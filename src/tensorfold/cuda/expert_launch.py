"""One-host expert-rank process owner: distinct visible GPUs and coordinated failure retirement."""
from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from tensorfold.cleanup import raise_failures


class UnresolvedRankSessions(RuntimeError):
    """Retain failed session owners until explicit recovery or rank-process containment.

    Callers retain this error and must not reap, rescan or signal its numeric
    session IDs. An unresolved retirement is never successful group shutdown.
    """
    def __init__(self, sessions, failure):
        super().__init__("expert session retirement unresolved; retain owners until process containment")
        self.sessions = tuple(sessions)
        self.__cause__ = failure


def _exited(process):
    # WNOWAIT reserves the session leader's numeric identity until all owned
    # compiler groups retire. Popen.poll/communicate/wait must not reap it early.
    status = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if status is None:
        return None
    return status.si_status if status.si_code == os.CLD_EXITED else -status.si_status


@contextmanager
def _session_members(session):
    """Pin the proc directory before pidfd_open: reused PIDs cannot redirect inspection or signaling."""
    members = []
    with ExitStack() as handles:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                directory = os.open(entry, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                try:
                    # Inspect the pinned directory first, avoiding pidfds for
                    # unrelated users/kernel tasks. Re-read after pinning PID.
                    def fields(directory=directory):
                        with os.fdopen(os.open("stat", os.O_RDONLY, dir_fd=directory), "rb") as source:
                            return source.read(4096).rsplit(b") ", 1)[1].split()
                    metadata = fields()
                    if int(metadata[3]) != session or metadata[0] in (b"Z", b"X"):
                        continue
                    pidfd = os.pidfd_open(int(entry.name))
                    try:
                        metadata = fields()
                        if int(metadata[3]) == session and metadata[0] not in (b"Z", b"X"):
                            members.append(pidfd)
                            handles.callback(os.close, pidfd)
                            pidfd = None
                    finally:
                        if pidfd is not None:
                            os.close(pidfd)
                finally:
                    os.close(directory)
            except (FileNotFoundError, ProcessLookupError):
                continue  # exact inspected process retired; never target its replacement
        yield members


def _retire(process, deadline):
    if process.returncode is not None:
        raise RuntimeError("expert rank was reaped before its session retired")
    while True:
        with _session_members(process.pid) as members:
            if time.monotonic() >= deadline:
                for pidfd in members:
                    try:
                        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        with _session_members(process.pid) as members:
            live = bool(members)
        if _exited(process) is not None and not live:
            process.wait(timeout=0)  # no further inspection/signaling of this reserved SID
            return
        if time.monotonic() >= deadline + 10:
            raise RuntimeError("expert session failed to retire; unreaped leader retains its identity")
        time.sleep(.01)


def supervise(commands, environments, *, grace=10.0):
    """Exclusively own direct-child waits and Linux rank sessions through group retirement."""
    if len(commands) != len(environments) or not commands:
        raise ValueError("one environment per expert-rank command is required")
    if isinstance(grace, bool) or not isinstance(grace, (int, float)) or not math.isfinite(grace) or grace < 0:
        raise ValueError("expert rank retirement grace must be a finite nonnegative duration")
    if not all(hasattr(os, name) for name in ("waitid", "WNOWAIT", "pidfd_open")) or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("expert launcher requires Linux waitid/WNOWAIT and pidfd session ownership")
    children, primary, result = [], None, 1
    try:
        for command, environment in zip(commands, environments):
            children.append(subprocess.Popen(command, env=environment, start_new_session=True))
        departure = None
        while True:
            statuses = [_exited(child) for child in children]
            if any(code not in (None, 0) for code in statuses):
                result = abs(next(code for code in statuses if code not in (None, 0)))
                break
            if all(code == 0 for code in statuses):
                result = 0
                break
            if any(code is not None for code in statuses):
                departure = departure or time.monotonic() + grace
                if time.monotonic() >= departure:
                    break  # a successful follower departure cannot leave an idle leader indefinitely
            time.sleep(.05)
    except BaseException as error:  # noqa: BLE001 - cleanup preserves primary cancellation/spawn failure
        primary = error
    errors = []
    unresolved = []
    deadline = time.monotonic() + grace
    for child in children:
        try:
            _exited(child)  # confirm our unreaped child identity before any numeric group signal
        except BaseException as error:  # noqa: BLE001 - never signal an identity with unresolved ownership
            unresolved.append(child)
            errors.append(UnresolvedRankSessions((child,), error))
            continue
        try:
            os.killpg(child.pid, signal.SIGTERM)  # unreaped leader still reserves this group identity
        except ProcessLookupError:
            pass
        except BaseException as error:  # noqa: BLE001 - still attempt owned pidfd retirement below
            errors.append(error)
    for child in children:
        if any(child is owner for owner in unresolved):
            continue  # ownership could not be established: no later numeric SID actions
        try:
            _retire(child, deadline)
        except BaseException as error:  # noqa: BLE001 - retain child owners and surface all cleanup failures
            errors.append(UnresolvedRankSessions((child,), error))
    if errors and primary is None and result:
        primary = RuntimeError(f"expert rank group failed with status {result}")
    raise_failures(primary, errors)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True, help="distinct CUDA GPU UUIDs or ordinal device IDs")
    parser.add_argument("--expert-shards", required=True)
    parser.add_argument("--rank-entrypoint", default=os.environ.get("TENSORFOLD_EXPERT_ENTRYPOINT"),
                        help="trusted Python serving script for each rank; default package CLI outside container")
    parser.add_argument("--master-port", type=int, default=29551)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="serve MODEL and ordinary serving options")
    args = parser.parse_args(argv)
    devices = [device.strip() for device in args.devices.split(",")]
    if not 1 <= len(devices) <= 4 or any(not d.strip() for d in devices) or len(set(devices)) != len(devices):
        parser.error("--devices requires one through four distinct GPU identities")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command[:1] != ["serve"] or any(flag.split("=", 1)[0] in
            {"--expert-ranks", "--expert-shards", "--rank", "--master", "--master-port", "--tp"} for flag in command):
        parser.error("supply serve MODEL options without rank/topology flags; the launcher owns those")
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be 1 through 65535")
    if args.rank_entrypoint is not None and not Path(args.rank_entrypoint).is_file():
        parser.error("--rank-entrypoint must name a trusted existing Python serving script")
    commands, environments = rank_commands(devices, command, args.expert_shards, args.master_port,
                                           entrypoint=args.rank_entrypoint)
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        return supervise(commands, environments)
    finally:
        signal.signal(signal.SIGTERM, previous)


def rank_commands(devices, command, shards, port, *, entrypoint=None):
    """Preserve serving entrypoint and argv boundaries; operator-controlled paths never enter a shell."""
    prefix = [sys.executable, str(entrypoint)] if entrypoint is not None else [sys.executable, "-m", "tensorfold"]
    commands, environments = [], []
    for rank, device in enumerate(devices):
        commands.append([*prefix, *command,
                         "--expert-ranks", str(len(devices)), "--expert-shards", shards,
                         "--rank", str(rank), "--master", "127.0.0.1", "--master-port", str(port)])
        environments.append({**os.environ, "CUDA_VISIBLE_DEVICES": device.strip()})
    return commands, environments


if __name__ == "__main__":
    raise SystemExit(main())
