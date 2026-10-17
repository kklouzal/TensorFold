"""One-host expert-rank process owner: distinct visible GPUs and coordinated failure retirement."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


def supervise(commands, environments, *, grace=10.0):
    """Own every child process group; any failure retires the entire rank group."""
    if len(commands) != len(environments) or not commands:
        raise ValueError("one environment per expert-rank command is required")
    children = []
    failure = None
    try:
        for command, environment in zip(commands, environments):
            children.append(subprocess.Popen(command, env=environment, start_new_session=True))
        while True:
            statuses = [child.poll() for child in children]
            if any(code not in (None, 0) for code in statuses):
                failure = next(code for code in statuses if code not in (None, 0))
                break
            if all(code == 0 for code in statuses):
                return 0
            if statuses[0] == 0:
                deadline = time.monotonic() + grace
                while time.monotonic() < deadline and any(child.poll() is None for child in children):
                    time.sleep(.05)
                if all(child.poll() == 0 for child in children):
                    return 0
                failure = 1
                break
            time.sleep(.05)
    finally:
        running = [child for child in children if child.poll() is None]
        for child in running:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # child already completed after the poll
        deadline = time.monotonic() + grace
        for child in children:
            try:
                child.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
    return abs(failure or 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True, help="distinct CUDA GPU UUIDs or ordinal device IDs")
    parser.add_argument("--expert-shards", required=True)
    parser.add_argument("--master-port", type=int, default=29551)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="serve MODEL and ordinary serving options")
    args = parser.parse_args(argv)
    devices = args.devices.split(",")
    if not 2 <= len(devices) <= 4 or any(not d.strip() for d in devices) or len(set(devices)) != len(devices):
        parser.error("--devices requires two through four distinct GPU identities")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command[:1] != ["serve"] or any(flag.split("=", 1)[0] in
            {"--expert-ranks", "--expert-shards", "--rank", "--master", "--master-port", "--tp"} for flag in command):
        parser.error("supply serve MODEL options without rank/topology flags; the launcher owns those")
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be 1 through 65535")
    commands, environments = [], []
    for rank, device in enumerate(devices):
        commands.append([sys.executable, "-m", "tensorfold", *command,
                         "--expert-ranks", str(len(devices)), "--expert-shards", args.expert_shards,
                         "--rank", str(rank), "--master", "127.0.0.1", "--master-port", str(args.master_port)])
        environments.append({**os.environ, "CUDA_VISIBLE_DEVICES": device.strip()})
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        return supervise(commands, environments)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
