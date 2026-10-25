#!/usr/bin/env python3
"""Supervise the ordinary TensorFold CUDA CLI ranks on one multi-GPU host.

Example: python tools/launch_cuda_tp.py --gpus 0,1,2,3,4,5,6,7 -- MODEL --context 4096 --no-drafts
Arguments after ``--`` are passed to ``tensorfold serve`` on every rank.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

from tensorfold.cuda.cohort import FAILURE_FD_ENV
from tensorfold.cuda.expert_launch import supervise
from tensorfold.cleanup import raise_failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Comma-separated visible CUDA device IDs or UUIDs, rank order")
    parser.add_argument("--master-port", type=int, default=29551)
    parser.add_argument("serve_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    devices = [device.strip() for device in args.gpus.split(",")]
    if len(devices) not in (2, 4, 8) or any(not item.strip() for item in devices) or len(set(devices)) != len(devices):
        parser.error("--gpus requires 2, 4 or 8 distinct device IDs")
    tail = args.serve_args
    if tail[:1] == ["--"]:
        tail = tail[1:]
    if not tail or not 1 <= args.master_port <= 65535:
        parser.error("supply MODEL and serve options after --, and a valid master port")
    if any(option.split("=", 1)[0] in ("--tp", "--rank", "--master", "--master-port") for option in tail):
        parser.error("the launcher supplies --tp, --rank, --master and --master-port")
    # Keep the failure channel process-owned until every rank/session retires.
    # A fatal worker can wake the supervisor even when another rank is stuck
    # inside a collective and cannot produce a normal operation acknowledgment.
    previous = [None, None]
    failures = [None] * 4
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    primary, result = None, None
    stopping = False
    try:
        commands, environments = [], []
        for rank in [*range(1, len(devices)), 0]:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[rank], **{FAILURE_FD_ENV: str(write_fd)})
            command = [sys.executable, "-m", "tensorfold", "serve", *tail, "--tp", str(len(devices)),
                       "--rank", str(rank), "--master", "127.0.0.1", "--master-port", str(args.master_port)]
            commands.append(command)
            environments.append(env)

        def failed():
            if stopping:
                return True
            try:
                return bool(os.read(read_fd, 1))
            except BlockingIOError:
                return False

        def stop(signum, _frame):
            nonlocal stopping
            stopping = True

        for index, signum in enumerate((signal.SIGINT, signal.SIGTERM)):
            previous[index] = signal.getsignal(signum)
            signal.signal(signum, stop)
        result = supervise(commands, environments, grace=30.0, failure_probe=failed,
                           pass_fds=(write_fd,), stop_on_departure=True)
        if stopping:
            result = 0  # preserve the launcher's ordinary signal-stop status after successful retirement
    except BaseException as error:
        primary = error
    for index, signum in enumerate((signal.SIGINT, signal.SIGTERM)):
        if previous[index] is not None:
            try:
                signal.signal(signum, previous[index])
            except BaseException as error:
                failures[index] = error
    for index, fd in enumerate((write_fd, read_fd), 2):
        try:
            os.close(fd)
        except BaseException as error:
            failures[index] = error
    try:
        errors = [error for error in failures if error is not None]
    except BaseException as error:
        if primary is not None:
            raise primary from error
        raise
    raise_failures(primary, errors)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
