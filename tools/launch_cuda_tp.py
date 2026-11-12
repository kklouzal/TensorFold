#!/usr/bin/env python3
"""Supervise the ordinary TensorFold CUDA CLI ranks on one multi-GPU host.

V100 NUMA recipe: --gpus GPU-uuid0,...,GPU-uuid7 --numa-plan /sealed/plan.json
--numa-plan-sha256 SHA -- MODEL --context 4096 --no-drafts. Other declared
legacy regions may omit the plan; the new V100 campaign/recipe requires it.
Arguments after ``--`` are passed to ``tensorfold serve`` on every rank.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.cohort import FAILURE_FD_ENV
from tensorfold.cuda.expert_launch import supervise


def main(argv=None, *, _command_prefix=None, _shutdown_probe=None, _failure_probe=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Comma-separated visible CUDA device IDs or UUIDs, rank order")
    parser.add_argument("--numa-plan", help="Absolute sealed GPU-UUID/NUMA rank plan; required by the V100 NUMA recipe")
    parser.add_argument("--numa-plan-sha256", help="Exact lowercase SHA256 of the rank plan")
    parser.add_argument("--master-port", type=int, default=29551)
    parser.add_argument("serve_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    devices = [device.strip() for device in args.gpus.split(",")]
    if len(devices) not in (2, 4, 8) or any(not item.strip() for item in devices) or len(set(devices)) != len(devices):
        parser.error("--gpus requires 2, 4 or 8 distinct device IDs")
    if (args.numa_plan is None)!=(args.numa_plan_sha256 is None):
        parser.error('--numa-plan and --numa-plan-sha256 must be supplied together')
    if args.numa_plan is not None:
        if os.environ.get('NCCL_IGNORE_CPU_AFFINITY','0')!='0':
            parser.error('NUMA rank placement requires NCCL_IGNORE_CPU_AFFINITY=0')
        from tensorfold.cuda.rank_placement import load_plan, validate_plan
        validate_plan(load_plan(args.numa_plan,args.numa_plan_sha256),devices)
    tail = args.serve_args
    if tail[:1] == ["--"]:
        tail = tail[1:]
    if not tail or not 1 <= args.master_port <= 65535:
        parser.error("supply MODEL and serve options after --, and a valid master port")
    if any(option.split("=", 1)[0] in ("--tp", "--rank", "--master", "--master-port") for option in tail):
        parser.error("the launcher supplies --tp, --rank, --master and --master-port")
    if _command_prefix is None:
        prefix = [sys.executable, "-m", "tensorfold", "serve"]
    else:
        if (type(_command_prefix) not in (list, tuple) or not _command_prefix
                or any(type(value) is not str or not value or "\0" in value for value in _command_prefix)):
            raise ValueError("private rank command prefix must be a nonempty argument sequence")
        prefix = list(_command_prefix)
    if any(probe is not None and not callable(probe) for probe in (_shutdown_probe, _failure_probe)):
        raise ValueError('private launch probes must be callable before cohort acquisition')
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
            command = [*prefix, *tail, "--tp", str(len(devices)),
                       "--rank", str(rank), "--master", "127.0.0.1", "--master-port", str(args.master_port)]
            if args.numa_plan is not None:
                env['NCCL_IGNORE_CPU_AFFINITY']='0'
                command=[sys.executable,'-m','tensorfold.cuda.rank_placement','exec',
                         '--plan',args.numa_plan,'--sha256',args.numa_plan_sha256,
                         '--rank',str(rank),'--',*command]
            commands.append(command)
            environments.append(env)

        def failed():
            try:
                worker_failed = bool(os.read(read_fd, 1))
            except BlockingIOError:
                worker_failed = False
            client_failed = False if _failure_probe is None else _failure_probe()
            if type(client_failed) is not bool:
                raise ValueError('private failure probe must return a boolean')
            return worker_failed or client_failed

        def stop(signum, _frame):
            nonlocal stopping
            stopping = True

        def normal_stop():
            client_done = False if _shutdown_probe is None else _shutdown_probe(stopping)
            if type(client_done) is not bool:
                raise ValueError('private shutdown probe must return a boolean')
            return stopping or client_done

        for index, signum in enumerate((signal.SIGINT, signal.SIGTERM)):
            previous[index] = signal.getsignal(signum)
            signal.signal(signum, stop)
        result = supervise(commands, environments, grace=30.0, failure_probe=failed,
                           pass_fds=(write_fd,), shutdown_probe=normal_stop,
                           shutdown_rank=len(commands) - 1)
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
