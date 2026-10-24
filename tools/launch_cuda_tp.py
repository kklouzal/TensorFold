#!/usr/bin/env python3
"""Supervise the ordinary TensorFold CUDA CLI ranks on one multi-GPU host.

Example: python tools/launch_cuda_tp.py --gpus 0,1,2,3,4,5,6,7 -- MODEL --context 4096 --no-drafts
Arguments after ``--`` are passed to ``tensorfold serve`` on every rank.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Comma-separated visible CUDA device IDs or UUIDs, rank order")
    parser.add_argument("--master-port", type=int, default=29551)
    parser.add_argument("serve_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    devices = args.gpus.split(",")
    if len(devices) not in (2, 4, 8) or any(not item.strip() for item in devices) or len(set(devices)) != len(devices):
        parser.error("--gpus requires 2, 4 or 8 distinct device IDs")
    tail = args.serve_args
    if tail[:1] == ["--"]:
        tail = tail[1:]
    if not tail or not 1 <= args.master_port <= 65535:
        parser.error("supply MODEL and serve options after --, and a valid master port")
    if any(option.split("=", 1)[0] in ("--tp", "--rank", "--master", "--master-port") for option in tail):
        parser.error("the launcher supplies --tp, --rank, --master and --master-port")
    children = []
    stopping = False

    def stop(signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    result = 0
    try:
        for rank in [*range(1, len(devices)), 0]:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[rank])
            command = [sys.executable, "-m", "tensorfold", "serve", *tail, "--tp", str(len(devices)),
                       "--rank", str(rank), "--master", "127.0.0.1", "--master-port", str(args.master_port)]
            children.append(subprocess.Popen(command, env=env, start_new_session=True))
        while not stopping:
            exited = [child.returncode for child in children if child.poll() is not None]
            if exited:
                result = next((code for code in exited if code), 0)
                break
            time.sleep(0.2)
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        for child in children:
            try:
                child.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
