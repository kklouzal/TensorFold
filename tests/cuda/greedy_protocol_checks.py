"""Shared TP4 test lifecycle; callers supply their distinct qualification worker."""

import multiprocessing
import queue
import socket
import time

import pytest
import torch


def check_four_rank_protocol(_qualify):
    world = 4
    if torch.cuda.device_count() < world:
        pytest.skip("requires four CUDA devices")
    from tensorfold.cuda.sm70 import attention_native, family_native, gdn_native

    family_native.extension()
    attention_native.extension()
    gdn_native.extension()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    children = [context.Process(target=_qualify, args=(rank, world, port, results)) for rank in range(world)]
    observed = []
    try:
        for child in children:
            child.start()
        end = time.monotonic()+240
        while len(observed) < world and time.monotonic() < end:
            try:
                observed.append(results.get(timeout=1))
            except queue.Empty:
                if all(not child.is_alive() for child in children):
                    break
        assert sorted(observed) == [(rank, None) for rank in range(world)], observed
        for child in children:
            child.join(timeout=10)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.pid is not None:
                if child.is_alive():
                    child.kill()
                child.join(timeout=10)
        results.close()
        results.join_thread()

