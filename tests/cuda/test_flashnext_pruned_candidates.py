"""Real NCCL graph packets for mapped, unequal and very small draft shards."""

import multiprocessing
import queue
import socket
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _participant(rank, world, port, results):
    from tensorfold.cuda.comm import NCCL
    from tensorfold.families.qwen4_exp.cuda.decode import SamplingShard, choose_gathered, tp_sample_rows
    from tensorfold.families.qwen4_exp.cuda.forward import candidates
    from tensorfold.families.qwen4_exp.cuda.state import CAND
    from tensorfold.engine.exact_sampling import Sampling

    owner, failed = NCCL(), False
    try:
        torch.cuda.set_device(rank)
        owner.open(rank, world, "127.0.0.1", port, timeout_s=90, key="tf_pruned_candidates")
        w = SimpleNamespace(comm=owner, meta={"world": world, "vocab_offset": 0})
        for widths in ((1, 2, 7, 31), (31, 32, 33, 34)):
            width, rows = widths[rank], 3
            mapping = torch.arange(width, dtype=torch.int64, device="cuda") * world + rank
            source = torch.empty((rows, width), dtype=torch.float16, device="cuda")
            plan = SamplingShard(w, width, id_map=mapping)
            buffer = SimpleNamespace(cand=torch.empty((rows, 2 * CAND + 1), device="cuda"),
                                     cand_all=torch.empty(world * rows * (2 * CAND + 1), device="cuda"), world=world)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                source.zero_()
                candidates(w, buffer, source, rows, id_map=mapping)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                candidates(w, buffer, source, rows, id_map=mapping)
            for tied in (True, False):
                data = torch.zeros_like(source) if tied else torch.sin(
                    mapping.float()[None] * .37 + torch.arange(rows, device="cuda")[:, None])
                source.copy_(data)
                graph.replay()
                for policy in (None, Sampling(41, .7, 3, .9), Sampling(7, .8, 20, .95),
                               Sampling(19, .7, 40, .9)):
                    got = choose_gathered(w, buffer.cand_all, rows, [17, 18, 19], policy,
                                          with_prob=True, logits=source, plan=plan)
                    reference = tp_sample_rows(w, source, [17, 18, 19], policy, with_prob=True, plan=plan)
                    assert got[0] == reference[0]
                    np.testing.assert_allclose(got[1], reference[1], rtol=1e-6, atol=1e-7)
                    assert all(token % world in range(world) and token // world < widths[token % world]
                               for token in got[0])
            graph.reset()
        owner.ready("pruned-candidates-complete", every=1, timeout=60)
        results.put((rank, None))
    except BaseException:
        failed = True
        results.put((rank, traceback.format_exc()))
        raise
    finally:
        owner.close(abort=failed)


def test_four_rank_small_mapped_heads_capture_and_replay_exact_sampling():
    world = 4
    if torch.cuda.device_count() < world:
        pytest.skip("requires four CUDA devices")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    children = [context.Process(target=_participant, args=(rank, world, port, results)) for rank in range(world)]
    observed = []
    try:
        for child in children:
            child.start()
        deadline = time.monotonic() + 150
        while len(observed) < world and time.monotonic() < deadline:
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
