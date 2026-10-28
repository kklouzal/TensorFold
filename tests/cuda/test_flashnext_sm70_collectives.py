"""Native V100 communication gates, independent of model weights and serving state."""

import multiprocessing
import queue
import socket
import time
import traceback

import pytest
import torch


def _status_cohort_oracle(owner):
    from tensorfold.families.qwen4_exp.cuda.forward import _kv_check
    from tensorfold.families.qwen4_exp.cuda.kv_status import KVStatusCohort
    from tensorfold.families.qwen4_exp.cuda.state import KVNumericError, State

    workspace = KVStatusCohort(6, owner, "cuda")
    states = []
    for _ in range(6):
        state = object.__new__(State)
        state.execution_dtype = torch.float16
        state.kv_key_dtype, state.kv_value_dtype = "fp16", "int8"
        state.kv_status = torch.tensor([0, -2], dtype=torch.int32, device="cuda")
        state._kv_peer_status = torch.empty((owner.world, 2), dtype=torch.int32, device="cuda")
        state._kv_pending, state._kv_error, state._resize_error = True, None, None
        state.compact_indexer = None  # Match the actual State invariant before kv_begin.
        state.pos, state.mtp_len = 9, 7
        states.append(state)
    for count in (1, 3, 6):
        segs = [(state, index, index + 1) for index, state in enumerate(states[:count])]
        for state, _, _ in segs:
            state.kv_begin()
        _kv_check(segs, comm=owner, workspace=workspace)
        assert all(not state._kv_pending for state, _, _ in segs)
    # One peer's codec error is known everywhere before any rank raises or
    # publishes the frame. A following collective proves all ranks completed it.
    if owner.rank == owner.world - 1:
        states[1].kv_status.copy_(torch.tensor([4, -1], dtype=torch.int32, device="cuda"))
    for state in states:
        state.kv_begin()
    with pytest.raises(KVNumericError):
        _kv_check([(state, index, index + 1) for index, state in enumerate(states)],
                  comm=owner, workspace=workspace)
    assert states[1]._kv_error is not None
    assert all(state._kv_error is None for index, state in enumerate(states) if index != 1)
    source = torch.tensor([owner.rank], dtype=torch.int32, device="cuda")
    peers = torch.empty(owner.world, dtype=torch.int32, device="cuda")
    owner.all_gather(source, peers)
    assert peers.cpu().tolist() == list(range(owner.world))



def _candidate_cohort_oracle(owner):
    import numpy as np
    from unittest.mock import patch
    from tensorfold.families.qwen4_exp.cuda.kv_status import KVStatusCohort
    from tensorfold.families.qwen4_exp.cuda.state import CAND, KVNumericError, State

    class Counted(KVStatusCohort):
        def _readback(self, count):
            self.reads += 1
            return super()._readback(count)

    rows, width = 3, 2 * CAND + 1
    workspace = Counted(rows, owner, "cuda", candidate_rows=rows)
    workspace.reads = 0
    states = []
    for _ in range(rows):
        state = object.__new__(State)
        state.execution_dtype = torch.float16
        state.kv_key_dtype, state.kv_value_dtype = "int8", "int8"
        state.kv_status = torch.tensor([0, -2], dtype=torch.int32, device="cuda")
        state._kv_pending, state._kv_error, state._resize_error = True, None, None
        state.compact_indexer = None
        state.pos, state.mtp_len = 9, 7
        states.append(state)
    local = torch.full((rows, width), float(owner.rank), dtype=torch.float32, device="cuda")
    ids = torch.arange(CAND, device="cuda", dtype=torch.int32) + owner.rank * CAND
    local[:, CAND:2*CAND].view(torch.int32).copy_(ids.expand(rows, CAND))
    packet = torch.empty(owner.world * rows * width, dtype=torch.float32, device="cuda")
    owner.all_gather(local, packet)  # Complete packet is produced before the tested status boundary.
    actual = workspace.read_candidates(states, packet, rows, comm=owner)
    expected = np.empty((owner.world, rows, width), dtype=np.float32)
    for rank in range(owner.world):
        expected[rank].fill(rank)
        expected[rank, :, CAND:2*CAND] = np.broadcast_to(
            np.arange(rank*CAND, (rank+1)*CAND, dtype=np.int32), (rows, CAND)).view(np.float32)
    assert workspace.reads == 1
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert actual.flags.owndata and all(not state._kv_pending for state in states)
    workspace.host.fill_(-37)
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    del actual  # Retire the sole escaped payload before the next candidate readback.
    if owner.rank == owner.world - 1:
        states[1].kv_status.copy_(torch.tensor([4, -1], dtype=torch.int32, device="cuda"))
    for state in states:
        state.kv_begin()
    with patch.object(torch.Tensor, "numpy", side_effect=AssertionError("candidate exposed before status")):
        with pytest.raises(KVNumericError):
            workspace.read_candidates(states, packet, rows, comm=owner)
    assert workspace.reads == 2 and states[1]._kv_error is not None
    # Expected numeric failure contains only the affected sequence, not the transport.
    source = torch.tensor([owner.rank], dtype=torch.int32, device="cuda")
    peers = torch.empty(owner.world, dtype=torch.int32, device="cuda")
    owner.all_gather(source, peers)
    assert peers.cpu().tolist() == list(range(owner.world))

def _collectives(rank, world, port, outcomes):
    """One process owns each CUDA context and communicator through retirement."""
    from tensorfold.cuda.comm import NCCL, agree_eos

    owner = NCCL()
    failed = False
    try:
        torch.cuda.set_device(rank)
        owner.open(rank, world, "127.0.0.1", port, timeout_s=60, key="tf_sm70_component_uid")
        for dtype in (torch.float16, torch.float32, torch.int64):
            source = torch.tensor([rank, rank + 11], dtype=dtype, device="cuda")
            destination = torch.empty(2 * world, dtype=dtype, device="cuda")
            owner.all_gather(source, destination)
            torch.cuda.synchronize()
            assert destination.cpu().tolist() == [v for r in range(world) for v in (r, r + 11)]
        _status_cohort_oracle(owner)
        _candidate_cohort_oracle(owner)
        agree_eos(owner.all_gather, (9, 1, 9), world=world, device=torch.device("cuda", rank))
        source = torch.tensor([rank, rank + 11], dtype=torch.float16, device="cuda")
        destination = torch.empty(2 * world, dtype=torch.float16, device="cuda")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            owner.all_gather(source, destination)
        side.synchronize()
        owner.ready("sm70-before-capture", every=1, timeout=30)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            owner.all_gather(source, destination)
        source.add_(3)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            graph.replay()
        side.synchronize()
        assert destination.cpu().tolist() == [v for r in range(world) for v in (r + 3, r + 14)]
        graph.reset()  # Retire captured communicator borrowers before native teardown.
        owner.ready("sm70-component-complete", every=1, timeout=30)
    except BaseException:
        failed = True
        outcomes.put((rank, traceback.format_exc()))
        raise
    finally:
        owner.close(abort=failed)
    outcomes.put((rank, None))


@pytest.mark.parametrize("world", [4, 8])
def test_v100s_exchange_half_float_and_integer_words_and_replay_graph(world):
    """Small real NCCL exchanges; this does not claim full model/TP8 qualification."""
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} visible CUDA GPUs")
    if any(torch.cuda.get_device_capability(rank) != (7, 0) for rank in range(world)):
        pytest.skip("SM70 foundation gate")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    outcomes = context.Queue()
    processes = [context.Process(target=_collectives, args=(rank, world, port, outcomes)) for rank in range(world)]
    results = []
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 100
        while len(results) < world and time.monotonic() < deadline:
            try:
                results.append(outcomes.get(timeout=min(1, max(0.01, deadline - time.monotonic()))))
            except queue.Empty:
                if all(not process.is_alive() for process in processes):
                    break
        assert len(results) == world, f"native participants did not retire: {results}"
        assert sorted(results) == [(rank, None) for rank in range(world)], results
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                process.join(timeout=10)
        outcomes.close()
        outcomes.join_thread()
