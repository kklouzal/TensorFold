"""Tiny real NCCL TP4/8 DSpark gates; no model files or engine changes."""

import multiprocessing
import queue
import socket
import time
import traceback

import pytest
import torch

from test_flashnext_dspark_cuda import assert_bits, target_state, tiny_features, tiny_provider


def _worker(rank, world, port, outcomes):
    from tensorfold.cuda.comm import NCCL
    from tensorfold.families.qwen4_exp.cuda.dspark_provider import DraftSample

    comm = NCCL()
    failed = False
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        comm.open(rank, world, '127.0.0.1', port, timeout_s=60, key='dspark_tiny_uid')
        p = tiny_provider(rank=rank, world=world, comm=comm)
        state = p.make_state(target_state(), capacity=64)
        f = tiny_features(p.config, 4).cuda()
        update = p.stage_context(state, f)
        state.target.pos = 3
        p.commit(state, update, 3)
        initial = state.keys[:, :3].clone(), state.values[:, :3].clone()
        forced = [7, 19, 32]
        def sample(logits, *, step, shard, position):
            assert position == 4 + step
            q = torch.zeros_like(logits, dtype=torch.float32)
            token = forced[step]
            if shard.start <= token < shard.stop:
                q[0, token - shard.start] = 1
            return DraftSample(torch.tensor([token], device='cuda'), q)
        out = p.propose(state, 4, block=3, sampler=sample).wait()
        assert out.ids.cpu().tolist() == forced
        reference = tiny_provider(device='cpu')
        rs = reference.make_state(target_state(), capacity=64)
        ru = reference.stage_context(rs, f.cpu())
        rs.target.pos = 3
        reference.commit(rs, ru, 3)
        def cpu_sample(logits, *, step, **_):
            q = torch.zeros_like(logits, dtype=torch.float32)
            q[0, forced[step]] = 1
            return DraftSample(torch.tensor([forced[step]]), q)
        expected = reference.propose(rs, 4, block=3, sampler=cpu_sample)
        torch.testing.assert_close(out.corrected_logits.cpu(), expected.corrected_logits[:, p.shard.start:p.shard.stop],
                                   rtol=.015, atol=.005)
        torch.testing.assert_close(out.confidence.cpu(), expected.confidence, rtol=.002, atol=.0005)
        first, width = p.local['first_kv'], p.local['kv_heads']
        torch.testing.assert_close(state.keys[:, :3].cpu(), rs.keys[:, :3, first:first + width], rtol=.005, atol=.001)
        torch.testing.assert_close(state.values[:, :3].cpu(), rs.values[:, :3, first:first + width], rtol=.005, atol=.001)
        # Every rank uses the exact same full hidden/confidence/global IDs.
        peers = torch.empty((world, 3), dtype=torch.int64, device='cuda')
        comm.all_gather(out.ids, peers)
        assert all(row == forced for row in peers.cpu().tolist())
        confidence = torch.empty((world, 3), dtype=torch.float32, device='cuda')
        comm.all_gather(out.confidence, confidence)
        for row in confidence:
            assert_bits(row, out.confidence)
        for fault in ('invalid-id', 'throw', 'wrong-shape', 'nan-q'):
            def bad(logits, *, step, shard, position):
                if rank == world - 1 and fault == 'throw':
                    raise RuntimeError('remote callback failure')
                if rank == world - 1 and fault == 'wrong-shape':
                    return None
                result = sample(logits, step=step, shard=shard, position=position)
                if rank == world - 1 and fault == 'invalid-id':
                    result.ids.fill_(p.config.vocab)
                if rank == world - 1 and fault == 'nan-q':
                    result.probabilities.fill_(float('nan'))
                return result
            with pytest.raises(ValueError, match='sampler'):
                p.propose(state, 4, block=3, sampler=bad)
            assert_bits(state.keys[:, :3], initial[0])
            assert_bits(state.values[:, :3], initial[1])
            assert state.length == 3 and state.pending is None
        # A post-error collective and healthy proposal prove no peer hung or
        # published a partial update before the sampling status agreement.
        after = p.propose(state, 4, block=3, sampler=sample)
        assert_bits(out.corrected_logits, after.corrected_logits)
        clone = p.copy_prefix(state, target_state(3), capacity=32)
        assert_bits(clone.keys[:, :3], state.keys[:, :3])
        assert clone.keys.data_ptr() != state.keys.data_ptr()
        p.close_state(clone)
        p.close_state(state)
        comm.ready('dspark-tiny-complete', every=1, timeout=30)
        torch.cuda.synchronize()
        outcomes.put((rank, None, torch.cuda.max_memory_allocated()))
    except BaseException:
        failed = True
        outcomes.put((rank, traceback.format_exc(), 0))
        raise
    finally:
        comm.close(abort=failed)


@pytest.mark.parametrize('world', [4, 8])
def test_native_tp_dspark_dense_gqa_markov_transactions_and_remote_errors(world):
    if torch.cuda.device_count() < world:
        pytest.skip(f'requires {world} visible CUDA GPUs')
    if any(torch.cuda.get_device_capability(i) != (7, 0) for i in range(world)):
        pytest.skip('SM70 DSpark scope')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    ctx = multiprocessing.get_context('spawn')
    outcomes = ctx.Queue()
    workers = [ctx.Process(target=_worker, args=(rank, world, port, outcomes)) for rank in range(world)]
    results = []
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 120
        while len(results) < world and time.monotonic() < deadline:
            try:
                results.append(outcomes.get(timeout=1))
            except queue.Empty:
                if all(not worker.is_alive() for worker in workers):
                    break
        assert len(results) == world, results
        assert all(error is None for _, error, _ in results), results
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.pid is not None:
                if worker.is_alive():
                    worker.kill()
                worker.join(timeout=10)
        outcomes.close()
        outcomes.join_thread()
