"""SM70 bound State captures preserve actual logits and accepted-prefix state."""

import faulthandler
import multiprocessing
import queue
import socket
import sys
import time
import traceback

import pytest
import torch

from test_flashnext_sm70_family import _small_weights

from tensorfold.families.qwen4_exp.cuda.decode import prefill
from tensorfold.families.qwen4_exp.cuda.forward import commit, compute, stage
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder, _slot
from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs
from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry
from tensorfold.families.qwen4_exp.cuda.state import CAND, Buffers

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA graph state gate")


@pytest.fixture(autouse=True)
def native_provider(monkeypatch):
    from tensorfold.cuda.sm70 import family_ops

    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    # Keep this short-context ownership fixture on the original Torch operator.
    # Qualified native attention (65..8192) has its independent exact oracle.
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)


def _assert_state(actual, expected):
    assert actual.pos == expected.pos and actual.cur == expected.cur
    for name in ("rec", "conv", "ple_tail"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    for a, b in zip(actual.kc, expected.kc):
        for name in ("k", "v", "ks", "vs"):
            torch.testing.assert_close(getattr(a, name)[:actual.pos], getattr(b, name)[:actual.pos], rtol=0, atol=0)
    for a, b in zip(actual.ikc, expected.ikc):
        torch.testing.assert_close(a[:actual.pos], b[:actual.pos], rtol=0, atol=0)
    for a, b in zip(actual.pooled, expected.pooled):
        torch.testing.assert_close(a[:actual.pos // actual.ratio], b[:actual.pos // actual.ratio], rtol=0, atol=0)


def _phase(w, label):
    if w.comm is not None:
        print(f"[bound-gate rank={w.comm.rank} t={time.monotonic():.3f}] {label}", file=sys.stderr, flush=True)


def _six_lane_geometry(w):
    # A target-only decoder has one anchor row, not two unbacked proposals.
    # Real stock-MTP fixtures retain the original varied R1/R2/R3 windows.
    geometry = MethodGeometry("none", 0) if w.mtp is None else MethodGeometry("mtp", 2)
    counts = (1,) * 6 if geometry.method == "none" else (1, 3, 2, 1, 2, 3)
    return geometry, counts


def _qualify_six_lanes(w, kv_dtype, reference_context=None):
    _phase(w, "verify:decoder")
    geometry, counts = _six_lane_geometry(w)
    decoder = MultiDecoder(w, slots=6, capacity=64, depth=geometry.maximum_proposals, prefill_rows=8, kv_dtype=kv_dtype,
                           graphs=False, keep=0)
    assert decoder.speculation == geometry
    graphs = RoundGraphs(decoder, 1024 * 2**20)
    try:
        _phase(w, "verify:capture")
        graphs.capture()
        _phase(w, "verify:captured")
        decoder.graphs = graphs
        for index, state in enumerate(decoder.free):
            e = _slot(w, state, decoder.buf, decoder.mbuf, decoder.pbuf, 64, 8)
            prefill(e, [4, 17, 3, 99, 33, 2, 7, 5, 6], None, mtp=False)
            if index % 2:
                state.rec[1 - state.cur[0]].copy_(state.rec[state.cur[0]])
                state.cur = [1 - value for value in state.cur]
        _phase(w, "verify:prefilled")
        # Reverse slot order and parity in both methods; real MTP also varies
        # row counts and truncates accepted prefixes within its allocated R3.
        states = list(reversed(decoder.free))
        reference = Buffers(w, decoder.buf.rows, 64, moe_prefill=True)
        for iteration in range(2):
            _phase(w, f"verify:iteration={iteration}:reference")
            windows = [[(11 * index + row + iteration * 17) % w.cfg.vocab for row in range(count)]
                       for index, count in enumerate(counts)]
            expected_states = [state.clone() for state in states]
            refs = stage(w, reference, list(zip(expected_states, windows)))
            expected = compute(w, refs, reference, context=reference_context).clone()
            _phase(w, f"verify:iteration={iteration}:replay")
            segs = stage(w, decoder.buf, list(zip(states, windows)), late=True)
            actual = graphs.verify_forward(segs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if w.comm is not None:
                words = w.comm.world * sum(counts) * (2 * CAND + 1)
                torch.testing.assert_close(decoder.buf.cand_all[:words].view(torch.int32),
                                           reference.cand_all[:words].view(torch.int32), rtol=0, atol=0)
            torch.testing.assert_close(decoder.buf.streams[:sum(counts)], reference.streams[:sum(counts)],
                                       rtol=0, atol=0)
            for index, ((state, a0, a1), expected_state) in enumerate(zip(segs, expected_states)):
                kept = min(counts[index], 1 + index % 2)
                commit(w, state, decoder.buf, counts[index], kept, at=a0)
                commit(w, expected_state, reference, counts[index], kept, at=a0)
                _assert_state(state, expected_state)
            _phase(w, f"verify:iteration={iteration}:exact")
        assert graphs.bound.hits > 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        _phase(w, "verify:close")
        graphs.close()
        decoder.graphs = None


def _qualify_mtp(w, cache, reference_context=None, *, slots=6):
    from tensorfold.families.qwen4_exp.cuda.mtp import mtp_compute, mtp_stage

    _phase(w, "mtp:decoder")
    decoder = MultiDecoder(w, slots=slots, capacity=32, depth=2, prefill_rows=8, kv_dtype=cache, graphs=False)
    graphs = RoundGraphs(decoder, 1024 * 2**20)
    reference = Buffers(w, decoder.buf.rows, 32, moe_prefill=decoder.mbuf.moe.plan.prefill)
    states = list(reversed(decoder.free))
    try:
        for state in states:
            prefill(_slot(w, state, decoder.buf, decoder.mbuf, decoder.pbuf, 32, 8),
                    [4, 17, 3, 99, 33, 2, 7, 5, 6], None)
        _phase(w, "mtp:prefilled")
        counts_by_step = ([1], [3], [2], [1]) if slots == 1 else (
            [1, 3, 2, 1, 2, 3], [1, 1, 1, 1, 1, 1], [3, 1, 2, 3, 2, 1], [1, 1, 1, 1, 1, 1])
        for iteration, counts in enumerate(counts_by_step):
            _phase(w, f"mtp:iteration={iteration}:reference")
            refs = [state.clone() for state in states]
            streams = torch.randn((sum(counts), w.cfg.streams * w.cfg.hidden), device=w.device,
                                  dtype=torch.float16) * 0.1
            windows, expected_windows, at = [], [], 0
            for index, (state, copied, count) in enumerate(zip(states, refs, counts)):
                tokens = [(13 * index + row) % w.cfg.vocab for row in range(count)]
                windows.append((state, tokens, streams[at:at + count]))
                expected_windows.append((copied, tokens, streams[at:at + count]))
                at += count
            expected = mtp_compute(w, mtp_stage(reference, expected_windows), reference,
                                   context=reference_context).clone()
            _phase(w, f"mtp:iteration={iteration}:replay")
            actual = graphs.mtp_step(mtp_stage(decoder.mbuf, windows))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(decoder.mbuf.streams[:sum(counts)], reference.streams[:sum(counts)],
                                       rtol=0, atol=0)
            if w.comm is not None:
                words = w.comm.world * len(states) * (2 * CAND + 1)
                torch.testing.assert_close(decoder.mbuf.cand_all[:words].view(torch.int32),
                                           reference.cand_all[:words].view(torch.int32), rtol=0, atol=0)
            for state, copied, count in zip(states, refs, counts):
                for item in (state, copied):
                    item.set_mtp_len(item.mtp_len + count)
                for name in ("k", "v", "ks", "vs"):
                    torch.testing.assert_close(getattr(state.mtp_kc, name)[:state.mtp_len],
                                               getattr(copied.mtp_kc, name)[:copied.mtp_len], rtol=0, atol=0)
                torch.testing.assert_close(state.mtp_pooled[:state.mtp_len // state.ratio],
                                           copied.mtp_pooled[:copied.mtp_len // copied.ratio], rtol=0, atol=0)
            _phase(w, f"mtp:iteration={iteration}:exact")
        expected_keys = 2
        assert len(graphs.mtp) == expected_keys and graphs.bound.hits == len(counts_by_step) - expected_keys
        assert {key[1] for key in graphs.mtp} == {1, 3}
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        _phase(w, "mtp:close")
        graphs.close()


@pytest.mark.parametrize("cache", ["fp16", "int8"])
@pytest.mark.parametrize("slots", [1, 6])
def test_bound_mtp_live_last_rows_and_valid_cache_prefixes_reuse_width_keys(cache, slots):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 bound MTP graph gate")
    from test_flashnext_sm70_mtp import _weights_with_stock_mtp_protocol

    _qualify_mtp(_weights_with_stock_mtp_protocol(), cache, slots=slots)


def _qualify_singleton_draft_protocol(w, cache):
    from tensorfold.cuda.streams import Stream

    reference, decoder = [MultiDecoder(w, slots=1, capacity=64, depth=2, confidence=0,
                                       prefill_rows=8, kv_dtype=cache, graphs=False, keep=0) for _ in range(2)]
    graphs = RoundGraphs(decoder, 1024 * 2**20)
    expected, actual = [Stream([4, 17, 3, 99, 33, 2, 7, 5, 6], 14, stop_eos=False) for _ in range(2)]
    expected_frames, actual_frames = [], []

    def trace_steps(dec, frames):
        forward = dec._mtp

        def traced(segs):
            rows = segs[-1][2]
            inputs = dec.mbuf.mtp_in[:rows].clone()
            state = segs[0][0]
            cache = tuple(getattr(state.mtp_kc, name)[:state.mtp_len].clone()
                          for name in ("k", "v", "ks", "vs"))
            pooled = state.mtp_pooled[:state.mtp_len // state.ratio].clone()
            logits = forward(segs)
            frames.append((*cache, pooled, inputs, dec.mbuf.streams[:rows].clone(), logits.clone()))
            return logits

        dec._mtp = traced

    trace_steps(reference, expected_frames)
    trace_steps(decoder, actual_frames)
    try:
        graphs.capture()
        decoder.graphs = graphs
        reference.admit(expected)
        decoder.admit(actual)
        for iteration in range(actual.count + 1):
            _phase(w, f"protocol:round={iteration}")
            reference.round()
            decoder.round()
            assert len(actual_frames) == len(expected_frames)
            for step, (received, wanted) in enumerate(zip(actual_frames, expected_frames)):
                for name, a, b in zip(("K", "V", "Kscale", "Vscale", "pooled", "inputs", "streams", "logits"),
                                      received, wanted):
                    torch.testing.assert_close(a, b, rtol=0, atol=0,
                                               msg=f"round={iteration}, MTPstep={step}, {name}")
            actual_frames.clear()
            expected_frames.clear()
            assert actual.out == expected.out and actual.drafts == expected.drafts
            assert actual.done == expected.done and actual.error is expected.error is None
            assert (actual.drafted, actual.accepted) == (expected.drafted, expected.accepted)
            _assert_state(actual.st, expected.st)
            assert actual.st.mtp_len == expected.st.mtp_len
            assert actual.st.mtp_drafted == expected.st.mtp_drafted
            for name in ("k", "v", "ks", "vs"):
                torch.testing.assert_close(getattr(actual.st.mtp_kc, name)[:actual.st.mtp_len],
                                           getattr(expected.st.mtp_kc, name)[:expected.st.mtp_len], rtol=0, atol=0)
            torch.testing.assert_close(actual.st.mtp_pooled[:actual.st.mtp_len // actual.st.ratio],
                                       expected.st.mtp_pooled[:expected.st.mtp_len // expected.st.ratio], rtol=0, atol=0)
            if actual.done:
                break
        assert actual.done and len(actual.out) == actual.count and actual.drafted > 0
        assert graphs.bound.hits > 0 and graphs.mtp
    finally:
        graphs.close()
        decoder.graphs = None


@pytest.mark.parametrize("cache", ["fp16", "int8"])
def test_bound_singleton_draft_protocol_matches_eager_tokens_drafts_and_committed_state(cache):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 bound MTP protocol gate")
    from test_flashnext_sm70_mtp import _weights_with_stock_mtp_protocol

    _qualify_singleton_draft_protocol(_weights_with_stock_mtp_protocol(), cache)


@pytest.mark.parametrize("kv_dtype", ["fp16", "int8"])
def test_bound_six_lane_graphs_match_logits_and_partial_commits_at_mixed_parities(kv_dtype):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 bound graph gate")
    _qualify_six_lanes(_small_weights(include_ple=True), kv_dtype)


def _distributed_graphs(rank, world, port, outcomes, reference_context=None):
    from tensorfold.cuda.comm import NCCL
    from tensorfold.cuda.sm70 import family_ops

    comm, failed = NCCL(), False
    faulthandler.dump_traceback_later(90)
    try:
        family_ops._NATIVE_FAMILY = True
        family_ops._NATIVE_ATTENTION = False
        torch.cuda.set_device(rank)
        comm.open(rank, world, "127.0.0.1", port, timeout_s=60, key="tf_bound_graphs_uid")
        print(f"[bound-gate rank={rank}] communicator-open", file=sys.stderr, flush=True)
        w = _small_weights(include_ple=True)
        width = w.cfg.vocab // world
        w.head.weight = w.head.weight[rank * width:(rank + 1) * width].contiguous()
        w.head.n = width
        w.comm = comm
        w.meta.update(world=world, vocab_offset=rank * width)
        _qualify_six_lanes(w, "int8", reference_context)
        from test_flashnext_sm70_mtp import _weights_with_stock_mtp_protocol

        draft = _weights_with_stock_mtp_protocol()
        draft.head.weight = draft.head.weight[rank * width:(rank + 1) * width].contiguous()
        draft.head.n, draft.comm = width, comm
        draft.meta.update(world=world, vocab_offset=rank * width)
        _qualify_mtp(draft, "int8", reference_context)
        _qualify_mtp(draft, "int8", reference_context, slots=1)
        _qualify_singleton_draft_protocol(draft, "int8")
        comm.ready("bound-graph-state-oracles-complete", every=1, timeout=60)
    except BaseException:
        failed = True
        outcomes.put((rank, traceback.format_exc()))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        comm.close(abort=failed)
    outcomes.put((rank, None))


@pytest.mark.parametrize("world", [4, 8])
def test_bound_six_lane_scores_and_state_match_on_real_tensor_parallel_ranks(world):
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} visible CUDA GPUs")
    if any(torch.cuda.get_device_capability(rank) != (7, 0) for rank in range(world)):
        pytest.skip("SM70 tensor-parallel graph gate")
    # Compile exact source identities before the bounded distributed lifecycle.
    # A fresh native build may exceed its120s deadline without any rank stall.
    from tensorfold.cuda.sm70 import attention_native, family_native, gdn_native

    family_native.extension()
    gdn_native.extension()
    attention_native.extension()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    outcomes = context.Queue()
    processes = [context.Process(target=_distributed_graphs, args=(rank, world, port, outcomes))
                 for rank in range(world)]
    results = []
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 120
        while len(results) < world and time.monotonic() < deadline:
            try:
                results.append(outcomes.get(timeout=min(1, max(0.01, deadline - time.monotonic()))))
            except queue.Empty:
                if all(not process.is_alive() for process in processes):
                    break
        assert len(results) == world, results
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


def test_bound_owner_is_invalidated_before_growth_replaces_cache_storage():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 bound graph gate")
    w = _small_weights(include_ple=True)
    decoder = MultiDecoder(w, slots=1, capacity=64, depth=2, prefill_rows=8, kv_dtype="fp16", graphs=False)
    state = decoder.free[0]
    state.resize(32)
    graphs = RoundGraphs(decoder, 128 * 2**20)
    try:
        graphs.capture()
        decoder.graphs = graphs
        assert graphs.bound.bindings
        old = state.version
        assert decoder._grow(state, 40, alone=True)
        assert state.version > old and state.capacity == 64
        assert graphs.bound.bindings == {} and graphs.bound.lanes == {}
    finally:
        graphs.close()
        decoder.graphs = None


def test_bound_floor_prefix_copy_and_growth_cross_the_8192_boundary():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 bound graph storage boundary")
    w = _small_weights(include_ple=True)
    decoder = MultiDecoder(w, slots=2, capacity=16384, depth=2, prefill_rows=8,
                           kv_dtype="int8", graphs=False, keep=0)
    assert decoder._prepare_graph_floor()
    assert all(state.capacity == 8192 for state in decoder.free)
    graphs = RoundGraphs(decoder, 1024 * 2**20)
    reference = Buffers(w, decoder.buf.rows, 16384, moe_prefill=True)
    source, target = decoder.free
    try:
        graphs.capture()
        decoder.graphs = graphs
        source.set_pos(8189)
        baseline = source.clone()
        expected = compute(w, stage(w, reference, [(baseline, [3, 5])]), reference).clone()
        actual = graphs.verify_forward(stage(w, decoder.buf, [(source, [3, 5])], late=True))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        commit(w, source, decoder.buf, 2, 2)
        commit(w, baseline, reference, 2, 2)
        _assert_state(source, baseline)
        assert source.pos == 8191
        snapshot = source.snapshot()
        target.copy_prefix(source, source.pos, 0)
        target.restore(snapshot)
        torch.testing.assert_close(target.rec[target.cur[0]], snapshot["rec"], rtol=0, atol=0)
        for name in ("k", "v", "ks", "vs"):
            torch.testing.assert_close(getattr(target.kc[0], name)[:source.pos],
                                       getattr(source.kc[0], name)[:source.pos], rtol=0, atol=0)
        # The padded third row completes a future pool in source. Prefix copy
        # carries only complete committed pools; the real next row rebuilds it.
        assert not torch.count_nonzero(target.pooled[0][8191 // target.ratio])
        old = target.version
        assert decoder._grow(target, 8194)
        assert target.version > old and target.capacity == 16384
        assert all(all(lane.state is not target for lane in lanes)
                   for _pieces, lanes in graphs.bound.bindings.values())
        baseline = target.clone()
        expected = compute(w, stage(w, reference, [(baseline, [29, 31])]), reference).clone()
        actual = graphs.verify_forward(stage(w, decoder.buf, [(target, [29, 31])], late=True))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        commit(w, target, decoder.buf, 2, 2)
        commit(w, baseline, reference, 2, 2)
        _assert_state(target, baseline)
        assert target.pos == 8193
    finally:
        graphs.close()
        decoder.graphs = None
