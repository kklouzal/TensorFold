"""Private target CUDA-graph tap export composed with existing State commits."""
from types import SimpleNamespace as NS

import pytest
import torch

from test_flashnext_dspark_runtime import attach, exact, target_equal, weights
from test_flashnext_dspark_taps import oracle
from tensorfold.cuda.sm70 import family_ops
from tensorfold.families.qwen4_exp.cuda import forward
from tensorfold.families.qwen4_exp.cuda.decode import prefill
from tensorfold.families.qwen4_exp.cuda.dspark_taps import TapWorkspace
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder, _slot
from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs
from tensorfold.families.qwen4_exp.cuda.state import Buffers, KVNumericError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native DSpark target graph composition")


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 target graph scope")
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    # Keep this short-context ownership fixture on the original Torch operator.
    # Qualified native attention (65..8192) has its independent exact oracle.
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)


def config():
    return NS(hidden=2560, taps=(1, 12, 23, 34, 45), capture_before=(2, 13, 24, 35, 46))


def write(owner, rows, h, pending):
    sink = owner.begin_graph(rows)
    for index in owner.before:
        sink.before_layer(index, h, pending)
    sink.finish_graph_sink()


@pytest.mark.parametrize("mode", (None, 2, 3))
def test_gpu_only_sink_captures_owned_hc_then_exports_real_rows_without_nan_padding(mode):
    rows = 18
    graph_owner, source = TapWorkspace(config(), rows, "cuda"), TapWorkspace(config(), rows, "cuda")
    h = torch.randn(rows, 10240, device="cuda", dtype=torch.float16) * 0.1
    inject = torch.rand(rows, 4, device="cuda", dtype=torch.float16)
    branch = torch.randn((rows, 11, 2560) if mode == 2 else (8, rows, 2560), device="cuda") * 0.1
    wts = torch.rand(rows, 11, device="cuda") if mode == 2 else None
    pending = None if mode is None else (mode, branch, wts, inject)
    graph = torch.cuda.CUDAGraph()
    write(graph_owner, rows, h, pending)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        write(graph_owner, rows, h, pending)
    counts, mapping, segs, end = (1, 3, 2, 1, 2, 3), [], [], 0
    states = [NS(pos=i, version=0, kv_identity=("int8", "int8"), cur=[0], _kv_error=None, _resize_error=None)
              for i in range(6)]
    for lane in (5, 0, 3, 1, 4, 2):
        count = counts[lane]
        mapping.extend(range(lane * 3, lane * 3 + count))
        segs.append((states[lane], end, end + count))
        end += count
    index = torch.tensor(mapping, dtype=torch.int64, device="cuda")
    try:
        for scale in (0.5, -0.125):
            h.mul_(scale)
            # Future padding must never enter context finite checks.
            for lane, count in enumerate(counts):
                h[lane * 3 + count:(lane + 1) * 3].fill_(float("nan"))
            graph_owner.prepare_replay()
            original = h.clone()
            graph.replay()
            expected = oracle(h, pending).index_select(0, index)
            batch = graph_owner.export_graph(source, segs, index)
            for slot in range(5):
                exact(batch.features.view(-1, 5, 2560)[:, slot], expected)
            exact(h, original)
            assert torch.isfinite(batch.features).all()
            batch.release()
        with pytest.raises(ValueError, match="graph plan"):
            graph_owner.close()
    finally:
        graph.reset()
        graph_owner.retire_graph()
        graph_owner.close()
        source.close()


def test_graph_copy_and_consumers_fence_cross_stream_reuse_without_borrowing_graph_features():
    graph_owner, source = TapWorkspace(config(), 3, "cuda"), TapWorkspace(config(), 3, "cuda")
    h = torch.randn(3, 10240, device="cuda", dtype=torch.float16)
    graph = torch.cuda.CUDAGraph()
    write(graph_owner, 3, h, None)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        write(graph_owner, 3, h, None)
    index = torch.arange(3, device="cuda")
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(producer):
            graph_owner.prepare_replay()
            graph.replay()
            batch = graph_owner.export_graph(source, (), index)
        with torch.cuda.stream(consumer):
            copied = batch.features.clone()
            batch.release()
        with torch.cuda.stream(producer):
            h.fill_(9)
            graph_owner.prepare_replay()
            graph.replay()
            # Source reuse waits for the first consumer, even though graph
            # storage could be overwritten immediately after its independent copy.
            second = graph_owner.export_graph(source, (), index)
            assert torch.all(second.features == 9)
            second.release()
        torch.cuda.synchronize()
        assert not torch.all(copied == 9)
    finally:
        graph.reset()
        graph_owner.retire_graph()
        graph_owner.close()
        source.close()


@pytest.mark.parametrize("cache", ("fp16", "int8"))
def test_six_lane_target_bound_replay_matches_eager_context_and_partial_commit_with_auxiliary(cache):
    w = weights()
    runtime = attach(w)
    dec = MultiDecoder(w, slots=6, capacity=64, depth=2, prefill_rows=8, kv_dtype=cache, graphs=False, keep=0)
    graphs = RoundGraphs(dec, 768 * 2**20)
    try:
        graphs.capture()
        assert graphs.verify and all(st.dspark.pending is None for st in dec.free)
        dec.graphs = graphs
        for i, st in enumerate(dec.free):
            e = _slot(w, st, dec.buf, dec.mbuf, dec.pbuf, 64, 8)
            prefill(e, [4, 17, 3, 99, 33, 2, 7, 5, 6], None, mtp=False)
            if i % 2:
                st.rec[1 - st.cur[0]].copy_(st.rec[st.cur[0]])
                st.cur = [1 - parity for parity in st.cur]
                # This fixture-only equivalent physical parity transition must
                # be explicitly acknowledged by its auxiliary stamp.
                from tensorfold.families.qwen4_exp.cuda.dspark_cuda import _Stamp
                st.dspark.stamp = _Stamp.read(st)
        states, counts = list(reversed(dec.free)), (1, 3, 2, 1, 2, 3)
        reference = Buffers(w, dec.buf.rows, 64, moe_prefill=dec.buf.moe.plan.prefill)
        for iteration in range(2):
            clones = [st.clone() for st in states]
            tokens = [[(11 * i + j + 17 * iteration) % w.cfg.vocab for j in range(count)]
                      for i, count in enumerate(counts)]
            expected_segs = forward.stage(w, reference, list(zip(clones, tokens)))
            # Identical fixed context isolates target/tap graph lifecycle from
            # the separately investigated native attention arithmetic policy.
            expected = forward.compute(w, expected_segs, reference, context=256).clone()
            segs = forward.stage(w, dec.buf, list(zip(states, tokens)), late=True)
            actual = graphs.verify_forward(segs)
            forward._kv_check(segs)
            exact(actual, expected)
            exact(dec.buf.streams[:sum(counts)], reference.streams[:sum(counts)])
            for i, ((st, a0, a1), cloned) in enumerate(zip(segs, clones)):
                assert st.dspark.length == st.pos and st.dspark.pending.rows == a1 - a0
                exact(st.dspark.pending.keys, cloned.dspark.pending.keys)
                exact(st.dspark.pending.values, cloned.dspark.pending.values)
                keep = min(counts[i], 1 + i % 2)
                forward.commit(w, st, dec.buf, counts[i], keep, at=a0)
                forward.commit(w, cloned, reference, counts[i], keep, at=a0)
                target_equal(st, cloned)
                assert st.dspark.length == cloned.dspark.length == st.pos
                exact(st.dspark.keys[:, :st.pos], cloned.dspark.keys[:, :st.pos])
                exact(st.dspark.values[:, :st.pos], cloned.dspark.values[:, :st.pos])
        assert graphs.bound.hits > 0 and graphs.room_bytes <= graphs.budget
        if cache == "int8":
            st = states[0]
            before = st.dspark.length
            segs = forward.stage(w, dec.buf, [(st, [4, 7])], late=True)
            assert graphs.verify_forward(segs) is not None
            st.kv_status[0].fill_(4)
            with pytest.raises(KVNumericError):
                forward._kv_check(segs)
            assert st.dspark.length == before and st.dspark.pending is not None
            runtime.abort(None, segs)
            st.reset(w)
            segs = forward.stage(w, dec.buf, [(st, [4])], late=True)
            assert graphs.verify_forward(segs) is not None
            forward._kv_check(segs)
            forward.commit(w, st, dec.buf, 1, 1)
            assert st.dspark.length == st.pos == 1
        print({"cache": cache, "graph_bytes": graphs.room_bytes, "keys": len(graphs.verify),
               "hits": graphs.bound.hits, "peak_allocated": torch.cuda.max_memory_allocated()}, flush=True)
    finally:
        graphs.close()
        dec.graphs = None
        runtime.close()
