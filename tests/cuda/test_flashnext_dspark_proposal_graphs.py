"""Captured GPU-only DSpark proposals: dynamic committed prefix and safe device greedy."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.dspark_config import DSparkConfig
from tensorfold.families.qwen4_exp.cuda.dspark_cuda import DSparkCUDA, DeviceWeights
from tensorfold.families.qwen4_exp.cuda.dspark_proposal_graphs import ProposalWorkspace, proposal_body
from tensorfold.families.qwen4_exp.cuda.dspark_weights import tensor_shapes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native DSpark proposal capture")


def provider():
    c = DSparkConfig(2560, 128, 5, 4, 1, 128, 128, 2, (0,), 127, 8, 3, 64, 1e-6, 10000)
    gen = torch.Generator(device="cuda").manual_seed(3803)
    arrays = {name: (torch.randn(shape, device="cuda", generator=gen) * 0.007
                    + (1 if "norm" in name else 0)).to(
                        torch.float32 if "norm" in name or name.startswith("confidence_head.") else torch.float16)
              for name, shape in tensor_shapes(c).items()}
    embedding = torch.randn(c.vocab, c.hidden, device="cuda", generator=gen).half() * 0.1
    head = torch.randn(c.vocab, c.hidden, device="cuda", generator=gen).half() * 0.05
    target = NS(hidden=c.hidden, vocab=c.vocab, layers=c.target_layers, streams=4)
    from tensorfold.families.qwen4_exp.cuda.sm70_mtp_weights import Dense
    dense_head = Dense(head)
    p = DSparkCUDA(DeviceWeights(c, arrays, "EXPLICIT-5LAYER-SYNTHETIC-DRAFT"), target=target,
                   embed=lambda ids: embedding.index_select(0, ids), head=lambda x, s: dense_head.matmul(x, f32=True),
                   target_identity="EXPLICIT-SYNTHETIC-SHARED-CALLBACKS", prefill_rows=64)
    return p


def target():
    return NS(pos=0, capacity=64, version=0, kv_identity=("int8", "int8"), cur=[0, 0],
              _kv_error=None, _resize_error=None, _kv_pending=False)


def exact(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("bucket", (4, 32))
def test_native_body_capture_tracks_dynamic_prefix_and_keeps_committed_cache_immutable(bucket):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 proposal graph scope")
    p, host = provider(), target()
    st = p.make_state(host, capacity=64)
    gen = torch.Generator(device="cuda").manual_seed(3805)
    features = torch.randn(8, 2560, device="cuda", generator=gen).half() * 0.03
    update = p.stage_context(st, features[:4])
    host.pos = 4
    p.commit(st, update, 4)
    st.keys[:, 4:].fill_(float("nan"))
    st.values[:, 4:].fill_(float("nan"))
    ws = ProposalWorkspace(p, st, 3, bucket)
    graph = torch.cuda.CUDAGraph()
    ws.allocate()
    assert ws.nbytes == ws.device_bytes(p.shard.width, 3, bucket, p.world)
    ws.prepare(4)
    proposal_body(p, ws)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        proposal_body(p, ws)
    try:
        for length, bonus in ((1, 4), (4, 17), (2, 9)):
            # Keep the same independently owned physical committed prefix while
            # deliberately moving its logical cursor within one graph bucket.
            host.pos = st.length = length
            before = st.keys.clone(), st.values.clone(), st.version
            ws.prepare(bonus)
            expected_ids = proposal_body(p, ws).clone()
            expected_scores = ws.corrected.clone()
            ws.prepare(bonus)
            graph.replay()
            actual_ids = ws.finish()
            assert actual_ids == expected_ids.tolist()
            exact(ws.corrected, expected_scores)
            exact(st.keys, before[0])
            exact(st.values, before[1])
            assert st.version == before[2]
            if bucket == length:
                eager = p.propose(st, bonus, block=3)
                assert actual_ids == eager.ids.tolist()
                exact(ws.corrected, eager.corrected_logits)
        assert ws.readbacks == ws.calls == 3 and ws.steps == 9
        print({"block": 3, "layers": 5, "bucket": bucket, "resident_bytes": ws.nbytes,
               "pinned_bytes": ws.host_bytes, "readbacks": ws.readbacks}, flush=True)
    finally:
        graph.reset()
        ws.close()
        p.close_state(st)


def test_native_nonfinite_status_is_sticky_safe_ids_prevent_bad_markov_lookup_and_replay_recovers():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 proposal graph scope")
    p, host = provider(), target()
    st = p.make_state(host, capacity=64)
    features = torch.zeros(4, 2560, dtype=torch.float16, device="cuda")
    update = p.stage_context(st, features)
    host.pos = 4
    p.commit(st, update, 4)
    ws, graph = ProposalWorkspace(p, st, 3, 32), torch.cuda.CUDAGraph()
    ws.allocate()
    ws.prepare(4)
    proposal_body(p, ws)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        proposal_body(p, ws)
    try:
        st.keys[:, :4].fill_(float("nan"))
        ws.prepare(4)
        graph.replay()
        with pytest.raises(ValueError, match="nonfinite"):
            ws.finish()
        assert ws.previous.item() == 0 and ws.readbacks == 1 and ws.calls == 0
        st.keys[:, :4].zero_()
        st.values[:, :4].zero_()
        ws.prepare(4)
        graph.replay()
        ids = ws.finish()
        assert len(ids) == 3 and all(0 <= i < p.config.vocab for i in ids)
        assert ws.readbacks == 2 and ws.calls == 1
        assert st.length == host.pos == 4 and st.pending is None
    finally:
        graph.reset()
        ws.close()
        p.close_state(st)


@pytest.mark.parametrize("cache", ("fp16", "int8"))
def test_existing_bound_owner_admits_proposals_reuses_hot_keys_then_retires_all_borrowed_cache_graphs(cache, monkeypatch):
    from test_flashnext_dspark_runtime import attach, weights
    from tensorfold.cuda.sm70 import family_ops
    from tensorfold.families.qwen4_exp.cuda import dspark_proposal_graphs as module, forward
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder, _slot
    from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs

    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    # Keep this short-context ownership fixture on the original Torch operator.
    # Qualified native attention (65..8192) has its independent exact oracle.
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)
    monkeypatch.setattr(module, "_ENABLED", True)
    w = weights()
    runtime = attach(w)
    dec = MultiDecoder(w, slots=6, capacity=64, depth=2, prefill_rows=8, kv_dtype=cache, graphs=False, keep=0)
    graphs = RoundGraphs(dec, 768 * 2**20)
    try:
        graphs.capture()
        proposals = runtime._proposal_graphs
        assert proposals is not None and len(proposals.slots) == 6
        assert all(st.dspark.pending is None and st.dspark.length == 0 for st in dec.free)
        before = graphs.room_bytes
        state = dec.free[0]
        engine = _slot(w, state, dec.buf, dec.mbuf, dec.pbuf, 64, 8)
        aux_before = state.dspark.keys.clone(), state.dspark.values.clone()
        # Existing key: account/memory_snapshot and admission votes are forbidden.
        with monkeypatch.context() as hot:
            hot.setattr(graphs.bound, "_account", lambda *a: (_ for _ in ()).throw(AssertionError("hot accounting")))
            hot.setattr(graphs.bound, "_agree", lambda *a: (_ for _ in ()).throw(AssertionError("hot admission vote")))
            for bonus in (4, 7):
                ids = runtime.draft(engine, bonus, 2, None, 0)
                assert len(ids) == 2 and all(0 <= x < w.cfg.vocab for x in ids)
        assert graphs.room_bytes == before
        exact(state.dspark.keys, aux_before[0])
        exact(state.dspark.values, aux_before[1])
        # Commit target-derived context, then the SAME static proposal key must
        # read its new device cursor/prefix without changing cache ownership.
        segs = forward.stage(w, dec.buf, [(state, [4, 7])], late=True)
        assert graphs.verify_forward(segs) is not None
        forward._kv_check(segs)
        forward.commit(w, state, dec.buf, 2, 2)
        ids = runtime.draft(engine, 19, 2, None, 0)
        assert len(ids) == 2 and state.dspark.length == state.pos == 2
        assert proposals.record()["readbacks"] == proposals.record()["calls"] == 3
        assert len(proposals.slots) == 6
        # Short blocks and stochastic/learned confidence retain the normal
        # authority; probing eligibility must not run a graph or publish state.
        assert proposals.draft(state, 19, 1, None, 0) is None
        from tensorfold.engine.exact_sampling import Sampling
        assert proposals.draft(state, 19, 2, Sampling(1), 0) is None
        assert proposals.draft(state, 19, 2, None, 0.5) is None
        with pytest.raises(ValueError, match="must retire"):
            runtime.close()
        assert runtime.provider is not None and not runtime.closed
        print({"cache": cache, "root_budget": graphs.budget, "root_retained_bytes": graphs.room_bytes,
               "proposals": proposals.record()}, flush=True)
        graphs.close()
        assert proposals.closed and not proposals.slots and runtime._proposal_graphs is None
        assert not graphs.verify and not graphs.pending
        # Retry close after retirement succeeds; no lost owner or poisoned cache.
        runtime.close()
        assert runtime.closed
    finally:
        graphs.close()
        runtime.close()
