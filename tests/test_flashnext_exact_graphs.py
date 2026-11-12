"""Exact unique-history graph geometry and live handoff contracts, CPU only."""

from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda import exact_graphs


def owner():
    states = [SimpleNamespace(kv_identity=("q8",), version=0, cur=[0, 0],
                              pos=8, mtp_len=7, image_positions=None) for _ in range(2)]
    source = SimpleNamespace(rows=10, rotor_lookup=False)
    model = SimpleNamespace(mtp=SimpleNamespace(layer=SimpleNamespace(linear=False)), comm=None)
    outer = SimpleNamespace(dec=SimpleNamespace(free=states, buf=source, mbuf=source, w=model),
                            verify={}, mtp={}, failure=None)
    graph = exact_graphs.ExactStateGraphs(outer)
    return graph, states


@pytest.mark.parametrize('field,value', [('sm70', False), ('gguf', False), ('owned', False),
    ('adaptive', False), ('tp', 4), ('etp', 2), ('indexer_history', 'ram')])
def test_partition_and_history_legacy_disables_exempt_only_the_exact_native_owner(field, value):
    from tensorfold.families.qwen4_exp.cuda.engine import _owned_graph_request
    request = dict(sm70=True, gguf=True, owned=True, adaptive=True, tp=8, etp=4, indexer_history='gpu')
    assert _owned_graph_request(**request) is True
    request[field] = value
    assert _owned_graph_request(**request) is False


def test_buffer_only_key_retains_rows_and_participants_without_state_addresses():
    graph, (a, b) = owner()
    first = graph._key("verify", [(a, 0, 1), (b, 1, 6)])
    a.pos += 20  # Live State properties belong to the eager handoff.
    assert graph._key("verify", [(a, 0, 1), (b, 1, 6)]) == first
    assert graph._key("verify", [(a, 0, 2), (b, 2, 6)]) == first
    assert graph._key("verify", [(a, 0, 2), (b, 2, 7)]) != first
    assert graph._key("verify", [(a, 0, 6)]) != first
    a.cur[0] = 1
    a.version += 1
    assert graph._key("verify", [(a, 0, 1), (b, 1, 6)]) == first
    assert graph._key("verify", [(b, 0, 1), (a, 1, 6)]) == first


def test_attention_mtp_key_does_not_depend_on_unused_target_recurrent_parity():
    graph, (state, _) = owner()
    first = graph._key("mtp", [(state, 0, 1)])
    state.cur[:] = [1, 1]
    assert graph._key("mtp", [(state, 0, 1)]) == first
    graph.dec.w.mtp.layer.linear = True
    assert graph._key("mtp", [(state, 0, 1)]) == first


@pytest.mark.parametrize("segments", [lambda a, b: [(a, 1, 2)], lambda a, b: [(a, 0, 0)],
                                     lambda a, b: [(a, 0, 11)],
                                     lambda a, b: [(a, 0, 2), (b, 1, 3)]])
def test_exact_key_rejects_invalid_live_row_coverage(segments):
    graph, (a, b) = owner()
    with pytest.raises(ValueError, match="compact nonempty live rows"):
        graph._key("verify", segments(a, b))


def test_exact_key_rejects_foreign_or_padded_state():
    graph, (state, _) = owner()
    with pytest.raises(ValueError, match="original decoder State"):
        graph._key("verify", [(SimpleNamespace(), 0, 1)])
    state.graph_rows = object()
    with pytest.raises(ValueError, match="padded State"):
        graph._key("verify", [(state, 0, 1)])


def test_live_miss_validates_current_coordinate_frame_and_never_captures(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import compact_indexer

    graph, (state, _) = owner()
    observed = []
    monkeypatch.setattr(compact_indexer, "validate_target_indexer_frame",
                        lambda buffers, segments: observed.append((buffers, segments)))
    monkeypatch.setattr(graph, "_capture", lambda *args: pytest.fail("live capture forbidden"))
    segments = [(state, 0, 5)]
    assert graph.forward("verify", segments) is None
    assert observed == [(graph.dec.buf, segments)]
    assert graph.misses == 1


def test_replay_keeps_live_owner_handoff_between_gpu_pieces(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import compact_indexer, history_owner

    graph, (state, _) = owner()
    order = []
    segments = [(state, 0, 5)]
    pieces = SimpleNamespace(graphs=["before", "middle", "after"],
                             inserts=[("owner_attention", "layer", 5, False),
                                      ("gdn_chain", "recurrent", 5)], out=object())
    graph.owner.verify[graph._key("verify", segments)] = pieces
    monkeypatch.setattr(compact_indexer, "validate_target_indexer_frame",
                        lambda buffers, segs: order.append("validate"))
    monkeypatch.setattr(exact_graphs, "_kv_begin", lambda segs: order.append("status"))
    monkeypatch.setattr(exact_graphs, "replay", lambda piece, comm: order.append(piece))
    monkeypatch.setattr(history_owner, "attention_layer",
                        lambda layer, weights, segs, buffers, rows, **kwargs:
                        order.append((segs[0][0].pos, rows, kwargs["mtp"])))
    monkeypatch.setattr(exact_graphs, "gdn_chain", lambda layer, weights, segs, buffers, rows:
                        order.append((segs[0][0].cur[:], rows)))
    state.pos = 300
    state.cur[:] = [1, 0]
    assert graph.forward("verify", segments) is pieces.out
    assert order == ["validate", "status", "before", (300, 5, False), "middle", ([1, 0], 5), "after"]
    assert graph.record()["capture_scope"] == "exact-row-gpu-pieces-with-eager-state-handoffs"
    assert graph.owner_inserts == graph.hits == 1
    assert graph.record()['serving_hits'] == {'verify': 1, 'mtp': 0}
    graph.preparing = True
    graph.forward('verify', segments)
    assert graph.hits == 2
    assert graph.record()['serving_hits'] == {'verify': 1, 'mtp': 0}
    graph.preparing = False
    monkeypatch.setattr(compact_indexer, "validate_mtp_indexer_frame", lambda buffers, segs: None)
    assert graph.forward('mtp', segments) is None
    assert graph.record()['serving_misses'] == {'verify': 0, 'mtp': 1}


def test_unique_capture_requires_actual_adopted_native_experts_and_gpu_history():
    from tensorfold.families.qwen4_exp.cuda import multi_graphs, owned_experts

    plan = object()
    bank = SimpleNamespace(_owned_expert_plan=plan, implementation=owned_experts.POLICY, capturable=False)
    weights = SimpleNamespace(meta={"history_layout": "unique", "execution_provider": "sm70",
                                    "owned_expert_plan": plan}, comm=object(), mtp=None,
                              layers=[SimpleNamespace(moe=SimpleNamespace(experts=bank), ple=None)])
    assert multi_graphs.capturable(weights) is None
    weights.dflash2 = object()
    assert "stock MTP" in multi_graphs.capturable(weights)
    weights.dflash2 = None
    weights.dspark = object()
    assert "stock MTP" in multi_graphs.capturable(weights)
    weights.dspark = None
    weights.meta["indexer_history"] = "ram"
    assert "GPU-resident" in multi_graphs.capturable(weights)
    weights.meta["indexer_history"] = "gpu"
    bank._owned_expert_plan = object()
    assert "prepared fixed native expert" in multi_graphs.capturable(weights)
    bank._owned_expert_plan = plan
    weights.meta["history_layout"] = "replicated"
    assert "capture-safe state" in multi_graphs.capturable(weights)


def test_capture_entry_refuses_serving_before_any_device_work():
    graph, _ = owner()
    with pytest.raises(RuntimeError, match="idle startup"):
        graph._capture("verify", (), (), None)


def test_buffer_only_graph_survives_state_invalidation_and_retires_at_owner_close(monkeypatch):
    graph, (state, other) = owner()
    order = []
    monkeypatch.setattr(exact_graphs.torch.cuda, "synchronize", lambda device: order.append("fence"))
    graph.dec.w.device = "cuda:0"
    key = graph._key("verify", [(state, 0, 1)])
    pieces = SimpleNamespace(graphs=[SimpleNamespace(reset=lambda: order.append("reset"))])
    graph.bindings[("verify", key)] = pieces
    graph.owner.verify[key] = pieces
    graph.invalidate(other)
    assert key in graph.owner.verify
    graph.invalidate(state)
    assert not order
    graph.close()
    assert order == ["fence", "reset"]
    assert not graph.bindings and not graph.owner.verify
