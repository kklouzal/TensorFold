"""Exact unique-history graph geometry and live handoff contracts, CPU only."""

from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda import exact_graphs


def owner(*, slots=2, depth=4, rows=10, mtp=True, policy=None):
    states = [SimpleNamespace(kv_identity=("q8",), version=0, cur=[0, 0],
                              pos=8, mtp_len=7, image_positions=None) for _ in range(slots)]
    source = SimpleNamespace(rows=rows, rotor_lookup=False)
    model = SimpleNamespace(mtp=SimpleNamespace(layer=SimpleNamespace(linear=False)), comm=None)
    outer = SimpleNamespace(dec=SimpleNamespace(free=states, buf=source, mbuf=source if mtp else None, w=model,
                                                depth=depth, exact_graph_policy=policy,
                                                streams={}, filling=[], kept=[]),
                            verify={}, mtp={}, failure=None, pending=[])
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


@pytest.mark.parametrize('role', ['verify', 'mtp'])
@pytest.mark.parametrize('split,exact,entry', [(False, False, True), (True, False, True),
                                             (False, True, True), (True, True, False)])
def test_only_exact_capture_handoff_omits_repeated_host_state_entry(monkeypatch, role, split, exact, entry):
    import torch
    from tensorfold.families.qwen4_exp.cuda import compact_indexer, forward, mtp

    events = []
    class GPUProgramBoundary(Exception):
        pass
    def gpu(*args, **kwargs):
        events.append('GPU-program')
        raise GPUProgramBoundary
    module = forward if role == 'verify' else mtp
    validator = 'validate_target_indexer_frame' if role == 'verify' else 'validate_mtp_indexer_frame'
    monkeypatch.setattr(compact_indexer, validator, lambda *args: events.append('coordinate-frame'))
    monkeypatch.setattr(module, '_kv_begin', lambda *args: events.append('State-transfer-entry'))
    monkeypatch.setattr(module, '_embed', gpu)
    buffers = SimpleNamespace(split=(lambda value: None) if split else None,
        exact_graph_handoffs=exact, compact_indexer_stage=object(), prefill=False,
        ids=torch.zeros(1, dtype=torch.int32), h=torch.empty((1, 4)), mtp_e=torch.empty((1, 1)))
    weights = SimpleNamespace(cfg=SimpleNamespace(streams=4), meta={}, mtp=object())
    with pytest.raises(GPUProgramBoundary):
        (forward.compute if role == 'verify' else mtp.mtp_compute)(
            weights, [(SimpleNamespace(), 0, 1)], buffers, check=False)
    assert events == (['coordinate-frame', 'State-transfer-entry'] if entry else []) + ['GPU-program']


def test_exact_replay_revalidates_current_frame_before_any_state_entry_or_gpu_piece(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import compact_indexer

    graph, (state, _) = owner()
    segments = [(state, 0, 1)]
    graph.owner.verify[graph._key('verify', segments)] = SimpleNamespace(
        graphs=[object()], inserts=[], out=object())
    failure = ValueError('issued coordinate frame became stale')
    def refuse(*args):
        raise failure
    monkeypatch.setattr(compact_indexer, 'validate_target_indexer_frame', refuse)
    monkeypatch.setattr(exact_graphs, '_kv_begin', lambda *args: pytest.fail('stale frame reached State entry'))
    monkeypatch.setattr(exact_graphs, 'replay', lambda *args: pytest.fail('stale frame reached GPU replay'))
    with pytest.raises(ValueError) as found:
        graph.forward('verify', segments)
    assert found.value is failure


def test_capture_handoff_lifetime_resets_after_piece_failure(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import multi_graphs

    graph, (state, _) = owner()
    graph.dec.w.device = 'cuda:0'
    graph.owner.room_bytes, graph.owner.budget = 0, 1024
    graph.owner._has_room = lambda: True
    graph.preparing = True
    graph.selected_keys = frozenset({('verify', 1, 1)})
    graph.dec.buf.split = None
    graph.dec.buf.exact_graph_handoffs = False
    events = []
    monkeypatch.setattr(graph, '_agree', lambda allowed: allowed)
    monkeypatch.setattr(exact_graphs, 'compute', lambda w, segs, b, **kw:
                        events.append(('run', b.split is not None, b.exact_graph_handoffs)))
    monkeypatch.setattr(exact_graphs, '_kv_check', lambda *args, **kw: events.append('eager-status-check'))
    monkeypatch.setattr(exact_graphs.torch.cuda, 'synchronize', lambda *args: events.append('eager-device-fence'))
    monkeypatch.setattr(multi_graphs, 'free_bytes', lambda: 1024)
    monkeypatch.setattr(exact_graphs.torch.cuda, 'memory_allocated', lambda device: 0)
    monkeypatch.setattr(exact_graphs.torch.cuda, 'memory_reserved', lambda device: 0)
    failure = RuntimeError('controlled capture failed')
    def pieces(run, buffers, *, retain_pending):
        assert retain_pending is True
        buffers.split = lambda value: None
        try:
            run()
            raise failure
        finally:
            buffers.split = None
    graph.owner._pieces = pieces
    monkeypatch.setattr(graph, '_restage_warm', lambda kind, segs, source:
                        events.append('private-reset-fresh-stage') or segs)
    with pytest.raises(RuntimeError) as found:
        graph._capture('verify', (1, 1), [(state, 0, 1)], graph.dec.buf)
    assert found.value is failure
    assert events == [('run', False, False), 'private-reset-fresh-stage', ('run', True, True)]
    assert graph.dec.buf.split is None and graph.dec.buf.exact_graph_handoffs is False


@pytest.mark.parametrize('piece_diagnostic', [False, True])
@pytest.mark.parametrize('budget,allowed,write_failure,physical_delta', [(1024, True, False, 128), (127, False, False, 128), (127, False, True, 128), (1024, True, False, -128)])
def test_startup_capture_preserves_per_key_allocation_receipt_before_fatal_budget_refusal(monkeypatch, budget, allowed, write_failure, physical_delta, piece_diagnostic):
    import json
    from tensorfold.families.qwen4_exp.cuda import multi_graphs

    graph, (state, _) = owner()
    graph.preparing = True
    graph.selected_keys = frozenset({('verify', 1, 1)})
    graph.dec.w.device = 'cuda:0'
    outer = graph.owner
    outer.room_bytes, outer.budget, outer.bytes, outer.measured = 0, budget, 0, []
    outer._has_room = lambda: True
    outer.retain_failure = lambda failure: setattr(outer, 'failure', failure)
    pieces = SimpleNamespace(graphs=[SimpleNamespace(reset=lambda: None)], out=object())
    graph.piece_diagnostic = piece_diagnostic and not write_failure
    def capture(*args, retain_pending, **kwargs):
        assert retain_pending is True
        if graph.piece_diagnostic:
            kwargs['observer'](0, ('gdn_chain', SimpleNamespace(index=3, tensor=object()), 1), {'physical_delta_bytes': -64})
        else:
            assert not kwargs
        outer.pending.append(pieces)
        return pieces
    outer._pieces = capture
    monkeypatch.setattr(exact_graphs, 'compute', lambda *args, **kw: None)
    monkeypatch.setattr(graph, '_restage_warm', lambda kind, segs, source: segs)
    monkeypatch.setattr(exact_graphs.torch.cuda, 'synchronize', lambda *args: None)
    free, allocated, reserved = iter([1024, 1024 - physical_delta]), iter([100, 132]), iter([200, 264])
    monkeypatch.setattr(multi_graphs, 'free_bytes', lambda: next(free))
    monkeypatch.setattr(exact_graphs.torch.cuda, 'memory_allocated', lambda device: next(allocated))
    monkeypatch.setattr(exact_graphs.torch.cuda, 'memory_reserved', lambda device: next(reserved))
    receipts = []
    diagnostic = OSError('controlled startup receipt write failure')
    def publish(fd, payload):
        if write_failure:
            raise diagnostic
        receipts.append((fd, payload))
        return len(payload)
    monkeypatch.setattr(exact_graphs.os, 'write', publish)
    if allowed:
        assert graph._capture('verify', (1, 1), [(state, 0, 1)], graph.dec.buf) is pieces
    else:
        with pytest.raises(MemoryError, match='rows=1 participants=1 pieces=1 prior_keys=0'):
            graph._capture('verify', (1, 1), [(state, 0, 1)], graph.dec.buf)
    assert graph.bindings[('verify', (1, 1))] is pieces  # Borrowers retained even on refusal.
    if write_failure:
        assert not receipts and outer.failure.__cause__ is diagnostic
        assert isinstance(outer.failure, MemoryError) and not outer.verify
        return
    if graph.piece_diagnostic:
        piece = json.loads(receipts.pop(0)[1])
        assert piece['schema'] == 'tensorfold.exact-graph-piece-v1'
        assert piece['terminating_insert'] == {'kind': 'gdn_chain', 'layer': 3, 'rows': 1}
        assert piece['accounting']['physical_delta_bytes'] == -64
        assert piece['diagnostic_only'] is True and piece['P3_valid'] is False
    assert len(receipts) == 1 and receipts[0][0] == 2 and len(receipts[0][1]) < 4096
    record = json.loads(receipts[0][1])
    assert record['physical_delta_bytes'] == physical_delta
    assert outer.measured[0]['bytes'] == max(0, physical_delta)
    assert record['allocated_delta_bytes'] == 32 and record['reserved_delta_bytes'] == 64
    assert record['prior_keys'] == 0 and record['admitted'] is allowed
    assert outer.measured[0]['accounting'] == record
    assert bool(outer.verify) is allowed
    if allowed:
        assert graph.record()['exact_graph_catalogue']['captured_keys'] == [['verify', 1, 1]]
        graph.close()
        assert not outer.verify
        assert graph.record()['exact_graph_catalogue']['captured_keys'] == [['verify', 1, 1]]


def test_private_reset_retires_real_unresolved_compact_staging_without_weakening_append_guard(monkeypatch):
    import torch
    from test_compact_indexer import cache, begin
    from tensorfold.families.qwen4_exp.cuda.compact_indexer import Staging

    graph, (state, _) = owner()
    graph.dec.w.device = 'cuda:0'
    graph.preparing = True
    entry = cache()
    staging = Staging(1, 12, 'cpu', mtp=False)
    begin(entry, staging, torch.ones((1, 128), dtype=torch.float16), torch.zeros((1, 3), dtype=torch.int64))
    with pytest.raises(ValueError, match='unresolved raw staging'):
        entry.validate_append(0, 1, (0, 0))
    events = []
    monkeypatch.setattr(exact_graphs, '_kv_check', lambda *a, **kw: events.append('validate-status'))
    monkeypatch.setattr(exact_graphs.torch.cuda, 'synchronize', lambda *a: events.append('retire-device-work'))
    def reset(weights, *, new_sequence):
        assert new_sequence is False and events == ['validate-status', 'retire-device-work']
        entry.reset((0, 0))
        events.append('reset-private-state')
    state.reset = reset
    graph._reset_warm_states([(state, 0, 1)])
    assert events == ['validate-status', 'retire-device-work', 'reset-private-state']
    assert entry.span is None and entry.n == 0
    entry.validate_append(0, 1, (0, 0))  # Original frontier guard now has a genuinely reset cache.


@pytest.mark.parametrize('role', ['verify', 'mtp'])
def test_reset_restages_fresh_coordinates_and_preserves_owned_mtp_hidden_input(monkeypatch, role):
    import torch
    from tensorfold.families.qwen4_exp.cuda import compact_indexer

    graph, (state, _) = owner()
    graph.dec.w.device = 'cuda:0'
    graph.preparing = True
    source = graph.dec.buf
    source.ids_host = torch.tensor([17, 19], dtype=torch.int32)
    source.mtp_in = torch.arange(6, dtype=torch.float16).view(2, 3)
    source.ple_late = None
    state.generation = 3
    events = []
    monkeypatch.setattr(exact_graphs, '_kv_check', lambda *a, **kw: events.append('old-status-valid'))
    monkeypatch.setattr(exact_graphs.torch.cuda, 'synchronize', lambda *a: events.append('fence'))
    def reset(weights, *, new_sequence):
        assert new_sequence is False
        state.generation += 1
        events.append('private-reset')
    state.reset = reset
    def stage(*args, **kwargs):
        windows = args[-1]
        assert windows[0][0] is state and windows[0][1] == [17, 19]
        if role == 'mtp':
            assert windows[0][2].data_ptr() == source.mtp_in.data_ptr()
            assert torch.equal(windows[0][2], torch.arange(6, dtype=torch.float16).view(2, 3))
        source.fresh_generation = state.generation
        events.append('fresh-stage')
        return [(state, 0, 2)]
    monkeypatch.setattr(exact_graphs, 'stage' if role == 'verify' else 'mtp_stage', stage)
    def validate(buffers, segs):
        assert buffers.fresh_generation == state.generation == 4
        events.append('fresh-coordinate-valid')
    monkeypatch.setattr(compact_indexer, 'validate_target_indexer_frame' if role == 'verify'
                        else 'validate_mtp_indexer_frame', validate)
    monkeypatch.setattr(exact_graphs, '_kv_begin', lambda *a: events.append('fresh-State-entry'))
    assert graph._restage_warm(role, [(state, 0, 2)], source) == [(state, 0, 2)]
    assert events == ['old-status-valid', 'fence', 'private-reset', 'fresh-stage',
                      'fresh-coordinate-valid', 'fresh-State-entry', 'fence']


def test_private_reset_never_runs_for_serving_or_failed_status(monkeypatch):
    graph, (state, _) = owner()
    monkeypatch.setattr(exact_graphs, '_kv_check', lambda *a, **kw: pytest.fail('serving reset reached State work'))
    with pytest.raises(RuntimeError, match='idle startup'):
        graph._reset_warm_states([(state, 0, 1)])
    graph.preparing = True
    graph.dec.streams[0] = object()
    with pytest.raises(RuntimeError, match='idle startup'):
        graph._reset_warm_states([(state, 0, 1)])
    graph.dec.streams.clear()
    failure = ValueError('actual warm status failed')
    def failed(*args, **kwargs):
        raise failure
    monkeypatch.setattr(exact_graphs, '_kv_check', failed)
    monkeypatch.setattr(exact_graphs.torch.cuda, 'synchronize', lambda *a: pytest.fail('failed work was reset'))
    state.reset = lambda *a, **kw: pytest.fail('failed work was cleared')
    with pytest.raises(ValueError) as found:
        graph._reset_warm_states([(state, 0, 1)])
    assert found.value is failure


def test_native_gdn_replay_uses_prepared_frame_without_upload_and_poison_stale_registration(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import compact_indexer, input_preparation

    graph, (state, _) = owner()
    graph.native_gdn_capture = True
    source, events = graph.dec.buf, []
    monkeypatch.setattr(graph, '_native_frame', lambda *a, **kw: pytest.fail('serving descriptor re-upload'))
    monkeypatch.setattr(compact_indexer, 'validate_target_indexer_frame', lambda *a: None)
    monkeypatch.setattr(exact_graphs, '_kv_begin', lambda *a: events.append('State-entry'))
    monkeypatch.setattr(exact_graphs, 'replay', lambda *a: events.append('replay'))
    segs = [(state, 0, 5)]
    pieces = SimpleNamespace(graphs=['one'], inserts=[], out=object())
    graph.owner.verify[graph._key('verify', segs)] = pieces
    for parity in (0, 1):
        state.cur[0] = parity
        source.gdn_tables = SimpleNamespace(validate_current=lambda *a: events.append(('prepared-parity', state.cur[0])))
        assert graph.forward('verify', segs) is pieces.out
    assert events == [('prepared-parity', 0), 'State-entry', 'replay',
                      ('prepared-parity', 1), 'State-entry', 'replay']
    failure = ValueError('fixed registration changed')
    def stale(*args):
        raise failure
    def poison(buffers, error):
        assert buffers is source and error is failure
        events.append('cohort-poisoned')
        raise error
    source.gdn_tables.validate_current = stale
    monkeypatch.setattr(input_preparation, 'reject_prepared', poison)
    with pytest.raises(ValueError) as caught:
        graph.forward('verify', segs)
    assert caught.value is failure and events[-1] == 'cohort-poisoned'
    assert graph.serving_misses['verify'] == 0


def test_prevalidated_gdn_capture_submits_existing_kernel_without_a_state_handoff(monkeypatch):
    from tests.test_sm70_gdn_batch import fixture
    from tensorfold.cuda.sm70 import gdn_native
    from tensorfold.families.qwen4_exp.cuda import forward

    static, source, states, actor = fixture(monkeypatch, width=5)
    table = static.native_frame([(states[0], 0, 5)], source, allow_single=True)
    operands = tuple(table.operands(li, source.proj[li, :5], source.gout[:5], source.gxs[:5])
                     for li in range(static.lin))
    source.gdn_capture_operands, source.exact_graph_handoffs = operands, True
    source.split = lambda *a: pytest.fail('prepared GDN must stay in its GPU piece')
    states[0].lin_index = {7: 0}
    calls = []
    monkeypatch.setattr(exact_graphs.torch.cuda, 'current_stream', lambda device: object())
    monkeypatch.setattr(gdn_native, 'extension', lambda: SimpleNamespace(batch_chain=lambda *a: calls.append(a)))
    monkeypatch.setattr(forward, '_mm', lambda *a: None)
    monkeypatch.setattr(forward, '_out_proj', lambda *a: 'output-projection')
    source.mixed, source.xs_mixed = source.h, source.gxs
    gdn = SimpleNamespace(proj=object(), conv=object(), a_log=None, a_neg=object(), dt_bias=object(),
                          norm=object(), out=object())
    layer = SimpleNamespace(index=7, gdn=gdn)
    assert forward.gdn_block(layer, SimpleNamespace(cfg=SimpleNamespace(eps=1e-6)),
                             [(states[0], 0, 5)], source, 5) == 'output-projection'
    assert len(calls) == 1 and calls[0][6] is operands[0][0] and calls[0][7] is operands[0][1]
    with pytest.raises(ValueError, match='current'):
        table.operands(0, source.proj[0, :5], source.gout[:5], source.gxs[:5])


@pytest.mark.parametrize('value', ['false', 'true', '2', '1\n'])
def test_piece_diagnostic_rejects_invalid_environment_before_capture(monkeypatch, value):
    monkeypatch.setenv('TENSORFOLD_EXACT_GRAPH_PIECE_DIAGNOSTIC', value)
    with pytest.raises(ValueError, match='must be 0 or 1'):
        owner()


def test_piece_diagnostic_is_explicit_and_startup_only(monkeypatch):
    monkeypatch.delenv('TENSORFOLD_EXACT_GRAPH_PIECE_DIAGNOSTIC', raising=False)
    graph, _ = owner()
    assert graph.piece_diagnostic is False
    monkeypatch.setenv('TENSORFOLD_EXACT_GRAPH_PIECE_DIAGNOSTIC', '1')
    graph, _ = owner()
    assert graph.piece_diagnostic is True


def test_default_catalogue_retains_all_original_MTP4_keys_and_order():
    graph, states = owner(slots=24, rows=120)
    shapes = graph._catalogue(states)
    assert shapes == ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                      (2, 1), (2, 5), (4, 1), (4, 5), (8, 1), (8, 5),
                      (16, 1), (16, 5), (24, 1), (24, 5))
    record = graph.record()['exact_graph_catalogue']
    assert record['policy'] == 'full-original-v1' and record['evidence_sha256'] is None
    assert len(record['available_keys']) == len(record['requested_keys']) == len(record['selected_keys']) == 30
    assert record['selected_keys'][:4] == [['verify', 1, 1], ['mtp', 1, 1], ['verify', 2, 1], ['mtp', 2, 1]]
    assert record['declined_keys'] == record['captured_keys'] == []


@pytest.mark.parametrize('key,mtp', [(['verify', 6, 1], True), (['verify', 3, 3], True), (['mtp', 1, 1], False)])
def test_requested_subset_geometry_and_role_validate_before_any_State_or_GPU_work(monkeypatch, key, mtp):
    policy = {'policy': 'paid-subset-v1', 'keys': [key], 'evidence_sha256': 'a'*64}
    graph, states = owner(policy=policy, mtp=mtp)
    monkeypatch.setattr(graph, '_reset_warm_states', lambda *a: pytest.fail('invalid subset touched State'))
    monkeypatch.setattr(exact_graphs, 'stage', lambda *a, **kw: pytest.fail('invalid subset staged GPU inputs'))
    with pytest.raises(ValueError, match='outside the actual startup catalogue'):
        graph.warm(states)
    assert graph.preparing is False and graph.bindings == {} and graph.selected_keys is None


def test_paid_subset_snapshot_is_immutable_and_only_selected_keys_may_capture(monkeypatch):
    policy = {'policy': 'paid-subset-v1', 'keys': [['mtp', 5, 1]], 'evidence_sha256': 'b'*64}
    graph, states = owner(policy=policy)
    policy['keys'][0][0] = 'verify'
    graph._catalogue(states)
    assert graph.selected_keys == frozenset({('mtp', 5, 1)})
    record = graph.record()['exact_graph_catalogue']
    assert record['requested_keys'] == record['selected_keys'] == [['mtp', 5, 1]]
    assert record['evidence_sha256'] == 'b'*64 and len(record['declined_keys']) == 13
    assert all(row['reason'] == 'excluded by paid-subset policy' for row in record['declined_keys'])
    record['selected_keys'][0][0] = 'verify'
    assert graph.selected_keys == frozenset({('mtp', 5, 1)})
    graph.preparing = True
    monkeypatch.setattr(graph, '_agree', lambda *a: pytest.fail('unselected key reached device agreement'))
    with pytest.raises(RuntimeError, match='selected startup catalogue key'):
        graph._capture('verify', (5, 1), [(states[0], 0, 5)], graph.dec.buf)


def test_subset_warm_eagerly_produces_fresh_target_HC_for_selected_MTP_only_shape(monkeypatch):
    import torch

    policy = {'policy': 'paid-subset-v1', 'keys': [['mtp', 5, 1], ['verify', 2, 1]], 'evidence_sha256': 'c'*64}
    graph, states = owner(policy=policy)
    source = graph.dec.buf
    source.ple_late = None
    source.streams = torch.full((10, 2), -999, dtype=torch.float16)
    events = []
    def stage(weights, buffers, windows, **kwargs):
        segs, end = [], 0
        for state, tokens in windows:
            segs.append((state, end, end+len(tokens)))
            end += len(tokens)
        events.append(('target-stage', end))
        return segs
    def reset(segs):
        events.append(('reset', segs[-1][2]))
    def forward(role, segs):
        rows = segs[-1][2]
        assert (role, rows, len(segs)) in graph.selected_keys
        events.append(('capture', role, rows))
        if role == 'verify':
            source.streams[:rows].fill_(21)
        return object()
    def compute(weights, segs, buffers, *, check):
        assert check is False and buffers is source and segs[-1][2] == 5
        events.append(('eager-target-HC', 5))
        source.streams[:5].fill_(51)
    def mtp_stage(buffers, windows):
        assert len(windows) == 1 and windows[0][1] == [0]*5
        assert windows[0][2].data_ptr() == source.streams.data_ptr()
        assert torch.equal(windows[0][2], torch.full((5, 2), 51, dtype=torch.float16))
        events.append(('MTP-stage-valid-HC', 5))
        return [(windows[0][0], 0, 5)]
    monkeypatch.setattr(graph, '_reset_warm_states', reset)
    monkeypatch.setattr(graph, 'forward', forward)
    monkeypatch.setattr(exact_graphs, 'stage', stage)
    monkeypatch.setattr(exact_graphs, 'compute', compute)
    monkeypatch.setattr(exact_graphs, 'mtp_stage', mtp_stage)
    monkeypatch.setattr(exact_graphs, '_kv_check', lambda segs, **kw: events.append(('status', segs[-1][2])))
    graph.warm(states)
    assert events == [('reset', 2), ('target-stage', 2), ('capture', 'verify', 2), ('status', 2),
                      ('reset', 5), ('target-stage', 5), ('eager-target-HC', 5), ('status', 5),
                      ('reset', 5), ('MTP-stage-valid-HC', 5), ('capture', 'mtp', 5), ('status', 5)]
    assert not graph.preparing


def test_observed_demand_extends_only_explicit_cold_shapes():
    request = {'policy': 'observed-demand-v1',
               'keys': [['mtp', 3, 3], ['verify', 28, 8]], 'evidence_sha256': 'd' * 64}
    graph, states = owner(slots=24, rows=120, policy=request)
    plans = graph._catalogue(states)
    assert plans == ((3, (1, 1, 1)), (8, (4, 4, 4, 4, 3, 3, 3, 3)))
    assert set(graph.requested_keys) == {('mtp', 3, 3), ('verify', 28, 8)}
    assert len(graph.available_keys) == 32
    assert graph.selected_keys == frozenset(graph.requested_keys)


@pytest.mark.parametrize('key', [['verify', 41, 8], ['verify', 28, 25], ['verify', 7, 8], ['mtp', 6, 1]])
def test_observed_demand_bounds_fail_before_warm_mutation(monkeypatch, key):
    request = {'policy': 'observed-demand-v1', 'keys': [key], 'evidence_sha256': 'e' * 64}
    graph, states = owner(slots=24, rows=120, policy=request)
    monkeypatch.setattr(graph, '_reset_warm_states', lambda *_: pytest.fail('invalid demand mutated State'))
    with pytest.raises(ValueError, match='observed graph demand'):
        graph.warm(states)
    assert not graph.preparing and graph.selected_keys is None


def test_ragged_warm_stages_matching_target_and_mtp_spans(monkeypatch):
    import torch
    request = {'policy': 'observed-demand-v1', 'keys': [['verify', 28, 8], ['mtp', 28, 8]],
               'evidence_sha256': 'f' * 64}
    graph, states = owner(slots=8, rows=40, policy=request)
    source = graph.dec.buf
    source.ple_late = None
    source.streams = torch.full((40, 2), -1, dtype=torch.float16)
    staged = []
    def stage_target(weights, buffers, windows, **kwargs):
        result, at = [], 0
        for state, tokens in windows:
            result.append((state, at, at + len(tokens)))
            at += len(tokens)
        staged.append(tuple(end - begin for _, begin, end in result))
        return result
    def capture(role, segs):
        assert (role, segs[-1][2], len(segs)) in graph.selected_keys
        if role == 'verify':
            for _, begin, end in segs:
                source.streams[begin:end].fill_(begin)
        return object()
    def stage_mtp(buffers, windows):
        result, at = [], 0
        for state, tokens, hidden in windows:
            assert hidden.shape[0] == len(tokens)
            assert torch.equal(hidden, torch.full_like(hidden, at))
            result.append((state, at, at + len(tokens)))
            at += len(tokens)
        staged.append(tuple(end - begin for _, begin, end in result))
        return result
    monkeypatch.setattr(graph, '_reset_warm_states', lambda *_: None)
    monkeypatch.setattr(exact_graphs, 'stage', stage_target)
    monkeypatch.setattr(exact_graphs, 'mtp_stage', stage_mtp)
    monkeypatch.setattr(exact_graphs, '_kv_check', lambda *a, **kw: None)
    monkeypatch.setattr(graph, 'forward', capture)
    graph.warm(states)
    assert staged == [(4, 4, 4, 4, 3, 3, 3, 3)] * 2
    assert not graph.preparing
