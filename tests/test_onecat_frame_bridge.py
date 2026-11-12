"""Frugal frame admission/alias/raw-binding controls; no GPU substitute."""
from types import SimpleNamespace

import pytest
import torch

from test_onecat_runtime_bridge import prototype
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFPacked
from tensorfold.families.qwen4_exp.cuda.onecat_experts import OneCatGU
from tensorfold.families.qwen4_exp.cuda.onecat_frame_experts import OneCatFrameOwner, POLICY


def prepared():
    # CPU metadata/lifecycle fixtures intentionally do not call native prepare.
    return OneCatFrameOwner(OneCatGU("tensorfold-fp32-epilogue", 117_976_064, 1 << 20), "cpu")


def test_frame_forecast_replaces_48_copies_without_allocating_before_admission():
    proto, _ = prototype()
    owner = prepared()
    profile = owner.forecast(proto, 32 << 20)
    assert profile["policy"] == POLICY
    assert profile["frame_slab_bytes"] == 117_964_800
    assert profile["shared_lookup_bytes"] == 11_264
    assert profile["packer_workspace_bytes"] == 0
    assert profile["additional_device_bytes"] == 117_976_064
    assert profile["original_raw_retained"] is True
    assert owner.frame is None and owner.sinks is None and not owner.started
    assert "max_host_chunk_payload_bytes_per_rank" not in profile


def test_frame_credit_refusal_precedes_allocation():
    proto, _ = prototype()
    owner = OneCatFrameOwner(OneCatGU("onecat-half-epilogue", 117_976_063, 1 << 20), "cpu")
    with pytest.raises(MemoryError):
        owner.forecast(proto, 32 << 20)
    assert owner.frame is None and not owner.started


def test_shared_frame_views_exact_allocation_raw_binding_and_no_false_hashes(monkeypatch):
    proto, partition = prototype()
    owner = prepared()
    owner.gu = SimpleNamespace(module=SimpleNamespace(__name__="CPU-transport-only-GU"))
    owner.down = SimpleNamespace(module=SimpleNamespace(__name__="CPU-transport-only-down"))
    owner.packer = SimpleNamespace(module=SimpleNamespace(__name__="CPU-transport-only-packer"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (1000000000, 2000000000))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 0)
    fences = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: fences.append(a))
    owner.begin_load(proto, 32 << 20)
    base = owner.frame.untyped_storage().data_ptr()
    for kind, (gate, up) in owner.views.items():
        spans = []
        for tensor in (gate.codes, gate.scales, up.codes, up.scales):
            assert tensor.untyped_storage().data_ptr() == base
            assert tensor.is_contiguous() and tensor.data_ptr() % 16 == 0
            spans.append((tensor.storage_offset(), tensor.storage_offset() + tensor.numel()))
        assert all(first[1] <= second[0] for first, second in zip(spans, spans[1:]))
        assert spans[-1][1] <= owner.frame.numel()
    # Four CPU-only raw representations shared by fixture metadata, not model
    # contents. Actual source reader/content authority is a separate GPU gate.
    raw = {}
    for kind in (18, 21, 22, 23):
        width = 10 * {18: 98, 21: 110, 22: 82, 23: 136}[kind]
        raw[kind] = GGUFPacked(torch.empty((40960, width), dtype=torch.uint8), kind, 40960, 2560)
    down = GGUFPacked(torch.empty((655360, 180), dtype=torch.uint8), 42, 655360, 160, 160)
    for layer in proto.layers:
        plan = layer.moe.experts
        bank = SimpleNamespace(gate=raw[plan.gate.source.ggml_type], up=raw[plan.up.source.ggml_type],
                               down=down, partition=partition)
        owner.bind_bank(plan, bank)
        assert bank._onecat_provider.packer is owner.packer
        assert bank._onecat_provider.device_tensors() == ()
    owner.finish_load()
    assert len(fences) == 1 and owner.complete
    owner.adopt(owner.spec)
    record = owner.record()
    from tensorfold.engine.model_identity import canonical_runtime_json
    canonical_runtime_json(record, max_bytes=4 << 20)
    assert record["schema"] == "tensorfold.onecat-active-frame-owner-v1"
    assert record["frame"]["bytes"] == 117_964_800
    assert len(record["raw_bindings"]) == 48
    assert "source_planes" not in record and "source_planes_sha256" not in record
    assert record["profile"]["additional_device_bytes"] == 117_976_064
    assert all(binding["source_scope"].startswith("descriptor hashes") for binding in record["raw_bindings"])
    owner.close()
    assert owner.lease.closed and owner.lease.retiring and not owner.lease.enabled
    with pytest.raises(RuntimeError):
        owner.adopt(owner.spec)


def test_frame_never_installs_cpu_loader_sink():
    owner = prepared()
    assert owner.sink(object()) is None


def test_existing_trusted_payload_accepts_frame_selector_without_public_parser():
    from tensorfold.cli import _OwnedExpertQualification

    spec = OneCatGU("onecat-half-epilogue", 117_976_064, 1 << 20)
    capability = _OwnedExpertQualification(lambda engine: None, grouped_gate_up=spec)
    assert capability.grouped_gate_up is spec


@pytest.mark.parametrize("override", [{"depth": -1}, {"depth": 16}, {"depth": True}, {"tp": 4},
                                      {"expert_tensor_parallel_size": 8}, {"adaptive_sessions": False},
                                      {"dspark_model_dir": "unused"}, {"dflash2_model_dir": "unused"}])
def test_engine_trial_rejects_unsupported_region_before_initialization(monkeypatch, override):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    monkeypatch.setattr(FlashNextEngine, "_initialize", lambda *a, **k: pytest.fail("initialized unsupported region"))
    arguments = dict(depth=0, tp=8, expert_tensor_parallel_size=4, adaptive_sessions=True,
                     execution_profile_manifest="unused", _owned_expert_native=True,
                     _owned_expert_gate_up=OneCatGU("tensorfold-fp32-epilogue", 117_976_064, 1 << 20))
    arguments.update(override)
    with pytest.raises(ValueError, match="1Cat qualification"):
        FlashNextEngine("unused", **arguments)


@pytest.mark.parametrize('depth', [0, 4, 15])
def test_stock_mtp_and_graph_request_reach_owned_initialization_without_semantic_substitution(monkeypatch, depth):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    observed = []
    monkeypatch.setattr(FlashNextEngine, '_initialize', lambda self, model, **options: observed.append(options))
    selection = OneCatGU('tensorfold-fp32-epilogue', 117_976_064, 1 << 20)
    engine = FlashNextEngine('unused', depth=depth, graphs=True, tp=8, expert_tensor_parallel_size=4,
        adaptive_sessions=True, execution_profile_manifest='unused', _owned_expert_native=True,
        _owned_expert_gate_up=selection)
    assert engine._owned_expert_gate_up_requested is selection
    assert observed[0]['depth'] == depth and observed[0]['graphs'] is True


def test_tensorfold_gu_has_no_packing_residency_and_explicit_down_selection():
    proto, _ = prototype()
    spec = OneCatGU('tensorfold', 1, 1 << 20, down_method='onecat-q8-slots')
    owner = OneCatFrameOwner(spec, 'cpu')
    profile = owner.forecast(proto, 32 << 20)
    assert profile['additional_device_bytes'] == profile['frame_slab_bytes'] == profile['shared_lookup_bytes'] == 0
    assert profile['tables'] == []
    assert profile['down_format_policy'] == {'20': 'onecat-q8-slots', '42': 'tensorfold'}
    assert profile['mtp_provider'] == 'tensorfold-fp32-mtp'
    assert spec.identity[-2:] == ('down', 'onecat-q8-slots')
    assert owner.frame is None and owner.packer is None


def test_down_selector_rejects_implicit_or_unknown_methods():
    for method in (None, True, 'guess'):
        with pytest.raises(ValueError, match='down method'):
            OneCatGU('tensorfold', 1, 1, down_method=method)


def test_raw_iq2_source_is_explicit_and_keeps_non_iq2_frame_admission():
    proto, _ = prototype()
    spec = OneCatGU('tensorfold-fp32-epilogue', 117_976_064, 1 << 20, iq2_source='raw')
    owner = OneCatFrameOwner(spec, 'cpu')
    profile = owner.forecast(proto, 32 << 20)
    assert spec.identity[-2:] == ('iq2-source', 'raw')
    assert profile['iq2_source'] == 'raw'
    # IQ4_XS still determines the one shared frame's maximum extent.
    assert profile['additional_device_bytes'] == 117_976_064
    for method, source in [('tensorfold', 'raw'), ('onecat-half-epilogue', 'guess'), ('onecat-half-epilogue', True)]:
        with pytest.raises(ValueError, match='IQ2 source'):
            OneCatGU(method, 1, 1, iq2_source=source)


@pytest.mark.parametrize('method,source,expected', [
    ('tensorfold', 'active-frame', ['tf-gu', 'shared', 'q8-down']),
    ('tensorfold-fp32-epilogue', 'active-frame', ['pack', 'onecat-gu', 'shared', 'q8-down']),
    ('tensorfold-fp32-epilogue', 'raw', ['raw-gu', 'shared', 'q8-down'])])
def test_independent_gu_down_dispatch_does_not_execute_superseded_work(method, source, expected):
    from tensorfold.families.qwen4_exp.cuda.onecat_experts import _Bank, _Lease

    calls = []
    marker = object()
    def operation(name):
        return lambda *a, **k: calls.append(name) or marker
    raw = SimpleNamespace(data=object(), column_offset=0)
    bank = SimpleNamespace(gate=raw, up=raw,
                           down=SimpleNamespace(data=object(), ggml_type=20, column_offset=0),
                           partition=SimpleNamespace(first=0), owned_shared_result=operation('shared'))
    view = SimpleNamespace(codes=object(), scales=object(), kind=22)
    lease = _Lease()
    lease.enabled = True
    provider = _Bank(OneCatGU(method, 1, 1, down_method='onecat-q8-slots', iq2_source=source), lease,
                     SimpleNamespace(run=operation('tf-gu'), gate_up=operation('onecat-gu')),
                     SimpleNamespace(run=operation('tf-down')), view, view, object(), bank.partition, bank,
                     packer=SimpleNamespace(pack_active=operation('pack')),
                     q8_down=SimpleNamespace(slots=operation('q8-down')),
                     raw_iq2_gu=SimpleNamespace(gate_up_raw_iq2=operation('raw-gu')))
    with pytest.raises(RuntimeError, match='role scratch'):
        provider.run_slots(bank, marker, marker, local=marker, activation=marker, out=marker, grouping=marker)
    assert calls == []
    assert provider.run_slots(bank, marker, marker, local=marker, activation=marker, out=marker,
                              grouping=marker, q8=marker, status=marker) is marker
    assert calls == expected
    calls.clear()
    bank.down.ggml_type = 42
    provider.run_slots(bank, marker, marker, local=marker, activation=marker, out=marker, grouping=marker)
    assert calls == [*expected[:-1], 'tf-down']


def test_candidate_prefill_is_independent_explicit_and_bound_to_profile():
    proto, _ = prototype()
    original = OneCatGU('tensorfold-fp32-epilogue', 117_976_064, 1 << 20)
    selected = OneCatGU('tensorfold-fp32-epilogue', 117_976_064, 1 << 20, prefill_method='candidate')
    assert original.prefill_method == 'tensorfold'
    assert selected.identity == original.identity + ('prefill', 'candidate')
    owner = OneCatFrameOwner(selected, 'cpu')
    profile = owner.forecast(proto, 32 << 20)
    assert profile['prefill_method'] == 'candidate'
    assert profile['additional_device_bytes'] == 117_976_064
    raw = OneCatGU('onecat-half-epilogue', 117_976_064, 1 << 20,
                   iq2_source='raw', down_method='onecat-q8-slots', prefill_method='candidate')
    assert raw.identity[-6:] == ('down', 'onecat-q8-slots', 'iq2-source', 'raw', 'prefill', 'candidate')
    for value in (None, True, 'auto'):
        with pytest.raises(ValueError, match='prefill method'):
            OneCatGU('tensorfold', 1, 1, prefill_method=value)


def test_union_frame_is_prepared_once_and_consumed_without_persistent_state():
    from tensorfold.families.qwen4_exp.cuda.onecat_experts import _Bank, _Lease

    calls = []
    marker = object()
    def operation(name):
        return lambda *a, **k: calls.append(name) or marker
    raw = SimpleNamespace(data=object(), column_offset=0)
    bank = SimpleNamespace(gate=raw, up=raw, down=SimpleNamespace(data=object(), ggml_type=42, column_offset=0),
                           partition=SimpleNamespace(first=0), owned_shared_result=operation('shared'))
    view = SimpleNamespace(codes=object(), scales=object(), kind=18)
    lease = _Lease()
    lease.enabled = True
    spec = OneCatGU('tensorfold-fp32-epilogue', 1, 1, prefill_method='candidate')
    provider = _Bank(spec, lease, SimpleNamespace(gate_up=operation('gu')), SimpleNamespace(run=operation('down')),
                     view, view, object(), bank.partition, bank,
                     packer=SimpleNamespace(pack_union=operation('union'), pack_active=operation('active')))
    before = provider.__dict__.copy()
    assert provider.prepare_prefill_frame(bank, marker) is True
    for _ in range(3):
        provider.run_slots(bank, marker, marker, local=marker, activation=marker, out=marker,
                           grouping=marker, _frame_prepared=True)
    assert calls == ['union', 'gu', 'shared', 'down', 'gu', 'shared', 'down', 'gu', 'shared', 'down']
    assert provider.__dict__ == before  # No validity cache crosses this Arena invocation.
    calls.clear()
    provider.run_slots(bank, marker, marker, local=marker, activation=marker, out=marker, grouping=marker)
    assert calls == ['active', 'gu', 'shared', 'down']  # Decode still packs its own panel.
    lease.retiring = True
    with pytest.raises(RuntimeError, match='resident bank'):
        provider.prepare_prefill_frame(bank, marker)
