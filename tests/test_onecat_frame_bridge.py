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


@pytest.mark.parametrize("override", [{"depth": 4}, {"tp": 4},
                                      {"expert_tensor_parallel_size": 8}, {"adaptive_sessions": False}])
def test_engine_trial_rejects_unsupported_region_before_initialization(monkeypatch, override):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    monkeypatch.setattr(FlashNextEngine, "_initialize", lambda *a, **k: pytest.fail("initialized unsupported region"))
    arguments = dict(depth=0, tp=8, expert_tensor_parallel_size=4, adaptive_sessions=True,
                     execution_profile_manifest="unused", _owned_expert_native=True,
                     _owned_expert_gate_up=OneCatGU("tensorfold-fp32-epilogue", 117_976_064, 1 << 20))
    arguments.update(override)
    with pytest.raises(ValueError, match="1Cat GU qualification"):
        FlashNextEngine("unused", **arguments)
