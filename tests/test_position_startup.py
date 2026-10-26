"""Actual startup/loader positional boundaries; CPU controls, no V100 claims."""
import pytest
import torch

from gguf_fixture import tiny_model
from tensorfold.families.qwen4_exp.position_profile import prepare_position_vectors, resolve_position_profile


class DeviceBoundary(Exception):
    pass


def engine_boundary(monkeypatch):
    from tensorfold.cuda import chat_template
    from tensorfold.families.qwen4_exp.cuda import weights

    events = []
    monkeypatch.setattr(chat_template, "prepare_chat_template", lambda *a, **k: None)
    def device(*a, **k):
        events.append("device")
        raise DeviceBoundary
    def retain(*a, **k):
        events.append("weights")
        pytest.fail("positional preflight retained weights")
    monkeypatch.setattr(torch.cuda, "set_device", device)
    monkeypatch.setattr(weights, "load", retain)
    return events


def test_exact_range_failure_reaches_neither_device_nor_weight_retention(tmp_path, monkeypatch):
    import gguf_fixture
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    # Preserve every actual toy tensor geometry; only the valid partial-rotary
    # policy changes. Its FP32 factor*frequency overflow is analogous to the
    # D64/factor32 witness, at this toy's D16/factor131072 dimensions.
    writer = gguf_fixture.write
    def source(path, tensors, fields):
        fields = dict(fields)
        fields.update({"qwen4exp.rope.freq_base": 3e38, "qwen4exp.rope.dimension_count": 16,
                       "qwen4exp.rope.dimension_sections": (3, 3, 2, 0)})
        return writer(path, tensors, fields)
    monkeypatch.setattr(gguf_fixture, "write", source)
    path = tiny_model(tmp_path / "range.gguf")
    events = engine_boundary(monkeypatch)
    with pytest.raises(ValueError, match="nonfinite or zero float32 inverse frequencies"):
        FlashNextEngine(path, depth=0, kv_dtype="fp16", max_len=64, yarn_factor=131072,
                        tp=8, rank=0, master="127.0.0.1", adaptive_sessions=True)
    assert events == []


def test_real_unique_engine_prepares_exact_cpu_vectors_before_device_boundary(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp import position_profile
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    path = tiny_model(tmp_path / "native.gguf")
    events = engine_boundary(monkeypatch)
    prepared = []
    actual = position_profile.prepare_position_vectors
    def prepare(*args, **kwargs):
        result = actual(*args, **kwargs)
        prepared.append(result)
        return result
    monkeypatch.setattr(position_profile, "prepare_position_vectors", prepare)
    with pytest.raises(DeviceBoundary):
        FlashNextEngine(path, depth=0, kv_dtype="fp16", max_len=64,
                        tp=8, rank=0, master="127.0.0.1", adaptive_sessions=True)
    assert events == ["device"] and len(prepared) == 1
    assert prepared[0].target_cpu.device.type == "cpu" and prepared[0].mtp_cpu is None
    assert prepared[0].profile.target.max_rows == 64


@pytest.mark.parametrize("world", [1, 8])
def test_legacy_replicated_yarn_has_no_new_binding_or_coordinate_allocations(tmp_path, monkeypatch, world):
    from tensorfold.families.qwen4_exp import position_profile
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    path = tiny_model(tmp_path / "legacy.gguf")
    events = engine_boundary(monkeypatch)
    monkeypatch.setattr(position_profile, "prepare_position_vectors",
                        lambda *a, **k: pytest.fail("legacy replica entered new bound policy"))
    with pytest.raises(DeviceBoundary):
        FlashNextEngine(path, depth=0, kv_dtype="fp16", max_len=64, yarn_factor=2,
                        tp=world, rank=0, master="127.0.0.1", adaptive_sessions=False)
    assert events == ["device"]
    with pytest.raises(ValueError, match="unique TP8"):
        FlashNextEngine(path, depth=0, kv_dtype="fp16", max_len=64,
                        tp=world, rank=0, master="127.0.0.1", adaptive_sessions=False,
                        position_profile_manifest=tmp_path / "new-profile.json")
    assert events == ["device"]


def test_direct_gguf_loader_rejects_stale_preparation_before_reader_acquisition(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import gguf_weights

    path = tiny_model(tmp_path / "source.gguf")
    profile = resolve_position_profile(path, requested_context=64)
    prepared = prepare_position_vectors(profile, mtp_active=False)
    prepared.target_cpu.add_(1)
    monkeypatch.setattr(gguf_weights, "Reader", lambda *a, **k: pytest.fail("stale positions acquired model reader"))
    with pytest.raises(ValueError, match="stale positional CPU preparation"):
        gguf_weights.load_gguf(path, "cpu", rope=profile.target.rope, position_profile=profile,
                               _position_prepared=prepared)


def test_generic_gguf_dispatch_rejects_orphan_preparation_before_reader_acquisition(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import gguf_weights, weights

    path = tiny_model(tmp_path / "source.gguf")
    profile = resolve_position_profile(path, requested_context=64)
    prepared = prepare_position_vectors(profile, mtp_active=False)
    monkeypatch.setattr(gguf_weights, "Reader", lambda *a, **k: pytest.fail("orphan preparation acquired model reader"))
    with pytest.raises(ValueError, match="lack their resolved profile"):
        weights.load(path, "cpu", mtp=False, _position_prepared=prepared)


def test_direct_gguf_loader_adopts_checked_target_vector_without_rebuilding(tmp_path, monkeypatch):
    from pathlib import Path
    from tensorfold.cuda.sm70 import gguf_ops
    from tensorfold.families.qwen4_exp.gguf import PackedTensor, decode_rows
    from tensorfold.families.qwen4_exp.cuda import gguf_weights
    from tensorfold.families.qwen4_exp.rope import RopeParameters

    path = tiny_model(tmp_path / "source.gguf")
    profile = resolve_position_profile(path, requested_context=64)
    prepared = prepare_position_vectors(profile, mtp_active=False)
    # The retained model loader is real; its GPU-only tile decoder is replaced
    # with the existing independent host decoder solely for this CPU ownership
    # control. This does not qualify CUDA decode or model execution.
    def host_decode(packed, kind, columns, *, column_offset=0, storage_columns=None, dtype=torch.float16, ids=None):
        raw = packed.numpy() if ids is None else packed.numpy()[ids.numpy()]
        descriptor = PackedTensor("control", (raw.shape[0], storage_columns or columns), kind, Path("unused"), 0, raw.nbytes)
        values = decode_rows(descriptor, raw)[:, column_offset:column_offset + columns].copy()
        return torch.from_numpy(values).to(dtype)
    monkeypatch.setattr(gguf_ops, "decode", host_decode)
    monkeypatch.setattr(RopeParameters, "inverse_frequencies",
                        lambda *a, **k: pytest.fail("loader rebuilt already checked positional vector"))
    loaded = gguf_weights.load_gguf(path, "cpu", rope=profile.target.rope, position_profile=profile,
                                    _position_prepared=prepared)
    assert loaded.inv_freq is prepared.target_cpu
    assert loaded.meta["position_profile"] is profile
    loaded.meta["ple_tables"].close(lambda: None)


@pytest.mark.parametrize("malformed", [False, True])
def test_coordinate_activation_is_shared_fresh_and_retires_owned_cpu_keys(tmp_path, monkeypatch, malformed):
    """Eight host control participants, not CUDA/NCCL execution evidence."""
    from concurrent.futures import ThreadPoolExecutor
    from datetime import timedelta
    from types import SimpleNamespace
    from tests.test_serving_profile_construction import Store
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import model_plan
    from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry
    from tensorfold.families.qwen4_exp.kv_formats import get_pair
    from tensorfold.families.qwen4_exp.position_profile import bind_position_profile
    from tensorfold.families.qwen4_exp.cuda import execution_identity
    import threading
    import secrets

    class StartupStore(Store):
        def wait(self, keys, duration):
            assert duration in (timedelta(hours=1), timedelta(seconds=60))
            with self.condition:
                assert self.condition.wait_for(lambda: all(key in self.data for key in keys), 5), keys
    path = tiny_model(tmp_path / "model.gguf")
    profile = resolve_position_profile(path, requested_context=64)
    store, participants = StartupStore(), []
    phase_state = threading.local()
    actual_phase = execution_identity._phase
    def phase(engine, channel, stage, operation, **kwargs):
        def observed():
            phase_state.stage = stage
            try:
                return operation()
            finally:
                phase_state.stage = None
        return actual_phase(engine, channel, stage, observed, **kwargs)
    monkeypatch.setattr(execution_identity, "_phase", phase)
    class PublishedMeta(dict):
        def __setitem__(self, key, value):
            if key in ("coordinate_profile_identity", "coordinate_profile_binding"):
                assert getattr(phase_state, "stage", None) == "agreement"
            super().__setitem__(key, value)
    for rank in range(8):
        model, _ = model_plan(path, (rank, 8), rope=profile.target.rope, position_profile=profile)
        prepared = prepare_position_vectors(profile, mtp_active=False)
        model.inv_freq = prepared.target_cpu
        bind_position_profile(model, profile, mtp_active=False, prepared=prepared)
        model.meta = PublishedMeta(model.meta)
        owner = object.__new__(FlashNextEngine)
        owner.rank, owner.tp, owner.depth, owner.max_len = rank, 8, 0, 65
        owner.comm = SimpleNamespace(store=store)
        owner._profile_build = owner._dflash_pairing = owner.multi = None
        owner.w = model
        owner.position_profile = profile
        owner._media_inputs_enabled = False
        owner.capacity_plan = {"position_profile": {"resolved": profile.receipt()}}
        owner.execution_provider, owner.sm70_profile = "sm70", "reference"
        owner.kv_pair, owner.method_geometry = get_pair("fp16"), MethodGeometry("none", 0)
        participants.append((owner, model))
    nonces = iter(("bad",) if malformed else ("0" * 64, "1" * 64))
    monkeypatch.setattr(secrets, "token_hex", lambda count: next(nonces))
    def bind(pair):
        owner, model = pair
        if malformed:
            with pytest.raises(ValueError, match="coordinate activation issuance is malformed"):
                owner._bind_coordinate_profile(model)
            return None
        owner._bind_coordinate_profile(model)
        return model.meta["coordinate_profile_identity"]
    with ThreadPoolExecutor(8) as workers:
        first = list(workers.map(bind, participants))
        if malformed:
            assert first == [None] * 8 and store.data == {} and store.maximum <= 32
            assert all("coordinate_profile_identity" not in model.meta for _, model in participants)
            return
        assert len(set(first)) == 1 and store.data == {}
        second = list(workers.map(bind, participants))
    assert len(set(second)) == 1 and second[0] != first[0]
    assert store.data == {} and store.maximum <= 32
    assert all(owner.coordinate_profile_binding["content_certificate"] is False for owner, _ in participants)
    assert all(owner.coordinate_profile_binding["qualification_certificate"] is False for owner, _ in participants)
