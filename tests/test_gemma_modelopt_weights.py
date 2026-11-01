"""Header/ownership/direct-calibration controls; no model/GPU performance proof."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest
import weakref

from gemma_modelopt_metadata_fixture import fixture
from tensorfold.checkpoint_quantization import effective_config
from tensorfold.families.gemma4.cuda.modelopt_weights import validate_headers, storage_bounds, tensor_contract
from tensorfold.families.gemma4.cuda.weights import Config


def test_independent_complete_pinned_header_inventory_matches_all11520_experts_and_bf16_exceptions():
    fixed = json.loads((Path(__file__).parent / "fixtures/gemma26b_modelopt_metadata.json").read_text())
    raw = effective_config(fixed["raw"], fixed["sidecar"], where="pinned26B")
    c = Config.from_dict(raw)
    experts, plain = tensor_contract(c)
    headers = {name: {"dtype": "BF16", "shape": list(shape)} for name, shape in plain.items()}
    for name, (n, k) in experts.items():
        for suffix, dtype, shape in [
            ("weight", "U8", [n, k // 2]),
            ("weight_scale", "F8_E4M3", [n, k // 16]),
            ("input_scale", "F32", []),
            ("weight_scale_2", "F32", []),
        ]:
            headers[name + "." + suffix] = {"dtype": dtype, "shape": shape}
    from tensorfold.families.gemma4.cuda.vision import vision_shapes

    for name, shape in vision_shapes(raw).items():
        headers["model." + name] = {"dtype": "BF16", "shape": list(shape)}
    headers["model.embed_vision.embedding_projection.weight"] = {
        "dtype": "BF16",
        "shape": [c.hidden, raw["vision_config"]["hidden_size"]],
    }
    semantic = {name: [v["dtype"], v["shape"]] for name, v in headers.items()}
    assert len(experts) == 11520 and len(headers) == fixed["tensor_count"] == 47033
    assert (
        hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        == fixed["semantic_header_sha256"]
    )
    assert validate_headers(raw, headers, vision=True) == c
    bound = storage_bounds(c, headers, vision=True)
    assert bound["resident"] > 0 and bound["mapped"] == 0 and bound["device_staging"] > 0


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "extra",
        "packed-native",
        "wrong-code",
        "wrong-scale",
        "bad-global",
        "awq",
        "acts-none",
        "missing-ignore",
        "asymmetric",
    ],
)
def test_invalid_or_incomplete_schema_rejected_before_payload(damage):
    raw, headers = fixture()
    key = "model.language_model.layers.0.experts.0.gate_proj"
    if damage == "missing":
        headers.pop(key + ".weight_scale")
    elif damage == "extra":
        headers[key + ".svdquant_lora_a"] = {"dtype": "BF16", "shape": [2, 64]}
    elif damage == "packed-native":
        headers["model.language_model.layers.0.self_attn.q_proj.weight"]["dtype"] = "U8"
    elif damage == "wrong-code":
        headers[key + ".weight"]["shape"] = [64, 31]
    elif damage == "wrong-scale":
        headers[key + ".weight_scale"]["dtype"] = "F32"
    elif damage == "bad-global":
        headers[key + ".input_scale"]["shape"] = [2]
    elif damage == "awq":
        raw["quantization_config"]["quant_algo"] = "NVFP4_AWQ"
    elif damage == "acts-none":
        raw["quantization_config"]["config_groups"]["group_0"]["input_activations"] = None
    elif damage == "asymmetric":
        raw["quantization_config"]["has_zero_point"] = True
    else:
        raw["quantization_config"]["ignore"] = []
    with pytest.raises(ValueError):
        validate_headers(raw, headers)


def test_modelopt_fields_remain_direct_and_scalar_rank_is_preserved():
    raw, headers = fixture()
    c = validate_headers(raw, headers)
    assert c.experts == 32
    assert all(
        item["shape"] == [] for name, item in headers.items() if name.endswith((".input_scale", ".weight_scale_2"))
    )


def test_native_finite_scratch_charges_bf16_abs_and_three_boolean_owners():
    import math

    raw, headers = fixture()
    c = validate_headers(raw, headers)
    largest = max(math.prod(item["shape"]) for item in headers.values() if item["dtype"] == "BF16")
    bound = storage_bounds(c, headers)
    # This fixture's embedding dominates pack scratch. The old2N+5 estimate
    # fails this check; no GPU allocation or cloned scan algorithm is needed.
    assert bound["device_staging"] == 5 * largest + 5


def test_activation_optin_preserves_other_modelopt_defaults_and_ct_convention(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    base = importlib.import_module("tensorfold.cuda.nvfp4.experts_a4")
    direct = importlib.import_module("tensorfold.cuda.nvfp4.modelopt_experts")
    gemma = importlib.import_module("tensorfold.families.gemma4.cuda.nvfp4_experts")
    wrapper = importlib.import_module("tensorfold.families.gemma4.cuda.modelopt_experts")
    choices = []

    def prepared(self, fields, width, dims, *, swiglu, shared_input=False):
        self.width, self.dims, self.count, self.device = width, dims, 1, torch.device("cpu")
        self.swiglu = swiglu
        choices.append(swiglu)

    monkeypatch.setattr(base.ExpertsA4, "__init__", prepared)
    original = direct.ModelOptExpertsA4((), 64, 64)
    assert choices[-1] is direct.swiglu_bf16
    with pytest.raises(ValueError):
        wrapper.ModelOptExperts(original)
    candidate = direct.ModelOptExpertsA4((), 64, 64, swiglu=gemma.geglu)
    assert wrapper.ModelOptExperts(candidate).bank is candidate
    with pytest.raises(ValueError):
        gemma.Experts(candidate)
    ct = base.ExpertsA4((), 64, 64, swiglu=gemma.geglu)
    assert gemma.Experts(ct).bank is ct
    with pytest.raises(ValueError):
        wrapper.ModelOptExperts(ct)


def fake_load(monkeypatch, tmp_path, *, native_error=None, drain_error=None, close_error=None):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    loader = importlib.import_module("tensorfold.families.gemma4.cuda.modelopt_weights")
    common = importlib.import_module("tensorfold.cuda.nvfp4.modelopt_experts")
    gemma = importlib.import_module("tensorfold.families.gemma4.cuda.nvfp4_experts")
    checkpoint = importlib.import_module("tensorfold.checkpoint_metadata")
    direct = importlib.import_module("tensorfold.cuda.direct_read")
    raw, headers = fixture()
    calls = []
    read_refs = {}
    (tmp_path / "model.safetensors").write_bytes(b"CPU transport fixture")
    scalars = [name for name in headers if name.endswith((".input_scale", ".weight_scale_2"))]
    scalarbytes = torch.tensor(
        [0.25 if name.endswith(".input_scale") else 2.0 for name in scalars], dtype=torch.float32
    ).view(torch.uint8)
    where = {name: (tmp_path / "model.safetensors", 4 * i, 4, "F32", []) for i, name in enumerate(scalars)}
    for name, item in headers.items():
        if name not in where:
            where[name] = (tmp_path / "model.safetensors", 0, 1, item["dtype"], item["shape"])

    class RawReader:
        def read(self, file, begin, size, device="cpu"):
            assert device == "cpu"
            return scalarbytes[begin : begin + size].clone()

        def close(self):
            calls.append("close")
            if close_error is not None:
                raise close_error

    class Source:
        def __init__(self, files, reader=None):
            self.reader = reader or RawReader()
            self.where = dict(where) if files else {}

        def get(self, name, device="cpu"):
            if native_error is not None and name == "model.language_model.norm.weight":
                raise native_error
            item = headers[name]
            dtype = direct.DTYPES[item["dtype"]]
            value = torch.ones(item["shape"], dtype=torch.float32).to(dtype)
            read_refs[name] = weakref.ref(value)
            return value

        def close(self):
            self.reader.close()

    def bank(reader, name, full, *, device, swiglu, field_owners):
        assert swiglu is gemma.geglu
        for expert in range(full.experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"{name}.experts.{expert}.{projection}"
                for suffix in ("weight", "weight_scale"):
                    key = prefix + "." + suffix
                    reader.get_rows(key, 0, headers[key]["shape"][0], device=device)
                assert reader.get(prefix + ".input_scale").item() == 0.25
                assert reader.get(prefix + ".weight_scale_2").item() == 2.0
        from tensorfold.cuda.nvfp4.host_layout import authority_fields_a4

        for index, (shape, dtype) in enumerate(authority_fields_a4(full.experts, full.moe_width, full.hidden)):
            field_owners[index] = torch.ones(shape, dtype=torch.uint8 if dtype == "U8" else torch.float32)
        value = common.ModelOptExpertsA4.__new__(common.ModelOptExpertsA4)
        value.width, value.dims, value.count, value.device = full.moe_width, full.hidden, full.experts, device
        value.swiglu = swiglu
        value.fields = tuple(field_owners)
        return value

    def drain(device):
        calls.append("drain")
        if drain_error is not None:
            raise drain_error

    monkeypatch.setattr(checkpoint, "read_config", lambda _: raw)
    monkeypatch.setattr(checkpoint, "read_json", lambda *args: None)
    monkeypatch.setattr(direct, "SafeTensors", Source)
    monkeypatch.setattr(common, "load_modelopt_a4", bank)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (12, 1))
    monkeypatch.setattr(torch.cuda, "synchronize", drain)
    status = loader.ModelOptLoadStatus()
    status.test_read_refs = read_refs
    return loader, status, calls


def test_success_consumes_complete_payload_and_retires_before_return(monkeypatch, tmp_path):
    loader, status, calls = fake_load(monkeypatch, tmp_path)
    result = loader.load(tmp_path, status=status)
    assert result.numerics == "modelopt_nvfp4" and result.precision == "checkpoint"
    assert result.consumption_receipt["tensor_count"] == result.consumption_receipt["consumed_count"]
    assert status.safe_to_release and status.transferred and not status.owners
    assert calls == ["drain", "close"]


def test_native_failure_preserves_primary_with_completed_cleanup(monkeypatch, tmp_path):
    primary = MemoryError("native payload read")
    loader, status, calls = fake_load(monkeypatch, tmp_path, native_error=primary)
    with pytest.raises(MemoryError) as caught:
        loader.load(tmp_path, status=status)
    assert caught.value is primary and calls == ["drain", "close"] and status.safe_to_release


def test_failed_drain_retains_owner_without_reader_close_or_cuda_retry(monkeypatch, tmp_path):
    primary = ValueError("payload")
    secondary = RuntimeError("CUDA drain")
    loader, status, calls = fake_load(monkeypatch, tmp_path, native_error=primary, drain_error=secondary)
    with pytest.raises(ValueError) as caught:
        loader.load(tmp_path, status=status)
    assert caught.value is primary and calls == ["drain"]
    assert not status.safe_to_release and status.owners and not status.transferred
    assert primary.__cause__ is not None


def test_close_failure_is_reported_after_gpu_retirement(monkeypatch, tmp_path):
    primary = OSError("reader close")
    loader, status, calls = fake_load(monkeypatch, tmp_path, close_error=primary)
    with pytest.raises(OSError) as caught:
        loader.load(tmp_path, status=status)
    assert caught.value is primary and calls == ["drain", "close"] and status.safe_to_release
    assert status.owners and not status.transferred


@pytest.mark.parametrize("value", [True, 0, None])
def test_global_asymmetry_flag_must_be_exact_false(value):
    raw, headers = fixture()
    raw["quantization_config"]["has_zero_point"] = value
    with pytest.raises(ValueError, match="untransformed"):
        validate_headers(raw, headers)


def test_late_native_registration_oom_and_failed_drain_keep_q_and_current_k(monkeypatch, tmp_path):
    primary = MemoryError("consumed registration")
    drain = RuntimeError("unknown CUDA")
    loader, status, calls = fake_load(monkeypatch, tmp_path, drain_error=drain)
    base = loader._PayloadReader
    key = "model.language_model.layers.0.self_attn.k_proj.weight"

    class FailingSet(set):
        def add(self, name):
            if name == key:
                raise primary
            return super().add(name)

    class Reader(base):
        def __init__(self, *args):
            super().__init__(*args)
            self.consumed = FailingSet()

    monkeypatch.setattr(loader, "_PayloadReader", Reader)
    with pytest.raises(MemoryError) as caught:
        loader.load(tmp_path, status=status)
    assert caught.value is primary and calls == ["drain"] and not status.safe_to_release
    for name in ["model.language_model.layers.0.self_attn.q_proj.weight", key]:
        assert status.test_read_refs[name]() is status.owners["native"][name]
    assert status.owners["pending_bank"] is not None
    assert status.owners["pending_experts"].bank is status.owners["pending_bank"]
    assert all(value is not None for value in status.owners["pending_fields"])
    assert status.owners["layers"] == [None]  # no anonymous partially published Layer


def test_post_layer_model_publication_oom_keeps_actual_layer_and_model_when_drain_fails(monkeypatch, tmp_path):
    primary = MemoryError("model metadata publication")
    drain = RuntimeError("unknown CUDA")
    loader, status, calls = fake_load(monkeypatch, tmp_path, drain_error=drain)
    actual = loader.Gemma4Weights

    class Failing(actual):
        def __setattr__(self, name, value):
            if name == "precision":
                raise primary
            return super().__setattr__(name, value)

    monkeypatch.setattr(loader, "Gemma4Weights", Failing)
    with pytest.raises(MemoryError) as caught:
        loader.load(tmp_path, status=status)
    assert caught.value is primary and calls == ["drain"]
    assert not status.safe_to_release
    model = status.owners["result"]
    layer = status.owners["layers"][0]
    assert model is not None and model.layers[0] is layer
    assert layer.q.weight is status.owners["native"]["model.language_model.layers.0.self_attn.q_proj.weight"]
    assert layer.experts.bank.fields[0] is not None


def test_gapped_multiple_scalar_windows_retain_only_compact_exact_f32_table():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    from types import SimpleNamespace

    loader = importlib.import_module("tensorfold.families.gemma4.cuda.modelopt_weights")
    experts = {f"e.{i}": (64, 64) for i in range(300)}
    names = [name + "." + suffix for name in experts for suffix in ("input_scale", "weight_scale_2")]
    where = {name: (Path("gapped"), index * 4100, 4, "F32", []) for index, name in enumerate(names)}
    headers = {name: {"dtype": "F32", "shape": []} for name in names}
    data = torch.zeros(len(names) * 4100, dtype=torch.uint8)
    expected = torch.tensor(
        [0.2500000298023224 if name.endswith(".input_scale") else 2.000000238418579 for name in names],
        dtype=torch.float32,
    )
    for index in range(len(names)):
        data[index * 4100 : index * 4100 + 4].copy_(expected[index : index + 1].view(torch.uint8))
    sizes = []

    def read(file, begin, size, device):
        assert device == "cpu"
        sizes.append(size)
        return data[begin : begin + size].clone()

    payload = loader._PayloadReader(
        SimpleNamespace(where=where, reader=SimpleNamespace(read=read)), headers, loader.ModelOptLoadStatus()
    )
    payload.prepare_scalars(experts)
    assert len(sizes) > 1 and max(sizes) <= 1 << 20
    assert payload.scalar_table.untyped_storage().nbytes() == 4 * len(names)
    assert torch.equal(payload.scalar_table.view(torch.int32), expected.view(torch.int32))
    assert all(
        value.untyped_storage().data_ptr() == payload.scalar_table.untyped_storage().data_ptr()
        for value in payload.scalars.values()
    )
    assert not payload.status.gpu_work_started and not torch.cuda.is_initialized()
