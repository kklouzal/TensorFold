"""Native target coordinator boundaries; CPU/tiny HF only, no provider algorithm."""

import copy
import hashlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from tensorfold.offline_quantization import target_calibration as target  # noqa: E402
from tensorfold.families.gemma4.dense_tensor_contract import text_tensor_contract  # noqa: E402
from test_gemma4_dense_loader import Tiny, geometry  # noqa: E402


def inputs():
    return [{"input_ids": torch.tensor([[2, 7, 11]]), "attention_mask": torch.tensor([[1, 1, 0]])}]


def freeze(batches, **limits):
    default = dict(max_batches=2, max_sequence_tokens=16, max_total_padded_tokens=16, max_cpu_bytes=1024)
    return target._freeze_batches(
        batches,
        target.CalibrationLimits(**(default | limits)),
        SimpleNamespace(vocab=128, max_position=32),
        dict(image_token_id=120, video_token_id=121, audio_token_id=122),
    )


def test_frozen_owned_cpu_batch_digest_and_source_tensor_mutation_isolation():
    batches = inputs()
    frozen, digests = freeze(batches)
    before = digests[0]
    batches[0]["input_ids"][0, 0] = 9
    assert frozen[0]["input_ids"][0, 0] == 2 and target._batch_digest(frozen[0]) == before
    assert frozen[0]["attention_mask"].dtype == torch.int64
    with pytest.raises(TypeError):
        frozen[0]["new_key"] = torch.tensor(0)
    frozen[0]["input_ids"][0, 0] = 8
    assert target._batch_digest(frozen[0]) != before  # replay check detects illicit private mutation


@pytest.mark.parametrize(
    "change", ["iterator", "media", "cache", "labels", "float-ids", "empty", "range", "mask", "position"]
)
def test_initial_text_schema_refuses_unsupported_or_invalid_batches(change):
    batches = inputs()
    if change == "iterator":
        batches = iter(batches)
    elif change == "media":
        batches[0]["input_ids"][0, 0] = 120
    elif change == "cache":
        batches[0]["past_key_values"] = object()
    elif change == "labels":
        batches[0]["labels"] = torch.zeros(1, 3, dtype=torch.int64)
    elif change == "float-ids":
        batches[0]["input_ids"] = batches[0]["input_ids"].float()
    elif change == "empty":
        batches[0]["input_ids"] = torch.empty(1, 0, dtype=torch.int64)
    elif change == "range":
        batches[0]["input_ids"][0, 0] = 128
    elif change == "mask":
        batches[0]["attention_mask"][0, 0] = 2
    else:
        batches[0]["position_ids"] = torch.tensor([[0, 1, 32]])
    with pytest.raises(ValueError):
        freeze(batches)


def test_admitted_data_caps_refuse_before_reductions_or_clones(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "min", lambda *_: pytest.fail("unadmitted content reduction"))
    monkeypatch.setattr(torch.Tensor, "clone", lambda *_: pytest.fail("unadmitted clone"))
    with pytest.raises(ValueError, match="admitted"):
        freeze(inputs(), max_cpu_bytes=1)


@pytest.mark.parametrize("budget", [0, -1, True])
def test_required_explicit_positive_dataset_limits(budget):
    with pytest.raises(ValueError):
        target.CalibrationLimits(budget, 1, 1, 1)


@pytest.fixture
def phase(monkeypatch):
    for key, name in target.BUDGET_ENV.items():
        monkeypatch.setenv(name, str(target.MINIMUM_PROFILE[key]))
    monkeypatch.setattr(
        target,
        "Path",
        lambda name: SimpleNamespace(
            read_text=lambda: str(20 * target.GIB if name.endswith("memory.max") else target.GIB)
        ),
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=128 * target.GIB))
    monkeypatch.setattr(torch.cuda.memory, "get_per_process_memory_fraction", lambda _: 20 / 128)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 3 * target.GIB)
    monkeypatch.setattr(target, "_meminfo", lambda: {"MemAvailable": 62 * target.GIB})
    calls = []
    return calls, lambda name: calls.append(name) or {"actual_guard_fixture": True}


def test_full_envelope_before_commit_but_no_second_full_envelope_afterwards(phase, monkeypatch):
    calls, admit = phase
    assert target._phase("before-native-load", admit)["host_available"] == 62 * target.GIB
    monkeypatch.setattr(target, "_meminfo", lambda: {"MemAvailable": 21 * target.GIB})
    assert target._phase("calibration-batch", admit)["host_available"] == 21 * target.GIB
    with pytest.raises(MemoryError):
        target._phase("before-native-load", admit)
    assert calls == ["before-native-load", "calibration-batch", "before-native-load"]


def test_no_resource_profile_lowering_or_cap_reset(phase, monkeypatch):
    _, admit = phase
    monkeypatch.setenv("TF_DIAG_CUDA_BUDGET_BYTES", str(16 * target.GIB))
    with pytest.raises(ValueError, match="cannot lower"):
        target._phase("before-native-load", admit)


def test_no_anonymous_admission_bool(phase):
    with pytest.raises(ValueError, match="receipt"):
        target._phase("before-native-load", lambda _: True)


def test_cuda_cap_must_already_be_installed(phase, monkeypatch):
    _, admit = phase
    monkeypatch.setattr(torch.cuda.memory, "get_per_process_memory_fraction", lambda _: 1)
    with pytest.raises(MemoryError, match="quota"):
        target._phase("before-native-load", admit)


def test_native_tiny_hf_inventory_tie_buffers_and_expected_unused_geometry():
    model = Tiny().to(torch.bfloat16)
    cfg = geometry()
    source = {
        name: {"dtype": "BF16", "shape": list(value.shape)}
        for name, value in model.state_dict().items()
        if name != "lm_head.weight"
    }
    contract, _, unused = text_tensor_contract(cfg)
    source.update({name: {"dtype": "BF16", "shape": list(contract[name])} for name in unused})
    result = target._native_inventory(model, cfg, source, "cpu")
    assert result["tied_head_identity"] and len(result["expected_unused_tensors"]) == 6
    assert result["nonpersistent_buffers"]
    bad = copy.deepcopy(source)
    del bad[next(iter(unused))]
    with pytest.raises(ValueError, match="inventory"):
        target._native_inventory(model, cfg, bad, "cpu")
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    with pytest.raises(ValueError, match="share one"):
        target._native_inventory(model, cfg, source, "cpu")


def test_owner_failed_drain_keeps_model_and_is_terminal(monkeypatch):
    owner = target.CalibratedTarget()
    model = object()
    owner.model = model
    owner.device_touched = True
    failure = RuntimeError("drain failed")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: (_ for _ in ()).throw(failure))
    with pytest.raises(RuntimeError) as caught:
        owner.close()
    assert caught.value is failure and owner.model is model
    with pytest.raises(RuntimeError, match="terminal"):
        owner.close()


def test_source_file_hash_and_observed_identity_check(tmp_path, monkeypatch):
    import json
    import struct

    body = b"\0\0"
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + body)
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(target.E2BConfig, "from_dict", lambda _: SimpleNamespace(profile="E4B"))
    with pytest.raises(ValueError, match="read-only"):
        target._source_snapshot(tmp_path)
    monkeypatch.setattr(target.os, "statvfs", lambda _: SimpleNamespace(f_flag=target.os.ST_RDONLY))
    _, _, files = target._source_snapshot(tmp_path)
    assert (
        files["model.safetensors"]["sha256"]
        == hashlib.sha256((tmp_path / "model.safetensors").read_bytes()).hexdigest()
    )
    target._recheck_source(tmp_path, files)
    (tmp_path / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="identity"):
        target._recheck_source(tmp_path, files)


def test_coordinator_replays_native_flags_and_returns_explicit_owner_without_export(monkeypatch, tmp_path):
    """Fake device/provider only; actual freeze/replay/control flow is exercised."""
    from contextlib import contextmanager
    import sys
    from types import ModuleType

    events = []
    cfg = SimpleNamespace(vocab=128, max_position=32)
    raw = dict(image_token_id=120, video_token_id=121, audio_token_id=122)
    monkeypatch.setattr(target, "read_config", lambda _: raw)
    monkeypatch.setattr(target, "_phase", lambda name, admit: events.append(name) or {"phase": name})
    monkeypatch.setattr(target, "_runtime_proof", lambda *_: {"provider": "synthetic-test-only"})
    monkeypatch.setattr(target, "_source_snapshot", lambda _: (cfg, {}, {}))
    monkeypatch.setattr(target, "_recheck_source", lambda *_: events.append("source-rechecked"))
    monkeypatch.setattr(target, "plan_nvfp4", lambda _: {"tensors": [{"action": "quantize", "shape": [64, 64]}]})
    monkeypatch.setattr(target, "scoped_local_hessian_config", lambda *_: {"method": "synthetic-config-only"})
    monkeypatch.setattr(
        target,
        "_native_inventory",
        lambda model, cfg, inventory, device: (
            events.append(("native", device)) or {"synthetic": True, "cuda_storage_proof": None}
        ),
    )
    original_to = torch.Tensor.to

    def to(value, *args, **kwargs):
        if args and str(args[0]) == "cuda:0":
            return value
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: events.append("drain"))

    class Native(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(128, 64)
            self.lm_head = torch.nn.Linear(64, 128, bias=False)
            self.lm_head.weight = self.embedding.weight

        def to(self, device):
            events.append(("to", str(device)))
            return self

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, **kwargs):
            assert kwargs["use_cache"] is False and kwargs["logits_to_keep"] == 1
            assert kwargs["output_hidden_states"] is kwargs["output_attentions"] is False
            assert kwargs["return_dict"] is True
            events.append(("forward", kwargs["input_ids"].tolist()))

    monkeypatch.setattr(target, "_load_native", lambda _: events.append("load") or Native())

    class Coverage:
        def __init__(self, plan, **kwargs):
            assert kwargs == dict(max_input_elements=192, max_batches=1)

        @contextmanager
        def observe_pass(self, model):
            events.append("pass")
            yield

        @contextmanager
        def batch(self, digest):
            assert len(digest) == 64
            yield

        def verify(self, model):
            events.append("coverage")
            return {"synthetic-observer-only": True}

    monkeypatch.setattr(target, "CalibrationCoverage", Coverage)
    package = ModuleType("modelopt")
    package.__path__ = []
    torch_package = ModuleType("modelopt.torch")
    torch_package.__path__ = []
    quant = ModuleType("modelopt.torch.quantization")
    quant.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG = {}

    def quantize(model, config, forward_loop):
        forward_loop(model)
        forward_loop(model)
        return model

    quant.quantize = quantize
    for name, module in [
        ("modelopt", package),
        ("modelopt.torch", torch_package),
        ("modelopt.torch.quantization", quant),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    owner = target.calibrate_target(
        tmp_path, inputs(), limits=target.CalibrationLimits(1, 16, 16, 1024), admit_phase=lambda _: {}
    )
    assert events.index("before-native-load") < events.index("load")
    assert events.count("pass") == 2
    assert len([item for item in events if isinstance(item, tuple) and item[0] == "forward"]) == 2
    assert owner.source_root == tmp_path.resolve()
    assert owner._runtime_namespace is not None
    assert owner.receipt["source_directory"] == str(tmp_path.resolve())
    assert owner.receipt["output_published"] is False and "text-only" in owner.receipt["data_scope"]
    assert "NOT consumed-byte" in owner.receipt["source_identity_scope"]
    owner.close()
    assert owner.model is None and events[-1] == "drain"
    assert owner.source_root is None and owner._runtime_namespace is None


@pytest.mark.parametrize("added", ["model-extra.safetensors", "generation_config.json"])
def test_source_namespace_additions_refused_before_native_load(tmp_path, monkeypatch, added):
    import json
    import struct

    body = b"\0\0"
    head = json.dumps({"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(head)) + head + body)
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(target.E2BConfig, "from_dict", lambda _: SimpleNamespace(profile="E4B"))
    monkeypatch.setattr(target.os, "statvfs", lambda _: SimpleNamespace(f_flag=target.os.ST_RDONLY))
    _, _, files = target._source_snapshot(tmp_path)
    (tmp_path / added).write_bytes(b"new namespace entry")
    with pytest.raises(ValueError, match="namespace"):
        target._recheck_source(tmp_path, files)


def test_native_load_info_refusal_clears_owned_model_alias(monkeypatch, tmp_path):
    import gc
    import weakref
    import transformers

    refs = []

    def load(*args, **kwargs):
        assert kwargs["local_files_only"] and not kwargs["trust_remote_code"]
        assert "device_map" not in kwargs and kwargs["use_safetensors"]
        model = torch.nn.Linear(2, 2)
        refs.append(weakref.ref(model))
        return model, {"unexpected_keys": ["not consumed"]}

    monkeypatch.setattr(transformers.Gemma4ForConditionalGeneration, "from_pretrained", load)
    with pytest.raises(ValueError, match="required tensors") as caught:
        target._load_native(tmp_path)
    assert caught.value.__traceback__ is not None
    gc.collect()
    assert refs[0]() is None


def test_contiguous_single_clone_accounts_logical_frozen_and_caller_coexistence():
    batches = [{"input_ids": torch.arange(6, dtype=torch.int64).reshape(1, 6)[:, ::2]}]
    assert not batches[0]["input_ids"].is_contiguous()
    frozen, _ = freeze(batches)
    assert frozen[0]["input_ids"].is_contiguous()
    assert frozen[0]["input_ids"].untyped_storage().nbytes() == 3 * 8
    assert batches[0]["input_ids"].untyped_storage().nbytes() == 6 * 8


def test_clean_physical_child_and_active_combined_runtime_namespaces(tmp_path, monkeypatch):
    """Actual child interpreter; role contract fixture, no provider/GPU claim."""
    import json
    import sys
    import types
    from tensorfold.families.gemma4.cuda import e2b_runtime

    root = tmp_path / "role"
    tools = root / "tools"
    tools.mkdir(parents=True)
    prefix = root / "prefix"
    prefix.mkdir()
    package = prefix / "modelopt"
    package.mkdir()
    (package / "__init__.py").write_text("# fixture")
    baseline = {"baseline-fixture": {"version": "1", "requires": []}}
    addition = {"version": "fixture-version", "requires": [], "sha256": "a" * 64}
    manifest = {
        "baseline": {"packages": baseline},
        "additions": {"nvidia-modelopt": addition},
        "source": {"source_commit": target.PRODUCER_COMMIT, "wheel_sha256": "a" * 64},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "seal.json").write_text('{"fixture":true}')
    (tools / "role.py").write_text(
        '''import json,hashlib,sys
from pathlib import Path
MODEL_OPT_COMMIT="'''
        + target.PRODUCER_COMMIT
        + """"
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def seal_inputs(prefix,path):return {"fixture":True}
def installed(paths=None):
 root=Path(__file__).resolve().parent.parent
 m=read(root/'manifest.json'); result=dict(m['baseline']['packages'])
 if str(root/'prefix') in (sys.path if paths is None else paths):
  result.update({name:{'version':row['version'],'requires':row['requires']} for name,row in m['additions'].items()})
 return result
def build_import_contract(path):return {}
def check_import_context(path,contract):
 if sys.path[:len(path)]!=path:raise ValueError("active path changed")
def audit(prefix,manifest):
 if installed()!=manifest['baseline']['packages']:raise ValueError('physical audit saw active producer union')
def provider_proof():
 if str(Path(__file__).resolve().parent.parent/'prefix') not in sys.path:raise ValueError('same-process proof missing active prefix')
"""
    )
    monkeypatch.setattr(target, "PRODUCER_ROOT", root)
    monkeypatch.setattr(e2b_runtime, "require_runtime", lambda *_: None)
    original_version = target.md.version
    monkeypatch.setattr(
        target.md, "version", lambda name: "fixture-version" if name == "nvidia-modelopt" else original_version(name)
    )
    fake = types.ModuleType("modelopt")
    fake.__file__ = str(package / "__init__.py")
    monkeypatch.setitem(sys.modules, "modelopt", fake)
    before = sys.path.copy()
    monkeypatch.syspath_prepend(str(prefix))
    proof = target._runtime_proof()
    assert str(prefix) in sys.path and sys.path[1:] == before
    assert (
        proof["physical_audit"]["manifest_sha256"] == hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    )
    assert proof["version"] == "fixture-version" and proof["active_namespace_sha256"]


@pytest.mark.parametrize("alias", [False, True])
def test_native_postcuda_storage_floor_is_unique_not_unrelated_allocator_bytes(monkeypatch, alias):
    model = Tiny().to(torch.bfloat16)
    cfg = geometry()
    inventory = {
        name: {"dtype": "BF16", "shape": list(value.shape)}
        for name, value in model.state_dict().items()
        if name != "lm_head.weight"
    }
    contract, _, unused = text_tensor_contract(cfg)
    inventory.update({name: {"dtype": "BF16", "shape": list(contract[name])} for name in unused})
    if alias:
        model.model.language_model.layers[1].input_layernorm.weight = model.model.language_model.layers[
            0
        ].input_layernorm.weight
    monkeypatch.setattr(torch.Tensor, "device", property(lambda _: torch.device("cuda:0")))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 2**30)
    if alias:
        with pytest.raises(ValueError, match="unique live"):
            target._native_inventory(model, cfg, inventory, "cuda:0")
    else:
        proof = target._native_inventory(model, cfg, inventory, "cuda:0")["cuda_storage_proof"]
        assert proof["unique_live_storage_bytes"] == proof["logical_persistent_bytes"]
        assert proof["unique_live_storage_bytes"] < proof["cuda_allocated_bytes"]
