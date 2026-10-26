"""Prepared profile files use actual metadata/GGUF/tokenizer boundary readers."""

import hashlib
import json

import pytest

from cuda_http_fixtures import model_directory
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.serving_profiles import prepare, read_manifest, relative_file
from gguf_fixture import write
from tensorfold.families.qwen4_exp.gguf import metadata_config


def profile_files(tmp_path, *, context=1024, theta=10000000.0, kv_heads=2):
    model = tmp_path / "target"
    model.mkdir()
    model_directory(model)
    # Real positional preparation reads authoritative GGUF geometry as well as
    # the sidecar. This small inventory exercises those readers without claiming
    # a complete runnable target payload or qualification.
    metadata = {"general.architecture": "qwen4exp", **{"qwen4exp." + key: value for key, value in {
        "embedding_length": 6144, "block_count": 2, "full_attention_interval": 2,
        "attention.key_length": 256, "rope.dimension_count": 64, "rope.freq_base": theta,
        "rope.dimension_sections": (11, 11, 10, 0), "context_length": context,
        "attention.head_count": 24, "attention.head_count_kv": kv_heads, "attention.layer_norm_rms_epsilon": 1e-6,
        "ssm.group_count": 8, "ssm.time_step_rank": 8, "ssm.state_size": 128, "ssm.inner_size": 1024,
        "ssm.conv_kernel": 4, "expert_count": 2, "expert_used_count": 1,
        "expert_feed_forward_length": 256, "expert_shared_feed_forward_length": 256,
        "hyper_connection.count": 4, "hyper_connection.low_rank": 32,
        "attention.indexer.head_count": 8, "attention.indexer.key_length": 128, "attention.indexer.top_k": 8,
        "ple.heads_per_ngram": 1, "ple.ngram_size": 3, "embedding_length_per_layer_input": 32,
        "ple.conv_kernel": 4,
    }.items()}}
    source = write(model / "model.gguf", [("token_embd.weight", (1,), 0, [0, 0, 0, 0])], metadata)
    config = metadata_config(source)
    config["gguf_file"] = source.name
    (model / "config.json").write_text(json.dumps(config))
    entries = [{"path": str(path.relative_to(tmp_path)), "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(model.iterdir())]
    raw = {"schema": "tensorfold.serving-profile", "version": 1, "id": "A", "model_dir": "target",
           "options": {"max_len": context, "kv_dtype": "int8", "depth": 0},
           "api": {"served": "model-A", "aliases": ["alias-A"]}, "artifacts": entries}
    path = tmp_path / "A.json"
    path.write_text(json.dumps(raw))
    return tmp_path, path, raw


@pytest.fixture
def profile(tmp_path):
    return profile_files(tmp_path)


def load(root, name="A.json", *, rank=0, waiting_input_bytes=4096, **kwargs):
    from tensorfold.families.qwen4_exp.gguf import GGUFCheckpoint

    actual = GGUFCheckpoint.open(root / "target/model.gguf").metadata
    fixed = {"tp": 8, "rank": rank, "adaptive_sessions": True, "indexer_history": "gpu"}
    if actual["qwen4exp.attention.head_count_kv"] == 2 and waiting_input_bytes is not None:
        fixed["max_waiting_input_bytes"] = waiting_input_bytes
    return prepare(root, name, initializer=FlashNextEngine._initialize, fixed_options=fixed, **kwargs)


def test_exact_manifest_identity_is_shared_by_all_rank_coordinates(profile):
    root, path, raw = profile
    leader = load(root)
    peer = load(root, rank=7, verify_content=False, expected=leader.wire())
    assert peer.identity == leader.identity
    assert peer.options["rank"] == 7 and leader.options["rank"] == 0
    assert leader.model_dir == root / "target"
    assert leader.options["max_len"] == 1024
    assert leader.record()["qualification"] == "prepared-identity-checked"
    assert leader.manifest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(TypeError):
        leader.options["max_len"] = 8


@pytest.mark.parametrize("mutation", ["missing-weight", "duplicate", "digest", "size", "context", "fixed-tp",
                                      "bool-depth", "unknown", "api-type", "api-nan", "api-seed"])
def test_bad_artifact_geometry_and_api_input_are_refused_before_engine_load(profile, mutation):
    root, path, raw = profile
    if mutation == "missing-weight":
        raw["artifacts"] = [item for item in raw["artifacts"] if not item["path"].endswith(".gguf")]
    elif mutation == "duplicate":
        raw["artifacts"].append(raw["artifacts"][0])
    elif mutation == "digest":
        raw["artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "size":
        raw["artifacts"][0]["size"] += 1
    elif mutation == "context":
        raw["options"]["max_len"] = 1025
    elif mutation == "fixed-tp":
        raw["options"]["tp"] = 4
    elif mutation == "bool-depth":
        raw["options"]["depth"] = True
    elif mutation == "unknown":
        raw["options"]["imaginary"] = 7
    elif mutation == "api-type":
        raw["api"]["default_thinking"] = 1
    elif mutation == "api-seed":
        raw["api"]["sampling"] = {"seed": 123}  # Seeds are request-scoped, not an App sampling default.
    else:
        raw["api"]["sampling"] = {"temperature": float("nan")}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load(root)


def test_prepared_file_replacement_is_detected_before_load_and_publication(profile):
    root, path, raw = profile
    prepared = load(root)
    target = prepared.model_dir / "model.gguf"
    original = target.read_bytes()
    target.unlink()
    target.write_bytes(original)
    with pytest.raises(ValueError, match="changed"):
        prepared.check_sources()
    with pytest.raises(ValueError, match="differ"):
        load(root, rank=1, verify_content=False, expected=prepared.wire())


@pytest.mark.parametrize("name", ["../A.json", "/A.json", "x//A.json", "./A.json", "x\\A.json", "A\x00.json"])
def test_administrator_selector_cannot_escape_profile_root(profile, name):
    with pytest.raises(ValueError):
        relative_file(profile[0], name)


def test_selector_symlink_escape_duplicate_json_and_nonregular_input(profile, tmp_path):
    root, path, raw = profile
    outside = tmp_path.parent / (tmp_path.name + "-outside.json")
    outside.write_text("{}")
    try:
        (root / "escape.json").symlink_to(outside)
        with pytest.raises(ValueError, match="escapes"):
            relative_file(root, "escape.json")
    finally:
        outside.unlink()
    path.write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError):
        read_manifest(path)
    with pytest.raises(ValueError):
        read_manifest(root)


def test_manifest_size_bound_precedes_json_allocation(profile):
    root, path, raw = profile
    path.write_bytes(b" " * ((1 << 20) + 1))
    with pytest.raises(ValueError, match="one-MiB"):
        load(root)


def test_boolean_schema_version_is_not_integer_version_one(profile):
    root, path, raw = profile
    raw["version"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="schema"):
        load(root)


def test_new_optional_load_input_cannot_bypass_the_prepared_inventory(profile):
    root, path, raw = profile
    prepared = load(root)
    (prepared.model_dir / "generation_config.json").write_text('{"eos_token_id":1}')
    with pytest.raises(ValueError, match="new selected"):
        prepared.check_sources()


def test_preparation_tool_publishes_the_real_verified_manifest_atomically(profile):
    from tools.prepare_serving_profile import create

    root, path, raw = profile
    destination = root / "generated.json"
    result = create(destination, identifier="generated", model=root / "target",
                    options=raw["options"], api={})
    assert result["manifest"] == "generated.json"
    actual = load(root, "generated.json")
    assert actual.identifier == "generated"
    assert not list(root.glob(".serving-*.json"))


def test_missing_startup_profile_root_refuses_before_any_engine_initialization(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(FlashNextEngine, "_initialize", lambda *args, **kwargs: called.append("unexpected load"))
    with pytest.raises(ValueError, match="directory"):
        FlashNextEngine(tmp_path, adaptive_sessions=True, serving_profile_root=tmp_path / "missing")
    assert not called
