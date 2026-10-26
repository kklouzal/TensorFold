"""Real positional authority at prepared-profile boundaries; no GPU qualification."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.serving_profiles import PATH_OPTIONS, prepare
from tensorfold.families.qwen4_exp.kv_formats import get_pair
from tensorfold.families.qwen4_exp.position_profile import resolve_position_profile
from test_serving_profile_manifest import load, profile_files
from tools.prepare_serving_profile import create

UNIQUE_DEPLOYMENT = MappingProxyType({"tp": 8, "rank": 0, "adaptive_sessions": True, "indexer_history": "gpu",
                                     "max_waiting_input_bytes": 4096})


def save(root, path, raw, *, add=()):
    paths = [root / item["path"] for item in raw["artifacts"]]
    paths.extend(file for file in add if file not in paths)
    raw["artifacts"] = [{"path": str(file.relative_to(root)), "size": file.stat().st_size,
                         "sha256": hashlib.sha256(file.read_bytes()).hexdigest()} for file in paths]
    path.write_text(json.dumps(raw))


def companion(root, *, factor=None):
    directory = root / "mtp"
    directory.mkdir()
    text = deepcopy(json.loads((root / "target/config.json").read_text())["text_config"])
    if factor is not None:
        text["rope_parameters"].update(rope_type="yarn", factor=factor,
                                        original_max_position_embeddings=text["max_position_embeddings"])
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": text}))
    (directory / "model.safetensors").write_bytes(b"payload import is a separate provider contract")
    return directory


def positional_manifest(root, *, rows, factor=1, mtp=None, epoch=0, limits=None):
    configs = {} if mtp is None else {"mtp": mtp}
    native = resolve_position_profile(root / "target", requested_context=262144, companion_configs=configs)

    def role(selected):
        rope = {"rope_type": "default" if factor == 1 else "yarn", "rope_theta": 10000000,
                "partial_rotary_factor": .25, "mrope_section": [11, 11, 10], "mrope_interleaved": True}
        if factor != 1:
            rope.update(factor=factor, original_max_position_embeddings=262144)
        return {"source_config_sha256": selected.source_config_sha256, "rope": rope,
                "coordinate_limits": [int(262144 * factor)] * 3 if limits is None else limits}

    body = {"schema_version": 1, "family": "qwen4_exp", "epoch": epoch, "requested_context": rows,
            "qualification": "unqualified-candidate", "target": role(native.target),
            "mtp": None if mtp is None else role(native.mtp), "dflash2": None}
    directory = root / "positions"
    directory.mkdir(exist_ok=True)
    path = directory / "selected.json"
    path.write_text(json.dumps(body))
    return path, body


def test_native_receipt_reports_exact_roles_codec_and_remains_unqualified(tmp_path):
    root, _, _ = profile_files(tmp_path)
    profile = load(root)
    position = profile.record()["position_profile"]
    assert position["resolved"]["requested_context"] == 1024
    assert position["resolved"]["target_attention"]["rope"]["original_max_position_embeddings"] == 1024
    assert position["binding"]["method"] == "none" and position["binding"]["lookahead_rows"] == 0
    assert position["binding"]["kv_identity"] == get_pair("int8").identity
    numerical = position["numerical_preparation"]
    assert numerical["device"] == "cpu" and numerical["finite_positive_vectors"] is True
    assert numerical["position_identity"] == position["binding"]["position_identity"]
    assert numerical["target_vector_bytes"] == 32 * 4 and numerical["mtp_extra_vector_bytes"] == 0
    assert position["resolved"]["qualification"] == position["binding"]["qualification"] == "unqualified-candidate"
    position["resolved"]["target_attention"]["rope"]["factor"] = 99
    assert profile.record()["position_profile"]["resolved"]["target_attention"]["rope"]["factor"] == 1


@pytest.mark.parametrize("factor,rows", [(2, 524288), (4, 1048576), (1.5, 393216), (2, 400001)])
def test_prepared_yarn_candidate_uses_resolved_limit_not_native_config_ceiling(tmp_path, factor, rows):
    root, path, raw = profile_files(tmp_path, context=262144)
    raw["options"].update(max_len=rows, yarn_factor=factor)
    save(root, path, raw)
    profile = load(root)
    peer = load(root, rank=7, verify_content=False, expected=profile.wire())
    resolved = profile.record()["position_profile"]["resolved"]
    assert profile.options["max_len"] == rows and peer.identity == profile.identity
    assert resolved["target_attention"]["max_rows"] == rows
    assert resolved["target_attention"]["rope"]["original_max_position_embeddings"] == 262144
    assert resolved["mtp_attention"]["rope"]["rope_type"] == "default"
    assert resolved["qualification"] == "unqualified-candidate"


def test_transform_changes_identity_even_when_context_and_artifacts_match(tmp_path):
    root, path, raw = profile_files(tmp_path, context=262144)
    native = load(root)
    raw["options"]["yarn_factor"] = 2
    save(root, path, raw)
    scaled = load(root)
    assert native.identity != scaled.identity and native.options["max_len"] == scaled.options["max_len"]
    assert native.artifacts == scaled.artifacts


def test_target_extension_does_not_silently_extend_mtp_or_substitute_none(tmp_path):
    root, path, raw = profile_files(tmp_path, context=262144)
    raw["options"].update(max_len=524288, yarn_factor=2, depth=2)
    save(root, path, raw)
    with pytest.raises(ValueError, match="explicit MTP"):
        load(root)
    draft = companion(root, factor=2)
    raw["options"]["mtp_model_dir"] = "mtp"
    save(root, path, raw, add=(draft / "config.json", draft / "model.safetensors"))
    actual = load(root)
    position = actual.record()["position_profile"]
    assert position["binding"]["method"] == "mtp" and position["binding"]["lookahead_rows"] == 2
    assert position["resolved"]["mtp_attention"]["origin"] == "actual-companion-with-explicit-source-native-defaults"
    assert position["resolved"]["mtp_attention"]["rope"]["factor"] == 2


def test_actual_initializer_accepts_relative_manifest_and_binds_epoch_and_full_closure(tmp_path):
    root, path, raw = profile_files(tmp_path, context=262144)
    draft = companion(root)
    positional, _ = positional_manifest(root, rows=524288, factor=2, mtp=draft, epoch=7)
    assert "position_profile_manifest" in PATH_OPTIONS
    raw["options"].update(max_len=524288, depth=2, mtp_model_dir="mtp",
                          position_profile_manifest="positions/selected.json")
    save(root, path, raw, add=(draft / "config.json", draft / "model.safetensors", positional))
    actual = load(root)
    assert actual.options["position_profile_manifest"] == positional
    position = actual.record()["position_profile"]
    assert position["binding"]["epoch"] == position["resolved"]["epoch"] == 7
    assert set(position["resolved"]["source_files"]) <= {item["path"] for item in actual.artifacts}
    assert {str(positional), str(draft / "config.json")} <= set(position["resolved"]["source_files"])
    peer = load(root, rank=6, verify_content=False, expected=actual.wire())
    assert peer.record()["position_profile"] == position


def test_missing_position_manifest_inventory_refuses_before_publication(tmp_path):
    root, path, raw = profile_files(tmp_path, context=262144)
    positional, _ = positional_manifest(root, rows=524288, factor=2)
    raw["options"].update(max_len=524288, position_profile_manifest="positions/selected.json")
    save(root, path, raw)
    with pytest.raises(ValueError, match="cover every"):
        load(root)
    save(root, path, raw, add=(positional,))
    load(root)


def test_changed_position_input_refuses_before_load_and_new_epoch_has_new_identity(tmp_path):
    root, path, raw = profile_files(tmp_path, context=262144)
    positional, body = positional_manifest(root, rows=262144, epoch=0)
    raw["options"]["position_profile_manifest"] = "positions/selected.json"
    save(root, path, raw, add=(positional,))
    old = load(root)
    body["epoch"] = 1
    positional.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="positional inputs"):
        old.check_sources()
    save(root, path, raw)
    new = load(root)
    assert old.identity != new.identity
    assert new.record()["position_profile"]["resolved"]["epoch"] == 1
    with pytest.raises(ValueError, match="differ"):
        load(root, rank=1, verify_content=False, expected=old.wire())


def test_producer_uses_real_manifest_resolver_and_publishes_complete_relative_inputs(tmp_path):
    root, _, raw = profile_files(tmp_path, context=262144)
    positional, _ = positional_manifest(root, rows=524288, factor=2)
    options = {**raw["options"], "max_len": 524288, "position_profile_manifest": "positions/selected.json"}
    output = root / "B.json"
    result = create(output, identifier="B", model=root / "target", options=options, api={}, fixed_options=UNIQUE_DEPLOYMENT)
    actual = prepare(root, output.name, initializer=FlashNextEngine._initialize, fixed_options=UNIQUE_DEPLOYMENT)
    assert str(positional) in {item["path"] for item in actual.artifacts}
    assert result["position_profile"] == actual.record()["position_profile"]
    assert result["position_profile"]["resolved"]["qualification"] == "unqualified-candidate"
    assert not list(root.glob(".serving-*.json"))


def test_dflash_separate_native_scalar_limit_refuses_target_only_extension(tmp_path):
    root, path, raw = profile_files(tmp_path, context=262144)
    draft = root / "draft"
    draft.mkdir()
    config = Path(__file__).parent / "fixtures/dflash2/config.json"
    (draft / "config.json").write_bytes(config.read_bytes())
    (draft / "model.safetensors").write_bytes(b"payload import is a separate provider contract")
    raw["options"].update(max_len=524288, yarn_factor=2, depth=2, dflash2_model_dir="draft")
    save(root, path, raw, add=(draft / "config.json", draft / "model.safetensors"))
    with pytest.raises(ValueError, match="supported absolute range"):
        load(root)


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_dflash_receipt_binds_fixed_four_row_compute_at_every_proposal_limit(tmp_path, depth):
    root, path, raw = profile_files(tmp_path)
    draft = root / "draft"
    draft.mkdir()
    (draft / "config.json").write_bytes((Path(__file__).parent / "fixtures/dflash2/config.json").read_bytes())
    (draft / "model.safetensors").write_bytes(b"payload import is a separate provider contract")
    raw["options"].update(depth=depth, dflash2_model_dir="draft")
    save(root, path, raw, add=(draft / "config.json", draft / "model.safetensors"))
    binding = load(root).record()["position_profile"]["binding"]
    assert binding["method"] == "dflash2" and binding["lookahead_rows"] == 3


def test_native_dspark_profile_preserves_existing_source_policy_without_new_role_claim(tmp_path):
    root, path, raw = profile_files(tmp_path)
    draft = root / "dspark"
    draft.mkdir()
    (draft / "config.json").write_text('{"architectures":["DSparkDraftModel"]}')
    (draft / "model.safetensors").write_bytes(b"payload import is a separate provider contract")
    raw["options"].update(depth=2, dspark_model_dir="dspark")
    save(root, path, raw, add=(draft / "config.json", draft / "model.safetensors"))
    position = load(root, waiting_input_bytes=None).record()["position_profile"]
    assert position["resolved"] is None
    assert position["binding"]["method"] == "dspark"
    assert position["binding"]["support"] == "legacy-source-policy"
    raw["options"]["max_len"] = 1025
    save(root, path, raw)
    with pytest.raises(ValueError, match="legacy.*position contract"):
        load(root, waiting_input_bytes=None)


def test_failed_extended_companion_preflight_keeps_previous_published_manifest(tmp_path):
    root, _, raw = profile_files(tmp_path, context=262144)
    output = root / "B.json"
    create(output, identifier="native", model=root / "target", options=raw["options"], api={}, fixed_options=UNIQUE_DEPLOYMENT)
    original = output.read_bytes()
    options = {**raw["options"], "depth": 2, "yarn_factor": 2, "max_len": 524288}
    with pytest.raises(ValueError, match="explicit MTP"):
        create(output, identifier="unimplemented-mtp-extension", model=root / "target", options=options, api={}, fixed_options=UNIQUE_DEPLOYMENT)
    assert output.read_bytes() == original and not list(root.glob(".serving-*.json"))



def test_legacy_mtp_missing_position_fields_uses_source_native_not_target_override(tmp_path):
    root, path, raw = profile_files(tmp_path, context=262144)
    draft = companion(root)
    source = draft / "config.json"
    config = json.loads(source.read_text())
    config["text_config"].pop("max_position_embeddings")
    config["text_config"].pop("rope_parameters")
    source.write_text(json.dumps(config))
    raw["options"].update(depth=2, mtp_model_dir="mtp")
    save(root, path, raw, add=(source, draft / "model.safetensors"))
    positions = load(root).record()["position_profile"]
    mtp = positions["resolved"]["mtp_attention"]
    expected = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert mtp["source_config_sha256"] == expected
    assert mtp["rope"]["rope_type"] == "default"
    assert mtp["rope"]["original_max_position_embeddings"] == 262144
    raw["options"].update(max_len=524288, yarn_factor=2)
    save(root, path, raw)
    with pytest.raises(ValueError, match="explicit MTP"):
        load(root)


def test_combined_fp32_frequency_overflow_refuses_before_manifest_publication(tmp_path):
    root, path, raw = profile_files(tmp_path, context=262144, theta=3e38)
    raw["options"].update(max_len=8388608, yarn_factor=32)
    save(root, path, raw)
    with pytest.raises(ValueError, match="frequenc"):
        load(root)
    output = root / "B.json"
    with pytest.raises(ValueError, match="frequenc"):
        create(output, identifier="overflow", model=root / "target", options=raw["options"], api={}, fixed_options=UNIQUE_DEPLOYMENT)
    assert not output.exists() and not list(root.glob(".serving-*.json"))


def test_expanded_row_allowance_is_independent_of_declared_thw_axis_limits(tmp_path):
    root, path, raw = profile_files(tmp_path, context=262144)
    positional, _ = positional_manifest(root, rows=64, limits=[16, 16, 16])
    raw["options"].update(max_len=64, position_profile_manifest="positions/selected.json")
    save(root, path, raw, add=(positional,))
    position = load(root).record()["position_profile"]
    target = position["resolved"]["target_attention"]
    assert target["max_rows"] == 64 and target["coordinate_limits_exclusive"] == [16, 16, 16]
    assert position["binding"]["actual_live_span_guard_required"] is True



def test_real_numerical_preparation_precedes_artifact_hashing_and_profile_issuance(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp import position_profile
    from tensorfold.families.qwen4_exp.cuda import serving_profiles

    root, _, _ = profile_files(tmp_path)
    events = []
    original_vectors = position_profile.prepare_position_vectors
    original_artifact = serving_profiles.artifact_identity

    def vectors(profile, *, mtp_active):
        prepared = original_vectors(profile, mtp_active=mtp_active)
        assert prepared.target_cpu.device.type == "cpu"
        events.append("vectors")
        return prepared

    def artifact(*args, **kwargs):
        events.append("artifact")
        return original_artifact(*args, **kwargs)

    monkeypatch.setattr(position_profile, "prepare_position_vectors", vectors)
    monkeypatch.setattr(serving_profiles, "artifact_identity", artifact)
    profile = load(root)
    assert events[0] == "vectors" and "artifact" in events and events[-1] == "vectors"
    assert profile.record()["position_profile"]["numerical_preparation"]["finite_positive_vectors"] is True


def test_real_prepared_position_receipts_reach_pending_and_effective_worker_profiles(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import serving_runtime
    from tensorfold.server import runtime
    from test_serving_profile_transitions import pipeline, wait_until

    for name in ("native", "extended"):
        directory = tmp_path / name
        directory.mkdir()
        root, path, raw = profile_files(directory, context=262144)
        raw["api"]["served"] = name
        if name == "extended":
            raw["options"].update(max_len=524288, yarn_factor=2)
        save(root, path, raw)
    # Only GPU load/allocation is doubled by this existing actor harness. Real
    # manifest/schema/source/position/numerical preparation remains in use.
    with pipeline(tmp_path, monkeypatch, world=8) as (engines, app):
        original_prepare = prepare

        def actual_prepare(root, name, **kwargs):
            kwargs["initializer"] = FlashNextEngine._initialize
            # The generic allocation harness passes only rank/world; restore
            # its actual already configured admission/history flags as startup
            # FlashNextEngine.__init__ does for the production owner.
            kwargs["fixed_options"] = {**kwargs["fixed_options"],
                "adaptive_sessions": engines[0].adaptive_sessions, "indexer_history": engines[0].indexer_history,
                "max_waiting_input_bytes": 4096}
            return original_prepare(root, name, **kwargs)

        monkeypatch.setattr(serving_runtime, "prepare", actual_prepare)
        engine = engines[0]
        worker, comm = engine.scheduler.thread, engine.comm
        previous = app.prepare({"prompt": "w1", "max_tokens": 2}, False)
        runtime.request(app, {"serving_profile": "extended/A.json"})
        owner = engine.serving_profiles
        wait_until(lambda: owner.record()["stage"] == "draining")
        pending = owner.record()["pending"]["position_profile"]
        assert pending["resolved"]["target_attention"]["rope"]["factor"] == 2
        assert pending["numerical_preparation"]["finite_positive_vectors"] is True
        previous.close()
        wait_until(lambda: owner.gate.epoch == 1)
        effective = owner.record()["effective"]["position_profile"]
        assert effective == pending and app.context_window == 524288
        runtime.request(app, {"serving_profile": "native/A.json"})
        wait_until(lambda: owner.gate.epoch == 2)
        native = owner.record()["effective"]["position_profile"]
        assert native["resolved"]["target_attention"]["rope"]["rope_type"] == "default"
        assert native["binding"]["position_identity"] != effective["binding"]["position_identity"]
        assert app.context_window == 262144 and engine.scheduler.thread is worker and engine.comm is comm



@pytest.mark.parametrize("depth", [None, 4])
def test_invalid_depth_or_fixed_dflash_block_maximum_refuses_in_preparation(tmp_path, depth):
    root, path, raw = profile_files(tmp_path)
    draft = root / "draft"
    draft.mkdir()
    (draft / "config.json").write_bytes((Path(__file__).parent / "fixtures/dflash2/config.json").read_bytes())
    (draft / "model.safetensors").write_bytes(b"payload import is a separate provider contract")
    raw["options"].update(depth=depth, dflash2_model_dir="draft")
    save(root, path, raw, add=(draft / "config.json", draft / "model.safetensors"))
    with pytest.raises(ValueError, match="depth|at most three"):
        load(root)



def test_legacy_tool_default_preserves_unbound_source_policy_and_never_stores_process_flags(tmp_path):
    root, _, raw = profile_files(tmp_path)
    output = root / "legacy.json"
    result = create(output, identifier="legacy", model=root / "target", options=raw["options"], api={})
    position = result["position_profile"]
    assert position["resolved"] is position["numerical_preparation"] is None
    assert position["binding"]["support"] == "legacy-source-policy"
    assert not {"tp", "rank", "adaptive_sessions", "indexer_history"} & json.loads(output.read_text())["options"].keys()


@pytest.mark.parametrize("fixed", [None, {"tp": 1, "adaptive_sessions": True},
    {"tp": 8, "adaptive_sessions": False}, {"tp": 8, "adaptive_sessions": True, "indexer_history": "ram"}])
def test_explicit_manifest_tool_refuses_missing_or_incompatible_deployment_without_publication(tmp_path, fixed):
    root, _, raw = profile_files(tmp_path, context=262144)
    positional_manifest(root, rows=262144)
    options = {**raw["options"], "position_profile_manifest": "positions/selected.json"}
    output = root / "new.json"
    with pytest.raises(ValueError, match="unique GPU-history"):
        create(output, identifier="new", model=root / "target", options=options, api={}, fixed_options=fixed)
    assert not output.exists() and not list(root.glob(".serving-*.json"))


def test_tool_deployment_flags_are_not_a_profile_controlled_override(tmp_path):
    root, _, raw = profile_files(tmp_path)
    output = root / "new.json"
    with pytest.raises(ValueError, match="process-fixed"):
        create(output, identifier="new", model=root / "target", options={**raw["options"], "tp": 8}, api={})
    with pytest.raises(ValueError, match="process-fixed"):
        create(output, identifier="new", model=root / "target", options=raw["options"], api={},
               fixed_options={"max_len": 20})
    assert not output.exists() and not list(root.glob(".serving-*.json"))


@pytest.mark.parametrize("fixed", [{"tp": 1, "adaptive_sessions": True, "indexer_history": "gpu"},
    {"tp": 8, "adaptive_sessions": False, "indexer_history": "gpu"}])
def test_maintained_legacy_cli_yarn_has_no_new_positional_role_claim(tmp_path, fixed):
    root, path, raw = profile_files(tmp_path, context=262144)
    raw["options"].update(yarn_factor=2, max_len=524288)
    save(root, path, raw)
    profile = prepare(root, path.name, initializer=FlashNextEngine._initialize, fixed_options=fixed)
    position = profile.record()["position_profile"]
    assert position["resolved"] is position["numerical_preparation"] is None
    assert position["binding"]["support"] == "legacy-source-policy"
    assert profile.options["max_len"] == 524288


@pytest.mark.parametrize("source_heads,declared_heads", [(1, 2), (2, 1)])
def test_sidecar_cannot_change_the_actual_unique_history_source_layout(tmp_path, source_heads, declared_heads):
    root, path, raw = profile_files(tmp_path, kv_heads=source_heads)
    target = root / "target/config.json"
    config = json.loads(target.read_text())
    config["text_config"]["num_key_value_heads"] = declared_heads
    target.write_text(json.dumps(config))
    save(root, path, raw)
    with pytest.raises(ValueError, match="KV head geometry conflicts"):
        load(root)



def test_legacy_kv_layout_stays_unbound_and_refuses_explicit_new_manifest(tmp_path):
    root, path, raw = profile_files(tmp_path, context=262144, kv_heads=1)
    position = load(root).record()["position_profile"]
    assert position["resolved"] is position["numerical_preparation"] is None
    assert position["binding"]["support"] == "legacy-source-policy"
    positional, _ = positional_manifest(root, rows=262144)
    raw["options"]["position_profile_manifest"] = "positions/selected.json"
    save(root, path, raw, add=(positional,))
    with pytest.raises(ValueError, match="unique GPU-history"):
        load(root)


def test_actual_tool_cli_uses_explicit_deployment_flags_without_serializing_them(tmp_path, monkeypatch, capsys):
    from tools import prepare_serving_profile

    root, _, raw = profile_files(tmp_path)
    options = root / "options.json"
    options.write_text(json.dumps(raw["options"]))
    output = root / "cli.json"
    monkeypatch.setattr("sys.argv", ["prepare_serving_profile.py", "--model", str(root / "target"),
        "--id", "cli", "--options", str(options), "--out", str(output), "--tp", "8", "--adaptive-sessions",
        "--max-waiting-input-bytes", "4096"])
    prepare_serving_profile.main()
    result = json.loads(capsys.readouterr().out)
    assert result["position_profile"]["resolved"]["qualification"] == "unqualified-candidate"
    assert result["position_profile"]["binding"]["kv_identity"] == get_pair("int8").identity
    assert not {"tp", "rank", "adaptive_sessions", "indexer_history"} & json.loads(output.read_text())["options"].keys()



def test_explicit_legacy_resolution_is_not_repeated_but_later_source_checks_remain(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import serving_profiles

    root, path, _ = profile_files(tmp_path)
    original = serving_profiles.resolve_profile_positions
    calls = []

    def resolve(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(serving_profiles, "resolve_profile_positions", resolve)
    prepared = prepare(root, path.name, initializer=FlashNextEngine._initialize, fixed_options={"tp": 1})
    assert prepared.record()["position_profile"]["resolved"] is None
    assert len(calls) == 2  # Initial binding and required final source check, each once.
    calls.clear()
    prepared.check_sources()
    assert len(calls) == 1
    calls.clear()
    serving_profiles._required_files(prepared.model_dir, prepared.options)
    assert len(calls) == 1  # Omitted argument still requests resolution.
    calls.clear()
    serving_profiles._required_files(prepared.model_dir, prepared.options, positions=None)
    assert not calls  # An explicitly resolved legacy result already established this invariant.


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_legacy_dflash_receipt_uses_actual_fixed_compute_footprint(tmp_path, depth):
    from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry

    root, path, raw = profile_files(tmp_path, kv_heads=1)
    draft = root / "draft"
    draft.mkdir()
    (draft / "config.json").write_bytes((Path(__file__).parent / "fixtures/dflash2/config.json").read_bytes())
    (draft / "model.safetensors").write_bytes(b"payload import is a separate provider contract")
    raw["options"].update(depth=depth, dflash2_model_dir="draft")
    save(root, path, raw, add=(draft / "config.json", draft / "model.safetensors"))
    receipt = load(root).record()["position_profile"]
    expected = MethodGeometry("dflash2", depth).receipt()["draft_compute_block_rows"] - 1
    assert receipt["resolved"] is None and receipt["binding"]["support"] == "legacy-source-policy"
    assert receipt["binding"]["lookahead_rows"] == expected == 3
