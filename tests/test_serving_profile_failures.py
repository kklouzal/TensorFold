"""Actual profile publication/cleanup boundaries under allocation/diagnostic faults."""

import inspect
import json
from types import SimpleNamespace

import pytest

from tensorfold.families import qwen4_exp
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.execution_identity import draft_artifact_files
from tensorfold.families.qwen4_exp.cuda.profile_construction import Construction, ConstructionFatal
from tensorfold.families.qwen4_exp.cuda.serving_profiles import PATH_OPTIONS, _required_files, prepare
from tensorfold.server import runtime
from test_serving_profile_manifest import profile_files
from test_serving_profile_construction import Comm, Store
from test_serving_profile_transitions import LoadDouble, pipeline, wait_until


def contains_exception(root, wanted):
    pending, seen = [root], set()
    while pending:
        value = pending.pop()
        if value is wanted:
            return True
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        pending.extend((value.__cause__, value.__context__))
        if isinstance(value, BaseExceptionGroup):
            pending.extend(value.exceptions)
    return False


@pytest.mark.parametrize("post_publication", [False, True])
def test_admin_wake_fault_keeps_original_failure_and_pending_transition_progress(tmp_path, monkeypatch, post_publication):
    with pipeline(tmp_path, monkeypatch, world=1) as (engines, app):
        owner = engines[0].serving_profiles
        waiting = engines[0].scheduler.waiting
        original, error = waiting.wake, MemoryError("wake publication")
        calls = []

        def wake():
            calls.append("wake")
            if len(calls) == 1:
                if post_publication:
                    original()
                raise error
            return original()

        monkeypatch.setattr(waiting, "wake", wake)
        with pytest.raises(MemoryError) as caught:
            runtime.request(app, {"serving_profile": "B.json"})
        assert caught.value is error and len(calls) >= 2
        wait_until(lambda: owner.gate.epoch == 1)
        assert owner.record()["pending"] is None and owner.failure is None
        assert len(engines[0].loads) == 1


def test_repeated_admin_wake_failure_has_terminal_containment_instead_of_stranding(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch, world=1) as (engines, app):
        owner = engines[0].serving_profiles
        errors = [MemoryError("first wake"), MemoryError("second wake")]
        calls, notices = [], []

        def wake():
            calls.append("wake")
            raise errors[min(len(calls)-1, 1)]

        monkeypatch.setattr(engines[0].scheduler.waiting, "wake", wake)
        monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", lambda: notices.append("fatal"))
        with pytest.raises(ConstructionFatal) as caught:
            runtime.request(app, {"serving_profile": "B.json"})
        assert all(contains_exception(caught.value, error) for error in errors)
        assert calls == ["wake", "wake"] and notices == ["fatal"]
        assert owner.record()["stage"] == "fatal" and owner.pending is None
        assert not owner.gate.record()["accepting"] and not engines[0].loads


@pytest.mark.parametrize("fault", ["format", "store", "notify"])
def test_construction_fatal_diagnostics_cannot_replace_primary_or_delay_notification(monkeypatch, fault):
    events = []
    diagnostic = MemoryError("diagnostic failure")
    notification = OSError("notification failure")

    class Primary(RuntimeError):
        def __str__(self):
            events.append("format")
            if fault == "format":
                raise diagnostic
            return "native submit failure"

    primary = Primary()

    def notify():
        events.append("notify")
        if fault == "notify":
            raise notification

    monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", notify)
    scope = Construction(Comm(0, 1, Store()), 22)

    def put(*args):
        events.append("store")
        if fault == "store":
            raise diagnostic

    monkeypatch.setattr(scope, "put", put)
    with pytest.raises(Primary) as caught:
        scope.fail(primary, fatal=True)
    assert caught.value is primary and scope.fatal
    assert events[0] == "notify"
    assert contains_exception(primary, notification if fault == "notify" else diagnostic)


def test_failed_expected_refusal_publication_escalates_before_peers_can_wait_forever(monkeypatch):
    primary, diagnostic = ValueError("expected before GO"), OSError("store refused failure record")
    notices = []
    scope = Construction(Comm(0, 1, Store()), 23)

    def put(*args):
        raise diagnostic

    monkeypatch.setattr(scope, "put", put)
    monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", lambda: notices.append("fatal"))
    with pytest.raises(ValueError) as caught:
        scope.fail(primary)
    assert caught.value is primary and scope.fatal and notices == ["fatal"]
    assert contains_exception(primary, diagnostic)


def test_real_engine_resource_cleanup_closes_native_owner_before_transport():
    events = []
    engine = object.__new__(FlashNextEngine)
    buffers = SimpleNamespace(owner_graphs=object(), history_owner=object())
    engine.multi = SimpleNamespace(graphs=SimpleNamespace(close=lambda: events.append("graphs")),
                                  recover_failed_slots=lambda: events.append("slots"), buf=buffers)
    engine.e = engine.serial = engine.w = None
    engine._owner_preparation = SimpleNamespace(close=lambda: events.append("owner"))
    engine.comm = SimpleNamespace(close=lambda **kwargs: events.append("comm"))
    engine._close_resources()
    assert events == ["graphs", "owner", "comm", "slots"]
    assert engine._owner_preparation is None and buffers.owner_graphs is buffers.history_owner is None


def test_native_catalogue_field_is_retired_before_non_native_profile_initialization(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch, world=1) as (engines, app):
        engine, events = engines[0], []
        catalogue = SimpleNamespace(close=lambda: events.append("closed"))
        engine._owner_preparation = catalogue
        original_initialize, original_close = LoadDouble._initialize, LoadDouble._close_resources

        def close(owner, **kwargs):
            if owner.w is not None:
                owner.w.meta, owner.w.device = {}, "cpu"
            if owner.multi is not None:
                owner.multi.recover_failed_slots = lambda: None
            FlashNextEngine._close_resources(owner, **kwargs)
            return original_close(owner, **kwargs)

        def initialize(owner, *args, **kwargs):
            assert owner._owner_preparation is None
            events.append("initialize")
            return original_initialize(owner, *args, **kwargs)

        monkeypatch.setattr(LoadDouble, "_close_resources", close)
        monkeypatch.setattr(LoadDouble, "_initialize", initialize)
        runtime.request(app, {"serving_profile": "B.json"})
        wait_until(lambda: engine.serving_profiles.gate.epoch == 1)
        assert events == ["closed", "initialize"] and engine._owner_preparation is None


def test_actual_draft_inventory_uses_single_payload_not_unconsumed_index(tmp_path):
    root, _, _ = profile_files(tmp_path)
    draft = root / "draft"
    draft.mkdir()
    (draft / "config.json").write_text("{}")
    (draft / "model.safetensors").write_bytes(b"actual selected payload")
    (draft / "other.safetensors").write_bytes(b"not consumed by these loaders")
    (draft / "model.safetensors.index.json").write_text('{"weight_map":{"x":"other.safetensors"}}')
    pairing = root / "pairing.json"
    pairing.write_text('{"draft_pairing":{}}')
    declared = draft_artifact_files(draft, "dflash2")
    assert set(declared) == {"config.json", "model.safetensors"}
    required = _required_files(root / "target", {"dflash2_model_dir": draft, "execution_profile_manifest": pairing})
    assert {str(draft / "config.json"), str(draft / "model.safetensors"), str(pairing)} <= required
    assert str(draft / "other.safetensors") not in required
    assert "dflash2_model_dir" in PATH_OPTIONS and "dflash_model_dir" not in PATH_OPTIONS


@pytest.mark.skipif("dflash2" not in qwen4_exp.CUDA_DRAFT_METHODS,
                    reason="DF provider is not in this isolated base; actual merged producer-signature control is required")
def test_merged_dflash2_signature_normalization_and_complete_pairing_inventory(tmp_path, monkeypatch):
    import hashlib

    root, manifest, raw = profile_files(tmp_path)
    signature = inspect.signature(FlashNextEngine._initialize)
    assert "dflash2_model_dir" in signature.parameters and "dflash_model_dir" not in signature.parameters
    draft = root / "draft"
    draft.mkdir()
    (draft / "config.json").write_text("{}")
    (draft / "model.safetensors").write_bytes(b"inventory-only fixture; provider payload validation is separate")
    pairing = root / "pairing.json"
    pairing.write_text('{"draft_pairing":{"draft_config":"draft/config.json","draft_weights":"draft/model.safetensors"}}')
    override = root / "selected-tokenizer.json"
    override.write_bytes((root / "target/tokenizer.json").read_bytes())
    monkeypatch.setattr("tensorfold.token_metadata.tokenizer_override", lambda: SimpleNamespace(path=override))
    raw["options"].update(dflash2_model_dir="draft", execution_profile_manifest="pairing.json")
    for path in (draft / "config.json", draft / "model.safetensors", pairing):
        raw["artifacts"].append({"path": str(path.relative_to(root)), "size": path.stat().st_size,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="cover every"):
        prepare(root, "A.json", initializer=FlashNextEngine._initialize, fixed_options={"tp": 8, "rank": 0})
    raw["artifacts"].append({"path": override.name, "size": override.stat().st_size,
                             "sha256": hashlib.sha256(override.read_bytes()).hexdigest()})
    manifest.write_text(json.dumps(raw))
    actual = prepare(root, "A.json", initializer=FlashNextEngine._initialize, fixed_options={"tp": 8, "rank": 0})
    assert actual.options["dflash2_model_dir"] == draft and actual.options["execution_profile_manifest"] == pairing
    assert str(override) in {item["path"] for item in actual.artifacts}
