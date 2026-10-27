"""Actual App resource ordering and conditional native-grammar admission, no CUDA."""
from dataclasses import asdict
from types import SimpleNamespace as NS
from types import MappingProxyType
import weakref

import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tokenizers import Tokenizer, decoders, models

from tensorfold.cuda import server
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.profile_binding import ProfileGate
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
from tensorfold.families.qwen4_exp.cuda.retention_resources import HostBaseline, tokenizer_presentation_bound
from tensorfold.server.api_records import ApiRecordStore
from tensorfold.server.errors import RequestError
from tensorfold.server.token_routes import count_chat
from tests.cuda_http_fixtures import model_directory
from tests.test_api_records import limits
from tests.test_cuda_admission import Engine
from tests.test_cuda_admission import http_server
from tests.test_conversations_api import call
from tests.test_serving_profile_transitions import PreparedDouble
from tests.test_retention_resources import resource_decoder


class ConstraintEngine(Engine):
    limit = 128

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None):
        return super().generate(prompt, max_tokens, sampling, on_tokens, draft)


class Compiler:
    def __init__(self):
        self.specs = []

    def compile(self, spec):
        self.specs.append(spec)
        return "compiled"


def selected_tokenizer():
    tok = Tokenizer(models.BPE({"a": 0, "b": 1, "h": 2}, []))
    tok.decoder = decoders.ByteLevel()
    return tok


def catalog(policy):
    return NS(policy=policy, world=1, states=[NS(limit=131, logical_context=128)], _manifest_bytes=1024,
              _buffer=None, _ram=None, has_retained=False, prefix_limit=2, minimum_future_bytes=1 << 20,
              host_resources=HostBaseline(0, 0), _cohort=NS(multi=resource_decoder()),
              check_health=lambda: None, evidence="source-candidate")


def configured_app(tmp_path, monkeypatch, policy, *, max_http_body_bytes=None, logical_api_limits=None,
                   output_queue_tokens=None, managed=False):
    model_directory(tmp_path)
    engine = ConstraintEngine()
    engine.retention_catalog = catalog(policy)
    engine.scheduler = Scheduler(NS(), max_streams=1, admission_policy="fifo", max_waiting_input_bytes=16384,
                                 output_queue_tokens=output_queue_tokens)
    engine.vision = NS(is_media_runtime=True, tokenizer=selected_tokenizer())
    engine.chat_template = server.ChatTemplate(tmp_path)
    if managed:
        engine.serving_profiles = NS(apps=weakref.WeakSet(), gate=ProfileGate())
    owned, admission = [], []
    def admit(total, *, materialized_owned_bytes):
        # The old early configure_requests ordering would already have bound it.
        assert engine.scheduler._request_limit is None
        assert owned[0].tok is engine.vision.tokenizer
        admission.append((total, materialized_owned_bytes))
        return {"release_qualified": False}
    monkeypatch.setattr("tensorfold.cuda.capacity.admit_host_commitment", admit)
    app = server.App(engine, tmp_path, "resource-test", max_tokens=3, max_pending_requests=2,
                     max_http_connections=4, max_http_body_bytes=max_http_body_bytes,
                     logical_api_limits=logical_api_limits, startup_owner=owned.append)
    return app, engine, admission


def test_actual_adopted_tokenizer_and_host_receipt_precede_scheduler_admission(tmp_path, monkeypatch):
    policy = RetentionPolicy("ram", 2, 10, 1 << 30, 4096, 8192, 0)
    app, engine, admission = configured_app(tmp_path, monkeypatch, policy)
    receipt = engine.retention_catalog.host_resource_envelope
    assert len(admission) == 1 and receipt["max_pending_requests"] == 2
    assert receipt["max_http_connections"] == 4 and receipt["global_waiting_input_bytes"] == 16384
    assert receipt["context_tokens"] == 128 and receipt["output_queue_tokens"] is None
    assert receipt["tokenizer"]["tokenizer_identity"] == tokenizer_presentation_bound(app.tok).tokenizer_identity
    assert receipt["authority"]["release_qualified"] is False
    assert engine.scheduler._request_limit is app._request_limit


def test_full_future_input_credit_caps_generation_owners_through_between_subruns(monkeypatch):
    from tensorfold.cuda.scheduler import AdmissionStatus, Waiting
    from tensorfold.families.qwen4_exp.cuda.retention_resources import finalize_retention_resources
    from tensorfold.server.errors import CapacityError

    selected = catalog(None)
    selected.states = [NS(limit=262147, logical_context=262144)]
    decoder = selected._cohort.multi = resource_decoder(262144)
    floor = decoder.input_reservation_bytes()
    monkeypatch.setattr("tensorfold.cuda.capacity.admit_host_commitment",
                        lambda total, **_kwargs: {"diagnostic_total": total})
    receipt = finalize_retention_resources(selected, tokenizer=selected_tokenizer(),
        max_pending_requests=4, max_waiting_input_bytes=16 << 20, max_http_connections=1)
    assert receipt["context_tokens"] == 262144
    assert selected.states[0].limit == 262147
    assert receipt["generation_owner_bound"]["mandatory_input_claim_bytes"] == floor
    assert receipt["generation_owner_bound"]["owners"] == 1
    assert receipt["global_model_cpu_projection_bytes"] == receipt["continuation"]["project_payload_bytes"]

    waiting = Waiting(policy="fifo", max_input_bytes=16 << 20)
    def entry():
        return NS(admission_status=AdmissionStatus(-1, 0.0, floor, None)), object()
    first, second = entry(), entry()
    claim = waiting.publish(first, reserved_bytes=floor, register=lambda _claim: None)
    claim.backend_retired(first[1])
    assert first[0].admission_status.phase == "between_subruns"
    with pytest.raises(CapacityError):
        waiting.publish(second, reserved_bytes=floor)
    claim.close()
    replacement = waiting.publish(second, reserved_bytes=floor)
    replacement.close()


def test_actual_output_credit_cap_reduces_admitted_mailbox_payload(tmp_path, monkeypatch):
    full_dir, bounded_dir = tmp_path / "full", tmp_path / "bounded"
    full_dir.mkdir()
    bounded_dir.mkdir()
    _, full, _ = configured_app(full_dir, monkeypatch, None)
    _, bounded, _ = configured_app(bounded_dir, monkeypatch, None, output_queue_tokens=4)
    full_receipt = full.retention_catalog.host_resource_envelope
    bounded_receipt = bounded.retention_catalog.host_resource_envelope
    assert full_receipt["output_queue_tokens"] is None
    assert bounded_receipt["output_queue_tokens"] == bounded.scheduler.output_queue_tokens == 4
    full_mailbox = full_receipt["continuation"]["mailbox"]
    bounded_mailbox = bounded_receipt["continuation"]["mailbox"]
    assert 0 < bounded_mailbox < full_mailbox
    assert bounded_receipt["global_host_commitment_bytes"] < full_receipt["global_host_commitment_bytes"]


def test_actual_typed_logical_api_limits_reach_pre_admission_receipt(tmp_path, monkeypatch):
    selected = limits()
    app, engine, admission = configured_app(tmp_path, monkeypatch, None, logical_api_limits=selected)
    receipt = engine.retention_catalog.host_resource_envelope
    assert app.logical_api_limits is selected and len(admission) == 1
    assert receipt["logical_api"]["limits"] == asdict(selected)
    assert receipt["global_logical_api_projection_bytes"] > 0
    assert receipt["max_http_body_bytes"] == app.max_http_body_bytes


def test_failed_host_admission_refuses_before_configuring_requests(tmp_path, monkeypatch):
    model_directory(tmp_path)
    engine = ConstraintEngine()
    engine.retention_catalog = catalog(RetentionPolicy("gpu", 1, 10))
    engine.scheduler = Scheduler(NS(), max_streams=1, admission_policy="fifo", max_waiting_input_bytes=4096)
    tok = selected_tokenizer()
    tok.decoder = decoders.Sequence([decoders.ByteLevel(), decoders.Replace("a", "large" * 1000)])
    engine.vision = NS(is_media_runtime=True, tokenizer=tok)
    owned = []
    primary = ValueError("actual host commitment does not fit")
    def refuse(*_args, **_kwargs):
        raise primary
    monkeypatch.setattr("tensorfold.cuda.capacity.admit_host_commitment", refuse)
    with pytest.raises(ValueError) as caught:
        server.App(engine, tmp_path, "resource-test", max_pending_requests=2, max_http_connections=4,
                   startup_owner=owned.append)
    assert caught.value is primary
    assert len(owned) == 1 and owned[0].tok is tok
    assert engine.scheduler._request_limit is None and not hasattr(engine.retention_catalog, "host_resource_envelope")


@pytest.mark.parametrize("tier,logical,refused", [("gpu", False, False), ("gpu", True, True),
                                                 ("ram", False, True), ("ram", True, True),
                                                 (None, True, False)])
def test_native_constraint_gate_uses_runtime_policy_and_trusted_logical_owner(tmp_path, monkeypatch, tier, logical, refused):
    policy = None if tier is None else RetentionPolicy(tier, 1, 10, **({"ram_bytes": 1 << 30,
        "chunk_bytes": 4096, "progress_bytes": 8192, "overlap_bytes": 0} if tier == "ram" else {}))
    app, engine, _ = configured_app(tmp_path, monkeypatch, policy)
    compiler = Compiler()
    app.grammars = compiler
    body = {"messages": [{"role": "user", "content": "a"}], "guided_choice": ["a", "b"],
            "metadata": {"logical_retained": "client hint has no authority"}}
    store = ApiRecordStore(tmp_path / "api.sqlite", limits=limits())
    try:
        with store.response_operation("trusted", "resp_test", previous_response_id=None, conversation_id=None,
                                       added=[{"role": "user", "content": "a"}], stored=logical) as operation, operation.execution_scope():
            if refused:
                with pytest.raises(RequestError, match="native matcher CPU-memory envelope"):
                    app.prepare(body, True)
                assert compiler.specs == []
            else:
                prepared = app.prepare(body, True)
                assert prepared.grammar[1] == "compiled" and len(compiler.specs) == 1
                prepared.close()
    finally:
        store.close()
    assert engine.calls == []


def test_ram_generation_refuses_before_strict_plan_but_counting_preserves_constraint_validation(tmp_path, monkeypatch):
    app, _, _ = configured_app(tmp_path, monkeypatch,
        RetentionPolicy("ram", 1, 10, 1 << 30, 4096, 8192, 0))
    compiler = Compiler()
    app.grammars = compiler
    body = {"messages": [{"role": "user", "content": "a"}], "guided_choice": ["a", "b"]}
    assert count_chat(app, body) > 0 and len(compiler.specs) == 1
    assert app.tokenize(body)["count"] > 0 and len(compiler.specs) == 2
    strict = {"messages": body["messages"], "tools": [{"type": "function", "function": {
        "name": "tool", "strict": True, "parameters": {"type": "object"}}}]}
    monkeypatch.setattr(app, "_strict_calls", lambda *_args: pytest.fail("native strict Plan reached before capability gate"))
    with pytest.raises(RequestError, match="native matcher CPU-memory envelope"):
        app.prepare(strict, True)
    assert len(compiler.specs) == 2


def test_ordinary_engine_keeps_unconfigured_constraints(tmp_path):
    model_directory(tmp_path)
    engine = ConstraintEngine()
    app = server.App(engine, tmp_path, "ordinary")
    compiler = Compiler()
    app.grammars = compiler
    prepared = app.prepare({"messages": [{"role": "user", "content": "a"}], "guided_choice": ["a"]}, True)
    assert prepared.grammar[1] == "compiled" and len(compiler.specs) == 1
    prepared.close()


def test_actual_route_aggregation_is_in_the_pre_admission_host_receipt(tmp_path, monkeypatch):
    from tensorfold.cuda.http import GENERATION_BODY_LIMIT
    normal, larger = tmp_path / "normal", tmp_path / "larger"
    normal.mkdir()
    larger.mkdir()
    _, first, _ = configured_app(normal, monkeypatch, RetentionPolicy("gpu", 1, 10))
    _, second, _ = configured_app(larger, monkeypatch, RetentionPolicy("gpu", 1, 10), max_http_body_bytes=2 * GENERATION_BODY_LIMIT)
    small = first.retention_catalog.host_resource_envelope
    large = second.retention_catalog.host_resource_envelope
    assert small["max_http_body_bytes"] == GENERATION_BODY_LIMIT
    assert large["max_http_body_bytes"] == 2 * GENERATION_BODY_LIMIT
    assert large["global_transport_projection_bytes"] == 2 * small["global_transport_projection_bytes"]


def test_unconfigured_fifo_retention_preserves_other_effective_decoders(tmp_path, monkeypatch):
    model_directory(tmp_path)  # WordLevel/no decoder has no new tier-presentation bound.
    engine = ConstraintEngine()
    engine.retention_catalog = catalog(None)
    engine.scheduler = Scheduler(NS(), max_streams=1, admission_policy="fifo", max_waiting_input_bytes=4096)
    monkeypatch.setattr("tensorfold.cuda.capacity.admit_host_commitment", lambda *_args, **_kwargs: {"release_qualified": False})
    app = server.App(engine, tmp_path, "ordinary-fifo", max_pending_requests=2, max_http_connections=2)
    assert engine.scheduler._request_limit is app._request_limit
    assert engine.retention_catalog.host_resource_envelope["tokenizer"]["decoder"]
    compiler = Compiler()
    app.grammars = compiler
    prepared = app.prepare({"messages": [{"role": "user", "content": "a"}], "guided_choice": ["a"]}, True)
    assert prepared.grammar[1] == "compiled"
    prepared.close()


@pytest.mark.parametrize("for_count", [False, True])
def test_prepared_grammar_rechecks_trusted_retention_before_matcher_or_model(tmp_path, monkeypatch, for_count):
    app, engine, _ = configured_app(tmp_path, monkeypatch, RetentionPolicy("gpu", 1, 10))
    compiler = Compiler()
    compiler.constraint = lambda *_args, **_kwargs: pytest.fail("native matcher reached before retained gate")
    app.grammars = compiler
    body = {"messages": [{"role": "user", "content": "a"}], "guided_choice": ["a"]}
    prepared = (app.prepare_for_count if for_count else app.prepare)(body, True)
    assert prepared.grammar[1] == "compiled"
    store = ApiRecordStore(tmp_path / "api.sqlite", limits=limits())
    try:
        with store.response_operation("trusted", "resp_new", previous_response_id=None, conversation_id=None,
            added=[{"role": "user", "content": "a"}], stored=True) as operation, operation.execution_scope():
            with pytest.raises(RequestError, match="native matcher CPU-memory envelope"):
                app.run(body, True, lambda _delta: None, prepared=prepared)
        assert engine.calls == []
    finally:
        prepared.close()
        store.close()


def test_prepare_cleanup_lookup_failure_preserves_validation_primary_and_owner(tmp_path, monkeypatch):
    model_directory(tmp_path)
    app = server.App(ConstraintEngine(), tmp_path, "cleanup")
    primary, cleanup = RequestError("validation refused"), MemoryError("retirement method lookup")
    owner = NS(closed=False)
    def retire(value):
        value.closed = True
        return True
    class LookupFault(server.PreparedRequest):
        def __getattribute__(self, name):
            if name == "release_media":
                raise cleanup
            return super().__getattribute__(name)
    prepared = LookupFault([1], 1, [], False, None, vision=owner, _media_release=retire)
    monkeypatch.setattr(app, "_prepare", lambda *_args, **_kwargs: prepared)
    def refuse(*_args, **_kwargs):
        raise primary
    monkeypatch.setattr(app, "check", refuse)
    with pytest.raises(RequestError) as caught:
        app.prepare({"messages": [{"role": "user", "content": "a"}]}, True)
    assert caught.value is primary and prepared in primary._tensorfold_retained_owners
    assert not owner.closed
    server.PreparedRequest.release_media(prepared)
    assert owner.closed


def test_actual_cuda_dispatch_and_host_projection_share_existing_body_caps(tmp_path, monkeypatch):
    from tensorfold.cuda import http
    model_directory(tmp_path)
    app = server.App(ConstraintEngine(), tmp_path, "caps")
    observed = []
    original = http.read_body
    def capped(handler, *, limit):
        observed.append((handler.path, limit))
        return original(handler, limit=limit)
    monkeypatch.setattr(http, "read_body", capped)
    with http_server(app) as port:
        assert call(port, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "a"}], "max_tokens": 1})[0] == 200
        assert call(port, "POST", "/v1/completions", {"prompt": "a", "max_tokens": 1})[0] == 200
        assert call(port, "POST", "/v1/tokenize", {"prompt": "a"})[0] == 200
    assert observed == [("/v1/chat/completions", 96 << 20), ("/v1/completions", 96 << 20), ("/v1/tokenize", 32 << 20)]
    assert app.max_http_body_bytes == http.GENERATION_BODY_LIMIT == 96 << 20


@pytest.mark.parametrize("fails", [False, True])
def test_actual_profile_stage_finalizes_adopted_frontend_before_publication(tmp_path, monkeypatch, fails):
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    app, engine, _ = configured_app(old_dir, monkeypatch, RetentionPolicy("gpu", 1, 10),
                                    output_queue_tokens=3, managed=True)
    old_tok, old_model = app.tok, app.model_dir
    model_directory(new_dir)  # Deliberately not the adopted ByteLevel frontend.
    selected = selected_tokenizer()
    selected.add_tokens(["actual new frontend"])
    engine.vision = NS(is_media_runtime=True, tokenizer=selected, check_media_health=lambda: None)
    engine.chat_template = server.ChatTemplate(new_dir)
    engine.context_window = 128
    engine.retention_catalog = catalog(RetentionPolicy("gpu", 1, 10))
    primary = ValueError("host admission refused staged profile")
    calls = []
    def admission(total, *, materialized_owned_bytes):
        assert app.tok is old_tok and app.model_dir == old_model
        calls.append(total)
        if fails:
            raise primary
        return {"release_qualified": False}
    monkeypatch.setattr("tensorfold.cuda.capacity.admit_host_commitment", admission)
    profile = PreparedDouble("new.json", "new", "a" * 64, "b" * 64, new_dir,
                              MappingProxyType({}), MappingProxyType({"served": "new"}), ())
    if fails:
        with pytest.raises(ValueError) as caught:
            app._profiles.stage(profile)
        assert caught.value is primary and not hasattr(engine.retention_catalog, "host_resource_envelope")
    else:
        binding = app._profiles.stage(profile)
        assert binding["tok"] is selected
        assert engine.retention_catalog.host_resource_envelope["tokenizer"]["tokenizer_identity"] == tokenizer_presentation_bound(selected).tokenizer_identity
        assert engine.retention_catalog.host_resource_envelope["output_queue_tokens"] == 3
        assert app.tok is old_tok  # stage only; publication remains the explicit owner action
        app._profiles.publish(binding)
        assert app.tok is selected and app.model_dir == new_dir
    assert len(calls) == 1
    if fails:
        assert app.tok is old_tok and app.model_dir == old_model
