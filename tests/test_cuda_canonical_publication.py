"""Canonical publication uses prepared renderer authority and actual response fields."""
import json

import pytest

from tensorfold.cuda import server
from tensorfold.server.cancellation import RequestCancelled
from tests.test_cuda_server_disconnect import PacedEngine, app_for


class PublishingEngine(PacedEngine):
    supports_canonical_prefix_archive = True

    def __init__(self, text="abc"):
        super().__init__(text)
        self.intents, self.published, self.cancelled = [], [], []
        self.app = None
        self.failure = None
        self.metadata_leases = []
        self.metadata_failure = None
        self.registrations = []

    def register_cache_metadata_owner(self, owner):
        from types import SimpleNamespace
        record = SimpleNamespace(owner=owner, closed=False, failed=False, retiring=False, consumer_done=False, close_failed=False)
        def close():
            record.consumer_done = True
            if not record.failed:
                record.closed = True
        record.close = close
        self.registrations.append(record)
        return record

    def reserve_cache_metadata(self, nbytes):
        from types import SimpleNamespace
        from tensorfold.engine.prefix_blocks import PrefixCapacity
        if self.metadata_failure is not None:
            raise PrefixCapacity("terminal cache failure")
        lease = SimpleNamespace(nbytes=nbytes, closed=False)
        def close():
            lease.closed = True
        lease.close = close
        self.metadata_leases.append(lease)
        return lease

    def cache_metadata_failed(self, primary, owner, *, close_failed=False):
        record = next(record for record in self.registrations if record.owner is owner)
        record.failed = True
        record.close_failed |= close_failed
        if self.metadata_failure is None:
            self.metadata_failure = (primary, owner)

    def cache_publication_intent(self, mode):
        intent = object()
        self.intents.append((mode, intent))
        return intent

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, _cache_publication=None, vision=None):
        if self.supports_canonical_prefix_archive is True:
            assert _cache_publication is self.intents[-1][1]
        else:
            assert _cache_publication is None
        if self.failure is not None:
            raise self.failure
        return super().generate(prompt, max_tokens, sampling, on_tokens, draft)

    def publish_canonical(self, tokens, prepared_media=None, *, intent, cancelled=None):
        assert self.app._turns().busy
        self.published.append((list(tokens), prepared_media, intent))
        return {"complete": True}

    def cancel_cache_publication(self, intent):
        self.cancelled.append(intent)


def app(tmp_path, text="abc"):
    engine = PublishingEngine(text)
    result = app_for(tmp_path, engine)
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    result.tok = Tokenizer(models.BPE({chr(i): i for i in range(128)}, [], unk_token="?"))
    result.tok.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    result.tok.decoder = decoders.Fuse()
    result.template = server.ChatTemplate(tmp_path)
    engine.app = result
    return result, engine


def test_chat_snapshot_ignores_mutated_body_and_retains_turn(tmp_path):
    frontend, engine = app(tmp_path)
    body = {"messages": [{"role": "user", "content": "original"}], "max_tokens": 3}
    prepared = frontend.prepare(body, True)
    body["messages"][0]["content"] = "mutated"
    result = frontend.run(body, True, lambda _: True, prepared=prepared)
    decoded = frontend.tok.decode(engine.published[0][0], skip_special_tokens=False)
    assert "original" in decoded and "mutated" not in decoded and "abc" in decoded
    assert result["stats"]["cache_publication"] == {"complete": True}
    assert not frontend._turns().busy and not engine.cancelled


def test_raw_text_reencodes_completed_string_instead_of_ids_concat(tmp_path):
    frontend, engine = app(tmp_path)
    body = {"prompt": "ab", "max_tokens": 3, "add_special_tokens": False}
    prepared = frontend.prepare(body, False)
    body["prompt"] = "wrong"
    frontend.run(body, False, lambda _: True, prepared=prepared)
    expected = frontend.tok.encode("ababc", add_special_tokens=False).ids
    assert engine.published[0][0] == expected
    assert engine.intents[0][0] == "frontend"


def test_raw_ids_have_exclusive_request_intent(tmp_path):
    frontend, engine = app(tmp_path)
    body = {"prompt": [65, 66], "max_tokens": 3}
    frontend.run(body, False, lambda _: True)
    assert engine.intents[0][0] == "raw_ids"
    assert engine.published[0][0] == [65, 66, 97, 98, 99]


def test_generation_failure_cancels_intent_and_releases_turn(tmp_path):
    frontend, engine = app(tmp_path)
    primary = RuntimeError("native generation failure")
    engine.failure = primary
    with pytest.raises(RuntimeError) as caught:
        frontend.run({"prompt": "x", "max_tokens": 3}, False, lambda _: True)
    assert caught.value is primary
    assert len(engine.cancelled) == 1 and not engine.published
    assert not frontend._turns().busy


def test_disconnect_cancels_without_publication(tmp_path):
    frontend, engine = app(tmp_path)
    with pytest.raises(RequestCancelled):
        frontend.run({"prompt": "x", "max_tokens": 3}, False, lambda _: False)
    assert len(engine.cancelled) == 1 and not engine.published


def test_accepted_tool_deltas_preserve_actual_id_and_argument_bytes():
    calls = []
    server._accepted_tool_delta(calls, {"tool_calls": [
        {"index": 0, "id": "actual_wire_id", "function": {"name": "f", "arguments": "{"}}]})
    server._accepted_tool_delta(calls, {"tool_calls": [
        {"index": 0, "function": {"arguments": '"x":1}'}}]})
    assert server._finished_wire_calls(calls) == [{"id": "actual_wire_id", "type": "function",
                      "function": {"name": "f", "arguments": '{"x":1}'}}]
    with pytest.raises(RuntimeError, match="ordered"):
        server._accepted_tool_delta(calls, {"tool_calls": [{"index": 2}]})


def test_template_clock_frozen_and_final_render_has_no_generation_opener(tmp_path):
    from datetime import datetime
    source = '{{ strftime_now("%Y-%m-%d") }}:{% for m in messages %}{{m.content}};{% endfor %}{% if add_generation_prompt %}OPEN{% endif %}'
    frontend, engine = app(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": source}))
    frontend.template = server.ChatTemplate(tmp_path)
    prepared = frontend.prepare({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}, True)
    from dataclasses import replace
    prepared.canonical = replace(prepared.canonical, render_time=datetime(2000, 1, 2))
    frontend.run({"messages": [], "max_tokens": 3}, True, lambda _: True, prepared=prepared)
    text = frontend.tok.decode(engine.published[0][0], skip_special_tokens=False)
    assert text.startswith("2000-01-02:") and "OPEN" not in text


def test_changed_frontend_authority_refused(tmp_path):
    frontend, engine = app(tmp_path)
    body = {"prompt": "x", "max_tokens": 3}
    prepared = frontend.prepare(body, False)
    frontend.tok = object()
    with pytest.raises(RuntimeError, match="authority changed"):
        prepared.canonical.encode(frontend, prepared, "a", None, [])


def test_chat_metadata_pressure_does_not_reject_generation(tmp_path, monkeypatch):
    from tensorfold.server.errors import CapacityError
    from tensorfold.server import host_payload

    frontend, engine = app(tmp_path)
    original = host_payload.json_wire_bound
    def no_metadata(*args, **kwargs):
        if "shape" in kwargs:
            raise CapacityError("no optional metadata room")
        return original(*args, **kwargs)
    monkeypatch.setattr(host_payload, "json_wire_bound", no_metadata)
    result = frontend.run({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
                          True, lambda _: True)
    assert result["content"] == "abc"
    assert result["stats"]["cache_publication"]["status"] == "metadata_unfunded"
    assert len(engine.cancelled) == 1 and not engine.published


@pytest.mark.parametrize("preserve", [False, True])
def test_actual_template_decides_wire_reasoning_retention(tmp_path, preserve):
    frontend, _ = app(tmp_path)
    source = '{% for m in messages %}{{m.content}}{% if preserve_reasoning %}{{m.reasoning_content|default("")}}{% endif %};{% endfor %}'
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": source}))
    frontend.template = server.ChatTemplate(tmp_path)
    body = {"messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"preserve_reasoning": preserve}, "max_tokens": 3}
    prepared = frontend.prepare(body, True)
    tokens, _ = prepared.canonical.encode(frontend, prepared, "VISIBLE", None, [], reasoning="REASON")
    text = frontend.tok.decode(tokens, skip_special_tokens=False)
    assert "VISIBLE" in text
    assert ("REASON" in text) is preserve


def test_stream_finalization_uses_accepted_text_and_emits_no_duplicate_tail(tmp_path):
    frontend, engine = app(tmp_path)
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "stream": True}
    prepared = frontend.prepare(body, True)
    body["stream"] = False  # publication mode belongs to prepare, not this mutable object
    deltas = []
    result = frontend.run(body, True, lambda delta: deltas.append(delta) or True, prepared=prepared)
    assert "".join(delta.get("content", "") for delta in deltas if delta) == "abc"
    assert result["final"] == {}
    assert "abc" in frontend.tok.decode(engine.published[0][0], skip_special_tokens=False)


def test_publication_error_remains_primary_when_intent_cancel_also_fails(tmp_path):
    frontend, engine = app(tmp_path)
    primary, cleanup = RuntimeError("publish failed"), RuntimeError("cancel failed")
    def publish(*args, **kwargs):
        raise primary
    def cancel(*args):
        raise cleanup
    engine.publish_canonical, engine.cancel_cache_publication = publish, cancel
    with pytest.raises(RuntimeError) as caught:
        frontend.run({"prompt": "hi", "max_tokens": 3}, False, lambda _: True)
    assert caught.value is primary
    assert not frontend._turns().busy
    assert caught.value.__cause__ is cleanup


def test_baseline_engine_without_publication_api_keeps_existing_finalization(tmp_path):
    engine = PacedEngine("abc")
    frontend = app_for(tmp_path, engine)
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}
    prepared = frontend.prepare(body, True)
    assert prepared.canonical is None
    result = frontend.run(body, True, lambda _: True, prepared=prepared)
    assert result["content"] == "abc" and "cache_publication" not in result["stats"]


def test_gate_subruns_share_one_intent(tmp_path, monkeypatch):
    frontend, engine = app(tmp_path)
    def two_runs(generate, prompt, max_tokens, gates, on_tokens, **kwargs):
        generate(prompt, 1, on_tokens)
        return generate([*prompt, ord("a")], 1, on_tokens)
    monkeypatch.setattr(server, "generate_gated", two_runs)
    frontend.run({"prompt": "hi", "max_tokens": 3}, False, lambda _: True)
    assert len(engine.calls) == 2 and len(engine.intents) == len(engine.published) == 1


def test_turn_release_failure_cannot_replace_publication_primary(tmp_path):
    frontend, engine = app(tmp_path)
    primary, cleanup = RuntimeError("publication"), RuntimeError("turn release")
    def publish(*args, **kwargs):
        raise primary
    def give():
        raise cleanup
    engine.publish_canonical = publish
    frontend._turns().give = give
    with pytest.raises(RuntimeError) as caught:
        frontend.run({"prompt": "hi", "max_tokens": 3}, False, lambda _: True)
    assert caught.value is primary and caught.value.__cause__ is cleanup
    assert len(engine.cancelled) == 1


def test_actual_stream_tool_ids_are_used_by_terminal_result_and_canonical_render(tmp_path):
    from tests.test_cuda_server_disconnect import TOOLS
    text = "<tool_call>\n<function=measure>\n</function>\n</tool_call>\x00"
    frontend, engine = app(tmp_path, text)
    source = '{% for m in messages %}{{m.content}}{{m.tool_calls|default([])|tojson}};{% endfor %}'
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": source}))
    frontend.template = server.ChatTemplate(tmp_path)
    deltas = []
    body = {"messages": [{"role": "user", "content": "measure"}], "tools": TOOLS,
            "stream": True, "max_tokens": len(text)}
    result = frontend.run(body, True, lambda delta: deltas.append(delta) or True)
    heads = [call for delta in deltas if delta for call in delta.get("tool_calls", ()) if "id" in call]
    assert heads and result["finish"] == "tool_calls"
    assert result["calls"][0]["id"] == heads[0]["id"]
    assert result["calls_streamed"] == 1
    canonical = frontend.tok.decode(engine.published[0][0], skip_special_tokens=False)
    assert heads[0]["id"] in canonical
    assert result["final"] == {}


def test_metadata_bound_precedes_snapshot_serialization(tmp_path, monkeypatch):
    frontend, _ = app(tmp_path)
    from tensorfold.engine.prefix_blocks import PrefixCapacity
    def refuse(nbytes):
        raise PrefixCapacity("no metadata credit")
    frontend.engine.reserve_cache_metadata = refuse
    def forbidden(*args, **kwargs):
        raise AssertionError("unfunded snapshot serialized")
    monkeypatch.setattr(server.json, "dumps", forbidden)
    captured = server._CanonicalInput.capture(frontend, "text", {"prompt": "x" * 50})
    assert captured.kind == "unfunded"


def test_metadata_lease_is_paid_before_snapshot_and_released_after_publication(tmp_path):
    frontend, engine = app(tmp_path)
    body = {"prompt": "hello", "max_tokens": 3}
    prepared = frontend.prepare(body, False)
    assert engine.metadata_leases and not any(lease.closed for lease in engine.metadata_leases)
    frontend.run(body, False, lambda _: True, prepared=prepared)
    assert all(lease.closed for lease in engine.metadata_leases)
    assert prepared.canonical is None


def test_unexpected_failure_retains_credit_and_latches_future_metadata(tmp_path):
    from tensorfold.engine.prefix_blocks import PrefixCapacity
    frontend, engine = app(tmp_path)
    engine.failure = RuntimeError("unexpected")
    with pytest.raises(RuntimeError) as caught:
        frontend.run({"prompt": "hi", "max_tokens": 3}, False, lambda _: True)
    assert engine.metadata_failure[0] is caught.value
    owner = engine.metadata_failure[1]
    assert owner.retained and any(not lease.closed for lease in engine.metadata_leases)
    with pytest.raises(PrefixCapacity):
        engine.reserve_cache_metadata(1)
    # The positive terminal engine/cache close owns this retirement.
    owner.registration.retiring = True
    owner.close()
    assert all(lease.closed for lease in engine.metadata_leases)


def test_cancel_retires_metadata_without_terminal_cache_failure(tmp_path):
    frontend, engine = app(tmp_path)
    with pytest.raises(RequestCancelled):
        frontend.run({"prompt": "hi", "max_tokens": 3}, False, lambda _: False)
    assert engine.metadata_failure is None
    assert all(lease.closed for lease in engine.metadata_leases)


def test_optional_credit_refusal_suppresses_raw_postpass_without_generation_failure(tmp_path):
    from tensorfold.engine.prefix_blocks import PrefixCapacity
    frontend, engine = app(tmp_path)
    def refuse(nbytes):
        raise PrefixCapacity("pinned byte budget")
    engine.reserve_cache_metadata = refuse
    result = frontend.run({"prompt": "hi", "max_tokens": 3}, False, lambda _: True)
    assert result["content"] == "abc" and not engine.published
    assert len(engine.intents) == len(engine.cancelled) == 1
    assert result["stats"]["cache_publication"]["status"] == "metadata_unfunded"


def test_registration_is_prepaid_before_any_metadata_reservation(tmp_path):
    frontend, engine = app(tmp_path)
    original = engine.reserve_cache_metadata
    def reserve(nbytes):
        assert engine.registrations and engine.registrations[0].owner.entries
        return original(nbytes)
    engine.reserve_cache_metadata = reserve
    frontend.run({"prompt": "hi", "max_tokens": 3}, False, lambda _: True)
    assert engine.registrations[0].closed


def test_capture_allocation_failure_retains_registered_owner_and_primary(tmp_path, monkeypatch):
    frontend, engine = app(tmp_path)
    primary = MemoryError("snapshot allocation")
    def fail(*args, **kwargs):
        raise primary
    monkeypatch.setattr(server.json, "dumps", fail)
    with pytest.raises(MemoryError) as caught:
        frontend.prepare({"prompt": "hi", "max_tokens": 3}, False)
    assert caught.value is primary and engine.metadata_failure[0] is primary
    owner = engine.metadata_failure[1]
    assert owner is engine.registrations[0].owner
    assert any(not lease.closed for lease in engine.metadata_leases)
    owner.registration.retiring = True
    owner.close()
    assert all(lease.closed for lease in engine.metadata_leases)


def test_expected_prepare_refusal_retires_credit_without_terminal_latch(tmp_path):
    from tensorfold.server.errors import RequestError
    frontend, engine = app(tmp_path)
    frontend.context_window = 1
    with pytest.raises(RequestError):
        frontend.prepare({"prompt": "hello", "max_tokens": 3}, False)
    assert engine.metadata_failure is None
    assert all(lease.closed for lease in engine.metadata_leases)


def test_final_render_credit_pressure_preserves_valid_reply(tmp_path):
    from tensorfold.engine.prefix_blocks import PrefixCapacity
    frontend, engine = app(tmp_path)
    original = engine.reserve_cache_metadata
    def reserve(nbytes):
        if engine.metadata_leases:
            raise PrefixCapacity("remaining byte grant is pinned")
        return original(nbytes)
    engine.reserve_cache_metadata = reserve
    result = frontend.run({"prompt": "hi", "max_tokens": 3}, False, lambda _: True)
    assert result["content"] == "abc" and not engine.published
    assert result["stats"]["cache_publication"]["status"] == "metadata_unfunded"
    assert engine.metadata_failure is None
    assert all(lease.closed for lease in engine.metadata_leases)


def test_media_metadata_lifetime_follows_original_claimed_owner():
    from types import SimpleNamespace
    events = []
    owner = object()
    media = SimpleNamespace(cache_metadata_owner=owner, cache_metadata_registration=None)
    prepared = server.PreparedRequest([1], 1, [], False, None, vision=media,
        _media_release=lambda _: False, _metadata_retire=lambda plan: events.append(plan))
    prepared.release_media()
    assert events == []
    prepared._media_release = lambda _: True
    prepared.release_media()
    prepared.release_media()
    assert events == [media] and prepared.vision is media


def test_failed_media_metadata_only_declares_completion_until_positive_retirement(tmp_path):
    frontend, engine = app(tmp_path)
    prepared = frontend.prepare({"prompt": "hi", "max_tokens": 3}, False)
    media, lease, registration = _native_style_media(frontend, engine, prepared)
    registration.failed = True
    prepared.release_media()
    assert registration.consumer_done and not lease.closed
    assert media.cache_metadata_owner.data is not None
    registration.retiring = True
    media.cache_metadata_owner.close()  # existing core owns positive terminal retirement
    assert lease.closed and media.cache_metadata_owner.data is None
    assert prepared.vision is media


def test_media_cleanup_notification_preserves_original_primary(tmp_path):
    frontend, engine = app(tmp_path)
    prepared = frontend.prepare({"prompt": "hi", "max_tokens": 3}, False)
    media, lease, registration = _native_style_media(frontend, engine, prepared)
    primary, notification, cause = RuntimeError("original replay close"), RuntimeError("notify"), ValueError("prior")
    primary.__cause__ = cause
    events = []
    def original_close(plan):
        raise primary
    def failed_notify(error, owner):
        assert error is primary and owner is media.cache_metadata_owner
        registration.failed = True
        events.append("notified")
        raise notification
    prepared._media_release, prepared._metadata_failed = original_close, failed_notify
    with pytest.raises(RuntimeError) as caught:
        prepared.release_media()
    assert caught.value is primary and events == ["notified"] and registration.consumer_done
    assert not lease.closed and media.cache_metadata_owner.data is not None
    assert any(error is cause for error in primary.__cause__.exceptions)
    assert any(error is notification for error in primary.__cause__.exceptions)


def test_settlement_timing_includes_canonical_publication(tmp_path):
    import time
    frontend, engine = app(tmp_path)
    publish = engine.publish_canonical
    def slow_publish(*args, **kwargs):
        time.sleep(0.01)
        return publish(*args, **kwargs)
    engine.publish_canonical = slow_publish
    result = frontend.run({"prompt": "hi", "max_tokens": 3}, False, lambda _: True)
    assert result["stats"]["frontend_settlement_s"] >= 0.01
    assert result["stats"]["request_wall_s"] >= result["stats"]["frontend_settlement_s"]


def test_unexpected_postprepare_check_failure_retains_credit_and_declares_cpu_done(tmp_path):
    frontend, engine = app(tmp_path)
    primary = MemoryError("postprepare check")
    def check(*args, **kwargs):
        raise primary
    frontend.check = check
    with pytest.raises(MemoryError) as caught:
        frontend.prepare({"prompt": "hi", "max_tokens": 3}, False)
    assert caught.value is primary and engine.metadata_failure[0] is primary
    owner = engine.metadata_failure[1]
    assert owner.registration.consumer_done
    assert any(not lease.closed for lease in engine.metadata_leases)
    owner.registration.retiring = True
    owner.close()
    assert all(lease.closed for lease in engine.metadata_leases)


def _native_style_media(frontend, engine, prepared):
    from types import SimpleNamespace
    from tensorfold.families.gemma4.cuda.vision import (
        PreparedGemmaVision, _CacheData, _CacheMetadata, rebind_prepared, retire_cache_metadata,
    )
    lease = engine.reserve_cache_metadata(1024)
    owner = _CacheMetadata(lease, 1024, engine.cache_metadata_failed)
    registration = engine.register_cache_metadata_owner(owner)
    owner.registration = registration
    owner.data = _CacheData((10, 11), (), (), (), "fixture", None)
    media = PreparedGemmaVision(tuple(prepared.prompt), (), (), (), (), _cache_metadata=owner, _cache_failure=None)
    frontend.vision = SimpleNamespace(
        canonical_tokens=lambda original, text: frontend.template.encode(frontend.tok, text),
        rebind_prepared=rebind_prepared, retire_cache_metadata=retire_cache_metadata)
    prepared.vision = media
    prepared._metadata_failed = engine.cache_metadata_failed
    prepared._metadata_retire = retire_cache_metadata
    assert prepared._media_release is None  # actual native Gemma eager ownership
    return media, lease, registration


def test_native_media_metadata_without_replay_callback_retires_after_success(tmp_path):
    frontend, engine = app(tmp_path)
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}
    prepared = frontend.prepare(body, True)
    media, lease, registration = _native_style_media(frontend, engine, prepared)
    frontend.run(body, True, lambda _: True, prepared=prepared)
    rebound = engine.published[0][1]
    assert rebound.cache_metadata_owner is media.cache_metadata_owner
    assert rebound.frames is media.frames and prepared.vision is media
    assert tuple(engine.published[0][0]) == rebound.token_ids
    assert lease.closed and registration.closed
    assert not media.cache_metadata_complete and not rebound.cache_metadata_complete
    assert media.cache_metadata_owner.data is None
    assert prepared._metadata_retire is prepared._metadata_failed is None


def test_native_media_failure_declares_cpu_done_and_retains_credit_until_terminal_close(tmp_path):
    frontend, engine = app(tmp_path)
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}
    prepared = frontend.prepare(body, True)
    media, lease, registration = _native_style_media(frontend, engine, prepared)
    primary = RuntimeError("native generation failed")
    engine.failure = primary
    with pytest.raises(RuntimeError) as caught:
        frontend.run(body, True, lambda _: True, prepared=prepared)
    assert caught.value is primary and registration.failed and registration.consumer_done
    assert not lease.closed and media.cache_metadata_owner.data is not None
    registration.retiring = True
    media.cache_metadata_owner.close()
    owner = engine.metadata_failure[1]
    owner.registration.retiring = True
    owner.close()
    assert all(credit.closed for credit in engine.metadata_leases)
    assert prepared.vision is media


@pytest.mark.parametrize("initially_unfunded", [False, True])
def test_stream_tool_metadata_pressure_keeps_generation_and_no_unpaid_capture(tmp_path, monkeypatch, initially_unfunded):
    from tensorfold.engine.prefix_blocks import PrefixCapacity
    from tests.test_cuda_server_disconnect import TOOLS
    text = "<tool_call>\n<function=measure>\n</function>\n</tool_call>\x00"
    frontend, engine = app(tmp_path, text)
    if initially_unfunded:
        def refuse(nbytes):
            raise PrefixCapacity("no initial snapshot credit")
        engine.reserve_cache_metadata = refuse
    else:
        original = server._CanonicalCredits.set
        attempts = []
        def set_credit(owner, role, amount, **kwargs):
            if role == "canonical-wire":
                attempts.append(amount)
                raise PrefixCapacity("no wire capture credit")
            return original(owner, role, amount, **kwargs)
        monkeypatch.setattr(server._CanonicalCredits, "set", set_credit)
    def unpaid(*args, **kwargs):
        raise AssertionError("unpaid wire copy")
    monkeypatch.setattr(server, "_accepted_tool_delta", unpaid)
    deltas = []
    result = frontend.run({"messages": [{"role": "user", "content": "measure"}],
        "tools": TOOLS, "stream": True, "max_tokens": len(text)}, True,
        lambda delta: deltas.append(delta) or True)
    assert result["finish"] == "tool_calls" and not engine.published
    assert result["stats"]["cache_publication"]["status"] == "metadata_unfunded"
    assert any(delta and delta.get("tool_calls") for delta in deltas)
    if not initially_unfunded:
        assert len(attempts) == 1  # all later fragments bypass optional capture
    assert engine.metadata_failure is None
    assert all(lease.closed for lease in engine.metadata_leases)


@pytest.mark.parametrize("stream", [False, True])
def test_tool_only_canonical_chat_uses_actual_wire_content_null(tmp_path, stream):
    from tests.test_cuda_server_disconnect import TOOLS
    text = "<tool_call>\n<function=measure>\n</function>\n</tool_call>\x00"
    frontend, engine = app(tmp_path, text)
    source = '{% for m in messages %}{% if m.role == "assistant" %}{% if m.content is none %}NULL{% elif m.content == "" %}EMPTY{% else %}TEXT{% endif %}{% endif %};{% endfor %}'
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": source}))
    frontend.template = server.ChatTemplate(tmp_path)
    render = frontend.template.render
    captured = []
    def observe_render(messages, **kwargs):
        if messages[-1]["role"] == "assistant":
            captured.append(messages[-1])
        return render(messages, **kwargs)
    frontend.template.render = observe_render
    result = frontend.run({"messages": [{"role": "user", "content": "measure"}],
        "tools": TOOLS, "stream": stream, "max_tokens": len(text)}, True, lambda _: True)
    assert (result["content"] or None) is None  # the actual HTTP wire projection
    assert captured[-1]["content"] is None
    canonical = frontend.tok.decode(engine.published[0][0], skip_special_tokens=False)
    # The shared authoritative normalizer maps wire null to lexical empty.
    assert "EMPTY" in canonical and "NULL" not in canonical
    expected = render([{"role": "user", "content": "measure"}, captured[-1]], tools=TOOLS,
                      enable_thinking=False, add_generation_prompt=False)
    assert engine.published[0][0] == frontend.template.encode(frontend.tok, expected)


@pytest.mark.parametrize("stream", [False, True])
def test_responses_canonical_history_matches_actual_wire_and_stored_next_turn(tmp_path, stream):
    from tests.test_cuda_server_disconnect import TOOLS, serving
    from tests.test_responses_api import call, stream_events
    from tensorfold.server.responses_translate import messages
    text = "<tool_call>\n<function=measure>\n</function>\n</tool_call>\x00"
    frontend, engine = app(tmp_path, text)
    source = '{% for m in messages %}{{m.role}}:{% if m.content is none %}NULL{% elif m.content == "" %}EMPTY{% else %}{{m.content}}{% endif %}{{m.tool_calls|default([])|tojson}};{% endfor %}{% if add_generation_prompt %}OPEN{% endif %}'
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": source}))
    frontend.template = server.ChatTemplate(tmp_path)
    body = {"model": "fake-cuda", "input": "measure", "stream": stream, "max_output_tokens": len(text),
            "tools": [{"type": "function", **TOOLS[0]["function"]}]}
    with serving(frontend) as port:
        status, raw = call(port, "POST", "/v1/responses", body)
        assert status == 200, raw
        response = stream_events(raw)[-1]["response"] if stream else json.loads(raw)
        converted = messages(response["output"])
        expected_render = frontend.template.render(
            [{"role": "user", "content": "measure"}, *converted], tools=TOOLS,
            enable_thinking=False, add_generation_prompt=False)
        expected_ids = frontend.template.encode(frontend.tok, expected_render)
        assert engine.published[-1][0] == expected_ids
        output_call = next(item for item in response["output"] if item["type"] == "function_call")
        assert output_call["call_id"] != output_call["id"]
        canonical = frontend.tok.decode(expected_ids, skip_special_tokens=False)
        assert output_call["call_id"] in canonical and output_call["id"] not in canonical
        assert "assistant:EMPTY" in canonical and "assistant:NULL" not in canonical
        engine.text = "z\x00"
        status, raw = call(port, "POST", "/v1/responses", {
            "model": "fake-cuda", "previous_response_id": response["id"],
            "input": [{"type": "function_call_output", "call_id": output_call["call_id"], "output": "ok"}],
            "max_output_tokens": 2})
        assert status == 200, raw
        assert engine.calls[-1]["prompt"][:len(expected_ids)] == expected_ids


def test_turn_failure_remains_primary_over_later_intent_cleanup_failure(tmp_path):
    frontend, engine = app(tmp_path)
    primary, secondary = RuntimeError("first turn release"), RuntimeError("later cancel")
    # An expected capacity decline leaves the intent for final cancellation.
    from tensorfold.engine.prefix_blocks import PrefixCapacity
    def refuse(nbytes):
        raise PrefixCapacity("no cache metadata room")
    engine.reserve_cache_metadata = refuse
    def give():
        raise primary
    def cancel(intent):
        raise secondary
    frontend._turns().give = give
    engine.cancel_cache_publication = cancel
    with pytest.raises(RuntimeError) as caught:
        frontend.run({"prompt": "hi", "max_tokens": 3}, False, lambda _: True)
    assert caught.value is primary and caught.value.__cause__ is secondary


def test_retired_prepared_still_generates_without_raw_postpass_or_unpaid_snapshot(tmp_path):
    frontend, engine = app(tmp_path)
    body = {"prompt": "hi", "max_tokens": 3}
    prepared = frontend.prepare(body, False)
    prompt, sampling, vision, binding = prepared.prompt, prepared.sampling, prepared.vision, prepared.profile_binding
    prepared.retire_canonical_cache()
    prepared.retire_canonical_cache()
    assert prepared.canonical is None and all(lease.closed for lease in engine.metadata_leases)
    assert prepared.prompt is prompt and prepared.sampling is sampling
    assert prepared.vision is vision and prepared.profile_binding is binding
    result = frontend.run(body, False, lambda _: True, prepared=prepared)
    assert result["content"] == "abc" and not engine.published
    assert result["stats"]["cache_publication"]["status"] == "canonical_metadata_retired"
    assert len(engine.intents) == len(engine.cancelled) == 1


def test_prepared_reuse_preserves_generation_and_explicit_nonpublication(tmp_path):
    frontend, engine = app(tmp_path)
    body = {"prompt": "hi", "max_tokens": 3}
    prepared = frontend.prepare(body, False)
    first = frontend.run(body, False, lambda _: True, prepared=prepared)
    second = frontend.run(body, False, lambda _: True, prepared=prepared)
    assert first["content"] == second["content"] == "abc"
    assert len(engine.published) == 1 and len(engine.intents) == 2
    assert engine.intents[0][1] is not engine.intents[1][1]
    assert second["stats"]["cache_publication"]["status"] == "canonical_metadata_retired"


def test_queued_reuse_cannot_borrow_active_runs_canonical_credit(tmp_path):
    import threading
    from tests.test_cuda_server_disconnect import until
    frontend, engine = app(tmp_path)
    engine.hold_at = 1
    body = {"prompt": "hi", "max_tokens": 3}
    prepared = frontend.prepare(body, False)
    results, failures = [], []
    def run():
        try:
            results.append(frontend.run(body, False, lambda _: True, prepared=prepared))
        except BaseException as error:
            failures.append(error)
    first, second = threading.Thread(target=run), threading.Thread(target=run)
    first.start()
    try:
        assert engine.held.wait(10)
        second.start()
        until(lambda: frontend._turns().parked == 1, "reused Prepared to wait behind active request")
    finally:
        engine.release.set()
        first.join(10)
        if second.ident is not None:
            second.join(10)
    assert not first.is_alive() and not second.is_alive() and not failures
    assert len(results) == 2 and all(result["content"] == "abc" for result in results)
    assert len(engine.published) == 1 and len(engine.intents) == 2
    assert sorted(result["stats"]["cache_publication"].get("status", "complete") for result in results) == [
        "canonical_metadata_retired", "complete"]


def test_unfunded_raw_ids_keep_raw_intent_mode(tmp_path):
    from tensorfold.engine.prefix_blocks import PrefixCapacity
    frontend, engine = app(tmp_path)
    def refuse(nbytes):
        raise PrefixCapacity("pinned cache metadata")
    engine.reserve_cache_metadata = refuse
    result = frontend.run({"prompt": [65, 66], "max_tokens": 3}, False, lambda _: True)
    assert engine.intents[0][0] == "raw_ids"
    assert result["stats"]["cache_publication"]["status"] == "metadata_unfunded"
    assert result["content"] == "abc" and not engine.published


def test_incomplete_tool_output_has_explicit_nonpublication_outcome(tmp_path):
    from tests.test_cuda_server_disconnect import TOOLS
    text = "<tool_call>\n<function=measure>\n"
    frontend, engine = app(tmp_path, text)
    result = frontend.run({"messages": [{"role": "user", "content": "measure"}],
        "tools": TOOLS, "stream": True, "max_tokens": len(text)}, True, lambda _: True)
    assert result["finish"] == "length" and not engine.published
    assert result["stats"]["cache_publication"]["status"] == "incomplete_tool_call"
    assert len(engine.cancelled) == 1 and all(lease.closed for lease in engine.metadata_leases)


def test_completed_history_template_refusal_preserves_valid_generation(tmp_path):
    frontend, engine = app(tmp_path)
    source = '{% if not add_generation_prompt and messages[-1].role == "assistant" %}{{raise_exception("completed history unsupported")}}{% endif %}{% for m in messages %}{{m.content}};{% endfor %}'
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": source}))
    frontend.template = server.ChatTemplate(tmp_path)
    result = frontend.run({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
                          True, lambda _: True)
    assert result["content"] == "abc" and not engine.published
    publication = result["stats"]["cache_publication"]
    assert publication["status"] == "canonical_template_refused"
    assert publication["error_type"] == "TemplateError" and not publication["complete"]
    assert engine.metadata_failure is None and all(lease.closed for lease in engine.metadata_leases)


def test_template_refusal_cannot_hide_chained_resource_failure(tmp_path):
    from jinja2.exceptions import TemplateError
    frontend, engine = app(tmp_path)
    render = frontend.template.render
    primary, resource = TemplateError("wrapped resource"), MemoryError("native allocation")
    def wrapped(messages, **kwargs):
        if kwargs.get("add_generation_prompt") is False:
            raise primary from resource
        return render(messages, **kwargs)
    frontend.template.render = wrapped
    with pytest.raises(TemplateError) as caught:
        frontend.run({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
                     True, lambda _: True)
    assert caught.value is primary and caught.value.__cause__ is resource
    assert engine.metadata_failure[0] is primary and not engine.published


def test_contained_template_error_frames_retire_before_metadata_refund(tmp_path):
    from jinja2.exceptions import TemplateError
    frontend, engine = app(tmp_path)
    render, reserve = frontend.template.render, engine.reserve_cache_metadata
    refused = []
    def wrapped(messages, **kwargs):
        if kwargs.get("add_generation_prompt") is False:
            error = TemplateError("unsupported canonical render")
            refused.append(error)
            raise error
        return render(messages, **kwargs)
    def reserved(nbytes):
        lease = reserve(nbytes)
        close = lease.close
        def checked_close():
            if refused:
                assert refused[0].__traceback__ is None
            close()
        lease.close = checked_close
        return lease
    frontend.template.render, engine.reserve_cache_metadata = wrapped, reserved
    result = frontend.run({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
                          True, lambda _: True)
    assert result["stats"]["cache_publication"]["status"] == "canonical_template_refused"
    assert refused[0].__traceback__ is None
    assert all(lease.closed for lease in engine.metadata_leases)


def test_internal_prepare_failure_has_prepaid_cleanup_and_original_primary(tmp_path, monkeypatch):
    frontend, engine = app(tmp_path)
    primary = MemoryError("internal snapshot serialization")
    def fail(*args, **kwargs):
        raise primary
    monkeypatch.setattr(server.json, "dumps", fail)
    with pytest.raises(MemoryError) as caught:
        frontend.run({"prompt": "hi", "max_tokens": 3}, False, lambda _: True)
    assert caught.value is primary
    assert engine.metadata_failure[0] is primary
    assert engine.metadata_failure[1].registration.consumer_done


@pytest.mark.parametrize("capability", [False, None, 1])
def test_inherited_methods_without_exact_archive_capability_keep_legacy_generation(tmp_path, capability):
    frontend, engine = app(tmp_path)
    engine.supports_canonical_prefix_archive = capability
    body = {"prompt": "hi", "max_tokens": 3}
    prepared = frontend.prepare(body, False)
    assert prepared.canonical is None and not engine.registrations
    result = frontend.run(body, False, lambda _: True, prepared=prepared)
    assert result["content"] == "abc"
    assert not engine.intents and not engine.published and "cache_publication" not in result["stats"]


def test_contained_capacity_frames_retire_before_metadata_credit_refund(tmp_path):
    from tensorfold.engine.prefix_blocks import PrefixCapacity
    frontend, engine = app(tmp_path)
    reserve = engine.reserve_cache_metadata
    refusal = PrefixCapacity("pinned future metadata")
    def reserved(nbytes):
        if engine.metadata_leases:
            raise refusal
        lease = reserve(nbytes)
        close = lease.close
        def checked_close():
            assert refusal.__traceback__ is None
            close()
        lease.close = checked_close
        return lease
    engine.reserve_cache_metadata = reserved
    result = frontend.run({"prompt": "hi", "max_tokens": 3}, False, lambda _: True)
    assert result["content"] == "abc"
    assert result["stats"]["cache_publication"]["status"] == "metadata_unfunded"
    assert refusal.__traceback__ is None and all(lease.closed for lease in engine.metadata_leases)


def test_refusal_frame_cleanup_failure_preserves_authentic_error_and_prior_cause(monkeypatch):
    from tensorfold.engine.prefix_blocks import PrefixCapacity
    primary, prior, cleanup = PrefixCapacity("owning refusal"), ValueError("prior cause"), MemoryError("retirement")
    primary.__cause__ = prior
    primary.add_note("keep original note")
    def fail(trace):
        raise cleanup
    monkeypatch.setattr(server, "clear_frames", fail)
    with pytest.raises(PrefixCapacity) as caught:
        server._retire_cache_refusal(primary)
    assert caught.value is primary and "keep original note" in primary.__notes__
    assert any(error is prior for error in primary.__cause__.exceptions)
    assert any(error is cleanup for error in primary.__cause__.exceptions)


def test_metadata_close_failure_is_sticky_and_never_retries_raw_credit(tmp_path):
    frontend, engine = app(tmp_path)
    prepared = frontend.prepare({"prompt": "hi", "max_tokens": 3}, False)
    owner = prepared.canonical.credit
    primary = RuntimeError("uncertain metadata close")
    attempts = []
    def fail_close():
        attempts.append("close")
        raise primary
    engine.metadata_leases[0].close = fail_close
    with pytest.raises(RuntimeError) as caught:
        prepared.retire_canonical_cache()
    assert caught.value is primary and owner.registration.close_failed
    with pytest.raises(RuntimeError) as again:
        owner.close()
    assert again.value is primary and attempts == ["close"]


def test_legacy_native_media_without_optional_owner_is_unchanged():
    from types import SimpleNamespace
    legacy = SimpleNamespace(token_ids=(1, 2), frames=("existing-frame",))
    def forbidden(plan):
        raise AssertionError("MoE optional helper invoked on legacy media")
    prepared = server.PreparedRequest([1, 2], 1, [], False, None, vision=legacy, _metadata_retire=forbidden)
    prepared.release_media()
    prepared.retire_canonical_cache()
    assert prepared.vision is legacy and prepared.vision.frames == ("existing-frame",)


def test_media_prepared_reuse_keeps_original_generation_plan(tmp_path):
    frontend, engine = app(tmp_path)
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}
    prepared = frontend.prepare(body, True)
    media, lease, registration = _native_style_media(frontend, engine, prepared)
    first = frontend.run(body, True, lambda _: True, prepared=prepared)
    assert prepared.vision is media and lease.closed
    second = frontend.run(body, True, lambda _: True, prepared=prepared)
    assert first["content"] == second["content"] == "abc"
    assert prepared.vision is media and media.token_ids == tuple(prepared.prompt)
    assert second["stats"]["cache_publication"]["status"] == "canonical_metadata_retired"


@pytest.mark.parametrize("value", [type("Literal", (str,), {})("literal"), ("tuple",), {1: "integer-key"}])
def test_non_json_internal_render_values_preserve_generation_without_snapshot_coercion(tmp_path, value):
    frontend, engine = app(tmp_path)
    body = {"messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"unused": value}, "max_tokens": 3}
    result = frontend.run(body, True, lambda _: True)
    assert result["content"] == "abc" and not engine.published
    assert result["stats"]["cache_publication"]["status"] == "snapshot_representation_unsupported"
    assert engine.metadata_failure is None


def test_expected_initial_validation_error_drops_snapshot_data_before_refund(tmp_path, monkeypatch):
    from tensorfold.server.errors import RequestError
    frontend, engine = app(tmp_path)
    captured = []
    capture = server._CanonicalInput.capture.__func__
    def observe(cls, *args, **kwargs):
        result = capture(cls, *args, **kwargs)
        captured.append(result)
        return result
    monkeypatch.setattr(server._CanonicalInput, "capture", classmethod(observe))
    with pytest.raises(RequestError) as caught:
        frontend.prepare({"messages": [{"role": "invalid-role", "content": "private" * 1000}],
                          "max_tokens": 3}, True)
    assert caught.value.__traceback__ is not None  # authentic caller diagnostic remains
    assert captured and captured[0].retired
    assert captured[0].payload == "" and captured[0].projection is None
    assert captured[0].template is captured[0].tokenizer is captured[0].credit is None
    assert all(lease.closed for lease in engine.metadata_leases)
    assert engine.metadata_failure is None


def test_retirement_invalidates_private_snapshot_alias_before_credit_close(tmp_path):
    frontend, engine = app(tmp_path)
    prepared = frontend.prepare({"prompt": "original", "max_tokens": 3}, False)
    alias = prepared.canonical
    assert alias.payload and not alias.retired
    close = engine.metadata_leases[0].close
    def checked_close():
        assert alias.retired and alias.payload == ""
        assert alias.template is alias.tokenizer is alias.credit is None
        close()
    engine.metadata_leases[0].close = checked_close
    prepared.retire_canonical_cache()
    assert prepared.canonical is None and prepared.prompt
    with pytest.raises(RuntimeError, match="retired"):
        alias.encode(frontend, prepared, "", None, [])


def test_nonstream_tool_uses_final_authority_without_wire_capture_credit(tmp_path, monkeypatch):
    from tensorfold.engine.prefix_blocks import PrefixCapacity
    from tests.test_cuda_server_disconnect import TOOLS
    frontend, engine = app(tmp_path, "<tool_call>\n<function=measure>\n</function>\n</tool_call>\x00")
    original = server._CanonicalCredits.set
    unused = []
    def credit(owner, role, amount, **kwargs):
        if role == "canonical-wire":
            unused.append(amount)
            raise PrefixCapacity("unused stream-only role")
        return original(owner, role, amount, **kwargs)
    monkeypatch.setattr(server._CanonicalCredits, "set", credit)
    result = frontend.run({"messages": [{"role": "user", "content": "measure"}],
        "tools": TOOLS, "stream": False, "max_tokens": 56}, True, lambda _: True)
    assert result["finish"] == "tool_calls" and result["calls"]
    assert not unused and len(engine.published) == 1
    assert result["stats"]["cache_publication"]["complete"]
