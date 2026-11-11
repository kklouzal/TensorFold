"""Canonical Responses history follows actual item order, IDs and finish semantics."""
from copy import deepcopy

import pytest

from tensorfold.server import responses_translate as rt
from tensorfold.server.errors import CapacityError, RequestError
from tensorfold.server.messages import normalize_messages


CALL = {"id": "call_actual", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":1}"}}


@pytest.mark.parametrize("message", [None, {}, {"content": None}, {"content": ""},
    {"content": None, "reasoning_content": ""}, {"content": None, "reasoning_content": "think"},
    {"content": "answer"}, {"tool_calls": [CALL]},
    {"content": None, "reasoning_content": "think", "tool_calls": [CALL]},
    {"content": "answer", "reasoning_content": "think", "tool_calls": [CALL]}])
@pytest.mark.parametrize("finish", [None, "stop", "length"])
def test_nonstream_projection_matches_actual_reply_stored_output(message, finish):
    reply = rt.Reply({"id": "resp_fixture"})
    final = reply.completion({"choices": [{"message": message, "finish_reason": finish}]})
    before = deepcopy(message)
    result = rt.canonical_output_messages(message, finish)
    assert result == rt.messages(final["output"])
    assert message == before


def test_completed_empty_answer_rule_is_shared_and_length_remains_empty():
    assert rt.needs_empty_output_message([], "stop") is True
    assert rt.needs_empty_output_message([], None) is True
    assert rt.needs_empty_output_message([], "length") is False
    assert rt.needs_empty_output_message([{"type": "reasoning"}], "stop") is True
    assert rt.needs_empty_output_message([{"type": "function_call"}], "stop") is False
    for finish, expected in (("stop", ["message"]), ("length", [])):
        reply = rt.Reply({"id": "resp_empty"})
        final = reply.finish(finish, None)
        assert [item["type"] for item in final["output"]] == expected
        assert rt.canonical_output_messages({}, finish) == rt.messages(final["output"])


def interleaved_reply():
    reply = rt.Reply({"id": "resp_interleaved", "previous_response_id": None})
    reply.delta({"reasoning_content": "first thought"})
    reply.delta({"content": "first answer"})
    reply.delta({"reasoning_content": "later thought"})
    reply.delta({"tool_calls": [{"index": 0, "id": "call_actual", "function": {"name": "lookup", "arguments": "{"}}]})
    reply.delta({"tool_calls": [{"index": 0, "function": {"arguments": "\"q\":1}"}}]})
    reply.delta({"content": "later answer"})
    return reply


def test_streaming_uses_actual_interleaved_items_and_call_id_without_uuid(monkeypatch):
    reply = interleaved_reply()
    existing = deepcopy(reply.items)
    def forbidden(*_):
        raise AssertionError("canonical projection generated an ID")
    monkeypatch.setattr(rt, "_id", forbidden)
    projected = rt.canonical_output_messages({"content": "flattened text cannot preserve order"}, "stop",
                                             output_items=reply.items)
    assert reply.items == existing
    assert [m.get("reasoning_content") for m in projected] == ["first thought", "later thought", None]
    assert projected[1]["tool_calls"][0]["id"] == "call_actual"
    tool_item = next(item for item in reply.items if item["type"] == "function_call")
    assert tool_item["id"] != tool_item["call_id"]
    assert projected[1]["tool_calls"][0]["id"] != tool_item["id"]
    assert projected == rt.messages(reply.items)
    final = reply.finish("stop", None)
    assert projected == rt.canonical_output_messages({}, "stop", output_items=final["output"])


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_streaming_pre_and_post_finish_projection_are_identical(finish):
    reply = rt.Reply({"id": "resp_reasoning"})
    reply.delta({"reasoning_content": "only thought"})
    before = deepcopy(reply.items)
    projected = rt.canonical_output_messages({}, finish, output_items=reply.items)
    assert reply.items == before  # Synthetic empty answer never mutates actual output.
    final = reply.finish(finish, None)
    assert projected == rt.messages(final["output"])
    assert projected == rt.canonical_output_messages({}, finish, output_items=reply.items)


def test_projection_is_funded_over_existing_source_before_derived_messages(monkeypatch):
    reply = interleaved_reply()
    events = []
    class Credit:
        def render_input(self, source):
            assert source is reply.items
            events.append("credit")
    original = rt.messages
    def convert(items):
        assert events == ["credit"]
        events.append("messages")
        return original(items)
    monkeypatch.setattr(rt, "messages", convert)
    rt.canonical_output_messages({}, "stop", output_items=reply.items, credit=Credit())
    assert events == ["credit", "messages"]
    class Decline:
        def render_input(self, source):
            raise CapacityError("optional projection unfunded")
    with pytest.raises(CapacityError, match="unfunded"):
        rt.canonical_output_messages({}, "stop", output_items=reply.items, credit=Decline())
    assert events == ["credit", "messages"]


def test_actual_missing_call_id_refuses_instead_of_uuid_fallback(monkeypatch):
    def forbidden(*_):
        raise AssertionError("UUID fallback")
    monkeypatch.setattr(rt, "_id", forbidden)
    with pytest.raises(RequestError, match="actual call IDs"):
        rt.canonical_output_messages({"tool_calls": [{"function": {"name": "f", "arguments": ""}}]}, "stop")
    call = {"id": "call_empty", "function": {"name": "f", "arguments": ""}}
    result = rt.canonical_output_messages({"tool_calls": [call]}, "length")
    assert result[0]["tool_calls"][0]["function"]["arguments"] == ""


def test_stored_history_and_durable_normalizer_roundtrip_keep_identical_chat(tmp_path):
    from tensorfold.server.responses import Store
    from tensorfold.server.api_records import ApiRecordStore
    from test_api_records import limits
    reply = interleaved_reply()
    final = reply.finish("stop", None)
    projected = rt.canonical_output_messages({}, "stop", output_items=reply.items)
    added = [{"role": "user", "content": "question"}]
    store = Store()
    store.put(final, added)
    assert normalize_messages(store.conversation(final["id"])) == normalize_messages([*added, *projected])
    durable = ApiRecordStore(tmp_path / "history.sqlite", limits=limits())
    try:
        actual = durable.normalize_items("alice", final["output"])
        assert normalize_messages(rt.messages(actual)) == normalize_messages(projected)
        followup = [{"type": "function_call_output", "call_id": "call_actual", "output": "tool result"}]
        request = rt.translate({"input": followup, "previous_response_id": final["id"]}, store)
        assert normalize_messages(request.chat["messages"]) == normalize_messages([*added, *projected,
            {"role": "tool", "tool_call_id": "call_actual", "content": "tool result"}])
    finally:
        durable.close()
