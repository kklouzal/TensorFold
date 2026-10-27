"""New tool definitions require text without changing historical call parsing."""

import json

import pytest

from tensorfold.server.messages import _normalize_tool_call_arguments
from tensorfold.server.tools import active_tool_specs, normalize_tool_specs, parse_tool_calls_from_content
from tests.test_cuda_tool_choice import Engine, app_for, ask


def definition(name, wrapped):
    function = {"name": name, "parameters": {"type": "object", "properties": {}}}
    return {"type": "function", "function": function} if wrapped else function


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("name", [123, 1.5, True, ["\x7f" * 128], {"nested": "name"}])
def test_new_input_definitions_refuse_nonstring_names_without_repr(wrapped, name):
    with pytest.raises(ValueError, match="tools\\[0\\].*function name"):
        active_tool_specs([definition(name, wrapped)], "required")


def test_definition_validation_never_calls_a_foreign_name_string_conversion():
    class Foreign:
        def __str__(self):
            raise AssertionError("untrusted name coercion")

    with pytest.raises(ValueError, match="function name"):
        normalize_tool_specs([definition(Foreign(), True)])


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("name", ["天气_" * 100, "x" * 4096, "  legacy-name  "])
def test_existing_string_extensions_keep_their_original_objects_and_fields(wrapped, name):
    tool = definition(name, wrapped)
    result = active_tool_specs([tool], "required")
    assert result == [tool] and result[0] is tool
    actual = tool["function"]["name"] if wrapped else tool["name"]
    assert actual == name


@pytest.mark.parametrize("stream", [False, True])
def test_http_rejects_new_invalid_name_before_generation_or_stream_headers(tmp_path, stream):
    engine = Engine()
    status, raw = ask(app_for(tmp_path, engine), stream=stream,
                      tools=[definition(["not-a-name"], True)])
    body = json.loads(raw)
    assert status == 400 and body["error"]["type"] == "invalid_request_error"
    assert "function name" in body["error"]["message"] and engine.calls == []


def test_permissive_historical_output_name_conversion_and_bad_arguments_replay_survive():
    tools = [definition("123", True)]
    content, calls = parse_tool_calls_from_content('<tool_call>{"name":123,"arguments":{"x":1}}</tool_call>', tools)
    assert not content and calls[0]["function"]["name"] == "123"
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}
    history = [{"role": "assistant", "tool_calls": [{"id": "old", "type": "function",
                "function": {"name": 123, "arguments": "malformed historical arguments"}}]}]
    replay = _normalize_tool_call_arguments(history)
    assert replay[0]["tool_calls"][0]["function"] == {
        "name": 123, "arguments": {"_invalid_arguments": "malformed historical arguments"}}
    assert history[0]["tool_calls"][0]["function"]["arguments"] == "malformed historical arguments"
