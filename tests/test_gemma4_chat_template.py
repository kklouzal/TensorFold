"""Google's current Gemma 4 chat template as the GB10 Gemma4 profile serves it (no GPU, no weights).

The profile passes ``deploy/gb10/gemma4-chat-template-google-4d7ae498.jinja`` with ``--chat-template``: the bytes of
``chat_template.jinja`` at google/gemma-4-26B-A4B-it revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 (last changed
by 35b4173cf6211bf5ee1f4c3c8d97cf2a0d89c122). Renders are compared with Hugging Face's own ``render_jinja_template``
(the apply_chat_template path) as an independent oracle; the server's reply parsing is checked on the prompts this
template ends with.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tensorfold.server.gemma_values import (
    GEMMA_NULLS, GemmaJSONStream, THOUGHT_CLOSE, THOUGHT_OPEN, prompt_opens_thought, split_gemma_thinking,
)

pytest.importorskip("jinja2")

ROOT = Path(__file__).resolve().parents[1]
NAME = "gemma4-chat-template-google-4d7ae498.jinja"
TEMPLATE = ROOT / "deploy" / "gb10" / NAME
SHA256 = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
BOS = "<bos>"

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
CLOCK = {"type": "function", "function": {
    "name": "get_time", "description": "Local time for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def _call(id_, name, city):
    # API form: arguments are a JSON string, as OpenAI clients send them
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps({"city": city})}}


def _tool_turn(reasoning="I need the weather tool."):
    return [{"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Weather in London?"},
            {"role": "assistant", "content": "", "reasoning_content": reasoning,
             "tool_calls": [_call("call_1", "get_weather", "London")]},
            {"role": "tool", "tool_call_id": "call_1", "content": "15 C, sunny"}]


def _as_template_input(messages):
    """The messages the template contract expects, built independently of the server's normalization."""
    out = []
    for message in messages:
        message = dict(message)
        if isinstance(message.get("content"), list):
            message["content"] = "".join(part["text"] for part in message["content"])
        if message.get("tool_calls"):
            message["tool_calls"] = [{**call, "function": {**call["function"],
                                                           "arguments": json.loads(call["function"]["arguments"])}}
                                     for call in message["tool_calls"]]
        out.append(message)
    return out


CONVERSATIONS = {
    "system_tools_user": ([{"role": "system", "content": "You are terse."},
                           {"role": "user", "content": "Weather in London?"}], [WEATHER]),
    "tool_call_and_response": (_tool_turn(), [WEATHER]),
    "parallel_calls_named_by_id": ([
        {"role": "user", "content": "Weather and time in Oslo?"},
        {"role": "assistant", "content": None, "reasoning_content": "Two tools.",
         "tool_calls": [_call("a", "get_weather", "Oslo"), _call("b", "get_time", "Oslo")]},
        {"role": "tool", "tool_call_id": "a", "content": "2 C"},
        {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "09:30"}]}], [WEATHER, CLOCK]),
    "multi_turn_after_tools": ([
        *_tool_turn(reasoning="OLD-REASONING"),
        {"role": "assistant", "content": "It is 15 C and sunny.", "reasoning_content": "OLD-ANSWER-REASONING"},
        {"role": "user", "content": "And tomorrow?"}], [WEATHER]),
    "plain_multi_turn": ([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
                          {"role": "user", "content": "Bye"}], None),
}


@pytest.fixture
def model_dir(tmp_path):
    # The checkpoint's own template must not participate once the explicit file is selected.
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"bos_token": BOS, "eos_token": "<eos>"}))
    (tmp_path / "chat_template.jinja").write_text("CHECKPOINT TEMPLATE {{ messages }}")
    return tmp_path


@pytest.fixture
def template(model_dir):
    from tensorfold.cuda.chat_template import prepare_chat_template
    return prepare_chat_template(model_dir, template_file=TEMPLATE)


def test_shipped_bytes_are_the_pinned_google_template_and_the_profile_selects_them():
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == SHA256
    compose = (ROOT / "deploy" / "gb10" / "compose.gemma4.yaml").read_text()
    yaml = pytest.importorskip("yaml")
    service = yaml.safe_load(compose)["services"]["tensorfold"]
    command = service["command"]
    target = command[command.index("--chat-template") + 1]
    mounts = [v for v in service["volumes"] if v.get("target") == target]
    assert target == "/templates/" + NAME and command.count("--chat-template") == 1
    assert mounts == [{"type": "bind", "source": "./" + NAME, "target": target, "read_only": True}]


def test_explicit_file_is_the_only_selected_source_for_both_routes(template):
    expected = {"name": "override", "source": str(TEMPLATE), "sha256": SHA256}
    assert template.provenance(tools=False) == template.provenance(tools=True) == expected
    assert "CHECKPOINT TEMPLATE" not in template.render([{"role": "user", "content": "x"}], tools=None,
                                                       enable_thinking=False)


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("case", sorted(CONVERSATIONS))
def test_server_render_equals_hugging_face_render(template, case, thinking):
    utils = pytest.importorskip("transformers.utils.chat_template_utils")
    messages, tools = CONVERSATIONS[case]
    ours = template.render(messages, tools=tools, enable_thinking=thinking)
    rendered, _ = utils.render_jinja_template([_as_template_input(messages)], tools=tools,
                                              chat_template=TEMPLATE.read_text(), add_generation_prompt=True,
                                              bos_token=BOS, eos_token="<eos>", enable_thinking=thinking)
    assert ours == rendered[0]


def test_trained_formats_tool_turn_with_preserved_thought(template):
    off = template.render(_tool_turn(), tools=[WEATHER], enable_thinking=False)
    on = template.render(_tool_turn(), tools=[WEATHER], enable_thinking=True)
    assert "<|turn>tool" not in off + on
    assert off.startswith(BOS + "<|turn>system\nYou are terse.<|tool>declaration:get_weather{")
    assert on.startswith(BOS + "<|turn>system\n<|think|>\nYou are terse.<|tool>declaration:get_weather{")
    turn = ('<|turn>model\n<|channel>thought\nI need the weather tool.\n<channel|>'
            '<|tool_call>call:get_weather{city:<|"|>London<|"|>}<tool_call|>'
            '<|tool_response>response:get_weather{value:<|"|>15 C, sunny<|"|>}<tool_response|>')
    # The tool result continues the model turn; the generation prompt opens no new turn.
    assert off.endswith("<turn|>\n" + turn)
    assert on.endswith("<turn|>\n" + turn + THOUGHT_OPEN + "\n")
    assert not prompt_opens_thought(off) and prompt_opens_thought(on)


def test_parallel_results_are_named_by_tool_call_id(template):
    messages, tools = CONVERSATIONS["parallel_calls_named_by_id"]
    text = template.render(messages, tools=tools, enable_thinking=False)
    assert text.endswith('<|tool_response>response:get_weather{value:<|"|>2 C<|"|>}<tool_response|>'
                         '<|tool_response>response:get_time{value:<|"|>09:30<|"|>}<tool_response|>')
    assert "<|channel>thought\nTwo tools.\n<channel|><|tool_call>call:get_weather" in text


def test_generation_prompts_after_a_user_turn(template):
    messages, tools = CONVERSATIONS["system_tools_user"]
    off = template.render(messages, tools=tools, enable_thinking=False)
    on = template.render(messages, tools=tools, enable_thinking=True)
    assert off.endswith("Weather in London?<turn|>\n<|turn>model\n" + THOUGHT_OPEN + "\n" + THOUGHT_CLOSE)
    assert on.endswith("Weather in London?<turn|>\n<|turn>model\n")
    assert not prompt_opens_thought(off) and not prompt_opens_thought(on)


def test_earlier_turn_thoughts_drop_unless_preserved(template):
    messages, tools = CONVERSATIONS["multi_turn_after_tools"]
    text = template.render(messages, tools=tools, enable_thinking=False)
    assert "OLD-REASONING" not in text and "OLD-ANSWER-REASONING" not in text
    # the final answer continues the tool-call model turn and closes it once
    assert ('<tool_response|>It is 15 C and sunny.<turn|>\n<|turn>user\nAnd tomorrow?<turn|>\n'
            '<|turn>model\n<|channel>thought\n<channel|>') in text
    kept = template.render(messages, tools=tools, enable_thinking=False, extra={"preserve_thinking": True})
    assert "<|channel>thought\nOLD-REASONING\n<channel|><|tool_call>" in kept and "OLD-ANSWER-REASONING" not in kept


def test_prompt_suffix_alone_opens_the_reply_channel():
    assert prompt_opens_thought("x<tool_response|>" + THOUGHT_OPEN + "\n")
    assert not prompt_opens_thought("x" + THOUGHT_OPEN + "\n" + THOUGHT_CLOSE)
    assert not prompt_opens_thought("<|turn>user\n" + THOUGHT_OPEN + "\n<turn|>\n<|turn>model\n")


def test_reply_inside_a_prompt_opened_channel():
    reply = "Weighing it.\n" + THOUGHT_CLOSE + "It is 15 C."
    assert split_gemma_thinking(reply, finished=True, opened=True) == ("Weighing it.\n", "It is 15 C.")
    assert split_gemma_thinking(reply, finished=True) == ("", reply)      # the defect a prompt-opened channel needs
    assert split_gemma_thinking("Weighing it.\n<chan", finished=False, opened=True) == ("Weighing it.\n", "")
    assert split_gemma_thinking("Out of tokens", finished=True, opened=True) == ("Out of tokens", "")
    call = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
    assert split_gemma_thinking("More.\n" + THOUGHT_CLOSE + call, finished=True, opened=True) == ("More.\n", call)
    assert split_gemma_thinking("More.\n" + call, finished=True, opened=True) == ("More.\n", call)


@pytest.mark.parametrize("null", GEMMA_NULLS)
def test_strict_value_stream_reads_only_the_templates_null_word(null):
    other = next(word for word in GEMMA_NULLS if word != null)
    assert GemmaJSONStream(null=null).feed("{a:" + null + ",b:[" + null + "]}", finished=True) == '{"a":null,"b":[null]}'
    with pytest.raises(ValueError, match="spells null as " + null):
        GemmaJSONStream(null=null).feed("{a:" + other + "}", finished=True)
    with pytest.raises(ValueError):
        GemmaJSONStream(null="NULL")


def test_strict_discovery_selects_json_null_for_this_template(template):
    from tensorfold.server import strict_gemma_tools as gemma
    rendered = template.render(gemma.PROBE, tools=gemma.PROBE_TOOLS, enable_thinking=False)
    rendered_false = template.render(gemma.false_probe(), tools=gemma.PROBE_TOOLS, enable_thinking=False)
    assert gemma.template_null(rendered, rendered_false) == "null"


# -- the CUDA server on this template (a scripted engine and toy tokenizer) ------------------------------------------

def _app(tmp_path, script, grammars=None):
    from tests.test_gemma_api_contract import app
    from tensorfold.cuda.chat_template import ChatTemplate

    tmp_path.mkdir(parents=True, exist_ok=True)
    result = app(tmp_path, script, grammars)
    result.template = ChatTemplate(tmp_path, template_file=TEMPLATE)
    return result


def _post(result, body):
    """(reasoning, content, [(name, arguments)], finish_reason) of one request, streamed or not."""
    from tests.test_cuda_admission import http_server, post
    from tests.test_strict_tools import _reply
    stream = bool(body.get("stream"))
    with http_server(result) as port:
        status, response = post(port, body, True)
    assert status == 200, response
    content, calls, finish = _reply(stream, response)
    if stream:
        parts = [choice["delta"] for line in response.splitlines()
                 if line.startswith("data: ") and line != "data: [DONE]"
                 for choice in json.loads(line[6:]).get("choices", [])]
    else:
        parts = [json.loads(response)["choices"][0]["message"]]
    return "".join(part.get("reasoning_content") or "" for part in parts), content or "", calls, finish


def _prompt_text(result):
    return result.tok.decode(result.engine.calls[0]["prompt"])


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_thinking_reply_after_tool_result_starts_in_the_opened_channel(tmp_path, stream):
    result = _app(tmp_path, "The tool says 15 C.\n" + THOUGHT_CLOSE + "It is 15 C and sunny in London.")
    body = {"messages": _tool_turn(), "tools": [WEATHER], "stream": stream, "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": True}}
    reasoning, content, calls, finish = _post(result, body)
    assert _prompt_text(result).endswith("<tool_response|>" + THOUGHT_OPEN + "\n")
    assert (reasoning, content, calls, finish) == ("The tool says 15 C.\n", "It is 15 C and sunny in London.", [],
                                                   "stop")


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_next_call_from_the_opened_channel(tmp_path, stream):
    call = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
    result = _app(tmp_path, "Paris too.\n" + THOUGHT_CLOSE + call)
    body = {"messages": _tool_turn(), "tools": [WEATHER], "stream": stream, "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": True}}
    reasoning, content, calls, finish = _post(result, body)
    assert reasoning == "Paris too.\n" and not content.strip() and finish == "tool_calls"
    assert [(name, json.loads(arguments)) for name, arguments in calls] == [("get_weather", {"city": "Paris"})]


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_thinking_off_reply_after_tool_result_and_thinking_on_after_user(tmp_path, stream):
    result = _app(tmp_path / "off", "It is 15 C.")
    reasoning, content, _, _ = _post(result, {"messages": _tool_turn(), "tools": [WEATHER], "stream": stream,
                                              "max_tokens": 200})
    assert _prompt_text(result).endswith("<tool_response|>") and (reasoning, content) == ("", "It is 15 C.")
    result = _app(tmp_path / "on", THOUGHT_OPEN + "\nShort.\n" + THOUGHT_CLOSE + "Hello.")
    reasoning, content, _, _ = _post(result, {"messages": [{"role": "user", "content": "Hi"}], "stream": stream,
                                              "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": True}})
    assert _prompt_text(result).endswith("<|turn>model\n") and (reasoning, content) == ("Short.\n", "Hello.")


def _strict_grammars():
    xgr = pytest.importorskip("xgrammar")
    pytest.importorskip("torch")
    from tests.test_gemma_api_contract import GemmaTokenizer
    from tests.test_strict_tools import STOP, V
    from tensorfold.engine import grammar

    return grammar.Grammars(xgr.TokenizerInfo(GemmaTokenizer.words, xgr.VocabType.RAW, vocab_size=V,
                                              stop_token_ids=[STOP]))


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_strict_call_from_the_opened_channel(tmp_path, stream):
    # A strict reply reads its answer through the server's strict split; the prompt opened its thought channel.
    from tests.test_strict_tools import tool
    result = _app(tmp_path, "Paris too.\n" + THOUGHT_CLOSE + '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}'
                  "<tool_call|>", _strict_grammars())
    body = {"messages": _tool_turn(), "tools": [tool("get_weather", WEATHER["function"]["parameters"])],
            "tool_choice": "required", "stream": stream, "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": True}}
    reasoning, content, calls, finish = _post(result, body)
    assert _prompt_text(result).endswith("<tool_response|>" + THOUGHT_OPEN + "\n")
    assert (reasoning, content, finish) == ("Paris too.\n", "", "tool_calls")
    assert [(name, json.loads(arguments)) for name, arguments in calls] == [("get_weather", {"city": "Paris"})]


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_strict_call_uses_this_templates_null(tmp_path, stream):
    from tests.test_strict_tools import tool

    grammars = _strict_grammars()
    schema = {"type": "object", "properties": {"enabled": {"type": "boolean"},
                                               "note": {"anyOf": [{"type": "null"}, {"type": "string"}]}},
              "required": ["enabled", "note"], "additionalProperties": False}
    result = _app(tmp_path, "<|tool_call>call:set_flag{enabled:true,note:null}<tool_call|>", grammars)
    body = {"messages": [{"role": "user", "content": "Enable it."}], "stream": stream, "max_tokens": 200,
            "tools": [tool("set_flag", schema)], "tool_choice": "required"}
    _, content, calls, finish = _post(result, body)
    assert not content and finish == "tool_calls"
    assert [(name, json.loads(arguments)) for name, arguments in calls] == [("set_flag", {"enabled": True, "note": None})]
    plan = result._strict_markup[0]
    assert plan.keywords == {"null": "null"}
    compiled = grammars.compile(plan([tool("set_flag", schema)], required=True, parallel=False).spec)
    good = "<|tool_call>call:set_flag{enabled:true,note:null}<tool_call|>"
    assert grammars.accepts(compiled, good) and not grammars.accepts(compiled, good.replace("null", "None"))
