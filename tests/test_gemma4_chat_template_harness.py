"""The GB10 Gemma 4 template derived from Google's 4d7ae498 template (no GPU, no weights).

``deploy/gb10/gemma4-chat-template-harness-20261119.jinja`` changes three things, listed in its header:
T1 thinking off, the generation after a tool result is prompted with the empty thought channel, as after a user turn;
T2 thinking off, current-turn model messages re-render that channel where it was prompted; T3 OpenAI form, a model
message's text precedes its tool calls and a turn ending in tool results stays open. Everything else must render as
Google's template does; Hugging Face's ``render_jinja_template`` is the independent oracle for the server render.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from tensorfold.server.gemma_values import THOUGHT_CLOSE, THOUGHT_OPEN, prompt_opens_thought

from tests.test_gemma4_chat_template import (
    BOS, CONVERSATIONS, ROOT, TEMPLATE as GOOGLE, WEATHER, _as_template_input, _call, _post, _tool_turn,
)

pytest.importorskip("jinja2")

NAME = "gemma4-chat-template-harness-20261119.jinja"
TEMPLATE = ROOT / "deploy" / "gb10" / NAME
SHA256 = "c59098e2887e6739a3a64aa74fd2dd5e2e98ab3fbf27937a34e0421f2f4fed33"
EMPTY = THOUGHT_OPEN + "\n" + THOUGHT_CLOSE
QUOTE = '<|"|>'

# A Harness-style opening (prompt, then a status frame); the server merges leading system messages, so cases compared
# with the Hugging Face oracle use ORACLE_PREAMBLE.
PREAMBLE = [{"role": "system", "content": "You are terse."}, {"role": "system", "content": "FRAME"},
            {"role": "user", "content": "Weather in London and Oslo?"}]
ORACLE_PREAMBLE = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Weather in London and Oslo?"}]

EXTRA = {
    "text_then_call": ([*ORACLE_PREAMBLE, {"role": "assistant", "content": "Let me check.",
                                    "tool_calls": [_call("c1", "get_weather", "London")]},
                        {"role": "tool", "tool_call_id": "c1", "content": "15 C"}], [WEATHER]),
    "two_steps": ([*ORACLE_PREAMBLE, {"role": "assistant", "content": None, "tool_calls": [_call("c1", "get_weather", "London")]},
                   {"role": "tool", "tool_call_id": "c1", "content": "15 C"},
                   {"role": "assistant", "content": None, "tool_calls": [_call("c2", "get_weather", "Oslo")]},
                   {"role": "tool", "tool_call_id": "c2", "content": "2 C"}], [WEATHER]),
    "native_answer": ([{"role": "user", "content": "Weather in London?"},
                       {"role": "assistant", "content": "It is 15 C.", "tool_calls": [_call("c1", "get_weather", "London")],
                        "tool_responses": [{"name": "get_weather", "response": "15 C"}]},
                       {"role": "user", "content": "Thanks"}], [WEATHER]),
}
CASES = {**CONVERSATIONS, **EXTRA}


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"bos_token": BOS, "eos_token": "<eos>"}))
    (tmp_path / "chat_template.jinja").write_text("CHECKPOINT TEMPLATE {{ messages }}")
    return tmp_path


@pytest.fixture
def ours(model_dir):
    from tensorfold.cuda.chat_template import prepare_chat_template
    return prepare_chat_template(model_dir, template_file=TEMPLATE)


@pytest.fixture
def google(model_dir):
    from tensorfold.cuda.chat_template import prepare_chat_template
    return prepare_chat_template(model_dir, template_file=GOOGLE)


def _raw_call(city):
    return f"<|tool_call>call:get_weather{{city:{QUOTE}{city}{QUOTE}}}<tool_call|>"


def test_pinned_bytes_and_selection(ours):
    assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == SHA256
    expected = {"name": "override", "source": str(TEMPLATE), "sha256": SHA256}
    assert ours.provenance(tools=False) == ours.provenance(tools=True) == expected


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("case", sorted(CASES))
def test_server_render_equals_hugging_face_render(ours, case, thinking):
    utils = pytest.importorskip("transformers.utils.chat_template_utils")
    messages, tools = CASES[case]
    rendered, _ = utils.render_jinja_template([_as_template_input(messages)], tools=tools,
                                              chat_template=TEMPLATE.read_text(), add_generation_prompt=True,
                                              bos_token=BOS, eos_token="<eos>", enable_thinking=thinking)
    assert ours.render(messages, tools=tools, enable_thinking=thinking) == rendered[0]


def test_t1_generation_after_a_tool_result_is_prompted_with_the_empty_channel(ours):
    text = ours.render(_tool_turn(reasoning=None), tools=[WEATHER], enable_thinking=False)
    assert text.endswith("<tool_response|>" + EMPTY) and not prompt_opens_thought(text)
    on = ours.render(_tool_turn(reasoning=None), tools=[WEATHER], enable_thinking=True)
    assert on.endswith("<tool_response|>" + THOUGHT_OPEN + "\n") and prompt_opens_thought(on)


def test_t2_each_step_extends_the_prompt_it_was_generated_from(ours):
    """Thinking off: prompt + the model's raw output is a prefix of the next step's prompt, so nothing re-renders."""
    messages = list(PREAMBLE)
    for index, (text, city) in enumerate([("", "London"), ("Let me check Oslo.", "Oslo"), ("", "Paris")]):
        before = ours.render(messages, tools=[WEATHER], enable_thinking=False)
        raw = text + _raw_call(city) + "<|tool_response>"
        messages += [{"role": "assistant", "content": text or None,
                      "tool_calls": [_call(f"c{index}", "get_weather", city)]},
                     {"role": "tool", "tool_call_id": f"c{index}", "content": "15 C"}]
        after = ours.render(messages, tools=[WEATHER], enable_thinking=False)
        assert after.startswith(before + raw)
        assert after.endswith("<tool_response|>" + EMPTY)


def test_t2_earlier_turns_render_as_google_renders_them(ours, google):
    for case in ("multi_turn_after_tools", "plain_multi_turn", "system_tools_user"):
        messages, tools = CONVERSATIONS[case]
        assert ours.render(messages, tools=tools, enable_thinking=False) == google.render(
            messages, tools=tools, enable_thinking=False)


def test_t2_current_reasoning_renders_once(ours):
    text = ours.render(_tool_turn(), tools=[WEATHER], enable_thinking=False)
    assert "<|turn>model\n" + THOUGHT_OPEN + "\nI need the weather tool.\n" + THOUGHT_CLOSE + "<|tool_call>" in text
    assert text.count(THOUGHT_OPEN) == 2   # the replayed thought and the T1 prompt


def test_t3_text_precedes_its_calls_and_the_turn_stays_open(ours):
    messages, tools = EXTRA["text_then_call"]
    for thinking in (False, True):
        text = ours.render(messages, tools=tools, enable_thinking=thinking)
        assert "Let me check." + _raw_call("London") + "<|tool_response>" in text
        assert "<turn|>" not in text.split("<|turn>user\nWeather in London and Oslo?<turn|>\n")[1]


def test_thinking_on_is_googles_render_except_t3(ours, google):
    for case, (messages, tools) in CASES.items():
        if case != "text_then_call":
            assert ours.render(messages, tools=tools, enable_thinking=True) == google.render(
                messages, tools=tools, enable_thinking=True), case


def test_native_form_keeps_googles_answer_placement(ours, google):
    messages, tools = EXTRA["native_answer"]
    for thinking in (False, True):
        assert ours.render(messages, tools=tools, enable_thinking=thinking) == google.render(
            messages, tools=tools, enable_thinking=thinking)
    last = messages[:2]
    off = ours.render(last, tools=tools, enable_thinking=False)
    assert off.endswith("<tool_response|>It is 15 C.<turn|>\n")   # a closed turn is not prompted (T1 needs an open one)


def test_strict_discovery_selects_json_null(ours):
    from tensorfold.server import strict_gemma_tools as gemma
    rendered = ours.render(gemma.PROBE, tools=gemma.PROBE_TOOLS, enable_thinking=False)
    rendered_false = ours.render(gemma.false_probe(), tools=gemma.PROBE_TOOLS, enable_thinking=False)
    assert gemma.template_null(rendered, rendered_false) == "null"


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_thinking_off_reply_after_tool_result_is_content(tmp_path, stream):
    from tests.test_gemma_api_contract import app
    from tensorfold.cuda.chat_template import ChatTemplate
    result = app(tmp_path, "It is 15 C.")
    result.template = ChatTemplate(tmp_path, template_file=TEMPLATE)
    reasoning, content, _, _ = _post(result, {"messages": _tool_turn(reasoning=None), "tools": [WEATHER],
                                              "stream": stream, "max_tokens": 200})
    assert result.tok.decode(result.engine.calls[0]["prompt"]).endswith("<tool_response|>" + EMPTY)
    assert (reasoning, content) == ("", "It is 15 C.")
