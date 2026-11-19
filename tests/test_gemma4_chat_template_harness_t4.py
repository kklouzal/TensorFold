"""GB10's Gemma 4 template with T4 (no GPU, no weights): ``deploy/gb10/gemma4-chat-template-harness-20261119-t4.jinja``.

T4: with thinking on, the generation prompt after a user turn opens the thought channel, as Google's template already
does after a tool result. Every other render equals ``gemma4-chat-template-harness-20261119.jinja`` (T1-T3, tested in
``test_gemma4_chat_template_harness.py``); Hugging Face's ``render_jinja_template`` is the independent oracle.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from tensorfold.server.gemma_values import THOUGHT_CLOSE, THOUGHT_OPEN, prompt_opens_thought

from tests.test_gemma4_chat_template import BOS, CONVERSATIONS, ROOT, WEATHER, _as_template_input, _post, _tool_turn
from tests.test_gemma4_chat_template_harness import EXTRA, TEMPLATE as T3

pytest.importorskip("jinja2")

TEMPLATE = ROOT / "deploy" / "gb10" / "gemma4-chat-template-harness-20261119-t4.jinja"
SHA256 = "c9bfe97c9d9e90beef33adfeaea4b38ec7e6b0410c3427cca6f82874fe950539"
CASES = {**CONVERSATIONS, **EXTRA}
OPEN = "<|turn>model\n" + THOUGHT_OPEN + "\n"


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
def t3(model_dir):
    from tensorfold.cuda.chat_template import prepare_chat_template
    return prepare_chat_template(model_dir, template_file=T3)


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


def test_t4_thinking_on_after_a_user_turn_opens_the_thought_channel(ours):
    for case in ("system_tools_user", "plain_multi_turn", "multi_turn_after_tools"):
        messages, tools = CONVERSATIONS[case]
        text = ours.render(messages, tools=tools, enable_thinking=True)
        assert text.endswith(OPEN) and prompt_opens_thought(text)


def test_everything_else_renders_as_before(ours, t3):
    for case, (messages, tools) in CASES.items():
        assert ours.render(messages, tools=tools, enable_thinking=False) == t3.render(
            messages, tools=tools, enable_thinking=False), case
        before = t3.render(messages, tools=tools, enable_thinking=True)
        after = ours.render(messages, tools=tools, enable_thinking=True)
        # Only the generation prompt after a user turn gains the opened channel.
        assert after == before or (before.endswith("<|turn>model\n") and after == before + THOUGHT_OPEN + "\n"), case


def test_strict_discovery_selects_json_null(ours):
    from tensorfold.server import strict_gemma_tools as gemma
    rendered = ours.render(gemma.PROBE, tools=gemma.PROBE_TOOLS, enable_thinking=False)
    rendered_false = ours.render(gemma.false_probe(), tools=gemma.PROBE_TOOLS, enable_thinking=False)
    assert gemma.template_null(rendered, rendered_false) == "null"


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_thinking_on_reply_after_user_starts_in_the_opened_channel(tmp_path, stream):
    from tests.test_gemma_api_contract import app
    from tensorfold.cuda.chat_template import ChatTemplate
    result = app(tmp_path, "Short.\n" + THOUGHT_CLOSE + "Hello.")
    result.template = ChatTemplate(tmp_path, template_file=TEMPLATE)
    reasoning, content, _, _ = _post(result, {"messages": [{"role": "user", "content": "Hi"}], "stream": stream,
                                              "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": True}})
    assert result.tok.decode(result.engine.calls[0]["prompt"]).endswith(OPEN)
    assert (reasoning, content) == ("Short.\n", "Hello.")


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_thinking_off_is_unchanged(tmp_path, stream):
    from tests.test_gemma_api_contract import app
    from tensorfold.cuda.chat_template import ChatTemplate
    result = app(tmp_path, "It is 15 C.")
    result.template = ChatTemplate(tmp_path, template_file=TEMPLATE)
    reasoning, content, _, _ = _post(result, {"messages": _tool_turn(reasoning=None), "tools": [WEATHER],
                                              "stream": stream, "max_tokens": 200})
    assert (reasoning, content) == ("", "It is 15 C.")
