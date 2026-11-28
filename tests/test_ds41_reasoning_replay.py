"""DeepSeek-V4.1 replay of earlier reasoning: accepted field names and recall of reasoning a client dropped."""

import json
import os
from pathlib import Path

import pytest

from tensorfold.families.deepseek_v4.serving_v41 import ReasoningMemory

MODEL = Path(os.environ.get("DS41_MODEL_DIR", "/home/kklouzal/models/ds41-tf"))
TOOLS = [{"type": "function", "function": {"name": "run_shell", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}}}}]
THOUGHT = "I should list the files first to see the layout."
CALL = {"id": "call_1", "type": "function", "function": {"name": "run_shell", "arguments": "{\"command\": \"ls\"}"}}


def history(system="You are an agent.", user="What does this project do?"):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def turn(**fields):
    return {"role": "assistant", "content": "", "tool_calls": [dict(CALL)], **fields}


def follow(messages, assistant):
    return [*messages, assistant, {"role": "tool", "tool_call_id": "call_1", "content": "README.md\nsrc\n"}]


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning", "thinking"])
def test_accepted_reasoning_fields(field):
    out = ReasoningMemory().restore(follow(history(), turn(**{field: THOUGHT})))
    assert out[2]["reasoning_content"] == THOUGHT


def test_dropped_reasoning_is_recalled_for_the_same_conversation_only():
    memory = ReasoningMemory()
    memory.remember(history(), {"role": "assistant", "content": None, "tool_calls": [CALL]}, THOUGHT)
    # the client drops the reasoning, renumbers the call and re-serializes its arguments
    replay = turn()
    replay["tool_calls"] = [{"id": "x9", "type": "function",
                             "function": {"name": "run_shell", "arguments": json.dumps({"command": "ls"}, indent=2)}}]
    assert memory.restore(follow(history(), replay))[2]["reasoning_content"] == THOUGHT
    # an identical call in another conversation recalls nothing
    other = memory.restore(follow(history(user="Something else entirely."), turn()))
    assert "reasoning_content" not in other[2]
    # a client's own reasoning wins over the remembered one
    own = memory.restore(follow(history(), turn(reasoning_content="mine")))
    assert own[2]["reasoning_content"] == "mine"
    # an empty field counts as dropped
    assert memory.restore(follow(history(), turn(reasoning_content="")))[2]["reasoning_content"] == THOUGHT


def test_memory_is_bounded_least_recently_used_first():
    memory = ReasoningMemory(entries=2, characters=1 << 20)
    for i in range(3):
        memory.remember(history(user=f"q{i}"), {"role": "assistant", "content": "a", "tool_calls": None}, f"t{i}")
    recall = lambda i: memory.restore([*history(user=f"q{i}"), {"role": "assistant", "content": "a"}])[2]
    assert "reasoning_content" not in recall(0)
    assert recall(1)["reasoning_content"] == "t1" and recall(2)["reasoning_content"] == "t2"
    small = ReasoningMemory(entries=10, characters=5)
    small.remember(history(), {"role": "assistant", "content": "a"}, "toolong")
    assert "reasoning_content" not in small.restore([*history(), {"role": "assistant", "content": "a"}])[2]


@pytest.mark.skipif(not (MODEL / "tokenizer.json").exists(), reason="DeepSeek-V4.1 tokenizer not available")
def test_template_renders_aliased_and_recalled_reasoning():
    pytest.importorskip("tokenizers")
    from tensorfold.families.deepseek_v4.serving_v41 import V41ChatTemplate

    template = V41ChatTemplate(MODEL)
    render = lambda messages: str(template.render(messages, tools=TOOLS, enable_thinking=True))
    full = render(follow(history(), turn(reasoning_content=THOUGHT)))
    assert THOUGHT in full
    assert render(follow(history(), turn(reasoning=THOUGHT))) == full
    assert render(follow(history(), turn(thinking=THOUGHT))) == full
    assert THOUGHT not in render(follow(history(), turn()))
    template.remember_reply(history(), "", [CALL], THOUGHT)
    assert render(follow(history(), turn())) == full
