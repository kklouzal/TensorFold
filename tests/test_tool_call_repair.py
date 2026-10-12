"""Qwen tool-call envelopes a model writes slightly off are still calls, streamed and final alike.

The model sometimes doubles a ``<tool_call>`` opener or closer, writes two ``<function=…>`` blocks in one envelope,
or ends without the final ``</tool_call>``. Each was returned as text (or merged into one call with mixed arguments),
so an agent client saw no call. These cases check the repair at every layer: the shared normalizer, both end
parsers, the streaming delta writer, the streamed content filter and the HTTP replies.
"""

import itertools
import json

import pytest

pytest.importorskip("jinja2")

from tensorfold.cuda.reply_text import hide_tool_calls, parse_tool_calls
from tensorfold.engine.tool_draft import ToolCallStreamer
from tensorfold.server import text as mac_text
from tensorfold.server.tools import normalize_qwen_tool_markup, parse_tool_calls_from_content
from tests.test_cuda_admission import http_server
from tests.test_cuda_server_errors import EOS, HI, app_for, events, request

TOOLS = [
    {"type": "function", "function": {"name": "bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["path"]}}},
]


def function(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<function={name}>\n{body}</function>"


CALL = f"<tool_call>\n{function('bash', command='ls -la')}\n</tool_call>"
LS = [("bash", {"command": "ls -la"})]
CASES = {
    "normal": ("Running it.\n\n" + CALL, LS),
    "two_calls": (CALL + "\n" + CALL.replace("ls -la", "pwd"), LS + [("bash", {"command": "pwd"})]),
    "repeated_opener": ("Running it.\n\n<tool_call>\n" + CALL, LS),
    "tripled_opener": ("<tool_call>\n<tool_call>\n" + CALL, LS),
    "repeated_closer": (CALL + "\n</tool_call>", LS),
    "missing_final_closer": (CALL[: -len("</tool_call>")], LS),
    "two_functions_one_envelope": (
        f"<tool_call>\n{function('bash', command='ls')}\n{function('read_file', path='/x', limit='5')}\n</tool_call>",
        [("bash", {"command": "ls"}), ("read_file", {"path": "/x", "limit": 5})]),
}
UNCHANGED = {   # text that is not a complete call stays text
    "prose_mentions_markup": "Use `<tool_call>` tags like this: <tool_call> ... then </tool_call>.",
    "truncated_mid_value": "<tool_call>\n<function=bash>\n<parameter=command>\nls -",
    "prose_between_unclosed_functions": (f"<tool_call>\n{function('bash', command='ls')}\nthen\n"
                                         f"{function('bash', command='pwd')}"),
}


def calls_of(calls):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls or []]


def assembled(parts):
    """(name, arguments) of each call streamed as OpenAI ``tool_calls`` delta parts, in index order."""

    calls = {}
    for part in parts:
        entry = calls.setdefault(part["index"], {"name": None, "arguments": ""})
        entry["name"] = part["function"].get("name") or entry["name"]
        entry["arguments"] += part["function"].get("arguments", "")
    return [(v["name"], json.loads(v["arguments"])) for _, v in sorted(calls.items())]


@pytest.mark.parametrize("name", sorted(CASES))
def test_both_end_parsers_read_the_intended_calls(name):
    text, expected = CASES[name]
    for parse in (parse_tool_calls, parse_tool_calls_from_content):
        content, calls = parse(text, TOOLS)
        assert calls_of(calls) == expected, parse.__module__
        assert "tool_call" not in content and "<function=" not in content


@pytest.mark.parametrize("name", sorted(UNCHANGED))
def test_text_that_is_not_a_complete_call_stays_text(name):
    text = UNCHANGED[name]
    assert normalize_qwen_tool_markup(text) == text
    content, calls = parse_tool_calls(text, TOOLS)
    assert calls is None and content == text.strip()


def test_the_normalizer_is_idempotent_and_closes_the_last_envelope():
    for text, _ in CASES.values():
        once = normalize_qwen_tool_markup(text)
        assert normalize_qwen_tool_markup(once) == once
    open_call = CASES["missing_final_closer"][0]
    assert normalize_qwen_tool_markup(open_call).endswith("</tool_call>")
    # lower() lengthens U+0130: the closer still lands after the last opener of the text itself
    for before in ("\u0130", "\u0130" * 3 + " <tool_call>\n" + function("bash", command="ls") + "\n</tool_call>\n"):
        content, calls = parse_tool_calls(before + open_call, TOOLS)
        assert calls_of(calls)[-1] == LS[0] and "tool_call" not in content


@pytest.mark.parametrize("name", sorted(CASES))
def test_streamed_deltas_equal_the_end_parsers_calls(name):
    text, expected = CASES[name]
    streamer = ToolCallStreamer(TOOLS)
    deltas = []
    for n in range(1, len(text) + 1):                  # one character at a time: every split point
        deltas += streamer.feed(text[:n])
    assert assembled(delta["tool_calls"][0] for delta in deltas) == expected


@pytest.mark.parametrize("hide", [hide_tool_calls, mac_text.hide_tool_calls], ids=["cuda", "mac"])
def test_a_doubled_closer_after_a_call_is_never_streamed_as_content(hide):
    text = CASES["repeated_closer"][0] + "\nDone."
    shown = [hide(text[:n], finished=False) for n in range(1, len(text) + 1)]
    assert all("</" not in s for s in shown)
    assert all(b.startswith(a) for a, b in itertools.pairwise(shown))   # visible text only grows
    assert hide(text, finished=True) == "\nDone."
    for parse in (parse_tool_calls, parse_tool_calls_from_content):   # streamed text agrees with the end parsers
        assert parse(text, TOOLS)[0] == "Done."


class Reply:
    """A fake engine whose reply is fixed text (the stand-in tokenizer's ids are characters)."""

    eos = (EOS,)

    def __init__(self, text):
        self.text = text

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        ids = [ord(c) for c in self.text]
        for i in range(0, len(ids), 3):                 # several rounds, as decoding gives
            on_tokens(ids[i:i + 3])
        on_tokens([EOS])
        return {"generated": len(ids) + 1}


def reply_events(tmp_path, text, **body):
    app = app_for(tmp_path)
    app.engine = Reply(text)
    with http_server(app) as port:
        streamed = events(request(port, {"messages": HI, "tools": TOOLS, "stream": True, **body})[2])
        status, _, plain = request(port, {"messages": HI, "tools": TOOLS, **body})
    assert status == 200
    return streamed, json.loads(plain)["choices"][0]


def stream_summary(frames):
    content, parts, finish = "", [], None
    for frame in frames:
        if frame == "[DONE]":
            continue
        choice = frame["choices"][0] if frame.get("choices") else {}
        delta = choice.get("delta", {})
        content += delta.get("content") or ""
        parts += delta.get("tool_calls") or []
        finish = choice.get("finish_reason") or finish
    return content, assembled(parts), finish


@pytest.mark.parametrize("name", ["repeated_opener", "repeated_closer", "missing_final_closer",
                                  "two_functions_one_envelope"])
def test_http_streamed_and_plain_replies_agree_on_the_calls(tmp_path, name):
    text, expected = CASES[name]
    streamed, plain = reply_events(tmp_path, text)
    content, calls, finish = stream_summary(streamed)
    assert (calls, finish) == (expected, "tool_calls")
    assert calls_of(plain["message"]["tool_calls"]) == expected and plain["finish_reason"] == "tool_calls"
    assert "tool_call" not in content and "tool_call" not in (plain["message"].get("content") or "")


def test_a_one_call_request_shows_an_unparseable_call_instead_of_an_empty_reply(tmp_path):
    text = f"<tool_call>\n{function('delete_everything', path='/')}\n</tool_call>"   # not an offered tool
    streamed, plain = reply_events(tmp_path, text, parallel_tool_calls=False)
    content, calls, finish = stream_summary(streamed)
    assert calls == [] and finish == "stop" and "delete_everything" in content
    assert "delete_everything" in plain["message"]["content"] and not plain["message"].get("tool_calls")


def test_a_held_back_tail_after_leading_spaces_is_streamed(tmp_path):
    streamed, _ = reply_events(tmp_path, "  See <")
    content, _, finish = stream_summary(streamed)
    assert content == "  See <" and finish == "stop"
