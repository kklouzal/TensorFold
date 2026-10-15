"""A reply that ended at an end token inside its think block: one warning line, and with
TENSORFOLD_FAILURE_CAPTURE_DIR set its request (images redacted), token ids and stats in a bounded directory. Nothing
for a normal reply, and the reply itself is never changed. CPU only."""

from __future__ import annotations

import json
import os
import stat

import pytest

pytest.importorskip("jinja2")

from tensorfold.server import failure_capture, thinking_notes
from tests.test_cuda_admission import http_server, post
from tests.test_cuda_tool_choice import END, EOS, Tokens, app_for

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


class Named(Tokens):
    def id_to_token(self, token):
        return {EOS: "<|im_end|>", END: "</think>"}.get(token, chr(token))


class Thinker:
    """Thinks, then (``closes``) writes an answer, or ends at its end token with the think block still open."""

    eos = (EOS,)

    def __init__(self, closes):
        self.closes = closes

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        reply = [ord(c) for c in "let me think"]
        reply += [END, ord("4")] if self.closes else []
        on_tokens(reply + [EOS])
        return {"rounds": 1, "decode_s": 0.1}


def ask(tmp_path, closes, *, stream=False, images=False):
    app = app_for(tmp_path, Thinker(closes))
    app.tok = Named()
    content = [{"type": "text", "text": "What is 2+2?"}, IMAGE] if images else "What is 2+2?"
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 64, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": True}}
    if images:                     # the stand-in renders text only: drop the image before the template sees it
        app.prepare = lambda b, chat: type(app).prepare(app, {**b, "messages": [{"role": "user", "content": "x"}]},
                                                       chat)
    with http_server(app) as port:
        return post(port, body, True)


def test_the_detection_needs_every_part_of_the_failure():
    yes = dict(finish="stop", thinking=True, content="", calls=None, reasoning="hmm", end_token=True, closed=False)
    assert thinking_notes.ended_thinking(**yes)
    for change in ({"finish": "length"}, {"thinking": False}, {"content": "4"}, {"calls": [{"id": "c"}]},
                   {"reasoning": " "}, {"end_token": False}, {"closed": True}):
        assert not thinking_notes.ended_thinking(**{**yes, **change}), change


@pytest.mark.parametrize("stream", [False, True])
def test_an_open_think_block_at_the_end_token_warns_once_and_captures_nothing_unasked(tmp_path, capsys, stream,
                                                                                     monkeypatch):
    monkeypatch.delenv(failure_capture.ENV, raising=False)
    status, body = ask(tmp_path, closes=False, stream=stream)
    output = capsys.readouterr().out
    assert status == 200 and output.count("inside its think block") == 1
    assert "end token <|im_end|> (id 0)" in output and "reply=13 tokens" in output
    assert "failure captured" not in output
    if not stream:                                       # the reply is what the model wrote: no answer, its reasoning
        choice = json.loads(body)["choices"][0]
        assert choice["finish_reason"] == "stop" and choice["message"]["content"] is None
        assert choice["message"]["reasoning_content"] == "let me think"


def test_a_closed_think_block_is_a_normal_reply(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(failure_capture.ENV, str(tmp_path / "captures"))
    status, body = ask(tmp_path, closes=True)
    assert status == 200 and json.loads(body)["choices"][0]["message"]["content"] == "4"
    assert "think block" not in capsys.readouterr().out and not (tmp_path / "captures").exists()


def test_the_capture_holds_the_redacted_request_its_tokens_and_stats(tmp_path, capsys, monkeypatch):
    folder = tmp_path / "captures"
    monkeypatch.setenv(failure_capture.ENV, str(folder))
    status, body = ask(tmp_path, closes=False, images=True)
    assert status == 200
    files = list(folder.glob("*" + failure_capture.SUFFIX))
    assert len(files) == 1 and stat.S_IMODE(os.stat(files[0]).st_mode) == 0o600
    record = json.loads(files[0].read_text())
    assert record["kind"] == "ended_in_think" and record["finish"] == "stop"
    assert record["end_token"] == {"id": EOS, "text": "<|im_end|>"}
    assert record["token_ids"] == [ord(c) for c in "let me think"] + [EOS]
    assert record["stats"]["request_id"] == record["request_id"] and record["stats"]["stats_version"] == 2
    assert record["body"]["messages"][0]["content"][1] == {"type": "image_url", "image_url": {"url": "<redacted>"}}
    assert "AAAA" not in files[0].read_text()
    assert f"failure captured: {files[0]}" in capsys.readouterr().out


def test_a_full_directory_skips_a_capture_rather_than_growing(tmp_path, capsys, monkeypatch):
    folder = tmp_path / "captures"
    monkeypatch.setenv(failure_capture.ENV, str(folder))
    monkeypatch.setattr(failure_capture, "MAX_FILES", 2)
    for _ in range(3):
        failure_capture.capture("ended_in_think", "req-x", {"messages": []}, {"token_ids": [1]})
    assert len(list(folder.glob("*" + failure_capture.SUFFIX))) == 2
    assert "failure capture skipped for req-x" in capsys.readouterr().out
    monkeypatch.setattr(failure_capture, "MAX_FILES", 100)
    monkeypatch.setattr(failure_capture, "MAX_BYTES", 10)
    assert failure_capture.capture("ended_in_think", "req-y", {"messages": []}, {"token_ids": [1]}) is None
    assert not list(folder.glob(".capture-*"))                              # no temporary left behind


def test_a_failed_write_prints_why_and_never_raises(tmp_path, capsys, monkeypatch):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    monkeypatch.setenv(failure_capture.ENV, str(blocker / "captures"))
    assert failure_capture.capture("ended_in_think", "req-z", {}, {}) is None
    assert "failure capture for req-z" in capsys.readouterr().out


def test_the_request_log_redaction_covers_videos_too():
    body = {"messages": [{"role": "user", "content": [IMAGE, {"type": "video_url", "video_url": {"url": "x.mp4"}},
                                                      {"type": "text", "text": "keep"}]}]}
    redacted = failure_capture.redact_images(body)["messages"][0]["content"]
    assert redacted[0]["image_url"] == redacted[1]["video_url"] == {"url": "<redacted>"}
    assert redacted[2] == {"type": "text", "text": "keep"} and body["messages"][0]["content"][0] == IMAGE
