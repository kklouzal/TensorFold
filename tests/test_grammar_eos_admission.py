"""Incompatible EOS suppression is refused before constrained generation or SSE headers on both HTTP backends."""

import json

import pytest

from tensorfold.engine import grammar
from tensorfold.server.errors import RequestError
from tensorfold.server.request_options import RequestOptions
from tests.http_fakes import post as mac_post
from tests.test_cuda_admission import http_server, post
from tests.test_server_openai_compat import FakeApp
from tests.test_strict_tools import ScriptEngine, TemplateTokenizer, QWEN_CALLS, _app, grammars, tool


CONTROLS = [
    {"response_format": {"type": "json_object"}},
    {"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}, "strict": True}}},
    {"guided_json": {"type": "object"}},
    {"guided_regex": "a+"},
    {"guided_choice": ["a", "b"]},
    {"guided_grammar": 'root ::= "a"'},
    {"structured_outputs": {"json_object": True}},
    {"tools": [tool("strict")]},
]


class MacApp(FakeApp, RequestOptions):
    accepts_sampling = True
    model_dir = None
    stop_ids = (0,)

    def __init__(self):
        super().__init__()
        self.tokenizer = TemplateTokenizer(QWEN_CALLS)
        self.calls = []

    def chat(self, messages, *, sampling=None, **kwargs):
        self.calls.append(sampling)
        return super().chat(messages, **kwargs)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("control", CONTROLS)
def test_cuda_prepare_and_http_reject_eos_suppression_before_engine(tmp_path, grammars, stream, control):
    engine = ScriptEngine("a")
    app = _app(tmp_path, engine, grammars)
    body = {"messages": [{"role": "user", "content": "Hi"}], "ignore_eos": True, "stream": stream, **control}
    with pytest.raises(RequestError, match="ignore_eos true cannot be combined"):
        app.prepare(body, True)
    with http_server(app) as port:
        status, text = post(port, body, True)
    assert status == 400
    assert "ignore_eos true cannot be combined" in json.loads(text)["error"]["message"]
    assert not text.startswith("data:") and not engine.calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("control", CONTROLS)
def test_mac_http_rejects_eos_suppression_before_chat_or_stream(stream, control):
    app = MacApp()
    status, text = mac_post(app, {"messages": [{"role": "user", "content": "Hi"}],
                                "ignore_eos": True, "stream": stream, **control})
    assert status == 400
    assert "ignore_eos true cannot be combined" in json.loads(text)["error"]["message"]
    assert not text.startswith("data:") and not app.calls


@pytest.mark.parametrize("control", [
    {}, {"response_format": {"type": "text"}}, {"tools": [tool("ordinary", strict=False)]},
    {"tools": [tool("inactive")], "tool_choice": "none"},
])
def test_ordinary_eos_suppression_survives_preparation_and_both_http_backends(tmp_path, grammars, control):
    engine = ScriptEngine("Hello")
    app = _app(tmp_path, engine, grammars)
    body = {"messages": [{"role": "user", "content": "Hi"}], "ignore_eos": True, **control}
    prepared = app.prepare(body, True)
    assert prepared.ignore_eos is True and prepared.grammar is None and prepared.calls is None
    with http_server(app) as port:
        assert post(port, body, True)[0] == 200
    assert len(engine.calls) == 1 and engine.calls[0]["constraint"] is None
    mac = MacApp()
    assert mac_post(mac, body)[0] == 200
    assert len(mac.calls) == 1 and mac.calls[0]["ignore_eos"] is True


def test_direct_request_constraints_reject_eos_suppression_before_compiler_access():
    # Direct MLX callers build their ChatJob's constraint here, without going through HTTP refusal().
    with pytest.raises(RequestError, match="ignore_eos true cannot be combined with tools"):
        grammar.request_constraint(object(), {"ignore_eos": True}, None, grammar.Spec("tools", "{}", "tools"))
    assert grammar.request_constraint(object(), {"ignore_eos": True}, None) is None
