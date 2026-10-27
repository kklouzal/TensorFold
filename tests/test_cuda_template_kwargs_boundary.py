"""Client template options cannot replace separately validated request inputs."""

import json

import pytest

from tensorfold.cuda import server
from tensorfold.server.errors import RequestError
from tests.test_cuda_admission import http_server
from tests.test_token_routes import HI, call, cuda_app


@pytest.mark.parametrize("reserved", ["messages", "tools"])
def test_reserved_override_refuses_before_template_or_tokenizer_materialization(tmp_path, reserved):
    app = cuda_app(tmp_path)
    app.template.selected = lambda **kw: pytest.fail("reserved override selected a template")
    app.tok.encode = lambda *a, **kw: pytest.fail("reserved override encoded text")
    override = [{"role": "user", "content": ["image"] * 5000}] if reserved == "messages" else "image" * 5000
    with pytest.raises(RequestError, match="cannot override request messages or tools"):
        app.prepare({"messages": HI, "chat_template_kwargs": {reserved: override, "add_vision_id": True}}, True)
    assert app.engine.prompts == []


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/tokenize"])
@pytest.mark.parametrize("reserved", ["messages", "tools"])
def test_actual_http_boundary_returns_the_same_invalid_request_envelope(tmp_path, route, reserved):
    app = cuda_app(tmp_path)
    body = {"messages": HI, "stream": True, "chat_template_kwargs": {reserved: []}}
    with http_server(app) as port:
        status, response = call(port, route, body)
    assert status == 400 and response["error"]["type"] == "invalid_request_error"
    assert "cannot override" in response["error"]["message"] and not app.engine.prompts


def test_special_token_and_checkpoint_options_retain_their_existing_forwarding(tmp_path):
    app = cuda_app(tmp_path)
    source = ("{{ bos_token }}|{{ preserve_thinking }}|{{ mode }}|{{ documents|length }}|{{ suffix }}|"
              "{% for m in messages %}{{ m.content }}{% endfor %}"
              "{% if add_generation_prompt %}assistant:{% endif %}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": source}))
    app.template = server.ChatTemplate(tmp_path)
    kwargs = {"bos_token": "CUSTOM", "preserve_thinking": True, "mode": "existing", "suffix": "done",
              "documents": ["one", "two"], "add_generation_prompt": False, "enable_thinking": False}
    prepared = app.prepare({"messages": HI, "chat_template_kwargs": kwargs}, True)
    assert app.tok.decode(prepared.prompt) == "CUSTOM|True|existing|2|done|Hi"
    assert kwargs["add_generation_prompt"] is False and kwargs["bos_token"] == "CUSTOM"


def test_tokenize_internal_generation_prompt_switch_stays_supported(tmp_path):
    app = cuda_app(tmp_path)
    on = app.tokenize({"messages": HI, "chat_template_kwargs": {"enable_thinking": False}})
    off = app.tokenize({"messages": HI, "add_generation_prompt": False,
                        "chat_template_kwargs": {"enable_thinking": False}})
    assert app.tok.decode(off["tokens"]) == "user:Hi;"
    assert app.tok.decode(on["tokens"]) == "user:Hi;assistant:"
    assert app.engine.prompts == []
