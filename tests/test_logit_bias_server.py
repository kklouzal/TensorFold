"""HTTP validation/forwarding and truthful unsupported-backend behavior."""
from types import SimpleNamespace

import pytest

from tensorfold.server.errors import RequestError
from tests.test_penalties_server import Penalized, _app


def app_for_bias(tmp_path):
    app = _app(tmp_path, Penalized())
    app.engine.target = SimpleNamespace(config=SimpleNamespace(vocab=256))
    return app


@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}]}),
    ("/v1/completions", {"prompt": "Hi"}),
    ("/v1/responses", {"input": "Hi", "store": False}),
])
def test_chat_completion_and_responses_deliver_bias_to_the_same_engine(tmp_path, path, body):
    from tests.test_cuda_admission import http_server
    from tests.test_openai_compliance import exchange

    app = app_for_bias(tmp_path)
    with http_server(app) as port:
        status, _, _ = exchange(port, "POST", path, {**body, "logit_bias": {"72": -100, "105": 2.5}})
    assert status == 200
    prompt, policy = app.engine.calls[-1]
    assert policy.bias == ((72, -100.0), (105, 2.5))
    assert policy.prompt_length == len(prompt)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("bias", [{"256": 0}, {"256": -100}, {"01": 1}, {"1": True}, {"1": 101}])
def test_invalid_bias_refuses_before_generate_or_sse(tmp_path, stream, bias):
    from tests.test_cuda_admission import http_server, post

    app = app_for_bias(tmp_path)
    app.tok.get_vocab_size = lambda **kwargs: 512  # added tokenizer rows cannot enlarge the real output matrix
    with http_server(app) as port:
        status, text = post(port, {"messages": [{"role": "user", "content": "Hi"}],
                                  "stream": stream, "logit_bias": bias}, True)
    assert status == 400 and "logit_bias" in text and not app.engine.calls
    assert not text.startswith("data:")


def test_empty_zero_and_null_bias_keep_existing_off_policy(tmp_path):
    app = app_for_bias(tmp_path)
    for bias in [None, {}, {"72": 0}]:
        assert app.penalties_for({"logit_bias": bias}) is None


@pytest.mark.parametrize("supported", [False, None])
def test_unsupported_engine_and_tp_refuse_active_bias(tmp_path, supported):
    from tests.test_cuda_request_policy import Engine

    app = app_for_bias(tmp_path)
    if supported is False:
        app.engine.penalties_supported = False
    else:
        app.engine = Engine("Hi")
        app.engine.target = SimpleNamespace(config=SimpleNamespace(vocab=256))
    with pytest.raises(RequestError, match="does not apply logit_bias"):
        app.penalties_for({"logit_bias": {"72": -100}})
    assert app.penalties_for({"logit_bias": {"72": 0}}) is None


def test_metal_refuses_active_bias_instead_of_silently_ignoring_it():
    from tests.test_openai_compliance import HI, FakeApp, exchange, mac_server

    with mac_server(FakeApp()) as port:
        status, _, text = exchange(port, "POST", "/v1/chat/completions",
                                   {"messages": HI, "logit_bias": {"1": -100}})
        assert status == 400 and "CUDA engines only" in text
        assert exchange(port, "POST", "/v1/chat/completions", {"messages": HI, "logit_bias": {}})[0] == 200
