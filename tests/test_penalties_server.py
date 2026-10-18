"""Penalty request fields: validated, defaulted, handed to engines that apply them, refused where none does."""

import json

import pytest

from tensorfold.engine.penalties import Penalties
from tensorfold.server.errors import RequestError
from tensorfold.server.request_options import parse_numbers


@pytest.mark.parametrize("field,value,words", [("presence_penalty", 2.5, "between -2 and 2"),
                                               ("frequency_penalty", -2.01, "between -2 and 2"),
                                               ("repetition_penalty", 0, "greater than 0"),
                                               ("repetition_penalty", "nan", "a finite number"),
                                               ("presence_penalty", True, "a finite number")])
def test_bad_penalties_are_refused(field, value, words):
    with pytest.raises(RequestError, match=f"{field} must be {words}"):
        parse_numbers({field: value})


def test_good_penalties_are_read():
    fields = parse_numbers({"presence_penalty": "1.5", "frequency_penalty": -2, "repetition_penalty": 1.05})
    assert (fields["presence_penalty"], fields["frequency_penalty"], fields["repetition_penalty"]) == (1.5, -2.0, 1.05)


def _app(tmp_path, engine):
    from tests.test_cuda_request_policy import TextTokenizer, app_for

    class Tokenizer(TextTokenizer):
        def token_to_id(self, text):
            return None

    app = app_for(tmp_path)
    app.tok = Tokenizer()
    app.engine = engine
    app.sampling = {"temperature": 0.0, "top_k": 0, "top_p": 1.0}       # App's defaults without a generation config
    return app


class Penalized:
    eos = (0,)

    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, penalties=None):
        self.calls.append((list(prompt), penalties))
        on_tokens([ord(c) for c in "Hi"])
        return {}


def test_the_cuda_server_hands_active_penalties_to_the_engine(tmp_path):
    from tests.test_cuda_admission import http_server, post

    app = _app(tmp_path, Penalized())
    body = {"messages": [{"role": "user", "content": "x"}]}
    with http_server(app) as port:
        assert post(port, body, True)[0] == 200
        assert post(port, {**body, "presence_penalty": 0, "frequency_penalty": 0.0, "repetition_penalty": 1}, True)[0] == 200
        assert post(port, {**body, "presence_penalty": 1.5, "temperature": 0.7}, True)[0] == 200
        app.sampling = {**app.sampling, "repetition_penalty": 1.05}        # generation_config / --repetition-penalty
        assert post(port, {**body, "frequency_penalty": 0.5}, True)[0] == 200
        assert post(port, {**body, "repetition_penalty": None}, True)[0] == 200
        for stream in (False, True):
            status, text = post(port, {**body, "presence_penalty": 3, "stream": stream}, True)
            assert status == 400 and "presence_penalty must be between -2 and 2" in json.loads(text)["error"]["message"]
    calls = app.engine.calls
    prompt = calls[0][0]
    assert [penalties for _, penalties in calls] == [
        None, None, Penalties(presence=1.5, prompt_length=len(prompt)),
        Penalties(frequency=0.5, repetition=1.05, prompt_length=len(prompt)),
        Penalties(repetition=1.05, prompt_length=len(prompt))]


def test_an_engine_without_penalties_refuses_active_ones_only(tmp_path):
    from tests.test_cuda_admission import http_server, post
    from tests.test_cuda_request_policy import Engine

    app = _app(tmp_path, Engine("Hi"))
    body = {"messages": [{"role": "user", "content": "x"}]}
    with http_server(app) as port:
        assert post(port, {**body, "presence_penalty": 0, "repetition_penalty": 1.0}, True)[0] == 200
        status, text = post(port, {**body, "repetition_penalty": 1.1}, True)
    assert status == 400 and "does not apply presence, frequency or repetition penalties" in text
    app.engine = Penalized()
    app.engine.penalties_supported = False             # e.g. a two-rank engine
    with http_server(app) as port:
        assert post(port, {**body, "presence_penalty": 1}, True)[0] == 400


def test_the_mac_server_refuses_penalties_it_cannot_apply():
    from tests.test_openai_compliance import HI, FakeApp, exchange, mac_server

    app = FakeApp()
    with mac_server(app) as port:
        assert exchange(port, "POST", "/v1/chat/completions",
                        {"messages": HI, "max_tokens": 8, "presence_penalty": 0, "repetition_penalty": 1})[0] == 200
        status, _, text = exchange(port, "POST", "/v1/chat/completions",
                                   {"messages": HI, "max_tokens": 8, "frequency_penalty": 0.5})
    assert status == 400 and "CUDA engines only" in text
