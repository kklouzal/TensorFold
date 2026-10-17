"""A reply names the model id its request asked for when the server answers to it, the served name when it names none, and
an id the server does not answer to is refused (404 model_not_found), on both backends and /v1/responses."""

import json
from types import SimpleNamespace

import pytest

from tensorfold.server.errors import ModelNotFound, RequestError
from tensorfold.server.http import requested_model
from tests.http_fakes import post

HI = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("asked, named", [("alias-a", "alias-a"), ("fake-27b", "fake-27b"), (None, "fake-27b")])
def test_requested_model_names_the_asked_id_or_the_served_one_when_omitted(asked, named):
    mlx = SimpleNamespace(served_name="fake-27b", model_ids=["fake-27b", "alias-a"])
    cuda = SimpleNamespace(served="fake-27b", model_ids=["fake-27b", "alias-a"])
    body = {} if asked is None else {"model": asked}
    assert requested_model(mlx, body) == requested_model(cuda, body) == named
    assert requested_model(cuda, {"model": None}) == "fake-27b"           # null is an omitted model


def test_an_unknown_or_malformed_model_is_refused_on_both_backends():
    for app in (SimpleNamespace(served_name="fake-27b", model_ids=["fake-27b", "alias-a"]),
                SimpleNamespace(served="fake-27b", model_ids=["fake-27b", "alias-a"])):
        with pytest.raises(ModelNotFound, match="`other` does not exist") as found:
            requested_model(app, {"model": "other"})
        assert (found.value.code, found.value.param) == ("model_not_found", "model")
        with pytest.raises(RequestError, match="must be a string") as malformed:
            requested_model(app, {"model": 7})
        assert not isinstance(malformed.value, ModelNotFound)


def test_mac_chat_completion_and_stream_name_the_alias():
    pytest.importorskip("mlx.core")                  # the Mac server's fakes
    from tests.test_lane_server import make_app

    app = make_app()
    try:
        status, text = post(app, {"model": "alias-a", "messages": HI, "max_tokens": 2})
        assert status == 200 and json.loads(text)["model"] == "alias-a"
        status, text = post(app, {"model": "alias-a", "messages": HI, "max_tokens": 2, "stream": True})
        models = {json.loads(line[6:])["model"] for line in text.splitlines()
                  if line.startswith("data: {") and '"model"' in line}
        assert status == 200 and models == {"alias-a"}
        status, text = post(app, {"model": "someone-else", "messages": HI, "max_tokens": 2})
        assert status == 404 and json.loads(text)["error"]["code"] == "model_not_found"
    finally:
        app.close()


def test_mac_completion_and_responses_name_the_alias():
    pytest.importorskip("mlx.core")
    from tests.test_lane_server import make_app

    app = make_app()
    try:
        status, text = post(app, {"model": "alias-a", "prompt": "hi", "max_tokens": 2}, "/v1/completions")
        assert status == 200 and json.loads(text)["model"] == "alias-a"
        status, text = post(app, {"model": "alias-a", "input": "hi", "max_output_tokens": 2}, "/v1/responses")
        assert status == 200 and json.loads(text)["model"] == "alias-a"
    finally:
        app.close()
