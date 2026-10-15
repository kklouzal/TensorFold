"""OpenAI compatibility of both servers' replies, checked against the pinned spec's schemas (tests/openai_spec.py):
error objects, capacity refusals, unknown models, the models routes, choice logprobs, include_usage chunks, Responses
deletion, and the request fields this server accepts and ignores. CPU only (stand-in engines)."""

from __future__ import annotations

import http.client
import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest

pytest.importorskip("jinja2")

from tensorfold.server import stats
from tensorfold.server.errors import RETRY_AFTER_S, CapacityError
from tensorfold.server.http import make_handler as mac_handler
from tests.openai_spec import check, check_error, schemas
from tests.test_cuda_admission import http_server
from tests.test_cuda_server_errors import HI, app_for
from tests.test_server_openai_compat import FakeApp
from tests.test_token_routes import MacApp


def exchange(port, method, path, body=None, headers=None):
    """(status, lowercased headers, body text)."""

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        data = json.dumps(body).encode() if body is not None else None
        connection.request(method, path, data, {"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read().decode()
    finally:
        connection.close()


def frames(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]


@contextmanager
def mac_server(app):
    server = ThreadingHTTPServer(("127.0.0.1", 0), mac_handler(app))
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@contextmanager
def backend(name, tmp_path, app=None):
    """A served stand-in app of either server: (port, the id it serves)."""

    if name == "cuda":
        app = app if app is not None else app_for(tmp_path)
        app.aliases = ("org/alias",)
        with http_server(app) as port:
            yield port, app.served
    else:
        app = app if app is not None else MacApp()
        app.model_ids = [*app.model_ids, "org/alias"]
        with mac_server(app) as port:
            yield port, app.served_name


BACKENDS = ["cuda", "mlx"]


@pytest.mark.parametrize("name", BACKENDS)
def test_models_list_and_retrieve_are_spec_models_with_the_context_length(name, tmp_path):
    with backend(name, tmp_path) as (port, served):
        status, _, text = exchange(port, "GET", "/v1/models")
        listed = json.loads(text)
        check(listed, "ListModelsResponse")
        assert status == 200 and [m["id"] for m in listed["data"]] == [served, "org/alias"]
        assert {m["created"] for m in listed["data"]} == {int(stats.STARTED_UNIX_S)}      # stable while it runs
        assert all("max_model_len" in m for m in listed["data"])
        for model_id in (served, "org/alias", "org%2Falias"):
            status, _, text = exchange(port, "GET", f"/v1/models/{model_id}")
            check(json.loads(text), "Model")
            assert status == 200 and json.loads(text)["id"] == model_id.replace("%2F", "/")
        status, _, text = exchange(port, "GET", "/v1/models/someone-else")
        check_error(json.loads(text))
        assert status == 404 and json.loads(text)["error"]["code"] == "model_not_found"


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("method, path", [("GET", "/v1/nothing"), ("POST", "/v1/nothing"),
                                          ("DELETE", "/v1/nothing")])
def test_unknown_routes_are_spec_errors(name, method, path, tmp_path):
    with backend(name, tmp_path) as (port, _):
        status, _, text = exchange(port, method, path, {} if method == "POST" else None)
    check_error(json.loads(text))
    assert status == 404 and "unknown path" in json.loads(text)["error"]["message"]


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("route, body", [("/v1/chat/completions", {"messages": HI}),
                                         ("/v1/completions", {"prompt": "Hi"}),
                                         ("/v1/responses", {"input": "Hi"}),
                                         ("/tokenize", {"prompt": "Hi"})])
def test_an_unknown_model_is_404_on_every_route(name, route, body, tmp_path):
    with backend(name, tmp_path) as (port, served):
        status, _, text = exchange(port, "POST", route, {**body, "model": "someone-else", "max_tokens": 2})
        check_error(json.loads(text))
        error = json.loads(text)["error"]
        assert status == 404 and (error["code"], error["param"]) == ("model_not_found", "model")
        status, _, text = exchange(port, "POST", route, {**body, "model": 7, "max_tokens": 2})
        assert status == 400 and json.loads(text)["error"]["code"] is None
        status, _, _ = exchange(port, "POST", route, {**body, "model": "org/alias", "max_tokens": 2})
        assert status == 200


@pytest.mark.parametrize("name", BACKENDS)
def test_messages_routes_refuse_an_unknown_model_in_anthropic_s_shape(name, tmp_path):
    body = {"model": "someone-else", "max_tokens": 4, "messages": [{"role": "user", "content": "Hi"}]}
    with backend(name, tmp_path) as (port, _):
        for route in ("/v1/messages", "/v1/messages/count_tokens"):
            status, _, text = exchange(port, "POST", route, body)
            assert status == 404 and json.loads(text)["error"]["type"] == "not_found_error"


def busy(*args, **kwargs):
    raise CapacityError("the request queue is full; retry shortly")


@pytest.mark.parametrize("name, stream", [("mlx", False), ("cuda", False), ("cuda", True)])
def test_a_capacity_refusal_is_503_service_unavailable_with_retry_after(name, stream, tmp_path):
    """The Mac server sent a non-streamed capacity refusal as HTTP 400; both answer as OpenAI's 503 before a reply
    (a stream whose headers are out ends with an error event instead)."""

    app = MacApp() if name == "mlx" else app_for(tmp_path)
    if name == "mlx":
        app.chat = busy
    else:
        app.prepare = busy                  # refused while preparing: before a stream's headers
    with backend(name, tmp_path, app) as (port, _):
        status, headers, text = exchange(port, "POST", "/v1/chat/completions",
                                         {"messages": HI, "max_tokens": 2, "stream": stream})
        check_error(json.loads(text))
        error = json.loads(text)["error"]
        assert status == 503 and headers["retry-after"] == str(RETRY_AFTER_S)
        assert (error["type"], error["code"]) == ("service_unavailable_error", "server_is_overloaded")
        if name == "cuda":
            del app.prepare                 # the App's own preparation again
        status, headers, _ = exchange(port, "POST", "/v1/chat/completions", {"messages": HI, "n": 2})
        assert status == 400 and "retry-after" not in headers


def test_a_mac_decision_capacity_refusal_is_503():
    app = MacApp()
    app.decisions = busy
    with mac_server(app) as port:
        status, headers, text = exchange(port, "POST", "/v1/decisions", {"input": "x", "questions": []})
    check_error(json.loads(text))
    assert status == 503 and headers["retry-after"] == str(RETRY_AFTER_S)


@pytest.mark.parametrize("name", BACKENDS)
def test_refusals_and_failures_carry_every_error_field(name, tmp_path):
    class Failing(FakeApp):
        def chat(self, messages, **kwargs):
            raise RuntimeError("the engine failed")

    class FailingEngine:
        eos = (0,)

        def generate(self, *args, **kwargs):
            raise RuntimeError("the engine failed")

    with backend(name, tmp_path) as (port, _):
        status, _, text = exchange(port, "POST", "/v1/chat/completions", {"messages": HI, "n": 2})
        check_error(json.loads(text))
        assert status == 400 and json.loads(text)["error"]["type"] == "invalid_request_error"
        status, _, text = exchange(port, "POST", "/v1/chat/completions", None)
        check_error(json.loads(text))
        assert status == 400
    app = Failing() if name == "mlx" else app_for(tmp_path)
    if name == "cuda":
        app.engine = FailingEngine()
    with backend(name, tmp_path, app) as (port, _):
        status, _, text = exchange(port, "POST", "/v1/chat/completions", {"messages": HI, "max_tokens": 2})
    check_error(json.loads(text))
    assert status == 500 and json.loads(text)["error"]["type"] == "server_error"


@pytest.mark.parametrize("name", BACKENDS)
def test_replies_and_include_usage_chunks_match_the_spec(name, tmp_path):
    with backend(name, tmp_path) as (port, _):
        status, _, text = exchange(port, "POST", "/v1/chat/completions", {"messages": HI, "max_tokens": 8})
        reply = json.loads(text)
        check(reply, "CreateChatCompletionResponse")
        assert status == 200 and reply["choices"][0]["logprobs"] is None
        status, _, text = exchange(port, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 8})
        check(json.loads(text), "CreateCompletionResponse")
        assert json.loads(text)["choices"][0]["logprobs"] is None
        status, _, text = exchange(port, "POST", "/v1/chat/completions",
                                   {"messages": HI, "max_tokens": 8, "stream": True,
                                    "stream_options": {"include_usage": True}})
        chunks = frames(text)
        for chunk in chunks:
            check(chunk, "CreateChatCompletionStreamResponse")
        assert all(chunk["usage"] is None for chunk in chunks[:-1])
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0
        status, _, text = exchange(port, "POST", "/v1/chat/completions",
                                   {"messages": HI, "max_tokens": 8, "stream": True})
        assert all("usage" not in chunk for chunk in frames(text)[:-1])     # without include_usage: no null usage
        assert frames(text)[-1]["usage"]["completion_tokens"] > 0           # on the finish chunk, as before


@pytest.mark.parametrize("name", BACKENDS)
def test_deleting_a_stored_response_returns_the_spec_deleted_object(name, tmp_path):
    with backend(name, tmp_path) as (port, _):
        status, _, text = exchange(port, "POST", "/v1/responses", {"input": "Hi", "max_output_tokens": 4})
        response = json.loads(text)
        assert status == 200 and response["access_programs"] is None
        assert not [key for key in schemas()["Response"]["required"] if key not in response]
        status, _, text = exchange(port, "DELETE", f"/v1/responses/{response['id']}")
        check(json.loads(text), "DeleteResponseResponse")
        assert status == 200 and json.loads(text) == {"id": response["id"], "object": "response.deleted",
                                                      "deleted": True}
        status, _, text = exchange(port, "DELETE", f"/v1/responses/{response['id']}")
        check_error(json.loads(text))
        assert status == 404


def test_a_responses_refusal_names_its_input_field(tmp_path):
    app = app_for(tmp_path)
    app.context_window = 4
    with http_server(app) as port:
        status, _, text = exchange(port, "POST", "/v1/responses", {"input": "a long enough prompt to overflow"})
    error = json.loads(text)["error"]
    check_error(json.loads(text))
    assert status == 400 and (error["code"], error["param"]) == ("context_length_exceeded", "input")


# fields the server accepts and ignores (owner decision 2026-10-15): the prompt and sampling are those of the request
# without them, and the reply is an ordinary one
IGNORED = {"logit_bias": {"50256": -100}, "verbosity": "low", "web_search_options": {"search_context_size": "low"},
           "moderation": {"model": "omni-moderation-latest"}, "functions": [{"name": "f", "parameters": {}}],
           "function_call": "auto", "prediction": {"type": "content", "content": "Hi"}, "store": True,
           "metadata": {"a": "b"}, "user": "u", "safety_identifier": "s", "prompt_cache_key": "k",
           "prompt_cache_retention": "24h", "service_tier": "priority", "frequency_penalty": 0.5,
           "presence_penalty": 0.5, "stream_options": {"include_obfuscation": True}}
COMPLETION_IGNORED = {"echo": True, "suffix": "!", "best_of": 3, "logit_bias": {"1": 5}, "user": "u"}


def test_cuda_accepts_and_ignores_the_fields_it_does_not_run(tmp_path):
    app = app_for(tmp_path)
    with http_server(app) as port:
        assert exchange(port, "POST", "/v1/chat/completions", {"messages": HI, "max_tokens": 8})[0] == 200
        status, _, text = exchange(port, "POST", "/v1/chat/completions", {"messages": HI, "max_tokens": 8, **IGNORED})
        check(json.loads(text), "CreateChatCompletionResponse")
        assert status == 200
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        allowed = {"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": [
            {"type": "function", "function": {"name": "f"}}]}}
        assert exchange(port, "POST", "/v1/chat/completions",
                        {"messages": HI, "max_tokens": 8, "tools": tools})[0] == 200
        assert exchange(port, "POST", "/v1/chat/completions",
                        {"messages": HI, "max_tokens": 8, "tools": tools, "tool_choice": allowed})[0] == 200
        assert exchange(port, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 8})[0] == 200
        assert exchange(port, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 8, **COMPLETION_IGNORED})[0] == 200
    plain, ignored, tool, allowed_tool, completion, completion_ignored = app.engine.calls
    assert plain == ignored and tool == allowed_tool and completion == completion_ignored


def test_mlx_accepts_and_ignores_the_fields_it_does_not_run():
    app = FakeApp()
    with mac_server(app) as port:
        status, _, text = exchange(port, "POST", "/v1/chat/completions", {"messages": HI, "max_tokens": 8, **IGNORED})
        check(json.loads(text), "CreateChatCompletionResponse")
        assert status == 200 and app.messages == [{"role": "user", "content": "Hi"}]
        assert exchange(port, "POST", "/v1/completions", {"prompt": "Hi", "max_tokens": 8, **COMPLETION_IGNORED})[0] == 200


def test_the_spec_oracle_rejects_what_the_spec_forbids():
    """The subset validator is this file's oracle: it must catch the defects it is used to rule out."""

    from tests.openai_spec import errors

    assert errors({"error": {"message": "x", "type": "t"}}, schemas()["ErrorResponse"])         # no param or code
    assert errors({"error": "not found"}, schemas()["ErrorResponse"])                           # a bare string
    assert errors({"id": "m", "object": "model", "owned_by": "o"}, schemas()["Model"])          # no created
    assert errors({"id": "r", "object": "response", "deleted": True}, schemas()["DeleteResponseResponse"])
    choice = {"index": 0, "message": {"role": "assistant", "content": "x", "refusal": None}, "finish_reason": "stop"}
    reply = {"id": "c", "object": "chat.completion", "created": 1, "model": "m", "choices": [choice]}
    assert errors(reply, schemas()["CreateChatCompletionResponse"])                              # no logprobs
    assert not errors({**reply, "choices": [{**choice, "logprobs": None}]},
                      schemas()["CreateChatCompletionResponse"])


def test_mlx_completions_refuse_a_batch_of_prompts_rather_than_joining_them():
    app = MacApp()
    with mac_server(app) as port:
        status, _, text = exchange(port, "POST", "/v1/completions", {"prompt": ["Hi", "there"], "max_tokens": 4})
        check_error(json.loads(text))
        assert status == 400 and "one prompt a request" in json.loads(text)["error"]["message"]
        assert exchange(port, "POST", "/v1/completions", {"prompt": ["Hi"], "max_tokens": 4})[0] == 200
    assert app.messages == [{"role": "user", "content": "Hi"}]


@pytest.mark.parametrize("name", BACKENDS)
def test_every_reply_carries_an_x_request_id(name, tmp_path):
    import re

    generated = re.compile(r"req-[0-9a-f]{32}")
    with backend(name, tmp_path) as (port, _):
        for header, value, expected in (("X-Client-Request-Id", "client.abc_1:2-3", "client.abc_1:2-3"),
                                        ("x-request-id", "upstream-7", "upstream-7"),
                                        ("x-request-id", "has spaces", None),
                                        ("X-Client-Request-Id", "x" * 129, None)):
            for method, path, body in (("GET", "/v1/models", None), ("POST", "/v1/chat/completions",
                                                                     {"messages": HI, "max_tokens": 2}),
                                       ("POST", "/v1/responses", {"input": "Hi", "max_output_tokens": 2}),
                                       ("POST", "/v1/nothing", {}), ("POST", "/v1/chat/completions", {"n": 2})):
                _, headers, _ = exchange(port, method, path, body, {header: value})
                got = headers["x-request-id"]
                assert got == expected if expected else generated.fullmatch(got), (method, path, value, got)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:                                        # a reused connection: each request has an id of its own
            ids = []
            for _ in range(2):
                connection.request("GET", "/v1/models")
                response = connection.getresponse()
                response.read()
                ids.append(response.getheader("x-request-id"))
            assert len(set(ids)) == 2 and all(generated.fullmatch(i) for i in ids)
        finally:
            connection.close()


def test_cuda_input_tokens_counts_the_prompt_a_response_would_run_without_generating(tmp_path):
    from tensorfold.server.responses import store_for
    from tensorfold.server.responses_translate import translate

    app = app_for(tmp_path)
    body = {"model": "fake-cuda", "input": "Say hello.", "instructions": "Be brief.", "max_output_tokens": 4}
    expected = len(app.prepare(translate(body, store_for(app)).chat, True).prompt)
    with http_server(app) as port:
        status, _, text = exchange(port, "POST", "/v1/responses/input_tokens", body)
        counted = json.loads(text)
        check(counted, "TokenCountsResource")
        assert status == 200 and counted == {"object": "response.input_tokens", "input_tokens": expected}
        assert not app.engine.calls                                    # nothing generated
        status, _, text = exchange(port, "POST", "/v1/responses", body)
        assert json.loads(text)["usage"]["input_tokens"] == expected   # the prompt the response ran
        status, _, text = exchange(port, "POST", "/v1/responses/input_tokens", {**body, "model": "other"})
        assert status == 404 and json.loads(text)["error"]["code"] == "model_not_found"
        status, _, text = exchange(port, "POST", "/v1/responses/input_tokens", {"model": "fake-cuda"})
        check_error(json.loads(text))
        assert status == 400


def test_mac_input_tokens_renders_as_tokenize_does():
    app = MacApp()
    app.reasoning_effort = None                 # the server's --reasoning-effort, as ChatApp keeps it
    with mac_server(app) as port:
        status, _, text = exchange(port, "POST", "/v1/responses/input_tokens", {"input": "Say hello."})
        counted = json.loads(text)
        tokenized = json.loads(exchange(port, "POST", "/tokenize",
                                        {"messages": [{"role": "user", "content": "Say hello."}]})[2])
    check(counted, "TokenCountsResource")
    assert status == 200 and counted["input_tokens"] == tokenized["count"] > 0


def test_input_tokens_refuses_a_prompt_the_context_cannot_hold_on_both_servers(tmp_path):
    for name in BACKENDS:
        app = app_for(tmp_path) if name == "cuda" else MacApp()
        app.context_window = 4
        app.reasoning_effort = None
        with backend(name, tmp_path, app) as (port, _):
            status, _, text = exchange(port, "POST", "/v1/responses/input_tokens", {"input": "a long prompt indeed"})
        error = json.loads(text)["error"]
        check_error(json.loads(text))
        assert status == 400 and (error["code"], error["param"]) == ("context_length_exceeded", "input"), name


def test_input_tokens_answers_an_unexpected_failure(tmp_path):
    app = app_for(tmp_path)

    def broken(body, chat):
        raise LookupError("the template looked up nothing")

    app.prepare = broken
    with http_server(app) as port:
        status, _, text = exchange(port, "POST", "/v1/responses/input_tokens", {"input": "Hi"})
    check_error(json.loads(text))
    assert status == 400 and "looked up nothing" in json.loads(text)["error"]["message"]
