"""Standard lifecycle through the actual shared and CUDA HTTP dispatchers."""
from contextlib import contextmanager
import http.client
import json
import threading

import pytest

from tensorfold.cuda.http import make_handler as cuda_handler
from tensorfold.server.authentication import KeyStore
from tensorfold.server.http import Server, make_handler as mlx_handler
from tensorfold.server.api_records import ApiRecordStore
from tensorfold.server.errors import RequestError
from tests.test_api_records import begin, limits
from tests.test_cuda_tool_choice import Engine, app_for
from tests.test_responses_api import stream_events
from tests.test_server_openai_compat import FakeApp


@contextmanager
def serving(tmp_path, backend, **store_limits):
    app = app_for(tmp_path, Engine()) if backend == "cuda" else FakeApp()
    app.auth = KeyStore(["fixture-alice", "fixture-bob"])
    app.api_record_store = ApiRecordStore(tmp_path / "records.sqlite", limits=limits(**store_limits))
    server = Server(("127.0.0.1", 0), (cuda_handler if backend == "cuda" else mlx_handler)(app))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield app, server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert not thread.is_alive()
        app.api_record_store.close()


def call(port, method, path, body=None, key="fixture-alice", stream=False):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, json.dumps(body) if body is not None else None,
                           {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        reply = connection.getresponse()
        raw = reply.read().decode()
        return reply.status, raw if stream else json.loads(raw)
    finally:
        connection.close()


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_standard_conversation_and_item_lifecycle_and_owner_isolation(tmp_path, backend):
    with serving(tmp_path, backend) as (app, port):
        code, conversation = call(port, "POST", "/v1/conversations", {"metadata": {"topic": "one"},
                         "items": [{"role": "developer", "content": "Keep the explicit historical rule."}]})
        assert code == 200 and conversation["object"] == "conversation"
        path = "/v1/conversations/" + conversation["id"]
        assert call(port, "GET", path)[1] == conversation
        assert call(port, "GET", path, key="fixture-bob")[0] == 404
        assert call(port, "POST", path, {"metadata": {"topic": "two"}})[1]["metadata"] == {"topic": "two"}
        added = call(port, "POST", path + "/items", {"items": [{"role": "user", "content": "Question"}]})
        assert added[0] == 200 and added[1]["object"] == "list" and not added[1]["has_more"]
        identifier = added[1]["first_id"]
        assert call(port, "GET", path + "/items/" + identifier)[1]["content"][0]["text"] == "Question"
        page = call(port, "GET", path + "/items?limit=1&order=asc")[1]
        assert page["has_more"] and page["first_id"] != identifier
        assert call(port, "GET", path + "/items?after=" + page["last_id"] + "&order=asc")[1]["first_id"] == identifier
        assert call(port, "DELETE", path + "/items/" + identifier)[1]["object"] == "conversation"
        assert call(port, "GET", path + "/items/" + identifier)[0] == 404
        preserved = page["first_id"]
        assert call(port, "DELETE", path, key="fixture-bob")[0] == 404
        assert call(port, "DELETE", path)[1] == {"id": conversation["id"], "object": "conversation.deleted", "deleted": True}
        assert call(port, "GET", path)[0] == 404
        principal = app.auth.match_identity({"Authorization": "Bearer fixture-alice"})[1]
        assert app.api_record_store.get_item(principal, preserved)["role"] == "developer"


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
@pytest.mark.parametrize("stream", [False, True])
def test_response_conversation_instructions_branches_and_store_false(tmp_path, backend, stream):
    with serving(tmp_path, backend) as (app, port):
        conversation = call(port, "POST", "/v1/conversations", {"items": [{"role": "developer", "content": "historical"}]})[1]
        code, raw = call(port, "POST", "/v1/responses", {"input": "first", "conversation": conversation["id"],
                         "instructions": "request-only-secret", "stream": stream}, stream=stream)
        assert code == 200
        first = stream_events(raw)[-1]["response"] if stream else raw
        assert first["status"] == "completed" and first["conversation"] == {"id": conversation["id"]}
        assert call(port, "GET", "/v1/responses/" + first["id"])[1] == first
        assert call(port, "GET", "/v1/responses/" + first["id"], key="fixture-bob")[0] == 404
        assert call(port, "DELETE", "/v1/responses/" + first["id"], key="fixture-bob")[0] == 404
        code, second = call(port, "POST", "/v1/responses", {"input": "second", "previous_response_id": first["id"]})
        assert code == 200 and second["instructions"] is None
        if backend == "mlx":
            rendered = json.dumps(app.messages)
        else:
            rendered = app.tok.decode(app.engine.calls[-1][0])
        assert "historical" in rendered and "first" in rendered and "second" in rendered
        assert "request-only-secret" not in rendered
        inputs = call(port, "GET", "/v1/responses/" + second["id"] + "/input_items?order=asc")[1]
        assert len(inputs["data"]) == 4
        assert call(port, "DELETE", "/v1/responses/" + first["id"])[0] == 200
        assert call(port, "POST", "/v1/responses", {"input": "branch", "previous_response_id": second["id"]})[0] == 200
        code, unstored = call(port, "POST", "/v1/responses", {"input": "append", "conversation": {"id": conversation["id"]}, "store": False})
        assert code == 200 and call(port, "GET", "/v1/responses/" + unstored["id"])[0] == 404
        values = call(port, "GET", "/v1/conversations/" + conversation["id"] + "/items")[1]["data"]
        assert len(values) == 5
        assert call(port, "POST", "/v1/responses", {"input": "bad", "conversation": conversation["id"],
                        "previous_response_id": second["id"]})[0] == 400
        assert call(port, "POST", "/v1/responses", {"input": "foreign", "conversation": conversation["id"]}, key="fixture-bob")[0] == 404


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_standard_conflict_and_quota_refuse_before_generation(tmp_path, backend):
    with serving(tmp_path, backend, max_principal_conversations=1) as (app, port):
        code, conversation = call(port, "POST", "/v1/conversations")
        assert code == 200
        assert call(port, "POST", "/v1/conversations")[0] == 429
        assert call(port, "POST", "/v1/conversations", key="fixture-bob")[0] == 200
        principal = app.auth.match_identity({"Authorization": "Bearer fixture-alice"})[1]
        with begin(app.api_record_store, conversation=conversation["id"], principal=principal):
            code, conflict = call(port, "POST", "/v1/responses", {"input": "overlap", "conversation": conversation["id"]})
            assert code == 409 and "error" in conflict
            assert call(port, "DELETE", "/v1/conversations/" + conversation["id"])[0] == 409
        assert not app.engine.calls if backend == "cuda" else app.messages is None
        assert not app.api_record_store._operations


def test_stream_output_exhaustion_fails_without_publishing_completed_record(tmp_path):
    with serving(tmp_path, "mlx", max_record_bytes=1050) as (app, port):
        original = app.chat
        def large_delta(messages, *, on_delta=None, **kwargs):
            on_delta("generated" * 1000)
            return original(messages, on_delta=None, **kwargs)
        app.chat = large_delta
        code, raw = call(port, "POST", "/v1/responses", {"input": "hello", "stream": True}, stream=True)
        assert code == 200
        events = stream_events(raw)
        assert events[-1]["type"] == "response.failed"
        assert not app.api_record_store._operations
        failed = events[-1]["response"]
        assert call(port, "GET", "/v1/responses/" + failed["id"])[1] == failed
        assert app.api_record_store._usage()[0] == 1


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_failed_stream_record_is_retrievable_without_appending_conversation(tmp_path, backend):
    with serving(tmp_path, backend) as (app, port):
        if backend == "mlx":
            def failed(messages, *, on_delta=None, **kwargs):
                on_delta("partial answer")
                raise RuntimeError("model failed")
            app.chat = failed
        else:
            def failed(prompt, max_tokens, sampling, on_tokens, draft=True):
                on_tokens([ord("P")])
                raise RuntimeError("model failed")
            app.engine.generate = failed
        conversation = call(port, "POST", "/v1/conversations", {"items": [{"role": "developer", "content": "history"}]})[1]
        code, raw = call(port, "POST", "/v1/responses", {"input": "question", "conversation": conversation["id"], "stream": True}, stream=True)
        assert code == 200
        final = stream_events(raw)[-1]
        assert final["type"] == "response.failed" and final["response"]["error"]["message"] == "model failed"
        failed = final["response"]
        assert call(port, "GET", "/v1/responses/" + failed["id"])[1] == failed
        assert len(call(port, "GET", "/v1/responses/" + failed["id"] + "/input_items")[1]["data"]) == 2
        assert len(call(port, "GET", "/v1/conversations/" + conversation["id"] + "/items")[1]["data"]) == 1
        if backend == "mlx":
            assert failed["output"][0]["content"][0]["text"] == "partial answer"
        assert not app.api_record_store._operations


def test_failed_record_storage_exhaustion_reports_primary_and_retention_outcome(tmp_path):
    with serving(tmp_path, "mlx", max_record_bytes=1050) as (app, port):
        def failed(messages, *, on_delta=None, **kwargs):
            on_delta("partial")
            raise RuntimeError("model failure: " + "e" * 3000)
        app.chat = failed
        code, raw = call(port, "POST", "/v1/responses", {"input": "question", "stream": True}, stream=True)
        final = stream_events(raw)[-1]["response"]
        assert code == 200 and final["status"] == "failed"
        assert final["error"]["message"].startswith("model failure: " + "e" * 3000)
        assert "retention or retirement" in final["error"]["message"]
        assert call(port, "GET", "/v1/responses/" + final["id"])[0] == 404
        assert not app.api_record_store._operations


def test_observed_client_write_failure_retires_and_persists_cancelled_record(tmp_path, monkeypatch):
    from tensorfold.server import responses
    original = responses._client_write
    failed = False
    class BrokenSink:
        def write(self, _data):
            raise ConnectionResetError("controlled response socket failure")
    def client_write(handler, data):
        nonlocal failed
        if not failed and data.startswith(b"event: response.output_text.delta"):
            failed = True
            target, handler.wfile = handler.wfile, BrokenSink()
            try:
                return original(handler, data)
            finally:
                handler.wfile = target
        return original(handler, data)
    monkeypatch.setattr(responses, "_client_write", client_write)
    with serving(tmp_path, "mlx") as (app, port):
        code, raw = call(port, "POST", "/v1/responses", {"input": "question", "stream": True}, stream=True)
        events = stream_events(raw)
        identifier = events[0]["response"]["id"]
        assert code == 200 and failed
        result = call(port, "GET", "/v1/responses/" + identifier)
        assert result[0] == 200 and result[1]["status"] == "cancelled"
        assert "client disconnected" in result[1]["error"]["message"]
        assert not app.api_record_store._operations


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_nonfinite_client_numbers_refuse_before_record_admission_or_generation(tmp_path, backend):
    with serving(tmp_path, backend) as (app, port):
        for literal in ("NaN", "Infinity", "-Infinity", "1e309"):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                connection.request("POST", "/v1/responses", '{"input":"hello","temperature":' + literal + '}',
                                   {"Content-Type": "application/json", "Authorization": "Bearer fixture-alice"})
                result = connection.getresponse()
                assert result.status == 400
                assert json.loads(result.read())["error"]["type"] == "invalid_request_error"
            finally:
                connection.close()
        assert app.api_record_store._usage() == (0, 0, 0, 0) and not app.api_record_store._operations
        assert not app.engine.calls if backend == "cuda" else app.messages is None


def test_incomplete_reasoning_only_response_survives_actual_next_turn(tmp_path):
    observed = []
    def thinking(messages, *, on_delta=None, **kwargs):
        observed.append(messages)
        on_delta({"reasoning_content": "retained partial analysis"})
        return {"content": "", "finish_reason": "length", "prompt_tokens": 3,
                "completion_tokens": 2, "cached_tokens": 0}
    with serving(tmp_path, "mlx") as (app, port):
        app.chat = thinking
        code, raw = call(port, "POST", "/v1/responses", {"input": "question", "stream": True}, stream=True)
        first = stream_events(raw)[-1]["response"]
        assert code == 200 and first["status"] == "incomplete" and first["output"][0]["type"] == "reasoning"
        assert call(port, "POST", "/v1/responses", {"input": "continue", "previous_response_id": first["id"], "stream": True}, stream=True)[0] == 200
        assert observed[1][1] == {"role": "assistant", "content": "", "reasoning_content": "retained partial analysis"}


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_unmodified_openai_sdk_conversation_objects_and_items(tmp_path, backend):
    openai = pytest.importorskip("openai")
    with serving(tmp_path, backend) as (_, port), openai.OpenAI(
            api_key="fixture-alice", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0) as client:
        conversation = client.conversations.create(metadata={"suite": "standard"})
        added = client.conversations.items.create(conversation.id, items=[{"role": "user", "content": "hello"}])
        assert client.conversations.items.retrieve(added.first_id, conversation_id=conversation.id).id == added.first_id
        assert client.conversations.items.delete(added.first_id, conversation_id=conversation.id).id == conversation.id
        special = "msg imported/日本%25"
        client.conversations.items.create(conversation.id, items=[{"type": "message", "id": special,
            "role": "user", "content": [{"type": "input_text", "text": "encoded identity"}]}])
        assert client.conversations.items.retrieve(special, conversation_id=conversation.id).id == special
        assert client.conversations.items.delete(special, conversation_id=conversation.id).id == conversation.id
        assert call(port, "GET", "/v1/conversations/" + conversation.id + "/items/%FF")[0] == 400
        assert client.conversations.retrieve(conversation.id).metadata == {"suite": "standard"}
        assert client.conversations.delete(conversation.id).deleted is True
        empty = client.conversations.create(items=None)
        assert not list(client.conversations.items.list(empty.id))
        for invalid in ("text is not an array", False, 0, {}):
            assert call(port, "POST", "/v1/conversations", {"items": invalid})[0] == 400
            assert call(port, "POST", "/v1/conversations/" + empty.id + "/items", {"items": invalid})[0] == 400
        assert not list(client.conversations.items.list(empty.id))
        assert client.conversations.delete(empty.id).deleted


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_unmodified_sdk_response_lifecycle_stream_owner_and_standard_errors(tmp_path, backend):
    openai = pytest.importorskip("openai")
    with serving(tmp_path, backend, max_principal_conversations=1) as (app, port), openai.OpenAI(
            api_key="fixture-alice", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0) as alice, openai.OpenAI(
            api_key="fixture-bob", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0) as bob:
        model = "fake-cuda" if backend == "cuda" else "fake-model"
        conversation = alice.conversations.create(items=[{"role": "developer", "content": "historical"}])
        first = alice.responses.create(model=model, input="hello", instructions="this turn only", conversation=conversation.id)
        assert first.status == "completed" and first.output_text
        assert alice.responses.retrieve(first.id).id == first.id
        with pytest.raises(openai.NotFoundError):
            bob.responses.retrieve(first.id)
        with alice.responses.create(model=model, input="next", previous_response_id=first.id, stream=True) as stream:
            events = list(stream)
        assert events[-1].type == "response.completed"
        second = events[-1].response
        assert second.instructions is None and second.previous_response_id == first.id
        assert list(alice.responses.input_items.list(second.id, order="asc"))
        with pytest.raises(openai.BadRequestError):
            alice.responses.create(model=model, input="bad", conversation=conversation.id, previous_response_id=first.id)
        with pytest.raises(openai.RateLimitError):
            alice.conversations.create()
        # The pinned SDK's Responses.delete returns None; verify the actual
        # deletion independently through its standard retrieval operation.
        assert alice.responses.delete(first.id) is None
        with pytest.raises(openai.NotFoundError):
            alice.responses.retrieve(first.id)
        assert alice.responses.create(model=model, input="branch", previous_response_id=second.id).status == "completed"
        assert alice.conversations.delete(conversation.id).deleted
        assert not app.api_record_store._operations


def test_unmodified_sdk_conversation_overlap_returns_conflict_until_actual_generation_retires(tmp_path):
    openai = pytest.importorskip("openai")
    entered, release = threading.Event(), threading.Event()
    outcome = []
    with serving(tmp_path, "mlx") as (app, port), openai.OpenAI(
            api_key="fixture-alice", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0) as client:
        original = app.chat
        def blocked(messages, **kwargs):
            entered.set()
            assert release.wait(5)
            return original(messages, **kwargs)
        app.chat = blocked
        conversation = client.conversations.create()
        def run():
            try:
                outcome.append(client.responses.create(model="fake-model", input="first", conversation=conversation.id))
            except BaseException as error:
                outcome.append(error)
        worker = threading.Thread(target=run)
        worker.start()
        try:
            assert entered.wait(5)
            with pytest.raises(openai.ConflictError):
                client.responses.create(model="fake-model", input="overlap", conversation=conversation.id)
            with pytest.raises(openai.ConflictError):
                client.conversations.delete(conversation.id)
        finally:
            release.set()
            worker.join(5)
        assert not worker.is_alive() and len(outcome) == 1 and outcome[0].status == "completed"
        assert client.conversations.delete(conversation.id).deleted


@pytest.mark.parametrize("stream", [False, True])
def test_catalog_publication_refusal_is_containment_at_real_http_boundary(tmp_path, stream):
    primary = RequestError("catalog publication failed")
    def fail(*args, **kwargs):
        raise primary
    with serving(tmp_path, "mlx") as (app, port):
        app.api_record_store.bind_reference_listeners(unavailable=lambda *_args, **_kw: None, committed=fail)
        code, raw = call(port, "POST", "/v1/responses", {"input": "hello", "stream": stream}, stream=stream)
        if stream:
            assert code == 200
            final = stream_events(raw)[-1]
            assert final["type"] == "response.failed" and final["response"]["error"]["code"] == "server_error"
        else:
            assert code == 500 and raw["error"]["type"] == "server_error"
        assert app.api_record_store._failure is primary and not app.api_record_store._operations
        assert call(port, "POST", "/v1/conversations")[0] == 500
