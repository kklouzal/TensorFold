"""OpenAI's Responses API on both servers: each request runs as its chat completion on the server's own handler."""

from __future__ import annotations

import copy
import http.client
import io
import json
import math
import threading
import time
import traceback
import weakref
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

from tensorfold.server.errors import RequestError, error_body, invalid_request, retry_headers, server_error, status_of
from tensorfold.server.request_body import read_body
from tensorfold.server.responses_translate import Reply, _id, messages, translate
from tensorfold.server.authentication import current_principal

LIMIT = 32 * 1024**2


class _ClientGone(BrokenPipeError):
    """An actual public response-socket write failed, not record/media IO."""


def _client_write(handler, data):
    try:
        handler.wfile.write(data)
        handler.wfile.flush()
    except OSError as error:
        raise _ClientGone(str(error)) from error


def route(path: str) -> str | None:
    """A ``/v1/responses`` path's response id ("" for the collection), else None."""

    path = path.split("?", 1)[0].rstrip("/")
    for prefix in ("/v1/responses", "/responses"):
        if path == prefix:
            return ""
        rest = path[len(prefix) + 1:] if path.startswith(prefix + "/") else ""
        if rest and ("/" not in rest or rest.endswith("/input_items") and rest.count("/") == 1):
            return rest
    return None


# -- store ------------------------------------------------------------------------------------


@dataclass
class _Entry:
    response: dict[str, Any]
    parent: str | None
    messages: list[dict[str, Any]]                 # what this response added: its input, then its output
    principal: str
    size: int = field(default=0)


class Store:
    """The newest stored responses (``store`` true), in memory, for GET, DELETE and ``previous_response_id``."""

    def __init__(self, limit: int = 1024, max_bytes: int = 256 * 1024**2) -> None:
        self.limit, self.max_bytes = limit, max_bytes
        self.entries: OrderedDict[str, _Entry] = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()

    def put(self, response: dict[str, Any], added: list[dict[str, Any]], *, principal=None) -> None:
        principal = principal or current_principal() or "local-unauthenticated"
        entry = _Entry(copy.deepcopy(response), response.get("previous_response_id"),
                       copy.deepcopy(added + messages([item for item in response["output"]])),
                       principal)
        entry.size = len(json.dumps(entry.response)) + len(json.dumps(entry.messages))
        with self.lock:
            previous = self.entries.get(response["id"])
            if previous is not None and previous.principal != principal:
                raise RuntimeError("response identifier already belongs to another principal")
            if previous is not None:
                self.bytes -= previous.size
            self.entries[response["id"]] = entry
            self.bytes += entry.size
            while self.entries and (len(self.entries) > self.limit or self.bytes > self.max_bytes):
                self.bytes -= self.entries.popitem(last=False)[1].size

    def get(self, rid: str, *, principal=None) -> dict[str, Any] | None:
        principal = principal or current_principal() or "local-unauthenticated"
        with self.lock:
            entry = self.entries.get(rid)
            if entry is not None and entry.principal != principal:
                entry = None
        # Stored responses are owned snapshots, never mutated after publication.
        # Keeping this reference makes deletion/eviction safe while it is copied.
        return copy.deepcopy(entry.response) if entry is not None else None

    def delete(self, rid: str, *, principal=None) -> bool:
        principal = principal or current_principal() or "local-unauthenticated"
        with self.lock:
            entry = self.entries.get(rid)
            if entry is None or entry.principal != principal:
                return False
            self.entries.pop(rid)
            if entry is not None:
                self.bytes -= entry.size
            return entry is not None

    def conversation(self, rid: Any, *, principal=None) -> list[dict[str, Any]]:
        """Every message of a stored response's chain, oldest first (the chain stays newest in the store)."""

        if not isinstance(rid, str):
            raise RequestError("previous_response_id must be a string")
        principal = principal or current_principal() or "local-unauthenticated"
        chain: list[list[dict[str, Any]]] = []
        with self.lock:
            at: str | None = rid
            while at is not None:
                entry = self.entries.get(at)
                if entry is None or entry.principal != principal:
                    raise RequestError(f"previous response {at!r} is not stored here (store: false, deleted, "
                                       "or evicted): send the conversation's items instead")
                self.entries.move_to_end(at)
                chain.append(entry.messages)
                at = entry.parent
        return copy.deepcopy([m for part in reversed(chain) for m in part])


_STORES: "weakref.WeakKeyDictionary[Any, Store]" = weakref.WeakKeyDictionary()
_STORES_LOCK = threading.Lock()


def store_for(app: Any) -> Store:
    configured = getattr(app, "api_record_store", None)
    if configured is not None:
        return configured
    with _STORES_LOCK:
        found = _STORES.get(app)
        if found is None:
            found = _STORES[app] = Store()
        return found


# -- HTTP -------------------------------------------------------------------------------------


def _send(handler: Any, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode()
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        for name, value in retry_headers(status):       # a capacity refusal: when to retry
            handler.send_header(name, value)
        if handler.close_connection:                 # so a pooling client does not reuse the socket
            handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(data)
    except OSError:                                  # the client has gone
        handler.close_connection = True


def _refuse(handler: Any, message: str, status: int = 400, param: str | None = None) -> None:
    _send(handler, status, {"error": invalid_request(message, param)})


def _health_problem(store):
    if getattr(store, "durable_conversations", False):
        try:
            store.check_health()
        except Exception as error:
            traceback.print_exception(error)
            return server_error("logical API operation requires containment")
    return None


def _problem(exc, param=None, *, store=None):
    health = _health_problem(store)
    return (500, health) if health is not None else (getattr(exc, "status", status_of(exc)), error_body(exc, param))


def _refused(handler: Any, exc: RequestError, param: str | None = None, *, store=None) -> None:
    """A refusal's own status and error object (404 for an unknown model); ``param``, the field a context-length
    refusal names."""

    status, problem = _problem(exc, param, store=store)
    _send(handler, status, {"error": problem})


def _read(handler: Any, app: Any) -> tuple[Any, str]:
    """A request's JSON body and the served id it names; RequestError for a body that is not JSON, ModelNotFound for
    an id this endpoint does not serve."""

    from tensorfold.server.http import requested_model     # http imports this module

    configured = getattr(app, "api_record_store", None)
    limit = min(LIMIT, configured.limits.max_operation_bytes) if configured is not None else LIMIT
    payload = read_body(handler, limit=limit)
    try:
        body = _client_json(payload or b"{}")
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise RequestError("the request body is not JSON") from None
    return body, requested_model(app, body)


def _client_json(payload):
    """Strict finite JSON at the client boundary, before logical admission."""
    def constant(_value):
        raise ValueError("nonfinite numbers are not JSON values")
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("JSON number is outside the supported finite range")
        return result
    return json.loads(payload, parse_constant=constant, parse_float=number)


class Wire:
    """The chat handler's side of its socket: the status and headers, then the JSON body or each stream event. A
    stream's SSE comment lines (the CUDA server's keepalive while a prompt waits or fills) go to ``comment``, so the
    client of the translated stream hears them too."""

    def __init__(self, opened: Callable[[], None], event: Callable[[dict[str, Any] | None], None],
                 comment: Callable[[bytes], None]) -> None:
        self.opened, self.event, self.comment = opened, event, comment
        self.status: int | None = None
        self.stream = False
        self.disconnected = False
        self.data = bytearray()

    def write(self, data: bytes) -> int:
        self.data += data
        if self.status is None:
            end = self.data.find(b"\r\n\r\n")
            if end < 0:
                return len(data)
            head = bytes(self.data[:end]).decode("latin-1").split("\r\n")
            del self.data[:end + 4]
            self.status = int(head[0].split()[1])
            kinds = [v.strip() for k, _, v in (line.partition(":") for line in head[1:]) if k.lower() == "content-type"]
            self.stream = self.status == 200 and bool(kinds) and kinds[0].startswith("text/event-stream")
            if self.stream:
                try:
                    self.opened()
                except _ClientGone:
                    self.disconnected = True
                    raise
        while self.stream and (end := self.data.find(b"\n\n")) >= 0:
            block = bytes(self.data[:end])
            del self.data[:end + 2]
            for line in block.split(b"\n"):
                if line.startswith(b"data: ") or line.startswith(b":"):
                    try:
                        if line.startswith(b":"):
                            self.comment(line)
                        else:
                            self.event(None if line[6:] == b"[DONE]" else json.loads(line[6:]))
                    except _ClientGone:
                        self.disconnected = True
                        raise
        return len(data)

    def flush(self) -> None:
        pass


def _event_writer(handler: Any) -> Callable[[dict[str, Any]], None]:
    """Write one translated stream event to the client: its ``event:`` type line and JSON ``data:`` line."""

    def send(event: dict[str, Any]) -> None:
        _client_write(handler, f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())

    return send


def _event_wire(handler: Any, reply: Any, *, store=None) -> Wire:
    """The chat handler's ``Wire`` for a translated reply: a streamed chat reply opens the client's event stream
    (then ``reply.start``) and feeds ``reply.chunk``; the chat stream's comment lines pass through to the client."""

    def opened() -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True
        reply.start()

    def comment(line: bytes) -> None:
        _client_write(handler, line + b"\n\n")

    def event(payload):
        if payload is not None and "error" in payload:
            health = _health_problem(store)
            if health is not None:
                payload = {"error": health}
        reply.chunk(payload)

    return Wire(opened, event, comment)


def _run_chat(handler: Any, body: dict[str, Any], wire: Wire) -> None:
    """The server's own chat-completions handler on ``body``, over this connection, its response written to ``wire``."""

    data = json.dumps(body).encode()
    inner = type(handler).__new__(type(handler))
    inner.__dict__.update(handler.__dict__)
    inner.path, inner.rfile, inner.wfile, inner._headers_buffer = "/v1/chat/completions", io.BytesIO(data), wire, []
    inner.headers = http.client.HTTPMessage()
    inner.headers["Content-Type"] = "application/json"
    inner.headers["Content-Length"] = str(len(data))
    inner.log_request = lambda *args: None          # the request logs once, as itself
    inner.do_POST()


def post(handler: Any, app: Any) -> None:
    """POST /v1/responses."""

    store = store_for(app)
    principal = current_principal() or "local-unauthenticated"
    try:
        body, model = _read(handler, app)
        durable = getattr(store, "durable_conversations", False)
        from tensorfold.server.api_record_media import video_capable
        request = translate(body, store, principal=principal, history=[] if durable else None,
                            allow_video=durable and video_capable(app))
    except RequestError as exc:
        return _refused(handler, exc, store=store)
    except ValueError as exc:
        return _refused(handler, exc, store=store)
    except Exception as error:
        traceback.print_exception(error)
        return _send(handler, 500, {"error": server_error("logical API operation failed")})
    base = {"id": _id("resp"), "object": "response", "created_at": int(time.time()), "status": "in_progress",
            "error": None, "incomplete_details": None, "model": model, "output": [], "usage": None,
            "access_programs": None, **request.echo}
    try:
        scope = store.response_operation(principal, base["id"], previous_response_id=body.get("previous_response_id"),
                                         conversation_id=request.conversation_id, added=request.items,
                                         stored=request.store) if durable else nullcontext()
        with scope as operation:
            if operation is not None:
                operation.capture_media(app)
                instructions = ([{"role": "system", "content": body["instructions"]}] if body.get("instructions") else [])
                request.chat["messages"] = instructions + messages(list(operation.execution_input()), allow_video=video_capable(app))
                store.check_materialized(operation.history, operation.execution_history, operation.added, operation.snapshots, request.chat)
            failure_keep = None
            if operation is not None:
                def keep(final):
                    operation.commit(final)
                    operation.close()
                def failure_keep(final):
                    operation.fail(final)
                    operation.close()
            else:
                keep = (lambda final: store.put(final, request.added, principal=principal)) if request.store else None
            reply = Reply(base, _event_writer(handler) if request.stream else None, keep,
                          budget=operation.output_budget(base) if operation is not None else None,
                          failure_keep=failure_keep)
            wire = _event_wire(handler, reply, store=store)
            with operation.execution_scope() if operation is not None else nullcontext():
                try:
                    return _complete(handler, request, reply, wire, store=store)
                except _ClientGone:
                    reply.abandon(server_error("client disconnected before terminal response"))
                    handler.close_connection = True
                    return
                except RequestError as error:
                    if wire.stream:
                        reply.fail(_problem(error, store=store)[1])
                        return
                    raise
                except Exception as error:
                    if wire.stream:
                        traceback.print_exception(error)
                        reply.fail(server_error("logical API operation failed"))
                        return
                    raise
    except RequestError as exc:
        if 'wire' in locals() and wire.stream:
            reply.fail(_problem(exc, store=store)[1])
            return
        return _refused(handler, exc, store=store)
    except OSError as error:
        traceback.print_exception(error)
        return _send(handler, 500, {"error": server_error("logical API operation failed")})
    except Exception as error:
        traceback.print_exception(error)
        if 'wire' in locals() and wire.stream:
            reply.fail(server_error("logical API operation failed"))
        else:
            _send(handler, 500, {"error": server_error("logical API operation failed")})


def _complete(handler, request, reply, wire, *, store=None):
    _run_chat(handler, request.chat, wire)
    if wire.stream:                                   # the events are written (none more if the client left)
        if wire.disconnected:
            reply.abandon(server_error("client disconnected before terminal response"))
            handler.close_connection = True
            return
        if reply.final is None:
            # The inner handler can report a failure while writing its final
            # chunk by writing an HTTP error body to this private transport.
            # It cannot open a second public HTTP response after SSE began.
            raw = bytes(wire.data)
            if raw.startswith(b"HTTP/"):
                raw = raw.partition(b"\r\n\r\n")[2]
            try:
                payload = json.loads(raw)
                problem = payload.get("error") if isinstance(payload, dict) else None
            except (ValueError, UnicodeDecodeError):
                problem = None
            reply.fail(_health_problem(store) or problem or server_error("the reply ended before terminal publication"))
        return
    if wire.status is None:                           # the client left before the reply
        handler.close_connection = True
        return
    if wire.status != 200:                            # refused as a chat completion: the same refusal
        try:
            payload = json.loads(bytes(wire.data) or b"{}")
        except ValueError:
            payload = {"error": server_error(bytes(wire.data).decode("utf-8", "replace"))}
        problem = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(problem, dict) and problem.get("param") == "messages":    # the chat field is this one's input
            problem["param"] = "input"
        return _send(handler, wire.status, payload)
    _send(handler, 200, reply.completion(json.loads(bytes(wire.data))))


def input_tokens(handler: Any, app: Any) -> None:
    """POST /v1/responses/input_tokens: the request's input tokens as its chat completion renders them, without
    generating (OpenAI's ``response.input_tokens``)."""

    from tensorfold.server.token_routes import count_chat

    store = store_for(app)
    try:
        body, _model = _read(handler, app)
        principal = current_principal() or "local-unauthenticated"
        durable = getattr(store, "durable_conversations", False)
        from tensorfold.server.api_record_media import video_capable
        request = translate(body, store, principal=principal, history=[] if durable else None,
                            allow_video=durable and video_capable(app))
        scope = store.response_operation(principal, _id("resp"), previous_response_id=body.get("previous_response_id"),
                                         conversation_id=request.conversation_id, added=request.items,
                                         stored=False, read_only=True) if durable else nullcontext()
        with scope as operation:
            if operation is not None:
                operation.capture_media(app)
                instructions = ([{"role": "system", "content": body["instructions"]}] if body.get("instructions") else [])
                request.chat["messages"] = instructions + messages(list(operation.execution_input()), allow_video=video_capable(app))
                store.check_materialized(operation.history, operation.execution_history, operation.added, operation.snapshots, request.chat)
            count = count_chat(app, request.chat)
    except RequestError as exc:
        return _refused(handler, exc, "input", store=store)
    except ValueError as exc:
        return _refused(handler, exc, store=store)
    except (ConnectionError, TimeoutError):     # the client has gone or stalled mid-body: the connection closes
        raise
    except Exception as exc:          # noqa: BLE001  a template's own refusal, say: refused, as the chat route does
        print(f"[tensorfold] request error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exception(exc)
        return _refused(handler, exc, store=store)
    _send(handler, 200, {"object": "response.input_tokens", "input_tokens": count})


def get(handler: Any, app: Any, rid: str | None) -> None:
    """GET /v1/responses/{id}."""

    store = store_for(app)
    principal = current_principal() or "local-unauthenticated"
    if rid and rid.endswith("/input_items"):
        if not getattr(store, "durable_conversations", False):
            return _refuse(handler, "input item retrieval requires a durable API record store")
        from tensorfold.server.conversations import pagination
        try:
            result = store.response_items(principal, rid.split("/")[0], **pagination(handler.path))
        except RequestError as error:
            return _refused(handler, error, store=store)
        except Exception as error:
            traceback.print_exception(error)
            return _send(handler, 500, {"error": server_error("logical API operation failed")})
        return _send(handler, 200, result)
    try:
        found = store.get(rid, principal=principal) if rid else None
    except RequestError as error:
        return _refused(handler, error, store=store)
    except Exception as error:
        traceback.print_exception(error)
        return _send(handler, 500, {"error": server_error("logical API operation failed")})
    if found is None:
        return _refuse(handler, f"no stored response has id {rid!r}", 404, "response_id")
    _send(handler, 200, found)


def delete(handler: Any, app: Any, rid: str | None) -> None:
    """DELETE /v1/responses/{id}."""

    store = store_for(app)
    try:
        deleted = bool(rid) and store.delete(rid, principal=current_principal() or "local-unauthenticated")
    except RequestError as error:
        return _refused(handler, error, store=store)
    except Exception as error:
        traceback.print_exception(error)
        return _send(handler, 500, {"error": server_error("logical API operation failed")})
    if not deleted:
        return _refuse(handler, f"no stored response has id {rid!r}", 404, "response_id")
    _send(handler, 200, {"id": rid, "object": "response.deleted", "deleted": True})
