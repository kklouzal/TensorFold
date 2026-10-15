"""Anthropic HTTP routes on both servers, over their own chat-completions handler and connection."""
from __future__ import annotations

import json
from typing import Any

from tensorfold.server.anthropic_translate import Reply, error, translate
from tensorfold.server.errors import RequestError, status_of
from tensorfold.server.token_routes import count_chat
from tensorfold.server.request_body import read_body
from tensorfold.server.responses import LIMIT, _event_wire, _event_writer, _run_chat, _send


def route(path: str) -> bool:
    return path.split("?", 1)[0].rstrip("/") in ("/v1/messages", "/messages", "/v1/messages/count_tokens",
                                                 "/messages/count_tokens")


def post(handler: Any, app: Any) -> None:
    from tensorfold.server.http import requested_model

    count = handler.path.split("?", 1)[0].rstrip("/").endswith("/count_tokens")
    try:
        raw = read_body(handler, limit=LIMIT)
        body = json.loads(raw or b"{}")
        chat = translate(body, count=count)
        model = requested_model(app, body)          # an id this endpoint does not serve: not_found_error (404)
        late_system = getattr(app, "late_system", getattr(getattr(app, "template", None), "late_system", "system"))
        if late_system != "system" and any(m.get("role") == "system" for m in body["messages"]):
            raise RequestError("the model's chat template does not support mid-conversation system messages")
        if count:
            return _send(handler, 200, {"input_tokens": count_chat(app, chat)})
    except (RequestError, ValueError, UnicodeDecodeError) as exc:
        status = status_of(exc)
        return _send(handler, status, error(str(exc), status))

    send = _event_writer(handler)
    reply = Reply(model, send)
    wire = _event_wire(handler, reply)
    try:
        _run_chat(handler, chat, wire)
        if wire.stream or wire.status is None:
            return
        payload = json.loads(bytes(wire.data) or b"{}")
        if wire.status != 200:
            problem = payload.get("error") or {}
            message = problem.get("message", "request failed") if isinstance(problem, dict) else str(problem)
            return _send(handler, wire.status, error(message, wire.status))
        _send(handler, 200, reply.completion(payload))
    except OSError:
        handler.close_connection = True
    except (ValueError, KeyError, TypeError) as exc:
        if wire.stream:
            send(error(f"invalid model response: {exc}", 500))
        else:
            _send(handler, 500, error(f"invalid model response: {exc}", 500))
