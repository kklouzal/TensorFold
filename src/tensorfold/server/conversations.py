"""Standard Conversations objects/items over the configured durable logical store."""
from __future__ import annotations

import traceback
from urllib.parse import parse_qs, unquote, urlsplit

from tensorfold.server.authentication import current_principal
from tensorfold.server.errors import RequestError, server_error
from tensorfold.server.request_body import read_body
from tensorfold.server.responses import _client_json, _refused, _send


def route(path):
    path = path.split("?", 1)[0].rstrip("/")
    for prefix in ("/v1/conversations", "/conversations"):
        if path == prefix:
            return ()
        if path.startswith(prefix + "/"):
            raw = path[len(prefix) + 1:].split("/")
            try:
                parts = tuple(unquote(part, encoding="utf-8", errors="strict") for part in raw)
            except UnicodeDecodeError as cause:
                error = RequestError("path identifiers must use valid UTF-8 percent encoding")
                error.param = "path"
                raise error from cause
            if len(parts) == 1 or len(parts) in (2, 3) and parts[1] == "items":
                return parts
    return None


def metadata(value):
    if value is not None and (not isinstance(value, dict) or len(value) > 16 or not all(
            isinstance(k, str) and len(k) <= 64 and isinstance(v, str) and len(v) <= 512
            for k, v in value.items())):
        raise RequestError("metadata must be null or at most16 string pairs (keys64, values512 characters)")
    return value


def pagination(path):
    try:
        query = parse_qs(urlsplit(path).query, keep_blank_values=True, encoding="utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RequestError("item pagination must use valid UTF-8 percent encoding") from error
    if set(query) - {"after", "limit", "order", "include", "include[]"} or any(len(v) != 1 for k, v in query.items() if k not in {"include", "include[]"}):
        raise RequestError("unsupported or repeated item pagination parameter")
    included = query.get("include", []) + query.get("include[]", [])
    if any(value != "message.input_image.image_url" for value in included):
        raise RequestError("the requested item include is unsupported")
    try:
        limit = int(query.get("limit", ["20"])[0])
    except ValueError:
        raise RequestError("limit must be an integer from1 to100") from None
    return {"after": query.get("after", [None])[0], "limit": limit, "order": query.get("order", ["desc"])[0]}


def _body(handler, store):
    try:
        value = _client_json(read_body(handler, limit=store.limits.max_operation_bytes) or b"{}")
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise RequestError("the request body is not a JSON object") from None
    if not isinstance(value, dict):
        raise RequestError("the request body must be a JSON object")
    return value


def dispatch(handler, app, method):
    store = getattr(app, "api_record_store", None)
    try:
        parts = route(handler.path)
        if parts is None:
            return False
        if store is None:
            raise RequestError("conversations require a configured durable API record store")
        principal = current_principal() or "local-unauthenticated"
        if method == "GET" and len(parts) == 1:
            result = store.get_conversation(principal, parts[0])
        elif method == "GET" and len(parts) == 2:
            result = store.list_items(principal, parts[0], **pagination(handler.path))
        elif method == "GET" and len(parts) == 3:
            pagination(handler.path)  # validates the supported optional include
            result = store.conversation_item(principal, parts[0], parts[2])
        elif method == "DELETE" and len(parts) == 1:
            result = store.delete_conversation(principal, parts[0])
        elif method == "DELETE" and len(parts) == 3:
            result = store.delete_item(principal, parts[0], parts[2])
        elif method == "POST" and len(parts) in (0, 1, 2):
            body = _body(handler, store)
            allowed = {"items", "metadata"} if not parts else {"metadata"} if len(parts) == 1 else {"items"}
            if set(body) - allowed:
                raise RequestError("unsupported conversation request field")
            if not parts:
                items = [] if body.get("items") is None else body["items"]
                if not isinstance(items, list):
                    raise RequestError("items must be an array or null")
                with store.mutation(principal, items, create=True, maximum=20) as operation:
                    operation.capture_media(app)
                    result = store.create_conversation(principal, metadata=metadata(body.get("metadata")), items=operation.added, operation=operation)
            elif len(parts) == 1:
                result = (store.update_conversation(principal, parts[0], metadata(body["metadata"]))
                          if "metadata" in body else store.get_conversation(principal, parts[0]))
            else:
                if not isinstance(body.get("items"), list) or not body["items"]:
                    raise RequestError("items must be a nonempty list")
                with store.mutation(principal, body["items"], conversation=parts[0], maximum=20) as operation:
                    operation.capture_media(app)
                    result = store.add_items(principal, parts[0], operation.added, operation=operation)
                result = {"object": "list", "data": result, "first_id": result[0]["id"],
                          "last_id": result[-1]["id"], "has_more": False}
        else:
            from tensorfold.server.api_records import RecordNotFound
            raise RecordNotFound("the requested conversation route is unavailable")
    except RequestError as error:
        if getattr(error, "param", None) == "path":
            handler.close_connection = True
        _refused(handler, error, store=store)
    except Exception as error:
        traceback.print_exception(error)
        _send(handler, 500, {"error": server_error("logical API operation failed")})
    else:
        _send(handler, 200, result)
    return True
