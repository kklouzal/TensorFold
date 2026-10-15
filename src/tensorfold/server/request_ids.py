"""Every reply's ``x-request-id``, and the clock a request's timings start from, on both servers' handlers.

A client's own ``X-Client-Request-Id`` (OpenAI's header), else its ``x-request-id``, names the request when it is 1 to
128 characters of letters, digits and ``. _ : -``; anything else (absent, too long, other characters) gets a generated
``req-`` id, so a client value can never break the server's log line. The same id is the reply header, the
``request_id`` of the reply's ``tensorfold`` block and the id of the server's ``done`` line.
"""

from __future__ import annotations

import re
import time
import uuid
from http.server import BaseHTTPRequestHandler

MAX_ID = 128
_VALID = re.compile(r"[A-Za-z0-9._:-]{1,%d}" % MAX_ID)
CLIENT_HEADERS = ("X-Client-Request-Id", "x-request-id")


def new_request_id() -> str:
    return f"req-{uuid.uuid4().hex}"


def request_id(headers) -> str:
    """The request's id: a valid client-sent one, else a new one."""

    for name in CLIENT_HEADERS:
        value = headers.get(name) if headers is not None else None
        if isinstance(value, str) and _VALID.fullmatch(value):
            return value
    return new_request_id()


class RequestIds(BaseHTTPRequestHandler):
    """``request_id`` and ``arrived`` (perf_counter when the request's headers were read) for each request on a
    connection; every reply carries ``x-request-id``, error replies the parser sends included."""

    request_id: str | None = None
    arrived: float = 0.0

    def parse_request(self) -> bool:
        self.request_id = None                      # a reused connection's previous request must not lend its id
        self.arrived = time.perf_counter()
        parsed = super().parse_request()
        if parsed:
            self.request_id = request_id(self.headers)
        return parsed

    def send_response(self, code, message=None) -> None:
        super().send_response(code, message)
        if self.request_id is None:                 # refused before its headers were read: an id of its own
            self.request_id = new_request_id()
        self.send_header("x-request-id", self.request_id)


__all__ = ["CLIENT_HEADERS", "MAX_ID", "RequestIds", "new_request_id", "request_id"]
