"""Both servers' per-request handler bookkeeping: every reply's ``x-request-id``, the clock a request's timings start
from, the access log line (none for a successful poll), and the error object of a request whose route failed before
it answered.

A client's own ``X-Client-Request-Id`` (OpenAI's header), else its ``x-request-id``, names the request when it is 1 to
128 characters of letters, digits and ``. _ : -``; anything else (absent, too long, other characters) gets a generated
``req-`` id, so a client value can never break the server's log line. The same id is the reply header, the
``request_id`` of the reply's ``tensorfold`` block and the id of the server's ``done`` line.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler

from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.errors import server_error

MAX_ID = 128
_VALID = re.compile(r"[A-Za-z0-9._:-]{1,%d}" % MAX_ID)
CLIENT_HEADERS = ("X-Client-Request-Id", "x-request-id")
# the routes a scraper polls every few seconds (the Mac's health is also "/"): a successful poll prints no access line
POLLED = frozenset(("", "/health", "/v1/health", "/metrics", "/v1/metrics", "/stats", "/v1/stats"))


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
    """``request_id`` and ``arrived`` (perf_counter when the request line was read) for each request on a connection;
    every reply carries ``x-request-id``, error replies the parser sends included. A route that raises before its
    reply starts is answered HTTP 500 with OpenAI's ``server_error`` object (the connection then closes, and the
    server's ``handle_error`` prints the traceback, as it did before); once a reply has started, or when the client
    has gone, closing the connection is all that is left."""

    request_id: str | None = None
    arrived: float = 0.0
    replied = False                                 # this request's status line has been sent

    def handle_one_request(self) -> None:
        # a reused connection's previous request must not lend its id, not even to a refusal sent before
        # ``parse_request`` runs (a request line past 64 KiB: 414)
        self.request_id, self.replied = None, False
        try:
            super().handle_one_request()
        except Exception as exc:
            if self.replied or isinstance(exc, (ConnectionError, TimeoutError, RequestCancelled)):
                raise
            self._failed(exc)
            raise

    def _failed(self, exc: Exception) -> None:
        """A route's failure before its reply: HTTP 500 and the error object; its unread body forbids reuse."""

        self.close_connection = True
        data = json.dumps({"error": server_error(str(exc) or type(exc).__name__)}).encode()
        try:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        except OSError:                             # the client has gone as well: the failure is still raised
            pass

    def parse_request(self) -> bool:
        self.arrived = time.perf_counter()
        parsed = super().parse_request()
        if parsed:
            self.request_id = request_id(self.headers)
        return parsed

    def send_response(self, code, message=None) -> None:
        self.replied = True
        super().send_response(code, message)
        if self.request_id is None:                 # refused before its headers were read: an id of its own
            self.request_id = new_request_id()
        self.send_header("x-request-id", self.request_id)

    def log_request(self, code="-", size="-") -> None:
        # a scraper's polls would bury the requests; a poll that fails still prints
        if not (code == 200 and self.command == "GET" and self.path.split("?", 1)[0].rstrip("/") in POLLED):
            super().log_request(code, size)


__all__ = ["CLIENT_HEADERS", "MAX_ID", "POLLED", "RequestIds", "new_request_id", "request_id"]
