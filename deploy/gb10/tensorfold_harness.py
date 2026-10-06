#!/usr/bin/env python3
"""Run pinned TensorFold CUDA with exact, generation-free Harness token counting.

Invocation: python3 tensorfold_harness.py serve MODEL [native TensorFold flags]
Requires TensorFold 0.6.1 and --thinking-budget 0. Installed packages are unchanged.
The additional POST /v1/tokenize route accepts a native chat request and optional
boolean return_tokens / return_prompt fields. It counts the generation prompt,
including image expansion, even when that prompt exceeds the served context.
Native image size, count, pixel and visual-token safety limits remain in force.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import socket
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

TENSORFOLD_VERSION = "0.6.1"
BRIDGE_VERSION = 1
MAX_BODY_BYTES = 96 * 1024**2
MAX_RETURN_TOKENS = 16_384
PREPARATION_TIMEOUT_S = 60.0
BODY_TIMEOUT_S = 30.0


class _PreparationView:
    """Native _prepare policy view; context admission belongs to generation.

All state and helpers retain the initialized App's ownership. In the pinned
CUDA _prepare method, the only context lookup is the direct image preparation
limit. Removing it here permits oversized history counts without changing App,
its tokenizer, its engine, or any independent image safety limit.
"""

    def __init__(self, app: Any):
        self._app = app

    def __getattr__(self, name: str) -> Any:
        return getattr(self._app, name)

    def _context_limit(self) -> None:
        return None


class PreparationGate:
    """One preparation at a time, sixteen bounded waiters, owned by one App."""

    def __init__(self):
        self._slot = threading.BoundedSemaphore(1)
        self._waiters = threading.BoundedSemaphore(16)

    @contextmanager
    def enter(self, cancelled: Callable[[], bool], capacity_error: type[Exception],
              cancelled_error: type[Exception]):
        if not self._waiters.acquire(blocking=False):
            raise capacity_error("Harness token counting queue is full; retry shortly")
        acquired = False
        try:
            deadline = time.monotonic() + PREPARATION_TIMEOUT_S
            while not acquired:
                if cancelled():
                    raise cancelled_error("token counting client disconnected")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise capacity_error("Harness token counting capacity is busy; retry shortly")
                acquired = self._slot.acquire(timeout=min(0.1, remaining))
        finally:
            self._waiters.release()
        try:
            if cancelled():
                raise cancelled_error("token counting client disconnected")
            yield
        finally:
            self._slot.release()


def count_request(app: Any, body: Any, request_error: type[Exception],
                  capabilities: dict[str, bool] | None = None) -> dict[str, Any]:
    """Native prompt preparation only: never call run, generate, or engine prefill."""
    if not isinstance(body, dict):
        raise request_error("token counting request body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or model not in app.model_ids:
        raise request_error("model must identify a model served by this TensorFold endpoint")
    for key in ("return_tokens", "return_prompt"):
        if key in body and type(body[key]) is not bool:
            raise request_error(f"{key} must be a boolean")
    # Validate generation options exactly as native preparation does, but omit
    # the subsequent prompt-plus-reply admission check needed by App.prepare.
    problem = app._check_fields(body)
    if problem:
        raise request_error(problem)
    prepared = type(app)._prepare(_PreparationView(app), body, True)
    count = len(prepared.prompt)
    response: dict[str, Any] = {
        "count": count,
        "model": model,
        "context_length": app.effective_context_window,
        "harness_tokenizer_version": BRIDGE_VERSION,
    }
    if capabilities is not None:
        response["capabilities"] = capabilities
    # Full histories use count-only responses. Diagnostic prompt inspection is
    # bounded separately so a tokenizer probe cannot return a giant token dump.
    if body.get("return_tokens") or body.get("return_prompt"):
        if count > MAX_RETURN_TOKENS:
            raise request_error(f"prompt inspection is limited to {MAX_RETURN_TOKENS} tokens; request count only")
        if body.get("return_tokens"):
            response["tokens"] = prepared.prompt
        if body.get("return_prompt"):
            response["prompt"] = app.tok.decode(prepared.prompt, skip_special_tokens=False)
    return response


def make_handler_factory(native_factory: Callable[[Any], type]) -> Callable[[Any], type]:
    """Add one route to the native factory; every other route stays native."""
    def make_handler(app: Any) -> type:
        from tensorfold.server.cancellation import RequestCancelled, socket_cancellation
        from tensorfold.server.errors import CapacityError, RequestError, error_body

        if app.thinking_budget != 0:
            raise ValueError("Harness requires TensorFold --thinking-budget 0; budgets are selected per request")
        native_handler = native_factory(app)
        gate = PreparationGate()
        capabilities = {
            "structured_output": importlib.util.find_spec("xgrammar") is not None
            and "constraint" in inspect.signature(app.engine.generate).parameters,
            "vision": app.vision is not None,
            "video": bool(getattr(app.vision, "videos", False)),
        }

        class Handler(native_handler):
            def do_POST(self):
                route = self.path.split("?", 1)[0].rstrip("/")
                if route != "/v1/tokenize":
                    return super().do_POST()
                gone = socket_cancellation(self.connection)
                try:
                    with gate.enter(lambda: gone.cancelled, CapacityError, RequestCancelled):
                        if self.headers.get("Transfer-Encoding") is not None:
                            raise RequestError("token counting requires Content-Length, without Transfer-Encoding")
                        raw_length = self.headers.get("Content-Length")
                        try:
                            length = int(raw_length) if raw_length is not None else -1
                        except ValueError:
                            length = -1
                        if not 0 < length <= MAX_BODY_BYTES:
                            raise RequestError("token counting body must be nonempty and at most 96 MiB")
                        previous_timeout = self.connection.gettimeout()
                        self.connection.settimeout(BODY_TIMEOUT_S)
                        try:
                            payload = self.rfile.read(length)
                        finally:
                            self.connection.settimeout(previous_timeout)
                        if len(payload) != length:
                            raise RequestError("token counting request body ended before Content-Length")
                        try:
                            body = json.loads(payload, parse_constant=_invalid_json_number)
                        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
                            raise RequestError("token counting request body must be valid UTF-8 JSON") from exc
                        if gone.cancelled:
                            raise RequestCancelled("token counting client disconnected")
                        result = count_request(app, body, RequestError, capabilities)
                        if gone.cancelled:
                            raise RequestCancelled("token counting client disconnected")
                    return self._json(200, result)
                except RequestCancelled:
                    self.close_connection = True
                except (RequestError, socket.timeout) as exc:
                    self.close_connection = True
                    error = RequestError("token counting request body timed out") if isinstance(exc, socket.timeout) else exc
                    return self._json(503 if isinstance(error, CapacityError) else 400, {"error": error_body(error)})
                except Exception:
                    self.close_connection = True
                    # Do not expose request data or the private server state.
                    import traceback
                    print(f"TensorFold token preparation failed: {type(sys.exception()).__name__}", file=sys.stderr)
                    traceback.print_list(traceback.extract_tb(sys.exc_info()[2]), file=sys.stderr)
                    return self._json(500, {"error": {"type": "server_error", "message": "TensorFold token preparation failed"}})

        return Handler
    return make_handler


def _invalid_json_number(value: str) -> None:
    raise ValueError(f"invalid JSON number: {value}")


def main(argv: list[str] | None = None) -> int:
    import tensorfold
    from tensorfold import cli
    from tensorfold.cuda import http

    if tensorfold.__version__ != TENSORFOLD_VERSION:
        print(f"Harness tokenizer requires TensorFold {TENSORFOLD_VERSION}; found {tensorfold.__version__}", file=sys.stderr)
        return 2
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] != "serve":
        print("usage: tensorfold_harness.py serve MODEL [TensorFold CUDA flags]", file=sys.stderr)
        return 2
    args = cli.build_parser().parse_args(arguments)
    if args.thinking_budget != 0:
        print("Harness requires --thinking-budget 0; select thinking budgets in Harness", file=sys.stderr)
        return 2
    if args.backend == "mlx" or (args.backend == "auto" and sys.platform == "darwin"):
        print("Harness tokenizer launcher supports TensorFold CUDA only", file=sys.stderr)
        return 2
    http.make_handler = make_handler_factory(http.make_handler)
    return cli.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
