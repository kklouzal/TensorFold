"""OpenAI-compatible model, health, chat and completion endpoints with streaming, tool calls and reasoning text."""

from __future__ import annotations

import errno
import json
import math
import os
import selectors
import socket
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote

from tensorfold.cleanup import raise_failures
from tensorfold.engine import grammar
from tensorfold.server import anthropic, live, request_body, responses, token_routes
from tensorfold.server.tools import (active_tool_specs, parse_tool_calls_from_content, stream_tool_call_deltas,
                                     tool_choice_requires_call)
from tensorfold.server.decisions import DecisionError
from tensorfold.server.failure_capture import redact_images
from tensorfold.server.errors import (ModelNotFound, RequestError, error_body, invalid_request, retry_headers,
                                     server_error, status_of, unknown_route)
from tensorfold.server.request_options import parse_numbers, thinking_fields
from tensorfold.server.probabilities import probability_options
from tensorfold.server.messages import normalize_messages, validate_modalities
from tensorfold.server.tool_policy import ToolCallPolicy
from tensorfold.server.cancellation import RequestCancelled, socket_cancellation
from tensorfold.server import metrics, stats
from tensorfold.server.request_ids import RequestIds
from tensorfold.server.stacks import Rearming
from tensorfold.server.request_body import read_body

# TENSORFOLD_REQUEST_LOG=path appends every request body (one JSON a line), for exact replays of real traffic
_REQUEST_LOG = os.environ.get("TENSORFOLD_REQUEST_LOG", "")


def _memory(reset_peak: bool, *, admission: Any = None) -> dict[str, int]:
    """MLX's memory in bytes: live buffers, its cache of freed ones, and the peak (since the last reset)."""

    if admission is not None:
        return admission.memory_snapshot(reset_peak)
    try:
        import mlx.core as mx
    except ImportError:          # the CUDA server
        return {}
    memory = {"active": int(mx.get_active_memory()), "cache": int(mx.get_cache_memory()),
              "peak": int(mx.get_peak_memory())}
    if reset_peak:
        mx.reset_peak_memory()
    return memory


class Server(ThreadingHTTPServer):
    """Own accepted sockets and join their handlers before model teardown.

    Closing stops admission, cancels request work and interrupts socket IO. A
    failed or interrupted drain retains the journal for an explicit retry.
    Callers must retain the application until ``handlers_drained`` is true.
    There is no implicit timeout or worker-count limit. ``max_connections``
    optionally pauses acceptance while that many handler owners are retained;
    unaccepted connections wait in the existing bounded listen backlog.
    """

    request_queue_size = 128
    daemon_threads = False

    def __init__(self, *args: Any, max_connections: int | None = None, **kwargs: Any) -> None:
        if max_connections is not None and (type(max_connections) is not int or max_connections <= 0):
            raise ValueError("max_connections must be a positive integer or None")
        handler = args[1] if len(args) > 1 else kwargs.get("RequestHandlerClass")
        if getattr(handler, "requires_finite_limits", False):
            from tensorfold.server.request_limits import require_finite_endpoint_limits

            require_finite_endpoint_limits(max_pending_requests=handler.pending_request_limit,
                                           max_http_connections=max_connections)
        self.max_connections = max_connections
        self._capacity_changed = threading.Event() if max_connections is not None else None
        self._capacity_poll = 0.5
        if max_connections is not None:
            self._accept_gate = threading.Lock()
            self._accept_without_limit = self._handle_request_noblock
            self._handle_request_noblock = self._handle_request_with_capacity
            self.service_actions = self._service_actions_bounded
        self._connections: dict[Any, dict[str, Any]] = {}
        self._connection_lock = threading.RLock()
        self._close_lock = threading.Lock()
        self.stopping = threading.Event()
        self._serve_stop = threading.Event()
        self._serving_thread = None
        self._handler_failure = None
        self.handlers_drained = False
        super().__init__(*args, **kwargs)

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        if self.max_connections is not None:
            if type(poll_interval) not in (int, float) or not math.isfinite(poll_interval) or poll_interval <= 0:
                raise ValueError("bounded HTTP acceptance requires a positive finite poll_interval")
            self._capacity_poll = float(poll_interval)
        lease = threading.Lock()
        with lease:
            try:
                with self._connection_lock:
                    serving = self._serving_thread
                    if serving is not None and not serving[1].locked():
                        self._serving_thread = None
                    if self.stopping.is_set() or self._serving_thread is not None:
                        raise RuntimeError("this server has closed or is already serving")
                    self._serve_stop.clear()
                    self._serving_thread = threading.current_thread(), lease
                # Match socketserver's one-descriptor polling strategy while
                # owning the loop scope before any interruptible startup.
                selector_type = getattr(selectors, "PollSelector", selectors.SelectSelector)
                with selector_type() as selector:
                    selector.register(self, selectors.EVENT_READ)
                    while not self._serve_stop.is_set():
                        ready = selector.select(poll_interval)
                        if self._serve_stop.is_set():
                            break
                        if ready:
                            self._handle_request_noblock()
                        self.service_actions()
            finally:
                with self._connection_lock:
                    serving = self._serving_thread
                    if serving is not None and serving[1] is lease:
                        self._serving_thread = None

    def shutdown(self) -> None:
        """Stop a distinct serving scope, including interrupted startup."""
        with self._connection_lock:
            serving = self._serving_thread
            if serving is not None and serving[0] is threading.current_thread() and serving[1].locked():
                raise RuntimeError("the server loop cannot join itself")
            self._serve_stop.set()
            if self._capacity_changed is not None:
                self._capacity_changed.set()
        if serving is not None:
            # Native with cleanup releases this even if Python finalization
            # was interrupted before it could publish loop completion.
            with serving[1]:
                pass

    def handle_request(self) -> None:
        """Retain stdlib's single-request path and its completed-worker reclamation."""
        primary = None
        try:
            super().handle_request()
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                self.service_actions()
            except BaseException as cleanup:
                if primary is not None and primary is not cleanup:
                    raise primary from cleanup
                raise

    def _handle_request_with_capacity(self) -> None:
        """Wait without accepting at capacity; reselect after an owner retires.

        The connection lock linearizes full-state observation and clearing the
        notification against handler completion and shutdown. The existing
        poll interval also observes failed native handler/socket retirement.
        Reselecting avoids accepting from stale readiness after this wait.
        """
        with self._accept_gate:
            with self._connection_lock:
                if self.stopping.is_set() or self._serve_stop.is_set():
                    return
                full = len(self._connections) >= self.max_connections
                retired = full and any(j["closed"] or "close_error" in j for j in self._connections.values())
                if full and not retired:
                    self._capacity_changed.clear()
            if full:
                if not retired:
                    self._capacity_changed.wait(self._capacity_poll)
                self.service_actions()
                return
            self._accept_without_limit()

    def connection_snapshot(self) -> dict[str, int | None]:
        """Handler owners include sockets whose fallible retirement is pending."""
        with self._connection_lock:
            return {"in_use": len(self._connections), "limit": self.max_connections}

    @staticmethod
    def _interrupt(request: socket.socket) -> None:
        try:
            request.shutdown(socket.SHUT_RDWR)
        except OSError as error:
            # An already disconnected/closed socket has no outstanding IO.
            if error.errno not in (errno.ENOTCONN, errno.EBADF):
                raise

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        with self._connection_lock:
            if self.stopping.is_set():
                super().shutdown_request(request)
                return
            journal = {"entered": False, "start_error": None, "thread": None, "closed": False,
                       "start_attempted": False, "lease": None, "entry": None}
            self._connections[request] = journal
            try:
                journal["lease"] = threading.Lock()
                journal["entry"] = threading.Event()
                worker = threading.Thread(target=self.process_request_thread,
                                          args=(request, client_address), daemon=False)
                journal["thread"] = worker
                journal["start_attempted"] = True
                worker.start()
                journal["entry"].wait()         # entry publishes before the worker asks for this connection lock
            except BaseException as error:
                # start() can be interrupted after spawning but before return.
                # Do not assume absence of an ident means absence of a worker.
                journal["start_error"] = error
                self.stopping.set()
                raise

    def shutdown_request(self, request: socket.socket) -> None:
        with self._connection_lock:
            owned = request in self._connections
        if owned:
            self._interrupt(request)             # the worker alone closes its IO
        else:
            super().shutdown_request(request)

    def process_request_thread(self, request: socket.socket, client_address: Any) -> None:
        journal = self._connections[request]
        with journal["lease"]:
            try:
                journal["entry"].set()
                with self._connection_lock:
                    journal["entered"] = True
                if not self.stopping.is_set():
                    try:
                        self.finish_request(request, client_address)
                    except Exception:
                        self.handle_error(request, client_address)
            except BaseException as error:
                with self._connection_lock:
                    self._handler_failure = error
                    self.stopping.set()
            finally:
                try:
                    super().shutdown_request(request)
                except BaseException as error:
                    with self._connection_lock:
                        journal["close_error"] = error
                else:
                    with self._connection_lock:
                        journal["closed"] = True
                finally:
                    if self._capacity_changed is not None:
                        try:
                            self._capacity_changed.set()
                        except BaseException as error:
                            with self._connection_lock:
                                try:
                                    raise_failures(self._handler_failure, [error])
                                except BaseException as failure:
                                    self._handler_failure = failure
                                self.stopping.set()

    def service_actions(self) -> None:
        # Reap completed handlers at the server-loop control point. Retiring
        # from inside a handler would lose its join ownership before exit.
        with self._connection_lock:
            for request, journal in tuple(self._connections.items()):
                worker = journal["thread"]
                if journal["closed"] and not worker.is_alive():
                    with journal["lease"]:
                        pass
                    worker.join()
                    self._connections.pop(request)
            failure = self._handler_failure
        if failure is not None:
            raise failure

    def _service_actions_bounded(self) -> None:
        # Closed socket publication precedes the wakeup. Wait for the native
        # target scope outside the journal lock, including failed wakeup handling.
        with self._connection_lock:
            retired = [(request, journal) for request, journal in self._connections.items() if journal["closed"]]
        for request, journal in retired:
            with journal["lease"]:
                pass
            journal["thread"].join()
            with self._connection_lock:
                if self._connections.get(request) is journal:
                    self._connections.pop(request)
        with self._connection_lock:
            failure = self._handler_failure
            if failure is None:
                failure = next((j["close_error"] for j in self._connections.values() if "close_error" in j), None)
        if failure is not None:
            raise failure

    def server_close(self) -> None:
        with self._connection_lock:
            current = threading.current_thread()
            serving = self._serving_thread
            if ((serving is not None and serving[0] is current and serving[1].locked())
                    or any(j["thread"] is current for j in self._connections.values())):
                raise RuntimeError("the server loop or a connection handler cannot join itself")
        with self._close_lock:
            with self._connection_lock:
                self.stopping.set()
                if self._capacity_changed is not None:
                    self._capacity_changed.set()
                serving = self._serving_thread
                journals = tuple(self._connections.items())
            if serving is not None:
                self.shutdown()                  # only a distinct serving thread
            super().server_close()               # no inherited worker registry
            for request, _ in journals:
                self._interrupt(request)
            for request, journal in journals:
                if not journal["start_attempted"]:
                    # Construction cannot publish a native actor. Its socket
                    # was journaled first and remains owned through cleanup.
                    super().shutdown_request(request)
                    with self._connection_lock:
                        self._connections.pop(request)
                    continue
                if not journal["entry"].is_set():
                    raise RuntimeError("connection start did not settle; its owners remain retained") from journal["start_error"]
                # Thread.join can mark an active target stopped after SIGINT
                # on CPython. The native handler/socket scope retires first.
                with journal["lease"]:
                    pass
                journal["thread"].join()
                with self._connection_lock:
                    if request in self._connections:  # failed worker socket cleanup
                        if not journal["closed"]:
                            super().shutdown_request(request)
                        self._connections.pop(request)
            with self._connection_lock:
                if self._connections:
                    raise RuntimeError("accepted connection owners remain after shutdown")
                self.handlers_drained = True
                failure = self._handler_failure
            if failure is not None:
                raise failure


def requested_model(app: Any, body: Any) -> str:
    """The id a reply names: the request's ``model`` when this endpoint answers to it (``--name`` or an alias), the
    served name when the request omits it or sends null. Any other id is ModelNotFound (HTTP 404); a model that is
    not a string is a RequestError."""

    name = str(getattr(app, "served_name", "") or getattr(app, "served", "") or "")
    asked = body.get("model") if isinstance(body, dict) else None
    if asked is None:
        return name
    if not isinstance(asked, str):
        raise RequestError("model must be a string naming a served model")
    if getattr(app, "model_policies", False):       # GLM: ``ID@POLICY`` names a served id and a draft policy
        asked = asked.split("@", 1)[0]
    ids = getattr(app, "model_ids", None) or [name]
    if asked not in ids:
        raise ModelNotFound(f"The model `{_shown(asked)}` does not exist on this server, which serves "
                            f"{', '.join(ids)} (GET /v1/models)")
    return asked


SHOWN_ID = 256          # characters of an unknown model id its 404 repeats: an error stays small whatever is sent


def _shown(model: str) -> str:
    return model if len(model) <= SHOWN_ID else f"{model[:SHOWN_ID]}... ({len(model)} characters)"


def model_route(path: str) -> str | None:
    """A models path's model id ("" for the list), else None. An id may hold slashes (``org/name``) and %-escapes."""

    path = path.split("?", 1)[0].rstrip("/")
    for prefix in ("/v1/models", "/models"):
        if path == prefix:
            return ""
        if path.startswith(prefix + "/"):
            return unquote(path[len(prefix) + 1:])
    return None


def models_reply(app: Any, model_id: str, max_model_len: int | None) -> tuple[int, dict[str, Any]]:
    """(status, body) of GET /v1/models (``model_id`` "") or GET /v1/models/{model}: OpenAI's Model, plus vLLM's
    ``max_model_len`` (the served context length, null where none is known); ``created`` is this server's start, so a
    listing is stable while it runs. An id this endpoint does not serve is 404 ``model_not_found``."""

    def entry(name: str) -> dict[str, Any]:
        return {"id": name, "object": "model", "created": int(stats.STARTED_UNIX_S), "owned_by": "tensorfold",
                "max_model_len": max_model_len}

    if not model_id:
        return 200, {"object": "list", "data": [entry(name) for name in app.model_ids]}
    if model_id not in app.model_ids:
        return 404, {"error": error_body(ModelNotFound(f"The model `{_shown(model_id)}` does not exist on this "
                                                       "server (GET /v1/models)"))}
    return 200, entry(model_id)


def served_model_ids(served_name: str, aliases: list[str] | None = None) -> list[str]:
    """Return the OpenAI model ids this endpoint advertises."""

    ids: list[str] = []
    for value in [served_name, *(aliases or [])]:
        model_id = str(value or "").strip()
        if model_id and model_id not in ids:
            ids.append(model_id)
    return ids


def wants_usage_chunk(body: Any) -> bool:
    """Whether the request asked for the spec's usage-only chunk before [DONE] (stream_options.include_usage)."""

    options = body.get("stream_options") if isinstance(body, dict) else None
    return bool(isinstance(options, dict) and options.get("include_usage"))


def make_handler(app: Any) -> type[BaseHTTPRequestHandler]:
    defer_headers = (getattr(app, "max_pending_requests", None) is not None
                     or getattr(getattr(app, "scheduler", None), "max_engine_calls", None) is not None)
    class Handler(RequestIds, Rearming):              # USR1's stack dump armed again after each request
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[tensorfold] {self.address_string()} {format % args}")

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for name, value in retry_headers(status):        # a capacity refusal: when to retry
                self.send_header(name, value)
            if self.close_connection:                    # so a pooling client does not reuse the socket
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _discard_body(self) -> bool:
            """Read a refused request's body, so it cannot reach the next request on this connection."""

            try:
                request_body.discard(self)
            except RequestError as error:
                self._send_json({"error": error_body(error)}, status=400)
                return False
            return True

        def _route(self) -> str:
            # Tolerate query strings, trailing slashes and client URLs with or without the /v1 prefix.
            return self.path.split("?", 1)[0].rstrip("/")

        def do_GET(self) -> None:
            if not self._discard_body():
                return
            route = self._route()
            if responses.route(route):
                return responses.get(self, app, responses.route(route))
            if route in {"/metrics", "/v1/metrics"}:
                return metrics.send(self, app)
            if route in stats.ROUTES:               # the Mac's own: MLX memory (no peak reset) and the live meters
                scheduler = getattr(app, "scheduler", None)
                return stats.send(self, app, {"mlx": {
                    "memory": _memory(False, admission=getattr(app, "prompt_memory", None)),
                    "live": live.snapshot(scheduler) if hasattr(scheduler, "decoded") else None}})
            if route in {"", "/health"}:
                if getattr(getattr(app, "auth", None), "enabled", False):
                    self._send_json({"status": "ok"})
                    return
                warmup = getattr(app, "warmup_result", None)
                failure = warmup.exception() if warmup is not None and warmup.done() else None
                self._send_json(
                    {
                        "status": "ok",
                        "model": app.served_name,
                        "model_ids": app.model_ids,
                        "max_batch_size": app.max_batch_size,
                        "warming": bool(getattr(app, "warming", False)),
                        **({"warmup_error": f"saved-prefix warmup failed: {failure.error_type}"} if failure else {}),
                        "memory": _memory("reset_peak=1" in self.path, admission=getattr(app, "prompt_memory", None)),
                        **({"admission": capacities} if (capacities := metrics.capacity_snapshot(app, server=self.server)) else {}),
                        **({"live": live.snapshot(app.scheduler)} if getattr(app, "scheduler", None) is not None else {}),
                    }
                )
                return
            model_id = model_route(route)
            if model_id is not None or route.endswith("/models"):    # any path ending in /models lists them
                status, payload = models_reply(app, model_id or "", int(getattr(app, "context_window", 0) or 0) or None)
                self._send_json(payload, status)
                return
            self._send_json({"error": unknown_route("GET", self.path)}, status=404)

        def _legacy_prompt_to_text(self, prompt: Any) -> str:
            if isinstance(prompt, str):
                return prompt
            if isinstance(prompt, list):
                if all(isinstance(token_id, int) for token_id in prompt):
                    with app.tokenizer_lock:
                        return app.tokenizer.decode([int(token_id) for token_id in prompt])
                if len(prompt) != 1:            # OpenAI's batch of prompts wants a choice each: refused, as on CUDA
                    raise RequestError("prompt must be one prompt a request (a string or a list of token ids); "
                                       "send each prompt as its own request")
                return self._legacy_prompt_to_text(prompt[0])
            if prompt is None:
                return ""
            return str(prompt)

        def _legacy_prompt(self, prompt: Any) -> str | list[int]:
            """A completion's prompt as the model reads it: token ids as given, anything else as text."""

            if isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
                with app.tokenizer_lock:
                    tokenizer = app.tokenizer
                    if not hasattr(type(tokenizer), "__len__"):
                        tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
                    try:
                        vocab = len(tokenizer)
                    except TypeError:
                        vocab = int(tokenizer.vocab_size)
                if any(type(t) is not int or not 0 <= t < vocab for t in prompt):
                    raise RequestError(f"prompt token ids must be integers in the valid range 0 to {vocab - 1}")
                return list(prompt)
            return self._legacy_prompt_to_text(prompt)

        def do_DELETE(self) -> None:
            if not self._discard_body():
                return
            rid = responses.route(self._route())
            if rid is None:
                self._send_json({"error": unknown_route("DELETE", self.path)}, status=404)
                return
            responses.delete(self, app, rid)

        def do_POST(self) -> None:
            route = self._route()
            if route.endswith("/decisions"):
                return self._post_decisions(app)
            if anthropic.route(self.path):
                return anthropic.post(self, app)
            if responses.route(route) == "":         # a Response: this handler's chat completion, translated
                return responses.post(self, app)
            if responses.route(route) == "input_tokens":
                return responses.input_tokens(self, app)
            if route in token_routes.ROUTES:         # vLLM's /tokenize and /detokenize
                return self._post_tokenizer(route.endswith("/detokenize"))

            is_chat_completion = route.endswith("/chat/completions")
            is_text_completion = route.endswith("/completions") and not is_chat_completion
            if not is_chat_completion and not is_text_completion:
                if not self._discard_body():
                    return
                self._send_json({"error": unknown_route("POST", self.path)}, status=404)
                return
            field = "messages" if is_chat_completion else "prompt"   # the field an error's code names (OpenAI's param)

            try:
                body = parse_numbers(json.loads(read_body(self) or b"{}"))
                validate_modalities(body)
                probability_options(body)
                named = requested_model(app, body)      # the id the request asked for, as vLLM names it
                if _REQUEST_LOG and body.get("priority") != "background":   # batch jobs are not client traffic
                    with open(_REQUEST_LOG, "a") as handle:
                        handle.write(json.dumps(redact_images(body)) + "\n")
                raw_kw: dict[str, Any] = {}
                if is_chat_completion:
                    messages = normalize_messages(body.get("messages"), allow_images=getattr(app, "vision", None) is not None)
                    tools = active_tool_specs(body.get("tools"), body.get("tool_choice"))
                elif isinstance(body.get("messages"), list) and body["messages"]:
                    messages, tools = normalize_messages(body["messages"], allow_images=getattr(app, "vision", None) is not None), []
                elif getattr(app, "accepts_raw_prompt", False):
                    # a text completion reads its prompt raw, as vLLM and mlx_lm do: no chat template, no think block
                    messages, tools = [], []
                    raw_kw["prompt"] = self._legacy_prompt(body.get("prompt", ""))
                else:
                    messages = [{"role": "user", "content": self._legacy_prompt_to_text(body.get("prompt", ""))}]
                    tools = []
                if body.get("presence_penalty") or body.get("frequency_penalty") \
                        or body.get("repetition_penalty") not in (None, 1, 1.0):
                    raise RequestError("presence, frequency and repetition penalties are applied by the CUDA engines "
                                       "only: send them as 0, 0 and 1 (or omit them) on the Metal backend")
                max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
                temperature = float(body.get("temperature") or 0.0)
                # Preserve raw sampling and scheduling options; an absent temperature differs from temperature zero.
                sampling_fields = {k: body[k] for k in ("temperature", "top_p", "top_k", "min_p", "seed", "priority",
                                                        "draft", "thinking_budget", "ignore_eos", "stop",
                                                        *grammar.FIELDS)
                                   if k in body}
                problem = grammar.refusal(body, app)        # compiled before a stream's headers: a bad grammar is a 400
                if problem:
                    raise RequestError(problem)
                if "priority" not in body and body.get("service_tier") == "flex":
                    sampling_fields["priority"] = "background"      # OpenAI's flex tier: this server's background
                if tools and tool_choice_requires_call(body.get("tool_choice")):
                    sampling_fields["tool_call_required"] = True     # the engine opens the answer with a call
                sampling_fields.update(thinking_fields(body, getattr(app, "effort_levels", frozenset())))
                sampling_kw = ({"sampling": sampling_fields}
                               if getattr(app, "accepts_sampling", False) else {})
                if getattr(app, "accepts_cancellation", False):
                    sampling_kw["cancellation"] = socket_cancellation(self.connection, stopping=getattr(self.server, "stopping", None))
                if getattr(app, "accepts_request_ids", False):      # the reply's request id; timings from arrival
                    sampling_kw.update(request_id=self.request_id, arrived=self.arrived)
                stream = bool(body.get("stream", False))
                separate_usage = wants_usage_chunk(body)      # usage then rides its own chunk before [DONE]
                tool_policy = ToolCallPolicy(body)
            except RequestError as exc:
                self._send_json({"error": error_body(exc, field)}, status=status_of(exc))
                return
            except Exception as exc:
                self._send_json({"error": invalid_request(str(exc) or type(exc).__name__)}, status=400)
                return

            completion_id = (
                f"chatcmpl-{uuid.uuid4().hex}"
                if is_chat_completion
                else f"cmpl-{uuid.uuid4().hex}"
            )
            created = int(time.time())

            def usage_from_reply(reply: dict[str, Any]) -> dict[str, Any]:
                return {
                    "prompt_tokens": reply["prompt_tokens"],
                    "completion_tokens": reply["completion_tokens"],
                    "total_tokens": reply["prompt_tokens"] + reply["completion_tokens"],
                    "prompt_tokens_details": {"cached_tokens": reply["cached_tokens"]},
                    "completion_tokens_details": {"reasoning_tokens": reply.get("reasoning_tokens", 0)},
                }

            def response_extras(reply: dict[str, Any]) -> dict[str, Any]:
                extras: dict[str, Any] = {
                    "exact_mode": app.exact_mode.get("mode", "target-verified")
                }
                if reply.get("stop_sequence") is not None:
                    extras["stop_sequence"] = reply["stop_sequence"]
                if reply.get("batch_size"):
                    extras["tensorfold"] = {
                        "batch_size": reply["batch_size"],
                        "seconds": reply["seconds"],
                    }
                if reply.get("runtime"):
                    extras["tensorfold"] = reply["runtime"]
                if reply.get("speculative"):
                    extras["speculative"] = reply["speculative"]
                if reply.get("pass_economics"):
                    extras["pass_economics"] = reply["pass_economics"]
                return extras

            def attach_tool_calls(reply: dict[str, Any]) -> dict[str, Any]:
                return tool_policy.finish(reply, tools, parse_tool_calls_from_content)

            def stream_chunk(
                delta: str | dict[str, Any] = "",
                finish_reason: str | None = None,
            ) -> dict[str, Any]:
                if is_text_completion:
                    choice = {"index": 0, "text": delta, "finish_reason": finish_reason, "logprobs": None}
                else:
                    choice = {"index": 0,
                              "delta": delta if isinstance(delta, dict) else ({"content": delta} if delta else {}),
                              "finish_reason": finish_reason}
                frame = {"id": completion_id, "object": "text_completion" if is_text_completion else
                         "chat.completion.chunk", "created": created, "model": named, "choices": [choice]}
                if separate_usage:              # the spec: every chunk but the usage chunk has a null usage
                    frame["usage"] = None
                return frame

            try:
                if stream:
                    opened = False

                    def open_stream() -> None:
                        nonlocal opened
                        if not opened:
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream")
                            self.send_header("Cache-Control", "no-cache")
                            self.send_header("Connection", "close")
                            self.end_headers()
                            opened = True

                    if not defer_headers:
                        open_stream()

                    def emit(payload: dict[str, Any]) -> None:
                        if not opened:
                            open_stream()
                        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
                        self.wfile.flush()

                    def finish_stream(
                        finish_reason: str | None,
                        *,
                        error: BaseException | None = None,
                        extras: dict[str, Any] | None = None,
                        usage: dict[str, Any] | None = None,
                    ) -> None:
                        if error is not None:
                            payload = {"error": server_error(str(error))}
                        else:
                            payload = stream_chunk("", finish_reason or "length")
                            if extras:
                                payload.update(extras)
                        emit(payload)
                        if usage is not None:                   # the spec's own usage chunk, with the stats
                            emit({**stream_chunk(), "choices": [], "usage": usage,
                                  **({"tensorfold": extras["tensorfold"]} if extras and "tensorfold" in extras else {})})
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()

                    def on_delta(delta: str | dict[str, Any]) -> None:
                        # Text completions carry content strings only, excluding reasoning deltas as non-streamed replies do.
                        if is_text_completion and not isinstance(delta, str):
                            return
                        if defer_headers and is_chat_completion and not opened:
                            emit(stream_chunk({"role": "assistant"}))
                        emit(stream_chunk(delta))

                    try:
                        if tools:
                            streamed = [False]

                            def on_prose(delta: str | dict[str, Any]) -> None:
                                delta = tool_policy.delta(delta)
                                if not delta:
                                    return
                                if not streamed[0]:
                                    streamed[0] = True
                                    emit(stream_chunk({"role": "assistant"}))
                                emit(stream_chunk(delta))

                            extra = (
                                {"on_delta": on_prose}
                                if getattr(app, "streams_prose_with_tools", False) else {}
                            )
                            reply = attach_tool_calls(
                                app.chat(
                                    messages,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    tools=tools,
                                    **extra,
                                    **sampling_kw,
                                    **raw_kw,
                                )
                            )
                            tail = tool_policy.flush()
                            if tail:
                                if not streamed[0]:
                                    streamed[0] = True
                                    emit(stream_chunk({"role": "assistant"}))
                                emit(stream_chunk(tail))
                            tool_calls = reply.get("tool_calls")
                            if tool_calls and not reply.get("tool_calls_streamed"):
                                # (calls the app already streamed as they were written are not sent twice)
                                emit(stream_chunk({"role": "assistant"}))
                                for delta in stream_tool_call_deltas(tool_calls):
                                    emit(stream_chunk(delta))
                            elif reply.get("content") and not streamed[0]:
                                emit(stream_chunk(str(reply["content"])))
                        else:
                            if is_chat_completion and not defer_headers:
                                emit(stream_chunk({"role": "assistant"}))
                            reply = app.chat(
                                messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                on_delta=on_delta,
                                **sampling_kw,
                                **raw_kw,
                            )
                    except (BrokenPipeError, ConnectionResetError):
                        raise
                    except RequestCancelled:
                        self.close_connection = True
                        return
                    except RequestError as exc:
                        if not opened:
                            self._send_json({"error": error_body(exc, field)}, status=status_of(exc))
                            return
                        emit({"error": error_body(exc, field)})
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        return
                    except Exception as exc:
                        print(
                            f"[tensorfold] stream error: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        traceback.print_exc()
                        if not opened:
                            self._send_json({"error": server_error(str(exc))}, status=500)
                            return
                        try:
                            finish_stream(None, error=exc)
                        except BrokenPipeError:
                            pass
                        return
                    extras = response_extras(reply)
                    if not opened and is_chat_completion:
                        emit(stream_chunk({"role": "assistant"}))
                    counted = "prompt_tokens" in reply and "completion_tokens" in reply
                    usage = usage_from_reply({"cached_tokens": 0, **reply}) if counted else None
                    if usage is not None and not separate_usage:
                        # Clients that time the stream count tokens from here.
                        extras["usage"] = usage
                    finish_stream(reply.get("finish_reason") or "length", extras=extras,
                                  usage=usage if separate_usage else None)
                    return

                reply = attach_tool_calls(
                    app.chat(
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        tools=tools or None,
                        **sampling_kw,
                        **raw_kw,
                    )
                )
                if is_text_completion:
                    self._send_json(
                        {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": named,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": reply["content"],
                                    "finish_reason": reply["finish_reason"],
                                    "logprobs": None,
                                }
                            ],
                            "usage": usage_from_reply(reply),
                            **response_extras(reply),
                        }
                    )
                    return

                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": None if reply.get("tool_calls") else reply["content"],
                }
                if reply.get("reasoning"):
                    message["reasoning_content"] = reply["reasoning"]
                if reply.get("tool_calls"):
                    message["tool_calls"] = reply["tool_calls"]
                self._send_json(
                    {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": named,
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "logprobs": None,
                                "finish_reason": reply["finish_reason"],
                            }
                        ],
                        "usage": usage_from_reply(reply),
                        **response_extras(reply),
                    }
                )
            except (BrokenPipeError, ConnectionResetError, RequestCancelled):
                self.close_connection = True
            except RequestError as exc:          # a capacity refusal is a 503 here too, as when streaming
                self._send_json({"error": error_body(exc, field)}, status=status_of(exc))
            except Exception as exc:  # surface runner errors to the client
                print(f"[tensorfold] request error: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                try:
                    self._send_json({"error": server_error(str(exc))}, status=500)
                except Exception:
                    pass


        def _post_tokenizer(self, detokenize: bool) -> None:
            try:
                body = json.loads(read_body(self) or b"{}")
                requested_model(app, body)
                reply = token_routes.detokenize(app, body) if detokenize else token_routes.tokenize(app, body)
            except RequestError as exc:
                self._send_json({"error": error_body(exc)}, status=status_of(exc))
                return
            except Exception as exc:  # noqa: BLE001 - a body the tokenizer cannot read is a client error
                self._send_json({"error": invalid_request(str(exc))}, status=400)
                return
            self._send_json(reply)

        def _post_decisions(self, app: Any) -> None:
            decide = getattr(app, "decisions", None)
            if decide is None:
                if not self._discard_body():
                    return
                self._send_json({"error": unknown_route("POST", self.path)}, status=404)
                return
            try:
                body = parse_numbers(json.loads(read_body(self) or b"{}"))
                if not isinstance(body, dict):
                    raise RequestError("request body must be an object")
                requested_model(app, body)
            except RequestError as exc:
                self._send_json({"error": error_body(exc)}, status=status_of(exc))
                return
            except Exception as exc:  # noqa: BLE001 - a bad body is a client error
                self._send_json({"error": invalid_request(str(exc))}, status=400)
                return
            try:
                payload = decide(body)
            except RequestError as exc:           # a capacity refusal is a 503, as on the chat route
                self._send_json({"error": error_body(exc)}, status=status_of(exc))
                return
            except DecisionError as exc:
                self._send_json({"error": invalid_request(str(exc))}, status=400)
                return
            except Exception as exc:  # a scoring failure is the server's, not a bad body
                print(f"[tensorfold] request error: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                try:
                    self._send_json({"error": server_error(str(exc))}, status=500)
                except Exception:
                    pass
                return
            self._send_json(payload)

    from tensorfold.server.auth_http import handler
    return handler(Handler, app)
