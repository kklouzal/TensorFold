"""The CUDA server's HTTP side: OpenAI routes over ``App`` (tensorfold.cuda.server), streamed or not."""
from __future__ import annotations

import json
import time
import traceback
import uuid
from typing import TYPE_CHECKING, Any

from tensorfold.cuda import health
from tensorfold.server import anthropic, metrics, request_body, responses, runtime, stats, token_routes
from tensorfold.server.cancellation import RequestCancelled, socket_cancellation
from tensorfold.server.decisions import DecisionError
from tensorfold.server.errors import (RequestError, error_body, invalid_request, retry_headers, server_error, status_of,
                                     unknown_route)
from tensorfold.server.http import Server, model_route, models_reply, requested_model, wants_usage_chunk
from tensorfold.server.request_body import read_body
from tensorfold.server.request_ids import RequestIds
from tensorfold.server.stacks import Rearming
from tensorfold.server.token_routes import flag

if TYPE_CHECKING:
    from tensorfold.cuda.server import App


def usage_of(result: dict[str, Any]) -> dict[str, Any]:
    """A reply's usage as the Mac server reports it, the prompt tokens found cached included."""

    return {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            "prompt_tokens_details": {"cached_tokens": result.get("cached_tokens", 0)},
            "completion_tokens_details": {"reasoning_tokens": result.get("reasoning_tokens", 0)}}


def _error_message(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


def _log_error(exc: BaseException) -> None:
    print(f"[tensorfold] request error: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exception(exc)


def make_handler(app: App):
    defer_headers = getattr(app, "max_pending_requests", None) is not None
    class Handler(RequestIds, Rearming):              # USR1's stack dump armed again after each request
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):    # one line a request (none for a successful poll), as on the Mac
            print(f"[tensorfold] {self.address_string()} {fmt % args}", flush=True)

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for name, value in retry_headers(code):        # a capacity refusal: when to retry
                    self.send_header(name, value)
                if self.close_connection:                # so a pooling client does not reuse the socket
                    self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):          # the client has gone
                self.close_connection = True

        def _discard_body(self) -> bool:
            """Read a refused request's body, so it cannot reach the next request on this connection."""

            try:
                request_body.discard(self)
            except RequestError as error:
                self._json(400, {"error": error_body(error)})
                return False
            return True

        def _refuse(self, exc: RequestError, field: str | None = None) -> None:
            """A refusal before any reply: 400, 404 for an unknown model, 503 (Retry-After) for capacity."""

            self._json(status_of(exc), {"error": error_body(exc, field)})

        def _stream_error(self, error: dict[str, Any]) -> None:
            """End an open stream with an error event and ``[DONE]``, as the MLX server does."""

            try:
                self.wfile.write(f"data: {json.dumps({'error': error})}\n\ndata: [DONE]\n\n".encode())
                self.wfile.flush()
            except OSError:
                pass
            self.close_connection = True

        def do_GET(self):
            if not self._discard_body():
                return
            route = self.path.split("?", 1)[0].rstrip("/")
            if route in ("/metrics", "/v1/metrics"):
                return metrics.send(self, app)
            if route in stats.ROUTES:
                return stats.send(self, app)
            if route == runtime.ROUTE:
                reading = runtime.snapshot(app)
                if reading is None:
                    return self._json(404, {"error": unknown_route("GET", self.path)})
                return self._json(200, reading)
            model_id = model_route(route)
            if model_id is not None:
                self._json(*models_reply(app, model_id, getattr(app, "max_model_len", None)))
            elif route in ("/health", "/v1/health"):
                if getattr(getattr(app, "auth", None), "enabled", False):
                    return self._json(200, {"status": "ok"})
                snapshot = health.of(app).snapshot(app)
                capacities = metrics.capacity_snapshot(app, server=self.server)
                if capacities:
                    snapshot["admission"] = capacities
                self._json(200, snapshot)
            elif responses.route(self.path):
                responses.get(self, app, responses.route(self.path))
            else:
                self._json(404, {"error": unknown_route("GET", self.path)})

        def do_DELETE(self):
            if not self._discard_body():
                return
            rid = responses.route(self.path)
            if rid is None:
                return self._json(404, {"error": unknown_route("DELETE", self.path)})
            responses.delete(self, app, rid)

        def do_POST(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == runtime.ROUTE:
                try:
                    body = json.loads(read_body(self, limit=runtime.MAX_BODY_BYTES) or b"{}")
                    reply = runtime.request(app, body)
                except RequestError as error:
                    return self._refuse(error)
                except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
                    return self._json(400, {"error": invalid_request("the request body is not JSON")})
                except Exception as error:
                    _log_error(error)
                    return self._json(500, {"error": server_error(_error_message(error))})
                return self._json(200, reply)
            if path.endswith("/decisions"):
                return self._post_decisions()
            if anthropic.route(self.path):
                return anthropic.post(self, app)
            if responses.route(self.path) == "":         # a Response: this handler's chat completion, translated
                return responses.post(self, app)
            if responses.route(self.path) == "input_tokens":
                return responses.input_tokens(self, app)
            chat = path.endswith("/chat/completions")
            tokenizer = path in token_routes.ROUTES
            if not chat and not tokenizer and not path.endswith("/completions"):
                if not self._discard_body():
                    return
                return self._json(404, {"error": unknown_route("POST", self.path)})
            try:
                body = json.loads(read_body(self, limit=(32 if tokenizer else 96) << 20) or b"{}")
                model = requested_model(app, body)          # an id this endpoint does not serve: 404
                # llama.cpp's opt-in prefill progress frames, on a stream
                progress = isinstance(body, dict) and flag(body, "return_progress", False) and bool(body.get("stream"))
            except RequestError as error:
                return self._refuse(error)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._json(400, {"error": invalid_request("the request body is not JSON")})
            if tokenizer:                                   # vLLM's /tokenize and /detokenize
                try:
                    reply = app.detokenize(body) if path.endswith("/detokenize") else app.tokenize(body)
                except RequestError as exc:
                    return self._refuse(exc)
                except Exception as exc:
                    _log_error(exc)
                    return self._json(400, {"error": invalid_request(_error_message(exc))})
                return self._json(200, reply)
            field = "messages" if chat else "prompt"            # the field an error's code names (OpenAI's param)
            try:
                if defer_headers:
                    if not isinstance(body, dict):
                        raise RequestError("the request body must be a JSON object")
                    prepared = None  # the configured App.run owner admits before preparation
                else:
                    prepared = app.prepare(body, chat)
            except RequestError as exc:
                return self._refuse(exc, field)
            except Exception as exc:        # any other failure to read the request is refused too, as on MLX
                _log_error(exc)
                return self._json(400, {"error": invalid_request(_error_message(exc))})
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}" if chat else f"cmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            stream = bool(body.get("stream"))
            separate_usage = wants_usage_chunk(body)          # usage then rides its own chunk before [DONE]
            kind = "chat.completion.chunk" if chat else "text_completion"
            gone = socket_cancellation(self.connection, stopping=getattr(self.server, "stopping", None))          # the Mac server's check: the client has closed
            cancelled = lambda: gone.cancelled                  # noqa: E731

            def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
                choice = ({"index": 0, "delta": delta, "finish_reason": finish} if chat else
                          {"index": 0, "text": delta.get("content", ""), "logprobs": None, "finish_reason": finish})
                frame = {"id": rid, "object": kind, "created": created, "model": model, "choices": [choice]}
                if separate_usage:                  # the spec: every chunk but the usage chunk has a null usage
                    frame["usage"] = None
                return frame

            if stream:
                opened = False
                said_role = not chat                # a chat stream's reply opens with the assistant's role chunk

                def open_stream() -> None:
                    nonlocal opened
                    if not opened:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        opened = True

                def write(frame: dict[str, Any] | None) -> None:
                    """Open the stream if it is not, write the role chunk if it is not out yet, then ``frame``."""
                    nonlocal said_role
                    open_stream()
                    if not said_role:
                        said_role = True
                        self.wfile.write(f"data: {json.dumps(chunk({'role': 'assistant'}))}\n\n".encode())
                    if frame is not None:
                        self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
                    self.wfile.flush()

                if not defer_headers:
                    try:
                        write(None)
                    except OSError:                 # the client has gone: the run notices at its first write
                        pass

                def emit(delta: dict[str, Any] | None) -> bool:
                    """Write one delta (None: an SSE comment that keeps an open stream's connection alive)."""
                    try:
                        if delta is None:
                            if opened:                   # deferred headers stay deferred: a refusal can still be a 503
                                self.wfile.write(b": keepalive\n\n")
                                self.wfile.flush()
                            return True
                        write(chunk(delta))
                        return True
                    except OSError:             # reset, broken pipe, timed out, host unreachable: the client has gone
                        return False

                def show_progress(progress: dict[str, Any]) -> bool:
                    """llama.cpp's ``prompt_progress`` frame, which carries no output; False: the client has gone."""
                    frame = chunk({"role": "assistant", "content": None} if chat else {})
                    frame["prompt_progress"] = progress
                    try:
                        open_stream()
                        self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
                        self.wfile.flush()
                        return True
                    except OSError:
                        return False

                try:
                    result = app.run(body, chat, emit, prepared=prepared, cancelled=cancelled,
                                     request_id=self.request_id, arrived=self.arrived,
                                     **({"progress": show_progress} if progress else {}))
                except RequestCancelled:
                    self.close_connection = True
                    return
                except RequestError as exc:
                    if not opened:
                        return self._refuse(exc, field)
                    return self._stream_error(error_body(exc, field))
                except Exception as exc:
                    _log_error(exc)
                    if not opened:
                        return self._json(500, {"error": server_error(_error_message(exc))})
                    return self._stream_error(server_error(_error_message(exc)))
                try:
                    write(None)                         # a reply with no token yet: its headers and role now
                except OSError:
                    pass
                if result["final"]:
                    emit(result["final"])
                if result["calls"]:
                    for i, call in enumerate(result["calls"]):
                        if i < result.get("calls_streamed", 0):      # sent as deltas already
                            continue
                        emit({"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                              "function": {"name": call["function"]["name"],
                                                           "arguments": call["function"]["arguments"]}}]})
                end = chunk({}, result["finish"])
                if result.get("stop_sequence") is not None:
                    end["stop_sequence"] = result["stop_sequence"]
                end["tensorfold"] = result["stats"]
                frames = [end]
                if separate_usage:                              # the spec: usage rides its own chunk before [DONE]
                    # with the stats too, so a client pairs usage with the timings on one frame (llama.cpp's place)
                    frames.append({"id": rid, "object": kind, "created": created, "model": model,
                                   "choices": [], "usage": usage_of(result), "tensorfold": result["stats"]})
                else:
                    end["usage"] = usage_of(result)      # every stream, as the Mac server's: clients count from it
                try:
                    self.wfile.write("".join(f"data: {json.dumps(frame)}\n\n" for frame in frames).encode()
                                     + b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
                return
            try:
                result = app.run(body, chat, lambda delta: True, prepared=prepared, cancelled=cancelled,
                                 request_id=self.request_id, arrived=self.arrived)
            except RequestCancelled:
                self.close_connection = True
                return
            except RequestError as exc:
                return self._refuse(exc, field)
            except Exception as exc:
                _log_error(exc)
                try:
                    self._json(500, {"error": server_error(_error_message(exc))})
                except OSError:
                    pass
                return
            usage = usage_of(result)
            if chat:
                message: dict[str, Any] = {"role": "assistant", "content": result["content"] or None}
                if result["reasoning"]:
                    message["reasoning_content"] = result["reasoning"]
                if result["calls"]:
                    message["tool_calls"] = result["calls"]
                payload = {"id": rid, "object": "chat.completion", "created": created, "model": model,
                           "choices": [{"index": 0, "message": message, "logprobs": result.get("logprobs"),
                                        "finish_reason": result["finish"]}],
                           "usage": usage, "tensorfold": result["stats"]}
            else:
                payload = {"id": rid, "object": "text_completion", "created": created, "model": model,
                           "choices": [{"index": 0, "text": result["content"], "logprobs": None,
                                        "finish_reason": result["finish"]}],
                           "usage": usage, "tensorfold": result["stats"]}
            if result.get("stop_sequence") is not None:
                payload["stop_sequence"] = result["stop_sequence"]
            self._json(200, payload)

        def _post_decisions(self) -> None:
            decide = getattr(app, "decisions", None)
            if decide is None:
                if not self._discard_body():
                    return
                return self._json(404, {"error": unknown_route("POST", self.path)})
            try:
                body = json.loads(read_body(self) or b"{}")
                if not isinstance(body, dict):
                    raise RequestError("request body must be an object")
                requested_model(app, body)
            except RequestError as exc:
                return self._refuse(exc)
            except Exception as exc:        # noqa: BLE001 - a bad body is a client error
                return self._json(400, {"error": invalid_request(_error_message(exc))})
            try:
                payload = decide(body)
            except RequestError as exc:
                return self._refuse(exc)
            except DecisionError as exc:
                return self._json(400, {"error": invalid_request(_error_message(exc))})
            except Exception as exc:        # a scoring failure is the server's, not a bad body
                _log_error(exc)
                try:
                    return self._json(500, {"error": server_error(_error_message(exc))})
                except OSError:
                    return
            self._json(200, payload)

    engine = getattr(app, "engine", None)
    if (getattr(engine, "execution_provider", None) == "sm70"
            and (getattr(engine, "adaptive_sessions", False)
                 or (getattr(engine, "capacity_plan", None) or {}).get("session_capacity") is not None)):
        Handler.requires_finite_limits = True
        Handler.pending_request_limit = app.max_pending_requests
    if getattr(engine, "serving_profiles", None) is not None:
        from tensorfold.cuda.profile_app import bind_http

        bind_http(Handler, app)
    from tensorfold.server.auth_http import handler
    return handler(Handler, app)


def serve(app: App, host: str, port: int, *, server: Server | None = None) -> None:
    """Serve until interrupted (SIGTERM included)."""

    import signal

    def _terminate(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _terminate)
    server = server if server is not None else Server((host, port), make_handler(app))
    primary = None
    try:
        from contextlib import nullcontext
        auth = getattr(app, "auth", None)
        with auth.signals() if auth is not None else nullcontext():
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            server.server_close()
        except BaseException as cleanup:
            if primary is not None and primary is not cleanup:
                raise primary from cleanup
            raise
