#!/usr/bin/env python3
"""Run pinned TensorFold CUDA with exact, generation-free Harness token counting.

Invocation: python3 tensorfold_harness.py serve MODEL [native TensorFold flags]
Requires installed owned TensorFold 0.6.6 and --thinking-budget 0. Installed packages are unchanged.
The additional POST /v1/tokenize route accepts a native chat request and optional
boolean return_tokens / return_prompt fields. It counts the generation prompt,
including image expansion, even when that prompt exceeds the served context.
Native image size, count, pixel and visual-token safety limits remain in force.
Successful counting replies also carry the facts Harness discovers models from:
canonical reasoning efforts, the active chat template's fingerprint and
loaded-code provenance, resolved once when the handler is built.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import marshal
import socket
import sys
import threading
import time
from contextlib import contextmanager
from types import CodeType
from typing import Any, Callable

TENSORFOLD_VERSION = "0.6.6"
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
                  capabilities: dict[str, Any] | None = None, *,
                  chat_template_sha256: str | None = None,
                  provenance: dict[str, Any] | None = None) -> dict[str, Any]:
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
    if chat_template_sha256 is not None:
        response["chat_template_sha256"] = chat_template_sha256
    if provenance is not None:
        response["provenance"] = provenance
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


def capability_metadata(app: Any) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
    """Resolve native facts once, before any counting request is admitted.

The App exposes the template's canonical efforts. Forwarding a thinking keyword
does not prove the active template consumes it or supports both values, so
thinking controls/defaults and replay semantics remain absent. Harness derives
those from the fingerprint of a template whose contract it has pinned.
"""
    efforts = app.template.efforts
    capabilities: dict[str, Any] = {
        "structured_output": importlib.util.find_spec("xgrammar") is not None
        and "constraint" in inspect.signature(app.engine.generate).parameters,
        "vision": app.vision is not None,
        "video": bool(getattr(app.vision, "videos", False)),
        "reasoning_efforts": sorted(efforts),
    }
    if app.reasoning_effort is not None:
        from tensorfold.server.request_options import heard_effort
        # Resolution can return a name the template lacks (xhigh, none, or any
        # name when it names no levels); only a level it names is canonical.
        heard = heard_effort(None, app.reasoning_effort, efforts)
        if heard in efforts:
            capabilities["default_reasoning_effort"] = heard
    provenance: dict[str, Any] = {
        "server_version": TENSORFOLD_VERSION,
        "bridge_version": BRIDGE_VERSION,
        "server_type": _qualified_type(app),
        "engine_type": _qualified_type(app.engine),
        "tokenizer_type": _qualified_type(app.tok),
    }
    # These identify loaded Python code, not current files or captured engine
    # state. Marshal identities are scoped to the recorded Python/format build.
    for key, operation in (
        ("server_prepare_sha256", type(app)._prepare),
        ("server_check_fields_sha256", app._check_fields),
        ("engine_generate_sha256", app.engine.generate),
    ):
        code = getattr(operation, "__code__", None)
        if isinstance(code, CodeType):
            provenance[key] = hashlib.sha256(marshal.dumps(code)).hexdigest()
    provenance["code_identity"] = "python_runtime_bytecode"
    provenance["python_version"] = sys.version.split()[0]
    provenance["marshal_version"] = marshal.version
    provenance["tokenizer_sha256"] = hashlib.sha256(app.tok.to_str().encode("utf-8")).hexdigest()
    provenance["tokenizer_identity"] = "runtime_serialization"
    fingerprint, source_name = _template_fingerprint(app)
    if fingerprint is not None:
        provenance["template_source"] = source_name
        provenance["template_identity"] = "equivalent_active_compilation"
    return capabilities, fingerprint, provenance


def _qualified_type(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _template_fingerprint(app: Any) -> tuple[str | None, str | None]:
    """Identify source only when it compiles to the active root and block code.

ChatTemplate discards its source after compiling it, so a later file read alone
cannot identify the active template. Reusing its Jinja environment preserves
compiler settings, filters and globals; structural code equality rejects changed
executable source. Source precedence and default text decoding match
ChatTemplate's loader, so the candidate is the text it would compile. The hash
covers this equivalent source's UTF-8 encoding, not original captured file bytes.
"""
    active = app.template.template
    try:
        source_path = app.model_dir / "chat_template.jinja"
        if source_path.exists():
            source, source_name = source_path.read_text(), "chat_template.jinja"
        else:
            source = json.loads((app.model_dir / "tokenizer_config.json").read_text())["chat_template"]
            source_name = "tokenizer_config.chat_template"
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return None, None
    if not isinstance(source, str):
        return None, None
    from jinja2 import TemplateError
    try:
        candidate = active.environment.from_string(source)
    except TemplateError:
        return None, None
    if (not _template_function_matches(active.root_render_func, candidate.root_render_func)
            or candidate.globals != active.globals
            or candidate.blocks.keys() != active.blocks.keys()
            or any(not _template_function_matches(block, candidate.blocks[name])
                   for name, block in active.blocks.items())):
        return None, None
    return hashlib.sha256(source.encode("utf-8")).hexdigest(), source_name


def _template_function_matches(active: Any, candidate: Any) -> bool:
    """Compare the pinned Jinja module's code and its execution bindings."""
    if (active.__code__ != candidate.__code__
            or active.__closure__ is not None or candidate.__closure__ is not None):
        return False
    active_defaults, candidate_defaults = active.__defaults__ or (), candidate.__defaults__ or ()
    if (len(active_defaults) != len(candidate_defaults)
            or any(left is not right for left, right in zip(active_defaults, candidate_defaults))):
        return False
    active_keywords, candidate_keywords = active.__kwdefaults__ or {}, candidate.__kwdefaults__ or {}
    if (active_keywords.keys() != candidate_keywords.keys()
            or any(value is not candidate_keywords[key] for key, value in active_keywords.items())):
        return False

    def referenced_names(code: CodeType):
        yield from code.co_names
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                yield from referenced_names(constant)

    # Template.globals does not cover imported Jinja runtime helpers or the
    # generated functions' builtins. Compare only referenced bindings: root and
    # block functions in separate compiled modules have distinct unused objects.
    missing = object()
    for name in referenced_names(active.__code__):
        active_value = active.__globals__.get(name, missing)
        candidate_value = candidate.__globals__.get(name, missing)
        if active_value is not candidate_value:
            return False
        if active_value is missing and (active.__builtins__.get(name, missing)
                                       is not candidate.__builtins__.get(name, missing)):
            return False
    return True


def make_handler_factory(native_factory: Callable[[Any], type]) -> Callable[[Any], type]:
    """Add one route to the native factory; every other route stays native."""
    def make_handler(app: Any) -> type:
        from tensorfold.server import request_body
        from tensorfold.server.cancellation import RequestCancelled, socket_cancellation
        from tensorfold.server.errors import CapacityError, RequestError, error_body

        if app.thinking_budget != 0:
            raise ValueError("Harness requires TensorFold --thinking-budget 0; budgets are selected per request")
        native_handler = native_factory(app)
        gate = PreparationGate()
        capabilities, template_sha256, provenance = capability_metadata(app)

        class Handler(native_handler):
            def do_POST(self):
                route = self.path.split("?", 1)[0].rstrip("/")
                if route != "/v1/tokenize":
                    return super().do_POST()
                gone = socket_cancellation(self.connection, stopping=getattr(self.server, "stopping", None))
                try:
                    with gate.enter(lambda: gone.cancelled, CapacityError, RequestCancelled):
                        length = request_body.content_length(self, MAX_BODY_BYTES, require_nonempty=True)
                        previous_timeout = self.connection.gettimeout()
                        self.connection.settimeout(BODY_TIMEOUT_S)
                        try:
                            payload = self.rfile.read(length)
                        finally:
                            self.connection.settimeout(previous_timeout)
                        if len(payload) != length:
                            self.close_connection = True
                            raise RequestError("token counting request body ended before Content-Length")
                        try:
                            body = json.loads(payload, parse_constant=_invalid_json_number)
                        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
                            raise RequestError("token counting request body must be valid UTF-8 JSON") from exc
                        if gone.cancelled:
                            raise RequestCancelled("token counting client disconnected")
                        result = count_request(app, body, RequestError, capabilities,
                                               chat_template_sha256=template_sha256,
                                               provenance=provenance)
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
    from importlib import metadata

    try:
        installed_version = metadata.version("tensorfold")
    except metadata.PackageNotFoundError:
        print(f"Harness tokenizer requires installed TensorFold {TENSORFOLD_VERSION}", file=sys.stderr)
        return 2
    if tensorfold.__version__ != TENSORFOLD_VERSION or installed_version != TENSORFOLD_VERSION:
        print(f"Harness tokenizer requires owned TensorFold {TENSORFOLD_VERSION}; "
              f"found module {tensorfold.__version__}, installed {installed_version}", file=sys.stderr)
        return 2
    from tensorfold import cli
    from tensorfold.cuda import http, server
    from tensorfold.server import request_body
    from tensorfold.server.cancellation import socket_cancellation

    try:
        inspect.signature(request_body.content_length).bind(None, MAX_BODY_BYTES, require_nonempty=True)
        inspect.signature(socket_cancellation).bind(None, stopping=None)
        inspect.signature(server.App._prepare).bind(None, {}, True)
    except (AttributeError, TypeError, ValueError):
        print("Harness tokenizer requires the owned TensorFold request preparation/framing interfaces", file=sys.stderr)
        return 2
    if server.make_handler is not http.make_handler:
        print("Harness tokenizer requires coherent native CUDA HTTP handler exports", file=sys.stderr)
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
    from tensorfold.cleanup import raise_failures

    # CLI imports the server's reexport; bind both supported surfaces even if
    # cuda.server was already imported. Export restoration changes future
    # construction only; existing listeners retain their constructed handlers.
    native_factory = http.make_handler
    wrapped_factory = make_handler_factory(native_factory)
    primary = None
    try:
        http.make_handler = server.make_handler = wrapped_factory
        return cli.main(arguments)
    except BaseException as error:
        primary = error
        raise
    finally:
        failures = []
        for module in (http, server):
            try:
                module.make_handler = native_factory
            except BaseException as cleanup:
                failures.append(cleanup)
        if failures:
            raise_failures(primary, failures)


if __name__ == "__main__":
    raise SystemExit(main())
