"""Opt-in App/tokenizer/prepared-request ownership across structural drains."""

from __future__ import annotations

from functools import wraps
import inspect
from pathlib import Path
import threading
import weakref

from tensorfold.cleanup import finish, raise_failures
from tensorfold.cuda.profile_binding import ProfileUnavailable

DERIVED = ("grammars", "_thinking_markers", "_thinking_fixed", "_strict_markup", "_probability_decoder", "_form")
BINDING = ("model_dir", "vision", "tok", "template", "segments", "served", "aliases", "default_thinking",
           "sampling", "max_tokens", "reasoning_effort", "thinking_budget", "native_context_window", "context_window")


def _weak_call(call):
    owner = getattr(call, "__self__", None)
    if owner is None:
        return call
    ref, function = weakref.ref(owner), call.__func__

    def invoke(*args, **kwargs):
        target = ref()
        if target is None:
            raise RuntimeError("the App owning this model entrypoint has retired")
        return function(target, *args, **kwargs)
    return invoke


class AppProfiles:
    def __init__(self, app, profiles):
        if app._request_limit is None:
            raise ValueError("serving profiles require a finite max_pending_requests admission bound")
        self.app, self.profiles = weakref.ref(app), profiles
        self.limit = app._request_limit
        self.local = threading.local()
        self.defaults = {name: getattr(app, name) for name in BINDING if name not in
                         ("model_dir", "vision", "tok", "template", "segments", "native_context_window", "context_window")}
        self.defaults["sampling"] = dict(app.sampling)
        profiles.apps.add(app)
        self._wrap_prepare(app)
        self._wrap_run(app)
        for name in ("tokenize", "detokenize", "decisions"):
            setattr(app, name, profiles.gate.wrap(getattr(app, name)))
        app.profile_scope = profiles.gate.scope
        app._retire_serving_binding = self.retire
        app._stage_serving_binding = self.stage
        app._publish_serving_binding = self.publish

    def _wrap_prepare(self, app):
        original = app.prepare
        call, signature = _weak_call(original), inspect.signature(original)

        @wraps(original.__func__)
        def prepare(*args, **kwargs):
            with self.profiles.gate.scope() as scope:
                internal = bool(getattr(self.local, "internal_prepare", False))
                self.local.internal_prepare = False
                reservation = None if internal else self.limit.prepare()
                lease = None
                try:
                    result = call(*args, **kwargs)
                    result._profile_use = threading.Lock()
                    lease = scope.hold()
                    result.profile_binding = lease
                    result._admission = reservation
                    operations = (lease.close,) + (() if reservation is None else (reservation.close,))
                    result._profile_finalizer = weakref.finalize(result, finish, operations)
                    if internal:
                        self.local.prepared.append(result)
                    return result
                except BaseException as primary:
                    try:
                        finish(operation for operation in (
                            None if lease is None else lease.close,
                            None if reservation is None else reservation.close) if operation is not None)
                    except BaseException as cleanup:
                        raise_failures(primary, [cleanup])
                    raise
        prepare.__signature__ = signature
        app.prepare = prepare

    def _wrap_run(self, app):
        original = app.run
        call, signature = _weak_call(original), inspect.signature(original)

        @wraps(getattr(original, "__func__", original))
        def run(*args, **kwargs):
            prepared = kwargs.get("prepared")
            if prepared is not None and prepared.profile_binding is None:
                raise ProfileUnavailable("the prepared request has no serving profile binding")
            selected = None if prepared is None else prepared.profile_binding
            with self.profiles.gate.scope(selected):
                made = []
                previous = getattr(self.local, "prepared", None)
                self.local.prepared = made
                previous_internal = getattr(self.local, "internal_prepare", False)
                self.local.internal_prepare = prepared is None
                claimed = prepared is not None and prepared._profile_use.acquire(blocking=False)
                if prepared is not None and not claimed:
                    self.local.prepared = previous
                    self.local.internal_prepare = previous_internal
                    raise ProfileUnavailable("a prepared request is already being consumed")
                primary = None
                try:
                    with self.limit.request(None if prepared is None else prepared._admission):
                        return call(*args, **kwargs)
                except BaseException as error:
                    primary = error
                    raise
                finally:
                    self.local.prepared = previous
                    self.local.internal_prepare = previous_internal
                    if claimed:
                        prepared._profile_use.release()
                    try:
                        finish(item.close for item in ([prepared] if prepared is not None else []) + made)
                    except BaseException as cleanup:
                        if primary is not None:
                            raise_failures(primary, [cleanup])
                        raise
        run.__signature__ = signature
        app.run = run

    def retire(self):
        app = self.app()
        if app is not None:
            app.vision = app.tok = app.template = app.segments = None
            for name in DERIVED:
                app.__dict__.pop(name, None)

    def stage(self, profile):
        from tensorfold.cuda.server import _native_context, _segment_encoder
        from tensorfold.token_metadata import load_tokenizer

        app = self.app()
        if app is None:
            raise RuntimeError("the serving App retired during profile construction")
        view = object.__new__(type(app))
        view.__dict__.update(app.__dict__)
        for name in DERIVED:
            view.__dict__.pop(name, None)
        view.__dict__.update(self.defaults)
        view.__dict__.update(profile.api)
        view.aliases = tuple(view.aliases)
        view.sampling = {**self.defaults["sampling"], **profile.api.get("sampling", {})}
        view.vision = app.engine.vision
        view.model_dir = Path(profile.model_dir)
        if getattr(view.vision, "is_media_runtime", False):
            view.vision.check_media_health()
            view.tok = view.vision.tokenizer
        else:
            view.tok = load_tokenizer(view.model_dir)
        view.template = app.engine.chat_template
        view.segments = _segment_encoder(view.tok, view.template)
        view.native_context_window = _native_context(view.model_dir)
        view.context_window = app.engine.context_window
        fixed = view._fixed_thinking(False)
        if fixed is not None:
            view.default_thinking = fixed
        catalog = getattr(app.engine, "retention_catalog", None)
        if catalog is not None:
            view._finalize_retention_resources()
        return {name: getattr(view, name) for name in BINDING}

    def publish(self, binding):
        app = self.app()
        if app is None:
            raise RuntimeError("the serving App retired before profile publication")
        app.__dict__.update(binding)


def bind_http(handler, app):
    """Scope each model-bound HTTP transaction, after auth and before model lookup."""
    from tensorfold.server import runtime, stats
    from tensorfold.server.http import model_route

    gate = app.engine.serving_profiles.gate
    for name in ("do_GET", "do_POST"):
        original = getattr(handler, name)

        @wraps(original)
        def request(owner, _call=original, _name=name):
            route = owner.path.split("?", 1)[0].rstrip("/")
            scoped = (route != runtime.ROUTE if _name == "do_POST" else
                      model_route(route) is not None or route in stats.ROUTES or
                      route in ("/metrics", "/v1/metrics", "/health", "/v1/health"))
            if not scoped:
                return _call(owner)
            try:
                with gate.scope():
                    return _call(owner)
            except ProfileUnavailable as error:
                if owner._discard_body():
                    return owner._refuse(error)
        setattr(handler, name, request)
