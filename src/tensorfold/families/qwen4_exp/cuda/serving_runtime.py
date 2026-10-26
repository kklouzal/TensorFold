"""Structural profile owner for FlashNextEngine's persistent serving worker."""

from __future__ import annotations

from copy import deepcopy
import gc
import threading
import traceback
import weakref

from tensorfold.cuda.profile_binding import ProfileGate, retry_control
from tensorfold.cleanup import error_name, raise_failures

from .execution_identity import _unretired
from .profile_construction import Construction, ConstructionFatal
from .serving_profiles import FIXED, prepare, profile_root


class ServingProfiles:
    def __init__(self, engine, root, options):
        if (engine.execution_provider != "sm70" or not engine.adaptive_sessions
                or engine.indexer_history != "gpu" or engine.multi is None
                or not engine.multi.full_resident or options.get("expert_session") is not None
                or options.get("vram_experts") is not None):
            raise ValueError("serving profiles require resident adaptive Flash Next GGUF GPU histories on SM70")
        self._engine = weakref.ref(engine)
        self.root = profile_root(root)
        self.fixed = {name: options[name] for name in FIXED if name in options}
        self.gate = ProfileGate(wake=self.wake, fatal=self._poison)
        self.lock = threading.RLock()
        self.preparing = threading.Lock()
        self.pending = self.candidate = None
        self.sequence = 0
        self.failure = None
        self.apps = weakref.WeakSet()
        self.status = {"configured": None, "pending": None, "effective": {"id": "startup",
                       "model_dir": str(engine.model_dir), "context_tokens": engine.context_window,
                       "maximum_proposals": engine.depth}, "stage": "ready", "request_epoch": 0,
                       "error": None}
        scheduler = engine.scheduler
        if scheduler is not None:
            scheduler.serving_profiles = self
            scheduler._serving_profile_epoch = 0
            scheduler._serving_profile_available = True
        engine.generate = self.gate.wrap(engine.generate)
        engine.score_labels = self.gate.wrap(engine.score_labels)

    @property
    def engine(self):
        engine = self._engine()
        if engine is None:
            raise RuntimeError("the serving engine owner has retired")
        return engine

    def wake(self):
        engine = self._engine()
        scheduler = None if engine is None else engine.scheduler
        if scheduler is not None:
            scheduler.waiting.wake()

    def record(self):
        with self.lock:
            return {**deepcopy(self.status), **self.gate.record()}

    def request(self, name):
        if self.failure is not None:
            raise ConstructionFatal("serving profile ownership is poisoned; process containment is required") from self.failure
        if not self.preparing.acquire(blocking=False):
            raise ValueError("a serving profile is already being prepared")
        try:
            with self.lock:
                if self.pending is not None:
                    raise ValueError("a serving profile transition is already pending")
            # Full artifact hashing is host admin preparation, outside the GPU
            # worker and its lock; the old profile continues serving here.
            try:
                profile = prepare(self.root, name, initializer=self.engine._initialize, fixed_options=self.fixed)
            except BaseException as error:
                if _unretired(error):
                    self._poison(error)
                raise
            scheduler = self.engine.scheduler
            with scheduler._state_lock:
                if scheduler._closing:
                    from tensorfold.cuda.profile_binding import ProfileUnavailable

                    raise ProfileUnavailable("CUDA scheduler is closing; serving profile selection is rejected")
                if scheduler._pending_manifest is not None:
                    raise ValueError("finish pending execution evidence registration before selecting a serving profile")
                with self.lock:
                    self.sequence += 1
                    self.pending = profile
                    self.status.update(configured=name, pending=profile.record(), stage="prepared",
                                       request_epoch=self.sequence, error=None)
                retry_control(scheduler.waiting.wake, self._control_failed)
            return self.record()
        finally:
            self.preparing.release()

    def _control_failed(self, primary, secondary=None):
        try:
            self._poison(primary)
        except BaseException as containment:
            if secondary is not None:
                raise_failures(containment, [secondary])
            raise

    def error_outcome(self, error, *, states=(), state_retired=False):
        """Detach only successfully retired request errors from model-owning frames.

        The decoder certifies state cleanup first. Original exception identity,
        cause/context/groups and frame-free stack evidence survive transport;
        unknown resources or failed cleanup retain their raw owners under the
        fatal policy instead of allowing a second model to overlap them.
        """
        namespace = BaseException.__dict__["__dict__"].__get__(error)
        certified = dict.pop(namespace, "_tensorfold_request_state_retired", False) is True
        dict.pop(namespace, "_tensorfold_profile_error_retired", None)  # Never trust an earlier delivery.
        if self.failure is not None:
            return error  # Terminal ownership is retained until process containment.
        decoder = self.engine.multi
        if (_unretired(error) or not (certified or state_retired)
                or getattr(decoder, "_memory_error", None) is not None or getattr(decoder, "failed_slots", ())):
            self._poison(error)
        nodes, pending, seen = [], [error], set()
        try:
            while pending:
                item = pending.pop()
                if item is None or id(item) in seen:
                    continue
                seen.add(id(item))
                nodes.append(item)
                group = isinstance(item, BaseExceptionGroup)
                args = BaseException.args.__get__(item)
                if not group and any(type(arg) not in (str, int, float, bool, type(None)) for arg in args):
                    raise RuntimeError("request error payload retains unknown ownership")
                values = BaseException.__dict__["__dict__"].__get__(item)
                for name, value in dict.items(values):
                    if name == "__notes__" and type(value) is list and all(type(note) is str for note in value):
                        continue
                    if type(value) not in (str, int, float, bool, type(None)):
                        raise RuntimeError("request error metadata retains unknown ownership")
                pending.extend((BaseException.__cause__.__get__(item), BaseException.__context__.__get__(item)))
                if group:
                    pending.extend(BaseExceptionGroup.exceptions.__get__(item))
            import torch

            if self.engine.w is not None:
                torch.cuda.synchronize(self.engine.w.device)
            runtime = getattr(self.engine.w, "dflash2", None)
            if runtime is not None:
                runtime.retire_request_after_fence(tuple(states), fenced=True)
            stacks = [(item, "retired worker stack:\n" + (
                "".join(traceback.format_tb(item.__traceback__)) if item.__traceback__ is not None
                else error_name(item) + ": " + BaseException.__str__(item)))
                for item in nodes if item.__traceback__ is not None or dict.get(
                    BaseException.__dict__["__dict__"].__get__(item), "_tensorfold_worker_stack_note") is None]
            for item, stack in stacks:
                values = BaseException.__dict__["__dict__"].__get__(item)
                previous = dict.get(values, "_tensorfold_worker_stack_note")
                notes = dict.get(values, "__notes__", ())
                found = next((i for i, note in enumerate(notes) if note is previous), None)
                if found is None:
                    BaseException.add_note(item, stack)
                else:
                    notes[found] = stack
                values["_tensorfold_worker_stack_note"] = stack
            # Removing the TB, rather than only clearing inactive frames, also
            # prevents active Scheduler/TP frames from becoming caller-owned.
            for item in nodes:
                BaseException.__traceback__.__set__(item, None)
            return error
        except BaseException as retirement:
            try:
                self._poison(error)
            except BaseException as containment:
                raise_failures(containment, [retirement])

    def _vote(self, phase, local):
        comm = self.engine.comm
        if comm is None:
            return [local]
        return Construction(comm, f"{self.sequence}/{phase}").outcomes(local)

    def _poison(self, error):
        """Close admission/notify containment before allocating diagnostics."""
        previous = self.failure
        self.failure = error
        publication = notification = None
        try:
            self.gate.poison(error)
        except BaseException as failed:
            publication = failed
        engine = self.engine
        engine._abort_comm = True
        try:
            adapter = getattr(engine, "tp_decoder", None)
            if adapter is not None:
                adapter._poison_control(error)
            else:
                from tensorfold.cuda.cohort import notify_failure

                notify_failure()
        except BaseException as failed:
            notification = failed
        try:
            with self.lock:
                self.pending = self.candidate = None
                self.status.update(pending=None, stage="fatal",
                                   error={"type": error_name(error), "message": "unresolved cleanup ownership"})
            fatal = ConstructionFatal("serving profile cleanup retained unresolved owners; process containment is required")
            failures = [item for item in (error, previous, publication, notification) if item is not None and item is not fatal]
        except BaseException as diagnostic:
            # The original retained owner, native gate latch and cohort signal
            # already exist even if no contextual wrapper/list can be allocated.
            raise error from diagnostic
        self.failure = fatal  # The wrapper owns every raw outcome through its failure transport.
        raise_failures(fatal, failures)

    def preflight(self, wire, sequence):
        self.sequence = sequence
        error = None
        profile = None
        try:
            if self.engine.rank == 0:
                profile = self.pending
                if profile is None or profile.wire() != wire:
                    raise ValueError("the prepared coordinator profile changed")
                profile.check_sources()
            else:
                profile = prepare(self.root, wire["name"], initializer=self.engine._initialize,
                                  fixed_options=self.fixed, verify_content=False, expected=wire)
        except (ValueError, OSError, MemoryError) as failed:
            if _unretired(failed):
                self._poison(failed)
            error = {"type": type(failed).__name__, "message": str(failed)[:1024]}
        votes = self._vote("preflight", {"error": error})
        failures = [item["error"] for item in votes if item["error"] is not None]
        if not failures:
            self.candidate = profile
            self.gate.drain()
        return {"ready": not failures, "errors": failures}

    def phase(self, name):
        with self.lock:
            self.status["stage"] = name

    def _retire(self):
        engine = self.engine
        import torch

        torch.cuda.synchronize()
        adapter = getattr(engine, "tp_decoder", None)
        if adapter is not None:
            adapter.suspend_profile()
        elif engine.scheduler is not None:
            engine.scheduler.decoder = None
            engine.scheduler._arrival_original = None
            engine.scheduler._arrival_applied = False
        # No borrowed request, graph or captured bound decoder method may keep
        # the old model alive across the following replacement allocation.
        engine._drain_startup_reads()
        engine._close_resources(preserve_transport=True)
        for app in tuple(self.apps):
            app._retire_serving_binding()
        engine.multi = engine.e = engine.serial = engine.w = engine.vision = None
        engine.cache = []
        engine.points = engine.chat_template = None
        engine._owner_preparation = None  # __init__ is not rerun for the next profile.
        engine.capacity_plan = None
        for name in ("grammars", "_execution_input_authority", "_execution_input_authorities", "_execution_profile_identity",
                     "_execution_profile_artifacts", "_execution_bind_local", "_execution_preload", "_execution_profile_root"):
            engine.__dict__.pop(name, None)
        gc.collect()
        torch.cuda.empty_cache()

    def replace(self):
        engine = self.engine
        profile = self.candidate
        if profile is None or self.gate.borrowers:
            raise RuntimeError("a serving replacement requires a prepared, drained profile")
        self.gate.unavailable()
        with self.lock:
            self.status["effective"] = None
        scheduler = engine.scheduler
        if scheduler is not None:
            scheduler._serving_profile_available = False
        self.phase("retiring")
        self._retire()
        votes = self._vote("retired", {"retired": True})
        if not all(item == {"retired": True} for item in votes):
            raise ConstructionFatal("ranks did not retire the old model")
        frontier = None if engine.comm is None else Construction(engine.comm, f"{self.sequence}/load")
        error = None
        failed_owner = None
        staged = []
        try:
            profile.check_sources()
            engine._profile_build = self
            if frontier is None:
                engine._initialize(profile.model_dir, **profile.options)
            else:
                with frontier:
                    engine._initialize(profile.model_dir, **profile.options)
            for app in tuple(self.apps):
                staged.append((app, app._stage_serving_binding(profile)))
            import torch

            torch.cuda.synchronize()
            profile.check_sources()
        except BaseException as failed:
            import torch

            if _unretired(failed):
                self._poison(failed)
            expected = isinstance(failed, (ValueError, OSError, MemoryError, torch.cuda.OutOfMemoryError))
            failed_graph = getattr(engine.multi, "_failed_graphs", None)
            if not expected or failed_graph is not None or frontier is not None and frontier.fatal:
                if frontier is not None:
                    frontier.fail(failed, fatal=True)
                raise
            error = {"type": type(failed).__name__, "message": str(failed)[:1024]}
            if frontier is not None:
                frontier.fail(failed)
            torch.cuda.synchronize()
            # Constructor-only tensors/FeatureFrames remain owned through the
            # coordinated fence and provider-specific fenced retirement below.
            failed_owner = failed
        finally:
            engine._profile_build = None
        local = {"error": error, "step": 0 if frontier is None else frontier.step}
        votes = self._vote("loaded", local) if frontier is None else frontier.outcomes(local)
        failures = [item["error"] for item in votes if item["error"] is not None]
        if failures:
            staged.clear()
            self._retire()  # Includes DF retire_after_fence once integrated in Engine._close_resources.
            self._vote("refused-cleanup", {"retired": True})
            if failed_owner is not None:
                traceback.clear_frames(failed_owner.__traceback__)
                failed_owner = None
                gc.collect()
                torch.cuda.empty_cache()
            self.candidate = None
            return {"ready": False, "errors": failures}
        # Every rank has loaded, allocated, warmed, fenced and validated its
        # complete binding. Publication below is the first new admission point.
        adapter = getattr(engine, "tp_decoder", None)
        if adapter is not None:
            adapter.bind_profile(engine.multi)
        elif scheduler is not None:
            scheduler.decoder = engine.multi
            scheduler._arrival_original = engine.multi.arrived
            scheduler._arrival_applied = True
            engine.multi.arrived = scheduler._arrival_hook
        for app, binding in staged:
            app._publish_serving_binding(binding)
        epoch = self.gate.epoch + 1
        if scheduler is not None:
            scheduler.max_streams = engine.streams
            manifest = profile.options.get("execution_profile_manifest")
            scheduler._manifest_status.update(configured=None if manifest is None else str(manifest),
                effective=None if manifest is None else str(manifest), pending=False, error=None)
            scheduler._serving_profile_epoch = epoch
            scheduler._serving_profile_available = True
        with self.lock:
            self.status.update(effective=profile.record(), stage="ready", error=None)
        self.gate.publish(epoch)
        self.candidate = None
        return {"ready": True, "errors": []}

    def bind_execution_manifest(self, path):
        from .execution_identity import _ManifestRefusal, bind_execution_manifest

        engine = self.engine
        def operation():
            return bind_execution_manifest(engine, path)
        frontier = None if engine.comm is None else getattr(engine.comm, "_profile_construction", None)
        return (operation() if frontier is None else
                frontier.protocol("execution-manifest-binding", operation, refusal=_ManifestRefusal))

    def boundary(self, scheduler):
        try:
            return self._boundary(scheduler)
        except BaseException as error:
            if self.failure is not None:
                raise
            self._poison(error)

    def _boundary(self, scheduler):
        with self.lock:
            profile = self.pending
            stage = self.status["stage"]
        if profile is None:
            return
        adapter = getattr(self.engine, "tp_decoder", None)
        if stage == "prepared":
            result = (adapter.prepare_serving_profile(profile.wire(), self.sequence) if adapter is not None else
                      self.preflight(profile.wire(), self.sequence))
            if not result["ready"]:
                with self.lock:
                    self.pending = None
                    self.status.update(pending=None, stage="ready" if self.gate.available else "unavailable",
                                       error=result["errors"])
                return
            with self.lock:
                self.status["stage"] = "draining"
        if (self.gate.borrowers or scheduler.boxes or scheduler.held is not None
                or scheduler.waiting.qsize() or self.engine.multi is not None and self.engine.multi.live()):
            return
        catalog = getattr(self.engine, "retention_catalog", None)
        if catalog is not None:
            if adapter is not None:
                adapter.drop_optional_retention()
            else:
                catalog.drop_optional_prefixes()
                self.engine.multi._release_retained_states()
        if catalog is not None and catalog.has_retained:
            blocking = catalog.record()
            with self.lock:
                self.status["blocking_retention"] = blocking
            return
        with self.lock:
            self.status.pop("blocking_retention", None)
            self.status["stage"] = "loading"
        result = adapter.replace_serving_profile() if adapter is not None else self.replace()
        with self.lock:
            self.pending = None
            self.status.update(pending=None, stage="ready" if result["ready"] else "unavailable",
                               effective=profile.record() if result["ready"] else None,
                               error=None if result["ready"] else result["errors"])
