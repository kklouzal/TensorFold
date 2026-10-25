"""Construction-only CPU readiness frontiers around the existing communicator.

A durable GO obliges every rank to issue that collective. A later local load
refusal cannot cancel it. Only a not-yet-issued collective may be refused.
These wrappers are removed before publication: serving retains its usual
communication path, identity and ordering.
"""

from __future__ import annotations

from datetime import timedelta
import inspect
import json
import time

from tensorfold.cleanup import finish, raise_failures


class ConstructionRefused(ValueError):
    pass


class ConstructionFatal(RuntimeError):
    pass


def replay(graph, comm):
    """A captured NCCL graph launch has the same collective readiness duty."""
    construction = None if comm is None else getattr(comm, "_profile_construction", None)
    if construction is None or not construction.active:
        return graph.replay()
    return construction.launch(graph.replay)


class Construction:
    METHODS = ("all_gather", "barrier", "broadcast", "ready", "exchange", "send", "recv")

    def __init__(self, comm, epoch):
        self.comm, self.epoch = comm, epoch
        self.step = 0
        self.active = False
        self.fatal = False
        self.original = {}
        self.bindings = []

    def key(self, suffix):
        return f"tensorfold/flashnext/construct/{self.epoch}/{suffix}"

    def put(self, suffix, value):
        key = self.key(suffix)
        self.comm.store.set(key, json.dumps(value, separators=(",", ":")))

    def read(self, suffix):
        key = self.key(suffix)
        return json.loads(self.comm.store.get(key)) if self.comm.store.check([key]) else None

    def fail(self, error, *, fatal=False):
        self.fatal |= fatal
        notification = None
        attempted = self.fatal
        if attempted:
            try:
                from tensorfold.cuda.cohort import notify_failure

                notify_failure()
            except BaseException as failed:
                notification = failed
        try:
            self.put(f"failure/{self.comm.rank}", {"step": self.step, "fatal": self.fatal,
                      "error": type(error).__name__, "message": str(error)[:1024]})
        except BaseException as diagnostic:
            # A missing refusal publication can strand peers just like an
            # issued collective failure. Escalate before fallible transport.
            self.fatal = True
            if not attempted:
                try:
                    from tensorfold.cuda.cohort import notify_failure

                    notify_failure()
                except BaseException as failed:
                    notification = failed
            raise_failures(error, [item for item in (diagnostic, notification) if item is not None])
        if notification is not None:
            raise_failures(error, [notification])

    def frontier(self, method):
        step = self.step
        self.put(f"ready/{step}/{self.comm.rank}", method)
        deadline = time.monotonic() + timedelta(hours=1).total_seconds()
        while True:
            decision = self.read(f"go/{step}")
            if decision is not None:
                if decision != method:
                    raise ConstructionFatal("ranks disagreed on construction collective order")
                # GO remains binding even if a faster peer has refused step+1.
                return
            failures = [self.read(f"failure/{rank}") for rank in range(self.comm.world)]
            if any(item is not None and item["fatal"] for item in failures):
                raise ConstructionFatal("a rank failed after ambiguous construction/device work")
            if any(item is not None and item["step"] <= step for item in failures):
                raise ConstructionRefused("a peer refused construction before the next collective")
            ready = [self.read(f"ready/{step}/{rank}") for rank in range(self.comm.world)]
            if self.comm.rank == 0 and all(item is not None for item in ready):
                if any(item != method for item in ready):
                    raise ConstructionFatal("ranks disagreed on construction collective order")
                self.put(f"go/{step}", method)
                # All participants arrived here, hence retired step-1. Keep at
                # most the previous/current frontiers until terminal agreement.
                if step:
                    for suffix in [f"go/{step-1}", *(f"ready/{step-1}/{r}" for r in range(self.comm.world))]:
                        self.comm.store.delete_key(self.key(suffix))
                continue
            if time.monotonic() >= deadline:
                raise ConstructionFatal("construction readiness store timed out")
            time.sleep(0.005)

    def call(self, name, *args, **kwargs):
        original = self.original[name][1]
        if not self.active:
            return original(*args, **kwargs)
        if name in ("exchange", "send", "recv"):
            raise ConstructionRefused("structural profiles require collective resident-weight construction")
        if name != "ready":
            import torch

        # barrier's original implementation allocates scratch after entering
        # NCCL. Prepare it before READY so a local OOM remains a refusal.
        scratch = None
        if name == "barrier":
            scratch = (torch.zeros(1, dtype=torch.float32, device=self.comm.device),
                       torch.empty(self.comm.world, dtype=torch.float32, device=self.comm.device))
        label = name
        if name != "ready":
            label += f"/capture={int(torch.cuda.is_current_stream_capturing())}"
        self.frontier(label)
        try:
            if scratch is None:
                result = original(*args, **kwargs)
            else:
                result = self.original["all_gather"][1](*scratch)
                torch.cuda.synchronize()
        except BaseException as error:
            self.fail(error, fatal=True)
            raise
        finally:
            self.step += 1
        return result

    def __enter__(self):
        if hasattr(self.comm, "_profile_construction"):
            raise RuntimeError("construction scopes cannot overlap on a communicator")
        try:
            self.comm._profile_construction = self
            for name in self.METHODS:
                if hasattr(self.comm, name):
                    self.original[name] = (name in vars(self.comm), getattr(self.comm, name))
                    setattr(self.comm, name, lambda *args, _name=name, **kwargs: self.call(_name, *args, **kwargs))
            self.active = True
        except BaseException as primary:
            try:
                self.restore()
            except BaseException as cleanup:
                raise_failures(primary, [cleanup])
            raise
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.restore()
        except BaseException as cleanup:
            self.fatal = True
            if error is not None:
                raise_failures(error, [cleanup])
            raise

    def restore(self):
        self.active = False
        def restore_method(name, owned, original):
            if owned:
                setattr(self.comm, name, original)
            elif name in vars(self.comm):
                delattr(self.comm, name)
        operations = [lambda: vars(self.comm).pop("_profile_construction", None)]
        operations.extend(lambda name=name, owned=owned, original=original: restore_method(name, owned, original)
                          for name, (owned, original) in self.original.items())
        operations.extend(lambda binding=binding: self._restore_binding(binding) for binding in reversed(self.bindings))
        finish(operations)
        self.bindings.clear()  # Cached inactive comm wrappers must not retain model/graph owners.

    def launch(self, operation):
        return self.submit("graph-launch", operation)

    def submit(self, label, operation, *args, **kwargs):
        """One globally uniform native/subgroup/graph submission after prep.

        Every world rank must call this same stage, even for separate fixed
        subgroups. All allocations/validation precede entry; post-GO errors are
        fatal. Advancing before the operation permits nested instrumented calls.
        """
        if not self.active:
            return operation(*args, **kwargs)
        import torch

        self.frontier(f"{label}/capture={int(torch.cuda.is_current_stream_capturing())}")
        self.step += 1
        try:
            return operation(*args, **kwargs)
        except BaseException as error:
            self.fail(error, fatal=True)
            raise

    def protocol(self, label, operation, *args, refusal, **kwargs):
        """Enter an existing all-rank protocol only when every builder reached it.

        That protocol owns local outcome propagation once entered. Advancing
        first allows its own communicator calls to use subsequent frontiers.
        """
        if not self.active:
            return operation(*args, **kwargs)
        self.frontier(label)
        self.step += 1
        try:
            return operation(*args, **kwargs)
        except refusal:
            raise  # This protocol already voted and retired the refusal.
        except BaseException as error:
            self.fail(error, fatal=True)
            raise

    def _bound_operation(self, label, operation, refusal):
        if not self.active or not callable(operation):
            raise ValueError("construction bindings require an active context and a prepared callable")

        def bound(*args, **kwargs):
            if refusal is None:
                return self.submit(label, operation, *args, **kwargs)
            return self.protocol(label, operation, *args, refusal=refusal, **kwargs)
        return bound

    def bind_submit(self, owner, attribute, label):
        """Temporarily bind a writable project-owned collective/replay callable."""
        self._bind_attribute(owner, attribute, label, None)

    def bind_protocol(self, owner, attribute, label, *, refusal=()):
        """Temporarily bind capture/protocol entry, before any nested submits."""
        self._bind_attribute(owner, attribute, label, refusal)

    def _bind_attribute(self, owner, attribute, label, refusal):
        original = getattr(owner, attribute)
        dictionary = getattr(owner, "__dict__", None)
        descriptor = inspect.getattr_static(type(owner), attribute, None)
        owned = dictionary is None or attribute in dictionary or inspect.isdatadescriptor(descriptor)
        bound = self._bound_operation(label, original, refusal)
        self.bindings.append(("attribute", owner, attribute, owned, original, bound))
        setattr(owner, attribute, bound)

    def bind_mapping_submit(self, mapping, key, label):
        """Bind an existing committed replay entry; its key/value stays stable."""
        original = mapping[key]
        bound = self._bound_operation(label, original, None)
        self.bindings.append(("mapping", mapping, key, True, original, bound))
        mapping[key] = bound

    @staticmethod
    def _restore_binding(binding):
        kind, owner, key, owned, original, bound = binding
        if kind == "mapping":
            current = owner.get(key)
            if current is bound:
                owner[key] = original
            elif current is not original:
                raise RuntimeError("construction replay binding changed before retirement")
        elif owned:
            setattr(owner, key, original)
        elif key in vars(owner):
            delattr(owner, key)

    def retire_bindings(self, target):
        """Restore/drop this owner/map BEFORE close resets resources or clears it."""
        selected = [binding for binding in reversed(self.bindings) if binding[1] is target]
        finish(lambda binding=binding: self._restore_binding(binding) for binding in selected)
        self.bindings[:] = [binding for binding in self.bindings if binding[1] is not target]

    def outcomes(self, local):
        """All-rank terminal vote; no model is published on a partial success."""
        self.put(f"done/{self.comm.rank}", local)
        keys = [self.key(f"done/{rank}") for rank in range(self.comm.world)]
        self.comm.store.wait(keys, timedelta(hours=1))
        result = [json.loads(self.comm.store.get(key)) for key in keys]
        # An all-read acknowledgement prevents deleting another reader's keys.
        self.put(f"read/{self.comm.rank}", True)
        if self.comm.rank == 0:
            self.comm.store.wait([self.key(f"read/{rank}") for rank in range(self.comm.world)], timedelta(hours=1))
            steps = [item.get("step", self.step) for item in result]
            for step in range(max(0, min(steps)-1), max(steps)+1):
                if step >= 0:
                    for suffix in [f"go/{step}", *(f"ready/{step}/{r}" for r in range(self.comm.world))]:
                        self.comm.store.delete_key(self.key(suffix))
            for rank in range(self.comm.world):
                for prefix in ("done", "failure", "read"):
                    self.comm.store.delete_key(self.key(f"{prefix}/{rank}"))
        return result
