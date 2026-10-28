"""Finite diagnostic host counters. No device inspection or serving imports.

One installed plan owns fixed bounded counters per rank process. The serialized
coordinator writes; snapshots occur at warm/case/drain boundaries after writers
retire. Times include Python/host waits and profiler overhead, not GPU durations.
"""
from functools import wraps
from contextlib import contextmanager
import time

_LIMIT = (1 << 63) - 1
_PHASES = ("round", "selection", "compute_decode", "compute_prefill", "owner_prepare", "mtp_owner_prepare")


class _Counters:
    def __init__(self):
        self.calls = [0] * len(_PHASES)
        self.elapsed_ns = [0] * len(_PHASES)
        self.round_live = [0] * 26
        self.ready_segments = [0] * 26
        self.selected_segments = [0] * 26
        self.selected_rows = [0] * 34
        self.compute_rows = [0] * 34
        self.compute_segments = [0] * 26
        self.owner_rows = [0] * 34
        self.overflowed = False

    def add(self, values, index, amount=1):
        total = values[index] + amount
        if total > _LIMIT:
            self.overflowed = True
            return
        values[index] = total

    def bucket(self, values, number):
        self.add(values, min(max(0, number), len(values) - 1))

    def snapshot(self):
        return {"scope": "submitted host calls and selected batch geometry; not GPU completion/timing",
                "overflowed": self.overflowed, "phases": {
                    name: {"calls": self.calls[i], "host_elapsed_ns": self.elapsed_ns[i]}
                    for i, name in enumerate(_PHASES)},
                **{name: list(getattr(self, name)) for name in (
                    "round_live", "ready_segments", "selected_segments", "selected_rows",
                    "compute_rows", "compute_segments", "owner_rows")},
                "histogram_contract": "segments0..24+25overflow; rows0..32+33overflow; prefill counted separately",
                "bounded_state": "7 fixed histograms, 12 int64 accumulators, one overflow latch; no retained inputs"}


_owner = None


def observer_snapshot():
    return None if _owner is None else _owner.snapshot()


def observer_plan(record_function):
    """Return transparent (owner, name, original, wrapper) bindings, once."""
    global _owner
    if _owner is not None:
        raise RuntimeError("concurrent observer plan is already owned by this rank")
    from tensorfold.families.qwen4_exp.cuda import (
        multi, forward, multi_graphs, bound_graphs, history_owner, adaptive_controller,
    )

    state = _Counters()
    _owner = state
    bindings = []

    def raise_diagnostic(primary, diagnostic):
        try:
            errors = (diagnostic,)
        except BaseException as allocation:
            if primary is not None:
                raise primary from allocation
            raise diagnostic from allocation
        from tensorfold.cleanup import raise_failures

        raise_failures(primary, errors)

    @contextmanager
    def label(name):
        owner = record_function(name)
        owner.__enter__()
        primary = None
        try:
            yield
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                owner.__exit__(None, None, None)
            except BaseException as diagnostic:
                raise_diagnostic(primary, diagnostic)

    def finish(phase, before, primary):
        try:
            state.add(state.calls, phase)
            state.add(state.elapsed_ns, phase, time.perf_counter_ns() - before)
        except BaseException as diagnostic:
            raise_diagnostic(primary, diagnostic)

    def wrapped(original, phase, geometry=None):
        @wraps(original)
        def invoke(*args, **kwargs):
            if geometry is not None:
                geometry(args, kwargs)
            before = time.perf_counter_ns()
            primary = None
            try:
                with label("tensorfold/body/concurrent/" + _PHASES[phase]):
                    return original(*args, **kwargs)
            except BaseException as error:
                primary = error
                raise
            finally:
                finish(phase, before, primary)
        return invoke

    def round_geometry(args, _kwargs):
        decoder = args[0]
        state.bucket(state.round_live, sum(not stream.done for stream in decoder.streams.values()))

    original = multi.MultiDecoder._round
    bindings.append((multi.MultiDecoder, "_round", original, wrapped(original, 0, round_geometry)))

    original = adaptive_controller.AdaptiveController.select

    @wraps(original)
    def select(controller, work, plan, _original=original):
        state.bucket(state.ready_segments, len(work))
        before = time.perf_counter_ns()
        primary = None
        try:
            with label("tensorfold/body/concurrent/selection"):
                result = _original(controller, work, plan)
            rows = dict(work)
            state.bucket(state.selected_segments, len(result))
            state.bucket(state.selected_rows, sum(rows[sid] for sid in result))
            return result
        except BaseException as error:
            primary = error
            raise
        finally:
            finish(1, before, primary)

    bindings.append((adaptive_controller.AdaptiveController, "select", original, select))
    original_compute = forward.compute

    @wraps(original_compute)
    def compute(weights, segments, buffers, *args, **kwargs):
        prefill = bool(buffers.prefill)
        phase = 3 if prefill else 2
        if not prefill:
            state.bucket(state.compute_rows, segments[-1][2] if segments else 0)
            state.bucket(state.compute_segments, len(segments))
        before = time.perf_counter_ns()
        primary = None
        try:
            with label("tensorfold/body/concurrent/" + _PHASES[phase]):
                return original_compute(weights, segments, buffers, *args, **kwargs)
        except BaseException as error:
            primary = error
            raise
        finally:
            finish(phase, before, primary)

    for module in (forward, multi, multi_graphs, bound_graphs):
        if getattr(module, "compute", None) is original_compute:
            bindings.append((module, "compute", original_compute, compute))
    original_prepare = history_owner.prepare

    @wraps(original_prepare)
    def prepare(projection, stream_state, layer, weights, buffers, start, stop, *, mtp):
        state.bucket(state.owner_rows, stop - start)
        phase = 5 if mtp else 4
        before = time.perf_counter_ns()
        primary = None
        try:
            with label("tensorfold/body/concurrent/" + _PHASES[phase]):
                return original_prepare(projection, stream_state, layer, weights, buffers, start, stop, mtp=mtp)
        except BaseException as error:
            primary = error
            raise
        finally:
            finish(phase, before, primary)

    bindings.append((history_owner, "prepare", original_prepare, prepare))
    return bindings
