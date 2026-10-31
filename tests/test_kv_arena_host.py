"""Attention-cache arena failure paths on the host, with stand-ins for libcuda and torch: a failed map undoes its
partial steps, out of memory retries once the queued releases are done, a failed release leaves the rest queued and
never releases anything twice, and a device the admission cannot count for is refused (``tests/cuda/test_kv_arena.py``
runs the real driver)."""

import sys
import threading
from types import SimpleNamespace

import pytest

from tensorfold.cuda import kv_arena
from tensorfold.cuda.geometry import ARENA_GRANULE

G = 2 << 20


class _OutOfMemory(Exception):
    """Stands in for ``torch.OutOfMemoryError``."""


class _Cuda:
    def __init__(self, log):
        self.log, self.capturing = log, False

    def is_current_stream_capturing(self):
        return self.capturing

    def synchronize(self, device=None):
        self.log.append(("device sync", device))


class _Event:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def synchronize(self):
        self.log.append(("event", self.name))


def _driver(failures=None):
    """A ``_Driver`` without libcuda: ``call`` logs each driver call and raises the queued failures for its name."""

    log, failures = [], failures or {}
    d = object.__new__(kv_arena._Driver)
    d.torch = SimpleNamespace(cuda=_Cuda(log), OutOfMemoryError=_OutOfMemory)
    d.lock, d.pending, d.mapped, d.device, d.granularity = threading.RLock(), [], 0, 0, G
    d.retirement_error = None
    d.prop, d.access = kv_arena._Prop(), kv_arena._Access()
    handles = iter(range(100, 200))

    def call(name, *args):
        if name == "cuMemCreate":
            args[0]._obj.value = next(handles)          # the handle the driver hands back
            args = (args[0]._obj.value, *args[1:])
        log.append((name, *args))
        if failures.get(name):
            raise failures[name].pop(0)

    d.call = call
    return d, log


def _queue(d, name, base, sizes, event=True):
    """An arena whose last view went: its chunks (offset, bytes, handle) mapped from ``base``."""

    chunks, offset = [], 0
    for i, nbytes in enumerate(sizes):
        chunks.append((offset, nbytes, base + i))
        offset += nbytes
    d.pending.append((_Event(d.torch.cuda.log, name) if event else None, base, 1 << 30, chunks))
    d.mapped += offset


def _released(log):
    return [entry for entry in log if entry[0] in ("cuMemUnmap", "cuMemRelease", "cuMemAddressFree")]


@pytest.mark.parametrize("stage", ["capture", "event", "stream", "record"])
def test_finalizer_completion_failure_retains_identity_and_surfaces_at_owner_operations(monkeypatch, stage):
    d, log = _driver()
    failure = RuntimeError(f"{stage} failed")

    def operation(name, result):
        def invoke(*args):
            assert d.pending and d.pending[0][1:3] == [1000, G]  # ownership precedes fallible CUDA work
            if stage == name:
                raise failure
            return result
        return invoke

    event = SimpleNamespace(record=operation("record", None))
    d.torch.cuda.is_current_stream_capturing = operation("capture", False)
    d.torch.cuda.Event = operation("event", event)
    d.torch.cuda.current_stream = operation("stream", object())
    chunks = [(0, G, 7)]
    d.mapped = G
    d.retire(1000, G, chunks)                           # finalizer itself cannot propagate to an owning caller
    assert d.retirement_error is failure and d.pending[0][3] is chunks and d.mapped == G
    monkeypatch.setattr(kv_arena, "_DRIVERS", [d])
    monkeypatch.setattr(kv_arena, "_driver", lambda device: d)
    for owner_operation in (d.reap, d.pending_bytes, kv_arena.mapped_bytes, lambda: kv_arena.driver(0)):
        with pytest.raises(RuntimeError, match="completion; mappings retained") as caught:
            owner_operation()
        assert caught.value.__cause__ is failure
    assert _released(log) == [] and d.pending[0][3] is chunks


def test_cached_driver_startup_reaps_valid_retirements_before_admission(monkeypatch):
    d, log = _driver()
    _queue(d, "completed", 1000, [G])
    monkeypatch.setattr(kv_arena, "_driver", lambda device: d)
    assert kv_arena.driver(0) is d
    assert d.pending == [] and d.mapped == 0 and _released(log)


def test_a_failed_release_raises_leaves_the_rest_queued_and_never_releases_twice():
    d, log = _driver({"cuMemUnmap": [RuntimeError("cuMemUnmap failed")]})
    _queue(d, "a", 1000, [G, G])
    _queue(d, "b", 2000, [G])
    with pytest.raises(RuntimeError, match="cuMemUnmap"):
        d.reap()
    # a's last chunk failed to unmap: it left the queue and stays mapped; the rest of a and all of b wait
    assert d.mapped == 3 * G and d.pending_bytes() == 2 * G and [item[1] for item in d.pending] == [1000, 2000]
    d.reap()
    released = _released(log)
    assert len(released) == len(set(released))                         # nothing released twice
    assert ("cuMemRelease", 1001) not in released                       # the chunk that failed to unmap
    assert ("cuMemAddressFree", 1000, 1 << 30) in released and ("cuMemAddressFree", 2000, 1 << 30) in released
    assert d.pending == [] and d.mapped == G                            # what the failure kept stays counted


def test_a_failed_reservation_free_is_not_retried():
    d, log = _driver({"cuMemAddressFree": [RuntimeError("cuMemAddressFree failed")]})
    _queue(d, "a", 1000, [G])
    _queue(d, "b", 2000, [G])
    with pytest.raises(RuntimeError, match="cuMemAddressFree"):
        d.reap()
    assert [item[1] for item in d.pending] == [2000] and d.mapped == G
    d.reap()
    assert [entry for entry in _released(log) if entry[0] == "cuMemAddressFree"] == [
        ("cuMemAddressFree", 1000, 1 << 30), ("cuMemAddressFree", 2000, 1 << 30)]
    assert d.pending == [] and d.mapped == 0


def test_a_capture_time_release_waits_for_the_capture_then_syncs_the_device():
    d, log = _driver()
    _queue(d, "captured", 1000, [G], event=False)
    _queue(d, "a", 2000, [G])
    d.torch.cuda.capturing = True
    d.reap()                                                            # the capture-time one stays queued
    assert [item[1] for item in d.pending] == [1000] and ("event", "a") in log and d.mapped == G
    d.torch.cuda.capturing = False
    d.reap()
    assert ("device sync", 0) in log and d.pending == [] and d.mapped == 0


def _arena(d, capacity=10_000, row_bytes=2048):
    a = object.__new__(kv_arena.Arena)
    a.drv, a.capacity, a.row_bytes, a.base, a.mapped, a.chunks = d, capacity, row_bytes, 1 << 40, 0, []
    return a


@pytest.mark.parametrize("failing,undone", [("cuMemMap", [("cuMemRelease", 100)]),
                                            ("cuMemSetAccess", [("cuMemUnmap", 1 << 40, 4 * G),
                                                                ("cuMemRelease", 100)])])
def test_a_failed_map_undoes_its_partial_steps(failing, undone):
    d, log = _driver({failing: [RuntimeError(f"{failing} failed")]})
    a = _arena(d)
    with pytest.raises(RuntimeError, match=failing):
        a.ensure(4000)                                                  # 4,000 rows of 2 KiB: four granules
    assert log[-len(undone):] == undone and a.chunks == [] and a.mapped == 0 and d.mapped == 0


def test_a_failing_undo_is_noted_on_the_primary_error():
    d, _ = _driver({"cuMemMap": [RuntimeError("cuMemMap failed")], "cuMemRelease": [RuntimeError("release failed")]})
    with pytest.raises(RuntimeError, match="cuMemMap") as raised:
        _arena(d).ensure(10)
    assert any("cuMemRelease also failed" in note for note in raised.value.__notes__)


def test_out_of_memory_retries_once_after_the_queued_releases():
    d, log = _driver({"cuMemCreate": [_OutOfMemory("oom")]})
    _queue(d, "gone", 5000, [G])
    a = _arena(d)
    a.ensure(1)                                                         # queued release, failed create, then maps
    names = [entry[0] for entry in log]
    assert names.count("cuMemCreate") == 2 and names.index("cuMemAddressFree") < names.index("cuMemCreate")
    assert a.mapped == G and d.mapped == G and d.pending == []
    d, _ = _driver({"cuMemCreate": [_OutOfMemory("oom"), _OutOfMemory("oom again")]})
    with pytest.raises(_OutOfMemory, match="again"):
        _arena(d).ensure(1)


class _Fn:
    def __init__(self, body):
        self.body = body

    def __call__(self, *args):
        return self.body(*args)


def _libcuda(supported, granularity):
    def out(value):
        def body(pointer, *rest):
            pointer._obj.value = value
            return 0
        return body

    def name(code, pointer):
        pointer._obj.value = b"CUDA_ERROR_STAND_IN"
        return 0

    bodies = {"cuInit": lambda flags: 0, "cuDeviceGet": out(0), "cuDeviceGetAttribute": out(supported),
              "cuMemGetAllocationGranularity": out(granularity), "cuGetErrorName": name}
    lib = SimpleNamespace()
    for fn in ("cuInit", "cuDeviceGet", "cuDeviceGetAttribute", "cuMemGetAllocationGranularity", "cuMemAddressReserve",
               "cuMemAddressFree", "cuMemCreate", "cuMemRelease", "cuMemMap", "cuMemUnmap", "cuMemSetAccess",
               "cuGetErrorName"):
        setattr(lib, fn, _Fn(bodies.get(fn, lambda *args: 0)))
    return lib


@pytest.mark.parametrize("supported,granularity,refusal", [(0, G, "lacks virtual memory management"),
                                                           (1, 2 * ARENA_GRANULE, "do not divide"),
                                                           (1, 3 << 20, "do not divide"), (1, G, None),
                                                           (1, 64 << 10, None)])
def test_the_driver_refuses_a_device_the_admission_cannot_count_for(monkeypatch, supported, granularity, refusal):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(init=lambda: None)))
    monkeypatch.setattr(kv_arena.ctypes, "CDLL", lambda path: _libcuda(supported, granularity))
    if refusal is None:
        assert kv_arena._Driver(0).granularity == granularity           # a divisor: admission over-counts, safely
    else:
        with pytest.raises(RuntimeError, match=refusal):
            kv_arena._Driver(0)
