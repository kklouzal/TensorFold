"""Actual pinned descriptor lifetime, not numerical/performance/model qualification.

CUDA12.9 cudaLaunchHostFunc orders a host-only callback in a stream; later work
cannot proceed until it returns. The callback calls NO CUDA/Torch API. Event
queries plus explicit host barriers establish pending frontiers, never sleeps.
https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__EXECUTION.html
"""
import ctypes
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

from tensorfold.cuda.sm70.software_binding import issue_source_binding, verify_source_binding
from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
from tensorfold.families.qwen4_exp.cuda.state import State
from tests.test_native_owner_adapter import _weights
from tools.worker_lifetime import Task, raise_failures

DEADLINE = 20  # Failure/cleanup completion bound, never a latency pass criterion.
# Terminal test ownership only: retain an uncertain CUDA reader and its callback
# through process exit. At most one test may poison this module; no new work follows.
_UNRETIRED = [None, [None], [None]]  # owner, one gate, one accepted host Task


@pytest.fixture
def actual_target():
    if _UNRETIRED[0] is not None:
        pytest.exit("Q05 stream retirement is uncertain; retained owners require process exit", returncode=1)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires installed pinned SM70; collection is not qualification")
    return issue_source_binding(required=True)


def _owner(rows=8, fmt="int8"):
    w = _weights()
    w.device = torch.device("cuda")
    w.meta["attention_degree"] = 4
    states = [State(w, 257, 4, fmt, slot=slot) for slot in (1, 3)]
    owner = object.__new__(OwnerBuffers)
    owner._empty_resources()
    _UNRETIRED[0] = owner  # Fixed ownership root BEFORE allocation or foreign callback.
    try:
        owner.__init__(w, rows, NS(idw=2052))
        owner.prepare_native(states, buckets=(rows,))
    except BaseException as error:
        _finish(error, owner)
    return owner, states


def _words(owner, segments, layer, base, count, *, mtp=False):
    """Independent original Tensor-row oracle; no NumPy packing implementation."""
    expected = torch.zeros(owner.descriptors.shape, dtype=torch.int64)
    lane = 0
    for state, first, last in segments:
        begin, end = max(base, first), min(base + count, last)
        if begin >= end:
            continue
        ai = state.history_placement.target_layers if mtp else state.att_index[layer.index]
        cache = state.mtp_kc if mtp else state.kc[ai]
        position = state.mtp_pos if mtp else state.pos_dev
        live = owner.lane_rows.data_ptr() + lane * owner.lane_rows.element_size()
        for row in range(begin, end):
            expected[row - base] = torch.tensor((
                cache.k.data_ptr(), cache.v.data_ptr(), cache.ks.data_ptr(), cache.vs.data_ptr(),
                cache.page_offsets.data_ptr(), state.history_generations.data_ptr(), state.kv_status.data_ptr(),
                position.data_ptr(), live, state.slot_generation, state.history_generation,
                state.capacity, cache.capacity, cache.page_offsets.numel(), row - first,
                state.history_placement.indexer_owner(state.history_slot, ai, include_mtp=True),
            ), dtype=torch.int64)
        lane += 1
    return expected


def _wait_gpu(event):
    """Bounded completion polling; delay is pacing, not a race/pass threshold."""
    deadline, pause = time.monotonic() + DEADLINE, threading.Event()
    while not event.query():
        if time.monotonic() >= deadline:
            raise TimeoutError("Q05 device frontier did not complete")
        pause.wait(.001)


class _Gate:
    """Retains callback/CDLL until stream retirement; NO callback CUDA calls."""
    def __init__(self, stream):
        self.retired = False
        self.terminal = torch.cuda.Event()
        self.terminal.record(stream)
        _wait_gpu(self.terminal)  # allocate/warm event handle BEFORE held callback.
        binding = issue_source_binding(required=True)
        # Do not load another runtime. The required source issuer has audited the
        # process maps; NOLOAD retains only its already mapped owned provider.
        paths = {Path(line.split()[-1]).resolve() for line in Path("/proc/self/maps").read_text().splitlines()
                 if "libcudart.so" in line and len(line.split()) >= 6}
        assert len(paths) == 1, "exactly one audited loaded CUDA runtime required"
        path = next(iter(paths))
        assert path in binding._states, "runtime is outside issued software ownership"
        verify_source_binding(binding)
        self.library = ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
        version = self.library.cudaRuntimeGetVersion
        version.argtypes, version.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
        value = ctypes.c_int()
        assert version(ctypes.byref(value)) == 0 and value.value == 12090
        self.entered, self.release = threading.Event(), threading.Event()
        self.errors = [None]  # Fixed failure slot: no list growth across C boundary.
        timeout = TimeoutError("CUDA host-function gate was not released")
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

        @callback_type
        def callback(_unused):
            try:
                self.entered.set()
                if not self.release.wait(DEADLINE):
                    raise timeout
            except BaseException as error:
                # No exception crosses the C callback boundary. Owner reports it
                # after stream retirement; callback captures no device tensors.
                self.errors[0] = error

        self.callback, self.stream = callback, stream
        self.launch = self.library.cudaLaunchHostFunc
        self.launch.argtypes, self.launch.restype = [ctypes.c_void_p, callback_type, ctypes.c_void_p], ctypes.c_int
        self.queued = False

    def enqueue(self):
        if self.queued:
            raise RuntimeError("Q05 host gate is one-shot")
        # Caller already owns the fully initialized gate. Publish its raw callback
        # root before the foreign call, including ambiguous enqueue failure.
        _UNRETIRED[1][0] = self
        self.queued = True
        result = self.launch(ctypes.c_void_p(self.stream.cuda_stream), self.callback, None)
        if result != 0:
            raise RuntimeError("cudaLaunchHostFunc failed: " + str(result))

    def wait(self):
        if not self.entered.wait(DEADLINE):
            raise TimeoutError("CUDA host function did not reach its frontier")

    def close(self):
        primary, errors = None, []
        try:
            self.release.set()  # Independent release BEFORE completion polling.
        except BaseException as error:
            primary = error
        try:
            self.terminal.record(self.stream)
            _wait_gpu(self.terminal)
            self.retired = True
        except BaseException as error:
            errors.append(error)
        errors.extend(error for error in self.errors if error is not None)
        raise_failures(primary, errors)


def _finish(primary, owner):
    """Attempt independent cleanup; fixed roots survive interruption/uncertainty.

    A live Task can still enqueue work: physical owner.close is unsafe until its
    done journal proves callback quiescence. Thread.join/is_alive alone is NOT
    that proof (an interrupted CPython join can report a false stopped state).
    """
    _UNRETIRED[0] = owner
    errors, quiet = [], True
    for gate in _UNRETIRED[1]:
        if gate is not None:
            try:
                gate.release.set()
            except BaseException as error:
                errors.append(error)
    for task in _UNRETIRED[2]:
        if task is None:
            continue
        try:
            task.cancel_unstarted()
        except BaseException as error:
            errors.append(error)
        try:
            task.done.wait(DEADLINE)
        except BaseException as error:
            errors.append(error)
        if not task.done.is_set():
            quiet = False
            errors.append(TimeoutError("descriptor host callback did not retire"))
        elif task.started:
            try:
                task.thread.join(DEADLINE)
            except BaseException as error:
                errors.append(error)
        if task.error is not None:
            errors.append(task.error)
    for gate in _UNRETIRED[1]:
        if gate is not None:
            try:
                gate.close()
            except BaseException as error:
                errors.append(error)
            if not gate.retired:
                quiet = False
    if quiet:
        try:
            owner.close()
        except BaseException as error:
            quiet = False
            errors.append(error)
    # Otherwise close is deliberately deferred: a live host issuer or unknown
    # device reader forbids freeing this owner. Fixed roots remain through exit.
    if quiet:
        _UNRETIRED[1][0] = _UNRETIRED[2][0] = None
        _UNRETIRED[0] = None
    raise_failures(primary, errors)


@pytest.mark.parametrize("fmt", ["int8", "rotorquant-iso4-sm70-v1"])
@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("rows", [8, 256])
def test_alternating_tables_keep_pinned_identity_and_capacity_geometry(actual_target, fmt, mtp, rows):
    owner, states = _owner(rows, fmt)
    primary = None
    try:
        assert owner.descriptors_host.is_pinned()
        view, pointer = owner.descriptor_host_words, owner.descriptors_host.data_ptr()
        assert view.ctypes.data == pointer
        main, side = torch.cuda.current_stream(), torch.cuda.Stream()
        side.wait_stream(main)
        segments = [(states[0], 0, rows // 2), (states[1], rows // 2, rows)]
        layer = NS(index=2)
        results = []
        # Distinct clipped spans/generations, including inactive tails and both
        # nonzero slots. No per-iteration CPU table readback hides reuse ordering.
        for iteration, (base, count) in enumerate(((0, rows), (1, rows - 2), (rows // 2, 1)) * 2):
            stream = main if iteration % 2 == 0 else side
            with torch.cuda.stream(stream):
                for state in states:
                    state.begin_sequence_generation()
                expected = _words(owner, segments, layer, base, count, mtp=mtp)
                owner.stage_native_descriptors(segments, layer, base, count, mtp=mtp, bucket_rows=rows)
                snapshot = owner.descriptors.clone()
                owner.mark_native_submitted()  # clone is the last device-table reader.
                results.append((snapshot, expected))
            assert owner.descriptor_host_words is view and owner.descriptors_host.data_ptr() == pointer
        for snapshot, expected in results:
            assert torch.equal(snapshot.cpu(), expected)
        assert owner.native_record()["pinned_descriptor_bytes"] == 128 * rows
    except BaseException as error:
        primary = error
    _finish(primary, owner)


def test_pending_dma_retires_before_host_words_can_change(actual_target):
    owner, states = _owner()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    gate, task, primary = None, None, None
    reached = threading.Event()
    real = owner.descriptor_upload
    try:
        old = owner.descriptors_host.clone()
        # Initialize lazy event handles and all CUDA allocations before gating.
        real.record(side)
        real.synchronize()
        gate = _Gate(side)
        gate.enqueue()
        with torch.cuda.stream(side):
            owner.descriptors.copy_(owner.descriptors_host, non_blocking=True)
            real.record(side)
        gate.wait()
        assert not real.query()  # actual H2D completion is behind held callback.
        owner._upload_pending = True

        class Announce:
            def synchronize(self):
                reached.set()
                real.synchronize()  # REAL event; spy only identifies call entry.

            def record(self, stream):
                real.record(stream)

        owner.descriptor_upload = Announce()

        def stage():
            with torch.cuda.stream(side):
                owner.stage_native_descriptors([(states[1], 0, 1)], NS(index=2), 0, 1, bucket_rows=8)

        task = Task(stage, daemon=True)
        _UNRETIRED[2][0] = task  # BEFORE fallible start/possible callback entry.
        task.start()
        assert reached.wait(DEADLINE)
        assert not real.query() and torch.equal(owner.descriptors_host, old)
        gate.release.set()
        assert task.done.wait(DEADLINE)
        raise_failures(None, [task.error] if task.error is not None else [])
        _wait_gpu(real)
        expected = _words(owner, [(states[1], 0, 1)], NS(index=2), 0, 1)
        assert torch.equal(owner.descriptors.cpu(), expected)
    except BaseException as error:
        primary = error
    owner.descriptor_upload = real
    _finish(primary, owner)


def test_native_reader_event_orders_device_overwrite_without_host_reader_wait(actual_target):
    owner, states = _owner()
    gate, primary = None, None
    try:
        segments, layer = [(states[0], 0, 1)], NS(index=2)
        owner.stage_native_descriptors(segments, layer, 0, 1, bucket_rows=8)
        torch.cuda.synchronize()
        expected_old = _words(owner, segments, layer, 0, 1)
        reader, writer = torch.cuda.Stream(), torch.cuda.Stream()
        reader.wait_stream(torch.cuda.current_stream())
        old_read = torch.empty_like(owner.descriptors)
        completed = torch.cuda.Event()
        completed.record(writer)
        completed.synchronize()
        with torch.cuda.stream(reader):
            owner.mark_native_submitted()
        owner.native_done.synchronize()
        gate = _Gate(reader)
        gate.enqueue()
        with torch.cuda.stream(reader):
            old_read.copy_(owner.descriptors)  # Real queued DEVICE descriptor consumer.
            owner.mark_native_submitted()
        gate.wait()
        assert not owner.native_done.query()
        with torch.cuda.stream(writer):
            # Previous upload is already done; host staging may proceed, while
            # wait_native_reuse must order GPU table/lane writes after old_read.
            owner.stage_native_descriptors([(states[1], 0, 2)], layer, 0, 2, bucket_rows=8)
            completed.record(writer)
        assert not owner.native_done.query() and not completed.query()
        gate.release.set()
        _wait_gpu(completed)
        assert torch.equal(old_read.cpu(), expected_old)
        expected_new = _words(owner, [(states[1], 0, 2)], layer, 0, 2)
        assert torch.equal(owner.descriptors.cpu(), expected_new)
    except BaseException as error:
        primary = error
    _finish(primary, owner)


def test_real_storage_survives_authentic_host_retirement_failure_then_positive_close(actual_target):
    owner, states = _owner()
    view, mirror, state = weakref.ref(owner.descriptor_host_words), weakref.ref(owner.descriptors_host), weakref.ref(states[0])
    primary, failure = None, RuntimeError("injected host retirement failure with real device storage")
    original = owner.retire_native
    try:
        owner.stage_native_descriptors([(states[0], 0, 1)], NS(index=2), 0, 1, bucket_rows=8)
        submitted = torch.cuda.Event()
        submitted.record(torch.cuda.current_stream())
        submitted.synchronize()

        def fail():
            raise failure

        owner.retire_native = fail
        with pytest.raises(RuntimeError) as caught:
            owner.close()
        assert caught.value is failure
        assert not owner.closed and view() is not None and mirror() is not None and state() is states[0]
        assert owner.native_states[id(states[0])][0] is states[0]
        owner.retire_native = original
        owner.close()
        assert owner.closed and view() is None and mirror() is None
        assert owner not in states[0].history_native_readers
    except BaseException as error:
        primary = error
    owner.retire_native = original
    _finish(primary, owner)


def _failure_ids(error):
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend((current.__cause__, current.__context__))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return seen


@pytest.mark.parametrize("mode", ["raise", "status"])
def test_host_enqueue_is_rooted_before_ambiguous_foreign_failure(monkeypatch, mode):
    roots = [object(), [None], [None]]
    monkeypatch.setitem(globals(), "_UNRETIRED", roots)
    gate = object.__new__(_Gate)
    gate.queued, gate.stream, gate.callback = False, NS(cuda_stream=123), object()
    failure = RuntimeError("foreign enqueue failed")

    def launch(*_args):
        assert roots[1][0] is gate and gate.queued
        if mode == "raise":
            raise failure
        return 17

    gate.launch = launch
    with pytest.raises(RuntimeError) as caught:
        gate.enqueue()
    assert roots[1][0] is gate  # No traceback-lifetime dependency or list growth.
    if mode == "raise":
        assert caught.value is failure
    else:
        assert str(caught.value) == "cudaLaunchHostFunc failed: 17"


def test_host_cleanup_attempts_every_safe_owner_and_preserves_all_failures(monkeypatch):
    order = []
    primary = ValueError("original test failure")
    release, cancel, join, close, child, owner_error = (KeyboardInterrupt("release"), KeyboardInterrupt("cancel"),
                                                       KeyboardInterrupt("join"), OSError("gate close"),
                                                       RuntimeError("child"), OSError("owner close"))

    def fail(name, error):
        order.append(name)
        raise error

    owner = NS(close=lambda: fail("owner", owner_error))
    gate = NS(release=NS(set=lambda: fail("release", release)), retired=True,
              close=lambda: fail("gate", close))
    task = NS(cancel_unstarted=lambda: fail("cancel", cancel),
              done=NS(wait=lambda _: order.append("wait"), is_set=lambda: True),
              started=True, thread=NS(join=lambda _: fail("join", join)), error=child)
    roots = [owner, [gate], [task]]
    monkeypatch.setitem(globals(), "_UNRETIRED", roots)
    with pytest.raises(ValueError) as caught:
        _finish(primary, owner)
    assert caught.value is primary
    assert order == ["release", "cancel", "wait", "join", "gate", "owner"]
    assert {id(e) for e in (primary, release, cancel, join, close, child, owner_error)} <= _failure_ids(primary)
    assert roots[0] is owner and roots[1][0] is gate and roots[2][0] is task


def test_host_cleanup_defers_physical_close_until_task_and_gate_quiescent(monkeypatch):
    order = []
    owner = NS(close=lambda: pytest.fail("live issuer/unknown reader must retain its storage"))
    gate = NS(release=NS(set=lambda: order.append("release")), retired=False,
              close=lambda: order.append("gate"))
    task = NS(cancel_unstarted=lambda: order.append("cancel"),
              done=NS(wait=lambda _: order.append("wait"), is_set=lambda: False), started=True, error=None)
    roots = [owner, [gate], [task]]
    monkeypatch.setitem(globals(), "_UNRETIRED", roots)
    with pytest.raises(TimeoutError):
        _finish(None, owner)
    assert order == ["release", "cancel", "wait", "gate"]
    assert roots == [owner, [gate], [task]]
    task.done.is_set = lambda: True
    task.thread = NS(join=lambda _: order.append("join"))
    gate.retired = True
    owner.close = lambda: order.append("owner")
    _finish(None, owner)
    assert order[-1] == "owner" and roots == [None, [None], [None]]


def test_host_gate_close_reports_release_wait_and_callback_errors(monkeypatch):
    gate, order = object.__new__(_Gate), []
    release, wait, callback = KeyboardInterrupt("release"), OSError("completion"), RuntimeError("callback")

    def fail_release():
        order.append("release")
        raise release

    def fail_wait(_):
        order.append("wait")
        raise wait

    gate.release, gate.stream = NS(set=fail_release), object()
    gate.terminal = NS(record=lambda _: order.append("record"))
    gate.errors, gate.retired = [callback], False
    monkeypatch.setitem(globals(), "_wait_gpu", fail_wait)
    with pytest.raises(KeyboardInterrupt) as caught:
        gate.close()
    assert caught.value is release and order == ["release", "record", "wait"]
    assert {id(release), id(wait), id(callback)} <= _failure_ids(release)
    assert not gate.retired
