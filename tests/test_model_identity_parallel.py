"""Bounded independent-file hashing; Python FD transport is not native FD proof."""
import concurrent.futures
import hashlib
import threading
from pathlib import Path

import pytest

from snapshot_fd_transport import TransportOwner
from tensorfold.engine import model_identity as identity


@pytest.fixture(autouse=True)
def descriptor_transport(monkeypatch):
    monkeypatch.setattr('tensorfold.file_io._owned_slot', TransportOwner)


def inputs(tmp_path, count=4):
    files = {}
    for index in range(count):
        path = tmp_path / str(index)
        path.write_bytes(bytes([index]) * (1024 + index))
        files[str(index)] = path
    return files


def capture(files, **kwargs):
    return identity.capture_model_identity(files, runtime_identity={'loader': 'parallel-control'},
        authorize=kwargs.pop('authorize', lambda _name, path: path.resolve(strict=True)),
        max_file_bytes=kwargs.pop('max_file_bytes', 1 << 20),
        max_total_bytes=kwargs.pop('max_total_bytes', 1 << 20), **kwargs)


def roots(error, seen=None):
    seen = set() if seen is None else seen
    if error is None or id(error) in seen:
        return []
    seen.add(id(error))
    return [error, *roots(error.__cause__, seen), *roots(error.__context__, seen),
            *(item for child in getattr(error, 'exceptions', ()) for item in roots(child, seen))]


def test_two_real_hashes_overlap_and_identity_order_matches_independent_hashes(tmp_path, monkeypatch):
    files = inputs(tmp_path, 6)
    serial = capture(files)
    actual, barrier, lock = identity._hash_descriptor, threading.Barrier(2, timeout=10), threading.Lock()
    active = peak = 0
    def hash_(fd, size):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait()
            return actual(fd, size)
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(identity, '_hash_descriptor', hash_)
    parallel = capture(dict(reversed(list(files.items()))), hash_workers=2)
    assert peak == 2 and active == 0
    assert parallel.model_prefix == serial.model_prefix
    assert parallel.files == serial.files
    assert [entry.sha256 for entry in parallel.files] == [hashlib.sha256(files[name].read_bytes()).hexdigest() for name in sorted(files)]
    assert parallel.bytes_read == serial.bytes_read and parallel.hashes_reused == 0
    assert not any(thread.name.startswith('model-content') for thread in threading.enumerate())


@pytest.mark.parametrize('workers', [True, False, 0, 3, 2.0, None])
def test_worker_domain_rejects_before_authorize_or_hash(tmp_path, workers):
    files = inputs(tmp_path)
    def forbidden(*args):
        raise AssertionError('effects before worker domain validation')
    with pytest.raises(ValueError):
        capture(files, hash_workers=workers, authorize=forbidden)


def test_pair_budget_and_authorization_are_logical_and_precede_hash(tmp_path, monkeypatch):
    files, calls = inputs(tmp_path, 2), []
    monkeypatch.setattr(identity, '_hash_descriptor', lambda *_: pytest.fail('hash before pair admission'))
    def authorize(name, path):
        calls.append(name)
        return path
    with pytest.raises(ValueError, match='total byte budget'):
        capture(dict(reversed(list(files.items()))), authorize=authorize, hash_workers=2, max_total_bytes=1024)
    assert calls == ['0', '1']


def test_two_worker_failures_choose_logical_primary_and_retire_before_close(tmp_path, monkeypatch):
    files = inputs(tmp_path, 2)
    errors = [ValueError('logical zero'), OSError('logical one')]
    earlier = LookupError('preexisting cause')
    errors[0].__cause__ = earlier
    barrier, finished = threading.Barrier(2, timeout=10), []
    sizes = [path.stat().st_size for path in files.values()]
    def hash_(fd, size):
        barrier.wait()
        index = sizes.index(size)
        finished.append(index)
        raise errors[index]
    class Checked(TransportOwner):
        def close(self):
            assert sorted(finished) == [0, 1]
            super().close()
    monkeypatch.setattr(identity, '_hash_descriptor', hash_)
    monkeypatch.setattr('tensorfold.file_io._owned_slot', Checked)
    with pytest.raises(ValueError) as caught:
        capture(files, hash_workers=2)
    assert caught.value is errors[0]
    assert errors[1] in roots(caught.value) and earlier in roots(caught.value)


def test_parallel_reuse_never_starts_threads_or_reads_payload(tmp_path, monkeypatch):
    files = inputs(tmp_path)
    original = capture(files, hash_workers=2)
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', lambda **_: pytest.fail('threads for reused hashes'))
    monkeypatch.setattr(identity, '_hash_descriptor', lambda *_: pytest.fail('reused payload read'))
    reused = capture(files, hash_workers=2, reuse=original, reuse_is_immutable=True)
    assert reused.model_prefix == original.model_prefix
    assert reused.bytes_read == 0 and reused.hashes_reused == len(files)


def test_mutation_after_worker_hash_prevents_publication(tmp_path, monkeypatch):
    files = inputs(tmp_path, 2)
    actual = identity._hash_descriptor
    def hash_(fd, size):
        result = actual(fd, size)
        if size == 1024:
            files['0'].write_bytes(b'changed same owner')
        return result
    monkeypatch.setattr(identity, '_hash_descriptor', hash_)
    with pytest.raises(ValueError, match='changed during'):
        capture(files, hash_workers=2)


def test_submission_and_close_failures_keep_primary_after_accepted_worker_join(tmp_path, monkeypatch):
    files = inputs(tmp_path, 2)
    primary, close_error = RuntimeError('second submission'), OSError('consumed close')
    real_submit, finished = concurrent.futures.ThreadPoolExecutor.submit, []
    count = 0
    def submit(pool, function, *args):
        nonlocal count
        count += 1
        if count == 2:
            raise primary
        def work():
            try:
                return function(*args)
            finally:
                finished.append(True)
        return real_submit(pool, work)
    class FaultClose(TransportOwner):
        def close(self):
            assert finished == [True]
            super().close()
            raise close_error
    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, 'submit', submit)
    monkeypatch.setattr('tensorfold.file_io._owned_slot', FaultClose)
    with pytest.raises(RuntimeError) as caught:
        capture(files, hash_workers=2)
    assert caught.value is primary and close_error in roots(primary)
    assert not any(thread.name.startswith('model-content') for thread in threading.enumerate())


def test_execution_manifest_explicitly_uses_two_workers_only_at_capture_callsite():
    import ast
    source = Path(identity.__file__).parents[1] / 'families/qwen4_exp/cuda/execution_identity.py'
    tree = ast.parse(source.read_text())
    call = next(node for node in ast.walk(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_capture'))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'capture_model_identity')
    assert next(node.value.value for node in call.keywords if node.arg == 'hash_workers') == 2


def test_failure_after_enqueue_still_joins_unreturned_future_before_fd_close(tmp_path, monkeypatch):
    files = inputs(tmp_path, 2)
    primary = RuntimeError('submit failed after queue ownership')
    real_submit, finished = concurrent.futures.ThreadPoolExecutor.submit, []
    count = 0
    def submit(pool, function, *args):
        nonlocal count
        count += 1
        current = count
        def work():
            try:
                return function(*args)
            finally:
                finished.append(current)
        future = real_submit(pool, work)
        if current == 2:
            raise primary
        return future
    class Checked(TransportOwner):
        def close(self):
            assert sorted(finished) == [1, 2]
            super().close()
    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, 'submit', submit)
    monkeypatch.setattr('tensorfold.file_io._owned_slot', Checked)
    with pytest.raises(RuntimeError) as caught:
        capture(files, hash_workers=2)
    assert caught.value is primary
    assert not any(thread.name.startswith('model-content') for thread in threading.enumerate())


def test_shutdown_interruptions_and_close_error_retained_after_positive_join(tmp_path, monkeypatch):
    files = inputs(tmp_path, 2)
    first, second, close_error = KeyboardInterrupt('first join interruption'), OSError('second join interruption'), OSError('close')
    real_shutdown = concurrent.futures.ThreadPoolExecutor.shutdown
    count = 0
    def shutdown(pool, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            raise first
        if count == 2:
            raise second
        return real_shutdown(pool, **kwargs)
    class FaultClose(TransportOwner):
        def close(self):
            assert count == 3
            super().close()
            raise close_error
    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, 'shutdown', shutdown)
    monkeypatch.setattr('tensorfold.file_io._owned_slot', FaultClose)
    with pytest.raises(KeyboardInterrupt) as caught:
        capture(files, hash_workers=2)
    assert caught.value is first
    assert second in roots(first) and close_error in roots(first)
    assert not any(thread.name.startswith('model-content') for thread in threading.enumerate())


def test_unreturned_callback_completion_precedes_corrupted_join_and_fd_close(tmp_path, monkeypatch):
    files = inputs(tmp_path, 2)
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    primary, join_error = RuntimeError('unreturned accepted callback'), KeyboardInterrupt('interrupted join')
    actual_hash, actual_submit = identity._hash_descriptor, concurrent.futures.ThreadPoolExecutor.submit
    actual_shutdown, actual_work = concurrent.futures.ThreadPoolExecutor.shutdown, identity._HashWork
    pools, shutdowns, completed, closed, outcome = [], [], [], [], []
    class ObservedWork(actual_work):
        def __init__(self, job):
            super().__init__(job)
            if job[0] == '1':
                wait = self.done.wait
                def observed_wait(timeout=None):
                    waiting.set()
                    return wait(timeout)
                self.done.wait = observed_wait
    def hash_(fd, size):
        if size == 1025:
            entered.set()
            assert release.wait(10), 'test release deadline'
            try:
                return actual_hash(fd, size)
            finally:
                completed.append(True)
        return actual_hash(fd, size)
    submits = 0
    def submit(pool, function, *args):
        nonlocal submits
        submits += 1
        pools.append(pool)
        future = actual_submit(pool, function, *args)
        if submits == 2:
            assert entered.wait(10), 'actual callback entry deadline'
            raise primary
        return future
    def broken_join(pool, **kwargs):
        shutdowns.append(True)
        if len(shutdowns) == 1:
            raise join_error
        # Model the supported CPython interrupted-join bookkeeping defect:
        # the next join reports stopped without proving callback completion.
        return None
    class Checked(TransportOwner):
        def close(self):
            assert completed == [True]
            closed.append(True)
            super().close()
    def operation():
        try:
            capture(files, hash_workers=2)
        except BaseException as error:
            outcome.append(error)
    monkeypatch.setattr(identity, '_HashWork', ObservedWork)
    monkeypatch.setattr(identity, '_hash_descriptor', hash_)
    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, 'submit', submit)
    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, 'shutdown', broken_join)
    monkeypatch.setattr('tensorfold.file_io._owned_slot', Checked)
    owner = threading.Thread(target=operation)
    owner.start()
    try:
        assert waiting.wait(10), 'independent completion wait not entered'
        assert not shutdowns and not closed and not completed
    finally:
        release.set()
        owner.join(10)
        for pool in set(pools):
            actual_shutdown(pool, wait=True)
    assert not owner.is_alive() and outcome == [primary]
    assert join_error in roots(primary) and completed == [True] and len(closed) == 2


def test_preallocated_callback_cancel_prevents_late_fd_access():
    work = identity._HashWork(('cancelled', None, -1, (0, 0, 0)))
    work.cancel_unstarted()
    assert work.done.is_set() and work.job is None
    work.run()
    assert not work.entered and work.result is None and work.error is None


def test_post_drain_transport_exhaustion_keeps_logical_file_primary_and_statuses(tmp_path, monkeypatch):
    from tensorfold import cleanup
    files = inputs(tmp_path, 2)
    first, second, allocation = ValueError('logical0'), OSError('logical1'), MemoryError('transport')
    prior = LookupError('original cause')
    first.__cause__ = prior
    def hash_(fd, size):
        raise first if size == 1024 else second
    def exhausted(*args):
        raise allocation
    monkeypatch.setattr(identity, '_hash_descriptor', hash_)
    monkeypatch.setattr(cleanup, 'raise_failures', exhausted)
    with pytest.raises(ValueError) as caught:
        capture(files, hash_workers=2)
    assert caught.value is first and first.__cause__ is allocation
    frame = first.__traceback__
    while frame is not None and frame.tb_frame.f_code.co_name != '_hash_pair':
        frame = frame.tb_next
    assert frame is not None
    retained = frame.tb_frame.f_locals
    assert first in retained['statuses'] and second in retained['statuses']
    assert retained['prior_cause'] is prior
    assert all(work.done.is_set() for work in retained['works'])
