"""Concrete read-cleanup status controls with real stdlib Futures and threads.

Selected owned source methods execute without importing accelerator libraries.
Thread/stream substitutes test only resource and status contracts, never tensor
bytes, native IO, model quality, CUDA ordering or performance.
"""
from __future__ import annotations

import ast
import builtins
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def controls():
    spec = importlib.util.spec_from_file_location('read_cleanup_actual_transport', ROOT / 'src/tensorfold/cleanup.py')
    cleanup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cleanup)
    direct = ROOT / 'src/tensorfold/cuda/direct_read.py'
    selected = [node for node in ast.parse(direct.read_bytes()).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in ('ReadAhead', 'in_background', 'wait_all')]
    scope = dict(threading=threading, torch=NS(_C=NS()), Reader=object, raise_failures=cleanup.raise_failures)
    exec(compile(ast.fix_missing_locations(ast.Module([
        ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], [])),
        str(direct), 'exec'), scope)
    host = ROOT / 'src/tensorfold/cuda/exl3/host_experts.py'
    selected = [node for node in ast.parse(host.read_bytes()).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in ('CompactReadSession', '_finish_reads', '_joint_payloads')]
    scope.update(contextmanager=contextmanager, nullcontext=nullcontext, _READ_RUN=16 << 20,
                 _PackRanges=lambda pk: NS(pk=pk))
    exec(compile(ast.fix_missing_locations(ast.Module([
        ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], [])),
        str(host), 'exec'), scope)
    return NS(**scope), scope


def statuses(error):
    """Independent native exception graph oracle, including allocation fallback."""
    seen, pending = set(), [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        pending.extend((BaseException.__cause__.__get__(current), BaseException.__context__.__get__(current)))
        data = BaseException.__dict__['__dict__'].__get__(current)
        retained = dict.get(data, '_tensorfold_retained_failures', ())
        pending.extend(retained)
    return seen


class Controls(unittest.TestCase):
    def setUp(self):
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'numpy', 'triton', 'mlx', 'tensorfold', 'cuda', 'cupy', 'ctypes'}:
                raise AssertionError('SDK/native import forbidden in source controls')
            return original(name, *args, **kwargs)
        owner = patch.object(builtins, '__import__', guard)
        owner.start()
        self.addCleanup(owner.stop)
        self.api, self.scope = controls()

    def assertStatuses(self, caught, *errors):
        found = statuses(caught)
        self.assertTrue(all(id(error) in found for error in errors), 'concrete failure object lost')

    def failed(self, error):
        future = Future()
        future.set_exception(error)
        return future

    def test_wait_all_retains_every_distinct_terminal_future_error(self):
        first, second, root = ValueError('first'), OSError('second'), RuntimeError('prior cause')
        first.__cause__ = root
        futures = [self.failed(first), self.failed(second), self.failed(first)]
        with self.assertRaises(ValueError) as caught:
            self.api.wait_all(futures)
        self.assertIs(caught.exception, first)
        self.assertStatuses(caught.exception, first, second, root)
        self.assertEqual(futures, [])

    def test_malformed_notes_cannot_replace_multi_future_primary(self):
        first, second = RuntimeError('first'), OSError('second')
        first.__notes__ = object()
        with self.assertRaises(RuntimeError) as caught:
            self.api.wait_all([self.failed(first), self.failed(second)])
        self.assertIs(caught.exception, first)
        self.assertStatuses(caught.exception, first, second)

    def test_status_observation_interruption_retains_future_for_retry(self):
        primary, status = RuntimeError('job'), KeyboardInterrupt('status')
        future = self.failed(primary)
        original = future.done
        future.done = lambda: (_ for _ in ()).throw(status)
        futures = [future]
        with self.assertRaises(RuntimeError) as caught:
            self.api.wait_all(futures)
        self.assertStatuses(caught.exception, primary, status)
        self.assertEqual(futures, [future])
        future.done = original
        with self.assertRaises(RuntimeError):
            self.api.wait_all(futures)
        self.assertEqual(futures, [])

    def test_unpublished_job_observed_after_shutdown_failure_and_worker_reaped(self):
        primary, shutdown, job_error = ValueError('publication'), RuntimeError('shutdown'), OSError('worker')
        cause, context = LookupError('cause'), ArithmeticError('context')
        primary.__cause__, primary.__context__ = cause, context
        entered, released, completed = threading.Event(), threading.Event(), threading.Event()
        actual = ThreadPoolExecutor(max_workers=1)
        workers = []
        self.addCleanup(actual.shutdown)
        class Pool:
            def __init__(self, *args, **kwargs):
                pass
            def submit(self, callback):
                return actual.submit(callback)
            def shutdown(self, **kwargs):
                actual.shutdown(**kwargs)
                raise shutdown
        class Publication:
            def append(inner, value):
                self.assertTrue(entered.wait(2))
                released.set()
                raise primary
        def job():
            workers.append(threading.current_thread())
            entered.set()
            self.assertTrue(released.wait(2))
            completed.set()
            raise job_error
        with patch('concurrent.futures.ThreadPoolExecutor', Pool):
            with self.assertRaises(ValueError) as caught:
                self.api.in_background(job, Publication())
        self.assertIs(caught.exception, primary)
        self.assertStatuses(caught.exception, primary, cause, context, shutdown, job_error)
        self.assertTrue(completed.is_set())
        self.assertTrue(all(not thread.is_alive() for thread in workers))

    def test_close_preserves_pool_and_both_read_failures_with_retry_authority(self):
        first, second, shutdown = ValueError('read1'), OSError('read2'), RuntimeError('shutdown')
        owner = self.api.ReadAhead(reader=NS(), threads=1)
        one, two = self.failed(first), self.failed(second)
        owner._owned.update((one, two))
        owner.ahead.update(one=one, two=two)
        class Pool:
            calls = 0
            def shutdown(inner, **kwargs):
                inner.calls += 1
                if inner.calls == 1:
                    raise shutdown
        pool = owner.pool = Pool()
        with self.assertRaises(RuntimeError) as caught:
            owner.close()
        self.assertIs(caught.exception, shutdown)
        self.assertStatuses(caught.exception, shutdown, first, second)
        self.assertIs(owner.pool, pool)
        self.assertTrue(owner._owned)
        with self.assertRaises(BaseException):
            owner.close()
        self.assertIsNone(owner.pool)
        self.assertFalse(owner._owned)

    def test_completed_close_keeps_all_error_objects_after_clearing_journal(self):
        first, second = ValueError('read1'), OSError('read2')
        owner = self.api.ReadAhead(reader=NS(), threads=1)
        owner._owned.update((self.failed(first), self.failed(second)))
        with self.assertRaises(BaseException) as caught:
            owner.close()
        self.assertStatuses(caught.exception, first, second)
        self.assertFalse(owner._owned)
        owner.close()

    def test_success_reads_and_session_retirement_are_unchanged(self):
        owner = self.api.ReadAhead(reader=NS(read=lambda path, begin, size: bytes(range(begin, begin + size))),
                                   threads=2, run=4, gap=0)
        owner.queue([('one', 'fixture', 0, 4, None), ('two', 'fixture', 4, 8, None)], cut=lambda raw, _: raw)
        self.assertEqual(owner.take('one'), bytes(range(4)))
        self.assertEqual(owner.take('two'), bytes(range(4, 8)))
        owner.close()
        self.assertIsNone(owner.pool)
        self.assertEqual(owner.ahead, {})
        self.assertEqual(owner._owned, set())
        pack = object()
        session = self.api.CompactReadSession(pack)
        trace = []
        self.scope['_payload_iterator'] = lambda *args: NS(close=lambda: trace.append('iterator-close'))
        with self.api._joint_payloads(pack, [], session):
            self.assertTrue(session.active)
            trace.append('consumer')
        self.assertEqual(trace, ['consumer', 'iterator-close'])
        self.assertFalse(session.active)
        self.assertFalse(session.poisoned)
        session.close()
        self.assertTrue(session.closed)
        self.assertIsNone(session.pk)

    def test_session_context_exit_exposes_cleanup_without_foreign_formatting(self):
        primary, cleanup, cause = ValueError('consumer'), OSError('cleanup'), LookupError('cause')
        primary.__cause__ = cause
        cleanup.__notes__ = object()
        session = self.api.CompactReadSession(object())
        session.close = lambda: (_ for _ in ()).throw(cleanup)
        with self.assertRaises(ValueError) as caught:
            session.__exit__(ValueError, primary, None)
        self.assertIs(caught.exception, primary)
        self.assertStatuses(caught.exception, primary, cleanup, cause)

    def test_session_close_preserves_multiple_results_after_actual_thread_drain(self):
        first, second = ValueError('read1'), OSError('read2')
        session = self.api.CompactReadSession(object())
        actual = ThreadPoolExecutor(max_workers=2)
        workers, entered, released = [], threading.Barrier(3), threading.Event()
        self.addCleanup(actual.shutdown)
        def work(error):
            workers.append(threading.current_thread())
            entered.wait(2)
            self.assertTrue(released.wait(2))
            raise error
        one, two = actual.submit(work, first), actual.submit(work, second)
        session.ahead.pool = actual
        session.ahead._owned.update((one, two))
        session.ahead.ahead.update(one=one, two=two)
        session.queued.update({one: None, two: None})
        entered.wait(2)
        released.set()
        with self.assertRaises(BaseException) as caught:
            session.close()
        self.assertStatuses(caught.exception, first, second)
        self.assertTrue(session.closed)
        self.assertTrue(session.poisoned)
        self.assertIsNone(session.pk)
        self.assertTrue(all(not thread.is_alive() for thread in workers))

    def test_joint_consumer_failure_keeps_iterator_and_read_cleanup_statuses(self):
        primary, iterator_error, read_error = ValueError('consumer'), OSError('iterator'), RuntimeError('read')
        session = self.api.CompactReadSession(object())
        future = self.failed(read_error)
        self.scope['_payload_iterator'] = lambda *args: NS(close=lambda: (_ for _ in ()).throw(iterator_error))
        with self.assertRaises(ValueError) as caught:
            with self.api._joint_payloads(session.pk, [], session):
                session.ahead.ahead['one'] = future
                session.queued[future] = None
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertStatuses(caught.exception, primary, iterator_error, read_error)
        self.assertFalse(session.active)
        self.assertTrue(session.poisoned)
        self.assertFalse(session.queued)
        session.close()

    def test_joint_success_captures_first_cleanup_roots_before_future_cancel_callback(self):
        cleanup, cause = OSError('iterator'), LookupError('prior cause')
        cleanup.__cause__ = cause
        session = self.api.CompactReadSession(object())
        future = Future()
        callbacks = []
        def mutate(done):
            callbacks.append(done)
            cleanup.__cause__ = None
            cleanup.__context__ = None
        future.add_done_callback(mutate)
        self.scope['_payload_iterator'] = lambda *args: NS(close=lambda: (_ for _ in ()).throw(cleanup))
        with self.assertRaises(OSError) as caught:
            with self.api._joint_payloads(session.pk, [], session):
                session.ahead.ahead['pending'] = future
                session.queued[future] = None
        self.assertIs(caught.exception, cleanup)
        self.assertEqual(callbacks, [future])
        self.assertStatuses(caught.exception, cleanup, cause)
        self.assertFalse(session.active)
        self.assertTrue(session.poisoned)
        self.assertFalse(session.queued)
        session.close()


if __name__ == '__main__':
    unittest.main()
