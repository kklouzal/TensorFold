"""Actual ReadAhead close with real Futures/threads; no ndarray/native claims."""
from __future__ import annotations

import ast
from concurrent.futures import Future, ThreadPoolExecutor
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/tensorfold/families/qwen4_exp/read_ahead.py'
spec = importlib.util.spec_from_file_location('read_ahead_close_cleanup_source', ROOT / 'src/tensorfold/cleanup.py')
CLEANUP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLEANUP)
tree = ast.parse(SOURCE.read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ReadAhead')
SCOPE = {'ThreadPoolExecutor': ThreadPoolExecutor, 'threading': threading, 'raise_failures': CLEANUP.raise_failures}
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), 'exec'), SCOPE)
ReadAhead = SCOPE['ReadAhead']


def statuses(error):
    seen, pending = [], [error]
    while pending:
        value = pending.pop()
        if any(value is old for old in seen):
            continue
        seen.append(value)
        for related in (BaseException.__cause__.__get__(value), BaseException.__context__.__get__(value)):
            if related is not None:
                pending.append(related)
        if type(value) in (BaseExceptionGroup, ExceptionGroup):
            pending.extend(value.exceptions)
    return seen


class Close(unittest.TestCase):
    def test_final_close_publication_interrupt_preserves_completed_work_failure(self):
        owner = self.owner()
        work, interruption = OSError('completed read failure'), KeyboardInterrupt('final Condition exit')
        future = Future()
        future.set_exception(work)
        owner._outstanding.add(future)
        actual = owner._condition

        class FinalExit:
            armed = True
            def __enter__(self):
                return actual.__enter__()
            def __exit__(self, *args):
                result = actual.__exit__(*args)
                if self.armed and owner._closed:
                    self.armed = False
                    raise interruption
                return result
            def __getattr__(self, name):
                return getattr(actual, name)

        owner._condition = FinalExit()
        with self.assertRaises(KeyboardInterrupt) as caught:
            owner.close()
        self.assertIs(caught.exception, interruption)
        self.assertIn(work, statuses(caught.exception))
        self.assertTrue(owner._closed)
        self.assertFalse(owner._retired or owner._outstanding or owner._ahead)
        owner.close()

    def test_status_allocation_failure_after_journal_transfer_retains_future(self):
        closed = []
        owner = self.owner(lambda: closed.append(True))
        future = Future()
        future.set_result(None)
        owner._outstanding.add(future)
        primary = MemoryError('close status allocation')

        class FailedRecords:
            def __iter__(self):
                raise primary

        owner._close_statuses = FailedRecords()
        with self.assertRaises(MemoryError) as caught:
            owner.close()
        self.assertIs(caught.exception, primary)
        self.assertFalse(owner._closing or owner._closed or closed)
        self.assertIn(future, owner._retired)
        owner._close_statuses = []
        owner.close()
        self.assertTrue(owner._closed)
        self.assertEqual(closed, [True])

    def test_condition_exit_after_journal_clear_retains_future_and_allows_retry(self):
        closed = []
        owner = self.owner(lambda: closed.append(True))
        future = Future()
        future.set_result(None)
        owner._outstanding.add(future)
        actual = owner._condition
        primary = KeyboardInterrupt('Condition exit after releasing lock')

        class ExitOnce:
            armed = True
            def __enter__(self):
                return actual.__enter__()
            def __exit__(self, *args):
                result = actual.__exit__(*args)
                if self.armed:
                    self.armed = False
                    raise primary
                return result
            def __getattr__(self, name):
                return getattr(actual, name)

        owner._condition = ExitOnce()
        with self.assertRaises(KeyboardInterrupt) as caught:
            owner.close()
        self.assertIs(caught.exception, primary)
        self.assertFalse(owner._closing or owner._closed or closed)
        self.assertIn(future, owner._retired)
        owner.close()
        self.assertTrue(owner._closed)
        self.assertEqual(closed, [True])

    def owner(self, close=lambda: None):
        return ReadAhead(SimpleNamespace(gather=lambda _: None, close=close))

    def test_all_real_terminal_failures_and_cleanup_identity_survive(self):
        work = [OSError('read0'), EOFError('read1'), LookupError('read2')]
        cleanup = MemoryError('table close')

        def fail_close():
            raise cleanup

        owner = self.owner(fail_close)
        for index, error in enumerate(work):
            future = Future()
            future.set_exception(error)
            owner._ahead[str(index)] = future
        with self.assertRaises(BaseException) as caught:
            owner.close()
        actual = statuses(caught.exception)
        self.assertTrue(all(any(error is value for value in actual) for error in [*work, cleanup]))
        self.assertTrue(owner._closed and owner._cleanup_pending)
        self.assertFalse(owner._ahead or owner._retired or owner._outstanding)
        owner.table.close = lambda: None
        owner.close()
        self.assertFalse(owner._cleanup_pending)

    def test_first_native_roots_survive_successful_foreign_table_close(self):
        error, native, context = OSError(), EOFError(), LookupError()
        error.__cause__, error.__context__ = native, context

        def close():
            error.__cause__ = error.__context__ = None

        owner = self.owner(close)
        future = Future()
        future.set_exception(error)
        owner._outstanding.add(future)
        with self.assertRaises(OSError) as caught:
            owner.close()
        self.assertIs(caught.exception, error)
        self.assertIs(BaseException.__cause__.__get__(error), native)
        self.assertIs(BaseException.__context__.__get__(error), context)

    def test_failed_shutdown_retry_keeps_real_futures_and_prior_statuses(self):
        work, first, second = OSError(), RuntimeError('first shutdown'), EOFError('retry shutdown')
        native, context = LookupError(), ValueError()
        first.__cause__, first.__context__ = native, context
        closed = []
        owner = self.owner(lambda: closed.append(True))
        pool = owner._pool

        class Pool:
            calls = 0
            def shutdown(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise first
                first.__cause__ = first.__context__ = None
                if self.calls == 2:
                    raise second
                pool.shutdown(**kwargs)

        owner._pool = Pool()
        future = Future()
        future.set_exception(work)
        owner._retired.add(future)
        for expected in (first, second):
            with self.assertRaises(type(expected)) as caught:
                owner.close()
            self.assertIs(caught.exception, expected)
            self.assertIn(future, owner._retired)
            self.assertFalse(owner._closed or closed)
        with self.assertRaises(OSError) as caught:
            owner.close()
        actual = statuses(caught.exception)
        self.assertTrue(all(any(error is value for value in actual) for error in (work, first, second, native, context)))
        self.assertEqual(closed, [True])
        self.assertTrue(owner._closed)

    def test_real_running_future_drains_before_table_release(self):
        entered, release, table_closed = threading.Event(), threading.Event(), threading.Event()
        owner = self.owner(table_closed.set)

        def work():
            entered.set()
            if not release.wait(10):
                raise TimeoutError('real Future fixture release')
            self.assertFalse(table_closed.is_set())

        future = owner._pool.submit(work)
        owner._outstanding.add(future)
        self.assertTrue(entered.wait(3))
        with ThreadPoolExecutor(1) as control:
            done = control.submit(owner.close)
            try:
                with owner._condition:
                    self.assertTrue(owner._condition.wait_for(lambda: owner._closing, 3))
                self.assertFalse(done.done() or table_closed.is_set())
            finally:
                release.set()
            done.result(timeout=3)
        self.assertTrue(owner._closed and table_closed.is_set())


if __name__ == '__main__':
    unittest.main()
