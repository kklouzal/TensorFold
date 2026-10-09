"""Actual stdlib request journals; no HTTP/model/numerical runtime execution."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import signal
import threading
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def source(name):
    path = ROOT / 'tools' / (name + '.py')
    spec = importlib.util.spec_from_file_location('owned_benchmark_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / 'tools'))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


concurrent = source('bench_concurrent')
calibration = source('fit_draft_calibration')


class Lifetime(unittest.TestCase):
    def specs(self, count=4):
        return [(concurrent.PROMPTS[0], i) for i in range(count)]

    def args(self):
        return SimpleNamespace(base='unused', model='unused', tokens=32, streams=3, seed=7)

    def invoke(self, api, count=3):
        if api is concurrent:
            return api.together('unused', 'unused', self.specs(count), 32, 0.)
        return api.collect(self.args())

    def members(self, root):
        pending, result = [root], []
        while pending:
            value = pending.pop()
            if isinstance(value, BaseExceptionGroup):
                pending.extend(value.exceptions)
            else:
                result.append(value)
        return result

    def tracked(self, api, *, fail_start=None, primary=None):
        owners = []
        original = api.Task
        class Task(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                owners.append(self)
            def start(self):
                if len(owners) == fail_start:
                    raise primary
                super().start()
        return Task, owners

    def test_all_requests_start_together_and_results_preserve_request_order(self):
        lock, active, maximum = threading.Lock(), 0, 0
        def request(base, model, item, tokens, temperature, seed, gate=None):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(active, maximum)
            gate.wait(timeout=2)
            with lock:
                active -= 1
            return {'seed': seed}
        with patch.object(concurrent, 'stream', request):
            result = concurrent.together('unused', 'unused', self.specs(), 32, 0.)
        self.assertEqual([row['seed'] for row in result], list(range(4)))
        self.assertEqual(maximum, 4)

    def test_error_before_barrier_cannot_leave_peers_waiting_or_drop_results(self):
        def request(base, model, item, tokens, temperature, seed, draft, gate):
            if seed == 0:
                raise ValueError('request body rejected before barrier')
            gate.wait(timeout=2)
            return {'seed': seed}
        with patch.object(concurrent, '_stream', request):
            result = concurrent.together('unused', 'unused', self.specs(), 32, 0.)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(row.get('error') for row in result))
        self.assertEqual(concurrent.aggregates(result)['failed'], 4)

    def test_worker_base_exception_observed_after_every_owned_callback_completes(self):
        primary = KeyboardInterrupt('child request interrupted')
        Task, owners = self.tracked(concurrent)
        def request(base, model, item, tokens, temperature, seed, gate=None):
            if seed == 0:
                raise primary
            gate.wait(timeout=2)
            return {'seed': seed}
        with patch.object(concurrent, 'stream', request), patch.object(concurrent, 'Task', Task):
            with self.assertRaises(KeyboardInterrupt) as caught:
                concurrent.together('unused', 'unused', self.specs(), 32, 0.)
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(owners), 4)
        self.assertTrue(all(owner.done.is_set() and owner._target is None for owner in owners))

    def test_partial_start_failure_aborts_barrier_and_cancels_unstarted_owner(self):
        primary = RuntimeError('thread creation failed')
        Task, owners = self.tracked(concurrent, fail_start=3, primary=primary)
        def request(base, model, item, tokens, temperature, seed, gate=None):
            gate.wait(timeout=2)
            return {'seed': seed}
        with patch.object(concurrent, 'stream', request), patch.object(concurrent, 'Task', Task):
            with self.assertRaises(RuntimeError) as caught:
                concurrent.together('unused', 'unused', self.specs(), 32, 0.)
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(owners), 3)
        self.assertTrue(all(owner.done.is_set() for owner in owners))
        self.assertTrue(owners[2].start_cancelled)

    def test_invalid_stagger_and_empty_requests_start_no_owners(self):
        with patch.object(concurrent, 'Task', side_effect=AssertionError('no workers')):
            for value in (-1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    concurrent.together('unused', 'unused', self.specs(), 32, 0., value)
            self.assertEqual(concurrent.together('unused', 'unused', [], 32, 0.), [])

    def test_calibration_failure_fatal_without_completed_batch_report(self):
        primary = RuntimeError('calibration HTTP request failed')
        completed, lock = [], threading.Lock()
        def request(*args):
            with lock:
                completed.append(args)
            if args[-1] == 7:
                raise primary
        with patch.object(calibration, '_send', request), contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(RuntimeError) as caught:
                calibration.collect(self.args())
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(completed), 3)
        self.assertEqual(output.getvalue(), '')

    def test_calibration_success_every_original_prompt_temperature_seed(self):
        completed, lock = [], threading.Lock()
        def request(*args):
            with lock:
                completed.append(args)
        with patch.object(calibration, '_send', request), contextlib.redirect_stdout(io.StringIO()) as output:
            calibration.collect(self.args())
        expected = [('unused', 'unused', kind, text, 32, temperature, 7 + i)
                    for temperature in (1., 0.) for i, (kind, text) in enumerate(calibration.CORPUS)]
        self.assertCountEqual(completed, expected)
        self.assertIn('sent 32 of 32', output.getvalue())

    def test_invalid_calibration_counts_reject_before_owner_creation(self):
        with patch.object(calibration, 'Task', side_effect=AssertionError('no workers')):
            for streams, tokens in [(0, 32), (-1, 32), (True, 32), (3, 0), (3, True)]:
                with self.assertRaises(ValueError):
                    calibration.collect(SimpleNamespace(streams=streams, tokens=tokens))

    def test_every_callback_error_observed_in_request_order(self):
        for api, count in ((concurrent, 4), (calibration, 3)):
            with self.subTest(api=api.__name__):
                errors = [RuntimeError('request ' + str(i)) for i in range(count)]
                Task, owners = self.tracked(api)
                def request(*args, **kwargs):
                    at = args[5] if api is concurrent else args[-1] - 7
                    raise errors[at]
                with patch.object(api, 'Task', Task), patch.object(api, 'stream' if api is concurrent else '_send', request), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    with self.assertRaises(RuntimeError) as caught:
                        self.invoke(api, count)
                self.assertIs(caught.exception, errors[0])
                self.assertEqual([owner.error for owner in owners], errors)
                self.assertTrue(all(owner.done.is_set() for owner in owners))
                self.assertEqual(len(primary_notes := caught.exception.__notes__), count - 1)
                self.assertTrue(primary_notes)
                self.assertEqual(output.getvalue(), '')

    def test_opaque_badnotes_and_metaclass_after_all_callback_completion(self):
        for api in (concurrent, calibration):
            with self.subTest(api=api.__name__):
                Task, owners = self.tracked(api)
                cause, context = OSError(), LookupError()
                class Meta(type):
                    def __getattribute__(cls, name):
                        if name == '__name__':
                            raise AssertionError('foreign exception metaclass queried')
                        return super().__getattribute__(name)
                class Opaque(KeyboardInterrupt, metaclass=Meta):
                    def __str__(self):
                        raise AssertionError('opaque primary formatted')
                    def __getattribute__(self, name):
                        if name == '__notes__':
                            assert all(owner.done.is_set() for owner in owners)
                            BaseException.__cause__.__set__(self, None)
                            BaseException.__context__.__set__(self, None)
                            return 123
                        return super().__getattribute__(name)
                primary = Opaque()
                failures = [primary, OSError(), EOFError()]
                def request(*args, **kwargs):
                    at = args[5] if api is concurrent else args[-1] - 7
                    if not at:
                        BaseException.__cause__.__set__(primary, cause)
                        BaseException.__context__.__set__(primary, context)
                    raise failures[at]
                with patch.object(api, 'Task', Task), patch.object(api, 'stream' if api is concurrent else '_send', request), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        self.invoke(api)
                self.assertIs(caught.exception, primary)
                members = self.members(BaseException.__cause__.__get__(primary))
                for error in (cause, context, *failures[1:]):
                    self.assertIn(error, members)
                self.assertTrue(any(type(error) is TypeError for error in members))

    def test_start_return_interrupted_owner_already_published_and_callback_drained(self):
        for api in (concurrent, calibration):
            with self.subTest(api=api.__name__):
                primary, child = KeyboardInterrupt(), OSError()
                release, completed = threading.Event(), threading.Event()
                owners = []
                original = api.Task
                class Task(original):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        owners.append(self)
                    def start(self):
                        super().start()
                        threading.Timer(.02, release.set).start()
                        raise primary
                def request(*args, **kwargs):
                    release.wait(timeout=2)
                    completed.set()
                    raise child
                with patch.object(api, 'Task', Task), patch.object(api, 'stream' if api is concurrent else '_send', request), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        self.invoke(api, 1)
                self.assertIs(caught.exception, primary)
                self.assertTrue(completed.is_set())
                self.assertEqual(len(owners), 1)
                self.assertTrue(owners[0].done.is_set())
                self.assertIs(BaseException.__cause__.__get__(primary), child)

    def test_completion_wait_interruption_retains_late_child_error(self):
        primary, child = KeyboardInterrupt(), OSError()
        release, entered = threading.Event(), threading.Event()
        owners = []
        original = concurrent.Task
        class Task(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                owners.append(self)
                wait = self.done.wait
                fired = []
                def interrupted_wait(*args, **kwargs):
                    if not fired:
                        fired.append(True)
                        release.set()
                        raise primary
                    return wait(*args, **kwargs)
                self.done.wait = interrupted_wait
        def request(*args, **kwargs):
            entered.set()
            release.wait(timeout=2)
            raise child
        with patch.object(concurrent, 'Task', Task), patch.object(concurrent, 'stream', request):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.invoke(concurrent, 1)
        self.assertIs(caught.exception, primary)
        self.assertTrue(entered.is_set() and owners[0].done.is_set())
        self.assertIs(BaseException.__cause__.__get__(primary), child)

    def test_start_refusal_before_callback_runs_retains_original_error(self):
        primary = RuntimeError('no accepted start')
        Task, owners = self.tracked(concurrent, fail_start=1, primary=primary)
        with patch.object(concurrent, 'Task', Task), patch.object(concurrent, 'stream', side_effect=AssertionError('cancelled callback ran')):
            with self.assertRaises(RuntimeError) as caught:
                self.invoke(concurrent, 1)
        self.assertIs(caught.exception, primary)
        self.assertTrue(owners[0].done.is_set() and owners[0].start_cancelled)
        self.assertFalse(owners[0].entered)

    def test_real_sigint_interrupted_join_does_not_retire_unfinished_callback(self):
        primary = KeyboardInterrupt('actual SIGINT during native thread join')
        release, entered, completed = threading.Event(), threading.Event(), threading.Event()
        owners, witness, delivered, timers = [], [], [], []
        original = concurrent.Task
        previous = signal.getsignal(signal.SIGINT)
        def interrupt(number, frame):
            delivered.append(number)
            raise primary
        signal.signal(signal.SIGINT, interrupt)
        class Task(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                owners.append(self)
            def start(self):
                super().start()
                self.assert_entered = entered.wait(timeout=2)
                for delay, callback in ((.02, lambda: os.kill(os.getpid(), signal.SIGINT)), (.10, release.set)):
                    timer = threading.Timer(delay, callback)
                    timers.append(timer)
                    timer.start()
                try:
                    self.thread.join()
                except BaseException:
                    witness.append((self.thread.is_alive(), self.done.is_set(), completed.is_set()))
                    raise
        def request(*args, **kwargs):
            entered.set()
            release.wait(timeout=2)
            completed.set()
            return {'seed': 0}
        try:
            with patch.object(concurrent, 'Task', Task), patch.object(concurrent, 'stream', request):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    self.invoke(concurrent, 1)
            self.assertIs(caught.exception, primary)
            self.assertEqual(delivered, [signal.SIGINT])
            self.assertEqual(len(witness), 1)
            self.assertEqual(witness[0][1:], (False, False))
            if sys.version_info[:2] == (3, 12):
                self.assertFalse(witness[0][0])
            self.assertTrue(completed.is_set() and owners[0].done.is_set())
        finally:
            release.set()
            for timer in timers:
                timer.join()
            signal.signal(signal.SIGINT, previous)

    def test_start_failure_cancellation_retries_abort_without_unprotected_broken_property(self):
        primary, cancellation = RuntimeError('second owner start failed'), KeyboardInterrupt('abort interrupted')
        Task, owners = self.tracked(concurrent, fail_start=2, primary=primary)
        original = threading.Barrier
        calls = []
        class Barrier(original):
            @property
            def broken(self):
                raise AssertionError('unprotected cancellation predicate read')
            def abort(self):
                calls.append('abort')
                if len(calls) == 1:
                    raise cancellation
                return super().abort()
        def request(*args, gate=None, **kwargs):
            gate.wait(timeout=2)
            return {'seed': args[5]}
        with patch.object(concurrent, 'Task', Task), patch.object(concurrent, 'stream', request), \
                patch.object(concurrent.threading, 'Barrier', Barrier):
            with self.assertRaises(RuntimeError) as caught:
                self.invoke(concurrent)
        self.assertIs(caught.exception, primary)
        self.assertTrue(all(owner.done.is_set() for owner in owners))
        self.assertGreaterEqual(len(calls), 2)
        self.assertIn(cancellation, self.members(BaseException.__cause__.__get__(primary)))

    def test_callback_failure_abort_interruption_does_not_leave_peers_waiting(self):
        primary, cancellation = KeyboardInterrupt(), OSError('callback abort failed once')
        primary.__notes__ = 123
        Task, owners = self.tracked(concurrent)
        original = threading.Barrier
        calls = []
        class Barrier(original):
            def abort(self):
                calls.append('abort')
                if len(calls) == 1:
                    raise cancellation
                return super().abort()
        def request(*args, gate=None, **kwargs):
            if args[5] == 0:
                raise primary
            gate.wait(timeout=2)
            return {'seed': args[5]}
        with patch.object(concurrent, 'Task', Task), patch.object(concurrent, 'stream', request), \
                patch.object(concurrent.threading, 'Barrier', Barrier):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.invoke(concurrent)
        self.assertIs(caught.exception, primary)
        self.assertTrue(all(owner.done.is_set() for owner in owners))
        self.assertGreaterEqual(len(calls), 2)
        self.assertIn(cancellation, self.members(BaseException.__cause__.__get__(primary)))

    def test_actual_abort_return_interruption_preserves_caller_primary_and_journal(self):
        for route in ('callback', 'startup', 'result'):
            with self.subTest(route=route):
                cause, context = EOFError(), LookupError()
                primary, cancellation = KeyboardInterrupt(), OSError('abort return interrupted')
                primary.__notes__ = 123
                Task, owners = self.tracked(concurrent, fail_start=2 if route == 'startup' else None,
                                            primary=primary)
                original = threading.Barrier
                fired, calls = [], []
                class Barrier(original):
                    def abort(self):
                        calls.append('abort')
                        return super().abort()
                def profile(frame, event, arg):
                    if event == 'return' and frame.f_code is Barrier.abort.__code__ and not fired:
                        fired.append(threading.get_ident())
                        raise cancellation
                def request(*args, gate=None, **kwargs):
                    if args[5] == 0 and route != 'startup':
                        if route == 'result':
                            return {'error': 'structured request failure'}
                        BaseException.__cause__.__set__(primary, cause)
                        BaseException.__context__.__set__(primary, context)
                        raise primary
                    gate.wait(timeout=2)
                    return {'seed': args[5]}
                previous_main, previous_workers = sys.getprofile(), threading.getprofile()
                try:
                    threading.setprofile(profile)
                    if route == 'startup':
                        BaseException.__cause__.__set__(primary, cause)
                        BaseException.__context__.__set__(primary, context)
                        sys.setprofile(profile)
                    with patch.object(concurrent, 'Task', Task), patch.object(concurrent, 'stream', request), \
                            patch.object(concurrent.threading, 'Barrier', Barrier):
                        with self.assertRaises(BaseException) as caught:
                            self.invoke(concurrent)
                finally:
                    sys.setprofile(previous_main)
                    threading.setprofile(previous_workers)
                expected = cancellation if route == 'result' else primary
                self.assertIs(caught.exception, expected)
                self.assertEqual(len(fired), 1)
                self.assertGreaterEqual(len(calls), 2)
                self.assertTrue(all(owner.done.is_set() and owner._target is None for owner in owners))
                members = self.members(BaseException.__cause__.__get__(expected))
                if route != 'result':
                    for error in (cause, context, cancellation):
                        self.assertIn(error, members)


if __name__ == '__main__':
    unittest.main()
