"""Decision cancellation diagnostics preserve primary and every owned status."""
import unittest
from unittest.mock import patch

import test_server_job_lifetime as fixture


def reachable(root, wanted):
    pending, seen = [root], set()
    while pending:
        error = pending.pop()
        if error is wanted:
            return True
        if id(error) in seen:
            continue
        seen.add(id(error))
        for slot in (BaseException.__cause__, BaseException.__context__):
            value = slot.__get__(error)
            if value is not None:
                pending.append(value)
        if isinstance(error, BaseExceptionGroup):
            pending.extend(BaseExceptionGroup.exceptions.__get__(error))
    return False


class DecisionCancelErrorTests(unittest.TestCase):
    def app(self, primary):
        def submit(job, index):
            if index == 3:
                raise primary
        return fixture.DecisionsTests().app(submit)

    def test_bad_notes_opaque_fields_all_cancels_and_prior_roots_retained(self):
        class Opaque(BaseException):
            def __str__(self):
                raise AssertionError("foreign formatting invoked")
            def add_note(self, note):
                raise AssertionError("foreign note hook invoked")
        class Meta(type):
            def __getattribute__(cls, name):
                if name == "__name__":
                    raise AssertionError("foreign class-name hook invoked")
                return super().__getattribute__(name)
        class Cleanup(Exception, metaclass=Meta):
            pass
        primary = Opaque()
        oldcause, oldcontext = RuntimeError("prior cause"), LookupError("prior context")
        primary.__cause__, primary.__context__, primary.__notes__ = oldcause, oldcontext, 7
        owner, body, jobs, _ = self.app(primary)
        failures = [Cleanup(), OSError("second cancel"), RuntimeError("third cancel")]
        attempted = []
        def cancel(value):
            attempted.append(value)
            primary.__cause__, primary.__context__ = None, None
            raise failures[len(attempted) - 1]
        owner.scheduler.cancel = cancel
        with self.assertRaises(Opaque) as caught:
            owner.decisions(body)
        self.assertIs(caught.exception, primary)
        self.assertEqual(attempted, [job.cancellation for job in jobs])
        for failure in (oldcause, oldcontext, *failures):
            self.assertTrue(reachable(primary, failure))
        self.assertTrue(any(isinstance(error, TypeError) for error in primary.__cause__.exceptions))

    def test_successful_cancellation_restores_native_suppression_and_prior_context(self):
        for suppressed in (False, True):
            primary = RuntimeError("submit failed")
            context = LookupError("original context")
            primary.__context__, primary.__suppress_context__ = context, suppressed
            owner, body, jobs, _ = self.app(primary)
            attempted = []
            def cancel(value):
                attempted.append(value)
                primary.__context__, primary.__suppress_context__ = None, not suppressed
            owner.scheduler.cancel = cancel
            with self.assertRaises(RuntimeError) as caught:
                owner.decisions(body)
            self.assertIs(caught.exception, primary)
            self.assertEqual(attempted, [job.cancellation for job in jobs])
            self.assertIs(primary.__context__, context)
            self.assertIs(primary.__suppress_context__, suppressed)

    def test_same_exception_cancellation_never_creates_self_cause(self):
        primary = RuntimeError("same submit and cancel status")
        owner, body, jobs, _ = self.app(primary)
        attempted = []
        def cancel(value):
            attempted.append(value)
            raise primary
        owner.scheduler.cancel = cancel
        with self.assertRaises(RuntimeError) as caught:
            owner.decisions(body)
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(attempted), len(jobs))
        self.assertIsNot(primary.__cause__, primary)

    def test_group_allocation_failure_keeps_primary_and_retained_statuses(self):
        primary = RuntimeError("submit failed")
        owner, body, jobs, _ = self.app(primary)
        failures = [OSError("cancel one"), LookupError("cancel two"), ValueError("cancel three")]
        attempted = []
        def cancel(value):
            attempted.append(value)
            raise failures[len(attempted) - 1]
        owner.scheduler.cancel = cancel
        # The source fixture deliberately does not register its loaded module;
        # patch the actual method globals instead of creating a duplicate class.
        allocation = MemoryError("labeled group allocation failure")
        with patch.dict(owner.decisions.__func__.__globals__, {"BaseExceptionGroup": lambda *args: (_ for _ in ()).throw(allocation)}):
            with self.assertRaises(RuntimeError) as caught:
                owner.decisions(body)
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, allocation)
        self.assertEqual(len(attempted), len(jobs))
        self.assertEqual(primary._tensorfold_decision_failures, failures)
