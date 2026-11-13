"""Completed-operation cleanup transport; no SDK or native runtime."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from tensorfold import cleanup


class CleanupTransportTests(unittest.TestCase):
    def test_nullable_slots_select_only_real_errors_without_changing_sequence(self):
        self.assertIsNone(cleanup.raise_failures(None, [None] * 5))
        first, second = OSError('first'), ValueError('second')
        statuses = [None, first, None, second, first, None]
        with self.assertRaises(OSError) as caught:
            cleanup.raise_failures(None, statuses)
        self.assertIs(caught.exception, first)
        self.assertIs(first.__cause__, second)
        self.assertNotIn('NoneType', ' '.join(first.__notes__))
        self.assertEqual(statuses, [None, first, None, second, first, None])

    def test_primary_with_only_nullable_slots_retains_original_native_cause(self):
        primary, cause = KeyboardInterrupt(), OSError()
        primary.__cause__ = cause
        with self.assertRaises(KeyboardInterrupt) as caught:
            cleanup.raise_failures(primary, (None, None))
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cause)
        self.assertFalse(hasattr(primary, '__notes__'))

    def test_falsey_exception_remains_a_failure(self):
        class Falsey(OSError):
            def __bool__(self):
                raise AssertionError('status truthiness must not be consulted')
        failure = Falsey()
        with self.assertRaises(Falsey) as caught:
            cleanup.raise_failures(None, [None, failure, None])
        self.assertIs(caught.exception, failure)

    def test_nullable_group_failure_preserves_primary_and_all_real_statuses(self):
        first, second, third = KeyboardInterrupt(), OSError(), ValueError()
        statuses, allocation = [None, first, None, second, third, None], MemoryError()
        with (
            patch.object(cleanup, 'BaseExceptionGroup', side_effect=allocation, create=True),
            self.assertRaises(KeyboardInterrupt) as caught,
        ):
            cleanup.raise_failures(None, statuses)
        self.assertIs(caught.exception, first)
        self.assertIs(first.__cause__, allocation)
        self.assertEqual(first.__dict__['_tensorfold_retained_failures'], [second, third])

    def test_primary_and_distinct_native_roots_and_statuses_survive(self):
        primary, cause, context = KeyboardInterrupt(), ValueError(), EOFError()
        first, second = OSError(), RuntimeError()
        primary.__cause__, primary.__context__ = cause, context
        with self.assertRaises(KeyboardInterrupt) as caught:
            cleanup.raise_failures(primary, [first, primary, second, first])
        self.assertIs(caught.exception, primary)
        self.assertEqual(BaseException.__cause__.__get__(primary).exceptions, (cause, context, first, second))

    def test_opaque_formatting_and_bad_notes_cannot_replace_primary(self):
        class Opaque(KeyboardInterrupt):
            def __str__(self):
                raise RuntimeError("formatting hook forbidden")

            def add_note(self, note):
                raise RuntimeError("annotation hook forbidden")

        primary, close = Opaque(), OSError()
        primary.__notes__ = 7
        with self.assertRaises(Opaque) as caught:
            cleanup.raise_failures(primary, [close])
        self.assertIs(caught.exception, primary)
        causes = BaseException.__cause__.__get__(primary).exceptions
        self.assertIs(causes[0], close)
        self.assertIsInstance(causes[1], TypeError)

    def test_group_allocation_failure_retains_original_formed_statuses(self):
        primary, previous, first, second = KeyboardInterrupt(), ValueError(), OSError(), EOFError()
        primary.__cause__ = previous
        allocation = MemoryError()
        with patch.object(cleanup, "BaseExceptionGroup", side_effect=allocation, create=True):
            with self.assertRaises(KeyboardInterrupt) as caught:
                cleanup.raise_failures(primary, [first, second])
        self.assertIs(caught.exception, primary)
        self.assertIs(BaseException.__cause__.__get__(primary), allocation)
        retained = BaseException.__dict__["__dict__"].__get__(primary)["_tensorfold_retained_failures"]
        self.assertEqual(retained, [previous, first, second])

    def test_same_identity_and_successful_empty_cleanup(self):
        self.assertIsNone(cleanup.raise_failures(None, []))
        primary = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt) as caught:
            cleanup.raise_failures(primary, [primary])
        self.assertIs(caught.exception, primary)
        self.assertIsNone(BaseException.__cause__.__get__(primary))

    def test_nested_constructor_owners_retained_and_normal_rollback_restores_roots(self):
        primary, first, second = KeyboardInterrupt(), object(), object()
        failures = [OSError(), RuntimeError()]
        for owner, failure in zip((first, second), failures):
            with self.assertRaises(KeyboardInterrupt) as caught:
                cleanup.rollback(owner, primary, lambda error=failure: (_ for _ in ()).throw(error))
            self.assertIs(caught.exception, primary)
        self.assertEqual(primary.__dict__["_tensorfold_retained_owners"], [first, second])
        cause, context = ValueError(), EOFError()
        primary = KeyboardInterrupt()
        primary.__cause__, primary.__context__, primary.__suppress_context__ = cause, context, False
        def close():
            primary.__cause__ = primary.__context__ = None
            primary.__suppress_context__ = True
        with self.assertRaises(KeyboardInterrupt):
            cleanup.rollback(first, primary, close)
        self.assertIs(primary.__cause__, cause)
        self.assertIs(primary.__context__, context)
        self.assertFalse(primary.__suppress_context__)

    def test_finish_attempts_all_and_restores_prior_roots_before_transport(self):
        primary, cause, context, close = KeyboardInterrupt(), ValueError(), EOFError(), OSError()
        primary.__cause__, primary.__context__ = cause, context
        calls = []
        def first():
            calls.append("first")
            raise primary
        def second():
            calls.append("second")
            primary.__cause__ = primary.__context__ = None
            raise close
        def third():
            calls.append("third")
        with self.assertRaises(KeyboardInterrupt) as caught:
            cleanup.finish([first, second, third])
        self.assertIs(caught.exception, primary)
        self.assertEqual(calls, ["first", "second", "third"])
        self.assertEqual(primary.__cause__.exceptions, (cause, context, close))


if __name__ == "__main__":
    unittest.main()
