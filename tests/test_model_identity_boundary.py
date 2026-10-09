"""Exact-byte identity, bounded JSON, actual filesystem and FD ownership."""
from __future__ import annotations

import dataclasses
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from snapshot_fd_transport import TransportOwner
from tensorfold.engine.model_identity import canonical_runtime_json, capture_model_identity


class JsonControls(unittest.TestCase):
    def test_exact_canonical_json_and_utf8_escaped_byte_budget(self):
        value = {'é': ['\0\b\t\n\f\r"\\', '😀', -0.0, 1, True, False, None], 'z': {}}
        expected = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        self.assertEqual(canonical_runtime_json(value, max_bytes=len(expected.encode())), expected)
        with self.assertRaises(ValueError):
            canonical_runtime_json(value, max_bytes=len(expected.encode()) - 1)

    def test_nonfinite_nonjson_unicode_cycles_depth_and_scheduled_nodes_refuse(self):
        cycle = []
        cycle.append(cycle)
        for value in ({'x': float('inf')}, {'x': float('nan')}, {'x': (1,)}, {1: 'x'},
                      {'x': object()}, {'x': '\ud800'}, {'x': cycle}):
            with self.assertRaises(ValueError):
                canonical_runtime_json(value)
        with self.assertRaises(ValueError):
            canonical_runtime_json({'x': [[1]]}, max_depth=2)
        with self.assertRaises(ValueError):
            canonical_runtime_json({'x': [[1] * 16, [1] * 16]}, max_nodes=20)
        shared = [1]
        self.assertEqual(canonical_runtime_json({'a': shared, 'b': shared}), '{"a":[1],"b":[1]}')


class IdentityControls(unittest.TestCase):
    def setUp(self):
        # Explicit Python transport substitution; native acquisition/retirement
        # atomicity is qualified separately by ROOT with the actual extension.
        substitution = patch('tensorfold.file_io._owned_slot', TransportOwner)
        substitution.start()
        self.addCleanup(substitution.stop)

    def fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix='model-identity-control-')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        files = {'weights': root / 'weights', 'tokenizer': root / 'tokenizer'}
        files['weights'].write_bytes(b'weights\0' * 37)
        files['tokenizer'].write_bytes(b'tokens')
        return root, files

    def capture(self, files, **kwargs):
        return capture_model_identity(files, runtime_identity=kwargs.pop('runtime_identity', {'loader': 'v1'}),
            authorize=kwargs.pop('authorize', lambda name, path: path.resolve(strict=True)),
            max_file_bytes=kwargs.pop('max_file_bytes', 1 << 20),
            max_total_bytes=kwargs.pop('max_total_bytes', 1 << 20), **kwargs)

    def test_full_hash_and_independent_envelope_identity(self):
        root, files = self.fixture()
        receipt = self.capture(files)
        entries = [{'name': name, 'size': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in sorted(files.items())]
        envelope = {'format': 'model-content-v1', 'runtime': '{"loader":"v1"}', 'files': entries}
        expected = hashlib.sha256(json.dumps(envelope, sort_keys=True, ensure_ascii=False,
                                             separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(receipt.model_prefix, 'model-content-v1:' + expected)
        self.assertEqual(receipt.bytes_read, sum(path.stat().st_size for path in files.values()))
        self.assertEqual(receipt.hashes_reused, 0)
        receipt.verify_unchanged()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            receipt.total_bytes = 0

    def test_empty_and_multichunk_inputs_keep_exact_incremental_read_contract(self):
        root, _ = self.fixture()
        empty, large = root / 'empty', root / 'large'
        empty.write_bytes(b'')
        raw = bytes(range(251)) * 5000
        large.write_bytes(raw)
        real_read, reads = os.read, []
        def read(fd, count):
            value = real_read(fd, count)
            reads.append((count, len(value)))
            return value
        with patch('tensorfold.engine.model_identity.os.read', side_effect=read):
            receipt = self.capture({'empty': empty, 'large': large},
                                   max_file_bytes=len(raw), max_total_bytes=len(raw))
        self.assertEqual(receipt.bytes_read, len(raw))
        self.assertEqual(reads, [(1, 0), (1 << 20, 1 << 20), (len(raw) - (1 << 20), len(raw) - (1 << 20)), (1, 0)])
        self.assertEqual([item.sha256 for item in receipt.files],
                         [hashlib.sha256(b'').hexdigest(), hashlib.sha256(raw).hexdigest()])
        zero = self.capture({'empty': empty}, max_file_bytes=0, max_total_bytes=0)
        self.assertEqual((zero.total_bytes, zero.bytes_read), (0, 0))

    def test_same_content_relocation_and_dict_order_preserve_identity(self):
        root, files = self.fixture()
        first = self.capture(files, runtime_identity={'b': 2, 'a': 1})
        relocated = root / 'relocated'
        relocated.mkdir()
        copied = {}
        for name, path in reversed(list(files.items())):
            copied[name] = relocated / path.name
            copied[name].write_bytes(path.read_bytes())
        second = self.capture(copied, runtime_identity={'a': 1, 'b': 2})
        self.assertEqual(first.model_prefix, second.model_prefix)

    def test_each_input_name_bytes_runtime_and_closure_change_identity(self):
        _, files = self.fixture()
        first = self.capture(files)
        self.assertNotEqual(first.model_prefix, self.capture(files, runtime_identity={'loader': 'v2'}).model_prefix)
        self.assertNotEqual(first.model_prefix, self.capture({'renamed': files['weights'], 'tokenizer': files['tokenizer']}).model_prefix)
        self.assertNotEqual(first.model_prefix, self.capture({'weights': files['weights']}).model_prefix)
        files['tokenizer'].write_bytes(b'TOKENS')
        self.assertNotEqual(first.model_prefix, self.capture(files).model_prefix)
        with self.assertRaises(ValueError):
            first.verify_unchanged()

    def test_process_owned_immutable_reuse_reads_zero_bytes(self):
        _, files = self.fixture()
        first = self.capture(files)
        with patch('tensorfold.engine.model_identity.os.read', side_effect=AssertionError('reused input read')):
            second = self.capture(files, reuse=first, reuse_is_immutable=True)
        self.assertEqual(first.model_prefix, second.model_prefix)
        self.assertEqual(second.bytes_read, 0)
        self.assertEqual(second.hashes_reused, 2)
        for kwargs in ({'reuse': first}, {'reuse': {'files': []}, 'reuse_is_immutable': True},
                       {'reuse': copy.copy(first), 'reuse_is_immutable': True}):
            with self.assertRaises(ValueError):
                self.capture(files, **kwargs)
        for changes in ({'_issued': object()}, {'model_prefix': 'forged'},
                        {'files': (dataclasses.replace(first.files[0], sha256='0' * 64),)}):
            with self.assertRaises(ValueError):
                dataclasses.replace(first, **changes)

    def test_changed_input_or_runtime_cannot_reuse_old_hash(self):
        _, files = self.fixture()
        first = self.capture(files)
        files['weights'].write_bytes(b'changed')
        second = self.capture(files, reuse=first, reuse_is_immutable=True)
        self.assertEqual(second.hashes_reused, 1)
        self.assertEqual(second.bytes_read, 7)
        self.assertNotEqual(first.model_prefix, second.model_prefix)
        third = self.capture(files, reuse=second, reuse_is_immutable=True, runtime_identity={'loader': 'v2'})
        self.assertEqual(third.hashes_reused, 0)

    def test_byte_name_count_authorization_and_nonregular_bounds(self):
        root, files = self.fixture()
        for kwargs in ({'max_file_bytes': 1}, {'max_total_bytes': 1}, {'max_files': 1},
                       {'max_name_bytes': 1}, {'max_file_bytes': False}, {'max_total_bytes': -1},
                       {'authorize': lambda name, path: Path('relative')}):
            with self.assertRaises(ValueError):
                self.capture(files, **kwargs)
        fifo = root / 'fifo'
        os.mkfifo(fifo)
        for path in (fifo, root):
            with self.assertRaises(ValueError):
                self.capture({'input': path})
        link = root / 'unresolved-link'
        link.symlink_to(files['weights'])
        with self.assertRaises(OSError):
            self.capture({'input': link}, authorize=lambda name, path: path)

    def test_hash_mutation_growth_truncation_and_path_replacement_refuse(self):
        for mode in ('mutate', 'grow', 'truncate', 'replace'):
            root, files = self.fixture()
            selected = files['tokenizer']
            real_read = os.read
            fired = []
            def read(fd, count):
                if not fired:
                    fired.append(True)
                    if mode == 'replace':
                        replacement = root / 'replacement'
                        replacement.write_bytes(b'tokens')
                        replacement.replace(selected)
                    else:
                        selected.write_bytes({'mutate': b'TOKENS', 'grow': b'tokens!', 'truncate': b'x'}[mode])
                return real_read(fd, count)
            with patch('tensorfold.engine.model_identity.os.read', side_effect=read):
                with self.assertRaises(ValueError):
                    self.capture(files)

    def test_real_close_secondary_preserves_opaque_hash_failure(self):
        _, files = self.fixture()
        class Opaque(KeyboardInterrupt):
            def __str__(self):
                raise AssertionError('foreign primary formatting')
            def add_note(self, text):
                raise AssertionError('foreign primary method')
        primary, calls = Opaque(), []
        real_close = os.close
        def close(fd):
            real_close(fd)
            calls.append(fd)
            raise OSError('real descriptor closed, then reported failure')
        with patch('tensorfold.engine.model_identity.os.read', side_effect=primary), \
                patch('tensorfold.engine.model_identity.os.close', side_effect=close):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.capture(files)
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(calls), 1)
        self.assertTrue(primary.__notes__)

    def test_owner_is_retained_before_first_fstat_and_hash_work(self):
        from tensorfold.engine import model_identity
        _, files = self.fixture()
        owners, closed = [], []
        primary = KeyboardInterrupt('first fstat interrupted')
        class Observed(TransportOwner):
            def __init__(self, *args):
                super().__init__(*args)
                owners.append(self)
            def close(self):
                closed.append(self)
                super().close()
        original_close = model_identity._close
        def close(scope, failure):
            self.assertIs(scope._records[0].owner, owners[0])
            self.assertIs(failure, primary)
            return original_close(scope, failure)
        with patch('tensorfold.file_io._owned_slot', Observed), \
                patch('tensorfold.file_io.os.fstat', side_effect=primary), \
                patch.object(model_identity, '_close', side_effect=close), \
                patch('tensorfold.engine.model_identity.os.read', side_effect=AssertionError('hash started')):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.capture(files)
        self.assertIs(caught.exception, primary)
        self.assertEqual(closed, owners)
        self.assertEqual(len(owners), 1)
        self.assertTrue(owners[0].closed)

    def test_before_entry_owner_close_interrupt_retries_owner_and_fails_operation(self):
        _, files = self.fixture()
        primary, owners = KeyboardInterrupt('before close entry'), []
        class Interrupted(TransportOwner):
            def __init__(self, *args):
                super().__init__(*args)
                self.calls = 0
                owners.append(self)
            def close(self):
                self.calls += 1
                if self.calls == 1:
                    raise primary
                super().close()
        with patch('tensorfold.file_io._owned_slot', Interrupted):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.capture(files)
        self.assertIs(caught.exception, primary)
        self.assertEqual(owners[0].calls, 2)
        self.assertTrue(owners[0].closed)

    def test_unretired_owner_journal_is_bounded_and_preserves_all_failure_context(self):
        from tensorfold.engine import model_identity
        _, files = self.fixture()
        primary = KeyboardInterrupt('hash read interrupted')
        previous = RuntimeError('preexisting failure context')
        first, second = KeyboardInterrupt('first close entry'), KeyboardInterrupt('second close entry')
        primary.__cause__, primary.__notes__ = previous, 123
        owners = []
        class Interrupted(TransportOwner):
            def __init__(self, *args):
                super().__init__(*args)
                self.calls = 0
                owners.append(self)
            def close(self):
                self.calls += 1
                if self.calls <= 2:
                    raise (first, second)[self.calls - 1]
                super().close()
        with patch('tensorfold.file_io._owned_slot', Interrupted), \
                patch('tensorfold.engine.model_identity.os.read', side_effect=primary):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.capture(files)
        self.assertIs(caught.exception, primary)
        self.assertEqual(primary.__notes__, 123)
        self.assertEqual(primary.__cause__.exceptions[:3], (previous, first, second))
        self.assertIsInstance(primary.__cause__.exceptions[3], TypeError)
        scope = primary._tensorfold_model_fd_scope
        self.assertEqual(scope.maximum, 1)
        self.assertEqual(len(scope._records), 1)
        self.assertEqual(primary._tensorfold_model_fd_owners, (owners[0],))
        self.assertFalse(owners[0].closed)
        self.assertEqual(scope.drain(), [])
        model_identity._close(scope, primary)
        self.assertTrue(owners[0].closed)
        self.assertNotIn('_tensorfold_model_fd_scope', primary.__dict__)
        self.assertNotIn('_tensorfold_model_fd_owners', primary.__dict__)

    def test_successful_cleanup_preserves_suppressed_native_context_without_explicit_cause(self):
        from tensorfold.engine import model_identity
        primary = json.JSONDecodeError('invalid metadata', '{', 1)
        context = StopIteration('parser context')
        primary.__cause__, primary.__context__, primary.__suppress_context__ = None, context, True
        class Scope:
            live_owners = ()
            def drain(self):
                return []
        model_identity._close(Scope(), primary)
        self.assertIsNone(primary.__cause__)
        self.assertIs(primary.__context__, context)
        self.assertIs(primary.__suppress_context__, True)
        self.assertNotIn('_tensorfold_model_fd_failures', primary.__dict__)

    def test_actual_successful_close_C_RETURN_callback_restores_all_native_transport_fields(self):
        from tensorfold import file_io
        from tensorfold.engine import model_identity
        _, files = self.fixture()
        for original_suppress in (False, True):
            primary = ValueError('original operation')
            cause, context = RuntimeError('original cause'), StopIteration('original parser context')
            primary.__cause__, primary.__context__, primary.__suppress_context__ = cause, context, original_suppress
            scope = file_io.FileStreams(max_files=1)
            record, _ = scope.open_descriptor(files['tokenizer'], os.O_RDONLY)
            previous_profile, fired = sys.getprofile(), []
            def profile(frame, event, function):
                if event == 'c_return' and function is os.close:
                    fired.append(True)
                    primary.__cause__ = None
                    primary.__context__ = None
                    primary.__suppress_context__ = not original_suppress
            try:
                sys.setprofile(profile)
                model_identity._close(scope, primary)
            finally:
                sys.setprofile(previous_profile)
                self.assertEqual(scope.drain(), [])
            self.assertEqual(fired, [True])
            self.assertTrue(record.done)
            self.assertTrue(record.owner.closed)
            self.assertIs(primary.__cause__, cause)
            self.assertIs(primary.__context__, context)
            self.assertIs(primary.__suppress_context__, original_suppress)

    def test_cleanup_captures_original_cause_and_context_before_foreign_callbacks(self):
        from tensorfold.engine import model_identity
        primary = KeyboardInterrupt('original operation')
        cause, context = RuntimeError('original cause'), LookupError('original context')
        replacement, cleanup = OSError('callback changed roots'), ValueError('cleanup status')
        primary.__cause__, primary.__context__ = cause, context
        class Scope:
            live_owners = ()
            def drain(self):
                primary.__cause__ = replacement
                primary.__context__ = replacement
                return [cleanup]
        with self.assertRaises(KeyboardInterrupt) as caught:
            model_identity._close(Scope(), primary)
        self.assertIs(caught.exception, primary)
        self.assertEqual(primary.__cause__.exceptions, (cause, context, replacement, cleanup))

    def test_cleanup_group_allocation_failure_keeps_same_primary_and_all_formed_statuses(self):
        from tensorfold.engine import model_identity
        primary = KeyboardInterrupt('original operation')
        cause, context = RuntimeError('original cause'), LookupError('original context')
        cleanup, allocation = OSError('cleanup status'), MemoryError('group allocation refused')
        primary.__cause__, primary.__context__ = cause, context
        class Scope:
            live_owners = ()
            def drain(self):
                return [cleanup]
        with patch.object(model_identity, 'BaseExceptionGroup', side_effect=allocation, create=True):
            with self.assertRaises(KeyboardInterrupt) as caught:
                model_identity._close(Scope(), primary)
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, allocation)
        self.assertEqual(primary._tensorfold_model_fd_failures, [cause, context, cleanup])

    def test_consumed_close_error_never_closes_reused_descriptor_and_keeps_primary_cause(self):
        _, files = self.fixture()
        primary, secondary = KeyboardInterrupt('hash failed'), OSError('consumed EIO')
        previous = RuntimeError('original cause')
        primary.__cause__ = previous
        owners, replacements = [], []
        class Consumed(TransportOwner):
            def __init__(self, *args):
                super().__init__(*args)
                self.calls = 0
                owners.append(self)
            def open(self, path, flags, mode=0o600):
                self.path = path
                return super().open(path, flags, mode)
            def close(self):
                self.calls += 1
                descriptor = self.fileno()
                super().close()
                replacement = TransportOwner()
                replacement.open(self.path, os.O_RDONLY)
                replacements.append(replacement)
                self.assertion_descriptor = descriptor
                raise secondary
        try:
            with patch('tensorfold.file_io._owned_slot', Consumed), \
                    patch('tensorfold.engine.model_identity.os.read', side_effect=primary):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    self.capture(files)
            self.assertIs(caught.exception, primary)
            self.assertEqual(primary.__cause__.exceptions, (previous, secondary))
            self.assertEqual(owners[0].calls, 1)
            self.assertTrue(owners[0].closed)
            self.assertEqual(replacements[0].fileno(), owners[0].assertion_descriptor)
            self.assertEqual(os.read(replacements[0].fileno(), 6), b'tokens')
        finally:
            for owner in replacements:
                owner.close()

    def test_same_primary_cleanup_object_and_prior_selfcause_do_not_create_selfcause(self):
        _, files = self.fixture()
        primary = KeyboardInterrupt('opaque read and cleanup failure')
        primary.__cause__ = primary
        class Consumed(TransportOwner):
            def close(self):
                super().close()
                raise primary
        with patch('tensorfold.file_io._owned_slot', Consumed), \
                patch('tensorfold.engine.model_identity.os.read', side_effect=primary):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.capture(files)
        self.assertIs(caught.exception, primary)
        self.assertIsNone(primary.__cause__)

    def test_acquisition_failure_is_contained_before_fstat_or_hash(self):
        from tensorfold.engine import model_identity
        _, files = self.fixture()
        primary = MemoryError('owner acquisition interrupted')
        original_close, journals = model_identity._close, []
        def close(scope, error):
            journals.append(scope)
            self.assertIs(error, primary)
            self.assertIsNone(scope._records[0].owner)
            return original_close(scope, error)
        with patch('tensorfold.file_io._owned_slot', side_effect=primary), \
                patch('tensorfold.file_io.os.fstat', side_effect=AssertionError('unowned fstat')), \
                patch.object(model_identity, '_close', side_effect=close), \
                patch('tensorfold.engine.model_identity.os.read', side_effect=AssertionError('unowned read')):
            with self.assertRaises(MemoryError) as caught:
                self.capture(files)
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(journals), 1)
        self.assertTrue(journals[0].retired)

    def test_closed_slot_is_published_before_open_and_after_acquisition_interrupt_cleanup(self):
        from tensorfold import file_io
        _, files = self.fixture()
        primary, records, owners = KeyboardInterrupt('after owner open acquisition'), [], []
        OriginalRecord = file_io._Record
        class Record(OriginalRecord):
            def __init__(self):
                super().__init__()
                records.append(self)
        class OpenInterrupted(TransportOwner):
            def __init__(self):
                super().__init__()
                owners.append(self)
            def open(self, *args, **kwargs):
                self_test.assertIs(records[0].owner, self)
                self_test.assertTrue(self.closed)
                super().open(*args, **kwargs)
                raise primary
        self_test = self
        with patch.object(file_io, '_Record', Record), patch.object(file_io, '_owned_slot', OpenInterrupted), \
                patch('tensorfold.file_io.os.fstat', side_effect=AssertionError('interrupted acquisition reached fstat')), \
                patch('tensorfold.engine.model_identity.os.read', side_effect=AssertionError('interrupted acquisition reached hash')):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.capture(files)
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(owners), 1)
        self.assertTrue(owners[0].closed)
        self.assertTrue(records[0].done)

    def test_published_closed_slot_open_refusal_keeps_no_descriptor(self):
        from tensorfold import file_io
        _, files = self.fixture()
        primary, owners = OSError('ordinary open refused'), []
        class Refused(TransportOwner):
            def __init__(self):
                super().__init__()
                owners.append(self)
            def open(self, *args, **kwargs):
                raise primary
        with patch.object(file_io, '_owned_slot', Refused), \
                patch('tensorfold.file_io.os.fstat', side_effect=AssertionError('failed open reached fstat')):
            with self.assertRaises(OSError) as caught:
                self.capture(files)
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(owners), 1)
        self.assertTrue(owners[0].closed)

    def test_issued_receipt_verification_reports_consumed_close_failure(self):
        _, files = self.fixture()
        receipt = self.capture(files)
        secondary, owners = OSError('verification close failed'), []
        class Consumed(TransportOwner):
            def __init__(self, *args):
                super().__init__(*args)
                owners.append(self)
            def close(self):
                super().close()
                raise secondary
        with patch('tensorfold.file_io._owned_slot', Consumed):
            with self.assertRaises(OSError) as caught:
                receipt.verify_unchanged()
        self.assertIs(caught.exception, secondary)
        self.assertEqual(len(owners), 1)
        self.assertTrue(owners[0].closed)


if __name__ == '__main__':
    unittest.main()
