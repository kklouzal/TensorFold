"""Checkpoint stream error ownership with labeled stdlib FD transport only."""

import errno
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from snapshot_fd_transport import TransportOwner
from tensorfold import file_io
from tensorfold.cuda import tensor_file


class StreamControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='checkpoint-stream-control-')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'header.safetensors'
        header = json.dumps({'data': {'dtype': 'U8', 'shape': [4], 'data_offsets': [0, 4]}}).encode()
        self.path.write_bytes(struct.pack('<Q', len(header)) + header + b'data')
        self.owners = []
        controls = self

        class Owner(TransportOwner):
            def __init__(self):
                super().__init__()
                controls.owners.append(self)
                controls.addCleanup(self.close)
            def open(self, *args, **kwargs):
                super().open(*args, **kwargs)
                self.original_fd = self.fileno()

        self.owner = Owner
        factory = patch.object(file_io, '_owned_slot', Owner)
        factory.start()
        self.addCleanup(factory.stop)

    def assert_retired(self, owner):
        self.assertTrue(owner.closed)
        with self.assertRaises(OSError) as caught:
            os.fstat(owner.original_fd)
        self.assertEqual(caught.exception.errno, errno.EBADF)

    @staticmethod
    def roots(primary):
        return (BaseException.__cause__.__get__(primary), BaseException.__context__.__get__(primary),
                BaseException.__suppress_context__.__get__(primary))

    def test_actual_header_reader_success_consumes_owner(self):
        start, header = tensor_file.read_header(self.path, {'U8': 1})
        self.assertGreater(start, 8)
        self.assertEqual(header['data']['data_offsets'], [0, 4])
        self.assertEqual(len(self.owners), 1)
        self.assert_retired(self.owners[0])

    def test_successful_actual_close_callback_restores_all_original_native_fields(self):
        for suppress in (False, True):
            primary = KeyboardInterrupt('consumer interrupted')
            cause, context = ValueError('cause'), OSError('context')
            BaseException.__cause__.__set__(primary, cause)
            BaseException.__context__.__set__(primary, context)
            BaseException.__suppress_context__.__set__(primary, suppress)
            fired = []
            def profile(frame, event, function):
                if event == 'c_return' and function is os.close:
                    fired.append(True)
                    BaseException.__cause__.__set__(primary, None)
                    BaseException.__context__.__set__(primary, None)
                    BaseException.__suppress_context__.__set__(primary, not suppress)
            previous = sys.getprofile()
            try:
                sys.setprofile(profile)
                with self.assertRaises(KeyboardInterrupt) as caught:
                    with tensor_file._regular_stream(self.path):
                        raise primary
            finally:
                sys.setprofile(previous)
            self.assertIs(caught.exception, primary)
            self.assertEqual(fired, [True])
            self.assertEqual(self.roots(primary), (cause, context, suppress))
            self.assert_retired(self.owners[-1])

    def test_plain_malformed_json_retains_ordinary_suppressed_parser_context(self):
        raw = b'{"bad":'
        self.path.write_bytes(struct.pack('<Q', len(raw)) + raw)
        with self.assertRaises(json.JSONDecodeError) as caught:
            tensor_file.read_header(self.path, {'U8': 1})
        cause, context, suppress = self.roots(caught.exception)
        self.assertIsNone(cause)
        self.assertIsInstance(context, StopIteration)
        self.assertIs(suppress, True)
        self.assert_retired(self.owners[0])

    def test_consumed_close_error_preserves_original_roots_before_callback_and_bad_notes(self):
        primary = KeyboardInterrupt('consumer interrupted')
        cause, context, cleanup = ValueError('prior cause'), LookupError('prior context'), OSError('consumed close')
        BaseException.__cause__.__set__(primary, cause)
        BaseException.__context__.__set__(primary, context)
        primary.__notes__ = 123
        original = self.owner
        class Broken(original):
            def close(self):
                if not self.closed:
                    super().close()
                    BaseException.__cause__.__set__(primary, None)
                    BaseException.__context__.__set__(primary, None)
                    raise cleanup
        with patch.object(file_io, '_owned_slot', Broken):
            with self.assertRaises(KeyboardInterrupt) as caught:
                with tensor_file._regular_stream(self.path):
                    raise primary
        self.assertIs(caught.exception, primary)
        members = BaseException.__cause__.__get__(primary).exceptions
        self.assertIs(members[0], cause)
        self.assertIs(members[1], context)
        self.assertIn(cleanup, members)
        self.assertTrue(any(type(error) is TypeError for error in members))
        self.assert_retired(self.owners[0])

    def test_cleanup_only_raises_exact_first_consumed_close_error(self):
        cleanup = OSError('consumed close failure')
        original = self.owner
        class Broken(original):
            def close(self):
                if not self.closed:
                    super().close()
                    raise cleanup
        with patch.object(file_io, '_owned_slot', Broken):
            with self.assertRaises(OSError) as caught:
                tensor_file.read_header(self.path, {'U8': 1})
        self.assertIs(caught.exception, cleanup)
        self.assert_retired(self.owners[0])

    def test_unfinished_owner_scope_is_retained_on_exact_primary_until_explicit_drain(self):
        primary, cleanup = KeyboardInterrupt('consumer failed'), OSError('pre-entry close refused')
        refused = [True]
        calls = []
        original = self.owner
        class Broken(original):
            def close(self):
                if refused[0] and not self.closed:
                    calls.append(True)
                    raise cleanup
                super().close()
        try:
            with patch.object(file_io, '_owned_slot', Broken):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    with tensor_file._regular_stream(self.path):
                        raise primary
            self.assertIs(caught.exception, primary)
            self.assertEqual(len(calls), 2)
            self.assertFalse(self.owners[0].closed)
            namespace = BaseException.__dict__['__dict__'].__get__(primary)
            scope = dict.__getitem__(namespace, '_tensorfold_checkpoint_file_scope')
            self.assertIs(scope.live_owners[0], self.owners[0])
            self.assertEqual(BaseException.__cause__.__get__(primary).exceptions, (cleanup,))
            refused[0] = False
            self.assertEqual(scope.drain(), [])
            self.assertTrue(scope.retired)
            self.assert_retired(self.owners[0])
        finally:
            refused[0] = False

    def test_group_allocation_refusal_preserves_exact_primary_and_every_formed_status(self):
        primary = KeyboardInterrupt('consumer failed')
        cause, context, cleanup = ValueError('prior cause'), LookupError('prior context'), OSError('consumed close')
        primary.__cause__, primary.__context__ = cause, context
        allocation = MemoryError('group refused')
        original = self.owner
        class Broken(original):
            def close(self):
                if not self.closed:
                    super().close()
                    raise cleanup
        with patch.object(file_io, '_owned_slot', Broken), \
                patch.object(tensor_file, 'BaseExceptionGroup', side_effect=allocation, create=True):
            with self.assertRaises(KeyboardInterrupt) as caught:
                with tensor_file._regular_stream(self.path):
                    raise primary
        self.assertIs(caught.exception, primary)
        self.assertIs(BaseException.__cause__.__get__(primary), allocation)
        namespace = BaseException.__dict__['__dict__'].__get__(primary)
        failures = dict.__getitem__(namespace, '_tensorfold_checkpoint_file_failures')
        self.assertEqual(failures, [cause, context, cleanup])
        self.assert_retired(self.owners[0])

    def test_opaque_exception_and_hostile_dictionary_do_not_replace_primary(self):
        hooks = []
        class Opaque(KeyboardInterrupt):
            def __getattribute__(self, name):
                if name in ('__dict__', '__cause__', '__context__', '__suppress_context__', '__notes__'):
                    hooks.append(name)
                    raise LookupError('opaque getter')
                return super().__getattribute__(name)
            def __setattr__(self, name, value):
                if name in ('__cause__', '__context__', '__suppress_context__'):
                    hooks.append(name)
                    raise LookupError('opaque setter')
                super().__setattr__(name, value)
        class Hostile(dict):
            def __setitem__(self, *args):
                raise LookupError('dictionary publication refused')
            def get(self, *args):
                raise LookupError('dictionary getter refused')
            def pop(self, *args):
                raise LookupError('dictionary removal refused')
        primary = Opaque('opaque consumer error')
        primary.__dict__ = Hostile()
        cleanup = OSError('consumed close')
        original = self.owner
        class Broken(original):
            def close(self):
                if not self.closed:
                    super().close()
                    raise cleanup
        with patch.object(file_io, '_owned_slot', Broken):
            with self.assertRaises(KeyboardInterrupt) as caught:
                with tensor_file._regular_stream(self.path):
                    raise primary
        self.assertIs(caught.exception, primary)
        members = BaseException.__cause__.__get__(primary).exceptions
        self.assertIs(members[0], cleanup)
        self.assertIsInstance(members[1], LookupError)
        self.assertEqual(hooks, ['__notes__'])
        self.assert_retired(self.owners[0])

    def test_consumed_close_error_does_not_retry_reused_descriptor(self):
        cleanup = OSError('close consumed original descriptor')
        replacements, calls = [], []
        original = self.owner
        class Broken(original):
            def close(self):
                if not self.closed:
                    descriptor = self.fileno()
                    super().close()
                    calls.append(descriptor)
                    replacement = os.open(self_test.path, os.O_RDONLY | os.O_NOFOLLOW)
                    replacements.append(replacement)
                    self_test.assertEqual(replacement, descriptor)
                    raise cleanup
        self_test = self
        try:
            with patch.object(file_io, '_owned_slot', Broken):
                with self.assertRaises(OSError) as caught:
                    tensor_file.read_header(self.path, {'U8': 1})
            self.assertIs(caught.exception, cleanup)
            self.assertTrue(self.owners[0].closed)
            self.assertEqual(calls, [replacements[0]])
            self.assertEqual(len(os.read(replacements[0], 8)), 8)
        finally:
            for descriptor in replacements:
                os.close(descriptor)

    def test_same_primary_close_failure_never_creates_self_cause(self):
        primary = KeyboardInterrupt('same consumer and close failure')
        primary.__cause__ = primary
        original = self.owner
        class Broken(original):
            def close(self):
                if not self.closed:
                    super().close()
                    raise primary
        with patch.object(file_io, '_owned_slot', Broken):
            with self.assertRaises(KeyboardInterrupt) as caught:
                with tensor_file._regular_stream(self.path):
                    raise primary
        self.assertIs(caught.exception, primary)
        self.assertIsNone(BaseException.__cause__.__get__(primary))
        self.assert_retired(self.owners[0])


if __name__ == '__main__':
    unittest.main()
