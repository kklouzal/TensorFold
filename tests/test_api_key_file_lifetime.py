"""The new API-key file boundary uses the fork's published owner and cleanup containment."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tensorfold.server import authentication


def stamp(*, mtime=1):
    return SimpleNamespace(st_dev=1, st_ino=1, st_mtime_ns=mtime, st_size=20, st_mode=0o100600)


class KeyFileLifetime(unittest.TestCase):
    def setUp(self):
        self.store = authentication.KeyStore()
        self.store.key_file = Path('controlled-key-file')

    def scope(self, *, failure=None, retired=True, read_error=None):
        store = self.store

        class Scope:
            def __init__(self, *, max_files):
                self.retired = retired
                self.max_files = max_files
                self.opened = self.drained = False

            def open(self, path, flags, mode, *, max_bytes):
                assert store._read_scope is self
                assert self.max_files == 1 and max_bytes == 1 << 20
                self.opened = True
                if read_error is not None:
                    raise read_error
                owner = SimpleNamespace(fileno=lambda: 1)
                return SimpleNamespace(owner=owner, stream=BytesIO(b'alpha: controlled-key\n')), stamp()

            def drain(self):
                self.drained = True
                return [failure] if failure is not None else []

        return Scope

    def test_scope_is_published_before_acquisition_and_retires_after_digest_snapshot(self):
        with patch.object(authentication, 'FileStreams', self.scope()), \
             patch.object(authentication.os, 'fstat', return_value=stamp()):
            keys, _ = self.store._read()
        self.assertIsNone(self.store._read_scope)
        self.assertEqual(keys, ((authentication._digest('controlled-key'), 'alpha'),))

    def test_failed_open_is_drained_and_primary_is_preserved(self):
        failure = OSError('controlled acquisition failure')
        with patch.object(authentication, 'FileStreams', self.scope(read_error=failure)), \
             self.assertRaises(OSError) as caught:
            self.store._read()
        self.assertIs(caught.exception, failure)
        self.assertIsNone(self.store._read_scope)

    def test_unretired_cleanup_is_contained_and_scope_cannot_be_overwritten(self):
        failure = OSError('controlled retirement failure')
        with patch.object(authentication, 'FileStreams', self.scope(failure=failure, retired=False)), \
             patch.object(authentication.os, 'fstat', return_value=stamp()), \
             self.assertRaises(RuntimeError) as caught:
            self.store._read()
        self.assertIs(caught.exception.__cause__, failure)
        scope = self.store._read_scope
        self.assertTrue(scope.drained)
        with self.assertRaises(RuntimeError):
            self.store._read()
        self.assertIs(self.store._read_scope, scope)

    def test_cleanup_failure_remains_an_error_even_after_resource_retirement(self):
        failure = OSError('controlled completed-close failure')
        with patch.object(authentication, 'FileStreams', self.scope(failure=failure)), \
             patch.object(authentication.os, 'fstat', return_value=stamp()), \
             self.assertRaises(RuntimeError) as caught:
            self.store._read()
        self.assertIs(caught.exception.__cause__, failure)
        self.assertIsNone(self.store._read_scope)

    def test_changed_key_file_is_not_published(self):
        with patch.object(authentication, 'FileStreams', self.scope()), \
             patch.object(authentication.os, 'fstat', return_value=stamp(mtime=2)), \
             self.assertRaisesRegex(ValueError, 'changed while being read'):
            self.store._read()
        self.assertEqual(self.store._file, ())
        self.assertIsNone(self.store._read_scope)

    def test_unretired_reload_blocks_later_requests_inside_reload_throttle(self):
        store = authentication.KeyStore(["controlled-static"], clock=lambda: 0.0)
        store.key_file = Path('controlled-key-file')
        failure = authentication._KeyFileCleanupError('controlled reload retirement failure')
        owner = object()

        def fail_read():
            store._read_scope = owner
            raise failure

        with patch.object(Path, 'stat', return_value=stamp()), patch.object(store, '_read', fail_read):
            with self.assertRaises(RuntimeError) as first:
                store.match({'x-api-key': 'controlled-static'})
            self.assertIs(first.exception, failure)
            self.assertFalse(store._valid)
            self.assertEqual(store._file, ())
            with self.assertRaisesRegex(RuntimeError, 'unretired resource owner'):
                store.match({'x-api-key': 'controlled-static'})
            self.assertIs(store._read_scope, owner)

    def test_retired_reload_error_stays_closed_until_valid_file_read(self):
        now = [0.0]
        store = authentication.KeyStore(["controlled-static"], clock=lambda: now[0])
        store.key_file = Path('controlled-key-file')
        failure = authentication._KeyFileCleanupError('controlled completed retirement failure')
        with patch.object(Path, 'stat', return_value=stamp()), \
             patch.object(store, '_read', side_effect=failure):
            with self.assertRaises(RuntimeError) as first:
                store.match({'x-api-key': 'controlled-static'})
            self.assertIs(first.exception, failure)
            self.assertIsNone(store.match({'x-api-key': 'controlled-static'}))
        now[0] = 1.0
        keys = ((authentication._digest('controlled-next'), 'next'),)
        with patch.object(Path, 'stat', return_value=stamp()), patch.object(store, '_read', return_value=(keys, ())):
            self.assertEqual(store.match({'x-api-key': 'controlled-next'}), 'next')
            self.assertEqual(store.match({'x-api-key': 'controlled-static'}), 'cli-1')
