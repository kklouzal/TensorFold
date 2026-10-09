"""Actual persistence callers with real files and labeled FD transport only."""

import ast
import __future__
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from snapshot_fd_transport import TransportOwner
import test_prefix_snapshot_authority_boundary as prefix_fixture
from tensorfold.server import checkpoints
from tensorfold.engine import snapshot_file


class CallerControls(unittest.TestCase):
    def setUp(self):
        # Reuse the declared source fixture, not its inherited test methods.
        prefix_fixture.PrefixControls.setUp(self)
        self.cache = lambda tokens: prefix_fixture.PrefixControls.cache(self, tokens)
        self.save = lambda *args, **kwargs: prefix_fixture.PrefixControls.save(self, *args, **kwargs)
        self.load = lambda *args, **kwargs: prefix_fixture.PrefixControls.load(self, *args, **kwargs)
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/tensorfold/server/scheduler.py').read_text())
        scheduler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Scheduler')
        methods = [node for node in scheduler.body if isinstance(node, ast.FunctionDef) and node.name in ('_read_disk_block', '_persist')]
        namespace = {'Any': object, 'time': time}
        exec(compile(ast.Module(body=methods, type_ignores=[]), '<actual snapshot caller>', 'exec',
                     flags=__future__.annotations.compiler_flag), namespace)
        self.methods = namespace
        module = patch.dict(sys.modules, {'tensorfold.engine.prefix_snapshots': self.api})
        module.start()
        self.addCleanup(module.stop)

    def broken_slots(self, error):
        class Slot(TransportOwner):
            def close(self):
                before = self.closed
                super().close()
                if not before:
                    raise error
        return patch('tensorfold.file_io._owned_slot', Slot)

    def test_actual_disk_index_propagates_consumed_header_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(Path(directory), 'current', [1, 2])
            error = OSError('owned header close failed')
            blocks = self.api.DiskBlocks(Path(directory), 'current', registry=self.registry)
            with self.broken_slots(error), self.assertRaises(OSError) as caught:
                blocks.blocks()
            self.assertIs(caught.exception, error)
            self.assertTrue(error.__notes__)
            self.assertEqual(self.codec.reads, [])
            self.assertTrue(path.is_file())

    def test_actual_scheduler_restore_propagates_early_owned_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(Path(directory), 'current', [1, 2])
            inserted = []
            owner = SimpleNamespace(checkpoints=SimpleNamespace(longest=lambda *_: 0,
                                     insert=lambda *args, **kwargs: inserted.append(args)),
                                    disk_blocks=SimpleNamespace(best=lambda *_: (path, [1, 2]), touch=lambda *_: None),
                                    session_blocks=None, prompt_memory=None, model_id='current',
                                    snapshot_registry=self.registry, snapshot_codec=self.codec)
            for kind in (OSError, ValueError):
                error = kind('early owned close failed')
                with self.subTest(kind=kind):
                    with self.broken_slots(error), self.assertRaises(kind) as caught:
                        self.methods['_read_disk_block'](owner, [1, 2, 3])
                    self.assertIs(caught.exception, error)
                    self.assertTrue(error.__notes__)
            self.assertEqual(inserted, [])
            self.assertEqual(self.codec.reads, [])

    def test_actual_save_spill_and_prune_wrappers_propagate_cleanup_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = checkpoints.CheckpointStore(2, list)
            entry = checkpoints.CheckpointEntry([1, 2], self.cache([1, 2]), [1, 2], nbytes=16)
            store._entries.append(entry)
            operations = (
                lambda: checkpoints.save_conversations(store, root, 'current', registry=self.registry, codec=self.codec),
                lambda: checkpoints.spill_conversation(entry, root, 'current', limit_bytes=256,
                                                       registry=self.registry, codec=self.codec),
            )
            for operation in operations:
                error = OSError('save owned close failed')
                with self.broken_slots(error), self.assertRaises(OSError) as caught:
                    operation()
                self.assertIs(caught.exception, error)
                self.assertTrue(error.__notes__)
            path = self.save(root, 'current', [1, 2])
            error = OSError('prune header close failed')
            with self.broken_slots(error), self.assertRaises(OSError) as caught:
                checkpoints.prune_conversations(root, 'current', 256)
            self.assertIs(caught.exception, error)
            self.assertTrue(path.is_file())

    def test_actual_scheduler_persist_propagates_owned_close_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = SimpleNamespace(snapshot_registry=self.registry, snapshot_codec=self.codec,
                                    snapshot_dir=Path(directory), model_id='current')
            error = OSError('system block close failed')
            with self.broken_slots(error), self.assertRaises(OSError) as caught:
                self.methods['_persist'](owner, [1, 2], self.cache([1, 2]))
            self.assertIs(caught.exception, error)
            self.assertTrue(error.__notes__)

    def test_nested_eviction_and_refusal_observe_annotated_operation_failure(self):
        def fail(_):
            raise error
        store = checkpoints.CheckpointStore(2, list, on_evict=fail)
        entry = checkpoints.CheckpointEntry([1, 2], [], [1, 2])
        for operation in (lambda: store._evicted([entry]), lambda: store.refuse([1, 2], [], 16, 'budget')):
            error = OSError('spill cleanup failed')
            error.add_note('owned cleanup failed')
            with self.assertRaises(OSError) as caught:
                operation()
            self.assertIs(caught.exception, error)
        self.assertEqual(store.spilled, 0)
        self.assertEqual(store.refused, 0)

    def test_existing_plain_full_disk_spill_refusal_policy_stays_bounded(self):
        def fail(_):
            raise OSError('plain full disk')
        store = checkpoints.CheckpointStore(2, list, on_evict=fail)
        with patch('builtins.print'):
            store._evicted([checkpoints.CheckpointEntry([1, 2], [], [1, 2])])
            store.refuse([1, 2], [], 16, 'budget')
        self.assertEqual(store.spilled, 0)
        self.assertEqual(store.refused, 1)

    def test_classifier_known_native_cause_precedes_hostile_key_inspection(self):
        calls = []
        class Key:
            def __hash__(self):
                return hash('__notes__')

            def __eq__(self, value):
                calls.append(value)
                raise LookupError('foreign notes key')
        error, cleanup, context = OSError('primary'), OSError('cleanup'), ValueError('context')
        error.__cause__, error.__context__, error.__suppress_context__ = cleanup, context, False
        dict.__setitem__(BaseException.__dict__['__dict__'].__get__(error), Key(), None)
        with self.assertRaises(OSError) as caught:
            snapshot_file.reject_snapshot_cleanup_failure(error)
        self.assertIs(caught.exception, error)
        self.assertIs(BaseException.__cause__.__get__(error), cleanup)
        self.assertIs(BaseException.__context__.__get__(error), context)
        self.assertIs(BaseException.__suppress_context__.__get__(error), False)
        self.assertEqual(calls, [])

    def test_classifier_no_cause_collision_and_string_subclass_notes_without_hooks(self):
        calls = []
        class Key:
            def __hash__(self):
                return hash('__notes__')

            def __eq__(self, value):
                calls.append(value)
                raise LookupError('foreign collision')
        class NotesKey(str):
            def __hash__(self):
                return str.__hash__(self)

            def __eq__(self, value):
                calls.append(value)
                raise LookupError('foreign string comparison')
        error, context = OSError('ordinary miss'), ValueError('suppressed parser context')
        error.__context__, error.__suppress_context__ = context, True
        dict.__setitem__(BaseException.__dict__['__dict__'].__get__(error), Key(), None)
        self.assertIsNone(snapshot_file.reject_snapshot_cleanup_failure(error))
        self.assertIs(BaseException.__context__.__get__(error), context)
        self.assertIs(BaseException.__suppress_context__.__get__(error), True)
        self.assertIsNone(BaseException.__cause__.__get__(error))
        for notes in ([], ['owned cleanup failed'], 'malformed notes'):
            error = OSError('native notes')
            dict.__setitem__(BaseException.__dict__['__dict__'].__get__(error), NotesKey('__notes__'), notes)
            if notes == []:
                self.assertIsNone(snapshot_file.reject_snapshot_cleanup_failure(error))
            else:
                with self.assertRaises(OSError) as caught:
                    snapshot_file.reject_snapshot_cleanup_failure(error)
                self.assertIs(caught.exception, error)
        self.assertEqual(calls, [])

    def test_classifier_metadata_iteration_failure_preserves_primary_and_status(self):
        error, context, inspection = OSError('primary'), ValueError('prior context'), MemoryError('notes view')
        error.__context__, error.__suppress_context__ = context, True
        class FailedView:
            @staticmethod
            def items(fields):
                raise inspection
        with patch.object(snapshot_file, 'dict', FailedView, create=True):
            with self.assertRaises(OSError) as caught:
                snapshot_file.reject_snapshot_cleanup_failure(error)
        self.assertIs(caught.exception, error)
        cause = BaseException.__cause__.__get__(error)
        self.assertIsInstance(cause, BaseExceptionGroup)
        self.assertEqual(cause.exceptions, (context, inspection))

    def test_classifier_empty_alias_cannot_hide_another_native_notes_field(self):
        calls = []
        alternate_hash = 17 if hash('__notes__') != 17 else 18
        class NotesKey(str):
            def __hash__(self):
                calls.append('hash')
                return alternate_hash

            def __eq__(self, value):
                calls.append('equality')
                raise LookupError('foreign string comparison')
        for alias, literal in (([], ['cleanup failed']), ([], 'malformed'),
                               (['cleanup failed'], []), ([], [])):
            error = OSError('notes aliases')
            fields = BaseException.__dict__['__dict__'].__get__(error)
            fields[NotesKey('__notes__')] = alias
            fields['__notes__'] = literal
            self.assertIs(dict.get(fields, '__notes__'), literal)
            calls.clear()
            if alias == literal == []:
                self.assertIsNone(snapshot_file.reject_snapshot_cleanup_failure(error))
            else:
                with self.assertRaises(OSError) as caught:
                    snapshot_file.reject_snapshot_cleanup_failure(error)
                self.assertIs(caught.exception, error)
            self.assertEqual(calls, [])

    def test_classifier_same_primary_profile_event_stays_failed_at_nested_catches(self):
        for context, suppressed in ((None, False), (None, True),
                                    (ValueError('prior context'), False), (ValueError('prior context'), True)):
            error = OSError('primary')
            error.__context__, error.__suppress_context__ = context, suppressed
            fields = BaseException.__dict__['__dict__'].__get__(error)
            fired = []

            def profiler(frame, event, function):
                if event == 'c_call' and getattr(function, '__name__', None) == 'items' and getattr(function, '__self__', None) is fields:
                    fired.append(True)
                    BaseException.__context__.__set__(error, None)
                    BaseException.__suppress_context__.__set__(error, not suppressed)
                    raise error

            previous_profiler = sys.getprofile()
            try:
                sys.setprofile(profiler)
                with self.assertRaises(OSError) as caught:
                    snapshot_file.reject_snapshot_cleanup_failure(error)
            finally:
                sys.setprofile(previous_profiler)
            self.assertIs(caught.exception, error)
            self.assertEqual(fired, [True])
            self.assertIs(BaseException.__context__.__get__(error), context)
            self.assertIs(BaseException.__suppress_context__.__get__(error), suppressed)
            cause = BaseException.__cause__.__get__(error)
            if context is None:
                marker = cause
            else:
                self.assertIsInstance(cause, BaseExceptionGroup)
                self.assertIs(cause.exceptions[0], context)
                marker = cause.exceptions[1]
            self.assertIsInstance(marker, snapshot_file._SnapshotCleanupInspectionFailure)
            self.assertIsNone(BaseException.__cause__.__get__(marker))
            self.assertIsNone(BaseException.__context__.__get__(marker))
            with self.assertRaises(OSError) as nested:
                snapshot_file.reject_snapshot_cleanup_failure(error)
            self.assertIs(nested.exception, error)

    def test_classifier_transport_failure_retains_formed_status_references(self):
        error, context = OSError('primary'), ValueError('prior context')
        inspection, transport = MemoryError('notes view'), MemoryError('transport')
        error.__context__, error.__suppress_context__ = context, True
        class FailedView:
            @staticmethod
            def items(fields):
                raise inspection
        with (patch.object(snapshot_file, 'dict', FailedView, create=True),
              patch.object(snapshot_file.SnapshotStreams, '_raise', side_effect=transport)):
            try:
                snapshot_file.reject_snapshot_cleanup_failure(error)
            except BaseException as caught:
                self.assertIs(caught, error)
                frames = []
                trace = caught.__traceback__
                while trace is not None:
                    frames.append(trace.tb_frame.f_locals)
                    trace = trace.tb_next
            else:
                self.fail('metadata transport failure returned a cache miss')
        self.assertIs(BaseException.__cause__.__get__(error), transport)
        self.assertTrue(any(frame.get('metadata_failure') is inspection and frame.get('previous') == (None, context, True)
                            for frame in frames))


if __name__ == '__main__':
    unittest.main()
