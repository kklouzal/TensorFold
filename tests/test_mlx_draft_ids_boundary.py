"""Real-file parser controls with explicit Python descriptor-owner substitute."""
from __future__ import annotations

import ast
import importlib.util
import io
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = REPO / 'src/tensorfold/families/qwen4_exp/draft_head.py'
source_tree = ast.parse(source.read_bytes())
source_tree.body = [node for node in source_tree.body
                    if isinstance(node, ast.FunctionDef) and node.name in ('_drain_ids', '_listed_ids')
                    or isinstance(node, (ast.Import, ast.ImportFrom)) and not (
                        isinstance(node, ast.Import) and any(alias.name.startswith(('mlx', 'numpy')) for alias in node.names)
                        or isinstance(node, ast.ImportFrom) and node.module.startswith(('mlx', 'numpy')))]
candidate = SimpleNamespace()
exec(compile(ast.fix_missing_locations(source_tree), str(source), 'exec'), candidate.__dict__)
candidate.listed_ids = candidate._listed_ids
wrapper = next(node for node in ast.parse(source.read_bytes()).body
               if isinstance(node, ast.FunctionDef) and node.name == 'draft_ids')
candidate.VOCAB_FILE = REPO / 'src/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt'
wrapper_tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                                wrapper], type_ignores=[])
exec(compile(ast.fix_missing_locations(wrapper_tree), str(source), 'exec'), candidate.__dict__)
core = load('_draft_id_python_streams', REPO / 'src/tensorfold/file_io.py')


class Slot:
    """FileIO owns an actual FD; no native C-return/interruption qualification."""
    made = []
    fail_close = []
    def __init__(self):
        self.file = None
        type(self).made.append(self)
    @property
    def closed(self):
        return self.file is None or self.file.closed
    def open(self, path, flags, mode):
        self.file = open(path, 'rb', buffering=0, opener=lambda name, unused: os.open(name, flags, mode))
    def fileno(self):
        return self.file.fileno()
    def close(self):
        if self.fail_close:
            raise self.fail_close.pop(0)
        self.file.close()


def original(text, multiple):
    values = {int(token) for token in text.split()}
    if not values:
        raise ValueError
    extra, trial = [], 0
    while (len(values) + len(extra)) % multiple:
        if trial not in values:
            extra.append(trial)
        trial += 1
    return sorted(values | set(extra))


class DraftBoundary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'ids.txt'
        Slot.made, Slot.fail_close = [], []
        self.owner = patch.object(core, '_owned_slot', Slot)
        self.modules = patch.dict(sys.modules, {'tensorfold.file_io': core})
        self.owner.start()
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(self.owner.stop)

    def call(self, contents, *, vocab=256, multiple=64):
        self.path.write_bytes(contents.encode('utf8') if isinstance(contents, str) else contents)
        return candidate.listed_ids(self.path, vocab=vocab, multiple=multiple)

    def test_legacy_int_spelling_dedup_sort_padding_and_independent_forest(self):
        for text in ('0 2 2', ' +١_٢ \u2003００７ -0 ', '0_1 +2\t3', '0000 0_0 +000_002'):
            self.assertEqual(self.call(text, multiple=4), original(text, 4))
        rng = random.Random(830771)
        for _ in range(400):
            ids = [rng.randrange(256) for _ in range(rng.randrange(1, 150))]
            text = ' \n'.join(str(value) for value in ids)
            self.assertEqual(self.call(text), original(text, 64))
        self.assertTrue(all(owner.closed for owner in Slot.made))

    def test_shipped_default_identity_and_smallest_unused_padding(self):
        path = REPO / 'src/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt'
        text = path.read_text(encoding='utf8')
        expected = original(text, 64)
        actual = candidate.listed_ids(path, vocab=1 << 32, multiple=64)
        self.assertEqual(actual, expected)
        self.assertEqual(len({int(token) for token in text.split()}), 79591)
        self.assertEqual(len(actual) % 64, 0)
        self.assertTrue(all(owner.closed for owner in Slot.made))

    def test_public_wrapper_original_uint32_conversion_follows_owner_retirement(self):
        calls = []
        def convert(ids, *, dtype):
            self.assertTrue(all(owner.closed for owner in Slot.made))
            calls.append((ids, dtype))
            return tuple(ids)
        with patch.object(candidate, 'np', SimpleNamespace(array=convert, uint32='U32'), create=True):
            self.path.write_text('0 7 7', encoding='utf8')
            self.assertEqual(candidate.draft_ids(self.path, multiple=4, vocab=16), (0, 1, 2, 7))
            self.assertEqual(calls, [([0, 1, 2, 7], 'U32')])
            self.path.write_text('16', encoding='utf8')
            with self.assertRaises(ValueError):
                candidate.draft_ids(self.path, multiple=4, vocab=16)
            self.assertEqual(len(calls), 1)
        self.assertTrue(all(owner.closed for owner in Slot.made))

    def test_runtime_startup_passes_current_head_rows_before_mapping(self):
        runtime = ast.parse((REPO / 'src/tensorfold/families/qwen4_exp/runtime.py').read_bytes())
        cls = next(node for node in runtime.body if isinstance(node, ast.ClassDef) and node.name == 'FlashNext')
        constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        calls = [node for node in ast.walk(constructor) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'draft_ids']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args, [])
        self.assertEqual([keyword.arg for keyword in calls[0].keywords], ['vocab'])
        self.assertEqual(ast.dump(calls[0].keywords[0].value, include_attributes=False),
                         ast.dump(ast.parse('model.lm_head.weight.shape[0]', mode='eval').body,
                                  include_attributes=False))
        statements = ast.get_source_segment((REPO / 'src/tensorfold/families/qwen4_exp/runtime.py').read_text(),
                                            constructor)
        self.assertLess(statements.index('ids = draft_ids('), statements.index('ids.flags.writeable = False'))
        self.assertLess(statements.index('ids.flags.writeable = False'), statements.index('self._draft_ids = mx.array(ids)'))

    def test_malformed_and_outside_vocabulary_refuse_and_retire(self):
        for text in ('', '-1', '256', '1.0', '1e2', '#2', '1__2', '_1', '1_', '+', '--1', b'\xff'):
            with self.subTest(text=text), self.assertRaises((ValueError, UnicodeError)):
                self.call(text)
            self.assertTrue(all(owner.closed for owner in Slot.made))
        for values in ((True, 64), (256, 0), (256, -1), (256, 257), (256.0, 64), (256, 64.0)):
            with self.assertRaises(ValueError):
                candidate.listed_ids(self.path, vocab=values[0], multiple=values[1])
        with self.assertRaises(ValueError):
            self.call('0 1 2', vocab=3, multiple=2)

    def test_long_tokens_and_utf8_cross_chunk_have_bounded_live_state(self):
        text = '0' * (3 * (64 << 10)) + '_０ 7 7'
        previous = sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(0)
            self.assertEqual(self.call(text, multiple=2), original(text, 2))
        finally:
            sys.set_int_max_str_digits(previous)
        self.assertTrue(all(owner.closed for owner in Slot.made))

    def test_configured_decimal_digit_limit_includes_unicode_excludes_sign_and_underscores(self):
        previous = sys.get_int_max_str_digits()
        try:
            for limit in (640, 4300):
                sys.set_int_max_str_digits(limit)
                for text in ('0' * limit, '+' + '٠_' * (limit - 1) + '٠'):
                    self.assertEqual(self.call(text, multiple=2), original(text, 2))
                for text in ('0' * (limit + 1), '-' + '０_' * limit + '０'):
                    with self.assertRaises(ValueError):
                        original(text, 2)
                    with self.assertRaises(ValueError):
                        self.call(text, multiple=2)
        finally:
            sys.set_int_max_str_digits(previous)
        self.assertTrue(all(owner.closed for owner in Slot.made))

    def test_regular_fifo_and_growth_refuse_before_unbounded_reads(self):
        fifo = self.path.with_name('pipe')
        os.mkfifo(fifo)
        with self.assertRaises(ValueError):
            candidate.listed_ids(fifo, vocab=256, multiple=64)
        self.assertTrue(all(owner.closed for owner in Slot.made))
        reads = []
        def buffered(raw, mode):
            stream = io.BufferedReader(raw)
            class Growing:
                def read(self, size):
                    reads.append(size)
                    result = stream.read(size)
                    if len(reads) == 1:
                        with self.path.open('ab') as writer:
                            writer.write(b' 9')
                    return result
                def close(self):
                    stream.close()
            view = Growing()
            view.path = self.path
            return view
        with patch.object(core, '_buffered', buffered), self.assertRaises(ValueError):
            self.call('2')
        self.assertEqual(reads, [1, 1])
        self.assertTrue(all(owner.closed for owner in Slot.made))

    def test_close_failure_makes_success_an_operation_error(self):
        cleanup = OSError('explicit cleanup failure')
        Slot.fail_close = [cleanup]
        try:
            self.call('2')
        except BaseException as actual:
            self.assertIs(actual, cleanup)
        else:
            self.fail('fallible close was ignored')
        self.assertTrue(all(owner.closed for owner in Slot.made))

    def test_group_allocation_keeps_primary_scope_and_native_prior_frames(self):
        primary, native, context, cleanup, allocation = ValueError(), LookupError(), EOFError(), OSError(), MemoryError()
        BaseException.__cause__.__set__(primary, native)
        BaseException.__context__.__set__(primary, context)
        scope = SimpleNamespace(drain=lambda: [cleanup], retired=True)
        with patch.object(candidate, 'BaseExceptionGroup', side_effect=allocation, create=True):
            try:
                candidate._drain_ids(scope, primary)
            except BaseException as actual:
                self.assertIs(actual, primary)
                self.assertIs(BaseException.__cause__.__get__(actual), allocation)
                trace = BaseException.__traceback__.__get__(actual)
                frames = []
                while trace is not None:
                    frames.append(trace.tb_frame.f_locals)
                    trace = trace.tb_next
                owner = next(frame for frame in frames if frame.get('scope') is scope
                             and frame.get('native_cause') is native)
                self.assertIs(owner['native_cause'], native)
                self.assertIs(owner['native_context'], context)
                self.assertEqual(owner['errors'], [cleanup])
            else:
                self.fail('group exhaustion replaced/ignored primary')

    def test_primary_selected_cleanup_keeps_its_existing_native_fields(self):
        primary, native, context, secondary = OSError(), LookupError(), EOFError(), ArithmeticError()
        BaseException.__cause__.__set__(primary, native)
        BaseException.__context__.__set__(primary, context)
        scope = SimpleNamespace(drain=lambda: [primary, secondary], retired=True)
        try:
            candidate._drain_ids(scope, None)
        except BaseException as actual:
            self.assertIs(actual, primary)
            self.assertEqual(BaseException.__cause__.__get__(actual).exceptions, (native, context, secondary))
        else:
            self.fail('cleanup primary or prior native statuses disappeared')


if __name__ == '__main__':
    unittest.main()
