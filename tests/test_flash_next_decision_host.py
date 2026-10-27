"""Decision admission and collector tests using source seams without GPU-runtime imports."""

import ast
import os
from pathlib import Path
import sys
import threading
import types
import unittest

ROOT = Path(os.environ.get('TF_DECISION_TEST_SRC', Path(__file__).resolve().parents[1] / 'src'))


def load_source(name, path, injected=None):
    module = types.ModuleType(name)
    sys.modules[name] = module
    module.__dict__.update(injected or {})
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom) or not node.level]
    exec(compile(tree, str(ROOT / path), 'exec'), module.__dict__)
    return module


probabilities = load_source('decision_host_probabilities', 'tensorfold/engine/probabilities.py')
prefixes = load_source('decision_host_prefixes', 'tensorfold/families/qwen4_exp/cuda/prefixes.py')


class Sampling:
    def __init__(self, **values):
        self.__dict__.update(values)


def source_class(path, name, methods, injected):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    for node in ast.walk(cls):
        if isinstance(node, ast.FunctionDef):
            node.body = [item for item in node.body if not isinstance(item, ast.ImportFrom)]
    namespace = {'__name__': 'decision_host_methods', **injected}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(module, str(ROOT / path), 'exec'), namespace)
    return namespace[name]


Engine = source_class('tensorfold/families/qwen4_exp/cuda/engine.py', 'FlashNextEngine',
                      {'_limit', 'context_window', 'supports_logprobs', 'decision_label_refusal', 'score_labels'},
                      {'Sampling': Sampling, 'LabelProbabilities': probabilities.LabelProbabilities, 'threading': threading})


def engine():
    instance = Engine()
    instance._lifecycle = threading.Condition()
    instance._closing = instance._closed = False
    instance._abort_comm = instance._receiving = False
    instance._calls = {}
    instance.tp, instance.depth, instance.max_len = 1, 6, 64
    instance.w = types.SimpleNamespace(cfg=types.SimpleNamespace(vocab=32))
    instance.cache = [('kept', object())]
    instance.finite = True
    instance.calls = []
    def generate(prompt, count, sampling, emit, **options):
        instance.calls.append((prompt, count, options))
        probe = options['probabilities']
        probe.add_labels(len(prompt), [float(token) if instance.finite else float('nan') for token in probe.labels], 10.)
    instance.generate = generate
    return instance


class DecisionHostTests(unittest.TestCase):
    def test_collector_only_accepts_its_own_position(self):
        a, b = probabilities.LabelProbabilities([2, 3], 5), probabilities.LabelProbabilities([7], 6)
        a.add_labels(6, [9.], 11.)
        self.assertIsNone(a.label_logits)
        a.add_labels(5, [2., 3.], 10.)
        self.assertEqual(a.label_logits, [2., 3.])
        self.assertIsNone(b.label_logits)


    def test_solo_collector_does_not_request_a_decode_loop(self):
        item = engine()
        self.assertEqual(item.score_labels([1], [7]), ([7.], 10.))
        self.assertEqual(item.calls[0][1], 1)
        self.assertIs(item.calls[0][2]['draft'], False)


    def test_bad_label_is_rejected_before_queueing(self):
        item = engine()
        with self.assertRaises(ValueError):
            item.score_labels([2], [32])
        self.assertEqual(item.calls, [])

    def test_invalid_prompt_and_label_ids_reject_before_generation(self):
        for prompt, labels in (([], [1]), ([1], []), ([32], [1]), ([-1], [1]),
                               ([1.5], [1]), ([True], [1]), ([1], [False]), ([1], [2.5])):
            with self.subTest(prompt=prompt, labels=labels):
                item = engine()
                with self.assertRaises(ValueError):
                    item.score_labels(prompt, labels)
                self.assertEqual(item.calls, [])

    def test_a_fresh_score_preserves_existing_kept_prefixes(self):
        item = engine()
        kept = list(item.cache)
        self.assertEqual(item.score_labels([1, 2], [3, 4]), ([3., 4.], 10.))
        self.assertEqual(item.cache, kept)
        self.assertEqual(len(item.calls), 1)

    def test_nonfinite_score_is_rejected(self):
        item = engine()
        item.finite = False
        with self.assertRaises(ValueError):
            item.score_labels([1], [7])
        self.assertEqual(item._calls, {})

    def test_two_ranks_are_rejected_before_queueing(self):
        item = engine()
        item.tp = 2
        with self.assertRaises(ValueError):
            item.score_labels([1], [7])
        self.assertEqual(item.calls, [])


    def test_fresh_score_can_reclaim_an_idle_kept_slot(self):
        begun = []
        idle = types.SimpleNamespace(begin_sequence_generation=lambda: begun.append(True))
        owner = types.SimpleNamespace(kept=[([1], idle, {}, None)], free=[], depth=6, used={}, clock=0,
                                      _busy=lambda: set())
        owner._drop_kept = lambda state: setattr(owner, 'kept', [k for k in owner.kept if k[1] is not state])
        state, hit, cached = prefixes.slot_for(owner, [2], False)
        self.assertIs(state, idle)
        self.assertIsNone(hit)
        self.assertEqual(cached, 0)
        self.assertEqual(owner.kept, [])
        self.assertEqual(begun, [True])

    def test_decision_prompt_respects_the_existing_live_decode_budget(self):
        seen = []
        def budget(rows, live, *rest):
            seen.append(live)
            return 64 if live else rows
        Decoder = source_class('tensorfold/families/qwen4_exp/cuda/multi.py', 'MultiDecoder', {'_pass_rows', 'pass_rows_for'},
                               {'pass_limit': budget, 'PASS_MIN': 64})
        item = Decoder()
        item.controller = None
        item.streams = {1: types.SimpleNamespace(done=False)}
        item.filling = [types.SimpleNamespace(probabilities=probabilities.LabelProbabilities([1], 2))]
        item.prefill_rows, item.share, item.round_s, item.row_s = 1024, .25, .02, .0001
        self.assertEqual(item._pass_rows(), 64)
        self.assertEqual(seen, [True])


if __name__ == '__main__':
    unittest.main()
