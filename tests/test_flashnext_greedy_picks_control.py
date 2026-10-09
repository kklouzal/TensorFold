"""Actual draft caller source controls, using opaque words and no numerical SDK.

ROOT native/model receipts must independently qualify score/probability bits.
These controls preserve greedy max/LSE/exp and its single batched readback,
while requiring complete-source repair and unsigned64 keyed identity.
"""
import ast
import __future__
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'src/tensorfold/families/qwen4_exp/cuda/multi.py'


class Row(list):
    def __getitem__(self, key):
        if key is None:
            return self
        if isinstance(key, list):
            return Row([self[index] for index in key])
        result = super().__getitem__(key)
        return Row(result) if isinstance(key, slice) else result
    def astype(self, dtype):
        return self
    def view(self, dtype):
        return self
    def __eq__(self, token):
        return SimpleNamespace(hits=[index for index, value in enumerate(self) if value == token])


class Tensor:
    def __init__(self, name, state):
        self.name, self.state = name, state
        self.shape = (len(state['samplings']), state['width'])
    def float(self):
        return self
    def contiguous(self):
        return self
    def view(self, dtype):
        self.state['word_views'].append((self.name, dtype))
        return self
    def to(self, dtype):
        self.state['word_storage'].append((self.name, dtype))
        return self
    def __getitem__(self, key):
        return Tensor('fullrow', self.state)
    def max(self, **kwargs):
        self.state['maximum'] = kwargs
        return Tensor('top', self.state), Tensor('column', self.state)
    def cpu(self):
        self.state['host_reads'] += 1
        return self
    def numpy(self):
        if self.name == 'fullrow':
            return Row(range(self.state['width']))
        k = self.state['k']
        rows = [Row([2.] * k + list(range(k)) + [3., self.state['column'], self.state['normalizer']])
                for _ in self.state['samplings']]
        class Packet(list):
            def __getitem__(self, key):
                if isinstance(key, tuple):
                    return Row([row[key[1]] for row in self])
                return super().__getitem__(key)
        return Packet(rows)


def run(samplings, mapping=None, *, normalizer=4., width=8, column=5):
    cls = next(node for node in ast.parse(SOURCE.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == 'MultiDecoder')
    picks = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_picks')
    state = dict(samplings=samplings, topk=[], host_reads=0, k=0, repairs=0,
                 full_choices=0, word_views=[], word_storage=[], normalizer=normalizer, exp_inputs=[], width=width, column=column)
    def topk(row, k, **kw):
        state['topk'].append((k, kw))
        state['k'] = k
        return Tensor('values', state), Tensor('indices', state)
    def cat(parts, **kw):
        state['payload'] = [part.name for part in parts]
        return Tensor('packed', state)
    def repair(values, ids, positions, sampling, width, full):
        state['repairs'] += 1
        return [int(ids[0])]
    uint64 = object()
    def keys(positions, *, dtype):
        if dtype is not uint64:
            raise AssertionError('key positions lost unsigned64 representation')
        return positions
    def choose(values, ids, positions, sampling):
        state['full_choices'] += 1
        if len(values) != 8 or len(ids) != 8:
            raise AssertionError('top_k0 chooser did not receive the complete source')
        return [int(ids[-1])]
    def exponential(value):
        state['exp_inputs'].append(value)
        return .25
    torch = SimpleNamespace(topk=topk, cat=cat, int32='i32', int64='i64',
                            logsumexp=lambda row, **kw: Tensor('lse', state))
    np = SimpleNamespace(exp=exponential, int32='i32', int64='i64', float32='f32', uint64=uint64,
        nonzero=lambda value: (value.hits,), isfinite=lambda values: SimpleNamespace(all=lambda: all(
            value == value and value not in (float('inf'), -float('inf')) for value in values))
            if isinstance(values, list) else True,
        arange=lambda width, **kw: Row(range(width)), asarray=keys)
    namespace = dict(torch=torch, np=np, MARGIN=8, choose_rows=choose, repaired_choose=repair,
                     validate_rows=lambda *args: None, validate_policy=lambda *args: None)
    exec(compile(ast.Module(body=[picks], type_ignores=[]), str(SOURCE), 'exec',
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), namespace)
    host = Row(mapping) if mapping is not None else None
    owner = SimpleNamespace(draft_host=host, _draft_map=SimpleNamespace(ids=lambda width: host) if host is not None else None)
    values = namespace['_picks'](owner, Tensor('logits', state), [(1 << 64) - 1] * len(samplings), samplings)
    return state, values


class Controls(unittest.TestCase):
    def test_greedy_default_and_zero_temperature_preserve_original_arithmetic_and_one_readback(self):
        for policies in ([None] * 4, [SimpleNamespace(temperature=0, top_k=7)] * 4):
            state, values = run(policies)
            self.assertEqual(state['topk'], [])
            self.assertEqual(state['payload'], ['top', 'column', 'lse'])
            self.assertEqual(state['host_reads'], 1)
            self.assertEqual(state['repairs'], 0)
            self.assertEqual(state['exp_inputs'], [3. - 4.] * 4)
            self.assertEqual(values, [(5, .25)] * 4)
            self.assertEqual(state['word_storage'], [('top', 'i64'), ('lse', 'i64')])

    def test_greedy_mapping_keeps_signed64_identity(self):
        state, values = run([None] * 2, [0, 10, 20, 30, 40, (1 << 63) - 1, 60, 70])
        self.assertEqual(values, [((1 << 63) - 1, .25)] * 2)
        self.assertEqual(state['topk'], [])

    def test_unsorted_mapped_greedy_keeps_the_first_physical_maximum(self):
        state, values = run([None], [9, 2], width=2, column=0)
        self.assertEqual(values, [(9, .25)])
        self.assertEqual(state['topk'], [])
        self.assertEqual(state['host_reads'], 1)
        self.assertEqual(state['exp_inputs'], [3. - 4.])

    def test_positive_mixed_sampler_uses_boundary_repair_and_integer_columns(self):
        state, values = run([None, SimpleNamespace(temperature=.6, top_k=2)])
        self.assertEqual(state['topk'], [(8, dict(dim=-1, sorted=False))])
        self.assertEqual(state['payload'], ['values', 'indices', 'top', 'column', 'lse'])
        self.assertEqual(state['repairs'], 1)
        self.assertEqual(state['full_choices'], 0)
        self.assertEqual(values, [(5, .25), (0, .25)])
        self.assertNotIn(('indices', 'i64'), state['word_storage'])  # native topk columns already int64

    def test_topk0_uses_complete_original_source_and_full_selected_probability(self):
        state, values = run([SimpleNamespace(temperature=.6, top_k=0)])
        self.assertEqual(state['full_choices'], 1)
        self.assertEqual(state['host_reads'], 2)
        self.assertEqual(state['exp_inputs'], [7. - 4.])
        self.assertEqual(values, [(7, .25)])

    def test_nonfinite_original_fp32_normalizer_stops_confidence(self):
        for normalizer in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaisesRegex(ValueError, 'normalization is nonfinite'):
                run([None], normalizer=normalizer)

    def test_mapped_slot_borrows_the_startup_immutable_owner(self):
        text = SOURCE.read_text()
        self.assertIn('e._draft_map = self._draft_map', text)
        self.assertIn('self._draft_map = TokenMap(w.draft_ids', text)
        self.assertNotIn('_draft_sorted', text)
        self.assertIn('greedy keeps the first physical FP32 maximum', text)
        self.assertNotIn('w.draft_ids.cpu().numpy()', text)


if __name__ == '__main__':
    unittest.main()
