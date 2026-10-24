"""Exercise DSpark dispatch through the existing keyed verifier without CUDA."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'src/tensorfold/families/qwen4_exp/cuda'
if not ROOT.exists():
    from importlib.util import find_spec
    ROOT = Path(find_spec('tensorfold.families.qwen4_exp.cuda').origin).parent


def method(file, name, cls=None, **namespace):
    namespace.setdefault("DEPTH", 3)
    namespace.setdefault("CONFIDENCE", 0.0)
    tree = ast.parse((ROOT / file).read_text())
    nodes = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls).body if cls else tree.body
    node = next(x for x in nodes if isinstance(x, ast.FunctionDef) and x.name == name)
    node.decorator_list = []
    compiled = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(compiled), str(ROOT / file), 'exec'), namespace)
    return namespace[name]


class Dispatch(unittest.TestCase):
    def test_serial_uses_last_pending_token_and_runtime_confidence(self):
        calls = []
        runtime = NS(draft=lambda *a: calls.append(a) or [17, 19])
        e = NS(w=NS(dspark=runtime))
        fn = method('decode.py', 'draft', absorb=lambda *a: self.fail('stock MTP touched'))
        policy = object()
        self.assertEqual(fn(e, None, [101, 102], 99, 3, policy, .7), [17, 19])
        self.assertEqual(calls, [(e, 102, 3, policy, .7)])

    def test_zero_room_neither_invokes_runtime_nor_absorbs(self):
        e = NS(w=NS(dspark=NS(draft=lambda *a: self.fail('proposal without room'))))
        fn = method('decode.py', 'draft', absorb=lambda *a: self.fail('stock absorb'))
        self.assertEqual(fn(e, None, [], 1, 0, None), [])
        with self.assertRaisesRegex(ValueError, 'pending'):
            fn(e, None, [], 1, 1, None)

    def test_existing_stock_dispatch_and_absolute_position(self):
        calls = []
        e = NS(w=NS(), st=NS(), sample=lambda x, positions, policy, **kw: calls.append((positions, kw)) or [71])
        fn = method('decode.py', 'draft', absorb=lambda *a: [1])
        self.assertEqual(fn(e, None, [4], 37, 1, None), [71])
        self.assertEqual(calls, [([37], {'draft': True})])

    def test_multi_without_stock_buffers_proposes_after_committed_prefix(self):
        calls = []
        runtime = NS(draft=lambda *a: calls.append(a) or [41])
        obj = NS(w=NS(dspark=runtime), depth=3, confidence=.2, buf=None, pbuf=None, capacity=4096, prefill_rows=2048)
        state, policy = object(), object()
        s = NS(sid=1, count=20, out=[1, 2, 3], st=state, sampling=policy, drafts=[999])
        fn = method('multi.py', '_draft_all', cls='MultiDecoder', _slot=lambda w, st, *a: NS(w=w, st=st))
        fn(obj, [(s, 0, [8, 9])])
        self.assertEqual(s.drafts, [41])
        self.assertIs(calls[0][0].st, state)
        self.assertEqual(calls[0][1:], (9, 3, policy, .2))

    def test_multi_remaining_room_clears_old_proposals(self):
        runtime = NS(draft=lambda *a: self.fail('no output room'))
        obj = NS(w=NS(dspark=runtime), depth=7)
        s = NS(count=3, out=[1, 2], drafts=[999])
        fn = method('multi.py', '_draft_all', cls='MultiDecoder')
        fn(obj, [(s, 0, [7])])
        self.assertEqual(s.drafts, [])

    def test_copy_context_is_temporary_and_aux_provider_can_be_skipped(self):
        calls = []
        runtime = NS(draft=lambda *a: self.fail('valid copy ignored'))
        obj = NS(w=NS(dspark=runtime), depth=3)
        s = NS(sid=7, count=20, out=[1], context=[11], drafts=[], copies=NS(propose=lambda *a: calls.append(a) or [31, 32, 33]))
        fn = method('multi.py', '_draft_all', cls='MultiDecoder', COPY_MATCH=8)
        fn(obj, [(s, 0, [17])], {7: [19]})
        self.assertEqual(s.context, [11])
        self.assertEqual(s.drafts, [31, 32, 33])
        self.assertEqual(calls[0][1], 8)

    def test_failed_copy_restores_context_and_keeps_no_proposals(self):
        def broken(*_):
            raise RuntimeError('copy failed')
        obj = NS(w=NS(dspark=object()), depth=3)
        s = NS(sid=7, count=20, out=[1], context=[11], drafts=[999], copies=NS(propose=broken))
        fn = method('multi.py', '_draft_all', cls='MultiDecoder', COPY_MATCH=8)
        with self.assertRaisesRegex(RuntimeError, 'copy failed'):
            fn(obj, [(s, 0, [17])], {7: [19]})
        self.assertEqual(s.context, [11])
        self.assertEqual(s.drafts, [])


class CoupledVerifier(unittest.TestCase):
    def test_dspark_mismatch_commits_only_accepted_inputs_and_skips_stock_final_absorb(self):
        state = NS(pos=11)
        w = NS(mtp=None)
        calls = []
        e = NS(w=w, st=state, buf=NS(streams=[0, 0, 0, 0]), last_streams=None,
               forward=lambda tokens: calls.append(('inputs', list(tokens))) or [0, 0, 0, 0],
               sample=lambda *a, **kw: [21, 31, 41, 51])
        def committed(_w, st, _b, rows, keep):
            calls.append(('commit', rows, keep))
            st.pos += keep
        def result(*a):
            return a
        fn = method('decode.py', 'mtp_decode', draft=lambda *a: [21, 999, 41], commit=committed,
                    absorb=lambda *a: self.fail('DSpark final stock absorb'), DecodeResult=result,
                    torch=NS(cuda=NS(synchronize=lambda: None)), time=NS(perf_counter=lambda: 1.0))
        value = fn(e, 10, 3, None, depth=3)
        self.assertEqual(value[0], [10, 21, 31])
        self.assertEqual(value[2:5], (1, 3, 1))
        self.assertEqual(calls, [('inputs', [10, 21, 999, 41]), ('commit', 4, 2)])
        self.assertEqual(state.pos, 13)


class AdmissionGeometry(unittest.TestCase):
    def test_shared_capacity_and_serial_twin_are_charged_before_admission(self):
        from dataclasses import dataclass
        import sys
        import types
        from unittest.mock import patch

        @dataclass
        class Geometry:
            bytes_at: object
            reserve: int
            minimum_slots: int

        tree = ast.parse((ROOT / 'engine.py').read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'FlashNextEngine')
        init = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_initialize')
        block = next(x for x in init.body if isinstance(x, ast.If) and ast.unparse(x.test) == 'dspark_plan is not None'
                     and any(isinstance(y, ast.Assign) and any(isinstance(z, ast.Name) and z.id == 'dspark_geometry'
                                                               for z in y.targets) for y in x.body))
        code = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), 'engine-admission', 'exec')
        cap = types.ModuleType('tensorfold.cuda.capacity')
        cap.Geometry = Geometry
        from itertools import product
        from tensorfold.families.qwen4_exp.cuda import dspark_proposal_graphs as proposal_module
        for slots, graphs, enabled in product((1, 2, 6), (False, True), (False, True)):
            calls = []
            plan = NS(config=NS(context=262144, block=7), resident_bytes=1000, loading_host_bytes=91,
                      capacity_bytes=lambda total, **kw: calls.append((total, kw)) or 7*total + 31)
            namespace = dict(dspark_plan=plan, streams=slots, each=4, rows0=2048, graphs=graphs, depth=3, tp=8,
                             resident_windows=None,
                             geometry=lambda text: Geometry(lambda total: 13*total + 97, 17, 21),
                             numeric_capacity={'resident_extra': 29}, host_capacity={'host_extra_staging': 43},
                             max_len=None, text={'max_position_embeddings':262144},
                             __package__='tensorfold.families.qwen4_exp.cuda')
            other = types.ModuleType('tensorfold.families.qwen4_exp.cuda.multi')
            other.FIRST = 256
            with patch.dict(sys.modules, {'tensorfold.cuda.capacity': cap, 'tensorfold.families.qwen4_exp.cuda.multi':other}), patch.object(proposal_module, '_ENABLED', enabled):
                exec(code, namespace)
            result = namespace['geometry']({})
            copies = 2 if slots == 1 else 1
            for total in (0, 256, 8193):
                self.assertEqual(result.bytes_at(total), 13*total+97 + 7*(total*copies + max(0,slots-1)*min(256,total))+31)
                self.assertEqual(calls[-1], (total*copies + max(0,slots-1)*min(256,total), {'main_rows': max(8, 4) if slots == 1 else slots*4,
                                                         'prompt_rows': 2048, 'slots': slots}))
            self.assertEqual((result.reserve, result.minimum_slots), (17, 21))
            self.assertEqual(namespace['max_len'], 262140)
            self.assertEqual(namespace['numeric_capacity']['resident_extra'], 1029)
            self.assertEqual(namespace['host_capacity']['host_extra_staging'], 134)
            expected_host = proposal_module.graph_host_bytes(slots, 3, 8, 262140) if graphs and enabled and slots > 1 else 0
            self.assertEqual(namespace['host_capacity'].get('host_workspace_bytes', 0), expected_host)


if __name__ == '__main__':
    unittest.main()
