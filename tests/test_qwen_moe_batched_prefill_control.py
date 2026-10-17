"""Pure host controls for admitted batch capacity and unsupported MoE TP.

These execute actual scheduler/early-refusal bodies, without importing or
simulating a tensor runtime. Native state/tap equivalence has CUDA tests.
"""
from __future__ import annotations
import __future__

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

BASE = Path(__file__).resolve().parents[1] / "src/tensorfold/families/qwen3_5/cuda"


def function(name):
    path = BASE / ("multi.py" if name == "_batch" else "prefill.py")
    node = deepcopy(next(n for n in ast.walk(ast.parse(path.read_text()))
                         if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    scope = {"STEP": 1024, "CHUNK": 4096}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    return scope[name]


class Controls(unittest.TestCase):
    def test_all_piece_rows_fit_the_admitted_forward_during_decode_and_prompt_only(self):
        # Several complete pieces followed by a partial piece: the cap applies
        # to the aggregate allocation, not independently to each request.
        lengths = [100, 300, 700, 1300]
        filling = [NS(background=False, vision=None, st=NS(pos=0), stops=[], prompt=range(n)) for n in lengths]
        cases = [(512, True, 512), (512, False, 512), (1024, True, 1024),
                 (2048, True, 1024), (2048, False, 2048), (None, True, 1024), (None, False, sum(lengths))]
        batch = function("_batch")
        for admitted, active, expected_rows in cases:
            with self.subTest(admitted=admitted, active=active):
                weights = NS() if admitted is None else NS(prompt_rows=admitted)
                decoder = NS(w=weights, filling=filling, streams={0: NS(done=not active)})
                result = batch(decoder)
                self.assertEqual(sum(stop - stream.st.pos for stream, stop in result), expected_rows)
                self.assertEqual([stream for stream, _ in result], filling[:len(result)])
                self.assertGreater(len(result), 1)
                self.assertTrue(all(stream.st.pos < stop <= len(stream.prompt) for stream, stop in result))
                self.assertTrue(all(stop == len(stream.prompt) for stream, stop in result[:-1]))

    def test_routed_tp_refuses_before_config_state_or_device_access(self):
        # No config/device/state exists: unsupported TP must refuse before any
        # preparation or mutation, rather than entering partially sharded MoE.
        weights = NS(layers=[NS(moe=None), NS(moe=object())])
        with self.assertRaisesRegex(ValueError, "routed experts run on one GPU"):
            function("prefill_rows")(weights, [], tp=True, capture_taps=True)


if __name__ == "__main__":
    unittest.main()
