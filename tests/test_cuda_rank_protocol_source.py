"""Actual rank protocol AST and bounded allocation refusal, no SDK."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import struct
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from tensorfold.cuda import rank_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]


def actual_function(path, name, namespace, *, owner=None):
    tree = ast.parse((ROOT / path).read_bytes())
    nodes = tree.body if owner is None else next(node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module([ast.ImportFrom("__future__", [ast.alias("annotations")], 0), node], [])), path, "exec"), namespace)
    return namespace[name]


class RankProtocolTests(unittest.TestCase):
    def setUp(self):
        sampling = ModuleType("tensorfold.engine.exact_sampling")
        sampling.Sampling = lambda *values: tuple(values)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {sampling.__name__: sampling}).start()
        self.request = actual_function("src/tensorfold/families/nemotron_h/cuda/app.py", "_request",
                                       {"ints": protocol.ints, "grammar": protocol.grammar, "finite": protocol.finite})
        self.unpack = actual_function("src/tensorfold/families/nemotron_h/cuda/app.py", "_unpack",
                                      {"request_json": protocol.request_json, "_request": self.request,
                                       "MAX_GRAMMAR_ITEMS": protocol.MAX_GRAMMAR_ITEMS})
        self.body = {"prompt": [1, 2], "max_tokens": 5, "draft": True, "cached": 0, "stop_eos": True,
                     "sampling": [1, 1.0, 20, 0.95, 0.0], "grammar": []}

    def test_valid_exact_request_and_explicit_stop(self):
        value = self.unpack(json.dumps(self.body).encode(), maximum=64, vocab=30)
        self.assertEqual(value, ([1, 2], 5, (1, 1.0, 20, 0.95, 0.0), True, 0, [], True))
        self.assertIsNone(self.unpack(b'{"stop":true}', maximum=64, vocab=30))
        for stop in (0, 1, "true", None):
            with self.assertRaises(ValueError):
                self.unpack(json.dumps({"stop": stop}), maximum=64, vocab=30)

    def test_refuses_schema_counts_types_tokens_sampling_and_grammar(self):
        for key, value in (("prompt", []), ("prompt", [True]), ("prompt", [30]), ("prompt", list(range(65))),
                           ("max_tokens", 0), ("max_tokens", 63), ("max_tokens", True), ("cached", 2),
                           ("draft", 1), ("stop_eos", None), ("sampling", [1, float("nan"), 20, 1.0, 0]),
                           ("sampling", [True, 1.0, 20, 1.0, 0]), ("sampling", [1, 1.0, -1, 1.0, 0]),
                           ("sampling", [1, 1.0, 20, 1.0, 2]), ("grammar", [5, 0]),
                           ("grammar", [0, 31]), ("grammar", [0, 0, 256]), ("grammar", [0, 0, 255])):
            body = {**self.body, key: value}
            with self.assertRaises((ValueError, UnicodeDecodeError), msg=(key, value)):
                self.unpack(json.dumps(body), maximum=64, vocab=30)
        with self.assertRaises(ValueError):
            self.unpack(json.dumps({**self.body, "extra": 1}), maximum=64, vocab=30)

    def test_size_and_nesting_refused_before_json_allocation(self):
        with patch.object(protocol, "MAX_MESSAGE_BYTES", 64):
            with patch.object(protocol.json, "loads", side_effect=AssertionError("decoder must not enter")):
                for raw in (b" " * 65, " " * 65, '{"prompt":[[[1]]]}'):
                    with self.assertRaises(ValueError):
                        protocol.request_json(raw, counts={"prompt": 64})
        for raw in ('{"cached":1,"cached":2}', '{"cached":123456789012345678901}',
                    '{"cached":1e9999}', '{"cached":Infinity}', '{"cached":0.' + '0' * 25 + '}'):
            with self.assertRaises(ValueError):
                protocol.request_json(raw, counts={"prompt": 64})
        with patch.object(protocol.json, "loads", side_effect=AssertionError("decoder must not enter")):
            for raw in ('{"prompt":[1,2,3]}', '{"unknown":[1]}', '{"prompt":["text"]}'):
                with self.assertRaises(ValueError):
                    protocol.request_json(raw, counts={"prompt": 2})

    def test_sender_validates_before_store_publication(self):
        share = actual_function("src/tensorfold/families/nemotron_h/cuda/app.py", "_share",
                                {"packed_grammar": lambda *args: [], "_request": self.request,
                                 "bounded_json": protocol.bounded_json},
                                owner="NemotronEngine")
        writes = []
        owner = SimpleNamespace(e=SimpleNamespace(c=SimpleNamespace(vocab=30)), context_window=64, served=0,
                                _key=lambda count: str(count), comm=SimpleNamespace(store=SimpleNamespace(set=lambda *args: writes.append(args))))
        with self.assertRaises(ValueError):
            share(owner, [30], 1, None, True, 0)
        self.assertEqual(writes, [])
        value = share(owner, [1], 1, None, True, 0)
        self.assertEqual(value[:2], ([1], 1))
        self.assertEqual(len(writes), 1)

    def test_glm_count_rejected_before_payload_allocation(self):
        allocations = []
        class Tensor:
            def __init__(self, values):
                self.values = list(values)
            def __getitem__(self, index):
                return SimpleNamespace(item=lambda: self.values[index])
        runtime = SimpleNamespace(int32="i32", tensor=lambda values, **kw: Tensor(values),
                                  empty=lambda shape, **kw: (allocations.append(shape) or Tensor([0] * shape[0])),
                                  zeros=lambda shape, **kw: (_ for _ in ()).throw(AssertionError("payload allocated")))
        def gather(send, got):
            got.values[0] = 100
        share = actual_function("src/tensorfold/families/glm5_next/cuda/engine.py", "_share", {"ints": protocol.ints}, owner="GlmEngine")
        owner = SimpleNamespace(rank=1, torch=runtime, comm=SimpleNamespace(all_gather=gather))
        with self.assertRaises(ValueError):
            share(owner, None, maximum=19)
        self.assertEqual(allocations, [(2,)])
        owner.rank = 0
        allocations.clear()
        with self.assertRaises(ValueError):
            share(owner, [1 << 31], maximum=19)
        self.assertEqual(allocations, [])

    def test_glm_sampling_refused_before_follower_wakeup_or_mutation(self):
        generate = actual_function("src/tensorfold/families/glm5_next/cuda/engine.py", "_generate",
                                   {"ints": protocol.ints, "finite": protocol.finite,
                                    "encode_policy": lambda spec: [0, 0, 0, 0],
                                    "_f64_ints": lambda value: list(struct.unpack("<2i", struct.pack("<d", float(value)))),
                                    "packed_grammar": lambda *args: [], "MAX_GRAMMAR_ITEMS": protocol.MAX_GRAMMAR_ITEMS},
                                   owner="GlmEngine")
        calls = []
        owner = SimpleNamespace(limit=64, serial_only=True, request=SimpleNamespace(),
                                w=SimpleNamespace(cfg=SimpleNamespace(vocab=30)), _effective=lambda code: code,
                                _resume=lambda *args: None, _ring=lambda: calls.append("ring"),
                                _share=lambda *args, **kwargs: calls.append("share"),
                                _run=lambda *args: (calls.append("run") or {}))
        for field, value in (("temperature", float("nan")), ("temperature", float("inf")),
                             ("top_p", float("nan")), ("min_p", float("nan")), ("min_p", -1.), ("min_p", 2.)):
            values = {"seed": 1, "temperature": 1., "top_k": 20, "top_p": .95, "min_p": 0., field: value}
            with self.assertRaises(ValueError):
                generate(owner, [1], 1, SimpleNamespace(**values), lambda tokens: None)
            self.assertEqual(calls, [])
        generate(owner, [1], 1, SimpleNamespace(seed=1, temperature=1., top_k=20, top_p=.95, min_p=0.), lambda tokens: None)
        self.assertEqual(calls, ["ring", "share", "share", "run"])


if __name__ == "__main__":
    unittest.main()
