"""Execute actual Nemotron request-local mask control, no numerical runtime."""

import ast
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from types import SimpleNamespace, ModuleType
import unittest

REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "src/tensorfold/families/nemotron_h/cuda/decode.py"


class Injected(RuntimeError):
    pass


class CleanupFailure(RuntimeError):
    pass


class Tensor:
    def __getitem__(self, index):
        return self

    def clone(self):
        return self

    def copy_(self, *a, **kw):
        return self

    def fill_(self, *a, **kw):
        return self


class Engine:
    def __init__(self, primary=None):
        self.masked = False
        self.primary = primary
        self.close_error = None
        self.pos = 0
        self.parity = 0
        self.prev_keep = 0
        self.prefill_rows = 32
        self.max_rows = 8
        self.p_hidden = Tensor()
        self.hidden = Tensor()
        self.sampled = Tensor()
        self.c = SimpleNamespace(eos=())
        self.plain_prefill_observed_mask = None

    def reset(self):
        self.pos = self.parity = self.prev_keep = 0

    def restore(self, *args):
        pass

    def set_sampling(self, *args):
        pass

    def mask(self, constraint, window):
        if constraint is None:
            if self.close_error:
                raise self.close_error
            self.masked = False
        else:
            self.masked = True

    def prefill_chunk(self, *args, **kwargs):
        self.plain_prefill_observed_mask = self.masked
        if self.primary:
            raise self.primary

    def prefill_token(self):
        return 17

    def snapshot(self):
        return {"host": (0, 0, 0)}

    def forward(self, *args, **kwargs):
        pass

    def tokens(self):
        return [17, 18]

    def commit(self, *args):
        self.pos += 1


class Draft:
    def __init__(self):
        self.pos = 0
        self._count = 1
        self._copied = SimpleNamespace(synchronize=lambda: None)

    def restore(self, *a):
        pass

    def reset(self):
        pass

    def round(self, *args):
        self.pos += 1

    def drafts(self):
        return [17]

    def absorb_rows(self, *a):
        pass

    def snapshot(self):
        return {"pos": 0}


class Grammar:
    def window(self, tokens, *args):
        return SimpleNamespace(tokens=tokens, rows=[0])

    def advance(self, *args):
        pass


def scope():
    raw = PATH.read_text()
    tree = ast.parse(raw)
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name in ("Prefilled", "DecodeResult", "CopyIndex", "prefill", "draft_decode")
    ]
    module = ModuleType("nemotron_grammar_source_current")
    sys.modules[module.__name__] = module
    module.__dict__.update(
        dataclass=dataclass,
        field=field,
        time=time,
        torch=SimpleNamespace(
            no_grad=lambda: lambda fn: fn,
            cuda=SimpleNamespace(
                synchronize=lambda: None, current_stream=lambda: SimpleNamespace(synchronize=lambda: None)
            ),
        ),
    )
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            "maintained Nemotron primitive",
            "exec",
            flags=__import__("__future__").annotations.compiler_flag,
        ),
        module.__dict__,
    )
    return module.__dict__


class Controls(unittest.TestCase):
    def test_prefill_failure_clears_request_mask(self):
        primary = Injected("prefill fails")
        eng = Engine(primary)
        with self.assertRaises(Injected) as got:
            scope()["prefill"](eng, None, [1, 2], None, constraint=Grammar())
        self.assertIs(got.exception, primary)
        self.assertEqual(eng.masked, False)

    def test_callback_failure_clears_request_mask(self):
        ns = scope()
        eng = Engine()
        primary = Injected("callback fails")
        pref = ns["Prefilled"]([1], 17, Tensor(), {}, None)

        def callback(tokens):
            raise primary

        with self.assertRaises(Injected) as got:
            ns["draft_decode"](eng, Draft(), pref, 4, None, copy=False, constraint=Grammar(), on_tokens=callback)
        self.assertIs(got.exception, primary)
        self.assertEqual(eng.masked, False)

    def test_new_plain_prefill_clears_stale_request_selection_before_sampling(self):
        eng = Engine()
        eng.masked = True
        scope()["prefill"](eng, None, [1, 2], None)
        self.assertEqual(eng.plain_prefill_observed_mask, False)

    def test_cleanup_failure_preserves_primary_with_cause(self):
        primary = Injected("primary")
        eng = Engine(primary)
        eng.close_error = CleanupFailure("cleanup")
        with self.assertRaises(CleanupFailure):
            scope()["prefill"](eng, None, [1, 2], None, constraint=Grammar())
        eng = Engine()
        eng.close_error = CleanupFailure("cleanup")
        ns = scope()
        pref = ns["Prefilled"]([1], 17, Tensor(), {}, None)

        def callback(tokens):
            raise primary

        with self.assertRaises(Injected) as got:
            ns["draft_decode"](eng, Draft(), pref, 4, None, copy=False, constraint=Grammar(), on_tokens=callback)
        self.assertIs(got.exception, primary)
        self.assertIs(got.exception.__cause__, eng.close_error)


if __name__ == "__main__":
    unittest.main()
