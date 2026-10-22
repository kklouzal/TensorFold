"""Execute maintained finish/MTP control flow without a tensor runtime or GPU."""

from __future__ import annotations
import __future__

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

BASE = Path(__file__).resolve().parents[1] / "src/tensorfold/families/qwen4_exp/cuda"


def maintained(filename, names, scope):
    path = BASE / filename
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    # Execute trusted project AST functions only; this fixture accepts no external code.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",  # noqa: S102
                 flags=__future__.annotations.compiler_flag), scope)
    return scope


class Rows:
    """Only row selection/storage needed by the controls; no tensor arithmetic is simulated."""

    device = "cpu"

    def __init__(self, values, indices=None):
        self.values = values
        self.indices = list(range(len(values))) if indices is None else indices

    def __getitem__(self, index):
        selected = self.indices[index]
        return Rows(self.values, selected if isinstance(index, slice) else [selected])

    def copy_(self, other):
        assert len(self.indices) == len(other.indices)
        for index, value in zip(self.indices, other.items(), strict=True):
            self.values[index] = value

    def index_select(self, dim, indices):
        assert dim == 0
        values = self.items()
        return Rows([values[index] for index in indices])

    def view(self, *shape):
        return self

    def items(self):
        return [self.values[index] for index in self.indices]


@pytest.mark.parametrize("prefill,ends,expected", [(False, (), [0, 1, 2, 3, 4]),
                                                  (True, (), [4]), (True, (1,), [1]), (True, (0, 3), [0, 3])])
def test_finish_scores_only_paired_selected_streams_and_pss(prefill, ends, expected):
    calls = []
    cfg = NS(eps=1e-6, streams=4, low=320)
    b = NS(prefill=prefill, h=Rows(list(range(5))), streams=Rows([-1] * 5), pss=Rows(list(range(5))),
           mixed=Rows([-1] * 5), xs_mixed=Rows([-1] * 5), logits=Rows([-1] * 5))

    def readout(mixer, buffers, rows, count, *_):
        calls.append(("readout", rows.items(), buffers.pss[:count].items()))

    def mm(rows, head, sums, out, buffers):
        calls.append(("head", len(rows.indices)))
        return out

    scope = maintained("forward.py", {"finish", "_final_readout"}, {
        "torch": NS(tensor=lambda values, **_: values, long="i64"),
        "_writeback": lambda *args: calls.append(("writeback", args[2])), "_readout": readout, "_mm": mm,
    })
    output = scope["finish"](NS(cfg=cfg, head=object(), comm=None), object(), b, 5, object(), ends=ends)
    assert len(output.indices) == len(expected)
    assert calls == [("writeback", 5), ("readout", expected, expected), ("head", len(expected))]
    assert b.streams.items() == list(range(5))


def test_stream_only_finish_returns_before_any_readout_resource_access():
    calls = []
    b = NS(h=Rows([4, 9, 2]), streams=Rows([-1] * 3))
    # No prefill, pss, mixed, head or logits exist: omitted readout must not touch them.
    scope = maintained("forward.py", {"finish"}, {
        "_writeback": lambda *args: calls.append(args),
    })
    cfg, pending = object(), object()
    assert scope["finish"](NS(cfg=cfg), object(), b, 3, pending, logits=False) is None
    assert b.streams.items() == [4, 9, 2]
    assert calls == [(b.streams, b, 3, cfg, pending)]


@pytest.mark.parametrize("logits,last_only,check", [(False, True, False), (False, False, True),
                                                  (True, True, False), (True, True, True), (True, False, False)])
def test_mtp_absorb_omits_mixer_but_scoring_retains_it_and_validation(logits, last_only, check):
    calls = []
    names = ("ids", "mtp_e", "mixed", "xs_mixed", "mtp_eo", "mtp_in", "mtp_hn", "mtp_xh",
             "mtp_hs", "h", "logits", "last")
    b = NS(**{name: Rows([0] * 16) for name in names}, prefill=False)
    c = NS(eps=1e-6, streams=4, hidden=32)
    m = NS(norm_e=object(), norm_h=object(), fc_e=object(), fc_h=object(), layer=object(), mixer=object())
    w = NS(cfg=c, mtp=m, comm=None, head=NS(n=1), draft_head=NS(n=1))
    segs = [(object(), 0, 3)]

    def mm(rows, weight, sums, out, buffers):
        calls.append(("mm", weight))
        return out

    scope = maintained("mtp.py", {"mtp_compute"}, {
        "_kv_begin": lambda *_: calls.append(("begin",)),
        "_kv_check": lambda *_, **__: calls.append(("check",)),
        "_embed": lambda *_: None,
        "_mm": mm,
        "glue": NS(rmsnorm=lambda *_, out, xs: (out, xs), add_streams=lambda *_: None),
        "layer_forward": lambda *_, **__: None,
        "finish": lambda *_, **kw: calls.append(("finish", kw["logits"])),
        "_final_readout": lambda *_: calls.append(("readout",)),
    })
    result = scope["mtp_compute"](w, segs, b, logits=logits, last_only=last_only, check=check)
    expected = [("begin",), ("mm", m.fc_e), ("mm", m.fc_h), ("finish", False)]
    if logits:
        expected += [("readout",), ("mm", w.draft_head if last_only else w.head)]
    if not logits or not last_only or check:
        expected += [("check",)]
    assert calls == expected
    assert (result is None) == (not logits)
