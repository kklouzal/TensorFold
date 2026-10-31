# ruff: noqa: E402 - optional Torch must be checked before backend imports.
"""Configured output bounds constrain new draws while preserving owned windows."""
from types import SimpleNamespace as NS

import pytest
pytest.importorskip("torch")

from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry


@pytest.mark.parametrize("method", ["mtp", "dflash2"])
@pytest.mark.parametrize("capacity,proposals", [(1, 0), (2, 1), (4, 3)])
def test_legal_new_draw_width_fits_explicit_capacity(method, capacity, proposals):
    decoder = object.__new__(MultiDecoder)
    decoder.capacity, decoder.depth, decoder.buf, decoder.controller = 32, 3, NS(rows=4), None
    decoder.w, decoder.fills = NS(dflash2=object() if method == "dflash2" else None), {}
    decoder.speculation = MethodGeometry(method, 3)
    stream = Stream([1], 10, st=NS(pos=1), draft=True)
    stream.output_capacity = capacity
    assert decoder._proposal_count(stream, 8) == proposals
    stream.out, stream.drafts = [2], list(range(3, 3 + proposals))
    before = list(stream.drafts)
    assert decoder.output_need(stream) == proposals + 1
    stream.production_budget = proposals
    assert not decoder._runnable(stream) and stream.drafts == before
    stream.production_budget = proposals + 1
    assert decoder._runnable(stream) and stream.drafts == before


def test_dflash_two_remaining_outputs_reserves_both_before_any_proposal_draw():
    decoder = object.__new__(MultiDecoder)
    decoder.capacity, decoder.depth, decoder.buf, decoder.controller = 32, 3, NS(rows=4), None
    decoder.w, decoder.fills, decoder.speculation = NS(dflash2=object()), {}, MethodGeometry("dflash2", 3)
    stream = Stream([1], 3, st=NS(pos=1), draft=True, out=[2])
    stream.output_capacity = 4
    assert decoder.output_need(stream) == 2
