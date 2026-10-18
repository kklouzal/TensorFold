"""Bookkeeping contracts for the bounded canonical head comparison tool."""

import argparse
from collections import Counter
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "gemma4_cuda_head_bench.py"
_SPEC = importlib.util.spec_from_file_location("gemma_head_bench", _PATH)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def test_orders_balance_positions_and_reverse_each_pair():
    names = ("reference", "reuse2", "reuse4", "reuse8", "reuse16")
    orders = list(bench._orders(names, len(names)))
    for index in range(0, len(orders), 2):
        assert orders[index + 1] == list(reversed(orders[index]))
    for position in range(len(names)):
        assert Counter(order[position] for order in orders) == Counter({name: 2 for name in names})


@pytest.mark.parametrize("text", ["", "1", "3,3", "3,5,11,16", "3,wat", "128"])
def test_verify_width_configuration_is_bounded(text):
    with pytest.raises(argparse.ArgumentTypeError):
        bench._widths(text)
    assert bench._widths("3,5,11") == (3, 5, 11)
    assert bench._widths("16") == (16,)


def test_timed_wrapper_keeps_complete_ids_and_drains_both_ends(monkeypatch):
    class Engine:
        drains = 0
        def drain(self):
            self.drains += 1
        def generate(self, prompt, count, sampling, callback, *, draft, stop_eos):
            assert prompt == [5] and count == 3 and sampling is None
            assert draft is True and stop_eos is True
            callback([7])
            callback([8, 9])
            return {"decode_s": 1.5}
    ticks = iter((10., 12.))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks))
    engine = Engine()
    output, stats = bench._run(engine, [5], 3, None, True)
    assert output == [7, 8, 9] and engine.drains == 2
    assert stats["wall_s"] == 2. and stats["output_tokens"] == 3
    assert stats["decode_tps"] == 2 / 1.5


def test_draft_only_restricts_timing_without_changing_reference_capture_policy():
    assert bench._timed_modes(True, 0) == (True,)
    assert bench._timed_modes(True, 1) == (True,)
    assert bench._timed_modes(False, 0) == (False, True)
    assert bench._timed_modes(False, 1) == (True, False)
