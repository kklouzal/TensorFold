"""Concurrent Flash Next keeps every stream slot: replacing a kept prompt never loses the displaced state."""

import importlib
from types import SimpleNamespace

import pytest

from tests.test_flashnext_prefix_copy import slot as state_slot

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

pytestmark = pytest.mark.torch


def decoder(module, free, kept, keep=8):
    dec = module.MultiDecoder.__new__(module.MultiDecoder)
    dec.streams, dec.free, dec.kept, dec.keep = {}, list(free), list(kept), keep
    dec.filling, dec.fills = [], {}
    return dec


def test_the_same_prompt_twice_keeps_both_slots(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    old, fresh = [state_slot("bf16", 1024, 0) for _ in range(2)]
    prompt = [1, 2, 3]
    dec = decoder(multi, [fresh], [(prompt, old, {}, None)])
    chosen, resume, cached = dec._slot_for(prompt, True)          # equal, not a strict prefix: a fresh prefill
    assert chosen is fresh and resume is None and cached == 0
    dec._remember(prompt, chosen, {"pos": len(prompt), "mtp_len": 0, **chosen.kv_snapshot_metadata()}, None)
    dec.streams = {0: SimpleNamespace(st=chosen)}
    other, _, _ = dec._slot_for([9, 9], True)                      # the second slot is still there
    assert other is old


def test_a_displaced_state_shared_or_busy_stays_out_of_the_free_list(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    old, fresh, busy = [state_slot("bf16", 1024, 0) for _ in range(3)]
    dec = decoder(multi, [], [([1], old, {}, None), ([2], old, {}, None), ([3], busy, {}, None)], keep=8)
    dec.streams = {0: SimpleNamespace(st=busy)}
    dec._remember([1], fresh, {"pos": 1, "mtp_len": 0, **fresh.kv_snapshot_metadata()}, None)                            # old still backs [2]: not free
    assert old not in dec.free
    dec._remember([3], fresh, {"pos": 1, "mtp_len": 0, **fresh.kv_snapshot_metadata()}, None)                            # busy is a live stream's: not free
    assert busy not in dec.free


def test_invalid_snapshot_rejects_before_displacing_a_kept_slot(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    old, fresh = [state_slot("bf16", 1024, 0) for _ in range(2)]
    dec = decoder(multi, [], [([1], old, {}, None)])
    kept = list(dec.kept)
    snapshot = {"pos": 1, "mtp_len": 0, **fresh.kv_snapshot_metadata()}
    snapshot["kv_status"] = (False, -2)
    with pytest.raises(ValueError, match="validated status"):
        dec._remember([1], fresh, snapshot, None)
    assert dec.kept == kept and dec.free == []
