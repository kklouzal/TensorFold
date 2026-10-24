"""Admission, phase and lifetime boundaries for reference-prefill batching."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda import gguf_prefill


def make_plan(pairs=128, width=80, rows=2048):
    return gguf_prefill.PrefillPlan(width, 2560, rows, 11, pairs, 8 << 30)


@pytest.mark.parametrize("width,original", [(80, 16), (160, 8), (320, 4), (640, 2)])
def test_budget_selects_largest_batch_with_an_explicit_original_fallback(width, original):
    assert gguf_prefill.original_pairs(width, 2560) == original
    for pairs in (128, 64, 32, original):
        exact = make_plan(pairs, width).workspace_bytes
        selected = gguf_prefill.choose(width=width, dims=2560, rows=2048, slots=11, budget_bytes=exact)
        assert selected.pairs == pairs and selected.workspace_bytes == exact
        if pairs == original:
            with pytest.raises(ValueError, match="cannot fit"):
                gguf_prefill.choose(width=width, dims=2560, rows=2048, slots=11, budget_bytes=exact - 1)
        else:
            smaller = gguf_prefill.choose(width=width, dims=2560, rows=2048, slots=11, budget_bytes=exact - 1)
            assert smaller.pairs < pairs


def test_peak_charges_replacement_overlap_and_keeps_caller_buffers_separate():
    selected = make_plan()
    receipt = selected.receipt()
    assert receipt["decoded_replacement_bytes"] == 8 * 128 * 80 * 2560
    assert receipt["decoded_replacement_bytes"] > receipt["nominal_three_decoded_banks_bytes"]
    assert receipt["workspace_bytes"] == sum(receipt[key] for key in (
        "decoded_replacement_bytes", "tile_bytes", "shared_bytes", "blas_scratch_bytes"))
    assert not receipt["caller_output_included"]
    assert receipt["resident_bytes"] == receipt["host_bytes"] == 0
    with pytest.raises(FrozenInstanceError):
        selected.pairs = 32


def test_private_policy_is_off_by_default_and_uses_only_header_geometry(monkeypatch):
    assert gguf_prefill.WORKSPACE_BUDGET_BYTES is None
    assert gguf_prefill.plan({}, rows=2048, world=8) is None
    monkeypatch.setattr(gguf_prefill, "WORKSPACE_BUDGET_BYTES", 2 << 30)
    text = dict(hidden_size=2560, moe_intermediate_size=640, shared_expert_intermediate_size=640,
                num_experts_per_tok=10)
    for world in (1, 2, 4, 8):
        result = gguf_prefill.plan(text, rows=2048, world=world)
        assert result.width == 640 // world and result.slots == 11
        assert result.rows == 2048 and result.workspace_bytes <= 2 << 30
    for bad in (0, 3, True):
        with pytest.raises(ValueError, match="world"):
            gguf_prefill.plan(text, rows=2048, world=bad)
    with pytest.raises(ValueError, match="equal"):
        gguf_prefill.plan({**text, "shared_expert_intermediate_size": 1280}, rows=2048, world=8)


@pytest.mark.parametrize("field,value", [("width", 0), ("dims", True), ("rows", -1), ("slots", 0),
                                         ("pairs", 24), ("budget_bytes", 1)])
def test_invalid_plan_is_rejected_before_any_allocation(field, value):
    with pytest.raises(ValueError):
        replace(make_plan(), **{field: value})


@pytest.mark.torch
def test_admitted_plan_changes_only_prompt_batch_and_is_bound_once(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts

    bank = GGUFExperts(None, None, None, (), count=512, width=80, dims=2560)
    calls = []
    monkeypatch.setattr(bank, "forward_reference", lambda *args, **kwargs: calls.append((args, kwargs)))
    selected = make_plan()
    bank.prepare_prefill(selected)
    assert bank.pairs == 16
    x, picks, out = SimpleNamespace(shape=(1, 2560)), SimpleNamespace(shape=(1, 11)), object()
    bank.forward_prefill(x, picks, out)
    assert calls.pop() == ((x, picks, out), {})
    bank.forward(x, picks, out)
    assert calls.pop() == ((x, picks, out), {})
    with pytest.raises(ValueError, match="already prepared"):
        bank.prepare_prefill(selected)
    with pytest.raises(ValueError, match="rows exceed"):
        bank.forward_prefill(SimpleNamespace(shape=(2049, 2560)), picks, out)
    with pytest.raises(ValueError, match="slots exceed"):
        bank.forward_prefill(x, SimpleNamespace(shape=(1, 12)), out)
    other = GGUFExperts(None, None, None, (), count=512, width=160, dims=2560)
    with pytest.raises(ValueError, match="geometry"):
        other.prepare_prefill(selected)


@pytest.mark.torch
def test_exact_workspace_delta_is_admitted_without_changing_resident_or_host_bytes(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import gguf_admission, gguf_weights

    prototype = SimpleNamespace(cfg=SimpleNamespace(streams=4, hidden=2560, rotary_dim=64,
                                                    moe_width=80, top_k=10),
                                head=SimpleNamespace(n=31040), draft_head=None, draft_ids=None,
                                meta={"draft_host_id_bytes": 0})
    planner = SimpleNamespace(matrices=[], arrays=[], tables=[])
    monkeypatch.setattr(gguf_weights, "model_plan", lambda *args, **kwargs: (prototype, planner))
    baseline = gguf_weights.weight_byte_plan("unused")
    selected = make_plan()
    actual = gguf_weights.weight_byte_plan("unused", prefill_plan=selected)
    assert actual["workspace_bytes"] - baseline["workspace_bytes"] == selected.workspace_bytes - (32 << 20)
    for key in ("resident_bytes", "host_resident_bytes", "host_staging_bytes", "staging_bytes"):
        assert actual[key] == baseline[key]
    options = gguf_admission.capacity_options("unused", 0, 8, prefill_plan=selected)
    assert options["device_workspace_bytes"] == actual["workspace_bytes"]
    with pytest.raises(ValueError, match="geometry"):
        gguf_weights.weight_byte_plan("unused", prefill_plan=make_plan(width=160))
    with pytest.raises(ValueError, match="immutable"):
        gguf_weights.weight_byte_plan("unused", prefill_plan={})


@pytest.mark.torch
def test_loader_checks_the_plan_before_constructing_an_io_reader(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import gguf_weights

    prototype = SimpleNamespace(cfg=SimpleNamespace(moe_width=80, hidden=256, top_k=1))
    monkeypatch.setattr(gguf_weights, "model_plan", lambda *args, **kwargs: (prototype, None))

    def reached_reader():
        raise RuntimeError("valid plan reached the reader")

    monkeypatch.setattr(gguf_weights, "Reader", reached_reader)
    with pytest.raises(ValueError, match="geometry"):
        gguf_weights.load_gguf("unused", _prefill_plan=make_plan())
    valid = gguf_prefill.PrefillPlan(80, 256, 7, 2, 128, 128 << 20)
    with pytest.raises(RuntimeError, match="reached the reader"):
        gguf_weights.load_gguf("unused", _prefill_plan=valid)


@pytest.mark.parametrize("width", [80, 160])
@pytest.mark.parametrize("maximum", [32, 64, 128])
def test_live_slot_count_uses_only_needed_batches_under_the_frozen_maximum(width, maximum):
    selected = make_plan(maximum, width)
    original = gguf_prefill.original_pairs(width, 2560)
    receipt = selected.receipt()
    for rows in (0, 1, 2, 3, 6, 18, 33, 128, 512, 2048):
        expected = min(maximum, max(original, 1 << (max(1, rows * 11) - 1).bit_length()))
        assert gguf_prefill.call_pairs(selected, rows, 11) == expected
        assert selected.receipt() == receipt
    assert gguf_prefill.call_pairs(selected, 1, 11) == 16
    with pytest.raises(ValueError, match="rows exceed"):
        gguf_prefill.call_pairs(selected, 2049, 11)


@pytest.mark.torch
def test_each_prepared_call_validates_actual_geometry_and_bounds_exactly_once(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts

    bank = GGUFExperts(None, None, None, (), count=512, width=80, dims=2560)
    bank.prepare_prefill(make_plan())
    seen = []
    original = gguf_prefill.PrefillPlan.validate
    def record(self, **kwargs):
        seen.append(kwargs)
        return original(self, **kwargs)
    monkeypatch.setattr(gguf_prefill.PrefillPlan, "validate", record)
    calls = []
    monkeypatch.setattr(bank, "forward_reference", lambda *args, **kwargs: calls.append(kwargs))
    picks = SimpleNamespace(shape=(1, 11))
    bank.forward_prefill(SimpleNamespace(shape=(1, 2560)), picks)
    assert seen == [dict(width=80, dims=2560, rows=1, slots=11)]
    assert calls == [{}]
    seen.clear()
    bank.width = 160
    with pytest.raises(ValueError, match="geometry"):
        bank.forward_prefill(SimpleNamespace(shape=(1, 2560)), picks)
    assert len(seen) == 1 and len(calls) == 1
