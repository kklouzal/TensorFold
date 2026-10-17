"""Draft residual selection and prepared MoE dispatch, without a CUDA device."""

from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.cuda import moe  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Weights  # noqa: E402


def _weights(layers=64, **kwargs):
    return Weights(NS(), None, [None] * layers, None, None, **kwargs)


def test_dense_dflash2_default_and_moe_dflash_v1_metadata_taps():
    assert _weights().draft_taps == (5, 19, 33, 47, 61)
    target = _weights(40, draft_tap_layers=(1, 10, 19, 28, 37))
    assert target.draft_taps == (1, 10, 19, 28, 37)
    assert target.__dict__["draft_taps"] is target.draft_taps


@pytest.mark.parametrize("taps", [(), (1, 1), (3, 1), (-1, 2), (1, 40), (True, 2), (1.5, 2), ("1", 2)])
def test_invalid_taps_refused_before_target_execution(taps):
    with pytest.raises(ValueError, match="draft target taps"):
        _ = _weights(40, draft_tap_layers=taps).draft_taps


def test_binding_snapshots_selector_and_rejects_later_conflict():
    target = _weights(40)
    source = [1, 10, 19, 28, 37]
    selected = target.bind_draft_taps(source)
    source[0] = 0
    assert selected == (1, 10, 19, 28, 37)
    assert target.draft_taps is selected
    assert target.bind_draft_taps(tuple(selected)) is selected
    with pytest.raises(ValueError, match="already bound"):
        target.bind_draft_taps((0, 10, 19, 28, 37))
    assert target.draft_taps is selected
    assert target.draft_tap_layers is selected


def test_binding_cannot_replace_an_observed_default():
    target = _weights()
    selected = target.draft_taps
    with pytest.raises(ValueError, match="already bound"):
        target.bind_draft_taps((1, 10, 19, 28, 37))
    assert target.draft_taps is selected
    assert target.draft_tap_layers == (5, 19, 33, 47, 61)


def test_invalid_binding_does_not_publish_a_selector():
    target = _weights(40)
    with pytest.raises(ValueError, match="draft target taps"):
        target.bind_draft_taps((1, 10, 19, 28, 40))
    assert "draft_taps" not in target.__dict__
    assert target.draft_tap_layers == (5, 19, 33, 47, 61)
    assert target.bind_draft_taps((1, 10, 19, 28, 37)) == (1, 10, 19, 28, 37)


def test_swiglu_preserves_bf16_silu_rounding():
    # The separate SiLU rounding changes the result at these ordinary inputs.
    gate = torch.linspace(-8, 8, 1024, dtype=torch.float64).bfloat16()
    up = torch.linspace(0.25, 4, 1024, dtype=torch.float64).bfloat16()
    g = gate.double()
    silu = (g / (1 + torch.exp(-g))).bfloat16()
    expected = (silu.double() * up.double()).bfloat16()
    got = moe.swiglu_bf16(gate, up)
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))
    fused = ((g / (1 + torch.exp(-g))) * up.double()).bfloat16()
    assert not torch.equal(expected.view(torch.int16), fused.view(torch.int16))


def test_prepared_expert_operation_uses_existing_scratch_and_ordered_combine(monkeypatch):
    calls = []
    bank = NS(width=8, dims=4)
    router = torch.zeros(5, 4, dtype=torch.bfloat16)
    x = torch.ones(2, 4, dtype=torch.bfloat16)

    def operation(rows, router_rows, experts, buf, top_k, count):
        calls.append((rows, router_rows, experts, buf, top_k, count))
        buf.y[:2].fill_(2)
        buf.wts[:2].fill_(0.25)
        return buf

    def combine(y, weights):
        return (y.float() * weights[:, :, None]).sum(1).bfloat16()

    monkeypatch.setattr(moe, "_scratch", {})
    monkeypatch.setattr(moe.grouped, "Plan", lambda *a, **kw: None)
    monkeypatch.setattr(moe, "combine", combine)
    layer = moe.Routed(router, bank, 2, run=operation)
    assert moe.Routed(router, bank, 2).run is moe.moe
    assert torch.equal(moe.run(x, layer), torch.full_like(x, 1.5))
    assert torch.equal(moe.run(x, layer), torch.full_like(x, 1.5))
    assert calls[0][1] is router and calls[0][2] is bank
    assert calls[0][3] is calls[1][3]
    assert calls[0][4:] == (2, 4)
