"""The project-owned CUDA oracle consumes YaRN amplitude without changing default-RoPE arithmetic."""

from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen3_5.cuda import reference


def test_zero_position_scales_only_the_rotary_dimensions_and_preserves_default_bits():
    x = torch.arange(-16, 16, dtype=torch.float32).reshape(2, 2, 8)
    pos, inv = torch.zeros(2, dtype=torch.int64), torch.tensor([1.0, 0.125])
    assert torch.equal(reference._rope(x, pos, inv).view(torch.int32), x.view(torch.int32))
    want = x.clone()
    want[..., :4] *= 1.25                                    # position zero: rotation is identity before amplitude
    got = reference._rope(x, pos, inv, 1.25)
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))


def test_reference_attention_forwards_the_configured_amplitude_to_queries_and_keys(monkeypatch):
    cfg = NS(heads=1, kv_heads=1, head_dim=4, eps=1e-6, rope_scale=1.25)
    at = NS(q="q", k="k", v="v", q_norm=None, k_norm=None, o="o")
    state = NS(pos=0, kv=[(torch.zeros(0, 1, 4, dtype=torch.bfloat16),
                           torch.zeros(0, 1, 4, dtype=torch.bfloat16))])
    seen, rotate = [], reference._rope

    def recorded(x, pos, inv, amplitude=1.0):
        seen.append(amplitude)
        return rotate(x, pos, inv, amplitude)

    monkeypatch.setattr(reference, "_rope", recorded)
    monkeypatch.setattr(reference, "_linear", lambda x, projection: torch.ones(
        x.shape[0], 8 if projection == "q" else 4, dtype=torch.bfloat16))
    reference._attention(NS(attn=at), torch.ones(2, 4), state, 0, cfg, torch.tensor([1.0]))
    assert seen == [1.25, 1.25]
