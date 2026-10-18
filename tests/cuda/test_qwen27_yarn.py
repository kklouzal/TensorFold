"""Qwen3.8 dense static YaRN on CUDA: the rotary amplitude scales q/k rotation as Transformers does, default RoPE keeps
its bits, and a YaRN Config carries the policy's frequencies."""

from __future__ import annotations

import json
import math

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda import glue
from tensorfold.families.qwen4_exp.rope import RopeParameters


def _inputs(rows=7, heads=4, kv_heads=1, d=256, seed=3):
    g = torch.Generator().manual_seed(seed)
    qg = (torch.randn(rows, heads * 2 * d, generator=g)).to(torch.bfloat16).cuda()
    k = (torch.randn(rows, kv_heads * d, generator=g)).to(torch.bfloat16).cuda()
    qn = (1 + 0.1 * torch.randn(d, generator=g)).to(torch.bfloat16).cuda()
    kn = (1 + 0.1 * torch.randn(d, generator=g)).to(torch.bfloat16).cuda()
    pos = torch.tensor([0, 1, 5, 4095, 262143, 600000, 1048575][:rows], dtype=torch.int32).cuda()
    return qg, k, qn, kn, pos


def _reference(x, norm, pos, inv, amp, eps=1e-6):
    """fp32 rmsnorm (rounded to bf16, as the kernel stores normed rows) then rotate-half on the first 2 * len(inv)
    dims with cos/sin times the amplitude, Transformers' partial rotary embedding."""

    xf = x.float()
    xn = (xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + eps) * norm.float()).to(torch.bfloat16).float()
    half = inv.numel()
    ang = pos.float()[:, None, None] * inv[None, None, :]
    cos, sin = torch.cos(ang) * amp, torch.sin(ang) * amp
    a, b = xn[..., :half], xn[..., half:2 * half]
    out = xn.clone()
    out[..., :half] = a * cos - b * sin
    out[..., half:2 * half] = b * cos + a * sin
    return out


def test_unit_amplitude_keeps_default_rope_bits():
    qg, k, qn, kn, pos = _inputs()
    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64))
    kw = dict(heads=4, kv_heads=1, head_dim=256)
    default = glue.attn_prep(qg, k, qn, kn, pos, inv, 1e-6, **kw)
    unit = glue.attn_prep(qg, k, qn, kn, pos, inv, 1e-6, amplitude=1.0, **kw)
    assert all(torch.equal(a, b) for a, b in zip(default, unit))


def test_yarn_frequencies_and_amplitude_match_the_reference():
    raw = {"text_config": {"head_dim": 256, "max_position_embeddings": 262144,
                           "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                                               "partial_rotary_factor": 0.25, "mrope_interleaved": True}}}
    policy = RopeParameters.from_config(raw, 4.0)
    inv = policy.inverse_frequencies(torch).cuda()
    amp = policy.attention_factor
    assert amp == pytest.approx(1 + 0.1 * math.log(4.0))
    qg, k, qn, kn, pos = _inputs()
    q, key = glue.attn_prep(qg, k, qn, kn, pos, inv, 1e-6, heads=4, kv_heads=1, head_dim=256, amplitude=amp)
    want_q = _reference(qg.view(7, 4, 2, 256)[:, :, 0], qn, pos, inv, amp)
    want_k = _reference(k.view(7, 1, 256), kn, pos, inv, amp)
    for got, want in ((q, want_q), (key, want_k)):
        err = (got.float() - want).abs().max().item()
        assert err <= 2 ** -7 * want.abs().max().item(), err                 # bf16 rounding of the rotated rows
    assert not torch.equal(q, glue.attn_prep(qg, k, qn, kn, pos, inv, 1e-6, heads=4, kv_heads=1, head_dim=256)[0])


def test_yarn_config_carries_the_policy_frequencies(tmp_path):
    from tensorfold.families.qwen3_5.cuda.weights import Config, _rotary_frequencies

    text = {"model_type": "qwen3_5_text", "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 4,
            "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "max_position_embeddings": 262144, "eos_token_id": 248044,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                                "mrope_section": [11, 11, 10], "mrope_interleaved": True}}
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": text}))
    default = Config.read(tmp_path)
    assert default.yarn is None and default.rope_scale == 1.0
    policy = RopeParameters.from_config({"text_config": text}, 4.0)
    cfg = Config.read(tmp_path, rope=policy)
    assert cfg.yarn is policy and cfg.rope_scale == pytest.approx(policy.attention_factor) and cfg.rope_dims == 64
    assert torch.equal(_rotary_frequencies(cfg, "cuda").cpu(), policy.inverse_frequencies(torch))
    base = _rotary_frequencies(default, "cuda")
    assert torch.equal(base.cpu(), (1e7 ** (-torch.arange(0, 32, dtype=torch.float64) / 32)).to(torch.float32))
