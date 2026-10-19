"""Qwen3.8 dense static YaRN on CUDA: the rotary amplitude scales q/k rotation as Transformers does, default RoPE keeps
its bits, a YaRN Config carries the policy's frequencies, and every forward path rotates with its amplitude."""

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


def _yarn_model(policy):
    """test_qwen27_prefill's two-layer model (DeltaNet, then attention) under the policy's frequencies and amplitude."""

    from tensorfold.families.qwen3_5.cuda.qmm_fast import prepare
    from tensorfold.families.qwen3_5.cuda.weights import GDN, Attention, Config, Layer, QLinear, Weights

    gen = torch.Generator(device="cuda").manual_seed(21)

    def qlinear(n, k):
        words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen,
                              device="cuda", dtype=torch.int64).to(torch.int32)
        scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 + 0.001).bfloat16()
        biases = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 - 0.0015).bfloat16()
        return QLinear(words, scales, biases)

    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    gdn = GDN(qlinear(384, 128), qlinear(128, 128), qlinear(1, 128), qlinear(1, 128),
              qlinear(128, 128), torch.randn(384, 4, generator=gen, device="cuda").bfloat16() * 0.1,
              torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda"), norm)
    attn = Attention(qlinear(2 * 2 * 128, 128), qlinear(128, 128), qlinear(128, 128), qlinear(128, 2 * 128),
                     norm, norm)
    layers = [Layer(True, norm, norm, gdn, None, qlinear(128, 128), qlinear(128, 128), qlinear(128, 128)),
              Layer(False, norm, norm, None, attn, qlinear(128, 128), qlinear(128, 128), qlinear(128, 128))]
    config = Config(hidden=128, intermediate=128, layers=2, heads=2, kv_heads=1, head_dim=128, vocab=256,
                    k_heads=1, v_heads=1, dk=128, dv=128, conv_kernel=4, interval=2, eps=1e-6,
                    rope_dims=policy.rotary_dim, rope_theta=policy.theta, eos=(0,), yarn=policy,
                    rope_scale=float(policy.attention_factor))
    w = Weights(config, qlinear(256, 128), layers, norm, qlinear(256, 128), policy.inverse_frequencies(torch).cuda())
    prepare(w)
    return w


def test_every_forward_path_rotates_with_the_yarn_amplitude(monkeypatch):
    """Prompt chunks, batched prompt rows, decode/verify trees and concurrent windows each pass the Config's amplitude
    to the rotation (whose arithmetic under it is checked above), and their row invariance holds under YaRN."""

    from tensorfold.families.qwen3_5.cuda.forward import State, multi_tree_forward, tree_forward
    from tensorfold.families.qwen3_5.cuda.multi import private
    from tensorfold.families.qwen3_5.cuda.prefill import prefill_chunk, prefill_rows

    policy = RopeParameters.from_config({"text_config": {
        "head_dim": 128, "max_position_embeddings": 262144,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25}}}, 4.0)
    w = _yarn_model(policy)
    seen, rotate = [], glue.attn_prep

    def recorded(*args, **kwargs):
        seen.append(kwargs.get("amplitude"))
        return rotate(*args, **kwargs)

    def amplitudes():
        out = set(seen)
        seen.clear()
        return out

    monkeypatch.setattr(glue, "attn_prep", recorded)
    ids = torch.randint(1, 256, (23,), generator=torch.Generator().manual_seed(5)).to(torch.int32).cuda()
    prompts = [ids, ids[:9]]
    alone = []
    for prompt in prompts:
        st = State(w)
        alone.append((prefill_chunk(w, prompt, st)[0], st))
    assert amplitudes() == {policy.attention_factor}
    together = prefill_rows(w, [(prompt.tolist(), State(w), 0) for prompt in prompts])
    assert amplitudes() == {policy.attention_factor}
    for (h, _), (row, _, _) in zip(alone, together):
        assert torch.equal(h.reshape(-1), row.reshape(-1))
    tokens, parents = [7, 8, 9, 10], [-1, 0, 0, 1]
    singles = [tree_forward(w, torch.tensor(tokens, dtype=torch.int32, device="cuda"), parents,
                            private(st, st.pos + 16))[0] for _, st in alone]
    assert amplitudes() == {policy.attention_factor}
    logits, _, _, starts = multi_tree_forward(w, [(tokens, parents, private(st, st.pos + 16)) for _, st in alone])
    assert amplitudes() == {policy.attention_factor}
    for s, single in enumerate(singles):
        assert torch.equal(logits[starts[s]:starts[s + 1]], single), s
