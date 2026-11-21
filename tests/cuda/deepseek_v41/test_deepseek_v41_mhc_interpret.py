"""CPU Triton-interpreter coverage of the vendored mHC, router, C2 and Engram kernels.

Runs only with TRITON_INTERPRET=1 (CPU tensors reach the Triton kernels through ``on_triton_device``). Oracles and
tolerances are the ported upstream tests' (test_deepseek_v41_mhc/_compression/_engram); CUDA-graph replays are
CUDA-only and stay in those files. hc_transition has no upstream oracle test: its two production paths (V3 for
1..8 tokens, windowed per-token kernels otherwise) are documented as the same per-token arithmetic, so they are
compared bitwise, and both against the unfused expand/mixes/collapse composition at the mHC tolerances.
"""

import copy
import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET", "0") != "1" or torch.cuda.is_available(),
    reason="CPU Triton-interpreter test (TRITON_INTERPRET=1, no CUDA)",
)

if os.environ.get("TRITON_INTERPRET", "0") == "1":
    from test_deepseek_v41_compression import _inputs, _Reference
    from test_deepseek_v41_engram import _auxiliaries
    from v41_engram_oracle import EngramHashState

    from tensorfold.families.deepseek_v4.cuda.kernels import mhc_fused
    from tensorfold.families.deepseek_v4.cuda.kernels.compression import V41C2State
    from tensorfold.families.deepseek_v4.cuda.kernels.engram import V41EngramHashState, apply_engram_gate
    from tensorfold.families.deepseek_v4.cuda.kernels.mhc import (
        collapse_hc,
        expand_hc,
        fp32_projection,
        hc_mixes,
        route_experts,
    )
    from tensorfold.families.deepseek_v4.cuda.kernels.router import route_topk


@pytest.mark.parametrize("tokens", [1, 3])
def test_mhc_mixes_collapse_expand_on_interpreter(tokens):
    # Oracles/tolerances of test_v41_mhc_coefficients_delayed_collapse_and_post.
    generator = torch.Generator().manual_seed(41)
    residual = torch.randn(tokens, 4, 5120, generator=generator).half()
    fn = torch.randn(24, 20480, generator=generator) * 0.005
    scales = torch.tensor([0.3, 0.2, 0.1])
    base = torch.randn(24, generator=generator) * 0.4
    norm = torch.rand(5120, generator=generator) + 0.5
    x = residual.flatten(1).float()
    mixes = F.linear(x, fn) * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20)
    expected_pre = torch.sigmoid(mixes[:, :4] * scales[0] + base[:4]) + 1e-6
    expected_post = 2 * torch.sigmoid(mixes[:, 4:8] * scales[1] + base[4:8])
    expected_comb = (mixes[:, 8:] * scales[2] + base[8:]).view(-1, 4, 4).softmax(-1) + 1e-6
    expected_comb /= expected_comb.sum(-2, keepdim=True) + 1e-6
    for _ in range(19):
        expected_comb /= expected_comb.sum(-1, keepdim=True) + 1e-6
        expected_comb /= expected_comb.sum(-2, keepdim=True) + 1e-6
    pre, post, comb = hc_mixes(residual, fn, scales, base)
    for actual, expected in ((pre, expected_pre), (post, expected_post), (comb, expected_comb)):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    previous = torch.zeros(tokens, 4)
    previous[:, 0] = 1
    torch.testing.assert_close(collapse_hc(residual, previous), residual[:, 0], rtol=0, atol=0)
    normalized = collapse_hc(residual, previous, norm)
    f = residual[:, 0].float()
    expected = (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-20) * norm).half()
    torch.testing.assert_close(normalized, expected, rtol=0.001, atol=2e-6)
    output = expand_hc(normalized, residual, post, comb)
    expected = (
        torch.einsum("tij,tih->tjh", comb, residual.float()) + post[..., None] * normalized[:, None].float()
    ).half()
    torch.testing.assert_close(output, expected, rtol=0.002, atol=3e-6)
    inplace = residual.clone()
    assert expand_hc(normalized, inplace, post, comb, inplace=True) is inplace
    torch.testing.assert_close(inplace, output, rtol=0, atol=0)


@pytest.mark.parametrize("normalized,n,k", [(False, 384, 5120), (True, 24, 20480)])
def test_fp32_projection_and_routes_batch_invariant_on_interpreter(normalized, n, k):
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(6, k, generator=generator).half()
    weight = torch.randn(n, k, generator=generator) * 0.005
    actual = fp32_projection(x, weight, normalized=normalized)
    separate = torch.cat([fp32_projection(x[i : i + 1], weight, normalized=normalized) for i in range(6)])
    torch.testing.assert_close(actual, separate, rtol=0, atol=0)
    expected = F.linear(x.float(), weight)
    if normalized:
        expected *= torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    if not normalized:
        bias = torch.randn(n, generator=generator) * 0.01
        probability, ids = route_experts(x, weight, bias)
        rows = [route_experts(x[i : i + 1], weight, bias) for i in range(6)]
        torch.testing.assert_close(ids, torch.cat([r[1] for r in rows]), rtol=0, atol=0)
        torch.testing.assert_close(probability, torch.cat([r[0] for r in rows]), rtol=0, atol=0)


def test_route_topk_matches_unfused_route_on_interpreter():
    # route_experts' unfused epilogue (torch softplus/sqrt/topk/renormalize) is the reference.
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(5, 384, generator=generator) * 4
    logits[0, :3] = torch.tensor([25.0, 30.0, -30.0])  # both softplus threshold branches
    bias = torch.randn(384, generator=generator) * 0.01
    probs, ids = route_topk(logits, bias)
    scores = F.softplus(logits).sqrt()
    expected_ids = (scores + bias).topk(6, dim=-1).indices
    expected = scores.gather(1, expected_ids)
    expected = expected / (expected.sum(-1, keepdim=True) + 1e-20) * 1.5
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(probs, expected, rtol=2e-6, atol=2e-7)


def _transition_inputs(tokens, mix, seed=0):
    generator = torch.Generator().manual_seed(seed)
    h = mhc_fused.H
    values = {
        "residual": torch.randn(tokens, 4, h, generator=generator),
        "y": torch.randn(tokens, h, generator=generator).half(),
        "post": torch.rand(tokens, 4, generator=generator) * 2,
        "comb": torch.rand(tokens, 4, 4, generator=generator),
        "collapse_pre": torch.rand(tokens, 4, generator=generator),
        "norm_weight": torch.rand(h, generator=generator),
    }
    if mix:
        values.update(
            fn=torch.randn(24, 4 * h, generator=generator) * 0.01,
            scale=torch.rand(3, generator=generator),
            base=torch.randn(24, generator=generator),
        )
    return values


def _windowed(v, window):
    previous = mhc_fused._WINDOW
    mhc_fused._WINDOW = window
    try:
        # A 9-token call cannot select V3; this is the >8-token production path.
        return mhc_fused.hc_transition(**v)
    finally:
        mhc_fused._WINDOW = previous


@pytest.mark.parametrize("mix", [True, False])
def test_hc_transition_v3_equals_windowed_path_and_unfused_composition(mix):
    v = _transition_inputs(9, mix)
    windowed = dict(v, residual=v["residual"].clone())
    out_w, mixes_w = _windowed(windowed, 4)  # three windows, one partial
    full = dict(v, residual=v["residual"].clone())
    out_f, mixes_f = _windowed(full, 2048)
    torch.testing.assert_close(windowed["residual"], full["residual"], rtol=0, atol=0)
    torch.testing.assert_close(out_w, out_f, rtol=0, atol=0)
    for t in range(0, 9, 4):  # V3 on 1..4-token slices, same per-token arithmetic
        part = {k: (x[t : t + 4].clone() if k not in ("norm_weight", "fn", "scale", "base") else x)
                for k, x in v.items()}
        out_3, mixes_3 = mhc_fused.hc_transition(**part)
        torch.testing.assert_close(part["residual"], full["residual"][t : t + 4], rtol=0, atol=0)
        torch.testing.assert_close(out_3, out_f[t : t + 4], rtol=0, atol=0)
        if mix:
            for a, b in zip(mixes_3, mixes_f):
                torch.testing.assert_close(a, b[t : t + 4], rtol=0, atol=0)
    if mix:
        for a, b in zip(mixes_w, mixes_f):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        assert mixes_w is None and mixes_f is None
    # Unfused composition (FP32 residual): expand, then delayed collapse+norm, then next mixes.
    expanded = expand_hc(v["y"], v["residual"].clone(), v["post"], v["comb"])
    torch.testing.assert_close(full["residual"], expanded, rtol=2e-6, atol=2e-6)
    collapsed = collapse_hc(expanded, v["collapse_pre"], v["norm_weight"])
    torch.testing.assert_close(out_f, collapsed, rtol=0.001, atol=2e-6)
    if mix:
        for a, b in zip(mixes_f, hc_mixes(expanded[:6], v["fn"], v["scale"], v["base"])):
            torch.testing.assert_close(a[:6], b, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("length,chunk", [(64, 64), (257, 63)])
def test_c2_pooling_and_commit_on_interpreter(length, chunk):
    norm = torch.linspace(0.5, 1.5, 512)
    cpu, state = _Reference(norm), V41C2State(norm)
    kv, score = _inputs(length, 41)
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        expected, mask = cpu.feed(kv[start:end], score[start:end], start)
        actual, complete = state(kv[start:end], score[start:end], torch.arange(start, end))
        torch.testing.assert_close(complete, mask, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0.001, atol=2e-6)
        state.commit(torch.tensor([end - start], dtype=torch.int32))
        torch.testing.assert_close(state.partial, cpu.partial, rtol=0, atol=0)
        assert state.position.item() == end
    # Speculative prefix commit (both parities), then a replacement input.
    for accepted in (1, 4):
        reference, draft = copy.deepcopy(cpu), copy.deepcopy(state)
        tail_kv, tail_score = _inputs(6, 90 + accepted)
        actual, complete = draft(tail_kv, tail_score, torch.arange(length, length + 6))
        expected, mask = copy.deepcopy(reference).feed(tail_kv, tail_score, length)
        torch.testing.assert_close(complete, mask, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0.001, atol=2e-6)
        draft.commit(torch.tensor([accepted], dtype=torch.int32))
        reference.feed(tail_kv[:accepted], tail_score[:accepted], length)
        torch.testing.assert_close(draft.partial, reference.partial, rtol=0, atol=0)


def test_c2_device_checks_fail_on_interpreter():
    state = V41C2State(torch.linspace(0.5, 1.5, 512))
    kv, score = _inputs(4, 3)
    with pytest.raises(Exception, match="device check failed"):
        state(kv, score, torch.tensor([0, 1, 3, 4]))  # non-contiguous position chain
    state = V41C2State(torch.linspace(0.5, 1.5, 512))
    state(kv, score, torch.arange(4))
    with pytest.raises(Exception, match="device check failed"):
        state.commit(torch.tensor([5], dtype=torch.int32))  # more than the pending length


@pytest.mark.parametrize("length,chunk", [(64, 64), (257, 63)])
def test_engram_hash_and_commit_on_interpreter(length, chunk):
    values = _auxiliaries()
    cpu = EngramHashState(*values, pad_token_id=2)
    state = V41EngramHashState(*values, max_tokens=2048)
    generator = torch.Generator().manual_seed(41)
    tokens = torch.randint(129280, (1, length), generator=generator)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, 61:64] = False
    mask[:, -2] = False
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        expected = cpu(tokens[:, start:end], start, mask[:, start:end])
        actual = state(tokens[0, start:end], torch.arange(start, end), mask[0, start:end])
        torch.testing.assert_close(actual, expected[0], rtol=0, atol=0)
        state.commit(torch.tensor([end - start], dtype=torch.int32))
        torch.testing.assert_close(state.history[:3], cpu.history[0], rtol=0, atol=0)
        assert state.history[3].item() == cpu.next_pos
    # Draft verification then partial acceptance, as test_v41_hash_commits_only_accepted_inputs.
    candidate = torch.arange(6) * 37 + 5
    before = copy.deepcopy(cpu)
    actual = state(candidate, torch.arange(length, length + 6))
    torch.testing.assert_close(actual, before(candidate[None], length)[0], rtol=0, atol=0)
    state.commit(torch.tensor([2], dtype=torch.int32))
    cpu(candidate[None, :2], length)
    torch.testing.assert_close(state.history[:3], cpu.history[0], rtol=0, atol=0)
    with pytest.raises(Exception, match="device check failed"):
        state(torch.tensor([129280]), torch.tensor([length + 2]))  # token outside the map


@pytest.mark.parametrize("inplace", [False, True])
def test_engram_gate_on_interpreter(inplace):
    tokens = 3
    generator = torch.Generator().manual_seed(41)
    residual = torch.randn(tokens, 4, 5120, generator=generator).half()
    projection = torch.randn(tokens, 25600, generator=generator).half()
    q_weight = torch.rand(4, 5120, generator=generator)
    k_weight = torch.rand(4, 5120, generator=generator)
    residual[0].zero_()
    projection[0, : 4 * 5120].zero_()
    mask = torch.ones(tokens, dtype=torch.bool)
    mask[1] = False
    h, key = residual.float(), projection[:, : 4 * 5120].float().view(tokens, 4, 5120)
    rstd = torch.rsqrt(h.square().mean(-1) + 1e-20) * torch.rsqrt(key.square().mean(-1) + 1e-20)
    dot = (h * (q_weight * k_weight) * key).sum(-1) * rstd * 5120**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    gate.masked_fill_(~mask[:, None], 0)
    expected = (h + gate[..., None] * projection[:, 4 * 5120 :].float()[:, None]).half()
    x = residual.clone()
    actual = apply_engram_gate(x, projection, q_weight, k_weight, mask, inplace=inplace)
    assert (actual.data_ptr() == x.data_ptr()) == inplace
    torch.testing.assert_close(actual, expected, rtol=0.001, atol=2e-6)
    torch.testing.assert_close(actual[1], residual[1], rtol=0, atol=0)
