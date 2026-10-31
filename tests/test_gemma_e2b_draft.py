"""Stateful proposals against a stateless Speculators-derived Torch oracle.

Reference: vllm-project/speculators 07b4fcf5e8d45d7d45da4e00464a484a7a8fb8ae,
models/dflash{,2}/model_definitions.py and models/dflash/attention.py.
The oracle recomputes all context K/V and uses an explicit anchor-prefix mask;
it does not reuse the implementation's rings, scratch buffers or convolution.
"""
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.rejection import draw, proposal_row
from tensorfold.families.gemma4.cuda.e2b_draft import E2BChains, E2BDFlash2
from tensorfold.families.gemma4.e2b_dflash_metadata import E2BDraftConfig
from test_gemma_e2b_dflash2_metadata import config


@pytest.fixture
def head(tmp_path):
    raw = config()
    cfg = E2BDraftConfig.from_dict(raw)
    generator = torch.Generator().manual_seed(793)
    tensors = {n: (torch.randn(s, generator=generator) * 0.125).to(torch.bfloat16)
               for n, s in cfg.shapes().items()}
    for name, tensor in tensors.items():
        if "norm.weight" in name:
            tensor.copy_(1 + (tensor.float() * 0.25).to(torch.bfloat16))
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(raw))
    embedding = torch.randn((cfg.vocab, cfg.hidden), generator=generator).to(torch.bfloat16)
    result = E2BDFlash2(tmp_path, embedding, embedding, context_limit=32)
    result._test_generator = generator
    return result


def norm(x, weight, eps):
    layer = Qwen3RMSNorm(weight.numel(), eps=eps)
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    return layer(x)


def reference_conv(x, delta, base, branch, group_size):
    """Generic tap loop, as opposed to the implementation's two-tap expression."""
    groups = x.shape[-1] // group_size
    original = x.reshape(len(x), groups, group_size)
    output = torch.zeros_like(original)
    for tap in range(base.shape[1]):
        shifted = torch.cat((torch.zeros_like(original[:tap]), original[:-tap]), 0) if tap else original
        coefficient = base[branch, tap].reshape(1, groups, group_size) + delta[:, branch, tap, :, None]
        output = output + shifted * coefficient
    return output.reshape_as(x)


def reference_rope(x, positions, theta):
    half = x.shape[-1] // 2
    frequencies = theta ** -(torch.arange(half).float() * 2 / x.shape[-1])
    phase = positions.float()[:, None] * frequencies
    phase = torch.cat((phase, phase), -1)
    rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
    return x * phase.cos().to(x.dtype)[None] + rotated * phase.sin().to(x.dtype)[None]


def reference_forward(head, taps, anchor, length):
    """Recompute the complete prefix independently of append/snapshot/state."""
    cfg, w = head.cfg, head.weights
    prefix_length = len(taps)
    context = norm(F.linear(taps, w["fc.weight"]), w["hidden_norm.weight"], cfg.eps)
    tokens = torch.tensor([anchor] + [cfg.mask_id] * (length - 1))
    noise = F.embedding(tokens, head.embedding_weight)
    positions = torch.arange(prefix_length + length)
    for i, kind in enumerate(cfg.layer_types):
        base = f"layers.{i}."
        hidden = norm(noise, w[base + "input_layernorm.weight"], cfg.eps)
        delta = F.linear(hidden, w[base + "attention_conv.kernel_projection.weight"]).reshape(
            length, 2, 2, cfg.hidden // cfg.group_size)
        prepared = reference_conv(hidden, delta, w[base + "attention_conv.base_kernel"], 0, cfg.group_size)
        combined = torch.cat((context, prepared))
        q = F.linear(prepared, w[base + "self_attn.q_proj.weight"]).reshape(length, cfg.heads, cfg.head_dim)
        k = F.linear(combined, w[base + "self_attn.k_proj.weight"]).reshape(
            prefix_length + length, cfg.kv_heads, cfg.head_dim)
        v = F.linear(combined, w[base + "self_attn.v_proj.weight"]).reshape(
            prefix_length + length, cfg.kv_heads, cfg.head_dim)
        q = reference_rope(norm(q, w[base + "self_attn.q_norm.weight"], cfg.eps).transpose(0, 1),
                           positions[-length:], cfg.theta)
        k = reference_rope(norm(k, w[base + "self_attn.k_norm.weight"], cfg.eps).transpose(0, 1),
                           positions, cfg.theta)
        allowed = torch.ones((length, prefix_length + length), dtype=torch.bool)
        if kind == "sliding_attention":
            allowed[:, :max(0, prefix_length - cfg.window)] = False
        attended = F.scaled_dot_product_attention(q[None], k[None], v.transpose(0, 1)[None],
                                                 attn_mask=allowed[None, None], enable_gqa=True)[0]
        attended = attended.transpose(0, 1).reshape(length, cfg.hidden)
        noise = noise + reference_conv(F.linear(attended, w[base + "self_attn.o_proj.weight"]),
                                       delta, w[base + "attention_conv.base_kernel"], 1, cfg.group_size)
        hidden = norm(noise, w[base + "post_attention_layernorm.weight"], cfg.eps)
        delta = F.linear(hidden, w[base + "mlp_conv.kernel_projection.weight"]).reshape(
            length, 2, 2, cfg.hidden // cfg.group_size)
        prepared = reference_conv(hidden, delta, w[base + "mlp_conv.base_kernel"], 0, cfg.group_size)
        mlp = F.linear(F.silu(F.linear(prepared, w[base + "mlp.gate_proj.weight"])) *
                       F.linear(prepared, w[base + "mlp.up_proj.weight"]), w[base + "mlp.down_proj.weight"])
        noise = noise + reference_conv(mlp, delta, w[base + "mlp_conv.base_kernel"], 1, cfg.group_size)
    hidden = norm(noise[1:], w["norm.weight"], cfg.eps)
    return F.linear(hidden, head.head_weight), hidden


@pytest.mark.parametrize("prefix", [1, 3, 4, 5, 9, 17])
def test_cached_draft_matches_stateless_bf16_reference(head, prefix):
    taps = torch.randn((prefix, 2 * head.cfg.hidden), generator=head._test_generator).to(torch.bfloat16)
    state = head.make_state()
    head.append(state, taps, prompt=True)
    anchor = 5
    packet = head.propose(state, anchor, 3)
    expected, hidden = reference_forward(head, taps, anchor, 4)
    reference = E2BChains(expected, hidden, head.weights, anchor, head.cfg.selector_top_k, None, False)
    packet._readback()
    reference._readback()
    # Same BF16 projection/norm/SDPA contracts: no empirically fitted tolerance.
    np.testing.assert_array_equal(packet.host[0], reference.host[0])
    np.testing.assert_array_equal(packet.host[1], reference.host[1])
    assert packet.finish(0, 3)[0] == reference.finish(0, 3)[0]


def test_proposals_do_not_commit_noise_and_snapshot_is_owned(head):
    state = head.make_state()
    taps = torch.randn((7, 32), generator=head._test_generator).to(torch.bfloat16)
    head.append(state, taps)
    held = state.snapshot()
    head.propose(state, 7, 3).finish(0, 3)
    assert state.length == 7
    after = state.snapshot()
    for (k, v), (ak, av) in zip(held[1], after[1]):
        assert torch.equal(k, ak) and torch.equal(v, av)
    head.append(state, taps[:2])
    state.restore(held)
    restored = state.snapshot()
    assert restored[0] == 7
    for (k, v), (rk, rv) in zip(held[1], restored[1]):
        assert torch.equal(k, rk) and torch.equal(v, rv)
    state.reset()
    assert state.length == 0
    assert all(k.shape[1] == 0 for k, _ in state.snapshot()[1])


def test_partial_append_and_cache_wrap_preserve_full_attention(head):
    taps = torch.randn((17, 32), generator=head._test_generator).to(torch.bfloat16)
    first, second = head.make_state(), head.make_state()
    head.append(first, taps)
    for chunk in taps.split(3):
        head.append(second, chunk, prompt=True)
    assert first.nbytes == head.cfg.state_bytes(32)
    for (k, v), (ck, cv) in zip(first.snapshot()[1], second.snapshot()[1]):
        assert torch.equal(k, ck) and torch.equal(v, cv)
    assert first.snapshot()[1][0][0].shape[1] == 4
    assert first.snapshot()[1][1][0].shape[1] == 17
    assert head.propose(first, 5, 3).finish(0, 3)[0] == head.propose(second, 5, 3).finish(0, 3)[0]


def test_sampled_packet_uses_actual_predecessor_conditional_q(head):
    state = head.make_state()
    head.append(state, torch.randn((5, 32), generator=head._test_generator).to(torch.bfloat16))
    sampling = Sampling(temperature=0.8, top_p=0.9, top_k=0, min_p=0.0, seed=73)
    packet = head.propose(state, 5, 3, sampled=True, sampling=sampling)
    packet._readback()
    sampled = packet.sample(0, 3, sampling, first_position=6)
    previous = 0
    for i, token in enumerate(sampled.tokens):
        ids, scores = packet.host
        q = proposal_row(scores[i, previous].astype(np.float64), ids[i], sampling)
        assert q == sampled.q_rows[i]
        assert token == draw(q, sampling.seed, 6 + i)
        previous = int(np.flatnonzero(ids[i] == token)[0])


def test_tail_context_bounds_and_bad_snapshot_preserve_state(head):
    state = head.make_state()
    taps = torch.randn((31, 32), generator=head._test_generator).to(torch.bfloat16)
    head.append(state, taps)
    assert head.propose(state, 5, 3) is None  # one remaining position is anchor only
    with pytest.raises(ValueError):
        head.append(state, taps[:2])
    held = state.snapshot()
    malformed = (held[0], (*held[1][:-1], (held[1][-1][0][..., :-1], held[1][-1][1])))
    with pytest.raises(ValueError):
        state.restore(malformed)
    assert state.length == 31
    for (k, v), (ak, av) in zip(held[1], state.snapshot()[1]):
        assert torch.equal(k, ak) and torch.equal(v, av)


def test_selected_context_bounds_counterfactual_sampled_positions(head):
    head.context_limit = 8  # checkpoint itself permits32; the admitted request pool is smaller
    state = head.make_state()
    head.append(state, torch.randn((6, 32), generator=head._test_generator).to(torch.bfloat16))
    sampling = Sampling(temperature=0.8, seed=73)
    packet = head.propose(state, 5, 3, sampled=True, sampling=sampling)
    assert packet.rows == 1  # anchor+one mask row fit, rather than the full four-row training block
    assert len(packet.sample(0, 3, sampling, first_position=7).tokens) == 1


@pytest.mark.parametrize("bits", [4, True, "8"])
def test_unsupported_proposal_precision_is_rejected_before_checkpoint_access(tmp_path, bits):
    weight = torch.zeros((16, 16), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="precision"):
        E2BDFlash2(tmp_path / "absent", weight, weight, 32, bits=bits)


def test_fp8_device_requirement_and_unchanged_bf16_default(head, tmp_path):
    assert head.bits == 16 and head.private_head is None and not head.linears
    assert head.resident_bytes == head.cfg.parameter_bytes
    # The fixture checkpoint is valid; refusal occurs before any FP8 provider is built.
    with pytest.raises(ValueError, match="require CUDA"):
        E2BDFlash2(tmp_path, head.embedding_weight, head.head_weight, 32, bits=8)


def selector_reference(logits, hidden, weights, anchor, count):
    """Full vocabulary lexsort and scalar edge walk, independent of packet guards."""
    raw = logits.float().numpy()
    ids = np.stack([np.lexsort((np.arange(raw.shape[1]), -row))[:count] for row in raw])
    projected = F.linear(hidden, weights["candidate_selector.hidden_projection.weight"])
    dtype = torch.promote_types(logits.dtype, projected.dtype)
    scores = torch.empty((len(hidden), count, count), dtype=dtype)
    for row in range(len(hidden)):
        for previous in range(count):
            predecessor = anchor if row == 0 else int(ids[row - 1, previous])
            context = weights["candidate_selector.predecessor_codebook"][predecessor] * projected[row]
            for current in range(count):
                candidate = int(ids[row, current])
                edge = (context * weights["candidate_selector.successor_codebook"][candidate]).sum()
                scores[row, previous, current] = logits[row, candidate] + edge
    return ids, scores.float().numpy()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_unary_boundary_plateaus_have_canonical_full_vocabulary_membership(dtype):
    vocab, width, rank, rows, kept = 80, 4, 4, 3, 4
    hidden = (torch.arange(rows * width).reshape(rows, width) % 5 / 8).to(torch.bfloat16)
    weights = {
        "candidate_selector.hidden_projection.weight":
            (torch.arange(rank * width).reshape(rank, width) % 3 / 8).to(torch.bfloat16),
        "candidate_selector.predecessor_codebook":
            (torch.arange(vocab * rank).reshape(vocab, rank) % 7 / 8).to(torch.bfloat16),
        "candidate_selector.successor_codebook":
            (torch.arange(vocab * rank).reshape(vocab, rank) % 5 / 8).to(torch.bfloat16),
    }
    logits = torch.zeros((rows, vocab), dtype=dtype)
    logits[1, :61] = 7
    logits[1, 70:73] = torch.tensor([10, 9, 8], dtype=dtype)
    logits[2] = 1
    logits[2, 79] = 4
    expected_ids, expected_scores = selector_reference(logits, hidden, weights, 5, kept)
    packet = E2BChains(logits, hidden, weights, 5, kept, None, False)
    packet._readback()
    np.testing.assert_array_equal(packet.host[0], expected_ids)
    # Repaired predecessors must also repair every dependent next-row edge.
    np.testing.assert_array_equal(packet.host[1], expected_scores)
    np.testing.assert_array_equal(packet.host[0][0], [0, 1, 2, 3])
    np.testing.assert_array_equal(packet.host[0][1], [70, 71, 72, 0])
    np.testing.assert_array_equal(packet.host[0][2], [79, 0, 1, 2])
    assert packet._source is None  # complete logits/codebooks retired after synchronous repair


def test_all_tied_unary_and_transition_scores_use_exact_token_id_order():
    logits = torch.zeros((3, 80), dtype=torch.bfloat16)
    hidden = torch.zeros((3, 4), dtype=torch.bfloat16)
    weights = {
        "candidate_selector.hidden_projection.weight": torch.zeros((4, 4), dtype=torch.bfloat16),
        "candidate_selector.predecessor_codebook": torch.zeros((80, 4), dtype=torch.bfloat16),
        "candidate_selector.successor_codebook": torch.zeros((80, 4), dtype=torch.bfloat16),
    }
    packet = E2BChains(logits, hidden, weights, 7, 4, None, False)
    assert packet.finish(0, 3)[0] == [0, 0, 0]
    np.testing.assert_array_equal(packet.host[0], np.broadcast_to(np.arange(4), (3, 4)))
    np.testing.assert_array_equal(packet.host[1], np.zeros((3, 4, 4), dtype=np.float32))
    sampling = Sampling(temperature=1.0, top_k=2, seed=19)
    sampled_packet = E2BChains(logits, hidden, weights, 7, 4, None, True)
    sampled = sampled_packet.sample(0, 3, sampling, 11)
    expected = proposal_row(np.zeros(4), np.arange(4), sampling)
    assert all(row == expected for row in sampled.q_rows)


def test_strict_unary_edge_keeps_normal_single_readback_support():
    logits = torch.arange(80, dtype=torch.float32)[None].repeat(3, 1)
    hidden = torch.zeros((3, 4), dtype=torch.bfloat16)
    weights = {
        "candidate_selector.hidden_projection.weight": torch.zeros((4, 4), dtype=torch.bfloat16),
        "candidate_selector.predecessor_codebook": torch.zeros((80, 4), dtype=torch.bfloat16),
        "candidate_selector.successor_codebook": torch.zeros((80, 4), dtype=torch.bfloat16),
    }
    packet = E2BChains(logits, hidden, weights, 7, 4, None, False)
    packet._readback()
    assert not packet.repaired_support
    np.testing.assert_array_equal(packet.host[0], np.broadcast_to([79, 78, 77, 76], (3, 4)))
    assert packet.finish(0, 3)[0] == [79, 79, 79]
