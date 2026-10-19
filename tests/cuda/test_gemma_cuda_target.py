"""Actual-CUDA target identity and cache lifecycle checks on a tiny random target with the served attention geometry.

The attention kernels themselves are checked against a float64 oracle in tests/test_gemma_cuda_attention.py; here the
whole layer-major forward (affine projections, experts, norms, attention, persistent caches) must give identical bits
for any row partition, rollback and snapshot restore.
"""

from __future__ import annotations

import pytest
import torch

from tensorfold.families.gemma4.cuda.experts import ExpertLinear, Experts
from tensorfold.families.gemma4.cuda.forward import Gemma4Target, apply_rope, rms_norm
from tensorfold.families.gemma4.cuda.linear import Linear
from tensorfold.families.gemma4.cuda.state import Snapshot, capacity_bytes
from tensorfold.families.gemma4.cuda.weights import Config, Gemma4Weights, Layer


def config():
    # The served head geometry (16 q heads; sliding 8x256; full 2x512, 128 rotated columns) on a tiny body.
    return Config(hidden=128, layers=2, vocab=128, heads=16, kv_heads=8, global_kv_heads=2,
                  head_dim=256, global_head_dim=512, intermediate=128, moe_width=64, experts=4, top_k=2,
                  window=8, max_position=4096, eps=1e-6, softcap=30.,
                  layer_types=("sliding_attention", "full_attention"), rope_theta=10000.,
                  global_rope_theta=1000000., global_rotary_fraction=.25, vision_bidirectional=True)


def packed(shape, bits, generator):
    words = torch.randint(0, 2**32, (*shape[:-1], shape[-1] * bits // 32),
                          dtype=torch.int64, generator=generator).to(dtype=torch.uint32, device="cuda")
    scales = (torch.randn((*shape[:-1], shape[-1] // 64), generator=generator) * .0007).to("cuda", torch.bfloat16)
    bias = (torch.randn(scales.shape, generator=generator) * .003).to("cuda", torch.bfloat16)
    return (ExpertLinear if len(shape) == 3 else Linear)(words, scales, bias, bits=bits, group=64)


def target(seed=319):
    c = config()
    generator = torch.Generator().manual_seed(seed)

    def norm(width=c.hidden):
        return (torch.randn((width,), generator=generator) * .1 + 1).to("cuda", torch.bfloat16)

    layers = []
    for i in range(c.layers):
        hk, d = c.attention_geometry(i)

        def linear(n, k, bits=4):
            return packed((n, k), bits, generator)

        banks = [packed((c.experts, *shape), bits, generator) for shape, bits in (
            ((c.moe_width, c.hidden), 4), ((c.moe_width, c.hidden), 6), ((c.hidden, c.moe_width), 6))]
        layers.append(Layer(
            q=linear(c.heads * d, c.hidden), k=linear(hk * d, c.hidden),
            v=linear(hk * d, c.hidden, 6) if c.sliding(i) else None, o=linear(c.hidden, c.heads * d),
            gate=linear(c.intermediate, c.hidden), up=linear(c.intermediate, c.hidden),
            down=linear(c.hidden, c.intermediate, 6), router=linear(c.experts, c.hidden), experts=Experts(*banks),
            input_norm=norm(), attention_norm=norm(), mlp_norm=norm(), expert_norm=norm(),
            dense_output_norm=norm(), expert_output_norm=norm(), output_norm=norm(), q_norm=norm(d), k_norm=norm(d),
            router_scale=norm(), expert_scale=norm(c.experts),
            scalar=torch.tensor([.99], device="cuda", dtype=torch.bfloat16)))
    weights = Gemma4Weights(c, packed((c.vocab, c.hidden), 6, generator), norm(), layers, torch.device("cuda:0"), {})
    return Gemma4Target(weights)


@pytest.mark.parametrize("width", [64, 256, 512, 2816])
def test_norm_uses_direct_scale_and_bf16_boundary(width):
    torch.manual_seed(82)
    x = torch.randn((5, width), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((width,), device="cuda", dtype=torch.bfloat16)
    actual = rms_norm(x, w, 1e-6)
    expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * w
    # FP32 reduction trees differ between the independent Torch oracle and
    # Triton. Two BF16 storage boundaries bound the comparison to two ULPs.
    torch.testing.assert_close(actual, expected, atol=0, rtol=2**-6)
    serial = torch.cat([rms_norm(row[None], w, 1e-6) for row in x])
    assert torch.equal(actual, serial)


def test_proportional_rope_leaves_nonrotary_pairs_unchanged():
    model = target()
    inverse = model._inverse[1]
    assert torch.count_nonzero(inverse).item() == 64
    x = torch.randn((4, 2, 512), device="cuda", dtype=torch.bfloat16)
    actual = apply_rope(x, inverse, 31)
    assert torch.equal(actual[:, :, 64:256], x[:, :, 64:256])
    assert torch.equal(actual[:, :, 320:], x[:, :, 320:])
    assert not torch.equal(actual[:, :, :64], x[:, :, :64])


def test_layer_major_rows_match_serial_logits_taps_rollback_and_restore():
    model = target()
    a, b = model.make_state(128, rows=17), model.make_state(128, rows=17)
    model.forward(list(range(17)), a)
    model.forward_serial(list(range(17)), b)
    snapshot = a.snapshot()
    tokens = torch.tensor([29, 31, 7, 90, 54], device="cuda", dtype=torch.int64)
    batch = model.forward(tokens, a)
    serial = model.forward_serial(tokens, b)
    assert torch.equal(batch.hidden, serial.hidden)
    assert torch.equal(batch.logits, serial.logits)
    for layer in batch.taps:
        assert torch.equal(batch.taps[layer], serial.taps[layer])
    a.rollback(19)
    b.rollback(19)
    left = model.forward([93], a)
    right = model.forward([93], b)
    assert torch.equal(left.logits, right.logits)
    a.restore(snapshot)
    replay = model.forward(tokens, a)
    assert torch.equal(batch.logits, replay.logits)
    assert a.position == 22
    assert a.nbytes - a.pos.numel() * 4 == capacity_bytes(model.config, 128, 17)


def test_ring_wrap_prefill_chunks_and_bounded_rollback():
    model = target()
    tokens = [(7 * i + 3) % 128 for i in range(100)]
    chunked = model.make_state(256, rows=48)
    out = [model.forward(tokens[i:i + 48], chunked).logits for i in range(0, 96, 48)]
    out.append(model.forward(tokens[96:], chunked).logits)
    one = model.make_state(256, rows=4)
    serial = model.forward_serial(tokens, one).logits
    assert torch.equal(torch.cat(out), serial)
    assert chunked.ring == 64 and one.ring == 64                 # window 8 + rows, rounded: the ring wrapped
    model.forward([87, 88, 89, 90], one, output_logits=False)
    with pytest.raises(ValueError, match="retained"):
        one.rollback(20)
    assert one.position == 104


def test_staged_forward_matches_forward_and_admits_only_covering_grids():
    model = target()
    tokens = [(5 * i + 1) % 128 for i in range(600)]
    eager, staged = model.make_state(1100, rows=128), model.make_state(1100, rows=128)
    for state in (eager, staged):
        for i in range(0, 600, 120):
            model.forward(tokens[i:i + 120], state, output_logits=False, output_taps=False)
    step = torch.tensor([17, 4, 90], device="cuda", dtype=torch.int64)
    expected = model.forward(step, eager)
    staged.pos.fill_(staged.position)
    hidden, logits, taps = model.forward_staged(step, staged, chunks=3)     # a bucket wider than the 2 chunks used
    staged.advance(3)
    assert torch.equal(hidden, expected.hidden) and torch.equal(logits, expected.logits)
    assert all(torch.equal(taps[layer], expected.taps[layer]) for layer in expected.taps)
    for chunks in (1, 4):                       # short of the last row; beyond the admitted context
        with pytest.raises(ValueError, match="grid"):
            model.forward_staged(step, staged, chunks=chunks)
    with pytest.raises(ValueError, match="ring reserve"):
        model.forward_staged(torch.zeros(129, device="cuda", dtype=torch.int64), staged, chunks=3)
    assert not staged.poisoned and staged.position == 603


def test_bad_visual_split_does_not_modify_state():
    model = target()
    state = model.make_state(64)
    with pytest.raises(ValueError, match="complete"):
        model.forward([1, 2], state, vision_blocks=((0, 4),))
    assert state.position == 0 and not state.poisoned


def test_embedding_is_scaled_once_and_tied_head_is_softcapped_once():
    model = target()
    ids = torch.tensor([12, 57], device="cuda", dtype=torch.int64)
    raw = model.weights.embed.embedding(ids)
    assert torch.equal(model.raw_embed_tokens(ids), raw)
    scale = torch.tensor(128 ** 0.5, dtype=torch.bfloat16, device="cuda")
    assert torch.equal(model.embed_tokens(ids), raw * scale)
    h = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
    expected = torch.tanh(model.weights.embed(h) / 30) * 30
    assert torch.equal(model.project_logits(h), expected)


def test_interrupted_snapshot_publication_poisoned_until_complete_restore():
    model = target()
    state = model.make_state(64)
    model.forward([1, 2, 3], state, output_logits=False)
    saved = state.snapshot()
    keys, values, raw, inv = saved._layers
    broken = Snapshot(saved.position, saved.high_water, saved.context, saved.rows, saved.config,
                      (keys, values, raw[:, :2], inv))
    with pytest.raises(RuntimeError):
        state.restore(broken)                                    # fails after publication began
    assert state.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        model.forward([4], state)
    state.restore(saved)
    assert not state.poisoned and state.position == 3
    model.forward([4], state)
