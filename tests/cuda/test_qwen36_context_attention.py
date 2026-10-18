"""DFlash v1 block attention over preallocated drafter stores: exact-softmax bound and position-only invariance."""

from types import SimpleNamespace as NS

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5_moe.cuda import context_attention as ca

D, LAYERS = 128, 3
CH = ca.CHUNK


def _store(kv_heads, capacity, gen):
    k = torch.randn(LAYERS, kv_heads, capacity, D, generator=gen, device="cuda").bfloat16()
    return NS(k=k, v=torch.randn(k.shape, generator=gen, device="cuda").bfloat16())


def _reference(q, k, v, stores, lengths, layer, length, scale):
    """Each stream's block over [its committed rows | its own block rows], noncausal, in float64."""
    group, outs = q.shape[0] // k.shape[0], []
    for j, (st, s) in enumerate(zip(stores, lengths)):
        rows = slice(j * length, (j + 1) * length)
        kk = torch.cat((st.k[layer, :, :s], k[:, rows]), 1).double().repeat_interleave(group, 0)
        vv = torch.cat((st.v[layer, :, :s], v[:, rows]), 1).double().repeat_interleave(group, 0)
        att = (q[:, rows].double() @ kk.transpose(1, 2)) * scale
        outs.append((att.softmax(-1) @ vv).transpose(0, 1).reshape(length, -1))
    return torch.cat(outs)


def _inputs(heads, kv_heads, length, streams, gen):
    q = torch.randn(heads, streams * length, D, generator=gen, device="cuda").bfloat16()
    k = torch.randn(kv_heads, streams * length, D, generator=gen, device="cuda").bfloat16()
    v = torch.randn(kv_heads, streams * length, D, generator=gen, device="cuda").bfloat16()
    return q, k, v


def _run(q, k, v, stores, lengths, layer, length, kv_heads):
    launch = ca.Round(stores, lengths, heads=q.shape[0], kv_heads=kv_heads, dim=D, length=length, device=q.device)
    return ca.block_attention(q, k, v, launch, layer, D ** -0.5)


@pytest.mark.parametrize("heads,kv_heads,length", [(32, 4, 16), (32, 4, 8), (32, 4, 2), (32, 4, 128), (16, 2, 33)])
def test_matches_exact_softmax_across_chunk_boundaries(heads, kv_heads, length):
    gen = torch.Generator(device="cuda").manual_seed(heads + length)
    lengths = [1, 63, CH - 1, CH, CH + 1, 3 * CH + 17]
    stores = [_store(kv_heads, n + 5 + 64 * i, gen) for i, n in enumerate(lengths)]
    q, k, v = _inputs(heads, kv_heads, length, len(lengths), gen)
    for layer in (0, LAYERS - 1):
        out = _run(q, k, v, stores, lengths, layer, length, kv_heads)
        ref = _reference(q, k, v, stores, lengths, layer, length, D ** -0.5)
        # BF16 probabilities in the value product and BF16 output rounding: a few BF16 ulps of max|v| ~ 4.5.
        assert (out.double() - ref).abs().max().item() < 2e-2


def test_a_stream_gets_the_same_bits_alone_among_others_and_at_any_capacity():
    gen = torch.Generator(device="cuda").manual_seed(3)
    heads, kv_heads, length = 32, 4, 16
    lengths = [2 * CH + 5, 700, CH + 1]
    stores = [_store(kv_heads, n + 9, gen) for n in lengths]
    q, k, v = _inputs(heads, kv_heads, length, 3, gen)
    together = _run(q, k, v, stores, lengths, 1, length, kv_heads)
    for j in range(3):
        rows = slice(j * length, (j + 1) * length)
        wide = _store(kv_heads, lengths[j] + 4000, gen)          # same committed rows inside a larger store
        wide.k[:, :, :lengths[j]] = stores[j].k[:, :, :lengths[j]]
        wide.v[:, :, :lengths[j]] = stores[j].v[:, :, :lengths[j]]
        for store in (stores[j], wide):
            alone = _run(q[:, rows].contiguous(), k[:, rows].contiguous(), v[:, rows].contiguous(), [store],
                         lengths[j:j + 1], 1, length, kv_heads)
            assert torch.equal(together[rows], alone), j


def test_repeated_launches_are_deterministic():
    gen = torch.Generator(device="cuda").manual_seed(11)
    lengths = [5 * CH + 3]
    stores = [_store(4, lengths[0], gen)]
    q, k, v = _inputs(32, 4, 16, 1, gen)
    first = _run(q, k, v, stores, lengths, 2, 16, 4)
    for _ in range(3):
        assert torch.equal(_run(q, k, v, stores, lengths, 2, 16, 4), first)
