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


def _reference(q, k, v, stores, lengths, layer, length, scale, window=None):
    """Each stream's block over [its committed rows | its own block rows], noncausal, in float64, and each output's
    error bound from the arithmetic the PTX ISA guarantees (bf16 mma: products exact in FP32, accumulation at least
    single precision in an unspecified order, so a sum of m terms errs by at most about m * 2**-24 of its absolute
    sum). BF16 softmax weights in the value product and the BF16 output (unit roundoff 2**-8): 2**-8 * (softmax @ |v|
    + |out|). FP32 terms, relative to softmax @ |v|: each logit (D products, the scale) errs by (D + 1) 2**-24 M,
    M = scale * max sum_d |q_d k_d| over the stream, which moves its weight and the normaliser (2 (D + 3) M with the
    max subtraction and exp argument); the row sum and the value accumulation over the stream's n keys and their
    online rescaling (n each, generously 4 n 2**-24 = n 2**-22 together); exp and the final division (128 2**-22
    covers them with slack)."""
    group, outs, bounds = q.shape[0] // k.shape[0], [], []
    for j, (st, s) in enumerate(zip(stores, lengths)):
        rows = slice(j * length, (j + 1) * length)
        lo = 0 if window is None else max(0, s - window) // CH * CH        # the window's first attended key
        kk = torch.cat((st.k[layer, :, lo:s], k[:, rows]), 1).double().repeat_interleave(group, 0)
        vv = torch.cat((st.v[layer, :, lo:s], v[:, rows]), 1).double().repeat_interleave(group, 0)
        qq = q[:, rows].double()
        att = ((qq @ kk.transpose(1, 2)) * scale).softmax(-1)
        out = (att @ vv).transpose(0, 1).reshape(length, -1)
        spread = (att @ vv.abs()).transpose(0, 1).reshape(length, -1)
        m = scale * (qq.abs() @ kk.abs().transpose(1, 2)).max().item()
        outs.append(out)
        bounds.append(2**-8 * (spread + out.abs()) + 2**-22 * (s - lo + length + 2 * (D + 3) * m + 128) * spread)
    return torch.cat(outs), torch.cat(bounds)


def _inputs(heads, kv_heads, length, streams, gen):
    q = torch.randn(heads, streams * length, D, generator=gen, device="cuda").bfloat16()
    k = torch.randn(kv_heads, streams * length, D, generator=gen, device="cuda").bfloat16()
    v = torch.randn(kv_heads, streams * length, D, generator=gen, device="cuda").bfloat16()
    return q, k, v


def _run(q, k, v, stores, lengths, layer, length, kv_heads, window=None):
    launch = ca.Round(stores, lengths, heads=q.shape[0], kv_heads=kv_heads, dim=D, length=length, device=q.device,
                      window=window)
    return ca.block_attention(q, k, v, launch, layer, D ** -0.5)


# (8, 4, 16) and (12, 4, 33): group x padded rows not a multiple of the 128-row query tile (dead tile rows).
@pytest.mark.parametrize("heads,kv_heads,length", [(32, 4, 16), (32, 4, 8), (32, 4, 2), (32, 4, 128), (16, 2, 33),
                                                  (8, 4, 16), (12, 4, 33)])
def test_matches_exact_softmax_across_chunk_boundaries(heads, kv_heads, length):
    gen = torch.Generator(device="cuda").manual_seed(heads + length)
    lengths = [1, 63, CH - 1, CH, CH + 1, 3 * CH + 17]
    stores = [_store(kv_heads, n + 5 + 64 * i, gen) for i, n in enumerate(lengths)]
    q, k, v = _inputs(heads, kv_heads, length, len(lengths), gen)
    for layer in (0, LAYERS - 1):
        out = _run(q, k, v, stores, lengths, layer, length, kv_heads)
        ref, bound = _reference(q, k, v, stores, lengths, layer, length, D ** -0.5)
        assert ((out.double() - ref).abs() <= bound).all()


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


@pytest.mark.parametrize("window", [CH, 2 * CH])
def test_a_window_attends_the_recent_chunks_exactly_and_by_length_alone(window):
    """Keys from chunk max(0, n - window) // CH on, within the same float64 bound; a stream's bits depend only on its
    length and the window (alone, among others, in a larger store)."""
    gen = torch.Generator(device="cuda").manual_seed(window)
    heads, kv_heads, length = 32, 4, 16
    lengths = [5, CH, CH + 1, 2 * CH + 17, 3 * CH + 5]
    stores = [_store(kv_heads, n + 7, gen) for n in lengths]
    q, k, v = _inputs(heads, kv_heads, length, len(lengths), gen)
    out = _run(q, k, v, stores, lengths, 1, length, kv_heads, window)
    ref, bound = _reference(q, k, v, stores, lengths, 1, length, D ** -0.5, window)
    assert ((out.double() - ref).abs() <= bound).all()
    full = _run(q, k, v, stores, lengths, 1, length, kv_heads)
    rows = slice(4 * length, 5 * length)                    # the longest stream sees fewer keys with the window
    assert not torch.equal(out[rows], full[rows])
    for j in range(len(lengths)):
        rows = slice(j * length, (j + 1) * length)
        alone = _run(q[:, rows].contiguous(), k[:, rows].contiguous(), v[:, rows].contiguous(), [stores[j]],
                     lengths[j:j + 1], 1, length, kv_heads, window)
        assert torch.equal(out[rows], alone), j


def test_a_window_must_be_whole_chunks():
    gen = torch.Generator(device="cuda").manual_seed(1)
    for bad in (0, CH - 1, CH + 64, 1.5 * CH):
        with pytest.raises(ValueError, match="window"):
            ca.Round([_store(4, 8, gen)], [4], heads=32, kv_heads=4, dim=D, length=16, device="cuda", window=bad)
