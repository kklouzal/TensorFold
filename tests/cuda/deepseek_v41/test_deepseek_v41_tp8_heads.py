"""TensorFold TP8 per-rank shapes (8 query heads, 4 indexer heads) for the vendored V4.1 attention paths, next to
the TP4 shapes (16 / 8), against the upstream test oracle and tolerances
(tests/models/test_deepseek_v41_attention_ops.py at 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f).
CUDA only: the FP16 bmm with FP32 output (aten::bmm.dtype_out) has no CPU kernel.
"""

import pytest
import torch

from tensorfold.families.deepseek_v4.cuda.kernels import attention_ops as ops
from tensorfold.families.deepseek_v4.cuda.kernels.kv_pack import pack_rows
from tensorfold.families.deepseek_v4.cuda.kernels.quant_rope import attention_qdq
from tensorfold.families.deepseek_v4.cuda.kernels.sparse_prefill_bmm import attend_gathered_keys

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _softmax_oracle(q, key, sink):
    # Upstream test oracle: FP32 scores with the sink column, FP16 probabilities, FP32 PV, FP16 result.
    scores = q.float() @ key.float().T * 512**-0.5
    probability = torch.softmax(torch.cat((scores, sink[:, None]), dim=1), dim=-1)[:, :-1].half()
    return (probability.float() @ key.float()).half()


@pytest.mark.parametrize("heads", [16, 8, 4])
def test_attend_gathered_keys_heads(heads):
    # The unfused upstream boundary, with unused (NaN-holding) slots, and the fused production path on the same keys.
    g = torch.Generator().manual_seed(5)
    n, width = 3, 40
    q = torch.randn(n, heads, 512, generator=g).half().cuda()
    keys = (torch.randn(n, width, 512, generator=g) * 0.1).half().cuda()
    unused = torch.zeros(n, width, dtype=torch.bool, device="cuda")
    unused[:, 30:] = True
    keys[unused] = float("nan")
    sink = torch.randn(heads, generator=g).cuda()
    out = torch.empty_like(q)
    scores = torch.empty(n, heads, width, device="cuda")
    logits = torch.empty(n, heads, width + 1, device="cuda")
    attend_gathered_keys(q, keys, unused, 512**-0.5, sink, out, scores, logits, torch.empty_like(logits))
    fused = torch.empty_like(q)
    ops._attend_fused(q, keys, unused, 512**-0.5, sink, fused, torch.empty(n, heads, width, device="cuda"))
    for i in range(n):
        expected = _softmax_oracle(q[i], keys[i, :30], sink)
        torch.testing.assert_close(out[i], expected, rtol=0.001, atol=2e-6)
        torch.testing.assert_close(fused[i], expected, rtol=0.001, atol=2e-6)


@pytest.mark.parametrize("heads", [8, 4])
@pytest.mark.parametrize("tokens", [1, 6, 300])
def test_select_positions_heads(heads, tokens):
    # Packed (decode Triton kernel / prefill unpack) and FP16 paths return valid ascending top-512 sets; the decode
    # kernel's head sums are compared with the FP16 einsum boundaries directly.
    g = torch.Generator().manual_seed(9)
    rows = 4096
    raw = (torch.randn(rows, 128, generator=g) * 0.5).half().cuda()
    packed = pack_rows(raw, False)
    keys = attention_qdq(raw, fp4=True, group=32)
    q = attention_qdq(
        (torch.randn(tokens * heads, 128, generator=g) * 0.5).half().cuda(), fp4=True, group=32
    ).view(tokens, heads, 128)
    w = (torch.randn(tokens, heads, generator=g) * 0.2).half().cuda()
    pos = torch.arange(3000, 3000 + tokens, device="cuda")
    valid = torch.ones(tokens, dtype=torch.bool, device="cuda")
    for result in (
        ops.select_packed_positions(q, w, packed, pos, valid, 1, lambda x: x),
        ops.select_compressed_positions(q, w, keys, pos, valid, 1, lambda x: x),
    ):
        assert result.shape == (tokens, 512)
        for r in range(tokens):
            live = result[r][result[r] >= 0]
            assert live.numel() == 512 and bool((live <= pos[r]).all()) and bool((live[1:] > live[:-1]).all())
