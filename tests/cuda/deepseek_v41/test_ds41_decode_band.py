"""TensorFold-owned DS41 decode fusions are bitwise their unfused compositions (CUDA, or CPU under TRITON_INTERPRET=1):
rope_qdq == attention_qdq(apply_rope), route_local == route_topk + Target.moe's former EP mapping, add_halves ==
a.float() + b.float()."""

import pytest
import torch

from tensorfold.families.deepseek_v4.cuda.kernels import band, quant_rope as q, router
from tensorfold.families.deepseek_v4.cuda.kernels.rope_qdq import rope_qdq
from tensorfold.families.deepseek_v4.cuda.kernels.triton_utils import INTERPRET

DEVICE = "cpu" if INTERPRET else "cuda"
pytestmark = pytest.mark.skipif(not INTERPRET and not torch.cuda.is_available(), reason="CUDA or TRITON_INTERPRET")


def bits(x):
    return x.contiguous().view(-1).view(torch.int16 if x.element_size() == 2 else torch.int32)


@pytest.mark.parametrize("d,heads,group,fp4,inverse,compressed", [
    (512, 1, 128, False, False, False), (512, 1, 128, False, False, True), (512, 8, 128, False, True, True),
    (128, 4, 32, True, False, True), (128, 4, 32, True, False, False)])
def test_rope_qdq_is_the_composition(d, heads, group, fp4, inverse, compressed):
    torch.manual_seed(d + heads)
    table = q.rope_table(16384, compressed=compressed, device=DEVICE)
    x = (torch.randn(9, heads, d, device=DEVICE) * 3).half()
    x[0] = 0                                                   # all-zero groups keep their minimum scale
    positions = torch.tensor([0, 1, 7, 100, 4095, 16383, 16384, 40000, 65535], dtype=torch.int64, device=DEVICE)
    expected = q.attention_qdq(q.apply_rope(x, positions, table, inverse=inverse), fp4=fp4, group=group)
    assert torch.equal(bits(rope_qdq(x, positions, table, inverse=inverse, fp4=fp4, group=group)), bits(expected))


@pytest.mark.parametrize("rows", [1, 7, 32, 300])
@pytest.mark.parametrize("first", [0, 192])
def test_route_local_is_route_topk_plus_the_ep_mapping(rows, first):
    torch.manual_seed(rows + first)
    logits = torch.randn(rows, 384, device=DEVICE) * 3
    logits[0, :9] = logits[0, 0]                               # ties pick the lowest expert id
    bias = torch.randn(384, device=DEVICE) * 0.1
    p, ids = router.route_topk(logits, bias, 6, 1.5)
    local = ids.to(torch.int32) - first
    mine = (local >= 0) & (local < 192)
    want_ids = torch.where(mine, local, torch.zeros_like(local))
    want_w = torch.where(mine, p.float(), torch.zeros_like(p, dtype=torch.float32))
    got_ids, got_w = band.route_local(logits, bias, 6, 1.5, first, 192)
    assert torch.equal(got_ids, want_ids) and torch.equal(bits(got_w), bits(want_w))


def test_add_halves_is_the_fp32_sum():
    a, b = (torch.randn(33, 5120, device=DEVICE) * 100).half(), torch.randn(33, 5120, device=DEVICE).half()
    assert torch.equal(bits(band.add_halves(a, b)), bits(a.float() + b.float()))
