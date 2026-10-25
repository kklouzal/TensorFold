"""Real SM70-only library backend gate; no math fallback is enabled."""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="requires actual V100 memory-efficient attention execution")


@pytest.mark.parametrize("rows", (1, 4, 64, 256))
def test_fp16_h16_d72_noncausal_uses_only_efficient_backend_and_preserves_constant_values(rows):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as F

    query = torch.zeros((1, 16, rows, 72), dtype=torch.float16, device="cuda")
    key = query.clone()
    expected = torch.arange(72, dtype=torch.float16, device="cuda").div_(128).expand(1, 16, rows, 72).contiguous()
    with torch.inference_mode(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        output = F.scaled_dot_product_attention(query, key, expected, dropout_p=0., is_causal=False)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
