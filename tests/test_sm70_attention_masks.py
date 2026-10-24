"""A context bucket must not read discarded or sparse-excluded value rows."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("sparse", [False, True])
def test_retained_prefix_ignores_nonfinite_values_outside_attention_mask(device, sparse):
    from tensorfold.cuda.sm70.family_ops import attention

    if device == "cuda" and (not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0)):
        pytest.skip("SM70 CUDA attention adapter")
    keys, heads, dims = 8, 3, 32
    query = torch.zeros((1, heads, dims), dtype=torch.float16, device=device)
    key = torch.zeros((keys, 1, dims), dtype=torch.float16, device=device)
    value = torch.ones_like(key)
    value[0].fill_(2)
    value[2].fill_(4)
    # A restored prefix ends at row2. Its retained cache allocation still
    # contains abandoned later rows, and a graph continues using all8 rows.
    value[3:].fill_(float("nan"))
    if sparse:
        value[1].fill_(float("inf"))
    scratch = SimpleNamespace(out=torch.empty_like(query), qsa=sparse,
                              ids=torch.tensor([[0, 2]], device=device, dtype=torch.int32),
                              nk=torch.tensor([2], device=device, dtype=torch.int32),
                              sparse=torch.tensor([1], device=device, dtype=torch.int32))
    position = torch.tensor([2], device=device, dtype=torch.int32)
    actual = attention(query, key, value, position, scratch, 1, dims ** -0.5, context=keys).clone()
    # Match the public FP16-probability / FP32-accumulation arithmetic.
    included = [0, 2] if sparse else [0, 1, 2]
    probability = torch.tensor(1 / len(included), dtype=torch.float16).float()
    expected = (sum(float(value[index, 0, 0]) for index in included) * probability).half()
    torch.testing.assert_close(actual, torch.full_like(query, expected), atol=0, rtol=0)
    # The fix excludes unread rows; a nonfinite visible row still propagates.
    value[0].fill_(float("nan"))
    assert attention(query, key, value, position, scratch, 1, dims ** -0.5, context=keys).isnan().all()
