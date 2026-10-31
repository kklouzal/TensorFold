"""Owned projection selection/parameter lifetime; native kernel GPU gates are separate."""
import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.gemma4.cuda.e2b_rows import E2BRowLinear, E2BRowPolicy, install_row_linears  # noqa: E402 - optional runtime admitted above


def test_install_borrows_loaded_parameters_and_keeps_prompt_native():
    native = torch.nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)
    model = torch.nn.Sequential(native)
    policy = E2BRowPolicy("route")
    value = torch.randn(1, 3, 32, dtype=torch.bfloat16)
    expected = native(value)
    install_row_linears(model, policy)
    assert model[0].weight is native.weight
    assert torch.equal(model(value), expected)


def test_decode_projection_invokes_fixed_row_provider_and_preserves_shape(monkeypatch):
    import tensorfold.cuda.moe as moe
    calls = []
    native = torch.nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)
    policy = E2BRowPolicy("route")
    policy.prompt = False
    candidate = E2BRowLinear(native, policy)

    def provider(rows, weights, output):
        calls.append((rows.shape, weights is native.weight, output.dtype))
        output.copy_(rows.float() @ weights.float().T)

    monkeypatch.setattr(moe, "route_logits", provider)
    value = torch.randn(1, 8, 32, dtype=torch.bfloat16)
    result = candidate(value)
    assert result.shape == (1, 8, 64) and result.dtype == torch.bfloat16
    assert calls == [(torch.Size([8, 32]), True, torch.float32)]


def test_native_padded_projection_uses_same_matrix_shape_for_all_rows(monkeypatch):
    native = torch.nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)
    policy = E2BRowPolicy("padded")
    policy.prompt = False
    candidate = E2BRowLinear(native, policy)
    calls = []
    original = torch.nn.functional.linear

    def observe(value, weight, bias=None):
        calls.append(tuple(value.shape))
        return original(value, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", observe)
    value = torch.randn(1, 8, 32, dtype=torch.bfloat16)
    for rows in range(1, 9):
        actual = candidate(value[:, :rows])
        assert actual.shape == (1, rows, 64)
    assert calls == [(8, 32)] * 8


@pytest.mark.parametrize("width,bias,dtype", [(33, False, torch.bfloat16), (32, True, torch.bfloat16),
                                           (32, False, torch.float32)])
def test_unsupported_projection_rejected_before_install(width, bias, dtype):
    native = torch.nn.Linear(width, 64, bias=bias, dtype=dtype)
    with pytest.raises(ValueError, match="BF16"):
        E2BRowLinear(native, E2BRowPolicy())
