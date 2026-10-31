"""Borrowed norm parameters and native/fused dispatch contracts."""
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.e2b_norm import E2BRowRMSNorm, install_row_norms  # noqa: E402 - optional runtime admitted above


@pytest.mark.parametrize("weighted", [False, True])
def test_native_prompt_norm_is_unchanged_and_parameters_are_borrowed(weighted):
    native = Gemma4RMSNorm(32, with_scale=weighted).to(torch.bfloat16).eval()
    model = torch.nn.Sequential(native)
    policy = SimpleNamespace(prompt=True)
    value = torch.randn(1, 3, 32, dtype=torch.bfloat16)
    expected = native(value)
    install_row_norms(model, policy)
    assert isinstance(model[0], E2BRowRMSNorm)
    if weighted:
        assert model[0].weight is native.weight
    assert model[0].state_dict().keys() == native.state_dict().keys()
    assert torch.equal(model(value), expected)


@pytest.mark.parametrize("weighted", [False, True])
def test_decode_reuses_unrounded_fp32_ct_norm_contract(monkeypatch, weighted):
    from tensorfold.families.gemma4.cuda import forward
    native = Gemma4RMSNorm(32, eps=1e-6, with_scale=weighted).to(torch.bfloat16)
    owned = E2BRowRMSNorm(native, SimpleNamespace(prompt=False))
    value = torch.randn(1, 8, 3, 32, dtype=torch.bfloat16)
    calls = []

    def provider(value, weight, eps):
        calls.append((weight, eps))
        return native(value)

    monkeypatch.setattr(forward, "ct_rms_norm", provider)
    assert torch.equal(owned(value), native(value))
    assert len(calls) == 1 and calls[0][1] == native.eps
    assert calls[0][0] is (native.weight if weighted else None)


def test_unsupported_activation_precision_is_not_silently_reduced():
    native = Gemma4RMSNorm(32).to(torch.bfloat16)
    owned = E2BRowRMSNorm(native, SimpleNamespace(prompt=False))
    with pytest.raises(ValueError, match="BF16"):
        owned(torch.ones(1, 1, 32, dtype=torch.float32))
