"""Borrowed norm parameters and native/fused dispatch contracts."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.e2b_norm import E2BRowRMSNorm, install_row_norms  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.e2b_rows import E2BRowPolicy  # noqa: E402 - optional runtime admitted above


@pytest.mark.parametrize("weighted", [False, True])
def test_native_prompt_norm_is_unchanged_and_parameters_are_borrowed(weighted):
    native = Gemma4RMSNorm(32, with_scale=weighted).to(torch.bfloat16).eval()
    model = torch.nn.Sequential(native)
    policy = E2BRowPolicy()
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
    policy = E2BRowPolicy()
    policy.prompt = False
    owned = E2BRowRMSNorm(native, policy)
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
    policy = E2BRowPolicy()
    policy.prompt = False
    owned = E2BRowRMSNorm(native, policy)
    with pytest.raises(ValueError, match="BF16"):
        owned(torch.ones(1, 1, 32, dtype=torch.float32))


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("prompt", [False, True])
def test_prompt_fusion_is_opt_in_and_decode_routing_preserved(monkeypatch, fuse, prompt):
    from tensorfold.families.gemma4.cuda import forward
    native = Gemma4RMSNorm(32).to(torch.bfloat16)
    policy = E2BRowPolicy(fuse_prompt_norms=fuse)
    policy.prompt = prompt
    owned = E2BRowRMSNorm(native, policy)
    value = torch.randn(1, 3, 32, dtype=torch.bfloat16)
    calls = []

    def provider(value, weight, eps):
        calls.append((weight is native.weight, eps))
        return native(value)

    monkeypatch.setattr(forward, "ct_rms_norm", provider)
    assert torch.equal(owned(value), native(value))
    assert len(calls) == int(fuse or not prompt)


def test_reference_override_keeps_enabled_prompt_fusion_native(monkeypatch):
    from tensorfold.families.gemma4.cuda import forward
    native = Gemma4RMSNorm(32).to(torch.bfloat16)
    policy = E2BRowPolicy(fuse_prompt_norms=True)
    owned = E2BRowRMSNorm(native, policy)
    value = torch.randn(1, 3, 32, dtype=torch.bfloat16)
    monkeypatch.setattr(forward, "ct_rms_norm", lambda *_: pytest.fail("native oracle called fusion"))
    with policy.native_reference():
        assert policy.fuse_prompt_norms is True
        assert torch.equal(owned(value), native(value))


def test_decode_fast_path_does_not_read_new_reference_or_fusion_fields(monkeypatch):
    from tensorfold.families.gemma4.cuda import forward
    accesses = []

    class DecodePolicy:
        @property
        def prompt(self):
            accesses.append("prompt")
            return False

        def __getattr__(self, name):
            pytest.fail("decode touched new policy field " + name)

    native = Gemma4RMSNorm(32).to(torch.bfloat16)
    owned = E2BRowRMSNorm(native, DecodePolicy())
    value = torch.randn(1, 8, 32, dtype=torch.bfloat16)
    calls = []

    def provider(actual, weight, eps):
        calls.append((actual is value, weight is native.weight, eps))
        return native(actual)

    monkeypatch.setattr(forward, "ct_rms_norm", provider)
    assert torch.equal(owned(value), native(value))
    assert accesses == ["prompt"]
    assert calls == [(True, True, native.eps)]
