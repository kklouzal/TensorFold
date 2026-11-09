"""CPU dispatch controllers; CUDA Dense32 tests remain the native math oracle."""

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.sm70 import gguf_ops  # noqa: E402 - requires the optional Torch guard above
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense  # noqa: E402


@pytest.fixture
def dense(monkeypatch):
    value = GGUFDense(torch.ones((4, 8), dtype=torch.float16))
    value.implementation = "native32"
    # Exercise eligibility and mutation metadata without invoking CUDA math.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    return value


@pytest.mark.parametrize("rows", [1, 3, 18, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_native_staging_keeps_input_stride_output_dtype_and_owner_version(dense, monkeypatch, rows, dtype):
    x = torch.ones((rows, 16), dtype=torch.float16)[:, ::2]
    storage = torch.full((rows, 8), -7, dtype=dtype)
    out = storage[:, :4]
    version = storage._version
    calls = []

    def native(value, weight, destination):
        calls.append((value, weight, destination))
        # A data_ptr-style write does not update Torch's version counter.
        destination.numpy()[:] = 11
        return destination

    monkeypatch.setattr(gguf_ops, "dense32", native)
    assert dense.matmul(x, out=out, f32=True) is out
    assert calls == [(x, dense.weight, out)]
    assert out.dtype == dtype and out._version == storage._version == version + 1
    assert torch.all(out == 11) and torch.all(storage[:, 4:] == -7)


@pytest.mark.parametrize("f32", [False, True])
def test_missing_output_uses_requested_dtype(dense, monkeypatch, f32):
    x = torch.ones((3, 8), dtype=torch.float16)

    def native(value, weight, out):
        out.numpy()[:] = 2
        return out

    monkeypatch.setattr(gguf_ops, "dense32", native)
    result = dense.matmul(x, f32=f32)
    assert result.shape == (3, 4) and result.dtype == (torch.float32 if f32 else torch.float16)
    assert result._version == 1 and torch.all(result == 2)


def test_existing_aten_mutation_is_not_counted_twice(dense, monkeypatch):
    out = torch.empty((3, 4), dtype=torch.float16)
    version = out._version
    monkeypatch.setattr(gguf_ops, "dense32", lambda x, w, destination: destination.fill_(2))
    dense.matmul(torch.ones((3, 8), dtype=torch.float16), out=out)
    assert out._version == version + 1


@pytest.mark.parametrize("case", ["empty", "large", "width", "dtype", "rank", "column-output",
                                  "overlap-rows", "input-alias", "weight-alias", "out-dtype", "out-shape",
                                  "input-grad", "weight-grad", "out-grad", "neg-input", "neg-weight", "neg-output",
                                  "input-parameter", "weight-parameter", "output-parameter", "cpu", "inference-output"])
def test_reference_route_keeps_generic_layout_autograd_alias_and_error_contracts(dense, monkeypatch, case):
    x = torch.ones((3, 8), dtype=torch.float16)
    out = torch.empty((3, 4), dtype=torch.float16)
    if case == "empty":
        x, out = x[:0], out[:0]
    elif case == "large":
        x, out = x[:1].expand(33, 8), out[:1].expand(33, 4)
    elif case == "width":
        x = x[:, :7]
    elif case == "dtype":
        x = x.float()
    elif case == "rank":
        x = x[0]
    elif case == "column-output":
        out = torch.empty((4, 3), dtype=torch.float16).T
    elif case == "overlap-rows":
        out = out[:1].expand(3, 4)
    elif case == "input-alias":
        out = x[:, :4]
    elif case == "weight-alias":
        out = dense.weight[:3, :4]
    elif case == "out-dtype":
        out = out.double()
    elif case == "out-shape":
        out = out[:2]
    elif case == "input-grad":
        x.requires_grad_()
    elif case == "weight-grad":
        dense.weight.requires_grad_()
    elif case == "out-grad":
        out.requires_grad_()
    elif case == "neg-input":
        x = torch._neg_view(x)
    elif case == "neg-weight":
        dense.weight = torch._neg_view(dense.weight)
    elif case == "neg-output":
        out = torch._neg_view(out)
    elif case == "input-parameter":
        x = torch.nn.Parameter(x, requires_grad=False)
    elif case == "weight-parameter":
        dense.weight = torch.nn.Parameter(dense.weight, requires_grad=False)
    elif case == "output-parameter":
        out = torch.nn.Parameter(out, requires_grad=False)
    elif case == "cpu":
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: False))
    elif case == "inference-output":
        with torch.inference_mode():
            out = torch.empty((3, 4), dtype=torch.float16)
    calls, sentinel = [], object()
    monkeypatch.setattr(dense, "matmul_reference", lambda value, **kwargs: calls.append((value, kwargs)) or sentinel)
    monkeypatch.setattr(gguf_ops, "dense32", lambda *a: pytest.fail("ineligible native call"))
    assert dense.matmul(x, out=out, f32=True) is sentinel
    assert calls == [(x, {"out": out, "f32": True})]


def test_inference_tensor_has_no_version_and_keeps_native_path(dense, monkeypatch):
    monkeypatch.setattr(gguf_ops, "dense32", lambda x, w, out: out.fill_(2))
    with torch.inference_mode():
        x = torch.ones((3, 8), dtype=torch.float16)
        out = dense.matmul(x)
        assert out.is_inference() and torch.all(out == 2)


@pytest.mark.parametrize("inference", [False, True])
def test_disabled_grad_keeps_native_path_and_normal_tensor_version(dense, monkeypatch, inference):
    x = torch.ones((3, 8), dtype=torch.float16, requires_grad=True)
    out = torch.empty((3, 4), dtype=torch.float16, requires_grad=True)
    version = out._version

    def native(value, weight, destination):
        destination.detach().numpy()[:] = 2
        return destination

    monkeypatch.setattr(gguf_ops, "dense32", native)
    with torch.inference_mode() if inference else torch.no_grad():
        assert dense.matmul(x, out=out) is out
    assert out._version == version + 1 and torch.all(out == 2)


def test_reference_error_type_message_and_output_survive_invalid_input(dense, monkeypatch):
    monkeypatch.setattr(gguf_ops, "dense32", lambda *a: pytest.fail("invalid native call"))
    out = torch.full((3, 4), -7, dtype=torch.float16)
    invalid = torch.ones((3, 7), dtype=torch.float16)
    with pytest.raises(ValueError) as original:
        dense.matmul_reference(invalid, out=out)
    with pytest.raises(ValueError) as actual:
        dense.matmul(invalid, out=out)
    assert str(actual.value) == str(original.value) and torch.all(out == -7)


@pytest.mark.parametrize("operand", ["input", "weight"])
def test_gradients_keep_reference_autograd(dense, monkeypatch, operand):
    monkeypatch.setattr(gguf_ops, "dense32", lambda *a: pytest.fail("autograd native call"))
    x = torch.ones((3, 8), dtype=torch.float16, requires_grad=operand == "input")
    dense.weight.requires_grad_(operand == "weight")
    dense.matmul(x).sum().backward()
    value, expected = (x, 4) if operand == "input" else (dense.weight, 3)
    assert torch.equal(value.grad, torch.full_like(value, expected))


@pytest.mark.parametrize("layout", ["columns", "negative"])
def test_generic_output_layout_keeps_reference_values(dense, monkeypatch, layout):
    monkeypatch.setattr(gguf_ops, "dense32", lambda *a: pytest.fail("generic layout native call"))
    x = torch.arange(24, dtype=torch.float16).view(3, 8)
    owner = torch.full((3, 8), -7, dtype=torch.float16)
    out = owner[:, ::2] if layout == "columns" else torch._neg_view(owner[:, :4])
    assert dense.matmul(x, out=out) is out
    assert torch.equal(out, x.sum(dim=1, keepdim=True).expand(3, 4))
    untouched = owner[:, 1::2] if layout == "columns" else owner[:, 4:]
    assert torch.all(untouched == -7)
