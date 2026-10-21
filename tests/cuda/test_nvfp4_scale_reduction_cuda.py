"""CUDA validation keeps its original Torch reduction and boundary."""

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.nvfp4.format import validate_native_a4_scales

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA validation requires a device")


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize("byte", [0, 0x7e, 0x7f, 0xff])
def test_cuda_uses_torch_reduction_without_numpy_or_private_cpu_copy(dtype, byte, monkeypatch):
    value = torch.tensor([0, byte, 0x7e], dtype=torch.uint8, device="cuda").view(dtype)
    before = value.view(torch.uint8).clone()
    version, pointer = value._version, value.data_ptr()
    monkeypatch.setattr(torch.Tensor, "numpy", lambda *args, **kwargs: pytest.fail("CUDA NumPy bridge"))
    monkeypatch.setattr(torch.Tensor, "clone", lambda *args, **kwargs: pytest.fail("CUDA private copy"))
    if byte <= 0x7e:
        validate_native_a4_scales(value, name="native CUDA scales")
    else:
        with pytest.raises(ValueError, match="finite unsigned E4M3"):
            validate_native_a4_scales(value, name="native CUDA scales")
    assert value._version == version and value.data_ptr() == pointer
    assert torch.equal(value.view(torch.uint8), before)
