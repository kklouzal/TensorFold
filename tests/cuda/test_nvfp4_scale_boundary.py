"""External native UE4M3 scales: real GPU admission, packing and MMA consumption."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native NVFP4 MMA requires an SM12.x GPU", allow_module_level=True)

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4.linear import Fp4Linear


@pytest.mark.parametrize("bad", [0x7F, 0x80, 0xFF])
@pytest.mark.parametrize("dtype", ["u8", "e4m3"])
def test_invalid_gpu_scale_payload_is_rejected_before_pack4(bad, dtype, monkeypatch):
    weight = torch.zeros((64, 32), dtype=torch.uint8, device="cuda")
    scale = torch.full((64, 4), 0x38, dtype=torch.uint8, device="cuda")
    scale[-1, -1] = bad
    value = scale if dtype == "u8" else scale.view(torch.float8_e4m3fn)
    before = scale.clone()
    monkeypatch.setattr(checkpoint, "pack4", lambda *args: pytest.fail("invalid GPU scales reached pack4"))
    with pytest.raises(ValueError, match="finite unsigned E4M3"):
        Fp4Linear.from_checkpoint(weight, value, 1.0, act=.125)
    assert torch.equal(scale, before)


@pytest.mark.parametrize("dtype", ["u8", "e4m3"])
def test_finite_zero_and_max_ue4m3_survive_native_packing_and_mma(dtype):
    weight = torch.zeros((64, 32), dtype=torch.uint8, device="cuda")
    scale = (torch.arange(256, device="cuda") % 127).to(torch.uint8).reshape(64, 4)
    value = scale if dtype == "u8" else scale.view(torch.float8_e4m3fn)
    before = scale.clone()
    linear = Fp4Linear.from_checkpoint(weight, value, 1.0, act=.125)
    restored = linear.bs.view(1, 1, 64, 4).permute(0, 2, 1, 3).reshape(64, 4)
    assert torch.equal(restored, before) and torch.equal(scale, before)
    result = linear(torch.ones((2, 64), dtype=torch.bfloat16, device="cuda"))
    assert torch.equal(result, torch.zeros_like(result))
