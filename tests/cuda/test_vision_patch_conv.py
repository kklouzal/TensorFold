"""The real Qwen3.8 vision patch projection must have a default cuDNN engine.

The NGC inference base omits precompiled cuDNN engines. A mixed base/wheel
installation can pass attention tests but fail this exact BF16 Conv3d shape.
These synthetic tests load neither TensorFold nor a checkpoint; their GPU
tensors occupy less than 16 MiB. The CPU oracle is an independent FP32 matrix
projection, valid because each convolution kernel covers one complete patch.
"""

import pytest
import torch
from torch.nn import functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


@pytest.mark.parametrize("patches", [4, 16])
@pytest.mark.parametrize("zero_input", [True, False], ids=["startup-zero", "image-random"])
def test_qwen_vision_patch_conv3d_default_engine(patches, zero_input):
    generator = torch.Generator(device="cpu").manual_seed(19)
    patch_shape = (3, 2, 16, 16)
    inputs = torch.randn((patches, *patch_shape), generator=generator).to(torch.bfloat16)
    weight = (torch.randn((1152, *patch_shape), generator=generator) * 0.02).to(torch.bfloat16)
    bias = (torch.randn((1152,), generator=generator) * 0.02).to(torch.bfloat16)
    if zero_input:
        inputs.zero_()

    reference = F.linear(inputs.float().flatten(1), weight.float().flatten(1), bias.float())
    projection = torch.nn.Conv3d(3, 1152, kernel_size=(2, 16, 16),
                                 stride=(2, 16, 16), bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        projection.weight.copy_(weight)
        projection.bias.copy_(bias)
    projection = projection.cuda().eval()
    with torch.inference_mode(), torch.backends.cudnn.flags(
            enabled=True, benchmark=False, deterministic=False):
        output = projection(inputs.cuda())
        torch.cuda.synchronize()

    assert output.shape == (patches, 1152, 1, 1, 1)
    assert output.dtype == torch.bfloat16
    actual = output.cpu().float().flatten(1)
    assert torch.isfinite(actual).all()
    if zero_input:
        assert torch.equal(actual, bias.float().expand(patches, -1))
    else:
        # One BF16 ULP at the oracle's maximum magnitude; the same explicit
        # output-scale bound used for this model's existing BF16 kernel tests.
        # FP32 accumulation and bias fusion may round differently across engines.
        maximum_error = (actual - reference).abs().max().item()
        bound = reference.abs().max().item() * 2 ** -7
        assert maximum_error <= bound, (maximum_error, bound)
