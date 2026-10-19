"""Actual GPU operator checks for TensorFold's separate CT numerical policy."""

from types import SimpleNamespace

import torch

from tensorfold.families.gemma4.cuda import glue
from tensorfold.families.gemma4.cuda.forward import Gemma4Target, ct_rms_norm, rms_norm
from tensorfold.families.gemma4.cuda.nvfp4_weights import _Router
from test_gemma_cuda_target import config


def test_ct_rms_single_cast_and_partition_identity():
    torch.manual_seed(391)
    x = torch.randn((17,2816),device="cuda",dtype=torch.bfloat16)
    weight = torch.randn((2816,),device="cuda",dtype=torch.bfloat16)
    actual = ct_rms_norm(x,weight,1e-6)
    independent = ((x.double() * (x.double().square().mean(-1,keepdim=True)+1e-6).pow(-.5))
                   * weight.double()).to(x.dtype)
    # One output BF16 quantization step, with a conservative one-ULP envelope
    # for FP32 reduction/rsqrt relative to the independent FP64 reference.
    torch.testing.assert_close(actual,independent,atol=0,rtol=2**-7)
    assert torch.equal(actual,torch.cat([ct_rms_norm(row[None],weight,1e-6) for row in x]))
    assert not torch.equal(actual,rms_norm(x,weight,1e-6))


def test_ct_geglu_preserves_bf16_activation_before_product():
    # Reuse the native expert adapter's boundary-discriminating input range.
    # The CT policy's fused GeGLU (glue.geglu) must equal PyTorch's BF16 GELU
    # then BF16 product: two storage boundaries, not one.
    gate = torch.linspace(-4.,4.,2048,device="cuda").to(torch.bfloat16)
    up = torch.linspace(.125,3.,2048,device="cuda").to(torch.bfloat16)
    activated = torch.nn.functional.gelu(gate,approximate="tanh")
    reference = activated * up
    one_round = (torch.nn.functional.gelu(gate.float(),approximate="tanh") * up.float()).to(torch.bfloat16)
    actual = glue.geglu(gate,up)
    assert torch.equal(actual,reference)
    assert not torch.equal(actual,one_round)


def test_publisher_plain_router_fp32_and_fixed_k_order():
    torch.manual_seed(62)
    weight = torch.randn((128,2816),device="cuda",dtype=torch.bfloat16)
    x = torch.randn((129,2816),device="cuda",dtype=torch.bfloat16)
    router = _Router(weight)
    actual = router(x)
    assert actual.dtype == torch.float32
    assert not torch.equal(actual,actual.to(torch.bfloat16).float())
    assert torch.equal(actual,torch.cat([router(row[None]) for row in x]))
    reference = x.double() @ weight.double().T
    # BF16 products are exact in FP32; a conservative gamma_K envelope
    # covers every product addition, including each MMA internal reduction.
    products = x.double().abs() @ weight.double().abs().T
    bound = (2816*2**-24/(1-2816*2**-24))*products
    assert torch.all((actual.double()-reference).abs() <= bound)


def test_ct_router_root_then_learned_scale_and_fp32_full_softmax():
    torch.manual_seed(293)
    c = config()
    captured = []
    logits = torch.tensor([[1.,.5,-.25,-1.]],device="cuda",dtype=torch.float32)
    def router(x):
        captured.append(x)
        return logits
    model = object.__new__(Gemma4Target)
    model.config = c
    model._router_root = torch.tensor(c.hidden**-.5,device="cuda",dtype=torch.bfloat16)
    layer = SimpleNamespace(router=router,
        router_scale=torch.randn((64,),device="cuda",dtype=torch.bfloat16),
        expert_scale=torch.tensor([.75,1.25,.5,2.],device="cuda",dtype=torch.bfloat16))
    h = torch.randn((1,64),device="cuda",dtype=torch.bfloat16)
    picks,weights = model._route_ct(h,layer)
    expected_input = ct_rms_norm(h,None,c.eps) * model._router_root
    expected_input = expected_input * layer.router_scale
    assert torch.equal(captured[0],expected_input)
    probabilities = torch.softmax(logits,dim=-1)
    selected = probabilities[:,:2]
    expected = selected/selected.sum(-1,keepdim=True) * layer.expert_scale[:2].float()
    assert picks.dtype == torch.int32 and weights.dtype == torch.float32
    assert torch.equal(picks,torch.tensor([[0,1]],device="cuda",dtype=torch.int32))
    assert torch.equal(weights,expected)
