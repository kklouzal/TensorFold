"""Actual GPU operator checks for TensorFold's separate CT numerical policy."""

from types import SimpleNamespace

import torch

from tensorfold.families.gemma4.cuda.forward import Gemma4Target, _geglu_ct, ct_rms_norm, rms_norm
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
    # PyTorch owns GELU; this checks the target's two BF16 storage boundaries.
    gate = torch.linspace(-4.,4.,2048,device="cuda").to(torch.bfloat16)
    up = torch.linspace(.125,3.,2048,device="cuda").to(torch.bfloat16)
    activated = torch.nn.functional.gelu(gate,approximate="tanh")
    reference = activated * up
    one_round = (torch.nn.functional.gelu(gate.float(),approximate="tanh") * up.float()).to(torch.bfloat16)
    actual = _geglu_ct(gate,up)
    assert torch.equal(actual,reference)
    assert not torch.equal(actual,one_round)


def test_ct_router_root_then_learned_scale_and_fp32_full_softmax():
    torch.manual_seed(293)
    c = config()
    model = object.__new__(Gemma4Target)
    model.config = c
    model._router_root_value = float(torch.tensor(c.hidden**-.5,dtype=torch.bfloat16))
    weight = (torch.randn((c.experts,c.hidden),device="cuda") * .2).to(torch.bfloat16)
    layer = SimpleNamespace(router=_Router(weight),
        router_scale=(torch.rand((c.hidden,),device="cuda") + .5).to(torch.bfloat16),
        expert_scale=(torch.rand((c.experts,),device="cuda") + .5).to(torch.bfloat16))
    h = torch.randn((9,c.hidden),device="cuda",dtype=torch.bfloat16)
    picks,weights = model._route_ct(h,layer)
    assert picks.dtype == torch.int32 and weights.dtype == torch.float32
    assert picks.shape == weights.shape == (9,c.top_k)
    # Independent BF16 tensor-operation router input, FP64 logits and FP64 selection.
    root = torch.tensor(c.hidden**-.5,device="cuda",dtype=torch.bfloat16)
    router_input = (ct_rms_norm(h,None,c.eps) * root) * layer.router_scale
    logits = router_input.double() @ weight.double().T
    order = torch.argsort(logits,dim=-1,descending=True,stable=True)
    margin = (torch.gather(logits,1,order[:,c.top_k-1:c.top_k]) - torch.gather(logits,1,order[:,c.top_k:c.top_k+1]))
    assert torch.all(margin > 1e-3)        # no near-tie at the top-k boundary for this seed
    assert torch.equal(torch.sort(picks,-1).values,torch.sort(order[:,:c.top_k],-1).values.to(torch.int32))
    probabilities = torch.softmax(logits,dim=-1)
    selected = torch.gather(probabilities,1,picks.long())
    expected = selected/selected.sum(-1,keepdim=True) * layer.expert_scale[picks.long()].double()
    torch.testing.assert_close(weights.double(),expected,rtol=1e-4,atol=0)
    # Each row's routing is independent of the other rows of the call.
    for row in range(9):
        alone = model._route_ct(h[row:row+1],layer)
        assert torch.equal(alone[0],picks[row:row+1]) and torch.equal(alone[1],weights[row:row+1])
