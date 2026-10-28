"""Strict pure-region graph/eager oracle on actual513-router/11-selected-slot GGUF geometry."""
from types import SimpleNamespace

import pytest
import torch

from gguf_fixture import blocks
from tensorfold.cuda import moe
from tensorfold.cuda.sm70 import family_ops,routing_native
from tensorfold.families.qwen4_exp.cuda.forward import moe_block
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense,GGUFExperts,GGUFPacked
from tensorfold.families.qwen4_exp.cuda.moe_region_graphs import MoERegion

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(7,0),
                            reason='actual SM70 capture component')


@torch.inference_mode()
def test_actual513_router_11slots_r32_pure_region_replay_preserves_original_bits(monkeypatch):
    # Owned synthetic packed bank; independent native arithmetic controls remain
    # unchanged. This oracle isolates capture/replay/input/route ownership.
    rows,dims,width,experts=32,2560,80,512
    packed=[]
    for seed,n,k,crop in ((811,experts*width,dims,0),(812,experts*width,dims,0),
                           (813,experts*dims,256,16)):
        raw=blocks(12,n,k,seed=seed)
        packed.append(GGUFPacked(torch.tensor(raw,device='cuda'),12,raw.shape[0],
                                  width if crop else k,crop))
    torch.manual_seed(814)
    shared=[GGUFDense(torch.randn(shape,device='cuda',dtype=torch.float16)*.01)
            for shape in ((width,dims),(width,dims),(dims,width))]
    # Match the frozen batch/GDN execution policy; keep the old rowwise router.
    assert routing_native._NATIVE_LOGITS is False
    monkeypatch.setattr(family_ops,'_NATIVE_FAMILY',True)
    monkeypatch.setattr(GGUFExperts,'implementation','packed_exact')
    monkeypatch.setattr(GGUFDense,'implementation','native32')
    bank=GGUFExperts(*packed,shared,count=experts,width=width,dims=dims)
    cfg=SimpleNamespace(experts=experts,top_k=10,num_experts=experts,num_experts_per_tok=10,
                        hidden_size=dims,moe_intermediate_size=width,execution_dtype='float16')
    weights=SimpleNamespace(cfg=cfg,comm=None)
    layer=SimpleNamespace(index=0,moe=SimpleNamespace(experts=bank,
                          router=torch.randn((experts+1,dims),device='cuda',dtype=torch.float32)*.01))
    buffers=SimpleNamespace(prefill=False,mixed=torch.randn((rows,dims),device='cuda',dtype=torch.float16)*.03,
                            moe=moe.MoEBuffers(rows,cfg,'cuda',prefill=False))
    assert layer.moe.router.shape==(513,2560)
    assert buffers.moe.pick.shape==(32,11) and buffers.moe.y.shape==(32,11,2560)
    buffers.mixed[0].zero_()  # Exact router ties must retain lower-id ordering.
    region=MoERegion(weights,layer,buffers,rows)
    primary=None
    try:
        region.prepare()
        for scale in (1.0,-.5,.125):
            buffers.mixed.mul_(scale)
            moe_block(layer,weights,buffers,rows)
            expected=[tensor.clone() for tensor in (buffers.moe.logits,buffers.moe.pick,
                                                    buffers.moe.wts,buffers.moe.y)]
            region.replay()
            torch.cuda.synchronize()
            for actual,reference in zip((buffers.moe.logits,buffers.moe.pick,buffers.moe.wts,buffers.moe.y),expected):
                torch.testing.assert_close(actual.view(torch.uint8),reference.view(torch.uint8),rtol=0,atol=0)
        assert region.hits==3
    except BaseException as error:
        primary=error
    try:
        region.close()
    except BaseException as error:
        if primary is not None:
            raise primary from error
        raise
    if primary is not None:
        raise primary
    assert region.closed
