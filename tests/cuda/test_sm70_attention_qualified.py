"""Installed qualified attention: actual-end R1 oracle, capture and scope fallback."""

import pytest
import torch

from tensorfold.cuda.sm70 import attention_native as native
from tensorfold.cuda.sm70 import attention_qualification as policy
from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.families.qwen4_exp.cuda.attention import AttnScratch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1, 3, 4, 24])
@pytest.mark.parametrize("kb,vb", [(0,0),(0,8),(8,0),(8,8)])
def test_qualified_capture_matches_original_each_query(rows, kb, vb):
    if torch.cuda.get_device_capability() != (7,0):
        pytest.skip("SM70 qualification")
    torch.manual_seed(901+rows)
    q=torch.randn((rows,3,256),device="cuda",dtype=torch.float16)*3
    k=torch.randn((8192,1,256),device="cuda",dtype=torch.float16)*3
    v=torch.randn_like(k)*.2
    ks=vs=None
    if kb:
        q=ops.h32(q).half()
        k,ks=ops.quantize_int8(k)
    if vb:
        v,vs=ops.quantize_int8(v)
    scratch=AttnScratch(rows,3,256,8192,"cuda",budget=2048,ratio=4,dtype=torch.float16)
    scratch.sparse.zero_()
    position=torch.tensor([128],device="cuda",dtype=torch.int32)
    def run():
        return native.attention(q,k,v,position,scratch,rows,.0625,scratch.out,8192,ks,vs,kb,vb)
    run()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    try:
        for end in (128,177,640,1025,2731,4097,5096,6145,8192):
            first=end-rows+1
            assert policy.supports(rows=rows,heads=3,kv_heads=1,head_dim=256,context=8192,
                                   first_end=first,identity=policy.runtime_identity(q.device),capability=(7,0))
            position.fill_(first-1)
            graph.replay()
            actual=scratch.out.clone()
            for row in range(rows):
                count=first+row
                kd=ops.dequantize_int8(k[:count],ks[:count]) if kb else k[:count]
                vd=ops.dequantize_int8(v[:count],vs[:count]) if vb else v[:count]
                probability=(ops._matmul32(q[row],kd[:,0].T)*.0625).softmax(-1).half()
                expected=ops._matmul32(probability,vd[:,0])
                if vb:
                    expected=ops.h32(expected)
                assert torch.equal(actual[row],expected.half()),(rows,kb,vb,end,row)
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("end,heads", [(64,3),(8193,3),(128,6)])
def test_unqualified_scope_uses_original_torch_without_native(monkeypatch,end,heads):
    monkeypatch.setattr(ops,"_NATIVE_FAMILY",True)
    monkeypatch.setattr(ops,"_NATIVE_ATTENTION",True)
    def forbidden(*args,**kwargs):
        raise AssertionError("unqualified operator entered native attention")
    monkeypatch.setattr(native,"attention",forbidden)
    q=torch.randn((1,heads,256),device="cuda",dtype=torch.float16)
    k=torch.randn((end,1,256),device="cuda",dtype=torch.float16)
    v=torch.randn_like(k)
    position=torch.tensor([end-1],device="cuda",dtype=torch.int32)
    scratch=AttnScratch(1,heads,256,end,"cuda",budget=0,ratio=4,dtype=torch.float16)
    expected=ops.attention(q,k,v,position,scratch,1,.0625,context=end,_native_allowed=False).clone()
    actual=ops.attention(q,k,v,position,scratch,1,.0625,context=end)
    assert torch.equal(actual,expected)
