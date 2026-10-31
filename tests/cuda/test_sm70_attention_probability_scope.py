"""All29 archived softmax tests retained as qualified exact or explicit fallback controls."""

from unittest.mock import patch

import pytest
import torch

from tensorfold.cuda.sm70 import attention_native, attention_qualification as policy, family_ops as ops
from test_sm70_attention_native import inputs, query_reference, reference


pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7,0), reason="SM70 probability scope")


def qualified(q,key,position,rows,context):
    return policy.supports(rows=rows,heads=q.shape[1],kv_heads=key.shape[1],head_dim=q.shape[2],
        context=context,first_end=int(position.cpu()[0])+1,identity=policy.runtime_identity(q.device),
        capability=torch.cuda.get_device_capability(q.device))


def original_dispatch(q,key,value,position,scratch,rows,context,ks=None,vs=None,kb=0,vb=0):
    expected=reference(q,key,value,position,scratch,rows,1/16,context=context,
                       ks=ks,vs=vs,k_bits=kb,k_codec=0,v_bits=vb,v_codec=0).clone()

    def forbidden(*args,**kwargs):
        raise AssertionError("unqualified frame entered native attention")

    with patch.object(ops,"_NATIVE_FAMILY",True), patch.object(ops,"_NATIVE_ATTENTION",True), \
            patch.object(attention_native,"attention",forbidden):
        actual=ops.attention(q,key,value,position,scratch,rows,1/16,context=context,
                            ks=ks,vs=vs,k_bits=kb,k_codec=0,v_bits=vb,v_codec=0)
    assert torch.equal(actual,expected)


@pytest.mark.parametrize("length", [33,65,127,128,129,257,511,1023,2048])
def test_archived_short_probability_cases_keep_exact_or_original_scope(length):
    q,key,value,position,scratch,_,_=inputs(length+32,rows=3)
    position.fill_(length-3)
    if not qualified(q,key,position,3,length):
        original_dispatch(q,key,value,position,scratch,3,length)
        assert not hasattr(scratch,"native_workspace")
        return
    attention_native.attention(q,key,value,position,scratch,3,1/16,scratch.out,length,None,None,0,0)
    workspace=scratch.native_workspace
    for row in range(3):
        end=length-2+row
        expected=workspace.scores[row,:,:end].softmax(-1).half()
        assert torch.equal(workspace.probabilities[row,:,:end],expected)


@pytest.mark.parametrize("length", [2048,2049,2050,4095,4096,8191,8192,9216])
@pytest.mark.parametrize("sparse", [False,True])
def test_archived_register_probability_cases_keep_exact_or_original_scope(length,sparse):
    q,key,value,position,scratch,_,_=inputs(length+32,rows=3,sparse=sparse)
    position.fill_(length-3)
    if sparse:
        for row in range(3):
            count=min(scratch.budget,length-2+row)
            scratch.nk[row]=count
            scratch.ids[row,:count]=torch.arange(count,device="cuda",dtype=torch.int32)
    if not qualified(q,key,position,3,length):
        with pytest.raises(ValueError,match="qualified"):
            attention_native.attention(q,key,value,position,scratch,3,1/16,scratch.out,length,None,None,0,0)
        original_dispatch(q,key,value,position,scratch,3,length)
        return
    attention_native.attention(q,key,value,position,scratch,3,1/16,scratch.out,length,None,None,0,0)
    workspace=scratch.native_workspace
    for row in range(3):
        end=length-2+row
        expected=workspace.scores[row,:,:end].softmax(-1).half()
        assert torch.equal(workspace.probabilities[row,:,:end],expected)


@pytest.mark.parametrize("kb,vb", [(0,0),(8,8),(0,8),(8,0)])
def test_archived_regime_transitions_capture_only_qualified_bucket_and_keep_long_fallback(kb,vb):
    q,key,value,position,scratch,ks,vs=inputs(16384,kb=kb,vb=vb)

    def run(context):
        return attention_native.attention(q,key,value,position,scratch,3,1/16,scratch.out,context,ks,vs,kb,vb)

    position.fill_(2045)
    assert not qualified(q,key,position,3,16384)
    with pytest.raises(ValueError,match="qualified"):
        run(16384)
    assert not hasattr(scratch,"native_workspace")
    run(8192)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(8192)
    try:
        for start in (2045,2046,2047,2048,4093,4094,8190,9213,9214,9215,9216):
            position.fill_(start)
            context=start+3
            if not qualified(q,key,position,3,context):
                # A cached8192key never permits replay at an unsupported end.
                original_dispatch(q,key,value,position,scratch,3,context,ks,vs,kb,vb)
                continue
            expected=query_reference(q,key,value,position,scratch,3,1/16,ks,vs,kb,vb)
            eager=run(context).clone()
            probabilities=scratch.native_workspace.probabilities[:,:,:context].clone()
            graph.replay()
            assert torch.equal(scratch.out,eager) and torch.equal(scratch.out,expected)
            for row in range(3):
                end=start+row+1
                assert torch.equal(scratch.native_workspace.probabilities[row,:,:end],probabilities[row,:,:end])
    finally:
        torch.cuda.synchronize()
        graph.reset()
