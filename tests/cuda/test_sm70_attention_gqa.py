"""Grouped attention scope after retiring the unqualified packed-PV experiment.

The complete former experiment and273 original cases remain byte-for-byte in
task-artifacts/v100-optimize/family/packed-gqa-candidate. That experiment proved
parity with its old native control, not with original Torch/full-model quality.
These maintained controls cover every former shape combination through the
qualified native or original-Torch dispatcher, without restoring that provider.
Full-forward fallback state is covered by test_flashnext_attention_fallback;
qualified captures retain an independent original per-query Torch oracle.
"""

import pytest
import torch

from tensorfold.cuda.sm70 import attention_native, family_ops as ops
from test_sm70_attention_native import inputs, query_reference, reference


pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 attention scope")


def test_retired_packed_gqa_workspace_is_not_a_runtime_capability():
    assert not hasattr(attention_native, "GQAPackedWorkspace")


@pytest.mark.parametrize("group", [1, 3, 6, 12, 16])
@pytest.mark.parametrize("rows", [1, 3, 18])
@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("kb,vb", [(0,0),(8,8),(0,8),(8,0)])
@pytest.mark.parametrize("sparse", [False, True])
def test_all_legacy_group_shapes_have_explicit_native_or_original_torch_route(monkeypatch, group, rows, dim, kb, vb, sparse):
    kvheads = 2 if group in (1, 6) and dim == 128 else 1
    q, key, value, position, scratch, ks, vs = inputs(512, rows=rows, dim=dim,
        qheads=group*kvheads, kvheads=kvheads, kb=kb, vb=vb, sparse=sparse)
    scale = dim**-0.5
    qualified = group == 3 and kvheads == 1 and dim == 256
    if qualified:
        expected = query_reference(q,key,value,position,scratch,rows,scale,ks,vs,kb,vb)
    else:
        expected = reference(q,key,value,position,scratch,rows,scale,context=512,
                             ks=ks,vs=vs,k_bits=kb,k_codec=0,v_bits=vb,v_codec=0).clone()
        with pytest.raises(ValueError, match="qualified"):
            attention_native.attention(q,key,value,position,scratch,rows,scale,scratch.out,512,ks,vs,kb,vb)
        assert not hasattr(scratch,"native_workspace")

        def forbidden(*args,**kwargs):
            raise AssertionError("unsupported grouped shape entered native attention")

        monkeypatch.setattr(attention_native,"attention",forbidden)
    monkeypatch.setattr(ops,"_NATIVE_FAMILY",True)
    monkeypatch.setattr(ops,"_NATIVE_ATTENTION",True)
    actual = ops.attention(q,key,value,position,scratch,rows,scale,context=512,
                          ks=ks,vs=vs,k_bits=kb,k_codec=0,v_bits=vb,v_codec=0)
    assert torch.equal(actual,expected)


@pytest.mark.parametrize("group", [3,12])
@pytest.mark.parametrize("rows", [1,3,18])
@pytest.mark.parametrize("kb,vb", [(0,0),(8,8),(0,8),(8,0)])
def test_legacy_capture_shapes_are_qualified_or_refused_before_native_work(group,rows,kb,vb):
    q,key,value,position,scratch,ks,vs = inputs(8192,rows=rows,qheads=group,kb=kb,vb=vb,sparse=True)
    scratch.nk.fill_(64)
    scratch.ids[:,:64] = torch.arange(64,device="cuda",dtype=torch.int32)
    position.fill_(125)
    if group != 3:
        with pytest.raises(ValueError,match="qualified"):
            attention_native.attention(q,key,value,position,scratch,rows,1/16,scratch.out,8192,ks,vs,kb,vb)
        assert not hasattr(scratch,"native_workspace")
        return

    def run(context):
        return attention_native.attention(q,key,value,position,scratch,rows,1/16,scratch.out,context,ks,vs,kb,vb)

    run(8192)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(8192)
    try:
        for start in (125,2047,2048,4095,8192-rows):
            position.fill_(start)
            expected=query_reference(q,key,value,position,scratch,rows,1/16,ks,vs,kb,vb)
            graph.replay()
            assert torch.equal(scratch.out,expected)
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("sparse", [False,True])
@pytest.mark.parametrize("dim", [128,256])
def test_unsupported_grouped_original_torch_still_masks_nonfinite_future(sparse,dim,monkeypatch):
    q,key,value,position,scratch,ks,vs = inputs(256,rows=3,dim=dim,qheads=12)
    position.fill_(63)
    key[66:].fill_(torch.nan)
    value[66:].fill_(torch.nan)
    if sparse:
        scratch.sparse.fill_(1)
        scratch.nk.fill_(4)
        scratch.ids[:,:4] = torch.tensor([0,2,4,6],device="cuda",dtype=torch.int32)
        key[1].fill_(torch.nan)
        value[1].fill_(torch.nan)
    expected=reference(q,key,value,position,scratch,3,dim**-0.5,context=256).clone()
    monkeypatch.setattr(ops,"_NATIVE_FAMILY",True)
    monkeypatch.setattr(ops,"_NATIVE_ATTENTION",True)
    actual=ops.attention(q,key,value,position,scratch,3,dim**-0.5,context=256)
    assert actual.isfinite().all() and torch.equal(actual,expected)


def test_group_above_old_packed_fragment_refuses_native_and_keeps_original_torch(monkeypatch):
    q,key,value,position,scratch,ks,vs = inputs(128,rows=1,dim=128,qheads=17)
    with pytest.raises(ValueError,match="qualified"):
        attention_native.attention(q,key,value,position,scratch,1,128**-0.5,scratch.out,128,ks,vs,0,0)
    expected=reference(q,key,value,position,scratch,1,128**-0.5,context=128).clone()
    monkeypatch.setattr(ops,"_NATIVE_FAMILY",True)
    monkeypatch.setattr(ops,"_NATIVE_ATTENTION",True)
    actual=ops.attention(q,key,value,position,scratch,1,128**-0.5,context=128)
    assert torch.equal(actual,expected)
