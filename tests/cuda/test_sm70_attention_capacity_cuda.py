"""A full native KV promise retains only the qualified attention workspace."""

from copy import copy
import gc
import json
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import attention_native as native, family_ops as ops

FULL = 262144
LIMIT = 768 << 20


@pytest.fixture(scope="module", autouse=True)
def bounded_allocator():
    previous = torch.cuda.get_per_process_memory_fraction()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.set_per_process_memory_fraction(min(previous,
        LIMIT / torch.cuda.get_device_properties(0).total_memory))
    try:
        yield
    finally:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(previous)


def scratch(rows):
    return SimpleNamespace(capacity=FULL, qsa=False,
        out=torch.empty((rows, 3, 256), device="cuda", dtype=torch.float16),
        sparse=torch.zeros(rows, device="cuda", dtype=torch.int32),
        nk=torch.zeros(rows, device="cuda", dtype=torch.int32),
        ids=torch.zeros((rows, 2052), device="cuda", dtype=torch.int32))


def expected(q, key, value, pos, owner, ks, vs, kb, vb):
    result=[]
    for row in range(q.shape[0]):
        view=copy(owner)
        view.sparse,view.nk,view.ids=owner.sparse[row:row+1],owner.nk[row:row+1],owner.ids[row:row+1]
        end=pos+row+1
        result.append(ops.attention(q[row:row+1],key[:end],value[:end],
            torch.tensor([pos+row],device="cuda",dtype=torch.int32),view,1,0.0625,
            out=torch.empty_like(q[row:row+1]),context=end,ks=None if ks is None else ks[:end],
            vs=None if vs is None else vs[:end],k_bits=kb,k_codec=0,v_bits=vb,v_codec=0,
            _native_allowed=False))
    return torch.cat(result)


@pytest.mark.parametrize("kb,vb",[(0,0),(8,8),(0,8),(8,0)])
def test_large_cache_capped_workspace_exact_query_oracle_and_capture(kb,vb):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(926)
    rows=3
    q=torch.randn((rows,3,256),device="cuda",dtype=torch.float16)
    key=torch.zeros((FULL,1,256),device="cuda",dtype=torch.int8 if kb else torch.float16)
    value=torch.zeros((FULL,1,256),device="cuda",dtype=torch.int8 if vb else torch.float16)
    raw_key=torch.randn((8192,1,256),device="cuda",dtype=torch.float16)
    raw_value=torch.randn_like(raw_key)*0.2
    ks=vs=None
    if kb:
        codes,scale=ops.quantize_int8(raw_key)
        key[:8192].copy_(codes)
        ks=torch.zeros((FULL,1,8),device="cuda",dtype=torch.float16)
        ks[:8192].copy_(scale)
        q=ops.h32(q).half()
    else:
        key[:8192].copy_(raw_key)
    if vb:
        codes,scale=ops.quantize_int8(raw_value)
        value[:8192].copy_(codes)
        vs=torch.zeros((FULL,1,8),device="cuda",dtype=torch.float16)
        vs[:8192].copy_(scale)
    else:
        value[:8192].copy_(raw_value)
    owner=scratch(rows)
    position=torch.tensor([64],device="cuda",dtype=torch.int32)
    def run(context=8192):
        return native.attention(q,key,value,position,owner,rows,0.0625,owner.out,context,ks,vs,kb,vb)
    for pos in (64,127,639,8189):
        position.fill_(pos)
        authority=expected(q,key,value,pos,owner,ks,vs,kb,vb)
        assert torch.equal(run(),authority)
    workspace=owner.native_workspace
    assert workspace.capacity==8192 and workspace.scores.shape[-1]==8192
    assert sum(t.numel()*t.element_size() for t in (workspace.scores,workspace.probabilities,workspace.partials))==native.workspace_bytes(rows,3,256,FULL)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    try:
        for pos in (127,639):
            position.fill_(pos)
            authority=expected(q,key,value,pos,owner,ks,vs,kb,vb)
            graph.replay()
            assert torch.equal(owner.out,authority)
        with pytest.raises(ValueError,match="qualified"):
            run(8193)
        assert owner.native_workspace is workspace
        if (kb,vb)==(0,0):
            bad=torch.empty((rows,3,22,256),device="cuda",dtype=torch.float32)
            with pytest.raises(RuntimeError,match="workspace geometry"):
                native.extension().run(q,key,value,position,owner.sparse,owner.nk,owner.ids,
                    workspace.scores,workspace.probabilities,bad,owner.out,None,None,
                    rows,2731,kb,vb,False,0.0625)
    finally:
        torch.cuda.synchronize()
        graph.reset()
    assert torch.cuda.max_memory_allocated()<LIMIT


@pytest.mark.parametrize("rows",[1,127,256])
def test_largest_original_stable_sort_branch_stays_below_service_reserve(rows):
    # QSA's full native-window fallback uses ceil(262144/4) block scores.
    # These cover ATen's large-segment/full-sort/segmented-sort branches.
    gc.collect()
    torch.cuda.empty_cache()
    scores=torch.zeros((rows,65536),device="cuda",dtype=torch.float32)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    chosen=torch.argsort(scores,dim=-1,descending=True,stable=True)
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()
    print(json.dumps({"sort_rows":rows,"block_count":65536,"peak_allocated":peak}),flush=True)
    assert chosen.shape==scores.shape
    assert peak<LIMIT
