"""Contiguous tail cuts versus the retained independent c928 indexed program."""
from types import SimpleNamespace as N

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.compact_indexer import Cache, Span
from compact_tail_reference import indexed_tail


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def case(count, shift, *, base=False, strided=False, device='cpu'):
    length = 12 + count
    start = max(0, 12 + shift)
    rows = max(1, length - start + 2)
    entry = Cache(128, 128, device, torch.zeros(2,dtype=torch.int64,device=device),
                  torch.tensor([0,-2],dtype=torch.int32,device=device), layer=3, proposal_rows=32, mtp=True)
    # Quiet/signaling/negative NaNs, signed zero and subnormal payloads. No
    # floating arithmetic initializes these representation-sensitive values.
    codes = torch.tensor([0x7c01,0x7e35,0xfeee,0x8000,0x0001,0x3c00],dtype=torch.int32,device=device).to(torch.int16)
    def values(n):
        raw = codes.repeat((n * 256 + len(codes)-1)//len(codes))[:n*256].reshape(n,256).view(torch.float16)
        return raw[:,::2] if strided else raw[:,:128].contiguous()
    raw = values(rows)
    coords = torch.arange(rows*6,dtype=torch.int64,device=device).reshape(rows,6)
    coords[:,0] = -(1<<63)
    coords[:,2] = (1<<63)-1
    coords = coords[:,::2] if strided else coords[:,:3].contiguous()
    entry.protected.copy_(values(4))
    entry.protected_coordinates.copy_(torch.tensor([[-(1<<63),(1<<63)-1,91+i] for i in range(4)],device=device))
    entry.raw.fill_(13)
    entry.coordinates.fill_(27)
    entry.n = start + rows
    if base:
        entry.base,entry.proposal_count,entry.base_expected = start,rows,(0,0)
        entry.base_raw.copy_(entry.protected)
        entry.base_coordinates.copy_(entry.protected_coordinates)
        entry.proposal_raw[:rows].copy_(raw)
        entry.proposal_coordinates[:rows].copy_(coords)
    else:
        entry.span = Span(start,raw,coords,None,0,False,(0,0),resolved=start+rows)
    return entry,length


@pytest.mark.parametrize('count',[1,2,3])
@pytest.mark.parametrize('shift',[-7,0,1,2,3])
@pytest.mark.parametrize('base',[False,True])
@pytest.mark.parametrize('strided',[False,True])
def test_all_legal_prefix_suffix_cuts_preserve_bits_and_inputs(count,shift,base,strided):
    entry,length=case(count,min(shift,count),base=base,strided=strided)
    before=[t.clone() for t in (entry.raw,entry.coordinates,entry.protected,entry.protected_coordinates)]
    expected=indexed_tail(entry,length)
    actual=entry.tail(length)
    for left,right in zip(actual,expected,strict=True):
        exact(left,right)
    for value,old in zip((entry.raw,entry.coordinates,entry.protected,entry.protected_coordinates),before,strict=True):
        exact(value,old)
    assert actual[0].data_ptr()!=entry.raw.data_ptr() and actual[1].data_ptr()!=entry.coordinates.data_ptr()
    assert torch.count_nonzero(actual[0][count:].view(torch.int16))==0
    assert torch.count_nonzero(actual[1][count:])==0


def outcome(function, entry, length):
    try:
        value=function(entry,length)
        return value,None
    except Exception as error:
        return None,(type(error),str(error))


@pytest.mark.parametrize('bad',['empty-raw','empty-coordinates','float-raw','float-coordinates','rejected','epoch','big-int'])
def test_original_eager_errors_and_precedence_remain_authoritative(bad):
    entry,length=case(2,2)
    if bad=='empty-raw':
        entry.span=Span(length,torch.empty(0,128).half(),torch.empty(0,3,dtype=torch.int64),None,0,False,(0,0))
    elif bad=='empty-coordinates':
        entry.span.coordinates=torch.empty(0,3,dtype=torch.int64)
    elif bad=='float-raw':
        entry.span.raw=entry.span.raw.float()
    elif bad=='float-coordinates':
        entry.span.coordinates=entry.span.coordinates.float()
    elif bad=='rejected':
        entry.span.resolved=length-1
    elif bad=='epoch':
        entry.span.staging=N(target_epoch=2,mtp_epoch=2)
        entry.span.epoch=1
    else:
        entry.capacity=1<<65
        length=(1<<63)+2
        entry.span=Span(length,torch.zeros(1,128).half(),torch.zeros(1,3,dtype=torch.int64),None,0,False,(0,0))
    _,expected=outcome(lambda c,n:indexed_tail(c,n),entry,length)
    _,actual=outcome(lambda c,n:c.tail(n),entry,length)
    assert expected is not None and actual==expected


def test_clone_and_aligned_zero_paths_stay_exact():
    entry,_=case(2,0)
    for length in (entry.n,8):
        actual,expected=entry.tail(length),indexed_tail(entry,length)
        for a,b in zip(actual,expected,strict=True):
            exact(a,b)


def test_autograd_and_subclasses_keep_original_indexed_path():
    entry,length=case(2,0)
    entry.span.raw.requires_grad_(True)
    assert not Cache._tail_slices_supported(entry,entry.protected,entry.protected_coordinates,
                                           entry.span.raw,entry.span.coordinates,entry.span.start,length)
    class SubCache(Cache):
        def _tail_slices_supported(self,*args):
            raise AssertionError('new private helper must not collide with subclass override')
    entry.span.raw.requires_grad_(False)
    entry.__class__=SubCache
    for a,b in zip(entry.tail(length),indexed_tail(entry,length),strict=True):
        exact(a,b)


@pytest.mark.parametrize('retirement_error',[False,True])
def test_partial_copy_failure_keeps_read_lease_and_original_failure(monkeypatch,retirement_error):
    entry,length=case(3,1)
    primary=MemoryError('second tail copy')
    secondary=RuntimeError('tail read retirement')
    calls=[]
    original=torch.Tensor.copy_
    def copy(target,*args,**kwargs):
        calls.append('copy')
        if len(calls)==2:
            raise primary
        return original(target,*args,**kwargs)
    def done(tensors,**kwargs):
        calls.append(('retire',len(tensors),kwargs['span']))
        if retirement_error:
            raise secondary
    monkeypatch.setattr(torch.Tensor,'copy_',copy)
    monkeypatch.setattr(entry,'_read_done',done)
    before_n,before_resolved=entry.n,entry.span.resolved
    with pytest.raises(MemoryError) as caught:
        entry.tail(length)
    assert caught.value is primary and calls[-1]==('retire',4,entry.span)
    assert entry.n==before_n and entry.span.resolved==before_resolved
    if retirement_error:
        assert entry._retirement_error is primary


def test_negative_view_metadata_keeps_original_indexed_path():
    entry,length=case(2,0)
    entry.span.raw=torch._neg_view(entry.span.raw)
    assert entry.span.raw.is_neg()
    assert not Cache._tail_slices_supported(entry,entry.protected,entry.protected_coordinates,
                                           entry.span.raw,entry.span.coordinates,entry.span.start,length)
    for a,b in zip(entry.tail(length),indexed_tail(entry,length),strict=True):
        exact(a,b)
