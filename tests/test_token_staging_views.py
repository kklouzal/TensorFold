"""Cached CPU staging borrowers retain original ordering, conversion and fences."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from tensorfold.families.qwen4_exp.cuda import forward,mtp


class Event:
    def __init__(self):self.operations=[]
    def synchronize(self):self.operations.append('wait')
    def record(self):self.operations.append('record')


def buffers(rows):
    b=NS(rows=rows,ids_host=torch.full((rows,),-77,dtype=torch.int32),ids=torch.zeros(rows,dtype=torch.int32),
        last_host=torch.full((rows,),-78,dtype=torch.int32),last=torch.zeros(rows,dtype=torch.int32),
        staged=Event(),mtp_in=torch.zeros(rows,2),mtp_indexer_position=None)
    b.ids_host_array,b.last_host_array=b.ids_host.numpy(),b.last_host.numpy()
    return b


@pytest.mark.parametrize('sizes',[(1,),(1,)*8,(256,),(5,)*24,(1,)*256,(256,)*16])
def test_target_and_mtp_payloads_match_original_conversion_and_keep_suffix(sizes,monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import compact_indexer
    monkeypatch.setattr(compact_indexer,'stage_text_indexer_frame',lambda *a:None)
    count=sum(sizes);b=buffers(count+1);windows=[];at=0
    for size in sizes:
        state=NS(pos=0,capacity=count+1,mtp_len=0,media_input_span=None)
        tokens=list(range(at,at+size));windows.append((state,tokens));at+=size
    array=b.ids_host_array;segs=forward._stage(NS(layers=[]),b,windows)
    expected=np.asarray([t for _,tokens in windows for t in tokens],dtype=np.int32)
    assert np.array_equal(array[:count],expected) and b.ids_host[count]==-77
    assert b.ids_host_array is array and b.staged.operations==['wait','record']
    method=buffers(count+1);ids_array,last_array=method.ids_host_array,method.last_host_array
    streams=[torch.ones(size,2) for size in sizes]
    actual=mtp._mtp_stage(method,[(state,tokens,stream)for(state,tokens),stream in zip(windows,streams)])
    assert actual==segs and np.array_equal(ids_array[:count],expected)
    assert last_array[:len(sizes)].tolist()==[end-1 for _,_,end in segs]
    assert method.ids_host[count]==-77 and method.last_host[len(sizes)]==-78
    assert method.ids_host_array is ids_array and method.last_host_array is last_array
    assert method.staged.operations==['wait','record']


def test_mtp_retains_int_coercion_before_staged_write_and_alias_snapshot_order():
    b=buffers(4);state=NS(mtp_len=0,capacity=4,media_input_span=None)
    class Token:
        def __int__(self):
            assert b.staged.operations==['wait']
            return 19
    source=b.mtp_in[:2];source.copy_(torch.tensor([[1.,2.],[3.,4.]]))
    expected=source.clone()
    mtp._mtp_stage(b,[(state,[Token(),Token()],source)])
    assert b.ids[:2].tolist()==[19,19] and torch.equal(b.mtp_in[:2],expected)
    assert b.staged.operations==['wait','record']
