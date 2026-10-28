"""Strict SM70 tail bytes and cache publication; no throughput/quality claim."""
import pytest
import torch

from compact_tail_reference import indexed_tail
from test_compact_tail_slices import case, exact
from tensorfold.families.qwen4_exp.cuda.compact_indexer import Cache
from tensorfold.families.qwen4_exp.cuda.capture_lifetime import on_stream

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(7,0),
                             reason='actual SM70 required; skip is not acceptance')


@torch.inference_mode()
def test_all_legal_tail_cuts_are_bitwise_indexed_reference_with_extreme_payloads():
    for count in (1,2,3):
        for shift in (-7,0,1,2,3):
            for base in (False,True):
                for strided in (False,True):
                    entry,length=case(count,min(shift,count),base=base,strided=strided,device='cuda')
                    tensors=[v for v in vars(entry).values() if isinstance(v,torch.Tensor)]
                    if entry.span is not None:
                        tensors += [entry.span.raw,entry.span.coordinates]
                    before=[v.clone() for v in tensors]
                    expected=indexed_tail(entry,length)
                    actual=entry.tail(length)
                    for left,right in zip(actual,expected,strict=True):
                        exact(left,right)
                    for value,old in zip(tensors,before,strict=True):
                        exact(value,old)
                    assert torch.count_nonzero(actual[0][count:].view(torch.int16))==0
                    assert torch.count_nonzero(actual[1][count:])==0
    torch.cuda.synchronize()


@torch.inference_mode()
def test_cross_stream_tail_retires_all_inputs_and_snapshot_commit_preserves_bytes():
    entry,length=case(3,1,device='cuda')
    # Use finite inputs for the existing native publication contract; payload
    # preservation itself is independently covered by the first case.
    entry.protected.fill_(.25)
    entry.span.raw.copy_(torch.arange(entry.span.raw.numel(),device='cuda').reshape_as(entry.span.raw).half())
    expected=indexed_tail(entry,length)
    entry._mark()  # Seeded external inputs are now issued on this source stream.
    side=torch.cuda.Stream()
    actual=on_stream(torch.cuda,side,lambda:entry.tail(length))
    entry.wait_native_reuse()  # Original stream joins the side's recorded reader.
    for a,b in zip(actual,expected,strict=True):
        exact(a,b)
    plan=entry.prepare_commit(length,(0,0))
    entry.publish_commit(plan,length,(0,0))
    torch.cuda.synchronize()
    assert entry.n==length and entry.span.resolved==length and entry.status.tolist()==[0,-2]
    snapshot=entry.snapshot(length)
    for key,expected_value in zip(('raw','coordinates'),expected,strict=True):
        exact(snapshot[key],expected_value)
    restored=Cache(128,128,'cuda',torch.zeros(2,dtype=torch.int64,device='cuda'),
                   torch.tensor([0,-2],dtype=torch.int32,device='cuda'),layer=3)
    restored.restore(snapshot,length,(0,0))
    torch.cuda.synchronize()
    assert restored.n==length
    exact(restored.raw,expected[0])
    exact(restored.coordinates,expected[1])
