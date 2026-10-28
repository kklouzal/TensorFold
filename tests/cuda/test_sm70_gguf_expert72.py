"""R48/64/72 original-kernel exactness with genuine checkpoint expert banks."""
import pytest
import torch

from test_sm70_gguf_exact_actual import MODEL, make
from test_sm70_gguf_exact import bits
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts
from tensorfold.cuda.sm70 import gguf_ops

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or not MODEL.is_file(),reason="actual SM70/checkpoint required")


@pytest.fixture(scope="module",params=[0,2])
def actual_large_bank(request):
    bank,metadata=make(request.param,8)
    bank.packed_rows=72
    bank.implementation="packed_exact"
    return bank,metadata


@pytest.mark.parametrize("rows",[48,64,72])
@pytest.mark.parametrize("dtype",[torch.float16,torch.float32])
@pytest.mark.parametrize("index_dtype",[torch.int32,torch.int64])
def test_actual_large_rows_keep_original_bmm_bits(actual_large_bank,rows,dtype,index_dtype):
    bank,_=actual_large_bank
    torch.manual_seed(9211+rows)
    x=(torch.randn(rows,2560,device="cuda")*.1).half()
    picks=(torch.arange(rows*11,device="cuda",dtype=index_dtype).reshape(rows,11)+rows).remainder(10)
    picks[:,-1]=10
    expected=bank.forward_reference(x,picks).to(dtype)
    actual=torch.empty_like(expected)
    assert bank.forward(x,picks,actual) is actual
    bits(actual,expected)


@pytest.mark.parametrize("rows",[64,72])
@pytest.mark.parametrize("dtype",[torch.float16,torch.float32])
@pytest.mark.parametrize("index_dtype",[torch.int32,torch.int64])
def test_large_capture_badids_shared_zero_alias_and_dynamic_routes(actual_large_bank,rows,dtype,index_dtype):
    bank,_=actual_large_bank
    torch.manual_seed(8421+rows)
    x=(torch.randn(rows,2560,device="cuda")*.1).half()
    picks=torch.arange(rows*11,device="cuda",dtype=index_dtype).reshape(rows,11).remainder(11)
    out=torch.empty((rows,11,2560),device="cuda",dtype=dtype)
    def run():gguf_ops.experts_exact(x,picks,bank.gate,bank.up,bank.down,bank.shared,out)
    run()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    picks.copy_(picks.roll(1,1))
    picks[0,0]=-1
    picks[-1,0]=2**30 if index_dtype==torch.int32 else 2**40
    graph.replay()
    valid=(picks>=0)&(picks<10)
    shared=picks==10
    # Use an independent reference call only for valid inputs. Invalid IDs keep
    # their original NaN containment contract, not a cross-runtime payload claim.
    sanitized=picks.clone()
    sanitized[~(valid|shared)]=0
    expected=bank.forward_reference(x,sanitized).to(dtype)
    bits(out[valid],expected[valid])
    bits(out[shared],torch.zeros_like(out[shared]))
    assert bool(torch.isnan(out[~(valid|shared)]).all())
    with pytest.raises(RuntimeError,match="overlap"):
        bank.forward_exact(x,picks[:,:1].contiguous(),x[:,None])
    valid_picks=sanitized
    bits(bank.forward(x,valid_picks).to(dtype),bank.forward_reference(x,valid_picks).to(dtype))


def test_existing_code_attributes_are_loaded_before_large_row_work():
    info=gguf_ops.extension().prepare_experts_exact72()
    assert info[0]==22 and info[4]>=128 and info[5]==70


def test_reference_fallback_above_private_cap_is_retained(actual_large_bank):
    bank,_=actual_large_bank
    for rows in [96,256]:
        x=(torch.randn(rows,2560,device="cuda")*.1).half()
        picks=torch.full((rows,1),10,device="cuda",dtype=torch.int32)
        bits(bank.forward(x,picks),bank.forward_reference(x,picks))
    assert GGUFExperts.packed_rows==32  # Fixture's row cap is instance-owned only.
