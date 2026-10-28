"""Private API/resource controls; hardware arithmetic is separately unverified."""
import ast
from pathlib import Path
import pytest
from tensorfold.cuda.sm70 import router_blas


@pytest.mark.parametrize("rows",[1,2,8,16,24,32])
def test_prepared_pointer_router_storage_is_fixed_and_no_weight_replication(rows):
    assert router_blas.workspace_bytes(rows)==rows*((2560+516)*4+24)
    assert router_blas.HOST_BYTES_PER_PLAN==16384


@pytest.mark.parametrize("rows",[0,33,-1,True,1.5])
def test_bad_router_capacity_refuses_before_extension_or_allocation(rows,monkeypatch):
    monkeypatch.setattr(router_blas,"extension",lambda:pytest.fail("bad capacity reached extension"))
    with pytest.raises(ValueError):
        router_blas.prepare(None,capacity=rows)


def test_current_policy_remains_uncoupled_and_canonical_row_router_unchanged():
    root=Path(__file__).parents[1]/"src/tensorfold"
    policy=(root/"cuda/sm70/execution_policy.py").read_text()
    assert 'routing_native._NATIVE_LOGITS = padded_linears._ENABLED = enabled' in policy
    current=(root/"cuda/moe.py").read_text()
    assert 'out[row:row + 1].copy_(value[row:row + 1] @ weight.T)' in current
    tree=ast.parse((root/"cuda/sm70/router_blas.py").read_text())
    assert not any(isinstance(n,ast.Assign)and any(isinstance(t,ast.Attribute)for t in n.targets)for n in tree.body)
