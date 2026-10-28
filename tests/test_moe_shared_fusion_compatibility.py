"""Profile/selector/funding intersections, without another kernel implementation."""
import ast
from pathlib import Path
from types import SimpleNamespace as N

import pytest
import torch

from tensorfold.cuda.sm70.execution_policy import SHARED_FUSION_TRIAL,ROUTER_BLAS_TRIAL
from tensorfold.families.qwen4_exp.cuda.moe_region_graphs import validate_region
from tensorfold.families.qwen4_exp.cuda import moe_region_catalogue as catalogue
from tensorfold.families.qwen4_exp.cuda import prefix_policy


class Tensor:
    shape=(32,2560)
    dtype=torch.float16
    device='cuda:0'
    is_cuda=True
    def is_contiguous(self):
        return True


@pytest.mark.parametrize('profile,implementation',[(ROUTER_BLAS_TRIAL,'packed_exact'),
                                                 (SHARED_FUSION_TRIAL,'packed_blocks2w')])
def test_numeric_profile_and_captured_expert_selector_are_paired(profile,implementation):
    weights=N(cfg=N(experts=512,top_k=10),meta={'sm70_profile':profile})
    bank=N(kernel='gguf',implementation=implementation,packed_rows=32,count=513,dims=2560,width=80)
    layer=N(moe=N(experts=bank))
    buffers=N(prefill=False,mixed=Tensor(),moe=N(logits=Tensor(),pick=Tensor(),wts=Tensor(),y=Tensor()))
    validate_region(weights,layer,buffers,32)
    bank.implementation='packed_exact' if implementation=='packed_blocks2w' else 'packed_blocks2w'
    with pytest.raises(ValueError,match='GGUF bank'):
        validate_region(weights,layer,buffers,32)


def test_fusion_and_router_prefix_state_epochs_remain_distinct(monkeypatch):
    assert prefix_policy.state_epoch(ROUTER_BLAS_TRIAL)!=prefix_policy.state_epoch(SHARED_FUSION_TRIAL)
    monkeypatch.setenv(catalogue.ENV,'1')
    assert catalogue.requested(ROUTER_BLAS_TRIAL) and catalogue.requested(SHARED_FUSION_TRIAL)


def test_fusion_eager_credit_and_capture_host_allowance_are_distinct():
    path=Path(__file__).resolve().parents[1]/'src/tensorfold/families/qwen4_exp/cuda/engine.py'
    tree=ast.parse(path.read_text())
    init=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='FlashNextEngine'
              for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_initialize')
    host=next(n for n in ast.walk(init) if isinstance(n,ast.If) and
              isinstance(n.test,ast.Call) and isinstance(n.test.func,ast.Name) and n.test.func.id=='getattr' and
              len(n.test.args)>1 and isinstance(n.test.args[1],ast.Constant) and n.test.args[1].value=='_moe_region_graphs_requested')
    fusion=next(n for n in ast.walk(init) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and
                any(isinstance(v,ast.Name) and v.id=='SHARED_FUSION_TRIAL' for v in n.test.comparators) and
                any(isinstance(v,ast.ImportFrom) and v.module=='tensorfold.cuda.sm70.gguf_ops' for v in n.body))
    options={'device_workspace_bytes':19,'host_workspace_bytes':137}
    scope={'self':N(_moe_region_graphs_requested=True,sm70_profile=SHARED_FUSION_TRIAL),
           'SHARED_FUSION_TRIAL':SHARED_FUSION_TRIAL,'tp':8,'options':options,
           '__package__':'tensorfold.families.qwen4_exp.cuda'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[host,fusion],type_ignores=[])),'combined-funding','exec'),scope)
    assert options['device_workspace_bytes']==19+(2<<20)  # Existing ordinary eager credit, not capture arena.
    assert options['host_workspace_bytes']==137+8*(1<<30)+8*(64<<10)
