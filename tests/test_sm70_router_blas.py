"""Exact row-canonical logits and routing; unsupported arithmetic is not waived."""
import pytest
import torch
import os
import json
from pathlib import Path
from tensorfold.cuda.sm70 import router_blas, routing_native
from tensorfold.cuda.sm70.execution_policy import _family_runtime

pytestmark=pytest.mark.skipif(not torch.cuda.is_available()or torch.cuda.get_device_capability()!=(7,0),
                             reason="actual pinned SM70 router API oracle")


def canonical(x,weight):
    value=x.float()
    out=torch.empty((x.shape[0],513),dtype=torch.float32,device=x.device)
    for row in range(x.shape[0]):
        out[row:row+1].copy_(value[row:row+1]@weight.T)
    return out


@pytest.mark.parametrize("rows",[1,2,3,8,16,24,32])
@pytest.mark.parametrize("case",["random","cancellation","near-tie","signed-zero"])
def test_pointer_batched_api_matches_row_canonical_bits_and_stable_route(rows,case):
    _family_runtime(torch)  # Assert targetTrue/accumFalse; never change math REGION.
    torch.manual_seed(9800+rows)
    x=(torch.randn((rows,2560),device="cuda")*.25).half()
    weight=torch.randn((513,2560),device="cuda",dtype=torch.float32)*.01
    if case=="cancellation":
        weight[:,::4]=1
        weight[:,1::4]=-1
        weight[:,2::4]=2**-20
        weight[:,3::4]=-2**-20
    elif case=="near-tie":
        weight[1:11].copy_(weight[0:1].expand(10,-1))
        weight[1,0]=torch.nextafter(weight[0,0],torch.full_like(weight[0,0],float("inf")))
    elif case=="signed-zero":
        x.zero_()
        x[:,::2].neg_()
        weight.zero_()
        weight[::2].neg_()
    reference=canonical(x,weight)
    plan=router_blas.prepare(weight,capacity=32)
    assert plan.gpu_bytes()==router_blas.workspace_bytes(32)
    output=torch.full_like(reference,float("nan"))
    assert plan.run(x,output).data_ptr()==output.data_ptr()
    torch.cuda.synchronize()
    picks0=torch.empty((rows,11),dtype=torch.int32,device="cuda")
    picks1=torch.empty_like(picks0)
    w0=torch.empty((rows,11),dtype=torch.float32,device="cuda")
    w1=torch.empty_like(w0)
    routing_native.route(reference,picks0,w0)
    routing_native.route(output,picks1,w1)
    torch.cuda.synchronize()
    equal=torch.equal(output.view(torch.int32),reference.view(torch.int32))
    same_picks=torch.equal(picks0,picks1)
    same_weights=torch.equal(w0.view(torch.int32),w1.view(torch.int32))
    directory=Path(os.environ.get("TENSORFOLD_ROUTER_ORACLE_EVIDENCE","/tmp/router-oracle"))/f"R{rows}-{case}"
    directory.mkdir(parents=True,exist_ok=False)
    facts={"rows":rows,"case":case,"logit_bits_exact":equal,"picks_exact":same_picks,
           "weight_bits_exact":same_weights,"target_reduced_fp16":torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
           "target_fp16_accum":torch.backends.cuda.matmul.allow_fp16_accumulation,"torch_git":torch.version.git_version,
           "scope":"operator numerical candidate only; mismatches remain failure, no approximate waiver"}
    if not equal or not same_picks or not same_weights:
        import numpy as np
        for name,value in (("input",x),("weight",weight),("reference",reference),("actual",output),
                           ("picks_reference",picks0),("picks_actual",picks1),("weights_reference",w0),("weights_actual",w1)):
            np.save(directory/(name+".npy"),value.cpu().numpy(),allow_pickle=False)
        changed=(output.view(torch.int32)!=reference.view(torch.int32)).nonzero().cpu().tolist()
        facts["changed_logit_count"]=len(changed)
        facts["first_changed_coordinates"]=changed[:16]
    (directory/"result.json").write_text(json.dumps(facts,indent=2)+"\n")
    # All three strict guards remain; diagnostic output never repairs a pass.
    assert equal and same_picks and same_weights


@pytest.mark.parametrize("rows", [1, 3, 32])
def test_model_shared_role_preserves_independent_plan_results_and_caller_output(rows):
    """Only sharing/ownership changes; the historical row-logit oracle stays strict."""
    _family_runtime(torch)
    torch.manual_seed(12000 + rows)
    weights = [torch.randn((513, 2560), device="cuda", dtype=torch.float32) * .01 for _ in range(2)]
    x = torch.randn((rows, 2560), device="cuda").half()
    role = router_blas.RolePlans(weights, 32)
    outputs = [torch.empty((rows, 513), device="cuda", dtype=torch.float32) for _ in weights]
    references = [torch.empty_like(value) for value in outputs]
    for weight, output in zip(weights, references):
        router_blas.prepare(weight, capacity=32).run(x, output)
    for index in (0, 1, 0):
        role.run(weights[index], x, outputs[index])
        torch.cuda.synchronize()
        assert torch.equal(outputs[index].view(torch.int32), references[index].view(torch.int32))
    # A subsequent other-weight call must not overwrite an earlier caller's out.
    assert torch.equal(outputs[1].view(torch.int32), references[1].view(torch.int32))
    assert role.calls == 3 and not role.overflowed
