"""Full-count metadata is checked separately from subset numerical evidence."""
from types import SimpleNamespace as NS
import pytest

from tools import v100_backend_probe as probe
GGUFExperts = pytest.importorskip("tensorfold.families.qwen4_exp.cuda.gguf_weights").GGUFExperts


def model():
    bank=object.__new__(GGUFExperts)
    bank.count,bank.width,bank.dims,bank.packed_rows,bank.implementation=513,80,2560,72,"packed_exact"
    def packed(rows,columns,kind):
        return NS(n=rows,k=columns,ggml_type=kind,data=NS(shape=(rows,128),is_cuda=True))
    bank.gate,bank.up,bank.down=packed(40960,2560,18),packed(40960,2560,18),packed(1310720,80,20)
    return NS(sm70_profile="source-dense-owner-expert72-trial",execution_rows=64,
        capacity_plan={"execution_envelope":{"target_rows":64}},
        w=NS(layers=[NS(moe=NS(experts=bank)) for _ in range(48)],
             meta={"sm70_family_trial":{"expert_code":[22,64,0,0,1024,70]}}))


def test_full_model_metadata_receipt_preserves_subset_numerical_scope():
    receipt=probe.large_row_expert_model_receipt(model())
    assert receipt["total_including_shared"]==513 and receipt["target_banks"]==48
    assert receipt["execution_rows"]==64 and receipt["expert_native_cap"]==72
    assert "subset" in receipt["scope"]


@pytest.mark.parametrize("bad",["count","cap","rows","extent","residency","code"])
def test_wrong_real_model_geometry_never_becomes_a_full_count_receipt(bad):
    engine=model()
    bank=engine.w.layers[0].moe.experts
    if bad=="count":
        bank.count=11
    elif bad=="cap":
        bank.packed_rows=32
    elif bad=="rows":
        engine.execution_rows=32
    elif bad=="extent":
        bank.down.data.shape=(25600,128)
    elif bad=="residency":
        bank.gate.data.is_cuda=False
    else:
        engine.w.meta["sm70_family_trial"]["expert_code"]=None
    with pytest.raises(ValueError):
        probe.large_row_expert_model_receipt(engine)
