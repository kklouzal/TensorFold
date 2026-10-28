"""Original exact experts gain only an explicit preadmitted cap72 profile."""
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("torch")
from . import test_sm70_family_trial_policy as family  # noqa: E402
from tensorfold.cuda.sm70 import family_native, gguf_ops, greedy_native  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import prefix_policy  # noqa: E402

policy, source = family.policy, family.source


@pytest.mark.parametrize("failure", [False, True])
def test_expert_code_preparation_precedes_cap_publication_and_keeps_dense_unchanged(source, monkeypatch, failure):
    module=source.module
    monkeypatch.setenv(module.PROFILE_ENV, module.OWNER_EXPERT72_TRIAL)
    bank=source.policy.weights.GGUFExperts
    before=(family.selectors(source.policy), bank.packed_rows, source.policy.weights.GGUFDense.implementation, module._SELECTED)
    monkeypatch.setattr(family_native,"extension",lambda:NS(prepare_value_guard=lambda:None,
        prepare_owner_prelude=lambda:None,prepare_owner_batch=lambda:None))
    calls=[]
    error=RuntimeError("existing expert code preparation refused")
    def prepare():
        assert (family.selectors(source.policy),bank.packed_rows,source.policy.weights.GGUFDense.implementation,module._SELECTED)==before
        calls.append("expert")
        if failure:
            raise error
        return [22,40,0,0,1024,70]
    monkeypatch.setattr(gguf_ops,"extension",lambda:NS(prepare_dense32=lambda:calls.append("dense") or 4,
                                                       prepare_experts_exact72=prepare))
    monkeypatch.setattr(greedy_native,"extension",lambda:None)
    if failure:
        with pytest.raises(RuntimeError) as caught:
            module.configure(source.text,tp=8,torch=family.torch)
        assert caught.value is error and bank.packed_rows==before[1]
    else:
        assert module.configure(source.text,tp=8,torch=family.torch)==module.OWNER_EXPERT72_TRIAL
        assert bank.packed_rows==72 and bank.implementation=="packed_exact"
        assert source.policy.weights.GGUFDense.implementation=="native32"
        assert module._FAMILY_SOFTWARE["expert72_code_summary"][0]==22
        monkeypatch.setenv(module.PROFILE_ENV,"reference")
        module.configure({},tp=8,torch=family.torch)
        assert bank.packed_rows==32 and bank.implementation=="reference"
    assert calls==["dense","expert"]


def test_cap72_epoch_is_separate_from_selected_batch_reference():
    assert prefix_policy.state_epoch(prefix_policy.OWNER_EXPERT72_PROFILE)!=prefix_policy.state_epoch(prefix_policy.OWNER_BATCH_PRELUDE_PROFILE)
    assert 72*11<=65535 and 72*11*80*2 < 32*2**20
