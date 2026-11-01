"""CPU-only local-mask ownership/dispatch/native-reference contracts."""
from contextlib import contextmanager

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tensorfold.families.gemma4.cuda import assistant_local as candidate
from tensorfold.families.gemma4.cuda.assistant_attention import AssistantAttentionPolicy
from tensorfold.families.gemma4.cuda.e2b_attention import E2BTextAttention
from tests.test_gemma_assistant import tiny


@pytest.mark.parametrize('selection',['native','none-default','none-efficient'])
def test_local_dispatch_frame_padding_global_and_native_reference(selection,monkeypatch):
    head,held=tiny(query_heads=4,kv_heads=2)
    frame=candidate.LocalMaskFrame(selection)
    policy=AssistantAttentionPolicy()
    native=head.model.model.layers[0].self_attn
    module=candidate.AssistantLocalAttention(native,policy,frame)
    global_module=candidate.AssistantLocalAttention(head.model.model.layers[3].self_attn,policy,frame)
    mask=torch.ones(1,1,1,4,dtype=torch.bool)
    hidden=torch.zeros(1,1,16,dtype=torch.bfloat16)
    forced=[]
    active=[]
    @contextmanager
    def backend(backends):
        forced.append(backends)
        active.append(True)
        try:
            yield
        finally:
            active.pop()
    seen=[]
    def base(self,h,pos,attn,kv,**kwargs):
        seen.append((self.layer_type,attn,bool(active)))
        return h,None
    monkeypatch.setattr(candidate,'sdpa_kernel',backend)
    monkeypatch.setattr(E2BTextAttention,'forward',base)
    module(hidden,None,mask,held['shared_kv_states'])
    assert seen[-1][1] is mask and not forced  # no prepared frame, native path
    with frame.bind(None):
        module(hidden,None,mask,held['shared_kv_states'])
        assert seen[-1][1] is (mask if selection=='native' else None)
        assert seen[-1][2] is (selection=='none-efficient')
        with policy.native_reference():
            module(hidden,None,mask,held['shared_kv_states'])
            assert seen[-1][1] is mask and seen[-1][2] is False
        global_module(hidden,None,mask,held['shared_kv_states'])
        assert seen[-1][1] is mask and seen[-1][2] is False
        with frame.bind(torch.ones(1,9)):
            module(hidden,None,mask,held['shared_kv_states'])
            assert seen[-1][1] is mask and seen[-1][2] is False
        module(hidden.expand(1,2,16),None,mask,held['shared_kv_states'])
        assert seen[-1][1] is mask
    assert not frame.active and not frame.unpadded and not active
    assert module.state_dict().keys()==native.state_dict().keys()
    assert module.q_proj.weight.data_ptr()==native.q_proj.weight.data_ptr()


def test_frame_nested_failure_restores_owner_metadata_and_retains_no_tensors():
    frame=candidate.LocalMaskFrame('none-default')
    with frame.bind(None):
        with pytest.raises(RuntimeError):
            with frame.bind(torch.ones(1,3)):
                assert frame.active and not frame.unpadded
                raise RuntimeError('proposal failed')
        assert frame.active and frame.unpadded
    assert not frame.active and not frame.unpadded
    assert not any(isinstance(value,torch.Tensor) for value in vars(frame).values())


@torch.inference_mode()
def test_native_bf16_cpu_math_oracle_and_global_or_padded_unchanged():
    head,held=tiny(query_heads=4,kv_heads=2)
    policy=AssistantAttentionPolicy()
    local=head.model.model.layers[0].self_attn
    frame=candidate.LocalMaskFrame('none-default')
    module=candidate.AssistantLocalAttention(local,policy,frame)
    hidden=torch.randn(1,1,16,dtype=torch.bfloat16)
    positions=head.model.model.rotary_emb(hidden,torch.tensor([[9]]),'sliding_attention')
    mask=torch.ones(1,1,1,4,dtype=torch.bool)
    with sdpa_kernel([SDPBackend.MATH]):
        expected=local(hidden_states=hidden,position_embeddings=positions,attention_mask=mask,
                       shared_kv_states=held['shared_kv_states'])[0]
        with frame.bind(None):
            actual=module(hidden,positions,mask,held['shared_kv_states'])[0]
        assert torch.equal(actual,expected)
        mask[..., -1]=False
        expected=local(hidden_states=hidden,position_embeddings=positions,attention_mask=mask,
                       shared_kv_states=held['shared_kv_states'])[0]
        with frame.bind(torch.ones(1,9)):
            actual=module(hidden,positions,mask,held['shared_kv_states'])[0]
        assert torch.equal(actual,expected)


@pytest.mark.parametrize('selection',[None,False,'flash','none-cudnn',1])
def test_only_finite_private_policies_admitted(selection):
    with pytest.raises(ValueError):
        candidate.LocalMaskFrame(selection)
