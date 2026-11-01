"""Private candidate routing/ownership/native-reference integration, CPU only."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.gemma4.cuda.assistant import Gemma4AssistantHead
from tensorfold.families.gemma4.cuda.assistant_masked import TensorMaskedEmbedder
from tests.test_gemma_assistant import tiny


@pytest.fixture
def construct(monkeypatch):
    import safetensors
    import transformers
    from tensorfold.families.gemma4.cuda import assistant as implementation
    original,held = tiny(query_heads=4,kv_heads=2)
    target = original.target
    target.config = SimpleNamespace(hidden=32,vocab=64)
    target.device = torch.device('cpu')
    cfg = SimpleNamespace(backbone_hidden=32,vocab=64,block=8,parameter_bytes=64,
                          state_bytes=lambda context:0,memory_bytes=lambda:(64,128))
    monkeypatch.setattr(implementation,'checked_headers',lambda path:(cfg,{}))
    monkeypatch.setattr(implementation,'checkpoint_path',lambda *args:'fixture.safetensors')
    class Source:
        def __enter__(self):
            return self
        def __exit__(self,*args):
            pass
        def get_tensor(self,name):
            return torch.arange(64)
    monkeypatch.setattr(safetensors,'safe_open',lambda *args,**kwargs:Source())
    class Loader:
        @staticmethod
        def from_pretrained(*args,**kwargs):
            return original.model,{}
    monkeypatch.setattr(transformers,'Gemma4AssistantForCausalLM',Loader)
    def create(**flags):
        head = Gemma4AssistantHead('fixture',target,64,**flags)
        head.bind_state(SimpleNamespace(owner=target))
        return head,held,original.model
    return create


@pytest.mark.parametrize('flag',['assistant_attention_gqa','assistant_device_chain','assistant_sampled_q'])
@pytest.mark.parametrize('value',[0,1,None,'true'])
def test_flags_fail_before_any_metadata_or_allocation(flag,value,monkeypatch):
    from tensorfold.families.gemma4.cuda import assistant as implementation
    def forbidden(*args):
        raise AssertionError('invalid flag reached metadata/allocation')
    monkeypatch.setattr(implementation,'checked_headers',forbidden)
    with pytest.raises(ValueError,match='must be boolean'):
        Gemma4AssistantHead('fixture',object(),64,**{flag:value})


def test_disabled_defaults_leave_native_modules_and_exports_unchanged(construct):
    head,_,native = construct()
    assert head._attention_policy is None
    assert head.model is native and head.model.masked_embedding is head._native_masked_embedding
    assert not head._device_chain and not head._sampled_q and head._reference_depth == 0
    assert all(type(layer.self_attn).__name__ == 'Gemma4TextAttention' for layer in native.model.layers)


@torch.inference_mode()
def test_all_flags_native_reference_nested_failure_preserves_aliases_and_fusion(construct):
    head,held,_ = construct(assistant_attention_gqa=True,assistant_device_chain=True,assistant_sampled_q=True)
    candidate = head.model.masked_embedding
    assert isinstance(candidate,TensorMaskedEmbedder)
    assert candidate.centroids is head._native_masked_embedding.centroids
    assert candidate.token_ordering.data_ptr() == head._native_masked_embedding.token_ordering.data_ptr()
    assert candidate.state_dict().keys() == head._native_masked_embedding.state_dict().keys()
    assert head._attention_policy.reference_depth == 0
    weights = {name:param.data_ptr() for name,param in head.model.named_parameters()}
    with head.native_reference():
        assert head.model.masked_embedding is head._native_masked_embedding
        assert head._reference_depth == head._attention_policy.reference_depth == 1
        with pytest.raises(RuntimeError,match='oracle failed'):
            with head.native_reference():
                assert head._reference_depth == head._attention_policy.reference_depth == 2
                raise RuntimeError('oracle failed')
        assert head._reference_depth == head._attention_policy.reference_depth == 1
    assert head.model.masked_embedding is candidate
    assert head._reference_depth == head._attention_policy.reference_depth == 0
    assert {name:param.data_ptr() for name,param in head.model.named_parameters()} == weights
    assert head._attention_policy.prompt is True
    assert held['position'] == 9


@pytest.mark.parametrize('sampled,opt_q,opt_device,expected',[
    (True,True,True,'q'),(False,True,True,'device'),(True,False,True,'device'),
    (False,True,False,'native'),(True,False,False,'native')])
def test_exact_routing_and_oracle_forces_native(construct,monkeypatch,sampled,opt_q,opt_device,expected):
    from tensorfold.families.gemma4.cuda import assistant_masked,assistant_sampled
    import tensorfold.cuda.sampling as sampling
    head,_,_ = construct(assistant_device_chain=opt_device,assistant_sampled_q=opt_q)
    seen = []
    sentinel = object()
    def q(*args,**kwargs):
        seen.append('q')
        assert kwargs['enabled'] is True
        return sentinel
    def device(*args,**kwargs):
        seen.append('device')
        return SimpleNamespace(readback=lambda:(7,8))
    def native(logits,positions,policy):
        seen.append('native')
        return [int(logits.argmax())]
    monkeypatch.setattr(assistant_sampled,'sampled_assistant_chain',q)
    monkeypatch.setattr(assistant_masked,'device_greedy_chain',device)
    monkeypatch.setattr(sampling,'sample_rows',native)
    policy = Sampling(seed=1,temperature=.7)
    result = head.propose(head.make_state(),11,2,sampled=sampled,sampling=policy)
    assert seen == ([expected] if expected != 'native' else ['native']*2)
    assert (result is sentinel) is (expected == 'q')
    seen.clear()
    with head.native_reference():
        result = head.propose(head.make_state(),11,2,sampled=sampled,sampling=policy)
    assert seen == ['native']*2
    if sampled:
        packet = result.sample(0,2,policy,10)
        assert all(row.probability(token) == 1.0 for token,row in zip(packet.tokens,packet.q_rows))


def test_late_candidate_installation_failure_does_not_publish_model_alias(construct,monkeypatch):
    from tensorfold.families.gemma4.cuda import assistant_attention
    def fail(*args,**kwargs):
        raise RuntimeError('candidate installer failed')
    monkeypatch.setattr(assistant_attention,'install_assistant_attention',fail)
    with pytest.raises(RuntimeError,match='installer failed') as caught:
        construct(assistant_attention_gqa=True)
    trace = caught.value.__traceback__
    constructor = None
    while trace is not None:
        if trace.tb_frame.f_code is Gemma4AssistantHead.__init__.__code__:
            constructor = trace.tb_frame
            break
        trace = trace.tb_next
    assert constructor is not None
    assert not hasattr(constructor.f_locals['self'], 'model')
    assert all(name not in constructor.f_locals for name in ('model','loaded','native_masked','attention_policy'))


@pytest.mark.parametrize('values',[{'assistant_q_temperature_factor':.5},
    {'assistant_q_top_k':16},{'assistant_q_temperature_factor':True},
    {'assistant_q_top_k':True}])
def test_nondefault_or_malformed_q_parameters_fail_before_metadata(values,monkeypatch):
    from tensorfold.families.gemma4.cuda import assistant as implementation
    def forbidden(*args):
        raise AssertionError('q parameter boundary reached metadata/allocation')
    monkeypatch.setattr(implementation,'checked_headers',forbidden)
    with pytest.raises(ValueError):
        Gemma4AssistantHead('fixture',object(),64,**values)


def test_q_parameter_routing_keeps_native_greedy_and_native_reference(construct,monkeypatch):
    from tensorfold.families.gemma4.cuda import assistant_sampled
    import tensorfold.cuda.sampling as sampling
    head,_,_=construct(assistant_sampled_q=True,assistant_q_temperature_factor=.75,assistant_q_top_k=16)
    calls=[]
    def q(*args,**kwargs):
        calls.append(kwargs)
        return object()
    monkeypatch.setattr(assistant_sampled,'sampled_assistant_chain',q)
    monkeypatch.setattr(sampling,'sample_rows',lambda logits,*args:[int(logits.argmax())])
    policy=Sampling(seed=1,temperature=1,top_k=50,top_p=1)
    head.propose(head.make_state(),11,2,sampled=True,sampling=policy)
    assert calls==[{'enabled':True,'temperature_factor':.75,'proposal_top_k':16}]
    with torch.inference_mode():
        head.propose(head.make_state(),11,2,sampled=False)
        with head.native_reference():
            head.propose(head.make_state(),11,2,sampled=True,sampling=policy)
    assert len(calls)==1
