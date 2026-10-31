"""CPU native assistant metadata, recurrent protocol and exact delta-q controls."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.gemma4.assistant_metadata import AssistantConfig, checked_headers
from tensorfold.families.gemma4.cuda.assistant import Gemma4AssistantHead, _Chains

# Original metadata/header DATA at google/gemma-4-E4B-it-assistant@8d0031ea.
FIXTURES = Path(__file__).parent / 'fixtures'


def metadata():
    return json.loads((FIXTURES / 'gemma4_assistant_config.json').read_text())


def test_checkpoint_geometry_and_accounting(tmp_path, monkeypatch):
    import tensorfold.cuda.capacity as capacity

    (tmp_path / 'config.json').write_text(json.dumps(metadata()))
    monkeypatch.setattr(capacity, 'headers', lambda directory: json.loads(
        (FIXTURES / 'gemma4_assistant_headers.json').read_text()))
    cfg, header = checked_headers(tmp_path)
    assert len(header) == 50
    assert cfg.parameter_bytes == 159132680
    assert cfg.memory_bytes() == (159132680, 318265360)
    assert cfg.state_bytes(131072) == 0
    for value in (0, -1, True, 131073):
        with pytest.raises(ValueError):
            cfg.state_bytes(value)
    with pytest.raises(ValueError):
        cfg.memory_bytes(8)


@pytest.mark.parametrize('field,value', [('hidden_size',512), ('num_kv_shared_layers',3),
    ('max_position_embeddings',32768), ('vocab_size',32000), ('attention_bias',0),
    ('final_logit_softcapping',30.0), ('rms_norm_eps',1e-5), ('use_bidirectional_attention','all')])
def test_text_metadata_refuses_changed_architecture(field, value):
    raw = metadata()
    raw['text_config'][field] = value
    with pytest.raises(ValueError):
        AssistantConfig.from_dict(raw)


@pytest.mark.parametrize('field,value', [('use_ordered_embeddings',False), ('num_centroids',1024),
    ('centroid_intermediate_top_k',16), ('backbone_hidden_size',1536), ('quantization_config',{'bits':8})])
def test_head_metadata_refuses_changed_architecture(field, value):
    raw = metadata()
    raw[field] = value
    with pytest.raises(ValueError):
        AssistantConfig.from_dict(raw)


def tiny():
    torch.manual_seed(1729)
    text = Gemma4TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, global_head_dim=16,
        vocab_size=64, layer_types=['sliding_attention']*3+['full_attention'],
        num_kv_shared_layers=4, sliding_window=4, hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0, use_double_wide_mlp=False,
        rope_parameters={'sliding_attention': {'rope_type':'default','rope_theta':10000.0},
            'full_attention': {'rope_type':'proportional','rope_theta':1000000.0,'partial_rotary_factor':0.5}})
    cfg = Gemma4AssistantConfig(text_config=text.to_dict(), backbone_hidden_size=32,
        num_centroids=8, centroid_intermediate_top_k=2, use_ordered_embeddings=True)
    model = Gemma4AssistantForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    model.masked_embedding.token_ordering.copy_(torch.randperm(64))
    shared = {kind: tuple(torch.randn(1,1,n,d,dtype=torch.bfloat16) for _ in range(2))
              for kind,n,d in [('sliding_attention',4,8),('full_attention',9,16)]}
    held = dict(position=9,last_hidden_state=torch.randn(1,1,32,dtype=torch.bfloat16),
                attention_mask=torch.ones(1,9,dtype=torch.long),shared_kv_states=shared)
    embedding = Gemma4TextScaledWordEmbedding(64,32,0,embed_scale=32**0.5).to(dtype=torch.bfloat16)
    target = SimpleNamespace(embedding_weight=embedding.weight,
                             model=SimpleNamespace(get_input_embeddings=lambda: embedding),
                             assistant_state=lambda state: held)
    head = Gemma4AssistantHead.__new__(Gemma4AssistantHead)
    head.cfg = SimpleNamespace(vocab=64,backbone_hidden=32)
    head.model, head.target = model,target
    head.context_limit,head.block,head.device = 64,8,torch.device('cpu')
    head._bound = None
    head.bind_state(SimpleNamespace(owner=target))
    return head,held


@pytest.mark.parametrize('sampled',[False,True])
@torch.inference_mode()
def test_native_oracle_fixed_anchor_scaled_embedding_frozen_kv_and_actual_delta_q(sampled, monkeypatch):
    import tensorfold.cuda.sampling as sampling_module

    def cpu_argmax(logits, positions, policy):
        # Only the sampler's CUDA transport boundary is substituted; native
        # assistant math/recurrence remains real CPU. GPU sampler is parent gate.
        assert policy is None and len(positions) == 1
        assert bool(torch.isfinite(logits).all())
        return [int(logits.argmax(dim=-1).item())]

    monkeypatch.setattr(sampling_module, 'sample_rows', cpu_argmax)
    head,held = tiny()
    native = deepcopy(head.model)
    hidden = held['last_hidden_state'].clone()
    anchor,expected = 11,[]
    for _ in range(3):
        # Pinned HF SinglePositionMultiTokenCandidateGenerator's public native recurrence.
        embedding = head.target.model.get_input_embeddings()(torch.tensor([[anchor]]))
        result = native(inputs_embeds=torch.cat([embedding,hidden],-1),position_ids=torch.tensor([[9]]),
            attention_mask=held['attention_mask'],shared_kv_states=held['shared_kv_states'],use_cache=False)
        anchor = int(result.logits.argmax().item())
        hidden = result.last_hidden_state
        expected.append(anchor)
    before = {key:tuple(t.clone() for t in value) for key,value in held['shared_kv_states'].items()}
    original_hidden = held['last_hidden_state'].clone()
    state,policy,history = head.make_state(),Sampling(seed=123,temperature=.7),object()
    captured = []
    handle = head.model.pre_projection.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].clone()))
    try:
        chains = head.propose(state,11,3,history,sampled=sampled,sampling=policy)
    finally:
        handle.remove()
    expected_embedding = head.target.model.get_input_embeddings()(torch.tensor([[11]]))
    assert torch.equal(captured[0][...,:32], expected_embedding)
    assert not torch.equal(captured[0][...,:32], head.target.embedding_weight[11].reshape(1,1,-1))
    assert torch.equal(captured[0][...,32:], original_hidden)
    replay = head.propose(state,11,3,history,sampled=sampled,sampling=policy)
    assert chains.tokens == replay.tokens == tuple(expected)
    if sampled:
        packet = chains.sample(0,3,policy,10,history)
        assert packet.tokens == tuple(expected)
        for token,row in zip(expected,packet.q_rows):
            np.testing.assert_array_equal(row.ids,[token])
            np.testing.assert_array_equal(row.probs,[1.0])
            assert not row.ids.flags.writeable and not row.probs.flags.writeable
    else:
        assert chains.finish(0,3,history)[0] == expected
    assert torch.equal(held['last_hidden_state'],original_hidden)
    for key,value in before.items():
        assert all(torch.equal(a,b) for a,b in zip(value,held['shared_kv_states'][key]))
    state.reset()
    state.restore(state.snapshot())
    assert state.nbytes == 0
    with pytest.raises(ValueError):
        state.restore((1,))
    with pytest.raises(RuntimeError):
        head.bind_state(head._bound)


def test_packet_policy_and_proposal_boundaries():
    head,_ = tiny()
    state = head.make_state()
    assert head.propose(state,0,0) is None
    for anchor,count in [(True,1),(-1,1),(64,1),(0,8),(0,-1)]:
        with pytest.raises(ValueError):
            head.propose(state,anchor,count)
    head.append(state,torch.zeros(3,32,dtype=torch.bfloat16),prompt=True)
    with pytest.raises(ValueError):
        head.append(state,torch.zeros(3,32))
    chains = _Chains([7,8],None,True)
    with pytest.raises(ValueError):
        chains.finish(0,2)
    with pytest.raises(ValueError):
        chains.sample(0,2,None,10)


@pytest.mark.parametrize('problem',['permutation','missing','unexpected','mismatched','load-error'])
def test_constructor_rejects_bad_indexing_or_incomplete_native_load(tmp_path, monkeypatch, problem):
    import safetensors
    import transformers
    from tensorfold.families.gemma4.cuda import assistant as implementation

    cfg = AssistantConfig()
    monkeypatch.setattr(implementation,'checked_headers',lambda directory:(cfg,{}))
    monkeypatch.setattr(implementation,'checkpoint_path',lambda *args:tmp_path/'model.safetensors')
    class Header:
        def __enter__(self):
            return self
        def __exit__(self,*args):
            return None
        def get_tensor(self,name):
            ordering = torch.arange(cfg.vocab)
            if problem == 'permutation':
                ordering[1] = 0
            return ordering
    monkeypatch.setattr(safetensors,'safe_open',lambda *args,**kwargs:Header())
    calls = []
    class Model:
        def eval(self):
            return self
        def to(self,device):
            calls.append('upload')
            raise AssertionError('invalid native load must fail before upload')
    class Loader:
        @staticmethod
        def from_pretrained(*args,**kwargs):
            assert kwargs['output_loading_info'] is True and kwargs['trust_remote_code'] is False
            assert kwargs['local_files_only'] is True
            calls.append('load')
            keys = {'missing':'missing_keys','unexpected':'unexpected_keys',
                    'mismatched':'mismatched_keys','load-error':'error_msgs'}
            return Model(),{keys[problem]:['invalid checkpoint']}
    monkeypatch.setattr(transformers,'Gemma4AssistantForCausalLM',Loader)
    target = SimpleNamespace(config=SimpleNamespace(hidden=2560,vocab=cfg.vocab),
                             embedding_weight=torch.empty(1,dtype=torch.bfloat16),device=torch.device('cpu'))
    with pytest.raises(ValueError):
        Gemma4AssistantHead(tmp_path,target,131072)
    assert calls == ([] if problem == 'permutation' else ['load'])


@torch.inference_mode()
def test_deterministic_context_tail_omits_only_unreachable_descendants(monkeypatch):
    import tensorfold.cuda.sampling as sampling_module
    monkeypatch.setattr(sampling_module,'sample_rows',lambda logits,positions,policy:[int(logits.argmax())])
    head,held = tiny()
    state = head.make_state()
    head.context_limit = 10
    assert held['position'] == 9
    assert head.propose(state,11,7) is None
    head.context_limit = 12
    short = head.propose(state,11,7).tokens
    assert len(short) == 2
    head.context_limit = 64
    full = head.propose(state,11,7).tokens
    assert full[:2] == short
