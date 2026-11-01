"""Actual-q recurrent proposal integration; native HF CPU with CUDA boundary substituted."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.rejection import draw, proposal_row
from tensorfold.families.gemma4.cuda import assistant_sampled as candidate
from tests.test_gemma_assistant import tiny


@pytest.fixture
def cpu_effective_rows(monkeypatch):
    import tensorfold.cuda.sampling as sampling
    def cpu_metadata(logits, positions, policy):
        assert not logits.is_cuda and logits.ndim == 2 and logits.shape[0] == len(positions)
        sampling.validate_policy(positions,policy)
    # Reuse the REAL effective_rows/topk/canonical repair/proposal-row code. Only
    # its CUDA geometry check is substituted for the native CPU integration.
    monkeypatch.setattr(sampling,'validate_rows',cpu_metadata)


@torch.inference_mode()
@pytest.mark.parametrize('temperature',[.7,1.0])
@pytest.mark.parametrize('factor,qk',[(1.0,None),(.5,50),(.75,50),(.75,16)])
def test_native_recurrence_actual_sampled_tokens_laws_positions_and_no_resampling(cpu_effective_rows,monkeypatch,temperature,factor,qk):
    head,held = tiny()
    native = deepcopy(head.model)
    embedding = head.target.model.get_input_embeddings()
    policy = Sampling(seed=62941,temperature=temperature,top_k=50,top_p=.95)
    proposal=replace(policy,temperature=temperature*factor,top_k=policy.top_k if qk is None else qk)
    previous,state,expected,expected_q,argmax = 11,held['last_hidden_state'].clone(),[],[],[]
    for index in range(3):
        result = native(inputs_embeds=torch.cat([embedding(torch.tensor([[previous]])),state],-1),
                        position_ids=torch.tensor([[9]]),shared_kv_states=held['shared_kv_states'],
                        attention_mask=held['attention_mask'],use_cache=False)
        # Independent complete-vocabulary CPU q oracle; producer uses bounded
        # GPU top-k+guard and canonical membership repair through effective_rows.
        values = result.logits.float().numpy().reshape(-1).astype(np.float64)
        row = proposal_row(values,np.arange(64),proposal)
        previous = draw(row,policy.seed,10+index)
        state = result.last_hidden_state
        expected.append(previous)
        expected_q.append(row)
        argmax.append(int(result.logits.argmax()))
    draws = []
    def recorded_draw(row,seed,position):
        draws.append(position)
        return draw(row,seed,position)
    monkeypatch.setattr(candidate,'draw',recorded_draw)
    history = object()
    chain = candidate.sampled_assistant_chain(head.model,embedding,held['last_hidden_state'],
        held['shared_kv_states'],held['attention_mask'],11,3,9,policy,history,enabled=True,
        temperature_factor=factor,proposal_top_k=qk)
    packet = chain.sample(0,3,policy,10,history)
    assert chain.sampling is policy and chain.proposal_sampling == proposal
    assert policy.temperature == temperature and policy.top_k == 50
    if proposal != policy:
        with pytest.raises(ValueError):
            chain.sample(0,3,proposal,10,history)
    assert packet.tokens == tuple(expected)
    assert packet.q_rows == tuple(expected_q)
    assert any(token != best for token,best in zip(expected,argmax))
    assert draws == [10,11,12]
    assert chain.sample(0,3,policy,10,history) == packet
    assert draws == [10,11,12]  # transport must NEVER draw twice
    clipped = chain.sample(0,2,policy,10,history)
    assert clipped.tokens == packet.tokens[:2] and clipped.q_rows == packet.q_rows[:2]
    assert all(not row.ids.flags.writeable and not row.probs.flags.writeable for row in packet.q_rows)
    for args in [(0,3,policy,11,history),(0,3,Sampling(seed=1,temperature=temperature),10,history),
                 (0,3,policy,10,object()),(1,3,policy,10,history),(0,-1,policy,10,history)]:
        with pytest.raises(ValueError):
            chain.sample(*args)
    with pytest.raises(ValueError):
        chain.finish(0,3,history)


@pytest.mark.parametrize('enabled,temperature',[(False,1.0),(0,1.0),(True,0.0),(True,-1.0),(True,float('nan'))])
def test_optin_positive_temperature_boundary_before_model_use(enabled,temperature):
    head,held = tiny()
    class Forbidden:
        config = head.model.config
        def __call__(self,*args,**kwargs):
            raise AssertionError('invalid policy must fail before model work')
    with pytest.raises(ValueError):
        candidate.sampled_assistant_chain(Forbidden(),head.target.model.get_input_embeddings(),
            held['last_hidden_state'],held['shared_kv_states'],held['attention_mask'],11,3,9,
            Sampling(seed=1,temperature=temperature),enabled=enabled)


@pytest.mark.parametrize('factor,k',[(True,None),(0,None),(-1,None),(float('nan'),None),
    (float('inf'),None),(10**1000,None),(1,True),(1,-1),(1,1.5),(1,65)])
def test_private_parameter_metadata_refuses_invalid_values(factor,k):
    with pytest.raises(ValueError):
        candidate.validate_proposal_parameters(factor,k,vocab=64)


@pytest.mark.parametrize('temperature,factor',[(1e308,2.0),(1e-300,1e-300)])
def test_temperature_product_rejects_overflow_and_zero_underflow(temperature,factor):
    target=Sampling(seed=1,temperature=temperature,top_k=50,top_p=1)
    with pytest.raises(ValueError,match='product'):
        candidate.make_proposal_sampling(target,factor,50,vocab=64)
    assert target.temperature==temperature


def test_default_proposal_policy_is_same_target_object_and_topk_zero_valid():
    target=Sampling(seed=11,temperature=1,top_k=50,top_p=1)
    assert candidate.make_proposal_sampling(target,vocab=64) is target
    proposal=candidate.make_proposal_sampling(target,.75,0,vocab=64)
    assert proposal.temperature==.75 and proposal.top_k==0 and proposal.seed==target.seed
    assert target.top_k==50 and target.temperature==1


@torch.inference_mode()
@pytest.mark.parametrize('count', [8, 15])
def test_explicit_extended_recurrence_laws_draws_prefixes_and_legacy_limit(cpu_effective_rows, count):
    head, held = tiny()
    embedding = head.target.model.get_input_embeddings()
    policy = Sampling(seed=62941, temperature=1, top_k=50, top_p=.95)
    hidden, previous, expected, laws = held['last_hidden_state'], 11, [], []
    for index in range(count):
        result = head.model(inputs_embeds=torch.cat((embedding(torch.tensor([[previous]])), hidden), -1),
                            position_ids=torch.tensor([[9]]), shared_kv_states=held['shared_kv_states'],
                            attention_mask=held['attention_mask'], use_cache=False)
        row = proposal_row(result.logits.float().numpy().reshape(-1).astype(np.float64), np.arange(64), policy)
        previous = draw(row, policy.seed, 10 + index)
        expected.append(previous)
        laws.append(row)
        hidden = result.last_hidden_state
    args = (head.model, embedding, held['last_hidden_state'], held['shared_kv_states'],
            held['attention_mask'], 11, count, 9, policy)
    with pytest.raises(ValueError, match='admitted maximum'):
        candidate.sampled_assistant_chain(*args, enabled=True)  # legacy default stays seven
    packet = candidate.sampled_assistant_chain(*args, enabled=True, max_count=15).sample(0, count, policy, 10)
    assert packet.tokens == tuple(expected) and packet.q_rows == tuple(laws)
    replay = candidate.sampled_assistant_chain(*args, enabled=True, max_count=15).sample(0, count, policy, 10)
    assert replay == packet
    assert policy == Sampling(seed=62941, temperature=1, top_k=50, top_p=.95)


@pytest.mark.parametrize('maximum', [0, 16, True, 1.5])
def test_invalid_extended_maximum_fails_before_model(maximum):
    head, held = tiny()
    with pytest.raises(ValueError, match='maximum'):
        candidate.sampled_assistant_chain(head.model, head.target.model.get_input_embeddings(),
            held['last_hidden_state'], held['shared_kv_states'], held['attention_mask'],
            11, 1, 9, Sampling(seed=1), enabled=True, max_count=maximum)
