"""Independent native HF floor/ordering oracle and queued-chain CPU controls."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.models.gemma4_assistant.modeling_gemma4_assistant import Gemma4AssistantMaskedEmbedder

from tensorfold.families.gemma4.cuda.assistant_masked import TensorMaskedEmbedder, DeviceGreedyChain, device_greedy_chain


def native_embedder():
    cfg = SimpleNamespace(num_centroids=8,centroid_intermediate_top_k=2,
                          get_text_config=lambda:SimpleNamespace(hidden_size=16,vocab_size=64))
    native = Gemma4AssistantMaskedEmbedder(cfg).to(dtype=torch.bfloat16)
    native.token_ordering.copy_(torch.arange(64))
    return native


@torch.inference_mode()
@pytest.mark.parametrize('batch,rows',[(1,1),(2,3)])
def test_complete_dense_logits_and_shared_authorities_match_native(batch,rows):
    torch.manual_seed(626)
    native = native_embedder()
    native.token_ordering.copy_(torch.randperm(64))
    candidate = TensorMaskedEmbedder(native)
    hidden = torch.randn(batch,rows,16,dtype=torch.bfloat16)
    head = torch.randn(64,16,dtype=torch.bfloat16)
    actual,expected = candidate(hidden,head),native(hidden,head)
    assert torch.equal(actual,expected)
    assert actual.shape == (batch,rows,64)
    assert candidate.centroids is native.centroids
    assert candidate.token_ordering.data_ptr() == native.token_ordering.data_ptr()
    assert candidate.state_dict().keys() == native.state_dict().keys()
    assert candidate.state_dict()['token_ordering'].data_ptr() == native.token_ordering.data_ptr()
    assert torch.equal(actual.argmax(-1),expected.argmax(-1))


@torch.inference_mode()
@pytest.mark.parametrize('large_negative',[False,True])
def test_full_canonical_id_ties_including_unselected_finite_floor(large_negative):
    native = native_embedder()
    native.centroids.weight.zero_()
    native.centroids.weight[:,0].copy_(torch.arange(8))
    hidden = torch.zeros(1,1,16,dtype=torch.bfloat16)
    hidden[...,0] = 1
    head = torch.zeros(64,16,dtype=torch.bfloat16)
    if large_negative:
        head[:,0] = -3e38
    actual = TensorMaskedEmbedder(native)(hidden,head)
    expected = native(hidden,head)
    assert torch.equal(actual,expected)
    # Selected cluster order is7 then6. Full target-ID tie policy chooses48;
    # at huge negative magnitudes floor rounds back to selected min, so token0
    # OUTSIDE selected centroids is the authoritative full-vocabulary winner.
    assert int(actual.argmax()) == (0 if large_negative else 48)


def test_all_bf16_floor_bits_match_independent_python_scalar_conversion():
    values = torch.arange(65536,dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    expected = torch.tensor([float(value)-1.0 for value in values.to(torch.float64).tolist()],dtype=torch.bfloat16)
    actual = (values.float()-1.0).to(torch.bfloat16)
    finite = torch.isfinite(values)
    assert int(finite.sum()) == 65280
    assert torch.equal(actual.view(torch.uint16)[finite],expected.view(torch.uint16)[finite])


@pytest.mark.parametrize('invalid',[float('nan'),float('inf'),float('-inf')])
@torch.inference_mode()
def test_nonfinite_native_dense_values_remain_invalid_for_packet_guard(invalid):
    native = native_embedder()
    hidden = torch.ones(1,1,16,dtype=torch.bfloat16)
    head = torch.full((64,16),invalid,dtype=torch.bfloat16)
    expected = native(hidden,head)
    actual = TensorMaskedEmbedder(native)(hidden,head)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
    assert not bool(torch.isfinite(actual).all())


@pytest.mark.parametrize('flags',[[3,1,1],[3,0,0]])
def test_device_packet_checks_every_flag_before_publishing_ids(flags):
    packet = DeviceGreedyChain(torch.tensor([[1,0,1],flags],dtype=torch.int64))
    with pytest.raises(ValueError):
        packet.readback()
    assert packet._tokens is None and packet.words is not None


@torch.inference_mode()
def test_device_recurrent_chain_native_math_one_readback_and_frozen_position(monkeypatch):
    import tensorfold.cuda.sampling as sampling
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding
    torch.manual_seed(37)
    text = Gemma4TextConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=4,num_attention_heads=2,
        num_key_value_heads=1,head_dim=8,global_head_dim=16,vocab_size=64,num_kv_shared_layers=4,
        layer_types=['sliding_attention']*3+['full_attention'],sliding_window=4,hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,use_double_wide_mlp=False,
        rope_parameters={'sliding_attention':{'rope_type':'default','rope_theta':10000.0},
            'full_attention':{'rope_type':'proportional','rope_theta':1000000.0,'partial_rotary_factor':.5}})
    cfg = Gemma4AssistantConfig(text_config=text.to_dict(),backbone_hidden_size=32,num_centroids=8,
        centroid_intermediate_top_k=2,use_ordered_embeddings=True)
    native = Gemma4AssistantForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    native.masked_embedding.token_ordering.copy_(torch.randperm(64))
    candidate = deepcopy(native)
    candidate.masked_embedding = TensorMaskedEmbedder(candidate.masked_embedding)
    embedding = Gemma4TextScaledWordEmbedding(64,32,0,32**.5).to(dtype=torch.bfloat16)
    hidden = torch.randn(1,1,32,dtype=torch.bfloat16)
    shared = {kind:tuple(torch.randn(1,1,n,d,dtype=torch.bfloat16) for _ in range(2))
              for kind,n,d in [('sliding_attention',4,8),('full_attention',9,16)]}
    mask = torch.ones(1,9,dtype=torch.long)
    reference,previous,state = [],11,hidden.clone()
    for _ in range(6):
        result = native(inputs_embeds=torch.cat([embedding(torch.tensor([[previous]])),state],-1),
            position_ids=torch.tensor([[9]]),shared_kv_states=shared,attention_mask=mask,use_cache=False)
        previous = int(result.logits.argmax())
        state = result.last_hidden_state
        reference.append(previous)
    positions = []
    def cpu_transport(logits,starts,keyed,samplings):
        # Substitute CUDA validation/transport only. Reuse REAL existing greedy
        # packet math, not a custom selection algorithm, on native CPU logits.
        positions.append(keyed)
        item = sampling._group_packet(logits,starts,0,[0])
        return SimpleNamespace(launched=[item])
    monkeypatch.setattr(sampling,'launch_streams',cpu_transport)
    packet = device_greedy_chain(candidate,embedding,hidden,shared,mask,11,6,9)
    assert packet._tokens is None and packet.words.shape == (6,3)
    assert positions == [[[9]]]*6
    assert packet.readback() == tuple(reference)
    assert packet.words is None
    assert packet.readback() == tuple(reference)
    np.testing.assert_array_equal(reference,packet.readback())


def test_precision_boundary_refuses_unproved_fp32_floor():
    native = native_embedder()
    with pytest.raises(ValueError):
        TensorMaskedEmbedder(native.float())
    native = native_embedder()
    with pytest.raises(ValueError):
        TensorMaskedEmbedder(native)(torch.ones(1,1,16),torch.ones(64,16))
