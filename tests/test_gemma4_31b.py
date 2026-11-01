"""Pinned31B geometry and owned native-HF glue; no full weights or GPU claim."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.families.gemma4.cuda.e2b_metadata import E2BConfig, is_dense_gemma, is_31b
from tensorfold.families.gemma4.dense_tensor_contract import text_tensor_contract

FIXTURE = Path(__file__).parent/'fixtures/gemma4_31b_aeon_config.json'


def config31():
    return json.loads(FIXTURE.read_text())


def test_pinned_native31b_geometry_and_context_are_not_e4b_or_draft_minimum():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == 'f4cf6df224883b55932c921091b032cdd10af944b111417fcd50f3cee77b1ce6'
    raw = config31()
    original = copy.deepcopy(raw)
    cfg = E2BConfig.from_architecture(raw)
    assert raw == original and is_dense_gemma(raw) and is_31b(raw)
    assert (cfg.profile,cfg.hidden,cfg.layers,cfg.heads,cfg.kv_heads,cfg.global_kv_heads) == ('31B',5376,60,32,16,4)
    assert (cfg.per_layer_input,cfg.shared_layers,cfg.window,cfg.max_position) == (0,0,1024,262144)
    assert cfg.attention_k_eq_v and cfg.bidirectional_vision and not cfg.has_audio
    assert cfg.tap_layers == ()
    assert cfg.kv_heads_for('full_attention') == 4 and cfg.kv_heads_for('sliding_attention') == 16
    assert cfg.cache_bytes(131072,8) == 10*131072*4*512*4+50*(1023+8)*16*256*4
    assert cfg.cache_bytes(262144,8) > cfg.cache_bytes(131072,8)
    assert cfg.prompt_kv_repeat_bytes(131072,512) == 8*1024**3
    assert cfg.prompt_kv_repeat_bytes(262144,512) == 16*1024**3
    # Q_acc/scaledQ + expandedFP32K/V + original groupedFP32K/V,
    # causal/additive masks/row predicate; no reuse credit across owners.
    expected = 2*512*32*512*4+2*8192*32*512*4+2*8192*4*512*4+5*512*8192+32*512+4
    assert cfg.prompt_math_accumulator_bytes(8192,512) == expected
    assert cfg.prompt_math_accumulator_bytes(131072,512) > 16*1024**3
    assert cfg.prompt_math_accumulator_bytes(262144,512) > 32*1024**3


def test_storage_boundary_keeps_awq_declaration_intact():
    raw = config31()
    before = copy.deepcopy(raw)
    assert E2BConfig.from_checkpoint(raw).profile == '31B'
    assert raw == before and raw['quantization_config']['quant_algo'] == 'NVFP4_AWQ'
    with pytest.raises(ValueError, match='unquantized'):
        E2BConfig.from_dict(raw)


@pytest.mark.parametrize('field,value', [('num_key_value_heads',4),('num_global_key_value_heads',16),
                                        ('num_kv_shared_layers',1),('hidden_size_per_layer_input',1),
                                        ('sliding_window',512),('max_position_embeddings',131072),
                                        ('num_hidden_layers',True),('attention_k_eq_v',False),
                                        ('use_bidirectional_attention',None)])
def test_exact31b_rejects_wrong_text_contract(field,value):
    raw = config31()
    raw['text_config'][field] = value
    with pytest.raises(ValueError):
        E2BConfig.from_architecture(raw)


def test_31b_has_no_audio_or_fabricated_ple_inventory():
    raw = config31()
    raw['audio_config'] = {}
    with pytest.raises(ValueError, match='vision-only'):
        E2BConfig.from_architecture(raw)
    shapes, active, unused = text_tensor_contract(E2BConfig.from_architecture(config31()))
    assert len(active) == 410 and not unused
    assert all('per_layer' not in name for name in shapes)
    assert 'model.language_model.layers.5.self_attn.v_proj.weight' not in shapes
    assert shapes['model.language_model.layers.5.self_attn.k_proj.weight'] == (2048,5376)
    assert shapes['model.language_model.layers.0.self_attn.k_proj.weight'] == (4096,5376)
    assert shapes['model.language_model.layers.0.self_attn.v_proj.weight'] == (4096,5376)


def test_visual_partition_never_splits_native_bidir_blocks():
    pytest.importorskip('torch')
    from tensorfold.families.gemma4.cuda.e2b_engine import prefill_boundaries
    ids = [2]*1400
    assert prefill_boundaries(ids,None) == (512,1024,1400)
    assert prefill_boundaries(ids,None,blocks=((400,680),)) == (400,912,1400)
    # A legitimate adjacent-image native group may exceed ordinary text rows.
    assert prefill_boundaries(ids,None,blocks=((0,1128),)) == (1128,1400)
    for end in prefill_boundaries(ids,None,blocks=((400,1128),)):
        assert not 400 < end < 1128
    with pytest.raises(ValueError, match='ordered'):
        prefill_boundaries(ids,None,blocks=((100,200),(150,300)))


@pytest.mark.parametrize('layer_index',[0,5])
def test_owned_attention_matches_native_raw_k_equals_v_glue(monkeypatch,layer_index):
    torch = pytest.importorskip('torch')
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4 import modeling_gemma4 as hf
    from tensorfold.families.gemma4.cuda import e2b_attention as owned
    from tensorfold.families.gemma4.cuda.e2b_rows import E2BRowPolicy

    cfg = Gemma4TextConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=6,
                          num_attention_heads=32,num_key_value_heads=16,num_global_key_value_heads=4,
                          head_dim=16,global_head_dim=32,num_kv_shared_layers=0,hidden_size_per_layer_input=0,
                          sliding_window=1024,vocab_size=128,attention_k_eq_v=True,
                          layer_types=['sliding_attention']*5+['full_attention'],attn_implementation='sdpa')
    torch.manual_seed(318)
    native = hf.Gemma4TextAttention(cfg,layer_index).to(dtype=torch.bfloat16).eval()
    with torch.no_grad():
        native.k_norm.weight.copy_(torch.linspace(0.2,1.7,native.head_dim))
    policy = E2BRowPolicy()
    policy.prompt = False
    adapter = owned.E2BTextAttention(native,policy)
    interface = hf.ALL_ATTENTION_FUNCTIONS.get_interface('sdpa', hf.eager_attention_forward)

    def oracle(q,k,v,mask,**_):
        # Native attention provider is the oracle here; this control isolates
        # project-owned projection/norm/RoPE/K=V glue, not the GPU kernel.
        return interface(native,q,k,v,mask,dropout=0.0,scaling=1.0,is_causal=False)[0]

    monkeypatch.setattr(owned,'chronological_attention',oracle)
    x = torch.randn(1,3,64,dtype=torch.bfloat16)
    positions = torch.arange(3).unsqueeze(0)
    rotary = hf.Gemma4TextRotaryEmbedding(cfg)
    embeddings = rotary(x,positions,native.layer_type)
    mask = torch.ones(1,1,3,3,dtype=torch.bool).tril()
    with torch.inference_mode():
        expected,_ = native(x,embeddings,mask,{})
        actual,_ = adapter(x,embeddings,mask,{})
    assert torch.equal(actual,expected)
    assert (adapter.v_proj is None) == (layer_index == 5)


def test_large_visual_chunk_admission_checks_actual_remaining_quota(monkeypatch):
    torch = pytest.importorskip('torch')
    from tensorfold.cuda import capacity
    from tensorfold.families.gemma4.cuda.e2b_target import E2BTarget
    target = object.__new__(E2BTarget)
    target.config = E2BConfig.from_architecture(config31())
    target._packed_storage = SimpleNamespace(workspace=lambda _: 1)
    monkeypatch.setattr(capacity,'available_bytes',lambda _:10**12)
    monkeypatch.setattr(torch.cuda.memory,'get_per_process_memory_fraction',lambda _:0.5)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(total_memory=1024**3))
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda _:512*1024**2)
    with pytest.raises(ValueError,match='remaining decoder workspace'):
        target._require_large_chunk_workspace(SimpleNamespace(position=0),1128)


def test_no_ple_encoded_forward_uses_native_vision_mask_builder():
    torch = pytest.importorskip('torch')
    from transformers import Gemma4Config,Gemma4ForConditionalGeneration,Gemma4TextConfig
    from tensorfold.families.gemma4.cuda.e2b_target import E2BTarget,E2BState,E2BEncodedMedia
    from tensorfold.families.gemma4.cuda.e2b_rows import E2BRowPolicy

    text = Gemma4TextConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=2,
                           num_attention_heads=4,num_key_value_heads=2,num_global_key_value_heads=1,
                           head_dim=16,global_head_dim=32,num_kv_shared_layers=0,hidden_size_per_layer_input=0,
                           layer_types=['sliding_attention','full_attention'],sliding_window=8,
                           vocab_size=128,vocab_size_per_layer_input=128,attention_k_eq_v=True,
                           use_bidirectional_attention='vision',attn_implementation='sdpa')
    vision = dict(hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,
                  num_key_value_heads=4,head_dim=8,global_head_dim=8,patch_size=2,pooling_kernel_size=1,
                  position_embedding_size=8,default_output_length=4,use_clipped_linears=False,
                  standardize=False,rope_parameters={'rope_type':'axial','rope_theta':100.0})
    config = Gemma4Config(text_config=text.to_dict(),vision_config=vision,audio_config=None,
                         image_token_id=100,video_token_id=101,audio_token_id=102)
    torch.manual_seed(319)
    model = Gemma4ForConditionalGeneration(config).eval()
    target = object.__new__(E2BTarget)
    target.model,target.device = model,torch.device('cpu')
    target.config = SimpleNamespace(profile='31B',vocab=128,per_layer_input=0,bidirectional_vision=True,softcap=30)
    target._closed,target._collect_taps,target._taps = False,False,{}
    target.tap_layers = ()
    target._row_policy = E2BRowPolicy()
    ids = (3,4,5,6)
    embedding = model.get_input_embeddings()(torch.tensor([ids]))
    types = torch.tensor([[0,1,1,0]])
    encoded = E2BEncodedMedia(ids,None,None,embedding,None,types)
    state = E2BState(config,32,owner=target)
    with torch.inference_mode():
        expected = model.model(inputs_embeds=embedding,mm_token_type_ids=types,
                               position_ids=torch.arange(4).unsqueeze(0),use_cache=False,return_dict=True)
        result = target.forward(list(ids),state,prompt=True,encoded=encoded,output_logits=False)
    assert torch.equal(result.hidden,expected.last_hidden_state.squeeze(0))
    assert state.position == 4
