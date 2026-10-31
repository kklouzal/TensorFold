"""Tiny native HF binding and actual bounded CPU byte-consumption controls."""
import copy
import gc
import weakref
import hashlib
import json
import struct
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from transformers import Gemma4TextConfig  # noqa: E402
from transformers.initialization import no_init_weights  # noqa: E402
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel  # noqa: E402
from tensorfold.cuda.capacity import headers  # noqa: E402
from tensorfold.cuda.nvfp4.format import LinearMetadata  # noqa: E402
from tensorfold.families.gemma4.cuda.dense_loader import _bind_native, _VerifiedTensors, _manifest  # noqa: E402
from tensorfold.families.gemma4.cuda.dense_packed import PackedProjectionSpec  # noqa: E402
from tensorfold.families.gemma4.dense_tensor_contract import text_tensor_contract  # noqa: E402


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = Gemma4TextModel(Gemma4TextConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, global_head_dim=32, num_kv_shared_layers=2,
            sliding_window=8, hidden_size_per_layer_input=8, vocab_size_per_layer_input=128,
            vocab_size=128, max_position_embeddings=128, enable_moe_block=False, use_double_wide_mlp=False,
            attention_k_eq_v=False, layer_types=['sliding_attention', 'full_attention']*2,
            attn_implementation='sdpa'))
        self.lm_head = torch.nn.Linear(64, 128, bias=False)
        self.lm_head.weight = self.get_input_embeddings().weight

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def forward(self, ids):
        return self.lm_head(self.model.language_model(input_ids=ids, use_cache=False).last_hidden_state)


def geometry():
    return SimpleNamespace(hidden=64, vocab=128, layers=4, heads=4, kv_heads=2, shared_layers=2,
                           head_dim=16, global_head_dim=32, intermediate=128, per_layer_input=8,
                           layer_types=['sliding_attention', 'full_attention']*2)


class Source:
    def __init__(self, values):
        self.values, self.consumed = values, {}

    def tensor(self, name, device, *, discard=False):
        self.consumed[name] = 'synthetic-test-only'
        return None if discard else self.values[name].clone().to(device)


@torch.inference_mode()
@pytest.mark.parametrize('retained', [False, True])
def test_native_tiny_model_bind_all_buffers_tied_identity_and_projection_without_mirror(retained):
    torch.manual_seed(171)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        reference = Tiny().eval()
        with no_init_weights(), torch.device('cpu'):
            candidate = Tiny().eval()
    finally:
        torch.set_default_dtype(previous)
    originals = {name: value.clone() for name, value in candidate.named_buffers()}
    assert originals and all(value.device.type == 'cpu' for value in originals.values())
    values = {name: value.clone() for name, value in reference.state_dict().items() if name != 'lm_head.weight'}
    key = 'model.language_model.layers.0.mlp.gate_proj'
    values.pop(key+'.weight')
    values[key+'.weight_packed'] = torch.zeros(128,32,dtype=torch.uint8)
    values[key+'.weight_scale'] = torch.ones(128,4).to(torch.float8_e4m3fn)
    values[key+'.weight_scale_2'] = torch.ones(())
    metadata = LinearMetadata({'quant_method':'modelopt','config_groups':{'g':{
        'targets':[key],'weights':{'num_bits':4,'type':'float','group_size':16},'input_activations':None}}})
    parts = {'weight_packed':('U8',[128,32]),'weight_scale':('F8_E4M3',[128,4]),'weight_scale_2':('F32',[])}
    descriptor = PackedProjectionSpec.from_header(key, parts, metadata, expected_shape=(128,64))
    cfg = geometry()
    contract, _, unused = text_tensor_contract(cfg)
    if retained:
        for name in unused:
            values[name] = torch.zeros(contract[name], dtype=torch.bfloat16)
    inventory = {name:{'dtype':('U8' if value.dtype==torch.uint8 else 'F8_E4M3' if value.dtype==torch.float8_e4m3fn
                              else 'F32' if value.dtype==torch.float32 else 'BF16'), 'shape':list(value.shape)}
                 for name,value in values.items()}
    source = Source(values)
    calls = []

    def fake_factory(spec, payload, policy):
        calls.append(set(payload))
        # This is solely native binding/caller integration. Independent real
        # quantized GPU kernels and their numerics are NOT tested here.
        return copy.deepcopy(reference.model.language_model.layers[0].mlp.gate_proj)

    details = _bind_native(candidate,cfg,inventory,{key:descriptor},source,'cpu',SimpleNamespace(prompt=False),
                           'retained-original-bf16' if retained else 'omitted',packed_factory=fake_factory)
    assert set(source.consumed)==set(inventory) and len(calls)==1
    assert candidate.lm_head.weight is candidate.get_input_embeddings().weight
    assert details['tied_head_identity'] and details['packed_projections']==1
    assert details['expected_unused_tensors']==(6 if retained else 0)
    for name,value in candidate.named_buffers():
        assert torch.equal(value,originals[name])
    ids=torch.tensor([[2,7,11,3,6]])
    assert torch.equal(reference(ids),candidate(ids))


def test_missing_native_payload_refused_before_any_tensor_is_read():
    model=Tiny().to(torch.bfloat16)
    values={name:value for name,value in model.state_dict().items() if name!='lm_head.weight'}
    inventory={name:{'dtype':'BF16','shape':list(value.shape)} for name,value in values.items()}
    del inventory['model.language_model.norm.weight']
    source=Source(values)
    with pytest.raises(ValueError,match='complete native') as caught:
        _bind_native(model,geometry(),inventory,{},source,'cpu',None,'omitted')
    assert not source.consumed
    assert caught.value.__traceback__ is not None
    held = weakref.ref(model)
    model = None
    gc.collect()
    assert held() is None  # retained primary traceback must not own skeleton


def checkpoint(tmp_path):
    value=torch.arange(8,dtype=torch.bfloat16)
    body=value.view(torch.uint8).numpy().tobytes()
    header=json.dumps({'weight':{'dtype':'BF16','shape':[8],'data_offsets':[0,len(body)]}},separators=(',',':')).encode()
    (tmp_path/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+body)
    return value,body,headers(tmp_path)


def test_actual_reader_hashes_exact_consumed_bytes_and_detects_corruption(tmp_path):
    value,body,inventory=checkpoint(tmp_path)
    digest=hashlib.sha256(body).hexdigest()
    source=_VerifiedTensors(tmp_path,inventory,{'weight':digest})
    try:
        assert torch.equal(source.tensor('weight','cpu'),value)
        assert source.consumed=={'weight':digest} and source.device_touched is None
        with pytest.raises(ValueError,match='twice'):
            source.tensor('weight','cpu')
    finally:
        source.close()
    source=_VerifiedTensors(tmp_path,inventory,{'weight':'0'*64})
    try:
        with pytest.raises(ValueError,match='payload hash mismatch'):
            source.tensor('weight','cpu')
        assert not source.consumed
    finally:
        source.close()


def test_manifest_metadata_identity_must_match_before_any_binding(tmp_path):
    _,body,inventory=checkpoint(tmp_path)
    config=b'{"model_type":"synthetic-test"}'
    (tmp_path/'config.json').write_bytes(config)
    manifest={'schema_version':1,'tensor_sha256':{'weight':hashlib.sha256(body).hexdigest()},
              'metadata_sha256':{'config.json':hashlib.sha256(config).hexdigest()},'unused_shared_kv_policy':'omitted'}
    path=tmp_path/'tensorfold_quantization_manifest.json'
    path.write_text(json.dumps(manifest))
    assert _manifest(tmp_path,inventory)==manifest
    (tmp_path/'config.json').write_text('{"model_type":"changed"}')
    with pytest.raises(ValueError,match='metadata hash mismatch'):
        _manifest(tmp_path,inventory)


def test_no_init_context_preserves_real_native_audio_vision_and_embedding_buffers():
    from transformers import Gemma4AudioConfig, Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4AudioRelPositionalEncoding, Gemma4VisionRotaryEmbedding, Gemma4TextScaledWordEmbedding)
    constructors = [lambda: Gemma4AudioRelPositionalEncoding(Gemma4AudioConfig()),
                    lambda: Gemma4VisionRotaryEmbedding(Gemma4VisionConfig()),
                    lambda: Gemma4TextScaledWordEmbedding(8,16,0,embed_scale=4.0)]
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        for create in constructors:
            normal = create()
            with no_init_weights(), torch.device('cpu'):
                empty_parameters = create()
            expected, actual = dict(normal.named_buffers()), dict(empty_parameters.named_buffers())
            assert expected and set(expected) == set(actual)
            for name in expected:
                assert expected[name].dtype == actual[name].dtype
                assert torch.equal(expected[name],actual[name]) and bool(torch.isfinite(actual[name]).all())
    finally:
        torch.set_default_dtype(previous)


def test_metadata_hashing_enforces_existing_owned_size_bound(tmp_path, monkeypatch):
    from tensorfold.families.gemma4.cuda import dense_loader
    path = tmp_path/'metadata'
    path.write_bytes(b'abc')
    monkeypatch.setattr(dense_loader,'MAX_HEADER_BYTES',3)
    assert dense_loader._file_hash(path)==hashlib.sha256(b'abc').hexdigest()
    path.write_bytes(b'abcd')
    with pytest.raises(ValueError,match='size bound'):
        dense_loader._file_hash(path)
    path.write_bytes(b'')
    assert dense_loader._file_hash(path)==hashlib.sha256(b'').hexdigest()
