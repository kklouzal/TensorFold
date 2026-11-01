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
@pytest.mark.parametrize('weight_name', ['weight_packed', 'weight'])
def test_native_tiny_model_bind_all_buffers_tied_identity_and_projection_without_mirror(retained, weight_name):
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
    values[key+'.'+weight_name] = torch.zeros(128,32,dtype=torch.uint8)
    values[key+'.weight_scale'] = torch.ones(128,4).to(torch.float8_e4m3fn)
    values[key+'.weight_scale_2'] = torch.ones(())
    metadata = LinearMetadata({'quant_method':'modelopt','config_groups':{'g':{
        'targets':[key],'weights':{'num_bits':4,'type':'float','group_size':16},'input_activations':None}}})
    parts = {weight_name:('U8',[128,32]),'weight_scale':('F8_E4M3',[128,4]),'weight_scale_2':('F32',[])}
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


@pytest.mark.parametrize('filename',['quantization_config.json','quant_config.json'])
def test_manual_manifest_must_bind_each_present_supported_quantization_sidecar(tmp_path,filename):
    _,body,inventory=checkpoint(tmp_path)
    config=b'{"model_type":"synthetic-test"}'
    (tmp_path/'config.json').write_bytes(config)
    sidecar=b'{"quant_method":"modelopt","quant_algo":"NVFP4"}'
    (tmp_path/filename).write_bytes(sidecar)
    manifest={'schema_version':1,'tensor_sha256':{'weight':hashlib.sha256(body).hexdigest()},
              'metadata_sha256':{'config.json':hashlib.sha256(config).hexdigest()},'unused_shared_kv_policy':'omitted'}
    path=tmp_path/'tensorfold_quantization_manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match=filename):
        _manifest(tmp_path,inventory)
    manifest['metadata_sha256'][filename]=hashlib.sha256(sidecar).hexdigest()
    path.write_text(json.dumps(manifest))
    assert _manifest(tmp_path,inventory)==manifest
    (tmp_path/filename).write_bytes(b'{"quant_method":"modelopt","quant_algo":"FP8"}')
    with pytest.raises(ValueError,match='metadata hash mismatch'):
        _manifest(tmp_path,inventory)


def test_sidecar_binding_uses_same_finite_inventory_before_topology_and_at_manifest_check():
    import ast
    from tensorfold.families.gemma4.cuda import dense_loader
    assert dense_loader.BINDING_METADATA == ('config.json','hf_quant_config.json','quantization_config.json',
                                            'quant_config.json','model.safetensors.index.json')
    # Execute the actual pre-topology comprehension without importing/allocating
    # a native E4B model; both paths must reference this authoritative inventory.
    import inspect
    tree=ast.parse(inspect.getsource(dense_loader.load_dense_quantized))
    assignment=next(node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                    and any(isinstance(target,ast.Name) and target.id=='initial_metadata' for target in node.targets))
    names=[]
    namespace={'BINDING_METADATA':dense_loader.BINDING_METADATA,'root':object(),
               '_file_hash':lambda path: names.append(path) or 'hash',
               'checkpoint_path':lambda root,name:type('Path',(),{'exists':lambda self:True,'name':name})()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'<actual-loader-initial-metadata>','exec'),namespace)
    assert len(namespace['initial_metadata'])==5 and len(names)==5


def test_publisher_file_hash_binds_derived_tensor_hash_and_actual_consumed_bytes(tmp_path):
    value, _, inventory = checkpoint(tmp_path)
    path = tmp_path/'model.safetensors'
    source = _VerifiedTensors(tmp_path, inventory, None,
                             file_hashes={path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
                             file_sizes={path.name: path.stat().st_size})
    try:
        assert source.verified_files == {}  # payload work deferred until native topology validation
        assert torch.equal(source.tensor('weight', 'cpu'), value)
        assert source.verified_files[path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert source.consumed['weight'] == hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    finally:
        source.close()


def test_publisher_wrong_full_file_identity_fails_before_any_consumption(tmp_path):
    _, _, inventory = checkpoint(tmp_path)
    path = tmp_path/'model.safetensors'
    source = _VerifiedTensors(tmp_path, inventory, None,
                             file_hashes={path.name: '0'*64}, file_sizes={path.name: path.stat().st_size})
    try:
        with pytest.raises(ValueError, match='whole-file SHA256'):
            source.tensor('weight', 'cpu')
        assert not source.consumed and source.device_touched is None
    finally:
        source.close()


def test_publisher_verified_digest_still_refuses_changed_tensor_upload(tmp_path):
    _, _, inventory = checkpoint(tmp_path)
    path = tmp_path/'model.safetensors'
    source = _VerifiedTensors(tmp_path, inventory, None,
                             file_hashes={path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
                             file_sizes={path.name: path.stat().st_size})
    try:
        source.hashes = source._publisher_hashes(source.file_hashes, source.file_sizes)
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with pytest.raises(ValueError, match='payload hash mismatch'):
            source.tensor('weight', 'cpu')
        assert not source.consumed
    finally:
        source.close()


def test_manifest_absence_does_not_allow_arbitrary_publisher_config(tmp_path):
    checkpoint(tmp_path)
    (tmp_path/'config.json').write_text('{"model_type":"gemma4"}')
    (tmp_path/'hf_quant_config.json').write_text('{"producer":{"name":"modelopt"}}')
    with pytest.raises(ValueError, match='authorized publisher metadata pin'):
        _manifest(tmp_path, headers(tmp_path))


def test_real_native_meta_skeleton_initializes_only_small_buffer_values():
    from transformers import Gemma4Config
    from tensorfold.families.gemma4.cuda.dense_loader import _native_skeleton
    text = Gemma4TextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                           global_head_dim=32, num_kv_shared_layers=2, sliding_window=8,
                           hidden_size_per_layer_input=8, vocab_size_per_layer_input=128,
                           vocab_size=128, max_position_embeddings=128, enable_moe_block=False,
                           use_double_wide_mlp=False, attention_k_eq_v=False,
                           layer_types=['sliding_attention','full_attention']*2)
    raw = Gemma4Config(text_config=text.to_dict(), vision_config=None, audio_config=None).to_dict()
    candidate = _native_skeleton(raw)
    assert all(p.device.type == 'meta' for p in candidate.parameters())
    values = dict(candidate.named_buffers())
    nonpersistent = {name: value for name, value in values.items() if name not in candidate.state_dict()}
    assert nonpersistent and all(v.device.type == 'cpu' for v in nonpersistent.values())
    assert all(torch.isfinite(v).all() for v in nonpersistent.values())
    assert candidate.model.language_model.embed_tokens.embed_scale.item() == 8.0
    assert candidate.model.language_model.embed_tokens_per_layer.embed_scale.item() == torch.tensor(8**.5, dtype=torch.bfloat16).item()


def test_publisher_hashed_header_cannot_relabel_stale_same_shape_tensor_offsets(tmp_path):
    path = tmp_path/'model.safetensors'
    def encode(swapped):
        entries = {'a': {'dtype':'BF16','shape':[1],'data_offsets':[2,4] if swapped else [0,2]},
                   'b': {'dtype':'BF16','shape':[1],'data_offsets':[0,2] if swapped else [2,4]}}
        raw = json.dumps(entries,separators=(',',':')).encode()
        return struct.pack('<Q',len(raw))+raw+torch.tensor([1.,2.],dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()
    trusted = encode(False)
    stale = encode(True)
    assert len(trusted) == len(stale)
    path.write_bytes(stale)
    source = _VerifiedTensors(tmp_path,headers(tmp_path),None,
                             file_hashes={path.name:hashlib.sha256(trusted).hexdigest()},
                             file_sizes={path.name:len(trusted)})
    try:
        path.write_bytes(trusted)
        with pytest.raises(ValueError,match='hashed header differs'):
            source.tensor('a','cpu')
        assert not source.consumed and source.device_touched is None
    finally:
        source.close()


def test_meta_skeleton_media_buffer_branches_match_public_native_cpu_components():
    from transformers import Gemma4Config, Gemma4AudioConfig, Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4AudioRelPositionalEncoding, Gemma4AudioAttention,
        Gemma4TextRotaryEmbedding, Gemma4VisionRotaryEmbedding,
    )
    from tensorfold.families.gemma4.cuda.dense_loader import _native_skeleton
    text = Gemma4TextConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=2,
                           num_attention_heads=4,num_key_value_heads=2,head_dim=16,global_head_dim=32,
                           num_kv_shared_layers=0,sliding_window=8,hidden_size_per_layer_input=0,
                           vocab_size_per_layer_input=128,vocab_size=128,max_position_embeddings=128,
                           enable_moe_block=False,use_double_wide_mlp=False,attention_k_eq_v=False,
                           layer_types=['sliding_attention','full_attention'])
    vision = Gemma4VisionConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=1,
                               num_attention_heads=4,num_key_value_heads=4,head_dim=16,global_head_dim=16,
                               position_embedding_size=64)
    audio = Gemma4AudioConfig(hidden_size=64,num_hidden_layers=1,num_attention_heads=4,
                             subsampling_conv_channels=[4,4],output_proj_dims=64)
    raw = Gemma4Config(text_config=text.to_dict(),vision_config=vision.to_dict(),audio_config=audio.to_dict()).to_dict()
    previous = torch.get_default_dtype()
    candidate = _native_skeleton(raw)
    assert torch.get_default_dtype() == previous
    assert all(p.device.type == 'meta' for p in candidate.parameters())
    seen = set()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device('cpu'), no_init_weights():
            for module in candidate.modules():
                if isinstance(module, Gemma4VisionRotaryEmbedding):
                    reference = Gemma4VisionRotaryEmbedding(module.config,device='cpu')
                    seen.add('vision')
                elif isinstance(module, Gemma4TextRotaryEmbedding):
                    reference = Gemma4TextRotaryEmbedding(module.config,device='cpu')
                    seen.add('text')
                elif isinstance(module, Gemma4AudioRelPositionalEncoding):
                    reference = Gemma4AudioRelPositionalEncoding(candidate.config.audio_config)
                    seen.add('audio-relative')
                elif isinstance(module, Gemma4AudioAttention):
                    # Same tiny public component constructor, not a whole CPU
                    # model or a copied sine/softcap/reference algorithm.
                    reference = Gemma4AudioAttention(candidate.config.audio_config,layer_idx=0)
                    seen.add('audio-softcap')
                else:
                    continue
                for name in module._non_persistent_buffers_set:
                    actual, expected = module._buffers[name], reference._buffers[name]
                    assert actual.device.type == 'cpu' and actual.dtype == expected.dtype
                    assert torch.equal(actual,expected), (type(module).__name__,name)
    finally:
        torch.set_default_dtype(previous)
    assert seen == {'text','vision','audio-relative','audio-softcap'}


def test_owned_manifest_extra_publisher_fields_cannot_override_tensor_sha(tmp_path):
    from tensorfold.families.gemma4.cuda.dense_loader import _verified_source
    _, _, inventory = checkpoint(tmp_path)
    path = tmp_path/'model.safetensors'
    manifest = {'source_kind':'publisher-pinned','tensor_sha256':{'weight':'0'*64},
                'file_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest()},
                'file_sizes':{path.name:path.stat().st_size}}
    source = _verified_source(tmp_path,inventory,manifest,'actual-owned-manifest-sha')
    try:
        with pytest.raises(ValueError,match='payload hash mismatch'):
            source.tensor('weight','cpu')
        assert not source.consumed and source.verified_files == {}
    finally:
        source.close()


def test_owned_manifest_appearing_after_initial_absence_cannot_bypass_publisher_pins(tmp_path,monkeypatch):
    from pathlib import Path
    from tensorfold.families.gemma4.cuda import dense_loader as loader
    raw = json.loads((Path(__file__).parent/'fixtures/gemma4_e4b_quant_plan_config.json').read_text())
    raw['quantization_config'] = {'quant_method':'modelopt','quant_algo':'NVFP4','group_size':16}
    _, _, inventory = checkpoint(tmp_path)
    (tmp_path/'config.json').write_text(json.dumps(raw))
    (tmp_path/'hf_quant_config.json').write_text('{}')
    model_file = tmp_path/'model.safetensors'
    owned = {'schema_version':1,'tensor_sha256':{'weight':'0'*64},
             'metadata_sha256':{name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
                                for name in ('config.json','hf_quant_config.json')},
             'unused_shared_kv_policy':'omitted','source_kind':'publisher-pinned',
             'file_sha256':{model_file.name:hashlib.sha256(model_file.read_bytes()).hexdigest()},
             'file_sizes':{model_file.name:model_file.stat().st_size}}
    def observe_headers(_):
        # This occurs after the loader snapshots absent MANIFEST authority.
        (tmp_path/loader.MANIFEST).write_text(json.dumps(owned))
        return inventory
    monkeypatch.setattr(loader,'read_config',lambda _:raw)
    monkeypatch.setattr(loader,'headers',observe_headers)
    status = loader.DenseLoadStatus()
    assert not (tmp_path/loader.MANIFEST).exists()
    with pytest.raises(ValueError,match='authorized publisher metadata pin'):
        loader.load_dense_quantized(tmp_path,device='cuda:0',prompt_policy=None,status=status)
    assert (tmp_path/loader.MANIFEST).exists()
    assert status.safe_to_release and status.model is status.source is None
