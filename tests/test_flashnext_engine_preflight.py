"""Unsupported Flash Next combinations fail before model-shape or CUDA setup."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip('torch')


@pytest.mark.parametrize('storage,options,message', [
    ('exl3', {'tp': 2, 'rank': 1, 'master': '192.0.2.1', 'vram_experts': 1.}, 'one CUDA GPU'),
    ('modelopt', {'tp': 2, 'rank': 1, 'master': '192.0.2.1', 'vram_experts': 1.}, 'one CUDA GPU'),
    ('mlx', {'vision': True, 'streams': 1, 'vram_experts': 1.}, 'image input'),
    ('exl3', {'ple_on_ssd': True, 'vram_experts': 1.}, '--ple-on-ssd'),
])
def test_unsupported_combinations_refuse_before_rope_ram_cuda_or_weights(monkeypatch, tmp_path, storage, options, message):
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine, weights

    metadata = {'model_type': 'qwen4_exp'}
    if storage == 'exl3':
        metadata['quantization_config'] = {'quant_method': 'exl3', 'version': '1.4.2'}
    elif storage == 'modelopt':
        metadata['quantization_config'] = {'quant_method': 'modelopt', 'quant_algo': 'NVFP4'}
    (tmp_path / 'config.json').write_text(json.dumps(metadata))
    reached = []

    def forbidden(*args, **kwargs):
        reached.append(True)
        raise AssertionError('unsupported mode reached shape validation or accelerator setup')

    monkeypatch.setattr(qwen4_exp, 'rope_parameters', forbidden)
    # Keep the actual pure budget/format preflight: it owns the one-GPU
    # refusal. Only shape, accelerator and payload work are forbidden here.
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    monkeypatch.setattr(weights, 'load', forbidden)
    with pytest.raises(ValueError, match=message):
        engine.FlashNextEngine(tmp_path, **options)
    assert reached == []


def test_invalid_pair_rejects_before_supported_mode_or_shape_setup(monkeypatch, tmp_path):
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine

    def forbidden(*args, **kwargs):
        raise AssertionError('invalid pair reached shape validation or accelerator setup')

    monkeypatch.setattr(qwen4_exp, 'rope_parameters', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    with pytest.raises(ValueError, match='KV|dtype|format'):
        engine.FlashNextEngine(tmp_path, kv_dtype='invalid')


@pytest.mark.parametrize('mode', [None, False, 0, 'cpu', 'RAM'])
def test_invalid_indexer_history_mode_refuses_before_model_or_cuda(monkeypatch, tmp_path, mode):
    from tensorfold.cuda import chat_template
    from tensorfold.families.qwen4_exp.cuda import engine

    def forbidden(*args, **kwargs):
        pytest.fail('invalid indexer history reached model or accelerator setup')

    monkeypatch.setattr(chat_template, 'prepare_chat_template', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    with pytest.raises(ValueError, match='indexer history'):
        engine.FlashNextEngine(tmp_path, indexer_history=mode)


def test_ram_indexer_history_requires_gguf_before_model_or_cuda(monkeypatch, tmp_path):
    from tensorfold.cuda import chat_template
    from tensorfold.families.qwen4_exp import gguf
    from tensorfold.families.qwen4_exp.cuda import engine

    def forbidden(*args, **kwargs):
        pytest.fail('unsupported RAM indexer history reached model or accelerator setup')

    monkeypatch.setattr(gguf, 'resolve_gguf', lambda path: None)
    monkeypatch.setattr(chat_template, 'prepare_chat_template', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    with pytest.raises(ValueError, match='RAM indexer history requires.*GGUF SM70'):
        engine.FlashNextEngine(tmp_path, indexer_history='ram')


@pytest.mark.parametrize('key,value', [('fp16', 'fp16'), ('fp16', 'int8'), ('int8', 'fp16'), ('int8', 'int8')])
def test_sm70_accepts_qualified_cache_sides_before_model_setup(monkeypatch, tmp_path, key, value):
    from tensorfold.cuda import chat_template
    from tensorfold.families.qwen4_exp import gguf
    from tensorfold.families.qwen4_exp.cuda import engine

    class NextStage(Exception):
        pass

    def next_stage(*args, **kwargs):
        raise NextStage

    monkeypatch.setattr(gguf, 'resolve_gguf', lambda path: tmp_path / 'model.gguf')
    monkeypatch.setattr(chat_template, 'prepare_chat_template', next_stage)
    with pytest.raises(NextStage):
        engine.FlashNextEngine(tmp_path, kv_key_dtype=key, kv_value_dtype=value)


@pytest.mark.parametrize('gguf,options', [
    (True, {'kv_dtype': 'bf16'}),
    (True, {'kv_dtype': 'rotorquant8'}),
    (False, {'kv_dtype': 'fp16'}),
    (False, {'kv_key_dtype': 'fp16', 'kv_value_dtype': 'int8'}),
])
def test_execution_provider_refuses_incompatible_cache_payload_before_cuda(monkeypatch, tmp_path, gguf, options):
    from tensorfold.cuda import chat_template
    from tensorfold.families.qwen4_exp import gguf as gguf_module
    from tensorfold.families.qwen4_exp.cuda import engine

    def forbidden(*args, **kwargs):
        raise AssertionError('incompatible cache payload reached model or CUDA setup')

    monkeypatch.setattr(gguf_module, 'resolve_gguf', lambda path: tmp_path / 'model.gguf' if gguf else None)
    monkeypatch.setattr(chat_template, 'prepare_chat_template', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    with pytest.raises(ValueError, match='SM70|provider'):
        engine.FlashNextEngine(tmp_path, **options)


@pytest.mark.parametrize('world,rows', [(1, 17), (4, 256), (8, 2048)])
def test_sm70_prompt_replay_scratch_admission_matches_allocated_tensor_payload(world, rows):
    from tensorfold.families.qwen4_exp.cuda.engine import _sm70_prefill_gdn_bytes
    from tensorfold.families.qwen4_exp.cuda.gdn import GDNScratch

    metadata = {'linear_num_key_heads': 16, 'linear_num_value_heads': 48,
                'linear_key_head_dim': 128, 'linear_value_head_dim': 128}
    scratch = GDNScratch(rows, 'cpu', 16 // world, 48 // world, dtype=torch.float16)
    allocated = sum(tensor.numel() * tensor.element_size() for tensor in vars(scratch).values())
    assert _sm70_prefill_gdn_bytes(metadata, world, rows) == allocated


@pytest.mark.parametrize('options', [
    {'kv_dtype': 'rotorquant-planar4-sm70-v1'},
    {'kv_key_dtype': 'rotorquant-planar4-sm70-v1', 'kv_value_dtype': 'int8'},
    {'kv_key_dtype': 'int8', 'kv_value_dtype': 'rotorquant-planar4-sm70-v1'},
])
def test_compact_codec_rejects_modern_provider_before_target_metadata_or_weights(monkeypatch, tmp_path, options):
    from tensorfold.cuda import chat_template
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp import gguf
    from tensorfold.families.qwen4_exp.cuda import engine, weights

    def forbidden(*args, **kwargs):
        pytest.fail('compact codec reached modern target metadata, CUDA setup, or weights')

    monkeypatch.setattr(gguf, 'resolve_gguf', lambda path: None)
    monkeypatch.setattr(chat_template, 'prepare_chat_template', forbidden)
    monkeypatch.setattr(qwen4_exp, 'rope_parameters', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    monkeypatch.setattr(weights, 'load', forbidden)
    with pytest.raises(ValueError, match='compact Planar4.*GGUF SM70'):
        engine.FlashNextEngine(tmp_path, **options)


def test_compact_gguf_refuses_blackwell_before_provider_setup_or_weight_loading(monkeypatch, tmp_path):
    from gguf_fixture import tiny_model
    from tensorfold.cuda import chat_template, capacity
    from tensorfold.cuda.sm70 import execution_policy
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp import gguf
    from tensorfold.families.qwen4_exp.cuda import engine, weights
    from tensorfold.families.qwen4_exp.rope import RopeParameters
    from tests.test_cuda_capacity import small_config

    text = small_config()
    text['head_dim'] = 256
    text['shared_expert_intermediate_size'] = 1024
    (tmp_path / 'config.json').write_text(json.dumps(text))
    # Layout eligibility now reads authoritative source headers before CUDA;
    # keep this hardware-ordering fixture backed by a real GGUF inventory.
    tiny_model(tmp_path / 'target.gguf')
    monkeypatch.setattr(gguf, 'resolve_gguf', lambda path: tmp_path / 'target.gguf')
    monkeypatch.setattr(chat_template, 'prepare_chat_template', lambda *a, **kw: None)
    monkeypatch.setattr(qwen4_exp, 'rope_parameters', lambda *a: RopeParameters.from_config(text))
    monkeypatch.setattr(capacity, 'config', lambda *a: text)
    monkeypatch.setattr(weights, 'draft_token_ids', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a: (12, 0))
    def forbidden(*args, **kwargs):
        pytest.fail('compact GGUF reached provider setup, admission, or target weights on Blackwell')
    monkeypatch.setattr(execution_policy, 'configure', forbidden)
    monkeypatch.setattr(weights, 'load', forbidden)
    monkeypatch.setattr(capacity, 'admit', forbidden)
    with pytest.raises(ValueError, match='actual SM70'):
        engine.FlashNextEngine(tmp_path, kv_dtype='rotorquant-planar4-sm70-v1', depth=0)
