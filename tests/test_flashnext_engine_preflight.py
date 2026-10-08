"""Unsupported Flash Next combinations fail before model-shape or CUDA setup."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip('torch')


@pytest.mark.parametrize('storage,options,message', [
    ('exl3', {'tp': 2, 'rank': 1, 'master': '192.0.2.1', 'vram_experts': 1.}, 'one GPU'),
    ('modelopt', {'tp': 2, 'rank': 1, 'master': '192.0.2.1', 'vram_experts': 1.}, 'one GPU'),
    ('mlx', {'vision': True, 'streams': 1, 'vram_experts': 1.}, 'image input'),
    ('exl3', {'ple_on_ssd': True, 'vram_experts': 1.}, '--ple-on-ssd'),
])
def test_unsupported_combinations_refuse_before_rope_ram_cuda_or_weights(monkeypatch, tmp_path, storage, options, message):
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp import ram_experts
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
    monkeypatch.setattr(ram_experts, 'check', forbidden)
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
