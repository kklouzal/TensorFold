"""Offline plan contracts using public checkpoint header DATA, no weights/GPU."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tensorfold.offline_quantization import NVFP4_RECIPE, plan_nvfp4
from tensorfold.offline_quantization import plan as planning

FIXTURES = Path(__file__).parent/'fixtures'


def data(role):
    prefix = 'gemma4_e4b_quant_plan' if role == 'target' else 'gemma4_assistant'
    return tuple(json.loads((FIXTURES/f'{prefix}_{suffix}.json').read_text()) for suffix in ('config', 'headers'))


@pytest.fixture(params=['target', 'assistant'])
def source(request, tmp_path, monkeypatch):
    import tensorfold.cuda.capacity as capacity
    role = request.param
    config, inventory = data(role)
    text = json.dumps(config, sort_keys=True, indent=2)+'\n'
    (tmp_path/'config.json').write_text(text)
    monkeypatch.setattr(planning, 'headers', lambda _: copy.deepcopy(inventory))
    monkeypatch.setattr(capacity, 'headers', lambda _: copy.deepcopy(inventory))
    return role, tmp_path, config, inventory


def test_plan_is_deterministic_header_only_and_explicitly_not_conversion(source):
    role, path, config, inventory = source
    before = copy.deepcopy((config, inventory))
    files = sorted(path.iterdir())
    plan = plan_nvfp4(path)
    assert plan == plan_nvfp4(path)
    assert (config, inventory) == before
    assert sorted(path.iterdir()) == files
    assert plan['status'] == 'plan-only' and plan['execution_implemented'] is False
    assert plan['source']['weight_payload_verification'] == 'pending'
    assert plan['source']['config_sha256'] == hashlib.sha256((path/'config.json').read_bytes()).hexdigest()
    assert plan['calibration']['status'] == 'required-not-run'
    assert plan['required_gates']
    assert plan['recipe']['producer']['commit'] == 'e352caca3c36236fc15adea2627a9d42788380c4'
    assert plan['recipe']['name'] == NVFP4_RECIPE
    assert plan['recipe']['weight_scale_algorithm'] == 'local_hessian'
    assert plan['recipe']['fp8_scale_sweep'] is True
    assert plan['recipe']['activation_global'] == 'static calibrated per projection'
    assert plan['recipe']['batch_amax_fallback'] is False
    assert plan['coverage']['packed_embedding_and_ordered_head_adapter'].startswith('not implemented')
    assert json.loads(json.dumps(plan)) == plan
    selected = [item for item in plan['tensors'] if item['action'] == 'quantize']
    assert len(selected) == (258 if role == 'target' else 22)
    assert plan['coverage']['stored_core_projections'] == (294 if role == 'target' else 22)
    assert plan['coverage']['source_tensors'] == len(inventory)
    assert sum(item['source_bytes'] for item in plan['tensors']) == (15992314836 if role == 'target' else 159132680)
    assert plan['estimates']['mixed_export_payload_bytes'] == (10400130020 if role == 'target' else 143494328)
    for item in selected:
        n, k = item['shape']
        assert item['estimated_export_bytes'] == n*k//2+n*k//16+8


def test_target_active_vs_unused_and_explicit_bf16_boundaries(source):
    role, path, _, _ = source
    plan = plan_nvfp4(path)
    by_name = {item['name']: item for item in plan['tensors']}
    if role == 'target':
        assert plan['coverage']['expected_unused_tensors'] == 54
        for index in range(42):
            for suffix in ('k_proj', 'v_proj', 'k_norm'):
                name = f'model.language_model.layers.{index}.self_attn.{suffix}.weight'
                assert (by_name[name]['action'] == 'expected-unused') is (index >= 24)
        for name in ('model.language_model.embed_tokens.weight', 'model.language_model.embed_tokens_per_layer.weight',
                     'model.language_model.layers.0.per_layer_input_gate.weight',
                     'model.language_model.per_layer_model_projection.weight'):
            assert by_name[name]['action'] == 'exclude'
        for name, item in by_name.items():
            if name.startswith(('model.audio_tower.', 'model.vision_tower.', 'model.embed_audio.', 'model.embed_vision.')):
                assert item['action'] == 'exclude' and item['estimated_export_bytes'] == item['source_bytes']
    else:
        assert plan['coverage']['expected_unused_tensors'] == 0
        for name in ('model.embed_tokens.weight', 'masked_embedding.centroids.weight', 'masked_embedding.token_ordering'):
            assert by_name[name]['action'] == 'exclude'
        assert by_name['masked_embedding.token_ordering']['source_dtype'] == 'I64'
        assert by_name['pre_projection.weight']['action'] == by_name['post_projection.weight']['action'] == 'quantize'


@pytest.mark.parametrize('recipe', ['nvfp4', 'max', 'local_hessian+gptq', None])
def test_unknown_recipe_rejected_before_any_source_read(tmp_path, recipe):
    with pytest.raises(ValueError, match='recipe'):
        plan_nvfp4(tmp_path, recipe=recipe)


def test_quantized_source_or_external_quantization_sidecar_refused(source):
    _, path, config, _ = source
    (path/'config.json').write_text(json.dumps({**config, 'quantization_config': {'quant_method': 'modelopt', 'quant_algo': 'NVFP4'}}))
    with pytest.raises(ValueError, match='BF16|unquantized'):
        plan_nvfp4(path)


def test_wrong_architecture_rejected_before_headers(tmp_path, monkeypatch):
    (tmp_path/'config.json').write_text(json.dumps({'architectures': ['Other'], 'model_type': 'other'}))
    monkeypatch.setattr(planning, 'headers', lambda _: pytest.fail('wrong architecture read tensor inventory'))
    with pytest.raises(ValueError):
        plan_nvfp4(tmp_path)


@pytest.mark.parametrize('mutation', ['missing-unused', 'extra-text', 'shape', 'quantized-dtype', 'missing-audio'])
def test_source_inventory_cannot_silently_lose_or_reinterpret_tensors(tmp_path, monkeypatch, mutation):
    config, inventory = data('target')
    (tmp_path/'config.json').write_text(json.dumps(config))
    name = 'model.language_model.layers.24.self_attn.k_proj.weight'
    if mutation == 'missing-unused':
        del inventory[name]
    elif mutation == 'extra-text':
        inventory['model.language_model.unrecognized.weight'] = {'dtype': 'BF16', 'shape': [1]}
    elif mutation == 'shape':
        inventory[name]['shape'][0] += 1
    elif mutation == 'quantized-dtype':
        inventory[name]['dtype'] = 'U8'
    else:
        inventory = {k: v for k, v in inventory.items() if not k.startswith('model.audio_tower.')}
    monkeypatch.setattr(planning, 'headers', lambda _: inventory)
    with pytest.raises(ValueError):
        plan_nvfp4(tmp_path)



def test_partial_media_inventory_is_only_a_plan_with_mandatory_validation_gate(tmp_path, monkeypatch):
    config, inventory = data('target')
    (tmp_path/'config.json').write_text(json.dumps(config))
    audio = sorted(name for name in inventory if name.startswith('model.audio_tower.'))
    assert len(audio) > 1
    inventory = {name: info for name, info in inventory.items() if name not in audio[1:]}
    monkeypatch.setattr(planning, 'headers', lambda _: inventory)
    plan = plan_nvfp4(tmp_path)
    assert plan['status'] == 'plan-only' and plan['execution_implemented'] is False
    assert plan['coverage']['media_tensor_validation'] == 'prefix-only; full native-model consumption pending'
    assert 'complete native vision/audio/projector tensor consumption before calibration or export' in plan['required_gates']
    assert plan['calibration']['status'] == 'required-not-run'


def test_config_change_during_inspection_rejected(source, monkeypatch):
    _, path, _, _ = source
    original = planning.read_text
    count = 0

    def changed(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)+(' ' if count > 1 else '')

    monkeypatch.setattr(planning, 'read_text', changed)
    with pytest.raises(ValueError, match='changed'):
        plan_nvfp4(path)


def test_module_import_requires_no_accelerator_producer_or_network():
    script = '''
import sys
class Reject:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers', 'modelopt', 'huggingface_hub'):
            raise RuntimeError('planner imported runtime/provider: '+fullname)
sys.meta_path.insert(0, Reject())
from tensorfold.offline_quantization import plan_nvfp4, NVFP4_RECIPE
assert callable(plan_nvfp4) and NVFP4_RECIPE.endswith('-v1')
'''
    completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
