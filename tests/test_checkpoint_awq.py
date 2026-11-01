"""Actual standard AEON AWQ declarations and explicit input-transform admission."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tensorfold.checkpoint_quantization import effective_config, LinearQuantization
from tensorfold.cuda.nvfp4.format import LinearMetadata

FIXTURE = json.loads((Path(__file__).parent/'fixtures/gemma4_awq_header_cases.json').read_text())


def metadata(case):
    sidecar = case['sidecar']
    # Actual config.json has the algorithm/producer but omits sidecar's detailed
    # pre_quant_scale/group-size fields; reconciliation must preserve both.
    raw = {'quantization_config': {'quant_method': 'modelopt', 'quant_algo': 'NVFP4_AWQ',
                                  'producer': sidecar['producer'],
                                  'ignore': sidecar['quantization']['exclude_modules']}}
    block = effective_config(raw, sidecar, where='pinned-AEON')['quantization_config']
    return block, LinearMetadata(block)


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda case: case['role']+':'+case['module'])
def test_actual_pinned_awq_headers_keep_transform_and_require_capable_consumer(case):
    block, parsed = metadata(case)
    assert block['quant_algo'] == 'NVFP4_AWQ' and block['pre_quant_scale'] is True
    scheme = LinearQuantization(block).resolve(case['module'])
    assert scheme['weights']['num_bits'] == 4 and scheme['input_activations']['num_bits'] == 4
    assert scheme['pre_quant_scale'] is True
    tensors = {key: (value[0], value[1]) for key, value in case['parts'].items()}
    assert parsed.validate(case['module'], tensors, checkpoint=True, allow_pre_quant_scale=True) == ('nvfp4', 4)
    with pytest.raises(ValueError, match='explicit input-transform consumer'):
        parsed.validate(case['module'], tensors, checkpoint=True)


@pytest.mark.parametrize('header', [('F32', [21504]), ('BF16', [1]), ('BF16', []),
                                  ('BF16', [1, 21504]), ('BF16', [21503])])
def test_pre_quant_scale_is_original_per_input_bf16_vector_not_scalar_or_output_scale(header):
    case = FIXTURE['cases'][0]
    _, parsed = metadata(case)
    tensors = {key: (value[0], value[1]) for key, value in case['parts'].items()}
    tensors['pre_quant_scale'] = header
    with pytest.raises(ValueError, match=r'BF16\[K\]'):
        parsed.validate(case['module'], tensors, checkpoint=True, allow_pre_quant_scale=True)


@pytest.mark.parametrize('value', [False, None, 1, 'true'])
def test_awq_cannot_be_declared_without_its_transform_semantics(value):
    raw = {'quantization_config': {'quant_method': 'modelopt', 'quant_algo': 'NVFP4_AWQ',
                                  'pre_quant_scale': value}}
    with pytest.raises(ValueError, match='pre_quant_scale'):
        effective_config(raw, None, where='conflicting-AWQ')


def test_awq_explicit_group_cannot_disable_transform():
    group = {'targets': ['Linear'], 'pre_quant_scale': False}
    raw = {'quantization_config': {'quant_method': 'modelopt', 'quant_algo': 'NVFP4_AWQ',
                                  'config_groups': {'g': group}}}
    with pytest.raises(ValueError, match='pre_quant_scale'):
        effective_config(raw, None, where='conflicting-group')


def test_asymmetric_and_svd_variants_are_not_implicitly_admitted():
    block, _ = metadata(FIXTURE['cases'][0])
    changed = deepcopy(block)
    changed['has_zero_point'] = True
    with pytest.raises(ValueError, match='symmetric'):
        LinearMetadata(changed)
    with pytest.raises(ValueError, match='unsupported ModelOpt'):
        effective_config({'quantization_config': {'quant_method': 'modelopt', 'quant_algo': 'NVFP4_SVDQUANT'}},
                         None, where='unsupported-low-rank')


def test_awq_excluded_bf16_tower_is_unchanged_and_never_scales_plain_weights():
    _, parsed = metadata(FIXTURE['cases'][0])
    assert parsed.validate('model.vision_tower.patch_embedding', {'weight': ('BF16', [768, 768])},
                           checkpoint=True) == ('bf16', None)
    with pytest.raises(ValueError, match='input-transform consumer'):
        parsed.validate('model.vision_tower.patch_embedding',
                        {'weight': ('BF16', [768, 768]), 'pre_quant_scale': ('BF16', [768])},
                        checkpoint=True, allow_pre_quant_scale=True)


def test_plain_nvfp4_cannot_hide_an_awq_buffer_and_calibrations_stay_scalar():
    case = FIXTURE['cases'][0]
    block, _ = metadata(case)
    block['quant_algo'] = 'NVFP4'
    block['config_groups']['group_0'].pop('pre_quant_scale')
    tensors = {key: (value[0], value[1]) for key, value in case['parts'].items()}
    with pytest.raises(ValueError, match='input-transform consumer'):
        LinearMetadata(block).validate(case['module'], tensors, checkpoint=True, allow_pre_quant_scale=True)
    _, awq = metadata(case)
    tensors['input_scale'] = 'F32', [2]
    with pytest.raises(ValueError, match='one floating-point calibration'):
        awq.validate(case['module'], tensors, checkpoint=True, allow_pre_quant_scale=True)
