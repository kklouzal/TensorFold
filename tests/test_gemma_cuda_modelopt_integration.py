"""Standalone mixed ModelOpt policy/admission boundaries; no model or CUDA load."""

import sys
from types import SimpleNamespace

import pytest

from gemma_cuda_target_metadata_test import tiny_config
from tensorfold.cuda import capacity as common
from tensorfold.families import gemma4
from tensorfold.families.gemma4.cuda import capacity
from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
from tensorfold.families.gemma4.cuda.weights import Config


def raw_modelopt():
    raw = tiny_config()
    del raw['quantization']
    raw['quantization_config'] = {'quant_method': 'modelopt', 'quant_algo': 'NVFP4', 'kv_cache_scheme': None}
    return raw


def test_family_native_modelopt_gate_and_no_automatic_drafter(monkeypatch):
    import tensorfold.families as families
    raw = raw_modelopt()
    gemma4.check_metadata(raw, 'cuda')
    monkeypatch.setattr(families, 'read_config', lambda _: raw)
    assert gemma4.cuda_drafter('.') == ''


@pytest.mark.parametrize('field,value', [('quant_algo', 'NVFP4_AWQ'), ('kv_cache_scheme', {'bits': 8}),
    ('pre_quant_scale', True), ('transform_config', {'x': 1}), ('sparsity_config', {'x': 1}),
    ('has_zero_point', True), ('has_zero_point', 0), ('has_zero_point', None)])
def test_family_rejects_different_math(field, value):
    raw = raw_modelopt()
    raw['quantization_config'][field] = value
    with pytest.raises(ValueError, match='ModelOpt'):
        gemma4.check_metadata(raw, 'cuda')


def _capacity_boundary(monkeypatch, *, uma):
    import tensorfold.checkpoint_metadata as metadata
    raw = raw_modelopt()
    cfg = Config.from_dict(raw)
    from tensorfold.families.gemma4.cuda import modelopt_weights
    bounds = {'resident': 1000, 'device_staging': 200, 'host_staging': 300, 'mapped': 0}
    monkeypatch.setattr(metadata, 'read_config', lambda _: raw)
    pytest.importorskip('torch')
    monkeypatch.setattr(common, 'headers', lambda _: {})
    monkeypatch.setattr(modelopt_weights, 'validate_headers', lambda *a, **k: cfg)
    monkeypatch.setattr(modelopt_weights, 'storage_bounds', lambda *a, **k: bounds)
    monkeypatch.setattr(common, 'unified', lambda _: uma)
    observed = []
    monkeypatch.setattr(common, 'admit_host_commitment', lambda amount: observed.append(amount))
    monkeypatch.setattr(capacity, 'expert_workspace', lambda *a: 12345)
    def admitted(directory, context, explicit, runtime, geometry, rule, **kwargs):
        weight, host = kwargs['weight_estimator'](directory)
        return dict(weight=weight, host=host, geometry=geometry,
                    sampling=(kwargs['device_workspace_bytes'], kwargs['host_workspace_bytes']))
    monkeypatch.setattr(common, 'admit', admitted)
    return observed


@pytest.mark.parametrize('uma,staging', [(True, 500), (False, 200)])
def test_mixed_storage_bounds_and_no_double_uma_host_charge(monkeypatch, tmp_path, uma, staging):
    observed = _capacity_boundary(monkeypatch, uma=uma)
    result = capacity.admit(tmp_path, 128, True, max_rows=1)
    assert (result['weight'].resident, result['weight'].staging, result['host']) == (1000, staging, 300)
    assert observed == [300]
    vocab = Config.from_dict(raw_modelopt()).vocab
    assert result['sampling'] == (result['logit_bias_workspace']['device_bytes'],
                                  result['logit_bias_workspace']['host_bytes'])
    assert result['sampling'][0] == 16 * vocab  # R1 bias copies/gather; no rejection q/p owners
    assert result['sampling'][1] >= 12 * vocab
    assert result['geometry'].needed(128) > 12345


def test_host_cgroup_refusal_precedes_device_admission(monkeypatch, tmp_path):
    _capacity_boundary(monkeypatch, uma=True)
    def refuse(_):
        raise ValueError('host cgroup exhausted')
    monkeypatch.setattr(common, 'admit_host_commitment', refuse)
    monkeypatch.setattr(common, 'admit', lambda *a, **k: pytest.fail('device admission after host refusal'))
    with pytest.raises(ValueError, match='host cgroup'):
        capacity.admit(tmp_path, 128, True, max_rows=1)


def test_modelopt_noncheckpoint_math_refused_before_cuda(monkeypatch, tmp_path):
    import torch
    from tensorfold.checkpoint_metadata import read_config
    from tensorfold.cuda import precision
    monkeypatch.setattr(sys.modules[read_config.__module__], 'read_config', lambda _: raw_modelopt())
    monkeypatch.setattr(precision, 'mode', lambda: precision.FULL)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a: pytest.fail('CUDA touched'))
    with pytest.raises(ValueError, match='checkpoint activation'):
        Gemma4Engine(tmp_path, drafter='incompatible', drafter_bits=8)


def test_modelopt_plain_prompt_and_decode_do_not_quantize_activation_groups(monkeypatch):
    from tensorfold.families.gemma4.cuda.forward import Gemma4Target, checkpoint
    target = object.__new__(Gemma4Target)
    target._native, target._plain_prompt = False, True
    monkeypatch.setattr(checkpoint, 'matmul_group', lambda *a, **k: pytest.fail('BF16 shared projections grouped as FP4'))
    calls = []
    class Linear:
        def __call__(self, x):
            calls.append(('decode', x))
            return x
        def prefill(self, x):
            calls.append(('prompt', x))
            return x
    lins = [Linear(), Linear()]
    assert target._project('x', lins, True) == ['x', 'x']
    assert target._project('x', lins, False) == ['x', 'x']
    assert calls == [('prompt', 'x')] * 2 + [('decode', 'x')] * 2


def test_unknown_modelopt_constructor_retirement_retains_primary_and_engine():
    target = object.__new__(Gemma4Engine)
    target._modelopt_load_status = SimpleNamespace(safe_to_release=False, owners={'weight': object()})
    primary = ValueError('original loader failure')
    with pytest.raises(ValueError) as caught:
        target._retain_modelopt_failure(primary)
    assert caught.value is primary
    assert any(owner is target for owner in primary._tensorfold_retained_owners)
