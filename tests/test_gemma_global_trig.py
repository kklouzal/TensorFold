"""Default-off trig-table admission and constructor contracts, without CUDA execution."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tensorfold.families.gemma4.cuda import capacity
from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
from test_gemma_cuda_modelopt_integration import _capacity_boundary, raw_modelopt


@pytest.mark.parametrize('value', [1, None, 'true'])
@pytest.mark.parametrize('flag', ['global_trig', 'global_trig_prompt'])
def test_constructor_flags_are_strict(value, flag, tmp_path):
    from tensorfold.families.gemma4.cuda.forward import Gemma4Target
    with pytest.raises(ValueError, match='bool'):
        Gemma4Engine(tmp_path, **{flag: value})
    with pytest.raises(ValueError, match='bool'):
        Gemma4Target(None, **{flag: value})
    if flag == 'global_trig':
        with pytest.raises(ValueError, match='bool'):
            capacity.admit(tmp_path, 128, True, global_trig=value)


def test_target_requires_ct_and_an_explicit_valid_context():
    from tensorfold.families.gemma4.cuda.forward import Gemma4Target
    weights = SimpleNamespace(numerics='ct_nvfp4', config=SimpleNamespace(max_position=131072))
    for context in (None, 0, 131073, True):
        with pytest.raises(ValueError, match='CT arithmetic.*admitted context'):
            Gemma4Target(weights, global_trig=True, prompt_context=context)
    weights.numerics = 'modelopt_nvfp4'
    with pytest.raises(ValueError, match='CT arithmetic'):
        Gemma4Target(weights, global_trig=True, prompt_context=131072)


def test_capacity_charges_table_once_not_per_layer_or_snapshot(monkeypatch, tmp_path):
    from tensorfold.families.gemma4.cuda import nvfp4_weights
    from tensorfold.families.gemma4.cuda.weights import Config
    import tensorfold.families as families
    _capacity_boundary(monkeypatch, uma=True)
    cfg = replace(Config.from_dict(raw_modelopt()), global_head_dim=512, global_kv_heads=2,
                  max_position=131072)
    monkeypatch.setattr(families, 'quant_method', lambda _: 'compressed-tensors')
    monkeypatch.setattr(nvfp4_weights, 'validate_headers', lambda *a, **kw: cfg)
    monkeypatch.setattr(nvfp4_weights, 'ct_tensor_contract', lambda _: ({}, {}))
    for keep in (0, 3):
        baseline = capacity.admit(tmp_path, 131072, True, max_rows=1, keep=keep)
        for prepared, trig, expected in ((False, True, 67108864),
                                          (True, False, 536870912),
                                          (True, True, 603979776)):
            candidate = capacity.admit(tmp_path, 131072, True, max_rows=1, keep=keep,
                                       global_trig=trig, prepare_global_prompt=prepared)
            assert candidate['geometry'].needed(131072) - baseline['geometry'].needed(131072) == expected
            assert candidate['weight'] == baseline['weight'] and candidate['host'] == baseline['host']
    monkeypatch.setattr(families, 'quant_method', lambda _: 'modelopt')
    with pytest.raises(ValueError, match='CT arithmetic'):
        capacity.admit(tmp_path, 131072, True, global_trig=True)


def test_table_refuses_invalid_bounds_and_cpu_frequency_before_allocation():
    import torch
    from tensorfold.families.gemma4.cuda.attention import GlobalTrig
    inverse = torch.zeros(256, dtype=torch.float32)
    for context in (True, 0, -1, 2**31, 128):
        with pytest.raises(ValueError, match='CUDA FP32 frequencies.*valid context'):
            GlobalTrig(inverse, context)


def test_target_refuses_state_context_different_from_table():
    from tensorfold.families.gemma4.cuda.forward import Gemma4Target
    target = object.__new__(Gemma4Target)
    target._global_trig = SimpleNamespace(context=128)
    with pytest.raises(ValueError, match='admitted table context'):
        target.make_state(127)


@pytest.mark.parametrize('prepared,trig', [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize('prompt_trig', [False, True])
def test_ct_target_constructs_shared_prompt_and_trig_owners(monkeypatch, prepared, trig, prompt_trig):
    import torch
    from tensorfold.families.gemma4.cuda import attention
    from tensorfold.families.gemma4.cuda.forward import Gemma4Target
    from tensorfold.families.gemma4.cuda.weights import Config
    cfg = replace(Config.from_dict(raw_modelopt()), global_head_dim=512, global_kv_heads=2)
    weights = SimpleNamespace(numerics='ct_nvfp4', config=cfg, norm=torch.ones(cfg.hidden),
                              embed=SimpleNamespace(weight=torch.empty((1, cfg.hidden))))
    owners = []

    def table(inverse, context):
        owner = SimpleNamespace(inverse=inverse, context=context)
        owners.append(owner)
        return owner

    monkeypatch.setattr(attention, 'GlobalTrig', table)
    target = Gemma4Target(weights, prepare_global_prompt=prepared, global_trig=trig,
                          global_trig_prompt=prompt_trig, prompt_context=128)
    assert target._global_trig_prompt is prompt_trig
    assert len(owners) == int(trig)
    assert (target._global_prompt_scratch is not None) is prepared
    if prepared:
        assert sum(t.numel() * t.element_size() for t in target._global_prompt_scratch) == 128 * 2 * 512 * 4
    if trig:
        assert target._global_trig is owners[0]
        assert target._global_trig.context == 128


@pytest.mark.parametrize('prompt,prompt_trig,prepared,trig,expected', [
    (True, True, False, False, 'raw'),
    (True, True, False, True, 'trig'),
    (True, False, False, True, 'raw'),
    (False, False, False, True, 'trig'),
    (False, True, False, True, 'trig'),
    (True, True, True, False, 'prepared'),
    (True, False, True, True, 'prepared'),
    (False, False, True, True, 'trig'),
    (False, True, True, False, 'raw'),
])
def test_ct_full_attention_dispatch_preserves_prompt_decode_matrix(
        monkeypatch, prompt, prompt_trig, prepared, trig, expected):
    import torch
    from tensorfold.families.gemma4.cuda import attention
    from tensorfold.families.gemma4.cuda.forward import Gemma4Target
    from tensorfold.families.gemma4.cuda.weights import Config
    cfg = replace(Config.from_dict(raw_modelopt()), layers=1, layer_types=('full_attention',),
                  global_head_dim=512)
    layer = SimpleNamespace(input_norm=None, q=None, k=None, q_norm=None, k_norm=None)
    target = object.__new__(Gemma4Target)
    target.config = cfg
    target.weights = SimpleNamespace(numerics='ct_nvfp4', layers=[layer])
    target._norm = lambda h, w, eps: h
    target._project = lambda *args: (None, None)
    target._inverse = [None]
    target._global_trig_prompt = prompt_trig
    target._global_trig = object() if trig else None
    target._global_prompt_scratch = object() if prepared else None
    state = SimpleNamespace(slot={0: (False, 0)}, raw=[None], inv=[None], pos=None, position=31)
    monkeypatch.setattr(attention, 'prep_full', lambda *args, **kw: None)

    class Observed(Exception):
        pass

    def raw(*args, **kwargs):
        assert expected == ('trig' if 'trig' in kwargs else 'raw')
        if 'trig' in kwargs:
            assert kwargs['trig'] is target._global_trig
        raise Observed

    def materialized(*args, **kwargs):
        assert expected == 'prepared'
        assert kwargs['scratch'] is target._global_prompt_scratch
        assert kwargs['length'] == 34
        assert kwargs['round_normalized'] is False
        raise Observed

    monkeypatch.setattr(attention, 'full', raw)
    monkeypatch.setattr(attention, 'full_prepared', materialized)
    with pytest.raises(Observed):
        target._layers(torch.zeros((3, cfg.hidden)), state, 3, 1, None, None, 0, False, False, prompt)
