"""Owned configuration/API boundaries; these controls do not simulate CUDA arithmetic."""
import copy
import json
import warnings
from types import SimpleNamespace as NS

import pytest

from tensorfold import cli
from tensorfold.cuda import native_allocator as policy
from tensorfold.cuda.allocation_bound import UnsupportedAllocator


@pytest.fixture
def clean_environment(monkeypatch):
    for name in policy.ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def fake_runtime(*, backend='native', capability=(12, 1), changed=None, segments=None, setter=None):
    state = {'allocator_settings': policy._fresh_settings(), 'segments': [] if segments is None else segments}
    if changed is not None:
        changed(state)
    calls = []
    def snapshot():
        calls.append('snapshot')
        return copy.deepcopy(state)
    def configure(value):
        calls.append(('setter', value))
        if setter is not None:
            setter(state)
        else:
            state['allocator_settings'].update(expandable_segments=True, PYTORCH_CUDA_ALLOC_CONF=value)
    runtime = NS(cuda=NS(memory=NS(get_allocator_backend=lambda: backend, _snapshot=snapshot),
                        get_device_capability=lambda device: capability),
                 _C=NS(_accelerator_setAllocatorSettings=configure))
    return runtime, calls, state


def test_one_setting_after_fresh_proof_without_allocating_a_tensor(clean_environment):
    runtime, calls, _ = fake_runtime()
    runtime.cuda.is_initialized = lambda: pytest.fail('initialized state does not prove process ownership')
    result = policy.enable_fresh_expansion(runtime)
    assert calls == ['snapshot', ('setter', policy.SETTING), 'snapshot']
    assert result['policy'] == 'expandable' and result['backend'] == 'native'
    assert result['settings'] == dict(policy._fresh_settings(), expandable_segments=True,
                                      PYTORCH_CUDA_ALLOC_CONF=policy.SETTING)


@pytest.mark.parametrize('name', policy.ENVIRONMENT)
@pytest.mark.parametrize('value', ['', '0', '1'])
def test_explicit_override_refused_before_native_inspection(clean_environment, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    runtime, calls, _ = fake_runtime()
    with pytest.raises(UnsupportedAllocator, match='environment'):
        policy.enable_fresh_expansion(runtime)
    assert calls == []


@pytest.mark.parametrize('kind', ['backend', 'device', 'segments', 'schema', 'setting', 'boolean', 'divisions', 'segment-schema'])
def test_nonfresh_or_unsupported_policy_refused_before_setter(clean_environment, kind):
    def changed(state):
        if kind == 'schema':
            state['allocator_settings']['unknown'] = 1
        elif kind == 'setting':
            state['allocator_settings']['max_split_size'] = 64 << 20
        elif kind == 'boolean':
            state['allocator_settings']['pinned_num_register_threads'] = True
        elif kind == 'divisions':
            state['allocator_settings']['roundup_power2_divisions']['1'] = True
        elif kind == 'segment-schema':
            state['segments'] = {}
    runtime, calls, _ = fake_runtime(backend='cudaMallocAsync' if kind == 'backend' else 'native',
        capability=(12, 0) if kind == 'device' else (12, 1), segments=[{'device': 1}] if kind == 'segments' else None,
        changed=changed)
    with pytest.raises(UnsupportedAllocator):
        policy.enable_fresh_expansion(runtime)
    assert not any(type(call) is tuple for call in calls)


@pytest.mark.parametrize('kind', ['parse', 'warning', 'fallback', 'unrelated', 'allocation'])
def test_failure_is_terminal_and_no_second_setting_or_reset(clean_environment, kind):
    primary = RuntimeError('partial parse')
    def configure(state):
        state['allocator_settings']['PYTORCH_CUDA_ALLOC_CONF'] = policy.SETTING
        if kind == 'parse':
            raise primary
        if kind == 'warning':
            warnings.warn('unsupported expansion', UserWarning, stacklevel=1)
        state['allocator_settings']['expandable_segments'] = kind != 'fallback'
        if kind == 'unrelated':
            state['allocator_settings']['release_lock_on_cudamalloc'] = True
        if kind == 'allocation':
            state['segments'] = [{'device': 0}]
    runtime, calls, _ = fake_runtime(setter=configure)
    expected = RuntimeError if kind == 'parse' else UserWarning if kind == 'warning' else UnsupportedAllocator
    with pytest.raises(expected) as caught:
        policy.enable_fresh_expansion(runtime)
    if kind == 'parse':
        assert caught.value is primary
    assert [call for call in calls if type(call) is tuple] == [('setter', policy.SETTING)]


def test_cli_has_one_positive_option_and_default_is_absent():
    parser = cli.build_parser()
    assert not hasattr(parser.parse_args(['serve', 'owner/model']), 'native_allocator')
    assert parser.parse_args(['serve', 'owner/model', '--native-expandable-segments']).native_allocator == 'expandable'
    with pytest.raises(SystemExit):
        parser.parse_args(['serve', 'owner/model', '--no-native-expandable-segments'])


@pytest.mark.parametrize('change', ['mlx', 'family', 'streams', 'tp', 'full', 'affine', 'fp8'])
def test_cli_rejects_unsupported_explicit_request_before_weight_resolution(tmp_path, clean_environment, change):
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5',
        'quantization_config': {'quant_method': 'modelopt'}}))
    family = NS(model_type='qwen3_5', title='Qwen dense', package=NS())
    args = cli.build_parser().parse_args(['serve', str(tmp_path), '--native-expandable-segments'])
    backend = 'mlx' if change == 'mlx' else 'cuda'
    if change == 'family':
        family.model_type = 'qwen4_exp'
    elif change == 'streams':
        args.parallel = '2'
    elif change == 'tp':
        args.tp = 2
    elif change == 'full':
        args.precision = 'full'
    elif change == 'affine':
        (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5'}))
    elif change == 'fp8':
        args.prefill_fp8 = True
    with pytest.raises(ValueError, match='native-expandable-segments'):
        cli._check_serve_options(args, family, backend, tmp_path)


def test_cli_validates_environment_before_native_metadata_or_downloads(tmp_path, clean_environment, monkeypatch):
    family = NS(model_type='qwen3_5', title='Qwen dense', package=NS())
    args = cli.build_parser().parse_args(['serve', str(tmp_path), '--native-expandable-segments'])
    monkeypatch.setenv('PYTORCH_ALLOC_CONF', '')
    with pytest.raises(UnsupportedAllocator, match='environment'):
        cli._check_serve_options(args, family, 'cuda', tmp_path)


@pytest.mark.parametrize('requested', [None, 'expandable'])
def test_dense_factory_forwards_only_the_explicit_policy(tmp_path, monkeypatch, requested):
    from tensorfold.families import qwen3_5
    from tensorfold.families.qwen3_5.cuda import engine

    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5'}))
    monkeypatch.setattr(qwen3_5, 'gb10', lambda: True)
    monkeypatch.setattr(engine, 'Qwen27Engine', lambda *args, **kwargs: kwargs)
    options = {} if requested is None else {'native_allocator': requested}
    result = qwen3_5.cuda_engine(tmp_path, no_drafts=True, **options)
    if requested is None:
        assert 'native_allocator' not in result
    else:
        assert result['native_allocator'] == requested
