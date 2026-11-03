"""Live-only CT derived ownership, raw snapshots, admission and entry contracts."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from gemma_cuda_target_metadata_test import tiny_config
from tensorfold.families.gemma4.cuda import attention, capacity
from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
from tensorfold.families.gemma4.cuda.forward import Gemma4Target
from tensorfold.families.gemma4.cuda.state import State, capacity_bytes, derived_capacity_bytes
from tensorfold.families.gemma4.cuda.weights import Config
from test_gemma_cuda_modelopt_integration import _capacity_boundary


def config():
    return replace(Config.from_dict(tiny_config()), global_head_dim=512, global_kv_heads=2)


@pytest.mark.parametrize('mode', [True, None, '', 'KV', 'raw', 1])
def test_modes_are_strict_before_runtime(mode, tmp_path):
    with pytest.raises(ValueError, match='global_cache'):
        Gemma4Engine(tmp_path, global_cache=mode)
    with pytest.raises(ValueError, match='global_cache'):
        Gemma4Target(None, global_cache=mode)
    with pytest.raises(ValueError, match='global_cache'):
        capacity.admit(tmp_path, 128, True, global_cache=mode)
    with pytest.raises(ValueError, match='global_cache'):
        State(config(), 'cpu', 128, global_cache=mode)


@pytest.mark.parametrize('policy', ['mlx', 'modelopt_nvfp4'])
@pytest.mark.parametrize('mode', ['k', 'kv'])
def test_target_rejects_other_arithmetic(policy, mode):
    with pytest.raises(ValueError, match='native CT arithmetic'):
        Gemma4Target(SimpleNamespace(numerics=policy), global_cache=mode)


@pytest.mark.parametrize('mode,copies', [('k', 1), ('kv', 2)])
def test_live_bytes_raw_snapshots_restore_reset_and_rollback(mode, copies):
    cfg, owner = config(), object()
    state = State(cfg, 'cpu', 128, rows=5, global_cache=mode, cache_owner=owner)
    assert state.nbytes == capacity_bytes(cfg, 128, 5) + 4 + 128 * 2 * 512 * 2 * copies
    state.raw.fill_(3)
    state.inv.fill_(.5)
    state.keys.zero_()
    state.values.zero_()
    state.raw.fill_(3)
    state.inv.fill_(.5)
    state.advance(5)
    saved = state.snapshot()
    assert len(saved._layers) == 4
    assert saved._layers[2].shape[1] == 5
    assert saved._layers[2].data_ptr() != state.raw.data_ptr()
    pointers = (state.raw.data_ptr(), state.inv.data_ptr(), state.derived_keys.data_ptr(),
                None if state.derived_values is None else state.derived_values.data_ptr())
    state.raw.zero_()
    state.restore(saved)
    assert state.position == 5 and not state.global_cache_ready
    assert torch.equal(state.raw[:, :5], saved._layers[2])
    state.rollback(3)
    assert state.position == 3 and not state.global_cache_ready
    state.reset()
    assert state.position == 0 and state.global_cache_ready and not state.poisoned
    assert pointers == (state.raw.data_ptr(), state.inv.data_ptr(), state.derived_keys.data_ptr(),
                        None if state.derived_values is None else state.derived_values.data_ptr())


def owned_target(mode, state):
    target = object.__new__(Gemma4Target)
    target._global_cache, target._global_cache_owner = mode, state.cache_owner
    target.device = torch.device('cpu')
    target.weights = SimpleNamespace(layers=[SimpleNamespace(k_norm=object()) for _ in range(state.config.layers)])
    target._inverse = [object() for _ in range(state.config.layers)]
    return target


@pytest.mark.parametrize('mode', ['k', 'kv'])
def test_restore_prepares_each_full_layer_once_and_failure_poisoning(monkeypatch, mode):
    cfg = replace(config(), layers=3, layer_types=('full_attention',) * 3)
    state = State(cfg, 'cpu', 128, rows=5, global_cache=mode, cache_owner=object())
    state.keys.zero_()
    state.values.zero_()
    state.raw.fill_(3)
    state.inv.fill_(.5)
    state.advance(5)
    saved = state.snapshot()
    state.restore(saved)
    target = owned_target(mode, state)
    calls = []

    def prepare(raw, inv, norm, freq, keys, values, *, rows):
        assert rows == 5 and keys.shape == raw.shape
        assert (values is not None) == (mode == 'kv')
        calls.append(norm)

    monkeypatch.setattr(attention, 'prepare_global_cache', prepare)
    target.prepare_global_cache(state)
    assert state.global_cache_ready and len(calls) == 3
    target.prepare_global_cache(state)
    assert len(calls) == 3
    state.restore(saved)
    primary = RuntimeError('second layer preparation failed')

    def fail(*args, **kwargs):
        calls.append(None)
        if len(calls) == 5:
            raise primary

    monkeypatch.setattr(attention, 'prepare_global_cache', fail)
    with pytest.raises(RuntimeError) as caught:
        target.prepare_global_cache(state)
    assert caught.value is primary and state.poisoned and not state.global_cache_ready
    with pytest.raises(RuntimeError, match='poisoned'):
        target.prepare_global_cache(state)
    state.restore(saved)
    monkeypatch.setattr(attention, 'prepare_global_cache', prepare)
    target.prepare_global_cache(state)
    assert not state.poisoned and state.global_cache_ready


def test_foreign_preparation_owner_is_rejected_without_writes():
    state = State(config(), 'cpu', 128, global_cache='k', cache_owner=object())
    target = owned_target('k', state)
    target._global_cache_owner = object()
    with pytest.raises(ValueError, match='another target preparation contract'):
        target.prepare_global_cache(state)
    assert state.position == 0 and not state.poisoned


@pytest.mark.parametrize('mode,expected', [('k', 1342177280), ('kv', 2684354560)])
def test_five_layer_live_admission_not_snapshot_multiplied(monkeypatch, tmp_path, mode, expected):
    from tensorfold.families.gemma4.cuda import nvfp4_weights
    import tensorfold.families as families
    _capacity_boundary(monkeypatch, uma=True)
    cfg = replace(config(), layers=30, max_position=131072,
                  layer_types=('sliding_attention',) * 25 + ('full_attention',) * 5)
    monkeypatch.setattr(families, 'quant_method', lambda _: 'compressed-tensors')
    monkeypatch.setattr(nvfp4_weights, 'validate_headers', lambda *a, **kw: cfg)
    monkeypatch.setattr(nvfp4_weights, 'ct_tensor_contract', lambda _: ({}, {}))
    assert derived_capacity_bytes(cfg, 131072, mode) == expected
    for keep in (0, 3):
        for prompt, trig in ((False, False), (True, True)):
            kwargs = dict(max_rows=5, keep=keep, prepare_global_prompt=prompt, global_trig=trig)
            raw = capacity.admit(tmp_path, 131072, True, **kwargs)
            cached = capacity.admit(tmp_path, 131072, True, global_cache=mode, **kwargs)
            assert cached['geometry'].needed(131072) - raw['geometry'].needed(131072) == expected
            assert cached['weight'] == raw['weight'] and cached['host'] == raw['host']
    monkeypatch.setattr(families, 'quant_method', lambda _: 'modelopt')
    with pytest.raises(ValueError, match='native CT arithmetic'):
        capacity.admit(tmp_path, 131072, True, global_cache=mode)


@pytest.mark.parametrize('entry', ['forward', 'forward_staged', 'graph'])
def test_all_execution_entries_prepare_before_work(monkeypatch, entry):
    from tensorfold.families.gemma4.cuda.graphs import Graphs
    state = State(config(), 'cpu', 128, rows=5, global_cache='k', cache_owner=object())
    state.keys.zero_()
    state.values.zero_()
    state.raw.zero_()
    state.inv.fill_(1)
    state.advance(3)
    state.restore(state.snapshot())
    target = owned_target('k', state)
    target.config = state.config
    observed = []
    primary = RuntimeError('stop before execution')

    def prepare(actual):
        assert actual is state and not actual.global_cache_ready
        observed.append('prepare')
        raise primary

    monkeypatch.setattr(target, 'prepare_global_cache', prepare)
    if entry == 'graph':
        graphs = object.__new__(Graphs)
        graphs.target, graphs.state, graphs.max_rows = target, state, 5
        def run():
            return graphs.forward([4], state, output_taps=False)
    elif entry == 'forward_staged':
        def run():
            return target.forward_staged(torch.tensor([4]), state, chunks=1)
    else:
        def run():
            return target.forward(torch.tensor([4]), state)
    with pytest.raises(RuntimeError) as caught:
        run()
    assert caught.value is primary and observed == ['prepare']


@pytest.mark.parametrize('mode', ['k', 'kv'])
@pytest.mark.parametrize('prompt', [False, True])
@pytest.mark.parametrize('consume_prompt', [False, True])
@pytest.mark.parametrize('rows', [3, 17, 1024])
def test_append_all_rows_and_explicit_prompt_consumption(monkeypatch, mode, prompt, consume_prompt, rows):
    cfg = replace(config(), layers=1, layer_types=('full_attention',), max_position=2048)
    state = State(cfg, 'cpu', 2048, rows=rows, global_cache=mode, cache_owner=object())
    state.position = 31
    target = owned_target(mode, state)
    target._global_cache_prompt = consume_prompt
    target.config = cfg
    layer = SimpleNamespace(input_norm=None, q=None, k=None, q_norm=None, k_norm=None)
    target.weights = SimpleNamespace(numerics='ct_nvfp4', layers=[layer])
    target._norm = lambda h, w, eps: h
    target._project = lambda *args: (None, None)
    calls = []
    monkeypatch.setattr(attention, 'prep_full', lambda *args, **kw: calls.append('raw'))

    def append(*args, **kwargs):
        assert kwargs['rows'] == rows and kwargs['pos'] is state.pos
        assert args[4].data_ptr() == state.derived_keys[0].data_ptr()
        assert (args[5] is not None) == (mode == 'kv')
        calls.append('append')

    class Stop(Exception):
        pass

    def consume(*args, **kwargs):
        assert calls == ['raw', 'append']
        if prompt and not consume_prompt:
            assert 'prepared' not in kwargs and 'prepared_keys' not in kwargs
        elif mode == 'k':
            assert kwargs['prepared_keys'].data_ptr() == state.derived_keys[0].data_ptr()
        else:
            assert kwargs['prepared'][0].data_ptr() == state.derived_keys[0].data_ptr()
            assert kwargs['prepared'][1].data_ptr() == state.derived_values[0].data_ptr()
        raise Stop

    monkeypatch.setattr(attention, 'prepare_global_cache', append)
    monkeypatch.setattr(attention, 'full', consume)
    with pytest.raises(Stop):
        target._layers(torch.zeros((rows, cfg.hidden)), state, rows, 3, None, None, 0, False, False, prompt)


@pytest.mark.parametrize('entry', ['forward', 'forward_staged'])
def test_raw_target_cannot_advance_derived_state_then_return_it_stale(entry):
    state = State(config(), 'cpu', 128, rows=5, global_cache='k', cache_owner=object())
    owner = owned_target('k', state)
    raw_target = object.__new__(Gemma4Target)
    raw_target.config, raw_target.device = state.config, torch.device('cpu')
    if entry == 'forward':
        def run():
            return raw_target.forward(torch.tensor([4]), state)
    else:
        def run():
            return raw_target.forward_staged(torch.tensor([4]), state, chunks=1)
    with pytest.raises(ValueError, match='target'):
        run()
    assert state.position == 0 and state.global_cache_ready and not state.poisoned
    owner.prepare_global_cache(state)  # legitimate owner remains usable, no stale rows were published


@pytest.mark.parametrize('value', [None, 1, 'true'])
def test_prompt_policy_strict_bool_before_runtime(value, tmp_path):
    with pytest.raises(ValueError, match='global_cache_prompt must be a bool'):
        Gemma4Engine(tmp_path, global_cache='k', global_cache_prompt=value)
    with pytest.raises(ValueError, match='global_cache_prompt must be a bool'):
        Gemma4Target(None, global_cache='k', global_cache_prompt=value)
    with pytest.raises(ValueError, match='global_cache_prompt must be a bool'):
        capacity.admit(tmp_path, 128, True, global_cache='k', global_cache_prompt=value)


@pytest.mark.parametrize('kwargs', [
    {'global_cache': 'none'},
    {'global_cache': 'k', 'prepare_global_prompt': True},
    {'global_cache': 'kv', 'prepare_global_prompt': True},
    {'global_cache': 'k', 'global_trig': True},
    {'global_cache': 'kv', 'global_trig': True},
])
def test_prompt_policy_rejects_missing_or_redundant_owners(kwargs, tmp_path):
    with pytest.raises(ValueError, match='requires k/kv without shared prepared prompt or trig tables'):
        Gemma4Engine(tmp_path, global_cache_prompt=True, **kwargs)
    with pytest.raises(ValueError, match='requires k/kv without shared prepared prompt or trig tables'):
        Gemma4Target(None, global_cache_prompt=True, **kwargs)
    with pytest.raises(ValueError, match='requires k/kv without shared prepared prompt or trig tables'):
        capacity.admit(tmp_path, 128, True, global_cache_prompt=True, **kwargs)


@pytest.mark.parametrize('mode', ['k', 'kv'])
def test_prompt_policy_constructor_and_admission_have_no_added_buffers(monkeypatch, tmp_path, mode):
    from tensorfold.families.gemma4.cuda import nvfp4_weights
    import tensorfold.families as families
    cfg = config()
    weights = SimpleNamespace(numerics='ct_nvfp4', config=cfg, norm=torch.ones(cfg.hidden),
                              embed=SimpleNamespace(weight=torch.empty((1, cfg.hidden))))
    old = Gemma4Target(weights, global_cache=mode)
    new = Gemma4Target(weights, global_cache=mode, global_cache_prompt=True)
    a, b = old.make_state(128, rows=17), new.make_state(128, rows=17)
    assert a.nbytes == b.nbytes and new._global_cache_prompt and not old._global_cache_prompt
    assert new._global_prompt_scratch is None and new._global_trig is None
    _capacity_boundary(monkeypatch, uma=True)
    monkeypatch.setattr(families, 'quant_method', lambda _: 'compressed-tensors')
    monkeypatch.setattr(nvfp4_weights, 'validate_headers', lambda *a, **kw: cfg)
    monkeypatch.setattr(nvfp4_weights, 'ct_tensor_contract', lambda _: ({}, {}))
    for keep in (0, 3):
        prior = capacity.admit(tmp_path, 128, True, global_cache=mode, keep=keep)
        current = capacity.admit(tmp_path, 128, True, global_cache=mode, global_cache_prompt=True, keep=keep)
        assert prior['geometry'].needed(128) == current['geometry'].needed(128)
        assert prior['weight'] == current['weight'] and prior['host'] == current['host']
