"""Sparkulator factory/admission and committed-feature plumbing; CPU only."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
from tensorfold.families.gemma4.cuda import capacity
from tensorfold.families.gemma4.sparkulator_metadata import SparkulatorConfig

FIXTURE = Path(__file__).parent / 'fixtures/gemma4_sparkulator_26b_config.json'


def test_final_hidden_dispatch_uses_accepted_prefix_not_taps():
    engine = object.__new__(Gemma4Engine)
    observed = []
    engine.head = SimpleNamespace(feature_source='final_hidden', tap_layers=(),
        append=lambda state, hidden, **kwargs: observed.append((state, hidden, kwargs)))
    result = SimpleNamespace(hidden=torch.arange(24, dtype=torch.bfloat16).reshape(6, 4),
                             taps={1: torch.zeros(6, 4)})
    state = object()
    engine._append_head(state, result, [1] * 6, 2)
    assert observed[0][0] is state and observed[0][1].shape == (2, 4)
    assert torch.equal(observed[0][1], result.hidden[:2])


def test_prefill_and_decode_share_final_hidden_append_without_tap_requests(monkeypatch):
    from test_gemma_cuda_engine import _engine
    engine = _engine(monkeypatch, with_head=True)
    engine.head.feature_source, engine.head.tap_layers = 'final_hidden', ()
    engine._tap_features = False
    original = engine.target.forward
    roles = []
    def forward(tokens, state, **kwargs):
        assert kwargs.get('output_taps') is False
        result = original(tokens, state, **kwargs)
        result.taps = {1: result.hidden + 1000}  # would corrupt the head if the old tap route were used
        roles.append(kwargs.get('prompt', False))
        return result
    monkeypatch.setattr(engine.target, 'forward', forward)
    outputs = []
    engine.generate([3, 5, 8], 8, None, lambda ids: outputs.extend(ids), stop_eos=False)
    assert roles[0] is True and all(role is False for role in roles[1:])
    assert engine._draft.taps == engine._state.tokens
    assert len(outputs) == 8
    assert len(engine._draft.taps) == engine._state.position


def spark_admission(monkeypatch):
    from tensorfold import checkpoint_metadata
    from tensorfold.cuda import capacity as common
    from tensorfold.families.gemma4 import sparkulator_metadata
    from tensorfold.families.gemma4.cuda import modelopt_weights
    from tensorfold.families.gemma4.cuda.weights import Config
    raw = json.loads((Path(__file__).parent / 'fixtures/gemma26b_modelopt_metadata.json').read_text())['raw']
    cfg = Config.from_dict(raw)
    spark = SparkulatorConfig.from_dict(json.loads(FIXTURE.read_text()))
    monkeypatch.setattr(checkpoint_metadata, 'read_config', lambda _: raw)
    monkeypatch.setattr(capacity, 'companion_kind', lambda _: 'sparkulator')
    monkeypatch.setattr(common, 'headers', lambda _: {})
    monkeypatch.setattr(modelopt_weights, 'validate_headers', lambda *a, **k: cfg)
    monkeypatch.setattr(modelopt_weights, 'storage_bounds', lambda *a, **k: {
        'resident': 1000, 'device_staging': 200, 'host_staging': 300, 'mapped': 0})
    monkeypatch.setattr(sparkulator_metadata, 'checked_headers', lambda _: (spark, {}))
    monkeypatch.setattr(common, 'unified', lambda _: False)
    monkeypatch.setattr(common, 'admit_host_commitment', lambda _: None)
    def admit(directory, context, explicit, runtime, geometry, rule, **kwargs):
        weight, host = kwargs['weight_estimator'](directory)
        return {'context_window': context, 'geometry': geometry, 'weights': weight, 'host': host,
                'p_workspace': kwargs['device_workspace_bytes'],
                'host_workspace': kwargs['host_workspace_bytes']}
    monkeypatch.setattr(common, 'admit', admit)
    return cfg, spark


def test_spark_live_derived_cache_not_multiplied_by_snapshots(monkeypatch, tmp_path):
    cfg, spark = spark_admission(monkeypatch)
    small = capacity.admit(tmp_path, 65536, True, max_rows=7, keep=1, drafter_dir=tmp_path/'spark')
    large = capacity.admit(tmp_path, 65536, True, max_rows=7, keep=3, drafter_dir=tmp_path/'spark')
    from tensorfold.families.gemma4.cuda.state import capacity_bytes, widest_forward
    expected = 2 * (capacity_bytes(cfg, 65536, widest_forward(7)) + spark.snapshot_bytes(65536))
    assert large['geometry'].needed(65536) - small['geometry'].needed(65536) == expected
    assert small['weights'].resident == 1000 + spark.memory_bytes(16)[0]
    assert small['host'] >= spark.host_loading_bytes
    assert small['p_workspace'] > 0
    assert spark.workspace_bytes(65536) >= 6 * (1 << 30)  # native MATH fallback, no invented efficient guarantee


def test_spark_bits8_rejected_before_common_admission(monkeypatch, tmp_path):
    spark_admission(monkeypatch)
    from tensorfold.cuda import capacity as common
    monkeypatch.setattr(common, 'admit', lambda *a, **k: pytest.fail('allocation after invalid bits'))
    with pytest.raises(ValueError, match=r'requires.*(BF16|bits16)'):
        capacity.admit(tmp_path, 65536, True, max_rows=7, drafter_dir=tmp_path/'spark', drafter_bits=8)


def test_engine_factory_binds_after_state_and_prepares_no_tap_graphs(monkeypatch, tmp_path):
    _factory(monkeypatch, tmp_path)


def test_unsafe_spark_constructor_retains_external_status_without_cuda_cleanup(monkeypatch, tmp_path):
    _factory(monkeypatch, tmp_path, fail=True)


def _factory(monkeypatch, tmp_path, *, fail=False, global_gqa=False, sampled_q=False):
    from tensorfold import checkpoint_metadata, token_metadata
    from tensorfold.cuda import chat_template, precision
    from tensorfold.families.gemma4.cuda import forward, graphs, modelopt_weights, sparkulator
    from tensorfold.families.gemma4.cuda.weights import Config
    main = json.loads((Path(__file__).parent / 'fixtures/gemma26b_modelopt_metadata.json').read_text())['raw']
    native = json.loads(FIXTURE.read_text())
    cfg = Config.from_dict(main)
    draft_dir = tmp_path / 'spark'
    monkeypatch.setattr(checkpoint_metadata, 'read_config', lambda path: native if Path(path) == draft_dir else main)
    monkeypatch.setattr(precision, 'mode', lambda: precision.CHECKPOINT)
    events = []
    def admit(*args, **kwargs):
        assert kwargs['max_rows'] == 7 and kwargs['drafter_bits'] == 16
        assert kwargs.get('spark_global_gqa', False) is global_gqa
        assert kwargs.get('spark_sampled_q', False) is sampled_q
        return {'context_window': 65536}
    monkeypatch.setattr(capacity, 'admit', admit)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda _: None)
    monkeypatch.setattr(chat_template, 'prepare_chat_template', lambda *a, **k: object())
    monkeypatch.setattr(token_metadata, 'resolve_eos', lambda *a, **k: (1, 106, 50))
    monkeypatch.setattr(token_metadata, 'load_tokenizer', lambda _: SimpleNamespace(token_to_id=lambda _: None))
    monkeypatch.setattr(modelopt_weights, 'load', lambda *a, **k: SimpleNamespace(config=cfg))
    state = object()
    class Target:
        def __init__(self, weights):
            self.device, self.config = torch.device('cpu'), cfg
        def make_state(self, context, rows):
            events.append('main-state')
            return state
    monkeypatch.setattr(forward, 'Gemma4Target', Target)
    selected_gqa, selected_sampled = global_gqa, sampled_q
    primary = RuntimeError('synthetic unsafe head upload')
    partial = object()
    class Head:
        feature_source, tap_layers = 'final_hidden', ()
        def __init__(self, directory, target, context_limit, *, block, status, global_gqa=False, sampled_q=False):
            assert sampled_q is selected_sampled
            assert global_gqa is selected_gqa
            assert events == ['main-state'] and block == 7 and context_limit == 65536
            events.append('head')
            if fail:
                status.safe_to_release = False
                status.model = partial
                raise primary
        def bind_state(self, value):
            assert value is state
            events.append('bind')
        def make_state(self):
            events.append('head-state')
            return object()
    monkeypatch.setattr(sparkulator, 'Sparkulator', Head)
    class Graphs:
        def __init__(self, *args, **kwargs):
            events.append('graphs')
        def prepare(self, *, taps):
            assert taps == (False,)
            events.append('no-taps')
    monkeypatch.setattr(graphs, 'Graphs', Graphs)
    monkeypatch.setattr(torch.cuda, 'Stream', lambda **kw: SimpleNamespace(wait_event=lambda _: None))
    monkeypatch.setattr(torch.cuda, 'Event', lambda: SimpleNamespace(record=lambda: None))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: events.append('empty'))
    if fail:
        with pytest.raises(RuntimeError) as caught:
            Gemma4Engine(tmp_path, drafter=str(draft_dir), context=65536,
                         **({'spark_global_gqa': True} if global_gqa else {}),
                         **({'spark_sampled_q': True} if sampled_q else {}))
        assert caught.value is primary and events == ['main-state', 'head']
        retained = primary._tensorfold_retained_owners
        assert any(owner._spark_load_status.model is partial for owner in retained if isinstance(owner, Gemma4Engine))
    else:
        engine = Gemma4Engine(tmp_path, drafter=str(draft_dir), context=65536,
                         **({'spark_global_gqa': True} if global_gqa else {}),
                         **({'spark_sampled_q': True} if sampled_q else {}))
        assert events == ['main-state', 'head', 'bind', 'head-state', 'graphs', 'no-taps', 'empty']
        assert engine.max_rows == 7 and not engine._tap_features


def test_head_close_only_after_successful_main_stream_retirement():
    import threading
    engine = object.__new__(Gemma4Engine)
    engine._owner, engine._head_kind = None, 'sparkulator'
    engine._closed_lock, engine._prefix_lock = threading.Lock(), threading.Lock()
    engine._prefixes, engine._draft = [object()], object()
    engine._prefix_failed_bytes = 0
    events = []
    engine.drain = lambda: events.append('drain')
    engine.head = SimpleNamespace(close=lambda: events.append('head-close'))
    engine.close()
    assert events == ['drain', 'head-close'] and engine._draft is None
    primary = RuntimeError('unknown drain failure')
    engine._draft = object()
    events.clear()
    def failed():
        raise primary
    engine.drain = failed
    with pytest.raises(RuntimeError) as caught:
        engine.close()
    assert caught.value is primary and events == [] and engine._draft is not None


@pytest.mark.parametrize('flag', [None, 1, 'true'])
def test_gqa_selection_bool_boundary_before_runtime(flag, tmp_path):
    with pytest.raises(ValueError, match='bool'):
        Gemma4Engine(tmp_path, spark_global_gqa=flag)


def test_engine_true_only_forwards_gqa_to_admission_and_native_head(monkeypatch, tmp_path):
    _factory(monkeypatch, tmp_path, global_gqa=True)


def test_gqa_profile_changes_only_explicit_selected_workspace(monkeypatch, tmp_path):
    _, spark = spark_admission(monkeypatch)
    baseline = capacity.admit(tmp_path, 65536, True, max_rows=7, drafter_dir=tmp_path/'spark')
    candidate = capacity.admit(tmp_path, 65536, True, max_rows=7, drafter_dir=tmp_path/'spark', spark_global_gqa=True)
    assert candidate['weights'] == baseline['weights']
    assert baseline['geometry'].needed(65536) - candidate['geometry'].needed(65536) == (
        spark.workspace_bytes(65536) - spark.workspace_bytes(65536, global_gqa=True))
    assert 'spark_global_gqa' not in baseline and candidate['spark_global_gqa']['unqualified_prototype']


def test_gqa_refuses_other_companion_or_unqualified_context_before_allocation(monkeypatch, tmp_path):
    spark_admission(monkeypatch)
    from tensorfold.cuda import capacity as common
    monkeypatch.setattr(common, 'admit', lambda *a, **k: pytest.fail('allocation after unsupported GQA selection'))
    for context in (None, 0, True, 262144):
        with pytest.raises(ValueError, match='explicit context'):
            capacity.admit(tmp_path, context, True, max_rows=7, drafter_dir=tmp_path/'spark', spark_global_gqa=True)
    monkeypatch.setattr(capacity, 'companion_kind', lambda _: 'dflash')
    with pytest.raises(ValueError, match='actual Sparkulator'):
        capacity.admit(tmp_path, 65536, True, max_rows=7, drafter_dir=tmp_path/'df', spark_global_gqa=True)


@pytest.mark.parametrize('flag', [None, 1, 'true'])
def test_sampled_q_selection_bool_boundary_before_runtime(flag, tmp_path):
    with pytest.raises(ValueError, match='bool'):
        Gemma4Engine(tmp_path, spark_sampled_q=flag)


def test_engine_true_only_sampled_q_forwarding(monkeypatch, tmp_path):
    _factory(monkeypatch, tmp_path, global_gqa=True, sampled_q=True)


def test_sampled_q_adds_preparation_and_immutable_row_owners_without_changing_target(monkeypatch, tmp_path):
    _, spark = spark_admission(monkeypatch)
    from tensorfold.cuda.sampling import effective_workspace
    from tensorfold.families.gemma4.cuda import e2b_engine
    estimate, observed = 100003, []
    def retained(rows, width, *, proposal_policy):
        observed.append((rows, width, proposal_policy))
        return estimate
    monkeypatch.setattr(e2b_engine, 'assistant_q_host_bytes', retained)
    baseline = capacity.admit(tmp_path, 65536, True, max_rows=7, drafter_dir=tmp_path/'spark')
    selected = capacity.admit(tmp_path, 65536, True, max_rows=7, drafter_dir=tmp_path/'spark', spark_sampled_q=True)
    assert selected['weights'] == baseline['weights']
    assert selected['geometry'].needed(65536) == baseline['geometry'].needed(65536)
    assert selected['p_workspace'] - baseline['p_workspace'] == effective_workspace(1, spark.vocab, top_k=256)['device_bytes']
    assert selected['spark_sampled_q']['proposal_support_max'] == 256
    assert observed == [(6, 256, True)]
    assert selected['spark_sampled_q']['retained_q_host_bytes'] == estimate
    assert selected['host_workspace'] - baseline['host_workspace'] == (
        effective_workspace(1, spark.vocab, top_k=256)['host_bytes'] + estimate)
    assert 'spark_sampled_q' not in baseline


def test_sampled_q_wrong_pair_and_overwidth_refused_before_common(monkeypatch, tmp_path):
    spark_admission(monkeypatch)
    from tensorfold.cuda import capacity as common
    monkeypatch.setattr(common, 'admit', lambda *a, **k: pytest.fail('allocation after unsupported sampled-q selection'))
    with pytest.raises(ValueError, match='2..16'):
        capacity.admit(tmp_path, 65536, True, max_rows=17, drafter_dir=tmp_path/'spark', spark_sampled_q=True)
    monkeypatch.setattr(capacity, 'companion_kind', lambda _: 'dflash')
    with pytest.raises(ValueError, match='actual Sparkulator'):
        capacity.admit(tmp_path, 65536, True, max_rows=7, drafter_dir=tmp_path/'df', spark_sampled_q=True)


def test_gqa_wrong_torch_rejected_before_engine_device_or_main_weights(monkeypatch, tmp_path):
    from tensorfold import checkpoint_metadata
    from tensorfold.families.gemma4.cuda import modelopt_weights
    native = json.loads(FIXTURE.read_text())
    main = json.loads((Path(__file__).parent / 'fixtures/gemma26b_modelopt_metadata.json').read_text())['raw']
    directory = tmp_path/'spark'
    monkeypatch.setattr(checkpoint_metadata, 'read_config', lambda path: native if Path(path) == directory else main)
    monkeypatch.setattr(torch.version, 'git_version', 'unqualified-test-git')
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a: pytest.fail('device work before Torch proof'))
    monkeypatch.setattr(modelopt_weights, 'load', lambda *a, **k: pytest.fail('main payload before Torch proof'))
    with pytest.raises(ValueError, match='qualified Torch git'):
        Gemma4Engine(tmp_path, drafter=str(directory), context=131072, spark_global_gqa=True)


def test_gqa_wrong_torch_rejected_before_capacity_header_or_small_budget(monkeypatch, tmp_path):
    from tensorfold.cuda import capacity as common
    monkeypatch.setattr(torch.version, 'git_version', 'unqualified-test-git')
    monkeypatch.setattr(common, 'headers', lambda *a: pytest.fail('header/candidate budget before Torch proof'))
    monkeypatch.setattr(common, 'admit', lambda *a, **k: pytest.fail('common admission before Torch proof'))
    with pytest.raises(ValueError, match='qualified Torch git'):
        capacity.admit(tmp_path, 131072, True, spark_global_gqa=True)


def test_disabled_gqa_never_queries_qualified_torch_guard(monkeypatch, tmp_path):
    from tensorfold.families.gemma4 import sparkulator_metadata
    monkeypatch.setattr(sparkulator_metadata, 'validate_global_gqa_torch',
                        lambda *a: pytest.fail('disabled GQA queried optional runtime guard'))
    with monkeypatch.context() as factory_patch:
        _factory(factory_patch, tmp_path)
    spark_admission(monkeypatch)
    capacity.admit(tmp_path, 65536, True, max_rows=7, drafter_dir=tmp_path/'spark')


def test_gqa_geometry_plans_requested_window_with_larger_native(monkeypatch, tmp_path):
    spark_admission(monkeypatch)
    from tensorfold.cuda.capacity import make_plan, choose

    result = capacity.admit(tmp_path, 131072, True, max_rows=7,
                            drafter_dir=tmp_path/'spark', spark_global_gqa=True)
    geometry = result['geometry']
    assert geometry.maximum_slots == 131072
    # The real profile raises if common planning probes an unsupported slot count.
    plan = make_plan(262144, 131072, True, 10**15, result['weights'], geometry)
    assert choose(plan) == plan.largest == 131072
    assert plan.native == 262144
    baseline = capacity.admit(tmp_path, 131072, True, max_rows=7, drafter_dir=tmp_path/'spark')
    assert baseline['geometry'].maximum_slots is None
