"""Window dispatch/publication contracts; CPU tests make no CUDA parity claim."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.families.qwen4_exp.cuda import history_owner as history
from tensorfold.families.qwen4_exp.cuda import owner_preparation as preparation
from tensorfold.families.qwen4_exp.kv_formats import get_pair

FAMILY = 'sm70-paged-owner-rn-ex2-p2051-v1'


def fixture(rows, *, mtp=False, owned_pages=(0, 2), pos=127):
    cfg = NS(heads=3, head_dim=256, index_heads=4, index_dim=128,
             eps=1e-6, mrope_section=(11, 11, 10), rope_attention_factor=1.)
    placement = NS(group_size=4, page_tokens=128, target_layers=1,
        physical_page_rows=lambda capacity, page: min(128, capacity-page*128),
        indexer_owner=lambda *a, **k: 0)
    cache = history.OwnedKVCache(385, 1, 256, 'cpu', pair=get_pair('int8'),
        placement=placement, slot=0, layer=0, rank=0,
        owners=[0 if page in owned_pages else 1 for page in range(4)])
    for tensor in (cache.k, cache.v, cache.ks, cache.vs):
        tensor.fill_(7)
    events = []
    expected = (7, 3)
    class Entry:
        def pool(self, *args):
            events.append(('pool', args))

        def prepare_commit(self, end, generation):
            events.append(('prepare_commit', end, generation))
            return 'prepared'

        def publish_commit(self, prepared, end, generation):
            events.append(('publish_commit', prepared, end, generation))

    def append(*args, **kwargs):
        events.append(('append', args, kwargs))
        return Entry()
    state = NS(pos=pos, mtp_len=pos, att_index={0: 0}, kc=[cache], mtp_kc=cache,
        history_rank=0, history_placement=placement, history_slot=0,
        slot_generation=expected[0], history_generation=expected[1],
        history_generations=torch.tensor(expected), kv_status=torch.tensor([0, -2], dtype=torch.int32),
        compact_indexer=NS(append=append), pooled=[object()], mtp_pooled=object(),
        target_tentative_length=pos)
    frequency = torch.linspace(.0001, .03, 64)
    layer = NS(index=-1 if mtp else 0, attn=NS(q_scale=torch.ones(256),
        k_scale=torch.ones(256), iq_scale=torch.ones(128), ik_scale=object()))
    projection = torch.linspace(-.7, .8, rows*2688).half().reshape(rows, 2688)
    axes = torch.arange(rows*3, dtype=torch.int64).reshape(rows, 3)
    profile = NS(inv_freq=frequency, sections=(11, 11, 10), amplitude=1.)
    role = 'mtp' if mtp else 'decode'
    buffers = NS(prefill=False, owner_role=role, owner_kernel_family=FAMILY,
        owner_graph_family='uncaptured', history_owner=NS(owned_q8_preparation=None),
        q=torch.empty(rows, 3, 256, dtype=torch.float16),
        iq=torch.empty(rows, 4, 128, dtype=torch.float16),
        target_indexer_coordinates=axes, mtp_indexer_coordinates=axes,
        target_indexer_position=profile, mtp_indexer_position=profile,
        compact_indexer_stage=object())
    return projection, state, layer, NS(cfg=cfg, inv_freq=frequency), buffers, events


def tensors(item):
    _, state, _, _, buffers, _ = item
    cache = state.kc[0]
    return (buffers.q, buffers.iq, cache.k, cache.v, cache.ks, cache.vs, state.kv_status)


@pytest.mark.parametrize('mtp', [False, True])
@pytest.mark.parametrize('dispatch', ['leaf-loop-v1', 'grid-v1'])
@pytest.mark.parametrize('rows,owned_pages', [(1, (0,)), (5, ()), (129, (0, 2)), (256, (0, 1, 2))])
@torch.no_grad()
def test_native_row_views_and_window_versions_match_independent_reference(rows, owned_pages, mtp, dispatch):
    old = fixture(rows, mtp=mtp, owned_pages=owned_pages)
    new = fixture(rows, mtp=mtp, owned_pages=owned_pages)
    before = [[t._version for t in tensors(item)] for item in (old, new)]
    calls = []
    module = NS(owned_q8_prepare=lambda *args: calls.append(args),
                owned_q8_prepare_window=lambda *args: calls.append(args))
    p, state, layer, weights, buffers, _ = new
    history._prepare_reference(*old[:5], 0, rows, mtp=mtp)
    result = ops.owned_q8_prepare_window(module, p, buffers.target_indexer_coordinates,
        weights, state.kc[0], state, layer, buffers.q, buffers.iq,
        frequency=weights.inv_freq, sections=(11, 11, 10), amplitude=1., start=state.pos, dispatch=dispatch)
    assert result.shape == (rows, 128)
    step = rows if dispatch == 'grid-v1' else 1
    assert len(calls) == (1 if dispatch == 'grid-v1' else rows)
    for row, args in zip(range(0, rows, step), calls):
        for value, base in ((args[0], p), (args[1], buffers.target_indexer_coordinates),
                            (args[6], buffers.q), (args[7], buffers.iq)):
            assert value.shape[0] == step and value.is_contiguous()
            assert value.data_ptr() == base[row:row+step].data_ptr()
        assert args[17] == state.pos+row
        assert args[21] == layer.index
    deltas = [[t._version-v for t, v in zip(tensors(item), versions)]
              for item, versions in zip((old, new), before)]
    assert deltas[0] == deltas[1]
    assert deltas[0][-1] == 20  # Independently executed Torch window, not18*rows.


@pytest.mark.parametrize('mtp', [False, True])
@pytest.mark.parametrize('dispatch', ['leaf-loop-v1', 'grid-v1'])
@torch.no_grad()
def test_selected_roles_keep_one_raw_transaction_and_commit_only_mtp(monkeypatch, mtp, dispatch):
    item = fixture(5, mtp=mtp)
    p, state, _layer, _weights, buffers, events = item
    module = object()
    buffers.history_owner = NS(owned_q8_window_preparation=module,
        owned_q8_window_role=buffers.owner_role, rows=5, native_kernel_family=FAMILY,
        owned_q8_window_dispatch=dispatch)
    invocations = []
    def native(*args, **kwargs):
        invocations.append((args, kwargs))
        return p[:, -128:]
    monkeypatch.setattr(ops, 'owned_q8_prepare_window', native)
    history.prepare(*item[:5], 0, 5, mtp=mtp)
    assert len(invocations) == 1 and invocations[0][1]['start'] == 127
    assert invocations[0][1].get('dispatch', 'leaf-loop-v1') == dispatch
    assert [event[0] for event in events] == (['append', 'pool', 'prepare_commit', 'publish_commit']
                                              if mtp else ['append', 'pool'])
    assert events[0][1][0:2] == (1 if mtp else 0, 127)
    assert events[0][2] == {'mtp': mtp, 'expected': (7, 3)}
    assert (state.pos, state.mtp_len) == (127, 127)
    assert state.target_tentative_length == (127 if mtp else 132)
    if mtp:
        assert events[-2][1:] == (132, (7, 3))
        assert events[-1][1:] == ('prepared', 132, (7, 3))


@torch.no_grad()
def test_cross_row_alias_and_invalid_window_fail_before_guard_or_launch(monkeypatch):
    p, state, layer, weights, buffers, _ = fixture(5)
    module = NS(owned_q8_prepare=lambda *a: pytest.fail('native mutation'))
    monkeypatch.setattr(ops, '_guard', lambda *a: pytest.fail('guard mutation'))
    # Current row0 output aliases a later projection row, outside one-row checks.
    buffers.q = p.reshape(-1)[2688:2688+5*3*256].reshape(5, 3, 256)
    with pytest.raises(ValueError, match='separately owned'):
        ops.owned_q8_prepare_window(module, p, buffers.target_indexer_coordinates, weights,
            state.kc[0], state, layer, buffers.q, buffers.iq,
            frequency=weights.inv_freq, sections=(11, 11, 10), amplitude=1., start=127)
    buffers.q = torch.empty(5, 3, 256, dtype=torch.float16)
    with pytest.raises(ValueError, match='logical'):
        ops.owned_q8_prepare_window(module, p, buffers.target_indexer_coordinates, weights,
            state.kc[0], state, layer, buffers.q, buffers.iq,
            frequency=weights.inv_freq, sections=(11, 11, 10), amplitude=1., start=384)


def test_opt_in_is_strict_and_false_preserves_baseline_identity():
    spec = {'kernel_family': FAMILY, 'rn_heads_per_block': 12, 'graph_family': 'uncaptured',
            'graph_budget_bytes': 0, 'trial': True}
    assert preparation.parse_preparation(spec) == preparation.parse_preparation({**spec, 'owned_q8_window_prepare': False})
    assert preparation.parse_preparation({**spec, 'owned_q8_window_prepare': True})['owned_q8_window_prepare']
    for value in (None, 0, 1, 'true'):
        with pytest.raises(ValueError, match='Q8 window'):
            preparation.parse_preparation({**spec, 'owned_q8_window_prepare': value})
    with pytest.raises(ValueError, match='Q8 window'):
        preparation.parse_preparation({**spec, 'trial': False, 'owned_q8_window_prepare': True})


def test_grid_selector_requires_window_flag_and_preserves_leaf_loop_identity():
    spec = {'kernel_family': FAMILY, 'rn_heads_per_block': 12, 'graph_family': 'uncaptured',
            'graph_budget_bytes': 0, 'trial': True, 'owned_q8_window_prepare': True}
    assert preparation.parse_preparation(spec) == preparation.parse_preparation({**spec, 'owned_q8_window_dispatch': 'leaf-loop-v1'})
    assert preparation.parse_preparation({**spec, 'owned_q8_window_dispatch': 'grid-v1'})['owned_q8_window_dispatch'] == 'grid-v1'
    for value in (None, False, 1, 'grid', 'grid-v2'):
        with pytest.raises(ValueError, match='dispatch'):
            preparation.parse_preparation({**spec, 'owned_q8_window_dispatch': value})
    for flag in (False, None):
        unused = {**spec, 'owned_q8_window_dispatch': 'grid-v1'}
        if flag is None:
            unused.pop('owned_q8_window_prepare')
        else:
            unused['owned_q8_window_prepare'] = flag
        with pytest.raises(ValueError, match='dispatch'):
            preparation.parse_preparation(unused)


def test_grid_capability_is_checked_before_residency(monkeypatch):
    from contextlib import nullcontext

    from tensorfold.cuda.sm70 import family_native
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda d: (7, 0))
    monkeypatch.setattr(torch.cuda, 'device', lambda d: nullcontext())
    monkeypatch.setattr(family_native, 'extension', lambda: NS(
        prepare_owned_q8=lambda: pytest.fail('unavailable grid warmed old entry')))
    with pytest.raises(ValueError, match='built native window entry'):
        ops.prepare_owned_q8('cuda:0', window_dispatch='grid-v1')


@pytest.mark.parametrize('role', ['decode', 'mtp'])
@pytest.mark.parametrize('dispatch', ['leaf-loop-v1', 'grid-v1'])
def test_cold_capability_reuses_one_residency_operation(monkeypatch, role, dispatch):
    owner = object.__new__(history.OwnerBuffers)
    owner._empty_resources()
    owner.prefill, owner.device, owner.rows, owner.heads, owner.head_dim = False, 'cpu', 5, 3, 256
    owner.native_pair = get_pair('int8')
    owner.partials, owner.partial = torch.empty(8, 5, 12, 256), torch.empty(5, 12, 256)
    module = NS(__name__='resident')
    calls = []
    monkeypatch.setattr(ops, 'prepare_owned_q8', lambda d, **kw: (calls.append((d, kw)), module)[1])
    owner.configure_native_family(FAMILY, 12, owned_q8_window_prepare=True,
        owned_q8_window_role=role, owned_q8_window_dispatch=dispatch)
    assert calls == [('cpu', {'window_dispatch': 'grid-v1'} if dispatch == 'grid-v1' else {})]
    assert owner.owned_q8_window_preparation is owner.owned_q8_preparation is module
    assert owner.owned_q8_window_role == role
    assert owner.owned_q8_window_dispatch == dispatch
    with pytest.raises(RuntimeError, match='immutable'):
        owner.configure_native_family(FAMILY, 12, owned_q8_window_prepare=True, owned_q8_window_role=role)


@pytest.mark.parametrize('fault', ['prefill', 'rows', 'heads', 'dimensions', 'codec', 'role'])
def test_unavailable_explicit_request_fails_cold_before_residency(monkeypatch, fault):
    owner = object.__new__(history.OwnerBuffers)
    owner._empty_resources()
    owner.prefill, owner.device, owner.rows, owner.heads, owner.head_dim = False, 'cpu', 5, 3, 256
    owner.native_pair = get_pair('int8')
    role = 'decode'
    if fault == 'prefill':
        owner.prefill = True
    elif fault == 'rows':
        owner.rows = 257
    elif fault == 'heads':
        owner.heads = 12
    elif fault == 'dimensions':
        owner.head_dim = 128
    elif fault == 'codec':
        owner.native_pair = None
    else:
        role = 'prefill'
    monkeypatch.setattr(ops, 'prepare_owned_q8', lambda d: pytest.fail('cold native mutation'))
    with pytest.raises(ValueError, match='Q8 window'):
        owner.configure_native_family(FAMILY, 12, owned_q8_window_prepare=True, owned_q8_window_role=role)
    assert not owner._native_family_configured
    assert owner.owned_q8_window_preparation is None


@pytest.mark.parametrize('guard', ['grad-enabled', 'input-grad', 'weight-grad', 'inference-output'])
def test_unsupported_autograd_context_keeps_original_fallback_before_mutation(monkeypatch, guard):
    p, state, layer, weights, buffers, _ = fixture(2)
    if guard == 'input-grad':
        p.requires_grad_(True)
    elif guard == 'weight-grad':
        layer.attn.q_scale.requires_grad_(True)
    elif guard == 'inference-output':
        with torch.inference_mode():
            buffers.q = torch.empty_like(buffers.q)
    module = NS(owned_q8_prepare=lambda *a: pytest.fail('native mutation'))
    monkeypatch.setattr(ops, '_guard', lambda *a: pytest.fail('guard mutation'))
    with torch.set_grad_enabled(guard == 'grad-enabled'):
        assert ops.owned_q8_prepare_window(module, p, buffers.target_indexer_coordinates, weights,
            state.kc[0], state, layer, buffers.q, buffers.iq, frequency=weights.inv_freq,
            sections=(11, 11, 10), amplitude=1., start=127) is None
