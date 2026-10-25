"""CPU dispatch proof; native mathematics and GPU ownership have separate tests."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import family_native, family_ops
from tensorfold.families.qwen4_exp.cuda import attention, glue
from tensorfold.families.qwen4_exp.cuda.indexer_history import HistoryLayer


@pytest.fixture
def case():
    capacity, width, rows = 257, 128, 3
    owner = SimpleNamespace(limit=1024, check=lambda: None,
                            ring=torch.zeros(2, 12, width, dtype=torch.float16),
                            origins=torch.tensor([9, 19], dtype=torch.int64),
                            frames={kind: SimpleNamespace(gpu=torch.zeros(2, count, width, dtype=torch.float16))
                                    for kind, count in [('prefill', 64), ('decode', 16)]})
    return SimpleNamespace(history=HistoryLayer(owner, 1), capacity=capacity, rows=rows,
                           p=torch.zeros(rows, 1408, dtype=torch.float16),
                           pos=torch.tensor([21], dtype=torch.int32),
                           scale=torch.ones(width), freq=torch.ones(32),
                           q=torch.empty(rows, 3, width, dtype=torch.float16),
                           k=torch.empty(capacity, 1, width, dtype=torch.float16),
                           v=torch.empty(capacity, 1, width, dtype=torch.float16),
                           iq=torch.empty(rows, 2, width, dtype=torch.float16),
                           pooled=torch.empty(65, width, dtype=torch.float16),
                           status=torch.tensor([0, -2], dtype=torch.int32),
                           live=torch.tensor([2], dtype=torch.int32),
                           delta=torch.tensor([1], dtype=torch.int32), scratch=SimpleNamespace(ratio=4))


def prepare(case, *, kind='decode', capacity=None, **options):
    arguments = dict(q_heads=3, kv_heads=1, head_dim=128, index_heads=2, index_dim=128,
                     status=case.status, valid_rows=case.live, layer=-1, delta=case.delta,
                     history_kind=kind, history_capacity=case.capacity if capacity is None else capacity)
    arguments.update(options)
    return family_ops.attn_prep(case.p, case.pos, case.scale, case.scale, case.scale, case.freq,
                               case.q, case.k, case.v, case.iq, case.history, 1e-6, **arguments)


@pytest.mark.parametrize('kind', ['prefill', 'decode'])
@pytest.mark.parametrize('kb,vb,kf,vf', [(0, 0, 0, 0), (8, 8, 0, 0), (4, 4, 4, 5), (0, 4, 0, 4)])
def test_staged_preparation_preserves_formats_origin_and_fixed_stage_without_profile_gate(
        case, monkeypatch, kind, kb, vb, kf, vf):
    calls, marker = [], object()
    native = SimpleNamespace(prep_staged=lambda *args: calls.append(args) or marker)
    monkeypatch.setattr(family_native, 'extension', lambda: native)
    monkeypatch.setattr(family_ops, '_NATIVE_FAMILY', False)
    ks, vs = object(), object()
    assert prepare(case, kind=kind, k_bits=kb, v_bits=vb, k_codec=kf, v_codec=vf,
                   ks=ks, vs=vs, independent_rows=kind == 'decode') is marker
    args = calls[0]
    assert args[0] is case.p and args[7] is case.k and args[8] is case.v
    stage = case.history.stage(kind)
    assert args[10].data_ptr() == stage.data_ptr() and args[10].shape == stage.shape
    assert args[11] is (ks if kb else None) and args[12] is (vs if vb else None)
    assert args[13] is case.status and args[14] is case.delta and args[15] is case.live
    assert args[16:24] == (3, 1, 128, 2, 128, kb, vb, -1)
    assert args[26:29] == (kind == 'decode', kf, vf)
    assert args[29].data_ptr() == case.history.origin.data_ptr()


@pytest.mark.parametrize('kind', ['prefill', 'decode'])
def test_staged_pool_consumes_carry_before_ring_publication_and_keeps_current_logical_capacity(case, monkeypatch, kind):
    calls, marker = [], object()
    native = SimpleNamespace(pool_staged=lambda *args: calls.append(('pool', args)),
                             ring_publish=lambda *args: calls.append(('publish', args)) or marker)
    monkeypatch.setattr(family_native, 'extension', lambda: native)
    monkeypatch.setattr(family_ops, '_NATIVE_FAMILY', False)
    assert family_ops.qsa_pool(case.history, case.pooled, case.pos, case.scale, case.freq, 1e-6,
                               case.scratch, case.rows, history_kind=kind, history_capacity=case.capacity,
                               status=case.status, valid_rows=case.live, layer=-1, delta=case.delta) is marker
    assert [name for name, _ in calls] == ['pool', 'publish']
    pool, publish = calls[0][1], calls[1][1]
    assert pool[0].data_ptr() == case.history.stage(kind).data_ptr()
    assert pool[0].shape == case.history.stage(kind).shape
    assert pool[1].data_ptr() == case.history.ring.data_ptr()
    assert pool[2] is case.pooled and pool[3] is case.pos
    assert pool[6] is case.delta and pool[7:10] == (4, case.rows, case.capacity)
    # 257 current raw rows means 64 logical RMS rows, despite the 65 pooled
    # slots, 12 ring rows, small stage and 1024-row full archive reservation.
    assert pool[9] not in (65, 12, 16, 64, case.history.owner.limit)
    assert pool[12] is case.status and pool[13] is case.live and pool[14] == -1
    assert pool[15].data_ptr() == case.history.origin.data_ptr()
    assert publish[:2] == pool[:2]
    assert publish[2] is case.pos and publish[3:5] == (case.rows, case.capacity)
    assert publish[5] is case.status and publish[6] is case.live and publish[7] == -1
    assert publish[8] is pool[15]


@pytest.mark.parametrize('options,match', [
    ({'kind': None}, 'explicit prefill/decode'),
    ({'kind': 'guess'}, 'explicit prefill/decode'),
    ({'capacity': 256}, 'K/V cache capacity'),
    ({'rope': object()}, 'text rotary'),
])
def test_invalid_staged_preparation_is_rejected_before_loading_native(case, monkeypatch, options, match):
    def unexpected():
        pytest.fail('invalid dispatch reached native extension loading')
    monkeypatch.setattr(family_native, 'extension', unexpected)
    with pytest.raises(ValueError, match=match):
        prepare(case, **options)


@pytest.mark.parametrize('capacity', [None, 0, True, 1025])
def test_staged_pool_requires_current_bounded_capacity(case, monkeypatch, capacity):
    monkeypatch.setattr(family_native, 'extension', lambda: pytest.fail('invalid capacity reached native'))
    with pytest.raises(ValueError, match='current positive cache capacity'):
        family_ops.qsa_pool(case.history, case.pooled, case.pos, case.scale, case.freq, 1e-6,
                            case.scratch, case.rows, history_kind='decode', history_capacity=capacity)


@pytest.mark.parametrize('history', [False, True])
def test_glue_only_forwards_history_options_for_typed_history(case, monkeypatch, history):
    calls, marker = [], object()
    monkeypatch.setattr(glue._half_ops, 'attn_prep', lambda *args, **kwargs: calls.append((args, kwargs)) or marker)
    cache = case.history if history else torch.empty(case.capacity, 128, dtype=torch.float16)
    assert glue.attn_prep(case.p, case.pos, case.scale, case.scale, case.scale, case.freq,
                         case.q, case.k, case.v, case.iq, cache, 1e-6, q_heads=3, kv_heads=1,
                         head_dim=128, index_heads=2, index_dim=128,
                         history_kind='prefill', history_capacity=case.capacity) is marker
    options = calls[0][1]
    assert ('history_kind' in options) is history
    assert ('history_capacity' in options) is history
    if history:
        assert options['history_kind'] == 'prefill' and options['history_capacity'] == case.capacity


def test_bf16_glue_rejects_history_handle_before_triton_dispatch(case):
    with pytest.raises(ValueError, match='FP16 SM70'):
        glue.attn_prep(case.p.bfloat16(), case.pos, case.scale, case.scale, case.scale, case.freq,
                       case.q, case.k, case.v, case.iq, case.history, 1e-6, q_heads=3, kv_heads=1,
                       head_dim=128, index_heads=2, index_dim=128)


@pytest.mark.parametrize('history', [False, True])
def test_attention_pool_wrapper_forwards_state_controls_only_for_typed_history(case, monkeypatch, history):
    calls, marker = [], object()
    monkeypatch.setattr(attention._half_ops, 'qsa_pool', lambda *args, **kwargs: calls.append(kwargs) or marker)
    cache = case.history if history else torch.empty(case.capacity, 128, dtype=torch.float16)
    assert attention.qsa_pool(cache, case.pooled, case.pos, case.scale, case.freq, 1e-6,
                              case.scratch, case.rows, history_kind='decode', history_capacity=case.capacity,
                              status=case.status, valid_rows=case.live, layer=-1) is marker
    assert ('history_kind' in calls[0]) is history
    if history:
        assert calls[0]['history_capacity'] == case.capacity and calls[0]['layer'] == -1
        assert calls[0]['status'] is case.status and calls[0]['valid_rows'] is case.live
    else:
        assert not {'status', 'valid_rows', 'layer', 'history_capacity'} & calls[0].keys()


@pytest.mark.parametrize('history', [False, True])
def test_select_pools_before_scoring_and_preserves_explicit_phase_controls(case, monkeypatch, history):
    calls = []
    monkeypatch.setattr(attention, 'qsa_pool', lambda *args, **kwargs: calls.append(('pool', kwargs)))
    monkeypatch.setattr(attention, 'qsa_rows', lambda *args, **kwargs: calls.append(('rows', kwargs)))
    cache = case.history if history else torch.empty(case.capacity, 128, dtype=torch.float16)
    attention.qsa_select(case.iq, cache, case.pooled, case.pos, case.scale, case.freq, 1e-6,
                         case.scratch, case.rows, context=24, history_kind='prefill',
                         history_capacity=case.capacity, status=case.status, valid_rows=case.live, layer=-1)
    assert [name for name, _ in calls] == ['pool', 'rows']
    assert calls[1][1] == {'context': 24}
    assert ('history_kind' in calls[0][1]) is history
    if history:
        assert calls[0][1]['history_kind'] == 'prefill'
        assert calls[0][1]['history_capacity'] == case.capacity
        assert calls[0][1]['status'] is case.status and calls[0][1]['valid_rows'] is case.live


def test_tensor_pool_uses_unchanged_native_entrypoint(case, monkeypatch):
    calls, marker = [], object()
    native = SimpleNamespace(pool=lambda *args: calls.append(args) or marker)
    monkeypatch.setattr(family_ops, '_fused', lambda _: native)
    raw = torch.empty(case.capacity, 128, dtype=torch.float16)
    assert family_ops.qsa_pool(raw, case.pooled, case.pos, case.scale, case.freq, 1e-6,
                               case.scratch, case.rows, delta=case.delta) is marker
    assert len(calls[0]) == 10 and calls[0][0] is raw
    assert calls[0][6:8] == (4, case.rows)
