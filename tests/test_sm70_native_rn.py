"""NativeRN policy/phase/slab contracts; no device or analytical-bound claim."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import paged_attention as native
from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController
from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
from tensorfold.families.qwen4_exp.cuda.owner_execution import (
    KERNEL_FAMILY, RN_KERNEL_FAMILY, RN_GRAPH_FAMILY, GRAPH_FAMILY, REFERENCE,
)
from tensorfold.families.qwen4_exp.cuda.owner_preparation import parse_preparation
from tensorfold.families.qwen4_exp.kv_formats import get_pair


def specification(heads=1, graph=False):
    return {'kernel_family': RN_KERNEL_FAMILY, 'graph_family': RN_GRAPH_FAMILY if graph else 'uncaptured',
            'graph_budget_bytes': 100 if graph else 0, 'rn_heads_per_block': heads, 'trial': True}


@pytest.mark.parametrize('heads', [1, 4, 12])
@pytest.mark.parametrize('graph', [False, True])
def test_distinct_numerical_and_five_phase_graph_identity_require_explicit_heads(heads, graph):
    parsed = parse_preparation(specification(heads, graph))
    assert parsed['rn_heads_per_block'] == heads
    assert parsed['kernel_family'] != KERNEL_FAMILY
    assert parsed['graph_family'] != GRAPH_FAMILY


@pytest.mark.parametrize('change', [{'rn_heads_per_block': True}, {'rn_heads_per_block': 0},
                                   {'rn_heads_per_block': 3}, {'rn_heads_per_block': None},
                                   {'kernel_family': []}, {'graph_family': GRAPH_FAMILY},
                                   {'graph_family': RN_GRAPH_FAMILY}])
def test_invalid_numerical_contract_rejected_before_preparation(change):
    with pytest.raises(ValueError):
        parse_preparation({**specification(), **change})


def test_old_family_cannot_silently_inherit_rn_geometry_or_graph_identity():
    with pytest.raises(ValueError):
        parse_preparation({**specification(), 'kernel_family': KERNEL_FAMILY})
    with pytest.raises(ValueError):
        parse_preparation({**specification(graph=True), 'kernel_family': KERNEL_FAMILY})


def test_controller_selects_only_actual_prepared_family_and_retains_reference_default():
    trial = {'kernel_family': RN_KERNEL_FAMILY, 'graph_family': 'uncaptured'}
    control = AdaptiveController(slots=8, row_budget=40, proposal_limit=4, prefill_tokens=256)
    with pytest.raises(ValueError, match='not prepared'):
        control.configure_execution_families(native=True, trial=trial)
    assert control.record()['execution_families']['effective'] == REFERENCE
    assert not control.native_prepared
    control.configure_execution_families(native=True, native_family=RN_KERNEL_FAMILY)
    assert control.effective_family == REFERENCE
    control.configure_execution_families(native=True, native_family=RN_KERNEL_FAMILY, graph_rows=40,
                                         trial={'kernel_family': RN_KERNEL_FAMILY, 'graph_family': RN_GRAPH_FAMILY})
    record = control.record()['execution_families']
    assert record['prepared_kernel_family'] == RN_KERNEL_FAMILY and record['effective']['graph_family'] == RN_GRAPH_FAMILY
    with pytest.raises(ValueError, match='not prepared'):
        control._check_family({'kernel_family': KERNEL_FAMILY, 'graph_family': 'uncaptured'}, 40)
    with pytest.raises(ValueError, match='graph family'):
        control._check_family({'kernel_family': RN_KERNEL_FAMILY, 'graph_family': GRAPH_FAMILY}, 40)
    control._check_family(dict(REFERENCE), 40)


def workspace(rows=5):
    weights = NS(cfg=NS(heads=3, head_dim=256), device='cpu', meta={'rank': 0},
                 comm=NS(world=8, rank=0, all_gather=None))
    result = OwnerBuffers(weights, rows, NS(idw=2052))
    result.prepare_native(pair=get_pair('int8'))
    return result


@pytest.mark.parametrize('rows', [1, 5, 40, 160, 256])
def test_half_probability_borrows_only_receive_slab_and_exact_prefix(rows):
    owner = workspace(rows)
    old_bytes = owner.nbytes
    owner.configure_native_family(RN_KERNEL_FAMILY, 1)
    for count in (1, rows):
        view = owner.native_views(count)
        p = view['probability']
        assert p.dtype == torch.float16 and p.shape == (count, 12, 2051) and p.is_contiguous()
        assert p.untyped_storage().data_ptr() == owner.partials.data_ptr()
        assert not torch._C._overlaps(p, view['local_values'])
        assert p.nbytes == native.rn_probability_bytes(count) <= view['all_values'].nbytes
    assert old_bytes == owner.nbytes
    with pytest.raises(RuntimeError, match='immutable'):
        owner.configure_native_family(KERNEL_FAMILY)
    owner.close()



@pytest.mark.parametrize('family', [None, KERNEL_FAMILY, RN_KERNEL_FAMILY], ids=['default', 'WMMA', 'RN'])
@pytest.mark.parametrize('rows', [1, 5])
def test_native_view_returns_keep_order_shape_stride_aliases_and_freshness(family, rows):
    owner = workspace(rows)
    try:
        if family is not None:
            owner.configure_native_family(family, 1 if family == RN_KERNEL_FAMILY else None)
        policy = (owner.native_kernel_family, owner.native_heads_per_block, owner._native_family_configured)
        paid = owner.nbytes
        keys = ('q_all', 'selected_all', 'row_table', 'local_stats', 'all_stats', 'global_stats',
                'local_values', 'all_values', 'out')
        if family == RN_KERNEL_FAMILY:
            keys = ('probability',)+keys
        for count in sorted({1, rows}):
            views = owner.native_views(count)
            assert tuple(views) == keys
            shapes = {'q_all': (8, count, 3, 256),
                      'selected_all': (8, count, owner.selection.shape[1]),
                      'row_table': (count, owner.descriptors.shape[1]),
                      'local_stats': (count, 12, 2), 'all_stats': (8, count, 12, 2),
                      'global_stats': (count, 12, 2), 'local_values': (count, 12, 256),
                      'all_values': (8, count, 12, 256), 'out': (count, 3, 256)}
            storage = {'q_all': owner.queries, 'selected_all': owner.selections,
                       'row_table': owner.descriptors, 'local_stats': owner.partial,
                       'all_stats': owner.partials, 'global_stats': owner.global_stats,
                       'local_values': owner.partial, 'all_values': owner.partials, 'out': owner.queries}
            if family == RN_KERNEL_FAMILY:
                shapes['probability'] = (count, 12, 2051)
                storage['probability'] = owner.partials
            for name, shape in shapes.items():
                tensor = views[name]
                assert tuple(tensor.shape) == shape and tensor.is_contiguous() and tensor.storage_offset() == 0
                stride, expected_stride = 1, []
                for dimension in reversed(shape):
                    expected_stride.append(stride)
                    stride *= dimension
                assert tensor.stride() == tuple(reversed(expected_stride))
                assert tensor.untyped_storage().data_ptr() == storage[name].untyped_storage().data_ptr()
                assert tensor.dtype == (torch.float16 if name == 'probability' else storage[name].dtype)
            again = owner.native_views(count)
            assert again is not views and tuple(again) == keys
            views.pop('out')
            assert tuple(owner.native_views(count)) == keys
        assert owner.nbytes == paid
        assert (owner.native_kernel_family, owner.native_heads_per_block, owner._native_family_configured) == policy
        for invalid in (False, 0, rows+1):
            with pytest.raises(ValueError, match='prepared bounded'):
                owner.native_views(invalid)
    finally:
        owner.close()


def test_foreign_or_insufficient_probability_storage_rejected_before_policy_publication():
    owner = workspace()
    owner.partials = owner.partial
    with pytest.raises(ValueError, match='disjoint'):
        owner.configure_native_family(RN_KERNEL_FAMILY, 1)
    assert owner.native_kernel_family == KERNEL_FAMILY and not owner._native_family_configured
    owner.close()


def test_five_phases_keep_probability_live_until_all_pv_reads_then_receive_overwrite(monkeypatch):
    owner = workspace()
    owner.configure_native_family(RN_KERNEL_FAMILY, 4)
    view = owner.native_views(5)
    events = []
    def gather(send, received):
        events.append('gather')
        received.copy_(send.expand(8, *send.shape))
    owner.native_comm.all_gather = gather
    def stats(q, selected, table, out, **controls):
        assert controls['heads_per_block'] == 4
        events.append('stats')
        out.fill_(1)
    def merge(gathered, table, out, **controls):
        assert torch.all(gathered == 1)
        events.append('merge')
        out.fill_(2)
    def probability(q, selected, table, global_stats, out, **controls):
        assert torch.all(global_stats == 2)
        events.append('probability')
        out.fill_(.5)
    def pv(probability, selected, table, out, **controls):
        assert torch.all(probability == .5)
        assert not torch._C._overlaps(probability, out)
        events.append('pv')
        out.fill_(3)
    def finish(gathered, table, out, **controls):
        assert torch.all(gathered == 3)
        events.append('finish')
        out.fill_(4)
    for name, call in [('rn_local_statistics', stats), ('rn_merge_statistics', merge),
                       ('rn_probabilities', probability), ('rn_partial_values', pv), ('merge_values', finish)]:
        monkeypatch.setattr(native, name, call)
    returned = owner.native_run(5)
    assert returned.data_ptr() == view['out'].data_ptr() and torch.all(returned == 4)
    assert events == ['stats', 'gather', 'merge', 'probability', 'pv', 'gather', 'finish']
    owner.close()


@pytest.mark.parametrize('heads', [True, 0, 3, None])
def test_invalid_block_geometry_never_acquires_native_code(monkeypatch, heads):
    monkeypatch.setattr(native, 'extension', lambda: pytest.fail('bad heads acquired native module'))
    with pytest.raises(ValueError, match='explicitly'):
        native.rn_local_statistics(None, None, None, None, group_start=0, page_tokens=128, layer=None,
                                   key_bits=8, key_codec=0, heads_per_block=heads)


def test_rn_source_metadata_cannot_relabel_an_old_native_binary(monkeypatch):
    from tensorfold.cuda import build
    from tensorfold.cuda.sm70 import provider, source_cache, paged_attention_descriptor as descriptor
    monkeypatch.setattr(provider, '_native', lambda: NS(__name__='pinned-source-foundation'))
    monkeypatch.setattr(source_cache, 'stage_sources', lambda sources, **kwargs: [str(p) for p in sources])
    old = NS(descriptor_abi=descriptor.ABI_VERSION, descriptor_words=descriptor.WORDS,
             descriptor_schema_id=descriptor.SCHEMA_ID)
    monkeypatch.setattr(build, 'load', lambda *args, **kwargs: old)
    native.extension.cache_clear()
    try:
        with pytest.raises(RuntimeError, match='numerical-family'):
            native.extension()
    finally:
        native.extension.cache_clear()


def test_source_owned_dense_tail_domain_keeps_2051_pitch_without_changing_old_family():
    identity = native.rn_preparation_identity()
    assert identity['max_dense'] == identity['max_selected'] == native.RN_MAX_SELECTED == 2051
    assert native.rn_probability_bytes(256) == 49224 * 256
    assert KERNEL_FAMILY == 'sm70-paged-owner-q8-compact-v1'
