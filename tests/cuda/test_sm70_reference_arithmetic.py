"""Pinned SM70 graph bit/resource/lease controls; no model/P3 qualification."""
import json

import pytest
import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.sm70.reference_owner_tiles import ReferenceTiles, prepare
from tensorfold.families.qwen4_exp.cuda.reference_arithmetic import ReferenceArithmetic
from tensorfold.families.qwen4_exp.cuda.history_owner import (
    OwnedKVCache, _encode, partial_statistics, normalized_partial,
)
from tensorfold.families.qwen4_exp.cuda.history_placement import HistoryPlacement
from tensorfold.families.qwen4_exp.kv_formats import get_pair

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='requires actual pinned SM70 arithmetic capture')


def bits(actual, expected):
    assert torch.equal(actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32))


def retire_and_record(owner, tiles, record_testsuite_property, label, primary=None):
    # Publish measured residency even when explicit-budget admission failed.
    # Failure of the diagnostic sink must not prevent safe retirement or mask
    # the preparation primary; failed close retains the borrowed slab owners.
    cleanup = []
    try:
        record = owner.record()
        record['failure_type'] = None if owner.failure is None else type(owner.failure).__name__
        record_testsuite_property(label, json.dumps(record, sort_keys=True))
    except BaseException as error:
        cleanup.append(error)
    try:
        owner.close()
        tiles.begin_retirement()
        tiles.release_views()
    except BaseException as error:
        cleanup.append(error)
    raise_failures(primary, cleanup)


@pytest.fixture(scope='module')
def arithmetic(record_testsuite_property):
    prepare()
    slab = torch.empty((8, 1, 12, 256), dtype=torch.float32, device='cuda')
    tiles = ReferenceTiles(slab)
    # This nominal test allowance is the actual available physical device
    # capacity, not an assumed graph footprint or a serving-budget default.
    budget = int(torch.cuda.mem_get_info(slab.device)[0])
    assert 0 < budget < 2**63
    owner = ReferenceArithmetic(tiles, budget_bytes=budget)
    primary = None
    try:
        owner.prepare(quiescent=lambda: True)
        assert owner.charged_bytes <= budget
        assert set(owner.graphs) == {'stats', 'probability', 'pv'}
        owner.validate_adoption()
        yield owner, tiles
    except BaseException as error:
        primary = error
    finally:
        retire_and_record(owner, tiles, record_testsuite_property, 'reference_arithmetic_nominal_preparation', primary)


def test_explicit_insufficient_budget_refuses_with_retained_record_and_safe_close(record_testsuite_property):
    prepare()
    slab = torch.empty((8, 1, 12, 256), dtype=torch.float32, device='cuda')
    tiles = ReferenceTiles(slab)
    owner = ReferenceArithmetic(tiles, budget_bytes=1)
    primary = None
    try:
        with pytest.raises(ValueError, match='explicit aggregate preparation allowance') as refused:
            owner.prepare(quiescent=lambda: True)
        assert owner.failure is refused.value
        assert not owner.prepared and not owner.closed
        assert owner.charged_bytes > owner.budget == 1
        assert owner.after is not None and owner.inputs and owner.outputs and len(owner.graphs) == 3
    except BaseException as error:
        primary = error
    finally:
        retire_and_record(owner, tiles, record_testsuite_property, 'reference_arithmetic_one_byte_refusal', primary)
    assert owner.closed and owner.inputs == owner.outputs == {} and tiles.workspace is None


@pytest.mark.parametrize('format_name', ['int8', 'rotorquant-iso4-sm70-v1'])
@pytest.mark.parametrize('count', [1, 127, 128, 129, 256, 259, 2051])
def test_actual_three_islands_match_original_two_pass_bits_and_changed_inputs(arithmetic, format_name, count):
    run_case(arithmetic, format_name, count)


@pytest.mark.parametrize('mode', ['empty', 'nonfinite', 'signed-zero'])
def test_exceptional_inputs_preserve_original_raw_bits_and_status(arithmetic, mode):
    run_case(arithmetic, 'int8', 129, mode)


def run_case(arithmetic, format_name, count, mode='normal'):
    owner, tiles = arithmetic
    pair = get_pair(format_name)
    placement = HistoryPlacement(8, 2, 12, degree=4)
    cache = OwnedKVCache(4096, 1, 256, 'cuda', pair=pair, placement=placement, slot=0, layer=0, rank=0)
    torch.manual_seed(3811)
    key = torch.randn((4096, 1, 256), dtype=torch.float16, device='cuda') * .03
    value = torch.randn_like(key) * .02
    status = torch.tensor([0, -2], dtype=torch.int32, device='cuda')
    k, ks = _encode(key, pair.key, status, 3, error=2)
    v, vs = _encode(value, pair.value, status, 3, error=4)
    cache.append(0, k, v, ks, vs)
    generations = torch.zeros(2, dtype=torch.int64, device='cuda')
    context = tiles.context(status=status, generations=generations, slot_generation=0, owner_generation=0, layer=3)
    query = torch.randn((12, 256), dtype=torch.float16, device='cuda') * .02
    if mode == 'nonfinite':
        query[0, 0] = torch.inf
        query[1, 0] = torch.nan
    elif mode == 'signed-zero':
        query.zero_()
        query[:, ::2] = -0.0
    selected = torch.arange(count, device='cuda', dtype=torch.int64)
    maximum = torch.linspace(-1, 1, 12, device='cuda')
    denominator = torch.linspace(.5, 2, 12, device='cuda')
    stats = torch.empty((12, 2), device='cuda')
    destinations = (stats[:, 0], stats[:, 1])
    result = torch.empty((12, 256), device='cuda')
    for stale in (False, True):
        generations[1].fill_(int(stale))
        visible = selected % 7 != 0
        if mode == 'empty':
            visible.zero_()
        status.copy_(torch.tensor([0, -2], dtype=torch.int32, device='cuda'))
        expected_stats = partial_statistics(query, cache, selected, visible, .0625, tile_preparer=context)
        expected_values = normalized_partial(query, cache, selected, visible, .0625, maximum, denominator, tile_preparer=context)
        expected_status = status.clone()
        status.copy_(torch.tensor([0, -2], dtype=torch.int32, device='cuda'))
        tiles.arithmetic = owner
        try:
            actual_stats = partial_statistics(query, cache, selected, visible, .0625, tile_preparer=context, out=destinations)
            actual_values = normalized_partial(query, cache, selected, visible, .0625, maximum, denominator,
                                                tile_preparer=context, out=result)
            assert actual_stats is destinations and actual_values is result
            for actual, expected in zip(actual_stats, expected_stats):
                bits(actual, expected)
            bits(actual_values, expected_values)
            assert torch.equal(status, expected_status)
        finally:
            tiles.arithmetic = None
        query.mul_(.5)
        maximum.add_(.25)
        denominator.mul_(1.5)
        selected = selected.flip(0)
