"""CPU preparation/operation protocol and eager oracle; no graph/GPU proof."""
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.reference_arithmetic import (
    ReferenceArithmetic, statistics_step, probability_step, value_step,
)
from tensorfold.families.qwen4_exp.cuda.owner_preparation import (
    ArithmeticRequest, OwnerPreparation, parse_preparation, ARITHMETIC_POLICY,
)
from tensorfold.families.qwen4_exp.cuda import history_owner as reference

pytestmark = pytest.mark.torch


def specification(request=None):
    return {'kernel_family': 'sm70-paged-owner-q8-compact-v1', 'graph_family': 'uncaptured',
            'graph_budget_bytes': 0, 'trial': False,
            **({'reference_arithmetic': request} if request is not None else {})}


@pytest.mark.parametrize('budget', [None, 0, -1, True, 2**63, 1.5])
def test_explicit_positive_preparation_budget_required(budget):
    with pytest.raises(ValueError):
        ArithmeticRequest.parse({'policy': ARITHMETIC_POLICY, 'graph_budget_bytes': budget})


def test_private_capability_and_typed_budget_default_off():
    base = specification()
    assert 'reference_arithmetic' not in parse_preparation(base)
    request = {'policy': ARITHMETIC_POLICY, 'graph_budget_bytes': 1234567}
    prepared = parse_preparation(specification(request))
    assert prepared['reference_arithmetic'] == request
    with pytest.raises(ValueError, match='trusted private'):
        OwnerPreparation(prepared, role_rows={'decode': 160, 'prefill': 256}, manifest_sha256='a' * 64)
    owner = OwnerPreparation(prepared, role_rows={'decode': 160, 'prefill': 256},
                             manifest_sha256='a' * 64, reference_tiles=True)
    assert owner.reference_arithmetic_request == ArithmeticRequest(1234567)


class Tiles:
    def __init__(self, count, arithmetic=None):
        torch.manual_seed(3001 + count)
        self.keys = torch.randn((count, 256), dtype=torch.float16) * .01
        self.v = torch.randn_like(self.keys) * .02
        self.half = torch.empty((128, 256), dtype=torch.float16)
        self.mask = torch.empty(128, dtype=torch.bool)
        self.workspace = self.half.view(torch.float32)
        self.arithmetic, self.events = arithmetic, []

    @contextmanager
    def tile(self, cache, positions, visible):
        count = positions.numel()
        self.events.append(('keys', count))
        self.half[:count].copy_(self.keys[positions])
        self.mask[:count].copy_(visible)
        owner = self
        class Tile:
            key = owner.half[:count]
            mask = owner.mask[:count]
            def values(self):
                owner.events.append(('values', count))
                owner.half[:count].copy_(owner.v[positions])
                return owner.half[:count]
        item = Tile()
        item.count = count
        yield item


def eager_protocol(tiles, monkeypatch):
    owner = ReferenceArithmetic(tiles, budget_bytes=1)  # Explicit CPU protocol fixture, never capture.
    owner.prepared = True
    owner.inputs = {'query': torch.zeros((12, 256), dtype=torch.float16),
                    'maximum': torch.zeros(12), 'denominator': torch.zeros(12),
                    'global_maximum': torch.zeros(12), 'global_denominator': torch.ones(12),
                    'result': torch.zeros((12, 256))}
    def stats():
        assert owner._operation_lease.locked()
        tiles.events.append(('stats', 128))
        owner.outputs['stats'] = statistics_step(owner.inputs['query'], tiles.half, tiles.mask,
                                                 owner.inputs['maximum'], owner.inputs['denominator'], .0625)
    def probability():
        assert owner._operation_lease.locked()
        tiles.events.append(('probability', 128))
        owner.outputs['probability'] = probability_step(owner.inputs['query'], tiles.half, tiles.mask,
                                                       owner.inputs['global_maximum'], owner.inputs['global_denominator'], .0625)
    def pv():
        assert owner._operation_lease.locked()
        tiles.events.append(('pv', 128))
        owner.outputs['pv'] = value_step(owner.outputs['probability'], tiles.half, owner.inputs['result'])
    owner.graphs = {name: NS(replay=call, reset=lambda: None)
                    for name, call in [('stats', stats), ('probability', probability), ('pv', pv)]}
    monkeypatch.setattr(owner, 'eligible', lambda *a: True)  # CPU sequencing only, not CUDA substitution.
    tiles.arithmetic = owner
    return owner


def bits(actual, expected):
    assert torch.equal(actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32))


@pytest.mark.parametrize('count', [0, 1, 127, 128, 129, 256, 259, 2048, 2051])
def test_full_tile_three_stage_order_feedback_tail_and_owned_handoff(count, monkeypatch):
    old, tiles = Tiles(count), Tiles(count)
    arithmetic = eager_protocol(tiles, monkeypatch)
    query = torch.randn((12, 256), dtype=torch.float16) * .01
    ids, visible = torch.arange(count), torch.arange(count) % 7 != 0
    cache = NS(capacity=max(1, count))
    expected = reference.partial_statistics(query, cache, ids, visible, .0625, tile_preparer=old)
    destination = torch.empty((12, 2))
    target = (destination[:, 0], destination[:, 1])
    actual = reference.partial_statistics(query, cache, ids, visible, .0625, tile_preparer=tiles, out=target)
    assert actual is target
    for a, e in zip(actual, expected):
        bits(a, e)
    maximum, denominator = torch.linspace(-1, 1, 12), torch.linspace(.5, 2, 12)
    expected_pv = reference.normalized_partial(query, cache, ids, visible, .0625, maximum, denominator, tile_preparer=old)
    pv = torch.empty((12, 256))
    returned = reference.normalized_partial(query, cache, ids, visible, .0625, maximum, denominator, tile_preparer=tiles, out=pv)
    assert returned is pv
    bits(pv, expected_pv)
    assert not arithmetic._operation_lease.locked()
    for index, event in enumerate(tiles.events):
        if event[0] == 'probability':
            assert tiles.events[index + 1:index + 3] == [('values', 128), ('pv', 128)]
    assert arithmetic.replays['stats'] == arithmetic.replays['probability'] == arithmetic.replays['pv'] == count // 128


def test_operation_reentry_retirement_and_alias_refuse_before_shared_input_write(monkeypatch):
    tiles = Tiles(128)
    owner = eager_protocol(tiles, monkeypatch)
    query, ids, visible = torch.ones((12, 256), dtype=torch.float16), torch.arange(128), torch.ones(128, dtype=torch.bool)
    out = (torch.empty(12), torch.empty(12))
    before = owner.inputs['query'].clone()
    owner._operation_lease.acquire()
    try:
        with pytest.raises(RuntimeError, match='borrowed'):
            owner.statistics(query, NS(capacity=128), ids, visible, .0625, tiles, out=out)
        with pytest.raises(RuntimeError, match='active operation'):
            owner.close()
        assert torch.equal(before, owner.inputs['query'])
    finally:
        owner._operation_lease.release()
    with pytest.raises(ValueError, match='disjoint'):
        owner.statistics(query, NS(capacity=128), ids, visible, .0625, tiles,
                         out=(owner.inputs['maximum'], owner.inputs['denominator']))
    owner.close()
    assert owner.closed and owner.inputs == owner.outputs == {}


def test_adoption_storage_resize_rebind_and_retirement_failure_keep_owners(monkeypatch):
    tiles = Tiles(128)
    owner = eager_protocol(tiles, monkeypatch)
    owner.tensor_leases = [owner._lease(t) for t in (tiles.half, tiles.mask, *owner.inputs.values())]
    owner.validate_adoption()
    owner.inputs['query'] = owner.inputs['query'].clone()
    with pytest.raises(RuntimeError, match='catalogue'):
        owner.validate_adoption()
    owner.inputs['query'] = owner.tensor_leases[2][0]
    storage = owner.inputs['query'].untyped_storage()
    storage.resize_(storage.nbytes() + 128)
    with pytest.raises(RuntimeError, match='storage'):
        owner.validate_adoption()
    error = OSError('graph reset failed')
    owner.graphs['stats'].reset = lambda: (_ for _ in ()).throw(error)
    with pytest.raises(OSError, match='graph reset failed'):
        owner.close()
    assert owner.failure is error and not owner.closed
    assert owner.tiles is tiles and owner.inputs and owner.outputs == {} and owner.tensor_leases
    assert 'stats' in owner.graphs and not owner._operation_lease.locked()
    owner.graphs['stats'].reset = lambda: None
    owner.close()
    assert owner.closed and not owner.tensor_leases


def test_unpaid_or_overlapping_handoff_never_mutates_shared_inputs(monkeypatch):
    tiles = Tiles(128)
    owner = eager_protocol(tiles, monkeypatch)
    query = torch.ones((12, 256), dtype=torch.float16)
    cache, ids, visible = NS(capacity=128), torch.arange(128), torch.ones(128, dtype=torch.bool)
    assert owner.statistics(query, cache, ids, visible, .0625, tiles) is None
    assert not owner.inputs['query'].count_nonzero()
    output = torch.empty(1).expand(12, 256)
    with pytest.raises(ValueError, match='disjoint'):
        owner.normalized(query, cache, ids, visible, .0625, torch.zeros(12), torch.ones(12), tiles, out=output)
    assert not owner.inputs['query'].count_nonzero()


def test_query_alignment_class_preserved_with_eager_fallback_before_staging(monkeypatch):
    tiles = Tiles(128)
    owner = ReferenceArithmetic(tiles, budget_bytes=1)
    owner.prepared, owner.prepared_stream = True, object()
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: owner.prepared_stream)
    storage = torch.empty(12 * 256 + 256, dtype=torch.float16)
    offset = (-storage.data_ptr() // storage.element_size()) % 256
    aligned = storage[offset:offset + 12 * 256].view(12, 256)
    misaligned = storage[offset + 1:offset + 1 + 12 * 256].view(12, 256)
    cache, ids = NS(capacity=128), torch.arange(128)
    assert aligned.data_ptr() % 512 == 0 and misaligned.is_contiguous()
    assert owner.eligible(aligned, cache, ids, .0625)
    assert not owner.eligible(misaligned, cache, ids, .0625)
    assert owner.statistics(misaligned, cache, ids, ids >= 0, .0625, tiles,
                            out=(torch.empty(12), torch.empty(12))) is None
    assert owner.inputs == {} and owner.failure is None
