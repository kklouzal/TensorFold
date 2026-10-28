"""Catalogue cold funding/lifecycle and required eager-path controls."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda import capacity
from tensorfold.families.qwen4_exp.cuda import candidate_region_catalogue as module


@pytest.fixture
def setup(monkeypatch):
    log = []
    stream = SimpleNamespace(cuda_stream=1)
    cuda = SimpleNamespace(synchronize=lambda device: log.append('fence'),
                           graph_pool_handle=lambda: (0, 9),
                           is_current_stream_capturing=lambda: False)
    monkeypatch.setattr(module, 'torch', SimpleNamespace(cuda=cuda,
        are_deterministic_algorithms_enabled=lambda: False, is_autocast_enabled=lambda device: False))
    monkeypatch.setattr(module, 'host_memory', lambda: {'VmRSS': 100, 'VmHWM': 100})
    monkeypatch.setattr(capacity, 'admit_host_commitment', lambda n: log.append(('host', n)) or {'admitted': True})
    weights = SimpleNamespace(meta={'sm70_profile': module.PROFILE, 'execution_provider': 'sm70',
        'execution_dtype': 'float16', 'history_layout': 'unique', 'world': 8, 'vocab_offset': 0,
        'sm70_candidate_region_graphs_trial': True}, device='cuda:0', comm=SimpleNamespace(world=8),
        head=SimpleNamespace(n=64), draft_head=None, draft_ids=None, draft_filter=None, mtp=object())
    failed = []

    def buffer():
        return SimpleNamespace(prefill=False, rows=32, logits=torch.zeros(32, 64),
                               input_preparation_context=SimpleNamespace(fail=failed.append))

    dec = SimpleNamespace(w=weights, streams=[], filling=False, kept=[], free=[object() for _ in range(24)],
                          round_tables=SimpleNamespace(slots=24), buf=buffer(), mbuf=buffer())
    parent = SimpleNamespace(dec=dec, allocator_backend='native', side=stream, admitted=False,
                             capture_observation=lambda: len(log),
                             charge_capture=lambda before, label: log.append(('charge', label)))

    class Region:
        def __init__(self, weights, buffers, logits, rows, *, id_map, offset, side, pool):
            self.buffers, self.logits, self.rows, self.id_map = buffers, logits, rows, id_map
            self.graph, self.pool, self.prepared, self.closed, self.hits = object(), pool, False, False, 0

        def prepare(self):
            self.prepared = True

        def replay(self):
            self.hits += 1

        def _release_fenced(self):
            log.append(('release', self.rows))
            self.closed = True
            self.graph = None

    monkeypatch.setattr(module, 'CandidateRegion', Region)
    owner = module.Catalogue(parent)
    return owner, parent, cuda, log, failed


def test_exact64_graphs_share_only_candidate_pool_and_charge_before_publication(setup):
    owner, parent, _, log, _ = setup
    owner.prepare()
    assert owner.ready and len(owner.regions) == 64
    assert {region.pool for region in owner.regions.values()} == {(0, 9)}
    assert len([item for item in log if isinstance(item, tuple) and item[0] == 'charge']) == 64
    assert log[0] == ('host', 8 * module.HOST_ALLOWANCE)
    assert log[-1] == ('host', 8 * module.HOST_ALLOWANCE)
    with pytest.raises(RuntimeError, match='physical admission'):
        owner.local_packet(parent.dec.w, parent.dec.buf, parent.dec.buf.logits[:3], 3, None, 0)
    parent.admitted = True
    assert owner.local_packet(parent.dec.w, parent.dec.buf, parent.dec.buf.logits[:3], 3, None, 0)
    snapshot = owner.record()
    assert snapshot['frames']['target'][3] == 1
    owner.local_packet(parent.dec.w, parent.dec.buf, parent.dec.buf.logits[:3], 3, None, 0)
    assert snapshot['frames']['target'][3] == 1  # Telemetry snapshots do not alias mutable counters.


def test_nested_capture_and_other_buffers_keep_original_body(setup):
    owner, parent, cuda, _, _ = setup
    owner.prepare()
    cuda.is_current_stream_capturing = lambda: True
    assert not owner.local_packet(parent.dec.w, parent.dec.buf, parent.dec.buf.logits[:3], 3, None, 0)
    assert not owner.local_packet(parent.dec.w, object(), None, 3, None, 0)
    assert not owner.local_packet(parent.dec.w, parent.dec.buf, None, 33, None, 0)
    assert all(region.hits == 1 for region in owner.regions.values())


def test_profile_change_poison_notifies_after_releasing_packet_lease(setup):
    owner, parent, _, _, failed = setup
    owner.prepare()
    parent.admitted = True
    parent.dec.w.meta['execution_provider'] = 'other'
    with pytest.raises(RuntimeError, match='profile') as caught:
        owner.local_packet(parent.dec.w, parent.dec.buf, parent.dec.buf.logits[:3], 3, None, 0)
    assert owner.failure is caught.value and parent.dec._memory_error is caught.value
    assert failed == [caught.value] and not owner.lock.locked()


def test_capture_refusal_retains_partial_regions_until_fenced_close(setup):
    owner, parent, _, log, _ = setup

    def refuse(before, label):
        if label[-1] == 2:
            raise MemoryError('capture budget')

    parent.charge_capture = refuse
    with pytest.raises(MemoryError, match='budget'):
        owner.prepare()
    assert len(owner.regions) == 2 and owner.retiring and not owner.ready
    owner.close()
    assert owner.closed and not owner.regions and not owner.pools
    assert log[-3:] == ['fence', ('release', 1), ('release', 2)]


def test_failed_fence_preserves_all_graphs(setup):
    owner, _, cuda, _, _ = setup
    owner.prepare()
    cuda.synchronize = lambda device: (_ for _ in ()).throw(OSError('fence'))
    with pytest.raises(OSError, match='fence'):
        owner.close()
    assert len(owner.regions) == 64 and owner.pools and not owner.closed


def test_private_intent_and_profile_are_explicit(monkeypatch):
    monkeypatch.setenv(module.ENV, '1')
    with pytest.raises(ValueError, match='profile'):
        module.requested('other')
    assert module.requested(module.PROFILE)
    monkeypatch.setenv(module.ENV, '2')
    with pytest.raises(ValueError, match='0 or1'):
        module.requested(module.PROFILE)
