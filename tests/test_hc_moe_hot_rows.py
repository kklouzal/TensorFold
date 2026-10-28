"""Actual hot-row catalogue factory/ownership plumbing; math and admission are separate GPU gates."""
from types import SimpleNamespace as N

import pytest
import torch

from tensorfold.cuda import capacity
from tensorfold.cuda.sm70.execution_policy import SHARED_FUSION_TRIAL
from tensorfold.families.qwen4_exp.cuda import hc_moe_region as leaf
from tensorfold.families.qwen4_exp.cuda import moe_region_catalogue as module
from test_moe_region_catalogue import model


@pytest.fixture
def factory(monkeypatch):
    w, b = model()
    w.meta.update(sm70_profile=SHARED_FUSION_TRIAL, sm70_hc_moe_region_graphs_trial=True,
                  sm70_hc_moe_shared_pool_trial=True)
    b.h, b.pss, b.inj_a = torch.zeros(32, 4), torch.zeros(32, 4), torch.zeros(32, 4)
    b.g_branch = torch.zeros(8 * 32 * 3)
    side = N(synchronize=lambda: None)
    parent = N(dec=N(w=w, buf=b, streams=[], filling=False, kept=[], free=list(range(24)),
                     round_tables=N(slots=24)), admitted=False, allocator_backend='native',
               side=side, bytes=0, future_bytes=0, budget=1 << 40,
               capture_observation=lambda: {'general_available_bytes': 1 << 40})
    phases, pools, role_calls = [], [], []
    def charge(before, key):
        phases.append(key)
        parent.bytes += 1
        return {'key': key, 'charged_bytes': 1}
    parent.charge_capture = charge
    monkeypatch.setattr(module, '_host_memory', lambda: {'VmRSS': 100, 'VmHWM': 100})
    monkeypatch.setattr(capacity, 'admit_host_commitment', lambda count: {'admitted': True})
    monkeypatch.setattr(module.torch.cuda, 'synchronize', lambda *args: None)
    monkeypatch.setattr(module.torch.cuda, 'current_stream', lambda *args: N(cuda_stream=7))
    def pool():
        value = (0, len(pools) + 100)
        pools.append(value)
        return value
    monkeypatch.setattr(module.torch.cuda, 'graph_pool_handle', pool)
    monkeypatch.setattr(module, 'on_stream', lambda cuda, side, operation: operation())
    monkeypatch.setattr(module, 'validate_region', lambda *args: None)
    monkeypatch.setattr(leaf, 'validate_hc_region', lambda *args: None)
    class Bindings:
        def __init__(self, *args):
            self.validations = 0
        def validate(self):
            self.validations += 1
    monkeypatch.setattr(leaf, 'HCModelBindings', Bindings)
    class Roles:
        def __init__(self, weights, rows):
            self.owners = weights
            role_calls.append((weights, rows))
    monkeypatch.setattr(module, 'RolePlans', Roles)
    class Region:
        def __init__(self, weights, layer, buffers, rows, *, side, pool, capture_router):
            self.pool, self.router, self.layer, self.rows = pool, capture_router, layer, rows
            self.branch = buffers.g_branch[:8 * rows * 3].view(8, rows, 3)
            self.injection = buffers.inj_a[:rows]
            self.branch_signature = module._span(self.branch)
            self.injection_signature = module._span(self.injection)
            self.graph, self.hits = object(), 0
        def prepare(self):
            pass
        def replay(self, mode, branch, injection):
            assert mode == 3 and branch is self.branch and injection is self.injection
        def replay_prevalidated(self):
            self.hits += 1
        def _release_fenced(self):
            self.graph = None
    monkeypatch.setattr(leaf, 'HCReadoutMoERegion', Region)
    return parent, phases, pools, role_calls


def test_actual_cold_factory_has768_keys769_phases_and_one_uniquely_owned_pool(factory):
    parent, phases, pools, roles = factory
    cat = module.Catalogue(parent)
    parent.moe_regions = cat
    cat.prepare()
    assert cat.ready and cat.independent and cat.row_keys == module.HOT_ROWS
    assert len(cat.regions) == 768 and set(cat.regions) == {(r, i) for r in module.HOT_ROWS for i in range(48)}
    assert len(pools) == 1 and cat.pools == {'shared': pools[0]}
    assert {region.pool for region in cat.regions.values()} == {pools[0]}
    assert len(roles) == 1 and roles[0][1] == 32 and len(roles[0][0]) == 48
    assert all(region.router is cat.capture_router for region in cat.regions.values())
    assert len(phases) == len(cat.measured) == cat.plan['maximum_graphs'] + 1 == 769
    assert cat.plan['future_host_bytes_per_rank'] == 1 << 30
    record = cat.record()
    assert record['rows'] == module.HOT_ROWS
    assert record['graph_contract'] == 'mlp-hc-mode3-moe-hot16-independent-pool-v1'
    assert record['funding']['pool_policy'] == 'one-independent-no-private-output-pool'


def test_hot_rows_preserve_all_n1_keys_and_full_ordered_issued_cut_cycles(factory):
    parent, _, _, _ = factory
    cat = module.Catalogue(parent)
    cat.prepare()
    parent.admitted = True  # CPU plumbing proxy only; actual Engine admission remains a GPU gate.
    w, b = parent.dec.w, parent.dec.buf
    for rows in (1, 2, 3, 7, 12, 16, 29):
        assert not cat.begin(w, b, rows) and not cat.lease and not cat.frame_lock.locked()
    for rows in (5, 32, 9):
        assert cat.begin(w, b, rows)
        for layer in w.layers:
            region = cat.regions[rows, layer.index]
            assert cat.defer_hc(layer, b, rows, 3, region.branch, region.injection)
            assert cat.run(layer, b, rows)
        cat.finish()
    assert cat.frames == 3 and cat.hits == 144 and not cat.lease
    assert {r: v for r, v in cat.row_frames.items() if v} == {5: 1, 9: 1, 32: 1}


def test_old_cold_factory_keeps_three_distinct_row_pools_when_optin_off(factory):
    parent, phases, pools, _ = factory
    parent.dec.w.meta['sm70_hc_moe_shared_pool_trial'] = False
    cat = module.Catalogue(parent)
    cat.prepare()
    assert not cat.independent and cat.row_keys == module.ROWS
    assert len(cat.regions) == 144 and len(phases) == 145 and len(pools) == 3
    assert set(cat.pools) == {30, 31, 32}
    assert all(region.pool == cat.pools[r] for (r, _), region in cat.regions.items())


def test_shared_optin_requires_existing_fused_hc_policy(monkeypatch):
    for name in (module.ENV, module.HC_ENV, module.SHARED_ENV):
        monkeypatch.setenv(name, '1')
    assert module.requested_shared(SHARED_FUSION_TRIAL)
    with pytest.raises(ValueError):
        module.requested_shared(module.PROFILE)
    monkeypatch.setenv(module.HC_ENV, '0')
    with pytest.raises(ValueError):
        module.requested_shared(SHARED_FUSION_TRIAL)
    monkeypatch.setenv(module.SHARED_ENV, '2')
    with pytest.raises(ValueError):
        module.requested_shared(SHARED_FUSION_TRIAL)
    assert module.funding(independent=True)['maximum_graphs'] == 768
    assert module.funding()['maximum_graphs'] == 144
    with pytest.raises(ValueError):
        module.funding(independent=1)


def test_actual_accounting_counts_shared_family_once_and_rejects_cross_family_alias(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import multi_graphs
    owner = N(failure=None, failure_frames=None, allocator_backend='native', pool=(0, 1),
              dec=N(w=N(device=N(index=0))), moe_regions=N(pools={'shared': (0, 2)}),
              candidate_regions=N(pools={'candidate-finish': (0, 3)}), private_spare_bytes=None)
    visited = []
    def snapshot(*, mempool_id, include_traces):
        visited.append(mempool_id)
        return [{'device': 0, 'segment_pool_id': mempool_id,
                 'total_size': 100 + 10 * mempool_id[1], 'allocated_size': 100, 'active_size': 100}]
    monkeypatch.setattr(multi_graphs.torch.cuda, 'memory_snapshot', snapshot)
    monkeypatch.setattr(multi_graphs.torch.cuda, 'memory_reserved', lambda: 1000)
    monkeypatch.setattr(multi_graphs.torch.cuda, 'memory_allocated', lambda: 500)
    monkeypatch.setattr(multi_graphs, 'free_bytes', lambda: 5000)
    monkeypatch.setattr(multi_graphs, 'live_bytes', lambda: 5500)
    multi_graphs.RoundGraphs._account_pool(owner)
    assert visited == [(0, 1), (0, 2), (0, 3)]
    assert owner.private_spare_bytes == owner.allocator_record['private_pool_spare_bytes'] == 60
    assert [r['family'] for r in owner.private_pool_records] == ['n1', 'moe-target-shared', 'candidate-candidate-finish']
    owner.candidate_regions.pools['candidate-finish'] = (0, 2)
    with pytest.raises(MemoryError, match='cannot be accounted'):
        multi_graphs.RoundGraphs._account_pool(owner)
