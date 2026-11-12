"""CPU-only owner selection, paid arenas and ordered phase orchestration."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import attention_backends as backend, paged_attention as native
from tensorfold.families.qwen4_exp.cuda import owner_preparation as preparation
from tensorfold.families.qwen4_exp.cuda.owner_execution import GRAPH_FAMILY


def specification(**changes):
    value = dict(kernel_family=preparation.ATEN_KERNEL_FAMILY, graph_family=preparation.UNCAPTURED,
                 graph_budget_bytes=0, trial=True, aten_tiles={'panel_rows': 2, 'tensor_budget_bytes': 1 << 30})
    value.update(changes)
    return value


def test_trial_identity_and_aggregate_credit_preserve_other_s15_controls():
    selected = preparation.parse_preparation(specification(native_gdn_capture=True, native_hc_writeback_capture=True))
    assert selected['native_gdn_capture'] and selected['native_hc_writeback_capture']
    owner = preparation.OwnerPreparation(selected, role_rows={'decode': 4, 'mtp': 4, 'prefill': 8}, manifest_sha256='a' * 64)
    identity = owner.numerical_selection['attention']
    assert identity['family'] == backend.OWNER_FAMILY and identity['local_output'] == 'normalized-half'
    assert identity['key_capacity_policy'] == 'fixed2051-mask-v1'
    needed = sum(v['persistent_bytes'] for v in owner.aten_requirements.values()) + max(
        v['operator_temporary_bytes'] for v in owner.aten_requirements.values())
    with pytest.raises(MemoryError, match='aggregate explicit tensor credit'):
        preparation.OwnerPreparation(specification(aten_tiles={'panel_rows': 2, 'tensor_budget_bytes': needed - 1}),
            role_rows={'decode': 4, 'mtp': 4, 'prefill': 8}, manifest_sha256='a' * 64)


@pytest.mark.parametrize('change', [{'trial': False}, {'aten_tiles': None},
    {'aten_tiles': {'panel_rows': True, 'tensor_budget_bytes': 1024}},
    {'aten_tiles': {'panel_rows': 2, 'tensor_budget_bytes': 0}},
    {'reference_arithmetic': {'policy': preparation.ARITHMETIC_POLICY, 'graph_budget_bytes': 1}},
    {'rn_heads_per_block': 4}, {'graph_family': GRAPH_FAMILY, 'graph_budget_bytes': 1}])
def test_unqualified_aten_contract_refuses_invalid_or_unpaid_modes(change):
    with pytest.raises(ValueError):
        preparation.parse_preparation(specification(**change))


def test_owner_payload_forecast_has_one_authoritative_panel_peak():
    rows, panel = 5, 2
    facts = backend.owner_requirements(rows, panel)
    expected = rows * 12 * 256 * 2 + rows * 12 * 4 + panel * 2051 * 256 * 4 + panel * 2051
    assert facts['persistent_bytes'] == expected
    assert facts['operator_temporary_bytes'] == backend.tensor_requirements(
        backend.TileSpec(panel, 1, 2051, 12, 1, .0625, masked=True))['known_peak_tensor_bytes']


def test_fixed_panel_descriptors_and_query_loan_retire_before_reuse(monkeypatch):
    descriptors, calls = [], []
    class Primitive:
        def __init__(self, spec, device, budget, identity):
            self.spec, self.device, self.device_budget_bytes, self.identity = spec, device, budget, identity
            descriptors.append(self)
        def run(self, q, k, v, *, valid_keys):
            calls.append(('operator', self.spec.batch, self.spec.key_rows))
            # Packed queries must survive reuse of original q_all as output.
            assert torch.equal(q, torch.full_like(q, 7))
            return backend.Partial(torch.ones_like(q), torch.zeros(self.spec.batch, 12, 1), valid_keys.any(-1))
    monkeypatch.setattr(backend, 'Prepared', Primitive)
    monkeypatch.setattr(backend, 'prepare', lambda spec, device, device_budget_bytes: Primitive(spec, device, device_budget_bytes, {}))
    def pack(q, table, out, **kwargs):
        out.fill_(7)
    def gather(selected, table, key, value, valid, **kwargs):
        key.zero_(); value.fill_(1); valid.zero_(); valid[:, :, 0] = True
    def merge(stats, table, global_stats, **kwargs):
        global_stats[:, :, 0].zero_(); global_stats[:, :, 1].fill_(4)
    def weight(normalized, lse, global_stats, table, out, **kwargs):
        out.copy_(normalized.float() * .25)
    def final(gathered, table, out, **kwargs):
        out.copy_(gathered[:4, :, :3].sum(0).half())
    for name, function in [('pack_owner_queries', pack), ('gather_owner_q8', gather),
                           ('merge_statistics', merge), ('weight_owner_normalized', weight), ('merge_values', final)]:
        monkeypatch.setattr(native, name, function)
    rows = 5
    q = torch.zeros(8, rows, 3, 256).half()
    views = dict(q_all=q, row_table=torch.zeros(rows, 16, dtype=torch.int64),
        selected_all=torch.zeros(8, rows, 2054, dtype=torch.int32), local_stats=torch.empty(rows, 12, 2),
        all_stats=torch.empty(8, rows, 12, 2), global_stats=torch.empty(rows, 12, 2),
        local_values=torch.empty(rows, 12, 256), all_values=torch.empty(8, rows, 12, 256),
        out=torch.empty(rows, 3, 256).half())
    def communicate(send, destination):
        calls.append(('collective', send.shape[-1]))
        destination.copy_(send[None].expand_as(destination))
    owner = NS(rank=0, native_page_tokens=128, native_layer=torch.tensor([3], dtype=torch.int32),
               queries=q, gather_native=communicate)
    tile = backend.OwnerTiles(rows, 2, 'cpu')
    before = len(descriptors)
    result = tile.run(owner, rows, views, 2051)
    assert torch.equal(result, torch.ones_like(result))
    assert len(descriptors) == before and [c[1] for c in calls if c[0] == 'operator'] == [2, 2, 1]
    assert all(c[2] == 2051 for c in calls if c[0] == 'operator')
    assert [c[1] for c in calls if c[0] == 'collective'] == [2, 256]
    with pytest.raises(ValueError, match='row/key envelope'):
        tile.run(owner, rows, views, 8)
    tile.close()
    assert tile.panels is tile.keys is tile.values is None
