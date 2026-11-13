"""Paid M32 row-boundary/layout contracts; CPU checks do not qualify CUDA math."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import padded_linears as p, owner_preparation as prep
from tests.test_sm70_padded_linears import metadata, local_config


def buffers(rows=120):
    w = NS(cfg=local_config(8), device=torch.device('cpu'), head=NS(n=248320 // 8), meta={'world': 8})
    b = p.PaddedLinears(w, rows, owned_panels=True)
    fields = {name: b.tensor(name, shape, torch.float32 if size == 4 else torch.float16)
              for name, (shape, size) in b.layout.items()}
    return b, fields


@pytest.mark.parametrize('rows', [1, 31, 32, 33, 64, 65, 96, 120])
def test_each_paid_logical_extent_matches_forecast_and_reconciles_before_state(rows):
    b, _ = buffers(rows)
    plan = p.PreparedM32(metadata(), {'decode': rows}, 'cpu')
    plan.adopt('decode', b)
    plan.verify_buffers('decode', b)
    assert b.physical_rows == ((rows + 31) // 32) * 32
    assert plan.reconciled_bytes['decode'] == p.workspace_bytes_from_text(metadata(), rows, 8,
        enabled=True, owned_panels=True)
    assert p.workspace_bytes_from_text(metadata(), rows, 8, enabled=True) == (0 if rows > 32 else
        p.workspace_bytes_from_text(metadata(), rows, 8, enabled=True, owned_panels=True))
    assert p._ENABLED is False
    plan.close()
    assert plan.buffers == {} and plan.closed


def test_actual_48_layer_forecast_is_not_lost_at_120_row_capacity():
    cfg = metadata()
    cfg['layer_types'] = ['linear_attention'] * 36 + ['full_attention'] * 12
    assert p.workspace_bytes_from_text(cfg, 120, 8, enabled=True, owned_panels=True) == 3974912
    plan = p.PreparedM32(cfg, {'decode': 120, 'mtp': 120, 'prefill': 256}, 'cpu')
    assert sum(plan.forecast.values()) == 7949824
    assert set(plan.role_rows) == {'decode', 'mtp'}


@pytest.mark.parametrize('rows', [1, 31, 32, 33, 64, 65, 96, 120])
@pytest.mark.parametrize('owned_input', [False, True])
def test_active_prefix_and_all_four_panels_preserve_values_without_extending_storage(rows, owned_input):
    b, fields = buffers()
    source = fields['act'] if owned_input else torch.empty((120, 320), dtype=torch.float16)
    for row in range(120):
        source[row].fill_(row / 256)
    original = source.clone()
    target = fields['mixed']
    target.fill_(-1)
    class Matrix:
        k, n = 320, 2560
        def __init__(self):
            self.destinations = []
        def can_matmul_prepared(self, dtype):
            return dtype == torch.float16
        def matmul_prepared(self, x, out):
            assert x.shape == (32, self.k) and out.shape == (32, self.n)
            self.destinations.append(out.data_ptr())
            # An independent known-answer projection: each output selects one
            # original input column with exact binary scaling.
            out.copy_((x[:, torch.arange(self.n) % self.k].float() / 8).half())
    matrix = Matrix()
    result = b.matmul(source[:rows], matrix, target[:rows])
    assert result.data_ptr() == target.data_ptr()
    torch.testing.assert_close(result, (source[:rows, torch.arange(2560) % 320].float() / 8).half(), atol=0, rtol=0)
    torch.testing.assert_close(source, original, atol=0, rtol=0)
    assert matrix.destinations == [target.data_ptr() + start * 2560 * 2 for start in range(0, rows, 32)]
    assert b.owners['mixed'].shape == (128, 2560)


def test_paid_offsets_reject_unaligned_foreign_and_out_of_extent_views():
    owner = torch.zeros((128, 16), dtype=torch.float16)
    for start in (0, 32, 64, 96):
        panel = p.PaddedLinears._owned_panel_span(owner[start:start + 1], [owner])
        assert panel.shape == (32, 16) and panel.data_ptr() == owner[start:].data_ptr()
    for value in (owner[1:2], owner[112:120], torch.zeros((1, 16)), owner[:, :8]):
        assert p.PaddedLinears._owned_panel_span(value, [owner]) is None
    assert p.PaddedLinears._owned_panel_span(owner[96:128], [owner]).numel() == 32 * 16
    assert p.PaddedLinears._matrix_view(owner[32:33], [owner]) is None  # old contract preserved


def test_aliasing_and_unowned_outputs_decline_before_any_write():
    b, f = buffers()
    class Matrix:
        k = n = 2560
        can_matmul_prepared = staticmethod(lambda _: True)
        matmul_prepared = staticmethod(lambda *_: pytest.fail('invalid layout reached operator'))
    assert b.matmul(f['mixed'][:64], Matrix(), f['mixed'][32:96]) is None
    assert b.matmul(f['mixed'][:1], Matrix(), torch.empty((1, 2560), dtype=torch.float16)) is None


def test_explicit_prepared_policy_is_numerical_and_cannot_hide_competing_dense_owner():
    spec = dict(kernel_family=prep.RN_KERNEL_FAMILY, rn_heads_per_block=12,
                graph_family='uncaptured', graph_budget_bytes=0, trial=True,
                prepared_linears=p.OWNED_POLICY)
    owner = prep.OwnerPreparation(spec, role_rows={'decode': 120, 'mtp': 120, 'prefill': 256}, manifest_sha256='a' * 64)
    assert owner.numerical_selection['prepared_linears']['reference_bit_identity'] is False
    for change in ({'trial': False}, {'prepared_linears': True}, {'prepared_linears': 'optimized'},
        {'dense_projections': {'backend': 'cublas', 'geometry': 'fixed-m32', 'roles': ['decode'], 'panel_rows': 32}}):
        with pytest.raises(ValueError, match='explicit RN trial'):
            prep.parse_preparation({**spec, **change})
    with pytest.raises(ValueError, match='through120'):
        p.PreparedM32(metadata(), {'decode': 121}, 'cpu')


def test_real_concat_adapter_passes_original_full_parts_to_each_prepared_panel(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFConcat
    from tensorfold.cuda.sm70 import gguf_ops
    b, fields = buffers()
    matrix = GGUFConcat((GGUFDense(torch.zeros((1536, 320), dtype=torch.float16)),
                         GGUFDense(torch.zeros((1024, 320), dtype=torch.float16))))
    matrix.can_matmul_prepared = lambda dtype: dtype == torch.float16  # CPU boundary control only
    calls = []
    def concat(source, weights, target):
        assert source.shape == (32, 320) and target.shape == (32, 2560)
        weights = tuple(weights)
        assert len(weights) == 2 and weights[0] is matrix.parts[0].weight and weights[1] is matrix.parts[1].weight
        calls.append(target.data_ptr())
        target[:, :1536].fill_(1)
        target[:, 1536:].fill_(2)
        return target
    monkeypatch.setattr(gguf_ops, 'concat32', concat)
    result = b.matmul(fields['act'], matrix, fields['mixed'])
    assert len(calls) == 4 and result.shape == (120, 2560)
    assert torch.all(result[:, :1536] == 1) and torch.all(result[:, 1536:] == 2)


def test_rank_receipt_reports_paid_roles_without_invocation_or_performance_claim():
    from tools.v100_owned_qualification import _prepared_m32_receipt
    b, _ = buffers()
    plan = p.PreparedM32(metadata(), {'decode': 120}, 'cpu')
    plan.adopt('decode', b)
    plan.verify_buffers('decode', b)
    plan.bound_weights = True  # CPU receipt boundary double, not native warmup
    engine = NS(w=NS(_private_prepared_m32=plan, meta={'private_prepared_m32_policy': p.owned_numerical_identity()}),
        _owner_preparation=NS(specification={'prepared_linears': p.OWNED_POLICY}, prepared_m32=plan),
        multi=NS(buf=NS(padded_linears=b), mbuf=None))
    receipt = _prepared_m32_receipt(engine)
    assert receipt['roles']['decode']['physical_rows'] == 128
    assert receipt['invocation_evidence'] is False and receipt['P3_valid'] is False
    engine.multi.buf.padded_linears = None
    with pytest.raises(ValueError, match='paid role'):
        _prepared_m32_receipt(engine)
