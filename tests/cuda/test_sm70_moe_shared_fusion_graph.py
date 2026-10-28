"""Selected IQ/shared-fused eager versus capture, with both real output storages."""
from types import SimpleNamespace

import pytest
import torch

from test_sm70_shared_fusion_current import bank
from tensorfold.cuda import moe
from tensorfold.cuda.sm70 import family_ops, routing_native
from tensorfold.cuda.sm70.execution_policy import SHARED_FUSION_TRIAL
from tensorfold.cuda.sm70.router_blas import RolePlans
from tensorfold.families.qwen4_exp.cuda.forward import moe_block
from tensorfold.families.qwen4_exp.cuda.moe_region_graphs import MoERegion

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='actual SM70 selected fusion capture')


@pytest.fixture(scope='module')
def current_bank():
    torch.manual_seed(2741)
    value = bank(18, 20)  # Actual current IQ gate/down domain; not synthetic Q4_K two-warp claims.
    value.implementation = 'packed_blocks2w'
    return value


@pytest.mark.parametrize('role', ('target', 'mtp'))
@pytest.mark.parametrize('rows', (30, 31, 32))
@torch.inference_mode()
def test_selected_fusion_capture_is_bitwise_fused_eager_and_preserves_tails(monkeypatch, current_bank, role, rows):
    monkeypatch.setattr(family_ops, '_NATIVE_FAMILY', True)
    assert routing_native._NATIVE_LOGITS is False
    cfg = SimpleNamespace(experts=512, top_k=10, num_experts=512, num_experts_per_tok=10,
                          hidden_size=2560, moe_intermediate_size=80, execution_dtype='float16')
    weights = SimpleNamespace(cfg=cfg, comm=None, meta={'sm70_profile': SHARED_FUSION_TRIAL})
    layer = SimpleNamespace(index=0, moe=SimpleNamespace(experts=current_bank,
        router=torch.randn((513, 2560), device='cuda', dtype=torch.float32) * .001))
    buffers = SimpleNamespace(prefill=False, mixed=torch.empty((32, 2560), device='cuda', dtype=torch.float16),
        moe=moe.MoEBuffers(32, cfg, 'cuda', prefill=role == 'target'))
    buffers.router_blas = RolePlans((layer.moe.router,), 32)
    assert buffers.moe.y.dtype == (torch.float16 if role == 'target' else torch.float32)
    original = torch.randn((rows, 2560), device='cuda', dtype=torch.float16) * .001
    buffers.mixed.fill_(float('nan'))
    buffers.mixed[:rows].copy_(original)
    region = MoERegion(weights, layer, buffers, rows)
    primary = None
    close_failures = [None]
    try:
        region.prepare()
        assert region.capture_router is not buffers.router_blas
        for scale in (1., -.5, .125):
            buffers.mixed[:rows].copy_(original * scale)
            packet = (buffers.moe.logits, buffers.moe.pick, buffers.moe.wts, buffers.moe.y)
            for tensor in packet:
                tensor[rows:].fill_(-71)
            moe_block(layer, weights, buffers, rows, _local_only=True)
            expected = [tensor[:rows].clone() for tensor in packet]
            tails = [tensor[rows:].clone() for tensor in packet]
            region.replay()
            torch.cuda.synchronize()
            for value, reference, tail in zip(packet, expected, tails, strict=True):
                assert torch.equal(value[:rows].contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))
                assert torch.equal(value[rows:].contiguous().view(torch.uint8), tail.contiguous().view(torch.uint8))
                if value.dtype != torch.int32:
                    assert torch.isfinite(value[:rows]).all()
            assert torch.isnan(buffers.mixed[rows:]).all()
        assert region.hits == 3 and region.capture_router.calls == 2
    except BaseException as error:
        primary = error
    try:
        region.close()
    except BaseException as error:
        close_failures[0] = error
    from tensorfold.cleanup import raise_failures

    if close_failures[0] is not None:
        raise_failures(primary, close_failures)
    if primary is not None:
        raise primary
