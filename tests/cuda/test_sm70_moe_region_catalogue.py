"""144 pure-MoE graphs: one-GPU arithmetic/capture oracle, admission PROXY only.

No State/KV allocation, NCCL, Engine admission or real native24 capacity is
qualified here. The actual RoundGraphs accounting methods observe this isolated
fixture with future_bytes=0;24 free placeholders establish only its cold gate.
"""
from types import SimpleNamespace as NS
import json

import pytest
import torch

from gguf_fixture import blocks
from tensorfold.cuda import moe
from tensorfold.cuda.sm70 import family_ops, routing_native
from tensorfold.cuda.sm70.router_blas import PROFILE, RolePlans
from tensorfold.families.qwen4_exp.cuda.capture_lifetime import finish_forward_frontiers
from tensorfold.families.qwen4_exp.cuda.forward import moe_block
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFExperts, GGUFPacked
from tensorfold.families.qwen4_exp.cuda.moe_region_catalogue import Catalogue, ROWS
from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="actual SM70 pure-MoE catalogue component")


def _packet(buffers):
    return (buffers.moe.logits, buffers.moe.pick, buffers.moe.wts, buffers.moe.y)


def _poison_outputs(buffers):
    # Distinct finite sentinels also expose any writes beyond exact R.
    for tensor, value in zip(_packet(buffers), (-121.25, -2147483648, -913.25, -71.5), strict=True):
        tensor.fill_(value)


def _exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


@torch.inference_mode()
def test_actual144_exact_row_graphs_match_batched_eager_changed_inputs_and_preserve_tails(monkeypatch):
    rows, dims, width, experts = 32, 2560, 80, 512
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is True
    assert torch.backends.cuda.matmul.allow_fp16_accumulation is False
    assert routing_native._NATIVE_LOGITS is False
    assert torch.cuda.memory.get_allocator_backend() == "native"
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    monkeypatch.setattr(GGUFExperts, "implementation", "packed_exact")
    monkeypatch.setattr(GGUFDense, "implementation", "native32")
    packed = []
    for seed, count, columns, crop in ((811, experts * width, dims, 0),
                                        (812, experts * width, dims, 0),
                                        (813, experts * dims, 256, 16)):
        raw = blocks(12, count, columns, seed=seed)
        packed.append(GGUFPacked(torch.tensor(raw, device="cuda"), 12, raw.shape[0],
                                  width if crop else columns, crop))
    torch.manual_seed(814)
    shared = [GGUFDense(torch.randn(shape, device="cuda", dtype=torch.float16) * .01)
              for shape in ((width, dims), (width, dims), (dims, width))]
    bank = GGUFExperts(*packed, shared, count=experts, width=width, dims=dims)
    cfg = NS(experts=experts, top_k=10, num_experts=experts, num_experts_per_tok=10,
             hidden_size=dims, moe_intermediate_size=width, execution_dtype="float16")
    layers = [NS(index=index, moe=NS(experts=bank,
                 router=torch.randn((experts + 1, dims), device="cuda", dtype=torch.float32) * .01))
              for index in range(48)]
    assert len({layer.moe.router.data_ptr() for layer in layers}) == 48
    device = torch.device("cuda", torch.cuda.current_device())
    weights = NS(cfg=cfg, layers=layers, device=device, comm=None, meta={"world": 1,
                 "sm70_profile": PROFILE, "execution_provider": "sm70", "sm70_moe_region_graphs_trial": True})
    buffers = NS(prefill=False, rows=rows,
                 mixed=torch.zeros((rows, dims), device=device, dtype=torch.float16),
                 moe=moe.MoEBuffers(rows, cfg, device, prefill=True), input_preparation_context=None)
    buffers.router_blas = RolePlans(tuple(layer.moe.router for layer in layers), rows)
    assert buffers.moe.pick.shape == (32, 11) and buffers.moe.y.shape == (32, 11, 2560)
    assert buffers.moe.y.dtype == torch.float16 and buffers.prefill is False
    # Canonical eager view borrows the same original packet/input objects. It
    # deliberately has no catalogue attribute, so actual moe_block runs eager.
    eager = NS(prefill=False, mixed=buffers.mixed, moe=buffers.moe, router_blas=buffers.router_blas)
    assert not hasattr(eager, "moe_region_graphs")
    inputs = torch.randn((48, rows, dims), device=device, dtype=torch.float16) * .03
    dec = NS(w=weights, buf=buffers, streams={}, filling={}, kept={},
             free=[object() for _ in range(24)], round_tables=NS(slots=24), _memory_error=None)
    # Use the actual parent methods without constructing BoundStateGraphs or
    # pretending placeholders are admitted States. No N1 graph is captured.
    parent = object.__new__(RoundGraphs)
    parent.dec, parent.budget, parent.future_bytes = dec, 2 << 30, 0
    parent.allocator_backend, parent.spare = "native", None
    parent.pool, parent.side = torch.cuda.graph_pool_handle(), torch.cuda.Stream(device=device)
    parent.bytes = parent.room_bytes = parent.settled_bytes = 0
    parent.private_spare_bytes, parent.private_pool_records = None, []
    parent.allocator_record = {}
    parent.failure = parent.failure_frames = parent.failure_traceback = None
    parent.failure_cause = parent.failure_context = None
    parent.admitted = False
    parent.bound = parent.candidate_regions = None
    parent.verify, parent.mtp, parent.pending = {}, {}, []
    catalogue = Catalogue(parent)
    parent.moe_regions = catalogue  # All owned pools enter real accounting before prepare.
    journal = ([None] * 3, [None], [None] * 2, [None] * 3)
    finish = catalogue.finish
    def close(_primary):
        catalogue.close()
    primary = None
    diagnostic = None
    try:
        catalogue.prepare()
        keys = {(count, index) for count in (30, 31, 32) for index in range(48)}
        assert set(catalogue.regions) == keys and set(catalogue.pools) == set(ROWS)
        assert len({tuple(pool) for pool in catalogue.pools.values()}) == 3
        assert catalogue.capture_router is not buffers.router_blas
        assert len(catalogue.capture_router.owners) == len(catalogue.capture_router.plans) == 48
        assert catalogue.capture_router.calls == 2 * 144
        assert parent.bytes <= parent.budget and parent.future_bytes == 0
        assert len(parent.verify) == len(parent.mtp) == 0
        parent.admitted = True  # PROXY aggregate admission only; no real24 State/KV proof.
        buffers.moe_region_graphs = catalogue
        for scale in (1.0, -.5, .125):
            for count in ROWS:
                assert catalogue.begin(weights, buffers, count)
                for layer in layers:
                    buffers.mixed.fill_(float("nan"))
                    buffers.mixed[:count].copy_(inputs[layer.index, :count])
                    buffers.mixed[:count].mul_(scale)
                    buffers.mixed[0].zero_()  # Exact logits ties: lower expert IDs first.
                    _poison_outputs(buffers)
                    moe_block(layer, weights, eager, count, _local_only=True)
                    expected = [tensor[:count].clone() for tensor in _packet(buffers)]
                    _poison_outputs(buffers)
                    tails = [tensor[count:].clone() for tensor in _packet(buffers)]
                    moe_block(layer, weights, buffers, count, _local_only=True)
                    torch.cuda.synchronize(device)
                    for tensor, active, tail in zip(_packet(buffers), expected, tails, strict=True):
                        _exact(tensor[:count], active)
                        _exact(tensor[count:], tail)
                    assert torch.all(torch.isfinite(buffers.moe.logits[:count]))
                    assert torch.all(torch.isfinite(buffers.moe.wts[:count]))
                    assert torch.all(torch.isfinite(buffers.moe.y[:count]))
                    assert torch.isnan(buffers.mixed[count:]).all()
                    assert torch.equal(buffers.moe.pick[0, :10], torch.arange(10, device=device, dtype=torch.int32))
                    assert torch.all(buffers.moe.pick[:count, -1] == 512)
                catalogue.finish()
        assert catalogue.frames == 9 and catalogue.hits == 3 * 144
        assert catalogue.row_frames == {30: 3, 31: 3, 32: 3}
        assert all(region.hits == 4 for region in catalogue.regions.values())
        assert buffers.router_blas.calls == 3 * 144
        assert catalogue.capture_router.calls == 2 * 144  # No runtime main-role fallback in graph replay.
        diagnostic = {"scope": "singleGPU144graph oracle; State/NCCL/Engine/native24 admission UNQUALIFIED",
                      "graphs": 144, "row_keys": list(ROWS), "runtime_frames": catalogue.frames,
                      "runtime_replays": catalogue.hits, "capture_role_calls": catalogue.capture_router.calls,
                      "main_eager_role_calls": buffers.router_blas.calls,
                      "capture_charge_bytes_proxy": parent.bytes, "future_service_bytes_proxy": parent.future_bytes}
    except BaseException as error:
        primary = error
    # Both frontiers are attempted using fixed outcomes, retaining the actual
    # primary even if finish/fence/reset/notification also fails.
    finish_forward_frontiers(primary, finish, close, None, journal)
    if primary is not None:
        raise primary
    assert catalogue.closed and catalogue.capture_router is None and not catalogue.regions and not catalogue.pools
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is True
    assert torch.backends.cuda.matmul.allow_fp16_accumulation is False
    print(json.dumps(diagnostic, sort_keys=True), flush=True)
