"""One fresh HC/MoE pool across independent exact-R graph cycles.

This is a single-GPU, test-only pool configuration: 192 keys, 48 ordered
synthetic layers and IQ18/20 banks. Every key receives prepared external inputs
and its outputs are consumed before another graph. No graph-private scratch
escapes to an owner State. No State, attention, NCCL, Engine, native24 capacity
or shipping pool accounting is qualified by this oracle.
"""
import json
import time
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda import moe
from tensorfold.cuda.sm70 import family_ops, routing_native
from tensorfold.cuda.sm70.execution_policy import SHARED_FUSION_TRIAL, _family_runtime
from tensorfold.cuda.sm70.router_blas import RolePlans
from tensorfold.families.qwen4_exp.cuda.capture_lifetime import finish_forward_frontiers, on_stream
from tensorfold.families.qwen4_exp.cuda.forward import hc_block, moe_block
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFConcat, GGUFDense, GGUFExperts
from tensorfold.families.qwen4_exp.cuda.hc_moe_region import HCReadoutMoERegion
from tensorfold.families.qwen4_exp.cuda.weight_types import HC
from test_sm70_hc_moe_region_tp8 import _equal, _snapshot, _tensors
from test_sm70_shared_fusion_current import bank as iq_bank


ROWS = (5, 9, 19, 32)
LAYERS = 48
# Every R is revisited after a different-sized complete cycle. Runtime starts
# with the last capture group; it never assumes all192 captures replay in order.
CYCLES = ((32, 1.0), (5, -.5), (19, 0.0), (9, .125),
          (5, 1.0), (32, 0.0), (9, -.5), (19, .125))


def _pool_observation(pool):
    # Pinned04cf supports a pool filter and disabling trace materialization.
    # Retain only a fixed-size summary, never the segment/block snapshot.
    segments = torch.cuda.memory_snapshot(mempool_id=pool, include_traces=False)
    return {'segments': len(segments), 'total_size': sum(row['total_size'] for row in segments),
            'allocated_size': sum(row['allocated_size'] for row in segments),
            'scope': 'observed private-pool reservation/live payload; not accounting or admission proof'}


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                    reason='actual pinned SM70 required; skip is not acceptance')
@torch.inference_mode()
def test_one_hc_moe_pool_preserves_iq18_20_exact_rows_across_ordered_cycles(monkeypatch):
    # Owner cells and cleanup outcomes precede allocations/accepted captures.
    owner = NS(device=None, regions=[None] * (len(ROWS) * LAYERS),
               initialized=[False] * (len(ROWS) * LAYERS), capture_router=None,
               pool=None, side=None, closed=False)
    journal = ([None] * 3, [None], [None] * 2, [None] * 3)
    primary, diagnostic = None, None

    def retire(_primary):
        if owner.device is None:
            return  # No test-owned GPU storage or capture could be acquired.
        for region, initialized in zip(owner.regions, owner.initialized, strict=True):
            if initialized:
                region.retiring = True
        # Failure here retains all graphs, the shared role and the pool.
        torch.cuda.synchronize(owner.device)
        for region, initialized in zip(owner.regions, owner.initialized, strict=True):
            if initialized:
                region._release_fenced()
        # No unconditional plan-release frontier: every graph reset must have
        # returned positively before these shared pointer-table owners retire.
        for index in range(len(owner.regions)):
            owner.regions[index] = None
        owner.capture_router = owner.side = owner.pool = None
        owner.closed = True

    try:
        owner.device = torch.device('cuda', torch.cuda.current_device())
        _family_runtime(torch)
        assert torch.cuda.memory.get_allocator_backend() == 'native'
        assert routing_native._NATIVE_LOGITS is False
        monkeypatch.setattr(family_ops, '_NATIVE_FAMILY', True)
        monkeypatch.setattr(GGUFDense, 'implementation', 'native32')
        monkeypatch.setattr(GGUFConcat, 'implementation', 'native32')
        monkeypatch.setattr(GGUFExperts, 'implementation', 'packed_blocks2w')
        torch.manual_seed(14201)
        value = iq_bank(18, 20)
        assert value.gate.ggml_type == value.up.ggml_type == 18 and value.down.ggml_type == 20
        assert value.implementation == 'packed_blocks2w' and value.count == 513
        capacity, dims = 34, 2560
        cfg = NS(hidden=dims, streams=4, low=320, eps=1e-6, execution_dtype='float16',
                 experts=512, top_k=10, num_experts=512, num_experts_per_tok=10,
                 hidden_size=dims, moe_intermediate_size=80)
        layers = []
        for index in range(LAYERS):
            hc = HC(GGUFConcat((GGUFDense(torch.randn((320, 10240), device=owner.device,
                                                       dtype=torch.float16) * .005),
                               GGUFDense(torch.randn((4, 10240), device=owner.device,
                                                       dtype=torch.float16) * .005))),
                    GGUFDense(torch.randn((10240, 320), device=owner.device, dtype=torch.float16) * .005),
                    torch.full((10240,), .05, device=owner.device, dtype=torch.float32), True)
            layers.append(NS(index=index, mlp_hc=hc, moe=NS(experts=value,
                router=torch.randn((513, dims), device=owner.device, dtype=torch.float32) * .01)))
        assert len({layer.moe.router.data_ptr() for layer in layers}) == LAYERS
        assert len({layer.mlp_hc.up.weight.data_ptr() for layer in layers}) == LAYERS
        assert len({layer.mlp_hc.down.parts[0].weight.data_ptr() for layer in layers}) == LAYERS
        # TP8-shaped prepared branch storage only. There is no communicator and
        # no collective in this pool oracle; the real TP8 oracle is separate.
        weights = NS(cfg=cfg, layers=layers, device=owner.device, x3=None, mtp=None,
                     comm=NS(world=8, rank=0, device=owner.device),
                     meta={'world': 8, 'rank': 0, 'sm70_profile': SHARED_FUSION_TRIAL,
                           'execution_provider': 'sm70', 'sm70_family_trial': {
                               'experts': 'packed_blocks2w',
                               'numerical_policy': 'router-cublas-generic-packed-shared-fusion-v1'}})
        b = NS(prefill=False, rows=capacity, hc_plans=None, padded_linears=None, split=None)
        shapes = {'h': (capacity, 10240), 'pss': (capacity, 10, 4), 'normed': (capacity, 10240),
                  'xs_normed': (capacity, 320), 'dn': (capacity, 324), 'dn_mix': (capacity, 320),
                  'act': (capacity, 320), 'xs_act': (capacity, 10), 'inj_a': (capacity, 4),
                  'inj_m': (capacity, 4), 'up': (capacity, 10240), 'mixed': (capacity, dims),
                  'xs_mixed': (capacity, 80), 'g_branch': (8 * capacity * dims,),
                  'part_moe': (capacity, dims), 'g_moe': (8 * capacity * dims,)}
        f32 = ('pss', 'xs_normed', 'xs_act', 'xs_mixed', 'g_branch', 'part_moe', 'g_moe')
        for name, shape in shapes.items():
            setattr(b, name, torch.empty(shape, device=owner.device,
                                        dtype=torch.float32 if name in f32 else torch.float16))
        b.moe = moe.MoEBuffers(capacity, cfg, owner.device, prefill=True)
        assert b.moe.y.dtype == torch.float16 and b.prefill is False
        routers = tuple(layer.moe.router for layer in layers)
        b.router_blas = RolePlans(routers, 32)
        current = torch.cuda.current_stream(owner.device)
        owner.side = torch.cuda.Stream(device=owner.device)
        owner.side.wait_stream(current)

        def capture_role():
            owner.capture_router = object.__new__(RolePlans)
            owner.capture_router.__init__(routers, 32)
        on_stream(torch.cuda, owner.side, capture_role)
        owner.side.synchronize()
        assert owner.capture_router is not b.router_blas
        owner.pool = torch.cuda.graph_pool_handle()
        # Fresh independent marker handles, not actual N1/candidate owners.
        other_pool_markers = (torch.cuda.graph_pool_handle(), torch.cuda.graph_pool_handle())
        pool_id = tuple(owner.pool)
        assert pool_id not in {tuple(pool) for pool in other_pool_markers}
        h_inputs = torch.randn((LAYERS, capacity, 10240), device=owner.device, dtype=torch.float16) * .03
        inj_inputs = (2 * torch.sigmoid(torch.randn((LAYERS, capacity, 4), device=owner.device))).half()
        branches = torch.randn((LAYERS, 8, capacity, dims), device=owner.device, dtype=torch.float32) * .01

        def seed(index, rows, scale):
            for tensor in _tensors(b).values():
                tensor.fill_(-2147483648 if tensor.dtype == torch.int32 else float('nan'))
            b.h[:rows].copy_(h_inputs[index, :rows] * scale)
            b.inj_a[:rows].copy_(inj_inputs[index, :rows])
            branch = b.g_branch[:8 * rows * dims].view(8, rows, dims)
            branch.copy_(branches[index, :, :rows] * scale)
            return branch

        torch.cuda.synchronize(owner.device)
        memory_before_capture = torch.cuda.mem_get_info(owner.device)
        capture_begin_ns = time.monotonic_ns()
        for row_index, rows in enumerate(ROWS):
            for layer in layers:
                index = row_index * LAYERS + layer.index
                seed(layer.index, rows, 1.0)
                # Constructor is metadata-only; publish its partial owner
                # before initialization/prepare can allocate or submit work.
                region = object.__new__(HCReadoutMoERegion)
                owner.regions[index] = region
                region.__init__(weights, layer, b, rows, side=owner.side,
                                pool=owner.pool, capture_router=owner.capture_router)
                owner.initialized[index] = True
                region.prepare()
                assert tuple(region.graph.pool()) == pool_id
                assert {tuple(pool) for pool in region.graph.pools()} == {pool_id}
                assert region.local_buffers.router_blas is owner.capture_router
                assert region.local_buffers.mixed is b.mixed and region.local_buffers.moe is b.moe
        torch.cuda.synchronize(owner.device)
        capture_end_ns = time.monotonic_ns()
        pool_after_capture = _pool_observation(owner.pool)
        memory_after_capture = torch.cuda.mem_get_info(owner.device)
        # In this observed pure component, no private temporary Tensor remains
        # application-live; reserved bytes remain owned by the captured graphs.
        assert pool_after_capture['allocated_size'] == 0
        assert owner.capture_router.calls == 2 * len(owner.regions)
        assert len(owner.regions) == 192 and all(owner.initialized)
        replay_count = 0
        runtime_begin_ns = time.monotonic_ns()
        for rows, scale in CYCLES:
            # Each complete cycle retains real layer order. R groups are
            # independent and can be selected in a different capture order.
            for layer in layers:
                branch = seed(layer.index, rows, scale)
                initial = _snapshot(b)
                hc_block(layer.mlp_hc, b, rows, cfg.eps, cfg.streams, cfg.low,
                         3, b.inj_a[:rows], b.inj_m, b.h, branch=branch)
                moe_block(layer, weights, b, rows, _local_only=True)
                expected = _snapshot(b)
                branch = seed(layer.index, rows, scale)
                region = owner.regions[ROWS.index(rows) * LAYERS + layer.index]
                region.replay(3, branch, b.inj_a[:rows])
                torch.cuda.synchronize(owner.device)  # Never overlap replays from this shared pool.
                for name, tensor in _tensors(b).items():
                    _equal(tensor, expected[name])
                    at = 8 * rows * dims if name in ('g_branch', 'g_moe') else rows
                    _equal(tensor[at:], initial[name][at:])
                for name in ('inj_a', 'g_branch', 'dn', 'dn_mix', 'part_moe', 'g_moe'):
                    _equal(expected[name], initial[name])
                assert torch.isfinite(b.mixed[:rows]).all()
                assert torch.isfinite(b.moe.logits[:rows]).all() and torch.isfinite(b.moe.y[:rows]).all()
                assert torch.all(b.moe.pick[:rows, -1] == 512)
                if scale == 0.0:
                    assert torch.equal(b.moe.pick[0, :10],
                                       torch.arange(10, device=owner.device, dtype=torch.int32))
                replay_count += 1
                # No per-key outputs survive to the following graph. These
                # ordinary eager clones are never put in the pool/owner state.
                del initial, expected
        torch.cuda.synchronize(owner.device)
        runtime_end_ns = time.monotonic_ns()
        pool_after_runtime = _pool_observation(owner.pool)
        memory_after_runtime = torch.cuda.mem_get_info(owner.device)
        assert pool_after_runtime['allocated_size'] == 0
        assert replay_count == len(CYCLES) * LAYERS == 384
        assert all(region.hits == 2 for region in owner.regions)
        assert b.router_blas.calls == replay_count and owner.capture_router.calls == 384
        _family_runtime(torch)
        assert value.implementation == 'packed_blocks2w'
        diagnostic = {'graphs': 192, 'actual_pool': pool_id, 'pool_count': 1,
                      'capture_preparation_host_elapsed_ns': capture_end_ns - capture_begin_ns,
                      'runtime_oracle_host_elapsed_ns': runtime_end_ns - runtime_begin_ns,
                      'timing_scope': 'monotonic host wall; capture includes constructor/warm/capture/driver checks; runtime includes eager references, clones, synchronization, validation and any lazy graph instantiation; NOT shipping throughput',
                      'private_pool_after_capture': pool_after_capture,
                      'private_pool_after_runtime': pool_after_runtime,
                      'raw_mem_get_info': {'before_capture': memory_before_capture,
                                           'after_capture': memory_after_capture,
                                           'after_runtime': memory_after_runtime,
                                           'units': 'free,total bytes; raw observations, no accounting/admission claim'},
                      'captured_R': list(ROWS), 'runtime_R': [row for row, _scale in CYCLES],
                      'complete_ordered_cycles': len(CYCLES), 'replays': replay_count,
                      'distinct_router_weights': 48, 'distinct_HC_weights': 48,
                      'bank_encodings': [18, 18, 20], 'selected_profile': SHARED_FUSION_TRIAL,
                      'scope': 'singleGPUtest-onlysharedpool; syntheticTP8shapedbranch; noState/NCCL/Engine/native24/accounting/P3qualification'}
    except BaseException as error:
        primary = error
    # One positive global fence, all graph resets, then shared-role/pool release.
    # A failed fence/reset stops dependent releases and retains the owner in the
    # propagated traceback, while the helper preserves the original primary.
    finish_forward_frontiers(primary, retire, None, None, journal)
    if primary is not None:
        raise primary
    assert owner.closed and owner.capture_router is None and owner.pool is None
    assert all(region is None for region in owner.regions)
    print(json.dumps(diagnostic, sort_keys=True), flush=True)
