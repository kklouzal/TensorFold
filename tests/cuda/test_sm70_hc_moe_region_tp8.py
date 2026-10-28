"""Strict TP8 HC/pure-MoE capture oracle; no Engine/catalogue/capacity claim.

Both collectives remain outside capture. Original FP16 HC calls and actual
513-router/11-slot GGUF operators are the eager oracle, including every public
HC/MoE output, protected input, and untouched tail. No State is constructed.
"""
import json
import socket
import time
from types import SimpleNamespace as NS

import pytest
import torch


HC_PACKET = ('h', 'pss', 'normed', 'xs_normed', 'dn', 'dn_mix', 'act', 'xs_act',
             'inj_a', 'inj_m', 'up', 'mixed', 'xs_mixed', 'g_branch', 'part_moe', 'g_moe')


def _tensors(b):
    return {**{name: getattr(b, name) for name in HC_PACKET},
            **{'moe.' + name: getattr(b.moe, name) for name in ('logits', 'pick', 'wts', 'y', 'act')}}


def _snapshot(b):
    return {name: tensor.clone() for name, tensor in _tensors(b).items()}


def _equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _worker(rank, port):
    from gguf_fixture import blocks
    from tensorfold.cleanup import raise_failures
    from tensorfold.cuda import moe
    from tensorfold.cuda.comm import NCCL
    from tensorfold.cuda.sm70 import family_ops, routing_native
    from tensorfold.cuda.sm70.router_blas import PROFILE, RolePlans
    from tensorfold.families.qwen4_exp.cuda import glue
    from tensorfold.families.qwen4_exp.cuda.forward import hc_block, moe_block
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFConcat, GGUFExperts, GGUFPacked
    from tensorfold.families.qwen4_exp.cuda.hc_moe_region import HCReadoutMoERegion
    from tensorfold.families.qwen4_exp.cuda.weight_types import HC

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    owner, region, primary = NCCL(), None, None
    secondary = [None]
    collective_calls = [0]
    def gather(send, recv):
        collective_calls[0] += 1
        owner.all_gather(send, recv)
    try:
        owner.open(rank, 8, '127.0.0.1', port, timeout_s=90, key='tf_hc_moe_mode3')
        assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is True
        assert torch.backends.cuda.matmul.allow_fp16_accumulation is False
        assert torch.cuda.memory.get_allocator_backend() == 'native'
        assert routing_native._NATIVE_LOGITS is False
        family_ops._NATIVE_FAMILY = True
        GGUFDense.implementation, GGUFConcat.implementation = 'native32', 'native32'
        GGUFExperts.implementation = 'packed_exact'
        capacity, dims, width, experts = 34, 2560, 80, 512
        packed = []
        for seed, count, columns, crop in ((811, experts * width, dims, 0),
                                            (812, experts * width, dims, 0),
                                            (813, experts * dims, 256, 16)):
            raw = blocks(12, count, columns, seed=seed + rank * 11)
            packed.append(GGUFPacked(torch.tensor(raw, device=device), 12, raw.shape[0],
                                      width if crop else columns, crop))
        torch.manual_seed(814)
        shared = [GGUFDense(torch.randn(shape, device=device, dtype=torch.float16) * .01)
                  for shape in ((width, dims), (width, dims), (dims, width))]
        bank = GGUFExperts(*packed, shared, count=experts, width=width, dims=dims)
        hc = HC(GGUFConcat((GGUFDense(torch.randn((320, 10240), device=device, dtype=torch.float16) * .005),
                           GGUFDense(torch.randn((4, 10240), device=device, dtype=torch.float16) * .005))),
                GGUFDense(torch.randn((10240, 320), device=device, dtype=torch.float16) * .005),
                torch.full((10240,), .05, device=device, dtype=torch.float32), True)
        cfg = NS(hidden=dims, streams=4, low=320, eps=1e-6, execution_dtype='float16',
                 experts=experts, top_k=10, num_experts=experts, num_experts_per_tok=10,
                 hidden_size=dims, moe_intermediate_size=width)
        layer = NS(index=0, mlp_hc=hc, moe=NS(experts=bank,
                   router=torch.randn((513, dims), device=device, dtype=torch.float32) * .01))
        comm = NS(world=8, rank=rank, device=device, all_gather=gather)
        weights = NS(cfg=cfg, layers=[layer], device=device, comm=comm, x3=None,
                     meta={'world': 8, 'rank': rank, 'sm70_profile': PROFILE, 'execution_provider': 'sm70'})
        h = torch.randn((capacity, 10240), device=device, dtype=torch.float16) * .03
        injection = (2 * torch.sigmoid(torch.randn((capacity, 4), device=device))).half()
        # Rank-dependent prior partials exercise the actual rank-ordered mode3 sum.
        torch.manual_seed(911 + rank)
        partial = torch.randn((capacity, dims), device=device, dtype=torch.float32) * .01
        replay_count = 0
        for role in ('target', 'mtp'):
            # MTP really is outside weights.layers; its captured weights must
            # still be bound rather than accidentally guarded by target layers.
            weights.layers = [layer] if role == 'target' else []
            weights.mtp = None if role == 'target' else NS(layer=layer)
            b = NS(prefill=False, rows=capacity, hc_plans=None, padded_linears=None, split=None)
            shapes = {'h': (capacity, 10240), 'pss': (capacity, 10, 4), 'normed': (capacity, 10240),
                      'xs_normed': (capacity, 320), 'dn': (capacity, 324), 'dn_mix': (capacity, 320),
                      'act': (capacity, 320), 'xs_act': (capacity, 10), 'inj_a': (capacity, 4),
                      'inj_m': (capacity, 4), 'up': (capacity, 10240), 'mixed': (capacity, dims),
                      'xs_mixed': (capacity, 80), 'g_branch': (8 * capacity * dims,),
                      'part_moe': (capacity, dims), 'g_moe': (8 * capacity * dims,)}
            for name, shape in shapes.items():
                dtype = torch.float32 if name in ('pss', 'xs_normed', 'xs_act', 'xs_mixed',
                                                  'g_branch', 'part_moe', 'g_moe') else torch.float16
                setattr(b, name, torch.empty(shape, device=device, dtype=dtype))
            b.moe = moe.MoEBuffers(capacity, cfg, device, prefill=role == 'target')
            b.router_blas = RolePlans((layer.moe.router,), 32)
            assert b.moe.y.dtype == (torch.float16 if role == 'target' else torch.float32)
            for rows in (30, 31, 32):
                def seed(scale):
                    for index, tensor in enumerate(_tensors(b).values()):
                        tensor.fill_(-101 - index)
                    b.h[:rows].copy_(h[:rows] * scale)
                    b.inj_a[:rows].copy_(injection[:rows])
                    send = partial[:rows] * scale
                    gather(send, b.g_branch[:8 * rows * dims])
                    return b.g_branch[:8 * rows * dims].view(8, rows, dims)
                # Prepare consumes only disposable initialized startup scratch.
                seed(1.0)
                region = HCReadoutMoERegion(weights, layer, b, rows)
                before = collective_calls[0]
                region.prepare()
                assert collective_calls[0] == before, 'capture crossed a collective boundary'
                assert region.capture_router is not b.router_blas
                for scale in (1.0, -.5, 0.0):
                    branch = seed(scale)
                    initial = _snapshot(b)
                    hc_block(layer.mlp_hc, b, rows, cfg.eps, cfg.streams, cfg.low,
                             3, b.inj_a[:rows], b.inj_m, b.h, branch=branch)
                    moe_block(layer, weights, b, rows, _local_only=True)
                    expected = _snapshot(b)
                    glue.moe_partial(b.moe.y[:rows], b.moe.wts[:rows], b.part_moe, rows)
                    gather(b.part_moe[:rows], b.g_moe[:8 * rows * dims])
                    expected_after = (b.part_moe.clone(), b.g_moe.clone())
                    branch = seed(scale)
                    before = collective_calls[0]
                    region.replay(3, branch, b.inj_a[:rows])
                    assert collective_calls[0] == before
                    observed = _snapshot(b)  # Before either downstream operation may overwrite a mistake.
                    glue.moe_partial(b.moe.y[:rows], b.moe.wts[:rows], b.part_moe, rows)
                    gather(b.part_moe[:rows], b.g_moe[:8 * rows * dims])
                    torch.cuda.synchronize(device)
                    for name in expected:
                        _equal(observed[name], expected[name])
                        at = 8 * rows * dims if name in ('g_branch', 'g_moe') else rows
                        _equal(expected[name][at:], initial[name][at:])
                    for name in ('inj_a', 'g_branch', 'dn', 'dn_mix', 'part_moe', 'g_moe'):
                        _equal(expected[name], initial[name])
                    _equal(b.part_moe, expected_after[0])
                    _equal(b.g_moe, expected_after[1])
                    assert torch.isfinite(b.moe.y[:rows]).all()
                    if scale == 0.0:
                        assert torch.equal(b.moe.pick[0, :10], torch.arange(10, device=device, dtype=torch.int32))
                    assert torch.all(b.moe.pick[:rows, -1] == 512)
                    replay_count += 1
                assert region.hits == 3 and region.capture_router.calls == 2
                region.close()
                region = None
        assert replay_count == 18 and collective_calls[0] == 78
        owner.ready('hc-moe-oracle-complete', every=1, timeout=60)
        print(json.dumps({'rank': rank, 'exact_replays': replay_count, 'collective_calls': collective_calls[0],
                          'world': 8, 'rows': [30, 31, 32], 'roles': ['target-half', 'mtp-float'],
                          'scope': 'HC/MoE component only; no State/catalogue/Engine/native24 qualification'}), flush=True)
    except BaseException as error:
        primary = error
    # Abort a failed cohort before fencing its possibly blocked prior collective.
    # A failed abort stops dependent resource release; process supervision retains
    # the local owners until all children actually exit.
    try:
        if primary is not None:
            owner.close(abort=True)
        if region is not None:
            region.close()
        if primary is None:
            owner.close()
    except BaseException as error:
        secondary[0] = error
    if secondary[0] is not None:
        raise_failures(primary, secondary)
    if primary is not None:
        raise primary


def test_actual_tp8_mode3_hc_and_pure_moe_graph_preserve_all_outputs_and_tails():
    if torch.cuda.device_count() < 8 or any(torch.cuda.get_device_capability(i) != (7, 0) for i in range(8)):
        pytest.skip('requires eight actual SM70 devices and the pinned source foundation')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    context = None
    primary = None
    try:
        context = torch.multiprocessing.spawn(_worker, args=(port,), nprocs=8, join=False)
        deadline = time.monotonic() + 240
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError('HC/MoE TP8 oracle did not retire')
    except BaseException as error:
        primary = error
    if primary is not None:
        if context is None:
            raise RuntimeError('HC/MoE child launch is uncertain; retain this failed test owner') from primary
        secondary = [None]
        try:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            for process in context.processes:
                process.join(timeout=10)
            if any(process.is_alive() or process.exitcode is None for process in context.processes):
                raise RuntimeError('HC/MoE child retirement is uncertain')
        except BaseException as error:
            secondary[0] = error
        if secondary[0] is not None:
            from tensorfold.cleanup import raise_failures
            raise_failures(primary, secondary)
        raise primary
