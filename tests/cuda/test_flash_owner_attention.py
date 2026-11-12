"""Sealed Flash owner controls; four logical ranks use copies on one CUDA stream.

This exercises real OwnerBuffers/FlashOwnerTiles and descriptor-owned Q8 data.
The collective shim is not NCCL/TP8 model or performance qualification.
"""
import os
from pathlib import Path
import threading

import pytest
import torch

from tensorfold.cuda.sm70 import attention_backends as backend
from tensorfold.cuda.sm70 import flash_v100_artifact as artifact
from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
from types import SimpleNamespace as NS
from tests.cuda.test_sm70_paged_attention import shard, expected_value

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='requires actual SM70')


@pytest.fixture(scope='module')
def artifacts():
    path = os.environ.get('TENSORFOLD_FLASH_NATIVE_MANIFEST')
    digest = os.environ.get('TENSORFOLD_FLASH_NATIVE_MANIFEST_SHA256')
    if not path or not digest: pytest.skip('requires explicitly staged sealed operator')
    checked = artifact.validate_manifest(Path(path), digest)
    owners = [artifact.load_native(checked) for _ in range(4)]
    try: yield owners
    finally:
        for owner in owners: owner.close()


def make_owner(native_artifact, degree, rank, group, rows):
    state = shard(degree, rank, group=group, rows=5)
    state.position.fill_(500); state.live.fill_(rows)
    text = {'num_attention_heads': 24, 'num_key_value_heads': 2, 'head_dim': 256,
            'hidden_size': 6144, 'indexer_budget': 2048, 'indexer_compress_ratio': 4}
    owner = OwnerBuffers.from_text(text, rows=5, prefill=True, idw=2052, world=8,
        rank=group+rank, device='cuda:0', comm=NS(world=8, rank=group+rank))
    # Fixture owns all descriptor-pointed data through positive owner retirement.
    owner.descriptors = state.table
    owner.global_stats = torch.empty(5, 12, 2, device='cuda')
    owner.native_layer, owner.native_page_tokens = state.layer, 128
    owner.native_buckets = (1, 2, 5)
    owner.native_pair = NS(key=NS(bits=8, codec=0), value=NS(bits=8, codec=0))
    forecast = backend.flash_owner_requirements(5, 3)
    owner.configure_native_family(backend.FLASH_OWNER_FAMILY, flash_artifact=native_artifact,
        flash_panel_rows=3, flash_tensor_budget_bytes=forecast['persistent_bytes'])
    views = owner.flash_tiles.frames[rows][0]
    owner.queries.zero_(); views['selected_all'].zero_()
    views['selected_all'][0, :, :10].copy_(state.selection[0, :rows, :10])
    views['selected_all'][0, :, -2:].copy_(state.selection[0, :rows, -2:])
    # A serving call must consume only the cold frame/prefix catalogue.
    owner.native_views = lambda rows: pytest.fail('runtime view construction')
    return owner, state, views


def four_rank_run(owners, rows):
    barrier, boot = threading.Barrier(4, timeout=20), threading.Barrier(4, timeout=20)
    sends, errors = [[None]*4 for _ in range(2)], []
    default = torch.cuda.default_stream()
    def worker(rank):
        owner = owners[rank]
        try:
            with torch.cuda.stream(default):
                owner.handoff_native_thread()
                phase = 0
                def gather(send, destination):
                    nonlocal phase
                    current = phase; phase += 1
                    sends[current][rank] = send
                    barrier.wait()
                    destination.zero_()
                    if current == 0: destination[..., 0].fill_(-torch.inf)
                    group = owner.rank // 4 * 4
                    for peer in range(4): destination[group+peer].copy_(sends[current][peer])
                    # All reads enter the same CUDA stream before any peer may
                    # overwrite its phase-borrowed SEND/RECEIVE storage.
                    barrier.wait()
                owner.gather_native = gather
                boot.wait()
                owner.native_run(rows)
        except BaseException as error:
            errors.append(error); barrier.abort(); boot.abort()
    threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(4)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if errors: raise errors[0]
    torch.cuda.synchronize()


@pytest.mark.parametrize('degree,group,rows', [(1, 0, 1), (2, 4, 2), (4, 0, 5), (4, 4, 5)])
def test_fixed_panel_tails_and_four_unique_owner_merge(artifacts, degree, group, rows):
    frames = [make_owner(artifacts[rank], degree, rank, group, rows) for rank in range(4)]
    try:
        four_rank_run([frame[0] for frame in frames], rows)
        expected = expected_value((8, 0)).to('cuda').expand(rows, 3, 256)
        for owner, state, views in frames:
            assert state.status.cpu().tolist() == [0, -2]
            assert int(state.live) == rows # Private ghost compute never pads State.
            assert torch.equal(views['out'], expected)
            assert owner.flash_tiles.requirements['panel_rows'] == 3
    finally:
        for owner, _, _ in frames: owner.close()


def single_owner_gather(owner):
    def gather(send, destination):
        destination.zero_()
        if send.shape[-1] == 2: destination[..., 0].fill_(-torch.inf)
        destination[owner.rank].copy_(send)
    owner.gather_native = gather


def test_captured_owner_reads_changed_q8_and_qsa_mask(artifacts):
    owner, state, views = make_owner(artifacts[0], 1, 0, 0, 2)
    single_owner_gather(owner)
    graph = torch.cuda.CUDAGraph()
    def pipeline():
        owner.queries.zero_() # Models the producer of the next query frame.
        return owner.native_run(2)
    try:
        pipeline()
        allocated = torch.cuda.memory_allocated()
        with torch.cuda.graph(graph): pipeline()
        state.vs.fill_(768.)
        state.ks.fill_(1.)
        views['selected_all'][0, :, :10].fill_(-1)
        views['selected_all'][0, :, 0] = 0
        graph.replay()
        expected = (expected_value((8, 0)).float()*3).half().to('cuda').expand(2,3,256)
        assert state.status.cpu().tolist() == [0,-2] and torch.equal(views['out'], expected)
        # Graph-private allocations, if any, are separate from eager run's
        # promised device-allocation-free prepared path.
        owner.queries.zero_(); before = torch.cuda.memory_allocated(); owner.native_run(2)
        assert torch.cuda.memory_allocated() == before and allocated > 0
    finally:
        torch.cuda.synchronize(); graph.reset(); owner.close()


@pytest.mark.parametrize('field,flag', [('query', 1), ('keys', 2), ('values', 4)])
def test_nonfinite_producer_status_remains_authoritative(artifacts, field, flag):
    owner, state, _ = make_owner(artifacts[0], 1, 0, 0, 2)
    single_owner_gather(owner)
    try:
        if field == 'query': owner.queries.fill_(torch.nan)
        else: getattr(state, 'ks' if field == 'keys' else 'vs').fill_(torch.nan)
        owner.native_run(2)
        # Failed scratch may be finite; publication authority is still status,
        # never a claimed universal upstream NaN propagation guarantee.
        assert state.status.cpu().tolist() == [flag, 3]
    finally: owner.close()


def test_bad_coordinate_generation_never_becomes_valid_empty_data(artifacts):
    owner, state, _ = make_owner(artifacts[0], 1, 0, 0, 1)
    single_owner_gather(owner)
    try:
        state.generations[0] = 1
        owner.native_run(1)
        assert state.status.cpu().tolist()[0] & 32
    finally: owner.close()


def test_same_live_vectors_preserve_bits_across_row_buckets_and_panel_positions(artifacts):
    owner, state, _ = make_owner(artifacts[0], 1, 0, 4, 5)
    single_owner_gather(owner)
    generator = torch.Generator(device='cuda').manual_seed(117)
    master = torch.randn(8, 1, 3, 256, generator=generator, device='cuda').half() * .125
    pattern = ((torch.arange(state.k.numel(), device='cuda') % 127) - 63).to(torch.int8).view_as(state.k)
    state.k.copy_(pattern); state.ks.fill_(8.)
    state.v.copy_(-pattern); state.vs.fill_(1.)
    reference = None
    try:
        for rows in (1, 2, 5):
            state.live.fill_(rows)
            views = owner.flash_tiles.frames[rows][0]
            views['q_all'].copy_(master.expand(8, rows, 3, 256))
            views['selected_all'].zero_()
            views['selected_all'][0, :, :10].copy_(state.selection[0, :rows, :10])
            views['selected_all'][0, :, -2:].copy_(state.selection[0, :rows, -2:])
            owner.native_run(rows)
            output = views['out'].clone()
            lse = owner.flash_tiles.lse[:rows].clone()
            assert state.status.cpu().tolist() == [0, -2]
            if reference is None: reference = output[:1].clone(), lse[:1].clone()
            assert torch.equal(output.view(torch.int16), reference[0].expand_as(output).view(torch.int16))
            assert torch.equal(lse.view(torch.int32), reference[1].expand_as(lse).view(torch.int32))
    finally: owner.close()
