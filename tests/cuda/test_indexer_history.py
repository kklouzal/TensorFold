"""Actual NUMA/pinned D2H capture and arbitrary-prefix RAM-history lifecycle."""
import pytest
import torch

from tensorfold.cuda.numa_memory import HostAllocator, configure_gpu_numa
from tensorfold.families.qwen4_exp.cuda.indexer_history import HistoryPlan, HistorySession

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RAM indexer owner")


def test_numa_ram_history_capture_replay_partial_commit_clone_and_restore():
    device = torch.device("cuda", torch.cuda.current_device())
    target = configure_gpu_numa(device)
    plan = HistoryPlan(2, 128, 4, 8, 2, page_size=target.page_size)
    allocator = HostAllocator(target, byte_limit=3 * plan.host_bytes(64), pinned_limit=3 * plan.pinned_bytes)
    session = HistorySession(plan, allocator, device, maximum_owners=3)
    owner = session.new(64)
    status = torch.zeros(2, dtype=torch.int32, device=device)
    source = torch.arange(2 * plan.decode_stage_rows * 128, device=device).reshape(
        2, plan.decode_stage_rows, 128).to(torch.float16)
    graph = torch.cuda.CUDAGraph()
    try:
        lease = owner.begin("decode", 0, 2)
        with torch.cuda.graph(graph):
            owner.frames["decode"].gpu.copy_(source)
            lease.enqueue()
        for position, count in ((0, 2), (1, 2), (3, 1)):
            current = lease if position == 0 else owner.begin("decode", position, count)
            source.add_(.5)
            graph.replay()
            # A ready event from a preceding replay is insufficient without
            # this actual replay's external submission record.
            with pytest.raises(RuntimeError, match="DMA"):
                current.validated(fenced=True)
            current.mark_submitted()
            status.cpu()                 # The existing producing-stream status/sampling wait, exactly once.
            current.validated(fenced=True)
            current.commit(1 if position == 0 else count)
            expected = source[:, :count].cpu()
            assert torch.equal(owner.archive[:, position:position + count].view(torch.int16),
                               expected.view(torch.int16))
        snapshot = owner.snapshot([2, 3])
        cloned = owner.clone()
        copied = session.new(64)
        copied.copy_prefix(owner, [2, 3])
        owner.reset()
        owner.restore(snapshot, [2, 3])
        for actual in (owner, copied):
            status.cpu()
            for layer, length in enumerate((2, 3)):
                assert torch.equal(actual.ring[layer, :length].cpu().view(torch.int16),
                                   actual.archive[layer, :length].view(torch.int16))
        assert cloned.archive.data_ptr() != owner.archive.data_ptr()
        assert owner.device_nbytes == plan.device_nbytes
        assert owner.host_nbytes == plan.host_bytes(64)
        print({"placement": target.receipt, "allocation": allocator.receipt,
               "device_bytes": owner.device_nbytes, "host_bytes": owner.host_nbytes}, flush=True)
    finally:
        graph.reset()                     # Captured host/device addresses retire before their owner.
        session.close()
    assert session.closed and owner.host_nbytes == owner.device_nbytes == 0
