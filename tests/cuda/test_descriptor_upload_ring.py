"""Real pinned-DMA ring reuse across streams, independent of model arithmetic."""
import pytest
import torch
from tensorfold.families.qwen4_exp.cuda.descriptor_upload import DescriptorUploadRing


@pytest.mark.parametrize("slots", [2, 4, 8])
def test_real_dma_ring_preserves_every_descriptor_version_across_streams(slots):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires real SM70 and the pinned source foundation")
    torch.cuda.set_device(0)
    ring = DescriptorUploadRing(32, slots, "cuda:0")
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    target = torch.empty((32, 16), dtype=torch.int64, device="cuda:0")
    outputs = torch.empty((slots * 8, 32, 16), dtype=torch.int64, device="cuda:0")
    try:
        for index in range(slots * 8):
            stream = streams[index % 2]
            with torch.cuda.stream(stream):
                host = ring.acquire()
                ring.before_gpu_write(stream)
                host.fill_(index + 1)
                target.copy_(host, non_blocking=True)
                # Same-stream consumer before next writer, and the event
                # includes this copy so the cross-stream join protects it.
                outputs[index].copy_(target)
                ring.publish(stream)
        ring.drain()
        torch.cuda.synchronize()
        expected = torch.arange(1, slots * 8 + 1, dtype=torch.int64).view(-1, 1, 1).expand(-1, 32, 16)
        assert torch.equal(outputs.cpu(), expected)
        assert all(host.is_pinned() for host in ring.views)
        assert not any(ring.pending)
    finally:
        torch.cuda.synchronize()
        ring.drain(globally_fenced=True)
