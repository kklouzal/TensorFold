"""Actual loaded CUDA registration over verified node-local owned host pages."""

import ctypes
import gc
import json
from pathlib import Path

import pytest
import torch

from tensorfold.cuda import numa_memory as numa

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host registration")


def test_registered_stage_transfer_capture_and_exported_view_retirement():
    placement = numa.configure_gpu_numa(torch.device("cuda", 0))
    count = 8 << 20
    archive_bytes = 2 << 30
    allocator = numa.HostAllocator(placement,
        byte_limit=numa.backing_bytes(archive_bytes)+numa.backing_bytes(count), pinned_limit=count)
    archive = allocator((archive_bytes//2,), torch.float16, tag="two-GiB-pageable-archive")
    stage = allocator((count,), torch.uint8, pinned=True, tag="bounded-registered-stage")
    assert archive.receipt["before_pin"]["node_pages"] == {placement.node: archive_bytes//placement.page_size}
    assert stage.tensor.is_pinned()
    assert stage.receipt["before_pin"]["node_pages"] == stage.receipt["after_pin"]["node_pages"]
    assert stage.receipt["after_pin"]["node_pages"] == {placement.node: count//placement.page_size}
    assert "libcudart" in Path(stage.receipt["cuda_runtime"]["library"]).name
    assert stage.receipt["cuda_runtime"]["version"] // 1000 == 12
    stream = torch.cuda.Stream()
    source = (torch.arange(count, device="cuda", dtype=torch.int64)%251).to(torch.uint8)
    device_copy = torch.empty_like(source)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            stage.tensor.copy_(source, non_blocking=True)
        stream.synchronize()  # The caller's explicit consumption fence.
        with placement.scope():
            assert stage.tensor.numpy().tobytes() == bytes(i%251 for i in range(count))
            ctypes.memset(stage.tensor.data_ptr(), 113, count)
        with torch.cuda.stream(stream):
            device_copy.copy_(stage.tensor, non_blocking=True)
        stream.synchronize()
        assert bool((device_copy == 113).all())
        with torch.cuda.graph(graph, stream=stream):
            stage.tensor.copy_(source, non_blocking=True)
        for value in (17, 211):
            source.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            with placement.scope():
                assert stage.tensor.numpy().tobytes() == bytes([value])*count
        print(json.dumps({"archive": archive.receipt, "stage": stage.receipt,
                          "loaded_cuda_libraries": sorted({line.split()[-1] for line in
                              Path("/proc/self/maps").read_text().splitlines() if "libcudart.so" in line})}), flush=True)
        alias = stage.tensor[:17]
        before = allocator.used
        graph.reset()
        stage.close()  # Every GPU reader/writer has completed before unregister.
        assert allocator.pinned == 0 and allocator.used == before
        with placement.scope():
            assert alias.numpy().tobytes() == bytes([211])*17
        assert not alias.is_pinned()
        del alias
        gc.collect()
        assert allocator.used == numa.backing_bytes(archive_bytes)
    finally:
        torch.cuda.synchronize()
        graph.reset()
        stage.close()
        archive.close()
        allocator.close()
    assert allocator.used == 0
