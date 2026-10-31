"""Owned NUMA ranges, physical placement, grants and exported-view lifetimes."""

import gc
import errno
import os
from pathlib import Path
import threading
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tensorfold.cuda import numa_memory as numa

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux NUMA allocation contract")


def test_gpu_uuid_discovery_uses_kernel_pci_identity_with_reordered_visible_devices():
    root = Path("/proc/driver/nvidia/gpus")
    if not root.is_dir():
        pytest.skip("CPU container has no read-only NVIDIA kernel inventory")
    for file in root.glob("*/information"):
        fields = dict(line.split(":", 1) for line in file.read_text().splitlines() if ":" in line)
        uuid = fields["GPU UUID"].strip()
        target = numa.configure_gpu_numa(0, torch_module=SimpleNamespace(
            cuda=SimpleNamespace(get_device_properties=lambda _device: SimpleNamespace(uuid=uuid))))
        assert target.pci_bus_id == file.parent.name
        assert target.node == int(Path(f"/sys/bus/pci/devices/{file.parent.name}/numa_node").read_text())
        assert set(target.cpus) <= os.sched_getaffinity(0)


def target(node):
    file = Path(f"/sys/devices/system/node/node{node}/cpulist")
    if not file.is_file():
        pytest.skip("requested NUMA node is absent on this host")
    cpus = tuple(sorted(numa._ids(file.read_text())
                        & os.sched_getaffinity(0)))
    if not cpus:
        pytest.skip("NUMA node is outside this CPU test's cpuset")
    return numa.Placement("GPU-test", "test", node, cpus, (node,), os.sysconf("SC_PAGE_SIZE"))


def test_affinity_scope_restores_after_body_failure_and_on_owned_worker():
    selected = target(1)
    previous = os.sched_getaffinity(0)
    with pytest.raises(LookupError, match="body"):
        with selected.scope():
            assert os.sched_getaffinity(0) == set(selected.cpus)
            raise LookupError("body")
    assert os.sched_getaffinity(0) == previous
    observations = []
    def worker():
        before = os.sched_getaffinity(0)
        with selected.scope():
            observations.append(os.sched_getaffinity(0))
        observations.append(os.sched_getaffinity(0) == before)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert observations == [set(selected.cpus), True]


def test_node_budget_is_aggregate_and_uses_explicit_caller_allowances():
    rows = [{"rank": 0, "node": 0, "bytes": 7}, {"rank": 1, "node": 0, "bytes": 9},
            {"rank": 2, "node": 1, "bytes": 11}]
    assert numa.verify_node_grants(rows, {0: 16, 1: 11})["admitted"]
    with pytest.raises(MemoryError):
        numa.verify_node_grants(rows, {0: 15, 1: 12})
    with pytest.raises(ValueError):
        numa.verify_node_grants(rows+[rows[0]], {0: 100, 1: 100})
    assert not numa.node_memory_snapshot(0)["snapshot_is_reservation"]


@pytest.mark.parametrize("node", [0, 1])
def test_native_private_mapping_is_fully_written_and_physically_node_local(node):
    selected = target(node)
    receipt = selected.probe()
    assert receipt["verified"] and receipt["node_pages"] == {node: 1}
    with selected.scope():
        size = (4096+17)*selected.page_size  # Cross the bounded move_pages query batch.
        mapping = numa._native().Mapping(size, node)
        assert memoryview(mapping).readonly is False
        assert bytes(memoryview(mapping)[:1024]) == bytes(1024)
        assert numa._verify(mapping, selected)["node_pages"] == {node: size//selected.page_size}


def test_exported_tensor_retains_native_mapping_and_actual_grant_until_last_alias():
    import torch

    selected = target(1)
    charge = numa.backing_bytes(4096, page_size=selected.page_size)
    allocator = numa.HostAllocator(selected, byte_limit=charge, pinned_limit=0)
    owner = allocator((2048,), torch.float16)
    alias = owner.tensor[13:]
    assert allocator.used == charge
    owner.close()
    assert allocator.buffers == [] and allocator.used == charge
    alias[0] = 17
    assert alias[0].item() == 17
    with pytest.raises(MemoryError):
        allocator((2048,), torch.float16)
    del alias
    gc.collect()
    assert allocator.used == 0
    for _ in range(5):
        replacement = allocator((2048,), torch.float16)
        assert allocator.used == charge
        replacement.close()
        gc.collect()
        assert allocator.used == 0 and allocator.buffers == []
    allocator.close()


def test_failed_binding_cannot_publish_or_leak_a_mapping(monkeypatch):
    import torch

    selected = target(0)
    allocator = numa.HostAllocator(selected, byte_limit=16384, pinned_limit=0)
    original = numa._native()
    monkeypatch.setattr(numa, "_native", lambda: SimpleNamespace(Mapping=lambda *args: (_ for _ in ()).throw(
        PermissionError("bind blocked"))))
    with pytest.raises(PermissionError):
        allocator((1024,), torch.float16)
    assert allocator.used == 0
    allocator.close()
    assert original is not None


def test_default_gpu_mode_import_has_no_cuda_initialization():
    subprocess.run([sys.executable, "-c", "import torch; from tensorfold.cuda import numa_memory; "
                    "assert not torch.cuda.is_initialized()"], check=True)


def test_guarded_vma_can_prove_actual_pages_when_query_capability_is_denied():
    selected = target(1)
    with selected.scope():
        mapping = numa._native().Mapping(selected.page_size*17, selected.node)
        def denied():
            raise PermissionError(errno.EPERM, "query denied")
        proxy = SimpleNamespace(address=mapping.address, extent=mapping.extent, nodes=denied)
        proof = numa._verify(proxy, selected)
        assert proof["method"] == "guarded-vma-numa_maps"
        assert proof["node_pages"] == {1: 17}


def test_nonlocal_pages_and_unknown_query_failures_do_not_become_affinity_proof():
    selected = target(1)
    with pytest.raises(RuntimeError, match="not entirely"):
        numa._verify(SimpleNamespace(extent=selected.page_size*2, nodes=lambda: {0: 1, 1: 1}), selected)
    def broken():
        raise OSError(errno.EFAULT, "invalid range")
    with pytest.raises(OSError) as caught:
        numa._verify(SimpleNamespace(nodes=broken), selected)
    assert caught.value.errno == errno.EFAULT


def test_restore_failure_preserves_primary_and_reports_cleanup(monkeypatch):
    selected = target(1)
    previous = os.sched_getaffinity(0)
    actual = os.sched_setaffinity
    count = 0
    def fail_restore(pid, cpus):
        nonlocal count
        count += 1
        if count > 1:
            raise OSError("restore failed")
        return actual(pid, cpus)
    primary = LookupError("body")
    monkeypatch.setattr(os, "sched_setaffinity", fail_restore)
    try:
        with pytest.raises(LookupError) as caught:
            with selected.scope():
                raise primary
        assert caught.value is primary
        assert isinstance(caught.value.__cause__, OSError)
    finally:
        actual(0, previous)


def test_failed_unregister_retains_owner_and_grants_until_explicit_retirement():
    import torch
    from tensorfold.cuda.expert_pin import PinCleanupError

    selected = target(1)
    charge = numa.backing_bytes(4096, page_size=selected.page_size)
    allocator = numa.HostAllocator(selected, byte_limit=charge, pinned_limit=4096)
    owner = allocator((2048,), torch.float16)
    tensor, mapping = owner.tensor, owner._mapping
    owner._pin = SimpleNamespace(close=lambda: 17)
    owner._pin_charge = 4096
    with pytest.raises(PinCleanupError) as caught:
        owner.close()
    assert caught.value.owner is owner
    assert owner.tensor is tensor and owner._mapping is mapping
    assert allocator.used == charge and allocator.pinned == 4096 and owner in allocator.buffers
    owner._pin = None  # Only a mocked registration existed in this CPU test.
    owner.close()
    del tensor, mapping
    gc.collect()
    assert allocator.used == allocator.pinned == 0
    allocator.close()


def test_backing_budget_counts_page_rounding_and_unfaulted_guard_ranges():
    page = os.sysconf("SC_PAGE_SIZE")
    assert numa.backing_bytes(0) == 0
    assert numa.backing_bytes(1) == numa.backing_bytes(page) == 3*page
    assert numa.backing_bytes(page+1) == 4*page
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError):
            numa.backing_bytes(value)


def test_closed_session_does_not_reopen_allocation_authority():
    import torch
    selected = target(0)
    allocator = numa.HostAllocator(selected, byte_limit=16384, pinned_limit=0)
    allocator((1,), torch.float16)
    allocator.close()
    with pytest.raises(RuntimeError, match="closed"):
        allocator((1,), torch.float16)
