"""Physical stage metadata and existing cohort protocol; no GPU qualification."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import io
import json
import threading
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from tensorfold.families.qwen4_exp.cuda import stage_topology as stage


def records():
    return [{"schema_version": 1, "rank": rank, "host": "a" * 64,
             "uuid": f"GPU-00000000-0000-0000-0000-{rank + 1:012x}",
             "name": "NVIDIA Tesla V100-SXM2-32GB", "capability": [7, 0],
             # Reported usable memory may be below the nominal SKU capacity.
             "memory_bytes": 32 * 2**30 - 256 * 2**20} for rank in range(8)]


def test_valid_stage_receipt_binds_order_without_exposing_boot_or_gpu_ids():
    rows = records()
    receipt = stage.validate_pp1(rows)
    assert receipt["server_count"] == 1 and receipt["world_size"] == receipt["tp"] == 8
    assert len(set(receipt["rank_gpu_sha256"])) == 8
    assert rows[0]["uuid"] not in json.dumps(receipt)
    assert "pending" in receipt["qualification"]
    alias = deepcopy(rows)
    alias[0]["uuid"] = UUID(rows[0]["uuid"][4:])
    assert stage.validate_pp1(alias) == receipt


@pytest.mark.parametrize("change", ["short", "rank", "rank-bool", "host", "host-missing", "uuid-empty",
    "uuid-malformed", "uuid-alias", "capability", "capability-bool", "pcie", "16gb", "other-device",
    "memory-zero", "memory-bool", "memory-overflow", "name-large", "version", "host-format", "extra"])
def test_invalid_stage_cannot_supply_an_admission_receipt(change):
    rows = records()
    if change == "short":
        rows.pop()
    elif change == "rank":
        rows[3]["rank"] = 4
    elif change == "rank-bool":
        rows[1]["rank"] = True
    elif change == "host":
        rows[4]["host"] = "b" * 64
    elif change == "host-missing":
        rows[4].pop("host")
    elif change == "uuid-empty":
        rows[4]["uuid"] = ""
    elif change == "uuid-malformed":
        rows[4]["uuid"] = "GPU-not-a-uuid"
    elif change == "uuid-alias":
        rows[4]["uuid"] = rows[0]["uuid"][4:].upper()
    elif change == "capability":
        rows[4]["capability"] = [8, 0]
    elif change == "capability-bool":
        rows[4]["capability"] = [True, 0]
    elif change == "pcie":
        rows[4]["name"] = "Tesla V100-PCIE-32GB"
    elif change == "16gb":
        rows[4]["name"] = "Tesla V100-SXM2-16GB"
    elif change == "other-device":
        rows[4]["name"] = "NVIDIA H100"
    elif change == "memory-zero":
        rows[4]["memory_bytes"] = 0
    elif change == "memory-bool":
        rows[4]["memory_bytes"] = True
    elif change == "memory-overflow":
        rows[4]["memory_bytes"] = 2**63
    elif change == "name-large":
        rows[4]["name"] += "x" * 256
    elif change == "version":
        rows[4]["schema_version"] = True
    elif change == "host-format":
        for row in rows:
            row["host"] = "not-a-fingerprint"
    elif change == "extra":
        rows[4]["unexpected"] = "other protocol"
    with pytest.raises(ValueError):
        stage.validate_pp1(rows)


@pytest.mark.parametrize("raw", [b"", b"invalid", b"\xff", b"a" * 129])
def test_host_identity_rejects_missing_malformed_or_oversized_input(raw, monkeypatch):
    monkeypatch.setattr(stage, "_regular_stream", lambda path: io.BytesIO(raw))
    with pytest.raises(ValueError):
        stage.host_identity()


@pytest.mark.parametrize("form", ["stdlib", "string", "torch-native"])
def test_capture_uses_assigned_device_uuid_and_kernel_identity(monkeypatch, form):
    boot = UUID("953e0462-dfab-541a-e631-573d64f8c5ad")
    opened, devices = [], []

    def source(path):
        opened.append(path)
        return io.BytesIO((str(boot) + "\n").encode())

    # Match the recorded binding's bare hyphenated spelling with a synthetic
    # device ID. This is a CPU provider-boundary stand-in, not a native object.
    spelling = str(UUID(int=17))
    class NativeUUID:
        def __str__(self):
            return spelling
    device_uuid = {"stdlib": UUID(spelling), "string": "GPU-" + spelling, "torch-native": NativeUUID()}[form]
    props = NS(uuid=device_uuid, name="Tesla V100-SXM2-32GB", major=7, minor=0, total_memory=32 << 30)
    def properties(device):
        devices.append(device)
        return props

    monkeypatch.setattr(stage, "_regular_stream", source)
    provider = NS(cuda=NS(get_device_properties=properties), _C=NS(_CUuuid=NativeUUID))
    value = stage.capture_rank(3, provider)
    assert devices == [0] and value["uuid"] == spelling
    assert value["host"] == hashlib.sha256(boot.bytes).hexdigest()
    assert opened == ["/proc/sys/kernel/random/boot_id"]  # Container machine-id is not a host binding.


@pytest.mark.parametrize("spelling", ["", "not-a-uuid", "GPU-invalid", "x" * 65])
def test_native_property_formatter_still_passes_the_strict_scalar_validator(monkeypatch, spelling):
    class NativeUUID:
        def __str__(self):
            return spelling
    monkeypatch.setattr(stage, "host_identity", lambda: "a" * 64)
    provider = NS(cuda=NS(get_device_properties=lambda device: NS(uuid=NativeUUID())),
                  _C=NS(_CUuuid=NativeUUID))
    with pytest.raises(ValueError, match="GPU UUID"):
        stage.capture_rank(0, provider)


def test_unknown_and_subclass_uuid_objects_are_not_stringified_or_accepted(monkeypatch):
    class NativeUUID:
        def __str__(self):
            pytest.fail("an unrecognized UUID object was stringified")
    class Subclass(NativeUUID):
        pass
    class Spoof:
        __module__ = "torch._C"
        def __str__(self):
            pytest.fail("a name-only UUID spoof was stringified")
    Spoof.__name__ = "_CUuuid"
    monkeypatch.setattr(stage, "host_identity", lambda: "a" * 64)
    for value in (Subclass(), Spoof(), object(), b"00000000-0000-0000-0000-000000000011"):
        provider = NS(cuda=NS(get_device_properties=lambda device, value=value: NS(uuid=value)),
                      _C=NS(_CUuuid=NativeUUID))
        with pytest.raises(ValueError, match="GPU UUID"):
            stage.capture_rank(0, provider)
        with pytest.raises(ValueError, match="GPU UUID"):
            stage.canonical_gpu_uuid(value)
    # A raw native-like object is forbidden in an untrusted rank record too.
    with pytest.raises(ValueError, match="GPU UUID"):
        stage.canonical_gpu_uuid(NativeUUID())


def test_captured_native_identity_retains_duplicate_and_ordered_rank_refusals(monkeypatch):
    spelling = str(UUID(int=17))
    class NativeUUID:
        def __str__(self):
            return spelling
    props = NS(uuid=NativeUUID(), name="Tesla V100-SXM2-32GB", major=7, minor=0, total_memory=32 << 30)
    provider = NS(cuda=NS(get_device_properties=lambda device: props), _C=NS(_CUuuid=NativeUUID))
    monkeypatch.setattr(stage, "host_identity", lambda: "a" * 64)
    rows = records()
    rows[0] = stage.capture_rank(0, provider)
    stage.validate_pp1(rows)
    rows[1] = stage.capture_rank(1, provider)
    with pytest.raises(ValueError, match="distinct physical GPU UUIDs"):
        stage.validate_pp1(rows)
    rows = records()
    rows[0] = stage.capture_rank(1, provider)
    with pytest.raises(ValueError, match="ordered ranks"):
        stage.validate_pp1(rows)


def test_missing_device_uuid_is_an_agreed_metadata_refusal(monkeypatch):
    monkeypatch.setattr(stage, "host_identity", lambda: "a" * 64)
    with pytest.raises(ValueError, match="GPU UUID"):
        stage.capture_rank(0, NS(cuda=NS(get_device_properties=lambda device: NS())))


def test_stage_frames_and_keys_fit_existing_startup_protocol_envelopes():
    from tensorfold.families.qwen4_exp.cuda.execution_identity import phase_failure_value_bytes
    from tensorfold.families.qwen4_exp.cuda.retention_resources import (
        INITIAL_COHORT_VALUE_BYTES, native_sync_control_payload, native_sync_fixed_payload,
    )

    assert phase_failure_value_bytes() <= 8192 < INITIAL_COHORT_VALUE_BYTES
    fixed = native_sync_fixed_payload(world=8, ram=False, slots=1, fd_limit=1024)
    # Retired before communicator/loading phases. Even their whole bootstrap
    # key allowance plus two outcome and two retirement waves fits the existing cap.
    assert (2 * 8 + 2) + 4 * 8 <= fixed["stored_key_bound"] <= fixed["value_map_key_highwater_bound"]
    control = native_sync_control_payload(world=8, chunk_bytes=0, ram=False, slots=1)
    assert 5 * 8 * 8192 <= control["constructor_value_bytes"]


class Store:
    """In-memory test transport with the real Store's byte-valued reads."""
    def __init__(self, *, fail_capture=False, oversize=False):
        self.condition, self.values = threading.Condition(), {}
        self.aborted, self.fail_capture = False, fail_capture
        self.oversize = oversize
        self.max_value = self.highwater = 0

    def set(self, key, value):
        raw = value.encode() if isinstance(value, str) else value
        with self.condition:
            if self.aborted:
                raise OSError("test supervisor contained cohort")
            self.values[key] = raw
            self.max_value = max(self.max_value, len(raw))
            self.highwater = max(self.highwater, len(self.values))
            self.condition.notify_all()
            if self.fail_capture and key.endswith("/capture/3"):
                raise OSError("test partial publication failure")

    def get(self, key):
        with self.condition:
            if self.aborted:
                raise OSError("test supervisor contained cohort")
            if self.oversize and key.endswith("/capture/3"):
                return b" " * 8193
            return self.values[key]

    def wait(self, keys, timeout):
        with self.condition:
            if not self.condition.wait_for(lambda: self.aborted or all(key in self.values for key in keys), timeout=5):
                raise TimeoutError("test stage cohort did not complete")
            if self.aborted:
                raise OSError("test supervisor contained cohort")

    def delete_key(self, key):
        with self.condition:
            return self.values.pop(key, None) is not None

    def abort(self):
        with self.condition:
            self.aborted = True
            self.condition.notify_all()


@pytest.mark.parametrize("failure", [None, "capture", "host", "uuid", "publication", "oversize",
                                     "legacy", "mixed", "required-type"])
def test_all_rank_agreement_refusal_retirement_and_fatal_containment(failure, monkeypatch):
    from tensorfold.cuda import cohort

    rows = records()
    if failure == "host":
        rows[4]["host"] = "b" * 64
    if failure == "uuid":
        rows[4]["uuid"] = rows[0]["uuid"][4:]

    def capture(rank, torch):
        if failure == "legacy":
            pytest.fail("uniform legacy admission must not query physical GPU eligibility")
        if failure == "capture" and rank == 3:
            raise ValueError("PP1 stage identity unavailable on rank3")
        return deepcopy(rows[rank])

    monkeypatch.setattr(stage, "capture_rank", capture)
    store = Store(fail_capture=failure == "publication", oversize=failure == "oversize")
    monkeypatch.setattr(cohort, "notify_failure", store.abort)  # Emulate the actual supervisor's cohort containment.
    engines = [NS(rank=rank, tp=8, _startup_store=store, _abort_comm=False) for rank in range(8)]
    errors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        participation = [True] * 8
        if failure == "legacy":
            participation = [False] * 8
        elif failure == "mixed":
            participation[4] = False
        elif failure == "required-type":
            participation[4] = 1
        pending = [pool.submit(stage.prepare_pp1, engine, None, required=participation[engine.rank])
                   for engine in engines]
        for future in pending:
            if failure and failure != "legacy":
                with pytest.raises((ValueError, OSError)) as caught:
                    future.result(timeout=10)
                errors.append(caught.value)
            else:
                assert future.result(timeout=10) == (None if failure == "legacy" else stage.validate_pp1(rows))
    if failure in ("publication", "oversize"):
        assert store.aborted and all(engine._abort_comm for engine in engines)
        if failure == "oversize":
            assert any("received cohort value exceeds" in str(error) for error in errors)
    else:
        assert not store.values and not store.aborted
        assert not any(engine._abort_comm for engine in engines)
    assert store.max_value <= 8192
    assert store.highwater <= 32  # Two W-outcome phases plus the existing two retirement waves.


@pytest.mark.torch
@pytest.mark.parametrize("mode", ["canonical", "legacy", "mixed"])
def test_real_cpu_store_preserves_common_envelope_and_retires_keys(mode, monkeypatch):
    """Host SDK transport exercise; not selected-x86-ABI or V100 evidence.

    Clients with either use_libuv request still use the selected server's Store
    protocol. A test-only wait deadline bounds failures; no CUDA is acquired.
    """
    from datetime import timedelta
    import torch
    from torch.distributed import TCPStore

    rows = records()
    monkeypatch.setattr(stage, "capture_rank", lambda rank, torch: deepcopy(rows[rank]))
    server = TCPStore("127.0.0.1", 0, 8, True, wait_for_workers=False,
                      timeout=timedelta(seconds=5), use_libuv=False)
    stores = [server] + [TCPStore("127.0.0.1", server.port, 8, False, wait_for_workers=False,
                                  timeout=timedelta(seconds=5), use_libuv=bool(rank % 2))
                         for rank in range(1, 8)]
    baseline = server.num_keys()
    engines = []
    for rank, native in enumerate(stores):
        def bounded_wait(keys, timeout, native=native):
            assert timeout in (timedelta(seconds=3600), timedelta(seconds=60))
            native.wait(keys, timedelta(seconds=5))
        wrapper = NS(set=native.set, get=native.get, delete_key=native.delete_key, wait=bounded_wait)
        engines.append(NS(rank=rank, tp=8, _startup_store=wrapper, _abort_comm=False))
    flags = [mode != "legacy"] * 8
    if mode == "mixed":
        flags[4] = False
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = [pool.submit(stage.prepare_pp1, engine, None, required=flags[engine.rank]) for engine in engines]
        for future in pending:
            if mode == "mixed":
                with pytest.raises(ValueError, match="different admission participation"):
                    future.result(timeout=10)
            else:
                assert future.result(timeout=10) == (None if mode == "legacy" else stage.validate_pp1(rows))
    assert server.num_keys() == baseline
    assert not torch.cuda.is_initialized()
