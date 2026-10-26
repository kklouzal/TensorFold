# ruff: noqa: E402 - selected optional Torch must precede CUDA-family imports.
"""Pinned native independent-field limits; no oversized daemon writes."""

from datetime import timedelta
import json
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

from tensorfold.engine.model_identity import canonical_runtime_json
from tensorfold.families.qwen4_exp.cuda.retained_catalog import make_retention_catalog, RetentionError, RetainedCatalog
from tensorfold.families.qwen4_exp.cuda.retained_parts import manifest_bound
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy

pytest_plugins = ("test_state_retention",)


@pytest.fixture
def native_catalog(materialized):
    server = torch.distributed.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=10))
    baseline = server.num_keys()
    weights = SimpleNamespace(meta={"rank": 0}, comm=SimpleNamespace(rank=0, world=1, store=server))
    policy = RetentionPolicy("ram", 1, 10, 1 << 30, 16 << 20, 32 << 20, 0)
    catalog = make_retention_catalog(policy, weights, [materialized], prefix_limit=0, recovery_versions=0)
    try:
        yield catalog
    finally:
        catalog.close(timeout=5)
        assert server.num_keys() == baseline


def test_existing_native_backend_caps_effective_chunk_without_changing_administrator_maximum(native_catalog):
    value = native_catalog
    assert value.comm.store.libuvBackend is True
    assert value.policy.chunk_bytes == 16 << 20
    assert value.transfer_chunk_bytes == value._buffer.nbytes == 8 << 20
    assert value.record()["native_store_value_limit_bytes"] == 8 << 20


def test_key_and_value_are_independent_native_fields_with_utf8_and_native_prefix(native_catalog):
    value = b"x" * (8 << 20)
    key = "\u00e9" * 4095 + "a"  #8191 UTF8 bytes, plus the native '/'.
    native_catalog._store_payload(key, value)
    assert len(key.encode()) + len(value) > 8 << 20  #No whole-wire8MiB cap.
    with pytest.raises(RetentionError, match="key field"):
        native_catalog._store_payload(key + "b", value)
    with pytest.raises(RetentionError, match="value field"):
        native_catalog._store_payload(key, value + b"x")
    # Validation precedes Store.set/get; no oversized daemon operation occurs.


@pytest.mark.parametrize("maximum", [4 << 20, 16 << 20])
def test_manifest_paging_reserves_actual_envelope_before_worstcase_outer_escaping(native_catalog, maximum):
    native_catalog.transfer_chunk_bytes = min(maximum, native_catalog._native_value_limit)
    width = native_catalog._manifest_width()
    def encoded(count):
        return canonical_runtime_json({"ok": True, "value": {"fragment": "\\" * count}}, max_bytes=16 << 20)
    accepted = encoded(width)
    native_catalog._store_payload("owned/fragment", accepted)
    assert len(accepted.encode()) <= 8 << 20
    with pytest.raises(RetentionError, match="value field"):
        native_catalog._store_payload("owned/fragment", encoded(width + 1))


def test_effective_chunk_drives_actual_bounded_copy_iteration(native_catalog):
    size = native_catalog.transfer_chunk_bytes + 17
    raw = torch.zeros(size, dtype=torch.uint8)
    class Builder:
        def __init__(self):
            self.chunks = []
        def write(self, chunk):
            self.chunks.append(len(chunk))
    writer = Builder()
    native_catalog._copy_source(raw, writer, native_catalog._buffer)
    assert writer.chunks == [8 << 20, 17]


def test_manifest_source_slice_retires_before_next_slice_allocation(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import retained_catalog

    class Slice(str):
        live = 0
        def __new__(cls, value):
            result = super().__new__(cls, value)
            cls.live += 1
            return result
        def __del__(self):
            type(self).live -= 1
    class Document(str):
        def __getitem__(self, index):
            assert Slice.live == 0, "previous source slice survived until next allocation"
            return Slice(super().__getitem__(index))
    local = {"metadata": {"schema": "wire-lifetime-only"}, "specs": [], "keys": [], "objects": []}
    original = retained_catalog._json
    monkeypatch.setattr(retained_catalog, "_json", lambda value: Document(original(value)) if value is local else original(value))
    # Real JSON roundtrip gives received strings separate from the source
    # slice; no model algorithm or alternate transport implementation.
    monkeypatch.setattr(retained_catalog, "_phase", lambda _engine, _channel, _stage, operation, **_kw:
                        [json.loads(json.dumps(operation()))])
    owner = object.__new__(RetainedCatalog)
    owner._store, owner.rank, owner._cohort = None, 0, None
    owner._manifest_bounds, owner.transfer_chunk_bytes, owner._native_value_limit = (manifest_bound(0),), 7, None
    assert owner._manifest("owned-wire", local) == [local]
    assert Slice.live == 0
