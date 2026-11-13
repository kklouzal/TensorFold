"""CPU fixture/observer boundaries only; never native or TP8 adoption proof."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from tests.g3_packed_native_support import CAPACITY, LAYERS, SHAPES, CutCopies, cut_rows, load_input, payload_bound


def test_finite_packed_shapes_and_paid_payload():
    assert SHAPES["tails"] == (63, 64, 65)
    assert (len(SHAPES["packed24"]), len(SHAPES["packed32"])) == (24, 32)
    assert all(sum(rows) <= CAPACITY for rows in SHAPES.values())
    assert sum(SHAPES["packed24"]) == sum(SHAPES["packed32"]) == CAPACITY
    assert payload_bound(32)["total"] == 216639189 < 256 * (1 << 20)
    for lengths in SHAPES.values():
        for rows in lengths:
            points = cut_rows(rows)
            assert len(points) <= 3 and tuple(sorted(set(points))) == points
            assert all(0 < row < rows for row in points)


def test_real_dispatch_copy_once_then_snapshot_survives_source_rewrite():
    cuts = [SimpleNamespace(rec=torch.empty(LAYERS, 6, 128, 128))]
    source = torch.randn(6, 128, 128)
    source[0, 0, 0] = -0.0
    expected = source.clone()
    with CutCopies(cuts, gpu_budget_bytes=2 * source.numel() * 4, host_budget_bytes=1 << 20) as observer:
        for layer in range(LAYERS):
            cuts[0].rec[layer].copy_(source)
        source.fill_(9)
    observer.assert_bits()
    assert observer.logical_clone_bytes == 2 * source.numel() * 4
    for layer in range(LAYERS):
        assert torch.equal(cuts[0].rec[layer].view(torch.int32), expected.view(torch.int32))
    # This uses the protected public mode's real redispatch, not mocked ATen.
    assert len(observer.sources) == LAYERS


def test_copy_observer_refuses_unpaid_headers_and_clone_storage():
    cuts = [SimpleNamespace(rec=torch.empty(LAYERS, 6, 128, 128))]
    with pytest.raises(MemoryError, match="Python-header"):
        CutCopies(cuts, gpu_budget_bytes=1 << 20, host_budget_bytes=0)
    source = torch.zeros(6, 128, 128)
    with CutCopies(cuts, gpu_budget_bytes=0, host_budget_bytes=1 << 20), pytest.raises(MemoryError, match="GPU credit"):
        cuts[0].rec[0].copy_(source)
    assert torch.equal(cuts[0].rec[0], source)  # Original copy forwarded before failed observation.


def bind_input(monkeypatch, tmp_path, value):
    raw = json.dumps(value).encode()
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    monkeypatch.setenv("TENSORFOLD_G3_NATIVE_INPUT", str(path))
    monkeypatch.setenv("TENSORFOLD_G3_NATIVE_INPUT_BYTES", str(len(raw)))
    monkeypatch.setenv("TENSORFOLD_G3_NATIVE_INPUT_SHA256", hashlib.sha256(raw).hexdigest())
    return path


def input_value():
    return {
        "schema": "tensorfold.g3-packed-native-input-v1",
        "source_uid": "a" * 64,
        "image_id": "sha256:" + "b" * 64,
        "cuda_home": "/usr/local/cuda-12.9",
        "nvcc_sha256": "c" * 64,
        "artifact": {},
        "fixture_budget_bytes": 256 * (1 << 20),
        "journal_host_budget_bytes": 1 << 26,
        "observer_host_budget_bytes": 1 << 20,
    }


def test_input_actual_identity_and_held_bytes(monkeypatch, tmp_path):
    value = input_value()
    path = bind_input(monkeypatch, tmp_path, value)
    assert load_input() == value
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="size"):
        load_input()


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_uid", "x"),
        ("image_id", "sha256:x"),
        ("nvcc_sha256", False),
        ("cuda_home", "relative"),
        ("fixture_budget_bytes", True),
        ("journal_host_budget_bytes", 0),
        ("unexpected", 0),
    ],
)
def test_input_rejects_unbound_metadata(monkeypatch, tmp_path, key, value):
    data = input_value()
    data[key] = value
    bind_input(monkeypatch, tmp_path, data)
    with pytest.raises(ValueError, match="externally bound"):
        load_input()
