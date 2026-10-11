"""Normal private FORMAT2 snapshot publication cleans owned files on write failure."""

from __future__ import annotations

from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine import prefix_snapshots as ps  # noqa: E402
from tests.snapshot_fixtures import fixture_registry  # noqa: E402 (requires native MLX)
from tensorfold.server.checkpoints import CheckpointEntry, spill_conversation  # noqa: E402

MODEL = "/models/qwen|mlx=1"


class Layer:
    def __init__(self) -> None:
        self.state = mx.arange(64).reshape(8, 8)
        self.offset = 8


def _disk_full(monkeypatch):
    """``mx.save_safetensors`` that writes some bytes, then fails as a full disk does."""

    def fail(path, arrays, metadata=None):
        with open(path, "wb") as handle:
            handle.write(b"\0" * 4096)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ps.mx, "save_safetensors", fail)


def test_a_failed_write_removes_its_partial_and_raises(tmp_path, monkeypatch):
    _disk_full(monkeypatch)
    with pytest.raises(OSError):
        ps.save_snapshot(tmp_path, MODEL, [1, 2, 3], [Layer()], registry=fixture_registry([Layer()]))
    assert list(tmp_path.iterdir()) == []


def test_a_failed_spill_leaves_the_directory_as_it_was(tmp_path, monkeypatch):
    ps.save_snapshot(tmp_path, MODEL, [1, 2], [Layer()], registry=fixture_registry([Layer()]))
    before = sorted(p.name for p in tmp_path.iterdir())
    _disk_full(monkeypatch)
    entry = CheckpointEntry(tokens=[1, 2, 3, 4], cache=[Layer()], last_prompt=[1, 2, 3, 4], nbytes=512)
    assert spill_conversation(entry, tmp_path, MODEL, limit_bytes=1 << 30, registry=fixture_registry([Layer()])) is False
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_each_write_uses_a_private_owned_directory(tmp_path, monkeypatch):
    written = []
    save = ps.mx.save_safetensors

    def record(path, arrays, metadata=None):
        written.append(Path(path))
        save(path, arrays, metadata=metadata)

    monkeypatch.setattr(ps.mx, "save_safetensors", record)
    target = ps.save_snapshot(tmp_path, MODEL, [4, 5], [Layer()], registry=fixture_registry([Layer()]))
    assert len(written) == 1 and written[0].name == "payload.safetensors"
    assert written[0].parent.name.startswith(".tensorfold-snapshot-")
    assert not written[0].parent.exists()
    assert [p.name for p in tmp_path.iterdir()] == [target.name]
