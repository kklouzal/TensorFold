"""Neutral GGUF inventory and explicit family admission use the same bytes."""
import pytest

from tensorfold import checkpoint_metadata, gguf_file
from tensorfold.families.qwen4_exp.gguf import metadata_config, metadata_config_from_checkpoint, open_checkpoint
from gguf_fixture import tiny_model, write
from test_flashnext_gguf import fixture


def test_other_architecture_inventory_is_not_model_support(tmp_path):
    path = fixture(tmp_path / "other.gguf", [("w", (1,), 0, bytes(4))],
                   metadata={"general.architecture": "other-family"})
    checkpoint = gguf_file.GGUFCheckpoint.open(path)
    assert checkpoint.metadata["general.architecture"] == "other-family"
    assert checkpoint.tensors["w"].nbytes == 4
    for read in (open_checkpoint, metadata_config, checkpoint_metadata.read_config):
        with pytest.raises(ValueError, match="architecture|Flash Next"):
            read(path)
    with pytest.raises(ValueError, match="architecture"):
        metadata_config_from_checkpoint(checkpoint, path)


def test_discovery_reuses_inventory_once_and_preserves_second_shard(tmp_path, monkeypatch):
    source = tiny_model(tmp_path / "source.gguf")
    fields = dict(gguf_file.GGUFCheckpoint.open(source).metadata)
    first = tmp_path / "split-00001-of-00002.gguf"
    second = tmp_path / "split-00002-of-00002.gguf"
    common = {**fields, "split.count": 2, "split.tensors.count": 1}
    write(first, [], {**common, "split.no": 0})
    write(second, [("token_embd.weight", (32, 1), 0, [0] * 128)], {**common, "split.no": 1})
    original = gguf_file._shard
    reads = []

    def counted(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(gguf_file, "_shard", counted)
    config = checkpoint_metadata.read_config(second)
    assert reads == [first, second]
    assert config["gguf_entrypoint"] == str(second)
    assert config["text_config"]["gguf_entrypoint"] == str(second)
    reads.clear()
    assert metadata_config(second) == config
    assert reads == [first, second]


def test_inventory_handoff_supports_mtp_tensor_validation_without_reopen(tmp_path, monkeypatch):
    from test_flashnext_gguf_mtp import fixture as mtp_fixture

    def positions(metadata, _entries):
        metadata["qwen4exp.context_length"] = 262144
        metadata["qwen4exp.rope.dimension_sections"] = (2, 1, 1, 0)

    _, path, _ = mtp_fixture(tmp_path, mutate=positions)
    checkpoint = gguf_file.GGUFCheckpoint.open(path)
    expected = metadata_config(path, role="mtp")
    monkeypatch.setattr(gguf_file.GGUFCheckpoint, "open", lambda _: pytest.fail("reopened validated inventory"))
    assert metadata_config_from_checkpoint(checkpoint, path, role="mtp") == expected
    assert metadata_config_from_checkpoint(checkpoint, str(path), role="mtp") == expected
    with pytest.raises(ValueError, match="role"):
        metadata_config_from_checkpoint(checkpoint, path, role="other")


def test_inventory_handoff_rejects_unrelated_entry_without_filesystem_read(tmp_path, monkeypatch):
    source = tiny_model(tmp_path / "source.gguf")
    checkpoint = gguf_file.GGUFCheckpoint.open(source)
    unrelated = tmp_path / "not-opened.gguf"

    def forbidden(*args, **kwargs):
        pytest.fail("inventory identity must be checked lexically without new IO")

    monkeypatch.setattr(gguf_file.GGUFCheckpoint, "open", forbidden)
    from pathlib import Path

    with monkeypatch.context() as patch:
        patch.setattr(Path, "resolve", forbidden)
        patch.setattr(Path, "stat", forbidden)
        patch.setattr(Path, "open", forbidden)
        with pytest.raises(ValueError, match="entrypoint.*validated inventory"):
            metadata_config_from_checkpoint(checkpoint, unrelated)
