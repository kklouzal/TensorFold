"""Preparation copies only MTP ranges, resumes complete tensors, and checks output."""

import importlib.util
import json
from pathlib import Path
import struct

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tensorfold.cuda.tensor_file import parse_header_bytes, read_header  # noqa: E402

spec = importlib.util.spec_from_file_location("v100_mtp_prepare", Path(__file__).parents[1] / "tools/v100_mtp_prepare.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": {
        "mtp": {"num_hidden_layers": 1, "layer_types": ["full_attention"]}}}))
    tensors = {f"mtp.tensor{i:02}.weight": torch.full((2, 4), i / 32, dtype=torch.bfloat16) for i in range(31)}
    tensors["model.unrelated.weight"] = torch.ones((32, 32), dtype=torch.bfloat16)
    save_file(tensors, str(root / "part-1.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "part-1.safetensors" for name in tensors}}))
    return root, {name: value for name, value in tensors.items() if name.startswith("mtp.")}


class RecordingSource(tool.Source):
    def __init__(self, directory, *, fail_offset=None):
        super().__init__(directory=directory)
        self.reads = []
        self.fail_offset = fail_offset

    def read(self, name, start, count, **kwargs):
        if name.endswith("safetensors") and start == self.fail_offset:
            raise OSError("controlled interrupted tensor read")
        self.reads.append((name, start, count))
        return super().read(name, start, count, **kwargs)


def test_import_bytes_and_reuse_do_not_read_unrelated_trunk(tmp_path):
    root, expected = source(tmp_path)
    reader = RecordingSource(root)
    out = tmp_path / "import"
    receipt = tool.prepare(reader, out)
    assert receipt["plan"]["tensor_count"] == 31
    assert receipt["plan"]["payload_bytes"] == 31 * 16
    assert receipt["execution_qualification"] == "unrun"
    assert (out / "config.json").read_bytes() == (root / "config.json").read_bytes()
    with safe_open(out / "model.safetensors", framework="pt") as imported:
        assert set(imported.keys()) == set(expected)
        for name, value in expected.items():
            torch.testing.assert_close(imported.get_tensor(name), value, atol=0, rtol=0)
    base, header = read_header(root / "part-1.safetensors", tool.SIZES)
    lo, hi = header["model.unrelated.weight"]["data_offsets"]
    for name, start, count in reader.reads:
        if name == "part-1.safetensors":
            assert start + count <= base + lo or start >= base + hi
            assert count <= tool.METADATA_LIMIT
    reader = RecordingSource(root)
    assert tool.prepare(reader, out) == receipt
    assert all(start < base for name, start, _ in reader.reads if name.endswith("safetensors"))


def test_interrupted_import_resumes_completed_tensors_without_early_publication(tmp_path):
    root, _ = source(tmp_path)
    base, header = read_header(root / "part-1.safetensors", tool.SIZES)
    second = base + header["mtp.tensor01.weight"]["data_offsets"][0]
    out = tmp_path / "import"
    with pytest.raises(OSError, match="controlled interrupted"):
        tool.prepare(RecordingSource(root, fail_offset=second), out)
    assert not out.exists()
    assert len(list((tmp_path / "import.partial").glob("*.tensor"))) == 1
    reader = RecordingSource(root)
    tool.prepare(reader, out)
    first = base + header["mtp.tensor00.weight"]["data_offsets"][0]
    assert all(start != first for name, start, _ in reader.reads if name.endswith("safetensors"))
    assert not (tmp_path / "import.partial").exists()


@pytest.mark.parametrize("changed", ["config.json", "model.safetensors"])
def test_existing_output_must_keep_both_configuration_and_tensor_identity(tmp_path, changed):
    root, _ = source(tmp_path)
    out = tmp_path / "import"
    tool.prepare(RecordingSource(root), out)
    with (out / changed).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="immutable source plan"):
        tool.prepare(RecordingSource(root), out)


def test_bounded_range_header_uses_same_schema_and_extents_as_local_reader(tmp_path):
    root, _ = source(tmp_path)
    file = root / "part-1.safetensors"
    with file.open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        raw = stream.read(count)
    expected = read_header(file, tool.SIZES)
    assert parse_header_bytes(raw, file.stat().st_size, tool.SIZES, label="range-only") == expected
    with pytest.raises((ValueError, IOError)):
        parse_header_bytes(raw, file.stat().st_size - 1, tool.SIZES, label="truncated range source")
    with pytest.raises(ValueError, match="bytes"):
        parse_header_bytes(None, file.stat().st_size, tool.SIZES, label="invalid")


@pytest.mark.parametrize("name", ["../escape.safetensors", "/absolute.safetensors", "bad?query", "bad\\path"])
def test_index_path_refused_before_preparation_writes(tmp_path, name):
    root, _ = source(tmp_path)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"mtp.x": name}}))
    with pytest.raises(ValueError, match="relative checkpoint"):
        tool.prepare(RecordingSource(root), tmp_path / "import")
    assert not (tmp_path / "import.partial").exists()


def test_remote_preparation_requires_an_immutable_commit():
    with pytest.raises(ValueError, match="immutable"):
        tool.Source(repository="Qwen/Qwen3.8-Flash-Next", revision="main")


def test_trickled_body_cannot_extend_cooperative_deadline(tmp_path, monkeypatch):
    root, _ = source(tmp_path)
    reader = RecordingSource(root)
    now = [0.0]
    reader.deadline = 2.0
    monkeypatch.setattr(tool.time, "monotonic", lambda: now[0])

    class Trickle:
        calls = 0
        headers = {}

        def read1(self, count):
            self.calls += 1
            now[0] += 0.75
            return b"x"

    body = Trickle()
    with pytest.raises(TimeoutError, match="deadline"):
        reader._body(body, 1024, exact=True)
    assert body.calls == 3


def test_transfer_encoding_rejected_before_an_unbounded_chunk_parser(tmp_path):
    from types import SimpleNamespace

    root, _ = source(tmp_path)
    response = SimpleNamespace(headers={"Transfer-Encoding": "chunked"},
                               read1=lambda _count: pytest.fail("chunk parser was entered"))
    with pytest.raises(ValueError, match="transfer-encoded"):
        RecordingSource(root)._body(response, 1024, exact=True)


def test_persisted_resume_metadata_is_size_bounded(tmp_path):
    root, _ = source(tmp_path)
    work = tmp_path / "import.partial"
    work.mkdir()
    with (work / "plan.json").open("wb") as stream:
        stream.truncate(tool.MAX_HEADER_BYTES + 1)
    with pytest.raises(ValueError, match="metadata size"):
        tool.prepare(RecordingSource(root), tmp_path / "import")


def test_retry_finishes_publication_barrier_after_rename_succeeded(tmp_path, monkeypatch):
    root, _ = source(tmp_path)
    out = tmp_path / "import"
    original = tool._sync_directory
    calls, failed = [], [False]

    def interrupt_parent(path):
        calls.append(path)
        if path == tmp_path and not failed[0]:
            failed[0] = True
            raise OSError("controlled parent publication barrier failure")
        original(path)

    monkeypatch.setattr(tool, "_sync_directory", interrupt_parent)
    with pytest.raises(OSError, match="publication barrier"):
        tool.prepare(RecordingSource(root), out)
    assert out.exists()
    calls.clear()
    reader = RecordingSource(root)
    tool.prepare(reader, out)
    assert calls == [out, tmp_path]


@pytest.mark.parametrize("mutation", ["extra", "bool_bytes", "wrong_count", "invalid_digest", "oversized"])
def test_completed_tensor_receipts_have_a_cumulative_bounded_canonical_schema(tmp_path, mutation):
    root, _ = source(tmp_path)
    base, header = read_header(root / "part-1.safetensors", tool.SIZES)
    second = base + header["mtp.tensor01.weight"]["data_offsets"][0]
    out = tmp_path / "import"
    with pytest.raises(OSError):
        tool.prepare(RecordingSource(root, fail_offset=second), out)
    record = next(path for path in (tmp_path / "import.partial").glob("*.json") if path.name != "plan.json")
    value = json.loads(record.read_text())
    if mutation == "extra":
        value["unexpected"] = "unretained"
    elif mutation == "bool_bytes":
        value["bytes"] = True
    elif mutation == "wrong_count":
        value["bytes"] += 1
    elif mutation == "invalid_digest":
        value["sha256"] = "invalid"
    else:
        value["unexpected"] = "x" * 4096
    record.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="receipt|metadata size"):
        tool.prepare(RecordingSource(root), out)
    assert not out.exists()
