"""CPU packing oracle and range-only checkpoint loads for affine expert spill."""

import json
import struct

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

# Optional Torch must be gated before importing the CUDA package's CPU helpers.
from tensorfold.cuda.host_experts import load_host_experts, pack_cpu  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.reader import _Reader  # noqa: E402


def triple(e=2, n=64, k=128, gs=32, seed=19):
    rng = np.random.default_rng(seed)
    words = torch.from_numpy(rng.integers(0, 2**32, (e, n, k // 8), dtype=np.uint32).view(np.int32))
    raw = []
    for _ in range(2):
        bits = rng.integers(0, 2**16, (e, n, k // gs), dtype=np.uint16)
        bits.reshape(-1)[:8] = (0, 0x8000, 1, 0x7F80, 0xFF80, 0x7FC1, 0xFFFF, 0x3F80)
        raw.append(torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16))
    return words, *raw


def reference_pack(words, scales, biases, gs):
    """Scalar destination-to-source oracle; nibble extraction avoids bit tricks."""

    e, n, k8 = words.shape
    h, nb, kg = gs // 32, n // 32, k8 * 8 // gs
    weight_words = 128 * h
    out = np.empty((e, nb, kg, weight_words + 32), dtype=np.uint32)
    bits = words.numpy().view(np.uint32)
    sb = [t.view(torch.int16).numpy().view(np.uint16) for t in (scales, biases)]
    order = (0, 2, 4, 6, 1, 3, 5, 7)
    for expert in range(e):
        for block in range(nb):
            for group in range(kg):
                for destination in range(weight_words):
                    frame, lane, quad = destination // 128, destination % 128 // 4, destination % 4
                    flat = frame * 128 + quad * 32 + lane
                    q, r, tj = flat % 4, flat // 4 % 8, flat // 32
                    row, kk = (tj // h) * 8 + r, q * h + tj % h
                    value = int(bits[expert, block * 32 + row, group * 4 * h + kk])
                    out[expert, block, group, destination] = sum(
                        ((value >> (4 * source)) & 15) << (4 * target) for target, source in enumerate(order)
                    )
                for p in range(4):
                    for kind in range(2):
                        for t in range(4):
                            row = block * 32 + t * 8 + p * 2
                            out[expert, block, group, weight_words + p * 8 + kind * 4 + t] = int(
                                sb[kind][expert, row, group]
                            ) | (int(sb[kind][expert, row + 1, group]) << 16)
    return torch.from_numpy(out.view(np.int32))


@pytest.mark.parametrize("gs", [32, 64])
@pytest.mark.parametrize("shape", [(1, 32, 64), (3, 64, 128)])
@pytest.mark.parametrize("unsigned", [False, True])
def test_pack_matches_scalar_oracle_and_preserves_inputs(gs, shape, unsigned):
    source = triple(*shape, gs)
    if unsigned:
        source = (source[0].view(torch.uint32), *source[1:])
    before = [t.view(torch.int32 if i == 0 else torch.int16).clone() for i, t in enumerate(source)]
    actual = pack_cpu(*source, gs)
    assert actual.dtype == torch.int32 and actual.is_contiguous() and actual.device.type == "cpu"
    assert torch.equal(actual, reference_pack(*source, gs))
    for i, tensor in enumerate(source):
        assert torch.equal(tensor.view(torch.int32 if i == 0 else torch.int16), before[i])


def test_output_reuse_and_input_overlap_rejection():
    source = triple(1, 32, 32)
    out = torch.empty((1, 1, 1, 160), dtype=torch.int32)
    assert pack_cpu(*source, 32, out=out) is out
    assert torch.equal(out, reference_pack(*source, 32))
    aliased = out.reshape(-1)[:128].reshape(1, 32, 4)
    with pytest.raises(ValueError, match="overlap"):
        pack_cpu(aliased, *source[1:], 32, out=out)


@pytest.mark.parametrize("gs", [0, 16, 128, True, 32.0])
def test_group_boundary(gs):
    with pytest.raises(ValueError, match="group size"):
        pack_cpu(*triple(1, 32, 64), gs)


def test_invalid_tensor_boundaries():
    words, scales, biases = triple(1, 32, 64)
    with pytest.raises(ValueError, match="int32/uint32"):
        pack_cpu(words.float(), scales, biases, 32)
    with pytest.raises(ValueError, match="BF16"):
        pack_cpu(words, scales.half(), biases, 32)
    with pytest.raises(ValueError, match="BF16"):
        pack_cpu(words, scales[..., :1].contiguous(), biases, 32)
    with pytest.raises(ValueError, match="contiguous"):
        pack_cpu(words.transpose(1, 2), scales, biases, 32)
    with pytest.raises(ValueError, match="positive"):
        pack_cpu(words[:0], scales[:0], biases[:0], 32)
    with pytest.raises(ValueError, match="output"):
        pack_cpu(words, scales, biases, 32, out=torch.empty((1, 1, 1, 160), dtype=torch.int32))


class RecordingReader:
    def __init__(self, tensors):
        self.tensors = tensors
        self.reads = []

    def info(self, name):
        tensor = self.tensors[name]
        return {"shape": tuple(tensor.shape), "dtype": "U32" if tensor.dtype == torch.int32 else "BF16"}

    def get_rows(self, name, lo, hi, device="cpu"):
        assert device == "cpu"
        self.reads.append((name, lo, hi))
        return self.tensors[name][lo:hi].clone()


def checkpoint_tensors(base="model.layers.0.mlp", e=3, width=32, dims=64):
    tensors = {}
    for index, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
        n, k = (dims, width) if index == 2 else (width, dims)
        for kind, count in (("switch_mlp", e), ("shared_expert", 1)):
            source = triple(count, n, k, seed=31 + index + count)
            for component, tensor in zip(("weight", "scales", "biases"), source):
                tensors[f"{base}.{kind}.{projection}.{component}"] = tensor if kind == "switch_mlp" else tensor[0]
    return tensors


@pytest.mark.parametrize("chunk", [1, 2, 8])
@pytest.mark.parametrize("base", ["model.layers.0.mlp", "language_model.mtp.layers.0.mlp"])
def test_load_host_experts_keeps_native_layout_shared_id_and_chunk_bound(chunk, base):
    tensors = checkpoint_tensors(base)
    reader = RecordingReader(tensors)
    host = load_host_experts(reader, base, chunk_experts=chunk)
    expected = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        source = [
            torch.cat(
                (
                    tensors[f"{base}.switch_mlp.{projection}.{part}"],
                    tensors[f"{base}.shared_expert.{projection}.{part}"][None],
                )
            )
            for part in ("weight", "scales", "biases")
        ]
        expected.append(reference_pack(*source, 32))
    assert torch.equal(host.up, torch.stack(expected[:2], dim=3))
    assert torch.equal(host.down, expected[2].unsqueeze(3))
    assert (host.gs, host.width, host.dims, host.count, host.swiglu, host.limit) == (32, 32, 64, 4, True, 0.0)
    assert host.up.is_contiguous() and host.down.is_contiguous()
    assert host.up.device.type == host.down.device.type == "cpu"
    assert not host.up.is_pinned() and not host.down.is_pinned()
    assert all(hi - lo <= chunk for name, lo, hi in reader.reads if ".switch_mlp." in name)


def test_invalid_model_shape_fails_before_payload_reads():
    tensors = checkpoint_tensors()
    tensors["model.layers.0.mlp.shared_expert.up_proj.weight"] = tensors[
        "model.layers.0.mlp.shared_expert.up_proj.weight"
    ][:16]
    reader = RecordingReader(tensors)
    with pytest.raises(ValueError, match="shapes|widths"):
        load_host_experts(reader, "model.layers.0.mlp")
    assert reader.reads == []
    with pytest.raises(ValueError, match="positive"):
        load_host_experts(RecordingReader(checkpoint_tensors()), "model.layers.0.mlp", chunk_experts=True)


@pytest.mark.parametrize("size", [0, -1, True, 1.0])
def test_invalid_metadata_dimension_fails_before_payload_read(size):
    reader = RecordingReader(checkpoint_tensors())
    actual_info = reader.info

    def info(name):
        result = actual_info(name)
        if name == "model.layers.0.mlp.switch_mlp.gate_proj.weight":
            result["shape"] = (size, *result["shape"][1:])
        return result

    reader.info = info
    with pytest.raises(ValueError, match="positive integers"):
        load_host_experts(reader, "model.layers.0.mlp")
    assert reader.reads == []


def write_checkpoint(tmp_path, tensors):
    header, payload, offset = {}, [], 0
    dtypes = {torch.int32: "U32", torch.bfloat16: "BF16"}
    for name, tensor in tensors.items():
        data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": dtypes[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        payload.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(payload))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}})
    )


def test_range_reader_reads_only_requested_first_axis_bytes(tmp_path):
    words, scales, _ = triple(3, 32, 64)
    write_checkpoint(tmp_path, {"words": words, "scales": scales})
    reader = _Reader(tmp_path, "cpu")
    requested = []
    actual_read = reader.io.read

    def record(path, offset, size, device):
        requested.append((offset, size, device))
        return actual_read(path, offset, size, device)

    reader.io.read = record
    try:
        info = reader.info("words")
        got = reader.get_rows("words", 1, 2)
        assert torch.equal(got, words[1:2])
        row_bytes = words[0].numel() * words.element_size()
        assert requested == [(info["byte_offset"] + row_bytes, row_bytes, "cpu")]
        assert torch.equal(reader.get_rows("scales", 2, 3).view(torch.int16), scales[2:3].view(torch.int16))
        before = len(requested)
        assert reader.get_rows("words", 1, 1).shape == (0, 32, 8)
        for lo, hi in [(-1, 1), (0, 4), (2, 1), (True, 2), (0, 1.0)]:
            with pytest.raises(ValueError, match="rows"):
                reader.get_rows("words", lo, hi)
        assert len(requested) == before
    finally:
        reader.close()


def test_range_reader_validates_metadata_before_allocation(tmp_path):
    words, _, _ = triple(1, 32, 64)
    write_checkpoint(tmp_path, {"words": words})
    reader = _Reader(tmp_path, "cpu")
    try:
        base, header = reader._header("model.safetensors")
        header["words"]["data_offsets"][1] += 4
        with pytest.raises(ValueError, match="byte range"):
            reader.get_rows("words", 0, 1)
        reader.headers.clear()
        (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", 2**63))
        with pytest.raises(ValueError, match="header"):
            reader.info("words")
    finally:
        reader.close()


def test_range_reader_preserves_owning_hf_blob_symlink_contract(tmp_path):
    model = tmp_path / "models--owner--checkpoint"
    snapshot = model / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (model / "blobs").mkdir()
    words, _, _ = triple(2, 32, 64)
    write_checkpoint(snapshot, {"words": words})
    shard = snapshot / "model.safetensors"
    target = model / "blobs" / "content-addressed-weights"
    shard.rename(target)
    shard.symlink_to("../../blobs/content-addressed-weights")
    reader = _Reader(snapshot, "cpu")
    try:
        assert reader.info("words")["path"] == target
        assert torch.equal(reader.get_rows("words", 1, 2), words[1:2])
        other = tmp_path / "models--other--checkpoint" / "blobs"
        other.mkdir(parents=True)
        outside = other / "unrelated-weights"
        outside.write_bytes(target.read_bytes())
        shard.unlink()
        shard.symlink_to(outside)
        with pytest.raises(ValueError, match="authorized roots"):
            reader.get_rows("words", 0, 1)
    finally:
        reader.close()


@pytest.mark.parametrize(
    "path", ["../outside.safetensors", "nested/../../outside.safetensors", "/absolute.safetensors"]
)
def test_range_reader_rejects_index_path_escape(tmp_path, path):
    words, _, _ = triple(1, 32, 64)
    write_checkpoint(tmp_path, {"words": words})
    reader = _Reader(tmp_path, "cpu")
    reader.where["words"] = path
    try:
        with pytest.raises(ValueError, match="absolute|parent directories"):
            reader.get_rows("words", 0, 1)
    finally:
        reader.close()


def test_range_reader_allows_nested_model_files(tmp_path):
    words, _, _ = triple(1, 32, 64)
    write_checkpoint(tmp_path, {"words": words})
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "model.safetensors").rename(nested / "weights.safetensors")
    reader = _Reader(tmp_path, "cpu")
    reader.where["words"] = "nested/weights.safetensors"
    try:
        assert torch.equal(reader.get_rows("words", 0, 1), words)
    finally:
        reader.close()


def test_range_reader_rejects_oversized_header_without_reading_it(tmp_path):
    words, _, _ = triple(1, 32, 64)
    write_checkpoint(tmp_path, {"words": words})
    with (tmp_path / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", 65 << 20))
        stream.truncate((65 << 20) + 8)
    reader = _Reader(tmp_path, "cpu")
    try:
        with pytest.raises(ValueError, match="invalid safetensors header"):
            reader.info("words")
    finally:
        reader.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", []),
        ("shape", [1, 32, 0]),
        ("shape", [True, 32, 8]),
        ("data_offsets", [False, 1024]),
        ("data_offsets", [-4, 1020]),
    ],
)
def test_range_reader_rejects_malformed_tensor_metadata(tmp_path, field, value):
    words, _, _ = triple(1, 32, 64)
    write_checkpoint(tmp_path, {"words": words})
    reader = _Reader(tmp_path, "cpu")
    try:
        _, header = reader._header("model.safetensors")
        header["words"][field] = value
        with pytest.raises(ValueError, match="dtype|shape|byte range"):
            reader.get_rows("words", 0, 1)
    finally:
        reader.close()


@pytest.mark.parametrize("advisory_error", [None, OSError("advice unsupported"), AttributeError("no advice")])
def test_row_only_buffered_shard_is_advised_and_owned_fd_always_closed(tmp_path, monkeypatch, advisory_error):
    import os

    words, _, _ = triple(2, 32, 64)
    write_checkpoint(tmp_path, {"words": words})
    reader = _Reader(tmp_path, "cpu")
    reader.io.direct = False
    opened, advised, closed = [], [], []
    actual_open, actual_close = os.open, os.close
    def open_owned(path, flags):
        assert path == tmp_path / "model.safetensors" and flags == os.O_RDONLY
        fd = actual_open(path, flags)
        opened.append(fd)
        return fd
    def advise(fd, offset, length, advice):
        advised.append((fd, offset, length, advice))
        if advisory_error is not None:
            raise advisory_error
    def close_owned(fd):
        closed.append(fd)
        actual_close(fd)
    try:
        assert torch.equal(reader.get_rows("words", 1, 2), words[1:2])
        assert reader.touched == {"model.safetensors"} and reader.reads.ahead == {}
        monkeypatch.setattr(os, "open", open_owned)
        monkeypatch.setattr(os, "posix_fadvise", advise)
        monkeypatch.setattr(os, "close", close_owned)
        reader.release()
        assert len(opened) == 1 and closed == opened
        assert advised == [(opened[0], 0, 0, os.POSIX_FADV_DONTNEED)]
        assert not reader.touched
        with pytest.raises(OSError):
            os.fstat(opened[0])                      # real descriptor was closed, including on advisory failure
    finally:
        reader.close()


def test_reader_release_reports_close_failure_without_retrying_owned_fd(tmp_path, monkeypatch):
    import os

    words, _, _ = triple(1, 32, 64)
    write_checkpoint(tmp_path, {"words": words})
    reader = _Reader(tmp_path, "cpu")
    reader.io.direct = False
    opened, closed = [], []
    actual_open, actual_close = os.open, os.close
    def open_owned(path, flags):
        fd = actual_open(path, flags)
        opened.append(fd)
        return fd
    def advise(*args):
        raise OSError("optional advice failed")
    def close_owned(fd):
        closed.append(fd)
        actual_close(fd)
        raise OSError("owned descriptor close failed")
    try:
        reader.get_rows("words", 0, 1)
        monkeypatch.setattr(os, "open", open_owned)
        monkeypatch.setattr(os, "posix_fadvise", advise)
        monkeypatch.setattr(os, "close", close_owned)
        with pytest.raises(OSError, match="owned descriptor close failed"):
            reader.release()
        assert len(opened) == 1 and closed == opened
        with pytest.raises(OSError):
            os.fstat(opened[0])
    finally:
        reader.close()
