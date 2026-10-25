"""Pinned original-source wire oracles for the SC117 mixed GGUF encodings."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gguf_fixture import write
from tensorfold.families.qwen4_exp.gguf import FORMATS, GGUFCheckpoint, PackedTensor, decode_rows

KINDS = (14, 18, 21, 22, 23, 42)
FIXTURE = Path(__file__).parent / "fixtures/gguf_mixed_ggml.npz"


@pytest.fixture
def wire():
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        yield fixture


@pytest.mark.parametrize("kind", KINDS)
def test_original_ggml_dequantizers_match_all_fp32_bits(kind, wire):
    raw, bits = wire[f"raw_{kind}"], wire[f"bits_{kind}"]
    source = PackedTensor("wire", bits.shape, kind, Path("unused"), 0, raw.nbytes)
    np.testing.assert_array_equal(decode_rows(source, raw).view(np.uint32), bits)
    # Different row boundaries and noncontiguous transport retain source bits.
    for row in (0, 3, 7):
        np.testing.assert_array_equal(decode_rows(source, raw[row:row + 1]).view(np.uint32), bits[row:row + 1])
    reversed_raw = raw[::-1]
    np.testing.assert_array_equal(decode_rows(source, reversed_raw).view(np.uint32), bits[::-1])
    np.testing.assert_array_equal(decode_rows(source, raw[:0]).view(np.uint32), bits[:0])


@pytest.mark.parametrize("kind", KINDS)
def test_new_formats_are_inventoried_and_bounded_before_decode(kind, wire, tmp_path):
    raw, bits = wire[f"raw_{kind}"], wire[f"bits_{kind}"]
    path = write(tmp_path / "wire.gguf", [("wire", bits.shape, kind, raw)],
                 {"general.architecture": "qwen4exp"})
    tensor = GGUFCheckpoint.open(path).tensors["wire"]
    assert tensor.row_bytes == raw.shape[1] and tensor.nbytes == raw.nbytes
    with path.open("rb") as stream:
        stream.seek(tensor.offset)
        saved = np.frombuffer(stream.read(tensor.nbytes), np.uint8).reshape(raw.shape)
    np.testing.assert_array_equal(decode_rows(tensor, saved).view(np.uint32), bits)
    with pytest.raises(ValueError, match="exact packed row bytes"):
        decode_rows(tensor, raw[:, :-1])
    with pytest.raises(ValueError, match="exact packed row bytes"):
        decode_rows(tensor, raw.astype(np.int8))
    with pytest.raises(ValueError, match="complete"):
        tensor.column_span(16, 80)
    with pytest.raises(ValueError, match="staging budget"):
        tensor.read_rows(None, 0, 8, max_bytes=raw.nbytes - 1)
    with pytest.raises(ValueError, match="outside"):
        tensor.row_span(7, 2)
    # All source blocks are read unchanged, including a tensor beyond4GiB.
    large = PackedTensor("large", (2**32 + 9, bits.shape[1]), kind, path, 2**32 + 32,
                         (2**32 + 9) * raw.shape[1])
    assert large.row_span(2**32, 3) == (2**32 + 32 + 2**32 * raw.shape[1], 3 * raw.shape[1])


def test_fixture_provenance_and_complete_sc117_type_set(wire):
    provenance = json.loads(str(wire["provenance"]))
    assert provenance["revision"] == "b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9"
    assert provenance["source_sha256"]["ggml-quants.c"] == (
        "5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab")
    assert {"BF16", "F32", "F16", "Q2_0", "Q4_K", "Q5_K", "Q6_K", "Q8_0",
            "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS"} <= {v[0] for v in FORMATS.values()}


def test_generated_device_codebooks_are_current():
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "tools/generate_gguf_decode_tables.py"), "--check"], check=True)
