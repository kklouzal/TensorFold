"""FP8 n-gram checkpoint admission, byte-exact CUDA staging and tiny complete-model equivalence.

The independent reference stores torch's FP8->fp32 scale product rounded once
into BF16. It never uses the production n-gram decoder or its rounding LUT.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import struct

import numpy as np
import pytest

pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from tests.cuda.nvfp4_tiny import write  # noqa: E402

from tensorfold.cleanup import raise_failures  # noqa: E402
from tensorfold.families.qwen4_exp import check  # noqa: E402
from tensorfold.families.qwen4_exp.host_table import BF16Table, FP8Table  # noqa: E402

# Binary non-power-of-two scale, with exact BF16 halfway cases and ordinary
# finite model-sized rows. 1.0 * this scale rounds to the even BF16 neighbour.
TABLE_SCALE = np.float32(1.00390625 / 32768)
PROMPT = [5, 17, 99, 250, 0, 64, 30, 11, 12, 13, 3, 200]


def _entries(directory: Path):
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    assert len(shards) == 1, "the immutable tiny fixture changed its one-shard contract"
    raw = (directory / shards.pop()).read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    return {name: (info["dtype"], list(info["shape"]), raw[8 + size + info["data_offsets"][0]:
                                                      8 + size + info["data_offsets"][1]])
            for name, info in header.items() if name != "__metadata__"}


def _write(directory: Path, config: dict, entries: dict):
    directory.mkdir()
    header, pieces, offset = {}, [], 0
    for name, (dtype, shape, raw) in entries.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        pieces.append(raw)
        offset += len(raw)
    encoded = json.dumps({"__metadata__": {"format": "pt"}, **header}, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    shard = "model-00001-of-00001.safetensors"
    (directory / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(pieces))
    (directory / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": offset}, "weight_map": {name: shard for name in entries}}))
    (directory / "config.json").write_text(json.dumps(config))
    return directory


def _reference(codes: np.ndarray, scale: np.float32) -> torch.Tensor:
    # CPU torch's native numeric conversion is independent of FP8Table.e4m3/lut.
    return (torch.from_numpy(codes.copy()).view(torch.float8_e4m3fn).float()
            * torch.tensor(float(scale), dtype=torch.float32)).to(torch.bfloat16)


def _pair(root: Path, *, prefix: str = "", nested: bool = False, scale: np.float32 | None = TABLE_SCALE):
    template = write(root / "template", hidden=512, ple_bf16=True)
    entries = _entries(template)
    table = next(name for name in entries if ".ngram_embedding.shard_0.weight" in name)
    dims = entries[table][1]
    # Every finite e4m3 byte, including signed zero/subnormals/extremes, appears;
    # omitted NaN codes are covered separately by the maintained CPU oracle.
    finite = np.array([code for code in range(256) if code & 0x7F != 0x7F], dtype=np.uint8)
    codes = np.resize(finite, np.prod(dims)).reshape(dims)
    effective = np.float32(1.0 if scale is None else scale)
    rows = _reference(codes, effective)
    if scale is not None and scale == TABLE_SCALE:
        midpoint = rows.reshape(-1)[0x38]  # byte0x38 decodes to1.0; this product is a BF16 halfway case.
        assert midpoint.view(torch.int16) == torch.tensor(2.0**-15, dtype=torch.bfloat16).view(torch.int16)
    stem = table.removesuffix("shard_0.weight")
    if nested:
        table = stem + "shards.0.weight"
        entries.pop(stem + "shard_0.weight")
    fp8 = {**entries, table: ("F8_E4M3", dims, codes.tobytes())}
    bf16 = {**entries, table: ("BF16", dims, rows.view(torch.uint8).numpy().tobytes())}
    if scale is not None:
        fp8[stem + "weight_scale"] = ("F32", [1], np.asarray([scale], dtype=np.float32).tobytes())
    config = json.loads((template / "config.json").read_text())
    quantized = {name.removesuffix(".weight"): {"quant_algo": "NVFP4", "group_size": 16}
                 for name, (dtype, _, _) in entries.items()
                 if name.endswith(".weight") and ".mlp.experts." in name and dtype == "U8"}
    config["quantization"]["quant_algo"] = "MIXED_PRECISION"
    config["quantization"]["quantized_layers"] = quantized

    def rename(name):
        return prefix + name[len("model."):] if prefix and name.startswith("model.") else name

    def named(values):
        return {rename(name): value for name, value in values.items()}

    reference_config = json.loads(json.dumps(config))
    reference_config["quantization"]["quantized_layers"] = {rename(name): value for name, value in quantized.items()}
    config["quantization"]["quantized_layers"] = {
        **reference_config["quantization"]["quantized_layers"], rename(stem.rstrip(".")): {"quant_algo": "FP8"}}
    actual = _write(root / "fp8", config, named(fp8))
    reference = _write(root / "bf16", reference_config, named(bf16))
    return actual, reference, rows, rename(stem)


@contextmanager
def _loaded(directory: Path):
    from tensorfold.families.qwen4_exp.cuda.weights import load

    weights = load(directory, mtp=True, draft_vocab=128)
    primary = None
    try:
        yield weights
    except BaseException as failure:
        primary = failure
        raise
    finally:
        failures = []
        try:
            weights.meta["ple_tables"].close(lambda: torch.cuda.synchronize(weights.device))
        except BaseException as failure:
            failures.append(failure)
        raise_failures(primary, failures)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA loading and staging")
@pytest.mark.parametrize("prefix,nested", [("", False), ("model.language_model.", True)],
                         ids=["plain-flat", "published-nested"])
def test_fp8_table_admission_loading_and_staging_are_bf16_byte_exact(tmp_path, prefix, nested):
    from tensorfold.families.qwen4_exp.cuda import glue
    from tensorfold.families.qwen4_exp.cuda.forward import stage_ple_rows
    from tensorfold.families.qwen4_exp.cuda.state import Buffers

    actual, reference, rows, _ = _pair(tmp_path, prefix=prefix, nested=nested)
    check(actual)
    check(reference)
    with _loaded(actual) as weights:
        ple = next(layer.ple for layer in weights.layers if layer.ple is not None)
        assert type(ple.table) is FP8Table and ple.table.rows == rows.shape[0] and ple.table.width == rows.shape[1]
        assert getattr(ple.table, "weight_scale", 1.0) == 1.0, "the scale was already applied before BF16 rounding"
        ids = np.array([0, 1, 7, 31, rows.shape[0] - 1, 0, 17, 251], dtype=np.int64).reshape(4, ple.ngram.heads)
        expected = rows[torch.from_numpy(ids.reshape(-1))]
        assert np.array_equal(ple.table.gather(ids), expected.view(torch.uint16).numpy())
        buffers = Buffers(weights, rows=6, capacity=64)
        buffers.ple_v.fill_(-123.0)
        at, count = ple.ngram.heads, ids.size
        stage_ple_rows(ple, buffers, ids, at=at)
        torch.cuda.synchronize(weights.device)
        assert torch.equal(buffers.ple_v[at:at + count].cpu().view(torch.int16), expected.view(torch.int16))
        assert torch.all(buffers.ple_v[:at] == -123.0) and torch.all(buffers.ple_v[at + count:] == -123.0)
        glue.ple_embed_bf16(4, buffers.ple_v[at:at + count], ple.ngram.heads, ple.ngram.dims,
                            buffers.ple_emb[:4], buffers.xs_ple[:4], scale=getattr(ple.table, "weight_scale", 1.0))
        assert torch.equal(buffers.ple_emb[:4].cpu().view(torch.int16), expected.reshape(4, -1).view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA model execution")
@pytest.mark.parametrize("seed", [None, 7], ids=["greedy", "keyed"])
def test_fp8_table_tiny_logits_and_complete_generation_equal_converted_bf16(tmp_path, seed):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode

    actual, reference, _, _ = _pair(tmp_path, prefix="model.language_model.", nested=True)
    sampling = None if seed is None else Sampling(seed=seed, top_k=20, top_p=0.95)
    with _loaded(actual) as fp8, _loaded(reference) as bf16:
        assert type(next(layer.ple.table for layer in bf16.layers if layer.ple is not None)) is BF16Table
        engines = [Engine(weights, capacity=256, max_rows=8, prefill_rows=16, graphs=False) for weights in (fp8, bf16)]
        first = [prefill(engine, PROMPT, sampling) for engine in engines]
        assert first[0] == first[1]
        # Compare the full-vocabulary row, not only its winning token.
        logits = [engine.forward([pending]).detach().clone().cpu().view(torch.int16)
                  for engine, pending in zip(engines, first, strict=True)]
        assert torch.equal(*logits)
        results = []
        for engine in engines:
            start = prefill(engine, PROMPT, sampling)
            serial = serial_decode(engine, start, 16, sampling).tokens
            assert len(serial) == 16
            for depth, confidence in ((2, 0.0), (4, 0.3)):
                start = prefill(engine, PROMPT, sampling)
                assert mtp_decode(engine, start, 16, sampling, depth=depth, confidence=confidence).tokens == serial
            results.append(serial)
        assert results[0] == results[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA weights loader")
def test_fp8_table_without_a_scale_uses_declared_unit_scale(tmp_path):
    actual, _, rows, _ = _pair(tmp_path, scale=None)
    check(actual)
    with _loaded(actual) as weights:
        table = next(layer.ple.table for layer in weights.layers if layer.ple is not None)
        ids = np.array([0, rows.shape[0] - 1, 0], dtype=np.int64)
        assert np.array_equal(table.gather(ids), rows[torch.from_numpy(ids)].view(torch.uint16).numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA weights loader")
def test_fp8_table_non_scalar_scale_is_rejected_before_row_consumption(tmp_path):
    actual, _, _, stem = _pair(tmp_path)
    entries = _entries(actual)
    entries[stem + "weight_scale"] = ("F32", [2], np.asarray([1.0, 2.0], dtype=np.float32).tobytes())
    config = json.loads((actual / "config.json").read_text())
    malformed = _write(tmp_path / "non-scalar", config, entries)
    check(malformed)
    from tensorfold.families.qwen4_exp.cuda.weights import load

    with pytest.raises(ValueError, match="one n-gram table scale"):
        load(malformed, mtp=True, draft_vocab=128)


def test_fp8_metadata_permission_is_limited_to_ngram_tables(tmp_path):
    actual, _, _, _ = _pair(tmp_path)
    config = json.loads((actual / "config.json").read_text())
    config["quantization"]["quantized_layers"]["model.layers.0.linear_attn.in_proj_qkv"] = {"quant_algo": "FP8"}
    (actual / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Flash Next kernels read NVFP4"):
        check(actual)
