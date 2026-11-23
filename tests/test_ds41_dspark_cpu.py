"""CPU checks of the DeepSeek-V4.1 DSpark head loader pieces (decode tables, layout, index reader, verification)."""

import json
import math
import struct

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import mxfp4_ds41 as mx
from tensorfold.families.deepseek_v4.cuda import dspark_weights as dw


def e4m3fn(code):
    sign = -1.0 if code & 0x80 else 1.0
    e, m = (code >> 3) & 15, code & 7
    return sign * (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


def test_fp8_block_decode_every_code_and_scale():
    valid = np.array([c for c in range(256) if (c & 0x7F) != 0x7F], dtype=np.uint8)
    codes = np.resize(valid, (32, 64))                           # every finite code, in two 32x32 blocks
    scales = np.array([[0, 140]], dtype=np.uint8)
    got = dw.fp8_block_decode(codes, scales)
    for (r, c), code in np.ndenumerate(codes):
        assert float(got[r, c]) == e4m3fn(int(code)) * 2.0 ** (int(scales[r // 32, c // 32]) - 127)
    with pytest.raises(ValueError):
        dw.fp8_block_decode(np.full((32, 32), 0x7F, dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8))


def test_audit_counts_fp16_effects():
    exact = torch.tensor([1.0, 2.0 ** -20, 2.0 ** -26, 70000.0, 1 + 2.0 ** -12, 0.0])
    _, a = dw.audit_fp16("t", "F32", exact)
    assert (a.overflow, a.flushed, a.subnormal, a.nonzero) == (1, 1, 1, 5)
    assert a.inexact == 3          # overflow, flush and the 1 + 2^-12 rounding


def test_mxfp4_decode_table():
    codes = torch.arange(256, dtype=torch.uint8).view(1, 16, 16).reshape(16, 16)
    scales = torch.full((16, 1), 127 - 3, dtype=torch.uint8)
    got = mx.decode(codes, scales, torch.float64)
    for r in range(16):
        for j in range(16):
            b = int(codes[r, j])
            assert got[r, 2 * j] == mx.E2M1[b & 15] / 8 and got[r, 2 * j + 1] == mx.E2M1[b >> 4] / 8


def test_layout_partitions_every_rank():
    seen_experts = {}
    for rank in range(8):
        lay = dw.Layout(rank, 8, 4)
        assert lay.heads == 8 and len(lay.experts) == 64 and len(lay.expert_rows) == 576
        for e in lay.experts:
            seen_experts.setdefault(e, set()).add((lay.expert_rows.start, lay.expert_rows.stop))
    assert sorted(seen_experts) == list(range(128))
    assert all(len(v) == 4 and sorted(v)[0][0] == 0 and sorted(v)[-1][1] == 2304 for v in seen_experts.values())
    one = dw.Layout()
    assert one.heads == 64 and len(one.experts) == 128 and len(one.expert_rows) == 2304
    with pytest.raises(ValueError):
        dw.Layout(0, 8, 1)


def test_checkpoint_reader_index(tmp_path):
    tensors = {"mtp.0.a": np.arange(6, dtype=np.float32).reshape(2, 3), "mtp.0.b": np.array([1, 2, 3], dtype=np.uint8)}
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": "F32" if arr.dtype == np.float32 else "U8", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    body = json.dumps(header).encode()
    (tmp_path / "x.safetensors").write_bytes(struct.pack("<Q", len(body)) + body + b"".join(blobs))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"mtp.0.a": "x.safetensors", "mtp.0.b": "x.safetensors", "embed": "missing.safetensors"}}))
    ck = dw.Checkpoint(tmp_path)
    assert np.array_equal(ck.array("mtp.0.a"), tensors["mtp.0.a"]) and ck.shape("mtp.0.b") == (3,)
    assert math.prod(ck.shape("mtp.0.a")) == 6


def test_greedy_accept():
    from tensorfold.families.deepseek_v4.cuda.dspark_head import greedy_accept
    drafts = torch.tensor([[1, 2, 3, 4, 5], [1, 9, 3, 4, 5], [7, 2, 3, 4, 5]])
    target = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    assert greedy_accept(drafts, target).tolist() == [5, 1, 0]
