"""Generic packed/shared-fusion current INT32/513/top11 safety and empirical error.

No tolerance or quality acceptance is inferred from these safety controls.
"""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from gguf_fixture import blocks
from test_sm70_gguf_exact import bits
from tensorfold.cuda.sm70 import gguf_ops
from tensorfold.cuda.sm70.execution_policy import _family_runtime
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFPacked, GGUFExperts, GGUFDense
from tensorfold.families.qwen4_exp.gguf import FORMATS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual pinned SM70 required")


def error_record(actual, reference):
    actual, reference = actual.cpu().double(), reference.cpu().double()
    assert torch.isfinite(actual).all().item() and torch.isfinite(reference).all().item()
    delta = actual - reference
    norm = float(torch.linalg.vector_norm(reference))
    return {"max_abs": float(delta.abs().max()), "rms": float(delta.square().mean().sqrt()),
            "relative_l2": float(torch.linalg.vector_norm(delta)) / norm if norm else None,
            "half_bin_mismatches": int((actual.half().view(torch.int16) != reference.half().view(torch.int16)).sum()),
            "values": actual.numel()}


def raw_packed(x, picks, gate, up, down, shared, out):
    return gguf_ops.experts(x, picks, gate, up, down, shared, out, blockwise=True, projection_warps=2)


def raw_rows(kind, rows, width, experts, per_expert, seed):
    block, size = FORMATS[kind][1:]
    if kind in (18, 21, 22, 23, 42):
        with np.load(Path(__file__).parents[1] / "fixtures/gguf_mixed_ggml.npz", allow_pickle=False) as wire:
            source = wire[f"raw_{kind}"]
        repeats = -(-width // (source.shape[1] // size * block))
        raw = np.ascontiguousarray(np.tile(source[(np.arange(rows) + seed) % len(source)], (1, repeats))[:, :width // block * size])
    else:
        raw = blocks(kind, rows, width, seed=seed)
    # Distinct expert outer scales prevent a wrong low/high expert address from
    # accidentally passing because the preserved wire patterns repeat.
    scales = (.001 + np.repeat(np.arange(experts), per_expert) * .000001).astype(np.float16)
    raw.reshape(rows, width // block, size)[:, :, :2] = scales.view(np.uint8).reshape(rows, 1, 2)
    return raw


def bank(gate_kind, down_kind, offset=48):
    width, hidden, experts = 80, 2560, 512
    physical = -(-(offset + width) // FORMATS[down_kind][1]) * FORMATS[down_kind][1]
    raw = [raw_rows(gate_kind, experts * width, hidden, experts, width, 31),
           raw_rows(gate_kind, experts * width, hidden, experts, width, 32),
           raw_rows(down_kind, experts * hidden, physical, experts, hidden, 33)]
    packed = [GGUFPacked(torch.from_numpy(value).to("cuda"), kind, value.shape[0], columns, crop)
              for value, kind, columns, crop in zip(raw, (gate_kind, gate_kind, down_kind),
                                                   (hidden, hidden, width), (0, 0, offset), strict=True)]
    shared = [GGUFDense(torch.randn(shape, device="cuda", dtype=torch.float16) * .002)
              for shape in ((width, hidden), (width, hidden), (hidden, width))]
    for value in shared:
        value.implementation = "native32"
    value = GGUFExperts(*packed, shared, count=experts, width=width, dims=hidden)
    assert value.count == experts + 1 == 513
    return value


@pytest.mark.parametrize("kinds", [(18, 20), (21, 42), (22, 20), (23, 42), (8, 8)])
def test_actual_domain_int32_fp32_row_isolation_and_error_records(kinds, record_property):
    _family_runtime(torch)
    torch.manual_seed(13001)
    value = bank(*kinds)
    x = torch.randn((32, 2560), device="cuda", dtype=torch.float16) * .001
    picks = torch.arange(32 * 11, device="cuda", dtype=torch.int32).reshape(32, 11) % 512
    picks[:, -1] = 512
    picks[0, 0] = 511
    out = torch.empty((32, 11, 2560), device="cuda", dtype=torch.float32)
    assert value.forward_blocks2w(x, picks, out) is out
    original = value.forward_reference(x, picks)
    record_property("shared_fusion_current_error", json.dumps({"kinds": kinds, "rows": 32,
        "routed_experts": 512, "slots": 11, "picks": "int32", "output": "float32",
        "original_bmm": error_record(out, original), "quality": "unqualified, empirical only"}))
    serial = torch.cat([value.forward_blocks2w(x[i:i+1], picks[i:i+1]) for i in range(32)])
    bits(out, serial)
    half = torch.empty_like(out, dtype=torch.float16)
    assert value.forward_blocks2w(x, picks, half) is half
    bits(half, out.half())
    assert torch.isfinite(half).all().item()
    # Shared math is intentionally fused and may differ from Dense reduction.
    # Report it, never impose an invented approximation tolerance.
    for rows in (1, 3, 30, 31):
        bits(value.forward_blocks2w(x[:rows], picks[:rows]), out[:rows])


def test_current_capture_replay_and_invalid_alias_bounds():
    _family_runtime(torch)
    torch.manual_seed(13002)
    value = bank(8, 8, offset=16)
    x = torch.randn((32, 2560), device="cuda", dtype=torch.float16) * .001
    picks = torch.arange(32 * 11, device="cuda", dtype=torch.int32).reshape(32, 11) % 512
    picks[:, -1] = 512
    value.forward_blocks2w(x, picks)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = value.forward_blocks2w(x, picks)
    x.mul_(.5)
    picks[:, 0] = 511
    graph.replay()
    bits(result, value.forward_blocks2w(x, picks))
    bad = picks.clone()
    bad[0, 0], bad[0, 1] = -1, 2147483647
    core = torch.empty_like(result)
    raw_packed(x, bad, value.gate, value.up, value.down, value.shared, core)
    assert torch.isnan(core[0, :2]).all().item()
    bits(core[:, -1], value.forward_blocks2w(x, picks)[:, -1])
    # The public selector keeps large rows on reference. The existing raw
    # generic API instead bounds pair-count65535, not WMMA's separateR32 cap.
    value.implementation = "packed_blocks2w"
    full, routes = x[:1].repeat(33, 1), picks[:1].repeat(33, 1)
    bits(value.forward(full, routes), value.forward_reference(full, routes))
    with pytest.raises(RuntimeError, match="slot geometry"):
        raw_packed(x[:1], torch.zeros((1, 65536), device="cuda", dtype=torch.int32),
            value.gate, value.up, value.down, value.shared, torch.empty(0, device="cuda"))
    with pytest.raises(RuntimeError, match="overlap"):
        raw_packed(x, picks[:, :1].contiguous(), value.gate, value.up, value.down, value.shared, x[:, None])
