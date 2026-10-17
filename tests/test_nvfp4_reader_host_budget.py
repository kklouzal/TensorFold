"""Independent physical Reader backing and nonduplicated NVFP4 host admission."""

import ast
import math
from pathlib import Path

import pytest

from tensorfold.cuda import capacity
from tensorfold.cuda.read_budget import READER_HOST_RING_BYTES, reader_host_bytes
from tensorfold.families.qwen3_5.cuda import nvfp4_memory
from tensorfold.families.qwen4_exp import ram_experts


def test_physical_reader_bound_matches_existing_owned_allocation_geometry():
    path = Path(__file__).parents[1] / "src/tensorfold/cuda/direct_read.py"
    tree = ast.parse(path.read_text())
    values = {node.targets[0].id: node.value for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("PIECE", "ALIGN")}
    assert ast.unparse(values["PIECE"]) == "64 << 20"
    assert ast.literal_eval(values["ALIGN"]) == 4096
    reader = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Reader")
    upload = next(node for node in reader.body if isinstance(node, ast.FunctionDef) and node.name == "_to_device")
    allocation = next(node for node in ast.walk(upload) if isinstance(node, ast.Call)
                      and ast.unparse(node.func) == "torch.empty" and "pin_memory" in
                      {kw.arg for kw in node.keywords})
    assert ast.unparse(allocation.args[0]) == "(PIECE + 3 * ALIGN,)"
    assert any(ast.unparse(node) == "range(2)" for node in ast.walk(upload) if isinstance(node, ast.Call))
    logical = (64 << 20) + 3 * 4096
    physical_each = 1 << (logical - 1).bit_length()
    assert physical_each == 128 << 20
    assert READER_HOST_RING_BYTES == 2 * physical_each == 256 << 20


@pytest.mark.parametrize("raw,published", [(0, 0), (17, 31), (31, 17), (1 << 30, 7)])
def test_host_backing_counts_coexisting_ring_and_successive_phase_max_once(raw, published):
    assert reader_host_bytes(raw, published) == (256 << 20) + max(raw, published)


@pytest.mark.parametrize("raw,published", [(-1, 0), (True, 0), (0, 1.0)])
def test_invalid_host_byte_bounds_are_refused(raw, published):
    with pytest.raises(ValueError, match="nonnegative integer"):
        reader_host_bytes(raw, published)


def modelopt_headers(width=32, dims=32):
    text = {"num_hidden_layers": 1, "num_experts": 1, "moe_intermediate_size": width, "hidden_size": dims}
    entries = {}
    for projection, n, k in (("gate_proj", width, dims), ("up_proj", width, dims), ("down_proj", dims, width)):
        base = "model.layers.0.mlp.experts.0." + projection
        for suffix, dtype, shape in (("weight", "U8", [n, k // 2]), ("weight_scale", "F8_E4M3", [n, k // 16]),
                                     ("weight_scale_2", "F32", [1])):
            entries[base + "." + suffix] = {"dtype": dtype, "shape": shape}
    return text, entries


def test_small_modelopt_ring_and_publication_are_host_only(monkeypatch):
    text, entries = modelopt_headers()
    monkeypatch.setattr(ram_experts, "headers", lambda _: entries)
    layout = ram_experts._nvfp4_layout("opaque", "auto", text, mtp=False)
    field_bytes = 2 * 144 * 4
    assert layout.loading_device_bytes == 64 << 20
    assert layout.loading_bytes == (256 << 20) + field_bytes + 9 * 4096
    assert layout.loading_bytes > layout.loading_device_bytes
    assert layout.host_bytes == 3 * 144 * 4 + 12


def test_stacked_mtp_host_bound_counts_one_original_f32_expert_row(monkeypatch):
    text, entries = modelopt_headers(width=32, dims=32)
    entries["mtp.layers.0.mlp.experts.gate_up_proj"] = {"dtype": "F32", "shape": [1, 64, 32]}
    entries["mtp.layers.0.mlp.experts.down_proj"] = {"dtype": "F32", "shape": [1, 32, 32]}
    monkeypatch.setattr(ram_experts, "headers", lambda _: entries)
    layout = ram_experts._nvfp4_layout("opaque", "auto", text, mtp=True)
    assert layout.loading_bytes == (256 << 20) + 64 * 32 * 4 + 9 * 4096
    assert layout.loading_device_bytes == 64 << 20


def test_qwen_shared_host_estimator_keeps_ring_separate_from_device_peak(monkeypatch):
    entries = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [2, 64]},
               "model.norm.weight": {"dtype": "BF16", "shape": [64]},
               "lm_head.weight": {"dtype": "BF16", "shape": [2, 64]}}
    monkeypatch.setattr(capacity, "headers", lambda _: entries)
    transform = lambda name, info: (math.prod(info["shape"]) * 2, 0)
    original = capacity._estimate_weights_from_headers(entries, transform)
    weights, host = nvfp4_memory.weight_estimate("opaque", transform)
    assert host == (256 << 20) + original.staging
    assert weights.staging == nvfp4_memory.scheduled_loading_bytes(entries, transform)
    assert weights.staging < 256 << 20
