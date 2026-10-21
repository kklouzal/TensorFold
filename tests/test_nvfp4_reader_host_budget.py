"""Independent physical Reader backing and nonduplicated NVFP4 host admission."""

import ast
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda import capacity
from tensorfold.cuda.read_budget import (
    READER_HOST_RING_BYTES,
    READER_PIECE_BYTES,
    pinned_backing_bytes,
    reader_host_bytes,
)
from tensorfold.families.qwen3_5.cuda import nvfp4_memory
from tensorfold.families.qwen4_exp import ram_experts


def test_physical_reader_bound_matches_existing_owned_allocation_geometry():
    path = Path(__file__).parents[1] / "src/tensorfold/cuda/direct_read.py"
    tree = ast.parse(path.read_text())
    values = {node.targets[0].id: node.value for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("PIECE", "ALIGN")}
    assert ast.unparse(values["PIECE"]) == "READER_PIECE_BYTES"
    assert any(isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "read_budget"
               and any(alias.name == "READER_PIECE_BYTES" and alias.asname is None for alias in node.names)
               for node in tree.body)
    budget_tree = ast.parse((path.parent / "read_budget.py").read_bytes())
    piece = next(node.value for node in budget_tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "READER_PIECE_BYTES" for target in node.targets))
    assert ast.unparse(piece) == "64 << 20" and READER_PIECE_BYTES == 64 << 20
    assert ast.literal_eval(values["ALIGN"]) == 4096
    reader = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Reader")
    construction = next(node for node in reader.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    staged = next(node.value for node in ast.walk(construction) if isinstance(node, ast.Assign)
                  and any(ast.unparse(left) == "self.staged" for left in node.targets))
    assert ast.unparse(staged) == "os.name == 'nt'"
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
    assert reader_host_bytes(raw, published, staged=False) == (256 << 20) + max(raw, published)


@pytest.mark.parametrize("raw,published", [(0, 0), (17, 31), (64 << 20, 7), ((64 << 20) + 1, 11)])
def test_staged_blocking_uploads_charge_cached_buckets_alongside_cpu_publication(raw, published):
    bucket = 0 if raw == 0 else 1 << (raw - 1).bit_length()
    assert reader_host_bytes(raw, published, staged=True) == 2 * bucket + published
    assert pinned_backing_bytes(raw) == bucket


@pytest.mark.parametrize("raw,published", [(-1, 0), (True, 0), (0, 1.0)])
def test_invalid_host_byte_bounds_are_refused(raw, published):
    with pytest.raises(ValueError, match="nonnegative integer"):
        reader_host_bytes(raw, published)


def modelopt_headers(path, width=32, dims=32):
    text = {"num_hidden_layers": 1, "num_experts": 1, "moe_intermediate_size": width, "hidden_size": dims}
    (path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": text,
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "W4A16_NVFP4",
            "config_groups": {"experts": {"targets": ["Linear"],
                "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                "input_activations": None}}}}))
    text = capacity.config(path)  # actual canonical resolver supplies _quantization
    entries = {}
    for projection, n, k in (("gate_proj", width, dims), ("up_proj", width, dims), ("down_proj", dims, width)):
        base = "model.layers.0.mlp.experts.0." + projection
        for suffix, dtype, shape in (("weight", "U8", [n, k // 2]), ("weight_scale", "F8_E4M3", [n, k // 16]),
                                     ("weight_scale_2", "F32", [1])):
            entries[base + "." + suffix] = {"dtype": dtype, "shape": shape}
    return text, entries


def test_small_modelopt_ring_and_publication_are_host_only(monkeypatch, tmp_path):
    text, entries = modelopt_headers(tmp_path)
    monkeypatch.setattr(ram_experts, "headers", lambda _: entries)
    layout = ram_experts._nvfp4_layout("opaque", "auto", text, mtp=False)
    field_bytes = 2 * 144 * 4
    assert layout.loading_device_bytes == 64 << 20
    expected = 2 * 512 + field_bytes if os.name == "nt" else (256 << 20) + field_bytes
    assert layout.loading_bytes == expected + 9 * 4096
    assert layout.host_bytes == 3 * 144 * 4 + 12


def test_large_modelopt_gpu_scratch_does_not_become_a_host_reserve(monkeypatch, tmp_path):
    text, entries = modelopt_headers(tmp_path, width=4096, dims=4096)
    monkeypatch.setattr(ram_experts, "headers", lambda _: entries)
    layout = ram_experts._nvfp4_layout("opaque", "auto", text, mtp=False)
    packed_field = 2 * (4096 // 32) * (4096 // 32) * 144 * 4
    raw = 4096 * 4096 // 2
    expected = 2 * raw + packed_field if os.name == "nt" else (256 << 20) + packed_field
    assert layout.loading_bytes == expected + 9 * 4096
    assert layout.loading_device_bytes == 128 * 4096 * 4096 + 8 * layout.entry_bytes
    assert layout.loading_bytes < layout.loading_device_bytes
    assert layout.staging_bytes == 1 << (2 * layout.entry_bytes - 1).bit_length()


def test_stacked_mtp_host_bound_counts_one_original_f32_expert_row(monkeypatch, tmp_path):
    text, entries = modelopt_headers(tmp_path, width=32, dims=32)
    entries["mtp.layers.0.mlp.experts.gate_up_proj"] = {"dtype": "F32", "shape": [1, 64, 32]}
    entries["mtp.layers.0.mlp.experts.down_proj"] = {"dtype": "F32", "shape": [1, 32, 32]}
    monkeypatch.setattr(ram_experts, "headers", lambda _: entries)
    layout = ram_experts._nvfp4_layout("opaque", "auto", text, mtp=True)
    expected = 2 * (64 * 32 * 4) + 2 * 144 * 4 if os.name == "nt" else (256 << 20) + 64 * 32 * 4
    assert layout.loading_bytes == expected + 9 * 4096
    assert layout.loading_device_bytes == 64 << 20


def test_qwen_shared_host_estimator_keeps_ring_separate_from_device_peak(monkeypatch):
    entries = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [2, 64]},
               "model.norm.weight": {"dtype": "BF16", "shape": [64]},
               "lm_head.weight": {"dtype": "BF16", "shape": [2, 64]}}
    monkeypatch.setattr(capacity, "headers", lambda _: entries)
    transform = lambda name, info: (math.prod(info["shape"]) * 2, 0)
    original = capacity._estimate_weights_from_headers(entries, transform)
    weights, host = nvfp4_memory.weight_estimate("opaque", transform)
    expected = 2 * 256 + original.staging if os.name == "nt" else (256 << 20) + original.staging
    assert host == expected
    assert weights.staging == nvfp4_memory.scheduled_loading_bytes(entries, transform)
    assert weights.staging < 256 << 20


def _source_assignment(path, target, *, required_key=None):
    tree = ast.parse(path.read_text())
    return next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(ast.unparse(left) == target for left in node.targets)
                and (required_key is None or isinstance(node.value, ast.Dict)
                     and any(isinstance(key, ast.Constant) and key.value == required_key for key in node.value.keys)))


@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("kind", ["nvfp4", "affine4"])
def test_actual_group_and_leader_source_charge_each_reader_owner_once(ranks, kind):
    root = Path(__file__).parents[1] / "src/tensorfold/families/qwen4_exp/cuda"
    loading = (256 << 20) + 64 if kind == "nvfp4" else 64
    base = SimpleNamespace(host_bytes=1000, staging_bytes=20, loading_bytes=loading,
                           metadata_host_bytes=30, control_host_bytes=40, gpu_bytes=50,
                           metadata_device_bytes=60, control_device_bytes=70)
    session = SimpleNamespace()
    group = _source_assignment(root / "expert_parallel.py", "self.group_host_bytes")
    exec(compile(ast.fix_missing_locations(ast.Module([group], [])), "<actual Session group bound>", "exec"),  # noqa: S102 - bounded owned bookkeeping AST, not checkpoint code.
         {"self": session, "base": base, "ranks": ranks})
    assert session.group_host_bytes == 1000 + ranks * (20 + loading + 30 + 40 + (4 << 20))
    tree = ast.parse((root / "engine.py").read_text())
    initial = _source_assignment(root / "engine.py", "host_capacity", required_key="host_resident")
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "expert_session is not None"
                  and any(isinstance(child, ast.Assign) and any(ast.unparse(left) ==
                          "host_capacity['host_resident']" for left in child.targets) for child in node.body))
    for owner in (None, session):
        scope = {"ram_layout": base, "scratch_host": 7, "expert_session": owner}
        exec(compile(ast.fix_missing_locations(ast.Module([initial, branch], [])),  # noqa: S102 - actual owned admission bookkeeping.
                     "<actual leader capacity bound>", "exec"), scope)
        capacity = scope["host_capacity"]
        assert capacity["resident_extra"] == 50 + 60 + 70  # device budget is unchanged
        if owner is None:
            assert capacity["host_resident"] == 1000 + 30 + 20 + 40 + 7
            assert capacity["host_extra_staging"] == loading
        else:
            assert capacity["host_resident"] == session.group_host_bytes + 7
            assert capacity["host_extra_staging"] == 256 << 20
            reader_count = ranks + 1 if kind == "nvfp4" else 1
            nonring = 1000 + ranks * (20 + 64 + 30 + 40 + (4 << 20)) + 7
            assert capacity["host_resident"] + capacity["host_extra_staging"] == nonring + reader_count * (256 << 20)
