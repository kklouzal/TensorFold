"""Both real host estimates and the preliminary pin phase count bulk owners once."""

import ast
import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda import capacity
from tensorfold.cuda.nvfp4 import host_layout
from tensorfold.cuda.nvfp4.host_layout import (
    authority_fields_a4,
    bulk_host_bytes_a4,
    bulk_layout_a4,
    bulk_plan_host_bytes_a4,
    scalar_batch_host_bytes_a4,
)
from tensorfold.cuda.read_budget import READER_PIECE_BYTES, reader_host_bytes
from tensorfold.families.qwen3_5.cuda import nvfp4_memory
from tensorfold.families.qwen3_5.cuda.nvfp4_memory import weight_estimate


def bank(count, width=64, dims=128, *, layer=0):
    result = {}
    for expert in range(count):
        for p, n, k in (("gate_proj", width, dims), ("up_proj", width, dims), ("down_proj", dims, width)):
            base = f"model.layers.{layer}.mlp.experts.{expert}.{p}."
            result[base + "weight_packed"] = {"dtype": "U8", "shape": [n, k // 2]}
            result[base + "weight_scale"] = {"dtype": "F8_E4M3", "shape": [n, k // 16]}
            result[base + "weight_global_scale"] = {"dtype": "F32", "shape": [1]}
            result[base + "input_global_scale"] = {"dtype": "F32", "shape": [1]}
    return result


def test_actual_q36_window_leaves_all_alignment_inside_the_existing_cap():
    layout = bulk_layout_a4(257, 512, 2048)
    assert layout.scale_bytes == 50_528_256 and layout.chunk_experts == 10
    assert layout.payload_bytes == 66_281_472 <= READER_PIECE_BYTES
    assert layout.payload_bytes + 3 * layout.code_field_bytes > READER_PIECE_BYTES
    assert bulk_layout_a4(1023, 512, 2048) is None
    # An owned subset can still be eligible when its complete source bank is
    # outside the window. The estimator must not return zero for that bank.
    raw, resident, plan = bulk_host_bytes_a4(bank(1023, 512, 2048))
    assert resident > raw >= layout.payload_bytes and plan > 0


def test_fixed_native_snapshot_is_separate_from_the_torch_index_and_reused_by_chunks(monkeypatch):
    assert host_layout.NATIVE_A4_INDEX_SNAPSHOT_BYTES == 8184
    charged = [bulk_plan_host_bytes_a4(count, 1000) for count in (1, 2, 1023)]
    assert bulk_plan_host_bytes_a4(0, 1000) == 0
    monkeypatch.setattr(host_layout, "NATIVE_A4_INDEX_SNAPSHOT_BYTES", 0)
    assert [value - bulk_plan_host_bytes_a4(count, 1000)
            for count, value in zip((1, 2, 1023), charged, strict=True)] == [8184] * 3
    assert bulk_plan_host_bytes_a4(0, 1000) == 0


def test_native_snapshot_does_not_borrow_the_offload_raw_window():
    layout = bulk_layout_a4(815, 64, 768)
    assert layout.chunk_experts == 808
    assert layout.payload_bytes == 67_107_840 <= READER_PIECE_BYTES
    assert layout.payload_bytes + host_layout.NATIVE_A4_INDEX_SNAPSHOT_BYTES > READER_PIECE_BYTES


@pytest.mark.parametrize("width,dims", [(64, 64), (128, 192), (512, 2048)])
def test_subset_envelope_covers_every_admissible_count_and_never_exceeds_raw_window(width, dims):
    entries = bank(257, width, dims)
    raw, resident, planning = bulk_host_bytes_a4(entries)
    for count in range(1, 258):
        layout = bulk_layout_a4(count, width, dims)
        if layout is not None:
            assert layout.payload_bytes <= raw <= READER_PIECE_BYTES
            authority = sum(math.prod(shape) * (1 if dtype == "U8" else 4)
                            for shape, dtype in authority_fields_a4(count, width, dims))
            assert resident >= authority + layout.payload_bytes
    assert planning > 6 * 257 * 8 + 8 * 257


def test_ineligible_geometry_has_no_added_payload_or_plan_and_cumulative_scan_maps_are_charged():
    assert bulk_host_bytes_a4(bank(1, 8192, 8192)) == (0, 0, 0)
    assert bulk_host_bytes_a4({}) == (0, 0, 0)
    one = bank(1)
    many = {name: info for layer in range(1024) for name, info in bank(1, layer=layer).items()}
    assert bulk_host_bytes_a4(many)[2] > bulk_host_bytes_a4(one)[2]


@pytest.mark.parametrize("large_head", [False, True])
@pytest.mark.parametrize("staged", [False, True])
def test_resident_increment_is_exact_phase_widening_and_offload_only_adds_plan_metadata(monkeypatch, large_head, staged):
    entries = bank(8)
    entries.update({"model.embed_tokens.weight": {"dtype": "BF16", "shape": [2, 128]},
                    "lm_head.weight": {"dtype": "BF16", "shape": [2, 128]},
                    "model.norm.weight": {"dtype": "BF16", "shape": [128]}})
    if large_head:
        entries["lm_head.weight"] = {"dtype": "BF16", "shape": [4096, 128]}
    monkeypatch.setattr(capacity, "headers", lambda _: entries)
    monkeypatch.setattr(nvfp4_memory, "reader_host_bytes", partial(reader_host_bytes, staged=staged))

    def transform(name, info):
        return math.prod(info["shape"]) * capacity.SIZES[info["dtype"]], 0

    old = capacity._estimate_weights_from_headers(entries, transform)
    largest = max(math.prod(info["shape"]) * capacity.itemsize(info, name) for name, info in entries.items())
    prior = reader_host_bytes(largest, max(old.staging, largest), staged=staged)
    _raw, resident_phase, planning = bulk_host_bytes_a4(entries)
    delta = reader_host_bytes(largest, max(old.staging, largest, resident_phase), staged=staged) - prior
    resident = weight_estimate("same-header-snapshot", transform, scalar_batches=True, details=True)
    _, offload_host = weight_estimate("same-header-snapshot", transform, scalar_batches=True, bulk_resident=False)
    assert resident.bulk_host_increment == delta
    assert resident.host_bytes == prior + scalar_batch_host_bytes_a4(entries) + planning + delta
    assert offload_host == resident.host_bytes - delta
    if large_head:
        assert delta == 0  # a proved disjoint late-read/publication phase dominates


@pytest.mark.parametrize("resident", [False, True])
def test_actual_estimator_keeps_one_native_snapshot_outside_the_resident_overlap_delta(monkeypatch, resident):
    entries = bank(8)
    entries.update({"model.embed_tokens.weight": {"dtype": "BF16", "shape": [2, 128]},
                    "lm_head.weight": {"dtype": "BF16", "shape": [2, 128]},
                    "model.norm.weight": {"dtype": "BF16", "shape": [128]}})
    monkeypatch.setattr(capacity, "headers", lambda _: entries)

    def transform(_name, info):
        return math.prod(info["shape"]) * capacity.SIZES[info["dtype"]], 0

    charged = weight_estimate("same-header-snapshot", transform, scalar_batches=True,
                              bulk_resident=resident, details=True)
    monkeypatch.setattr(host_layout, "NATIVE_A4_INDEX_SNAPSHOT_BYTES", 0)
    uncharged = weight_estimate("same-header-snapshot", transform, scalar_batches=True,
                                bulk_resident=resident, details=True)
    assert charged.host_bytes - uncharged.host_bytes == 8184
    assert charged.bulk_host_increment == uncharged.bulk_host_increment
    # Preliminary pin admission removes this exact delta, so the independent
    # snapshot charge remains in both the resident and offload future bound.
    assert ((charged.host_bytes - charged.bulk_host_increment)
            - (uncharged.host_bytes - uncharged.bulk_host_increment)) == 8184


@pytest.mark.parametrize("staged", [False, True])
def test_smallest_resident_bank_charges_its_cpu_authority_alongside_worst_alignment_owners(monkeypatch, staged):
    entries = bank(1, 64, 64)
    raw, resident_phase, _plan = bulk_host_bytes_a4(entries)
    authority = sum(math.prod(shape) * (1 if dtype == "U8" else 4)
                    for shape, dtype in authority_fields_a4(1, 64, 64))
    assert authority == 6936
    assert resident_phase == raw + authority > raw
    entries.update({"model.embed_tokens.weight": {"dtype": "BF16", "shape": [2, 64]},
                    "lm_head.weight": {"dtype": "BF16", "shape": [2, 64]},
                    "model.norm.weight": {"dtype": "BF16", "shape": [64]}})
    monkeypatch.setattr(capacity, "headers", lambda _: entries)
    monkeypatch.setattr(nvfp4_memory, "reader_host_bytes", partial(reader_host_bytes, staged=staged))
    transform = lambda _name, info: math.prod(info["shape"]) * capacity.SIZES[info["dtype"]]
    original = capacity._estimate_weights_from_headers(entries, lambda name, info: (transform(name, info), 0))
    largest = max(transform(name, info) for name, info in entries.items())
    prior_temporary = max(original.staging, largest)
    assert resident_phase > prior_temporary
    report = weight_estimate("smallest-valid-bank", lambda name, info: (transform(name, info), 0),
                             scalar_batches=True, details=True)
    assert report.bulk_host_increment == resident_phase - prior_temporary


@pytest.mark.parametrize("rejection", [0, 450, 600])
@pytest.mark.parametrize("snapshot", [0, 8184])
def test_actual_preliminary_pin_call_removes_only_known_bulk_delta_before_the_rejection_max(rejection, snapshot):
    source = Path(__file__).parents[1] / "src/tensorfold/families/qwen3_5_moe/cuda/nvfp4_engine.py"
    node = next(node for node in ast.walk(ast.parse(source.read_bytes()))
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pins" for t in node.targets))
    seen = []
    scope = {"ExpertPinPolicy": lambda authority, **kwargs: seen.append((authority, kwargs)),
             "plan": SimpleNamespace(host_bytes=1000, authority_allocations=(1000,)),
             "host_extra": 123, "host_weight_staging": 500 + snapshot,
             "resident_report": SimpleNamespace(bulk_host_increment=50),
             "self": SimpleNamespace(rejection_workspace={"host_bytes": rejection}), "vram_experts_pin": "auto"}
    exec(compile(ast.Module([node], []), str(source), "exec"), scope)  # noqa: S102 - exact owned pin admission expression.
    assert seen == [(1000, {"mode": "auto", "future_bytes": 123 + max(450 + snapshot, rejection),
                           "allocation_bytes": (1000,)})]
