"""Unique histories retain selector storage and omit unreachable legacy arrays."""

from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda import geometry
from tensorfold.cuda.sm70 import family_ops
from tensorfold.families.qwen4_exp.cuda.attention import AttnScratch, attention
from tensorfold.families.qwen4_exp.cuda.history_placement import HistoryPlacement
from tests.test_adaptive_session_capacity import TEXT

pytestmark = pytest.mark.torch


def _bytes(scratch):
    return sum(value.nbytes for value in vars(scratch).values() if isinstance(value, torch.Tensor))


def _scratch(rows, *, selection_only=False):
    return AttnScratch(rows, 3, 256, 4096, "cpu", budget=2048, ratio=4,
                       dtype=torch.float16, selection_only=selection_only)


def test_selection_only_role_omits_only_legacy_chunk_radix_and_codec_arrays(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: NS(multi_processor_count=80))
    legacy, selector = _scratch(8), _scratch(8, selection_only=True)
    removed = ("po", "pm", "pl", "out", "hist", "suf", "centroids", "thresholds")
    assert not set(removed) & set(vars(selector))
    assert _bytes(legacy) - _bytes(selector) == sum(getattr(legacy, name).nbytes for name in removed)
    for name in ("ids", "nk", "sparse", "scores"):
        assert torch.equal(getattr(legacy, name), getattr(selector, name))
    # The explicit role cannot accidentally enter a legacy output consumer.
    query = torch.zeros((1, 3, 256), dtype=torch.float16)
    cache = torch.zeros((4096, 1, 256), dtype=torch.float16)
    with pytest.raises(ValueError, match="selection-only"):
        attention(query, cache, cache, torch.zeros(1, dtype=torch.int32), selector, 1, .0625)


def test_sparse_selection_at_dense_boundary_is_identical_for_both_storage_roles(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: NS(multi_processor_count=80))
    legacy, selector = _scratch(8), _scratch(8, selection_only=True)
    generator = torch.Generator().manual_seed(493)
    query = torch.randn((8, 4, 128), generator=generator).half()
    pooled = torch.randn((1024, 128), generator=generator).half()
    position = torch.tensor([2047], dtype=torch.int32)
    for scratch in (legacy, selector):
        family_ops.qsa_rows(query, pooled, position, scratch, 8, context=2055)
    for name in ("ids", "nk", "sparse", "scores"):
        assert torch.equal(getattr(legacy, name), getattr(selector, name))


@pytest.mark.parametrize("mtp", [False, True])
def test_geometry_removes_only_proven_detailed_terms_not_coarse_output_allowance(monkeypatch, mtp):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: NS(multi_processor_count=80))
    rows, capacity = 12, 4096
    options = dict(streams=3, each=4, keep=8, mtp=mtp, world=8, full_windows=3, execution_rows=rows,
                   prefill_rows=256, kv_pair=((8, 32, 2),) * 2)
    legacy = geometry.indexed_stream_geometry(TEXT, **options).breakdown_at(capacity)
    unique = geometry.indexed_stream_geometry(TEXT, **options, history_placement=HistoryPlacement(8, 2, 12),
                                              rank=0).breakdown_at(capacity)
    construction_delta = output_delta = table_padding = 0
    for count in [rows] * (1 + int(mtp)) + [256]:
        full, selected = _scratch(count), _scratch(count, selection_only=True)
        construction_delta += _bytes(full) - _bytes(selected)
        output_delta += full.out.nbytes
        table_padding += (geometry.CENTROID_TABLE_BYTES - full.centroids.nbytes
                          + geometry.THRESHOLD_TABLE_BYTES - full.thresholds.nbytes)
    # AttnScratch.out is within an inherited coarse extent bound. Preserve that
    # allowance until a separate proof assigns it; detailed removed arrays are
    # accounted exactly by this construction-derived delta.
    # Existing geometry names CUDA allocation-block padding for the fixed
    # codebooks; CPU tensor nbytes deliberately excludes that separate charge.
    assert legacy["attention_scratch"] - unique["attention_scratch"] == construction_delta - output_delta + table_padding
