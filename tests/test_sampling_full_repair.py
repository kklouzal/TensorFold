"""Exact NumPy-only full-row repair against the legacy full lexsort oracle.

Load the owned pure-NumPy function ASTs from the active package source. This
boundary suite requires no Torch installation, device or synthetic model.
Actual sampler/CUDA integration remains separately required.
"""
from __future__ import annotations

import ast
import importlib.util
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(importlib.util.find_spec("tensorfold.cuda.sampling").origin)


@pytest.fixture
def sampler():
    names = {"valid_scores", "ambiguous_rows", "_canonical_repair_order", "canonical_candidates"}
    nodes = [node for node in ast.parse(SOURCE.read_bytes()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {"np": np, "Callable": Callable}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)  # noqa: S102 - owned ASTs
    return namespace


def legacy(values, ids, kept):
    """Independent incumbent: no partition/cutoff/partial-selector mechanism."""
    return np.lexsort((ids, -values))[:kept]


def check_order(sampler, values, ids, kept):
    before, ids_before = values.tobytes(), ids.tobytes()
    values.flags.writeable = ids.flags.writeable = False
    expected = legacy(values, ids, kept)
    actual = sampler["_canonical_repair_order"](values, ids, kept)
    np.testing.assert_array_equal(actual, expected)
    assert values[actual].tobytes() == values[expected].tobytes()
    assert ids[actual].tobytes() == ids[expected].tobytes()
    assert values.tobytes() == before and ids.tobytes() == ids_before


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("width", [65, 512, 4095, 4096, 8192, 262144])
def test_random_complete_rows_match_full_lexsort_at_sparse_and_dense_prefixes(sampler, dtype, width):
    rng = np.random.default_rng(20261021 + width)
    values = rng.standard_normal(width).astype(dtype)
    ids = rng.permutation(width).astype(np.int64) + (1 << 54)
    for kept in sorted({1, min(64, width), min(256, width), max(1, width // 4 - 1),
                        width // 4, width // 2, width - 1, width}):
        check_order(sampler, values, ids, kept)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["all-tied", "plateau", "signed-zero", "masked", "extreme", "duplicate-id"])
def test_adversarial_ties_masks_score_bits_and_global_ids(sampler, dtype, kind):
    width = 8192
    rng = np.random.default_rng(20261021)
    values = rng.integers(-4, 5, width).astype(dtype)
    ids = np.iinfo(np.int64).max - rng.permutation(width).astype(np.int64)
    if kind == "all-tied":
        values.fill(3)
    elif kind == "plateau":
        values.fill(1)
        values[::79] = 2
    elif kind == "signed-zero":
        values.fill(0)
        values[::2] = -0.0
    elif kind == "masked":
        values.fill(-np.inf)
        values[::1024] = np.arange(8, dtype=dtype)
    elif kind == "extreme":
        info = np.finfo(dtype)
        values = np.resize(np.array([info.max, -info.max, info.tiny, -info.tiny,
            np.nextafter(dtype(0), dtype(1)), -np.nextafter(dtype(0), dtype(1)),
            0.0, -0.0, -np.inf], dtype=dtype), width)
    else:
        # Production IDs are unique, but preserve even the incumbent's stable
        # ordering for equal score/ID keys, including distinct signed-zero bits.
        values.fill(0)
        values[::2] = -0.0
        ids = np.arange(width, dtype=np.int64) % 13
    for kept in (1, 64, 256, 2047, 2048, width - 1, width):
        check_order(sampler, values, ids, kept)


def test_noncontiguous_complete_views_remain_unchanged(sampler):
    rng = np.random.default_rng(71)
    base = rng.standard_normal(32768).astype(np.float32)
    id_base = rng.permutation(32768).astype(np.int64) + (1 << 55)
    values, ids = base[1::2][::-1], id_base[1::2][::-1]
    before, ids_before = base.tobytes(), id_base.tobytes()
    check_order(sampler, values, ids, 64)
    assert base.tobytes() == before and id_base.tobytes() == ids_before


class NumpyCalls:
    def __init__(self):
        self.sort_widths = []

    def __getattr__(self, name):
        return getattr(np, name)

    def lexsort(self, keys):
        self.sort_widths.append(len(keys[0]))
        return np.lexsort(keys)


@pytest.mark.parametrize("count", [64, 256])
def test_actual_full_row_callback_repairs_with_exact_owned_outputs_and_small_final_sort(sampler, count):
    width = 262144
    rng = np.random.default_rng(31)
    full = rng.integers(-3, 4, width).astype(np.float32)
    mapping = rng.permutation(width).astype(np.int64) + (1 << 54)
    candidates = legacy(full, mapping, count + 1)
    provisional, ids = full[candidates][None], mapping[candidates][None]
    full.flags.writeable = mapping.flags.writeable = False
    reads = []
    def complete(row):
        reads.append(row)
        return full, mapping
    calls = NumpyCalls()
    sampler["np"] = calls
    got_values, got_ids = sampler["canonical_candidates"](provisional, ids, count, width, complete)
    expected = legacy(full, mapping, count)
    assert reads == [0] and calls.sort_widths == [count]
    assert got_values[0].tobytes() == full[expected].tobytes()
    np.testing.assert_array_equal(got_ids[0], mapping[expected])
    assert not np.shares_memory(got_values, full) and not np.shares_memory(got_ids, mapping)
    got_values.fill(99)
    assert full.max() == 3


def test_ordinary_provisional_candidate_path_is_unchanged(sampler):
    values = np.arange(65, dtype=np.float32)[None]
    ids = np.arange(65, dtype=np.int64)[None]
    def unexpected(*args):
        pytest.fail("ordinary candidate rows must not enter full-row repair")
    sampler["_canonical_repair_order"] = unexpected
    calls = NumpyCalls()
    sampler["np"] = calls
    result = sampler["canonical_candidates"](values, ids, 64, 262144, unexpected)
    order = legacy(values[0], ids[0], 64)
    np.testing.assert_array_equal(result[0][0], values[0, order])
    assert calls.sort_widths == [65]


@pytest.mark.parametrize("fault", ["nan", "positive-inf", "all-masked", "score-shape", "id-shape"])
def test_full_row_domain_or_shape_failure_precedes_partial_selection(sampler, fault):
    width, count = 4096, 64
    full, mapping = np.zeros(width, dtype=np.float32), np.arange(width, dtype=np.int64)
    if fault == "nan":
        full[-1] = np.nan
    elif fault == "positive-inf":
        full[-1] = np.inf
    elif fault == "all-masked":
        full.fill(-np.inf)
    elif fault == "score-shape":
        full = full[:-1]
    else:
        mapping = mapping[:-1]
    def unexpected(*args):
        pytest.fail("invalid complete rows must fail before selection")
    sampler["_canonical_repair_order"] = unexpected
    provisional = np.zeros((1, count + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="sampling needs|complete immutable row"):
        sampler["canonical_candidates"](provisional, np.arange(count + 1)[None], count, width,
                                          lambda row: (full, mapping))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_invalid_provisional_batch_never_reads_complete_source(sampler, bad):
    values = np.zeros((2, 65), dtype=np.float32)
    values[1].fill(bad)
    def unexpected(*args):
        pytest.fail("invalid provisional batch must fail before a source callback")
    with pytest.raises(ValueError, match="sampling needs"):
        sampler["canonical_candidates"](values, np.tile(np.arange(65), (2, 1)), 64, 4096, unexpected)


def test_full_row_callback_error_retains_identity_and_native_cause(sampler):
    primary, cause = RuntimeError("source readback failed"), OSError("native transport cause")
    primary.__cause__ = cause
    def complete(row):
        raise primary
    with pytest.raises(RuntimeError) as caught:
        sampler["canonical_candidates"](np.zeros((1, 65), dtype=np.float32), np.arange(65)[None], 64, 4096, complete)
    assert caught.value is primary and caught.value.__cause__ is cause
