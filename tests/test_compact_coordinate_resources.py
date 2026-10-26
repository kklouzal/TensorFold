"""Coordinate payload accounting against actual canonical issuance/storage."""

import inspect
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.compact_indexer import Staging, issue_text_indexer_coordinates
from tensorfold.families.qwen4_exp.cuda.history_placement import coordinate_source_host_bytes, compact_stage_bytes
from test_qwen4_exp_position_coordinates import policy


@pytest.mark.parametrize("rows", [1, 2, 32, 2048])
def test_host_peak_contains_the_actual_simultaneously_owned_numpy_payload(monkeypatch, rows):
    position = policy(limit=rows + 64)
    state = NS(coordinate_profile_identity="a" * 64, slot_generation=3, media_identity=None,
               image_positions=None, rope_delta=5, capacity=rows + 64, pos_dev=torch.zeros(1, dtype=torch.int32))
    original, observed = torch.tensor, []
    def tensor(data, *args, **kwargs):
        caller = inspect.currentframe().f_back
        parent = caller.f_back
        try:
            arrays = [data, *caller.f_locals.values(), *parent.f_locals.values()]
            # The issuer's raw/host_positions are aliases, not extra matrices.
            payloads = {(value.__array_interface__["data"][0], value.nbytes)
                        for value in arrays if isinstance(value, np.ndarray)}
            observed.append(sum(size for _pointer, size in payloads))
            return original(data, *args, **kwargs)
        finally:
            del arrays, caller, parent
    monkeypatch.setattr(torch, "tensor", tensor)
    issued = issue_text_indexer_coordinates(state, position, 7, rows, max_rows=rows)
    assert observed == [68 * rows]
    assert issued.tensor.dtype == torch.int32 and issued.tensor.numel() * issued.tensor.element_size() == 12 * rows
    assert issued.tensor[0].tolist() == [12, 12, 12]
    expected = 68 * rows if rows >= 2 else 80  # The 3-int64 reduction result dominates only at one row.
    assert coordinate_source_host_bytes(rows) == expected >= observed[0]


@pytest.mark.parametrize("rows", [False, 0, -1, 1.5])
def test_host_source_bound_rejects_invalid_geometry(rows):
    with pytest.raises(ValueError, match="coordinate source rows"):
        coordinate_source_host_bytes(rows)


@pytest.mark.parametrize("rows,mtp", [(1, False), (32, True), (2048, True)])
def test_owned_target_mtp_and_prefill_coordinates_are_already_in_compact_staging(rows, mtp):
    count = 12 + int(mtp)
    roles = [Staging(rows, count, "cpu", mtp=False), Staging(rows, count, "cpu", mtp=mtp)]
    expected = compact_stage_bytes(rows, count, mtp=False) + compact_stage_bytes(rows, count, mtp=mtp)
    if mtp:
        roles.append(Staging(rows, count, "cpu", mtp=True, mtp_only=True))
        expected += compact_stage_bytes(rows, count, mtp=True, mtp_only=True)
    assert sum(role.nbytes for role in roles) == expected
    coordinates = sum(role.coordinates.nbytes + role.mtp_coordinates.nbytes for role in roles)
    assert coordinates == rows * 24 * (4 if mtp else 2)
    assert coordinate_source_host_bytes(rows) > 0  # A host-only delta; no additional GPU owner is constructed.
