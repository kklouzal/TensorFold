"""Actual MTP text producer retains cache coordinates and conservative provenance."""
from types import SimpleNamespace as NS

import pytest
torch = pytest.importorskip("torch")

from tests.test_compact_position_frames import prepared  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import mtp, compact_indexer, history_owner  # noqa: E402


def case():
    buffers, position, states, explicit, segments = prepared(mtp=True, context_identity="1" * 64)
    buffers.mtp_in = torch.full((4, 16), 31, dtype=torch.float16)
    buffers.ids_host = torch.zeros(4, dtype=torch.int32)
    buffers.ids = torch.zeros(4, dtype=torch.int32)
    buffers.last_host = torch.zeros(2, dtype=torch.int32)
    buffers.last = torch.zeros(2, dtype=torch.int32)
    buffers.staged = NS(synchronize=lambda: None, record=lambda: None)
    for state in states:
        state.mtp_boundary = None
    hidden = torch.arange(64, dtype=torch.float16).view(4, 16)
    windows = [(states[0], [41, 42], hidden[:2]), (states[1], [99, 100], hidden[2:])]
    return buffers, position, states, explicit, segments, hidden, windows


def test_actual_text_mtp_stage_grants_native_recipe_without_next_token_coordinate_shift():
    buffers, _position, states, _explicit, expected_segments, hidden, windows = case()
    segments = mtp.mtp_stage(buffers, windows)
    assert segments == expected_segments
    assert buffers.mtp_indexer_coordinates.tolist() == [[7, 7, 7], [8, 8, 8], [12, 12, 12], [13, 13, 13]]
    assert compact_indexer._scalar_text_frame(buffers.mtp_indexer_frame)
    assert history_owner._native_prelude_position(buffers.mtp_indexer_position)
    assert torch.equal(buffers.mtp_in, hidden) and buffers.ids.tolist() == [41, 42, 99, 100]
    assert [state.mtp_len for state in states] == [7, 12]
    assert compact_indexer.validate_mtp_indexer_frame(buffers, segments).data_ptr() == buffers.mtp_indexer_coordinate_storage.data_ptr()
    buffers.input_preparation_context.close()


def test_explicit_nonmedia_custom_coordinates_keep_original_unqualified_recipe():
    buffers, _position, _states, explicit, _segments, hidden, windows = case()
    segments = mtp.mtp_stage(buffers, windows, indexer_coordinates=explicit)
    assert not compact_indexer._scalar_text_frame(buffers.mtp_indexer_frame)
    assert compact_indexer.validate_mtp_indexer_frame(buffers, segments).tolist() == [
        [7, 21, 31], [8, 22, 32], [12, 21, 31], [13, 22, 32]]
    assert torch.equal(buffers.mtp_in, hidden)
    buffers.input_preparation_context.close()


def test_deferred_boundary_keeps_actual_non_scalar_coordinate_transfer():
    buffers, position, states, explicit, _segments, _hidden, windows = case()
    states[0].mtp_boundary = NS(row=states[0].mtp_len, generation=states[0].slot_generation,
                               coordinate_record=explicit[0].source, position=position)
    segments = mtp.mtp_stage(buffers, windows)
    assert not compact_indexer._scalar_text_frame(buffers.mtp_indexer_frame)
    assert compact_indexer.validate_mtp_indexer_frame(buffers, segments).tolist() == [
        [7, 21, 31], [8, 8, 8], [12, 12, 12], [13, 13, 13]]
    buffers.input_preparation_context.close()


def test_ready_media_requires_actual_coordinates_before_staging_any_payload():
    buffers, _position, states, _explicit, _segments, _hidden, windows = case()
    states[0].media_input_span = object()
    with pytest.raises(ValueError, match="ready media MTP hidden rows"):
        mtp.mtp_stage(buffers, windows)
    assert bool((buffers.mtp_in == 31).all()) and bool((buffers.ids == 0).all())
    assert buffers.mtp_indexer_frame is None
    buffers.input_preparation_context.close()

def test_mtp_role_counter_is_bounded_and_overflow_becomes_explicitly_unknown():
    weights = NS(meta={"sm70_owner_prelude_submitted_frames": 0,
                       "sm70_owner_prelude_submitted_mtp_frames": (1 << 63) - 2,
                       "sm70_owner_prelude_mtp_counter_overflowed": False})
    history_owner._record_native_prelude(weights, mtp=False)
    assert weights.meta["sm70_owner_prelude_submitted_mtp_frames"] == (1 << 63) - 2
    history_owner._record_native_prelude(weights, mtp=True)
    assert weights.meta["sm70_owner_prelude_submitted_mtp_frames"] == (1 << 63) - 1
    assert not weights.meta["sm70_owner_prelude_mtp_counter_overflowed"]
    history_owner._record_native_prelude(weights, mtp=True)
    assert weights.meta["sm70_owner_prelude_submitted_mtp_frames"] == (1 << 63) - 1
    assert weights.meta["sm70_owner_prelude_mtp_counter_overflowed"]
    assert weights.meta["sm70_owner_prelude_submitted_frames"] == 3
