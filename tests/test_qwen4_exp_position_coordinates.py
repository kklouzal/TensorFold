"""Coordinate provenance comes from actual bounded CPU rows and controlled copies."""

import gc
import weakref

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.position_coordinates import (  # noqa: E402 - optional backend guard
    CoordinateProof, CoordinateTransfer, IssuedCoordinates, copy_coordinate_batch, copy_coordinates,
    issue_coordinates, restore_coordinates,
)
from tensorfold.families.qwen4_exp.position_profile import PositionOperands, PositionRole  # noqa: E402
from tensorfold.families.qwen4_exp.rope import RopeParameters  # noqa: E402


def policy(name="target", *, limit=128, identity="c" * 64, epoch=0):
    rope = RopeParameters("default", 1e7, 64, limit, limit)
    domain = PositionRole(name, rope, limit, (limit, limit, limit), "d" * 64, "control")
    return PositionOperands(torch.ones(32), 1., (11, 11, 10), domain, identity, epoch)


def issue(rows=None, **changes):
    if rows is None:
        rows = np.array([[7, 7, 7], [8, 3, 4], [8, 3, 5], [9, 9, 9]], dtype=np.int64)
    options = dict(position=policy(), profile_identity="a" * 64, generation=3, media_identity="b" * 64,
                   row_start=7, max_rows=4, device="cpu")
    options.update(changes)
    return issue_coordinates(rows, **options)


def validate(value, position=None, **changes):
    options = dict(profile_identity="a" * 64, generation=3, media_identity="b" * 64,
                   row_start=7, row_count=4)
    options.update(changes)
    value.proof.validate(value.tensor, policy() if position is None else position, **options)


def test_actual_host_bounds_bind_immutable_request_and_device_storage():
    raw = np.array([[7, 7, 7], [8, 3, 4], [8, 3, 5], [9, 9, 9]], dtype=np.int64)
    issued = issue(raw)
    raw.fill(99)
    assert issued.tensor.dtype == torch.int32 and issued.tensor.tolist()[1] == [8, 3, 4]
    assert issued.proof.minimum == (7, 3, 4) and issued.proof.maximum == (9, 9, 9)
    validate(issued)
    with pytest.raises(TypeError, match="immutable"):
        issued.proof.maximum = (0, 0, 0)
    with pytest.raises(TypeError, match="actual rows"):
        CoordinateProof(minimum=(0, 0, 0), maximum=(1, 1, 1))


@pytest.mark.parametrize("rows", [np.array([[2**32, 0, 0]], dtype=np.int64),
                                  np.array([[2**64 - 1, 0, 0]], dtype=np.uint64),
                                  np.array([[-1, 0, 0]], dtype=np.int64),
                                  np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.bool_),
                                  np.zeros((5, 3), dtype=np.int64), np.zeros((3, 1), dtype=np.int64)])
def test_invalid_rows_are_refused_before_narrowing_or_target_allocation(rows, monkeypatch):
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("target allocation before row/domain validation"))
    with pytest.raises(ValueError):
        issue(rows)


@pytest.mark.parametrize("changes", [dict(generation=4), dict(generation=3.), dict(profile_identity="e" * 64),
                                    dict(media_identity=None), dict(row_start=8), dict(row_count=3)])
def test_context_or_row_mismatch_cannot_reuse_a_proof(changes):
    with pytest.raises(ValueError, match="generation|another row"):
        validate(issue(), **changes)


def test_role_epoch_storage_and_foreign_process_are_validated_before_use(monkeypatch):
    issued = issue()
    for other in (policy("mtp"), policy(epoch=1), policy(identity="e" * 64)):
        with pytest.raises(ValueError, match="another row"):
            validate(issued, other)
    with pytest.raises(ValueError, match="storage changed"):
        issued.proof.validate(issued.tensor.clone(), policy(), profile_identity="a" * 64,
                              generation=3, media_identity="b" * 64, row_start=7, row_count=4)
    import tensorfold.families.qwen4_exp.cuda.position_coordinates as implementation

    monkeypatch.setattr(implementation.os, "getpid", lambda: -1)
    monkeypatch.setattr(implementation, "_signature", lambda *a: pytest.fail("device access before PID refusal"))
    with pytest.raises(ValueError, match="current-process"):
        validate(issued)


def test_controlled_small_boundary_copy_changes_dtype_role_and_does_not_pin_parent():
    source = issue()
    parent = weakref.ref(source.tensor)
    copied = copy_coordinates(source, source_position=policy(), position=policy("mtp"), row_start=10, row_count=1)
    assert copied.tensor.dtype == torch.int64 and copied.tensor.tolist() == [[9, 9, 9]]
    assert copied.tensor.data_ptr() != source.tensor.data_ptr()
    validate(copied, policy("mtp"), row_start=10, row_count=1)
    del source
    gc.collect()
    assert parent() is None
    validate(copied, policy("mtp"), row_start=10, row_count=1)


def test_controlled_staging_copy_binds_actual_destination_and_refuses_source_alias():
    source = issue()
    class Frame:
        pass
    owner = Frame()
    frame = torch.empty((6, 3), dtype=torch.int64)
    view = frame[1:5]
    staged = copy_coordinates(source, source_position=policy(), position=policy(), row_start=7, row_count=4,
                              destination=view, owner=owner)
    assert staged.tensor is view
    validate(staged, owner=owner)
    with pytest.raises(ValueError, match="reusable frame"):
        validate(staged, owner=Frame())
    torch.testing.assert_close(staged.tensor, source.tensor.long(), rtol=0, atol=0)
    with pytest.raises(ValueError, match="overlap"):
        copy_coordinates(source, source_position=policy(), position=policy(), row_start=7, row_count=4,
                         dtype=torch.int32, destination=source.tensor, owner=owner)
    for start, count in ((6, 1), (11, 1), (7, 5)):
        with pytest.raises(ValueError, match="outside"):
            copy_coordinates(source, source_position=policy(), position=policy(), row_start=start, row_count=count)
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert ref() is None
    with pytest.raises(ValueError, match="retired"):
        validate(staged)


def test_raw_record_or_unissued_object_cannot_authorize_copy():
    for proof in ({"minimum": (0, 0, 0)}, object.__new__(CoordinateProof)):
        source = IssuedCoordinates(torch.zeros(4, 3, dtype=torch.int32), proof)
        with pytest.raises(ValueError, match="issued"):
            copy_coordinates(source, source_position=policy(), position=policy(), row_start=7, row_count=4)


def test_empty_span_never_interprets_unused_coordinate_values():
    issued = issue(np.empty((0, 3), dtype=np.int64))
    validate(issued, row_count=0)
    assert issued.proof.minimum is None and issued.proof.maximum is None


def test_protected_snapshot_restore_copies_into_current_admitted_generation():
    source = issue()
    restored = restore_coordinates(source, source_position=policy(), position=policy(),
        profile_identity="a" * 64, media_identity="b" * 64, generation=8, row_start=7, row_count=4)
    assert restored.tensor.data_ptr() != source.tensor.data_ptr()
    validate(restored, generation=8)
    validate(source)
    with pytest.raises(ValueError, match="generation"):
        validate(restored)
    for changes in (dict(profile_identity="f" * 64), dict(media_identity=None), dict(generation=True)):
        context = dict(profile_identity="a" * 64, media_identity="b" * 64, generation=8)
        context.update(changes)
        with pytest.raises(ValueError, match="admitted|generation"):
            restore_coordinates(source, source_position=policy(), position=policy(),
                                row_start=7, row_count=4, **context)
    with pytest.raises(ValueError, match="profile identities"):
        restore_coordinates(source, source_position=policy(), position=policy(epoch=1),
            profile_identity="a" * 64, media_identity="b" * 64, generation=8, row_start=7, row_count=4)


def test_same_destination_reuse_invalidates_old_live_frame_bounds():
    class Frame:
        pass
    first, second = Frame(), Frame()
    destination = torch.empty(4, 3, dtype=torch.int64)
    old = copy_coordinates(issue(), source_position=policy(), position=policy(), row_start=7, row_count=4,
                           destination=destination, owner=first)
    newer_source = issue(np.full((4, 3), 31, dtype=np.int64), generation=4, media_identity="e" * 64)
    current = copy_coordinates(newer_source, source_position=policy(), position=policy(), row_start=7, row_count=4,
                               destination=destination, owner=second)
    validate(current, owner=second, generation=4, media_identity="e" * 64)
    assert old.proof.maximum == (9, 9, 9) and old.tensor.max() == 31
    with pytest.raises(ValueError, match="storage changed"):
        validate(old, owner=first)


def test_in_place_mutation_cannot_keep_an_old_coordinate_proof():
    issued = issue()
    issued.tensor.add_(100)
    with pytest.raises(ValueError, match="storage changed"):
        validate(issued)


def test_issuance_and_controlled_copies_track_versions_inside_inference_mode():
    class Frame:
        pass
    owner = Frame()
    with torch.inference_mode():
        source = issue()
        copied = copy_coordinates(source, source_position=policy(), position=policy(), row_start=7, row_count=4)
        assert not torch.is_inference(source.tensor) and not torch.is_inference(copied.tensor)
        validate(source)
        validate(copied)
        untracked = torch.zeros(4, 3, dtype=torch.int64)
        with pytest.raises(ValueError, match="track Tensor mutations"):
            copy_coordinates(source, source_position=policy(), position=policy(), row_start=7, row_count=4,
                             destination=untracked, owner=owner)
    assert torch.count_nonzero(untracked) == 0


def transfer(source, destination, **changes):
    options = dict(source=source, source_position=policy(), position=policy(), row_start=7, row_count=4,
                   destination=destination, profile_identity="a" * 64, generation=3, media_identity="b" * 64,
                   dtype=destination.dtype)
    options.update(changes)
    return CoordinateTransfer(**options)


def test_packed_batch_mints_every_segment_after_final_shared_version():
    class Frame:
        pass
    owner = Frame()
    packed = torch.empty(8, 3, dtype=torch.int64)
    first, second = issue(), issue(np.full((4, 3), 31, dtype=np.int64), generation=4, media_identity="e" * 64)
    issued = copy_coordinate_batch((transfer(first, packed[:4]),
                                   transfer(second, packed[4:], generation=4, media_identity="e" * 64)),
                                  owner=owner, max_rows=8)
    validate(issued[0], owner=owner)
    validate(issued[1], owner=owner, generation=4, media_identity="e" * 64)
    assert issued[0].tensor._version == issued[1].tensor._version == packed._version == 2
    validate(first)
    validate(second, generation=4, media_identity="e" * 64)


def test_packed_batch_validates_late_segments_and_overlaps_before_first_write():
    class Frame:
        pass
    owner = Frame()
    packed = torch.zeros(8, 3, dtype=torch.int64)
    source = issue()
    for second in (transfer(source, packed[4:], row_start=8), transfer(source, packed[3:7])):
        with pytest.raises(ValueError, match="outside|overlap"):
            copy_coordinate_batch((transfer(source, packed[:4]), second), owner=owner, max_rows=8)
        assert torch.count_nonzero(packed) == 0 and packed._version == 0


def test_packed_batch_refuses_writing_into_another_segments_immutable_source():
    class Frame:
        pass
    first, second = issue(), issue(np.full((4, 3), 31, dtype=np.int64))
    saved = second.tensor.clone()
    with pytest.raises(ValueError, match="immutable source"):
        copy_coordinate_batch((transfer(first, second.tensor), transfer(second, torch.empty_like(second.tensor))),
                              owner=Frame(), max_rows=8)
    torch.testing.assert_close(second.tensor, saved, rtol=0, atol=0)
    validate(first)
    validate(second)


def test_packed_batch_reserves_mutation_versions_before_writing():
    from tensorfold.cuda.sm70.tensor_mutation import MAX_MUTATION_VERSION

    class Frame:
        pass
    packed = torch.zeros(8, 3, dtype=torch.int64)
    torch._C._autograd._unsafe_set_version_counter([packed], [MAX_MUTATION_VERSION - 1])
    with pytest.raises(ValueError, match="version exhausted"):
        copy_coordinate_batch((transfer(issue(), packed[:4]), transfer(issue(), packed[4:])),
                              owner=Frame(), max_rows=8)
    assert torch.count_nonzero(packed) == 0 and packed._version == MAX_MUTATION_VERSION - 1
