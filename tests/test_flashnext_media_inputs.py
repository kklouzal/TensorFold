"""Merged visual rows retain request identity and bounded consumer ownership."""

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.media_inputs import ReadyInputSpan


def prepared(**changes):
    fields = dict(start=7, token_ids=(10, 99, 99, 11), values=torch.arange(32).reshape(4, 8).half(),
                  positions=torch.tensor([[7, 7, 7], [8, 3, 4], [8, 3, 5], [9, 9, 9]], dtype=torch.int32),
                  rope_delta=-1, profile_identity="a" * 64, generation=3, media_identity="b" * 64,
                  max_rows=4, hidden_size=8, vocab_size=100, roles=("target", "mtp"))
    fields.update(changes)
    return ReadyInputSpan(**fields)


def borrow(span, role, **changes):
    identity = dict(profile_identity="a" * 64, generation=3, media_identity="b" * 64)
    identity.update(changes)
    return span.borrow(role, **identity)


def test_same_visual_ids_select_distinct_real_embeddings_and_preserve_coordinates():
    span = prepared()
    target = borrow(span, "target")
    mtp = borrow(span, "mtp")
    visual = mtp.rows(8, (99, 99))
    assert visual.data_ptr() == span.values[1:].data_ptr()
    assert not torch.equal(visual[0], visual[1])
    assert span.token_ids == (10, 99, 99, 11) and span.end == 11
    assert span.positions.tolist() == [[7, 7, 7], [8, 3, 4], [8, 3, 5], [9, 9, 9]]
    # Several slices use one consumer lease; the caller owns its hidden-row RoPE.
    torch.testing.assert_close(mtp.rows(10, (11,)), target.rows(10, [11]))
    target.complete()
    with pytest.raises(ValueError, match="not completed"):
        span.retire()
    mtp.complete()
    span.retire()
    assert span.closed and span.values is None and span.positions is None and span._storage is None
    with pytest.raises(ValueError, match="retired"):
        mtp.rows(8, (99,))
    span.retire()


@pytest.mark.parametrize("changes", [dict(profile_identity="c" * 64), dict(generation=4),
                                    dict(generation=3.), dict(media_identity="c" * 64), dict(media_identity=None)])
def test_stale_request_identity_cannot_take_consumer_or_change_span(changes):
    span = prepared()
    with pytest.raises(ValueError, match="another request"):
        borrow(span, "target", **changes)
    assert all(v is None for v in span._leases.values())
    span.retire(cancelled=True)


@pytest.mark.parametrize("start,ids", [(6, (10,)), (11, (11,)), (8, (10,)), (8, (99, 11)), (8, ()),
                                     (7, (10.,)), (7, (True,))])
def test_consumer_rejects_missing_lookahead_wrong_offset_or_wrong_original_ids(start, ids):
    span = prepared()
    lease = borrow(span, "mtp")
    with pytest.raises(ValueError, match="original expanded"):
        lease.rows(start, ids)
    lease.complete()
    span.retire(cancelled=True)


def test_consumers_are_bounded_and_cancellation_waits_for_existing_staging():
    span = prepared()
    target = borrow(span, "target")
    for role in ("target", "unexpected"):
        with pytest.raises(ValueError, match="undeclared or already borrowed"):
            borrow(span, role)
    with pytest.raises(ValueError, match="not completed"):
        span.retire(cancelled=True)
    target.complete()
    target.complete()
    with pytest.raises(ValueError, match="already completed"):
        target.rows(7, (10,))
    span.retire(cancelled=True)
    assert span.closed


def test_storage_rebinding_poison_retains_original_storage_for_fenced_retirement():
    span = prepared()
    old = span._storage
    lease = borrow(span, "target")
    span.values.set_(torch.zeros_like(span.values))
    with pytest.raises(ValueError, match="storage changed"):
        lease.rows(7, (10,))
    assert span.poisoned and not span.closed and span._storage is old
    with pytest.raises(ValueError, match="poisoned"):
        span.retire(cancelled=True)


def test_coordinates_on_another_device_are_refused_before_event_creation():
    with pytest.raises(ValueError, match="share the value device"):
        prepared(positions=torch.empty((4, 3), dtype=torch.int32, device="meta"))


def test_cancellation_flag_cannot_skip_an_unused_consumer_through_coercion():
    span = prepared()
    with pytest.raises(ValueError, match="explicit boolean"):
        span.retire(cancelled="false")
    assert not span.closed
    span.retire(cancelled=True)


@pytest.mark.parametrize("changes", [dict(max_rows=3), dict(start=-1), dict(start=2**63 - 2),
                                    dict(generation=True), dict(profile_identity="model-name"),
                                    dict(media_identity="content-name"), dict(token_ids=(10, True, 99, 11)),
                                    dict(values=torch.zeros(4, 8)), dict(positions=torch.zeros(4, 3)),
                                    dict(roles=("target", "mtp", "extra"))])
def test_prepared_span_rejects_unadmitted_geometry_identity_and_representations(changes):
    with pytest.raises(ValueError):
        prepared(**changes)
