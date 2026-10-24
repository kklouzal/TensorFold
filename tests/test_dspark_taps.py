"""Pending HC storage boundaries, owned tap features and accepted-prefix leases."""
from types import SimpleNamespace as NS
import weakref

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.families.qwen4_exp.cuda.dspark_taps import TapWorkspace, byte_plan  # noqa: E402


def config(hidden=256, before=(2, 13, 24, 35, 46)):
    return NS(hidden=hidden, taps=tuple(index - 1 for index in before), capture_before=before)


def state(position=0):
    return NS(pos=position, version=0, kv_identity=("fp16", "int8"), cur=[0], _kv_error=None, _resize_error=None)


def bits(a):
    return a.contiguous().view(torch.int16 if a.dtype == torch.float16 else torch.int32)


def materialize(h, pending):
    """Independent NumPy FP32 operations with the target's explicit Half stores."""
    if pending is None:
        return h.clone()
    mode, value, weights, inject = pending
    if mode == 2:
        slots = value.numpy().astype(np.float32)
        w = weights.numpy().astype(np.float32)
        total = np.zeros((h.shape[0], h.shape[1] // 4), dtype=np.float32)
        for slot in range(slots.shape[1]):
            total = np.add(total, np.multiply(slots[:, slot], w[:, slot, None], dtype=np.float32), dtype=np.float32)
    elif mode in (3, 4):
        values = value.numpy().astype(np.float32)
        total = values[0].copy()
        for rank in range(1, len(values)):
            total = np.add(total, values[rank], dtype=np.float32)
    else:
        total = value.numpy()
    update = total.astype(np.float16).astype(np.float32)
    addition = (update[:, None] * inject.numpy().astype(np.float32)[:, :, None]).astype(np.float16)
    result = (h.numpy().reshape(h.shape[0], 4, -1).astype(np.float32)
              + addition.astype(np.float32)).astype(np.float16).reshape(h.shape)
    return torch.from_numpy(result)


@pytest.mark.parametrize("rows", [1, 3, 18])
@pytest.mark.parametrize("mode", [None, 1, 2, 3, 4])
def test_pending_safe_features_match_materialized_oracle_without_touching_target_operands(rows, mode):
    torch.manual_seed(224)
    hidden = 2560
    h = (torch.randn(rows, hidden * 4) * .7).half()
    inject = torch.rand(rows, 4).half()
    pending = None
    if mode is not None:
        shape = (rows, 11, hidden) if mode == 2 else ((8, rows, hidden) if mode in (3, 4) else (rows, hidden))
        value = torch.randn(shape) * .2
        weights = torch.rand(rows, 11) if mode == 2 else None
        pending = mode, value, weights, inject
    inputs = [h] + ([] if pending is None else [p for p in pending[1:] if p is not None])
    original = [p.clone() for p in inputs]
    workspace = TapWorkspace(config(hidden), rows, "cpu")
    capture = workspace.begin(rows)
    expected = materialize(h, pending).view(rows, 4, hidden).float().mean(1).half()
    for index in workspace.before:
        capture.before_layer(index, h, pending)
    batch = capture.finish()
    for slot in range(5):
        torch.testing.assert_close(bits(batch.features.view(rows, 5, hidden)[:, slot]), bits(expected), rtol=0, atol=0)
    for actual, previous in zip(inputs, original):
        torch.testing.assert_close(bits(actual), bits(previous), rtol=0, atol=0)
    batch.release()
    workspace.close()


def test_negative_pending_and_current_ple_boundaries_are_distinct_and_features_are_owned():
    rows, hidden = 3, 256
    workspace = TapWorkspace(config(hidden, (2,)), rows, "cpu")
    h = torch.full((rows, 4 * hidden), .25, dtype=torch.float16)
    pending = (1, torch.ones(rows, hidden), None, torch.ones(rows, 4, dtype=torch.float16))
    capture = workspace.begin(rows)
    capture.before_layer(2, h, pending)
    batch = capture.finish()
    assert torch.all(batch.features == 1.25)
    stale = h.view(rows, 4, hidden).float().mean(1).half()
    assert not torch.equal(batch.features, stale)
    h.add_(7)  # represent the current layer's later PLE contribution
    assert not torch.equal(batch.features, h.view(rows, 4, hidden).float().mean(1).half())
    assert torch.all(batch.features == 1.25)
    batch.release()


def test_byte_plan_is_exact_for_owned_storage_and_explicitly_bounds_2048_rows():
    c = config(2560)
    owner = TapWorkspace(c, 2048, "cpu", packed_rows=18)
    plan = byte_plan(c, 2048, packed_rows=18)
    assert owner.nbytes == plan.device_bytes == 115671040 + 18 * (5 * 2560 * 2 + 8)
    assert plan.host_bytes == owner.host_indices.numel() * owner.host_indices.element_size() == 144
    assert plan.transient_bytes == 2048 * 2560 * 20
    assert plan.admission_bytes == plan.device_bytes + plan.transient_bytes
    owner.close()
    for rows in (0, 2049, True):
        with pytest.raises(ValueError, match="2048"):
            byte_plan(c, rows)


def test_capture_order_completeness_duplicate_alias_and_feature_lease_guards():
    owner = TapWorkspace(config(), 3, "cpu")
    h = torch.zeros(3, 1024, dtype=torch.float16)
    capture = owner.begin(3)
    with pytest.raises(ValueError, match="incomplete"):
        capture.finish()
    with pytest.raises(ValueError, match="alias"):
        capture.before_layer(2, owner.materialized)
    capture.before_layer(1, h)
    capture.before_layer(2, h)
    with pytest.raises(ValueError, match="duplicate"):
        capture.before_layer(2, h)
    for index in owner.before[1:]:
        capture.before_layer(index, h)
    batch = capture.finish()
    with pytest.raises(ValueError, match="borrowed"):
        owner.begin(1)
    with pytest.raises(ValueError, match="active"):
        owner.close()
    batch.release()
    with pytest.raises(ValueError, match="released"):
        _ = batch.features
    again = owner.begin(1)
    again.abort()
    owner.close()


def captured(owner, rows, segments=()):
    h = torch.arange(rows, dtype=torch.float16)[:, None].expand(-1, 4 * owner.hidden).contiguous()
    capture = owner.begin(rows, segments)
    for index in owner.before:
        capture.before_layer(index, h)
    return capture.finish()


def test_six_lane_compaction_preserves_each_owner_prefix_and_rejects_mislabeled_rows():
    owner = TapWorkspace(config(), 18, "cpu", packed_rows=18)
    states = [state(index) for index in range(6)]
    source = [(st, 3 * index, 3 * index + 3) for index, st in enumerate(states)]
    batch = captured(owner, 18, source)
    original = batch.features.clone()
    counts, targets, indices, at = [1, 3, 2, 1, 2, 3], [], [], 0
    for st, count in zip(reversed(states), counts):
        targets.append((st, at, at + count))
        begin = states.index(st) * 3
        indices.extend(range(begin, begin + count))
        at += count
    batch.compact(indices, targets)
    torch.testing.assert_close(batch.features, original[indices], rtol=0, atol=0)
    for (st, _, _), count in zip(targets, counts):
        assert batch.segment(st).shape == (count, 1280)
    batch.release()
    batch = captured(owner, 18, source)
    with pytest.raises(ValueError, match="leading prefix"):
        batch.compact([3], [(states[0], 0, 1)])
    with pytest.raises(ValueError, match="ownership"):
        batch.compact([0], ())
    batch.release()


def test_accepted_prefix_access_and_cache_version_identity_do_not_expose_rejected_rows():
    owner, target = TapWorkspace(config(), 3, "cpu"), state(7)
    batch = captured(owner, 3, [(target, 0, 3)])
    target.pos = 9
    with pytest.raises(ValueError, match="accepted-prefix"):
        _ = batch.features
    with pytest.raises(ValueError, match="committed"):
        batch.segment(target)
    assert batch.segment(target, accepted=2).shape == (2, 1280)
    with pytest.raises(ValueError, match="committed"):
        batch.segment(target, accepted=3)
    target.version = 1
    with pytest.raises(ValueError, match="target storage"):
        batch.segment(target, accepted=2)
    target.version = 0
    batch.release()


def test_cpu_capture_cannot_claim_a_cuda_graph_plan():
    owner = TapWorkspace(config(before=(2,)), 1, "cpu")
    capture = owner.begin(1)
    capture.before_layer(2, torch.zeros(1, 1024, dtype=torch.float16))
    with pytest.raises(ValueError, match="actual CUDA capture"):
        capture.finish_capture(NS(replay=lambda: None))
    capture.abort()


def test_target_numeric_failure_or_storage_invalidation_never_blocks_lease_cleanup():
    owner, target = TapWorkspace(config(), 3, "cpu"), state(7)
    batch = captured(owner, 3, [(target, 0, 3)])
    target._kv_error = RuntimeError("target failed")
    with pytest.raises(ValueError, match="target storage"):
        _ = batch.features
    batch.release()
    target._kv_error = None
    capture = owner.begin(3, [(target, 0, 3)])
    target.version += 1
    with pytest.raises(ValueError, match="target storage"):
        capture.finish()
    capture.abort()
    owner.close()
    assert owner.nbytes == 0


def test_same_cursor_reset_or_restore_invalidates_old_features_without_state_mutation():
    owner, target = TapWorkspace(config(), 3, "cpu"), state()
    batch = captured(owner, 3, [(target, 0, 3)])
    target.cur = [0]  # State.reset/restore replace this list, even with unchanged storage and cursor.
    with pytest.raises(ValueError, match="target storage"):
        _ = batch.features
    batch.release()
    current = captured(owner, 3, [(target, 0, 3)])
    assert current.features.shape == (3, 1280)
    current.release()


def test_released_feature_batch_does_not_keep_retired_device_storage_alive():
    owner = TapWorkspace(config(), 3, "cpu")
    storage = weakref.ref(owner.features)
    batch = captured(owner, 3)
    batch.release()
    owner.close()
    assert storage() is None
