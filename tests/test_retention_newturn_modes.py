# ruff: noqa: E402 - optional torch must be checked before CUDA-family imports.
"""Independent new-turn mode boundaries through actual CPU State transfers.

The existing tiny family fixtures exercise the real State and method adapters.
Their replicated CPU caches are component oracles, not SM70/TP8 qualification.
"""

import pytest
torch = pytest.importorskip("torch")

from tensorfold.cleanup import finish, rollback
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.media_mtp import MTPBoundary
from tensorfold.families.qwen4_exp.cuda.position_coordinates import copy_coordinates, issue_coordinates
from tensorfold.families.qwen4_exp.cuda.retained_catalog import (
    LogicalReference, RetentionCapacity, RetentionError, make_retention_catalog,
)
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
from tensorfold.families.qwen4_exp.cuda.state import State
from tensorfold.families.qwen4_exp.cuda.state_retention import (
    commit_state_restore,
    new_turn_gpu_snapshot,
    state_inventory,
    state_restore_plan,
)
from test_flashnext_dflash2_state import family as dflash_family  # noqa: F401 - fixture registration
from test_flashnext_media_mtp import MEDIA, PROFILE, bind_reference_inputs
from test_flashnext_mtp_inputs import family as mtp_family  # noqa: F401 - fixture registration


def states(request, method, enabled):
    fixture = request.getfixturevalue(method + "_family")
    weights = fixture[0] if method == "dflash" else fixture
    source, destination = (State(weights, 16, 4, "fp16", limit=16) for _ in range(2))
    for state, marker in ((source, 3), (destination, -7)):
        for cache in [*state.kc, *([state.mtp_kc] if hasattr(state, "mtp_kc") else [])]:
            for field in ("k", "v", "ks", "vs"):
                tensor = getattr(cache, field)
                if tensor is not None:
                    tensor.fill_(marker)
        for tensor in [state.rec, state.conv, state.ple_tail, *state.pooled]:
            tensor.fill_(marker)
        if hasattr(state, "mtp_pooled"):
            state.mtp_pooled.fill_(marker)
    if method == "mtp":
        for state, mode in ((source, enabled), (destination, False)):
            bind_reference_inputs(weights, state)
            state.media_inputs_enabled = True
            state.bind_media_context(PROFILE, MEDIA)
            state.media_mtp_enabled = mode
    source.pos = source.target_tentative_length = 4
    source.pos_dev.fill_(4)
    if method == "mtp":
        source.mtp_len = 4 if enabled else 0
        source.mtp_pos.fill_(source.mtp_len)
    else:
        window = source.dflash2
        if enabled:
            rows = torch.arange(2 * window.config.layers * 4 * window.kv.shape[-2] * window.kv.shape[-1])
            rows = rows.reshape(2, window.config.layers, 4, *window.kv.shape[-2:]).half()
            window.append(rows, 0)
        source.dflash2_runtime.set_enabled(source, enabled)
        destination.dflash2_runtime.set_enabled(destination, False)
    return source, destination


def request_for(source, *, enabled):
    participation = source.media_mtp_enabled if source.dflash2 is None else source.dflash2.enabled
    old = Stream([1] * 4, 8, draft=participation, st=source)
    fresh = Stream([1] * 4 + [2], 8, draft=enabled)
    return old, fresh


def deferred_hc(source):
    source.mtp_len = 3
    source.mtp_pos.fill_(3)
    issued = issue_coordinates(torch.tensor([[3, 3, 3]], dtype=torch.int64),
        position=source.media_mtp_position, profile_identity=PROFILE, generation=source.slot_generation,
        media_identity=MEDIA, row_start=3, max_rows=1, device="cpu")
    issued = copy_coordinates(issued, source_position=source.media_mtp_position,
                              position=source.media_mtp_position, row_start=3, row_count=1, dtype=torch.int64)
    hidden = torch.arange(source.mtp_hidden_width).reshape(1, -1).to(source.execution_dtype)
    boundary = MTPBoundary(3, hidden, issued, source.media_mtp_position, source.slot_generation)
    source.mtp_boundary = boundary
    return boundary


def transfer(source, destination, old, fresh, path):
    if path == "gpu-prefix":
        snapshot = new_turn_gpu_snapshot(source, destination, fresh)
        destination.copy_prefix(source, snapshot["pos"], snapshot["mtp_len"], copy_dflash=fresh.draft)
        destination.restore(snapshot)
        return None
    inventory = state_inventory(source, stream=old, recovery_versions=0)
    records = ((inventory.metadata, inventory.specs),)
    plan = state_restore_plan(destination, records, stream=fresh, continuation="new-turn", recovery_versions=0)
    for (_rank, number), part in plan.targets.items():
        part.tensor.copy_(inventory.parts[number].tensor)
    return commit_state_restore(plan)


def retire_catalog(owner, borrower, parent, target):
    if owner is None:
        return
    operations = []
    if borrower is not None and not borrower.closed:
        operations.append(borrower.close)
    for handle in (target, parent):
        if handle is not None and owner._handles.get(handle.serial) is handle:
            operations.append(lambda handle=handle: owner.release_owner(handle))
    operations.append(lambda: owner.close(timeout=5))
    finish(operations)


@pytest.mark.parametrize("method", ["mtp", "dflash"])
@pytest.mark.parametrize("path", ["gpu-prefix", "ram"])
@pytest.mark.parametrize("parent_enabled", [False, True])
def test_parent_to_serial_new_turn_copies_only_target_state(request, method, path, parent_enabled):
    source, destination = states(request, method, parent_enabled)
    old, fresh = request_for(source, enabled=False)
    source_k = source.kc[0].k[:4].clone()
    source_rec = source.rec[source.cur[0]].clone()
    handoff = transfer(source, destination, old, fresh, path)
    assert destination.pos == destination.pos_dev.item() == 4
    assert torch.equal(destination.kc[0].k[:4], source_k)
    assert torch.equal(destination.rec[0], source_rec)
    assert torch.equal(source.kc[0].k[:4], source_k)
    assert source.pos == 4 and fresh.draft is False and fresh.out == fresh.drafts == []
    assert destination.mtp_len == destination.mtp_drafted == 0 and destination.mtp_boundary is None
    if method == "mtp":
        assert destination.mtp_pos.item() == 0 and not destination.media_mtp_enabled
        assert torch.all(destination.mtp_kc.k == -7)  # No parent draft cache enters a serial request.
    else:
        assert not destination.dflash2.enabled
        assert destination.dflash2.end == destination.dflash2.available == 0
        assert source.dflash2.enabled is parent_enabled and source.dflash2.end == (4 if parent_enabled else 0)
    if handoff is not None:
        assert handoff["pending_hc"] is None and handoff["draws"] is None and handoff["recoveries"] == ()


@pytest.mark.parametrize("path", ["gpu-prefix", "ram"])
def test_serial_new_turn_discards_parent_deferred_media_hc_without_relabeling_source(request, path):
    source, destination = states(request, "mtp", True)
    boundary = deferred_hc(source)
    hidden = boundary.hidden.clone()
    old, fresh = request_for(source, enabled=False)
    transfer(source, destination, old, fresh, path)
    assert destination.pos == 4 and destination.mtp_len == destination.mtp_pos.item() == 0
    assert destination.mtp_boundary is None and not destination.media_mtp_enabled
    assert source.mtp_boundary is boundary and source.mtp_len == 3
    assert torch.equal(boundary.hidden, hidden) and torch.equal(boundary.coordinates, torch.tensor([[3, 3, 3]]))


@pytest.mark.parametrize("method", ["mtp", "dflash"])
@pytest.mark.parametrize("path", ["gpu-prefix", "ram"])
def test_cold_parent_to_enabled_new_turn_refuses_before_destination_writes(request, method, path):
    source, destination = states(request, method, False)
    destination.media_mtp_enabled = method == "mtp"
    if method == "dflash":
        destination.dflash2_runtime.set_enabled(destination, True)
    old, fresh = request_for(source, enabled=True)
    before = destination.kc[0].k.clone(), destination.rec.clone()
    with pytest.raises(ValueError):
        transfer(source, destination, old, fresh, path)
    assert torch.equal(destination.kc[0].k, before[0]) and torch.equal(destination.rec, before[1])
    assert destination.pos == destination.pos_dev.item() == 0
    assert destination._kv_error is None and destination._history_error is None


@pytest.mark.parametrize("method", ["mtp", "dflash"])
@pytest.mark.parametrize("path", ["gpu-prefix", "ram"])
def test_enabled_parent_to_enabled_new_turn_preserves_committed_method_prefix(request, method, path):
    source, destination = states(request, method, True)
    destination.media_mtp_enabled = method == "mtp"
    if method == "dflash":
        destination.dflash2_runtime.set_enabled(destination, True)
    old, fresh = request_for(source, enabled=True)
    transfer(source, destination, old, fresh, path)
    assert destination.pos == 4 and fresh.out == fresh.drafts == []
    if method == "mtp":
        assert destination.mtp_len == destination.mtp_pos.item() == 4 and destination.mtp_drafted == 0
        assert torch.equal(destination.mtp_kc.k[:4], source.mtp_kc.k[:4])
    else:
        assert destination.dflash2.enabled and destination.dflash2.end == 4
        assert torch.equal(destination.dflash2.snapshot()["kv"], source.dflash2.snapshot()["kv"])


@pytest.mark.parametrize("explicit", [False, True])
def test_ordinary_dflash_prefix_copy_keeps_enabled_destination_requirement(request, explicit):
    source, destination = states(request, "dflash", True)
    before = destination.kc[0].k.clone()
    with pytest.raises(ValueError, match="valid enabled method state"):
        destination.copy_prefix(source, 4, 0, **({"copy_dflash": True} if explicit else {}))
    assert torch.equal(destination.kc[0].k, before) and destination.pos == 0


@pytest.mark.parametrize("mode", ["serial-new-turn", "enabled-new-turn", "same-request"])
def test_catalog_coordinate_quota_uses_current_method_participation(request, mode):
    from types import SimpleNamespace

    source, destination = states(request, "mtp", True)
    boundary = deferred_hc(source)
    old, fresh = request_for(source, enabled=mode != "serial-new-turn")
    destination.media_mtp_enabled = mode != "serial-new-turn"
    catalog = borrower = parent = target = None
    owners = []

    def retire():
        owner = catalog if catalog is not None else next(iter(owners), None)
        retire_catalog(owner, borrower, parent, target)

    try:
        # Only the actual transfer chunk pair is admitted. A serial request
        # restores no coordinate proof; enabled/current-request HC still needs
        # its real additional host/device coordinate storage.
        policy = RetentionPolicy("ram", 2, 30, 128 << 20, 37, 74, 0)
        catalog = make_retention_catalog(policy, SimpleNamespace(meta={"rank": 0}, comm=None),
            [source, destination], prefix_limit=0, recovery_versions=0, register=owners.append)
        parent = catalog.reserve(LogicalReference("trusted", "response:parent", 1),
                                 future_bytes=catalog.future_claim())
        catalog.capture_idle(old, handle=parent)
        saved = catalog.suspend(parent)
        assert saved.retired.wait(5) and catalog.saved(parent), saved.failure
        if mode == "same-request":
            borrower = catalog.borrow(parent)
            current = old
        else:
            target = catalog.reserve(LogicalReference("trusted", "response:fresh", 1),
                                     future_bytes=catalog.future_claim())
            borrower = catalog.borrow(parent, continuation="new-turn", target_handle=target)
            current = fresh
        before = destination.kc[0].k.clone()
        if mode != "serial-new-turn":
            with pytest.raises(RetentionError, match="coordinate restore exceeds"):
                borrower.claim(destination, stream=current)
            assert torch.equal(destination.kc[0].k, before) and destination.pos == 0
        else:
            assert borrower.claim(destination, stream=current) is None
            assert borrower.restore_handle.transfer.retired.wait(5)
            handoff = catalog.restored(borrower)
            assert handoff is not None
            assert destination.pos == 4 and destination.mtp_len == destination.mtp_pos.item() == 0
            assert destination.mtp_boundary is None and handoff["pending_hc"] is None
            assert torch.equal(destination.kc[0].k[:4], source.kc[0].k[:4])
        assert source.mtp_boundary is boundary and source.mtp_len == 3
    except BaseException as primary:
        rollback(owners, primary, retire)
    else:
        retire()


@pytest.mark.parametrize("missing_hc_byte", [False, True])
def test_gpu_only_media_fork_uses_paid_snapshot_and_hc_pools_without_ram_phase_scratch(request, missing_hc_byte):
    from types import SimpleNamespace

    source, destination = states(request, "mtp", True)
    original = deferred_hc(source)
    destination.media_mtp_enabled = True
    old, fresh = request_for(source, enabled=True)
    # Independent actual single-version payload: recurrence, convolution,
    # PLE tail and the deferred hidden/int64-coordinate row. No CPU→int32
    # issue buffer exists in this issued-coordinate GPU-prefix path.
    snapshot_bytes = (source.rec[0].nbytes + source.conv.nbytes + source.ple_tail.nbytes
                      + original.hidden.nbytes + original.coordinates.nbytes)
    hc_bytes = original.hidden.nbytes + original.coordinates.nbytes
    catalog = borrower = parent = target = None
    owners = []

    def retire():
        owner = catalog if catalog is not None else next(iter(owners), None)
        retire_catalog(owner, borrower, parent, target)

    try:
        catalog = make_retention_catalog(RetentionPolicy("gpu", 2, 30),
            SimpleNamespace(meta={"rank": 0}, comm=None), [source, destination],
            prefix_limit=1, recovery_versions=0, register=owners.append)
        catalog.configure_device_resources(recovery_pool_bytes=2 * snapshot_bytes,
            pending_hc_bytes=2 * hc_bytes - int(missing_hc_byte), snapshot_bytes=snapshot_bytes)
        parent = catalog.reserve(LogicalReference("trusted", "response:gpu-parent", 1),
                                 future_bytes=catalog.future_claim())
        catalog.capture_idle(old, handle=parent)
        target = catalog.reserve(LogicalReference("trusted", "response:gpu-fork", 1),
                                 future_bytes=catalog.future_claim())
        borrower = catalog.borrow(parent, continuation="new-turn", target_handle=target)
        before = destination.kc[0].k.clone()
        if missing_hc_byte:
            with pytest.raises(RetentionCapacity):
                borrower.claim(destination, stream=fresh)
            assert destination.pos == 0 and torch.equal(destination.kc[0].k, before)
        else:
            handoff = borrower.claim(destination, stream=fresh)
            assert handoff is not None and handoff["forked"] and catalog.policy.overlap_bytes is None
            assert destination.pos == 4 and destination.mtp_len == 3
            assert torch.equal(destination.kc[0].k[:4], source.kc[0].k[:4])
            assert destination.mtp_boundary is not original
            assert torch.equal(destination.mtp_boundary.hidden, original.hidden)
            assert torch.equal(destination.mtp_boundary.coordinates, original.coordinates)
        assert source.mtp_boundary is original and source.mtp_len == 3
    except BaseException as primary:
        rollback(owners, primary, retire)
    else:
        retire()
