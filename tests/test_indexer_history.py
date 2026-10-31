"""RAM-history journal/lifecycle proof without CUDA or actual NUMA syscalls."""
from contextlib import contextmanager
import ctypes

import pytest
import torch

from tensorfold.cuda.numa_memory import backing_bytes
from tensorfold.families.qwen4_exp.cuda.indexer_history import HistoryPlan, HistorySession


class Target:
    def __init__(self):
        self.depth = self.entries = 0

    @contextmanager
    def scope(self):
        self.depth += 1
        self.entries += 1
        try:
            yield self
        finally:
            self.depth -= 1


class Buffer:
    def __init__(self, tensor):
        self.tensor = tensor
        self.closed = False
        self.fail = False

    def close(self):
        if self.fail:
            self.fail = False
            raise RuntimeError("controlled host retirement failure")
        self.closed = True
        self.tensor = None


class Allocator:
    def __init__(self, fail_at=None):
        self.target = Target()
        self.buffers = []
        self.fail_at = fail_at
        self.closed = False

    def __call__(self, shape, dtype, *, pinned=False, tag=""):
        resource = Buffer(torch.zeros(shape, dtype=dtype))
        self.buffers.append(resource)
        if len(self.buffers) == self.fail_at:
            raise RuntimeError("controlled allocation failure")
        return resource

    def close(self):
        for resource in self.buffers:
            resource.close()
        self.closed = True


@pytest.fixture
def session():
    value = HistorySession(HistoryPlan(3, 4, 4, 8, 2), Allocator(), "cpu", maximum_owners=6)
    try:
        yield value
    finally:
        value.close()


def data(start, rows, layer, dim=4):
    return (torch.arange(start, start + rows)[:, None] * 4
            + torch.arange(dim)[None] + layer * 256).to(torch.float16)


def produce(owner, start, rows, *, kind="prefill", layers=(0, 1, 2), validate=True):
    frame = owner.begin(kind, start, rows, layers=layers)
    for layer in layers:
        offset = start - int(owner.origins_host[layer])
        frame.stage(layer)[offset:offset + rows].copy_(data(start, rows, layer))
    frame.enqueue(layers)
    frame.mark_submitted()
    if validate:
        frame.validated(fenced=True)
    return frame


def test_plan_exact_backing_and_context_independent_device_storage():
    plan = HistoryPlan(13, 128, 4, 2048, 8)
    assert plan.ring_rows == 2052 and plan.decode_stage_rows == 16
    assert plan.device_nbytes == plan.device_bytes_per_layer * 13
    for capacity in (256, 262148):
        expected = sum(backing_bytes(2 * 13 * rows * 128) for rows in
                       (capacity, 2048, 16, 2052)) + backing_bytes(8 * 13)
        assert plan.host_bytes(capacity) == expected
    assert plan.pinned_bytes == sum(-(-size // plan.page_size) * plan.page_size for size in
        (2 * 13 * 2048 * 128, 2 * 13 * 16 * 128, 2 * 13 * 2052 * 128, 8 * 13))


@pytest.mark.parametrize("name,value", [("layers", 0), ("index_dim", -1), ("ratio", True),
                                        ("prefill_rows", 0), ("decode_rows", 1.5)])
def test_plan_refuses_invalid_dimensions(name, value):
    values = dict(layers=3, index_dim=4, ratio=4, prefill_rows=8, decode_rows=2)
    values[name] = value
    with pytest.raises(ValueError):
        HistoryPlan(**values)


def test_owner_actual_bytes_and_layer_handles_match_plan(session):
    owner = session.new(64)
    assert owner.device_nbytes == session.plan.device_nbytes
    assert owner.host_nbytes == session.plan.host_bytes(64)
    assert sum(t.numel() * t.element_size() for t in owner.device_tensors()) == owner.device_nbytes
    handle = owner.layer(1)
    assert handle.ring.data_ptr() == owner.ring[1].data_ptr()
    assert handle.stage("decode").shape == (4, 4)
    assert handle.origin.shape == (1,) and handle.origin.dtype == torch.int64


def test_validation_is_required_before_history_can_be_snapshotted(session):
    owner = session.new(64)
    frame = produce(owner, 0, 8, validate=False)
    with pytest.raises(ValueError, match="unvalidated"):
        owner.snapshot([8] * 3)
    with pytest.raises(ValueError, match="fence"):
        frame.validated(fenced=False)
    frame.validated(fenced=True)
    frame.commit(3)
    assert owner.snapshot([3] * 3)["lengths"] == (3, 3, 3)
    assert owner._published == [3] * 3
    for index in range(3):
        torch.testing.assert_close(owner.archive[index, :8], data(0, 8, index), rtol=0, atol=0)


def test_boolean_fence_cannot_publish_unsubmitted_or_incomplete_dma(session):
    owner = session.new(64)
    frame = owner.begin("prefill", 0, 2)
    frame.gpu.fill_(3)
    frame.enqueue()
    with pytest.raises(RuntimeError, match="DMA"):
        frame.validated(fenced=True)
    frame.mark_submitted()
    frame.event = type("Pending", (), {"query": lambda _: False})()
    with pytest.raises(RuntimeError, match="DMA"):
        frame.validated(fenced=True)
    frame.event = type("Ready", (), {"query": lambda _: True})()
    frame.validated(fenced=True)
    assert owner.snapshot([2] * 3)["lengths"] == (2, 2, 2)


def test_main_and_mtp_journals_have_independent_origins_and_cursors(session):
    owner = session.new(64)
    produce(owner, 0, 8, layers=(0, 1))
    produce(owner, 0, 7, layers=(2,))
    owner.commit([8, 8, 7])
    produce(owner, 7, 2, kind="decode", layers=(2,), validate=False)
    assert int(owner.layer(2).origin[0]) == 7
    frame = produce(owner, 9, 2, kind="decode", layers=(2,), validate=False)
    assert int(owner.layer(2).origin[0]) == 7
    frame.validated(fenced=True)
    owner.commit([8, 8, 10])
    torch.testing.assert_close(owner.archive[2, :11], data(0, 11, 2), rtol=0, atol=0)
    assert owner.snapshot([8, 8, 10])["lengths"] == (8, 8, 10)


def test_main_lease_selection_survives_an_intervening_mtp_phase(session):
    owner = session.new(64)
    main = produce(owner, 0, 8, layers=(0, 1))
    mtp = produce(owner, 0, 7, layers=(2,))
    mtp.commit(6)
    main.commit(8)
    assert owner._published == [8, 8, 6]


def test_reset_invalidates_old_frame_leases_without_erasing_restorable_archive(session):
    owner = session.new(64)
    frame = produce(owner, 0, 8)
    metadata = owner.snapshot([8] * 3)
    owner.reset()
    with pytest.raises(RuntimeError, match="stale"):
        frame.commit(1)
    owner.restore(metadata, [8] * 3)


def test_origins_mirror_cannot_be_rewritten_while_its_upload_is_pending(session):
    owner = session.new(64)
    owner._origin_event = type("Pending", (), {"query": lambda _: False})()
    before = owner.origins_host.clone()
    with pytest.raises(RuntimeError, match="still borrowed"):
        owner.begin("decode", 0, 1)
    assert not any(owner._pending.values()) and torch.equal(before, owner.origins_host)


def test_bounded_stage_refuses_a_gap_or_overflow_before_journal_mutation(session):
    owner = session.new(64)
    produce(owner, 0, 8, layers=(0,))
    before = owner.origins.clone(), owner.origins_host.clone()
    with pytest.raises(ValueError, match="gap"):
        owner.begin("decode", 8, 1, layers=(0, 1))
    assert not any(owner._pending.values())
    assert torch.equal(owner.origins, before[0]) and torch.equal(owner.origins_host, before[1])
    produce(owner, 8, 4, kind="decode", layers=(0,), validate=False)
    old = dict(owner._pending["decode"])
    with pytest.raises(ValueError, match="bounded"):
        owner.begin("decode", 12, 1, layers=(0,))
    assert owner._pending["decode"] == old
    owner.abort(fenced=True)


def test_ring_bound_retains_carry_for_every_partial_commit():
    plan = HistoryPlan(1, 1, 4, 8, 2)
    for position in range(plan.ring_rows, 3 * plan.ring_rows):
        for rows in range(1, 9):
            ring = {row % plan.ring_rows: row for row in range(position - plan.ring_rows, position)}
            ring.update({row % plan.ring_rows: row for row in range(position, position + rows)})
            for keep in range(rows + 1):
                end = position + keep
                for required in range(end // plan.ratio * plan.ratio, end):
                    assert ring[required % plan.ring_rows] == required


def test_arbitrary_prefix_restore_rebuilds_modulo_ring_and_reset_retains_bytes(session):
    owner = session.new(64)
    produce(owner, 0, 8)
    produce(owner, 8, 8)
    produce(owner, 16, 8)
    owner.commit([24] * 3)
    metadata = owner.snapshot([17, 5, 21])
    owner.reset()
    owner.restore(metadata, [17, 5, 21])
    for layer, end in enumerate((17, 5, 21)):
        for row in range(max(0, end-session.plan.ring_rows), end):
            torch.testing.assert_close(owner.ring[layer, row % session.plan.ring_rows],
                                       data(row, 1, layer)[0], rtol=0, atol=0)


def test_prefix_copy_clone_and_rewind_preserve_independent_ownership(session):
    source = session.new(64)
    produce(source, 0, 8)
    source.commit([3] * 3)
    cloned = source.clone()
    assert cloned._published == [3] * 3 and cloned._available == [8] * 3
    assert cloned.archive.data_ptr() != source.archive.data_ptr()
    other = session.new(32)
    other.copy_prefix(source, [7, 4, 2])
    for layer, count in enumerate((7, 4, 2)):
        torch.testing.assert_close(other.archive[layer, :count], data(0, count, layer), rtol=0, atol=0)
    other.restore(other.snapshot([3, 3, 1]), [3, 3, 1])
    produce(other, 3, 2, kind="decode", layers=(0,))
    assert source.snapshot([8] * 3)["lengths"] == (8, 8, 8)


def test_foreign_snapshot_or_missing_prefix_is_rejected_before_restore(session):
    owner = session.new(64)
    produce(owner, 0, 8)
    metadata = owner.snapshot([7] * 3)
    old = owner.ring.clone()
    with pytest.raises(ValueError, match="snapshot"):
        owner.restore({**metadata, "index_dim": 9}, [7] * 3)
    with pytest.raises(ValueError, match="unvalidated"):
        owner.snapshot([9] * 3)
    assert torch.equal(old, owner.ring)


def test_archive_memcpy_runs_only_inside_local_scope_and_never_waits(monkeypatch, session):
    owner = session.new(64)
    original = ctypes.memmove
    calls = []

    def checked(*args):
        assert session.allocator.target.depth == 1
        calls.append(args[-1])
        return original(*args)

    monkeypatch.setattr(ctypes, "memmove", checked)
    monkeypatch.setattr(owner, "_synchronize", lambda: pytest.fail("validated archival must not add a wait"))
    produce(owner, 0, 8)
    assert calls == [8 * 4 * 2] * 3


def test_partial_allocation_and_failed_retirement_remain_journaled():
    allocator = Allocator(fail_at=3)
    session = HistorySession(HistoryPlan(2, 4, 4, 8, 2), allocator, "cpu", maximum_owners=2)
    with pytest.raises(RuntimeError, match="allocation failure"):
        session.new(32)
    assert len(session._owners) == 1
    session.close()
    assert session.closed and allocator.closed and all(buffer.closed for buffer in allocator.buffers)
    allocator = Allocator()
    session = HistorySession(HistoryPlan(2, 4, 4, 8, 2), allocator, "cpu", maximum_owners=2)
    owner = session.new(32)
    allocator.buffers[0].fail = True
    with pytest.raises(RuntimeError, match="retirement failure"):
        session.close()
    assert not owner.closed and not session.closed and owner.host_nbytes > 0
    session.close()
    assert owner.closed and session.closed and owner.device_nbytes == owner.host_nbytes == 0


def test_bounded_catalogue_and_closed_owners_reject_new_work(session):
    owners = [session.new(16) for _ in range(session.maximum_owners)]
    with pytest.raises(RuntimeError, match="catalogue"):
        session.new(16)
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        owners[0].begin("decode", 0, 1)
    with pytest.raises(RuntimeError, match="closed"):
        session.new(16)
