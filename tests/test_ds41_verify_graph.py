"""DSpark graph-owner boundaries, with independent target and head witnesses."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v4.cuda.dspark_graph import SpecShape, VerifyCommit, VerifyEntry
from tensorfold.families.deepseek_v4.cuda.graphs import SlotPages


def owners(device):
    events = []
    slots, width, context = 6, 3, 16_384
    scratch = slots - 1
    pages = SlotPages(slots, width, device)
    owned = {0: [0, 1, 2], 2: [9, 10, 11], 3: [12]}
    for slot, row in owned.items():
        pages.set(slot, row)

    class Target:
        units = 80
        index_context = context
        g = SimpleNamespace(hidden=2)
        hash_state = SimpleNamespace(max_tokens=16)
        pool = SimpleNamespace(owned=owned)
        w = SimpleNamespace(head=object())

        def begin(self, step):
            events.append(("begin", self.index_context, step.qsl_cpu))
            self.last = step
            return step

        @staticmethod
        def layers_with_draft_taps(step, stop, tap_rows, out):
            assert stop == Target.units
            assert step.qsl.numel() == step.num_reqs + 1
            values = (step.ids + step.positions).half()
            out.copy_(values.index_select(0, tap_rows)[:, None].expand_as(out))

        @staticmethod
        def end(step, tap_rows):
            return torch.stack((step.ids, step.positions, step.ids + step.positions), 1).float()

        @staticmethod
        def embed(ids):
            return ids[:, None].half()

        @staticmethod
        def _linear(value, weight):
            assert weight is Target.w.head
            return value

    target = Target()
    target.device = device

    class Spec:
        def validate_step(self, owner, step):
            if owner is not target:
                raise ValueError("DSpark verify state belongs to another target")
            assert step.spec in (self, None)

        def commit(self, slots, counts):
            events.append(("commit", slots.tolist(), counts.tolist()))

    spec = Spec()
    spec.slots = scratch

    class Head:
        ring = torch.full((slots, 3, 256, 512), -1, dtype=torch.float16, device=device)
        rotary = torch.full((slots, 256), -1, dtype=torch.int64, device=device)

        @staticmethod
        def draft(anchors, positions, owner, embed, project, *, depth):
            events.append(("draft", depth))
            return project(embed(anchors)).long() + torch.arange(1, depth + 1, device=device)

        def insert_context(self, owners, out, pos, *, padding):
            slot, rows = padding
            destinations = torch.where(owners == slot, rows, pos % 256)
            self.ring[owners, 0, destinations, 0] = out[:, 0]
            self.rotary[owners, destinations] = pos

    def gather(packet, out):
        out.copy_(packet[None].expand_as(out))

    engine = SimpleNamespace(target=target, rank=0, world=8, streams=4, slots=slots, context=context,
                             slot_pages=pages, geometry=SimpleNamespace(bos=1, vocab=101),
                             comm=SimpleNamespace(all_gather=gather))
    return engine, Head(), spec, events


def inspect(shape, output, head, wanted_ids, wanted_positions):
    step = shape.step
    assert step.qsl.tolist() == list(range(0, shape.bucket * shape.rows + 1, shape.rows))
    assert step.ids.view(shape.bucket, shape.rows)[:len(wanted_ids)].tolist() == wanted_ids
    assert step.positions.view(shape.bucket, shape.rows)[:len(wanted_positions)].tolist() == wanted_positions
    assert step.valid.tolist() == [True] * (len(wanted_ids) * shape.rows) + [False] * (
        (shape.bucket - len(wanted_ids)) * shape.rows)
    assert (step.block_table[len(wanted_ids):] == -1).all()
    assert output.local_logits[:, 2].long().tolist() == (step.ids + step.positions).tolist()
    assert output.compact.shape == (8, shape.bucket * shape.rows, 3)
    # An anchor at position zero ties columns 0 and 2, so the first wins.
    assert output.compact[0, :, 1].long().tolist() == [0 if p == 0 else 2 for p in step.positions.tolist()]
    n = len(wanted_ids) * shape.rows
    assert head.rotary[shape.scratch, n:n + shape.rows].tolist() == list(range(shape.rows))


def test_verify_frame_checked_prefix_commit_and_depth_zero():
    engine, head, spec, events = owners(torch.device("cpu"))
    shape = SpecShape(engine, head, spec, 4, 3, 8192)
    entries = [VerifyEntry(0, 65, 11, VerifyCommit(63, 4, 2)), VerifyEntry(2, 14, 31)]
    output = shape.run(entries)
    assert events == [("commit", [0, 5, 5, 5], [2, -1, -1, -1]), ("draft", 3),
                      ("begin", 8192, [0, 4, 8, 12, 16])]
    assert engine.target.index_context == engine.context
    assert shape.step.spec is spec and not shape.step.single_token_slots
    inspect(shape, output, head, [[11, 12, 13, 14], [31, 32, 33, 34]], [[65, 66, 67, 68], [14, 15, 16, 17]])
    assert shape.step.block_table[:2].tolist() == [[0, 1, 2], [9, 10, 11]]

    events.clear()
    serial = SpecShape(engine, head, spec, 4, 0, 8192)
    output = serial.run([VerifyEntry(2, 14, 31, VerifyCommit(14, 4, 0))])
    assert events == [("commit", [2, 5, 5, 5], [0, -1, -1, -1]), ("begin", 8192, [0, 1, 2, 3, 4])]
    assert serial.step.spec is None and serial.step.single_token_slots and output.drafts.shape == (4, 0)
    inspect(serial, output, head, [[31]], [[14]])


def test_refuses_invalid_frames_before_target_or_commit():
    engine, head, spec, events = owners(torch.device("cpu"))
    shape = SpecShape(engine, head, spec, 4, 3, 8192)
    for entries in ([VerifyEntry(0, 65, 11)] * 2, [VerifyEntry(5, 0, 11)], [VerifyEntry(3, 61, 11)],
                    [VerifyEntry(4, 0, 11)], [VerifyEntry(0, 0, 101)], [VerifyEntry(0, -1, 11)],
                    [VerifyEntry(0, 65, 11, VerifyCommit(63, 4, 1))],
                    [VerifyEntry(0, 65, 11, VerifyCommit(59, 4, 6))]):
        with pytest.raises(ValueError):
            shape.run(entries)
    for args in ((3, 2, 8192), (4, -1, 8192), (4, 6, 8192), (4, 2, 8191), (4, 5, 8192)):
        with pytest.raises(ValueError):
            SpecShape(engine, head, spec, *args)
    assert events == []


def test_rejects_wrong_same_extent_owner_and_foreign_static_storage():
    device = torch.device("cpu")
    engine, head, spec, events = owners(device)
    other, _, _, other_events = owners(device)
    with pytest.raises(ValueError, match="belongs to another target"):
        SpecShape(other, head, spec, 4, 0, 8192)
    with pytest.raises(ValueError, match="belongs to another target"):
        SpecShape(other, head, spec, 4, 3, 8192)

    original_ring = head.ring
    head.ring = torch.empty_like(original_ring, device="meta")
    with pytest.raises(ValueError, match="prepared draft ring"):
        SpecShape(engine, head, spec, 4, 3, 8192)
    head.ring = original_ring[:, :, :, :1]
    with pytest.raises(ValueError, match="prepared draft ring"):
        SpecShape(engine, head, spec, 4, 3, 8192)
    head.ring = original_ring
    engine.slot_pages.table = engine.slot_pages.table.long()
    with pytest.raises(ValueError, match="contiguous device page storage"):
        SpecShape(engine, head, spec, 4, 0, 8192)
    assert events == other_events == []


def test_capture_prewarms_only_scratch_with_live_commit_buffers(monkeypatch):
    engine, head, spec, events = owners(torch.device("cpu"))
    shape = SpecShape(engine, head, spec, 4, 3, 8192)
    active, commits = [], []
    current = SimpleNamespace(wait_stream=lambda stream: None)
    startup = SimpleNamespace(wait_stream=lambda stream: None, synchronize=lambda: None)

    @contextmanager
    def choose(stream):
        active.append(stream)
        try:
            yield
        finally:
            assert active.pop() is stream

    class Graph:
        @staticmethod
        def replay():
            events.append(("replay",))

    original = spec.commit

    def commit(slots, counts):
        assert slots is shape.commit_slots and slots.dtype == torch.int32
        assert counts.data_ptr() == shape._staged[3].data_ptr() and counts.dtype == torch.int64
        assert slots.stride() == counts.stride() == (1,)
        commits.append((active[-1] if active else current, slots.tolist(), counts.tolist()))
        original(slots, counts)

    monkeypatch.setattr(spec, "commit", commit)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current)
    monkeypatch.setattr(torch.cuda, "stream", choose)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, **kwargs: nullcontext())

    shape.capture(None, startup)
    assert commits == [(startup, [5, 5, 5, 5], [-1, -1, -1, -1])]
    assert events[0] == ("commit", [5, 5, 5, 5], [-1, -1, -1, -1])
    assert len([event for event in events if event[0] == "commit"]) == 1

    events.clear()
    shape.run([VerifyEntry(2, 16, 33, VerifyCommit(14, 4, 2))])
    shape.run([VerifyEntry(0, 65, 11)])
    assert commits[1:] == [(current, [2, 5, 5, 5], [2, -1, -1, -1])]
    assert events == [("commit", [2, 5, 5, 5], [2, -1, -1, -1]), ("replay",), ("replay",)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph required")
def test_captured_shape_replays_changed_live_slots_and_commits_once():
    with torch.inference_mode():
        engine, head, spec, events = owners(torch.device("cuda"))
        shape = SpecShape(engine, head, spec, 4, 3, 8192)
        shape.capture(torch.cuda.graph_pool_handle(), torch.cuda.Stream())
        events.clear()
        first = shape.run([VerifyEntry(0, 65, 11), VerifyEntry(2, 14, 31)])
        inspect(shape, first, head, [[11, 12, 13, 14], [31, 32, 33, 34]], [[65, 66, 67, 68], [14, 15, 16, 17]])
        assert events == []
        second = shape.run([VerifyEntry(2, 16, 33, VerifyCommit(14, 4, 2))])
        inspect(shape, second, head, [[33, 34, 35, 36]], [[16, 17, 18, 19]])
        assert events == [("commit", [2, 5, 5, 5], [2, -1, -1, -1])]
        assert first.compact.data_ptr() == second.compact.data_ptr()
        assert first.local_logits.data_ptr() == second.local_logits.data_ptr()
