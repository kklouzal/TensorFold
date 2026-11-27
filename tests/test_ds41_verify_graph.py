"""DSpark graph-owner boundaries, with independent target and head witnesses."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v4.cuda.dspark_graph import FixedVerifyOwner, SpecShape, VerifyCommit, VerifyEntry
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
        g = SimpleNamespace(hidden=2, head_dim=512)
        hash_state = SimpleNamespace(max_tokens=16, err=torch.zeros(1, dtype=torch.int32, device=device))
        c2 = {}
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


@pytest.mark.parametrize("depth", (1, 2, 3, 4, 5))
def test_fixed_shape_has_two_real_requests_and_one_pad(depth):
    from tensorfold.families.deepseek_v4.cuda.engine import MAX_TOKENS

    engine, head, spec, events = owners(torch.device("cpu"))
    engine.target.hash_state.max_tokens = MAX_TOKENS
    shape = SpecShape(engine, head, spec, 3, depth, engine.context)
    entries = (VerifyEntry(0, 65, 11), VerifyEntry(2, 14, 31))
    output = shape.run(entries)
    inspect(shape, output, head, [list(range(11, 12 + depth)), list(range(31, 32 + depth))],
            [list(range(65, 66 + depth)), list(range(14, 15 + depth))])
    assert output.drafts.shape == (3, depth)
    assert events[0] == ("draft", depth)


def test_fixed_bucket_coverage_for_every_active_count():
    from tensorfold.families.deepseek_v4.cuda.graphs import DecodeGraphs, buckets, shape_keys

    graphs = DecodeGraphs.__new__(DecodeGraphs)
    graphs.sizes = buckets(32)
    capture_keys = set(shape_keys(32, 131_072))
    expected_buckets = [*range(1, 17), *([24] * 8), *([32] * 8)]
    for active, expected in enumerate(expected_buckets, 1):
        bucket = graphs.bucket(active)
        assert bucket == expected and (bucket, 131_072) in capture_keys


def test_verify_uses_the_installed_embedding_when_requested(monkeypatch):
    engine, head, spec, _ = owners(torch.device("cpu"))
    original = head.draft

    def draft(anchors, positions, owners, embed, project, *, depth):
        assert embed is None
        return original(anchors, positions, owners, engine.target.embed, project, depth=depth)

    monkeypatch.setattr(head, "draft", draft)
    shape = SpecShape(engine, head, spec, 1, 3, engine.context, use_prepared_embedding=True)
    output = shape.run([VerifyEntry(0, 65, 11)])
    assert output.drafts.tolist() == [[12, 13, 14]]
    assert shape.step.ids.tolist() == [11, 12, 13, 14]


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
    with torch.inference_mode():
        target_scratch = torch.ones(1)
    begin = engine.target.begin

    def clear_target_scratch(step):
        # Ordinary warm-up has already allocated the target's shared scratch.
        target_scratch.zero_()
        return begin(step)

    monkeypatch.setattr(engine.target, "begin", clear_target_scratch)
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
    assert target_scratch.item() == 0
    assert not torch.is_inference_mode_enabled() and not shape._staged.is_inference()
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
        engine, head, spec, events = owners(torch.device("cuda", torch.cuda.current_device()))
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


@pytest.mark.parametrize("active", (1, 2, 3))
def test_fixed_vector_owner_settles_once_and_omits_scratch_from_commit(active):
    engine, head, spec, events = owners(torch.device("cpu"))
    shape = SpecShape(engine, head, spec, 3, 3, engine.context)
    owner = FixedVerifyOwner(shape)
    assert events.pop() == ("commit", [5, 5, 5], [-1, -1, -1])
    slots = (0, 2, 3)[:active]
    entries = tuple(VerifyEntry(slot, 3, 11 + slot) for slot in slots)
    owner.verify_batch(0, entries)
    assert owner.pending == (0, slots)
    with pytest.raises(ValueError, match="verify sequence"):
        owner.verify_batch(1, entries)
    with pytest.raises(ValueError, match="pending slots"):
        owner.commit_batch(1, tuple((slot, 1) for slot in slots))
    counts = (2, 0, 4)[:active]
    decisions = tuple(zip(slots, counts, strict=True))
    owner.commit_batch(0, decisions)
    assert events[-1] == ("commit", [*slots, *([5] * (3 - active))], [*counts, *([-1] * (3 - active))])
    assert owner.pending is None and owner.next_sequence == 1
    with pytest.raises(ValueError, match="commit owners"):
        owner.commit_batch(0, decisions)


def test_invalid_native_commit_does_not_read_an_unowned_candidate():
    from tensorfold.families.deepseek_v4.cuda.kernels.mu_state import (
        ERR_CONTINUE, _mu_commit_c2_accepted, _mu_commit_history_accepted,
    )
    from tensorfold.families.deepseek_v4.cuda.kernels.triton_utils import INTERPRET

    if not INTERPRET and not torch.cuda.is_available():
        pytest.skip("Triton interpreter or a CUDA device is required")
    device = "cpu" if INTERPRET else "cuda"
    width, candidates = 8, 10
    history = torch.full((2, 4), 17, dtype=torch.int64, device=device)
    choice = torch.zeros((2, candidates, 4), dtype=torch.int64, device=device)
    lengths = torch.tensor([0, 3], dtype=torch.int64, device=device)
    slots = torch.arange(2, dtype=torch.int32, device=device)
    counts = torch.tensor([0, 1 << 35], dtype=torch.int64, device=device)
    errors = torch.zeros((), dtype=torch.int32, device=device)
    _mu_commit_history_accepted[(2,)](history, choice, lengths, slots, counts, errors, S=2, J=candidates)
    assert (history == 17).all() and errors.item() == ERR_CONTINUE
    partial = torch.full((2, 2, width), 17.0, device=device)
    positions = torch.full((2,), 17, dtype=torch.int64, device=device)
    vectors = torch.zeros((2, candidates, 2, width), device=device)
    offsets = torch.zeros((2, candidates), dtype=torch.int64, device=device)
    errors.zero_()
    _mu_commit_c2_accepted[(2,)](partial, positions, vectors, offsets, lengths, slots, counts, errors,
                                  S=2, J=candidates, D=width, BLOCK=width)
    assert (partial == 17).all() and (positions == 17).all() and errors.item() == ERR_CONTINUE


@pytest.mark.parametrize("sticky_owner", ("engram", "compressor"))
def test_verify_refuses_poisoned_rank_before_commit_writes(sticky_owner):
    engine, head, spec, events = owners(torch.device("cpu"))
    engine.target.c2[0] = SimpleNamespace(err=torch.zeros(1, dtype=torch.int32))
    shape = SpecShape(engine, head, spec, 3, 3, engine.context)
    owner = FixedVerifyOwner(shape)
    owner.verify_batch(0, (VerifyEntry(0, 3, 11),))
    events.clear()
    error = engine.target.hash_state.err if sticky_owner == "engram" else engine.target.c2[0].err
    error.fill_(4)

    with pytest.raises(RuntimeError, match="poisoned verification state"):
        owner.commit_batch(0, ((0, 0),))
    assert owner.pending == (0, (0,)) and events == []
    with pytest.raises(RuntimeError, match="poisoned verification state"):
        shape.run([VerifyEntry(0, 3, 11, VerifyCommit(0, 4, 3))])
    assert events == []


def test_verify_commit_does_not_release_local_pending_on_post_commit_failure(monkeypatch):
    engine, head, spec, events = owners(torch.device("cpu"))
    owner = FixedVerifyOwner(SpecShape(engine, head, spec, 3, 3, engine.context))
    owner.verify_batch(0, (VerifyEntry(0, 3, 11),))

    def fail(*args):
        engine.target.hash_state.err.fill_(4)

    monkeypatch.setattr(spec, "commit", fail)
    with pytest.raises(RuntimeError, match="DSpark commit failed"):
        owner.commit_batch(0, ((0, 0),))
    assert owner.pending == (0, (0,))


@pytest.mark.parametrize("broken_ack", (False, True))
def test_engine_waits_for_exact_followers_before_completing_vector_commit(monkeypatch, broken_ack):
    import array

    from tensorfold.cuda import cohort
    from tensorfold.families.deepseek_v4.cuda.engine import (
        _HEAD, DS41Engine, OP_ATTACH, OP_DRAFT_COMMIT_BATCH, OP_DRAFT_VERIFY_BATCH,
    )

    notices, echoes, commits = [], [], []
    monkeypatch.setattr(cohort, "notify_failure", lambda: notices.append(True))
    peers = []

    def ack(payload):
        echoes.append(payload)

    class Verifier:
        shape = SimpleNamespace(bucket=3)
        pending = None

        def __init__(self, rank):
            self.rank = rank

        def verify_batch(self, sequence, entries):
            self.pending = sequence, tuple(entry.slot for entry in entries)
            return self.pending

        def commit_batch(self, sequence, rows):
            assert self.pending == (sequence, tuple(slot for slot, _ in rows))
            commits.append((self.rank, sequence, rows))
            self.pending = None

    def run_followers(frame):
        payload = array.array("i")
        payload.frombytes(frame[_HEAD.size:])
        header = _HEAD.unpack_from(frame)
        for peer in peers[1:]:
            peer._run(*header, payload)

    def join():
        assert {row[0] for row in commits} == set(range(8))
        assert all(frame == echoes[0] for frame in echoes)
        return echoes[:-1] if broken_ack else echoes

    for rank in range(8):
        owner = DS41Engine.__new__(DS41Engine)
        owner.rank, owner.world, owner.part = rank, 8, None
        owner._draft_verify = Verifier(rank)
        owner.pipes = SimpleNamespace(send=run_followers, acknowledge=ack, acknowledgments=join, broken=False)
        peers.append(owner)
    leader = peers[0]
    assert leader.announce(OP_DRAFT_VERIFY_BATCH, 2, 2, ids=iter((0, 39, 4, 2, 18, 8))) == (2, (0, 2))
    with pytest.raises(RuntimeError, match="pending vector"):
        leader.announce(OP_ATTACH, 4)
    if broken_ack:
        with pytest.raises(RuntimeError, match="acknowledgment"):
            leader.announce(OP_DRAFT_COMMIT_BATCH, 2, 2, ids=iter((0, 0, 2, 3)))
        assert leader.pipes.broken and notices == [True]
    else:
        leader.announce(OP_DRAFT_COMMIT_BATCH, 2, 2, ids=iter((0, 0, 2, 3)))
        assert not leader.pipes.broken and notices == []
    assert all(row[2] == ((0, 0), (2, 3)) for row in commits)


def test_engine_ordinary_announce_keeps_its_existing_wire(monkeypatch):
    import array

    from tensorfold.families.deepseek_v4.cuda.engine import _HEAD, DS41Engine, OP_ATTACH, OP_DRAFT_VERIFY_BATCH

    engine = DS41Engine.__new__(DS41Engine)
    sent = []
    engine.pipes = SimpleNamespace(send=sent.append)
    monkeypatch.setattr(engine, "_attach", lambda slot, tokens: (slot, tokens))
    assert engine.announce(OP_ATTACH, 2, 64) == (2, 64)
    assert sent == [_HEAD.pack(OP_ATTACH, 2, 64, 0, 0) + array.array("i").tobytes()]
    with pytest.raises(ValueError, match="control frame"):
        engine.announce(OP_DRAFT_VERIFY_BATCH, 0, 1, ids=(0, 1, 2))
    assert len(sent) == 1
