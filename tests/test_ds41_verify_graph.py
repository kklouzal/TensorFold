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
    spec._target = target

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
        bucket = 3
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


def test_grammar_verify_gathers_only_flagged_lane_rows_in_vector_order():
    import array

    from tensorfold.families.deepseek_v4.cuda.dspark_graph import VerifyOutput
    from tensorfold.families.deepseek_v4.cuda.engine import (
        DS41Engine, OP_DRAFT_COMMIT_BATCH, OP_DRAFT_VERIFY_BATCH, _HEAD,
    )

    local_logits = torch.arange(12, dtype=torch.float32).reshape(-1, 1)
    result = VerifyOutput(torch.empty((8, 12, 3)), torch.empty((3, 3), dtype=torch.long), local_logits)
    frames, gathers, checked = [], [], []

    def verify(sequence, entries):
        checked.append((sequence, [entry.slot for entry in entries]))
        return result

    def gather(local):
        gathers.append(local.flatten().tolist())
        return torch.cat([local, local + 100], dim=1)

    def engine(rank):
        owner = DS41Engine.__new__(DS41Engine)
        owner.rank, owner.world, owner.part = rank, 8, None
        owner._draft_verify = SimpleNamespace(bucket=3, depth=3, pending=None, verify_batch=verify)
        owner.pipes = SimpleNamespace(send=frames.append)
        owner.coll = SimpleNamespace(gather=gather)
        return owner

    leader, follower = engine(0), engine(1)
    entries = (3, 24, 11, 2, 30, 12, 1, 16, 13)
    output = leader.announce(OP_DRAFT_VERIFY_BATCH, 9, 3, 0b101, ids=entries)
    payload = array.array("i", entries)
    assert frames == [_HEAD.pack(OP_DRAFT_VERIFY_BATCH, 9, 3, 0b101, 0) + payload.tobytes()]
    assert follower._run(*_HEAD.unpack_from(frames[0]), payload).full_logits is None
    expected = [0., 1., 2., 3., 8., 9., 10., 11.]
    assert output.full_logits.tolist() == [[value, value + 100] for value in expected]
    assert gathers == [expected, expected] and checked == [(9, [3, 2, 1]), (9, [3, 2, 1])]
    with pytest.raises(ValueError, match="control frame"):
        leader.announce(OP_DRAFT_VERIFY_BATCH, 9, 3, 0b1000, ids=entries)
    with pytest.raises(ValueError, match="control frame"):
        leader.announce(OP_DRAFT_COMMIT_BATCH, 9, 3, 1, ids=(3, 1, 2, 1, 1, 1))
    assert len(frames) == 1


def test_exact_live_widths_share_one_pending_transaction():
    engine, head, spec, events = owners(torch.device("cpu"))
    shapes = tuple(SpecShape(engine, head, spec, width, 3, engine.context) for width in (1, 2, 3))
    owner = FixedVerifyOwner(shapes[2], narrower=shapes[:2])
    entries = tuple(VerifyEntry(slot, 3, 11 + slot) for slot in (0, 2, 3))
    for sequence, count in enumerate((2, 1, 3, 1)):
        out = owner.verify_batch(sequence, entries[:count])
        assert out.drafts.shape == (count, 3)
        assert events[-1] == ("begin", engine.context, list(range(0, 4 * count + 1, 4)))
        with pytest.raises(RuntimeError, match="pending verification"):
            owner.release_graphs()
        owner.commit_batch(sequence, tuple((entry.slot, 0) for entry in entries[:count]))
    assert owner.next_sequence == 4 and owner.pending is None
    owner.release_graphs()


def test_fixed_scheduler_cancels_a_lane_and_publishes_only_after_commit():
    from threading import Event
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.deepseek_v4.cuda.engine import (
        FixedDraftDecoder, OP_DRAFT_COMMIT_BATCH, OP_DRAFT_VERIFY_BATCH,
    )

    decoder = FixedDraftDecoder.__new__(FixedDraftDecoder)
    decoder.eos = (99,)
    decoder.verify_rounds_by_active = [0] * 4
    sent, emitted = [], []
    cancel = Event()
    first = Stream([1] * 10, 3, cancel=Event(), out=[5],
                   emit=lambda tokens: emitted.append((sent[-1][0], tokens)))
    second = Stream([1] * 10, 6, cancel=cancel, out=[6])
    for slot, stream in enumerate((first, second)):
        stream.slot = slot
        stream.draft_seed = True
    # Row 2/3 are not reachable on the first stream due to the cap. The
    # second stream cancels after the vector starts, before any commit.
    compact = torch.tensor([[[1., 10., 0.], [1., 11., 0.], [0., 0., 1.], [0., 0., 1.],
                             [1., 12., 0.], [1., 13., 0.], [1., 14., 0.], [1., 15., 0.]]])

    def announce(op, seq, count, grammar=0, *, ids):
        assert grammar == 0
        sent.append((op, seq, count, tuple(ids)))
        assert emitted == []
        if op == OP_DRAFT_VERIFY_BATCH:
            cancel.set()
            return SimpleNamespace(compact=compact, drafts=torch.tensor([[10, 11, 12], [12, 13, 14]]))

    decoder.e = SimpleNamespace(_draft_verify=SimpleNamespace(next_sequence=7, depth=3, bucket=3), context=32,
                                target=SimpleNamespace(pool=SimpleNamespace(owned={0: [0], 1: [1]})),
                                announce=announce)
    lanes = [first, second]
    entries = [decoder._draft_entry(stream) for stream in lanes]
    assert entries == [(0, 10, 5), (1, 10, 6)]
    assert decoder._fixed_round(lanes, entries) == lanes
    assert sent == [(OP_DRAFT_VERIFY_BATCH, 7, 2, (0, 10, 5, 1, 10, 6)),
                    (OP_DRAFT_COMMIT_BATCH, 7, 2, (0, 2, 1, 0))]
    assert emitted == [(OP_DRAFT_COMMIT_BATCH, [10, 11])]
    assert (first.drafted, first.accepted, second.drafted) == (3, 2, 0)
    assert decoder.verify_rounds_by_active == [0, 0, 1, 0]


def test_strict_dsml_speculation_masks_only_reachable_rows_and_commits_selected_tokens():
    xgr = pytest.importorskip("xgrammar")

    from tensorfold.cuda.streams import Stream
    from tensorfold.engine.grammar import Grammars
    from tensorfold.families.deepseek_v4.cuda.dspark_serving import decide_constrained
    from tensorfold.families.deepseek_v4.cuda.engine import FixedDraftDecoder
    from tensorfold.families.deepseek_v4.cuda.round_sampling import compact_packet, greedy_tokens
    from tensorfold.families.deepseek_v4.strict_tools_v41 import CLOSE, INVOKE_END, OPEN, Plan

    tools = [{"type": "function", "function": {"name": "now", "strict": True}}]
    plan = Plan(tools, required=False, parallel=False)
    header = plan.thinking_headers[0][len(OPEN):]
    words = ["", "thinking\n", "</think>", OPEN, header, f"{INVOKE_END}\n{CLOSE}", "invalid"]
    info = xgr.TokenizerInfo(words, xgr.VocabType.RAW, vocab_size=len(words), stop_token_ids=[0])
    grammars = Grammars(info)
    matcher = grammars.constraint(
        grammars.compile(plan.spec), think_end=2, answer_start=3,
        answer_opener=plan.opener, answer_headers=plan.thinking_headers,
        decode=lambda ids, **_kwargs: "".join(words[t] for t in ids),
    )
    request = Stream([1] * 192, count=12, constraint=matcher)
    request.forced = request.head = None
    request.pos = 0
    decoder = FixedDraftDecoder.__new__(FixedDraftDecoder)
    decoder._draft_turn = 0
    decoder._admitted(request)
    assert request.draft_seed
    matcher.advance([1])
    assert not matcher.active

    logits = torch.full((4, len(words)), float("-inf"))
    logits[0, 3] = logits[1, 4] = logits[2, 6] = 10.
    logits[2, 5] = 5.
    logits[3].fill_(float("nan"))  # follows an invalid proposal and is unreachable
    choices = greedy_tokens(compact_packet(logits, 0).unsqueeze(0), 4)
    chosen = decide_constrained(matcher, logits, choices, 1, [3, 4, 6], 8, (0,))
    assert chosen == [3, 4, 5]
    assert matcher.observed == 1 and not matcher.active    # speculative traversal rolled itself back
    matcher.advance(chosen)
    assert matcher.active and not matcher.finished
    _, calls, complete = plan.parse_finished("".join(words[t] for t in chosen))
    assert complete and calls[0]["function"]["name"] == "now"

    logits = torch.full((4, len(words)), float("-inf"))
    logits[0, 0] = 2.
    logits[1:].fill_(float("nan"))  # a terminating proposal has no child row
    choices = greedy_tokens(compact_packet(logits, 0).unsqueeze(0), 4)
    chosen = decide_constrained(matcher, logits, choices, 5, [0, 6, 6], 8, (0,))
    assert chosen == [0]
    matcher.advance(chosen)
    assert matcher.finished


@pytest.mark.parametrize("cancel", [False, True])
def test_constrained_verify_errors_and_cancellation_settle_only_the_affected_lane(monkeypatch, cancel):
    from threading import Event

    from tensorfold.cuda.streams import Stream
    from tensorfold.families.deepseek_v4.cuda import dspark_serving as serving
    from tensorfold.families.deepseek_v4.cuda.engine import (
        FixedDraftDecoder, OP_DRAFT_COMMIT_BATCH, OP_DRAFT_VERIFY_BATCH,
    )

    decoder = FixedDraftDecoder.__new__(FixedDraftDecoder)
    decoder.eos = (99,)
    decoder.verify_rounds_by_active = [0] * 4
    sent, emitted, advanced = [], [], []
    first_constraint = SimpleNamespace(advance=lambda tokens: advanced.append((0, tokens)))
    cancel_event = Event()
    rejected = RuntimeError("a test grammar rejected a chosen token")

    def last_advance(tokens):
        assert not cancel
        advanced.append((2, tokens))
        raise rejected

    last_constraint = SimpleNamespace(advance=last_advance)
    streams = [Stream([1] * 10, 2, cancel=event, constraint=constraint, out=[slot + 5],
                      emit=lambda tokens, lane=slot: emitted.append((lane, sent[-1][0], tokens)))
               for slot, event, constraint in ((0, None, first_constraint), (1, None, None),
                                                (2, cancel_event, last_constraint))]
    compact = torch.tensor([[[1., float(index + 10), 0.] for index in range(12)]])
    full_logits = torch.arange(8, dtype=torch.float32).view(-1, 1)

    def decision(constraint, logits, choices, anchor, drafts, room, eos):
        assert room == 1 and eos == (99,)
        if constraint is first_constraint:
            assert logits.flatten().tolist() == [0., 1., 2., 3.] and anchor == 5
            return [10]
        assert constraint is last_constraint and anchor == 7
        assert logits.flatten().tolist() == [4., 5., 6., 7.]
        if cancel:
            cancel_event.set()
        return [18]

    def announce(op, sequence, count, grammar=0, *, ids):
        sent.append((op, sequence, count, grammar, tuple(ids)))
        assert not emitted
        if op == OP_DRAFT_VERIFY_BATCH:
            return SimpleNamespace(
                compact=compact, drafts=torch.tensor([[10, 11, 12], [14, 15, 16], [18, 19, 20]]),
                full_logits=full_logits,
            )

    monkeypatch.setattr(serving, "decide_constrained", decision)
    decoder.e = SimpleNamespace(_draft_verify=SimpleNamespace(next_sequence=5, depth=3), announce=announce)
    assert decoder._fixed_round(streams, [(0, 10, 5), (1, 10, 6), (2, 10, 7)]) == streams
    assert sent == [(OP_DRAFT_VERIFY_BATCH, 5, 3, 0b101, (0, 10, 5, 1, 10, 6, 2, 10, 7)),
                    (OP_DRAFT_COMMIT_BATCH, 5, 3, 0, (0, 1, 1, 1, 2, 0))]
    assert advanced == ([(0, [10])] if cancel else [(0, [10]), (2, [18])])
    assert streams[-1].error is (None if cancel else rejected)
    assert emitted == [(0, OP_DRAFT_COMMIT_BATCH, [10]), (1, OP_DRAFT_COMMIT_BATCH, [14])]


def test_fixed_scheduler_verifies_disjoint_lanes_through_a_paused_prompt(monkeypatch):
    from tensorfold.families.deepseek_v4.cuda.engine import Decoder, FixedDraftDecoder

    calls = []
    decoder = FixedDraftDecoder.__new__(FixedDraftDecoder)
    decoder.e = SimpleNamespace(_draft_verify=SimpleNamespace(bucket=3))
    decoder._broken = decoder.inflight = decoder.split = None
    decoder._draft_turn = 0
    decoder.streams = {}
    decoder.filling = [SimpleNamespace(sid=i, draft_seed=True, draft_turn=0, done=False, cancel=None, out=[10])
                       for i in range(3)]
    first, second, third = decoder.filling
    decoder.finishing = (first, ())

    def filled(stream, _result):
        decoder.filling.remove(stream)
        decoder.streams[stream.sid] = stream
        calls.append(("publish-first", stream.sid))
        return []

    def fill():
        stream = decoder.filling[0]
        calls.append(("one-prompt-group", stream.sid))
        if stream is second and decoder.split is None:
            decoder.split = (second, (), 6)
            return []
        decoder.split = None
        decoder.finishing = (stream, ())
        return []

    def fixed(live, entries):
        assert entries == tuple(s.sid for s in live)
        calls.append(("verify-and-commit", len(live)))
        return []

    decoder._filled = filled
    decoder._fill = fill
    decoder._draft_entry = lambda s: s.sid
    decoder._fixed_round = fixed
    monkeypatch.setattr(Decoder, "round", lambda _self: calls.append("ordinary") or [])

    for _ in range(4):
        assert decoder.round() == []
    assert calls == [("publish-first", first.sid), ("verify-and-commit", 1), ("one-prompt-group", second.sid),
                     ("verify-and-commit", 1), ("one-prompt-group", second.sid),
                     ("publish-first", second.sid), ("verify-and-commit", 2), ("one-prompt-group", third.sid),
                     ("publish-first", third.sid), ("verify-and-commit", 3)]
    # A fourth filling request does not invalidate any ready draft rings.
    decoder.filling.append(SimpleNamespace(sid=3, draft_seed=True, draft_turn=decoder._draft_turn,
                                          done=False, cancel=None, out=[10]))
    assert decoder.round() == []
    assert calls[-2:] == [("verify-and-commit", 3), ("one-prompt-group", 3)]


def test_vision_prefill_does_not_disable_ready_text_draft_rounds(monkeypatch):
    from tensorfold.families.deepseek_v4.cuda.engine import Decoder, FixedDraftDecoder

    decoder = FixedDraftDecoder.__new__(FixedDraftDecoder)
    decoder.e = SimpleNamespace(_draft_verify=SimpleNamespace(bucket=3))
    decoder._broken = decoder.inflight = decoder.finishing = decoder.split = None
    decoder._draft_turn = 0
    options = dict(draft=True, background=False, owed=False, token_limit=None, head=None, pos=0, prompt=[1, 2],
                   forced=None, constraint=None, history=None, probabilities=None, sampling=None,
                   done=False, cancel=None, out=[10])
    text = SimpleNamespace(sid=0, vision=None, **options)
    image = SimpleNamespace(sid=1, vision=object(), **options)
    decoder._admitted(text)
    decoder._admitted(image)
    assert text.draft_seed and not image.draft_seed
    decoder.streams, decoder.filling = {0: text}, [image]
    calls = []
    decoder._draft_entry = lambda s: s.sid
    decoder._fixed_round = lambda active, entries: calls.append(("verify", active, entries)) or []
    decoder._fill = lambda: calls.append(("prefill", decoder.filling[:])) or []

    def ordinary(active):
        assert active is decoder and active.filling == [image]
        return ["ordinary target result"]

    monkeypatch.setattr(Decoder, "round", ordinary)
    assert decoder.round() == []
    assert calls == [("verify", (text,), (0,)), ("prefill", [image])]
    assert text.draft_seed
    text.done = True
    assert decoder.round() == ["ordinary target result"]


def test_mixed_round_drains_old_ordinary_work_then_admits_draft_and_advances_background(monkeypatch):
    from threading import Event

    from tensorfold.cuda.streams import Stream
    from tensorfold.families.deepseek_v4.cuda.engine import Decoder, FixedDraftDecoder

    calls = []
    decoder = FixedDraftDecoder.__new__(FixedDraftDecoder)
    decoder._broken = decoder.ahead = decoder.split = decoder._draft_round_ms = None
    decoder._draft_turn = 0
    decoder.eos = (99,)
    decoder.rounds = SimpleNamespace(
        add=lambda bucket, ms: calls.append(("ordinary_time", bucket, ms)),
        estimate=lambda bucket: 13 if bucket == 2 else pytest.fail("wrong ordinary bucket"),
    )
    decoder.e = SimpleNamespace(
        _draft_verify=SimpleNamespace(bucket=3, depth=3), context=128, graphs=None,
        target=SimpleNamespace(pool=SimpleNamespace(owned={0: [0], 1: [1], 2: [2]})),
    )

    def stream(sid, out, *, draft_seed=False, **options):
        value = Stream([1] * (63 if sid == 2 else 12), 9, sid=sid, out=out, **options)
        value.slot, value.forced = sid, None
        value.draft_seed, value.draft_turn = draft_seed, 0
        return value

    ordinary = stream(1, [4], background=True, emit=lambda values: calls.append(("emit", values)))
    retired = stream(9, [5], background=True, emit=lambda _values: pytest.fail("emitted a stale lane"))
    joining = stream(0, [6], draft_seed=True)
    tail = stream(2, [7], draft_seed=True)
    cancelled = stream(3, [8], draft_seed=True, cancel=Event())
    cancelled.cancel.set()
    filling = stream(4, [], background=True)
    decoder.streams = {1: ordinary, 2: tail, 3: cancelled}
    decoder.filling = [joining, filling]
    decoder.finishing = (joining, "finished prompt")
    packet = torch.tensor([[[1., 11., 0.], [1., 12., 0.]]])
    started = SimpleNamespace(elapsed_time=lambda _end: 10)
    ended = SimpleNamespace(synchronize=lambda: calls.append("drain"))
    decoder.inflight = ([retired, ordinary], [13, 13], packet, None, None, 0., (started, ended, 2), ())

    def finish_prompt(value, completed):
        assert value is joining and completed == "finished prompt"
        assert decoder.inflight is None and ordinary.out == [4, 12]
        decoder.filling.remove(value)
        decoder.streams[value.sid] = value
        calls.append("finish_prompt")
        return []

    def verify(live, entries):
        assert live == (joining,) and entries == ((0, 12, 6),)
        calls.append("verify")
        return []

    def launch(active, live, pending):
        assert active is decoder and live == [ordinary, tail] and not pending
        assert decoder.inflight is None and not tail.draft_seed
        decoder.inflight = object()
        calls.append("launch")

    def fill_prompt():
        assert decoder.finishing is None and decoder.filling == [filling]
        assert decoder._prefill_round_ms() >= 13
        calls.append("fill")
        return []

    decoder._filled, decoder._fixed_round, decoder._fill = finish_prompt, verify, fill_prompt
    monkeypatch.setattr(Decoder, "_launch", launch)
    assert decoder.round() == [cancelled]
    assert calls == ["drain", ("ordinary_time", 2, 10), ("emit", [12]), "finish_prompt", "verify", "launch", "fill"]
    assert joining.draft_seed and cancelled.done and decoder._draft_round_ms is None


def test_full_draft_vectors_do_not_starve_waiting_lane_when_short_requests_churn():
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.deepseek_v4.cuda.engine import FixedDraftDecoder

    decoder = FixedDraftDecoder.__new__(FixedDraftDecoder)
    decoder.e = SimpleNamespace(_draft_verify=SimpleNamespace(bucket=3))
    decoder._draft_turn = 0
    decoder._draft_entry = lambda stream: stream.sid
    decoder.streams = {}

    def add(sid):
        value = Stream([1] * 10, count=2, sid=sid, out=[10])
        value.forced = value.head = None
        value.pos = 0
        decoder._admitted(value)
        decoder.streams[sid] = value

    for sid in range(4):
        add(sid)
    next_sid, selected_long = 4, []
    for turn in range(8):
        ready, ordinary, cancelled = decoder._decode_groups()
        assert len(ready) == 3 and not ordinary and not cancelled
        assert all(stream.draft_seed for stream in decoder.streams.values())
        for stream, entry in ready:
            assert entry == stream.sid
            if stream.sid == 0:
                selected_long.append(turn)
                continue
            del decoder.streams[stream.sid]
            add(next_sid)
            next_sid += 1
    assert selected_long[0] == 0 and selected_long[-1] >= 6
    assert all(b - a <= 2 for a, b in zip(selected_long, selected_long[1:]))


def test_cold_long_prompt_seeds_the_whole_draft_window_across_singleton_tail():
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.deepseek_v4.cuda.engine import FixedDraftDecoder
    from tensorfold.families.deepseek_v4.cuda.prefix import chunk_end, chunk_splits

    decoder = FixedDraftDecoder.__new__(FixedDraftDecoder)
    decoder._draft_turn = 0
    tapped = []

    def step(entries, ids, logits_rows, flags, *, tap_from=None):
        (slot, start, rows), = entries
        assert slot == 0 and rows == len(ids) and flags == 0
        if tap_from is not None:
            tapped.extend(range(start + tap_from, start + rows))

    decoder.e = SimpleNamespace(step=step)
    prompt_length = 24 * 8192 + 1
    stream = Stream([10] * prompt_length, count=32)
    stream.forced = stream.head = None
    stream.slot = stream.pos = 0
    decoder._admitted(stream)
    bounds = chunk_splits([], stream.pos, prompt_length, 8192)
    assert bounds[-1] == prompt_length - 1
    start = stream.pos
    while start < prompt_length:
        stop = chunk_end(bounds, start, prompt_length)
        decoder._text_chunk(stream, start, stop, stop == prompt_length, 0)
        start = stop
    assert stream.draft_seed
    assert tapped == list(range(prompt_length - 128, prompt_length))


@pytest.mark.parametrize("pause_after_units", [0, 20])
def test_singleton_tap_packet_bypasses_decode_graph_and_seeds_draft_head(pause_after_units):
    import array

    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine, F_DRAFT_TAPS, OP_LAYERS, OP_STEP, _HEAD

    engine = DS41Engine.__new__(DS41Engine)
    engine.rank, engine.world, engine.slots, engine.units = 0, 8, 4, 80
    engine.geometry = SimpleNamespace(hidden=8)
    engine.expert_observer = None
    engine.part = engine.layer_marks = None
    frames, seeded, graph_calls, segments = [], [], [], []
    engine.pipes = SimpleNamespace(send=frames.append)

    def taps(state, stop, indices, out, marks):
        assert state["step"].positions.tolist() == [65]
        assert indices.tolist() == [0] and marks is None
        out.fill_(7)
        segments.append(stop)

    def insert_context(slots, values, positions):
        seeded.append((slots.tolist(), positions.tolist(), values.tolist()))

    engine._draft_verify = SimpleNamespace(
        pending=None, head=SimpleNamespace(insert_context=insert_context),
    )
    engine.target = SimpleNamespace(
        device=torch.device("cpu"), pool=SimpleNamespace(owned={2: [12, 13]}),
        begin=lambda step: {"step": step}, layers_with_draft_taps=taps,
        end=lambda _state, rows: rows.tolist(),
    )
    engine.coll = SimpleNamespace(gather=lambda value: value)
    engine.graphs = SimpleNamespace(sizes=(1,), run=lambda *args, **kwargs: graph_calls.append(args) or "graph")

    result = engine.step([(2, 65, 1)], [10], [0], pause_after_units, tap_from=0)
    if pause_after_units:
        assert result is None and not seeded and engine.part[2] == (2,)
        assert engine.announce(OP_LAYERS, 80) == [0]
        assert engine.part is None and segments == [pause_after_units, 80]
    else:
        assert result == [0] and segments == [80]
    assert not graph_calls
    assert seeded == [([2], [65], [[7.0] * 24])]
    payload = array.array("i", [2, 65, 1, 10, 0, 0]).tobytes()
    assert frames[0] == _HEAD.pack(OP_STEP, 1, 1, 1, F_DRAFT_TAPS | pause_after_units) + payload
    assert engine.step([(2, 65, 1)], [10], [0]) == "graph"
    assert len(graph_calls) == 1


def test_draft_trace_distinguishes_shadow_admission_and_active_fallback(monkeypatch):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.deepseek_v4.cuda import engine

    def stream(**options):
        result = Stream([10] * 192, count=32, **options)
        result.forced = result.head = None
        result.pos = 0
        return result

    decoder = engine.FixedDraftDecoder.__new__(engine.FixedDraftDecoder)
    decoder._draft_turn = 0
    monkeypatch.setattr(engine, "TRACE_DRAFT", False)
    hidden = stream()
    decoder._admitted(hidden)
    assert hidden.draft_seed and "dspark_admission" not in hidden.stats()

    monkeypatch.setattr(engine, "TRACE_DRAFT", True)
    excluded = stream(sampling=SimpleNamespace(temperature=float("nan")))
    excluded.head, excluded.pos = object(), len(excluded.prompt)
    shadow = engine.Decoder.__new__(engine.Decoder)
    shadow._admitted(excluded)
    assert excluded.stats()["dspark_shadow"]
    assert excluded.stats()["dspark_admission"] == ("sampling", "exact_cache_hit")

    eligible = stream()
    decoder._admitted(eligible)
    assert eligible.stats()["dspark_admission"] == ()
    assert "dspark_shadow" not in eligible.stats()
    monkeypatch.setattr(engine.Decoder, "_launch", lambda *_args: None)
    decoder._launch([eligible], {})
    assert not eligible.draft_seed and eligible.stats()["dspark_fallback"] == "ordinary_decode"


def test_paused_prefill_disjointness_is_checked_before_vector_transport():
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine, OP_DRAFT_VERIFY_BATCH

    engine = DS41Engine.__new__(DS41Engine)
    engine.part = (object(), object(), (4, 8))
    engine._draft_verify = SimpleNamespace(bucket=3, pending=None)
    engine.pipes = SimpleNamespace(send=lambda _frame: pytest.fail("overlapping verify reached followers"))
    assert engine._check_draft_vector(OP_DRAFT_VERIFY_BATCH, 0, 2, 0, 0, (3, 14, 5, 7, 9, 2)) == (
        engine._draft_verify, 3,
    )
    with pytest.raises(RuntimeError, match="overlaps a paused prompt slot"):
        engine.announce(OP_DRAFT_VERIFY_BATCH, 0, 2, ids=(3, 14, 5, 8, 9, 2))


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


def _sampled_rows(*rows):
    return torch.tensor(rows, dtype=torch.float32)


@pytest.fixture
def cpu_effective_rows(monkeypatch):
    import tensorfold.cuda.sampling as sampling

    def cpu_metadata(logits, positions, policy):
        assert not logits.is_cuda and logits.ndim == 2 and logits.shape[0] == len(positions)
        sampling.validate_policy(positions, policy)
    # The real effective_rows/nucleus/rejection code runs; only its CUDA geometry check is substituted.
    monkeypatch.setattr(sampling, "validate_rows", cpu_metadata)


def test_sampled_verify_keeps_the_target_law_for_point_mass_proposals(cpu_effective_rows):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.deepseek_v4.cuda.dspark_serving import decide_sampled

    first = [2., 1., 0., -1., -3.]
    second = [0., 3., 1., 0., 0.]
    logits = _sampled_rows(first, second, second, second)
    target = torch.softmax(torch.tensor(first, dtype=torch.float64), -1).numpy()
    draws, followers = 6000, {}
    counts = [0] * 5
    for seed in range(draws):
        chosen = decide_sampled(Sampling(seed, temperature=1.0, top_k=0, top_p=1.0), logits, 7, [1, 1, 1], 40, 8, ())
        assert 1 <= len(chosen) <= 4
        counts[chosen[0]] += 1
        if chosen[0] == 1 and len(chosen) > 1:
            followers[chosen[1]] = followers.get(chosen[1], 0) + 1
    # Pearson chi-square with 4 degrees of freedom: p = 1e-4 at 23.5.
    chi = sum((c - draws * p) ** 2 / (draws * p) for c, p in zip(counts, target))
    assert chi < 23.5, (counts, chi)
    # The accepted first proposal's child follows the second row's law.
    law = torch.softmax(torch.tensor(second, dtype=torch.float64), -1).numpy()
    n = sum(followers.values())
    chi = sum((followers.get(t, 0) - n * p) ** 2 / (n * p) for t, p in enumerate(law))
    assert chi < 23.5, (followers, chi)


def test_sampled_verify_honors_filters_room_and_end_tokens(cpu_effective_rows):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.deepseek_v4.cuda.dspark_serving import decide_sampled

    peaked = [2., 1., 0., -1., -3.]                # p(0) ~ 0.64: top_p 0.5 keeps only token 0
    logits = _sampled_rows(peaked, peaked, peaked, peaked)
    for seed in range(50):
        narrow = Sampling(seed, temperature=1.0, top_k=0, top_p=0.5)
        assert decide_sampled(narrow, logits, 7, [1, 0, 0], 40, 8, ()) == [0]
        assert decide_sampled(narrow, logits, 7, [0, 0, 0], 40, 8, ()) == [0, 0, 0, 0]
        assert decide_sampled(narrow, logits, 7, [0, 0, 0], 40, 2, ()) == [0, 0]
        assert decide_sampled(narrow, logits, 7, [0, 0, 0], 40, 8, (0,)) == [0]
    with pytest.raises(ValueError, match="geometry"):
        decide_sampled(Sampling(1, temperature=1.0, top_k=0, top_p=1.0), logits[:3], 7, [0, 0, 0], 40, 8, ())


def test_fixed_round_gathers_sampled_lane_rows_and_commits_rejection_output(cpu_effective_rows):
    from tensorfold.cuda.streams import Stream
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.deepseek_v4.cuda.engine import (
        FixedDraftDecoder, OP_DRAFT_COMMIT_BATCH, OP_DRAFT_VERIFY_BATCH,
    )

    decoder = FixedDraftDecoder.__new__(FixedDraftDecoder)
    decoder.eos = (99,)
    decoder.verify_rounds_by_active = [0] * 4
    sent = []
    greedy = Stream([1] * 10, 8, out=[5])
    sampled = Stream([1] * 10, 8, Sampling(3, temperature=1.0, top_k=0, top_p=0.5), out=[6])
    for slot, stream in enumerate((greedy, sampled)):
        stream.slot, stream.draft_seed = slot, True
    compact = torch.tensor([[[1., 10., 0.], [1., 11., 0.], [1., 12., 0.], [1., 13., 0.],
                             [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]]])
    peaked = [-9.] * 16
    peaked[4], peaked[2] = 2., 1.                  # only token 4 survives top_p 0.5
    full = torch.tensor([peaked] * 4)

    def announce(op, seq, count, mask=0, *, ids):
        sent.append((op, seq, count, mask, tuple(ids)))
        if op == OP_DRAFT_VERIFY_BATCH:
            return SimpleNamespace(compact=compact, drafts=torch.tensor([[10, 11, 12], [4, 4, 2]]),
                                   full_logits=full)

    decoder.e = SimpleNamespace(_draft_verify=SimpleNamespace(next_sequence=4, depth=3, bucket=3), context=64,
                                target=SimpleNamespace(pool=SimpleNamespace(owned={0: [0], 1: [1]})),
                                announce=announce)
    lanes = [greedy, sampled]
    entries = [decoder._draft_entry(stream) for stream in lanes]
    decoder._fixed_round(lanes, entries)
    assert sent[0][3] == 0b10
    assert sent[1] == (OP_DRAFT_COMMIT_BATCH, 4, 2, 0, (0, 4, 1, 3))
    assert greedy.out == [5, 10, 11, 12, 13] and sampled.out == [6, 4, 4, 4]
