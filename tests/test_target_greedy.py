"""Target feedback ownership/control with independent full-vocabulary CPU oracles.

CPU kernels below are control substitutes; actual CUDA arithmetic is covered by
its maintained SM70 component test and remains a required target gate.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.sm70 import greedy_native  # noqa: E402
from tensorfold.cuda.streams import Stream, accept  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import target_greedy as target  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.kv_status import KVStatusCohort  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import KVNumericError  # noqa: E402
from test_flashnext_kv_status_cohort import states_for  # noqa: E402


class Box:
    def __init__(self, **values):
        self.__dict__.update(values)


def original_packet(shard, offset):
    maximum = shard.max(dim=1).values
    ids = torch.arange(offset, offset + shard.shape[1], dtype=torch.int64)
    chosen = [int(ids[row == value].min()) if not torch.isnan(value) else np.iinfo(np.int64).max
              for row, value in zip(shard, maximum)]
    result = torch.empty((shard.shape[0], 2), dtype=torch.int64)
    result[:, 0] = maximum.view(torch.int32).to(torch.int64)
    result[:, 1] = torch.tensor(chosen, dtype=torch.int64)
    return result


class CPUKernels:
    def target_local(self, logits, packet, offset, vocabulary):
        packet.copy_(original_packet(logits, offset))

    def target_accept(self, frames, inputs, eos, feedback, room, vocabulary, stop_eos):
        scores = frames[:, :, 0].to(torch.int32).view(torch.float32).numpy()
        ids = frames[:, :, 1].numpy()
        chosen, flags = [], 0
        for values, tokens in zip(scores.T, ids.T):
            if np.isnan(values).any():
                flags |= 1
            if np.isposinf(values).any():
                flags |= 2
            if np.isneginf(values).all():
                flags |= 4
            token = int(tokens[np.lexsort((tokens, -values))[0]])
            if not 0 <= token < vocabulary:
                flags |= 8
            if len(set(tokens.tolist())) != len(tokens):
                flags |= 16
            chosen.append(token if 0 <= token < vocabulary else 0)
        path, end = accept(inputs.tolist(), list(range(-1, len(inputs) - 1)), chosen, room,
                           eos.tolist() if stop_eos else ())
        committed = [int(inputs[row]) for row in path[1:]] + [end]
        feedback.zero_()
        feedback[0], feedback[1] = len(path), flags
        feedback[2:2 + len(committed)] = torch.tensor(committed, dtype=torch.int32)


def setup(monkeypatch, shards, window, *, eos=(), room=None, sampling=None, status=None):
    world, rows = len(shards), len(window)
    assert world == 8
    widths = tuple(shard.shape[1] for shard in shards)
    packets = torch.stack([original_packet(shard, sum(widths[:rank])) for rank, shard in enumerate(shards)])
    words = torch.zeros((world, 1, 2), dtype=torch.int32) if status is None else status
    if status is None:
        words[:, :, 1] = -2
    states = states_for(words[0].tolist(), world)
    state = states[0]
    state.media_profile_identity = None
    calls = []
    def gather(send, receive):
        if send.dtype == torch.int64:
            calls.append("maxima")
            torch.testing.assert_close(send, packets[0], rtol=0, atol=0)
            receive.copy_(packets.reshape(-1))
        else:
            calls.append("status")
            receive.copy_(words.reshape(-1))
    comm = Box(world=world, rank=0, all_gather=gather)
    weights = Box(device="cpu", cfg=Box(vocab=sum(widths), eos=tuple(eos)), comm=comm, dspark=None,
                  meta={"sm70_target_greedy_trial": {"policy": target.POLICY}})
    buffer = Box(rows=rows, cand=torch.empty(rows, 65), cand_all=torch.empty(world * rows * 65),
                 ids=torch.tensor(window, dtype=torch.int32))
    checked = []
    plan = Box(mapping=None, graph_ids_safe=True, offset=0, widths=widths, total_width=sum(widths),
               check_model=lambda w, logits: checked.append(tuple(logits.shape)))
    decoder = Box(w=weights, buf=buffer, _sample_plan=plan, graphs=None,
                  kv_cohort=KVStatusCohort(1, comm, "cpu", candidate_rows=rows), confidence=.7)
    monkeypatch.setattr(greedy_native, "extension", lambda: CPUKernels())
    # Only substitute the CUDA tensor-location check; real policy, status and
    # final feedback checks and the actual production owner all run normally.
    monkeypatch.setattr(target, "validate_rows", lambda logits, positions, policy: target.validate_policy(positions, policy))
    owner = target.TargetGreedy(decoder)
    stream = Stream([], (rows if room is None else room) + 1, sampling=sampling, st=state,
                    out=[window[0]], drafts=list(window[1:]))
    return owner, decoder, stream, packets, calls, checked


def scores(rows, widths, pattern):
    rng = np.random.default_rng(912)
    shards = [torch.from_numpy(rng.normal(size=(rows, width)).astype(np.float32)) for width in widths]
    if pattern == "ties":
        for shard in shards:
            shard.zero_()
    elif pattern == "zeros":
        for shard in shards:
            shard.view(torch.int32).reshape(-1).copy_(torch.from_numpy(np.resize(
                np.asarray([0, 0x80000000], dtype=np.uint32).view(np.int32), shard.numel())))
    elif pattern == "subnormal":
        for shard in shards:
            shard.view(torch.int32).reshape(-1).copy_(torch.from_numpy(np.resize(np.asarray(
                [0, 1, 0x80000001, 0x007fffff, 0x807fffff], dtype=np.uint32).view(np.int32), shard.numel())))
    return shards


def full_reference(shards):
    full = torch.cat(shards, dim=1).numpy()
    ids = np.arange(full.shape[1], dtype=np.int64)
    return [int(ids[np.lexsort((ids, -row))[0]]) for row in full]


@pytest.mark.parametrize("rows", [1, 2, 3, 32])
@pytest.mark.parametrize("pattern", ["random", "ties", "zeros", "subnormal"])
@pytest.mark.parametrize("kind", ["full", "partial", "room", "eos"])
def test_actual_owner_matches_full_vocabulary_and_original_chain_oracle(monkeypatch, rows, pattern, kind):
    shards = scores(rows, (1, 7, 17, 31, 33, 41, 67, 71), pattern)
    sampled = full_reference(shards)
    window = [11] + sampled[:-1]
    if kind == "partial" and rows > 1:
        window[max(1, rows // 2)] = (window[max(1, rows // 2)] + 1) % sum(x.shape[1] for x in shards)
    room = max(1, rows // 2) if kind == "room" else rows
    eos = (sampled[max(0, rows // 2 - 1)],) if kind == "eos" else ()
    expected = accept(window, list(range(-1, rows - 1)), sampled, room, eos)
    committed = [window[row] for row in expected[0][1:]] + [expected[1]]
    owner, decoder, stream, _packets, calls, checked = setup(monkeypatch, shards, window, eos=eos, room=room)
    assert owner.eligible(decoder, [stream], pieces=(), grammars={}, dflash=None)
    owner.gather(shards[0], rows)
    actual, new = owner.finish(decoder, [(stream.st, 0, rows)], stream, shards[0])
    assert actual == expected and new == committed
    assert calls == ["maxima", "status"] and checked == [(rows, shards[0].shape[1])]
    assert owner.completed == 1 and decoder.confidence == .7
    assert not stream.st._kv_pending and stream.drafts == window[1:]
    decoder.kv_cohort.host.fill_(-55)
    assert new == committed  # Escaped IDs are ordinary owning Python integers.


@pytest.mark.parametrize("fault", ["nan", "inf", "all-masked", "duplicate", "invalid-id"])
def test_every_produced_row_is_validated_even_after_a_first_draft_mismatch(monkeypatch, fault):
    shards = scores(3, (71,) * 8, "random")
    if fault == "nan":
        shards[7][2, -1] = float("nan")
    elif fault == "inf":
        shards[6][2, 0] = float("inf")
    elif fault == "all-masked":
        for shard in shards:
            shard[2].fill_(-float("inf"))
    mismatch = (full_reference(shards)[0] + 1) % sum(shard.shape[1] for shard in shards)
    owner, decoder, stream, packets, _calls, _checked = setup(monkeypatch, shards, [11, mismatch, 17])
    if fault == "duplicate":
        packets[7, 2, 1] = packets[0, 2, 1]
    elif fault == "invalid-id":
        packets[7, 2, 0] = torch.tensor(1e30, dtype=torch.float32).view(torch.int32)
        packets[7, 2, 1] = owner.w.cfg.vocab
    owner.gather(shards[0], 3)
    with pytest.raises(ValueError):
        owner.finish(decoder, [(stream.st, 0, 3)], stream, shards[0])
    assert owner.completed == 0 and stream.out == [11]


def test_codec_status_retains_precedence_over_bad_scores_and_bad_feedback(monkeypatch):
    shards = scores(3, (7,) * 8, "ties")
    shards[7][2, 0] = float("nan")
    status = torch.zeros((8, 1, 2), dtype=torch.int32)
    status[:, :, 1] = -2
    status[7, 0] = torch.tensor([4, -1], dtype=torch.int32)
    owner, decoder, stream, _packets, _calls, _checked = setup(monkeypatch, shards, [3, 2, 1], status=status)
    owner.gather(shards[0], 3)
    with pytest.raises(KVNumericError, match="rank 7"):
        owner.finish(decoder, [(stream.st, 0, 3)], stream, shards[0])
    assert stream.st._kv_error is not None and owner.completed == 0


@pytest.mark.parametrize("path", ["stochastic", "invalid-policy", "constraint", "owed", "probabilities", "token-limit",
                                     "media", "copies", "many", "pieces", "graph", "dflash"])
def test_required_original_paths_are_declined_without_mutating_transcript(monkeypatch, path):
    owner, decoder, stream, _packets, _calls, _checked = setup(monkeypatch, scores(2, (7,) * 8, "random"), [3, 2])
    options = {"pieces": (), "grammars": {}, "dflash": None}
    live = [stream]
    if path == "stochastic":
        stream.sampling = Sampling(17, .7)
    elif path == "invalid-policy":
        stream.sampling = Sampling(17, float("nan"))
    elif path == "constraint":
        stream.constraint = object()
    elif path == "owed":
        stream.owed = [2]
    elif path == "probabilities":
        stream.probabilities = []
    elif path == "token-limit":
        stream.token_limit = lambda *_args: (1, False)
    elif path == "media":
        stream.vision = object()
    elif path == "copies":
        stream.copies = object()
    elif path == "many":
        live = [stream, stream]
    elif path == "pieces":
        options["pieces"] = [object()]
    elif path == "graph":
        decoder.graphs = object()
    else:
        options["dflash"] = object()
    before = (list(stream.out), list(stream.drafts), stream.st.pos)
    assert not owner.eligible(decoder, live, **options)
    assert before == (stream.out, stream.drafts, stream.st.pos)


def test_unknown_native_submission_failure_is_not_an_optional_fallback(monkeypatch):
    owner, decoder, stream, _packets, _calls, _checked = setup(monkeypatch, scores(2, (7,) * 8, "random"), [3, 2])
    primary = MemoryError("native writer failed after submission")
    def fail(*_args): raise primary
    owner.native.target_local = fail
    with pytest.raises(MemoryError) as caught:
        owner.gather(torch.zeros((2, 7)), 2)
    assert caught.value is primary and owner.error is primary
    stream.sampling = Sampling(17, .7)
    with pytest.raises(RuntimeError, match="requires model retirement") as stopped:
        owner.eligible(decoder, [stream], pieces=(), grammars={}, dflash=None)
    assert stopped.value.__cause__ is primary


def test_retirement_keeps_eos_and_borrowed_storage_until_positive_model_fence(monkeypatch):
    owner, _decoder, _stream, _packets, _calls, _checked = setup(monkeypatch, scores(1, (7,) * 8, "random"), [3])
    eos = owner.eos
    with pytest.raises(ValueError, match="actual model device fence"):
        owner.retire_after_fence(fenced=False)
    assert owner.eos is eos
    owner.retire_after_fence(fenced=True)
    assert owner.retired and all(getattr(owner, name) is None for name in ("eos", "local", "frames", "feedback"))


def test_feedback_owner_reuses_existing_data_envelope_and_returns_no_pinned_alias():
    words = torch.zeros((8, 1, 2), dtype=torch.int32)
    words[:, :, 1] = -2
    states = states_for(words[0].tolist(), 8)
    comm = Box(rank=0, world=8, all_gather=lambda send, receive: receive.copy_(words.reshape(-1)))
    owner = KVStatusCohort(1, comm, "cpu", candidate_rows=32)
    value = torch.arange(34, dtype=torch.int32)
    received = owner.read_feedback(states, value, comm=comm)
    np.testing.assert_array_equal(received, value.numpy())
    assert received.flags.owndata and not np.shares_memory(received, owner.host.numpy())
    owner.host.fill_(-1)
    np.testing.assert_array_equal(received, value.numpy())
    device, host = target.workspace_bytes(248320)
    assert device == 4 * 248320 and host > device


@pytest.mark.parametrize("graph_head", [False, True])
def test_actual_forward_epilogue_routes_only_selected_target_without_topk_packet(monkeypatch, graph_head):
    from tensorfold.families.qwen4_exp.cuda import forward

    shards = scores(3, (7,) * 8, "random")
    owner, decoder, _stream, _packets, calls, _checked = setup(monkeypatch, shards, [3, 2, 1])
    decoder.w.cfg.eps, decoder.w.cfg.streams, decoder.w.cfg.low = 1e-6, 1, 1
    decoder.w.head = object()
    buffer = decoder.buf
    buffer.streams, buffer.h = torch.empty((3, 1)), torch.zeros((3, 1))
    buffer.prefill, buffer.mixed, buffer.xs_mixed, buffer.logits = False, torch.empty(3, 1), torch.empty(3, 1), shards[0]
    monkeypatch.setattr(forward, "_writeback", lambda *_args, **_kw: None)
    monkeypatch.setattr(forward, "_readout", lambda *_args, **_kw: None)
    monkeypatch.setattr(forward, "_mm", lambda *_args, **_kw: shards[0])
    monkeypatch.setattr(forward, "candidates", lambda *_args, **_kw: pytest.fail("built top32/LSE target packet"))
    result = forward.finish(decoder.w, object(), buffer, 3, None,
                            **({"gather_candidates": False} if graph_head else {"target_greedy": owner}))
    assert result is shards[0] and calls == ([] if graph_head else ["maxima"])


@pytest.mark.parametrize("use_graph", [False, True])
def test_actual_round_uses_gpu_accepted_path_and_keeps_original_draft_inputs(monkeypatch, use_graph):
    from types import MethodType
    from tensorfold.families.qwen4_exp.cuda import multi

    shards = scores(3, (7,) * 8, "random")
    sampled = full_reference(shards)
    window = [11, sampled[0], (sampled[1] + 1) % 56]
    owner, shell, stream, _packets, calls, _checked = setup(monkeypatch, shards, window, room=10)
    decoder = object.__new__(multi.MultiDecoder)
    decoder.w, decoder.buf, decoder.kv_cohort = shell.w, shell.buf, shell.kv_cohort
    decoder.w.meta["target_greedy_owner"] = owner
    decoder._sample_plan, decoder._execution_plan = shell._sample_plan, None
    decoder.controller, decoder.graphs, decoder.pdl = None, None, False
    decoder.pbuf, decoder.eos = Box(), shell.w.cfg.eos
    decoder.streams, decoder.filling, decoder.held = {stream.sid: stream}, [], {}
    decoder.converged, decoder.round_s, decoder.row_s, decoder.confidence = False, None, None, .7
    decoder._make_room = lambda: []
    decoder._runnable = lambda value: True
    decoder._rotor_lookup = lambda *args: False
    decoder._migrate_boundary = lambda *args: None
    decoder.buf.h = torch.empty((3, 1), dtype=torch.float16)
    decoder.kv_host = torch.zeros((1, 2), dtype=torch.int32)
    commits, drafted = [], []
    def stage(weights, buffer, windows, **options):
        assert windows == [(stream.st, window)]
        buffer.ids.copy_(torch.tensor(window, dtype=torch.int32))
        return [(stream.st, 0, 3)]
    def compute(weights, segments, buffer, **options):
        assert options == {"check": False, "target_greedy": owner}
        owner.gather(shards[0], 3)
        return shards[0]
    def commit(weights, state, buffer, produced, keep, **options):
        commits.append((produced, keep))
        state.pos += keep
    monkeypatch.setattr(multi, "stage", stage)
    monkeypatch.setattr(multi.pdl, "ON", False)
    monkeypatch.setattr(multi, "compute", compute)
    monkeypatch.setattr(multi, "commit", commit)
    if use_graph:
        from tensorfold.families.qwen4_exp.cuda.prefix_policy import OWNER_TARGET_GREEDY_PROFILE

        decoder.w.meta["sm70_profile"] = OWNER_TARGET_GREEDY_PROFILE
        def replay(segments):
            assert segments == [(stream.st, 0, 3)]
            calls.append("graph")
            return shards[0]
        decoder.graphs = Box(bound=Box(unique=True, native_owner=True), verify_forward=replay)
        monkeypatch.setattr(multi, "compute", lambda *a, **kw: pytest.fail("eligible graph was declined"))
    monkeypatch.setattr(multi, "choose_gathered", lambda *a, **kw: pytest.fail("CPU chooser after device verification"))
    monkeypatch.setattr(multi, "accept", lambda *a, **kw: pytest.fail("CPU acceptance after device verification"))
    decoder._candidate_readback = lambda *a, **kw: pytest.fail("copied full target score packet")
    decoder._draft_all = lambda streams, new: drafted.append((streams, new, decoder.confidence))
    decoder._ends = MethodType(multi.MultiDecoder._ends, decoder)
    expected_path, end = accept(window, [-1, 0, 1], sampled, 10)
    committed = [window[row] for row in expected_path[1:]] + [end]
    assert decoder._round() == []
    assert stream.out == [11] + committed and stream.committed == window[:len(expected_path)]
    assert commits == [(3, len(expected_path))]
    assert drafted[0][0][0][2] == committed and drafted[0][2] == .7
    assert (stream.rounds, stream.drafted, stream.accepted) == (1, 2, len(expected_path) - 1)
    assert calls == (["graph", "maxima", "status"] if use_graph else ["maxima", "status"])


@pytest.mark.parametrize("profile,unique,native,allowed", [
    ("source-owner-target-guard-trial", True, True, True),
    ("source-owner-target-greedy-trial", True, True, True),
    ("source-owner-target-greedy-trial", False, True, False),
    ("source-owner-target-greedy-trial", True, False, False),
    ("source-owner-pieces-trial", True, True, False),
    ("source-target-greedy-trial", True, True, False)])
def test_graph_target_authority_requires_the_explicit_native_piece_composition(monkeypatch, profile, unique, native, allowed):
    owner, decoder, stream, _packets, _calls, _checked = setup(monkeypatch, scores(2, (7,) * 8, "random"), [3, 2])
    decoder.w.meta["sm70_profile"] = profile
    decoder.graphs = Box(bound=Box(unique=unique, native_owner=native))
    assert owner.eligible(decoder, [stream], pieces=(), grammars={}, dflash=None) is allowed
