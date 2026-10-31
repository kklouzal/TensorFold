"""Current-operation DFlash draws, rejection, stop and publication ownership."""

from types import SimpleNamespace as NS
import threading

import numpy as np
import pytest
import torch

from tensorfold.engine.rejection import ChainPacket, ProbabilityRow
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda import multi
from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry

from test_flashnext_adaptive_controller import decoder_round  # noqa: F401 (pytest fixture)


@pytest.fixture
def dflash_round(request, monkeypatch):
    decoder, executed = request.getfixturevalue("decoder_round")
    events = []
    decoder.depth, decoder.capacity, decoder.logical_context = 2, 1004, 1000
    decoder.buf.rows = 6
    decoder.speculation = MethodGeometry("dflash2", 2)
    decoder._sample_plan = None
    for stream in decoder.streams.values():
        state = stream.st
        state.slot_generation = 3
        state.dflash2 = NS(end=state.pos, generation=3, enabled=True,
                          profile_identity="fixture-only", check=lambda **_kwargs: None)
        state.discard_target_tentative = lambda state=state: events.append(("discard", state))
        take = stream.take

        def publish(tokens, eos, take=take):
            events.append(("publish", list(tokens)))
            return take(tokens, eos)

        stream.take = publish

    def propose(states, anchors, counts, samplings, positions):
        events.append(("draw", list(counts), list(positions), list(anchors)))
        return [ChainPacket(tuple([20, 30][:count]), tuple(ProbabilityRow([token], [1.0])
                   for token in [20, 30][:count]), position) for count, position in zip(counts, positions)]

    def maintain(accepted):
        events.append(("maintain", [(state, count) for state, count in accepted]))
        for state, count in accepted:
            assert state.dflash2.end + count == state.pos
            state.dflash2.end += count

    decoder.w.dflash2 = NS(propose_many=propose, commit_many=maintain)

    def compute(weights, segments, buffers, **kwargs):
        events.append(("forward", [end - begin for _, begin, end in segments]))
        logits = torch.full((segments[-1][2], 64), -torch.inf)
        for _state, begin, end in segments:
            for row, token in enumerate([20, 30, 40][:end - begin]):
                logits[begin + row, token] = 0
        return logits

    def commit(weights, state, buffers, rows, kept, **kwargs):
        assert kwargs["commit_dflash"] is False
        events.append(("target_commit", state, kept))
        state.pos += kept

    monkeypatch.setattr(multi, "compute", compute)
    monkeypatch.setattr(multi, "commit", commit)
    # This suite exercises the coordinator on CPU. Actual CUDA extraction has
    # its own provider tests; keep its shipping CUDA boundary intact.
    monkeypatch.setattr("tensorfold.cuda.sampling.sample_rows",
                        lambda logits, positions, sampling: logits.argmax(-1).tolist())
    monkeypatch.setattr(multi, "launch_streams", lambda *args: pytest.fail("legacy sample matching launched"))
    decoder._draft_all = lambda *args, **kwargs: pytest.fail("a next-round DFlash chain was drawn")
    return decoder, executed, events


def test_draws_use_current_frozen_budget_and_verify_in_the_same_round(dflash_round):
    decoder, executed, events = dflash_round
    decoder.controller.request(proposal_limit=0, row_budget=2)
    decoder.round()
    assert events[0] == ("draw", [0, 0], [2, 2], [10, 10])
    assert events[1] == ("forward", [1, 1])
    assert executed[-1] == [decoder.streams[sid].st for sid in [0, 1]]
    assert all(event[0] != "publish" for event in events[:5])
    assert [kind for kind, *_ in events] == ["draw", "forward", "target_commit", "target_commit",
                                           "maintain", "publish", "publish"]
    assert all(not stream.drafts for stream in decoder.streams.values() if stream.sid in (0, 1))
    assert all(decoder.streams[sid].st.dflash2.end == 2 for sid in (0, 1))


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_stop_limits_all_state_before_any_publication(dflash_round, count):
    decoder, _executed, events = dflash_round
    for stream in decoder.streams.values():
        stream.token_limit = lambda new, prefix: (count, True)
    ended = decoder.round()
    assert [stream.sid for stream in ended] == [0, 1]
    for stream in ended:
        assert stream.out == [10, *[20, 30, 40][:count]]
        assert stream.committed == [10, 20, 30][:count]
        assert stream.st.pos == stream.st.dflash2.end == 1 + count
    kinds = [event[0] for event in events]
    if count:
        assert kinds.index("maintain") < kinds.index("publish")
    else:
        assert "publish" not in kinds and "target_commit" not in kinds


def test_reply_context_and_last_row_remainder_cap_counts_before_drawing(dflash_round):
    decoder, _executed, events = dflash_round
    decoder.controller.request(row_budget=5)
    decoder.streams[0].count = 2  # one output remains after the already emitted anchor
    decoder.logical_context = 4
    decoder.round()
    assert events[0][1] == [0, 1, 1]
    assert events[1] == ("forward", [1, 2, 2])
    assert sum(count + 1 for count in events[0][1]) == 5


def test_request_draft_false_runs_serial_without_proposals_or_conditioning(dflash_round):
    decoder, _executed, events = dflash_round
    for stream in decoder.streams.values():
        stream.draft = False
        stream.st.dflash2.enabled = False
    decoder.round()
    assert all(event[0] not in ("draw", "maintain") for event in events)
    assert all(stream.st.pos == 2 and stream.st.dflash2.end == 1 for stream in decoder.streams.values())


def test_mixed_serial_and_enabled_requests_keep_distinct_method_participation(dflash_round):
    decoder, _executed, events = dflash_round
    serial = decoder.streams[0]
    serial.draft, serial.st.dflash2.enabled = False, False
    decoder.round()
    assert events[0][1] == [2, 1]
    maintained = next(event[1] for event in events if event[0] == "maintain")
    assert all(state is not serial.st for state, _rows in maintained)
    assert serial.st.pos == 2 and serial.st.dflash2.end == 1
    assert decoder.streams[1].st.pos == decoder.streams[1].st.dflash2.end == 4


def test_sampled_rejection_commits_only_anchor_and_publishes_target_residual(dflash_round, monkeypatch):
    decoder, _executed, events = dflash_round
    for stream in decoder.streams.values():
        stream.sampling = Sampling(1234, .7)
    # Independent p assigns zero mass to the proposed20, so first-candidate
    # rejection and the point-mass residual50 are certain for every RNG value.
    monkeypatch.setattr("tensorfold.cuda.sampling.effective_rows",
                        lambda logits, sampling: [ProbabilityRow([50], [1.0]) for _ in logits])
    decoder.round()
    assert events[0][1] == [2, 2]
    for sid in (0, 1):
        stream = decoder.streams[sid]
        assert stream.out == [10, 50] and stream.committed == [10]
        assert stream.st.pos == stream.st.dflash2.end == 2


def test_grammar_pruning_retains_original_q_packet_and_unreachable_tails(dflash_round):
    decoder, _executed, events = dflash_round

    def mask(logits, *args):
        logits.fill_(-torch.inf)
        logits[:, 50] = 0  # independent grammar permits only50, rejecting q's20
        return logits

    for stream in decoder.streams.values():
        stream.constraint = NS(window=lambda tokens, parents: NS(tokens=tokens[:1]),
                               mask=mask, advance=lambda tokens: None)
    decoder.round()
    assert events[0][1] == [2, 2] and events[1] == ("forward", [1, 1])
    assert decoder.streams[0].out == [10, 50]
    assert decoder.streams[1].out == [10, 50]


def join_replay(decoder, source, monkeypatch):
    """The real prompt-to-decode coordinator adopts Stream.continued's transcript."""
    replay = source.continued()
    replay.sid = source.sid
    replay.st = NS(pos=len(replay.prompt), slot_generation=3,
                   discard_target_tentative=lambda: None)
    replay.st.dflash2 = NS(end=replay.st.pos, generation=3, enabled=source.draft,
                          profile_identity="fixture-only", check=lambda **_kw: None)
    decoder.streams, decoder.filling = {}, [replay]
    engine = NS(mbuf=None, buf=decoder.buf, sample=lambda *_a, **_k: pytest.fail("replay anchor was redrawn"))
    decoder.fills = {replay.sid: [engine, False, 0, [], None]}
    monkeypatch.setattr(multi, "_kv_check", lambda *_a, **_k: None)
    monkeypatch.setattr(multi.image_rows, "finish", lambda *_a: None)
    decoder._joined([(replay, 0, len(replay.prompt))], torch.zeros((1, 64)), [torch.zeros((1, 1))], 0.)
    return replay


@pytest.mark.parametrize("before,after", [(1, 0), (0, 3)])
def test_actual_selector_rejection_replay_forces_published_ids_across_policy_changes(
        dflash_round, monkeypatch, before, after):
    from tensorfold.families.qwen4_exp.cuda.dflash2_provider import selector_chain

    decoder, _executed, events = dflash_round
    from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController

    decoder.controller = AdaptiveController(slots=5, row_budget=8, proposal_limit=3, prefill_tokens=2048)
    decoder.controller.work_observations.configure_indexer(ratio=4, budget=2048, heads=4, target_caches=12)
    decoder.depth, decoder.buf.rows, decoder.speculation = 3, 8, MethodGeometry("dflash2", 3)
    source = decoder.streams[0]
    source.prompt, source.out, source.context = [1] * 10, [12], [1] * 10 + [12]
    source.st.pos = source.st.dflash2.end = 10
    source.sampling, source.background = Sampling(0, 1.), True
    emitted, stops, draws = [], [], []
    source.emit = lambda new: emitted.append(list(new)) or False
    source.token_limit = lambda new, prefix: stops.append((list(new), list(prefix))) or (len(new), False)
    decoder.streams = {source.sid: source}
    p = ProbabilityRow(np.arange(16), np.ones(16))
    values = np.log(np.asarray([.9] + [.1 / 15] * 15)).astype(np.float32)

    def propose(states, anchors, counts, samplings, positions):
        draws.append(list(counts))
        return [selector_chain(np.broadcast_to(np.arange(16), (count, 16)),
                   np.broadcast_to(values, (count, 16, 16)), sampling=sampling, first_position=position)
                if count else ChainPacket((), (), position)
                for count, sampling, position in zip(counts, samplings, positions)]

    decoder.w.dflash2.propose_many = propose
    monkeypatch.setattr(multi, "compute", lambda weights, segments, buffers, **kwargs:
                        torch.zeros((segments[-1][2], 16)))  # independent uniform-target fixture, including bonus row
    monkeypatch.setattr("tensorfold.cuda.sampling.effective_rows", lambda logits, sampling: [p for _ in logits])
    decoder.controller.request(proposal_limit=before, row_budget=before + 1)
    decoder.round()
    canonical = list(source.out)
    assert canonical[:2] == [12, 5 if before else 13]  # The independent D14 seed0 witness.
    delivered = list(emitted)
    replay = join_replay(decoder, source, monkeypatch)
    assert replay.out == [12] and replay.owed == canonical[1:]
    assert emitted == delivered
    decoder.controller.request(proposal_limit=after, row_budget=after + 1)
    draws.clear()
    stops.clear()
    while replay.owed:
        decoder.round()
    assert replay.out == canonical and replay.error is None
    assert replay.st.pos == replay.st.dflash2.end == len(source.prompt) + len(canonical) - 1
    assert not draws and not stops and emitted == delivered
    decoder.round()  # Only the new suffix uses the current frozen policy.
    assert draws == [[after]] and len(emitted) > len(delivered)
    assert decoder.controller.record()["work_observations"]["totals"]["target_replay_rows_submitted"] == len(canonical) - 1


def test_replay_first_anchor_multiple_yields_grammar_and_cancellation(dflash_round, monkeypatch):
    decoder, _executed, events = dflash_round
    source = decoder.streams[0]
    source.out, source.context = [10, 5, 6, 7], [1, 10, 5, 6, 7]
    delivered, grammar = [], []
    source.emit = lambda new: delivered.append(list(new))
    source.cancel = threading.Event()

    class Grammar:
        def fresh(self):
            return Grammar()

        def advance(self, tokens):
            grammar.extend(tokens)

        def mask(self, *_a):
            pytest.fail("canonical replay invoked grammar sampling mask")

        def window(self, *_a):
            pytest.fail("canonical replay built a proposal grammar window")

    source.constraint = Grammar()
    source.token_limit = lambda *_a: pytest.fail("canonical replay invoked text-stop callback")
    replay = join_replay(decoder, source, monkeypatch)
    decoder.streams = {replay.sid: replay}
    decoder.controller.request(proposal_limit=0, row_budget=1)
    decoder.round()
    assert replay.out == [10, 5] and replay.owed == [6, 7]
    again = join_replay(decoder, replay, monkeypatch)  # Always-yields during reconstruction too.
    while again.owed:
        decoder.round()
    assert again.out == source.out and not delivered and again.error is None
    assert grammar == [10, 5, 10, 5, 6, 7]
    # Cancellation at the coordinator boundary does not wait for owed tokens
    # or invoke a fresh-output callback to discover disconnection.
    cancelled = join_replay(decoder, again, monkeypatch)
    events.clear()
    cancelled.cancel.set()
    assert decoder.round() == [cancelled]
    assert cancelled.done and cancelled.owed == [5, 6, 7] and not events and not delivered


def test_first_anchor_only_replay_and_context_edge_allow_only_one_fresh_row(dflash_round, monkeypatch):
    decoder, _executed, events = dflash_round
    source = decoder.streams[0]
    source.count = 2
    decoder.logical_context = len(source.prompt) + source.count
    delivered = []
    source.emit = lambda new: delivered.append(list(new))
    replay = join_replay(decoder, source, monkeypatch)
    assert replay.out == [10] and not replay.owed and not delivered
    decoder.round()
    assert next(event[1] for event in events if event[0] == "forward") == [1]
    assert replay.done and len(replay.out) == 2 and replay.st.pos == replay.st.dflash2.end == 2
    assert delivered == [[20]]


def test_reconstruction_failure_does_not_consume_owed_or_publish(dflash_round, monkeypatch):
    decoder, _executed, events = dflash_round
    source = decoder.streams[0]
    source.out, source.context = [10, 5], [1, 10, 5]
    delivered = []
    source.emit = lambda new: delivered.append(new)
    replay = join_replay(decoder, source, monkeypatch)
    primary = RuntimeError("accepted-feature reconstruction failed")

    def fail(_accepted):
        raise primary

    decoder.w.dflash2.commit_many = fail
    with pytest.raises(RuntimeError) as caught:
        decoder.round()
    assert caught.value is primary
    assert replay.out == [10] and replay.owed == [5] and not delivered
    assert replay.st.pos == 2 and replay.st.dflash2.end == 1
