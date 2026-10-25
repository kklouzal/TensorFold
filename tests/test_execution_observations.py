"""Independent enumerated causal-work oracle and bounded observation contracts."""

from types import SimpleNamespace as NS

import pytest

from tensorfold.families.qwen4_exp.cuda.execution_observations import ExecutionObservations, indexer_counts


@pytest.mark.parametrize("ratio,budget", [(3, 8), (4, 8), (4, 11), (4, 2048), (7, 23)])
@pytest.mark.parametrize("position,rows", [(0, 0), (0, 17), (2, 21), (10, 12), (2046, 8), (262140, 4)])
def test_constant_time_counts_match_an_independent_per_query_causal_enumeration(ratio, budget, position, rows):
    groups, scores, keys = 0, 0, 0
    for end in range(position + 1, position + rows + 1):
        complete_groups = [list(range(first, first + ratio)) for first in range(0, end - ratio + 1, ratio)]
        tail = list(range(len(complete_groups) * ratio, end))
        groups += len(complete_groups)
        if len(complete_groups) > budget // ratio:
            scores += len(complete_groups)
            selected = complete_groups[:budget // ratio]
        else:
            selected = complete_groups
        keys += sum(len(group) for group in selected) + len(tail)
    assert indexer_counts(position, rows, ratio, budget) == (groups, scores, keys)


def work(maximum=3):
    result = ExecutionObservations(maximum)
    result.configure_indexer(ratio=4, budget=11, heads=4, target_caches=12)
    return result


def test_submitted_target_and_mtp_roles_do_not_count_replay_as_new_logical_input():
    observed = work()
    original = NS(prompt=[1, 2, 3, 4, 5], cached=2, carry=None)
    replay = NS(prompt=[1, 2, 3, 4, 5, 8], cached=3, carry={"rounds": 1})
    observed.admission(original)
    observed.admission(replay)
    state = NS(pos=10, mtp_len=9)
    observed.target([(state, 0, 3)], prefill=True, replay_rows=3)
    observed.target([(state, 0, 2)])
    observed.mtp([(state, 0, 3)], role="prefill")
    observed.mtp([(state, 0, 1)], role="refresh", zero_proposal_rows=1)
    observed.mtp([(state, 0, 2)], role="proposal")
    observed.publish()
    totals = observed.record()["totals"]
    assert totals["request_admissions"] == totals["replay_admissions"] == 1
    assert totals["logical_prompt_tokens"] == 5
    assert totals["request_prefix_hit_tokens"] == 2 and totals["replay_prefix_hit_tokens"] == 3
    assert totals["target_prefill_rows_submitted"] == totals["target_replay_rows_submitted"] == 3
    assert totals["target_decode_rows_submitted"] == 2
    assert [totals["mtp_" + role + "_rows_submitted"] for role in ("prefill", "refresh", "proposal")] == [3, 1, 2]
    assert totals["mtp_zero_proposal_refresh_rows"] == 1
    assert totals["indexer_query_rows_submitted"] == (3 + 2) * 12 + 3 + 1 + 2
    assert observed.record()["expert_group_max"] is None


def test_completed_initial_mtp_spans_use_the_recorded_pre_forward_position():
    observed = work()
    state = NS(mtp_len=100)
    observed.mtp([(state, 0, 1)], role="refresh", position=7)
    observed.mtp([(state, 0, 2)], role="proposal", position=8)
    observed.publish()
    expected = indexer_counts(7, 3, 4, 11)
    reading = observed.record()["totals"]
    assert reading["indexer_visible_pooled_groups"] == expected[0]
    assert reading["attention_selected_key_rows"] == expected[2]
    assert state.mtp_len == 100


def test_acceptance_histogram_is_fixed_to_the_allocated_method_envelope_above32():
    observed = work(37)
    for offered, accepted in [(0, 0), (3, 0), (3, 2), (3, 3), (37, 37)]:
        observed.verification(offered, accepted)
    observed.publish()
    record = observed.record()
    assert len(record["windows_by_offered_candidates"]) == 38
    assert record["windows_by_offered_candidates"][3] == 3
    assert record["accepted_candidates_by_offered_length"][3] == 5
    assert record["accepted_prefix_histogram_before_text_limits"][0] == 2
    assert record["accepted_prefix_histogram_before_text_limits"][37] == 1
    assert record["totals"]["verified_candidates"] == 46
    assert record["totals"]["accepted_candidates_before_text_limits"] == 42
    assert "after_reply_and_eos_before_text" in record["acceptance_scope"]
    for offered, accepted in [(38, 0), (3, 4), (-1, 0), (True, 0)]:
        with pytest.raises(ValueError, match="allocated"):
            observed.verification(offered, accepted)


def test_publication_and_returned_receipts_do_not_expose_mutable_accumulation():
    observed = work()
    initial = observed.record()
    observed.verification(3, 2)
    assert observed.record() == initial
    observed.publish()
    first = observed.record()
    first["totals"]["verified_candidates"] = 999
    first["windows_by_offered_candidates"][3] = 999
    assert observed.record()["totals"]["verified_candidates"] == 3
    assert observed.record()["windows_by_offered_candidates"][3] == 1


def test_partial_or_unconfigured_work_is_explicitly_unknown_and_counter_exhaustion_is_visible():
    observed = ExecutionObservations(3)
    observed.target([(NS(pos=10), 0, 2)])
    primary = RuntimeError("partial forward")
    observed.incomplete(primary)
    observed.totals["verified_candidates"] = (1 << 63) - 1
    observed.verification(3, 2)
    observed.publish()
    record = observed.record()
    assert record["totals"]["indexer_unknown_rows"] == 2
    assert record["totals"]["incomplete_forward_submissions"] == 1
    assert record["counter_overflow"] and not record["counts_complete"]
    assert record["totals"]["verified_candidates"] == (1 << 63) - 1


def test_observation_failure_preserves_the_primary_forward_failure(monkeypatch):
    observed = work()
    primary, secondary = RuntimeError("forward"), MemoryError("observation")
    def fail(*_args):
        raise secondary
    monkeypatch.setattr(observed, "_add", fail)
    with pytest.raises(RuntimeError) as caught:
        observed.incomplete(primary)
    assert caught.value is primary and primary.__cause__ is secondary


def test_retired_profile_is_required_to_change_indexer_geometry():
    observed = work()
    with pytest.raises(ValueError, match="retired profile"):
        observed.configure_indexer(ratio=4, budget=2048, heads=4, target_caches=12)


def test_preparation_reset_discards_warmup_counts_but_preserves_bound_geometry():
    observed = work()
    observed.admission(NS(prompt=[1, 2, 3], carry=None, cached=1))
    observed.verification(3, 2)
    observed.publish()
    observed.reset()
    reading = observed.record()
    assert reading["indexer_geometry"] == {"ratio": 4, "budget": 11, "heads": 4, "target_caches": 12}
    assert not any(reading["totals"].values())
    assert reading["windows_by_offered_candidates"] == [0, 0, 0, 0]


def test_admission_publication_failure_rolls_back_counters_and_latches_incompleteness(monkeypatch):
    observed = work()
    before = observed.record()
    original = observed.publish
    primary = MemoryError("controlled snapshot publication")
    def fail_after_publication():
        original()
        raise primary
    monkeypatch.setattr(observed, "publish", fail_after_publication)
    with pytest.raises(MemoryError) as caught:
        observed.admission(NS(prompt=[1, 2, 3], cached=1, carry=None))
    assert caught.value is primary
    after = observed.record()
    assert after["totals"] == before["totals"]
    assert after["windows_by_offered_candidates"] == before["windows_by_offered_candidates"]
    assert after["observation_failure"] and not after["counts_complete"]


def test_cached_mtp_boundary_refresh_is_counted_at_its_pre_restore_position():
    observed = work()
    stream = NS(prompt=[1, 2, 3, 4], cached=2, carry=None, st=NS(mtp_len=100))
    observed.admission(stream, prefix_mtp_position=7, zero_proposals=True)
    reading = observed.record()
    assert reading["totals"]["mtp_refresh_rows_submitted"] == 1
    assert reading["totals"]["mtp_zero_proposal_refresh_rows"] == 1
    assert reading["totals"]["indexer_visible_pooled_groups"] == 2
    assert reading["totals"]["attention_selected_key_rows"] == 8


@pytest.mark.parametrize("failure_type", [MemoryError, KeyboardInterrupt])
@pytest.mark.parametrize("phase", ["callback", "publication"])
@pytest.mark.parametrize("cleanup_failed", [False, True])
def test_failed_admission_observation_retires_its_borrowed_state_and_preserves_failures(
        monkeypatch, failure_type, phase, cleanup_failed):
    pytest.importorskip("torch")
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda import multi
    from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController

    decoder = object.__new__(multi.MultiDecoder)
    decoder.controller = AdaptiveController(slots=1, row_budget=4, proposal_limit=3, prefill_tokens=2)
    decoder.controller.work_observations.configure_indexer(ratio=4, budget=11, heads=4, target_caches=12)
    state = NS(history_slot=0, slot_generation=0, limit=128, capacity=128, pos=0, mtp_len=0,
               kv_pair=object(), image_positions=None)
    state.begin_sequence_generation = lambda: setattr(state, "slot_generation", state.slot_generation + 1)
    decoder.w = NS(comm=None, meta={"execution_provider": "cuda"})
    decoder.capacity, decoder.depth, decoder.prefill_rows = 128, 3, 2
    decoder.kv_pair = state.kv_pair
    decoder.free, decoder.states, decoder.failed_slots = [state], (state,), []
    decoder.streams, decoder.filling, decoder.fills, decoder.held = {}, [], {}, {}
    decoder.kept, decoder.used, decoder.clock = [], {}, 0
    decoder.graphs, decoder.full_resident = None, True
    decoder.buf, decoder.pbuf, decoder.mbuf = NS(rows=4), NS(rows=2), None
    decoder._draft_map = decoder.points = decoder.vision = None
    decoder.next_id, decoder.copy = 0, False
    decoder._grow = lambda *_args, **_kwargs: True
    primary, cleanup = failure_type("admission observation failed"), RuntimeError("exclusive retirement failed")
    retired = []
    def shrink(selected, *, force=False):
        retired.append((selected, force))
        if cleanup_failed:
            raise cleanup
    decoder._shrink = shrink
    def prepared(engine, prompt, **kwargs):
        state.prepared = True
        return 0
    monkeypatch.setattr(multi, "prefill_begin", prepared)
    monkeypatch.setattr(multi.image_rows, "begin", lambda *args: None)
    observed = decoder.controller.work_observations
    before = observed.record()["totals"]
    def fail(*args, **kwargs):
        raise primary
    monkeypatch.setattr(observed, "admission" if phase == "callback" else "publish", fail)
    with pytest.raises(failure_type) as caught:
        decoder.admit(Stream([11, 12, 13, 14], 3, draft=False))
    assert caught.value is primary and state.prepared
    assert decoder.streams == {} and decoder.filling == [] and decoder.fills == {}
    assert observed.record()["totals"] == before
    assert observed.record()["observation_failure"] and not observed.record()["counts_complete"]
    assert retired == [(state, True)]
    if cleanup_failed:
        assert primary.__cause__ is cleanup
        assert decoder.free == [] and decoder.failed_slots == [state]
    else:
        assert decoder.free == [state] and decoder.failed_slots == []
