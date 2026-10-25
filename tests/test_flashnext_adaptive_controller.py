"""Execution bounds and distributed decisions; no GPU/performance claims."""

from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController


def controller(slots=40, rows=32, proposals=3):
    return AdaptiveController(slots=slots, row_budget=rows, proposal_limit=proposals, prefill_tokens=2048)


@pytest.mark.parametrize("slots", [1, 18, 32, 40, 81])
def test_every_capacity_endpoint_is_served_without_a_session_count_ceiling(slots):
    control = controller(slots)
    served = []
    sessions = list(range(slots))
    for _ in range(-(-slots // 8)):
        plan = control.freeze(sessions)
        control.accept(plan)
        chosen = control.select([(sid, 4) for sid in sessions], plan)
        assert 1 <= len(chosen) <= 8
        served.extend(chosen)
    assert set(served) == set(sessions)


def test_single_request_runs_immediately_and_a_demand_spike_stays_inside_the_arena():
    control = controller()
    single = control.freeze([0])
    control.accept(single)
    assert control.select([(0, 4)], single) == [0]
    maximum = control.freeze(list(range(40)))
    control.accept(maximum)
    selected = control.select([(sid, 4) for sid in range(40)], maximum)
    assert selected == list(range(1, 9))
    assert maximum["row_budget"] == 32
    assert control.maximum["row_budget"] == 32


def test_requests_coalesce_and_only_a_boundary_changes_effective_values():
    control = controller()
    control.request(session_ceiling=2, proposal_limit=0)
    record = control.request(prefill_tokens=512, row_budget=1)
    assert control.admission_limit() == 40
    assert record["pending"] == record["configured"]
    assert record["effective"]["proposal_limit"] == 3
    control.boundary()
    assert control.admission_limit() == 2
    assert control.record()["effective"] == {"proposal_limit": 0, "row_budget": 1,
                                            "prefill_tokens": 512, "session_ceiling": 2}
    # Existing sessions remain executable after admission stops at the lower
    # ceiling. Session count is not used as a memory reclamation instruction.
    plan = control.freeze(list(range(40)))
    control.accept(plan)
    assert control.select([(sid, 1) for sid in range(40)], plan) == [0]


def test_a_wide_oldest_window_cannot_be_starved_by_smaller_late_windows():
    control = controller(rows=5)
    sessions = list(range(7))
    work = [(0, 4), *[(sid, 1) for sid in sessions[1:]]]
    first = control.freeze(sessions)
    control.accept(first)
    assert control.select(work, first) == [0, 1]
    second = control.freeze(sessions)
    control.accept(second)
    assert control.select(work, second) == [2, 3, 4, 5, 6]
    third = control.freeze(sessions)
    control.accept(third)
    assert control.select(work, third) == [0, 1]


@pytest.mark.parametrize("change", [{"row_budget": True}, {"session_ceiling": 0}, {"prefill_tokens": -1},
                                    {"proposal_limit": 4}, {"row_budget": 2}, {"session_ceiling": 41}])
def test_invalid_requests_preserve_all_configured_effective_and_pending_values(change):
    control = controller()
    before = control.record()
    with pytest.raises(ValueError):
        control.request(**change)
    assert control.record() == before


def test_peers_apply_the_leaders_plan_and_reject_stale_or_oversized_plans():
    leader, peer = controller(), controller()
    leader.request(proposal_limit=0, row_budget=3)
    plan = leader.freeze([0, 1, 2, 3])
    for control in (leader, peer):
        control.accept(plan)
        assert control.select([(sid, 1) for sid in range(4)], plan) == [0, 1, 2]
    assert leader.effective == peer.effective
    with pytest.raises(ValueError, match="epoch"):
        peer.accept(plan)
    with pytest.raises(ValueError, match="maximum"):
        peer.accept({**plan, "epoch": 2, "row_budget": 33})
    assert peer.epoch == 1


def test_nonround_boundary_sync_preserves_newer_pending_requests_and_plan_epoch():
    leader, peer = controller(), controller()
    leader.request(row_budget=16, proposal_limit=0, prefill_tokens=512)
    leader.boundary()
    frozen = leader.freeze_boundary()
    leader.request(proposal_limit=1, row_budget=8)
    before = leader.record()
    for control in (leader, peer):
        control.apply_boundary(frozen)
        assert control.effective == frozen["settings"] and control.epoch == 0
    assert leader.record()["pending"] == before["pending"]
    assert leader.record()["configured"] == before["configured"]
    leader.boundary()
    next_boundary = leader.freeze_boundary()
    peer.apply_boundary(next_boundary)
    assert peer.effective["proposal_limit"] == 1 and peer.effective["row_budget"] == 8
    with pytest.raises(ValueError, match="epoch"):
        peer.apply_boundary({**next_boundary, "epoch": 1})


def test_empty_step_retains_the_ready_profile_and_reports_work_categories_separately():
    control = controller()
    plan = control.freeze([])
    control.accept(plan)
    assert control.select([], plan) == []
    control.observe(active=1, scheduled=1, target_rows=4, prefill_tokens=128,
                    committed_tokens=2, max_history_tokens=262140, elapsed_s=0.125)
    record = control.record()
    assert record["total_target_rows"] == 4
    assert record["total_prefill_tokens"] == 128
    assert record["total_committed_tokens"] == 2
    assert record["allocated_maximum"] == control.maximum
    assert record["last_step"]["max_history_tokens"] == 262140


@pytest.mark.parametrize("provisional", [0, 1])
def test_zero_proposal_steps_absorb_mtp_conditioning_without_sampling(monkeypatch, provisional):
    import torch
    from tensorfold.families.qwen4_exp.cuda import multi
    from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry

    decoder = object.__new__(multi.MultiDecoder)
    decoder.controller = controller(slots=1, rows=4)
    decoder.controller.request(proposal_limit=0)
    decoder.controller.boundary()
    decoder.depth, decoder.capacity = 3, 1000
    decoder.speculation = MethodGeometry("mtp", 3)
    decoder.w, decoder.mbuf = SimpleNamespace(dspark=None), object()
    decoder.buf = SimpleNamespace(streams=torch.tensor([[1.0], [2.0]]), rows=4)
    decoder.kv_host, decoder.graphs = None, None
    state = SimpleNamespace(mtp_len=5, mtp_drafted=provisional, pos=6)
    state.set_mtp_len = lambda length: setattr(state, "mtp_len", length)
    stream = SimpleNamespace(st=state, sid=0, out=[], count=100, drafts=[9])
    staged, settled = [], []

    def stage(buffer, windows):
        staged.extend(windows)
        return [(state, 0, 2)]

    class Words:
        def __init__(self, segments, host, absorbed):
            self.absorbed = absorbed

        def settle(self):
            self.absorbed()
            settled.append(True)

    monkeypatch.setattr(multi, "mtp_stage", stage)
    monkeypatch.setattr(multi, "KVWords", Words)
    def mtp(segments, *, logits=True):
        assert not logits  # Maintenance does not calculate an unused vocabulary projection.
        return None

    decoder._mtp = mtp
    decoder._picks = lambda *args: pytest.fail("a zero-proposal step consumed a sample")
    decoder._draft_all([(stream, 0, [7, 8])])
    assert len(staged) == 1 and staged[0][1] == [7, 8]
    assert state.mtp_len == 7 - provisional and state.mtp_drafted == 0
    assert stream.drafts == [] and settled == [True]


@pytest.fixture
def decoder_round(monkeypatch):
    import torch
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda import multi

    decoder = object.__new__(multi.MultiDecoder)
    decoder.controller = controller(slots=5, rows=6, proposals=2)
    decoder.controller.work_observations.configure_indexer(ratio=4, budget=2048, heads=4, target_caches=12)
    decoder._execution_plan, decoder._memory_error = None, None
    decoder.w = SimpleNamespace(comm=None, meta={})
    decoder.buf = SimpleNamespace(h=torch.zeros((6, 1), dtype=torch.float16))
    decoder.pbuf, decoder.kv_host = SimpleNamespace(), None
    decoder.mbuf = None
    decoder.pdl, decoder.converged = False, False
    decoder.filling, decoder.held, decoder.streams = [], {}, {}
    decoder.kept, decoder.fills = [], {}
    decoder.arrived = lambda: False
    decoder.graphs = None
    decoder._rotor_lookup = lambda *args: False
    decoder._make_room = lambda: []
    decoder._timed = lambda *args: None
    decoder._ends = lambda stream: ()
    decoder._draft_all = lambda *args, **kwargs: None
    for sid in range(5):
        stream = Stream([1], 100)
        stream.sid, stream.st = sid, SimpleNamespace(pos=1, discard_target_tentative=lambda: None)
        stream.out, stream.context, stream.drafts = [10], [1, 10], [20, 30]
        decoder.streams[sid] = stream
    executed = []

    def stage(weights, buffer, windows, **kwargs):
        segments, at = [], 0
        executed.append([state for state, _ in windows])
        for state, tokens in windows:
            segments.append((state, at, at + len(tokens)))
            at += len(tokens)
        return segments

    monkeypatch.setattr(multi, "stage", stage)
    monkeypatch.setattr(multi, "compute", lambda weights, segs, buffer, **kwargs:
                        torch.zeros((segs[-1][2], 64)))
    monkeypatch.setattr(multi, "commit", lambda weights, state, buffer, rows, kept, **kwargs:
                        setattr(state, "pos", state.pos + kept))
    monkeypatch.setattr(multi, "KVWords", lambda *args: SimpleNamespace(settle=lambda: None))
    monkeypatch.setattr(multi, "launch_streams", lambda logits, starts, positions, sampling:
                        [[50] * len(group) for group in positions])
    monkeypatch.setattr(multi, "collect_streams", lambda launched, words: launched)
    return decoder, executed


def test_decoder_round_executes_only_the_bounded_fair_microbatch(decoder_round):
    decoder, executed = decoder_round
    for expected in ([0, 1], [2, 3], [4, 0]):
        assert decoder.round() == []
        assert executed[-1] == [decoder.streams[sid].st for sid in expected]
        assert decoder.controller.record()["last_step"]["target_rows"] == 6
    assert [len(stream.out) for stream in decoder.streams.values()] == [3, 2, 2, 2, 2]


def test_useful_output_metrics_exclude_owed_replay(decoder_round):
    decoder, _ = decoder_round
    for stream in decoder.streams.values():
        stream.owed = [50]
    decoder.round()
    assert decoder.controller.record()["total_committed_tokens"] == 0
    assert decoder.streams[0].out == [10, 50]


def test_useful_output_metrics_exclude_failed_grammar_publication(decoder_round):
    from tensorfold.engine.grammar import GrammarError

    decoder, _ = decoder_round

    def failed_advance(tokens):
        raise GrammarError("independent test grammar refuses this publication")

    decoder.streams[0].constraint = SimpleNamespace(
        window=lambda tokens, parents: SimpleNamespace(tokens=tokens),
        mask=lambda logits, *args: logits, advance=failed_advance)
    decoder.round()
    assert decoder.streams[0].out == [10]
    assert decoder.controller.record()["total_committed_tokens"] == 1


@pytest.mark.parametrize("count", [0, 1, 3])
def test_logical_stop_precedes_cache_commit_even_inside_an_all_accepted_block(decoder_round, monkeypatch, count):
    from tensorfold.families.qwen4_exp.cuda import multi

    decoder, _ = decoder_round
    monkeypatch.setattr(multi, "launch_streams", lambda *args: [[20, 30, 40], [20, 30, 40]])
    for stream in decoder.streams.values():
        stream.token_limit = lambda new, prefix: (count, True)
    ended = decoder.round()
    assert ended == [decoder.streams[0], decoder.streams[1]]
    for stream in ended:
        assert stream.st.pos == 1 + count
        assert stream.committed == [10, 20, 30][:count]
        assert stream.out == [10, *[20, 30, 40][:count]]
    record = decoder.controller.record()
    assert record["total_target_rows"] == 6
    assert record["total_committed_tokens"] == 2 * count


@pytest.mark.parametrize("first_output", [False, True])
def test_standalone_prefill_metrics_include_first_outputs_once(decoder_round, monkeypatch, first_output):
    import torch
    decoder, _ = decoder_round
    stream = decoder.streams[0]
    stream.out, stream.owed = [], []
    decoder.streams, decoder.filling = {}, [stream]
    decoder._pieces = lambda: [(stream, 0, 2)]
    decoder._note_passed = decoder._read_ahead = lambda *args: None
    decoder._end_rows, decoder._cuts = lambda *args: [1], lambda *args: []

    def absorb(*args):
        stream.st.pos += 2
        return [torch.zeros((1, 1))]

    def joined(*args):
        if first_output:
            stream.take([50])
        return []

    decoder._absorb, decoder._joined = absorb, joined
    assert decoder._pass() == []
    record = decoder.controller.record()
    assert record["total_prefill_tokens"] == 2
    assert record["total_committed_tokens"] == int(first_output)
    assert record["last_step"]["elapsed_s"] > 0


def test_graph_padding_that_exceeds_the_execution_arena_declines_before_staging():
    from tensorfold.families.qwen4_exp.cuda.bound_graphs import BoundStateGraphs

    graph = object.__new__(BoundStateGraphs)
    graph.closed = False
    graph.dec = SimpleNamespace(depth=3, buf=SimpleNamespace(rotor_lookup=False, rows=4))
    states = [SimpleNamespace(image_positions=None), SimpleNamespace(image_positions=None)]
    # Two actual rows fit the arena, while full-width graph padding needs eight.
    assert graph._forward("verify", [(states[0], 0, 1), (states[1], 1, 2)]) is None


def profile_packet(control):
    """Synthetic bound proofs test selection; they are not hardware evidence."""
    from tensorfold.families.qwen4_exp.cuda.adaptive_controller import IDENTITY_DIGESTS

    identity = {name: "a" * 64 for name in IDENTITY_DIGESTS}
    identity.update(context_tokens=262144, resident_sessions=control.slots, history_layout="unique", method="mtp")
    work = {"active_requests": 1, "decode_requests": 1, "prefill_requests": 0,
            "max_history_tokens": 8192, "history_tokens": 8192, "retained_sessions": control.slots,
            "pending_prompt_tokens": 0, "pending_arrivals": False, "indexer_work": None, "expert_group_max": None}
    region = {**work, "indexer_work": {"coverage": "all-supported"},
              "expert_group_max": {"coverage": "all-supported"}}
    execution = {"row_budget": 4, "proposal_limit": 1, "prefill_tokens": 512}
    evidence, refs = {}, {}
    for index, role in enumerate(("correctness", "resources", "performance"), 1):
        digest = str(index) * 64
        refs[role] = digest
        evidence[digest] = {"schema": "tensorfold.execution-profile-evidence", "version": 1,
                            "kind": role, "identity": identity, "qualification": "live", "verdict": "passed",
                            "workload": region, "execution": execution,
                            "proof": {"protocol": "tensorfold.execution-profile-" + role + "-v1",
                                      "record_sha256": "f" * 64,
                                      "artifacts": [{"path": role + ".json", "size": 1, "sha256": "f" * 64}]}}
    table = {"schema": "tensorfold.execution-profiles", "version": 1, "identity": identity,
             "profiles": [{"id": "bound-synthetic-control", "workload": region,
                           "execution": execution, "evidence": refs}]}
    return identity, evidence, table, work


def test_only_exact_bound_workload_selects_a_registered_profile_and_fallback_stays_static():
    control = controller()
    identity, evidence, table, work = profile_packet(control)
    with pytest.raises(ValueError, match="unbound"):
        control.install_profiles(table)
    control.bind_identity(identity, evidence=evidence)
    control.install_profiles(table)
    plan = control.freeze([0], workload=work)
    control.accept(plan)
    assert plan["profile_id"] == "bound-synthetic-control" and plan["proposal_limit"] == 1
    assert control.record()["selection"] == "qualified-workload-region"
    other = control.freeze([0], workload={**work, "max_history_tokens": 8193})
    control.accept(other)
    assert other["profile_id"] is None
    assert other["proposal_limit"] == 3 and other["row_budget"] == 32


@pytest.mark.parametrize("changed", ["identity", "execution", "workload", "proof", "missing_receipt"])
def test_table_cannot_self_certify_a_new_scope_or_unbound_evidence(changed):
    from copy import deepcopy

    control = controller()
    identity, evidence, table, _ = profile_packet(control)
    if changed == "proof":
        evidence[next(iter(evidence))]["proof"] = {}
        with pytest.raises(ValueError, match="proof"):
            control.bind_identity(identity, evidence=evidence)
        return
    control.bind_identity(identity, evidence=evidence)
    table = deepcopy(table)
    profile = table["profiles"][0]
    if changed == "identity":
        table["identity"]["target"] = "b" * 64
    elif changed == "execution":
        profile["execution"]["proposal_limit"] = 2
    elif changed == "workload":
        profile["workload"]["indexer_work"] = 17
    else:
        profile["evidence"]["correctness"] = "9" * 64
    with pytest.raises(ValueError):
        control.install_profiles(table)
    assert control.record()["registered_profiles"] == 0


def test_profile_selection_respects_requested_caps_and_unknown_work_axes():
    control = controller()
    identity, evidence, table, work = profile_packet(control)
    control.bind_identity(identity, evidence=evidence)
    control.install_profiles(table)
    control.request(prefill_tokens=128)
    assert control.freeze([0], workload=work)["profile_id"] is None
    # A measurement requiring actual selected-indexer work must not match an
    # observation that did not obtain that quantity from the execution path.
    table["profiles"][0]["workload"]["indexer_work"] = 17
    for receipt in evidence.values():
        receipt["workload"] = table["profiles"][0]["workload"]
    control.bind_identity(identity, evidence=evidence)
    control.install_profiles(table)
    control.request(prefill_tokens=2048)
    assert control.freeze([0], workload=work)["profile_id"] is None


def test_new_placement_requires_new_identity_and_cannot_reuse_old_proofs():
    control = controller()
    identity, evidence, table, _ = profile_packet(control)
    control.enable_migration(1)
    control.bind_identity(identity, evidence=evidence)
    control.install_profiles(table)
    control.request_migration(4)
    assert control.record()["profile_identity"] is None
    assert control.record()["registered_profiles"] == 0
    new_identity = {**identity, "placement": "b" * 64}
    with pytest.raises(ValueError, match="exact live qualification"):
        control.bind_identity(new_identity, evidence=evidence)
    control.bind_identity(new_identity)
    with pytest.raises(ValueError, match="differs"):
        control.install_profiles(table)


def test_manifest_replacement_is_atomic_and_rejects_a_newer_pending_migration():
    from copy import deepcopy

    control = controller()
    identity, evidence, table, _ = profile_packet(control)
    control.enable_migration(1)
    control.bind_identity(identity, evidence=evidence)
    control.install_profiles(table)
    before = control.record()
    bad = deepcopy(table)
    bad["profiles"][0]["evidence"]["resources"] = "e" * 64
    with pytest.raises(ValueError):
        control.replace_verified_profiles(identity, evidence=evidence, table=bad, expected_migration_epoch=0)
    assert control.record() == before
    control.request_migration(4)
    with pytest.raises(ValueError, match="migration is pending"):
        control.replace_verified_profiles(identity, evidence=evidence, table=table, expected_migration_epoch=0)
    assert control.record()["migration"]["pending"]


def test_changed_runtime_facts_invalidate_proofs_without_changing_allocated_or_session_promises():
    control = controller()
    identity, evidence, table, work = profile_packet(control)
    control.bind_identity(identity, evidence=evidence)
    control.install_profiles(table)
    selected = control.freeze([0], workload=work)
    control.accept(selected)
    control.request(session_ceiling=2)
    before = control.record()
    record = control.invalidate_binding("observed runtime/library identity changed: " + "x" * 1000)
    assert record["profile_identity"] is None and record["registered_profiles"] == 0
    assert record["selected_profile"] is None and record["profile_table_sha256"] is None
    assert not control.evidence
    assert len(record["binding_invalidated_reason"]) == 512
    for field in ("configured", "pending", "effective", "allocated_maximum", "migration", "epoch"):
        assert record[field] == before[field]
    with pytest.raises(ValueError, match="unbound"):
        control.install_profiles(table)
    control.bind_identity(identity, evidence=evidence)
    assert control.record()["binding_invalidated_reason"] is None


def test_eight_rank_profile_commit_defers_http_migration_until_every_rank_acknowledges():
    from concurrent.futures import ThreadPoolExecutor
    import threading

    controls = [controller(slots=1) for _ in range(8)]
    packets = []
    for control in controls:
        control.enable_migration(1)
        packets.append(profile_packet(control))
    voted = [threading.Event() for _ in controls]
    final_vote, committed = threading.Barrier(8), threading.Barrier(9)
    commit_allowed, release = threading.Event(), threading.Event()

    def rank_commit(rank):
        control = controls[rank]
        identity, evidence, table, _ = packets[rank]
        with control.profile_transaction():
            final_vote.wait(timeout=5)
            voted[rank].set()
            assert commit_allowed.wait(5)
            control.replace_verified_profiles(identity, evidence=evidence, table=table, expected_migration_epoch=0)
            committed.wait(timeout=5)
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=9) as workers:
        futures = [workers.submit(rank_commit, rank) for rank in range(8)]
        try:
            assert all(event.wait(5) for event in voted)
            # A real HTTP thread races after the final vote but before commit.
            # It must return while all worker transactions remain reserved.
            receipt = workers.submit(controls[0].request_migration, 4).result(timeout=1)
            assert receipt["deferred_migration"] == 4
            assert receipt["migration"]["request_epoch"] == 0
            commit_allowed.set()
            committed.wait(timeout=5)
            assert all(control.record()["registered_profiles"] == 1 for control in controls)
            assert all(control.record()["migration"]["request_epoch"] == 0 for control in controls)
        finally:
            commit_allowed.set()
            release.set()
        for future in futures:
            future.result(timeout=5)
    assert controls[0].record()["migration"]["request_epoch"] == 1
    assert controls[0].record()["registered_profiles"] == 0
    assert all(control.record()["migration"]["request_epoch"] == 0 for control in controls[1:])
    # The next coordinator plan publishes that one request to the whole cohort.
    plan = controls[0].freeze([])
    for control in controls:
        control.accept(plan)
        assert control.record()["migration"]["requested_degree"] == 4
        assert control.record()["registered_profiles"] == 0


def test_profile_transaction_coalesces_and_cancels_deferred_requests_without_touching_effective_values():
    control = controller()
    control.enable_migration(1)
    before = control.record()
    with control.profile_transaction():
        control.request_migration(4)
        control.request_migration(2)
        receipt = control.request_migration(1)
        assert receipt["deferred_migration"] == 1
        assert receipt["migration"] == before["migration"]
        for field in ("effective", "epoch", "applied_migration", "allocated_maximum"):
            assert receipt[field] == before[field]
    after = control.record()
    assert after["migration"] == before["migration"]
    assert after["deferred_migration"] is None and not after["profile_transaction_active"]


def test_refused_profile_transaction_preserves_primary_and_publishes_deferred_migration_after_abort():
    control = controller()
    control.enable_migration(1)
    primary = ValueError("the cohort rejected this profile packet")
    with pytest.raises(ValueError) as caught:
        with control.profile_transaction():
            control.request_migration(4)
            raise primary
    assert caught.value is primary
    record = control.record()
    assert not record["profile_transaction_active"] and record["deferred_migration"] is None
    assert record["migration"]["requested_degree"] == 4 and record["migration"]["pending"]


def test_measured_covering_region_serves_changing_histories_without_extrapolation():
    control = controller()
    identity, evidence, table, work = profile_packet(control)
    region = table["profiles"][0]["workload"]
    region["max_history_tokens"] = {"min": 8192, "max": 16384}
    region["history_tokens"] = {"min": 8192, "max": 16384}
    control.bind_identity(identity, evidence=evidence)
    control.install_profiles(table)
    for position in (8192, 8193, 10240, 16384):
        plan = control.freeze([0], workload={**work, "max_history_tokens": position, "history_tokens": position})
        control.accept(plan)
        assert plan["profile_id"] == "bound-synthetic-control"
    assert control.freeze([0], workload={**work, "max_history_tokens": 16385,
                                        "history_tokens": 16385})["profile_id"] is None


@pytest.mark.parametrize("bad_bound", [{"min": 12, "max": 11}, {"min": True, "max": 14},
                                       {"min": 0}, {"min": -1, "max": 1}, None])
def test_malformed_measured_region_is_rejected_without_weakening_existing_table(bad_bound):
    control = controller()
    identity, evidence, table, _ = profile_packet(control)
    table["profiles"][0]["workload"]["max_history_tokens"] = bad_bound
    control.bind_identity(identity, evidence=evidence)
    with pytest.raises(ValueError, match="workload axis"):
        control.install_profiles(table)


def test_overlapping_regions_require_a_measured_order_instead_of_table_order():
    from copy import deepcopy

    control = controller()
    identity, evidence, table, _ = profile_packet(control)
    extra = deepcopy(table["profiles"][0])
    extra["id"] = "unjustified-overlapping-winner"
    extra["workload"]["max_history_tokens"] = {"min": 8192, "max": 16384}
    table["profiles"].append(extra)
    control.bind_identity(identity, evidence=evidence)
    with pytest.raises(ValueError, match="overlap"):
        control.install_profiles(table)


def test_migration_runs_one_bundle_per_boundary_and_retains_partial_progress():
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    decoder = object.__new__(MultiDecoder)
    decoder.controller = controller(slots=2)
    decoder.controller.enable_migration(1)
    decoder.controller.request_migration(4)
    decoder._migration_cursor, decoder.streams = 0, {}
    moved = []
    states = []
    for slot in range(2):
        state = SimpleNamespace(history_placement=SimpleNamespace(degree=1), mtp_drafted=2, mtp_len=10)
        state.set_mtp_len = lambda length, state=state: setattr(state, "mtp_len", length)

        def begin(degree, state=state):
            assert state.mtp_drafted == 0 and state.mtp_len == 8
            state._history_migration = {"degree": degree, "remaining": 2}

        def step(state=state, slot=slot):
            pending = state._history_migration
            if pending["remaining"]:
                pending["remaining"] -= 1
                moved.append(slot)
                return True
            state.history_placement.degree = pending["degree"]
            state._history_migration = None
            return False

        state.begin_history_migration, state.migrate_history_bundle = begin, step
        states.append(state)
    decoder.states = tuple(states)
    for _ in range(7):
        before = len(moved)
        plan = decoder.controller.freeze([])
        decoder.controller.accept(plan)
        decoder._migrate_boundary(plan["migration"])
        assert len(moved) - before <= 1
    assert moved == [0, 1, 0, 1]
    assert all(state.history_placement.degree == 4 for state in states)
    assert not decoder.maintenance_pending()


def test_blocked_migration_does_not_request_an_idle_busy_loop():
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    decoder = object.__new__(MultiDecoder)
    decoder.controller = controller(slots=1)
    decoder.controller.enable_migration(1)
    decoder.controller.request_migration(2)
    decoder._migration_cursor, decoder.streams = 0, {}
    state = SimpleNamespace(history_placement=SimpleNamespace(degree=1), mtp_drafted=0)
    state.begin_history_migration = lambda degree: setattr(state, "_history_migration", {"target": degree})
    state.migrate_history_bundle = lambda: False
    decoder.states = (state,)
    plan = decoder.controller.freeze([])
    decoder.controller.accept(plan)
    decoder._migrate_boundary(plan["migration"])
    assert not decoder.maintenance_pending()
    assert decoder.controller.record()["migration"]["blocked"]
    assert not decoder.controller.freeze([])["migration"]["pending"]
    previous = decoder.controller.record()["migration"]["request_epoch"]
    decoder.controller.request_migration(2)
    retry = decoder.controller.freeze([])["migration"]
    assert retry["pending"] and retry["request_epoch"] == previous + 1


@pytest.mark.parametrize("arrival", ["after_freeze", "after_accept"])
def test_new_migration_request_cannot_change_a_frozen_cohort_bundle_or_be_cleared(arrival):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    decoders, calls = [], []
    for rank in range(2):
        decoder = object.__new__(MultiDecoder)
        decoder.controller = controller(slots=1)
        decoder.controller.enable_migration(1)
        decoder._migration_cursor, decoder.streams = 0, {}
        state = SimpleNamespace(history_placement=SimpleNamespace(degree=1), mtp_drafted=0)

        def begin(degree, state=state, rank=rank):
            calls.append((rank, degree))
            state._history_migration = {"target": degree}

        def step(state=state):
            state.history_placement.degree = state._history_migration["target"]
            state._history_migration = None
            return False

        state.begin_history_migration, state.migrate_history_bundle = begin, step
        decoder.states = (state,)
        decoders.append(decoder)
    leader, peer = decoders
    leader.controller.request_migration(2)
    old = leader.controller.freeze([])
    if arrival == "after_freeze":
        leader.controller.request_migration(4)
    for decoder in decoders:
        decoder.controller.accept(old)
    if arrival == "after_accept":
        leader.controller.request_migration(4)
    for decoder in decoders:
        decoder._migrate_boundary(old["migration"])
        # Resolving the old generation must not clear the leader's new request.
        decoder.controller.migration_status(request_epoch=old["migration"]["request_epoch"], pending=False)
    assert calls == [(0, 2), (1, 2)]
    latest = leader.controller.record()["migration"]
    assert latest["pending"] and latest["requested_degree"] == 4 and latest["request_epoch"] == 2
    new = leader.controller.freeze([])
    for decoder in decoders:
        decoder.controller.accept(new)
        decoder._migrate_boundary(new["migration"])
    assert calls == [(0, 2), (1, 2), (0, 4), (1, 4)]


@pytest.mark.parametrize("failed", [False, True])
def test_complete_round_timing_includes_idle_migration_and_failed_work(decoder_round, monkeypatch, failed):
    from tensorfold.families.qwen4_exp.cuda import multi

    decoder, _ = decoder_round
    decoder.streams = {}
    decoder.controller.enable_migration(1)
    decoder.controller.request_migration(2)
    now = [10.0]
    monkeypatch.setattr(multi.time, "perf_counter", lambda: now[0])

    def migrate(migration):
        now[0] += 3.0
        if failed:
            raise RuntimeError("a migration failed after doing bounded work")

    decoder._migration_bundle = migrate
    if failed:
        with pytest.raises(RuntimeError):
            decoder.round()
    else:
        assert decoder.round() == []
    record = decoder.controller.record()["last_round"]
    assert record["elapsed_s"] == record["migration_s"] == 3.0
    assert record["outcome"] == ("RuntimeError" if failed else "ok")
    assert record["prefill_tokens"] == record["committed_tokens"] == 0


def test_telemetry_failure_preserves_an_original_inference_failure(decoder_round):
    decoder, _ = decoder_round
    primary, secondary = RuntimeError("native inference failed"), OSError("observation publication failed")

    def fail_round():
        raise primary

    def fail_observation(**kwargs):
        raise secondary

    decoder._round = fail_round
    decoder.controller.observe_round = fail_observation
    with pytest.raises(RuntimeError) as caught:
        decoder.round()
    assert caught.value is primary and caught.value.__cause__ is secondary


@pytest.mark.parametrize("enabled", [False, True])
def test_request_lifecycle_keeps_enabled_zero_proposal_mtp_warm_only_until_retirement(
        decoder_round, monkeypatch, enabled):
    """Run serving orchestration; token-recording operators replace GPU math only."""
    import torch
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda import multi
    from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry

    decoder, _ = decoder_round
    decoder.controller = controller(slots=1, rows=3, proposals=2)
    decoder.controller.work_observations.configure_indexer(ratio=4, budget=2048, heads=4, target_caches=12)
    decoder.full_resident = True
    decoder.depth, decoder.capacity, decoder.prefill_rows = 2, 64, 2
    decoder.share, decoder.round_s, decoder.row_s = 0.0, None, None
    decoder.confidence, decoder.copy, decoder.next_id = 0.0, False, 0
    decoder.controller.request(proposal_limit=0)
    decoder.controller.boundary()
    decoder.points = decoder.vision = decoder._draft_map = None
    decoder.streams, decoder.filling, decoder.fills, decoder.passed = {}, [], {}, {}
    decoder.keep, decoder.used, decoder.clock = 0, {}, 0
    decoder.buf.rows, decoder.buf.streams = 6, torch.zeros((6, 1))
    decoder.pbuf.rows, decoder.pbuf.streams = 2, torch.zeros((2, 1))
    decoder.mbuf = SimpleNamespace(h=torch.zeros((6, 1), dtype=torch.float16), rows=6)
    decoder.speculation = MethodGeometry("mtp", 2)
    decoder.w.meta["execution_provider"] = "sm70"
    decoder._grow = lambda *args, **kwargs: True
    decoder._read_ahead = lambda pieces: None
    decoder._picks = lambda *args: pytest.fail("a draft=False request sampled a proposal")
    del decoder._draft_all  # Exercise canonical maintenance, not the other fixture's operator stub.

    class State:
        history_slot = slot_generation = 0
        from tensorfold.families.qwen4_exp.kv_formats import get_pair
        kv_pair = get_pair("fp16")
        image_positions = None
        reset_calls = 0

        def begin_sequence_generation(self):
            self.slot_generation += 1

        def reset(self, weights, *, new_sequence=True):
            if new_sequence:
                self.begin_sequence_generation()
            self.reset_calls += 1
            self.pos = self.mtp_len = self.mtp_drafted = 0
            self.mtp_history = []

        def snapshot(self):
            return {"pos": self.pos, "mtp_len": self.mtp_len}

        def kv_validate_snapshot(self, snapshot):
            assert 0 <= snapshot["mtp_len"] <= snapshot["pos"]

        def set_mtp_len(self, length):
            assert length == len(self.mtp_history), "MTP publication skipped or repeated conditioning inputs"
            self.mtp_len = length

    state = State()
    state.pos = state.mtp_len = state.mtp_drafted = 0
    state.mtp_history = []
    decoder.free, decoder.states, decoder.kv_pair = [state], (state,), state.kv_pair
    decoder._shrink = lambda st: st.reset(decoder.w)
    staged = {}

    def append_conditioning(st, tokens):
        assert len(st.mtp_history) == st.mtp_len
        st.mtp_history.extend(tokens)

    def mtp_stage(buffer, windows):
        staged[id(buffer)] = [(st, list(tokens)) for st, tokens, _features in windows]
        segments, at = [], 0
        for st, tokens, _features in windows:
            segments.append((st, at, at + len(tokens)))
            at += len(tokens)
        return segments

    def mtp_compute(weights, segments, buffer, *, logits=True, **kwargs):
        assert not logits
        for st, tokens in staged[id(buffer)]:
            append_conditioning(st, tokens)
        return None

    def initial_mtp(engine, tokens, features):
        append_conditioning(engine.st, tokens)
        return torch.zeros((1, 64))

    class Words:
        def __init__(self, segments, host, absorbed=None):
            self.absorbed = absorbed

        def settle(self):
            if self.absorbed is not None:
                callback, self.absorbed = self.absorbed, None
                callback()

    monkeypatch.setattr(multi, "mtp_stage", mtp_stage)
    monkeypatch.setattr(multi, "mtp_compute", mtp_compute)
    monkeypatch.setattr(multi.Engine, "mtp_forward", initial_mtp)
    monkeypatch.setattr(multi.Engine, "sample", lambda *args, **kwargs: [40])
    monkeypatch.setattr(multi, "KVWords", Words)
    monkeypatch.setattr(multi, "collect_streams", lambda launched, words: (words.settle(), launched)[1])
    monkeypatch.setattr(multi, "_kv_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(multi.image_rows, "begin", lambda *args: None)
    monkeypatch.setattr(multi.image_rows, "finish", lambda *args: None)

    stream = Stream([11, 12, 13, 14, 15], 3, draft=enabled)
    decoder.admit(stream)
    assert decoder.fills[stream.sid][1] is enabled
    for _ in range(8):
        done = decoder.round()
        if done:
            break
    else:
        pytest.fail("the bounded prefill/decode lifecycle did not finish")
    assert stream.error is None
    assert stream.out == [40, 50, 50] and stream.drafts == []
    assert state.pos == len(stream.prompt) + len(stream.out) - 1
    if enabled:
        # Zero proposals keep conditioning fresh while this request can run
        # another round; completion needs no idle endpoint maintenance.
        assert state.mtp_history == stream.prompt[1:] + stream.out[:-1]
        assert state.mtp_len == state.pos - 1 and state.mtp_drafted == 0
    else:
        assert state.mtp_history == [] and state.mtp_len == 0
    observation = decoder.controller.record()["work_observations"]
    totals = observation["totals"]
    assert totals["logical_prompt_tokens"] == len(stream.prompt)
    assert totals["target_prefill_rows_submitted"] == len(stream.prompt)
    assert totals["target_decode_rows_submitted"] == 2
    assert totals["mtp_prefill_rows_submitted"] == (4 if enabled else 0)
    assert totals["mtp_refresh_rows_submitted"] == (2 if enabled else 0)
    assert totals["mtp_zero_proposal_refresh_rows"] == (2 if enabled else 0)
    assert totals["mtp_proposal_rows_submitted"] == 0
    assert observation["counts_complete"]
    decoder.finish(done)
    assert not hasattr(decoder, "retained_states")
    assert state.reset_calls == 2 and state.pos == state.mtp_len == 0


def test_round_publishes_actual_logical_work_and_verifier_histogram(decoder_round):
    decoder, executed = decoder_round
    decoder.round()
    reading = decoder.controller.record()["work_observations"]
    totals = reading["totals"]
    assert len(executed[0]) == 2
    assert totals["target_decode_rows_submitted"] == 6
    assert totals["indexer_query_rows_submitted"] == 6 * 12
    assert totals["verification_windows"] == 2
    assert totals["verified_candidates"] == 4
    assert totals["accepted_candidates_before_text_limits"] == 0
    assert reading["windows_by_offered_candidates"] == [0, 0, 2]
    assert reading["accepted_prefix_histogram_before_text_limits"] == [2, 0, 0]
    assert reading["expert_group_max"] is None and reading["counts_complete"]


def test_verifier_histogram_precedes_text_stop_and_useful_output_is_not_fabricated(decoder_round):
    decoder, _ = decoder_round
    stream = next(iter(decoder.streams.values()))
    decoder.streams = {stream.sid: stream}
    stream.drafts = [50, 50]
    stream.token_limit = lambda _new, _prefix: (0, True)
    decoder.round()
    reading = decoder.controller.record()
    work = reading["work_observations"]
    assert work["totals"]["verified_candidates"] == 2
    assert work["totals"]["accepted_candidates_before_text_limits"] == 2
    assert work["accepted_prefix_histogram_before_text_limits"] == [0, 0, 1]
    assert stream.out == [10] and reading["total_committed_tokens"] == 0


def test_partial_forward_remains_unknown_and_preserves_original_failure(decoder_round, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import multi

    decoder, _ = decoder_round
    primary = KeyboardInterrupt("controlled partial submission")
    def fail(*_args, **_kwargs):
        raise primary
    monkeypatch.setattr(multi, "compute", fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        decoder.round()
    assert caught.value is primary
    work = decoder.controller.record()["work_observations"]
    assert work["totals"]["incomplete_forward_submissions"] == 1
    assert work["totals"]["target_decode_rows_submitted"] == 0
    assert not work["counts_complete"]


def test_partial_prefill_interrupt_is_visible_without_counting_unsubmitted_rows(decoder_round, monkeypatch):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda import multi

    decoder, _ = decoder_round
    stream = Stream([11, 12], 3)
    stream.sid, stream.st = 9, SimpleNamespace(pos=0)
    decoder.streams, decoder.filling = {}, [stream]
    decoder._pass_rows = lambda: 2
    decoder._pieces = lambda rows=None: [(stream, 0, 2)]
    decoder._note_passed = lambda pieces: None
    decoder._read_ahead = lambda pieces: None
    decoder._end_rows = lambda pieces, segments: [1]
    decoder._cuts = lambda pieces, segments: []
    primary = KeyboardInterrupt("partial prompt forward")
    def fail(*args, **kwargs):
        raise primary
    monkeypatch.setattr(multi, "compute", fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        decoder._pass()
    assert caught.value is primary
    reading = decoder.controller.record()["work_observations"]
    assert reading["totals"]["incomplete_forward_submissions"] == 1
    assert reading["totals"]["target_prefill_rows_submitted"] == 0
    assert not reading["counts_complete"]


def test_synthetic_warmup_is_cleared_from_every_serving_measurement_plane():
    control = controller(slots=5, rows=6, proposals=2)
    control.work_observations.configure_indexer(ratio=4, budget=2048, heads=4, target_caches=12)
    control.observe(active=1, scheduled=1, target_rows=3, prefill_tokens=5, committed_tokens=2,
                    max_history_tokens=5, elapsed_s=0.1)
    control.observe_round(before=(0, 0, 0), elapsed_s=0.2, migration_s=0.0, outcome="ok")
    control.work_observations.admission(SimpleNamespace(prompt=[0] * 5, carry=None, cached=0))
    settings, epoch = control.configured.copy(), control.epoch
    control.reset_observations()
    reading = control.record()
    assert reading["total_committed_tokens"] == reading["total_prefill_tokens"] == reading["total_target_rows"] == 0
    assert reading["last_step"] == reading["last_round"] == {}
    assert not any(reading["work_observations"]["totals"].values())
    assert control.configured == settings and control.epoch == epoch
    assert reading["work_observations"]["indexer_geometry"]["target_caches"] == 12
