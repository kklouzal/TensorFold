"""Synthetic preparation/proof controls; no native hardware qualification."""

import pytest
from types import SimpleNamespace

from tensorfold.families.qwen4_exp.cuda.owner_execution import GRAPH_FAMILY, KERNEL_FAMILY, REFERENCE, UNCAPTURED
from test_flashnext_adaptive_controller import controller, profile_packet
from test_flashnext_tp_scheduler import cohort  # noqa: F401 - reuse the real eight-thread control cohort fixture


def native_packet(control, *, graph=UNCAPTURED):
    identity, evidence, table, work = profile_packet(control)
    table["profiles"][0]["execution"].update(kernel_family=KERNEL_FAMILY, graph_family=graph)
    return identity, evidence, table, work


def test_preparing_native_owners_does_not_select_or_qualify_them():
    control = controller()
    control.configure_execution_families(native=True, graph_rows=32)
    plan = control.freeze([0])
    assert {name: plan[name] for name in REFERENCE} == REFERENCE
    control.accept(plan)
    assert control.record()["performance_qualification"] == "pending"
    assert control.record()["execution_families"]["native_prepared"]


@pytest.mark.parametrize("prepared,graph_rows,graph,error", [
    (False, 0, UNCAPTURED, "not prepared"),
    (True, 0, GRAPH_FAMILY, "does not cover"),
    (True, 2, GRAPH_FAMILY, "does not cover"),
])
def test_verified_profile_cannot_select_an_unprepared_or_undersized_family(prepared, graph_rows, graph, error):
    control = controller()
    control.configure_execution_families(native=prepared, graph_rows=graph_rows)
    identity, evidence, table, _ = native_packet(control, graph=graph)
    control.bind_identity(identity, evidence=evidence)
    with pytest.raises(ValueError, match=error):
        control.install_profiles(table)
    assert control.record()["registered_profiles"] == 0


@pytest.mark.parametrize("graph", [UNCAPTURED, GRAPH_FAMILY])
def test_only_exact_verified_family_scope_can_select_prepared_native_execution(graph):
    control = controller()
    control.configure_execution_families(native=True, graph_rows=32)
    identity, evidence, table, work = native_packet(control, graph=graph)
    control.bind_identity(identity, evidence=evidence)
    control.install_profiles(table)
    plan = control.freeze([0], workload=work)
    assert plan["kernel_family"] == KERNEL_FAMILY and plan["graph_family"] == graph
    control.accept(plan)
    assert control.record()["performance_qualification"] == "registered-workload-region"
    other = control.freeze([0], workload={**work, "history_tokens": 8193})
    assert {name: other[name] for name in REFERENCE} == REFERENCE


def test_family_cannot_be_added_to_an_old_proof_by_a_table_request():
    control = controller()
    control.configure_execution_families(native=True, graph_rows=32)
    identity, evidence, table, _ = profile_packet(control)
    control.bind_identity(identity, evidence=evidence)
    table["profiles"][0]["execution"] = {**table["profiles"][0]["execution"],
                                            "kernel_family": KERNEL_FAMILY, "graph_family": GRAPH_FAMILY}
    with pytest.raises(ValueError, match="exact-scope"):
        control.install_profiles(table)


def test_native_plan_without_a_qualified_profile_requires_explicit_startup_trial():
    control = controller()
    control.configure_execution_families(native=True, graph_rows=32)
    plan = {**control.freeze([0]), "kernel_family": KERNEL_FAMILY, "graph_family": GRAPH_FAMILY}
    with pytest.raises(ValueError, match="qualified profile or explicit startup trial"):
        control.accept(plan)
    trial = {"kernel_family": KERNEL_FAMILY, "graph_family": GRAPH_FAMILY}
    control.configure_execution_families(native=True, graph_rows=32, trial=trial)
    plan = control.freeze([0])
    control.accept(plan)
    record = control.record()
    assert record["selection"] == "qualification-trial" and record["performance_qualification"] == "pending"
    with pytest.raises(RuntimeError, match="before startup"):
        control.configure_execution_families(native=False)


def test_a_new_admin_request_does_not_retroactively_reject_a_frozen_qualified_family():
    control = controller()
    control.enable_migration(1)
    control.configure_execution_families(native=True, graph_rows=32)
    identity, evidence, table, work = native_packet(control, graph=GRAPH_FAMILY)
    control.bind_identity(identity, evidence=evidence)
    control.install_profiles(table)
    plan = control.freeze([0], workload=work)
    control.request_migration(4)
    boundary = control.freeze_boundary({name: plan[name] for name in control.maximum},
                                       family={name: plan[name] for name in REFERENCE}, profile_id=plan["profile_id"])
    control.apply_boundary(boundary)
    control.accept(plan)
    assert control.record()["migration"]["pending"]
    assert control.record()["execution_families"]["effective"]["kernel_family"] == KERNEL_FAMILY
    next_plan = control.freeze([0], workload=work)
    assert {name: next_plan[name] for name in REFERENCE} == REFERENCE


def test_numeric_live_update_returns_to_static_family_until_an_eligible_plan_is_selected():
    control = controller()
    control.configure_execution_families(native=True, graph_rows=32)
    identity, evidence, table, work = native_packet(control, graph=GRAPH_FAMILY)
    control.bind_identity(identity, evidence=evidence)
    control.install_profiles(table)
    control.accept(control.freeze([0], workload=work))
    control.request(prefill_tokens=128)
    control.boundary()
    assert {name: control.freeze_boundary()[name] for name in REFERENCE} == REFERENCE


@pytest.mark.parametrize("trial", [False, True])
def test_every_tp_rank_applies_the_selected_family_to_buffers_before_model_work(cohort, trial):  # noqa: F811
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    family = {"kernel_family": KERNEL_FAMILY, "graph_family": GRAPH_FAMILY}
    work = None
    for peer in cohort:
        decoder = peer.decoder
        decoder.controller = AdaptiveController(slots=2, row_budget=8, proposal_limit=1, prefill_tokens=16)
        decoder.controller.configure_execution_families(native=True, graph_rows=8, trial=family if trial else None)
        decoder.buf, decoder.mbuf, decoder.pbuf = SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
        decoder.apply_execution_family = lambda dec=decoder: MultiDecoder.apply_execution_family(dec)
        if not trial:
            identity, evidence, table, work = native_packet(decoder.controller, graph=GRAPH_FAMILY)
            # This fixture's allocated proposal/prefill maxima are smaller than
            # the generic controller fixture used by the synthetic packet.
            execution = table["profiles"][0]["execution"]
            execution.update(prefill_tokens=16, row_budget=4, proposal_limit=1)
            decoder.controller.bind_identity(identity, evidence=evidence)
            if peer is cohort[0]:
                decoder.controller.install_profiles(table)

        def accept(plan, dec=decoder):
            dec.controller.accept(plan)
            dec.apply_execution_family()

        decoder.accept_execution = accept
        decoder.freeze_execution = lambda dec=decoder, observed=work: dec.controller.freeze(
            sorted(dec.streams), workload=None if trial else observed)
        original = decoder.round

        def model_round(dec=decoder, run=original):
            for buffers in (dec.buf, dec.mbuf, dec.pbuf):
                assert buffers.owner_kernel_family == KERNEL_FAMILY
                assert buffers.owner_graph_family == GRAPH_FAMILY
            return run()

        decoder.round = model_round
    root = cohort[0]
    stream = Stream([1], 1)
    root.admit(stream)
    assert root.round() == [stream]
    assert all(peer.decoder.controller.effective_family == family and peer.failure is None for peer in cohort)
    root.finish([stream])


def test_tp_family_digest_ignores_mapping_insertion_order_across_processes():
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder

    values = []
    for family in (dict(REFERENCE), dict(reversed(tuple(REFERENCE.items())))):
        decoder = object.__new__(TensorParallelDecoder)
        decoder.requests = {}
        decoder.decoder = SimpleNamespace(controller=SimpleNamespace(
            epoch=0, effective={"row_budget": 8, "proposal_limit": 1, "prefill_tokens": 16, "session_ceiling": 2},
            last_served=None, effective_family=family))
        values.append(decoder._state_digest())
    assert values[0] == values[1]
