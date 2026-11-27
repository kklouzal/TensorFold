from types import SimpleNamespace

import pytest

from tensorfold.families.deepseek_v4.cuda.expert_live import LiveExpertPlacement
from tensorfold.families.deepseek_v4.cuda.expert_rebalance_wire import PeerRefusal, Phase, Refusal, placement_digest
from tensorfold.families.deepseek_v4.cuda.expert_transfer import (
    expert_groups_changed,
    local_expert_layout,
    plan_expert_transfers,
)


def setup_transfer(etp):
    row = tuple(range(384))
    current = (row,) * 40
    cut = 384 // (8 // etp)
    changed = list(row)
    changed[0], changed[cut] = changed[cut], changed[0]
    desired = (tuple(changed), *current[1:])
    plan = plan_expert_transfers(current, desired, etp=etp)
    actions = []
    decoder = SimpleNamespace(broken=None, retire_placement_prefixes=lambda: actions.append("retire"))
    engine = SimpleNamespace(rank=0, expert_tensor_parallel_size=etp, decoder=decoder)
    live = LiveExpertPlacement(engine, epoch=4, placement=current, operation=33)

    class History:
        def validate_live_placement(self, candidate, *, expected_epoch):
            assert candidate == plan and expected_epoch == 4
            actions.append("validate")
            return True

        def commit_live_placement(self, candidate, *, expected_epoch):
            assert candidate == plan and expected_epoch == 4 and live.epoch == 5
            actions.append("commit")
            return 5

    return live, plan, History(), actions


@pytest.mark.parametrize("etp", (1, 2, 4))
def test_unchanged_local_slots_keep_the_arithmetic_route(etp):
    _, plan, _, _ = setup_transfer(etp)
    for owner in range(8 // etp):
        assert local_expert_layout(plan.current[0], owner=owner, group_slots=plan.group_slots) is None
        layout = local_expert_layout(plan.proposed[0], owner=owner, group_slots=plan.group_slots)
        if owner >= 2:
            assert layout is None
        else:
            selected, inverse = layout
            offset = owner * plan.group_slots
            assert selected == plan.proposed[0][offset:offset + plan.group_slots]
            assert all(inverse[logical] == slot for slot, logical in enumerate(selected))
            assert inverse.count(-1) == 384 - plan.group_slots


def test_apply_rebuilds_with_cohort_wide_canonical_group_membership(monkeypatch):
    import torch

    _, move, _, _ = setup_transfer(4)
    canonical = move.current[0]
    within = (canonical[1], canonical[0], *canonical[2:])
    mixed = (move.proposed[0], within, *move.proposed[2:])
    restored = (canonical, *mixed[1:])
    assert not expert_groups_changed(within, group_slots=move.group_slots)
    assert local_expert_layout(within, owner=0, group_slots=move.group_slots) is not None
    stream = SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)

    for rank in range(8):
        layers = [SimpleNamespace(expert_to_local=None, expert_groups_changed=False) for _ in move.current]
        graph_actions = []
        engine = SimpleNamespace(rank=rank, expert_tensor_parallel_size=4, graphs=object(),
                                 _draft_verify=SimpleNamespace(release_graphs=lambda: graph_actions.append("release")),
                                 w=SimpleNamespace(group=rank // 4, layers=layers),
                                 prefix=SimpleNamespace(retire_placement_prefixes=lambda: None),
                                 comm=SimpleNamespace(device="cpu", check=lambda: None, barrier=lambda: None))
        live = LiveExpertPlacement(engine, epoch=0, placement=move.current, operation=33)
        observed = []

        def capture():
            assert engine.graphs is None
            assert graph_actions[-2:] == ["release", "move"]
            graph_actions.append("capture")
            observed.append([layer.expert_groups_changed for layer in layers])
            engine.graphs = object()

        engine.recapture_expert_graphs = capture
        for epoch, proposed in enumerate((mixed, restored), 1):
            plan = plan_expert_transfers(live.placement, proposed, etp=4)
            pending = SimpleNamespace(plan=plan, movement=SimpleNamespace(enqueue=lambda: graph_actions.append("move")),
                                      transaction=SimpleNamespace(target_epoch=epoch,
                                                                  target_digest=placement_digest(proposed, etp=4)))
            live._apply_prepared(pending)
            assert live.epoch == epoch and live.placement == proposed
            assert (layers[1].expert_to_local is not None) == (rank < 4)
        assert observed == [[True, *([False] * 39)], [False] * 40]


@pytest.mark.parametrize("etp", (1, 2, 4))
def test_live_placement_commits_only_after_all_rank_apply_receipts(etp):
    live, plan, history, actions = setup_transfer(etp)

    def round_trip(phase, transaction, ids=()):
        assert (transaction.current_epoch, transaction.target_epoch, transaction.nonce) == (4, 5, 1)
        actions.append(phase)
        if phase == Phase.READY:
            assert len(ids) == 40 * 384 and live.placement == plan.current
        else:
            assert phase == Phase.APPLIED and not ids
            live.epoch, live.placement = 5, plan.proposed

    live._round_trip = round_trip
    assert live.apply(plan, expected_epoch=4, history=history)
    assert actions == ["validate", Phase.READY, "retire", Phase.APPLIED, "commit"]
    assert live.engine.decoder.broken is None


def test_live_placement_aborts_a_clean_preflight_refusal_without_retirement():
    live, plan, history, actions = setup_transfer(2)

    def round_trip(phase, transaction, ids=()):
        actions.append(phase)
        if phase == Phase.READY:
            raise PeerRefusal(((3, Refusal.PREFLIGHT),))
        assert phase == Phase.ABORTED and not ids

    live._round_trip = round_trip
    with pytest.raises(PeerRefusal):
        live.apply(plan, expected_epoch=4, history=history)
    assert actions == ["validate", Phase.READY, Phase.ABORTED]
    assert live.engine.decoder.broken is None and live.placement == plan.current


def test_live_placement_poisons_the_cohort_after_a_partial_apply():
    live, plan, history, actions = setup_transfer(4)
    failure = RuntimeError("missing apply receipt")

    def round_trip(phase, transaction, ids=()):
        actions.append(phase)
        if phase == Phase.APPLIED:
            raise failure

    live._round_trip = round_trip
    with pytest.raises(RuntimeError, match="missing apply receipt"):
        live.apply(plan, expected_epoch=4, history=history)
    assert actions == ["validate", Phase.READY, "retire", Phase.APPLIED]
    assert live.engine.decoder.broken is failure


@pytest.mark.parametrize("degree", (None, 1, 2, 4))
def test_live_placement_uses_degree_specific_model_sidecar(tmp_path, monkeypatch, degree):
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine

    options = []
    monkeypatch.setattr(DS41Engine, "_initialize", lambda self, model_dir, **kw: options.append(kw))

    DS41Engine(tmp_path, rank=0, tp=8, serve=True, expert_tensor_parallel_size=degree)
    assert options[0]["expert_balance_state"] == tmp_path / f"expert-balance-etp{degree or 4}.json"
    assert options[0]["expert_balance_live"] is True

    explicit = tmp_path / "store" / "explicit.json"
    DS41Engine(tmp_path, rank=0, tp=8, serve=True, expert_balance_state=explicit)
    assert options[1]["expert_balance_state"] == explicit


def test_expert_balance_opt_out_ignores_model_sidecar(tmp_path, monkeypatch):
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine

    options = []
    monkeypatch.setattr(DS41Engine, "_initialize", lambda self, model_dir, **kw: options.append(kw))
    path = tmp_path / "expert-balance-etp4.json"
    path.write_text("not valid state")

    DS41Engine(tmp_path, rank=0, tp=8, serve=True, no_expert_balance=True)
    assert options[0]["expert_balance_state"] is None and options[0]["expert_balance_live"] is False
    with pytest.raises(ValueError, match="disabled expert balancing"):
        DS41Engine(tmp_path, rank=0, tp=8, serve=True, no_expert_balance=True, expert_balance_state=path)


def test_expert_shutdown_flushes_both_snapshots_without_proposing(monkeypatch):
    import torch
    from tensorfold.families.deepseek_v4.cuda.expert_observer import ExpertObserver

    actions = []
    observations, device = iter(([1], [2])), object()
    observer = object.__new__(ExpertObserver)
    observer.write = None
    observer.copy_pending = True

    def request_snapshot():
        actions.append("copy")
        return True

    observer.usage = SimpleNamespace(poll_snapshot=lambda: SimpleNamespace(tolist=lambda: next(observations)),
                                     request_snapshot=request_snapshot, counts=SimpleNamespace(device=device))

    def record(rows):
        actions.append(rows)
        return True

    observer.history = SimpleNamespace(record=record, flush=lambda: actions.append("flush"))
    def stream(bound_device):
        assert bound_device is device
        return SimpleNamespace(synchronize=lambda: actions.append("fence"))

    monkeypatch.setattr(torch.cuda, "current_stream", stream)
    observer._finish_pending()
    assert actions == [[1], "copy", "fence", [2], "flush"]
