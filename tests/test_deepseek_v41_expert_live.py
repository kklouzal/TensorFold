from types import SimpleNamespace

import pytest

from tensorfold.families.deepseek_v4.cuda.expert_live import LiveExpertPlacement
from tensorfold.families.deepseek_v4.cuda.expert_rebalance_wire import PeerRefusal, Phase, Refusal
from tensorfold.families.deepseek_v4.cuda.expert_transfer import plan_expert_transfers


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
