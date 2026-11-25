"""CPU history resumption and committed placement contracts."""
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event

import pytest

from tensorfold.families.deepseek_v4.cuda import expert_balance_state
from tensorfold.families.deepseek_v4.cuda.expert_balance_state import ExpertBalanceState
from tensorfold.families.deepseek_v4.cuda.expert_transfer import plan_expert_transfers


def state(path, *, checkpoint="fixture", etp=4, current=None):
    return ExpertBalanceState(path, checkpoint=checkpoint, etp=etp, layers=1, experts=4,
                              half_life_routes=4, current=current)


def test_history_resume_requires_fresh_routes_in_both_phases(tmp_path):
    history = state(tmp_path / "expert-load.json")
    initial = history.path.read_bytes()
    assert history.record((((4, 4, 0, 0),), ((0, 0, 0, 0),)))
    assert history.path.read_bytes() == initial
    assert history.live_recommendation(minimum_routes=4) is None
    history.flush()
    history.close()
    resumed = state(history.path)
    assert resumed.snapshot() == history.snapshot()
    assert resumed.snapshot()[1] == ((8,), (0,))
    assert resumed.record((((0, 0, 0, 0),), ((4, 4, 0, 0),)))
    assert resumed.live_recommendation(minimum_routes=4) is None
    assert resumed.record((((4, 4, 0, 0),), ((0, 0, 0, 0),)))
    epoch, plan = resumed.live_recommendation(minimum_routes=4)
    assert resumed.placement_snapshot() == (epoch, plan.current) == (0, ((0, 1, 2, 3),))
    assert plan.proposed != plan.current
    resumed.flush()
    resumed.close()
    again = state(history.path)
    try:
        assert again.snapshot() == resumed.snapshot()
    finally:
        again.close()


def test_observation_invalidates_a_plan_still_being_computed(tmp_path, monkeypatch):
    history = state(tmp_path / "history.json")
    assert history.record((((4, 4, 0, 0),), ((4, 4, 0, 0),)))
    assert history.live_recommendation(minimum_routes=4) is not None
    original = expert_balance_state.plan_expert_placement
    planning, observed = Event(), Event()

    def blocked_plan(*args, **kwargs):
        planning.set()
        assert observed.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(expert_balance_state, "plan_expert_placement", blocked_plan)
    with ThreadPoolExecutor(max_workers=1) as worker:
        proposal = worker.submit(history.live_recommendation, minimum_routes=4)
        assert planning.wait(timeout=5)
        history.record((((0, 0, 4, 4),), ((0, 0, 4, 4),)))
        observed.set()
        assert proposal.result(timeout=5) is None
    history.close()


def test_history_refuses_mismatched_checkpoint_and_bad_counts(tmp_path):
    history = state(tmp_path / "history.json")
    history.flush()
    history.close()
    with pytest.raises(ValueError, match="another checkpoint"):
        state(history.path, checkpoint="other")
    history = state(history.path)
    before = history.snapshot()
    with pytest.raises(ValueError, match="counts values"):
        history.record((((0, 0, -1, 0),), ((0, 0, 0, 0),)))
    assert history.snapshot() == before
    history.close()


def test_active_writer_holds_stable_lock_until_close(tmp_path):
    history = state(tmp_path / "history.json")
    marker = tmp_path / "history.json.lock"
    inode = marker.stat().st_ino
    with pytest.raises(RuntimeError, match="already has a writer"):
        state(history.path)
    history.close()
    with pytest.raises(RuntimeError, match="closed"):
        history.flush()
    other = state(history.path)
    try:
        assert marker.stat().st_ino == inode
        assert marker.read_bytes() == b""
    finally:
        other.close()


def test_constructor_failure_releases_lock_and_symlink_is_refused(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("bad JSON")
    with pytest.raises(ValueError, match="history JSON"):
        state(path)
    path.unlink()
    usable = state(path)
    usable.close()
    target = tmp_path / "untouched"
    target.write_text("prior")
    other = tmp_path / "other.json"
    marker = tmp_path / "other.json.lock"
    marker.symlink_to(target)
    with pytest.raises(OSError):
        state(other)
    assert marker.is_symlink() and target.read_text() == "prior"


def test_live_open_migrates_v1_loads_and_exposures_without_selecting_old_proposal(tmp_path):
    path = tmp_path / "history.json"
    before = {
        "version": 1, "checkpoint": "fixture", "etp": 4, "layers": 1, "experts": 4,
        "half_life_routes": 4, "loads": [[[0.5, 0.5, 0, 0]], [[0.5, 0.5, 0, 0]]],
        "exposures": [[8], [8]],
    }
    path.write_text(json.dumps(before))
    advisory = path.with_name("history.json.placement.json")
    prior_advisory = b'{"version":1,"checkpoint":"fixture","etp":4,"placement":[[0,2,1,3]]}'
    advisory.write_bytes(prior_advisory)
    with pytest.raises(ValueError, match="another checkpoint"):
        state(path, checkpoint="other")
    assert json.loads(path.read_bytes()) == before

    live = state(path)
    assert live.snapshot() == ((((0.5, 0.5, 0, 0),), ((0.5, 0.5, 0, 0),)), ((8,), (8,)))
    assert live.placement_snapshot() == (0, ((0, 1, 2, 3),))
    assert live.live_recommendation(minimum_routes=4) is None
    assert advisory.read_bytes() == prior_advisory
    saved = json.loads(path.read_bytes())
    assert saved == {**before, "version": 2, "epoch": 0, "placement": [[0, 1, 2, 3]]}
    live.close()


def test_live_commit_is_epoch_bound_and_resumes_exact_placement(tmp_path):
    path = tmp_path / "history.json"
    live = state(path)
    assert live.record((((4, 4, 0, 0),), ((4, 4, 0, 0),)))
    epoch, proposal = live.live_recommendation(minimum_routes=4)
    assert epoch == 0 and live.validate_live_placement(proposal, expected_epoch=epoch)
    assert live.placement_snapshot() == (epoch, proposal.current)
    assert proposal.proposed != proposal.current
    with pytest.raises(ValueError, match="active placement epoch"):
        live.validate_live_placement(proposal, expected_epoch=1)
    wrong_old = plan_expert_transfers(proposal.proposed, proposal.current, etp=4)
    with pytest.raises(ValueError, match="active placement epoch"):
        live.validate_live_placement(wrong_old, expected_epoch=epoch)

    expected_loads = live.snapshot()
    assert live.commit_live_placement(proposal, expected_epoch=epoch) == 1
    assert live.placement_snapshot() == (1, proposal.proposed)
    no_change = plan_expert_transfers(proposal.proposed, proposal.proposed, etp=4)
    assert not live.validate_live_placement(no_change, expected_epoch=1)
    saved = path.read_bytes()
    assert live.commit_live_placement(no_change, expected_epoch=1) == 1
    with pytest.raises(ValueError, match="active placement epoch"):
        live.commit_live_placement(proposal, expected_epoch=epoch)
    assert path.read_bytes() == saved
    assert live.live_recommendation(minimum_routes=4) is None
    live.close()

    with pytest.raises(ValueError, match="conflicts with the committed epoch"):
        state(path, current=proposal.current)
    resumed = state(path, current=proposal.proposed)
    assert resumed.placement_snapshot() == (1, proposal.proposed)
    assert resumed.snapshot() == expected_loads
    resumed.close()


@pytest.mark.parametrize("after_replace", [False, True])
def test_failed_live_commit_prevents_any_further_publication(tmp_path, monkeypatch, after_replace):
    path = tmp_path / "history.json"
    live = state(path)
    epoch, start = live.placement_snapshot()
    assert epoch == 0
    proposed = ((0, 2, 1, 3),)
    plan = plan_expert_transfers(start, proposed, etp=4)
    atomic_replace = expert_balance_state._atomic_replace

    def fail(destination, data):
        if after_replace:
            atomic_replace(destination, data)
        raise OSError("simulated publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(expert_balance_state, "_atomic_replace", fail)
        with pytest.raises(OSError, match="publication failure"):
            live.commit_live_placement(plan, expected_epoch=0)
        with pytest.raises(RuntimeError, match="publication failed"):
            live.flush()
        with pytest.raises(RuntimeError, match="publication failed"):
            live.placement_snapshot()
    live.close()
    resumed = state(path)
    assert resumed.placement_snapshot() == ((1, proposed) if after_replace else (0, start))
    resumed.close()
