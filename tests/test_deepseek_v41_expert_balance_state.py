"""CPU history resumption and committed placement contracts."""
import json

import pytest

from tensorfold.families.deepseek_v4.cuda import expert_balance_state
from tensorfold.families.deepseek_v4.cuda.expert_balance_state import ExpertBalanceState
from tensorfold.families.deepseek_v4.cuda.expert_placement import decode_placement
from tensorfold.families.deepseek_v4.cuda.expert_transfer import plan_expert_transfers


def state(path, *, checkpoint="fixture", etp=4, current=None, live=False):
    return ExpertBalanceState(path, checkpoint=checkpoint, etp=etp, layers=1, experts=4,
                              half_life_routes=4, current=current, live=live)


def test_history_resume_requires_fresh_routes_in_both_phases(tmp_path):
    history = state(tmp_path / "expert-load.json")
    assert history.record((((4, 4, 0, 0),), ((0, 0, 0, 0),)))
    assert not history.path.exists()
    assert history.recommendation(minimum_routes=4) is None
    history.flush()
    history.close()
    resumed = state(history.path)
    assert resumed.snapshot() == history.snapshot()
    assert resumed.snapshot()[1] == ((8,), (0,))
    assert resumed.record((((0, 0, 0, 0),), ((4, 4, 0, 0),)))
    assert resumed.recommendation(minimum_routes=4) is None
    assert resumed.record((((4, 4, 0, 0),), ((0, 0, 0, 0),)))
    assert resumed.recommendation() is None
    proposal = resumed.recommendation(minimum_routes=4)
    assert proposal is not None
    resumed.publish_recommendation(proposal)
    next_placement = decode_placement(resumed.recommendation_path.read_bytes(), checkpoint="fixture", layers=1,
                                      experts=4, etp=4, source="test proposal")
    assert next_placement != resumed.current == ((0, 1, 2, 3),)
    resumed.flush()
    resumed.close()
    again = state(history.path)
    try:
        assert again.snapshot() == resumed.snapshot()
    finally:
        again.close()


def test_observation_invalidates_unpublished_proposal(tmp_path):
    history = state(tmp_path / "history.json")
    assert history.record((((4, 4, 0, 0),), ((4, 4, 0, 0),)))
    proposal = history.recommendation(minimum_routes=4)
    assert proposal is not None
    history.record((((0, 0, 4, 4),), ((0, 0, 4, 4),)))
    with pytest.raises(ValueError, match="stale"):
        history.publish_recommendation(proposal)
    assert not history.recommendation_path.exists()
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
    history = state(path)
    assert history.record((((4, 4, 0, 0),), ((4, 4, 0, 0),)))
    history.publish_recommendation(history.recommendation(minimum_routes=4))
    history.flush()
    before = json.loads(path.read_bytes())
    advisory = history.recommendation_path.read_bytes()
    snapshot = history.snapshot()
    history.close()
    assert before["version"] == 1 and "placement" not in before

    live = state(path, live=True)
    assert live.snapshot() == snapshot
    assert live.placement_snapshot() == (0, ((0, 1, 2, 3),))
    assert live.live_recommendation(minimum_routes=4) is None
    assert live.recommendation_path.read_bytes() == advisory
    saved = json.loads(path.read_bytes())
    assert saved == {**before, "version": 2, "epoch": 0, "placement": [[0, 1, 2, 3]]}
    live.close()
    with pytest.raises(ValueError, match="requires live expert balance"):
        state(path)
    assert json.loads(path.read_bytes()) == saved


def test_live_commit_is_epoch_bound_and_resumes_exact_placement(tmp_path):
    path = tmp_path / "history.json"
    live = state(path, live=True)
    assert live.record((((4, 4, 0, 0),), ((4, 4, 0, 0),)))
    epoch, proposal = live.live_recommendation(minimum_routes=4)
    assert epoch == 0 and live.validate_live_placement(proposal, expected_epoch=epoch)
    assert proposal.current == live.current
    assert proposal.proposed != live.current
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
        state(path, current=proposal.current, live=True)
    resumed = state(path, current=proposal.proposed, live=True)
    assert resumed.placement_snapshot() == (1, proposal.proposed)
    assert resumed.snapshot() == expected_loads
    resumed.close()


@pytest.mark.parametrize("after_replace", [False, True])
def test_failed_live_commit_prevents_any_further_publication(tmp_path, monkeypatch, after_replace):
    path = tmp_path / "history.json"
    live = state(path, live=True)
    start = live.current
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
    resumed = state(path, live=True)
    assert resumed.placement_snapshot() == ((1, proposed) if after_replace else (0, start))
    resumed.close()
