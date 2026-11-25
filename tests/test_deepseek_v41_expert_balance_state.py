"""CPU history resumption and restart-only recommendation contracts."""
import pytest

from tensorfold.families.deepseek_v4.cuda.expert_balance_state import ExpertBalanceState
from tensorfold.families.deepseek_v4.cuda.expert_placement import decode_placement


def state(path, *, checkpoint="fixture", etp=4):
    return ExpertBalanceState(path, checkpoint=checkpoint, etp=etp, layers=1, experts=4,
                              half_life_routes=4, current=None)


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
