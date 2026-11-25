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
    assert state(history.path).snapshot() == resumed.snapshot()


def test_observation_invalidates_unpublished_proposal(tmp_path):
    history = state(tmp_path / "history.json")
    assert history.record((((4, 4, 0, 0),), ((4, 4, 0, 0),)))
    proposal = history.recommendation(minimum_routes=4)
    assert proposal is not None
    history.record((((0, 0, 4, 4),), ((0, 0, 4, 4),)))
    with pytest.raises(ValueError, match="stale"):
        history.publish_recommendation(proposal)
    assert not history.recommendation_path.exists()


def test_history_refuses_mismatched_checkpoint_and_bad_counts(tmp_path):
    history = state(tmp_path / "history.json")
    history.flush()
    with pytest.raises(ValueError, match="another checkpoint"):
        state(history.path, checkpoint="other")
    before = history.snapshot()
    with pytest.raises(ValueError, match="counts values"):
        history.record((((0, 0, -1, 0),), ((0, 0, 0, 0),)))
    assert history.snapshot() == before
