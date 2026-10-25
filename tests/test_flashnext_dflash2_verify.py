"""Target-distribution, pruning, ownership and output-room integration controls."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.engine.exact_sampling import Sampling, uniform  # noqa: E402
from tensorfold.engine.rejection import ACCEPT, ChainPacket, ProbabilityRow  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.dflash2_verify import bind_packet, verify_batch  # noqa: E402


def state(position=7):
    window = SimpleNamespace(generation=1, end=position, profile_identity="current-runtime-rank-profile",
                             check=lambda **_kwargs: None)
    return SimpleNamespace(pos=position, slot_generation=1, dflash2=window)


def point(token):
    return ProbabilityRow([token], [1.0])


def test_stochastic_verification_uses_actual_conditional_q_and_target_rows(monkeypatch):
    from tensorfold.cuda import sampling as sampler

    st = state()
    p, q = ProbabilityRow([0, 1], [.1, .9]), ProbabilityRow([0, 1], [.9, .1])
    seed = next(seed for seed in range(100) if uniform(seed ^ ACCEPT, 8, np.array([0]))[0] > 1 / 9)
    sampling = Sampling(temperature=1, top_k=2, seed=seed)
    packet = bind_packet(st, ChainPacket((0,), (q,), 8))
    observed = []

    def actual_target(rows, settings):
        observed.append((rows.shape, settings))
        return [p, point(0)]

    monkeypatch.setattr(sampler, "effective_rows", actual_target)
    result = verify_batch(SimpleNamespace(comm=None), torch.zeros((2, 2)), [(st, 0, 2)],
                          [packet], [sampling], [10], [()])[0]
    assert result.path == (0,) and result.terminal == 1 and result.outputs == (1,)
    assert observed == [(torch.Size([2, 2]), sampling)]
    assert packet.chain.q_rows[0] is q and q.probability(0) == .9


@pytest.mark.parametrize("room,path,outputs", [(1, (0,), (1,)), (2, (0, 1), (1, 2))])
def test_greedy_bonus_and_last_allowed_pending_anchor(monkeypatch, room, path, outputs):
    from tensorfold.cuda import sampling as sampler

    st = state()
    packet = bind_packet(st, ChainPacket((1,), (point(1),), 8))
    monkeypatch.setattr(sampler, "sample_rows", lambda *_args: [1, 2])
    result = verify_batch(SimpleNamespace(comm=None), torch.zeros((2, 3)), [(st, 0, 2)],
                          [packet], [None], [room], [()])[0]
    assert result.path == path and result.outputs == outputs


def test_grammar_pruned_descendants_keep_original_q_and_are_unreachable(monkeypatch):
    from tensorfold.cuda import sampling as sampler

    st = state()
    original = ChainPacket((0, 1, 2), (point(0), point(1), point(2)), 8)
    monkeypatch.setattr(sampler, "sample_rows", lambda *_args: [4])
    result = verify_batch(SimpleNamespace(comm=None), torch.zeros((1, 5)), [(st, 0, 1)],
                          [bind_packet(st, original)], [None], [10], [()])[0]
    assert result.path == (0,) and result.outputs == (4,)
    assert len(original.tokens) == len(original.q_rows) == 3


@pytest.mark.parametrize("mutation", ["slot", "profile", "rows", "position", "unbound", "other_window", "closed_window"])
def test_bad_or_retired_batch_fails_before_any_target_collective(monkeypatch, mutation):
    from tensorfold.cuda import sampling as sampler

    st = state()
    packet = bind_packet(st, ChainPacket((1,), (point(1),), 8))
    if mutation == "slot":
        st.slot_generation += 1
    elif mutation == "profile":
        st.dflash2.profile_identity = "replacement-runtime"
    elif mutation == "position":
        st.pos += 1
    elif mutation == "unbound":
        packet = packet.chain
    elif mutation == "other_window":
        st.dflash2 = state().dflash2  # Same generation, position and profile; distinct logical owner.
    elif mutation == "closed_window":
        st.dflash2 = None
    monkeypatch.setattr(sampler, "effective_shards", lambda *_a, **_k: pytest.fail("collective was entered"))
    rows = 3 if mutation == "rows" else 2
    plan = SimpleNamespace(check_model=lambda *_args: None)
    with pytest.raises(ValueError):
        verify_batch(SimpleNamespace(comm=object()), torch.zeros((rows, 3)), [(st, 0, rows)],
                     [packet], [Sampling(temperature=1, seed=0)], [10], [()], plan=plan)


def test_empty_batch_does_not_require_or_touch_collective_owner():
    assert verify_batch(SimpleNamespace(comm=object()), torch.empty((0, 3)), [], [], [], [], []) == []


def actual_window_state():
    from tensorfold.families.qwen4_exp.cuda.dflash2_config import DFlash2Config
    from tensorfold.families.qwen4_exp.cuda.dflash2_runtime import DraftWindow
    from test_flashnext_dflash2 import small_config

    window = DraftWindow(DFlash2Config.from_dict(small_config()), tp=(0, 1), context=64, device="cpu", generation=1)
    window.end = window.available = 7
    return SimpleNamespace(pos=7, slot_generation=1, dflash2=window)


@pytest.mark.parametrize("terminal", ["closed", "poisoned"])
def test_actual_window_terminal_state_invalidates_previously_bound_packet_before_sampling(monkeypatch, terminal):
    from tensorfold.cuda import sampling as sampler

    st = actual_window_state()
    packet = bind_packet(st, ChainPacket((1,), (point(1),), 8))
    setattr(st.dflash2, terminal, True)
    monkeypatch.setattr(sampler, "effective_shards", lambda *_a, **_k: pytest.fail("target collective entered"))
    plan = SimpleNamespace(check_model=lambda *_args: pytest.fail("target sampler plan entered"))
    with pytest.raises(ValueError, match="poisoned"):
        verify_batch(SimpleNamespace(comm=object()), torch.zeros((2, 3)), [(st, 0, 2)],
                     [packet], [Sampling(temperature=1, seed=0)], [10], [()], plan=plan)


def test_actual_pending_feature_transaction_stays_live_through_verification(monkeypatch):
    from tensorfold.cuda import sampling as sampler
    from tensorfold.families.qwen4_exp.cuda.dflash2_features import FeatureFrame
    from tensorfold.families.qwen4_exp.cuda.dflash2_runtime import PendingFeatures

    st = actual_window_state()
    window = st.dflash2
    packet = bind_packet(st, ChainPacket((1,), (point(1),), 8))
    frame = FeatureFrame(window.config, 2, "cpu")
    frame.generation, frame.readers = 1, 1
    window.pending = PendingFeatures(frame, frame.generation, window.generation, window.end,
                                     frame.features[:2].flatten(1))
    with pytest.raises(ValueError, match="accepted-prefix decision"):
        window.check()  # Proposal construction still requires resolved state.
    monkeypatch.setattr(sampler, "sample_rows", lambda *_args: [1, 2])
    result = verify_batch(SimpleNamespace(comm=None), torch.zeros((2, 3)), [(st, 0, 2)],
                          [packet], [None], [10], [()])[0]
    assert result.outputs == (1, 2) and result.path == (0, 1)
    assert window.pending is not None and frame.readers == 1  # the coordinator still owns commit/stop decision
