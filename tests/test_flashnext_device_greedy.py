"""Exact full-shard maxima and deferred MTP publication without a CUDA device."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.device_greedy import GreedyHandoff, WORDS, recorded_token, workspace_bytes  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.kv_status import KVStatusCohort  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import CAND, KVNumericError, State  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402


def frame(scores, ids, norms=None):
    result = np.zeros((len(scores), WORDS), dtype=np.int64)
    result[:, 0] = np.asarray(scores, dtype=np.float32).view(np.int32)
    result[:, 1] = ids
    result[:, 2] = np.asarray(scores if norms is None else norms, dtype=np.float32).view(np.int32)
    result[:, 4] = -1
    return result


def test_original_maxima_use_global_minimum_id_for_ties():
    packet = frame([3, 3, 2], [777, 9, 17])
    assert recorded_token(packet, 9) == 9
    with pytest.raises(ValueError, match="differs"):
        recorded_token(packet, 777)


@pytest.mark.parametrize("scores,norms", [([float("nan"), 3], [0, 3]),
    ([float("inf"), 3], [float("inf"), 3]), ([-float("inf")]*2, [-float("inf")]*2),
    ([3, 2], [float("nan"), 3])])
def test_complete_domain_nonfinite_errors_survive_a_finite_winning_shard(scores, norms):
    with pytest.raises(ValueError):
        recorded_token(frame(scores, [9, 17], norms), 9)


@pytest.mark.parametrize("width,depth,world", [(1, 1, 2), (9949, 3, 8), (31040, 7, 8)])
def test_workspace_retains_every_step_and_exact_global_ids(width, depth, world):
    device, host = workspace_bytes(width, depth, world)
    assert host == depth * (world * WORDS + 1) * 8
    assert device >= host + width * 17


def test_actual_tensor_handoff_repairs_more_than_top32_ties_and_unsorted_ids(monkeypatch):
    original = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: original(*args, **{
        k: v for k, v in kwargs.items() if k != "pin_memory"}))
    width, world = 71, 2
    logits = torch.zeros((1, width), dtype=torch.float32)
    logits[:, 40:] = -3
    ids = torch.arange(400, 400+width).flip(0)
    remote = frame([0], [17], [np.log(40)])

    def gather(send, receive):
        receive.view(world, WORDS)[0].copy_(send)
        receive.view(world, WORDS)[1].copy_(torch.from_numpy(remote[0]))

    comm = NS(world=world, rank=0, all_gather=gather)
    plan = NS(width=width, offset=0, device_map=ids, check_model=lambda w, values: None)
    candidates = torch.zeros(world * (2*CAND+1), dtype=torch.float32)
    candidates.view(world, -1)[:, -1] = logits.logsumexp(-1)
    dec = NS(w=NS(comm=comm, device="cpu", cfg=NS(vocab=1000)), depth=3, _draft_plan=plan,
             mbuf=NS(cand_all=candidates))
    owner = GreedyHandoff(NS(dec=dec))
    owner.storage.zero_()
    owner.gather(0, logits, NS(kv_status=torch.tensor([0, -1], dtype=torch.int32)))
    assert owner.input_token.item() == 17
    assert recorded_token(owner.frames[0].numpy(), int(owner.tokens[0])) == 17


def test_recorded_remote_failure_prevents_any_position_publication(monkeypatch):
    state = object.__new__(State)
    state._kv_error = state._resize_error = None
    state._kv_pending = True
    state.kv_status = torch.tensor([0, -1], dtype=torch.int32)
    state.kv_key_dtype = state.kv_value_dtype = "int8"
    state.pos, state.mtp_len, state.mtp_drafted = 19, 7, 0
    cohort = object.__new__(KVStatusCohort)
    cohort.rank, cohort.world = 0, 2
    owner = object.__new__(GreedyHandoff)
    owner.depth, owner.world = 3, 2
    owner.w = NS(device="cpu")
    owner.dec = NS(kv_cohort=cohort)
    owner.storage = torch.zeros(3 * (2 * WORDS + 1), dtype=torch.int64)
    frames = owner.storage[:3*2*WORDS].view(3, 2, WORDS)
    for step in range(3):
        frames[step].copy_(torch.from_numpy(frame([3, 2], [9, 17])))
    frames[1, 1, 3] = 4  # healthy first step, remote MTP error in the second
    owner.storage[-3:] = 9
    owner.host = torch.empty_like(owner.storage)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: NS(synchronize=lambda: None))
    with pytest.raises(KVNumericError, match="rank 1"):
        owner.finish(state, 3, 7, 2)
    assert state.mtp_len == 7 and state.mtp_drafted == 0 and state._kv_error is not None


def test_fallback_eligibility_does_not_rewind_or_publish_state():
    owner = object.__new__(GreedyHandoff)
    state = NS(pos=12, mtp_len=10, mtp_drafted=2, _kv_pending=False, _kv_error=None, _resize_error=None)
    stream = NS(st=state, sid=1, count=9, out=[1], sampling=None, copies=None, constraint=None)
    owner.error, owner.depth = None, 3
    owner.plan = NS(graph_ids_safe=True)
    owner.dec = NS(confidence=0, w=NS(mtp=NS(layer=NS(linear=False))), kv_cohort=object(),
                   _rotor_lookup=lambda value: False)
    owner.bound = NS(mtp_ready=lambda *args: False)
    assert owner.eligible([(stream, 0, [2])], None) is None
    assert (state.mtp_len, state.mtp_drafted) == (10, 2)


def policy_owner(sampling, position=12):
    owner = object.__new__(GreedyHandoff)
    state = NS(pos=position, mtp_len=10, mtp_drafted=2, _kv_pending=False, _kv_error=None, _resize_error=None)
    stream = NS(st=state, sid=1, count=9, out=[1], sampling=sampling, copies=None, constraint=None)
    owner.error, owner.depth = None, 3
    owner.plan = NS(graph_ids_safe=True)
    owner.dec = NS(confidence=0, w=NS(mtp=NS(layer=NS(linear=False))), kv_cohort=object(),
                   _rotor_lookup=lambda value: False)
    owner.bound = NS(mtp_ready=lambda *args: True)
    return owner, stream


@pytest.mark.parametrize("policy", [None, Sampling(0, temperature=0), Sampling(0, temperature=-1)])
def test_only_valid_greedy_sampling_can_enter_device_handoff(policy):
    owner, stream = policy_owner(policy)
    assert owner.eligible([(stream, 0, [2])], None) == (stream, 0, [2], 3, 8)


@pytest.mark.parametrize("policy", [
    Sampling(0, temperature=float("nan")), Sampling(0, temperature=-float("inf")),
    Sampling(0, temperature=0, top_p=2), Sampling(0, temperature=0, min_p=-1),
    Sampling(1.5, temperature=0), Sampling(True, temperature=0),
    Sampling(0, temperature=0, top_p=float("nan")), object(),
])
def test_invalid_sampling_falls_back_to_original_validation_before_any_device_work(policy):
    owner, stream = policy_owner(policy)
    owner.bound.mtp_ready = lambda *args: pytest.fail("invalid policies must not inspect or stage graph keys")
    assert owner.eligible([(stream, 0, [2])], None) is None
    assert (stream.st.mtp_len, stream.st.mtp_drafted, stream.st._kv_error) == (10, 2, None)


@pytest.mark.parametrize("position", [-2, (1 << 64)-2])
def test_all_known_draft_positions_keep_unsigned64_validation(position):
    owner, stream = policy_owner(None, position)
    assert owner.eligible([(stream, 0, [2])], None) is None


def test_declined_policy_preserves_original_absorb_codec_error_precedence(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import multi

    policy = Sampling(0, temperature=float("nan"))
    owner, stream = policy_owner(policy)
    assert owner.eligible([(stream, 0, [2])], None) is None
    monkeypatch.setattr(multi, "validate_rows", lambda *args: None)
    events = []

    def failed_absorb():
        events.append("codec")
        raise KVNumericError("original failed absorb wins")

    with pytest.raises(KVNumericError, match="original failed absorb wins"):
        multi.MultiDecoder._picks(NS(), torch.zeros(1, 3), [13], [policy], NS(settle=failed_absorb))
    assert events == ["codec"]
    with pytest.raises(ValueError, match="finite temperature"):
        multi.MultiDecoder._picks(NS(), torch.zeros(1, 3), [13], [policy],
                                 NS(settle=lambda: events.append("healthy codec")))
    assert events == ["codec", "healthy codec"]


@pytest.mark.parametrize("failed", [False, True])
def test_device_handoff_observes_each_submitted_graph_span_without_a_new_readback(monkeypatch, failed):
    from tensorfold.families.qwen4_exp.cuda import mtp
    from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController

    controller = AdaptiveController(slots=1, row_budget=4, proposal_limit=3, prefill_tokens=4)
    controller.work_observations.configure_indexer(ratio=4, budget=11, heads=4, target_caches=12)
    state = NS(mtp_len=9, mtp_drafted=1, _kv_error=None)
    state.set_mtp_len = lambda value: setattr(state, "mtp_len", value)
    stream = NS(st=state, drafts=[])
    owner = object.__new__(GreedyHandoff)
    owner.busy, owner.error, owner.completed, owner.steps = False, None, 0, 0
    owner.dec = NS(controller=controller, mbuf=NS(streams=torch.zeros((2, 1))),
                   buf=NS(streams=torch.zeros((2, 1))))
    submitted = []
    def submit(segments, *, mtp_positions, device_tokens):
        submitted.append((mtp_positions[0], segments[-1][2]))
        return torch.zeros((1, 2))
    owner.dec.graphs = NS(mtp_step=submit)
    owner.storage = torch.zeros(1)
    owner.input_token = torch.zeros(1, dtype=torch.int32)
    owner.eligible = lambda streams, new: (stream, 0, [40, 50], 3, 8)
    primary = RuntimeError("controlled failed handoff")
    def gather(step, logits, state):
        if failed:
            raise primary
    owner.gather = gather
    owner.finish = lambda state, count, start, kept: [60, 70, 80]
    monkeypatch.setattr(mtp, "mtp_stage", lambda buffer, windows:
                        [(windows[0][0], 0, len(windows[0][1]))])
    if failed:
        with pytest.raises(RuntimeError) as caught:
            owner.try_draft([(stream, 0, [40, 50])], None)
        assert caught.value is primary and state._kv_error is not None
    else:
        assert owner.try_draft([(stream, 0, [40, 50])], None)
    controller.work_observations.publish()
    reading = controller.record()["work_observations"]
    assert reading["totals"]["mtp_refresh_rows_submitted"] == 2
    assert reading["totals"]["mtp_proposal_rows_submitted"] == (0 if failed else 2)
    assert submitted == ([(8, 2)] if failed else [(8, 2), (10, 1), (11, 1)])
    assert reading["totals"]["incomplete_forward_submissions"] == int(failed)
    assert reading["expert_group_max"] is None and not owner.busy
