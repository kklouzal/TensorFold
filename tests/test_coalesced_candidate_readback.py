"""Status-first packet packing and exact original chooser math, without GPU claims."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from .test_flashnext_kv_status_cohort import states_for
from tensorfold.families.qwen4_exp.cuda.kv_status import KVStatusCohort, workspace_bytes, _CANDIDATE_HANDOFF_BYTES
from tensorfold.families.qwen4_exp.cuda.state import CAND, KVNumericError
from tensorfold.families.qwen4_exp.cuda.decode import choose_gathered
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.engine.exact_sampling import Sampling


def prepared(*, world=2, rows=3, failure=False):
    words = torch.zeros((world, rows, 2), dtype=torch.int32)
    words[..., 1] = -2
    if failure:
        words[-1, 0] = torch.tensor([4, -1], dtype=torch.int32)
    states = states_for(words[0].tolist(), world)
    comm = NS(rank=0, world=world, all_gather=lambda source, dest: dest.copy_(words.reshape(-1)))
    owner = KVStatusCohort(rows, comm, "cpu", candidate_rows=rows)
    packet = np.empty((world, rows, 2 * CAND + 1), dtype=np.float32)
    packet[:, :, :CAND] = -np.arange(CAND, dtype=np.float32)[None, None, :] / 32
    packet[:, :, CAND:2*CAND] = np.broadcast_to(
        np.arange(world*CAND, dtype=np.int32).reshape(world, 1, CAND), (world, rows, CAND)).view(np.float32)
    packet[:, :, -1] = np.float32(3.25)
    if rows > 1:
        packet[1, 1, 0] = 2
    if rows > 2:
        packet[:, 2, :CAND] = 1  # Complete-shard ties retain the original global-ID rule.
    return owner, comm, states, torch.from_numpy(packet), packet


def test_one_readback_retains_all_fp32_packet_bits_and_survives_arena_reuse(monkeypatch):
    owner, comm, states, device, packet = prepared()
    calls = []
    original = owner._readback
    monkeypatch.setattr(owner, "_readback", lambda count: calls.append(count) or original(count))
    actual = owner.read_candidates(states, device, 3, comm=comm)
    assert calls == [2*3*2 + 2*3*(2*CAND+1)]
    assert np.array_equal(actual.view(np.uint32), packet.view(np.uint32))
    assert actual.flags.owndata and not any(state._kv_pending for state in states)
    owner.host.fill_(19)
    owner.receive.fill_(31)
    assert np.array_equal(actual.view(np.uint32), packet.view(np.uint32))
    # Only one escaped owner is live; retire it before another candidate packet.
    del actual
    owner.read_candidates(states, device, 3, comm=comm)
    device_bytes, host_bytes = workspace_bytes(3, 2, candidate_rows=3)
    assert device_bytes == owner.send.nbytes + owner.receive.nbytes
    assert host_bytes == owner.host.nbytes + packet.nbytes + np.empty((0,0,0), dtype=np.float32).__sizeof__() + _CANDIDATE_HANDOFF_BYTES


@pytest.mark.parametrize("malformed", [False, True])
def test_numeric_error_precedes_packet_interpretation_or_malformed_error(monkeypatch, malformed):
    owner, comm, states, device, _ = prepared(failure=True)
    if malformed:
        device = device.double()
    calls = []
    original = torch.Tensor.numpy
    monkeypatch.setattr(torch.Tensor, "numpy", lambda value: calls.append(value) or original(value))
    with pytest.raises(KVNumericError):
        owner.read_candidates(states, device, 3, comm=comm)
    assert calls == [] and states[0]._kv_error is not None


@pytest.mark.parametrize("failure", ["gather", "copy", "readback"])
def test_transport_failure_poison_retains_primary_and_refuses_owner_reuse(monkeypatch, failure):
    owner, comm, states, device, _ = prepared()
    primary = RuntimeError("injected readback transport failure")
    def fail(*_args, **_kwargs):
        raise primary
    if failure == "gather":
        comm.all_gather = fail
    elif failure == "readback":
        # Fail the actual IO primitive inside _readback so its error owner latches it.
        original = torch.Tensor.copy_
        monkeypatch.setattr(torch.Tensor, "copy_", lambda target, source, **kw:
                            fail() if target.data_ptr() == owner.host.data_ptr() else original(target, source, **kw))
    else:
        original = torch.Tensor.copy_
        candidate_start = owner.receive.data_ptr() + 2*3*2*4
        monkeypatch.setattr(torch.Tensor, "copy_", lambda target, source, **kw:
                            fail() if target.data_ptr() == candidate_start else original(target, source, **kw))
    with pytest.raises(RuntimeError) as caught:
        owner.read_candidates(states, device, 3, comm=comm)
    assert caught.value is primary and owner._readback_error is primary
    with pytest.raises(RuntimeError, match="requires retirement") as stopped:
        owner.read_candidates(states, device, 3, comm=comm)
    assert stopped.value.__cause__ is primary


def test_owned_snapshot_keeps_original_ties_ids_and_fp64_confidence_without_second_cpu_copy(monkeypatch):
    owner, comm, states, device, _ = prepared()
    packet = owner.read_candidates(states, device, 3, comm=comm)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _self: True))
    weights, plan = NS(meta={"world": 2}), NS(widths=(CAND, CAND), graph_ids_safe=True)
    expected = choose_gathered(weights, device, 3, [1,2,3], Sampling(17, 0), with_prob=True, plan=plan)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda _self: pytest.fail("second candidate CPU copy"))
    actual = choose_gathered(weights, device, 3, [1,2,3], Sampling(17, 0), with_prob=True, plan=plan, gathered=packet)
    assert actual == expected
    assert actual[0] == [0, CAND, 0]
    with pytest.raises(ValueError, match="must own"):
        choose_gathered(weights, device, 3, [1,2,3], Sampling(17, 0), plan=plan, gathered=packet.view())


@pytest.mark.parametrize("samplings,enabled", [
    ([Sampling(17, 0)]*2, True), ([None]*2, True), ([Sampling(17, 0)], False),
    ([Sampling(17, 1)]*2, False), ([Sampling(17, 0), Sampling(18, 0)], False)])
def test_only_actual_multi_state_homogeneous_greedy_cohorts_use_coalescing(samplings, enabled):
    decoder = object.__new__(MultiDecoder)
    decoder.w = NS(meta={"sm70_owner_batch_prelude_trial": True}, comm=object(), draft_head=None)
    decoder.kv_cohort = NS(candidate_rows=32)
    assert decoder._coalesced_candidates(samplings) is enabled
    decoder.w.meta.clear()
    assert not decoder._coalesced_candidates(samplings)
