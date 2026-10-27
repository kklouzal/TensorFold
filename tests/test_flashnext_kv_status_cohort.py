"""The packed TP status path retains the existing per-sequence failure contract."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.forward import _kv_check
from tensorfold.families.qwen4_exp.cuda.kv_status import KVStatusCohort, workspace_bytes
from tensorfold.families.qwen4_exp.cuda.state import State


def states_for(words, world):
    states = []
    for bits, layer in words:
        st = object.__new__(State)
        st.execution_dtype = torch.float16
        st.kv_key_dtype, st.kv_value_dtype = "fp16", "int8"
        st.kv_status = torch.tensor([bits, layer], dtype=torch.int32)
        st._kv_peer_status = torch.empty((world, 2), dtype=torch.int32)
        st._kv_pending, st._kv_error, st._resize_error = True, None, None
        st.compact_indexer = None
        st.pos, st.mtp_len = 17, 13
        states.append(st)
    return states


def run(words, rank, packed):
    world, slots, _ = words.shape
    states = states_for(words[rank].tolist(), world)
    calls = []

    def gather(source, destination):
        calls.append(source.clone())
        if packed:
            torch.testing.assert_close(source, words[rank], atol=0, rtol=0)
            destination.copy_(words.reshape(-1))
        else:
            column = len(calls) - 1
            torch.testing.assert_close(source, words[rank, column], atol=0, rtol=0)
            destination.copy_(words[:, column])

    comm = SimpleNamespace(rank=rank, world=world, all_gather=gather)
    workspace = KVStatusCohort(slots, comm, "cpu") if packed else None
    error = None
    try:
        _kv_check([(st, row, row + 1) for row, st in enumerate(states)], comm=comm, workspace=workspace)
    except RuntimeError as caught:
        error = caught
    return states, calls, error


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("slots", [1, 3, 6])
@pytest.mark.parametrize("failure", ["none", "remote", "local_and_remote", "all"])
def test_packed_status_is_identical_to_sequence_checks_including_error_precedence(world, slots, failure):
    rank = 0
    words = torch.zeros((world, slots, 2), dtype=torch.int32)
    words[:, :, 1] = -2
    if failure in ("remote", "local_and_remote"):
        words[-1, 0] = torch.tensor([4, -1])
    if failure == "local_and_remote":
        words[rank, -1] = torch.tensor([2, 7])
    if failure == "all":
        words[:, :, 0] = 8
        words[:, :, 1] = torch.arange(slots)
    expected, separate, reference_error = run(words, rank, False)
    actual, gathered, received_error = run(words, rank, True)
    assert len(separate) == slots and len(gathered) == 1
    assert type(received_error) is type(reference_error) and str(received_error) == str(reference_error)
    for state, reference in zip(actual, expected):
        assert state._kv_pending is reference._kv_pending is False
        assert type(state._kv_error) is type(reference._kv_error)
        assert str(state._kv_error) == str(reference._kv_error)
        if state._kv_error is not None:
            assert state._kv_error is received_error


def test_workspace_size_matches_retained_arrays_and_is_independent_of_context():
    workspace = KVStatusCohort(6, SimpleNamespace(rank=0, world=8), "cpu")
    assert workspace_bytes(6, 8) == (432, 384)
    assert workspace.device_bytes == workspace.send.nbytes + workspace.receive.nbytes == 432
    assert workspace.host_bytes == workspace.host.nbytes == 384


@pytest.mark.parametrize("slots,world", [(0, 8), (6, 1), (True, 8), (6, False)])
def test_invalid_geometry_refuses_before_allocating(slots, world):
    with pytest.raises(ValueError, match="positive slots"):
        workspace_bytes(slots, world)


def test_cohort_never_silently_changes_its_admitted_world_or_execution_policy():
    comm = SimpleNamespace(rank=0, world=8)
    workspace = KVStatusCohort(1, comm, "cpu")
    states = states_for([[0, -2]], 8)
    with pytest.raises(ValueError, match="slots/rank/world"):
        workspace.check(states, comm=SimpleNamespace(rank=1, world=8))
    states[0].execution_dtype = torch.bfloat16
    with pytest.raises(ValueError, match="FP16"):
        workspace.check(states, comm=comm)
