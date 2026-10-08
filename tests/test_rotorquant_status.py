"""CPU protocol evidence for sticky status and rank-coherent frame containment.

Fake collectives verify participation/error policy, not NCCL or two-GPU ordering.
No weights, runtime installation or CUDA initialization is needed.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda.forward import _kv_check  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import KVNumericError, State  # noqa: E402
from tensorfold.families.qwen4_exp.kv_formats import get_pair  # noqa: E402


def state(status=(0, -2), world=2, pair=None):
    st = State.__new__(State)
    st.kv_pair = pair or get_pair("bf16" if status is None else "int8", value_override=None if status is None else "rotorquant8")
    st.kv_key_dtype, st.kv_value_dtype = st.kv_pair.key_dtype, st.kv_pair.value_dtype
    st.pos, st.mtp_len, st.capacity = 7, 5, 16
    st.kv_identity = (st.kv_pair.identity, "stored-basis-native64-v1")
    st.kv_status = None if status is None else torch.tensor(status, dtype=torch.int32)
    st._kv_peer_status = torch.empty((world, 2), dtype=torch.int32) if world > 1 else None
    st._kv_pending, st._kv_error = status is not None, None
    return st


class Collective:
    world = 2

    def __init__(self, statuses, rank):
        self.statuses = torch.tensor(statuses, dtype=torch.int32)
        self.rank, self.calls = rank, 0

    def all_gather(self, local, peers):
        assert torch.equal(local, self.statuses[self.rank])
        assert peers.shape == self.statuses.shape
        self.calls += 1
        peers.copy_(self.statuses)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("statuses", [((4, -1), (0, -2)), ((0, -2), (2, 9)), ((1, 3), (8, 7))])
def test_every_rank_joins_collective_and_latches_local_or_peer_numeric_failure(rank, statuses):
    st = state(statuses[rank])
    collective = Collective(statuses, rank)
    with pytest.raises(KVNumericError, match="requires reset"):
        _kv_check([(st, 0, 1)], comm=collective)
    assert collective.calls == 1
    assert st._kv_error is not None
    assert torch.equal(st.kv_status, torch.tensor(statuses[rank], dtype=torch.int32))
    with pytest.raises(KVNumericError):
        st.kv_begin()
    assert not torch.cuda.is_initialized()


def test_valid_collective_keeps_stable_peer_buffer_and_clears_pending_only():
    st = state()
    collective = Collective(((0, -2), (0, -2)), 0)
    pointer = st._kv_peer_status.data_ptr()
    _kv_check([(st, 0, 1)], comm=collective)
    assert collective.calls == 1 and st._kv_peer_status.data_ptr() == pointer
    assert not st._kv_pending and st._kv_error is None
    assert st.kv_status.tolist() == [0, -2]


def test_native_states_do_not_join_numeric_collective():
    st = state(None)
    collective = Collective(((0, -2), (0, -2)), 0)
    _kv_check([(st, 0, 1)], comm=collective)
    assert collective.calls == 0 and not st._kv_pending


@pytest.mark.parametrize("peer_shape", [None, (1, 2), (2, 1)])
def test_invalid_peer_buffer_is_rejected_before_collective(peer_shape):
    st = state()
    st._kv_peer_status = None if peer_shape is None else torch.empty(peer_shape, dtype=torch.int32)
    collective = Collective(((0, -2), (0, -2)), 0)
    with pytest.raises(ValueError, match="peer validation buffer"):
        _kv_check([(st, 0, 1)], comm=collective)
    assert collective.calls == 0


def test_all_local_invalid_participants_latch_before_first_error_is_raised():
    first, second = state((1, 4), world=1), state((2, 5), world=1)
    with pytest.raises(KVNumericError, match="layer=4"):
        _kv_check([(first, 0, 1), (second, 1, 2)])
    assert first._kv_error is not None and second._kv_error is not None


@pytest.mark.parametrize("dtype", ["bf16", "int8", "int4"])
def test_native_snapshot_metadata_requires_identity_and_validated_status(dtype):
    st = state(None, world=1, pair=get_pair(dtype))
    metadata = st.kv_snapshot_metadata()
    assert metadata == {"kv_identity": (st.kv_pair.identity, "stored-basis-native64-v1"), "kv_status": (0, -2)}
    st.kv_validate_snapshot({"pos": 7, "mtp_len": 5, **metadata})
    with pytest.raises(ValueError, match="matching ordered formats"):
        st.kv_validate_snapshot({"pos": 7, "mtp_len": 5})


@pytest.mark.parametrize("bad_identity", [
    (get_pair("rotorquant8", value_override="int8").identity, "stored-basis-native64-v1"),
    (get_pair("int8", value_override="rotorquant8-norm").identity, "stored-basis-native64-v1"),
    (get_pair("int8", value_override="rotorquant8").identity, "different-working-policy"),
])
def test_swapped_norm_or_working_policy_snapshot_is_rejected_before_state_changes(bad_identity):
    st = state(world=1)
    snap = {"pos": 7, "mtp_len": 5, **st.kv_snapshot_metadata()}
    snap["kv_identity"] = bad_identity
    with pytest.raises(ValueError, match="matching ordered formats"):
        st.kv_validate_snapshot(snap)
    assert (st.pos, st.mtp_len) == (7, 5)
