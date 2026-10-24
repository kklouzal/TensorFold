"""The real Flash Next request/settings protocol with eight CPU-controlled peers."""

from datetime import timedelta
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

torch = pytest.importorskip("torch")


class Store:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value.encode() if isinstance(value, str) else value

    def wait(self, keys, duration):
        assert duration == timedelta(hours=1)
        assert all(key in self.values for key in keys)

    def get(self, key):
        return self.values[key]

    def delete_key(self, key):
        del self.values[key]


def engine(rank, world, store):
    result = object.__new__(FlashNextEngine)
    result.tp, result.rank, result.served = world, rank, 0
    result.comm = SimpleNamespace(store=store)
    result._lifecycle = threading.Condition()
    result._closing = False
    result._shutdown_sent = False
    result.points = lambda prompt: [300, 700]
    return result


@pytest.mark.parametrize("world", [2, 4, 8])
def test_every_follower_consumes_its_own_request_and_shutdown(world):
    store = Store()
    leader = engine(0, world, store)
    followers = [engine(rank, world, store) for rank in range(1, world)]
    expected = leader._share([17] * 800, 16, None, True, 300)
    assert len(store.values) == world - 1
    for follower in reversed(followers):
        assert follower._receive() == expected
    assert store.values == {}
    leader.served = 1
    leader._send_shutdown()
    assert len(store.values) == world - 1
    for follower in followers:
        follower.served = 1
        assert follower._receive() is None
    assert store.values == {}
    leader._send_shutdown()  # One publication after quiescence; no duplicate stop frames.
    assert store.values == {}


def test_two_rank_store_key_is_backward_compatible():
    assert engine(0, 2, Store())._key(11, 1) == "tensorfold/flashnext/request/11"
    assert engine(1, 2, Store())._key(11) == "tensorfold/flashnext/request/11"
    assert engine(7, 8, Store())._key(11) == "tensorfold/flashnext/request/11/rank/7"


class CPUWords:
    int64 = torch.int64
    equal = staticmethod(torch.equal)

    @staticmethod
    def tensor(values, **options):
        return torch.tensor(values, dtype=options["dtype"])

    @staticmethod
    def empty(shape, **options):
        return torch.empty(shape, dtype=options["dtype"])


@pytest.mark.parametrize("bad_rank,field", [(None, None), (7, 0), (6, 8), (5, 9), (4, 10)])
def test_settings_consensus_checks_every_rank(bad_rank, field):
    result = engine(0, 8, Store())
    result.depth, result.confidence, result.max_len = 0, 0.5, 4096
    result.streams, result.graphs_enabled, result.prefill_rows = 1, False, 256
    result.execution_provider = "sm70"
    result.kv_pair = SimpleNamespace(handshake=123)
    result.rope = SimpleNamespace(metadata=lambda: {"rope_type": "default"})

    def gather(mine, received):
        rows = received.view(8, -1)
        rows.copy_(mine.expand(8, -1))
        if bad_rank is not None:
            rows[bad_rank, field] += 1

    result.comm.all_gather = gather
    if bad_rank is None:
        result._same_settings(CPUWords, np.array([1, 2, 5], dtype=np.int64))
    else:
        with pytest.raises(RuntimeError, match=f"rank {bad_rank}"):
            result._same_settings(CPUWords, np.array([1, 2, 5], dtype=np.int64))


def test_draft_ids_with_equal_counts_and_sums_have_distinct_handshake_words():
    captured = []
    result = engine(0, 8, Store())
    result.depth, result.confidence, result.max_len = 2, 0.5, 4096
    result.streams, result.graphs_enabled, result.prefill_rows = 1, False, 256
    result.kv_pair = SimpleNamespace(handshake=123)
    result.rope = SimpleNamespace(metadata=lambda: {})

    def gather(mine, received):
        captured.append(mine.clone())
        received.view(8, -1).copy_(mine.expand(8, -1))

    result.comm.all_gather = gather
    result._same_settings(CPUWords, np.array([1, 4], dtype=np.int64))
    result._same_settings(CPUWords, np.array([2, 3], dtype=np.int64))
    assert captured[0][6] == captured[1][6]
    assert captured[0][7] != captured[1][7]


def test_copy_draft_policy_changes_the_concurrent_rank_handshake(monkeypatch):
    captured = []
    result = engine(0, 8, Store())
    result.depth, result.confidence, result.max_len = 2, 0.5, 4096
    result.streams, result.graphs_enabled, result.prefill_rows = 2, False, 256
    result.kv_pair = SimpleNamespace(handshake=123)
    result.rope = SimpleNamespace(metadata=lambda: {})

    def gather(mine, received):
        captured.append(mine.clone())
        received.view(8, -1).copy_(mine.expand(8, -1))

    result.comm.all_gather = gather
    monkeypatch.delenv("TENSORFOLD_MTP_COPY", raising=False)
    result._same_settings(CPUWords, None)
    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    result._same_settings(CPUWords, None)
    assert captured[0][9] != captured[1][9]


def _half_status_state():
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    st = object.__new__(State)
    st.execution_dtype = torch.float16
    st.kv_pair = get_pair('fp16')
    st.kv_key_dtype = st.kv_value_dtype = 'fp16'
    st.kv_status = torch.tensor([0, -2], dtype=torch.int32)
    st._kv_peer_status = torch.empty((8, 2), dtype=torch.int32)
    st._kv_pending, st._kv_error, st._resize_error = True, None, None
    st.pos, st.mtp_len = 9, 0
    return st


def test_half_multi_segment_status_collectives_finish_before_remote_error_is_raised():
    from tensorfold.families.qwen4_exp.cuda.forward import _kv_check
    from tensorfold.families.qwen4_exp.cuda.state import KVNumericError

    calls = []

    def gather(source, destination):
        calls.append(source.clone())
        destination.copy_(source.expand(8, -1))
        destination[7].copy_(torch.tensor([2, 3]))

    states = [_half_status_state(), _half_status_state()]
    with pytest.raises(KVNumericError, match='rank 7') as caught:
        _kv_check([(states[0], 0, 1), (states[1], 1, 2)],
                  comm=SimpleNamespace(world=8, all_gather=gather))
    assert len(calls) == 2
    assert all(st._kv_error is caught.value for st in states)


def test_tp_concurrent_mtp_reports_peer_numeric_failure_before_publishing_logits(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import multi
    from tensorfold.families.qwen4_exp.cuda.state import KVNumericError

    def gather(source, destination):
        destination.copy_(source.expand(8, -1))
        destination[7].copy_(torch.tensor([4, -1]))

    decoder = object.__new__(multi.MultiDecoder)
    decoder.w = SimpleNamespace(comm=SimpleNamespace(world=8, all_gather=gather))
    decoder.live = lambda: 0
    decoder.mbuf = SimpleNamespace(h=torch.empty(1, dtype=torch.float16))
    decoder.graphs = None
    logits = torch.zeros((1, 64))
    monkeypatch.setattr(multi, 'mtp_compute', lambda *args, **kwargs: logits)
    with pytest.raises(KVNumericError, match='rank 7'):
        decoder._mtp([(_half_status_state(), 0, 1)])
    assert decoder.mbuf.attn_step is None
