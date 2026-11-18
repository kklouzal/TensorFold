"""A recoverable failure on some TP ranks only must retire the cohort, never leave the others inside a collective."""

import threading
from types import SimpleNamespace

import pytest
from test_flashnext_tp_scheduler import BatchStore, Decoder

from tensorfold.families.qwen4_exp.cuda import tp_scheduler
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder


class TimedStore(BatchStore):
    """A wait that reports torch's own timeout (store_wait.timed_out) once its duration passes."""

    def wait(self, keys, duration):
        from torch.distributed import DistStoreError

        with self.condition:
            if not self.condition.wait_for(lambda: all(key in self.data for key in keys), duration.total_seconds()):
                milliseconds = int(duration.total_seconds() * 1000)
                raise DistStoreError(f"wait timeout after {milliseconds}ms, keys: " + ", ".join("/" + k for k in keys))


class RankLocal(Decoder):
    """``round`` raises only on the ranks in ``failing``; the others would enter the round's next collective."""

    def __init__(self, rank, world, store, failing, *, pipes=False):
        super().__init__(rank, world, store, pipes=pipes)
        self.failing = failing

    def round(self):
        if self.rank in self.failing:
            raise ValueError("compact append staging was reused before its cut/commit resolved")
        return []


def _engine():
    return SimpleNamespace(_abort_comm=False, _closing=False, _lifecycle=threading.Condition())


def _cohort(world, failing, monkeypatch, *, pipes=False):
    monkeypatch.setattr(tp_scheduler, "ERROR_AGREEMENT_S", 0.5)
    store = TimedStore()
    return store, [TensorParallelDecoder(RankLocal(rank, world, store, failing, pipes=pipes), _engine())
                   for rank in range(world)]


def _apply(peer, outcome):
    try:
        peer.apply(peer.receive())
        outcome.append(None)
    except BaseException as error:  # noqa: BLE001 - asserted by the caller
        outcome.append(error)


def test_a_peer_failing_alone_retires_the_cohort_instead_of_waiting_for_its_next_command(monkeypatch):
    # rank 1 raises, the leader does not: in production the leader then sits in the round's next collective and never
    # compares acknowledgments, so the peer is the only rank that can end the wait.
    _store, (leader, peer) = _cohort(2, {1}, monkeypatch)
    outcome = []
    thread = threading.Thread(target=_apply, args=(peer, outcome))
    thread.start()
    leader.decoder.round = lambda: threading.Event().wait(2) and []   # the stuck leader: no comparison in time
    stuck = threading.Thread(target=lambda: pytest.raises(RuntimeError, leader.round), daemon=True)
    stuck.start()
    thread.join(10)
    assert not thread.is_alive()
    (error,) = outcome
    assert type(error) is RuntimeError and "did not confirm" in str(error)
    assert type(error.__cause__) is ValueError and "staging was reused" in str(error.__cause__)
    assert peer.failure is error and peer.engine._abort_comm      # poisoned: the supervisor ends every rank
    stuck.join(10)


def test_a_leader_failing_alone_stops_waiting_for_acknowledgments_after_the_bound(monkeypatch):
    _store, (leader, peer) = _cohort(2, {0}, monkeypatch)
    peer.decoder.round = lambda: threading.Event().wait(5) and []     # the peer sits in the skipped collective
    thread = threading.Thread(target=lambda: peer.apply(peer.receive()), daemon=True)
    thread.start()
    with pytest.raises(RuntimeError, match="did not acknowledge within 0.5 s") as caught:
        leader.round()
    assert leader.failure is caught.value and leader.engine._abort_comm


@pytest.mark.parametrize("pipes", [False, True], ids=["store", "control-pipes"])
def test_ranks_failing_alike_keep_the_recoverable_request_failure(monkeypatch, pipes):
    store, ranks = _cohort(3, {0, 1, 2}, monkeypatch, pipes=pipes)
    leader, peers = ranks[0], ranks[1:]
    outcomes = [[] for _ in peers]
    threads = [threading.Thread(target=_apply, args=(peer, outcome)) for peer, outcome in zip(peers, outcomes)]
    [thread.start() for thread in threads]
    with pytest.raises(ValueError, match="staging was reused"):
        leader.round()
    [thread.join(10) for thread in threads]
    assert outcomes == [[None], [None]]                       # confirmed alike: every peer goes on serving
    assert all(rank.failure is None and not rank.engine._abort_comm for rank in ranks)
    assert store.data == {}                                   # the confirmation keys are retired by their peers
    leader.shutdown()
