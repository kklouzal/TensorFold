"""Per-request snapshot refusals that only an owner rank can see end the request alike on every TP rank."""

import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.torch


class _Cohort:
    """In-process stand-in for the TP all-gather of one integer per rank."""

    def __init__(self, world):
        self.world, self.values = world, [None] * world
        self.barrier = threading.Barrier(world)

    def gather(self, rank, values):
        self.values[rank] = list(values)
        self.barrier.wait()
        result = [list(item) for item in self.values]
        self.barrier.wait()
        return result


class _State:
    def __init__(self, refuse):
        self.refuse, self.pos, self.mtp_len, self.compact_indexer = refuse, 600, 601, object()
        self.validated = 0

    def snapshot(self):
        return {"pos": self.pos, "mtp_len": self.mtp_len}

    def indexer_snapshot_metadata(self, pos, mtp_len):
        if self.refuse:   # the owner rank's compact tail: its staging was reused by a later pass
            raise ValueError("compact append staging was reused before its cut/commit resolved")
        return {"compact_indexer": (pos, mtp_len)}

    def kv_validate_snapshot(self, snap):
        self.validated += 1


def _rank(rank, cohort, refuse, monkeypatch_gather):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    m = object.__new__(MultiDecoder)
    m.w = SimpleNamespace(comm=SimpleNamespace(rank=rank, world=cohort.world, all_gather=None))
    m.admission_policy = "fifo"
    m.streams, m.passed, m.remembered, m.dropped = {}, {}, [], []
    s = Stream(list(range(1000)), 8, None, draft=True)
    s.sid, s.st = 7, _State(refuse)
    m.filling = [s]
    m.fills = {s.sid: [None, True, 600, None, "tail"]}
    m._keep_at = lambda stream: 999
    m._remember = lambda ids, st, snap, tail: m.remembered.append(len(ids))
    m._drop_kept = lambda st: m.dropped.append(st)
    return m, s


@pytest.mark.parametrize("refusing", [(), (1,), (0, 1)])
def test_cancel_snapshot_outcome_is_agreed_by_every_rank(monkeypatch, refusing):
    from tensorfold.families.qwen4_exp.cuda import multi

    world = 2
    cohort = _Cohort(world)
    ranks = [_rank(rank, cohort, rank in refusing, None) for rank in range(world)]
    by_send = {}

    def gather_ints(torch, gather, values, world):
        rank = by_send[threading.get_ident()]
        return cohort.gather(rank, values)

    monkeypatch.setattr(multi, "gather_ints", gather_ints)
    results, errors = [None] * world, []

    def run(rank):
        by_send[threading.get_ident()] = rank
        try:
            m, s = ranks[rank]
            results[rank] = m.cancel([s])
        except BaseException as error:  # noqa: BLE001 - reported below
            errors.append(error)

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(world)]
    [thread.start() for thread in threads]
    [thread.join(30) for thread in threads]
    assert not errors and all(not thread.is_alive() for thread in threads)
    outcomes = [(type(s.error).__name__ if s.error is not None else None, s.done, m.remembered, len(m.dropped))
                for m, s in ranks]
    assert outcomes[0] == outcomes[1]                     # the TP scheduler digest sees one outcome
    if refusing:
        assert outcomes[0] == ("ValueError", True, [], 1)
        for rank, (_m, s) in enumerate(ranks):
            cause = s.error.__cause__
            assert (cause is not None) == (rank in refusing)  # the local refusal stays attached where it occurred
            assert str(list(refusing)) in str(s.error)
    else:
        assert outcomes[0] == (None, True, [600], 0)


def test_single_rank_cancel_keeps_its_local_refusal(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import multi

    monkeypatch.setattr(multi, "gather_ints", lambda *args, **kwargs: pytest.fail("no collective without a cohort"))
    m, s = _rank(0, _Cohort(1), True, None)
    m.w = SimpleNamespace(comm=None)
    m.cancel([s])
    assert type(s.error) is ValueError and "staging was reused" in str(s.error) and m.remembered == [] and m.dropped
