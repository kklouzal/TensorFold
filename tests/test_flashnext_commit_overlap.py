"""MultiDecoder._draft_all runs the round's target commits once: during the TP refresh, else before drafting."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import multi


def decoder(*, comm, proposals):
    d = object.__new__(multi.MultiDecoder)
    d.controller, d.depth, d.capacity = None, 3, 1000
    d.w, d.mbuf = NS(dspark=None, comm=comm), NS()
    d.buf = NS(streams=torch.zeros(4, 1), rows=4)
    d.kv_host, d.graphs = None, None
    d._proposal_count = lambda s, remaining: proposals
    d._proposal_limit = lambda: 0                 # the chain itself is not under test
    d._pad_rows = lambda *args: None
    return d


@pytest.mark.parametrize("comm,proposals", [(object(), 1), (object(), 0), (None, 1), (None, 0)])
def test_commits_run_once_inside_the_tp_refresh_wait_else_before_drafting(monkeypatch, comm, proposals):
    events = []
    state = NS(mtp_len=5, mtp_drafted=0, pos=6, indexer_history_owner=None)
    state.set_mtp_len = lambda n: events.append(("absorb", n))
    stream = NS(st=state, sid=0, out=[], count=100, drafts=[], sampling=None)
    d = decoder(comm=comm, proposals=proposals)

    def stage(buffer, windows, **kwargs):
        events.append("stage")
        return [(state, 0, 2)]

    def mtp(segs, *, logits=True, pick=False, before_wait=None):
        events.append("launch")
        if comm is not None and logits and before_wait is not None:
            before_wait()                          # _mtp runs the hook between launch and the status wait
        events.append("wait")
        return torch.zeros(1, 4) if logits else None

    class Words:
        def __init__(self, segments, host, absorbed=None):
            self.absorbed = absorbed

        def settle(self):
            if self.absorbed is not None:
                self.absorbed()

    monkeypatch.setattr(multi, "mtp_stage", stage)
    monkeypatch.setattr(multi, "KVWords", Words)
    d._mtp = mtp
    d._draft_all([(stream, 0, [7, 8])], commits=lambda inflight=False: events.append(("commit", inflight)))
    commits = [event for event in events if type(event) is tuple and event[0] == "commit"]
    events = [event for event in events if not (type(event) is tuple and event[0] == "absorb")]
    assert len(commits) == 1
    if comm is not None and proposals:
        assert events == ["stage", "launch", ("commit", True), "wait"]
    elif comm is not None:                         # no hook ran (no logits): the commits follow the refresh
        assert events == ["stage", "launch", "wait", ("commit", False)]
    else:                                          # single device: the incumbent order
        assert events[0] == ("commit", False)


def test_commits_precede_drafting_for_ram_indexer_history(monkeypatch):
    events = []
    state = NS(mtp_len=5, mtp_drafted=0, pos=6, indexer_history_owner=object())
    stream = NS(st=state, sid=0, out=[], count=100, drafts=[], sampling=None)
    d = decoder(comm=object(), proposals=1)
    monkeypatch.setattr(multi, "mtp_stage", lambda buffer, windows, **kw: events.append("stage") or [(state, 0, 2)])
    monkeypatch.setattr(multi, "KVWords", lambda segments, host, absorbed=None: NS(settle=lambda: None))
    d._mtp = lambda segs, **kw: (events.append("launch"), kw.get("before_wait") and kw["before_wait"](),
                                 torch.zeros(1, 4))[2]
    d._draft_all([(stream, 0, [7, 8])], commits=lambda inflight=False: events.append(("commit", inflight)))
    assert events[0] == ("commit", False) and events.count(("commit", False)) == 1
