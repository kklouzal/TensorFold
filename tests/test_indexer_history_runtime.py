"""The existing forward/graph protocol selects only live RAM-history journals."""
from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda import history_runtime as runtime
from tensorfold.families.qwen4_exp.cuda.indexer_history import HistoryPlan, HistorySession
from tests.test_indexer_history import Allocator


@pytest.fixture
def owned():
    session = HistorySession(HistoryPlan(3, 4, 4, 8, 2), Allocator(), "cpu", maximum_owners=4)
    history = session.new(64)
    state = SimpleNamespace(indexer_history_owner=history, kc=[object(), object()], mtp_ikc=object(),
                            pos=0, mtp_len=0)
    try:
        yield state, history
    finally:
        session.close()


def test_default_gpu_states_are_execution_neutral():
    segs = [(SimpleNamespace(indexer_history_owner=None), 0, 2)]
    assert runtime.begin(segs, positions=[]) == ()
    runtime.enqueue(segs)
    runtime.submitted(segs)
    runtime.abort(segs)


def test_main_vs_mtp_explicit_layers_and_validation(owned):
    state, history = owned
    segs = [(state, 0, 2)]
    runtime.begin(segs, prefill=True)
    history.frames["prefill"].gpu.fill_(7)
    runtime.enqueue(segs, prefill=True)
    runtime.submitted(segs, prefill=True)
    history.validated(fenced=True)
    assert history.snapshot([2, 2, 0])["lengths"] == (2, 2, 0)
    runtime.begin(segs, mtp=True)
    history.frames["decode"].gpu[2].fill_(11)
    runtime.enqueue(segs, mtp=True)
    runtime.submitted(segs, mtp=True)
    history.validated(fenced=True)
    assert history.snapshot([2, 2, 2])["lengths"] == (2, 2, 2)


def test_bound_padding_requires_known_logical_counts(owned):
    state, history = owned
    state.graph_rows = object()
    segs = [(state, 0, 4)]
    with pytest.raises(ValueError, match="host logical"):
        runtime.begin(segs)
    leases = runtime.begin(segs, counts=[1])
    assert len(leases) == 1 and history._pending["decode"] == {0: (0, 1), 1: (0, 1)}
    runtime.abort(segs, fenced=True)
    state.history_rows = 2
    runtime.begin(segs)
    assert history._pending["decode"] == {0: (0, 2), 1: (0, 2)}


@pytest.mark.parametrize("changes", [{"counts": [5]}, {"counts": [True]}, {"counts": []},
                                     {"positions": [-1]}, {"positions": []}])
def test_bad_host_frame_metadata_is_rejected_without_archiving(owned, changes):
    state, history = owned
    with pytest.raises(ValueError):
        runtime.begin([(state, 0, 4)], **changes)
    assert history.snapshot([0, 0, 0])["lengths"] == (0, 0, 0)


def test_repeated_mtp_calls_retain_one_dynamic_origin_until_validation(owned):
    state, history = owned
    segs = [(state, 0, 1)]
    runtime.begin(segs, mtp=True)
    history.frames["decode"].gpu[2, 0].fill_(1)
    runtime.enqueue(segs, mtp=True)
    runtime.submitted(segs, mtp=True)
    state.mtp_len = 1
    runtime.begin(segs, mtp=True)
    history.frames["decode"].gpu[2, 1].fill_(2)
    runtime.enqueue(segs, mtp=True)
    runtime.submitted(segs, mtp=True)
    assert int(history.origins[2]) == 0
    history.validated(fenced=True)
    assert history.archive[2, 0, 0] == 1 and history.archive[2, 1, 0] == 2


def test_duplicate_owner_segments_are_rejected(owned):
    state, _ = owned
    with pytest.raises(ValueError, match="one indexer history owner"):
        runtime.begin([(state, 0, 1), (state, 1, 2)])
