"""CPU protocol/fault tests; no CUDA context or NCCL execution is simulated as proof."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
import threading

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.profile_construction import (
    Construction, ConstructionFatal, ConstructionRefused, replay,
)


class Store:
    def __init__(self):
        self.condition = threading.Condition()
        self.data = {}
        self.maximum = 0

    def set(self, key, value):
        with self.condition:
            self.data[key] = value.encode("utf-8") if isinstance(value, str) else value
            self.maximum = max(self.maximum, len(self.data))
            self.condition.notify_all()

    def check(self, keys):
        with self.condition:
            return all(key in self.data for key in keys)

    def get(self, key):
        with self.condition:
            return self.data[key]

    def delete_key(self, key):
        with self.condition:
            self.data.pop(key, None)

    def wait(self, keys, duration):
        assert duration == timedelta(hours=1)
        with self.condition:
            assert self.condition.wait_for(lambda: all(key in self.data for key in keys), 5), keys


@pytest.mark.parametrize("value,expected", [("native-\u00e9-contract", b"native-\xc3\xa9-contract"),
                                           (b"\x00\xff\x00", b"\x00\xff\x00")])
def test_store_producer_matches_native_utf8_bytes_contract(value, expected):
    store = Store()
    store.set("value", value)
    assert type(store.data["value"]) is bytes and store.data["value"] == expected
    assert type(store.get("value")) is bytes and store.get("value") == expected
    assert store.check(["value"]) and store.maximum == 1
    store.delete_key("value")
    assert not store.check(["value"]) and store.data == {}


class Comm:
    def __init__(self, rank, world, store):
        self.rank, self.world, self.store = rank, world, store
        self.calls = []

    def all_gather(self, *args):
        self.calls.append(args)


@pytest.fixture(autouse=True)
def no_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def test_eight_rank_frontiers_have_bounded_keys_and_restore_original_hot_path():
    store = Store()
    comms = [Comm(rank, 8, store) for rank in range(8)]
    originals = [comm.all_gather for comm in comms]

    def run(comm):
        scope = Construction(comm, 7)
        with scope:
            for step in range(40):
                comm.all_gather(step)
        return scope.outcomes({"step": scope.step, "error": None})

    with ThreadPoolExecutor(8) as workers:
        results = list(workers.map(run, comms))
    assert all(result == [{"step": 40, "error": None}] * 8 for result in results)
    assert all(comm.calls == [(step,) for step in range(40)] for comm in comms)
    assert all(comm.all_gather == original for comm, original in zip(comms, originals))
    assert all("_profile_construction" not in vars(comm) for comm in comms)
    assert store.data == {}
    assert store.maximum <= 8 * 4 + 2


def test_local_load_refusal_reaches_peer_before_any_collective_is_issued():
    store = Store()
    comms = [Comm(rank, 2, store) for rank in range(2)]

    def run(comm):
        scope = Construction(comm, 3)
        with scope:
            if comm.rank == 0:
                scope.fail(MemoryError("weight allocation refused"))
            else:
                with pytest.raises(ConstructionRefused):
                    comm.all_gather("unissued")
        return scope.outcomes({"step": scope.step, "error": "refused"})

    with ThreadPoolExecutor(2) as workers:
        list(workers.map(run, comms))
    assert all(not comm.calls for comm in comms)
    assert store.data == {}


def test_go_is_binding_when_fast_peer_refuses_a_later_step():
    store = Store()
    comms = [Comm(rank, 2, store) for rank in range(2)]
    refused = threading.Event()

    class Delayed(Construction):
        def frontier(self, method):
            super().frontier(method)
            assert refused.wait(5)

    def run(comm):
        scope = (Delayed if comm.rank == 0 else Construction)(comm, 4)
        with scope:
            comm.all_gather("obligated")
            if comm.rank == 1:
                scope.fail(MemoryError("following allocation"))
                refused.set()
        return scope.outcomes({"step": scope.step, "error": "refused"})

    with ThreadPoolExecutor(2) as workers:
        list(workers.map(run, comms))
    assert [comm.calls for comm in comms] == [[("obligated",)], [("obligated",)]]


def test_issued_collective_failure_is_fatal_and_not_a_load_refusal(monkeypatch):
    notices = []
    monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", lambda: notices.append("fatal"))
    comm = Comm(0, 1, Store())

    def fail(*args):
        raise RuntimeError("transport ambiguity")

    comm.all_gather = fail
    scope = Construction(comm, 8)
    with pytest.raises(RuntimeError, match="transport"):
        with scope:
            comm.all_gather()
    assert scope.fatal and notices == ["fatal"]
    assert comm.all_gather is fail


def test_collective_order_mismatch_fails_without_issuing():
    store = Store()
    scopes = [Construction(Comm(rank, 2, store), 9) for rank in range(2)]

    def run(scope):
        try:
            scope.frontier("A" if scope.comm.rank == 0 else "B")
        except ConstructionFatal:
            scope.fail(ValueError("order"))
            return "fatal"
        except ConstructionRefused:
            return "refused"

    with ThreadPoolExecutor(2) as workers:
        assert set(workers.map(run, scopes)) == {"fatal", "refused"}
    assert all(not scope.comm.calls for scope in scopes)


def test_graph_launch_has_readiness_but_original_graph_remains_owned():
    store = Store()
    comm = Comm(0, 1, store)
    events = []
    graph = SimpleNamespace(replay=lambda: events.append("launch"))
    scope = Construction(comm, 10)
    with scope:
        replay(graph, comm)
    assert scope.step == 1
    replay(graph, comm)
    assert events == ["launch", "launch"]
    scope.outcomes({"step": 1, "error": None})
    assert store.data == {}


@pytest.mark.parametrize("failure,fatal", [(ValueError("agreed refusal"), False), (MemoryError("unvoted failure"), True)])
def test_existing_protocol_must_own_its_refusals_after_the_entry_frontier(monkeypatch, failure, fatal):
    notices = []
    monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", lambda: notices.append("fatal"))
    scope = Construction(Comm(0, 1, Store()), 12)

    def operation():
        raise failure

    with scope:
        with pytest.raises(type(failure)) as raised:
            scope.protocol("owned-protocol", operation, refusal=ValueError)
    assert raised.value is failure
    assert scope.fatal is fatal and bool(notices) is fatal
    if not fatal:
        scope.outcomes({"step": scope.step, "error": "agreed refusal"})


def test_uniform_native_subgroup_capture_replay_bindings_restore_direct_calls():
    store = Store()
    scopes = [Construction(Comm(rank, 8, store), 15) for rank in range(8)]

    def run(scope):
        events = []

        class HeadGroup:
            def all_gather(self, value):
                events.append(("group", scope.comm.rank // 4, value))

        head = HeadGroup()

        class Manager:
            def capture(self, value):
                head.all_gather("stats-" + value)
                head.all_gather("pv-" + value)

        manager = Manager()
        original_head, original_capture = head.all_gather, manager.capture
        def replay_call(value):
            events.append(("replay", value))
        replays = {"bucket": replay_call}
        with scope:
            scope.bind_submit(head, "all_gather", "owner/head-gather")
            scope.bind_protocol(manager, "capture", "owner/capture", refusal=())
            scope.bind_mapping_submit(replays, "bucket", "owner/replay/bucket")
            head.all_gather("query")
            manager.capture("bucket")
            replays["bucket"]("bucket")
            assert scope.step == 5
        assert head.all_gather == original_head and "all_gather" not in vars(head)
        assert manager.capture == original_capture and "capture" not in vars(manager)
        assert replays["bucket"] is replay_call and scope.bindings == []
        head.all_gather("live")
        assert scope.step == 5  # No construction wrapper/branch remains in the serving call.
        scope.outcomes({"step": 5, "error": None})
        return events

    with ThreadPoolExecutor(8) as workers:
        results = list(workers.map(run, scopes))
    for rank, events in enumerate(results):
        assert events == [("group", rank // 4, "query"), ("group", rank // 4, "stats-bucket"),
                          ("group", rank // 4, "pv-bucket"), ("replay", "bucket"),
                          ("group", rank // 4, "live")]
    assert store.data == {}


def test_native_close_retires_bindings_before_reset_without_resurrecting_graphs():
    import gc
    import weakref

    class Graph:
        def replay(self):
            pass

    graph = Graph()
    observed = weakref.ref(graph)
    replays = {"bucket": graph.replay}
    scope = Construction(Comm(0, 1, Store()), 16)
    with scope:
        scope.bind_mapping_submit(replays, "bucket", "owner/replay")
        scope.retire_bindings(replays)
        assert replays["bucket"].__self__ is graph
        replays.clear()  # Native close/reset happens only after the construction borrow retires.
        del graph
        gc.collect()
        assert observed() is None and scope.bindings == []
    assert replays == {}
    scope.outcomes({"step": scope.step, "error": None})


def test_changed_replay_map_fails_closed_without_reinstalling_a_removed_entry():
    scope = Construction(Comm(0, 1, Store()), 17)
    replays = {"bucket": lambda: None}
    with pytest.raises(RuntimeError, match="binding changed"):
        with scope:
            scope.bind_mapping_submit(replays, "bucket", "owner/replay")
            replays.clear()  # Deliberately violate the required pre-close retire hook.
    assert scope.fatal and replays == {} and scope.bindings


def test_native_callable_slot_is_restored_even_when_derived_owner_has_a_dictionary():
    class Slotted:
        __slots__ = ("submit",)

    class Owner(Slotted):
        pass

    owner = Owner()
    def original(value):
        return value + 1
    owner.submit = original
    scope = Construction(Comm(0, 1, Store()), 18)
    with scope:
        scope.bind_submit(owner, "submit", "owner/slot")
        assert owner.submit(3) == 4
    assert owner.submit is original and scope.bindings == []
    scope.outcomes({"step": scope.step, "error": None})
