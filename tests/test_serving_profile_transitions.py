"""Real Engine entrypoint/Scheduler/TP/App lifecycle with fake model allocations.

The loader/resource doubles deliberately do not establish GGUF/CUDA loading,
quality, resident-byte recovery or V100 A-B-A qualification.
"""

from contextlib import contextmanager
import gc
import http.client
import json
import queue
from pathlib import Path
import threading
import time
from types import MappingProxyType, SimpleNamespace
import weakref

import pytest
import torch

from cuda_http_fixtures import model_directory
from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.cuda.profile_binding import ProfileGate, ProfileUnavailable
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.server import App
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.serving_profiles import ServingProfile
from tensorfold.families.qwen4_exp.cuda.serving_runtime import ServingProfiles
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder
from tensorfold.server import runtime
from tensorfold.server import responses
from tensorfold.server.errors import CapacityError
from test_cuda_admission import http_server
from test_flashnext_tp_scheduler import Decoder
from test_serving_profile_construction import Comm, Store


class PreparedDouble(ServingProfile):
    def check_sources(self):
        pass  # Actual file/schema coverage is independent in test_serving_profile_manifest.py.


class Weight:
    def __init__(self):
        self.device = "cpu"
        self.meta = {}


class LoadDouble(FlashNextEngine):
    """Replace only GPU loader/retirement; exercise the real orchestration."""

    def _initialize(self, model_dir, *, max_len=65, depth=0, kv_dtype="int8", **options):
        assert self.multi is self.w is None
        assert self.tp_decoder is None or self.tp_decoder.decoder is None
        assert not self.serving_profiles.gate.available
        assert all(ref() is None for ref in self.old_weights)
        self.loads.append((Path(model_dir).name, max_len, depth, kv_dtype, threading.get_ident()))
        if self.rank == 3 and self.refuse:
            self.refuse = False
            raise torch.cuda.OutOfMemoryError("rank3 refused weight allocation before its next collective")
        if self.comm is not None:
            self.comm.all_gather("loader-cohort-shape")
        self.max_len, self.depth = max_len + depth + 1, depth
        self.model_dir = Path(model_dir)
        self.chat_template = ChatTemplate(model_dir)
        self.w = Weight()
        self.w.comm = self.comm
        self.multi = Decoder(self.rank, self.tp, None if self.comm is None else self.comm.store)
        self.multi.w = self.w
        self.multi.full_resident = True
        self.streams = 2
        self.capacity_plan = {"profile": Path(model_dir).name, "kv_dtype": kv_dtype, "context": max_len}
        if self.comm is not None:
            self.comm.all_gather("allocated-warm-ready")

    def _close_resources(self, *, preserve_transport=False):
        assert preserve_transport
        self.retirements.append(None if self.w is None else weakref.ref(self.w))
        if self.w is not None:
            self.old_weights.append(weakref.ref(self.w))

    def _drain_startup_reads(self):
        pass


def wait_until(check):
    deadline = time.monotonic() + 7
    while not check():
        assert time.monotonic() < deadline
        time.sleep(0.005)


@contextmanager
def pipeline(tmp_path, monkeypatch, world=8, pending=8, configure=None, profile_api=None, sampling=None):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    directories = {}
    for name in ("A", "B"):
        directory = tmp_path / name
        directory.mkdir()
        directories[name] = model_directory(directory)
    if configure is not None:
        configure(directories)
    store = Store()
    engines = []
    old_weights = []
    for rank in range(world):
        engine = object.__new__(LoadDouble)
        engine.rank, engine.tp = rank, world
        engine._abort_comm = engine._closing = engine._closed = False
        engine._receiving = False
        engine._lifecycle = threading.Condition()
        engine._calls = {}
        engine._profile_build = None
        engine.serving_profiles = None
        engine.model_dir = directories["A"]
        engine.max_len, engine.depth = 1025, 0
        engine.execution_provider = "sm70"
        engine.adaptive_sessions, engine.indexer_history = True, "gpu"
        engine.eos, engine.concurrent, engine.streams = (0,), True, 2
        engine.vision = engine.e = engine.serial = None
        engine.chat_template = ChatTemplate(engine.model_dir)
        engine.comm = Comm(rank, world, store) if world > 1 else None
        engine.w = Weight()
        engine.w.comm = engine.comm
        engine.multi = Decoder(rank, world, store)
        engine.multi.w = engine.w
        engine.multi.full_resident = True
        engine.multi.controller = None
        engine.old_weights = old_weights
        old_weights.append(weakref.ref(engine.w))
        engine.capacity_plan = {"profile": "A"}
        engine.loads, engine.retirements, engine.refuse = [], [], False
        engine.tp_decoder = TensorParallelDecoder(engine.multi, engine) if world > 1 else None
        engine.scheduler = Scheduler(engine.tp_decoder if world > 1 else engine.multi, max_streams=2) if rank == 0 else None
        engine.serving_profiles = ServingProfiles(engine, tmp_path, {"rank": rank, "tp": world})
        engines.append(engine)
    del engine

    def prepare(_root, name, *, fixed_options, **kwargs):
        name = name.removesuffix(".json")
        options = {**fixed_options, "max_len": 80 if name == "B" else 1024,
                   "depth": 2 if name == "B" else 0, "kv_dtype": "fp16" if name == "B" else "int8"}
        return PreparedDouble(name+".json", name, "a"*64, name*64, directories[name],
                              MappingProxyType(options), MappingProxyType(
                                  ({"served": "model-B"} if name == "B" else {}) if profile_api is None
                                  else profile_api.get(name, {})), ())

    monkeypatch.setattr("tensorfold.families.qwen4_exp.cuda.serving_runtime.prepare", prepare)
    errors = []

    def follow(engine):
        try:
            while True:
                body = engine.tp_decoder.receive()
                if body is None:
                    return
                engine.tp_decoder.apply(body)
        except BaseException as error:
            errors.append(error)

    followers = [threading.Thread(target=follow, args=(engine,), daemon=True) for engine in engines[1:]]
    for actor in followers:
        actor.start()
    app = App(engines[0], directories["A"], "model-A", max_tokens=3, max_pending_requests=pending, sampling=sampling)
    engines[0].scheduler.start()
    try:
        yield engines, app
    finally:
        engines[0].scheduler.close()
        if engines[0].tp_decoder is not None:
            engines[0].tp_decoder.shutdown()
        for actor in followers:
            actor.join(8)
            assert not actor.is_alive()
        assert not errors
        assert store.data == {}


@pytest.mark.parametrize("world", [1, 8])
def test_prepared_request_drains_then_switches_a_b_a_on_same_worker(tmp_path, monkeypatch, world):
    with pipeline(tmp_path, monkeypatch, world) as (engines, app):
        owner = engines[0].serving_profiles
        app.vision = engines[0].vision = Weight()
        engines[0].old_weights.append(weakref.ref(app.vision))
        app.grammars = object()
        engines[0].grammars = object()
        worker = engines[0].scheduler.thread
        comms = [engine.comm for engine in engines]
        adapters = [engine.tp_decoder for engine in engines]
        prepared = app.prepare({"prompt": "w1", "max_tokens": 2}, False)
        assert owner.gate.borrowers == 1
        runtime.request(app, {"serving_profile": "B.json"})
        wait_until(lambda: owner.record()["stage"] == "draining")
        assert not engines[0].loads
        with pytest.raises(ProfileUnavailable):
            app.prepare({"prompt": "w2"}, False)
        with pytest.raises(ProfileUnavailable, match="lease"):
            engines[0].scheduler.submit([1], 1, None, False, lambda _: False, profile_epoch=0)
        result = app.run({"prompt": "w1", "max_tokens": 2}, False, lambda event: True, prepared=prepared)
        assert result["completion_tokens"] == 2
        wait_until(lambda: owner.record()["stage"] == "ready" and owner.gate.epoch == 1)
        assert app.served == "model-B" and app.context_window == 80
        assert app.vision is None and not hasattr(app, "grammars") and not hasattr(engines[0], "grammars")
        assert all(engine.depth == 2 and engine.capacity_plan["kv_dtype"] == "fp16" for engine in engines)
        assert all(engine._abort_comm is False for engine in engines)
        assert all(engine.comm is comm for engine, comm in zip(engines, comms))
        assert all(engine.tp_decoder is adapter for engine, adapter in zip(engines, adapters))
        assert engines[0].scheduler.thread is worker and worker.is_alive()
        runtime.request(app, {"serving_profile": "A.json"})
        wait_until(lambda: owner.gate.epoch == 2)
        assert app.served == "model-A" and app.context_window == 1024
        assert all(engine.depth == 0 and engine.capacity_plan["kv_dtype"] == "int8" for engine in engines)
        assert all(row[-1] == worker.ident for row in engines[0].loads)
        if world > 1:
            assert engines[0].tp_decoder.next_wire == 1
        with pytest.raises(ProfileUnavailable):
            app.run({"prompt": "w1"}, False, lambda event: True, prepared=prepared)


def test_rank_load_refusal_releases_old_model_and_retry_stays_in_process(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch) as (engines, app):
        owner = engines[0].serving_profiles
        worker = engines[0].scheduler.thread
        engines[3].refuse = True
        runtime.request(app, {"serving_profile": "B.json"})
        wait_until(lambda: owner.record()["stage"] == "unavailable")
        assert not owner.gate.available
        assert all(engine.multi is engine.w is None for engine in engines)
        assert all(engine.tp_decoder.decoder is None for engine in engines)
        assert runtime.snapshot(app)["capacity_plan"] is None
        from tensorfold.cuda.http import make_handler

        assert make_handler(app).requires_finite_limits  # Unavailability cannot drop endpoint policy.
        assert all(engine._abort_comm is False for engine in engines)
        with pytest.raises(ProfileUnavailable):
            app.tokenize({"prompt": "w1"})
        runtime.request(app, {"serving_profile": "B.json"})
        wait_until(lambda: owner.gate.epoch == 1)
        assert app.served == "model-B"
        assert engines[0].scheduler.thread is worker and worker.is_alive()


def test_unused_prepared_binding_has_explicit_and_gc_retirement(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch) as (engines, app):
        gate = engines[0].serving_profiles.gate
        prepared = app.prepare({"prompt": "w1"}, False)
        assert gate.borrowers == 1
        prepared.close()
        prepared.close()
        assert gate.borrowers == 0
        prepared = app.prepare({"prompt": "w1"}, False)
        ref = weakref.ref(prepared)
        del prepared
        gc.collect()
        assert ref() is None and gate.borrowers == 0


def test_peer_preflight_refusal_preserves_the_live_profile_and_all_borrowers(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch) as (engines, app):
        from tensorfold.families.qwen4_exp.cuda import serving_runtime

        original = serving_runtime.prepare

        def refuse(root, name, **kwargs):
            if kwargs["fixed_options"]["rank"] == 4:
                raise ValueError("rank4 selected metadata differs")
            return original(root, name, **kwargs)

        monkeypatch.setattr(serving_runtime, "prepare", refuse)
        prepared = app.prepare({"prompt": "w1"}, False)
        owner = engines[0].serving_profiles
        runtime.request(app, {"serving_profile": "B.json"})
        wait_until(lambda: owner.record()["pending"] is None)
        assert owner.record()["stage"] == "ready" and owner.record()["error"]
        assert all(engine.w is not None and not engine.retirements and not engine.loads for engine in engines)
        assert all(engine.serving_profiles.gate.available and engine.serving_profiles.gate.accepting for engine in engines)
        assert owner.gate.borrowers == 1
        prepared.close()


def test_last_rank_readiness_is_required_before_any_new_model_admission(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch) as (engines, app):
        owner = engines[0].serving_profiles
        entered, release = threading.Event(), threading.Event()
        original = engines[7].comm.all_gather

        def slow(*args):
            if args == ("allocated-warm-ready",):
                entered.set()
                assert release.wait(5)
            return original(*args)

        engines[7].comm.all_gather = slow
        try:
            runtime.request(app, {"serving_profile": "B.json"})
            assert entered.wait(5)
            assert all(not engine.serving_profiles.gate.available for engine in engines)
            assert runtime.snapshot(app)["capacity_plan"] is None
            with pytest.raises(ProfileUnavailable):
                app.tokenize({"prompt": "w1"})
        finally:
            release.set()
        wait_until(lambda: owner.gate.epoch == 1)
        assert all(engine.serving_profiles.gate.available for engine in engines)


def test_only_existing_binding_can_extend_a_draining_request():
    gate = ProfileGate()
    binding = gate.acquire()
    gate.drain()
    with pytest.raises(ProfileUnavailable):
        gate.acquire()
    with gate.scope(binding) as nested:
        assert nested.epoch == 0 and gate.borrowers == 1
    binding.close()
    gate.unavailable()
    gate.publish(1)
    with pytest.raises(ProfileUnavailable):
        gate.acquire(binding)


def test_interrupted_lexical_retirement_releases_native_claim_and_wakes_pending_drain(monkeypatch):
    wakes = []
    gate = ProfileGate(wake=lambda: wakes.append("wake"))
    original = gate.retire
    calls = []

    def interrupted():
        calls.append("retire")
        if len(calls) == 1:
            raise KeyboardInterrupt("before bookkeeping")
        return original()

    monkeypatch.setattr(gate, "retire", interrupted)
    with pytest.raises(KeyboardInterrupt):
        with gate.scope() as scope:
            gate.drain()
    assert not scope.scope.locked() and gate.borrowers == 0
    assert calls == ["retire", "retire"] and wakes == ["wake"]
    assert gate.current() is None


def test_lost_standalone_holder_is_collected_without_a_stuck_profile_claim():
    gate = ProfileGate()
    held = gate.acquire()
    ref = weakref.ref(held)
    assert gate.borrowers == 1
    del held
    gc.collect()
    assert ref() is None and gate.borrowers == 0


def test_standalone_preparations_share_one_finite_budget_with_running_requests(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch, pending=2) as (engines, app):
        first = app.prepare({"prompt": "w1", "max_tokens": 1}, False)
        second = app.prepare({"prompt": "w2"}, False)
        assert app.pending_requests == 2
        with pytest.raises(CapacityError):
            app.prepare({"prompt": "w3"}, False)
        assert app.pending_requests == 2 and engines[0].serving_profiles.gate.borrowers == 2
        observed = []

        def accept(event):
            observed.append(app.pending_requests)
            return True

        app.run({"prompt": "w1"}, False, accept, prepared=first)
        assert observed and set(observed) == {2}  # The prepared owner transfers; no third admission.
        assert app.pending_requests == 1
        second.close()
        assert app.pending_requests == 0


def test_internal_prepare_uses_admitted_owner_but_callback_prepare_is_independent(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch, pending=1) as (engines, app):
        refused = []

        def accept(event):
            with pytest.raises(CapacityError):
                app.prepare({"prompt": "w3"}, False)
            refused.append(True)
            return True

        result = app.run({"prompt": "w1", "max_tokens": 1}, False, accept)
        assert result["completion_tokens"] == 1 and refused
        assert app.pending_requests == 0 and engines[0].serving_profiles.gate.borrowers == 0


def test_prepared_admission_retires_after_explicit_close_gc_and_prepare_failure(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch, pending=1) as (engines, app):
        prepared = app.prepare({"prompt": "w1"}, False)
        assert app.pending_requests == 1
        del prepared
        gc.collect()
        assert app.pending_requests == 0
        with pytest.raises(ValueError):
            app.prepare({"prompt": None}, False)
        assert app.pending_requests == 0


def test_adopted_preparation_keeps_its_budget_until_a_blocked_consumer_retires(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch, pending=1) as (engines, app):
        prepared = app.prepare({"prompt": "w1", "max_tokens": 1}, False)
        entered, release = threading.Event(), threading.Event()
        outcomes = []

        def accept(event):
            if not entered.is_set():
                entered.set()
                assert release.wait(5)
            return True

        def run():
            try:
                outcomes.append(app.run({"prompt": "w1"}, False, accept, prepared=prepared))
            except BaseException as error:
                outcomes.append(error)

        actor = threading.Thread(target=run)
        actor.start()
        try:
            assert entered.wait(5)
            wait_until(lambda: not engines[0].scheduler.boxes)
            assert app.pending_requests == 1
            with pytest.raises(CapacityError):
                app.prepare({"prompt": "w3"}, False)
        finally:
            release.set()
            actor.join(7)
        assert not actor.is_alive() and len(outcomes) == 1 and isinstance(outcomes[0], dict)
        assert app.pending_requests == 0


def test_stale_queued_epoch_refuses_before_the_new_decoder_reads_tokens(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch) as (engines, app):
        owner = engines[0].serving_profiles
        runtime.request(app, {"serving_profile": "B.json"})
        wait_until(lambda: owner.gate.epoch == 1)
        stale = Stream([123], 1, profile_epoch=0)
        box = queue.Queue()
        engines[0].scheduler.waiting.put((stale, box))
        kind, error = box.get(timeout=5)
        assert kind == "error" and isinstance(error, ProfileUnavailable)
        assert stale.st is None and not any(row[0] == "admit" for row in engines[0].multi.observed)
        assert stale.continued().profile_epoch == 0
        packed = engines[0].tp_decoder._pack(stale)
        assert engines[0].tp_decoder._unpack(packed).profile_epoch == 0


def test_http_model_metadata_and_tokenizer_are_leased_but_stored_json_is_independent(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch) as (engines, app):
        owner = engines[0].serving_profiles
        prepared = app.prepare({"prompt": "w1"}, False)
        responses.store_for(app).put({"id": "resp_saved", "output": [], "model": "model-A"}, [])
        with http_server(app) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                connection.request("POST", "/v1/runtime", json.dumps({"serving_profile": "B.json"}),
                                   {"Content-Type": "application/json"})
                reply = connection.getresponse()
                assert reply.status == 200
                reply.read()
                wait_until(lambda: owner.record()["stage"] == "draining")
                for method, path, body in [("GET", "/v1/models", None),
                                           ("POST", "/tokenize", json.dumps({"prompt": "w1"})),
                                           ("GET", "/stats", None)]:
                    connection.request(method, path, body)
                    reply = connection.getresponse()
                    assert reply.status == 503 and reply.getheader("Retry-After")
                    assert json.loads(reply.read())["error"]["code"] == "model_profile_unavailable"
                connection.request("GET", "/v1/responses/resp_saved")
                reply = connection.getresponse()
                assert reply.status == 200 and json.loads(reply.read())["model"] == "model-A"
                connection.request("GET", "/v1/runtime")
                reply = connection.getresponse()
                assert reply.status == 200
                assert json.loads(reply.read())["serving_profile"]["stage"] == "draining"
                prepared.close()
                wait_until(lambda: owner.gate.epoch == 1)
                connection.request("GET", "/v1/models")
                reply = connection.getresponse()
                assert reply.status == 200
                assert json.loads(reply.read())["data"][0]["id"] == "model-B"
            finally:
                connection.close()


@pytest.mark.parametrize("error_type", [ValueError, MemoryError])
def test_unretired_preparation_failure_is_retained_fatal_and_never_a_retryable_refusal(
        tmp_path, monkeypatch, error_type):
    from tensorfold.families.qwen4_exp.cuda.profile_construction import ConstructionFatal

    with pipeline(tmp_path, monkeypatch, world=1) as (engines, app):
        notices = []
        monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", lambda: notices.append("fatal"))
        unsafe = error_type("prepared file owner failed retirement")
        retained = object()
        unsafe._tensorfold_retained_owners = [retained]

        def fail(*args, **kwargs):
            raise unsafe

        monkeypatch.setattr("tensorfold.families.qwen4_exp.cuda.serving_runtime.prepare", fail)
        with http_server(app) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                connection.request("POST", "/v1/runtime", json.dumps({"serving_profile": "B.json"}),
                                   {"Content-Type": "application/json"})
                reply = connection.getresponse()
                assert reply.status == 500
                assert json.loads(reply.read())["error"]["type"] == "server_error"
            finally:
                connection.close()
        owner = engines[0].serving_profiles
        assert owner.failure.__cause__ is unsafe and unsafe._tensorfold_retained_owners == [retained]
        assert owner.record()["stage"] == "fatal" and not owner.gate.accepting
        assert engines[0]._abort_comm and notices == ["fatal"]
        assert not engines[0].loads and not engines[0].retirements
        with pytest.raises(ConstructionFatal):
            owner.request("A.json")


@pytest.mark.parametrize("phase", ["prepared", "draining"])
def test_unexpected_preflight_or_retirement_control_failure_closes_profile_admission(phase):
    from tensorfold.families.qwen4_exp.cuda.profile_construction import ConstructionFatal

    class EngineOwner:
        pass

    engine = EngineOwner()
    error = OSError("control or retirement did not complete")
    notices = []

    def fail(*args):
        raise error

    engine.tp_decoder = SimpleNamespace(prepare_serving_profile=fail, replace_serving_profile=fail,
                                       _poison_control=lambda original: notices.append(original))
    engine.multi = None
    engine._abort_comm = False
    owner = object.__new__(ServingProfiles)
    owner._engine = weakref.ref(engine)
    owner.lock = threading.RLock()
    owner.gate = ProfileGate()
    owner.failure = None
    owner.sequence = 0
    owner.pending = SimpleNamespace(wire=lambda: {})
    owner.candidate = None
    owner.status = {"stage": phase}
    scheduler = SimpleNamespace(boxes={}, held=None, waiting=queue.Queue())
    with pytest.raises(ConstructionFatal):
        owner.boundary(scheduler)
    assert owner.failure.__cause__ is error and notices == [error]
    assert owner.status["stage"] == "fatal" and not owner.gate.accepting
    assert owner.pending is None and engine._abort_comm
