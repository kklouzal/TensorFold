"""Independent controls for the concrete profile D13 failure/binding findings."""

from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
from types import SimpleNamespace
import weakref

import pytest

from tensorfold.cuda import profile_binding
from tensorfold.cuda.profile_binding import ProfileGate, ProfileUnavailable
from tensorfold.families.qwen4_exp.cuda import serving_profiles, serving_runtime
from tensorfold.server import runtime
from test_serving_profile_manifest import profile_files
from test_serving_profile_transitions import LoadDouble, pipeline, wait_until


@pytest.fixture
def review_profile(tmp_path):
    return profile_files(tmp_path)


@pytest.mark.parametrize("consumer", ["manifest", "producer"])
def test_growing_artifact_cannot_extend_hash_work_beyond_declared_extent(tmp_path, monkeypatch, consumer):
    size = (2 << 20) + 7
    path = tmp_path / "growing.bin"
    path.write_bytes(b"x" * size)
    calls = []

    @contextmanager
    def growing(filename):
        with path.open("rb") as real:
            class Stream:
                def fileno(self):
                    return real.fileno()

                def read(self, count):
                    calls.append(count)
                    assert len(calls) <= 3, "hashing followed a growing file beyond its prepared extent"
                    return b"x" * count  # Appending keeps every requested byte available.
            yield Stream()

    monkeypatch.setattr(serving_profiles, "_regular_stream", growing)
    with pytest.raises(ValueError, match="grew"):
        if consumer == "manifest":
            serving_profiles._artifact({"path": str(path), "size": size,
                "sha256": hashlib.sha256(b"x" * size).hexdigest()}, tmp_path, verify_content=True)
        else:
            from tools import prepare_serving_profile as producer

            monkeypatch.setattr(producer, "_required_files", lambda *_: {str(path)})
            producer.create(tmp_path / "B.json", identifier="B", model=tmp_path, options={}, api={})
    assert calls == [2 << 20, 7, 1] and sum(calls) == size + 1


def test_manifest_producer_hashes_content_once_and_schema_preflight_does_not_rehash(review_profile, monkeypatch):
    from tools import prepare_serving_profile as producer

    root, _, raw = review_profile
    original = serving_profiles.artifact_identity
    reads = []

    def observe(path, **kwargs):
        reads.append((str(path), kwargs.get("verify_content", True)))
        return original(path, **kwargs)

    monkeypatch.setattr(producer, "artifact_identity", observe)
    monkeypatch.setattr(serving_profiles, "artifact_identity", observe)
    producer.create(root / "B.json", identifier="B", model=root / "target", options=raw["options"], api={})
    count = len(raw["artifacts"])
    assert len([item for item in reads if item[1]]) == count
    assert len([item for item in reads if not item[1]]) == count


def test_producer_unlink_failure_preserves_preflight_primary(review_profile, monkeypatch):
    from tools import prepare_serving_profile as producer

    root, _, raw = review_profile
    primary, cleanup = ValueError("preflight failed"), OSError("unlink failed")
    original = Path.unlink

    def refuse(*args, **kwargs):
        raise primary

    def unlink(path, *args, **kwargs):
        if path.name.startswith(".serving-"):
            raise cleanup
        return original(path, *args, **kwargs)

    monkeypatch.setattr(producer, "prepare", refuse)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(ValueError) as caught:
        producer.create(root / "B.json", identifier="B", model=root / "target", options=raw["options"], api={})
    assert caught.value is primary and caught.value.__cause__ is cleanup
    assert not (root / "B.json").exists()


def test_producer_directory_close_failure_preserves_fsync_primary(review_profile, monkeypatch):
    from tools import prepare_serving_profile as producer

    root, _, raw = review_profile
    primary, cleanup = OSError("directory fsync failed"), OSError("directory close failed")
    original_sync, original_close = os.fsync, os.close

    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise primary
        return original_sync(fd)

    def close(fd):
        directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        original_close(fd)
        if directory:
            raise cleanup

    monkeypatch.setattr(producer.os, "fsync", sync)
    monkeypatch.setattr(producer.os, "close", close)
    with pytest.raises(OSError) as caught:
        producer.create(root / "B.json", identifier="B", model=root / "target", options=raw["options"], api={})
    assert caught.value is primary and caught.value.__cause__ is cleanup
    assert (root / "B.json").is_file()  # Publication preceded the explicitly failed durability acknowledgment.


@pytest.mark.parametrize("kind", [KeyboardInterrupt, MemoryError])
def test_before_release_failure_keeps_standalone_close_retryable(monkeypatch, kind):
    original = profile_binding._release_hold

    def interrupt(*args):
        raise kind("before native release")

    monkeypatch.setattr(profile_binding, "_release_hold", interrupt)
    gate = ProfileGate()
    lease = gate.acquire()
    gate.drain()
    with pytest.raises(kind):
        lease.close()
    assert lease.scope.locked() and lease._finalizer.alive
    assert not gate.record()["accepting"]  # Repeated release failure is explicit fatal containment.
    monkeypatch.setattr(profile_binding, "_release_hold", original)
    lease.close()
    assert not lease.scope.locked() and gate.borrowers == 0 and not lease._finalizer.alive


def test_abandoned_holder_is_not_strongly_retained_by_its_journal_or_finalizer():
    gate = ProfileGate()
    held = gate.acquire()
    native = weakref.ref(held.scope)
    owner = weakref.ref(held)
    held._finalizer.detach()  # Independent missing-callback oracle, not the close implementation.
    del held
    gc.collect()
    assert owner() is None and native() is None and gate.borrowers == 0


def test_transient_wake_allocation_failure_retries_and_preserves_the_original_error():
    calls = []
    primary = MemoryError("before wake publication")

    def wake():
        calls.append("wake")
        if len(calls) == 1:
            raise primary

    gate = ProfileGate(wake=wake)
    held = gate.acquire()
    gate.drain()
    with pytest.raises(MemoryError) as caught:
        held.close()
    assert caught.value is primary and calls == ["wake", "wake"] and gate.borrowers == 0
    assert not gate._poisoned.locked()
    held.close()


def test_repeated_wake_failure_poison_is_not_a_silent_idle_drain():
    calls, failures = [], []

    def wake():
        calls.append("wake")
        raise MemoryError("wake allocation failed")

    gate = ProfileGate(wake=wake, fatal=lambda error: failures.append(error))
    held = gate.acquire()
    gate.drain()
    with pytest.raises(MemoryError):
        held.close()
    assert calls == ["wake", "wake"] and len(failures) == 1
    assert not gate.record()["available"] and gate.borrowers == 0
    with pytest.raises(ProfileUnavailable):
        gate.acquire()
    gate.wake = None
    held.close()


def test_prepared_close_can_retry_before_native_release(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch, world=1) as (engines, app):
        original = profile_binding._release_hold
        notices = []
        monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", lambda: notices.append("fatal"))
        prepared = app.prepare({"prompt": "w1"}, False)

        def fail(*args):
            raise KeyboardInterrupt("before native release")

        monkeypatch.setattr(profile_binding, "_release_hold", fail)
        with pytest.raises(KeyboardInterrupt):
            prepared.close()
        assert prepared._profile_finalizer.alive and app.pending_requests == 0
        monkeypatch.setattr(profile_binding, "_release_hold", original)
        prepared.close()
        assert not prepared._profile_finalizer.alive
        assert engines[0].serving_profiles.gate.borrowers == 0 and notices


def test_poison_wrapper_allocation_cannot_leave_model_admission_open(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch, world=1) as (engines, app):
        owner = engines[0].serving_profiles
        original, allocation = ValueError("unresolved ownership"), MemoryError("wrapper allocation")
        original._tensorfold_retained_owners = [object()]
        notices = []
        monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", lambda: notices.append("fatal"))

        def no_wrapper(*args):
            raise allocation

        monkeypatch.setattr(serving_runtime, "ConstructionFatal", no_wrapper)
        with pytest.raises(ValueError) as caught:
            owner._poison(original)
        assert caught.value is original and caught.value.__cause__ is allocation
        assert owner.failure is original and notices == ["fatal"] and engines[0]._abort_comm
        assert not owner.gate.record()["accepting"] and not owner.gate.record()["available"]
        with pytest.raises(ProfileUnavailable):
            app.prepare({"prompt": "w2"}, False)


def test_partial_sampling_uses_startup_defaults_on_every_profile(tmp_path, monkeypatch):
    defaults = {"temperature": 0.7, "top_k": 11, "top_p": 0.8, "min_p": 0.02}
    api = {"A": {"sampling": {"temperature": 0.3}}, "B": {"sampling": {"top_p": 0.9}}}
    with pipeline(tmp_path, monkeypatch, world=1, profile_api=api, sampling=defaults) as (engines, app):
        owner = engines[0].serving_profiles
        for name, epoch, temperature, top_p in [("B", 1, 0.7, 0.9), ("A", 2, 0.3, 0.8)]:
            runtime.request(app, {"serving_profile": name+".json"})
            wait_until(lambda: owner.gate.epoch == epoch)
            prepared = app.prepare({"prompt": "w1", "seed": 123}, False)
            try:
                assert prepared.sampling.temperature == temperature
                assert prepared.sampling.top_p == top_p and prepared.sampling.top_k == 11
                assert prepared.sampling.min_p == 0.02
            finally:
                prepared.close()


def test_tool_format_cache_tracks_template_and_tokenizer_across_a_b_a(tmp_path, monkeypatch):
    from tokenizers import Tokenizer

    def configure(directories):
        for name, directory in directories.items():
            tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
            tok.add_tokens(["<tool_call>", "<|tool_call>"])
            tok.save(str(directory / "tokenizer.json"))
            body = ('<tool_call>{"name":"{{ m.tool_calls[0].function.name }}","arguments":{}}</tool_call>'
                    if name == "A" else '<|tool_call>fn={{ m.tool_calls[0].function.name }}<tool_call|>')
            template = ('{% for m in messages %}{% if m.tool_calls|default([]) %}' + body +
                        '{% else %}{{ m.content }} {% endif %}{% endfor %}header')
            (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))

    with pipeline(tmp_path, monkeypatch, world=1, configure=configure) as (engines, app):
        owner, turns = engines[0].serving_profiles, app.turns
        tools = [{"type": "function", "function": {"name": "actual_fn"}}]
        expected = {"A": ("<tool_call>", '{"name":"', '"'), "B": ("<|tool_call>", "fn=", "<")}
        app._call_gate([0], tools)
        assert app._form == expected["A"]
        for name, epoch in [("B", 1), ("A", 2)]:
            runtime.request(app, {"serving_profile": name+".json"})
            wait_until(lambda: owner.gate.epoch == epoch)
            assert "_form" not in app.__dict__
            gate = app._call_gate([0], tools)
            assert app._form == expected[name] and app.turns is turns
            assert gate.opener == app.tok.token_to_id(expected[name][0])


def test_failed_constructor_frames_survive_fence_and_resource_retirement(tmp_path, monkeypatch):
    class ConstructorOnly:
        pass

    with pipeline(tmp_path, monkeypatch, world=1) as (engines, app):
        engine, alive = engines[0], []
        original_close = LoadDouble._close_resources

        def initialize(owner, *args, **kwargs):
            frame = ConstructorOnly()
            owner.constructor_only = weakref.ref(frame)
            raise MemoryError("partial constructor")

        def close(owner, **kwargs):
            ref = getattr(owner, "constructor_only", None)
            if ref is not None:
                alive.append(ref() is not None)
            return original_close(owner, **kwargs)

        monkeypatch.setattr(LoadDouble, "_initialize", initialize)
        monkeypatch.setattr(LoadDouble, "_close_resources", close)
        runtime.request(app, {"serving_profile": "B.json"})
        wait_until(lambda: engine.serving_profiles.record()["stage"] == "unavailable")
        assert alive == [True] and engine.constructor_only() is None


def test_single_rank_successful_switch_restores_the_new_arrival_hook_on_close(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch, world=1) as (engines, app):
        engine = engines[0]
        runtime.request(app, {"serving_profile": "B.json"})
        wait_until(lambda: engine.serving_profiles.gate.epoch == 1)
        scheduler, decoder = engine.scheduler, engine.multi
        original = scheduler._arrival_original
        assert scheduler._arrival_applied and decoder.arrived == scheduler._arrival_hook
        scheduler.close()
        assert decoder.arrived == original and not scheduler._arrival_applied


def test_control_predicate_allocation_failure_closes_the_gate_and_notifies_containment():
    class Engine:
        pass

    error = MemoryError("reader inspection failed")

    class Gate(ProfileGate):
        @property
        def borrowers(self):
            raise error

    engine = Engine()
    notices = []
    engine.tp_decoder = SimpleNamespace(_poison_control=lambda cause: notices.append(cause))
    engine.multi = None
    engine._abort_comm = False
    owner = object.__new__(serving_runtime.ServingProfiles)
    owner._engine = weakref.ref(engine)
    owner.lock = threading.RLock()
    owner.gate = Gate()
    owner.failure = None
    owner.sequence = 0
    owner.pending = SimpleNamespace()
    owner.candidate = None
    owner.status = {"stage": "draining"}
    scheduler = SimpleNamespace(boxes={}, held=None)
    with pytest.raises(serving_runtime.ConstructionFatal):
        owner.boundary(scheduler)
    assert owner.failure.__cause__ is error and notices == [error] and engine._abort_comm
    assert not owner.gate.record()["available"] and owner.status["stage"] == "fatal"
