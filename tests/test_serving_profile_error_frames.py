"""Retained caller errors never become invisible owners of a replaced model."""

import threading
from types import SimpleNamespace

import pytest

from tensorfold.cuda.streams import mark_error_state_retired
from tensorfold.families.qwen4_exp.cuda.profile_construction import ConstructionFatal
from tensorfold.engine.grammar import GrammarError
from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController
from tensorfold.server import runtime
from tensorfold.server.cancellation import RequestCancelled
from test_flashnext_tp_scheduler import Decoder
from test_serving_profile_transitions import pipeline, wait_until


@pytest.mark.parametrize("world", [1, 8])
@pytest.mark.parametrize("kind", [GrammarError, ValueError, RequestCancelled])
def test_caller_keeps_retired_admission_error_tree_while_a_b_a_releases_old_model(
        tmp_path, monkeypatch, world, kind):
    original = Decoder.admit

    def refuse(self, stream):
        if stream.prompt == [999]:
            detail = ExceptionGroup("validation causes", [ValueError("detail one"), RuntimeError("detail two")])
            error = mark_error_state_retired(kind("request refused before state mutation"))
            raise error from detail
        return original(self, stream)

    monkeypatch.setattr(Decoder, "admit", refuse)
    with pipeline(tmp_path, monkeypatch, world=world) as (engines, app):
        error = None
        try:
            engines[0].generate([999], 1, None, lambda _: False)
        except kind as caught:
            error = caught
        assert error is not None and isinstance(error.__cause__, ExceptionGroup)
        assert [str(child) for child in error.__cause__.exceptions] == ["detail one", "detail two"]
        assert any("retired worker stack" in note for note in error.__notes__)
        assert all(engine._abort_comm is False for engine in engines)
        for name, epoch in [("B", 1), ("A", 2)]:
            runtime.request(app, {"serving_profile": name+".json"})
            wait_until(lambda: engines[0].serving_profiles.gate.epoch == epoch)
        # The fake loader independently asserts every previous weight weakref
        # is dead before each replacement allocation, while error stays alive.
        assert error is not None and all(ref() is None for ref in engines[0].old_weights)


@pytest.mark.parametrize("world", [1, 8])
@pytest.mark.parametrize("raised", [False, True])
def test_expected_round_and_stream_errors_retire_frames_after_finish_or_drop(tmp_path, monkeypatch, world, raised):
    original = Decoder.round

    def fail(self):
        if self.streams:
            if raised:
                raise GrammarError("round grammar failed")
            done = list(self.streams.values())
            for stream in done:
                stream.st.old_model = self.w  # An actual strong state reference must not survive finish transport.
                stream.error = GrammarError("stream grammar failed")
                stream.done = True
            return done
        return original(self)

    monkeypatch.setattr(Decoder, "round", fail)
    with pipeline(tmp_path, monkeypatch, world=world) as (engines, app):
        error = None
        try:
            engines[0].generate([1], 1, None, lambda _: False)
        except GrammarError as caught:
            error = caught
        assert error is not None and any("retired worker stack" in note for note in error.__notes__)
        assert all(engine._abort_comm is False for engine in engines)
        runtime.request(app, {"serving_profile": "B.json"})
        wait_until(lambda: engines[0].serving_profiles.gate.epoch == 1)
        assert all(ref() is None for ref in engines[0].old_weights)


def test_runtime_projection_is_a_reader_and_draining_status_does_not_extend_it(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch, world=1) as (engines, app):
        engine = engines[0]
        engine.multi.controller = AdaptiveController(slots=2, row_budget=8, proposal_limit=1, prefill_tokens=16)
        entered, release = threading.Event(), threading.Event()
        readings = []

        def placement():
            entered.set()
            assert release.wait(5)
            return {"profile": "A"}

        engine.multi.history_record = placement
        actor = threading.Thread(target=lambda: readings.append(runtime.snapshot(app)))
        actor.start()
        try:
            assert entered.wait(5)
            runtime.request(app, {"serving_profile": "B.json"})
            wait_until(lambda: engine.serving_profiles.record()["stage"] == "draining")
            assert not engine.loads
            status = runtime.snapshot(app)
            assert status["serving_profile"]["stage"] == "draining" and status["capacity_plan"] is None
        finally:
            release.set()
            actor.join(7)
        assert not actor.is_alive() and readings[0]["placement"] == {"profile": "A"}
        wait_until(lambda: engine.serving_profiles.gate.epoch == 1)
        assert all(ref() is None for ref in engine.old_weights)


def test_shared_exception_retires_every_supplied_state_and_renewed_traceback(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch, world=1) as (engines, app):
        owner, calls = engines[0].serving_profiles, []
        engines[0].w.dflash2 = SimpleNamespace(
            retire_request_after_fence=lambda states, **kwargs: calls.append((states, kwargs["fenced"])))
        error = GrammarError("shared error")
        a, b, c = object(), object(), object()
        # One finish operation has cleaned both states, even though their error
        # identity is shared. Neither call may reuse an exception-local shortcut.
        owner.error_outcome(error, states=(a,), state_retired=True)
        owner.error_outcome(error, states=(b,), state_retired=True)
        try:
            raise mark_error_state_retired(error)
        except GrammarError as current:
            owner.error_outcome(current, states=(c,))
        assert calls == [((a,), True), ((b,), True), ((c,), True)]
        assert error.__traceback__ is None
        assert "_tensorfold_request_state_retired" not in error.__dict__


def test_previous_cleanup_certificate_cannot_authorize_a_new_unclean_operation(tmp_path, monkeypatch):
    with pipeline(tmp_path, monkeypatch, world=1) as (engines, app):
        owner, calls, notices = engines[0].serving_profiles, [], []
        engines[0].w.dflash2 = SimpleNamespace(
            retire_request_after_fence=lambda states, **kwargs: calls.append(states))
        monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", lambda: notices.append("fatal"))
        error = mark_error_state_retired(GrammarError("one instance reused"))
        owner.error_outcome(error)
        with pytest.raises(ConstructionFatal):
            try:
                raise error  # No decoder cleanup occurred in this new operation.
            except GrammarError as current:
                owner.error_outcome(current)
        assert calls == [()] and notices == ["fatal"]
        assert error.__traceback__ is not None and not owner.gate.record()["available"]


@pytest.mark.parametrize("world", [1, 8])
def test_reused_error_preserves_each_callers_fresh_retirement_signal(tmp_path, monkeypatch, world):
    # Production TP ranks are separate processes: reuse is local to each rank,
    # never one Python exception concurrently shared by simulated peer threads.
    errors = [GrammarError("reused admission error") for _ in range(world)]
    error = errors[0]
    original = Decoder.admit

    def admit(self, stream):
        if stream.prompt == [999]:
            raise mark_error_state_retired(errors[self.rank])
        return original(self, stream)

    monkeypatch.setattr(Decoder, "admit", admit)
    with pipeline(tmp_path, monkeypatch, world=world) as (engines, app):
        for _ in range(3):
            with pytest.raises(GrammarError) as caught:
                engines[0].generate([999], 1, None, lambda _: False)
            assert caught.value is error
            assert all(not engine._abort_comm for engine in engines)
            notes = [note for note in error.__notes__ if note.startswith("retired worker stack:")]
            assert len(notes) == 1  # Renewed stacks replace only our old evidence, never recursively embed it.
        runtime.request(app, {"serving_profile": "B.json"})
        wait_until(lambda: engines[0].serving_profiles.gate.epoch == 1)
        assert all(ref() is None for ref in engines[0].old_weights)


@pytest.mark.parametrize("after_insert", [False, True])
def test_event_registration_failure_rolls_back_borrow_and_both_journals(tmp_path, monkeypatch, after_insert):
    with pipeline(tmp_path, monkeypatch, world=1) as (engines, app):
        primary = MemoryError("event journal insertion")

        class FailedInsertion(dict):
            def __setitem__(self, key, value):
                if after_insert:
                    super().__setitem__(key, value)
                raise primary

        scheduler = engines[0].scheduler
        scheduler._error_retirements = FailedInsertion()
        with pytest.raises(MemoryError) as caught:
            engines[0].generate([1], 1, None, lambda _: False)
        assert caught.value is primary
        assert app.pending_requests == 0 and not scheduler._request_boxes and not scheduler._error_retirements
        assert scheduler.waiting.qsize() == 0 and not scheduler._closing
        assert engines[0]._abort_comm is False and engines[0].serving_profiles.failure is None
