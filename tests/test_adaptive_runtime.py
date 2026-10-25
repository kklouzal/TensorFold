"""Runtime controls preserve request and derived-cache state and apply only bounded batch settings."""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController
from tensorfold.server import runtime, stats
from tensorfold.server.auth_http import gated
from tensorfold.server.errors import RequestError


def app_for():
    controller = AdaptiveController(slots=40, row_budget=32, proposal_limit=3, prefill_tokens=512)
    shared = SimpleNamespace(pos=128)
    active = SimpleNamespace(st=shared, waiting=False)
    decoder = SimpleNamespace(controller=controller, streams={4: active}, filling=[],
                              kept=[([1], shared, {}, None), ([1, 2], shared, {}, None)])
    scheduler = SimpleNamespace(decoder=decoder, waiting=queue.Queue(), held=None, max_streams=40)
    scheduler.request_runtime = lambda publish: publish()
    engine = SimpleNamespace(scheduler=scheduler, context_window=262144,
                             capacity_plan={"resident_capacity": 40, "evidence_class": "header_plan"})
    return SimpleNamespace(engine=engine, served_name="test-target", max_model_len=262144)


def test_runtime_updates_are_pending_and_do_not_modify_retained_state():
    app = app_for()
    decoder = app.engine.scheduler.decoder
    original_streams, original_kept = decoder.streams, decoder.kept
    before = runtime.snapshot(app)
    assert before["execution"]["timing_scope"] == "host_wall_including_observed_waits"
    assert before["execution"]["device_step_elapsed_s"] is None
    assert before["sessions"]["retained"] == 0
    assert before["sessions"]["active"] == 1
    assert before["sessions"]["derived_prefix_slots"] == 1
    assert before["sessions"]["derived_prefix_entries"] == 2
    update = runtime.request(app, {"row_budget": 16, "proposal_limit": 0, "session_ceiling": 1})
    assert update["execution"]["effective"]["session_ceiling"] == 40
    assert update["execution"]["pending"]["session_ceiling"] == 1
    decoder.controller.boundary()
    assert runtime.snapshot(app)["execution"]["effective"]["proposal_limit"] == 0
    assert decoder.streams is original_streams and decoder.kept is original_kept
    assert app.engine.context_window == 262144
    assert app.engine.capacity_plan["resident_capacity"] == 40


@pytest.mark.parametrize("body", [None, [], {}, {"row_budget": None}, {"proposal_limit": True},
                                  {"row_budget": 33}, {"context_tokens": 8192}, {"kv_format": "q4"},
                                  {"row_budget": 2, "proposal_limit": 3}, {"profile_table": []},
                                  {"profile_table": {}, "row_budget": 8}, {"attention_degree": True},
                                  {"attention_degree": 2, "proposal_limit": 1}])
def test_bad_or_unimplemented_settings_fail_without_publishing(body):
    app = app_for()
    before = runtime.snapshot(app)
    with pytest.raises(RequestError):
        runtime.request(app, body)
    assert runtime.snapshot(app) == before


def test_published_receipts_are_owned_and_capacity_is_not_cached_forever():
    app = app_for()
    first = stats.snapshot(app)
    app.engine.capacity_plan["future_commitment_bytes"] = 42
    second = stats.snapshot(app)
    assert "future_commitment_bytes" not in first["capacity_plan"]
    assert second["capacity_plan"]["future_commitment_bytes"] == 42
    second["runtime"]["execution"]["effective"]["row_budget"] = 1
    assert runtime.snapshot(app)["execution"]["effective"]["row_budget"] == 32


def test_ceiling_decrease_only_blocks_new_admissions_at_coordinator_boundary():
    app = app_for()
    decoder = app.engine.scheduler.decoder
    decoder.live = lambda: 2
    scheduler = Scheduler(decoder, max_streams=40)
    try:
        runtime.request(app, {"session_ceiling": 1})
        assert scheduler._stream_limit() == 40
        scheduler._execution_boundary()
        assert scheduler._stream_limit() == 1
        assert scheduler._admit() == []
        assert decoder.live() == 2
        assert len(decoder.kept) == 2
        runtime.request(app, {"session_ceiling": 40})
        scheduler._execution_boundary()
        assert scheduler._stream_limit() == 40
    finally:
        scheduler.close()


def test_legacy_engine_has_no_adaptive_controls_and_route_is_authenticated():
    app = SimpleNamespace(engine=SimpleNamespace())
    assert runtime.snapshot(app) is None
    with pytest.raises(RequestError, match="no adaptive runtime"):
        runtime.request(app, {"row_budget": 1})
    assert gated(runtime.ROUTE, metrics_open=True)


def test_http_runtime_round_trip_and_rejections():
    from tests.test_cuda_admission import http_server
    from tests.test_cuda_server_errors import app_for as http_app
    from tests.test_openai_compliance import exchange
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as directory:
        app = http_app(Path(directory))
        app.engine = app_for().engine
        with http_server(app) as port:
            status, _, text = exchange(port, "GET", runtime.ROUTE)
            assert status == 200 and json.loads(text)["sessions"]["allocated_slots"] == 40
            status, _, text = exchange(port, "POST", runtime.ROUTE, {"proposal_limit": 0})
            assert status == 200 and json.loads(text)["execution"]["pending"]["proposal_limit"] == 0
            status, _, _ = exchange(port, "POST", runtime.ROUTE, {"context_tokens": 4096})
            assert status == 400


@pytest.mark.parametrize("method,body", [("GET", None), ("POST", '{"proposal_limit":0}')])
def test_open_metrics_does_not_open_runtime_control(method, body):
    from tensorfold.server.authentication import KeyStore
    from tests.test_api_keys_http import request as http_request, serving

    with serving("cuda", KeyStore(["fixture-key"], metrics_open=True)) as (app, port):
        app.engine = app_for().engine
        before = runtime.snapshot(app)
        assert http_request(port, runtime.ROUTE, method=method, body=body)[0] == 401
        assert runtime.snapshot(app) == before
        headers = {"Authorization": "Bearer fixture-key"}
        assert http_request(port, runtime.ROUTE, headers, method=method, body=body)[0] == 200


@pytest.mark.parametrize("baseline,prefix,new,expected", [
    ([], [65], [66, 67, 68], (2, True)),
    ([65], [66], [67], (1, True)),
    ([], [], [65, 66], (2, False)),
    ([], [65, 0], [66, 67, 68], (2, True)),
])
def test_precommit_stop_cut_matches_tokenwise_output_and_handles_exact_end(baseline, prefix, new, expected):
    from tensorfold.cuda.reply_text import StopStrings, stop_prefix_limit

    class Characters:
        def decode(self, tokens, **kwargs):
            return "".join(chr(token) for token in tokens)

    stops = StopStrings(("ABC",), Characters(), skip=(0,))
    original = (list(baseline), list(prefix), list(new))
    assert stop_prefix_limit(stops, baseline, prefix, new) == expected
    assert (baseline, prefix, new) == original


def test_precommit_stop_storage_is_bounded_by_the_existing_tokenizer_tail():
    from tensorfold.cuda.reply_text import StopStrings, stop_prefix_limit

    class Characters:
        def decode(self, tokens, **kwargs):
            assert len(tokens) <= 11
            return "".join(chr(token) for token in tokens)

    stops = StopStrings(("ABC",), Characters())
    assert stop_prefix_limit(stops, [88] * 10000, [89] * 10000 + [65], [66, 67, 68]) == (2, True)


def test_idle_maintenance_is_woken_and_bounded_without_new_requests():
    import threading

    class RetainedIdle:
        def __init__(self):
            self.remaining = self.rounds = 0
            self.finished = threading.Event()

        def live(self):
            return 0

        def maintenance_pending(self):
            return self.remaining > 0

        def round(self):
            assert self.remaining > 0
            self.rounds += 1
            self.remaining -= 1
            if not self.remaining:
                self.finished.set()
            return []

        def finish(self, _done):
            pass

        def drop(self):
            return []

    decoder = RetainedIdle()
    scheduler = Scheduler(decoder).start()
    try:
        scheduler.request_runtime(lambda: setattr(decoder, "remaining", 3))
        assert decoder.finished.wait(5)
    finally:
        scheduler.close()
    assert decoder.rounds == 3


def test_wake_markers_coalesce_and_are_not_counted_as_queued_requests():
    from tensorfold.cuda.scheduler import Waiting

    waiting = Waiting()
    request = (SimpleNamespace(background=False), object())
    for _ in range(100):
        waiting.wake()
    assert waiting.qsize() == 0
    assert len(waiting.queue) == 1
    waiting.put(request)
    assert waiting.qsize() == 1 and waiting.foreground()
    assert waiting.get_nowait() is request
    assert waiting.is_wake(waiting.get_nowait())
    assert waiting.qsize() == 0 and waiting.empty()


def test_closed_scheduler_refuses_before_runtime_publication():
    published = []
    scheduler = Scheduler(SimpleNamespace())
    try:
        with pytest.raises(RuntimeError, match="closing"):
            scheduler.request_runtime(lambda: published.append(True))
    finally:
        scheduler.close()
    assert not published


def test_idle_derived_prefix_bytes_are_not_intentional_retained_sessions():
    app = app_for()
    decoder = app.engine.scheduler.decoder
    decoder.streams = {}
    reading = runtime.snapshot(app)
    assert reading["sessions"]["lifetime"] == "request_scoped"
    assert reading["sessions"]["retained"] == reading["sessions"]["active"] == 0
    assert reading["sessions"]["derived_prefix_slots"] == 1
    assert reading["sessions"]["derived_prefix_entries"] == 2
    assert reading["sessions"]["allocated_slots"] == 40


def test_manifest_install_runs_on_owner_boundary_and_preserves_newer_request():
    app = app_for()
    decoder = app.engine.scheduler.decoder
    decoder.execution_manifest_installer = object()
    scheduler = Scheduler(decoder, max_streams=40)
    scheduler._closing = False  # explicit test owner, no worker/thread is launched
    observed = []

    def install(name):
        observed.append(name)
        if name == "first.json":
            scheduler.request_execution_manifest("second.json")
        if name == "invalid.json":
            raise ValueError("wrong target identity")

    decoder.install_execution_manifest = install
    try:
        scheduler.request_execution_manifest("first.json")
        assert not observed
        scheduler._execution_boundary()
        reading = scheduler.execution_manifest_record()
        assert reading["effective"] == "first.json" and reading["pending"]
        assert reading["configured"] == "second.json"
        scheduler._execution_boundary()
        assert observed == ["first.json", "second.json"]
        assert scheduler.execution_manifest_record()["effective"] == "second.json"
        assert not scheduler.execution_manifest_record()["pending"]
        scheduler.request_execution_manifest("invalid.json")
        scheduler._execution_boundary()
        reading = scheduler.execution_manifest_record()
        assert reading["effective"] == "second.json" and reading["error"] == "wrong target identity"
        for invalid in ("/etc/passwd", "../manifest.json", "a/../manifest.json", "a//b", "x\\y", ""):
            with pytest.raises(ValueError, match="relative"):
                scheduler.request_execution_manifest(invalid)
    finally:
        scheduler.close()
