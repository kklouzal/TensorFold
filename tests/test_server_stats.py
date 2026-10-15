"""GET /stats (schema v1), the /metrics families rendered from the same snapshot, the reply's ``tensorfold`` block v2
and llama.cpp-shaped prefill progress: what Harness reads for load gating and per-call timings. CPU only."""

from __future__ import annotations

import json
import re
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")

from tensorfold.cuda import health
from tensorfold.server import metrics, stats
from tensorfold.server.cancellation import RequestCancelled
from tests.test_cuda_admission import http_server
from tests.test_cuda_server_errors import HI, app_for
from tests.test_openai_compliance import exchange, frames, mac_server
from tests.test_token_routes import MacApp

SECTIONS = {"object", "schema_version", "time", "server", "model", "limits", "load", "kv", "prefix_cache", "totals"}
LOAD = {"requests_in_flight", "requests_running", "requests_waiting", "lanes", "held_for_memory",
        "http_connections_in_use", "stream_tokens"}
TOTALS = {"requests", "finished", "cancelled", "failed", "prompt_tokens", "cached_prompt_tokens", "generated_tokens",
          "generated_tokens_live", "queue_seconds", "prefill_seconds", "decode_seconds", "rounds", "drafted",
          "accepted", "preemptions", "memory_waits", "memory_stops"}
SAMPLE = re.compile(r'[a-zA-Z_:][a-zA-Z0-9_:]*(\{([a-zA-Z_][a-zA-Z0-9_]*="([^"\\]|\\.)*",?)*\})? -?[0-9.e+]+')


def get_json(port, path):
    status, headers, text = exchange(port, "GET", path)
    return status, headers, json.loads(text)


def prometheus_families(text):
    """Every family's TYPE, after checking the text is valid exposition (HELP and TYPE once each before samples)."""

    types, helped = {}, set()
    for line in text.splitlines():
        if line.startswith("# HELP "):
            name = line.split()[2]
            assert name not in helped, f"HELP twice for {name}"
            helped.add(name)
        elif line.startswith("# TYPE "):
            _, _, name, kind = line.split()
            assert name not in types and name in helped, f"TYPE out of order for {name}"
            types[name] = kind
        else:
            assert SAMPLE.fullmatch(line), f"not a sample line: {line!r}"
            base = re.sub(r"(_bucket|_sum|_count)$", "", line.split("{")[0].split()[0])
            assert line.split("{")[0].split()[0] in types or base in types, f"sample before its TYPE: {line!r}"
    return types


def test_cuda_stats_is_schema_v1_and_counts_what_health_and_metrics_count(tmp_path):
    app = app_for(tmp_path)
    app.aliases = ("alias",)
    app.context_window = 4096
    with http_server(app) as port:
        status, _, idle = get_json(port, "/stats")
        assert status == 200 and set(idle) == SECTIONS and set(idle["load"]) == LOAD and set(idle["totals"]) == TOTALS
        assert (idle["object"], idle["schema_version"]) == ("tensorfold.stats", 1)
        assert idle["server"]["backend"] == "cuda" and idle["server"]["instance_id"] == stats.INSTANCE_ID
        assert idle["server"]["started_unix_s"] == round(stats.STARTED_UNIX_S, 3) and idle["time"]["uptime_s"] >= 0
        assert idle["model"]["id"] == "fake-cuda" and idle["model"]["aliases"] == ["alias"]
        assert idle["model"]["context_length"] == 4096 and idle["limits"]["lanes"] == 1   # one request at a time
        assert idle["kv"] is None and idle["prefix_cache"] is None and idle["load"]["lanes"] is None
        assert idle["totals"]["requests"] == 0 and idle["totals"]["generated_tokens_live"] == 0
        assert get_json(port, "/v1/stats")[2]["server"] == idle["server"]
        status, headers, text = exchange(port, "POST", "/v1/chat/completions", {"messages": HI, "max_tokens": 8})
        usage = json.loads(text)["usage"]
        after = get_json(port, "/stats")[2]
        healthy = get_json(port, "/health")[2]
        scrape = exchange(port, "GET", "/metrics")[2]
    totals = after["totals"]
    assert totals["requests"] == 1 and totals["finished"] == {"stop": 1, "length": 0, "tool_calls": 0}
    assert totals["prompt_tokens"] == usage["prompt_tokens"] == healthy["prompt_tokens_total"]
    assert totals["generated_tokens"] == usage["completion_tokens"] == healthy["completion_tokens_total"]
    assert totals["cancelled"] == totals["failed"] == 0 and totals["queue_seconds"] >= 0
    assert f"{metrics.PREFIX}prompt_tokens_total {totals['prompt_tokens']}" in scrape
    assert f'{metrics.PREFIX}request_success_total{{finished_reason="stop"}} 1' in scrape


def test_mac_stats_reads_its_scheduler_and_adds_its_memory():
    app = MacApp()
    app.scheduler = SimpleNamespace(active=1, waiting=2, filling=[object()], cancelled=0, preemptions=0)
    app.max_batch_size, app.context_window = 4, 8192
    with mac_server(app) as port:
        status, _, body = get_json(port, "/stats")
    assert status == 200 and set(body) == SECTIONS | {"mlx"} and body["server"]["backend"] == "mlx"
    assert body["load"]["requests_running"] == 2 and body["load"]["requests_waiting"] == 2
    assert body["load"]["requests_in_flight"] == 4 and body["load"]["lanes"] == {"decode": 1, "prefill": 1,
                                                                                  "memory_wait": None}
    assert body["limits"]["lanes"] == 4 and body["model"]["context_length"] == 8192
    assert body["totals"]["generated_tokens_live"] is None and body["kv"] is None   # nothing the Mac cannot count


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_stats_is_gated_like_metrics(backend):
    from tensorfold.server.authentication import KeyStore
    from tests.test_api_keys_http import request, serving

    with serving(backend, KeyStore(["fixture-key"])) as (_, port):
        assert request(port, "/stats")[0] == request(port, "/v1/stats")[0] == 401
        assert request(port, "/stats", {"Authorization": "Bearer fixture-key"})[0] == 200
        assert request(port, "/v1/models/fixture")[0] == 401
    with serving(backend, KeyStore(["fixture-key"], metrics_open=True)) as (_, port):
        assert request(port, "/stats")[0] == request(port, "/v1/stats")[0] == request(port, "/metrics")[0] == 200


class Decoder:
    """A concurrent decoder's tables as /stats reads them, with qwen4_exp's ``usage`` report."""

    keep = 8

    def __init__(self):
        self.streams = {0: SimpleNamespace(waiting=False, context=list(range(30)), prompt=[1] * 20),
                        1: SimpleNamespace(waiting=True, context=list(range(10)), prompt=[1] * 10)}
        self.filling = [SimpleNamespace(context=[], prompt=[1] * 50)]

    def live(self):
        return len(self.streams) + len(self.filling)

    def usage(self):
        return {"kv": {"budget_bytes": 1000, "held_bytes": 600, "active_bytes": 400, "usage_ratio": 0.4,
                       "bytes_per_token": 10, "free_tokens_estimate": 60},
                "prefix_cache": {"entries": 3, "capacity": 8, "slots": 2, "tokens": 70},
                "memory_waits": 5, "memory_stops": 1}


class Waiting:
    def qsize(self):
        return 1


def concurrent_app(tmp_path):
    app = app_for(tmp_path)
    app.engine.scheduler = SimpleNamespace(decoder=Decoder(), waiting=Waiting(), held=object(), max_streams=4,
                                           yields=2)
    app.engine.kv_pair = SimpleNamespace(key_dtype="rotorquant8-norm", value_dtype="rotorquant8-norm")
    app.engine.depth = 4
    return app


def test_a_concurrent_engine_s_load_memory_and_prefix_cache(tmp_path):
    app = concurrent_app(tmp_path)
    body = stats.snapshot(app)
    assert body["limits"]["lanes"] == 4 and body["limits"]["prefix_cache_entries"] == 8
    assert body["model"]["kv_dtype"] == "rotorquant8-norm" and body["model"]["mtp_drafts"] == 4
    assert body["load"]["lanes"] == {"decode": 1, "prefill": 1, "memory_wait": 1}
    assert (body["load"]["requests_running"], body["load"]["requests_waiting"]) == (3, 2)   # queued + held
    assert body["load"]["held_for_memory"] is True and body["load"]["stream_tokens"] == [30, 10, 50]
    assert body["kv"]["usage_ratio"] == 0.4 and body["prefix_cache"]["tokens"] == 70
    assert (body["totals"]["memory_waits"], body["totals"]["memory_stops"], body["totals"]["preemptions"]) == (5, 1, 2)
    text = metrics.render(app)
    types = prometheus_families(text)
    for name, kind in (("kv_cache_usage_perc", "gauge"), ("kv_memory_budget_bytes", "gauge"),
                       ("kv_memory_held_bytes", "gauge"), ("kv_memory_active_bytes", "gauge"),
                       ("prefix_cache_entries", "gauge"), ("prefix_cache_capacity", "gauge"),
                       ("prefix_cache_tokens", "gauge"), ("prefix_cache_queries_total", "counter"),
                       ("prefix_cache_hits_total", "counter"), ("lanes", "gauge"), ("lanes_max", "gauge"),
                       ("requests_held_for_memory", "gauge"), ("memory_waits_total", "counter"),
                       ("memory_stops_total", "counter"), ("num_preemptions_total", "counter"),
                       ("request_success_total", "counter"), ("client_disconnections_total", "counter"),
                       ("request_queue_time_seconds", "histogram"), ("request_prefill_time_seconds", "histogram"),
                       ("request_time_per_output_token_seconds", "histogram"), ("process_start_time_seconds", "gauge"),
                       ("build_info", "gauge")):
        assert types.get(metrics.PREFIX + name) == kind, name
    assert f"{metrics.PREFIX}kv_cache_usage_perc 0.4\n" in text          # vLLM's meaning: one unlabeled fraction
    assert f'{metrics.PREFIX}lanes{{phase="memory_wait"}} 1' in text
    assert re.search(r'build_info\{version="[^"]+",backend="cuda",instance="[0-9a-f]{32}",model="fake-cuda",'
                     r'kv_dtype="rotorquant8-norm",mtp_drafts="4",lanes="4"\} 1', text)


def test_an_idle_bare_app_still_renders_valid_exposition():
    prometheus_families(metrics.render(SimpleNamespace()))
    prometheus_families(metrics.render(SimpleNamespace(served='x"y\\z')))     # an id needing escapes in build_info


def test_outcomes_count_how_each_request_ended(tmp_path):
    app = app_for(tmp_path)
    with health.of(app).running(3, [1, 2]) as request:
        request.outcome = "length"
    with pytest.raises(RequestCancelled), health.of(app).running(4, []):
        raise RequestCancelled("the client left")
    with pytest.raises(RuntimeError), health.of(app).running(5, [9]):
        raise RuntimeError("the engine failed")
    totals = stats.snapshot(app)["totals"]
    assert totals["requests"] == 3 and totals["finished"]["length"] == 1
    assert (totals["cancelled"], totals["failed"]) == (1, 1) and totals["prompt_tokens"] == 12
    text = metrics.render(app)
    assert f"{metrics.PREFIX}client_disconnections_total 1" in text
    assert f'{metrics.PREFIX}request_success_total{{finished_reason="error"}} 1' in text


def test_a_cuda_request_whose_client_left_while_it_waited_is_an_abort(tmp_path):
    app = app_for(tmp_path)
    with pytest.raises(RequestCancelled):
        app.run({"messages": HI, "max_tokens": 4}, True, lambda delta: True, cancelled=lambda: True)
    assert stats.snapshot(app)["totals"]["cancelled"] == 1 and not app.engine.calls


class Timed:
    """An engine that reports its own timings, a lane engine's."""

    eos = (0,)

    def __init__(self, tokens=4):
        self.tokens = tokens

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        on_tokens([ord("a")])
        time.sleep(0.02)
        on_tokens([ord("b")] * (self.tokens - 2) + [0])
        return {"prefill_s": 0.01, "decode_s": 0.5, "rounds": 2, "drafted": 3, "accepted": 1, "cached": 0,
                "queue_s": 0.25}


def test_the_reply_block_v2_names_the_request_and_times_it_from_arrival(tmp_path, capsys):
    app = app_for(tmp_path)
    app.engine = Timed(tokens=4)
    prepare = app.prepare

    def slow_prepare(body, chat):
        time.sleep(0.1)                     # template, tokens and images: before the old clocks started
        return prepare(body, chat)

    app.prepare = slow_prepare
    with http_server(app) as port:
        status, headers, text = exchange(port, "POST", "/v1/chat/completions", {"messages": HI, "max_tokens": 8},
                                         {"X-Client-Request-Id": "harness-turn-7"})
        block = json.loads(text)["tensorfold"]
        streamed = exchange(port, "POST", "/v1/chat/completions",
                            {"messages": HI, "max_tokens": 8, "stream": True,
                             "stream_options": {"include_usage": True}})[2]
    assert status == 200 and headers["x-request-id"] == block["request_id"] == "harness-turn-7"
    assert block["stats_version"] == 2 and block["prepare_s"] >= 0.1
    assert block["ttft_s"] >= block["prepare_s"]                     # from arrival, preparation included (F4)
    assert block["queue_s"] >= 0.25                                   # the engine's queue, plus any turn wait
    assert block["decode_tps"] == round(3 / 0.5, 2)                   # tokens after the first over decode_s
    assert "done harness-turn-7 " in capsys.readouterr().out          # the server's done line names it too
    events = frames(streamed)
    finish, usage_frame = events[-2], events[-1]
    assert usage_frame["choices"] == [] and usage_frame["usage"] is not None
    assert usage_frame["tensorfold"] == finish["tensorfold"] and finish["usage"] is None   # F3: on the usage frame
    snapshot = stats.snapshot(app)["totals"]
    assert snapshot["prefill_seconds"] == 0.02 and snapshot["decode_seconds"] == 1.0 and snapshot["rounds"] == 4


def test_a_reply_without_tokens_or_decode_time_leaves_those_timings_out(tmp_path):
    app = app_for(tmp_path)

    class Silent:
        eos = (0,)

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            return {}

    app.engine = Silent()
    result = app.run({"messages": HI, "max_tokens": 4}, True, lambda delta: True)
    block = result["stats"]
    assert "ttft_s" not in block and "decode_tps" not in block and block["request_id"].startswith("req-")
    assert set(block) >= {"stats_version", "request_id", "queue_s", "prepare_s", "token_sha"}


# -- llama.cpp's prompt_progress (opt-in ``return_progress``) -------------------------------------------------------


def harness_accepts(events):
    """Harness's llama.cpp-dialect checks (OpenAiChatStreamParser.ParsePromptProgress and its frame rules)."""

    previous = None
    for event in events:
        progress = event.get("prompt_progress")
        if progress is None:
            continue
        choice, = event["choices"]
        delta = choice["delta"]
        assert choice["index"] == 0 and choice["finish_reason"] is None and delta.get("role") == "assistant"
        assert all(delta.get(key) is None for key in ("content", "reasoning_content", "reasoning"))
        assert "tool_calls" not in delta
        for key in ("total", "cache", "processed"):
            assert isinstance(progress[key], int) and progress[key] >= 0
        assert isinstance(progress["time_ms"], (int, float)) and progress["time_ms"] >= 0
        assert progress["cache"] <= progress["processed"] <= progress["total"]
        if previous is not None:
            assert progress["total"] == previous["total"]
            assert all(progress[key] >= previous[key] for key in ("cache", "processed", "time_ms"))
        previous = progress
    return previous


class Prefilling:
    """A concurrent engine that reports its prompt's prefill as it goes, as Flash Next's scheduler does."""

    eos = (0,)
    concurrent = True

    def __init__(self):
        self.progressed = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, progress=None):
        self.progressed.append(progress is not None)
        if progress is not None:
            for processed in (2, 5, len(prompt)):
                if progress(len(prompt), 2, processed, processed * 10.0):
                    return {}
        on_tokens([ord("o"), ord("k"), 0])
        return {}


@pytest.mark.parametrize("chat", [True, False])
def test_return_progress_streams_llamacpp_prompt_progress_before_the_reply(tmp_path, chat):
    app = app_for(tmp_path)
    app.engine = Prefilling()
    body = {"messages": HI} if chat else {"prompt": "Hi"}
    with http_server(app) as port:
        status, _, text = exchange(port, "POST", "/v1/chat/completions" if chat else "/v1/completions",
                                   {**body, "max_tokens": 8, "stream": True, "return_progress": True,
                                    "stream_options": {"include_usage": True}})
        plain = exchange(port, "POST", "/v1/chat/completions", {"messages": HI, "max_tokens": 8, "stream": True})[2]
        unasked = exchange(port, "POST", "/v1/chat/completions",
                           {"messages": HI, "max_tokens": 8, "return_progress": True})[0]
        malformed = exchange(port, "POST", "/v1/chat/completions",
                             {"messages": HI, "max_tokens": 8, "stream": True, "return_progress": "yes"})[0]
    events = frames(text)
    marked = [i for i, event in enumerate(events) if "prompt_progress" in event]
    assert status == 200 and len(marked) == 3
    if chat:
        last = harness_accepts(events)
        assert last["processed"] == last["total"] and last["cache"] == 2
        first_text = next(i for i, e in enumerate(events) if e["choices"] and e["choices"][0]["delta"].get("content"))
        assert max(marked) < first_text
    else:
        assert all(events[i]["choices"][0]["text"] == "" for i in marked)
    assert all(events[i]["usage"] is None for i in marked)
    assert "prompt_progress" not in plain and unasked == 200 and malformed == 400
    assert app.engine.progressed == [True, False, False]


def test_an_engine_without_progress_sends_no_progress_frames(tmp_path):
    app = app_for(tmp_path)                 # a serial engine: no progress parameter
    with http_server(app) as port:
        text = exchange(port, "POST", "/v1/chat/completions",
                        {"messages": HI, "max_tokens": 8, "stream": True, "return_progress": True})[2]
    assert "prompt_progress" not in text and frames(text)[-1]["choices"][0]["finish_reason"] == "stop"


def test_the_scheduler_reports_admission_and_each_reported_pass():
    from tensorfold.cleanup import finish
    from tensorfold.cuda.scheduler import Scheduler

    class Passes:
        """Fills a prompt 3 rows a round, reporting each pass, then decodes one token."""

        def __init__(self):
            self.streams, self.filling = {}, []

        def live(self):
            return len(self.streams) + len(self.filling)

        def admit(self, s):
            s.cached, s.rows = 1, 1
            self.filling.append(s)

        def round(self):
            done = []
            for s in list(self.filling):
                s.rows = min(len(s.prompt), s.rows + 3)
                if s.on_fill is not None:
                    s.on_fill(s.rows)
                if s.rows == len(s.prompt):
                    self.filling.remove(s)
                    s.take([7], eos=(7,))
                    done.append(s)
            return done

        def cancel(self, streams):
            return []

        def finish(self, done):
            pass

        def drop(self):
            return []

    scheduler = Scheduler(Passes(), max_streams=2).start()
    try:
        heard = []
        out = []
        stats_ = scheduler.submit(list(range(8)), 4, None, True, lambda new: out.extend(new),
                                  progress=lambda *values: heard.append(values) or False)
    finally:
        finish([scheduler.close])
    assert [(total, cached, rows) for total, cached, rows, _ in heard] == [(8, 1, 1), (8, 1, 4), (8, 1, 7), (8, 1, 8)]
    assert all(ms >= 0 for *_, ms in heard) and out == [7] and stats_["queue_s"] >= 0


def test_a_replayed_stream_sums_its_queue_time():
    from tensorfold.cuda.streams import Stream

    first = Stream([1, 2], 4, None)
    first.queue_s = 0.5
    again = first.continued()
    again.queue_s = 0.25
    assert again.stats()["queue_s"] == 0.75 and again.on_fill is None      # a replay reports no progress


def test_a_prefix_cache_use_publishes_a_new_sorted_list():
    from tensorfold.families.qwen4_exp.cuda import prefixes

    slot_a, slot_b = object(), object()
    owner = SimpleNamespace(used={}, clock=0, kept=[([1, 2, 3], slot_a, None, None), ([1], slot_b, None, None)])
    before = owner.kept
    owner.clock = 5
    prefixes._use(owner, slot_b)
    assert owner.kept is not before and before == [([1, 2, 3], slot_a, None, None), ([1], slot_b, None, None)]
    assert [k[0] for k in owner.kept] == [[1, 2, 3], [1]]          # values 3 and 5+1, ascending


def test_the_footprint_reader_is_resolved_once_and_never_loads_a_library_off_macos(monkeypatch):
    import ctypes
    import sys

    from tensorfold.server import memory_budget

    if sys.platform == "darwin":
        pytest.skip("macOS reads its footprint")
    memory_budget._footprint_reader.cache_clear()
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: (_ for _ in ()).throw(AssertionError("dlopen on a scrape")))
    try:
        assert memory_budget.process_footprint() is None and memory_budget.process_footprint() is None
    finally:
        memory_budget._footprint_reader.cache_clear()


def test_a_scrape_stays_the_same_size_however_many_requests_finished(tmp_path):
    """The O(1) contract (owner bound 2026-10-15): nothing a scrape reads or writes grows with traffic."""

    app = concurrent_app(tmp_path)
    sizes = []
    for _ in range(2):
        for _ in range(500):
            metrics.note(app, prompt=10, generation=5, decode=0.1, latency=0.2, ttft=0.05, queue=0.01, prefill=0.02,
                         outcome="stop")
        sizes.append((len(json.dumps(stats.snapshot(app))), len(metrics.render(app).splitlines())))
    assert sizes[0][1] == sizes[1][1] and abs(sizes[0][0] - sizes[1][0]) < 64 and sizes[1][0] < 4096


def test_in_flight_counts_requests_waiting_for_a_one_at_a_time_turn(tmp_path):
    app = app_for(tmp_path)                   # no concurrent scheduler: requests take turns
    app.turns = SimpleNamespace(parked=2)
    with health.of(app).running(3, []):
        load = stats.snapshot(app)["load"]
    assert (load["requests_running"], load["requests_waiting"], load["requests_in_flight"]) == (1, 2, 3)


def documented_fields():
    """docs/api.md's /stats table, its shorthand expanded: ``.x`` continues the previous field's section and
    ``{a,b}`` lists siblings."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs" / "api.md").read_text()
    table = text.split("\n## Statistics\n", 1)[1].split("\n## ", 1)[0]
    fields, last = set(), ""
    for line in table.splitlines():
        if not line.startswith("| `"):
            continue
        for name in re.findall(r"`([^`]+)`", line.split("|")[1]):
            name = last.rsplit(".", 1)[0] + name if name.startswith(".") else name
            last = name
            head, _, siblings = name.partition("{")
            fields.update([head + part for part in siblings.rstrip("}").split(",")] if siblings else [name])
    return fields


def field_paths(value, prefix=""):
    """{dotted path: value} of every field of a /stats body, sections and their fields alike."""

    found = {}
    for key, item in value.items():
        found[prefix + key] = item
        if isinstance(item, dict):
            found.update(field_paths(item, prefix + key + "."))
    return found


def test_the_api_reference_documents_every_stats_field_and_no_other(tmp_path):
    cuda = field_paths(stats.snapshot(concurrent_app(tmp_path)))     # kv and prefix_cache filled (usage's keys)
    app = MacApp()
    app.scheduler = SimpleNamespace(active=1, waiting=0, filling=[], cancelled=0, preemptions=0)
    with mac_server(app) as port:
        mac = field_paths(get_json(port, "/stats")[2])
    documented = documented_fields()
    assert documented <= set(cuda) | set(mac), documented - set(cuda) - set(mac)
    for path in (*cuda, *mac):          # a field, a field inside a documented one, or a section of documented fields
        assert any(path == name or path.startswith(name + ".") or name.startswith(path + ".")
                   for name in documented), f"/stats field {path} is not in docs/api.md"


def test_successful_polls_print_no_access_line_on_the_mac_either(capsys):
    from tensorfold.server.live import ChunkRate, Meter

    app = MacApp()
    app.scheduler = SimpleNamespace(active=0, waiting=0, filling=[], cancelled=0, preemptions=0, decoded=Meter(),
                                    prefilled=ChunkRate())
    with mac_server(app) as port:
        for path in ("/stats", "/v1/stats", "/metrics", "/v1/metrics?x=1", "/health"):
            assert exchange(port, "GET", path)[0] == 200, path
        assert exchange(port, "GET", "/v1/models")[0] == 200
        assert exchange(port, "GET", "/stats/missing")[0] == 404
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[tensorfold] 127.0.0.1 ")]
    assert any('"GET /v1/models HTTP/1.1" 200' in line for line in lines), lines
    assert any('"GET /stats/missing HTTP/1.1" 404' in line for line in lines), lines
    assert not any(poll in line for line in lines for poll in ("/stats HTTP", "/v1/stats HTTP", "/metrics",
                                                               "/health")), lines
