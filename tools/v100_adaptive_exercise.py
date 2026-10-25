"""Record adaptive-service lifecycle scenarios using explicit supplied requests.

Every request file is JSON chat-request data. The tool preserves its messages,
sampling, stop settings and reply allowance; it adds only SSE/usage/token-ID
diagnostics. /tokenize must prove prompt + reply fits the effective context.
The reserved name ``overrun`` instead requires the server's context refusal.

Use --request short=short.json --request intermediate=middle.json
--request long=long.json --request stop=stop.json. Additional distinct supplied
requests can populate retained slots. Missing inputs or unobserved conditions
are reported unsupported, never inferred from allocated slot counts. This tool
records behavior and timings; it does not grant quality/profile eligibility.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
import urllib.error

if __package__:
    from . import openai_protocol as protocol
    from . import v100_adaptive_qualification as qualification
    from . import v100_rotorquant_live as live
    from .benchmark_output import write_json
    from .worker_lifetime import Task, drain, raise_failures
else:
    import openai_protocol as protocol
    import v100_adaptive_qualification as qualification
    import v100_rotorquant_live as live
    from benchmark_output import write_json
    from worker_lifetime import Task, drain, raise_failures


SCENARIOS = ("requests", "mixed", "burst", "cancel", "reuse", "retention", "migration",
             "oscillation", "limits", "zero-proposal", "stop", "eos", "context")


class Unsupported(RuntimeError):
    """An input/capability/observed state required by this exercise is absent."""

    def __init__(self, message, *, evidence=None):
        super().__init__(message)
        self.evidence = evidence


def load_requests(named, *, budget_bytes):
    qualification.positive(budget_bytes, "request-data byte budget")
    result, remaining = {}, budget_bytes
    for entry in named:
        name, separator, filename = entry.partition("=")
        if not separator or re.fullmatch(r"[A-Za-z0-9_-]{1,48}", name) is None or name in result:
            raise ValueError("unique safe request NAME=JSON_FILE entries required")
        path = Path(filename)
        with path.open("rb") as source:
            raw = source.read(remaining + 1)
        if len(raw) > remaining:
            raise ValueError("request data exceeds cumulative byte budget")
        remaining -= len(raw)
        body = protocol.object_json(raw)
        if type(body.get("max_tokens")) is not int or body["max_tokens"] <= 0:
            raise ValueError("each supplied request needs a positive integer max_tokens")
        messages = body.get("messages")
        if (type(messages) is not list or not messages or any(
                type(message) is not dict or type(message.get("content")) is not str for message in messages)
                or body.get("tools") or body.get("functions") or type(body.get("n", 1)) is not int
                or body.get("n", 1) != 1):
            raise ValueError("single text-only chat requests are required")
        if "stream_options" in body and type(body["stream_options"]) is not dict:
            raise ValueError("stream_options must be an object when supplied")
        result[name] = {"body": body, "source": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    if not result:
        raise ValueError("at least one explicit request file is required")
    return result


class Suite:
    def __init__(self, args, requests):
        self.args, self.requests, self.out = args, requests, args.out
        self.limits = protocol.from_arguments(args)
        self.client = live.Client(args.url, self.out, timeout=args.timeout, limits=self.limits)
        self.lock, self.calls, self.sequence, self.cleanup_calls = threading.Lock(), [], 0, 0
        self.active_read_control = None  # one caller-owned group, retained through worker drain
        self.effective = None
        self.initial_runtime = None
        self.changed_settings, self.changed_degree = False, False
        self.cancelled_group = False
        self.report = {"schema": "tensorfold.v100-adaptive-exercise", "schema_version": 1,
                       "evidence": "recorded-HTTP-scenarios", "complete": False, "passed": False,
                       "qualification": "not-granted-by-this-tool", "scenarios": [],
                       "selected_scenarios": list(args.scenarios),
                       "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "http_recorder_sha256": hashlib.sha256(Path(live.__file__).read_bytes()).hexdigest(),
                       "inputs": {name: {key: value for key, value in row.items() if key != "body"}
                                  for name, row in requests.items()},
                       "budgets": {"calls": args.max_calls, "cleanup_calls": args.max_cleanup_calls,
                                   "inflight": args.max_inflight, "request_data_bytes": args.request_mib << 20,
                                   "batch_response_bytes": args.batch_response_mib << 20,
                                   "response_bytes_each": self.limits.response_bytes,
                                   "sse_events_each": args.max_events},
                       "http_lifetime": "cooperative response-body deadline/cancel; workers drain before controls restore; "
                                        "an in-flight read may last its socket timeout; acquisition is not whole-wall bounded",
                       "pending_gates": ["independent-target-and-codec-quality-oracles",
                                         "stochastic-acceptance-distribution", "native-SM70-memory-diagnostics",
                                         "measured-static-versus-adaptive-performance-frontier",
                                         "continuous-hardware-thermal-clock-link-telemetry"]}

    def save(self):
        with self.lock:
            self.report["calls"] = list(self.calls)
        write_json(self.out / "suite.json", self.report)

    def call(self, label, body=None, *, path=None, cancel=False, object_response=False,
             first=None, cleanup=False, expected_refusal=None):
        with self.lock:
            if cleanup:
                if self.cleanup_calls >= self.args.max_cleanup_calls:
                    raise RuntimeError("cleanup HTTP-call budget exhausted")
                self.cleanup_calls += 1
            elif self.sequence - self.cleanup_calls >= self.args.max_calls:
                raise RuntimeError("scenario HTTP-call budget exhausted")
            number, self.sequence = self.sequence, self.sequence + 1
        exchange = f"{number:06d}-{label}"
        started, arrivals = time.perf_counter(), []

        def observe(_event, pieces):
            if pieces:
                if len(arrivals) >= self.args.max_events:
                    raise ValueError("SSE timing-event budget exhausted")
                arrivals.append([time.perf_counter() - started, sum(len(piece) for piece in pieces)])
                if first is not None:
                    first.set()

        receipt = {"exchange": exchange, "started_monotonic_s": started,
                   "scope": "HTTP including recording; SSE text-piece arrivals are not exact token times"}
        primary = None
        read_control = self.active_read_control or live.ReadControl(time.monotonic() + self.args.operation_timeout)
        try:
            result = self.client.call(exchange, body, path=path, cancel=cancel,
                                      object_response=object_response, observer=observe, read_control=read_control)
            receipt["status"] = "intentionally-cancelled" if cancel else "complete"
            return result
        except BaseException as error:
            primary = error
            if isinstance(error, urllib.error.HTTPError) and error.code == 400 and expected_refusal is not None:
                try:
                    with (self.out / exchange / "response.raw").open("rb") as source:
                        raw = source.read(self.limits.response_bytes + 1)
                    if len(raw) > self.limits.response_bytes:
                        raise ValueError("refusal response exceeds protocol byte budget")
                    refused = json.loads(raw)
                except BaseException as receipt_error:
                    raise error from receipt_error
                if type(refused) is dict and type(refused.get("error")) is dict and refused["error"].get("code") == expected_refusal:
                    receipt.update(status="expected-refusal", code=expected_refusal)
                    primary = None
                    return refused
            receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            receipt.update(elapsed_s=time.perf_counter() - started, text_arrivals=arrivals)
            with self.lock:
                self.calls.append({key: value for key, value in receipt.items() if key != "text_arrivals"})
            directory = self.out / exchange
            if directory.is_dir():
                try:
                    write_json(directory / "timing.json", receipt)
                except BaseException as publication:
                    raise_failures(primary, [publication])

    def stats(self, label, *, cleanup=False):
        stats = self.call(label, cleanup=cleanup)
        current = qualification.profile(stats)
        if self.effective is not None and current != self.effective:
            raise ValueError("server instance/model/context/full-window capacity changed during exercise")
        return stats

    def runtime(self, label, *, cleanup=False):
        try:
            result = self.call(label, path="/v1/runtime", object_response=True, cleanup=cleanup)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise Unsupported("service has no adaptive runtime receipt") from error
            raise
        if result.get("object") != "tensorfold.runtime" or result.get("context_tokens") != self.effective["context_tokens"]:
            raise ValueError("runtime context/profile differs from the effective service contract")
        sessions = result.get("sessions") or {}
        for key in ("retained", "active", "decode", "prefill"):
            value = sessions.get(key)
            if type(value) is not int or not 0 <= value <= self.effective["resident_capacity"]:
                raise ValueError("runtime session count is missing or outside computed capacity: " + key)
        slots = (result.get("placement") or {}).get("slots") or []
        for slot in slots:
            if slot.get("history_poisoned"):
                raise ValueError("runtime reports poisoned retained history")
            cached = slot.get("target_cached_length")
            if type(cached) is not int or not 0 <= cached <= self.effective["context_tokens"]:
                raise ValueError("target cached position exceeds its logical context commitment")
        migration_limit = (result.get("capacity_plan") or {}).get("resource_ledger_bytes", {}).get("migration_overlap_limit")
        if type(migration_limit) is int and migration_limit >= 0:
            occupancy = [slot.get("migration_bytes", 0) for slot in slots]
            if any(type(value) is not int or value < 0 for value in occupancy) or sum(occupancy) > migration_limit:
                raise ValueError("observed migration occupancy exceeds the admitted rank limit")
        return result

    def wait_runtime(self, label, condition, *, cleanup=False, observe=None):
        deadline = time.monotonic() + self.args.poll_timeout
        latest = None
        while True:
            latest = self.runtime(label, cleanup=cleanup)
            if observe is not None:
                observe(latest)
            if condition(latest):
                return latest
            blocked = ((latest.get("execution") or {}).get("migration") or {}).get("blocked")
            if blocked:
                raise Unsupported("runtime migration remains blocked: " + str(blocked))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Unsupported("required runtime condition was not observed before the polling deadline")
            time.sleep(min(self.args.poll_interval, remaining))

    def control(self, changes, *, cleanup=False):
        try:
            return self.call("runtime-update", changes, path="/v1/runtime", object_response=True, cleanup=cleanup)
        except urllib.error.HTTPError as error:
            if error.code in (400, 404):
                raise Unsupported("runtime does not support requested control " + ",".join(changes)) from error
            raise

    def choose(self, role=None):
        candidates = [name for name, row in self.requests.items() if name != "overrun" and row.get("fits")]
        if role is not None:
            if role not in candidates:
                raise Unsupported("explicit fitting request file required for role: " + role)
            return role
        if not candidates:
            raise Unsupported("no supplied request fits the full prompt-plus-reply window")
        return candidates[0]

    def request(self, name, label, *, first=None, cancel=False):
        row = self.requests[name]
        if not row.get("fits"):
            raise Unsupported("supplied prompt-plus-reply does not fit: " + name)
        original = row["body"]
        body = {**copy.deepcopy(original), "stream": True, "return_token_ids": True,
                "stream_options": {**original.get("stream_options", {}), "include_usage": True}}
        result = self.call(label, body, first=first, cancel=cancel)
        if cancel:
            if result.get("intentionally_incomplete") is not True:
                raise ValueError("cancellation failed to interrupt an active stream")
            return {"request": name, "intentionally_cancelled": True}
        completed = live.completion(result, original["max_tokens"])
        if result["usage"]["prompt_tokens"] != row["prompt_tokens"]:
            raise ValueError("generation prompt count differs from the exact tokenizer endpoint")
        if row["prompt_tokens"] + len(completed["ids"]) > self.effective["context_tokens"]:
            raise ValueError("completed request exceeded its context allowance")
        return {"request": name, "tokens": len(completed["ids"]), "finish": completed["finish"],
                "token_ids_sha256": hashlib.sha256(json.dumps(completed["ids"], separators=(",", ":")).encode()).hexdigest(),
                "cached_prompt_tokens": (result.get("tensorfold") or {}).get("cached"),
                "runtime_timings": {key: value for key, value in (result.get("tensorfold") or {}).items()
                                    if key in ("prefill_s", "decode_s", "ttft_s", "decode_tps", "queue_s", "drafted", "accepted")}}

    def require_concurrency(self, count):
        if count > self.args.max_inflight or count * self.limits.response_bytes > self.args.batch_response_mib << 20:
            raise Unsupported("requested concurrent response/worker budgets need an explicit increase")

    def concurrent(self, names, *, burst=False, degree_sequence=()):
        count = len(names)
        self.require_concurrency(count)
        ready, abort, first = threading.Event(), threading.Event(), threading.Event()
        control = live.ReadControl(time.monotonic() + self.args.operation_timeout, abort)
        self.active_read_control = control
        results, tasks, primary = [None] * count, [], None
        observed = {"peak_active": 0, "peak_prefill": 0, "peak_decode": 0,
                    "mixed_prefill_decode": False, "partial_migration": False, "first_before_burst": False,
                    "retained_counts_while_active": []}

        def work(index):
            if index or not burst:
                ready.wait()
            if not abort.is_set():
                results[index] = self.request(names[index], "concurrent-request", first=first if index == 0 else None)

        def sample():
            state = self.runtime("concurrent-runtime")
            sessions = state["sessions"]
            for axis in ("active", "prefill", "decode"):
                observed["peak_" + axis] = max(observed["peak_" + axis], sessions[axis])
            if sessions["active"] and sessions["retained"] not in observed["retained_counts_while_active"]:
                observed["retained_counts_while_active"].append(sessions["retained"])
            observed["mixed_prefill_decode"] |= sessions["prefill"] > 0 and sessions["decode"] > 0
            slots = (state.get("placement") or {}).get("slots") or []
            observed["partial_migration"] |= any(slot.get("pending_attention_degree") is not None for slot in slots)
            return state

        try:
            for index in range(count):
                task = Task(lambda index=index: work(index), name=f"adaptive-exercise-{index}")
                tasks.append(task)
                task.start()
            if burst:
                deadline = time.monotonic() + self.args.poll_timeout
                while not first.is_set() and not tasks[0].done.is_set():
                    control.check()
                    state = sample()
                    observed["first_before_burst"] |= state["sessions"]["active"] == 1
                    if time.monotonic() >= deadline:
                        raise Unsupported("first active request was not observed before the burst")
                    time.sleep(self.args.poll_interval)
                if tasks[0].done.is_set():
                    observed["first_before_burst"] = False
            ready.set()
            for degree in degree_sequence:
                self.changed_degree = True
                self.control({"attention_degree": degree})
                sample()
            while not all(task.done.is_set() for task in tasks):
                control.check()
                failed = next((task.error for task in tasks if task.done.is_set() and task.error is not None), None)
                if failed is not None:
                    raise failed
                sample()
                time.sleep(self.args.poll_interval)
        except BaseException as error:
            primary = error
        finally:
            if primary is not None:
                self.cancelled_group = True
                abort.set()
            ready.set()
            try:
                drain(tasks, primary)
            finally:
                self.active_read_control = None
        return {"offered": count, "completed": len([row for row in results if row is not None]),
                "results": results, "observed": observed}

    def restore(self):
        if self.initial_runtime is None:
            return
        failures = []
        if self.cancelled_group:
            try:
                self.wait_runtime("cancelled-group-server-retirement", lambda value: value["sessions"]["active"] == 0,
                                  cleanup=True)
            except BaseException as error:
                failures.append(error)
        if self.changed_settings:
            settings = self.initial_runtime["execution"]["effective"]
            changes = {key: settings[key] for key in ("prefill_tokens", "proposal_limit", "row_budget", "session_ceiling")}
            try:
                self.control(changes, cleanup=True)
                self.wait_runtime("restored-settings", lambda value: value["execution"]["effective"] == settings,
                                  cleanup=True)
            except BaseException as error:
                failures.append(error)
        if self.changed_degree:
            migration = self.initial_runtime["execution"].get("migration")
            if migration is not None:
                degree = migration["requested_degree"]
                try:
                    self.control({"attention_degree": degree}, cleanup=True)
                    self.wait_runtime("restored-degree", lambda value: migration_done(value, degree), cleanup=True)
                except BaseException as error:
                    failures.append(error)
        if failures:
            raise_failures(None, failures)


def migration_done(runtime, degree):
    migration = (runtime.get("execution") or {}).get("migration")
    slots = (runtime.get("placement") or {}).get("slots") or []
    return bool(migration and migration.get("requested_degree") == degree and migration.get("pending") is False
                and slots and all(slot.get("attention_degree") == degree
                                  and slot.get("pending_attention_degree") is None for slot in slots))


def exercise_scenario(suite, scenario):
    capacity = suite.effective["resident_capacity"]
    name = suite.choose()
    if scenario == "requests":
        return {"results": [suite.request(case, "supplied-request") for case in suite.requests
                            if case != "overrun" and suite.requests[case]["fits"]]}
    if scenario == "mixed":
        short, long = suite.choose("short"), suite.choose("long")
        result = suite.concurrent([short, long], burst=True)
        if not result["observed"]["mixed_prefill_decode"]:
            raise Unsupported("mixed prefill/decode overlap was not observed; exact supplied requests were retained", evidence=result)
        return result
    if scenario == "burst":
        suite.require_concurrency(capacity)
        result = suite.concurrent([name] * capacity, burst=True)
        if not result["observed"]["first_before_burst"] or result["observed"]["peak_active"] < capacity:
            raise Unsupported("offered burst did not produce observed one-to-capacity active execution", evidence=result)
        return result
    if scenario == "cancel":
        before = suite.stats("before-cancel")
        result = suite.request(name, "cancelled-request", cancel=True)
        suite.wait_runtime("cancel-retirement", lambda value: value["sessions"]["active"] == 0)
        after = suite.stats("after-cancel")
        start, end = (body.get("totals", {}).get("cancelled") for body in (before, after))
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError("cancelled request did not retire and increment the cancellation counter")
        return {**result, "cancel_counter_delta": end - start, "recovery": suite.request(name, "post-cancel-reuse")}
    if scenario == "reuse":
        before = suite.runtime("before-slot-reuse")
        names = [case for case, row in suite.requests.items() if case != "overrun" and row["fits"]]
        results = [suite.request(case, "reused-slot") for _ in range(suite.args.cycles) for case in names]
        after = suite.runtime("after-slot-reuse")
        initial = {row["slot"]: row["slot_generation"] for row in (before.get("placement") or {}).get("slots", [])}
        changed = [row["slot"] for row in (after.get("placement") or {}).get("slots", [])
                   if row.get("slot") in initial and row.get("slot_generation", -1) > initial[row["slot"]]]
        if not changed:
            raise Unsupported("completed requests supplied no observable slot-generation reuse evidence")
        return {"results": results, "slots_with_new_generation": changed}
    if scenario == "retention":
        names = [case for case, row in suite.requests.items() if case != "overrun" and row["fits"]]
        if len(names) < capacity:
            raise Unsupported("all-resident test requires at least capacity distinct explicitly supplied request files")
        for case in names[:capacity]:
            suite.request(case, "retention-prime")
        before = suite.runtime("all-resident-before")
        if before["sessions"]["retained"] != capacity or before["sessions"]["active"] != 0:
            raise Unsupported("actual runtime does not retain every computed resident session")
        result = suite.concurrent([names[0]])
        if capacity not in result["observed"]["retained_counts_while_active"]:
            raise Unsupported("all-retained/one-active state was not observed during execution", evidence=result)
        after = suite.runtime("all-resident-after")
        if after["sessions"]["retained"] != capacity:
            raise ValueError("all-resident/one-active workload lost a retained session")
        return {"retained_before": capacity, "retained_after": capacity, "result": result}
    if scenario in ("migration", "oscillation"):
        suite.request(suite.choose("long"), "migration-retained-context")
        initial = suite.runtime("before-migration")
        if initial["sessions"]["retained"] == 0:
            raise Unsupported("migration exercise requires an actually retained supplied long context")
        if initial["execution"].get("migration") is None:
            raise Unsupported("no bounded runtime ownership migration capability")
        if scenario == "oscillation":
            result = suite.concurrent([name] * min(2, capacity), degree_sequence=(1, 4, 2, 4, 1))
            suite.wait_runtime("oscillation-settled", lambda value: migration_done(value, 1))
            return result
        transitions, observed = [], {"partial": False}

        def observe(state):
            observed["partial"] |= any(row.get("pending_attention_degree") is not None
                                       for row in (state.get("placement") or {}).get("slots", []))

        for degree in (1, 2, 1, 4, 2, 4, 1):
            suite.changed_degree = True
            suite.control({"attention_degree": degree})
            state = suite.wait_runtime("migration-progress", lambda value: migration_done(value, degree), observe=observe)
            transitions.append({"degree": degree, "execution_epoch": state["execution"]["epoch"]})
            suite.request(name, "post-migration-request")
        result = {"transitions": transitions, "partial_migration_observed": observed["partial"]}
        if not observed["partial"]:
            raise Unsupported("degree changes completed but partial migration was not observed", evidence=result)
        return result
    if scenario == "limits":
        before = suite.runtime("before-ceiling-change")
        original = before["sessions"]["admission_ceiling"]
        suite.changed_settings = True
        suite.control({"session_ceiling": 1})
        lower = suite.wait_runtime("lower-ceiling", lambda value: value["sessions"]["admission_ceiling"] == 1)
        if lower["sessions"]["retained"] != before["sessions"]["retained"]:
            raise ValueError("ceiling decrease changed retained session ownership")
        suite.control({"session_ceiling": original})
        suite.wait_runtime("raised-ceiling", lambda value: value["sessions"]["admission_ceiling"] == original)
        return {"lowered_to": 1, "restored_to": original, "retained_preserved": before["sessions"]["retained"]}
    if scenario == "zero-proposal":
        if ((suite.initial_runtime.get("capacity_plan") or {}).get("speculation") or {}).get("method") in (None, "none"):
            raise Unsupported("no identified warm speculative method in the current capacity receipt")
        suite.changed_settings = True
        original = suite.initial_runtime["execution"]["effective"]["proposal_limit"]
        suite.control({"proposal_limit": 0})
        suite.wait_runtime("zero-proposal-effective", lambda value: value["execution"]["effective"]["proposal_limit"] == 0)
        zero = suite.request(name, "zero-proposal-request")
        suite.control({"proposal_limit": original})
        suite.wait_runtime("proposal-reactivation", lambda value: value["execution"]["effective"]["proposal_limit"] == original)
        return {"zero_proposal": zero, "warm_reactivation": suite.request(name, "warm-reactivated-request")}
    if scenario == "stop":
        stop = suite.choose("stop")
        if not suite.requests[stop]["body"].get("stop"):
            raise Unsupported("stop role must supply an explicit stopping string")
        result = suite.request(stop, "stop-string-request")
        if result["finish"] != "stop":
            raise Unsupported("supplied stop string was not reached by this request")
        return result
    if scenario == "eos":
        eos = suite.choose("eos")
        body = suite.requests[eos]["body"]
        if body.get("ignore_eos") is True or body.get("stop"):
            raise Unsupported("EOS role must permit model stop tokens without a user stop string")
        result = suite.request(eos, "eos-request")
        if result["finish"] != "stop":
            raise Unsupported("model stop token was not reached within the supplied reply budget", evidence=result)
        return result
    if scenario == "context":
        exact = [case for case, row in suite.requests.items()
                 if row.get("fits") and row["prompt_tokens"] + row["body"]["max_tokens"] == suite.effective["context_tokens"]]
        results = [suite.request(case, "exact-context-request") for case in exact]
        refusal = None
        if "overrun" in suite.requests:
            row = suite.requests["overrun"]
            if row["fits"]:
                raise ValueError("overrun input must exceed the effective prompt-plus-reply window")
            refused = suite.call("context-overrun", row["body"], expected_refusal="context_length_exceeded")
            if type(refused.get("error")) is not dict or refused["error"].get("code") != "context_length_exceeded":
                raise ValueError("context-overrun request was accepted")
            refusal = "HTTP400-context_length_exceeded"
        if not exact or refusal is None:
            raise Unsupported("context gate needs explicit exact-window and overrun requests; completed traces retained")
        if any(result["finish"] != "length" or result["tokens"] != suite.requests[result["request"]]["body"]["max_tokens"]
               for result in results):
            raise Unsupported("exact-window allowance was tested but generation stopped before context exhaustion",
                              evidence={"exact_window_requests": results, "overrun_refusal": refusal})
        return {"exact_window_requests": results, "overrun_refusal": refusal}
    raise ValueError("unknown scenario: " + scenario)


def run(args):
    requests = load_requests(args.request, budget_bytes=args.request_mib << 20)
    if args.out.exists():
        raise ValueError("fresh exercise output directory required")
    args.out.mkdir(parents=True)
    suite, primary = Suite(args, requests), None
    try:
        initial = suite.stats("stats-initial")
        qualification.require_idle(initial)
        suite.effective = qualification.profile(initial)
        suite.report["startup_materialization"] = qualification.materialization(initial, suite.effective)
        suite.initial_runtime = suite.runtime("runtime-initial")
        suite.report.update(initial_stats=initial, initial_runtime=suite.initial_runtime)
        model = suite.effective["model"]
        aliases = [model["id"], *model.get("aliases", [])]
        for name, row in requests.items():
            if row["body"].get("model") not in aliases:
                raise ValueError("supplied model must exactly match a served ID/alias; request files are never rewritten")
            tokenized = suite.call("tokenize-" + name, row["body"], path="/v1/tokenize", object_response=True)
            count, ids = tokenized.get("count"), tokenized.get("tokens")
            if (type(count) is not int or count <= 0 or type(ids) is not list or len(ids) != count
                    or any(type(token) is not int or token < 0 for token in ids)
                    or tokenized.get("max_model_len") != suite.effective["context_tokens"]):
                raise ValueError("tokenizer must report exact IDs/count and the effective context")
            row.update(prompt_tokens=count, fits=count + row["body"]["max_tokens"] <= suite.effective["context_tokens"])
            suite.report["inputs"][name].update(prompt_tokens=count, max_tokens=row["body"]["max_tokens"], fits=row["fits"])
        suite.save()
        for scenario in args.scenarios:
            outcome = {"scenario": scenario, "status": "running"}
            suite.report["scenarios"].append(outcome)
            suite.save()
            try:
                outcome.update(result=exercise_scenario(suite, scenario), status="passed")
            except Unsupported as error:
                outcome.update(status="unsupported", reason=str(error))
                if error.evidence is not None:
                    outcome["observed_evidence"] = error.evidence
            except BaseException as error:
                outcome.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
                raise
            suite.save()
        final = suite.stats("stats-final")
        qualification.require_idle(final)
        suite.report.update(final_stats=final, complete=True,
                            passed=all(row["status"] == "passed" for row in suite.report["scenarios"]))
    except BaseException as error:
        primary = error
        suite.report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        failures = []
        suite.report["unrun_scenarios"] = [scenario for scenario in args.scenarios
                                           if scenario not in {row["scenario"] for row in suite.report["scenarios"]}]
        try:
            suite.restore()
        except BaseException as error:
            failures.append(error)
            suite.report.update(complete=False, passed=False,
                                restoration_error={"type": type(error).__name__, "message": str(error)})
        try:
            suite.save()
        except BaseException as error:
            failures.append(error)
        if failures:
            raise_failures(primary, failures)
    return 0 if suite.report["passed"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--request", action="append", required=True, metavar="NAME=JSON_FILE")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS), help="comma-separated explicit exercises")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--request-mib", type=int, default=64)
    parser.add_argument("--max-inflight", type=int, default=256)
    parser.add_argument("--batch-response-mib", type=int, default=1024)
    parser.add_argument("--max-calls", type=int, default=10000)
    parser.add_argument("--max-cleanup-calls", type=int, default=1000)
    parser.add_argument("--max-events", type=int, default=100000)
    parser.add_argument("--timeout", type=float, default=900, help="socket inactivity timeout, not a whole-request wall guarantee")
    parser.add_argument("--operation-timeout", type=float, default=900,
                        help="cooperative response-body deadline per call/group; one read can overrun by socket timeout")
    parser.add_argument("--poll-timeout", type=float, default=120)
    parser.add_argument("--poll-interval", type=float, default=.1)
    protocol.add_arguments(parser)
    args = parser.parse_args(argv)
    args.scenarios = args.scenarios.split(",")
    if not args.scenarios or len(set(args.scenarios)) != len(args.scenarios) or set(args.scenarios) - set(SCENARIOS):
        parser.error("unique supported scenario names required")
    for key in ("cycles", "request_mib", "max_inflight", "batch_response_mib", "max_calls", "max_cleanup_calls", "max_events"):
        qualification.positive(getattr(args, key), key)
    if any(not math.isfinite(value) or value <= 0 for value in (args.timeout, args.operation_timeout,
                                                               args.poll_timeout, args.poll_interval)):
        parser.error("finite positive timeouts and polling interval required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
