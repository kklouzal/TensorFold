"""Bounded burst measurement, separate from runtime scheduling and model math.

Timestamps describe committed callback delivery, not draft acceptance or GPU
kernel completion. A positive common interval and actual admission observation
are required before a sustained case can claim the offered decode concurrency.
"""
from __future__ import annotations

from array import array
from concurrent.futures import ThreadPoolExecutor
import math
import sys
import threading
import time

from tensorfold.cleanup import error_name, raise_failures

GRID = (1, 2, 4, 8, 16, 24)
MAXIMUM = max(GRID)
STACK_BYTES = 64 << 20
TIMELINE_EVENTS = 8192
POLL_SECONDS = .01
SUSTAINED_MINIMUM = 256
SUSTAINED_PROMPT = (
    "Write a detailed laboratory notebook containing 120 numbered entries about designing, "
    "building, testing, and maintaining a small solar-powered weather station. Each entry "
    "must contain at least two complete explanatory sentences with concrete technical detail. "
    "Do not summarize the remaining entries or use ellipses. Start with entry 1 and continue "
    "in numerical order, explaining each measurement and practical decision."
)


def control_payload_bytes(tokens):
    """Selected CPython payload bound for fixed timing/control cells, not RSS.

    Each token permits one chunk event: two uint64 cells. Each bounded admission
    transition owns N(ticket,phase) tuples plus monotonic time and one bounded
    three-integer decode interval. The256-byte scalar/slot allowance covers
    their selected integer/tuple/list entries together (no tensor owners).
    Futures, locks, threads and their fixed control graphs have the same explicit
    per-slot allowance (4096 scalar/entry cells). Worker C stacks are separate.
    Report objects/encoding/decoded text are funded by the ordinary caller terms.
    """
    if type(tokens) is not int or not 1 <= tokens <= 512:
        raise ValueError("finite timing token bound must be1..512")
    cells = TIMELINE_EVENTS * (2 + 2 * MAXIMUM) + MAXIMUM * 4096
    return MAXIMUM * (2 * sys.getsizeof(array('Q', [0]) * tokens)) + 256 * cells


def input_credit(context):
    # Exact native text input claim formula, using an upper-bound context row
    # count and the target's actual (sub-2**30) token-ID integer representation.
    header = sys.getsizeof([])
    pointer = sys.getsizeof([None]) - header
    return MAXIMUM * (header + (context + 1) * pointer + context * sys.getsizeof(248319))


class TimedTokens:
    """One worker owns IDs and fixed-size chunk cells until its Future retires."""
    def __init__(self, maximum, vocab, abort):
        if type(maximum) is not int or not 1 <= maximum <= 512 or type(vocab) is not int or vocab <= 0:
            raise ValueError("invalid finite token collector domain")
        self.maximum, self.vocab, self.abort = maximum, vocab, abort
        self.ids = []
        self.times = array('Q', [0]) * maximum
        self.counts = array('Q', [0]) * maximum
        self.events = 0
        self.closed = False

    def __call__(self, chunk):
        if self.closed:
            raise RuntimeError("token callback after retirement")
        if (type(chunk) is not list or len(chunk) > self.maximum - len(self.ids)
                or any(type(token) is not int or not 0 <= token < self.vocab for token in chunk)):
            raise ValueError("generation exceeded finite output/token contract")
        if chunk:
            now = time.monotonic_ns()
            self.ids.extend(chunk)
            self.times[self.events] = now
            self.counts[self.events] = len(self.ids)
            self.events += 1
        return self.abort.is_set()

    def close(self):
        self.closed = True

    def record(self):
        return {"token_ids": self.ids, "chunk_monotonic_ns": self.times[:self.events].tolist(),
                "chunk_cumulative_committed_tokens": self.counts[:self.events].tolist()}


def _distribution(values):
    ordered = sorted(values)
    return {"minimum": ordered[0], "p50": ordered[(len(ordered) - 1) // 2],
            "p95": ordered[math.ceil(.95 * len(ordered)) - 1], "maximum": ordered[-1]}


def metrics(rows, started, ended):
    """Independent counts from emitted committed IDs; no draft/stat TPS inputs."""
    complete = [row for row in rows if row["complete"] and row.get("chunk_monotonic_ns")]
    if len(complete) != len(rows):
        return {"measurable": False, "reason": "one or more requests failed or emitted no committed token"}
    begin = max(row["chunk_monotonic_ns"][0] for row in complete)
    end = min(row["chunk_monotonic_ns"][-1] for row in complete)
    totals, rates, ttft, latencies = [], [], [], []
    for row in complete:
        times, counts = row["chunk_monotonic_ns"], row["chunk_cumulative_committed_tokens"]
        common = sum(count - (counts[index - 1] if index else 0)
                     for index, (at, count) in enumerate(zip(times, counts)) if begin < at <= end)
        duration = times[-1] - times[0]
        rate = None if duration <= 0 else (counts[-1] - counts[0]) * 1e9 / duration
        row["timing"] = {"committed_tokens": len(row["token_ids"]),
            "ttft_ns": times[0] - row["offered_monotonic_ns"],
            "delivery_decode_ns": duration, "delivery_decode_tokens": counts[-1] - counts[0],
            "delivery_decode_tps": rate, "common_window_committed_tokens": common,
            "common_window_tps": None if end <= begin else common * 1e9 / (end - begin),
            "whole_job_ns": row["finished_monotonic_ns"] - row["offered_monotonic_ns"]}
        totals.append(common)
        if rate is not None:
            rates.append(rate)
        ttft.append(row["timing"]["ttft_ns"])
        latencies.append(row["timing"]["whole_job_ns"])
    intervals = sorted([(row["chunk_monotonic_ns"][0], 1) for row in complete]
                       + [(row["chunk_monotonic_ns"][-1], -1) for row in complete])
    active = peak = 0
    for _, change in intervals:
        active += change
        peak = max(peak, active)
    return {"measurable": True, "timing_scope": "committed callback delivery; first chunk excluded from decode rate",
        "common_decode_start_monotonic_ns": begin, "common_decode_end_monotonic_ns": end,
        "common_decode_ns": max(0, end - begin), "common_window_tokens_per_request": totals,
        "common_window_aggregate_tps": None if end <= begin else sum(totals) * 1e9 / (end - begin),
        "peak_overlapping_delivery_decode_intervals": peak,
        "whole_wall_ns": ended - started,
        "whole_wall_aggregate_committed_tps": sum(len(row["token_ids"]) for row in complete) * 1e9 / (ended - started),
        "per_request_decode_tps": None if len(rates) != len(rows) else _distribution(rates),
        "decode_rate_jain_fairness": None if len(rates) != len(rows) or not sum(rate * rate for rate in rates)
            else sum(rates) ** 2 / (len(rates) * sum(rate * rate for rate in rates)),
        "ttft_ns": _distribution(ttft), "whole_job_ns": _distribution(latencies),
        "backend_stats_seconds": {key: _distribution([row["stats"][key] for row in complete])
            if all(type(row.get("stats", {}).get(key)) in (int, float)
                and math.isfinite(row["stats"][key]) and row["stats"][key] >= 0 for row in complete) else None
            for key in ("queue_s", "prefill_s", "decode_s")}}



def decoding_tickets(entries, streams):
    """Borrow scalar Stream state only; exclude prefill/wait/done/retained rows.

    These are observations of the actor-owned rows, not a lock or a new model
    scheduling path. Charged admission by itself includes prefill and delivery
    retirement. Identity in the live decode map and committed output distinguish
    actual decoding; no tensor values or GPU synchronization are inspected.
    """
    return tuple(entry[1] for entry in entries
        if (stream := entry[2][0]) is streams.get(stream.sid)
        and not stream.done and not stream.waiting and stream.started > 0
        and stream.committed_output_tokens > 0)


def correlated_overlap(rows, measurement, intervals):
    """Require a measured decode-active interval inside committed delivery.

    Reads are bracketed; the interval starts AFTER the first complete read and
    ends BEFORE the last read. The finite fully resident/no replay role makes
    decode entry->done monotone; observed waiting/requeue is ineligible. These
    are measurement witnesses, never a universal worst-case timing proof. Every
    request must contribute committed output inside the correlated interval.
    """
    if not measurement.get("measurable"):
        return None
    begin, end = measurement["common_decode_start_monotonic_ns"], measurement["common_decode_end_monotonic_ns"]
    selected = None
    for first, last, samples in intervals:
        left, right = max(begin, first), min(end, last)
        if right <= left or samples < 2:
            continue
        counts = [sum(count - (previous[index - 1] if index else 0)
                    for index, (at, count) in enumerate(zip(row["chunk_monotonic_ns"], previous))
                    if left < at <= right)
                  for row in rows for previous in (row["chunk_cumulative_committed_tokens"],)]
        if not all(counts):
            continue
        if selected is None or right - left > selected["duration_ns"]:
            selected = {"start_monotonic_ns": left, "end_monotonic_ns": right,
                "duration_ns": right - left, "observations_in_retained_run": samples,
                "committed_tokens_per_request": counts,
                "aggregate_committed_tps": sum(counts) * 1e9 / (right - left),
                "per_request_committed_tps": [count * 1e9 / (right - left) for count in counts],
                "scope": "correlated sampled live-decode observations and committed callback delivery"}
    return selected

def owned_pool(timeout):
    """Create all N explicitly sized worker stacks after caller admission.

    The process owns this preparation stage; no other thread factory runs while
    the process-global stack-size setting is temporarily changed. No model work
    starts here. On preparation failure the barrier is broken and every started
    worker retires before the error escapes.
    """
    previous = threading.stack_size()
    pool, barrier = None, threading.Barrier(MAXIMUM + 1)
    primary, failures = None, [None, None]
    try:
        threading.stack_size(STACK_BYTES)
        pool = ThreadPoolExecutor(max_workers=MAXIMUM, thread_name_prefix="finite-caller")
        futures = [pool.submit(barrier.wait, timeout) for _ in range(MAXIMUM)]
        barrier.wait(timeout)
        for future in futures:
            future.result()
    except BaseException as error:
        primary = error
    try:
        threading.stack_size(previous)
    except BaseException as error:
        if primary is None:
            primary = error
        else:
            failures[0] = error
    if primary is not None:
        barrier.abort()
        if pool is not None:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except BaseException as error:
                failures[1] = error
    if primary is None:
        return pool
    try:
        errors = [error for error in failures if error is not None]
    except BaseException as allocation:
        raise primary from allocation
    raise_failures(primary, errors)


def run_case(pool, *, count, maximum_tokens, vocab, limit, invoke, observe, write, minimum_tokens, timeout):
    """Offer exactly N jobs, drain ALL siblings, persist raw outcomes, then gate.

    invoke runs canonical Engine.generate and presentation decode while the
    worker's real RequestLimit owner remains live. observe copies scalar queue
    state only. No runtime timing, status, cache or scheduling is changed here.
    """
    if count not in GRID or minimum_tokens not in (1, SUSTAINED_MINIMUM):
        raise ValueError("unsupported finite case")
    abort, barrier = threading.Event(), threading.Barrier(count + 1)
    collectors = [TimedTokens(maximum_tokens, vocab, abort) for _ in range(count)]
    rows = [{"request": index, "complete": False, "cancel_requested": False} for index in range(count)]
    errors, futures, drain_errors = [None] * count, [None] * count, [None] * count
    diagnostic_errors = [None] * count
    timeline, previous, peak = [], None, 0
    decode_intervals, current_decode_interval = [], None
    initial = observe()
    started = time.monotonic_ns()
    primary = None

    def work(index):
        row, collector = rows[index], collectors[index]
        try:
            barrier.wait(timeout)
            row["offered_monotonic_ns"] = time.monotonic_ns()
            with limit.request():
                row["logical_admitted_monotonic_ns"] = time.monotonic_ns()
                def progress(total, cached, processed, milliseconds):
                    # Scheduler progress follows the token callback convention:
                    # True requests cancellation; False continues prefill.
                    row["prefill_progress_callbacks"] = row.get("prefill_progress_callbacks", 0) + 1
                    row["last_prefill_progress"] = {"total": total, "cached": cached,
                        "processed": processed, "backend_elapsed_ms": milliseconds,
                        "observed_monotonic_ns": time.monotonic_ns()}
                    return abort.is_set()
                row.update(invoke(collector, progress))
            row["complete"] = not abort.is_set()
            row["cancel_requested"] = abort.is_set()
        except BaseException as error:
            errors[index] = error
            # Cancellation/notification must precede fallible diagnostics.
            abort.set()
            barrier.abort()
            try:
                row["failure"] = {"type": error_name(error), "message": str(error)}
            except BaseException as diagnostic:
                diagnostic_errors[index] = diagnostic
        finally:
            row["finished_monotonic_ns"] = time.monotonic_ns()
            collector.close()

    try:
        for index in range(count):
            futures[index] = pool.submit(work, index)
        barrier.wait(timeout)
        while not all(future.done() for future in futures):
            observation_started = time.monotonic_ns()
            observed = observe()
            states = observed["states"]
            at = time.monotonic_ns()
            if observed.get("nonmonotone_decode_states"):
                raise ValueError("finite decode observation encountered waiting/requeued live state")
            peak = max(peak, sum(phase == "active" for _, phase in states))
            if len(observed["decoding"]) == count:
                if current_decode_interval is None:
                    if len(decode_intervals) >= TIMELINE_EVENTS:
                        raise ValueError("finite decode observation intervals exceeded declared cells")
                    current_decode_interval = [at, observation_started, 1]
                    decode_intervals.append(current_decode_interval)
                else:
                    current_decode_interval[1] = observation_started
                    current_decode_interval[2] += 1
            else:
                current_decode_interval = None
            if states != previous:
                if len(timeline) >= TIMELINE_EVENTS:
                    raise ValueError("finite admission timeline exceeded declared cells")
                timeline.append((time.monotonic_ns(), states))
                previous = states
            if time.monotonic_ns() - started > timeout * 1e9:
                raise TimeoutError("finite burst deadline elapsed; sibling drain required")
            abort.wait(POLL_SECONDS) if not abort.is_set() else time.sleep(POLL_SECONDS)
    except BaseException as error:
        primary = error
        abort.set()
        barrier.abort()
    finally:
        # Future cancellation is only a request. result() proves every started
        # callback/engine scope has retired before result graphs are reused.
        for index, future in enumerate(futures):
            if future is not None:
                if abort.is_set():
                    future.cancel()
                if not future.cancelled():
                    while True:
                        try:
                            future.result()
                            break
                        except BaseException as error:
                            if drain_errors[index] is None:
                                drain_errors[index] = error
                            abort.set()
                            barrier.abort()
                            # A caller interrupt can leave Future.result before
                            # completion. It requests cancellation, not worker
                            # retirement: keep waiting before closing callbacks.
                            if future.done():
                                break
        for collector in collectors:
            collector.close()
    if primary is None:
        for error in errors:
            if error is not None:
                primary = error
                break
    if primary is None:
        for error in drain_errors:
            if error is not None:
                primary = error
                break
    try:
        ended = time.monotonic_ns()
        final = observe()
        for row, collector in zip(rows, collectors):
            row.update(collector.record())
        result = {"schema": "tensorfold.finite-concurrency-case-v2", "offered": count,
            "maximum_funded_callers": MAXIMUM, "logical_admitted": sum("logical_admitted_monotonic_ns" in row for row in rows),
            "completed": sum(row["complete"] for row in rows), "errors": sum(error is not None for error in errors),
            "cancel_requested": sum(row["cancel_requested"] for row in rows),
            "peak_observed_backend_active": peak, "admission_poll_seconds": POLL_SECONDS,
            "timeline": timeline, "decode_active_observed_intervals": decode_intervals,
            "diagnostic_failure_types": [error_name(error) if error is not None else None for error in diagnostic_errors],
            "initial_queue": initial, "final_queue": final,
            "controller_failure_type": None if primary is None else error_name(primary),
            "retirement_failures": [{"type": type(error).__name__, "message": str(error)} for error in drain_errors if error is not None],
            "backend_outcome_delta": {key: final["outcomes"][key] - value for key, value in initial["outcomes"].items()},
            "requests": rows, "metrics": metrics(rows, started, ended), "passed": False}
        if primary is None and (limit.used or final["states"]):
            primary = RuntimeError("finite jobs retired with logical/backend ownership still live")
        if primary is None and (result["logical_admitted"] != count or result["completed"] != count
                or result["backend_outcome_delta"].get("admitted") != count
                or any(value for key, value in result["backend_outcome_delta"].items() if key != "admitted")):
            primary = ValueError("finite offered/admitted/completed outcomes differ")
        measured = result["metrics"]
        result["correlated_decode_overlap"] = correlated_overlap(rows, measured, decode_intervals)
        if primary is None and (not measured["measurable"] or any(len(row["token_ids"]) < minimum_tokens for row in rows)):
            primary = ValueError("natural output ended before the declared measurement length")
        if primary is None and minimum_tokens == SUSTAINED_MINIMUM and result["correlated_decode_overlap"] is None:
            primary = ValueError("offered concurrency did not establish actual all-request decode overlap")
        result["passed"] = primary is None
        try:
            write(result)
        except BaseException as diagnostic:
            if primary is not None:
                raise primary from diagnostic
            raise
        if primary is not None:
            for diagnostic in diagnostic_errors:
                if diagnostic is not None:
                    raise primary from diagnostic
            raise primary
        return {key: result[key] for key in ("offered", "completed", "peak_observed_backend_active", "metrics", "passed")}
    except BaseException as diagnostic:
        if primary is not None and diagnostic is not primary:
            raise primary from diagnostic
        raise
