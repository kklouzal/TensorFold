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


FAST_READY_SECONDS = 30.0
FAST_DECODE_SECONDS = 10.0


class ScreenTokens(TimedTokens):
    """Publish first delivery once; distinguish observed stop from natural EOS."""
    def __init__(self, maximum, vocab, stop):
        super().__init__(maximum, vocab, stop)
        self.first_ready = threading.Event()
        self.stop_observed = False
        self.native_stream = None

    def bind_native_stream(self, stream):
        if self.native_stream is not None and self.native_stream is not stream:
            raise RuntimeError("screen callback changed its native Stream owner")
        self.native_stream = stream

    def native_record(self, stats):
        """Read only the retired Stream's bounded CPU output, never its State.

        Scheduler suppresses further callback deliveries once cancellation is
        observed. This retains the original native IDs too, including in-flight
        commits with no delivery timestamp. Such residuals never enter windowTPS.
        """
        if not self.closed:
            raise RuntimeError("native output cannot be read before callback retirement")
        stream = self.native_stream
        try:
            if stream is None:
                committed = stats.get("committed_output_tokens")
                if type(committed) is not int or not 0 <= committed <= self.maximum:
                    return {"native_token_ids": None, "native_committed_tokens": None,
                        "native_undelivered_tokens": None,
                        "native_trace_source": "unavailable: no retired Stream or exact native committed count; callback evidence retained"}
                if committed != len(self.ids):
                    raise RuntimeError("native committed IDs unavailable after suppressed delivery")
                return {"native_token_ids": list(self.ids), "native_committed_tokens": committed,
                    "native_undelivered_tokens": 0, "native_trace_source": "complete callback IDs/count agreement"}
            out, committed = stream.out, stream.committed_output_tokens
            if (type(out) is not list or len(out) > self.maximum or type(committed) is not int
                    or committed != len(out) or any(type(token) is not int or not 0 <= token < self.vocab for token in out)):
                raise ValueError("retired native Stream exceeds the finite committed-output contract")
            ids = list(out)
            if ids[:len(self.ids)] != self.ids:
                raise ValueError("callback delivery is not a prefix of the native committed IDs")
            return {"native_token_ids": ids, "native_committed_tokens": committed,
                "native_undelivered_tokens": committed - len(self.ids),
                "native_undelivered_commit_timestamps": None,
                "native_trace_source": "retired owned Stream.out; undelivered IDs untimed/excluded windowTPS"}
        finally:
            self.native_stream = None  # Retire the Stream/State aliases before another cohort.

    def __call__(self, chunk):
        stop = super().__call__(chunk)
        if self.events and not self.first_ready.is_set():
            self.first_ready.set()
        if stop:
            self.stop_observed = True
        return stop


def screen_control_payload_bytes():
    # Additional Event/Condition/lock/deque/flags, one owned Stream reference
    # and up to two512-pointer snapshots per worker, plus coordinator cells.
    # These fit64 selected-runtime256-byte cells per worker; timing/IDs from
    # callback delivery and all report graphs have their existing terms.
    return (MAXIMUM * 64 + 64) * 256


def run_screen_case(pool, *, count, maximum_tokens, vocab, limit, invoke, observe, write,
                    timeout, readiness_seconds=FAST_READY_SECONDS, decode_seconds=FAST_DECODE_SECONDS,
                    failure_notify=None):
    """Bounded time-window screen; partial output never becomes a full reply.

    This is a distinct operational protocol. The full run_case remains intact.
    Only the coordinator requests the expected stop, after freezing its exact
    measurement interval. Worker callbacks/progress acknowledge that request;
    every Future/real permit/backend claim retires before the next cohort.
    """
    if (count not in GRID or type(maximum_tokens) is not int or maximum_tokens not in (320, 512)
            or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= threading.TIMEOUT_MAX
                   for value in (timeout, readiness_seconds, decode_seconds))):
        raise ValueError("invalid fast screen bounds")
    stop, failed, barrier = threading.Event(), threading.Event(), threading.Barrier(count + 1)
    collectors = [ScreenTokens(maximum_tokens, vocab, stop) for _ in range(count)]
    rows = [{"request": index, "complete": False, "backend_retired": False,
             "intentional_cancelled": False, "quality_assessed": False} for index in range(count)]
    errors, drains, futures = [None] * count, [None] * count, [None] * count
    notification_errors = [None] * (count + 1)
    timeline, intervals, previous, active_interval = [], [], None, None
    screen = {"protocol": "fast-grid-v1", "target_decode_seconds": decode_seconds,
        "readiness_cap_seconds": readiness_seconds, "maximum_output_tokens": maximum_tokens,
        "begin_monotonic_ns": None, "end_monotonic_ns": None,
        "cancel_requested_monotonic_ns": None, "stop_reason": None,
        "full_qualification": False, "quality_assessed": False}
    initial = observe()
    started, peak, peak_decode, primary = time.monotonic_ns(), 0, 0, None

    def notify(index):
        if failure_notify is not None:
            try:
                if not failure_notify():
                    raise RuntimeError("fast controller lost its supervised failure capability")
            except BaseException as error:
                notification_errors[index] = error

    def cancel(reason, end):
        # Freeze both endpoints before any callback can observe stop. Chunks
        # arriving after end remain raw evidence, outside measured decode TPS.
        if screen["stop_reason"] is None:
            screen["end_monotonic_ns"] = end
            screen["stop_reason"] = reason
            screen["cancel_requested_monotonic_ns"] = time.monotonic_ns()
            stop.set()

    def work(index):
        row, collector = rows[index], collectors[index]
        try:
            barrier.wait(min(30, timeout))
            row["offered_monotonic_ns"] = time.monotonic_ns()
            with limit.request():
                row["logical_admitted_monotonic_ns"] = time.monotonic_ns()
                def progress(total, cached, processed, milliseconds):
                    row["prefill_progress_callbacks"] = row.get("prefill_progress_callbacks", 0) + 1
                    row["last_prefill_progress"] = {"total": total, "cached": cached, "processed": processed,
                        "backend_elapsed_ms": milliseconds, "observed_monotonic_ns": time.monotonic_ns()}
                    value = stop.is_set()
                    if value:
                        collector.stop_observed = True
                    return value  # The canonical Scheduler cancels on True.
                row.update(invoke(collector, progress))
            row["backend_retired"] = True
            row["intentional_cancelled"] = collector.stop_observed and not failed.is_set()
            row["complete"] = not collector.stop_observed and not failed.is_set()
            row["termination"] = ("intentional_screen_cancel" if row["intentional_cancelled"] else
                                  "sibling_failure_cancel" if collector.stop_observed else row.get("natural_stop", "natural_return"))
        except BaseException as error:
            errors[index] = error
            failed.set()
            stop.set()
            barrier.abort()
            notify(index)  # Parent containment observes faults before any sibling can block drain.
        finally:
            row["finished_monotonic_ns"] = time.monotonic_ns()
            collector.close()

    try:
        for index in range(count):
            futures[index] = pool.submit(work, index)
        barrier.wait(min(30, timeout))
        while not all(future.done() for future in futures):
            read_started = time.monotonic_ns()
            observed = observe()
            at = time.monotonic_ns()
            states, decoding = observed["states"], observed["decoding"]
            peak = max(peak, sum(phase == "active" for _, phase in states))
            peak_decode = max(peak_decode, len(decoding))
            if observed.get("nonmonotone_decode_states") and not stop.is_set():
                raise ValueError("fast fixed-resident screen encountered waiting/requeued decode state")
            if len(decoding) == count and not stop.is_set():
                if active_interval is None:
                    if len(intervals) >= TIMELINE_EVENTS:
                        raise ValueError("fast decode interval storage exhausted")
                    active_interval = [at, read_started, 1]
                    intervals.append(active_interval)
                else:
                    active_interval[1], active_interval[2] = read_started, active_interval[2] + 1
            elif not stop.is_set():
                if screen["begin_monotonic_ns"] is not None:
                    cancel("natural_decode_membership_end", active_interval[1] if active_interval is not None else read_started)
                active_interval = None
            if states != previous:
                if len(timeline) >= TIMELINE_EVENTS:
                    raise ValueError("fast admission timeline storage exhausted")
                timeline.append((at, states))
                previous = states
            if not stop.is_set():
                if screen["begin_monotonic_ns"] is None:
                    if (active_interval is not None and active_interval[2] >= 2
                            and active_interval[1] > active_interval[0]
                            and all(collector.first_ready.is_set() for collector in collectors)):
                        screen["begin_monotonic_ns"] = time.monotonic_ns()
                    elif at - started >= readiness_seconds * 1e9:
                        cancel("readiness_timeout", None)
                else:
                    target_end = screen["begin_monotonic_ns"] + int(decode_seconds * 1e9)
                    if at >= target_end and active_interval is not None and active_interval[1] >= target_end:
                        # The read START must confirm all-N activity through
                        # target_end. A slow observation ending after the target
                        # is insufficient; wait for the next confirming read.
                        cancel("decode_window_elapsed", target_end)
                    elif any(future.done() for future in futures):
                        cancel("natural_reply_end", active_interval[1] if active_interval is not None else read_started)
            if at - started > timeout * 1e9:
                raise TimeoutError("fast screen overall case deadline elapsed; worker drain required")
            time.sleep(POLL_SECONDS)
    except BaseException as error:
        primary = error
        failed.set()
        stop.set()
        barrier.abort()
        notify(count)
    finally:
        # A requested cancellation/timeout is not retirement. Interrupted waits
        # keep waiting until each actual Future has completed; no buffer reuse.
        for index, future in enumerate(futures):
            if future is None:
                continue
            if failed.is_set():
                future.cancel()
            if not future.cancelled():
                while True:
                    try:
                        future.result()
                        break
                    except BaseException as error:
                        if drains[index] is None:
                            drains[index] = error
                        failed.set()
                        stop.set()
                        barrier.abort()
                        notify(count)
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
        for error in drains:
            if error is not None:
                primary = error
                break
    post_diagnostic, summary = None, None
    try:
        retired = time.monotonic_ns()
        final = observe()
        drained = (not limit.used and not final["states"] and final["backend_live"] == 0
            and final["backend_retiring"] == 0 and final["backend_mailboxes"] == 0
            and all(future is not None and future.done() for future in futures))
        if primary is None and not drained:
            primary = RuntimeError("fast screen retained worker/request/backend owners after drain")
        for row, collector in zip(rows, collectors):
            row.update(collector.record())
            if drained:
                row.update(collector.native_record(row.get("stats", {})))
            else:
                row["native_trace_unavailable"] = "backend retirement was not proved; native output not inspected"
        if primary is not None:
            screen["stop_reason"] = "runtime_or_controller_failure"
            screen["end_monotonic_ns"] = screen["begin_monotonic_ns"]
        if screen["stop_reason"] is None:
            screen["stop_reason"] = "natural_replies_before_window_end"
            screen["end_monotonic_ns"] = None if screen["begin_monotonic_ns"] is None else min(
                [intervals[-1][1] if intervals else retired]
                + [row["chunk_monotonic_ns"][-1] for row in rows if row["chunk_monotonic_ns"]])
        begin, end = screen["begin_monotonic_ns"], screen["end_monotonic_ns"]
        screen["readiness_ns"] = None if begin is None else begin - started
        screen["duration_ns"] = 0 if begin is None or end is None else max(0, end - begin)
        screen["short_window"] = screen["duration_ns"] < int(decode_seconds * 1e9)
        screen["retired_monotonic_ns"] = retired
        requested = screen["cancel_requested_monotonic_ns"]
        screen["cancel_and_drain_ns"] = None if requested is None else retired - requested
        screen["whole_case_ns"] = retired - started
        screen["readiness_or_drain_in_decode_tps"] = False
        # The metrics view asserts only that emitted data is known AFTER real
        # retirement. Actual rows retain complete=False for intentional stops.
        views = [{**row, "complete": row["backend_retired"] and errors[index] is None} for index, row in enumerate(rows)]
        delivery = metrics(views, started, retired)
        correlated = None
        if delivery.get("measurable") and screen["duration_ns"] > 0:
            clipped = {**delivery, "common_decode_start_monotonic_ns": max(begin, delivery["common_decode_start_monotonic_ns"]),
                "common_decode_end_monotonic_ns": min(end, delivery["common_decode_end_monotonic_ns"])}
            correlated = correlated_overlap(rows, clipped, intervals)
        for row, view in zip(rows, views):
            if "timing" in view:
                row["delivery_timing_diagnostic"] = view["timing"]
            counts, times = row["chunk_cumulative_committed_tokens"], row["chunk_monotonic_ns"]
            measured = sum(value - (counts[index - 1] if index else 0) for index, (at, value) in enumerate(zip(times, counts))
                           if begin is not None and end is not None and begin < at <= end)
            row["measured_window_committed_tokens"] = measured
            row["committed_tokens_after_window"] = sum(value - (counts[index - 1] if index else 0)
                for index, (at, value) in enumerate(zip(times, counts)) if end is not None and at > end)
            row["measured_window_tps"] = None if screen["duration_ns"] <= 0 else measured * 1e9 / screen["duration_ns"]
        outcome = {key: final["outcomes"][key] - value for key, value in initial["outcomes"].items()}
        screen["native_commit_trace_complete"] = all(type(row.get("native_committed_tokens")) is int
            and type(row.get("native_token_ids")) is list for row in rows)
        valid = (primary is None and drained and screen["native_commit_trace_complete"] and outcome.get("admitted") == count
            and outcome.get("refused", 0) == outcome.get("expired", 0) == 0
            and screen["duration_ns"] > 0 and correlated is not None
            and all(row["measured_window_committed_tokens"] > 0 for row in rows))
        screen["measurement_valid"] = valid
        screen["window_aggregate_committed_tps"] = None if screen["duration_ns"] <= 0 else sum(
            row["measured_window_committed_tokens"] for row in rows) * 1e9 / screen["duration_ns"]
        screen["correlated_all_decode_interval"] = correlated
        screen["all_workers_permits_backend_owners_drained"] = drained
        result = {"schema": "tensorfold.fast-grid-case-v1", "measurement_role": "operational screen; not full qualification or answer quality",
            "offered": count, "logical_admitted": sum("logical_admitted_monotonic_ns" in row for row in rows),
            "backend_outcome_delta": outcome, "completed": sum(row["complete"] for row in rows),
            "intentionally_cancelled": sum(row["intentional_cancelled"] for row in rows),
            "errors": sum(error is not None for error in errors), "peak_observed_backend_active": peak,
            "peak_observed_decoding": peak_decode, "initial_queue": initial, "final_queue": final,
            "timeline": timeline, "decode_active_observed_intervals": intervals, "screen": screen,
            "requests": rows, "delivery_metrics_diagnostic": delivery, "passed": valid,
            "failure_types": [None if error is None else error_name(error) for error in errors],
            "failure_notification_secondary_types": [None if error is None else error_name(error) for error in notification_errors]}
        write(result)
        summary = {key: result[key] for key in ("offered", "completed", "intentionally_cancelled", "passed", "screen")}
    except BaseException as diagnostic:
        if primary is None:
            primary = diagnostic
        else:
            post_diagnostic = diagnostic
    if primary is not None:
        try:
            secondary = [error for error in notification_errors if error is not None]
            if post_diagnostic is not None:
                secondary.append(post_diagnostic)
        except BaseException as allocation:
            raise primary from allocation
        raise_failures(primary, secondary)
    return summary
