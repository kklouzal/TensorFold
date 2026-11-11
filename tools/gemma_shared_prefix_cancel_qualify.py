"""Private-arm observed-active Gemma cancellation; no server instrumentation.

Historical original SHA ac3c93b4df4354d0a8cbf7fd5fb9d666b4a950f301a0d66b44545cc8faf8330f.

Decode: visible generated delta + increased health token counter + running.
Prefill: native return_progress interior chunk + running before any output.
These are wire/server observations, never GPU-inflight or physical-retirement proof.
"""

from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

BUDGET = 4096
MAX_OBSERVED_BEFORE_CLOSE = 256
MAX_EVENTS = 1024
STREAM_BYTES = 4 << 20
HEALTH_BYTES = 64 << 10

if __package__:
    from . import gemma_shared_prefix_qualify as core
    from . import openai_protocol as protocol
    from .v100_rotorquant_live import Client, RecordedResponse, ReadControl, idle
    from .benchmark_output import write_json
else:
    import gemma_shared_prefix_qualify as core
    import openai_protocol as protocol
    from v100_rotorquant_live import Client, RecordedResponse, ReadControl, idle
    from benchmark_output import write_json


class Inconclusive(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_deadline(deadline, now):
    if now >= deadline:
        raise TimeoutError("declared cooperative observation/cleanup deadline exceeded")


def trigger_event(phase, event, pieces):
    if event.get("usage") is not None or any(
        choice.get("finish_reason") for choice in event.get("choices", [])
    ):
        raise Inconclusive("generation finished before an observed-active disconnect")
    if phase == "decode":
        return any(pieces)
    progress = event.get("prompt_progress")
    if progress is not None:
        require(
            type(progress) is dict
            and all(
                type(progress.get(key)) is int
                for key in ("total", "cache", "processed")
            ),
            "typed native prefill progress required",
        )
        require(
            0 <= progress["cache"] <= progress["processed"] <= progress["total"],
            "invalid progress ordering",
        )
        return progress["cache"] < progress["processed"] < progress["total"]
    if any(pieces):
        raise Inconclusive(
            "prefill completed before a wire-visible interior progress observation"
        )
    return False


def validate_active(phase, before, health, event, pieces):
    require(health.get("ok") is True, "production health error during stream")
    if (
        not health.get("busy")
        or type(health.get("requests_running")) is not int
        or health["requests_running"] < 1
    ):
        raise Inconclusive("stream was no longer running at the active observation")
    require(
        health.get("requests_total") == before.get("requests_total"),
        "another request completed during isolated stream",
    )
    require(
        type(health.get("completion_tokens_total")) is int
        and type(before.get("completion_tokens_total")) is int,
        "typed live generated-token counters required",
    )
    delta = health["completion_tokens_total"] - before["completion_tokens_total"]
    if phase == "decode":
        require(any(pieces), "visible emitted text delta required")
        if not 1 <= delta <= MAX_OBSERVED_BEFORE_CLOSE:
            raise Inconclusive(
                "visible decode was not observed well before the4096-token budget"
            )
        return {
            "phase": "decode",
            "visible_text": True,
            "server_generated_tokens_since_before": delta,
        }
    require(
        trigger_event("prefill", event, pieces)
        and event["prompt_progress"]["total"] == 65536,
        "interior native progress of the actual65536-ID request required",
    )
    if delta != 0:
        raise Inconclusive(
            "output already began before the prefill disconnect observation"
        )
    return {
        "phase": "prefill",
        "native_prompt_progress": event["prompt_progress"],
        "server_generated_tokens_since_before": 0,
    }


def retirement_state(before, after):
    prior = before.get("totals", {}).get("cancelled")
    current = after.get("totals", {}).get("cancelled")
    require(
        type(prior) is int and type(current) is int,
        "typed cancellation counters required",
    )
    require(
        current <= prior + 1,
        "unexpected extra cancellation in isolated observed request",
    )
    return idle(after) and current == prior + 1


def validate_recovery(reply):
    message = reply["choices"][0]["message"]
    require(
        (message.get("content") or "").strip() == "4"
        and reply["choices"][0]["finish_reason"] == "stop",
        "consumed recovery answer/normal EOS failed",
    )


def validate_cache_settled(stats, arm):
    cache = stats["prefix_cache"]
    require(
        type(cache.get("charged_bytes")) is int
        and type(cache.get("byte_budget")) is int
        and 0
        <= cache["charged_bytes"] + cache.get("failed_snapshot_bytes", 0)
        <= cache["byte_budget"],
        "observed cache byte credit invalid",
    )
    if arm == "candidate":
        require(
            cache.get("store_schema") == "gemma-canonical-block-archive-v1"
            and all(
                key in cache and cache[key] is None
                for key in ("entries", "max_entries", "capacity", "tokens")
            ),
            "candidate archive schema/nullable counts invalid",
        )
        require(
            cache.get("metadata_status") == "ready", "candidate cache metadata failed"
        )
        for key in (
            "pins",
            "pending_batches",
            "reserved_payload_bytes",
            "uncertain_payload_bytes",
            "temporary_bytes",
            "metadata_owners",
            "host_metadata_claim_bytes",
        ):
            require(
                type(cache.get(key)) is int and cache[key] == 0,
                "cache obligations not positively settled: " + key,
            )
    return cache


def observe_and_disconnect(
    args, body, path, before, protocol, RecordedResponse, ReadControl, write_json, evidence
):
    deadline = time.monotonic() + args.observe_seconds
    control = ReadControl(deadline)
    request = urllib.request.Request(
        args.url + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    response = None
    close_attempted = False
    buffer = io.BytesIO()
    record = {
        "events": [],
        "health_observations": [],
        "request": body,
        "path": path,
        "phase": args.phase,
        "budget": BUDGET,
        "maximum_generated_at_observation": MAX_OBSERVED_BEFORE_CLOSE,
    }
    primary = None
    errors = [None, None]
    try:
        response = urllib.request.urlopen(
            request, timeout=control.acquisition_timeout(args.socket_timeout)
        )
        wrapped = RecordedResponse(response, buffer, STREAM_BYTES, control)
        protocol.response_type(wrapped, "text/event-stream")
        record["headers"] = list(response.headers.items())
        for event in protocol.sse_objects(
            wrapped, limits=protocol.Limits(STREAM_BYTES, 65536)
        ):
            require(
                len(record["events"]) < MAX_EVENTS,
                "bounded SSE observation count exceeded",
            )
            record["events"].append(event)
            _usage, _runtime, pieces = protocol.chunk_fields(event)
            if not trigger_event(args.phase, event, pieces):
                continue
            # Health stays in bounded memory: no disk publication can delay the
            # immediate close after this last running/visible observation.
            raw_health = io.BytesIO()
            health_request = urllib.request.Request(args.url + "/health")
            with urllib.request.urlopen(
                health_request, timeout=control.acquisition_timeout(args.socket_timeout)
            ) as health_response:
                health_reader = RecordedResponse(
                    health_response, raw_health, HEALTH_BYTES, control
                )
                protocol.response_type(health_reader, "application/json")
                while health_reader.read1(4096):
                    pass
                health = protocol.object_json(raw_health.getvalue())
            record["health_observations"].append(
                {
                    "monotonic_s": time.monotonic(),
                    "body": health,
                    "raw_utf8": raw_health.getvalue().decode(),
                }
            )
            record["active_observation"] = validate_active(
                args.phase, before, health, event, pieces
            )
            control.check()
            record["close_started_monotonic_s"] = time.monotonic()
            close_attempted = True
            response.close()
            record["close_returned_monotonic_s"] = time.monotonic()
            record["intentionally_incomplete"] = True
            return record
        raise Inconclusive("SSE completed before an observed-active disconnect trigger")
    except BaseException as error:
        primary = error
        _operation_cause = BaseException.__cause__.__get__(error)
        _operation_context = BaseException.__context__.__get__(error)
        _operation_suppress = BaseException.__suppress_context__.__get__(error)
        raise
    finally:
        if response is not None and not close_attempted:
            try:
                close_attempted = True
                response.close()
            except BaseException as error:
                errors[0] = error
        if primary is not None:
            BaseException.__cause__.__set__(primary, _operation_cause)
            BaseException.__context__.__set__(primary, _operation_context)
            BaseException.__suppress_context__.__set__(primary, _operation_suppress)
        try:
            evidence.raw("stream-response.raw", buffer.getvalue())
            write_json(args.output / "stream-observations.json", record)
        except BaseException as error:
            errors[1] = error
        core.finish_errors(primary, errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-uid", required=True)
    parser.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--socket-timeout", type=float, default=5.0)
    parser.add_argument("--observe-seconds", type=float, default=60.0)
    parser.add_argument("--cleanup-seconds", type=float, default=60.0)
    parser.add_argument("--prefill-token", type=int, default=239)
    args = parser.parse_args(argv)
    parsed = urllib.parse.urlsplit(args.url)
    require(
        parsed.scheme == "http"
        and parsed.hostname in ("127.0.0.1", "localhost")
        and parsed.port is not None
        and parsed.port != 8888
        and parsed.username is None
        and parsed.password is None
        and not parsed.path.strip("/")
        and not parsed.query
        and not parsed.fragment,
        "explicit private loopback endpoint required; public8888 forbidden; executor binds CID/image/source/config",
    )
    require(
        0 < args.socket_timeout <= 10
        and 0 < args.observe_seconds <= 120
        and 0 < args.cleanup_seconds <= 120,
        "bounded declared socket/cooperative timeouts required",
    )
    require(
        len(args.expected_source_uid) == 64
        and all(c in "0123456789abcdef" for c in args.expected_source_uid),
        "complete expected source UID required",
    )
    require(
        0 <= args.prefill_token < 262144
        and args.prefill_token not in (1, 2, 50, 98, 100, 101, 106),
        "ordinary synthetic prefill token required",
    )
    evidence = core.Evidence(args.output)
    client = Client(
        args.url,
        args.output,
        timeout=args.socket_timeout,
        limits=protocol.Limits(STREAM_BYTES, 65536),
    )
    report = {
        "status": "running",
        "phase": args.phase,
        "limits": {
            "socket_inactivity_s": args.socket_timeout,
            "observation_cooperative_s": args.observe_seconds,
            "retirement_cooperative_s": args.cleanup_seconds,
            "stream_bytes": STREAM_BYTES,
            "health_bytes": HEALTH_BYTES,
            "sse_events": MAX_EVENTS,
            "external_bound": "Executor must impose a whole-process deadline; socket inactivity and inter-read checks are not a hard whole-operation bound.",
        },
        "scope": "Visible output/interior server prefill progress plus running health, cancellation counter+idle and consumed recovery. No independent GPU-inflight or physical-retirement claim.",
    }

    def save():
        write_json(args.output / "summary.json", report)

    primary = None
    errors = [None, None]
    try:
        before = client.call("stats-before")
        health = client.call("health-before", path="/health")
        require(health.get("ok") is True, "healthy production required")
        if not idle(before) or health.get("busy") is not False:
            raise Inconclusive("production was not naturally idle; no stream started")
        revision = before.get("server", {}).get("source_revision", "")
        require(
            args.expected_source_uid[:16] in revision,
            "production source revision differs",
        )
        report.update(before_stats=before, before_health=health)
        base = {
            "model": "gemma4-26b-a4b",
            "temperature": 0,
            "max_tokens": BUDGET,
            "ignore_eos": True,
            "draft": True,
            "stream": True,
            "return_token_ids": True,
            "stream_options": {"include_usage": True},
            "return_progress": True,
        }
        if args.phase == "decode":
            body = {
                **base,
                "chat_template_kwargs": {"enable_thinking": True},
                "messages": [
                    {
                        "role": "user",
                        "content": "List positive integers in ascending order separated by spaces. Continue without commentary.",
                    }
                ],
            }
            path = "/v1/chat/completions"
        else:
            body = {**base, "prompt": [args.prefill_token] * 65536}
            path = "/v1/completions"
        report["request"] = body
        save()
        report["disconnect"] = observe_and_disconnect(
            args,
            body,
            path,
            health,
            protocol,
            RecordedResponse,
            ReadControl,
            write_json,
            evidence,
        )
        deadline = time.monotonic() + args.cleanup_seconds
        attempt = 0
        while True:
            check_deadline(deadline, time.monotonic())
            require(attempt < 512, "bounded retirement poll count exceeded")
            control = ReadControl(deadline)
            state = client.call(
                "stats-retirement-" + str(attempt), read_control=control
            )
            attempt += 1
            if retirement_state(before, state):
                break
            time.sleep(0.1)
        validate_cache_settled(state, args.arm)
        report.update(cancelled_idle=state, retirement_polls=attempt)
        recovery = client.call(
            "recovery",
            {
                "model": "gemma4-26b-a4b",
                "messages": [
                    {"role": "user", "content": "Return only the answer to 2+2."}
                ],
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0,
                "max_tokens": 32,
                "draft": True,
                "return_token_ids": True,
            },
        )
        validate_recovery(recovery)
        final = client.call("stats-final")
        require(idle(final), "recovery did not leave production idle")
        validate_cache_settled(final, args.arm)
        report.update(
            status="passed-observed-cancel-recovery",
            recovery=recovery,
            final_stats=final,
        )
    except BaseException as error:
        primary = error
        try:
            report["status"] = "inconclusive" if isinstance(error, Inconclusive) else "failed"
            report["error"] = {"type": type(error).__name__, "message": str(error)[:4096]}
        except BaseException as diagnostic:
            errors[0] = diagnostic
    try:
        save()
    except BaseException as publication:
        errors[1] = publication
    core.finish_errors(primary, errors)


if __name__ == "__main__":
    main()
