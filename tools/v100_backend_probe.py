#!/usr/bin/env python3
"""Finite SC117 IQ3_S/Q8-MTP TP8 diagnostic through the ordinary CUDA backend.

This produces bounded text and raw token/statistics receipts. It does not qualify
HTTP serving, full-context workloads, numerical quality or throughput. Run only
the reviewed installed wheel, with all eight V100s exclusively available.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time
import traceback
import zipfile

from tensorfold.cleanup import raise_failures

if __package__:
    from .benchmark_output import write_json
else:
    from benchmark_output import write_json

CONTEXT = 262144
WORLD = 8
DEPTH = 2
CONFIDENCE = 0.7
INPUT_CREDIT = 16 << 20
MAX_PROMPT_BYTES = 4096
MAX_RENDERED_BYTES = 65536
MAX_PROMPT_TOKENS = 4096
MAX_TOKENS = 512
MAX_RECEIPT_BYTES = 1 << 20
LEADER_RECEIPT_BYTES = 512 << 10
FOLLOWER_RECEIPT_BYTES = 64 << 10
SUPERVISOR_RECEIPT_BYTES = 64 << 10
FATAL_REPORT_GRACE_SECONDS = 30.0
DEFAULT_PROMPT = "In one short sentence, explain why the sky appears blue."
TARGET_TEMPLATE_SHA = "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce"
SOURCE_FILES = (
    "tensorfold/__init__.py", "tensorfold/families/qwen4_exp/__init__.py",
    "tensorfold/families/qwen4_exp/cuda/engine.py",
    "tensorfold/families/qwen4_exp/cuda/retention_resources.py",
    "tensorfold/families/qwen4_exp/cuda/tp_scheduler.py",
    "tensorfold/cuda/expert_launch.py",
    "tensorfold/cuda/scheduler.py", "tensorfold/server/request_limits.py",
    "tensorfold/token_metadata.py", "tensorfold/cuda/chat_template.py",
    "tensorfold/cuda/sm70/execution_policy.py",
    "tensorfold/families/qwen4_exp/cuda/prefix_policy.py",
    "tensorfold/families/qwen4_exp/cuda/state.py",
    "tensorfold/families/qwen4_exp/cuda/multi.py",
    "tensorfold/families/qwen4_exp/cuda/execution_identity.py",
    "tensorfold/families/qwen4_exp/cuda/owner_preparation.py",
    "tensorfold/families/qwen4_exp/cuda/owner_execution.py",
    "tensorfold/families/qwen4_exp/cuda/owner_graphs.py",
    "tensorfold/families/qwen4_exp/cuda/history_owner.py",
    "tensorfold/cuda/sm70/paged_attention.py",
    "tensorfold/engine/model_identity.py",
)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_report(path, value, *, allowance):
    """Bound the aggregate live report; disk checkpoints do not create live trees.

    The leader owns512KiB, seven followers64KiB each, and the supervisor64KiB:
    together they fit the caller's1MiB global receipt allowance. Overflow is a
    failed diagnostic; preserve its complete evidence separately for review.
    """
    encoded = json.dumps(value, indent=1, allow_nan=False)
    size = len(encoded) + 1  # ensure_ascii=True makes every serialized character one byte.
    del encoded
    if size > allowance:
        write_json(Path(path).with_suffix(".overflow.json"), value)
        raise ValueError(f"finite aggregate report slice exceeded ({size}>{allowance} bytes); complete overflow evidence saved")
    write_json(path, value)


def finish_failures(primary, slots):
    """Collect already retired statuses without replacing an earlier failure."""
    try:
        errors = [failure for failure in slots if failure is not None]
    except BaseException as allocation:
        if primary is not None:
            raise primary from allocation
        for failure in slots:
            if failure is not None:
                raise failure from allocation
        raise
    raise_failures(primary, errors)


def regular_file(path):
    value = Path(path).expanduser().resolve(strict=True)
    if not stat.S_ISREG(value.stat().st_mode):
        raise ValueError("probe input must be an existing regular file: " + str(value))
    return value


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="existing pinned SC117 IQ3_S first GGUF shard")
    parser.add_argument("--mtp", type=Path, required=True, help="existing pinned SC117 Q8 MTP GGUF")
    parser.add_argument("--wheel", type=Path, required=True, help="exact installed whole TensorFold wheel")
    parser.add_argument("--source-hold", type=Path, required=True, help="frozen source/provenance HOLD, copied and hashed")
    parser.add_argument("--execution-profile-manifest", type=Path,
                        help="explicit immutable loader-derived execution manifest; native owner trials remain unqualified")
    parser.add_argument("--image-identity", required=True, help="actual immutable base image sha256 identity")
    parser.add_argument("--base-commit", help="optional base commit; never a claim that dirty source equals it")
    parser.add_argument("--out", type=Path, required=True, help="new private result directory")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--master-port", type=int, default=29607)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--instrumented-departure-grace-seconds", type=float,
                        help="explicit profiler trial only: extend the natural child-exit wait, bounded by --timeout-seconds; default30s, failure teardown stays30s")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=128, help="bounded reply allowance, 1..512")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="1..16 sequential repetitions using one loaded engine/admission; cache state is retained and reported")
    parser.add_argument("--expect-prefix-reuse", action="store_true",
                        help="private cached FAMILY trial: require cold first MTP and maximal legal prefix on later identical prompts")
    parser.add_argument("--thinking", action="store_true", help="pass enable_thinking=True to the unchanged target template")
    parser.add_argument("--serial-reference", action="store_true", help="also run a separately labeled draft=False reply with MTP loaded")
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_identity):
        parser.error("--image-identity must be an actual immutable sha256 image identity")
    if args.base_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", args.base_commit):
        parser.error("--base-commit must be 40 lowercase hexadecimal characters")
    if (not 1 <= args.tokens <= MAX_TOKENS or not args.prompt
            or len(args.prompt.encode("utf-8")) > MAX_PROMPT_BYTES):
        parser.error("a nonempty prompt of at most4096 UTF8 bytes and 1..512 reply tokens are required")
    if not 1 <= args.repetitions <= 16:
        parser.error("--repetitions must be 1..16")
    if args.expect_prefix_reuse and args.repetitions < 2:
        parser.error("--expect-prefix-reuse requires at least2 repetitions")
    if (not 1 <= args.master_port <= 65535 or not math.isfinite(args.timeout_seconds)
            or args.timeout_seconds <= 0 or args.rank is not None and args.rank not in range(WORLD)):
        parser.error("invalid rendezvous port, finite positive timeout, or rank")
    departure = args.instrumented_departure_grace_seconds
    if departure is not None and (not math.isfinite(departure) or not 0 < departure <= args.timeout_seconds):
        parser.error("instrumented departure grace must be positive, finite and no larger than --timeout-seconds")
    args.devices = args.gpus.split(",")
    if (len(args.devices) != WORLD or len(set(args.devices)) != WORLD
            or any(not value or value != value.strip() for value in args.devices)):
        parser.error("--gpus requires eight distinct nonempty GPU identities")
    from tensorfold.cuda.sm70.execution_policy import PROFILE_ENV

    # The existing constructor owns supported values, runtime qualification and
    # its unset default. Recording intent must not select a different profile.
    args.requested_sm70_profile = os.environ.get(PROFILE_ENV)
    if args.expect_prefix_reuse:
        from tensorfold.families.qwen4_exp.cuda.prefix_policy import FAMILY_PREFIX_PROFILE

        if args.requested_sm70_profile != FAMILY_PREFIX_PROFILE:
            parser.error("--expect-prefix-reuse requires the explicit cached FAMILY trial profile")
    for name in ("model", "mtp", "wheel", "source_hold"):
        setattr(args, name, regular_file(getattr(args, name)))
    if args.source_hold.stat().st_size > 16 << 20:
        parser.error("source HOLD exceeds the16MiB artifact bound")
    args.execution_manifest_identity = None
    if args.execution_profile_manifest is not None:
        from tensorfold.families.qwen4_exp.cuda.execution_identity import _json, _manifest_schema

        args.execution_profile_manifest = regular_file(args.execution_profile_manifest)
        document, raw = _json(args.execution_profile_manifest, with_content=True)
        _manifest_schema(document)
        args.execution_manifest_identity = {"path": str(args.execution_profile_manifest), "bytes": len(raw),
                                            "sha256": hashlib.sha256(raw).hexdigest()}
    args.out = args.out.expanduser().absolute()
    return args


def installed_identity(wheel):
    """Verify affected normal imports against the declared whole-wheel artifact."""
    import tensorfold

    if "site-packages" not in Path(tensorfold.__file__).parts:
        raise RuntimeError("backend probe requires normal installed TensorFold imports")
    distribution = importlib.metadata.distribution("tensorfold")
    rows = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in SOURCE_FILES:
            path = Path(distribution.locate_file(name))
            actual = digest(path)
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            if actual != expected:
                raise ValueError("installed source differs from the declared wheel: " + name)
            rows[name] = actual
    records = [value for value in distribution.files or () if str(value).endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError("one installed wheel RECORD is required")
    return {"package": str(Path(tensorfold.__file__).resolve()), "version": distribution.version,
            "record_sha256": digest(distribution.locate_file(records[0])), "affected_files": rows}


class TokenCollector:
    """One synchronous callback, with no decoding, queue or retained chunk copies."""

    def __init__(self, maximum, vocab):
        if type(maximum) is not int or not 1 <= maximum <= MAX_TOKENS or type(vocab) is not int or vocab <= 0:
            raise ValueError("finite callback requires a supported token allowance and vocabulary")
        self.maximum, self.vocab, self.ids, self.closed = maximum, vocab, [], False
        self.first_nonempty_monotonic_ns = None

    def __call__(self, chunk):
        if self.closed:
            raise RuntimeError("generation callback used after retirement")
        if (type(chunk) is not list or len(chunk) > self.maximum - len(self.ids)
                or any(type(token) is not int or not 0 <= token < self.vocab for token in chunk)):
            raise ValueError("generation callback exceeded its finite output/token contract")
        if chunk and self.first_nonempty_monotonic_ns is None:
            self.first_nonempty_monotonic_ns = time.monotonic_ns()
        self.ids.extend(chunk)
        return False

    def close(self):
        self.closed = True


def record_execution_profile(engine, report, *, requested):
    """Observe the constructor-selected dispatch without changing native flags."""
    from tensorfold.cuda.sm70 import family_ops, routing_native
    from tensorfold.families.qwen4_exp.cuda import gguf_weights, padded_linears
    from tensorfold.families.qwen4_exp.cuda import prefix_policy

    report["requested_sm70_profile"] = requested
    report["sm70_profile"] = engine.sm70_profile
    report["prefix_policy"] = {"identity": prefix_policy.identity(engine.sm70_profile),
                               "state_epoch": prefix_policy.state_epoch(engine.sm70_profile)}
    report["native_flags"] = {
        "family": family_ops._NATIVE_FAMILY,
        "private_attention": family_ops._NATIVE_ATTENTION,
        "routing": routing_native._NATIVE_LOGITS,
        "padded": padded_linears._ENABLED,
        "gguf_dense_implementation": gguf_weights.GGUFDense.implementation,
        "gguf_concat_implementation": gguf_weights.GGUFConcat.implementation,
        "gguf_experts_implementation": gguf_weights.GGUFExperts.implementation,
        "gguf_experts_packed_rows": gguf_weights.GGUFExperts.packed_rows,
    }
    # Preparation intent does not prove which family the adopted buffers use.
    # Retain one bounded observation per fixed execution role. The full prepared
    # identity is already in capacity_plan; bind it without duplicating that
    # potentially large allocation record in every rank's finite report.
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest as identity_digest

    preparation = getattr(engine, "_owner_preparation", None)
    decoder = getattr(engine, "multi", None)
    roles = {}
    for role, name in (("decode", "buf"), ("mtp", "mbuf"), ("prefill", "pbuf")):
        buffers = None if decoder is None else getattr(decoder, name, None)
        if buffers is not None and buffers.history_owner is not None:
            roles[role] = {"kernel_family": buffers.owner_kernel_family,
                           "graph_family": buffers.owner_graph_family,
                           "collective_family": buffers.history_owner.native_collective}
    report["owner_execution"] = {
        "prepared_identity": None if preparation is None else {
            "sha256": identity_digest(preparation.identity()),
            "prepared": preparation.prepared, "closed": preparation.closed,
            "trial": preparation.specification["trial"]},
        "roles": roles}
    if requested is not None and requested != engine.sm70_profile:
        raise ValueError("requested SM70 profile differs from the actual constructor selection")


def prepared_prompt(engine, tokenizer, args):
    template = engine.chat_template
    if template is None or template.provenance()["sha256"] != TARGET_TEMPLATE_SHA:
        raise ValueError("finite probe requires the unchanged pinned SC117 target chat template")
    rendered = template.render([{"role": "user", "content": args.prompt}], tools=None,
                               enable_thinking=args.thinking)
    if len(rendered.encode("utf-8")) > MAX_RENDERED_BYTES:
        raise ValueError("rendered prompt exceeds the finite caller byte bound")
    tokens = tokenizer.encode(rendered, add_special_tokens=False).ids
    if not tokens or len(tokens) > MAX_PROMPT_TOKENS:
        raise ValueError("rendered prompt exceeds the finite caller token bound")
    return tokens, {"messages": [{"role": "user", "content": args.prompt}],
                    "enable_thinking": args.thinking, "template": template.provenance(),
                    "rendered": rendered, "token_ids": tokens}


def check_engine(engine):
    """Refuse a changed execution profile before host finalization/admission."""
    plan = engine.capacity_plan
    solved, full = plan["session_capacity"], plan["kv_full_windows"]
    if (engine.tp != WORLD or engine.depth != DEPTH or engine.confidence != CONFIDENCE
            or engine.context_window != CONTEXT or engine.indexer_history != "gpu"
            or engine.kv_pair.key_dtype != "int8" or engine.kv_pair.value_dtype != "int8"
            or engine.w.mtp is None or engine.scheduler is None or engine.retention_catalog is None
            or engine.w.meta.get("execution_provider") != "sm70"
            or engine.w.meta.get("execution_dtype") != "float16"
            or (engine.w.meta["full"].hidden, engine.w.meta["full"].layers, engine.w.meta["full"].vocab)
                != (2560, 48, 248320)
            or solved["context_tokens"] != CONTEXT or solved["configured_ceiling"] is not None
            or solved["resident_capacity"] < 1
            or solved["full_context_commitments"] != solved["resident_capacity"]
            or not full["materialized"]):
        raise ValueError("loaded engine differs from native262144/INT8/TP8/auto/Q8-MTP2 profile")
    if engine.scheduler._request_limit is not None:
        raise ValueError("finite backend host admission must precede logical request publication")


def generate_reply(engine, tokenizer, tokens, *, count, draft, limit, result):
    from torch.cuda import nvtx
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.token_metadata import decode_presentation, tokenizer_vocab_size

    collector = TokenCollector(count, tokenizer_vocab_size(tokenizer))
    started = time.monotonic_ns()
    result.update(label="mtp-target" if draft else "serial-reference", draft=draft,
                  token_ids=collector.ids, complete=False,
                  nvtx_range="tensorfold/mtp-target" if draft else "tensorfold/serial-reference")
    try:
        # The logical caller owns callback, generation and final decode together.
        with limit.request():
            primary, marker_failures = None, [None]
            nvtx.range_push(result["nvtx_range"])
            try:
                stats = engine.generate(tokens, count, Sampling(seed=17, temperature=0, top_k=0, top_p=1),
                                        collector, draft=draft, stop_eos=True)
                text = decode_presentation(tokenizer, collector.ids)
            except BaseException as error:
                primary = error
            finally:
                try:
                    nvtx.range_pop()
                except BaseException as error:
                    marker_failures[0] = error
            finish_failures(primary, marker_failures)
            first = collector.first_nonempty_monotonic_ns
            result.update(text=text, stats=stats, elapsed_ns=time.monotonic_ns() - started,
                          first_committed_token_elapsed_ns=None if first is None else first - started,
                          timing_scope="diagnostic wall interval including final decode; not qualified throughput")
            if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_RECEIPT_BYTES:
                raise ValueError("generated result exceeds the finite caller receipt bound")
        if limit.used:
            raise RuntimeError("logical request ownership survived completed generation and decode")
        result["complete"] = True
        return result
    finally:
        collector.close()


def repeat_replies(engine, tokenizer, tokens, args, report, limit):
    """Run bounded sequential jobs under one already admitted engine/owner cap."""
    report["completed_iterations"] = 0
    for iteration in range(1, args.repetitions + 1):
        # Only the current pair stays live. Every previous reply is already
        # durable; no summary/list of prior pairs grows with repetitions.
        report["replies"].clear()
        report["iteration"] = iteration
        for draft in ((True, False) if args.serial_reference else (True,)):
            row = {"iteration": iteration}
            report["replies"].append(row)
            label = "mtp-target" if draft else "serial-reference"
            name = label if args.repetitions == 1 else f"iteration-{iteration:04d}-{label}"
            primary, diagnostics = None, [None]
            try:
                generate_reply(engine, tokenizer, tokens, count=args.tokens, draft=draft, limit=limit, result=row)
            except BaseException as error:
                primary = error
            if getattr(args, "expect_prefix_reuse", False) and draft and primary is None:
                expected = 0 if iteration == 1 else len(tokens) - 1
                observed = row["stats"].get("cached")
                matched = type(observed) is int and observed == expected
                row["prefix_reuse_check"] = {"expected": expected, "observed": observed, "passed": matched,
                    "scope": "same rendered prompt; complete reusable prefix before pending anchor"}
                if not matched:
                    primary = ValueError("cached FAMILY trial differs from the declared cold/full-prefix case")
            try:
                write_json(args.out / (name + ".json"), row)
            except BaseException as error:
                diagnostics[0] = error
            finish_failures(primary, diagnostics)
            write_report(args.out / "progress.json", report, allowance=LEADER_RECEIPT_BYTES)
            print(json.dumps({"event": "reply", **row}, ensure_ascii=False, allow_nan=False), flush=True)
            if not row["text"].strip():
                raise ValueError("finite diagnostic completed without visible text; raw IDs/stats were preserved")
            row = None  # Retire the last alias before the next pair clears its list.
        report["completed_iterations"] = iteration


def run_leader(engine, tokenizer, args, report):
    from tensorfold.families.qwen4_exp.cuda.retention_resources import (
        FiniteBackendCaller, finalize_backend_resources,
    )
    from tensorfold.server.request_limits import RequestLimit

    check_engine(engine)
    if getattr(args, "expect_prefix_reuse", False):
        from tensorfold.families.qwen4_exp.cuda.prefix_policy import FAMILY_PREFIX_PROFILE, FAMILY_PREFIX_STATE

        if engine.sm70_profile != FAMILY_PREFIX_PROFILE:
            raise ValueError("prefix-reuse expectation requires the explicit cached FAMILY trial profile")
        if any(FAMILY_PREFIX_STATE not in state.kv_identity for state in engine.retention_catalog.states):
            raise ValueError("cached FAMILY State owners lack the declared prefix policy epoch")
    tokens, prompt = prepared_prompt(engine, tokenizer, args)
    report["prompt"] = prompt
    caller = FiniteBackendCaller(max_input_tokens=MAX_PROMPT_TOKENS,
        max_output_tokens=args.tokens, retained_results=2 if args.serial_reference else 1,
        max_source_bytes=MAX_PROMPT_BYTES + MAX_RENDERED_BYTES, max_receipt_bytes=MAX_RECEIPT_BYTES)
    # The real catalog/baseline/native/RPC envelope must pass physical admission
    # before any logical request owner or backend generation is published.
    limit = RequestLimit(1)
    try:
        report["host_admission"] = finalize_backend_resources(engine.retention_catalog,
            tokenizer=tokenizer, scheduler=engine.scheduler, request_limit=limit, caller=caller)
        write_report(args.out / "admitted.json", report, allowance=LEADER_RECEIPT_BYTES)
        engine.scheduler.configure_requests(limit)
        repeat_replies(engine, tokenizer, tokens, args, report, limit)
    finally:
        limit.close()


def worker(args):
    from tensorfold import hub
    from tensorfold.cuda import precision, prompt_precision
    from tensorfold.cuda.cohort import FAILURE_FD_ENV
    from tensorfold.families.qwen4_exp import cuda_engine
    from tensorfold.serve_options import drafter_metadata_source
    from tensorfold.token_metadata import load_tokenizer, tokenizer_identity

    descriptor = os.environ.get(FAILURE_FD_ENV, "")
    if not descriptor.isdecimal() or not stat.S_ISFIFO(os.fstat(int(descriptor)).st_mode):
        raise ValueError("probe workers require their supervised process-lifetime failure pipe")
    report = {"schema": "tensorfold.v100-finite-backend-probe-v1", "rank": args.rank,
              "argv": sys.argv, "complete": False, "replies": [],
              "repetitions": args.repetitions,
              "process_identity": {"pid": os.getpid(), "parent_pid": os.getppid(),
                                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
              "qualification": "finite-backend-diagnostic-only; HTTP/design qualification remains open"}
    engine, primary, failures = None, None, [None] * 3
    allowance = FOLLOWER_RECEIPT_BYTES if args.rank else LEADER_RECEIPT_BYTES
    try:
        report["installed"] = installed_identity(args.wheel)
        report["execution_profile_manifest"] = getattr(args, "execution_manifest_identity", None)
        model = hub.resolve(str(args.model), download=False)
        companion = drafter_metadata_source(str(args.mtp))
        report["prepared_model_dir"] = str(model)
        precision.set_mode(precision.CHECKPOINT)
        prompt_precision.set_fp8(prompt_precision.FP8_BY_DEFAULT)
        engine = cuda_engine(model, drafter=str(companion), draft_method="mtp", tp=WORLD, rank=args.rank,
            master="127.0.0.1", master_port=args.master_port, mtp_drafts=DEPTH, mtp_confidence=CONFIDENCE,
            context=CONTEXT, context_explicit=True, parallel="auto", kv_full_windows="auto",
            kv_dtype="int8", indexer_history="gpu", max_waiting_input_bytes=INPUT_CREDIT,
            output_queue_tokens=args.tokens,
            **({"execution_profile_manifest": args.execution_profile_manifest}
               if getattr(args, "execution_profile_manifest", None) is not None else {}))
        record_execution_profile(engine, report, requested=args.requested_sm70_profile)
        report["capacity_plan"] = engine.capacity_plan
        write_report(args.out / f"rank-{args.rank}-ready.json", report, allowance=allowance)
        if args.rank:
            engine.follow()
        else:
            tokenizer = load_tokenizer(model)
            report["tokenizer_identity"] = tokenizer_identity(tokenizer)
            run_leader(engine, tokenizer, args, report)
    except BaseException as error:
        primary = error
        try:
            report["failure"] = {"type": type(error).__name__, "message": str(error),
                                 "traceback": traceback.format_exc()}
            write_report(args.out / f"rank-{args.rank}-failure.json", report, allowance=allowance)
        except BaseException as diagnostic:
            failures[0] = diagnostic
    finally:
        if engine is not None:
            try:
                engine.close()
            except BaseException as error:
                failures[1] = error
    try:
        if failures[0] is not None or failures[1] is not None:
            report["cleanup_failures"] = [{"type": type(error).__name__, "message": str(error)}
                                          for error in failures[:2] if error is not None]
        report["complete"] = primary is None and not any(error is not None for error in failures)
        write_report(args.out / f"rank-{args.rank}-final.json", report, allowance=allowance)
    except BaseException as diagnostic:
        failures[2] = diagnostic
    finish_failures(primary, failures)
    return 0


def launch(args, argv):
    from tensorfold.cuda.cohort import FAILURE_FD_ENV
    from tensorfold.cuda.expert_launch import supervise

    parameters = inspect.signature(supervise).parameters
    reports_exit = "on_exit" in parameters
    supports_departure = "departure_grace" in parameters
    if args.instrumented_departure_grace_seconds is not None and not supports_departure:
        raise ValueError("installed supervisor does not support instrumented departure grace; use its reviewed whole wheel")
    args.out.mkdir(mode=0o700)
    with args.source_hold.open("rb") as source:
        hold = source.read((16 << 20) + 1)
    if len(hold) > 16 << 20:
        raise ValueError("source HOLD exceeds the16MiB artifact bound")
    (args.out / "source-hold").write_bytes(hold)
    commands, environments = [], []
    primary, failures, result = None, [None] * 6, 1
    signums = (signal.SIGINT, signal.SIGTERM)
    previous = [None, None]
    rank_order = [*range(1, WORLD), 0]
    started, stopped, expired = time.monotonic(), False, False
    first_fatal = None
    notification_count = 0
    receipt = {"schema": "tensorfold.v100-finite-backend-launch-v1", "argv": [sys.executable, *sys.argv],
        "image_identity": args.image_identity, "base_commit": args.base_commit,
        "source_hold": {"path": str(args.source_hold), "sha256": hashlib.sha256(hold).hexdigest()},
        "wheel": {"path": str(args.wheel), "sha256": digest(args.wheel)},
        "tool_sha256": digest(__file__), "installed": installed_identity(args.wheel),
        "execution_profile_manifest": getattr(args, "execution_manifest_identity", None),
        "settings": {"context": CONTEXT, "tp": WORLD, "depth": DEPTH, "confidence": CONFIDENCE,
            "kv_dtype": "int8", "parallel": "auto", "kv_full_windows": "auto", "indexer_history": "gpu",
            "max_waiting_input_bytes": INPUT_CREDIT, "logical_callers": 1,
            "requested_sm70_profile": args.requested_sm70_profile,
            "instrumented_departure_grace_seconds": args.instrumented_departure_grace_seconds,
            "repetitions": args.repetitions,
            "output_queue_tokens": args.tokens, "prefill_rows_env": os.environ.get("TENSORFOLD_PREFILL_ROWS", "2048")},
        "raw_output": "worker stdout/stderr inherit the launcher's streams; capture both in the execution log",
        "complete": False, "qualification": "supplemental finite backend; complete HTTP/design goal open",
        "supervision": {"available": False, "reason": "installed supervisor has no exit reporter"}}
    read_fd = write_fd = None

    def record_exit(record):
        record["available"] = True
        record["rank_order"] = rank_order
        receipt["supervision"] = record

    supervision_options = {"on_exit": record_exit} if reports_exit else {}
    if args.instrumented_departure_grace_seconds is not None:
        supervision_options["departure_grace"] = args.instrumented_departure_grace_seconds

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    def failed():
        nonlocal expired, first_fatal, notification_count
        now = time.monotonic()
        expired = now - started >= args.timeout_seconds
        if stopped or expired:
            return True
        try:
            notification = os.read(read_fd, 4096)
        except BlockingIOError:
            notification = b""
        if notification:
            notification_count += len(notification)
            first = first_fatal is None
            if first:
                first_fatal = now
            receipt["fatal_notification"] = {
                "first_elapsed_s": first_fatal - started, "bytes_observed": notification_count,
                "last_observed_bytes_hex": notification.hex(), "report_grace_seconds": FATAL_REPORT_GRACE_SECONDS,
                "state": "fatal; awaiting failure reports/owned retirement; no recovery"}
            if first:
                write_report(args.out / "fatal-notification.json", receipt, allowance=SUPERVISOR_RECEIPT_BYTES)
                print(json.dumps({"event": "fatal-notification", **receipt["fatal_notification"]}), flush=True)
        # Fatal notifications remain latched. Allow the already failing ranks
        # to publish diagnostics before group containment, without retrying or
        # resuming model work. Stops and the overall deadline take precedence.
        return first_fatal is not None and now - first_fatal >= FATAL_REPORT_GRACE_SECONDS

    try:
        # Install the lifetime guard before acquiring either descriptor. Even
        # diagnostic construction failure must reach both independent closes.
        read_fd, write_fd = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
        for rank in rank_order:
            commands.append([sys.executable, str(Path(__file__).resolve()), *argv, "--rank", str(rank)])
            environments.append(dict(os.environ, CUDA_VISIBLE_DEVICES=args.devices[rank],
                                     **{FAILURE_FD_ENV: str(write_fd)}))
        receipt["commands"] = commands
        write_report(args.out / "launch.json", receipt, allowance=SUPERVISOR_RECEIPT_BYTES)
        for index, signum in enumerate(signums):
            previous[index] = signal.getsignal(signum)
            signal.signal(signum, stop)
        result = supervise(commands, environments, grace=30, failure_probe=failed, pass_fds=(write_fd,),
                           **supervision_options)
        failed()  # Consume a final notification after all owned rank sessions have retired.
        if result == 0 and (first_fatal is not None or stopped or expired):
            result = 1
        receipt.update(exit_code=result, stopped=stopped, timed_out=expired,
                       elapsed_s=time.monotonic() - started, complete=result == 0 and not stopped and not expired)
    except BaseException as error:
        primary = error
        try:
            receipt["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        except BaseException as diagnostic:
            failures[0] = diagnostic
    finally:
        # Four fixed retirements avoid allocating cleanup iteration containers.
        if previous[0] is not None:
            try:
                signal.signal(signums[0], previous[0])
            except BaseException as error:
                failures[1] = error
        if previous[1] is not None:
            try:
                signal.signal(signums[1], previous[1])
            except BaseException as error:
                failures[2] = error
        if write_fd is not None:
            try:
                os.close(write_fd)
            except BaseException as error:
                failures[3] = error
        if read_fd is not None:
            try:
                os.close(read_fd)
            except BaseException as error:
                failures[4] = error
    try:
        write_report(args.out / "launch-final.json", receipt, allowance=SUPERVISOR_RECEIPT_BYTES)
    except BaseException as diagnostic:
        failures[5] = diagnostic
    finish_failures(primary, failures)
    return result if result else (1 if stopped or expired else 0)


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    args = arguments(raw)
    return worker(args) if args.rank is not None else launch(args, raw)


if __name__ == "__main__":
    raise SystemExit(main())
