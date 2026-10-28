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
MAX_TOKENS = 2048
LEADER_RECEIPT_BYTES = 512 << 10
FOLLOWER_RECEIPT_BYTES = 256 << 10
SUPERVISOR_RECEIPT_BYTES = 64 << 10
MAX_RECEIPT_BYTES = LEADER_RECEIPT_BYTES + (WORLD - 1) * FOLLOWER_RECEIPT_BYTES + SUPERVISOR_RECEIPT_BYTES
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
    "tensorfold/cuda/sm70/router_blas.py", "tensorfold/cuda/sm70/router_blas.cpp",
    "tensorfold/cuda/moe.py",
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
    "tensorfold/families/qwen4_exp/cuda/target_greedy.py",
    "tensorfold/families/qwen4_exp/cuda/batch_prelude.py",
    "tensorfold/families/qwen4_exp/cuda/kv_status.py",
    "tensorfold/families/qwen4_exp/cuda/forward.py",
    "tensorfold/cuda/sm70/greedy_native.py", "tensorfold/cuda/sm70/greedy_bindings.cpp",
    "tensorfold/cuda/sm70/greedy.cu",
)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_report(path, value, *, allowance):
    """Bound the aggregate live report; disk checkpoints do not create live trees.

    The leader owns512KiB, seven followers256KiB each, and the supervisor64KiB:
    their sum is the caller's funded global receipt allowance. Overflow is a
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
    parser.add_argument("--triton-cache-root", type=Path,
                        help="opt-in persistent cache root: each child inherits TRITON_CACHE_DIR=ROOT/rank-N before startup; Triton retains its source/platform/configuration keys")
    parser.add_argument("--router-blas-trial", action="store_true",
                        help="explicit experimental pointer-batched FP32 router; requires matching numeric profile, native Dense/batch and resident24")
    parser.add_argument("--owner-batch-prelude-trial", action="store_true",
                        help="private exact multi-State native prelude composed with selected Dense; explicit matching profile and native-Dense intent required; unqualified")
    parser.add_argument("--gdn-state-batch-trial", action="store_true",
                        help="private exact FP16 nv6 State-batch screen; requires explicit matching APP environment and batch/Dense/24 intent")
    parser.add_argument("--native-dense-trial", action="store_true",
                        help="private exact native32 Dense composition with native target/MTP prelude; requires explicit matching SM70 profile and session ceiling24; unqualified")
    parser.add_argument("--overlap-content-capture", action="store_true",
                        help="private TP8 owner MTP diagnostic: enable funded one-worker fresh artifact hashing during startup; inherited by all ranks before imports; default serial hashing")
    parser.add_argument("--startup-content-workers", type=int, choices=(1,3), default=1,
                        help="explicit1(default) or3 independent whole-file startup SHA workers; requires --overlap-content-capture, all resources prepaid")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--session-ceiling", type=int, choices=(24,),
                        help="optional explicit24-session qualification region; default retains automatic capacity")
    parser.add_argument("--master-port", type=int, default=29607)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--instrumented-departure-grace-seconds", type=float,
                        help="explicit profiler trial only: extend the natural child-exit wait, bounded by --timeout-seconds; default30s, failure teardown stays30s")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=128, help="bounded reply allowance:1..512 ordinary/v1; exactly2048 requires --fast-grid-protocol fast-grid-v2")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="1..16 sequential repetitions using one loaded engine/admission; cache state is retained and reported")
    parser.add_argument("--expect-prefix-reuse", action="store_true",
                        help="private cached FAMILY trial: require cold first MTP and maximal legal prefix on later identical prompts")
    parser.add_argument("--concurrency-grid", action="store_true",
                        help="finite bursts at1,2,4,8,16,24; sustained replies require>=256 committed tokens each plus positive correlated decode overlap; EOS unchanged")
    parser.add_argument("--fast-grid", action="store_true",
                        help="operational screen: one2-tokenNotebookbringup, allsixNs,30sreadiness/target10sdecode each, expectedcancel+drain; requiresgrid and320or512(v1)/2048(v2)outputcells; notfullqualification")
    parser.add_argument("--fast-grid-protocol", choices=("fast-grid-v1", "fast-grid-v2"), default="fast-grid-v1",
                        help="explicit protocol: v1 uses320/512outputcells; v2 funds2048cells for full10s >85tokens/s screens; sameprompts/window/EOS/quality")
    parser.add_argument("--quality-spotchecks", action="store_true",
                        help="after the unchanged concurrency grid, run six fixed independent answer/format checks with320-token allowances; requires grid and --tokens>=320")
    parser.add_argument("--thinking", action="store_true", help="pass enable_thinking=True to the unchanged target template")
    parser.add_argument("--serial-reference", action="store_true", help="also run a separately labeled draft=False reply with MTP loaded")
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_identity):
        parser.error("--image-identity must be an actual immutable sha256 image identity")
    if args.base_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", args.base_commit):
        parser.error("--base-commit must be 40 lowercase hexadecimal characters")
    if (not 1 <= args.tokens <= MAX_TOKENS or not args.prompt
            or len(args.prompt.encode("utf-8")) > MAX_PROMPT_BYTES
            or args.tokens > 512 and not (args.fast_grid and args.fast_grid_protocol == "fast-grid-v2" and args.tokens == 2048)):
        parser.error("a nonempty4096-byte prompt and1..512tokens required; explicitfast-grid-v2allows2048")
    if not 1 <= args.repetitions <= 16:
        parser.error("--repetitions must be 1..16")
    if args.concurrency_grid and (args.tokens < 256 or args.expect_prefix_reuse):
        parser.error("--concurrency-grid requires --tokens256..512 and cannot impose sequential prefix-reuse assertions")
    allowed_fast_tokens = (320, 512) if args.fast_grid_protocol == "fast-grid-v1" else (2048,)
    if args.fast_grid_protocol != "fast-grid-v1" and not args.fast_grid:
        parser.error("--fast-grid-protocol fast-grid-v2 requires --fast-grid")
    if args.fast_grid and (not args.concurrency_grid or args.tokens not in allowed_fast_tokens or args.serial_reference or args.repetitions != 1):
        parser.error("--fast-grid requires --concurrency-grid, protocol-outputcap320or512(v1)/2048(v2), --repetitions1 and MTPonly")
    if args.quality_spotchecks and (not args.concurrency_grid or args.tokens < 320):
        parser.error("--quality-spotchecks requires --concurrency-grid and --tokens>=320 for the declared caller funding")
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
    if args.triton_cache_root is not None:
        args.triton_cache_root = args.triton_cache_root.expanduser().resolve()
        if args.rank is not None:
            expected = str(args.triton_cache_root / f"rank-{args.rank}")
            if os.environ.get("TRITON_CACHE_DIR") != expected:
                parser.error("worker TRITON_CACHE_DIR must be inherited from the launcher before startup")
    from tensorfold.cuda.sm70.execution_policy import PROFILE_ENV

    # The existing constructor owns supported values, runtime qualification and
    # its unset default. Recording intent must not select a different profile.
    args.requested_sm70_profile = os.environ.get(PROFILE_ENV)
    if args.native_dense_trial:
        from tensorfold.cuda.sm70.execution_policy import OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL

        if args.requested_sm70_profile not in (OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL) or args.session_ceiling != 24:
            parser.error("--native-dense-trial requires explicit source-owner-target-dense-trial and --session-ceiling24")
    if args.owner_batch_prelude_trial:
        from tensorfold.cuda.sm70.execution_policy import OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL

        if args.requested_sm70_profile not in (OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL) or not args.native_dense_trial or args.session_ceiling != 24:
            parser.error("--owner-batch-prelude-trial requires matching source-dense-owner-batch-prelude-trial, --native-dense-trial and --session-ceiling24")
    from tensorfold.cuda.sm70.execution_policy import ROUTER_BLAS_TRIAL

    if args.router_blas_trial != (args.requested_sm70_profile == ROUTER_BLAS_TRIAL):
        parser.error("--router-blas-trial and explicit source-batch-router-blas-trial must agree")
    if args.router_blas_trial and (not args.owner_batch_prelude_trial or not args.native_dense_trial or args.session_ceiling != 24):
        parser.error("router trial requires native Dense, owner batch prelude and session-ceiling24")
    gdn_flag = os.environ.get("TENSORFOLD_SM70_GDN_STATE_BATCH")
    if args.gdn_state_batch_trial:
        if gdn_flag != "1" or not args.owner_batch_prelude_trial or not args.fast_grid or not args.quality_spotchecks:
            parser.error("--gdn-state-batch-trial requires private env1, batch/Dense/24, FASTv2 and unchanged quality checks")
        if args.fast_grid_protocol != "fast-grid-v2":
            parser.error("GDN State batch first screen requires closed FASTv2/2048")
    elif gdn_flag is not None:
        parser.error("private GDN environment requires explicit --gdn-state-batch-trial intent")

    if args.expect_prefix_reuse:
        from tensorfold.families.qwen4_exp.cuda.prefix_policy import state_epoch

        if state_epoch(args.requested_sm70_profile) is None:
            parser.error("--expect-prefix-reuse requires an explicit cached SM70 trial profile")
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
    if args.startup_content_workers != 1 and not args.overlap_content_capture:
        parser.error("--startup-content-workers3 requires --overlap-content-capture")
    if args.overlap_content_capture:
        if args.execution_profile_manifest is None:
            parser.error("--overlap-content-capture requires an explicit execution manifest")
        if args.rank is not None and os.environ.get("TENSORFOLD_STARTUP_CONTENT_OVERLAP") != "1":
            parser.error("worker startup content overlap must be inherited from the launcher")
        if args.rank is not None and os.environ.get("TENSORFOLD_STARTUP_CONTENT_WORKERS", "1") != str(args.startup_content_workers):
            parser.error("worker content hash worker budget must match inherited startup environment")
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
    from tensorfold.cuda.sm70.execution_policy import PACKED_GRAPH_TRIAL, OWNER_GRAPH_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL

    if engine.sm70_profile in (PACKED_GRAPH_TRIAL, OWNER_GRAPH_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
        from tensorfold.families.qwen4_exp.cuda.multi_graphs import (
            UNIQUE_CAPTURE_CONTRACT, UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT,
        )

        contract = (UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT if engine.sm70_profile in (OWNER_GRAPH_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL)
                    else UNIQUE_CAPTURE_CONTRACT)

        decoder = engine.multi
        if (not engine.graphs_enabled or decoder.graphs is None or not decoder.graph_record.get("captured")
                or not decoder.graphs.verify or not decoder.graphs.mtp
                or engine.w.meta.get("unique_capture_contract") != contract):
            raise ValueError("requested graph trial did not obtain an admitted unique-history graph owner")
        report["unique_graph_trial"] = {"contract": contract,
            "verify_keys": len(decoder.graphs.verify), "mtp_keys": len(decoder.graphs.mtp),
            "admission": engine.capacity_plan["multi_graphs"]}


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


def check_engine(engine, *, session_ceiling=None, require_scheduler=True):
    """Refuse a changed execution profile before host finalization/admission."""
    plan = engine.capacity_plan
    solved, full = plan["session_capacity"], plan["kv_full_windows"]
    if (engine.tp != WORLD or engine.depth != DEPTH or engine.confidence != CONFIDENCE
            or engine.context_window != CONTEXT or engine.indexer_history != "gpu"
            or engine.kv_pair.key_dtype != "int8" or engine.kv_pair.value_dtype != "int8"
            or engine.w.mtp is None or (require_scheduler and engine.scheduler is None) or engine.retention_catalog is None
            or engine.w.meta.get("execution_provider") != "sm70"
            or engine.w.meta.get("execution_dtype") != "float16"
            or (engine.w.meta["full"].hidden, engine.w.meta["full"].layers, engine.w.meta["full"].vocab)
                != (2560, 48, 248320)
            or solved["context_tokens"] != CONTEXT or solved["configured_ceiling"] != session_ceiling
            or (solved["resident_capacity"] < 1 if session_ceiling is None else solved["resident_capacity"] != session_ceiling)
            or solved["full_context_commitments"] != solved["resident_capacity"]
            or not full["materialized"]):
        raise ValueError("loaded engine differs from native262144/INT8/TP8/auto/Q8-MTP2 profile")
    if require_scheduler and engine.scheduler._request_limit is not None:
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

    if getattr(args, "session_ceiling", None) is None:
        check_engine(engine)
    else:
        check_engine(engine, session_ceiling=args.session_ceiling)
    if getattr(args, "concurrency_grid", False):
        return run_concurrency_leader(engine, tokenizer, args, report)
    if getattr(args, "expect_prefix_reuse", False):
        from tensorfold.families.qwen4_exp.cuda.prefix_policy import state_epoch

        epoch = state_epoch(engine.sm70_profile)
        if epoch is None:
            raise ValueError("prefix-reuse expectation requires an explicit cached SM70 trial profile")
        if any(epoch not in state.kv_identity for state in engine.retention_catalog.states):
            raise ValueError("cached State owners lack the selected prefix policy epoch")
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
        if report.get("sm70_profile") == "source-target-greedy-trial":
            owner = engine.w.meta.get("target_greedy_owner")
            if owner is not None:
                report["target_greedy_final"] = owner.record()
    finally:
        limit.close()




def _owned_attributes(value):
    """Inspect existing Python fields without invoking descriptors or device work."""
    try:
        result = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        return {}
    return result if type(result) is dict else {}

def write_moe_capture_ledger(engine, output, rank):
    """One bounded cold evidence file; never duplicate145 phase rows in replies."""
    decoder = _owned_attributes(engine).get("multi")
    owner = _owned_attributes(_owned_attributes(decoder).get("graphs"))
    catalogue = owner.get("moe_regions")
    fields = _owned_attributes(catalogue)
    if catalogue is None:
        return None
    measurements = fields.get("measured")
    if fields.get("ready") is not True or type(measurements) is not list or len(measurements) != 145:
        raise ValueError("required complete145-phase MoE capture ledger is unavailable")
    payload = {"schema": "tensorfold.cold-moe-capture-ledger-v1", "rank": rank,
               "funding": fields.get("plan"), "capture_phases": measurements,
               "private_pools": owner.get("private_pool_records"),
               "host_before": fields.get("host_before"), "host_after": fields.get("host_after"),
               "scope": "coldpreparation only; baselinecanonical buffers/weights alreadyfunded; newGPUphasecharges usemax physical/general/owned deltas, never sum overlappingviews"}
    encoded = (json.dumps(payload, indent=1, allow_nan=False) + "\n").encode("ascii")
    if len(encoded) > 1 << 20:
        raise ValueError("cold MoE capture ledger exceeds its inclusive graph-host allowance slice")
    path = Path(output) / f"rank-{rank}-moe-capture-ledger.json"
    primary = None
    close_failures = [None]  # Reserve before acquiring the output stream.
    file = path.open("xb")
    try:
        if file.write(encoded) != len(encoded):
            raise OSError("short cold MoE capture ledger write")
    except BaseException as error:
        primary = error
    try:
        file.close()
    except BaseException as error:
        close_failures[0] = error
    finish_failures(primary, close_failures)
    catalogue._host_gate()  # Include the bounded cold diagnostic buffer in the observed operating gate.
    return {"path": str(path), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest(),
            "phase_records": len(measurements), "scope": "cold rankevidence; graphhostallowance, not256KiB replypayload"}


def graph_execution_snapshot(engine):
    """Bounded host counters, not a device-completion or timing instrument."""
    decoder = _owned_attributes(engine).get("multi")
    owner = _owned_attributes(decoder).get("graphs")
    fields = _owned_attributes(owner)
    batch = _owned_attributes(_owned_attributes(decoder).get("buf")).get("gdn_state_batch")
    submitted = _owned_attributes(batch).get("submitted")
    if owner is None and submitted is None:
        return None
    counters = {}
    moe_regions = _owned_attributes(fields.get("moe_regions"))
    for name in ("frames", "hits"):
        value = moe_regions.get(name)
        if not moe_regions.get("counter_overflowed", False) and type(value) is int and 0 <= value <= (1 << 63) - 1:
            counters["moe_regions." + name] = value
    if type(submitted) is int and 0 <= submitted <= (1 << 63) - 1:
        counters["gdn.batch_submitted_layers"] = submitted
    names = ("hits", "misses", "lane_declines", "verify_hits", "mtp_hits",
             "eager_declines", "verify_misses", "mtp_misses")
    for role, values in (("owner", fields), ("bound", _owned_attributes(fields.get("bound")))):
        for name in names:
            value = values.get(name)
            if type(value) is int and value >= 0:
                counters[role + "." + name] = value
    weights = _owned_attributes(decoder).get("w")
    metadata = _owned_attributes(weights).get("meta", {})
    target = metadata.get("target_greedy_owner") if type(metadata) is dict else None
    prelude = metadata.get("sm70_owner_prelude_submitted_frames") if type(metadata) is dict else None
    if type(prelude) is int and prelude >= 0:
        counters["owner.prelude_submitted_frames"] = prelude
    mtp_prelude = metadata.get("sm70_owner_prelude_submitted_mtp_frames") if type(metadata) is dict else None
    mtp_overflowed = metadata.get("sm70_owner_prelude_mtp_counter_overflowed", False) if type(metadata) is dict else False
    if not mtp_overflowed and type(mtp_prelude) is int and mtp_prelude >= 0:
        counters["owner.prelude_mtp_submitted_frames"] = mtp_prelude
        if type(prelude) is int and prelude >= mtp_prelude:
            counters["owner.prelude_target_submitted_frames"] = prelude - mtp_prelude
    if target is not None:
        from tensorfold.families.qwen4_exp.cuda.target_greedy import TargetGreedy

        if type(target) is TargetGreedy:
            value = object.__getattribute__(target, "completed")
            if type(value) is int and value >= 0:
                counters["target.completed"] = value
    for field, label in (("sm70_owner_prelude_submitted_batches", "owner.prelude_submitted_batches"),
                         ("sm70_owner_prelude_submitted_mtp_batches", "owner.prelude_submitted_mtp_batches"),
                         ("sm70_owner_prelude_max_batch_segments", "owner.prelude_max_batch_segments")):
        value = metadata.get(field) if type(metadata) is dict else None
        if type(value) is int and value >= 0:
            counters[label] = value
    if type(metadata) is dict and metadata.get("sm70_owner_prelude_batch_counter_overflowed"):
        raise ValueError("batch native preparation diagnostic counter overflowed")
    for field, label in (("buf", "target"), ("mbuf", "mtp")):
        plans = _owned_attributes(_owned_attributes(decoder).get(field)).get("router_blas")
        info = _owned_attributes(plans)
        calls = info.get("calls")
        if info.get("overflowed"):
            raise ValueError("router batched diagnostic counter overflowed")
        if type(calls) is int and 0 <= calls <= (1 << 63) - 1:
            counters["router_blas." + label + ".calls"] = calls
    keys = {name: len(fields[name]) for name in ("verify", "mtp") if type(fields.get(name)) is dict}
    result = {"counters": counters, "captured_keys": keys}
    if mtp_overflowed:
        result["unavailable_counters"] = ["owner.prelude_mtp_submitted_frames", "owner.prelude_target_submitted_frames"]
    return result

def graph_execution_delta(before, after):
    """Differences of available counters from the same owner before cleanup.

    The finite caller never replaces its engine/profile. Missing counters are
    not treated as zero; captured-key availability is reported separately.
    """
    if before is None or after is None:
        return None
    return {name: value - before["counters"][name] for name, value in after["counters"].items()
            if name in before["counters"]}


BATCH_EVIDENCE_CONTROL_BYTES = 8192  # Fixed host-only receipt fields, bounded int63 counters and reason literals.


def batch_preparation_evidence(profile, count, before, after):
    """Qualify actual cohort dispatch after drain; never infer missing counts as zero."""
    from tensorfold.cuda.sm70.execution_policy import OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL

    if profile not in (OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
        return None
    result = {"passed": False, "available": False, "cohort_N": count,
              "scope": "submitted native batches during readiness/decode/cancel/drain; not device timing"}
    fields = ("owner.prelude_submitted_batches", "owner.prelude_submitted_mtp_batches")
    snapshots = (before, after)
    if any(type(snapshot) is not dict or type(snapshot.get("counters")) is not dict
           or any(type(snapshot["counters"].get(field)) is not int
                  or not 0 <= snapshot["counters"][field] <= (1 << 63) - 1 for field in fields)
           for snapshot in snapshots):
        result["reason"] = "required batch counters unavailable or invalid"
        return result
    total, mtp = (after["counters"][field] - before["counters"][field] for field in fields)
    if any(snapshot["counters"][fields[1]] > snapshot["counters"][fields[0]] for snapshot in snapshots) or not 0 <= mtp <= total:
        result["reason"] = "batch role counters regressed or exceeded total"
        return result
    result.update(available=True, total_batches=total, target_batches=total - mtp, mtp_batches=mtp)
    result["passed"] = (total == 0 if count == 1 else total - mtp > 0 and mtp > 0)
    if not result["passed"]:
        result["reason"] = ("N1 must retain scalar preparation" if count == 1
                            else "multi-State cohort requires actual target and MTP native batches")
    return result

GDN_EVIDENCE_CONTROL_BYTES = 8192


def gdn_batch_evidence(count, before, after):
    """Exact host submission counter evidence, never device timing or CUDA completion."""
    field = "gdn.batch_submitted_layers"
    result = {"passed": False, "available": False, "cohort_N": count,
              "scope": "target-layer batch submissions during this readiness/decode/cancel/drain cohort; not device time"}
    if any(type(snapshot) is not dict or type(snapshot.get("counters")) is not dict
           or type(snapshot["counters"].get(field)) is not int
           or not 0 <= snapshot["counters"][field] <= (1 << 63) - 1 for snapshot in (before, after)):
        result["reason"] = "required GDN submission counter unavailable or invalid"
        return result
    delta = after["counters"][field] - before["counters"][field]
    result.update(available=True, submitted_layers=delta)
    result["passed"] = delta == 0 if count == 1 else delta > 0
    if not result["passed"]:
        result["reason"] = "N1 must remain original; multi-State cohort must actually submit batch GDN"
    return result


def check_gdn_batch_engine(engine):
    """Check actual factory result/funding before any finite logical caller admission."""
    from tensorfold.families.qwen4_exp.cuda.gdn_state_batch import GDNStateBatch, workspace_bytes

    decoder = engine.multi
    owner = decoder.buf.gdn_state_batch
    host_bytes = workspace_bytes(36)[1]
    if (engine.w.meta.get("sm70_gdn_state_batch_trial") is not True or type(owner) is not GDNStateBatch
            or engine.execution_rows != 32 or engine.capacity_plan.get("execution_envelope", {}).get("target_rows") != 32
            or owner.layers != 36 or owner.host.shape != (36, 24, 10) or owner.frame.shape != (24, 4)
            or owner.host.device.type != "cpu" or owner.frame.device.type != "cpu"
            or engine.w.cfg.nk != 2 or engine.w.cfg.nv != 6
            or engine.capacity_plan.get("host_workspace_bytes_estimate", 0) < 8 * host_bytes):
        raise ValueError("actual GDN batch factory/funding differs from row32/native24 FP16nv6 contract")
    return {"enabled": True, "layers": 36, "max_states": 24, "target_rows": 32,
            "extra_device_bytes": 0, "funded_global_host_bytes": 8 * host_bytes,
            "dispatch": "privateFP16nv6; N1/prefill/longchains/modelgraph original",
            "pointees": "actual original State FP32rec/F16conv/replayscratch; weak static cache/strong current frame"}


def check_router_blas_engine(engine):
    """Read actual prepared owners and prepaid charges on every rank before submit."""
    from tensorfold.cuda.sm70.router_blas import PROFILE, RolePlans, role_bytes, role_payload_bytes

    if engine.sm70_profile != PROFILE or engine.execution_rows != 32:
        raise ValueError("actual router factory differs from private row32 profile")
    roles = (("target", engine.multi.buf.router_blas, 48), ("mtp", engine.multi.mbuf.router_blas, 1))
    for _name, owner, count in roles:
        device, host = role_bytes(32, count)
        if (type(owner) is not RolePlans or len(owner.owners) != count or len(owner.plans) != count
                or set(owner.plans) != {id(weight) for weight in owner.owners}
                or owner.gpu_bytes != device or owner.host_bytes != host
                or owner.gpu_payload_bytes != role_payload_bytes(32, count) or owner.overflowed):
            raise ValueError("actual router role ownership/resources differ from admitted plan")
    charged_device = sum(role_bytes(32, count)[0] for _, _, count in roles)
    charged_host = WORLD * sum(role_bytes(32, count)[1] for _, _, count in roles)
    if (engine.capacity_plan.get("device_workspace_bytes_estimate", 0) < charged_device
            or engine.capacity_plan.get("host_workspace_bytes_estimate", 0) < charged_host):
        raise ValueError("actual router role funding is missing before logical admission")
    return {"enabled": True, "target_weights": 48, "mtp_weights": 1, "target_rows": 32,
            "device_charge_per_rank": charged_device, "raw_device_payload_per_rank": 801024,
            "global_host_charge": charged_host, "scope": "eager target/MTP only; graph/prefill canonical; experimental numeric implementation"}


def concurrency_module():
    if __package__:
        from . import finite_concurrency
    else:
        import finite_concurrency
    return finite_concurrency


def waiting_input_credit(args):
    return concurrency_module().input_credit(CONTEXT) if getattr(args, "concurrency_grid", False) else INPUT_CREDIT


def run_quality_spotchecks(engine, tokenizer, args, report, limit, prepared, quality):
    """Reuse the admitted owner; keep one raw reply and write every mismatch."""
    status = report["quality"]
    for case_id, tokens, prompt in prepared:
        for draft in ((True, False) if args.serial_reference else (True,)):
            mode = "mtp-target" if draft else "serial-reference"
            row = {"case_id": case_id, "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt["rendered"].encode("utf-8")).hexdigest(),
                "source_prompt_sha256": hashlib.sha256(prompt["messages"][0]["content"].encode("utf-8")).hexdigest(),
                "model_identity": status["model_identity"], "dataset": quality.VERSION,
                "dataset_sha256": status["dataset_sha256"],
                "quality_helper_sha256": status["helper_sha256"],
                "max_tokens": quality.OUTPUT_TOKENS, "stop_eos": True,
                "quality": {"passed": False, "assessment_complete": False},
                "measurement_role": "quality spotcheck; excluded from performance metrics"}
            primary, diagnostics = None, [None]
            try:
                generate_reply(engine, tokenizer, tokens, count=quality.OUTPUT_TOKENS, draft=draft,
                               limit=limit, result=row)
                row["quality"] = quality.assess(case_id, row["text"])
            except BaseException as error:
                primary = error
            try:
                # Raw text/IDs also persist on generation or oracle failure.
                write_report(args.out / f"quality-{case_id}-{mode}.json", row, allowance=MAX_RECEIPT_BYTES)
            except BaseException as diagnostic:
                diagnostics[0] = diagnostic
            finish_failures(primary, diagnostics)
            status["completed_requests"] += 1
            status["failed_requests"] += not row["quality"]["passed"]
            status["last_result"] = {"case_id": case_id, "draft": draft, "passed": row["quality"]["passed"]}
            row = None  # No previous result graph survives the next generation.
            write_report(args.out / "progress.json", report, allowance=LEADER_RECEIPT_BYTES)
    status["passed"] = status["failed_requests"] == 0
    write_report(args.out / "quality-summary.json", status, allowance=LEADER_RECEIPT_BYTES)
    if not status["passed"]:
        raise ValueError("fixed quality spot checks failed; all raw answers and differences were retained")



def run_fast_grid(engine, tokenizer, args, report, limit, pool, prepared, finite, vocab):
    """Screen all six offered Ns; independent answer quality stays separate."""
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.cuda.cohort import notify_failure
    from tensorfold.token_metadata import decode_presentation

    _label, tokens, prompt = next(row for row in prepared if row[0] == "sustained")
    report["fast_screen"] = {"protocol": getattr(args, "fast_grid_protocol", "fast-grid-v1"), "readiness_cap_seconds": finite.FAST_READY_SECONDS,
        "target_decode_seconds": finite.FAST_DECODE_SECONDS, "output_cell_bound": args.tokens,
        "completed_cohorts": 0, "invalid_measurements": 0,
        "full_qualification": False, "quality_assessed": False,
        "prompt_sha256": hashlib.sha256(prompt["rendered"].encode("utf-8")).hexdigest(),
        "schedule": "one Notebook2-tokenMTP bringup, then1/2/4/8/16/24; no per-cohort warmup/shortbringupsweep; optional quality onlyafterall6"}
    warmup = {}
    generate_reply(engine, tokenizer, tokens, count=2, draft=True, limit=limit, result=warmup)
    warmup["measurement_role"] = "one cheap real NotebookMTP readiness/prefill; not answerquality or timed decode"
    write_report(args.out / "fast-notebook-bringup.json", warmup, allowance=MAX_RECEIPT_BYTES)
    report["fast_screen"]["bringup"] = {"stats": warmup["stats"],
        "first_committed_token_elapsed_ns": warmup["first_committed_token_elapsed_ns"],
        "prompt_tokens": len(tokens), "committed_tokens": len(warmup["token_ids"])}
    del warmup

    def invoke(collector, progress):
        stats = engine.generate(tokens, args.tokens, Sampling(seed=17, temperature=0, top_k=0, top_p=1),
                                collector, draft=True, stop_eos=True, progress=progress)
        natural = ("natural_eos" if collector.ids and collector.ids[-1] in getattr(engine, "eos", ()) else
                   "output_cap" if len(collector.ids) == args.tokens else "backend_return")
        return {"stats": stats, "text": decode_presentation(tokenizer, collector.ids), "natural_stop": natural}

    def observe():
        _, entries = engine.scheduler.waiting.requests(include_reserved=True)
        streams = engine.retention_catalog._cohort.multi.streams
        for entry in entries:
            stream = entry[2][0]
            callback = stream.delivery_owner.emit
            if type(callback) is not finite.ScreenTokens:
                raise ValueError("fast native output requires its actual owned ScreenTokens callback")
            callback.bind_native_stream(stream)
        record = engine.scheduler.queue_record()
        return {"states": tuple((entry[1], entry[2][0].admission_status.phase) for entry in entries),
            "decoding": finite.decoding_tickets(entries, streams),
            "nonmonotone_decode_states": tuple(entry[1] for entry in entries
                if (stream := entry[2][0]).started > 0 and stream.committed_output_tokens > 0 and not stream.done
                and (stream.waiting or streams.get(stream.sid) is not stream)),
            "outcomes": record["outcomes"], "charged_live_input_bytes": record["charged_live_input_bytes"],
            "backend_live": engine.retention_catalog._cohort.multi.live(),
            "backend_retiring": len(engine.scheduler._retiring), "backend_mailboxes": len(engine.scheduler._request_boxes)}

    for count in finite.GRID:
        graph_before = graph_execution_snapshot(engine)
        def write_case(result):
            graph_after = graph_execution_snapshot(engine)
            result.update(workload="fast-sustained", prompt=prompt, max_tokens=args.tokens, stop_eos=True,
                profile=engine.sm70_profile, draft=True,
                graph_execution={"before_requests": graph_before, "after_requests": graph_after,
                    "delta": graph_execution_delta(graph_before, graph_after),
                    "scope": "this screen cohort including readiness and measured decode/cancel/drain; excludesconstructor/cheapbringup/otherNs/quality; countersnotdevicetiming"})
            evidence = batch_preparation_evidence(engine.sm70_profile, count, graph_before, graph_after)
            if evidence is not None:
                result["batch_native_preparation"] = evidence
                if not evidence["passed"]:
                    result["passed"] = False
                    result["screen"]["measurement_valid"] = False
            if getattr(args, "gdn_state_batch_trial", False):
                gdn_evidence = gdn_batch_evidence(count, graph_before, graph_after)
                result["gdn_state_batch"] = gdn_evidence
                if not gdn_evidence["passed"]:
                    result["passed"] = False
                    result["screen"]["measurement_valid"] = False
            write_report(args.out / f"fast-mtp-target-n{count:02d}.json", result,
                         allowance=finite.MAXIMUM * MAX_RECEIPT_BYTES)
        summary = finite.run_screen_case(pool, count=count, maximum_tokens=args.tokens, vocab=vocab,
            limit=limit, invoke=invoke, observe=observe, write=write_case, timeout=args.timeout_seconds,
            failure_notify=notify_failure, protocol=getattr(args, "fast_grid_protocol", "fast-grid-v1"))
        report["fast_screen"]["completed_cohorts"] += 1
        report["fast_screen"]["invalid_measurements"] += not summary["passed"]
        report["fast_screen"]["last_case"] = summary
        report["concurrency"]["completed_cases"] += 1
        write_report(args.out / "progress.json", report, allowance=LEADER_RECEIPT_BYTES)
    report["fast_screen"]["all_required_N_attempted"] = report["fast_screen"]["completed_cohorts"] == len(finite.GRID)
    return report["fast_screen"]["invalid_measurements"]

def run_concurrency_leader(engine, tokenizer, args, report):
    """One physical envelope and real logical cap; no fake HTTP or runtime patch."""
    from types import SimpleNamespace
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.retention_resources import FiniteBackendCaller, finalize_backend_resources
    from tensorfold.server.request_limits import RequestLimit
    from tensorfold.token_metadata import decode_presentation, tokenizer_vocab_size
    from tensorfold.cuda.sm70.execution_policy import OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL

    finite = concurrency_module()
    prepared = []
    for label, prompt in (("bringup", args.prompt), ("sustained", finite.SUSTAINED_PROMPT)):
        tokens, receipt = prepared_prompt(engine, tokenizer, SimpleNamespace(prompt=prompt, thinking=args.thinking))
        prepared.append((label, tokens, receipt))
    quality, quality_prepared, quality_control = None, None, 0
    if getattr(args, "quality_spotchecks", False):
        if __package__:
            from . import finite_quality as quality
        else:
            import finite_quality as quality
        quality_prepared = []
        for case_id, prompt, _expected in quality.CASES:
            tokens, receipt = prepared_prompt(engine, tokenizer, SimpleNamespace(prompt=prompt, thinking=args.thinking))
            quality_prepared.append((case_id, tokens, receipt))
        if len(prepared) + len(quality_prepared) > finite.MAXIMUM:
            raise ValueError("prepared quality/performance ID owners exceed the funded24-owner envelope")
        quality_control = (quality.control_payload_bytes() + sys.getsizeof(quality_prepared)
                           + sum(sys.getsizeof(row) for row in quality_prepared))
        report["quality"] = {"dataset": quality.VERSION,
            "dataset_sha256": hashlib.sha256(json.dumps(quality.CASES, ensure_ascii=True, separators=(",", ":")).encode("ascii")).hexdigest(),
            "helper_sha256": digest(quality.__file__),
            "prepared_prompts": [receipt for _, _, receipt in quality_prepared],
            "prepared_input_owners_total": len(prepared) + len(quality_prepared),
            "additional_control_payload_bytes": quality_control,
            "completed_requests": 0, "failed_requests": 0, "passed": False,
            "model_identity": {"model_path": str(args.model), "mtp_path": str(args.mtp),
                "prepared_model_dir": report.get("prepared_model_dir"), "profile": engine.sm70_profile,
                "context": CONTEXT, "tp": WORLD, "kv_dtype": "int8", "mtp_drafts": DEPTH,
                "mtp_confidence": CONFIDENCE, "thinking": args.thinking,
                "sampling": {"seed": 17, "temperature": 0, "top_k": 0, "top_p": 1},
                "tokenizer_identity": report["tokenizer_identity"], "native_flags": report["native_flags"],
                "wheel_sha256": digest(args.wheel), "source_hold_sha256": digest(args.source_hold)},
            "scope": "six deterministic answer/format spot checks only; no broad quality or performance qualification"}
    caller = FiniteBackendCaller(max_input_tokens=MAX_PROMPT_TOKENS, max_output_tokens=args.tokens,
        retained_results=finite.MAXIMUM, max_source_bytes=MAX_PROMPT_BYTES + MAX_RENDERED_BYTES,
        max_receipt_bytes=(finite.MAXIMUM + 1) * MAX_RECEIPT_BYTES,
        max_concurrent_requests=finite.MAXIMUM, worker_stack_bytes=finite.STACK_BYTES,
        control_payload_bytes=finite.control_payload_bytes(args.tokens) + quality_control
            + (finite.screen_control_payload_bytes(args.tokens) if getattr(args, "fast_grid", False) else 0)
            + (BATCH_EVIDENCE_CONTROL_BYTES if engine.sm70_profile in (OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL) else 0)
            + (GDN_EVIDENCE_CONTROL_BYTES if getattr(args, "gdn_state_batch_trial", False) else 0))

    limit, pool = RequestLimit(finite.MAXIMUM), None
    primary, failures = None, [None, None]
    report["concurrency"] = {"grid": finite.GRID, "caller": caller.__dict__,
        "prompts": [receipt for _, _, receipt in prepared], "completed_cases": 0,
        "schedule": "fixed ascending N, bringup then sustained, MTP then optional serial, repeated; simultaneous burst barrier",
        "stack_size_before_pool": __import__("threading").stack_size(),
        "helper_sha256": digest(finite.__file__), "eos": "ordinary stop_eos=True",
        "qualification": "measurement harness; no performance or quality qualification until matched actual runs"}
    try:
        report["host_admission"] = finalize_backend_resources(engine.retention_catalog,
            tokenizer=tokenizer, scheduler=engine.scheduler, request_limit=limit, caller=caller)
        write_report(args.out / "admitted.json", report, allowance=LEADER_RECEIPT_BYTES)
        engine.scheduler.configure_requests(limit)
        pool = finite.owned_pool(min(30, args.timeout_seconds))
        vocab = tokenizer_vocab_size(tokenizer)

        def observe():
            _, entries = engine.scheduler.waiting.requests(include_reserved=True)
            states = tuple((entry[1], entry[2][0].admission_status.phase) for entry in entries)
            record = engine.scheduler.queue_record()
            return {"states": states,
                    "decoding": finite.decoding_tickets(entries, engine.retention_catalog._cohort.multi.streams),
                    "nonmonotone_decode_states": tuple(entry[1] for entry in entries
                        if (stream := entry[2][0]).started > 0 and stream.committed_output_tokens > 0
                        and not stream.done and (stream.waiting
                            or engine.retention_catalog._cohort.multi.streams.get(stream.sid) is not stream)),
                    "outcomes": record["outcomes"],
                    "charged_live_input_bytes": record["charged_live_input_bytes"]}

        invalid_screens = 0
        if getattr(args, "fast_grid", False):
            invalid_screens = run_fast_grid(engine, tokenizer, args, report, limit, pool, prepared, finite, vocab)
        else:
            for iteration in range(1, args.repetitions + 1):
                for label, tokens, prompt in prepared:
                    for draft in ((True, False) if args.serial_reference else (True,)):
                        mode = "mtp-target" if draft else "serial-reference"
                        # Real one-token warmup, durably reported then retired. All
                        # variants use the same prompt/cache schedule; no cache clear.
                        warmup = {}
                        generate_reply(engine, tokenizer, tokens, count=1, draft=draft, limit=limit, result=warmup)
                        write_json(args.out / f"iteration-{iteration:04d}-{label}-{mode}-warmup.json", warmup)
                        del warmup

                        def invoke(collector, progress):
                            stats = engine.generate(tokens, args.tokens,
                                Sampling(seed=17, temperature=0, top_k=0, top_p=1), collector,
                                draft=draft, stop_eos=True, progress=progress)
                            return {"stats": stats, "text": decode_presentation(tokenizer, collector.ids)}

                        for count in finite.GRID:
                            name = f"iteration-{iteration:04d}-{label}-{mode}-n{count:02d}"
                            graph_before = graph_execution_snapshot(engine) if label == "sustained" else None
                            def write_case(result):
                                if label == "sustained":
                                    graph_after = graph_execution_snapshot(engine)
                                    result["graph_execution"] = {"before_requests": graph_before,
                                        "after_requests": graph_after,
                                        "delta": graph_execution_delta(graph_before, graph_after),
                                        "scope": "this measured sustained burst only; after constructor, bringup and explicit tool warmup; before optional quality and cleanup; host counters only"}
                                result.update(iteration=iteration, workload=label, draft=draft,
                                    prompt=prompt, stop_eos=True, max_tokens=args.tokens,
                                    profile=engine.sm70_profile)
                                write_report(args.out / (name + ".json"), result,
                                             allowance=finite.MAXIMUM * MAX_RECEIPT_BYTES)
                            summary = finite.run_case(pool, count=count, maximum_tokens=args.tokens, vocab=vocab,
                                limit=limit, invoke=invoke, observe=observe, write=write_case,
                                minimum_tokens=1 if label == "bringup" else finite.SUSTAINED_MINIMUM,
                                timeout=args.timeout_seconds)
                            report["concurrency"]["completed_cases"] += 1
                            report["concurrency"]["last_case"] = {"name": name, **summary}
                            write_report(args.out / "progress.json", report, allowance=LEADER_RECEIPT_BYTES)
        if quality is not None:
            run_quality_spotchecks(engine, tokenizer, args, report, limit, quality_prepared, quality)
        if invalid_screens:
            raise ValueError("fast screen has zero/no-overlap/readiness failures; allsixNs and independentquality were attempted")
    except BaseException as error:
        primary = error
    finally:
        if pool is not None:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except BaseException as error:
                failures[0] = error
        try:
            limit.close()
        except BaseException as error:
            failures[1] = error
    finish_failures(primary, failures)


def worker(args):
    from tensorfold import hub
    from tensorfold.cuda import precision, prompt_precision
    from tensorfold.cuda.cohort import FAILURE_FD_ENV, notify_failure
    from tensorfold.families.qwen4_exp import cuda_engine
    from tensorfold.cuda.sm70.execution_policy import PACKED_GRAPH_TRIAL, OWNER_GRAPH_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL
    from tensorfold.serve_options import drafter_metadata_source
    from tensorfold.token_metadata import load_tokenizer, tokenizer_identity

    descriptor = os.environ.get(FAILURE_FD_ENV, "")
    if not descriptor.isdecimal() or not stat.S_ISFIFO(os.fstat(int(descriptor)).st_mode):
        raise ValueError("probe workers require their supervised process-lifetime failure pipe")
    report = {"schema": "tensorfold.v100-finite-backend-probe-v1", "rank": args.rank,
              "argv": sys.argv, "complete": False, "replies": [],
              "repetitions": args.repetitions,
              "process_identity": {"pid": os.getpid(), "parent_pid": os.getppid(),
                                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                   "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR")},
              "qualification": "finite-backend-diagnostic-only; HTTP/design qualification remains open"}
    engine, primary, failures = None, None, [None] * 4
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
            context=CONTEXT, context_explicit=True,
            parallel=getattr(args, "session_ceiling", None) or "auto", kv_full_windows="auto",
            kv_dtype="int8", indexer_history="gpu", max_waiting_input_bytes=waiting_input_credit(args),
            output_queue_tokens=args.tokens, graphs=args.requested_sm70_profile in (PACKED_GRAPH_TRIAL, OWNER_GRAPH_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL),
            **({"execution_profile_manifest": args.execution_profile_manifest}
               if getattr(args, "execution_profile_manifest", None) is not None else {}))
        record_execution_profile(engine, report, requested=args.requested_sm70_profile)
        if getattr(args, "gdn_state_batch_trial", False):
            check_engine(engine, session_ceiling=24, require_scheduler=args.rank == 0)
            report["gdn_state_batch"] = check_gdn_batch_engine(engine)
        if getattr(engine, "sm70_profile", None) == ROUTER_BLAS_TRIAL:
            report["router_blas"] = check_router_blas_engine(engine)
        report["moe_capture_ledger"] = write_moe_capture_ledger(engine, args.out, args.rank)
        report["capacity_plan"] = engine.capacity_plan
        report["graph_execution"] = {"before_requests": graph_execution_snapshot(engine),
            "after_requests": None, "delta": None,
            "scope": "existing host counters after constructor warmup and before cleanup; includes every tool request, including bringup, explicit tool warmups and optional quality; measured-case snapshots are separate; no device timing"}
        write_report(args.out / f"rank-{args.rank}-ready.json", report, allowance=allowance)
        if args.rank:
            engine.follow()
        else:
            tokenizer = load_tokenizer(model)
            report["tokenizer_identity"] = tokenizer_identity(tokenizer)
            run_leader(engine, tokenizer, args, report)
        execution = report["graph_execution"]
        execution["after_requests"] = graph_execution_snapshot(engine)
        execution["delta"] = graph_execution_delta(execution["before_requests"], execution["after_requests"])
    except BaseException as error:
        primary = error
        try:
            # Signal the existing supervised cohort before fallible diagnostics
            # or backend cleanup can block. Its latched grace bounds retirement.
            if not notify_failure():
                raise RuntimeError("finite worker lost its supervised failure capability")
        except BaseException as notification:
            failures[3] = notification
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
        if failures[0] is not None or failures[1] is not None or failures[3] is not None:
            report["cleanup_failures"] = [{"type": type(error).__name__, "message": str(error)}
                                          for error in (failures[0], failures[1], failures[3]) if error is not None]
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
        "tool_sha256": digest(__file__),
        "concurrency_helper_sha256": digest(concurrency_module().__file__) if args.concurrency_grid else None,
        "installed": installed_identity(args.wheel),
        "execution_profile_manifest": getattr(args, "execution_manifest_identity", None),
        "settings": {"context": CONTEXT, "tp": WORLD, "depth": DEPTH, "confidence": CONFIDENCE,
            "kv_dtype": "int8", "parallel": getattr(args, "session_ceiling", None) or "auto",
            "session_ceiling": getattr(args, "session_ceiling", None), "kv_full_windows": "auto", "indexer_history": "gpu",
            "max_waiting_input_bytes": waiting_input_credit(args),
            "logical_callers": 24 if args.concurrency_grid else 1,
            "max_receipt_bytes": (25 if args.concurrency_grid else 1) * MAX_RECEIPT_BYTES,
            "concurrency_grid": [1, 2, 4, 8, 16, 24] if args.concurrency_grid else None,
            "quality_spotchecks": getattr(args, "quality_spotchecks", False),
            "fast_grid_protocol": getattr(args, "fast_grid_protocol", "fast-grid-v1"),
            "fast_grid": getattr(args, "fast_grid", False),
            "measurement_role": "fastoperational-screen-notfullqualification" if getattr(args, "fast_grid", False) else "fullfinitequalification-protocol",
            "requested_sm70_profile": args.requested_sm70_profile,
            "triton_cache_root": str(args.triton_cache_root) if args.triton_cache_root is not None else None,
            "overlap_content_capture": getattr(args, "overlap_content_capture", False),
            "startup_content_workers": getattr(args, "startup_content_workers", 1),
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
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.devices[rank],
                               **{FAILURE_FD_ENV: str(write_fd)})
            if getattr(args, "overlap_content_capture", False):
                environment["TENSORFOLD_STARTUP_CONTENT_OVERLAP"] = "1"
                if getattr(args, "startup_content_workers", 1) != 1:
                    environment["TENSORFOLD_STARTUP_CONTENT_WORKERS"] = str(args.startup_content_workers)
            if args.triton_cache_root is not None:
                environment["TRITON_CACHE_DIR"] = str(args.triton_cache_root / f"rank-{rank}")
            environments.append(environment)
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
