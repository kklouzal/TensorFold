"""Fixed-region, single-stream Gemma dense comparison using the maintained HTTP reader.

prepare URL MODEL SNAPSHOT CORPUS MANIFEST freezes prompts, native sampling and ABBA order.
run URL MANIFEST META OUT --cycle C --position P records one independently started server.
analyze MANIFEST OUT RUN... reports paired cycle log-ratio intervals, never an equivalence verdict.

The primary/operator starts each candidate. This tool never starts containers, clears server state
through private endpoints, or changes model configuration. Fresh requests first send an ordinary
raw non-BOS token request, then require actual cached=0. Warm requests repeat the same body.
Timeouts are socket-inactivity limits; an external watchdog owns whole-request stall policy.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import time
import urllib.parse
import urllib.request

if __package__:
    from . import campaign_bench as campaign
    from . import openai_protocol as protocol
    from .benchmark_output import write_json
else:
    import campaign_bench as campaign
    import openai_protocol as protocol
    from benchmark_output import write_json

EVICTION_TOKENS = [1000]  # Gemma BOS=2; a raw non-BOS entry cannot share even the chat turn checkpoint.
METRICS = {
    "prefill_tps": "higher", "decode_tps": "higher", "ttft_client_s": "lower",
    "complete_body_tps": "higher", "complete_body_s": "lower",
}
IDENTITIES = ("image_id", "source_revision", "model_revision", "launch", "instance_id")
MATERIAL_IDENTITIES = ("image_id", "source_revision", "model_revision", "launch", "drafter_revision",
                       "precision", "kv_dtype", "context", "runtime_pins", "environment")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def base_url(value):
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("use an HTTP(S) base URL without embedded credentials")
    if parsed.query or parsed.fragment or parsed.path.rstrip("/").endswith("/v1"):
        raise ValueError("benchmark base URL excludes /v1, queries and fragments")
    return value.rstrip("/")


def json_endpoint(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        protocol.response_type(response, "application/json")
        raw = response.read(protocol.MAX_RESPONSE_BYTES + 1)
    if len(raw) > protocol.MAX_RESPONSE_BYTES:
        raise ValueError("metadata response exceeds benchmark storage budget")
    return protocol.object_json(raw)


def sampling_policy(root, policy):
    """Freeze actual checkpoint values or explicitly requested HF sampling defaults.

    HF do_sample=False is native greedy. Conditional sampled coverage explicitly
    supplies its temperature/top-k/top-p instead of claiming default sampling.
    """
    path = root / "generation_config.json"
    if policy == "checkpoint":
        generation = json.loads(path.read_text())
        return ({k: generation[k] for k in ("temperature", "top_k", "top_p")},
                {"kind": "checkpoint", "sha256": digest(path),
                 "do_sample": generation.get("do_sample")}, "native")
    if policy != "hf-defaults" or path.exists():
        raise ValueError("hf-defaults requires an absent generation_config.json; retain checkpoint authority")
    import transformers
    from transformers import AutoConfig, GenerationConfig
    config = AutoConfig.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    generation = GenerationConfig.from_model_config(config)
    # HF5.17 leaves unset members None until its generation preparation fills them.
    defaults = GenerationConfig._get_default_generation_params()
    def effective(key):
        value = getattr(generation, key)
        return defaults[key] if value is None else value
    return ({k: effective(k) for k in ("temperature", "top_k", "top_p")},
            {"kind": "HF GenerationConfig.from_model_config", "transformers": transformers.__version__,
             "do_sample": effective("do_sample"),
             "defaults_source_sha256": hashlib.sha256(inspect.getsource(GenerationConfig._get_default_generation_params).encode()).hexdigest(),
             "native_omitted_api_policy": "greedy; no checkpoint sampling overrides",
             "sampled_region": "explicit conditional HF sampling parameters"}, "sampled_hf_defaults")


def prepare(url, model, snapshot, corpus_path, out, *, depths, samples, cycles, reply_tokens, thinking, tasks,
            baseline, candidate, generation_policy="checkpoint"):
    positive(samples, "samples")
    positive(cycles, "cycles")
    if not reply_tokens or len(set(reply_tokens)) != len(reply_tokens):
        raise ValueError("reply budgets must be nonempty and unique")
    for tokens in reply_tokens:
        positive(tokens, "reply tokens")
    if not baseline or not candidate or baseline == candidate:
        raise ValueError("comparison needs two distinct nonempty variant names")
    root = Path(snapshot)
    config = json.loads((root / "config.json").read_text())
    card, sampling_source, sampling_label = sampling_policy(root, generation_policy)
    context = config.get("text_config", config)["max_position_embeddings"]
    positive(context, "checkpoint context")
    if (isinstance(card["temperature"], bool) or not isinstance(card["temperature"], (int, float))
            or not math.isfinite(card["temperature"]) or card["temperature"] <= 0
            or type(card["top_k"]) is not int or card["top_k"] <= 0
            or isinstance(card["top_p"], bool) or not isinstance(card["top_p"], (int, float))
            or not 0 < card["top_p"] <= 1):
        raise ValueError("checkpoint has no valid native sampling policy")
    data, corpus_sha = campaign.corpus_text(corpus_path)
    if not data:
        raise ValueError("prompt corpus is empty")
    items = []
    for depth in depths:
        positive(depth, "prompt depth")
        if depth + max(reply_tokens) > context:
            raise ValueError("prompt plus reply exceeds checkpoint context")
        for task in tasks:
            if task not in campaign.TASKS:
                raise ValueError("unknown maintained benchmark task: " + task)
            for mode in thinking:
                kwargs = {"enable_thinking": mode}

                def count(messages):
                    response = json_endpoint(url + "/v1/tokenize",
                                             {"model": model, "messages": messages,
                                              "chat_template_kwargs": kwargs})
                    return positive(response["count"], "rendered prompt count")

                messages, n = campaign.size_prompt(count, str(depth), task, data, depth)
                items.append({"id": f"{depth}-{task}-{'think' if mode else 'nothink'}",
                              "depth": depth, "task": task, "thinking": mode,
                              "messages": messages, "prompt_tokens": n})
    doc = {
        "schema": 1, "model": model, "model_revision": root.name, "checkpoint_context": context,
        "config_sha256": digest(root / "config.json"),
        "generation_config_sha256": sampling_source.get("sha256"),
        "sampling_source": sampling_source, "sampling_modes": ["greedy", sampling_label],
        "corpus_sha256": corpus_sha, "native_sampling": card, "reply_tokens": reply_tokens,
        "samples_per_start": samples, "baseline": baseline, "candidate": candidate,
        "cycles": [[baseline, candidate, candidate, baseline] if c % 2 == 0
                   else [candidate, baseline, baseline, candidate] for c in range(cycles)],
        "prompts": items, "prepared_server_stats": json_endpoint(url + "/v1/stats"),
        "comparison_contract": {
            "concurrency": 1, "arrival": "closed-loop; no overlapping requests",
            "fresh": "ordinary raw non-BOS one-token eviction then actual cached==0 required",
            "warm": "immediate same-body repeat; observed cached>0 required",
            "confidence_unit": "independent start positions aggregated within paired ABBA cycle",
            "selection": "lexicographic complete_body_s, ttft_client_s, decode_tps per fixed-output region; no regional losses offset",
            "quality": "cold/warm repeat token identity per seeded request; cross-configuration quality gated separately",
        },
    }
    write_json(out, doc)
    return doc


def body_for(manifest, item, sampling, sample, tokens):
    card = {"temperature": 0} if sampling == "greedy" else manifest["native_sampling"]
    return {"model": manifest["model"], "messages": item["messages"],
            "chat_template_kwargs": {"enable_thinking": item["thinking"]},
            "max_tokens": tokens, "ignore_eos": True, "return_token_ids": True,
            "seed": 1234 + sample, **card}


def measured_record(raw, expected_prompt, maximum, phase):
    ids = campaign.reply_token_ids(raw)
    usage, stats = raw["usage"], raw["stats"]
    protocol.completion_usage(usage, maximum, prompt=True)
    n, prompt = usage["completion_tokens"], usage["prompt_tokens"]
    if type(prompt) is not int or prompt != expected_prompt or n != maximum:
        raise ValueError("served prompt/reply token counts differ from the frozen workload")
    cached = stats.get("cached")
    if type(cached) is not int or not 0 <= cached <= prompt:
        raise ValueError("server did not report a valid cached prompt count")
    if (phase == "fresh" and cached != 0) or (phase == "warm" and cached == 0):
        raise ValueError("observed prefix cache state differs from the requested benchmark phase")
    prefill, decode = stats.get("prefill_s"), stats.get("decode_s")
    if isinstance(prefill, bool) or not isinstance(prefill, (float, int)) or not math.isfinite(prefill) or prefill < 0:
        raise ValueError("missing nonnegative finite engine timing: prefill_s")
    if prompt > cached and prefill == 0:
        raise ValueError("uncached prefill duration is below reported timer resolution")
    for name, value in [("decode_s", decode), ("decode_tps", stats.get("decode_tps"))]:
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError("missing positive finite engine timing: " + name)
    wall = raw["done"] - raw["sent"]
    if wall <= 0 or raw["first"] is None:
        raise ValueError("response has no valid client emission/complete timing")
    return {
        "token_ids": ids, "usage": usage, "stats": stats,
        "content": raw["content"], "reasoning": raw["reasoning"], "finish": raw["finish"],
        "tool_call_deltas": raw["tool_call_deltas"], "pieces": raw["pieces"],
        "client": {k: raw[k] - raw["sent"] if raw[k] is not None else None
                   for k in ("headers", "first", "last", "done")},
        "metrics": {"prefill_tps": (prompt - cached) / prefill if prompt > cached else None,
                    "decode_tps": stats["decode_tps"], "ttft_client_s": raw["first"] - raw["sent"],
                    "complete_body_tps": n / wall, "complete_body_s": wall},
        "timing_scope": "engine timings plus client SSE-arrival/complete-body timings; chunks can batch tokens",
    }


def run(url, manifest_path, meta_path, out, *, cycle, position, timeout):
    manifest = json.loads(Path(manifest_path).read_text())
    if not 0 <= cycle < len(manifest["cycles"]) or not 0 <= position < 4:
        raise ValueError("cycle/position outside frozen ABBA plan")
    meta = json.loads(Path(meta_path).read_text())
    if any(not meta.get(k) for k in IDENTITIES) or meta["model_revision"] != manifest["model_revision"]:
        raise ValueError("record image/source/model/launch/instance identities and match the frozen model revision")
    doc = {"schema": 1, "manifest_sha256": digest(manifest_path), "tool_sha256": digest(__file__),
           "cycle": cycle, "position": position, "variant": manifest["cycles"][cycle][position],
           "meta": meta, "host_before": campaign.host_snapshot(gpu=True),
           "server_before": json_endpoint(url + "/v1/stats"), "records": [], "status": "running"}
    write_json(out, doc)
    started = time.perf_counter()
    try:
        # Prime longest-sequence/verify shapes; keep the entire warmup fact separate from measured rows.
        longest = max(manifest["prompts"], key=lambda x: x["prompt_tokens"])
        doc["warmup"] = []
        for sampling in manifest.get("sampling_modes", ["greedy", "native"]):
            warm = campaign.stream_chat(url, body_for(manifest, longest, sampling, 0, max(manifest["reply_tokens"])), timeout=timeout,
                                        prompt_tokens=longest["prompt_tokens"], label="excluded warmup")
            campaign.reply_token_ids(warm)
            doc["warmup"].append(warm)
            write_json(out, doc)
        for item in manifest["prompts"]:
            for sampling in manifest.get("sampling_modes", ["greedy", "native"]):
                for tokens in manifest["reply_tokens"]:
                    for sample in range(manifest["samples_per_start"]):
                        request = body_for(manifest, item, sampling, sample, tokens)
                        campaign.stream_chat(
                            url, {"model": manifest["model"], "prompt": EVICTION_TOKENS,
                                  "max_tokens": 1, "temperature": 0, "ignore_eos": True}, timeout=timeout,
                            route="/v1/completions",
                            label="excluded prefix eviction")
                        previous = None
                        for phase in ("fresh", "warm"):
                            row = {"prompt": item["id"], "depth": item["depth"], "task": item["task"],
                                   "thinking": item["thinking"], "sampling": sampling, "sample": sample,
                                   "reply_tokens": tokens, "phase": phase, "status": "running", "offered_unix_s": time.time()}
                            doc["records"].append(row)
                            write_json(out, doc)
                            with campaign.MemSampler() as memory:
                                raw = campaign.stream_chat(url, request, timeout=timeout,
                                                           prompt_tokens=item["prompt_tokens"],
                                                           label=f"{item['id']} {sampling} reply{tokens} {sample} {phase}")
                            row["raw_stream_result"] = raw
                            row.update(measured_record(raw, item["prompt_tokens"], tokens, phase))
                            del row["raw_stream_result"]  # validated fields retained once; invalid raw responses stay on failure
                            if previous is not None and previous != row["token_ids"]:
                                raise ValueError("cold/warm repetition changed token IDs for an identical seeded request")
                            previous = row["token_ids"]
                            row["mem_min_available_mib"] = memory.low
                            row["status"] = "complete"
                            write_json(out, doc)
        doc["status"] = "complete"
        doc["host_after"] = campaign.host_snapshot(gpu=True)
        doc["server_after"] = json_endpoint(url + "/v1/stats")
    except BaseException as error:
        doc["status"] = "error"
        doc["error"] = {"type": type(error).__name__, "message": str(error),
                        "elapsed_s": time.perf_counter() - started}
        if doc["records"] and doc["records"][-1]["status"] == "running":
            doc["records"][-1]["status"] = "error"
        write_json(out, doc)
        raise
    write_json(out, doc)
    return doc


def analyze(manifest_path, paths):
    manifest = json.loads(Path(manifest_path).read_text())
    expected_hash = digest(manifest_path)
    runs = [json.loads(Path(p).read_text()) for p in paths]
    cells = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    seen_positions, seen_instances, issues = set(), set(), []
    versions, variant_identities = set(), {}
    for path, run_doc in zip(paths, runs):
        if run_doc["manifest_sha256"] != expected_hash:
            raise ValueError("run uses a different frozen workload")
        pair = (run_doc["cycle"], run_doc["position"])
        if pair in seen_positions:
            raise ValueError("duplicate independent start position")
        seen_positions.add(pair)
        instance = run_doc["meta"].get("instance_id")
        if any(not run_doc["meta"].get(k) for k in IDENTITIES):
            raise ValueError("run omits required source/image/launch/instance identity")
        if not instance or instance in seen_instances:
            raise ValueError("each confidence position needs a distinct server instance identity")
        seen_instances.add(instance)
        material = {k: run_doc["meta"].get(k) for k in MATERIAL_IDENTITIES}
        prior = variant_identities.setdefault(run_doc["variant"], material)
        if material != prior:
            raise ValueError("material configuration changed within one variant label")
        if run_doc["variant"] != manifest["cycles"][pair[0]][pair[1]]:
            raise ValueError("run label differs from the frozen comparison order")
        versions.add(run_doc["tool_sha256"])
        if run_doc["status"] != "complete":
            issues.append({"file": str(path), "reason": run_doc.get("error", "incomplete run")})
        expected_rows = len(manifest["prompts"]) * 2 * len(manifest["reply_tokens"]) * manifest["samples_per_start"] * 2
        if len(run_doc["records"]) != expected_rows:
            issues.append({"file": str(path), "reason": "request schedule incomplete",
                           "expected": expected_rows, "actual": len(run_doc["records"])})
        per_start = collections.defaultdict(lambda: collections.defaultdict(list))
        for row in run_doc["records"]:
            if row["status"] != "complete":
                issues.append({"file": str(path), "row": row["prompt"], "reason": row["status"]})
                continue
            key = "/".join(str(row[k]) for k in ("depth", "task", "thinking", "sampling", "reply_tokens", "phase"))
            for metric, value in row["metrics"].items():
                if value is not None:
                    per_start[key][metric].append(value)
        for key, metrics in per_start.items():
            for metric, values in metrics.items():
                cells[key][metric][run_doc["cycle"]].append(
                    (run_doc["variant"], statistics.fmean(values)))
    if len(versions) > 1:
        raise ValueError("mixed benchmark tool versions")
    missing = [(c, p) for c, order in enumerate(manifest["cycles"]) for p in range(len(order))
               if (c, p) not in seen_positions]
    result = {}
    for key, metrics in cells.items():
        result[key] = {}
        for metric, cycles in metrics.items():
            logs = []
            for cycle, values in cycles.items():
                base = [v for label, v in values if label == manifest["baseline"]]
                candidate = [v for label, v in values if label == manifest["candidate"]]
                if len(base) == 2 and len(candidate) == 2:
                    logs.append(statistics.fmean(map(math.log, candidate))
                                - statistics.fmean(map(math.log, base)))
            if logs:
                interval = campaign.interval(logs)
                middle, half = interval["mean"], interval["half95"]
                result[key][metric] = {"candidate_over_baseline": math.exp(middle),
                                       "low95": math.exp(middle - half) if half is not None else None,
                                       "high95": math.exp(middle + half) if half is not None else None,
                                       "independent_cycles": len(logs), "raw_cycle_log_ratios": logs,
                                       "preferred_direction": METRICS[metric]}
    regressions = []
    for region, metrics in result.items():
        for metric, estimate in metrics.items():
            low, high = estimate["low95"], estimate["high95"]
            if low is not None and ((METRICS[metric] == "lower" and low > 1)
                                    or (METRICS[metric] == "higher" and high < 1)):
                regressions.append({"region": region, "metric": metric, **estimate})
    return {"schema": 1, "complete": not issues and not missing, "issues": issues,
            "variant_identities": variant_identities,
            "missing_start_positions": missing, "cells": result,
            "objective_order": ["complete_body_s", "ttft_client_s", "decode_tps"],
            "regional_regressions": regressions,
            "interpretation": "Paired cycle ratios only; uncertainty is not equivalence or a global-fastest claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("url", "model", "snapshot", "corpus", "out"):
        p.add_argument(name)
    p.add_argument("--depths", default="512,4096,16384")
    p.add_argument("--tasks", default="code,prose")
    p.add_argument("--thinking", choices=("both", "on", "off"), default="both")
    p.add_argument("--samples", type=int, default=1,
                   help="correlated samples per independent start; CI units remain paired cycles")
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--reply-tokens", default="8,32,256")
    p.add_argument("--generation-policy", choices=("checkpoint", "hf-defaults"), default="checkpoint",
                   help="hf-defaults explicitly samples HF conditional defaults when checkpoint card is absent")
    p.add_argument("--baseline", default="baseline")
    p.add_argument("--candidate", default="candidate")
    r = commands.add_parser("run")
    for name in ("url", "manifest", "meta", "out"):
        r.add_argument(name)
    r.add_argument("--cycle", type=int, required=True)
    r.add_argument("--position", type=int, required=True)
    r.add_argument("--timeout", type=float, default=campaign.TIMEOUT_S)
    r.add_argument("--progress")
    r.add_argument("--hot-flag")
    a = commands.add_parser("analyze")
    a.add_argument("manifest")
    a.add_argument("out")
    a.add_argument("runs", nargs="+")
    args = parser.parse_args()
    if args.command == "prepare":
        thinking = [False, True] if args.thinking == "both" else [args.thinking == "on"]
        prepare(base_url(args.url), args.model, args.snapshot, args.corpus, args.out,
                depths=[int(x) for x in args.depths.split(",")], samples=args.samples, cycles=args.cycles,
                reply_tokens=[int(x) for x in args.reply_tokens.split(",")],
                thinking=thinking, tasks=args.tasks.split(","),
                baseline=args.baseline, candidate=args.candidate, generation_policy=args.generation_policy)
    elif args.command == "run":
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("timeout must be positive and finite")
        campaign.Progress.configure(args.progress)
        campaign.HotGate.configure(args.hot_flag)
        run(base_url(args.url), args.manifest, args.meta, args.out,
            cycle=args.cycle, position=args.position, timeout=args.timeout)
    else:
        write_json(args.out, analyze(args.manifest, args.runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
