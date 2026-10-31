"""Finite exact Gemma proposal qualification through one production CLI engine.

Run with --plan sealed.json --out NEW_DIRECTORY. The plan binds the consumed
prompts, installed source, named policies and every position before model work.
Proof instrumentation is removed before timing. This tool reports observations;
statistical selection and protected GPU admission belong to the operator.
Native wall timing spans generate through drain, after App.prepare/hot waits;
it is not an HTTP/client TTFT measurement. Prefix warmups have separate rows.
The screen warms each timed position. The holdout instead uses one baseline
warm per workload after both policies' full proofs, then six alternating
ABBA/BAAB blocks under warm-prompt/proof-prepared-kernel conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import struct
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from .benchmark_output import write_json
    from .campaign_bench import HotGate, Progress, host_snapshot
    from .campaign_kl import with_app
else:
    from benchmark_output import write_json
    from campaign_bench import HotGate, Progress, host_snapshot
    from campaign_kl import with_app


POLICIES = {f"q{q}-g{guard}-{repair}": (q == 64, guard, repair == "partial")
            for q in (256, 64) for guard in (1, 8, 32) for repair in ("lex", "partial")}
BASELINE = "q256-g1-lex"
WORKLOADS = {"code-sampled": 48, "history-chat-sampled": 8451}
SOURCES = ("tensorfold.cuda.speculative", "tensorfold.cuda.sampling",
           "tensorfold.families.gemma4.cuda.engine", "tensorfold.families.gemma4.cuda.dflash",
           "tensorfold.engine.rejection", "tensorfold.engine.exact_sampling")
COUNTS = ("rounds", "drafted", "accepted", "min_rows")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, limit):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"input exceeds its {limit}-byte bound: {path}")
    return raw


def load_plan(path):
    path = Path(path)
    raw = _read(path, 65536)
    plan = json.loads(raw)
    required = {"schema", "kind", "artifact_id", "source_sha256", "prompts_file", "policies",
                "reference", "blocks", "tokens", "serve_args", "port"}
    if not isinstance(plan, dict) or set(plan) != required or type(plan["schema"]) is not int or plan["schema"] != 1:
        raise ValueError("exact schema1 qualification plan fields required")
    policies = plan["policies"]
    if (not isinstance(policies, list) or not policies or any(type(p) is not str or p not in POLICIES for p in policies)
            or len(set(policies)) != len(policies) or plan["reference"] != BASELINE or BASELINE not in policies):
        raise ValueError("unique finite policies including the named q256/G1/lex reference required")
    if type(plan["tokens"]) is not int or type(plan["port"]) is not int or not 1 <= plan["port"] <= 65535:
        raise ValueError("integer reply budget and loopback port required")
    blocks = plan["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("explicit ordered blocks required")
    for block in blocks:
        if (not isinstance(block, dict) or set(block) != {"seed", "order"} or type(block["seed"]) is not int
                or not isinstance(block["order"], list) or any(type(p) is not str for p in block["order"])
                or set(block["order"]) != set(policies)):
            raise ValueError("every block must name its seed and complete policy order")
    if plan["kind"] == "screen":
        if (set(policies) != set(POLICIES) or plan["tokens"] != 64 or len(blocks) != 1
                or blocks[0]["seed"] != 7 or blocks[0]["order"] != policies + list(reversed(policies))):
            raise ValueError("screen requires all12 policies,64 tokens,seed7 and the declared order plus reverse")
    elif plan["kind"] == "holdout":
        other = [p for p in policies if p != BASELINE]
        if (len(other) != 1 or plan["tokens"] != 256 or [b["seed"] for b in blocks] != [13, 31, 47, 83, 101, 127]
                or [b["order"] for b in blocks] not in (
                    [[BASELINE, other[0], other[0], BASELINE], [other[0], BASELINE, BASELINE, other[0]]] * 3,
                    [[other[0], BASELINE, BASELINE, other[0]], [BASELINE, other[0], other[0], BASELINE]] * 3)):
            raise ValueError("holdout requires256 tokens and six seeded alternating ABBA/BAAB blocks")
    else:
        raise ValueError("kind must be screen or holdout")
    identity = plan["artifact_id"]
    if (type(identity) is not str or not identity.startswith("sha256:") or len(identity) != 71
            or any(c not in "0123456789abcdef" for c in identity[7:])):
        raise ValueError("operator-verified immutable sha256 image identity required")
    sources = plan["source_sha256"]
    if (not isinstance(sources, dict) or set(sources) != {"tool", *SOURCES}
            or any(type(s) is not str or len(s) != 64 or any(c not in "0123456789abcdef" for c in s)
                   for s in sources.values())):
        raise ValueError("exact tool and affected installed source SHA256 bindings required")
    serve = plan["serve_args"]
    if (not isinstance(serve, list) or not serve or serve[0] != "serve"
            or any(type(a) is not str for a in serve)
            or any(a.split("=", 1)[0] in {"--api-key", "--api-key-file"} for a in serve)):
        raise ValueError("nonsecret production serve arguments required; no HTTP authentication is used here")
    entry = plan["prompts_file"]
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"} or type(entry["path"]) is not str:
        raise ValueError("prompt path and consumed-byte SHA256 required")
    source = Path(entry["path"])
    prompt_raw = _read(source if source.is_absolute() else path.parent / source, 1 << 20)
    if _sha(prompt_raw) != entry["sha256"]:
        raise ValueError("prompt input hash changed")
    items = json.loads(prompt_raw).get("items")
    if (not isinstance(items, list) or len(items) != 2 or any(not isinstance(i, dict) for i in items)
            or {i.get("id") for i in items} != set(WORKLOADS)):
        raise ValueError("the exact code/history prompt pair is required")
    for item in items:
        if item.get("thinking") is not False or not isinstance(item.get("messages"), list):
            raise ValueError("declared nonthinking synthetic messages required")
    return plan, items, raw, prompt_raw


@contextmanager
def _bindings(changes):
    """Own only the finite internal module/class substitutions listed by this tool."""
    from tensorfold.cleanup import raise_failures

    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in changes]
    primary, errors = None, []
    try:
        try:
            for owner, name, value in changes:
                setattr(owner, name, value)
            yield
        except BaseException as error:  # noqa: BLE001 - restore all bindings before propagating interrupts/primary
            primary = error
    finally:
        for owner, name, value in reversed(originals):
            try:
                setattr(owner, name, value)
            except BaseException as error:  # noqa: BLE001 - every restoration failure belongs to this outcome
                errors.append(error)
        try:
            raise_failures(primary, errors)
        finally:
            primary = errors = originals = None


def legacy_order(values, ids, kept):
    """Independent historical oracle: no partial/partition threshold mechanism."""
    import numpy as np

    return np.lexsort((ids, -values))[:kept]


@contextmanager
def policy(speculative, sampling, name, chain_type, partial):
    bounded, guard, use_partial = POLICIES[name]

    def construct(*args, **kwargs):
        if not bounded and kwargs.get("sampled"):
            kwargs.pop("sampling", None)  # the unchanged default is the historical canonical256 support
        return chain_type(*args, **kwargs)

    with _bindings([(speculative, "_PROPOSAL_GUARD", guard), (speculative, "Chains", construct),
                    (sampling, "_canonical_repair_order", partial if use_partial else legacy_order)]):
        yield


@contextmanager
def diagnostics(speculative, sampling, chain_type, assets, outdir):
    """Proof-only q-byte hashes and already-read repair assets; no extra device readback."""
    import numpy as np

    digest = hashlib.sha256()
    state = {"q_packets": 0, "q_rows": 0, "proposal_repairs": 0, "target_repairs": 0}
    sample = chain_type.sample

    def observed_sample(*args, **kwargs):
        packet = sample(*args, **kwargs)
        digest.update(struct.pack("<QQ", packet.first_position, len(packet.tokens)))
        digest.update(np.asarray(packet.tokens, dtype="<i8").tobytes())
        for row in packet.q_rows:
            digest.update(struct.pack("<Q", len(row.ids)))
            digest.update(np.asarray(row.ids, dtype="<i8").tobytes())
            digest.update(np.asarray(row.probs, dtype="<f8").tobytes())
        state["q_packets"] += 1
        state["q_rows"] += len(packet.q_rows)
        state["q_sha256"] = digest.hexdigest()
        return packet

    def observed_canonical(kind, original):
        def canonical(values, ids, count, width, full_row):
            def complete(row):
                data, tokens = full_row(row)
                state[kind + "_repairs"] += 1
                if len(assets) < 2:
                    name = f"repair-{len(assets)}.npz"
                    target = outdir / name
                    asset = {"path": name, "status": "WRITING", "kind": kind, "width": width,
                             "kept": min(count, width), "dtype": str(data.dtype)}
                    assets.append(asset)
                    with target.open("xb") as stream:
                        np.savez(stream, values=data, ids=tokens)
                    asset.update(status="COMPLETE", sha256=_sha(target.read_bytes()))
                return data, tokens
            return original(values, ids, count, width, complete)
        return canonical

    with _bindings([(chain_type, "sample", observed_sample),
                    (speculative, "canonical_candidates", observed_canonical("proposal", speculative.canonical_candidates)),
                    (sampling, "canonical_candidates", observed_canonical("target", sampling.canonical_candidates))]):
        yield state


def _matrix(plan):
    rows = []
    def add(**fields):
        rows.append({"case": len(rows), "status": "NOT_RUN", **fields})
    for workload in WORKLOADS:
        add(phase="cold", workload=workload, policy=BASELINE, mode="greedy-serial", seed=7)
        for name in plan["policies"]:
            for mode in ("greedy-draft", "sampled-draft"):
                add(phase="proof", workload=workload, policy=name, mode=mode, seed=7)
    for workload in WORKLOADS:
        if plan["kind"] == "holdout":
            # This prepares the workload's complete prefix. Both policies and
            # fixed request shapes were already exercised by the full proofs.
            add(phase="warm", workload=workload, policy=BASELINE, mode="sampled-draft",
                seed=13, block=0, position=-1)
        for block, entry in enumerate(plan["blocks"]):
            for position, name in enumerate(entry["order"]):
                for phase in ("warm", "timed") if plan["kind"] == "screen" else ("timed",):
                    add(phase=phase, workload=workload, policy=name, mode="sampled-draft",
                        seed=entry["seed"], block=block, position=position)
    return rows


def _attempt(engine, prepared, row, report, destination, *, audit=None):
    from tensorfold.cleanup import raise_failures

    row["status"] = "RUNNING"
    write_json(destination, report)
    output, first, last = [], None, None
    primary, errors = None, []
    retired = False

    def receive(tokens):
        nonlocal first, last
        now = time.perf_counter()
        if first is None:
            first = now
        last = now
        output.extend(tokens)
        return False

    try:
        try:
            row["hot_wait_s"] = HotGate.wait()
            engine.drain()
            if row["phase"] == "cold":
                with engine._prefix_lock:
                    engine._prefix = None
            row["host_before"] = host_snapshot()
            began = time.perf_counter()
            stats = engine.generate(prepared.prompt, prepared.max_tokens, prepared.sampling, receive,
                                    draft=row["mode"] != "greedy-serial", stop_eos=False,
                                    penalties=prepared.penalties)
            returned = time.perf_counter()
            engine.drain()
            retired = True
            settled = time.perf_counter()
            json.dumps(stats, allow_nan=False)  # malformed metrics must not poison the owning error report
            if any(type(stats[key]) is not int or stats[key] < 0 for key in COUNTS):
                raise ValueError("native proposal counters must be nonnegative integers")
            row["stats"] = stats
            row.update(generated_s=returned - began, settled_wall_s=settled - began,
                       deferred_tail_s=settled - returned, first_s=None if first is None else first - began,
                       last_s=None if last is None else last - began)
            if len(output) != prepared.max_tokens:
                raise RuntimeError("request did not return its complete declared token count")
            expected_cache = 0 if row["phase"] == "cold" else len(prepared.prompt)
            if row["phase"] != "warm" and row["stats"]["cached"] != expected_cache:
                raise RuntimeError("request did not exercise its declared prefix-cache state")
            if audit is not None and row["mode"] == "sampled-draft" and not audit["q_packets"]:
                raise RuntimeError("sampled proof did not exercise actual q packets")
        except BaseException as error:  # noqa: BLE001 - retain partial IDs and primary while settling native work
            primary = error
        if not retired:
            try:
                engine.drain()
            except BaseException as error:  # noqa: BLE001 - completion failure is part of the owning outcome
                errors.append(error)
        row.update(token_ids=output, token_sha256=_sha(json.dumps(output, separators=(",", ":")).encode()),
                   status="ERROR" if primary is not None or errors else "PASS")
        if audit is not None:
            row["proof"] = dict(audit)
        if primary is not None or errors:
            row["errors"] = ["".join(traceback.format_exception(e))
                             for e in ([primary] if primary is not None else []) + errors]
        try:
            write_json(destination, report)
        except BaseException as error:  # noqa: BLE001 - preserve request/settlement errors if evidence fails
            raise_failures(primary, [*errors, error])
        raise_failures(primary, errors)
        Progress.update("idle", case=row["case"], qualification_phase=row["phase"], workload=row["workload"])
        print(json.dumps({k: row[k] for k in ("case", "phase", "workload", "policy", "mode", "status",
                                             "token_sha256")}), flush=True)
    finally:
        primary = errors = None


def _compare(rows, reference, *, proof):
    for row in rows:
        if row["status"] != "PASS":
            raise RuntimeError("incomplete qualification matrix")
        if row["phase"] == "cold":
            continue
        context = f"case{row['case']} {row['workload']}/{row['policy']}/{row['mode']}"
        if proof:
            baseline = next(r for r in rows if r["phase"] == "proof" and r["workload"] == row["workload"]
                            and r["mode"] == row["mode"] and r["policy"] == reference)
            if row["mode"] == "greedy-draft":
                serial = next(r for r in rows if r["phase"] == "cold" and r["workload"] == row["workload"])
                if row["token_ids"] != serial["token_ids"]:
                    row["comparison"] = "ERROR"
                    raise RuntimeError(f"{context}: greedy drafting changed complete serial token IDs")
            elif row["proof"] != baseline["proof"]:
                # Repair counts legitimately differ; only logical packet bytes and geometry must match.
                for key in ("q_packets", "q_rows", "q_sha256"):
                    if row["proof"].get(key) != baseline["proof"].get(key):
                        row["comparison"] = "ERROR"
                        raise RuntimeError(f"{context}: candidate changed actual logical q bytes")
        else:
            baseline = next(r for r in rows if r["phase"] == "timed" and r["workload"] == row["workload"]
                            and r["seed"] == row["seed"] and r["block"] == row["block"]
                            and r["policy"] == reference)
        if row["token_ids"] != baseline["token_ids"]:
            row["comparison"] = "ERROR"
            raise RuntimeError(f"{context}: candidate changed complete same-policy token IDs")
        if any(row["stats"][key] != baseline["stats"][key] for key in COUNTS):
            row["comparison"] = "ERROR"
            raise RuntimeError(f"{context}: candidate changed mode-correct proposal/acceptance geometry")
        row["comparison"] = "PASS"


def _anchor_phases(proof_rows, timing_rows):
    """Removing proof observers must preserve each covered request's complete result.

    Holdout seeds deliberately differ from the seed7 proof. They retain their
    own cross-arm replay checks, without claiming q-byte proof at those seeds.
    """
    def key(row):
        return row["workload"], row["policy"], row["mode"], row["seed"]
    anchors = {key(row): row for row in proof_rows if row["phase"] == "proof"}
    for row in timing_rows:
        anchor = anchors.get(key(row))
        if anchor is None:
            row["proof_anchor"] = {"status": "NOT_COVERED", "reason": "no proof for this request seed"}
            continue
        row["proof_anchor"] = {"status": "ERROR", "case": anchor["case"]}
        context = f"case{row['case']} {row['workload']}/{row['policy']}/{row['mode']}"
        if row["token_ids"] != anchor["token_ids"]:
            raise RuntimeError(f"{context}: proof-to-timing complete token IDs changed")
        if any(row["stats"][field] != anchor["stats"][field] for field in COUNTS):
            raise RuntimeError(f"{context}: proof-to-timing proposal/acceptance geometry changed")
        row["proof_anchor"]["status"] = "MATCH"


def qualify(app, plan, items, report, outdir, modules):
    import numpy as np
    import torch
    import triton

    engine = app.engine
    if (not isinstance(engine, modules[SOURCES[2]].Gemma4Engine) or engine.w.numerics != "ct_nvfp4"
            or engine.max_rows != 5 or engine.graphs is None or engine.head is None):
        raise ValueError("this finite comparison requires original Gemma CT NVFP4,rows5,graphs and a companion")
    speculative, sampling = modules[SOURCES[0]], modules[SOURCES[1]]
    chain_type, partial = speculative.Chains, sampling._canonical_repair_order
    report["runtime"] = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                         "numpy": np.__version__, "triton": triton.__version__, "device": torch.cuda.get_device_name(),
                         "capability": torch.cuda.get_device_capability(),
                         "affinity": sorted(os.sched_getaffinity(0)), "context": engine.context_window,
                         "verify_rows": engine.max_rows,
                         "source_paths": {name: str(module.__file__) for name, module in modules.items()}}
    bodies = {item["id"]: item for item in items}
    prepared = {}
    for row in report["records"]:
        key = row["workload"], row["mode"], row["seed"]
        if key in prepared:
            continue
        body = {"messages": bodies[key[0]]["messages"], "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": plan["tokens"], "ignore_eos": True, "temperature": 0 if key[1] == "greedy-serial"
                or key[1] == "greedy-draft" else 1., "top_k": 64, "top_p": .95, "min_p": 0., "seed": key[2]}
        request = app.prepare(body, True)
        if (len(request.prompt) != WORKLOADS[key[0]] or request.max_tokens != plan["tokens"]
                or request.thinking or not request.ignore_eos or request.grammar is not None
                or request.vision is not None or request.penalties is not None):
            raise ValueError("prepared request differs from the declared text/default-penalty region")
        if key[1] == "sampled-draft":
            actual = vars(request.sampling) if request.sampling is not None else None
            if actual != {"seed": key[2], "temperature": 1., "top_k": 64, "top_p": .95, "min_p": 0.}:
                raise ValueError("prepared sampling differs from the fixed T1/k64/p.95 request")
        elif request.sampling is not None:
            raise ValueError("greedy controls must use the canonical greedy request policy")
        prepared[key] = request
    report["prepared"] = [{"workload": k[0], "mode": k[1], "seed": k[2], "prompt_tokens": len(v.prompt),
                           "prompt_sha256": _sha(json.dumps(v.prompt, separators=(",", ":")).encode()),
                           "sampling": None if v.sampling is None else vars(v.sampling)} for k, v in prepared.items()]
    assets = report["repair_assets"]
    destination = outdir / "report.json"
    proof_rows = [r for r in report["records"] if r["phase"] in {"cold", "proof"}]
    for row in proof_rows:
        with policy(speculative, sampling, row["policy"], chain_type, partial), \
                diagnostics(speculative, sampling, chain_type, assets, outdir) as audit:
            _attempt(engine, prepared[row["workload"], row["mode"], row["seed"]], row, report, destination,
                     audit=audit)
    _compare(proof_rows, plan["reference"], proof=True)
    timing_rows = [r for r in report["records"] if r["phase"] in {"warm", "timed"}]
    for row in timing_rows:
        with policy(speculative, sampling, row["policy"], chain_type, partial):
            _attempt(engine, prepared[row["workload"], row["mode"], row["seed"]], row, report, destination)
    _compare(timing_rows, plan["reference"], proof=False)
    _anchor_phases(proof_rows, timing_rows)


def main(argv=None):
    from tensorfold.cleanup import raise_failures

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan, items, raw, prompt_raw = load_plan(args.plan)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "plan.json").write_bytes(raw)
    (args.out / "prompts.json").write_bytes(prompt_raw)
    report = {"schema": 1, "status": "RUNNING", "plan_sha256": _sha(raw), "prompts_sha256": _sha(prompt_raw),
              "artifact_id": plan["artifact_id"], "records": _matrix(plan), "repair_assets": [],
              "scope": "finite unprofiled exact-policy screen/holdout; statistical selection is external",
              "instrumentation": "timing retains finite policy factory/binding shims and stats; q/repair observers "
                                 "are proof-only; selected shipping source still requires its final recheck",
              "cold_scope": "one deliberate prefix-cold preparation/workload; kernel/grid first use retained in proof",
              "cache_condition": ("warm complete prompt and proof-prepared kernels; one baseline seed13 warm/workload, "
                                  "without per-position same-completion warming" if plan["kind"] == "holdout"
                                  else "per-position complete-request warmup before each timed request"),
              "reserved_guard": 32}
    destination = args.out / "report.json"
    write_json(destination, report)
    primary, errors = None, []
    try:
        try:
            modules = {name: importlib.import_module(name) for name in SOURCES}
            actual = {name: _sha(Path(module.__file__).read_bytes()) for name, module in modules.items()}
            actual["tool"] = _sha(Path(__file__).read_bytes())
            report["source_sha256"] = actual
            if actual != plan["source_sha256"]:
                raise ValueError("installed source differs from the sealed plan")
            HotGate.configure(None)
            speculative = modules[SOURCES[0]]
            # Admission and final CLI retirement see the maximum workspace. Inner arm reductions fit it.
            with _bindings([(speculative, "_PROPOSAL_GUARD", 32)]):
                with_app(plan["serve_args"], plan["port"],
                         lambda app: qualify(app, plan, items, report, args.out, modules))
            report["status"] = "PASS"
        except BaseException as error:  # noqa: BLE001 - terminal report retains all failures/interrupts and not-run rows
            primary = error
            report.update(status="ERROR", error="".join(traceback.format_exception(error)))
        try:
            write_json(destination, report)
        except BaseException as error:  # noqa: BLE001 - evidence failure cannot replace model/cleanup primary
            errors.append(error)
        raise_failures(primary, errors)
    finally:
        primary = errors = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
