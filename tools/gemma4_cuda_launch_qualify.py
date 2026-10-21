"""Finite production-App Gemma launch screen; no automatic selection or holdout claim.

One immutable rows/precision pair owns each engine. Exact H16/H8 banks share
the original BF16 weight; only prepared requests run, with ignore_eos retained.
All preparation, proof, warm, timed and not-run positions are declared first.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import stat
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from . import gemma4_cuda_head_bench as head_bench
    from . import gemma4_cuda_proposal_qualify as proposal
else:
    import gemma4_cuda_head_bench as head_bench
    import gemma4_cuda_proposal_qualify as proposal


SOURCES = (*proposal.SOURCES, "tensorfold.families.gemma4.cuda.head",
           "tensorfold.families.gemma4.cuda.graphs", "tensorfold.families.gemma4.cuda.state",
           "tensorfold.families.gemma4.cuda.forward", "tensorfold.families.qwen3_5.cuda.b16")
SOURCE_KEYS = {"tool", "proposal_helper", "head_helper", "b16.cu", "b16.cpp", *SOURCES}
TEMPLATE_SHA256 = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
MODES = ("greedy-serial", "greedy-draft", "sampled-serial", "sampled-draft")
HEAD_ORDER = (16, 8, 8, 16)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _hash(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _read(path, limit):
    """Read actual bounded bytes from a retained regular FD, including HF symlink targets."""
    from tensorfold.cleanup import raise_failures

    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    stream, primary, errors, raw = None, None, [], None
    try:
        try:
            _require(stat.S_ISREG(os.fstat(fd).st_mode), "input must resolve to a regular file")
            stream = os.fdopen(fd, "rb", closefd=False)
            raw = stream.read(limit + 1)
            _require(len(raw) <= limit, "input exceeds its actual-byte limit")
        except BaseException as error:  # noqa: BLE001 - preserve input failure through both owned closes
            primary = error
        if stream is not None:
            try:
                stream.close()
            except BaseException as error:  # noqa: BLE001 - cleanup belongs to this input operation
                errors.append(error)
        try:
            os.close(fd)
        except BaseException as error:  # noqa: BLE001 - never retry a potentially reused descriptor
            errors.append(error)
        raise_failures(primary, errors)
        return raw
    finally:
        primary = errors = None


def _option(argv, flag):
    _require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv), f"one explicit {flag} value required")
    return argv[argv.index(flag) + 1]


def load_plan(path):
    path = Path(path)
    raw = _read(path, 65536)
    plan = json.loads(raw)
    fields = {"schema", "kind", "artifact_id", "source_sha256", "prompts_file", "prompt_sha256",
              "assets", "policy", "configurations", "serial_ids", "tokens", "seed", "serve_args", "port"}
    _require(type(plan) is dict and set(plan) == fields and type(plan["schema"]) is int
             and plan["schema"] == 1 and plan["kind"] == "launch-screen", "exact launch-screen schema1 required")
    _require(type(plan["tokens"]) is int and plan["tokens"] == 64 and type(plan["seed"]) is int
             and plan["seed"] == 7, "screen requires64 reply tokens and seed7")
    _require(type(plan["policy"]) is str and plan["policy"] in proposal.POLICIES, "one sealed finite sampler policy required")
    identity = plan["artifact_id"]
    _require(type(identity) is str and identity.startswith("sha256:") and _hash(identity[7:]),
             "actual immutable image sha256 identity required")
    sources = plan["source_sha256"]
    _require(type(sources) is dict and set(sources) == SOURCE_KEYS and all(_hash(v) for v in sources.values()),
             "exact installed/tool/helper source hashes required")
    configs = plan["configurations"]
    _require(type(configs) is list and 1 <= len(configs) <= 4, "one through four explicit constructor configurations required")
    pairs = []
    for cfg in configs:
        _require(type(cfg) is dict and set(cfg) == {"rows", "bits"} and type(cfg["rows"]) is int
                 and cfg["rows"] in (3, 5, 11, 16) and type(cfg["bits"]) is int and cfg["bits"] in (8, 16),
                 "only rows3/5/11/16 and proposal bits8/16 are supported")
        pairs.append((cfg["rows"], cfg["bits"]))
    _require(len(set(pairs)) == len(pairs), "constructor configurations must be unique")
    _require(type(plan["port"]) is int and 1 <= plan["port"] <= 65535, "explicit loopback port required")
    serve = plan["serve_args"]
    _require(type(serve) is list and len(serve) >= 2 and serve[0] == "serve" and all(type(x) is str for x in serve),
             "production serve argument list required")
    forbidden = {"--draft-rows", "--draft-depth", "--drafter-bits", "--api-key", "--api-key-file", "--no-thinking",
                 "--no-drafts", "--tokenizer", "--vision-offload"}
    _require(not any(x.split("=", 1)[0] in forbidden for x in serve), "base arguments contain controlled or secret options")
    bound = {"--thinking-budget", "--context", "--drafter", "--chat-template"}
    _require(not any("=" in x and x.split("=", 1)[0] in bound for x in serve),
             "bound input options require their single explicit argument form")
    _require("--thinking" in serve and _option(serve, "--thinking-budget") == "0"
             and _option(serve, "--context") in ("0", "262144") and "--vision" in serve,
             "native context, resident vision and globally enabled unlimited thinking must be retained")
    assets = plan["assets"]
    _require(type(assets) is dict and set(assets) == {"model_config", "drafter_config", "tokenizer", "template"},
             "exact model/drafter/tokenizer/template input bindings required")
    consumed = {}
    for name, entry in {**assets, "prompts": plan["prompts_file"]}.items():
        _require(type(entry) is dict and set(entry) == {"path", "sha256"} and type(entry["path"]) is str
                 and _hash(entry["sha256"]), "input path and SHA256 required")
        actual = Path(entry["path"])
        actual = actual if actual.is_absolute() else path.parent / actual
        data = _read(actual, 64 << 20 if name == "tokenizer" else 1 << 20)
        _require(proposal._sha(data) == entry["sha256"], f"consumed {name} input hash changed")
        consumed[name] = {"path": str(actual.resolve()), "sha256": entry["sha256"], "bytes": len(data)}
        if name == "prompts":
            prompt_raw = data
    _require(assets["template"]["sha256"] == TEMPLATE_SHA256, "the fixed production Google template is required")
    expected = {"model_config": Path(serve[1]) / "config.json",
                "drafter_config": Path(_option(serve, "--drafter")) / "config.json",
                "tokenizer": Path(serve[1]) / "tokenizer.json", "template": Path(_option(serve, "--chat-template"))}
    _require(all(str(value.resolve()) == consumed[name]["path"] for name, value in expected.items()),
             "serve inputs differ from the consumed immutable assets")
    body = json.loads(prompt_raw)
    items = body.get("items") if type(body) is dict else None
    _require(type(items) is list and len(items) == 2 and all(type(i) is dict for i in items)
             and {i.get("id") for i in items} == set(proposal.WORKLOADS), "exact code/history prompt pair required")
    _require(all(i.get("thinking") is False and type(i.get("messages")) is list for i in items),
             "declared nonthinking voice-like messages required")
    _require(type(plan["prompt_sha256"]) is dict and set(plan["prompt_sha256"]) == set(proposal.WORKLOADS)
             and all(_hash(v) for v in plan["prompt_sha256"].values()), "prepared reference prompt ID hashes required")
    serial = plan["serial_ids"]
    _require(type(serial) is dict and set(serial) == set(proposal.WORKLOADS), "sealed cross-job serial references required")
    for modes in serial.values():
        _require(type(modes) is dict and set(modes) == {"greedy-serial", "sampled-serial"}
                 and all(type(ids) is list and len(ids) == 64 and all(type(t) is int and 0 <= t < 262144 for t in ids)
                         for ids in modes.values()), "complete native-vocabulary serial reference IDs required")
    return plan, items, raw, prompt_raw, consumed


def _matrix(plan):
    rows = []
    for cfg_index, _cfg in enumerate(plan["configurations"]):
        for workload in proposal.WORKLOADS:
            for head in (16, 8):
                for mode, phase in (("greedy-serial", "cold"), ("greedy-draft", "proof"),
                                    ("sampled-draft", "cold"), ("sampled-draft", "proof"),
                                    ("sampled-serial", "proof")):
                    rows.append({"configuration": cfg_index, "workload": workload, "head": head,
                                 "mode": mode, "phase": phase, "status": "NOT_RUN"})
        for workload in proposal.WORKLOADS:
            for position, head in enumerate(HEAD_ORDER):
                for phase in ("warm", "timed"):
                    rows.append({"configuration": cfg_index, "workload": workload, "head": head,
                                 "mode": "sampled-draft", "phase": phase, "position": position, "status": "NOT_RUN"})
    for index, row in enumerate(rows):
        row.update(case=index, seed=7, policy=plan["policy"])
    return rows


def _prepare(app, plan, items):
    prepared, metadata = {}, []
    for item in items:
        for mode in MODES:
            sampled = mode.startswith("sampled")
            request = app.prepare({"messages": item["messages"], "max_tokens": 64, "ignore_eos": True,
                                   "chat_template_kwargs": {"enable_thinking": False}, "temperature": 1. if sampled else 0,
                                   "top_k": 64, "top_p": .95, "min_p": 0., "seed": 7}, True)
            digest = proposal._sha(json.dumps(request.prompt, separators=(",", ":")).encode())
            _require(len(request.prompt) == proposal.WORKLOADS[item["id"]] and digest == plan["prompt_sha256"][item["id"]]
                     and request.max_tokens == 64 and not request.thinking and request.ignore_eos
                     and request.grammar is None and request.vision is None and request.penalties is None,
                     "production prepared request differs from the sealed text/input/cache region")
            actual = None if request.sampling is None else vars(request.sampling)
            _require(actual == ({"seed": 7, "temperature": 1., "top_k": 64, "top_p": .95, "min_p": 0.}
                                if sampled else None), "prepared sampling differs from the fixed request policy")
            prepared[item["id"], mode] = request
            metadata.append({"workload": item["id"], "mode": mode, "prompt_tokens": len(request.prompt),
                             "prompt_sha256": digest, "sampling": actual})
    return prepared, metadata


def _attempt(engine, request, row, report, destination, *, audit=None):
    """The frozen proposal helper cannot dispatch sampled-serial; keep this explicit four-mode boundary."""
    import torch

    from tensorfold.cleanup import raise_failures

    row["status"] = "RUNNING"
    proposal.write_json(destination, report)
    output, primary, errors, retired = [], None, [], False
    first = last = None
    def receive(tokens):
        nonlocal first, last
        now = time.perf_counter()
        first = now if first is None else first
        last = now
        output.extend(tokens)
        return False
    try:
        try:
            row["hot_wait_s"] = proposal.HotGate.wait()
            engine.drain()
            if row["phase"] == "cold":
                with engine._prefix_lock:
                    engine._prefix = None
            row["host_before"] = proposal.host_snapshot()
            row["allocator_before"] = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}
            torch.cuda.reset_peak_memory_stats()
            proposal.Progress.update("prefill", request=str(row["case"]), prompt_tokens=len(request.prompt), max_tokens=64)
            began = time.perf_counter()
            stats = engine.generate(request.prompt, 64, request.sampling, receive,
                                    draft=row["mode"].endswith("-draft"), stop_eos=False, penalties=request.penalties)
            returned = time.perf_counter()
            engine.drain()
            retired = True
            settled = time.perf_counter()
            row["allocator_after"] = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                                      "peak_allocated": torch.cuda.max_memory_allocated(),
                                      "peak_reserved": torch.cuda.max_memory_reserved()}
            json.dumps(stats, allow_nan=False)
            _require(all(type(stats[k]) is int and stats[k] >= 0 for k in proposal.COUNTS), "invalid native counters")
            row.update(stats=stats, generated_s=returned - began, settled_wall_s=settled - began,
                       deferred_tail_s=settled - returned, first_s=None if first is None else first - began,
                       last_s=None if last is None else last - began)
            _require(len(output) == 64, "request did not return all64 declared tokens")
            _require(row["phase"] == "warm" or stats["cached"] == (0 if row["phase"] == "cold" else len(request.prompt)),
                     "request did not exercise its declared prefix-cache state")
            if row["mode"].endswith("-draft"):
                _require(stats["drafted"] > 0, "requested draft route was not exercised")
            if audit is not None and row["mode"] == "sampled-draft":
                _require(audit["q_packets"] > 0, "sampled proof did not exercise q packets")
        except BaseException as error:  # noqa: BLE001 - own partial IDs through settlement and publication
            primary = error
        if not retired:
            try:
                engine.drain()
            except BaseException as error:  # noqa: BLE001 - deferred failure belongs to this request
                errors.append(error)
        row.update(token_ids=output, token_sha256=proposal._sha(json.dumps(output, separators=(",", ":")).encode()),
                   status="ERROR" if primary is not None or errors else "PASS")
        if audit is not None:
            row["proof"] = dict(audit)
        if primary is not None or errors:
            row["errors"] = ["".join(traceback.format_exception(e))
                             for e in ([primary] if primary is not None else []) + errors]
        try:
            proposal.write_json(destination, report)
        except BaseException as publication:  # noqa: BLE001 - evidence may not replace the request failure
            raise_failures(primary, [*errors, publication])
        raise_failures(primary, errors)
        proposal.Progress.update("idle", case=row["case"], qualification_phase=row["phase"])
    finally:
        primary = errors = None


@contextmanager
def _inputs(engine):
    """Proof-only counterpart of c36 capture; it observes the prepared request instead of running its own."""
    import torch

    from tensorfold.cleanup import raise_failures

    captured, state = {}, {"graph_calls": 0}
    target, draft, graph = engine.target.project_logits, engine.head.project_logits, engine.graphs.forward
    hooks = [(owner, name, name in vars(owner), vars(owner).get(name)) for owner, name in
             ((engine.target, "project_logits"), (engine.head, "project_logits"), (engine.graphs, "forward"))]
    def remember(kind, hidden):
        if kind not in captured or hidden.shape[0] > captured[kind].shape[0]:
            captured[kind] = hidden.detach().clone()
    def observe(kind, call):
        def observed(hidden):
            if not torch.cuda.is_current_stream_capturing():
                remember(kind, hidden)
            return call(hidden)
        return observed
    def observed_graph(*args, **kwargs):
        result = graph(*args, **kwargs)
        remember("target", result.hidden)
        state["graph_calls"] += 1
        return result
    primary, errors = None, []
    try:
        try:
            engine.target.project_logits = observe("target", target)
            engine.head.project_logits = observe("dflash", draft)
            engine.graphs.forward = observed_graph
            yield captured, state
        except BaseException as error:  # noqa: BLE001 - restore every borrowed method before propagation
            primary = error
    finally:
        for owner, name, owned, original in reversed(hooks):
            try:
                if owned:
                    setattr(owner, name, original)
                elif name in vars(owner):
                    delattr(owner, name)
            except BaseException as error:  # noqa: BLE001 - all restoration failures belong to the outcome
                errors.append(error)
        try:
            engine.drain()
        except BaseException as error:  # noqa: BLE001 - observation does not prove native completion
            errors.append(error)
        try:
            raise_failures(primary, errors)
        finally:
            primary = errors = None


def _bits(engine, heads, incumbent, captured, bits, *, draft):
    import torch

    selected = {kind: value for kind, value in captured.items() if kind == "target" or bits == 16}
    _require("target" in selected and (not draft or bits == 8 or "dflash" in selected),
             "actual head-input roles were not captured")
    metadata = {}
    with torch.cuda.stream(engine._stream), torch.inference_mode():
        for kind, hidden in selected.items():
            expected = incumbent(hidden)
            for head in heads.values():
                _require(torch.equal(head(hidden).view(torch.int16), expected.view(torch.int16)),
                         "exact head changed actual BF16 score bits")
            metadata[kind] = {"shape": list(hidden.shape),
                              "sha256": proposal._sha(hidden.contiguous().view(torch.uint8).cpu().numpy().tobytes())}
            del expected
    engine.drain()
    return metadata


def _compare(rows, serial_ids):
    _require(all(r["status"] == "PASS" for r in rows), "incomplete launch matrix")
    for row in rows:
        if row["mode"].endswith("-serial"):
            _require(row["token_ids"] == serial_ids[row["workload"]][row["mode"]],
                     "serial IDs changed across geometry/precision/exact heads")
    for row in rows:
        key = row["configuration"], row["workload"], row["mode"]
        reference = next(r for r in rows if (r["configuration"], r["workload"], r["mode"]) == key
                         and r["head"] == 16 and r["phase"] in ("cold", "proof"))
        _require(row["token_ids"] == reference["token_ids"], "same-config replay or exact-head IDs changed")
        if row["mode"] == "greedy-draft":
            _require(row["token_ids"] == serial_ids[row["workload"]]["greedy-serial"], "greedy draft changed serial IDs")
        if row["mode"] == "sampled-draft" and "proof" in row:
            for name in ("q_packets", "q_rows", "q_sha256"):
                _require(row["proof"].get(name) == reference["proof"].get(name), "same-config exact-head q bytes changed")
        if row["mode"].endswith("-draft"):
            _require(all(row["stats"][k] == reference["stats"][k] for k in proposal.COUNTS),
                     "same-config exact-head proposal counters changed")
        row["comparison"] = "PASS"


def qualify(app, cfg_index, plan, items, report, outdir, modules):
    import numpy as np
    import torch
    import triton

    engine, cfg = app.engine, plan["configurations"][cfg_index]
    _require(isinstance(engine, modules[proposal.SOURCES[2]].Gemma4Engine) and engine.w.numerics == "ct_nvfp4"
             and engine.max_rows == cfg["rows"] and engine.head is not None and engine.head.bits == cfg["bits"]
             and engine.context_window == 262144 and engine.vision is not None and engine.graphs is not None,
             "constructed engine differs from the fixed production launch descriptor")
    head_module = modules["tensorfold.families.gemma4.cuda.head"]
    _require(isinstance(engine.w.embed, head_module._RowsHead), "prepared GB10 exact BF16 head required")
    incumbent = engine.w.embed.incumbent
    heads = {n: engine.w.embed if engine.w.embed.reuse_rows == n else head_module.prepare_head(incumbent, reuse_rows=n)
             for n in (16, 8)}
    prepared, metadata = _prepare(app, plan, items)
    info = report["configurations"][cfg_index]
    info.update(status="READY", prepared=metadata, runtime={"python": sys.version, "torch": torch.__version__,
                "cuda": torch.version.cuda, "numpy": np.__version__, "triton": triton.__version__,
                "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                "affinity": sorted(os.sched_getaffinity(0)), "context": engine.context_window,
                "verify_rows": engine.max_rows, "drafter_bits": engine.head.bits}, capacity_plan=engine.capacity_plan)
    speculative, sampling = modules[proposal.SOURCES[0]], modules[proposal.SOURCES[1]]
    chain_type, partial = speculative.Chains, sampling._canonical_repair_order
    destination = outdir / "report.json"
    rows = [r for r in report["records"] if r["configuration"] == cfg_index]
    try:
        with proposal.policy(speculative, sampling, plan["policy"], chain_type, partial):
            for row in rows:
                if row["phase"] == "warm" and "proof_complete" not in info:
                    _compare([r for r in rows if r["phase"] in ("cold", "proof")], plan["serial_ids"])
                    info["proof_complete"] = True
                try:
                    proposal.HotGate.wait()
                    proposal.Progress.update("load", configuration=cfg_index, preparation="exact-head graph binding")
                    row["head_preparation"] = head_bench._bind_head(engine, heads[row["head"]])
                    # A changed bank has an explicit cold proof or untimed warm
                    # request; no first-use graph capture enters a timed row.
                    if row["head_preparation"]["changed"] and row["phase"] not in ("cold", "warm"):
                        raise RuntimeError("matrix changed heads without an explicit cold/warm preparation position")
                    request = prepared[row["workload"], row["mode"]]
                    if row["phase"] in ("cold", "proof"):
                        captured = {}
                        try:
                            with _inputs(engine) as (captured, state), proposal.diagnostics(
                                    speculative, sampling, chain_type, report["repair_assets"], outdir) as audit:
                                _attempt(engine, request, row, report, destination, audit=audit)
                            _require(state["graph_calls"] > 0, "proof did not execute graph-returned target head inputs")
                            row["head_bits"] = _bits(engine, heads, incumbent, captured, cfg["bits"],
                                                     draft=row["mode"].endswith("-draft"))
                        finally:
                            captured.clear()
                    else:
                        _attempt(engine, request, row, report, destination)
                except BaseException as error:  # preparation/proof failure is an actual failed position
                    if row["status"] != "ERROR":
                        row.update(status="ERROR", errors=["".join(traceback.format_exception(error))])
                    raise
        _compare(rows, plan["serial_ids"])
        info["status"] = "COMPLETE"
    finally:
        heads.clear()


def _sources(modules):
    result = {name: proposal._sha(Path(module.__file__).read_bytes()) for name, module in modules.items()}
    result.update(tool=proposal._sha(Path(__file__).read_bytes()),
                  proposal_helper=proposal._sha(Path(proposal.__file__).read_bytes()),
                  head_helper=proposal._sha(Path(head_bench.__file__).read_bytes()))
    native = Path(modules["tensorfold.families.qwen3_5.cuda.b16"].__file__)
    result.update({name: proposal._sha(native.with_name(name).read_bytes()) for name in ("b16.cu", "b16.cpp")})
    return result


def main(argv=None):
    from tensorfold.cleanup import raise_failures

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan, items, raw, prompts, consumed = load_plan(args.plan)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "plan.json").write_bytes(raw)
    (args.out / "prompts.json").write_bytes(prompts)
    report = {"schema": 1, "status": "RUNNING", "plan_sha256": proposal._sha(raw), "prompts_sha256": proposal._sha(prompts),
              "artifact_id": plan["artifact_id"], "consumed_inputs": consumed, "records": _matrix(plan),
              "configurations": [{**cfg, "status": "NOT_RUN"} for cfg in plan["configurations"]], "repair_assets": [],
              "scope": "64-token seed7 production-template launch screen only; no constructor-cluster inference or selection",
              "objective": "per-region warm settled native-generate wall time; HTTP/tokenization and startup separate",
              "timing": "no proof/head/q observers in timed positions; finite selected policy factory and stats retained",
              "allocator_scope": "Torch allocator counters outside the timed interval; not unified physical-memory peaks",
              "reserved_guard": 32}
    destination = args.out / "report.json"
    proposal.write_json(destination, report)
    primary, errors = None, []
    try:
        try:
            modules = {name: importlib.import_module(name) for name in SOURCES}
            report["source_sha256"] = _sources(modules)
            _require(report["source_sha256"] == plan["source_sha256"], "actual installed/tool source differs from sealed plan")
            proposal.HotGate.configure(None)
            speculative = modules[proposal.SOURCES[0]]
            with proposal._bindings([(speculative, "_PROPOSAL_GUARD", 32)]):
                for index, cfg in enumerate(plan["configurations"]):
                    info = report["configurations"][index]
                    info.update(status="LOADING", host_before=proposal.host_snapshot())
                    proposal.write_json(destination, report)
                    proposal.HotGate.wait()
                    proposal.Progress.update("load", configuration=index)
                    began = time.perf_counter()
                    def work(app, index=index, info=info, began=began):
                        info["construction_s"] = time.perf_counter() - began
                        qualify(app, index, plan, items, report, args.out, modules)
                    serve = [*plan["serve_args"], "--draft-rows", str(cfg["rows"]), "--drafter-bits", str(cfg["bits"])]
                    proposal.with_app(serve, plan["port"], work)
                    info.update(status="RETIRED", owner_s=time.perf_counter() - began, host_after=proposal.host_snapshot())
                    proposal.write_json(destination, report)
                    proposal.Progress.update("idle", configuration=index)
            _compare(report["records"], plan["serial_ids"])
            report["status"] = "PASS"
        except BaseException as error:  # noqa: BLE001 - retain failed/partial/not-run records and exact native primary
            primary = error
            report.update(status="ERROR", error="".join(traceback.format_exception(error)))
        try:
            proposal.write_json(destination, report)
        except BaseException as error:  # noqa: BLE001 - evidence publication is part of the owning result
            errors.append(error)
        raise_failures(primary, errors)
    finally:
        primary = errors = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
