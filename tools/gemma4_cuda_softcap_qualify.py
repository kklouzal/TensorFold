"""Sealed, same-engine Gemma softcap holdout; --plan PLAN --out NEW_DIRECTORY.

Six seeds have predeclared alternating ABBA/BAAB orders in each of two regions.
Every position runs a complete 256-token q-byte proof/warm request, removes the
observer, then times its complete cached replay including App.prepare and native
settlement. Actual arm transitions retire the target graph bank and prepare a
fresh exact R16 wrapper. Prefix-cold proofs and graph preparation are separate
observations. Statistical selection, physical-memory monitoring, exclusive GPU
admission and the explicit operation deadline belong to the outer operator.
"""
from __future__ import annotations
import __future__

import argparse
import ast
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from . import gemma4_cuda_launch_qualify as launch
    from . import model_phase_bench as phase
else:
    import gemma4_cuda_launch_qualify as launch
    import model_phase_bench as phase


proposal, head_bench = launch.proposal, launch.head_bench
SOURCES = (*launch.SOURCES, "tensorfold.families.gemma4.cuda.glue")
SOURCE_KEYS = launch.SOURCE_KEYS | {"launch_helper", "phase_helper", SOURCES[-1]}
SEEDS = (13, 31, 47, 83, 101, 127)
ARMS = ("torch", "fused")
BASELINE_COMMIT = "3b8cee825b21b2c88197cc811efa2544858d7ad7"
BASELINE_FORWARD_SHA256 = "de88c303fb302e7b2ff6350a5f2ffa893623186783de9037e7e1eb1518196aad"
BASELINE_METHOD = """def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
    logits = self.weights.embed(hidden)
    return torch.tanh(logits / self.config.softcap) * self.config.softcap
"""
FIXED = {"--backend": "cuda", "--precision": "checkpoint", "--draft-rows": "5", "--drafter-bits": "16",
         "--parallel": "1", "--kv-dtype": "bf16", "--thinking-budget": "0"}


def load_plan(path):
    path = Path(path)
    raw = launch._read(path, 65536)
    plan = json.loads(raw)
    fields = {"schema", "kind", "artifact_id", "source_sha256", "assets", "prompts_file", "prompt_sha256",
              "serve_args", "port", "tokens", "blocks"}
    launch._require(type(plan) is dict and set(plan) == fields and type(plan["schema"]) is int
                    and plan["schema"] == 1 and plan["kind"] == "softcap-holdout", "exact softcap-holdout schema1 required")
    launch._require(type(plan["tokens"]) is int and plan["tokens"] == 256, "exact256-token requests required")
    a, b = list(ARMS), list(reversed(ARMS))
    orders = [a + b, b + a] * 3
    blocks = plan["blocks"]
    launch._require(type(blocks) is list and len(blocks) == 6 and all(type(block) is dict
                    and set(block) == {"seed", "order"} and type(block["seed"]) is int for block in blocks)
                    and tuple(block["seed"] for block in blocks) == SEEDS
                    and [block["order"] for block in blocks] in (orders, list(reversed(orders))),
                    "six seeded alternating ABBA/BAAB blocks required")
    identity, sources = plan["artifact_id"], plan["source_sha256"]
    launch._require(type(identity) is str and identity.startswith("sha256:") and launch._hash(identity[7:]),
                    "immutable image SHA256 required")
    launch._require(type(sources) is dict and set(sources) == SOURCE_KEYS and all(launch._hash(v) for v in sources.values()),
                    "exact installed and orchestration source hashes required")
    launch._require(type(plan["port"]) is int and 1 <= plan["port"] <= 65535, "explicit loopback port required")
    serve = plan["serve_args"]
    launch._require(type(serve) is list and len(serve) >= 2 and serve[0] == "serve"
                    and all(type(arg) is str for arg in serve), "production serve argument list required")
    forbidden = {"--api-key", "--api-key-file", "--no-thinking", "--no-drafts", "--tokenizer", "--vision-offload",
                 "--draft-depth", "--kv-key-dtype", "--kv-value-dtype"}
    bound = {*FIXED, "--context", "--drafter", "--chat-template"}
    launch._require(not any(arg.split("=", 1)[0] in forbidden or ("=" in arg and arg.split("=", 1)[0] in bound)
                           for arg in serve), "controlled options require their single explicit argument form")
    launch._require(all(launch._option(serve, key) == value for key, value in FIXED.items())
                    and launch._option(serve, "--context") in ("0", "262144")
                    and serve.count("--vision") == serve.count("--thinking") == 1,
                    "fixed native-context BF16 production region required")
    assets = plan["assets"]
    launch._require(type(assets) is dict and set(assets) == {
        "model_config", "drafter_config", "tokenizer", "template", "baseline_source"},
        "exact model/drafter/tokenizer/template/historical baseline bindings required")
    consumed, payloads = {}, {}
    for name, entry in {**assets, "prompts": plan["prompts_file"]}.items():
        launch._require(type(entry) is dict and set(entry) == {"path", "sha256"} and type(entry["path"]) is str
                        and launch._hash(entry["sha256"]), "input path and SHA256 required")
        target = Path(entry["path"])
        target = target if target.is_absolute() else path.parent / target
        data = launch._read(target, 64 << 20 if name == "tokenizer" else 1 << 20)
        launch._require(proposal._sha(data) == entry["sha256"], f"consumed {name} input hash changed")
        consumed[name] = {"path": str(target.resolve()), "sha256": entry["sha256"], "bytes": len(data)}
        payloads[name] = data
    expected = {"model_config": Path(serve[1]) / "config.json", "tokenizer": Path(serve[1]) / "tokenizer.json",
                "drafter_config": Path(launch._option(serve, "--drafter")) / "config.json",
                "template": Path(launch._option(serve, "--chat-template"))}
    launch._require(all(str(value.resolve()) == consumed[name]["path"] for name, value in expected.items())
                    and assets["template"]["sha256"] == launch.TEMPLATE_SHA256, "serve inputs or fixed template differ")
    launch._require(assets["baseline_source"]["sha256"] == BASELINE_FORWARD_SHA256,
                    "named historical forward source required")
    baseline = next(node for node in ast.parse(payloads["baseline_source"]).body
                    if isinstance(node, ast.ClassDef) and node.name == "Gemma4Target")
    method = next(node for node in baseline.body if isinstance(node, ast.FunctionDef) and node.name == "project_logits")
    launch._require(ast.dump(method) == ast.dump(ast.parse(BASELINE_METHOD).body[0]),
                    "historical baseline method AST differs from original inline expression")
    body = json.loads(payloads["prompts"])
    items = body.get("items") if type(body) is dict else None
    launch._require(type(items) is list and len(items) == 2 and all(type(item) is dict for item in items)
                    and {item.get("id") for item in items} == set(proposal.WORKLOADS), "exact code/history pair required")
    for item in items:
        launch._require(item.get("thinking") is False and type(item.get("messages")) is list and item["messages"]
                        and all(type(message) is dict and set(message) == {"role", "content"}
                                and type(message["content"]) is str for message in item["messages"]),
                        "plain text nonthinking messages required")
    launch._require(type(plan["prompt_sha256"]) is dict and set(plan["prompt_sha256"]) == set(proposal.WORKLOADS)
                    and all(launch._hash(v) for v in plan["prompt_sha256"].values()), "prepared prompt ID hashes required")
    return plan, items, raw, payloads["prompts"], consumed, method


def matrix(plan):
    rows = []
    current = None
    for workload in proposal.WORKLOADS:
        for block, entry in enumerate(plan["blocks"]):
            for position, arm in enumerate(entry["order"]):
                changed = arm != current
                for stage in ("proof", "timed"):
                    cold = stage == "proof" and changed
                    rows.append({"id": f"{workload}-b{block}-p{position}-{stage}", "case": len(rows), "workload": workload,
                                 "block": block, "seed": entry["seed"], "position": position, "arm": arm,
                                 "stage": stage, "phase": "cold" if cold else "warm", "status": "NOT_RUN",
                                 "rebind": changed and stage == "proof"})
                current = arm
    return rows


def sources(modules):
    actual = launch._sources(modules)
    actual.update(tool=proposal._sha(Path(__file__).read_bytes()),
                  launch_helper=proposal._sha(Path(launch.__file__).read_bytes()),
                  phase_helper=proposal._sha(Path(phase.__file__).read_bytes()))
    return actual


def bind(engine, arm, modules, reference):
    """A fresh exact wrapper forces the reviewed helper to retire stale graphs."""
    head = modules["tensorfold.families.gemma4.cuda.head"]
    forward = modules["tensorfold.families.gemma4.cuda.forward"]
    engine.drain()
    incumbent = engine.w.embed.incumbent
    method = reference if arm == "torch" else forward.Gemma4Target.project_logits
    engine.target.project_logits = method.__get__(engine.target, forward.Gemma4Target)
    engine.head.project_logits = engine.target.project_logits
    return head_bench._bind_head(engine, head.prepare_head(incumbent, reuse_rows=16))


def request(app, row, item):
    body = {"messages": item["messages"], "max_tokens": 256, "ignore_eos": True, "draft": True,
            "chat_template_kwargs": {"enable_thinking": False}, "temperature": 1., "top_k": 64,
            "top_p": .95, "min_p": 0., "seed": row["seed"]}

    def prepare(body, chat):
        prepared = app.prepare(body, chat)
        launch._require(not prepared.thinking and prepared.penalties is None and vars(prepared.sampling) == {
                            "seed": row["seed"], "temperature": 1., "top_k": 64, "top_p": .95, "min_p": 0.},
                        "prepared sampling differs from sealed region")
        return prepared
    count = proposal.WORKLOADS[row["workload"]]
    return SimpleNamespace(engine=app.engine, prepare=prepare), {
        "id": row["id"], "phase": row["phase"], "body": body, "prompt_tokens": count,
        "cached_tokens": 0 if row["phase"] == "cold" else count}


def compare(rows):
    launch._require(len(rows) == 96 and all(row["status"] == "COMPLETE" for row in rows), "incomplete softcap matrix")
    for row in rows:
        row["comparison"] = "FAILED"
        reference = next(other for other in rows if other["workload"] == row["workload"]
                         and other["seed"] == row["seed"] and other["arm"] == "torch" and other["stage"] == "proof")
        launch._require(row["token_ids"] == reference["token_ids"] and all(
            row["stats"][name] == reference["stats"][name] for name in proposal.COUNTS),
            "softcap changed complete IDs or proposal/acceptance counters")
        if row["stage"] == "proof":
            launch._require(all(row["proof"].get(name) == reference["proof"].get(name)
                                for name in ("q_packets", "q_rows", "q_sha256")), "softcap changed actual q bytes")
        row["comparison"] = "MATCH"


def qualify(app, plan, items, report, outdir, modules, baseline):
    import torch
    import triton

    from tensorfold.cleanup import raise_failures

    engine = app.engine
    head = modules["tensorfold.families.gemma4.cuda.head"]
    launch._require(isinstance(engine, modules[proposal.SOURCES[2]].Gemma4Engine)
                    and engine.w.numerics == "ct_nvfp4" and engine.max_rows == 5 and engine.context_window == 262144
                    and engine.head is not None and engine.head.bits == 16 and engine.vision is not None
                    and engine.graphs is not None and isinstance(engine.w.embed, head._RowsHead)
                    and engine.w.embed.reuse_rows == 16 and torch.cuda.get_device_capability() == (12, 1)
                    and engine.target._softcap_logits is modules[SOURCES[-1]].softcap30,
                    "constructed engine differs from original GB10 softcap region")
    speculative, sampling = modules[proposal.SOURCES[0]], modules[proposal.SOURCES[1]]
    launch._require(speculative._PROPOSAL_GUARD == 32, "shipping G32 proposal policy required")
    report.update(runtime={"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                          "triton": triton.__version__, "device": torch.cuda.get_device_name(),
                          "capability": torch.cuda.get_device_capability(), "affinity": sorted(os.sched_getaffinity(0)),
                          "context": engine.context_window, "verify_rows": 5, "drafter_bits": 16},
                  capacity_plan=engine.capacity_plan)
    # Compile only the pinned, AST-verified historical method, with its actual
    # Torch global and future-annotations mode. The rest of that file never runs.
    namespace = {"torch": torch}
    code = compile(ast.Module(body=[baseline], type_ignores=[]), "<historical Gemma project_logits>", "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, namespace)  # noqa: S102 - Only the SHA256-pinned, exact-AST historical method executes.
    reference = namespace["project_logits"]
    target, old_head = engine.target, engine.w.embed
    hooks = [(owner, "project_logits", "project_logits" in vars(owner), vars(owner).get("project_logits"))
             for owner in (target, engine.head)]
    primary, errors = None, []
    try:
        report["prepared_inputs"] = {}
        for item in items:
            first = next(row for row in report["records"] if row["workload"] == item["id"])
            proxy, case = request(app, first, item)
            prepared = proxy.prepare(case["body"], True)
            digest = proposal._sha(json.dumps(prepared.prompt, separators=(",", ":")).encode())
            launch._require(len(prepared.prompt) == proposal.WORKLOADS[item["id"]]
                            and digest == plan["prompt_sha256"][item["id"]], "prepared prompt differs from sealed region")
            report["prepared_inputs"][item["id"]] = {"tokens": len(prepared.prompt), "sha256": digest}
        for row in report["records"]:
            try:
                if row["rebind"]:
                    row["binding_hot_wait_s"] = phase.HotGate.wait()
                    row["binding"] = bind(engine, row["arm"], modules, reference)
                    launch._require(row["binding"]["changed"], "softcap transition retained stale target graphs")
                proxy, case = request(app, row, next(item for item in items if item["id"] == row["workload"]))
                if row["stage"] == "proof":
                    with proposal.diagnostics(speculative, sampling, speculative.Chains,
                                              report["repair_assets"], outdir) as audit:
                        row["proof"] = audit
                        phase._attempt(proxy, case, row, report, outdir / "report.json", torch)
                    launch._require(audit["q_packets"] > 0 and audit["q_rows"] > 0
                                    and launch._hash(audit.get("q_sha256")), "proof did not exercise actual q packets")
                else:
                    phase._attempt(proxy, case, row, report, outdir / "report.json", torch)
                # The shared attempt already hashes its actual prepared IDs
                # after timing. Reuse that evidence; do not hash in hot prepare.
                launch._require(row["prompt_sha256"] == plan["prompt_sha256"][row["workload"]],
                                "prepared prompt differs from sealed region")
                launch._require(all(type(row["stats"].get(key)) is int and row["stats"][key] >= 0
                                    for key in proposal.COUNTS) and row["stats"]["drafted"] > 0,
                                "request did not exercise valid native proposal counters")
            except BaseException as error:
                row.update(status="FAILED", qualification_error=phase._failure(error))
                raise
        compare(report["records"])
    except BaseException as error:  # noqa: BLE001 - preserve primary through owned native retirement
        primary = error
    try:
        engine.drain()
    except BaseException as error:  # noqa: BLE001 - failed drain retains owners for CLI retirement
        errors.append(error)
    else:
        try:
            engine.graphs = None
            engine.w.embed = old_head
            with engine._prefix_lock:
                engine._prefix = None
            for owner, name, owned, previous in hooks:
                if owned:
                    setattr(owner, name, previous)
                elif name in vars(owner):
                    delattr(owner, name)
        except BaseException as error:  # noqa: BLE001 - method restoration is part of the owning outcome
            errors.append(error)
    report["restoration"] = {"status": "FAILED" if errors else "COMPLETE", "errors": [phase._failure(e) for e in errors]}
    raise_failures(primary, errors)


def run(plan_path, destination):
    from tensorfold.cleanup import raise_failures

    plan, items, raw, prompts, consumed, baseline = load_plan(plan_path)
    outdir = Path(destination)
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "plan.json").write_bytes(raw)
    (outdir / "prompts.json").write_bytes(prompts)
    report = {"schema": 1, "status": "STARTING", "plan": plan, "plan_sha256": proposal._sha(raw),
              "consumed_inputs": consumed, "records": matrix(plan), "repair_assets": [],
              "objective": "per-region cached complete request wall from App.prepare through native settlement",
              "timing": "prepare validation and bounded progress retained equally; q observers proof-only; no HTTP",
              "conditioning": "every timed request follows its same-seed complete q-byte proof/warm request",
              "reference": {"commit": BASELINE_COMMIT, "forward_sha256": BASELINE_FORWARD_SHA256,
                            "method_ast_sha256": proposal._sha(ast.dump(baseline).encode())},
              "scope": "same-engine paired observations; no automatic statistical or global-optimum claim"}
    output = outdir / "report.json"
    phase.write_json(output, report)
    primary, errors, began = None, [], None
    try:
        modules = {name: importlib.import_module(name) for name in SOURCES}
        report["source_sha256"] = sources(modules)
        report["source_paths"] = {name: module.__file__ for name, module in modules.items()}
        launch._require(report["source_sha256"] == plan["source_sha256"], "installed or tool source differs from sealed plan")
        phase.HotGate.configure(None)
        phase.Progress.configure(str(outdir / "progress.json"))
        report["startup_hot_wait_s"] = phase.HotGate.wait()
        began = time.perf_counter()

        def work(app):
            report["startup_s"] = time.perf_counter() - began
            report["status"] = "RUNNING"
            qualify(app, plan, items, report, outdir, modules, baseline)
        phase.with_app(plan["serve_args"], plan["port"], work)
        report["status"] = "COMPLETE"
    except BaseException as error:  # noqa: BLE001 - retain startup, close, interruption and native failures
        primary = error
        report.update(status="FAILED", error=phase._failure(error))
    report["cli_lifetime_s"] = None if began is None else time.perf_counter() - began
    try:
        phase.write_json(output, report)
    except BaseException as error:  # noqa: BLE001 - publication cannot replace an earlier primary
        errors.append(error)
    raise_failures(primary, errors)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run(args.plan, args.out)


if __name__ == "__main__":
    main()
