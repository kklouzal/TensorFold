#!/usr/bin/env python3
"""Hash the actual local target/MTP loader inputs for an explicit native-owner trial.

This is CPU-only preparation, with no download, GPU initialization or invented
qualification evidence. Keep the selected files immutable through model startup.
The Engine independently verifies their complete contents before serving.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from tensorfold import hub
from tensorfold.engine.model_identity import canonical_runtime_json
from tensorfold.families.qwen4_exp.cuda import execution_identity
from tensorfold.families.qwen4_exp.cuda.owner_execution import FULL_COLLECTIVE, GRAPH_FAMILY, KERNEL_FAMILY, UNCAPTURED
from tensorfold.families.qwen4_exp.cuda.serving_profiles import artifact_identity, stamp
from tensorfold.serve_options import drafter_metadata_source

if __package__:
    from .benchmark_output import write_json
else:
    from benchmark_output import write_json


def create(model, mtp, output, *, graph_budget_bytes=0):
    if type(graph_budget_bytes) is not int or graph_budget_bytes < 0:
        raise ValueError("native graph budget requires a nonnegative integer byte allowance")
    output = Path(output).absolute()
    # These are the same prepared directories selected by the finite worker.
    model = hub.resolve(str(model), download=False)
    mtp = drafter_metadata_source(str(mtp))
    selected = SimpleNamespace(model_dir=model, mtp_model_dir=mtp,
                               method_geometry=SimpleNamespace(method="mtp"))
    files, _ = execution_identity.artifact_files(selected)
    if output.resolve() in {path.resolve() for path in files.values()}:
        raise ValueError("trial manifest cannot replace a selected loader input")
    artifacts, identities = {}, {}
    for name, path in sorted(files.items()):
        identity = artifact_identity(path)
        artifacts[name] = {key: identity[key] for key in ("size", "sha256")}
        identities[path] = tuple(identity["stat"])
    document = {"schema": execution_identity.SCHEMA, "version": execution_identity.VERSION,
                "artifacts": artifacts, "preparation": {
                    "kernel_family": KERNEL_FAMILY, "graph_family": GRAPH_FAMILY if graph_budget_bytes else UNCAPTURED,
                    "collective_family": FULL_COLLECTIVE, "graph_budget_bytes": graph_budget_bytes, "trial": True}}
    execution_identity._manifest_schema(document)
    canonical_runtime_json(document, max_bytes=4 << 20)
    if any(stamp(path) != previous for path, previous in identities.items()):
        raise ValueError("selected loader input changed during trial manifest preparation")
    write_json(output, document)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="existing target first shard or prepared directory")
    parser.add_argument("--mtp", type=Path, required=True, help="existing selected stock MTP GGUF")
    parser.add_argument("--output", type=Path, required=True, help="new administrator-owned trial manifest JSON")
    parser.add_argument("--native-graph-budget-mib", type=int, default=0,
                        help="explicit aggregate per-rank owner attention graph cap in MiB; zero retains eager trial")
    args = parser.parse_args(argv)
    create(args.model, args.mtp, args.output, graph_budget_bytes=args.native_graph_budget_mib * (1 << 20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
