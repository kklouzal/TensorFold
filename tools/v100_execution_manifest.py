#!/usr/bin/env python3
"""Execution-profile manifests for the SC117 V100 owned-routed configuration.

``catalogue`` prints the exact-graph key catalogue of the selected configuration
(deterministic in lanes and depth); the tracked preparation packet
``deploy/v100/execution-profiles/sc117-tp8etp4-mtp4-p24.preparation.json`` holds
its output. ``manifest`` writes a complete ``--execution-profile-manifest``: the
artifact inventory the loader verifies (``execution_identity.artifact_files`` and
``artifact_record`` on the actual target GGUF, prepared metadata and MTP drafter)
plus that preparation. Run ``manifest`` with the serving image and the serving
``TF_GGUF_ASSETS_DIR`` so the prepared metadata is the file set the server reads.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def catalogue(lanes: int, depth: int) -> list[list]:
    """Verify and MTP keys (role, total rows, participants) of the selected catalogue.

    One participant: every row count 1..depth+1. p = 2..lanes: MTP proposal keys
    (p, p); full-width verify and MTP refresh (p * (depth + 1), p); intermediate
    verify (3p, p), (ceil(3.5p), p), (4p, p) and MTP refresh (3p, p), which batched
    frames pad to (docs/observed-graph-demand.md#intermediate-row-keys). Keys above
    the full width do not exist; sorted as the loader's policy parser sorts them.
    """
    if type(lanes) is not int or type(depth) is not int or lanes < 1 or depth < 1:
        raise ValueError("lanes and depth must be positive integers")
    width = depth + 1
    keys = {(role, rows, 1) for rows in range(1, width + 1) for role in ("verify", "mtp")}
    for p in range(2, lanes + 1):
        keys |= {("mtp", p, p), ("verify", width * p, p), ("mtp", width * p, p), ("mtp", 3 * p, p)}
        keys |= {("verify", rows, p) for rows in (3 * p, math.ceil(3.5 * p), 4 * p)}
    return [list(key) for key in sorted(key for key in keys if key[1] <= width * key[2])]


def manifest(target: Path, drafter: Path, preparation: dict) -> dict:
    from types import SimpleNamespace

    from tensorfold import hub
    from tensorfold.families.qwen4_exp.cuda.execution_identity import artifact_files, artifact_record
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import parse_preparation

    if parse_preparation(preparation) is None:
        raise ValueError("the preparation packet must select an explicit preparation object")
    prepared = hub.resolve(str(target), download=False)
    engine = SimpleNamespace(model_dir=prepared, method_geometry=SimpleNamespace(method="mtp"),
                             mtp_model_dir=Path(drafter), media_profile=None)
    files, _ = artifact_files(engine)
    artifacts = {name: artifact_record(name, path) for name, path in sorted(files.items())}
    return {"schema": "tensorfold.execution-profile-manifest", "version": 1, "artifacts": artifacts,
            "preparation": preparation}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    keys = sub.add_parser("catalogue", help="print the selected exact-graph key catalogue as JSON")
    keys.add_argument("--lanes", type=int, default=24)
    keys.add_argument("--depth", type=int, default=4)
    make = sub.add_parser("manifest", help="write a complete execution-profile manifest")
    make.add_argument("--target", type=Path, required=True, help="first target GGUF shard (or its directory)")
    make.add_argument("--drafter", type=Path, required=True, help="MTP drafter GGUF directory or file")
    make.add_argument("--preparation", type=Path, required=True, help="tracked preparation packet JSON")
    make.add_argument("--out", type=Path, required=True, help="new manifest file (never overwritten)")
    args = parser.parse_args(argv)
    if args.command == "catalogue":
        json.dump(catalogue(args.lanes, args.depth), sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")
        return 0
    preparation = json.loads(args.preparation.read_text(encoding="utf-8"))
    value = manifest(args.target, args.drafter, preparation)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=1, allow_nan=False)
        out.write("\n")
    print(json.dumps({"out": str(args.out), "artifacts": len(value["artifacts"]),
                      "exact_graph_keys": len(preparation.get("exact_graph_policy", {}).get("keys", []))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
