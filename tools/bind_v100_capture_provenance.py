#!/usr/bin/env python3
"""Bind a post-build packaging recapture to unchanged actual native artifacts.

No build fact is rewritten. The original receipt is copied byte-for-byte into
its own provenance directory; the new companion records both identities and the
immutable original phase evidence. Run before freezing the SDK artifact context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy/v100"))
from source_foundation import file_identity, read_json  # noqa: E402


def bind(original_capture, current_capture, native_phase_directory, output):
    original, original_raw = read_json(original_capture)
    current, _ = read_json(current_capture)
    for receipt in (original, current):
        if receipt.get("schema") != "tensorfold.v100-source-build" or type(receipt.get("version")) is not int or receipt["version"] != 1:
            raise ValueError("capture provenance requires completed source-build receipts")
    native_original = {key: value for key, value in original.items() if key != "producer_inputs"}
    native_current = {key: value for key, value in current.items() if key != "producer_inputs"}
    missing = object()
    changed = sorted(key for key in native_original.keys() | native_current.keys()
                     if native_original.get(key, missing) != native_current.get(key, missing))
    if native_original != native_current:
        raise ValueError("native build identities changed across packaging recapture: " + ", ".join(changed))
    phase = native_phase_directory.resolve(strict=True)
    if not phase.is_dir():
        raise ValueError("original native phase evidence must be a directory")
    evidence = {}
    for path in sorted(phase.rglob("*")):
        if path.is_symlink():
            raise ValueError("native phase evidence cannot introduce aliases")
        if path.is_file():
            evidence[str(path.relative_to(phase))] = file_identity(path)
    if not evidence:
        raise ValueError("original native phase evidence cannot be empty")
    preserved = output.parent / "original-capture-v8" / "source-build.json"
    preserved.parent.mkdir(parents=True, exist_ok=True)
    if preserved.exists() and preserved.read_bytes() != original_raw:
        raise ValueError("existing original capture provenance has different bytes")
    if not preserved.exists():
        with preserved.open("xb") as stream:
            stream.write(original_raw)
    result = {"schema": "tensorfold.v100-capture-provenance", "version": 1,
        "scope": "post-build capture/packaging revision; no native rebuild or relabeled compile inputs",
        "unchanged_native_fields": sorted(set(original) - {"producer_inputs"}),
        "original_capture": {"path": str(preserved.relative_to(output.parent)), **file_identity(preserved)},
        "current_capture": {"path": str(current_capture.relative_to(output.parent)), **file_identity(current_capture)},
        "native_phase": {"path": str(phase.relative_to(output.parent.resolve())), "files": evidence},
        "binder": {"path": "tools/bind_v100_capture_provenance.py", **file_identity(Path(__file__))}}
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if output.exists() and output.read_bytes() != raw:
        raise ValueError("existing capture provenance cannot be replaced with different facts")
    if not output.exists():
        with output.open("xb") as stream:
            stream.write(raw)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-capture", type=Path, required=True)
    parser.add_argument("--current-capture", type=Path, required=True)
    parser.add_argument("--native-phase-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bind(args.original_capture.resolve(strict=True), args.current_capture.resolve(strict=True),
         args.native_phase_directory, args.output.resolve())


if __name__ == "__main__":
    main()
