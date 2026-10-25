#!/usr/bin/env python3
"""Prepare a local Flash Next serving manifest; never downloads model weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from tensorfold import hub
from tensorfold.cleanup import raise_failures
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.serving_profiles import (
    PATH_OPTIONS, SCHEMA, _path, _required_files, artifact_identity, prepare, read_manifest, stamp,
)


def _close_preserving(primary, operation):
    prior = None if primary is None else (BaseException.__cause__.__get__(primary),
        BaseException.__context__.__get__(primary), BaseException.__suppress_context__.__get__(primary))
    try:
        operation()
    except BaseException as cleanup:
        if primary is not None:
            BaseException.__cause__.__set__(primary, prior[0])
            BaseException.__context__.__set__(primary, prior[1])
            BaseException.__suppress_context__.__set__(primary, prior[2])
            raise_failures(primary, [cleanup])
        raise
    else:
        if primary is not None:
            BaseException.__cause__.__set__(primary, prior[0])
            BaseException.__context__.__set__(primary, prior[1])
            BaseException.__suppress_context__.__set__(primary, prior[2])


def create(destination, *, identifier, model, options, api):
    destination = Path(destination).absolute()
    root = destination.parent.resolve(strict=True)
    model_dir = hub.resolve(str(_path(root, str(model))), download=False)
    normalized = dict(options)
    for name in PATH_OPTIONS & normalized.keys():
        if normalized[name] is not None:
            normalized[name] = _path(root, normalized[name])
    artifacts = []
    hashed_stats = {}
    for filename in sorted(_required_files(model_dir, normalized)):
        identity = artifact_identity(filename)
        if stamp(filename) != tuple(identity["stat"]):
            raise ValueError("selected artifact changed during profile preparation")
        hashed_stats[filename] = tuple(identity["stat"])
        artifacts.append({key: identity[key] for key in ("path", "size", "sha256")})
    body = {"schema": SCHEMA, "version": 1, "id": identifier, "model_dir": str(model_dir),
            "options": options, "api": api, "artifacts": artifacts}
    temporary = None
    primary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=root, prefix=".serving-", suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(body, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Content was hashed above. This real preflight checks the generated
        # schema/geometry/API/closure and records immutable file identities.
        profile = prepare(root, temporary.name, initializer=FlashNextEngine._initialize,
                          fixed_options={}, verify_content=False)
        if any(stamp(path) != value for path, value in hashed_stats.items()):
            raise ValueError("selected artifact changed before manifest publication")
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
        temporary = None
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        sync_error = None
        try:
            os.fsync(directory)
        except BaseException as error:
            sync_error = error
            raise
        finally:
            _close_preserving(sync_error, lambda: os.close(directory))
        return {**profile.record(), "manifest": destination.name}
    except BaseException as error:
        primary = error
        raise
    finally:
        if temporary is not None:
            _close_preserving(primary, lambda: temporary.unlink(missing_ok=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--options", type=Path, required=True, help="bounded JSON object of supported engine options")
    parser.add_argument("--api", type=Path, help="optional bounded JSON object of API binding defaults")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    options, _ = read_manifest(args.options)
    api = {} if args.api is None else read_manifest(args.api)[0]
    print(json.dumps(create(args.out, identifier=args.id, model=args.model, options=options, api=api), sort_keys=True))


if __name__ == "__main__":
    main()
