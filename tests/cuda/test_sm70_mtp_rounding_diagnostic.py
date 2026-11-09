"""Explicit operator-selected diagnostic capture; never a quality/parity gate.

General CI without complete operator declarations skips this CUDA diagnostic.
The actual owned runner requires zero skips and verifies its full capture.
"""
import hashlib
import json
import math
import os
import re
from pathlib import Path
import stat

ENV = {"case": "TENSORFOLD_MTP_ROUNDING_CASE", "out": "TENSORFOLD_MTP_ROUNDING_OUT",
       "image_id": "TENSORFOLD_MTP_ROUNDING_IMAGE", "source_commit": "TENSORFOLD_MTP_ROUNDING_SOURCE"}


def operator_arguments(environment):
    present = {name: environment.get(key) for name, key in ENV.items()}
    if all(value is None for value in present.values()):
        return None
    if any(type(value) is not str or not value for value in present.values()):
        raise ValueError("complete explicit diagnostic operator declarations required")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", present["image_id"]) or not re.fullmatch(r"[0-9a-f]{40}", present["source_commit"]):
        raise ValueError("exact operator image/source identities required")
    if present["case"] not in ("reference", "prepared"):
        raise ValueError("unknown original fixture case")
    path = Path(present["out"])
    if not path.is_absolute() or not path.is_relative_to("/evidence") or path == Path("/evidence") or ".." in path.parts:
        raise ValueError("diagnostic output must be a fresh owned /evidence subdirectory")
    return ["--case", present["case"], "--out", present["out"], "--image-id", present["image_id"],
            "--source-commit", present["source_commit"]]


def verify_tensor_artifacts(report, directory):
    """Check NPY headers and raw identities streaming; never allocate header-sized arrays."""
    import numpy as np
    from gb10_io import owned_descriptor, borrowed_stream, stamp
    pending, records = [report], []
    while pending:
        value = pending.pop()
        if type(value) is dict:
            if {"shape", "dtype", "sha256", "file"} <= set(value):
                records.append(value)
            else:
                pending.extend(value.values())
        elif type(value) is list:
            pending.extend(value)
    if not records or len(records) > 2048:
        raise ValueError("missing/excess diagnostic tensor artifacts")
    names, total = set(), 0
    for record in records:
        name = record["file"]
        if type(name) is not str or Path(name).name != name or not name.endswith(".npy") or name in names:
            raise ValueError("invalid/duplicate diagnostic artifact path")
        names.add(name)
        with owned_descriptor(directory / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC) as fd:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > 64 << 20:
                raise ValueError("diagnostic artifact extent/type differs")
            with borrowed_stream(fd) as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError("unsupported diagnostic NPY schema")
                shape, fortran, dtype = reader(stream)
                if (fortran or dtype.hasobject or list(shape) != record["shape"] or dtype.str != record["dtype"]
                        or any(type(n) is not int or n < 0 for n in shape)):
                    raise ValueError("diagnostic tensor header differs")
                length = math.prod(shape) * dtype.itemsize
                total += length
                if total > 64 << 20 or stream.tell() + length != before.st_size:
                    raise ValueError("diagnostic tensor extent differs/exceeds bound")
                checksum, remaining = hashlib.sha256(), length
                while remaining:
                    data = stream.read(min(65536, remaining))
                    if not data:
                        raise ValueError("truncated diagnostic tensor")
                    remaining -= len(data)
                    checksum.update(data)
                if checksum.hexdigest() != record["sha256"] or stamp(os.fstat(fd)) != stamp(before):
                    raise ValueError("diagnostic tensor changed or SHA differs")
    return len(records)


def test_operator_selected_four_cell_capture(record_property):
    import pytest
    arguments = operator_arguments(os.environ)
    if arguments is None:
        pytest.skip("explicit operator diagnostic declarations absent; CUDA capture UNRUN")
    from gb10_io import opened_file
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("actual pinned SM70 diagnostic required; collection is not capture")
    # Import exact staged /operator tool, not an old /canonical/tools copy.
    import v100_mtp_rounding_diagnostic as diagnostic
    diagnostic.main(arguments)
    directory = Path(os.environ[ENV["out"]])
    with opened_file(directory / "diagnostic.json", "rb") as stream:
        raw = stream.read((8 << 20) + 1)
    if len(raw) > 8 << 20:
        raise ValueError("diagnostic report exceeds structured bound")
    report = json.loads(raw)
    status = diagnostic.validate_report(report, case=os.environ[ENV["case"]], image_id=os.environ[ENV["image_id"]],
                                        source_commit=os.environ[ENV["source_commit"]])
    count = verify_tensor_artifacts(report, directory)
    record_property("capture_status", status)
    record_property("verified_tensor_artifacts", count)
    record_property("original_numeric_gate", "OPEN_UNCHANGED_STRICT_FAILURES")
    record_property("original_prestate_match", "UNPROVEN_NO_ORIGINAL_CAUSALITY_CLAIM")
