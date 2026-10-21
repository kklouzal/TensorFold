"""Explicit canonical-Reader ownership, bounded run reads and consumption."""

import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.direct_read import Reader, SafeTensors


def files(tmp_path, *, begin=0, direct=False):
    path = tmp_path / "raw"
    path.write_bytes(b"x" * begin + bytes(range(32)))
    owner = SafeTensors.__new__(SafeTensors)
    owner.reader = Reader()
    owner.reader.direct = direct and owner.reader.direct
    owner.where = {f"field.{i}": (path, begin + 8 * i, 8, "U8", [2, 4]) for i in range(4)}
    return owner, path


@pytest.mark.parametrize("direct", [False, True])
def test_fresh_run_backings_do_not_alias_files_other_reads_or_mutable_reader_state(tmp_path, direct):
    owner, path = files(tmp_path, direct=direct)
    try:
        plan = owner.prepare_cpu_run(tuple(owner.where), dtype="U8", field_bytes=8, max_read_bytes=24)
        assert plan is not None and not owner.reader.staging
        a = plan.read(0, 2)
        b = plan.read(0, 2)
        assert a.device.type == b.device.type == "cpu" and a.dtype == b.dtype == torch.uint8
        assert bytes(a.tolist()) == bytes(b.tolist()) == bytes(range(16))
        assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
        a.fill_(255)
        assert bytes(b.tolist()) == path.read_bytes()[:16] == bytes(range(16))
        owner.close()
        assert bytes(b.tolist()) == bytes(range(16))  # backing is independent of reader lifetime
    finally:
        owner.close()


@pytest.mark.parametrize("fault", ["gap", "shard", "unaligned", "custom_reader"])
def test_valid_unsupported_geometry_or_reader_is_explicit_ineligibility_without_io(tmp_path, fault):
    owner, _path = files(tmp_path, begin=int(fault == "unaligned"))
    if fault == "gap":
        owner.where["field.2"] = (owner.where["field.2"][0], 17, 8, "U8", [2, 4])
    elif fault == "shard":
        owner.where["field.2"] = (Path("not-opened"), 16, 8, "U8", [2, 4])
    elif fault == "custom_reader":
        owner.reader = SimpleNamespace(read=lambda *_a, **_k: pytest.fail("read unsupported reader"))
    assert owner.prepare_cpu_run(tuple(owner.where), dtype="U8", field_bytes=8, max_read_bytes=32) is None


@pytest.mark.parametrize("fault", ["unknown", "duplicate", "dtype", "size", "offset", "boolean", "bound", "empty"])
def test_malformed_owned_plan_never_becomes_a_topology_miss(tmp_path, fault):
    owner, _ = files(tmp_path)
    names, budget = list(owner.where), 32
    if fault == "unknown": names.append("absent")
    elif fault == "duplicate": names.append(names[0])
    elif fault == "dtype": owner.where[names[-1]] = (Path("not-opened"), 24, 8, "F32", [2])
    elif fault == "size": owner.where[names[-1]] = (Path("not-opened"), 24, 9, "U8", [9])
    elif fault == "offset": owner.where[names[-1]] = (Path("not-opened"), -1, 8, "U8", [8])
    elif fault == "boolean": owner.where[names[-1]] = (Path("not-opened"), True, 8, "U8", [8])
    elif fault == "bound": budget = 7
    else: names.clear()
    with pytest.raises((ValueError, KeyError)):
        owner.prepare_cpu_run(names, dtype="U8", field_bytes=8, max_read_bytes=budget)


@pytest.mark.parametrize("first,stop", [(-1, 1), (0, 0), (0, 5), (True, 1), (0, 4), (0, 1.0)])
def test_run_checks_complete_interval_and_byte_cap_before_opening_file(tmp_path, first, stop):
    owner, path = files(tmp_path)
    plan = owner.prepare_cpu_run(tuple(owner.where), dtype="U8", field_bytes=8, max_read_bytes=24)
    path.unlink()
    with pytest.raises(ValueError, match="interval|byte bound"):
        plan.read(first, stop)


@pytest.mark.parametrize("direct", [False, True])
def test_fresh_truncation_is_fatal_and_not_a_planning_miss(tmp_path, direct):
    owner, path = files(tmp_path, direct=direct)
    plan = owner.prepare_cpu_run(tuple(owner.where), dtype="U8", field_bytes=8, max_read_bytes=32)
    path.write_bytes(struct.pack("<Q", 1))
    try:
        with pytest.raises((ValueError, OSError), match="range|exceeds|short|file"):
            plan.read(0, 4)
    finally:
        owner.close()
