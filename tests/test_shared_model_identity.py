"""Real local opened-file ownership with one bounded cohort content pass."""

import copy
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from tensorfold.engine.model_identity import capture_shared_model_identity


class Exchange:
    """A test-only owned phase transport; JSON never carries local authority."""

    def __init__(self, world, alter=None):
        self.barrier = threading.Barrier(world, timeout=10)
        self.values = {}
        self.alter = alter

    def phase(self, rank, stage, operation):
        try:
            value = operation()
            if self.alter is not None:
                value = self.alter(rank, stage, value)
            outcome = (True, json.loads(json.dumps(value)))
        except BaseException as error:
            outcome = (False, error)
        self.values[stage, rank] = outcome
        self.barrier.wait()
        rows = [self.values[stage, index] for index in range(self.barrier.parties)]
        self.barrier.wait()
        for ok, result in rows:
            if not ok:
                raise result
        return [result for _ok, result in rows]


def run_capture(tmp_path, *, world=8, alter=None, reuse=None):
    files = {}
    for name, data in {"target/source": b"actual complete target bytes", "draft/source": b"actual draft bytes"}.items():
        path = tmp_path / name.replace('/', '-')
        if not path.exists():
            path.write_bytes(data)
        files[name] = path
    exchange = Exchange(world, alter)

    def rank_capture(rank):
        return capture_shared_model_identity(files, runtime_identity={"runtime": "same-exact-build"},
            authorize=lambda name, path: path.resolve(strict=True), max_file_bytes=100,
            max_total_bytes=200, rank=rank, world=world,
            exchange=lambda stage, op: exchange.phase(rank, stage, op),
            reuse=None if reuse is None else reuse[rank], reuse_is_immutable=reuse is not None)

    with ThreadPoolExecutor(max_workers=world) as pool:
        tasks = [pool.submit(rank_capture, rank) for rank in range(world)]
        outcomes = []
        for task in tasks:
            try:
                outcomes.append(task.result(timeout=15))
            except BaseException as error:
                outcomes.append(error)
    return outcomes, files


def test_eight_local_issued_authorities_share_one_hash_and_rebind_without_payload_reads(tmp_path):
    receipts, files = run_capture(tmp_path)
    assert all(not isinstance(value, BaseException) for value in receipts), receipts
    assert len({receipt.model_prefix for receipt in receipts}) == 1
    assert sum(receipt.bytes_read for receipt in receipts) == sum(path.stat().st_size for path in files.values())
    assert receipts[0].hashes_reused == 0
    assert all(receipt.bytes_read == 0 and receipt.hashes_reused == 2 for receipt in receipts[1:])
    for receipt in receipts:
        receipt.verify_unchanged()
        with pytest.raises(ValueError, match="not issued"):
            copy.copy(receipt).verify_unchanged()
    rebound, _files = run_capture(tmp_path, reuse=receipts)
    assert all(not isinstance(value, BaseException) for value in rebound), rebound
    assert all(receipt.bytes_read == 0 and receipt.hashes_reused == 2 for receipt in rebound)


def test_one_copied_reuse_authority_is_an_agreed_refusal(tmp_path):
    receipts, _files = run_capture(tmp_path)
    receipts[3] = copy.copy(receipts[3])
    outcomes, _files = run_capture(tmp_path, reuse=receipts)
    assert all(isinstance(value, ValueError) for value in outcomes), outcomes


@pytest.mark.parametrize("mutation", ["host", "runtime", "closure", "stat", "post-hash-change"])
def test_cohort_differences_or_input_change_refuse_all_ranks(tmp_path, mutation):
    def alter(rank, stage, value):
        if rank == 1 and stage == "inventory" and mutation != "post-hash-change":
            if mutation == "closure":
                value["files"].pop("draft/source")
            elif mutation == "stat":
                value["files"]["draft/source"]["stat"][1] += 1
            elif mutation == "host":
                value["host"] = ["different", "actual-host"]
            else:
                value["runtime"] = "different exact build"
        if rank == 0 and stage == "content" and mutation == "post-hash-change":
            (tmp_path / "target-source").write_bytes(b"replaced after actual content read")
        return value

    outcomes, _files = run_capture(tmp_path, alter=alter)
    assert all(isinstance(value, ValueError) for value in outcomes), outcomes
