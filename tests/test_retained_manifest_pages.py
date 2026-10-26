# ruff: noqa: E402 - optional torch capability must be checked before CUDA-family imports.
"""Owned metadata paging crosses the old packet cap on a real CPU Store.

Wire-shape fixture only: these descriptors are not a model checkpoint or a
qualification record. Actual State/tensor restoration is covered separately.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
from torch.distributed import TCPStore

from tensorfold.engine.model_identity import canonical_runtime_json
from tensorfold.families.qwen4_exp.cuda.execution_identity import _retire_store
from tensorfold.families.qwen4_exp.cuda.retained_catalog import RetainedCatalog
from tensorfold.families.qwen4_exp.cuda.retained_parts import PartSpec, manifest_bound


def test_descriptor_table_larger_than_one_packet_uses_bounded_retired_fragments():
    count = 25000
    specs = [asdict(PartSpec(f"wire.{index}", "state", "float16", (1,), 2, ("a" * 64,) * 7))
             for index in range(count)]
    body = {"metadata": {"schema": "wire-fixture-only"}, "specs": specs,
            "keys": ["b" * 64] * count,
            "objects": [{"identity": "c" * 64, "checksum": "d" * 64, "bytes": 2}] * count}
    document = json.dumps(body, separators=(",", ":"))
    assert len(document) > 16 << 20
    with pytest.raises(ValueError, match="byte"):
        canonical_runtime_json({"ok": True, "value": {"document": document}}, max_bytes=16 << 20)
    del document
    server = TCPStore("127.0.0.1", 0, None, True, timedelta(seconds=10), wait_for_workers=False)
    baseline = server.num_keys()
    owners = []
    for rank in range(2):
        connection = TCPStore("127.0.0.1", server.port, None, False, timedelta(seconds=10)).clone()
        owner = object.__new__(RetainedCatalog)  # Codec-only actual methods; no workers/model allocation.
        owner.rank, owner._store = rank, connection
        owner._cohort = SimpleNamespace(rank=rank, tp=2, multi=None, comm=SimpleNamespace(store=connection))
        owner._manifest_bounds = (manifest_bound(count),) * 2
        owner.transfer_chunk_bytes, owner._native_value_limit = 128 << 10, 8 << 20
        owners.append(owner)

    def run(owner):
        result = RetainedCatalog._manifest(owner, "wire-test", body)
        _retire_store(owner._cohort, "wire-test", stages=("manifest-header",))
        return len(result), len(result[0]["specs"]), result[1]["specs"][-1]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, owners))
    expected = {**specs[-1], "shape": list(specs[-1]["shape"]), "semantic": list(specs[-1]["semantic"])}
    assert all(size == 2 and records == count and last == expected for size, records, last in results)
    assert server.num_keys() == baseline
