"""Multi-process small all-gather vs NCCL on SM70 GPUs with an NVLink cohort.

Each rank is a separate process with one visible GPU (``small_gather_worker.py``),
as in tensor-parallel serving. Checks bitwise equality with NCCL and with a
deterministic oracle (eager, in place, offset storage, capacity edge, NCCL above
capacity) and varying-data CUDA-graph replays; the row all-reduce against the
gathered partials summed in rank order and rounded to FP16 (eager and replayed
between gathers), the bulk row all-reduce (direct calls at 5-160 rows and 64 rows inside the graph)
against the same oracle, once from a fresh generation
sequence and once seeded just before the 32-bit generation wrap.
``TF_TEST_SMALL_GATHER_WORLD`` selects the rank count (default: all visible
GPUs, at most 8); the visible GPUs must form a cohort with NVLink routes.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensorfold.cuda import small_gather

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("small_gather_worker.py")


def _devices():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [value for value in visible.split(",") if value] if visible else [
        str(index) for index in range(torch.cuda.device_count())]
    world = int(os.environ.get("TF_TEST_SMALL_GATHER_WORLD", min(8, len(devices))))
    return devices[:world]


def _port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (7, 0) or len(_devices()) < 2,
    reason="needs at least two SM70 GPUs")


@pytest.mark.parametrize("seed", [-1, 0xFFFFFFFF - 3], ids=["fresh", "near-wrap"])
def test_small_gather_matches_nccl_bitwise_eager_and_in_graph(seed):
    devices = _devices()
    world, port = len(devices), _port()
    bulk = 16 * world * (160 * 2560 * 4 // (16 * world))  # FP32 partials of up to 160 x 2560 rows per rank
    environment = {**os.environ, small_gather.ENV: str(small_gather.MAX_CAPACITY), small_gather.BULK_ENV: str(bulk),
                   "PYTHONUNBUFFERED": "1",
                   "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    workers = [subprocess.Popen([sys.executable, str(WORKER), str(rank), str(world), str(port), str(seed), "200"],
                                env={**environment, "CUDA_VISIBLE_DEVICES": devices[rank]},
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
               for rank in range(world)]
    reports, failures = [], []
    for rank, worker in enumerate(workers):
        try:
            out, err = worker.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            for other in workers:
                other.kill()
            pytest.fail(f"rank {rank} timed out")
        if worker.returncode:
            failures.append(f"rank {rank} exited {worker.returncode}: {err[-4000:]}")
            continue
        reports.append(json.loads(out.strip().splitlines()[-1]))
    assert not failures, failures
    enabled = {report["enabled"] for report in reports}
    assert len(enabled) == 1, "ranks disagreed on selecting the small all-gather"
    if enabled == {False}:
        pytest.skip("the visible GPUs lack the NVLink routes the small all-gather requires")
    assert [report["errors"] for report in reports] == [[]] * world
    assert all(report["capacity"] == small_gather.MAX_CAPACITY and report["eager_cases"] > 0
               and report["graph_replays"] == 200 and report["reduced_shapes"] >= 5 for report in reports)
    if 2560 % (8 * world) == 0:
        assert all(report["bulk_shapes"] == 7 for report in reports)
