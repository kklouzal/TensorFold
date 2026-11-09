"""Closed benchmark DATA validation; no model/runtime/capability activation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import v100_owned_model_contract as c


def file(path, *, rank=None):
    return {"path": path, "bytes": 10, "sha256": "a" * 64, **({} if rank is None else {"rank": rank})}


def artifact(path, *, rank=None):
    row = file(path, rank=rank)
    row["file"] = row.pop("path")
    return row


@pytest.fixture
def operation():
    first = "/models/hf/hub/model/snapshot/IQ3_S/model-00001-of-00002.gguf"
    frontend = "/frontend/" + "b" * 64
    return {
        "benchmark_plan": file("/operator/benchmark-plan.json"),
        "benchmark_client": file("/operator/benchmark-client.json"),
        "model_entrypoint": first,
        "model_assets": [file(first), file(first.replace("00001", "00002"))],
        "frontend_root": frontend,
        "frontend_assets": [file(frontend + "/" + name) for name in sorted(c.FRONTEND_NAMES)],
        "execution_manifest": None,
        "trace": {
            "prefill_start": 8192,
            "host_obligation_bytes": 8 << 30,
            "device_headroom_bytes": 512 << 20,
            "disk_bytes": 16 << 30,
        },
    }


@pytest.fixture
def client():
    return {
        "schema": c.BENCHMARK_CLIENT_SCHEMA,
        "benchmark_plan_sha256": "a" * 64,
        "base": "http://127.0.0.1:8000",
        "model": "SC117",
        "expected_service": dict(c.SERVICE),
        "readiness_timeout_s": 600,
        "whole_timeout_s": 3600,
        "socket_timeout_s": 300,
        "response_bytes": 16 << 20,
        "sse_line_bytes": 64 << 10,
    }


@pytest.fixture
def canonical_plan(tmp_path):
    # Canonical loader, not a second implementation of its construction/validator.
    from tools.bench_concurrent import load_matrix

    rows = []
    for ident, size, warmup in ((0, 64, True), (3, 16384, False)):
        ids = [ident + 1] + [7] * (size - 1)
        rows.append(
            {
                "id": ident,
                "level": 1,
                "input_tokens": size,
                "tokens": 2,
                "repetition": 0,
                "warmup": warmup,
                "requests": [
                    {
                        "prefix": [ident + 1],
                        "input_tokens": size,
                        "index": 0,
                        "prompt_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                    }
                ],
            }
        )
    value = {
        "schema": "tensorfold.phase-matrix-plan-v1",
        "run_identity": "c" * 32,
        "model": "SC117",
        "profile": "q8",
        "admin_cap": 24,
        "levels": [1],
        "context": 262144,
        "tokens": 2,
        "reps": 1,
        "inputs": [64, 16384],
        "tokenizer_sha256": "d" * 64,
        "filler_ids": [7],
        "input_scope": "synthetic DATA only",
        "cohorts": rows,
        "excluded_levels": [],
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(value))
    return load_matrix(path)


def test_actual_canonical_subset_and_explicit_grants_admit(operation, client, canonical_plan):
    assert c.validate_benchmark_operation(operation) is operation
    assert c.validate_benchmark_client(client) is client
    selection = c.validate_benchmark_operation_plan(operation, canonical_plan, client)
    assert selection == {
        "cohort_id": 3,
        "input_tokens": 16384,
        "prompt_sha256": canonical_plan["cohorts"][1]["requests"][0]["prompt_sha256"],
        "prefill_start": 8192,
        "request_count": 2,
    }
    assert c.benchmark_operation_assets(operation) == [
        *operation["model_assets"],
        *operation["frontend_assets"],
        operation["benchmark_plan"],
        operation["benchmark_client"],
    ]
    # N and grants are DATA, not the proposed first fixture's literal values.
    operation["trace"].update(prefill_start=128, host_obligation_bytes=1, device_headroom_bytes=2, disk_bytes=3)
    assert c.validate_benchmark_operation_plan(operation, canonical_plan, client)["prefill_start"] == 128


@pytest.mark.parametrize(
    "key,bad",
    [
        ("host_obligation_bytes", True),
        ("host_obligation_bytes", 0),
        ("device_headroom_bytes", -1),
        ("device_headroom_bytes", 1 << 63),
        ("disk_bytes", None),
        ("prefill_start", False),
        ("prefill_start", 262144),
    ],
)
def test_trace_grants_are_explicit_typed_and_bounded(operation, key, bad):
    operation["trace"][key] = bad
    with pytest.raises(ValueError):
        c.validate_benchmark_operation(operation)


@pytest.mark.parametrize("fault", ["extra", "missing", "teacher", "manifest", "asset", "plan-path", "plan-bytes"])
def test_closed_operation_rejects_unbound_or_foreign_roles(operation, fault):
    if fault == "extra":
        operation["argv"] = ["anything"]
    elif fault == "missing":
        del operation["trace"]["disk_bytes"]
    elif fault == "teacher":
        operation["teacher_plan"] = operation.pop("benchmark_plan")
    elif fault == "manifest":
        operation["execution_manifest"] = file("/operator/candidate.json")
    elif fault == "asset":
        operation["frontend_assets"][0]["path"] = "/outside/config.json"
    elif fault == "plan-path":
        operation["benchmark_plan"]["path"] = "/operator/other.json"
    else:
        operation["benchmark_plan"]["bytes"] = (8 << 20) + 1
    with pytest.raises(ValueError):
        c.validate_benchmark_operation(operation)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("response_bytes", True),
        ("sse_line_bytes", 0),
        ("sse_line_bytes", 1 << 30),
        ("whole_timeout_s", float("nan")),
        ("socket_timeout_s", float("inf")),
        ("readiness_timeout_s", 4000),
        ("base", "http://localhost:8000"),
        ("model", "bad\0model"),
    ],
)
def test_benchmark_client_boundary_refuses_invalid_fields(client, key, bad):
    client[key] = bad
    with pytest.raises(ValueError):
        c.validate_benchmark_client(client)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("response_bytes", 1),
        ("response_bytes", (1 << 20) - 1),
        ("response_bytes", (16 << 20) + 1),
        ("response_bytes", (sys.maxsize // (1 << 20) + 1) * (1 << 20)),
        ("sse_line_bytes", 1),
        ("sse_line_bytes", 1023),
        ("sse_line_bytes", (64 << 10) + 1),
    ],
)
def test_byte_grants_cannot_round_or_truncate_canonical_cli_units(client, key, bad):
    client[key] = bad
    with pytest.raises(ValueError):
        c.validate_benchmark_client(client)


@pytest.mark.parametrize("response,line", [(16 << 20, 64 << 10), (1 << 20, 1 << 10)])
def test_valid_byte_units_reach_real_protocol_limits_exactly(client, response, line):
    from types import SimpleNamespace
    from tools import openai_protocol

    client.update(response_bytes=response, sse_line_bytes=line)
    c.validate_benchmark_client(client)
    limits = openai_protocol.from_arguments(
        SimpleNamespace(response_mib=response // (1 << 20), sse_line_kib=line // (1 << 10))
    )
    assert type(limits.response_bytes) is type(limits.line_bytes) is int
    assert limits.response_bytes == response and limits.line_bytes == line


@pytest.mark.parametrize(
    "fault",
    [
        "wrong-plan",
        "q4",
        "cap",
        "context",
        "levels",
        "bool-level",
        "bool-reps",
        "reply",
        "warmup-order",
        "double-measured",
        "only-warmup",
        "input-metadata",
        "bool-index",
        "threshold",
    ],
)
def test_closed_additions_reject_other_regions_without_shadowing_loader(operation, client, canonical_plan, fault):
    plan = canonical_plan
    if fault == "wrong-plan":
        client["benchmark_plan_sha256"] = "b" * 64
    elif fault == "q4":
        plan["profile"] = "q4"
    elif fault == "cap":
        plan["admin_cap"] = 32
    elif fault == "context":
        plan["context"] = 262144.0
    elif fault == "levels":
        plan["levels"] = [1, 8]
    elif fault == "bool-level":
        plan["levels"] = [True]
    elif fault == "bool-reps":
        plan["reps"] = True
    elif fault == "reply":
        plan["tokens"] = 3
    elif fault == "warmup-order":
        plan["cohorts"].reverse()
    elif fault == "double-measured":
        plan["cohorts"][0]["warmup"] = False
    elif fault == "only-warmup":
        plan["cohorts"].pop()
    elif fault == "input-metadata":
        plan["inputs"] = [16384]
    elif fault == "bool-index":
        plan["cohorts"][0]["requests"][0]["index"] = False
    else:
        operation["trace"]["prefill_start"] = 16384
    with pytest.raises(ValueError):
        c.validate_benchmark_operation_plan(operation, plan, client)


def test_canonical_loader_still_owns_full_prompt_hash_before_contract(tmp_path):
    from tools.bench_concurrent import load_matrix

    value = {
        "schema": "tensorfold.phase-matrix-plan-v1",
        "profile": "q8",
        "admin_cap": 24,
        "context": 262144,
        "run_identity": "a" * 32,
        "filler_ids": [7],
        "cohorts": [
            {
                "id": 0,
                "level": 1,
                "tokens": 2,
                "input_tokens": 3,
                "requests": [{"prefix": [1], "input_tokens": 3, "prompt_sha256": "0" * 64}],
            }
        ],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_matrix(path)


def test_fixed_argv_has_no_generic_command_or_teacher_flags(operation, client):
    uuids = ["GPU-" + f"{i:08x}" + "-1234-1234-1234-123456789abc" for i in range(8)]
    argv = c.benchmark_qualifier_argv(operation, client, uuids, "/evidence/benchmark")
    assert argv[:6] == ["--gpus", ",".join(uuids), "--master-port", "29551", "--out", "/evidence/benchmark"]
    assert argv[6:14] == [
        "--benchmark-plan",
        "/operator/benchmark-plan.json",
        "--benchmark-client",
        "/operator/benchmark-client.json",
        "--scheduler-trace",
        "/evidence/benchmark/trace",
        "--scheduler-trace-prefill-start",
        "8192",
    ]
    assert argv[14] == "--"
    for flag, value in [
        ("--parallel", "24"),
        ("--kv-full-windows", "24"),
        ("--context", "262144"),
        ("--draft-method", "none"),
        ("--expert-tensor-parallel-size", "4"),
    ]:
        assert argv[argv.index(flag) + 1] == value
    assert "--no-graphs" in argv and "--teacher-plan" not in argv
    with pytest.raises(ValueError):
        c.benchmark_qualifier_argv(operation, client, uuids, "/other")
    uuids[0] = "GPU-" + "-" * 36
    with pytest.raises(ValueError):
        c.benchmark_qualifier_argv(operation, client, uuids, "/evidence/benchmark")


def make_result(operation):
    label = "c1-prefill-start-ge-" + str(operation["trace"]["prefill_start"]) + "-round"
    return {
        "schema": c.BENCHMARK_RESULT_SCHEMA,
        "benchmark_plan_sha256": "a" * 64,
        "benchmark_client_sha256": "a" * 64,
        "launch_exit": 0,
        "P3_valid": False,
        "client_report": artifact("benchmark/benchmark-client-report.json"),
        "matrix_report": artifact("benchmark/matrix-results.json"),
        "startup_receipts": [artifact(f"benchmark/rank-{i}.json", rank=i) for i in range(8)],
        "rank_receipts": [artifact(f"benchmark/benchmark-rank{i}/summary.json", rank=i) for i in range(8)],
        "child_native_records": [
            artifact(f"child-native/benchmark-closed-rank-{i}-pid-{100 + i}.json", rank=i) for i in range(8)
        ],
        "trace": {
            k: artifact("benchmark/trace/" + name)
            for k, name in [
                ("inventory", "inventory.json"),
                ("selection", "prefill-selection.json"),
                ("receipt", label + "-receipt.json"),
                ("chrome", label + ".json"),
            ]
        },
    }


def test_result_and_outer_identity_bind_exact_plan_source_threshold_and_evidence(operation):
    result = make_result(operation)
    assert c.validate_benchmark_model_result(result) is result
    outer = {
        "schema": c.BENCHMARK_HOST_RESULT_SCHEMA,
        "mode": c.BENCHMARK_MODE,
        "source_commit": "f" * 40,
        "image_id_declared": "sha256:" + "e" * 64,
        "operation": operation,
        "source_after_match": True,
        "content_binding_complete": True,
        "content_binding": artifact("model-content-binding.json"),
        "parent_cuda_uninitialized": True,
        "model_result": result,
        "primary_error": None,
        "post_errors": [],
        "scope": "diagnostic only",
    }
    expected = {"operation": operation, "source_commit": "f" * 40, "image_id": "sha256:" + "e" * 64}
    assert c.validate_benchmark_host_result(outer, expected) is outer
    for fault in ("source", "input", "threshold", "disk"):
        bad = deepcopy(outer)
        if fault == "source":
            bad["source_after_match"] = False
        elif fault == "input":
            bad["model_result"]["benchmark_plan_sha256"] = "d" * 64
        elif fault == "threshold":
            bad["model_result"]["trace"]["receipt"]["file"] = "benchmark/trace/c1-prefill-start-ge-1-round-receipt.json"
            bad["model_result"]["trace"]["chrome"]["file"] = "benchmark/trace/c1-prefill-start-ge-1-round.json"
        else:
            bad["operation"]["trace"]["disk_bytes"] = 1
            expected = {**expected, "operation": bad["operation"]}
        with pytest.raises(ValueError):
            c.validate_benchmark_host_result(bad, expected)


@pytest.mark.parametrize("fault", ["missing-rank", "wrong-rank", "wrong-tag", "P3", "trace-role", "extra"])
def test_result_cannot_hide_partial_or_foreign_evidence(operation, fault):
    value = make_result(operation)
    if fault == "missing-rank":
        value["rank_receipts"].pop()
    elif fault == "wrong-rank":
        value["startup_receipts"][1]["rank"] = 0
    elif fault == "wrong-tag":
        value["child_native_records"][0]["file"] = "child-native/teacher-closed-rank-0-pid-100.json"
    elif fault == "P3":
        value["P3_valid"] = True
    elif fault == "trace-role":
        value["trace"]["selection"] = artifact("elsewhere.json")
    else:
        value["argv"] = ["anything"]
    with pytest.raises(ValueError):
        c.validate_benchmark_model_result(value)


def test_teacher_entrypoints_stay_teacher_only_and_keep_response_ceiling(operation, client):
    with pytest.raises(ValueError):
        c.validate_operation(operation)
    teacher = {k: v for k, v in client.items() if k not in ("sse_line_bytes", "benchmark_plan_sha256")}
    teacher.update(schema="tensorfold.owned-teacher-client-v1", teacher_plan_sha256="a" * 64)
    with pytest.raises(ValueError):
        c.validate_teacher_client(teacher)
    teacher["response_bytes"] = 1 << 20
    assert c.validate_teacher_client(teacher) is teacher


def test_contract_import_is_inert_in_fresh_process():
    root = Path(__file__).resolve().parents[1]
    code = 'import sys;sys.path.insert(0,sys.argv[1]);from tools import v100_owned_model_contract;assert "torch" not in sys.modules'
    subprocess.run([sys.executable, "-I", "-c", code, str(root)], check=True, timeout=10)


def test_trace_transport_roundtrip_has_one_shared_typed_owner(operation):
    trace = json.loads(json.dumps(operation["trace"], sort_keys=True, separators=(",", ":")))
    assert c.validate_benchmark_trace(trace) == operation["trace"]
    trace["reserve_device_bytes"] = trace["device_headroom_bytes"]
    with pytest.raises(ValueError):
        c.validate_benchmark_trace(trace)


def test_benchmark_sources_extend_required_closure_without_changing_teacher_minimum():
    source = {name: {"bytes": 10, "sha256": "b" * 64} for name in c.REQUIRED_TOOLS}
    assert c.validate_sources(source) is source
    with pytest.raises(ValueError):
        c.validate_benchmark_sources(source)
    source.update(
        {
            name: {"bytes": 10, "sha256": "c" * 64}
            for name in (
                "tools/bench_concurrent.py",
                "tools/v100_adaptive_qualification.py",
                "tools/openai_protocol.py",
                "tools/worker_lifetime.py",
                "tools/benchmark_output.py",
            )
        }
    )
    assert c.validate_benchmark_sources(source) is source


def test_benchmark_host_preserves_same_fixed_model_and_namespace_guards(operation):
    uuids = ["GPU-" + f"{i:08x}" + "-1234-1234-1234-123456789abc" for i in range(8)]
    required = c.REQUIRED_TOOLS | {
        "tools/bench_concurrent.py",
        "tools/v100_adaptive_qualification.py",
        "tools/openai_protocol.py",
        "tools/worker_lifetime.py",
        "tools/benchmark_output.py",
    }
    inner = {
        "schema": c.BENCHMARK_SCHEMA,
        "mode": c.BENCHMARK_MODE,
        "operation": operation,
        "source_files": {name: {"bytes": 10, "sha256": "a" * 64} for name in required},
        "UUIDs": uuids,
        "SM": [7, 0],
        "image_id": "sha256:" + "a" * 64,
    }
    host = {
        "uuids": uuids,
        "image_id": inner["image_id"],
        "verification_launcher": {},
        "resources": {
            "nano_cpus": 64 * 10**9,
            "memory_bytes": 340 << 30,
            "memory_swap_bytes": 340 << 30,
            "ipc": "host",
        },
        "timeouts": {"run": 7800, "retire": 90},
        "environment": {"TF_GGUF_ASSETS_DIR": "/frontend"},
        "mounts": [
            {"source": "/held/models", "target": "/models/hf/hub", "read_only": True},
            {"source": "/held/frontend", "target": "/frontend", "read_only": True},
        ],
    }
    assert c.validate_benchmark_host_operation(host, inner) is inner
    for fault in ("writable", "candidate", "mode", "device-order"):
        wrong_host, wrong_inner = deepcopy(host), deepcopy(inner)
        if fault == "writable":
            wrong_host["mounts"][1]["read_only"] = False
        elif fault == "candidate":
            wrong_host["environment"]["TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN"] = "grouped-v1"
        elif fault == "mode":
            wrong_inner["mode"] = c.MODE
        else:
            wrong_inner["UUIDs"].reverse()
        with pytest.raises(ValueError):
            c.validate_benchmark_host_operation(wrong_host, wrong_inner)
