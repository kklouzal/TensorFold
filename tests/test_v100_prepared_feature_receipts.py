"""CPU-only close-journal/receipt contracts; no CUDA or Graph emulation claim."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tools import launch_cuda_tp as launcher
from tools import v100_owned_qualification as owned


class Graph:
    def __init__(self):
        self.prepared, self.closed, self.failure = True, False, None
        self.replays = dict(stats=0, probability=0, pv=0)
        self.charged = 100
        self.source = "a" * 64

    def record(self):
        return {"policy": "reference-aten128-three-islands-v1", "source_sha256": self.source,
                "replays": dict(self.replays), "before": dict(free=1000, allocated=200, reserved=300),
                "after": dict(free=900, allocated=300, reserved=400),
                "graph_budget_bytes": 1000, "charged_preparation_bytes": self.charged,
                "retained_tensor_payload_bytes": 100, "private_pool_spare_bytes": None}


class Engine:
    def __init__(self, graphs=True):
        self.rank, self._closed, self._abort_comm = 0, False, False
        self.close_calls, self.close_error, self.restore_error, self.prove_close = 0, None, None, True
        self._owned_expert_plan = self._owned_expert_gate_up_requested = None
        roles = {role: Graph() for role in ("decode", "mtp", "prefill")} if graphs else {}
        owners = {role: NS(reference_tiles=NS(arithmetic=graph)) for role, graph in roles.items()}
        self.multi = NS(**{key: NS(history_owner=owners[role]) for key, role in
                         (("buf", "decode"), ("pbuf", "prefill"), ("mbuf", "mtp"))}) if graphs else None
        self.w = NS(mtp=object())
        self._owner_preparation = NS(reference_arithmetic_request=NS(graph_budget_bytes=1000,
            policy="reference-aten128-three-islands-v1") if graphs else None,
            reference_arithmetic=roles, workspaces=owners, reference_arithmetic_source="a" * 64,
            reference_arithmetic_bytes=300, before=dict(physical_free_bytes=1000,
                torch_allocated_bytes=200, torch_reserved_bytes=300),
            after=dict(physical_free_bytes=700, torch_allocated_bytes=500, torch_reserved_bytes=600),
            private_spare_bytes=None)
        self._owned_qualification_host = (10000000, "b" * 64)
        self.retention_catalog = NS(host_resources=NS(baseline_host_bytes=10000000))

    def __delattr__(self, name):
        if name == "close" and self.restore_error is not None:
            raise self.restore_error
        object.__delattr__(self, name)

    def close(self, *, abort=False):
        self.close_calls += 1
        self._abort_comm = self._abort_comm or abort
        if self.close_error is not None:
            raise self.close_error
        if self.prove_close:
            for graph in self._owner_preparation.reference_arithmetic.values():
                graph.closed = True
            self._owner_preparation = self.multi = self.w = None
            self._closed = True
        return "retired"


def resources():
    # Explicit test admission fixture; the actual helper policy runs pinned.
    return dict(receipt_bytes_limit=36864, total_global_host_obligation_bytes=10000000,
                input_sha256="b" * 64, maximum_roles=3, additional_device_bytes=0)


def activate(engine):
    for graph in engine._owner_preparation.reference_arithmetic.values():
        graph.replays = dict(stats=1, probability=2, pv=3)


@pytest.fixture(autouse=True)
def notification(monkeypatch):
    monkeypatch.setattr(owned, "notify_failure", lambda: True)


@pytest.mark.parametrize("instance_attribute", [False, True])
def test_normal_close_proves_all_roles_after_runtime_releases_journal(tmp_path, instance_attribute):
    engine = Engine()
    if instance_attribute:
        original = engine.close
        engine.close = lambda: original()
    old = engine.close
    marker = owned.prepare_prepared_features(engine, resources(), tmp_path, None)
    assert marker["expected_roles"] == ["decode", "prefill", "mtp"]
    prepared = json.loads((tmp_path / marker["directory"] / "prepared.json").read_text())
    assert prepared == marker and all(not r["closed"] for r in prepared["roles"].values())
    activate(engine)
    assert engine.close() == "retired"
    assert engine.close == old and ("close" in vars(engine)) is instance_attribute
    final = json.loads((tmp_path / marker["directory"] / "summary.json").read_text())
    assert final["closed"] and not final["failed"]
    assert all(r["closed"] and r["replays"] == dict(stats=1, probability=2, pv=3)
               for r in final["roles"].values())
    assert final["preparation"]["charged_preparation_bytes"] == 300
    assert final["preparation"]["private_pool_spare_bytes"] is None
    assert engine._owner_preparation is None  # Observer retained only the Graphs needed after close.


def test_graph_off_gate_up_on_has_explicit_null_graphs(monkeypatch, tmp_path):
    engine = Engine(graphs=False)
    monkeypatch.setattr(owned, "_grouped_gate_up_receipt", lambda e, rows: {"requested_rows": rows, "actual_rows": rows})
    marker = owned.prepare_prepared_features(engine, resources(), tmp_path, 2)
    engine.close()
    final = json.loads((tmp_path / marker["directory"] / "summary.json").read_text())
    assert marker["roles"] is None and marker["expected_roles"] == []
    assert final["roles"] is None and final["grouped_gate_up"] == {"requested_rows": 2, "actual_rows": 2}


def test_no_features_or_unpaid_resources_refuse_before_install(tmp_path):
    engine = Engine(graphs=False)
    old = engine.close
    with pytest.raises(ValueError, match="require Graph or"):
        owned.prepare_prepared_features(engine, resources(), tmp_path, None)
    assert engine.close == old and not list(tmp_path.iterdir())
    engine = Engine()
    engine._owned_qualification_host = (1, "b" * 64)
    with pytest.raises(ValueError, match="host admission"):
        owned.prepare_prepared_features(engine, resources(), tmp_path, None)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("counter", ["stats", "probability", "pv"])
def test_each_configured_role_requires_positive_each_counter(tmp_path, counter):
    engine = Engine()
    activate(engine)
    engine._owner_preparation.reference_arithmetic["prefill"].replays[counter] = 0
    observer = owned._PreparedFeatures(engine, tmp_path, resources(), None)
    observer.install()
    with pytest.raises(RuntimeError, match="lacks stats"):
        engine.close()
    assert engine._closed and engine._abort_comm
    assert not (observer.directory / "summary.json").exists()
    assert "close" not in vars(engine)


@pytest.mark.parametrize("corruption", ["source", "charged", "failure", "closed"])
def test_final_role_metadata_and_retirement_fail_closed(tmp_path, corruption):
    engine = Engine()
    activate(engine)
    observer = owned._PreparedFeatures(engine, tmp_path, resources(), None)
    observer.install()
    graph = observer.graphs["decode"]
    if corruption == "source":
        graph.source = "c" * 64
    elif corruption == "charged":
        graph.charged += 1
    elif corruption == "failure":
        graph.failure = OSError("Graph failure")
    else:
        original = engine.close
        def close_without_graph_retirement():
            result = original()
            graph.closed = False
            return result
        # Test original close's contradictory retirement result, not hook replacement.
        observer.original_close = close_without_graph_retirement
    with pytest.raises((ValueError, RuntimeError)):
        engine.close()
    assert engine._abort_comm and not (observer.directory / "summary.json").exists()


def test_primary_close_failure_keeps_owners_and_never_retries_even_if_notification_fails(monkeypatch, tmp_path):
    engine = Engine()
    primary, notify = OSError("close failed"), RuntimeError("notification failed")
    engine.close_error = primary
    observer = owned._PreparedFeatures(engine, tmp_path, resources(), None)
    observer.install()
    monkeypatch.setattr(owned, "notify_failure", lambda: (_ for _ in ()).throw(notify))
    with pytest.raises(OSError) as caught:
        engine.close()
    assert caught.value is primary and primary.__cause__ is not None
    assert observer.engine is engine and len(observer.graphs) == 3 and not observer.closed
    with pytest.raises(OSError) as repeated:
        engine.close()
    assert repeated.value is primary and engine.close_calls == 1
    assert not (observer.directory / "summary.json").exists()


def test_unproved_engine_close_cannot_publish_success(tmp_path):
    engine = Engine()
    engine.prove_close = False
    observer = owned._PreparedFeatures(engine, tmp_path, resources(), None)
    observer.install()
    with pytest.raises(RuntimeError, match="Engine close"):
        engine.close()
    assert observer.engine is engine and len(observer.graphs) == 3
    assert not (observer.directory / "summary.json").exists()


def test_semantic_primary_survives_restore_and_notification_failures(monkeypatch, tmp_path):
    engine = Engine()
    restore, notify = OSError("restore failed"), OSError("notify failed")
    observer = owned._PreparedFeatures(engine, tmp_path, resources(), None)
    observer.install()
    engine.restore_error = restore
    monkeypatch.setattr(owned, "notify_failure", lambda: (_ for _ in ()).throw(notify))
    with pytest.raises(RuntimeError, match="lacks stats") as caught:
        engine.close()
    assert caught.value is observer.failure
    def contains(error, sought):
        return error is sought or any(contains(e, sought) for e in getattr(error, "exceptions", ()))
    assert contains(caught.value.__cause__, restore) and contains(caught.value.__cause__, notify)
    assert not (observer.directory / "summary.json").exists()


def test_summary_publication_failure_reports_failure_after_restoring_original(monkeypatch, tmp_path):
    engine = Engine()
    activate(engine)
    observer = owned._PreparedFeatures(engine, tmp_path, resources(), None)
    observer.install()
    write = owned.write_json
    failure = OSError("summary publication failed")
    def broken(path, value, **kwargs):
        if Path(path).name == "summary.json":
            raise failure
        return write(path, value, **kwargs)
    monkeypatch.setattr(owned, "write_json", broken)
    with pytest.raises(OSError) as caught:
        engine.close()
    assert caught.value is failure and engine._closed and engine._abort_comm
    assert "close" not in vars(engine) and len(observer.graphs) == 3


def test_existing_namespace_and_summary_are_preserved(tmp_path):
    engine = Engine()
    observer = owned._PreparedFeatures(engine, tmp_path, resources(), None)
    observer.directory.mkdir()
    with pytest.raises(FileExistsError):
        observer.install()
    assert "close" not in vars(engine)
    observer.directory.rmdir()
    observer.install()
    target = observer.directory / "summary.json"
    target.write_text("user bytes")
    activate(engine)
    with pytest.raises(ValueError, match="already exists"):
        engine.close()
    assert target.read_text() == "user bytes"


@pytest.mark.parametrize("extra", [[], ["--normalization-parity"], ["--scheduler-trace", "trace"],
                                    ["--history-input", "data"], ["--router-row-input", "data"]])
def test_parent_capture_exclusions_precede_output_creation(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(launcher, "supervise", lambda *a, **k: pytest.fail("invalid mode spawned"))
    args = ["--prepared-feature-receipts", "--gpus", "0,1,2,3,4,5,6,7", "--out", str(tmp_path / "out")]
    if extra:
        args += ["--benchmark-plan", "plan", *extra]
    with pytest.raises(SystemExit):
        owned.main([*args, "--", "model"])
    assert not (tmp_path / "out").exists()


def test_parent_flag_precedes_separator_for_every_rank(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(launcher, "supervise", lambda cmd, env, **kw: commands.extend(cmd) or 0)
    owned.main(["--prepared-feature-receipts", "--benchmark-plan", "plan", "--gpus", "0,1,2,3,4,5,6,7",
                "--out", str(tmp_path / "out"), "--", "model"])
    assert len(commands) == 8
    assert all(c.index("--prepared-feature-receipts") < c.index("--") for c in commands)


def test_falsey_metadata_primary_is_not_replaced_by_cleanup_selection(tmp_path):
    class FalseyError(Exception):
        def __bool__(self):
            return False
    engine = Engine()
    activate(engine)
    observer = owned._PreparedFeatures(engine, tmp_path, resources(), None)
    observer.install()
    primary = FalseyError("actual graph metadata failure")
    observer.graphs["decode"].record = lambda: (_ for _ in ()).throw(primary)
    with pytest.raises(FalseyError) as caught:
        engine.close()
    assert caught.value is primary and engine._abort_comm
    assert "close" not in vars(engine)


def test_schema_bound_covers_worst_accepted_three_role_metadata(tmp_path, record_testsuite_property):
    engine = Engine()
    maximum = 2**63 - 1
    preparation = engine._owner_preparation
    preparation.reference_arithmetic_request.graph_budget_bytes = maximum
    preparation.reference_arithmetic_request.policy = "\uffff" * 128
    for graph in preparation.reference_arithmetic.values():
        graph.replays = dict(stats=maximum, probability=maximum, pv=maximum)
        original = graph.record
        def worst(original=original):
            record = original()
            record.update(policy="\uffff" * 128, graph_budget_bytes=maximum,
                          retained_tensor_payload_bytes=maximum, private_pool_spare_bytes=maximum,
                          before=dict.fromkeys(("free", "allocated", "reserved"), maximum),
                          after=dict.fromkeys(("free", "allocated", "reserved"), maximum))
            return record
        graph.record = worst
    observer = owned._PreparedFeatures(engine, tmp_path, resources(), None)
    largest_role = max(len(json.dumps(v, ensure_ascii=True, indent=1).encode()) for v in observer.marker["roles"].values())
    record_testsuite_property("prepared_feature_worst_role_bytes", largest_role)
    record_testsuite_property("prepared_feature_worst_marker_bytes", len(observer._encoded(observer.marker)))
    assert largest_role < 8192
    assert len(observer._encoded(observer.marker)) < resources()["receipt_bytes_limit"]
    observer.install()
    engine.close()
    final = json.loads((observer.directory / "summary.json").read_text())
    assert len(json.dumps(final, ensure_ascii=True, indent=1).encode()) < resources()["receipt_bytes_limit"]


def test_resource_helper_pays_two_trees_and_owned_source_bytes(monkeypatch):
    calls = []
    def policy(capacity, limit):
        calls.append((capacity, limit))
        return dict(per_rank_payload_bytes=100, global_host_obligation_bytes=800, receipt_bytes_limit=36864)
    monkeypatch.setattr(owned, "_history_resources", policy)
    result = owned._prepared_feature_resources()
    assert calls == [(32, 4096)]
    assert result["per_rank_payload_bytes"] == 200 + Path(owned.__file__).stat().st_size
    assert result["global_host_obligation_bytes"] == 8 * result["per_rank_payload_bytes"]
    assert result["additional_device_bytes"] == 0 and result["maximum_roles"] == 3


def test_worker_admits_and_publishes_prepared_marker_through_existing_callback(monkeypatch, tmp_path):
    import os
    from tensorfold import cli
    from tensorfold.cuda.cohort import FAILURE_FD_ENV
    from tools import bench_concurrent
    engine = Engine()
    seen = []
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    monkeypatch.setenv(FAILURE_FD_ENV, str(write_fd))
    monkeypatch.setattr(bench_concurrent, "load_matrix", lambda _: {"cohorts": [{"level": 1}]})
    monkeypatch.setattr(owned, "_history_resources", lambda *a: dict(per_rank_payload_bytes=100,
        global_host_obligation_bytes=800, receipt_bytes_limit=36864))
    monkeypatch.setattr(owned, "prepare_benchmark", lambda *a: seen.append("benchmark"))
    for name in ("_half_policy", "_router_policy", "_empty_prefix_policy"):
        monkeypatch.setattr(owned, name, lambda _: {})
    def publish(e, output, **kwargs):
        assert seen == ["benchmark"]
        marker = kwargs["prepared_features"]
        assert marker["resources"]["total_global_host_obligation_bytes"] == e._owned_qualification_host[0]
        assert (Path(output) / marker["directory"] / "prepared.json").is_file()
        seen.append("published")
    monkeypatch.setattr(owned, "publish_receipt", publish)
    def serve(args):
        engine._owned_qualification_host = args._owned_expert_qualification.host_obligation
        engine.retention_catalog.host_resources.baseline_host_bytes = engine._owned_qualification_host[0]
        args._owned_expert_qualification.before_listener(engine)
        activate(engine)
        engine.close()
        return 0
    monkeypatch.setattr(cli, "cmd_serve", serve)
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    try:
        assert owned.worker(["--prepared-feature-receipts", "--benchmark-plan", str(plan),
            "--out", str(tmp_path), "--", "model", "--backend", "cuda", "--tp", "8",
            "--rank", "0", "--expert-tensor-parallel-size", "4"]) == 0
        assert seen == ["benchmark", "published"]
        assert (tmp_path / "prepared-features-rank-0/summary.json").is_file()
    finally:
        os.close(write_fd)
        os.close(read_fd)


@pytest.mark.parametrize("corruption", ["roles", "owner", "duplicate", "budget", "counter"])
def test_invalid_preparation_is_rejected_before_namespace_or_hook(tmp_path, corruption):
    engine = Engine()
    preparation = engine._owner_preparation
    old = engine.close
    if corruption == "roles":
        del preparation.reference_arithmetic["mtp"]
    elif corruption == "owner":
        engine.multi.buf.history_owner = object()
    elif corruption == "duplicate":
        preparation.reference_arithmetic["mtp"] = preparation.reference_arithmetic["decode"]
        preparation.workspaces["mtp"].reference_tiles.arithmetic = preparation.reference_arithmetic["decode"]
    elif corruption == "budget":
        preparation.reference_arithmetic_request.graph_budget_bytes = 299
    else:
        preparation.reference_arithmetic["decode"].replays["stats"] = False
    with pytest.raises(ValueError):
        owned.prepare_prepared_features(engine, resources(), tmp_path, None)
    assert engine.close == old and not list(tmp_path.iterdir())


def test_aborted_close_is_not_a_successful_feature_retirement(tmp_path):
    engine = Engine()
    activate(engine)
    observer = owned._PreparedFeatures(engine, tmp_path, resources(), None)
    observer.install()
    with pytest.raises(RuntimeError, match="aborted cohort"):
        engine.close(abort=True)
    assert engine._closed and observer.engine is engine
    assert not (observer.directory / "summary.json").exists()
