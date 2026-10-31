"""Opt-in profile input identity and transactional registration without a GPU."""

import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from tensorfold.families.qwen4_exp.cuda import execution_identity as identity
from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController
from tensorfold.families.qwen4_exp.cuda.speculation import MethodGeometry

_RUNTIME_OBSERVER = identity._runtime


def test_backend_tuning_inputs_bind_without_exposing_values_or_process_labels():
    environment = {"NCCL_SOCKET_NTHREADS": "2", "PYTORCH_CUDA_ALLOC_CONF": "backend:native",
                   "CUDA_FORCE_PTX_JIT": "0", "TENSORFOLD_TP_FAILURE_FD": "42",
                   "TENSORFOLD_SOURCE_REVISION": "documentary", "HF_TOKEN": "not-an-execution-input"}
    original = identity.tuning_environment_identity(environment)
    assert len(original) == 64 and all(character in "0123456789abcdef" for character in original)
    for name, value in (("NCCL_SOCKET_NTHREADS", "4"),
                        ("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync"), ("CUDA_FORCE_PTX_JIT", "1")):
        assert identity.tuning_environment_identity({**environment, name: value}) != original
    for name in ("TENSORFOLD_TP_FAILURE_FD", "TENSORFOLD_SOURCE_REVISION", "HF_TOKEN"):
        assert identity.tuning_environment_identity({**environment, name: "different"}) == original
    assert "not-an-execution-input" not in original


@pytest.fixture
def packet(tmp_path, monkeypatch):
    files = {}
    for name, value in {"target/weights.gguf": b"all original checkpoint bytes",
                        "metadata/config.json": b"{}", "metadata/tokenizer.json": b"{}",
                        "metadata/tokenizer_config.json": b"{}"}.items():
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(value)
        files[name] = path
    source, library = tmp_path / "source.py", tmp_path / "kernel.so"
    source.write_bytes(b"same canonical source")
    library.write_bytes(b"same loaded native kernel")
    checkpoint = NS(tensors={"weight": NS(ggml_type=8, shape=(32,), nbytes=34)})
    monkeypatch.setattr(identity, "artifact_files", lambda e: (files, checkpoint))
    monkeypatch.setattr(identity, "_source_files", lambda: {"source/source.py": source})
    monkeypatch.setattr(identity, "_runtime", lambda e, **kw: (
        "same host", {"fp16_accumulation": False}, {"torch": "pinned"}, {"rank": 0, "uuid": "real-observed"},
        {"library/kernel.so": library}))
    controller = AdaptiveController(slots=1, row_budget=1, proposal_limit=0, prefill_tokens=256)
    controller.enable_migration(1)
    cache = NS(owners=[0, 0], offsets=(0, 128), reserved_rows=129)
    state = NS(history_slot=0, capacity=129, kc=[cache],
               history_placement=NS(degree=1, page_tokens=128))
    engine = NS(model_dir=tmp_path, execution_provider="sm70", indexer_history="gpu", rank=0, tp=1,
                comm=None, multi=NS(controller=controller, states=[state], full_windows=1, keep=8,
                                   share=0.0, draft_host=None),
                mtp_model_dir=None, dspark_model_dir=None,
                w=NS(meta={"history_layout": "unique", "attention_degree": 1},
                     cfg=NS(execution_dtype="float16", vocab=128)), chat_template=None,
                method_geometry=MethodGeometry("none", 0), kv_pair=NS(identity="q8-explicit"),
                capacity_plan={"execution_envelope": {"target_rows": 1}}, points=None,
                context_window=128, streams=1)
    body = {"schema": identity.SCHEMA, "version": 1, "artifacts": {
        name: {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in files.items()}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(body))
    return engine, path, body, files


@pytest.mark.parametrize("migration", [True, False])
def test_exact_content_pass_is_opt_in_then_current_process_authority_reuses_hashes(packet, migration):
    engine, path, _, files = packet
    if not migration:
        engine.multi.controller.migration = engine.multi.controller.applied_migration = None
    identity.begin_execution_manifest(engine, path)
    first = identity.bind_execution_manifest(engine, path)
    assert first["verified_input_files"] == len(files)
    assert first["content_passes"] == 1
    assert first["identity"]["draft"] == "none"
    assert engine._execution_input_authority.bytes_read > sum(p.stat().st_size for p in files.values())
    second = identity.bind_execution_manifest(engine, path, rebind=True)
    assert second["identity"] == first["identity"]
    assert second["content_passes"] == second["bytes_read_this_binding"] == 0
    assert second["hashes_reused_this_binding"] == len(files) + 2
    assert engine.multi.controller.record()["profile_identity"] == first["identity"]


@pytest.mark.parametrize("change", ["missing", "extra", "wrong-hash", "wrong-size"])
def test_checkpoint_manifest_cannot_omit_inputs_or_authorize_unrelated_files(packet, change):
    engine, path, body, _ = packet
    if change == "missing":
        body["artifacts"].pop("metadata/tokenizer.json")
    elif change == "extra":
        body["artifacts"]["target/unused"] = {"size": 1, "sha256": "0" * 64}
    elif change == "wrong-hash":
        body["artifacts"]["target/weights.gguf"]["sha256"] = "0" * 64
    else:
        body["artifacts"]["target/weights.gguf"]["size"] += 1
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        identity.begin_execution_manifest(engine, path)
        identity.bind_execution_manifest(engine, path)
    assert engine.multi.controller.record()["profile_identity"] is None


def test_file_change_between_loader_boundary_and_hashing_is_rejected(packet):
    engine, path, _, files = packet
    identity.begin_execution_manifest(engine, path)
    files["target/weights.gguf"].write_bytes(b"changed during weight loading")
    with pytest.raises(ValueError, match="size|changed across loading"):
        identity.bind_execution_manifest(engine, path)


def test_unverified_disk_receipt_cannot_replace_current_process_hash_authority(packet):
    engine, path, _, _ = packet
    identity.begin_execution_manifest(engine, path)
    identity.bind_execution_manifest(engine, path)
    engine._execution_input_authority = {"sha256": "0" * 64}
    with pytest.raises(ValueError, match="current-process|content verification"):
        identity.bind_execution_manifest(engine, path, rebind=True)


def test_bad_replacement_table_preserves_previously_bound_registry(packet):
    engine, path, body, _ = packet
    identity.begin_execution_manifest(engine, path)
    first = identity.bind_execution_manifest(engine, path)
    before = engine.multi.controller.record()
    body["profiles"] = {"schema": "tensorfold.execution-profiles", "version": 1,
                        "identity": {**first["identity"], "source": "0" * 64}, "profiles": []}
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="identity"):
        identity.bind_execution_manifest(engine, path, rebind=True)
    assert engine.multi.controller.record() == before


def test_actual_owner_offsets_and_stationary_degree_are_bound_not_just_layout_name(packet):
    engine, path, _, _ = packet
    identity.begin_execution_manifest(engine, path)
    old = identity.bind_execution_manifest(engine, path)["identity"]
    engine.multi.controller.request_migration(4)
    with pytest.raises(ValueError, match="pending|mixed"):
        identity.bind_execution_manifest(engine, path, rebind=True)
    engine.multi.controller.migration_status(pending=False,
        request_epoch=engine.multi.controller.record()["migration"]["request_epoch"])
    state = engine.multi.states[0]
    state.history_placement.degree = 4
    state.kc[0].owners = [0, 1]
    state.kc[0].offsets = (0, -1)
    state.kc[0].reserved_rows = 128
    engine.w.meta["attention_degree"] = 4
    new = identity.bind_execution_manifest(engine, path, rebind=True)["identity"]
    assert new["history_layout"] == old["history_layout"]
    assert new["source"] == old["source"]
    assert new["placement"] != old["placement"]


def test_partial_page_move_refuses_binding_even_without_global_pending_flag(packet):
    engine, path, _, _ = packet
    identity.begin_execution_manifest(engine, path)
    engine.multi.states[0]._history_migration = {"target": 4}
    with pytest.raises(ValueError, match="partially migrated"):
        identity.bind_execution_manifest(engine, path)


def test_json_boundary_rejects_duplicate_keys_and_traversal(tmp_path):
    path = tmp_path / "packet.json"
    path.write_text('{"version":1,"version":1}')
    with pytest.raises(ValueError):
        identity._json(path)
    manifest = {"evidence": [{"path": "../outside.json", "sha256": "0" * 64}]}
    with pytest.raises(ValueError, match="traverse"):
        identity._evidence(manifest, tmp_path)


def test_topology_command_retains_bounded_output_and_owns_failure():
    import sys

    assert identity._command([sys.executable, "-c", "print('topology')"]) == "topology\n"
    with pytest.raises(ValueError, match="output bound"):
        identity._command([sys.executable, "-c", "print('x' * 70000)"])


@pytest.mark.parametrize("role", ["correctness", "resources", "performance"])
def test_role_evidence_reads_actual_raw_artifacts_and_validates_measurements(packet, role):
    engine, manifest_path, _, _ = packet
    identity.begin_execution_manifest(engine, manifest_path)
    actual = identity.bind_execution_manifest(engine, manifest_path)["identity"]
    root = manifest_path.parent
    reference, protocol = root / "reference.bin", root / "protocol.py"
    reference.write_bytes(b"independent reference output and hybrid state")
    protocol.write_bytes(b"registered predeclared comparison protocol")
    reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    sample = dict(elapsed_seconds=2.0, prompt_tokens=8, committed_tokens=16,
                  offered=1, completed=1, rejected=0, timeouts=0, ttft_seconds=.1, tpot_seconds=.01)
    measurements = {
        "correctness": {"comparisons": [{"contract": "accepted-prefix-hybrid-state",
                        "reference_sha256": reference_sha, "candidate_sha256": reference_sha}]},
        "resources": {"ranks": [{"rank": 0, "budget_bytes": 1000, "measured_peak_bytes": 800,
                                  "protected_future_bytes": 100, "materialized_sessions": 1, "context_tokens": 128}]},
        "performance": {"pairs": [{"reference": sample, "candidate": {**sample, "elapsed_seconds": 1.0}}] * 2,
                        "analysis": {"objective": "committed_decode_tokens_per_second", "confidence": .95,
                                     "lower_gain": .1, "upper_gain": 1.5, "protocol_source_sha256": protocol_sha}},
    }
    raw = root / "raw.json"
    raw_body = {"schema": "tensorfold.execution-profile-" + role, "version": 1,
                "identity": actual, "workload": {}, "execution": {}, "measurements": measurements[role]}
    raw.write_text(json.dumps(raw_body))
    receipt = {"schema": "tensorfold.execution-profile-evidence", "version": 1, "kind": role,
               "qualification": "live", "verdict": "passed", "identity": actual,
               "workload": {}, "execution": {}, "proof": {
                   "protocol": "tensorfold.execution-profile-" + role + "-v1",
                   "record_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "artifacts": [
                       {"path": p.name, "size": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in (raw, reference, protocol)]}}
    proof = root / "proof.json"
    proof.write_text(json.dumps(receipt))
    proof_sha = hashlib.sha256(proof.read_bytes()).hexdigest()
    manifest = {"evidence": [{"path": proof.name, "sha256": proof_sha}]}
    assert identity._evidence(manifest, root, identity=actual, ranks=1) == {proof_sha: receipt}
    # A retained "passed" label cannot hide changed/missing source evidence.
    reference.write_bytes(b"changed reference")
    with pytest.raises(ValueError, match="size|content"):
        identity._evidence(manifest, root, identity=actual, ranks=1)


def test_plain_passed_attestation_without_raw_measurements_is_not_a_proof(tmp_path):
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({"kind": "correctness", "identity": {}, "proof": {"passed": True}}))
    manifest = {"evidence": [{"path": proof.name, "sha256": hashlib.sha256(proof.read_bytes()).hexdigest()}]}
    with pytest.raises(ValueError, match="raw artifacts"):
        identity._evidence(manifest, tmp_path, identity={})


def test_query_primary_survives_independent_child_cleanup_failures(monkeypatch):
    primary = TimeoutError("query timed out")
    kill_error, close_error = OSError("kill failed"), OSError("pipe close failed")

    def fail(error):
        def operation(*args, **kwargs):
            raise error
        return operation

    child = NS(poll=lambda: None, kill=fail(kill_error), wait=lambda **kw: 0,
               stdout=NS(close=fail(close_error)))
    monkeypatch.setattr(identity.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(identity.selectors, "DefaultSelector", fail(primary))
    with pytest.raises(TimeoutError) as caught:
        identity._command(["nvidia-smi"])
    assert caught.value is primary

    retained, pending = [], [primary]
    while pending:
        error = pending.pop()
        if error is None or any(error is prior for prior in retained):
            continue
        retained.append(error)
        pending.extend((error.__cause__, error.__context__))
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
    assert kill_error in retained
    assert close_error in retained
    assert primary._tensorfold_retained_owners == [child]


@pytest.mark.parametrize("divergent", [None, "host", "file", "uuid"])
def test_rank_zero_hashes_once_only_after_shared_opened_identity_agreement(packet, monkeypatch, divergent):
    from concurrent.futures import ThreadPoolExecutor
    import copy
    import threading

    first, path, _, files = packet
    second = copy.copy(first)
    second.multi = copy.copy(first.multi)
    second.multi.states = copy.deepcopy(first.multi.states)
    second.multi.controller = AdaptiveController(slots=1, row_budget=1, proposal_limit=0, prefill_tokens=256)
    second.multi.controller.enable_migration(1)
    engines = [first, second]
    lock, values = threading.Condition(), {}

    class Store:
        def set(self, key, value):
            with lock:
                values[key] = value
                lock.notify_all()

        def get(self, key):
            with lock:
                return values[key]

        def check(self, keys):
            with lock:
                return all(key in values for key in keys)

        def wait(self, keys, timeout):
            with lock:
                if not lock.wait_for(lambda: all(key in values for key in keys), timeout=5):
                    raise TimeoutError("test rank store wait timed out")

        def delete_key(self, key):
            with lock:
                return values.pop(key, None) is not None

    class Comm:
        store = Store()

    other = path.parent / "other.gguf"
    other.write_bytes(files["target/weights.gguf"].read_bytes())
    old_artifacts = identity.artifact_files
    monkeypatch.setattr(identity, "artifact_files", lambda e: (
        {**files, "target/weights.gguf": other} if divergent == "file" and e.rank else old_artifacts(e)[0],
        old_artifacts(e)[1]))
    old_runtime = identity._runtime

    def runtime(engine, **kwargs):
        host, numerics, observed, topology, libraries = old_runtime(engine, **kwargs)
        return ((host + "other" if divergent == "host" and engine.rank else host), numerics, observed,
                {**topology, "rank": engine.rank,
                 "uuid": "same" if divergent == "uuid" else "GPU-" + str(engine.rank)}, libraries)

    monkeypatch.setattr(identity, "_runtime", runtime)
    captures, capture = [], identity._capture

    def observed_capture(*args, **kwargs):
        captures.append(1)
        return capture(*args, **kwargs)

    monkeypatch.setattr(identity, "_capture", observed_capture)
    for rank, engine in enumerate(engines):
        engine.rank, engine.tp, engine.comm = rank, 2, Comm()
        identity.begin_execution_manifest(engine, path)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(identity.bind_execution_manifest, e, path) for e in engines]
        if divergent:
            for future in futures:
                with pytest.raises(ValueError, match="one host|distinct GPU"):
                    future.result(timeout=10)
            assert not captures
        else:
            results = [future.result(timeout=10) for future in futures]
            assert results[0]["identity"] == results[1]["identity"]
            assert len(captures) == 1
            assert first._execution_input_authority.bytes_read > 0
            assert second._execution_input_authority is None
            assert not values  # No inventory, outcome, readiness or retirement keys survive.
            for _ in range(12):
                repeated = [workers.submit(identity.bind_execution_manifest, e, path, rebind=True) for e in engines]
                results = [future.result(timeout=10) for future in repeated]
                assert all(result["bytes_read_this_binding"] == 0 for result in results)
                assert not values


def test_new_lazy_library_hashes_only_new_bytes_and_updates_actual_kernel_identity(packet, monkeypatch):
    engine, path, _, _ = packet
    initial_runtime = identity._runtime
    library = path.parent / "later.so"
    library.write_bytes(b"newly mapped native implementation")
    enabled = [False]

    def runtime(engine, **kwargs):
        host, numeric, observed, topology, libraries = initial_runtime(engine, **kwargs)
        if enabled[0]:
            libraries = {**libraries, "library/later.so": library}
        return host, numeric, observed, topology, libraries

    monkeypatch.setattr(identity, "_runtime", runtime)
    identity.begin_execution_manifest(engine, path)
    first = identity.bind_execution_manifest(engine, path)
    enabled[0] = True
    second = identity.bind_execution_manifest(engine, path, rebind=True)
    assert second["identity"]["kernel_build"] != first["identity"]["kernel_build"]
    assert second["identity"]["target"] == first["identity"]["target"]
    assert second["bytes_read_this_binding"] == library.stat().st_size
    assert second["content_passes"] == 0
    third = identity.bind_execution_manifest(engine, path, rebind=True)
    assert third["bytes_read_this_binding"] == 0


@pytest.mark.parametrize("field", ["numerics", "runtime", "topology"])
def test_mutable_runtime_change_invalidates_stale_proof_even_if_replacement_is_bad(packet, monkeypatch, field):
    engine, path, body, _ = packet
    initial_runtime = identity._runtime
    changed = [False]

    def runtime(engine, **kwargs):
        observed = list(initial_runtime(engine, **kwargs))
        if changed[0]:
            index = {"numerics": 1, "runtime": 2, "topology": 3}[field]
            observed[index] = {**observed[index], "actual_observed_change": 1}
        return tuple(observed)

    monkeypatch.setattr(identity, "_runtime", runtime)
    identity.begin_execution_manifest(engine, path)
    old = identity.bind_execution_manifest(engine, path)["identity"]
    changed[0] = True
    body["profiles"] = {"schema": "tensorfold.execution-profiles", "version": 1, "identity": old, "profiles": []}
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="identity"):
        identity.bind_execution_manifest(engine, path, rebind=True)
    assert engine.multi.controller.record()["profile_identity"] is None
    body.pop("profiles")
    path.write_text(json.dumps(body))
    new = identity.bind_execution_manifest(engine, path, rebind=True)
    assert new["identity"][field] != old[field]
    assert new["bytes_read_this_binding"] == 0


def test_store_retirement_failure_contains_future_registrations_without_new_namespaces(packet, monkeypatch):
    engine, path, _, _ = packet
    values = {}

    class Store:
        def set(self, key, value):
            values[key] = value

        def check(self, keys):
            return all(key in values for key in keys)

    engine.comm = NS(store=Store())

    def bind(engine, path, *, channel, rebind, manifest_root):
        engine.comm.store.set(channel + "/rank/0", "bounded failed operation inventory")
        return {}

    def failed_retirement(engine, channel):
        raise OSError("store key retirement failed")

    monkeypatch.setattr(identity, "_bind_execution_manifest", bind)
    monkeypatch.setattr(identity, "_retire_store", failed_retirement)
    with pytest.raises(OSError, match="retirement"):
        identity.bind_execution_manifest(engine, path)
    before = dict(values)
    for _ in range(12):
        with pytest.raises(RuntimeError, match="contained"):
            identity.bind_execution_manifest(engine, path, rebind=True)
        assert values == before
    assert len(values) == 2


def _eight_rank_cohort(packet, monkeypatch):
    import copy
    import threading

    template, path, _, _ = packet
    condition, values, poison = threading.Condition(), {}, []
    aborted = [False]
    faults = []

    class Store:
        def set(self, key, value):
            with condition:
                if aborted[0]:
                    raise RuntimeError("test supervisor contained the cohort")
                values[key] = value
                condition.notify_all()
                if any(fault(key) for fault in faults):
                    raise OSError("rank store failed after partial outcome publication")

        def get(self, key):
            with condition:
                if aborted[0]:
                    raise RuntimeError("test supervisor contained the cohort")
                return values[key]

        def check(self, keys):
            with condition:
                return all(key in values for key in keys)

        def wait(self, keys, timeout):
            with condition:
                if not condition.wait_for(lambda: aborted[0] or all(key in values for key in keys), timeout=5):
                    raise TimeoutError("test cohort phase did not complete")
                if aborted[0]:
                    raise RuntimeError("test supervisor contained the cohort")

        def delete_key(self, key):
            with condition:
                return values.pop(key, None) is not None

    store = Store()
    engines = []
    for rank in range(8):
        engine = copy.copy(template)
        engine.rank, engine.tp, engine.comm = rank, 8, NS(store=store)
        engine.multi = copy.copy(template.multi)
        engine.multi.states = copy.deepcopy(template.multi.states)
        engine.multi.controller = AdaptiveController(slots=1, row_budget=1, proposal_limit=0, prefill_tokens=256)
        engine.multi.controller.enable_migration(1)

        def notify(error, rank=rank):
            with condition:
                poison.append((rank, error))
                aborted[0] = True
                condition.notify_all()

        engine.multi.poison_control = notify
        identity.begin_execution_manifest(engine, path)
        engines.append(engine)
    original_runtime = identity._runtime

    def runtime(engine, **kwargs):
        host, numerics, observed, topology, libraries = original_runtime(engine, **kwargs)
        return host, numerics, observed, {**topology, "rank": engine.rank, "uuid": "GPU-" + str(engine.rank)}, libraries

    monkeypatch.setattr(identity, "_runtime", runtime)
    return engines, path, values, poison, faults


@pytest.mark.parametrize("stage", ["preflight", "content", "verification", "evidence"])
def test_one_rank_read_failure_votes_refusal_without_mismatched_retirement_wait(packet, monkeypatch, stage):
    from concurrent.futures import ThreadPoolExecutor

    engines, path, values, poison, _ = _eight_rank_cohort(packet, monkeypatch)
    with ThreadPoolExecutor(max_workers=8) as workers:
        assert all(f.result(timeout=10) for f in [workers.submit(identity.bind_execution_manifest, e, path) for e in engines])
        before = [e.multi.controller.record() for e in engines]
        original_phase = identity._phase

        def phase(engine, channel, name, operation, **kwargs):
            if name == stage and engine.rank == (0 if stage == "content" else 3):
                def operation():
                    raise FileNotFoundError("rank-local missing runtime or proof input")
            return original_phase(engine, channel, name, operation, **kwargs)

        monkeypatch.setattr(identity, "_phase", phase)
        futures = [workers.submit(identity.bind_execution_manifest, e, path, rebind=True) for e in engines]
        for future in futures:
            with pytest.raises(ValueError, match="rank-local missing"):
                future.result(timeout=10)
    assert not poison
    assert not values
    assert [e.multi.controller.record() for e in engines] == before


def test_missing_relative_manifest_is_agreed_diagnostic_and_preserves_registry(packet, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    engines, path, values, poison, _ = _eight_rank_cohort(packet, monkeypatch)
    with ThreadPoolExecutor(max_workers=8) as workers:
        for result in [workers.submit(identity.bind_execution_manifest, e, path) for e in engines]:
            result.result(timeout=10)
        before = [e.multi.controller.record() for e in engines]
        pending = [workers.submit(FlashNextEngine._install_execution_manifest, e, "missing.json") for e in engines]
        for result in pending:
            with pytest.raises(ValueError, match="FileNotFoundError"):
                result.result(timeout=10)
    assert not values and not poison
    assert [e.multi.controller.record() for e in engines] == before


def test_uncertain_phase_publication_notifies_supervisor_without_entering_retirement(packet, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    engines, path, values, poison, faults = _eight_rank_cohort(packet, monkeypatch)
    faults.append(lambda key: key.endswith("/phase/preflight/3"))
    with ThreadPoolExecutor(max_workers=8) as workers:
        pending = [workers.submit(identity.bind_execution_manifest, e, path) for e in engines]
        for result in pending:
            with pytest.raises((OSError, RuntimeError)):
                result.result(timeout=10)
    assert poison
    assert not any("/retired/" in key or "/ready/retirement/" in key for key in values)
    assert all(e._abort_comm for e in engines)


def test_eight_rank_commit_defers_http_migration_between_final_vote_and_publication(packet, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    engines, path, values, poison, _ = _eight_rank_cohort(packet, monkeypatch)
    with ThreadPoolExecutor(max_workers=8) as workers:
        for result in [workers.submit(identity.bind_execution_manifest, e, path) for e in engines]:
            result.result(timeout=10)
        ready, release = threading.Event(), threading.Event()
        gate = threading.Barrier(8, action=ready.set)
        original_phase = identity._phase

        def phase(engine, channel, name, operation, **kwargs):
            if name == "commit":
                gate.wait(timeout=5)
                assert release.wait(timeout=5)
            return original_phase(engine, channel, name, operation, **kwargs)

        monkeypatch.setattr(identity, "_phase", phase)
        pending = [workers.submit(identity.bind_execution_manifest, e, path, rebind=True) for e in engines]
        try:
            assert ready.wait(timeout=5)
            requested = engines[0].multi.controller.request_migration(4)
            assert requested["profile_transaction_active"]
            assert requested["deferred_migration"] == 4
            assert requested["migration"]["request_epoch"] == 0
        finally:
            release.set()
        results = [future.result(timeout=10) for future in pending]
    assert all(r["identity"] == results[0]["identity"] for r in results)
    assert engines[0].multi.controller.record()["migration"]["request_epoch"] == 1
    assert not values and not poison


@pytest.mark.torch
@pytest.mark.parametrize("fact", ["fp16-policy", "configured-power", "qsa-policy"])
@pytest.mark.parametrize("uuid_form", ["torch", "nvidia"])
def test_registration_observes_uncached_backend_policy_and_fresh_configured_gpu_operation(
        packet, monkeypatch, fact, uuid_form):
    from uuid import UUID

    import torch
    from tensorfold.cuda.sm70 import attention_qualification

    engine, path, body, _ = packet
    engine.w.device = torch.device("cpu")
    engine.w.cfg.eos = (2,)
    engine.rope = NS(metadata=lambda: {"rope_type": "default"})
    engine.confidence, engine.graphs_enabled = .5, False
    uuid = "953e0462-dfab-541a-e631-573d64f8c5ad"
    props = NS(uuid=UUID(uuid) if uuid_form == "torch" else "GPU-" + uuid,
               name="V100", total_memory=32 << 30, multi_processor_count=80)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: props)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (7, 0))
    monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", lambda: "Cublas")
    monkeypatch.setattr(torch._C, "_cuda_getCublasWorkspaceSize", lambda: 1024)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False)
    monkeypatch.setattr(identity.importlib.metadata, "version", lambda name: "pinned-version")
    monkeypatch.setattr(identity.importlib.metadata, "distributions", lambda: [])
    power = ["200"]
    queries = []

    def command(argv):
        queries.append(list(argv))
        if any("power.limit" in part for part in argv):
            assert "--id=GPU-" + uuid in argv
            return "GPU-" + uuid + "," + power[0] + ",N/A,N/A,Default,Enabled\n"
        return "fixed physical topology\n"

    monkeypatch.setattr(identity, "_command", command)
    native = path.parent / "kernel.so"

    def small(name, limit=65536):
        name = str(name)
        if name.endswith("/maps"):
            return ("0-1 r-xp 0 00:00 0 " + str(native) + "\n").encode()
        if name.endswith("/cpuinfo"):
            return b"model name : qualified CPU\n"
        if name.endswith("/meminfo"):
            return b"MemTotal: 1024 kB\n"
        return b"same immutable host identity"

    monkeypatch.setattr(identity, "_small", small)
    monkeypatch.setattr(identity, "_runtime", _RUNTIME_OBSERVER)
    authority = attention_qualification.runtime_identity
    authority.cache_clear()
    try:
        cached = authority(engine.w.device)
        assert cached["fp16_reduced_precision"] is False
        identity.begin_execution_manifest(engine, path)
        old = identity.bind_execution_manifest(engine, path)["identity"]
        if fact == "fp16-policy":
            monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", True)
            assert authority(engine.w.device)["fp16_reduced_precision"] is False
            assert authority.__wrapped__(engine.w.device)["fp16_reduced_precision"] is True
        elif fact == "qsa-policy":
            from tensorfold.families.qwen4_exp import qsa_selection

            monkeypatch.setattr(qsa_selection, "POLICY", qsa_selection.POLICY + "-changed")
        else:
            power[0] = "140"
        body["profiles"] = {"schema": "tensorfold.execution-profiles", "version": 1,
                            "identity": old, "profiles": []}
        path.write_text(json.dumps(body))
        with pytest.raises(ValueError, match="identity"):
            identity.bind_execution_manifest(engine, path, rebind=True)
        assert engine.multi.controller.record()["profile_identity"] is None
        body.pop("profiles")
        path.write_text(json.dumps(body))
        result = identity.bind_execution_manifest(engine, path, rebind=True)
        field = "numerics" if fact in ("fp16-policy", "qsa-policy") else "topology"
        assert result["identity"][field] != old[field]
        assert result["bytes_read_this_binding"] == 0
        operating_queries = [argv for argv in queries if any("power.limit" in part for part in argv)]
        assert len(operating_queries) == 3  # Initial capture and both registrations observe configured operation.
        assert "N/A" in engine._execution_bind_local["topology"]["configured_gpu_operation"]
    finally:
        authority.cache_clear()


def test_native_retained_owner_cannot_be_hidden_by_exception_attribute_overrides(packet):
    engine, _, _, _ = packet
    notified = []
    engine.multi.poison_control = notified.append

    class HiddenCause(OSError):
        def __getattribute__(self, name):
            if name in ("__cause__", "__context__", "_tensorfold_model_fd_scope"):
                return None
            return super().__getattribute__(name)

    primary = HiddenCause("input failed with retained native owner")
    cause = OSError("descriptor retirement did not finish")
    namespace = BaseException.__dict__["__dict__"].__get__(cause, type(cause))
    namespace["_tensorfold_model_fd_scope"] = object()
    BaseException.__cause__.__set__(primary, cause)

    def operation():
        raise primary

    with pytest.raises(HiddenCause) as caught:
        identity._phase(engine, None, "preflight", operation)
    assert caught.value is primary
    assert notified == [primary]
    assert engine._execution_manifest_uncertain


@pytest.mark.parametrize("changed", ["decode-share", "draft-map"])
def test_actual_decode_share_and_cpu_draft_map_reject_old_profile_without_checkpoint_rehash(packet, changed):
    import numpy as np

    engine, path, body, _ = packet

    def selection(values):
        result = np.asarray(values, dtype=np.int64)
        result.flags.writeable = False
        return result

    engine.multi.draft_host = selection([1, 17, 63])
    identity.begin_execution_manifest(engine, path)
    old = identity.bind_execution_manifest(engine, path)["identity"]
    if changed == "decode-share":
        engine.multi.share = .5
    else:
        engine.multi.draft_host = selection([1, 18, 63])
    body["profiles"] = {"schema": "tensorfold.execution-profiles", "version": 1,
                        "identity": old, "profiles": []}
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="identity"):
        identity.bind_execution_manifest(engine, path, rebind=True)
    assert engine.multi.controller.record()["profile_identity"] is None
    body.pop("profiles")
    path.write_text(json.dumps(body))
    result = identity.bind_execution_manifest(engine, path, rebind=True)
    field = "runtime" if changed == "decode-share" else "numerics"
    assert result["identity"][field] != old[field]
    for unchanged in ("target", "draft", "source", "kernel_build", "context_tokens", "resident_sessions"):
        assert result["identity"][unchanged] == old[unchanged]
    assert result["bytes_read_this_binding"] == 0
    assert not engine.multi.draft_host.flags.writeable


@pytest.mark.parametrize("encoding", ["<i8", ">i8"])
def test_selected_draft_hash_has_canonical_endian_and_full_vocab_has_explicit_count(packet, encoding):
    import numpy as np
    import struct

    engine, _, _, _ = packet
    assert identity._draft_vocabulary(engine) == {"mode": "full-vocabulary", "vocabulary_size": 128}
    engine.multi.draft_host = np.array([1, 17, 63], dtype=encoding)
    engine.multi.draft_host.flags.writeable = False
    observed = identity._draft_vocabulary(engine)
    assert observed == {"mode": "selected-global-id-map", "vocabulary_size": 128,
                        "selected_count": 3, "encoding": "little-endian-int64",
                        "sha256": hashlib.sha256(struct.pack("<3q", 1, 17, 63)).hexdigest()}
    engine.multi.draft_host = None
    engine.w.cfg.vocab = 129
    assert identity._draft_vocabulary(engine) == {"mode": "full-vocabulary", "vocabulary_size": 129}


def test_overrideable_exception_accessors_are_not_used_for_refusal_classification(packet):
    engine, _, _, _ = packet
    inspected, notified = [], []
    engine.multi.poison_control = notified.append

    class ThrowingAccessors(OSError):
        def __getattribute__(self, name):
            if name in ("__cause__", "__context__", "__dict__", "_tensorfold_retained_owners",
                        "_tensorfold_model_fd_scope", "_tensorfold_checkpoint_file_scope"):
                inspected.append(name)
                raise RuntimeError("untrusted exception accessor must not run")
            return super().__getattribute__(name)

    primary = ThrowingAccessors("ordinary input read refusal")

    def operation():
        raise primary

    with pytest.raises(ValueError, match="ordinary input read refusal") as caught:
        identity._phase(engine, None, "preflight", operation)
    assert caught.value.__cause__ is primary
    assert not inspected and not notified


def test_group_children_cannot_be_hidden_by_metaclass_mro_or_exceptions_properties(packet):
    engine, _, _, _ = packet
    notified, inspected = [], []
    engine.multi.poison_control = notified.append

    class HideMro(type):
        @property
        def __mro__(cls):
            inspected.append("mro")
            return (OSError, BaseException, object)

    class HiddenGroup(ExceptionGroup, metaclass=HideMro):
        @property
        def exceptions(self):
            inspected.append("exceptions")
            return ()

    retained = OSError("native checkpoint owner still retained")
    BaseException.__dict__["__dict__"].__get__(retained, type(retained))["_tensorfold_checkpoint_file_scope"] = object()
    primary = OSError("input operation failed")
    BaseException.__cause__.__set__(primary, HiddenGroup("cleanup", [retained]))

    def operation():
        raise primary

    with pytest.raises(OSError) as caught:
        identity._phase(engine, None, "preflight", operation)
    assert caught.value is primary
    assert notified == [primary]
    assert not inspected


@pytest.mark.parametrize("fault", ["format", "inspection"])
def test_refusal_diagnostic_failure_preserves_original_primary_and_contains_cohort(packet, monkeypatch, fault):
    engine, _, _, _ = packet
    notified = []
    engine.multi.poison_control = notified.append
    diagnostic = RuntimeError("diagnostic formation failed")

    class BadFormat(OSError):
        def __str__(self):
            raise diagnostic

    primary = BadFormat() if fault == "format" else OSError("original input failure")
    if fault == "inspection":
        def failed_inspection(error):
            raise diagnostic
        monkeypatch.setattr(identity, "_unretired", failed_inspection)

    def operation():
        raise primary

    with pytest.raises(OSError) as caught:
        identity._phase(engine, None, "preflight", operation)
    assert caught.value is primary
    assert BaseException.__cause__.__get__(primary, type(primary)) is diagnostic
    assert notified == [primary]
