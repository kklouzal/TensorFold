"""Stdlib-only host/controller contracts; no Docker/GPU or qualification claim."""
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import v100_native_contract_host as host

U = ["GPU-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-000000000002"]
CID = "b" * 64


def file_record(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def inner(mode="cuda"):
    return {"schema": "tensorfold.native-contract-result-v1", "mode": mode,
            "pytest_exit": 0, "JUnit": {"tests": "2", "errors": "0", "failures": "0", "skipped": "0"},
            "post_errors": [], "primary_error": None, "diagnostic_errors": [], "context_fatal": False,
            "source_after_match": True, "first_failure_observed": False, "source_commit": "c" * 40,
            "image_id_declared": "sha256:" + "a" * 64, "modules": ["tests/test_contract.py"]}


def document(spec, state=None):
    r = spec["resources"]
    return {"Id": CID, "Name": "/" + spec["name"], "Image": spec["image_id"],
            "Config": {"User": "1000:1000", "Env": [k + "=" + v for k, v in spec["environment"].items()],
                       "Entrypoint": [spec["entrypoint"]], "Cmd": spec["argv"], "WorkingDir": "/canonical",
                       "Labels": {"org.tensorfold.native-contract": host.ownership_label(spec)}},
            "HostConfig": {"Runtime": "nvidia" if spec["uuids"] else "runc", "NetworkMode": "none",
                           "ReadonlyRootfs": True, "Privileged": False, "Devices": [], "CapAdd": None, "PidMode": "",
                           "SecurityOpt": ["label=disable"] if r["ipc"] == "host" else None,
                           "DeviceRequests": [{"DeviceIDs": spec["uuids"], "Driver": "nvidia", "Count": 0,
                                               "Capabilities": [["gpu"]]}] if spec["uuids"] else [],
                           "NanoCpus": r["nano_cpus"], "Memory": r["memory_bytes"], "MemorySwap": r["memory_swap_bytes"],
                           "IpcMode": r["ipc"], "Tmpfs": r["tmpfs"], "Ulimits": r["ulimits"]},
            "Mounts": [{"Type": "bind", "Source": m["source"], "Destination": m["target"], "RW": not m["read_only"]}
                       for m in spec["mounts"]],
            "State": state or {"Status": "created", "Running": False, "Pid": 0, "ExitCode": 0, "OOMKilled": False}}


def state(running=False):
    return {"Status": "running" if running else "exited", "Running": running,
            "Pid": 42 if running else 0, "ExitCode": 0, "OOMKilled": False}


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.actions = []

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.actions.append("wait")
        self.returncode = 0
        return 0

    def terminate(self):
        self.actions.append("terminate")

    def kill(self):
        self.actions.append("kill")


class FakeSystem:
    def __init__(self, spec, output, states=None):
        self.spec, self.output = spec, output
        self.states = states or [state()]
        self.group = "/system.slice/docker-" + CID + ".scope"
        self.started = False
        self.calls, self.snapshots = [], 0
        self.process = FakeProcess()
        self.mid_error = self.close_error = self.after_error = self.bind_error = None
        self.bad_create = None
        self.cancel_at_create = False

    def command(self, argv, timeout=30):
        self.calls.append(argv)
        if argv[:2] == ["docker", "create"]:
            if self.cancel_at_create:
                identity = json.loads((self.output / "controller-identity.json").read_text())
                host.request_cancel(self.output, identity["run_token"])
            return CID + "\n"
        return ""

    def inspect(self, _cid):
        if not self.started:
            value = document(self.spec)
            if self.bad_create:
                self.bad_create(value)
            return value
        current = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return document(self.spec, current)

    def start(self, _cid, _log):
        self.started = True
        (self.output / "named-suite-result.json").write_text(json.dumps(inner("cuda" if self.spec["uuids"] else "cpu")))
        return self.process

    def snapshot(self, _spec):
        self.snapshots += 1
        if self.snapshots > 1 and self.after_error:
            raise self.after_error
        return {"frozen": True}

    def health(self, _spec, owned=None):
        if owned and self.mid_error:
            raise self.mid_error
        return {"facts": self.spec["health_baseline"], "raw_compute": ""}

    def bind(self, _cid, _document, _spec):
        if self.bind_error:
            raise self.bind_error
        return {"owned": self.group}

    def sample(self, _spec):
        return {"owned": self.group}

    def observe(self, _cid, _spec):
        return {"phase": None, "compiler_processes": []}

    def close(self):
        self.calls.append(["close"])
        if self.close_error:
            raise self.close_error


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "attempt"
        self.source, self.operator, self.cache = (self.root / n for n in ("source", "operator", "cache"))
        for p in (self.source, self.operator, self.cache):
            p.mkdir()
        (self.source / "empty.py").write_bytes(b"")
        (self.source / "test.py").write_text("pass\n")
        manifest = {"schema": "tensorfold.reviewed-test-only-input-fixture-v1", "base_commit": "c" * 40,
                    "base_app_image": "sha256:" + "a" * 64, "snapshot_clean_commit": False,
                    "source_host": str(self.source), "archive_sha256": "d" * 64,
                    "files": {p.name: {"type": "file", "bytes": p.stat().st_size,
                              "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "stored_mode": p.stat().st_mode & 0o7777}
                              for p in self.source.iterdir()},
                    "reviewed_test_hashes": {"test.py": hashlib.sha256((self.source / "test.py").read_bytes()).hexdigest()}}
        inventory = self.operator / "source.json"
        inventory.write_text(json.dumps(manifest))
        driver = self.operator / "v100_native_contract_driver.py"
        driver.write_text("# immutable test driver DATA\n")
        inner_spec = self.operator / "spec.json"
        inner_spec.write_text(json.dumps({"schema": "tensorfold.v100-native-contract.v1", "mode": "cuda",
                                          "UUIDs": U[:], "SM": [7, 0], "image_id": "sha256:" + "a" * 64,
                                          "source_commit": "c" * 40, "modules": ["tests/test_contract.py"]}))
        inner_identity = file_record(inner_spec)
        self.spec = {"schema": host.SCHEMA, "name": "native-test", "image_id": "sha256:" + "a" * 64,
                     "uuids": U[:], "source": {"root": str(self.source), "manifest": file_record(inventory)},
                     "inputs": [file_record(driver), inner_identity], "mounts": [
                         {"source": str(p), "target": t, "read_only": ro} for p, t, ro in
                         ((self.source, "/canonical", True), (self.operator, "/operator", True),
                          (self.output, "/evidence", False), (self.cache, "/cache", False))],
                     "environment": {"NVIDIA_VISIBLE_DEVICES": ",".join(U), "CUDA_VISIBLE_DEVICES": ",".join(U),
                                     "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"},
                     "entrypoint": "/opt/venv/bin/python", "argv": ["-B", "/operator/v100_native_contract_driver.py", "--spec", "/operator/spec.json",
                                                              "--spec-sha256", inner_identity["sha256"], "--spec-bytes", str(inner_identity["bytes"]),
                                                              "--source", "/canonical", "--evidence", "/evidence"],
                     "resources": {"nano_cpus": 24 * 10**9, "memory_bytes": 256 * 2**30, "memory_swap_bytes": 256 * 2**30,
                                   "ipc": "host", "tmpfs": {"/tmp": "rw,exec,nosuid,nodev,size=8g,mode=1777"},
                                   "ulimits": [{"Name": "memlock", "Soft": -1, "Hard": -1}]},
                     "health_baseline": {u: [i, 0, 0, 0, 0, "No"] for i, u in enumerate(U)},
                     "timeouts": {"run": 30, "poll": .01, "monitor": 1, "retire": 1},
                     "disk_bytes": {"start": 0, "running": 0},
                     "diagnostics": {"sample_limit": 128, "process_limit": 16, "read_limit_bytes": 4096}}

    def test_multi_uuid_and_cpu_create_admission(self):
        for cpu in (False, True):
            spec = copy.deepcopy(self.spec)
            if cpu:
                spec["uuids"] = []
                spec["environment"].update(NVIDIA_VISIBLE_DEVICES="void", CUDA_VISIBLE_DEVICES="")
            host.validate_spec(spec)
            argv = host.create_argv(spec, self.output)
            self.assertEqual("--gpus" in argv, not cpu)
            self.assertEqual(argv[argv.index("--runtime") + 1], "runc" if cpu else "nvidia")
            host.admit(document(spec), spec, CID)
            host.suite_result(inner("cpu" if cpu else "cuda"), "cpu" if cpu else "cuda")

    def test_cpu_inner_route_is_bound_before_start_and_not_native_qualification(self):
        spec = copy.deepcopy(self.spec)
        spec["uuids"] = []
        spec["environment"].update(NVIDIA_VISIBLE_DEVICES="void", CUDA_VISIBLE_DEVICES="")
        path = self.operator / "spec.json"
        data = json.loads(path.read_text())
        data.update(mode="cpu", UUIDs=[], SM=[])
        path.write_text(json.dumps(data))
        record = file_record(path)
        spec["inputs"][1] = record
        spec["argv"][5], spec["argv"][7] = record["sha256"], str(record["bytes"])
        result = host.run(spec, self.output, FakeSystem(spec, self.output))
        self.assertEqual(result["inner"]["mode"], "cpu")
        created = json.loads((self.output / "created.json").read_text())
        self.assertEqual(created["HostConfig"]["DeviceRequests"], [])
        self.assertEqual(created["HostConfig"]["Runtime"], "runc")

    def test_cpu_host_cannot_start_a_cuda_inner_or_stale_spec(self):
        spec = copy.deepcopy(self.spec)
        spec["uuids"] = []
        spec["environment"].update(NVIDIA_VISIBLE_DEVICES="void", CUDA_VISIBLE_DEVICES="")
        system = FakeSystem(spec, self.output)
        with self.assertRaisesRegex(ValueError, "host/inner"):
            host.run(spec, self.output, system)
        self.assertFalse(self.output.exists())
        self.assertFalse(system.calls)
        self.assertFalse(system.started)

    def test_rejects_create_mismatches_before_start(self):
        mutations = [lambda d: d.update(Image="sha256:" + "f" * 64),
                     lambda d: d["HostConfig"]["DeviceRequests"][0].update(DeviceIDs=U[:1]),
                     lambda d: d["Mounts"][0].update(RW=True),
                     lambda d: d["Config"].update(Cmd=["wrong"]),
                     lambda d: d["HostConfig"].update(Memory=1),
                     lambda d: d["HostConfig"]["DeviceRequests"][0].update(Count=False),
                     lambda d: d["HostConfig"]["DeviceRequests"][0].update(Options={"unexpected": "override"}),
                     lambda d: d["HostConfig"].update(SecurityOpt=["seccomp=unconfined"])]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                system = FakeSystem(self.spec, self.output)
                system.bad_create = mutation
                with self.assertRaises(ValueError):
                    host.run(self.spec, self.output, system)
                self.assertFalse(system.started)
                self.assertTrue((self.output / "host-result.json").exists())
                import shutil
                shutil.rmtree(self.output)

    def test_invalid_spec_fails_without_effects(self):
        for key, value in (("GPU_UUID", U[0]), ("cases", "old49 narrative"), ("uuids", [U[0], U[0]]),
                           ("timeouts", {"run": float("nan"), "poll": 1, "monitor": 1, "retire": 1})):
            with self.subTest(key=key):
                spec = copy.deepcopy(self.spec)
                spec[key] = value
                system = FakeSystem(spec, self.output)
                with self.assertRaises(ValueError):
                    host.run(spec, self.output, system)
                self.assertFalse(self.output.exists())
                self.assertFalse(system.calls)

    def test_terminal_before_pid_binding_is_not_a_false_live_pid(self):
        system = FakeSystem(self.spec, self.output)
        result = host.run(self.spec, self.output, system)
        self.assertTrue(result["startup"]["terminal_before_binding"])
        self.assertEqual(result["container"]["State"]["Pid"], 0)
        self.assertTrue(result["source_match"])

    def test_startup_departure_race_is_rechecked_against_terminal_container(self):
        system = FakeSystem(self.spec, self.output, [state(True), state()])
        system.bind_error = host.ProcessDeparted("init departed")
        result = host.run(self.spec, self.output, system)
        self.assertTrue(result["startup"]["terminal_before_binding"])

    def test_running_observations_remain_bounded_data(self):
        system = FakeSystem(self.spec, self.output, [state(True), state()])
        with patch.object(host.time, "sleep"):
            result = host.run(self.spec, self.output, system)
        self.assertEqual(len(result["samples"]), 1)
        self.assertTrue((self.output / result["samples"][0]["file"]).is_file())

    def test_generation_channel_cancel_before_start_never_spawns(self):
        system = FakeSystem(self.spec, self.output)
        system.cancel_at_create = True
        with self.assertRaises(InterruptedError):
            host.run(self.spec, self.output, system)
        self.assertFalse(system.started)
        self.assertIsNotNone(json.loads((self.output / "host-result.json").read_text())["primary_error"])

    def test_cancel_is_idempotent_and_stale_identity_cannot_touch_new_generation(self):
        self.output.mkdir(mode=0o700)
        token, newer = "1" * 64, "2" * 64
        identity = self.output / "controller-identity.json"
        identity.write_text(json.dumps({"schema": "tensorfold.native-contract-control-v1", "run_token": token}))
        self.assertFalse(host.request_cancel(self.output, token)["already_requested"])
        self.assertTrue(host.request_cancel(self.output, token)["already_requested"])
        identity.write_text(json.dumps({"schema": "tensorfold.native-contract-control-v1", "run_token": newer}))
        with self.assertRaisesRegex(ValueError, "stale/reused"):
            host.request_cancel(self.output, token)
        self.assertFalse((self.output / ("cancel-" + newer + ".json")).exists())
        with host._io().owned_descriptor(self.output, os.O_RDONLY | os.O_DIRECTORY) as fd:
            self.assertFalse(host.cancellation(fd, newer))

    def test_authentic_primary_and_all_cleanup_errors_are_retained(self):
        system = FakeSystem(self.spec, self.output, [state(True), state()])
        primary, after, close = RuntimeError("foreign PID"), OSError("source after"), KeyboardInterrupt("close")
        system.mid_error, system.after_error, system.close_error = primary, after, close
        with self.assertRaises(RuntimeError) as caught:
            host.run(self.spec, self.output, system)
        self.assertIs(caught.exception, primary)
        found, pending = set(), [primary]
        while pending:
            error = pending.pop()
            if error is None or id(error) in found:
                continue
            found.add(id(error))
            pending += [error.__cause__, error.__context__]
            if isinstance(error, BaseExceptionGroup):
                pending += list(error.exceptions)
        self.assertTrue({id(primary), id(after), id(close)} <= found)
        result = json.loads((self.output / "host-result.json").read_text())
        self.assertFalse(result["source_match"])
        self.assertTrue(any("source after" in x for x in result["cleanup_errors"]))

    def test_suite_result_rejects_mode_skips_bool_and_diagnostic_failures(self):
        for key, value in (("mode", "cpu"), ("context_fatal", True), ("first_failure_observed", True), ("post_errors", ["bad"]),
                           ("source_after_match", False), ("diagnostic_errors", ["failed observation"])):
            data = inner()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                host.suite_result(data, "cuda")
        for field, value in (("skipped", "1"), ("tests", False), ("tests", "0")):
            data = inner()
            data["JUnit"][field] = value
            with self.assertRaises(ValueError):
                host.suite_result(data, "cuda")

    def test_source_snapshot_rejects_mutation_and_allows_empty_file(self):
        system = object.__new__(host.System)
        system.spec_identity = None
        system.command = lambda *_: self.spec["image_id"]
        system.snapshot(self.spec)
        (self.source / "test.py").write_text("changed\n")
        with self.assertRaises(ValueError):
            system.snapshot(self.spec)

    def test_gpu_health_requires_exact_uuid_membership_and_owned_cgroup(self):
        system = object.__new__(host.System)
        gpu = "\n".join(f"{i}, {u}, 0, 0, 0, 0, No" for i, u in enumerate(U))
        processes = ", 9, python, 1MiB\n"
        owned = "/system.slice/docker-" + CID + ".scope"
        answers = iter((gpu, U[1] + processes))
        system.command = lambda *_: next(answers)

        @contextlib.contextmanager
        def fd(_pid):
            yield 99

        with patch.object(host, "_pidfd", fd), patch.object(host.select, "select", return_value=([], [], [])), \
             patch.object(host, "read_kernel", return_value=("0::" + owned + "/child\n").encode()):
            system.health(self.spec, owned)
            answers = iter((gpu, U[1] + processes))
            with patch.object(host, "read_kernel", return_value=b"0::/foreign\n"), self.assertRaisesRegex(ValueError, "foreign"):
                system.health(self.spec, owned)
        answers = iter((gpu, "GPU-00000000-0000-0000-0000-000000000003" + processes))
        with patch.object(host, "_pidfd", side_effect=AssertionError("foreign GPU must reject before PID access")), \
             self.assertRaisesRegex(ValueError, "UUID"):
            system.health(self.spec, owned)

    def test_help_import_and_spawn_name_are_inert_without_torch(self):
        script = """import runpy,sys,subprocess,pathlib
subprocess.Popen=lambda *a,**k: (_ for _ in ()).throw(AssertionError('spawn'))
pathlib.Path.mkdir=lambda *a,**k: (_ for _ in ()).throw(AssertionError('mkdir'))
runpy.run_module('tools.v100_native_contract_host',run_name='__mp_main__')
assert 'torch' not in sys.modules
"""
        p = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        p = subprocess.run([sys.executable, "-B", "-m", "tools.v100_native_contract_host", "--help"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--cancel-token", p.stdout)

    def test_inline_code_or_unbound_inner_spec_refuses_before_effects(self):
        for change in (lambda s: s.update(argv=["-c", "raise SystemExit(0)"]),
                       lambda s: s["argv"].__setitem__(5, "f" * 64)):
            spec = copy.deepcopy(self.spec)
            change(spec)
            with self.assertRaises(ValueError):
                host.run(spec, self.output, FakeSystem(spec, self.output))
            self.assertFalse(self.output.exists())

    def test_created_container_must_not_already_be_running(self):
        system = FakeSystem(self.spec, self.output)
        system.bad_create = lambda d: d.update(State=state(True))
        # Retirement sees a terminal state after the deliberate initial mismatch.
        calls = 0
        original = system.inspect
        def inspect(cid):
            nonlocal calls
            calls += 1
            return original(cid) if calls == 1 else document(self.spec, state())
        system.inspect = inspect
        with self.assertRaisesRegex(ValueError, "before admission"):
            host.run(self.spec, self.output, system)
        self.assertFalse(system.started)

    def test_atomic_cancel_preserves_publication_and_cleanup_errors(self):
        self.output.mkdir(mode=0o700)
        token = "1" * 64
        (self.output / "controller-identity.json").write_text(json.dumps({
            "schema": "tensorfold.native-contract-control-v1", "run_token": token}))
        primary, cleanup = OSError("cancel publication"), OSError("temporary cleanup")
        with patch.object(host.os, "link", side_effect=primary), patch.object(host.os, "unlink", side_effect=cleanup):
            with self.assertRaises(OSError) as caught:
                host.request_cancel(self.output, token)
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cleanup)
        self.assertFalse((self.output / ("cancel-" + token + ".json")).exists())

    def test_persistence_failure_keeps_existing_primary_and_emits_fallback(self):
        import io
        system = FakeSystem(self.spec, self.output, [state(True), state()])
        primary, persistence = RuntimeError("GPU ownership"), OSError("result fsync")
        system.mid_error = primary
        publish = host.write_json
        def write(path, value):
            if path.name == "host-result.json":
                raise persistence
            publish(path, value)
        with patch.object(host, "write_json", write), patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(RuntimeError) as caught:
                host.run(self.spec, self.output, system)
        self.assertIs(caught.exception, primary)
        self.assertIn("NATIVE_HOST_OUTCOME_UNPUBLISHED", stderr.getvalue())
        self.assertIs(primary.__cause__, persistence)

    def test_wait_is_attempted_even_after_owned_attach_terminate_raises(self):
        system = FakeSystem(self.spec, self.output)
        primary = subprocess.TimeoutExpired("attach", 1)
        second, terminate = subprocess.TimeoutExpired("attach", 1), OSError("terminate")
        waits = iter((primary, second, 0))
        def wait(timeout):
            system.process.actions.append("wait")
            result = next(waits)
            if isinstance(result, BaseException):
                raise result
            system.process.returncode = result
            return result
        def stop():
            system.process.actions.append("terminate")
            raise terminate
        system.process.wait, system.process.terminate = wait, stop
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            host.run(self.spec, self.output, system)
        self.assertIs(caught.exception, primary)
        self.assertEqual(system.process.actions, ["wait", "wait", "terminate", "wait"])
        self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
        self.assertIn(terminate, primary.__cause__.exceptions)

    def test_control_reads_handle_short_reads_without_accepting_extra_data(self):
        self.output.mkdir(mode=0o700)
        token = "1" * 64
        identity = self.output / "controller-identity.json"
        identity.write_text(json.dumps({"schema": "tensorfold.native-contract-control-v1", "run_token": token}))
        original = host.os.read
        with patch.object(host.os, "read", side_effect=lambda fd, n: original(fd, min(n, 3))):
            self.assertTrue(host.request_cancel(self.output, token)["requested"])
        identity.write_text(identity.read_text() + "{}")
        with self.assertRaises(ValueError):
            host.request_cancel(self.output, token)

    def test_real_backend_rejects_wrong_host_owner_before_effects(self):
        with patch.object(host.os, "getuid", return_value=0), self.assertRaisesRegex(ValueError, "UID1000"):
            host.System(self.output)
        self.assertFalse(self.output.exists())

    def test_pytest_options_cannot_narrow_modules_even_when_inherited(self):
        spec = copy.deepcopy(self.spec)
        spec["environment"]["PYTEST_ADDOPTS"] = "-k narrowed"
        with self.assertRaisesRegex(ValueError, "pytest options"):
            host.validate_spec(spec)
        doc = document(self.spec)
        doc["Config"]["Env"].append("PYTEST_ADDOPTS=-s")
        with self.assertRaisesRegex(ValueError, "pytest options"):
            host.admit(doc, self.spec, CID)

    def test_reviewed_fixture_identity_never_becomes_clean_commit(self):
        files, provenance = host.source_inventory(self.spec)
        self.assertFalse(provenance["snapshot_clean_commit"])
        self.assertEqual(files["test.py"]["type"], "file")
        manifest = Path(self.spec["source"]["manifest"]["path"])
        raw = json.loads(manifest.read_text())
        raw["snapshot_clean_commit"] = True
        manifest.write_text(json.dumps(raw))
        self.spec["source"]["manifest"] = file_record(manifest)
        with self.assertRaisesRegex(ValueError, "not a clean commit"):
            host.validate_data(self.spec)

    def test_unrenderable_primary_keeps_identity_and_durable_incomplete_report(self):
        import io
        render = ValueError("primary repr failed")
        class Unrenderable(RuntimeError):
            def __repr__(self):
                raise render
        primary, cleanup = Unrenderable("actual primary"), OSError("cleanup")
        system = FakeSystem(self.spec, self.output, [state(True), state()])
        system.mid_error, system.close_error = primary, cleanup
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(Unrenderable) as caught:
                host.run(self.spec, self.output, system)
        self.assertIs(caught.exception, primary)
        self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
        self.assertIn(render, primary.__cause__.exceptions)
        self.assertIn(cleanup, primary.__cause__.exceptions)
        record = json.loads((self.output / "host-result-incomplete.json").read_text())
        self.assertFalse(record["report_complete"])
        self.assertEqual(record["primary_type"], "Unrenderable")
        self.assertIn("NATIVE_HOST_OUTCOME_UNPUBLISHED", stderr.getvalue())

    def test_unrenderable_cleanup_and_publication_failures_do_not_mask_primary(self):
        import io
        render, publication = ValueError("cleanup repr failed"), OSError("no evidence write")
        class Unrenderable(OSError):
            def __repr__(self):
                raise render
        primary, cleanup = RuntimeError("actual primary"), Unrenderable("cleanup")
        system = FakeSystem(self.spec, self.output, [state(True), state()])
        system.mid_error, system.close_error = primary, cleanup
        original = host.write_json
        def publish(path, value):
            if path.name.startswith("host-result"):
                raise publication
            original(path, value)
        with patch.object(host, "write_json", publish), patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(RuntimeError) as caught:
                host.run(self.spec, self.output, system)
        self.assertIs(caught.exception, primary)
        self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
        self.assertTrue(all(e in primary.__cause__.exceptions for e in (cleanup, render, publication)))
        self.assertIn("Unrenderable", stderr.getvalue())
        self.assertNotIn("cleanup repr failed", stderr.getvalue())  # fallback does not call original repr

    def test_large_raw_asset_is_streamed_without_materializing_a_payload(self):
        import tracemalloc
        asset = self.operator / "tokenizer.json"
        # Same extent as the actual approved frontend; synthetic opaque bytes,
        # not a claim that this fixture is a functional tokenizer.
        length, checksum = 20217442, hashlib.sha256()
        with asset.open("wb") as stream:
            remaining = length
            while remaining:
                block = b"x" * min(65536, remaining)
                stream.write(block)
                checksum.update(block)
                remaining -= len(block)
        item = {"path": str(asset), "bytes": length, "sha256": checksum.hexdigest()}
        self.spec["inputs"].append(item)
        host.validate_spec(self.spec)
        host._io()  # Warm library imports; measure verifier-owned Python buffers.
        original, sizes = host.os.read, []
        def read(fd, count):
            sizes.append(count)
            return original(fd, count)
        tracing = tracemalloc.is_tracing()
        if not tracing:
            tracemalloc.start()
        tracemalloc.reset_peak()
        baseline = tracemalloc.get_traced_memory()[0]
        try:
            with patch.object(host.os, "read", read):
                verified = host.check_raw_file(item)
            peak = tracemalloc.get_traced_memory()[1] - baseline
        finally:
            if not tracing:
                tracemalloc.stop()
        self.assertIs(verified, item)  # Identity metadata only; no raw-byte result.
        self.assertTrue(sizes and max(sizes) <= 65536)
        # Sensitivity control for a forbidden 20MiB joined/list payload. This is
        # a Python allocation check, NOT an RSS/native-allocator admission bound.
        self.assertLess(peak, 1 << 20)
        structured = host.bound_file
        def no_raw_materialization(record):
            if record["path"] == str(asset):
                self.fail("raw asset used structured materializing reader")
            return structured(record)
        with patch.object(host, "bound_file", no_raw_materialization):
            host.validate_data(self.spec)
            system = object.__new__(host.System)
            system.spec_identity = None
            system.command = lambda *_: self.spec["image_id"]
            system.snapshot(self.spec)

    def test_structured_json_limit_is_unchanged_and_refuses_before_read(self):
        item = {"path": str(self.operator / "too-large.json"), "bytes": host.LIMIT + 1, "sha256": "a" * 64}
        with patch.object(host, "read_bytes", side_effect=AssertionError("oversized structured data must not be read")):
            with self.assertRaisesRegex(ValueError, "structured JSON"):
                host.bound_file(item)

    def test_raw_wrong_extent_or_digest_refuses(self):
        asset = self.operator / "raw.dat"
        asset.write_bytes(b"raw immutable bytes")
        original = file_record(asset)
        for changed in ({**original, "bytes": original["bytes"] - 1},
                        {**original, "bytes": original["bytes"] + 1},
                        {**original, "sha256": "0" * 64}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                host.check_raw_file(changed)
        host.check_raw_file(original)
        empty = self.operator / "empty.dat"
        empty.write_bytes(b"")
        host.check_raw_file(file_record(empty))

    def test_raw_mid_read_mutation_truncation_extension_and_path_replacement_refuse(self):
        original_read = host.os.read
        for mode in ("rewrite", "truncate", "extend", "replace"):
            with self.subTest(mode=mode):
                asset = self.operator / (mode + ".dat")
                contents = b"q" * (2 * 65536 + 7)
                asset.write_bytes(contents)
                item, initial = file_record(asset), asset.stat()
                changed = False
                def read(fd, count):
                    nonlocal changed
                    value = original_read(fd, count)
                    if not changed:
                        changed = True
                        if mode == "truncate":
                            os.truncate(asset, 0)
                        elif mode == "extend":
                            with asset.open("ab") as stream:
                                stream.write(b"more")
                        elif mode == "replace":
                            replacement = self.operator / "replacement.dat"
                            replacement.write_bytes(contents)
                            os.replace(replacement, asset)
                        else:
                            with asset.open("r+b") as stream:
                                stream.write(b"z")  # Already consumed byte: stamps must reject it.
                            os.utime(asset, ns=(initial.st_atime_ns, initial.st_mtime_ns + 2000000000))
                    return value
                with patch.object(host.os, "read", read), self.assertRaises(ValueError):
                    host.check_raw_file(item)
                self.assertTrue(changed)

    def test_raw_symlink_refuses_without_following_target(self):
        actual = self.operator / "actual.dat"
        actual.write_bytes(b"data")
        alias = self.operator / "alias.dat"
        alias.symlink_to(actual)
        record = {**file_record(actual), "path": str(alias)}
        with self.assertRaises(OSError):
            host.check_raw_file(record)

    @unittest.skipUnless(hasattr(os, "pidfd_open"), "Linux owned-child pidfd required")
    def test_real_owned_child_pidfd_proves_exit_without_numeric_pid_signal(self):
        child = subprocess.Popen([sys.executable, "-B", "-c", "import time;time.sleep(30)"])
        try:
            with host._pidfd(child.pid) as fd:
                self.assertFalse(select.select([fd], [], [], 0)[0])
                child.terminate()  # Owned unreaped Popen child, not a prechecked borrowed PID.
                child.wait(timeout=5)
                self.assertTrue(select.select([fd], [], [], 0)[0])
            with self.assertRaises(OSError):
                os.fstat(fd)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
