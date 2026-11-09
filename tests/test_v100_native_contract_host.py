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
    value = {"Id": CID, "Name": "/" + spec["name"], "Image": spec["image_id"],
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
    if "verification_launcher" in spec:
        value["HostConfig"].update(CapAdd=["CAP_CHECKPOINT_RESTORE"], CapDrop=["ALL"])
        value["NetworkSettings"] = {"Networks": {"none": {}}}
    return value


def verification_data(spec):
    contract = spec["verification_launcher"]
    expected = host.validate_data(spec)
    return {"schema": "tensorfold.verification-launcher-preflight-v1", "mode": host.VERIFICATION_MODE,
            "image_id": spec["image_id"], "source_commit": expected["source_commit"],
            "verification_launcher": copy.deepcopy(contract), "PID": 1, "uids": [1000]*4, "gids": [1000]*4,
            "caps": {name: host.VERIFICATION_CAP for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")},
            "NoNewPrivs": "0", "loader_environment": {"LD_LIBRARY_PATH": None},
            "launcher": {**{k: contract["launcher"][k] for k in ("path", "bytes", "sha256", "file_capability_hex")},
                         "mode": 0o555, "uid": 0, "source": {"path": host.VERIFICATION_SOURCE,
                         "bytes": contract["launcher"]["source_bytes"], "sha256": contract["launcher"]["source_sha256"],
                         "mode": 0o444, "uid": 0}},
            "python": {**contract["python"], "resolved_path": "/usr/bin/python3.12", "mode": 0o755, "uid": 0},
            "runtime": expected["runtime"], "providers": {"schema": "tensorfold.native-elf-observation-v3",
                "identity_method": "kernel-vma-backed-fd-v1", "PID": 1, "missing_required_provider_paths": [],
                "required_providers": contract["providers"], "mapped_elfs": [
                    {"path": path, **record} for path, record in contract["providers"].items()]}}


def status_text():
    return "Uid:\t1000\t1000\t1000\t1000\nGid:\t1000\t1000\t1000\t1000\nNoNewPrivs:\t0\n"+"".join(
        name+":\t"+host.VERIFICATION_CAP+"\n" for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"))


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
        self.write_verification, self.bad_verification, self.verification_error = True, None, None

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
        result = inner("cuda" if self.spec["uuids"] else "cpu")
        if "verification_launcher" in self.spec:
            result["image_id_declared"] = self.spec["image_id"]
            if self.write_verification:
                receipt = verification_data(self.spec)
                if self.bad_verification:
                    self.bad_verification(receipt)
                (self.output / "verification-launcher-preflight.json").write_text(json.dumps(receipt))
        (self.output / "named-suite-result.json").write_text(json.dumps(result))
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

    def verification_process(self, _cid, _spec):
        if self.verification_error:
            raise self.verification_error
        return {"host_PID": 42, "status": host.verification_status(status_text())}

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

    def verification_spec(self):
        spec = copy.deepcopy(self.spec)
        spec["image_id"] = "sha256:"+"f"*64
        spec["entrypoint"] = host.VERIFICATION_ENTRY
        spec["verification_launcher"] = {"mode": host.VERIFICATION_MODE, "base_image_id": self.spec["image_id"],
            "launcher": {"path": host.VERIFICATION_ENTRY, "bytes": 16976, "sha256": "1"*64,
                         "source_bytes": 8000, "source_sha256": "2"*64, "file_capability_hex": host.VERIFICATION_XATTR},
            "python": {"path": host.VERIFICATION_PYTHON, "bytes": 8020928, "sha256": "3"*64},
            "providers": {"/opt/venv/lib/python3.12/site-packages/torch/_C.cpython-312-x86_64-linux-gnu.so": {"bytes": 1024, "sha256": "4"*64},
                          "/opt/venv/lib/python3.12/site-packages/tensorfold/_fd_owner.abi3.so": {"bytes": 37936, "sha256": "7"*64}}}
        path = self.operator/"spec.json"
        data = json.loads(path.read_text())
        data.update(image_id=spec["image_id"], verification_launcher=copy.deepcopy(spec["verification_launcher"]),
                    app_root="/opt/venv/lib/python3.12/site-packages/tensorfold",
                    expected_app={"__init__.py": {"bytes": 10, "sha256": "5"*64, "mode": 0o644}},
                    sdk_manifest={"path": "/opt/tensorfold-pins/v100-source-foundation.json", "sha256": "6"*64},
                    fd={"sha256": "7"*64, "source_sha256": "8"*64},
                    torch={"git_version": "9"*40, "cuda": "12.9"})
        path.write_text(json.dumps(data))
        record = file_record(path)
        spec["inputs"][1] = record
        spec["argv"][5], spec["argv"][7] = record["sha256"], str(record["bytes"])
        return spec

    def test_regular_json_and_cap0_route_remain_unchanged(self):
        original = copy.deepcopy(self.spec)
        self.assertIs(host.validate_spec(self.spec), self.spec)
        self.assertEqual(self.spec, original)
        self.assertNotIn("--cap-add", host.create_argv(self.spec, self.output))
        self.assertNotIn("--cap-drop", host.create_argv(self.spec, self.output))
        host.admit(document(self.spec), self.spec, CID)

    def test_explicit_verification_create_and_actual_admission(self):
        spec = self.verification_spec()
        host.validate_spec(spec)
        self.assertEqual(host.source_inventory(spec)[1]["base_app_image"], self.spec["image_id"])
        self.assertEqual(host.validate_data(spec)["image_id"], spec["image_id"])
        argv = host.create_argv(spec, self.output)
        self.assertEqual(argv[argv.index("--cap-add")+1], "CHECKPOINT_RESTORE")
        self.assertEqual(argv[argv.index("--cap-drop")+1], "ALL")
        host.admit(document(spec), spec, CID)

    def test_verification_schema_refuses_unknown_injected_or_unowned_inputs(self):
        spec = self.verification_spec()
        changes = [lambda d: d.update(verification_launcher=None),
                   lambda d: d["verification_launcher"].update(mode="service"),
                   lambda d: d["verification_launcher"]["launcher"].update(file_capability_hex="00"),
                   lambda d: d["verification_launcher"]["python"].update(path="/usr/bin/python3"),
                   lambda d: d["verification_launcher"].update(providers={}),
                   lambda d: d.update(entrypoint=host.VERIFICATION_PYTHON),
                   lambda d: d["environment"].update(PYTHONPATH=""),
                   lambda d: d["environment"].update(LD_PRELOAD="/tmp/evil.so"),
                   lambda d: d["environment"].update(PYTHONOPTIMIZE="1")]
        for change in changes:
            bad = copy.deepcopy(spec)
            change(bad)
            with self.subTest(change=change), self.assertRaises(ValueError):
                host.validate_spec(bad)
        for target in ("/proc", "/proc/self/status", "/opt/qualification", "/usr/bin/python3.12", "/tmp"):
            bad = copy.deepcopy(spec)
            bad["mounts"].append({"source": str(self.cache), "target": target, "read_only": True})
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "shadows"):
                host.validate_spec(bad)

    def test_verification_inspect_refuses_extra_privilege_or_changed_isolation(self):
        spec = self.verification_spec()
        changes = [lambda d: d["HostConfig"].update(CapAdd=["CAP_SYS_ADMIN"]),
                   lambda d: d["HostConfig"].update(CapAdd=["CAP_CHECKPOINT_RESTORE", "CAP_SYS_ADMIN"]),
                   lambda d: d["HostConfig"].update(CapDrop=[]),
                   lambda d: d["HostConfig"].update(Privileged=True),
                   lambda d: d["HostConfig"].update(PidMode="host"),
                   lambda d: d["HostConfig"].update(ReadonlyRootfs=False),
                   lambda d: d["Config"].update(User="0:0"),
                   lambda d: d["HostConfig"].update(GroupAdd=["0"]),
                   lambda d: d["HostConfig"].update(Init=True),
                   lambda d: d["NetworkSettings"]["Networks"].update(bridge={}),
                   lambda d: d["Config"]["Env"].append("PYTHONPATH=/unmanaged")]
        for change in changes:
            doc = document(spec)
            change(doc)
            with self.subTest(change=change), self.assertRaises(ValueError):
                host.admit(doc, spec, CID)

    def test_verification_operator_closure_is_frozen_bound_and_has_no_unmanaged_python(self):
        spec = self.verification_spec()
        for path in self.operator.iterdir():
            path.chmod(0o444)
        self.operator.chmod(0o555)
        try:
            self.assertEqual(set(host.verification_operator(spec)), {"source.json", "spec.json", "v100_native_contract_driver.py"})
            self.operator.chmod(0o755)
            (self.operator/"json.py").write_text("raise RuntimeError('unmanaged')")
            (self.operator/"json.py").chmod(0o444)
            self.operator.chmod(0o555)
            with self.assertRaisesRegex(ValueError, "unmanaged"):
                host.verification_operator(spec)
        finally:
            self.operator.chmod(0o755)
            for path in self.operator.iterdir():
                path.chmod(0o644)

    def test_verification_lineage_is_exact_base_derivative(self):
        spec = self.verification_spec()
        base = {"Id": spec["verification_launcher"]["base_image_id"], "RootFS": {"Layers": ["sha256:"+"1"*64]}}
        candidate = {"Id": spec["image_id"], "RootFS": {"Layers": [*base["RootFS"]["Layers"], "sha256:"+"2"*64]}}
        self.assertEqual(host.verification_image(candidate, base, spec)["base_image_id"], base["Id"])
        candidate["RootFS"]["Layers"][0] = "sha256:"+"3"*64
        with self.assertRaises(ValueError):
            host.verification_image(candidate, base, spec)

    def test_real_snapshot_binds_verification_lineage_and_operator_without_rewriting_fixture(self):
        spec = self.verification_spec()
        system = object.__new__(host.System)
        system.spec_identity = None
        layers = ["sha256:"+"1"*64]
        def command(argv, *_args):
            if argv[argv.index("--format")+1] == "{{.Id}}":
                return spec["image_id"]
            image = argv[-1]
            return json.dumps({"Id": image, "RootFS": {"Layers": layers+(["sha256:"+"2"*64] if image == spec["image_id"] else [])}})
        system.command = command
        for path in self.operator.iterdir():
            path.chmod(0o444)
        self.operator.chmod(0o555)
        try:
            snapshot = system.snapshot(spec)
            self.assertEqual(snapshot["source_fixture"]["base_app_image"], self.spec["image_id"])
            self.assertEqual(snapshot["verification_image"]["base_layers"], layers)
            self.assertIn("v100_native_contract_driver.py", snapshot["verification_operator"])
        finally:
            self.operator.chmod(0o755)
            for path in self.operator.iterdir():
                path.chmod(0o644)

    def test_prequalification_receipt_binds_all_declared_identity_and_provider_facts(self):
        spec = self.verification_spec()
        expected = host.validate_data(spec)
        raw = verification_data(spec)
        host.verification_receipt(raw, spec, expected)
        changes = [lambda r: r.update(PID=True), lambda r: r.update(uids=[0]*4),
                   lambda r: r["caps"].update(CapBnd="0"*16),
                   lambda r: r.update(loader_environment={"LD_LIBRARY_PATH": ""}),
                   lambda r: r["launcher"]["source"].update(sha256="f"*64),
                   lambda r: r["python"].update(resolved_path="/unmanaged/python"),
                   lambda r: r["runtime"].update(sdk_sha256="f"*64),
                   lambda r: r["providers"].update(missing_required_provider_paths=["missing"]),
                   lambda r: r["providers"]["mapped_elfs"][0].update(sha256="f"*64)]
        for change in changes:
            bad = copy.deepcopy(raw)
            change(bad)
            with self.subTest(change=change), self.assertRaises(ValueError):
                host.verification_receipt(bad, spec, expected)

    def test_actual_proc_status_requires_uid_gid_only_cap40_and_no_new_privileges_zero(self):
        host.verification_status(status_text())
        for bad in (status_text().replace("CapEff:", "unexpected:"),
                    status_text().replace(host.VERIFICATION_CAP, "0"*16),
                    status_text()+"Uid: 1000 1000 1000 1000\n",
                    status_text().replace("NoNewPrivs:\t0", "NoNewPrivs:\t1")):
            with self.assertRaises(ValueError):
                host.verification_status(bad)

    def test_live_verification_status_stays_bound_to_original_pidfd_and_cgroup(self):
        spec = self.verification_spec()
        system = object.__new__(host.System)
        system.pidfd, system.init_pid = 99, 42
        system.group = "/system.slice/docker-"+CID+".scope"
        system.inspect = lambda _cid: document(spec, state(True))
        answers = [status_text().encode(), ("0::"+system.group+"\n").encode()]
        with patch.object(host.select, "select", return_value=([], [], [])), patch.object(host, "read_kernel", side_effect=answers):
            self.assertEqual(system.verification_process(CID, spec)["host_PID"], 42)
        with patch.object(host.select, "select", return_value=([], [], [])), patch.object(host, "read_kernel", side_effect=PermissionError("nondumpable")):
            with self.assertRaises(PermissionError):
                system.verification_process(CID, spec)
        changed = document(spec, state(True))
        changed["State"]["Pid"] = 43
        system.inspect = lambda _cid: changed
        with patch.object(host.select, "select", return_value=([], [], [])):
            with self.assertRaisesRegex(ValueError, "pinned PID"):
                system.verification_process(CID, spec)

    def test_verification_fast_terminal_still_requires_prequal_receipt(self):
        spec = self.verification_spec()
        result = host.run(spec, self.output, FakeSystem(spec, self.output))
        self.assertEqual(result["verification_preflight"]["verification_launcher"], spec["verification_launcher"])
        self.assertEqual(result["container"]["State"]["Pid"], 0)

    def test_missing_prequal_receipt_fails_even_if_pytest_result_passed(self):
        spec = self.verification_spec()
        system = FakeSystem(spec, self.output)
        system.write_verification = False
        with self.assertRaises(FileNotFoundError) as missing:
            host.run(spec, self.output, system)
        self.assertEqual(Path(missing.exception.filename), self.output/"verification-launcher-preflight.json")
        report = json.loads((self.output/"host-result.json").read_text())
        self.assertIn("FileNotFoundError", report["primary_error"])
        self.assertEqual(report["container"]["State"]["Pid"], 0)

    def test_nondumpable_status_denial_is_not_bypassed(self):
        spec = self.verification_spec()
        system = FakeSystem(spec, self.output, [state(True), state()])
        system.verification_error = PermissionError("owned proc status denied")
        with self.assertRaises(PermissionError):
            host.run(spec, self.output, system)
        report = json.loads((self.output/"host-result.json").read_text())
        self.assertIn("owned proc status denied", report["primary_error"])
        self.assertEqual(report["container"]["State"]["Pid"], 0)

    def test_invalid_verification_receipt_preserves_cleanup_failure(self):
        spec = self.verification_spec()
        system = FakeSystem(spec, self.output)
        system.bad_verification = lambda r: r["caps"].update(CapEff="0"*16)
        system.close_error = RuntimeError("owned observer close failed")
        with self.assertRaisesRegex(ValueError, "capabilities") as caught:
            host.run(spec, self.output, system)
        self.assertIs(caught.exception.__cause__, system.close_error)
        report = json.loads((self.output/"host-result.json").read_text())
        self.assertIn("capabilities", report["primary_error"])
        self.assertTrue(any("owned observer close failed" in value for value in report["cleanup_errors"]))
        self.assertEqual(report["container"]["State"]["Pid"], 0)

    def test_verification_receipt_changed_after_prequalification_is_rejected(self):
        spec = self.verification_spec()
        system = FakeSystem(spec, self.output, [state(True), state()])
        def mutate(_cid, _spec):
            path = self.output/"verification-launcher-preflight.json"
            raw = json.loads(path.read_text())
            raw["loader_environment"] = {"LD_LIBRARY_PATH": "/restored"}
            path.write_text(json.dumps(raw))
            return {"phase": None, "compiler_processes": []}
        system.observe = mutate
        with self.assertRaisesRegex(ValueError, "loader environment"):
            host.run(spec, self.output, system)

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


    def model_spec(self):
        c = host.model_contract
        spec = self.verification_spec()
        ids = ['GPU-00000000-0000-0000-0000-' + format(i, '012x') for i in range(8)]
        spec['uuids'] = ids
        spec['environment'].update(CUDA_VISIBLE_DEVICES=','.join(ids), NVIDIA_VISIBLE_DEVICES=','.join(ids), TF_GGUF_ASSETS_DIR='/frontend')
        spec['health_baseline'] = {u: [i, 0, 0, 0, 0, 'No'] for i, u in enumerate(ids)}
        spec['resources'].update(nano_cpus=64*10**9, memory_bytes=340 << 30, memory_swap_bytes=340 << 30)
        spec['timeouts'].update(run=7800, retire=90, poll=1, monitor=120)
        modelroot = '/models/hf/hub/models--fixture/snapshots/' + 'a'*40
        frontend = '/frontend/' + 'b'*64
        def asset(path):
            return {'path': path, 'bytes': 1, 'sha256': '1'*64}
        shards = [asset(modelroot + '/fixture-0000' + str(i) + '-of-00002.gguf') for i in (1, 2)]
        operation = {'teacher_plan': None, 'teacher_client': None, 'model_entrypoint': shards[0]['path'],
                     'model_assets': shards, 'frontend_root': frontend,
                     'frontend_assets': [asset(frontend+'/'+name) for name in sorted(c.FRONTEND_NAMES)], 'execution_manifest': None}
        for name in ('teacher-plan.json', 'teacher-client.json'):
            path = self.operator / name
            path.write_text('{}\n')
            record = file_record(path)
            spec['inputs'].append(record)
            operation['teacher_plan' if 'plan' in name else 'teacher_client'] = {**record, 'path': '/operator/'+name}
        contract_path = self.operator / 'v100_owned_model_contract.py'
        contract_path.write_text('# immutable contract fixture\n')
        spec['inputs'].append(file_record(contract_path))
        sources = {}
        for name in c.REQUIRED_TOOLS:
            source = self.source / name
            source.parent.mkdir(exist_ok=True)
            staged = self.operator / Path(name).name
            source.write_bytes(staged.read_bytes() if staged.exists() else b'# canonical source fixture\n')
            sources[name] = {key: file_record(source)[key] for key in ('bytes', 'sha256')}
        manifest_path = Path(spec['source']['manifest']['path'])
        manifest = json.loads(manifest_path.read_text())
        for name in sources:
            path = self.source/name
            manifest['files'][name] = {'type': 'file', 'stored_mode': path.stat().st_mode & 0o7777, **sources[name]}
        manifest_path.write_text(json.dumps(manifest))
        spec['source']['manifest'] = file_record(manifest_path)
        inner_path = self.operator/'spec.json'
        data = json.loads(inner_path.read_text())
        data.pop('modules')
        data.update(schema=c.SCHEMA, mode=c.MODE, UUIDs=ids, operation=operation, source_files=sources)
        inner_path.write_text(json.dumps(data))
        spec['inputs'][1] = file_record(inner_path)
        spec['argv'][5], spec['argv'][7] = spec['inputs'][1]['sha256'], str(spec['inputs'][1]['bytes'])
        spec['mounts'] += [{'source': str(self.root/'readonly-model'), 'target': modelroot, 'read_only': True},
                           {'source': str(self.root/'readonly-frontend'), 'target': '/frontend', 'read_only': True}]
        return spec, data

    def model_outcome(self, expected):
        c = host.model_contract
        def ref(name, rank=None):
            value = {'file': name, 'bytes': 1, 'sha256': 'a'*64}
            return value if rank is None else {**value, 'rank': rank}
        model = {'schema': c.RESULT_SCHEMA, 'teacher_plan_sha256': expected['operation']['teacher_plan']['sha256'],
                 'teacher_client_sha256': expected['operation']['teacher_client']['sha256'], 'launch_exit': 0,
                 'client_report': ref('teacher/teacher-client-report.json'), 'teacher_authority': ref('teacher/teacher-authority.json'),
                 'rank_receipts': [ref(f'teacher/teacher-rank{i}/summary.json', i) for i in range(8)],
                 'startup_receipts': [ref(f'teacher/rank-{i}.json', i) for i in range(8)],
                 'child_native_records': [ref(f'child-native/teacher-closed-rank-{i}-pid-{i+2}.json', i) for i in range(8)]}
        assets = expected['operation']
        identities = [*assets['model_assets'], *assets['frontend_assets'], assets['teacher_plan'], assets['teacher_client']]
        if 'reference' in assets:
            reference = assets['reference']
            identities += [{'path': reference['root']+'/'+name, **reference['files'][name]} for name in sorted(reference['files'])]
            identities += [reference['host_result'], reference['completion']]
        binding = {'schema': 'tensorfold.owned-model-content-binding-v1', 'completed': True,
                   'assets': [{'identity': item, 'stat': [1, 2, 3, item['bytes'], 4, 5]} for item in identities]}
        record = self.output/'model-content-binding.json'
        record.write_text(json.dumps(binding))
        return {'schema': c.HOST_RESULT_SCHEMA, 'mode': c.MODE, 'source_commit': expected['source_commit'],
                'image_id_declared': expected['image_id'], 'operation': assets, 'source_after_match': True,
                'content_binding_complete': True, 'content_binding': {**{k: file_record(record)[k] for k in ('bytes', 'sha256')}, 'file': record.name},
                'parent_cuda_uninitialized': True, 'model_result': model, 'primary_error': None, 'post_errors': [],
                'scope': 'unit boundary fixture only'}

    def test_model_mode_binds_exact_eight_ro_inputs_and_canonical_operator(self):
        spec, data = self.model_spec()
        host.validate_spec(spec)
        expected = host.validate_data(spec)
        self.assertEqual(expected['operation'], data['operation'])
        self.assertNotIn('modules', expected)
        host.admit(document(spec), spec, CID)
        argv = host.create_argv(spec, self.output)
        self.assertEqual(argv[argv.index('--network')+1], 'none')
        self.assertEqual(argv.count('--cap-add'), 1)
        self.assertEqual(argv[argv.index('--cpus')+1], '64')
        self.assertEqual(argv[argv.index('--memory')+1], str(340 << 30))

    def test_model_mode_refuses_writable_assets_wrong_floors_missing_ranks_and_selection(self):
        spec, data = self.model_spec()
        c = host.model_contract
        for change in ('writable-model', 'low-memory', 'short-deadline', 'four-ranks', 'optimized', 'private-select'):
            bad = copy.deepcopy(spec)
            if change == 'writable-model':
                bad['mounts'][-2]['read_only'] = False
            elif change == 'low-memory':
                bad['resources']['memory_bytes'] -= 1
            elif change == 'short-deadline':
                bad['timeouts']['run'] = 600
            elif change == 'four-ranks':
                bad['uuids'] = bad['uuids'][:4]
            elif change == 'optimized':
                bad['environment']['TENSORFOLD_SM70_PROFILE'] = 'optimized'
            else:
                bad['environment']['TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS'] = 'native-serial-v1'
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.validate_host_operation(bad, data)

    def test_typed_model_result_is_not_a_junit_success_substitute(self):
        spec, _ = self.model_spec()
        expected = host.validate_data(spec)
        self.output.mkdir()
        result = self.model_outcome(expected)
        c = host.model_contract
        self.assertIs(c.validate_host_result(result, expected), result)
        for change in ({'parent_cuda_uninitialized': False}, {'content_binding_complete': False},
                       {'post_errors': [{'error': 'retirement'}]}, {'primary_error': {'error': 'model'}},
                       {'model_result': {**result['model_result'], 'rank_receipts': result['model_result']['rank_receipts'][:-1]}},
                       {'model_result': {**result['model_result'], 'launch_exit': 1}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.validate_host_result({**result, **change}, expected)
        with self.assertRaises(ValueError):
            c.validate_host_result(inner(), expected)

    def test_model_host_lifecycle_requires_typed_result_content_receipt_and_retirement(self):
        spec, _ = self.model_spec()
        system = FakeSystem(spec, self.output)
        original = system.start
        def start(cid, log):
            process = original(cid, log)
            expected = host.validate_data(spec)
            (self.output/'owned-teacher-operation-result.json').write_text(json.dumps(self.model_outcome(expected)))
            return process
        system.start = start
        outcome = host.run(spec, self.output, system=system)
        self.assertEqual(outcome['inner']['mode'], host.model_contract.MODE)
        self.assertEqual(outcome['inner']['model_result']['launch_exit'], 0)
        self.assertEqual(outcome['container']['State']['Pid'], 0)
        self.assertIn(['close'], system.calls)


    def replay_host_spec(self):
        spec, data = self.model_spec()
        names = ['teacher/teacher-authority.json', *[f'teacher/rank-{i}.json' for i in range(8)],
                 *[f'teacher/teacher-rank{i}/summary.json' for i in range(8)],
                 *[f'teacher/teacher-rank0/frame-{i:06d}.npy' for i in range(4)]]
        reference = {'root': '/reference', 'authority': {'path': '/reference/teacher/teacher-authority.json', 'bytes': 1, 'sha256': 'b'*64},
                     'files': {name: {'bytes': 1, 'sha256': 'b'*64} for name in names},
                     'host_result': {'path': '/reference/host-result.json', 'bytes': 1, 'sha256': 'c'*64},
                     'completion': {'path': '/operator/reference-capture-completion.json', 'bytes': 1, 'sha256': 'd'*64}}
        path = self.operator/'reference-capture-completion.json'
        path.write_text('{}')
        record = file_record(path)
        reference['completion'].update({key: record[key] for key in ('bytes', 'sha256')})
        spec['inputs'].append(record)
        data['operation']['reference'] = reference
        spec['mounts'].append({'source': str(self.root/'readonly-reference'), 'target': '/reference', 'read_only': True})
        path = self.operator/'spec.json'
        path.write_text(json.dumps(data))
        record = file_record(path)
        spec['inputs'][1] = record
        spec['argv'][5], spec['argv'][7] = record['sha256'], str(record['bytes'])
        return spec, data

    def test_replay_reference_bind_rejects_writable_or_source_aliases(self):
        spec, data = self.replay_host_spec()
        host.validate_spec(spec)
        host.validate_data(spec)
        for kind in ('writable', 'alias', 'ancestor', 'missing'):
            changed = copy.deepcopy(spec)
            if kind == 'writable':
                changed['mounts'][-1]['read_only'] = False
            elif kind == 'alias':
                changed['mounts'][3]['source'] = changed['mounts'][-1]['source']
            elif kind == 'ancestor':
                changed['mounts'][3]['source'] = str(self.root)
            else:
                changed['mounts'].pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                host.model_contract.validate_host_operation(changed, data)

    def test_replay_host_receipt_requires_the_entire_32_file_owned_content_pass(self):
        spec, _ = self.replay_host_spec()
        system = FakeSystem(spec, self.output)
        original = system.start
        def start(cid, log):
            process = original(cid, log)
            expected = host.validate_data(spec)
            (self.output/'owned-teacher-operation-result.json').write_text(json.dumps(self.model_outcome(expected)))
            return process
        system.start = start
        result = host.run(spec, self.output, system=system)
        binding = json.loads((self.output/'model-content-binding.json').read_text())
        self.assertEqual(len(binding['assets']), 32)
        self.assertEqual(len({row['identity']['path'] for row in binding['assets']}), 32)
        self.assertEqual(result['container']['State']['Pid'], 0)

    def test_positive_raw_host_closure_and_external_cli_completion_are_separate(self):
        spec, _ = self.model_spec()
        expected = host.validate_data(spec)
        self.output.mkdir()
        inner_result = self.model_outcome(expected)
        closure = {'schema': 'tensorfold.native-contract-host-result-v1', 'report_complete': True,
                   'source_match': True, 'health_match': True, 'primary_error': None, 'cleanup_errors': [],
                   'attach_exit': 0, 'container': document(spec, state()), 'inner': inner_result}
        c = host.model_contract
        self.assertIs(c.validate_completed_teacher_host(closure, authority_sha256='a'*64, authority_bytes=1), inner_result)
        for change in ({'report_complete': False}, {'primary_error': 'failure'}, {'cleanup_errors': ['failure']},
                       {'attach_exit': False}, {'container': {**closure['container'], 'State': {**state(), 'Pid': 123}}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.validate_completed_teacher_host({**closure, **change}, authority_sha256='a'*64, authority_bytes=1)
        with self.assertRaises(ValueError):
            c.validate_completed_teacher_host(closure, authority_sha256='b'*64, authority_bytes=1)
        # A raw Host/attach success cannot substitute for actual external CLI0.
        with self.assertRaises(ValueError):
            c.validate_teacher_completion(closure, host_result_sha256='c'*64, authority_sha256='a'*64)


    def benchmark_host_spec(self):
        spec, data = self.model_spec()
        c, op = host.model_contract, data['operation']
        for role in ('plan', 'client'):
            old = self.operator/f'teacher-{role}.json'
            spec['inputs'] = [item for item in spec['inputs'] if item['path'] != str(old)]
            path = self.operator/f'benchmark-{role}.json'
            path.write_text('{}\n')
            record = file_record(path)
            spec['inputs'].append(record)
            op['benchmark_'+role] = {**record, 'path': '/operator/'+path.name}
            op.pop('teacher_'+role)
        op['trace'] = {'prefill_start': 1, 'host_obligation_bytes': 4096,
                       'device_headroom_bytes': 8192, 'disk_bytes': 1 << 20}
        for name in ('bench_concurrent.py', 'v100_adaptive_qualification.py', 'openai_protocol.py',
                     'worker_lifetime.py', 'benchmark_output.py'):
            source = self.source/'tools'/name
            source.write_bytes(b'# reviewed benchmark dependency\n')
            data['source_files']['tools/'+name] = {k: file_record(source)[k] for k in ('bytes', 'sha256')}
        manifest_path = Path(spec['source']['manifest']['path'])
        manifest = json.loads(manifest_path.read_text())
        for name in data['source_files']:
            path = self.source/name
            manifest['files'][name] = {'type': 'file', 'stored_mode': path.stat().st_mode & 0o7777,
                                       **data['source_files'][name]}
        manifest_path.write_text(json.dumps(manifest))
        spec['source']['manifest'] = file_record(manifest_path)
        data.update(schema=c.BENCHMARK_SCHEMA, mode=c.BENCHMARK_MODE)
        inner_path = self.operator/'spec.json'
        inner_path.write_text(json.dumps(data))
        spec['inputs'][1] = file_record(inner_path)
        spec['argv'][5], spec['argv'][7] = spec['inputs'][1]['sha256'], str(spec['inputs'][1]['bytes'])
        return spec, data

    def benchmark_host_outcome(self, expected):
        c, op = host.model_contract, expected['operation']
        def ref(name, rank=None):
            return {'file': name, 'bytes': 1, 'sha256': 'a'*64, **({} if rank is None else {'rank': rank})}
        model = {'schema': c.BENCHMARK_RESULT_SCHEMA, 'benchmark_plan_sha256': op['benchmark_plan']['sha256'],
            'benchmark_client_sha256': op['benchmark_client']['sha256'], 'launch_exit': 0, 'P3_valid': False,
            'client_report': ref('benchmark/benchmark-client-report.json'), 'matrix_report': ref('benchmark/matrix-results.json'),
            'rank_receipts': [ref(f'benchmark/benchmark-rank{i}/summary.json', i) for i in range(8)],
            'startup_receipts': [ref(f'benchmark/rank-{i}.json', i) for i in range(8)],
            'child_native_records': [ref(f'child-native/benchmark-closed-rank-{i}-pid-{i+2}.json', i) for i in range(8)],
            'trace': {'inventory': ref('benchmark/trace/inventory.json'),
                      'selection': ref('benchmark/trace/prefill-selection.json'),
                      'receipt': ref('benchmark/trace/c1-prefill-start-ge-1-round-receipt.json'),
                      'chrome': ref('benchmark/trace/c1-prefill-start-ge-1-round.json')}}
        binding = {'schema': 'tensorfold.owned-model-content-binding-v1', 'completed': True,
                   'assets': [{'identity': item} for item in c.benchmark_operation_assets(op)]}
        path = self.output/'model-content-binding.json'
        path.write_text(json.dumps(binding))
        return {'schema': c.BENCHMARK_HOST_RESULT_SCHEMA, 'mode': c.BENCHMARK_MODE,
            'source_commit': expected['source_commit'], 'image_id_declared': expected['image_id'], 'operation': op,
            'source_after_match': True, 'content_binding_complete': True,
            'content_binding': {'file': path.name, **{k: file_record(path)[k] for k in ('bytes', 'sha256')}},
            'parent_cuda_uninitialized': True, 'model_result': model, 'primary_error': None, 'post_errors': [],
            'scope': 'fake boundary result, not model qualification'}

    def test_closed_benchmark_host_binds_new_mode_sources_and_two_fixed_data_files(self):
        spec, data = self.benchmark_host_spec()
        expected = host.validate_data(spec)
        self.assertEqual(expected['mode'], host.model_contract.BENCHMARK_MODE)
        self.assertEqual(expected['operation'], data['operation'])
        self.assertNotIn('modules', expected)
        bad = json.loads(json.dumps(data))
        bad['schema'] = host.model_contract.SCHEMA
        inner = self.operator/'spec.json'
        inner.write_text(json.dumps(bad))
        spec['inputs'][1] = file_record(inner)
        spec['argv'][5], spec['argv'][7] = spec['inputs'][1]['sha256'], str(spec['inputs'][1]['bytes'])
        with self.assertRaises(ValueError):
            host.validate_data(spec)

    def test_closed_benchmark_host_uses_typed_fixed_result_and_retires_actual_owned_handle(self):
        spec, _ = self.benchmark_host_spec()
        system = FakeSystem(spec, self.output)
        original = system.start
        def start(cid, log):
            process = original(cid, log)
            expected = host.validate_data(spec)
            (self.output/'owned-benchmark-operation-result.json').write_text(json.dumps(self.benchmark_host_outcome(expected)))
            return process
        system.start = start
        result = host.run(spec, self.output, system=system)
        self.assertEqual(result['inner']['mode'], host.model_contract.BENCHMARK_MODE)
        self.assertEqual(result['container']['State']['Pid'], 0)
        self.assertFalse((self.output/'owned-teacher-operation-result.json').exists())
        self.assertIn(['close'], system.calls)

    def test_closed_benchmark_host_rejects_suite_zero_and_wrong_mode_receipt_after_retirement(self):
        spec, _ = self.benchmark_host_spec()
        system = FakeSystem(spec, self.output)
        original = system.start
        def start(cid, log):
            process = original(cid, log)
            expected = host.validate_data(spec)
            result = self.benchmark_host_outcome(expected)
            result['mode'] = host.model_contract.MODE
            (self.output/'owned-benchmark-operation-result.json').write_text(json.dumps(result))
            return process
        system.start = start
        with self.assertRaises(ValueError):
            host.run(spec, self.output, system=system)
        self.assertIn(['close'], system.calls)
        self.assertTrue(host.terminal(json.loads((self.output/'host-result.json').read_text())['container']))


if __name__ == "__main__":
    unittest.main()
