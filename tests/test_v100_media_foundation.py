"""Owned media acquisition boundaries; no processor/model/codec qualification."""

import hashlib
import importlib.util
import copy
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/v100"))
try:
    SPEC = importlib.util.spec_from_file_location("v100_media_foundation", ROOT / "deploy/v100/media_foundation.py")
    MEDIA = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MEDIA)
finally:
    sys.path.pop(0)


def pin(payload=b"exact source archive"):
    return {"name": "owned", "artifact": "owned.tar.gz", "url": "https://github.com/owner/project/source.tar.gz",
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_partial_transfer_is_resumed_and_exact_bytes_publish_atomically(tmp_path, monkeypatch):
    payload = b"exact source archive"
    partial = tmp_path / "owned.tar.gz.part"
    partial.write_bytes(payload[:5])
    seen = []

    def transfer(command, **kwargs):
        seen.append((command, kwargs))
        assert partial.read_bytes() == payload[:5]
        partial.write_bytes(payload)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(MEDIA.subprocess, "run", transfer)
    assert MEDIA.fetch_artifact(pin(payload), tmp_path, deadline_s=60) == "downloaded"
    assert (tmp_path / "owned.tar.gz").read_bytes() == payload and not partial.exists()
    assert seen[0][0][seen[0][0].index("--continue-at") + 1] == "-"
    assert seen[0][0][seen[0][0].index("--max-filesize") + 1] == str(len(payload))
    assert seen[0][1]["timeout"] == 90
    assert MEDIA.fetch_artifact(pin(payload), tmp_path) == "reused" and len(seen) == 1


@pytest.mark.parametrize("failure", ["transport", "identity"])
def test_failed_or_changed_download_retains_partial_without_publishing(tmp_path, monkeypatch, failure):
    def transfer(command, **kwargs):
        (tmp_path / "owned.tar.gz.part").write_bytes(b"wrong source payload")
        return subprocess.CompletedProcess(command, 28 if failure == "transport" else 0, "", "")

    monkeypatch.setattr(MEDIA.subprocess, "run", transfer)
    with pytest.raises((RuntimeError, ValueError), match="partial retained"):
        MEDIA.fetch_artifact(pin(), tmp_path)
    assert (tmp_path / "owned.tar.gz.part").exists() and not (tmp_path / "owned.tar.gz").exists()


def test_dangling_partial_alias_is_refused_before_transport(tmp_path, monkeypatch):
    (tmp_path / "owned.tar.gz.part").symlink_to(tmp_path / "outside")
    monkeypatch.setattr(MEDIA.subprocess, "run", lambda *args, **kwargs: pytest.fail("transport reached"))
    with pytest.raises(ValueError, match="bounded regular"):
        MEDIA.fetch_artifact(pin(), tmp_path)


@pytest.mark.parametrize("change", ["traversal", "credentials", "origin", "hash", "bytes"])
def test_malformed_pins_refuse_before_transport(tmp_path, monkeypatch, change):
    row = pin()
    if change == "traversal":
        row["artifact"] = "../outside"
    elif change == "credentials":
        row["url"] = "https://token@github.com/owner/source"
    elif change == "origin":
        row["url"] = "https://unapproved.invalid/source"
    elif change == "hash":
        row["sha256"] = "short"
    else:
        row["bytes"] = True
    monkeypatch.setattr(MEDIA.subprocess, "run", lambda *args, **kwargs: pytest.fail("transport reached"))
    with pytest.raises(ValueError):
        MEDIA.fetch_artifact(row, tmp_path)


def test_source_extraction_uses_exact_archive_and_cumulative_budget(tmp_path):
    downloads, sources = tmp_path / "downloads", tmp_path / "sources"
    downloads.mkdir()
    initial = tmp_path / "root"
    initial.mkdir()
    (initial / "source.txt").write_bytes(b"canonical source")
    archive = downloads / "owned.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        writer.add(initial, arcname="selected-source")
    row = pin(archive.read_bytes())
    row["extraction_max_bytes"] = 1000
    target = MEDIA.extract_source(row, downloads, sources)
    assert (target / "source.txt").read_bytes() == b"canonical source"
    assert MEDIA.extract_source(row, downloads, sources) == target
    (target / "source.txt").write_bytes(b"unrecorded source modification")
    with pytest.raises(ValueError, match="source input changed"):
        MEDIA.extract_source(row, downloads, sources)
    row["name"] = "bounded"
    row["extraction_max_bytes"] = 1
    with pytest.raises(ValueError, match="extraction budget"):
        MEDIA.extract_source(row, downloads, sources)


def test_source_cache_accepts_only_original_or_exact_completed_generated_inventory(tmp_path):
    downloads, sources = tmp_path / "downloads", tmp_path / "sources"
    downloads.mkdir()
    initial = tmp_path / "root"
    initial.mkdir()
    (initial / "source.py").write_text("immutable = True\n")
    archive = downloads / "owned.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        writer.add(initial, arcname="selected-source")
    row = pin(archive.read_bytes())
    row["extraction_max_bytes"] = 1000
    target = MEDIA.extract_source(row, downloads, sources)
    generated = target / "generated.o"
    generated.write_bytes(b"completed build output")
    with pytest.raises(ValueError, match="unrecorded generated"):
        MEDIA.verify_source(row, downloads, sources)
    receipt = tmp_path / "completed.json"
    receipt.write_bytes(MEDIA.canonical(MEDIA.source_inventory(row, downloads, sources)))
    MEDIA.verify_source(row, downloads, sources, receipt)
    generated.write_bytes(b"substituted cache bytes")
    with pytest.raises(ValueError, match="last completed stage"):
        MEDIA.verify_source(row, downloads, sources, receipt)


def test_source_cache_refuses_new_external_symlink(tmp_path):
    downloads, sources = tmp_path / "downloads", tmp_path / "sources"
    downloads.mkdir()
    initial = tmp_path / "root"
    initial.mkdir()
    (initial / "source.py").write_text("immutable = True\n")
    archive = downloads / "owned.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        writer.add(initial, arcname="selected-source")
    row = pin(archive.read_bytes())
    row["extraction_max_bytes"] = 1000
    target = MEDIA.extract_source(row, downloads, sources)
    (target / "external").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="introduce aliases"):
        MEDIA.source_inventory(row, downloads, sources)


def media_manifest(tmp_path, monkeypatch):
    artifact = tmp_path / "owned-media.so"
    artifact.write_bytes(b"actual installed media input")
    runtime = {"actual": "CPU processor and decoder identity"}
    pins = {"schema": "tensorfold.v100-media-inputs", "version": 1}
    monkeypatch.setattr(MEDIA, "runtime_inventory", lambda: runtime)
    monkeypatch.setattr(MEDIA, "installed_paths", lambda *args: {str(artifact)})
    manifest = {"schema": "tensorfold.v100-media-install", "version": 1,
                "pins_sha256": hashlib.sha256(MEDIA.canonical(pins)).hexdigest(),
                "packages": ["av"], "runtime": runtime,
                "artifacts": [{"path": str(artifact), **MEDIA.file_identity(artifact)}],
                "release_qualified": False, "pending_gates": ["V100", "compiler-6111901"]}
    return manifest, pins, artifact


def test_media_audit_binds_actual_files_runtime_and_pending_gates(tmp_path, monkeypatch):
    manifest, pins, _ = media_manifest(tmp_path, monkeypatch)
    result = MEDIA.audit_install(manifest, pins)
    assert result["software_integrity"] == "passed" and result["release_qualified"] is False
    assert result["pending_gates"] == ["V100", "compiler-6111901"]


@pytest.mark.parametrize("change", ["bytes", "runtime", "extra", "duplicate", "schema", "version", "pins", "qualified"])
def test_media_audit_refuses_changed_or_forged_inputs(tmp_path, monkeypatch, change):
    manifest, pins, artifact = media_manifest(tmp_path, monkeypatch)
    if change == "bytes":
        artifact.write_bytes(b"substituted media library")
    elif change == "runtime":
        monkeypatch.setattr(MEDIA, "runtime_inventory", lambda: {"actual": "different"})
    elif change == "extra":
        monkeypatch.setattr(MEDIA, "installed_paths", lambda *args: {str(artifact), "/extra-input"})
    elif change == "duplicate":
        manifest["artifacts"].append(copy.deepcopy(manifest["artifacts"][0]))
    elif change == "schema":
        manifest["schema"] = "unknown"
    elif change == "version":
        manifest["version"] = True
    elif change == "pins":
        pins["unexpected"] = "changed"
    else:
        manifest["release_qualified"] = True
    with pytest.raises(ValueError):
        MEDIA.audit_install(manifest, pins)


@pytest.fixture
def media_driver(tmp_path):
    """Real shell/receipt code; only external build/admission tools are spies."""
    import json
    import os

    state = tmp_path / "media state"
    tools = tmp_path / "tools"
    tools.mkdir()
    log = tmp_path / "commands.jsonl"
    probe = tools / "receipt_probe.py"
    probe.write_text(r"""import builtins
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time

script, arguments = Path(sys.argv[1]), sys.argv[2:]
sys.path.insert(0, str(script.parent))
spec = importlib.util.spec_from_file_location("media_under_io_probe", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
phase = arguments[0]
state = Path(arguments[arguments.index("--state") + 1])
stage = arguments[arguments.index("--source-name") + 1]
base = state / "receipts" / ("build-environment-" + stage)
pending, target = base.with_suffix(".pending"), base.with_suffix(".json")
fault = os.environ.get("TF_TEST_IO_FAILURE", "")
fired = False

def event(operation, path, **extra):
    with builtins.open(os.environ["TF_TEST_IO_LOG"], "a") as sink:
        sink.write(json.dumps({"operation": operation, "path": str(path), "phase": phase,
                               "time_ns": time.monotonic_ns(), **extra}) + "\n")

def fail_if(match):
    global fired
    if match and (not fired or fault == "complete-dir-fsync-permanent"):
        fired = True
        raise OSError("injected " + fault)

class Written:
    def __init__(self, stream, path):
        self.stream, self.path = stream, Path(path)
    def __enter__(self):
        self.stream.__enter__()
        return self
    def __exit__(self, *args):
        return self.stream.__exit__(*args)
    def fileno(self):
        return self.stream.fileno()
    def write(self, payload):
        event("write", self.path)
        fail_if(fault == "pending-write" and self.path.suffix == ".pending"
                or fault == "complete-write" and self.path.suffix == ".publishing")
        return self.stream.write(payload)
    def flush(self):
        self.stream.flush()
        event("flush-done", self.path)

def opened(path, mode="r", **kwargs):
    stream = builtins.open(path, mode, **kwargs)
    return Written(stream, path) if mode in ("wb", "xb") else stream
module.open = opened
actual_fsync, actual_replace, actual_unlink = os.fsync, os.replace, os.unlink

def fsync(fd):
    path = Path(os.readlink("/proc/self/fd/" + str(fd)))
    directory = stat.S_ISDIR(os.fstat(fd).st_mode)
    new_target = target.exists() and json.loads(target.read_text())["MAX_JOBS"] == "24"
    fail_if(fault == "pending-file-fsync" and not directory and path.suffix == ".pending"
            or fault == "pending-dir-fsync" and directory and phase == "begin-build-environment"
            or fault == "complete-file-fsync" and not directory and path.suffix == ".publishing"
            or fault == "complete-dir-fsync" and directory and phase == "complete-build-environment"
               and new_target and pending.exists()
            or fault == "complete-dir-fsync-permanent" and directory and phase == "complete-build-environment"
               and (new_target or fired)
            or fault == "retire-dir-fsync" and directory and phase == "complete-build-environment"
               and new_target and not pending.exists())
    actual_fsync(fd)
    event("fsync-done", path, directory=directory)

def replace(source, destination):
    fail_if(fault == "complete-rename" and Path(source).suffix == ".publishing")
    actual_replace(source, destination)
    event("replace-done", source, target=str(destination))

def unlink(path, **kwargs):
    actual_unlink(path, **kwargs)
    event("unlink-done", path)
module.os.fsync, module.os.replace, module.os.unlink = fsync, replace, unlink
sys.argv = [str(script), *arguments]
module.main()
""")
    shim = f'''#!{sys.executable}
import json
import os
from pathlib import Path
import sys
import time

name, args = Path(sys.argv[0]).name, sys.argv[1:]
if name == "python" and args[0].endswith("media_foundation.py") and args[1] in ("begin-build-environment", "complete-build-environment"):
    os.execv({sys.executable!r}, [{sys.executable!r}, str(Path(__file__).with_name("receipt_probe.py")), *args])
if name == "python" and args[0] == "-":
    os.execv({sys.executable!r}, [{sys.executable!r}, *args])
with open(os.environ["TF_TEST_LOG"], "a") as sink:
    sink.write(json.dumps({{"tool": name, "args": args, "time_ns": time.monotonic_ns(), "controls": {{key: os.environ.get(key) for key in
        ("MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL", "OMP_NUM_THREADS")}}}}) + "\\n")
failure = os.environ.get("TF_TEST_FAILURE")
if failure == name or failure == "install" and "install" in args or failure == "wheel" and "wheel" in args:
    sys.exit(7)
if name == "python" and args[0].endswith("source_foundation.py"):
    Path(args[args.index("--output") + 1]).write_text('{{"tool":"external-boundary-only"}}\\n')
elif name == "python" and args[0].endswith("media_foundation.py") and args[1] == "record-source":
    Path(args[args.index("--receipt") + 1]).write_text(json.dumps({{"source": os.environ["TF_TEST_STAGE"]}}) + "\\n")
'''
    for name in ("python", "make", "ninja", "meson"):
        path = tools / name
        path.write_text(shim)
        path.chmod(0o755)
    for name in ("torchvision", "transformers", "dav1d", "ffmpeg", "pyav"):
        source = state / "sources" / name
        source.mkdir(parents=True)
        if name == "ffmpeg":
            (source / "configure").write_text(shim)
            (source / "configure").chmod(0o755)
            for license in ("COPYING.LGPL", "LICENSE.md"):
                (source / license).write_text("license boundary fixture\n")
        elif name == "dav1d":
            (source / "doc").mkdir()
            for license in ("COPYING", "doc/PATENTS"):
                (source / license).write_text("license boundary fixture\n")

    def run(stage, jobs="24", failure="", io_failure=""):
        environment = {**os.environ, "TF_MEDIA_STATE": str(state), "TF_MEDIA_PREFIX": str(state / "native"),
                       "TF_PROJECT_ROOT": str(ROOT), "PATH": str(tools) + os.pathsep + os.environ.get("PATH", ""),
                       "TF_TEST_LOG": str(log), "TF_TEST_STAGE": stage, "TF_TEST_FAILURE": failure,
                       "TF_TEST_IO_LOG": str(tmp_path / "io.jsonl"), "TF_TEST_IO_FAILURE": io_failure,
                       "CMAKE_BUILD_PARALLEL_LEVEL": "99", "OMP_NUM_THREADS": "77"}
        if jobs is None:
            environment.pop("MAX_JOBS", None)
        else:
            environment["MAX_JOBS"] = jobs
        result = subprocess.run(["bash", str(ROOT / "deploy/v100/media_build.sh"), stage],
                                env=environment, capture_output=True, text=True)
        return result, [json.loads(line) for line in log.read_text().splitlines()]
    return run, state


@pytest.mark.parametrize("stage", ["torchvision", "transformers", "dav1d", "ffmpeg", "pyav"])
@pytest.mark.parametrize("jobs", [None, "", "4", "24"])
def test_media_driver_propagates_actual_controls_and_records_success(media_driver, stage, jobs):
    import json

    run, state = media_driver
    result, commands = run(stage, jobs)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {name: jobs or "4" for name in ("MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL", "OMP_NUM_THREADS")}
    assert all(command["controls"] == expected for command in commands)
    for command in commands:
        if command["tool"] in ("make", "ninja"):
            assert "-j" + (jobs or "4") in command["args"]
    if stage == "dav1d":
        ninja = next(i for i, row in enumerate(commands) if row["tool"] == "ninja")
        install = next(i for i, row in enumerate(commands) if row["tool"] == "meson" and row["args"][0] == "install")
        assert ninja < install and "--no-rebuild" in commands[install]["args"]
    if stage == "ffmpeg":
        install = next(row for row in commands if row["tool"] == "make" and "install" in row["args"])
        assert "-j" + (jobs or "4") in install["args"]
    completed = state / "receipts" / ("build-environment-" + stage + ".json")
    value = json.loads(completed.read_text())
    assert {name: value[name] for name in expected} == expected
    assert value["producer_script_sha256"] == MEDIA.file_identity(ROOT / "deploy/v100/media_build.sh")["sha256"]
    assert value["producer_helper_sha256"] == MEDIA.file_identity(Path(MEDIA.__file__))["sha256"]
    source = MEDIA.file_identity(state / "receipts" / ("source-" + stage + ".json"))
    assert value["source_receipt_sha256"] == source["sha256"]
    assert not completed.with_suffix(".pending").exists()
    assert MEDIA.build_environments(state, {stage: source}, value["producer_script_sha256"],
                                    MEDIA.file_identity(Path(MEDIA.__file__))["sha256"]) == {stage: value}


@pytest.mark.parametrize("stage,failure", [("torchvision", "wheel"), ("ffmpeg", "make"), ("ffmpeg", "install"),
                                          ("dav1d", "ninja"), ("dav1d", "install"), ("pyav", "make"), ("pyav", "wheel")])
def test_media_failure_cannot_relabel_an_old_completed_stage(media_driver, stage, failure):
    run, state = media_driver
    first, _ = run(stage, "4")
    assert first.returncode == 0
    receipt = state / "receipts" / ("build-environment-" + stage + ".json")
    before = receipt.read_bytes()
    second, commands = run(stage, "24", failure)
    assert second.returncode == 7
    assert receipt.read_bytes() == before
    assert receipt.with_suffix(".pending").is_file()
    source = MEDIA.file_identity(state / "receipts" / ("source-" + stage + ".json"))
    with pytest.raises(ValueError, match="unfinished"):
        MEDIA.build_environments(state, {stage: source}, MEDIA.file_identity(ROOT / "deploy/v100/media_build.sh")["sha256"],
                                 MEDIA.file_identity(Path(MEDIA.__file__))["sha256"])
    if failure == "ninja":
        last = next(i for i, row in enumerate(commands) if row["tool"] == "ninja" and row["controls"]["MAX_JOBS"] == "24")
        assert not any(row["tool"] == "meson" and row["args"][0] == "install" for row in commands[last + 1:])


@pytest.mark.parametrize("jobs", ["0", "-1", "25", "2147483648", "9" * 1000, "04", " 4", "4.0", "4;false"])
def test_media_jobs_reject_before_acquisition_or_state_creation(tmp_path, jobs):
    import os

    state, prefix = tmp_path / "absent-state", tmp_path / "absent-prefix"
    environment = {**os.environ, "MAX_JOBS": jobs, "TF_MEDIA_STATE": str(state), "TF_MEDIA_PREFIX": str(prefix),
                   "TF_PROJECT_ROOT": str(tmp_path / "missing-project")}
    result = subprocess.run(["bash", str(ROOT / "deploy/v100/media_build.sh"), "torchvision"],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 2 and "MAX_JOBS must be 1..24" in result.stderr
    assert not state.exists() and not prefix.exists()


@pytest.mark.parametrize("change", ["controls", "stage", "script", "helper", "source", "range", "version", "extra"])
def test_media_environment_refuses_forged_or_stale_facts(media_driver, change):
    import json

    run, state = media_driver
    assert run("transformers")[0].returncode == 0
    target = state / "receipts/build-environment-transformers.json"
    value = json.loads(target.read_text())
    if change == "controls":
        value["CMAKE_BUILD_PARALLEL_LEVEL"] = "4"
    elif change == "stage":
        value["stage"] = "ffmpeg"
    elif change == "script":
        value["producer_script_sha256"] = "0" * 64
    elif change == "helper":
        value["producer_helper_sha256"] = "0" * 64
    elif change == "source":
        value["source_receipt_sha256"] = "0" * 64
    elif change == "range":
        value["MAX_JOBS"] = "25"
    elif change == "version":
        value["version"] = True
    else:
        value["unexpected"] = "not a stage fact"
    target.write_text(json.dumps(value))
    source = MEDIA.file_identity(state / "receipts/source-transformers.json")
    with pytest.raises(ValueError, match="stage/source/policy"):
        MEDIA.build_environments(state, {"transformers": source}, MEDIA.file_identity(ROOT / "deploy/v100/media_build.sh")["sha256"],
                                 MEDIA.file_identity(Path(MEDIA.__file__))["sha256"])


def test_media_export_embeds_the_actual_completed_stage_environments(media_driver, monkeypatch):
    import json

    run, state = media_driver
    stages = ("torchvision", "transformers", "dav1d", "ffmpeg", "pyav")
    for stage in stages:
        result, _ = run(stage)
        assert result.returncode == 0, result.stdout + result.stderr
    wheels = state / "wheels"
    for name in ("av", "torchvision", "transformers"):
        (wheels / (name + "-boundary.whl")).write_bytes(b"external wheel boundary; no media build proof")
    # Independent source/Ninja admission controls retain their own real tests.
    # This test exercises only actual shell environment publication and export.
    monkeypatch.setattr(MEDIA, "verify_source", lambda *args: None)
    monkeypatch.setattr(MEDIA, "ninja_tool_identity", lambda *args: {"tool": "external-boundary-only"})
    pins = {"artifacts": [{"name": stage, "kind": "source-archive"} for stage in stages]}
    for control in ("MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL", "OMP_NUM_THREADS"):
        monkeypatch.setenv(control, "1")  # Export-time settings cannot relabel completed 24-job stages.
    result = MEDIA.capture_build(pins, state, ROOT)
    expected = {stage: json.loads((state / "receipts" / ("build-environment-" + stage + ".json")).read_text())
                for stage in stages}
    assert result["build_environments"] == expected
    assert all(value["MAX_JOBS"] == "24" for value in expected.values())
    assert result["release_qualified"] is False


@pytest.mark.parametrize("jobs", ["1", "9", "10", "19", "20"])
def test_media_accepted_job_range_boundaries_are_recorded(media_driver, jobs):
    import json

    run, state = media_driver
    result, commands = run("transformers", jobs)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(all(value == jobs for value in command["controls"].values()) for command in commands)
    value = json.loads((state / "receipts/build-environment-transformers.json").read_text())
    assert value["MAX_JOBS"] == jobs


@pytest.mark.parametrize("fault", ["pending-write", "pending-file-fsync", "pending-dir-fsync", "complete-write",
                                  "complete-file-fsync", "complete-rename", "complete-dir-fsync",
                                  "retire-dir-fsync", "complete-dir-fsync-permanent"])
def test_media_durable_publication_failure_retains_gate_and_prior_facts(media_driver, fault):
    run, state = media_driver
    first, _ = run("ffmpeg", "4")
    assert first.returncode == 0, first.stderr
    target = state / "receipts/build-environment-ffmpeg.json"
    prior = target.read_bytes()
    second, commands = run("ffmpeg", "24", io_failure=fault)
    assert second.returncode != 0 and "injected " + fault in second.stderr
    assert target.read_bytes() == prior
    assert target.with_suffix(".pending").is_file()
    if fault.startswith("pending-"):
        assert not any(row["controls"]["MAX_JOBS"] == "24" and row["tool"] in ("configure", "make") for row in commands)
    else:
        assert target.with_suffix(".previous").read_bytes() == prior
    if fault.endswith("permanent"):
        assert "publication and recovery failed" in second.stderr
    source = MEDIA.file_identity(state / "receipts/source-ffmpeg.json")
    with pytest.raises(ValueError, match="unfinished"):
        MEDIA.build_environments(state, {"ffmpeg": source}, MEDIA.file_identity(ROOT / "deploy/v100/media_build.sh")["sha256"],
                                 MEDIA.file_identity(Path(MEDIA.__file__))["sha256"])


def test_media_durability_barriers_precede_build_and_completion_visibility(media_driver):
    import json

    run, state = media_driver
    result, commands = run("ffmpeg", "24")
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in (state.parent / "io.jsonl").read_text().splitlines()]
    begin = [row for row in events if row["phase"] == "begin-build-environment"]
    file_sync = next(row["time_ns"] for row in begin if row["operation"] == "fsync-done" and not row["directory"])
    parent_sync = next(row["time_ns"] for row in begin if row["operation"] == "fsync-done" and row["directory"])
    first_build = min(row["time_ns"] for row in commands if row["tool"] in ("configure", "make"))
    assert file_sync < parent_sync < first_build
    complete = [row for row in events if row["phase"] == "complete-build-environment"]
    completed_sync = next(row["time_ns"] for row in complete if row["operation"] == "fsync-done"
                          and Path(row["path"]).suffix == ".publishing")
    publication = next(row["time_ns"] for row in complete if row["operation"] == "replace-done"
                       and Path(row["path"]).suffix == ".publishing")
    published_parent = next(row["time_ns"] for row in complete if row["operation"] == "fsync-done"
                            and row["directory"] and row["time_ns"] > publication)
    retirement = next(row["time_ns"] for row in complete if row["operation"] == "unlink-done"
                      and Path(row["path"]).suffix == ".pending")
    retired_parent = next(row["time_ns"] for row in complete if row["operation"] == "fsync-done"
                          and row["directory"] and row["time_ns"] > retirement)
    assert completed_sync < publication < published_parent < retirement < retired_parent


def test_media_cannot_start_over_an_unretired_pending_attempt(media_driver):
    run, state = media_driver
    result, before = run("ffmpeg", "24", failure="make")
    assert result.returncode == 7
    pending = state / "receipts/build-environment-ffmpeg.pending"
    original = pending.read_bytes()
    retry, after = run("ffmpeg", "24")
    assert retry.returncode != 0 and pending.read_bytes() == original
    assert not any(row["tool"] in ("configure", "make") for row in after[len(before):])


@pytest.mark.parametrize("fault", ["complete-write", "complete-file-fsync", "complete-rename",
                                  "complete-dir-fsync", "retire-dir-fsync"])
def test_media_first_completion_failure_never_leaves_exportable_facts(media_driver, fault):
    run, state = media_driver
    result, _ = run("ffmpeg", "24", io_failure=fault)
    assert result.returncode != 0
    target = state / "receipts/build-environment-ffmpeg.json"
    assert not target.exists() and target.with_suffix(".pending").is_file()
    source = MEDIA.file_identity(state / "receipts/source-ffmpeg.json")
    with pytest.raises(ValueError, match="unfinished"):
        MEDIA.build_environments(state, {"ffmpeg": source}, MEDIA.file_identity(ROOT / "deploy/v100/media_build.sh")["sha256"],
                                 MEDIA.file_identity(Path(MEDIA.__file__))["sha256"])
