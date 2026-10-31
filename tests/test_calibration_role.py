"""Boundary tests for addition-only producer admission; no Torch/GPU imports."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import ModuleType
import zipfile

import pytest
from packaging.markers import default_environment

spec = importlib.util.spec_from_file_location("calibration_role",
    Path(__file__).resolve().parents[1] / "deploy/gb10/calibration/role.py")
role = importlib.util.module_from_spec(spec)
spec.loader.exec_module(role)


def wheel(tmp_path, *, member=None, name="nvidia-modelopt", requires=(), version="0.0.0"):
    path = tmp_path / f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    metadata = (f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                "Requires-Python: >=3.11\n" + "".join(f"Requires-Dist: {r}\n" for r in requires))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name.replace('-', '_')}-{version}.dist-info/METADATA", metadata)
        archive.writestr(member or "modelopt/__init__.py", "# fixture\n")
    return path


@pytest.mark.parametrize("member", ["activate.pth", "../escape.py", "/absolute.py",
    "sitecustomize.py", "payload/libcublasLt.so.13", "payload/libnvidia-ml.so.1"])
def test_rejects_activation_escape_and_replacement_provider(tmp_path, member):
    with pytest.raises(ValueError):
        role.wheel_record(wheel(tmp_path, member=member))


def test_narrow_provider_exception_and_cpu_payload(tmp_path):
    assert role.wheel_record(wheel(tmp_path, member="modelopt/templates/kernel.cu"))["name"] == "nvidia-modelopt"
    with pytest.raises(ValueError, match="provider package"):
        role.wheel_record(wheel(tmp_path, name="nvidia-cublas-cu13"))


def test_union_dependency_constraints_include_prerelease_baseline():
    packages = {"producer": {"version": "1", "requires": ["torch>=2.10", "missing; python_version<'3'"]},
                "torch": {"version": "2.16.0.dev20261014+cu134", "requires": []}}
    role.closure(packages, default_environment())
    packages["producer"]["requires"] = ["torch<2.0"]
    with pytest.raises(ValueError, match="unsatisfied dependency"):
        role.closure(packages, default_environment())
    packages["producer"]["requires"] = ["torch[extra]>=2"]
    packages["torch"]["requires"] = ["feature; extra=='extra'"]
    with pytest.raises(ValueError, match="unsatisfied dependency"):
        role.closure(packages, default_environment())
    packages["feature"] = {"version": "1", "requires": []}
    role.closure(packages, default_environment())


def test_real_report_preparation_records_baseline_edges_and_hashes(tmp_path, monkeypatch):
    artifact = wheel(tmp_path, requires=["torch>=2.10"])
    baseline = {"base": role.BASE, "environment": default_environment(), "files": {},
                "packages": {"torch": {"version": "2.16.0.dev20261014+cu134", "requires": []}}}
    report = {"environment": default_environment(), "install": [{
        "metadata": {"name": "nvidia-modelopt", "version": "0.0.0"},
        "download_info": {"url": artifact.as_uri(), "archive_info": {"hashes": {"sha256": role.sha(artifact)}}}}]}
    provenance = {"source_commit": role.MODEL_OPT_COMMIT, "source_archive_sha256": "a" * 64,
                  "wheel_sha256": role.sha(artifact)}
    paths = []
    for filename, content in [("baseline.json", baseline), ("report.json", report), ("source.json", provenance)]:
        path = tmp_path / filename
        path.write_text(json.dumps(content))
        paths.append(path)
    output = tmp_path / "generated"
    monkeypatch.setattr(role.sys, "argv", ["role.py", "prepare", "--baseline", str(paths[0]),
        "--report", str(paths[1]), "--wheels", str(tmp_path), "--provenance", str(paths[2]),
        "--output", str(output)])
    role.main()
    manifest = role.read(output / "manifest.json")
    assert manifest["baseline"]["packages"] == baseline["packages"]
    assert set(manifest["additions"]) == {"nvidia-modelopt"}
    assert role.sha(artifact) in (output / "requirements.lock").read_text()
    report["install"][0]["metadata"]["name"] = "torch"
    paths[1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="existing package"):
        role.prepare(paths[0], paths[1], tmp_path, paths[2], tmp_path / "bad")


def test_seal_detects_symlink_and_content_changes(tmp_path):
    path = tmp_path / "payload.py"
    path.write_text("first")
    before = role.tree(tmp_path)
    path.write_text("second")
    assert role.tree(tmp_path) != before
    (tmp_path / "alias").symlink_to(path)
    with pytest.raises(ValueError, match="aliases"):
        role.tree(tmp_path)


def test_successful_audit_checks_separate_prefix_and_complete_baseline(tmp_path, monkeypatch):
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    (prefix / "modelopt.py").write_text("# payload")
    old = {"torch": {"version": "2.16.0.dev20261014+cu134", "requires": []}}
    new = {"nvidia-modelopt": {"version": "0.0.0", "requires": ["torch>=2.10"]}}
    manifest = {"baseline": {"packages": old, "files": {}, "environment": default_environment()},
                "additions": new}
    calls = []
    monkeypatch.setattr(role.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    monkeypatch.setattr(role, "installed", lambda paths=None: new if paths else old)
    monkeypatch.setattr(role, "baseline_files", lambda: {})
    monkeypatch.setattr(role.md, "distributions", lambda: [])
    role.audit(prefix, manifest)
    assert calls == [[role.sys.executable, role.GUARD, "audit"]]
    monkeypatch.setattr(role, "baseline_files", lambda: {"changed": "digest"})
    with pytest.raises(ValueError, match="payloads changed"):
        role.audit(prefix, manifest)


def test_runner_forwards_literal_flags_and_finally_audits(tmp_path, monkeypatch):
    monkeypatch.setattr(role, "provider_proof", lambda: None)
    monkeypatch.setattr(role.sys, "path", role.sys.path.copy())
    monkeypatch.setattr(role.sys, "argv", role.sys.argv.copy())
    monkeypatch.setattr(role.subprocess, "run", lambda *args, **kwargs: None)
    audits, observed = [], []
    monkeypatch.setattr(role, "audit", lambda *args: audits.append("audited"))
    monkeypatch.setattr(role, "seal_inputs", lambda *args: {"sealed": True})
    monkeypatch.setattr(role.runpy, "run_path", lambda *args, **kwargs: observed.append(role.sys.argv.copy()))
    role.run_script("job.py", ["--model", "a b", "--output=/work/out"], tmp_path, {},
                    {"sealed": True}, tmp_path / "manifest.json")
    assert observed == [["job.py", "--model", "a b", "--output=/work/out"]]
    assert audits == ["audited"]
    assert str(tmp_path) not in role.sys.path


def test_runner_preserves_primary_and_attempts_all_failed_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(role, "provider_proof", lambda: None)
    monkeypatch.setattr(role.sys, "path", role.sys.path.copy())
    monkeypatch.setattr(role.sys, "argv", role.sys.argv.copy())
    monkeypatch.setattr(role.subprocess, "run", lambda *args, **kwargs: None)
    attempts = []
    def job(*args, **kwargs):
        role.sys.path.remove(str(tmp_path))
        raise RuntimeError("job failed")
    def audit(*args):
        attempts.append("baseline")
        raise RuntimeError("audit failed")
    def sealed(*args):
        attempts.append("seal")
        raise RuntimeError("seal failed")
    monkeypatch.setattr(role.runpy, "run_path", job)
    monkeypatch.setattr(role, "audit", audit)
    monkeypatch.setattr(role, "seal_inputs", sealed)
    with pytest.raises(BaseExceptionGroup) as failure:
        role.run_script("job.py", [], tmp_path, {}, {}, tmp_path / "manifest.json")
    assert str(failure.value.exceptions[0]) == "job failed"
    assert len(failure.value.exceptions[1].exceptions) == 3
    assert attempts == ["baseline", "seal"]


def test_provider_phase_text_cannot_replace_activation_ownership(monkeypatch):
    monkeypatch.setenv("TF_FOUNDATION_RUNTIME_PHASE", "cuda")
    monkeypatch.setenv("TF_FOUNDATION_STATE", "/fixture/state.json")
    monkeypatch.delitem(role.sys.modules, "tensorfold_foundation_bootstrap", raising=False)
    monkeypatch.setattr(role, "read", lambda path: {
        "files": {"profile": {"sha256": "expected"}}, "runtime_profile_path": "profile"})
    with pytest.raises(RuntimeError, match="same-process"):
        role.provider_proof()


def test_extension_seal_includes_executable_helpers_and_requirements(tmp_path, monkeypatch):
    tools, prefix = tmp_path / "tools", tmp_path / "prefix"
    tools.mkdir()
    prefix.mkdir()
    helper = tools / "role.py"
    helper.write_text("# first")
    monkeypatch.setattr(role, "__file__", str(helper))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    lock = tmp_path / "requirements.lock"
    lock.write_text("first")
    before = role.seal_inputs(prefix, manifest)
    helper.write_text("# second")
    assert role.seal_inputs(prefix, manifest)["tools"] != before["tools"]
    lock.write_text("second")
    assert role.seal_inputs(prefix, manifest)["requirements_sha256"] != before["requirements_sha256"]


def test_cli_metadata_options_and_job_flags_are_distinct():
    parser = role.argument_parser()
    assert parser.parse_args(["snapshot", "--output", "/work/out"]).output == Path("/work/out")
    job = parser.parse_args(["run", "--prefix", "/work/prefix", "job.py", "--model", "m", "--prefix", "job-value"])
    assert job.prefix == Path("/work/prefix")
    assert job.script_args == ["--model", "m", "--prefix", "job-value"]
    with pytest.raises(SystemExit):
        parser.parse_args(["prepare", "--output", "/work/incomplete"])


@pytest.mark.parametrize("name,version,admitted", [
    ("setuptools-scm", "9.0.0", True), ("setuptools-scm", "7.0.0", False),
    ("unreviewed-builder", "9.0.0", False)])
def test_bootstrap_solution_needs_no_modelopt_but_has_narrow_build_scope(tmp_path, name, version, admitted, monkeypatch):
    artifact = wheel(tmp_path, name=name, version=version)
    documents = {
        "baseline.json": {"base": role.BASE, "environment": default_environment(), "packages": {}, "files": {}},
        "report.json": {"environment": default_environment(), "install": [{
            "metadata": {"name": name, "version": version}, "download_info": {
                "url": f"https://files.pythonhosted.org/{artifact.name}",
                "archive_info": {"hashes": {"sha256": role.sha(artifact)}}}}]},
        "source.json": {"source_commit": role.MODEL_OPT_COMMIT, "source_archive_sha256": "a" * 64}}
    for filename, content in documents.items():
        (tmp_path / filename).write_text(json.dumps(content))
    monkeypatch.setattr(role.sys, "argv", ["role.py", "prepare", "--baseline", str(tmp_path / "baseline.json"),
        "--report", str(tmp_path / "report.json"), "--wheels", str(tmp_path),
        "--provenance", str(tmp_path / "source.json"), "--output", str(tmp_path / "out"), "--kind", "build-tools"])
    if admitted:
        role.main()
        manifest = role.read(tmp_path / "out/manifest.json")
        assert manifest["kind"] == "build-tools"
        assert set(manifest["additions"]) == {"setuptools-scm"}
    else:
        with pytest.raises(ValueError, match="build-tools|setuptools-scm>=8"):
            role.main()


def test_snapshot_cli_dispatch_uses_documented_output_option(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(role, "snapshot", lambda destination: calls.append(destination))
    monkeypatch.setattr(role.sys, "argv", ["role.py", "snapshot", "--output", str(tmp_path / "out")])
    role.main()
    assert calls == [tmp_path / "out"]


def test_snapshot_full_cli_uses_real_native_stdlib_distribution_discovery(tmp_path, monkeypatch):
    # Exercise real Python3.12 discovery and actual image metadata; the separate
    # root integration job covers multi-GB content hashing without repeating it.
    monkeypatch.setattr(role, "baseline_files", lambda: {})
    monkeypatch.setattr(role.sys, "argv", ["role.py", "snapshot", "--output", str(tmp_path / "out")])
    role.main()
    baseline = role.read(tmp_path / "out/baseline.json")
    assert baseline["packages"]["transformers"]["version"] == "5.17.0"
    assert baseline["packages"]["torch"]["version"] == role.md.version("torch")
    assert baseline["packages"]["pip"]["version"] == role.md.version("pip")
    assert baseline["environment"]["platform_machine"] == "aarch64"


def test_real_metadata_duplicate_path_is_distinct_from_shadow_installation(tmp_path):
    sites = [tmp_path / "site1", tmp_path / "site2"]
    for site in sites:
        metadata = site / "sample-1.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: sample\nVersion: 1\n")
        (metadata / "RECORD").write_text("sample-1.dist-info/METADATA,,\nsample-1.dist-info/RECORD,,\n")
    assert role.installed([str(sites[0]), str(sites[0])]) == {"sample": {"version": "1", "requires": []}}
    with pytest.raises(ValueError, match="duplicate active"):
        role.installed(list(map(str, sites)))


def test_public_wheel_cannot_alias_setuptools_vendor_code(tmp_path, monkeypatch):
    public = tmp_path / "wheel"
    vendor = tmp_path / "setuptools/_vendor/wheel"
    public.mkdir()
    vendor.mkdir(parents=True)
    for directory in (public, vendor):
        (directory / "__init__.py").write_text("# fixture")
    selected = role.sys.path.copy()
    contract = {"wheel": {"origin": public / "__init__.py", "roots": (public,)}}
    module = ModuleType("wheel")
    module.__file__ = str(vendor / "__init__.py")
    monkeypatch.setitem(role.sys.modules, "wheel", module)
    with pytest.raises(ValueError, match="selected origin"):
        role.check_import_context(selected, contract)


def test_actual_managed_setuptools_consumer_restores_namespace_without_hiding_shadow(tmp_path):
    # Fresh interpreter: pytest/builder fixture imports cannot preactivate
    # Setuptools. No packages installed; the exact image's backend is exercised.
    control = tmp_path / "control.py"
    consumer = tmp_path / "consumer.py"
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    consumer.write_text('''import importlib.metadata as md
import hashlib,json,sys
from pathlib import Path
import setuptools.build_meta
import wheel,packaging
print(json.dumps({"consumer_path":sys.path,"wheel_version":wheel.__version__,
"wheel_file":wheel.__file__,"packaging_file":packaging.__file__,
"setuptools_file":setuptools.__file__,"setuptools_version":setuptools.__version__,
"setuptools_sha256":hashlib.sha256(Path(setuptools.__file__).read_bytes()).hexdigest(),
"wheel_metadata":[str(d.locate_file(p)) for d in md.distributions()
if d.metadata['Name'].lower()=='wheel' for p in d.files or []
if len(p.parts)==2 and p.name=='METADATA']}))
''')
    control.write_text(f'''import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("producer_role",{str(spec.origin)!r})
role=importlib.util.module_from_spec(spec);spec.loader.exec_module(role)
before=sys.path.copy();baseline=role.installed()
def audit(*args):
    assert role.installed()==baseline
role.audit=audit
role.seal_inputs=lambda *args:{{"test_seal":True}}
role.run_script({str(consumer)!r},[],Path({str(prefix)!r}),{{}},{{"test_seal":True}},Path("unused"))
assert sys.path==before
assert role.installed()==baseline
print(json.dumps({{"restored_path":sys.path,"baseline_wheel":baseline['wheel']['version']}}))
''')
    result = subprocess.run([role.sys.executable, str(control)], check=True, text=True, capture_output=True)
    reports = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    during, after = reports
    assert during["wheel_version"] == after["baseline_wheel"] == "0.47.0"
    assert "/setuptools/_vendor/" not in during["wheel_file"]
    assert "/setuptools/_vendor/" not in during["packaging_file"]
    assert during["consumer_path"][-1].endswith("/setuptools/_vendor")
    assert len(set(during["wheel_metadata"])) == 2
    assert after["restored_path"] == during["consumer_path"][:-2]
    print(json.dumps({"actual_setuptools_consumer": during, "after_restoration": after}, sort_keys=True))


def test_owned_fork_pin_matches_planner_and_exact_hosted_incremental_bundle():
    from tensorfold.offline_quantization import plan
    root=Path(__file__).resolve().parents[1]
    record=json.loads((root/'deploy/gb10/calibration/modelopt-fork/source.json').read_text())
    assert role.MODEL_OPT_COMMIT==plan.PRODUCER_COMMIT==record['fork_commit']
    assert role.MODEL_OPT_UPSTREAM_COMMIT==plan.PRODUCER_UPSTREAM_COMMIT==record['upstream_commit']
    assert role.MODEL_OPT_UPSTREAM_REPOSITORY==plan.PRODUCER_UPSTREAM_REPOSITORY==record['upstream_repository']
    assert role.MODEL_OPT_REPOSITORY==plan.PRODUCER_REPOSITORY=='https://github.com/kklouzal/TensorFold'
    assert role.MODEL_OPT_SOURCE_TYPE==plan.PRODUCER_SOURCE_TYPE=='incremental-git-bundle'
    assert role.MODEL_OPT_BUNDLE_PATH==plan.PRODUCER_BUNDLE_PATH
    assert role.MODEL_OPT_BUNDLE_SHA256==plan.PRODUCER_BUNDLE_SHA256==record['bundle_sha256']
    assert role.sha(root/role.MODEL_OPT_BUNDLE_PATH)==record['bundle_sha256']
    assert role.MODEL_OPT_COMMIT!=role.MODEL_OPT_UPSTREAM_COMMIT
    dockerfile=(root/'deploy/gb10/Dockerfile.calibration').read_text()
    assert f'tensorfold.modelopt.source="{role.MODEL_OPT_COMMIT}"' in dockerfile
    assert f'tensorfold.modelopt.upstream="{role.MODEL_OPT_UPSTREAM_COMMIT}"' in dockerfile
