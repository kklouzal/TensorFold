"""Target wheel/marker validation and deployment artifact selection boundaries."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "deploy/gb10"
HISTORY = DEPLOY / "history/20261006"


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(DEPLOY))
    modules = []
    for name in ("lock_dependencies", "select_pins"):
        spec = importlib.util.spec_from_file_location("test_" + name, DEPLOY / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def report_and_pins(architecture="amd64", machine="x86_64"):
    suffix = "-amd64" if architecture == "amd64" else ""
    pins = json.loads((HISTORY / ("nightly-pins" + suffix + ".json")).read_text())
    # Archived schema-1 pins provide independent historical seeds. Reports
    # derived from them here are synthetic and do not select deployment inputs.
    pins.update({"schema_version": 2,
                 "base_index": "nvcr.io/nvidia/cuda-dl-base:synthetic@sha256:" + "b" * 64,
                 "foundation_roots": list(pins["wheels"])})
    report = {"environment": {"implementation_name": "cpython", "platform_system": "Linux",
                              "sys_platform": "linux", "python_version": "3.12", "python_full_version": "3.12.3",
                              "platform_machine": machine}, "install": []}
    for name, pin in pins["wheels"].items():
        report["install"].append({"metadata": {"name": name, "version": pin["version"]},
                                 "download_info": {"url": pin["url"],
                                                   "archive_info": {"hashes": {"sha256": pin["sha256"]}}}})
    for name in ("pytest", "pluggy", "iniconfig"):
        add(report, name)
    return report, pins


def add(report, name, version="1.0", requirements=None):
    filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    report["install"].append({"metadata": {"name": name, "version": version, "requires_dist": requirements or []},
                             "download_info": {"url": f"https://files.pythonhosted.org/{filename}",
                                               "archive_info": {"hashes": {"sha256": "a" * 64}}}})


def manifest(result, scripts, architecture="amd64"):
    return json.loads(result[scripts[0].platform_pins(architecture).manifest])


@pytest.mark.parametrize("architecture,machine", [("arm64", "aarch64"), ("amd64", "x86_64")])
def test_native_report_validates_selected_platform(scripts, architecture, machine):
    report, pins = report_and_pins(architecture, machine)
    result = scripts[0].render(report, pins, architecture)
    assert len(result) == 5
    assert all("cross-platform" not in content for content in result.values())


def test_cross_report_must_be_declared_and_target_markers_checked(scripts):
    report, pins = report_and_pins(machine="aarch64")
    add(report, "parent", requirements=['child>=2; platform_machine == "x86_64"'])
    with pytest.raises(ValueError, match="--cross-report"):
        scripts[0].render(report, pins, "amd64")
    with pytest.raises(ValueError, match="omitted active target dependency"):
        scripts[0].render(report, pins, "amd64", cross_report=True)
    add(report, "child", "2.0")
    result = scripts[0].render(report, pins, "amd64", cross_report=True)
    target = scripts[0].platform_pins("amd64")
    assert all("cross-platform" in content for name, content in result.items() if name != target.manifest)
    assert manifest(result, scripts)["cross_report"] is True
    assert manifest(result, scripts)["resolver_environment"]["platform_machine"] == "aarch64"
    assert report["environment"]["platform_machine"] == "aarch64"


def test_transitive_extras_require_real_dependencies(scripts):
    report, pins = report_and_pins()
    add(report, "parent", requirements=["child[accelerator]>=1"])
    add(report, "child", requirements=['leaf==2; extra == "accelerator"'])
    with pytest.raises(ValueError, match="omitted active target dependency"):
        scripts[0].render(report, pins, "amd64")
    add(report, "leaf", "1.0")
    with pytest.raises(ValueError, match="target dependency conflict"):
        scripts[0].render(report, pins, "amd64")
    report["install"][-1]["metadata"]["version"] = "2.0"
    report["install"][-1]["download_info"]["url"] = "https://files.pythonhosted.org/leaf-2.0-py3-none-any.whl"
    scripts[0].render(report, pins, "amd64")


@pytest.mark.parametrize("bad", ["hash", "architecture", "python", "duplicate", "origin"])
def test_malformed_or_changed_distribution_is_refused(scripts, bad):
    report, pins = report_and_pins()
    if bad == "hash":
        report["install"][0]["download_info"]["archive_info"]["hashes"]["sha256"] = "0" * 64
    elif bad == "architecture":
        report["install"][0]["download_info"]["url"] = pins["wheels"]["torch"]["url"].replace("x86_64", "aarch64")
    elif bad == "python":
        report["install"][0]["download_info"]["url"] = pins["wheels"]["torch"]["url"].replace("cp312", "cp311")
    elif bad == "duplicate":
        report["install"].append(copy.deepcopy(report["install"][0]))
    else:
        report["install"][-1]["download_info"]["url"] = "https://unapproved.invalid/iniconfig-1.0-py3-none-any.whl"
    with pytest.raises(ValueError):
        scripts[0].render(report, pins, "amd64")


def test_cross_kernel_markers_need_native_environment(scripts):
    report, pins = report_and_pins(machine="aarch64")
    add(report, "parent", requirements=['leaf; platform_release == "unknown"'])
    with pytest.raises(ValueError, match="native report"):
        scripts[0].render(report, pins, "amd64", cross_report=True)


def test_arm64_audited_nightlies_remain_fixed():
    pins = json.loads((HISTORY / "nightly-pins.json").read_text())
    assert pins["base"].endswith("sha256:b2d6023181b13e4ca5bc770cbad5e2585d199cf01cb757d58896345b457f7e68")
    assert {name: value["sha256"] for name, value in pins["wheels"].items()} == {
        "torch": "f36987f1c8dc4587b43be718752ca195398017ab7cc6e2c77bbfbcae59e31f20",
        "torchvision": "35caf9593e4c168b0eb1737ed76c05675149f01e9c59289a048fd106f13f554e",
        "triton": "e52b9595189715d91916d6ceaf04e8e5f4e4b5fbea629ccbab97343318e2c357",
    }


@pytest.mark.parametrize("base_key", ["base", "base_index", None])
@pytest.mark.parametrize("architecture,machine", [("arm64", "aarch64"), ("amd64", "x86_64")])
def test_select_pins_uses_native_runtime_and_all_preconditions(scripts, tmp_path, monkeypatch,
                                                            architecture, machine, base_key):
    selector = scripts[1]
    target = selector.platform_pins(architecture)
    monkeypatch.setattr(selector.platform, "system", lambda: "Linux")
    monkeypatch.setattr(selector.platform, "machine", lambda: machine)
    monkeypatch.setattr(selector, "sys", SimpleNamespace(version_info=SimpleNamespace(major=3, minor=12)))
    report, pins = report_and_pins(architecture, machine)
    (tmp_path / target.pins).write_text(json.dumps(pins))
    contents = scripts[0].render(report, pins, architecture)
    for name, content in contents.items():
        (tmp_path / name).write_text(content)
    selector.select(architecture, tmp_path, base_image=pins[base_key] if base_key else None)
    assert json.loads((tmp_path / "nightly-pins.json").read_text())["platform"] == f"linux/{architecture}"
    for source, generic in ((target.runtime, "requirements-runtime.lock"),
                            (target.verification, "requirements-test.lock"),
                            (target.foundation, "requirements-foundation.lock"),
                            (target.constraints, "foundation-constraints.txt"),
                            (target.manifest, "foundation-manifest.json")):
        assert (tmp_path / generic).read_text() == contents[source]
    (tmp_path / target.verification).unlink()
    before = (tmp_path / "nightly-pins.json").read_bytes()
    with pytest.raises(ValueError, match="missing or oversized"):
        selector.select(architecture, tmp_path)
    assert (tmp_path / "nightly-pins.json").read_bytes() == before


def test_select_wrong_architecture_does_not_mutate_files(scripts, tmp_path, monkeypatch):
    selector = scripts[1]
    monkeypatch.setattr(selector.platform, "system", lambda: "Linux")
    monkeypatch.setattr(selector.platform, "machine", lambda: "aarch64")
    with pytest.raises(ValueError, match="requested linux/amd64"):
        selector.select("amd64", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_generated_pair_rolls_back_on_second_replace_failure(scripts, tmp_path, monkeypatch):
    for name in ("one.lock", "two.lock"):
        (tmp_path / name).write_text("original")
    replace = Path.replace

    def fail_second(source, destination):
        if Path(destination).name == "two.lock":
            raise OSError("injected second replacement failure")
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_second)
    with pytest.raises(OSError, match="injected"):
        scripts[0].write_locks({"one.lock": "changed", "two.lock": "changed"}, tmp_path)
    assert all(path.read_text() == "original" for path in tmp_path.iterdir())


def test_foundation_roles_are_disjoint_and_constraints_cover_exact_closure(scripts):
    report, pins = report_and_pins()
    report["install"][0]["metadata"]["requires_dist"] = ["shared>=1"]
    add(report, "shared")
    add(report, "application", requirements=["shared>=1"])
    result = scripts[0].render(report, pins, "amd64")
    roles = manifest(result, scripts)["roles"]
    foundation = set(pins["wheels"]) | {"shared"}
    assert set(roles["foundation"]) == foundation
    assert set(roles["runtime"]) == {"application"}
    assert set(roles["verification"]) == {"pytest", "pluggy", "iniconfig"}
    assert set().union(*(set(packages) for packages in roles.values())) == {
        item["metadata"]["name"] for item in report["install"]}
    target = scripts[0].platform_pins("amd64")
    for role, filename in (("foundation", target.foundation), ("runtime", target.runtime),
                           ("verification", target.verification)):
        records = {line.split(" @ ", 1)[0] for line in result[filename].splitlines() if not line.startswith("#")}
        assert records == set(roles[role])
    constraints = {line for line in result[target.constraints].splitlines() if not line.startswith("#")}
    assert constraints == {f"{name}==={record['version']}" for name, record in roles["foundation"].items()}
    assert roles["foundation"]["torch"]["requires_dist"] == ["shared>=1"]


def test_foundation_roots_and_transitive_extras_are_protected(scripts):
    report, pins = report_and_pins()
    pins["foundation_roots"][0] = "torch[accelerator]"
    report["install"][0]["metadata"]["requires_dist"] = ['child[transitive]; extra == "accelerator"']
    add(report, "child", requirements=['nvidia-cublas-cu13==13.4.3; extra == "transitive"'])
    add(report, "nvidia-cublas-cu13", "13.4.3")
    add(report, "unrelated")
    roles = manifest(scripts[0].render(report, pins, "amd64"), scripts)["roles"]
    assert {"child", "nvidia-cublas-cu13"} <= roles["foundation"].keys()
    assert set(roles["runtime"]) == {"unrelated"}


@pytest.mark.parametrize("name", ["nvidia-cublas-cu13", "cuda-bindings", "pytorch-triton", "triton-custom", "torchaudio"])
def test_application_cannot_introduce_an_unprotected_accelerator_provider(scripts, name):
    report, pins = report_and_pins()
    add(report, "application", requirements=[name])
    add(report, name)
    with pytest.raises(ValueError, match="outside protected foundation"):
        scripts[0].render(report, pins, "amd64")


@pytest.mark.parametrize("roots", [None, [], ["torch", "torchvision"], ["torch<1", "torchvision", "triton"],
                                   ['torch; platform_machine == "aarch64"', "torchvision", "triton"]])
def test_missing_inactive_or_conflicting_foundation_roots_are_refused(scripts, roots):
    report, pins = report_and_pins()
    pins["foundation_roots"] = roots
    with pytest.raises(ValueError):
        scripts[0].render(report, pins, "amd64")


def test_foundation_root_extra_missing_dependency_is_still_refused(scripts):
    report, pins = report_and_pins()
    pins["foundation_roots"][0] = "torch[extra]"
    report["install"][0]["metadata"]["requires_dist"] = ['missing; extra == "extra"']
    with pytest.raises(ValueError, match="omitted active target dependency"):
        scripts[0].render(report, pins, "amd64")


def test_foundation_cannot_contain_verification_only_packages(scripts):
    report, pins = report_and_pins()
    report["install"][0]["metadata"]["requires_dist"] = ["pytest"]
    with pytest.raises(ValueError, match="verification-only"):
        scripts[0].render(report, pins, "amd64")


def test_runtime_cannot_depend_on_verification_only_package(scripts):
    report, pins = report_and_pins()
    add(report, "application", requirements=["pytest"])
    with pytest.raises(ValueError, match="verification-only packages cannot be runtime"):
        scripts[0].render(report, pins, "amd64")


def test_transitive_reach_does_not_replace_required_explicit_foundation_root(scripts):
    report, pins = report_and_pins()
    pins["foundation_roots"] = ["torchvision", "triton"]
    report["install"][1]["metadata"]["requires_dist"] = ["torch"]
    with pytest.raises(ValueError, match="foundation_roots omitted"):
        scripts[0].render(report, pins, "amd64")


@pytest.mark.parametrize("name", ["torch", "torchvision", "triton"])
@pytest.mark.parametrize("metadata", [{"source": {}}, {"artifact": "local.whl"},
                                      {"source": {}, "artifact": "local.whl"}])
def test_source_and_artifact_pin_metadata_is_rejected(scripts, name, metadata):
    report, pins = report_and_pins()
    pins["wheels"][name].update(metadata)
    with pytest.raises(ValueError):
        scripts[0].render(report, pins, "amd64")


def test_official_binary_preserves_actual_requires_dist_validation(scripts):
    report, pins = report_and_pins()
    report["install"][0]["metadata"]["requires_dist"] = ["nvidia-cuda-runtime-cu13==13.4.1"]
    add(report, "nvidia-cuda-runtime-cu13", "13.4.3")
    with pytest.raises(ValueError, match="target dependency conflict"):
        scripts[0].render(report, pins, "amd64")


def test_manifest_records_target_and_report_provenance_deterministically(scripts):
    report, pins = report_and_pins()
    result = scripts[0].render(report, pins, "amd64")
    actual = manifest(result, scripts)
    assert actual["schema_version"] == 1
    assert {key: actual[key] for key in ("platform", "python", "base", "base_index", "cuda", "cache_key")} == {
        key: pins[key] for key in ("platform", "python", "base", "base_index", "cuda", "cache_key")}
    assert actual["foundation_roots"] == pins["foundation_roots"]
    assert actual["resolver_environment"] == report["environment"]
    assert actual["resolver_report_sha256"] == hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert actual["resolver_report_hash_format"] == "canonical-json"
    reordered = dict(reversed(list(report.items())))
    assert scripts[0].render(reordered, pins, "amd64") == result
    raw = scripts[0].render(report, pins, "amd64", report_sha256="c" * 64)
    assert manifest(raw, scripts)["resolver_report_sha256"] == "c" * 64
    assert manifest(raw, scripts)["resolver_report_hash_format"] == "raw-file"


@pytest.mark.parametrize("bad", ["full_python", "requires_python", "requires_dist", "missing_nightly", "missing_test",
                                 "direct_dependency", "direct_root"])
def test_full_report_contract_remains_required(scripts, bad):
    report, pins = report_and_pins()
    if bad == "full_python":
        report["environment"]["python_full_version"] = "3.13.1"
    elif bad == "requires_python":
        report["install"][0]["metadata"]["requires_python"] = ">=3.13"
    elif bad == "requires_dist":
        report["install"][0]["metadata"]["requires_dist"] = {}
    elif bad == "missing_nightly":
        report["install"].pop(0)
    elif bad == "missing_test":
        report["install"].pop()
    elif bad == "direct_dependency":
        add(report, "parent", requirements=["pytest @ https://files.pythonhosted.org/pytest-1.0-py3-none-any.whl"])
    else:
        pins["foundation_roots"][0] = "torch @ " + pins["wheels"]["torch"]["url"]
    with pytest.raises(ValueError):
        scripts[0].render(report, pins, "amd64")


def test_cli_writes_all_five_artifacts_and_hashes_original_report(scripts, tmp_path, monkeypatch, capsys):
    generator = scripts[0]
    report, pins = report_and_pins()
    target = generator.platform_pins("amd64")
    report_path = tmp_path / "report.json"
    report_bytes = json.dumps(report, indent=3).encode() + b"\n"
    report_path.write_bytes(report_bytes)
    (tmp_path / target.pins).write_text(json.dumps(pins))
    monkeypatch.setattr(generator, "HERE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["lock_dependencies.py", "--report", str(report_path), "--architecture", "amd64"])
    generator.main()
    contents = {name: (tmp_path / name).read_text() for name in (
        target.foundation, target.runtime, target.verification, target.constraints, target.manifest)}
    assert manifest(contents, scripts)["resolver_report_sha256"] == hashlib.sha256(report_bytes).hexdigest()
    assert manifest(contents, scripts)["base"] == pins["base"]
    assert json.loads((tmp_path / target.pins).read_text()) == pins
    assert "3 foundation, 0 runtime and 3 test packages" in capsys.readouterr().out


def test_five_artifact_generation_restores_originals_and_removes_new_files_on_failure(scripts, tmp_path, monkeypatch):
    report, pins = report_and_pins()
    contents = scripts[0].render(report, pins, "amd64")
    filenames = list(contents)
    (tmp_path / filenames[0]).write_text("original foundation")
    replace = Path.replace

    def fail_manifest(source, destination):
        if Path(destination).name == filenames[-1]:
            raise OSError("injected manifest replacement failure")
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_manifest)
    with pytest.raises(OSError, match="injected manifest"):
        scripts[0].write_locks(contents, tmp_path)
    assert {path.name: path.read_text() for path in tmp_path.iterdir()} == {filenames[0]: "original foundation"}


@pytest.mark.parametrize("bad", ["base", "python", "missing", "oversized", "schema"])
def test_select_validates_every_input_before_copying(scripts, tmp_path, monkeypatch, bad):
    selector = scripts[1]
    report, pins = report_and_pins()
    target = selector.platform_pins("amd64")
    monkeypatch.setattr(selector.platform, "system", lambda: "Linux")
    monkeypatch.setattr(selector.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(selector, "sys", SimpleNamespace(version_info=SimpleNamespace(major=3, minor=12)))
    for name, content in scripts[0].render(report, pins, "amd64").items():
        (tmp_path / name).write_text(content)
    if bad == "python":
        monkeypatch.setattr(selector, "sys", SimpleNamespace(version_info=SimpleNamespace(major=3, minor=13)))
    elif bad == "missing":
        (tmp_path / target.manifest).unlink()
    elif bad == "oversized":
        (tmp_path / target.constraints).write_bytes(b"x" * (2 * 2**20 + 1))
    elif bad == "schema":
        pins["schema_version"] = 1
    (tmp_path / target.pins).write_text(json.dumps(pins))
    for name in ("nightly-pins.json", "requirements-foundation.lock", "requirements-runtime.lock",
                 "requirements-test.lock", "foundation-constraints.txt", "foundation-manifest.json"):
        (tmp_path / name).write_text("unchanged")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        selector.select("amd64", tmp_path, base_image="wrong" if bad == "base" else pins["base"])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def native_addon():
    # Identity-only synthetic record; no acquisition or installation occurs.
    artifact = "libnative-addon_1.2.3-1_amd64.deb"
    return {"artifact": artifact, "sha256": "a" * 64, "package": "libnative-addon", "version": "1.2.3-1",
            "url": "https://developer.download.nvidia.com/synthetic/" + artifact}


def test_native_addons_default_to_empty_and_preserve_exact_validated_records(scripts):
    report, pins = report_and_pins()
    assert manifest(scripts[0].render(report, pins, "amd64"), scripts)["native_addons"] == []
    addon = native_addon()
    pins["native_addons"] = [addon]
    assert manifest(scripts[0].render(report, pins, "amd64"), scripts)["native_addons"] == [addon]
    del addon["url"]
    assert manifest(scripts[0].render(report, pins, "amd64"), scripts)["native_addons"] == [addon]


def test_native_addon_record_count_is_bounded(scripts):
    report, pins = report_and_pins()
    pins["native_addons"] = [{"artifact": f"native-addon-{index}.deb", "sha256": "a" * 64,
                             "package": f"native-addon-{index}", "version": "1.0"} for index in range(32)]
    assert len(manifest(scripts[0].render(report, pins, "amd64"), scripts)["native_addons"]) == 32
    pins["native_addons"].append({"artifact": "native-addon-32.deb", "sha256": "a" * 64,
                                  "package": "native-addon-32", "version": "1.0"})
    with pytest.raises(ValueError, match="at most 32"):
        scripts[0].render(report, pins, "amd64")


@pytest.mark.parametrize("field,value", [
    ("artifact", "../addon.deb"), ("artifact", "addon.whl"), ("artifact", "addon.deb\n"),
    ("artifact", ".addon.deb"), ("artifact", None),
    ("sha256", "a" * 63), ("sha256", "A" * 64), ("sha256", None),
    ("package", "A-valid-addon"), ("package", "-invalid-addon"), ("package", "a"),
    ("package", "addon:amd64"), ("package", "addon\nmalicious"), ("package", None),
    ("version", ""), ("version", "1" * 257), ("version", "1.0\n2.0"), ("version", "1.0 --option"),
    ("version", None), ("unexpected", "field"),
])
def test_native_addon_identity_boundary_rejects_invalid_records(scripts, field, value):
    report, pins = report_and_pins()
    addon = native_addon()
    addon[field] = value
    pins["native_addons"] = [addon]
    with pytest.raises(ValueError):
        scripts[0].render(report, pins, "amd64")


@pytest.mark.parametrize("entries", [None, {}, [None], [{}]])
def test_native_addons_require_a_list_of_complete_records(scripts, entries):
    report, pins = report_and_pins()
    pins["native_addons"] = entries
    with pytest.raises(ValueError):
        scripts[0].render(report, pins, "amd64")


@pytest.mark.parametrize("field", ["artifact", "package"])
def test_native_addon_package_and_artifact_identities_are_unique(scripts, field):
    report, pins = report_and_pins()
    first, second = native_addon(), native_addon()
    second.update({"package": "other-addon", "artifact": "other-addon.deb"})
    second.pop("url")
    second[field] = first[field]
    pins["native_addons"] = [first, second]
    with pytest.raises(ValueError, match="duplicate native addon"):
        scripts[0].render(report, pins, "amd64")


@pytest.mark.parametrize("url", [
    "http://developer.download.nvidia.com/synthetic/libnative-addon_1.2.3-1_amd64.deb",
    "file:///temporary/libnative-addon_1.2.3-1_amd64.deb",
    "https://unapproved.invalid/libnative-addon_1.2.3-1_amd64.deb",
    "https://user:secret@developer.download.nvidia.com/libnative-addon_1.2.3-1_amd64.deb",
    "https://developer.download.nvidia.com/other.deb",
    "https://developer.download.nvidia.com/%20/libnative-addon_1.2.3-1_amd64.deb",
    "https://developer.download.nvidia.com/libnative-addon_1.2.3-1_amd64.deb#other",
    None,
])
def test_native_addon_acquisition_sources_must_be_official_https(scripts, url):
    report, pins = report_and_pins()
    addon = native_addon()
    addon["url"] = url
    pins["native_addons"] = [addon]
    with pytest.raises(ValueError, match="official HTTPS"):
        scripts[0].render(report, pins, "amd64")


@pytest.fixture
def selection_bundle(scripts, tmp_path, monkeypatch):
    selector = scripts[1]
    report, pins = report_and_pins()
    add(report, "application")
    pins["native_addons"] = [native_addon()]
    target = selector.platform_pins("amd64")
    contents = scripts[0].render(report, pins, "amd64")
    for name, content in contents.items():
        (tmp_path / name).write_text(content)
    (tmp_path / target.pins).write_text(json.dumps(pins))
    for name in ("nightly-pins.json", "requirements-foundation.lock", "requirements-runtime.lock",
                 "requirements-test.lock", "foundation-constraints.txt", "foundation-manifest.json"):
        (tmp_path / name).write_text("unpublished")
    monkeypatch.setattr(selector.platform, "system", lambda: "Linux")
    monkeypatch.setattr(selector.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(selector, "sys", SimpleNamespace(version_info=SimpleNamespace(major=3, minor=12)))
    return selector, target, pins, contents


@pytest.mark.parametrize("field", ["platform", "python", "base", "base_index", "cuda", "cache_key",
                                  "foundation_roots", "native_addons", "nccl_provider"])
def test_selection_rejects_manifest_identity_mismatch_before_copying(selection_bundle, tmp_path, field):
    selector, target, pins, contents = selection_bundle
    selected = json.loads(contents[target.manifest])
    selected[field] = [] if isinstance(selected[field], list) else "different"
    (tmp_path / target.manifest).write_text(json.dumps(selected))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="manifest differs from pins"):
        selector.select("amd64", tmp_path, base_image=pins["base"])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("field", ["base", "base_index", "cuda", "cache_key", "foundation_roots", "native_addons",
                                  "nccl_provider", "wheel_version", "wheel_sha256", "wheel_url"])
def test_old_consistent_artifacts_cannot_accompany_new_valid_pins(selection_bundle, tmp_path, field):
    selector, target, pins, _ = selection_bundle
    if field in {"base", "base_index"}:
        pins[field] = pins[field].rsplit("@", 1)[0] + "@sha256:" + "1" * 64
    elif field in {"cuda", "cache_key"}:
        pins[field] = "different"
    elif field == "foundation_roots":
        pins[field].append("another-root")
    elif field == "native_addons":
        pins[field][0]["sha256"] = "1" * 64
    elif field == "nccl_provider":
        pins[field] = "nvidia-nccl-cu13"
    else:
        pin = pins["wheels"]["torch"]
        if field == "wheel_version":
            pin["url"] = pin["url"].replace(pin["version"].replace("+", "%2B"), "99.0")
            pin["version"] = "99.0"
        elif field == "wheel_sha256":
            pin["sha256"] = "1" * 64
        else:
            pin["url"] = pin["url"].replace("download.pytorch.org", "download-r2.pytorch.org")
    (tmp_path / target.pins).write_text(json.dumps(pins))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="differs from pins"):
        selector.select("amd64", tmp_path, base_image=pins["base"])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("role", ["foundation", "runtime", "verification"])
def test_selection_binds_every_role_lock_to_its_manifest(selection_bundle, tmp_path, role):
    selector, target, pins, _ = selection_bundle
    filename = {"foundation": target.foundation, "runtime": target.runtime, "verification": target.verification}[role]
    path = tmp_path / filename
    path.write_text(path.read_text() + "--extra-index-url https://unapproved.invalid\n")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="lock differs from manifest role"):
        selector.select("amd64", tmp_path, base_image=pins["base"])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("bad", ["missing", "version", "duplicate", "weaker_operator", "extra"])
def test_selection_requires_exact_protected_constraints(selection_bundle, tmp_path, bad):
    selector, target, pins, _ = selection_bundle
    path = tmp_path / target.constraints
    constraints = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    if bad == "missing":
        constraints.pop()
    elif bad == "version":
        constraints[0] = constraints[0].split("===", 1)[0] + "===99.0"
    elif bad == "duplicate":
        constraints.append(constraints[0])
    elif bad == "weaker_operator":
        constraints[0] = constraints[0].replace("===", "==")
    else:
        constraints.append("unprotected===1.0")
    path.write_text("\n".join(constraints) + "\n")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="constraints differ"):
        selector.select("amd64", tmp_path, base_image=pins["base"])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("bad", ["schema", "schema_bool", "role", "duplicate", "missing_wheel", "identity", "origin"])
def test_selection_rejects_malformed_manifest_schema(selection_bundle, tmp_path, bad):
    selector, target, pins, contents = selection_bundle
    selected = json.loads(contents[target.manifest])
    if bad in {"schema", "schema_bool"}:
        selected["schema_version"] = True if bad == "schema_bool" else 2
    elif bad == "role":
        selected["roles"]["unexpected"] = {}
    elif bad == "duplicate":
        selected["roles"]["runtime"]["torch"] = selected["roles"]["foundation"]["torch"]
    elif bad == "missing_wheel":
        selected["roles"]["foundation"].pop("torch")
        (tmp_path / target.foundation).write_text("".join(
            line + "\n" for line in contents[target.foundation].splitlines() if not line.startswith("torch @ ")))
    elif bad == "identity":
        selected["roles"]["runtime"]["application"]["sha256"] = "invalid"
    else:
        selected["roles"]["runtime"]["application"]["url"] = "file:///unapproved/application.whl"
    (tmp_path / target.manifest).write_text(json.dumps(selected))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        selector.select("amd64", tmp_path, base_image=pins["base"])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("name", ["torch", "torchvision", "triton"])
@pytest.mark.parametrize("field", ["version", "url", "sha256", "source", "artifact"])
def test_selection_binds_every_audited_wheel_identity_field(selection_bundle, tmp_path, name, field):
    selector, target, pins, contents = selection_bundle
    selected = json.loads(contents[target.manifest])
    record = selected["roles"]["foundation"][name]
    record[field] = {"version": "99.0", "url": "https://download.pytorch.org/different.whl",
                     "sha256": "1" * 64, "source": {"commit": "1" * 40}, "artifact": "different.whl"}[field]
    (tmp_path / target.manifest).write_text(json.dumps(selected))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match=("unsupported source/artifact" if field in {"source", "artifact"}
                                          else "foundation wheel differs from pins")):
        selector.select("amd64", tmp_path, base_image=pins["base"])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_selection_publishes_validated_snapshots(selection_bundle, tmp_path, monkeypatch):
    selector, target, pins, contents = selection_bundle
    validate = selector._validate_artifacts

    def change_sources_after_validation(*args):
        validate(*args)
        for filename in contents:
            (tmp_path / filename).write_text("changed after snapshot")
        (tmp_path / target.pins).write_text("changed after snapshot")

    monkeypatch.setattr(selector, "_validate_artifacts", change_sources_after_validation)
    selector.select("amd64", tmp_path, base_image=pins["base"])
    assert json.loads((tmp_path / "nightly-pins.json").read_text()) == pins
    for filename, generic in ((target.foundation, "requirements-foundation.lock"),
                              (target.runtime, "requirements-runtime.lock"),
                              (target.verification, "requirements-test.lock"),
                              (target.constraints, "foundation-constraints.txt"),
                              (target.manifest, "foundation-manifest.json")):
        assert (tmp_path / generic).read_text() == contents[filename]


@pytest.mark.parametrize("host", ["download.pytorch.org", "download-r2.pytorch.org"])
def test_exact_official_binary_hosts_generate_and_select(scripts, tmp_path, monkeypatch, host):
    report, pins = report_and_pins()
    for pin in pins["wheels"].values():
        pin["url"] = pin["url"].replace("download.pytorch.org", host)
    for item in report["install"][:3]:
        item["download_info"]["url"] = item["download_info"]["url"].replace("download.pytorch.org", host)
    add(report, "runtime-wheel")
    report["install"][-1]["download_info"]["url"] = report["install"][-1]["download_info"]["url"].replace(
        "files.pythonhosted.org", host)
    result = scripts[0].render(report, pins, "amd64")
    records = manifest(result, scripts)["roles"]
    assert all(record["url"].startswith("https://" + host + "/")
               for record in records["foundation"].values())
    assert records["runtime"]["runtime-wheel"]["url"].startswith("https://" + host + "/")
    selector = scripts[1]
    target = selector.platform_pins("amd64")
    for filename, content in result.items():
        (tmp_path / filename).write_text(content)
    (tmp_path / target.pins).write_text(json.dumps(pins))
    monkeypatch.setattr(selector.platform, "system", lambda: "Linux")
    monkeypatch.setattr(selector.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(selector, "sys", SimpleNamespace(version_info=SimpleNamespace(major=3, minor=12)))
    selector.select("amd64", tmp_path, base_image=pins["base"])
    assert (tmp_path / "requirements-foundation.lock").read_text() == result[target.foundation]
    assert (tmp_path / "requirements-runtime.lock").read_text() == result[target.runtime]


@pytest.mark.parametrize("origin", ["file:///temporary/", "file://remote-host/temporary/",
                                   "ftp://download.pytorch.org/", "http://download.pytorch.org/",
                                   "https://download-r2.pytorch.org.unapproved.invalid/",
                                   "https://nightly.pytorch.org/", "https://files.pythonhosted.org/",
                                   "https://pypi.nvidia.com/",
                                   "https://user:secret@download-r2.pytorch.org/"])
def test_binary_report_origin_cannot_be_promoted_to_an_official_pin(scripts, origin):
    report, pins = report_and_pins()
    item = report["install"][0]
    filename = item["download_info"]["url"].rsplit("/", 1)[-1]
    item["download_info"]["url"] = origin + filename
    # Filename/version/hash still match. The original report origin must be
    # checked before canonicalizing to the audited official binary URL.
    with pytest.raises(ValueError, match="unapproved dependency origin"):
        scripts[0].render(report, pins, "amd64")


@pytest.mark.parametrize("origin", ["https://download-r2.pytorch.org.unapproved.invalid/",
                                   "https://other.pytorch.org/", "http://download-r2.pytorch.org/"])
def test_official_redirect_allowlist_does_not_authorize_related_hosts(scripts, origin):
    report, pins = report_and_pins()
    add(report, "runtime-wheel")
    filename = report["install"][-1]["download_info"]["url"].rsplit("/", 1)[-1]
    report["install"][-1]["download_info"]["url"] = origin + filename
    with pytest.raises(ValueError, match="unapproved dependency origin"):
        scripts[0].render(report, pins, "amd64")


def test_cli_uses_explicit_pins_and_external_output_directory(scripts, tmp_path, monkeypatch):
    generator = scripts[0]
    report, pins = report_and_pins()
    target = generator.platform_pins("amd64")
    repository = tmp_path / "repository"
    repository.mkdir()
    historical = repository / target.pins
    historical.write_text("historical pins must not be read or replaced")
    explicit = tmp_path / "selected-pins.json"
    explicit.write_text(json.dumps(pins))
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    output = tmp_path / "managed-build-context" / "pins"
    monkeypatch.setattr(generator, "HERE", repository)
    monkeypatch.setattr(sys, "argv", ["lock_dependencies.py", "--architecture", "amd64",
                                     "--report", str(report_path), "--pins", str(explicit),
                                     "--output-directory", str(output)])
    generator.main()
    assert {path.name for path in output.iterdir()} == {
        target.foundation, target.runtime, target.verification, target.constraints, target.manifest}
    selected = json.loads((output / target.manifest).read_text())
    assert selected["base"] == pins["base"]
    assert selected["foundation_roots"] == pins["foundation_roots"]
    assert selected["resolver_report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert historical.read_text() == "historical pins must not be read or replaced"
    assert list(repository.iterdir()) == [historical]
    assert json.loads(explicit.read_text()) == pins


@pytest.mark.parametrize("oversized", ["report", "pins"])
def test_cli_bounds_external_inputs_before_creating_output(scripts, tmp_path, monkeypatch, oversized):
    generator = scripts[0]
    report, pins = report_and_pins()
    report_path, pin_path = tmp_path / "report.json", tmp_path / "pins.json"
    report_path.write_text(json.dumps(report))
    pin_path.write_text(json.dumps(pins))
    (report_path if oversized == "report" else pin_path).write_bytes(b"x" * (2 * 2**20 + 1))
    output = tmp_path / "managed-build-context"
    monkeypatch.setattr(sys, "argv", ["lock_dependencies.py", "--architecture", "amd64",
                                     "--report", str(report_path), "--pins", str(pin_path),
                                     "--output-directory", str(output)])
    with pytest.raises(ValueError, match="2 MiB audit limit"):
        generator.main()
    assert not output.exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_native_nccl_provider_is_explicit_in_generated_manifest(scripts, explicit):
    report, pins = report_and_pins()
    if explicit:
        pins["nccl_provider"] = "native"
    result = scripts[0].render(report, pins, "amd64")
    assert manifest(result, scripts)["nccl_provider"] == "native"


@pytest.mark.parametrize("protection", ["root", "transitive"])
def test_selected_nccl_wheel_is_protected_and_selected(scripts, tmp_path, monkeypatch, protection):
    report, pins = report_and_pins()
    pins["nccl_provider"] = "nvidia-nccl-cu13"
    if protection == "root":
        pins["foundation_roots"].append("nvidia-nccl-cu13==1.0")
    else:
        report["install"][0]["metadata"]["requires_dist"] = ["nvidia-nccl-cu13==1.0"]
    add(report, "nvidia-nccl-cu13")
    result = scripts[0].render(report, pins, "amd64")
    selected = manifest(result, scripts)
    assert selected["nccl_provider"] == "nvidia-nccl-cu13"
    assert "nvidia-nccl-cu13" in selected["roles"]["foundation"]
    selector = scripts[1]
    target = selector.platform_pins("amd64")
    for filename, content in result.items():
        (tmp_path / filename).write_text(content)
    (tmp_path / target.pins).write_text(json.dumps(pins))
    monkeypatch.setattr(selector.platform, "system", lambda: "Linux")
    monkeypatch.setattr(selector.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(selector, "sys", SimpleNamespace(version_info=SimpleNamespace(major=3, minor=12)))
    selector.select("amd64", tmp_path, base_image=pins["base"])
    assert json.loads((tmp_path / "foundation-manifest.json").read_text())["nccl_provider"] == "nvidia-nccl-cu13"


@pytest.mark.parametrize("present", [False, True])
def test_selected_nccl_wheel_requires_protected_foundation_membership(scripts, present):
    report, pins = report_and_pins()
    pins["nccl_provider"] = "nvidia-nccl-cu13"
    if present:
        add(report, "nvidia-nccl-cu13")
    with pytest.raises(ValueError, match=r"NCCL wheel provider.*protected foundation closure"):
        scripts[0].render(report, pins, "amd64")


@pytest.mark.parametrize("bad", ["missing_provider", "missing_wheel"])
def test_selector_checks_nccl_binding_before_publication(selection_bundle, tmp_path, bad):
    selector, target, pins, contents = selection_bundle
    selected = json.loads(contents[target.manifest])
    if bad == "missing_provider":
        selected.pop("nccl_provider")
    else:
        pins["nccl_provider"] = selected["nccl_provider"] = "nvidia-nccl-cu13"
        (tmp_path / target.pins).write_text(json.dumps(pins))
    (tmp_path / target.manifest).write_text(json.dumps(selected))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match=r"nccl_provider|NCCL wheel provider"):
        selector.select("amd64", tmp_path, base_image=pins["base"])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("architecture,machine", [("arm64", "aarch64"), ("amd64", "x86_64")])
@pytest.mark.parametrize("name", ["cuda-toolkit", "nvidia-cudnn-cu13", "NVIDIA_CUDA_RUNTIME"])
def test_nvidia_index_accepts_only_canonical_vendor_namespaces(scripts, tmp_path, monkeypatch,
                                                              architecture, machine, name):
    report, pins = report_and_pins(architecture, machine)
    report["install"][0]["metadata"]["requires_dist"] = [name + "==1.0"]
    add(report, name)
    item = report["install"][-1]
    item["download_info"]["url"] = item["download_info"]["url"].replace("files.pythonhosted.org", "pypi.nvidia.com")
    before = copy.deepcopy(report)
    contents = scripts[0].render(report, pins, architecture)
    selected = manifest(contents, scripts, architecture)
    canonical = name.lower().replace("_", "-")
    assert selected["roles"]["foundation"][canonical]["url"] == item["download_info"]["url"]
    assert report == before
    selector = scripts[1]
    target = selector.platform_pins(architecture)
    for filename, content in contents.items():
        (tmp_path / filename).write_text(content)
    (tmp_path / target.pins).write_text(json.dumps(pins))
    monkeypatch.setattr(selector.platform, "system", lambda: "Linux")
    monkeypatch.setattr(selector.platform, "machine", lambda: machine)
    monkeypatch.setattr(selector, "sys", SimpleNamespace(version_info=SimpleNamespace(major=3, minor=12)))
    selector.select(architecture, tmp_path, base_image=pins["base"])
    assert (tmp_path / "requirements-foundation.lock").read_text() == contents[target.foundation]


@pytest.mark.parametrize("name", ["ordinary", "cudahelper", "nvidiahelper", "nvidia"])
def test_nvidia_index_rejects_nonnamespace_packages_in_generator_and_selector(scripts, tmp_path, monkeypatch, name):
    report, pins = report_and_pins()
    add(report, name)
    contents = scripts[0].render(report, pins, "amd64")
    item = report["install"][-1]
    item["download_info"]["url"] = item["download_info"]["url"].replace("files.pythonhosted.org", "pypi.nvidia.com")
    with pytest.raises(ValueError, match="unapproved dependency origin"):
        scripts[0].render(report, pins, "amd64")
    selector = scripts[1]
    target = selector.platform_pins("amd64")
    selected = json.loads(contents[target.manifest])
    selected["roles"]["runtime"][name]["url"] = item["download_info"]["url"]
    contents[target.manifest] = json.dumps(selected)
    contents[target.runtime] = contents[target.runtime].replace("files.pythonhosted.org", "pypi.nvidia.com")
    for filename, content in contents.items():
        (tmp_path / filename).write_text(content)
    (tmp_path / target.pins).write_text(json.dumps(pins))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    monkeypatch.setattr(selector.platform, "system", lambda: "Linux")
    monkeypatch.setattr(selector.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(selector, "sys", SimpleNamespace(version_info=SimpleNamespace(major=3, minor=12)))
    with pytest.raises(ValueError, match="invalid selected distribution origin"):
        selector.select("amd64", tmp_path, base_image=pins["base"])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("origin,suffix", [
    ("http://pypi.nvidia.com/", ""), ("https://pypi.nvidia.com.unapproved.invalid/", ""),
    ("https://other.nvidia.com/", ""), ("https://user:secret@pypi.nvidia.com/", ""),
    ("https://pypi.nvidia.com:444/", ""), ("https://pypi.nvidia.com:80/", ""),
    ("https://pypi.nvidia.com/encoded%20space/", ""), ("https://pypi.nvidia.com/", "#fragment"),
])
def test_nvidia_namespace_does_not_authorize_unsafe_urls(scripts, tmp_path, monkeypatch, origin, suffix):
    report, pins = report_and_pins()
    report["install"][0]["metadata"]["requires_dist"] = ["nvidia-cuda-runtime"]
    add(report, "nvidia-cuda-runtime")
    contents = scripts[0].render(report, pins, "amd64")
    item = report["install"][-1]
    original = item["download_info"]["url"]
    item["download_info"]["url"] = origin + original.rsplit("/", 1)[-1] + suffix
    with pytest.raises(ValueError):
        scripts[0].render(report, pins, "amd64")
    selector = scripts[1]
    target = selector.platform_pins("amd64")
    selected = json.loads(contents[target.manifest])
    selected["roles"]["foundation"]["nvidia-cuda-runtime"]["url"] = item["download_info"]["url"]
    contents[target.manifest] = json.dumps(selected)
    contents[target.foundation] = contents[target.foundation].replace(original, item["download_info"]["url"])
    for filename, content in contents.items():
        (tmp_path / filename).write_text(content)
    (tmp_path / target.pins).write_text(json.dumps(pins))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    monkeypatch.setattr(selector.platform, "system", lambda: "Linux")
    monkeypatch.setattr(selector.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(selector, "sys", SimpleNamespace(version_info=SimpleNamespace(major=3, minor=12)))
    with pytest.raises(ValueError):
        selector.select("amd64", tmp_path, base_image=pins["base"])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("name", ["cuda-bindings", "cuda-runtime", "cuda-other"])
def test_nvidia_index_cuda_exception_is_exact_toolkit_only(scripts, tmp_path, monkeypatch, name):
    report, pins = report_and_pins()
    report["install"][0]["metadata"]["requires_dist"] = [name]
    add(report, name)
    contents = scripts[0].render(report, pins, "amd64")
    original = report["install"][-1]["download_info"]["url"]
    rejected = original.replace("files.pythonhosted.org", "pypi.nvidia.com")
    report["install"][-1]["download_info"]["url"] = rejected
    with pytest.raises(ValueError, match="unapproved dependency origin"):
        scripts[0].render(report, pins, "amd64")
    selector = scripts[1]
    target = selector.platform_pins("amd64")
    selected = json.loads(contents[target.manifest])
    selected["roles"]["foundation"][name]["url"] = rejected
    contents[target.manifest] = json.dumps(selected)
    contents[target.foundation] = contents[target.foundation].replace(original, rejected)
    for filename, content in contents.items():
        (tmp_path / filename).write_text(content)
    (tmp_path / target.pins).write_text(json.dumps(pins))
    monkeypatch.setattr(selector.platform, "system", lambda: "Linux")
    monkeypatch.setattr(selector.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(selector, "sys", SimpleNamespace(version_info=SimpleNamespace(major=3, minor=12)))
    with pytest.raises(ValueError, match="distribution origin"):
        selector.select("amd64", tmp_path, base_image=pins["base"])
