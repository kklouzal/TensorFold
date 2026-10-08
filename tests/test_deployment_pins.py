"""Target wheel/marker validation and deployment artifact selection boundaries."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "deploy/gb10"


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
    pins = json.loads((DEPLOY / ("nightly-pins" + suffix + ".json")).read_text())
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
    report["install"].append({"metadata": {"name": name, "version": version, "requires_dist": requirements or []},
                             "download_info": {"url": f"https://files.pythonhosted.org/{name}-{version}-py3-none-any.whl",
                                               "archive_info": {"hashes": {"sha256": "a" * 64}}}})


@pytest.mark.parametrize("architecture,machine", [("arm64", "aarch64"), ("amd64", "x86_64")])
def test_native_report_validates_selected_platform(scripts, architecture, machine):
    report, pins = report_and_pins(architecture, machine)
    result = scripts[0].render(report, pins, architecture)
    assert len(result) == 2
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
    assert all("cross-platform" in content for content in result.values())
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
    pins = json.loads((DEPLOY / "nightly-pins.json").read_text())
    assert pins["base"].endswith("sha256:b2d6023181b13e4ca5bc770cbad5e2585d199cf01cb757d58896345b457f7e68")
    assert {name: value["sha256"] for name, value in pins["wheels"].items()} == {
        "torch": "f36987f1c8dc4587b43be718752ca195398017ab7cc6e2c77bbfbcae59e31f20",
        "torchvision": "35caf9593e4c168b0eb1737ed76c05675149f01e9c59289a048fd106f13f554e",
        "triton": "e52b9595189715d91916d6ceaf04e8e5f4e4b5fbea629ccbab97343318e2c357",
    }


@pytest.mark.parametrize("architecture,machine", [("arm64", "aarch64"), ("amd64", "x86_64")])
def test_select_pins_uses_native_runtime_and_all_preconditions(scripts, tmp_path, monkeypatch, architecture, machine):
    selector = scripts[1]
    target = selector.platform_pins(architecture)
    monkeypatch.setattr(selector.platform, "system", lambda: "Linux")
    monkeypatch.setattr(selector.platform, "machine", lambda: machine)
    for name in (target.pins, target.runtime, target.verification):
        (tmp_path / name).write_bytes((DEPLOY / name).read_bytes())
    selector.select(architecture, tmp_path)
    assert json.loads((tmp_path / "nightly-pins.json").read_text())["platform"] == f"linux/{architecture}"
    assert (tmp_path / "requirements-runtime.lock").read_bytes() == (DEPLOY / target.runtime).read_bytes()
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
