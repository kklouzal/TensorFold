"""Prepared extension admission, exact recipes and process-lifetime ownership."""
import hashlib
import json
import threading
from pathlib import Path

import pytest

from tensorfold.cuda import build
from tensorfold.cuda import native_artifact as native


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    package, directory = tmp_path / "package", tmp_path / "bundle"
    package.mkdir()
    directory.mkdir()
    source = package / "kernel.cu"
    source.write_text("// owned source\n")
    options = {"extra_cuda_cflags": ["-O3", "-gencode=arch=compute_121,code=sm_121"], "verbose": False}
    abi = {"host_machine": "fixture", "python_soabi": "fixture", "torch_version": "fixture"}
    manifest = {"schema_version": 1, "kind": "tensorfold-native-artifacts", "identity": {"wheel": "fixture"},
                "abi": abi, "package_root": str(package), "target_capability": [12, 1], "entries": {}}
    for name in ("first_sm_121", "second_sm_121", "third_sm_121a"):
        binary = directory / (name + ".so")
        binary.write_bytes(b"sealed generated output")
        manifest["entries"][name] = {"module_name": name, "binary": binary.name,
                                     "request": native.request(package, [str(source)], options)}
    path = directory / "manifest.json"

    def publish():
        path.write_bytes(native.canonical(manifest))
        activation = {"manifest": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "identity": manifest["identity"]}
        monkeypatch.setenv(native.ACTIVATION, json.dumps(activation))
        native._bundle.cache_clear()

    monkeypatch.setattr(native, "abi", lambda: abi)
    publish()
    yield manifest, source, options, path, publish
    native._bundle.cache_clear()


def test_same_native_module_is_strongly_owned_and_imported_once_across_threads(bundle, monkeypatch):
    _, source, options, _, _ = bundle
    calls, results, failures = [], [], []
    module = object()

    def imported(name, path):
        calls.append((name, path))
        return module

    monkeypatch.setattr(native, "_import", imported)
    barrier = threading.Barrier(8)

    def consume():
        try:
            barrier.wait(timeout=5)
            results.append(native.load_prepared("first_sm_121", [str(source)], options))
        except BaseException as error:  # noqa: BLE001 - Transfer every worker outcome to the asserting test owner.
            failures.append(error)

    threads = [threading.Thread(target=consume) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads) and not failures
    assert len(calls) == 1 and len(results) == 8 and all(result is module for result in results)


@pytest.mark.parametrize("fault", ["hash", "abi", "schema", "count", "target", "entry", "escape", "absolute"])
def test_corrupt_declared_bundle_fails_before_any_native_import(bundle, monkeypatch, fault, tmp_path):
    manifest, source, options, path, publish = bundle
    if fault == "hash":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        if fault == "abi":
            manifest["abi"] = {"host_machine": "other ABI"}
        elif fault == "schema":
            manifest["schema_version"] = True
        elif fault == "count":
            manifest["entries"].pop("second_sm_121")
        elif fault == "target":
            manifest["target_capability"] = [12, 0]
        elif fault == "entry":
            manifest["entries"]["first_sm_121"]["module_name"] = "other"
        else:
            foreign = tmp_path / "foreign.so"
            foreign.write_bytes(b"foreign")
            manifest["entries"]["first_sm_121"]["binary"] = ("../foreign.so" if fault == "escape" else str(foreign))
        publish()
    monkeypatch.setattr(native, "_import", lambda *args: pytest.fail("imported corrupt native artifact"))
    with pytest.raises(ValueError):
        native.load_prepared("first_sm_121", [str(source)], options)


def test_changed_canonical_flags_cannot_hide_behind_jit(bundle, monkeypatch):
    _, source, options, _, _ = bundle
    changed = dict(options, extra_cuda_cflags=["-O3", "--use_fast_math"])
    monkeypatch.setattr(native, "_import", lambda *args: pytest.fail("imported mismatched recipe"))
    with pytest.raises(ValueError, match="recipe changed"):
        native.load_prepared("first_sm_121", [str(source)], changed)


def test_distinct_supported_targets_custom_builds_and_workspaces_retain_jit(bundle, tmp_path):
    _, source, options, _, _ = bundle
    assert native.load_prepared("first_sm_120", [str(source)], options) is None
    workspace = tmp_path / "workspace.cu"
    workspace.write_text("// independent canonical workspace\n")
    assert native.load_prepared("first_sm_121", [str(workspace)], options) is None
    assert native.load_prepared("first_sm_121", [str(source)], dict(options, build_directory=tmp_path)) is None


def test_actual_gpu_minimum_is_checked_before_prepared_selection(monkeypatch):
    def refused(*args):
        raise RuntimeError("unsupported actual GPU")

    monkeypatch.setattr(build, "arch_flags", refused)
    monkeypatch.setattr(native, "load_prepared", lambda *args: pytest.fail("bypassed actual GPU contract"))
    with pytest.raises(RuntimeError, match="unsupported actual GPU"):
        build.load("test", [], prepared=True)


def test_canonical_recipe_can_import_without_cpp_extension_or_compiler(monkeypatch):
    sentinel, seen = object(), []
    monkeypatch.setenv(native.ACTIVATION, "test-owned-activation")
    monkeypatch.setattr(build, "arch_flags", lambda *args: ["-gencode=arch=compute_121,code=sm_121"])
    monkeypatch.setattr(native, "load_prepared", lambda *args: seen.append(args) or sentinel)
    monkeypatch.setattr(build, "_toolkit", lambda: pytest.fail("prepared import started compiler preparation"))
    assert build.load("test", ["source.cu"], prepared=True, extra_cuda_cflags=["-O3"]) is sentinel
    assert seen == [("test_sm_121", ["source.cu"], {
        "extra_cuda_cflags": ["-O3", "-gencode=arch=compute_121,code=sm_121"]})]


def test_regular_wheel_without_activation_keeps_jit_policy(monkeypatch):
    monkeypatch.delenv(native.ACTIVATION, raising=False)
    assert native.load_prepared("first_sm_121", ["unresolved-custom-source"], {}) is None


@pytest.mark.torch
def test_original_custom_load_kwargs_do_not_opt_into_matching_artifact(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch.utils.cpp_extension")
    monkeypatch.setattr(build, "arch_flags", lambda *args: ["-gencode=arch=compute_121,code=sm_121"])
    monkeypatch.setattr(build, "_toolkit", list)
    monkeypatch.setattr(build, "_announce", lambda *args: None)
    monkeypatch.setattr(native, "load_prepared", lambda *args: pytest.fail("custom request consumed bundle"))
    seen = []
    monkeypatch.setattr(torch, "load", lambda **options: seen.append(options) or "custom JIT")
    assert build.load("first", [], with_cuda=False, keep_intermediates=False,
                      build_directory=str(tmp_path)) == "custom JIT"
    assert seen[0]["with_cuda"] is False and seen[0]["keep_intermediates"] is False
    assert Path(seen[0]["build_directory"]).parent == tmp_path


@pytest.mark.torch
def test_same_canonical_recipes_supply_runtime_and_explicit_preparation(monkeypatch):
    pytest.importorskip("torch")
    from tensorfold.cuda.nvfp4 import checkpoint
    from tensorfold.families.gemma4.cuda import head
    from tensorfold.families.qwen3_5.cuda import b16

    operations = [(b16._ext, b16._build_options()), (head._cap30_extension, head._cap30_build_options()),
                  (checkpoint._ext, checkpoint._build_options((12, 1)))]
    monkeypatch.setattr(checkpoint, "_capability", lambda: (12, 1))
    captured = []
    monkeypatch.setattr(build, "load", lambda **options: captured.append(options) or "module")
    for operation, expected in operations:
        operation.cache_clear()
        try:
            assert operation() == "module" and captured[-1] == expected
            assert expected["prepared"] is True
        finally:
            operation.cache_clear()
    assert b16._build_options()["name"] == "tensorfold_qwen_b16_v6"
    assert [Path(path).name for path in b16._build_options()["sources"]] == ["b16.cpp", "b16.cu"]
    assert b16._build_options()["extra_cuda_cflags"] == ["-O3"]
    assert checkpoint._build_options((12, 1))["name"] == "tensorfold_nvfp4_ck_v7"
    assert [Path(path).name for path in checkpoint._build_options((12, 1))["sources"]] == [
        "checkpoint.cpp", "act.cu", "lane4.cu", "gemm_ck.cu", "gemm_ws.cu"]
    assert checkpoint._build_options((12, 1))["extra_cuda_cflags"] == ["-O3"]
    assert head._cap30_build_options()["extra_cuda_cflags"] == ["-O3", "-DTENSORFOLD_B16_CAP30=1", "--ftz=false"]
    assert checkpoint._build_options((8, 9))["arch_specific"] is False
    assert checkpoint._build_options((12, 1))["arch_specific"] is True
