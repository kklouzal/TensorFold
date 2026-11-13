"""CPU evidence-boundary controls; mocked records confer no native authority."""

import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda.sm70 import gdn_chunk64_artifact as artifact
from tools import v100_g3_native_observation as observed
from tools import v100_native_contract_driver as driver


def fixture(monkeypatch, tmp_path):
    kernels = []
    source_paths = []
    calls = []
    names = ("tilelang_cumsum", "tilelang_kkt", "tilelang_core")
    for index, name in enumerate(names):

        def get_function(symbol, query_imports, expected=name):
            assert symbol == expected and query_imports is False
            calls.append(symbol)
            return NS(owner=symbol)

        module = NS(kind="library", get_function=get_function)
        kernel = NS(
            prim_func=NS(attrs={"global_symbol": name}),
            adapter=NS(executable=NS(_jitted_mod=module)),
            get_kernel_source=lambda name=name: name,
        )
        kernels.append(kernel)
        path = Path("/tmp/tvm-debug-mode-tempdirs") / str(index) / "exported.so"
        source_paths.append(str(path))
    library = {"path": str(tmp_path / "compiler.so"), "bytes": 20, "sha256": "a" * 64}
    report = {
        "cache_directory": str(tmp_path),
        "native_libraries": {"compiler": library},
        "optional_native_libraries": {},
    }
    prepared = NS(
        core=NS(cumsum=kernels[0], kkt=kernels[1], core=kernels[2], compiler_report={"owned_artifact": report})
    )
    monkeypatch.setattr(artifact, "loaded_compiler", lambda value: [library])
    monkeypatch.setattr(
        driver,
        "_read_native_maps",
        lambda: "\n".join(
            f"{index + 1:x}-{index + 2:x} r-xp 00000000 01:01 2 {path}" for index, path in enumerate(source_paths)
        ),
    )
    monkeypatch.setattr(observed, "_exports", lambda path, unused: {"__tvm_ffi_" + names[source_paths.index(path)]})
    return prepared, source_paths, library, calls


def test_three_actual_callee_owners_and_unique_exports(monkeypatch, tmp_path):
    prepared, paths, library, calls = fixture(monkeypatch, tmp_path)
    assert set(observed.provider_paths(prepared)) == {*paths, library["path"]}
    assert calls == ["tilelang_cumsum", "tilelang_kkt", "tilelang_core"]


@pytest.mark.parametrize(
    "defect", ("missing_owner", "wrong_kind", "duplicate_export", "missing_mapping", "compiler_bytes")
)
def test_native_evidence_refuses_incomplete_or_mismatched_bindings(monkeypatch, tmp_path, defect):
    prepared, _paths, library, _ = fixture(monkeypatch, tmp_path)
    if defect == "missing_owner":
        prepared.core.core.adapter.executable._jitted_mod = None
    elif defect == "wrong_kind":
        prepared.core.core.adapter.executable._jitted_mod.kind = "c"
    elif defect == "duplicate_export":
        monkeypatch.setattr(observed, "_exports", lambda *args: {"__tvm_ffi_tilelang_core"})
    elif defect == "missing_mapping":
        monkeypatch.setattr(driver, "_read_native_maps", lambda: "")
    else:
        monkeypatch.setattr(
            driver,
            "observe_native_elfs",
            lambda *args, **kwargs: {
                "mapped_elfs": [{**library, "sha256": "b" * 64}],
                "missing_required_provider_paths": [],
            },
        )
    with pytest.raises(ValueError):
        observed.observe(prepared)


def test_public_retention_restores_after_exception_and_admits_before_import(monkeypatch):
    order = []
    state = [False]

    @contextmanager
    def keep():
        previous = state[0]
        state[0] = True
        try:
            yield
        finally:
            state[0] = previous

    utils = ModuleType("tvm.contrib.utils")
    utils.TempDirectory = NS(set_keep_for_debug=keep)
    monkeypatch.setitem(sys.modules, "tvm.contrib.utils", utils)

    def imported(name):
        assert order == ["admitted"] and name == "tilelang"
        order.append("imported")

    monkeypatch.setattr(observed.importlib, "import_module", imported)
    with (
        pytest.raises(RuntimeError, match="primary"),
        observed.retained_jit_files(NS(inspect=lambda: order.append("admitted"))),
    ):
        assert state[0] is True
        raise RuntimeError("primary")
    assert state[0] is False and order == ["admitted", "imported"]


def test_actual_installed_nm_reads_held_genuine_fd_elf_and_retires_descriptor(monkeypatch):
    import errno
    import os

    import tensorfold._fd_owner as genuine

    path = Path(genuine.__file__)
    assert str(path).startswith("/opt/venv/") and path.suffix == ".so"
    original = driver._owned_descriptor
    descriptors = []

    @contextmanager
    def held(*args, **kwargs):
        with original(*args, **kwargs) as descriptor:
            descriptors.append(descriptor)
            before = driver._file_stat(descriptor)
            yield descriptor
            assert driver._file_stat(descriptor) == before

    monkeypatch.setattr(driver, "_owned_descriptor", held)
    assert "PyInit__fd_owner" in observed._exports(str(path), driver)
    assert len(descriptors) == 1
    with pytest.raises(OSError) as caught:
        os.fstat(descriptors[0])
    assert caught.value.errno == errno.EBADF


@pytest.mark.parametrize("status,code", [("output-limit", -9), ("timeout", -9), ("failed", 1)])
def test_export_probe_rejects_incomplete_process_capture(monkeypatch, tmp_path, status, code):
    path = tmp_path / "controlled.so"
    path.write_bytes(b"bounded fixture")
    calls = []

    def incomplete(argv, **kwargs):
        calls.append((argv, kwargs))
        return {"status": status, "exit_code": code, "output_utf8": "PyInit__fd_owner\n"}

    monkeypatch.setattr(observed, "capture", incomplete)
    with pytest.raises(RuntimeError, match=status):
        observed._exports(str(path), driver)
    assert len(calls) == 1 and calls[0][1]["limit"] == 65536
    assert len(calls[0][1]["pass_fds"]) == 1
