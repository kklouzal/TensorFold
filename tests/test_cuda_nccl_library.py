"""NCCL provider selection without communicator or GPU initialization."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tensorfold.cuda import comm

pytestmark = pytest.mark.skipif(os.name == "nt", reason="NCCL is outside the native Windows runtime contract")


def forbidden(*args, **kwargs):
    pytest.fail("discovery continued after selecting an authoritative provider")


def namespace(monkeypatch, locations):
    monkeypatch.setattr(comm.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(submodule_search_locations=locations))


def library_file(root):
    path = root / "lib" / "libnccl.so.2"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"controlled library placeholder")
    return str(path)


@pytest.mark.parametrize("failure", [False, True])
def test_explicit_override_is_authoritative_even_if_invalid(monkeypatch, failure):
    override = "/operator/nccl/libnccl.so.2"
    primary, result, seen = OSError("controlled dependency load failure"), object(), []
    def load(path):
        seen.append(path)
        if failure:
            raise primary
        return result
    monkeypatch.setenv("TF_NCCL_LIB", override)
    monkeypatch.setattr(comm.ctypes, "CDLL", load)
    monkeypatch.setattr(comm, "_python_nccl_library", forbidden)
    monkeypatch.setattr(comm.glob, "glob", forbidden)
    monkeypatch.setattr(comm.ctypes.util, "find_library", forbidden)
    if failure:
        with pytest.raises(RuntimeError, match="TF_NCCL_LIB") as caught:
            comm._library()
        assert caught.value.__cause__ is primary
    else:
        assert comm._library() is result
    assert seen == [override]


@pytest.mark.parametrize("first_missing", [False, True])
def test_active_namespace_locations_precede_torch_and_system(monkeypatch, tmp_path, first_missing):
    first, second = tmp_path / "first", tmp_path / "second"
    chosen = library_file((second if first_missing else first) / "nccl")
    if first_missing:
        (first / "nccl" / "include").mkdir(parents=True)
    else:
        library_file(second / "nccl")
    namespace(monkeypatch, [str(first), str(second)])
    monkeypatch.setenv("TF_NCCL_LIB", "")
    monkeypatch.setattr(comm.glob, "glob", forbidden)
    monkeypatch.setattr(comm.ctypes.util, "find_library", forbidden)
    monkeypatch.setattr(comm.ctypes, "CDLL", lambda path: path)
    assert comm._library() == chosen


@pytest.mark.parametrize("regular_parent", [False, True])
def test_real_active_package_spec_does_not_execute_nvidia_python(monkeypatch, tmp_path, regular_parent):
    root = tmp_path / "site-packages"
    selected = library_file(root / "nvidia" / "nccl")
    if regular_parent:
        (root / "nvidia" / "__init__.py").write_text("raise AssertionError('discovery executed nvidia Python')\n")
    for name in ("nvidia", "nvidia.nccl"):
        # Ensure both preexisting modules and newly created entries restore.
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name)
    # A regular installed NVIDIA package overrides namespace portions even
    # later on sys.path. This fixture owns the active package search path;
    # it must not accidentally select the container's real regular package.
    monkeypatch.setattr(sys, "path", [str(root)])
    comm.importlib.invalidate_caches()
    monkeypatch.delenv("TF_NCCL_LIB", raising=False)
    monkeypatch.setattr(comm.ctypes, "CDLL", lambda path: path)
    assert comm._library() == selected
    assert "nvidia" not in sys.modules and "nvidia.nccl" not in sys.modules


def test_broken_active_package_does_not_select_an_unrelated_library(monkeypatch, tmp_path):
    (tmp_path / "nccl" / "include").mkdir(parents=True)
    namespace(monkeypatch, [str(tmp_path)])
    monkeypatch.delenv("TF_NCCL_LIB", raising=False)
    monkeypatch.setattr(comm.ctypes, "CDLL", forbidden)
    monkeypatch.setattr(comm.glob, "glob", forbidden)
    monkeypatch.setattr(comm.ctypes.util, "find_library", forbidden)
    with pytest.raises(RuntimeError, match=r"active nvidia\.nccl package has no"):
        comm._library()


def test_unloadable_active_wheel_preserves_actual_error_without_fallback(monkeypatch, tmp_path):
    path = library_file(tmp_path / "nccl")
    namespace(monkeypatch, [str(tmp_path)])
    monkeypatch.delenv("TF_NCCL_LIB", raising=False)
    monkeypatch.setattr(comm.glob, "glob", forbidden)
    monkeypatch.setattr(comm.ctypes.util, "find_library", forbidden)
    # Real dlopen of the invalid fixture proves the foreign failure is retained.
    with pytest.raises(RuntimeError, match="active Python environment") as caught:
        comm._library()
    assert isinstance(caught.value.__cause__, OSError)
    assert path in str(caught.value.__cause__)


@pytest.mark.parametrize("load_failure", [False, True])
def test_torch_bundle_precedes_system_and_must_itself_load(monkeypatch, load_failure):
    path = str(Path(comm.torch.__file__).parent / "lib" / "libnccl.so.2")
    primary = OSError("controlled bundled library failure")
    seen = []
    def load(selected):
        seen.append(selected)
        if load_failure:
            raise primary
        return selected
    monkeypatch.delenv("TF_NCCL_LIB", raising=False)
    monkeypatch.setattr(comm.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(comm.glob, "glob", lambda pattern: [path])
    monkeypatch.setattr(comm.ctypes.util, "find_library", forbidden)
    monkeypatch.setattr(comm.ctypes, "CDLL", load)
    if load_failure:
        with pytest.raises(RuntimeError, match="Torch bundle") as caught:
            comm._library()
        assert caught.value.__cause__ is primary
    else:
        assert comm._library() == path
    assert seen == [path]


def test_system_search_retains_expected_absence_fallback_and_deduplicates(monkeypatch):
    seen = []
    def load(path):
        seen.append(path)
        if path == "unavailable-soname":
            raise OSError("system linker candidate unavailable")
        return path
    monkeypatch.delenv("TF_NCCL_LIB", raising=False)
    monkeypatch.setattr(comm.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(comm.glob, "glob", lambda pattern: ["unavailable-soname", "/usr/lib/native/libnccl.so.2"]
                        if pattern == "/usr/lib/*/libnccl.so.2" else [])
    monkeypatch.setattr(comm.ctypes.util, "find_library", lambda name: "unavailable-soname")
    monkeypatch.setattr(comm.ctypes, "CDLL", load)
    assert comm._library() == "/usr/lib/native/libnccl.so.2"
    assert seen == ["unavailable-soname", "/usr/lib/native/libnccl.so.2"]


def test_missing_system_library_preserves_the_loader_error(monkeypatch):
    primary = OSError("controlled missing system library")
    monkeypatch.delenv("TF_NCCL_LIB", raising=False)
    monkeypatch.setattr(comm.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(comm.glob, "glob", lambda pattern: [])
    monkeypatch.setattr(comm.ctypes.util, "find_library", lambda name: "libnccl.so.2")
    def failed(path):
        raise primary
    monkeypatch.setattr(comm.ctypes, "CDLL", failed)
    with pytest.raises(RuntimeError, match="set TF_NCCL_LIB") as caught:
        comm._library()
    assert caught.value.__cause__ is primary


def test_namespace_discovery_failures_are_not_hidden_as_package_absence(monkeypatch):
    primary = PermissionError("controlled environment access failure")
    def failed(name):
        raise primary
    monkeypatch.delenv("TF_NCCL_LIB", raising=False)
    monkeypatch.setattr(comm.importlib.util, "find_spec", failed)
    monkeypatch.setattr(comm.ctypes, "CDLL", forbidden)
    with pytest.raises(PermissionError) as caught:
        comm._library()
    assert caught.value is primary
