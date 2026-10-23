"""CUDA extensions build for the GPU present, whatever architecture list the container sets (#56), and say what a
silent start does: a first build, or a lock a killed build left."""

import builtins
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda import build

NGC_LIST = "8.0 8.6 9.0 10.0 11.0 12.0+PTX"
HINT = "if no other build is running, a killed build left it: stop this start, delete the lock and start again"
SM121 = "-gencode=arch=compute_121,code=sm_121"


def _gpu(monkeypatch, capability, name="GPU"):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: name)


def test_every_cuda_extension_builds_through_the_helper():
    src = Path(__file__).resolve().parents[1] / "src" / "tensorfold"
    # This CPU-only pread reader has no GPU architecture or capability policy.
    cpu_reader = src / "families" / "qwen4_exp" / "native_ssd.py"
    assert "with_cuda=False" in cpu_reader.read_text()
    direct = [p.relative_to(src) for p in src.rglob("*.py")
              if p.name != "build.py" and p != cpu_reader
              and "from torch.utils.cpp_extension import load" in p.read_text()]
    assert direct == []


@pytest.mark.torch
def test_the_flags_name_only_this_gpu(monkeypatch):
    _gpu(monkeypatch, (12, 1))
    assert build.arch_flags() == ["-gencode=arch=compute_121,code=sm_121"]


@pytest.mark.torch
def test_an_older_gpu_is_refused_by_name(monkeypatch):
    _gpu(monkeypatch, (8, 6), "NVIDIA GeForce RTX 3090")
    with pytest.raises(RuntimeError, match=r"capability 8\.9 or newer \(FP8 MMA\).*RTX 3090.*is 8\.6"):
        build.arch_flags()


@pytest.mark.torch
def test_ada_builds_all_but_the_cluster_only_extensions(monkeypatch):
    _gpu(monkeypatch, (8, 9), "NVIDIA GeForce RTX 4090")
    assert build.arch_flags() == ["-gencode=arch=compute_89,code=sm_89"]
    with pytest.raises(RuntimeError, match=r"capability 9\.0 or newer \(thread-block clusters for these weights\)"
                                           r".*RTX 4090.*is 8\.9"):
        build.arch_flags(build.CLUSTERS)


@pytest.mark.torch
def test_the_container_list_adds_nothing(monkeypatch, tmp_path):
    import torch.utils.cpp_extension as ext

    _gpu(monkeypatch, (12, 1))
    seen = {}
    monkeypatch.setattr(ext, "load", lambda **kw: seen.update(kw) or "module")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", NGC_LIST)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))     # load looks up the build directory
    assert build.load(name="x", sources=[], extra_cuda_cflags=["-O3"]) == "module"
    assert seen["extra_cuda_cflags"] == ["-O3", "-gencode=arch=compute_121,code=sm_121"]
    assert ext._get_cuda_arch_flags(seen["extra_cuda_cflags"]) == []


@pytest.mark.torch
@pytest.mark.parametrize("explicit", [False, True])
def test_actual_architecture_and_instruction_target_have_distinct_modules_and_build_locks(ext, monkeypatch, tmp_path,
                                                                                       explicit):
    root = tmp_path / "explicit" if explicit else None
    for capability, specific in [((8, 9), False), ((12, 0), False), ((12, 0), True)]:
        _gpu(monkeypatch, capability)
        options = {"build_directory": root} if explicit else {}
        ext.build.load(name="tf_test", sources=ext.sources, arch_specific=specific, **options)
    records = [kwargs for _, kwargs in ext.calls]
    assert [row["name"] for row in records] == ["tf_test_sm_89", "tf_test_sm_120", "tf_test_sm_120a"]
    for row, target in zip(records, ("89", "120", "120a")):
        assert row["extra_cuda_cflags"] == [f"-gencode=arch=compute_{target},code=sm_{target}"]
        if explicit:
            assert Path(row["build_directory"]) == root / row["name"]
            assert Path(row["build_directory"]).is_dir()


@pytest.mark.torch
def test_one_device_observation_binds_flags_and_module_identity(ext, monkeypatch):
    import torch

    observations = []

    def capability():
        observations.append(True)
        return (8, 9) if len(observations) == 1 else (12, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    ext.build.load(name="tf_test", sources=ext.sources)
    assert observations == [True]
    kwargs = ext.calls[0][1]
    assert kwargs["name"] == "tf_test_sm_89" and "code=sm_89" in kwargs["extra_cuda_cflags"][0]


@pytest.fixture
def ext(tmp_path, monkeypatch):
    """Extension ``tf_test`` with its build directory under ``tmp_path`` and torch's build replaced by a recorder."""

    import torch.utils.cpp_extension as cpp_extension

    _gpu(monkeypatch, (12, 1))
    sources = []
    for name in ("ext.cpp", "ext.cu"):
        (tmp_path / name).write_text("// source\n")
        sources.append(str(tmp_path / name))
    calls, said = [], []

    def directory(name, verbose):
        (tmp_path / name).mkdir(exist_ok=True)
        return str(tmp_path / name)

    monkeypatch.setattr(cpp_extension, "_get_build_directory", directory)
    monkeypatch.setattr(cpp_extension, "load", lambda *a, **k: calls.append((a, k)) or "module")
    # These fixtures describe the audited FileBaton runtime independently of
    # the installed Torch. The real protocol is exercised separately below.
    monkeypatch.setattr(cpp_extension, "_jit_compile", lambda: cpp_extension.FileBaton)
    monkeypatch.setattr(build, "_say", said.append)
    monkeypatch.setattr(build, "_toolkit", lambda: [])               # the pip toolkit has tests of its own
    return SimpleNamespace(build=build, torch=cpp_extension, dir=Path(directory("tf_test_sm_121", False)), sources=sources,
                           calls=calls, said=said)


def _built(ext, mtime=None):
    module = ext.dir / f"tf_test_sm_121{getattr(ext.torch, 'LIB_EXT', '.so')}"
    module.write_bytes(b"")
    stamp = time.time() + 60 if mtime is None else mtime
    os.utime(module, (stamp, stamp))


@pytest.mark.torch
def test_every_other_argument_reaches_torch_unchanged(ext):
    flags = {"extra_cuda_cflags": ["-O3", "--fmad=false"], "extra_include_paths": ["inc"], "verbose": False}
    assert ext.build.load(name="tf_test", sources=ext.sources, **flags) == "module"
    assert ext.calls == [((), {"name": "tf_test_sm_121", "sources": ext.sources, **flags,
                               "extra_cuda_cflags": ["-O3", "--fmad=false", SM121]})]


@pytest.mark.torch
def test_a_first_build_says_so(ext):
    ext.build.load(name="tf_test", sources=ext.sources, verbose=False)
    assert ext.said == ["building CUDA extension tf_test_sm_121 (first start after an install or update; "
                        "later starts reuse it)"]
    assert len(ext.calls) == 1


@pytest.mark.torch
def test_a_built_extension_starts_quietly(ext):
    _built(ext)
    ext.build.load(name="tf_test", sources=ext.sources, verbose=False)
    assert ext.said == []
    assert len(ext.calls) == 1


@pytest.mark.torch
def test_a_source_newer_than_the_built_module_is_a_build(ext):
    _built(ext, mtime=1.0)
    ext.build.load(name="tf_test", sources=ext.sources, verbose=False)
    assert [line.split(" (")[0] for line in ext.said] == ["building CUDA extension tf_test_sm_121"]


@pytest.mark.torch
def test_a_lock_is_named_with_the_fix(ext):
    lock = ext.dir / "lock"
    lock.write_text("")
    ext.build.load(name="tf_test", sources=ext.sources, verbose=False)
    assert len(ext.said) == 1
    assert str(lock) in ext.said[0] and HINT in ext.said[0]
    assert "building" not in ext.said[0]
    assert len(ext.calls) == 1                       # torch still decides: it waits, then loads


@pytest.mark.torch
def test_a_build_directory_argument_is_the_one_checked(ext, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "tf_test_sm_121").mkdir()
    (other / "tf_test_sm_121" / "lock").write_text("")
    ext.build.load(name="tf_test", sources=ext.sources, build_directory=str(other), verbose=False)
    assert str(other / "tf_test_sm_121" / "lock") in ext.said[0]
    assert ext.calls[0][1]["build_directory"] == str(other / "tf_test_sm_121")


@pytest.mark.torch
def test_an_older_gpu_is_refused_before_any_line(ext, monkeypatch):
    _gpu(monkeypatch, (8, 6), "NVIDIA GeForce RTX 3090")
    with pytest.raises(RuntimeError, match="RTX 3090"):
        ext.build.load(name="tf_test", sources=ext.sources, verbose=False)
    assert ext.said == [] and ext.calls == []


@pytest.mark.torch
def test_a_wait_past_the_delay_repeats_the_hint(ext, monkeypatch):
    lock = ext.dir / "lock"
    lock.write_text("")
    repeated = threading.Event()

    def say(text):
        ext.said.append(text)
        if text.startswith("still waiting"):
            repeated.set()

    monkeypatch.setattr(ext.build, "_say", say)
    monkeypatch.setattr(ext.build, "LOCK_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(ext.torch, "load", lambda **k: repeated.wait(5))
    assert ext.build.load(name="tf_test", sources=ext.sources, verbose=False) is True
    assert len(ext.said) == 2
    assert str(lock) in ext.said[1] and HINT in ext.said[1]


@pytest.mark.torch
def test_a_finished_load_does_not_repeat_the_hint(ext, monkeypatch):
    (ext.dir / "lock").write_text("")
    monkeypatch.setattr(ext.build, "LOCK_WAIT_SECONDS", 0.05)
    ext.build.load(name="tf_test", sources=ext.sources, verbose=False)
    time.sleep(0.3)
    assert len(ext.said) == 1


@pytest.mark.torch
def test_no_repeat_once_the_lock_is_gone_or_replaced(ext):
    lock = ext.dir / "lock"
    lock.write_text("")
    os.utime(lock, (1.0, 1.0))
    seen = os.stat(lock)
    identity = (seen.st_ino, seen.st_mtime_ns)
    ext.build._still_waiting(str(lock), identity)
    assert len(ext.said) == 1                        # the same file still there: repeated
    lock.unlink()
    ext.build._still_waiting(str(lock), identity)
    lock.write_text("")                              # a new lock: this start took it and is building
    ext.build._still_waiting(str(lock), identity)
    assert len(ext.said) == 1


@pytest.mark.torch
@pytest.mark.parametrize("lookup", ["missing", "raises"])
def test_without_torchs_directory_lookup_the_build_goes_ahead_quietly(ext, monkeypatch, lookup):
    if lookup == "missing":
        monkeypatch.delattr(ext.torch, "_get_build_directory")
    else:
        monkeypatch.setattr(ext.torch, "_get_build_directory", lambda *a, **k: 1 / 0)
    (ext.dir / "lock").write_text("")
    assert ext.build.load(name="tf_test", sources=ext.sources, verbose=False) == "module"
    assert ext.said == []


@pytest.mark.torch
def test_the_lock_named_is_the_one_torch_waits_on(tmp_path, monkeypatch):
    """Real torch, nothing compiled: ``load`` waits on the lock the line names until it is deleted."""

    import torch.utils.cpp_extension as cpp_extension

    if not build._uses_file_baton(cpp_extension):
        assert "FileLock" in cpp_extension._jit_compile.__code__.co_names
        _actual_advisory_lock(tmp_path, monkeypatch, cpp_extension)
        return
    _gpu(monkeypatch, (12, 1))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "extensions"))
    lock = tmp_path / "extensions" / "tf_lock_probe_sm_121" / "lock"      # TORCH_EXTENSIONS_DIR/<name>/lock
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    source = tmp_path / "tf_lock_probe.cpp"
    source.write_text("int tf_lock_probe() { return 0; }\n")
    said, ended = [], []
    monkeypatch.setattr(build, "_say", said.append)

    def start():
        try:
            ended.append(build.load(name="tf_lock_probe", sources=[str(source)], verbose=False))
        except Exception as error:  # noqa: BLE001 - nothing was built, so the import after the wait fails
            ended.append(error)

    thread = threading.Thread(target=start, daemon=True)
    thread.start()
    thread.join(0.5)
    assert thread.is_alive() and not ended           # torch is waiting on that file
    assert len(said) == 1 and str(lock) in said[0]
    lock.unlink()
    thread.join(10)
    assert not thread.is_alive() and len(ended) == 1


@pytest.mark.torch
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("built", [False, True])
def test_advisory_lock_file_is_never_probed_for_wait_or_stale_advice(ext, monkeypatch, active, built):
    """A real advisory lock file persists after release; its inode proves no ownership."""

    FileLock = pytest.importorskip("filelock").FileLock
    monkeypatch.setattr(ext.torch, "_jit_compile", lambda: ext.torch.FileLock)
    lock = ext.dir / "lock"
    owner = FileLock(str(lock))
    owner.acquire(timeout=1)
    if not active:
        owner.release()
    if built:
        _built(ext)
    identity = os.stat(lock)
    contents = lock.read_bytes()
    real_stat = os.stat

    def no_lock_probe(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and os.fspath(path) == str(lock):
            pytest.fail("diagnostics must not infer advisory lock ownership from the file")
        return real_stat(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(build.os, "stat", no_lock_probe)
            assert ext.build._announce(ext.torch, "tf_test_sm_121", ext.sources, None) is None
        after = os.stat(lock)
        assert (after.st_ino, after.st_mtime_ns) == (identity.st_ino, identity.st_mtime_ns)
        assert lock.read_bytes() == contents
        assert all("wait" not in line and "delete" not in line and HINT not in line for line in ext.said)
        assert ext.said == ([] if built else [
            "building CUDA extension tf_test_sm_121 (first start after an install or update; later starts reuse it)"
        ])
    finally:
        owner.release()


@pytest.mark.torch
def test_active_advisory_build_waits_in_backend_without_obsolete_hint_or_timer(ext, monkeypatch):
    """The backend owns the OS lock; TensorFold neither probes it nor advises unlinking it."""

    FileLock = pytest.importorskip("filelock").FileLock
    monkeypatch.setattr(ext.torch, "_jit_compile", lambda: ext.torch.FileLock)
    _built(ext)
    lock = ext.dir / "lock"
    owner = FileLock(str(lock))
    entered, completed = threading.Event(), threading.Event()
    failures = []

    def no_timer(*args, **kwargs):
        pytest.fail("advisory lock files must not start the FileBaton stale-lock timer")

    def load_backend(**kwargs):
        ext.calls.append(((), kwargs))
        entered.set()
        with FileLock(str(lock)):
            return "module"

    def start():
        try:
            assert ext.build.load(name="tf_test", sources=ext.sources, verbose=False) == "module"
            completed.set()
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(build.threading, "Timer", no_timer)
    monkeypatch.setattr(ext.torch, "load", load_backend)
    owner.acquire(timeout=1)
    thread = threading.Thread(target=start)
    try:
        thread.start()
        assert entered.wait(2)
        assert thread.is_alive() and not completed.is_set()
        assert ext.said == []
    finally:
        owner.release()
        thread.join(timeout=2)
    assert not thread.is_alive() and completed.is_set() and not failures
    assert lock.exists()
    assert ext.calls[0][1]["extra_cuda_cflags"] == [SM121]
    assert ext.said == []


@pytest.mark.torch
def test_unknown_jit_lock_protocol_omits_file_baton_advice(ext, monkeypatch):
    monkeypatch.setattr(ext.torch, "_jit_compile", lambda: None)
    _built(ext)
    (ext.dir / "lock").write_text("")
    assert ext.build.load(name="tf_test", sources=ext.sources, verbose=False) == "module"
    assert ext.said == [] and len(ext.calls) == 1


@pytest.mark.torch
def test_real_advisory_torch_build_waits_without_file_baton_advice(tmp_path, monkeypatch):
    """Exercise the nightly's own lock without compiling an extension or loading CUDA."""

    import torch.utils.cpp_extension as cpp_extension

    names = getattr(getattr(cpp_extension._jit_compile, "__code__", None), "co_names", ())
    if "FileLock" not in names or "FileBaton" in names:
        pytest.skip("real advisory lock test requires the advisory-lock Torch runtime")
    _gpu(monkeypatch, (12, 1))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "extensions"))
    lock = tmp_path / "extensions" / "tf_advisory_probe_sm_121" / "lock"
    lock.parent.mkdir(parents=True)
    source = tmp_path / "tf_advisory_probe.cpp"
    source.write_text("int tf_advisory_probe() { return 0; }\n")
    compiled, finished = threading.Event(), threading.Event()
    said, failures = [], []
    monkeypatch.setattr(build, "_say", said.append)
    monkeypatch.setattr(build, "_toolkit", lambda: [])
    monkeypatch.setattr(cpp_extension, "_write_ninja_file_and_build_library", lambda **kwargs: compiled.set())
    monkeypatch.setattr(cpp_extension, "_import_module_from_library", lambda *args: "module")
    owner = cpp_extension.FileLock(str(lock))

    def start():
        try:
            assert build.load(name="tf_advisory_probe", sources=[str(source)], verbose=False) == "module"
            finished.set()
        except BaseException as error:
            failures.append(error)

    owner.acquire(timeout=1)
    thread = threading.Thread(target=start)
    try:
        thread.start()
        thread.join(0.1)
        assert thread.is_alive() and not compiled.is_set()
        assert all("wait" not in line and "delete" not in line and HINT not in line for line in said)
    finally:
        owner.release()
        thread.join(timeout=5)
    assert not thread.is_alive() and compiled.is_set() and finished.is_set() and not failures
    assert lock.exists()


def _pip_site(tmp_path, nvcc=True):
    """A venv's site-packages with torch beside NVIDIA's pip toolkit (bin/nvcc, lib/libcudart.so.13 only)."""

    site = tmp_path / "site"
    (site / "torch").mkdir(parents=True)
    (site / "torch" / "__init__.py").write_text("")
    home = site / "nvidia" / "cu13"
    (home / "bin").mkdir(parents=True)
    (home / "lib").mkdir()
    if nvcc:
        (home / "bin" / "nvcc").write_text("#!/bin/sh\n")
    (home / "lib" / "libcudart.so.13").write_text("")
    torch = SimpleNamespace(__file__=str(site / "torch" / "__init__.py"), version=SimpleNamespace(cuda="13.0"))
    ext = SimpleNamespace(CUDA_HOME=None, get_default_build_root=lambda: str(tmp_path / "ext"))
    return home, torch, ext


def test_with_no_toolkit_the_pip_one_beside_torch_builds_and_links(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TF_FOUNDATION_STATE", raising=False)  # Unmanaged pip-toolkit capability.
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    home, torch, ext = _pip_site(tmp_path)
    links = tmp_path / "ext" / "tensorfold_cudart"
    assert build.pip_toolkit(ext, torch) == [f"-L{links}"]
    assert ext.CUDA_HOME == str(home) == os.environ["CUDA_HOME"] and os.environ["PATH"].startswith(str(home / "bin"))
    assert (links / "libcudart.so").resolve() == (home / "lib" / "libcudart.so.13").resolve()
    assert f"CUDA compiler: NVIDIA's pip toolkit for CUDA 13.0 at {home}" in capsys.readouterr().out
    assert build.pip_toolkit(ext, torch) == []                      # found now: nothing more to do


def test_a_toolkit_torch_found_or_no_pip_toolkit_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("TF_FOUNDATION_STATE", raising=False)  # Unmanaged toolkit discovery.
    monkeypatch.delenv("CUDA_HOME", raising=False)
    home, torch, ext = _pip_site(tmp_path)
    ext.CUDA_HOME = "/usr/local/cuda"
    assert build.pip_toolkit(ext, torch) == [] and ext.CUDA_HOME == "/usr/local/cuda"
    _, torch, ext = _pip_site(tmp_path / "bare", nvcc=False)
    assert build.pip_toolkit(ext, torch) == [] and ext.CUDA_HOME is None and "CUDA_HOME" not in os.environ


def _actual_advisory_lock(tmp_path, monkeypatch, cpp_extension):
    """Current Torch's actual JIT/FileLock contention, without compiling code."""
    _gpu(monkeypatch, (12, 1))
    monkeypatch.setattr(build, "_toolkit", lambda: [])
    directory = tmp_path / "advisory-extension"
    directory.mkdir()
    source = tmp_path / "advisory.cpp"
    source.write_text("int fixture() { return 0; }\n")
    lock = directory / "tf_advisory_probe_sm_121" / "lock"
    lock.parent.mkdir()
    actual_lock = cpp_extension.FileLock
    reached_lock, finished = threading.Event(), threading.Event()
    acquired, compiled, ended, said = [], [], [], []

    def observed_lock(path, *args, **kwargs):
        acquired.append(Path(path))
        reached_lock.set()
        return actual_lock(path, *args, **kwargs)

    monkeypatch.setattr(cpp_extension, "FileLock", observed_lock)
    monkeypatch.setattr(cpp_extension, "_write_ninja_file_and_build_library", lambda *a, **k: compiled.append(k))
    monkeypatch.setattr(cpp_extension, "_import_module_from_library", lambda *a, **k: "fixture-module")
    monkeypatch.setattr(build, "_say", said.append)

    def run():
        try:
            ended.append(build.load(name="tf_advisory_probe", sources=[str(source)],
                                    build_directory=str(directory), verbose=False, with_cuda=False))
        except BaseException as error:
            ended.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=run, daemon=True)
    primary = None
    try:
        with actual_lock(str(lock)):
            worker.start()
            assert reached_lock.wait(3), "JIT did not reach its actual lock"
            assert acquired == [lock]
            worker.join(0.05)
            assert worker.is_alive() and not finished.is_set() and compiled == []
            assert lock.exists()
    except BaseException as error:
        primary = error
    finally:
        if worker.ident is not None:
            worker.join(10)
    if worker.is_alive():
        if primary is not None:
            primary.add_note("advisory JIT worker did not stop after releasing its owned lock")
            raise primary
        raise AssertionError("advisory JIT worker did not stop after releasing its owned lock")
    if primary is not None:
        raise primary
    assert finished.is_set() and ended == ["fixture-module"] and len(compiled) == 1
    assert all("delete" not in line and "killed build" not in line for line in said)
    # Advisory locks are acquired independently of a persistent marker file.
    assert lock.exists()
    with actual_lock(str(lock), timeout=1):
        pass


class _ManagedStream:
    def __init__(self, raw=b'{}', read_error=None, close_error=None, closing=None):
        self.raw, self.read_error, self.close_error, self.closing = raw, read_error, close_error, closing
        self.reads, self.closes = [], 0

    def read(self, count):
        self.reads.append(count)
        if self.read_error is not None:
            raise self.read_error
        return self.raw

    def close(self):
        self.closes += 1
        if self.closing is not None:
            self.closing()
        if self.close_error is not None:
            raise self.close_error


class _ManagedPath:
    def __init__(self, stream):
        self.stream, self.opens = stream, []

    def open(self, *args, **kwargs):
        self.opens.append((args, kwargs))
        return self.stream


def _managed_roots(error):
    if isinstance(error, BaseExceptionGroup):
        return [root for nested in error.exceptions for root in _managed_roots(nested)]
    return [error]


def test_managed_read_preserves_same_open_and_exact_read_contract():
    stream = _ManagedStream(raw=b'{"toolchain": {}}')
    path = _ManagedPath(stream)
    assert build._read_managed_state(path) == stream.raw
    assert path.opens == [(("rb",), {})]
    assert stream.reads == [64 * 2**20 + 1] and stream.closes == 1


@pytest.mark.parametrize("primary", [MemoryError("read exhaustion"), KeyboardInterrupt()])
def test_managed_read_failure_with_successful_close_keeps_exact_root(primary):
    prior = OSError("native")
    primary.__cause__ = prior
    stream = _ManagedStream(read_error=primary)
    try:
        build._read_managed_state(_ManagedPath(stream))
    except BaseException as error:  # noqa: BLE001 - assert exact exhaustion/interruption identity, including KeyboardInterrupt.
        assert error is primary and BaseException.__cause__.__get__(error) is prior
    else:
        pytest.fail("read failure was swallowed")
    assert stream.closes == 1


def test_managed_success_has_no_new_transport_dependency(monkeypatch):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "tensorfold.cleanup":
            pytest.fail("successful state read must not import failure transport")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    stream = _ManagedStream(raw=b'snapshot')
    assert build._read_managed_state(_ManagedPath(stream)) == b'snapshot'
    assert stream.closes == 1


def test_managed_transport_import_failure_still_keeps_read_primary(monkeypatch):
    original = builtins.__import__
    allocation = MemoryError("transport import")

    def fail(name, *args, **kwargs):
        if name == "tensorfold.cleanup":
            raise allocation
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail)
    primary = RuntimeError("read")
    stream = _ManagedStream(read_error=primary, close_error=OSError("close"))
    with pytest.raises(RuntimeError) as caught:
        build._read_managed_state(_ManagedPath(stream))
    assert caught.value is primary and BaseException.__cause__.__get__(primary) is allocation
    assert stream.closes == 1


@pytest.mark.parametrize("same_error", [False, True])
def test_managed_read_and_close_keep_primary_and_native_links(same_error):
    primary, prior, context = RuntimeError("read"), ValueError("native"), KeyError("context")
    cleanup = primary if same_error else OSError("close")
    primary.__cause__, primary.__context__, primary.__suppress_context__ = prior, context, True

    def rewrite_links():
        primary.__cause__ = ArithmeticError("foreign close changed native link")

    stream = _ManagedStream(read_error=primary, close_error=cleanup, closing=rewrite_links)
    with pytest.raises(RuntimeError) as caught:
        build._read_managed_state(_ManagedPath(stream))
    assert caught.value is primary and stream.closes == 1
    roots = _managed_roots(BaseException.__cause__.__get__(primary))
    assert any(error is prior for error in roots)
    if not same_error:
        assert any(error is context for error in roots) and any(error is cleanup for error in roots)


def test_managed_close_only_is_exact_failure_without_retry():
    cleanup = OSError("close")
    stream = _ManagedStream(raw=b'{}', close_error=cleanup)
    with pytest.raises(OSError) as caught:
        build._read_managed_state(_ManagedPath(stream))
    assert caught.value is cleanup and stream.closes == 1
    frames = []
    traceback = cleanup.__traceback__
    while traceback is not None:
        frames.append(traceback.tb_frame)
        traceback = traceback.tb_next
    assert any(frame.f_locals.get("stream") is stream for frame in frames)


def test_managed_hostile_exception_getters_cannot_skip_close():
    class HostileError(RuntimeError):
        def __getattribute__(self, name):
            if name in ("__cause__", "__context__", "__suppress_context__"):
                raise AssertionError("exception hook must not run")
            return super().__getattribute__(name)

    primary, cleanup = HostileError("read"), OSError("close")
    stream = _ManagedStream(read_error=primary, close_error=cleanup)
    try:
        build._read_managed_state(_ManagedPath(stream))
    except BaseException as error:  # noqa: BLE001 - inspect the exact primary and cleanup roots without invoking hostile getters.
        assert error is primary
        assert any(item is cleanup for item in _managed_roots(BaseException.__cause__.__get__(primary)))
    else:
        pytest.fail("owned read/close failures were swallowed")
    assert stream.closes == 1


def _managed_toolkit(monkeypatch, stream):
    original = build.Path
    selected = _ManagedPath(stream)
    monkeypatch.setenv("TF_FOUNDATION_STATE", "selected-managed-state")
    monkeypatch.setattr(build, "Path", lambda value: selected if value == "selected-managed-state" else original(value))
    return selected


def test_managed_state_cap_refuses_before_json_after_one_close(monkeypatch):
    stream = _ManagedStream(raw=b' ' * (64 * 2**20 + 1))
    selected = _managed_toolkit(monkeypatch, stream)
    with pytest.raises(ValueError, match="managed CUDA foundation state exceeds its bound"):
        build.pip_toolkit(None, None)
    assert selected.opens == [(("rb",), {})]
    assert stream.reads == [64 * 2**20 + 1] and stream.closes == 1


def test_managed_state_exact_cap_is_retained(tmp_path, monkeypatch):
    home = tmp_path / "native-toolkit"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "nvcc").touch()
    payload = json.dumps({"toolchain": {"cuda_home": str(home)}}).encode()
    stream = _ManagedStream(raw=payload + b' ' * (64 * 2**20 - len(payload)))
    _managed_toolkit(monkeypatch, stream)
    monkeypatch.setenv("CUDA_HOME", str(home))
    assert build.pip_toolkit(SimpleNamespace(CUDA_HOME=str(home)), None) == []
    assert stream.closes == 1


@pytest.mark.skipif(os.name != "posix", reason="ordinary POSIX symlink fixture")
def test_managed_state_follows_the_same_path_open_symlink(tmp_path, monkeypatch):
    home = tmp_path / "native-toolkit"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "nvcc").touch()
    source = tmp_path / "state.json"
    source.write_text(json.dumps({"toolchain": {"cuda_home": str(home)}}))
    selected = tmp_path / "state-link.json"
    selected.symlink_to(source)
    monkeypatch.setenv("TF_FOUNDATION_STATE", str(selected))
    monkeypatch.setenv("CUDA_HOME", str(home))
    assert build.pip_toolkit(SimpleNamespace(CUDA_HOME=str(home)), None) == []
    assert selected.is_symlink()


def test_managed_invalid_json_and_toolkit_mismatch_keep_existing_failures(tmp_path, monkeypatch):
    stream = _ManagedStream(raw=b'invalid-json')
    _managed_toolkit(monkeypatch, stream)
    with pytest.raises(json.JSONDecodeError):
        build.pip_toolkit(None, None)
    assert stream.closes == 1
    home = tmp_path / "missing-toolkit"
    home.mkdir()  # A valid declared root reaches the unchanged missing-nvcc refusal.
    stream = _ManagedStream(raw=json.dumps({"toolchain": {"cuda_home": str(home)}}).encode())
    _managed_toolkit(monkeypatch, stream)
    monkeypatch.setenv("CUDA_HOME", str(home))
    with pytest.raises(RuntimeError, match="managed CUDA foundation requires its selected native toolkit"):
        build.pip_toolkit(SimpleNamespace(CUDA_HOME=str(home)), None)
    assert stream.closes == 1
