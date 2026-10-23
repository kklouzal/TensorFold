"""Build the CUDA extensions for the GPU present, whatever architecture list the container sets, saying when a build runs or waits on a lock."""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import sys
import threading
from typing import Any

MIN_CAPABILITY = (8, 9)         # FP8 MMA and e4m3 conversions (Ada); kernels with clusters use them from 9.0
CLUSTERS = (9, 0)               # extensions built only on thread-block clusters (NVFP4) need Hopper or newer
# stop first: when the lock goes, a waiting start imports whatever module is there without building, even an old one
HINT = "if no other build is running, a killed build left it: stop this start, delete the lock and start again"
LOCK_WAIT_SECONDS = 60.0        # a start still waiting on the same lock this long says so again


def arch_flags(need: tuple[int, int] = MIN_CAPABILITY, arch_specific: bool = False) -> list[str]:
    """nvcc flags for this GPU alone (``arch_specific``: its ``a`` target); a GPU under ``need`` is refused by name."""

    import torch

    if arch_specific and need < CLUSTERS:
        need = CLUSTERS  # The architecture-specific target requires the cluster-generation toolchain contract.
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < need:
        why = "thread-block clusters" if need >= CLUSTERS else "FP8 MMA"
        raise RuntimeError(f"TensorFold's CUDA kernels need compute capability {need[0]}.{need[1]} or newer ({why}"
                           f"{' for these weights' if need > MIN_CAPABILITY else ''}); this GPU "
                           f"({torch.cuda.get_device_name()}) is {major}.{minor}")
    a = "a" if arch_specific else ""
    return [f"-gencode=arch=compute_{major}{minor}{a},code=sm_{major}{minor}{a}"]


def refuse_old_gpu(need: tuple[int, int] = MIN_CAPABILITY) -> None:
    """At startup, before any weight loads: a GPU older than ``need`` is refused by name (no GPU: skipped)."""

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        try:
            arch_flags(need)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from None


def load(name: str, sources: str | list[str], need: tuple[int, int] = MIN_CAPABILITY, *, arch_specific: bool = False,
         **kwargs: Any) -> Any:
    """Build for this GPU only, with a target-qualified JIT module and lock identity.

    Torch's public ``name`` argument controls both its default build directory
    and TORCH_EXTENSION_NAME. Keep them consistent: a waiter must never import
    another rank's single-target binary. Explicit build directories are roots
    beneath which each module/actual target gets its own build and lock files.
    """

    from torch.utils import cpp_extension

    flags = arch_flags(need, arch_specific)
    # arch_flags constructs this single target from one current-device query;
    # the `a` suffix also distinguishes architecture-specific instructions.
    name = name + "_" + flags[0].split(",code=", 1)[1]
    directory = kwargs.get("build_directory")
    if directory is not None:
        directory = Path(directory) / name
        directory.mkdir(parents=True, exist_ok=True)
        kwargs["build_directory"] = str(directory)
    kwargs["extra_cuda_cflags"] = [*kwargs.get("extra_cuda_cflags", []), *flags]
    links = _toolkit()
    if links:
        kwargs["extra_ldflags"] = [*kwargs.get("extra_ldflags", []), *links]
    held = _announce(cpp_extension, name, sources, kwargs.get("build_directory"))
    timer = None
    if held is not None:
        timer = threading.Timer(LOCK_WAIT_SECONDS, _still_waiting, held)
        timer.daemon = True
        timer.start()
    try:
        return cpp_extension.load(name=name, sources=sources, **kwargs)
    finally:
        if timer is not None:
            timer.cancel()


def _announce(cpp_extension: Any, name: str, sources: str | list[str],
              directory: str | None) -> tuple[str, tuple[int, int]] | None:
    """Say whether ``name`` builds or waits on a lock; return (lock path, (inode, mtime)) when it waits."""

    try:
        directory = directory or cpp_extension._get_build_directory(name, verbose=False)   # private in torch
    except Exception:  # noqa: BLE001 - no lookup in this torch: build without the lines
        return None
    if _uses_file_baton(cpp_extension):
        lock = os.path.join(directory, "lock")
        try:
            seen = os.stat(lock)
        except OSError:
            seen = None
        if seen is not None:
            _say(f"CUDA extension {name} waits on the build lock {lock}; {HINT}")
            return lock, (seen.st_ino, seen.st_mtime_ns)
    if _needs_build(os.path.join(directory, name + getattr(cpp_extension, "LIB_EXT", ".so")), sources):
        _say(f"building CUDA extension {name} (first start after an install or update; later starts reuse it)")
    return None


def _uses_file_baton(cpp_extension: Any) -> bool:
    """Only the audited FileBaton JIT protocol treats file existence as ownership.

    Newer Torch uses an OS advisory FileLock whose file persists while unlocked;
    observing or unlinking that file cannot establish or recover lock ownership.
    Resolve the active private implementation at the build boundary. Unknown or
    mixed protocols omit the FileBaton-specific diagnostic.
    """

    compile_code = getattr(getattr(cpp_extension, "_jit_compile", None), "__code__", None)
    names = getattr(compile_code, "co_names", ())
    return "FileBaton" in names and "FileLock" not in names


def _needs_build(module: str, sources: str | list[str]) -> bool:
    """No built module, or a source newer than it (ninja then compiles again)."""

    try:
        built = os.path.getmtime(module)
    except OSError:
        return True
    for source in [sources] if isinstance(sources, str) else sources:
        try:
            if os.path.getmtime(source) > built:
                return True
        except OSError:
            continue
    return False


def _still_waiting(lock: str, identity: tuple[int, int]) -> None:
    """Repeat the hint while the lock seen before the build is still the one there."""

    try:
        now = os.stat(lock)
    except OSError:
        return
    if (now.st_ino, now.st_mtime_ns) == identity:
        _say(f"still waiting after {LOCK_WAIT_SECONDS:g} s on the build lock {lock}; {HINT}")


def _toolkit() -> list[str]:
    """The pip route's tools: the venv's ninja on PATH, NVIDIA's pip toolkit, and the link flags it needs."""

    venv = Path(sys.executable).parent              # an inactive venv: ninja is beside its python, not on PATH
    tool = "ninja.exe" if os.name == "nt" else "ninja"              # the venv's ninja, named as Windows names it
    if shutil.which("ninja") is None and (venv / tool).is_file():
        os.environ["PATH"] = f"{venv}{os.pathsep}{os.environ.get('PATH', '')}"
    return list(_pip_flags())


@lru_cache(maxsize=1)
def _pip_flags() -> tuple[str, ...]:
    """Once a process: NVIDIA's pip toolkit when torch found none (the module keeps it after the first call)."""

    import torch
    from torch.utils import cpp_extension

    return tuple(pip_toolkit(cpp_extension, torch))


def _read_managed_state(path: Path) -> bytes:
    """Retire the same binary stream once, preserving read and close roots.

    Open/file-type/symlink behavior stays Path.open('rb'); no FD extension or
    new parser is needed. A failed close retains its stream in this owning
    traceback until explicit recovery or process retirement, never a retry.
    Failure transport is imported only after resource work has been attempted.
    """
    statuses = [None]
    primary = prior_cause = prior_context = None
    prior_suppress = False
    # Bind the native getters before acquisition; exception subclasses cannot
    # intercept them or allocate a fresh Python bound method during failure.
    cause = BaseException.__cause__.__get__
    context = BaseException.__context__.__get__
    suppress = BaseException.__suppress_context__.__get__
    stream = path.open("rb")
    raw = None
    try:
        raw = stream.read(64 * 2**20 + 1)
    except BaseException as error:  # noqa: BLE001 - every read/interruption root survives owned retirement.
        primary = error
        prior_cause = cause(error)
        prior_context = context(error)
        prior_suppress = suppress(error)
    finally:
        try:
            stream.close()
        except BaseException as error:  # noqa: BLE001 - one attempted close; unresolved stream stays frame-owned.
            statuses[0] = error
            if primary is None:
                primary = error
                prior_cause = cause(error)
                prior_context = context(error)
                prior_suppress = suppress(error)
    if primary is not None:
        try:
            from tensorfold.cleanup import raise_failures

            BaseException.__cause__.__set__(primary, prior_cause)
            BaseException.__context__.__set__(primary, prior_context)
            BaseException.__suppress_context__.__set__(primary, prior_suppress)
            raise_failures(primary, statuses if statuses[0] is not None else ())
        except BaseException as transport:  # Raw roots remain owned if failure transport exhausts.
            if transport is primary:
                raise
            raise primary from transport
    return raw


def pip_toolkit(cpp_extension: Any, torch: Any) -> list[str]:
    """No toolkit found (CUDA_HOME, nvcc, /usr/local/cuda): NVIDIA's pip one beside torch, its cudart linked by name."""

    if os.environ.get("TF_FOUNDATION_STATE"):
        raw = _read_managed_state(Path(os.environ["TF_FOUNDATION_STATE"]))
        if len(raw) > 64 * 2**20:
            raise ValueError("managed CUDA foundation state exceeds its bound")
        expected = Path(json.loads(raw)["toolchain"]["cuda_home"]).resolve(strict=True)
        if (os.environ.get("CUDA_HOME") is None or Path(os.environ["CUDA_HOME"]).resolve(strict=True) != expected
                or cpp_extension.CUDA_HOME is None or Path(cpp_extension.CUDA_HOME).resolve(strict=True) != expected
                or not (expected / "bin/nvcc").is_file()):
            raise RuntimeError("managed CUDA foundation requires its selected native toolkit; rebuild the image")
        return []
    if cpp_extension.CUDA_HOME is not None or not getattr(torch.version, "cuda", None):
        return []
    home = Path(torch.__file__).resolve().parents[1] / "nvidia" / f"cu{torch.version.cuda.split('.')[0]}"
    if os.name == "nt":                       # a Windows wheel: DLLs beside torch and in NVIDIA's cu13 folder
        search = [folder for folder in (home / "bin", home / "lib", Path(torch.__file__).resolve().parent / "lib")
                  if folder.is_dir()]
        for folder in search:                 # a built .pyd loads cudart64_13.dll by name at import
            try:
                os.add_dll_directory(str(folder))
            except (AttributeError, OSError):
                pass                          # old CPython or a refused folder: PATH is the fallback
            os.environ["PATH"] = f"{folder}{os.pathsep}{os.environ.get('PATH', '')}"
        return [f"/LIBPATH:{folder}" for folder in search if (folder / "cudart.lib").is_file()]
    if not (home / "bin" / "nvcc").is_file():
        return []
    cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"] = str(home)    # torch reads its CUDA_HOME at every build
    os.environ["PATH"] = f"{home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    _say(f"CUDA compiler: NVIDIA's pip toolkit for CUDA {torch.version.cuda} at {home}")
    versioned = sorted((home / "lib").glob("libcudart.so.*"))
    if (home / "lib" / "libcudart.so").exists() or not versioned:
        return []
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or cpp_extension.get_default_build_root()
    links = Path(root) / "tensorfold_cudart"                         # -lcudart wants the bare name
    links.mkdir(parents=True, exist_ok=True)
    bare = links / "libcudart.so"
    if not bare.is_symlink() or bare.resolve() != versioned[0].resolve():
        bare.unlink(missing_ok=True)
        try:
            bare.symlink_to(versioned[0])
        except FileExistsError:                                      # another start made the same link first
            pass
    return [f"-L{links}"]


def _say(text: str) -> None:
    """One ``[tensorfold]`` line, flushed like the CLI's."""

    print(f"[tensorfold] {text}", flush=True)


__all__ = ["CLUSTERS", "MIN_CAPABILITY", "arch_flags", "load", "pip_toolkit"]
