"""Finite G3 callee/export evidence through the maintained strict VMA-FD API.

Use only in the controlled diagnostic process. Public TVM debug retention keeps
actual JIT DSOs pathname-backed; this changes diagnostic retention, never math.
The ordinary driver/SDK path and deleted-mapping rejection remain unchanged.
"""

import hashlib
import importlib
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from .v100_adaptive_qualification import capture
else:
    from v100_adaptive_qualification import capture


@contextmanager
def retained_jit_files(compiler):
    # Admit actual owned libraries BEFORE importing TVM. prepare() repeats this
    # admission; the extra cold hash pass is explicit diagnostic overhead.
    compiler.inspect()
    importlib.import_module("tilelang")  # Owned entrypoint configures its verified vendored TVM path.
    from tvm.contrib.utils import TempDirectory

    with TempDirectory.set_keep_for_debug():
        yield


def _exports(path, driver):
    with driver._owned_descriptor(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW) as descriptor:
        before = driver._file_stat(descriptor)
        if not 0 < before["bytes"] <= 16 << 20:
            raise ValueError("G3 generated host-export ELF exceeds its finite diagnostic bound")
        result = capture(
            ["/usr/bin/nm", "-D", "--defined-only", f"/proc/self/fd/{descriptor}"],
            pass_fds=(descriptor,),
            timeout=15,
            limit=65536,
        )
        if driver._file_stat(descriptor) != before:
            raise ValueError("G3 generated ELF changed during dynamic-export inspection")
        if result["status"] != "ok" or result["exit_code"] != 0:
            raise RuntimeError("G3 dynamic-export capture failed: " + result["status"], result)
    return {row.split()[-1] for row in result["output_utf8"].splitlines() if row.split()}


def provider_paths(prepared):
    from tensorfold.cuda.sm70.gdn_chunk64_artifact import loaded_compiler

    from . import v100_native_contract_driver as driver

    report = prepared.core.compiler_report["owned_artifact"]
    libraries = loaded_compiler(report)
    entries = set()
    owners = []
    for kernel in (prepared.core.cumsum, prepared.core.kkt, prepared.core.core):
        symbol = str(kernel.prim_func.attrs["global_symbol"])
        if not re.fullmatch("tilelang_[A-Za-z0-9_]{1,128}", symbol) or symbol in entries:
            raise ValueError("G3 actual generated host entry differs from the finite owned kernels")
        executable = kernel.adapter.executable
        # Pinned TVM Executable retains its actual load_module return owner.
        # Cached JITKernel.from_database instead retains that Module directly.
        module = executable._jitted_mod if hasattr(executable, "_jitted_mod") else executable
        if module is None or module.kind != "library":
            raise ValueError("G3 actual callee has no positively loaded library owner")
        function = module.get_function(symbol, query_imports=False)
        owners.append((module, function))  # Strong owners through maps/export census.
        entries.add(symbol)
    generated = {}
    debug = Path(tempfile.gettempdir()) / "tvm-debug-mode-tempdirs"
    cache = Path(report["cache_directory"]).resolve(strict=True)
    for line in driver._read_native_maps().splitlines():
        row = line.split(None, 5)
        if len(row) != 6 or "x" not in row[1]:
            continue
        path = Path(row[5])
        if (path.name == "exported.so" and path.is_relative_to(debug)) or (
            path.name == "executable.so" and path.is_relative_to(cache)
        ):
            if str(path) in generated:
                continue
            exports = _exports(str(path), driver)
            matched = {name for name in entries if "__tvm_ffi_" + name in exports}
            if len(matched) != 1:
                raise ValueError("G3 actual DSO lacks one unique owned callee export")
            generated[str(path)] = matched.pop()
    if len(generated) != 3 or set(generated.values()) != entries:
        raise ValueError("G3 actual mapped exports differ from the three retained callee owners")
    paths = tuple(sorted({row["path"] for row in libraries} | set(generated)))
    if not 3 < len(paths) <= 8:
        raise ValueError("G3 actual provider closure exceeds its finite bound")
    return paths


def observe(prepared):
    from . import v100_native_contract_driver as driver
    from .v100_owned_model_contract import MODEL_OBSERVER_LIMITS

    paths = provider_paths(prepared)
    report = driver.observe_native_elfs([], provider_paths=paths, limits=MODEL_OBSERVER_LIMITS)
    # The unchanged driver holds actual map_files/path FDs throughout all
    # hashes/reopens/namespace checks. No cache-copy identity substitutes for it.
    allowed = prepared.core.compiler_report["owned_artifact"]
    expected = {
        row["path"]: row
        for row in (*allowed["native_libraries"].values(), *allowed["optional_native_libraries"].values())
    }
    for row in report["mapped_elfs"]:
        if row["path"] in expected:
            binding = expected[row["path"]]
            if (row["bytes"], row["sha256"]) != (binding["bytes"], binding["sha256"]):
                raise ValueError("G3 actual VMA backing differs from admitted compiler bytes/SHA")
    if report["missing_required_provider_paths"]:
        raise ValueError("G3 actual provider lacks a VMA-backed FD identity")
    report["required_provider_paths"] = list(paths)
    report["generated_source_sha256"] = {
        str(kernel.prim_func.attrs["global_symbol"]): hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()
        for kernel in (prepared.core.cumsum, prepared.core.kkt, prepared.core.core)
    }
    report["provider_scope"] = (
        "Three retained callee owners/unique actual dynamic exports and admitted compiler libraries; no quality/P3"
    )
    return report
