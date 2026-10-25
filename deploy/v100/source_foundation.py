#!/usr/bin/env python3
"""Reproducible input preparation for the separate native SM70 source foundation.

Build/CPU inspection and V100 execution are different gates. This producer never
changes a host driver or edits an installed framework/provider. Its outputs are
hash locks and receipts for owned source-build inputs.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import stat
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

MAX_DOCUMENT_BYTES = 16 << 20
INCLUDE_ROOTS = (("cccl", "/opt/tensorfold-native/cccl-3.5.0/include"),
                 ("cutlass", "/opt/tensorfold-native/cutlass-4.8.1/include"))

# CMake configure_file/copy outputs and torchgen side outputs intentionally not
# Ninja compilation targets. Authorities: aten/src/ATen/CMakeLists.txt,
# torch/CMakeLists.txt, tools/pyi/gen_pyi.py and torchgen/utils.py (Everything
# files are explicitly excluded from compilation there). All other generated
# source paths must be declared outputs of the selected Ninja build graph.
PYTORCH_SIDE_OUTPUTS = frozenset({
    "aten/src/ATen/Config.h", "aten/src/ATen/cuda/CUDAConfig.h", "compile_commands.json",
    "tools/shared/_utils_internal.py", "torch/_C/_nn.pyi", "torch/_VF.pyi", "torch/return_types.pyi",
    "torch/headeronly/version.h", "torch/csrc/autograd/generated/templates/VariableType.cpp",
    *{"torch/csrc/autograd/generated/" + name + "Everything.cpp" for name in (
        "ADInplaceOrViewType", "TraceType", "VariableType", "python_functions", "python_torch_functions")},
})


def expected_identity(artifact):
    size, sha = artifact.get("bytes"), artifact.get("sha256")
    if (type(size) is not int or size < 0 or not isinstance(sha, str) or len(sha) != 64
            or any(char not in "0123456789abcdef" for char in sha)):
        raise ValueError("artifact identity needs a nonnegative byte count and full SHA256")
    return {"bytes": size, "sha256": sha}


def file_identity(path, *, include_sha512=False):
    """Stream a stable regular artifact; source/model-sized buffers are never retained."""
    with open(path, "rb", opener=lambda name, flags: os.open(name, flags | os.O_NONBLOCK)) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("foundation artifact must be a regular file: " + str(path))
        digest, signed, seen = hashlib.sha256(), hashlib.sha512() if include_sha512 else None, 0
        while seen < before.st_size:
            block = stream.read(min(1 << 20, before.st_size - seen))
            if not block:
                break
            digest.update(block)
            if signed is not None:
                signed.update(block)
            seen += len(block)
        after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (seen != before.st_size or stream.read(1)
                or any(getattr(before, key) != getattr(after, key) for key in fields)):
            raise ValueError("foundation artifact changed during hashing: " + str(path))
    result = {"bytes": seen, "sha256": digest.hexdigest()}
    if signed is not None:
        result["sha512"] = signed.hexdigest()
    return result


def read_document(path):
    with open(path, "rb", opener=lambda name, flags: os.open(name, flags | os.O_NONBLOCK)) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_DOCUMENT_BYTES:
            raise ValueError("source foundation metadata must be a bounded regular file")
        raw = stream.read(info.st_size + 1)
        after = os.fstat(stream.fileno())
        def identity(item):
            return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns
        if len(raw) != info.st_size or identity(after) != identity(info):
            raise ValueError("source foundation metadata changed during read")
    return raw


def read_json(path):
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ValueError("duplicate source foundation metadata key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite source foundation metadata")

    raw = read_document(path)
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant), raw


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def effective_commit(source):
    return source.get("effective_commit", source["commit"])


def render_build_report(report, report_sha256, *, runtime=False, lint=False):
    if not isinstance(report, dict) or not isinstance(report.get("environment"), dict):
        raise ValueError("source-build dependency report must contain its native environment")
    environment = report.get("environment", {})
    if any(environment.get(key) != value for key, value in {
        "sys_platform": "linux", "platform_machine": "x86_64", "implementation_name": "cpython",
        "python_version": "3.12"}.items()):
        raise ValueError("resolve source-build dependencies on native Linux AMD64 CPython3.12")
    installs = report.get("install")
    if not isinstance(installs, list) or len(installs) > 4096:
        raise ValueError("source-build dependency report must contain a bounded install list")
    selected = {}
    for item in installs:
        metadata, download = item["metadata"], item["download_info"]
        name = canonicalize_name(metadata["name"], validate=True)
        if (name in selected or name in ("torch", "torchvision") or (name == "triton" and not runtime)
                or name.startswith(("nvidia-", "cuda-"))):
            raise ValueError("source-build tool closure contains a duplicate or binary accelerator provider: " + name)
        url = download["url"]
        if not isinstance(url, str) or not isinstance(metadata.get("version"), str):
            raise ValueError("source-build package version and URL must be strings")
        parsed = urlsplit(url)
        sha = download["archive_info"]["hashes"]["sha256"]
        wheel_name, wheel_version, _, _ = parse_wheel_filename(parsed.path.rsplit("/", 1)[-1])
        if (parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org" or parsed.username
                or parsed.password or parsed.fragment or parsed.query or any(c.isspace() for c in url)
                or wheel_name != name or wheel_version != Version(metadata["version"])
                or not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("source-build tools need exact official PyPI wheel hashes")
        selected[name] = {"version": metadata["version"], "url": url, "sha256": sha,
                          "requires_dist": metadata.get("requires_dist", [])}
    if runtime and lint:
        raise ValueError("source runtime and lint environments have separate dependency lifetimes")
    required = ({"spin", "uv", "lintrunner", "cmakelint", "packaging"} if lint else
                {"scikit-build-core", "cmake", "ninja", "numpy", "setuptools", "packaging", "pip", "pyyaml",
                 "six", "typing-extensions"})
    if not required <= selected.keys():
        raise ValueError("source-build report omitted required framework build tools")
    if runtime and (selected.get("triton", {}).get("version") != "3.8.0"
                    or not {"huggingface-hub", "tokenizers", "safetensors", "pytest", "ruff"} <= selected.keys()):
        raise ValueError("source runtime closure needs its pinned import-only Triton and application/test tools")
    lock = "# Generated by deploy/v100/source_foundation.py from the recorded native pip report.\n"
    lock += "".join(f"{name} @ {row['url']} --hash=sha256:{row['sha256']}\n" for name, row in sorted(selected.items()))
    schema = ("tensorfold.v100-source-lint-tools" if lint else
              "tensorfold.v100-source-runtime" if runtime else "tensorfold.v100-source-build-tools")
    receipt = {"schema": schema,
               "version": 1, "environment": environment,
               "resolver_report_sha256": report_sha256, "packages": selected,
               "lock_sha256": hashlib.sha256(lock.encode()).hexdigest()}
    return lock, receipt


def offline_lock(receipt, *, bootstrap=False):
    packages = receipt["packages"]
    names = ("packaging", "pip") if bootstrap else sorted(packages)
    result = "# Generated exact local-wheel closure; provenance URLs remain in its companion receipt.\n"
    for name in names:
        row = packages[name]
        version = str(Version(row["version"]))
        sha = row["sha256"]
        if canonicalize_name(name, validate=True) != name or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("offline wheel requirement has an invalid name/hash")
        result += f"{name}=={version} --hash=sha256:{sha}\n"
    return result


def capture_apt(directory, pins, expected, download_uris=None):
    packages, seen = [], {}
    origins = {}
    if download_uris is not None:
        for line in download_uris.read_text().splitlines():
            fields = shlex.split(line)
            if len(fields) != 4:
                raise ValueError("APT download provenance must contain one bounded URI record per package")
            parsed = urlsplit(fields[0])
            if (parsed.scheme not in ("http", "https") or parsed.hostname not in
                    ("archive.ubuntu.com", "security.ubuntu.com") or parsed.username or parsed.password
                    or parsed.query or parsed.fragment or fields[1] in origins):
                raise ValueError("APT build tools must come from their recorded official Ubuntu origin")
            origins[fields[1]] = ("https://" + parsed.netloc + parsed.path, int(fields[2]), fields[3])
    for path in sorted(directory.glob("*.deb")):
        fields = subprocess.check_output(["dpkg-deb", "--show", "--showformat=${Package}\n${Version}\n${Architecture}\n",
                                          str(path)]).decode().splitlines()
        if len(fields) != 3 or fields[0] in seen or fields[2] not in ("amd64", "all"):
            raise ValueError("APT source-build bundle has duplicate or foreign packages")
        seen[fields[0]] = fields[1]
        identity = file_identity(path, include_sha512=path.name in origins)
        signed = identity.pop("sha512", None)
        row = {"artifact": path.name, "package": fields[0], "version": fields[1],
               "architecture": fields[2], **identity}
        if path.name in origins:
            url, size, digest = origins.pop(path.name)
            algorithm, wanted = digest.split(":", 1)
            if algorithm != "SHA512" or size != row["bytes"]:
                raise ValueError("APT download provenance size/digest type disagrees with its artifact")
            if signed != wanted:
                raise ValueError("APT artifact differs from its signed-index download digest")
            row["url"] = url
        packages.append(row)
    wanted = {}
    for line in expected.read_text().splitlines():
        name, version = line.split("=", 1)
        wanted[name.split(":", 1)[0]] = version
    for row in pins["native_packages"]:
        wanted[row["package"]] = row["version"]
    if seen != wanted:
        raise ValueError("APT bundle must contain exactly the installed source-build package closure")
    for row in pins["native_packages"]:
        actual = next(item for item in packages if item["artifact"] == row["artifact"])
        if any(actual[key] != row[key] for key in ("package", "version", "bytes", "sha256")):
            raise ValueError("APT bundle native package differs from its official pinned artifact")
        actual["url"] = row["url"]
    if download_uris is not None and (origins or any("url" not in row for row in packages)):
        raise ValueError("APT download provenance does not cover exactly its source-build bundle")
    return {"schema": "tensorfold.v100-source-apt-bundle", "version": 1,
            "base": pins["base"], "packages": packages}


def dockerfiles(pins):
    """One preparation recipe for the source builder and installed development image."""
    common = f'''# Generated by source_foundation.py render-dockerfiles. Device/release gates remain separate.
# Supply --build-context source_artifacts=/path/to/the/verified/release/bundle.
FROM {pins["base"]} AS source-builder
COPY deploy/v100/source-apt-amd64.sha256 /opt/tensorfold-pins/source-apt-amd64.sha256
RUN --network=none --mount=from=source_artifacts,source=/build-debs,target=/debs \\
    cd /debs && sha256sum --strict --check /opt/tensorfold-pins/source-apt-amd64.sha256 \\
    && awk '{{print $2}}' /opt/tensorfold-pins/source-apt-amd64.sha256 > /tmp/source-apt-files \\
    && xargs -a /tmp/source-apt-files dpkg -i \\
    && dpkg --purge libnccl-dev libnccl2
COPY deploy/v100/requirements-source-bootstrap-offline-amd64.lock /opt/source-bootstrap.lock
RUN --network=none --mount=from=source_artifacts,source=/python-wheels,target=/python-wheels \\
    python3.12 -m venv /opt/source-venv \\
    && /opt/source-venv/bin/python -m pip install --no-cache-dir --no-index --find-links=/python-wheels \\
       --require-hashes -r /opt/source-bootstrap.lock
ENV PATH="/opt/source-venv/bin:${{PATH}}" CUDA_HOME=/usr/local/cuda \\
    TORCH_CUDA_ARCH_LIST=7.0 NCCL_VERSION=2.32.3 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 OMP_NUM_THREADS=4
WORKDIR /work
ENTRYPOINT ["/opt/source-venv/bin/python"]
'''
    verification = '''
FROM source-builder AS resolver
# The source runtime report is resolved here; binary accelerator wheels are excluded by its producer.

FROM resolver AS verification
COPY deploy/v100/requirements-source-runtime-offline-amd64.lock /locks/requirements-source-runtime.lock
RUN --network=none --mount=from=source_artifacts,source=/python-wheels,target=/python-wheels \\
    python3.12 -m venv /opt/venv \\
    && /opt/source-venv/bin/python -m pip --python /opt/venv install --no-cache-dir --no-index \\
       --find-links=/python-wheels --require-hashes -r /locks/requirements-source-runtime.lock
ENV PATH="/opt/venv/bin:${PATH}" \\
    TF_NCCL_LIB=/opt/tensorfold-native/nccl-2.32.3/lib/libnccl.so.2 \\
    TORCH_EXTENSIONS_DIR=/cache/sm70-source-cu1292-torch2141-cccl35-nccl2323
COPY --from=source_artifacts /native /opt/tensorfold-native
COPY --from=source_artifacts /probes /opt/tensorfold-probes
COPY --from=source_artifacts /build-inputs /opt/tensorfold-pins/build-inputs
COPY deploy/v100/source-pins.json /opt/tensorfold-pins/source-pins.json
COPY deploy/v100/source_foundation.py /opt/tensorfold-pins/source_foundation.py
RUN --network=none --mount=from=source_artifacts,source=/wheels,target=/source-wheels \\
    python -m pip install --no-deps /source-wheels/torch-*.whl \\
    && python -m pip check \\
    && python /opt/tensorfold-pins/source_foundation.py capture-install \\
       --pins /opt/tensorfold-pins/source-pins.json \\
       --build-receipt /opt/tensorfold-pins/build-inputs/source-build.json \\
       --receipt-root /opt/tensorfold-pins/build-inputs --wheel /source-wheels/torch-*.whl \\
       --output /opt/tensorfold-pins/v100-source-foundation.json \\
    && python /opt/tensorfold-pins/source_foundation.py audit
COPY . /src
RUN --network=none python -m pip wheel --no-build-isolation --no-deps /src --wheel-dir /wheels \\
    && python -m pip install --no-deps /wheels/tensorfold-*.whl \\
    && python -m pytest -c /dev/null -p no:cacheprovider -q /src/tests/test_v100_source_runtime.py
COPY tests /verification/tests
COPY tools /verification/tools
COPY tools/v100_probe.py /verification/v100_probe.py
RUN printf '[pytest]\\npythonpath = tests\\n' > /verification/pytest.ini
WORKDIR /verification
ENTRYPOINT ["/opt/venv/bin/python"]
'''
    return {"Dockerfile.source-builder": common, "Dockerfile": common + verification}


def runtime_identity():
    """Inspect this process's installed source framework and actual native providers.

    This runs in the image producer/audit process, never in a serving hot loop.
    Loading provider metadata does not create a CUDA context or prove device work.
    """
    import torch

    providers = ("libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9",
                 "/opt/tensorfold-native/nccl-2.32.3/lib/libnccl.so.2")
    handles = [ctypes.CDLL(name) for name in providers]
    properties = []
    for kind in range(3):
        value = ctypes.c_int()
        if handles[0].cublasGetProperty(kind, ctypes.byref(value)) != 0:
            raise ValueError("cuBLAS metadata query failed")
        properties.append(value.value)
    handles[1].cublasLtGetVersion.restype = ctypes.c_size_t
    handles[2].cudnnGetVersion.restype = ctypes.c_size_t
    nccl = ctypes.c_int()
    if handles[3].ncclGetVersion(ctypes.byref(nccl)) != 0:
        raise ValueError("NCCL metadata query failed")
    distribution = importlib.metadata.distribution("torch")
    mapped, driver = set(), set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/"):
            if parts[5].endswith(" (deleted)"):
                raise ValueError("foundation audit encountered a retired mapped artifact")
            path = Path(parts[5]).resolve(strict=True)
            if ".so" in path.name:
                # NVIDIA Container Toolkit supplies host driver DSOs at launch.
                # Bind their actual bytes in the audit result; they are not
                # immutable artifacts produced by this development image.
                owner = driver if path.name.startswith(("libcuda.so", "libnvidia-")) else mapped
                owner.add(str(path))
    return {"torch": str(torch.__version__), "distribution": distribution.version,
            "torch_path": str(Path(torch.__file__).resolve()), "cuda": torch.version.cuda,
            "torch_git": torch.version.git_version,
            "arch_flags": torch._C._cuda_getArchFlags(),
            "torch_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
            "cublas": properties, "cublas_lt": handles[1].cublasLtGetVersion(),
            "cudnn": handles[2].cudnnGetVersion(), "nccl": nccl.value,
            "framework_nccl": list(torch.cuda.nccl.version()),
            "framework_cudnn": torch.backends.cudnn.version(),
            "nccl_backend_available": torch.distributed.is_nccl_available(),
            "mapped_libraries": sorted(mapped), "host_driver_libraries": sorted(driver)}


def validate_runtime(identity):
    expected = {"torch": "2.14.1+tfcu129sm70", "distribution": "2.14.1+tfcu129sm70", "cuda": "12.9",
                "cublas": [12, 9, 2], "cublas_lt": 120902, "cudnn": 91002, "nccl": 23203,
                "framework_nccl": [2, 32, 3], "framework_cudnn": 91002, "nccl_backend_available": True}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("installed framework/provider versions differ from the selected source stack")
    if "sm_70" not in str(identity.get("arch_flags", "")).split():
        raise ValueError("source framework does not contain native SM70 targets")
    paths = identity["mapped_libraries"]
    for stem, prefix in (("libcublas", "/usr/local/cuda-12.9/"),
                         ("libcudnn", "/usr/lib/x86_64-linux-gnu/"),
                         ("libnccl", "/opt/tensorfold-native/nccl-2.32.3/lib/")):
        matches = [path for path in paths if Path(path).name.startswith(stem)]
        if not matches or any(not path.startswith(prefix) for path in matches):
            raise ValueError("mapped provider escaped its source-build native ownership: " + stem)
    if any("/site-packages/nvidia/" in path or "/torchvision" in path for path in paths):
        raise ValueError("binary-wheel accelerator providers cannot supply the source foundation")


def capture_install(pins, build_receipt, receipt_root, wheel_path=None):
    """Bind the actual installed wheel/native bytes to an administrator-built image.

    The source receipt is generated from the completed owned build, not accepted
    from a network caller. The image and receipt directory are the trust boundary.
    Device/model/compiler gates remain separate and cannot be self-certified here.
    """
    build, _ = read_json(build_receipt)
    if (build.get("schema") != "tensorfold.v100-source-build" or type(build.get("version")) is not int
            or build["version"] != 1
            or build.get("source_commits") != {name: effective_commit(source) for name, source in pins["sources"].items()}
            or build.get("targets") != pins["targets"]):
        raise ValueError("installed foundation needs the exact completed source-build receipt")
    name = build["wheel"]["artifact"]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("source wheel artifact must be a local basename")
    wheel = receipt_root / name if wheel_path is None else wheel_path
    if wheel.name != name or file_identity(wheel) != expected_identity(build["wheel"]):
        raise ValueError("installed framework source-wheel artifact differs from its build receipt")
    prefix = Path("/opt/tensorfold-native")
    installed = {str(path.relative_to(prefix)): file_identity(path) for path in prefix.rglob("*") if path.is_file()}
    if installed != build.get("native_prefix"):
        raise ValueError("installed native headers/libraries differ from the completed source build")
    producer = Path(__file__).resolve()
    if file_identity(producer) != build["producer_inputs"]["deploy/v100/source_foundation.py"]:
        raise ValueError("installed audit producer differs from the completed source-build input")
    identity = runtime_identity()
    validate_runtime(identity)
    if identity["torch_git"] != effective_commit(pins["sources"]["pytorch"]):
        raise ValueError("installed framework source revision differs from its completed build receipt")
    paths = {str(path.resolve()) for path in receipt_root.rglob("*") if path.is_file()}
    paths.add(str(producer))
    probes = build.get("probes", {})
    if set(probes) != {"cccl-primitives", "libraries-probe", "reconvergence-6111901"}:
        raise ValueError("completed build must identify every maintained standalone probe")
    for name, expected in probes.items():
        path = Path("/opt/tensorfold-probes") / name
        if file_identity(path) != expected_identity(expected):
            raise ValueError("installed standalone probe differs from its compiled receipt: " + name)
        paths.add(str(path.resolve(strict=True)))
    distribution = importlib.metadata.distribution("torch")
    for item in distribution.files or ():
        if str(item).endswith((".pyc", ".pyo")):
            continue
        paths.add(str(Path(distribution.locate_file(item)).resolve(strict=True)))
    for path in Path("/opt/tensorfold-native").rglob("*"):
        if path.is_file():
            paths.add(str(path.resolve(strict=True)))
    for directory, pattern in ((Path("/usr/local/cuda-12.9/targets/x86_64-linux/lib"), "*.so*"),
                               (Path("/usr/lib/x86_64-linux-gnu"), "libcudnn*.so*")):
        for path in directory.glob(pattern):
            if path.is_file():
                paths.add(str(path.resolve(strict=True)))
    paths.update(identity["mapped_libraries"])
    if not 1 <= len(paths) <= 100000:
        raise ValueError("installed foundation file inventory exceeded its bound")
    artifacts = [{"path": path, **file_identity(path)} for path in sorted(paths)]
    return {"schema": "tensorfold.v100-source-foundation-install", "version": 1,
            "pins_sha256": hashlib.sha256(canonical(pins)).hexdigest(),
            "build_receipt": {"path": str(build_receipt.resolve()), **file_identity(build_receipt)},
            "runtime": identity, "artifacts": artifacts,
            "include_roots": dict(INCLUDE_ROOTS),
            "evidence_class": "source-built-software-identity",
            "release_qualified": False,
            "pending_gates": ["compiler-6111901", "V100-device-graphs-collectives", "model-quality-performance"]}


def application_nccl_identity():
    """Check the real application consumer; a separate absolute CDLL is insufficient."""
    from tensorfold.cuda.comm import _library

    library = _library()
    path = Path(library._name).resolve(strict=True)
    expected = Path("/opt/tensorfold-native/nccl-2.32.3/lib/libnccl.so.2").resolve(strict=True)
    version = ctypes.c_int()
    if path != expected or library.ncclGetVersion(ctypes.byref(version)) != 0 or version.value != 23203:
        raise ValueError("application NCCL consumer differs from the selected source provider")
    return {"path": str(path), "version": version.value}


def audit_install(manifest, pins, *, process_mode=False):
    if (manifest.get("schema") != "tensorfold.v100-source-foundation-install"
            or type(manifest.get("version")) is not int or manifest["version"] != 1
            or manifest.get("pins_sha256") != hashlib.sha256(canonical(pins)).hexdigest()
            or manifest.get("evidence_class") != "source-built-software-identity"
            or manifest.get("release_qualified") is not False or manifest.get("include_roots") != dict(INCLUDE_ROOTS)):
        raise ValueError("unsupported installed source-foundation manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 100000:
        raise ValueError("installed foundation artifact inventory must be bounded")
    paths = set()
    for artifact in artifacts:
        path = Path(artifact["path"])
        if not path.is_absolute() or str(path) in paths or path.resolve(strict=True) != path:
            raise ValueError("installed foundation artifact paths must be unique canonical absolute paths")
        paths.add(str(path))
        if file_identity(path) != expected_identity(artifact):
            raise ValueError("installed source-foundation artifact changed: " + str(path))
    application = application_nccl_identity() if process_mode else None
    actual = runtime_identity()
    validate_runtime(actual)
    excluded = {"host_driver_libraries"}
    if process_mode:
        excluded.add("mapped_libraries")
    if ({key: value for key, value in actual.items() if key not in excluded} !=
            {key: value for key, value in manifest["runtime"].items() if key not in excluded}):
        raise ValueError("actual loaded source foundation differs from its installed identity")
    baseline, current = set(manifest["runtime"]["mapped_libraries"]), set(actual["mapped_libraries"])
    required = baseline if process_mode else current
    if not {*required, actual["torch_path"], manifest["build_receipt"]["path"]} <= paths:
        raise ValueError("installed source-foundation inventory omitted actual loaded/build artifacts")
    extra = []
    if process_mode:
        if not baseline <= current or application["path"] not in paths:
            raise ValueError("current process lost a known foundation/provider mapping")
        families = {Path(path).name.split(".so", 1)[0] for path in paths if ".so" in Path(path).name}
        for path in sorted(current - paths):
            if Path(path).name.split(".so", 1)[0] in families:
                raise ValueError("known foundation library family was relocated or substituted: " + path)
            extra.append({"path": path, **file_identity(path)})
    driver = [{"path": path, **file_identity(path)} for path in actual.get("host_driver_libraries", [])]
    runtime = ({"runtime": actual, "additional_process_libraries": extra, "host_driver_artifacts": driver,
                "application_nccl": application} if process_mode else actual)
    result = {"schema": ("tensorfold.v100-source-foundation-process-audit" if process_mode else
                         "tensorfold.v100-source-foundation-audit"), "version": 1,
            "software_integrity": "passed", "release_qualified": False,
            "evidence_class": manifest["evidence_class"], "pending_gates": manifest["pending_gates"],
            "runtime_sha256": hashlib.sha256(canonical(runtime)).hexdigest(), "artifact_count": len(paths),
            "host_driver_artifacts": driver}
    if process_mode:
        result.update(scope="current-process", additional_process_libraries=extra, application_nccl=application)
    return result


def audit_process(manifest, pins):
    """Explicit current-process authority; extra server/JIT code is separately bound.

    The caller privately owns the decoded protected manifest/pins for this call.
    Known image bytes/mappings remain mandatory. This is software-only evidence;
    numerical/device/compiler qualification belongs to the execution profile.
    """
    return audit_install(manifest, pins, process_mode=True)


def capture_build(pins, state, project):
    """Produce a completed-build receipt from actual outputs, never a success flag."""
    source = state / "sources/pytorch"
    cache_path = source / "build/CMakeCache.txt"
    cache = {}
    for line in read_document(cache_path).decode().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        name, value = line.split("=", 1)
        cache[name.split(":", 1)[0]] = value
    expected = {"CMAKE_CUDA_ARCHITECTURES": "70", "NCCL_VERSION": "2.32.3",
                "NCCL_VERSION_DEFINED": "1",
                "USE_CUDA": "1", "USE_CUDNN": "1", "USE_NCCL": "1", "USE_SYSTEM_NCCL": "1",
                "USE_FLASH_ATTENTION": "0",
                "CUB_INCLUDE_DIR": "/opt/tensorfold-native/cccl-3.5.0/include",
                "NCCL_INCLUDE_DIRS": "/opt/tensorfold-native/nccl-2.32.3/include",
                "NCCL_LIBRARIES": "/opt/tensorfold-native/nccl-2.32.3/lib/libnccl.so"}
    if any(cache.get(name) != value for name, value in expected.items()):
        raise ValueError("completed framework build has a different CMake provider/architecture configuration")
    wheels = list((state / "wheels").glob("torch-*.whl"))
    if len(wheels) != 1:
        raise ValueError("completed source build must produce exactly one framework wheel")
    name, version, _, _ = parse_wheel_filename(wheels[0].name)
    if name != "torch" or version != Version("2.14.1+tfcu129sm70"):
        raise ValueError("completed source framework wheel has the wrong name/version")
    capture_sources(pins, state / "sources", state / "receipts")
    inventories = {name: file_identity(state / "receipts" / (name + "-source-inventory.json"))
                   for name in pins["sources"]}
    generated_inputs = read_json(state / "receipts/generated-source-inputs.json")[0]
    symlinks = read_json(state / "receipts/source-symlink-closure.json")[0]
    tools = {}
    for path in ("/usr/local/cuda/bin/nvcc", "/usr/local/cuda/bin/ptxas", "/usr/local/cuda/nvvm/bin/cicc",
                 "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc", "/usr/bin/g++", "/usr/bin/gcc"):
        tools[path] = {"resolved_path": str(Path(path).resolve(strict=True)), **file_identity(path)}
    generated = {}
    for path in sorted((project / "deploy/v100").rglob("*")):
        if path.is_file() and path.suffix not in (".pyc", ".log") and "__pycache__" not in path.parts:
            generated[str(path.relative_to(project))] = file_identity(path)
    prefix = Path("/opt/tensorfold-native")
    native = {str(path.relative_to(prefix)): file_identity(path) for path in prefix.rglob("*") if path.is_file()}
    if not native:
        raise ValueError("completed source build has no selected native header/library installation")
    return {"schema": "tensorfold.v100-source-build", "version": 1, "targets": pins["targets"],
            "source_commits": {name: effective_commit(row) for name, row in pins["sources"].items()},
            "upstream_source_commits": {name: row["commit"] for name, row in pins["sources"].items()},
            "source_inventories": inventories, "wheel": {"artifact": wheels[0].name, **file_identity(wheels[0])},
            "generated_source_inputs": generated_inputs, "source_symlinks": symlinks,
            "probes": {name: file_identity(state / "build" / name) for name in
                       ("cccl-primitives", "libraries-probe", "reconvergence-6111901")},
            "cmake_cache": file_identity(cache_path), "cmake_configuration": cache,
            "build_environment": read_json(state / "receipts/framework-build-environment.json")[0],
            "toolchain": tools, "producer_inputs": generated, "native_prefix": native,
            "framework_private_cutlass": {"version": "4.6.1", "source": "pytorch/third_party/cutlass"},
            "native_device_execution": "unrun", "compiler_6111901": "unresolved-release-gate"}


def verify_native(pins, directory):
    for artifact in pins["native_packages"]:
        path = directory / artifact["artifact"]
        if path.name != artifact["artifact"] or not path.is_file():
            raise ValueError("native package must be a declared local regular artifact")
        if file_identity(path) != expected_identity(artifact):
            raise ValueError("native source-foundation package identity mismatch: " + path.name)


def verify_source_heads(pins, directory):
    for name, source in pins["sources"].items():
        if Path(name).name != name:
            raise ValueError("source checkout name must stay inside its declared directory")
        path = (directory / name).resolve(strict=True)
        head = subprocess.check_output(["git", "-c", "safe.directory=" + str(path), "-C", str(path),
                                        "rev-parse", "HEAD"]).decode().strip()
        if head != effective_commit(source):
            raise ValueError("source checkout does not match its selected immutable revision: " + name)


def acquire_sources(pins, directory, patch_directory, lint_tool=None):
    """Acquire immutable commits/submodules without rewriting an existing checkout."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in pins["sources"].items():
        if Path(name).name != name:
            raise ValueError("source checkout name must stay inside its declared directory")
        path = (directory / name).resolve()
        def git(*arguments):
            subprocess.run(["git", "-c", "safe.directory=" + str(path), "-C", str(path), *arguments], check=True)
        if not path.exists():
            path.mkdir()
            git("init")
            git("remote", "add", "origin", source["repository"])
            git("fetch", "--depth=1", "--filter=blob:none", "origin", source["commit"])
            git("checkout", "--detach", "FETCH_HEAD")
        head = subprocess.check_output(["git", "-c", "safe.directory=" + str(path), "-C", str(path),
                                        "rev-parse", "HEAD"]).decode().strip()
        origin = subprocess.check_output(["git", "-c", "safe.directory=" + str(path), "-C", str(path),
                                          "remote", "get-url", "origin"]).decode().strip()
        if origin != source["repository"] or head not in (source["commit"], effective_commit(source)):
            raise ValueError("existing source checkout does not match its immutable commit/origin: " + name)
        if head != effective_commit(source):
            apply_owned_patches(path, source, patch_directory, lint_tool=lint_tool)
        git("submodule", "update", "--init", "--recursive", "--jobs=4")
    verify_source_heads(pins, directory)


def apply_owned_patches(path, source, directory, *, lint_tool=None):
    """Reproduce the one recorded downstream source fork, before any compilation."""
    patches = source.get("owned_patches", [])
    if not isinstance(patches, list) or not 1 <= len(patches) <= 16:
        raise ValueError("effective source fork needs a bounded recorded patch list")
    if lint_tool is None or not lint_tool.is_file() or not os.access(lint_tool, os.X_OK):
        raise ValueError("owned source fork requires its prepared Spin/lintrunner environment before committing")
    if "Test Plan" not in source["fork_message"] or "```" not in source["fork_message"]:
        raise ValueError("owned source fork message must preserve its root-cause/fix and fenced Test Plan")
    total, arguments = 0, []
    for row in patches:
        name = row["artifact"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("owned source patch must be a declared local basename")
        patch = directory / name
        identity = file_identity(patch)
        total += identity["bytes"]
        if total > 1 << 20 or identity != expected_identity(row):
            raise ValueError("owned source patch bytes differ from their bounded recorded identity")
        arguments.append(str(patch.resolve(strict=True)))
    base = ["git", "-c", "safe.directory=" + str(path), "-C", str(path)]
    subprocess.run([*base, "diff", "--exit-code", "HEAD", "--"], check=True)
    subprocess.run([*base, "apply", "--check", *arguments], check=True)
    subprocess.run([*base, "apply", "--index", *arguments], check=True)
    environment = os.environ.copy()
    environment["PATH"] = str(lint_tool.parent) + os.pathsep + environment.get("PATH", "")
    changed = subprocess.check_output([*base, "diff", "--name-only", "-z", "HEAD"]).split(b"\0")
    changed = [name.decode("utf-8") for name in changed if name]
    # The supported Spin command dispatches the required lintrunner -a on the
    # actual uncommitted patch. No formatter change can silently enter a fork.
    subprocess.run([str(lint_tool), "quicklint", "--", "-a", *changed], cwd=path, env=environment, check=True)
    subprocess.run([*base, "diff", "--exit-code"], check=True)
    environment["GIT_AUTHOR_DATE"] = source["fork_date"]
    environment["GIT_COMMITTER_DATE"] = source["fork_date"]
    subprocess.run([*base, "-c", "user.name=TensorFold Source Builder", "-c",
                    "user.email=source-builder@tensorfold.invalid", "-c", "commit.gpgsign=false", "-c",
                    "core.hooksPath=/dev/null", "commit", "-m", source["fork_message"]],
                   env=environment, check=True)
    actual = subprocess.check_output([*base, "rev-parse", "HEAD"]).decode().strip()
    if actual != effective_commit(source):
        raise ValueError("reproduced owned source fork has a different immutable commit")


def generated_source_paths(root, build_directory):
    """Identify outputs declared by the canonical configured build, not source globs."""
    if build_directory is None or not (build_directory / "build.ninja").is_file():
        return set()
    cache = read_document(build_directory / "CMakeCache.txt").decode()
    if "CMAKE_HOME_DIRECTORY:INTERNAL=" + str(root.resolve()) + "\n" not in cache:
        raise ValueError("generated-source graph belongs to a different source checkout")
    raw = subprocess.check_output(["ninja", "-C", str(build_directory), "-t", "targets", "all"])
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise ValueError("generated-source graph exceeds its metadata bound")
    result = set(PYTORCH_SIDE_OUTPUTS)
    for line in raw.decode().splitlines():
        name, kind = line.rsplit(": ", 1)
        if kind == "phony":
            continue
        path = Path(name)
        if not path.is_absolute():
            path = build_directory / path
        path = path.absolute()
        if path.is_relative_to(root):
            result.add(str(path.relative_to(root)))
    return result


def verify_source_links(root, inventory):
    result = []
    root = root.resolve(strict=True)
    for row in inventory["files"]:
        if row["mode"] == "120000":
            target = (root / row["path"]).resolve(strict=True)
            if not target.is_relative_to(root):
                raise ValueError("build source symlink escapes the inventoried closure: " + row["path"])
            result.append({"path": row["path"], "target": str(target.relative_to(root))})
    return result


def capture_sources(pins, directory, receipts):
    """Validate all raw inputs before a standalone build stage can consume them."""
    verify_source_heads(pins, directory)
    receipts.mkdir(parents=True, exist_ok=True)
    generated_inputs, links = {}, {}
    for name, source in pins["sources"].items():
        root = (directory / name).resolve(strict=True)
        build = root / "build" if name == "pytorch" and (root / "build/build.ninja").is_file() else None
        generated = generated_source_paths(root, build) if name == "pytorch" else set()
        # CCCLClangdCompileInfo.cmake creates this IDE-only link even for a
        # header installation. It is metadata, not a compiler source input.
        metadata_links = ({"compile_commands.json": directory.parent / "build/cccl/compile_commands.json"}
                          if name == "cccl" else {})
        inventory = source_inventory(root, effective_commit(source), build_directory=build,
                                     generated_paths=generated, generated_inputs=generated_inputs,
                                     input_prefix=name + "/", cache_roots=(".lintbin",) if name == "pytorch" else (),
                                     metadata_links=metadata_links)
        links[name] = verify_source_links(root, inventory)
        raw = canonical(inventory) + b"\n"
        if hashlib.sha256(raw).hexdigest() != source["source_inventory_sha256"]:
            raise ValueError("source inputs differ from the exact recorded raw-byte inventory: " + name)
        (receipts / (name + "-source-inventory.json")).write_bytes(raw)
    (receipts / "generated-source-inputs.json").write_bytes(canonical(generated_inputs) + b"\n")
    (receipts / "source-symlink-closure.json").write_bytes(canonical(links) + b"\n")


def source_inventory(root, expected_commit, *, build_directory=None, generated_paths=(),
                     generated_inputs=None, input_prefix="", cache_roots=(), metadata_links=None):
    """Validate Git's complete recursive input tree and hash actual source bytes.

    Generated object/codegen outputs are excluded: their authoritative generators
    and inputs are tracked here, and output identities belong in build receipts.
    """
    root = root.resolve(strict=True)
    pending, repositories, files = [(root, "", expected_commit)], [], []
    while pending:
        repository, prefix, wanted = pending.pop()
        def git(*arguments):
            return subprocess.check_output(["git", "-c", "safe.directory=" + str(repository),
                                            "-C", str(repository), *arguments])
        actual = git("rev-parse", "HEAD").decode().strip()
        if actual != wanted:
            raise ValueError("source/submodule commit differs from its declared Git tree: " + prefix)
        repositories.append({"path": prefix or ".", "commit": actual})
        entries = [entry for entry in git("ls-tree", "-r", "-z", "HEAD").split(b"\0") if entry]
        tracked = {entry.split(b"\t", 1)[1].decode("utf-8") for entry in entries}
        # Include ignored files: an ignored C++ file can still enter a CMake
        # GLOB. Only explicit build outputs, source-derived Python caches and
        # the owned binary/linter cache directories are outside the Git input.
        for raw_name in git("ls-files", "--others", "-z").split(b"\0"):
            if not raw_name:
                continue
            name = raw_name.decode("utf-8")
            path, logical = repository / name, prefix + name
            if metadata_links and logical in metadata_links:
                wanted = str(metadata_links[logical])
                if not path.is_symlink() or os.readlink(path) != wanted:
                    raise ValueError("generated IDE metadata link differs from its declared build output")
                if generated_inputs is not None:
                    payload = os.fsencode(wanted)
                    generated_inputs[input_prefix + logical] = {
                        "kind": "generated-ide-metadata-link", "target": wanted,
                        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                continue
            if build_directory is not None and path.is_relative_to(build_directory):
                continue
            if not prefix and any(name.startswith(directory + "/") for directory in cache_roots):
                continue
            python_cache = False
            if "__pycache__" in path.parts and path.suffix == ".pyc":
                try:
                    original = Path(importlib.util.source_from_cache(str(path)))
                    original = str(original.relative_to(repository))
                    python_cache = original in tracked or prefix + original in generated_paths
                except ValueError:
                    pass
            if logical not in generated_paths and not python_cache:
                raise ValueError("untracked non-generated source input: " + input_prefix + logical)
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("generated source input escapes the inventoried closure: " + logical)
            if generated_inputs is not None:
                generated_inputs[input_prefix + logical] = file_identity(path)
        for entry in entries:
            if not entry:
                continue
            head, raw_name = entry.split(b"\t", 1)
            mode, kind, blob = head.decode().split()
            name = raw_name.decode("utf-8")
            path = repository / name
            logical = prefix + name
            if mode == "160000":
                pending.append((path, logical + "/", blob))
                continue
            if kind != "blob" or mode not in ("100644", "100755", "120000"):
                raise ValueError("unsupported source Git object: " + logical)
            if mode == "120000":
                content = os.fsencode(os.readlink(path))
                size = len(content)
                sha = hashlib.sha256(content)
                git_sha = hashlib.sha1(f"blob {size}\0".encode() + content)
            else:
                with open(path, "rb", opener=lambda name, flags: os.open(
                        name, flags | os.O_NONBLOCK | os.O_NOFOLLOW)) as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError("tracked source blob is not a regular file: " + logical)
                    size, seen = info.st_size, 0
                    sha, git_sha = hashlib.sha256(), hashlib.sha1(f"blob {size}\0".encode())
                    while seen < size:
                        block = stream.read(min(1 << 20, size - seen))
                        if not block:
                            break
                        sha.update(block)
                        git_sha.update(block)
                        seen += len(block)
                    after = os.fstat(stream.fileno())
                    if (seen != size or stream.read(1) or (after.st_size, after.st_mtime_ns, after.st_ctime_ns) !=
                            (info.st_size, info.st_mtime_ns, info.st_ctime_ns)):
                        raise ValueError("tracked source changed during inventory: " + logical)
            transformed = git_sha.hexdigest() != blob
            # Git checkout applies recorded attributes, notably CRLF for .bat
            # files. Reuse Git's clean conversion rather than guessing it; raw
            # SHA256 still binds the exact bytes consumed by this build.
            clean = (git("hash-object", "--path=" + name, str(path)).decode().strip()
                     if transformed and mode != "120000" else git_sha.hexdigest())
            if clean != blob:
                raise ValueError("source bytes differ from the pinned Git blob: " + logical)
            files.append({"path": logical, "mode": mode, "bytes": size, "sha256": sha.hexdigest(),
                          "git_blob": blob, "checkout_transform": transformed})
    return {"schema": "tensorfold.source-input-inventory", "version": 1,
            "repositories": sorted(repositories, key=lambda row: row["path"]),
            "files": sorted(files, key=lambda row: row["path"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render-build-tools")
    render.add_argument("--report", type=Path, required=True)
    render.add_argument("--lock", type=Path, required=True)
    render.add_argument("--receipt", type=Path, required=True)
    render.add_argument("--bootstrap-lock", type=Path)
    runtime = commands.add_parser("render-runtime")
    runtime.add_argument("--report", type=Path, required=True)
    runtime.add_argument("--lock", type=Path, required=True)
    runtime.add_argument("--receipt", type=Path, required=True)
    lint = commands.add_parser("render-lint-tools")
    lint.add_argument("--report", type=Path, required=True)
    lint.add_argument("--lock", type=Path, required=True)
    lint.add_argument("--receipt", type=Path, required=True)
    offline = commands.add_parser("offline-lock")
    offline.add_argument("--receipt", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    offline.add_argument("--bootstrap", action="store_true")
    apt = commands.add_parser("capture-apt")
    apt.add_argument("--directory", type=Path, required=True)
    apt.add_argument("--pins", type=Path, required=True)
    apt.add_argument("--expected-packages", type=Path, required=True)
    apt.add_argument("--output", type=Path, required=True)
    apt.add_argument("--checksums", type=Path, required=True)
    apt.add_argument("--download-uris", type=Path)
    docker = commands.add_parser("render-dockerfiles")
    docker.add_argument("--pins", type=Path, required=True)
    docker.add_argument("--directory", type=Path, required=True)
    verify = commands.add_parser("verify-native")
    verify.add_argument("--pins", type=Path, required=True)
    verify.add_argument("--directory", type=Path, required=True)
    heads = commands.add_parser("verify-source-heads")
    heads.add_argument("--pins", type=Path, required=True)
    heads.add_argument("--directory", type=Path, required=True)
    acquire = commands.add_parser("acquire-sources")
    acquire.add_argument("--pins", type=Path, required=True)
    acquire.add_argument("--directory", type=Path, required=True)
    acquire.add_argument("--lint-tool", type=Path)
    sources = commands.add_parser("capture-sources")
    sources.add_argument("--pins", type=Path, required=True)
    sources.add_argument("--directory", type=Path, required=True)
    sources.add_argument("--receipts", type=Path, required=True)
    capture = commands.add_parser("capture-source")
    capture.add_argument("--source", type=Path, required=True)
    capture.add_argument("--commit", required=True)
    capture.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("capture-build")
    build.add_argument("--pins", type=Path, required=True)
    build.add_argument("--state", type=Path, required=True)
    build.add_argument("--project", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    install = commands.add_parser("capture-install")
    install.add_argument("--pins", type=Path, required=True)
    install.add_argument("--build-receipt", type=Path, required=True)
    install.add_argument("--receipt-root", type=Path, required=True)
    install.add_argument("--wheel", type=Path)
    install.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--pins", type=Path, default=Path("/opt/tensorfold-pins/source-pins.json"))
    audit.add_argument("--manifest", type=Path, default=Path("/opt/tensorfold-pins/v100-source-foundation.json"))
    process_audit = commands.add_parser("audit-process")
    process_audit.add_argument("--pins", type=Path, default=Path("/opt/tensorfold-pins/source-pins.json"))
    process_audit.add_argument("--manifest", type=Path, default=Path("/opt/tensorfold-pins/v100-source-foundation.json"))
    args = parser.parse_args()
    if args.command == "verify-native":
        verify_native(read_json(args.pins)[0], args.directory)
        return
    if args.command == "offline-lock":
        args.output.write_text(offline_lock(read_json(args.receipt)[0], bootstrap=args.bootstrap))
        return
    if args.command == "capture-apt":
        captured = capture_apt(args.directory, read_json(args.pins)[0], args.expected_packages, args.download_uris)
        args.output.write_bytes(canonical(captured) + b"\n")
        args.checksums.write_text("".join(f"{row['sha256']}  {row['artifact']}\n" for row in captured["packages"]))
        return
    if args.command == "render-dockerfiles":
        for name, content in dockerfiles(read_json(args.pins)[0]).items():
            (args.directory / name).write_text(content)
        return
    if args.command == "verify-source-heads":
        verify_source_heads(read_json(args.pins)[0], args.directory)
        return
    if args.command == "acquire-sources":
        acquire_sources(read_json(args.pins)[0], args.directory, args.pins.parent / "framework-patches", args.lint_tool)
        return
    if args.command == "capture-sources":
        capture_sources(read_json(args.pins)[0], args.directory, args.receipts)
        return
    if args.command == "capture-source":
        captured = source_inventory(args.source, args.commit)
        args.output.write_bytes(canonical(captured) + b"\n")
        return
    if args.command == "capture-build":
        captured = capture_build(read_json(args.pins)[0], args.state, args.project)
        args.output.write_bytes(canonical(captured) + b"\n")
        return
    if args.command == "capture-install":
        captured = capture_install(read_json(args.pins)[0], args.build_receipt, args.receipt_root.resolve(strict=True),
                                   wheel_path=args.wheel)
        args.output.write_bytes(canonical(captured) + b"\n")
        return
    if args.command in ("audit", "audit-process"):
        manifest, raw = read_json(args.manifest)
        function = audit_process if args.command == "audit-process" else audit_install
        result = function(manifest, read_json(args.pins)[0])
        result["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        print(canonical(result).decode())
        return
    report, raw = read_json(args.report)
    lock, receipt = render_build_report(report, hashlib.sha256(raw).hexdigest(), runtime=args.command == "render-runtime",
                                        lint=args.command == "render-lint-tools")
    args.lock.write_text(lock)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if getattr(args, "bootstrap_lock", None) is not None:
        rows = ["# Generated by deploy/v100/source_foundation.py from the same native dependency report.\n"]
        for name in ("packaging", "pip"):
            package = receipt["packages"][name]
            rows.append(f"{name} @ {package['url']} --hash=sha256:{package['sha256']}\n")
        args.bootstrap_lock.write_text("".join(rows))


if __name__ == "__main__":
    main()
