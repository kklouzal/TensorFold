#!/usr/bin/env python3
"""Pinned CPU media inputs for the separate V100 source-image producer.

Network acquisition is bounded and resumable; only complete exact artifacts
publish. This tool does not select an alternate processor or qualify media/model
execution. Source builds and installed semantic controls are separate gates.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import os
from pathlib import Path
import stat
import subprocess
import tarfile
from urllib.parse import urlsplit

from source_foundation import canonical, expected_identity, file_identity, read_json, runtime_identity as foundation_runtime_identity


def source_inventory(row, downloads, sources):
    """Verify every archive input and separately bind completed generated cache bytes."""
    archive, root = downloads / row["artifact"], sources / row["name"]
    if root.is_symlink() or not root.is_dir():
        raise ValueError("media source root must be an owned non-aliased directory")
    if file_identity(archive) != expected_identity(row):
        raise ValueError("media source archive differs from its exact pin")
    originals, generated, names, total, unpacked = [], [], set(), 0, 0
    with tarfile.open(archive) as reader:
        for member in reader:
            unpacked += max(0, member.size)
            if len(originals) >= 200000 or unpacked > row["extraction_max_bytes"]:
                raise ValueError("media source archive exceeded its extraction budget")
            relative = Path(*Path(member.name).parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("media archive path escapes its source root")
            path = root / relative
            if member.isfile():
                with reader.extractfile(member) as stream:
                    wanted = {"bytes": member.size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
                if path.is_symlink() or file_identity(path) != wanted:
                    raise ValueError("media source input changed: " + str(path))
                originals.append({"path": str(relative), **wanted})
                names.add(str(relative))
            elif member.issym():
                if (not path.is_symlink() or os.readlink(path) != member.linkname
                        or not path.resolve(strict=True).is_relative_to(root.resolve(strict=True))):
                    raise ValueError("media source link differs or leaves its archive closure")
                originals.append({"path": str(relative), "symlink": member.linkname})
                names.add(str(relative))
            elif not member.isdir():
                raise ValueError("media source archive contains an unsupported entry type")
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if relative in names:
            continue
        if path.is_symlink():
            raise ValueError("generated media source cache may not introduce aliases")
        if path.is_dir():
            continue
        total += path.stat(follow_symlinks=False).st_size
        if len(generated) >= 200000 or total > 2 << 30:
            raise ValueError("generated media source cache exceeded its byte/file budget")
        identity = file_identity(path)
        generated.append({"path": relative, **identity})
    return {"schema": "tensorfold.v100-media-source-cache", "version": 1,
            "source": row["name"], "archive": expected_identity(row),
            "original_files": len(originals), "original_inventory_sha256": hashlib.sha256(canonical(originals)).hexdigest(),
            "generated_outputs": sorted(generated, key=lambda row: row["path"])}


def verify_source(row, downloads, sources, receipt=None):
    actual = source_inventory(row, downloads, sources)
    if receipt is not None and receipt.exists():
        if read_json(receipt)[0] != actual:
            raise ValueError("media source cache differs from its last completed stage; prepare a fresh source root")
    elif actual["generated_outputs"]:
        raise ValueError("unrecorded generated media source inputs; prepare a fresh source root")
    return actual


def prefix_inventory(prefix):
    result = {}
    for path in prefix.rglob("*"):
        if len(result) >= 100000:
            raise ValueError("media native prefix exceeded its file inventory budget")
        if path.is_symlink():
            if not path.resolve(strict=True).is_relative_to(prefix.resolve(strict=True)):
                raise ValueError("media native alias leaves its owned prefix")
            result[str(path.relative_to(prefix))] = {"symlink": os.readlink(path)}
        elif path.is_file():
            result[str(path.relative_to(prefix))] = file_identity(path)
    return result


def capture_build(pins, state, project):
    inputs = {}
    for row in pins["artifacts"]:
        if row["kind"] == "source-archive":
            receipt = state / "receipts" / ("source-" + row["name"] + ".json")
            verify_source(row, state / "downloads", state / "sources", receipt)
            inputs[row["name"]] = file_identity(receipt)
    wheels = sorted((state / "wheels").glob("*.whl"))
    if len(wheels) != 3 or {path.name.split("-", 1)[0] for path in wheels} != {"av", "torchvision", "transformers"}:
        raise ValueError("media export needs exactly its three completed source wheels")
    owned = ["source_foundation.py", "media_foundation.py", "media_build.sh", "media_probe.py", "Dockerfile.media", "media-pins.json",
             "media-native-apt.json", "media-native-apt.sha256", "media-runtime-amd64.json", "media-build-amd64.json",
             "media-build-constraints.txt", "requirements-media-runtime-amd64.lock",
             "requirements-media-runtime-offline-amd64.lock", "requirements-media-build-amd64.lock",
             "requirements-media-build-offline-amd64.lock"]
    return {"schema": "tensorfold.v100-media-build", "version": 1,
            "pins_sha256": hashlib.sha256(canonical(pins)).hexdigest(), "source_receipts": inputs,
            "source_wheels": {path.name: file_identity(path) for path in wheels},
            "native_prefix": prefix_inventory(state / "native"),
            "producer_inputs": {name: file_identity(project / "deploy/v100" / name) for name in owned},
            "release_qualified": False}


def runtime_inventory():
    """Describe actual registries and pinned consumers; registration alone is no decode proof."""
    import av
    import PIL
    from PIL import features
    import torch
    import torchvision
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    decoders = []
    for name in sorted(av.codecs_available):
        try:
            codec = av.Codec(name, "r")
        except av.codec.codec.UnknownCodecError:
            continue
        decoders.append({"name": name, "type": codec.type, "description": codec.long_name})
    demuxers = []
    for name in sorted(av.formats_available):
        container = av.ContainerFormat(name)
        if container.is_input:
            demuxers.append({"name": name, "description": container.long_name})
    prefix = Path("/opt/tensorfold-media/ffmpeg-9.0.2/lib")
    library = ctypes.CDLL(str(prefix / "libavutil.so.61"))
    library.avutil_configuration.restype = ctypes.c_char_p
    configuration = library.avutil_configuration().decode("utf-8")
    dav1d = ctypes.CDLL("/opt/tensorfold-media/dav1d-1.5.4/lib/libdav1d.so")
    dav1d.dav1d_version.restype = ctypes.c_char_p
    mapped = foundation_runtime_identity()["mapped_libraries"]
    providers = {}
    for stem in (*av.library_versions, "libdav1d"):
        matches = sorted(path for path in mapped if Path(path).name.startswith(stem + ".so"))
        owner = Path("/opt/tensorfold-media/dav1d-1.5.4/lib") if stem == "libdav1d" else prefix
        expected_root = owner.resolve(strict=True)
        if len(matches) != 1 or not Path(matches[0]).is_relative_to(expected_root):
            raise ValueError("loaded media provider differs from its owned prefix: " + stem)
        providers[stem] = matches[0]
    if not torchvision.extension._has_ops():
        raise ValueError("source CPU TorchVision native operators did not load")
    return {"packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "torchvision", "transformers", "av", "pillow")},
            "torch_source": torch.version.git_version, "torch_cuda": torch.version.cuda,
            "torchvision_cuda": torchvision.extension._check_cuda_version(),
            "processor_classes": [item.__module__ + "." + item.__name__ for item in
                                  (Qwen2VLImageProcessor, Qwen3VLVideoProcessor, Qwen4ExpVisionModel)],
            "image_backend": "torchvision", "decoder_backend": "pyav",
            "ffmpeg_versions": {key: list(value) for key, value in av.library_versions.items()},
            "loaded_media_providers": providers,
            "ffmpeg_configuration": configuration, "decoders": decoders, "demuxers": demuxers,
            "decoder_inventory_scope": "registered decoder names, not per-codec execution qualification",
            "cpu_av1_provider": "libdav1d",
            "dav1d_version": dav1d.dav1d_version().decode("utf-8"),
            "known_unavailable_cpu_decoders": {"av1": "FFmpeg native av1 is hardware-only; CPU AV1 uses libdav1d"},
            "pillow": {"version": PIL.__version__, "jpeg": features.version_codec("jpg"),
                       "zlib": features.version_codec("zlib"), "webp": features.version_module("webp")}}


def installed_paths(package_names, prefix, metadata):
    paths = {str(Path(__file__).resolve(strict=True))}
    for name in package_names:
        distribution = importlib.metadata.distribution(name)
        for item in distribution.files or ():
            if not str(item).endswith((".pyc", ".pyo")):
                paths.add(str(Path(distribution.locate_file(item)).resolve(strict=True)))
    for root in (prefix, metadata):
        for path in root.rglob("*"):
            if path.is_file() and path.name != "media-install.json":
                paths.add(str(path.resolve(strict=True)))
    # These exact Debian libraries are used by TorchVision's CPU image extension.
    for pattern in ("libjpeg.so*", "libpng16.so*", "libwebp*.so*", "libsharpyuv.so*"):
        for path in Path("/usr/lib/x86_64-linux-gnu").glob(pattern):
            if path.is_file():
                paths.add(str(path.resolve(strict=True)))
    if not 1 <= len(paths) <= 100000:
        raise ValueError("installed media inventory exceeded its file budget")
    return paths


def capture_install(pins, runtime, base_runtime, metadata, wheels):
    """Seal added media bytes without changing the selected Torch/native provider authority."""
    build = read_json(metadata / "media-build.json")[0]
    if (build.get("schema") != "tensorfold.v100-media-build" or type(build.get("version")) is not int
            or build["version"] != 1 or build.get("pins_sha256") != hashlib.sha256(canonical(pins)).hexdigest()
            or prefix_inventory(Path("/opt/tensorfold-media")) != build.get("native_prefix")
            or file_identity(Path(__file__)) != build["producer_inputs"]["media_foundation.py"]
            or file_identity(Path("/opt/tensorfold-pins/source_foundation.py")) != build["producer_inputs"]["source_foundation.py"]):
        raise ValueError("installed media bytes differ from their completed source-build receipt")
    if {path.name: file_identity(path) for path in wheels.glob("*.whl")} != build.get("source_wheels"):
        raise ValueError("installed media source-wheel inputs differ from the completed build")
    for name, expected in build["source_receipts"].items():
        if file_identity(metadata / ("source-" + name + ".json")) != expected_identity(expected):
            raise ValueError("installed media source receipt differs from the completed build")
    selected = sorted((runtime["packages"].keys() - base_runtime["packages"].keys()) |
                      {"torchvision", "transformers", "av"})
    for name, row in runtime["packages"].items():
        if importlib.metadata.version(name) != row["version"]:
            raise ValueError("installed support dependency differs from its media lock: " + name)
    for row in pins["artifacts"]:
        if row["kind"] in ("processor-config", "decoder-fixture") and file_identity(metadata / row["artifact"]) != expected_identity(row):
            raise ValueError("installed official processor configuration differs from its pin")
    actual = runtime_inventory()
    expected = {"torchvision": "0.29.1+tfcpu", "transformers": "5.21.0.dev0", "av": "19.0.2", "pillow": "13.0.0"}
    if (any(actual["packages"][name] != version for name, version in expected.items())
            or actual["torchvision_cuda"] != -1):
        raise ValueError("installed media providers differ from the pinned CPU source build")
    paths = installed_paths(selected, Path("/opt/tensorfold-media"), metadata)
    return {"schema": "tensorfold.v100-media-install", "version": 1,
            "pins_sha256": hashlib.sha256(canonical(pins)).hexdigest(),
            "packages": selected, "runtime": actual,
            "artifacts": [{"path": path, **file_identity(path)} for path in sorted(paths)],
            "evidence_class": "CPU-source-built-media-software", "release_qualified": False,
            "pending_gates": ["service-input-bounds-and-format-parity", "SC117-mmproj-model-equivalence",
                              "V100-device-resource-quality-performance", "compiler-6111901"]}


def audit_install(manifest, pins):
    if (manifest.get("schema") != "tensorfold.v100-media-install" or type(manifest.get("version")) is not int
            or manifest["version"] != 1 or manifest.get("release_qualified") is not False
            or manifest.get("pins_sha256") != hashlib.sha256(canonical(pins)).hexdigest()):
        raise ValueError("unsupported installed media software manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 100000:
        raise ValueError("media artifact inventory must be bounded")
    seen = set()
    for row in artifacts:
        path = Path(row["path"])
        if not path.is_absolute() or path.resolve(strict=True) != path or str(path) in seen:
            raise ValueError("media artifact paths must be unique canonical absolute paths")
        seen.add(str(path))
        if file_identity(path) != expected_identity(row):
            raise ValueError("installed media software artifact changed: " + str(path))
    if (installed_paths(manifest["packages"], Path("/opt/tensorfold-media"), Path("/opt/tensorfold-media-pins")) != seen
            or runtime_inventory() != manifest["runtime"]):
        raise ValueError("actual installed media software differs from its sealed inventory")
    return {"schema": "tensorfold.v100-media-audit", "version": 1, "software_integrity": "passed",
            "artifact_count": len(seen), "runtime_sha256": hashlib.sha256(canonical(manifest["runtime"])).hexdigest(),
            "release_qualified": False, "pending_gates": manifest["pending_gates"]}


def fetch_artifact(row, directory, *, deadline_s=900):
    name, url = row["artifact"], row["url"]
    wanted = expected_identity(row)
    if (not isinstance(name, str) or Path(name).name != name or name.startswith("-")
            or not isinstance(url, str) or any(char.isspace() for char in url)):
        raise ValueError("media input needs an exact local basename and bounded public URL")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment
            or parsed.hostname not in ("codeload.github.com", "github.com", "ffmpeg.org",
                                        "files.pythonhosted.org", "huggingface.co",
                                        "archive.ubuntu.com", "security.ubuntu.com",
                                        "download.videolan.org", "fate-suite.ffmpeg.org")):
        raise ValueError("media source input must use a declared official HTTPS origin")
    if type(deadline_s) is not int or not 1 <= deadline_s <= 3600 or not 0 < wanted["bytes"] <= 2 << 30:
        raise ValueError("media input transfer exceeds its declared deadline/byte budget")
    directory.mkdir(parents=True, exist_ok=True)
    target, partial = directory / name, directory / (name + ".part")
    try:
        complete = target.stat(follow_symlinks=False)
    except FileNotFoundError:
        complete = None
    if complete is not None:
        if not stat.S_ISREG(complete.st_mode):
            raise ValueError("cached media artifact must be a non-aliased regular file")
        if file_identity(target) != wanted:
            raise ValueError("cached media artifact identity differs from its pin: " + name)
        return "reused"
    try:
        info = partial.stat(follow_symlinks=False)
    except FileNotFoundError:
        info = None
    if info is not None:
        if not stat.S_ISREG(info.st_mode) or info.st_size > wanted["bytes"]:
            raise ValueError("media partial artifact must be a bounded regular file")
    command = ["curl", "--fail", "--location", "--proto", "=https", "--proto-redir", "=https",
               "--continue-at", "-", "--connect-timeout", "20", "--max-time", str(deadline_s),
               "--max-filesize", str(wanted["bytes"]), "--output", str(partial), url]
    result = subprocess.run(command, capture_output=True, text=True, timeout=deadline_s + 30)
    if result.returncode:
        raise RuntimeError(f"media transfer {name} refused (curl exit {result.returncode}); partial retained")
    if file_identity(partial) != wanted:
        raise ValueError("downloaded media artifact differs from its exact pin; partial retained: " + name)
    os.replace(partial, target)
    return "downloaded"


def extract_source(row, downloads, sources, receipt=None):
    """Extract only the pinned archive into one source root, never over a cache."""
    name = row["name"]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("media source name must remain inside its owned directory")
    archive = downloads / row["artifact"]
    target, pending = sources / name, sources / (name + ".extract")
    if pending.exists():
        raise ValueError("media source extraction is incomplete; inspect its retained pending root")
    if target.exists():
        verify_source(row, downloads, sources, receipt)
        return target
    if file_identity(archive) != expected_identity(row):
        raise ValueError("media source archive differs from its exact pin")
    maximum = row["extraction_max_bytes"]
    if type(maximum) is not int or not 0 < maximum <= 2 << 30:
        raise ValueError("media source needs a declared cumulative extraction bound")
    pending.mkdir(parents=True)
    with tarfile.open(archive) as stream:
        members, total = [], 0
        for member in stream:
            total += max(member.size, 0)
            if total > maximum or len(members) >= 200000:
                raise ValueError("media source archive exceeded its extraction budget")
            members.append(member)
        stream.extractall(pending, members=members, filter="data")
    roots = list(pending.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("media source archive must have one declared source root")
    roots[0].rename(target)
    pending.rmdir()
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "fetch-native", "extract", "verify-source", "record-source", "capture-build", "capture-install", "audit"))
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--downloads", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--base-runtime", type=Path)
    parser.add_argument("--metadata", type=Path, default=Path("/opt/tensorfold-media-pins"))
    parser.add_argument("--source-name")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--wheels", type=Path)
    args = parser.parse_args()
    pins = read_json(args.pins)[0]
    if args.command == "fetch-native":
        if (pins.get("schema") != "tensorfold.v100-source-apt-bundle" or type(pins.get("version")) is not int
                or pins["version"] != 1 or args.downloads is None):
            raise ValueError("native media acquisition needs its exact Debian artifact manifest and directory")
        for row in pins["packages"]:
            print(canonical({"artifact": row["artifact"], "status": fetch_artifact(row, args.downloads)}).decode())
        return
    if (pins.get("schema") != "tensorfold.v100-media-inputs" or type(pins.get("version")) is not int
            or pins["version"] != 1):
        raise ValueError("unsupported media source input manifest")
    if args.command == "capture-build":
        if args.state is None or args.project is None:
            raise ValueError("media build capture needs its completed state and canonical producer source")
        build = capture_build(pins, args.state, args.project)
        (args.state / "receipts/media-build.json").write_bytes(canonical(build) + b"\n")
        (args.state / "receipts/source-wheels.sha256").write_text("".join(
            row["sha256"] + "  " + name + "\n" for name, row in build["source_wheels"].items()))
        return
    if args.command == "capture-install":
        if args.runtime is None or args.base_runtime is None or args.wheels is None:
            raise ValueError("media install capture needs both dependency closures and actual source wheels")
        result = capture_install(pins, read_json(args.runtime)[0], read_json(args.base_runtime)[0], args.metadata, args.wheels)
        (args.metadata / "media-install.json").write_bytes(canonical(result) + b"\n")
        print(canonical({"artifact_count": len(result["artifacts"]), "release_qualified": False}).decode())
        return
    if args.command == "audit":
        print(canonical(audit_install(read_json(args.metadata / "media-install.json")[0], pins)).decode())
        return
    if args.command in ("verify-source", "record-source"):
        row = next(row for row in pins["artifacts"] if row["name"] == args.source_name and row["kind"] == "source-archive")
        if args.sources is None or args.downloads is None or args.receipt is None:
            raise ValueError("source cache verification needs its owned inputs and completed-stage receipt")
        if args.command == "verify-source":
            verify_source(row, args.downloads, args.sources, args.receipt)
        else:
            args.receipt.write_bytes(canonical(source_inventory(row, args.downloads, args.sources)) + b"\n")
        return
    if args.downloads is None:
        raise ValueError("media acquisition needs its declared download directory")
    for row in pins["artifacts"]:
        if args.command == "fetch":
            print(canonical({"artifact": row["artifact"], "status": fetch_artifact(row, args.downloads)}).decode())
        elif row.get("kind") == "source-archive":
            if args.sources is None:
                raise ValueError("source extraction needs its owned source directory")
            receipt = args.sources.parent / "receipts" / ("source-" + row["name"] + ".json")
            print(extract_source(row, args.downloads, args.sources, receipt))


if __name__ == "__main__":
    main()
