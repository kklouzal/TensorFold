"""Sealed ELF preload order and exact component-runtime provenance boundaries."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "deploy/gb10"


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(DEPLOY))
    spec = importlib.util.spec_from_file_location("test_foundation_runtime_module", DEPLOY / "foundation_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Synthetic activation controls model the pre-Torch startup boundary even
    # when unrelated tests already imported Torch in the shared pytest process.
    # The real ELF control below exercises actual sys.modules in a fresh child.
    monkeypatch.setattr(module, "sys", SimpleNamespace(modules={}))
    return module


@pytest.fixture
def prepared(runtime, tmp_path, monkeypatch):
    foundation = tmp_path / "foundation"
    site = foundation / "lib/python3.12/site-packages"
    (site / "torchvision").mkdir(parents=True)
    seals, dynamics, native, compiler = {}, {}, set(), set()

    def seal(path):
        real, info = path.resolve(strict=True), path.stat()
        seals[str(path)] = {"realpath": str(real), "size": info.st_size, "mtime_ns": info.st_mtime_ns,
                            "mode": stat.S_IMODE(info.st_mode), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def add(path, soname=None, needed=(), *, owned=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic ELF control: " + path.name.encode())
        seal(path)
        dynamics[str(path.resolve())] = {"soname": path.name if soname is None else soname, "needed": list(needed)}
        if owned:
            native.add(path)
        return path

    root = tmp_path / "native"
    for name in ("libstdc++.so.6", "libgcc_s.so.1", "libcudart.so.13", "libnvrtc.so.13",
                 "libnvrtc-builtins.so.13.4", "libnvJitLink.so.13", "libnccl.so.2"):
        add(root / name, owned=True)
    compiler.update(root / name for name in ("libstdc++.so.6", "libgcc_s.so.1"))
    dynamics[str(root / "libstdc++.so.6")]["needed"] = ["libgcc_s.so.1"]
    dynamics[str(root / "libnvrtc.so.13")]["needed"] = ["libstdc++.so.6"]
    cupti = add(root / "cupti/libcupti.so.13", owned=True)
    add(cupti.parent / "libnvperf_target.so", owned=True)
    add(cupti.parent / "libnvperf_host.so", needed=["libnvperf_target.so"], owned=True)
    add(root / "nsight/libnvperf_host.so", owned=True)
    add(root / "stubs/libcudart.so.13", owned=True)
    add(root / "compat/libcublas.so.13", owned=True)
    add(root / "libcuda.so.1", owned=True)
    add(site / "torch/_C.so", needed=["libcudart.so.13", "libnvrtc.so.13", "libnccl.so.2"])
    add(site / "torch/lib/libtorch_cuda.so", needed=["libnvJitLink.so.13"])
    members = [("torchvision.libs/libcudart.a6a4ec3c.so.13", "libcudart.so.13"),
               ("torchvision.libs/libnvjpeg.be6456a4.so.13", "libnvjpeg.so.13")]
    for member, soname in members:
        add(site / member, soname=soname)
    record = site / "torchvision-0.30.dist-info/RECORD"

    def rows():
        values = []
        for member, _ in members:
            data = (site / member).read_bytes()
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            values.append([member, "sha256=" + encoded, str(len(data))])
        return [*values, [str(record.relative_to(site)), "", ""]]

    def write_record(values):
        record.parent.mkdir(parents=True, exist_ok=True)
        stream = io.StringIO()
        csv.writer(stream, lineterminator="\n").writerows(values)
        record.write_text(stream.getvalue())
        seal(record)

    write_record(rows())
    dist = SimpleNamespace(metadata={"Name": "torchvision"}, version="0.30.0.dev20261007+cu134",
                           files=[Path(member) for member, _ in members] + [record.relative_to(site)],
                           locate_file=lambda member: site / member)
    monkeypatch.setattr(runtime.metadata, "distributions", lambda **kwargs: [dist])
    monkeypatch.setattr(runtime, "elf", lambda path: dict(dynamics[str(path.resolve())]))
    manifest = {"python": "3.12", "roles": {"foundation": {"torchvision": {
        "version": dist.version, "sha256": "a" * 64,
        "url": "https://download.pytorch.org/torchvision-0.30.0.dev20261007%2Bcu134-cp312-cp312-linux_x86_64.whl"}}}}
    value = SimpleNamespace(foundation=foundation, site=site, native=native, compiler=compiler, seals=seals, dynamics=dynamics,
                            root=root, cupti=cupti, nccl=root / "libnccl.so.2", members=members,
                            record=record, dist=dist, manifest=manifest, add=add, seal=seal,
                            rows=rows, write_record=write_record)
    value.capture = lambda: runtime.capture_runtime(foundation=foundation, manifest=manifest, native_files=native,
                                                    compiler_files=compiler, sealed_files=seals, nccl_library=value.nccl,
                                                    cupti_library=value.cupti)
    return value


def map_line(path, *, inode=None, device=None, deleted=False):
    info = path.stat()
    major, minor = (os.major(info.st_dev), os.minor(info.st_dev)) if device is None else device
    return (f"1000-2000 r-xp 00000000 {major:x}:{minor:x} {info.st_ino if inode is None else inode} "
            f"{path}{' (deleted)' if deleted else ''}\n")


def test_capture_uses_sealed_provider_set_and_actual_dependency_order(prepared):
    contract = prepared.capture()
    index, order = contract["index"], contract["torch"]["order"]
    assert contract["cupti_dir"] == str(prepared.cupti.parent)
    assert index["libnvperf_host.so"]["realpath"] == str(prepared.cupti.parent / "libnvperf_host.so")
    assert all("stubs" not in entry["path"] and "compat" not in entry["path"] for entry in index.values())
    assert "libcuda.so.1" not in index
    assert order.index("libgcc_s.so.1") < order.index("libstdc++.so.6") < order.index("libnvrtc.so.13")
    assert order.index("libnvrtc-builtins.so.13.4") < order.index("libnvrtc.so.13")
    assert {"libnvperf_host.so", "libnvperf_target.so"} <= index.keys()
    assert "libnvperf_host.so" not in order and "libnvperf_target.so" not in order
    assert {"libcupti.so.13", "libnvJitLink.so.13", "libnccl.so.2"} <= set(order)
    assert [entry["soname"] for entry in contract["vision"]["providers"]] == ["libcudart.so.13", "libnvjpeg.so.13"]
    persisted = json.dumps(contract)
    assert '"inode"' not in persisted and '"device"' not in persisted


@pytest.mark.parametrize("failure", ["missing", "ambiguous", "builtins_missing", "builtins_ambiguous", "cycle"])
def test_capture_refuses_incomplete_or_ambiguous_native_dependencies(prepared, failure):
    if failure == "missing":
        prepared.native.remove(prepared.root / "libgcc_s.so.1")
        prepared.compiler.remove(prepared.root / "libgcc_s.so.1")
    elif failure == "ambiguous":
        prepared.compiler.add(prepared.add(prepared.root / "other/libstdc++.so.6", owned=True))
    elif failure == "builtins_missing":
        prepared.native.remove(prepared.root / "libnvrtc-builtins.so.13.4")
    elif failure == "builtins_ambiguous":
        prepared.add(prepared.root / "libnvrtc-builtins.so.13.5", owned=True)
    else:
        prepared.dynamics[str(prepared.root / "libgcc_s.so.1")]["needed"] = ["libstdc++.so.6"]
    with pytest.raises(RuntimeError):
        prepared.capture()


def test_cxx_provider_comes_from_compiler_packages_not_sdk_private_bundles(prepared, runtime):
    private = prepared.add(prepared.root / "profiler-private/libstdc++.so.6", owned=True)
    contract = prepared.capture()
    selected = prepared.root / "libstdc++.so.6"
    assert contract["index"]["libstdc++.so.6"]["realpath"] == str(selected)
    with pytest.raises(RuntimeError, match="non-foundation"):
        runtime.mapped_runtime(contract, map_line(private))
    prepared.compiler.remove(selected)
    with pytest.raises(RuntimeError, match="missing base C\\+\\+ runtime provider"):
        prepared.capture()


def test_capture_does_not_discover_unsealed_native_root_files(prepared):
    extra = prepared.root / "libcublas.so.13"
    prepared.add(extra)
    prepared.dynamics[str(prepared.site / "torch/_C.so")]["needed"].append("libcublas.so.13")
    with pytest.raises(RuntimeError, match="no sealed provider"):
        prepared.capture()


def test_selected_cupti_scopes_duplicate_providers_and_needs_input_ownership(prepared):
    other = prepared.add(prepared.root / "nsight/libcupti.so.13", owned=True)
    contract = prepared.capture()
    assert contract["index"]["libcupti.so.13"]["realpath"] == str(prepared.cupti)
    assert all(entry["realpath"] != str(other) for entry in contract["index"].values())
    prepared.cupti = prepared.add(prepared.root / "outside-input/libcupti.so.13")
    with pytest.raises(RuntimeError, match="CUPTI library is outside"):
        prepared.capture()


def test_native_aliases_have_one_physical_provider_and_canonical_map_path(prepared, runtime):
    original = prepared.root / "libstdc++.so.6"
    alias = prepared.root / "libstdc++.so.6.0"
    alias.symlink_to(original.name)
    prepared.seal(alias)
    prepared.native.add(alias)
    contract = prepared.capture()
    assert runtime.mapped_runtime(contract, map_line(alias))[0]["path"] == str(original)


@pytest.mark.parametrize("version", [True, 2])
def test_hot_checks_reject_unsupported_contract_versions(prepared, runtime, version):
    contract = prepared.capture()
    contract["schema_version"] = version
    with pytest.raises(RuntimeError, match="unsupported"):
        runtime.activate_runtime(contract)
    with pytest.raises(RuntimeError, match="unsupported"):
        runtime.mapped_runtime(contract, "")


def test_selected_nccl_replaces_only_the_native_nccl_entry(prepared, runtime):
    old = prepared.nccl
    prepared.nccl = prepared.add(prepared.site / "nvidia/nccl/lib/libnccl.so.2")
    contract = prepared.capture()
    assert contract["index"]["libnccl.so.2"]["realpath"] == str(prepared.nccl)
    assert runtime.mapped_runtime(contract, map_line(prepared.nccl))[0]["path"] == str(prepared.nccl)
    with pytest.raises(RuntimeError, match="non-foundation"):
        runtime.mapped_runtime(contract, map_line(old))


@pytest.mark.parametrize("failure", ["record_hash", "member_hash", "member_size", "duplicate", "missing",
                                   "soname", "other_family", "symlink", "version", "origin"])
def test_vision_exceptions_require_exact_official_record_members(prepared, failure):
    rows = prepared.rows()
    if failure == "record_hash":
        prepared.seals[str(prepared.record)]["sha256"] = "0" * 64
    elif failure == "member_hash":
        rows[0][1] = "sha256=" + "a" * 43
        prepared.write_record(rows)
    elif failure == "member_size":
        rows[0][2] = "1"
        prepared.write_record(rows)
    elif failure == "duplicate":
        prepared.write_record([*rows, rows[0]])
    elif failure == "missing":
        prepared.write_record(rows[1:])
    elif failure == "soname":
        prepared.dynamics[str(prepared.site / prepared.members[0][0])]["soname"] = "libcudart.so.12"
    elif failure == "other_family":
        prepared.write_record([*rows, ["torchvision.libs/libcublas.abcdef12.so.13", "", ""]])
    elif failure == "symlink":
        path = prepared.site / prepared.members[0][0]
        target = path.with_name("real.so")
        path.rename(target)
        path.symlink_to(target.name)
    elif failure == "version":
        prepared.dist.version = "other-version"
    else:
        prepared.manifest["roles"]["foundation"]["torchvision"]["url"] = "https://unapproved.invalid/vision.whl"
    with pytest.raises((RuntimeError, ValueError)):
        prepared.capture()


def test_activate_uses_stat_only_and_returns_strong_owners(prepared, runtime, monkeypatch):
    contract = prepared.capture()
    calls = []

    class Handle:
        def __init__(self, path, *, mode):
            calls.append((path, mode))
            self.path = path

    def forbidden(*args, **kwargs):
        raise AssertionError("startup attempted content hashing or ELF inspection")

    monkeypatch.setattr(runtime, "elf", forbidden)
    monkeypatch.setattr(runtime.hashlib, "file_digest", forbidden)
    monkeypatch.setattr(runtime.ctypes, "CDLL", Handle)
    owners = runtime.activate_runtime(contract)
    assert isinstance(owners, tuple) and len(owners) == len(contract["torch"]["order"])
    assert calls == [(contract["index"][name]["realpath"], os.RTLD_GLOBAL | os.RTLD_NOW)
                     for name in contract["torch"]["order"]]
    mapped = runtime.mapped_runtime(contract, map_line(prepared.nccl))
    assert mapped[0]["mapped_object_verified"] is True


def test_activate_rejects_preimported_torch_and_preserves_cdll_error(prepared, runtime, monkeypatch):
    contract = prepared.capture()
    monkeypatch.setitem(runtime.sys.modules, "torch", SimpleNamespace())
    with pytest.raises(RuntimeError, match="already imported"):
        runtime.activate_runtime(contract)
    monkeypatch.delitem(runtime.sys.modules, "torch")
    failure = OSError("raw loader failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(runtime.ctypes, "CDLL", fail)
    with pytest.raises(OSError) as caught:
        runtime.activate_runtime(contract)
    assert caught.value is failure


def test_stat_change_fails_before_loading(prepared, runtime, monkeypatch):
    contract = prepared.capture()
    prepared.nccl.write_bytes(b"changed")
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda *args, **kwargs: pytest.fail("loaded changed providers"))
    with pytest.raises(RuntimeError, match="changed"):
        runtime.activate_runtime(contract)


def test_capture_refuses_a_file_changed_during_elf_inspection(prepared, runtime, monkeypatch):
    inspect = runtime.elf
    target = prepared.root / "libcudart.so.13"

    def change_after_read(path):
        result = inspect(path)
        if path == target:
            target.write_bytes(b"changed during ELF inspection")
        return result

    monkeypatch.setattr(runtime, "elf", change_after_read)
    with pytest.raises(RuntimeError, match="changed"):
        prepared.capture()


def test_inode_changes_are_process_local_and_stale_map_segments_fail(prepared, runtime, monkeypatch):
    contract = prepared.capture()
    path = prepared.root / "libstdc++.so.6"
    old_maps, info, data = map_line(path), path.stat(), path.read_bytes()
    path.rename(path.with_name("retired-provider"))
    path.write_bytes(data)
    path.chmod(stat.S_IMODE(info.st_mode))
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda *args, **kwargs: object())
    # Copying an image can change inode/device without changing the sealed
    # contract. An already-mapped old object must still fail its live comparison.
    assert runtime.activate_runtime(contract)
    with pytest.raises(RuntimeError, match="mapped runtime object differs"):
        runtime.mapped_runtime(contract, old_maps)
    assert runtime.mapped_runtime(contract, map_line(path))[0]["mapped_object_verified"] is True


@pytest.mark.parametrize("failure", ["inode", "device", "segments", "deleted", "unapproved_cxx", "stub"])
def test_mapped_provider_segment_and_physical_ownership_checks(prepared, runtime, failure):
    contract = prepared.capture()
    path = prepared.root / "libstdc++.so.6"
    text = map_line(path)
    if failure == "inode":
        text = map_line(path, inode=path.stat().st_ino + 1)
    elif failure == "device":
        text = map_line(path, device=(0xABC, 0xDEF))
    elif failure == "segments":
        text += map_line(path, inode=path.stat().st_ino + 1)
    elif failure == "deleted":
        text = map_line(path, deleted=True)
    elif failure == "unapproved_cxx":
        duplicate = prepared.add(prepared.root / "unapproved/libstdc++.so.6")
        text += map_line(duplicate)
    else:
        text = map_line(prepared.root / "stubs/libcudart.so.13")
    with pytest.raises(RuntimeError):
        runtime.mapped_runtime(contract, text)


def test_vision_mapping_requires_explicit_phase_and_allows_exact_two_roles(prepared, runtime):
    contract = prepared.capture()
    bundled = prepared.site / prepared.members[0][0]
    text = map_line(prepared.root / "libcudart.so.13") + map_line(bundled)
    with pytest.raises(RuntimeError, match="non-foundation"):
        runtime.mapped_runtime(contract, text)
    rows = runtime.mapped_runtime(contract, text, allow_vision=True)
    assert {row["provider_role"] for row in rows} == {"native_foundation", "official_vision_bundle"}
    assert len(runtime.mapped_runtime(contract, text + text, allow_vision=True)) == 2
    assert runtime.mapped_runtime(contract, map_line(prepared.root / "libcuda.so.1")) == []


@pytest.mark.skipif(sys.platform != "linux", reason="Linux ELF linker and /proc/self/maps integration")
def test_actual_elf_parser_dependency_order_and_maps(runtime, tmp_path):
    compiler = shutil.which("cc")
    if compiler is None or shutil.which("readelf") is None:
        pytest.skip("native cc/readelf unavailable for ELF control")
    source = tmp_path / "provider.c"
    source.write_text("int runtime_control(void) { return 7; }\n")
    library = tmp_path / "libnvJitLink.so.13"
    subprocess.run([compiler, "-shared", "-fPIC", "-Wl,-soname,libnvJitLink.so.13", "-o", str(library),
                    str(source)], check=True, capture_output=True, text=True)
    entry = {"path": str(library), **runtime._stat(library)[0],
             "sha256": hashlib.sha256(library.read_bytes()).hexdigest(), **runtime.elf(library),
             "kind": "cuda", "provider_role": "native_foundation"}
    assert entry["soname"] == "libnvJitLink.so.13"
    consumer_source = tmp_path / "consumer.c"
    consumer_source.write_text("extern int runtime_control(void); int consumer_control(void) { return runtime_control(); }\n")
    consumer = tmp_path / "libcusparse.so.13"
    subprocess.run([compiler, "-shared", "-fPIC", "-Wl,-soname,libcusparse.so.13", "-o", str(consumer),
                    str(consumer_source), "-L" + str(tmp_path), "-l:libnvJitLink.so.13",
                    "-Wl,-rpath," + str(tmp_path)], check=True, capture_output=True, text=True)
    consumer_entry = {"path": str(consumer), **runtime._stat(consumer)[0],
                      "sha256": hashlib.sha256(consumer.read_bytes()).hexdigest(), **runtime.elf(consumer),
                      "kind": "cuda", "provider_role": "native_foundation"}
    assert "libnvJitLink.so.13" in consumer_entry["needed"]
    entries = {entry["soname"]: entry, consumer_entry["soname"]: consumer_entry}
    assert runtime._dependency_order({consumer_entry["soname"]}, entries) == [entry["soname"], consumer_entry["soname"]]
    # Isolate the synthetic CUDA-named DSO from the parent test process, then
    # consume actual kernel maps rather than constructing segment identities.
    script = """import ctypes, json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
import foundation_runtime as runtime
entries = json.loads(sys.argv[1])
assert 'torch' not in sys.modules
order = runtime._dependency_order(set(entries), entries)
contract = {'schema_version': 1, 'index': entries,
            'torch': {'order': order, 'objects': []},
            'vision': {'record': next(iter(entries.values())), 'providers': []}}
owners = runtime.activate_runtime(contract)
assert owners[0].runtime_control() == owners[1].consumer_control() == 7
paths = {entry['realpath'] for entry in entries.values()}
text = '\\n'.join(row for row in Path('/proc/self/maps').read_text().splitlines()
                 if len(row.split(maxsplit=5)) == 6 and row.split(maxsplit=5)[5] in paths)
assert text
rows = runtime.mapped_runtime({'schema_version': 1, 'index': entries}, text=text)
assert len(rows) == 2 and all(row['mapped_object_verified'] for row in rows)
print(json.dumps(rows))
"""
    result = subprocess.run([sys.executable, "-c", script, json.dumps(entries), str(DEPLOY)],
                            check=True, capture_output=True, text=True)
    assert {row["path"] for row in json.loads(result.stdout)} == {str(library), str(consumer)}
    broken = tmp_path / "not-elf.so"
    broken.write_text("not ELF")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        runtime.elf(broken)
    assert caught.value.stderr
