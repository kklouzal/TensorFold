"""Stable SM70 inputs retain exact bytes and immutable publication ownership."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading

import pytest

from tensorfold.cuda.sm70.source_cache import stage_sources


def fixture(tmp_path):
    root = tmp_path / "cuda"
    (root / "sm70").mkdir(parents=True)
    (root / "kernels").mkdir()
    source, header = root / "sm70" / "operator.cu", root / "kernels" / "tensor_contracts.cuh"
    source.write_text('#include "../kernels/tensor_contracts.cuh"\n' + "// source\n" * 10000)
    header.write_text("// pinned contract\n")
    return root, source, header, tmp_path / "cache"


def stage(root, source, header, cache):
    return stage_sources([source], headers=[header], source_root=root, cache_root=cache)


def test_identical_reinstalled_bytes_reuse_stable_paths_and_mtimes(tmp_path):
    root, source, header, cache = fixture(tmp_path)
    (result,) = stage(root, source, header, cache)
    staged = Path(result)
    before = staged.stat().st_mtime_ns
    assert staged.read_bytes() == source.read_bytes()
    assert (staged.parent.parent / "kernels" / header.name).read_bytes() == header.read_bytes()
    os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 10**12))
    assert stage(root, source, header, cache) == [result]
    assert staged.stat().st_mtime_ns == before
    installed = source.read_bytes()
    header.write_text("// revised contract\n")
    assert stage(root, source, header, cache) != [result]
    assert source.read_bytes() == installed and staged.read_bytes() == installed


def test_concurrent_publishers_reuse_only_a_complete_verified_namespace(tmp_path):
    root, source, header, cache = fixture(tmp_path)
    barrier = threading.Barrier(8)

    def publish(_):
        barrier.wait()
        return stage(root, source, header, cache)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(publish, range(8)))
    assert all(paths == results[0] for paths in results)
    assert len(list((cache / "_sm70_sources").iterdir())) == 1
    assert Path(results[0][0]).read_bytes() == source.read_bytes()


@pytest.mark.parametrize("damage", ["bytes", "missing", "extra", "symlink", "manifest"])
def test_existing_immutable_namespace_is_never_repaired_or_silently_reused(tmp_path, damage):
    root, source, header, cache = fixture(tmp_path)
    (result,) = stage(root, source, header, cache)
    staged = Path(result)
    if damage == "bytes":
        staged.write_text("changed")
    elif damage == "missing":
        staged.unlink()
    elif damage == "extra":
        (staged.parent / "undeclared.h").write_text("// injected\n")
    elif damage == "symlink":
        staged.unlink()
        staged.symlink_to(source)
    else:
        (staged.parent.parent / "manifest.json").write_text("{}")
    with pytest.raises((ValueError, FileNotFoundError)):
        stage(root, source, header, cache)
    assert source.read_text().startswith("#include")


def test_declared_source_path_cannot_escape_the_cuda_root(tmp_path):
    root, source, header, cache = fixture(tmp_path)
    outside = tmp_path / "outside.cu"
    outside.write_text("// outside\n")
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        stage(root, source, header, cache)
    assert not cache.exists()


def test_cache_namespace_symlink_is_refused(tmp_path):
    root, source, header, cache = fixture(tmp_path)
    cache.mkdir()
    target = tmp_path / "other"
    target.mkdir()
    (cache / "_sm70_sources").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="owned directory"):
        stage(root, source, header, cache)
    assert list(target.iterdir()) == []


def test_failed_atomic_publication_retires_its_private_staging(tmp_path, monkeypatch):
    root, source, header, cache = fixture(tmp_path)

    def fail(*_args):
        raise PermissionError("publication refused")

    monkeypatch.setattr("tensorfold.cuda.sm70.source_cache.os.rename", fail)
    with pytest.raises(PermissionError, match="publication refused"):
        stage(root, source, header, cache)
    assert list((cache / "_sm70_sources").iterdir()) == []
