"""Exact persisted bytes, source provenance, resource limits and failure paths."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import struct

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.host_experts import load_host_experts  # noqa: E402
from tensorfold.cuda.packed_experts import PackedExpertCache, SourceIdentity  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.reader import _Reader  # noqa: E402
from test_host_experts import checkpoint_tensors, reference_pack, write_checkpoint  # noqa: E402


def prepared(tmp_path, *, base="model.layers.0.mlp"):
    model = tmp_path / "model"
    model.mkdir()
    tensors = checkpoint_tensors(base)
    write_checkpoint(model, tensors)
    return model, tensors, PackedExpertCache(tmp_path / "cache", 1 << 20)


def load(model, cache, **kwargs):
    reader = _Reader(model, "cpu")
    try:
        return load_host_experts(reader, kwargs.pop("base", "model.layers.0.mlp"), packed_cache=cache, **kwargs)
    finally:
        reader.close()


@pytest.mark.parametrize("base", ["model.layers.0.mlp", "language_model.mtp.layers.0.mlp"])
@pytest.mark.parametrize("ids,shared", [(None, True), ([2, 0], False), (range(1, 3), True)])
def test_cache_hit_equals_independent_oracle_without_pack_or_raw_payload_reads(tmp_path, monkeypatch, base, ids, shared):
    model, tensors, cache = prepared(tmp_path, base=base)
    fresh = load(model, cache, base=base, owned_ids=ids, include_shared=shared)
    order = list(range(3) if ids is None else ids) + ([3] if shared else [])
    expected = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        source = [torch.cat((tensors[f"{base}.switch_mlp.{projection}.{field}"],
                             tensors[f"{base}.shared_expert.{projection}.{field}"][None]))[order].contiguous()
                  for field in ("weight", "scales", "biases")]
        expected.append(reference_pack(*source, 32))
    def forbidden(*args, **kwargs):
        pytest.fail("a verified hit must not repack or consume the payload reader")
    monkeypatch.setattr("tensorfold.cuda.host_experts.pack_cpu", forbidden)
    monkeypatch.setattr(_Reader, "get_rows", forbidden)
    cached = load(model, cache, base=base, owned_ids=ids, include_shared=shared, chunk_experts=1)
    assert torch.equal(cached.up, torch.stack(expected[:2], dim=3))
    assert torch.equal(cached.down, expected[2].unsqueeze(3))
    assert torch.equal(cached.up, fresh.up) and cached.up.data_ptr() != fresh.up.data_ptr()
    assert cached.logical_ids == tuple(order)


@pytest.mark.parametrize("field", [f"{kind}.{projection}.{component}"
                                   for kind in ("switch_mlp", "shared_expert")
                                   for projection in ("gate_proj", "up_proj", "down_proj")
                                   for component in ("weight", "scales", "biases")])
def test_every_contributing_source_change_rekeys_even_when_size_mtime_preserved(tmp_path, field):
    model, _, cache = prepared(tmp_path)
    load(model, cache)
    reader = _Reader(model, "cpu")
    try:
        info = reader.info("model.layers.0.mlp." + field)
    finally:
        reader.close()
    path = info["path"]
    before = path.stat()
    with path.open("r+b") as stream:
        stream.seek(info["byte_offset"])
        value = stream.read(1)
        stream.seek(info["byte_offset"])
        stream.write(bytes([value[0] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    cached = load(model, cache)
    fresh = load(model, None)
    assert torch.equal(cached.up, fresh.up) and torch.equal(cached.down, fresh.down)
    assert len(list(cache.root.glob("*.tfc"))) == 2


def test_unowned_source_change_does_not_change_rank_local_identity(tmp_path):
    model, _, cache = prepared(tmp_path)
    first = load(model, cache, owned_ids=[1], include_shared=False)
    reader = _Reader(model, "cpu")
    try:
        info = reader.info("model.layers.0.mlp.switch_mlp.up_proj.weight")
    finally:
        reader.close()
    with info["path"].open("r+b") as stream:
        stream.seek(info["byte_offset"])
        stream.write(b"\0")
    second = load(model, cache, owned_ids=[1], include_shared=False)
    assert torch.equal(first.up, second.up)
    assert len(list(cache.root.glob("*.tfc"))) == 1


@pytest.mark.parametrize("damage", ["magic", "huge-header", "json", "shape", "checksum", "truncated", "extra"])
def test_damage_is_reported_and_replaced_with_canonical_bytes(tmp_path, capsys, damage):
    model, _, cache = prepared(tmp_path)
    expected = load(model, cache)
    capsys.readouterr()
    path, = cache.root.glob("*.tfc")
    raw = bytearray(path.read_bytes())
    if damage == "magic":
        raw[0] ^= 1
    elif damage == "huge-header":
        raw[16:24] = struct.pack("<Q", 2**63)
    elif damage == "json":
        raw[24] = 0
    elif damage == "shape":
        size = struct.unpack("<Q", raw[16:24])[0]
        header = json.loads(raw[24:24 + size])
        header["arrays"][0]["shape"][0] = 2**63
        changed = json.dumps(header).encode()
        raw = raw[:16] + struct.pack("<Q", len(changed)) + changed + raw[24 + size:]
    elif damage == "checksum":
        raw[-1] ^= 1
    elif damage == "truncated":
        raw = raw[:-1]
    else:
        raw += b"x"
    path.write_bytes(raw)
    got = load(model, cache)
    assert "rebuild:" in capsys.readouterr().out
    assert torch.equal(got.up, expected.up) and torch.equal(got.down, expected.down)
    assert "rebuild:" not in (load(model, cache), capsys.readouterr().out)[1]


def test_partial_reads_and_writes_make_complete_progress(tmp_path, monkeypatch):
    model, _, cache = prepared(tmp_path)
    actual_write, actual_read = os.write, os.readv
    monkeypatch.setattr(os, "write", lambda fd, buf: actual_write(fd, buf[:max(1, len(buf) // 3)]))
    first = load(model, cache)
    monkeypatch.setattr(os, "readv", lambda fd, bufs: actual_read(fd, [bufs[0][:max(1, len(bufs[0]) // 3)]]))
    second = load(model, cache)
    assert torch.equal(first.up, second.up) and torch.equal(first.down, second.down)


@pytest.mark.parametrize("failure", ["write", "fsync", "replace"])
def test_publishing_io_failure_propagates_and_removes_incomplete_file(tmp_path, monkeypatch, failure):
    model, _, cache = prepared(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("injected " + failure)
    monkeypatch.setattr(os, failure, fail)
    with pytest.raises(OSError, match="injected"):
        load(model, cache)
    assert not list(cache.root.glob(".tmp-*"))
    assert not list(cache.root.glob("*.tfc"))


def test_source_mutation_during_canonical_pack_aborts_publication(tmp_path, monkeypatch):
    model, _, cache = prepared(tmp_path)
    actual = _Reader.get_rows
    def changed(self, name, lo, hi, device="cpu"):
        got = actual(self, name, lo, hi, device)
        if name.endswith("gate_proj.weight"):
            path = self.info(name)["path"]
            os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1))
        return got
    monkeypatch.setattr(_Reader, "get_rows", changed)
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        load(model, cache)
    assert not list(cache.root.glob("*.tfc"))


def test_private_root_and_artifact_symlinks_refuse_access(tmp_path):
    model, _, cache = prepared(tmp_path)
    cache.root.chmod(0o777)
    with pytest.raises(PermissionError):
        load(model, cache)
    cache.root.chmod(0o700)
    load(model, cache)
    artifact, = cache.root.glob("*.tfc")
    artifact.unlink()
    artifact.symlink_to(model / "model.safetensors")
    with pytest.raises(OSError):
        load(model, cache)


def test_concurrent_publishers_and_readers_only_observe_complete_artifacts(tmp_path):
    model, _, cache = prepared(tmp_path)
    with ThreadPoolExecutor(4) as workers:
        results = list(workers.map(lambda _: load(model, cache), range(8)))
    assert all(torch.equal(results[0].up, result.up) and torch.equal(results[0].down, result.down)
               for result in results)
    assert len(list(cache.root.glob("*.tfc"))) == 1
    assert not list(cache.root.glob(".tmp-*"))


def test_budget_eviction_and_stale_temporaries_are_bounded(tmp_path):
    model, _, cache = prepared(tmp_path)
    load(model, cache)
    artifact, = cache.root.glob("*.tfc")
    size = artifact.stat().st_size
    (cache.root / ".tmp-dead-publisher").write_bytes(b"not published")
    cache.max_bytes = size + 4096
    load(model, cache, owned_ids=[0], include_shared=False)
    assert not list(cache.root.glob(".tmp-*"))
    assert sum(path.stat().st_size for path in cache.root.glob("*.tfc")) <= cache.max_bytes
    cache.max_bytes = 1
    load(model, cache, owned_ids=[2], include_shared=False)
    assert not list(cache.root.glob(".tmp-*"))


def test_environment_requires_explicit_positive_disk_budget(monkeypatch, tmp_path):
    monkeypatch.delenv("TENSORFOLD_PACKED_EXPERT_CACHE", raising=False)
    monkeypatch.delenv("TENSORFOLD_PACKED_EXPERT_CACHE_BYTES", raising=False)
    assert PackedExpertCache.from_environment() is None
    monkeypatch.setenv("TENSORFOLD_PACKED_EXPERT_CACHE", str(tmp_path / "cache"))
    for value in ("", "0", "-1", "1.5", "²"):
        monkeypatch.setenv("TENSORFOLD_PACKED_EXPERT_CACHE_BYTES", value)
        with pytest.raises(ValueError, match="positive byte count"):
            PackedExpertCache.from_environment()
    monkeypatch.setenv("TENSORFOLD_PACKED_EXPERT_CACHE_BYTES", " 16384 ")
    assert PackedExpertCache.from_environment().max_bytes == 16384


def test_budget_shrink_is_enforced_on_hits_without_republication(tmp_path):
    model, _, cache = prepared(tmp_path)
    load(model, cache)
    assert list(cache.root.glob("*.tfc"))
    cache.max_bytes = 1
    load(model, cache)
    assert not list(cache.root.glob("*.tfc"))


def test_oversized_existing_inventory_is_explicitly_refused(tmp_path):
    cache = PackedExpertCache(tmp_path / "cache", 1 << 20)
    for i in range(4097):
        (cache.root / (f"{i:064x}.tfc")).write_bytes(b"")
    with cache._locked(True) as root:
        with pytest.raises(OSError, match="inventory limit"):
            cache._prune(root)


def test_cache_read_io_errors_do_not_become_repack_misses(tmp_path, monkeypatch):
    model, _, cache = prepared(tmp_path)
    load(model, cache)
    def fail(*args, **kwargs):
        raise OSError("injected read failure")
    monkeypatch.setattr(os, "readv", fail)
    with pytest.raises(OSError, match="injected read failure"):
        load(model, cache)


def test_atomic_cache_replacement_during_read_cannot_publish_mixed_authority(tmp_path, monkeypatch):
    model, _, cache = prepared(tmp_path)
    expected = load(model, cache)
    artifact, = cache.root.glob("*.tfc")
    replacement = cache.root / "replacement"
    replacement.write_bytes(artifact.read_bytes())
    actual = os.readv
    def replaced(fd, buffers):
        if replacement.exists():
            os.replace(replacement, artifact)
        return actual(fd, buffers)
    monkeypatch.setattr(os, "readv", replaced)
    result = load(model, cache)
    assert torch.equal(result.up, expected.up) and torch.equal(result.down, expected.down)


def test_source_cleanup_preserves_primary_and_all_close_failures(tmp_path, monkeypatch):
    paths = [tmp_path / f"source-{i}" for i in range(2)]
    for path in paths:
        path.write_bytes(b"\0" * 4)
    ranges = [(str(i), dict(path=path, shape=(1,), dtype="I32", byte_offset=0), 0, 1)
              for i, path in enumerate(paths)]
    identity = SourceIdentity(ranges, {})
    with identity:
        fds = set(fd for fd, _ in identity.files.values())
        actual = os.close
        def fail_after_close(fd):
            actual(fd)
            if fd in fds:
                raise OSError(f"close {fd}")
        monkeypatch.setattr(os, "close", fail_after_close)
        try:
            identity.__exit__(ValueError, ValueError("primary source failure"), None)
        except ValueError as error:
            assert "primary source failure" in str(error)
            assert len(error.__cause__.exceptions) == 2
        else:
            pytest.fail("primary failure must be retained")


def test_publication_attempts_unlink_after_close_failure_and_keeps_both_errors(tmp_path, monkeypatch):
    model, _, cache = prepared(tmp_path)
    actual_close = os.close
    target = set()
    actual_open = os.open
    def opened(path, flags, *args, **kwargs):
        fd = actual_open(path, flags, *args, **kwargs)
        if str(path).startswith(".tmp-"):
            target.add(fd)
        return fd
    def close(fd):
        actual_close(fd)
        if fd in target:
            target.remove(fd)
            raise OSError("temporary close failure")
    def fsync(fd):
        raise OSError("primary fsync failure")
    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "close", close)
    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError, match="primary fsync failure") as caught:
        load(model, cache)
    assert "temporary close failure" in str(caught.value.__cause__)
    assert not list(cache.root.glob(".tmp-*"))
