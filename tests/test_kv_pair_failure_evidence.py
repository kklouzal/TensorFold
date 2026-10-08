"""Stdlib failure evidence identity/resource/error boundaries; no Torch."""
from __future__ import annotations

import hashlib
import io
import os
import stat
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests import kv_pair_failure_evidence as M


class Tensor:
    def __init__(self, count=2, size=2, storage=None):
        self.count, self.size, self.storage = count, size, count * size if storage is None else storage
        self.device = SimpleNamespace(type="cpu")
    def numel(self):
        return self.count
    def element_size(self):
        return self.size
    def is_contiguous(self):
        return True
    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: self.storage)


def save(root, *, case="tests/cuda/test.py::case[param] (call)", tensor=None, writer=None, copier=None):
    return M.save_failure_snapshot(root, "chunk-po", "kv-pair-v1-" + "f" * 64, "K-int8__V-int8", case,
                                   {"actual": tensor or Tensor()}, writer or (lambda payload, out: out.write(b"data")),
                                   lambda obj: isinstance(obj, Tensor), copier or (lambda obj: Tensor(obj.numel(), obj.element_size())))


class FailureSnapshotBoundaries(unittest.TestCase):
    def test_distinct_case_full_sha_identity_retained_and_existing_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = []
            def writer(payload, output):
                payloads.append(payload)
                output.write(b"data")
            for case in ("tests/cuda/test.py::case[17] (call)", "tests/cuda/test.py::case[513] (call)"):
                path = save(root, case=case, writer=writer)
                self.assertIn(hashlib.sha256(case.encode()).hexdigest(), path.name)
                self.assertEqual(payloads[-1]["pytest_case_identity"], case)
                self.assertEqual(payloads[-1]["logical_tensor_bytes"], 4)
            self.assertEqual(len(list(root.iterdir())), 2)
            copied = Mock()
            with self.assertRaises(FileExistsError):
                save(root, case=payloads[0]["pytest_case_identity"], copier=copied)
            copied.assert_not_called()
            self.assertTrue(all(p.read_bytes() == b"data" for p in root.iterdir()))

    def test_unsafe_labels_and_phase_identity_are_data_not_paths(self):
        case = "tests/cuda/test.py::a[`$(literal)/../x] (call)"
        path = M.filename("../../label", "../../pair", case)
        self.assertNotIn("/", path)
        self.assertIn(hashlib.sha256(case.encode()).hexdigest(), path)
        self.assertLess(len(path), 256)
        for invalid in ("", "x" * 4097, "node\0bad"):
            with self.assertRaises(ValueError):
                M.filename("label", "pair", invalid)

    def test_typed_logical_bounds_fail_before_copy_or_file_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not-created"
            copy = Mock()
            for tensor in (Tensor(count=True), Tensor(size=False), Tensor(count=-1), Tensor(count=(64 << 20) + 1, size=1)):
                with self.subTest(count=tensor.count, size=tensor.size), self.assertRaises((ValueError, MemoryError)):
                    save(root, tensor=tensor, copier=copy)
            copy.assert_not_called()
            self.assertFalse(root.exists())

    def test_file_and_batch_exhaustion_and_unexpected_nodes_fail_before_copy(self):
        for policy in ("files", "bytes", "symlink", "directory"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "prior.pt").write_bytes(b"data")
                copy = Mock()
                if policy == "symlink":
                    (root / "alias.pt").symlink_to(root / "prior.pt")
                elif policy == "directory":
                    (root / "unexpected").mkdir()
                with patch.object(M, "MAX_FILES", 1 if policy == "files" else 128), \
                        patch.object(M, "MAX_BATCH_BYTES", 4 if policy == "bytes" else 1 << 30), \
                        self.assertRaises((ValueError, MemoryError)):
                    save(root, copier=copy)
                copy.assert_not_called()
                self.assertEqual((root / "prior.pt").read_bytes(), b"data")

    def test_serialization_overflow_preserves_prior_and_cleans_owned_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prior.pt").write_bytes(b"prior")
            with patch.object(M, "MAX_FILE_BYTES", 8), self.assertRaises(MemoryError):
                save(root, writer=lambda payload, out: out.write(b"a" * 9))
            self.assertEqual([p.name for p in root.iterdir()], ["prior.pt"])
            self.assertEqual((root / "prior.pt").read_bytes(), b"prior")

    def test_unexpected_backing_storage_and_copy_failure_never_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "storage"):
                save(root, copier=lambda obj: Tensor(storage=128))
            primary = OSError("CPU copy fixture failure")
            with self.assertRaises(OSError) as caught:
                save(root, copier=Mock(side_effect=primary))
            self.assertIs(caught.exception, primary)
            self.assertEqual(list(root.iterdir()), [])

    def test_serialization_error_survives_secondary_owned_cleanup_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            primary = OSError("fixture save failure")
            with patch.object(M.os, "unlink", side_effect=OSError("fixture cleanup failure")), self.assertRaises(OSError) as caught:
                save(Path(directory), writer=Mock(side_effect=primary))
            self.assertIs(caught.exception, primary)
            self.assertTrue(any("cleanup failure" in n for n in primary.__notes__))

    def test_bounded_writer_exact_limit_short_write_and_no_expansion(self):
        buffer = io.BytesIO()
        writer = M.BoundedWriter(buffer, 4)
        self.assertEqual(writer.write(b"1234"), 4)
        self.assertEqual(writer.tell(), 4)
        with self.assertRaises(MemoryError):
            writer.write(b"5")
        self.assertEqual(buffer.getvalue(), b"1234")
        short = M.BoundedWriter(SimpleNamespace(write=lambda data: 1), 4)
        with self.assertRaises(OSError):
            short.write(b"12")


if __name__ == "__main__":
    unittest.main()


def test_real_torch_serialization_compacts_views_and_preserves_exact_values(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    backing = torch.arange(1024, dtype=torch.float32)
    view = backing[7:11]
    path = M.save_failure_snapshot(
        tmp_path, "compact-smoke", "kv-pair-v1-" + "f" * 64, "K-int8__V-int8", "actualTorch (call)",
        {"actual": view}, torch.save, torch.is_tensor,
        lambda tensor: tensor.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format),
    )
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    tensor = loaded["tensors"]["actual"]
    assert torch.equal(tensor, view)
    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size() == 16
    assert loaded["pytest_case_identity"] == "actualTorch (call)"
    assert loaded["logical_tensor_bytes"] == 16
    assert list(tmp_path.iterdir()) == [path]


class FailurePermissions(unittest.TestCase):
    def test_host_readability_is_preserved_under_restrictive_umask(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.umask(0o077)
            try:
                path = save(Path(directory))
            finally:
                os.umask(previous)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

    def test_permission_failure_closes_descriptor_and_cleans_only_owned_temporary(self):
        primary = OSError("chmod failed")
        copier = Mock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(M.os, "fchmod", side_effect=primary):
                with self.assertRaises(OSError) as failed:
                    save(root, copier=copier)
            self.assertIs(failed.exception, primary)
            copier.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])
