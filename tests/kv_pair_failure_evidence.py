"""Bounded, exclusive diagnostics for one owned sequential pytest worker.

Inputs are deterministic test fixtures, never model/user data. The caller
supplies Torch.save/is_tensor and a compact-copy function, and preserves its
original assertion. The compact copier must use detach().to(device="cpu",
copy=True, memory_format=torch.contiguous_format); every storage is checked.
No file may be overwritten. A repeated exact case/phase/label is an error;
the caller annotates the assertion when diagnostics cannot be retained.
Largest current po fixture is 2*(256*4*24*256*4) plus query/K/V, below64MiB.
Limits are per logical record64MiB, serialization80MiB, batch1GiB/128files.
Only this sequential worker may populate its private failure directory.
Files are0644 inside the owned0700 results tree so a root container can publish
synthetic fixture evidence for its authorized host owner to hash/review.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat

MAX_LOGICAL_BYTES = 64 << 20
MAX_FILE_BYTES = 80 << 20
MAX_BATCH_BYTES = 1 << 30
MAX_FILES = 128


def bounded_text(value, label, limit):
    if type(value) is not str or not value or len(value.encode("utf-8")) > limit or "\0" in value:
        raise ValueError("invalid bounded failure " + label)
    return value


def filename(name, pair_label, case_identity):
    bounded_text(name, "label", 512)
    bounded_text(pair_label, "pair label", 512)
    bounded_text(case_identity, "exact pytest case/phase", 4096)
    def safe(value, limit):
        return value if re.fullmatch(r"[A-Za-z0-9_-]{1," + str(limit) + r"}", value) else hashlib.sha256(value.encode()).hexdigest()
    case_sha = hashlib.sha256(case_identity.encode("utf-8")).hexdigest()
    return safe(name, 64) + "__" + safe(pair_label, 96) + "__" + case_sha + ".pt"


class BoundedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.bytes = stream, limit, 0

    def write(self, data):
        size = memoryview(data).nbytes
        if self.bytes + size > self.limit:
            raise MemoryError("failure serialization byte budget exhausted")
        written = self.stream.write(data)
        if written != size:
            raise OSError("short failure serialization write")
        self.bytes += written
        return written

    def flush(self):
        return self.stream.flush()

    def tell(self):
        return self.bytes


def save_failure_snapshot(directory, name, pair_identity, pair_label, case_identity, tensors, save, is_tensor, copy_tensor):
    """Return the owned saved path, or raise without overwriting prior evidence.

    Validate tensor bytes before detach/cpu/allocation; stream serialized output
    under exact byte budgets, atomically publish via exclusive hard link, and
    clean only the exclusively acquired temporary leaf on every failure.
    """
    target = filename(name, pair_label, case_identity)
    bounded_text(pair_identity, "pair identity", 128)
    if type(tensors) is not dict or not 1 <= len(tensors) <= 16:
        raise ValueError("failure evidence requires1..16 fixture tensors")
    logical_bytes = 0
    for key, tensor in tensors.items():
        bounded_text(key, "tensor label", 64)
        if not is_tensor(tensor):
            raise TypeError("failure evidence value is not a tensor")
        count, size = tensor.numel(), tensor.element_size()
        if type(count) is not int or type(size) is not int or count < 0 or not 1 <= size <= 16:
            raise ValueError("failure tensor byte geometry is invalid")
        logical_bytes += count * size
        if logical_bytes > MAX_LOGICAL_BYTES:
            raise MemoryError("failure logical tensor byte budget exhausted")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = target + ".tmp"
    acquired = False
    primary_error = None
    try:
        file_count, total_bytes = 0, 0
        with os.scandir(descriptor) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or not (entry.name.endswith(".pt") or entry.name.endswith(".pt.tmp")):
                    raise ValueError("private failure directory contains an unexpected node")
                file_count += 1
                total_bytes += info.st_size
                if info.st_size > MAX_FILE_BYTES or file_count >= MAX_FILES or total_bytes >= MAX_BATCH_BYTES:
                    raise MemoryError("failure batch file/byte budget exhausted")
        limit = min(MAX_FILE_BYTES, MAX_BATCH_BYTES - total_bytes)
        # Publication must never replace a prior same-node snapshot, and
        # acquisition must precede potentially allocating CPU copies.
        try:
            os.stat(target, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("exact failure snapshot already exists")
        output_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=descriptor)
        acquired = True
        try:
            os.fchmod(output_fd, 0o644)  # authorized host diagnostics remain readable under a restrictive umask
            stream = os.fdopen(output_fd, "wb")
        except BaseException as primary:
            try:
                os.close(output_fd)
            except BaseException as secondary:
                primary.add_note("owned failure output descriptor cleanup: " + repr(secondary))
            raise
        with stream:
            writer = BoundedWriter(stream, limit)
            copies = {}
            for key, tensor in tensors.items():
                copied = copy_tensor(tensor)
                expected_bytes = tensor.numel() * tensor.element_size()
                if (not is_tensor(copied) or copied.device.type != "cpu" or not copied.is_contiguous()
                        or copied.numel() != tensor.numel() or copied.element_size() != tensor.element_size()
                        or copied.untyped_storage().nbytes() != expected_bytes):
                    raise ValueError("compact failure CPU copy has unexpected geometry/storage")
                copies[key] = copied
            payload = {"schema": "tensorfold-kv-pair-failure-v1", "pair_identity": pair_identity,
                       "pair_label": pair_label, "label": name, "pytest_case_identity": case_identity,
                       "pytest_case_sha256": hashlib.sha256(case_identity.encode()).hexdigest(),
                       "logical_tensor_bytes": logical_bytes,
                       "tensors": copies}
            save(payload, writer)
            writer.flush()
            os.fsync(stream.fileno())
            if os.fstat(stream.fileno()).st_size != writer.bytes:
                raise ValueError("failure file size differs from serialized accounting")
        os.link(temporary, target, src_dir_fd=descriptor, dst_dir_fd=descriptor, follow_symlinks=False)
        os.unlink(temporary, dir_fd=descriptor)
        acquired = False
        return root / target
    except BaseException as primary:
        primary_error = primary
        if acquired:
            try:
                os.unlink(temporary, dir_fd=descriptor)
            except BaseException as secondary:
                primary.add_note("owned failure temporary cleanup: " + repr(secondary))
        raise
    finally:
        try:
            os.close(descriptor)
        except BaseException as secondary:
            if primary_error is None:
                raise
            primary_error.add_note("owned failure directory descriptor cleanup: " + repr(secondary))
