"""Bounded immutable RAM objects shared by prefix and retained-state manifests.

Only application payload bytes are accounted here. Complete future promises,
metadata, progress buffers and ordinary host obligations are admitted by the
catalog. No disk, pickle, device pointer or implicit process durability exists.
"""

from dataclasses import dataclass
import hashlib
import os
import threading

from tensorfold.cleanup import rollback


def digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass
class _Object:
    identity: str
    checksum: str
    data: bytearray
    references: int = 0
    readers: int = 0


class RamReader:
    """One sequential owner's reference, held through its last IO completion."""
    def __init__(self, store, key, value):
        self._store, self.key, self._value = store, key, value

    def read(self, offset, count):
        store, value = self._store, self._value
        if value is None:
            raise RuntimeError("RAM checkpoint reader is retired")
        store._check()
        if type(offset) is not int or type(count) is not int or offset < 0 or count < 0 or offset > len(value.data) - count:
            raise ValueError("RAM checkpoint read exceeds its immutable object")
        # One bounded caller-owned transfer chunk. No whole-session image.
        return bytes(memoryview(value.data)[offset:offset + count])

    def verify(self):
        if self._value is None:
            raise RuntimeError("RAM checkpoint reader is retired")
        self._store._check()
        if hashlib.sha256(self._value.data).hexdigest() != self._value.checksum:
            raise ValueError("RAM checkpoint object checksum mismatch")

    def close(self):
        store, value = self._store, self._value
        if value is None:
            return
        with store._lock:
            store._check()
            next_value, next_total = value.readers - 1, store._readers - 1
            try:
                value.readers, store._readers = next_value, next_total
                self._value = None
                store._reclaim(self.key, value)
            except BaseException as error:
                store._poison(error, self)

    def __enter__(self):
        return self

    def _close_after_error(self, primary):
        """Retain a failed loan before fallible exception-owner diagnostics."""
        try:
            self.close()
        except BaseException as cleanup:
            self._store._poison(primary, self, cleanup)
        raise primary

    def __exit__(self, _kind, error, _traceback):
        if error is not None:
            self._close_after_error(error)
        self.close()


class RamBuilder:
    """Unpublished sequential object; source remains authoritative on failure."""
    def __init__(self, store, identity, size):
        self.store, self.identity, self.size = store, identity, size
        self.data = None  # Store journals the reservation before allocating payload outside its lock.
        self.written, self.hasher = 0, hashlib.sha256()
        self.finished = False
        self.key = None  # Journals a completed publication through the return cut.

    def write(self, data):
        if self.finished:
            raise RuntimeError("RAM checkpoint builder is retired")
        value = memoryview(data).cast("B")
        if len(value) > self.size - self.written:
            raise ValueError("RAM checkpoint object write exceeds its declared size")
        self.data[self.written:self.written + len(value)] = value
        self.hasher.update(value)
        self.written += len(value)

    def finish(self):
        if self.finished or self.written != self.size:
            raise ValueError("RAM checkpoint object is incomplete or already published")
        checksum = self.hasher.hexdigest()
        key = hashlib.sha256((self.identity + ":" + checksum).encode("ascii")).hexdigest()
        store = self.store
        try:
            reader = store.acquire(key)  # Check reused bytes, not merely their old checksum label.
        except KeyError:
            reader = None
        try:
            with store._lock:
                store._check()
                old = store._objects.get(key)
                if old is not None and (old.identity != self.identity or old.checksum != checksum or len(old.data) != self.size):
                    raise ValueError("RAM checkpoint object identity collision")
                replacement = _Object(self.identity, checksum, self.data, references=1) if old is None else None
                prior_used, prior_data = store._used, self.data
                prior_refs = None if old is None else old.references
                next_refs = None if old is None else old.references + 1
                next_used = store._used if old is None else store._used - self.size
                try:
                    self.key = key
                    if old is None:
                        store._objects[key] = replacement
                    else:
                        old.references = next_refs
                    store._used = next_used
                    store._builders.remove(self)
                    self.finished, self.data = True, None
                except BaseException as error:
                    try:
                        if old is None:
                            if store._objects.get(key) is replacement:
                                store._objects.pop(key)
                        else:
                            old.references = prior_refs
                        store._builders.add(self)
                        store._used, self.data = prior_used, prior_data
                        self.finished, self.key = False, None
                    except BaseException as cleanup:
                        store._poison(error, self, cleanup)
                    raise
        except BaseException as error:
            if reader is not None:
                reader._close_after_error(error)
            raise
        else:
            if reader is not None:
                reader.close()
        return key

    def finish_into(self, keys, index):
        """Publish into an existing caller journal before returning an owned ref."""
        self.store._key_slot(keys, index)
        try:
            key = self.finish()
            keys[index] = key
        except BaseException as error:
            # finish journals its key even if interrupted at its return cut.
            if self.finished and self.key is not None and keys[index] != self.key:
                rollback(self, error, lambda: self.store.release(self.key))
            raise

    def close(self):
        if self.finished:
            return
        with self.store._lock:
            self.store._check()
            remaining = self.store._used - self.size
            try:
                self.store._builders.remove(self)
                self.store._used = remaining
                self.finished = True
                self.data = None
            except BaseException as error:
                self.store._poison(error, self)


class RamObjects:
    def __init__(self, capacity_bytes, *, max_objects, max_readers):
        if any(type(v) is not int or v <= 0 for v in (capacity_bytes, max_objects, max_readers)):
            raise ValueError("RAM object bytes/count/readers require explicit positive admission bounds")
        self.capacity, self.max_objects, self.max_readers = capacity_bytes, max_objects, max_readers
        self._pid, self._lock = os.getpid(), threading.RLock()
        self._objects, self._builders = {}, set()
        self._used = self._readers = 0
        self._failure = self._retained_reader = None

    def _check(self):
        if self._pid != os.getpid():
            raise RuntimeError("RAM checkpoint objects belong to another process")
        if self._failure is not None:
            raise RuntimeError("RAM checkpoint ownership is poisoned and remains retained") from self._failure

    def check_health(self):
        """Read-only public predicate for the catalog's failure classification."""
        self._check()

    def _poison(self, error, reader=None, secondary=None):
        if self._failure is None:
            self._failure = error
        self._retained_reader = reader
        try:
            rollback(self, error, self.close)
        except BaseException as reported:
            if secondary is not None:
                from tensorfold.cleanup import raise_failures

                raise_failures(reported, (secondary,))
            raise

    def begin(self, identity, size):
        self._check()
        if not digest(identity) or type(size) is not int or size < 0:
            raise ValueError("RAM checkpoint needs exact content/profile identity and nonnegative size")
        with self._lock:
            self._check()  # An actor-side failure may latch while this producer waits for the lock.
            prior_used, next_used = self._used, self._used + size
            if next_used > self.capacity or len(self._objects) + len(self._builders) >= self.max_objects:
                raise MemoryError("RAM checkpoint payload/object capacity is committed")
            builder = RamBuilder(self, identity, size)
            # Publish the complete reservation before allocating payload outside
            # the lock. Construction/arithmetic has not yet changed the store.
            try:
                self._builders.add(builder)
                self._used = next_used
            except BaseException as error:
                try:
                    self._builders.discard(builder)
                    self._used = prior_used
                except BaseException as cleanup:
                    self._poison(error, builder, cleanup)
                raise
        try:
            builder.data = bytearray(size)
        except BaseException as error:
            rollback(builder, error, builder.close)
        return builder

    def retain(self, key):
        with self._lock:
            self._check()
            value = self._objects[key]
            value.references += 1

    @staticmethod
    def _key_slot(keys, index):
        if type(keys) is not list or type(index) is not int or not 0 <= index < len(keys) or keys[index] is not None:
            raise ValueError("RAM reference publication requires an empty preallocated caller journal slot")

    def retain_into(self, key, keys, index):
        self._key_slot(keys, index)
        with self._lock:
            self._check()
            value = self._objects[key]
            prior, following = value.references, value.references + 1
            try:
                value.references = following
                keys[index] = key
            except BaseException as error:
                if keys[index] != key:
                    try:
                        value.references = prior
                    except BaseException as cleanup:
                        self._poison(error, value, cleanup)
                raise

    def release(self, key):
        with self._lock:
            self._check()
            value = self._objects[key]
            if value.references <= 0:
                raise RuntimeError("RAM checkpoint manifest reference underflow")
            value.references -= 1
            self._reclaim(key, value)

    def _reclaim(self, key, value):
        if value.references == 0 and value.readers == 0:
            remaining = self._used - len(value.data)
            try:
                self._objects.pop(key)
                self._used = remaining
            except BaseException as error:
                self._poison(error)

    def acquire(self, key, *, verify=True):
        with self._lock:
            self._check()
            if self._readers >= self.max_readers:
                raise MemoryError("RAM checkpoint reader capacity is busy")
            value = self._objects[key]
            # Fallible construction/arithmetic precedes publication. A failed
            # constructor owns no count, and cannot strand an invisible lease.
            reader = RamReader(self, key, value)
            next_value, next_total = value.readers + 1, self._readers + 1
            try:
                value.readers, self._readers = next_value, next_total
            except BaseException as error:
                self._poison(error, reader)
        try:
            if verify:
                reader.verify()  # No catalog/store lock spans checksum traversal.
        except BaseException as error:
            reader._close_after_error(error)
        return reader

    def record(self):
        with self._lock:
            self._check()
            return {"payload_bytes": self._used, "limit_bytes": self.capacity, "objects": len(self._objects),
                    "building": len(self._builders), "readers": self._readers}

    def descriptor(self, key):
        with self._lock:
            self._check()
            value = self._objects[key]
            return {"identity": value.identity, "checksum": value.checksum, "bytes": len(value.data)}

    def close(self):
        with self._lock:
            self._check()
            if self._builders or self._readers or any(obj.references for obj in self._objects.values()):
                raise RuntimeError("RAM checkpoints still have protected manifests/builders/readers")
            self._objects.clear()
            self._used = 0
