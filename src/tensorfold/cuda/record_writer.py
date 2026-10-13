"""Opt-in telemetry leaves the engine thread through a bounded queue that a daemon thread drains to disk.

``put`` never blocks: when the writer falls behind, the oldest queued record is dropped and counted (``dropped``), so
a slow disk costs records, never decode time or unbounded memory. Its destination is opened or created when the
writer is made, so a bad path fails at startup. A write that raises stops the writer: the error reaches
``threading.excepthook`` and ``failed``, later records are dropped and counted, and decoding goes on. ``close``
(the engine's shutdown) writes what is queued, then raises if the writer failed or could not finish in time.
"""

from __future__ import annotations

import json
import os
import queue
import threading

from tensorfold.cleanup import raise_failures

_STOP = object()


def close_all(writers, timeout: float = 10.0) -> None:
    """Close every writer, then raise their failures together (the first, with the others attached)."""

    errors = []
    for writer in writers:
        try:
            writer.close(timeout)
        except Exception as exc:                     # noqa: BLE001  every writer is closed before any is reported
            errors.append(exc)
    raise_failures(None, errors)


class RecordWriter:
    """Subclasses define ``write(record)``; ``flush()`` runs whenever the queue empties and ``end()`` as it stops."""

    def __init__(self, *, name: str, queue_size: int = 4096) -> None:
        self.records: queue.Queue = queue.Queue(maxsize=queue_size)
        self.dropped = 0
        self.failed: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def put(self, record) -> None:
        if self.failed is not None:                  # nothing drains the queue any more
            self.dropped += 1
            return
        try:
            self.records.put_nowait(record)
        except queue.Full:
            try:
                self.records.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.records.put_nowait(record)
            except queue.Full:
                self.dropped += 1

    def close(self, timeout: float = 10.0) -> None:
        """Write every queued record, then stop; raises if the writer failed or did not finish within ``timeout``."""

        if self.thread.is_alive() and self.failed is None:
            try:
                self.records.put(_STOP, timeout=timeout)
            except queue.Full:
                raise TimeoutError(f"{self.thread.name}: records still unwritten after {timeout} s") from None
            self.thread.join(timeout)
            if self.thread.is_alive():
                raise TimeoutError(f"{self.thread.name}: records still unwritten after {timeout} s")
        if self.failed is not None:
            raise RuntimeError(f"{self.thread.name} stopped: {self.failed!r} ({self.dropped} records dropped)") \
                from self.failed

    def write(self, record) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        pass

    def end(self) -> None:
        pass

    def _run(self) -> None:
        try:
            while True:
                record = self.records.get()
                if record is _STOP:
                    break
                self.write(record)
                if self.records.empty():
                    self.flush()
        except BaseException as exc:
            self.failed = exc
            raise
        finally:
            self.end()


class JsonLines(RecordWriter):
    """Append each record to ``path`` as one compact JSON line; ``stamp(record)`` adds fields as it is written."""

    def __init__(self, path: str, *, name: str, stamp=None, queue_size: int = 4096) -> None:
        self.path, self.stamp = path, stamp
        self.out = open(path, "a", encoding="utf-8")
        super().__init__(name=name, queue_size=queue_size)

    def write(self, record) -> None:
        if self.stamp is not None:
            self.stamp(record)
        self.out.write(json.dumps(record, separators=(",", ":")) + "\n")

    def flush(self) -> None:
        self.out.flush()

    def end(self) -> None:
        self.out.close()


class JsonFiles(RecordWriter):
    """Each record ``(name, document)`` becomes ``directory/name``, readable by its owner only (mode 0600).

    A file appears complete or not at all: it is written and synced under a hidden temporary name, then renamed; a
    failed write removes the temporary. Each document gains ``dropped``: documents lost to a full queue before it.
    """

    def __init__(self, directory: str, *, name: str, queue_size: int = 256) -> None:
        self.directory = directory
        os.makedirs(directory, mode=0o700, exist_ok=True)
        super().__init__(name=name, queue_size=queue_size)

    def write(self, record) -> None:
        name, document = record
        document["dropped"] = self.dropped
        path = os.path.join(self.directory, name)
        temporary = os.path.join(self.directory, f".{name}.part")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)                     # a stale temporary keeps its mode through O_CREAT
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                fd = -1
                json.dump(document, out, separators=(",", ":"), default=int)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, path)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
