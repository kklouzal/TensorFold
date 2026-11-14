"""Same-host control pipes between a tensor-parallel leader and its peers.

The leader-owned scheduler protocol publishes one command per operation and
joins one acknowledgment from every peer before the next command. Through the
rendezvous Store that costs a publish, a wait, a batch get and one delete per
key on the leader's serial path every round (~1.5-2 ms at C1 on 8x V100), while
every peer idles. A peer that shares the leader's kernel instead receives the
same bytes through a FIFO pair: one leader-to-peer command pipe and one
peer-to-leader acknowledgment pipe, framed as an 8-byte little-endian length
plus payload (at most ``MAX_MESSAGE_BYTES``, checked before allocation).

Rendezvous uses the Store once: the leader creates the FIFOs in a private
(0700) temporary directory, opens its acknowledgment read ends and publishes
the directory; each peer opens its command read end and acknowledgment write
end (which succeeds only while the leader's read end is open, so a same-named
path on another host cannot pass) and publishes whether it joined. Before its
first command the leader waits for every peer's answer, opens the joined
peers' command write ends, then removes the directory and the rendezvous keys.
A peer that cannot open the directory (another host) keeps the Store protocol;
the leader serves each peer through the transport that peer declared.

Process death closes the pipe ends: a peer reads end-of-file when the leader
departs (or, before the leader's first command opened the command pipe, sees its
acknowledgment pipe lose the leader's read end), and the leader reads end-of-file
(or EPIPE on write) when a peer departs, so neither waits for a dead process. Waits are bounded by the same
timeout the Store protocol used. A send or read interrupted mid-frame leaves the
stream unusable; such a channel is marked broken and only closed afterwards.
"""

from __future__ import annotations

import errno
import os
import secrets
import select
import shutil
import struct
import tempfile
import time
import weakref

from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
from tensorfold.cuda.store_fields import validate_store_key

_HEADER = struct.Struct("<Q")
_JOINED, _ABSENT = b"pipe", b"store"


class PeerDeparted(RuntimeError):
    """The other end of a control pipe closed (its process retired or died)."""


def _key(nonce, name):
    key = f"tensorfold/flashnext/pipes/{nonce}/{name}"
    validate_store_key(key)
    return key


def _directory_root():
    shm = "/dev/shm"
    return shm if os.path.isdir(shm) and os.access(shm, os.W_OK | os.X_OK) else None


def _close_all(descriptors, directory):
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass
    descriptors.clear()
    if directory:
        shutil.rmtree(directory[0], ignore_errors=True)
        directory.clear()


def _deadline_ms(deadline):
    return max(0, int((deadline - time.monotonic()) * 1000) + 1)


def _write(descriptor, data, deadline):
    view = memoryview(data)
    poller = None
    while view:
        try:
            written = os.write(descriptor, view)
        except BlockingIOError:
            if poller is None:
                poller = select.poll()
                poller.register(descriptor, select.POLLOUT)
            if time.monotonic() >= deadline:
                raise TimeoutError("tensor-parallel control pipe write timed out")
            poller.poll(_deadline_ms(deadline))
            continue
        except BrokenPipeError as error:
            raise PeerDeparted("tensor-parallel control peer departed") from error
        view = view[written:]


def _read_exact(descriptor, size, deadline, *, poller=None):
    """Exactly ``size`` bytes, None on end-of-file before the first byte."""
    buffer = bytearray(size)
    view, filled = memoryview(buffer), 0
    while filled < size:
        try:
            count = os.readv(descriptor, [view[filled:]])
        except BlockingIOError:
            if poller is None:
                poller = select.poll()
                poller.register(descriptor, select.POLLIN)
            if time.monotonic() >= deadline:
                raise TimeoutError("tensor-parallel control pipe read timed out") from None
            poller.poll(_deadline_ms(deadline))
            continue
        if count == 0:
            if filled == 0:
                return None
            raise PeerDeparted("tensor-parallel control pipe closed inside a frame")
        filled += count
    return bytes(buffer)


class ControlPipes:
    """One rank's ends of the leader/peer FIFO pairs (see module docstring)."""

    def __init__(self, store, rank, world, *, timeout_s):
        if type(world) is not int or world < 2 or type(rank) is not int or not 0 <= rank < world:
            raise ValueError("control pipes require a multi-rank cohort and a valid rank")
        self.store, self.rank, self.world, self.timeout_s = store, rank, world, float(timeout_s)
        self._descriptors, self._directory = [], []
        self._finalizer = weakref.finalize(self, _close_all, self._descriptors, self._directory)
        self.broken = False
        if rank == 0:
            self._open_leader()
        else:
            self._open_peer()

    # Leader ------------------------------------------------------------------
    def _open_leader(self):
        self.nonce = secrets.token_hex(16)
        self.writers, self.readers, self.peers = {}, {}, None
        root = _directory_root()
        try:
            directory = tempfile.mkdtemp(prefix="tensorfold-tp-", dir=root)
            self._directory.append(directory)
            for peer in range(1, self.world):
                os.mkfifo(os.path.join(directory, f"command-{peer}"), 0o600)
                os.mkfifo(os.path.join(directory, f"ack-{peer}"), 0o600)
                reader = os.open(os.path.join(directory, f"ack-{peer}"), os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
                self._descriptors.append(reader)
                self.readers[peer] = reader
            self.store.set(_key("current", "directory"), f"{self.nonce}:{directory}")
        except BaseException:
            self._finalizer()
            raise

    def _join(self):
        """Before the first command: learn each peer's transport and finish the pipe pairs."""
        if self.peers is not None:
            return
        from datetime import timedelta

        keys = [_key(self.nonce, f"peer/{peer}") for peer in range(1, self.world)]
        self.store.wait(keys, timedelta(seconds=self.timeout_s))
        answers = [self.store.get(key) for key in keys]
        peers = set()
        for peer, answer in enumerate(answers, 1):
            if answer == _JOINED:
                directory = self._directory[0]
                writer = os.open(os.path.join(directory, f"command-{peer}"),
                                 os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
                self._descriptors.append(writer)
                self.writers[peer] = writer
                peers.add(peer)
            elif answer != _ABSENT:
                raise ValueError("tensor-parallel control pipe rendezvous answer is invalid")
        for peer in set(self.readers) - peers:
            reader = self.readers.pop(peer)
            self._descriptors.remove(reader)
            os.close(reader)
        shutil.rmtree(self._directory.pop(), ignore_errors=True)
        for key in (*keys, _key("current", "directory")):
            self.store.delete_key(key)
        self.peers = frozenset(peers)

    def piped(self):
        """Peers served through pipes; resolves the rendezvous on first use."""
        self._join()
        return self.peers

    def send(self, payload):
        if self.broken:
            raise RuntimeError("tensor-parallel control pipes are broken; retire the cohort")
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError("tensor-parallel control frame exceeds the 96 MiB transport envelope")
        frame = _HEADER.pack(len(payload)) + payload
        deadline = time.monotonic() + self.timeout_s
        try:
            for peer in sorted(self.piped()):
                _write(self.writers[peer], frame, deadline)
        except BaseException:
            self.broken = True
            raise

    def acknowledgments(self):
        """One frame from every piped peer, in peer order."""
        if self.broken:
            raise RuntimeError("tensor-parallel control pipes are broken; retire the cohort")
        deadline = time.monotonic() + self.timeout_s
        try:
            pending = {self.readers[peer]: peer for peer in self.piped()}
            frames = {}
            poller = select.poll()
            for descriptor in pending:
                poller.register(descriptor, select.POLLIN)
            while pending:
                if time.monotonic() >= deadline:
                    raise TimeoutError("tensor-parallel acknowledgment wait timed out")
                for descriptor, _events in poller.poll(_deadline_ms(deadline)):
                    peer = pending.pop(descriptor)
                    poller.unregister(descriptor)
                    frames[peer] = self._frame(descriptor, deadline)
            return [frames[peer] for peer in sorted(frames)]
        except BaseException:
            self.broken = True
            raise

    # Peer --------------------------------------------------------------------
    def _open_peer(self):
        from datetime import timedelta

        self.reader = self.writer = None
        key = _key("current", "directory")
        self.store.wait([key], timedelta(seconds=self.timeout_s))
        nonce, _, directory = self.store.get(key).decode().partition(":")
        answer = _ABSENT
        try:
            reader = os.open(os.path.join(directory, f"command-{self.rank}"),
                             os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            self._descriptors.append(reader)
            writer = os.open(os.path.join(directory, f"ack-{self.rank}"), os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            self._descriptors.append(writer)
            self.reader, self.writer, answer = reader, writer, _JOINED
        except OSError as error:
            if error.errno not in (errno.ENOENT, errno.ENXIO, errno.ENOTDIR):
                self._finalizer()
                raise
            self._finalizer()  # Another host: keep the Store protocol for this peer.
        self.store.set(_key(nonce, f"peer/{self.rank}"), answer)

    @property
    def joined(self):
        return self.rank != 0 and self.reader is not None

    def receive(self, timeout_s):
        """The next command frame; None when the leader departed. TimeoutError after ``timeout_s`` idle."""
        if self.broken:
            raise RuntimeError("tensor-parallel control pipes are broken; retire the cohort")
        poller = select.poll()
        poller.register(self.reader, select.POLLIN)
        # The leader opens the command write ends only at its first command, and a FIFO
        # reports no hang-up to a reader that never had a writer. The acknowledgment pipe
        # has had the leader's read end since rendezvous: POLLERR on this write end means
        # that reader is gone, i.e. the leader departed, before or between commands.
        poller.register(self.writer, 0)
        events = poller.poll(int(timeout_s * 1000))
        if not events:
            raise TimeoutError("tensor-parallel command wait timed out")
        if not any(descriptor == self.reader for descriptor, _ in events):
            return None
        try:
            return self._frame(self.reader, time.monotonic() + self.timeout_s, departed=None)
        except BaseException:
            self.broken = True
            raise

    def acknowledge(self, payload):
        if self.broken:
            raise RuntimeError("tensor-parallel control pipes are broken; retire the cohort")
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError("tensor-parallel acknowledgment exceeds the 96 MiB transport envelope")
        try:
            _write(self.writer, _HEADER.pack(len(payload)) + payload, time.monotonic() + self.timeout_s)
        except BaseException:
            self.broken = True
            raise

    # Shared ------------------------------------------------------------------
    def _frame(self, descriptor, deadline, *, departed=PeerDeparted):
        header = _read_exact(descriptor, _HEADER.size, deadline)
        if header is None:
            if departed is None:
                return None
            raise departed("tensor-parallel control peer departed")
        (size,) = _HEADER.unpack(header)
        if size > MAX_MESSAGE_BYTES:
            raise ValueError("tensor-parallel control frame exceeds the 96 MiB transport envelope")
        payload = _read_exact(descriptor, size, deadline) if size else b""
        if payload is None:
            raise PeerDeparted("tensor-parallel control pipe closed inside a frame")
        return payload

    def close(self):
        """Close this rank's ends; the other side then reads end-of-file."""
        self._finalizer()
