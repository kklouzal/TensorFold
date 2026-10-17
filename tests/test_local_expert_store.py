"""Loopback listener admission and native-FD handoff failure containment; no GPU required."""
import errno
import gc
import os
from pathlib import Path
import socket
import sys

import pytest

from tensorfold.cuda import local_store


@pytest.mark.parametrize("options", [dict(master="0.0.0.0"), dict(master="192.0.2.1"), dict(master="remote.invalid"),
                                     dict(port=-1), dict(port=65536), dict(world=5), dict(rank=True),
                                     dict(world=2, port=0), dict(timeout_s=0), dict(timeout_s=float("nan"))])
def test_invalid_store_geometry_fails_before_native_import_or_listener(monkeypatch, options):
    def unexpected():
        pytest.fail("invalid rendezvous reached native import")
    monkeypatch.setattr(local_store, "_store_type", unexpected)
    with pytest.raises(ValueError):
        local_store.open_store(**{**dict(master="127.0.0.1", port=29551, world=1, rank=0, timeout_s=3), **options})


def test_client_has_no_listener_and_retains_bounded_worker_barrier_policy(monkeypatch):
    seen = []
    result = object()
    monkeypatch.setattr(local_store, "_store_type", lambda: lambda *args, **kw: seen.append((args, kw)) or result)
    assert local_store.open_store("localhost", 29551, 2, 1, timeout_s=3) is result
    args, options = seen[0]
    assert args == ("127.0.0.1", 29551, 2, False)
    assert options["wait_for_workers"] is False and options["use_libuv"] is False
    assert "master_listen_fd" not in options


@pytest.mark.parametrize("reuse", [False, True])
def test_failed_native_handoff_never_closes_or_retries_an_unproved_numeric_fd(monkeypatch, tmp_path, reuse):
    primary = RuntimeError("controlled native constructor failure")
    owned = []
    path = tmp_path / "unrelated"
    path.write_bytes(b"sentinel")

    def failed(*args, master_listen_fd, **kwargs):
        fd = master_listen_fd
        if reuse:
            os.close(fd)                         # a native constructor may adopt and close before raising
            replacement = os.open(path, os.O_RDONLY)
            assert replacement == fd
            owned.append(replacement)             # the test owns this unrelated descriptor, not the helper
        raise primary

    monkeypatch.setattr(local_store, "_store_type", lambda: failed)
    try:
        with pytest.raises(RuntimeError) as caught:
            local_store.open_store("127.0.0.1", 0, 1, 0, timeout_s=3)
        assert caught.value is primary
        cause = primary.__cause__
        assert isinstance(cause, local_store.UnresolvedStoreOwner)
        assert cause.owner.phase == "unresolved" and cause.owner.socket.fileno() == -1
        if reuse:
            assert os.read(owned[0], 8) == b"sentinel"
        else:
            # Native was a controlled stub that demonstrably did NOT adopt.
            # Only this test has that extra fact; production must contain the process.
            owned.append(cause.owner.descriptor)
            with socket.fromfd(owned[0], socket.AF_INET, socket.SOCK_STREAM) as borrowed:
                assert borrowed.getsockname()[0] == "127.0.0.1"
    finally:
        for fd in owned:
            os.close(fd)


def test_interruption_after_detach_before_native_call_retires_proven_owned_fd(monkeypatch):
    real_socket = socket.socket
    detached = []
    primary = KeyboardInterrupt("controlled detach return interruption")

    class Listener:
        def __init__(self, *args):
            self.value = real_socket(*args)
        def __getattr__(self, name):
            return getattr(self.value, name)
        def detach(self):
            detached.append(self.value.detach())
            raise primary

    monkeypatch.setattr(socket, "socket", Listener)
    monkeypatch.setattr(local_store, "_store_type", lambda: lambda *args, **kw: pytest.fail("native handoff entered"))
    with pytest.raises(KeyboardInterrupt) as caught:
        local_store.open_store("127.0.0.1", 0, 1, 0, timeout_s=3)
    assert caught.value is primary and len(detached) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(detached[0])
    assert closed.value.errno == errno.EBADF


def test_bind_and_cleanup_failures_keep_primary_and_unresolved_owner(monkeypatch):
    primary, cleanup = OSError("bind failure"), OSError("socket close failure")

    class Listener:
        def bind(self, address):
            raise primary
        def fileno(self):
            return 99
        def close(self):
            raise cleanup

    monkeypatch.setattr(socket, "socket", lambda *args: Listener())
    monkeypatch.setattr(local_store, "_store_type", lambda: lambda *a, **k: pytest.fail("native handoff entered"))
    with pytest.raises(OSError) as caught:
        local_store.open_store("127.0.0.1", 0, 1, 0, timeout_s=3)
    assert caught.value is primary
    assert isinstance(primary.__cause__, BaseExceptionGroup)
    assert primary.__cause__.exceptions[0] is cleanup
    unresolved = primary.__cause__.exceptions[1]
    assert isinstance(unresolved, local_store.UnresolvedStoreOwner) and unresolved.owner.phase == "unresolved"


@pytest.mark.parametrize("native", [False, True])
def test_failed_unresolved_owner_diagnostic_preserves_primary_and_retained_listener(monkeypatch, native):
    primary = RuntimeError("controlled startup failure")
    cleanup = OSError("controlled listener retirement failure")
    diagnostic = MemoryError("controlled diagnostic allocation failure")
    captured = []

    class Listener:
        def bind(self, address):
            if not native:
                raise primary
        def listen(self):
            pass
        def getsockname(self):
            return "127.0.0.1", 29551
        def fileno(self):
            return -1 if native else 99
        def detach(self):
            pass
        def close(self):
            raise cleanup

    def failed_native(*args, **kwargs):
        raise primary

    def failed_diagnostic(owner):
        captured.append(owner)
        raise diagnostic

    monkeypatch.setattr(socket, "socket", lambda *args: Listener())
    monkeypatch.setattr(local_store, "_store_type", lambda: failed_native)
    monkeypatch.setattr(local_store, "UnresolvedStoreOwner", failed_diagnostic)
    with pytest.raises(RuntimeError) as caught:
        local_store.open_store("127.0.0.1", 29551, 1, 0, timeout_s=3)
    assert caught.value is primary and primary.__cause__ is diagnostic
    frames, current = [], primary.__traceback__
    while current is not None:
        frames.append(current.tb_frame)
        current = current.tb_next
    assert any(frame.f_locals.get("owner") is captured[0] for frame in frames)
    assert captured[0].phase == "unresolved"
    if not native:
        assert diagnostic.__context__ is cleanup


def test_returned_native_owner_retires_after_interrupted_publication(monkeypatch):
    real_listener = local_store._Listener
    closed = []
    primary = KeyboardInterrupt("interrupted native owner publication")

    class Listener(real_listener):
        def __setattr__(self, name, value):
            if name == "phase" and value == "native":
                raise primary
            super().__setattr__(name, value)

    class Native:
        def __init__(self, descriptor):
            self.socket = socket.socket(fileno=descriptor)
        def __del__(self):
            self.socket.close()
            closed.append(True)

    monkeypatch.setattr(local_store, "_Listener", Listener)
    monkeypatch.setattr(local_store, "_store_type",
                        lambda: lambda *a, master_listen_fd, **k: Native(master_listen_fd))
    with pytest.raises(KeyboardInterrupt) as caught:
        local_store.open_store("127.0.0.1", 0, 1, 0, timeout_s=3)
    assert caught.value is primary and closed == [True]


@pytest.mark.skipif(sys.platform != "linux", reason="actual listener address oracle reads Linux procfs")
def test_pinned_native_store_binds_only_loopback_and_retires_listener():
    pytest.importorskip("torch.distributed")

    def bindings(port):
        found = []
        for name in ("/proc/net/tcp", "/proc/net/tcp6"):
            try:
                rows = Path(name).read_text().splitlines()[1:]
            except FileNotFoundError:
                continue
            for line in rows:
                fields = line.split()
                if fields[3] == "0A" and int(fields[1].split(":")[1], 16) == port:
                    found.append(fields[1].split(":")[0])
        return found

    store = local_store.open_store("127.0.0.1", 0, 1, 0, timeout_s=3)
    port = store.port
    assert bindings(port) == ["0100007F"]
    store.set("typed-control-probe", "ready")
    assert store.get("typed-control-probe") == b"ready"
    del store
    gc.collect()
    assert bindings(port) == []


def test_pinned_native_loopback_client_and_server_exchange_control_values():
    pytest.importorskip("torch.distributed")
    # Only select a currently available port here; the production listener's
    # exclusive bind remains the authority if another process wins that port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    server = local_store.open_store("127.0.0.1", port, 2, 0, timeout_s=3)
    client = local_store.open_store("127.0.0.1", port, 2, 1, timeout_s=3)
    server.set("server-control", b"typed-server")
    assert client.get("server-control") == b"typed-server"
    client.set("client-control", b"typed-client")
    assert server.get("client-control") == b"typed-client"
