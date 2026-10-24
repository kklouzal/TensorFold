"""Host-only controller fault coverage; never starts Docker or a subprocess."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gb10_controller as controller
import gb10_io as owned_io


def roots(error):
    """Inspect native transport links without relying on formatted messages."""
    seen, pending, result = set(), [error], []
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        pending.extend((BaseException.__cause__.__get__(current), BaseException.__context__.__get__(current)))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return result


class Pipe:
    def __init__(self, fd, failure=None):
        self.fd, self.failure, self.closed = fd, failure, 0

    def fileno(self):
        return self.fd

    def close(self):
        self.closed += 1
        if self.failure is not None:
            raise self.failure


class Client:
    def __init__(self, stdout=None, stderr=None, kill_error=None, wait_error=None):
        self.pid = 91
        self.stdout, self.stderr = stdout or Pipe(1), stderr or Pipe(2)
        self.kill_error, self.wait_error = kill_error, wait_error
        self.kills, self.waits = 0, []

    def kill(self):
        self.kills += 1
        if self.kill_error is not None:
            raise self.kill_error

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.wait_error is not None:
            raise self.wait_error
        return 0

    def poll(self):
        return None if self.wait_error is not None else 0


class Selector:
    def __init__(self, failure=None):
        self.entries, self.failure, self.closed = {}, failure, 0

    def register(self, pipe, event, name):
        self.entries[pipe] = NS(fileobj=pipe, data=name)

    def get_map(self):
        return self.entries

    def select(self, timeout):
        return [(key, 1) for key in self.entries.values()]

    def unregister(self, pipe):
        del self.entries[pipe]

    def close(self):
        self.closed += 1
        if self.failure is not None:
            raise self.failure


def prepared_host(monkeypatch, tmp_path, *, client=None, selector=None, read_error=None,
                  publication_error=None, spawn_error=None, clock=None):
    client, selector = client or Client(), selector or Selector()
    spawns, publications = [], []
    chunks = {1: iter((b"payload", b"")), 2: iter((b"",))}

    def spawn(argv, **options):
        spawns.append((argv, options))
        if spawn_error is not None:
            raise spawn_error
        return client

    def read(fd, count):
        if read_error is not None:
            raise read_error
        return next(chunks[fd])

    def publish(path, calls):
        publications.append((path, [dict(row) for row in calls]))
        if publication_error is not None:
            raise publication_error

    monkeypatch.setattr(controller, "subprocess", NS(Popen=spawn, PIPE=-1))
    monkeypatch.setattr(controller, "selectors", NS(DefaultSelector=lambda: selector, EVENT_READ=1))
    monkeypatch.setattr(controller, "os", NS(read=read))
    monkeypatch.setattr(controller, "time", NS(monotonic=clock or (lambda: 0)))
    monkeypatch.setattr(controller, "publish", publish)
    host = controller.Host(tmp_path, descriptors=(17,))
    return host, client, selector, spawns, publications


@pytest.mark.parametrize("prior", [False, True])
def test_host_success_preserves_record_bytes_arguments_and_lifecycle(monkeypatch, tmp_path, prior):
    host, client, selector, spawns, publications = prepared_host(monkeypatch, tmp_path)
    host.lifecycle_uncertain = prior
    argv = ["docker", "start", "exact-cid"]
    assert host.command(argv, timeout=20) == b"payload"
    assert spawns == [(argv, {"stdout": -1, "stderr": -1, "pass_fds": (17,)})]
    assert client.kills == 0 and client.waits == [20]
    assert selector.closed == client.stdout.closed == client.stderr.closed == 1
    assert host.uncertain is False and host.lifecycle_uncertain is prior
    row = publications[0][1][0]
    assert row["argv"] == argv and row["timeout_s"] == 20 and row["exit"] == 0
    assert row["complete"] is True and row["stdout_bytes"] == 7 and row["stderr_bytes"] == 0
    assert publications[0][0] == tmp_path / "commands.json" and host.clients_retired()


def test_read_primary_keeps_all_independent_retirement_and_publication_roots(monkeypatch, tmp_path):
    primary, cause = OSError("read"), ValueError("original native cause")
    primary.__cause__ = cause
    selector_error, kill_error, wait_error = OSError("selector"), OSError("kill"), subprocess.TimeoutExpired("client", 5)
    out_error, err_error, pub_error = OSError("stdout-close"), OSError("stderr-close"), OSError("publish")
    client = Client(Pipe(1, out_error), Pipe(2, err_error), kill_error, wait_error)
    host, _, selector, spawns, publications = prepared_host(
        monkeypatch, tmp_path, client=client, selector=Selector(selector_error),
        read_error=primary, publication_error=pub_error)
    with pytest.raises(OSError) as caught:
        host.command(["docker", "stop", "exact-cid"], timeout=20)
    assert caught.value is primary
    found = roots(primary)
    assert all(error in found for error in (cause, selector_error, kill_error, wait_error, out_error, err_error, pub_error))
    assert len(spawns) == 1 and client.kills == 1 and client.waits == [5]
    assert selector.closed == client.stdout.closed == client.stderr.closed == 1
    assert host.pending_clients == [client] and not host.clients_retired()
    assert host.uncertain is host.lifecycle_uncertain is True
    assert len(publications) == 1 and publications[0][1][0]["complete"] is False


@pytest.mark.parametrize("stage", ["selector", "stdout", "stderr", "publish"])
def test_successful_client_cleanup_failure_is_explicit_and_does_not_discharge_lifecycle(monkeypatch, tmp_path, stage):
    failure = OSError(stage)
    client = Client(Pipe(1, failure if stage == "stdout" else None), Pipe(2, failure if stage == "stderr" else None))
    host, _, selector, _, publications = prepared_host(
        monkeypatch, tmp_path, client=client, selector=Selector(failure if stage == "selector" else None),
        publication_error=failure if stage == "publish" else None)
    with pytest.raises(OSError) as caught:
        host.command(["docker", "start", "exact-cid"])
    assert caught.value is failure
    assert selector.closed == client.stdout.closed == client.stderr.closed == 1
    assert len(publications) == 1 and host.calls[0]["complete"] is False
    assert host.uncertain is host.lifecycle_uncertain is True


def test_spawn_failure_and_publication_failure_preserve_spawn_identity(monkeypatch, tmp_path):
    primary, secondary = OSError("spawn"), OSError("publish")
    host, client, _, spawns, publications = prepared_host(
        monkeypatch, tmp_path, spawn_error=primary, publication_error=secondary)
    with pytest.raises(OSError) as caught:
        host.command(["docker", "container", "inspect", "exact-cid"])
    assert caught.value is primary and secondary in roots(primary)
    assert len(spawns) == len(publications) == 1 and client.kills == 0
    assert host.calls[0]["client_publication_unproven"] is True and host.uncertain is True


def test_original_deadline_failure_still_retires_client_once(monkeypatch, tmp_path):
    ticks = iter((0, 21))
    host, client, selector, spawns, publications = prepared_host(monkeypatch, tmp_path, clock=lambda: next(ticks))
    with pytest.raises(RuntimeError, match="bounded Docker client timed out"):
        host.command(["docker", "container", "inspect", "exact-cid"], timeout=20)
    assert len(spawns) == len(publications) == client.kills == 1 and client.waits == [5]
    assert selector.closed == client.stdout.closed == client.stderr.closed == 1
    assert host.uncertain is True and host.calls[0]["complete"] is False


def test_buffer_metadata_failure_still_closes_both_pipes_and_publishes_failure(monkeypatch, tmp_path):
    failure = MemoryError("buffer fingerprint")
    host, client, selector, _, publications = prepared_host(monkeypatch, tmp_path)

    def failed_hash(value):
        raise failure

    monkeypatch.setattr(controller, "hashlib", NS(sha256=failed_hash))
    with pytest.raises(MemoryError) as caught:
        host.command(["docker", "container", "inspect", "exact-cid"])
    assert caught.value is failure
    assert selector.closed == client.stdout.closed == client.stderr.closed == 1
    assert len(publications) == 1 and publications[0][1][0]["complete"] is False
    assert host.uncertain is True


class RawOutput:
    def __init__(self, *, write_error=None, close_error=None):
        self.data, self.closes, self.flushes = bytearray(), 0, 0
        self.write_error, self.close_error = write_error, close_error

    def write(self, value):
        if self.write_error is not None:
            raise self.write_error
        self.data.extend(value.encode("utf-8"))
        return len(value)

    def flush(self):
        self.flushes += 1

    def fileno(self):
        return 71

    def close(self):
        self.closes += 1
        if self.close_error is not None:
            raise self.close_error


def fake_output(monkeypatch, output):
    acquired = []

    class OutputPath:
        def open(self, mode, **options):
            assert mode == "x" and options == {"encoding": "utf-8"}
            acquired.append(True)
            return output

    # Keep the actual shared owner and inject only its ordinary Path.open
    # resource acquisition, after the helper's documented Path normalization.
    monkeypatch.setattr(owned_io, "Path", lambda path: OutputPath())
    return acquired


@pytest.mark.parametrize("stage", ["serialize", "write", "fsync"])
def test_publisher_body_and_raw_close_failures_preserve_primary(monkeypatch, tmp_path, stage):
    primary, secondary = OSError(stage), OSError("raw-close")
    output = RawOutput(write_error=primary if stage == "write" else None, close_error=secondary)
    acquired = fake_output(monkeypatch, output)
    if stage == "serialize":
        def dump(value, stream, **options):
            stream.write('{"partial":')
            raise primary
        monkeypatch.setattr(controller, "json", NS(dump=dump))
    if stage == "fsync":
        def fsync(fd):
            raise primary
        monkeypatch.setattr(controller, "os", NS(fsync=fsync))
    with pytest.raises(OSError) as caught:
        controller.publish(tmp_path / "commands.json", {"key": "value"})
    assert caught.value is primary and secondary in roots(primary)
    assert acquired == [True] and output.closes == 1
    assert not (tmp_path / "commands.json").exists()


def test_publisher_keeps_wire_bytes_mode_and_both_durability_steps(monkeypatch, tmp_path):
    target, calls = tmp_path / "commands.json", []

    def fsync(fd):
        calls.append(os.fstat(fd).st_mode)
        os.fsync(fd)

    monkeypatch.setattr(controller, "os", NS(fsync=fsync, replace=os.replace, O_RDONLY=os.O_RDONLY, O_DIRECTORY=os.O_DIRECTORY))
    value = [{"z": "é", "complete": True, "null": None, "argv": ["docker", "inspect"]}]
    controller.publish(target, value)
    expected = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with owned_io.opened_file(target, "rb") as stream:
        actual = stream.read(len(expected) + 1)
    assert actual == expected
    assert target.stat().st_mode & 0o777 == 0o600 and len(calls) == 2
    assert not target.with_suffix(".json.new").exists()


def test_directory_fsync_and_close_failures_retain_published_bytes_and_primary(monkeypatch, tmp_path):
    primary, secondary = OSError("directory-fsync"), OSError("directory-close")
    closes = []

    def close(fd):
        closes.append(fd)
        raise secondary

    monkeypatch.setattr(owned_io, "os", NS(open=lambda *args, **options: 1337, close=close))

    def fsync(fd):
        if fd == 1337:
            raise primary

    monkeypatch.setattr(controller, "os", NS(fsync=fsync, replace=os.replace, O_RDONLY=os.O_RDONLY, O_DIRECTORY=os.O_DIRECTORY))
    target = tmp_path / "commands.json"
    with pytest.raises(OSError) as caught:
        controller.publish(target, {"complete": False})
    assert caught.value is primary and secondary in roots(primary) and closes == [1337]
    # Avoid the fault-injected helper's input descriptors for this assertion.
    assert target.is_file() and target.stat().st_size == len(b'{\n  "complete": false\n}\n')
