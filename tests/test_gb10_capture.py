"""Owned capture boundaries and failure transport; no model or build execution."""
from __future__ import annotations

import io as buffer_io
import json
import os
import stat
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import gb10_capture as capture

io = sys.modules[capture.new_file.__module__]
REVISION = "a" * 40
TREE = "b" * 40


def exception_roots(error):
    found, pending = set(), [error]
    while pending:
        item = pending.pop()
        if id(item) in found:
            continue
        found.add(id(item))
        pending.extend(value for value in (
            BaseException.__cause__.__get__(item), BaseException.__context__.__get__(item)) if value is not None)
        if isinstance(item, BaseExceptionGroup):
            pending.extend(item.exceptions)
        if isinstance(item, io.RetirementFailure):
            pending.extend(value for value in (
                item.original_cause, item.original_context, item.cleanup, item.transport)
                           if isinstance(value, BaseException))
    return found


class HostilePrimary(RuntimeError):
    def __getattribute__(self, name):
        if name in {"__cause__", "__context__", "__suppress_context__"}:
            raise AssertionError("native error fields must bypass attribute hooks")
        return super().__getattribute__(name)


def test_descriptor_retirement_preserves_native_primary_links(monkeypatch):
    primary, cleanup, cause = HostilePrimary("operation"), OSError("close"), ValueError("prior")
    BaseException.__cause__.__set__(primary, cause)
    closed = []
    monkeypatch.setattr(io.os, "open", lambda *_args, **_kwargs: 91)

    def close(fd):
        closed.append(fd)
        raise cleanup

    monkeypatch.setattr(io.os, "close", close)
    with pytest.raises(HostilePrimary) as caught, io.owned_descriptor("fixture", 0):
        raise primary
    assert caught.value is primary and closed == [91]
    assert {id(primary), id(cleanup), id(cause)} <= exception_roots(primary)


def test_raw_output_preserves_write_and_close_failures(monkeypatch):
    primary, cleanup = RuntimeError("write"), OSError("close")
    closes = []

    class Output:
        def close(self):
            closes.append(True)
            raise cleanup

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: Output())
    with pytest.raises(RuntimeError) as caught, io.new_file(Path("fixture")):
        raise primary
    assert caught.value is primary and closes == [True]
    assert id(cleanup) in exception_roots(primary)


@pytest.mark.parametrize("target", ["raise_failures", "_raise_cleanup"])
def test_transport_exhaustion_keeps_primary_and_preallocated_retirement_state(monkeypatch, target):
    primary, cleanup, prior, prior_context, exhausted = (
        RuntimeError("body"), OSError("close"), ValueError("prior"),
        LookupError("prior context"), MemoryError("transport"))
    primary.__cause__ = prior
    primary.__context__ = prior_context
    primary.__suppress_context__ = True
    closes = []

    class Output:
        def close(self):
            closes.append(True)
            raise cleanup

    resource = Output()

    def fail_transport(*_args):
        raise exhausted

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: resource)
    monkeypatch.setattr(io, target, fail_transport)
    with pytest.raises(RuntimeError) as caught, io.new_file(Path("fixture")):
        raise primary
    assert caught.value is primary and closes == [True]
    state = BaseException.__cause__.__get__(primary)
    assert type(state) is io.RetirementFailure
    assert state.original_cause is prior and state.original_context is prior_context
    assert state.original_suppress_context is True
    assert state.cleanup is cleanup and state.transport is exhausted and state.resource is resource
    # contextlib may remove the generator frame; callers may discard all traces.
    # Ownership and prior roots must survive both, not rely on incidental locals.
    for error in (primary, state, prior, prior_context, cleanup, exhausted):
        BaseException.__traceback__.__set__(error, None)
    assert {id(prior), id(prior_context), id(cleanup), id(exhausted)} <= exception_roots(primary)
    assert state.resource is resource


@pytest.mark.parametrize("owner", ["descriptor", "file", "borrowed", "archive"])
def test_retirement_carrier_is_allocated_before_resource_acquisition(monkeypatch, owner):
    exhausted, acquisitions = MemoryError("carrier"), []

    def fail_carrier():
        raise exhausted

    def unexpected_acquisition(*_args, **_kwargs):
        acquisitions.append(True)
        raise AssertionError("resource acquired before failure state")

    monkeypatch.setattr(io, "RetirementFailure", fail_carrier)
    monkeypatch.setattr(io.os, "open", unexpected_acquisition)
    monkeypatch.setattr(io.os, "fdopen", unexpected_acquisition)
    monkeypatch.setattr(Path, "open", unexpected_acquisition)
    monkeypatch.setattr(io.tarfile, "open", unexpected_acquisition)
    factory = {"descriptor": lambda: io.owned_descriptor("fixture", 0),
               "file": lambda: io.new_file(Path("fixture")),
               "borrowed": lambda: io.borrowed_stream(91),
               "archive": lambda: io.archive("fixture", "r")}[owner]
    with pytest.raises(MemoryError) as caught, factory():
        raise AssertionError("owner entered after failed carrier allocation")
    assert caught.value is exhausted and acquisitions == []


def test_failed_normal_close_retains_its_stream_and_original_exception(monkeypatch):
    cleanup, closes = OSError("close"), []

    class Output:
        def close(self):
            closes.append(True)
            raise cleanup

    resource = Output()
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: resource)
    with pytest.raises(OSError) as caught, io.new_file(Path("fixture")):
        pass
    assert caught.value is cleanup and closes == [True]
    state = BaseException.__cause__.__get__(cleanup)
    assert type(state) is io.RetirementFailure and state.resource is resource
    assert state.cleanup is cleanup and state.transport is None


def test_write_all_handles_partial_writes_and_rejects_invalid_progress():
    class Partial:
        def __init__(self):
            self.output = bytearray()

        def write(self, value):
            count = min(3, len(value))
            self.output.extend(value[:count])
            return count

    stream = Partial()
    io.write_all(stream, b"partial-boundary")
    assert bytes(stream.output) == b"partial-boundary"


@pytest.mark.parametrize("progress", [None, False, 0, -1, 5])
def test_invalid_write_progress_is_explicit_failure(progress):
    with pytest.raises(OSError, match="invalid write progress"):
        io.write_all(SimpleNamespace(write=lambda _value: progress), b"four")


def test_archive_aborts_without_success_eof_and_preserves_both_errors(monkeypatch):
    primary, cleanup = RuntimeError("add"), OSError("abort")
    calls = []

    class Archive:
        def __exit__(self, kind, value, traceback):
            calls.append((kind, value, traceback))
            raise cleanup

        def close(self):
            raise AssertionError("failure must use abort rather than success close")

    monkeypatch.setattr(io.tarfile, "open", lambda *_args, **_kwargs: Archive())
    with pytest.raises(RuntimeError) as caught, io.archive("fixture", "x"):
        raise primary
    assert caught.value is primary and calls == [(RuntimeError, primary, None)]
    assert id(cleanup) in exception_roots(primary)


def tree_fixture(tmp_path, setgid=False):
    root = tmp_path / "input"
    root.mkdir(mode=0o750)
    root.chmod(0o2750 if setgid else 0o750)
    (root / "nested").mkdir(mode=0o711)
    (root / "nested").chmod(0o2711 if setgid else 0o711)
    (root / "nested" / "empty").mkdir(mode=0o550)
    (root / "nested" / "empty").chmod(0o2550 if setgid else 0o550)
    data = b"calibrated-build-input\n" * 48000
    (root / "nested" / "payload").write_bytes(data)
    (root / "nested" / "payload").chmod(0o2640 if setgid else 0o640)
    return root, data


@pytest.mark.parametrize("setgid", [False, True])
def test_context_copy_preserves_bytes_modes_manifest_and_archive_selection(tmp_path, setgid):
    root, data = tree_fixture(tmp_path, setgid)
    output = tmp_path / "capture"
    record = capture.capture_context("foundation_pins", root, output)
    expected_files = {"nested/payload": {"sha256": capture.hashlib.sha256(data).hexdigest(),
                                         "size": len(data),
                                         "mode": stat.S_IMODE((root / "nested/payload").stat().st_mode)}}
    assert record["files"] == expected_files
    expected_directories = {relative: stat.S_IMODE((root / relative).stat().st_mode)
                            for relative in ("", "nested", "nested/empty")}
    assert record["directories"] == expected_directories
    assert bool(expected_directories[""] & stat.S_ISGID) is setgid
    assert (output / "payload/nested/payload").read_bytes() == data
    assert stat.S_IMODE((output / "payload/nested").stat().st_mode) == expected_directories["nested"]
    assert record["archive_sha256"] == capture.hashlib.sha256((output / "context.tar").read_bytes()).hexdigest()
    with tarfile.open(output / "context.tar") as archive:
        entries = archive.getmembers()
        assert [entry.name for entry in entries] == [".", "nested", "nested/empty", "nested/payload"]
        assert [entry.mode for entry in entries] == [expected_directories[""], expected_directories["nested"],
                                                    expected_directories["nested/empty"],
                                                    expected_files["nested/payload"]["mode"]]
        assert all(entry.mtime == 0 and entry.uid == 0 and entry.gid == 0 for entry in entries)
        stream = archive.extractfile(entries[-1])
        assert stream is not None
        with stream:
            assert stream.read() == data
    assert record["context_uid"] == capture.digest({key: value for key, value in record.items() if key != "context_uid"})
    with pytest.raises(FileExistsError):
        capture.capture_context("foundation_pins", root, output)


@pytest.mark.parametrize("kind", ["link", "fifo"])
def test_named_context_rejects_links_and_specials_before_copy(tmp_path, kind):
    root = tmp_path / "input"
    root.mkdir()
    if kind == "link":
        (root / "entry").symlink_to(tmp_path)
    else:
        os.mkfifo(root / "entry")
    with pytest.raises(ValueError, match="not links or specials"):
        capture.capture_context("foundation_artifacts", root, tmp_path / "output")


def test_context_rejects_source_mutation_during_copy(tmp_path, monkeypatch):
    root, _data = tree_fixture(tmp_path)
    original = capture.write_all
    mutated = False

    def write_and_mutate(stream, value):
        nonlocal mutated
        original(stream, value)
        if not mutated:
            mutated = True
            (root / "nested/payload").write_bytes(b"changed")

    monkeypatch.setattr(capture, "write_all", write_and_mutate)
    with pytest.raises(ValueError, match="changed"):
        capture.capture_context("foundation_pins", root, tmp_path / "output")
    assert mutated


def test_member_adapter_fd_and_archive_failures_all_survive(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    (root / "payload").write_bytes(b"payload")
    primary, adapter_error, fd_error, archive_error = (
        RuntimeError("add"), OSError("adapter"), OSError("fd"), OSError("archive"))
    borrowed_fds = []
    real_close = os.close

    class Adapter:
        def close(self):
            raise adapter_error

    class Archive:
        def addfile(self, entry, _stream=None):
            if entry.isfile():
                raise primary

        def __exit__(self, *_args):
            raise archive_error

        def close(self):
            raise AssertionError("abort path required")

    def borrow(fd, *_args, **_kwargs):
        borrowed_fds.append(fd)
        return Adapter()

    def close(fd):
        if fd in borrowed_fds:
            raise fd_error
        real_close(fd)

    monkeypatch.setattr(io.os, "fdopen", borrow)
    monkeypatch.setattr(io.os, "close", close)
    monkeypatch.setattr(io.tarfile, "open", lambda *_args, **_kwargs: Archive())
    try:
        with pytest.raises(RuntimeError) as caught:
            capture.capture_context("foundation_pins", root, tmp_path / "output")
        assert caught.value is primary
        assert {id(primary), id(adapter_error), id(fd_error), id(archive_error)} <= exception_roots(primary)
    finally:
        for fd in borrowed_fds:
            real_close(fd)


def fixture_archive(newline=b"\n"):
    entries = {
        "deploy/gb10/Dockerfile": b"ARG BASE_IMAGE=example/cuda@sha256:" + b"0" * 64 + newline,
        "pyproject.toml": b"[project]\nname='fixture'\n",
        "setup.py": b"# fixture\n",
        "src/tensorfold/_fd_owner.c": b"/* fixture */\n",
    }
    stream = buffer_io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in sorted(entries.items()):
            entry = tarfile.TarInfo(name)
            entry.mode, entry.size, entry.mtime = 0o644, len(data), 0
            archive.addfile(entry, buffer_io.BytesIO(data))
    return stream.getvalue(), entries


def setup_capture(tmp_path, monkeypatch, newline=b"\n"):
    repo = tmp_path / "repo"
    repo.mkdir()
    roots = {}
    for name in capture.NAMES:
        roots[name] = tmp_path / name
        roots[name].mkdir()
        (roots[name] / "input").write_bytes(name.encode())
    archive, entries = fixture_archive(newline)

    def git(_repo, *arguments):
        if arguments == ("status", "--porcelain"):
            return ""
        if arguments == ("rev-parse", REVISION + "^{tree}"):
            return TREE
        if arguments in (("rev-parse", "HEAD"), ("rev-parse", REVISION + "^{commit}")):
            return REVISION
        raise AssertionError(arguments)

    def run(argv, *, stdout, check):
        assert argv == ["git", "-C", str(repo.resolve()), "archive", "--format=tar", REVISION]
        assert check is True
        io.write_all(stdout, archive)

    monkeypatch.setattr(capture, "git", git)
    monkeypatch.setattr(capture.subprocess, "run", run)
    return repo, roots, entries


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_capture_preserves_schema_identity_and_path_independence(tmp_path, monkeypatch, newline):
    repo, roots, entries = setup_capture(tmp_path, monkeypatch, newline)
    values = []
    for output in (tmp_path / "first", tmp_path / "second"):
        result = capture.capture(repo, REVISION, output, architecture="arm64",
                                 foundation_pins=roots["foundation_pins"],
                                 foundation_artifacts=roots["foundation_artifacts"])
        status = json.loads((output / "capture-status.json").read_bytes())
        assert status["complete"] is True and status["revision"] == REVISION
        assert result == json.loads((output / "source.json").read_bytes())
        assert result["schema"] == 4 and result["tree"] == TREE
        assert result["architecture"] == "arm64" and result["platform"] == "linux/arm64"
        assert result["native_machine"] == "aarch64"
        assert result["files"] == {name: capture.hashlib.sha256(data).hexdigest() for name, data in entries.items()}
        assert result["modes"] == dict.fromkeys(entries, 0o644) and result["symlinks"] == {}
        assert set(result["named_contexts"]) == set(capture.NAMES)
        values.append(result)
    assert values[0] == values[1]


@pytest.mark.parametrize("change", ["dirty_before", "wrong_head", "dirty_after"])
def test_capture_requires_the_explicit_clean_commit_through_publication(tmp_path, monkeypatch, change):
    repo, roots, _entries = setup_capture(tmp_path, monkeypatch)
    original = capture.git
    statuses = 0

    def changed_git(repo, *arguments):
        nonlocal statuses
        if arguments == ("rev-parse", "HEAD") and change == "wrong_head":
            return "c" * 40
        if arguments == ("status", "--porcelain"):
            statuses += 1
            if change == "dirty_before" or (change == "dirty_after" and statuses == 2):
                return " M source"
        return original(repo, *arguments)

    monkeypatch.setattr(capture, "git", changed_git)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="clean worktree|changed during capture"):
        capture.capture(repo, REVISION, output, architecture="arm64",
                        foundation_pins=roots["foundation_pins"], foundation_artifacts=roots["foundation_artifacts"])
    assert not (output / "source.json").exists()
    if change == "dirty_after":
        assert json.loads((output / "capture-status.json").read_bytes())["complete"] is False
    else:
        assert not output.exists()


def test_capture_failure_and_failed_status_publication_keep_original_root(tmp_path, monkeypatch):
    repo, roots, _entries = setup_capture(tmp_path, monkeypatch)
    primary, diagnostic = RuntimeError("git archive"), OSError("status")

    def fail_archive(_argv, *, stdout, check):
        io.write_all(stdout, b"partial")
        raise primary

    def fail_status(path, value):
        assert path.name == "capture-status.json" and value["complete"] is False
        raise diagnostic

    monkeypatch.setattr(capture.subprocess, "run", fail_archive)
    monkeypatch.setattr(capture, "publish_document", fail_status)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError) as caught:
        capture.capture(repo, REVISION, output, architecture="arm64",
                        foundation_pins=roots["foundation_pins"], foundation_artifacts=roots["foundation_artifacts"])
    assert caught.value is primary and id(diagnostic) in exception_roots(primary)
    assert (output / "source.tar").read_bytes() == b"partial"
    assert not (output / "source.json").exists()


def test_status_is_not_published_if_its_output_close_fails(tmp_path, monkeypatch):
    repo, roots, _entries = setup_capture(tmp_path, monkeypatch)
    original_open = Path.open
    cleanup = OSError("status close")

    class ClosingFailure:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def write(self, value):
            return self.wrapped.write(value)

        def close(self):
            self.wrapped.close()
            raise cleanup

    def open_file(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        return ClosingFailure(stream) if path.name == "capture-status.json.new" and mode == "xb" else stream

    monkeypatch.setattr(Path, "open", open_file)
    output = tmp_path / "failed"
    with pytest.raises(OSError) as caught:
        capture.capture(repo, REVISION, output, architecture="arm64",
                        foundation_pins=roots["foundation_pins"], foundation_artifacts=roots["foundation_artifacts"])
    assert caught.value is cleanup
    assert not (output / "capture-status.json").exists()
    assert json.loads((output / "capture-status.json.new").read_bytes())["complete"] is True


def test_build_identity_keeps_platform_and_complete_named_context_contract(monkeypatch):
    observed = []
    monkeypatch.setattr(capture, "digest", lambda body: observed.append(body) or "identity")
    source = {"architecture": "amd64", "platform": "linux/amd64", "native_machine": "x86_64",
              "source_uid": "s", "archive_sha256": "a",
              "named_contexts": {"foundation_pins": {"context_uid": "p"},
                                 "foundation_artifacts": {"context_uid": "f"}}}
    assert capture.build_input_identity(source) == "identity"
    assert observed == [{"schema": 2, "platform": "linux/amd64", "source_uid": "s",
                         "source_archive_sha256": "a",
                         "named_contexts": {"foundation_artifacts": "f", "foundation_pins": "p"}}]
    with pytest.raises(ValueError, match="target.*differs"):
        capture.build_input_identity(dict(source, native_machine="aarch64"))
    with pytest.raises(ValueError, match="exactly"):
        capture.build_input_identity(dict(source, named_contexts={"foundation_pins": {"context_uid": "p"}}))
