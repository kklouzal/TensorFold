"""Immutable preparation publication and source option integration controls."""

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cli import build_parser
from tensorfold.serve_options import check
from tensorfold.families.qwen4_exp.cuda.sm70_weight_manifest import numerical_profile
from tensorfold.cuda.sm70.nvfp4_layout import LINEAR_REFERENCE


@pytest.fixture
def publisher():
    path = Path(__file__).resolve().parents[1] / "tools/prepare_sm70_nvfp4.py"
    spec = importlib.util.spec_from_file_location("sm70_manifest_publication_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepared_identity_is_reused_without_replacement_and_conflicting_identity_is_preserved(tmp_path, publisher):
    target = tmp_path / "family.json"
    body = {"source": "same", "version": 1}
    publisher.publish(target, body)
    first = target.stat()
    publisher.publish(target, {"version": 1, "source": "same"})
    assert target.stat().st_ino == first.st_ino and target.stat().st_mtime_ns == first.st_mtime_ns
    with pytest.raises(FileExistsError):
        publisher.publish(target, {"source": "other", "version": 1})
    assert json.loads(target.read_text()) == body
    assert list(tmp_path.iterdir()) == [target]


def test_growing_source_hash_is_bounded_to_opened_size_plus_one_byte(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import sm70_weight_manifest as authority

    path = tmp_path / "growing.bin"
    path.write_bytes(b"abc")
    original = authority._regular_stream
    calls, consumed = [], []
    class Growing:
        def __init__(self, stream):
            self.stream = stream
        def fileno(self):
            return self.stream.fileno()
        def readinto(self, value):
            calls.append(len(value))
            if len(calls) > 2:
                pytest.fail("preparation followed an indefinitely growing EOF")
            with path.open("ab") as extra:
                extra.write(b"x")
            count = self.stream.readinto(value)
            consumed.append(count)
            return count
        def read(self, size):
            assert size == 1
            result = self.stream.read(size)
            consumed.append(len(result))
            return result
    @contextmanager
    def opened(value):
        with original(value) as stream:
            yield Growing(stream)
    monkeypatch.setattr(authority, "_regular_stream", opened)
    with pytest.raises(ValueError, match="changed"):
        authority._digest(path)
    assert calls == [3] and sum(consumed) == 4


def test_failed_link_and_failed_temporary_retirement_preserve_primary_and_cleanup(tmp_path, publisher, monkeypatch):
    primary, cleanup = OSError("controlled link"), OSError("controlled temporary retirement")
    def link(*args):
        raise primary
    def unlink(self, *args, **kwargs):
        raise cleanup
    monkeypatch.setattr(publisher.os, "link", link)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(BaseException) as caught:
        publisher.publish(tmp_path / "family.json", {"source": "controlled"})
    # The shared cleanup transport retains original failure objects; neither
    # outcome becomes a success-shaped publication.
    pending, seen = [caught.value], set()
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        pending.extend((error.__cause__, error.__context__))
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
    assert id(primary) in seen and id(cleanup) in seen


@pytest.mark.parametrize("flags,reason", [([], "precision full"),
    (["--precision", "full"], "max-pending-requests"),
    (["--precision", "full", "--max-pending-requests", "3", "--max-http-connections", "4",
      "--indexer-history", "ram"], "compact GPU")])
def test_declared_source_options_refuse_unsupported_policy_before_engine_or_payload(tmp_path, flags, reason):
    declaration = tmp_path / "family.json"
    declaration.write_text(json.dumps({"numerical": numerical_profile(LINEAR_REFERENCE), "transforms": []}))
    args = build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--family-manifest",
                                      str(declaration), "--no-drafts", *flags])
    family = SimpleNamespace(model_type="qwen4_exp", title="controlled", package=SimpleNamespace())
    with pytest.raises(ValueError, match=reason):
        check(args, family, "cuda", tmp_path)
