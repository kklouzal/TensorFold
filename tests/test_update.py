"""Version tracking and `tensorfold update`, with GitHub and pip replaced by fakes (no network, no installs)."""

import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold import __version__, cli, update


SAMPLE = """# What's new in TensorFold

## 99.0.0 (1 Jan 2099)

- Faster everything.

## 0.3.5.1 (28 Sep 2026)

- The M1 and M2 fix.
"""


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "CACHE", tmp_path / "update-check.json")
    monkeypatch.setattr(update, "SEEN", tmp_path / "version-seen")
    monkeypatch.setattr("urllib.request.urlopen", _offline)


def _offline(*args, **kwargs):
    raise OSError("offline")


def _no_network(*args, **kwargs):
    raise AssertionError("the network was used")


def test_versions_compare_numerically():
    assert update.parse_version("v0.3.10") == (0, 3, 10)
    assert update.newer("v0.3.10", "0.3.9") and not update.newer("v0.3.1", "0.3.1")
    assert not update.newer("nightly", "0.3.1")


def test_latest_release_reads_github_once_a_day(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(json.dumps({"tag_name": "v9.9.9"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert update.latest_release() == "v9.9.9"
    monkeypatch.setattr("urllib.request.urlopen", _no_network)
    assert update.latest_release() == "v9.9.9"             # from the day's cache
    assert calls == ["https://api.github.com/repos/kklouzal/TensorFold/releases/latest"]
    assert json.loads(update.CACHE.read_text())["repo"] == "kklouzal/TensorFold"


def test_offline_is_silent(monkeypatch):
    def urlopen(request, timeout):
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert update.latest_release() is None
    assert update.notice(None) is None


def test_notice_only_for_a_newer_release():
    assert "run `tensorfold update`" in update.notice("v99.0.0")
    assert update.notice(f"v{__version__}") is None


def test_background_check_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(update, "latest_release", _no_network)
    assert update.check_in_background() is None


def test_background_check_prints_the_notice(monkeypatch, capsys):
    monkeypatch.delenv("TENSORFOLD_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(update, "latest_release", lambda: "v99.0.0")
    update.check_in_background().join(5)
    assert "TensorFold 99.0.0 is available" in capsys.readouterr().out


def test_update_installs_the_latest_tag_with_this_python(monkeypatch):
    commands = []
    monkeypatch.setattr(update, "latest_release", lambda **kwargs: "v99.0.0")
    monkeypatch.setattr(update, "_editable_clone", lambda: None)
    monkeypatch.setattr(update.subprocess, "call", lambda command, **kw: commands.append(command) or 0)
    monkeypatch.setattr(update.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="99.0.0\n", returncode=0))
    assert update.update() == 0
    assert commands == [[sys.executable, "-m", "pip", "install", "--upgrade",
                         "git+https://github.com/kklouzal/TensorFold.git@v99.0.0"]]


def test_check_only_and_current_install_nothing(monkeypatch, capsys):
    monkeypatch.setattr(update.subprocess, "call", _no_network)
    monkeypatch.setattr(update, "latest_release", lambda **kwargs: "v99.0.0")
    assert update.update(check_only=True) == 0
    monkeypatch.setattr(update, "latest_release", lambda **kwargs: f"v{__version__}")
    assert update.update() == 0
    assert "is the latest release" in capsys.readouterr().out


def test_editable_clone_fast_forwards_only_when_clean(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(update, "latest_release", lambda **kwargs: "v99.0.0")
    monkeypatch.setattr(update, "_editable_clone", lambda: tmp_path)
    monkeypatch.setattr(update.subprocess, "call", lambda command, **kw: commands.append(command) or 0)
    import uuid

    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    private = "refs/tensorfold/update/" + "a" * 32
    base, raw, commit = "1" * 40, "2" * 40, "3" * 40

    def result(command, **kw):
        if command[-1] == "--porcelain":
            return SimpleNamespace(stdout="", returncode=0)
        return SimpleNamespace(stdout={"HEAD^{commit}": base, private: raw, raw + "^{commit}": commit}[command[-1]],
                               returncode=0)

    monkeypatch.setattr(update.subprocess, "run", result)
    assert update.update() == 0
    assert commands == [["git", "-C", str(tmp_path), "update-ref", "--no-deref", private, base, ""],
                        ["git", "-C", str(tmp_path), "fetch", "--no-tags",
                         "https://github.com/kklouzal/TensorFold.git", "refs/tags/v99.0.0:" + private],
                        ["git", "-C", str(tmp_path), "merge", "--ff-only", commit],
                        ["git", "-C", str(tmp_path), "update-ref", "--no-deref", "-d", private, raw]]
    before = list(commands)
    monkeypatch.setattr(update.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=" M file.py\n", returncode=0))
    assert update.update() == 1                            # local changes: left alone
    assert commands == before


def test_the_cli_has_update_and_the_serve_switch(monkeypatch):
    monkeypatch.setattr(update, "latest_release", lambda **kwargs: f"v{__version__}")
    assert cli.main(["update", "--check"]) == 0
    assert cli.build_parser().parse_args(["serve", "owner/model", "--no-update-check"]).no_update_check


def test_whats_new_keeps_the_versions_after_this_one():
    notes = update.whats_new(SAMPLE, "0.3.5.1", "v99.0.0")
    assert notes.startswith("## 99.0.0") and "Faster everything" in notes and "M1 and M2" not in notes
    assert update.whats_new(SAMPLE, "0.3.5", "v0.3.5.1").startswith("## 0.3.5.1")
    assert update.whats_new(SAMPLE, "99.0.0", "v99.0.0") == ""


def test_update_shows_whats_new_from_the_new_tag(monkeypatch, capsys):
    fetched = []

    def urlopen(request, timeout):
        fetched.append(request.full_url)
        return io.BytesIO(SAMPLE.encode())

    monkeypatch.setattr(update, "latest_release", lambda **kwargs: "v99.0.0")
    monkeypatch.setattr(update, "_editable_clone", lambda: None)
    monkeypatch.setattr(update.subprocess, "call", lambda command, **kw: 0)
    monkeypatch.setattr(update.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="99.0.0\n", returncode=0))
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert update.update() == 0
    out = capsys.readouterr().out
    assert f"What's new since {__version__}" in out and "Faster everything" in out
    assert fetched == ["https://raw.githubusercontent.com/kklouzal/TensorFold/v99.0.0/CHANGELOG.md"]
    assert update.SEEN.read_text() == "99.0.0"             # the first run after it stays quiet


def test_an_editable_clone_reads_its_own_changelog(monkeypatch, tmp_path, capsys):
    (tmp_path / "CHANGELOG.md").write_text(SAMPLE)
    monkeypatch.setattr(update, "latest_release", lambda **kwargs: "v99.0.0")
    monkeypatch.setattr(update, "_editable_clone", lambda: tmp_path)
    monkeypatch.setattr(update.subprocess, "call", lambda command, **kw: 0)
    monkeypatch.setattr(update.subprocess, "run", lambda command, **k: SimpleNamespace(
        stdout="" if command[-1] == "--porcelain" else "a" * 40, returncode=0))
    monkeypatch.setattr("urllib.request.urlopen", _no_network)
    assert update.update() == 0
    assert "Faster everything" in capsys.readouterr().out


def test_offline_the_notes_link_still_prints(capsys):
    update.show_whats_new("0.3.5.1", "v99.0.0")
    out = capsys.readouterr().out
    assert "What's new" not in out and "blob/v99.0.0/CHANGELOG.md" in out


def test_the_first_run_of_a_version_points_at_its_notes_once(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_NO_UPDATE_CHECK", raising=False)
    assert f"TensorFold {__version__}" in update.first_run_notice()
    assert update.first_run_notice() is None
    update.SEEN.write_text("0.0.1")
    assert "what's new" in update.first_run_notice()       # updated some other way
    update.SEEN.write_text("99.0.0")
    assert update.first_run_notice() is None               # a downgrade
    monkeypatch.setenv("TENSORFOLD_NO_UPDATE_CHECK", "1")
    update.SEEN.write_text("0.0.1")
    assert update.first_run_notice() is None


def test_the_changelog_has_this_version():
    text = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
    assert re.search(rf"^## {re.escape(__version__)} ", text, re.M), f"CHANGELOG.md needs a ## {__version__} section"


@pytest.mark.parametrize("repo", [None, "ashhart/TensorFold", "another/TensorFold"])
def test_legacy_or_foreign_release_cache_never_advertises_a_native_update(monkeypatch, repo):
    cached = {"tag": "v1.0.7", "checked": time.time()}
    if repo is not None:
        cached["repo"] = repo
    update.CACHE.write_text(json.dumps(cached))
    calls = []

    def owned_release(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(b'{"tag_name":"v0.6.6"}')

    monkeypatch.setattr("urllib.request.urlopen", owned_release)
    assert update.latest_release() == "v0.6.6"
    assert calls == ["https://api.github.com/repos/kklouzal/TensorFold/releases/latest"]
    assert json.loads(update.CACHE.read_text())["repo"] == "kklouzal/TensorFold"


@pytest.mark.parametrize("tag", [
    None, True, 7, [], {}, "main", "python-0.6", "--upload-pack=other", "v1.0@{1}", "v1.0^{commit}",
    "v1.0..v2.0", "refs/tags/v1.0", "v1.0:branch", "v1.0#main", "v1.0?query", "v1.0.lock", "v1.0\n",
])
def test_invalid_release_metadata_cannot_reach_git_or_pip(monkeypatch, tag, capsys):
    calls = []

    def response(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(json.dumps({"tag_name": tag}).encode())

    monkeypatch.setattr("urllib.request.urlopen", response)
    monkeypatch.setattr(update, "_editable_clone", _no_network)
    monkeypatch.setattr(update.subprocess, "call", _no_network)
    assert update.update(force=True) == 1
    assert calls == ["https://api.github.com/repos/kklouzal/TensorFold/releases/latest"]
    assert "could not resolve a published release for kklouzal/TensorFold" in capsys.readouterr().err
    assert not update.CACHE.exists()


@pytest.mark.parametrize("tag", ["v0.6.6", "0.6.6", "v0.3.5.1", "v0.6.6-rc.1", "v0.6.6+build.7"])
def test_supported_version_tags_keep_their_exact_release_identity(monkeypatch, tag):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(json.dumps({"tag_name": tag}).encode()))
    assert update.latest_release() == tag


@pytest.mark.parametrize("cached", [[], {}, {"repo": "kklouzal/TensorFold", "tag": "v1.0#main", "checked": 0}])
def test_invalid_cache_offline_stays_silent_without_a_branch_fallback(cached):
    update.CACHE.write_text(json.dumps(cached))
    assert update.latest_release() is None
    assert update.notice(None) is None


def test_no_published_release_does_not_install_or_fetch_a_branch(monkeypatch, capsys):
    from urllib.error import HTTPError

    def missing(request, timeout):
        assert request.full_url == "https://api.github.com/repos/kklouzal/TensorFold/releases/latest"
        raise HTTPError(request.full_url, 404, "No published releases", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", missing)
    monkeypatch.setattr(update.subprocess, "call", _no_network)
    assert update.update() == 1
    assert "could not resolve a published release" in capsys.readouterr().err


def test_invalid_changelog_ref_is_refused_before_network_or_clone_read(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", _no_network)
    with pytest.raises(ValueError, match="release tag"):
        update.changelog("v1.0#main")
    with pytest.raises(ValueError, match="release tag"):
        update.changelog("v1.0^{commit}", tmp_path)
    assert update.notice("v99.0.0#main") is None


def test_editable_update_uses_owned_release_despite_foreign_origin_and_local_tag(tmp_path, monkeypatch):
    """Real isolated Git history proves source ownership beyond mocked command strings."""

    def git(*args):
        return subprocess.run(["git", *map(str, args)], text=True, capture_output=True, check=True).stdout.strip()

    def identity(repo):
        git("-C", repo, "config", "user.name", "TensorFold Test")
        git("-C", repo, "config", "user.email", "tensorfold-test@example.invalid")
        git("-C", repo, "config", "commit.gpgsign", "false")
        git("-C", repo, "config", "tag.gpgsign", "false")

    owned, foreign, clone = (tmp_path / name for name in ("owned", "foreign", "editable"))
    git("init", "--initial-branch=python-0.6", owned)
    identity(owned)
    (owned / "implementation.txt").write_text("Python baseline")
    (owned / "CHANGELOG.md").write_text(SAMPLE)
    git("-C", owned, "add", "implementation.txt", "CHANGELOG.md")
    git("-C", owned, "commit", "-m", "Python baseline")
    git("clone", owned, clone)
    git("clone", owned, foreign)
    identity(foreign)
    git("-C", foreign, "switch", "-c", "main")
    (foreign / "implementation.txt").write_text("Foreign native implementation")
    git("-C", foreign, "commit", "-am", "Foreign native release")
    git("-C", foreign, "tag", "v99.0.0")
    foreign_commit = git("-C", foreign, "rev-parse", "HEAD")
    git("-C", clone, "remote", "set-url", "origin", foreign)
    git("-C", clone, "fetch", "--tags", "origin")
    (owned / "implementation.txt").write_text("Owned Python release")
    git("-C", owned, "commit", "-am", "Owned Python release")
    git("-C", owned, "tag", "-a", "v99.0.0", "-m", "Owned Python release")
    owned_commit = git("-C", owned, "rev-parse", "HEAD")
    monkeypatch.setattr(update, "REPO_URL", str(owned))
    monkeypatch.setattr(update, "_editable_clone", lambda: clone)
    monkeypatch.setattr(update, "latest_release", lambda **_: "v99.0.0")
    original_run = update._run
    interleaved = []

    def competing_fetch(command, **kw):
        if "merge" in command:
            git("-C", clone, "fetch", "--no-tags", "origin", "refs/heads/main")
            assert git("-C", clone, "rev-parse", "FETCH_HEAD") == foreign_commit
            interleaved.append(command[-1])
        return original_run(command, **kw)

    monkeypatch.setattr(update, "_run", competing_fetch)
    assert update.update() == 0
    assert interleaved == [owned_commit]
    assert git("-C", clone, "rev-parse", "HEAD") == owned_commit
    assert git("-C", clone, "for-each-ref", "--format=%(refname)", "refs/tensorfold/update") == ""
    assert (clone / "implementation.txt").read_text() == "Owned Python release"
    assert git("-C", clone, "rev-parse", "refs/tags/v99.0.0") == foreign_commit
    assert git("-C", clone, "remote", "get-url", "origin") == str(foreign)


@pytest.mark.parametrize("fetch_code, merge_code, cleanup_code, expected", [
    (0, 0, 0, 0), (1, 0, 0, 1), (0, 7, 0, 7), (0, 0, 1, 1), (0, 7, 1, 7),
])
def test_private_updater_ref_retirement_preserves_operation_failures(monkeypatch, tmp_path, capsys,
                                                                    fetch_code, merge_code, cleanup_code, expected):
    base, raw, commit = "1" * 40, "2" * 40, "3" * 40
    calls, notes = [], []

    def oid(clone, expression):
        return base if expression == "HEAD^{commit}" else commit if expression.endswith("^{commit}") else raw

    def run(command, **kw):
        calls.append(command)
        if "fetch" in command:
            return fetch_code
        if "merge" in command:
            return merge_code
        return cleanup_code if "-d" in command else 0

    monkeypatch.setattr(update, "_git_oid", oid)
    monkeypatch.setattr(update, "_run", run)
    monkeypatch.setattr(update, "show_whats_new", lambda *a: notes.append(a))
    assert update._update_editable(tmp_path, "v99.0.0") == expected
    created, deleted = calls[0], calls[-1]
    assert created[-2:] == [base, ""]
    assert "--no-deref" in created and "--no-deref" in deleted and "-d" in deleted
    assert deleted[-2:] == [created[-3], raw]
    assert bool(notes) is (fetch_code == merge_code == cleanup_code == 0)
    if cleanup_code:
        assert "retained for inspection" in capsys.readouterr().err


def test_private_ref_reservation_failure_cannot_fetch_merge_or_delete_foreign_state(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(update, "_git_oid", lambda *a: "a" * 40)
    monkeypatch.setattr(update, "_run", lambda command, **kw: calls.append(command) or 1)
    assert update._update_editable(tmp_path, "v99.0.0") == 1
    assert len(calls) == 1
    assert calls[0][3:5] == ["update-ref", "--no-deref"]
    assert calls[0][-1] == ""


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_fetch_interruption_retains_primary_and_reports_private_ref_cleanup_failure(monkeypatch, tmp_path, cleanup_fails):
    primary = KeyboardInterrupt("fetch interrupted")
    calls = []
    monkeypatch.setattr(update, "_git_oid", lambda *a: "a" * 40)

    def run(command, **kw):
        calls.append(command)
        if "fetch" in command:
            raise primary
        return 1 if "-d" in command and cleanup_fails else 0

    monkeypatch.setattr(update, "_run", run)
    with pytest.raises(KeyboardInterrupt) as caught:
        update._update_editable(tmp_path, "v99.0.0")
    assert caught.value is primary
    assert calls[-1][-3:] == ["-d", calls[0][-3], "a" * 40]
    if cleanup_fails:
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert "retained for inspection" in str(caught.value.__cause__)


@pytest.mark.parametrize("output", ["", "HEAD", "--force", "a" * 39, "b" * 41, "a" * 40 + "\n" + "b" * 40])
def test_invalid_git_object_output_cannot_select_a_revision(monkeypatch, tmp_path, output):
    monkeypatch.setattr(update.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=output, returncode=0))
    monkeypatch.setattr(update, "_run", _no_network)
    assert update._update_editable(tmp_path, "v99.0.0") == 1


def test_interrupted_private_ref_claim_reports_ambiguous_ownership_without_deleting_it(monkeypatch, tmp_path, capsys):
    primary = KeyboardInterrupt("claim acknowledgement interrupted")
    calls, external = [], {}
    monkeypatch.setattr(update, "_git_oid", lambda *a: "a" * 40)

    def claim_then_interrupt(command, **kw):
        calls.append(command)
        external[command[-3]] = command[-2]  # the external mutation completed before acknowledgement was lost
        raise primary

    monkeypatch.setattr(update, "_run", claim_then_interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        update._update_editable(tmp_path, "v99.0.0")
    assert caught.value is primary
    assert len(calls) == 1 and external == {calls[0][-3]: "a" * 40}
    assert "claim outcome is unknown" in capsys.readouterr().err
    assert calls[0][-3] in str(caught.value.__cause__)


@pytest.mark.parametrize("raised_cleanup", [False, True])
def test_cleanup_failure_preserves_existing_primary_cause_and_context(monkeypatch, tmp_path, raised_cleanup):
    primary = KeyboardInterrupt("fetch failure")
    prior_cause, prior_context = ValueError("original cause"), OSError("original context")
    cleanup_error = OSError("ref delete failed")
    BaseException.__cause__.__set__(primary, prior_cause)
    BaseException.__context__.__set__(primary, prior_context)
    monkeypatch.setattr(update, "_git_oid", lambda *a: "a" * 40)

    def run(command, **kw):
        if "fetch" in command:
            raise primary
        if "-d" in command:
            BaseException.__cause__.__set__(primary, TypeError("foreign cause overwrite"))
            BaseException.__context__.__set__(primary, RuntimeError("foreign context overwrite"))
            if raised_cleanup:
                raise cleanup_error
            return 1
        return 0

    monkeypatch.setattr(update, "_run", run)
    with pytest.raises(KeyboardInterrupt) as caught:
        update._update_editable(tmp_path, "v99.0.0")
    assert caught.value is primary
    pending, retained = [caught.value.__cause__], []
    while pending:
        error = pending.pop()
        if error is None or any(error is item for item in retained):
            continue
        retained.append(error)
        pending.extend((error.__cause__, error.__context__))
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
    assert any(error is prior_cause for error in retained)
    assert any(error is prior_context for error in retained)
    if raised_cleanup:
        assert any(error is cleanup_error for error in retained)
    assert any(isinstance(error, RuntimeError) and "retained for inspection" in str(error) for error in retained)


@pytest.mark.parametrize("stage", ["claim", "cleanup"])
def test_diagnostic_output_failure_cannot_replace_primary_or_owned_ref_context(monkeypatch, tmp_path, stage):
    primary = KeyboardInterrupt("owned updater interruption")
    prior_cause, prior_context = ValueError("original cause"), OSError("original context")
    diagnostic = OSError("stderr unavailable")
    BaseException.__cause__.__set__(primary, prior_cause)
    BaseException.__context__.__set__(primary, prior_context)
    calls = []
    monkeypatch.setattr(update, "_git_oid", lambda *a: "a" * 40)

    def run(command, **kw):
        calls.append(command)
        if stage == "claim" and len(calls) == 1 or stage == "cleanup" and "fetch" in command:
            raise primary
        return 1 if "-d" in command else 0

    class FailedDiagnostics:
        def write(self, text):
            raise diagnostic

        def flush(self):
            pass

    monkeypatch.setattr(update, "_run", run)
    with monkeypatch.context() as scoped:
        scoped.setattr(update.sys, "stderr", FailedDiagnostics())
        with pytest.raises(KeyboardInterrupt) as caught:
            update._update_editable(tmp_path, "v99.0.0")
    assert caught.value is primary
    pending, retained = [caught.value.__cause__], []
    while pending:
        error = pending.pop()
        if error is None or any(error is item for item in retained):
            continue
        retained.append(error)
        pending.extend((error.__cause__, error.__context__))
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
    assert all(any(error is target for error in retained) for target in (prior_cause, prior_context, diagnostic))
    phrase = "claim outcome is unknown" if stage == "claim" else "cleanup failed"
    assert any(isinstance(error, RuntimeError) and phrase in str(error) for error in retained)
    assert (len(calls) == 1) if stage == "claim" else "-d" in calls[-1]


def test_diagnostic_sink_cannot_overwrite_retained_cleanup_native_fields(monkeypatch, tmp_path):
    primary = KeyboardInterrupt("fetch interrupted")
    cleanup = OSError("private ref delete failed")
    cleanup_cause = ValueError("cleanup provider cause")
    diagnostic = OSError("diagnostic sink failed")
    BaseException.__cause__.__set__(cleanup, cleanup_cause)
    monkeypatch.setattr(update, "_git_oid", lambda *a: "a" * 40)

    def run(command, **kw):
        if "fetch" in command:
            raise primary
        if "-d" in command:
            raise cleanup
        return 0

    class MutatingDiagnostics:
        def write(self, text):
            BaseException.__cause__.__set__(cleanup, TypeError("foreign overwritten cleanup cause"))
            BaseException.__context__.__set__(cleanup, RuntimeError("foreign overwritten cleanup context"))
            BaseException.__suppress_context__.__set__(cleanup, False)
            raise diagnostic

        def flush(self):
            pass

    monkeypatch.setattr(update, "_run", run)
    with monkeypatch.context() as scoped:
        scoped.setattr(update.sys, "stderr", MutatingDiagnostics())
        with pytest.raises(KeyboardInterrupt) as caught:
            update._update_editable(tmp_path, "v99.0.0")
    assert caught.value is primary
    assert cleanup.__cause__ is cleanup_cause
    assert cleanup.__context__ is primary
    assert cleanup.__suppress_context__ is True
