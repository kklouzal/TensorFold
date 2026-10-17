"""Check releases without blocking the server; update with pip or fast-forward a clean editable clone."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from tensorfold import __version__

REPO = "kklouzal/TensorFold"
RELEASES_API = f"https://api.github.com/repos/{REPO}/releases/latest"
REPO_URL = f"https://github.com/{REPO}.git"
CACHE = Path.home() / ".cache" / "tensorfold" / "update-check.json"
CACHE_SECONDS = 24 * 3600
CHANGELOG = "CHANGELOG.md"
SEEN = Path.home() / ".cache" / "tensorfold" / "version-seen"    # the last version whose news was shown


def parse_version(text: str) -> tuple[int, ...]:
    """``v0.3.1`` or ``0.3.1`` -> (0, 3, 1); anything after the numbers is ignored."""

    match = re.match(r"v?(\d+(?:\.\d+)*)", text.strip())
    if not match:
        raise ValueError(f"not a version: {text!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def newer(latest: str, current: str = __version__) -> bool:
    try:
        return parse_version(latest) > parse_version(current)
    except ValueError:
        return False


def _release_tag(value: Any) -> str:
    """A version-prefixed single Git ref component, also safe inside a pip VCS URL."""

    if (not isinstance(value, str)
            or re.fullmatch(r"v?[0-9]+(?:\.[0-9]+)*(?:[A-Za-z_+\-][A-Za-z0-9_+\-]*(?:\.[A-Za-z0-9_+\-]+)*)?", value) is None
            or value.endswith(".lock")):
        raise ValueError("release tag must be a version without Git or URL expressions")
    parse_version(value)
    return value


def latest_release(timeout: float = 3.0, *, use_cache: bool = True) -> str | None:
    """The owned fork's release tag, from its day's cache; None without a resolvable published release."""

    if use_cache:
        try:
            cached = json.loads(CACHE.read_text())
            if (isinstance(cached, dict) and cached.get("repo") == REPO
                    and 0 <= time.time() - float(cached["checked"]) < CACHE_SECONDS):
                return _release_tag(cached["tag"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    import urllib.request

    request = urllib.request.Request(RELEASES_API, headers={"Accept": "application/vnd.github+json",
                                                            "User-Agent": f"tensorfold/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            tag = _release_tag(json.loads(response.read())["tag_name"])
    except Exception:                  # offline, rate-limited or blocked: no answer is fine
        return None
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"repo": REPO, "tag": tag, "checked": time.time()}))
    except OSError:
        pass
    return tag


def notice(tag: str | None) -> str | None:
    try:
        tag = _release_tag(tag)
    except ValueError:
        return None
    if newer(tag):
        return (f"[tensorfold] TensorFold {tag.lstrip('v')} is available (this is {__version__}): run "
                f"`tensorfold update`, then restart the server")
    return None


def _checks_off() -> bool:
    return os.environ.get("TENSORFOLD_NO_UPDATE_CHECK", "").strip().lower() in ("1", "true", "yes", "on")


def whats_new(text: str, since: str, upto: str) -> str:
    """The changelog's sections for versions after ``since`` up to ``upto``, as written (newest first)."""

    lines, keep = [], False
    for line in text.splitlines():
        heading = re.match(r"##\s+v?(\d+(?:\.\d+)+)", line)
        if heading:
            keep = parse_version(since) < parse_version(heading.group(1)) <= parse_version(upto)
        elif line.startswith("# "):
            keep = False
        if keep:
            lines.append(line)
    return "\n".join(lines).strip()


def changelog(tag: str, clone: Path | None = None) -> str | None:
    """CHANGELOG.md at ``tag``: the clone's copy when there is one, else GitHub's; None when unreachable."""

    tag = _release_tag(tag)
    if clone is not None:
        try:
            return (clone / CHANGELOG).read_text()
        except OSError:
            return None
    import urllib.request

    request = urllib.request.Request(f"https://raw.githubusercontent.com/{REPO}/{tag}/{CHANGELOG}",
                                     headers={"User-Agent": f"tensorfold/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read().decode("utf-8", "replace")
    except Exception:                  # offline or blocked: the link printed after it still works
        return None


def _remember(version: str) -> None:
    try:
        SEEN.parent.mkdir(parents=True, exist_ok=True)
        SEEN.write_text(version)
    except OSError:
        pass


def show_whats_new(since: str, tag: str, clone: Path | None = None) -> None:
    """Print what changed between ``since`` and ``tag``, then where every release's notes live."""

    text = changelog(tag, clone)
    notes = whats_new(text, since, tag) if text else ""
    if notes:
        print(f"\nWhat's new since {since}:\n\n{notes}\n", flush=True)
    print(f"[tensorfold] every release's notes: https://github.com/{REPO}/blob/{tag}/{CHANGELOG}", flush=True)
    _remember(tag.lstrip("v"))


def first_run_notice() -> str | None:
    """One line the first time a version runs, so installs updated some other way still hear what's new."""

    if _checks_off():
        return None
    try:
        seen = SEEN.read_text().strip()
    except OSError:
        seen = ""
    if seen == __version__:
        return None
    _remember(__version__)
    if seen and not newer(__version__, seen):
        return None
    return (f"[tensorfold] this is TensorFold {__version__}; what's new: "
            f"https://github.com/{REPO}/blob/v{__version__}/{CHANGELOG}")


def check_in_background() -> threading.Thread | None:
    """Look for a newer release without delaying anything; print one line if there is one."""

    if _checks_off():
        return None

    def run() -> None:
        line = notice(latest_release())
        if line:
            print(line, flush=True)

    thread = threading.Thread(target=run, name="tensorfold-update-check", daemon=True)
    thread.start()
    return thread


def _editable_clone() -> Path | None:
    """The git clone this package runs from when it was installed with ``pip install -e``, else None."""

    try:
        from importlib.metadata import distribution

        direct = distribution("tensorfold").read_text("direct_url.json")
        info: dict[str, Any] = json.loads(direct) if direct else {}
    except Exception:
        info = {}
    if not (info.get("dir_info") or {}).get("editable"):
        return None
    source = Path(__file__).resolve()
    for folder in source.parents:
        if (folder / ".git").exists():
            return folder
    return None


def _run(command: list[str], **kwargs: Any) -> int:
    print("[tensorfold] " + " ".join(command), flush=True)
    return subprocess.call(command, **kwargs)


def _git_oid(clone: Path, expression: str) -> str | None:
    """Resolve a complete SHA-1/SHA-256 object name; malformed or failed Git output cannot select a revision."""

    result = subprocess.run(["git", "-C", str(clone), "rev-parse", "--verify", expression],
                            capture_output=True, text=True)
    oid = result.stdout.strip()
    if result.returncode != 0 or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", oid) is None:
        print(f"[tensorfold] could not resolve updater object {expression}", file=sys.stderr)
        return None
    return oid


def _failure_report(message: str) -> BaseException | None:
    """Diagnostic output is observational; transport its failure with the operation's owned status."""

    try:
        print(f"[tensorfold] {message}", file=sys.stderr)
    except BaseException as error:
        return error
    return None


def _update_editable(clone: Path, tag: str) -> int:
    """Own a validated release's private ref through immutable commit selection and retirement."""

    from uuid import uuid4
    from tensorfold.cleanup import raise_failures

    base = _git_oid(clone, "HEAD^{commit}")
    if base is None:
        return 1
    private = f"refs/tensorfold/update/{uuid4().hex}"
    # Journal possible ownership before any mutation; only an acknowledged successful claim permits deletion.
    claimed, owned_oid, code, primary, prior = None, base, 1, None, None
    cleanup, cleanup_failure, cleanup_prior = 0, None, None
    try:
        claimed = _run(["git", "-C", str(clone), "update-ref", "--no-deref", private, base, ""])
        if claimed == 0:
            fetched = _run(["git", "-C", str(clone), "fetch", "--no-tags", REPO_URL, f"refs/tags/{tag}:{private}"])
            raw = _git_oid(clone, private)
            if raw is not None:
                owned_oid = raw
            if fetched == 0 and raw is not None:
                commit = _git_oid(clone, f"{raw}^{{commit}}")
                if commit is not None:
                    code = _run(["git", "-C", str(clone), "merge", "--ff-only", commit])
                    if code != 0:
                        message = f"{clone} could not fast-forward to {tag} (Git status {code}): update it yourself"
                        diagnostic = _failure_report(message)
                        if diagnostic is not None:
                            raise_failures(diagnostic, [RuntimeError(message)])
    except BaseException as error:
        primary = error
        prior = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
                 BaseException.__suppress_context__.__get__(error))
        raise
    finally:
        message = None
        if claimed == 0:
            try:
                cleanup = _run(["git", "-C", str(clone), "update-ref", "--no-deref", "-d", private, owned_oid])
            except BaseException as error:
                cleanup_failure, cleanup = error, 1
                cleanup_prior = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
                                 BaseException.__suppress_context__.__get__(error))
            if cleanup != 0:
                message = f"temporary updater ref cleanup failed: {private} (Git status {cleanup}); retained for inspection"
        elif claimed is None:
            message = f"private updater ref claim outcome is unknown: {private}; retained for inspection if present"
        else:
            message = f"could not claim private updater ref {private} (Git status {claimed}); existing state retained"
        context = RuntimeError(message) if message is not None else None
        diagnostic = _failure_report(message) if message is not None else None
        if cleanup_failure is not None:
            BaseException.__cause__.__set__(cleanup_failure, cleanup_prior[0])
            BaseException.__context__.__set__(cleanup_failure, cleanup_prior[1])
            BaseException.__suppress_context__.__set__(cleanup_failure, cleanup_prior[2])
        if primary is not None:
            BaseException.__cause__.__set__(primary, prior[0])
            BaseException.__context__.__set__(primary, prior[1])
            BaseException.__suppress_context__.__set__(primary, prior[2])
            raise_failures(primary, [error for error in (cleanup_failure, context, diagnostic) if error is not None])
        if cleanup_failure is not None:
            raise_failures(cleanup_failure, [error for error in (context, diagnostic) if error is not None])
        if diagnostic is not None:
            raise_failures(diagnostic, [context])
    if code == 0 and cleanup != 0:
        return 1
    if code == 0:
        show_whats_new(__version__, tag, clone)
    return code


def update(*, check_only: bool = False, force: bool = False) -> int:
    """``tensorfold update``: install the newest release (or say this one is current)."""

    if os.environ.get("TF_FOUNDATION_STATE") and not check_only:
        print("[tensorfold] this managed CUDA image protects its foundation; rebuild the image with "
              "the selected foundation inputs to update TensorFold", file=sys.stderr)
        return 1
    tag = latest_release(timeout=10.0, use_cache=False)
    if tag is None:
        print(f"[tensorfold] could not resolve a published release for {REPO} ({RELEASES_API})", file=sys.stderr)
        return 1
    if not newer(tag) and not force:
        print(f"[tensorfold] TensorFold {__version__} is the latest release")
        return 0
    print(f"[tensorfold] TensorFold {tag.lstrip('v')} is available (this is {__version__})")
    if check_only:
        return 0
    clone = _editable_clone()
    if clone is not None:
        dirty = subprocess.run(["git", "-C", str(clone), "status", "--porcelain"], capture_output=True, text=True)
        if dirty.returncode != 0 or dirty.stdout.strip():
            print(f"[tensorfold] this is an editable install from {clone}, which has local changes: update it "
                  f"yourself (review the owned fork's release {tag})", file=sys.stderr)
            return 1
        return _update_editable(clone, tag)
    # Upgrade dependencies only when the new release requires different versions.
    code = _run([sys.executable, "-m", "pip", "install", "--upgrade", f"git+{REPO_URL}@{tag}"])
    if code == 0:
        installed = subprocess.run([sys.executable, "-c", "import tensorfold; print(tensorfold.__version__)"],
                                   capture_output=True, text=True).stdout.strip()
        print(f"[tensorfold] installed TensorFold {installed or tag.lstrip('v')}; restart any running server")
        try:
            CACHE.unlink()
        except OSError:
            pass
        show_whats_new(__version__, tag)
    return code
