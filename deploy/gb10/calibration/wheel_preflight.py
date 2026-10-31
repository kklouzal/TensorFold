"""Cheap stdlib-only source-build prerequisites; execute before role admission."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def verify_git_role():
    from git_os import verify
    verify()


def preflight(source: Path, output: Path) -> tuple[Path, Path, Path, str]:
    selected = shutil.which("git")
    if selected is None:
        raise RuntimeError("source wheel build requires the reviewed OS Git addon; git is unavailable")
    git = Path(selected).resolve(strict=True)
    if not git.is_file() or not os.access(git, os.X_OK):
        raise RuntimeError("selected Git executable is inaccessible")
    verify_git_role()
    version = subprocess.run([str(git), "--version"], check=True, text=True, capture_output=True).stdout.strip()
    if not version.startswith("git version ") or "\n" in version:
        raise RuntimeError("selected Git returned an invalid version identity")
    source = source.resolve(strict=True)
    if not source.is_dir() or not os.access(source, os.R_OK | os.W_OK | os.X_OK):
        raise RuntimeError("source checkout must be readable/writable/searchable for backend outputs")
    metadata = source / ".git"
    mode = os.R_OK | (os.X_OK if metadata.is_dir() else 0)
    if not metadata.exists() or not os.access(metadata, mode):
        raise RuntimeError("source Git metadata is unavailable or unreadable")
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError("source wheel output must be a new path; preserve previous outcomes")
    if not output.parent.is_dir() or not os.access(output.parent, os.W_OK | os.X_OK):
        raise RuntimeError("source wheel output parent must be writable/searchable")
    return source, output, git, version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output, git, version = preflight(args.source, args.output)
    print(json.dumps({"scope": "prerequisites/Git version probe only; no source build",
                      "source": str(source), "output": str(output), "git": str(git), "git_version": version}))


if __name__ == "__main__":
    main()
