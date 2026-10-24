"""Freeze one clean commit and two immutable native-Linux BuildKit contexts.

This creates build inputs, never a checkout overlay or a build process. Inputs
must stay immutable for the whole capture/build/audit interval. New, incomplete
outputs are retained for diagnosis; they are never retried or overwritten.
The format and identity algorithms match the earlier schema4 capture contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path

# Isolated source preparation runs before installed project dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gb10_io import (
    READ_FLAGS,
    RetirementFailure,
    _raise_cleanup,
    archive,
    borrowed_stream,
    new_file,
    owned_descriptor,
    read_text,
    sha256,
    stamp,
    write_all,
)

ARCHITECTURES = ("amd64", "arm64")
_ALIASES = {"amd64": ("x86_64", "amd64"), "arm64": ("aarch64", "arm64")}
NAMES = ("foundation_pins", "foundation_artifacts")


def declared_target(architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError("declared architecture must be amd64 or arm64")
    return {"architecture": architecture, "platform": "linux/" + architecture,
            "native_machine": _ALIASES[architecture][0]}


def build_input_identity(source):
    selected = declared_target(source.get("architecture"))
    if any(source.get(key) != value for key, value in selected.items()):
        raise ValueError("capture target architecture/platform/machine differs")
    contexts = source["named_contexts"]
    if set(contexts) != set(NAMES):
        raise ValueError("exactly foundation_pins and foundation_artifacts are required")
    body = {"schema": 2, "platform": selected["platform"], "source_uid": source["source_uid"],
            "source_archive_sha256": source["archive_sha256"],
            "named_contexts": {name: contexts[name]["context_uid"] for name in sorted(contexts)}}
    return digest(body)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def publish_document(path, value):
    # Serialize first. Publish only after the raw output's close succeeds.
    raw = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".new")
    with new_file(temporary) as stream:
        write_all(stream, raw)
    temporary.replace(path)


def regular_tree(root, destination=None):
    """Descriptor-walk an immutable regular tree, optionally copying each FD.

    Root aliases are resolved by capture_context only. Entries are never
    followed through links. Identity stamps are local race evidence, not UIDs.
    Destination is a fresh caller-owned tree, not an update/recovery target.
    """
    root = Path(root)
    files, directories, identities = {}, {}, {}
    with owned_descriptor(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC) as root_fd:
        def walk(fd, relative):
            before = os.fstat(fd)
            directories[relative] = stat.S_IMODE(before.st_mode)
            identities[relative] = stamp(before)
            names = sorted(os.listdir(fd))
            for name in names:
                key = name if not relative else relative + "/" + name
                prior = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(prior.st_mode):
                    with owned_descriptor(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                          dir_fd=fd) as child:
                        if stamp(os.fstat(child)) != stamp(prior):
                            raise ValueError("context directory changed before open: " + key)
                        if destination is not None:
                            (destination / key).mkdir()
                        walk(child, key)
                elif stat.S_ISREG(prior.st_mode):
                    with owned_descriptor(name, READ_FLAGS, dir_fd=fd) as child:
                        opened = os.fstat(child)
                        if not stat.S_ISREG(opened.st_mode) or stamp(opened) != stamp(prior):
                            raise ValueError("context file changed before open: " + key)
                        checksum = hashlib.sha256()
                        if destination is None:
                            while block := os.read(child, 1 << 20):
                                checksum.update(block)
                        else:
                            with new_file(destination / key) as target:
                                while block := os.read(child, 1 << 20):
                                    checksum.update(block)
                                    write_all(target, block)
                                os.fchmod(target.fileno(), stat.S_IMODE(opened.st_mode))
                        if stamp(os.fstat(child)) != stamp(opened):
                            raise ValueError("context file changed while reading: " + key)
                        identities[key] = stamp(opened)
                        files[key] = {"sha256": checksum.hexdigest(), "size": opened.st_size,
                                      "mode": stat.S_IMODE(opened.st_mode)}
                else:
                    raise ValueError("named contexts require regular files/directories, not links or specials: " + key)
                if stamp(os.stat(name, dir_fd=fd, follow_symlinks=False)) != stamp(prior):
                    raise ValueError("context entry changed during traversal: " + key)
            if sorted(os.listdir(fd)) != names or stamp(os.fstat(fd)) != stamp(before):
                raise ValueError("context directory changed during traversal: " + relative)
        walk(root_fd, "")
        if stamp(os.stat(root, follow_symlinks=False)) != stamp(os.fstat(root_fd)):
            raise ValueError("context root changed during traversal")
    return {"files": files, "directories": directories}, identities


def capture_context(name, root, output):
    if name not in NAMES:
        raise ValueError("unsupported named context")
    root = Path(root).resolve(strict=True)
    output = Path(output)
    if output.resolve().is_relative_to(root):
        raise ValueError("context capture destination must be outside its input tree")
    before, identities = regular_tree(root)
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "payload"
    payload.mkdir()
    copied, copied_identities = regular_tree(root, payload)
    after, final_identities = regular_tree(root)
    if before != copied or before != after or identities != copied_identities or identities != final_identities:
        raise ValueError("named context changed during capture")
    for relative in sorted(copied["directories"], key=lambda item: item.count("/"), reverse=True):
        (payload / relative).chmod(copied["directories"][relative])
    output_archive = output / "context.tar"
    with archive(output_archive, mode="x", format=tarfile.PAX_FORMAT) as target:
        for relative, mode in sorted(copied["directories"].items()):
            info = tarfile.TarInfo("." if not relative else relative)
            info.type, info.mode, info.mtime = tarfile.DIRTYPE, mode, 0
            target.addfile(info)
        for relative, record in sorted(copied["files"].items()):
            info = tarfile.TarInfo(relative)
            info.mode, info.size, info.mtime = record["mode"], record["size"], 0
            path = payload / relative
            with owned_descriptor(path, READ_FLAGS) as fd:
                before_file = os.fstat(fd)
                if not stat.S_ISREG(before_file.st_mode) or before_file.st_size != record["size"]:
                    raise ValueError("context archive member changed before open: " + relative)
                with borrowed_stream(fd) as stream:
                    target.addfile(info, stream)
                if (stamp(os.fstat(fd)) != stamp(before_file)
                        or stamp(os.stat(path, follow_symlinks=False)) != stamp(before_file)):
                    raise ValueError("context archive member changed during add: " + relative)
    if regular_tree(payload)[0] != copied or regular_tree(root) != (before, identities):
        raise ValueError("named context changed while archiving")
    record = {"schema": 1, "name": name, **copied, "archive_sha256": sha256(output_archive)}
    return {**record, "context_uid": digest(record)}


def git(repo, *arguments):
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def capture(repo, revision, output, *, architecture, foundation_pins, foundation_artifacts):
    selected = declared_target(architecture)
    repo = Path(repo).resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--revision requires an explicit full committed SHA")
    if git(repo, "rev-parse", revision + "^{commit}") != revision:
        raise ValueError("revision is not the requested commit")
    if git(repo, "rev-parse", "HEAD") != revision or git(repo, "status", "--porcelain"):
        raise ValueError("capture requires HEAD at the requested revision and a clean worktree")
    output = Path(output).absolute()
    failure = RetirementFailure()
    failure.resource = output
    output.mkdir(parents=True, exist_ok=False)
    status = {"revision": revision, "repo": str(repo), "started_unix_ns": time.time_ns(), "complete": False}
    try:
        source_archive = output / "source.tar"
        with new_file(source_archive) as stream:
            subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", revision], stdout=stream, check=True)
        payload = output / "payload"
        payload.mkdir()
        with archive(source_archive, "r") as stream:
            stream.extractall(payload, filter="data")
        files, symlinks, modes = {}, {}, {}
        for path in sorted(payload.rglob("*")):
            name = path.relative_to(payload).as_posix()
            if path.is_symlink():
                target = path.resolve(strict=True)
                if not target.is_relative_to(payload):
                    raise ValueError("archive link escapes the frozen source: " + name)
                symlinks[name] = str(path.readlink())
            elif path.is_file():
                files[name] = sha256(path)
                modes[name] = stat.S_IMODE(path.stat().st_mode)
        required = ("deploy/gb10/Dockerfile", "pyproject.toml", "setup.py", "src/tensorfold/_fd_owner.c")
        if any(name not in files for name in required):
            raise ValueError("frozen commit is missing a required regular build input")
        dockerfile = read_text(payload / "deploy/gb10/Dockerfile")
        base = re.search(r"^ARG BASE_IMAGE=([^\s]+@sha256:[0-9a-f]{64})$", dockerfile, re.MULTILINE)
        if base is None:
            raise ValueError("Dockerfile must declare the pinned immutable BASE_IMAGE")
        identity = digest({"files": files, "symlinks": symlinks, "modes": modes})
        source = {"schema": 4, "revision": revision, "tree": git(repo, "rev-parse", revision + "^{tree}"),
                  "source_uid": identity, "archive_sha256": sha256(source_archive), "files": files,
                  "symlinks": symlinks, "modes": modes, **selected, "base_image": base.group(1)}
        roots = {"foundation_pins": foundation_pins, "foundation_artifacts": foundation_artifacts}
        source["named_contexts"] = {
            name: capture_context(name, roots[name], output / "named-contexts" / name) for name in NAMES}
        source["build_input_uid"] = build_input_identity(source)
        if git(repo, "rev-parse", "HEAD") != revision or git(repo, "status", "--porcelain"):
            raise ValueError("worktree changed during capture; preserved attempt is not complete")
        publish_document(output / "source.json", source)
        complete = {**status, "complete": True, "source_uid": identity, "build_input_uid": source["build_input_uid"],
                    "finished_unix_ns": time.time_ns()}
    except BaseException as primary:
        failure.original_cause = BaseException.__cause__.__get__(primary)
        failure.original_context = BaseException.__context__.__get__(primary)
        failure.original_suppress_context = BaseException.__suppress_context__.__get__(primary)
        try:
            publish_document(output / "capture-status.json", status)
        except BaseException as diagnostic:  # noqa: BLE001 - status failure cannot replace the capture failure.
            failure.cleanup = diagnostic
            BaseException.__cause__.__set__(failure, failure.original_cause)
            BaseException.__context__.__set__(failure, failure.original_context)
            BaseException.__suppress_context__.__set__(failure, failure.original_suppress_context)
            BaseException.__cause__.__set__(primary, failure)
            try:
                _raise_cleanup(failure, diagnostic)
            except BaseException as transport:  # noqa: BLE001 - retained on the carrier before primary rethrow.
                failure.transport = (BaseException.__cause__.__get__(failure)
                                     if transport is failure else transport)
            raise primary from failure
        raise
    else:
        publish_document(output / "capture-status.json", complete)
    return source


def main():
    def interrupted(signum, _frame):
        raise InterruptedError("source capture interrupted by signal " + str(signum))
    for signum in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(signum, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new capture directory; never overwritten")
    parser.add_argument("--foundation-pins", type=Path, required=True)
    parser.add_argument("--foundation-artifacts", type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.repo, args.revision, args.output, architecture=args.architecture,
                     foundation_pins=args.foundation_pins, foundation_artifacts=args.foundation_artifacts)
    print(json.dumps({"revision": result["revision"], "source_uid": result["source_uid"],
                      "build_input_uid": result["build_input_uid"], "capture": str(args.output.absolute())}, sort_keys=True))


if __name__ == "__main__":
    main()
