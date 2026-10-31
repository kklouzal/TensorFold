"""Reviewed, additive OS Git role: inspect first, materialize offline, seal.

This does not resolve dependencies. Authenticated native APT simulation owns
the exact closure; dpkg owns installation. No existing package/file replacement
is admitted. The original SDK guard remains authoritative for its providers.
"""
from __future__ import annotations

import argparse
from email.parser import Parser
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile

PACKAGES = {"git", "git-man", "liberror-perl", "libcurl3t64-gnutls", "libssh-4",
            "librtmp1", "libbrotli1", "libpsl5t64", "libnghttp2-14", "libgssapi-krb5-2",
            "libkrb5-3", "libkeyutils1", "libk5crypto3", "libkrb5support0"}
GUARD = "/opt/tensorfold-pins/foundation_guard.py"
DEFAULT_STATE = Path("/opt/tensorfold-calibration/git-state.json")
FILTERS = [("path-exclude", "/usr/share/man/*"),
           ("path-exclude", "/usr/share/locale/*/LC_MESSAGES/*.mo"),
           ("path-exclude", "/usr/share/doc/*"),
           ("path-include", "/usr/share/doc/*/copyright"),
           ("path-include", "/usr/share/doc/*/changelog.*")]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, body):
    with Path(path).open("x") as stream:
        json.dump(body, stream, sort_keys=True, indent=2)
        stream.write("\n")


def run(*argv):
    return subprocess.run(list(argv), check=True, text=True, capture_output=True).stdout


def dpkg_state():
    raw = run("dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${Status}\n")
    result = {}
    for line in raw.splitlines():
        name, version, status = line.split("\t", 2)
        if name in result:
            raise ValueError("duplicate dpkg package identity")
        result[name] = {"version": version, "status": status}
    return result


def dpkg_config():
    """Pin this base's actual policy, not a general dpkg config interpreter."""
    directory = Path("/etc/dpkg/dpkg.cfg.d")
    paths = sorted(directory.iterdir())
    if {path.name for path in paths} != {"docker-apt-speedup", "excludes"}:
        raise ValueError("baseline dpkg configuration file set changed")
    paths.append(Path("/etc/dpkg/dpkg.cfg"))
    files, filters = {}, []
    for path in paths:
        text = path.read_text()
        files[str(path)] = digest(path)
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith(("path-exclude", "path-include")):
                option, separator, pattern = line.partition("=")
                if path.name != "excludes" or not separator:
                    raise ValueError("unsupported dpkg path-filter configuration")
                filters.append((option, pattern))
    if filters != FILTERS:
        raise ValueError("baseline dpkg path-filter rules changed")
    return {"files": files, "filters": [list(rule) for rule in filters]}


def filter_omissions(packages, contract):
    omitted = {}
    for package in packages:
        for row in package["payload"]:
            include, reason = True, None
            for option, pattern in contract["filters"]:
                if fnmatch.fnmatchcase(row["path"], pattern):
                    include, reason = option == "path-include", pattern
            if row["type"] != "directory" and not include:
                path = row["path"]
                if (not path.startswith(("/usr/share/doc/", "/usr/share/man/", "/usr/share/locale/"))
                        or "elf" in row or (row["type"] == "file" and row["mode"] & 0o111)):
                    raise ValueError(f"dpkg filter would omit a runtime executable/library: {path}")
                omitted[path] = {"rule": reason, "type": row["type"]}
    return omitted


def payload_path(name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe DEB member: {name}")
    return Path("/", *relative.parts)


def inspect_deb(path):
    """No extraction onto the host/target root; temporary container files only."""
    control = dict(Parser().parsestr(run("dpkg-deb", "--field", str(path))).items())
    scripts, payload = {}, []
    from tensorfold_foundation_runtime import elf, provider_kind
    for flag in ("--ctrl-tarfile", "--fsys-tarfile"):
        with tempfile.TemporaryFile() as output:
            subprocess.run(["dpkg-deb", flag, str(path)], check=True, stdout=output)
            output.seek(0)
            with tarfile.open(fileobj=output) as archive:
                for member in archive:
                    target = payload_path(member.name)
                    if flag == "--ctrl-tarfile":
                        if member.isfile() and target.name in {"preinst", "postinst", "prerm", "postrm", "triggers", "conffiles"}:
                            scripts[target.name] = archive.extractfile(member).read().decode("utf-8")
                        continue
                    if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
                        raise ValueError(f"unsupported special DEB member: {member.name}")
                    exists = os.path.lexists(target)
                    collision = exists and not (member.isdir() and target.is_dir() and not target.is_symlink())
                    record = {"path": str(target), "mode": member.mode, "size": member.size,
                              "type": "directory" if member.isdir() else "file" if member.isfile() else "symlink" if member.issym() else "hardlink",
                              "link": member.linkname, "old_path_collision": collision}
                    if provider_kind(target.name):
                        raise ValueError(f"new Git role contains a protected native provider: {target}")
                    if member.isfile():
                        stream = archive.extractfile(member)
                        with tempfile.NamedTemporaryFile() as candidate:
                            checksum = hashlib.sha256()
                            first = stream.read(65536)
                            is_elf = first.startswith(b"\x7fELF")
                            chunk = first
                            while chunk:
                                checksum.update(chunk)
                                if is_elf:
                                    candidate.write(chunk)
                                chunk = stream.read(65536)
                            record["sha256"] = checksum.hexdigest()
                            if is_elf:
                                candidate.flush()
                                record["elf"] = elf(Path(candidate.name))
                                soname = record["elf"]["soname"]
                                if soname and provider_kind(soname):
                                    raise ValueError(f"Git ELF replaces protected provider: {target}")
                    payload.append(record)
    return {"control": control, "maintainers": scripts, "payload": payload}


def plan(metadata, debs, acquisition, destination):
    summary = load(metadata / "summary.json")
    if not summary["dpkg_unchanged"] or (metadata / "dpkg-before.stdout").read_bytes() != (metadata / "dpkg-after.stdout").read_bytes():
        raise ValueError("APT simulation changed installed packages")
    old_versions = dict(line.split("\t", 1) for line in (metadata / "dpkg-before.stdout").read_text().splitlines())
    current = dpkg_state()
    config = dpkg_config()
    if (Path.home() / ".dpkg.cfg").exists():
        raise ValueError("OS Git installation requires no additional user dpkg configuration")
    if {name: row["version"] for name, row in current.items()} != old_versions:
        raise ValueError("inspection image differs from native APT baseline")
    if run("dpkg", "--print-architecture").strip() != "arm64":
        raise ValueError("Git role requires the recorded native ARM64 target")
    acquisition_rows = load(acquisition)["packages"]
    acquired = {row["name"]: row for row in acquisition_rows}
    if (len(acquisition_rows) != len(PACKAGES) or len(summary["new_packages"]) != len(PACKAGES)
            or set(acquired) != PACKAGES or {row["package"] for row in summary["new_packages"]} != PACKAGES):
        raise ValueError("unreviewed Git dependency closure")
    packages = []
    for item in summary["new_packages"]:
        name = item["package"]
        if name in {key.split(":")[0] for key in current}:
            raise ValueError(f"Git role replaces existing package: {name}")
        artifact = acquired[name]
        filename = artifact["filename"]
        if Path(filename).name != filename or not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
            raise ValueError("invalid Git artifact identity")
        path = debs / filename
        if path.stat().st_size != artifact["size"] or digest(path) != artifact["sha256"]:
            raise ValueError(f"Git artifact size/hash differs: {name}")
        source = Parser().parsestr((metadata / item["metadata_file"]).read_text())
        if (source["Package"] != name or source["Version"] != artifact["version"]
                or source["SHA256"] != artifact["sha256"] or source["Architecture"] != artifact["architecture"]):
            raise ValueError("Git acquisition differs from authenticated package metadata")
        details = inspect_deb(path)
        for key, expected in (("Package", name), ("Version", artifact["version"]), ("Architecture", artifact["architecture"])):
            if details["control"].get(key) != expected:
                raise ValueError("DEB control differs from authenticated metadata")
        if artifact["architecture"] not in {"arm64", "all"}:
            raise ValueError("foreign Git dependency architecture")
        packages.append({**artifact, **details})
    seen = set()
    for package in packages:
        for row in package["payload"]:
            if row["type"] != "directory":
                if row["path"] in seen:
                    raise ValueError("Git packages duplicate a non-directory payload path")
                seen.add(row["path"])
    write(destination, {"schema_version": 2, "scope": "reviewed OS Git only", "baseline": current,
          "dpkg_config": config, "filtered_paths": filter_omissions(packages, config),
          "inspector_sha256": digest(__file__),
          "acquisition_sha256": digest(acquisition), "apt_summary_sha256": digest(metadata / "summary.json"),
          "packages": packages})


def old_unchanged(before, after):
    for name, row in before.items():
        if after.get(name) != row:
            raise ValueError(f"existing dpkg version/status changed: {name}")


def file_identity(path):
    info = path.lstat()
    if path.is_symlink():
        return {"type": "symlink", "target": os.readlink(path)}
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Git state entry is not regular/link: {path}")
    return {"type": "file", "sha256": digest(path), "mode": stat.S_IMODE(info.st_mode)}


def install(manifest_path, debs, state_path):
    manifest = load(manifest_path)
    before = dpkg_state()
    if (manifest["schema_version"] != 2 or manifest["inspector_sha256"] != digest(__file__)
            or before != manifest["baseline"] or {p["name"] for p in manifest["packages"]} != PACKAGES):
        raise ValueError("Git installation baseline/closure differs")
    if dpkg_config() != manifest["dpkg_config"] or (Path.home() / ".dpkg.cfg").exists():
        raise ValueError("Git dpkg filter configuration changed")
    if filter_omissions(manifest["packages"], manifest["dpkg_config"]) != manifest["filtered_paths"]:
        raise ValueError("Git declared filter omissions differ")
    subprocess.run([sys.executable, GUARD, "audit"], check=True)
    paths = []
    for package in manifest["packages"]:
        path = debs / package["filename"]
        if digest(path) != package["sha256"] or path.stat().st_size != package["size"]:
            raise ValueError("Git DEB identity differs")
        if inspect_deb(path) != {key: package[key] for key in ("control", "maintainers", "payload")}:
            raise ValueError("Git payload/control or old-path state differs from reviewed plan")
        if any(row["old_path_collision"] for row in package["payload"]):
            raise ValueError("Git payload overwrites an existing non-directory")
        paths.append(str(path))
    failures = []
    try:
        subprocess.run(["dpkg", "--install", *paths], check=True, stdin=subprocess.DEVNULL,
                       env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
    except BaseException as error:
        failures.append(error)
    try:
        after = dpkg_state()
        old_unchanged(before, after)
        if dpkg_config() != manifest["dpkg_config"]:
            raise ValueError("dpkg configuration changed during Git installation")
        new = {name: row for name, row in after.items() if name not in before}
        expected = {package["name"]: package["version"] for package in manifest["packages"]}
        if ({name.split(":")[0]: row["version"] for name, row in new.items()} != expected
                or any(row["status"] != "install ok installed" for row in new.values())):
            raise ValueError("Git dependency set is incomplete, changed, or half-configured")
    except BaseException as error:
        failures.append(error)
    try:
        subprocess.run([sys.executable, GUARD, "audit"], check=True)
    except BaseException as error:
        failures.append(error)
    if failures:
        raise BaseExceptionGroup("OS Git installation/final audits failed", failures)
    files, omitted = {}, {}
    for package in manifest["packages"]:
        for row in package["payload"]:
            if row["type"] != "directory":
                path = Path(row["path"])
                if not os.path.lexists(path) and str(path) in manifest["filtered_paths"]:
                    omitted[str(path)] = manifest["filtered_paths"][str(path)]
                    continue
                files[str(path)] = file_identity(path)
    write(state_path, {"schema_version": 2, "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
                       "dpkg": after, "files": files, "omitted_by_dpkg": omitted,
                       "dpkg_config": manifest["dpkg_config"], "helper_sha256": digest(__file__)})


def verify(state_path=DEFAULT_STATE):
    state = load(state_path)
    if state["schema_version"] != 2 or state["helper_sha256"] != digest(__file__):
        raise ValueError("OS Git role helper identity changed")
    if digest(state["manifest"]) != state["manifest_sha256"] or dpkg_state() != state["dpkg"]:
        raise ValueError("OS Git role manifest/package state changed")
    if dpkg_config() != state["dpkg_config"]:
        raise ValueError("OS Git dpkg configuration changed")
    for name in state["omitted_by_dpkg"]:
        if os.path.lexists(name):
            raise ValueError(f"dpkg-excluded Git path unexpectedly materialized: {name}")
    for name, expected in state["files"].items():
        if file_identity(Path(name)) != expected:
            raise ValueError(f"OS Git role file/link changed: {name}")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "install", "verify"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--debs", type=Path)
    parser.add_argument("--acquisition", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("/opt/tensorfold-calibration/git-inputs/manifest.json"))
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()
    if args.command == "plan":
        plan(args.metadata, args.debs, args.acquisition, args.manifest)
    elif args.command == "install":
        install(args.manifest, args.debs, args.state)
    else:
        verify(args.state)


if __name__ == "__main__":
    main()
