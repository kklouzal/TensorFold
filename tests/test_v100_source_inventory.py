"""Actual source-byte provenance with Git-owned checkout transformations."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v100_source_inventory", ROOT / "deploy/v100/source_foundation.py")
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)


@pytest.fixture
def repository(tmp_path):
    path = tmp_path / "owned source"
    path.mkdir()

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(path), "-c", "commit.gpgsign=false",
            "-c", "user.name=Source fixture", "-c", "user.email=fixture@example.invalid", *arguments], stderr=subprocess.PIPE)

    git("init", "--object-format=sha1")
    git("config", "core.autocrlf", "false")
    (path / ".gitattributes").write_text("script.bat text eol=crlf\n")
    (path / "script.bat").write_bytes(b"echo source\n")
    (path / "code header.txt").write_bytes(b"exact selected bytes\n")
    git("add", ".")
    git("commit", "-m", "Synthetic source input fixture")
    (path / "script.bat").unlink()
    git("checkout", "--", "script.bat")
    return path, git, git("rev-parse", "HEAD").decode().strip()


def test_checkout_eol_conversion_is_verified_by_git_while_raw_source_sha_is_exact(repository):
    path, _, commit = repository
    actual = SOURCE.source_inventory(path, commit)
    files = {row["path"]: row for row in actual["files"]}
    assert (path / "script.bat").read_bytes() == b"echo source\r\n"
    assert files["script.bat"]["checkout_transform"] is True
    assert files["script.bat"]["sha256"] == hashlib.sha256(b"echo source\r\n").hexdigest()
    assert files["code header.txt"]["checkout_transform"] is False
    assert SOURCE.source_inventory(path, commit) == actual


def test_modified_source_or_wrong_declared_commit_cannot_inherit_pinned_provenance(repository):
    path, _, commit = repository
    with pytest.raises(ValueError, match="commit differs"):
        SOURCE.source_inventory(path, "0" * 40)
    (path / "script.bat").write_bytes(b"changed source\r\n")
    with pytest.raises(ValueError, match="pinned Git blob"):
        SOURCE.source_inventory(path, commit)


def test_source_head_validation_binds_each_declared_repository(repository):
    path, _, commit = repository
    pins = {"sources": {path.name: {"commit": commit}}}
    SOURCE.verify_source_heads(pins, path.parent)
    pins["sources"][path.name]["commit"] = "0" * 40
    with pytest.raises(ValueError, match="immutable revision"):
        SOURCE.verify_source_heads(pins, path.parent)


@pytest.mark.skipif(os.name != "posix", reason="the selected source stack targets Linux")
def test_symlink_identity_hashes_the_link_payload_without_reading_an_external_target(repository):
    path, git, _ = repository
    (path / "link").symlink_to("code header.txt")
    git("add", "link")
    git("commit", "-m", "Synthetic source link fixture")
    actual = SOURCE.source_inventory(path, git("rev-parse", "HEAD").decode().strip())
    link = next(row for row in actual["files"] if row["path"] == "link")
    assert link["mode"] == "120000"
    assert link["sha256"] == hashlib.sha256(b"code header.txt").hexdigest()
    assert SOURCE.verify_source_links(path, actual) == [{"path": "link", "target": "code header.txt"}]


@pytest.mark.parametrize("ignored", [False, True])
def test_untracked_compiler_source_cannot_hide_behind_git_ignore(repository, ignored):
    path, git, commit = repository
    if ignored:
        (path / ".gitignore").write_text("*.cpp\n")
        git("add", ".gitignore")
        git("commit", "-m", "Synthetic ignored-input fixture")
        commit = git("rev-parse", "HEAD").decode().strip()
    extra = path / "aten/src/ATen/core/untracked_probe.cpp"
    extra.parent.mkdir(parents=True)
    extra.write_text("int extra_controllable_input = 1;\n")
    with pytest.raises(ValueError, match="untracked non-generated source input"):
        SOURCE.source_inventory(path, commit)


def test_only_declared_generated_names_are_allowed_and_their_actual_bytes_are_bound(repository):
    path, _, commit = repository
    original = SOURCE.source_inventory(path, commit)
    generated = path / "generated/known.cpp"
    generated.parent.mkdir()
    generated.write_bytes(b"known derived compiler input")
    inputs = {}
    actual = SOURCE.source_inventory(path, commit, generated_paths={"generated/known.cpp"}, generated_inputs=inputs)
    assert actual == original
    assert inputs == {"generated/known.cpp": SOURCE.file_identity(generated)}
    (generated.parent / "unexpected.cpp").write_text("extra input\n")
    with pytest.raises(ValueError, match="untracked non-generated source input"):
        SOURCE.source_inventory(path, commit, generated_paths={"generated/known.cpp"})


def test_recursive_module_cannot_add_untracked_compiler_inputs(repository, tmp_path):
    path, git, _ = repository
    module = tmp_path / "module source"
    subprocess.run(["git", "clone", "--shared", str(path), str(module)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    git("-c", "protocol.file.allow=always", "submodule", "add", str(module), "nested")
    git("commit", "-m", "Synthetic recursive source fixture")
    (path / "nested/extra.cpp").write_text("int extra = 1;\n")
    with pytest.raises(ValueError, match="nested/extra.cpp"):
        SOURCE.source_inventory(path, git("rev-parse", "HEAD").decode().strip())


@pytest.mark.skipif(os.name != "posix", reason="the selected source producer targets Linux")
def test_tracked_fifo_refuses_without_stranding_source_capture(repository):
    path, _, commit = repository
    (path / "script.bat").unlink()
    os.mkfifo(path / "script.bat")
    program = '''import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("source",sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
try: m.source_inventory(Path(sys.argv[2]),sys.argv[3])
except (OSError,ValueError): print("refused")
else: raise AssertionError("FIFO admitted as source")
'''
    result = subprocess.run([sys.executable, "-c", program, str(ROOT / "deploy/v100/source_foundation.py"),
                             str(path), commit], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0 and result.stdout.strip() == "refused", result.stderr


@pytest.mark.skipif(os.name != "posix", reason="the selected source producer targets Linux")
def test_build_link_closure_is_separate_from_git_link_payload_identity(repository, tmp_path):
    path, git, _ = repository
    external = tmp_path / "external"
    external.write_bytes(b"external compiler input")
    (path / "external link").symlink_to(external)
    git("add", "external link")
    git("commit", "-m", "Synthetic external link fixture")
    inventory = SOURCE.source_inventory(path, git("rev-parse", "HEAD").decode().strip())
    with pytest.raises(ValueError, match="escapes the inventoried closure"):
        SOURCE.verify_source_links(path, inventory)


@pytest.mark.skipif(os.name != "posix", reason="the source-build driver targets Ubuntu Linux")
def test_fresh_acquire_initializes_receipts_and_captures_inputs_without_manual_steps(repository, tmp_path):
    path, _, commit = repository
    state, project = tmp_path / "state", tmp_path / "project"
    source = state / "sources/demo"
    source.parent.mkdir(parents=True)
    subprocess.run(["git", "clone", "--shared", str(path), str(source)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    inventory = SOURCE.source_inventory(source, commit)
    deployment = project / "deploy/v100"
    deployment.mkdir(parents=True)
    (deployment / "source_foundation.py").symlink_to(ROOT / "deploy/v100/source_foundation.py")
    (deployment / "source-pins.json").write_text(json.dumps({"sources": {"demo": {
        "commit": commit, "repository": str(path),
        "source_inventory_sha256": hashlib.sha256(SOURCE.canonical(inventory) + b"\n").hexdigest()}}}))
    environment = os.environ.copy()
    environment.update(TF_SOURCE_STATE=str(state), TF_PROJECT_ROOT=str(project),
                       PATH=str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", ""))
    result = subprocess.run(["bash", str(ROOT / "deploy/v100/source_build.sh"), "acquire"],
                            env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (state / "logs").is_dir()
    assert json.loads((state / "receipts/demo-source-inventory.json").read_text()) == inventory
