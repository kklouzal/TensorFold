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


@pytest.fixture
def owned_patch(repository, tmp_path):
    path, git, base = repository
    (path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(SourceFixture LANGUAGES CXX)\n"
        "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)\nadd_library(fixture STATIC fixture.cpp)\n")
    (path / "fixture.cpp").write_text("int fixture() { return 1; }\n")
    (path / ".gitignore").write_text("build/\nagent_space/\n")
    git("add", "CMakeLists.txt", "fixture.cpp", ".gitignore")
    git("commit", "-m", "Declare independent synthetic CMake fixture")
    base = git("rev-parse", "HEAD").decode().strip()
    git("remote", "add", "origin", str(path))
    git("checkout", "--detach", base)
    (path / "code header.txt").write_text("corrected owned bytes\n")
    (path / "new test.cpp").write_text("// independent regression fixture\n")
    git("add", "code header.txt", "new test.cpp")
    patches = tmp_path / "patches"
    patches.mkdir()
    patch = patches / "owned.patch"
    patch.write_bytes(git("diff", "--cached", "--binary", base))
    message = "Fix fixture source\n\nTest Plan:\n```bash\nfixture-check\n```\n"
    date = "1793049855 +0000"
    environment = {**os.environ, "GIT_AUTHOR_NAME": "TensorFold Source Builder",
                   "GIT_AUTHOR_EMAIL": "source-builder@tensorfold.invalid",
                   "GIT_COMMITTER_NAME": "TensorFold Source Builder",
                   "GIT_COMMITTER_EMAIL": "source-builder@tensorfold.invalid",
                   "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    final = subprocess.check_output(["git", "-C", str(path), "commit-tree",
                                     git("write-tree").decode().strip(), "-p", base],
                                    input=message.encode(), env=environment).decode().strip()
    git("reset", "--hard", final)
    inventory = SOURCE.source_inventory(path, final)
    source = {"commit": base, "effective_commit": final, "repository": str(path),
              "fork_date": date, "fork_message": message,
              "owned_patches": [{"artifact": patch.name, **SOURCE.file_identity(patch)}],
              "source_inventory_sha256": hashlib.sha256(SOURCE.canonical(inventory) + b"\n").hexdigest()}
    git("reset", "--hard", base)
    return path, git, source, patches


def test_owned_patch_staging_resumes_exact_index_without_reapplying(owned_patch):
    path, git, source, patches = owned_patch
    tree = SOURCE.stage_owned_patches(path, source, patches)
    index = git("ls-files", "--stage", "-z")
    assert SOURCE.stage_owned_patches(path, source, patches) == tree
    assert git("ls-files", "--stage", "-z") == index
    assert git("rev-parse", "HEAD").decode().strip() == source["commit"]
    with pytest.raises(ValueError, match="immutable revision"):
        SOURCE.verify_source_heads({"sources": {path.name: source}}, path.parent)


@pytest.mark.parametrize("kind", ["unstaged", "staged", "partial", "patch-bytes", "wrong-head"])
def test_owned_patch_resume_rejects_unknown_states_without_reverting(owned_patch, kind):
    path, git, source, patches = owned_patch
    SOURCE.stage_owned_patches(path, source, patches)
    if kind in ("unstaged", "staged"):
        (path / "code header.txt").write_text("unknown user bytes\n")
        if kind == "staged":
            git("add", "code header.txt")
    elif kind == "partial":
        git("restore", "--staged", "--source=HEAD", "code header.txt")
        git("restore", "code header.txt")
    elif kind == "patch-bytes":
        (patches / "owned.patch").write_text("unapproved patch\n")
    else:
        git("reset", "--hard", source["effective_commit"])
    before = git("diff", "HEAD", "--binary")
    index = git("ls-files", "--stage", "-z")
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        SOURCE.stage_owned_patches(path, source, patches)
    assert git("diff", "HEAD", "--binary") == before
    assert git("ls-files", "--stage", "-z") == index


def test_bootstrap_inventory_preserves_actual_head_and_final_gate(owned_patch, tmp_path):
    path, git, source, patches = owned_patch
    pins = {"sources": {path.name: source}}
    receipts = tmp_path / "receipts"
    SOURCE.capture_bootstrap_sources(pins, path.parent, patches, receipts)
    receipt = json.loads((receipts / "bootstrap-sources.json").read_text())
    assert receipt["phase"] == "precommit-build-inputs"
    assert receipt["sources"][path.name]["head"] == source["commit"]
    assert receipt["sources"][path.name]["staged_tree"] == git("write-tree").decode().strip()
    with pytest.raises(ValueError, match="immutable revision"):
        SOURCE.capture_sources(pins, path.parent, receipts)
    (path / "unapproved.cpp").write_text("// hidden compiler input\n")
    with pytest.raises(ValueError, match="untracked non-generated"):
        SOURCE.capture_bootstrap_sources(pins, path.parent, patches, receipts)


@pytest.mark.parametrize("capture", ["bootstrap", "final", "finalize", "source"])
@pytest.mark.parametrize("kind", ["staged", "intent-to-add"])
def test_effective_checkout_rejects_unrecorded_index_additions(owned_patch, tmp_path, capture, kind):
    path, git, source, patches = owned_patch
    git("reset", "--hard", source["effective_commit"])
    extra = path / "unexpected.cpp"
    extra.write_text("// staged file must not escape source inventory\n")
    git("add", *(["-N"] if kind == "intent-to-add" else []), "unexpected.cpp")
    pins = {"sources": {path.name: source}}
    receipts = tmp_path / "receipts"
    index = git("ls-files", "--stage", "-z")
    with pytest.raises(subprocess.CalledProcessError):
        if capture == "bootstrap":
            SOURCE.capture_bootstrap_sources(pins, path.parent, patches, receipts)
        elif capture == "final":
            SOURCE.capture_sources(pins, path.parent, receipts)
        elif capture == "finalize":
            SOURCE.finalize_sources(pins, path.parent, patches, receipts, tmp_path / "unavailable-spin")
        else:
            SOURCE.source_inventory(path, source["effective_commit"])
    assert extra.read_text() == "// staged file must not escape source inventory\n"
    assert git("ls-files", "--stage", "-z") == index


def test_acquisition_initializes_all_repositories_before_staging(owned_patch, tmp_path, monkeypatch):
    path, _, source, patches = owned_patch
    directory = tmp_path / "acquired"
    pins = {"sources": {"pytorch": source, "native": {"repository": str(path), "commit": source["commit"]}}}
    original = SOURCE.stage_owned_patches
    staged = []

    def apply(checkout, selected, patch_directory):
        assert (directory / "native/.git").is_dir()
        staged.append(checkout.name)
        return original(checkout, selected, patch_directory)

    monkeypatch.setattr(SOURCE, "stage_owned_patches", apply)
    SOURCE.acquire_sources(pins, directory, patches, stage_only=True)
    SOURCE.acquire_sources(pins, directory, patches, stage_only=True)
    assert staged == ["pytorch", "pytorch"]


def test_default_acquire_rejects_unknown_compiler_inputs_before_fork_commit(owned_patch, monkeypatch):
    path, git, source, patches = owned_patch
    extra = path / "unrecorded.cpp"
    extra.write_text("// unrelated user source\n")
    finalized = []
    monkeypatch.setattr(SOURCE, "apply_owned_patches", lambda *args, **kwargs: finalized.append(args))
    with pytest.raises(ValueError, match="untracked non-generated"):
        SOURCE.acquire_sources({"sources": {path.name: source}}, path.parent, patches)
    assert not finalized
    assert git("rev-parse", "HEAD").decode().strip() == source["commit"]
    assert extra.read_text() == "// unrelated user source\n"


def test_finalization_rejects_missing_real_build_before_lint_or_commit(owned_patch, tmp_path):
    path, git, source, patches = owned_patch
    lint = tmp_path / "spin"
    lint.write_text("#!/bin/sh\nexit 0\n")
    lint.chmod(0o755)
    with pytest.raises(ValueError, match="real framework bootstrap build"):
        SOURCE.apply_owned_patches(path, source, patches, lint_tool=lint)
    assert git("rev-parse", "HEAD").decode().strip() == source["commit"]


@pytest.mark.parametrize("nested", [False, True])
def test_dirty_gitmodules_rejected_before_any_submodule_update(repository, tmp_path, monkeypatch, nested):
    path, git, _ = repository
    child = tmp_path / "submodule-origin"
    subprocess.run(["git", "clone", str(path), str(child)], check=True, capture_output=True)
    git("-c", "protocol.file.allow=always", "submodule", "add", str(child), "module")
    git("commit", "-m", "Pin synthetic submodule")
    base = git("rev-parse", "HEAD").decode().strip()
    git("remote", "add", "origin", str(path))
    if nested:
        metadata = path / "module/.gitmodules"
    else:
        metadata = path / ".gitmodules"
    metadata.write_text('[submodule "unapproved"]\npath = unapproved\nurl = https://unapproved.invalid/repo\n')
    if nested:
        subprocess.run(["git", "-C", str(path / "module"), "add", ".gitmodules"], check=True)
    before = metadata.read_bytes()
    called = []
    run = SOURCE.subprocess.run

    def guarded(arguments, **kwargs):
        if "submodule" in arguments and "update" in arguments:
            called.append(arguments)
            raise AssertionError("untrusted .gitmodules must be rejected before external acquisition")
        return run(arguments, **kwargs)

    monkeypatch.setattr(SOURCE.subprocess, "run", guarded)
    with pytest.raises(subprocess.CalledProcessError):
        SOURCE.acquire_sources({"sources": {path.name: {"repository": str(path), "commit": base}}},
                               path.parent, tmp_path, stage_only=True)
    assert not called
    assert metadata.read_bytes() == before


@pytest.mark.parametrize("lint_exit", [0, 1])
def test_finalize_uses_real_cmake_graph_and_lint_before_deterministic_commit(owned_patch, tmp_path, monkeypatch, lint_exit):
    origin, _, source, patches = owned_patch
    root = tmp_path / "sources/pytorch"
    subprocess.run(["git", "clone", str(origin), str(root)], check=True, capture_output=True)
    source = {**source, "repository": str(origin)}
    SOURCE.stage_owned_patches(root, source, patches)
    # This is an independent tiny CMake project, not a fabricated compilation
    # database or a substitute qualification build of the upstream framework.
    subprocess.run(["cmake", "-S", str(root), "-B", str(root / "build"), "-G", "Ninja"],
                   check=True, capture_output=True)
    subprocess.run(["cmake", "--build", str(root / "build"), "--parallel", "1"], check=True, capture_output=True)
    lint = tmp_path / "spin"
    lint.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$TF_TEST_LINT_LOG"\nexit "$TF_TEST_LINT_EXIT"\n')
    lint.chmod(0o755)
    log = tmp_path / "lint-arguments.txt"
    monkeypatch.setenv("TF_TEST_LINT_LOG", str(log))
    monkeypatch.setenv("TF_TEST_LINT_EXIT", str(lint_exit))
    receipts = tmp_path / "receipts"
    pins = {"sources": {"pytorch": source}}
    if lint_exit:
        with pytest.raises(subprocess.CalledProcessError):
            SOURCE.finalize_sources(pins, root.parent, patches, receipts, lint)
    else:
        SOURCE.finalize_sources(pins, root.parent, patches, receipts, lint)
        SOURCE.finalize_sources(pins, root.parent, patches, receipts, lint)
        assert (receipts / "pytorch-source-inventory.json").is_file()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
    assert head == source["commit" if lint_exit else "effective_commit"]
    assert log.read_text().splitlines() == ["quicklint", "--", "-a", "code header.txt", "new test.cpp"]
