"""Reproducible source/wheel/image preparation for the private V100 comparison.

The frozen source is evidence and a build input, never an editable checkout.
Builds use a disposable derived staging directory. Checks import the installed
wheel; the source mount supplies tests/tools only. Each command has a fresh
output directory and retains its arguments, logs and terminal outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
import zipfile

if __package__:
    from .v100_short_container import OwnedContainer
    from .worker_lifetime import raise_failures
else:
    from v100_short_container import OwnedContainer
    from worker_lifetime import raise_failures

CORE = "sha256:829ebca249664e8014bbc883305d8ae7f15edcab3fea7b8b620a19c8e17f150e"


def identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def publish(path, value):
    # Exclusive creation makes retries use a new named attempt, preserving failures.
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def source_files(repo):
    raw = git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    names = sorted(set(os.fsdecode(name) for name in raw.split(b"\0") if name))
    result = {}
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source inventory contains a non-relative path")
        if relative.parts[0] == "task-artifacts":
            continue
        path = repo / relative
        if not path.exists() and not path.is_symlink():
            continue  # A tracked deletion is part of the recorded dirty state.
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            target = os.readlink(path)
            if Path(target).is_absolute() or not path.resolve(strict=True).is_relative_to(repo):
                raise ValueError("source symlink escapes the canonical checkout: " + name)
            raw_target = os.fsencode(target)
            result[name] = {"type": "symlink", "target": target,
                            "bytes": len(raw_target), "sha256": hashlib.sha256(raw_target).hexdigest()}
        elif stat.S_ISREG(info.st_mode):
            result[name] = {"type": "file", "mode": stat.S_IMODE(info.st_mode), **identity(path)}
        else:
            raise ValueError("source inventory contains a non-file: " + name)
    for name, expected in result.items():
        if expected["type"] == "symlink":
            destination = (repo / name).resolve(strict=True).relative_to(repo)
            if str(destination) not in result and not any(destination in Path(n).parents for n in result):
                raise ValueError("source symlink target is absent from the captured inventory: " + name)
    return result


def verify_source(root, files, *, derived=False):
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink() or path.is_file()}
    if not derived and actual != set(files):
        raise ValueError("frozen source contains missing or unlisted files")
    for name, expected in files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("manifest contains a non-relative source path")
        path = root / name
        if expected["type"] == "symlink":
            if not path.is_symlink() or os.readlink(path) != expected["target"]:
                raise ValueError("frozen symlink differs: " + name)
        elif path.is_symlink() or not path.is_file() or identity(path) != {k: expected[k] for k in ("bytes", "sha256")}:
            raise ValueError("frozen source differs: " + name)
        elif stat.S_IMODE(path.stat().st_mode) != (expected["mode"] if derived else expected["mode"] & ~0o222):
            raise ValueError("source file permissions differ: " + name)


def freeze(repo, output):
    repo = repo.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all").decode()
    before = source_files(repo)
    for name, expected in before.items():
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if expected["type"] == "symlink":
            destination.symlink_to(expected["target"])
        else:
            shutil.copyfile(repo / name, destination)
            destination.chmod(expected["mode"] & ~0o222)
    verify_source(source, before)
    if (source_files(repo) != before or git(repo, "rev-parse", "HEAD").decode().strip() != head
            or git(repo, "status", "--porcelain=v1", "--untracked-files=all").decode() != status):
        raise ValueError("canonical checkout changed during capture; retained snapshot is not accepted")
    record = {"schema": "tensorfold.onecat-source-v1", "repo": str(repo), "head": head,
              "status": status, "files": before, "source_uid": hashlib.sha256(
                  json.dumps(before, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "scope": "Exact canonical build inputs; dirty status is explicit; no target qualification"}
    publish(output / "source.json", record)
    for directory in sorted((p for p in source.rglob("*") if p.is_dir() and not p.is_symlink()), reverse=True):
        directory.chmod(0o555)
    source.chmod(0o555)
    return record


def frozen(root):
    record = json.loads((root / "source.json").read_text())
    if record.get("schema") != "tensorfold.onecat-source-v1":
        raise ValueError("campaign source manifest schema differs")
    digest = hashlib.sha256(json.dumps(record["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if record.get("source_uid") != digest:
        raise ValueError("campaign source inventory identity differs")
    verify_source(root / "source", record["files"])
    return record


def immutable_image(selected, *, build=False):
    record = json.loads(subprocess.check_output(["docker", "image", "inspect", selected]))
    if len(record) != 1 or not record[0].get("Id", "").startswith("sha256:"):
        raise ValueError("Docker did not resolve one immutable image")
    actual = record[0]
    reference = actual["Id"]
    if build:
        # Docker's local containerd store resolves its named index using BOTH
        # the retained tag and immutable descriptor; a bare ID/repo digest can
        # otherwise trigger a registry lookup even for an existing local image.
        descriptor = actual.get("Descriptor", {}).get("digest")
        tags = actual.get("RepoTags", [])
        if not tags or not descriptor or not descriptor.startswith("sha256:"):
            raise ValueError("local image build requires a named immutable Docker descriptor")
        reference = tags[0] + "@" + descriptor
        resolved = json.loads(subprocess.check_output(["docker", "image", "inspect", reference]))
        if len(resolved) != 1 or resolved[0].get("Id") != actual["Id"]:
            raise ValueError("named image descriptor differs from the selected image")
    return {"requested": selected, "id": actual["Id"], "reference": reference}


def logged(argv, output, timeout):
    started = time.monotonic()
    publish(output / "command.json", {"argv": argv, "timeout_s": timeout})
    failure = None
    code = None
    try:
        with (output / "console.log").open("xb") as log:
            code = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        if code:
            raise subprocess.CalledProcessError(code, argv)
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            publish(output / "command-result.json", {"exit_code": code, "elapsed_s": time.monotonic() - started,
                    "error": None if failure is None else repr(failure), "log": identity(output / "console.log")})
        except BaseException as error:
            if failure is not None:
                raise failure from error
            raise


def container(argv, output, timeout):
    owner = OwnedContainer()
    primary = None
    try:
        owner.create(argv, lambda command, result: publish(output / "create.json", {
            "argv": command, "cid": result.stdout.decode().strip(), "exit_code": result.returncode}))
        logged(["docker", "start", "--attach", owner.cid], output, timeout)
    except BaseException as error:
        primary = error
    finally:
        owner.retire()
    errors = [error for error in owner.errors if error is not None]
    failed = owner.exit_code(primary)
    try:
        publish(output / "container-result.json", {"cid": owner.cid, "state": owner.state, "exit_code": failed,
                "primary": None if primary is None else repr(primary), "cleanup": [repr(e) for e in errors]})
    except BaseException as error:
        errors.append(error)
    raise_failures(primary, errors)
    if failed:
        raise RuntimeError("owned container did not retire successfully")


def validate_wheel(path, source):
    expected_python = {str(p.relative_to(source / "src")) for p in (source / "src/tensorfold").rglob("*.py")}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):
            raise ValueError("wheel has duplicate or escaping members")
        if not expected_python.issubset(names):
            raise ValueError("wheel omits canonical Python modules")
        native = []
        members = {}
        for name in names:
            if not name.startswith("tensorfold/") or name.endswith("/"):
                continue
            data = archive.read(name)
            original = source / "src" / name
            if name.endswith(".so"):
                if name != "tensorfold/_fd_owner.abi3.so":
                    raise ValueError("wheel contains an unexpected prebuilt native module")
                native.append(name)
            elif not original.is_file() or data != original.read_bytes():
                raise ValueError("wheel member differs from canonical source: " + name)
            members[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if native != ["tensorfold/_fd_owner.abi3.so"]:
            raise ValueError("wheel lacks the owned stable-ABI FD extension")
        configuration = tomllib.loads((source / "pyproject.toml").read_text())
        for package, patterns in configuration["tool"]["setuptools"]["package-data"].items():
            prefix = Path(*package.split("."))
            for pattern in patterns:
                for candidate in (source / "src" / prefix).glob(pattern):
                    if candidate.is_file() and str(candidate.relative_to(source / "src")) not in names:
                        raise ValueError("wheel omits declared package data: " + str(candidate))
        for filename in ("THIRD_PARTY_NOTICES.md", "LICENSES/GGML-MIT.txt"):
            matches = [n for n in names if ".dist-info/" in n and n.endswith("/" + filename)]
            if len(matches) != 1 or archive.read(matches[0]) != (source / filename).read_bytes():
                raise ValueError("wheel third-party attribution differs: " + filename)
    return {"wheel": path.name, **identity(path), "members": members,
            "fd_owner_source": identity(source / "src/tensorfold/_fd_owner.c")}


def wheel(root, output, base, cpus, memory, timeout):
    record = frozen(root)
    base_binding = immutable_image(base)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output / "artifacts"
    artifacts.mkdir()
    with tempfile.TemporaryDirectory(prefix="onecat-derived-wheel-", dir=output) as temporary:
        stage = Path(temporary) / "source"
        shutil.copytree(root / "source", stage, symlinks=True)
        for path in [stage, *stage.rglob("*")]:
            if path.is_symlink():
                continue
            if path.is_dir():
                path.chmod(0o755)
            else:
                path.chmod(record["files"][str(path.relative_to(stage))]["mode"])
        argv = ["docker", "create", "--name", "tf-onecat-wheel-" + uuid.uuid4().hex[:12], "--runtime", "runc",
                "--network", "none", "--read-only", "--user", f"{os.getuid()}:{os.getgid()}",
                "--cpus", str(cpus), "--memory", memory, "--memory-swap", memory,
                "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=2g,mode=1777",
                "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "PIP_NO_INDEX=1",
                "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", f"MAX_JOBS={cpus}",
                "-v", str(stage) + ":/src:rw", "-v", str(artifacts) + ":/artifacts:rw", "-w", "/src",
                "--entrypoint", "/opt/venv/bin/python", base_binding["reference"], "-B", "-m", "pip", "wheel",
                "--no-build-isolation", "--no-deps", "--no-cache-dir", "/src", "--wheel-dir", "/artifacts"]
        container(argv, output, timeout)
        verify_source(stage, record["files"], derived=True)
    verify_source(root / "source", record["files"])
    candidates = list(artifacts.glob("tensorfold-*.whl"))
    if len(candidates) != 1:
        raise ValueError("wheel build must produce exactly one TensorFold artifact")
    receipt = validate_wheel(candidates[0], root / "source")
    receipt.update(source_uid=record["source_uid"], base_image=base_binding["id"], base_binding=base_binding)
    (artifacts / "application-wheel.sha256").write_text(receipt["sha256"] + "  " + receipt["wheel"] + "\n")
    publish(output / "wheel.json", receipt)
    return receipt


def image(root, wheel_output, output, base, cpus, memory, timeout):
    record = frozen(root)
    base_binding = immutable_image(base, build=True)
    artifact = json.loads((wheel_output / "wheel.json").read_text())
    if artifact["source_uid"] != record["source_uid"]:
        raise ValueError("wheel and image source captures differ")
    verified = validate_wheel(wheel_output / "artifacts" / artifact["wheel"], root / "source")
    if any(verified[k] != artifact[k] for k in ("wheel", "bytes", "sha256", "members")):
        raise ValueError("wheel changed after verification")
    output.mkdir(parents=True, exist_ok=False)
    argv = ["docker", "build", "--network=none", "--progress=plain", "--platform=linux/amd64",
            "--resource", f"cpu-quota={cpus * 100000}", "--resource", "cpu-period=100000",
            "--resource", "memory=" + memory, "--resource", "memory-swap=" + memory,
            "--build-arg", "APPLICATION_IMAGE=" + base_binding["reference"],
            "--build-context", "application_artifacts=" + str(wheel_output / "artifacts"),
            "--iidfile", str(output / "image-id.txt"), "--metadata-file", str(output / "build-metadata.json"),
            "--file", str(root / "source/deploy/v100/Dockerfile.application-wheel"), str(root / "source")]
    logged(argv, output, timeout)
    if immutable_image(base_binding["reference"])["id"] != base_binding["id"]:
        raise ValueError("base image identity changed during the application build")
    publish(output / "image.json", {"image": (output / "image-id.txt").read_text().strip(),
            "base": base_binding["id"], "base_binding": base_binding,
            "source_uid": record["source_uid"], "wheel_sha256": artifact["sha256"],
            "qualification": False})


_INSTALLED_CHECK = """import hashlib,json,pathlib,subprocess,sys
from tensorfold import _fd_owner
from tensorfold.cuda.sm70 import software_binding
import tensorfold
source=pathlib.Path('/source')
receipt=json.loads(pathlib.Path('/wheel/wheel.json').read_text())
installed=pathlib.Path(tensorfold.__file__).resolve().parent
if installed.is_relative_to(source): raise RuntimeError('checkout runtime import is forbidden')
expected=receipt['members']
actual={}
for p in installed.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
        relative='tensorfold/'+str(p.relative_to(installed))
        data=p.read_bytes()
        actual[relative]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
if actual != expected: raise RuntimeError('installed whole application differs from the verified wheel')
if _fd_owner._source_sha256 != receipt['fd_owner_source']['sha256']:
    raise RuntimeError('native FD source binding differs')
authority=software_binding.issue_source_binding(required=True)
software_binding.verify_source_binding(authority)
subprocess.run([sys.executable,'-m','pip','check'],check=True)
if sys.argv[1] == 'cuda':
    import torch
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,0):
        raise RuntimeError('native component check requires exactly one actual SM70 GPU')
print(json.dumps({'installed_root':str(installed),'files':len(actual),'software_binding':authority.record(),
                  'source_uid':receipt['source_uid'],'scope':'installed application/source binding; not model quality/P3'}))
if len(sys.argv)>2:
    subprocess.run([sys.executable,'-m','pytest','-c','/verification/pytest.ini',
        '-o','pythonpath=/source/tests /source','-p','no:cacheprovider','-q',
        '--maxfail=1','--tb=short','--junitxml=/evidence/tests.xml',*sys.argv[2:]],check=True)
"""


def check(root, wheel_output, output, selected_image, tests, gpu, cache, cpus, memory, timeout):
    record = frozen(root)
    image_binding = immutable_image(selected_image)
    artifact = json.loads((wheel_output / "wheel.json").read_text())
    if artifact["source_uid"] != record["source_uid"]:
        raise ValueError("installed check source/wheel identities differ")
    if not tests or any(not name.startswith("tests/") or ".." in Path(name).parts
                        or not (root / "source" / name).is_file() for name in tests):
        raise ValueError("installed checks require explicit existing source test files")
    output.mkdir(parents=True, exist_ok=False)
    publish(output / "check-inputs.json", {"source_uid": record["source_uid"], "image_id": image_binding["id"],
            "wheel_receipt": identity(wheel_output / "wheel.json"), "gpu": gpu,
            "tests": {name: identity(root / "source" / name) for name in tests}})
    argv = ["docker", "create", "--name", "tf-onecat-check-" + uuid.uuid4().hex[:12],
            "--network", "none", "--read-only", "--user", f"{os.getuid()}:{os.getgid()}",
            "--cpus", str(cpus), "--memory", memory, "--memory-swap", memory,
            "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=8g,mode=1777",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONOPTIMIZE=0", "-e", "OMP_NUM_THREADS=1",
            "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-e", f"MAX_JOBS={cpus}",
            "-e", f"CMAKE_BUILD_PARALLEL_LEVEL={cpus}", "-v", str(root / "source") + ":/source:ro",
            "-v", str(wheel_output) + ":/wheel:ro", "-v", str(output) + ":/evidence:rw", "-w", "/source"]
    if gpu is None:
        argv += ["--runtime", "runc", "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES="]
    else:
        if cache is None:
            raise ValueError("native checks require an explicit caller-owned cache directory")
        cache.mkdir(parents=True, exist_ok=True)
        argv += ["--runtime", "nvidia", "--gpus", '"device=' + gpu + '"', "--ipc", "host",
                 "--ulimit", "memlock=-1:-1", "--ulimit", "stack=67108864:67108864",
                 "-e", "NVIDIA_VISIBLE_DEVICES=" + gpu, "-e", "CUDA_VISIBLE_DEVICES=" + gpu,
                 "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "-e", "TORCH_CUDA_ARCH_LIST=7.0",
                 "-e", "TF_GGUF_ASSETS_DIR=/cache/gguf_assets", "-v", str(cache) + ":/cache:rw"]
    publish(output / "image-binding.json", image_binding)
    argv += ["--entrypoint", "/opt/venv/bin/python", image_binding["reference"], "-B", "-c", _INSTALLED_CHECK,
             "cpu" if gpu is None else "cuda", *("/source/" + name for name in tests)]
    container(argv, output, timeout)
    verify_source(root / "source", record["files"])


_PREPARE_INPUTS = """import hashlib,json,pathlib,sys
from types import SimpleNamespace
from tensorfold import hub
from tensorfold.families.qwen4_exp.cuda.execution_identity import artifact_files
sys.path.insert(0,'/tools')
from bench_concurrent import prepare_matrix
target,mtp,model,profile,inputs,tokens,reps,levels=sys.argv[1:]
prepared=hub.resolve(target,download=False)
engine=SimpleNamespace(model_dir=prepared,method_geometry=SimpleNamespace(method='mtp'),
                       mtp_model_dir=pathlib.Path(mtp),media_profile=None)
files,_=artifact_files(engine)
artifacts={}
for name,path in sorted(files.items()):
    before=path.stat()
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise RuntimeError('model artifact changed during qualification input preparation')
    artifacts[name]={'size':before.st_size,'sha256':digest}
manifest={'schema':'tensorfold.execution-profile-manifest','version':1,'artifacts':artifacts,
          'preparation':{'kernel_family':'sm70-paged-owner-rn-ex2-p2051-v1','graph_family':'uncaptured',
                         'graph_budget_bytes':0,'trial':True,'collective_family':'tp8-allgather-v1','rn_heads_per_block':12}}
matrix=prepare_matrix(prepared,model,profile,[int(n) for n in levels.split(',')],[int(inputs)],int(tokens),int(reps))
for name,value in [('profile.json',manifest),('matrix.json',matrix)]:
    with (pathlib.Path('/evidence')/name).open('x') as out:
        json.dump(value,out,indent=2,allow_nan=False);out.write('\\n')
print(json.dumps({'prepared_metadata':str(prepared),'artifact_count':len(artifacts),
                  'matrix_run_identity':matrix['run_identity'],'scope':'CPU source/content inputs only; no model execution'}))
"""


def installed_check_receipt(directory, source_uid, image_id):
    checked = json.loads((directory / "check-inputs.json").read_text())
    terminal = json.loads((directory / "container-result.json").read_text())
    state = terminal.get("state", {})
    if (checked.get("source_uid") != source_uid or checked.get("image_id") != image_id
            or type(terminal.get("exit_code")) is not int or terminal["exit_code"] != 0
            or terminal.get("primary") is not None or terminal.get("cleanup") != []
            or state.get("Status") != "exited" or state.get("Running") is not False
            or type(state.get("Pid")) is not int or state["Pid"] != 0
            or type(state.get("ExitCode")) is not int or state["ExitCode"] != 0 or state.get("OOMKilled") is not False):
        raise ValueError("campaign requires a successful installed check of this exact source/image")
    return {"root": str(directory), "inputs": identity(directory / "check-inputs.json"),
            "terminal": identity(directory / "container-result.json")}


def inputs(root, output, selected_image, installed_check, models, target, mtp, cache, model_name,
           profile, input_tokens, tokens, repetitions, levels, cpus, memory, timeout):
    record = frozen(root)
    binding = immutable_image(selected_image)
    accepted = installed_check_receipt(installed_check, record["source_uid"], binding["id"])
    models = models.resolve(strict=True)
    for artifact in (target, mtp):
        if (Path(artifact).is_absolute() or ".." in Path(artifact).parts
                or not (models / artifact).resolve(strict=True).is_relative_to(models)):
            raise ValueError("qualification input artifacts must belong to the read-only model mount")
    output.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    argv = ["docker", "create", "--name", "tf-onecat-inputs-" + uuid.uuid4().hex[:12],
            "--runtime", "runc", "--network", "none", "--read-only", "--user", f"{os.getuid()}:{os.getgid()}",
            "--cpus", str(cpus), "--memory", memory, "--memory-swap", memory,
            "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=2g,mode=1777", "-e", "NVIDIA_VISIBLE_DEVICES=void",
            "-e", "CUDA_VISIBLE_DEVICES=", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "TF_GGUF_ASSETS_DIR=/cache/gguf_assets", "-e", "HF_HUB_OFFLINE=1",
            "-v", str(models) + ":/models:ro", "-v", str(cache) + ":/cache:rw",
            "-v", str(root / "source/tools") + ":/tools:ro", "-v", str(output) + ":/evidence:rw",
            "-w", "/tmp", "--entrypoint", "/opt/venv/bin/python", binding["reference"], "-B", "-c", _PREPARE_INPUTS,
            "/models/" + target, "/models/" + mtp, model_name, profile, str(input_tokens), str(tokens),
            str(repetitions), levels]
    container(argv, output, timeout)
    publish(output / "inputs.json", {"source_uid": record["source_uid"], "image_binding": binding,
            "manifest": identity(output / "profile.json"), "matrix": identity(output / "matrix.json"),
            "installed_check": accepted,
            "quality_accepted": False, "P3_accepted": False})
    verify_source(root / "source", record["files"])


def commit_diagnostic_spec(events, nvtx):
    if type(nvtx) is not bool:
        raise ValueError("commit NVTX selection must be an explicit boolean")
    if events is None:
        if nvtx:
            raise ValueError("commit NVTX markers require explicit finite diagnostic events")
        return None
    if __package__:
        from .v100_commit_trace import resources
    else:
        from v100_commit_trace import resources
    budget = resources(events)
    global_bytes = 8 * (budget["retained_event_bytes"] + budget["serialization_bytes"] + 8192)
    if global_bytes > sys.maxsize:
        raise ValueError("commit diagnostic global host obligation exceeds its size domain")
    return {**budget, "nvtx": nvtx, "global_host_obligation_bytes": global_bytes,
            "diagnostic_only": True, "P3_valid": False}


def diagnostic_c1_c8_plan(plan):
    """Reuse exact canonical prompts/work, taking one complete C1/C8 cohort.

    This is an open finite benchmark observer operation, not the closed
    benchmark/trace client and not performance evidence.
    """
    if (type(plan) is not dict or plan.get("schema") != "tensorfold.phase-matrix-plan-v1"
            or type(plan.get("cohorts")) is not list):
        raise ValueError("commit diagnostic requires an existing finite matrix plan")
    selected = []
    warmup = next((row for row in plan["cohorts"] if row.get("warmup") is True
                   and row.get("level") == 1 and row.get("input_tokens") == 1024), None)
    if warmup is None:
        raise ValueError("commit diagnostic requires the existing explicit C1 warmup")
    selected.append(warmup)
    for level in (1, 8):
        row = next((row for row in plan["cohorts"] if row.get("warmup") is False
                    and row.get("level") == level and row.get("input_tokens") == 1024
                    and row.get("tokens") == 128), None)
        if row is None:
            raise ValueError("commit diagnostic requires complete 1024-input/128-output C1 and C8 work")
        selected.append(row)
    # Detach the derived DATA so neither caller nor filtering mutates the
    # original immutable plan. IDs/prefixes/full work hashes are preserved.
    derived = json.loads(json.dumps(plan, allow_nan=False))
    derived.update(levels=[1, 8], inputs=[1024], tokens=128, reps=1, excluded_levels=[],
                   cohorts=json.loads(json.dumps(selected, allow_nan=False)))
    return derived


def model_arguments(settings, plan, *, smoke_only=False, diagnostic_commit_events=None, diagnostic_commit_nvtx=False):
    """One explicit MTP4/graph-enabled arm; numerical acceptance is separate."""
    required = {"gpus", "models_root", "target", "mtp", "profile_manifest", "matrix_plan", "cache",
                "gu", "down", "device_budget_bytes", "global_host_bytes", "max_waiting_input_bytes",
                "prefill_rows", "mtp_confidence", "execution_rows", "graphs", "iq2_source", "prefill_method"}
    if type(settings) is not dict or set(settings) != required:
        raise ValueError("model arm requires the complete explicit campaign settings")
    devices = settings["gpus"]
    if (type(devices) is not list or len(devices) != 8 or len(set(devices)) != 8
            or any(type(value) is not str or not value or "," in value for value in devices)):
        raise ValueError("model arm requires eight distinct physical GPU IDs in rank order")
    for name in ("device_budget_bytes", "global_host_bytes", "max_waiting_input_bytes", "prefill_rows", "execution_rows"):
        if type(settings[name]) is not int or not 0 < settings[name] < (1 << 63):
            raise ValueError("model arm requires explicit bounded positive " + name)
    if settings["gu"] not in ("incumbent", "tensorfold", "tensorfold-fp32-epilogue", "onecat-half-epilogue"):
        raise ValueError("unsupported GU candidate")
    if settings["down"] not in ("tensorfold", "onecat-q8-slots"):
        raise ValueError("unsupported down candidate")
    if settings["prefill_method"] not in ("tensorfold", "candidate"):
        raise ValueError("unsupported target prefill candidate")
    if settings["iq2_source"] not in ("active-frame", "raw") or type(settings["graphs"]) is not bool:
        raise ValueError("explicit IQ2 source and graph policy required")
    if settings["gu"] in ("incumbent", "tensorfold") and settings["iq2_source"] != "active-frame":
        raise ValueError("raw IQ2 requires a OneCat GU candidate")
    if settings["gu"] == "incumbent" and settings["down"] != "tensorfold":
        raise ValueError("incumbent comparison must retain its complete TensorFold GU/down chain")
    if settings["gu"] == "incumbent" and settings["prefill_method"] != "tensorfold":
        raise ValueError("incumbent comparison must retain TensorFold prefill")
    if (type(settings["mtp_confidence"]) not in (int, float)
            or not 0 <= settings["mtp_confidence"] <= 1):
        raise ValueError("explicit MTP confidence must be finite in [0,1]")
    if plan.get("schema") != "tensorfold.phase-matrix-plan-v1" or plan.get("profile") not in ("q8", "q4"):
        raise ValueError("existing phase-matrix plan required")
    capacity = 24 if plan["profile"] == "q8" else 32
    if plan.get("admin_cap") != capacity or plan.get("context") != 262144:
        raise ValueError("matrix personal profile context/capacity differs")
    for name in ("target", "mtp"):
        path = Path(settings[name])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("model artifact must be a relative path under models_root")
    serve = ["/models/" + settings["target"], "--backend", "cuda", "--context", "262144",
             "--parallel", str(capacity), "--kv-full-windows", str(capacity),
             "--kv-dtype", "int8" if plan["profile"] == "q8" else "rotorquant-iso4-sm70-v1",
             "--indexer-history", "gpu", "--expert-tensor-parallel-size", "4",
             "--draft-method", "mtp", "--drafter", "/models/" + settings["mtp"],
             "--mtp-drafts", "4", "--mtp-confidence", str(settings["mtp_confidence"]),
             "--no-prefill-fp8", "--no-thinking", "--no-update-check", "--host", "127.0.0.1",
             "--port", "8000", "--name", plan["model"], "--max-pending-requests", str(capacity),
             "--max-http-connections", str(capacity + 1), "--max-waiting-input-bytes",
             str(settings["max_waiting_input_bytes"]), "--execution-profile-manifest", "/operator/profile.json"]
    if not settings["graphs"]:
        serve.append("--no-graphs")
    qualifier = ["/opt/venv/bin/python", "-B", "/tools/v100_owned_qualification.py", "--gpus", ",".join(devices),
                 "--out", "/evidence/qualification", "--execution-rows", str(settings["execution_rows"]),
                 "--graph-use-receipts"]
    commit = commit_diagnostic_spec(diagnostic_commit_events, diagnostic_commit_nvtx)
    if commit is not None:
        if not smoke_only and (plan.get("levels") != [1, 8] or len(plan.get("cohorts", [])) != 3):
            raise ValueError("commit diagnostic requires the separate finite C1/C8 matrix plan")
        qualifier += ["--commit-trace-events", str(diagnostic_commit_events)]
        if diagnostic_commit_nvtx:
            qualifier.append("--commit-trace-nvtx")
    if not smoke_only:
        qualifier += ["--benchmark-plan", "/operator/matrix.json"]
    if settings["gu"] == "incumbent":
        qualifier += ["--grouped-gate-up-rows", "4"]
    else:
        qualifier += ["--onecat-gu-method", settings["gu"], "--onecat-down-method", settings["down"],
                      "--onecat-iq2-source", settings["iq2_source"],
                      "--onecat-prefill-method", settings["prefill_method"],
                      "--onecat-device-budget-bytes", str(settings["device_budget_bytes"]),
                      "--onecat-global-host-bytes", str(settings["global_host_bytes"])]
    qualifier += ["--", *serve]
    return qualifier


_SMOKE = """import json,pathlib,sys,time,urllib.request
model=sys.argv[1]
records=[]
for prompt,expected in [('What is 2+2? Answer with only the number.','4'),
                        ('Reply with exactly the word READY and nothing else.','READY')]:
    pair=[]
    for drafting in (False,True):
        body={'model':model,'messages':[{'role':'user','content':prompt}],
              'temperature':0,'max_tokens':16,'stream':False,'draft':drafting,'return_token_ids':True}
        request=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=180) as response:
            raw=response.read((1<<20)+1)
        if len(raw)>1<<20: raise RuntimeError('bounded smoke response exceeded')
        result=json.loads(raw)
        pair.append({'request':body,'response':result})
        records.append(pair[-1])
        pathlib.Path('/evidence/smoke-results.json').write_text(json.dumps(records,indent=2)+'\\n')
        if result['choices'][0]['message']['content'].strip()!=expected:
            raise RuntimeError('exact known-answer functional check failed')
    if pair[0]['response']['tensorfold']['token_ids']!=pair[1]['response']['tensorfold']['token_ids']:
        raise RuntimeError('serial/MTP greedy smoke outputs differ')
print(json.dumps({'complete':True,'cases':len(records),'scope':'finite known-answer serial/MTP check; no P3/held-out quality claim'}))
"""


def validate_execution_binding(binding, manifest, *, digest):
    """Match the engine's canonical manifest identity and exact artifact census.

    ``digest`` is execution_identity.digest from the installed application.
    Reused immutable source/library aliases are valid authority observations;
    they do not reduce the required complete model artifact census.
    """
    if type(manifest) is not dict or type(manifest.get("artifacts")) is not dict or not manifest["artifacts"]:
        raise ValueError("readiness requires the actual nonempty execution manifest")
    artifacts = manifest["artifacts"]
    if any(type(value) is not dict or type(value.get("size")) is not int or value["size"] < 0
           for value in artifacts.values()):
        raise ValueError("execution manifest artifact byte sizes are invalid")
    expected_bytes = sum(value["size"] for value in artifacts.values())
    canonical = digest(manifest)
    if type(canonical) is not str or len(canonical) != 64 or any(c not in "0123456789abcdef" for c in canonical):
        raise ValueError("installed execution manifest digest did not return a SHA256 identity")
    if (type(binding) is not dict or binding.get("manifest_sha256") != canonical
            or type(binding.get("content_passes")) is not int or binding["content_passes"] != 1
            or type(binding.get("verified_input_files")) is not int or binding["verified_input_files"] != len(artifacts)
            or type(binding.get("verified_input_bytes")) is not int or binding["verified_input_bytes"] != expected_bytes
            or binding.get("shared_immutable_same_host") is not True):
        raise ValueError("actual execution manifest/content verification differs")
    for name in ("bytes_read_this_binding", "hashes_reused_this_binding"):
        if type(binding.get(name)) is not int or binding[name] < 0:
            raise ValueError("execution binding byte/reuse observations must be nonnegative integers")
    return {"canonical_manifest_sha256": canonical, "verified_input_files": len(artifacts),
            "verified_input_bytes": expected_bytes, "content_passes": 1,
            "bytes_read_this_binding": binding["bytes_read_this_binding"],
            "hashes_reused_this_binding": binding["hashes_reused_this_binding"],
            "scope": "exact startup input binding only; reuse observations do not establish model/quality/P3 acceptance"}


_WAIT_READY = """import errno,json,pathlib,sys,time,urllib.error,urllib.request
sys.path.insert(0,'/tools')
from v100_adaptive_qualification import profile,materialization
from v100_onecat_campaign import validate_execution_binding
from benchmark_output import write_json
from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
deadline=time.monotonic()+float(sys.argv[1])
def read(path):
    with urllib.request.urlopen('http://127.0.0.1:8000'+path,timeout=max(.01,min(2,deadline-time.monotonic()))) as response:
        raw=response.read((16<<20)+1)
    if len(raw)>16<<20: raise RuntimeError('readiness response exceeded existing metadata budget')
    return json.loads(raw)
while True:
    if time.monotonic()>=deadline: raise TimeoutError('HTTP readiness deadline expired')
    try:
        runtime=read('/v1/runtime')
        break
    except urllib.error.URLError as error:
        if not isinstance(error.reason,ConnectionRefusedError) or error.reason.errno!=errno.ECONNREFUSED: raise
        time.sleep(min(.25,max(0,deadline-time.monotonic())))
stats=read('/stats')
write_json(pathlib.Path('/evidence/readiness.json'),{'runtime':runtime,'stats':stats})
effective=profile(stats);materialization(stats,effective)
expected=24 if sys.argv[2]=='q8' else 32
if effective['context_tokens']!=262144 or effective['resident_capacity']!=expected:
    raise RuntimeError('actual full-window residency differs from the matrix profile')
binding=runtime['capacity_plan']['execution_profile_binding']
manifest=json.loads(pathlib.Path('/operator/profile.json').read_text())
validated=validate_execution_binding(binding,manifest,digest=digest)
write_json(pathlib.Path('/evidence/readiness-binding.json'),validated)
"""


def graph_use_receipts(output, source, settings):
    """Require completed serving use, separately from startup capture counters."""
    expected_sources = {"engine": "src/tensorfold/families/qwen4_exp/cuda/engine.py",
                        "observer_tool": "tools/v100_owned_qualification.py"}
    if settings["graphs"]:
        expected_sources.update(graph_controller="src/tensorfold/families/qwen4_exp/cuda/multi_graphs.py",
                                graph_policy="src/tensorfold/families/qwen4_exp/cuda/exact_graphs.py")
    expected_sources = {name: source["files"][path]["sha256"] for name, path in expected_sources.items()}
    selected = {"gu": settings["gu"], "down": settings["down"],
                "iq2_source": "not-applicable" if settings["gu"] == "incumbent" else settings["iq2_source"],
                "prefill_method": settings["prefill_method"]}
    result = []
    for rank in range(8):
        path = output / "qualification" / f"graph-use-rank-{rank}.json"
        if path.stat().st_size > 65536:
            raise ValueError("graph-use record exceeds its observer metadata bound")
        record = json.loads(path.read_text())
        selector = record.get("selector", {})
        graph = record.get("graph_use", {})
        startup = output / "qualification" / f"rank-{rank}.json"
        if (record.get("schema") != "tensorfold.graph-use-rank-v1" or type(record.get("rank")) is not int
                or record["rank"] != rank or record.get("world") != 8
                or record.get("closed") is not True or record.get("failed") is not False
                or record.get("primary_error_type") is not None or record.get("snapshot_error_type") is not None
                or type(record.get("mtp_depth")) is not int or record["mtp_depth"] != 4
                or record.get("mtp_active") is not True or record.get("runtime_sources") != expected_sources
                or record.get("startup_receipt_sha256") != identity(startup)["sha256"]
                or any(selector.get(key) != value for key, value in selected.items())):
            raise ValueError("graph-use record lacks the actual requested source/rank/MTP/retirement binding")
        if settings["gu"] == "incumbent" and selector.get("grouped_gate_up_rows") != 4:
            raise ValueError("incumbent graph receipt differs from the original Grouped4 chain")
        if rank and selector != result[0]["selector"]:
            raise ValueError("graph-use candidate selection differs across ranks")
        if (graph.get("requested") is not settings["graphs"] or graph.get("prepared") is not settings["graphs"]):
            raise ValueError("actual graph request/preparation differs from the selected arm")
        if settings["graphs"]:
            if graph.get("capture_during_serving") is not False:
                raise ValueError("serving graph capture invalidates the prepared execution comparison")
            for name in ("serving_hits", "serving_misses"):
                counts = graph.get(name)
                if (type(counts) is not dict or set(counts) != {"verify", "mtp"}
                        or any(type(n) is not int or n < 0 for n in counts.values())):
                    raise ValueError("per-role serving graph counters are incomplete")
            if any(graph["serving_hits"][role] == 0 for role in ("verify", "mtp")):
                raise ValueError("target and MTP graphs must both replay during actual serving on every rank")
        elif graph.get("serving_hits") is not None or graph.get("serving_misses") is not None:
            raise ValueError("eager arm must not claim graph serving counters")
        result.append({"rank": rank, "file": str(path.relative_to(output)), **identity(path),
                       "selector": selector, "graph_use": graph})
    return result


def commit_trace_receipts(output, source, spec):
    if __package__:
        from .v100_commit_trace import COLUMNS
    else:
        from v100_commit_trace import COLUMNS
    records = []
    for rank in range(8):
        path = output / "qualification" / f"commit-trace-rank-{rank}.json"
        if path.stat().st_size > spec["serialization_bytes"]:
            raise ValueError("commit diagnostic exceeds its admitted encoding bound")
        record = json.loads(path.read_text())
        if (record.get("schema") != "tensorfold.commit-batch-diagnostic-v1" or record.get("rank") != rank
                or record.get("world") != 8 or record.get("closed") is not True or record.get("complete") is not True
                or record.get("failure") is not None or record.get("engine_closed") is not True
                or record.get("engine_aborted") is not False or record.get("diagnostic_only") is not True
                or record.get("P3_valid") is not False or record.get("columns") != list(COLUMNS)
                or record.get("observer_source_sha256") != source["files"]["tools/v100_owned_qualification.py"]["sha256"]):
            raise ValueError("commit diagnostic lacks a positive bound closed-rank outcome")
        requested = record.get("requested", {})
        if any(requested.get(key) != spec[key] for key in
               ("maximum_events", "retained_event_bytes", "serialization_bytes", "nvtx", "global_host_obligation_bytes")):
            raise ValueError("commit diagnostic differs from the admitted finite request")
        events = record.get("events")
        if type(events) is not list or len(events) > spec["maximum_events"]:
            raise ValueError("commit diagnostic event count differs from its finite credit")
        for row in events:
            if (type(row) is not list or len(row) != len(COLUMNS)
                    or any(type(n) is not int or not 0 <= n <= sys.maxsize for n in row)
                    or row[2] < row[1] or row[4] < row[3] or row[5] != row[4] - row[3]):
                raise ValueError("commit diagnostic batch interval/count law differs")
        records.append({"rank": rank, "file": str(path.relative_to(output)), **identity(path),
                        "events": len(events), "committed_delta": sum(row[-1] for row in events),
                        "diagnostic_only": True, "P3_valid": False})
    if not records[0]["events"] or not records[0]["committed_delta"]:
        raise ValueError("leader commit diagnostic observed no actual committed work")
    return records


def cublas_workspace_configuration(selection):
    """Host-only selection of the pinned Torch public workspace policy."""
    if type(selection) is not str or selection not in ("inherit", "on", "off"):
        raise ValueError("cuBLAS workspace cache requires inherit, on or off")
    environment = {} if selection == "inherit" else {"TORCH_CUBLAS_WORKSPACE_CACHE": "1" if selection == "on" else "0"}
    return {"selection": selection, "environment": environment,
            "scope": "unchanged selected image environment" if selection == "inherit" else "explicit diagnostic runtime configuration"}


def model(root, output, selected_image, settings_path, installed_check, cpus, memory, timeout, startup_timeout, smoke_only,
          diagnostic_commit_events=None, diagnostic_commit_nvtx=False, *, cublas_workspace_cache="inherit", diagnostic_graph_pieces=False):
    workspace = cublas_workspace_configuration(cublas_workspace_cache)
    if type(diagnostic_graph_pieces) is not bool:
        raise ValueError("graph piece diagnostic selection must be boolean")
    configuration_diagnostic = cublas_workspace_cache != "inherit" or diagnostic_graph_pieces
    commit = commit_diagnostic_spec(diagnostic_commit_events, diagnostic_commit_nvtx)
    controller_path = Path(__file__).resolve()
    controller_source = {"path": str(controller_path), **identity(controller_path)}
    record = frozen(root)
    image_binding = immutable_image(selected_image)
    accepted = installed_check_receipt(installed_check, record["source_uid"], image_binding["id"])
    if settings_path.stat().st_size > 1 << 20:
        raise ValueError("model settings exceed bounded operator metadata")
    settings = json.loads(settings_path.read_text())
    plan_path = Path(settings["matrix_plan"]).resolve(strict=True)
    if plan_path.stat().st_size > 96 << 20:
        raise ValueError("matrix exceeds the existing qualifier DATA envelope")
    original_plan = json.loads(plan_path.read_text())
    plan = diagnostic_c1_c8_plan(original_plan) if commit is not None and not smoke_only else original_plan
    qualifier = model_arguments(settings, plan, smoke_only=smoke_only,
        diagnostic_commit_events=diagnostic_commit_events, diagnostic_commit_nvtx=diagnostic_commit_nvtx)
    models = Path(settings["models_root"]).resolve(strict=True)
    for name in ("target", "mtp"):
        artifact = models / settings[name]
        if not artifact.is_file() or not artifact.resolve(strict=True).is_relative_to(models):
            raise ValueError("selected model artifact is absent or escapes its read-only mount")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).strip():
        raise RuntimeError("model campaign requires idle GPUs; existing compute contexts remain untouched")
    output.mkdir(parents=True, exist_ok=False)
    operator = output / "operator"
    operator.mkdir()
    if commit is not None and not smoke_only:
        if json.loads(plan_path.read_text()) != original_plan:
            raise ValueError("source matrix changed before diagnostic derivation")
        publish(operator / "matrix.json", plan)
    else:
        shutil.copyfile(plan_path, operator / "matrix.json")
    if json.loads((operator / "matrix.json").read_text()) != plan:
        raise ValueError("matrix plan changed before publication as immutable operator input")
    manifest = Path(settings["profile_manifest"]).resolve(strict=True)
    if manifest.stat().st_size > 16 << 20:
        raise ValueError("execution manifest exceeds bounded operator metadata")
    shutil.copyfile(manifest, operator / "profile.json")
    for item in operator.iterdir():
        item.chmod(0o444)
    operator.chmod(0o555)
    cache = Path(settings["cache"]).absolute()
    cache.mkdir(parents=True, exist_ok=True)
    gpus = ",".join(settings["gpus"])
    argv = ["docker", "create", "--name", "tf-onecat-model-" + uuid.uuid4().hex[:12],
            "--runtime", "nvidia", "--gpus", '"device=' + gpus + '"', "--network", "none", "--read-only",
            "--user", f"{os.getuid()}:{os.getgid()}", "--cpus", str(cpus), "--memory", memory,
            "--memory-swap", memory, "--ipc", "host", "--ulimit", "memlock=-1:-1",
            "--ulimit", "nofile=1048576:1048576", "--ulimit", "core=0:0",
            "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=8g,mode=1777", "-e", "NVIDIA_VISIBLE_DEVICES=" + gpus,
            "-e", "CUDA_VISIBLE_DEVICES=" + gpus, "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            "-e", "TENSORFOLD_NO_UPDATE_CHECK=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "PYTHONUNBUFFERED=1", "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
            "-e", "MKL_NUM_THREADS=1", "-e", f"MAX_JOBS={cpus}", "-e", f"CMAKE_BUILD_PARALLEL_LEVEL={cpus}",
            "-e", "TF_GGUF_ASSETS_DIR=/cache/gguf_assets", "-e", "TENSORFOLD_SM70_PROFILE=reference",
            "-e", "TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN=grouped-v1",
            "-e", "TENSORFOLD_PREFILL_ROWS=" + str(settings["prefill_rows"]),
            *(["-e", "TENSORFOLD_EXACT_GRAPH_PIECE_DIAGNOSTIC=1"] if diagnostic_graph_pieces else []),
            *[item for key, value in workspace["environment"].items() for item in ("-e", key + "=" + value)],
            "-v", str(models) + ":/models:ro", "-v", str(root / "source/tools") + ":/tools:ro",
            "-v", str(operator) + ":/operator:ro", "-v", str(output) + ":/evidence:rw",
            "-v", str(cache) + ":/cache:rw", "-w", "/tools", "--entrypoint", qualifier[0],
            image_binding["reference"], *qualifier[1:]]
    publish(output / "model-inputs.json", {"settings": settings, "matrix": identity(operator / "matrix.json"),
            "manifest": identity(operator / "profile.json"), "image_binding": image_binding,
            "source_uid": record["source_uid"], "qualification": False, "P3_accepted": False,
            "installed_check": accepted,
            "cublas_workspace_cache": workspace, "controller_source": controller_source,
            "diagnostic_graph_pieces": diagnostic_graph_pieces,
            "commit_diagnostic": commit, "diagnostic_only": commit is not None or configuration_diagnostic,
            "P3_valid": False if commit is not None or configuration_diagnostic else None,
            "source_matrix": {"path": str(plan_path), **identity(plan_path)},
            "scope": "one complete candidate arm; matched repetition and quality acceptance are separate"})
    owner = OwnedContainer()
    primary = None
    errors = []
    started = time.monotonic()
    try:
        owner.create(argv, lambda command, result: publish(output / "create.json", {"argv": command,
            "cid": result.stdout.decode().strip(), "exit_code": result.returncode}))
        subprocess.run(["docker", "start", owner.cid], check=True, timeout=60)
        while len(list((output / "qualification").glob("rank-*.json"))) != 8:
            state = json.loads(subprocess.check_output(["docker", "inspect", "--format", "{{json .State}}", owner.cid], timeout=30))
            if not state["Running"]:
                raise RuntimeError("model exited before all eight startup receipts")
            if time.monotonic() - started > startup_timeout:
                raise TimeoutError("eight-rank model startup deadline expired")
            time.sleep(1)
        client = output / "client"
        client.mkdir()
        readiness = output / "readiness"
        readiness.mkdir()
        remaining = startup_timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("original startup deadline expired before HTTP readiness")
        logged(["docker", "exec", owner.cid, "/opt/venv/bin/python", "-B", "-c", _WAIT_READY,
                str(remaining), plan["profile"]], readiness, remaining + 5)
        if smoke_only:
            logged(["docker", "exec", owner.cid, "/opt/venv/bin/python", "-B", "-c", _SMOKE, plan["model"]], client, timeout)
        else:
            logged(["docker", "exec", owner.cid, "/opt/venv/bin/python", "-B", "/tools/bench_concurrent.py",
                    "http://127.0.0.1:8000", plan["model"], "--matrix-plan", "/operator/matrix.json",
                    "--output", "/evidence/matrix-results.json", "--timeout", str(timeout),
                    "--response-mib", "16", "--sse-line-kib", "64"], client, timeout)
            if json.loads((output / "matrix-results.json").read_text()).get("passed") is not True:
                raise RuntimeError("matrix protocol/work/overlap qualification did not pass")
            if commit is not None:
                publish(output / "diagnostic-matrix-result.json", {"diagnostic_only": True, "P3_valid": False,
                    "matrix": identity(output / "matrix-results.json"), "workload": "one canonical warmup then C1/C8,1024input/128output",
                    "scope": "instrumented commit-batch diagnostic; raw matrix rates are not P3 evidence"})
    except BaseException as error:
        primary = error
    finally:
        owner.retire()
    errors.extend(error for error in owner.errors if error is not None)
    graph_evidence = None
    commit_evidence = None
    if primary is None and not errors and owner.exit_code(None) == 0:
        try:
            graph_evidence = graph_use_receipts(output, record, settings)
            if commit is not None:
                commit_evidence = commit_trace_receipts(output, record, commit)
        except BaseException as error:
            errors.append(error)
    if owner.cid is not None:
        try:
            with (output / "server.log").open("xb") as log:
                subprocess.run(["docker", "logs", owner.cid], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        except BaseException as error:
            errors.append(error)
    try:
        verify_source(root / "source", record["files"])
        compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], timeout=30)
        if compute.strip():
            raise RuntimeError("post-run compute contexts remain; no P3 acceptance")
    except BaseException as error:
        errors.append(error)
    try:
        if identity(controller_path) != {key: controller_source[key] for key in ("bytes", "sha256")}:
            raise RuntimeError("campaign controller source changed during the owned model lifetime")
    except BaseException as error:
        errors.append(error)
    failed = owner.exit_code(primary) or bool(errors)
    try:
        publish(output / "model-result.json", {"cid": owner.cid, "state": owner.state,
                "exit_code": int(failed), "primary": None if primary is None else repr(primary),
                "cleanup": [repr(error) for error in errors], "elapsed_s": time.monotonic() - started,
                "graph_use": graph_evidence,
                "cublas_workspace_cache": workspace, "controller_source": controller_source,
            "diagnostic_graph_pieces": diagnostic_graph_pieces,
                "commit_diagnostic": commit, "diagnostic_only": commit is not None or configuration_diagnostic,
                "P3_valid": False if commit is not None or configuration_diagnostic else None,
                "commit_trace": commit_evidence,
                "quality_accepted": False, "P3_accepted": False})
    except BaseException as error:
        errors.append(error)
    raise_failures(primary, errors)
    if failed:
        raise RuntimeError("model campaign did not complete and retire successfully")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("freeze")
    capture.add_argument("--repo", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    for name in ("wheel", "image", "check", "inputs", "model"):
        item = sub.add_parser(name)
        item.add_argument("--source", type=Path, required=True)
        item.add_argument("--out", type=Path, required=True)
        if name in ("wheel", "image"):
            item.add_argument("--base", default=CORE)
        item.add_argument("--cpus", type=int, default=64 if name == "model" else 8)
        item.add_argument("--memory", default="340g" if name == "model" else "32g")
        item.add_argument("--timeout", type=int, default=7200 if name == "model" else 1200)
        if name in ("image", "check"):
            item.add_argument("--wheel", type=Path, required=True)
        if name == "check":
            item.add_argument("--image", required=True)
            item.add_argument("--test", action="append", required=True)
            item.add_argument("--gpu", help="one actual V100 UUID/ordinal; omit for CPU-only runc checks")
            item.add_argument("--cache", type=Path)
        if name == "model":
            item.add_argument("--image", required=True)
            item.add_argument("--settings", type=Path, required=True)
            item.add_argument("--installed-check", type=Path, required=True)
            item.add_argument("--startup-timeout", type=int, default=600)
            item.add_argument("--smoke-only", action="store_true", help="serial/MTP known answers only; no benchmark timings")
            item.add_argument("--diagnostic-graph-pieces", action="store_true",
                              help="startup per-piece signed allocation observations; unsynchronized snapshots, diagnostic only, never P3")
            item.add_argument("--cublas-workspace-cache", choices=("inherit", "on", "off"), default="inherit",
                              help="host-only pinned Torch workspace configuration; explicit values are diagnostic, never P3")
            item.add_argument("--diagnostic-commit-events", type=int,
                              help="finite commit-batch event credit; smoke or C1/C8 diagnostic only, never P3")
            item.add_argument("--diagnostic-commit-nvtx", action="store_true",
                              help="numeric NVTX commit markers; requires explicit finite diagnostic event credit")
        if name == "inputs":
            item.add_argument("--image", required=True)
            item.add_argument("--installed-check", type=Path, required=True)
            item.add_argument("--models", type=Path, required=True)
            item.add_argument("--target", required=True)
            item.add_argument("--mtp", required=True)
            item.add_argument("--cache", type=Path, required=True)
            item.add_argument("--name", required=True)
            item.add_argument("--profile", choices=("q8", "q4"), default="q8")
            item.add_argument("--input-tokens", type=int, default=1024)
            item.add_argument("--tokens", type=int, default=128)
            item.add_argument("--reps", type=int, default=3)
            item.add_argument("--levels", default="1,2,4,8,16,24")
    args = parser.parse_args(argv)
    if args.command != "freeze" and (args.cpus <= 0 or args.timeout <= 0):
        parser.error("CPU count and command timeout must be positive")
    if args.command == "freeze":
        freeze(args.repo.absolute(), args.out.absolute())
    elif args.command == "wheel":
        wheel(args.source.absolute(), args.out.absolute(), args.base, args.cpus, args.memory, args.timeout)
    elif args.command == "image":
        image(args.source.absolute(), args.wheel.absolute(), args.out.absolute(), args.base,
              args.cpus, args.memory, args.timeout)
    elif args.command == "check":
        check(args.source.absolute(), args.wheel.absolute(), args.out.absolute(), args.image, args.test,
              args.gpu, None if args.cache is None else args.cache.absolute(), args.cpus, args.memory, args.timeout)
    elif args.command == "inputs":
        inputs(args.source.absolute(), args.out.absolute(), args.image, args.installed_check.absolute(), args.models, args.target, args.mtp,
               args.cache.absolute(), args.name, args.profile, args.input_tokens, args.tokens, args.reps,
               args.levels, args.cpus, args.memory, args.timeout)
    else:
        model(args.source.absolute(), args.out.absolute(), args.image, args.settings.absolute(), args.installed_check.absolute(),
              args.cpus, args.memory, args.timeout, args.startup_timeout, args.smoke_only,
              args.diagnostic_commit_events, args.diagnostic_commit_nvtx,
              cublas_workspace_cache=args.cublas_workspace_cache,
              diagnostic_graph_pieces=args.diagnostic_graph_pieces)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
