# Additive ModelOpt calibration role (candidate)

This is a producer-only extension of verification image
`sha256:2a772b064f593ae9a4436562c4c26ceaa60cec55709992cbf966b74b16bc649c`.
It does not install packages on the host, replace the serving image, or modify
the original foundation manifest, bootstrap, prefixes, provider selection or seal.
The historical upstream-only producer image `91bc` passed CPU API admission,
including namespace, dependency, file-seal and provider audits. Its guarded GPU
probe later reached the Hessian pass and failed on Triton 3.9's removed block
pointer API. Those results do not qualify the new owned source candidate below.
The candidate's clean source wheel, producer image, CUDA/JIT execution, full
calibration, exported-model compatibility and quality/performance remain
**not yet verified**. See [current evidence](EVIDENCE.md).

The current pin is owned fork commit
`a851b028ecbd5f843eb1fc05b93cc2d0abed0de3`, distributed in the
[TensorFold-hosted incremental bundle](modelopt-fork/modelopt-triton39.bundle)
with SHA256 `adfee7d91e371c90eebe8c4da33c4eb1572760322187d8f67662990c34a9221d`.
The commit identifies that bundle's fork history, not a commit claimed to exist
on NVIDIA's or TensorFold's remote Git history. Its NVIDIA upstream is
`e352caca3c36236fc15adea2627a9d42788380c4`; the generated
[source record](modelopt-fork/source.json) and [reconstruction instructions](modelopt-fork/README.md)
preserve the source, patch and local signature scope. The compatibility change
is a source candidate, **not GPU qualified**. Base dependencies retain genuine
upstream metadata. Its HF extra supports Transformers `>=5.5,<5.19`, including
this baseline's `5.17.0`; stable ModelOpt 0.47 excludes 5.17. The full HF extra
also adds unrelated DeepSpeed/diffusion dependencies, so this role initially
requests the base wheel only. Actual quantization and unified HF export imports
are a required smoke gate; additional genuinely required dependencies must be
resolved explicitly, not hidden behind import fallbacks.

## Prepare native metadata before installation

The operator runs all commands in the **exact base image**, CPU only, with no
NVIDIA devices, the original `foundation_guard.py run` CPU activation, network
only for metadata/wheel acquisition, bounded memory/no swap and a two-CPU quota.
Preserve the host's existing affinity policy and record effective affinity:
this GB10's Docker enforcer selects `5-9,15-19` even when another set is requested.
The original runtime's constraints remain in force. The source checkout/archive
and built wheel are operator-provided immutable inputs, not an editable copy of
TensorFold. Preserve the exact archive SHA256, source commit, wheel SHA256,
build-tool versions/command/log and actual checkout-derived wheel version.
Do not invent a release number or edit ModelOpt metadata. Its setuptools build
requires `setuptools>=80` and `setuptools-scm>=8`; missing build dependencies
need their own constrained, hashed isolated build-tool solution. Do not install
them in the sealed baseline or use build isolation that silently changes Torch.

This exact base contains an older TensorFold application. The initial smoke
tests ModelOpt APIs without claiming the new producer pipeline is installed.
Before building a runnable new TensorFold producer, update the owned application
wheel first through `Dockerfile.application`, retain its image/source identity,
then capture a new baseline and explicitly admit that image in this role's
versioned base pin. Updating TensorFold after the producer seal invalidates its
baseline. A source-mounted development run must be labeled separately.

### Bootstrap the wheel builder

The base already has setuptools81 and wheel0.47. The missing build requirement
is setuptools-scm; installing the optional `build` frontend is unnecessary.
First capture the baseline with `role.py snapshot --output /work/baseline`. In that same native
CPU environment run the full binary-only resolver (no installation):

```bash
python3 -m pip install --dry-run --only-binary=:all: \
  --report /work/build-tools-resolver.json \
  --constraint /work/baseline/constraints.txt 'setuptools-scm>=8'
```

Acquire those report-selected wheels and write source provenance containing the
pinned ModelOpt `source_commit` and actual `source_archive_sha256`. Generate
the same independently audited role with `role.py prepare --kind build-tools`
and all five preparation flags below. This explicit kind permits only
setuptools-scm>=8 and its reviewed vcs-versioning dependency; any further
dependency requires a concrete review. The actual 2026-10-31 native report
selects setuptools-scm10.3.4 and vcs-versioning2.6.0. The latter's packaging>=26.2
is satisfied by unchanged baseline26.3. Hashes come from that actual report;
these descriptions do not substitute for the generated lock.

Use the generated build-tools context with the same Dockerfile/named context
and `--target build-tools`. Its CPU smoke imports public backend APIs without
building source or importing ModelOpt.

The original base has **no Git executable**. A successful Python backend import
does not qualify it for SCM source building. Source-wheel execution remains
blocked until the build-tools-only native Git addon is explicitly reviewed.
Use authenticated Ubuntu24.04/ARM64 metadata, simulate no-recommends/no-upgrade,
reject any proposed replacement of an installed package, and retain the exact
new DEB closure (Depends/Pre-Depends, architecture, versions, URLs and SHA256).
Materialize only reviewed artifacts offline; preserve every old dpkg version,
SDK/provider/Python baseline and seal added files/links. Git does not belong to
the final ModelOpt producer image. No host binary mount or SCM override is a
replacement for this prerequisite.

The read-only archive inspection command is:

```bash
python3 /producer-tools/git_os.py plan \
  --metadata /work/git-metadata-v1 --debs /work/git-debs-v1 \
  --acquisition /work/git-deb-acquisition-v1.json \
  --manifest /work/git-os-plan-v1.json
```

Execute it in the exact native container; it writes only the report and temporary
container inspection files. Review the complete control scripts, dependencies,
ELF SONAME/NEEDED entries, symlinks and existing-path collisions before install.
The report binds its inspector source SHA. The actual14-artifact report contains
1,273 payload entries and27 ELF files with no old non-directory collisions.
Expected maintainer effects include the shared `ldconfig` cache; original SDK
provider files and package versions remain protected independently.

The approved Git context contains `manifest.json` (that exact reviewed plan)
and `debs/`. Add these arguments **only** to the build-tools image command:

```bash
--target build-tools --build-arg CALIBRATION_PARENT=git-sdk \
--build-context git-inputs=/ABSOLUTE/REVIEWED_GIT_CONTEXT
```

The `git-sdk` stage materializes the14 artifacts offline through dpkg, checks
every old package's version/configuration status, and seals added files/links.
Both the pre/post SDK audits and complete configuration are required; a failed
install never publishes an OS seal. Python tools and dependency prefixes are
sealed afterward. The producer's default parent is `sdk`, so it inherits no Git.
The prerequisite checker verifies the Git OS seal before executing Git itself.

The base's existing dpkg configuration excludes man pages, translations and
most documentation, while reincluding copyright and `changelog.*`. Schema2
inspection pins the exact configuration-file hashes and those five ordered
rules. It declares every potentially filtered non-runtime path before install;
the OS seal records each actual omission and its rule. Missing executables,
ELF libraries, helpers, conffiles or reincluded documentation remain errors.
Present optional symlinks are sealed normally. No baseline filter is changed.
This follows dpkg's documented last-match path-filter semantics; dpkg may retain
extra directory/symlink entries to support included children.
[Ubuntu24.04 dpkg path filters](https://manpages.ubuntu.com/manpages/noble/man1/dpkg.1.html).

Before the expensive role audits/source build, run the stdlib-only prerequisite
check in the same CPU environment/user as the forthcoming build:

```bash
python3 /opt/tensorfold-calibration/tools/wheel_preflight.py \
  --source /source/Model-Optimizer --output /work/new-wheel-output
```

It checks the selected Git path and real version, source and Git-metadata
permissions, and the new output path's writable parent. It creates no output
directory and imports no build backend. The build consumer repeats the checks
at use and imports Setuptools only after them; it records the actual Git path
and version. A preflight pass is not a source/build/SCM-version qualification.

After prerequisite/artifact review and D13/D14 admission, the root operator may
run the sealed source-wheel consumer in CPU mode:

```bash
python3 /opt/tensorfold-calibration/tools/role.py run \
  /opt/tensorfold-calibration/tools/build_wheel.py \
  --source /source/Model-Optimizer --output /work/new-wheel-output \
  --source-archive-sha256 ACTUAL_ARCHIVE_SHA256
```

Execute that command through `foundation_guard.py run` with no NVIDIA devices,
bounded CPU resources and no network. Use the checkout owner's UID/GID and
writable user-owned output; keep Git's ownership checks intact. The source must
be the clean exact pinned Git checkout with sufficient history/tags, reconstructed
from the recorded upstream and owned bundle. Normal backend generated outputs are
allowed; uncommitted tracked changes and SCM pretend-version
overrides are rejected. The public PEP517 backend executes in the same managed
interpreter, retaining the append-only build-tools prefix. The wheel version is
whatever genuine setuptools-scm derives. Output `source.json` records the real
wheel version/hash, exact fork commit, upstream/bundle provenance and build-tool
versions for the subsequent producer solution.
Both static build requirements and the public backend's dynamic
`get_requires_for_build_wheel()` requirements must already be satisfied; no
implicit installation is allowed. Backend code execution is root-reviewed and
network-isolated; CPU API imports alone do not prove absence of compilation.
This build-tools image is not the final producer image and does not become a new
dependency baseline: producer dependencies are resolved against the original
exact baseline, using the built wheel as input.

The owned workflow is staged so a dependency conflict fails before installation:

1. Mount the canonical `deploy/gb10/calibration` read-only as `/producer-tools`
   and an empty writable evidence directory as `/work`.
   Run `python3 /producer-tools/role.py snapshot --output /work/baseline`, or reuse
   the exact unchanged baseline snapshot already captured for the builder.
2. With the source-pinned ModelOpt wheel in `/work/wheels`, run the full resolver:
   `python3 -m pip install --dry-run --report /work/resolver.json
   --only-binary=:all: --constraint /work/baseline/constraints.txt /work/wheels/EXACT_MODELOPT.whl`.
   Do not use `--no-deps`, `--ignore-installed`, `--upgrade`, or fake platform
   metadata in this resolution. Keep the full report and failure log.
3. Acquire each report-selected wheel exactly (no sdist dependency builds),
   verify the report's SHA256 and wheel tags, and retain it in `/work/wheels`.
   Write `/work/source.json` with `source_commit`, `source_archive_sha256`,
   `wheel_sha256` and the build provenance above. No unrecorded network inputs.
4. Generate the role from the actual report, not handwritten package pins:

   ```bash
   python3 /producer-tools/role.py prepare \
     --baseline /work/baseline/baseline.json --report /work/resolver.json \
     --wheels /work/wheels --provenance /work/source.json --output /work/generated
   ```

The generator validates the complete active dependency closure, including edges
satisfied by unchanged baseline packages and transitively requested extras.
The generated manifest records baseline-satisfied versions separately from new
distributions. Local artifacts are permitted only for the source-pinned ModelOpt
wheel. All other origins require HTTPS, wheel hashes and real native resolution.

## Build and API admission

After D13/D14 review of the actual source, report and generated artifacts,
assemble a named context containing `manifest.json`, `requirements.lock` and
`wheels/`, then the root operator may build:

```bash
set -euo pipefail
TF_CALIBRATION_BASE=local/tensorfold-e2b-verification-base:20261030
TF_CALIBRATION_BASE_ID=sha256:2a772b064f593ae9a4436562c4c26ceaa60cec55709992cbf966b74b16bc649c
test "$(docker image inspect --format '{{.Id}}' "$TF_CALIBRATION_BASE")" = "$TF_CALIBRATION_BASE_ID"
# Use the inspected default docker driver; a separate docker-container builder
# does not inherit this engine's local image store. Do not retag concurrently.
docker buildx build --builder default --pull=false --load --network=none \
  --build-arg CALIBRATION_BASE_IMAGE="$TF_CALIBRATION_BASE" \
  --build-context producer-inputs=/ABSOLUTE/GENERATED_INPUT_CONTEXT \
  -f deploy/gb10/Dockerfile.calibration -t local/tensorfold-calibration:candidate .
test "$(docker image inspect --format '{{.Id}}' "$TF_CALIBRATION_BASE")" = "$TF_CALIBRATION_BASE_ID"
```

Retain both inspected IDs and the complete build log. The tag is a local lookup
handle; it is not an invented registry digest. Its actual config ID must match
the fixed baseline before and after the build. The same checks and
`--pull=false` apply when selecting `--target build-tools`. Missing/mismatched
images, an unavailable selected builder or failed audits stop qualification.
The generated baseline payload hashes and original full SDK/provider audits
remain mandatory; local tag checks alone are not stack verification.

Installation uses `--no-deps` only to materialize the **already fully resolved,
independently checked**, hash-locked addition set offline. Both preinstallation
wheel validation and postinstallation union checks are mandatory. Baseline
packages cannot be reinstalled, shadowed or upgraded. `.pth`, site customization,
and bundled replacement CUDA/driver/C++ provider libraries are rejected. Only
the named Python ModelOpt/NVML binding distributions have an explicit exception
to the conservative `nvidia-*` package-name rule; they receive the same payload
checks. An independent file-content seal covers the complete new prefix; the
old foundation audit still covers exactly its original surfaces.
The independent seal also covers the role's executable tools and requirements
lock. Consumer script arguments are forwarded literally after the script path.

The CPU smoke imports quantize/export and the selected NVFP4 recipes, checks
effective combined dependencies, verifies mapped providers, and requires zero
CUDA initialization. It loads no model and executes no NVFP4 GEMM/calibration.
ModelOpt's optional real-GEMM TensorRT-LLM backend is not an implicit dependency
or a promised capability. Any ModelOpt JIT operator later needs separately
reviewed exact source/cache/compiler identity, bounded workspace and original
qualified SDK/provider proof before execution.

The protected startup mapping supplies the effective compiler/cache paths and
overrides raw cache environment requests. Isolate producer caches with a
dedicated host backing mounted at `/cache`, preserving that logical namespace.
Record compiled source, compiler flags, binary hashes, effective paths and
provider checks. A build-body receipt is not successful final qualification:
require the enclosing role's final audits and exit status as well. The bounded
FP8 helper has passed CPU build/load checks; its CUDA numerical execution is
still a separate gate.

For future GPU producer execution, preserve the **original** CUDA entrypoint →
NVIDIA entrypoint → `foundation_guard.py checked-run` chain, and pass
`python3 /opt/tensorfold-calibration/tools/role.py run SCRIPT` as the final
consumer. Never substitute a bare Python entrypoint. The producer appends its
prefix only after managed foundation activation. Actual weights, calibration
data, resource guard, recipe/export coverage and quality gates remain separate.

The consumer owns a temporary import namespace. Sealed Setuptools81 normally
appends its `_vendor` fallback; that exposes its bundled wheel0.46.3 metadata
alongside installed wheel0.47.0 without changing the effective public import.
The role allows only that exact append after the selected paths and checks the
loaded origins of the five adopted public build APIs. A public `wheel` or
`packaging` alias to vendored code remains an error. On every outcome the role
restores the entire original `sys.path` before independent package/file/provider
audits; it does not discard distinct-distribution errors by package name.
An actual fresh base-container control verified original public wheel0.47.0
and packaging module paths and the namespace restoration. It does not replace
the subsequent two-wheel role-image smoke or actual source-wheel build gates.
