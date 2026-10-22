# Managed CUDA foundation

The Docker build requires an immutable official NGC CUDA development base and
the selected genuine official nightly wheels. The selected base is CUDA13.4.3 cuDNN development. Official AMD64 roots use
Torch20261017, TorchVision20261007 and Tritonfc8; ARM64 uses its distinct official
Torch20261014/TorchVision20261006 artifacts. The selected pin manifest names
actual whole-wheel hashes. Native ARM image audits, device/provider checks and
the [GB10 bd1 model protocols](../../docs/gb10-nvfp4-validation.md) passed with
those recorded inputs; final composed-source gates remain separate. The base supplies the CUDA
headers, compiler, tools, cuDNN and NCCL. The selected nightly Torch/Triton and
all their Python dependency versions belong to a separate protected prefix,
`/opt/tensorfold-foundation`; TensorFold, application dependencies and test
packages belong to `/opt/tensorfold-venv`.

This protection covers ordinary managed installation and startup. It does not
sandbox a root user deliberately modifying files or bypassing managed installers.
A refresh must retain actual upstream dependency metadata. The official binary
Torch nightly's CUDA wheel requirement must not be removed merely to install
against a newer SDK. Use the genuine official nightly wheel and a verified compatible official base.
Any required vendor source change needs an explicit decision before patching.

## Prepare the tracked selected inputs

An ordinary build uses the six selected files already tracked for its architecture;
it does not need a new resolver report or a private wheelhouse. The pin JSON is
the authority for immutable base references, official wheel identities and the
two required native runtime addons. The existing `platform_pins.py` mapping
selects its companion locks, constraints and manifest.

Run the following from the repository root in Bash with Git, Python3.11+, curl,
GNU coreutils and Docker Buildx available. Choose `amd64` for a native AMD64
builder or `arm64` for a native ARM64 builder. A client can use a remote native
builder; the platform flag alone does not establish native hardware execution.
For a remote builder, set `TF_BUILD_JOBS` from that builder's available CPU count,
rather than the client's `nproc`. Compiler parallelism does not prove all cores
stay busy.

This prepares minimal named contexts outside the checkout and prints their
retained path before any metadata or network step can fail. The addon URLs and
whole-file hashes are read from the selected pin JSON; do not copy their values
into another configuration file.

```bash
set -euo pipefail

TF_BUILD_ARCH=amd64       # Use arm64 with the corresponding native builder.
TF_BUILD_JOBS="$(nproc)"  # Available CPUs for this local native builder.
TF_BUILD_INPUTS="$(mktemp -d -t tensorfold-foundation.XXXXXXXX)"
printf 'Foundation build inputs: %s\n' "$TF_BUILD_INPUTS"
TF_FOUNDATION_PINS="$TF_BUILD_INPUTS/pins"
TF_FOUNDATION_ARTIFACTS="$TF_BUILD_INPUTS/artifacts"
mkdir -p "$TF_FOUNDATION_PINS" "$TF_FOUNDATION_ARTIFACTS"

# Copy the six existing selected files using their maintained filename mapping.
# The same pin JSON supplies the base reference and both exact addon identities.
python3 - "$TF_BUILD_ARCH" "$TF_FOUNDATION_PINS" \
  "$TF_BUILD_INPUTS/base-image.txt" > "$TF_BUILD_INPUTS/native-addons.tsv" <<'PY'
import json
from pathlib import Path
import shutil
import sys

source = Path("deploy/gb10").resolve()
sys.path.insert(0, str(source))
from platform_pins import platform_pins

target = platform_pins(sys.argv[1])
destination = Path(sys.argv[2])
for field in ("pins", "foundation", "runtime", "verification", "constraints", "manifest"):
    filename = getattr(target, field)
    shutil.copyfile(source / filename, destination / filename)
pins = json.loads((destination / target.pins).read_text())
Path(sys.argv[3]).write_text(pins["base"] + "\n")
for addon in pins["native_addons"]:
    print(addon["artifact"], addon["url"], addon["sha256"], sep="\t")
PY

git rev-parse HEAD > "$TF_BUILD_INPUTS/source-revision.txt"
sha256sum "$TF_FOUNDATION_PINS"/* > "$TF_BUILD_INPUTS/pin-inputs.sha256"

# Download only the two addon records declared by the current selected pins.
# A canonical .deb name is published only after the complete checksum succeeds.
while IFS=$'\t' read -r TF_DEB_NAME TF_DEB_URL TF_DEB_SHA; do
  TF_DEB_TMP="$(mktemp "$TF_FOUNDATION_ARTIFACTS/.${TF_DEB_NAME}.XXXXXXXX")"
  curl --fail --show-error --location --proto '=https' --proto-redir '=https' \
    --output "$TF_DEB_TMP" "$TF_DEB_URL"
  printf '%s  %s\n' "$TF_DEB_SHA" "$TF_DEB_TMP" | \
    sha256sum --check | tee -a "$TF_BUILD_INPUTS/native-addon-checks.log"
  mv -- "$TF_DEB_TMP" "$TF_FOUNDATION_ARTIFACTS/$TF_DEB_NAME"
done < "$TF_BUILD_INPUTS/native-addons.tsv"

TF_BASE_IMAGE="$(cat "$TF_BUILD_INPUTS/base-image.txt")"
TF_SOURCE_REVISION="$(cat "$TF_BUILD_INPUTS/source-revision.txt")"
export TF_FOUNDATION_PINS TF_FOUNDATION_ARTIFACTS TF_SOURCE_REVISION TF_BUILD_JOBS
```

A failed transfer or checksum stops the shell before a canonical `.deb` name is
published. Keep the printed input directory and its checksum/acquisition records
for diagnosis. The image independently verifies the addon hashes and Debian
metadata. Dev/static, MAGMA and gfortran source-build packages are not required
by the selected official-wheel profile. Its complete original upstream Python
dependency metadata remains intact and is checked during the image build.

The preparation/acquisition excerpt was exercised on AMD64 at source
`16ae6dfc`; independent checks matched all six copied inputs and both complete
DEBs against the selected metadata. That component check excluded Docker build,
package installation, GPU and model execution. The complete cold-build procedure
and fresh foundation qualification have their separate gates.

## Refresh selected dependencies

An ordinary checkout uses the tracked selected files above. For an intentional
refresh, resolve on native Linux CPython3.12 for the selected architecture. Use the actual
immutable NGC child/index identity and exact official Torch, TorchVision and
Triton wheel identities. Pin schema2
records the immutable base, target, distinct cache key, explicit
`foundation_roots` with required extras, and exact whole-wheel hashes.

A pip dry-run report must include the actual selected full application
and verification dependency closure. Binary wheel acquisition remains official
HTTPS, including the official `download-r2.pytorch.org` redirect host; local
file wheel reports are refused.
Run the maintained generator with `packaging` installed:

```bash
python3 deploy/gb10/lock_dependencies.py --architecture amd64 \
  --pins generated-amd64/nightly-pins-amd64.json \
  --report generated-amd64/resolver.json --output-directory generated-amd64
```

Use `--architecture arm64` with its actual ARM64 wheel pins. Supported
`--cross-report` generation preserves the real AMD64 resolver environment and
checks target ARM64 markers/tags/closure; it is metadata validation, not ARM64
execution. A native ARM64 report can instead be generated on that host. The
outputs are separate foundation/runtime/test locks, exact protected constraints
and a provenance manifest. Do not hand-edit generated locks. The selector
validates the base and every relationship between pins, manifest and all locks
before publishing the native selection. Earlier schema1 data is archived byte-for-byte under [history/20261006](history/20261006/README.md); it is
evidence and a test oracle, not a deployment fallback.
A cross-architecture resolver report does not qualify native execution.

Put any explicitly declared native add-on `.deb` artifacts and acquisition
receipts in a minimal regular-file artifact directory.
`native_addons` records each artifact's SHA256, Debian package and exact version.
Only these artifacts are installed; missing Debian dependencies fail the build
rather than resolving another toolkit. Add-ons cannot replace installed base
packages. ARM64 remains a supported native build target; its image requires
actual official ARM64 wheel hashes and a validated native or explicitly cross-resolved report. An AMD64 wheel
or invented hash cannot qualify it.

## Build and audit

After the preparation block, keep the exported variables in the same shell.
Both input directories are explicit Docker named contexts; they contain no
model weights or private source checkout. This example builds the verification
target; use `--target runtime` and your own tag for serving:

```bash
docker buildx build --platform "linux/$TF_BUILD_ARCH" --load --target verification \
  --build-context "foundation_pins=$TF_FOUNDATION_PINS" \
  --build-context "foundation_artifacts=$TF_FOUNDATION_ARTIFACTS" \
  --build-arg "BASE_IMAGE=$TF_BASE_IMAGE" \
  --build-arg "SOURCE_REVISION=$TF_SOURCE_REVISION" \
  --build-arg "MAX_JOBS=$TF_BUILD_JOBS" \
  --tag tensorfold-fork:verification --file deploy/gb10/Dockerfile .
```

The optional `compose.build.yaml` overlay supports either maintained service
profile. Set `TF_FOUNDATION_PINS`, `TF_FOUNDATION_ARTIFACTS` and
`TF_SOURCE_REVISION`, then use, for example:

```bash
docker compose -f deploy/discrete-gpu/compose.yaml \
  -f deploy/gb10/compose.build.yaml build
```

Before any apt/Python preparation, the image records the original native SDK
package versions, file SHA256s and symlink targets, and holds those packages.
The builder checks this baseline after native add-ons and dependency installation.
It seals protected Python files, native providers, compiler dependencies and
artifact identities. App and test installers accept only their disjoint generated
locks, install with hashes and `--no-deps`, and perform full before/after content
audits. The app wheel receives a separate owned-package/payload check. `pip check`
continues to validate genuine upstream dependency metadata.
The C++ runtimes come from the protected compiler-package inventory; private
copies bundled with SDK tools remain protected but cannot supply that runtime.

`nccl_provider` explicitly selects `native` (default) or protected
`nvidia-nccl-cu13`. The image binds `TF_NCCL_LIB` to that physical library,
refuses a conflicting override, verifies its native version against the exact
nightly dependency when present, and refuses a second physical NCCL provider.
An older base NCCL cannot silently satisfy a newer nightly requirement.

The app venv publishes selected tools and cache identity from a compact owned
startup mapping before native imports, including direct Python/test commands.
The managed launcher additionally checks package origins/versions, file
mode/size/mtime/realpath, original SDK aliases and actual loaded CUDA providers.
It loads the required TorchVision native/image extensions and verifies their
registered CUDA image schemas before checking provider maps; this does not
execute GPU JPEG decoding. It validates the effective Driver API in fresh children and uses the base's
forward-compatibility libraries only after a successful actual-device probe.
NVIDIA's entrypoint still executes; final native-provider proof runs after its
environment transformations. A stale compatibility marker grants no admission.
The final consumer interpreter preloads the sealed core libraries with strong
process-lifetime owners before Torch; the supported `TORCH_USE_RTLD_GLOBAL`
selection avoids its wheel-provider preload. Preparatory launcher/driver/compiler
interpreters skip these native preloads. Only exact official TorchVision RECORD
members for bundled CUDART/NVJPEG are allowed as additional physical providers
under the approved no-override exception. CUDA bindings' static bundled CUDART
is recorded separately. The graph therefore does not claim every component uses
only the base runtime. Triton's supported compiler/tool/libdevice overrides are checked against actual
resolved providers. An owned compiler wrapper puts the selected SDK headers
first without changing Triton source.

Full file hashes are construction/managed-install/explicit-audit work. Startup
stat checks assume the managed files remain immutable; they are not a security
proof against a writer restoring timestamps. All caches include the sealed
runtime, package, native-provider, compiler and tool/header identities plus the
actual installed TensorFold wheel SHA256. A changed
foundation requires a fresh namespace and requalification.

Before sealing, the build normalizes owned/copied regular-file timestamps to
whole seconds, the precision retained by the verified image-layer export.
Owned JSON publications use the same preparation. Original SDK files and alias
targets are left unchanged. Startup still compares exact nanosecond timestamps;
it does not round a changed file's timestamp or relax content-hash audits.

CPU installed validation deliberately bypasses device admission while retaining
foundation activation. Override pytest's development `src` path so the installed
wheel is the package under test:

```bash
docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint python3 tensorfold-fork:verification \
  /opt/tensorfold-pins/foundation_guard.py audit
docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint python3 tensorfold-fork:verification \
  /opt/tensorfold-pins/foundation_guard.py run -- \
  python3 -m pytest -o pythonpath=tests
```

Native GPU validation uses the normal compatibility entrypoint and the same
`-o pythonpath=tests` override. Qualification must separately retain source,
image, actual provider/tool, native math/model/API and startup-cost receipts.
The foundation refresh changes dependencies, so historical results do not
establish its native/model or performance behavior. Actual latest image and
ARM64 hardware qualification remain separate gates until recorded.

`tensorfold update` refuses in-place mutation in a managed image. Rebuild with
the selected audited foundation inputs to update the application; unmanaged
editable/developer installations retain their existing update behavior.


## Prepared native bundle

The managed image implements a five-provider preparation candidate: the original
checkpoint packing/GEMM module, plain BF16/FP16 projection, and shared GDN,
prefill-attention and QMM modules. It extends the separate two-provider baseline;
that baseline's evidence does not qualify these additional prepared libraries.
The five-provider source/API/packaging cohort passed 342 focused CPU tests in
17.85 seconds on pinned AMD64 image `509b4135`, using four CPUs, 16 GiB and no GPU. Its 1,310
source/test files matched before/after and the declared read-only source;
21 actual project origins were checked. The controls use genuine scratch
same-version pip installation and controlled compiler/import callbacks, not
actual prepared CUDA compilation or binding import. Those target compile,
new installed-image, ARM64/native GPU and full-model performance gates remain
pending. No numerical C++/CUDA source or serving flag is changed. This does not
precompile every
extension, Triton kernel, graph or model plan, and it does not select the
experimental Gemma head epilogue or Flash's separate family-specific GDN module.

The `application-wheel` Docker stage builds and installs the exact owned wheel
under the existing full foundation audits. Its completed immutable layer can be
reused if later preparation fails. The `application` stage runs
`prepare_native.py` through CPU admission using the runtime owners' canonical
recipes, explicit SM12.1/SM12.1a targets and the selected Torch/toolchain without
a visible GPU. Ordinary wheel builds retain their Stable-ABI C owner and gain
no CUDA compiler requirement. AMD64 compilation does not qualify ARM64 imports
or actual SM12.1 execution.

One provider registry governs preparation, sealing, loading and audit. Final
owned binding sources determine required exports; preparation and audit compare
real imported symbols and callability. Unknown binding forms require a coherent
producer update rather than guessed exports. The bundle under
`/opt/tensorfold-native` binds the installed wheel, foundation/toolchain,
host/Python/Torch/CUDA ABI, exact target/flags/sources/owned headers and binary
size/hash. Foundation seals these inputs and publishes a compact build-owned
activation. Operators cannot choose arbitrary bundles.

Runtime GPU/minimum-capability checks precede artifact selection. Declared
canonical recipe or artifact mismatches fail explicitly. Ordinary unprepared or
custom calls, explicit build directories, other supported targets and separate
development workspaces retain their valid JIT path. Modules load once under a
process preparation lock and remain strongly owned through process exit. The
sealed file/stat lifetime covers sources and binaries; inference does not
repeatedly hash them.

Application installation uses `--no-deps --force-reinstall`, including same-version
wheels. Replacing an application after its bundle is sealed is refused before
opening the wheel or invoking pip: rebuild the image, install its wheel, then
prepare the bundle. Full audits retain both primary and audit failures. Partial
preparation invalidates the image and is not resumed from a leftover library or
lock file.
JSON publication closes before rename and preserves write and close failures;
if diagnostic grouping cannot allocate, the primary traceback retains both raw
failure roots rather than replacing the write error.

Fresh owned CPP/header inputs are prepared for whole-second image timestamps
before compilation and their first seal, preserving bytes and modes. Preparation
refuses an already sealed bundle. Strict post-seal file/stat validation remains
active; a later timestamp mutation is not normalized away.

For a prepared image, run the GPU-free consumer audit through the managed CPU
boundary:

```bash
python3 /opt/tensorfold-pins/foundation_guard.py run -- \
  python3 /opt/tensorfold-pins/prepare_native.py --audit
```

This checks installed recipes, input closure, exports, activation and provider
mapping without executing a GPU kernel. Target numerical, graph and lifetime
controls and complete model comparisons remain separate. AOT work directories
are image-owned and deleted after publication; runtime `/cache` is not a
resumable AOT object cache. Reuse only completed immutable layers or separately
proved compatible artifacts.

Report compiler-cache-cold and compiler-cache-warm process/model startup
separately, including build/preparation/import/storage, first graph/request and
steady request/resource/quality costs. Historical CPU tests, timed-out prototype
preparation and prewarmed model campaigns do not qualify this source.
