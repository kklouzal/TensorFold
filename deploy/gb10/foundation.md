# Managed CUDA foundation

The Docker build requires an immutable official NGC CUDA development base and
the selected genuine official nightly wheels. The selected base is CUDA13.4.3
cuDNN development. Both maintained architectures select Torch20261111,
TorchVision20261111, Triton3.9.0+gitb7ddcf86 and protected pip 26.3, with distinct
architecture-specific whole-wheel hashes and explicit resolver metadata. The
[refresh ledger](foundation-refresh-20261111.md) records the current resolver,
build and qualification status. Earlier native ARM/model results retain their
recorded inputs and do not qualify this refreshed foundation.
The declared native additions select cuSPARSELt0.10.0.12-1 and NVSHMEM3.8.0-1
as latest compatible candidates outside the original base package inventory.
Their documented/source ABI admission and the upstream Python distribution
metadata remain separate from actual native/provider/model qualification.
The GB10 runtime `e0bdd6b4` / source UID `1c166686` passed installed CPU and
selected AEON 26B/DFlash functional/media/cancellation/full 262,144-token context gates.
It is deployed as `0aa83b4b` on port 8888 with profile/gold/cache reuse verified and
replaced `7ea` retained stopped. The user explicitly deferred comparative P3;
no gain/equivalence is claimed. AMD64 native, other models and NVSHMEM distributed
execution remain outside these selected-profile passes. Exact hashes and
deployment receipts are in the ledger.

The base supplies the CUDA
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

An ordinary build uses the seven selected files already tracked for its architecture;
it does not need a new resolver report or a private wheelhouse. The pin JSON is
the authority for immutable base references, the CPython interpreter archive,
official wheel identities, owned source builds and the two required native
runtime addons. The existing `platform_pins.py` mapping selects its companion
locks, constraints and manifest.

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

# Copy the seven existing selected files using their maintained filename mapping.
# The same pin JSON supplies the base reference, the interpreter archive and both
# exact addon identities; all three downloads use the same checked loop below.
python3 - "$TF_BUILD_ARCH" "$TF_FOUNDATION_PINS" \
  "$TF_BUILD_INPUTS/base-image.txt" "$TF_BUILD_INPUTS/python-sha256.txt" \
  > "$TF_BUILD_INPUTS/native-addons.tsv" <<'PY'
import json
from pathlib import Path
import shutil
import sys

source = Path("deploy/gb10").resolve()
sys.path.insert(0, str(source))
from platform_pins import platform_pins

target = platform_pins(sys.argv[1])
destination = Path(sys.argv[2])
for field in ("pins", "foundation", "runtime", "verification", "constraints", "manifest", "build"):
    filename = getattr(target, field)
    shutil.copyfile(source / filename, destination / filename)
pins = json.loads((destination / target.pins).read_text())
Path(sys.argv[3]).write_text(pins["base"] + "\n")
Path(sys.argv[4]).write_text(pins["interpreter"]["sha256"] + "\n")
for addon in [*pins["native_addons"], pins["interpreter"]]:
    print(addon["artifact"], addon["url"], addon["sha256"], sep="\t")
PY

git rev-parse HEAD > "$TF_BUILD_INPUTS/source-revision.txt"
sha256sum "$TF_FOUNDATION_PINS"/* > "$TF_BUILD_INPUTS/pin-inputs.sha256"

# Download only the two addon records and the interpreter archive declared by the
# current selected pins. A canonical name is published only after the complete
# checksum succeeds.
while IFS=$'\t' read -r TF_DEB_NAME TF_DEB_URL TF_DEB_SHA; do
  TF_DEB_TMP="$(mktemp "$TF_FOUNDATION_ARTIFACTS/.${TF_DEB_NAME}.XXXXXXXX")"
  curl --fail --show-error --location --proto '=https' --proto-redir '=https' \
    --output "$TF_DEB_TMP" "$TF_DEB_URL"
  printf '%s  %s\n' "$TF_DEB_SHA" "$TF_DEB_TMP" | \
    sha256sum --check | tee -a "$TF_BUILD_INPUTS/native-addon-checks.log"
  mv -- "$TF_DEB_TMP" "$TF_FOUNDATION_ARTIFACTS/$TF_DEB_NAME"
done < "$TF_BUILD_INPUTS/native-addons.tsv"

TF_BASE_IMAGE="$(cat "$TF_BUILD_INPUTS/base-image.txt")"
TF_PYTHON_SHA256="$(cat "$TF_BUILD_INPUTS/python-sha256.txt")"
TF_SOURCE_REVISION="$(cat "$TF_BUILD_INPUTS/source-revision.txt")"
export TF_FOUNDATION_PINS TF_FOUNDATION_ARTIFACTS TF_SOURCE_REVISION TF_BUILD_JOBS TF_PYTHON_SHA256
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
refresh, resolve natively on the selected architecture with the pinned
interpreter. Use the actual immutable NGC child/index identity and exact
official Torch, TorchVision and Triton wheel identities. Pin schema3 records the
immutable base, target, CPython minor and exact `interpreter` archive, distinct
cache key, explicit `foundation_roots` with required extras, exact whole-wheel
hashes and `owned_builds`.

`resolve_dependencies.py` verifies the downloaded interpreter archive against
the pins, extracts it into a temporary directory and writes one pip dry-run
report for the foundation, application, verification and owned-build roots. Only
the owned packages may resolve to sdists. `--constraints-manifest` keeps every
version an earlier manifest records. Binary wheel acquisition remains official
HTTPS, including the official `download-r2.pytorch.org` redirect host; local
file wheel reports are refused. Run both tools with Python 3.12+ and
`packaging` installed:

```bash
python3 deploy/gb10/resolve_dependencies.py --architecture amd64 \
  --archive "$TF_FOUNDATION_ARTIFACTS/cpython-3.15.2+20261110-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz" \
  --constraints-manifest deploy/gb10/foundation-manifest-amd64.json \
  --output generated-amd64/resolver.json
python3 deploy/gb10/lock_dependencies.py --architecture amd64 \
  --report generated-amd64/resolver.json --output-directory generated-amd64
```

Use `--architecture arm64` with its ARM64 pins and archive on an ARM64 host. The
generator still accepts a declared `--cross-report` for wheel-only selections,
but pip's cross-platform mode cannot select the owned sdists, so the current
selection needs native reports. The outputs are separate foundation, runtime,
test and build locks, exact protected constraints and a provenance manifest. Do
not hand-edit generated locks. The selector validates the base, interpreter and
every relationship between pins, manifest and all locks before publishing the
native selection. Earlier schema1 data is archived byte-for-byte under [history/20261006](history/20261006/README.md); it is
evidence and a test oracle, not a deployment fallback.
A cross-architecture resolver report does not qualify native execution.

### Interpreter and owned source builds

Ubuntu 24.04 has no CPython 3.15. The image installs the pins'
python-build-standalone `install_only_stripped` archive at `/opt/python3.15`
(contents unchanged; its group-writable modes are reduced to the usual 0755/0644) after checking the `PYTHON_SHA256` build argument; the selector
binds that argument, the archive name and the running version to the pins, and the
seal records the archive with the native add-ons. The interpreter tree itself is
not sealed (the earlier apt interpreter was not either). Its sysconfig names
clang; owned builds use GCC, and runtime extension builds use the sealed
compiler wrappers.

`owned_builds` lists selected versions without a target wheel: pyyaml 6.0.3 and
xgrammar 0.2.9 have no cp315 wheels. Each record pins the PyPI sdist URL and
SHA256, its build requirements (the sdist's declared ones plus, for xgrammar,
the CMake 3.18+ and Ninja that scikit-build-core requests from a build host
without them) and at most `CFLAGS`/`CXXFLAGS`.
xgrammar builds with `CXXFLAGS=-Wno-error=odr`: GCC 13 LTO reports `-Wodr` for
`std::variant` across its C++17 core and C++20 tvm-ffi bindings, and upstream
`-Werror` makes it fatal (the warning stays visible). The resolver keeps these
sdists in the manifest's runtime role, not in the runtime lock. The discarded
`owned-builds` stage installs the build lock's wheels into a separate venv and
runs `owned_builds.py`, which fetches each sdist by hash and builds it with
`--no-build-isolation --check-build-dependencies --no-index`; nothing unpinned is
downloaded. The guard then installs the locked runtime wheels and exactly one
built wheel per owned record, with full audits, and records the wheel SHA256s in
the foundation state. pyyaml builds without libyaml headers in the base, so its
optional C loader is absent (TensorFold does not use it). On CPython 3.15 the
Torch metadata no longer requires `cupti-python` or `cuda-bindings`; TensorFold
imports neither and uses the base SDK's CUPTI.

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
  --build-arg "PYTHON_SHA256=$TF_PYTHON_SHA256" \
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
and compiler/runtime package versions, file SHA256s and symlink targets, and
holds those packages. Nonheld OS packages are upgraded before Python preparation;
the pristine baseline is checked immediately afterward and again after native
add-ons and dependency installation. Held packages are never unheld to satisfy
a new dependency closure.
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
under the approved no-override exception. (The CPython 3.12 foundation also
recorded CUDA bindings' static CUDART; on 3.15 Torch no longer installs them.) The graph therefore does not claim every component uses
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
