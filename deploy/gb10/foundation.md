# Managed CUDA foundation

The Docker build requires an immutable official NGC CUDA development base and
the selected genuine official nightly wheels. The selected base is CUDA13.4.3 cuDNN development. Official AMD64 roots use
Torch20261017, TorchVision20261007 and Tritonfc8; ARM64 uses its distinct official
Torch20261014/TorchVision20261006 artifacts. The selected pin manifest names
actual whole-wheel hashes. CPU provider selection passed separately; fresh
installed-image/GPU/model qualification and ARM64 execution remain separate gates. The base supplies the CUDA
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

## Prepare native inputs

Resolve on native Linux CPython3.12 for the selected architecture. Use the actual
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
before publishing the native selection. Earlier schema1 data is archived byte-for-byte under `history/20261006`; it is
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

Both input directories are explicit Docker named contexts; they contain no
model weights or private source checkout:

```bash
docker buildx build --platform linux/amd64 --load --target verification \
  --build-context foundation_pins=./generated-amd64 \
  --build-context foundation_artifacts=./artifacts-amd64 \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
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
docker run --rm --network none --entrypoint python3 tensorfold-fork:verification \
  /opt/tensorfold-pins/foundation_guard.py audit
docker run --rm --network none --entrypoint python3 tensorfold-fork:verification \
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
