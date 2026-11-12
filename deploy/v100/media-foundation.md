# CPU media software for the V100 source foundation

This additive image supplies the required CPU processor/decoder backend against
the [selected source Torch stack](source-foundation.md). It reuses the protected
published Transformers 5.22.2 provider already in that SDK and builds published
TorchVision 0.30.0.dev20261111 source8d995a1694b18f6944da14f4a45865d0277f8b20
on CPU, PyAV 19.0.2, FFmpeg 9.0.2 and dav1d 1.5.4. Pillow 13 and disjoint media
extras/build tools have generated hash locks constrained by the base runtime.

The Transformers source archive is reference-only. No separate Transformers
build stage or third replacement wheel is installed. Prepare/export/install
receipts bind its 2,810 inherited payload files, whole METADATA and genuine
Requires-Dist before/after. The two-wheel role validator refuses stale extra
Transformers wheels without deleting historical files. The input/configuration
manifest and exact source revisions remain in [media-pins.json](media-pins.json).
Current preparation/ownership controls passed 103 cases, and the six CPU build/
export phases completed with exit0. Image/install/decoder and model qualification
remain separate from those passes and from the historical receipts below.

The pinned image processor consumes CPU FP32 TorchVision transforms. This needs
the actual TorchVision backend, without TorchVision CUDA operators or NVJPEG.
The eventual FP16 vision encoder remains a separate GPU/model integration.
PyAV uses the owned CPU FFmpeg prefix, with dav1d for CPU AV1. FFmpeg retains its built-in CPU decoders
and demuxers, with external auto-detection, hardware acceleration, encoders,
muxers and devices disabled. No CUDA/NPP or additional encoder dependency is
introduced. The source Torch, CUDA and NCCL libraries are preserved. The refreshed source-media
candidate requires its own source-wheel/CPU-origin and installed native/model
qualification; historical receipts below retain their original inputs.

The earlier home-design checkpoints `14fda300` and `72972474` retain their
recorded processor/stack scope; they do not qualify this refreshed candidate.

## Independent input acquisition checkpoint

Input acquisition and extraction can run before the final SDK exists. They
require Python 3.12, `packaging` and curl 8.4 or newer for the
[in-transfer size limit](https://raw.githubusercontent.com/curl/curl/curl-8_5_0/docs/cmdline-opts/max-filesize.d);
installation, source compilation and
installed-media auditing still require their separately specified SDK/builder
prerequisites. The inspected source-builder and GB10 verification images lacked
curl. A separate existing acquisition-only image supplied Python 3.12.3 and
curl 8.5.0, with an exact cached `packaging` 25 wheel mounted read-only on
`PYTHONPATH`, without installing it or changing the SDK. The actual local tool
image ID was
`sha256:944aadd4f84258d1594ed9aa8b08514d664e3a00b46e0dddc6790103d191d70b`;
this is measured tool provenance, not a registry-pull or target-SDK claim.

Canonical producer/pins from commit
`9b651b8f456fadadec71d493351c6768b7190058` acquired all 70 exact files:
eight source/configuration/fixture artifacts, eleven Debian packages and 51
support wheels, totaling 395,910,778 logical artifact bytes. Forty-one wheels
were verified copies from the read-only SDK cache; ten new wheels totaled
13,023,907 bytes, with lengths recorded before transfer. All completed inputs
and parent directories were rehashed/fsynced. All five extracted source closures
passed canonical verification and durability checks. Producer/pin files and the
existing SDK were untouched; no model weights or host-package installation was
involved.

The first FFmpeg transfer timed out with curl exit 28. Its retained record does
not establish a more specific network cause. One explicit canonical retry
succeeded with the original URL, size and hash. A separate instruction-packet
collector initially rejected Transformers' legitimate `AGENTS.md` symlink to
`.ai/AGENTS.md`; canonical extraction/source verification had already accepted
it. Read-only follow-up verified the archive-declared alias and its regular
target inside the source closure, then completed instruction inspection and
fsync without changing source bytes or the link. Earlier failed/incomplete
records remain intact.

The completion receipt is
`native-media-foundation-v1/acquisition-completion-v2.json`, SHA256
`917f053a3ae808c7369aa284bd74d54939b111ebca11dcb587d8a91727f9f455`,
under the private audit root. It records input completion only, with
`release_qualified=false`. No `prepare`, source build, export, final image or
installed-media probe ran. Full native media/application controls, physical
V100/model quality and performance remain separate unfinished gates. Inspect
the captured applicable source instructions again before the future build;
input readiness does not waive them or the exact SDK/audit boundary.

## Reproduce the source build

Use the completed explicit source-SDK image and an immutable current producer
snapshot. Keep source/producer files unchanged for the whole operation; a
read-only mount does not prevent edits on its host. Select CPU/memory/job bounds
from the actual host and retain the completed environment receipt.

```bash
set -euo pipefail
: "${TF_MEDIA_SDK_IMAGE:?Set the explicit immutable source SDK image}"
: "${TF_MEDIA_PROJECT:?Set the immutable current producer snapshot path}"
: "${TF_MEDIA_STATE:?Set a new owned external media state directory}"
: "${TF_MEDIA_CPUS:?Set the actual CPU quota}"
: "${TF_MEDIA_MEMORY:?Set the actual container memory limit}"
mkdir -p "$TF_MEDIA_STATE/native"

docker run -d --name tensorfold-v100-media-build --runtime runc \
  --cpus "$TF_MEDIA_CPUS" --memory "$TF_MEDIA_MEMORY" --network bridge \
  -e "MAX_JOBS=${MAX_JOBS:-4}" -e NVIDIA_VISIBLE_DEVICES=void \
  -e CUDA_VISIBLE_DEVICES=-1 --entrypoint sleep \
  --mount type=bind,src="$TF_MEDIA_STATE",dst=/media \
  --mount type=bind,src="$TF_MEDIA_PROJECT",dst=/src,readonly \
  "$TF_MEDIA_SDK_IMAGE" infinity
docker exec tensorfold-v100-media-build ln -s /media/native /opt/tensorfold-media
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh acquire
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh prepare
docker network disconnect bridge tensorfold-v100-media-build
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh torchvision
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh dav1d
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh ffmpeg
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh pyav
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh export
docker build --network=none --build-arg "SOURCE_IMAGE=$TF_MEDIA_SDK_IMAGE" \
  --build-context "media_artifacts=$TF_MEDIA_STATE" \
  -f "$TF_MEDIA_PROJECT/deploy/v100/Dockerfile.media" \
  -t tensorfold-v100:media "$TF_MEDIA_PROJECT"
docker image inspect tensorfold-v100:media --format '{{.Id}}'
```

The current preparation/source-build/export receipts pass; final image/install/
decoder checks remain separate. The historical 04cf image/producer receipts remain
reference evidence; do not mix that older SDK with refreshed native/provider
inputs.

Create `$TF_MEDIA_STATE/native` before starting. Keep it outside the project and ordinary
Docker context. The owned prefix link is only a build-state mapping; the final
image contains a real `/opt/tensorfold-media` directory. `acquire` uses bounded,
resumable HTTPS transfers and only publishes exact complete artifacts. It
extracts one bounded source root per archive. Source reuse verifies every
original archive file/link and every recorded generated file. The stage wrapper
checks sources before building and records completed generated caches afterward;
changed or unrecorded cache inputs require an explicitly prepared fresh source
root. Interrupted extraction/build state is retained for diagnosis, never
silently treated as complete. Archive download retries are explicit.

`prepare` verifies the generated `media-native-apt.json` native-requirements
schema against the exact foundation APT manifest and installed versions. All
eleven requirements are already satisfied in the selected 191-package SDK, so
there is no native download/reinstall or base write. It snapshots the inherited
Transformers provider, installs only disjoint hash-locked media extras/build
tools and verifies the provider unchanged. CPU source builds then run offline. PyAV follows its maintained `make` entry point, with
supported Python/pip/quoted compiler flag variables and the exact build
constraints. `export` binds completed source receipts, exactly two wheel hashes,
the inherited-provider snapshot, producer inputs and the native prefix. Do not reuse an export after changing
any of its inputs. Regenerate support locks using the existing
`source_foundation.py render-media-build-tools`,
`render-media-runtime-support` and `offline-lock` commands from actual
matching-platform reports. Their companion JSON binds the source/runtime
identity. First constrain resolution with `runtime-constraints` from the
protected base runtime. Publish `media-build-constraints.txt` through the same
actual build report's `render-media-build-tools --constraints` output; do not
replace shared providers with a separately resolved closure. Active source/media Ninja pins stay
at the selected fixed 1.13.2 tool; build stages and export bind its actual wheel/
executable provenance through the shared producer guard. This does not qualify
media output or a completed SDK. Do not edit generated locks or installed tools.

## Build resource contract

Refreshed media producers accept explicit canonical `MAX_JOBS` values 1..64 and
retain 4 when unset or empty. The same value is exported as
`CMAKE_BUILD_PARALLEL_LEVEL` and `OMP_NUM_THREADS`, including Python/PyTorch/
CMake child builds such as TorchVision. FFmpeg/PyAV Make and dav1d Ninja receive
explicit `-j` bounds; FFmpeg install also retains that bound. Meson install uses
`--no-rebuild` after the required explicit Ninja build, avoiding an additional
implicitly sized rebuild. The option is supported by the pinned
[Meson 1.12.1 implementation](https://github.com/mesonbuild/meson/blob/1.12.1/mesonbuild/minstall.py).
No required compile or failure check is skipped.

PyAV's pinned Makefile runs one `python setup.py build_ext --inplace --debug`
recipe. Make's `-j` and the three inherited job variables alone do not set that
command's extension compiler parallelism. A future producer writes the bounded
job-specific `build/pyav-build-ext-N.cfg` with `[build_ext] parallel = N`, then
passes its absolute path through the selected setuptools 84.2.1
`DIST_EXTRA_CONFIG` mechanism, with parser/version evidence bound to the actual
build report. The earlier 80.9.0
[discovery implementation](https://github.com/pypa/setuptools/blob/v80.9.0/setuptools/_distutils/dist.py)
remains historical feature evidence.
This variable is set only for the PyAV stage and reaches both canonical Make
and the subsequent wheel build. The Makefile, vendor source, `--inplace --debug`,
CFLAGS/LDFLAGS and wheel command stay unchanged. `MAX_JOBS=1` remains an explicit
serial extension-compiler control. Cython's existing `cythonize` call and its
thread setting are unchanged; no 24-Cython-worker or build-speed claim follows.

Before compiler/installer work, each stage creates its exclusive pending
environment record, fsyncs the file and fsyncs its parent directory. A failed or
unretired pending attempt blocks a new build and export. After successful work
and source recording, completed facts are written to a separate file, fsynced,
atomically renamed and followed by a parent-directory fsync. Only then is the
pending marker removed and its directory retirement fsynced. Prior completed
facts have a durable bounded backup; a failed publication restores their
current path and retains/restores the pending gate. Restoration failures are
reported alongside the primary failure and require explicit owner recovery;
no automatic retry or older-receipt borrowing is allowed.

`receipts/build-environment-STAGE.json` binds the actual three job settings,
executed script/helper hashes and completed source-receipt hash. Export
validates those facts and embeds them in `media-build.json`; it never infers
them from the export process's current environment. These settings bound work; they do not
prove that 24 workers executed or establish a speed gain. Container CPU/memory
budgets require their separate actual-host/build evidence.

PyAV alone uses environment-receipt version 2, adding the actual config
path/size/SHA256, pinned setuptools version and parsed `effective_parallel`.
The durable pending record is intent; configuration is then written and fsynced
before compiler work. Setuptools discovery and the canonical `build_ext` flags
are parsed before and after the stage; a changed file, wrong parser version or
mismatched effective setting leaves the pending gate and refuses completion.
Export verifies the completed config/facts without inferring them from its own
job environment. These are parser-selection facts, not measured active workers.
Other stages retain their version-1 environment records.

The completed frozen `8e6c89` media build remains its actual serial-PyAV
reference: its logged extension command had `parallel=None`, despite inherited
MAX/CMake/OMP24. Its vendor bytes, wheel/export receipts, warnings and timing
history are retained. The corrected configuration requires a fresh canonical
producer/state and separately recorded execution; it is not a hotpatch or
relabel of that completed build, nor a new installed-media qualification.

The earlier four-CPU/24 GiB example above retains its historical scope. For a
new 24-job media invocation, set `-e MAX_JOBS=24` when creating a separately
resource-budgeted container and retain its actual resource receipt. Use a new
producer checkpoint and owned state. The active e90 SDK snapshot/build and
its receipts remain immutable; they are not relabelled or requalified by this
follow-on. No dependency, model, runtime, generated Dockerfile or framework
source pin changes are implied, and no actual media build is established by
shell-driver unit controls.

## Installed identity and controls

The image retains the protected original source-foundation audit. Its separate
media producer is `/opt/tensorfold-pins/media_foundation.py`; media inputs and
the installation inventory are under `/opt/tensorfold-media-pins`, protected
against group/world writes. The callable
`audit_install(manifest, pins)` verifies the complete added package/native/config
inventory and actual runtime identity. It returns software-only evidence with
`release_qualified=false`; serialized records grant no process authority or
execution-profile eligibility.

The historical original source audit rejected every TorchVision mapping, including these
CPU consumer extensions. The additive image retains that failure and the old
producer, installation manifest and actual cold-audit output under protected
`/opt/tensorfold-pins/audit-history`. Its explicit `revise-audit-policy` step
checks every original installed byte against that retained inventory, preserves
the original framework build receipt, and records separate original-build and
current-audit producer identities. The new policy accepts ordinary consumer
extensions under the existing process audit while still refusing bundled
accelerator-provider roots and substituted known library families. Exact CPU
TorchVision bytes and `CUDA=-1` remain the media audit's responsibility.
The corrected 04cf source SDK records the current source producer directly; its
build does not inherit the earlier strict-policy producer identity. Rebinding
the media image to that corrected SDK reuses the exact pinned media wheel and
native-prefix bytes with their original build receipts, followed by fresh
consumer/source audits and controls. This does not claim a clean media source
recompile against the corrected framework or transfer earlier application tests.
The original recorded media build used
`tensorfold-v100:software-binding-final@sha256:413c39b178688fba6890afb61264da11f2fc175054b6689241a1d5d5aee5de69`;
that predecessor and its strict-policy failure remain historical evidence.

The corrected rebind is local image ID
`sha256:f1c72b97ae8949a0aab0b1407edd4fc320671fb129e104e39095eb602ce2e94f`
(`tensorfold-v100:media-socket-04cf2557`), 23,316,923,844 uncompressed bytes.
It preserves all 34 corrected SDK layers and adds 312,481,365 bytes. The
maintained offline recipe completed the exact-wheel installation, retained
baseline/source policy rebind, media audit, official processor probe and source
process audit. Fresh separate media/source audits passed over 4,211/15,634
artifacts, as did the processor probe and 110 installed controls across nine
modules, with zero skips or warnings in 325.85 seconds. These component checks
remain distinct from the fresh complete application CPU gate recorded in the
[corrected-stack receipt](../../docs/verification-v100-sdk-rebind-cpu-20261026.json).
The later [portable-policy software receipt](../../docs/verification-v100-portable-policy-cpu-20261027.json)
records a separate 187-module application run on f8 and affected mapping-policy
checks on 6da using the unchanged whole application wheel and media bytes.
Actual successor target acceptance after GPU work, compiler/device/model/
resource/quality and performance gates remain open.
The selected consumer recipe is `Dockerfile.media-binding-verification`; its
default names this actual locally resolved OCI index. The former default
`tensorfold-v100:media-dav1d-candidate@sha256:7c547c67e4783f3767b0cb2f88587331b61cef6b4adb595548f7a19deab8ef91`
remains a historical component identity, with no registry publication claim.
Framework metadata queries materialize their own lazy cuDNN component before
collecting mapped-library identity. This changes audit observation order, not
framework/provider binaries. A fresh-process regression compares consecutive
queries; a same-process media/base binding control covers actual warm imports.

```bash
docker run --rm --network none tensorfold-v100:media \
  /opt/tensorfold-pins/media_foundation.py audit \
  --pins /opt/tensorfold-media-pins/media-pins.json
docker run --rm --network none tensorfold-v100:media \
  /opt/tensorfold-media-pins/media_probe.py --configs /opt/tensorfold-media-pins
docker run --rm --network none --cpus 4 --memory 24g --entrypoint python \
  --mount type=bind,src="$PWD",dst=/source,readonly tensorfold-v100:media \
  -m pytest -c /verification/pytest.ini -o 'pythonpath=/source/tests /source' \
  -p no:cacheprovider -q /source/tests/test_v100_media_foundation.py \
  /source/tests/test_v100_media_runtime.py
```

`media-install.json` publishes the complete actual decoder/demuxer inventory,
FFmpeg compile configuration/library versions, Pillow codec versions, provider
versions and processor classes. Demuxer names can contain comma-separated aliases.
An absent codec/container remains unsupported and must be rejected specifically
at the service boundary; the inventory is not an advertised upload policy.
Registration is not per-codec execution proof: FFmpeg's native `av1` entry requires
hardware acceleration. The selected CPU provider is source-built dav1d 1.5.4,
with generic x86-64 C flags and the upstream runtime SIMD dispatch. The earlier
candidate's actual AV1 `ENOSYS` failure is retained. A small pinned FATE bitstream
is decoded through the actual automatic PyAV provider, with every output pixel
compared to an independent libaom reference. libaom is a reference-only existing
host tool and is not added to the runtime. These controls do not establish all
AV1 profiles, service limits or full-model video quality.
Installed controls resolve the pinned official configurations, run CPU FP32
image resize/normalization/patch checks, video odd-frame repetition, independently
constructed Y4M RGB/timestamp decoding, and a tiny actual HF FP16 vision forward.
The tiny random model is a shape/import/finite-output fixture, not SC117 quality.

The first source builds predated the maintained preflight wrapper. Their exact
archive inputs were independently verified after the builds; subsequent complete
generated-cache receipts record that observation. This does not retroactively
claim that the new preflight checks ran. Historical failed PyAV Make flag quoting
is retained in the private evidence; the supported caller quoting was corrected
without editing PyAV source. A clean independent rebuild remains separate.

Alpha/grayscale/orientation parity, complete advertised-format fixtures, decoded
resource bounds, frame/FPS policy, service integration, actual SC117/mmproj
conversion, full-model/media quality, V100 execution/resources/performance and
compiler issue 6111901 remain separate pending gates. This image establishes no
resident session count or production media qualification. The GB10/Blackwell and
other platform paths are unchanged.
