# CPU media software for the V100 source foundation

This additive image supplies the required official processor backend against the
[selected source Torch stack](source-foundation.md). It builds TorchVision
0.29.1 on CPU, Transformers at `50e35cc0ca18ec19b80653690de553e6decf74f5`,
PyAV 19.0.2, FFmpeg 9.0.2 and dav1d 1.5.4 from their exact archived sources. Pillow 13.0.0
and the remaining Python support wheels have generated hash locks. The
[input manifest](media-pins.json) pins both official processor configuration
files, source archives, revisions and extraction bounds. The current home design
`14fda300` retains the inspected processor/stack contract from `72972474`.

The pinned image processor consumes CPU FP32 TorchVision transforms. This needs
the actual TorchVision backend, without TorchVision CUDA operators or NVJPEG.
The eventual FP16 vision encoder remains a separate GPU/model integration.
PyAV uses the owned CPU FFmpeg prefix, with dav1d for CPU AV1. FFmpeg retains its built-in CPU decoders
and demuxers, with external auto-detection, hardware acceleration, encoders,
muxers and devices disabled. No CUDA/NPP or additional encoder dependency is
introduced. The source Torch, CUDA and NCCL libraries are preserved.

## Reproduce the source build

Use the exact base from `Dockerfile.media`, a Linux AMD64 CPython 3.12 container,
and an owned external state directory. These CPU build/semantic controls used
four CPUs and at most 24 GiB; they are not throughput benchmarks. Do not change
source inputs/scripts during a stage. No GPU or host driver change is needed.

```bash
docker run -d --name tensorfold-v100-media-build --cpus 4 --memory 24g \
  --entrypoint sleep \
  --mount type=bind,src="$TF_MEDIA_STATE",dst=/media \
  --mount type=bind,src="$PWD",dst=/src,readonly \
  tensorfold-v100:source-socket-04cf2557@sha256:eb09ff449dba6acc89d7207547a10b767adde592f47fef839f33dc98f269df28 infinity
docker exec tensorfold-v100-media-build ln -s /media/native /opt/tensorfold-media
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh acquire
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh prepare
docker network disconnect bridge tensorfold-v100-media-build
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh torchvision
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh transformers
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh dav1d
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh ffmpeg
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh pyav
docker exec tensorfold-v100-media-build bash /src/deploy/v100/media_build.sh export
docker build --network=none --build-context "media_artifacts=$TF_MEDIA_STATE" \
  -f deploy/v100/Dockerfile.media -t tensorfold-v100:media .
docker image inspect tensorfold-v100:media --format '{{.Id}}'
```

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

`prepare` installs the eleven exact Debian artifacts and hash-locked Python
support/build closure; it does not select a new package version. CPU source
builds then run offline. PyAV follows its maintained `make` entry point, with
supported Python/pip/quoted compiler flag variables and the exact build
constraints. `export` binds completed source receipts, three wheel hashes,
producer inputs and the native prefix. Do not reuse an export after changing
any of its inputs. Regenerate support locks using the existing
`source_foundation.py render-runtime`/`render-build-tools` and `offline-lock` commands
from matching-platform resolver reports; the companion JSON records their
identities. Do not edit generated locks.

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
