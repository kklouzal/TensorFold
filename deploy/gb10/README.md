This integration reproduces the GB10 Flash Next deployment from repository
source. It includes the native SSD reader, multi-image/video and copy-draft
changes, xgrammar 0.2.8, and the Harness `/v1/tokenize` endpoint. The native
baseline is 262,144 prompt-plus-output tokens with four request slots.
`provenance.json` records the upstream commits, integrated patch hashes, model
revision, and the original container stack. The Dockerfile supports Linux ARM64
and AMD64 with NGC CUDA 13.4.1 and upstream PyTorch nightly 2.16.0.dev20261006,
matching TorchVision 0.30.0.dev20261006 and Triton 3.9.0+gitaad2a60d.
`nightly-pins.json` retains the validated ARM64 inputs;
`nightly-pins-amd64.json` records the corresponding official AMD64 inputs.
Both include immutable base digests and wheel SHA256 hashes. These are the
recorded 2026-10-06 inputs, rather than floating latest tags. The AMD64 dependency
closure began with an explicit cross-platform pip report; subsequent native
AMD64 image installation, installed-wheel audits, CUDA checks and actual
generations ran on the 16 GB Blackwell test host. Those results belong to their
recorded image/source identities. Fresh final-image checks for the latest source
remain tracked in [optimization validation](../../docs/optimization-validation.md).
The current Compose profile enables YaRN factor 2 with 524,288
prompt-plus-reply tokens and four request slots. Caches grow on demand within
the global memory gate; four simultaneously full-size contexts are not assumed
to fit the GB10.

Build from the repository root:

```bash
docker buildx build --load --target runtime \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  --tag tensorfold-gb10-fork:local --file deploy/gb10/Dockerfile .
docker compose -f deploy/gb10/compose.yaml config --quiet
```

The Dockerfile uses NVIDIA's NGC CUDA DL inference development base, pinned by
its immutable multiarchitecture OCI index. It selects the original ARM64 child
or the audited AMD64 child, then validates the actual Linux/Python architecture
before installing the matching locks. That supplies NVCC, CUDA headers and forward-compatibility
libraries. Python 3.12 and upstream nightly packages live in an isolated venv;
NGC's tightly coupled PyTorch plugins are not part of this image. All Python
runtime dependencies are pinned to wheel URLs and SHA256 hashes, including the
nightly's required CUDA libraries and Triton. The inference base's sparse/solver
development headers are supplied by those wheels; `CPATH` makes their audited
include directories available to NVCC and C++. Transformers, tokenizers, the Hub
client, xgrammar and core model-input dependencies retain their deployed pins
where compatible. The native reader's C++ source is included in the fork wheel.
The verification target adds its own pinned pytest tools, maintained tests and `CHANGELOG.md`, all readable by unprivileged workers, and contains no weights. It also includes the synthetic RAM-expert startup and cache-policy benchmarks in `tools/`. The frozen AMD64 verification image passed its built-image update and public-permissions checks without a checkout mount: 343 passes and eight explicit skips in the separate baked batch. The earlier qualified-fixture CPU results retain their original scope; this does not establish a new ARM64/GB10 run.
The opt-in, general CUDA host-RAM expert mode is documented in
[the RAM-expert recipe](../../docs/recipes/ram-experts.md); this ARM64 image
and its AMD64 counterpart use the model family's existing CUDA target. Hardware
architecture is selected from the actual GPU when owned CUDA extensions build.

Build AMD64 on the target's native Docker host with the same file:

```bash
docker buildx build --platform linux/amd64 --load --target runtime \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  --tag tensorfold-fork:amd64 --file deploy/gb10/Dockerfile .
```

For a persistent service on a separate-VRAM AMD64 host, use the
[discrete-GPU Compose profile](../discrete-gpu/README.md). Its context, slots
and expert-cache allocation have their own qualification scope.

The GB10 Compose profile contains its real local paths and model options. Select
paths, model, memory pool and context for the target machine using actual
admission and measured peaks. A 16 GiB card does not inherit the GB10's available
memory or the original MLX checkpoint's host-fit assumptions.

cuDNN uses the complete hash-pinned wheel provider. Build-time aliases in
`/opt/tensorfold-cudnn` cover its major, minor and full-version SONAME lookups;
that directory leads the library search path. The NGC inference base has a
reduced cuDNN library set, so ordinary wheel-directory precedence can mix
providers and omit the precompiled Conv3d engine needed by the vision tower.
Neither dependency tree is modified.

Compiled artifacts use `/cache/cu1341-torch216-dev20261006-triton39-aad2a60d/arm64/`
or the sibling `/amd64/`, separate from each other and older Torch/CUDA caches.
The Dockerfile defaults `MAX_JOBS` to eight. Set `--build-arg MAX_JOBS=N` when
building or `-e MAX_JOBS=N` when running to choose a bounded compiler worker
count appropriate to that host's CPU and memory. This setting does not prove
that a compiler used every worker. The small verification examples below use
one worker to limit memory peaks. New Torch versions use OS advisory
locks: a persistent unlocked `lock` file is normal and must not be deleted as a
recovery step. TensorFold emits old stale-file guidance only for the audited
FileBaton implementation.

The 2026-10-06 GB10 check used host driver 580.178.04. A fresh container check demonstrated
NGC's enabled CUDA forward compatibility with user-mode driver 615.71.09 and
the nightly's CUDA tensor operations on SM 12.1. No host driver change was made.
The project CUDA launcher reruns NVIDIA's shipped compatibility probe on each
startup and refuses probe failure before executing the original NGC entrypoint.
This preserves its validation across restarts, when NGC's cached marker survives
but its process environment loses the probe result.
Full model startup, memory, long-context quality and performance require their
own deployment validation.

NVIDIA's forward-compatibility support covers data-center GPUs, selected NGC
Server Ready RTX models and Jetson. A desktop GeForce Blackwell with an older
driver cannot assume the GB10's compatibility-library path. Confirm that target's
native driver and the fresh CUDA probe; a failed probe stops startup. See
[NVIDIA's compatibility contract](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

The pinned image was deployed on this GB10 on 2026-10-06 at native 262,144
tokens/four slots. Full-model startup, 17 API checks before and after a real
same-container restart, four simultaneous requests, 20 repeated requests, tool
result round trips, two-image input and MP4 video recognition passed. The final
library/launcher fix passed 111 CUDA regressions and eight launcher boundary
tests. The previous container and image were retained stopped for rollback.
These are bounded functional checks; 512k YaRN quality/capacity and sustained
memory/performance validation remain separate.

The factor-2 profile also passed startup, 17 short API checks and four concurrent
requests. One 523,882-token prompt plus 18 reply tokens retrieved the exact
synthetic passphrase at 60% depth without truncation. The trial took 442 seconds
and included three simultaneous short requests, all correct in about 2.3 seconds
while the large prompt was prefilling. This single retrieval is a bounded
capacity/quality check; it does not establish uniform long-context accuracy.

`compose.yaml` preserves the current sampler, MTP, vision and model/cache mounts,
request slots, memory/swap settings, CPU set, health checks, and restart policy.
Cache paths come from the runtime image's ABI-specific defaults. Its container
name and port are the existing production service's. Rendering the
configuration is read-only; applying it is a service replacement and belongs to
the deployment step after validation.

The current Compose project is named `tensorfold-gb10-yarn2`, separating the
retained native fork container from the new managed service. After a controlled
cutover, manage it from this repository with
`docker compose -f deploy/gb10/compose.yaml up -d`, or restart the existing
container with `docker restart qwen38-flash-next-tf`. Preserve the previous
container stopped under a distinct rollback name during deployment validation.
The earlier recipe and Harness launch scripts manage their captured images;
use this fork's Compose configuration for the new deployment.

The Harness endpoint counts the native rendered prompt and image expansion
without generation. It permits counting a history beyond the served context,
while retaining generation admission and independent image safety checks.
It requires `--thinking-budget 0`; callers select budgets per request. Returned
token/prompt inspection is capped at 16,384 tokens, and count-only responses
support larger histories. HTTP preparation has a bounded queue and cancellation.

YaRN is an optional CUDA text-RoPE policy for Flash Next. The CLI option
`--yarn-factor 2 --context 524288` requests a 2× window without modifying the
shared model snapshot. Native operation omits that option. A full 512k validation
must measure memory and long-context quality; four full 512k streams are not
assumed to fit. Four slots are retained for mixed workloads. Startup logged
29.8 GiB of cache room and 8.94 GiB for one full window; four full windows need
about 35.8 GiB before transient copies. Runtime growth/eviction/admission and
resource failure handling remain active. The singleton growth path can exceed
the soft memory gate, so reserves are not an unconditional headroom floor.

Qwen's [official model guidance](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices)
documents factor 2 for 524,288 tokens and factor 4 for approximately one million
tokens. Static YaRN can affect shorter-context quality. A frequency/kernel oracle
does not establish full-model quality at the expanded limit; long-context
retrieval and representative Harness tasks must be checked separately.

CPU verification, from the repository root:

```bash
docker buildx build --load --target verification \
  --tag tensorfold-gb10-fork:verification --file deploy/gb10/Dockerfile .
docker run --rm --network none --memory 3g --memory-swap 3g --pids-limit 512 \
  --cpus 2 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  tensorfold-gb10-fork:verification \
  tests/test_native_ssd.py tests/test_harness_tokenizer.py \
  tests/test_qwen4_exp_yarn.py -q
```

The native SSD tests use synthetic, isolated files and an independent byte
oracle. Harness tests use the native request preparation with a small tokenizer,
including oversized counts, image expansion, validation, framing, and bounded
preparation. YaRN tests compare frequencies against the pinned Transformers
reference and test context/memory admission.

Synthetic CUDA verification uses the NGC entrypoint to enable the same driver
compatibility setup as the runtime image. It allocates no full model weights:

```bash
docker run --rm --network none --gpus all --memory 4g --memory-swap 4g \
  --pids-limit 256 --cpus 2 -e MAX_JOBS=1 -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  --entrypoint python3 \
  tensorfold-gb10-fork:verification /opt/harness/cuda_entrypoint.py python3 -m pytest \
  tests/cuda/test_flashnext_yarn.py tests/cuda/test_flashnext_vision.py \
  tests/cuda/test_flashnext_forward.py tests/cuda/test_grammar_mask.py \
  tests/cuda/test_vision_patch_conv.py -q
```

These tests cover native byte equality, large-position YaRN, INT8/INT4 KV,
vision/MRoPE, concurrent forward paths, MTP and CUDA graphs. Full-model checks
require the checkpoint and an isolated validation slot or maintenance window.

On 2026-10-06, the pinned nightly image passed 1,543 CPU checks with 21
platform/fixture skips and all 88 synthetic CUDA checks on the GB10. Five
generation-free cases using the actual checkpoint tokenizer matched the running
service's prompts, token arrays and counts exactly, including tools and thinking.
An additional 19 CUDA grammar cases passed against an independent CPU matcher
oracle across FP32, BF16 and FP16 logits. These checks establish component
behavior; full-model startup and expanded
context quality, memory and performance are still deployment gates.

Later native AMD64 installed-origin CPU qualification completed 25 successful chunks against the
normal Source06 verification image: 6,214 non-skipped passes, zero failures/errors and 155 explicit
skips over its qualified 386-path Linux inventory. These counts retain that source-bound snapshot before the new maintained Rotor7 native GPU test; the changed runtime has separate qualification. The qualified read-only test fixture preserved installed
package/native origins; its full 1,033-file native UID-1000 hashes matched before and after.
The dated GB10 counts above retain their original scope. Current source/image identities, default-prefetch
generation observations and completed current CLI/API qualification are recorded in the
[README](../../README.md#verification-and-development) and
[optimization validation](../../docs/optimization-validation.md). The tested notification and full-request optional expert-arena alternatives retain the original notification and fixed-cache policies. The seven-bit decoder lookup was selected only for the [controlled AMD64 serial/MTP region](../../README.md#controlled-rotorquant7-decoder-comparison), without a new public flag; it does not change the full-prefetch Compose profile. The separate aggregate-room-clipping diagnostic was inapplicable in its measured workload, so no arena-lending/clipping path is retained. The frozen AMD64 shipping build/native audits, baked packaging and fresh seven-group API qualification are complete; [the shipping results](../../README.md#frozen-source-shipping-qualification) name commit `07f4294…` and its actual image/wheel identities. These AMD64 results do not add GB10 performance or context coverage; the closing documentation-only update leaves qualified runtime code unchanged.

The nightly Triton bindings emitted nanobind reference warnings at interpreter
shutdown after the CUDA suite, which exited successfully. Small isolated compiler
and GPU probes did not reproduce them. Sustained full-model memory behavior still
needs validation; the warnings are retained rather than suppressed.

To refresh dependency locks, resolve the updated pins in the matching Linux
CPython 3.12 environment with `pip install --dry-run --ignore-installed --report
REPORT.json`, including the three exact wheel pins and verification packages.
Run the generator in an environment with `packaging` installed:

```bash
# Existing ARM64 inputs and generated lock names stay unchanged.
python3 deploy/gb10/lock_dependencies.py --report REPORT.json

# AMD64 native report.
python3 deploy/gb10/lock_dependencies.py --architecture amd64 --report REPORT.json

# Reproduce the current audited AMD64 cross-report locks.
python3 deploy/gb10/lock_dependencies.py --architecture amd64 --cross-report \
  --report deploy/gb10/resolver-amd64.json
```

The generator checks wheel architecture/ABI, Python versions, approved origins,
nightly hashes and the full target dependency closure, including transitive
extras. A cross report requires explicit declaration and retains its actual
resolver architecture; kernel-dependent markers require native resolution.
`resolver-amd64.json` preserves the relevant metadata from the genuine pip report
and its original hash. Dependency selection/provenance does not replace native
image installation, payload hash verification or CUDA/model checks.
Update the image/cache identifiers with the pins and rerun CPU, GPU and model
gates. Do not hand-edit generated requirement locks.

## Deployment upkeep

This Compose file is the captured **GB10 deployment profile**. Its 114 GiB memory
limit, CPU set, local model path, 524,288-token context, and four-slot configuration
are specific to that host. Start a different machine with its own model, resource
budget, and context using the standalone command in the [fork README](../../README.md).
The runtime image's default entrypoint already invokes the CUDA compatibility
launcher and Harness; pass `serve` and its model/options after the image name.

The images default to the container's root user. Model files must be readable,
and the persistent `/cache` directory must be writable by the user selected for
the container. Keep a local model mount read-only when downloading is not
needed. Changing `--user` requires checking those paths and the runtime's home
directory; the verification stage makes its public project files readable to
unprivileged test workers. No model or kernel cache belongs in a source checkout.

For the GB10 profile, set `HF_CACHE`, `KERNEL_CACHE`, and `TENSORFOLD_IMAGE` for
the selected paths and built image, then use:

```bash
docker compose -f deploy/gb10/compose.yaml config --quiet
docker compose -f deploy/gb10/compose.yaml up -d
docker compose -f deploy/gb10/compose.yaml ps
docker compose -f deploy/gb10/compose.yaml logs --tail 100 -f tensorfold
curl -fsS http://127.0.0.1:8888/health
curl -fsS http://127.0.0.1:8888/v1/models
docker compose -f deploy/gb10/compose.yaml stop tensorfold
docker compose -f deploy/gb10/compose.yaml start tensorfold
```

Initial kernel compilation and full-model loading can take minutes. Read startup
logs and use the health endpoint to establish readiness before sending requests.
The profile's health check allows a 15-minute initial startup period. `stop` and
`start` retain the container; `restart tensorfold` restarts it. After rebuilding
with a new source commit and image tag, set `TENSORFOLD_IMAGE` and run `up -d`
to replace the service, then repeat health, model, generation and restart checks.
Retain the previous image tag for rollback until those checks pass. Updating
packages inside a running container would discard the build's recorded identity.

When changing features, flags, defaults, pins, deployment settings, or qualified
performance/validation results, update the root README and affected guide in the
same change. Follow [CONTRIBUTING.md](../../CONTRIBUTING.md) for source validation
and upstream merges. Documentation changes to `README.md` also change the wheel's
package metadata, so publish a fresh source capture and image identity while
reusing native compilation evidence only where its actual inputs still match.
