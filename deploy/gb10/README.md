This integration reproduces the GB10 Flash Next deployment from repository
source. It includes the native SSD reader, multi-image/video and copy-draft
changes, xgrammar 0.2.8, and the Harness `/v1/tokenize` endpoint. The native
baseline is 262,144 prompt-plus-output tokens with four request slots.
`provenance.json` records the upstream commits, integrated patch hashes, model
revision, and the original container stack. The new build uses NGC CUDA 13.4.1
and upstream ARM64 PyTorch nightly 2.16.0.dev20261006, matching TorchVision
0.30.0.dev20261006 and Triton 3.9.0+gitaad2a60d. `nightly-pins.json` records the
base digest, official wheel URLs/hashes, and the date of the latest-available
selection. The current Compose profile enables YaRN factor 2 with 524,288
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
ARM64 manifest digest. That supplies NVCC, CUDA headers and forward-compatibility
libraries. Python 3.12 and upstream nightly packages live in an isolated venv;
NGC's tightly coupled PyTorch plugins are not part of this image. All Python
runtime dependencies are pinned to wheel URLs and SHA256 hashes, including the
nightly's required CUDA libraries and Triton. The inference base's sparse/solver
development headers are supplied by those wheels; `CPATH` makes their audited
include directories available to NVCC and C++. Transformers, tokenizers, the Hub
client, xgrammar and core model-input dependencies retain their deployed pins
where compatible. The native reader's C++ source is included in the fork wheel.
The verification target adds its own pinned pytest tools and contains no weights.

cuDNN uses the complete hash-pinned wheel provider. Build-time aliases in
`/opt/tensorfold-cudnn` cover its major, minor and full-version SONAME lookups;
that directory leads the library search path. The NGC inference base has a
reduced cuDNN library set, so ordinary wheel-directory precedence can mix
providers and omit the precompiled Conv3d engine needed by the vision tower.
Neither dependency tree is modified.

Compiled artifacts use `/cache/cu1341-torch216-dev20261006-triton39-aad2a60d/`,
separate from the serving container's older Torch/CUDA caches. Compiler jobs are
bounded to one to limit startup memory peaks. New Torch versions use OS advisory
locks: a persistent unlocked `lock` file is normal and must not be deleted as a
recovery step. TensorFold emits old stale-file guidance only for the audited
FileBaton implementation.

The current GB10 host driver is 580.178.04. A fresh container check demonstrated
NGC's enabled CUDA forward compatibility with user-mode driver 615.71.09 and
the nightly's CUDA tensor operations on SM 12.1. No host driver change was made.
The project CUDA launcher reruns NVIDIA's shipped compatibility probe on each
startup and refuses probe failure before executing the original NGC entrypoint.
This preserves its validation across restarts, when NGC's cached marker survives
but its process environment loses the probe result.
Full model startup, memory, long-context quality and performance require their
own deployment validation.

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

The nightly Triton bindings emitted nanobind reference warnings at interpreter
shutdown after the CUDA suite, which exited successfully. Small isolated compiler
and GPU probes did not reproduce them. Sustained full-model memory behavior still
needs validation; the warnings are retained rather than suppressed.

To refresh dependency locks, resolve the updated pins in the ARM64 Python 3.12
foundation environment with `pip install --dry-run --report REPORT.json`, then
run `python3 deploy/gb10/lock_dependencies.py --report REPORT.json`. Include the
three exact wheel pins and verification packages in that resolution. The
generator rejects changed nightly hashes or a different Python/architecture.
Update the image/cache identifiers with the pins and rerun CPU, GPU and model
gates. Do not hand-edit generated requirement locks.
