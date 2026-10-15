# GB10 Flash Next deployment

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
recorded image/source identities. The current ARM64 deployment is recorded below;
the separate AMD64 shipping checks retain their own scope in
[optimization validation](../../docs/optimization-validation.md).
The current Compose profile enables YaRN factor 4 with 1,048,576
prompt-plus-reply tokens, `rotorquant8-norm` K/V and four request slots. Caches grow on demand within
the global memory gate; four simultaneously full-size contexts are not assumed
to fit the GB10.

New builds from the maintained `python-0.6` integration use TensorFold **0.6.6**.
The Harness checks both the installed package version and module version,
validates its preparation interfaces, and binds the CUDA HTTP factory and its
server export together. Native authentication also covers `/v1/tokenize`.
The historical deployment receipts below identify their original images;
qualify each new build's launcher, model behavior and performance separately.

The Python integration has passed native installed-package, launcher and
synthetic CUDA checks, including all 256 ordered KV-format pairs on both GB10
and RTX PRO 2000 Blackwell, with complete receipt and retirement reinspection.
Isolated GDN candidates also passed native correctness checks at SM 121;
full-model performance selection remains pending. The user authorized deployment
of the qualified Python image on 2026-10-12, based on these completed gates.
The live deployment and rollback details are recorded below. Controlled
baseline/performance comparisons remain separate and unrun. See the
[source/image-specific qualification](../../docs/optimization-validation.md#python-line-integration-2026-10-11).

## Decode-bandwidth round 2 deployment, 2026-10-14

The service runs source `a05c3219` (deploy commit `ff20feba`) in image `sha256:506b3913…`, container `875a8b95…`,
with `rotorquant8-norm` K/V, an 8 GiB reserve and kernel cache `cache-python-0.6-bw-a05c3219`. Attention value prep,
the eight-bit encoder, RoPE and the merge changed; the DeltaNet chain trees return to 8 value rows a warp on this chip
(4 on the RTX PRO 2000); the lane matmul and expert window launches are unchanged here (their new per-chip tables hold
only RTX PRO 2000 entries). Every reply keeps the previous build's tokens. On GB10 a staged test container reproduced
the reference replies at ~30k through ~650k, and a container A/B against the replaced build measured decode +1.2 to
+2.0% (geometric mean +1.7%), prompt passes +0.2 to +0.7%. After the swap, reference replies at ~30k and ~150k
matched, live tokenization, thinking-off, thinking and tool-call checks passed, and Harness reports the provider
compatible. Every Triton kernel run since the swap has machine code already run on this GB10. The replaced container
(`52b8f229`) is retained stopped as `qwen38-flash-next-tf-rollback-bw-20261014`. See
[the deployment receipt](deployment-bw2-20261014.json) and the
[selection evidence](../../docs/optimization-validation.md).

## Decode-bandwidth build deployment, 2026-10-14

From 21:46Z to 23:01Z the service ran source `52b8f229` (deploy commit `1314ca05`) in image `sha256:6c528ff6…`, container `16c40da8…`,
with `rotorquant8-norm` K/V and a new kernel cache (`cache-python-0.6-bw-52b8f229`). Six- and eight-bit decode
attention, the decode hyper-connection read-outs and the DeltaNet chain trees changed; every reply keeps the previous
build's tokens. On GB10 a staged test container reproduced the previous build's replies at ~30k, ~150k, ~300k and
~650k, and a container A/B measured decode +5.9% (5.4 to 6.5%) and prompt passes 1.44 to 1.59x. After the swap,
reference replies at ~30k and ~150k matched, live tokenization, thinking-off, thinking and tool-call checks passed,
and Harness reports the provider compatible. Every Triton kernel run since the swap has machine code already run on
this GB10. The replaced container is retained stopped as `qwen38-flash-next-tf-rollback-rq8-20261014`. The
memory reserve then rose from 6 to 8 GiB (owner decision): 110.11 GiB startup budget, 28.4 GiB for stream caches. See
[the deployment receipt](deployment-bw-20261014.json) and the
[selection evidence](../../docs/optimization-validation.md).

## RotorQuant 8 KV restart, 2026-10-14

By owner decision the service stores keys and values as `rotorquant8-norm`
again (it ran `rotorquant6-norm` from 2026-10-13). The image (`da53a063…`,
source `fd0205a0`), environment, binds and kernel cache are unchanged; only
`--kv-dtype` differs (compose `298054a4`). Container `145d4b86…` started at
04:35:32Z and was healthy by 04:38:02Z: startup estimate 100.01 GiB within
111.95 GiB, 30.0 GiB free for stream caches and 17.47 GiB for one stream at the
full 1,048,576 window. Live tokenization, thinking-off, thinking and tool-call
checks passed, and Harness reports the provider compatible. This source's
RotorQuant 8 KV kernels (42 of the 44 Triton entries compiled after the restart)
had not run on this GB10 before. A host monitor and `memguard.sh` (4 GiB floor)
watch the service. The GB10 had gone down at 02:57:49Z under the RotorQuant 6
service on a host-wide out-of-memory (NVRM `NV_ERR_NO_MEMORY`, then the kernel
OOM killer) while one conversation stood at ~608k tokens. RotorQuant 8 keys and
values take about 32% more memory per token. The RotorQuant 6 container is
retained stopped as `qwen38-flash-next-tf-rollback-rq6-20261014`. See
[the deployment receipt](deployment-rq8norm-20261014.json).

## Performance build deployment, 2026-10-14

The service now runs source `fd0205a0eb7959ebd71cfa83da56c53c4a17d09e` in ARM64
image `sha256:da53a063ac505851a60457b9bfdf03fff59c448e30c2c062189649a83042148d`
(`tensorfold-gb10-fork:local`), container `86d986371bdc…`, cut over at
2026-10-14T00:53:11Z. It adds the decode and prompt-pass stages that the
[2026-10-13 GB10 A/B](../../docs/optimization-validation.md) kept: one-launch
and multi-CTA sparse-attention select, static round tables with captured CUDA
graphs, one host wait a step, late n-gram reads, programmatic dependent launch
and the RotorQuant 6 prompt-pass gather. Its kernel sources equal the A/B
image's. Only the image and a new kernel-cache bind
(`cache-python-0.6-perf-20261013`) changed; `TENSORFOLD_MULTI_GRAPH_MIB` keeps
its 512 MiB default and `TENSORFOLD_MEMORY_RESERVE_GIB` stays 6. Startup
reported 96.76 GiB within the 112.31 GiB budget, 31.0 GiB for stream caches,
14.22 GiB for one stream at the full 1,048,576 window, and 61 verify and 30 MTP
step graphs in 504 MiB. A same-configuration test container and then the
service returned the reference tokens for warmup, ~30k and ~150k prompts; four
concurrent streams returned their one-stream tokens. Every Triton kernel they
compiled has machine code that had already run on this GB10. Live tokenization
metadata, thinking-off, thinking and tool-call checks passed, and Harness
reports the provider compatible. Four-stream throughput at the default graph
budget, full-model quality and filled-window capacity remain unmeasured. The
replaced container is retained stopped as
`qwen38-flash-next-tf-rollback-perf-20261013` (image `sha256:ca62f894…`,
audit cache directory). See
[the deployment receipt](deployment-perf-20261014.json).

## Audit build and RotorQuant 6 deployment, 2026-10-13

The service now runs source `cadcae5ce7e729cc4e6eadae44552f9b02dd1985` in ARM64
image `sha256:ca62f8944734e5526b099a4cd1801d58a043af372e32196a01f747276f136e74`
(`tensorfold-gb10-fork:local`), container `3a8d027f2753…`. It adds the audit
waves (reply correctness, fp32 logits, prefix-cache retention and kept points,
keepalive/cancel) and, by owner decision, stores keys and values as
`rotorquant6-norm` instead of `rotorquant8-norm`, with a new kernel-cache bind.
Startup reported 96.25 GiB within the 112.28 GiB budget and 14.22 GiB of cache
for one stream at the full 1,048,576 window. Live tokenization matched the previous
service's count and discovery metadata; short thinking-off, thinking and tool-call
checks passed, and Harness resolved its thinking profile. Full-window and
RotorQuant 6 quality and performance comparisons remain unrun. The replaced
container is retained stopped as `qwen38-flash-next-tf-rollback-audit-20261013`
(image `sha256:55a38d89…`, `rotorquant8-norm`, old cache directory). See
[the deployment receipt](deployment-audit-rq6-20261013.json).

Before this replacement the GB10 hung on unified-memory exhaustion. Unified GPU
allocations are not charged to the container's memory limit, so observe host
memory as well.

## Harness discovery bridge deployment, 2026-10-12

The service now runs source `9acb28c395683a7b2fddc70904942396f902c7ca` in ARM64
image `sha256:55a38d8985529e21370a6fd2acd4b7f8c88106661c01b7224f41e671d502d71f`
(`tensorfold-gb10-fork:local`), container
`d58000d027c802b00cca345b24022cf0e7eb40ff2bde7e56ed0305a9c7146847`. It changes
only the Harness launcher: `/v1/tokenize` replies now carry the discovery
metadata described below. The installed TensorFold package, entrypoint scripts,
dependencies, Compose arguments, environment, binds and limits are unchanged.
Live tokenization returned the fingerprint of the served template and its
reasoning efforts, with tokens identical to the previous service; a complete
generation passed, and Harness resolved its thinking profile from the metadata.
The replaced container's exact configuration is retained, stopped and outside
Compose ownership, as `qwen38-flash-next-tf-rollback-bridge-20261012` (image
`sha256:8e1fe707…`); that image lacks the metadata, so Harness refuses its
thinking-off maintenance against it. See
[the deployment receipt](deployment-harness-bridge-20261012.json).

## Python integration deployment, 2026-10-12

The service ran **TensorFold 0.6.6**, source
`2edfaa515d85ca48a57365552f38a97a17123c3c`, in qualified ARM64 image
`sha256:8e1fe707502d99e8565af2900bd862847141a9da30779b208c239cbb21ee44be`.
Later qualification-only documentation does not substitute a different image.
Container `612a761a5b98d8fab842cdb0b05c6a8d04edbfd33527ede6e3471cc64fc1f2f6`
serves the same model at `http://192.168.1.50:8888/v1`.

The replacement preserves the model revision, 1,048,576 context, YaRN-4,
RQ8-norm K/V, four slots, MTP-4/confidence 0.5/copy mode, 2,048 prefill rows,
vision/thinking, 114 GiB/no-extra-swap, CPU set, PID limits, health check and
all 81 effective environment values. The derived kernel-cache host bind is
isolated as described below. Local and LAN health/model identity, the installed
module/package version, Unicode tokenization, the independent answer 391 for
23 × 17, complete HTTP/SSE bodies and full token-ID sequences passed. Target-only,
MTP and warmed MTP produced the same 64 tokens; four concurrent 64-token requests
and their warmed repeats passed with exact fixed-seed sequences.

First-use MTP/shared batches were slow; their raw observations are retained.
Warmed functional repeats completed in approximately 1.00 s for one 64-token
reply and 2.03 s for four 64-token replies. These are finite functional
observations, not a controlled baseline comparison, statistical throughput gain
or full-window quality/capacity result. The same process stayed healthy with
zero restarts and the observed container flag `OOMKilled=false`. Image/video/tool/grammar
queries and filled-1M stress were not part of this deployment batch.

The exact previous image/configuration/cache is retained in
`qwen38-flash-next-tf-rollback-python-20261012`, created and stopped, outside
Compose ownership. Its immutable image is
`sha256:5b6627ba69d9cd7107016b55e37503a356e108a1ef5ec91bf6ce7740b478efec`.
Stop and verify retirement of the active service before starting that rollback
on the same port. Manage the service through this maintained Python
checkout's Compose file; the default `tensorfold-gb10-fork:local` tag now points
to the bridge image above. Keep both image IDs and cache directories.
See [the deployment receipt](deployment-python-20261012.json) for exact identities,
settings, successful checks and coverage limits.

## Current YaRN-4 / RotorQuant8-norm configuration

The service at **192.168.1.50:8888** uses **YaRN factor 4**, a configured
**1,048,576-token prompt-plus-reply limit**, and **`rotorquant8-norm` for both
keys and values**. The model revision, four slots, sampler, MTP/copy, SSD reads,
vision/video, thinking and resource limits are preserved. The current image is
the qualified Python deployment above. This configuration was introduced by
three Compose argument changes on 2026-10-10; the Python image replacement on
2026-10-12 retains those values. The existing Compose project name
`tensorfold-gb10-yarn2` is retained to preserve service ownership.

Startup identifies the two effective KV formats and **1.1386294** rotary
amplitude. The runtime reports **17.47 GiB** for one complete full-window growing
cache, versus the calculated INT8 forecast of 17.88 GiB at the same new context.
This forecast includes scales, BF16 indexer/pooled state and MTP; it excludes
weights, recurrent state, snapshots and transient prompt workspace. It is not
observed peak memory. Four slots do not imply four full windows fit.

See [the configuration-change receipt](deployment-rq8norm-yarn4-20261010.json)
for bounded functional/restart results, the initial verification rollback,
cold media timeout and remaining qualification limits. No matched current-image
INT8/RotorQuant speed comparison or filled-million-token quality claim is made.
Consumed JSON/SSE, same-policy draft/serial identity and positive prefix reuse
with a matching fresh reference passed before restart. The same-container
restart retained its ID/image/arguments, changed the process start and returned
healthy. After restart, shared-service validation passed image/video recognition
and counting, the typed `add_tags` call, xgrammar `TFOK` and four synchronized
short SSE arithmetic clients. The first image trial reached a 60-second client
watchdog; subsequent media checks used 180-second deadlines and retained exact
output oracles. No new resource failure was observed in the captured snapshots.
Individual cancellation retirement could not be attributed under shared traffic
and remains unqualified; no global-idle or four full-window claim follows.
The prior INT8/YaRN-2 qualification below retains its original scope.

## Earlier ARM64 redeployment with INT8 and YaRN-2

On **2026-10-10**, the GB10 at **192.168.1.50:8888** was redeployed from the
maintained fork with the same affine **4-bit/group-32** Flash Next checkpoint,
revision `2b170fa6309d5d1ee380b35636075fac7945f286`. Its approximately 113.2 GB
of checkpoint assets include the SSD-backed n-gram table; disk size is not a
resident-memory figure. The selected built artifacts are:

| Artifact | Recorded identity |
| --- | --- |
| Built source commit | `5fe0e89d4697cf2c2f152a720df7de1c0d90ec8c` |
| Native ARM64 runtime image | `sha256:5b6627ba69d9cd7107016b55e37503a356e108a1ef5ec91bf6ce7740b478efec` |
| Application wheel SHA256 | `4ea0280726172616432a20f770ffd4fdbd98be7ceadf1d1915267ae2ebbae391` |

All **465 packaged source files** were audited. The selected image is tagged
`tensorfold-gb10-fork:local`; the immutable image ID above identifies the tested
build rather than a later documentation commit or rebuild.
The maintained [deployment receipt](deployment-20261010.json) records the exact
configuration, measurements, completed checks and evidence identities.

That earlier plain Compose deployment preserved **four slots**, **524,288 configured
prompt-plus-reply tokens**, **YaRN-2**, INT8 K/V, MTP-4/confidence 0.5,
`TENSORFOLD_MTP_COPY=1`, SSD n-gram reads, vision/video, thinking and the existing
sampler defaults. The memory and swap limits remain **114 GiB** with no extra
swap budget. The CPU set `5-9,15-19` retains the ten fast cores. Selected changes
are **2,048 prefill rows** and **20 compiler jobs**; a job limit does not prove
all workers remain busy. This resident unified-memory deployment does not
enable RAM-backed expert offload.

Startup observations reported **28.8–29.1 GiB of cache room** and **8.94 GiB
for one full window**. Caches grow under the runtime memory gate; this does not
establish that four simultaneously full windows fit. No new filled-524k prompt
was exercised on this image. The historical 523,882-token retrieval below
belongs to its earlier image.

Actual short greedy HTTP/SSE generations kept the checkpoint, INT8 KV and
request settings fixed. Three 128-token draft/serial pairs matched their serial
targets and the earlier 512-row baseline outputs. One concurrent warmup was
excluded before three consumed batches of four 128-token replies:

| Selected 2,048-row workload | Observed client tokens/s |
| --- | ---: |
| One MTP request, median of three | 57.40 |
| One serial request, median of three | 36.21 |
| Four concurrent MTP requests, median aggregate | 115.16 |

The three concurrent samples were **115.16, 115.98 and 114.63 aggregate
tokens/s**. Completion usage includes reasoning tokens, and timings cover the
complete consumed HTTP response. They are not GPU emission timestamps or each
concurrent user's rate. The original four-draft configuration measured 114.23
aggregate tokens/s and its repeat measured 116.10, so these trials do **not**
establish a short-decode throughput improvement. Six drafts gave no repeatable
gain; disabling copy drafts gave about a 1% gain that did not survive the
baseline repeat, so four drafts and copy drafts remain selected.

For a fresh **1,886-token prompt plus 64 reply tokens**, serial requests reported
zero cached tokens. The 512-row baseline's reported prefill time was about
0.98 s; the selected 2,048-row confirmation median was **0.2355 s**, with samples
**0.2413, 0.2341 and 0.2355 s**, and the recorded output hash matched. This
prefill observation has a different timing scope from complete-response rates.
These are small, warm-cache samples from the recorded starts, with no
population-confidence or sustained full-context quality/performance claim.

The current image passed six feature groups: image and video recognition,
their generation-free `/v1/tokenize` counts (150 and 302), media capability
reporting, the required typed `add_tags` call (integer 42, list and boolean true),
and xgrammar's exact `TFOK` result. An ordinary thinking-enabled LAN arithmetic
request completed with **888** for 37 × 24. A same-container restart passed:
the container ID/image stayed fixed, its start time changed, Docker health
became healthy, and LAN health advertised the configured 524,288 context and
four slots. Restart loading took **25.9 s**, including **1.6 s** prompt warming.
**All six feature groups passed again after restart**, and a fresh ordinary
thinking-enabled LAN arithmetic request again completed with **888**. Final
observations showed no container OOM kill, zero container swap and **33.42 GiB
host MemAvailable**. Unified GPU allocations are not fully charged to the
cgroup, so host memory must also be observed; this one observation is not a
guaranteed free-memory floor. This record does not relabel the earlier exhaustive
AMD64 or historical GB10 checks as new runs.

Build from the repository root:

```bash
docker buildx build --platform linux/arm64 --load --target runtime \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  --build-arg MAX_JOBS=20 \
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

Generic image defaults put compiled artifacts under
`/cache/cu1341-torch216-dev20261006-triton39-aad2a60d/arm64/` or the sibling
`/amd64/`, separate from each other and older Torch/CUDA caches. This GB10's
Compose profile retains **ARM64-only, unsuffixed**
`/cache/cu1341-torch216-dev20261006-triton39-aad2a60d/` directories for Torch
extensions, Triton, CUDA, Inductor and XDG caches. Its default host bind is now
`/home/kklouzal/services/qwen3.8-flash-next/cache-python-0.6-perf-20261013`,
separate from the earlier images' `cache-python-0.6-audit-20261013`,
`cache-python-0.6-2edfaa51` and `cache` binds.
Retain those old directories for rollback. Use a new `KERNEL_CACHE` directory for
each subsequent image replacement; matching ABI or module names alone do not
establish native artifact compatibility. These deployment overrides do not
change the generic image defaults or authorize sharing native caches with AMD64.
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

The historical pinned image was deployed on this GB10 on 2026-10-06 at native 262,144
tokens/four slots. Full-model startup, 17 API checks before and after a real
same-container restart, four simultaneous requests, 20 repeated requests, tool
result round trips, two-image input and MP4 video recognition passed. The final
library/launcher fix passed 111 CUDA regressions and eight launcher boundary
tests. The previous container and image were retained stopped for rollback.
These are bounded functional checks; 512k YaRN quality/capacity and sustained
memory/performance validation remain separate.

That earlier image's factor-2 profile also passed startup, 17 short API checks and four concurrent
requests. One 523,882-token prompt plus 18 reply tokens retrieved the exact
synthetic passphrase at 60% depth without truncation. The trial took 442 seconds
and included three simultaneous short requests, all correct in about 2.3 seconds
while the large prompt was prefilling. This single retrieval is a bounded
capacity/quality check; it does not establish uniform long-context accuracy.

`compose.yaml` preserves the current sampler, MTP, vision and model/cache mounts,
request slots, memory/swap settings, CPU set, health checks, and restart policy.
Cache paths explicitly preserve this host's existing ARM64 namespace as described
above. Its container
name and port are the existing production service's. Rendering the
configuration is read-only; applying it is a service replacement and belongs to
the deployment step after validation.

The current Compose project is named `tensorfold-gb10-yarn2`, separating the
retained native fork container from the new managed service. After a controlled
cutover, manage it from this repository with
`docker compose -f deploy/gb10/compose.yaml up -d`, or restart the existing
container with `docker restart qwen38-flash-next-tf`. A simple rename of a
Compose-owned old container is not safe rollback preservation: its Compose
ownership labels can still cause replacement or removal. During the historical
2026-10-10 cutover,
the old container was renamed and then removed by that label match. The saved
old image `sha256:089bfba448433506ec3e2fe6332848824c299e7358f76996af44e4e6b6df389f`
and exact saved container/host configuration were used to recreate
`qwen38-flash-next-tf-rollback-20261010`, with only Compose ownership labels
removed. It is retained in **Created/stopped** state. Stop the current service
before starting a rollback on the same ports; validate its health and complete
generations. Capture the previous configuration before future replacements,
and keep rollback containers outside the active Compose ownership labels.
For the current Python deployment, use the old `5b6627ba…` image and
`qwen38-flash-next-tf-rollback-python-20261012` identified in
[the 2026-10-12 deployment section](#python-integration-deployment-2026-10-12).
The earlier recipe and Harness launch scripts manage their captured images;
use this fork's Compose configuration for the new deployment.

The Harness endpoint counts the native rendered prompt and image expansion
without generation. It permits counting a history beyond the served context,
while retaining generation admission and independent image safety checks.
It requires `--thinking-budget 0`; callers select budgets per request. Returned
token/prompt inspection is capped at 16,384 tokens, and count-only responses
support larger histories. HTTP preparation has a bounded queue and cancellation.

Each successful counting reply also carries the metadata Harness discovers
models from, resolved once when the handler is built; changing the template or
model state requires a restart. `capabilities.reasoning_efforts` lists the
efforts the initialized template names, and `default_reasoning_effort` appears
only when `--reasoning-effort` resolves to one of them. `chat_template_sha256`
hashes the UTF-8 encoding of the template source, read with ChatTemplate's own
precedence (`chat_template.jinja`, then `tokenizer_config.json`) and text
decoding. It is reported only when that source compiles, in the active Jinja
environment, to the active root and block code with the same defaults and
referenced bindings; changed, invalid or unreadable source omits it.
`provenance` labels that identity `equivalent_active_compilation` and records
the server, engine and tokenizer types, loaded tokenizer serialization and
Python bytecode hashes with their Python and marshal versions. The bytecode
hashes distinguish installed Python implementations; they do not identify engine
state or non-Python code. Thinking controls, default-thinking and
reasoning-replay facts stay absent: forwarding a keyword to Jinja proves neither
that the template consumes it nor that it can turn thinking off. Harness obtains
those from the contract it pins for the fingerprinted template. Without a
fingerprint Harness has pinned, Harness establishes no thinking control and
refuses requests that select thinking on or off, including its compaction,
memory, goal-ledger and skill-learning requests. The contract follows Harness's
`deploy/tensorfold/README.md`.

YaRN is an optional CUDA text-RoPE policy for Flash Next. The CLI option
`--yarn-factor 2 --context 524288` requests a 2× window without modifying the
shared model snapshot. Native operation omits that option. A full 512k validation
must measure memory and long-context quality; four full 512k streams are not
assumed to fit. Four slots are retained for mixed workloads. The historical
startup logged 29.8 GiB of cache room; the current startup observations above
reported 28.8–29.1 GiB. One full window needs 8.94 GiB; four full windows need
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
preparation, plus discovery metadata: template fingerprint precedence;
rejection of changed, invalid, unreadable or missing source and of changed
blocks or bindings; effort defaults; and code provenance. YaRN tests compare frequencies against the pinned Transformers
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
curl -fsS http://127.0.0.1:8888/stats
docker compose -f deploy/gb10/compose.yaml stop tensorfold
docker compose -f deploy/gb10/compose.yaml start tensorfold
```

The profile sets `TENSORFOLD_FAILURE_CAPTURE_DIR=/cache/failure-captures`, inside the kernel-cache bind, so on the
host it is the `failure-captures` directory under `KERNEL_CACHE`. The server writes there only when a reply fails in
a way it names: a reply that ended at an end token with its think block still open (finish `stop`, reasoning and
no answer) prints one warning line with its request id, and is captured there with its request (image and video parts
redacted), generated token IDs and statistics. A normal reply's request is never logged or captured (its `done` line
carries its request id, counts, timings and a token hash only). The directory holds at most 32 files and 256 MiB; a
capture that would pass either is skipped with a line saying so, so remove captures once read. Files are readable by
the container's user only; the captured prompts are the user's data. Remove the variable to keep the warning line
alone.

`GET /stats` (JSON) and `GET /metrics` (Prometheus) report load, KV memory, prefix-cache occupancy and totals for
monitoring and for Harness's background-work gating; neither is logged when it succeeds.

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
