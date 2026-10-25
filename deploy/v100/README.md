# Flash Next on eight V100 GPUs

The [adaptive implementation](../../docs/v100-adaptive-validation.md) changes
GGUF GPU-history admission to computed full-context capacity. Its unique-history
and compact-Planar4 paths require new native/model/quality/performance
qualification; historical live receipts below describe their original layouts.
The [governing specification](../../docs/v100-adaptive-inference-design.md)
defines the complete stage and acceptance gates.

This AMD64 development profile builds the installed TensorFold wheel with the
SM70 FP16 provider. Original GGUF encodings remain separate from affine/EXL3/
NVFP4 readers. The same family engine, scheduler, HTTP server, prefix states,
sampling and cleanup contracts are used. Full-model validation is recorded in
[the qualification note](../../docs/v100-flashnext-validation.md); do not infer
quality or throughput from the component matrix alone.

## Selected foundation

The native AMD64 CPython 3.12 profile pins the official CUDA **12.9.2** development
image by child and index digest, with official **Torch 2.14.1+cu126**,
**TorchVision 0.29.1+cu126** and **Triton 3.8.0** wheels by SHA256. CUDA 12.9.2 is
the compiler toolkit; Torch's wheel runtime is CUDA 12.6. They are deliberately
recorded separately. A newer CUDA toolkit alone does not prove that a framework
wheel contains Volta kernels. The probe requires `sm_70` in Torch's actual
compiled targets and executes the selected operators on a V100.

The cu126 wheels are selected because newer cu128/cu129 PyTorch wheels removed
Volta. See [PyTorch's packaging notice](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290)
and [CUDA 12.9.2 compiler targets](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-compiler-driver-nvcc/index.html#gpu-name-gpuname-arch).
Triton is retained as a framework dependency; this SM70 provider does not invoke
Triton kernels. The GB10 managed runtime, native-library policies and CUDA 13.x
locks are unchanged. This separate development profile is not yet a qualified
managed production foundation or a complete audit of all family APIs.

The lock files are generated with the existing `deploy/gb10/lock_dependencies.py`
renderer, using [pins.json](pins.json) and an actual Linux AMD64 pip resolver
report. Never hand-edit the generated locks. To regenerate, build the resolver
target, mount a new evidence directory, and resolve the three exact wheel URLs
in `pins.json` plus the required component/test dependencies with pip's
`--dry-run --ignore-installed --only-binary=:all: --report`. Pass that report,
the pins, `architecture="amd64"`, and its raw SHA256 to `render`, then publish
with `write_locks` into this directory. Preserve the report as private evidence;
the generated manifest retains its identity and dependency closure.

## Build and check

From the repository root:

```bash
docker build --target verification -f deploy/v100/Dockerfile -t tensorfold-v100:verified .
docker run --rm tensorfold-v100:verified -m pytest -q \
  tests/test_flashnext_gguf.py tests/test_flashnext_gguf_weights.py \
  tests/test_gguf_assets.py tests/test_sm70_mtp_weights.py \
  tests/test_flashnext_tp_control.py tests/test_flashnext_tp_scheduler.py
docker run --rm --gpus 'device=0' -v tensorfold-v100-cache:/cache \
  tensorfold-v100:verified -m pytest -q tests/cuda/test_sm70.py \
  tests/cuda/test_sm70_gguf.py tests/cuda/test_flashnext_sm70_family.py \
  tests/cuda/test_flashnext_sm70_mtp.py
docker run --rm --gpus all --ipc host -v tensorfold-v100-cache:/cache \
  tensorfold-v100:verified -m pytest -q tests/cuda/test_flashnext_sm70_collectives.py \
  tests/cuda/test_flashnext_sm70_tp_scheduler.py
docker run --rm --gpus 'device=0' -v tensorfold-v100-cache:/cache \
  tensorfold-v100:verified /verification/v100_probe.py --probe-gpu
docker run --rm tensorfold-v100:verified -m pytest -c /verification/pytest.ini -q \
  /src/tests/test_deployment_pins.py /src/tests/test_verification_public_permissions.py
```

The last command uses source-relative deployment fixtures while importing the
installed wheel. Other commands use tests outside the source import path.
Package data includes all native CUDA/C++ sources. The cache key binds source,
headers, architecture, compiler, flags and framework ABI; `MAX_JOBS=4` bounds
native builds. Keep this cache separate from other serving stacks.
The probe prints a component receipt with the provider, GGUF and GDN native
binary hashes, compiler identities, SM70 cubins and bounded execution checks.
Its optional `--checkpoint /models/first-shard.gguf` inventories a read-only
model mount without loading the full model.

## Serve the original checkpoint

Stop the existing GPU consumer before loading a second full model. Keep its
container and immutable image for rollback. Replace the example model path with
the first shard (a metadata-only first shard is supported); all sibling shards
must be present. The model mount remains read-only.

```bash
docker run -d --name tensorfold-flash-next --gpus all --ipc host \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 --ulimit core=0 \
  -p 127.0.0.1:8000:8000 \
  -e PYTHONUNBUFFERED=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 \
  -e TF_GGUF_ASSETS_DIR=/cache/gguf_assets \
  -v tensorfold-v100-cache:/cache -v /path/to/models:/models:ro \
  tensorfold-v100:verified /src/tools/launch_cuda_tp.py --gpus 0,1,2,3,4,5,6,7 -- \
  /models/Flash-Next-00001-of-00004.gguf --backend cuda \
  --host 0.0.0.0 --port 8000 --name flash-next \
  --context 4096 --parallel 1 --no-drafts --kv-dtype fp16 --no-update-check
```

`launch_cuda_tp.py` supervises ordinary CLI rank processes, exposes one GPU to
each, supplies a shared rendezvous, and retires all ranks on a worker failure or
termination. Rank zero serves HTTP. Use the existing endpoint authentication
options before changing the loopback-only host port publication.

GGUF tokenizer/config assets are reconstructed from the checkpoint into the
cache. The original four files are neither copied nor rewritten. Large routed
experts retain original packed bytes on GPU; bounded decoded tiles feed FP16
matmuls with FP32 accumulation. Ordinary projections are decoded once at load.
PLE remains a bounded, read-only host mapping. At TP8 the actual checkpoint
retains approximately 12.59 GiB of weight tensors per rank, before cache,
workspace, framework allocator and communication overhead.

The baseline command deliberately starts with a bounded context. Admission
uses the smallest rank budget, includes complete quantization blocks at partial
shards and accounts for replicated KV heads. Increase `--context` and
`--parallel` only with the resulting capacity and workload evidence. The explicit optimized profile and its graph qualification are described below.
A graph microtest alone does not establish full-model graph readiness.

For a matching separate stock MTP head, replace `--no-drafts` with
`--drafter /models/mtp-stock --mtp-drafts 2`. The loader accepts the existing
floating checkpoint directory or a shared-target GGUF companion entrypoint.
The revised design pins `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` at
`unsloth/Qwen3.8-Flash-Next-GGUF` revision
`766911a6b7369840a91dbcd95f9f997acaab6cd6`; verify the size and SHA256 in the
[specification](../../docs/v100-adaptive-inference-design.md#stock-mtp).
Reads and decode staging are bounded. Routed GGUF banks retain their packed
bytes, small projections execute in FP16, and the target embedding/head remain
shared. This importer explicitly selects QSA compression four and token budget
2,048; it records the companion's zero final compression entry without treating
it as a dense-attention instruction. Ordered target/draft token IDs must match.
`/stats.capacity_plan.mtp_import` identifies the import policy. CPU import
controls are separate from pending SM70 execution, target-conditioned refresh,
BF16-control comparison, quality, and performance qualification.

Prepare the official BF16 MTP control from only its indexed tensor ranges:

```bash
PYTHONPATH=src python tools/v100_mtp_prepare.py \
  --repository Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --out /models/mtp-stock-bf16
```

Run this from the checkout in the verification environment, with exclusive
ownership of the destination. `--source-dir /models/original` instead reads an
immutable local checkpoint. The tool copies original `mtp.*` bytes with an
8 MiB transfer bound, emits tensor/file checksums and source-header identities,
and resumes verified complete tensors from its `.partial` directory. It reads
source metadata and MTP ranges; complete target shards are unnecessary. Its
preparation envelope allows up to 16 GiB of selected payload and temporarily
holds both tensor fragments and the assembled control. The initial official
control contains 31 tensors and 5,214,301,696 payload bytes. Import preparation
does not establish QSA execution parity or transfer quality qualification from
the BF16 control to the quantized companion.
`--kv-dtype int8`, or mixed `--kv-key-dtype`/`--kv-value-dtype` FP16/INT8 pairs,
select the existing INT8 H32 codec and its scales; they do not mean FP8 or
RotorQuant. The `rotorquant-signed128-4` and `rotorquant-signed64-norm4` SM70 paths use
their existing signed bases, packed four-bit centroid indices and FP32 scales.
They have separate measured quantization-quality tradeoffs; see below. Other formats fail before allocating model weights
on SM70.

The maintained [Compose profile](compose.yaml) selects the matching stock MTP
head, automatic full-window session capacity and an 8,192-token window by default. Set `MODEL_ROOT` to the
host model directory and `GGUF_FILE` to its container first-shard path, then:

```bash
MODEL_ROOT=/path/to/models GGUF_FILE=/models/Flash-Next-00001-of-00004.gguf \
  docker compose -f deploy/v100/compose.yaml up -d
```

`TF_CONTEXT`, `TF_PARALLEL`, `TF_KV_DTYPE`, `TF_PORT`, `TF_BIND_ADDRESS`,
`TF_MODEL_NAME`, `MTP_DIRECTORY`, `TF_CACHE_VOLUME` and `TENSORFOLD_V100_IMAGE`
configure that profile. `TF_PARALLEL` and `TF_KV_FULL_WINDOWS` default to `auto`;
numeric overrides constrain automatic capacity rather than forcing unsafe
allocation. GGUF GPU-history slots reserve their full prompt-plus-reply windows.
The optional legacy RAM-indexer path requires explicit numeric settings and
retains its separately documented accounting; it does not inherit unique GPU
history capacity. Other providers' shared-growth contracts remain separate.

## Implementation and evidence

Native operators cover FP16 RMSNorm/RoPE, mixed-GGUF decode and GDN chain/replay.
Shared FP16 family paths cover HC, PLE, routing, causal indexed attention and
cache reads/writes, while preserving FP32 recurrence. Dense/expert adapters
retain the existing weights and forward interfaces. TP8 uses the ordinary NCCL
communicator and per-peer control acknowledgements. Concurrent TP uses the
existing scheduler and per-request state ownership.

Rebuild the image after source changes and validate the installed wheel.
Independent scalar/FP64 oracles, real checkpoint slices, tiny family execution,
and full-model answers have separate evidence scopes in the
[qualification note](../../docs/v100-flashnext-validation.md).

The qualified host deployment uses the same profile with `TF_CONTEXT=237598`,
`TF_PARALLEL=6` and `TF_KV_DTYPE=int8`. Its checks exercise bounded real prompts,
not a complete 237,598-token quality benchmark or six simultaneous full windows.
The operator environment pins image digest
`sha256:b223074bfe57d6a4c9a706a49b0eb97ecc0ed73d37ff93170ba419470f655d8d`.
The original service is preserved as a stopped rollback container. During first
startup, `/dev/sda` reported real SATA/Btrfs errors; see the qualification note.


## Select the drafter

Flash Next CUDA accepts `--draft-method auto|mtp|dspark`. `auto` preserves
checkpoint-based selection; explicit methods reject a checkpoint of the other
kind before loading the target. `--drafter` supplies the checkpoint directory.
DSpark currently uses the GGUF SM70 text provider, the complete target vocabulary
and the existing position-keyed target verifier. Stock MTP and DSpark remain
separate providers in the same engine and scheduler. `--no-drafts` selects serial
execution; `--mtp-drafts` controls proposal depth for either provider.

For Compose, keep the stock defaults or set:

```dotenv
TF_DRAFT_METHOD=dspark
DRAFTER_DIRECTORY=/models/dspark-checkpoint
TF_DRAFTS=3
TF_DRAFT_CONFIDENCE=0
```

`DRAFTER_DIRECTORY` overrides the legacy `MTP_DIRECTORY` variable; both paths
refer to directories inside the read-only model mount. DSpark checkpoint metadata
and tensor shapes are checked before target payload loading. Greedy, confidence0,
full-depth proposal blocks may use the admitted graph path; stochastic sampling,
short final blocks and declined graph keys use the ordinary proposal provider.
The target sampler remains the authority in each case. Experimental throughput
and outstanding numerical qualification are recorded in the
[validation note](../../docs/v100-flashnext-validation.md).


## Historical replicated-layout operator profile

Set `TENSORFOLD_SM70_PROFILE=optimized` before starting the ordinary CUDA
launcher, or `TF_SM70_PROFILE=optimized` for Compose. The general default remains
`reference`. The optimized profile checks the pinned Torch/cuBLAS/workspace and
Volta identity plus TP8 Flash Next geometry before admission and payload loading.
Its operator/runtime policy participates in the existing worker agreement.
These qualification results belong to the original replicated history layout.
The adaptive unique-history path currently declines graph capture and retains
its own source/native/model qualification gates.
It selects canonical `packed_exact` experts (R≤32), original shared Dense math,
prepared linears, native family operations and the admitted graph paths for
DSpark and MTP. Native attention is qualified only for local H3/KV1/D256 and
real query ends65–8192. Other decode windows use the original Torch operator at
each query's actual end and decline unsupported graphs before binding; the
allocated KV window and ordinary context limit remain unchanged.

C21 passed both fixed target-distribution suites325/389 independently, with
full top-1 agreement; see the [quality record](../../docs/v100-flashnext-validation.md).
Live method/depth performance and final service selection are still in progress.

## Equal full context windows

For adaptive resident GGUF with GPU history on TP1/2/4/8, use
`--parallel auto --kv-full-windows auto`, or supply an integer ceiling.
Every selected lane allocates the same complete admitted window, including MTP or DSpark
auxiliary KV. Normal prefix retirement keeps these allocations. The existing
4GiB device floor remains, with explicit 2GiB serving workspace and protected
cache replacement; graph publication cannot spend those reservations.

Compose defaults both `TF_PARALLEL` and `TF_KV_FULL_WINDOWS` to `auto`.
Counts need not be even or powers of two. The legacy RAM-indexer path uses
explicit numeric counts with N≥2 and its separate admission policy. Modern
providers and the one-GPU automatic expert-pool mode retain their existing
growth/full-window contracts.

Native attention scratch is capped at its qualified8192-token dispatch range,
while KV remains allocated at the full context limit. Longer queries retain
original Torch attention. Targeted large-cache/refusal/replay and fallback
allocation controls passed. Actual full-service allocations and short-prompt
timing are recorded in the C22/Q4 sections below.

## Measured full-window host configuration (C22)

The context-first candidate on this eight-V100 host uses stock MTP depth3,
confidence0, INT8 KV and three complete262144-token windows. All eight ranks
materialized three262148-slot target/MTP caches (8833339008owned cache bytes
per rank). The R12 concurrent graph workload retained653821400bytes within
692445232bytes of graph room, with no context or reserve reduction.

After building the pinned image and setting existing model/cache variables:

```bash
export TF_SM70_PROFILE=optimized TF_CONTEXT=262144 TF_PARALLEL=3
export TF_KV_FULL_WINDOWS=3 TF_KV_DTYPE=int8 TF_DRAFT_METHOD=mtp
export DRAFTER_DIRECTORY=/models/mtp-stock TF_DRAFTS=3 TF_DRAFT_CONFIDENCE=0
docker compose -f deploy/v100/compose.yaml up -d
```

DSpark remains maintained and selectable. Its best depth in the bounded2/3/7
screen was3: use `TF_DRAFT_METHOD=dspark`,
`DRAFTER_DIRECTORY=/models/dspark-shadowpa-3f74bdfa`, `TF_DRAFTS=3`,
`TF_PARALLEL=2`, `TF_KV_FULL_WINDOWS=2`, and `TF_CONTEXT=262140`.
The model root must contain the matching pinned checkpoint. Actual memory
admission, native profile pins and rank agreement remain authoritative; these
counts are specific to this checkpoint/host/configuration. Full-window cache
allocation is not a full262K generation or numerical-quality benchmark.

For the same128-input/512-output quiet greedy screen, MTP D3 measured
52.86/53.29/58.93decode tokens/s; DSpark D3 measured50.41/53.92/54.67.
MTP retained127prompt tokens on its final repeated prompt; DSpark's two-slot
configuration evicted that prefix. These are configuration screens, not an
isolated-method causal gain or a statistical equivalence claim.100single-request
decode tokens/s remains unmet. All final full-window API cases and plain service restoration/verification
passed; the [validation record](../../docs/v100-flashnext-validation.md)
tracks their final status.

The plain shipping service is healthy on port8000 with three equal262144-token
windows. Its128-input/512-output quiet decode measured54.04/54.42/59.78tokens/s;
all complete benchmark outputs and three simultaneous responses matched the
qualified candidate. Existing C3 remains stopped for rollback. The finite
closeouts are complete;100single-request tokens/s remains unmet.

## RotorQuant Q4 validation

Use `--kv-dtype rotorquant-signed128-4` for both sides, or combine it with
`--kv-key-dtype` / `--kv-value-dtype` to select independent FP16/INT8/Q4 sides.
The descriptor and codec identity are unchanged from the existing fork format.
The V100 implementation uses FP16 working Q/K/V operands with FP32 reductions
and inverse value rotation; it does not invoke Triton or reinterpret INT4 bytes.
Native/same-format live checks and actual four-window allocations passed;
full714quantization comparisons and quiet throughput are reported below.
The result is not INT8 quality equivalence.

### Q4 configuration and acceptance

Build/select a Q4-capable verified image first. The old C22 INT8 rollback
image rejects Q4; an existing host `.env` can otherwise keep that old image
pinned. The signed128 Q4 live configuration has four equal full native windows:

```bash
export TENSORFOLD_V100_IMAGE=local/tensorfold-v100:rotor-q4-r5
export TF_SM70_PROFILE=optimized TF_KV_DTYPE=rotorquant-signed128-4
export TF_CONTEXT=262144 TF_PARALLEL=4 TF_KV_FULL_WINDOWS=4
export TF_DRAFT_METHOD=mtp DRAFTER_DIRECTORY=/models/mtp-stock
export TF_DRAFTS=3 TF_DRAFT_CONFIDENCE=0
docker compose -f deploy/v100/compose.yaml up -d
```

Each rank materialized four262148-slot caches (context plus speculation),
8069964032owned bytes, with4GiBdevice floor and2GiBworkspace preserved.
Use `TF_KV_DTYPE=rotorquant-signed64-norm4` for the existing norm-preserving
variant. Both formats force cold preparation in serial and concurrent serving
until prompt-prefix reuse meets same-format identity checks. Request-local KV
caching, MTP, cache precision, copy/resize and mixed-side interfaces remain.
Repeated prompts therefore have cold TTFT rather than the INT8 prefix-hit cost.

Signed128 Q4 passed21serial/draft pairs,4concurrent replies,SSE,cancellation and
recovery. Short128-input/512-output greedy decode measured51.2476/50.2409/
50.2750tokens/s, with coldprefill1.6512/1.6418/1.6255s. These are descriptive
configuration screens; format/output/cache state differs from the INT8 screen,
so no isolated speed gain/equivalence or global optimum is claimed.
The broader quality suite measuredPPLratio1.006522/top1agreement.974293 for
signed128, versus1.008041/.976864 for signed64norm; old numerical comparison
gates failed for both. Their native implementation gates passed separately.
Production selection remains subject to the user's quality/latency preference.

The canonical live procedure compares against fresh replies in the SAMEformat,
not old INT8 IDs, and explicitly enforces the current prefix capability:

```bash
python tools/v100_rotorquant_live.py --url http://127.0.0.1:8001 \
  --cases-dir /path/to/frozen-request-data --output-dir /path/to/new-results \
  --parallel 4 --tokens 128 --context 262144 \
  --kv-dtype rotorquant-signed128-4 --prefix-reuse cold
```

For source-relative tool tests inside the verification image (installed wheel
imports, matching `/src` fixtures), use:

```bash
docker run --rm -e PYTHONPATH=/src tensorfold-v100:verified -m pytest \
  -c /verification/pytest.ini -q /src/tests/test_v100_rotorquant_quality.py \
  /src/tests/test_v100_rotorquant_live.py
```

A `/verification/tests` invocation alone lacks the tool-source fixtures; do not
repair installed code to compensate. Build from canonical source and preserve
the source/image identities reported by the quality and live tools.

Q4 DSpark was also checked through real requests. Three full262140windows
left too little graph room for the longer corpus and the cohort was retired.
The bounded compatible configuration uses3equal245713-token windows with
1GiBgraph allowance, preservingtheexisting4GiBfloor/2GiBworkspace, and passed
all27serial/draft/concurrent/SSE/cancel/recovery checks. It is not a claim of
three simultaneous full262KDSpark workloads or of better output quality.

## NUMA-local indexer archive

`--indexer-history ram` replaces full raw GPU indexer history with a bounded
GPU ring and staging buffers plus a lossless FP16 archive in local system RAM.
`--indexer-history gpu` is the default. Attention KV and all pooled indexer
keys remain on the GPU. The archive is independent of the selected KV format
and drafter. TP8 stock-MTP live checks passed four full INT8 windows and eight
260,060-token Q4 windows, including frozen same-format output comparisons,
concurrency, SSE, cancellation and recovery. These are allocation and bounded
workload checks; full-context payloads and RAM/DSpark remain unqualified.
The host currently runs INT8 with GPU history and three full native windows;
the preceding throughput measurements use their named configurations.

Each rank resolves its actual GPU UUID/PCI device to a NUMA node, binds owned
anonymous allocations before first write, verifies every backing page, and
registers only bounded transfer stages. CPU archive copies run on that node's
CPUs. Missing placement capabilities fail explicitly. CPU affinity alone does
not establish memory locality. On this host GPUs0–3 attach to node0 and
GPUs4–7 to node1; runtime discovery also supports CUDA ordinal remapping.

Docker's ordinary default seccomp policy blocks the required NUMA operations.
Generate the narrow override from a temporary CPU-only container using this
host's actual default policy and the intended immutable TensorFold image:

```bash
docker run -d --name tf-numa-default --network none \
  --entrypoint /bin/sleep "$TF_IMAGE" 300
sudo python3 tools/generate_numa_seccomp.py --container tf-numa-default \
  --output-directory /absolute/new/numa-profile
docker rm -f tf-numa-default
```

The generated profile preserves the captured filter and grants only static
node binding without migration, policy queries and self page-location queries.
It requires neither extra capabilities nor unconfined seccomp. Regenerate and
requalify after changes to Docker/runtime defaults, capabilities or CPU/node
allowances. Keep the selected node in the container's allowed memory set.

For a direct run, add `--security-opt
seccomp=/absolute/new/numa-profile/seccomp.json` to Docker and
`--indexer-history ram` to the TensorFold command. For Compose:

```bash
export TF_INDEXER_HISTORY=ram
export TF_PARALLEL=4 TF_KV_FULL_WINDOWS=4
export TF_NUMA_SECCOMP=/absolute/new/numa-profile/seccomp.json
docker compose -f deploy/v100/compose.yaml \
  -f deploy/v100/compose.numa.yaml up -d
```

Use an archive-capable image built from the matching canonical source. The
optional override is unnecessary for GPU-history mode. Startup reports host
archive/pinned staging reservations separately from GPU cache storage. Preserve
the existing device floor and workspace while selecting equal context windows;
saved raw-history bytes alone do not establish how many full windows fit.

With automatic graphs and every RAM window already materialized, startup may
refine graph room from the minimum physical spare across ranks after the same
device floor and future workspace/copy/snapshot promises. The original header
receipt is retained beside `observed_graph_admission`. Explicit graph ceilings
and GPU-history behavior retain their existing admission policy. This does not
make all graph shapes fit or guarantee unchanged throughput.
