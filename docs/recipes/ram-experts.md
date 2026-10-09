# CUDA experts backed by system RAM

`--vram-experts GIB|auto` keeps Flash Next's affine 4-bit or EXL3 expert weights in pageable
system RAM and allocates a bounded shared cache on the GPU. This is useful on
machines with separate host RAM and limited NVIDIA VRAM. It uses the ordinary
CUDA kernels; it has no GB10 hardware dependency. The existing CUDA compute
capability requirement (8.9 or newer) still applies.

For example, in an environment with TensorFold's CUDA runtime and compiler:

```bash
tensorfold serve /models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --backend cuda --vram-experts 8 --ple-on-ssd \
  --context 32768 --parallel 4 --kv-dtype int8
```

The `8` limits **packed GPU expert tensor storage**, in GiB. Startup rounds down
to whole expert slots and reports the actual pool size. CUDA context, allocator,
non-expert weights, activations, vision, and KV buffers also need VRAM. Startup
admits these separately and refuses a context that does not fit. Choose the pool
and context for the machine's reported capacity; the example is not a capacity
guarantee for a particular GPU. Omit the flag to keep the fully resident path.

On a GPU with separate VRAM, use `--vram-experts auto` to size the expert cache from the VRAM remaining after
the model's other allocations. Attention and routing weights, KV caches,
recurrent state, and compute buffers stay on the GPU. Model weights and the
expert pool are shared by all parallel slots; each slot needs separate sequence
state. Automatic sizing accounts for the configured context, KV formats, MTP,
all configured parallel slots' scratch and sequence state, retained prompt
states, and transient workspace before choosing
whole expert cells. The chosen size is fixed for the engine's lifetime and is
reported at startup. It leaves allocation headroom rather than treating every
physically free byte as usable expert storage.

Automatic sizing materializes EXL3's lazy prompt workspaces before measuring
free VRAM and uses one 512 MiB allocation margin in addition to the explicitly
budgeted future state and workspace. It sizes the pool once before the first
expert lease; it does not compete with requests for memory at runtime. Unified
memory devices retain the existing numeric-budget behavior and reject `auto`.

```bash
tensorfold serve /models/Qwen3.8-Flash-Next-exl3 \
  --backend cuda --vram-experts auto \
  --context 2048 --parallel 4 --kv-dtype int8
```

Automatic sizing does not change precision or routing. A context or mandatory
model working set that cannot fit still fails startup; system RAM spillover
applies to expert weights only. EXL3's separately mapped n-gram tables retain
their existing host-memory behavior.

The baseline RTX PRO 2000 Blackwell run (16 GB VRAM, 64 GB RAM) with
`turboderp/Qwen3.8-Flash-Next-exl3` at revision
`65c895314393431c09050b2e04e250836b3a6eb4`, INT8 KV, a 2,048-token context,
four slots, and YaRN factor 2 chose 2,341 expert cells: 5,753,241,600 bytes
(5.36 GiB). It left 134,469,952 bytes for full-window KV growth and its copy
overlap, 1,387,954,176 bytes for retained/in-flight states, and 2,068,480 bytes
for prompt temporaries, plus the 512 MiB margin. Less than one expert cell
remained after those reservations. The final compact arena holds 4,632 resident
cells, with at most 4,583 arbitrary experts borrowed by one lease, within the
same 5,753,241,600-byte (5.36 GiB) expert-storage budget. Its packed GPU storage
uses 5,752,012,800 bytes after whole-cell rounding. This changes cell geometry; the
four full-context slots' scratch and sequence-state reservation remains
separate from the pool. The original expert payload authority occupies
30,948,556,800 bytes (28.82 GiB) in pageable host RAM, before other host allocations.

For the baseline arena's repeated 128-token greedy reply, a 0.5 GiB pool measured
6.28–6.33 decode tokens/s without drafts and 7.34–7.35 with warmed MTP.
Automatic sizing measured 9.21 and 11.96–12.00 respectively, with identical
output tokens. Warm MTP reused 45 prompt tokens. The serial request's expert
hit rate rose from 24.03% to 67.49%. These are observations for this model,
runtime, and prompts, rather than throughput guarantees or measurements of the
final compact arena.

A later controlled comparison of the baseline and compact arena completed
20 matched trials per region in alternating baseline/candidate/candidate/baseline
order. Complete request medians were 8.09→11.64 tokens/s for serial greedy,
12.76→21.95 with MTP, and 48.82→81.71 aggregate tokens/s for four simultaneous
requests. All 240 timed requests produced the exact 64-token teacher output.
Expert bytes copied per output token fell by 52.3%, 60.9%, and 61.4%, respectively.
Constructor time increased from 225–228 seconds to 280 seconds. These results
apply to the tested 2,048-token INT8/YaRN-2 region and intermediate source build;
they do not prove that a full 524,288-token window fits this GPU. See
[optimization validation](../optimization-validation.md) for the run-order,
uncertainty, exact sampling checks, and remaining final-build gates.

Four simultaneous requests with 1,805–1,808 prompt tokens and 64 output tokens
also completed. The GPU-thread observer saw all four slots active; minimum
observed free VRAM was 1,042,087,936 bytes, with no OOM or swap use. This first
long-prompt batch included prompt work and was the first batch of that shape;
compilation overhead was not isolated. It is not a warm four-request throughput
measurement. All measurements used the pinned
PyTorch `2.16.0.dev20261006+cu134`/CUDA 13.4 runtime.

The adapters support Flash Next MLX affine 4-bit/group-32 and EXL3 checkpoints
on one CUDA GPU. They include every routed expert, the shared expert in each layer,
and the MTP expert layer when drafts are enabled. EXL3 retains each original
trellis stream and bit width, including supported half-bit widths; all experts
within a layer must use the same native codebook. NVFP4, tensor parallelism,
MLX, and other model families refuse this flag. `--ssd-experts` remains the MLX
checkpoint-streaming option and is now explicitly refused on CUDA. With affine
checkpoints, `--ple-on-ssd` controls the separate n-gram tables. EXL3 checkpoints
map their native n-gram tables directly and refuse `--ple-on-ssd`.

## Residency and execution

The affine loader reads two experts' projections at a time directly into CPU memory and
packs its existing bits into the CUDA kernel layout. It never uploads the full
expert stacks first and never dequantizes or requantizes them. A single fixed GPU
pool is shared across main and MTP layers. Two expert-sized pinned buffers stage
cache misses; cold expert weights stay pageable.

For EXL3, CPU authority contains compact original trellis bytes without padding.
Fixed GPU cells are sized for the largest gate/up/down bundle in the model. With
numeric budgets or `--vram-experts auto`, final startup sizing can use smaller routed cells when all
routed bundles have one size and every named shared bundle has one larger size.
It reserves a cell for each shared expert and uses the remaining space for routed
experts, within the original packed-weight and publication-device budgets. Other
distributions retain fixed cells. Diagnostics distinguish resident cell capacity
from the maximum number of arbitrary experts that one lease can safely borrow.
Numeric pools are finalized after loading, before the first lease, using a
single bootstrap cell so arena replacement never duplicates a full pool.
Prepared
FP16 scales, logical pointer/width tables, bounded pointer-publication buffers,
and wave controls are admitted separately from the packed-weight pool. Original
codebook markers and scale payloads are validated at loading; nonfinite scales
are rejected before becoming GPU metadata.

The EXL3 loader batches complete expert bundles into bounded reads of at most
16 MiB, borrowing views until their copies finish. Larger bundles use sequential
projection reads. Read failures drain outstanding work and preserve the original
error before releasing the loader's resources.

The cache uses bounded aging frequency counters, with recency and physical slot
as deterministic tie breakers. Frequently reused experts can remain resident
across layer calls. The distribution can change: counters age, and every selected
expert remains eligible for execution. The router's choices, top-k weights,
quantization, pair ordering, and reductions remain unchanged. A request cannot
restrict routing to cached experts.

Larger pools rank missing routed experts together. A bounded private startup
calibration selects between exact repeated scans and partition selection; it
retains raw timings and runtime provenance. Both preserve frequency, recency,
aging, protected entries, and physical-slot tie order. Small pools and single
misses retain their original selection path. Original ranking also remains
available when a large pool's representative
calibration cannot fit its optional startup budget. This does not limit the
supported pool size or invent a ranking cutoff. Single-wave EXL3 execution uses the
original GPU routing table after validating the host snapshot, avoiding a second
routing-table upload.

A prompt or mixed batch can select more experts than the pool holds. Execution
windows the existing grouped plan items and remaps their expert IDs to cache
slots. Each window finishes gate/up and down before its slots may be reused.
Transfer staging and stream changes are ordered with CUDA events. The ordinary
writeback runs once after all selected experts finish. Single-request decode,
concurrent decode/prefill, vision's text layers, MTP, and YaRN use the same path.
Host-coordinated forwards run eagerly; full-forward CUDA graphs are disabled.

The EXL3 adapter preserves native logical expert IDs and row/slot order. Each
wave publishes leased pointers, executes its selected pairs, and retains the
other slots' outputs until all waves finish. It then performs the native weighted
combination once. Duplicate valid expert IDs within a row are rejected before
native grouping; the existing native limit is 32 route slots including shared.

CPU expert payloads and aliases are immutable for the cache's model lifetime.
The cache bounds storage, frequency history, resident metadata, and staging;
explicit close waits for outstanding consumers. Failed scheduling contains the
affected cache instead of continuing with partly installed entries. Consumer
errors propagate while completion events keep outstanding work ordered.

## Memory and performance

Enough system RAM is still required for **all packed experts**, plus loading
scratch, pinned staging, resident n-gram pages if used, and the rest of the host
process. Startup checks the host estimate independently on discrete GPUs. On a
unified-memory GPU, host weights and GPU copies consume the same physical pool;
the estimate accounts for both. Moving weights to RAM on that hardware does not
create additional physical capacity.

Misses transfer weights and synchronize the host plan. Large prompts commonly
touch many experts, so a small cache can make prefill substantially slower.
This feature adds capacity; it does not promise a throughput improvement over
fully resident execution. It is an independently implemented host-store/cache
path, inspired by the tiered residency idea described in
[FreeToken's project documentation](https://github.com/FlashML-org/FreeToken).
It does not implement FreeToken's CPU/GPU compute split or elastic KV/cache
reallocation.

The checked-in tests use independent CPU packed-byte oracles and the unchanged
resident CUDA execution as an oracle, including forced eviction, shared experts,
MTP, mixed requests, stream ordering, mutation/resource failures, and memory
bounds. A complete trained EXL3 checkpoint also passed 32 paired resident/cache
checks on GB10: all 25,137 bundles and 30,948,556,800 original trellis bytes matched,
as did routing, full-vocabulary logits, greedy and seeded MTP, n-gram rows, prefix
restoration, and storage growth. That run used 109 cells in a 267,878,400-byte pool,
with no container swap and at least 18.39 GiB of sampled available host RAM.
A separate 33-check run also matched a 112×112 image prefill through the original
BF16 vision tower, with at least 17.31 GiB of sampled available host RAM. These
functional runs do not establish language quality, broad vision accuracy, or
discrete PCIe performance. Those require separate measurements. The GB10 recipe uses resident
experts, YaRN 2x, and four request slots.

For a comparison on that machine, use identical checkpoint, runtime, context,
sampling, concurrency, and prompt/token distributions. Record cold startup and
cold/warm prefill, time to first token, decode rate, p95/p99 stalls, host and GPU
peak memory, cache hit/miss/eviction counts, and copied bytes. Compare fully
resident execution when it fits and several admitted cache sizes; synthetic
hit rate alone does not establish an end-to-end speed gain.

## Direct GPU reads from host RAM

A GPU can read suitably mapped host buffers while computing. On a discrete
GPU, those weight bytes still travel over the CPU–GPU interconnect. Avoiding an
explicit upload can help a coalesced, one-use access, but repeated reads can
make an initial copy into VRAM cheaper. NVIDIA describes these conditions in
its [zero-copy guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#zero-copy).

The grouped kernels reuse weight tiles across up to 16 routed rows in decode
and 64 rows in prefill. Additional tiles and later requests can reuse an expert
again. The current miss path stages immutable packed weights and queues their
upload before the consuming kernels. Ordinary CPU tensors cannot be passed to
these kernels: they require CUDA weight tensors. Direct access would require a
validated mapped allocation/alias and matching lifetime rules, or hardware and
driver support for another host-access mode. Unified-memory page faults may
trigger migration, as described in the
[CUDA memory guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html).

A possible hybrid would stream genuinely cold, single-use experts from mapped
host buffers and promote experts when reuse justifies it. Reading remotely and
immediately uploading the same weights can move their bytes twice. That choice
needs a complete miss/decode/prefill comparison on the intended discrete GPU,
including mapping, staging, promotion and memory costs. GB10's shared-memory
measurements cannot establish that PCIe tradeoff. The flag rename changes the
budget's name; the validated upload-and-cache policy remains the same.

## Reproducing the synthetic checks

From the repository root, the CPU boundaries and policy replay checks are:

```bash
python3 -m pytest tests/test_ram_experts.py tests/test_host_experts.py \
  tests/test_expert_cache.py -q
```

In a supported CUDA environment, run the packed-byte, cache lifetime and
complete Flash Next checkpoint/forward oracles:

```bash
python3 -m pytest tests/cuda/test_host_experts.py \
  tests/cuda/test_expert_cache.py tests/cuda/test_flashnext_ram_experts.py -q
python3 -m pytest tests/test_exl3_host_experts.py tests/test_exl3_ram_admission.py \
  tests/test_exl3_read_recorder.py tests/test_exl3_scale_boundary.py -q
python3 -m pytest tests/cuda/test_exl3_host_cache_cuda.py \
  tests/cuda/test_flashnext_exl3_ram_engine.py -q
python3 tools/bench_ram_expert_startup.py --repeats 9
python3 tools/bench_ram_expert_policy.py --repeats 9
```

The benchmark tools use the checked-in synthetic fixtures and consume the
complete native results. They retain per-sample timings, paired uncertainty,
cache work and memory receipts; the policy comparison includes an all-hit
warm-cache region. Use the pinned verification image and NVIDIA launcher
described in the [container recipe](../../deploy/gb10/README.md) on GB10. These
checks complement measurements with a real checkpoint on the intended machine.

The pinned NGC/nightly build passed 1,666 applicable Linux CPU checks
(six platform/fixture skips) and 194 affected CUDA checks. The built verification
image also passed all 28 feature checks with bytewise tensor comparisons.
Complete synthetic startup and cache-policy replays matched native packed
weights, generated tokens, logits and residuals. Recorded conditions, numerical
oracles, paired timings and limits are in
[the validation receipt](ram-experts-validation.json).

The option is now `--vram-experts`; its Python budget keyword and admission
receipt key are `vram_experts`. The previous spelling is refused. The rename
passed 1,669 packaged Linux CPU checks (six skips) and both actual CUDA cold
load/YaRN/MTP cases; runtime help and source/dependency audits passed. See the
[rename validation receipt](vram-experts-rename-validation.json). The original
performance receipt above records its own earlier source revision.
