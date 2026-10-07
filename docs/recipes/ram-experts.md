# CUDA experts backed by system RAM

`--vram-experts GIB` keeps Flash Next's affine 4-bit expert weights in pageable
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

The first model adapter supports Flash Next MLX affine 4-bit/group-32 checkpoints
on one CUDA GPU. It includes every routed expert, the shared expert in each layer,
and the MTP expert layer when drafts are enabled. EXL3, NVFP4, tensor parallelism,
MLX, and other model families refuse this flag. `--ssd-experts` remains the MLX
checkpoint-streaming option and is now explicitly refused on CUDA. Neither flag
changes `--ple-on-ssd`, which controls the separate n-gram tables.

## Residency and execution

The loader reads two experts' projections at a time directly into CPU memory and
packs its existing bits into the CUDA kernel layout. It never uploads the full
expert stacks first and never dequantizes or requantizes them. A single fixed GPU
pool is shared across main and MTP layers. Two expert-sized pinned buffers stage
cache misses; cold expert weights stay pageable.

The cache uses bounded aging frequency counters, with recency and physical slot
as deterministic tie breakers. Frequently reused experts can remain resident
across layer calls. The distribution can change: counters age, and every selected
expert remains eligible for execution. The router's choices, top-k weights,
quantization, pair ordering, and reductions remain unchanged. A request cannot
restrict routing to cached experts.

A prompt or mixed batch can select more experts than the pool holds. Execution
windows the existing grouped plan items and remaps their expert IDs to cache
slots. Each window finishes gate/up and down before its slots may be reused.
Transfer staging and stream changes are ordered with CUDA events. The ordinary
writeback runs once after all selected experts finish. Single-request decode,
concurrent decode/prefill, vision's text layers, MTP, and YaRN use the same path.
Host-coordinated forwards run eagerly; full-forward CUDA graphs are disabled.

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
bounds. GB10 is available for functional CUDA validation; discrete PCIe transfer
throughput and a full host-offloaded production checkpoint require measurements
on the intended discrete-GPU machine. The existing live GB10 deployment keeps
its resident experts, YaRN 2x, and four request slots.

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
