# TensorFold V100 Adaptive Inference Design

**Platform:** Supermicro SYS-4029GP-TVRT with eight V100 SXM2 GPUs, 32 GiB each.\
**Purpose:** implementation requirements for maximum useful prompt-processing and decode throughput across the supported concurrency range, with automatic capacity planning and no runtime tuning restarts.\
**Design basis:** the complete discussion and the user-supplied startup accounting dated 2026-10-25 04:16 UTC.\
**Status:** consolidated implementation design. Reported observations, arithmetic estimates, proposed mechanisms, and required qualification are identified separately. The exact private fork and target/draft artifacts remain integration inputs.

## 1 Governing operating contract

The administrator selects the target checkpoint, context allowance, numerical formats, speculative method, and any service limits. TensorFold computes a truthful full-context session capacity and selects qualified execution plans for the actual workload.

The performance objective is the highest useful token throughput the selected model, quality, context, and hardware permit at every relevant concurrency step. Prompt-processing throughput and committed decode throughput are measured separately. Low-load responsiveness, aggregate throughput, and mixed-workload latency are evaluated together; one undifferentiated tokens-per-second counter is insufficient.

### Settled requirements

| Area | Governing requirement |
|---|---|
| Hardware | Optimize for this eight-V100 SXM2 server, including its actual NVLink, PCIe, NUMA, and SM70 behavior. |
| Target model | One distributed model, initially retaining TP8 weight placement and the qualified resident weight inventory. All eight ranks remain available at low concurrency. |
| Context | Equal administrator-selected context allowance for admitted sessions; prompt plus generated reply share that allowance. The reported model limit is 262,144 tokens. |
| Capacity | Maximum resident/admitted sessions is computed from the complete worst-rank budget. It is not a permanently fixed 20, 24, 32, or power-of-two count. |
| Speculation | Stock MTP is the first enabled method to qualify. DFlash is a separate selectable method with its own compatible checkpoint and allocation description. None remains an explicit selectable method. |
| Q4 | Integrate a pinned RotorQuant-family four-bit format first. SAW-style evaluation follows after the core system works. |
| Q8 | Retain a precisely specified ordinary Q8 format. Do not invent an unspecified native RotorQuant eight-bit encoding. |
| Histories | Remove redundant KV and raw/pooled indexer storage. The primary design keeps unique attention and indexer histories in VRAM. |
| Adaptation | Adapt batches, kernels, graphs, prefill budgets, proposal lengths, and justified attention placement to actual work. |
| Restarts | Runtime adaptation must not restart the container, workers, or CUDA context. A setting without a safe online transition stays at its qualified maximum-concurrency configuration. |
| Migration | Between-batch changes and bounded state-preserving cache migration are authorized. Preserve every retained session and its context commitment. |
| Simplicity | Refactor the serving path directly. Use one homogeneous numerical/speculative configuration for retained sessions initially. Avoid simultaneous speculative systems, duplicated full histories, and unnecessary framework layers. |
| Evidence | Capacity and throughput claims require complete accounting and representative execution. Header arithmetic, startup allocation, and live workload qualification are different evidence classes. |

The first full-context MTP qualification candidates are **16 Q8 sessions** and **24 RotorQuant Q4 sessions**, after unique-history storage is implemented. These are planning candidates, not measured maxima or hardcoded limits.

### Boundaries of the design

The target's sparse/indexed attention, recurrent equations, masks, pooling, positional conventions, and sampling semantics remain the defined computation. This design changes storage ownership and execution scheduling around those semantics. Quality-changing codec choices require separate qualification.

Hardware failure, driver failure, and software deployment are separate operational concerns. The no-restart requirement governs normal serving adaptation and supported administrator reconfiguration; it is not a claim of recovery from every GPU or process fault without restart.

Pin the exact target checkpoint and fork; cache geometry alone is not a model identifier. Public TensorFold upstream interfaces must not be assumed to match the custom fork described by the Torch-based receipt.

## 2 Hardware and software baseline

### Physical platform

The platform supports eight V100 SXM2 GPUs on the X10DGO-SXMV-P carrier with the X11DGO-T host board. It uses a cube-style NVLink mesh and switched PCIe 3.0 CPU-GPU paths; it is not an NVSwitch system. The advertised 300 GB/s NVLink figure is aggregate bidirectional bandwidth per GPU, not bandwidth for every GPU pair or for host transfers. Sources: [Supermicro system specification](https://www.supermicro.com/en/products/system/4U/4029/SYS-4029GP-TVRT.php), [system manual MNL-2115](https://www.supermicro.com/manuals/superserver/4U/MNL-2115.pdf), and the [joint Red Hat and Supermicro reference architecture](https://www.redhat.com/rhdc/managed-files/pa-supermicro-deep-learning-openshift-reference-architecture-f19703-201910-en_0.pdf).

The previously reviewed system topology has shared PCIe uplinks and switches serving GPU pairs. Capacity and performance work must not assume eight independent host x16 links. CPU model, DIMM population, actual RAM capacity, OS placement, and physical GPU enumeration must be inventoried on the machine used for qualification.

Required platform manifest:

- GPU UUID, device index, logical rank, memory capacity, ECC state, and supported peer access.
- NVLink link state and pairwise transfer measurements.
- PCIe switch/root association and NUMA affinity.
- Actual CPU topology, memory channels/population, and available host RAM.
- Driver, CUDA, compiler, framework, NCCL, kernel-library, and profiler versions.
- Power limits, sustained clocks, temperatures, throttling, and link errors during performance runs.

Use the real outputs of tools such as `nvidia-smi topo -m`, NVLink status queries, CPU/NUMA inventory, and transfer/collective benchmarks. Logical KV-head groups must remain consistent with the weight shards when physical device assignment is chosen.

### SM70 execution stack

Build and qualify native SM70 code. NVIDIA documents that CUDA 13 removes offline compilation support below compute capability 7.5 and identifies CUDA 12.9 and the R580 driver branch as the maintenance direction for older architectures. Pin compatible releases across the entire dependency chain; successful package installation does not prove that its generated or bundled kernels support V100. Source: [NVIDIA architecture support guidance](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/).

Volta's Tensor Core matrix operands are FP16, with FP16 or FP32 accumulation. Packed Q4/Q8 storage must be consumed through an explicitly supported computation path. Do not assume native INT4, INT8, FP8, or BF16 Tensor Core paths from newer GPUs. Source: [Volta tuning guide](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html).

Use a profiler release that still supports Volta. Upstream FlashAttention or a modern FA3/FA4/Triton execution recipe is not evidence of a working SM70 implementation for this custom sparse, rotated, quantized attention path. Verify each dependency and kernel backend explicitly.

Keep CPU scheduling and transfer workers NUMA-aware. Bound pinned host staging. Do not pin an entire evictable file mapping without a measured need and a host-memory budget. One worker per GPU is the initial process model; one coordinator controls distributed execution. An alternative process model requires its own synchronization and performance qualification.

## 3 Reported deployment and accounting evidence

This section preserves the supplied receipt as a baseline. The local evidence JSONs listed in Section 25 are on the user's machine and have not been independently inspected in this workspace.

### Current reported configuration

- Eight GPUs at 32 GiB each: 256 GiB nominal total VRAM.
- INT8 K/V, GPU-resident raw and pooled indexer history.
- Optional raw-history RAM archive disabled.
- Three sessions, each with 262,144 logical context tokens.
- 262,148 physical rows per session in this verification geometry.
- Twelve target attention caches and one stock-MTP cache.
- Thirty-six recurrent layers using fixed per-session state instead of complete KV histories.
- Two global KV heads: ranks 0–3 retain head 0 and ranks 4–7 retain head 1 in the reported layout. Each head therefore has four physical copies.
- Both indexer histories are replicated across all eight ranks.

The rank numbers above describe the reported logical placement, not a claim about the best physical NVLink grouping.

### Startup physical reconciliation

| Bucket | GiB per GPU |
|---|---:|
| Resident weights, retained loader payload plan | 13.421 |
| Context caches, exact owned tensors | 8.227 |
| Other active Torch tensors and rounding | 1.218 |
| Torch cached but currently unallocated blocks | 0.800 |
| CUDA, NCCL, driver, and libraries outside Torch | 0.945 |
| Physically free at this startup snapshot | 7.389 |
| Total | 32.000 |

Torch allocated memory is already contained in reserved memory. Do not add them. Graph captures and requests alter this snapshot; a startup receipt does not establish later graph occupancy or peak serving memory.

### Retained weight plan

| Component | GiB per GPU |
|---|---:|
| Routed packed experts | 9.700 |
| Backbone projections | 1.936 |
| Stock MTP | 0.667 |
| Embedding | 0.629 |
| Routers | 0.235 |
| Vocabulary heads | 0.195 |
| Shared experts | 0.055 |
| Small arrays and maps | 0.004 |
| Total, rounded | 13.421 |

The eight actual rank plans total approximately 107.371 GiB. This is one sharded model, not eight complete copies. The illustrative weight size 14,411,046,312 bytes is approximately 13.421333 GiB; actual per-rank payloads and runtime allocations must govern admission. KV quantization does not change model weight encoding.

The reported recurrent/tail/replay subset is 27.798 MiB per session per GPU. The explicit workspace plan is about 0.640 GiB per GPU, mostly expert-prefill tiles. These are components of existing accounting, not additional totals to add without reconciling lifetimes and categories.

### Current history footprint

| Three sessions at 262,148 physical rows | GiB per GPU |
|---|---:|
| INT8 K/V payload | 4.875 |
| K/V scales | 0.305 |
| Raw FP16 indexer history | 2.438 |
| Pooled FP16 indexer history | 0.609 |
| Total | 8.227 |

The logical-token rates below are from the reported formats. Pooled indexer cost is averaged across groups of four tokens.

| Storage per token on each reported rank | INT8 | Existing Signed128 Q4 |
|---|---:|---:|
| K+V payload and metadata per attention layer | 544 B | 272 B |
| Raw FP16 indexer per layer | 256 B | 256 B |
| Pooled FP16 indexer per layer, amortized | 64 B | 64 B |
| Thirteen caches with GPU history | 11,232 B | 7,696 B |
| Thirteen caches with optional raw-history RAM archive | 7,904 B | 4,368 B |

Current INT8 metadata uses FP16 scales per 32 values. Existing Signed128 Q4 uses FP32 scales per 128 values. RotorQuant's schema is separately defined in Section 13; equal byte counts do not make the formats interchangeable.

One current logical token costs 89,856 physical bytes across the box in INT8/GPU-history mode. Three sessions means three total, not three per GPU.

### Reported qualification history

| Reported configuration | Sessions | Context per session | Status in supplied receipt |
|---|---:|---:|---|
| INT8 and GPU history | 3 | 262,144 | Current qualified startup/request path |
| INT8 and raw-history RAM archive | 4 | 262,144 | Previously live-qualified |
| Existing Signed128 Q4 and GPU history | 4 | 262,144 | Previously live-qualified |
| Existing Signed128 Q4 and raw-history RAM archive | 8 | 260,060 | Previously live-qualified |

These Q4 qualifications do not qualify RotorQuant. Header-only INT8/GPU-history alternatives were 4×211,270, 5×168,236, and 6×139,312. They were not live-qualified and did not add a separate fixed graph reservation.

The old-layout six-million-token calculations remain valid for those layouts:

| Six million total logical tokens | Cache GiB per GPU | Cache plus weights GiB per GPU |
|---|---:|---:|
| INT8 and GPU history | 62.764 | 76.185 |
| Existing Q4 and GPU history | 43.005 | 56.426 |
| INT8 and raw-history RAM archive | 44.167 | 57.588 |
| Existing Q4 and raw-history RAM archive | 24.408 | 37.829 |

These exclude all other obligations and all exceed 32 GiB. They are not a universal upper bound for a new unique-history architecture. Six million near-native tokens also require at least 23 sessions under the 262,144-token per-request limit.

## 4 Accounting rules and complete resource ledger

The reported planner starts after initial runtime allocations and protects a 4 GiB/GPU device floor. Its observed serving budget is 27.086 GiB/GPU. It also protects 2 GiB/GPU of serving workspace and, for the three-slot geometry, about 0.273 GiB of future snapshots and 0.211 GiB of replacement overlap. Graph allowance is separately bounded.

These are resource promises and limits, not necessarily allocated slabs. Reconcile each tensor and promise once. Do not subtract an already included floor twice, or add planner promises to the physical startup snapshot.

Every retained or peak allocation must have:

- An owner rank and allocation size including real alignment and padding.
- A persistent, session, batch, graph, replacement, or migration lifetime.
- A reason it must overlap another allocation, or a proven reason it can reuse an arena.
- A maximum derived from the effective profile and implementation limits.
- A measurement or construction proof for allocations made by libraries outside the explicit allocator.

The per-rank budget includes target and draft weights; target and draft history; recurrent, convolution, pooling, and draft state; feature buffers; execution and verification scratch; accepted-prefix recovery; graphs; communication; replacement overlap; metadata; padding; and bounded migration occupancy.

Use maximum workspace size only for proven mutually exclusive lifetimes. Sum genuinely overlapping occupancy. Do not allocate a maximum slab independently per slot when one bounded execution arena suffices, and do not reuse storage while an asynchronous reader remains.

The current 2 GiB workspace allowance, 0.640 GiB explicit plan, and 27.798 MiB/session subset need direct lifecycle reconciliation before being combined. Snapshot and replacement costs must be recalculated for the selected session count and speculative method while preserving the required recovery semantics.

## 5 Performance objectives and workload definitions

### Separate quantities

| Quantity | Definition |
|---|---|
| Configured context allowance | Maximum prompt-plus-reply length promised to each session under the effective profile |
| Resident capacity | Maximum sessions whose complete context and necessary state can be retained under that promise |
| Resident sessions | Sessions currently holding retained history/state |
| Active requests | Admitted requests with pending or in-flight inference work |
| Scheduled decode requests | Ready requests included in the next execution step |
| Target query or verification rows | Exact target rows executed, including speculative candidates according to the method convention |
| Prefill tokens | Prompt tokens scheduled in the current step |
| Tentative cache rows | Uncommitted target/draft storage needed by the selected speculative procedure |
| Emitted length | Tokens delivered or committed to the externally visible sequence |
| Target-cached length | Positions already consumed into the target's cache and state |
| Draft-valid length | Target context through which the drafter has the required valid conditioning/state |

The final three counters are not universally equal. An emitted anchor or bonus token can still await target consumption. Each method must specify its position and commit convention.

One active request with all permitted histories resident is a required workload. It does not have the memory freedom of one resident request. Idle retained histories are not available workspace.

### Optimization objective

Measure prompt tokens processed per wall-clock second separately from committed output tokens per wall-clock second. Include useful per-request output rate, TTFT, TPOT percentiles, queue delay, and fairness. Rejected speculative candidates are not delivered output tokens.

At low activity, avoid artificial waits to fill batches. At high activity, maximize aggregate committed throughput within administrator-selected latency limits. Under mixed load, preserve progress for both prefill and decode. Optimize the complete iteration, including routing, features, draft maintenance, drafting, verification, rejection/replay, communication, sampling, and commit.

There is no established tokens/s target or universal winning plan for this exact fork. Report a measured performance frontier for relevant operating points rather than transferring modern-GPU benchmarks or treating peak theoretical arithmetic throughput as an inference result.

## 6 Administrator configuration and effective profiles

### Inputs

Expose only implemented, qualified choices:

- Target checkpoint/revision and actual weight encoding.
- Equal context allowance, within the target's supported position limit.
- Target KV format and its pinned implementation.
- Speculative method: none, stock MTP, or DFlash.
- Compatible DFlash artifact and supported draft weight/KV choices when applicable.
- Explicit history-residency policy.
- Optional session ceiling.
- Performance objective and any requested latency or waiting-time limits.

A draft sliding-window override is exposed only if the selected checkpoint and storage implementation support it. Target and draft weight formats, target KV, and draft KV are independent dimensions.

### Outputs

The engine resolves the full-context resident capacity, admission limit, allowed execution envelopes, actual row budgets, graph/kernel families, placement, and per-batch scheduling. It reports the binding rank/resource and qualification status.

Illustrative schema, not existing TensorFold flags:

```yaml
context_tokens: 262144
kv_format: q8
speculation:
  method: mtp
  proposal_length: auto
max_sessions: auto
history_residency: gpu
performance:
  objective: throughput
```

An explicit integer session setting is an administrator ceiling constrained by physical capacity. It does not force an unsafe allocation. A requested minimum capacity, if later offered, is a validation requirement: an infeasible request fails clearly instead of silently shortening context or changing quality.

Keep configured, pending, and effective profiles distinct. The profile key includes target and draft revisions, codecs, relevant numerical settings, hardware topology, kernel build, runtime dependencies, and execution geometry. A plan qualified for another checkpoint, codec, or GPU is not automatically transferable.

## 7 Automatic capacity planner

The planner calculates the largest supported resident count for a useful qualified execution envelope:

```text
Nmax = max N such that, for every rank r:
    target_weights[r] + draft_weights[r]
  + target_histories[r, N, context, codec]
  + draft_histories[r, N, draft_policy]
  + retained_state[r, N]
  + retained_recovery_checkpoints[r, N, method]
  + peak_execution_and_recovery[r, envelope]
  + graph_and_communication_residency[r]
  + replacement_obligations[r, N]
  + migration_overlap_limit[r]
  + metadata_and_padding[r]
  <= usable_serving_budget[r]
```

The execution envelope includes supported target rows, prompt token budgets, proposal limits, and allowed operation overlap. Capacity is not obtained by shrinking execution resources until speculation is nominally enabled but unusable.

The retained recovery term includes mandatory per-session checkpoint/snapshot versions and their required future occupancy. The peak term covers additional execution-time recovery work. Reconcile shared storage once; do not omit retained snapshots or charge the same checkpoint in both terms.

Implementation sequence:

1. Resolve actual model geometry, loader payloads, codecs, and speculative compatibility.
2. Measure or establish the runtime baseline and per-rank usable budget.
3. Evaluate a small set of qualified execution/placement envelopes for the selected objective.
4. Compute exact per-rank persistent and peak costs for candidate resident counts.
5. Apply implementation limits and an optional administrator ceiling.
6. Initialize the chosen arenas and bounded graph resources.
7. Verify measured allocation against the plan and publish the effective profile.

Use exact allocated strides, page/tail rounding, real rank ownership, and library residency. Logical token-rate formulas are cross-checks, not the admission allocator.

Every admitted session reserves its future growth to the selected complete window. Do not increase guaranteed capacity based on short current prompts, expected inactivity, presumed prefix sharing, or unimplemented future offload.

Computed capacity can be a non-threshold value. A result of 18 remains 18. The policy buckets are execution aids, not an instruction to round down or a permanent 32-session limit.

### Method-specific resource description

Each method supplies its retained weights and state, history geometry, proposal convention, target verification rows, per-session target/draft tentative rows, feature rows, recovery requirements, supported execution shapes, state-validity maintenance, and stop/commit rules.

Do not hardcode four guard rows into this interface. The reported +4 rows belong to the current verification geometry. Larger MTP depths and DFlash blocks require their actual tail policy and page/pooling rounding.

Clamp proposals to remaining context, reply budget, supported positions, and stopping rules. Extra physical rows do not extend advertised context or authorize evaluation at unsupported positions.

## 8 Capacity derivations and qualification candidates

### Unique-history rates

For the supplied geometry, one head's K+V costs 544 B/token/layer in Q8 and 272 B in the audited compact four-bit layout. Raw plus pooled indexer history is 320 B/token/layer.

With two global heads stored once, balanced over eight ranks, and one copy of indexer history:

```text
13-cache Q8 rate = 13 × (544 / 4 + 320 / 8) = 2,288 B/token/GPU
13-cache Q4 rate = 13 × (272 / 4 + 320 / 8) = 1,404 B/token/GPU

12-cache Q8 rate = 12 × (544 / 4 + 320 / 8) = 2,112 B/token/GPU
12-cache Q4 rate = 12 × (272 / 4 + 320 / 8) = 1,296 B/token/GPU
```

The 12-cache rates describe a target-only configuration. They cannot budget an instantly available full stock-MTP configuration. DFlash uses 12 target caches plus its own actual draft resources.

### Full-context MTP planning arithmetic

Assumptions: 13 caches; uniquely stored histories with ideal balance; 13.421333 GiB/GPU of weights; 27.086 GiB/GPU serving budget; 2 GiB workspace; and the entire reported 27.798 MiB/session/GPU state subset treated as additive. Real padding and remaining obligations are not included in the residual.

| Candidate | History GiB/GPU | Weights plus history GiB/GPU | Residual after stated workspace and state |
|---|---:|---:|---:|
| Q8 × 16 | 8.937500 | 22.358833 | 2.292823 |
| Q8 × 18 | 10.054688 | 23.476020 | 1.121343 |
| Q4 × 24 | 8.226563 | 21.647895 | 2.786589 |
| Q4 × 28 | 9.597656 | 23.018989 | 1.306909 |

The residual must still cover snapshots, graphs, replacement overlap, migration, metadata/padding, and remaining tensors. The additive interpretation and rank balance require verification. Therefore 16 Q8 and 24 Q4 are first qualification candidates, not established operational maxima.

| Context/session | Q8 history/session/GPU with MTP | Q4 history/session/GPU with MTP |
|---:|---:|---:|
| 65,536 | 0.139648 GiB | 0.085693 GiB |
| 131,072 | 0.279297 GiB | 0.171387 GiB |
| 262,144 | 0.558594 GiB | 0.342773 GiB |

Per-session state and fixed execution/recovery costs prevent a simple rule that halving context doubles capacity.

For 16 and 24 sessions, whole 13-layer indexer-history counts divide evenly across eight ranks: 26 and 39 per rank. Other computed counts may not. Use actual ownership and rank-specific weights, not the average, when accepting them.

The previous 20-Q8/32-Q4 target-only calculations are not the governing capacity contract. The governing contract is automatic capacity with the administrator-selected speculative capability and useful execution resources included.

## 9 Canonical paged history and state ownership

Store each target KV head once across its four-rank head group and each raw/pooled indexer history once across the box. Initial indexer computation remains at one canonical owner per session/layer. The target model's TP8 projections and output layout remain intact.

### Page and session metadata

Each logical page identifies its session generation, attention layer, global head, logical position range, valid rows, codec/schema, immutable rotation identity, physical owner/offset, and ownership generation. Include all payload and metadata needed to decode it. Keep stable device tables and arena addresses for captured execution.

Session state distinguishes:

- Externally committed/emitted sequence and per-request sampling state.
- Target-consumed/cached position and recurrent/convolution state.
- Draft-valid position and method-specific conditioning state.
- Committed and tentative KV/indexer lengths.
- Incomplete indexer pooling groups.
- Selected context/reply limits and stop state.
- Page ownership generation and slot generation.

Slot reuse must not permit a stale graph, transfer, or completion callback to write into a new request. Use explicit lifetime/generation checks and retire readers before recycling storage.

Choose an aligned page size compatible with quantization blocks and four-token indexer pooling. A 128-token page is an initial candidate, not a proven optimum. Charge real payload and metadata strides; a small odd-sized quantization structure must not cause unaccounted padding on every row.

### Initial owner execution

For each target attention layer:

1. Execute the existing TP8 projections.
2. Route each query head to the owner of its corresponding global KV head history.
3. Append incoming KV once at its owner, in the committed or tentative range required by the execution method; promote only the accepted prefix.
4. Construct the exact indexer query inputs and update raw/pooled indexer data once at its canonical owner, with per-row causal visibility and committed/tentative state handled by the same transaction.
5. Perform the original selection over unique raw/pooled histories.
6. Dispatch selected logical positions to the KV owners.
7. Compute attention and return outputs to their original TP positions.
8. Continue the existing target output projection and distributed execution.

Batch routing and collectives by layer. Do not invoke one collective per request. Preserve each query row's causal history boundary, masking, pooling, ties, deterministic logical position interpretation, and the exact sparse-selection algorithm. A later row's tentative append must not become visible to an earlier query. Replicated histories in the old layout do not establish that all indexer query inputs were already replicated.

Small transient projection duplication may remain initially if it avoids unnecessary changes and does not duplicate persistent history. Optimize that work only when profiling identifies a useful gain.

### Prefix reuse

Shared prefixes are an optional optimization, never a prerequisite for the full-context capacity guarantee. Correct reuse in this hybrid model includes compatible recurrent state, convolution tails, indexer pooling/selection state, codec identity, and speculative conditioning at the reuse boundary. Matching target KV bytes alone is insufficient.

## 10 Attention degree and balanced placement

Support target attention degree 1, 2, and 4 within each global KV-head group. Degree describes how many ranks hold disjoint portions of a session/layer/head history. It does not change target TP8 weight distribution.

Degree 1 computes attention locally at the owner. Degrees 2 and 4 distribute the selected positions, route the necessary queries, and combine partial attention. Byte balance does not ensure compute balance when sparse selected positions concentrate on a subset of pages.

### Proposed target-layer mapping

For local rank indices 0 through 3:

```text
base = (slot_phase + layer_phase) mod 4
owner(page, degree) = base XOR (page_index mod degree)
degree is one of 1, 2, or 4
```

Cycle the layer phase uniformly over the twelve target attention caches. If their page geometry and token-row costs are identical, the same logical page across a quartet of layers occupies each local rank equally for every supported degree. This preserves the session's aggregate per-rank KV reservation when degree changes.

Verify that assumption against the manifest. Include incomplete pages and any layer-specific data. Choose physical rank numbering so the nested degree-2 pairs use appropriate measured links.

The MTP cache is outside this twelve-layer proof. Initial MTP KV placement may remain page-striped over all four ranks in each head group; this is a proposed starting layout to qualify. Its indexer is separately placed and charged. DFlash layers receive an independent geometry/placement plan.

### Exact partial attention merge

For the same query and disjoint selected positions, each participating rank computes local maximum m_i, exponential denominator l_i, and weighted value numerator z_i. Merge:

```text
m = max_i(m_i)
l = sum_i(exp(m_i - m) × l_i)
z = sum_i(exp(m_i - m) × z_i)
output = z / l
```

Do not average independently normalized outputs. Empty shards contribute an empty softmax state, with correct handling of the all-empty case according to the original attention semantics.

Keep the indexer canonical during KV placement changes. The selected positions are routed using the actual current owner table. Splitting the indexer into independent local top-k operations would change the algorithm unless a separate exact global procedure is implemented.

### Placement policy

Measure degree 1, 2, and 4 at relevant activity, context, selected-work, and prefill/speculation shapes. No particular degree is established as the winner at a fixed threshold. All eight GPUs continue target-model work even when degree-1 attention concentrates on a smaller set of owners.

A fixed four-way context-sharded placement is a legitimate measured candidate, with no load-driven page movement. Adaptive placement is chosen only when its benefit exceeds its communication and migration costs.

## 11 Bounded live cache migration

Migration changes physical ownership within the running engine and keeps compressed bytes unchanged. It does not require a second complete history, a precision conversion, or a model restart.

### Resource and scheduling rules

The proposed initial additional-occupancy cap is 64 MiB per GPU. Qualify or adjust it; it is not a measured optimum. The cap covers uncommitted destination copies, staging/bounce space, metadata, and source-retirement lag. Count the entire temporary excess, not only a named buffer.

Before a move, verify current occupancy, future full-window reservations, final placement, and every intermediate overlap. Low activity does not release inactive sessions' pages. If a move does not fit, keep executing from the valid current placement.

For target-layer degree changes, migrate balanced bundles containing the same token-page index across matching layer quartets. Source and destination assignments distribute those pages evenly. Bound transfers by available time and occupancy; do not move an entire layer first and rely on later moves to repair temporary imbalance.

### Commit protocol

1. Select a worthwhile target placement and reserve the next bounded destination bundle.
2. Resolve the affected session's speculative work to its defined committed boundary.
3. Drain readers/writers of the affected pages through explicit stream/event dependencies.
4. Copy complete packed payload and metadata. Source pages remain authoritative until completion.
5. Publish a new page-ownership generation after the required participants acknowledge copy completion.
6. Retire old copies only after all readers of the previous generation have finished.
7. Replenish the bounded free-page window and schedule further bundles when appropriate.

Pause only the affected work to the extent the implementation permits. Other requests can run between transfer chunks. Overlap on independent streams is an optimization to qualify; deterministic communication ordering is required even when overlap is disabled.

### Intermediate ownership is normal executable state

Committed page moves remain authoritative. If migration stops midway, execute from the mixed owner table using the qualified union of current holders. Discard only uncommitted copies. Do not restore an old session-level mapping after the original pages have been freed.

Make the union communication path available before expansion. Keep a larger participating group available until contraction has moved every page that requires it. Stable table addresses do not replace synchronization of their contents.

Seal or quiesce mutable tail pages before copying them. Preserve valid lengths, incomplete pooled groups, slot generations, sampling state, and target/draft validity. Physical relocation must not change a codec rotation seed or basis.

### Cost policy

A threshold crossing triggers evaluation, not automatic movement. Consider expected workload persistence, remaining generation, measured transfer bandwidth, interruption, interference with other requests, and an uncertainty margin.

Under the twelve-layer geometry, a full owner-to-degree-4 redistribution moves about three quarters of the unique target KV: approximately 1.20 GiB for Q4 or 2.39 GiB for Q8 across the box per full session, excluding indexer movement. This is arithmetic transfer volume, not a timing prediction.

On a demand spike, stop initiating unnecessary chunks and continue from the valid actual placement. Coalesce pending changes and use hysteresis to prevent repeated movement around a threshold.

## 12 Speculative execution and accepted-prefix correctness

### One method interface

None, stock MTP, and DFlash provide method-specific resource and state requirements to the same capacity planner and scheduler. This is a small execution contract, not a requirement for a generic plugin framework.

Each method must define proposal construction, target row count, target/draft tentative storage, valid position offsets, recovery, feature/state maintenance, and commit behavior. Its reported limits must agree with the actual kernels and buffers.

### Stock MTP

Keep the stock MTP weights, thirteenth cache, and required per-session state resident when MTP is enabled. At the current convention, target rows per scheduled request equal one plus speculative depth:

| Depth | Target rows/request | Requests fitting in R32 |
|---:|---:|---:|
| 0 | 1 | 32 |
| 1 | 2 | 16 |
| 2 | 3 | 10 |
| 3 | 4 | 8 |

R32 is a per-step execution limit, not an eight-session resident limit at depth 3. Sixteen depth-3 requests need 64 rows if verified together; 24 need 96. Fair microbatching remains possible, with its actual latency and throughput measured.

Qualify useful R32/R64/R96/R128 or other row envelopes as whole implementations. Extend index ranges, kernel shapes, workspace extents, recovery buffers, and graph shapes together. A larger graph alone does not extend an R32 kernel.

### Warm capability versus current proposal length

Proposal length zero is an execution choice. Immediate reactivation requires the method's weights, required history/state, and execution prerequisites to remain present and current. Include any state-maintenance work during zero-proposal steps in the timing.

The planner must not reclaim those persistent resources while promising instant reuse. Removing the method is a configuration transition with a different capacity result and possible future reconstruction needs.

### Accepted-prefix transaction

Verification must commit only the accepted and permitted prefix across every state type:

- Target KV and raw/pooled indexer history.
- Recurrent state and convolution tails.
- Partial indexer pooling groups.
- Draft KV and conditioning state.
- Logical/emitted/target-cached/draft-valid positions.

Sampling/RNG state follows the method's proven sampling algorithm, not a generic rollback to the accepted-token count. Rejected proposals, acceptance tests, and correction sampling can legitimately consume random draws. Define per-request proposal, acceptance, and target/correction sampling streams or draw identifiers as needed, and preserve their algorithm-defined advancement. Do not rewind randomness merely because candidate tokens were rejected.

The 36 recurrent layers require accepted-prefix state selection or a proven checkpoint/replay procedure. Cropping rejected KV does not undo recurrence. Budget and measure the actual strategy. Longer blocks may increase stored intermediates or recomputation substantially.

EOS, stop strings, cancellation, context exhaustion, and reply-budget exhaustion can shorten the committed prefix further. Evaluated tokens beyond that prefix must not become visible or permanent state.

### Sampling and numerical guarantees

Speculation must verify against the configured target computation and sampling policy. Correct greedy matching is not a substitute for stochastic acceptance/rejection. Apply temperature, truncation, penalties, constraints, and stopping rules at the correct conditional prefix. A DFlash variant with a candidate selector needs its own correct proposal-probability treatment.

Lossless speculation is relative to the selected target numerical configuration; it does not make Q4 equal to Q8. Changed floating-point reduction order can affect numerical results. Define tolerances and supported sampling guarantees explicitly rather than promising universal bit-identical text.

Keep per-request RNG ownership independent of batch permutation and migration. Padded rows must not advance another request's RNG or state.

## 13 RotorQuant and future codec work

### Format requirements

Pin the exact RotorQuant-family four-bit variant and source revision. The audited reference provides planar3, iso3, planar4, and iso4 paths; the ordinary Q8 representation remains separately specified.

The inspected four-bit block occupies 68 bytes per 128 values: 64 bytes of codes plus two FP16 metadata fields, including its norm/reserved-field convention. For dimension 256, K and V together occupy 272 bytes per head/token/layer before allocation padding. The existing Signed128 format has different metadata and numerical semantics despite the same nominal footprint.

The supplied live qualification of Signed128 must not be transferred to RotorQuant. The actual port must independently establish format roundtrip correctness, zero-vector behavior, scales/norms, rotation/centroid conventions, quality, peak memory, prefill, and decode performance.

Sources: [RotorQuant project](https://github.com/scrya-com/rotorquant) and [audited reference schema](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-common.h).

### Reference integration boundaries

The audited llama.cpp reference commit includes deferred FP16 K allocation plus a compressed destination and a CPU conversion path. TensorFold must instead use direct GPU packing and bounded progressive-prefill storage. A full FP16 history shadow or full-history conversion peak invalidates a compact-only capacity plan. Source: [pinned cache allocation and conversion code](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/src/llama-kv-cache.cpp).

The inspected dispatcher routes Planar/Iso attention to the vector path. That reference path does not establish optimized SM70 prefill or large-verification execution. Implement and compare appropriate small-row and tiled paths in the TensorFold engine. Source: [pinned attention dispatch](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-cuda/fattn.cu).

### Rotation and compute

Preserve the target model's positional transformation order. A general codec rotation must not be assumed to commute with RoPE. Where the pinned codec permits it, rotate the query consistently with stored keys, accumulate values in the appropriate basis, and apply an inverse transform once to the output. Include every norm/scale correction; do not infer equivalence from orthogonality alone after quantization.

Unpack into register/shared-memory tiles as appropriate and avoid expanding the entire history per step. Keep payload and metadata layout explicit and aligned. GPU migration copies the packed representation without another quantization pass.

Evaluate the actual immediate-quantized-history prefill path at long context. Quality measured with FP16 prefill and deferred conversion does not establish the same behavior.

### Deferred comparison

SAW-style token-wise INT4 with lightweight key rotation remains the next codec candidate after the main system is working. Keep the page/ownership and scheduling contract independent of the pack/unpack implementation, without creating multiple permanent serving frameworks.

The SAW reference is not a drop-in V100 implementation; its modern-GPU attention paths and full-history prefill expansion behavior need their own port review. Sources: [SAW paper](https://arxiv.org/abs/2604.19157) and [reference implementation](https://github.com/togethercomputer/saw-int4). KIVI, KVQuant, TurboQuant, and other approaches are not prerequisites for this design. Revisit them only if measured quality or performance provides a concrete reason.

## 14 DFlash integration

DFlash uses target hidden features to condition a block-parallel drafter. The features are projected into the draft layers' own KV; this is not free reuse of target attention KV. The draft can share some target tensors only where the implementation actually shares their storage. Sources: [DFlash paper](https://arxiv.org/html/2602.06036v2) and [joint Z Lab, Modal, and SGLang implementation](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/).

### Compatibility manifest

Require exact target/draft revisions, feature tap layers and normalization, hidden dimensions, tokenizer/vocabulary, mask token, embedding/head conventions, logit transforms, positional encoding, architecture variant, anchor convention, and supported block/context sizes. Do not infer a compatible draft from a related model name.

The official repository lists model-specific artifacts and distinguishes DFlash from DFlash 2. The exact pairing for this deployment is unresolved. Source: [official DFlash repository](https://github.com/z-lab/dflash).

### Attention-mask contract

The drafter needs its checkpoint-defined non-causal/bidirectional interaction within the draft block, while target verification must retain the target's causal conditional prefix. Implement the corresponding SM70 draft-attention mask; do not reuse the target's causal sparse/indexed path unchanged merely because tensor dimensions match. Target KV, indexer selection/pooling, and recurrent execution must prevent later verification positions from influencing earlier conditional logits. Source for the draft computation: [DFlash paper](https://arxiv.org/html/2602.06036v2).

### Complete draft budget

Charge:

- Draft weights according to actual placement, including any replicated copies.
- Actual target embedding/head sharing or extra copies.
- Each draft layer's cache heads, dimension, precision, metadata, page strides, and retention rule.
- Bounded target-feature extraction/fusion and draft-KV materialization buffers.
- Draft activations, logits, scratch, and communication.
- Target verification, temporary tails, recurrent recovery, and graphs.

Target TP8 does not imply that every draft cost is divided by eight. Select and qualify an appropriate draft placement, then keep it fixed at its maximum-concurrency configuration unless a separate safe online transition is implemented.

### Feature and context handling

Extract the required target features in bounded prefill/verification chunks, project committed features into draft KV, and release temporary feature storage. Do not archive every target hidden state across a 262k prompt. Rejected candidate features must not become permanent draft context.

A checkpoint-supported draft window can reduce draft history without shortening target context. Charge each layer's actual policy; a mixed full/sliding stack is not uniformly bounded. Storage outside the window must actually be reclaimed or reused. A mask alone proves no physical memory reduction. Source: [NVIDIA DFlash geometry and window configuration](https://docs.nvidia.com/nemo/automodel/latest/recipes-e2e-examples/dflash-speculative-decoding).

Long absolute positions and acceptance at 262k require qualification. A full-prefix dense drafter can perform more historical attention work than the indexed target; include that cost in comparisons.

### Verification convention

Standard DFlash block size B includes the anchor, normally proposing B−1 tokens and consuming B target verification rows. Other variants must report their own convention. Source: [Speculators DFlash anchor documentation](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/).

| Standard block size | Candidate tokens | Target rows/request | Requests fitting in R32 |
|---:|---:|---:|---:|
| 4 | 3 | 4 | 8 |
| 8 | 7 | 8 | 4 |
| 16 | 15 | 16 | 2 |

Twenty-four block-16 requests require 384 target rows if verified together. Resident capacity can exceed the next verification batch, but the resulting latency/throughput must be measured.

Choose among block sizes supported by the checkpoint and implementation. Do not assume that every integer proposal length is valid or profitable.

### State maintenance and activation

If DFlash is enabled while proposals are temporarily skipped, maintain the required conditioning through committed target progress and track draft-valid position. Include the maintenance time.

Missing historical target features generally cannot be reconstructed from current target KV. Replaying only the last draft window from an empty target is not generally equivalent, because those features can depend on prior attention and recurrent state. A stale or unloaded draft requires a separately supported reconstruction or a pending profile change.

### Evidence and performance boundary

The original paper's serving gains were measured on modern GPUs and its long-context adaptation results do not qualify this target at 262k. A firsthand two-V100 benchmark supplies short-context single-request evidence, with benefits concentrated at small proposal widths and regressions at larger widths for some workloads. Its targets, weight formats, context, and engine differ from this deployment. Sources: [paper](https://arxiv.org/html/2602.06036v2), [V100 benchmark](https://github.com/thc1006/qwen3.6-speculative-decoding-v100), and [methodology](https://github.com/thc1006/qwen3.6-speculative-decoding-v100/blob/main/METHODOLOGY.md).

This supports implementation and measurement, not a predetermined DFlash win or a numerical concurrency claim. Validate SM70 FP16/FP32 numerics for the chosen draft rather than assuming a BF16-oriented reference can be cast safely.

## 15 SM70 kernel and model execution work

Retain the model's qualified packed weight inventory. All routed experts in the supplied retained plan remain resident initially. Do not reintroduce host expert paging or expand the full checkpoint to FP16 as an unaccounted optimization.

Required execution work:

- Implement direct GPU KV packing and tile-based unpacking with the exact codec metadata.
- Select vector/small-M versus tiled attention and projection paths from measured complete-step behavior.
- Group routed rows by expert and use actual group sizes to select GEMV, small-M, or grouped-GEMM execution. Global batch size alone is insufficient.
- Reuse decoded weight tiles and activation buffers where proven useful, with explicit lifetimes.
- Preserve required FP32 recurrent state and sensitive accumulation unless separately qualified.
- Compute logits for positions required by generation/verification; additional prompt log-probabilities are an explicit workload with their own budget.
- Reduce launch, routing, and host/device synchronization overhead when profiling identifies it.
- Audit warp masks, barriers, register/shared-memory pressure, occupancy, spills, memory coalescing, and actual bandwidth.

Volta has independent thread scheduling and explicit synchronization requirements; respect them in custom kernels. Native SM70 compilation and supported math/library paths are required for every kernel family. Source: [Volta tuning guide](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html).

Do not select a path solely because it uses more Tensor Cores or has a higher isolated kernel rate. Selection must include its packing, communication, state, and recovery costs.

## 16 Prefill and mixed-work scheduling

Use chunked prefill with bounded workspaces and continuous admission into scheduled work. Preserve causal order and the 36 recurrent layers' state transitions within a request. Independent requests can contribute rows to the same execution batch. Parallelizing a recurrent scan within one request requires an algorithm that preserves that layer's semantics; ordinary token batching does not establish this.

Make the prefill workspace proportional to the scheduled token budget and selected kernel shapes, not the number of configured sessions multiplied by every session's maximum chunk. Include routing tiles, attention/indexer scratch, recurrent intermediates, codec staging, target-feature extraction for the selected speculative method, and any required logits. A provisional chunk search might include 512-token chunks and total scheduled budgets from 1,024 through 8,192 tokens; these are measurement candidates, not settled limits or existing flags.

Qualify three distinct regimes:

| Regime | Scheduling aim | Measurements that decide the plan |
|---|---|---|
| Prefill only | Maximize completed prompt tokens/s within memory and fairness limits | Chunk size, total scheduled tokens, expert group sizes, history length, time to first token |
| Decode only | Maximize committed generated tokens/s with acceptable inter-token delay | Active requests, proposal acceptance, verification rows, indexed attention work, state/recovery cost |
| Mixed | Preserve useful decode progress while processing queued prompts efficiently | Prefill quota, decode service interval, prompt starvation, tail latency, combined resource pressure |

The scheduler must not force a low-load request to wait for a larger batch merely because a larger batch has a better throughput profile. At high load, use bounded prefill service and fair decode rotation so long prompts cannot indefinitely stall generation, and generation cannot indefinitely starve new prompts. Admin-selected latency objectives can restrict the eligible throughput profiles.

Keep temporary work bounded at long contexts: no full-context FP16 KV mirror, full-history dequantization buffer, or permanent target-hidden-state archive. An implementation that avoids a persistent mirror but materializes a huge temporary mirror still violates the memory goal. Compute only requested prompt log-probabilities and required generation/verification logits.

Include tokenization, request assembly, output delivery, pinned transfer buffers, and NUMA placement in end-to-end profiles. Overlap CPU work and independent GPU work where dependencies permit. Each overlap must be represented in the resource ledger; a buffer cannot be counted as reusable if another stream still consumes it.

## 17 Global adaptive controller

Use a small deterministic controller backed by qualified profiles. Begin with an offline benchmark table and lightweight runtime observations; a complex online learning system is not required.

### Two control timescales

| Timescale | Decisions | Safe application point |
|---|---|---|
| Every scheduled step | Eligible requests, proposal length, target verification rows, prefill token quota, kernel family, existing graph bucket | A globally agreed batch boundary |
| Sustained workload change | Attention owner degree, page placement, bounded graph replacement, resource-envelope changes | Qualified migration or quiescence protocol, with admission constraints respected |

The coordinator issues one execution plan and epoch for all TP ranks. Ranks do not independently change proposal lengths, graph shapes, communicators, or collective participation based on local measurements.

The controller observes actual active requests, history lengths, selected indexer work, routed expert group sizes, prefill/decode mix, acceptance, pending arrivals, and memory pressure. Active count selects a profile family; it is not a sufficient prediction of work by itself. A single request with a nearly full retained history can have different costs from a short request in the same count bucket.

### Required concurrency coverage

| Actual active requests | Required behavior |
|---|---|
| 0 | Retain promised session state; maintain a ready execution profile; perform permitted quiescent maintenance |
| 1, 2, 4, 8 | Measure low-load latency and throughput; compare ownership degrees and useful speculation widths |
| 12, 16, 20, 24, 28, 32 | Measure saturation, verification splitting, prefill interference, fairness, and speculation profitability wherever capacity permits |
| Other counts | Select a qualified covering profile with correct masking and memory bounds; include the exact computed capacity as a benchmark endpoint |
| Above 32 | Extend the same policy when shorter contexts or other qualified settings permit more sessions |

Clip the test set to the computed resident capacity, but do not round that capacity down to a listed threshold. If capacity is 18, test 18. If it is 40, test the higher regime as well.

Maintain a safe fallback for an immediate jump from one active request to the configured maximum. This can be a bounded sequence of smaller execution batches. Hysteresis must never leave an undersized buffer or unsupported graph serving an oversized batch.

Use hysteresis and a minimum expected benefit for costly placement changes. Evaluate the cost over a realistic expected residence time in the new regime. When evidence is insufficient, retain the current placement and use a safe batch plan. Do not repeatedly migrate large histories during oscillations around a threshold.

Keep statistics asynchronous where possible. Avoid a device-wide synchronization on every step just to feed the controller. Profile selection should use complete-step elapsed time, committed token count, and observed latency, including draft maintenance, communication, packing, recovery, and migration amortization.

## 18 CUDA graphs and communication

Initialize the supported communication groups, kernel families, descriptor bounds, and maximum required buffer capacities before they are needed. Keep weight TP8 fixed. Prebuild a bounded set of useful execution graphs and retain uncaptured execution of the same canonical plan, kernels, and allocation contracts. Graph replay and uncaptured execution are modes of one serving implementation, not separate backends.

Do not capture the Cartesian product of every session count, context length, proposal width, prefill size, codec, and placement. A small set of row/token buckets with masks and stable indirection is preferable when measurements justify it. Padding is not free; charge its actual compute and storage.

### Graph validity

Stable page-table pointers help, but they do not by themselves make a graph reusable across a layout change. Audit captured direct pointers, scalar arguments, launch dimensions, collective counts, participating ranks, stream dependencies, allocator pools, and object lifetimes. A graph is reusable only when all relevant captured assumptions remain valid.

Inactive or padded rows must be masked from KV writes, indexer updates, recurrent and convolution state, draft state, random-number consumption, stop processing, and committed output. A graph must also validate the session generation so reused slots cannot receive stale writes.

Count graph-private pools, persistent graph inputs, and live graph instances in the ledger. Share graph scratch only where replay ordering proves that lifetimes cannot overlap. Source: [PyTorch CUDA graph memory and lifetime documentation](https://docs.pytorch.org/docs/2.14/notes/cuda.html). The linked documentation explains the mechanisms; its package version is not a recommendation for the SM70 runtime.

### Collective ordering

All ranks must agree on captured versus uncaptured execution and issue collectives in a compatible order. The migration protocol must coordinate its communication with serving work rather than inserting rank-local copies into an otherwise captured collective schedule. Use explicit streams/events and a globally ordered plan. Source: [NCCL CUDA graph requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html).

Capture or replace graphs only under the framework's supported quiescence rules. Running capture on another host thread does not establish that concurrent unrelated CUDA work is safe. Use the ordinary uncaptured plan before or after recapture where needed; pause the affected device work during capture to the extent the framework requires.

## 19 Configuration changes without restart

Expose configured, pending, and effective values separately. Normal adaptation and supported administrative changes must not restart the container, worker process, or CUDA context. Do not disguise a worker replacement as a live configuration update.

| Adjustment | Implementation policy |
|---|---|
| Scheduled batch, prefill quota, supported proposal length, existing graph/kernel choice | Apply at a coordinated batch boundary within allocated limits |
| KV ownership and attention degree | Apply through bounded byte-preserving migration and a published mapping epoch |
| Resident-session ceiling decrease | Stop new admissions as needed; preserve existing sessions and their context commitments |
| Resident-session ceiling increase | Apply only after the complete maximum-context and execution-envelope proof succeeds |
| Larger context setting | Activate for all retained sessions together only after reserving every enlarged commitment and validating position/state compatibility; otherwise keep pending until the retained profile drains |
| Smaller context setting | Preserve existing promised windows and keep the change pending until the homogeneous retained profile drains |
| Structural codec, target weights, draft checkpoint, or speculative-method change | Use a qualified in-process transition; otherwise keep pending until retained sessions drain, then reinitialize the profile in process |
| A setting with no supported in-process transition | Fix it at the qualified maximum-concurrency configuration; do not present it as a live control |

Initially use one homogeneous retained-session profile. This makes equal context guarantees, recovery, capacity accounting, and graph ownership tractable. Supporting several simultaneous quality/context profiles is a separate extension with a combined budget, not an implicit consequence of paging.

Zero active requests does not imply zero retained sessions. Histories, draft conditioning, recurrent state, and recovery obligations can still prevent a structural change. Pending changes must state the blocking resource or retained-session condition without silently deleting sessions.

Moving already quantized page bytes between GPUs preserves that representation. Converting Q4 history to Q8 does not recover the information discarded by Q4. Likewise, enabling a new speculative method mid-session can require historical features or state that were never retained. Do not label either operation a lossless migration. Support a properly qualified reconstruction if useful; otherwise apply the profile after the affected sessions drain.

If old and new layouts or weights cannot coexist within bounded transition memory, use the empty-resident in-process transition. Keep requests queued or report the pending state according to the service policy; do not allocate an unbudgeted second model or claim an interruption-free transition that has not been implemented.

Within an enabled method, a zero-proposal step is an ordinary scheduling choice if its conditioning remains valid. Switching MTP to DFlash, changing draft checkpoints, or changing KV precision is a structural profile change and is not driven automatically by the active-count controller.

## 20 Host memory and deferred architecture alternatives

The existing PLE/ngram table remains a reported 26.822 GiB read-only file-backed host mapping. Rank mappings can share physical file-backed pages; eight mappings do not establish eight physical copies. Residency is evictable, and actual NUMA placement remains an integration measurement.

The new raw-indexer RAM archive is off in the supplied current configuration. If explicitly enabled later, it removes the growing raw indexer history from VRAM while attention KV and pooled indexer history remain on GPUs. The old implementation's approximately 13.038 MiB fixed staging/ring cost per session per GPU must be recomputed for the unique-history design rather than copied unchanged.

Qualify host archive page residency, NUMA binding, pinned-buffer caps, transfer concurrency, PCIe uplink contention, and serving tail latency. Do not assume nominal host RAM bandwidth is the available GPU transfer rate. Preserve the same context and exact selection semantics.

Paging attention KV through host RAM is a separate unimplemented architecture in the supplied receipt. It cannot be counted as present capacity or activated silently to rescue an overcommitted profile. The same applies to offloading retained expert weights.

Alternative weight topologies, such as independent smaller TP groups with expert parallelism, or pipeline layouts, require a new per-rank weight plan, routing/communication design, memory proof, and performance qualification. They are not prerequisites for the fixed-TP8 design and are not live active-count knobs in this plan.

## 21 Observability and qualification

### Runtime receipt

Publish a compact machine-readable effective-profile and capacity receipt, plus user-facing explanations of limiting resources. Include:

- Exact target/draft/codec identifiers, hardware/runtime manifest, selected context, configured ceiling, computed resident capacity, and qualification status.
- Retained, active, queued, and scheduled request counts; prefill tokens and target verification rows in the current execution plan.
- Current proposal policy, draft-valid state, acceptance distribution, recovery/replay work, and committed generated tokens.
- Current per-rank weights, history pages, state, graph pools, workspaces, external allocations, physical free memory, and protected commitments, with timestamps.
- Placement degree, owner-map epoch, migration bytes outstanding, transient occupancy, migration time, and admission restrictions.
- Useful prompt-processing tokens/s, committed decode tokens/s, time to first token, inter-token latency, per-request service/fairness, and error/cancellation counts.

Report startup and current runtime readings separately. Distinguish measured allocations from logical reservations and planned maxima. Do not add allocated bytes to reserved bytes, or report tentative/draft tokens as generated throughput.

### Qualification matrix

| Axis | Required cases |
|---|---|
| Precision and method | Each supported target KV format with none/MTP; DFlash only for an identified compatible draft and supported draft formats |
| Context | Short, intermediate, and near-262,144 windows; requested output must fit inside the selected window |
| Concurrency | All feasible requested thresholds, exact capacity endpoints, burst from 1 to maximum, and counts between buckets |
| Retention | One resident/one active; all resident/one active; all resident/all active; partially filled and near-full histories |
| Workload mix | Prefill only, decode only, mixed long prompts and decode, heterogeneous acceptance, and routed expert skew |
| Transitions | Every supported ownership change in both directions, partial migration, threshold oscillation, and changes while arrivals continue |
| Lifecycle | Completion, EOS, stop strings, context exhaustion, cancellation, slot reuse, admission-limit changes, and pending profile changes |
| Speculation | All-accept, early/late rejection, zero accepted candidates, zero-proposal warm operation, final bonus/anchor handling, and state recovery |
| Long-lived service | Repeated fill/drain cycles, graph creation/replacement, allocator fragmentation, bounded migration, and sustained load |

A valid example of a near-full window is a 260,096-token prompt plus up to 2,048 generated tokens, totaling 262,144. A 262,144-token prompt plus an additional reply exceeds that window.

### Correctness and quality gates

Compare unsharded versus sharded attention using the same configured codec and target semantics. Cover empty shards, extreme attention scores, pooled-history boundaries, partial pages, ties in indexer selection, long absolute positions, masks, and codec metadata. Quantization quality against the chosen higher-precision reference is a separate comparison.

Test speculative accepted-prefix recovery across attention KV, raw and pooled indexer histories, recurrent state, convolution tails, draft state, positions, and random-number streams. Include cancellation and slot-generation races. Compare greedy outputs where exactness is expected and validate the stochastic acceptance algorithm and resulting distribution with appropriate tests; individual sampled strings need not be identical across execution schedules.

For RotorQuant, qualify progressive long-prompt prefill and long decode, not just quantization error on isolated vectors. Exercise the real SM70 kernels. For DFlash, verify the exact target/draft pairing, supported context/window semantics, within-block draft masks, and conditioning after rejected or skipped proposals. Include a target-verification test that detects future-token leakage into earlier logits or indexer decisions.

Use focused synchronization/memory diagnostics for custom-kernel risks and lifecycle races. Do not add a large test framework that merely repeats implementation details. Each gate should address a specific correctness, resource, or performance risk.

### Performance gates

Use repeatable prompts, generation settings, target/draft revisions, warmup, stable clocks/thermal conditions, and timing boundaries. Repeat measurements enough to distinguish real improvements from noise. Compare adaptive execution against the best qualified static profile under the same context, quality, retention, and workload conditions.

Report prefill rate and decode rate separately. Report whole-step and end-to-end rates alongside isolated-kernel measurements. Account for quantization, draft maintenance, rejected work, collectives, recovery, scheduling, and amortized migration. Record latency distributions and fairness so a throughput gain cannot hide starvation.

The aim is the best measured eligible profile at each load regime. No design document can establish a globally optimal tokens/s figure before implementing and measuring the actual target on this machine.

## 22 Implementation sequence and exit gates

The sequence below resolves dependencies while delivering useful intermediate configurations. Work can proceed in parallel where the state and resource contracts are already agreed.

| Stage | Required work | Exit evidence |
|---:|---|---|
| 1 | Pin target/fork, hardware, runtime, codecs, and existing recovery behavior; reproduce the supplied accounting | Exact per-rank inventory and reproducible current-profile receipt |
| 2 | Introduce canonical page/session metadata, full-context commitments, and accepted-prefix state APIs using the existing Q8 path | Current semantics preserved; lifecycle and state recovery tests pass |
| 3 | Remove persistent KV and indexer replication with owner execution; retain fixed weight TP8 | Exact history bytes and rank balance measured; output/selection equivalence established |
| 4 | Add distributed attention degrees, exact merge, topology-aware mapping, and separate MTP-cache placement | Correct degree 1/2/4 operation; placement and communication profiles recorded |
| 5 | Port the selected RotorQuant Q4 format and direct SM70 pack/unpack/attention paths | Byte layout verified; quality and long-context behavior qualified; no hidden full-history mirror |
| 6 | Complete MTP-enabled capacity planning and useful verification/prefill envelopes, including larger or split verification batches | Computed capacity receipts for selected contexts; 16 Q8/24 Q4 candidates evaluated with MTP resources included |
| 7 | Add bounded CUDA graphs, the global batch controller, and mixed prefill/decode scheduling | Safe immediate load changes; measured profiles at every feasible concurrency threshold |
| 8 | Add bounded cache migration, epoch publication, and hysteresis | All intermediate states fit; serving remains correct; migration gains exceed measured costs in adopted cases |
| 9 | Integrate a compatible DFlash checkpoint, bounded target-feature/draft-KV handling, and its complete resource description | Correctness, capacity, and performance independently qualified; method-specific effective capacity exposed |
| 10 | Complete in-process administrative lifecycle, observability, sustained-load tests, and operational documentation | No normal tuning restart; retained commitments preserved; complete qualification receipt and reproducible benchmark suite |
| Later | Evaluate SAW-style quantization and additional architectural alternatives | Separate quality/capacity/performance evidence before inclusion in eligible profiles |

Do not let DFlash availability block the stock-MTP architecture. Conversely, do not declare the dynamic planner complete while it assumes MTP's cache/state geometry for every future method.

For every stage, retain the exact commit/build identity, workload, resource receipt, results, and known limits. Promote a profile from estimate to qualified only after its stated gates pass. Qualification at one context or concurrency does not silently qualify every larger one.

## 23 Integration inputs and unresolved measurements

These are concrete implementation inputs and measurement gates, not reasons to revisit settled choices or request permission again.

1. **Exact target and fork:** repository/revision, checkpoint and tokenizer identity, layer geometry, RoPE/indexer semantics, retained weight encodings, and the three source accounting artifacts.
2. **Actual machine inventory:** CPU models, RAM capacity/layout, GPU UUID/rank/topology mapping, PCIe paths, NUMA placement, firmware, and usable SM70 software stack.
3. **RotorQuant selection:** exact variant, block format, scaling/rotation rules, GPU implementation, and quality reference. A format name alone is insufficient.
4. **Speculative contracts:** stock-MTP proposal/anchor rules and recovery implementation; exact compatible DFlash checkpoint, conditioning, draft geometry, retention policy, and supported block sizes.
5. **Recovery commitments:** required snapshot counts, capture cadence, replacement overlap, failure behavior, and which buffers genuinely share lifetimes.
6. **Measured execution profiles:** useful row/prefill budgets, graph buckets, kernel breakpoints, ownership degree, draft width, migration budget, and hysteresis for each eligible regime.
7. **Service objective:** throughput-first defaults with bounded fairness, plus any administrator-specified time-to-first-token or inter-token-latency constraints.

The final Q8 and Q4 capacities, and the final DFlash capacities, are outputs of these inputs and tests. The initial 16/24 MTP planning candidates must remain labeled as candidates until the complete ledger and workload qualification support them.

## 24 Acceptance checklist

The implementation is complete when all applicable items below are satisfied:

- Administrator inputs produce an explicit effective context, quality/method profile, computed capacity, and limiting-resource explanation.
- Every admitted retained session can reach its promised context, including generated output, with the selected speculative capability and recovery contract available.
- Target weight TP8 remains stable; persistent attention and indexer histories use the intended unique storage and a verified balanced placement.
- RotorQuant Q4 and ordinary Q8 have explicit, tested byte layouts and SM70 kernels; prior Signed128 qualification is not reused as RotorQuant evidence.
- The resource ledger covers worst-rank occupancy and real concurrent lifetimes, including MTP/DFlash, graphs, tails, snapshots, migration, and non-Torch allocations.
- Useful prefill and verification work remains possible at computed capacity; a cache-only packing limit is never presented as serving capacity.
- Low-load execution uses appropriate batches, kernels, graphs, speculation, and placement without waiting unnecessarily for more requests.
- Every feasible concurrency step and exact endpoint has a measured eligible plan; shorter-context capacity can exceed 32 without a hard-coded ceiling.
- Sudden load increases, partial migrations, cancellations, and slot reuse are correct and bounded.
- Accepted-prefix commit/recovery covers all hybrid model and draft state; masked rows cannot mutate live sessions.
- No supported runtime adjustment requires a container, worker, or CUDA-context restart. Unsupported live knobs are fixed appropriately or remain pending until a supported in-process transition.
- Structural changes preserve retained-session commitments and never silently reduce quality, context, or availability of the selected method.
- Runtime receipts distinguish current allocations, future commitments, startup observations, estimates, and qualified limits.
- Performance is demonstrated using useful prompt and committed decode tokens, with latency, fairness, and all relevant overhead visible.
- DFlash remains independently qualified for the actual target and V100 stack; SAW-style evaluation follows the core implementation.

## 25 Evidence and source register

### User-supplied deployment evidence

The baseline values and historical qualification descriptions in this document come from the supplied TensorFold accounting receipt dated 2026-10-25 04:16 UTC. Its referenced files are on the user's TensorFold machine and were not directly inspected while preparing this document:

```text
/home/kklouzal/TensorFold/task-artifacts/v100-indexer-archive/production-gpu-history-smoke/stats-initial/result.json
/home/kklouzal/TensorFold/task-artifacts/v100-indexer-archive/report-facts/weights.json
/home/kklouzal/TensorFold/task-artifacts/v100-indexer-archive/report-facts/frontier-all-layouts.json
```

The unique-history capacity arithmetic is derived from that receipt's geometry. Proposed algorithms, policies, placement rules, and gates are implementation design, not observations from an already modified TensorFold deployment.

### Primary external references

| Reference | Use and boundary |
|---|---|
| [Supermicro system specification](https://www.supermicro.com/en/products/system/4U/4029/SYS-4029GP-TVRT.php) and [system manual](https://www.supermicro.com/manuals/superserver/4U/MNL-2115.pdf) | Platform components and physical topology; actual installed CPU/RAM still require inventory |
| [Supermicro/Red Hat reference architecture](https://www.redhat.com/rhdc/managed-files/pa-supermicro-deep-learning-openshift-reference-architecture-f19703-201910-en_0.pdf) | Primary supplementary evidence for the platform's GPU interconnect arrangement |
| [NVIDIA Volta tuning guide](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html) and [architecture support guidance](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/) | SM70 execution and toolchain constraints; do not transfer modern-GPU kernel claims |
| [NCCL graphs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html) and [PyTorch CUDA notes](https://docs.pytorch.org/docs/2.14/notes/cuda.html) | Collective capture, graph lifetime, and memory rules; validate against the deployed versions |
| [RotorQuant repository](https://github.com/scrya-com/rotorquant) and [audited format source](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-common.h) | Codec/reference implementation evidence; a custom TensorFold port still needs format and quality qualification |
| [DFlash paper](https://arxiv.org/html/2602.06036v2) and [official repository](https://github.com/z-lab/dflash) | Method and checkpoint compatibility; modern-GPU results do not establish this deployment's throughput |
| [DFlash anchor semantics](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/) and [NVIDIA draft geometry documentation](https://docs.nvidia.com/nemo/automodel/latest/recipes-e2e-examples/dflash-speculative-decoding) | Method-specific row, cache, and window accounting |
| [DFlash / Spec V2 serving implementation report](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/) | Bounded feature-to-draft-KV materialization ideas; not a qualified implementation for this target |
| [Firsthand V100 DFlash benchmark](https://github.com/thc1006/qwen3.6-speculative-decoding-v100) and [methodology](https://github.com/thc1006/qwen3.6-speculative-decoding-v100/blob/main/METHODOLOGY.md) | Limited feasibility/performance evidence on a different target and workload; not a capacity forecast |
| [vLLM context parallelism](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/), [dynamic speculation](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/), and [NanoCP paper](https://arxiv.org/abs/2605.21100) | Related implementation concepts; not drop-in proof for TensorFold, stock MTP, or this V100 topology |
| [SAW paper](https://arxiv.org/abs/2604.19157) and [reference code](https://github.com/togethercomputer/saw-int4) | Deferred codec comparison after the core system works |

External software, draft artifacts, and reference implementations must be pinned when implementation begins. Every published capacity or performance claim must identify whether it is a supplied observation, an arithmetic estimate, a header plan, or a live-qualified result.
