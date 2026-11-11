# ProviderOne Experimental Optimization Candidates

**Research basis:** 2026-10-29.  
**Governing baseline:** current ProviderOne V100 Adaptive Inference Design, including Section 1’s duplication policy and Section 20’s committed parallelism profiles; the research consolidation originally used revision 21.  
**Integrated inputs:** TensorFold Strata/1Cat Comparison, revision 1; TensorFold External Research Candidates, revision 2.

This is the unified experimental and qualification backlog: **41 repository-audit families (A01–A41) and 24 external research IDs (E01–E24; E10 withdrawn)**. It brings their implementation details, measurements, failed controls, numerical/resource boundaries, source pins and references into one subsystem-oriented document. The existing IDs remain stable for implementation traceability.

Retain ProviderOne's capacity-first, unique-history architecture. Validate complete finite execution and the selected Inco draft's range before accepting performance evidence. Prioritize single-bank storage and bounded scratch, indexed MoE, retention-aware prefill, mixed/tail graphs and exact QSA work. E21 deferred GDN materialization is a high-value feasibility experiment once those prerequisites hold; the other research entries remain conditional on their stated bottlenecks and proof gates.

Published algorithms, inspected implementations, open/unmerged experiments and ProviderOne-specific synthesis are distinguished throughout. A synthesis is not a claim of worldwide novelty. Reported source-project measurements were not reproduced here; this consolidation performs no V100 execution or new model benchmark and does not amend the governing design.

All **Design Section** references below retain the section numbering used in revision 21; current governing requirements, including Section 1’s duplication policy and Section 20’s profile catalog, take precedence. Local navigation uses the linked sections in this document. “Current,” “latest,” and source status quoted from retained reports mean the inspected snapshots, not an unpinned future release.

ProviderOne is being detached from upstream TensorFold as an independent inference engine. The governing design’s Section 1 now authorizes direct removal/replacement of inherited internals and phases delivery: SC117/V100 first, mainstream model/container/quantization support next, Blackwell later. These V100/SC117 candidates and receipts do not establish another family’s geometry, speculative pairing or hardware performance. Historical source/report names remain provenance.

“Existing owner,” “native path” and similar wording below identifies a current integration point, not a requirement to preserve inherited classes, PyTorch/Python/MLX, scheduling or cache structures. Implement the smallest directly owned solution that satisfies the governing behavior and measured resource/performance gates; replace current components and remove superseded code where appropriate. A common ProviderOne service may have concrete model/hardware-specialized execution paths without retaining parallel legacy runtimes. Small reproducible complete-serving gains remain worth evaluating; isolated gains and unbounded framework growth are not acceptance evidence.

## 1 Operating contract and evidence

### Mission, admission and numerical boundaries

The control remains one distributed target on one or two eight-GPU SM70 V100 SXM2 servers under the fixed matrix, native 262,144-token windows and separately qualified extensions, original selected mixed weights, unique canonical GPU-resident active histories, required shared-Q8 MTP and separately selected Inco DFlash2. Maximize qualified resident capacity, then aggregate committed decode at full activity. Active expert/KV host paging remains outside scope. Duplication experiments are permitted, but acceptance requires verified complete-serving gains weighed against actual added VRAM; ≥1 GB (1,000,000,000 bytes ≈0.931323 GiB) net added across the whole selected deployment receives exceptional scrutiny. Apply the threshold to individual candidates and the cumulative package, including induced resources and steady/peak increments; below-threshold copies are not automatically accepted. Follow Design Section 1 for byte accounting, lower-copy comparisons, capacity/quality/nonregression gates and the recorded decision.

Every candidate must pass the existing C1/2/4/8/16/24 matrix where capacity permits, recording infeasible points, and the decode nonregression gates, including near-full histories and one active request with other promised histories retained. If qualified capacity exceeds 24, also test its full count and relevant integer transitions; the milestone matrix is not an engine ceiling. Preserve FIFO admission, resident progress, media, prefix identity, accepted-prefix transactions and no-restart transitions. The supplied >250 single-request and >600 aggregate-at-24 decode targets remain unchanged. Bind the >3500 prefill reference to its actual workload; no missing condition is invented here.

The required supported matrix is Design Section 20’s **TP8×PP1/PP2 with ETP8/4/2**: three cases on one eight-GPU server and three on two physical eight-GPU servers, one TP8 stage/server. ETP1 and its E10 experiment are excluded from development and qualification. EP=8/ETP stays node-local; DPA, other nonexpert TP/PP degrees, full/expert-bank replicas and independent method parallelism are outside the implementation backlog. The selected ETP also applies to MoE MTP in its one TP8 method group; dense DFlash has no ETP axis. PP2 remains required work pending two-server/RDMA qualification; a fully qualified three-profile PP1 release need not wait for that hardware. TP4×PP2 alone remains an optional, low-priority future implementation/benchmark choice. Its conventional four-rank stages allow ETP4/2 only; ETP8 is excluded from that study. No TP4 or ETP1 work is required now; other-layout evidence does not expand the backlog.

Three distinctions govern the entries:

- **Same stored bits** does not guarantee the same floating-point computation. Reordered dot products, different GEMM shapes, moved casts and changed reductions need the existing numerical/teacher, task, sampling and acceptance qualification.
- **Algebraically equivalent** does not establish bitwise equivalence or finite-range safety. Proofs about real arithmetic need a reference-arithmetic boundary.
- **More average compression or throughput** does not prove more guaranteed full-window sessions. Charge worst-rank allocations, metadata, scratch, graph pools, lifetime overlap and worst-case future growth.

“Proceed” below means a focused evidence-producing experiment in ProviderOne, including direct replacement where justified. It does not require a second legacy runtime, an online autotuning framework or an upstream compatibility layer.

**Target contract.** The platform is eight 32-GiB SM70 V100 SXM2 GPUs with the reference NVLink cube topology requiring deployment measurement, not NVSwitch. Keep target/MTP Q8 or iso4 KV and the canonical **FP16 pooled indexer keys** distinct. Host-backed PLE lookup and protected completed/idle-session RAM or storage retention are intentional boundaries; executing expert/attention state remains GPU-resident.

ProviderOne-derived cost examples default to **TP8×PP1** unless a deployment is named. Recompute PP2 costs from hosted target layers and final-stage companion ownership; do not halve every resource.

| Metric or workload | Governing condition |
|---|---|
| Single-request decode | User-verified greater than 250 committed output tokens/s at 262K context and Q8 KV |
| Full C24 decode | User-verified greater than 600 aggregate committed output tokens/s, with 24 active requests at the same context/quantization |
| Prefill | Greater than 3500 tokens/s reference; bind its actual workload/concurrency/timing conditions rather than inventing them |
| Milestones | A/B C1/2/4/8/16/24 for prefill and decode, with decode nonregression in every cell and priority 1 < 2 < 4 < 8 < 16 < 24; no invented numeric weights |
| Capacity | Maximize qualified integer resident sessions first, then full-activity aggregate committed decode; extend testing through the actual maximum and integer transitions above 24 |
| Serving envelope | Near-full independent histories, one active request with other promised histories retained, misses and warm prefixes, mixed prefill/decode, media, cancellation, tier restoration and sustained progress |

At C24, a four-row verifier contains 96 target rows; MTP4's five-row verifier contains 120. Original, routed, padded and per-request rows are different quantities. Extend kernels, descriptors, scratch, state/recovery and captures together, or use qualified fair microbatches; another project's M5/M20/C4 limits cannot become ProviderOne's service ceiling.

### What the inspected projects optimize

| Project | Relevant implementation | Material difference from ProviderOne |
|---|---|---|
| Strata-V100 | C++ runtime; native mixed-GGUF readers; Volta prompt attention and grouped expert kernels; CPU/RAM/storage expert tiers; pipeline execution; host/disk conversation cache | Most published V100 work uses one or two 16-GiB PCIe cards, including an x4 host link. Expert streaming, CPU sharing and residency-aware decisions address constraints different from eight 32-GiB SXM2 GPUs with all selected experts resident. |
| 1Cat-vLLM | vLLM fork with bundled Flash-V100, FlashQLA, TurboMind and native SM70 quantized kernels; Flash-Next adapters; GGUF preparation; graphs and custom communication | Many Flash-Next experiments use TP4, FP16 or E4M3 KV, FP16-derived MTP and C1/C4. Other headline results use the separate dense 27B model and DFlash2. Some deployment profiles execute attention against host-backed authoritative history. |
| ProviderOne | One distributed target, TP8 baseline, unique Q8/iso4 attention history and FP16 pooled indexer history, bounded raw state, required shared-Q8 QSA MTP and separately qualified Inco draft, full-window capacity guarantees | Stronger requirements on independent-session capacity, active GPU residency, measured/tiered scrutiny of optional throughput copies, byte-preserving idle suspension, and no-restart adaptation. Kernel implementations remain the main comparison opportunity. |

Sources: [Strata README](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/README.md), [Strata V100 execution guide](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/docs/NVIDIA_V100.md), [1Cat README](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/README.md), [SC117 local deployment](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/flashnext_abliterated_local_20261014.md).

The SC117 deployment is particularly relevant but not a matched substitute: it records a different SC117 revision, four V100-SXM2-16GB GPUs, TP4, 80% GPU/20% host expert placement, E4M3 host authoritative target KV with a bounded GPU hot cache, disk PLE and one active sequence. Its approximately 5088.8 input tokens/s is input count divided by median first-choice latency on roughly 34.6K inputs; its 154.9 output tokens/s is measured at 32,774 input tokens. Neither is a 262K-full-history, 24-active Q8 measurement.

## 2 Unified candidate register

A identifies a repository-audit family, including already-covered obligations and concrete validity findings; E identifies additional research. The labels record provenance, not separate implementation systems.

**Evaluate** means compare under ProviderOne’s required behavioral/resource/performance gates; existing component ownership may be redesigned. **Covered** means the design already requires the principle, with the source supplying concrete implementation or regression evidence. External **P1** denotes a focused first experiment when its named bottleneck exists, **P2** a conditional follow-up, and **P3** a short feasibility/rejection study. None bypasses the correctness-first release order in §13 or authorizes changed precision, active-history offload, altered routing or a different default speculative method. Named tensor/endpoint/vision representations and read-cache copies within the supported matrix are eligible experiments under Design Section 1; this register establishes neither their acceptance nor another target/expert routing topology.

| ID | Mechanism or lesson | Origin | Disposition / first gate | Details |
|---|---|---|---|---|
| A01 | Retention-aware prefill, independent execution/page/checkpoint grids | Repository audit | Evaluate actual execution/chunk boundaries. | [§3](#3-prefill-prefix-caching-and-mixed-scheduling) |
| A02 | Immutable prefix checkpoints inside physical pages at useful rendered message boundaries | Repository audit | Evaluate the concrete policy; extend the existing coherent-boundary/COW contract. | [§3](#3-prefill-prefix-caching-and-mixed-scheduling) |
| A03 | Draft lookahead identity in prefix hashes | Repository audit | Evaluate dependency-aware reuse; never reuse a draft cache from a matching target prefix alone. | [§3](#3-prefill-prefix-caching-and-mixed-scheduling) |
| A04 | Mixed-prefill controller with separate fixed and per-row cost | Repository audit | Evaluate; do not let small chunks create a self-reinforcing overhead loop. | [§3](#3-prefill-prefix-caching-and-mixed-scheduling) |
| A05 | Piecewise graphs for short mixed batches, plus optimized resident decode/GDN rows | Repository audit | Evaluate independently of pure-decode FULL graphs. | [§3](#3-prefill-prefix-caching-and-mixed-scheduling) |
| A06 | Capacity-tail graphs and attention routes for partial verifier widths | Repository audit | Evaluate all actually executable widths, coordinates and near-limit states. | [§3](#3-prefill-prefix-caching-and-mixed-scheduling) |
| A07 | Balanced prefill chunks and measured pipeline handoff | Repository audit | Evaluate useful cost balance, preserving retained boundaries and FIFO fairness. | [§3](#3-prefill-prefix-caching-and-mixed-scheduling) |
| A08 | Indexed/grouped expert execution without expanded routed inputs | Repository audit | Evaluate; retain exact source formats and route semantics. | [§4](#4-packed-weights-moe-execution-and-placement) |
| A09 | Equal-byte IQ packets and single-bank original expert storage | Repository audit | High-value experimental candidate; distinguish current operators from unmerged storage integration. | [§4](#4-packed-weights-moe-execution-and-placement) |
| A10 | Small-M expert vector/joint gate-up routes and complete MoE crossover maps | Repository audit | Evaluate actual per-expert occupancy, TP8 shapes and routing skew. | [§4](#4-packed-weights-moe-execution-and-placement) |
| A11 | Fused small routing/alignment/unroute and winner-only routing normalization | Repository audit | Evaluate with exact IDs, tie rules and declared FP32 reduction order. | [§4](#4-packed-weights-moe-execution-and-placement) |
| A12 | Activation reuse, interleaved packet creation and gate/up/SwiGLU/quantizer fusion | Repository audit | Reuse where the selected activation profile already permits its quantizer. | [§4](#4-packed-weights-moe-execution-and-placement) |
| A13 | Codebook word loads, bank-aware small shared tables and ordered split-K completion | Repository audit | Evaluate measured instruction/synchronization costs; any added persistent full bank needs separate duplication qualification. | [§4](#4-packed-weights-moe-execution-and-placement) |
| A14 | Mixed-format joint projections and direct final-layout writes | Repository audit | Evaluate; include staging, copies and temporary memory. | [§4](#4-packed-weights-moe-execution-and-placement) |
| A15 | Packed input embeddings and vocabulary heads with true target/draft aliases | Repository audit | Evaluate actual source tensors; avoid automatic full FP16 expansion. | [§4](#4-packed-weights-moe-execution-and-placement) |
| A16 | Sharded DFlash selector codebooks | Repository audit | High-value memory candidate from an open PR; trade lookup communication against saved replication. | [§8](#8-vocabulary-sampling-and-speculative-methods) |
| A17 | Bit-preserving packed candidate communication and GPU compact sampling | Repository audit | Evaluate exact semantics and complete sampling cost, including dense fallback. | [§8](#8-vocabulary-sampling-and-speculative-methods) |
| A18 | Optional reduced draft vocabulary, lookup/ngram or tree proposals | Repository audit | Separate method experiments only; retain normalized actual q and exact target verification. | [§8](#8-vocabulary-sampling-and-speculative-methods) |
| A19 | Direct encoded device-history readers and batch KV appends | Repository audit | Covered; compare concrete readers without importing host authoritative history. | [§5](#5-qsa-selection-history-readers-and-attention) |
| A20 | Exact live-length QSA selection, register-resident queries and multiquery key reuse | Repository audit | Evaluate actual row/coordinate bounds; shared-key/selection caps remain separate. | [§5](#5-qsa-selection-history-readers-and-attention) |
| A21 | Selected-KV grouping, logical execution order and allocation invariance | Repository audit | Evaluate; fewer bytes alone does not guarantee a faster attention kernel. | [§5](#5-qsa-selection-history-readers-and-attention) |
| A22 | GDN row-strided inputs, direct outputs, fused preprocessing, indexed state and tuned value tiles | Repository audit | Evaluate actual local heads, state order, cast points and every accepted prefix. | [§6](#6-gdn-execution-replay-and-state-representation) |
| A23 | PLE integer input/hash fusion and reused asynchronous metadata | Repository audit | Evaluate; preserve EOS, signed wrapping, request boundaries and rollback. | [§7](#7-ple-preparation-and-lookup) |
| A24 | PLE prefill convolution without padded history, bounded channel chunks | Repository audit | Evaluate complete convolution/activation/state commit and peak memory. | [§7](#7-ple-preparation-and-lookup) |
| A25 | Mapped, direct-I/O, pinned packed and bounded-readahead PLE transport | Repository audit | Evaluate cold/warm NUMA/PCIe/storage conditions; keep table access distinct from active KV. | [§7](#7-ple-preparation-and-lookup) |
| A26 | Early prompt-only PLE requests and earlier consumer-safe buffer release | Repository audit | Evaluate dependency/lifetime proof; graph replay requires its own release protocol. | [§7](#7-ple-preparation-and-lookup) |
| A27 | Persistent request state and incremental CPU/worker metadata | Repository audit | Covered; explicitly avoid retransmitting full token history every decode step. | [§9](#9-resource-ownership-host-work-and-communication) |
| A28 | Direct GPU slot handoff, batched state copies and pooled checkpoint buffers | Repository audit | Evaluate where ownership changes actually require copies; prefer a move/rebind when legal. | [§9](#9-resource-ownership-host-work-and-communication) |
| A29 | GPU capture followed by deferred host checkpoint transfer | Repository audit | Evaluate optional-cache TTFT benefit; never weaken protected checkpoint durability or progress. | [§9](#9-resource-ownership-host-work-and-communication) |
| A30 | Shared scratch by lifetime, stream and actual partition capacity | Repository audit | Covered; compare concrete arena grouping and prevent cross-product maxima. | [§9](#9-resource-ownership-host-work-and-communication) |
| A31 | Bounded shared positional tables and analytic/native RoPE | Repository audit | Covered; add target/indexer/MTP parity and long-coordinate range cases. | [§10](#10-media-and-positional-execution) |
| A32 | Topology/payload/precision-aware collectives and HC boundary fusion | Repository audit | Evaluate against NCCL on the real eight-GPU cube. | [§9](#9-resource-ownership-host-work-and-communication) |
| A33 | Nonduplicating HC residual-row/sequence ownership | Repository audit | Research candidate for prefill; include complete HC communication and quality. | [§9](#9-resource-ownership-host-work-and-communication) |
| A34 | Async eager pipeline sends with immutable send buffers | Repository audit | Two-server TP8×PP2 transfer-lifetime study; graph/static buffers need earlier retirement, and the source does not qualify RDMA. | [§9](#9-resource-ownership-host-work-and-communication) |
| A35 | GGUF loader slicing, bounded conversion and preparation/tuning reuse | Repository audit | Covered; avoid full-expert transcoding on every rank and live-request autotuning. | [§11](#11-build-runtime-validity-and-source-components) |
| A36 | Bounded media microbatches, exact companion-tower loading and preprocessing cache | Repository audit | Covered; preserve the fixed vision envelope and correct cache identity. | [§10](#10-media-and-positional-execution) |
| A37 | Exact native Inco DFlash2 implementation and FP32-output/range safeguards | Repository audit | Immediate range/precision qualification prerequisite. | [§8](#8-vocabulary-sampling-and-speculative-methods) |
| A38 | High-expert-ID, padding, stale-row and unwritten-output regression probes | Repository audit | Concrete validity requirements before accepting performance evidence. | [§11](#11-build-runtime-validity-and-source-components) |
| A39 | Executed-route/build/capture census and low-overhead critical-path measurement | Repository audit | Covered; distinguish configured, initialized, captured and actually replayed paths. | [§11](#11-build-runtime-validity-and-source-components) |
| A40 | Per-rank NUMA placement, exact-size pin registration and bounded asynchronous host work | Repository audit | Covered; measure physical placement and actual pinned bytes, including allocator rounding and hidden copies. | [§9](#9-resource-ownership-host-work-and-communication) |
| A41 | Accepted-prefix recurrent replay instead of per-proposal state snapshots | Repository audit | Evaluate the concrete memory/compute tradeoff; retain exact verifier inputs, checkpoint boundaries and complete transaction semantics. | [§6](#6-gdn-execution-replay-and-state-representation) |
| E01 | Current-query witness cutoff before dense QSA score writeback | Published principles + ProviderOne adaptation | P1: exact selection-chain experiment | [§5](#5-qsa-selection-history-readers-and-attention) |
| E02 | Certified immutable pooled-key block envelopes | ProviderOne synthesis from retrieval bounds | P2: bound-tightness pilot | [§5](#5-qsa-selection-history-readers-and-attention) |
| E03 | One raw GDN K-Gram for three value-head siblings | ProviderOne synthesis from grouped geometry | P1: inspect executed redundancy, then prototype | [§6](#6-gdn-execution-replay-and-state-representation) |
| E04 | GDN diagonal similarity and fused inference WY | Derivation + unmerged implementation | P2: numerical and native-kernel research | [§6](#6-gdn-execution-replay-and-state-representation) |
| E05 | Work-centric/hybrid residual-wave grouped experts | Stream-K adaptation | P1 only with measured wave/straggler loss | [§4](#4-packed-weights-moe-execution-and-placement) |
| E06 | Exact joint block verification | Published sampling algorithm | P2: MTP-first proof and device-cost study | [§8](#8-vocabulary-sampling-and-speculative-methods) |
| E07 | Exact grammar-support projection before the vocabulary head | ProviderOne synthesis | P1 for narrow structured-output states | [§8](#8-vocabulary-sampling-and-speculative-methods) |
| E08 | Pre-draw heterogeneous verifier-row allocation | TETRIS-inspired policy | P2: concrete extension of width adaptation | [§8](#8-vocabulary-sampling-and-speculative-methods) |
| E09 | Route-aware partitioning of independently necessary full microbatches | Research-inspired ProviderOne scheduling synthesis | P2: trace-only cost/predictability test first | [§4](#4-packed-weights-moe-execution-and-placement) |
| E10 | Withdrawn: full-expert ownership (ETP1) | Historical ProviderOne synthesis | Excluded; no implementation, profiling or benchmark work | [§4](#4-packed-weights-moe-execution-and-placement) |
| E11 | Shared-prefix decode computation, beyond cached prefill | Hydragen/Cascade adaptation | P2: shared-page workloads only | [§5](#5-qsa-selection-history-readers-and-attention) |
| E12 | Lossless, tile-addressable static weight entropy coding | Published compression + source-format adaptation | P3: actual-artifact entropy/cost census | [§4](#4-packed-weights-moe-execution-and-placement) |
| E13 | Lossless idle-checkpoint and backing-tier compression | Published transfer codecs + lifecycle synthesis | P2 for queued/retained service; no active-capacity claim | [§9](#9-resource-ownership-host-work-and-communication) |
| E14 | Stock-driver VMM for contiguous virtual history | vAttention-inspired allocator experiment | P3: granularity/cost proof first | [§9](#9-resource-ownership-host-work-and-communication) |
| E15 | Selective multiword FP16 Tensor Core products | V100 numerical-linear-algebra research | P2 only for an actual FP32-input bottleneck | [§4](#4-packed-weights-moe-execution-and-placement) |
| E16 | Activation-dependent partial-sum lookup | LUT-GEMM/CodeGEMM adaptation | P3: likely difficult for the selected IQ3_S | [§4](#4-packed-weights-moe-execution-and-placement) |
| E17 | SM-aware mixed prefill/decode attention | POD-Attention algorithm adaptation | P2: complementary-resource mixed workloads | [§3](#3-prefill-prefix-caching-and-mixed-scheduling) |
| E18 | Exact distributed selection over unique pooled-key shards | Upstream algorithm + ProviderOne architecture experiment | P2: fixed-placement cost study | [§5](#5-qsa-selection-history-readers-and-attention) |
| E19 | Actual-draft-token PLE lookup lookahead | ProviderOne synthesis from deterministic lookup | P1 if target PLE is on the critical path | [§7](#7-ple-preparation-and-lookup) |
| E20 | Fixed-function NVDEC with exact device-resident video preparation | Verified V100 capability + pipeline adaptation | P2 for qualified video workloads | [§10](#10-media-and-positional-execution) |
| E21 | Deferred FP32 GDN state materialization with bounded update records | ReplaySSM/TreeWY adaptation | P1: numerical, traffic and lifecycle feasibility | [§6](#6-gdn-execution-replay-and-state-representation) |
| E22 | Token-exact forced spans through bounded teacher-forced chunks | Compressed-FSM inspiration + stricter ProviderOne eligibility | P1 for genuinely singleton token paths | [§8](#8-vocabulary-sampling-and-speculative-methods) |
| E23 | Certified original-weight vocabulary-head pruning | Exact MIPS + ProviderOne synthesis | P3: bound-tightness and sampler-eligibility pilot | [§8](#8-vocabulary-sampling-and-speculative-methods) |
| E24 | Bit-preserving encoding of active FP32 GDN checkpoints | Scientific lossless coding + ProviderOne synthesis | P3: actual-state entropy/sector-cost census | [§6](#6-gdn-execution-replay-and-state-representation) |

The mechanisms remain independent experiments even where they share an operator:

| Area | Distinctions to preserve |
|---|---|
| QSA | A20 scores/selects live work; E01 avoids dense score writeback; E02 may skip key reads with certificates; E18 changes fixed unique ownership/global selection; A21 groups selected attention; E11 reuses shared-prefix reads across actual overlapping selections |
| GDN | A41 replays retained verifier inputs after acceptance; E21 defers materialization across rounds; TreeWY is a per-round reconstruction comparator; E03 shares raw Gram arithmetic; E04 changes a factorization; E24 preserves serialized state bits |
| Head and sampling | A15 packs original head weights; A17 compacts sampling/transport after logits; E07 restricts exact legal support before projection; E22 batches forced token execution; E23 certifies omitted dot products; A18 changes proposal methods/vocabularies |
| Storage | A09 is equal-byte IQ repacking; E12 entropy-codes known static weights; E13 compresses idle backing; E24 reserves raw active-state capacity and targets traffic |
| Ownership | A32 changes collective schedules; A33 changes HC row ownership; E18 changes indexer ownership. Their input-layout assumptions and costs cannot be combined implicitly |
| PLE | A25 chooses transport; A26 starts known prompt work early; E19 uses actual draft IDs to create a target-lookup lead window |

## 3 Prefill, prefix caching and mixed scheduling

### A01 Retention-aware prefill boundaries

1Cat found a scheduler bottleneck that required no faster arithmetic kernel. Prefix caching retained sparse recurrent-state checkpoints, but align-mode scheduling still split the entire model forward at every 816-token state block. An 8192-token budget therefore became many smaller forwards, repeating projection, expert and collective overhead. Its repair schedules across discarded state boundaries and stops at every boundary actually required for replay, shared prefixes or configured periodic retention.

The paired Flash-Next NVFP4/TP4/MTP4/FP16-KV report records:

| Actual input rows | Prefix-on control prefill | Retention-aware prefill |
|---:|---:|---:|
| 8192 | 3060 tokens/s | 5356 tokens/s |
| 32768 | 3046 tokens/s | 5068 tokens/s |
| 131040 | 2818 tokens/s | 4666 tokens/s |

The 8K row uses warmed cold requests; 32K/128K combine two cold requests per arm. Both arms preserve measured warm-prefix reuse. There was one successful startup per configuration, and concurrent prefill/256K were outside that report's qualification. PR #754 is merged; its body retains earlier draft-stage wording, so merge metadata and the measurement scope must be read separately.

**Design mapping:** Design Sections 9, 16 and 17 already require coherent checkpoints and bounded chunking, but do not explicitly forbid splitting every model forward at checkpoints that will immediately be discarded.

**Implementation addition:** Separate physical KV page boundaries, recurrent checkpoint-retention boundaries and execution-chunk boundaries. Crossing a discarded checkpoint boundary must not force a complete model forward. End exactly at every retained coherent checkpoint and preserve MTP back-off, rollback, cancellation and fairness boundaries. Measure actual scheduled chunk sizes and forwards per prompt with prefix caching enabled, including zero-hit cold requests. Keep cold prefill and warm reuse in the same acceptance matrix.

Evidence: [retention-aware prefill report](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_flash_next_prefix_prefill.md), [merged PR #754](https://github.com/1CatAI/1Cat-vLLM/pull/754).

### A02 Useful partial-page prefix checkpoints

The pinned 1Cat release profile combines 2048-token recurrent blocks with **partial prefix-cache entries** at the rendered system/message boundaries, rounded to its 64-token GDN chunk. It copies the partially filled attention page, coherent recurrent state and required draft sliding-window blocks into cache-owned storage. A hit continues on private copies. Entries become eligible only after their producing worker step, and reclaim releases them before displacing running work.

Its eight-turn agent example reports 99–100% of the reusable prefix hit, leaving approximately 30–60 replayed tokens beyond new input; four conversations' median TTFT decreases from 1.07 to 0.71 s. These are application/cache results, not faster arithmetic over uncached prompt tokens.

This complements retention-aware large prefill: cross discarded checkpoints freely, but preserve a small set of useful exact boundaries. ProviderOne already requires coherent internal-group reuse, partial-page COW and bounded useful checkpoints. The addition is an explicit boundary-selection policy and measurement of avoided replay. Its checkpoint must additionally include ProviderOne's pooled QSA/raw tails, PLE and selected method state; the generic full-attention/Mamba/SWA implementation is not a complete serializer for ProviderOne.

Do not copy 1Cat's fixed “three scheduler steps” deferred-release assumption into another execution engine. ProviderOne must retire the actual producing/copying readers and publish all-rank readiness. Similarly, its quarter-pool cache bound is an implementation choice, not ProviderOne's capacity formula.

Sources: [current release/cache profile](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/serving/sm70_release_profile.md), [partial-entry ownership](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/vllm/v1/core/partial_prefix_cache.py), [allocation, copying and hit selection](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/vllm/v1/core/kv_cache_manager.py), [partial-cache tests](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/tests/v1/core/test_partial_prefix_cache.py).

### A03 Dependency-aware draft prefix identity

Open 1Cat [PR #1276](https://github.com/1CatAI/1Cat-vLLM/pull/1276), head bd3b78bcd4b11981dd45fbb968e9f55ffd5833e3, includes the following token in EAGLE/MTP block identity. That dependency explains why simply reusing the last matching target block can be invalid, and why conservative implementations discard a large matched block.

ProviderOne already retains MTP's actual next-position embedding dependency. A dependency-aware cache key can safely recover reuse otherwise lost to blanket back-off. Include visual next-row identity and the whole effective profile; matching a text token alone cannot validate a visual-conditioned boundary. Prefix identities must not depend on fresh-request sampling settings when those settings do not affect model state.

### A04 Overhead-aware mixed-prefill control

1Cat's original proportional budget divided an entire mixed step's time by prefill rows. Fixed decode/launch overhead then looked like expensive prefill, shrinking chunks and increasing the number of expensive steps. Its affine replacement models fixed plus per-token cost and guarantees progress by a scheduling quantum. Later work excludes graph-executed samples from the eager cost fit and adds a floor near the fixed-cost crossover.

The pinned packaged profile nevertheless disables the mixed latency target: on that model, roughly 130 ms of fixed mixed-step overhead caused about 100-row chunks and poor prefill. A positive target remains optional for latency-sensitive service. This is a workload tradeoff, not a universal argument to disable mixed-load fairness.

ProviderOne should model execution regime, active/resident counts, history range and pending work separately. Preserve oldest-eligible admission, reserve resident service, and avoid both indefinitely blocking decode behind a large prompt and fragmenting prefill into launch-bound slivers. Include incoming TTFT, longest resident update gap and total completion time alongside throughput.

Sources: [current controller](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/vllm/v1/core/sched/mixed_prefill.py), [mixed-load tradeoffs](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/mixed_prefill_latency.md), [affine repair #1288](https://github.com/1CatAI/1Cat-vLLM/pull/1288), [graph-aware repair #1290](https://github.com/1CatAI/1Cat-vLLM/pull/1290), [current profile](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/serving/sm70_release_profile.md).

### A05–A06 Resident-row fast paths and mixed/tail graphs

A mixed batch must not send resident verifier rows through scalar attention once a long prefill joins it. 1Cat gathers those rows into request-major groups, retains per-row device-authoritative causal lengths, and caches the row plan once per step. Its GDN verifier similarly consumes only the rows named in speculative metadata and merges outputs by token index. That is useful even when the overall mixed step remains eager.

Newer piecewise captures cover short mixed steps through 256 total rows while keeping pure verifier FULL graphs. The reported graph allocation rises from 0.50 to 0.59 GiB; costs and work coverage matter more than a graph-count headline. Reported MT-Bench throughput also decreases in that PR, so its other workload gains cannot waive ProviderOne's per-count decode gate.

Capacity-tail work is separately important. 1Cat found that ordinary DFlash q1–q7 tails did not register graphs and that attention manifests ended below the service's declared context. Its recovery improves a recorded near-262K q7-draft complete round by about 4–6 ms, with short controls essentially unchanged. This is the dense 27B DFlash contract, not the selected Inco block-four contract; copy the boundary-coverage method, not its shapes or manifest.

Sources: [mixed rows](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_mixed_batch_decode_rows.md), [short mixed graphs #1290](https://github.com/1CatAI/1Cat-vLLM/pull/1290), [tail graphs and final guards](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_dflash2_tail_graphs_20260911.md).

### A07 Balanced prefill chunks and stage cost

Balanced prefill chunks also avoid one very small final chunk or a badly imbalanced pipeline stage. Both repositories measure such policies. Balance **actual read rows and complete stage cost**, preserving retained checkpoints, media indivisibility and fairness; do not assume the largest chunk or an even row split is fastest. The new dense-attention split-KV path for short query tails over long prefixes is a related shape-specialized candidate, not a replacement for QSA.

Sources: [Strata pipeline planning](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/include/strata/program/prefill_plan.hpp), [1Cat measured prefill worklog](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/benchmarks/flashnext_prefill/OPTIMIZATION_20261014.md), [short-prefix split-KV #1298](https://github.com/1CatAI/1Cat-vLLM/pull/1298).

### E17 SM-aware mixed prefill/decode attention

**Addition.** Scheduling prefill and decode in one batch does not ensure that their attention kernels share an SM effectively. POD-Attention supplies a concrete CTA-parallel fusion: a launched CTA receives prefill or decode work according to bounded software scheduling. Its published measurements are on A100; the inspected source requires Ampere or newer. This is an algorithm reference for a native SM70 experiment, not an installable V100 kernel. [S27]

Keep the existing fast prefill and decode arithmetic inside one qualified operator plan. Assign independent work through device counters/descriptors, without a CPU decision per CTA or a model-wide persistent scheduler. Use operation-specific useful tiles and preserve per-request masks, causal lengths, selected pages, output ownership and complete finalization. The work must already be dependency-ready; fusion cannot bypass recurrent or collective dependencies.

**V100 resource boundary.** A fused kernel's register, thread and shared-memory reservation can prevent co-residency even if standalone decode uses little memory. If a prefill tile consumes nearly the full 96-KiB SM70 shared-memory budget, merely combining grids cannot provide two concurrent CTAs. Measure smaller useful tiles and the actual common launch footprint, including lost prefill efficiency. More CTAs or theoretical occupancy is not a win by itself.

The source uses SM-ID-based tickets. PTX documents that SM identifiers need not be contiguous, their extent can exceed the active SM count, and a thread's SM may change after preemption. Size/bound bookkeeping using the supported identifier range; do not index an assumed 80-entry array. Treat SM identity as a placement hint: unique task assignment and progress cannot rely on permanent binding. Reset counters and retire scratch correctly on graph replay; no spinning global barrier. [S36]

**First test / stop condition.** Compare serial specialized kernels, eligible independent-stream execution and a bounded native fused plan at identical mixed workloads. Measure complete mixed-step time, incoming TTFT, resident inter-token gaps, prefill completion, HBM/SM use and capacity. Preserve the pure-decode plan. Stop if the two workloads compete for the same bottleneck, common resource reservations prevent useful overlap, or fusion harms resident service.

## 4 Packed weights, MoE execution and placement

### A08 Indexed grouped expert input

1Cat's NVFP4 prefill path retains stable expert ordering but passes original token-row indices into TurboMind rather than creating a full [tokens × top-k, hidden] FP16 activation matrix. Its reported 8192-token operator chain falls from 6.026752 to 4.235264 ms. Under its TP4/no-MTP/NVFP4 contract, 64K prefill rises from 5777.43 to 6241.48 tokens/s. Outputs and routing metadata match its control.

For ProviderOne geometry, an explicitly materialized 8192 × 10 × 2560 FP16 routed-input array alone is **400 MiB**, excluding output, alignment and other staging. That is arithmetic, not evidence ProviderOne currently allocates it. Indexed loads can remove this particular expansion; they do not remove necessary expert outputs or accepted-prefix recovery.

**Design mapping:** Design Section 15 calls for grouped MoE and device-driven routing, but the avoidance of full input-row expansion is not an explicit implementation requirement.

**Implementation addition:** Prefer caller-owned expert row lists/offsets and kernels that read canonical activation rows through indices. Reuse activations across gate/up, retain routing probabilities and tie/order semantics, and reduce outputs in a declared order. Quantify metadata, indexing, padding, coalescing and scratch costs over the entire FFN chain. At 24 requests, calibrate on actual per-expert occupancy and routing skew, not the total scheduled token count alone.

The indexed NVFP4 kernel is not an IQ3_S decoder. Port its dataflow to each actual SC117 format rather than changing target quantization to make the kernel applicable.

Evidence: [merged indexed-prefill PR #390](https://github.com/1CatAI/1Cat-vLLM/pull/390), [native small grouped implementation](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/csrc/sm70_turbomind/ops/gguf_small_grouped_sm70.cu).

### A10/A13/A14 Mixed-format readers, fusion and crossover maps

1Cat's GGUF work distinguishes canonical affine, LUT and lattice representations from source-sized native readers. Compatible gate/up pairs share an activation tile while decoding each projection through its own format. Joint GDN Q/K/V/Z/a/b work writes the final logical layout directly, removing launches and concatenation. QPN layouts organize small-M work around Volta's m8n8k4 instruction and reuse activation fragments.

The native linear body also performs ordered split-K completion inside the last CTA, with publication fences and replay-reset completion tickets. This is concrete graph-safe implementation detail, not a recommendation to combine arbitrary reductions or use unordered floating atomics.

A 27B IQ3_S/DFlash2 projection-batch report measures about 1.58 ms saved per full round at 1K/8K inputs; that model has different hidden and layer geometry from Flash-Next. The directly relevant Flash-Next operator report instead records C1 26.072 ms/round and 92.267 tokens/s, and C4 52.059 ms/round and 280.441 aggregate tokens/s under its stated short-history/FP16-KV workload. These values are separate development records, not additive speedups.

**Design mapping:** Design Sections 2 and 15 allow format-preserving repacking, small-M kernels and fusion. They do not name these concrete SM70 readers or require a source-format × actual-M × local-shape crossover ledger.

**Implementation addition:** Build one dispatcher using actual source formats, sharded dimensions, rows, strides, precision and workspace. Compare native packed SIMT/dp4a only when its numerical profile is authorized, native FP16 MMA, and bounded decode-to-FP16 plus FP32-accumulation BLAS. Select by whole-chain measurements, including preparation and output-layout work. Preserve original scale products and cast points unless a separately qualified numerical representation is selected.

Sources: [source-sized mixed readers](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/gguf_native_mixed_pair.md), [QPN joint projection contract](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/gguf_qpn_qkvz.md), [shared activation linear source](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/csrc/sm70_turbomind/ops/gguf_linear_shared_a_sm70.cuh), [Flash-Next operator integration](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_flashnext_gguf_mtp4_operator_routes.md).

Strata's relevant Volta grouped expert code is in **src/kernels/cuda/iq_kernels.cu**, including gemm_iq_f16_grouped and its second tile. The similarly named src/prefill/wmma_gemm.cu is an AMD/RDNA path and is not the V100 implementation.

Strata's second tile supports IQ3_XXS/IQ3_S/IQ2_S gate/up and IQ4_NL down, with two operand orientations selected around measured routed-row occupancy. Reported dual-card prefill improvements at equal prefill caps are 2.83–3.85%; larger single-card gains also change the chunk cap. This is a useful original-format grouped FP16 reference for SC117, not a universal tile threshold for TP8.

Source: [actual grouped expert code](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/src/kernels/cuda/iq_kernels.cu#L5228), [controlled prefill study](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/benchmarks/v100-iq3_s-prefill-decode-2026-10-23.md).

### Storage and arithmetic gates for packed routes

1Cat's canonical lattice layout can increase IQ3 storage to four bits/weight and expands nested scales into FP16. Some native integrations additionally retain source-sized banks beside canonical banks. One Flash-Next packed GDN experiment adds **725.625 MiB/rank at TP4**, reducing reported KV token capacity by about 20.3%; another promotion reports approximately 1.25 GiB/rank additional packed storage under its own configuration. These are separate records and cannot be summed.

ProviderOne should use replacement layouts or aliases where possible and release superseded representations. Retaining two full representations to accelerate different row bands is a separate duplication experiment under Design Section 1, with exceptional scrutiny when the whole-deployment increment reaches 1 GB. Recompute the complete TP8 ledger; the published TP4 per-rank overhead does not transfer unchanged.

Evidence: [canonical lattice storage](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/gguf_turbomind_lattice.md), [GDN batch memory tradeoff](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_flash_next_mtp4_batch_gdn.md).

**dp4a is a real option, with activation rounding.** 1Cat quantizes activation rows to Q8_1, fuses direct-route gate/up and consumes quantized intermediates in down/unroute. Its complete-model C1/C4 timing improves in a recorded screen, but its teacher logits and acceptance are not bit-identical to the FP16-activation control. ProviderOne already requires a separately selected/qualified activation-numerical profile for this choice. Do not treat original packed weights as proof that activation numerics are unchanged.

Source: [1Cat Q8_1 expert experiments](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/gguf_dp4a_small_m.md).

### A09 Equal-byte original IQ banks

**Storage-preserving packets** provide a distinct single-bank option. Open 1Cat [PR #897](https://github.com/1CatAI/1Cat-vLLM/pull/897), head 881ad90eae7ee4c7159ff7973bad0261b1a30ac8, proposes invertible IQ3_XXS/IQ3_S/IQ2_S permutations retaining exactly 98/110/82 bytes per 256 weights. Original scale/grid/sign information is preserved; optional rounded scale variants have separate tuning identities.

The related [PR #959](https://github.com/1CatAI/1Cat-vLLM/pull/959), head f02e37e081029791542a5dd43ca41d3a661ecc79, connects an equal-byte IQ3_XXS expert bank without canonical stats or an extra original bank. Its illustrative TP4 E512/N160/K2560 bank is 80,281,600 bytes versus 104,857,600 canonical weight bytes alone. Reported grouped operator rows 1/5/8/16/512 improve, but those are **input-row points, not client concurrency**, and complete-model quality/speed remain unmeasured.

This is directly aligned with ProviderOne's preference for a one-bank control and provides a lower-copy comparison for any dual-bank proposal. Test original-block bit inverses, official decoded values, packed-row TP tails, real route skew and complete FFN chains at TP8. Source-sized storage does not by itself establish identical canonical arithmetic or a supported MMA layout. Use replacement banks and release superseded representations after bounded validation; retaining both requires separate duplication qualification.

The existing merged “joint original experts” path instead retains approximately **4.20 GiB/rank at TP4** for 30 useful IQ2_S/IQ3_S layers alongside canonical storage. An additional IQ3_XXS bank would cost about 2.55 GiB/rank and is not retained there by default. These are separate from the GDN packed-bank examples above. They make a single-bank implementation especially important.

Sources: [merged joint bank tradeoff](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_gguf_joint_expert_dispatch.md), [original expert operators](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_gguf_joint_original_experts.md).

### A10–A11 Small routing, alignment, restoration and down projections

1Cat supplies a two-kernel small-routing chain: stable expert alignment plus activation gather, followed by weighted restoration without an inverse sort. Its later implementation reproduces Torch's four-accumulator FP32 reduction order and reports zero output differences in its operator domain. M1..32, hidden 2560 and E512 are explicit bounds. The advertised multi-millisecond 48-layer saving is an extrapolation; an earlier whole-model version still regresses C4.

Its router top-10 path uses lane-local sorted candidates and winner merging, exponentiating only selected winners when renormalization cancels the full softmax denominator. A selector-plus-Q8_1 encoder saves a launch boundary when the expert path already consumes those packets. Standalone selection regresses the default M5 MoE chain, while the combined encoder helps. Raw-logit and rounded-softmax tie semantics must be reconciled with the selected ProviderOne reference.

For down projections, a canonical small-M IQ4_NL/Q2_0 vector route wins at original M1/M5 and loses substantially at M20. Padding singleton expert intervals to M8 can waste more work than a Tensor Core kernel saves. Measure original rows, routed rows, active experts, empty intervals, padding and reduction together.

Sources: [exact small-routing chain](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_small_moe_routing.md), [router comparison](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_router_operators.md), [small down route](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_gguf_small_grouped_down.md).

### A12–A14 Cross-operation dataflow and small-table reuse

Strata's pinned default table selects more than the prefill tiles. It enables register/shared reuse in grouped expert decode, query-register QSA scores, interleaved Q8_1 packet production, gate/up/SwiGLU/quantizer fusion, HC down/up/norm schedules, decode attention and multi-block greedy selection. Several older guides call these opt-in; **the pinned SM70 default table is authoritative**.

Concrete mechanisms include:

- Stage a grouped expert's activation packets in a bank-aware shared layout and reuse decoded weights across its rows.
- Create ordinary and interleaved Q8_1 activation images in one launch when both consumers need them.
- Hoist HC up-projection loads while preserving the previous sum tree; split HC down work by useful token groups.
- Keep QSA query slices in registers and reorganize shuffle reductions without changing the declared pairings.
- Batch independent sessions' score/select/attention launches, preserving private page tables and lengths.
- Omit a separate commit graph only for an unconditional one-row-per-session path whose writes already constitute the canonical commit.

ProviderOne's FP16-activation path cannot silently acquire Q8_1 activations to use these kernels. Its full MTP4 verifier is five rows/request, while several Strata paths cover at most four queries or eight total rows. Also, calling three QSA stages in batch means one launch **per stage**, not a single fused score/top-k/attention kernel.

Sources: [current default map](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/src/core/nv_defaults.cpp), [verify dispatch and commit guards](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/src/core/verify.cpp), [grouped expert reuse](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/docs/EXPERTDEC.md), [query registers](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/docs/QSAROWS.md), [HC kernels](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/src/kernels/cuda/fused_gr.cu).

Codebook word copying, lane-aware shared replicas, wide packed readers and same-launch ordered reductions are additional implementation candidates. They preserve small tables or source records; full weight duplication remains a separate, explicitly priced experiment under Design Section 1. Some expanded-table experiments lose once initialization and real expert traffic are included. Screen these with cold rotated weight banks and complete chains.

Sources: [word loads](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/gguf_turbomind_codebook_words.md), [bank layout experiment](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/gguf_turbomind_lattice_banks.md), [rejected expanded codebooks](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/gguf_turbomind_lattice_grouped.md).

### A15 Packed embeddings, vocabulary heads and true aliases

1Cat's packed embedding work avoids constructing a full FP16 vocabulary table merely because a model adapter initially chose a dense embedding class. Its separate Q6_K Flash-Next LM-head work reduces head storage and operator service, including repeated MTP head calls. Actual target/draft sharing, tied weights, source-scale arithmetic and vocabulary layout must remain explicit.

The dense 27B embedding example has width 5120; it is not ProviderOne's width-2560 table. The Flash-Next Q6_K head report is more relevant: the subsequently inspected SC117 header identifies its distinct target head as Q6_K. Reconcile exact shape/block bytes with the runtime import before applying a reader. The reported packed-head path also retains raw plus canonical streams, so its entire footprint must be compared against a one-bank alternative.

Sources: [packed embeddings](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/gguf_packed_embeddings.md), [Flash-Next head](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_gguf_canonical_head.md), [head preparation](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/gguf_head_layout_preparation.md).

### E05 Work-centric grouped expert scheduling

**Addition.** Stream-K divides aggregate inner-loop work rather than relying entirely on output-tile waves. Adapt that decomposition to compatible routed-expert cohorts, particularly measured residual waves and highly uneven expert populations. This is more specific than ordinary grouped dispatch or a generic persistent kernel. [S07]

Flatten useful output/K intervals into a bounded task schedule. Compare a hybrid that leaves full waves under normal output ownership and redistributes only the tail. Retain the exact expert IDs, source formats and routing weights. Use device-owned descriptors; use affine scheduling when adequate and a task counter only when its measured benefit pays for coordination.

Every partial needs an explicit output/K interval and declared FP32 merge order. No unordered floating atomics, model-wide spinning barrier or assumption that all CTAs are resident. Keep scope local to a projection/FFN stage.

**Cost.** Partial traffic, descriptors, completion state, padding and larger instruction/register footprints can cost more than the idle tail. Tiny TP8 down-shard K80 may provide too little work to amortize a split; a fully utilized grouped route has little opportunity.

**First test / stop condition.** Replay real routed populations through the qualified grouped route, ordinary split-K and the bounded hybrid. Measure useful work, final-wave duration and the entire routing→gate/up→activation→down→combine chain. Stop without a whole-chain gain and preserved resident capacity.

### E09 Route-aware full-microbatch partitioning

**Origin and addition.** Research on expert activation patterns and batch clustering motivates a narrower ProviderOne adaptation: reduce repeated expert-bank reads across complete-model microbatches that are independently necessary. Reordering requests by coactivation is not a new general invention. [S11]; [S12]

A QSA-only row cap should split that operator while retaining a legal full MoE batch. Do not create whole-model microbatches to manufacture this opportunity. If full-forward splitting is genuinely required, preserve its shape limits and group ready requests using prior committed per-layer routing signatures.

Estimate total per-bin expert read cost from actual bank bytes and row-tile reuse, plus imbalance, padding and planning costs. A unique-(layer,expert)-bytes sum is only a proxy; repeated tile loads and cache behavior determine the result. Keep each request's required rows coherent and preserve slot IDs/owner mappings despite row permutation.

Signatures scale with requests × layers × top-k. One 32-bit last-committed signature at 24×48×10 is 46,080 B before counters, not merely 24×10. Keep metadata device-side or reuse existing transfers. A prediction sketch may be approximate because it selects scheduling only; current router outputs remain authoritative.

**First test / stop condition.** Replay identical traces through a bounded greedy/local-swap partitioner before writing kernels. Compare actual HBM bytes, whole-step time, fairness and prediction decay. Add no wait for future requests and no auxiliary embedding model. Stop if full batching is already legal, predictions are weak, or planning/padding/imbalance costs exceed reuse.

### E10 Withdrawn: full-expert ownership

The prior ETP1/full-expert-owner proposal is removed from the active backlog to reduce development time. No implementation, profiling, qualification or benchmark work is required. Retain E10 only as a historical identifier; expert-placement work is limited to the required ETP8/4/2 matrix. Related MoE placement background [S13] does not reinstate this proposal.

### E12 Lossless static-weight entropy coding

**Addition.** Beyond equal-byte permutation, investigate a smaller entropy-coded sole representation of fixed weight records, decoded in bounded tiles directly into the selected compute path. Static artifact bytes are known, so their exact compressed footprint can support a capacity proof.

DFloat11 compresses BF16 exponent information; NeuZip distinguishes lossless exponent coding from lossy inference variants. Neither establishes a ratio for FP16 or IQ3_S. Tile-addressable ANS work supplies another pipeline reference on newer GPUs. Its headline batch numbers are not used here; its retained HTML Qwen prose/table figures disagree, reinforcing the need for an independent byte/cost census. [S16]; [S17]; [S18]

**First experiment.** Measure actual per-tensor entropy and achievable independently decodable tile sizes, including signs, grids, scales, offsets, tables, padding and restart states. FP16 has five exponent bits, not BF16's eight; IQ records may already be near their useful entropy. Do not expand IQ weights into FP16 merely to report compression relative to the larger representation.

Only proceed when measured saving survives all metadata and worst-rank placement. Retain one compressed bank and bounded decoded tiles, releasing superseded storage. V100 needs its own register/shared software pipeline; newer-GPU producer/consumer kernels are not a ready dependency. Decode must reproduce original bits and their qualified scale/cast arithmetic.

**Stop condition.** Stop at the census if net saving is negligible. Otherwise compare complete cold-bank expert/head chains across real row counts, including decompression, tile reuse and scratch. Stop this storage-saving variant if reduced bytes cost decode throughput or require a persistent decoded shadow; a dual-bank speed variant is a separately priced duplication experiment under Design Section 1. Dynamic KV's unknown future entropy cannot justify the same capacity argument.

### E15 Selective multiword FP16 Tensor Core products

**Addition.** FP32 accumulation does not preserve FP32 input precision. V100-tested numerical-linear-algebra research uses sums of lower-precision components to improve the performance/accuracy tradeoff. Naive double-FP16 is not uniformly as accurate as FP32; blocked summation matters. [S23]

When one operand genuinely needs FP32 information and the other is already exact FP16, split only the FP32 operand into bounded high/low FP16 components. Two native FP16 Tensor Core products and an FP32 combination can avoid a second persistent weight bank. Range/scaling and omitted residual error need proof; two halves do not represent arbitrary FP32 inputs exactly.

Target only measured FP32-input GEMMs, potentially in recurrent prefill/state work. A route already using qualified FP16 inputs has little reason to add this machinery. Keeping an overflowing projection's output in FP32 is a separate issue and does not require multiword inputs.

**First test / stop condition.** Use actual matrices, exponent/cancellation extremes, nonzero recurrent states and real local dimensions. Include split/combine, graph and scratch costs. Compare native FP32, the selected reference and multiword variants through long-state/logit/task/acceptance gates. Stop without a complete-chain gain and a qualified numerical profile.

### E16 Activation-dependent partial-sum lookup

**Addition.** LUT-GEMM/CodeGEMM form activation-derived dot-combination tables, rather than simply decoding a weight codebook before multiplication. That is a different compute algorithm from the existing direct IQ readers. Their source formats and shipped reduction arithmetic are not ProviderOne's default. [S24]; [S25]

IQ3_S is a difficult fit: its 512 four-value magnitude grids, independent signs and block/subblock scales are not a plain three-bit scalar LUT. A naive signed-grid table has 512×16 FP32 entries, or 32 KiB for just one four-activation subgroup/row. Scaling that across K and verifier rows quickly exceeds useful shared memory. Tiny local expert populations may not reuse the table enough.

A possible bounded experiment factorizes demanded combinations into smaller exact tables. It must preserve decoded FP16 values and cast points; pulling a scale outside a dot can change rounding. Count table construction, deduplication, scale handling and ordered FP32 reduction. Do not change SC117 into another quantizer to make the kernel fit.

**First test / stop condition.** Prove representation/arithmetic compatibility and count demanded combinations before writing a serving kernel. Stop if exact tables are too large, scale/cast semantics cannot be retained, or build+lookup loses to direct MMA. This is a small feasibility exercise, not a proposed wholesale expert rewrite.

## 5 QSA selection, history readers and attention

A19 retains direct encoded device-history readers and batched KV appends under the selected codec. Decoder format, exact logical order, unique ownership and current causal length remain independent obligations; a host-authoritative source path is not imported.

**Q8 is not a shared codec contract.** Strata's INT8 KV uses one FP16 scale per 64 values, yielding 528 bytes per head's K+V and 1056 bytes for two heads per layer/token. ProviderOne's pinned ordinary Q8 contract is 544 bytes per head's K+V. 1Cat often uses FP8 E4M3 or FP16. These formats need different writers, readers, quality tests and capacity arithmetic. E4M3 is not signed INT8 Q8.

Source: [Strata INT8 schema](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/include/strata/kernels/kv_q8.hpp).

### A20–A21 Live scoring, exact selection and grouped selected attention

Three mechanisms merit independent evaluation.

**Live-length scoring.** Strata avoids scanning beyond visible pooled groups; its tiled SIMT scorer also reuses keys across prompt queries. The Oct 25 two-card report shows 65K prefill 27.609 to 26.212 seconds and 131K 58.216 to 51.909 seconds. That scorer changes FP32 summation order; one synthetic 128K selection set differs. Retain exact target masks and numerical/quality qualification rather than interpreting these speedups as byte-equivalent target computation.

**Exact top-k narrowing.** 1Cat's native selector uses a monotone coarse key to locate a candidate bucket, resolves the cutoff using full FP32 scores and stable lower-index ties, and preserves increasing logical output order. Oversized or ambiguous buckets stay exact. Its M20/65,688-live-score operator drops from 191.376 to 108.288 microseconds; short rows remain essentially unchanged. This is an especially attractive candidate because it improves the selection algorithm without treating approximate top-k as equivalent.

**Grouped selected attention.** 1Cat groups eight adjacent query rows, forms the union of selected four-token groups and carries per-query visibility masks. Its original NVFP4/FP16-KV report shows about 29.93% more 64K prefill throughput. A four-token selection group is an execution unit; ProviderOne need not replace its larger physical allocation pages with four-token pages.

The later allocation-invariance audit is essential: ordering work by physical hash slots changed floating reduction order after page relocation despite identical logical selections. Its repair orders by logical ownership and carries physical addresses only as payload. This is directly relevant to ProviderOne's live migration, restored slot mappings and prefix sharing.

The expanded history pass also recovered 1Cat's [historical sglang-sxm2 operator comparison](https://github.com/1CatAI/1Cat-vLLM/blob/87ffd9b08870730529229ebdf0813d277de417b3/docs/benchmarks/qsa_sxm2_20261011/README.md), subsequently removed from main by PR #1169. At its frozen earlier revisions, the 32K/four-request/five-query-row FP16 chain measured 340.99 microseconds for 1Cat device history versus 179.20 for SGLang; the 8K/single-request/five-row case favored 1Cat, 66.56 versus 78.85. The report localizes multi-request scoring and long top-k, and shows large page-geometry sensitivity. It provides another concrete operator comparator for A20/A21, not a current-main or full-engine ranking. Its random inputs do not qualify cutoff ties; preserve ProviderOne's deterministic logical selection rather than copying unordered collection. Rebenchmark current revisions and native allocation geometry before using the historical gap.

**Design mapping:** Design Section 15 broadly permits tile reuse and Design Section 9 requires exact causal selection. Add explicit measured live-range bounds, query-group reuse and an allocation-order invariant. Preserve one canonical history copy and bounded scratch. Test relocation, page aliases, partial groups, tails, all-empty shards and masked queries. The 1Cat shared-key scorer currently admits at most 64 query rows; that cap cannot serve every ProviderOne full-concurrency step unchanged.

Sources: [Strata scorer measurements](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/docs/NVIDIA_V100.md#tiled-prompt-selection), [1Cat exact selector report](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/flashnext_qsa_router_structure.md), [native multiquery scorer](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/csrc/sm70_turbomind/ops/qsa_indexer_sm70.cu), [grouped QSA PR #387](https://github.com/1CatAI/1Cat-vLLM/pull/387), [physical-allocation invariance repair](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_qsa_page4_allocation_invariance.md).

### E01 Current-query witnesses and fused exact QSA filtering

**Addition.** The existing exact top-k candidate A20 narrows selection after scoring. Establish a valid cutoff before dense score writeback, then fuse original scoring with exact filtering and bounded selection.

DoSA supplies witness/certificate reasoning; NVIDIA GVR V2 supplies self-sampling and multi-threshold exact-selection ideas. Their implementations and model geometries do not establish an SM70 implementation or ProviderOne gain. [S01]; [S02]

Retain 512 distinct logical group IDs from an eligible committed selection, rescore them with the current query and reference-equivalent four-head arithmetic, and take their minimum as a lower bound on the current 512th score. A different gather/GEMM shape must preserve the reference reduction/cast sequence; otherwise use certified lower score enclosures for the cutoff and obtain reference scores for final selection. An upward witness error can make pruning unsafe. Validate generation and per-query causal visibility; supplement an incomplete witness set with other distinct visible IDs. Below 512 visible groups, ordinary all-visible behavior applies.

Scan the remaining domain with the unchanged scorer. Reject only scores strictly below the valid cutoff. Merge the reference-scored witness records with all remaining-domain survivors, including each logical ID exactly once, for exact stable selection. Process excess candidates through bounded tiles and exact merges. Never truncate an overflowing survivor buffer. Keep lower-logical-ID cutoff ties and increasing logical attention order. A self-sampled witness is another mode of the same algorithm, not a second candidate.

Previous winners are hints, not reused answers. Weak correlation affects cost, not correctness. Every verifier row needs its own current-query cutoff; rollback, prefix forks, media positions and slot reuse cannot validate stale visibility. Hints are disposable and need not become required continuation state.

**Cost.** A 512-ID copy costs 2 KiB/cache: ideally 78 KiB/rank at 24 MTP sessions and 13 indexed caches/session, before metadata. Rescoring 512 of 65,536 pooled keys is 0.78125% of the native full domain. All remaining keys still receive their four dot products; the intended saving is score traffic, selection passes and scratch.

**First test / stop condition.** Compare the complete scoring+selection chain, including witness work, bytes written and candidate fraction. Exercise equal scores, adjacent FP32 values, masks, partial groups, changed activity and physical relocation. Require identical logical winners. Stop if compaction/rescoring costs exceed the saved work.

### E02 Certified pooled-key block envelopes

**Addition and origin.** Combine Quest-style page summaries with witnessed exact rejection. Quest's approximate page budget is not adopted; this proposal skips a block only when it is provably unable to contain a winning QSA group. [S03]; [S01]

For each completed block of canonical pooled keys, store coordinate minima and maxima. For query head h, a real-arithmetic dot upper bound is

\[
u_h(B)=\sum_d \max(q_{hd}m_d,\ q_{hd}M_d).
\]

Apply the profile's actual nonnegative QSA head aggregation and scaling to the ReLU head bounds. The implemented upper enclosure must additionally cover the reference dot/reduction/scaling roundoff. Reject the block only when this conservative bound is strictly below E01's witnessed cutoff; otherwise score its visible keys normally. Equal or numerically ambiguous bounds remain eligible.

Summarize the final normalized/rotated stored keys, not raw pre-RoPE or pre-pooling values. Revision 21 keeps pooled keys FP16 even when target/MTP KV is Q8. Their exact coordinate endpoints are representable; a narrower summary or another source dtype requires outward-rounded endpoints. Nearest-rounded bound arithmetic alone is insufficient. Invalid/nonfinite certificates cannot authorize skipping.

Summarize only complete immutable indexer blocks and fully score the open ending block. Bind metadata to profile, grouping origin and cache/session generations. Summaries are reconstructible acceleration metadata; they cannot replace authoritative history.

**Cost.** At 64 pooled keys/block, keys occupy 16 KiB and two 128-coordinate FP16 endpoints occupy 512 B: 3.125% of pooled-key payload. At native context this adds 0.5 MiB/cache, or 19.5 MiB/rank for 24 MTP sessions under balanced ownership, before headers/alignment. This must fit without reducing the qualified integer resident maximum.

**First test / stop condition.** Compute certificate survival rates on real traces before implementing the full kernel. Test blocks 8/16/32/64, metadata traffic and finite-precision bounds. High-dimensional boxes can be very loose. Abandon the representation if most blocks survive or saved key reads do not pay for bounds and metadata. A centroid/radius variant belongs to this same family.

**Lower-metadata variant: bound changes from a fully scored anchor query.** This additional synthesis combines retrieval certificates with bounded-change reasoning. For an immutable block, retain an upper enclosure P_B of its maximum score at anchor query p, a certified key-norm maximum R_B, and the actual anchor query. ReLU is 1-Lipschitz, so for the four equally weighted heads,

\[
U_B(q)=P_B+\frac{R_B}{\sqrt{128}}\sum_{h=1}^{4}\|q_h-p_h\|_2+E_B
\]

is an upper enclosure when E_B conservatively covers the difference between both reference floating-point scorers and their real-arithmetic forms. Skip only when U_B is strictly below a valid current cutoff. Compare final normalized/rotated query operands against unchanged stored keys; the new/open block is scored normally. Never substitute small positional distance for a measured query-vector bound.

Keep the anchor fixed until a complete qualifying scan refreshes its maxima; partially skipped work cannot silently advance it. Bind anchor/blocks to visibility, profile and generations. One FP32 maximum per 64-key block is 4 KiB/cache at native length, plus norm/query/error metadata, rather than a coordinate box. Near-query continuity is unproven and the norm bound may be useless in 128 dimensions: pilot survival and refresh costs before retaining this variant. It uses previous work as a certificate, never as an assumed current answer.

### E18 Exact selection over unique pooled-key shards

**Addition.** Revision 21 keeps each indexer history and selection at one canonical owner and explicitly defers distributed exact selection. A concrete alternative stores each pooled key once across a fixed degree of owners. Each owner computes complete per-key QSA scores, selects its local top-512, and participates in exact global merging.

Current upstream vLLM contains a local-top-k/global-merge implementation absent from the inspected 1Cat snapshot. It packs score/global-ID candidates, all-gathers them and selects global winners. Its first implementation targets Hopper/Blackwell and uses CuteDSL; only the algorithm and ownership examples transfer. [S28]

**Exactness requirement.** Use one total order at both local and global stages: descending reference score, then ascending logical global ID. An item outside its owner's local top-k has at least k items ahead of it locally and cannot be in global top-k. A stable global selector cannot recover correct tie IDs already discarded by an inconsistent local selector. Send all visible local items when fewer than k exist, with explicit padding/empty-shard handling.

Compute all four query-head contributions for each owned key using the qualified scorer. Do not select on partial-head scores or add a new cross-rank floating reduction. Preserve causal frontiers, complete pooled-group identity, tail treatment and final increasing logical attention order. Changed batch/GEMM shapes still need the reference-arithmetic qualification.

**Cost.** For k=512 and an eight-byte FP32-score/INT32-ID record, local candidate input is 4 KiB/query/owner. Gathered candidates occupy 16 KiB/query at degree four or 32 KiB/query at degree eight. At 120 rows this is 1.875 or 3.75 MiB per participant if AllGather is used, before local selection, merge, communication and output scratch. A four-head width-128 FP16 query is 1 KiB/row; an FP32 query is 2 KiB. Inspect where queries already reside before charging or adding a broadcast. Gather-to-coordinator is a separate schedule comparison, not an assumed saving.

Begin with one fixed, qualified sharding scheme and a complete worst-rank ledger; retain a single physical history, bounded raw pending state and matching checkpoint/restore ownership. This changes the initial canonical-owner contract and needs separate architecture qualification. Do not silently turn it into live degree changes or remote scans of the same canonical bank.

**First test / stop condition.** Compare actual C1 and C24 critical paths at native histories, then every required count. At C24, independent canonical owners may already balance the box; sharding can add communication without reducing total work. Stop if that existing balance wins, gather/merge dominates, capacity falls, or migration would be needed before a fixed layout qualifies.

### E11 Shared-prefix decode computation

**Addition.** Prefix caching already avoids repeated prefill. Hydragen/Cascade instead reuse the attention computation's key reads across concurrent queries sharing actual prefix storage. This is a conditional decode optimization, not another cache-hit policy. [S14]; [S15]

Group queries referring to the same immutable canonical prefix pages. For QSA, preserve each row's own selected groups and visibility; shared prefix identity does not imply shared selections. Evaluate shared selected tiles with row-specific masks, and evaluate unique suffix/tail work separately. Compatible pooled-key scoring can likewise batch current queries against identical prefix keys.

Merge partial softmax maximum, denominator and unnormalized value sums in the existing qualified order/precision. Do not copy a subroutine that downcasts normalized partial outputs and then claim full FP32 equivalence. Charge masks, grouping, scratch and extra passes; use one authoritative page object.

**First test / stop condition.** Test shared selected-page intersections, not just common prompt lengths. Compare 0/25/50/90% shared-prefix traffic, branch/restore/COW cases and per-row masks. No added batch delay or warm-request FIFO jump. Stop when selection overlap is insufficient or grouping costs exceed saved reads. All-unique full-context capacity/throughput remains a separate mandatory control.

### A08/A10/A21 Native register, launch and tail evidence

1Cat issue #1132 includes a two-file D128 dense/paged-attention patch that retains running P×V output in WMMA accumulators and lifts a minimum-two-block launch hint when shared memory already permits only one block. Either change alone need not win: register accumulation under the old cap increases spills. The author reports bit-identical operator cases and 8–12% better long-prompt prefill in a single-GPU Qwen3-8B setup, with no gain on its shortest prompt. The retained test run also has 80 unchanged environment/route failures; it is not a wholly passing qualification. This supplies an additional A21 register/resource implementation to compare, not a speedup established for ProviderOne's different QSA geometry. Inspect registers, spills, shared memory and the complete reduction/cast sequence together.

Issue #441 and its follow-ups distinguish a merged narrow pre-Ampere QSA prefill dispatch from separate D128 tile and compact grouped-NVFP4 proposals. The original thread's claim about the live amd/nvidia file was corrected later; follow current dispatch. A D128 tile microbenchmark wins while its 27B end-to-end test is neutral. A later grouped-NVFP4 report uses a compact static group grid, byte-equal in-place fragment packing and weight reuse across up to eight rows. These reinforce A08/A10/A21; the described K-multiple-of-64 requirement does not cover ProviderOne's TP8 K80 down shard. Synthetic NVFP4 bytes, another checkpoint or a report without a complete portable patch does not qualify the selected IQ formats.

Sources: [register/launch patch #1132](https://github.com/1CatAI/1Cat-vLLM/issues/1132), [pre-Ampere discussion and corrected follow-ups #441](https://github.com/1CatAI/1Cat-vLLM/issues/441).

## 6 GDN execution, replay and state representation

### A22 Native GDN interfaces, metadata and head geometry

The bundled **FlashQLA SM70** code has original TileLang prefill, native CUDA indexed-state paths, direct output aliases and dedicated decode/verification interfaces. 1Cat separately tunes prefill thread counts and avoids repeated metadata construction across compatible GDN groups. These are concrete alternatives to an unoptimized recurrence implementation.

Its original-prefill 256-thread screen admits FP16 Q/K/V, FP32 state/gate/beta, local H4/HV12, dimensions 128 and 4096–8192 chunk rows. It reports 8192-row indexed-state time 1.749 to 1.127 ms, with exact measured outputs/state. The default stays 128 threads; the subsequent 5941.04 prompt tokens/s result combines earlier HC changes and is not an isolated GDN speedup.

At TP8, ProviderOne's per-rank GDN head geometry differs from TP4. Validate the actual imported head order and state layout before porting a shape-specialized kernel. Kernel availability under a matching Python name is not evidence that this route executes.

HC is also a real target. 1Cat fuses communication, normalization and the four-stream down/up/mixing chain. Its full-mesh HCX reports contain both passing operator screens and model comparisons that fail teacher-distribution gates. Several more aggressive polling, exchange and fusion schemes lose to the installed chain. Keep one measured implementation and explicit graph ownership; do not introduce a second coordination framework from these exploratory reports.

Sources: [GDN bindings](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/vllm/model_executor/layers/fla/ops/sm70/gdn_prefill.py), [native FlashQLA interface](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/flash_qla/ops/gated_delta_rule/chunk/sm70/fused_fwd.py), [prefill thread qualification](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_gdn_prefill_threads.md), [HCX results and rejected screens](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/flashnext_hcx_local_schedule.md).

### A22 Producer strides, direct outputs and convolution state

GDN additionally benefits from accepting the producer's feature-contiguous row stride, writing caller-owned output directly, sharing compatible per-step metadata and fusing convolution/gating/initialization only once. Preserve logical convolution length independently of physical history capacity. The dedicated TP4 H4/HV12 value-tile schedules are not ProviderOne TP8 H2/HV6 schedules. Default-only tests miss mixed batches and large expert-count routes.

Sources: [row-strided GDN](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_mtp_strided_qkv.md), [direct output](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_mtp_gdn_output.md), [logical convolution regression](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_gdn_preprocess.md), [GDN range coverage](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_dflash2_m48_scaling_20260927.md).

### E03 Share the raw GDN Gram across value-head siblings

**Addition.** The model has three value heads per Q/K head; TP8 has H2/HV6. In a route that computes raw K K-transpose before value-head beta/gates, those three siblings can share identical raw FP32 Gram results. Existing head-layout and metadata-reuse requirements do not specify sharing this arithmetic.

The inspected FLA experiment launches this work per value head, maps each to its K head, computes raw Gram, then applies distinct gates. This is evidence of a transferable redundancy, not proof ProviderOne's executed kernel contains it. [S06]

Organize bounded work per K head/chunk: compute Gram once, then perform three separate value-head coupling transforms, solves, WY outputs and state updates. Preserve imported head order. Distinct beta/decay values prohibit sharing their gated matrices, inverses or states.

At chunk 64, a full FP32 Gram is 16 KiB/K head; its lower-triangle payload is 8,320 B. Prefer shared/register reuse and streamed sibling consumers. Retaining every prompt Gram or all sibling solves at once can defeat the memory/occupancy benefit.

**Numerical boundary.** Sharing can retain raw FP32 bits only if operands, dot sequence and cast points match. If the selected route first rounds beta-scaled K into FP16, factoring beta out is a different numerical computation.

**First test / stop condition.** Inspect the loaded route first. Compare same-input three-sibling work with deliberately different gates, real H2/HV6 order, tails and initial states. Measure complete GDN stages, scratch, spills and long-state behavior. Stop if the work is already shared or added Gram traffic/resource pressure exceeds recomputation.

### E04 GDN similarity transform and fused inference WY

**Addition.** Move separable cumulative gate factors outside pairwise coupling/triangular-solve work, then fuse compatible inference intermediates where profitable. For a gated matrix N related to ungated M by diagonal similarity,

\[
N=G M G^{-1},\qquad N^{-1}=G M^{-1}G^{-1}.
\]

The derivation and experimental implementation are primary sources; their timing is not V100 evidence. FLA PR797 is closed and unmerged at inspected head 7004d193f9e26aa706fb8fa9b85022e80eec04c1. An earlier redundant-inverse removal is already merged; it is not this candidate. [S04]; [S05]; [S06]

Derive the actual transformation from the pinned Qwen equations and casts, including its softplus decay, beta and FP32 state. Do not substitute a blog's illustrative gate or materialize tiny G/huge inverse factors in FP16. Use a stable representation of required gate differences/scales and restore every W/U factor correctly.

The inspected fused experiment does not supply ProviderOne's H2/HV6 mapping, and its zero-filled tail lanes need correct masking in gate reductions. Handle underflowed/zero cumulative scales under the reference semantics; never form an inverse of a represented zero scale. A finite log-domain formulation needs its own qualification. A direct port is therefore insufficient. Native CUDA/WMMA and ordinary FP32 SIMT solve work remain possible; BF16/TF32 Tensor Core instructions are unavailable.

**First test / stop condition.** Compare chunks 16/32/64 at actual local geometry. Include initial state, extreme decay, partial chunks, convolution, long recurrence and accepted-prefix recovery. Count complete GDN/prefill work and peak storage. Stop if changed numerics fail qualification, fusion loses useful parallelism, or reduced exponent work merely exposes a larger bottleneck.

### A41 Accepted-prefix recurrent replay

ReVolta's optional GDN path retains two running/checkpoint state blocks instead of nine for seven proposals. Verification retains bounded per-layer inputs without storing every intermediate state. After sampling, the same verifier schedule replays those inputs, writing only the accepted running state and any crossed retained prefix boundary. This is a concrete implementation of ProviderOne Design Section 12's existing checkpoint/replay option, with a materially different memory/compute tradeoff from per-proposal state selection. It does not replay the entire target model or resample proposals.

The fork reports GPU token-pool capacity rising from 1,379,634 to 1,542,558 (+11.8%) with replay in its stated greedy configuration. It also reports approximately 0.8 ms/step replay cost with probabilistic drafting, and 1–4% slower greedy throughput at one to four requests. Replay is default-off. Those measurements cannot waive ProviderOne's decode gates or be converted directly into additional full Q8 sessions.

Qualify stored inputs, baseline state, request/slot mapping, accepted counts after stops, every retained boundary, mixed/padded/partial steps, cancellation and convolution/PLE coordination. Charge input buffers, output scratch, state versions and replay graphs. The reference lazily captures replay graphs on a later live step, adding a reported 35–90 ms once per shape; ProviderOne must prepare bounded qualified captures through its existing resource protocol before relying on their latency/capacity. Do not import unbounded live capture or reuse graphs after recorded inputs/storage change. A capacity benefit qualifies only with complete state parity and the existing performance contract.

Sources: [pinned guide and measured tradeoffs](https://github.com/Leonccaa/ReVolta/blob/b7a6fe8bb18ecb77b4dbbc4273ed4171a8613973/docs/dflash2.md), [feature capture](https://github.com/Leonccaa/ReVolta/blob/b7a6fe8bb18ecb77b4dbbc4273ed4171a8613973/vllm/models/qwen4_exp/nvidia/model.py), [hybrid cache ownership](https://github.com/Leonccaa/ReVolta/blob/b7a6fe8bb18ecb77b4dbbc4273ed4171a8613973/vllm/v1/core/kv_cache_utils.py), [accepted-state replay and graph keys](https://github.com/Leonccaa/ReVolta/blob/b7a6fe8bb18ecb77b4dbbc4273ed4171a8613973/vllm/v1/worker/gpu/gdn_spec_replay.py), [recorded-input/replay kernel caller](https://github.com/Leonccaa/ReVolta/blob/b7a6fe8bb18ecb77b4dbbc4273ed4171a8613973/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py), [replay regression cases](https://github.com/Leonccaa/ReVolta/blob/b7a6fe8bb18ecb77b4dbbc4273ed4171a8613973/tests/v1/worker/test_gdn_spec_state_replay.py).

The ReVolta model/profile and method comparisons are retained with A37 in [§8](#8-vocabulary-sampling-and-speculative-methods); its replay measurements above use that separate source configuration.

### E21 Deferred GDN state materialization across decode rounds

**Addition.** A41 saves snapshots by replaying retained verifier inputs after acceptance. ReplaySSM goes further: keep one FP32 checkpoint and a short buffer of corrections/keys/decays, compute outputs without writing a full state on most steps, and materialize only at bounded flush boundaries. Speculative output evaluation can reuse one checkpoint read across its query rows. This changes the recurring dataflow, rather than only the recovery procedure. [S37]; [S38]

Start with the required chain-shaped MTP path. TreeWY offers a related accepted-state reconstruction alternative; its paper reports better chain performance from ReplaySSM and does not establish a throughput advantage for branching trees. It is a reference for the representation, not a reason to add a tree method. [S39]

**ProviderOne experiment.** Use native SM70 kernels, the actual TP8 H2/HV6 mapping, imported head order and existing gate/normalization casts. Keep the checkpoint and initially the cached correction vectors in FP32. The paper's FP16 correction cache and the inspected speculative wrapper's TF32 default are independent numerical choices, not a ready V100 port. No low-precision checkpoint/read shadow is part of this candidate.

For a logical budget L of committed records and at most T new consumed verifier rows, reserve at least L+T records and choose L≥T. Before a step with h+T>L, fold only the h already committed records into the checkpoint; tentative records remain outside it. Commit only the actual target-consumed accepted prefix, including the engine's anchor/correction convention, after stops resolve. This permits bounded pointer-based rejection without inverting a rounded state.

Derive cursor invariants from ProviderOne's own row definitions. The inspected source names its logical threshold differently from the paper and rounds its physical ring to a power of two; copying one formula into the other convention can overwrite live records. Use per-row device cursors/flush flags, fixed bounded launches and ordinary lifetime dependencies. Test wrap, partial acceptance and flush on the same step. No host acceptance readback is needed.

**Scale and cost.** Under TP8×PP1, one target recurrent-state version is 36×6×128×128×4 = 13.5 MiB/request/rank, or 324 MiB/rank at C24. This is a geometry calculation, not additional memory to add to the current ledger. With L16/T5 and a 32-record physical ring, shared FP16 keys plus FP32 corrections alone occupy 94.5 MiB/rank at C24; add decays, cursors, padding, convolution and execution scratch. Removing some existing versions can save capacity, but only the reconciled net allocation proves that.

Preserve coherent prefix checkpoints and suspension. Either materialize the exact requested boundary through a qualified path, or serialize checkpoint, committed ring, cursors and complete dependent state under a versioned identity. Rejected records cannot enter a retained prefix. Immutable prefix readers need a private writable continuation; flushing cannot mutate a leased checkpoint. Keep the existing convolution/PLE and draft recovery rules. Enabling/disabling this representation needs an in-process, budgeted boundary conversion.

**First test / stop condition.** Compare snapshots, A41 replay, per-round accepted-state reconstruction and deferred flushing on real long-state traces. Sweep small windows, including every supported T and mixed per-row cursors; include flush cost and p99 step/service gaps. A tentative refinement is to stagger optional early flushes of committed records to avoid synchronized flush bursts, while retaining the hard space invariant. Accept it only if total work and latency improve.

Require long-context teacher/task/acceptance qualification: algebraic equivalence does not preserve the original FP32 rounding path, and caching corrections in FP32 alone does not prove parity. Include extreme/zero represented decays and cumulative underflow; never divide by a zero cumulative scale. Stop if numerical error, correction-cache traffic, register pressure, flush peaks or C1/2/4/8/16/24 performance defeats the complete benefit. H100/B300/B200 evidence supplies no V100 multiplier.

### E24 Bit-preserving encoding of active FP32 GDN checkpoints

**Origin and addition.** Scientific lossless codecs, including ndzip-gpu's Volta work, show that GPU bit packing can trade cheap integer work for fewer transferred floating-point bytes. Test that trade inside the recurrent-state read path. E12 concerns static weights and E13 idle backing; this proposal targets the active FP32 GDN checkpoint, particularly its repeated reads under E21. The adaptation and format sketch here are ProviderOne synthesis, not an ndzip result on model state. [S44]

Begin with an actual state-bit census. A simple candidate uses independent 256-word blocks and a small mode table: raw words, common exponent, or a narrow losslessly coded exponent range. Preserve every sign and mantissa bit and reconstruct all 32 bits before the unchanged arithmetic. Store whatever base/escape information the selected mode needs. Unusual exponents, signed zero, subnormals and nonfinite payloads must round-trip; raw mode is always available. Do not round values or assume state resembles a smooth scientific grid.

Reserve a full 1,024-byte payload slot for every block plus worst-case mode/metadata space. Read only the populated sectors in compressed modes; decode into bounded register/shared tiles directly for the GDN consumer. Writes encode the exact produced FP32 words. Publish payload/modes through the existing state transaction and protect shared mode-word updates between producer CTAs. No persistent decoded state mirror, full-state expansion pass, CPU boundary or dependence on future compressibility is permitted.

Under TP8×PP1, a two-bit mode table for one target state version at C24 has 81 KiB/rank of logical payload before alignment and other metadata. This small overhead still belongs in admission. Fixed raw-capacity slots mean **no guaranteed active-capacity increase**; the proposed benefit is less HBM traffic. Count actual 32-byte sectors, repeated header loads, scratch and retirement, not ideal bit counts.

**First test / stop condition.** Capture representative state words under the selected target, nonzero long-context histories, media and every activity point. Estimate net sectors saved, then prototype one fused reader/writer against raw FP32. Compare eager/raw, deferred/raw and deferred/encoded paths; measure complete flush cycles, spills and whole-round throughput. Test bit-for-bit codec round trips independently of E21's separate arithmetic qualification.

Stop at the census if state entropy is unfavorable, or after the reader prototype if integer/decode work and lost coalescing exceed saved traffic. Any mode that needs more than its raw slot falls back before publication. Moving spare integer work beside FP32/Tensor Core work is an opportunity to measure, not a claim that execution resources or bandwidth are free.

## 7 PLE preparation and lookup

### A25 Packed-table, mapped and queued-I/O transports

The design already recognizes PLE as a possible low-load bottleneck. These projects provide concrete alternatives worth measuring under this server's RAM, NUMA and shared-PCIe limits.

| Mechanism | Relevant evidence | ProviderOne disposition |
|---|---|---|
| Strata bounded page cache and queued direct I/O | PleReader deduplicates covering pages, handles cross-page rows and sends work to an I/O thread; its DirectFile path supports queued io_uring/O_DIRECT. | Benchmark against the existing mapped-table page-cache path under actual cold/warm row access; direct I/O is not automatically faster than warm RAM. |
| 1Cat pinned packed-table GPU lookup | Rank-local pinned IQ4_NL storage; GPU gathers/decodes requested rows during decode without CPU request/flag exchange. Reported C1 round 22.501 to 18.615 ms and C4 53.927 to 46.072 under its TP4/C1–C4 contract. | High-value decode candidate if locked-RAM, locality and 24-request PCIe service fit. This is intentional host PLE access, not offloading active attention. Requalify target numerics and concurrency; natural long outputs differed in the report. |
| CPU fetch of packed rows followed by GPU decoding | 1440 packed bytes/token versus 5120 FP16 bytes/token for the report's TP4 local shape. Reported round savings accompany a failed model-distribution gate. | Keep as a candidate requiring correction and model requalification, not a qualified speedup. |

For the pinned-table route, the table remains packed and is partitioned across workers; it need not become a full FP16 table or a full copy per GPU. The source route is restricted to TP4 and at most four scheduled requests. ProviderOne must qualify TP8/24, actual row ownership, common-result distribution and contention with media and checkpoint transfers.

Sources: [Strata PLE reader](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/src/ngram/ple_reader.cpp), [pinned packed PLE](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/gguf_pinned_ple_decode.md), [packed result experiment and failed gate](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/gguf_packed_ple_results.md), [mapped row gather](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/ple_mapped_row_gather.md).

### A23–A26 Input fusion, direct convolution and prompt lookahead

Evaluate the following preparation and lifetime mechanisms:

1. Fuse integer ngram input gathering, EOS reset, history masking, wrapping hashes and request boundaries. Reuse device request-index metadata and pinned asynchronous transfers rather than copying the same small index list twice.
2. Read prefill convolution taps directly from current rows plus the nine-row history instead of building a padded history tensor. Preserve width four/dilation three, SiLU and the final state. A channel-chunked fallback prevents concurrent wide short-convolution temporaries from exploding.
3. For known prompt-only rows, send CPU PLE work at scheduling time so its token request is not trapped behind the previous GPU forward. Release mapped result storage after the actual consumer copy retires, rather than always after the whole model. Graph replay and mixed/speculative batches need separate paths.
4. Bound mapped-table readahead and copy in address order under RAM pressure. Advising every missing page far ahead can have those pages reclaimed before consumption, causing repeated major faults.

The direct-convolution report's M7344 complete stage falls from 7.827 to 0.941 ms and temporary memory from 287.7 to 143.4 MiB under its TP4 scope. This is an operator/state-chain result, not a ProviderOne model gain. Early PLE requests can remove a pipeline bubble yet leave overall prefill unchanged when a later stage is critical. Bounded-readahead results depend on cold file cache, cgroup pressure and storage; warm-memory lookup need not improve.

Sources: [input preparation](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_ple_input_prepare.md), [integer ngram fusion](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_ple_ngram_batch.md), [metadata reuse](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/ple_metadata_request_indices.md), [convolution/worklog](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/benchmarks/flashnext_prefill/OPTIMIZATION_20261014.md), [early request #1217](https://github.com/1CatAI/1Cat-vLLM/pull/1217), [bounded readahead #1235](https://github.com/1CatAI/1Cat-vLLM/pull/1235).

### E19 Actual-draft-token PLE lookahead

**Origin and addition.** A26 starts known prompt PLE work early. Extend the lookup lead window using actual MTP draft tokens: MTP disables PLE, while target verification still needs it. As sampled proposal IDs become available, derive the exact target n-gram addresses on private tentative state and prefetch immutable rows while later proposals are drafted. Engram supplies a deterministic asynchronous-lookup precedent; this specific MTP/target overlap is a ProviderOne synthesis. Its measurements on other models/hardware do not qualify this server. [S29]

Keep the target's own hashes, offsets, signed wrapping/modulo, resets, source decode and expanded-token/media semantics. Do not change proposal draws/q, invent authoritative tokens or advance committed PLE hash/convolution state. Notify only the already selected verification prefix. Correction/bonus rows become eligible only when their actual token IDs exist.

Rows in the selected proposal block are normally processed during target verification even when those proposals are subsequently rejected. Ordinary rejection therefore does not make those lookups wasted; cancellation, truncation or never-verified rows can. Distinguish that waste in measurements. Retire all asynchronous readers before discarding or reusing notices/staging.

Use timely existing ID transfers where possible, or bounded pinned notices and stream/event dependencies without a global synchronization. Compare host hashing of compact token IDs with device hashing plus address transfer. Coalesce reads through one bounded lookup/I/O owner; replace inherited ownership where it impedes the qualified path. Do not split an efficient monolithic graph merely to expose expensive per-token notifications. MTP is the first useful scope; a parallel DFlash call may offer too little early visibility.

**First test / stop condition.** Trace token-ID-ready, address-ready, row-ready and target-PLE-consumer times. Charge CPU wakeups, graph effects, I/O, pinning, H2D and full-concurrency interference under warm/cold memory/storage. The consuming path still joins a required read if a prefetch is late. Stop if PLE is not critical, no useful lead exists, notices cost more than overlap, or the bounded pipeline weakens capacity/progress.

## 8 Vocabulary sampling and speculative methods

### A37 Selected Inco draft range and correctness boundary

**Unmerged Strata PR #68**, pinned head **baede60aa1d8ccf5a7b8ed83822da4f1229f92ef**. It implements the exact Inco revision selected by ProviderOne, a parallel five-layer block, target HC feature taps, bounded ring attention and realized-predecessor conditional proposal probabilities.

Its source contains a focused overflow reproducer: FP16 matrix inputs with FP32 accumulation can still produce a projection output above FP16 range. Storing that output/residual/KV in FP16 creates infinities/NaNs before a later normalization that would otherwise remain finite. Saturating/clamping changes channel ratios and fails the fixture. The branch consequently retains FP32 projection outputs, other activations/residuals and draft KV, while using FP16 matrix operands. The report notes real-checkpoint dynamic-range problems; we have not reproduced them in ProviderOne.

**Design implication:** Design Section 14's selected FP16 draft weights and especially its FP16 KV/5-MiB-per-rank window are a candidate profile, not a proven safe representation. Add finite-range probes and real-checkpoint, real-SC117-feature parity before fixing those storage choices. If FP16 storage fails, qualify the necessary FP32 outputs/KV or another independently validated representation, preserving the source's logical equations. Recompute every affected activation/graph/rollback/allocation term. An FP32 KV window has twice the FP16 payload before guards; that is an accounting consequence, not a new measured capacity.

This does not establish a ProviderOne defect or justify replacing its FP16 matrix operands with a slow whole-model FP32 implementation. It identifies a concrete precision boundary that needs testing. Correct proposal probabilities and target sampling remain mandatory even when draft arithmetic changes.

The same branch is also **negative performance evidence**: across three complete 128-request dataset pairs, acceptance increases 2.95–6.15%, while request throughput decreases 1.45–6.92%. In six coding prompts, paired geometric-mean decode decreases 4.14%. It lacks concurrent/batch support and pipeline-2 integration; cached-prefix missing features can reduce early acceptance. Those limitations conflict with ProviderOne's full-prefix/continuation conditioning and 24-active contract. Borrow its range/rejection/ring tests and projection fusion, not its incomplete serving architecture.

Sources: [PR #68](https://github.com/jmnargi/Strata-V100/pull/68), [exact report](https://github.com/jmnargi/Strata-V100/blob/baede60aa1d8ccf5a7b8ed83822da4f1229f92ef/benchmarks/v100-iq3_s-dflash2-2026-10-20.md), [overflow reproducer](https://github.com/jmnargi/Strata-V100/blob/baede60aa1d8ccf5a7b8ed83822da4f1229f92ef/tests/core/dflash2_overflow_test.cu), [native draft](https://github.com/jmnargi/Strata-V100/blob/baede60aa1d8ccf5a7b8ed83822da4f1229f92ef/src/core/dflash2.cu), [limits](https://github.com/jmnargi/Strata-V100/blob/baede60aa1d8ccf5a7b8ed83822da4f1229f92ef/docs/DFLASH2.md).

### A37 ReVolta integration and method comparison

Open [issue #1302](https://github.com/1CatAI/1Cat-vLLM/issues/1302) links a Flash-Next/Inco implementation in **ReVolta**, inspected here at `b7a6fe8bb18ecb77b4dbbc4273ed4171a8613973`. It uses the same Inco checkpoint revision, completed-HC mean features with an explicit admission declaration, bounded QSA/drafter rings, hybrid cache-group integration, actual draft-dtype block alignment and batched verifier admission. Unlike the Strata branch, it reports concurrent operation. These are useful A37 integration/test references; they are outside 1Cat's inspected main.

The reference profile uses TP4/DCP2, an eight-token block/seven proposals, a 512K allowance, FP16 draft KV, other target weights/KV and optional CPU KV offload. Its replicated draft-cache placement under DCP is not ProviderOne's TP8 placement proof. Raising a config position limit does not independently establish long-position quality. Its use of FP16 KV also does not resolve Strata's output-range finding: compare feature reduction, cast/output boundaries, finite values and actual checkpoint paths directly.

The reported one-request English code/reasoning gains do not persist uniformly: its early mixed C4 result is 238 versus 236 tokens/s for MTP4, while C8 is 331 versus 357. Real sampled agent replay is about 0.97× MTP4, with a reported 95% interval of 0.90–1.01. MTP4 remains its default. Later active-count width ladders verify fewer proposals at selected counts while still generating all seven; this is a concrete A06/A18 policy reference, with possible wasted draft work. ProviderOne must calibrate its own legal widths, proposal accounting, conditioning and graph/resource envelopes at C1/2/4/8/16/24.

Its accepted-state replay mechanism, measured costs and pinned implementation links are in [§6](#6-gdn-execution-replay-and-state-representation).

### A16 Shard the Inco selector codebooks

Open [1Cat PR #1301](https://github.com/1CatAI/1Cat-vLLM/pull/1301), head d8ed559e21705566f983c58dc5de0f361b63f3ac, vocabulary-shards the predecessor/successor codebooks and gathers looked-up rows through zero-filled ownership plus all-reduce. It retains exact finite nonzero values; signed zero can change. Two small collectives replace full per-rank table replication. Its new-main evidence is CPU tests; reference-fork V100 results are not a measured result on the current main branch.

The dimensions match ProviderOne's selected Inco manifest: two 248320×256 FP16 tables total **254,279,680 bytes/rank** in its reference replicated placement. TP8 row sharding would retain **31,784,960 bytes/rank**, an arithmetic saving of **222,494,720 bytes = 212.1875 MiB/rank**, before padding, communication and lookup scratch. The draft-weight reference would decrease from 460,659,200 to 238,164,480 bytes/rank if all other placements stayed unchanged.

This is a genuine candidate to improve capacity without changing target weights or KV quantization. Qualify its actual conditional-selector access pattern at C1..24; do not assume the lookup collectives are free. Release the full tables rather than keeping both representations. Report post-import allocation, not just the source-byte subtraction.

### A17 Exact compact sampling, transport and logprob budgets

Packed candidate all-gather can carry token IDs and original FP16/FP32 value bits in one int32 message stream. 1Cat measures roughly 5–8 microseconds saved in qualified TP2/TP4 operator cases. This is lossless communication packing, not score quantization. Requalify vocabulary offsets, sentinel IDs, ties, NaN payloads, signed zero and TP8 graph teardown.

The inspected compact top-k/top-p repair classifies no-filter, top-k-only and oversized rows on GPU rather than rejecting an entire mixed batch with a device-to-host Boolean. An exact dense reference still handles difficult support/tie cases. The pinned implementation **still contains a remaining host synchronization** for fallback classification; it is not fully sync-free. A changed probability distribution, omitted support or approximate reranker cannot be advertised as an exact optimization.

Budget requested prompt/output logprobs separately. An open prompt-logprob repair shows how a nominal 1024-row chunk can create more than a gigabyte of temporary per-rank vocabulary tensors absent from startup profiling. Chunk by an actual admitted byte budget, release prior temporaries, and avoid a full-vocabulary request widening unrelated requests. Full logprobs inherently produce large output; do not promise compact transfer for that API case.

Sources: [packed candidate transport](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_compact_candidate_gather.md), [row classification #1267](https://github.com/1CatAI/1Cat-vLLM/pull/1267), [current sampler](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/vllm/v1/sample/ops/topk_topp_triton.py), [bounded prompt logprobs #1225](https://github.com/1CatAI/1Cat-vLLM/pull/1225), [sampling contract regressions](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_mrv2_sampling_contract_repairs.md).

### A18 Separately qualified proposal and numerical policies

Reduced-vocabulary MTP, ngram/lookup augmentation, DDTree, deeper drafts and alternative selector orders remain separate method candidates. Exact rejection can accommodate a changed normalized proposal q, but acceptance, support, constraints, conditioning maintenance and whole-round cost must all qualify. They cannot replace required full-contract MTP or Inco DFlash implementation by default. Open candidate-head work also records a slower C8 route despite good filtered-distribution probes.

Sources: [reduced vocabulary #1218](https://github.com/1CatAI/1Cat-vLLM/pull/1218), [lookup probability repair](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_dflash2_fastpath_numerics.md), [candidate-head #953](https://github.com/1CatAI/1Cat-vLLM/pull/953), [lookup method](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_dflash2_lookup_augmented.md).

**MTP selection reuse changes the drafter.** 1Cat exposes reuse of MTP step-0 sparse indices during later steps and shares scratch whose lifetimes do not overlap. ProviderOne's governing initial policy explicitly recomputes MTP QSA selection. Scratch sharing can be adopted after a lifetime proof; selection reuse requires separate method/acceptance/proposal-probability qualification and must not become an accidental default.

**Deeper drafting is conditional.** v100-skinny demonstrates that an eight-row Volta tile can make a wider native-MTP verifier economical on its dense 27B workload. It does not make seven proposals optimal for Flash-Next or 24 active requests. ProviderOne's adaptive proposal policy is appropriate; calibrate useful committed tokens per complete round and every larger execution envelope.

Sources: [MTP implementation](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/vllm/models/qwen4_exp/nvidia/mtp.py#L503), [v100-skinny](https://github.com/dnv2003/v100-skinny).

### E07 Grammar-legal-support vocabulary projection

**Origin and addition.** Combine exact grammar/tokenizer state with indexed original-weight head projection. Compute only vocabulary rows that can have nonzero probability under the already selected target policy. This differs from compact sampling after full logits and from approximate/reduced vocabularies. XGrammar supplies grammar/mask/lifecycle primitives; it does not demonstrate this proposed restricted head. [S09]

**Essential eligibility condition.** The exact hard support mask must precede any support-dependent top-k/top-p or other operation that could depend on excluded scores, or omitted scores must otherwise be proven irrelevant. If truncation precedes masking, computing only legal logits can change the distribution or hide empty-support behavior.

Use the existing vocabulary shards and packed source rows, with bounded legal-ID lists. Apply biases, penalties, temperature and truncation in their declared order. Preserve every legal token. For speculative rows, fork grammar state along the actual candidate prefixes and preserve stop/error handling.

A legal singleton removes head work only if the API requests the final conditioned distribution. Raw logits still require the requested dot-product scores; unmasked logprobs additionally require the full normalizer. Skipping omitted logits, or the legal singleton’s own projection, also requires a qualified finite/range and error-behavior proof: a unique legal ID alone cannot hide a nonfinite legal score or reference validation of masked scores. Where that behavior is not proven irrelevant, use the ordinary qualified head. Preserve required RNG, state, conditioning and response semantics. Broad-support requests use the ordinary qualified head plan. A speculative row following an illegal proposal is not a valid conditioned prefix; preserve reference rejection/error behavior and never normalize empty support.

**Cost.** The 248,320-token vocabulary needs 31,040 B/bitmask; 120 masks cost 3,724,800 B before ownership/padding. Count mask construction/transfer, compaction, indexed quantized-row reads and lost dense-MMA reuse. Avoid per-token CPU/GPU ping-pong; immutable grammar work may overlap transformer execution.

**First test / stop condition.** Compare support sizes 1/8/32/256/4096/all on real JSON/tool schemas and mixed free text. Require identical legal support and qualified logits/probabilities, including overflow/nonfinite and empty-support/error cases. Stop if support is usually broad or mask/index overhead exceeds complete-round savings.

### E22 Token-exact forced spans as bounded teacher-forced chunks

**Origin and addition.** Compressed-FSM jump-forward decoding replaces multiple serial generation rounds with a short extend operation. The published SGLang route operates on forced strings and retokenizes text; its paper explicitly discusses distorted probabilities. ProviderOne should evaluate a narrower variant: only a run whose next **token ID** is uniquely determined at every step under the actual final sampling policy. This extends E07's singleton head saving to the transformer execution schedule. [S40]; [S41]

Advance a private grammar/request-state copy through the unique-token path, ending before branching, empty support, EOS/stop, a reply/context limit or the admitted chunk cap. A unique character/string path is insufficient: multiple tokenizations, tokens crossing a grammar boundary and UTF-8 byte fallbacks can remain valid. Never retokenize already committed token IDs, rewrite the prompt or choose a convenient tokenization.

Eligibility inherits E07's sampler-order, finite-range and error-behavior conditions. Check minimum-token/EOS rules, hard masks, penalties and all request-dependent constraints at each future prefix; a grammar singleton does not override an empty or invalid final support. Preserve the reference's per-sequence RNG advancement even where the outcome is predetermined.

Execute the known token sequence through a bounded causal chunk, producing all required KV, pooled groups/tails, GDN, convolution, PLE, positions and selected-method conditioning. Existing native prefill kernels can turn repeated small projections/collectives into useful multirow work; no model layers disappear. Keep the precise target-consumed versus generated frontier, and compute the terminal hidden/logit state needed for the next branching choice. Future complete groups must remain invisible to earlier queries.

Use the existing active slot, row budget, graph families and commit path. No request termination/re-enqueue, new endpoint, FIFO jump or additional speculative model is needed. The selected MTP/DFlash state must remain valid or receive its existing qualified catch-up; skipped proposal work is not permission to lose full-prefix conditioning.

Requested raw logits/unmasked logprobs still require their scores/normalizers; disable omitted-head work where those obligations cannot be met. Bound chunk duration for other requests' progress, cancellation and streaming. Publish only the committed prefix; never charge forced output as new user-input prefill or fabricate zero-cost decode.

**First test / stop condition.** Census actual consecutive singleton-token runs in JSON/tool workloads, then compare lengths 2/4/8 within the resource envelope against sequential E07 and ordinary speculative execution. Require exact token/stop/accounting behavior and qualified chunk-versus-recurrent state/logits. Stop if token-level runs are rare, model-state maintenance dominates, or batching/catch-up delays other sessions enough to erase the gain.

### E23 Certified original-weight vocabulary-head pruning

**Origin and addition.** Exact maximum-inner-product search uses geometric upper bounds to avoid reading vectors that cannot win. Apply that principle to the unchanged 248,320-row output head: a cheap incumbent search plus conservative bucket bounds may avoid whole original-weight row blocks. This is target-head computation avoidance, distinct from compact sampling after full logits, E07's hard legal support and A18's changed draft vocabulary. [S42]

Start with greedy or a fixed finite top-k policy. Partition original decoded operand rows into bounded buckets, retaining the sole packed head. For a bucket B with center c and certified radius r≥max_i∈B ||w_i−c||₂, the real-arithmetic bound is

\[
\max_{i\in B} h^\top w_i \le h^\top c+r\|h\|_2 .
\]

Cover the reference packed decoder, operand casts and GEMM/reduction rounding with an outward error enclosure. Apply valid upper bounds for every bias/penalty/mask/temperature transformation used before selection; token-specific transforms cannot be ignored. Score seed rows with reference arithmetic, then skip only buckets strictly below the certified kth incumbent. Resolve ties using the actual sampler contract, and score all ambiguous buckets normally. A learned or temporal shortlist can choose seeds but cannot authorize pruning.

Use owner-local exact top-k certificates followed by the existing exact global merge, or explicitly budget any global cutoff exchanges. No per-bucket collective belongs in the proposed cheap bound pass. For an eligible fixed top-k distribution, certify the complete support before normalizing or applying later top-p. General nucleus sampling, full-vocabulary normalization and requested unmasked logprobs require additional proof/work and initially use the ordinary head. Speculative target/proposal probabilities and rejection corrections must still have their exact normalized support; a greedy winner certificate does not supply stochastic p/q.

**Cost.** With balanced 31,040-row vocabulary shards and 256-row local buckets, 122 FP32 centers of width 2,560 need 1,249,280 B/rank before radii, membership, worklists and alignment. Contiguous buckets need no full ID permutation; offline clustering may tighten bounds but needs its own indexing/layout accounting. This summary is not a second full head. Do not change source quantization or normalize head rows to improve retrieval.

**First test / stop condition.** Test bound survival on recorded hidden states and actual decoded head rows before a serving kernel. Compare small greedy/top-k workloads and the full verifier row matrix against the existing packed dense head. High-dimensional radii may be too loose, and irregular sparse row evaluation can lose to dense Tensor Cores at C24. Stop if bounds rarely prune enough to repay their own reads, fallback duplication and lost reuse, or if metadata reduces the qualified integer capacity.

### E06 Exact joint block verification

**Addition.** Block Verification Accelerates Speculative Decoding changes the verification coupling across a draft block while preserving the target distribution under its stated conditional laws. This is different from adding another draft model or merely increasing draft width. Its reported wall-clock results are predominantly small-batch/short-prompt evidence. [S08]

Start with the required MTP path. Retain the actual conditional proposal probabilities, transformed target distributions and correction rule required by the proof. Perform prefix-ratio/coupling operations on device with bounded state. The Inco method needs a separate derivation against its realized-predecessor selector-conditioned q; raw pre-selector logits cannot stand in for that law.

Use the main exact block algorithm. A different greedy-block appendix variant requires its own distribution modification across rounds and is not a drop-in shortcut. Higher expected acceptance also does not guarantee lower wall-clock cost.

**Cost / numerical boundary.** Additional probability retention, normalization/correction work, stable ratio evaluation and graph storage. RNG use and seedwise outputs can change while the target distribution remains identical. State commit must still honor stops and the permitted prefix.

**First test / stop condition.** Exhaustively enumerate small-vocabulary multi-round laws, then test zero support, extreme ratios, constraints, all rejection positions and ragged replay. Measure complete rounds at all required counts. Stop if proof assumptions fail or acceptance gains fail to pay for verification and memory costs.

### E08 Pre-draw heterogeneous verifier-row allocation

**Addition.** Allocate legal per-request speculative prefixes under one measured target-row budget instead of assigning every active request the same width. TETRIS motivates the allocation problem; its optimality assumptions do not establish an optimal ProviderOne controller. [S10]

Choose widths from committed history and prior-round statistics before fresh proposal draws. Selecting lengths from newly drawn token confidence is not automatically unbiased. Give every scheduled request its required base service, then allocate optional prefix rows using acceptance-survival estimates and complete-round cost.

Use a bounded deterministic optimization, initially piloted through 24 requests and existing legal widths/graph buckets; extend the policy and qualification through the actual admitted capacity. Optimize expected committed tokens per unit complete-round time, including draft work, padding, routing, verifier and recovery. Enforce service-gap limits and preserve FIFO admission. Reserve the full resource envelope; a shorter current draft does not release another request's promised context.

**First test / stop condition.** Compare uniform and heterogeneous widths on mixed code/prose/constraint/context traces. Report all per-request service gaps, unused draft work and controller overhead. Verify the sampling/stopping-time argument and state transitions. Stop if the existing uniform plan wins after overhead, or benefit requires speculative starvation or new unbounded graph shapes.

## 9 Resource ownership, host work and communication

### A30 Scratch ownership, capacity classes and live generations

1Cat's arena audit removes decode-workspace duplication across compatible row capacities and retains graph-referenced generations safely. Its first implementation accidentally multiplied a previous long-context capacity by a later larger short-request batch. The repair groups partition capacities rather than sharing a single set of maxima. Separate device, stream, head geometry, dtype and partition layout remain necessary.

DFlash auxiliary features can be retained directly in the dtype consumed by the projection after the required original-precision additions. Direct concatenation can avoid a full larger-dtype intermediate. This is conditional on the selected feature contract, not permission to narrow arbitrary target state.

Strata's lean WMMA prefill path removes gather, intermediate and dequant-ring buffers that its direct reader never uses. Smaller actual-length loans avoid crossing a staging threshold because of rounding. The transferable requirement is **allocate only the selected path's real scratch**, without borrowing ProviderOne's guaranteed histories or resident experts. Strata's observed cache-placement-dependent speedups do not transfer directly.

Sources: [arena audit and failed intermediate](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_memory_arena.md), [Strata lean/loan measurements](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/docs/NVIDIA_V100.md), [path-specific prefill planner](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/include/strata/prefill/vram_plan.hpp).

### E14 Contiguous virtual history with stock-driver VMM

**Addition.** vAttention separates contiguous virtual addresses from physical page backing. That may simplify address arithmetic or remove a measured paging-related kernel penalty while retaining unique physical histories. Its published allocator is not a V100/QSA qualification. [S21]

Query actual VMM support and allocation granularity on the selected driver/devices. A reserved virtual range is not physical capacity or a reservation guarantee. Pre-reserve actual physical resources under the existing ledger. Bound map/unmap/access/IPC overhead and retire readers before changing mappings. [S22]

Use only a stock supported configuration for the first experiment. vAttention's smaller-page results involve a modified driver; they cannot be assumed on this stack. Derive rounding for every target/indexer/draft object, degree placement and COW boundary. A nominally contiguous view does not remove QSA's sparse selected loads or causal masks.

**First test / stop condition.** Start with a byte/granularity ledger and a minimal mapping microbenchmark. Then compare an affected packed-history kernel at identical physical residency, graph lifecycle and boundaries. Stop if existing stable arenas already remove the cost, rounding reduces session capacity, or benefit requires another allocator/driver framework or active mapping stalls.

### A32–A33 Topology collectives and HC row ownership

Open [1Cat PR #1245](https://github.com/1CatAI/1Cat-vLLM/pull/1245), head 205ac6f550a5870b7ac06dc9286ea1a16069559e, extends low-latency collectives beyond a fully connected TP4 NVLink group. It detects peer routes, generates and verifies schedules, compares complete calls with the installed path, and caches selections subject to startup requalification.

Its small-message results include an eight-GPU mixed NVLink/PCIe system. **The default uses FP16 sums/partials**. FP32 matrix accumulation does not specify collective arithmetic: preserve ProviderOne's selected collective dtype/reduction contract and qualify any change. The optional FP32-partial route is a useful comparator; it reports 9.083 versus 29.678 microseconds for its 5-KiB input comparison, including comparison conversions. At larger messages, that route wins at 384 KiB but loses to NCCL at 448 and 512 KiB. These are measured points on another system, not a universal threshold or an end-to-end token/s gain.

Compare profile-compatible variants, the FP32-partial comparator, NCCL and any existing native protocol across ProviderOne's actual payloads, rank mapping and cube links. Preserve communicator order, publication, graph reuse, cancellation, bounded IPC buffers and forward progress. No runtime compilation or unbounded schedule search belongs in active-count adaptation. Do not disable IOMMU/ACS or change host security configuration as an automatic optimization.

The narrower merged HC low-latency/sharded implementations are also useful references, but replicated HC variants and remote repeated weight reads can lose the capacity/latency tradeoff. Sequence-owned prefill residuals are a separate candidate that can reduce local HC work and temporary memory while adding reduce-scatter/all-gather. Compare the complete HC chain and numerical result rather than a projection-only microbenchmark.

Sources: [sharded HC LL](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_hc_ll_shards.md), [HC topology alternatives](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_hc_batch_topology.md), [HC row-ownership screens and negative results](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/benchmarks/flashnext_prefill/OPTIMIZATION_20261014.md).

### A32/A39 Executed collective fusion and exact metadata

1Cat uses custom SM70 push/ring collectives and fuses qualified collective-plus-norm boundaries. A concrete integration defect prevented a fusion because GGUF's FP32 epsilon became Python 9.999999974752427e-7 while a matcher required the literal 1e-6. Using the actual metadata value restored 106 replacements/rank in a dense-27B TP4 experiment, saving about 0.47 ms per full round.

The lesson is to verify executed fusion patterns and native operator routes, not round metadata to force a match. ProviderOne should inventory executed kernel/codec families, unintentional generic paths, graph replay, and fusion counts by workload. Preparation and configured capability reports alone do not prove execution.

Design Section 18's NCCL baseline remains sound. A custom TP8 collective is a separately measured alternative requiring this cube topology, every message range, graph replay, cancellation and collective-order qualification. TP4 full-mesh HCX or an all-reduce specialized for a different model cannot establish a TP8 win.

Sources: [collective/norm matcher and measured integration](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/gguf_collective_norm.md), [HCX schedule](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/flashnext_hcx_local_schedule.md), [native collective source](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/csrc/custom_all_reduce.cu).

### A27–A29/A34/A40 Host metadata, handoff, NUMA and pinning

1Cat's merged scheduler repair stops serializing the complete token history to its V2 runner after a skipped pipeline step; persistent worker state already contains it. Its measured submit cost previously rose by milliseconds at long context. ProviderOne should send deltas/frontiers and changed block descriptors, with full reconstruction only on the explicit restore/new-request path.

Strata demonstrates direct device-to-device session-slot handoff, batched synchronization for many small copies, move-owned checkpoints, and pooled checkpoint storage. Its deferred input anchor captures immutable state on GPU and copies it to host after the first output, reducing TTFT while retaining the copy cost later. Apply that only where checkpoint policy permits; protected suspension still requires complete verified backing before releasing the source or promising a freed slot.

An open eager-PP-send change overlaps a previous send with the next forward only when its immutable fresh output tensor remains owned by the send. Graph/static output buffers keep the original wait because replay overwrites them. Only the buffer-lifetime/overlap principle transfers to ProviderOne’s two-server TP8×PP2; the source’s local pipeline does not qualify RDMA, TP8 handoff or cross-node transactions. No TP4 mode is required.

Sources: [history metadata #1263](https://github.com/1CatAI/1Cat-vLLM/pull/1263), [Strata handoff and checkpoint controls](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/docs/BATCHING.md), [deferred anchor implementation](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/src/program/generate.cpp), [TTFT experiment](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/benchmarks/v100-incremental-ttft-2026-10-21.md), [eager send #1294](https://github.com/1CatAI/1Cat-vLLM/pull/1294).

Generic vLLM CPU-offload connectors and Strata's delta/prefix archives supply useful group-local accounting, byte-preserving copy and parent-object lifetime examples. They do not establish ProviderOne's protected cross-rank retained-session contract. No active-session time slicing or host attention follows from adopting a checkpoint-copy primitive.

Two issue-backed host improvements make A40 more concrete. A four-V100 C4140 report measures per-rank NUMA binding raising local host-memory placement from 72.6% to 98.2%, with about 1.9% more decode; a container-wide CPU set did not reproduce the locality improvement. ProviderOne must observe actual allocation placement and bind the relevant rank/PLE/I/O workers before allocation, using this server's topology. The percentages are that report's results, not a forecast.

The pinned main also contains an exact-size PLE registration repair: its tested Torch host allocator rounded an 11.92-GiB request to a 16-GiB pinned allocation. Registering ordinary CPU storage at its actual size removes that waste, verifies that the tensor is recognized as pinned, and unregisters only after its owner retires. Otherwise a mapped-view helper can silently create a second full pinned copy. This is a concrete allocator/lifetime candidate; do not assume every Torch version rounds identically or unpin while a GPU reader remains live.

Sources: [NUMA field report #678](https://github.com/1CatAI/1Cat-vLLM/issues/678), [host-allocation report and corroboration #1099](https://github.com/1CatAI/1Cat-vLLM/issues/1099), [current exact registration and placement](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/vllm/models/qwen4_exp/nvidia/ple_layer.py), [per-rank NUMA configuration](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/vllm/config/parallel.py).

### E13 Lossless idle-checkpoint/backing compression

**Addition.** Compress the exact serialized bytes of completed/idle retained state and reusable immutable checkpoints. Preserve KV scales/rotation, recurrent/PLE/draft state and metadata byte-for-byte. This can reduce RAM/storage occupancy or restore traffic without changing active attention.

Invariant Bit Packing studies lossless transfer compression on A100; nvCOMP documents a CUDA-12/Volta-capable library family. Individual algorithms, binary builds, scratch and asynchronous behavior still require qualification. V100 has no newer dedicated decompression engine. [S19]; [S20]

Compare NUMA-local CPU compression after D2H with bounded GPU compression before transfer. The former can save backing bytes without saving D2H traffic; the latter competes for active GPU resources. Choose based on complete contention-aware cost, not compressor throughput alone:

\[
T_\mathrm{compress}+T_\mathrm{compressed\ transfer}+T_\mathrm{restore}
+T_\mathrm{interference}<T_\mathrm{ordinary\ transfer/restore}.
\]

Keep exact codec/version/size/checksum identity. Validate backing before releasing the authoritative source; reserve compression overlap, worst-case expansion and restore staging. Compression cannot strand FIFO progress or block unrelated decode. Future incompressible state retains its worst-case backing obligation; measured average ratios do not enlarge guaranteed GPU-resident capacity.

**First test / stop condition.** Benchmark actual Q8/iso4 plus hybrid-state checkpoints, raw storage as control, warm/cold tiers, concurrent decoding and cancellation/corruption. Stop if ratios are weak or compression increases restore/queue latency enough to erase the space/traffic benefit.

## 10 Media and positional execution

### A31 Bounded positional caches and analytic RoPE

1Cat explicitly bounds MRoPE cache allocation to the permitted position envelope rather than always allocating a video-expanded table, and its MTP code reuses compatible cosine/sine tables. Strata supplies both precomputed and analytic RoPE/YaRN paths from one validated immutable configuration. Validate pair-index conventions, correction ramps, amplitude and large positions across target attention, pooled indexer and MTP; Inco's separate theta/window remains independent.

ProviderOne supports video, so it cannot adopt a video-disabled cache bound wholesale. Size tables from the maximum actual temporal/spatial/text coordinates of the declared media envelope, including proposals and pooled-group coordinates, and account for all retained dtypes. Check analytic/table numerical parity and prohibit silent clamping of a logically live position.

Sources: [MRoPE bound #1019](https://github.com/1CatAI/1Cat-vLLM/pull/1019), [native MTP YaRN validation](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/vllm/config/speculative.py), [MTP table sharing](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/vllm/models/qwen4_exp/nvidia/mtp.py), [Strata single scaling contract](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/include/strata/kernels/rope_scaling.hpp).

### A36 Exact media integration, cache identity and bounded encoding

The Qwen4Exp GGUF media work validates a companion safetensors file against exactly the active towers, uses the same language-module owner for weight/preparation paths, and bounds encoder item microbatches. ProviderOne can reuse those integration/peak-memory principles without duplicating the language model or releasing its required vision reserve.

Processor/output caching must key the actual decoder/processor configuration, expanded media identity, positions and model profile. Cross-request cache sharing cannot equate a reused URL with identical decoded bytes. Repeated image encoding and excess padded patches are useful trace targets. Generic encoder CUDA graphs are a candidate only for compatible shapes and within ProviderOne's bounded capture/vision envelope.

The broader 1Cat H3/VSA/TeaCache/Cache-DiT, LoRA, media-generation and Rust frontend work was screened. Their approximate generation caches, host weight lanes, extra model servers and unrelated architectures are not better substitutes for this target's exact language path. Generic prepared epilogues, alias validation and activation sharding are already represented in the applicable candidate families. A frontend/API rewrite may address a measured bottleneck, correctness gap or concrete inherited complexity under ProviderOne’s implementation freedom; preserve the required wire/lifecycle behavior and prove the resulting resource/performance case. A language change alone supplies no performance evidence.

Sources: [native GGUF vision handoff](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/benchmarks/flashnext_prefill/VISION_9683.md), [encoder graph design](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/cuda_graphs_multimodal.md), [shared SM70 epilogues](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/minimax_h3/SM70_EPILOGUES.md), [H3 quality limits](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/minimax_h3/GENERAL_SM70.md).

### E20 Fixed-function NVDEC and device-resident video preparation

**Capability and addition.** NVIDIA's support matrix explicitly lists Tesla V100 with one third-generation NVDEC. Eligible video can be decoded into GPU surfaces and prepared for the unchanged vision encoder, potentially removing CPU decode and full-frame RGB H2D transfers. This is a concrete input-pipeline capability beyond generic encoder microbatches and graphs. It adds no vision/model replica. [S30]; [S31]

**Bind a compatible stack.** Query cuvidGetDecoderCaps for the exact codec, chroma, bit depth and dimensions in the actual container. V100's matrix does not support AV1; a newer SDK's codec enum is not hardware support. Video Codec SDK 13.0 lists Volta, Linux driver 570+ and CUDA 11+ and is a C NVDECODE candidate for the existing R580/CUDA 12.9 stack. SDK 13.1 requires newer architectures/driver/toolkit, and current PyNvVideoCodec does not list Volta. Do not select those current bindings by name alone. Pin and qualify direct C NVDECODE or a compatible FFmpeg build plus native SM70 processing. Video Codec SDK and CUDA Toolkit version numbers describe different products. [S32]; [S33]; [S34]

Expose the required driver video libraries while preserving existing compute/utility capabilities. In the documented container environment, selecting driver capabilities replaces the default set rather than extending it automatically. Check the actual container/device configuration. [S35]

**Input and memory contract.** Preserve selected frames, presentation timestamps/order, B-frame reorder, crop/orientation, YUV range/matrix/chroma siting, bit depth, RGB conversion, resize/antialias rounding and patch order. Sparse frame sampling still requires decoding necessary unsampled reference frames. Generic GPU conversion/resizing is not automatically equivalent to the selected CPU processor; decoder/processor identity and parity must qualify before cache reuse.

Charge codec-required DPB/reference surfaces, mapped outputs, pitch/alignment, bitstream queues, conversion/patch buffers and all overlap against the fixed media envelope. Start on the existing encoder owner; distributing decode across GPUs requires accounting for peer frame transport. NVDEC is separate from inference SMs but shares memory/bandwidth/power, and CUDA preprocessing consumes SM resources. Eight GPUs do not authorize eight maximum-resolution working sets.

**First test / stop condition.** Check capability/build, temporal/pixel parity and peak surfaces first. Then measure complete media TTFT, CPU use, transferred bytes and interference with all active sessions. Stop if the new pipeline changes inputs, reduces qualified resident capacity, or merely moves an already hidden CPU cost onto contended GPU resources. Pure text decode receives no performance credit from this media-only path.

## 11 Build, runtime validity and source components

### A35 Source components and integration boundaries

| Component | Role | Integration boundary |
|---|---|---|
| 1Cat bundled TurboMind SM70 | Packed affine/LUT/lattice, grouped MoE and small-M native readers | Port relevant primitives into one ProviderOne dispatcher; preserve source-scale semantics and account replacement-format memory. |
| 1Cat bundled Flash-V100 | Volta paged attention, grouped verification and QSA-specific extensions | Dense attention support alone does not supply ProviderOne's QSA/iso4/unique-owner path. Qualify exact attention numerics and codec hooks. |
| 1Cat bundled FlashQLA | GDN prefill, indexed state and native verification/decode | Match the actual recurrent state/head ordering and TP8 geometry. |
| v100-skinny | QPN operand layouts and shape-specific SM70 kernels decoding FP8/NVFP4 storage into FP16 MMA operands | Most direct for additional NVFP4 profiles; principles can inform original-format readers without requantizing SC117. |
| marlin_v100 | Separate SM70 dense/MoE development workspace | Useful comparator for supported packed encodings; it is not a universal decoder for learned GGUF codebooks. |
| Strata native IQ kernels | Direct original-block grouped expert kernels and Volta attention | Evaluate the actual CUDA SM70 specializations; similarly named AMD or newer-CUDA paths are not equivalent. |
| Linux io_uring | Queued PLE storage reads | Optional I/O implementation after cold/warm contention tests; no requirement to replace efficient mapped warm reads. |

The inspected projects commonly validate CUDA 12.8/Torch 2.10 builds. That supplies a reproducible comparison stack, not a reason to silently downgrade ProviderOne's qualified dependencies or assume these extensions compile unchanged with its newer source-build candidate. Keep native and Python components matched; preserve third-party provenance/notices. Avoid installing several overlapping inference frameworks into the serving critical path.

Sources: [1Cat source/component inventory](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/README.md), [FlashQLA compiler integration](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/flash_qla/compiler.py), [FlashAttention V100 upstream](https://github.com/ai-bond/flash-attention-v100), [marlin_v100](https://github.com/zhinianqin/marlin_v100), [v100-skinny](https://github.com/dnv2003/v100-skinny).

### A35/A39 Build identity, preparation caches and attribution

The complete component/build pass retains TurboMind, FlashQLA, Flash-V100, the independent FlashInfer-SM70 prototype, CUB/CUTLASS, original GGUF readers and Linux I/O as useful sources. Generalized FlashInfer planning and low-communication MMA dataflow are candidates; modern-only backends do not become SM70-capable by weakening guards. Optional standalone Marlin/v100-skinny comparisons still require actual codec and geometry admission. Strata's io_uring path is a bounded I/O option, not a requirement for every warm PLE read.

Pin linked native/Python sources together and test the installed artifact outside its checkout. Compilation caching differs from CUDA graph capture and from FX/AOT object serialization. Do not deserialize stale workspace addresses or carry a tuned reduction partition across incompatible numerical/shape/build keys. The projects record both stale source/binary mismatches and AOT reload output failures; a warm cache requires its own correctness check.

Strata declares MIT and 1Cat Apache-2.0 at their roots; bundled components have their own notices/provenance. Preserve applicable file-level attribution when porting a primitive.

Sources: [Strata build](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/CMakeLists.txt), [1Cat build](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/CMakeLists.txt), [compile-cache findings](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_release151_compile_cache.md), [Strata license](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/LICENSE), [1Cat license](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/LICENSE).

### A38 Repaired expert-bound defect and invalidated control

The later 1Cat Flash-Next worklog finds a legacy MMQ guard that excluded expert IDs 256–511 despite a 512-expert MTP bank. Poisoned graph-owned output remained unwritten for those routed IDs. Small M1/M5/M20 checks had missed the large-batch route.

The pinned main source now passes the actual expert-bank capacity and rejects IDs outside it. Thus this is a **historical control defect, repaired in the inspected main**, not an assertion that current main still skips those experts. The worklog explicitly disqualifies its earlier 5.159-second/~6708-token/s client baseline as fully qualified MTP prefill until corrected full-model timing and acceptance are available.

Audit any imported prefill number against its binary and executed MTP route. The independent GDN/convolution/reader operator measurements remain evidence within their own domains; do not discard unrelated results or reuse the flawed whole-model control. Add changed-graph tests for expert IDs 0/255/256/511, empty and tail groups, actual K tails, NaN-poisoned output and every live row written.

Sources: [root cause and invalidated control](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/benchmarks/flashnext_prefill/OPTIMIZATION_20261014.md), [corrected MMQ expert bound](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/csrc/libtorch_stable/quantization/gguf/moe.cuh).

### A38–A39 Call-shape, capture, API and capacity regression evidence

Open [1Cat PR #1300](https://github.com/1CatAI/1Cat-vLLM/pull/1300) identifies a current-main call mismatch: two GGUF expert-cache callers pass weights while one cache class accepts only layer/IDs. The pinned files confirm the signature mismatch. It affects the specified cache-backed subpaths, not every GGUF or all-resident ProviderOne configuration; no V100 boot was reproduced here. Historical qualified wheels and current main are distinct evidence.

Source: [caller](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/vllm/model_executor/layers/quantization/gguf_moe.py), [callee](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/vllm/model_executor/layers/quantization/gguf_expert_cache.py).

The issue sweep supplies further regression cases within A38/A39: capture-time dummy lengths can freeze the wrong attention route; debug tensors can alias or change graph execution; a shrinking speculative batch can expose stale collective/state participation; and chunk-boundary changes can alter retrieval. These are reported failures with specific historical builds, not assertions that every pinned route remains broken. Test actual replay, changed active counts and observation neutrality. A separate scratch-bound report sizes selection with one group and the guard with the real group count; the lesson is to make feasibility and execution use the same actual count and swizzle-expanded bytes, preserving the guard.

The field report advertising four 262K users explicitly clarifies that its 336,934-token pool cannot hold four full 262K histories. Its separate offload test stalls with deferred requests and no running work. This reinforces the distinction between offered concurrency and simultaneous full-window capacity, and the need for bounded transfer reservations, readiness/failure handling and queue progress. Also verify streamed responses end to end: engine completion with an unclosed/empty HTTP stream and invalid tool-call continuation fields are API failures even when GPU output exists.

Sources: [capture metadata #728](https://github.com/1CatAI/1Cat-vLLM/issues/728), [instrumentation #418](https://github.com/1CatAI/1Cat-vLLM/issues/418), [shrinking MTP batches #534](https://github.com/1CatAI/1Cat-vLLM/issues/534), [chunk-boundary retrieval #1003](https://github.com/1CatAI/1Cat-vLLM/issues/1003), [scratch feasibility #1270](https://github.com/1CatAI/1Cat-vLLM/issues/1270), [capacity/offload field report #1285](https://github.com/1CatAI/1Cat-vLLM/issues/1285), [stream completion #1248](https://github.com/1CatAI/1Cat-vLLM/issues/1248), [tool-call continuation #240](https://github.com/1CatAI/1Cat-vLLM/issues/240).

Other corrections retained by this pass:

- **Strata default status:** use its current SM70 table, not older opt-in prose.
- **1Cat INT8-G64:** explicitly a proposal, not an implemented/qualified KV format. Existing FP8/FP16 codec interfaces are useful; they do not supply ProviderOne's exact group32 Q8 codec.
- **Mixed-prefill policy:** historical adaptive-budget results and the newest default-disabled profile describe different policies; retain both tradeoffs.
- **Graph counters:** Python dispatch/capture counters do not count repeated graph execution. Observe replay coverage separately.
- **Grouped attention/head expansion:** the open 12/24-head page4 extension covers multiples of six; it does not automatically support ProviderOne's TP8 three-query-head local geometry or unique-owner attention.
- **M ranges:** a native launch bound is not a performance envelope. Keep original, routed, padded and per-request rows distinct.
- **Sparse/compact guards:** several paths retain a synchronized exact fallback. Do not report them as end-to-end sync-free.
- **FP32 wording:** FP32 accumulation does not prove FP32 logits, partial transport, intermediate score storage or output storage. Inspect each boundary.
- **Open precision repair:** PR #844 remains unqualified after severe model-speed regressions; overflow recovery already merged is distinct from its remaining finite-score proposal.
- **Reported maxima:** offered client concurrency, GPU-admitted requests, simultaneously active decode and common-window committed throughput are different quantities.
- **Acceptance:** accepted/drafted fraction, emitted tokens/round and a ratio of mean acceptance lengths are different metrics.
- **Method compatibility:** Strata pipelined batching may disable MTP; it cannot satisfy ProviderOne by silently dropping the required method.
- **Queue policy:** Strata's short-first admission and inactive-session caching are not ProviderOne's oldest-eligible FIFO contract.
- **Memory reuse:** head unloading/reloading, graph failure followed by process restart, CPU expert execution and active host-KV streaming remain incompatible with settled ProviderOne promises.
- **Microbenchmarks:** all expanded-table, persistent, atomics, extra-stream, DMA and higher-occupancy variants retain negative complete-chain evidence where they lost. Ideal saved bytes/launches do not establish a win.

Sources: [INT8 proposal](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/architecture/int8_g64_codec.md), [current release evidence levels](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/serving/sm70_release_profile.md), [head geometry PR #1236](https://github.com/1CatAI/1Cat-vLLM/pull/1236), [finite-score PR #844](https://github.com/1CatAI/1Cat-vLLM/pull/844), [Strata batching limits](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/docs/BATCHING.md), [head-alias limits](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/docs/PREFILL_HEAD_ALIAS.md).

## 12 Hardware coverage and screened alternatives

### V100 capability coverage

The design already covers residency, graphs, NUMA, native mixed readers, Tensor Core/SIMT crossover, packing, bank conflicts, register pressure, output layouts and topology collectives. Those remain obligations; they are not counted again as external discoveries.

| Capability or boundary | Consequence for these candidates |
|---|---|
| Native FP16 Tensor Cores with FP32 accumulation | E03–E05/E15 may use wider or better reused tiles; input/output range and cast points remain separate. |
| FP32 recurrence with short bounded update records | E21 trades repeated state stores/snapshots for output evaluation and periodic flushes; the source's TF32/BF16 paths are not SM70 implementations. |
| Integer/FP32 work, ballots, shuffles and shared memory | E01/E02/E12/E16 can exploit metadata/bit operations; DP4A does not grant free exact floating-activation quantization. |
| Lossless integer bit packing of FP32 state words | E24 can preserve numerical bits while reducing useful sectors read; raw-capacity slots and all codec costs remain reserved. |
| Register/shared/L1 resource limits and explicit synchronization | New fusion or lookup tables must fit the actual SM70 resource envelope; more reuse can lower occupancy or spill. |
| HBM2 and cache hierarchy | E01 targets writes; E02 key reads; E09/E11/E12 repeated or encoded reads. Measure bytes and full-step latency rather than theoretical peak. |
| NVLink cube, P2P and local copy engines | Required ETP4/2 and A32/A33 must use measured owner/consumer layouts and link costs. A published NVSwitch/multimem schedule is not this topology. |
| CPU SIMD/NUMA, PCIe and storage | E13 uses detected host capabilities and bounded transport. It cannot become per-step active KV/expert offload. |
| VMM and graph-stable addresses | E14 requires actual support/granularity/lifetime proof; it is not automatic demand-paging capacity. |
| SM-ID registers and bounded software work assignment | E17 can use placement-aware tickets; no permanent SM binding, contiguous-ID assumption or progress dependence on co-residency. |
| V100 fixed-function NVDEC | E20 targets eligible video ingest with a separately qualified C API/build, exact preprocessing and a bounded surface pool. |
| No native BF16/TF32/FP8/INT4/INT8 Tensor Cores, cp.async, TMA, TMEM/WGMMA or thread-block clusters | Newer research may supply algorithms, not executable V100 kernels. No ISA gate is weakened. |

Sources: archived Volta/PTX guides, NVIDIA's video support matrix and the governing design's platform manifest. [S26]; [S30]; [S36]

### Architecture substitutions with incompatible contracts

1. **1Cat DCP2** shards target main KV but replicates compressed indexer keys, rings, GDN/PLE state and draft QSA. It reports capacity gains, C1 decode regressions around 3% on some integrated controls, and a repeated-2K TTFT regression of 16–18%. ProviderOne already has the stronger unique-history capacity goal and exact partial-attention merge. These results do not justify replacing its control with that replication pattern. A new read-copy variant remains eligible under Design Section 1, with all repeated state charged and decode nonregression required.
2. **Strata expert tiers and residency-aware routing** solve small-VRAM execution. They can alter CPU/GPU rounding and, in some service configurations, residency-biased routing. ProviderOne preserves selected experts/routing and keeps all routed experts in its retained SC117 plan.
3. **Strata lazy vision** unloads an encoder and returns memory to expert caches, but its measured path requires one-request-at-a-time mode and changes encoder process lifetime. ProviderOne promises a fixed qualified vision plan and useful media service at full residency; it cannot spend that reserve on additional guaranteed sessions.
4. **Conversation caches** in Strata provide useful delta/checkpoint backing and branch preservation, but many entries are best-effort and may be dropped for later recomputation. Its documented Responses endpoint is stateless and does not retain previous_response_id. ProviderOne's protected queued checkpoints and standard stored-response chains require stronger semantics.
5. **Persistent megakernels** remain research in the inspected 1Cat Flash-Next note. It explicitly reports no working persistent SM70 kernel or end-to-end speedup from that proposed redesign. Typical remaining graph-entry gaps are too small to cover its full performance deficit. Do not start a model-wide megakernel rewrite from a proposal.
6. **TurboQuant** in Strata changes codec, keeps an INT8 MTP cache and reported slower prefill/decode than INT8 in its shown test. It is not evidence to replace the selected Q8/iso4 plan.

Sources: [DCP2 ownership and measured limits](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/qwen4exp_qsa_dcp2.md), [Strata cache and lazy vision behavior](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/README.md), [persistent execution proposal](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/flashnext_persistent_execution.md), [TurboQuant measurements](https://github.com/jmnargi/Strata-V100/blob/2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926/benchmarks/v100-turboquant-2026-10-25.md).

### Other screened directions

| Direction | Disposition |
|---|---|
| FLUTE repacking, generic fusion/overlap, CoCoNet, FLUX/TokenWeave motifs | Useful references; overlap A08–A14/A32–A33 and existing hardware rules. A modern implementation is not another SM70 capability. |
| DoSA's full signed-head certificate pipeline | Conditional reference within E01/E02. Its signed 32/64-head benefits do not transfer automatically to four equally positive QSA heads; its public implementation identifies GB200 hardware. |
| Approximate attention eviction, cross-layer/stale-index substitution, Quest's fixed page budget, epsilon/delta approximate retrieval | Changes target selection/output semantics. Not an exact default optimization. |
| Recent FP16 KV tails before quantization, KIVI/KVQuant-style new codecs, different activation/weight quantizers | Separately selected quality-profile research only. They do not satisfy “same specified quantizations” merely because benchmarks improve. |
| Other target parallelism, expert shadow routing, repeated nonexpert request groups, full/expert-bank replicas or duplicate prefill/decode engines | Outside the current supported matrix/backlog; only TP4×PP2 with ETP4/2 is retained as optional future work under Design Section 20. ETP1 is excluded. |
| Tensor representation, endpoint/read-cache or vision copies within the selected matrix | Eligible only under Design Section 1; ≥1 GB net extra across all GPUs in the selected deployment receives exceptional scrutiny. Copies do not authorize another group/routing topology. |
| Active CPU experts or active KV offload | Remain outside the settled GPU-resident execution contract; the revised duplication policy does not authorize paging. |
| Reversible GDN rollback by algebraically inverting rounded state | Not lossless: rounding discarded information and near-singular updates can amplify errors. Retain qualified snapshots/replay. |
| Dynamic-shape persistent whole-model runtimes, blanket manual SASS rewrites | Added complexity without an established incremental opportunity. A local measured kernel experiment is sufficient. |
| Compression based on average future KV entropy | Cannot increase guaranteed full-window admission without a valid worst-case bound and storage plan. |
| Disabling quality/reliability checks or changing router choices to match faster kernels | Outside the governing contract. |
| Omitting PV work because probability operands round to zero | Profiling hypothesis only. Exact zero in one operand does not prove an omitted MMA preserves exceptional values or signed-zero behavior. Keep the reference denominator/rescaling and prove the full operation equivalent before any V-load elision; no probability threshold approximation is admitted. |
| Unrestricted string-level grammar jump-forward | E22 admits only token-exact singleton paths. SGLang documents retokenization/probability limitations; neither forced bytes nor a valid final string proves the original token distribution. [S40]; [S41] |
| Gumbel/A* lazy full-support sampling | The perturb-and-search principle can sample without evaluating every density term, but does not by itself supply ProviderOne's finite-precision sampler, top-p normalizer, RNG or speculative p/q/correction contract. Future research after E23's basic bounds prove useful; no stochastic shortcut is adopted. [S43] |
| Lossless compression inside active collectives | UCCL-Zip v2 reports negative NVLink and ring-all-reduce results; its two-shot improvement starts at much larger messages with NVLink disabled. Do not import its framework or headline inference percentage for this eight-V100 cube. Existing bounded transfer/compression work can use the algorithm only if a measured message regime reverses that cost. [S45] |
| Treating Volta's four quadpair MMAs as a new accelerator discovery | Native m8n8k4 already underlies A10/A14 and the inspected TurboMind 8×32 mapping. Different experts per quadpair do not automatically remove the eight-row padding or save operations over full output widths. Consider a narrowly measured tail-packing variant only after the existing packed routes leave an identified hole. [S46] |

## 13 Experiment order, qualification and design integration

Correctness and valid controls precede optimization. The release order below reconciles the repository audit's final priorities with the external research priorities. Actual ProviderOne traces determine ordering within a band; P1 denotes an early feasibility study, not permission to skip prerequisites or promote an unqualified path.

### Release and implementation order

| Order | Work | Required decision evidence |
|---:|---|---|
| 0 | Validate every executed target/MTP expert ID/row, loaded routes, current controls and selected Inco dynamic range; verify API completion and state transactions | Complete finite computation with valid q, state and response semantics; repaired historical controls cannot remain timing baselines |
| 1 | Remove dead/duplicate buffers; evaluate A09 equal-byte single banks, A15 packed heads, A16 selector sharding, A30 lifetime-safe scratch and A41 replay; run E21 state/traffic feasibility | Preserve or increase qualified full-window capacity after all metadata, graph, scratch and overlap charges; replay/materialization must also pass decode gates |
| 2 | A08 indexed grouped MoE, A10–A14 small routing/vector/joint paths and source-format × TP8-shape × actual-M crossovers; trace E05/E09 opportunities | Complete FFN chains improve with fewer copied bytes, preserved numerics, bounded work and no per-count decode loss |
| 3 | A01 retention-aware prefill, A02 useful partial checkpoints and A03 dependency-aware prefix identity | Cold/miss and warm-prefix correctness, actual larger useful chunks/fewer forwards and reduced real agent-turn replay; preserve every retained boundary |
| 4 | A04 overhead-aware mixed scheduling, A05/A06 mixed/tail graph coverage and resident-row fast paths; conditional E17 | Improve mixed traffic with bounded graph memory, incoming TTFT and longest resident service gap; preserve pure-decode gates |
| 5 | A19–A21 direct packed history, live-length exact selection and multiquery reuse; E01 witnesses, E02 certified bounds, E11 shared-prefix work and E18 fixed distributed selection where indicated | Identical logical winners/masks/order; relocation/restore/alias/tail tests; unique histories, real row budgets and full-context/full-concurrency prefill/decode gains |
| 6 | A22 native GDN strides/outputs/preprocess, E03/E04 arithmetic studies and E21 materialization; A23–A26 PLE hash/convolution/transport/prompt lookahead plus E19 draft-ID lookahead | Numerical/long-state/recovery parity and complete-round gains after CPU/NUMA/PCIe/lookup/flush contention; preserved media, mixed and rollback behavior |
| 7 | A32/A33 topology collectives, exact metadata fusion and HC ownership; A27–A29/A40 bounded host/handoff/pinning | Beat the best qualified path for each actual payload/precision regime on the cube while retaining graph/order/cancellation/progress and protected checkpoint durability |
| Workload-gated | E07/E22 structured-output head/forced spans; E23 certified head pruning; A31/A36/E20 media/positions; E13 backing compression; E12/E14/E15/E16/E24 feasibility | Use each candidate's exact eligibility, proof and rejection gate; include complete catch-up/codec/transport/peak-resource costs |
| Parallelism profiles | Required TP8×PP1/PP2 with ETP8/4/2; one TP8 stage/server, selected ETP for MTP and fixed TP8 dense DFlash | Three PP1 cases, then three TP8×PP2 cases with actual second-server/RDMA evidence; all selected ETP degrees and complete capacity/performance/state gates |
| Later | A18 alternate proposal/vocabulary methods, A34 optional eager-send tuning within TP8×PP2, optional future TP4×PP2 with ETP4/2 under Section 20, persistent local pipelines or ancillary runtime rewrites | A concrete performance, correctness or inherited-complexity justification and independently complete method/architecture/no-restart qualification; required direct refactors need not wait for this band, and isolated kernel results do not justify a model-wide rewrite |

Add a tensor-copy arm only within the supported matrix at the stage justified by its measured bottleneck, not solely because of its size. Its receipt must state gross/net aggregate and per-rank steady/peak bytes, cumulative threshold tier, repeated matched gains, qualified capacity change, lower-copy alternatives for big-ticket items and acceptance/fallback rationale. Experiments do not silently change the capacity-first default.

Matrix decisions use priority **1 < 2 < 4 < 8 < 16 < 24** without invented weights. Gains in one cell do not excuse decode loss in another. Extend coverage through 96/120 target rows, all smaller legal tails, real per-expert distributions, qualified microbatches and the actual maximum admitted activity.

### First measurements, rejection budgets and experiment receipt

After the release prerequisites, use these candidate-specific first measurements. They are a feasibility checklist, not a second global implementation order:

1. **E21:** reconcile current GDN state traffic/versions; test FP32 corrections, ring/flush invariants and numerical feasibility at actual local geometry.
2. **E01/E03:** trace score-write/selection cost and witness quality; establish whether the GDN route repeats raw Gram work.
3. **E05:** quantify expert-wave/straggler time before building a scheduler.
4. **E07/E22:** measure legal-support sizes, consecutive singleton-token runs and eligibility under the actual sampler order.
5. **E12/E24:** census static-bank entropy and active FP32 state bits independently, with actual tile/sector/metadata costs. Neither first test needs serving integration.
6. **E19:** measure the actual draft-ID-to-target-PLE lead window and lookup critical-path share before changing notices or graphs.
7. **E20:** verify video capability, stack eligibility, pixel/timestamp parity and worst-case surfaces before testing media overlap.
8. **E23:** reject loose head bounds offline before building irregular projection or sampling code.

E02 follows only after conservative bounds prove both safe and tight enough. E04/E06/E15 require numerical or sampling proof before extensive optimization. E08/E09 should first beat simple fixed policies in bounded trace replay. E11/E13 require representative sharing/queue traces; E14/E16 get strict feasibility budgets. E17 needs demonstrated complementary mixed work and a viable common SM70 resource footprint. E18 starts with fixed placement and a communication/selection cost pilot, including the already-balanced C24 case.

Evaluate E21 and E24 separately before their combination; one changes arithmetic/dataflow and the other must preserve serialized bits. E22's state catch-up and constrained-output accounting belong in its complete cost. A capability or algebraic proof can justify a prototype without predicting that prototype's throughput.

Report effects on the deciding workload. A media-ingest or backing-tier gain is valuable, but cannot be reported as progress against the pure-text full-concurrency decode target. Speculative PLE work counts only through its complete-round benefit, including notices and contention. No headline gain is added to another candidate's percentage.

For every experiment retain:

| Decision evidence | Required content |
|---|---|
| Exact scope | Artifact/profile, executed route, source/build identity, shapes, histories, active/resident counts and cache/media/constraint conditions |
| Cost and capacity | Complete operation/round latency; physical bytes, bandwidth/traffic, launches, padding, scratch, graph and overlap peaks; worst-rank integer capacity |
| Correctness | Stated exactness boundary, reference tests, masks/ties/casts, long state, sampling/RNG, accepted-prefix and failure/lifetime behavior |
| Service behavior | FIFO and per-request progress, input/first-token/completion latency, inter-token gaps and every required activity point |
| Decision | Retain, revise or reject, with the specific falsified assumption; no adding operator speedup percentages together |

The backlog is additional research, not a forecast that all entries will win. The useful result of a cheap experiment may be a decisive rejection. Adopt a candidate through explicit ProviderOne ownership, numerical/profile identities, resource proofs and performance gates; existing owners may be replaced directly.

### Mapping into the governing implementation design

These mechanisms were assembled against revision 21; the current ProviderOne design, including independent implementation ownership and the phased roadmap, governs. Source mechanisms are conditional refinements, not automatic amendments.

| Governing design area | Integrated implementation addition |
|---|---|
| Design Sections 9/16 | Separate physical pages, useful coherent retention boundaries and execution chunks; preserve logical order under physical relocation; measure real replay avoided |
| Design Section 15 | Original-format grouped readers, indexed routing, complete TP8/M crossover maps, mixed projection fusion, direct GDN state/output interfaces, and the separately qualified arithmetic/work-avoidance experiments |
| Design Sections 18/21 | Executed operator/fusion/capture census, metadata-exact matching, bounded scratch/lifetimes and topology/payload/precision evidence |
| Design Section 20 | Compare mapped, queued direct-I/O and pinned packed PLE under host/NUMA/PCIe budgets; keep idle backing distinct from active expert/KV service |
| Design Section 21 | Physical-allocation permutation, discarded-checkpoint chunking, extra-bank memory, every actual row/count regime, long recurrence, sampler/RNG, accepted-prefix and finite-range validity |

Neither inspected project proves the full >250 C1 / >600 aggregate C24 decode contract at native near-full Q8 histories together with the separately bound >3500 prefill reference. The user's hardware verification remains the target. No reported percentage is added to another candidate's percentage, and no throughput figure converts directly into guaranteed full-window capacity.

## 14 Research provenance and primary sources

### Input and implementation snapshot identity

| Input or implementation | Inspected identity |
|---|---|
| TensorFold Strata/1Cat Comparison | Revision 1; 105,450 bytes; A01–A41 |
| TensorFold External Research Candidates | Revision 2; 80,809 bytes; E01–E24 and S01–S46 |
| Governing TensorFold design | Revision 21 |
| Strata-V100 main snapshot | 2c9b1dd7bda3102c4a8e65fbe56c45f5c6882926 |
| 1Cat-vLLM main snapshot | 4d57473db67eb77a13018d179e4d95169e82044f |
| Strata Inco implementation branch | baede60aa1d8ccf5a7b8ed83822da4f1229f92ef; unmerged PR #68 |
| ReVolta reference subset | b7a6fe8bb18ecb77b4dbbc4273ed4171a8613973 |
| ReplaySSM author source | 73a9592e9d8a2268620e8883a82f5cec7442ed36 |

Repository heads/statuses were resolved by the source audits. Their findings concern these snapshots, not every future release. This integration does not refresh repository heads or independently reproduce retained measurements.

### Repository audit coverage and evidence limits

The repository audit inventories **all 9,802 tracked files** in the two nontruncated trees, retrieves and searches **3,632 files / 61,042,759 bytes**, and reviews the relevant implementation chains, native dispatch, tests, build definitions, benchmark records and historical decisions. Every retrieved main-snapshot file matches its Git blob hash. Coverage comprises 1,087 Strata files and 2,545 1Cat files, including **all 354 text/data files under 1Cat's design directory**, the entire bundled TurboMind tree, FlashQLA, Flash-V100, FlashInfer-SM70, the Qwen4Exp modules, and the selected serving/cache/speculation paths. Strata coverage includes 254 src files, 152 include files, its English operating guides, benchmark experiments/results and relevant tests.

The published pull-request catalogs were paginated to completion: **80 Strata PRs and 1,118 1Cat PRs**. At retrieval, Strata had no open PRs; 1Cat had 78. Titles and bodies were screened, with direct inspection of the applicable candidate descriptions, changed-file records and selected branch implementations. Unmerged Strata Inco DFlash2 and 1Cat's open storage, selector, I/O and topology proposals are distinguished from the pinned main snapshots.

Its issue/release scope also screens **all 191 non-PR 1Cat issues**, selected discussion threads and **all 10 published GitHub release entries**; Strata's corresponding issue/release collections contain no non-PR issues or published releases. Issue #1302 points to another exact-family implementation in ReVolta, a 1Cat reference fork. A targeted review retrieves and Git-hash-verifies **19 files / 997,679 bytes** at its separate commit `b7a6fe8bb18ecb77b4dbbc4273ed4171a8613973`, and examines the 11 linked integration commit catalogs. This is a relevant reference subset, not a comprehensive third-repository audit. It adds the distinct accepted-prefix recurrent-replay candidate and concrete native/host qualification cases.

This is a comprehensive **candidate-discovery and design-comparison audit**, not a claim that every line in 9,802 files was manually reviewed. Unrelated models, AMD/SYCL-only arithmetic, media generation, UI assets, translations and generic examples are inventoried and dispositioned; applicable mechanisms from those areas were screened for transfer. Native performance reports were checked against current callers and negative findings. Reported tests remain the projects' evidence; no GPU tests or model benchmarks were executed here.

The [retained source-coverage JSON](https://chatgpt.com/library?file_id=libfile_b1877bbc3be88191be892c567707c432&account_id=personal) records every tracked path, blob identity, retrieval/verification status, scope disposition and reviewed PR inventory. A retrieved file may have been searched as part of a corpus rather than individually reviewed line by line; that distinction is explicit.

**Repository-audit closure:** repeated source-coverage sweeps found no additional unclassified family within the pinned repository scope after A01–A41 were assembled. This conclusion is bounded by the pinned snapshots and retrieved public branches; it does not certify ProviderOne's implementation or establish that an undiscovered optimization cannot exist.

### Repository-to-design coverage

| Audited subsystem | Main evidence/call chain | ProviderOne mapping |
|---|---|---|
| Source/import/build/codec preparation | Both build systems; all bundled TurboMind files; GGUF adapters/readers; preparation and storage tests | Design Sections 2, 4, 7, 13, 15 |
| Dense/expert/router/head kernels | Native packed and grouped CUDA, current dispatch, cold-bank benchmarks, inverse/shape/graph tests | Design Sections 12, 14, 15, 21 |
| QSA selection/attention/cache | Qwen4Exp module tree, Flash-V100, device/host readers, exact selector, planner and physical-allocation tests | Design Sections 9–11, 13, 15 |
| GDN/PLE/state | FlashQLA/native recurrence, caller strides, state/recovery tests, PLE transports/convolution | Design Sections 9, 12, 14–17 |
| Speculation/sampling | Both MTP paths, 1Cat DFlash/compact verification, Strata unmerged Inco draft, linked ReVolta feature/cache/replay subset, rejection/range/graph tests | Design Sections 12, 14, 21 |
| Prefix/queue/tiering | Scheduler/core managers, partial entries, Strata archive/slot/copy paths and failure tests | Design Sections 6–9, 16–17, 19 |
| Graphs/communication/allocator | Worker/model runners, captures, lifetime grouping, native collectives, topology branch | Design Sections 4, 10–11, 18–21 |
| Media/position/host work | Exact companion loading, encoder bounds, positional caches/YaRN, I/O/pin budgets and API staging | Design Sections 2, 6, 15–17 |
| Measurement/provenance | All 1Cat design records, applicable benchmark/prototype corpora, V100 reports, full PR/issue/release catalogs and selected discussions | Design Sections 5, 21, 23–25 |
| Excluded architectures/products | Other model bring-ups, AMD/SYCL, H3/media generation, UI/Rust/LoRA and modern-GPU-only paths | No architecture replacement; applicable primitive families retained above |

All retained candidates have a disposition, a numerical/resource boundary and an implementation/measurement gate. Source absence, closed/unmerged work, partial tests and failed measurements are visible evidence limits rather than silently counted features.

### Audit method and closure record

1. **Inventory/documents:** complete trees and all 1Cat design text/data; added missing scheduling, cache, small-routing, memory and PLE candidates.
2. **Native/caller/tests:** traced actual route guards and caller layouts; found current-default/prose differences, small-M limits, TP8 head mismatches, shared scratch hazards and exact fallback synchronization.
3. **History/branches:** completed both PR catalogs and applicable branch/source review; added equal-byte storage, topology planning, selector sharding, lookahead reuse, bounded readahead and exact Inco overflow evidence.
4. **Benchmark validity:** checked historical decisions against source, paired controls and negative results; isolated the repaired expert-capacity defect, current call-shape regression and nontransferable benchmark conditions.
5. **Expanded experiment closure:** searched the additional Strata and 1Cat native benchmark/prototype corpora and remaining model tests; every mechanism mapped to the register or an explicit exclusion.
6. **Reverse coverage check:** checked each ProviderOne hot-path/state/resource area against the repository map and revisited open candidates, default maps and negative evidence; no additional unclassified candidate family.
7. **Consistency check:** rechecked source status, artifact/codec/method distinctions, metrics, byte arithmetic, links and unchanged ProviderOne constraints; clarified that matrix-accumulation precision does not prescribe collective precision and kept the prefill reference's unspecified conditions explicit.
8. **Public issue/release extension:** screened the complete remaining catalogs and selected discussions; traced the linked exact-family ReVolta subset and added A41, register/resource tuning, exact pinning and further regression cases.
9. **Repeated closure and consistency:** reconciled the extended evidence against every candidate and ProviderOne subsystem, rechecked source identities, conditions, byte arithmetic and unchanged constraints, and repeated the coverage check after the additions. No further unclassified candidate family was identified.

These are analysis passes, not GPU qualification runs.

### External primary sources and inspected status

The source research checked these links and recorded the status boundaries below. Mutable documentation is background, not a deployment version pin; this consolidation preserves those references without claiming a new live-source audit.

| Ref | Source and evidence boundary |
|---|---|
| S01 | DoSA full paper, including numerical certificate conditions; inspected public implementation branch 582a8a151d5b14bd3b9d79e932a6390638d11e5a in llm-sys/dosa-sglang. Algorithm transfer only. |
| S02 | NVIDIA GVR V2 exact Top-K article; unordered-index/tie contract needs ProviderOne adaptation. |
| S03 | Quest paper; approximate page selection is not adopted. |
| S04–S06 | Author GDN derivation/report and FLA code/PRs; PR797 closed/unmerged, head 7004d193f9e26aa706fb8fa9b85022e80eec04c1; redundant-inverse PR433 already merged. |
| S07 | Stream-K author publication page/paper; dense work-decomposition evidence. |
| S08 | ICLR block-verification paper, version 3; theorem assumptions and main algorithm matter. |
| S09 | XGrammar engine integration; grammar primitives, not a restricted-head benchmark. |
| S10 | TETRIS ACL2025 paper; allocation assumptions differ from ProviderOne's complete-round objective. |
| S11–S12 | Activation-pattern paper and author batch-clustering post; other models/topologies, no transferable percentage. |
| S13 | Tutel author source; placement/communication background; E10 is withdrawn from the active backlog. |
| S14–S15 | Hydragen author implementation and FlashInfer Cascade article; shared-prefix attention computation and precision boundaries. |
| S16–S18 | DFloat11, NeuZip and tile-ANS paper; distinguish formats and lossless/lossy modes. S18's retained HTML Qwen batch figures disagree between prose and table. |
| S19–S20 | Invariant Bit Packing primary paper and nvCOMP prerequisites; A100 study versus separately qualified Volta-capable software. |
| S21–S22 | vAttention author repository and CUDA VMM documentation; stock versus modified-driver granularity distinguished. |
| S23 | Fasi et al. multiword Tensor Core error analysis/experiments, including V100. |
| S24–S25 | LUT-GEMM/CodeGEMM official source; inspected heads 647d49fcf0b054b6dafee04b5ce1c8c2adb047e5 and 8e9f34f5bec121793b3d19178470f3f036ac6702. |
| S26 | NVIDIA archived Volta tuning guide. |
| S27 | POD-Attention ASPLOS2025 paper; inspected microsoft/vattention head 71a0e91aa46ff8fa985bcca3327efe0ab9929a39 requires Ampere. Borrow scheduling, not its binary or speed ratios. |
| S28 | Upstream vLLM sparse-indexer merge and ownership callers at fa3b9d1377219247060d3c8b49eae199c9e7e154; distinct from the pinned 1Cat code. Hopper/Blackwell-oriented implementation. |
| S29 | Engram primary paper, deterministic lookup/transfer sections; another model/hash and H800 evaluation. |
| S30–S31 | NVIDIA Tesla V100 video matrix and NVDEC API guide; exact hardware capability and surface lifetime remain runtime checks. |
| S32–S34 | Video Codec SDK13.0/13.1 and current PyNvVideoCodec requirements; distinguish the supported C API candidate from unsupported newer stack choices. |
| S35 | NVIDIA Container Toolkit video driver-library exposure; preserve all required capabilities. |
| S36 | Archived PTX SM-ID/identifier-extent semantics; not an SM pinning guarantee. |
| S37–S38 | ReplaySSM paper v1, GDN/flush appendices and author source at 73a9592e9d8a2268620e8883a82f5cec7442ed36. Inspected standard/speculative GDN kernels and cursor wrappers; numerical choices and threshold naming need independent SM70 integration. |
| S39 | TreeWY v2, revised 2026-10-22; B200/other-model study, unmerged reference, greedy evaluation and changed numerical paths. Its chain comparison favors ReplaySSM. |
| S40–S41 | SGLang paper v2 and author compressed-FSM article; published string/retokenization method and explicit probability limitations, rather than proof of E22's stricter token-exact path. |
| S42 | Ram/Gray KDD2012 author-hosted exact inner-product search paper; geometric bound, not a vocabulary-head/V100 benchmark. |
| S43 | Maddison/Tarlow/Minka A* Sampling primary paper; perturbation/search principle only, not an integrated speculative verifier. |
| S44 | ndzip-gpu SC2021 primary proceedings; lossless scientific floating-point compression on Volta/Ampere, not measured GDN-state compressibility or a fused inference reader. |
| S45 | UCCL-Zip v2, especially Sections 5.2.2/6; negative NVLink/ring evidence and separate large-message two-shot experiment. No transfer of its headline inference estimate. |
| S46 | NVIDIA CUTLASS Volta MMA atom description, checked against the already inventoried native TurboMind MMA/scheduler. Hardware granularity alone does not establish a new kernel advantage. |

[S01]: https://arxiv.org/html/2610.13990v1
[S02]: https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog29_GVR_V2_Self_Sampling_Exact_TopK_for_Sparse_Attention.html
[S03]: https://arxiv.org/abs/2406.10774
[S04]: https://veitner.bearblog.dev/simple-math-to-speed-up-gdn-prefill/
[S05]: https://hypnopump.github.io/blog/accelerating-gdn-inference.html
[S06]: https://github.com/fla-org/flash-linear-attention/pull/797
[S07]: https://mgarland.org/papers/2023/streamk/
[S08]: https://arxiv.org/html/2403.10444v3
[S09]: https://xgrammar.mlc.ai/docs/latest/using_xgrammar/engine_integration.html
[S10]: https://aclanthology.org/2025.acl-long.1598/
[S11]: https://arxiv.org/abs/2604.23150
[S12]: https://blog.doubleword.ai/moe-expert-coactivations
[S13]: https://www.microsoft.com/en-us/research/publication/tutel-adaptive-mixture-of-experts-at-scale/
[S14]: https://github.com/ScalingIntelligence/hydragen
[S15]: https://flashinfer.ai/2024/02/02/cascade-inference.html
[S16]: https://papers.nips.cc/paper_files/paper/2025/hash/8e8b4449c80879215ff2ee47b6dd01a6-Abstract-Conference.html
[S17]: https://arxiv.org/html/2410.20650v1
[S18]: https://arxiv.org/html/2606.15789v1
[S19]: https://akkamath.github.io/files/EuroSys26_IBP.pdf
[S20]: https://docs.nvidia.com/cuda/nvcomp/installation.html
[S21]: https://github.com/microsoft/vattention
[S22]: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html
[S23]: https://nhigham.com/wp-content/uploads/2023/02/fhlm23.pdf
[S24]: https://github.com/naver-aics/lut-gemm/tree/647d49fcf0b054b6dafee04b5ce1c8c2adb047e5
[S25]: https://github.com/naver-aics/codegemm/tree/8e9f34f5bec121793b3d19178470f3f036ac6702
[S26]: https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html
[S27]: https://www.microsoft.com/en-us/research/wp-content/uploads/2025/03/POD-Attention-ASPLOS25.pdf
[S28]: https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/model_executor/layers/sparse_attn_indexer.py
[S29]: https://arxiv.org/html/2601.07372v1
[S30]: https://developer.nvidia.com/video-encode-decode-support-matrix
[S31]: https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html
[S32]: https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/read-me/index.html
[S33]: https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/read-me/index.html
[S34]: https://docs.nvidia.com/video-technologies/pynvvideocodec/read-me/system-requirements-common.html
[S35]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
[S36]: https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#special-registers-nsmid
[S37]: https://arxiv.org/html/2610.13778v1
[S38]: https://github.com/Johnny-Liou/ReplaySSM/tree/73a9592e9d8a2268620e8883a82f5cec7442ed36
[S39]: https://arxiv.org/pdf/2608.20961v2
[S40]: https://arxiv.org/html/2312.07104v2
[S41]: https://www.lmsys.org/blog/2024-02-05-compressed-fsm/
[S42]: https://rithram.github.io/research/papers/2012/RG_KDD12.pdf
[S43]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/5449-a-sampling.pdf
[S44]: https://sc21.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap595.html
[S45]: https://arxiv.org/html/2604.17172v2
[S46]: https://docs.nvidia.com/cutlass/4.5.3/media/docs/cpp/cute/0t_mma_atom.html

Additional inspected implementation/provenance links: [DoSA branch](https://github.com/llm-sys/dosa-sglang/tree/582a8a151d5b14bd3b9d79e932a6390638d11e5a); [FLA experiment code](https://github.com/hypnopump/flash-linear-attention/blob/7004d193f9e26aa706fb8fa9b85022e80eec04c1/fla/ops/gated_delta_rule/chunk_fwd.py); [already-merged inverse removal](https://github.com/fla-org/flash-linear-attention/pull/433).

Additional implementation links: [POD source requirements](https://github.com/microsoft/vattention/blob/71a0e91aa46ff8fa985bcca3327efe0ab9929a39/pod_attn/README.md); [POD scheduling implementation](https://github.com/microsoft/vattention/blob/71a0e91aa46ff8fa985bcca3327efe0ab9929a39/pod_attn/pod_attn/fused_fwd_kernel.h); [upstream local causal lengths](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/attention/backends/mla/indexer.py); [single-owner compressed slot mapping](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/attention/backends/mla/compressor_utils.py).

Gap-review implementation links: [ReplaySSM speculative GDN kernel and cursors](https://github.com/Johnny-Liou/ReplaySSM/blob/73a9592e9d8a2268620e8883a82f5cec7442ed36/vllm/third_party/flash_linear_attention/ops/gdn_replayssm_spec_decode.py); [standard GDN path](https://github.com/Johnny-Liou/ReplaySSM/blob/73a9592e9d8a2268620e8883a82f5cec7442ed36/vllm/third_party/flash_linear_attention/ops/fused_recurrent_replayssm.py).
