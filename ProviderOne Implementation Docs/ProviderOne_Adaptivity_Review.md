# ProviderOne Adaptivity Review

4 November 2026, America/Phoenix. Adopted adaptivity scope, with 5 November 2026 UTC source/architecture alignment against the owned `python-0.6` implementation, integrated into the governing documents.

## Decision

The implementation plan uses a small scheduler and prepared execution plans. Live KV degree conversion and migration are optional benchmark-gated experiments, excluded from initial implementation obligations. Graph-set replacement belongs to preparation or explicit qualified administration; ordinary activity selects existing ready graphs. These document updates establish scope, not completed implementation or measured performance.

An optimized engine still encounters different operator shapes at one and 24 active requests. It also encounters different history lengths, expert populations, speculative acceptance, prompt/decode mixes and media work. Responding to those differences can improve performance. Most responses can be deterministic dispatch or ordinary bounded scheduling, with no persistent-state change or online tuning system.

Preserve the mission: maximize qualified full-window GPU-resident capacity, then committed decode throughput at full activity, with the existing lower-concurrency, quality, latency and fairness gates. Small reproducible gains remain eligible; there is no arbitrary percentage cutoff. The control for any adaptive optimization must be the best qualified simple implementation under the same commitments.

The historical review basis is design revision 30, experimental register revision 7, parallelism research revision 8, and numerical companion revision 6. The adopted scope is synchronized across their updated documents. The required matrix remains TP8×PP1 on one server and TP8×PP2 on two servers, each with ETP8/4/2. ETP1 remains excluded; one-server TP4×PP2 with ETP4/2 remains optional future work.

## Current implementation and cross-hardware application

The current source baseline is only [TensorFold `python-0.6` at `c83187878423388715571d194536eba836a64804`](https://github.com/kklouzal/TensorFold/tree/c83187878423388715571d194536eba836a64804). It contains unique-history/full-window planning, bounded deterministic controls, prepared native-owner/partial-graph mechanisms, shared logical Responses/Conversations storage and common prefix/retained-state RAM paths. These are implementation starting points with scoped evidence, not a newly qualified V100 service. There is no evidence of a complex online-learning subsystem that must be replaced merely to simplify policy.

Apply the adopted fixed-placement scope directly to the existing source: allocate the selected map during preparation, with degree four as the first unmeasured control; remove initial live-degree publication and only the actual dedicated migration liability. Preserve independent frame/mutation authority, exact merging, generations, asynchronous retirement/failure containment, complete restore, logical/physical guards and surviving host/native resource highwaters. Reuse bounded prepared-plan selection and its exact evidence gate. The old narrow cu126 attention control does not qualify the owned CUDA12.9 stack or 262K context.

ProviderOne’s shared API/admission/state/artifact contracts apply across actual implementations; kernels, representation, build and memory accounting specialize for each device. Preserve selected existing GB10 numerical/deployment contracts during SC117/V100 delivery. The accepted Gemma head-only W4A16 trade remains scoped to that profile; unselected A8, N32, draft-graph and other candidates remain conditional. Owned RTX 4090 and possible Ampere support need actual legal operators and target qualification. Separate modern/SM70 immutable builds and unified/discrete memory ledgers; no lowest-capability kernel or universal tensor layout is required.

The governing design’s canonical-document/package/CLI/updater handoff and affected-profile gates preserve completed shared work without carrying old test counts into a new source/image. All original numerical, capacity, quality and SC117 performance obligations remain intact.

## Normal serving decisions

Use known request metadata and already-produced compact results. A single coordinator issues a consistent plan; device-side routing handles device-only details. Every decision below stays inside the prepared numerical, resource, communicator and buffer contracts.

| Mechanism | Adopted disposition | Smallest useful implementation and reason | Implementation location |
|---|---|---|---|
| FIFO admission, ready membership and completion handoff | Keep | Admit the oldest eligible work when capacity is available. Schedule ready GPU-admitted sequences and retire completions promptly. This is basic serving behavior. | Design §§6,16–17 |
| Decode batch and microbatch size/membership | Keep | Choose legal ready groups within actual row limits. Avoid batch-filling delays and fixed whole-wave barriers. Split only the constrained operator where the rest of the model can retain a larger legal batch. | Design §§12,14,16–17,20; Research §§8–9 |
| Verification-row quota | Keep | Vary rows used within a qualified maximum. Count anchor/proposal rows and multiplicity correctly. A larger arriving batch uses legal microbatches rather than enlarging an arena during execution. | Design §§7,12,14,17–18 |
| Prefill chunk size | Keep as deterministic planning | Start from a measured efficient chunk range, then clip for remaining work, required retained checkpoints, media boundaries and fairness. Physical KV pages, checkpoint retention and execution chunks are separate grids. | Design §§9,16; A01,A07 |
| Mixed prefill/decode quotas | Keep, simplify | Use prepared pure-prefill and mixed-work policies. Preserve decode progress and a prefill progress/aging floor. An online fitted cost model is unnecessary until the simple policy misses a measured goal. | Design §§16–17; A04–A05 |
| PP2 ready queues and causal prompt chunks | Keep | Overlap independent ready groups and known causal prompt chunks. Retain one unresolved speculative verification round per sequence. Predictive delay scheduling is a later response to measured bubbles. | Design §20; Research §§8–9,12 |
| MTP proposal depth | Keep | Select from a small measured width set before fresh proposal draws. Use complete-round cost and committed output. Active count alone is insufficient. | Design §§12,17; A18 |
| DFlash verified prefix length | Keep with precise cost accounting | Under the initial block-four plan, positive prefix lengths still execute the four-row draft. Shortening can save verifier/recovery work, without saving those draft rows. | Design §14 |
| Warm zero-proposal operation | Keep its existing contract | Temporarily skip proposals while preserving required conditioning for immediate reactivation. Include refresh/materialization cost. Warm zero proposals is a different timing/resource baseline from an administrator-selected no-speculation profile. | Design §§12,14,19 |
| Remaining context/reply/stop clipping | Keep as correctness | Clip to actual legal rows, positions and method shapes. A fixed four-row DFlash call may be ineligible near the limit. This remains necessary even if one width wins every ordinary workload. | Design §§7,12,14 |
| GEMV/small-M/tiled/Tensor Core kernel selection | Keep as shape dispatch | Reuse prepared tactics keyed by actual source format, shape, dtype, stride, alignment, workspace and build. Actual routed rows matter more than client count. Keep expert row classification device-side. | Design §15; A08–A14,A22 |
| GDN prefill/decode/verification kernel regime | Keep as prepared operator dispatch | Row geometry and valid state boundaries select a qualified kernel/chunk plan. Do not add a controller that changes recurrence precision or state representation with load. | Design §§12,15–16; A22,A41 |
| Live-length QSA scoring, selected-work grouping and local tiles | Keep | Execute only visible groups and legal selected positions; choose useful local tile/query grouping from prepared implementations. Exact selection and logical reduction order remain unchanged. | Design §§9,15; A19–A21 |
| Existing CUDA graph bucket | Keep | Select a bounded prepared graph using actual execution geometry and complete validity assumptions. Different integer request counts can share one winner. Charge padding and actual graph-pool lifetimes. | Design §§17–18; A05–A06 |
| Captured versus uncaptured launch | Keep the existing single path | Launch the same optimized kernels without capture where rare shapes or padding make that preferable. This needs no separate legacy serving runtime. | Design §18 |
| Legal media microbatch, attention tile and language quota | Keep | Use actual grids/frames/segments and prepared implementations. Preserve exact media fidelity, preprocessing, positions and fixed encoder ownership. | Design §§2,16–17; A36 |
| CPU/media work issuance | Persistent pools, variable issued work | Initialize bounded NUMA-bound pools. Let queues supply ready tasks; bound tasks/bytes in flight. Extra feedback-based quotas need actual contention evidence. Pool size, affinity and library thread settings stay fixed during normal serving. | Design §§2,16,19; A40 |
| Exact sampler/grammar fast-path eligibility | Keep semantic dispatch | Requested sampling, legal support and requested logprobs determine eligible exact paths. Never change the requested distribution to make an execution plan faster. | Design §§12,14,21; A17,E07,E22,E23 |
| Output backpressure and admitted fairness | Keep as lifecycle scheduling | Bound output and service gaps; stop a slow client from monopolizing resources. Readiness, cancellation and event retirement do not require a learned scheduler. | Design §§12,16–17 |

Sarathi-Serve provides a primary precedent for bounded chunked prefill and responsive mixed scheduling [S1]. Dynamic speculation has a documented batch-dependent verification cost [S2]. NVIDIA recommends reusing prepared cuBLASLt algorithm decisions [S3] and supports graph bucketing as a tradeoff between padding, flexibility and memory [S4]. These justify the mechanisms, without establishing their fastest parameters on this V100 deployment.

## Conditional experiments and excluded automatic changes

The reviewed design already forbade serving-time compiler/algorithm searches and fixed structural profiles. Those safeguards remain. The adopted scope also removes costly state/graph transitions from initial obligations until their need is demonstrated.

| Mechanism | Adopted disposition | Reason and experiment admission condition | Implementation location |
|---|---|---|---|
| Live attention ownership degrees 1/2/4 | Optional benchmark-gated experiment | Qualify a fixed unique layout first, with four-way striping within each KV-head quartet as the first practical control. Fixed alternatives can be tested if a bottleneck supports them. No degree is declared a measured winner. | Design §10; Research §12 |
| KV redistribution, balanced/incomplete migration bundles and mixed-holder execution | Excluded from initial obligations; gated experiment | Destination/source overlap, mapping generations, union communication, reader retirement, partial execution and oscillation handling are a substantial development surface. Require net gains over the best fixed layout including transition cost. | Design §11; §22 stage 4 and named “Optional migration study”; §24 |
| Load-driven canonical indexer owner changes | Keep out of the baseline | Canonical ownership is already fixed. E18 can test a fixed distributed indexer without first building a live repartitioner. | Design §§9–10; E18 |
| Graph replacement/recapture because activity changes | Excluded from normal serving | Ordinary activity selects prepared graphs. Supported parameter updates and dynamic inputs are distinct from rebuilding a graph. Preparation or explicit qualified administration can replace the graph set. | Design §17 preparation/administrator row; §§18–19 |
| Resource-envelope changes because activity changes | Use one prepared maximum | Ordinary adaptation changes used quotas inside one prepared maximum envelope. Actual maximum/arena/capacity changes follow explicit qualified activation. The reviewed wording did not establish that live arena resizing or an unsafe allocator was implemented. | Design §§4,7,17–19 |
| Thread-pool, affinity or NUMA retuning with GPU count | Exclude from normal serving | Pools, affinity and library thread settings remain fixed after setup; issued tasks/bytes vary. This clarifies worker-quota wording, with no claim that thread churn was implemented. A bounded idle worker can sleep without a topology controller. | Design §§2,16,19 |
| Online affine mixed-step cost fitting | Keep A04 optional | Fixed overhead divided by too few prefill rows can produce a shrinking-chunk feedback loop. Compare against simple regime quotas before fitting a live model. | A04 |
| Per-request heterogeneous speculative widths | Keep E08 optional | Potentially useful with heterogeneous acceptance and a binding row budget. First beat a small uniform-width table. Choose widths before fresh draws and include padding, scheduling and fairness. | E08 |
| Route-signature-aware whole-model microbatch partitioning | Keep E09 at trace feasibility | There is no opportunity when the full model can already batch all ready work. Do not manufacture splitting because one QSA operator has a row cap; preserve the larger legal MoE batch. | E09 |
| Predictive PP delay/chunk optimizer | Defer until real PP2 bubbles remain | A dependency-ready scheduler and sensible chunks come first. Published PP results use other models/A100s; second-server traces determine whether a predictor earns its costs. | Research §9; A07; [S5] |
| Mixed prefill/decode CTA scheduler | Keep E17 conditional | Compare ordinary specialized kernels and legal stream overlap first. Shared register/shared-memory reservations can erase the hoped-for overlap. It is a local kernel experiment, not a model-wide runtime. | E17 |
| Genuine DFlash block 2/3/8 execution | Defer additional block implementations | Changing a noncausal block changes interactions, convolutions, masks, proposal probabilities, buffers and graphs. First evaluate block 4 plus legal prefix truncation. | Design §14 |
| Live expert-weight reshuffling/EPLB | Do not add now | Required row grouping is separate. ETP8 has EP1; ETP4/2 need sustained owner-service skew after good kernels/static mapping before moving weights is useful. One-copy moves still need transfers and epochs. | Research §10; Design §§19–20 |
| Creating/releasing immutable tensor copies with activity | Keep copies as prepared profile choices | Packing, staging, graph pointers and readers introduce transition costs. Accept a copy under the existing whole-deployment 1-GB policy; dynamic population is not an initial obligation. | Design §1; Numerical selective-copy table |
| Online PLE transport or compression-algorithm shopping | Do not add | Select a qualified transport/codec at preparation. Bounded read submission, raw codec modes and input-dependent output lengths are ordinary execution within that choice. | A25,E13,E24 |

## Why live KV movement deserves special scrutiny

SC117 uses sparse selected attention. At the native context, a QSA cache has 65,536 completed pooled groups to score, but selected KV attention uses at most 2,048 complete-group members plus three visible tail rows. The canonical pooled-key working set is 16 MiB/cache, 192 MiB across the twelve target caches, or 208 MiB with MTP. These are working-set sizes for complete scans, not unavoidable bytes per committed token.

Changing KV attention ownership does not parallelize that canonical indexer scan. A slow long-context request could therefore trigger a large KV move while leaving its dominant operation unchanged. Profile scoring/top-k, selected KV attention and communication separately before building an ownership controller.

Using the documented two KV heads, twelve target QSA caches and 262,144 positions:

| PP1 target history calculation | Q8 | iso4 |
|---|---:|---:|
| Unique target KV per full session, excluding indexer/MTP |3.187500 GiB|1.593750 GiB|
| Owner-to-four-way redistribution moving three quarters |2.390625 GiB|1.1953125 GiB|
| Transfer if all 24 complete target histories move |57.375 GiB|28.6875 GiB|

The last row applies only if all 24 histories are actually moved. It excludes metadata, staging/retirement, indexer moves and any separate MTP work. A smaller affected set transfers less. These are transfer volumes, not timings or simultaneous duplicate residency.

One unique history exists at every degree. Changing degree does not create a generic storage multiplier; worst-rank balance, page rounding and concentrated method/vision resources can differ. Derive the exact stage ledger, particularly PP2's six target QSA caches per server. A fixed map still needs valid page descriptors, slot generations, exact partial-softmax merges, prefix/restore ownership and correct retirement. Demoting migration does not remove those independent correctness requirements.

Fixed ownership also does not require one invariant local kernel split. Local CTA tiles, query grouping and selected work lists can change without moving retained history. Gathering selected KV to fewer compute ranks is a possible bounded experiment, but it pays per-step reads/transfers and staging; it does not obtain degree-one local-history performance for free. Flash-Decoding illustrates execution decomposition without moving the stored sequence [S6]. LoongServe explicitly addresses movement costs in a different dense-attention/A800 architecture [S7]; NanoCP studies a different data/expert-parallel arrangement [S8]. Neither certifies a live-migration gain for this profile.

If migration is eventually tested, use a conservative payback condition:

`future saved complete-serving time > observation + decision + quiescence + transfer + interference + graph/metadata transition costs`.

Include a later reverse move when the trace requires one. With a measured total transition cost M and saving delta per useful step, a positive delta requires at least M/delta useful steps to repay the move. A crossed concurrency threshold is not a dwell-time forecast. C1→C24 must execute safely immediately from the valid map; migration cannot be a prerequisite for making progress.

The reviewed migration plan's 64 MiB/GPU overlap cap is up to 512 MiB across one server if it is a separate obligation. Removing mandatory migration can release only its actual dedicated liability after accounting for shared arenas. It does not prove an extra integer session or justify reducing device-floor/workspace protections; an adopted migration experiment must restore its complete resource charges.

## Speculation is worth adapting with a small policy

The objective is committed output per complete elapsed round. For an ordinary linear speculative round before stop/length limits, a useful illustrative model is:

`rate(k,x) = [1 + sum(j=1..k) P(accepted prefix >= j | prior information,x)] / T_complete(k,x)`.

The denominator includes draft work, conditioning maintenance, target verification, refresh/recovery, communication, sampling/commit and scheduling. Acceptance probabilities need not be independent. For a batch, compare aggregate committed output over a common wall interval, subject to per-request service constraints. This is an explanation of what to measure; runtime selection can use a prepared table rather than solve an optimization problem each step.

Choose width and batch shape together. Under a 32-row limit, deeper proposals fit fewer sequences per microbatch. Whether a shallower wide batch or deeper microbatches win depends on complete costs, useful acceptance and kernel reuse. No monotone rule such as always reducing speculation at high concurrency is justified. The Inco author report retains gains at C32 on GB300/FP8 [S9]; that is a counterexample to a universal rule, not a V100 prediction.

For the initial DFlash implementation, k=1/2/3 still computes its four-row parallel draft. Truncation saves target verification and related work. Warm k=0 skips drafting while maintaining the selected method's conditioning. Compare it with both positive widths and a separately selected true no-speculation control. Automatic unloading would lose immediate readiness and can require unavailable historical features or expensive reconstruction; keep that in explicit qualified profile transitions.

A cheap useful refinement is terminal consumer liveness. After a verdict proves that request-scoped computation is disposable, omit further speculative refresh/materialization that has no continuation, protected retained compute checkpoint or chosen coherent prefix-cache consumer. Retire issued work normally. A retained session can still need conditioning, and features produced before terminality was known are already-paid work. Stored logical API items alone do not demand a permanent computed-state checkpoint.

## Capacity, caching and tiers need ordinary lifecycle logic

| Mechanism | Recommendation | Exact boundary |
|---|---|---|
| Automatic integer Nmax derivation | Keep at preparation/profile activation | Derive from complete worst-rank full-window state, method/media, graph, execution and overlap costs. It is a deterministic planner, not a per-step load optimizer. |
| Physical page growth and pending rings | Keep | Allocate from reserved typed capacity; valid current length does not enlarge guaranteed admission. |
| Prefix matching and useful coherent checkpoints | Keep high priority | Reuse the deepest complete compatible hybrid boundary. Use a bounded prepared policy for message/chunk/end checkpoints. Avoid an online checkpoint-density predictor initially. |
| Physical sharing/COW of prefix pages | Keep optional as already written | Computation reuse can use bounded private copies/restore without requiring cross-session physical sharing. If sharing is adopted, references/COW/future private growth must be budgeted. |
| Prefix victim selection and reclaim | Keep simple and pressure driven | Reclaim unleased reusable entries before protected continuation. A bounded fixed policy is enough until traces justify another one. |
| Idle retained-session suspension | Keep on demand | Suspend complete idle state to reserved RAM/optional storage when capacity or an explicit lifecycle action requires it. Preserve warmth when capacity is spare. Runnable generation is not routinely rotated through tiers. |
| Dirty-page write-back | Keep | Reuse verified immutable backing and write changed state at suspension. This avoids I/O without a per-token host mirror or a predictor. |
| Restore/source choice | Keep ownership driven | A resident continuation, a compatible leased prefix and an authoritative suspended checkpoint are distinct exact paths. Hits do not change FIFO priority. |
| Restore prefetch and transfer overlap | Keep bounded | Prefetch oldest eligible work into reserved host staging. Limit in-flight bytes/I/O and protect progress. Do not scale prefetch without bound with queue depth. |
| Transfer chunk/concurrency quotas | Simple prepared limit first | Observe PLE/media/decode interference. Add cheap backpressure only when needed; a separate continuous bandwidth optimizer is not an initial obligation. |
| Retention/backing quotas and resident ceiling | Administrator contracts | Preserve accepted claims on reductions; increases require full proof. Low current activity and favorable compression averages cannot enlarge promised GPU capacity. |

SGLang's official cache design distinguishes write-through and pressure-triggered write-back [S10]. vLLM provides ordinary prefix-cache ownership/reclaim machinery [S11]. ProviderOne still needs its stronger complete GDN/PLE/speculative continuation contract. Simplifying adaptive policy does not permit losing protected state.

## Prepared choices and explicit administration

| Decision | Adopted boundary |
|---|---|
| TP/PP/ETP, participating servers and communicators | Select a supported profile at initialization; preserve all six required cases. |
| PP2 layer cut, ETP rank map and weight owners | Qualify offline; hold fixed during ordinary requests. A different measured cut remains a new prepared map. |
| Target/MTP/DFlash/vision weights and immutable extra representations | Prepare and account once; no per-load representation churn. |
| Selected method, checkpoint and numerical/activation policy | Profile selection. Required/default MTP and separate DFlash remain separate qualified profiles. |
| Context/RoPE/YaRN, KV codec and media fidelity | Explicit numerical/service contract. Activity never shortens context or lowers precision/resolution/frame policy. |
| Full resource maximum, arenas and protected reserves | Establish before publishing capacity; vary actual used work inside it. |
| Graph set, compilation and GEMM/packed-kernel tuning | Prepare offline or during qualified activation; reuse by complete identity. Bounded readiness validation is different from repeated exhaustive tuning. |
| NUMA, CPU pools, affinity, SIMD paths and library threads | Detect/measure at setup, bind before allocation and retain during serving. |
| Vision/input/output endpoint placement | Fixed qualified ownership; issue ready work to it. |
| PLE mapped/pinned/direct-I/O transport and optional video backend | Qualified source/hardware-specific choice. Deduplication, legal submission and codec capability dispatch remain ordinary execution. |
| NCCL groups/rail map and optional custom collective table | Initialize once; use NCCL selection or prepared measured tactics. No environment mutation, communicator recreation or runtime schedule search on activity changes. |
| GDN snapshot/replay/deferred representation and flush parameters | Fixed qualified numerical/resource representation; execute required accepted-boundary replay/flushes inside it. |
| Supported no-restart administrator changes | Keep the existing drain/empty-resident in-process activation where appropriate. No arbitrary occupied-state conversion framework is implied. |

NCCL already has an operation/topology-aware cost model and scheduler [S12]. A measured selective override can correct a real crossover; another global serving controller would duplicate responsibility. cuBLASLt similarly recommends reusing algorithm decisions [S3].

## Expert adaptation and source formats

Keep actual-row compaction, grouping and local kernel dispatch. The model's selected expert IDs/gates are authoritative. E05 can test a local task schedule if measured tail waves or stragglers remain. A global weight-moving controller is a different feature.

The inspected SC117 source contains 144 routed tensors: 48 layers × gate/up/down. Each tensor has 512 experts, a common shape and one encoding. Expert source bytes are therefore equal within each layer/projection. Mixed formats across layers/projections do not establish per-expert byte heterogeneity inside a layer. Equal-count per-layer placement is the source-byte-balanced control; actual transformed allocations still require their ledger. Avoid writing a byte-weighted optimizer for heterogeneity this artifact does not have.

ETP8 has EP1, so no competing expert-owner group exists to rebalance. ETP4/2 can show service skew because expert activation/coactivation differs. Measure owner-group critical time, actual rows and how long skew persists before considering a more informed static map. If a static map already suffices, omit online remapping. If live movement is revisited, charge source-preserving transfer/repacking, staging, epoch publication, graph/readers, planning and reversal.

Some EPLB implementations use redundant experts; others permit zero redundant experts [S13]. Neither form is free. The existing whole-deployment decimal 1-GB threshold applies to copies and cumulative induced allocations, with every smaller copy also requiring benefit. Popularity alone does not establish recoverable complete-round time.

## Complete experimental register disposition

Every A01–A41 and E01–E24 entry is covered below. Classification records the adopted scope boundary; it does not accept an experimental kernel or alter its existing proof/performance gate.

| Entries | Adopted scope and qualification boundary |
|---|---|
| A01,A02,A07 | Useful prefill/checkpoint boundary planning and cheap prepared chunk policies. Preserve coherent state and avoid discarded-boundary splits. |
| A03 | Exact dependency-aware prefix identity. Correctness, not another adaptive controller. |
| A04 | Optional online mixed cost fit; simple quotas first. |
| A05,A06 | Prepared mixed/tail graphs and fast kernels, then cheap dispatch. No activity-triggered recapture obligation. |
| A08,A10,A11,A12,A13,A14 | Native indexed/grouped kernels, routing, fusion and measured shape tactics. Runtime actual-row execution; preparation chooses representations and numerical boundaries. |
| A09,A15,A16 | Static storage/alias/codebook-sharding choices. Accept source parity, exact allocations and service benefit; avoid dynamic duplicate banks. |
| A17 | Exact compact sampling/transport and logprob capability dispatch. Never silently change sampling. |
| A18 | Separately qualified proposal/method/numerical profiles. Cheap width within a selected profile remains useful; automatic method substitution is excluded. |
| A19,A20,A21,A22 | Direct history, exact live work, selected grouping and native GDN interfaces. Ordinary input-dependent kernels and prepared dispatch. |
| A23,A24,A26 | Exact PLE preparation/convolution and known-input lookahead. Bounded event-driven I/O; no authoritative token prediction. |
| A25 | Prepared PLE transport choice; bounded submission/deduplication/readahead inside it. No online transport shopping. |
| A27,A28,A29,A34 | Incremental metadata, ownership handoff and lifetime-safe overlap. Events/reader retirement drive execution. |
| A30,A31 | Prepared scratch/positional bounds; actual live ranges and qualified phase generation. No full-window per-session expansion or activity-based position retuning. |
| A32,A33 | Prepared collective/HC-layout tactics. Choose cheap operation-compatible plans; no repeated topology planner or weight movement. |
| A35,A37,A38,A39,A40 | Import/build/range/caller/route/host qualification and diagnostics. Preserve; do not turn every diagnostic into a synchronous control input. |
| A36 | Bounded exact media scheduling and correct cache identity. Preserve fidelity and fixed placement. |
| A41 | Prepared accepted-prefix recovery strategy. Actual replay to accepted boundaries is required state work, not a separate optimization policy. |
| E01,E02 | Optional exact witnesses/certified bounds. Current-query data can reduce work without changing history layout. Stop if rescoring/summaries/certification costs exceed saving. Use bounded scratch and exact overflow handling. |
| E03,E04 | Fixed qualified GDN arithmetic/kernel experiments. Different gates/casts/FP32 rounding need proof; no load-driven precision switch. |
| E05 | Conditional projection-local grouped task scheduling. Trace wave loss first; static/affine task assignment wins where adequate. |
| E06 | Separately qualified exact verifier algorithm. No need for a global adaptive controller. |
| E07,E22 | Optional grammar-support/forced-token exact execution. Actual hard support determines eligibility; finite/error behavior, RNG, stop/logprob and state contracts still apply. |
| E08 | Optional heterogeneous width policy after simple-table controls. |
| E09 | Optional route-aware partitioning only when full-model microbatches are independently necessary. |
| E10 | Withdrawn ETP1 work. Keep excluded. |
| E11 | Optional reuse of shared selected tiles across real immutable shared prefixes. No batch delay or FIFO jump. Equal text length does not prove selected-set overlap. |
| E12,E15,E16 | Static compressed representation or fixed arithmetic/lookup algorithm experiments. Input-dependent tile work does not require online profile switching. Census/proof before serving implementation. |
| E13 | Optional fixed idle-checkpoint codec/path. Raw mode and encoded size may depend on actual bits; reserve worst-case backing. |
| E14 | Optional allocator/VMM feasibility. Address reservation is not physical capacity; avoid a live remapper without a paging bottleneck. |
| E17 | Optional local mixed-kernel task scheduling after simpler mixed execution loses. |
| E18 | Optional fixed distributed indexer. Compare long-history C1 benefit with C24's existing independent-owner balance; no live sharding transition is needed to test it. |
| E19 | Optional actual-draft-ID PLE lookahead. Start immutable reads when IDs are known; measure lead and contention. |
| E20 | Optional prepared NVDEC pipeline; codec/format checks and legal media-job scheduling are ordinary execution. Fidelity and surface budgets remain fixed. |
| E21 | Optional deferred GDN state representation with a fixed qualified L and ring/flush parameters. The required space-triggered flush when h+T > L materializes only already committed records; it is state execution. Required coherent-boundary materialization also preserves retained state. Changing L or early flush cadence can regroup FP32 arithmetic; optional staggering needs measured bursts and numerical qualification. |
| E23 | Optional certified head pruning in proven sampler regimes. Current hidden state/bounds drive an exact kernel; no automatic approximate vocabulary. |
| E24 | Optional fixed bit-preserving active-state codec. Per-block mode selection is codec execution; full raw-capacity slots still prevent a guaranteed admission gain. |

## Minimal implementation policy

Prepare one selected structural/numerical/media profile with a fixed full-window capacity and maximum execution envelope. Retain only a small set of measured, nondominated kernel/graph/batch/width plans. Several counts may use the same plan; that is a valid optimization result.

At a coordinated boundary:

1. Resolve completions, cancellations and output readiness; wake eligible FIFO work immediately when a claim releases.
2. Select ready GPU-admitted sequences and legal rows under service and memory bounds.
3. Choose a prepared batch/width/prefill/media plan jointly using cheap known metadata; use prior completed-round acceptance only where it improves a measured decision.
4. Publish one consistent plan to participating ranks. Device routing handles device-only row populations without a host readback per expert/layer.
5. Execute, commit the exact permitted prefix, retire readers and publish bounded output.
6. Update already-available compact counts; sample timing/critical-path diagnostics asynchronously.

No work waits merely to make a larger batch or adapt to a predicted future arrival. No kernel search, state conversion, allocation-envelope change, graph rebuild, communicator rebuild or weight redistribution is needed for an ordinary active-count transition.

Prune dominated plans offline within each legal validity region: if a tactic never wins after complete cost and gates, remove its dispatch branch while preserving required tail/API coverage. Every retained runtime observation answers a concrete decision question. Prefer ready counts, known lengths/shapes and prior completed-round summaries; keep device-only routing details on device. Sample diagnostic timing independently instead of fetching it synchronously every round.

## Applied change map

The governing design and companions incorporate the following planning changes. Experimental implementations and their performance remain subject to qualification.

| Location | Applied planning contract |
|---|---|
| Design §17 title and opening | Work scheduling and prepared execution selection, with one coordinator and globally agreed plans. |
| Design §17 timescales | Normal steps select ready work, legal widths/quotas and prepared kernel/graph plans. Prepared profile/admin boundaries handle maximum-envelope or graph-set replacement. Live placement changes remain a benchmark-gated experiment. |
| Design §§1,6,10–11,19–22,24; Research §12; numerical companion | Require a qualified fixed unique KV layout first; four-way striping is the first unmeasured control. Live degree conversion/migration, union maps and transition sweeps are excluded from initial obligations. Their complete correctness/resource contract applies if the experiment is adopted. |
| Design §§4,7,17–19 | State that used work varies inside a prepared maximum. Actual capacity/arena changes follow explicit qualified activation; no claims are released because current activity is low. |
| Design §§16,19 | Clarify persistent CPU pools and queue/task/byte quotas. Dynamic thread/affinity retuning is not normal load response. |
| Design §§12,14,17; Experimental §8 | Use a small measured width policy with DFlash fixed-block cost, mandatory warm-zero maintenance and stop/position eligibility. Omit terminal refresh only after proving no continuation or protected compute/prefix consumer. |
| Design §§15,18; Research §12; A32/A35 | Call shape/tactic dispatch ordinary prepared execution. Preserve existing no-live-search safeguards. Full tuning reruns follow changed identity or evidence, rather than every launch or request. |
| Design §20; Research §10 | Keep expert placement fixed in normal operation. Use equal-count source-byte-balanced SC117 mapping and actual-row scheduling first; any coactivation map or live rebalancer needs a demonstrated service bottleneck. |
| Experimental A04,E08,E09,E17,E18,E21 | Optional experiments follow simple controls. E21 uses a fixed qualified L with required space/coherent-boundary materialization of committed records; optional early staggering needs measured bursts and numerical qualification. |
| Design §17; Experimental §§1,13 | Prune dominated plans offline, preserve legal coverage and tie each cheap observation to a concrete decision. |
| All companions | Adopted scope synchronized; all 65 experimental IDs, source pins, six structural cases and evidence/resource limits retained. |

## Qualification and ordering

First profile the actual installed ProviderOne build and identify the complete-round critical path. For a latency optimization at fixed workload and capacity, bound the complete-round work and interference it can remove; capacity and admission benefits require their separate resource proof. Then establish the optimized fixed control: correct compact history, native kernels, useful batch/chunk sizes, prepared graphs/tactics and full state commitments.

Run matched C1/2/4/8/16/24 A/B measurements, full actual capacity and the release integer sweep. Preserve native context, selected KV/method/media, genuine histories, and the one-active-with-other-full-residents case. Include pure prefill, pure decode, mixed arrivals/media, cache hits/misses, FIFO backlog and idle restoration. Never credit a cache hit as computed cold prefill or rejected drafts as output.

Compare three things separately: the best fixed plan, the cheap dispatch policy, and any expensive stateful experiment with all transition costs included. Vary history and acceptance as well as C. Include sudden C1→maximum changes and oscillating/short-lived regimes. Every per-count decode nonregression and latency/fairness gate remains in force.

Prepare the simplest winning choices first. Only then test optional online fits, heterogeneous allocation, route-aware grouping, distributed indexers or migration when traces establish remaining opportunity. No paper supplies this SC117/V100 engine's winning width, graph bucket, owner degree or throughput; those are qualification outputs.

## Sources and evidence boundaries

The four current governing files are the requirements baseline:

- [ProviderOne V100 Adaptive Inference Design](https://chatgpt.com/library?file_id=libfile_409ab65a4194819194b7337392258936&account_id=personal)
- [Experimental Optimization Candidates](https://chatgpt.com/library?file_id=libfile_f592a45084988191a3072db8ec831c85&account_id=personal)
- [ProviderOne V100 Parallelism Research](https://chatgpt.com/library?file_id=libfile_03ac2f10310081919a03e292e09f3b8e&account_id=personal)
- [SC117 Memory and Parallelism Options](https://chatgpt.com/library?file_id=libfile_024132db763081918939c76377983933&account_id=personal)

External research supports mechanism-level decisions. Reported hardware/model speedups retain their own scope. The transfer calculations and equal expert-stride check use the pinned SC117 geometry and the inspected 144 routed tensor records, not measurements of executed ProviderOne kernels.

| Source | Applicable evidence |
|---|---|
| S1 [Sarathi-Serve OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal) | Chunked prefill and bounded mixed scheduling address a throughput/latency tradeoff. |
| S2 [vLLM Dynamic Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/) | Verification cost changes with batch and speculative length; a small policy table is a concrete implementation precedent. |
| S3 [cuBLAS 12.9 heuristic cache](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#heuristics-cache) | Query and reuse algorithm decisions; repeated heuristic work can cost host time. |
| S4 [NVIDIA graph dynamic patterns](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/handling-dynamic-patterns.html) | Prepared buckets, padding, eager/static portions and dynamic inputs have explicit cost/lifetime tradeoffs. Its illustrative pool costs are not a universal multiplicative allocation rule. |
| S5 [Revisiting Pipeline Parallelism for LLM Serving OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/hwang) | Online stage imbalance can justify dynamic chunks/grouping on its Qwen 2.5/A100 setup. |
| S6 [Flash-Decoding author explanation](https://crfm.stanford.edu/2023%2F10%2F12/flashdecoding) | Local sequence decomposition and softmax-state merging need not move the underlying stored sequence. |
| S7 [LoongServe SOSP 2024](https://arxiv.org/html/2404.09526v2) | Elastic parallelism must address reactive movement costs; its dense-attention/A800 mechanisms are not a drop-in sparse recurrent path. |
| S8 [NanoCP](https://arxiv.org/html/2605.21100v1) | Dynamic CP can be useful in a different data/expert-parallel design; it does not expand this frozen target matrix. |
| S9 [IncoAI exact-family draft model card](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) | Block/draft contract and source-specific GB300 evaluations; high concurrency does not universally eliminate speculation's benefit. |
| S10 [SGLang HiCache system design](https://docs.sglang.ai/advanced_features/hicache_design.html) | Write strategies, rank coordination and pressure-driven backing support ordinary bounded cache lifecycle reasoning. |
| S11 [vLLM prefix-cache design](https://docs.vllm.ai/en/latest/design/prefix_caching/) | Bounded state ownership/free-queue/reclaim primitives without a learned cache controller. |
| S12 [NVIDIA NCCL tuning](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/) | Existing operation/message/topology-aware selection and narrow prepared tuning overrides. |
| S13 [vLLM expert parallelism/EPLB](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) and [DeepSeek EPLB](https://github.com/deepseek-ai/EPLB) | Physical balancing has load/planning/mapping costs; not every implementation requires a permanent extra expert copy. |
| S14 [TETRIS ACL 2025](https://aclanthology.org/2025.acl-long.1598/) | Heterogeneous speculative allocation is a genuine optional policy direction, with its own assumptions. |
| S15 [Exact speculative decoding ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html) | Verification is a sampling algorithm; altered widths/proposals must preserve its conditional laws. |
| S16 [ReplaySSM author paper](https://arxiv.org/html/2610.13778v1) | Deferred materialization changes state dataflow and carries numerical/flush tradeoffs; it is not a load-driven precision permission. |
| S17 [Linux NUMA policy](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html) | Thread/allocation policy and existing page placement are distinct; retain setup-time locality. |
| S18 [Pinned selected SC117 artifact](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/tree/90a3350cd800ec1fe0d1660a38414ad9e985782f/IQ3_S) | Source directory/geometry and fixed tensor encodings; header arithmetic does not establish runtime allocation or speed. |

The reviewed revisions remain the historical evidence baseline. The updated governing design and companions apply the adopted scope while preserving independent correctness, retention and qualification requirements. The source alignment inspects the pinned implementation and published evidence; it runs no ProviderOne/V100 or other hardware benchmark and does not qualify an installed deployment.
