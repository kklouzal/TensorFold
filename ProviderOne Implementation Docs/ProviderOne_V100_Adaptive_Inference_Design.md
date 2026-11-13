# ProviderOne V100 Adaptive Inference Design

**Current platform:** one or two Supermicro SYS-4029GP-TVRT servers, each with eight 32 GiB V100 SXM2 GPUs; one eight-GPU TP stage per server.  
**Initial target:** non-Swift `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF`, `IQ3_S`.  
**Required:** QSA MTP, separately selectable IncoAI DFlash2, image/video → text.  
**Scope:** SC117/V100 delivery contract, shared ProviderOne architecture and preservation of selected implementations across owned hardware; new model/format/device qualification follows Section 1.  
**Basis:** discussion and startup receipt dated 2026-10-25 04:16 UTC.  
**Review:** 2026-11-05 UTC; end-to-end source/architecture audit of `python-0.6` at `c83187878423388715571d194536eba836a64804`, retaining the adopted prepared-plan/fixed-placement scope and separate qualification gates.  
**Evidence:** supplied observations, artifact/header measurements, arithmetic estimates and V100 workload qualification are distinct (Section 25).

## 1 Governing operating contract

Build ProviderOne as an independent inference engine. The immediate new delivery is fully qualified Qwen3.8-Flash-Next/SC117 on V100; preserve the fork’s selected existing implementations and their scoped deployment contracts while developing it. Administrators select target/runtime weights, context/RoPE, KV codec, speculative artifact/method and service limits; ProviderOne computes full-context capacity and selects qualified workload plans. Initial profiles use SC117 IQ3_S, shared-Q8 MTP and IncoAI DFlash2. Additional artifacts use the same contracts.

**Mission:** maximize qualified GPU-resident sessions at the selected context, weight/KV quantizations, speculative method and media envelope, then maximize aggregate committed decode tokens/s with all sessions active. Select ready work and prepared execution plans for actual GPU-admitted activity, preserving context, quality, fairness and no-restart commitments. Initial serving uses fixed unique history placement; live ownership redistribution remains a benchmark-gated experiment. Serve excess requests through bounded FIFO admission; qualified RAM/storage suspension preserves idle retained conversations (Section 17). Report the measured context/capacity/throughput tradeoff; never silently sacrifice one to improve another (Section 5).

### ProviderOne direction, implementation freedom and roadmap

Extract the highest useful concurrency and end-to-end prefill/decode performance the selected hardware can sustain, including small repeatable gains. No arbitrary percentage cutoff dismisses a gain that survives measurement noise and the complete quality, capacity, fairness and lifecycle gates. Resolve the context/capacity/latency tradeoff explicitly; microbenchmarks and theoretical utilization alone cannot justify a change.

ProviderOne may remove or replace inherited schedulers, loaders, cache/checkpoint managers, model wrappers, kernels, orchestration, dependencies, module layouts and build systems. Python/PyTorch, MLX, upstream flags, internal APIs and TensorFold-specific abstractions are implementation choices, not preservation requirements. Rework the directly owned path and remove superseded code/dependencies; do not retain a second legacy serving stack or compatibility shim merely to ease detachment. Reuse code only where it earns its place. Preserve this design’s required observable behavior, model semantics, OpenAI-compatible interfaces and active session commitments through qualified development/deployment boundaries. Retain source provenance and applicable third-party notices; historical repository links and measured paths are not renamed evidence.

| Delivery phase | Scope and exit boundary |
|---|---|
| SC117 on V100 — current priority | Complete the selected SC117 model, QSA MTP, separately selected Inco DFlash2, image/video, Q8/iso4, FIFO/prefix/tiered state, no-restart scheduling/prepared-plan selection, fixed unique KV placement and Section 20’s six required configurations. Meet the measured correctness, capacity and performance gates before broad model rollout. Directly refactor inherited structure now where it impedes this work. |
| Existing selected models/backends — throughout the transition | Carry forward source-selected implementations, independent oracles and recorded deployment contracts, especially the owned GB10 paths. Revalidate affected shared changes against each affected profile’s accepted baseline and evidence scope. Existing support is inventoried; no existing profile is silently reset or represented as newly qualified by this document. |
| Mainstream Hugging Face models and checkpoint encodings | Complete missing family/format coverage and qualify new device/model/recipe pairings, including GGUF/safetensors. Existing family implementations remain starting points. The requested list below is representative, not a permanent product limit or a universal-support claim. |
| Hardware specialization and expansion | Preserve selected Blackwell implementations while delivering SC117/V100. Optimize concrete profiles for owned GB10 and RTX 4090 hardware; add missing Blackwell, Ada or Ampere paths only with their own device/operator/build/resource and end-to-end evidence. New SC117 pairings and further devices follow the current delivery priority; existing deployment preservation applies throughout. |

Release claims are deployment-specific. One-server delivery can proceed after all three PP1 profiles and required SC117 feature gates pass; the three PP2 profiles remain required work, explicitly pending actual two-server qualification. This does not waive the SC117-first roadmap or any correctness, capacity, quality or performance gate.

Requested follow-on model labels: **DeepSeek 4.1 Flash; MiniMax 3.1 Flash; Qwen 3.8 27B; Qwen 3.6 35B A3B; Gemma4 31B IT; Gemma4 26B A4B IT; Gemma4 E4B; Gemma4 E2B**. Bind each to an exact publisher/repository, immutable revision, configuration, tokenizer/template, weight recipe and any compatible vision/speculative companion when that work begins; these user-specified labels are not independently verified artifact identities. The branch already contains implementations and recipes for several requested families; each exact pairing keeps its own current status. Support must include full inference, state/cache/continuation, exposed API behavior, quality and load qualification, not just successful weight loading.

Separate only the contracts needed by real implementations: source container/recipe decoding; model-family operators and typed persistent/recovery state; hardware-specific kernels, packing, communication and graphs; and the shared admission/resource/API lifecycle. Reuse orchestration where contracts match while specializing hot execution freely. Do not force a universal intermediate tensor layout, scalar dispatcher per token, backend emulation or a framework around hypothetical models. Add a family/backend boundary when a concrete implementation needs it.

### Shared architecture and cross-hardware optimization

Share behavior, ownership and work elimination where their contracts match. Specialize the actual compute, representation and transport at preparation boundaries; one semantic serving path can have several selected device implementations. Architecture-specific kernels and distinct immutable deployment builds are valid specializations. A common lowest-capability kernel, universal tensor layout or extra per-token polymorphic layer is not a portability requirement.

| Responsibility | Minimal ownership and dependency boundary |
|---|---|
| API, admission and lifecycle | Shared principal/request identity, bounded FIFO/credits, logical records, cancellation, streaming publication, output backpressure and retention policy. Keep one authoritative owner for each claim and commit; device/model adapters provide concrete execution and state completion. |
| Container and artifact preparation | Neutral bounded file/header/shard/metadata validation and content identity. Decode the actual container/recipe, then select the model family; a generic GGUF read must not require Flash-Next or import a CUDA runtime. Keep model-specific transforms with the family/recipe. |
| Model family and speculative method | Exact operators, positions, masks, typed persistent/recovery state, proposal/verification/commit semantics and source mappings. Different families can use different state geometry and storage. |
| Device execution | Native operator families, packing/layouts, launch/graph plans, workspace and communication/host-memory behavior. Bind selected implementations and limits before execution; keep device-only routing and intermediate state on device. |
| Resource and evidence contracts | Shared accounting rules, identity/receipt formats and qualification procedures, with device/profile-specific limits, allocation backends and measurements. Reuse matching proofs; retain actual representation and lifetime differences. |

Foundational container/metadata helpers stay independent of device and family modules. Load native dependencies only for the selected backend. Refactor verified duplicated common policy at its real owner, including the current generic helpers imported through CUDA/Flash-Next; do not create a large registry or universal intermediate representation to perform that move. Retain the selected specialized implementations and independent oracles until their replacements pass the affected gates. A whole-language/runtime rewrite is justified only by an identified contract or measured bottleneck.

Carry forward cross-hardware work elimination already present in the branch: bounded/lazy metadata and format checks before weight allocation; content hashing shared by ranks where the storage contract permits it; reusable prepared transformations and kernel artifacts; exact dependency-keyed prefix reuse; bounded admission/input/output credits; compact device results and sampling; and host-only advisory status. Keep the current shared Responses/Conversations logical store and dispatch rather than creating a separate ProviderOne API implementation per device. Complete computed-state preservation remains a different contract from logical API records.

For each shared optimization, identify all affected released profiles and call sites, compare against their accepted same-profile baselines, and verify their numerical, resource, lifecycle and end-to-end behavior. Local kernel gains need integration evidence; a hardware-local change need not become global policy. Existing owner-accepted deployment tradeoffs remain scoped to that profile; they do not grant a general regression allowance or change SC117’s capacity-first and decode-nonregression gates. Required new profiles remain pending until qualified. Unavailable future hardware does not block an independently complete V100 release or justify claims about that hardware.

### ProviderOne source and release handoff

Use the owned `python-0.6` lineage as the implementation starting point. This review excludes all other branches. Record the exact starting commit, local dirty patch, installed package/native origins and immutable image identities; refresh the source-to-design status map when that branch changes. Historical upstream/source snapshots remain provenance, not implementation instructions.

Maintain one canonical implementation design in the ProviderOne repository. The inspected branch’s root `TensorFold_V100_Adaptive_Inference_Design.md` and `docs/v100-adaptive-inference-design.md` are equivalent apart from title whitespace and lag this governing document. At repository handoff, install the current contract at one maintained location; make any second location a link or mechanically generated, checked projection. This document task does not modify the repository or silently resolve an installed deployment.

Rebrand the owned package/imports, native extension names, CLI/entry points, image/deployment names, updater/remote targets, help/recipes/API discovery, cache/artifact identities and maintained documentation coherently. Update actual callers and packaged native sources; remove superseded names/paths directly once verified. Preserve third-party licenses, attribution and historical immutable evidence. Cache reuse requires matching semantic/source/representation/build identity; a product-name change alone neither proves compatibility nor requires recomputing identical content. Test the built installed artifact outside the checkout, with no editable overlay, including CLI discovery, native module origins, readiness, shutdown and the advertised endpoint/model capabilities.

All SC117 geometry, HC/QSA/GDN/PLE state, MTP/DFlash requirements, 262,144 baseline, Q8/iso4 codecs and Section 20’s TP/ETP/PP matrix are **current model/hardware profile requirements**, not universal properties of every future model. Dense models have no routed-expert ETP axis; models without a compatible trained speculative companion must not be forced into SC117’s method. Future attention/recurrent/state architectures, context limits, codecs, degrees and residency policies need their own explicit resource and numerical contract. These roadmap permissions do not expand the present SC117 configuration backlog, authorize active paging now or weaken an admitted commitment.

For each released model × container/quantization recipe × runtime representation × hardware × method profile, record requested, implemented, qualified and effective status, measured maximum concurrency and per-count performance. Validate geometry/divisibility, exact operators/state, finite ranges, long context, tools/media where exposed, prefix/restore/branch/cancellation, graphs and full peak allocations. Reuse evidence only for demonstrably identical contracts. The C1/2/4/8/16/24 milestone method and priority continue where capacity permits; always measure actual maximum, retain smaller-count results and record infeasible points. SC117’s absolute speed targets and historical byte formulas do not certify another model or device.

### Settled requirements

| Area | Governing requirement |
|---|---|
| Hardware | Optimize for this server's measured NVLink/PCIe/NUMA and SM70 behavior (Section 2). |
| Target | Non-Swift SC117 IQ3_S; qualified mixed-format import; TP8 control with all eight ranks available at low load. Implement and qualify Section 20’s explicit TP/ETP/PP profile matrix in the same engine. |
| Residency | One distributed target is the control, with at most one selected speculative method. Optional target/expert/endpoint/vision or other copies require the duplication policy below; ≥1 GB net added across the whole selected deployment receives exceptional scrutiny. Charge all copies and bounded activation/communication/recovery storage, plus any separately adopted migration overlap. Qualified no-speculation transition releases both speculative systems. |
| Context | Equal administrator-selected windows shared by expanded text/media, retained conversation and reasoning/reply. Native 262,144 rows; required RoPE/YaRN extension needs separate row/coordinate qualification (Section 6). |
| Capacity | Compute the complete worst-rank integer maximum; 20/24/32 and power-of-two counts are not fixed limits (Sections 7–8). |
| Speculation | Required/default shared-Q8 QSA MTP; separately selectable IncoAI DFlash2; explicit no-speculation reference/admin profile cannot replace MTP implementation (Sections 12, 14). |
| KV | Ordinary precisely specified Q8 or RotorQuant `iso4` K/V. No unspecified native RotorQuant Q8. RotorQuant first; SAW later (Section 13). |
| Histories | Unique canonical attention KV/pooled indexer and bounded raw pending state in VRAM during execution; optional read copies follow the duplication policy and cannot increase guaranteed capacity. Qualified idle retained sessions may suspend to RAM/storage (Section 17); raw-history archiving stays off. |
| Multimodal | Required text/images/video → text, pinned vision/processor, bounded execution and separate MTP/DFlash qualification. Native audio/media generation excluded (Sections 2, 16). |
| Adaptation | Bounded ready-work scheduling and prepared batch/kernel/graph/proposal choices inside one qualified maximum envelope; FIFO admission and demand-driven idle-session tier transfers. Live KV redistribution is optional (Sections 10–18). |
| Restarts | No container/worker/CUDA-context restart. Non-live settings stay at qualified maximum-concurrency values during adaptation or await a supported administrator in-process transition (Section 19). |
| History placement | Qualify a fixed unique KV layout first; four-way striping is the initial control, not a measured winner. Live degree changes/migration carry no initial development or release obligation; any later promotion must preserve every admitted lifetime/context commitment (Sections 10–11). |
| Simplicity | Refactor one serving path directly; initially one homogeneous numerical/speculative resident profile. No simultaneous speculative systems, unqualified GPU history replication or unnecessary framework layers; Section 17 bounds cross-tier checkpoint copies. |
| Evidence | Complete accounting and representative execution support capacity/performance claims; header arithmetic, startup allocation and live qualification remain distinct (Sections 21, 25). |

SC117 capacity requires reconciled post-import target/method/vision, state, recovery, execution and placement allocations. Section 8 gives geometry-only cross-checks; 20-Q8/32-Q4 are reference points, not admission promises or ceilings.

### Duplication and VRAM tradeoffs

One-copy placement is the control, not a blanket prohibition. Permit experiments with named tensor/endpoint/vision copies, alternate compute representations and read caches within Section 20’s supported matrix. This policy does not authorize another target layout, additional model/expert banks or altered expert routing. Permission to test is not acceptance: retain a copy only when verified complete-serving gains justify its measured VRAM cost under the mission and qualification gates.

**Escalation threshold: ≥1 GB of net added VRAM across the entire selected deployment**—all eight GPUs for PP1 or all sixteen for PP2—not per GPU or per server. Use decimal GB: 1,000,000,000 bytes ≈0.931323 GiB; report exact bytes and GiB as well. Below this threshold, weigh verified performance gains carefully against the added memory. At or above it, scrutinize the proposal with extreme prejudice: require a compelling, repeatable benefit, explicit capacity/latency tradeoffs and comparison with feasible lower-copy or no-copy approaches. This is a higher evidence bar, not a hard cap or automatic rejection. Apply it to each candidate and the cumulative accepted duplication package; several smaller copies cannot evade it by being evaluated separately.

Use matched, fully specified control/candidate profiles differing only in the declared optimization and its required execution changes. Publish gross added copies, any real released allocations and the net increment in both steady residency and concurrent peak, as whole-system totals and per-rank deltas/peaks. Escalate if either aggregate increment reaches the threshold. Include actual encodings, decoded/repacked banks, scales, metadata, alignment, graph/private pools, communication/staging and other induced resources; do not count existing replicas again, subtract unrealized savings or charge host mappings as GPU copies. Preserve one authoritative mutable state and correct replica validity/invalidation and retirement. All ordinary worst-rank, future-context, workspace/floor and lifetime-overlap reservations still apply.

Require matched, repeated C1/2/4/8/16/24 prefill/decode A/B measurements, extended through the qualified maximum, with identical context, quantizations, selected method, media and retained-history commitments. Measure committed tokens/s, complete-round latency, mixed-load fairness and actual maximum resident capacity; isolated kernel gains are insufficient. Preserve numerical/quality gates, every admitted commitment and decode nonregression; higher concurrency keeps priority 1<2<4<8<16<24. A small copy is not automatically worthwhile, and a large speedup does not hide lost capacity. Record the chosen capacity/performance tradeoff explicitly; a faster reduced-capacity profile cannot silently replace the capacity-first default. Name accepted copies, byte budgets, evidence and fallback in the effective profile/receipt. Otherwise keep the control.

### Boundaries of the design

Preserve the target's sparse/indexed attention, recurrent equations, masks, pooling, positions, and sampling semantics while changing storage ownership and scheduling. Qualify quality-changing codecs separately.

No-restart applies to serving adaptation and supported administrator changes. Hardware/driver faults and software deployment have separate lifecycles; universal GPU/process fault recovery without restart is not promised.

Preserve the selected model’s tokenizer, operators and qualified numerical behavior; declare every conversion and compute choice in the effective profile. ProviderOne emerges from the heavily diverged owned TensorFold `python-0.6` fork, with upstream detachment and rebranding planned. Section 2 binds the inspected source snapshot; local dirty state, installed builds and actual execution remain separately bound. Older TensorFold snapshots retain their historical evidence scope. Their internal APIs impose no compatibility obligation. Source implementation, recorded tests, installed-artifact verification and actual hardware qualification remain distinct statuses.

[Experimental_Optimization_Candidates.md](Experimental_Optimization_Candidates.md) is the companion experimental backlog. Its source findings inform correctness gates; its proposed mechanisms remain conditional. Promote an experiment only through the numerical, resource, lifecycle and performance gates here, then update the effective implementation contract and receipt. Its revision-21 design references retain their original section meanings; this design governs subsequent corrections.

## 2 Fixed server and model profiles

### Authoritative implementation snapshot

The implementation audit reads only [`kklouzal/TensorFold`, `python-0.6`, commit `c83187878423388715571d194536eba836a64804`](https://github.com/kklouzal/TensorFold/tree/c83187878423388715571d194536eba836a64804), whose commit timestamp is 2026-11-05 02:41:32 UTC. The repository tree was complete at that pin. Current-source statements below refer to this snapshot, not every later branch head or the installed machines. The audit inspects task-relevant implementation, callers, maintained tests/contracts and build/deployment evidence; it performs no V100, GB10 or RTX benchmark.

| Surface | Starting status and implementation consequence |
|---|---|
| Common serving and state lifecycle | Source implements bounded FIFO/input-output credits, standard Responses/Conversations logical records, a common prefix/retained-state catalog and RAM save/restore. Preserve those owners and verify the concrete ProviderOne integration; logical records alone do not qualify complete GPU continuation. |
| V100 capacity/history/profile work | Compact unique history, full-window capacity planning, exact identity/profile binding, native preparation and state/control mechanisms exist in source with scoped CPU/compile controls. They are starting implementations, not a qualified eight-V100 execution envelope. |
| Required TP8/PP/ETP configurations | Section 20 remains the delivery contract. Existing smaller-world providers, flags, metadata or CPU tests do not establish all six required target/MTP expert and pipeline execution paths. Complete and qualify missing degrees, stage ownership and multi-server transactions. |
| Selected GB10 Gemma deployment | The repository records an accepted Gemma4 CT/ModelOpt deployment with prepared prompt work, Delta4/R16, N64 experts and a head-only W4A16 DFlash policy; its companion transformer remains FP8 and KV BF16. Preserve the exact selected profile and evidence limits; neither the head policy nor those numerics automatically apply to SC117/V100. |
| Experiments and historical receipts | Default-off/unselected A8, draft-graph, persistent-cache or other candidates retain their explicit status. A historical CPU/native/model pass or reported cutover is evidence only for its recorded source/image/workload. No old suite total is assigned to this HEAD by implication. |

The branch’s [V100 validation record](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/docs/v100-adaptive-validation.md) distinguishes source/CPU milestones and still-pending physical allocation, actual model/state/quality and performance gates. Its later startup diagnostic does not establish ready/model-output acceptance. The [current README](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/README.md) and [GB10 qualification record](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/docs/gemma4-cuda-validation.md) record selected production policies, disclosed tradeoffs and remaining API/cache/media checks. Reuse matching evidence and completed source work; repeat affected checks after integration or changed identities, rather than restarting development or declaring the combined new deployment qualified.

### Physical platform

Eight V100 SXM2 GPUs use the X10DGO-SXMV-P carrier and X11DGO-T host board, a cube-style NVLink mesh, and switched PCIe 3.0 CPU-GPU paths. There is no NVSwitch. The advertised 300 GB/s NVLink bandwidth is aggregate bidirectional per GPU, not per pair or host transfer. Sources: [Supermicro specification](https://www.supermicro.com/en/products/system/4U/4029/SYS-4029GP-TVRT.php) and [manual MNL-2115, GPU Add-on Module/GPU Mapping](https://www.supermicro.com/manuals/superserver/4U/MNL-2115.pdf).

TVRT has shared PCIe uplinks and switches serving GPU pairs, not eight independent host x16 links. The product page's generic TRT/TRT2/TRT3 PCIe chart does not establish TVRT topology. Inventory installed CPUs/DIMMs, NUMA placement, and physical GPU enumeration to choose bindings and verify links.

Required platform manifest:

- GPU UUID, device index, logical rank, memory capacity, ECC state, and supported peer access.
- NVLink link state and pairwise transfer measurements.
- PCIe switch/root association and NUMA affinity.
- Actual CPU topology, memory channels/population, and available host RAM.
- Driver, CUDA, compiler, framework, NCCL, kernel-library, and profiler versions.
- Power limits, sustained clocks, temperatures, throttling, and link errors during performance runs.

Record `nvidia-smi topo -m`, NVLink status, CPU/NUMA inventory, and transfer/collective measurements. Physical assignment must keep logical KV-head groups consistent with weight shards.

### SM70 execution stack

Use the newest stable CUDA 12.x and mutually compatible SM70 libraries/tools; build and qualify native SM70 code. CUDA 13 removes offline compilation below 7.5; NVIDIA directs older GPUs to CUDA 12.9/R580. Resolve the newest appropriate R580 patch. Pin packages, source revisions, flags, host compiler, linked libraries and loaded paths. Installation/CUDA compatibility alone does not qualify V100 execution. Sources: [NVIDIA architecture guidance](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/); [CUDA archive](https://developer.nvidia.com/cuda-toolkit-archive).

Volta Tensor Cores use FP16 matrix operands with FP16 or FP32 accumulation. Packed Q4/Q8 needs an explicitly supported compute path; do not assume newer GPUs' native INT4, INT8, FP8, or BF16 Tensor Core paths. Source: [Volta tuning guide](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html).

Use Volta-capable profilers and verify every dependency/kernel backend. FlashAttention or modern FA3/FA4/Triton recipes do not establish SM70 support for this sparse, rotated, quantized attention path.

The externally researched component table below retains its 2026-10-29 basis. The inspected branch now supplies an exact SM70 source-build starting control in its [source pins](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/deploy/v100/source-pins.json) and [source foundation record](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/deploy/v100/source-foundation.md): CUDA12.9.2/nvcc12.9.86, PyTorch2.14.1 with effective owned Torch `04cf2557f315a350e7eea7ab064288def65db75a`, CCCL3.5.0, NCCL2.32.3, application CUTLASS4.8.1 and cuDNN9.10.2.21. The owned wheel/SDK and native CPU/ABI/Store controls are recorded with exact hashes; they are not an eight-V100 combined-stack/model/performance qualification. Keep the framework’s separately pinned internal dependencies coherent. Start from these completed controls, resolve remaining compiler/driver/device gates, and compare newer eligible candidates before replacing them. The pinned R580 driver is a candidate with installation unperformed, not an observed server configuration.

| Component | Candidate/control revision | SM70 condition |
|---|---|---|
| CUDA Toolkit | 12.9.2, Update 2 | Newest published 12.x toolkit; build native `sm_70` objects with its supported host compiler |
| cuBLAS and cuBLASLt | 12.9.2.10 | Recorded CUDA 12.9.2 component/build candidate; resolve later eligible 12.x patches and qualify actual FP16/FP32 algorithms, layouts and workspace bounds |
| CUTLASS C++ | 4.8.1 | Use retained SM70 C++ TensorOp/WMMA components; the name of its 2.x kernel API does not require a 2.x package |
| NCCL | 2.32.3, release tag `v2.32.3-1` | Released CUDA 12.x candidate; verify the exact release build's SM70 objects and V100 collective/graph execution |
| CCCL, CUB, Thrust, libcu++ | 3.5.0 required-primitives SM70 build candidate | CUDA 12.x compilation remains supported; SM70 is outside regular upstream validation, so qualify every required primitive and the coherent framework build |
| cuDNN, when required by the framework or a selected operator | 9.10.2 for CUDA 12.x | Its matrix supports Volta with CUDA 12.9; 9.11.0 removes Volta |
| Nsight Compute | 2025.2.1, build 2025.2.1.3 | 2025.3 removes Volta |
| Nsight Systems | 2025.3.2, build 2025.3.2.474 | 2025.4 and later remove Volta |
| Compute Sanitizer | 2025.2.1 release, CUDA package 12.9.79 | 2025.3 removes Volta |
| PyTorch, while retained by the current implementation | 2.14.1 source-build candidate against the selected CUDA 12.9 stack | Its published Volta wheel uses CUDA 12.6; inspect source-build and transitive dependency compatibility instead of assuming the wheel supplies this stack |

Toolkit/library versions: [CUDA 12.9.2 component manifest](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-toolkit-release-notes/index.html), [cuBLAS patch releases](https://docs.nvidia.com/cuda/cublas-patch-release-notes/), [CUTLASS releases](https://github.com/NVIDIA/cutlass) and [4.8.1 SM70 functionality](https://github.com/NVIDIA/cutlass/blob/v4.8.1/media/docs/cpp/functionality.md), [NCCL release](https://github.com/NVIDIA/nccl/releases/tag/v2.32.3-1) and [exact-release build rules](https://github.com/NVIDIA/nccl/blob/v2.32.3-1/makefiles/common.mk), [CCCL 3.5.0 release](https://github.com/NVIDIA/cccl/releases/tag/v3.5.0) and [exact-release architecture policy](https://github.com/NVIDIA/cccl/blob/v3.5.0/README.md), [cuDNN support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html) and [9.11 removal](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.11.0/release-notes.html). The mutable cuBLAS notes also list patch 12.9.3 for Toolkit 12.9 Update 2; a release-note label is not the exact package build ID. Resolve availability, SM70 support and the combined manifest before replacing the recorded control. The selected NCCL source includes SM70 under CUDA 12.x; this does not replace exact-release binary inspection.

CCCL 3.x omits older-toolkit-only architectures from regular testing without disabling every path. Qualify 3.5.0 scan/sort/compaction/reduction and warp/block primitives for numerics/selection, graphs/streams, bounded scratch and whole-step performance; resolve vendored headers coherently. If it fails, select the newest earlier passing release. CCCL 2.8.5 is a comparison baseline, not a ceiling or LTS claim. Pin one deployment release; no serving-time library fallback. Sources: [CCCL 2.8.5](https://github.com/NVIDIA/cccl/releases/tag/v2.8.5); [issue 6226](https://github.com/NVIDIA/cccl/issues/6226); [clarification 6229](https://github.com/NVIDIA/cccl/pull/6229).

Tools/framework evidence: [Nsight Compute history](https://developer.nvidia.com/nsight-compute-history) and [2025.3 support boundary](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_3), [Nsight Systems boundary](https://docs.nvidia.com/nsight-systems/ReleaseNotes/), [Sanitizer history](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html), [NVIDIA published package builds](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/), [PyTorch 2.14.1](https://github.com/pytorch/pytorch/releases/tag/v2.14.1), and [PyTorch CUDA 12 packaging policy](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432). A package index identifies a build; it does not select the server's OS. Source builds must resolve framework-vendored headers and transitive libraries coherently, rather than overriding individual libraries inside an incompatible wheel.

Keep dependencies immutable during serving. Library/compiler/driver replacement follows Section 1's deployment lifecycle and invalidates affected algorithm, graph, numerical, resource and performance qualification. Adaptation selects qualified plans; it never upgrades dependencies.

PyTorch [2.15.0](https://github.com/pytorch/pytorch/releases/tag/v2.15.0), released 2026-10-28, drops the legacy CUDA 12.6 wheel line described above. Its availability does not prove or disprove a CUDA 12.9/SM70 source build. Keep 2.14.1 as the recorded source-build control; assess newer source releases against the actual fork and full dependency manifest before selecting one. In particular, requalify its changed NCCL backend/device binding, communicator allocation and AOT interfaces. No framework upgrade or SM70 source-build qualification is implied by this review.

### CPU, NUMA and transport placement

Initially retain one worker per GPU and one distributed coordinator, with persistent bounded CPU service/media pools; other process models require qualification. Pool sizes, affinities, NUMA and library thread settings remain fixed during ordinary serving. Queue-driven work issuance is the baseline; vary optional task/byte quotas only where measured contention justifies it. Record physical cores/SMT siblings, allowed CPU/memory cpusets, GPU PCI bus/NUMA affinity, memory channels and active NIC/NVMe paths. Bind launch/progress, lookup and media workers deliberately; reserve CPU service for communication, tokenization and output instead of letting eight workers each consume every core. Set framework/OpenMP/BLAS/media thread budgets before worker initialization; measure SMT, spinning and blocking policies during preparation rather than assuming more threads help.

Set NUMA task policy in each allocating thread or before creating workers/threads; existing sibling threads retain their policy. Apply it before allocating/first-touching anonymous staging/work arenas and bounded pinned pools near the intended GPU path. Verify placement with `numastat` and `/proc/PID/numa_maps`; CPU binding does not relocate pages. Regular file-backed PLE page-cache placement differs from anonymous VMA policy. Initially retain one shared mapped table; measure cold/warm placement, remote traffic, lookup ownership and common-result distribution without extra GPU weight copies. Measure automatic NUMA balancing and huge-page policies. Pin an entire evictable mapping only with demonstrated benefit and sufficient host budget.

Verify peer access and measured transfer/collective paths inside the actual container, not only on the host. Check cpuset/NUMA permissions, shared-memory/locked-memory limits and topology visibility. Investigate unexpected host staging or poor P2P, including relevant ACS/IOMMU settings, against NVIDIA's platform requirements; topology labels alone do not prove correct direct transfers. Record approved host configuration changes and requalify, rather than blindly applying system-wide tuning. GPUDirect RDMA is required for the two-server TP8×PP2 profile in Section 20; qualify it with the same native SM70 stack. It is not a prerequisite for one-server PP1. GDS remains outside the required implementation.

Sources: [Linux NUMA memory policy](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html), [PyTorch CPU/NUMA tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html), [CUDA NUMA guidance](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html#numa-best-practices), and [NCCL transport diagnostics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html).

### Model profile contract

Validate each target/speculative manifest using family-specific operators and shared lifecycle, accounting and coordinated-plan authority. Use concrete family/device allocators and ready-work execution where their representations or lifetimes differ, with one authoritative owner per claim and no duplicate legacy serving stack. Initial artifact values do not establish another artifact's geometry, numerics, tokenizer or compatibility.

| Manifest field | Required content |
|---|---|
| Engine/build identity | Actual ProviderOne repository/checkout, branch and commit, dirty-patch hash if applicable, installed host/native build hashes and the code owning import, codecs, state, scheduling, graphs and API lifecycle |
| Source identity | Target/MTP/DFlash and visual repositories, immutable revisions, files/hashes, actual encodings/quantization metadata, tokenizer/template, and matching processor/decoder identity |
| Model geometry/state | Family-specific layer/operators, attention/cache and recurrent state, dimensions, vocabulary/context, tokenization/positions and optional media/speculation; HC, QSA, GDN, experts and PLE only where actually present |
| Runtime representation | Per-tensor retained format, scale layout, sharding/replication, conversion or repacking recipe, compute and accumulation precision, activation-quantization policy, and exact allocation sizes |
| Speculative pairing | Embedded or separately supplied MTP; or draft architecture, feature taps/transforms, embedding/head ownership, tokenizer alignment, masks, position/window rules, selector/proposal probabilities, and commit/recovery convention |
| Device/build contract | Exact GPU capability/features and physical limits, host ISA/ABI, selected native target and feature suffix, toolchain/framework/library/extension identities, memory/interconnect behavior and bounded execution plans |
| Qualification identity | Source/conversion hashes, kernel/runtime build, selected target/draft KV formats, capacity ledger, and measured execution profiles for this numerical pairing |

Before device allocation, validate tensor names/shapes/transforms/operators and derive planner dimensions from the manifest. Preserve explicit authorization for executable model/tokenizer code and actual borrowed-file validation; metadata or a new family label does not grant code-execution authority. Specialize kernels without publisher/filename hardcoding. Unsupported geometry/operators require implementation before admission; the two-head placement proof and 12/13-cache formulas apply only to matching geometry.

Keep architecture, container, quantization recipe, runtime weights, compute precision and target/draft KV independent. **GGUF and safetensors are tensor containers; NVFP4 and MXFP8 are numerical/scaling formats**, not competing file containers. A safetensors checkpoint may require separate model configuration, tokenizer/processor and quantization metadata; validate the shard index and actual tensors/scales/transforms before allocation. GGUF metadata also does not implement the model’s operators. Weight quantization does not select KV. Sources: [GGUF container](https://huggingface.co/docs/hub/en/gguf), [safetensors specification](https://github.com/safetensors/safetensors/blob/main/README.md) and [NVIDIA quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html).

GGUF/safetensors support must handle their applicable single/sharded layouts and mixed per-tensor encodings through checked bounded imports. Pin exporter/recipe, block sizes, scale tensors, axes, zero points/codebooks, activation quantization and any calibration/QAT transforms. NVFP4, MXFP8, ordinary FP8 and integer/GGUF recipes require their own decoders and compute admission, not a generic low-bit reinterpretation. Add further mainstream recipes as concrete qualified profiles; neither filename parsing nor a generic loader establishes universal Hugging Face model support.

A compact source can use a native accelerated path where the selected device and libraries support it, or a separately qualified supported compute representation. V100 has no native FP8/NVFP4/MXFP8 Tensor Core execution: use supported bounded decode/conversion without assuming a full FP16 bank is affordable. Declare activation and accumulation behavior, preserve selected source semantics or explicitly qualify a changed numerical profile, prepare/reuse transformations outside serving, and charge cold/peak preparation plus steady execution. MXFP8 import/compute is follow-on work; the detailed NVFP4 choices below are not an implemented MXFP8 decoder.

Blackwell is not one interchangeable backend. NVIDIA lists GB10 at CC12.1 and RTX 5090/RTX PRO 2000 Blackwell at CC12.0, distinct from data-center CC10.x. Qualify the exact GPU/host architecture, memory/coherence and P2P paths, libraries, shapes and instruction support. In particular, the inspected Transformer Engine matrix limits its MXFP8 recipe to CC10.x and treats CC12.x NVFP4 separately; this is a library boundary, not proof that all Blackwell hardware lacks the format. Do not import data-center kernels or assume one library’s feature matrix is the ISA. Blackwell may use a separate appropriate toolkit/build from V100’s native SM70 CUDA12.9 stack; no single global CUDA restriction is required. Sources: [NVIDIA device capabilities](https://developer.nvidia.com/cuda/gpus), [Blackwell tuning](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) and [Transformer Engine support matrix](https://nvidia.github.io/TransformerEngine/support_matrix.html).

### Owned-hardware profiles and legal specialization

Inventory the actual owned machines first. The immediate V100 server, Dell Pro Max GB10 and GeForce RTX 4090 are distinct execution targets. RTX 5090/RTX PRO Blackwell and data-center Blackwell need their own concrete inventory and qualification; references or compiled code do not establish ownership or tested support. Ampere is an allowed concrete extension when needed, with no invented current performance result.

| Target | Device-specific execution direction | Preparation and qualification boundary |
|---|---|---|
| Current V100 servers, SM70/AMD64 | Native FP16 Tensor Cores, FP32 arithmetic and qualified original-format/decoded-weight kernels; retain the SC117 Q8/iso4, state and six-profile obligations | Selected CUDA12.9/owned-Torch source foundation; actual device, full-model and complete-resource gates remain required |
| Owned GB10, SM121/ARM64 | Preserve selected native low-precision, BF16/FP32, packing and measured launch/grouping paths under each exact model/recipe contract | Separate modern ARM64 foundation, coherent unified-memory accounting and actual GB10 installed/model/service evidence; no transferred V100 numerics |
| Owned RTX 4090, Ada SM89/discrete VRAM | Native FP16/BF16/TF32 and FP8 opportunities; FP4 storage uses a separately qualified supported compute representation | Native AMD64/SM89 artifacts and actual model/capacity/PCIe/service tests; Blackwell-run cross-build/oracles do not qualify RTX 4090 execution |
| Concrete Ampere extension | Distinguish SM80 A100/A30, SM86 RTX 30/A-series and SM87 Orin where requested; exploit legal FP16/BF16/TF32 and asynchronous copy, with device-specific limits | Implement and qualify all selected operators/build paths for the actual device; the current generic backend’s SM89 minimum cannot simply be lowered to obtain complete Ampere support |

Sources: [NVIDIA device capabilities](https://developer.nvidia.com/cuda/gpus), [Ampere tuning](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html), [Ada tuning](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html), and exact [PTX instruction/target rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html). Extra numerical modes are capabilities to evaluate, never automatic changes to admitted weights/activation/KV/state precision. Ampere has no native FP8/FP4 MMA; Ada has no native FP4 MMA. Legal stored formats can still use a qualified reconstruction path. Do not force SM70’s instruction/tile choices onto later hardware or use native low-precision paths on unsupported targets.

The pinned [modern build guide](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/deploy/gb10/README.md) selects CUDA13.4.3 with separate ARM64/AMD64 Torch2.16 nightlies, TorchVision and Triton3.9 inputs. These are selected foundation inputs, distinct from the source/image identities of earlier accepted services; their updated qualification remains explicit. Preserve separate immutable SM70 and modern deployment builds and architecture-specific generated locks/cache identities. Share source and semantic contracts without mixing incompatible runtime libraries or binary caches. Record host ISA/ABI, effective CPU dispatch, device limits, feature-specific compiler target, native provider bytes and actual loaded libraries. Rebuild/requalify affected artifacts after relevant changes; never mutate a running dependency prefix.

Select required features per operator at preparation. The pinned generic [`cuda/build.py`](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/src/tensorfold/cuda/build.py) defaults to SM89, and [`cuda/precision.py`](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/src/tensorfold/cuda/precision.py) currently implements native NVFP4 for major12 and FP8 from SM89. Those predicates describe existing implementations, not the complete capability of every device. An Ampere or data-center Blackwell profile needs a complete legal operator chain, including prefill/decode, expert tails, attention/state, selected draft, sampler and vision when exposed. Changing a guard, importing a module or compiling a kernel does not supply that chain.

Build native code for actual selected capabilities. Architecture-specific `a` targets have restricted compatibility; family-specific `f` targets are portable only within their documented family/toolchain rules. SM100-family `tcgen05`/TMEM, SM120/121 block-scaled MMA and measured GB10 PDL/tile choices are separate implementation opportunities. Verify the emitted code, shape/resource limits and whole-serving benefit for each selected target. Native compilation/PTX compatibility and performance qualification are separate. Sources: [NVIDIA feature-target compatibility](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/) and [Blackwell compatibility](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html).

GB10 CPU and GPU allocations compete for one physical memory pool. Account OS/services, mapped/host/device-owned weights, preparation, staging and execution by actual backing and overlap, without inventing a separate additive RAM-plus-VRAM capacity or double-charging aliases. Discrete GPUs need separate host/device ledgers, registration, DMA and permitted overlap. Preserve each existing profile’s explicit residency contract; SC117/V100 executing expert/KV offload remains excluded. See the [GB10 memory class](https://docs.nvidia.com/dgx/dgx-spark/hardware.html), [Dell device documentation](https://www.dell.com/en-us/infohub/articles/dell-pro-max-gb10-deploying-llms) and pinned [discrete-device deployment contract](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/deploy/discrete-gpu/README.md). Measure the actual Dell inventory rather than copying another chassis’s power, clocks or memory allowance.

Qualify the real driver/device/host combination. The GB10 compatibility-driver arrangement does not establish a GeForce driver upgrade path; NVIDIA limits [forward-compatibility packages](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html) to documented systems. Current CUDA native Linux ARM64/AMD64 support does not establish native Windows or WSL2 qualification. Device support and host/OS support are separate declared combinations.

### Prepared artifact and execution identity

Bind container/recipe and decoded-value identity separately from device packing and executable identity. A reusable transformed artifact includes its source hashes, actual scales/transforms, shape/orientation/padding, selected numerical behavior and layout. A compiled/algorithm/graph choice additionally binds host/GPU target and feature set, ABI, compiler/flags, linked dependencies, operator inputs/layouts, workspace and captured assumptions. Resolve these once at preparation/profile activation. Do not key support or cache validity by a broad “CUDA”/“Blackwell” label, filename, client count or product name.

Preserve the current [execution-profile manifest](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/docs/execution-profile-manifest.md) distinction between exact byte identity and substantive qualification evidence: an empty table or matching digest does not qualify execution. Preserve same-host rank-shared hashing of immutable opened artifacts rather than rereading complete weights on every rank. Its current same-host/container/inode contract does not extend unchanged across PP2 servers. Verify each independently stored node copy under its own open-file/immutability contract, agree content/profile digests globally and publish only after all required nodes acknowledge; do not require different servers to share inode or host identities. Hashing, compilation and exhaustive tactic selection stay outside normal request execution.

Keep immutable software/provider identity separate from fresh host observations. The same unchanged image can report different legal CPU dispatch or driver/device information on another machine; record those conditions and requalify affected execution instead of forcing CPU features or hashing a host-dependent diagnostic string as software identity. Preserve exact provider/header/ABI and source-owned resource checks. Stable configured power/resource controls can belong to measurement identity; instantaneous clocks, temperatures, queue timing and P-states are observations. Preserve host-only cached/advisory runtime status and reader/drain ownership so health/discovery/metrics polling does not introduce a GPU synchronization or extend a retired engine’s lifetime. Required readiness and resource checks remain authoritative.

### NVFP4 sources on V100

Import NVFP4 through an explicit SM70 representation. E2M1 values use two-level scaling, commonly an E4M3 scale per 16 values and an FP32 tensor scale. Exporter layouts/recipes vary: decode the actual scale tensors, orientations, transforms, padding, and mixed-precision exclusions. Sources: [NVIDIA NVFP4 format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) and [scaling/recipe documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html).

| Qualified runtime representation | Implementation and accounting |
|---|---|
| Retained packed NVFP4 values/scales with software tile decode | Custom SM70 kernels decode bounded weight tiles into supported FP16 operands, with sensitive scale arithmetic/accumulation as required. Charge packing, scales, tiles, conversion work and actual per-rank retention. This preserves compact storage without claiming native NVFP4 matrix throughput. |
| Converted supported packed weight format | Stream through bounded import buffers and materialize the chosen SM70 format. Repacking that preserves decoded values is distinct from requantization; requantization changes the numerical profile and needs separate quality, acceptance, memory and performance qualification. |
| Resident FP16 for selected tensors | Permit only where the full retained and peak allocation plan fits and measurement justifies it. Do not silently expand an entire target because its source encoding lacks a native instruction. |

Initially retain packed NVFP4 weights and decode bounded tiles into a qualified SM70 FP16 compute path. Select another qualified representation only when measured throughput and quality/resource constraints justify it, using the same engine's loader/packing and kernel choices. Source file size is not a VRAM budget; native-Blackwell benchmarks do not predict V100 performance.

Inspect activation quantization, calibration transforms, and QAT computation. Weight decoding alone does not reproduce native W4A4. Declare activation/accumulation policy and separately qualify departures from the source recipe. Conversion cannot restore lost quantization information; validate finite range and reconstruction error before serving.

Prepare/hash converted or repacked artifacts once per source/recipe with bounded host/device memory; reuse them, never convert on admission. Budget concurrent preparation. Record source/derived identities and reject unsupported/infeasible choices without substituting model, precision or offload policy.

### Initial target artifact

Use non-Swift `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF` revision `90a3350cd800ec1fe0d1660a38414ad9e985782f`, directory `IQ3_S`. Load both files prefixed `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-`. Sources: [pinned directory](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/tree/90a3350cd800ec1fe0d1660a38414ad9e985782f/IQ3_S) and immutable [first](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/raw/90a3350cd800ec1fe0d1660a38414ad9e985782f/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf)/[second](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/raw/90a3350cd800ec1fe0d1660a38414ad9e985782f/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf) file pointers.

| Shard suffix | Exact file bytes | SHA256 |
|---|---:|---|
| `00001-of-00002.gguf` | 55,063,448,064 | `40f369eca301ec6ea318dbb671b3132c26a2f0b89aa5458a7051b33215ee296c` |
| `00002-of-00002.gguf` | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| Total | 83,863,586,496 | 78.104051 GiB of files |

The publisher identifies the first shard as transformer weights and the second as the IQ4_NL PLE/ngram table. Their file sizes are 51.281832 and 26.822219 GiB, respectively; separate vision and speculative artifacts are excluded. File size is not GPU residency. Identify PLE by tensor name/geometry, never a general shard-number rule.

Both selected binary tensor directories have now been inspected through bounded range reads: 1,223 transformer tensors and one host PLE tensor. Their declared payload sizes reconcile with consecutive offsets and file lengths. Transformer payload is 55,052,423,680 bytes: routed experts 50,292,326,400 and other tensors 4,760,097,280. PLE payload is 28,800,138,240 bytes. See the [SC117 numerical companion](https://chatgpt.com/library?file_id=libfile_024132db763081918939c76377983933&account_id=personal) for the census and derived copy prices. This is source-header evidence, not full-file hash/value, tokenizer, import or runtime qualification. Before allocation, independently validate complete coverage, shapes, block sizes, split metadata, offsets, alignment, scales/codebooks and full-file hashes, and generate the per-tensor source-to-runtime inventory.

### Required target geometry

The initial implementation targets the following base-family geometry from `Qwen/Qwen3.8-Flash-Next` revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Assert it against the selected SC117 binary metadata during import; publisher lineage alone cannot validate every field. Source: [pinned base configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json).

| Property | Required value or interpretation |
|---|---|
| Architecture | GGUF `qwen4exp`; base configuration `qwen4_exp` |
| Target depth | 48 blocks: 36 Gated DeltaNet/recurrent blocks and 12 QSA blocks, every fourth block |
| Hidden and residual geometry | Hidden width 2,560; four hyper-connection streams; low-rank width 320 |
| Target attention | 24 query heads; two global KV heads; head dimension 256 |
| QSA indexer | Four query heads; one history head of dimension 128; compression ratio 4; 512 selected complete four-token groups plus up to three visible tail tokens |
| Recurrent geometry | 16 key heads and 48 value heads, each dimension 128; convolution width 4; FP32 state |
| MoE | 512 routed experts, ten selected per token; routed and shared intermediate width 640 |
| Context | Native configured allowance 262,144; extended allowances use Section 6's qualified RoPE/YaRN profiles. Prompt and generated output share the selected allowance. |
| Positional encoding | Target theta 10,000,000; 64 rotary dimensions out of each 256-dimensional head; preserve the supplied mRoPE conventions |
| Vocabulary | 248,320 entries; target embedding and output head are distinct tensors |
| PLE | N-gram size 3; depthwise causal convolution width 4, dilation 3, nine prior activation rows; base config layer identifier 2 is one-based and corresponds to GGUF `blk.1` |

Preserve all four residual streams within target computation; do not average them or treat them as independent requests. DFlash's defined reduction applies separately at specified completed blocks.

### Mixed weight import

`IQ3_S` is a learned mixed allocation. Initial decoders must support the publisher's BF16, F32, F16, Q2_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS list. Reconcile the inspected two-shard census with the importer’s exact counts/payloads; the card histogram is not a complete census. Sources: [pinned SC117 inventory and method](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/blob/90a3350cd800ec1fe0d1660a38414ad9e985782f/README.md).

Preserve the selected learned values, scales and codebooks through layout repacking. Use artifact-specific packed offsets and hashes. An unsupported format must not trigger uniform requantization, a different publisher's pack, or whole-model FP16 expansion. Any changed numerical representation is a separately selected and qualified profile.

Inventory every retained/decoded/replicated/sharded/aliased tensor per rank, including speculation and vision. Bind Section 15's decay/head-order transforms; identify the host-mapped PLE tensor explicitly. Qualify BF16→FP16 range, sensitive accumulation and indexer selection stability. The historical 107.371 GiB receipt is not this runtime plan.

### Tokenization and positions

Extract and hash the selected GGUF tokenizer/chat template, including reasoning and tool-call formatting; do not substitute a generic or another publisher's template. Count the fully rendered expanded input, retained conversation, special tokens, and generated reasoning/output against context/reply limits. Drafts use target token IDs.

Preserve distinct generation-stop, BOS/padding, and PLE-reset roles. The base reference uses ordinary `im_end` 248046 and end-of-text 248044, with both listed as generation terminators; verify the selected metadata/template before binding these roles. QSA's 2,048-token complete-group budget can require 2,051 positions with the incomplete visible tail. Sources: [base generation configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/generation_config.json) and the pinned base configuration.

### Multimodal profile

Image and video understanding with text output is required. At the same pinned SC117 revision, use `mmproj-Qwen3.8-Flash-Next-BF16.gguf`: **907,542,944 bytes**, SHA256 `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355`. It contains the encoder/projector; its 0.845215 GiB file size is not a per-GPU allocation. Source: [immutable visual file pointer](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/raw/90a3350cd800ec1fe0d1660a38414ad9e985782f/mmproj-Qwen3.8-Flash-Next-BF16.gguf).

Pin its tensor inventory, matching official processor configuration/code, selected template, and FP16 SM70 conversion recipe. Validate base geometry: 27 vision layers, width 1,152, 16 heads, projected width 2,560, spatial patch 16, merge 2, temporal patch 2; image/video IDs 248056/248057 and vision delimiters 248053/248054. Preserve interleaved M-RoPE sections `[11,11,10]` and continuation deltas. The pinned [base configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) and [target/vision implementation](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py) define reference semantics; Section 16 specifies bounded execution and scheduling. Exact SC117 multimodal quality and V100 performance require qualification.

#### Pinned processor contract

Use the official base processor files at `de4b8e4d43b917e7706784d8bb445c9af86a3540` with Transformers reference `50e35cc0ca18ec19b80653690de553e6decf74f5`:

| File | Bytes | SHA256 |
|---|---:|---|
| [preprocessor_config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/preprocessor_config.json) | 390 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| [video_preprocessor_config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/video_preprocessor_config.json) | 385 | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |

Both select `Qwen3VLProcessor`. Pin the torchvision `Qwen2VLImageProcessor` resolved from the legacy `Fast` name and `Qwen3VLVideoProcessor`; no silent preprocessing-backend substitution.

| Contract | Image | Video |
|---|---|---|
| Pixel processing | RGB, channel-first; resize/rescale/normalize enabled; bicubic with antialias; rescale by 1/255, then mean/std 0.5 per channel with reference FP32 normalization | Same numerical settings |
| Resize bounds | 65,536–16,777,216 pixel-area bounds per image; preserve aspect ratio and snap spatial sizes to multiples of 32 | 4,096–25,165,824 temporal-spatial pixel budget across sampled frames; not a per-frame budget |
| Temporal grouping | Repeat still-image content across temporal patch size 2; grid_t=1 | Pair frames; repeat the final frame for odd counts |
| Reference sampling | Not applicable | 2 fps; automatic sampling clamps to 4–768 frames and source length, with rounded uniform indices |

Pin `cap_pixels_per_frame=false`: the inspected revision resolves its `None` default to false. Its optional `max_video_tokens=768` is not a universal context limit. Explicit `num_frames` bypasses the automatic frame-range clamp, so enforce administrator bounds on every entry path. Require reliable FPS/timestamps or an explicitly selected assumed-FPS policy; do not silently apply upstream's 24-fps fallback. A one-frame video requires explicit still-image conversion or rejection.

Define decoded pixel-domain, orientation and channel semantics before processing: arrays/tensors must be RGB rather than BGR and must not be rescaled twice. Match reference image and video alpha/grayscale handling separately. Pin decoder behavior and retain fixtures for backend differences. Service media limits remain separate from upstream defaults; Section 16 admits the actual resized/padded grids, markers and timestamps.

Sources: pinned [image processor](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py), [video processor](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py), [image backend](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/image_processing_backends.py), and [processor expansion](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen3_vl/processing_qwen3_vl.py).

### MTP provenance boundary

Use the separately pinned shared-Q8 MTP companion in Section 12; target shard metadata declaring an MTP layer does not supply that companion. Confirm tensor coverage and reject duplicate/ambiguous MTP imports. Section 14 pins IncoAI DFlash2. Both bind the actual SC117 embedding/output tensors without replacing the target trunk or vocabulary weights.

## 3 Historical deployment and accounting evidence

This section preserves the supplied earlier-deployment receipt for accounting and geometry checks. It is not a weight, vision, capacity, or performance receipt for the selected SC117 profile. Its local JSONs (Section 25) have not been independently inspected in this workspace.

### Historical reported configuration

- Eight GPUs at 32 GiB each: 256 GiB nominal total VRAM.
- INT8 K/V, GPU-resident raw and pooled indexer history.
- Optional raw-history RAM archive disabled.
- Three sessions, each with 262,144 logical context tokens.
- 262,148 physical rows per session in this verification geometry.
- Twelve target attention caches and one stock-MTP cache.
- Thirty-six recurrent layers using fixed per-session state instead of complete KV histories.
- Two global KV heads: ranks 0–3 retain head 0 and ranks 4–7 retain head 1 in the reported layout. Each head therefore has four physical copies.
- Both indexer histories are replicated across all eight ranks.

These are reported logical ranks, not a qualified physical NVLink grouping.

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

Torch allocated memory is included in reserved memory; never add them. Requests and graph capture alter the snapshot, which does not establish serving peaks or later graph occupancy.

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

The eight rank plans total approximately 107.371 GiB for one sharded model. The illustrative 14,411,046,312-byte weight size is approximately 13.421333 GiB; admission uses actual per-rank payloads and allocations. KV quantization does not change weight encoding.

This loader receipt is not the selected SC117/MTP/vision inventory. Reconcile its own historical categories separately; build the selected profile's receipt from actual imports and allocations.

The reported recurrent/tail/replay subset is 27.798 MiB/session/GPU; the explicit workspace plan is about 0.640 GiB/GPU, mostly expert-prefill tiles. Reconcile their existing categories and lifetimes before combining them; they are not additional totals.

### Historical history footprint

| Three sessions at 262,148 physical rows | GiB per GPU |
|---|---:|
| INT8 K/V payload | 4.875 |
| K/V scales | 0.305 |
| Raw FP16 indexer history | 2.438 |
| Pooled FP16 indexer history | 0.609 |
| Total | 8.227 |

Reported logical-token rates, with pooled indexer cost amortized over four-token groups:

| Storage per token on each reported rank | INT8 | Existing Signed128 Q4 |
|---|---:|---:|
| K+V payload and metadata per attention layer | 544 B | 272 B |
| Raw FP16 indexer per layer | 256 B | 256 B |
| Pooled FP16 indexer per layer, amortized | 64 B | 64 B |
| Thirteen caches with GPU history | 11,232 B | 7,696 B |
| Thirteen caches with optional raw-history RAM archive | 7,904 B | 4,368 B |

INT8 uses FP16 scales per 32 values; existing Signed128 Q4 uses FP32 scales per 128 values. Equal byte counts do not make either interchangeable with RotorQuant (Section 13).

In this historical layout, one logical token costs 89,856 physical bytes across the box in INT8/GPU-history mode. Three sessions means three total, not three per GPU.

### Reported qualification history

| Reported configuration | Sessions | Context per session | Status in supplied receipt |
|---|---:|---:|---|
| INT8 and GPU history | 3 | 262,144 | Reported live-qualified startup/request path |
| INT8 and raw-history RAM archive | 4 | 262,144 | Previously live-qualified |
| Existing Signed128 Q4 and GPU history | 4 | 262,144 | Previously live-qualified |
| Existing Signed128 Q4 and raw-history RAM archive | 8 | 260,060 | Previously live-qualified |

These Q4 results do not qualify RotorQuant. INT8/GPU-history header estimates of 4×211,270, 5×168,236, and 6×139,312 were not live-qualified and included no extra fixed graph reservation.

Six-million-token arithmetic for the reported layouts:

| Six million total logical tokens | Cache GiB per GPU | Cache plus weights GiB per GPU |
|---|---:|---:|
| INT8 and GPU history | 62.764 | 76.185 |
| Existing Q4 and GPU history | 43.005 | 56.426 |
| INT8 and raw-history RAM archive | 44.167 | 57.588 |
| Existing Q4 and raw-history RAM archive | 24.408 | 37.829 |

All exceed 32 GiB before other obligations; they do not bound a new unique-history architecture. Six million near-native tokens require at least 23 sessions under the 262,144-token per-sequence context limit.

## 4 Accounting rules and complete resource ledger

User authority updated on 2026-11-13: retain at least24 Q8 or32 Q4 sessions at the model's advertised262,144-token logical context, including the actual speculative physical tail. Once those complete resident GPU windows are materialized, otherwise unused GPU memory may fund graphs and other execution. An arbitrary unused4 GiB floor is not a separate permanent obligation in that region. Required execution workspace (at least2 GiB), snapshots, replacement/copy overlap, future growth and all actual library/driver/private-pool residency remain funded once, on the worst rank. Do not trade away context, required session capacity, recovery or live workspaces. This scoped authority supersedes other blanket device-floor wording in this design for that complete resident region; other profiles retain their existing protections.

The cold loader may retain its existing conservative4 GiB admission baseline only when it does not rule out the required feasible full-context capacity. Reconcile measured post-materialization physical headroom and any absolute project grant against that original baseline; preserve both receipts, and do not add materialized owners or allocator reservations twice. The historical planner observed a27.086 GiB/GPU serving budget after initial runtime allocations; that is not universal. Its old three-slot0.273 GiB snapshot and0.211 GiB replacement promises remain historical and must be recomputed for actual state and lifetimes. A smaller checkpoint alone does not waive required service obligations.

Reconcile each allocation and resource promise once. Promises need not be allocated slabs; do not add them to the startup snapshot or subtract an included floor twice.

Publish the derivation of Section7's `usable_serving_budget` from measured physical capacity, external/runtime occupancy and required outstanding promises, identifying what is already charged on either side of the inequality. Bind the applicable policy: measured complete-window headroom in the user-authorized region above, otherwise the existing device floor, while preserving required workspace in both. Record which arenas satisfy each promise and what unallocated memory must remain. Do not subtract a promise again after charging its arena, or treat unrelated storage as satisfying it. Historical free/reserved readings do not define current liability.

Use checked 64-bit arithmetic for source-file offsets, allocation sizes, byte strides and page/arena address products; widen operands before multiplication or addition. Narrow local indices remain appropriate where bounds are proven for the effective profile. Reject overflow or out-of-range descriptors before allocation or launch. The selected shard exceeds 4 GiB; full-context arena addressing also requires checked wide products.

Every retained or peak allocation must have:

- An owner rank and allocation size including real alignment and padding.
- A persistent, session, batch, graph, replacement, or migration lifetime.
- A reason it must overlap another allocation, or a proven reason it can reuse an arena.
- A maximum derived from the effective profile and implementation limits.
- A measurement or construction proof for allocations made by libraries outside the explicit allocator.

Preserve the current branch’s process-lived host/native resource reservations and highwaters across profile replacement: Store/control maps and waits, descriptors/FD metadata, RPC payloads and queued input/output claims. Source/ABI payload bounds, allocator/RSS/kernel buffers and measured physical/cgroup availability are different evidence. Refine a conservative bound only with an equally complete justified proof; lowering current load, hiding a control or retiring one profile does not release a surviving process-lived resource. Runtime/ABI-specific equations stay scoped to that implementation and must be replaced or revalidated for a changed backend. Use the existing ledger rather than adding a second certifier.

Per-rank budgets cover target/method/vision weights and histories; recurrent/convolution/raw-pending/pooling/draft/sampling/constraint/output/media-position state; bounded language/visual execution/features; verification/recovery; graphs/communication; supported replacement overlap and any adopted migration overlap; metadata/padding. Host media/replay/pinned staging and Section 17's queued inputs/retained checkpoints are separate bounded resources. Tiered retention increases logical session capacity, not GPU-resident KV capacity; charge transfer metadata/staging and permitted overlap in the GPU ledger.

Take the maximum for proven mutually exclusive workspaces and sum overlapping occupancy. Use one bounded execution arena where it can replace per-slot maximum slabs; reuse only after all asynchronous readers retire.

Use fixed-size typed history-page arenas and bounded execution/graph arenas. Track free classes/counts per rank, not aggregate bytes; reserve matching layer bundles before execution. Full admitted windows must remain allocatable despite fragmentation and required state/restore overlap, without fresh general-purpose device allocation per page at maximum residency. Charge separately any supported graph replacement or optional migration; ordinary activity neither rebuilds the graph set nor resizes these arenas.

Before publishing capacity, initialize or conservatively budget every eligible kernel/communicator/library family's lazy residency. Every selected plan must fit admitted commitments and the overlap envelope. Deleting a graph does not prove its pool was released. Enforce these invariants in the existing allocator/planner.

Reconcile the workspace protection with actual language/visual arena lifetimes. The historical 0.640 GiB workspace plan and 27.798 MiB/session subset cannot be added wholesale to a newly itemized ledger. Recalculate snapshots/replacement for the selected profile without weakening recovery.

## 5 Performance objectives and workload definitions

### Separate quantities

| Quantity | Definition |
|---|---|
| Configured context allowance | Maximum prompt-plus-reply length promised to each session under the effective profile |
| Resident capacity | Maximum simultaneously GPU-resident logical sequences with complete context/state commitments; distinct from total retained sessions and waiting requests |
| Resident sequences | Request-scoped or retained sequences currently holding GPU history/state and GPU commitments |
| Retained sessions | Logical conversations preserved across requests, on GPU or in a qualified RAM/storage checkpoint |
| Waiting requests | Accepted fresh or resume work awaiting GPU admission/restoration under Section 17's bounded FIFO policy |
| Active requests | GPU-admitted requests with pending or in-flight inference work; waiting/restoring requests are reported separately |
| Active sequences | GPU-admitted independently advancing sequences with pending or in-flight inference work; exclude suspended/restoring sequences from activity buckets |
| Scheduled decode sequences | Ready sequences included in the next execution step |
| Target query or verification rows | Exact target rows executed, including speculative candidates according to the method convention |
| Prefill rows | Expanded prompt rows scheduled in the current step; distinguish text tokens, visual rows, and required special/timestamp tokens |
| Tentative cache rows | Uncommitted target/draft storage needed by the selected speculative procedure |
| Committed output length | Tokens accepted into the server's canonical transcript and eligible for publication |
| Delivered output | The portion actually sent to the client; network delivery is separate from model-state commit |
| Target-cached length | Positions already consumed into the target's cache and state |
| Draft-valid length | Target context through which the drafter has the required valid conditioning/state |

Committed, target-cached, and draft-valid lengths can differ: an accepted anchor/bonus may await target consumption. Each method defines positions and commit; asynchronous delivery does not change them.

Capacity counts independent logical sequences. Session/threshold/row examples assume one sequence per HTTP request. Batched prompts, multiple completions, or branches require separate history, recurrent/draft/RNG state, recovery, and full-context reservations in admission and scheduling. Reject unsupported multiplicity. Prefix sharing cannot reduce the guarantee without a complete branching/copy-on-write budget.

Test one active request with all permitted GPU histories resident. Idle GPU-only retained history remains committed memory; tiered state releases its GPU claim only through Section 17's checkpoint protocol. Backlogged FIFO and tiered-retention workloads require separate end-to-end receipts.

### Optimization objective

Hold context/position, quantizations, method and media bounds fixed. Establish the largest qualified resident count with useful execution resources; measure sustained committed decode with all sessions active and near-full histories. Tune every active count through capacity, including full idle histories elsewhere. Report per-session rate/latency; empty-context batch rates do not establish full-load performance.

Administrators choose context using the measured context/capacity/throughput frontier. Load adaptation must not resize windows, downgrade Q8/Q4 or disable the selected method. It selects safe batches, proposals, prefill/media quotas and prepared kernel/graph plans within the selected profile. History ownership remains fixed initially; Section 11 gates any future live redistribution.

Measure text prompt tokens/s, expanded visual rows/s, and committed output tokens/s separately, including per-request output rate, TTFT, TPOT percentiles, queue delay and fairness. Report media pixels/frames and preprocessing/encoding time; visual rows are not ordinary text-token throughput. Rejected candidates are not delivered output.

Count each target prompt row once; report prefix/media hits, replay and MTP/other conditioning separately. Include media preparation/encoding, transfers and maintenance in complete-operation timing. Track successful-task latency and reasoning/output counts so shorter answers are not mistaken for higher token/s.

Avoid batch-filling delay at low activity; maximize aggregate committed throughput within administrator latency limits at high activity. Mixed load must advance both prefill and decode. Optimize whole iterations, including routing, features, draft maintenance/drafting, verification, rejection/replay, communication, sampling, and commit.

Establish this fork's measured performance frontier against the user-verified hardware targets below. Select plans from measured results; neither modern-GPU benchmarks nor peak arithmetic establishes a universally winning plan.

### Verified hardware targets and milestone policy

The user has verified the following performance on this exact eight-V100-32-GiB hardware. Adopt these as concrete implementation performance objectives alongside the capacity, quality and lifecycle contract.

| Benchmark | Required performance objective | Context and KV |
|---|---|---|
| Single active request decode | Greater than 250 committed output tokens/s | 262K context, Q8 KV |
| 24 simultaneously active requests decode | Greater than 600 aggregate committed output tokens/s | 262K context per request, Q8 KV |
| Prefill | Greater than 3,500 processed prompt tokens/s | Bind the reference workload/concurrency and cache conditions in its receipt |

Use the native 262,144-row profile for the design's 262K qualification. Qualify the 24-active Q8 operating point with full-window reservations and useful execution resources; it is a specific implementation objective, not a new hard ceiling. A capacity/resource shortfall is an explicit gap to resolve, never permission to shorten context, change quantization or count queued requests as active. Continue computing the maximum under Section 7.

The reference measurements are user-verified hardware evidence; independently reproducing them in this implementation remains required. Bind target/runtime weight and speculative-method identities, configured allowance and actual cached lengths/positions, resident versus active counts, prompt/output lengths, sampling, media, cache conditions, clocks and measurement boundaries in the comparison receipts. The supplied figures do not identify every reference setting; do not invent those bindings or require the prefill threshold at an unspecified concurrency. Retain near-full-history/full-active qualification and the one-active-with-idle-residents case separately. When supplying these targets, the user reported implementation progress but decode/prefill still far below the objectives; no current implementation rates were supplied.

**Milestone A/B contract.** Test every optimization and development milestone at **1, 2, 4, 8, 16 and 24 active concurrent requests**, recording both prefill and aggregate committed decode tokens/s at each point. Require a repeatable decode and/or prefill improvement, with no decode regression at any tested count beyond a predeclared measurement-noise tolerance. Higher-concurrency gains cannot conceal a lower-concurrency decode regression. Apply priority **1 < 2 < 4 < 8 < 16 < 24** when prioritizing eligible results and implementation work. This ordering supplies no numerical weights; any weighted score must declare weights consistent with it and retain the per-count results and hard gates. The milestone matrix supplements Section 17's release sweep of every supported integer count.

Select qualified kernel/graph/batch/proposal plans by load within the unchanged effective numerical profile. Keep the fastest eligible plan at each regime; record unachieved targets and measured gaps explicitly. Performance completion requires matched receipts exceeding each bound target; completing implementation milestones alone does not satisfy it. Section 21 defines reproducible receipts and acceptance checks.

## 6 Administrator configuration and effective profiles

### Inputs

Expose only implemented, qualified choices:

- A qualified Qwen3.8-Flash-Next target manifest and runtime weight representation, initially non-Swift SC117 IQ3_S.
- Equal context allowance and a qualified native or extended RoPE/YaRN profile, with explicit target/MTP/draft coordinate bounds; 262,144 is the native baseline, not a permanent engine ceiling.
- Target KV format and its pinned implementation.
- Speculative method: none, stock MTP, or DFlash.
- Compatible MTP/DFlash artifact and supported method-weight/KV choices; initial MTP KV matches target KV. Initial DFlash FP16 KV is a candidate subject to Section 14's range gate, not an already-qualified storage guarantee.
- Explicit history-residency policy and implemented media limits/processing policy; the effective profile includes fixed vision placement and bounded visual execution.
- Structural target profile: fixed `tensor_parallel_size=8`, `pipeline_parallel_size` in `{1,2}` and `expert_tensor_parallel_size` in `{8,4,2}`. PP equals participating server count; each server supplies one eight-GPU stage. EP is derived as `8/ETP` within each stage; no DPA, full-model replica or independent method-degree controls. Validate Section 20’s exact hardware/operator/state plan before allocation.
- Optional session ceiling.
- Performance objective and any requested latency or waiting-time limits.
- Explicit retained-session lifetime, bounded FIFO admission/output queues and per-principal limits; qualified GPU-only or RAM/optional-storage retention with separate session/input/backing budgets (Section 17).

A draft sliding-window override is exposed only if the selected checkpoint and storage implementation support it. Target and draft weight formats, target KV, and draft KV are independent dimensions. Validate each requested sampling/constraint policy against the effective verifier's capabilities before admission or session mutation; reject unsupported combinations with a specific reason.

### Outputs

At preparation/profile activation the engine resolves full-context resident capacity, admission and media-resource limits, a maximum language/visual execution envelope, row/patch bounds, graph/kernel families and fixed vision/history placement. Per-batch scheduling chooses used work and ready plans inside those bounds; an active-count change does not recalculate Nmax or enlarge the maximum envelope. It reports the binding rank/resource and qualification status, resolved server/GPU counts, TP8/PP/ETP and stage-local EP, stage/rank maps, companion/endpoint placement and any copy budget.

Illustrative schema, not existing ProviderOne flags:

```yaml
target:
  repository: SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF
  revision: 90a3350cd800ec1fe0d1660a38414ad9e985782f
  weight_directory: IQ3_S
context_tokens: 262144
position_profile: native  # immutable resolved RoPE/MTP/draft manifest; qualified YaRN profiles may extend this
kv_format: q8
parallelism:
  tensor_parallel_size: 8
  expert_tensor_parallel_size: 8
  pipeline_parallel_size: 1  # 2 requires two physical servers, eight GPUs per stage
speculation:
  method: mtp
  artifact: shared_q8_mtp  # pinned manifest in Section 12
  kv_format: target
  proposal_length: auto
multimodal:
  enabled: true
  artifact: sc117_vision  # pinned manifest in Section 2
  limits: qualified_media_envelope  # concrete bounds required before admission
max_sessions: auto
history_residency: gpu  # or a separately qualified tiered retained-session policy
admission:
  policy: fifo
  queue_limits: qualified_request_and_input_budget  # concrete count/byte/timeout bounds
performance:
  objective: max_sessions_then_full_load_decode_throughput
```

An integer session setting caps physically supported capacity; it cannot force unsafe allocation. Any future minimum-capacity setting must fail if infeasible, without silently shortening context or changing quality.

Keep configured, pending, and effective profiles distinct. Key each by target/method/vision hashes and revisions, conversion/runtime representations, tokenizer/template/processor, media policy, feature/position contracts, codecs, arithmetic, topology, kernel build, dependencies, and execution geometry. Prior geometry may seed tuning; release a new pairing only after its affected acceptance, quality, allocation, and performance gates pass.

### RoPE/YaRN context profiles

Implement context extension in the same engine. The pinned official model card recommends RoPE scaling/YaRN beyond 262,144 tokens; it establishes no reliable maximum for SC117 IQ3_S, either KV codec or speculative companion. Native/extended profiles require distinct qualification identities. Sources: [official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/README.md).

**Configuration.** Resolve native/default or YaRN into an immutable manifest: algorithm/revision, original reference length, requested maximum rows, coordinate bounds, theta, rotary width/layout, factor, ramp, amplitude/attention scaling and rounding/clamping. Record numerical defaults separately for target attention/indexer, MTP attention/indexer and DFlash. Before allocation, reject nonfinite parameters, nonpositive reference lengths/amplitudes, invalid theta/factors/ramp bounds or inconsistent geometry; warnings are insufficient. Resolve GGUF/external conflicts explicitly or reject; never scale twice. Automatic factor selection may depend on the configured maximum, never live load or prompt length.

First YaRN candidates use the pinned native reference length 262,144, theta 10,000,000 and 64 rotary dimensions, verified against import/reference code. Windows 524,288/1,048,576 imply factors two/four; these are qualification targets, not released limits or quality guarantees. Keep reference length separate from runtime maximum. Resolve beta/ramp/amplitude defaults from the selected implementation and qualify borrowed tuning. Other qualified, including non-power-of-two, windows remain eligible.

**Numerical execution.** Match YaRN interpolation/amplitude, partial rotary dimensions and interleaved M-RoPE `[11,11,10]`; preserve nonrotary channels, THW assignment, timestamps and continuation delta. Target attention/indexer use the declared compatible transform; indexer rotation remains 64 of 128 dimensions with FP32 pooling at group-first coordinates. Define amplitude/score scaling once per operator: cos/sin amplitude s multiplies only the rotary Q·K contribution by s², leaving the nonrotary contribution unchanged. Uniform full-score scaling is not equivalent; test both contributions. Library support alone does not qualify this model/indexer/M-RoPE combination. Longer text context changes neither vision positions nor QSA grouping/selection budgets.

Preserve Section 13's RoPE/codec order. Compare Q8 and iso4 with immediate quantization under identical positions; YaRN neither compresses history nor restores quantization loss. Distinguish row counts, integer absolute/M-RoPE positions and physical ring slots. Build phases/range reduction in qualified FP32-or-higher arithmetic, never FP16/BF16 absolute positions; check large-coordinate precision, launch/index limits and byte addressing. Budget bounded shared/tiled/on-demand phase generation, never uncharged per-session full-window replicas; include bounds/epochs in graph validity.

**Speculation.** Every released extended point requires MTP attention/indexer positions, feature alignment, full-prefix initialization/refresh, sampling and recovery qualification with each selected KV codec; declare MTP settings explicitly rather than assuming target overrides propagate. DFlash2 keeps separate theta, absolute scalar rows and its bounded 2,048-row window. The window does not qualify positions beyond the configured 262,144 maximum. Validate an explicit draft extension policy and its target-feature effects, acceptance and whole-round speed; original frequencies may remain only if qualified at extended coordinates. Copying target YaRN or wrapping positions modulo the ring is insufficient. Verification preserves the selected extended target distribution, not its task quality. Reject unsupported target/method/position combinations before admission; never substitute methods.

**Admission/capacity.** Bound extended context by positional quality, implementation limits and the complete worst-rank proof, not scaling factor alone. Expanded input, retained conversation and reasoning/reply share one window. Validate target/MTP axes and draft absolute positions, including lookahead, separately from rows; guard rows extend no position limit. Charge positional buffers, pooled scan/top-k, descriptors/rounding, recurrent/PLE recovery and media/replay costs. Section 8 rates apply only to unchanged geometry/codecs. Bounded selected attention does not bound pooled-indexer scan work.

**Transitions/identity.** Fix frequencies, amplitude and algorithm for the lifetime of retained state, including suspended checkpoints; active-count adaptation never retunes them. An allowance increase with unchanged transform needs Section 19's GPU/backing reservation and compatibility proof for every affected retained session. A transform/reference-policy change initially drains affected sequences and retires state/graphs before in-process activation. Updating constants or re-rotating keys alone cannot convert quantized KV/pooled keys, recurrence, features and draft conditioning. Future replay conversion needs a bounded full-state proof. Include the complete positional manifest/epoch in session/prefix/checkpoint/draft/graph/tuning identities; forbid incompatible reuse. Expose configured/pending/effective state without container/worker/CUDA-context restart.

**Release evidence.** Test native/extended windows at native/new-limit boundaries, arbitrary chunk/group/ring splits, long generation, rejection/rollback and media continuation. Compare reference/one-shot/chunked logits and state under identical settings. Predeclare quality thresholds for short-context regression, retrieval across depths, multi-needle and order-sensitive reasoning plus representative user tasks; one needle test is insufficient. Cover Q8/Q4, required MTP, separate DFlash and image/video cases. For every released profile, run Section 21's integer activity sweep and near-full-history/full-concurrency tests; report acceptance, prefill/committed decode, TTFT/TPOT, fairness and resources. Keep unsupported candidates visibly unqualified; never silently clamp.

Reference algorithm background: [YaRN paper](https://arxiv.org/abs/2309.00071) and [Transformers RoPE parameters](https://huggingface.co/docs/transformers/main/en/internal/rope_utils). The initial frequency/amplitude construction reference is [`_compute_yarn_parameters` at the pinned Transformers revision](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/modeling_rope_utils.py); resolve and record its factor, original length, beta values, truncation and mscale/attention-factor conventions explicitly. This routine is independent of current sequence length; generic dynamic-NTK behavior is not a substitute. Pin the exact algorithm implementation/hash used for release; these references do not establish V100 performance or SC117 extended-context quality.

### Retention and admission

Preserve existing lifetimes. Request-scoped slots release state after completion/cancellation and retirement of readers, transfers and callbacks. Only intentionally retained sessions survive requests under close/expiry rules; missing expiry does not make stateless requests persistent. Inactivity cannot delete retained state; Section 17 permits qualified idle-state suspension while preserving its lifetime/context. Completion, cancellation, disconnection, GPU demotion and retained-session destruction are distinct.

Use Section 17's bounded FIFO admission queue, explicit queue-full response and eligibility/aging rules. Bound media preparation before slot assignment, then validate exact expanded prompt/positions, output budget, media limits, and sequence count. Pending media work must retain its declared resource claim. Resumption preserves the retained logical/backing claim and acquires a GPU claim if suspended. Serialize or reject concurrent state mutations per sequence under the API contract. Each GPU-admitted request-scoped sequence retains its complete-window reservation until slot release; waiting input alone holds no GPU claim.

## 7 Automatic capacity planner

At preparation/profile activation, for the selected context and positional/numerical/method/media profile, the planner calculates the largest supported resident count for a useful qualified maximum execution envelope. At that count, select the measured envelope with the best aggregate committed decode throughput under service limits; retain useful prefill/media progress and fairness. Report the measured capacity/throughput tradeoff when a smaller count is faster, without silently lowering the requested capacity objective:

```text
Nmax = max N such that, for every rank r:
    target_weights[r] + method_weights[r] + vision_weights[r]
  + target_histories[r, N, context, codec, ownership]
  + method_histories[r, N, method_policy]
  + retained_state_and_media_metadata[r, N]
  + retained_recovery_checkpoints[r, N, method]
  + peak_language_visual_execution_and_recovery[r, envelope]
  + graph_and_communication_residency[r]
  + positional_tables_and_scratch[r, position_profile, envelope]
  + replacement_obligations[r, N]
  + migration_overlap_limit[r]
  + metadata_and_padding[r]
  <= usable_serving_budget[r]
```

Execution envelopes bound target/expanded-prompt rows, proposals, the largest media unit, visual microbatches and overlap. Enforce separate host/pinned/replay bounds. Full residency must support useful selected speculation and promised media work; visual encoding may queue instead of multiplying its workspace by all sessions.

Ledger terms are disjoint allocation/lifetime categories. The initial fixed-placement profile has no dedicated live-ownership migration liability: its migration-overlap term is zero. Required suspension/restore, COW, recovery and supported replacement overlap remain charged in their own categories; a shared arena cannot be credited twice. An optional migration implementation must add its actual full intermediate liability before activation. Charge exclusive positional tables/scratch there; reference shared language/graph arenas without double-counting. Target weights exclude separately charged speculation/vision; actual embedding/head aliases add no payload. Use fresh SC117 allocations, not the historical 13.421 GiB/GPU aggregate with another MTP plan.

For the compact indexer, growing histories contain attention KV and completed pooled keys. Raw pending rings and saved tail versions belong to retained state/recovery; raw prefill/verification spans belong to bounded execution.

Retained recovery includes mandatory per-session checkpoint/snapshot versions and future occupancy; peak recovery covers additional execution work. Reconcile shared storage once without omitting retained snapshots or charging a checkpoint twice. Section 9/17's prefix and queued-state catalog accounts shared physical objects once while preserving independent future-window/branch claims; charge catalog/lease/dirty metadata, bounded reusable cache and snapshot/COW overlap.

Implementation sequence:

1. Resolve target/method/vision/processor manifests, actual imported allocations, codecs, media bounds and speculative compatibility; include conversion, activation policy and media replay/staging costs.
2. Measure runtime baseline and per-rank usable budget while preserving device-floor, workspace and recovery protections; establish host/pinned/media-store bounds.
3. Evaluate a small set of prepared execution plans and fixed placements for the selected objective; retain one qualified maximum resource envelope and only useful plan breakpoints.
4. Compute exact per-rank persistent and peak costs for candidate resident counts.
5. Apply implementation limits and an optional administrator ceiling.
6. Initialize the chosen arenas and bounded graph resources.
7. Verify measured allocation against the plan and publish the effective profile.

Admission uses exact allocated strides, page/tail rounding, rank ownership, and library residency. Logical token rates are cross-checks.

Reserve every GPU-admitted sequence's complete future window until its GPU entitlement is released. GPU-only retained sequences keep it for their retained lifetime; qualified tiered retained sessions keep full backing claims and reacquire full GPU claims before restoration (Section 17). Short prompts, presumed sharing or unimplemented offload cannot increase GPU-resident capacity.

Capacity may fall between policy thresholds: retain a computed 18. Buckets aid execution; they neither round down admission nor impose a permanent 32-session ceiling.

### Method-specific resource description

Each method declares weights/state, history geometry, proposals/verification rows, target/draft tentative and feature rows, recovery, supported shapes, validity maintenance and stop/commit rules.

Guard rows follow the method's tail policy and page/pooling rounding. The reported +4 belongs to the historical receipt's verification geometry, not every MTP depth or DFlash block.

Clamp proposals to remaining context/reply budget, supported positions, and stops. Physical guard rows neither extend advertised context nor permit unsupported positions.

## 8 Capacity derivations and profile cross-checks

### Unique-history rates

For the required geometry, one head's K+V costs 544 B/row/layer in Q8 and 272 B in the pinned compact Q4 layout. Completed FP16 indexer keys cost 256 B per four-token group, amortized to 64 B/token/layer. Raw pending state is bounded rather than growing with context (Section 9). Rows include visual expansion; verify imported geometry and physical strides before applying these rates.

For PP1, with two global KV heads stored once, balanced over eight ranks, and one copy of pooled indexer history:

```text
13-cache Q8 rate = 13 × (544 / 4 + 64 / 8) = 1,872 B/token/GPU
13-cache Q4 rate = 13 × (272 / 4 + 64 / 8) =   988 B/token/GPU

12-cache Q8 rate = 12 × (544 / 4 + 64 / 8) = 1,728 B/token/GPU
12-cache Q4 rate = 12 × (272 / 4 + 64 / 8) =   912 B/token/GPU
```

Twelve caches cover the target only; immediately available stock MTP needs its additional resources. DFlash adds its actual draft resources to twelve target caches.

For a head-sharded TP8 FP32 GDN state, the current-state payload alone is `36 × 6 × 128 × 128 × 4 = 14,155,776 bytes = 13.5 MiB/session/rank`. Charge convolution/PLE state, layout padding, checkpoints and speculative/replay versions separately. This is a geometry cross-check, not the historical 27.798 MiB subset or a complete per-session ledger.

### Full-context history arithmetic

The following tables describe **PP1 ideal average growing-history costs**, excluding raw rings/staging/recovery versions, guards, page rounding and owner imbalance. Section 20 gives PP2’s stage-specific rates; do not apply a thirteen-cache PP1 mean to every PP2 rank. MTP assumes thirteen QSA caches whose K/V use the selected target codec; pooled indexer keys and raw pending state remain FP16. For a different MTP cache format, sum twelve target caches and the actual MTP cache independently.

| Resident sequences at 262,144 positions | Q8 MTP history GiB/rank | Q4 MTP history GiB/rank |
|---:|---:|---:|
| 8 | 3.656250 | 1.929688 |
| 12 | 5.484375 | 2.894531 |
| 16 | 7.312500 | 3.859375 |
| 20 | 9.140625 | 4.824219 |
| 24 | 10.968750 | 5.789063 |
| 28 | 12.796875 | 6.753906 |
| 32 | 14.625000 | 7.718750 |

| Context/session | Q8 MTP history/session/rank | Q4 MTP history/session/rank |
|---:|---:|---:|
| 65,536 | 0.114258 GiB | 0.060303 GiB |
| 131,072 | 0.228516 GiB | 0.120605 GiB |
| 262,144 | 0.457031 GiB | 0.241211 GiB |
| 524,288 | 0.914062 GiB | 0.482422 GiB |
| 1,048,576 | 1.828125 GiB | 0.964844 GiB |

The two extended-window rows are arithmetic examples pending Section 6's positional/quality qualification, not advertised supported lengths.

Add the selected profile's actual target/MTP/vision weights, all retained state, recovery, execution, graph/communication residency and overlap obligations. Neither a count in this table nor an earlier loader receipt establishes SC117 capacity. Fixed/session costs also prevent a rule that halving context doubles capacity.

Under Section 9's indexer mapping, sixteen and twenty-four MTP sessions balance to 26 and 39 indexed caches per rank. Twenty sessions produce 32 or 33 per rank; twenty-eight produce 45 or 46. Use actual worst-rank ownership, weight placement and visual residency, not averages or rounding to policy thresholds.

### Selected DFlash history cross-check

Conditional on qualifying the TP8/FP16 Inco draft KV profile and context C of at least 2,048 tokens, the ideal unique growing-history cost per resident sequence per GPU is:

```text
target Q8 KV + pooled indexer: (1,728 × C) bytes + 5 MiB of draft KV
target Q4 KV + pooled indexer:   (912 × C) bytes + 5 MiB of draft KV
```

At C=262,144, this is 0.426758 GiB/session/GPU for Q8 target KV or 0.227539 GiB for Q4. FP32 draft KV doubles the 5 MiB draft term to 10 MiB; another qualified representation uses its actual strides. Add target-only weights, the selected draft weight plan (approximately 0.429022 GiB/rank for the FP16 reference), and the remaining ledger, including bounded raw indexer state. Shorter contexts use draft retention up to `min(C,2048)` plus tentative/rollback rows. These cross-checks exclude page rounding and do not establish final session counts.

## 9 Canonical paged history and state ownership

For each TP8 stage, store each target KV head once across its four-rank group and each owned layer’s completed pooled indexer history once across the stage, with bounded raw pending state at the same owner. Section 20 splits PP2’s layer histories between its two eight-GPU stages; no target weight label alone establishes their history or method placement. Section 17 governs complete idle-session suspension/restoration; executing attention never reads a partially restored or host-paged history. Initially compute each session/layer indexer at one canonical owner in that layer’s eight-GPU stage. Retain target TP8 projections and output layout. Its four indexer query heads and aggregation are distinct from the 24 target query heads and two KV heads; rank-local or per-KV-head approximations are not equivalent.

Within each eight-GPU stage, assign the indexer owner as `(local_indexed_cache_count × slot_id + local_cache_ordinal) mod 8`, with profile-fixed local ordinals and slot IDs. Charge actual ownership across every configured slot and preserve its phase on reuse. PP1 uses thirteen caches with QSA-MTP or twelve with target-only/DFlash; PP2’s initial 24/24 split uses six on the first stage and seven with MTP, otherwise six, on the last. Total-count divisibility does not replace this ownership calculation.

### Page and session metadata

Each logical page records its session generation, attention layer, global head, absolute sequence-row range, valid rows, codec/schema, immutable rotation identity, physical owner/offset, ownership generation, and decoding metadata. Keep sequence-row indexing distinct from target multimodal coordinates. Device tables and arena addresses remain stable for captured execution.

Session state distinguishes:

- Canonical committed transcript, externally delivered-output position, per-sequence sampling/constraints and Section 12’s complete output bookkeeping.
- Target-consumed/cached position and recurrent/convolution state.
- Admitted input range and its committed or pending boundary, including partially processed prefill.
- PLE rolling original expanded token/hash inputs and convolution tail; media content/profile identity, modality spans/grids, target M-RoPE continuation state and required replay-source ownership.
- Draft-valid position and method-specific conditioning state.
- Committed and tentative KV/pooled-indexer lengths and raw pending-group state.
- Stable logical grouping origin and pending members' logical positions and M-RoPE coordinates.
- Selected context/reply limits and stop state.
- Page ownership generation and slot generation.

Guard slot reuse with lifetime/generation checks and reader retirement so stale graphs, transfers, or completion callbacks cannot write into a new sequence.

Choose token-page lengths aligned with four-token indexer pooling; qualify 128-token pages as an initial candidate. Quantization blocks partition the head/value dimension, not the token axis: preserve their independent alignment and row strides without imposing a 128-token page requirement merely because iso4 uses 128-value blocks. Charge actual payload/metadata strides, including row padding caused by odd-sized quantization structures.

### Compact indexer construction

Keep raw indexer keys unnormalized and unrotated. Average each complete four-row group in FP32, cast to the configured raw dtype, apply key RMSNorm, then rotate at the first member's coordinates. Reuse the target's 64-dimensional rotary transform within the 128-dimensional indexer head, including its theta and interleaved M-RoPE. Normalize/rotate the four query heads at the query coordinates; score a group by `sum_h ReLU(q_h · k_group) / sqrt(128)` with reference FP32 score/reduction arithmetic. Select up to 512 visible complete groups, expand all four members of each, and append the up-to-three visible tail positions. Sources: [pinned indexer](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/attention/qsa/qsa_indexer.py) and [pooling helper](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/attention/qsa/kernel.py).

For valid cache length `n` measured from that cache's fixed grouping origin, retain `floor(n/4)` completed pooled keys and `n mod 4` pending raw rows. Use a four-row FP16 raw ring per indexed cache/sequence: 1,024 payload bytes before metadata, placement, padding and recovery versions. Preserve required member coordinates and logical positions. Release completed raw members once pooling and all required tail/checkpoint captures have finished and readers retire. Source: [pinned ring metadata](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/attention/qsa/metadata.py).

This fixed grouping assumes each query sees a contiguous valid causal prefix from the declared origin. Batch padding and other sequences never enter that prefix. Arbitrary holes, shifted visibility or origin resets require a separately proven pooling/selection policy; a causal length alone cannot describe them.

Protect the pre-step pending tail and bounded new raw span until compression and accepted-prefix commit resolve; writing an entire verification span into the same four-slot ring can overwrite required carry-in members. Form crossing groups from the saved tail plus new rows. If `a` new cache rows become valid, retain `floor((n+a)/4)` complete pools and the ending `(n+a) mod 4` raw tail; invalidate groups containing rejected members. Each query sees only groups complete at its own causal frontier, even when later pools are already computed. Support arbitrary chunk splits; padded rows do not advance grouping.

Checkpoint each recoverable boundary's pooled-valid frontier, pending raw values and coordinates with its other state. Whole-append rollback restores this checkpoint and discards later pools. A long pending append uses protected checkpoint tails plus bounded current-span storage. Use each cache's own validity frontier, especially MTP's shifted/deferred rows; its target-conditioned refresh rebuilds affected groups. Physical migration preserves grouping origin and membership.

Preserve the branch’s issued-coordinate/frame authority, mutation visibility and separate target/MTP staging lifetimes. Revoke stale authority before fallible replacement, bind every issued batch segment to the final coordinate/mutation identity and validate it before persistent computation. Inference mode, native writes or graph replay must not hide mutation from that authority mechanism. Stable addresses alone do not establish a valid frame. Preserve these contracts through media splits, rollback, restore and slot reuse independently of optional ownership migration. The [compact-indexer contract](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/docs/compact-indexer.md) and current `compact_indexer.py`/`state.py` are concrete starting controls, replaceable only with an equivalent cheaper qualified implementation.

### Initial owner execution

For each target attention layer:

1. Execute the existing TP8 projections.
2. Route each query head to the qualified holders of its corresponding global KV head history under the fixed owner map.
3. Append incoming KV once at its canonical page owner, in the committed or tentative range required by the execution method; promote only the accepted prefix.
4. Construct exact indexer queries and update the bounded raw pending state and pooled keys once at their canonical owner under the same committed/tentative transaction.
5. Select visible complete pooled groups, expand all their members, and append the visible incomplete tail.
6. Dispatch selected logical positions to the KV owners.
7. Compute local attention, merge disjoint selected-position partials under Section 10, and return outputs to their original TP positions.
8. Continue the existing target output projection and distributed execution.

Batch routing and collectives by layer, not request. Preserve per-row causal boundaries, masking, pooling, ties, deterministic logical positions, and exact sparse selection. Later tentative appends must remain invisible to earlier queries. Replicated histories do not establish replicated indexer query inputs.

Pin score-tie handling and logical selected-token accumulation order in the numerical profile. Physical page addresses, free-list order, slot reuse or ownership migration must not silently choose a different reduction order; degree-specific partial merges use their declared qualified reduction trees. Where a source leaves equal-score ordering unspecified, declare and qualify the chosen rule rather than claiming bitwise reference equivalence. Test identical logical histories under permuted physical allocations.

Transient projection duplication is initially permitted; canonical growing histories remain uniquely owned. Additional bounded read copies require Section 1’s duplication policy, complete validity/invalidation rules and measured benefit.

### Prefix reuse

Compatible prompt-prefix reuse is a required implementation/qualification workstream. It avoids repeated model work; physical cross-session page sharing remains optional and full-context capacity assumes cache misses and private growth. Use one page/state manager for active histories, reusable prefixes and Section 17's queued checkpoints, with explicit ownership classes rather than duplicate cache stores.

Within an explicitly allowed cache-sharing domain, match the exact rendered/expanded token prefix and complete dependent identity: target/runtime weights and arithmetic, tokenizer/template, codec/rotation, RoPE/M-RoPE/positions, model/feature contracts and required companion state. Include immutable media content, processor/settings, grids/order and target continuation state; equal placeholder IDs with different pixels cannot hit. Use canonical serialized metadata and collision-resistant prefix/page keys, validating identity/lengths rather than trusting a filename or short hash. Bound tokenization/render/media-result caches separately and key each by its own complete dependencies. Reuse model state, never a previous response as the new answer.

Bind any next-token or next-visual-embedding lookahead already consumed by MTP/draft state at a cached boundary. Equal target prefixes do not validate companion state dependent on different following input. Match those dependencies, restore an earlier common checkpoint, or use the qualified bounded companion reconstruction path; do not discard otherwise reusable target state unnecessarily.

Select the deepest stored boundary with every required component: target KV/pooled keys/raw group tail, GDN and PLE recurrence/convolution/token-hash state, positions, and method conditioning or a qualified bounded reconstruction path. A later recurrent checkpoint cannot be cropped backward, and taking the minimum of component lengths does not create a missing checkpoint. Reuse inside a group requires the matching raw-tail snapshot or replay from an earlier complete boundary; pooled means cannot reconstruct discarded keys. Retain a bounded set of useful message/chunk/end boundaries and budget their actual state versions.

An exact whole-prompt hit still needs valid next-logit/hidden state under the method's cached-versus-committed convention, or a qualified earlier boundary from which to execute required rows. Never consume the final prompt row twice into recurrence or pretend KV alone supplies its logits. Honor requested prompt log-probabilities: skip work only when the cache contains the exact required values. Canonical rendering changes require longest-prefix matching again, not reuse by chat/session label alone.

For a new request or branch, initialize its own sampling/RNG, constraints, penalties, stops and output state from its API contract and prompt; do not inherit another request's generation bookkeeping. An interrupted same-request continuation preserves generation bookkeeping; a new turn in a retained conversation initializes new-request bookkeeping while reusing compatible model state. Read-only shared prefix pages require reference counts and copy-on-write of mutable/partial pages plus private recurrent/draft/sampling state; otherwise use a budgeted independent copy or move exclusively owned idle state. No write may invalidate a longer retained branch or another reader.

Retain useful prefixes only within a bounded reclaimable budget after guaranteed GPU/host/backing and execution obligations. Reclaim unleased reusable entries before suspending useful work; cache residency cannot strand promised full-window page classes. Publish entries only at validated state boundaries, and acquire a lease before selecting/restoring them. A reusable-cache miss/corrupt entry permits ordinary qualified prefill; loss of the sole required queued checkpoint is a session failure. Section 17 defines shared tier transfer and backing-object lifetimes.

References: [TensorFold CUDA prefix selection](https://github.com/kklouzal/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/families/qwen4_exp/cuda/prefixes.py), [Flash Next cache tests](https://github.com/kklouzal/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/tests/cuda/test_flashnext_prompt_cache.py), [vLLM prefix identity/block ownership](https://docs.vllm.ai/en/latest/design/prefix_caching/), and [SGLang hybrid reuse boundaries](https://www.sglang.io/blog/unified-radix-cache). These inform integration; they do not qualify SC117/SM70/TP8 snapshots.

## 10 Fixed attention ownership and exact merging

For each released TP8 stage/profile, first qualify one fixed unique KV ownership map within its two four-rank KV-head groups. Four-way page striping within each group is the initial control to measure; it is not a demonstrated performance winner. Derive the exact worst-rank ledger and compare another fixed degree only where a measured bottleneck gives it a plausible case. Supporting every degree 1/2/4 or converting between them live is not an initial implementation or release obligation. These are internal placement choices, not another public TP/ETP/PP axis.

The pinned source already owns unique history: `history_placement.py` defaults target degree to 1, `engine.py` constructs that map, `multi.py` enables manual migration for qualifying quartets, and the engine charges a 64 MiB/rank migration allowance. Refactor this existing owner path to allocate the selected fixed map directly at prepared activation, with degree4 as the first control. Initial serving exposes no live-degree update; remove its unnecessary control/lifetime and only its actual dedicated migration liability. Preserve all prefix/COW/restore/recovery/progress obligations and rederive worst-rank capacity. Existing migration algorithms may remain research/reference material where useful; no duplicate serving path or general migration framework is required. See pinned [placement](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/src/tensorfold/families/qwen4_exp/cuda/history_placement.py), [engine](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/src/tensorfold/families/qwen4_exp/cuda/engine.py) and [multi-rank owner](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/src/tensorfold/families/qwen4_exp/cuda/multi.py).

The selected ownership map stays fixed during ordinary activity changes. Local CTA/tile decomposition, query grouping and selected-work lists can vary through prepared kernels without moving retained history. A bounded gather to fewer compute ranks would pay its actual per-step reads/transfers and staging; it does not obtain owner-local degree-one execution for free. Adopt such a dataflow only after complete-chain measurements. Balanced stored bytes can still yield uneven selected-position work.

### Initial fixed target-layer mapping

For local rank indices 0 through 3 in each KV-head group, use this first-control map:

```text
base = (slot_phase + layer_phase) mod 4
owner(page) = base XOR (page_index mod 4)
```

Use profile-fixed slot/layer phases for PP1’s twelve target QSA caches or PP2’s six per stage. Charge actual page classes, partly valid tails, guards, slot populations and all transient layer skew. The earlier twelve-layer uniform-phase balance argument is not a blanket worst-rank proof for a six-layer stage or another layout. Map physical ranks using measured links; reserve required page classes before prefill/verification advances layers.

If a fixed degree-1/2 alternative is tested, derive its complete source/owner/compute plan and exact slot/layer/page maxima separately. The generalized research map `base XOR (page_index mod degree)` requires its common-degree/equal-footprint assumptions; no current requirement authorizes independent live per-layer choices. Select any accepted alternative at prepared profile activation.

The MTP cache is outside the twelve-target-layer proof. Qualify its fixed four-way KV striping within each head group and charge its indexer separately. DFlash uses its own declared head/window map. Exact prefix/restore ownership, page generations and reader retirement remain mandatory even with no live migration.

### Exact partial attention merge

For the same query and disjoint selected positions, each participating rank computes local maximum m_i, exponential denominator l_i, and weighted value numerator z_i. Merge:

```text
m = max_i(m_i)
l = sum_i(exp(m_i - m) × l_i)
z = sum_i(exp(m_i - m) × z_i)
output = z / l
```

Do not average independently normalized outputs. Empty shards contribute an empty softmax state, with correct handling of the all-empty case according to the original attention semantics. Qualify the declared precision/reduction tree, masking and logical accumulation order.

Keep the pooled indexer canonical in the initial plan and route selected positions using the fixed owner table. Live KV placement changes do not parallelize its full pooled scan/top-k. E18’s fixed distributed indexer is a separate conditional experiment with exact global selection and its own map/ledger; no live indexer repartitioner is required.

### Placement adoption gate

Measure the fixed control through actual capacity and C1/2/4/8/16/24, including near-full histories, one active with idle residents, mixed prefill/media and required speculation. All eight GPUs retain their target-model roles. Adopt a fixed alternative only through the existing numerical, capacity and complete-serving performance gates. Section 11’s optional live ownership machinery requires additional transition/payback evidence after the best fixed plan and cheap dispatch have been optimized.

## 11 Conditional live cache migration

Live KV degree changes, migration bundles, mixed-holder execution and hysteresis are a benchmark-gated experiment, with no initial development, profiling sweep or release obligation. Initial serving uses Section 10’s qualified fixed map and makes immediate progress after activity changes. If the experiment is promoted, all requirements in this section become mandatory for that implementation. Complete idle-session suspension/restoration under Section 17 remains required independently.

Migration changes physical ownership while preserving compressed bytes, without a second complete history, precision conversion, or model restart.

### Resource and scheduling rules

Qualify or adjust an initial additional-occupancy cap of 64 MiB/GPU. Count all temporary excess: uncommitted destination copies, staging/bounce space, metadata, and source-retirement lag.

Before moving, verify current occupancy, future full-window reservations, final placement, and every intermediate overlap. Inactive sessions retain their pages. If the move cannot fit, execute from the valid current placement.

For matching layer quartets with equal footprints and common old/new degrees, migrate the same token-page index as a balanced bundle and publish it together. PP2’s six target QSA layers leave an incomplete quartet; unequal footprints, slot populations and tail allocations can also defeat the uniform proof. Precompute an exact per-rank source/destination/intermediate allocation plan for those cases. Use a bounded composite bundle across compatible local phases/slots, or charge the worst-rank skew of a partial bundle; never assume missing phases or borrow another stage’s headroom. Reserve destination page classes and every overlap before copying. If no plan fits, defer the move and serve the valid current map. Bound transfers by time and occupancy; never move an entire layer first and repair temporary imbalance later.

### Commit protocol

1. Select a worthwhile target placement and reserve the next bounded destination bundle.
2. Resolve the affected session's speculative work to its defined committed boundary.
3. Drain readers/writers of the affected pages through explicit stream/event dependencies.
4. Copy complete packed payload and metadata. Source pages remain authoritative until completion.
5. Publish a new page-ownership generation after the required participants acknowledge copy completion.
6. Retire old copies only after all readers of the previous generation have finished.
7. Replenish the bounded free-page window and schedule further bundles when appropriate.

Pause only affected work where supported; other requests can run between transfer chunks. Qualify independent-stream overlap, retaining deterministic communication ordering with or without overlap.

### Intermediate ownership is normal executable state

After partial migration, execute committed moves through the qualified union of current holders in the mixed owner table. Discard only uncommitted copies; never restore mappings whose source pages were freed.

New pages follow the published future-allocation policy; existing pages follow recorded owners. Neither follows a stale session-level degree. Appends during partial migration retain balanced-bundle reservations.

Prepare the union communication path before expansion; retain the larger group until contraction moves every page requiring it. Synchronize table contents even when addresses are stable.

Seal or quiesce mutable tails before copying. Preserve lengths, raw pending tails, pooling origin/membership, generations, sampling state, target/draft validity, codec basis, media identity and target/draft position mappings. Ownership migration never reprocesses media or reinterprets sequence positions.

### Cost policy

Before implementation/adoption, demonstrate a fixed-placement bottleneck and complete-serving opportunity that survives cheap dispatch improvements. Before each adopted move, assess workload persistence, remaining useful work, measured transfer/retirement time, interruption/interference, decision cost and uncertainty. Require expected future serving-time savings to exceed observation, decision, quiescence, transfer, contention, graph/metadata transition and any later reverse-move cost; a concurrency threshold alone is not a dwell-time forecast. No arbitrary percentage cutoff dismisses a small reproducible net gain.

Under the twelve-layer geometry, full owner-to-degree-4 redistribution moves about three quarters of unique target KV: approximately 1.20 GiB Q4 or 2.39 GiB Q8 across the box per full session, excluding indexer movement. Moving all 24 full target histories would transfer 57.375 GiB Q8 or 28.6875 GiB iso4, only if all are actually moved; indexer and separate MTP work remain excluded. These are transfer volumes, not simultaneous duplicate residency or timing predictions.

On demand spikes, stop unnecessary new chunks and use the valid actual placement. Coalesce pending changes and apply hysteresis against repeated threshold movement.

## 12 Speculative execution and accepted-prefix correctness

### One method interface

None, stock MTP, and DFlash supply resource/state requirements to one planner and scheduler through a small execution contract. MTP implementation and qualification are mandatory for the initial SC117 release; DFlash is the separately selected companion workstream. Do not load both speculative systems in one initial resident profile.

Each method defines proposal construction, target row count, target/draft tentative storage, valid position offsets, recovery, feature/state maintenance, and commits, with limits matching actual kernels and buffers.

Resolve MTP per target: use compatible embedded tensors or an explicitly paired companion, without loading both. Family identity does not transfer the initial companion's tensor count, QSA semantics, or footprint. DFlash eligibility requires the Section 14 feature and probability contract.

### Stock MTP

The first packed MTP candidate is `unsloth/Qwen3.8-Flash-Next-GGUF`, revision `766911a6b7369840a91dbcd95f9f997acaab6cd6`, file `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`. Its file size is 2,786,568,256 bytes and SHA256 is `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6`. The inspected tensor payload is 2,775,621,632 bytes, or 2.584999 GiB: nineteen Q8_0 tensors, eleven F32 tensors, and two BF16 tensors. Source: [pinned MTP companion directory and README](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6/MTP).

The official control source is the 31 `mtp.*` tensors in `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d43b917e7706784d8bb445c9af86a3540`: 5,214,301,696 bytes of BF16 payload across the source shards. Extract only the required tensors with bounded preparation memory, retain a checksummed import manifest, and convert to the qualified SM70 representation. The shared-BF16 GGUF companion is also available for import parity checks. Source: [official tensor index](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/model.safetensors.index.json).

Both paths share the actual SC117 embedding/output tensors without replacing the target trunk. Reconcile fused/split projections, transformed RMS weights, native fusion and MTP hyper-connections. Capture completed block-47 HC residuals before the final target HC mixer: all four streams form the 10,240-wide input. Preserve the selected SGLang reference's hidden RMS normalization over that entire flattened width, its 10,240 learned weights and order, epsilon 1e-6, and one application of the `1 + weight` convention; normalize input embeddings over 2,560 separately. The pinned GGUF conversion already folds the Gemma offset into gamma: use that gamma directly, or convert once to zero-centered weights; never add one twice. Only then reshape HC into four streams for the shared hidden projection. Another engine's per-stream normalization is not equivalent merely because tensor shapes match. Never substitute the target's 2,560-wide mixed output, stream mean or DFlash features. MTP disables PLE; target PLE still advances/rolls back during verification. Qualify packed and BF16-derived control profiles separately without permanently loading both.

### Explicit MTP attention policy

The initial ProviderOne MTP profile uses **QSA with compression four and complete-group token budget 2,048**, matching the inspected model-config/SGLang path. The inspected shared-Q8/BF16 GGUF companions have a final MTP compression-ratio metadata entry of zero, which selects dense attention in the inspected llama.cpp path despite the presence of indexer tensors. Therefore the importer must bind and validate the declared QSA policy against the reference; blindly interpreting that metadata would produce a different algorithm. Sources: [SGLang MTP reference](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/models/qwen4_exp_mtp.py) and [llama.cpp target/MTP implementation](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/src/models/qwen4exp.cpp).

Recompute MTP QSA selection for each row/step. Disable `index_share_for_mtp_iteration` in [pinned SGLang parity controls](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/speculative/eagle_worker_v2.py); later selection reuse requires separate qualification.

Verify QSA-MTP import/execution parity using identical decoded weights, QSA policy, numerical settings, inputs, masking, and selected-tail semantics. Compare packed-Q8 and BF16-derived control profiles for acceptance, useful throughput, and quality; different weight encodings need not yield identical proposals. Dense MTP requires a separately selected profile with its own attention-work, allocation, and throughput proof. The receipt's thirteenth indexer allocation does not establish which path executed.

Enabled MTP retains its weights, thirteenth QSA cache and per-session state. Initially its K/V codec matches the selected target Q8 or RotorQuant iso4 codec, with FP16 pooled indexer history and bounded raw pending state; record and qualify this binding separately from MTP weight precision. A different MTP KV format requires its own ledger and acceptance profile. Target rows per scheduled sequence equal one plus speculative depth:

| Depth | Target rows/sequence | Sequences fitting in R32 |
|---:|---:|---:|
| 0 | 1 | 32 |
| 1 | 2 | 16 |
| 2 | 3 | 10 |
| 3 | 4 | 8 |

R32 limits each step, not residency to eight sessions at depth 3. Sixteen depth-3 sequences verified together need 64 rows; 24 need 96. Measure fair microbatching's latency and throughput.

Qualify R32/R64/R96/R128 or other useful envelopes by extending index ranges, kernels, workspace, recovery buffers, and graphs together. A larger graph alone does not extend an R32 kernel.

### Native MTP initialization and refresh

Construct MTP history by pairing completed target HC row `h_t` with the actual merged input embedding at following position `t+1`, aligned with its token ID and the pinned position/cache-slot convention. Text uses the token embedding; visual rows use projected media embeddings. Non-final chunks use actual next prompt input; only a completed prompt with generation permitted uses a sampled text anchor. If its token or visual embedding is unavailable, retain the bounded HC boundary feature and defer the MTP row. Sources: [pinned MTP worker prefill/refresh path](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/speculative/eagle_worker_v2.py) and [chunk-boundary token handling](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/speculative/eagle_utils.py).

Until lookahead input commits under the retention contract, keep dependent MTP boundary state tentative and retain its HC feature plus actual next embedding, or a budgeted recovery path. Cancellation/replacement invalidates that dependency. The reference shifts multimodal embeddings with next-token IDs, but its terminal text lookup cannot be copied at a chunk boundary whose next row is visual; supply the correct embedding or disallow that split until qualified (Section 16).

Populate full-prefix MTP KV and completed pooled indexer history in bounded feature batches, using Section 9's raw-state transaction; the final target hidden state or DFlash's window-only initialization is insufficient. Optimize unused outputs only after proving equivalent cache construction and next-proposal state.

After verification, restore MTP's valid pre-draft pooled frontier and pending raw tail, then reconstruct retained new MTP rows and pools from verified target HC features and aligned accepted/correction/bonus tokens. Cropping provisional MTP hidden states/KV to accepted length does not perform this target-conditioned refresh. Maintain it during zero-proposal steps when promising immediate reactivation. Charge execution and temporary storage without an unnecessary full-prefix target-feature archive.

Verify one-token prompts, text/visual chunk boundaries, timestamps/delimiters, resumed mixed prefixes, every rejection offset, correction/bonus handling and zero-to-positive transitions against the selected reference semantics.

### Warm capability versus current proposal length

Zero proposals retains weights, valid history/state and execution prerequisites for immediate reactivation; include maintenance time. Warm zero proposals is not the timing/resource baseline of a separately selected no-speculation profile. Reclaiming method resources requires a configuration transition and may require reconstruction.

After a verdict establishes completion, EOS, cancellation or exhausted budget, omit subsequent speculative refresh/materialization only when no admitted continuation, protected retained compute checkpoint or chosen coherent reusable-prefix entry will consume it. Request-scoped disposable state normally needs no post-verdict warm maintenance. Retire already-issued work/readers normally; earlier feature capture can be omitted only when plan construction already proves the absence of every consumer. Stored logical API items alone do not promise a computed-state checkpoint.

### Accepted-prefix transaction

Verification must commit only the accepted and permitted prefix across every state type:

- Target KV, pooled indexer history and bounded raw pending state.
- Recurrent state and convolution tails.
- PLE rolling token/hash state and PLE convolution tail.
- Pooling origin, completed-group frontier and pending members' coordinates.
- Draft KV and conditioning state.
- Logical/emitted/target-cached/draft-valid positions, multimodal continuation metadata and any media/replay dependencies.
- Enabled grammar/constraint cursors, repetition/frequency/presence counts, incremental detokenization/stops and structured-output bookkeeping defined below.

Isolate tentative updates to this bookkeeping and commit only the permitted prefix, or reconstruct it from the canonical transcript before reuse. Rejected candidates must not alter later penalties, constraints, stops or published text. Charge retained state and bounded tentative work to their actual host/device owners.

Follow the method's proven sampling algorithm, preserving its RNG advancement through per-sequence proposal, acceptance, and target/correction streams or draw identifiers. Rejected proposals, acceptance tests, and correction sampling may consume draws; rejection does not authorize rewinding RNG to the accepted-token count.

The 36 recurrent layers require accepted-prefix state selection or proven checkpoint/replay; KV cropping cannot undo recurrence. Budget and measure the strategy, including longer blocks' additional intermediates or recomputation.

EOS, cancellation, and context/reply-budget exhaustion can further shorten the committed prefix. Later evaluated tokens must not become visible or permanent state. Preserve the service's token-versus-text stop boundary; an intra-token byte stop does not permit KV cropping at a character offset.

### Commit and output publication

One coordinator resolves verification, limits, stops, and method-defined target/draft state before publishing the accepted prefix to the output queue. Never stream tentative tokens. The server's committed transcript governs retained-session resumption, including after disconnection; network delivery is not model-state authority.

Apply cancellation at a coordinated scheduling boundary. Finish or safely retire the issued plan and prescribed communication before releasing pages/slots. Cancellation cannot retract published output. Generation/lifetime checks must prevent late completions writing into reused slots.

Use bounded asynchronous per-request output queues. Backpressure pauses that request or applies its timeout/cancellation policy without blocking unrelated GPU progress. Detect stop strings across token, speculative-block, and stream-chunk boundaries, withholding text that could expose a partial stop marker. Test against the non-speculative path.

For exposed reasoning/tool-call output, retain incremental parser/channel state, stable endpoint-specific response/item/call identifiers and bounded queued/published event or delta frontiers separately from token counts. Advance canonical parser/publication state only for the committed prefix allowed by the endpoint’s token/byte-stop and channel contract; keep lookahead or tentative parser state private and commit/recover only that prefix. Preserve this state or a qualified bounded reconstruction across same-request continuation, including partial markers and argument fragments; new requests initialize their own output state. Internal publication accounting follows Section 17’s delivery/retry limits and adds no reconnect/replay guarantee. Source: [standard function-call streaming lifecycle](https://developers.openai.com/api/docs/guides/function-calling).

### Sampling and numerical guarantees

Verify speculation against the configured target computation and sampling policy. Greedy matching does not establish stochastic acceptance/rejection. Apply temperature, truncation, penalties, constraints, and stops at the correct conditional prefix. Candidate-selector DFlash requires the corresponding proposal probabilities.

Lossless speculation is relative to the configured numerical target, not Q4/Q8 equivalence. Declare tolerances and sampling guarantees for reduction-order differences; universal bit-identical text is not promised.

Keep per-sequence RNG ownership independent of batch permutation and migration. Padded rows must not advance another sequence's RNG or state.

## 13 RotorQuant and future codec work

### Initial Q8 contract

Preserve the qualified ordinary Q8 profile’s numerical encoding/decoding contract; replace its inherited implementation as needed. Pin writer, reader, attention-consumption source and receipt-build revisions. Before kernel or ownership changes, record signed-code interpretation/range; grouping/layout; scale calculation, FP16 storage and reconstruction; rounding/ties; saturation; zero blocks; and nonfinite handling. Its 544 bytes per target-head K+V token establish capacity, not external-format compatibility.

Paged storage, packing, and attention must reproduce the pinned semantics. Verify byte compatibility for identical representations and qualify execution-induced numerical differences. Migration copies bytes/metadata without requantization. Require source-confirmed equivalence before calling it GGML Q8_0. A different codec requires a separately selected, qualified numerical profile; never reinterpret retained pages silently.

### Format requirements

The initial Q4 profile is **K=iso4, V=iso4**, specifically `GGML_TYPE_ISO4_0` / `block_iso4_0` at reference commit `08e025c06ab521e4fa9e5c08b80af57614543e53`. This starting profile is neither a measured V100 winner nor a universal upstream default. The author's limited four-bit comparison favors IsoQuant; qualify actual SC117 long-context quality and throughput.

Each iso4 block represents 128 values with 32 quaternion rotations and occupies 68 bytes: 64 bytes of codes plus two FP16 metadata fields. Preserve the pinned centroid/quaternion tables, reconstruction-norm correction in `norm`, and reference zero initialization of the reserved `rnorm` field. A dimension-256 K or V head uses two blocks; K and V together occupy 272 bytes per head/token/layer before allocation padding. The existing Signed128 format has different metadata and numerical semantics despite the same nominal footprint.

Bind the reference's `TURBO4_USE_4BIT=1` layout explicitly: `block_iso4_0` aliases a structure whose legacy three-bit-plus-QJL alternative also occupies 68 bytes. The selected profile uses head-axis, rotated K and V blocks, with normalization/correction and the 32-entry quaternion table restarted per 128-value block. The reference's no-rotation transposed-V helpers are a different path, not interchangeable readers/writers under the same schema identity. Pin flags, axis and basis with the codec; a type name and byte count alone cannot establish compatibility.

RotorQuant requires independent qualification of format roundtrips, zero vectors, scales/norms, rotations/centroids, quality, peak memory, prefill, and decode; Signed128's live qualification does not transfer.

Sources: [RotorQuant project](https://github.com/scrya-com/rotorquant) and [audited reference schema](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-common.h).

Use planar4/planar4 as the first equal-memory comparison after iso4; its simpler two-dimensional rotation does not establish a faster V100 path. Do not start with unsupported mixed variants. Administrators select qualified codec profiles; activity thresholds adjust execution within that codec, without silently requantizing retained pages. Exact implementation references: [GPU packing and norm correction](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-cuda/set-rows-planar-iso.cuh), [centroids and rotations](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-cuda/planar-iso-constants.cuh), and [attention transforms](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-cuda/fattn-common.cuh).

### Reference integration boundaries

The audited llama.cpp commit allocates deferred FP16 K plus a compressed destination and provides CPU conversion. Use direct GPU packing and bounded progressive-prefill storage in ProviderOne; a full FP16 history shadow or full-history conversion peak invalidates compact-only capacity. Source: [pinned cache allocation and conversion code](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/src/llama-kv-cache.cpp).

The reference dispatcher uses vector Planar/Iso attention, which does not establish optimized SM70 prefill or large verification. Implement and compare small-row and tiled ProviderOne paths. Source: [pinned attention dispatch](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-cuda/fattn.cu).

### Rotation and compute

Preserve target positional-transform order; general codec rotations need not commute with RoPE. Where the pinned codec permits, rotate queries consistently with stored keys, accumulate values in the proper basis, and inverse-transform the output once. Apply every norm/scale correction; orthogonality alone does not prove post-quantization equivalence.

Unpack into register/shared-memory tiles without expanding full history per step. Specify aligned payload/metadata layouts; migrate packed bytes without requantization.

Qualify long-context prefill with immediate history quantization; FP16-prefill/deferred-conversion quality does not transfer.

### Deferred comparison

SAW-style token-wise INT4 with lightweight key rotation remains the next codec candidate after the main system works. Keep page/ownership and scheduling contracts independent of pack/unpack, within one serving framework.

Review SAW's modern-GPU attention and full-history prefill expansion for the V100 port; it is not a drop-in implementation. Sources: [SAW paper](https://arxiv.org/abs/2604.19157) and [reference implementation](https://github.com/togethercomputer/saw-int4). Consider KIVI, KVQuant, TurboQuant, or other non-prerequisite approaches only for a concrete measured quality/performance need.

## 14 DFlash profiles and initial DFlash 2 integration

### Additional draft profiles

Use the same interface for administrator-selected DFlash checkpoints and source/runtime formats. Validate feature/HC boundaries, shapes, tokenizer/vocabulary and embedding/head alignment, masks, RoPE/window bounds, block conventions, selector/proposal probabilities and commit/recovery. Family resemblance is insufficient; narrowed vocabularies need explicit token/probability mapping and verification.

Derive physical retention per layer from its actual full-prefix, sliding-window, or other supported policy. Full-prefix storage grows with context; windows save memory only through bounded allocation. Do not transfer the initial Inco profile's geometry, state, block convention, or memory totals to another draft. Budget actual persistent recurrence/convolution state, feature buffers and selector tables. Reused target embeddings/heads use real storage sharing or qualified owner-served access with bounded activation transfer by default; deliberate copies require Section 1’s duplication policy and explicit accounting instead of being reported as aliases. Independently learned draft tensors are distinct parameters and require their own numerical and residency inventory.

Qualify acceptance and useful throughput against exact target weights, effective arithmetic, KV codec, and generation policy. Matching dimensions do not preserve profitable proposal length, capacity, or state compatibility. Cache identity includes the numerical target/draft and feature contract; model changes cannot reuse KV, recurrence, or draft conditioning based only on shapes.

### Selected artifact and alternatives

Use **incoai/Qwen3.8-Flash-Next-DFlash2**, revision **a22a1744e85c17364dcc54a6a420af7135d4c7f6**, as the first DFlash candidate to implement and qualify against the pinned SC117 target. Import its source `config.json`, `model.safetensors`, and feature-capture contract. The initial control candidate is **fixed TP8, FP16 matrix weights/operands, a physically bounded draft window, and block size four**. Section 20 requires separately qualified draft placement and feature/endpoint interfaces for every additional target profile; changing target ETP does not change this dense draft’s geometry. FP16 activation/output/KV storage is conditional on the range gate below; the following FP16 payloads are reference calculations, not established safe runtime allocations.

Select execution settings from V100 measurements. Published evaluation uses the official FP8 target on a GB300 at TP1, without disclosing the exact training target revision/precision or establishing SC117, RotorQuant, TP8, or V100 performance. Source: [author's model card](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2).

| Candidate family | Disposition for this deployment |
|---|---|
| IncoAI exact-family DFlash 2 | Selected: source checkpoint, precise capture patch, native bounded window, and model-family serving evidence |
| PixelML Qwen3.8-Flash-Next NVFP4 DFlash | Independent older candidate with different feature/anchor conventions, full-prefix draft history, and limited task evidence; not the initial high-concurrency choice |
| hotocooo DavidAU GSQ-RCO draft | Fine-tuned for a different derivative/quantization; an abliterated label does not establish superiority for SC117 |
| GGUF/EXL3 and engine-specific conversions | Alternative encodings of those drafts; no independent throughput or acceptance advantage follows from repackaging |

Sources: [PixelML artifact](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash), [DavidAU-specific draft](https://huggingface.co/hotocooo/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-DFlash), and [IncoAI GGUF conversion](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF). No published result establishes this exact SC117 IQ3_S/V100/multimodal pairing; qualify it independently.

### Exact draft manifest

| Property | Value |
|---|---|
| Architecture | DFlash2DraftModel |
| Source tensor file | `model.safetensors`, 1,206,140,008 bytes; SHA256 `d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f` |
| Source parameters and dtype | 603,065,600 BF16 parameters; 1,206,131,200 payload bytes |
| Hidden / intermediate / depth | 2,560 / 7,680 / five layers |
| Attention heads | 32 query heads; eight KV heads; dimension 128 |
| Target feature taps | Completed zero-based blocks 5, 15, 25, 35, 45 |
| Feature input | Mean over four HC streams at each tap; concatenate five width-2,560 vectors to width 12,800 |
| Draft attention | All five layers use a 2,048-token sliding window and checkpoint-defined non-causal block attention |
| Position configuration | Published configuration maximum 262,144; draft RoPE theta 10,000. Extended absolute positions need Section 6's separately qualified draft policy; the 2,048-row ring does not extend this limit. |
| Configured / initial serving block | Configuration says eight; initial qualification candidate uses four |
| Candidate selector | Rank 256; top 16 candidates |
| Dynamic convolutions | Kernel width two; group size 16; block-local and stateless across blocks |
| Vocabulary / mask token | 248,320 / 248077 |
| Embedding and output projection | Borrow the selected target's actual tensors; absent from the draft safetensors payload |

Sources: [pinned configuration](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/blob/a22a1744e85c17364dcc54a6a420af7135d4c7f6/config.json), [pinned tensor file](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/blob/a22a1744e85c17364dcc54a6a420af7135d4c7f6/model.safetensors), and [capture patch](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/blob/a22a1744e85c17364dcc54a6a420af7135d4c7f6/patches/qwen38-dflash2-capture.patch).

### Target feature capture

Capture completed outputs at the five specified blocks, average over four HC streams, concatenate in declared order, and apply trained feature projection/normalization. Do not substitute pre-layer residuals, final target hidden states, or learned HC contractions.

The reference capture patch asserts TP1/CP1. Implement TP8 capture for ProviderOne's actual residual sharding/replication and verify parity across ranks, chunk sizes, prefill, and verification. Removing the assertion cannot establish equivalent features. The author's GGUF card identifies missing HC-mean capture as a llama.cpp blocker; downloadable weights alone are insufficient.

Keep native four-stream target computation; take the mean only for draft features. Qualify exact SC117 IQ3_S features and acceptance for both target KV codecs, including visual prefixes, chunking, corrections and continuation positions.

### Physical draft memory plan

Reference placement shards attention/MLP projections and replicates feature projection, dynamic-convolution parameters, selector codebooks/projection, and normalization arrays. Conditional on the qualified FP16 weight representation, the TP8 payload is:

| Weight group | Source payload bytes | Bytes per TP8 rank |
|---|---:|---:|
| Attention and MLP projections | 851,968,000 | 106,496,000 |
| Remaining replicated parameters | 354,163,200 | 354,163,200 |
| Total | 1,206,131,200 | **460,659,200 = 0.429022 GiB** |

The two selector codebooks account for 254,279,680 bytes per replicated rank. Dividing the whole checkpoint by eight would undercount this placement. Source: [pinned SGLang DFlash model implementation](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/models/dflash.py).

Account for padding, qualified replacement-format residency, activations, graphs, communication, feature buffers and sampling scratch. KV-materialization helpers use source-weight aliases or bounded tile decoding by default; target embedding/head access shares actual storage or uses the qualified owner service. Any retained alternative representation or endpoint copy requires Section 1’s duplication policy and explicit non-alias accounting. A replacement FP32 or fused runtime representation releases its superseded GPU copy after bounded conversion/validation. Additional retained full stacked or decoded weight copies require Section 1’s duplication policy; charge conversion overlap and tile workspace regardless of acceptance.

At TP8, each draft rank can own one of its eight KV heads. An unreplicated FP16 window costs:

```text
2 × 5 layers × 8 KV heads × 128 values × 2 bytes × 2,048 positions
= 41,943,040 bytes = 40 MiB/session across the draft
= 5 MiB/session/GPU at TP8
```

Enforce bounded physical allocation for the committed window plus tentative rows, page alignment, and rejection/rollback protection. Future writes must not overwrite committed entries needed after rejection. Retire committed entries only at the accepted-prefix boundary through bounded tentative pages or an equivalent proven ring transaction.

**Finite-range gate.** FP32 accumulation does not make a subsequent FP16 output cast safe. Probe real-checkpoint SC117 HC features, projected outputs/residuals, pre/post-normalization values, selector logits/probabilities and materialized KV through prefill, long decode, rejection and window wrap. The companion's A37 records an unmerged exact-Inco Strata implementation with an overflow reproducer and FP32 outputs/KV; that is concrete test evidence, not a reproduced ProviderOne defect or a blanket FP32-weight requirement. Source: [pinned overflow fixture](https://github.com/jmnargi/Strata-V100/blob/baede60aa1d8ccf5a7b8ed83822da4f1229f92ef/tests/core/dflash2_overflow_test.cu).

If FP16 storage fails, qualify necessary FP32 outputs/KV or another validated representation before enabling this draft. Do not silently clamp/saturate or accept nonfinite proposal probabilities. FP32 window payload is 10 MiB/session/rank rather than 5 MiB; recompute weight replacements, activations, rollback, graphs, transfer/checkpoint schema and capacity for the actual profile. Finite values alone do not establish parity, useful acceptance or a throughput gain.

Preserve absolute scalar sequence-row positions and draft theta 10,000 when reusing ring slots; visual rows count in that sequence. Target theta is 10,000,000 and its multimodal coordinates/continuation delta remain separate (Section 16). Budget or avoid full-length trigonometric tables: 128 FP16 entries × 262,144 positions cost 64 MiB/rank, doubled for FP32. Share applicable tables across the five draft layers.

### Bounded prefill and warm maintenance

Apply feature FC/hidden norm, then each layer's K/V projections. Apply per-head K RMSNorm and draft RoPE only to K. This checkpoint has no V scale; honor one only if declared. Release features after bounded KV construction. For complete prompts, materialize only the ending window when lifecycle/projection semantics permit. Target prefill still covers the full prompt; draft conditioning needs no full-prefix attention pass.

A full FP16 [262144,12800] feature array costs 6.25 GiB/replica. Use bounded target-prefill chunks or proven necessary tail captures, accounting for overlap with target execution.

Prefix reuse, resumption and interrupted/streamed prefill require the correct ending conditioning window. Target KV generally cannot reconstruct missing features; a fresh target pass over only the last 2,048 tokens is not equivalent. Reuse compatible draft state or replay from a valid target-state boundary.

During skipped proposals, maintain the enabled method's window through committed target progress, track draft-valid position, and include maintenance time. Rejected features/tentative KV never become permanent conditioning.

### Block and verification policy

Start with block four: one anchor plus three proposals and four target verification rows per sequence, as in the exact-family published setup. Its QSA block limit is implementation-specific, not architectural.

| Scheduled sequences at block four | Target verification rows |
|---:|---:|
| 8 | 32 |
| 16 | 64 |
| 24 | 96 |
| 32 | 128 |

To shorten proposals initially, choose a count of one, two, or three before drawing; retain four-row draft computation and verify that prefix. This preserves one draft graph and reduces target work without saving skipped draft-row computation. Zero proposals remains a warm scheduling choice.

Genuine block-two/block-three/block-eight execution remains optional follow-up work after the fixed-block-four/truncated-prefix control. Before promotion, qualify its changed non-causal interactions, grouped-convolution boundaries, masks, proposal probabilities, scratch and graphs. Configured block eight is not qualification for target verification. Near the position limit, skip fixed-block drafting that would evaluate unsupported positions unless a smaller valid block is qualified.

Longer blocks require per-sequence QSA/recurrence support and aggregate row capacity; a larger R32 envelope alone is insufficient.

### Selector, masks, and sampling

Implement the selector and block-local two-tap dynamic convolutions, respecting block membership and padding. These convolutions create no persistent draft recurrent tail.

The selector defines conditional candidate paths; rejection sampling must use their actual conditional proposal probabilities, not pre-selector logits. Preserve algorithm-defined RNG advancement. Source: [author's DFlash 2 method description](https://inco.ai/blog/dflash2/).

Implement and qualify selector-aware stochastic verification on SM70 for every exposed sampling policy. The pinned worker's `_validate_phase1_sampling_support` can fall back to greedy argmax when non-greedy verification is unavailable, explicitly changing the requested distribution. ProviderOne must reject unsupported settings before admission or state mutation; this reference fallback is not permitted. Source: [pinned DFlash worker](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/speculative/dflash_worker_v2.py).

Use the checkpoint's non-causal draft mask while preserving per-row target causality across KV, QSA pooled/tail selection, recurrence, and PLE. Future candidates must not influence earlier target logits.

Keep vocabulary projection sharded where possible and merge local top-16 candidate sets without gathering all draft logits per rank. Bind codebook rows to actual target token IDs and mask token.

Budget target verification/correction sampling separately: one FP32 [128,248320] matrix costs 121.25 MiB, with potentially overlapping transformed logits, probabilities, masks, and correction intermediates. Qualify exact temperature, top-k/top-p, penalties, constraints, and correction sampling; greedy top-k establishes neither general sampling correctness nor peak memory. Source: [pinned speculative worker](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/speculative/dflash_worker_v2.py).

### Performance and capacity qualification

Start with FP16 matrix operands and appropriate FP32 accumulation/reductions; qualify weight conversions and all activation/output/KV cast boundaries under the finite-range gate. GB300, BF16, TRT-LLM, FlashInfer, and Triton reference paths do not establish SM70 support. Qualify kernels, HC capture, selector, masks, window wraparound, every rejection position, skipped-proposal maintenance, and high absolute positions.

Compute capacity from twelve target attention/indexer caches plus actual DFlash weights, bounded draft KV, features, target-state recovery, execution resources, graphs, and overlaps. The MTP thirteenth-cache formula does not apply. A DFlash-only profile may release MTP resources through the permitted in-process transition.

Compare committed throughput against no speculation and imported MTP with identical target numerics, reasoning/sampling modes, histories, and workloads. Select the best qualified execution within the enabled profile, including zero proposals when beneficial.

Inco is a SC117 candidate, not guaranteed useful speculation. Poor acceptance may justify exact-target fine-tuning using verified SC117 numerical-target HC features/continuations/distributions and its intended embedding/head, consistently training/validating backbone, block convolutions and selector. Training cost and V100 speedup remain unestablished.

## 15 SM70 kernel and model execution work

Retain each profile's qualified runtime weight inventory, including explicit NVFP4 decode/conversion choices. All routed experts in the selected SC117 retained plan remain resident. Do not introduce unbudgeted host expert paging or full-checkpoint FP16 expansion; each artifact must fit its complete serving plan.

Required model contract:

- Implement qwen4exp's four HC streams, target per-stream normalization, gated query projection, PLE injection and all selected mixed weight formats. Its 12,288-wide query/gate projection contains 24 query heads plus their gate. Preserve MTP's distinct flattened-HC norm axes (Section 12); reconcile HF `1 + weight` conventions with converted GGUF values exactly once.

After the declared head-order transforms, TP8 projections have three target query heads and two GDN Q/K heads with six V heads per rank; routed-expert intermediate shards are 80/160/320 values wide for ETP8/4/2 respectively. Attention owners may receive queries from multiple projection ranks. The indexer's four query heads plus one raw-key head form 640 global projection values: assemble complete heads before normalization, rotation, scoring and ReLU aggregation, rather than assuming a TP8 output slice contains whole indexer heads. Qualify packed-block slices, padding and K tails at these actual shapes; another engine's TP4 or K-multiple-of-64 route is not sufficient.

Profile indexer-query assembly, pooled scan/top-k, selected-group expansion and KV gather, selected-position routing, and KV attention separately. Reuse history tiles across query rows only when causal masks permit. Bound selected-index and unpacking work by tiles and row budgets; never expand selected KV for an entire long prefill batch. Increasing KV attention degree does not remove a canonical indexer bottleneck. Further indexer parallelism requires measured whole-step benefit and exact global selection.

Every kernel family requires native SM70 compilation and supported math/library paths. Respect Volta's independent thread scheduling and explicit synchronization requirements ([Volta tuning guide](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html)). Select paths using complete packing, communication, state, and recovery costs, not Tensor Core utilization or isolated kernel rate alone.

### Performance engineering contract

Apply these rules to target, MTP, DFlash and vision under Sections 4–7's capacity/quality commitments. Justify hot-path compute, traffic, launch and synchronization costs; exceptions require measured justification and bounded resources/lifetimes.

| Area | Required implementation rule |
|---|---|
| Device residency | Keep reusable KV/state, routing/indexer results and speculative features on GPU until consumers retire. Pack KV directly there and unpack only required tiles. CPU tokenization, bounded media preparation, the host/NVMe-backed PLE table, committed output and Section 17's controlled session checkpoints are intentional boundaries; additional copies require Section 1’s measured tradeoff policy, and session commitments remain protected. |
| Host feedback | Audit explicit/implicit `.item()`, `.cpu()`, `.tolist()`, NumPy conversion, host branches on device values and CPU-created indexing. Avoid per-layer/row readbacks. Prefer device-resident counters/indices and fixed-capacity outputs; dynamic `nonzero`/masked selection or tensor slice bounds can introduce synchronization. Preserve the exact operator, ordering and masks when changing representation. |
| Transfers | Coalesce necessary control/input/output transfers; normally return committed token IDs/compact status rather than full logits/features. Budget explicit log-probability requests separately. Use bounded preallocated pinned staging and useful asynchronous copies; protect buffers through DMA/consumer retirement and await D2H completion before CPU reads. No on-demand pin/register/unregister cycle per token. |
| Sampling/output | Prefer qualified GPU sampling, penalties/constraints and verification to avoid host round-trips of vocabulary logits. Preserve selector probabilities, grammar/stop semantics and RNG. CPU text assembly/decisions consume bounded compact committed results/status; reject unsupported sampling without changing policy. |
| Inference runtime | Where PyTorch is used, set evaluation mode and disable autograd in every inference worker/thread; qualify thread-local `inference_mode` compatibility. Keep host orchestration, serialization, logging and locks off the per-layer critical path; replace or move measured hotspots into suitable native code. Bound queues and diagnostics; retain launch/error and asynchronous completion checks. |
| Allocation/initialization | Reuse Section 4's arenas, handles and descriptors. No per-token general-purpose device allocation/free, `empty_cache`, full-buffer clearing, weight repacking or compiler/algorithm search. Initialize only required live ranges before first read; never rely on uninitialized or stale masked state. Cache immutable preparation by complete source/numerical/layout/build identity. |
| Memory traffic | Produce the next consumer's layout directly; remove redundant copies, dtype conversions and transposes. Coalesce aligned loads/stores within bounds, respecting packed metadata/strides. Reuse bounded weight/history tiles and activation buffers with proven lifetimes; measure useful versus transferred bytes, cache misses and shared-memory bank conflicts. No full-history decoded shadows. |
| Work/fusion | Compute only consumed outputs and required conditioning/recovery. Fuse compatible normalization/gating/packing/elementwise work when saved launches/traffic exceed register/spill/occupancy costs. Preserve cast points and state transactions. Fusion, more streams, occupancy and Tensor Core utilization are not objectives by themselves. |
| Shape/routing | Select GEMV/vector/small-M or tiled/Tensor Core paths by complete-step measurements at actual row counts. Group MoE work by actual rows per expert, preserving routing scores/ties; device-driven compaction/grouped execution must not require a host count readback per expert/layer. Charge padding and avoid kernels over empty work when measured worthwhile. |
| Overlap/progress | Overlap independent preparation/copy/compute/communication only with explicit stream/event and allocator-lifetime dependencies. Check default-stream serialization and actual copy-engine capabilities. More streams can contend for HBM, SMs, PCIe or NVLink; qualify under mixed/full load and maintain NCCL progress/order. Never require concurrent scheduling for correctness. |
| Preparation/warmup | Warm released shape/kernel/communicator paths and required module/library residency before timing/capture and admission. Persist qualified conversion, compile and algorithm artifacts; serving-time activity selects ready plans. `torch.compile` is optional and must prove native SM70 coverage, bounded recompilation/caches and compatibility with custom operators; default backend selection does not guarantee support. |

Sources: [CUDA transfer/memory practices](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html), [sync-free PyTorch operations](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/sync-free-code.html), [inference-mode semantics](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html), [stream/allocator lifetimes](https://docs.pytorch.org/docs/2.14/notes/cuda.html), and [CUDA lazy loading](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html).

### Hardware capability and numerical checks

At initialization, record GPU attributes and CPU instruction capabilities; map each operator to its compiled implementation, dtype/accumulator, supported shapes and resource bounds. Inspect native SM70 objects and representative disassembly/profiler traces; an API call or `sm_70` build alone proves neither Tensor Core execution nor optimal code.

| V100 mechanism | Use and boundary |
|---|---|
| FP16 Tensor Cores / SIMT | Qualify FP16 operands with FP32 accumulation for suitable dense/grouped tiles; use SIMT/vector paths where small/ragged work wins. Retain FP32 recurrent state and sensitive reductions. |
| Integer/warp operations | Exploit bit unpacking, integer address work and masked warp vote/shuffle/scan for packing/routing. SM70 supports packed-byte `dp4a`, but it is not INT8 Tensor Core math; quantizing floating activations to use it changes the numerical profile and requires separate qualification. |
| Register/shared-memory reuse | Tune tiles, registers, spills, coalescing, instruction-level parallelism and shared-memory/L1 carveout together. Dynamic shared memory above 48 KiB requires opt-in and bound checks; do not maximize occupancy blindly. Observe Volta's explicit warp/barrier synchronization rules. |
| Unsupported newer mechanisms | Do not assume native BF16/TF32/FP8/NVFP4 or integer Tensor Core modes, structured sparsity, `cp.async`, TMA, `ldmatrix`, warp-group MMA or newer cache controls on SM70. A modern API may compile to another path; validate what actually executes. |
| CPU/media acceleration | Qualify detected SIMD and codec/backend paths for PLE decoding, tokenization and media work, including numerical preprocessing parity, CPU frequency effects and NUMA traffic. Do not assume a Xeon generation/ISA or hardware video decoder from the chassis name. |

Include compiler/library math modes in the numerical profile. Verify cuBLAS/PyTorch settings and custom casts: declared FP32 accumulation must exclude reduced-precision intermediate reductions. Fast-math, approximate transcendentals, flush-to-zero and changed reduction/fusion order require affected parity, sampling, task-quality and speculative-acceptance gates; never enable them globally merely for speed. Weight/KV quantization remains administrator-selected.

Sources: [Volta tuning](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html), [PTX instruction target requirements](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html), and [cuBLAS numerical modes](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html).

### Library and kernel responsibilities

| Work | Implementation requirement |
|---|---|
| Dense target, MTP, DFlash and vision projections | Compare rank-local cuBLASLt/cuBLAS FP16-input, FP32-accumulation GEMM against qualified custom kernels at actual row/patch counts |
| Packed routed experts | Use SM70 packed GEMV/small-M and grouped/tiled kernels; CUTLASS C++ components support fused tile decode and Tensor Core work |
| Gated DeltaNet and its causal convolution | Qualify native SM70 chunkwise prefill and fused recurrent decode/verification kernels in the same dispatcher. Preserve width-four causal history, 16 Q/K-to-48 V-head grouping, exact gates/normalization and FP32 recurrent state. Select chunk sizes by whole-step measurements with bounded scratch. |
| Routing and reductions | Use qualified CUB scan, sort, compaction, and warp/block primitives where useful; preserve exact selection/tie rules and caller-owned temporary storage |
| RotorQuant and indexed attention | Implement packed history, selection, masking, and distributed merge directly; dense BLAS or generic sparse-matrix libraries do not supply these semantics |
| Vision encoder/projector | Implement patch projection, positional interpolation/rotary, LayerNorm, GELU/MLP, noncausal segmented attention and merger. Use bounded exact attention tiles and qualified FP16 kernels; generic FlashAttention support is not an SM70 qualification. |

Bind the pinned HF Gated DeltaNet equations and cast points: SiLU causal convolution; Q/K L2 normalization with epsilon 1e-6 and query scaling by `1/sqrt(128)`; `beta=sigmoid(b)`; FP32 log decay `g=-exp(A_log)*softplus(a+dt_bias)`; and RMS normalization before sigmoid output gating. Three V heads share each Q/K head before TP partitioning. The gated output RMS norm uses its direct learned weight, with no Gemma offset. Qualify fused or FP16 Tensor Core reformulations against this numerical contract, including FP32 final state. The reference's chunk size 64 is a comparison point, not a fixed V100 setting. Source: [pinned recurrent and convolution implementation](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py).

Bind the recurrent representation during import. The pinned qwen4exp GGUF converter stores `A_neg = −exp(A_log)`; for that encoding use `g = A_neg × softplus(a + dt_bias)` without exponentiating the stored value again. It also changes V-head order from `[16 K groups, 3 V heads]` to `[3 V subheads, 16 K groups]`. Resolve the coupled V/z/a/b projections, decay/bias vectors, V-convolution channels and output-projection transforms, including declared folded transforms, before choosing Q/K broadcasting, TP head mapping and state layout. Any remap must preserve decoded quantized values. Verify each artifact's actual recipe rather than infer it from matching shapes. Sources: [pinned conversion](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/conversion/qwen.py), [qwen4exp binding](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/conversion/qwen4exp.py), and [runtime decay use](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/src/models/qwen4exp.cpp).

PLE convolves normalized gated-value activations across all four 2,560-wide HC streams. Preserve width 4, dilation 3 (taps at `t`, `t−3`, `t−6`, `t−9`), SiLU after convolution, and addition to the gated value. Its logical prior-activation tail contains `9 × 10,240` values per sequence before TP placement. Pin the qualified activation/storage dtype; charge actual shards/replicas, padding and tentative/recovery versions separately from n-gram token/hash state. This tail differs from the recurrent layers' undilated convolution state. Source: [pinned PLE implementation](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py).

cuBLASLt provides reusable algorithms and workspace constraints; CUTLASS supports SM70 FP16 TensorOp/WMMA, and its grouped GEMM can be specialized for Volta. Sources: [cuBLAS API](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html), [CUTLASS functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html), and [grouped-GEMM specialization](https://github.com/NVIDIA/cutlass/issues/888). Inventory existing calls and traces before replacing work already using an appropriate library; these options still require measured ProviderOne benefit.

Keep handles/descriptors and bounded per-stream scratch persistent; tune outside serving. Key choices by shape, layout/strides, alignment, numerics, workspace, architecture and exact library build. Concurrent scratch must be separate or provably serialized. Set cuBLAS stream before rebinding workspace; stream changes reset it. Budget lazy residency/scratch before admission; larger workspaces require the resource-envelope protocol.

Compare fused packed-weight tile decode with bounded FP16 staging/library GEMM where row reuse amortizes staging. Count reads/writes/launches and peak memory; no full-model/KV expansion for BLAS. Specify FP32 accumulation/reduction to prevent silent numerical changes. Select expert kernels from actual routed rows without per-layer host readback merely to populate library calls.

Use cuDNN or ordinary cuSPARSE only for a matching operator with measured complete-step benefit. cuBLASXt's competing multi-GPU distribution policy is outside fixed TP8. cuSPARSELt requires SM80 or newer structured-sparsity hardware; cuBLASDx 0.6.0 removes Volta. Neither is a baseline dependency. Sources: [cuSPARSELt support](https://docs.nvidia.com/cuda/cusparselt/index.html) and [cuBLASDx release notes](https://docs.nvidia.com/cuda/cublasdx/0.6.0/release_notes.html).

## 16 Prefill and mixed-work scheduling

Use chunked prefill, bounded workspaces, and continuous admission into scheduled work. Preserve causal order and all 36 recurrent layers' transitions within each sequence. Independent sequences may share an execution batch; parallel recurrent scans within one sequence require a semantics-preserving algorithm, not ordinary token batching.

Preserve the API's input-commit and prefill-cancellation contract. Track the admitted input range, its committed/pending boundary, target-consumed position, and draft-valid position through every chunk. Request-scoped cancellation discards the slot after work retires. Retained-session cancellation must retain the specified input progress or restore its checkpoint consistently across transcript, KV/indexer, recurrent/PLE, draft, sampling/output bookkeeping and multimodal state. Whole-append rollback requires protected, budgeted pre-append recovery state, including overwritten draft-window entries; a KV-length reset is insufficient.

Resumption validates the canonical expanded token prefix, immutable media identity, processor and position state, then continues from the recorded boundary; submitted input is not automatically fully cached. Test cancellation/resumption at chunk, pooling, page and media boundaries. Bind the existing behavior during fork integration without implicitly changing retention or input-commit policy.

Size prefill workspace for scheduled rows and kernel shapes, not configured sessions times every session's maximum chunk. Include routing tiles, attention/indexer scratch, recurrent intermediates, codec staging, speculative target-feature extraction, required logits and permitted visual overlap. Candidate measurements include 512-row chunks and total scheduled budgets of 1,024–8,192 rows; these are not settled limits or existing flags.

| Regime | Scheduling aim | Deciding measurements |
|---|---|---|
| Prefill only | Maximize completed expanded prompt rows/s within memory and fairness limits | Chunk size, scheduled rows, expert group sizes, history length, time to first token; separate text/visual work |
| Decode only | Maximize committed generated tokens/s with acceptable inter-token delay | Active requests, proposal acceptance, verification rows, indexed attention work, state/recovery cost |
| Mixed | Preserve decode progress while efficiently processing queued prompts | Prefill quota, decode service interval, prompt starvation, tail latency, combined resource pressure |
| Media preparation/encoding | Feed ready visual rows within the declared media envelope | Decode/resize/encode/transfer time, pixels/frames, encoder batches, staging occupancy and interference |

Do not delay low-load requests to fill a higher-throughput batch. At high load, bound prefill service and rotate decode fairly: long prompts cannot indefinitely stall generation, nor generation starve new prompts. Administrator latency objectives restrict eligible throughput profiles.

For long contexts, prohibit full-context FP16 KV mirrors, full-history dequantization buffers, and permanent target-hidden-state archives, including huge temporary equivalents. Compute only requested prompt log-probabilities and required generation/verification logits.

Include tokenization, CPU media decode/preprocessing, request assembly, output delivery, pinned transfers, and NUMA placement in end-to-end profiles. Overlap independent CPU/GPU work where dependencies permit, and charge concurrent buffer lifetimes; storage remains occupied while another stream consumes it.

### Multimodal execution

Use one admission/scheduling/state path for text/media. Before publishing capacity, declare media formats, upload/decoded-size bounds, image counts/pixels, video frames/duration/sampling/timestamps, expanded-row allowance, encoder placement/numerics and CPU/GPU staging/replay/work budgets. Reject violations specifically. Media fidelity/resolution/frame policy are request/profile semantics, never activity controls.

#### Preparation, positions and context

Apply Section 2's pinned processor contract and preserve media order when interleaved with text. Record selected frames/timestamps and validate grids, placeholder counts and projected rows before target execution.

The expanded sequence is the context/cache unit. An image grid contributes `grid_t × grid_h × grid_w / merge_size²` visual placeholder rows; video also contributes timestamp and delimiter text. Fully rendered input plus generated reasoning/output must fit the equal selected window. Validate target and draft position ranges separately: cache-row ordinals and target M-RoPE coordinates are distinct. Count visual rows once in target prefill/history, with separate media-work metrics.

Preserve original expanded token IDs for PLE/hash state while replacing visual input embeddings with projected media embeddings. Repeated placeholder IDs do not identify media. Keep modality spans, grids, absolute offsets, M-RoPE coordinates/continuation delta, processor identity and immutable content identity through prefill, cancellation, resumption, prefix reuse and migration. Sources: pinned [processor expansion](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen3_vl/processing_qwen3_vl.py) and [target visual/PLE/position implementation](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py).

#### Bounded visual execution

Initially qualify one resident FP16 encoder/projector copy on one fixed GPU, choosing the rank by topology, worst-rank headroom and mixed-work latency. Qualified sharding of that same encoder may replace this placement after implementation and measured benefit; additional encoder replicas are separate experiments under Section 1’s duplication policy. Publish its complete placement/resource plan before admission. Placement stays fixed during serving. Route projected embeddings through the target TP8 input contract; eight target ranks do not require eight visual copies.

Use persistent bounded CPU preprocessing pools with fixed setup-time affinity/thread budgets. At coordinated boundaries, issue ready tasks and select legal image/frame microbatches, prepared attention tiles and language-prefill quotas inside reserved limits. Add feedback-based optional-task/byte quotas only where measured CPU/PCIe contention improves complete mixed service. Independent images or valid temporal/attention segments may batch together; arbitrary spatial splitting must not remove noncausal attention edges. Use exact bounded attention tiling, preserving segment boundaries, positional interpolation and merger order, rather than an unbounded score tensor or an approximation.

Charge placed vision weights, encoder activations, projected embeddings, staging, graph pools and communication for permitted overlap. Encoding concurrency may be lower than resident session capacity. Queue visual work fairly while preserving every full-context commitment; the largest supported media work unit must fit at full residency. Idle visual execution does not release promised reserve for additional guaranteed contexts.

Use bounded host/pinned buffers and immutable bounded replay sources or reproducible equivalents. Release embeddings after target/speculative conditioning unless explicitly retained/budgeted; a `[262144,2560]` FP16 archive costs 1.25 GiB/copy and is not default. Bound media caches by content, processor/settings, vision weights/numerics, grids/order; capacity must hold on misses. Target-prefix reuse also requires Section 9's complete state/profile identity.

#### Multimodal speculation

Apply Section 12's actual next-position embedding contract for MTP, including visual lookahead and tentative chunk-tail state. Qualify splits inside visual spans, timestamp/delimiter transitions, multiple media items, cancellation/resumption and zero-proposal maintenance; placeholder-token lookup is insufficient.

DFlash consumes the target's visually conditioned completed-layer HC means. Its draft RoPE uses scalar absolute sequence-row positions, separately from target M-RoPE coordinates/deltas. Visual rows count toward the bounded conditioning window. Keep draft materialization, correction, rollback and warm maintenance aligned while the verifier retains target positions. This feature contract needs no separate draft vision encoder. Source: pinned [DFlash model](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/models/dflash.py) and [worker position materialization](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/speculative/dflash_worker_v2.py). Text-only qualification cannot establish multimodal correctness, capacity or acceptance.

## 17 Work scheduling and prepared execution selection

Use a small deterministic scheduler with offline-qualified execution plans and cheap known metadata. Prepare one selected numerical/structural/media profile, fixed unique history ownership and a maximum execution/resource envelope before publishing capacity. Ordinary activity changes select ready work and prepared plans inside those bounds. No online learning, serving-time algorithm search, load-triggered graph rebuild, maximum-arena resize, communicator recreation or physical-weight/history reshuffling is required.

### Decision boundaries

| Boundary | Decisions | Required contract |
|---|---|---|
| Every scheduled step | Ready admitted sequences, legal microbatch/verification rows, proposal length, prefill/media work quotas, prepared kernel family and existing graph bucket | One globally agreed plan inside prepared capacities; preserve causal/state/collective rules |
| Completion, cancellation or idle-slot demand | Event-driven FIFO handoff, bounded reusable-cache reclaim and complete idle-state suspension/restore | Preserve Section 17’s lifetime/backing and all-rank publication rules; do not wait for a controller polling tick |
| Preparation or explicit supported administration | Maximum envelope/capacity, graph-set replacement, fixed placement and numerical/structural profile activation | Section 19’s qualified readiness/quiescence/drain protocol; no ordinary active-count trigger |
| Optional future live KV placement | Only a separately adopted Section 11 experiment | Complete intermediate-state, memory, payback and burst/oscillation proof; not a release prerequisite |

One coordinator issues a plan and epoch for participating ranks. They must not independently change proposal lengths, graph shapes, communicator membership or collective participation. A sequence retains at most one unresolved speculative verification round under the initial PP2 plan; independent ready groups and known causal prompt chunks can overlap.

Choose batch shape, legal proposal width, prefill/media quota and kernel/graph together from a small measured map. Basic inputs are readiness, actual scheduled rows, valid histories, remaining limits and already-known media geometry. Device routing handles actual expert-row populations without a host readback per expert/layer. Prior completed-round acceptance or sampled stage/queue timings may refine a choice only after demonstrating decision value. Diagnostics need not become synchronous control inputs.

Optimize committed output per complete elapsed round, including draft work, warm conditioning maintenance, target verification, refresh/recovery, communication, sampling/commit and scheduling. Prefer a small uniform-width table first. Neither acceptance percentage nor active count alone establishes the best width; do not impose a monotone count-to-width ladder. DFlash’s initial positive prefix lengths still execute its four-row draft. A04’s online cost fit, E08’s heterogeneous widths and E09’s route-aware grouping remain conditional experiments that must beat the simple controls without extra batch delay or starvation.

Start mixed scheduling with measured pure-prefill and bounded mixed-work policies, preserving decode service and a prefill progress/aging floor. Separate useful retention boundaries from physical pages and execution chunks. PP2 first uses bounded dependency-ready queues; predictive/delay scheduling needs actual second-server bubble evidence. CPU/media pools remain initialized and bound while queued work varies.

Prune dominated plans offline: one winning tactic can cover several counts. Retain another tactic, graph bucket or observed feature only when repeatable complete-serving benefit justifies its decision, padding and residency costs under the existing gates. Small reproducible gains count; do not add a universal percentage threshold.

### Required concurrency coverage

| Actual active sequences, one per initial request | Required behavior |
|---|---|
| 0 | Retain promised state and a ready prepared profile; perform permitted lifecycle/preparation work |
| 1, 2, 4, 8 | Measure low-load latency/throughput, fixed-placement work and useful speculative widths |
| 12, 16, 20, 24, 28, 32 | Measure saturation, verification splitting, prefill interference, fairness and complete speculation profitability wherever capacity permits |
| Other counts | Use a correctly masked qualified covering plan; measure every integer count and investigate breakpoints |
| Above 32 | Extend the same policy when shorter contexts or other qualified settings permit |

For every released context/precision/method, measure every integer active count through capacity. Detailed long-context/lifecycle tests cover named thresholds, endpoints and observed breakpoints; test 18 when capacity is 18 and extend above 32 when feasible. This validates the admitted range; it does not require a distinct graph/tactic at every count. Masked covering graphs prove safety, not intermediate throughput. Every optimization/milestone also runs Section 5’s fixed C1/2/4/8/16/24 A/B matrix and decode nonregression gate; insufficient capacity is an explicit gap.

Handle an immediate jump from one active sequence to the maximum using the valid fixed ownership and prepared execution path, with bounded smaller microbatches when needed. No history rewrite or recapture is required before progress. Keep statistics asynchronous where possible; compare cheap dispatch against the best qualified fixed execution under identical retained histories. If Section 11’s experiment is later adopted, include its complete transition/interference costs and apply its dwell/hysteresis rules separately.

### FIFO admission and tiered retained sessions

**Default policy.** Use one bounded FIFO admission queue with continuous batching. Admit the oldest eligible request when a qualified GPU slot/execution envelope is available. Completion/idling hands the slot onward; runnable generation is not routinely preempted for tier rotation. FIFO orders admission, not completion; up to the qualified GPU-resident maximum can execute together. Fresh queued work has no computed KV to offload: retain only bounded input/media sources and metadata.

FIFO avoids repeated full-history transfers but guarantees neither equal wall-clock active time nor bounded waiting behind arbitrarily long work. Preserve administrator context/reply/work limits and cancellation policies; never truncate accepted generation to advertise fairness. GPU-only operation supports bounded new-request queuing; qualified tiering additionally lets idle retained conversations release GPU slots while preserving state in RAM or administrator-selected local storage.

**Existing mechanisms and evidence.** The inspected public [TensorFold README at `cb2ebf0540f42604e2759b2ddef497861e928248`](https://github.com/kklouzal/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/README.md) documents shared rounds/queuing and MLX prefix spill/snapshots; it does not establish this CUDA/V100 full-session capability in the deployed fork. Inspect the actual admission, cancellation and checkpoint paths, then replace or refactor them directly to satisfy ProviderOne’s contract; existing lane abstractions need not survive. [vLLM V1](https://docs.vllm.ai/en/latest/usage/v1_guide/) uses preemption without requiring its former KV swap mechanism; its [optimization guide](https://docs.vllm.ai/en/latest/configuration/optimization/) describes recomputation as the default. [SGLang HiCache](https://docs.sglang.ai/advanced_features/hicache_design.html) supplies tiering concepts, and its [hybrid-state work](https://www.sglang.io/blog/hicache) emphasizes recurrent/draft recovery. These are references, not drop-in SM70/qwen4exp support. [Fair-serving research](https://www.usenix.org/conference/osdi24/presentation/sheng) distinguishes cost-aware execution fairness from simple request ordering; its bounds do not automatically apply here.

Current integration uses Section 2’s pinned `python-0.6` source: bounded FIFO/live credits, shared Responses/Conversations records and common protected-retention/prefix catalog with RAM backing already exist. Preserve/refactor those owners and qualify their actual CUDA/model/resource behavior. GPU/RAM are the implemented typed retention tiers; optional storage needs its own implementation and gates. Generic best-effort prefix spill, logical API persistence and a protected computed continuation retain different promises. A legacy stream-growth horizon is not a full-window admission proof. Earlier [CUDA scheduler](https://github.com/kklouzal/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/cuda/scheduler.py), [CheckpointStore](https://github.com/kklouzal/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/server/checkpoints.py) and [MLX snapshot serializer](https://github.com/kklouzal/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/engine/prefix_snapshots.py) findings remain historical source evidence; they neither describe every current owner nor qualify the installed ProviderOne service.

#### OpenAI-compatible API and internal session identity

Use standard OpenAI-compatible endpoints, schemas and continuation fields; no ProviderOne session/revision headers, private client session protocol or fixed GPU-lane identifier is required. Clients use the local base URL, normal authentication and model IDs from `GET /v1/models`. Pin the supported API/SDK schema and publish/test its endpoint/feature matrix, including each media content type's exact decoder/processor mapping. Required video understanding stays a release gate: distinguish native video/file transport from a standard image-frame/timestamp workflow, and separately qualify the latter if used. Neither invent a private video field nor silently replace Section 2's pinned video semantics to claim API compatibility. Compatibility means correct wire and lifecycle behavior for supported operations, not a claim that this model/server implements every OpenAI modality, hosted tool or product feature.

| Client interface | Conversation and cache behavior |
|---|---|
| `POST /v1/chat/completions` | Client sends `messages`; automatically match compatible rendered prefixes. A response `id` identifies that completion, not a stable chat or hidden append target. |
| `POST /v1/completions` | Accept the standard full `prompt` contract with automatic prefix reuse; do not infer conversation identity from equal prompts. |
| `POST /v1/responses` | Implement/qualify the standard `previous_response_id` chain; a new response returns a new `id`. Resolve retained logical items and render the actual next input before selecting model state. |
| Standard Conversations API | Reuse/refactor the current shared `/v1/conversations` logical store and item dispatcher; qualify its installed wire/storage/lifecycle behavior, then pass its `id` through Responses' `conversation` field. Do not substitute a proprietary field. |
| Internal identities | Owner-scoped logical sequence IDs, mutation/checkpoint generations and GPU slots protect state and placement; prefix/page keys identify reusable computation. These remain distinct. |

OpenAI's [conversation-state guide](https://developers.openai.com/api/docs/guides/conversation-state) defines manual Chat history, response chains and durable Conversations IDs. Reject simultaneous `conversation`/`previous_response_id`. Responses `instructions` is request-scoped and never inherited through `previous_response_id` ([instruction scope](https://developers.openai.com/api/docs/guides/text)); distinguish it from historical input items. Rerender changed/omitted instructions before prefix selection; obsolete instruction state cannot be reused. Repeated `user`, `metadata` or `prompt_cache_key` never authorizes hidden Chat history. A cache hint aids lookup only within its allowed domain; validate content/profile and authorization.

The current `python-0.6` implementation already shares Responses/Conversations logical storage and item dispatch across endpoint adapters (Section 2). Preserve/refactor that implementation and independently qualify the installed ProviderOne standard SDK/wire, durable storage, ownership, create/retrieve/update/delete/items and bounded-capacity behavior before advertising stable client conversation IDs. The earlier [public TensorFold API contract](https://github.com/kklouzal/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/docs/api.md), which rejected `conversation`, remains historical; its rejected operations and route/storage structure impose no current constraint. Storing a response or conversation retains logical API items, not an automatic permanent GPU slot or full KV checkpoint. Budget those records and parent/media dependencies separately; an explicit computed-state retention promise additionally needs Section 17's backing claim. A valid item history can rebuild a reusable-prefix miss, while required queued/retained inference checkpoints remain protected. Logical API identity can survive computed-cache eviction; each rebuilt execution still validates the requested profile and actual rendered prefix.

Serialize mutations of a shared conversation under an internal owner/generation transaction, with bounded FIFO waiting or a documented standard-shaped conflict; no external revision handshake. Response-chain branches require independent sequence/state claims and must not mutate their parent. Distinguish a new turn from resumption of the same operation: initialize new-request sampling/constraints/output state for the former and preserve it for the latter. Validate principal access on every ID lookup. Unknown/deleted/expired references return explicit API errors, never a new unrelated conversation. Honor `store` and declared retention/deletion semantics; reserve response/item/manifest/media bytes and dependency lifetimes before promising retention. Apply the documented deletion semantics to descendants and independently retained items: preserve needed dependencies only where allowed; otherwise mark affected references unavailable explicitly. Never leave a dangling continuation or override standard deletion behavior.

Preserve standard response IDs, error envelopes/statuses, tool-call IDs, streaming events/deltas/termination and usage/cached-token fields. Queue waiting, restore and cache management must work with unmodified SDK calls; expose detailed queue/tier state through operations metrics rather than invented generation events or fake tokens. Request IDs are diagnostics, not conversation handles or automatic deduplication. Follow supported endpoint retry/cancellation/retrieval semantics; never infer retries from equal input or promise unlimited/exactly-once network delivery. Test disconnect/retry races without duplicate internal commit/publication, and report the standard API's limits where the caller cannot reconcile an unknown result.

#### Queue, residency and execution

Keep logical session ID/generation independent of reusable GPU slot ID/generation. A retained session can have no pending request; an unresolved request can be waiting for restoration. Allow at most one mutating request per logical sequence under Section 6.

| State | Meaning |
|---|---|
| Waiting | Accepted request awaiting GPU admission; either fresh input or a retained-state resume |
| GPU-ready / running | Required state is on GPU; execution requires an issued plan |
| Idle retained | No GPU-admitted runnable request; model state remains retained and may stay on GPU while capacity is spare |
| Suspending | Quiescent state is being saved; incomplete destination cannot replace the authoritative GPU source |
| Suspended | Complete required retained state is in RAM/storage, ready for a future queued resume |
| Restoring | Destination resources are reserved; incomplete GPU state cannot execute |
| Blocked / terminal | Input/output/I/O blocking or explicit failure, completion or cancellation; expose the reason and retire resources under lifetime rules |

Record the authoritative checkpoint generation/tier, complete effective-profile/build/schema identity, component sizes/offsets/checksums and retained validity/recovery frontiers. Never use incomplete transfer state as a resume point.

Assign FIFO tickets on validated enqueue; resumes join the same queue as new requests. Media readiness determines eligibility. A newer warm GPU-resident resume cannot bypass older eligible work merely to exploit locality. Work awaiting client input/media readiness or an explicitly unavailable tier is ineligible, retains its recorded ordering information and cannot block unrelated ready work indefinitely. Never bypass an eligible head merely for throughput; bound readiness/I/O waits and expose timeout/failure rather than silently excluding work. Validate infeasible context/profile requests before acceptance rather than leaving them permanently at the head.

Among GPU-admitted runnable sequences, rotate bounded prefill/decode opportunities fairly through ProviderOne’s scheduler, replacing inherited scheduling where needed, covering long prompts, heterogeneous acceptance and media bursts. Give every ready sequence progress; bound prefill chunks/visual units and consecutive decode rounds so one cannot monopolize execution. Equal turns do not imply equal committed tokens/s or latency. Measure service and waiting lag, and bound outstanding requests/bytes per principal so queue flooding cannot consume the entire service. Use no complex new scheduling framework. If future requirements demand preemption of long-running jobs for a tighter waiting bound, separately qualify that policy and its throughput/transfer tradeoff; do not silently introduce it into FIFO.

#### Capacity and commitments

Publish separate limits for GPU-resident sequences, retained logical sessions across all tiers, and accepted waiting requests/input bytes. Tiering increases retention capacity, not simultaneously executing GPU capacity or compute throughput. The Section 7 planner still acquires each GPU-admitted/restored sequence's full-window worst-rank commitment and required execution envelope. Queue length must not select an oversized GPU batch.

Fresh request-scoped waiting work reserves bounded input/media/queue resources, not hypothetical KV it has not computed. For every tiered retained session, additionally reserve a RAM or storage destination sufficient for its complete future context, continuation/recovery state and metadata, even while GPU-resident. No reliance on a short current history, unreserved future SSD spill or presumed shared prefix. Release its GPU entitlement only after complete checkpoint publication and reader retirement; preserve the logical context/lifetime and backing claim. GPU-only retained sessions keep their GPU guarantee. Before promising FIFO progress, prove that eligible work can obtain a recyclable slot within the declared waiting policy: tiered idle state can demote, active work can finish, or a known close/expiry can release capacity. If indefinite GPU-only reservations occupy every slot, do not admit more work with a progress guarantee; reject it or explicitly report blocked/bounded waiting under the API policy. Never force eviction/deletion or quietly overcommit.

Bound host RAM, storage, pinned staging, transfer descriptors, open files, queued I/O, media replay, old checkpoint generations/replacement and suspend/restore overlap. Subtract OS, PLE page cache, decoder workers and ordinary host obligations from RAM budgets; do not rely on OS swap. RAM need not all be pinned. Preserve the existing 4 GiB/2 GiB GPU protections. Reserve progress headroom: saving/releasing an idle victim must not require first restoring another session. Initial save-release-restore can be serialized when overlap cannot fit; no second full GPU history is required. Required tier-transfer overlap has its own bound. If Section 11’s optional ownership migration is adopted, its cap does not cover tier transfers; charge their combined permitted overlap. With migration disabled, retain every tier/recovery/progress obligation.

#### Complete state and transfer protocol

At request completion/cancellation, retire issued work and detach bounded host output/recovery obligations from the GPU slot. Release request-scoped GPU commitments promptly; no mandatory offload is needed if the state is disposable. If the session is explicitly retained, preserve its complete required checkpoint before freeing the last GPU copy. A queued same-session continuation may take its valid resident state directly when FIFO selects it; otherwise suspend idle retained state as needed to admit older work. Keep idle state warm only while capacity is spare; a reuse prediction or grace period cannot indefinitely delay an eligible queued request. Use write-back at suspension, reusing verified backing and writing only changed state under the integration contract below. All attention still executes with required materialized history on GPU; this is whole-session suspension, not CPU attention, expert paging, raw-indexer archiving or managed-memory oversubscription.

Use the profile's typed component inventory: each field is required retained state, qualified bounded rebuild input, or disposable execution scratch; never omit method state merely because an older serializer marked it nonstored. Save the complete Section 9/12/16 continuation/recovery contract: packed target and enabled-method KV, completed pooled keys, raw pending values/coordinates/group frontiers, recurrent/convolution/PLE state, draft-valid conditioning/window, deferred HC/next-embedding dependencies, target/draft/media positions, transcript/input frontiers, RNG streams/draw identifiers, constraints/penalties, detokenization/stops, reasoning/tool parser and channel state, response/item/call identities, queued/published event frontiers, undelivered committed output and media/replay references. Necessary host-side state may remain resident under explicit ownership. Include protected pre-append recovery versions when cancellation leaves an outstanding rollback obligation. Add no full feature/media-embedding archive unless continuation actually requires it. Shared target/draft/vision weights, PLE tables and reusable execution scratch are not per-session checkpoint payload.

1. The coordinator selects an idle victim/destination, reserves backing/staging and issues a transfer epoch. Serialize suspension with mutation, cancellation and any separately adopted ownership migration. Resolve/retire issued work at its method-defined consistent boundary and drain affected readers/writers/transfers through stream/events.
2. Preserve legal pending anchor/bonus and distinct committed/target-cached/draft-valid lengths; do not generate or consume RNG just to equalize them. Copy only valid packed rows and required state/tails in bounded coalesced chunks without requantization; exclude or initialize serialized padding and use qualified NUMA-local pinned staging and asynchronous host I/O. No GDS dependency.
3. Validate sizes, page ranges, checksums and a complete all-rank/component manifest. Publish one checkpoint generation only after every required piece succeeds; storage commit ordering must satisfy any explicitly promised durability. Until then GPU state remains authoritative. A failed save retains it and aborts eviction.
4. Retire readers/callbacks, detach or transfer reusable-page references, invalidate the GPU slot generation, then release its state/commitment and wake FIFO admission without waiting for a polling tick. Keep any leased shared pages alive under their remaining owners. The suspended checkpoint is required retained state, not an evictable best-effort prefix entry.
5. At FIFO admission of a resume, validate profile/schema and checked descriptor/size bounds before allocation. Reserve destination page classes/state, its complete future GPU window and bounded I/O/execution resources; verify checksums and restore every required materialized component before execution. Translate logical pages to the selected GPU slot/qualified owner mapping, validating worst-rank indexer/KV balance for arbitrary cohorts. Never deserialize device pointers or reinterpret ring slots as positions.
6. All ranks acknowledge readiness before GPU-ready publication. Partial/failed restoration releases only tentative destination resources after readers retire; the lower-tier checkpoint stays authoritative. Track GPU-authoritative mutable versions after resumption; verified immutable backing remains reusable, while stale mutable versions cannot satisfy a current checkpoint. Charge backing/old generations until safely retired. A new turn resets request-specific generation state; continuing an interrupted request preserves it.

Use byte-preserving continuation initially; KV-only restore, silent method changes or replay from an incomplete recurrent/media boundary are invalid. Recomputing required continuation needs independent complete-state, media, numerical and sampling qualification. A reusable-prefix miss uses Section 9's ordinary qualified prefill, not protected-checkpoint loss recovery. Shared prefixes require reference counts/lifetime guarantees; capacity assumes none.

#### Queue and cache integration

Use one canonical catalog entry per page/state object with content/profile identity, generation, validity, owner/references, leases, dirty status and per-tier residency/backing. Account separately for permitted physical/staging copies. Share one allocator/catalog and bounded transfer executor between prefix reuse and queued sessions; implement their ownership contracts directly, replacing inherited stores or abstractions where needed. Logical session/request manifests reference these objects and their small private mutable state. Distinguish protected active/queued continuation from reclaimable reusable-prefix residency: LRU eviction or failed best-effort spill must never delete a required queued object's last authoritative copy. On request completion, only explicitly retained sessions stay protected; released state may become bounded reusable cache without retaining an accidental session guarantee.

Reuse verified immutable backing on repeated suspension: transfer only newly appended/dirty pages, partial-page replacements and changed recurrent/draft/request state. A mutable GPU tail never overwrites a backing object referenced by an older checkpoint or fork. Publish a new manifest over immutable objects only after all referenced pieces validate; retire superseded objects after manifests, readers and transfers release their references. First writes and cache misses still require complete transfers; budget/measure that cold path and report actual dirty versus reused bytes rather than assuming delta savings. Metadata, reference maps, hashes, old checkpoint generations and copy-on-write peaks belong in the ledger.

Queue admission resolves the dependency path once: fresh work takes a compatible leased GPU/RAM/storage prefix hit or performs ordinary prefill; an exact queued continuation restores its authoritative manifest; an already GPU-resident valid continuation needs no transfer. A GPU-ready sequence omitted from a microbatch remains GPU-resident and does not become a paging victim. Stage bounded lookahead reads for the oldest eligible requests into host buffers when they can hide I/O; acquire the full GPU claim before restoring GPU pages. Prefetch must neither leapfrog FIFO nor evict active state, consume reserved progress buffers or expand unboundedly with queue depth.

Coalesce duplicate reads of the same immutable object and share completed buffers under reader leases; cancellation releases only that consumer's references. Do not hold the scheduler/catalog lock across disk I/O, DMA waits or serialization. Separate bounded I/O progress from launch/communication service, and rate-limit copies so PLE lookups, media processing and full-load decode retain their qualified service envelope. Do not build a second whole-session host image merely to send it to disk when bounded streaming is sufficient.

Apply one end-to-end transition contract to fresh prefill, prefix hits and authoritative-checkpoint resumption. Record reason/ticket and prevent new work before quiescing. Required retained state must be saved before slot release; all-rank restore readiness precedes execution. Hits change preparation cost, not FIFO priority, reservations, publication, cancellation or numerical/lifecycle gates; all paths use the same target/speculative execution.

#### Transfer cost, API and failure behavior

Measure simultaneous eight-rank D2H/H2D/storage paths under shared PCIe/NUMA and PLE/media contention; NVLink bandwidth is not host bandwidth. Section 8's native-context MTP growing history alone is 3.656250 GiB/session across the box for Q8 or 1.929688 GiB for iso4. Out-and-back therefore transfers at least 7.312500/3.859375 GiB before state, rounding and storage I/O. Use sustained measured rates and actual per-rank bytes, never a fixed millisecond restore promise; these costs motivate FIFO rather than routine rotation.

Administrators set bounded accepted/waiting counts/input bytes, per-principal limits, retention tier and RAM/storage budgets/path, and service/queue timeout limits. Resolve transfer chunks/overlap from qualified measurements. Begin with RAM suspension, then qualify optional storage. Quota reductions preserve accepted claims; disabling/changing a tier or numerical/model profile drains or safely relocates affected checkpoints, including suspended sessions. Queue saturation returns a specific response; deadlines/cancellation retire queued inputs without allocating GPU state. A hard wait SLO requires admission/workload bounds and measured completion/restore limits; FIFO alone provides none.

Expose queue order/age, lifecycle counts, backing bytes, GPU residency, restore delay, traffic and blocking reasons through existing request/cancellation and administrative status surfaces. Preserve standard API semantics while waiting; implement missing lifecycle operations directly. Resumption publishes canonical output without duplicates/retractions; define pause/heartbeat/idle timeouts without fake tokens. Bounded slow-client backpressure/cancellation must prevent indefinite GPU monopolization. Retire transfers/generations before cancellation/expiry reclamation.

On RAM/storage exhaustion, partial writes, corruption or I/O timeout, preserve an authoritative source, bound retries, stop affected admissions and expose failure without discarding context or blocking unrelated GPU work. A missing/corrupt sole checkpoint is an explicit session failure, never silent recomputation. RAM snapshots are not process-crash durable; local storage alone promises no restart recovery. Enforce ownership, file access and close/expiry cleanup. Normal queue/tier operations require no container/worker/CUDA-context restart.

## 18 CUDA graphs and communication

The pinned branch already has fixed native-owner workspace preparation and bounded partial attention graphs through `owner_execution.py`, `owner_graphs.py` and `paged_attention.py`. Query projection, indexer selection, other model-state updates and sampling remain outside those partial graphs. Reuse this scope and the content-bound execution-profile gate; no eligible measured V100 plan ships by default. The earlier private cu126 attention control is a separate context ≤ 8,192 / rows ≤ 24 envelope and is ineligible under the selected owned-Torch/CUDA12.9 identity. Do not relax its guard to claim native262K or new-stack qualification. Port useful math into the owned implementation and pass the actual row/context/codec, resource and complete-model gates. Prepared or compiled does not mean selected or qualified; whole-model capture is not required when a smaller captured region or uncaptured execution wins.

Initialize communication groups, kernel families, descriptor bounds, and maximum buffer capacities for the validated effective-profile envelope before use. A larger profile or ceiling requires new bound validation even if its bytes fit. Keep the selected structural weight/stage/owner plan fixed during ordinary adaptation (Section 20). Prepare a bounded graph set and retain uncaptured execution of the same plan, kernels and allocation contracts within the canonical serving path. Ordinary activity selects ready instances; graph-set construction/replacement belongs to qualified preparation or explicit supported administration, not a sustained-load trigger.

Prefer replay when saved submission time outweighs padding, graph memory and update costs; graphs do not accelerate kernel bodies. Use bounded instances, stable arenas and device-updated inputs/counters/masks. Remove CPU dependencies from capture or keep that logic outside; preserve Section 12's graph-safe RNG/state ownership. Warmup must not mutate live sessions or their RNG. Key validity by the complete effective profile and captured assumptions, not batch count alone; record breaks, recapture causes and replay coverage. Capture/recompile follows Section 19's coordinated resource protocol, never unbounded per-token cache-miss handling.

Use a small measured set of masked row/token and legal visual-shape/segment buckets with stable indirection; discard dominated buckets/tactics and let one winner cover multiple counts; do not capture every combination of session count, context, proposal width, prefill size, media shape, codec, and placement. Charge padding's actual compute and storage.

### Graph validity

Stable page-table pointers alone do not establish reuse across layout changes. Validate captured direct pointers, scalar arguments, launch dimensions, collective counts, participating ranks, stream dependencies, allocator pools, and object lifetimes. All captured assumptions must remain valid.

Mask inactive/padded rows from KV/indexer writes, recurrent/convolution state, draft state, RNG, stop processing, and committed output. Validate session generation to prevent stale writes into reused slots; asynchronous media completions obey the same generation and reader-retirement rules.

Charge graph-private pools, persistent inputs, and live instances in the ledger. Share scratch only when replay ordering proves nonoverlapping lifetimes. Source: [PyTorch CUDA graph memory/lifetime documentation](https://docs.pytorch.org/docs/2.14/notes/cuda.html); qualify the actual framework build under Section 2.

### Collective ordering

Use NCCL for normal collectives under the selected structural plan, with TP8 as the control. Initialize that profile’s stage-local TP/ETP/EP, pipeline, method and bounded attention groups before serving with topology-aware rank mapping; initialize only required group families, not every possible profile at once. Use AllReduce when every participant needs the result, ReduceScatter when the next operation consumes shards, and AllGather where replication is needed. Immediate ReduceScatter-plus-AllGather does not itself eliminate communication. Required state transfers and any separately adopted bounded ownership migration use matched NCCL P2P or qualified peer copies. Group mutually dependent two-sided sends/receives with `ncclGroupStart`/`ncclGroupEnd`, matching endpoint counts and datatypes within the globally agreed communication phase. Sources: [collective semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) and [P2P operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html).

Start with NCCL's operation/message/topology-aware selection; do not duplicate its policy in a serving controller. Prepare any measured custom collective rule table offline, keyed by actual operation, bytes, dtype, ranks/topology and build, then validate identity/readiness at startup. Full tuning reruns follow a changed qualification identity or concrete evidence, not every request or ordinary launch. Measure small decode-message latency, larger prefill traffic, and GPU contention during communication/computation overlap. Tune counts/bytes, scheduling, and graph integration before adding frameworks. Do not force one algorithm/protocol across message sizes without evidence, recreate communicators on active-count changes, or use changing process environment variables as runtime controls. Device-initiated NCCL, MSCCL++, or NVSHMEM needs a measured benefit justifying its additional contracts. NVLS requires newer NVSwitch/Hopper hardware and is unavailable on this V100 mesh ([NCCL controls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)).

All ranks agree on captured/uncaptured execution and compatible collective order. The issued plan fixes groups, ordering, counts, and empty contributions: cancellation, empty shards, or partial migration in an adopted experiment cannot independently suppress a worker's collective. Coordinate required state transfers, any adopted ownership migration, visual-input distribution and serving with explicit streams/events and a globally ordered plan; do not inject rank-local copies or visual broadcasts into captured collective schedules. Source: [NCCL CUDA graph requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html).

Preparation/administrator capture or replacement follows framework quiescence rules. Another host thread does not make unrelated concurrent CUDA work safe. Use uncaptured execution around recapture where needed and pause affected device work as required.

## 19 Configuration changes without restart

Preserve early distributed/control ownership and private synthetic warmup. Retire synthetic state/readers before publishing public FIFO admission; do not weaken real principal/ticket/credit checks for constructor warmup. Physical guard rows remain separate from admitted logical context and positions.

A clean preflight resource refusal can leave the valid state pending. After uncertain device/collective submission, transfer, publication or reader retirement, quarantine affected owners and retain their buffers/claims until retirement is proven. Never issue ordinary success, restore a freed source mapping or reuse a slot after ambiguous completion. Notify the existing cohort supervisor where peers could be stranded in a collective; preserve bounded request-local failure containment where safe. These safety/failure contracts are independent of optional migration and of the no-restart promise for normal supported administration.

Expose configured, pending, and effective values separately. Normal adaptation and supported administrator changes must not restart containers, workers, or CUDA contexts; worker replacement is not a live update.

Drains cover every affected sequence, including retained RAM/storage checkpoints: request-scoped work must finish/cancel and retire; retained sessions follow close/expiry/reconstruction rules. Zero GPU residency is insufficient while suspended sessions retain incompatible state. Zero activity leaves retained history, draft, recurrent/position/media and recovery obligations intact. No retained sessions does not authorize unloading request-scoped or in-flight state.

| Adjustment | Implementation policy |
|---|---|
| Scheduled batch, prefill/visual quota, legal media microbatch, issued CPU task/byte quota within persistent pools, supported proposal length, existing graph/kernel choice | Apply at a coordinated batch boundary inside the prepared maximum; pool topology and library thread settings stay fixed |
| KV ownership and attention degree | Initial map is fixed. Select an accepted alternative at prepared profile activation/drain; live changes are optional Section 11 work, exposed only after complete qualification |
| Maximum execution/resource envelope or graph set | Prepare and validate at explicit supported profile/administrator activation, with quiescence/overlap proof; normal load changes select used quotas and ready plans only |
| Resident-sequence ceiling decrease | Restrict new admissions; preserve existing sequences and context commitments |
| Resident-sequence ceiling increase | Require the complete maximum-context and execution-envelope proof |
| Larger context with unchanged effective positional transform | Activate for all affected GPU/tier-retained sequences together only after enlarging GPU/backing claims as applicable and validating qualified positions, state and execution bounds; otherwise keep pending until the affected profile drains |
| RoPE/YaRN algorithm, frequencies, factor, original-reference length or amplitude change | Initially drain affected GPU and tier-retained state and retire dependent graphs/caches; validate and activate the prepared profile in process. Never change from active-count adaptation or reuse incompatible state (Section 6). |
| Smaller context | Preserve existing windows; keep pending until the homogeneous profile, including suspended retained state, drains |
| ETP, PP/server count, layer cut or expert/endpoint weight placement within Section 20’s matrix | Select at initialization; later change only through a qualified in-process complete-state transition or drain. No automatic PP/server/ETP switching is required. Reject unsupported degrees rather than exposing pending or fallback layouts. |
| Codec, target/visual weights, processor/numerics, structural encoder placement, draft checkpoint or speculative method | Use a qualified in-process transition, or drain all affected GPU/tier-retained sequences and reinitialize in process |
| Retention tier/path or lower queue/backing budgets | Preserve accepted claims; drain or safely relocate required checkpoints before retiring a tier/path (Section 17) |
| No supported in-process transition | Fix the setting at its qualified maximum-concurrency value; expose no live control |

Initially use one homogeneous context/quality profile for recovery, capacity and graph ownership; simultaneous profiles need separate budgeting.

Structural drains stop admissions that could indefinitely extend the old profile. Define resume/closure behavior while existing sessions finish or close under their retention contract. An idle retained session, including a suspended checkpoint, can keep the change pending: report its blocking condition or resource without forced deletion, restart, or a promise to switch at the next zero-active instant.

Byte-preserving GPU page moves retain the quantized representation. Q4-to-Q8 conversion cannot recover discarded information; enabling another speculative method may require historical features/state never retained. Neither is a lossless migration. Use qualified reconstruction where supported; otherwise drain affected sessions. Within an enabled method, zero-proposal operation is a scheduling choice only while conditioning remains valid. Ordinary work scheduling does not change MTP/DFlash, draft checkpoints, KV precision or retained numerical/state representation. Snapshot/replay/deferred-GDN choices and any E21 ring/flush parameters are fixed qualified profile decisions. Required space-triggered flushes materialize only committed records; changing cadence or optional early staggering needs affected FP32/lifecycle qualification.

Target/runtime-weight changes initially drain affected GPU/tier-retained sequences, retire graphs/readers/state and load a prepared replacement. A saved checkpoint cannot be restored under incompatible replacement weights or positions. Saved API items alone do not pin a GPU profile: keep their source identity and use them with another requested profile only through a qualified render/rebuild path, without treating old computed state as compatible. Protected retained compute claims still require the drain/reconstruction contract. If old/new layouts cannot coexist in bounded memory, use an empty-resident in-process transition; uninterrupted service is not promised. Queue or report pending per service policy. Bind queued work to its complete requested model/context/RoPE/KV/method/tokenizer/processor/media profile; reject or retain incompatible work pending. Retained media cannot be reprocessed under changed settings without a qualified transition.

Publish a replacement/capacity only after all ranks finish loading, allocation, target/draft binding, bound initialization and required library/kernel residency. Retire/rebuild affected graphs and tuning plans. On recoverable preparation/load failure, coherently retire partial state and expose failed/pending, never unready capacity. Releasing the old model ends its service guarantee. Retry/reload a prepared profile in process without two full resident models.

## 20 Host memory and parallelism profiles

The historical PLE/ngram mapping is reported as 26.822 GiB. The inspected second shard contains the IQ4_NL `per_layer_token_embd.weight` table with 28,800,138,240 payload bytes; revalidate exact tensor identity/offsets at import. Rank file mappings can share physical pages; eight mappings do not imply eight physical copies. Residency is evictable; measure actual NUMA placement.

Profile cold/warm lookups, page faults, CPU execution, NUMA access, and bounded host-to-device transfers during prefill/decode. Preserve all sixteen bigram/trigram hashes, offsets, resets, IQ4_NL decoding, and PLE convolution. Batch lookups and eliminate proven duplicate TP-rank work, routing common results where beneficial. Shared file pages do not eliminate duplicate lookup/transfer work; PLE can dominate low-load latency after KV deduplication.

Budget host media decode, normalized pixels, optional retained projected embeddings, immutable replay sources/caches and pinned transfer stages separately from PLE and history. Bound CPU worker concurrency, queued decoded media and every buffer lifetime; do not allocate maximum video working sets for all resident sessions. These media resources do not provide attention-KV capacity.

Keep the raw-indexer RAM archive disabled. Compact indexing has no growing raw array. The historical 13.038 MiB/session/GPU ring/staging cost belongs to the full-raw layout, not an initial compact allocation or capacity-saving option.

Section 17 adds qualified idle-session RAM/optional-storage checkpoints with FIFO restoration. Per-step host paging of executing attention KV or retained expert weights remains outside scope; it cannot supply GPU-resident capacity or rescue overcommitment.

### Supported configuration matrix

The required supported target matrix is **TP8 × PP1/PP2 with ETP8/4/2**: six profiles. ETP1 is excluded from implementation and qualification scope. TP is fixed nonexpert tensor parallelism, ETP splits each routed expert and EP distributes different experts. PP1 runs on one eight-V100 server. PP2 runs one eight-GPU stage on each of two physical servers, sixteen GPUs total; it is not a four-GPU-per-stage mode on one server.

| Servers | Total GPUs | Nonexpert TP | PP | Expert ETP | Derived EP per stage |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 1 | 8 | 1 |
| 1 | 8 | 8 | 1 | 4 | 2 |
| 1 | 8 | 8 | 1 | 2 | 4 |
| 2 | 16 | 8 | 2 | 8 | 1 |
| 2 | 16 | 8 | 2 | 4 | 2 |
| 2 | 16 | 8 | 2 | 2 | 4 |

Resolve `server_count=PP`, `world_size=8×PP` and `EP=8/ETP`. ETP and EP stay inside each server; never multiply TP by ETP or stretch an expert group across nodes. Each target layer has one expert representation, split among its stage’s ranks. ETP8/4/2 gives intermediate widths 80/160/320 and groups owning 512/256/128 experts per layer. Nonexpert operators retain TP8 with their declared small rank-local/owner-served exceptions. Shared experts are part of that nonexpert plan, not another routed-expert bank.

Use one serving path and only the three degree fields shown in Section 6. Reject other nonexpert TP degrees, other PP degrees, DPA, independent full-model or expert-bank replicas, cross-node EP, heterogeneous stage widths, dedicated target/draft GPU layouts and independent method parallelism settings. Apart from the optional TP4×PP2 study below, these are not deferred implementation commitments or automatic alternatives. Prepared operator tactics, accepted fixed attention layouts, separately gated live-placement experiments and measured tensor-copy optimizations remain eligible within this fixed matrix and Section 1’s policy; they cannot silently alter the target group/routing topology.

### Ownership and companion placement

PP1 owns all 48 target blocks on one server. PP2 initially owns blocks 0–23 on the first server and 24–47 on the second, with all eight GPUs participating in their stage’s nonexpert operators. A measured layer-cut rebalance remains within TP8×PP2, but requires a new qualified weight/state/resource map; no general heterogeneous-stage framework or launch knob is required. Bind physical GPU UUIDs and logical ranks on every node, using each server’s validated eight-GPU mesh and suitable local ETP4 quads/ETP2 pairs.

Declare producer/consumer row layouts, global/local expert IDs, selected routes, casts, shared-expert contribution and the qualified output reduction. Complete replicated inputs may avoid a new input all-to-all; otherwise assemble/dispatch the actual required rows. Empty groups retain the agreed collective order. Preserve every original selected expert/gate and execute each assignment once. ETP changes arithmetic/reduction order despite unchanged packed values; qualify the complete model at each degree, including mixed block formats and all tails. Keep expert-to-group placement fixed during ordinary serving. The 144 routed SC117 source tensors each use one shape/encoding across 512 experts, so equal-count per-layer placement is source-byte-balanced; mixed formats vary across layers/projections, not expert bytes within one tensor. Validate actual runtime padding/representation and profile owner-service/coactivation skew before a more informed static map or local E05 scheduling. Live weight reshuffling/EPLB remains a separate measured structural experiment, not a load-response obligation.

Keep each layer’s canonical KV, pooled indexer, bounded raw tails, GDN/conv/PLE and recovery state on its owning stage. Each stage retains two KV-head groups of four ranks, a qualified fixed unique KV map and a fixed initial indexer owner. Four-way striping is the first control; other fixed layouts or live degree changes do not add mandatory profile axes. The TP8 operator geometry remains the same at PP1 and PP2; stage-local layer counts, ownership phases and memory costs change. No active attention or expert paging across host memory or the network is introduced.

The selected MTP or DFlash method uses one TP8 group on the sole server for PP1 or the final server for PP2. **MoE MTP uses the selected ETP and its derived stage-local EP**, avoiding a separate target×method degree matrix; validate the pinned companion geometry and all three degree implementations. Dense DFlash has no ETP axis. Keep one selected method resident, with a unique valid history/state plan and explicit artifact/precision/position binding. MTP’s QSA and full-prefix initialization/refresh obligations remain unchanged.

The first stage owns target input embedding and PLE lookup; the final stage owns target output/head and companion execution. Use actual local aliases or bounded owner-served embedding vectors, preserving MTP’s full flattened block 47 HC and DFlash’s five specified completed-layer means. Forward required earlier-stage DFlash features and media/position metadata with bounded lifetimes. A measured packed embedding copy on the final node is permitted under Section 1’s byte/quality gates, not required or reported as an alias. Accepted immutable copies belong to the prepared effective profile; dynamic creation/release with active count is not required. Keep one qualified vision service at the input stage; no extra encoder or full target copy is implied by PP2.

### Two-server execution and resource proof

Move the full four-stream HC boundary—10,240 values per row—between stages in its qualified dtype, never its stream mean. FP16 is 20 KiB/row; FP32 is 40 KiB. Pipeline independent ready decode groups and causal prefill chunks with bounded queues, useful row budgets, fairness and graph-safe communication. Initially permit at most one unresolved speculative verification round per sequence; start its next sample-dependent round only after the prior verdict and method-defined state commit. Commit/rollback/cancellation, RNG, cached/committed frontiers and output publication form one transaction across both servers. Lookup/feature/send/feedback services must make progress during pipeline waits; retire readers before buffer reuse. Count all network service and speculative maintenance in complete-round timing.

For PP2, record NIC count/rate, GPU/NIC PCIe-switch and NUMA locality, actual GPU Direct RDMA transport and supported driver/peer-memory path, NCCL version, rail/rank mapping, measured small-message latency and aggregate/contended bandwidth. Use the existing native SM70 stack and suitable GPU-tray NIC paths. Qualify each node’s GPU peer access and each required cross-node transfer, not just NIC RDMA capability. Four NICs do not prove four-way striping: use the actual qualified transfer plan, avoid redundant full-boundary sends and charge its buffers. Selective GPUDirect pin/registration, completion ordering and graph lifetimes must remain valid. Sources: [CUDA 12.9 GPUDirect RDMA](https://docs.nvidia.com/cuda/archive/12.9.1/gpudirect-rdma/index.html), [NCCL network controls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) and [Supermicro GPU-tray RDMA slots](https://www.supermicro.com/flyer/f_NVLink-Family.pdf).

The target source payload remains 51.271565 GiB across the selected deployment, before actual packing and required rank-local copies; PP2 splits that one target by depth. Do not divide every runtime bucket by sixteen: the last stage owns the entire companion, target output/head and some workspaces; one fixed vision service can bind another rank. Solve admission against every physical GPU on both servers, preserving the same context, codec, method, recovery, device-floor and workspace guarantees.

With the initial 24/24 split and stage-local QSA-MTP, PP2 growing history rates are 864/1008 B/token/rank first/last for Q8 and 456/532 for iso4. At 24 sessions ×262,144 positions:

| Deployment / stage | Q8 history GiB/rank | iso4 history GiB/rank |
|---|---:|---:|
| PP1, all 13 caches on one eight-GPU stage | 10.968750 | 5.789063 |
| PP2, first stage: six target caches | 5.062500 | 2.671875 |
| PP2, last stage: six target plus one MTP cache | 5.906250 | 3.117188 |

These are history-only ideal stage means, not certified allocations or capacity. PP2 current FP32 GDN matrix state is 6.75 MiB/session/rank for 18 recurrent layers/stage, versus 13.5 MiB at PP1. DFlash instead adds its bounded window/features only on the final stage; keep its range-qualified storage and all other state/recovery costs explicit. Recompute local ownership, guards, padding, retained versions and peaks rather than copying PP1 averages.

### Validation and qualification

Require exactly PP participating physical servers and exactly eight suitable GPU ranks per stage, with world size 8×PP; reject mismatches and never substitute local TP4×PP2 for two-server PP2. Reject invalid degree values, rank/layer/expert coverage, unsupported formats or insufficient worst-rank headroom before admission. Initialize only the selected profile’s stage-local and pipeline communicator/graph families. Cache/checkpoint numerical compatibility is separate from physical mapping epochs; matching shapes alone do not permit old-state reuse.

Implement the three PP1 ETP cases now. The three TP8×PP2 cases remain **pending two-server hardware and execution qualification** until that environment is available; unit/static checks are not multi-node performance or correctness evidence. Track requested, implemented, qualified and effective status without claiming unavailable modes work. ETP1 requires no kernels, launch option, profiling or qualification work.

Apply the existing C1/2/4/8/16/24 prefill/decode and full-capacity gates to each deployment, with identical selected numerics and fully retained histories; extend activity through the actual maximum. Compare ETP degrees on the same PP/hardware with matched workloads and complete method costs. Report PP1-versus-PP2 as hardware scaling, not an isolated software optimization; invent no two-server token/s promise. Structural selection stays fixed during ordinary active-count adaptation. Later ETP, PP/server or weight-placement changes need Section 19’s qualified in-process transition or drain; no automatic multi-node expansion/contraction is required.

### Optional future TP4×PP2 study

Retain **TP4×PP2 with ETP4 or ETP2 only** as a low-priority future implementation/benchmark choice after the required TP8 matrix. It is not a current support, development or release gate. A conventional one-server, eight-GPU layout has four ranks/stage: ETP4×EP1 or ETP2×EP2. ETP8 cannot fit that stage and is excluded from this future study; ETP1 is also excluded.

No TP4-specific kernels, group framework, launch modes or benchmark sweep are required now. Before any future promotion, define exact target/method/endpoint/history owners, numerical and graph contracts, complete transactions and worst-rank resources; TP8×PP2’s two-server budgets do not certify a four-rank stage. Apply the existing matched C1/2/4/8/16/24, capacity, quality, fairness, no-restart and duplication gates.

## 21 Observability and qualification

### Runtime receipt

Publish a compact machine-readable effective-profile/capacity receipt and user-facing limiting-resource explanation. Include:

- Exact target/MTP/DFlash/visual/processor/codec IDs, hardware/runtime manifest, immutable resolved RoPE/YaRN and companion position manifests/epochs, native/reference versus configured/qualified maxima, context, media envelope/placement, configured ceiling, computed resident capacity, and qualification status.
- Standard response/conversation references, internal request/session generations and lifecycle status; separate request-scoped/retained, GPU-admitted active, FIFO-waiting, idle, suspending/suspended/restoring and scheduled sequences; queue order/age, per-principal use, tier budgets/bytes, checkpoint generations, slot-release-to-admission delay, transfer/restore time, I/O failures and blocking reasons; text/visual prefill rows, frames/pixels, cache hits, maintenance/replay and verification work.
- Proposal policy, draft-valid state, acceptance distribution, recovery/replay work, and committed generated tokens.
- Timestamped per-rank weights, history pages, state, graph pools, language/visual workspaces, external allocations, physical free memory and protected commitments; bounded host/pinned/media-store use and queue occupancy. For duplication, include named copies, gross/net whole-system and per-rank steady/peak increments, cumulative package, ≥1-GB review tier, matched gains, capacity cost and acceptance/fallback rationale (Section 1).
- Fixed placement/map identity, slot/ownership generations and admission restrictions; if optional live migration is adopted, additionally report its requested/effective degree, mapping epoch, outstanding bytes, transient occupancy and complete transition time. Include Section 20’s server/world-size, TP8/PP/ETP and stage-local EP, stage cuts, UUID/GPU/NIC maps, RDMA/rail measurements and method/endpoint/history plans; distinguish geometrically valid, implemented, qualified and selected profiles.
- Useful text/visual prefill and committed decode rates, media decode/encode/transfer time, TTFT, inter-token latency, per-request service/fairness and error/cancellation counts.

Separate startup/current readings, measured allocations, logical reservations, and planned maxima. Do not add allocated to reserved bytes or count tentative/draft tokens as generated throughput.

### Qualification matrix

| Axis | Required cases |
|---|---|
| Artifacts, precision and method | Every released target/runtime-weight pairing and supported target KV with none/MTP; every DFlash pairing/draft format; selected NVFP4 import/conversion/activation semantics |
| Context and RoPE/YaRN | Native and every released extended positional profile; short/intermediate/near-full histories, native/new-limit boundaries, coordinate-versus-row bounds, invalid/nonfinite profile rejection, partial-RoPE amplitude parity, Q8/Q4 quality, required MTP and separately qualified DFlash, media, profile identity and in-process changes; input plus output fits the allowance |
| Concurrency | Integer sweep from 1 through capacity; detailed thresholds, exact endpoints and measured breakpoints; burst from 1 to maximum |
| Retention | Request-scoped release and intentional retention; one resident/one active, all resident/one active, all resident/all active; partial/near-full histories; supported multi-sequence accounting |
| FIFO and tiered retention | Fresh/resume order, bounded eligibility, no warm-resume jump, fair admitted progress, completion/idle slot handoff and immediate wakeup, no routine runnable preemption; above-GPU-capacity backlog, idle demotion and RAM/optional-storage restore; complete-state parity for Q8/iso4, MTP/DFlash, native/extended positions and media; arbitrary owner/slot phases, cancellation/expiry/load-change races, quota reduction, no transfer-progress deadlock, all-slot idle GPU-only blockage versus guaranteed FIFO progress, exhausted/unhealthy tiers, corrupt/partial checkpoints and bounded retries |
| OpenAI API/session identity | Unmodified SDK Chat/Completions/Responses calls, full-history automatic reuse, stored response chains/branches, request-scoped instructions changed/omitted across turns and resulting prefix misses, owner isolation, reference expiry/deletion, store/usage/errors and streamed/nonstreamed lifecycle; no custom session headers. Separately qualify standard Conversations/item APIs and conversation/previous-response exclusivity, concurrent mutations, durability/quotas and cache-independent logical retention, record/compute-claim separation and deleted-parent descendant lifetimes before advertising stable conversation handles |
| Prefix/cache integration | Exact rendered/media/profile identities; no-hit/partial/full-prompt hits; new-turn versus interrupted-request RNG/constraint state; earlier forks and immutable longer branches; all hybrid/draft components at a common valid cut; COW/reference/lease races, concurrent load coalescing, dirty-page reuse, cache eviction during save/restore, protected-checkpoint survival, reclaimed page-class guarantees and bounded cold-path execution |
| Workload mix | Prefill only, decode only, mixed long prompts/decode, heterogeneous acceptance, routed expert skew; visual bursts with decode fairness |
| Media | Standard wire-to-processor mapping and explicit native-video/frame-sequence distinction; single/multiple images, aspect ratios/resolutions, interleaved text/media, video temporal order/timestamps, exact grids/rounding/temporal padding, boundary limits, explicit frame-count paths and invalid metadata |
| Multimodal lifecycle | Cancellation/resumption inside visual spans and at media boundaries; same-token/different-media prefix misses; retained replay identity, stale media completions and bounded cache misses; PP2 forwards exact early-stage draft features/media positions and includes terminal speculation in the resource proof |
| Ownership and transitions | Fixed-map page/slot phases, exact merge, allocation permutation and prefix/restore ownership are required. Only an adopted Section 11 experiment additionally requires every live degree transition both ways, partial/union maps, full/incomplete bundles including PP2’s remainder, intermediate worst-rank peaks and oscillation tests. PP2 independently requires cross-node send/feedback/cancel/restore retirement, one unresolved round per sequence, independent-group overlap and complete-state failure/pending behavior |
| Lifecycle | Completion, all profile-defined generation terminators, PLE resets, cross-token/block stop strings, context exhaustion, partial-prefill cancellation/resumption, cancellation around commit/publication, slow/disconnected clients, slot reuse, bounded admissions and pending drains |
| Speculation | Text/visual MTP aligned prefill/target-conditioned refresh and DFlash HC/position alignment; all-accept, early/late rejection, zero accepted candidates, warm zero-proposal operation, final bonus/anchor handling and recovery |
| Sampling policy | Every exposed greedy/stochastic policy; supported constraints/penalties; rejection, cancellation and resumption preserve canonical bookkeeping; unavailable verifier capability rejects before state mutation |
| Profile changes | A-to-B-to-A without worker/context restart; enlarged execution/descriptor bounds; recoverable load failure after old-profile retirement and in-process recovery; repeated switches without uncharged residual allocations |
| Long-lived service | Repeated fill/drain and slot permutations, typed page-class availability, prepared graph lifetime, lazy-library residency and full resident commitments plus the largest permitted media unit. Exercise supported preparation/admin replacements; live migration sweeps apply only if that experiment is adopted |

A 260,096-row expanded prompt plus up to 2,048 generated tokens fits 262,144; a 262,144-row prompt plus a reply does not.

### Correctness and quality gates

Exercise import and device-address calculations across the `2^31` and `2^32` byte boundaries and the largest supported offset, including intermediate products and descriptor serialization. Prove bounds for narrower local indices.

Exercise every valid target/MTP expert ID, especially 0/255/256/511, across actual routed-row regimes, empty/tail groups, TP8 K80 shards and mixed-format blocks. For all six required Section 20 cases, verify global-to-local expert IDs and exact-once assignment across every stage-local ETP/EP map, with the selected MTP degree and pipeline transaction where applicable. Poison output/scratch before controlled tests to prove every live row is written and padded/stale rows cannot leak into a result. Feasibility guards and executed kernels must use the same actual expert/group counts, packing and expanded scratch sizes. A repaired skipped-output or nonfinite route invalidates affected timing controls; rerun matched baseline/candidate correctness and performance before claiming improvement.

Compare chunkwise prefill and recurrent/verification outputs, convolution tails and final states against the pinned reference with fresh/nonzero initial states, partial chunks, mixed sequence lengths, masked rows, resets, text/visual rows and every accepted speculative prefix. Include PLE's dilation-three, nine-row history across chunk and rejection boundaries. Import parity must cover the stored decay representation and complete coupled head ordering before TP partitioning. Measure recurrence/convolution time and bounded workspace separately in prefill, decode and mixed workloads.

Compare unsharded/sharded attention with identical configured codec and target semantics. Cover empty shards, extreme scores, pooled-history boundaries, partial pages, indexer ties, long absolute positions, masks, and codec metadata. Separately compare quantization quality against the chosen higher-precision reference. Verify all four pooling phases, spans crossing multiple groups, every accepted prefix, whole-append rollback and reuse cuts inside groups. Compare pooled values, selected positions, ending raw tails and group-first coordinates against the pinned semantics for text and visual rows.

Test accepted-prefix recovery across attention KV, pooled indexer history and raw pending state, recurrent state, Gated DeltaNet/PLE convolution tails, PLE token/hash history, draft state, target/draft positions, media identity, sampling/constraint/output bookkeeping and RNG streams, including cancellation and slot-generation races. Test that terminal disposable work omits only maintenance with no continuation/checkpoint/cache consumer, while retained sessions and warm zero-proposal execution preserve valid conditioning. Compare greedy outputs where exactness is expected; validate stochastic acceptance and resulting distributions without requiring identical sampled strings across schedules. Exercise partial reasoning/tool delimiters and argument fragments, byte-stop cuts, multiple calls, rejection, cancellation and same-request resumption around publication; preserve stable IDs and prevent duplicate internal items/deltas.

Qualify RotorQuant's actual SM70 kernels through progressive long-prompt prefill and long decode, beyond isolated-vector error, including the selected schema/flag and both 128-value blocks of each D256 head. Verify MTP import fusion, flattened 10,240-wide hidden normalization, complete learned norm vectors, single Gemma-offset conversion, HC transforms, actual next-position visual embeddings and explicit QSA execution against identical decoded weights/numerics. Qualify MTP with both target-selected KV codecs; compare weight formats by quality, acceptance and throughput without requiring identical proposals. For DFlash 2, verify all five completed-layer HC means across TP8/chunking, every activation/output/KV range boundary, block-local convolutions, selector probabilities, 2,048-window wrap/rollback, separate RoPE, and conditioning after rejected/skipped proposals. Detect future-token leakage into earlier target logits or indexer decisions, including the up-to-three-token QSA tail.

Compare visual preprocessing, patch/grid order, projected embeddings, target positions and representative logits with the pinned reference. Evaluate SC117 image/video understanding on OCR, charts, screenshots, documents, spatial relationships, multi-image comparisons and temporal video tasks. Verify MTP/DFlash conditional acceptance and target-equivalent output with visual prefixes; family compatibility and unchanged vision weights do not establish quality after language-model modification.

For Section 17, compare suspended/resumed continuation with uninterrupted execution under identical profiles and sampling ownership, including legal pending anchors, stopped output and partial-prefill retention. Measure FIFO waiting/TTFT, slot handoff/idle-reclaim latency, admitted fairness, full-load committed decode, tail restore latency, PCIe/NUMA/storage contention and bytes per completed request above GPU capacity. Separate requested logical-input rate from executed prefill; report reusable hit rows by GPU/RAM/storage tier, avoided work, dirty versus reused backing bytes, load coalescing and cache-management overhead. Include cold/miss paths, repeated cycles and slow clients; claim neither a hard wait bound nor faster decode from added storage alone.

Use focused synchronization/memory diagnostics for kernel risks and lifecycle races. Each gate must address a concrete correctness, resource, or performance risk; avoid tests that merely repeat implementation details.

Qualify Section 2's combined dependency manifest: inspect SM70 coverage and loaded libraries, exercise selected GEMM algorithms and every communicator at actual dtypes/message sizes, and test captured/uncaptured serving. Framework/extensions must not silently introduce unsupported kernels/backends. Use NVTX/Nsight Systems for host gaps, collective critical paths, transfers, and overlap; Nsight Compute for bandwidth, Tensor Core/SIMT activity, occupancy, and spills; focused Compute Sanitizer checks and NCCL correctness/performance tests. Separate instrumented runs from throughput receipts.

Resolve applicable compiler issues before release. NVIDIA issue 6111901, introduced in CUDA 12.8 and fixed in 13.2.2, concerns failed reconvergence under nested thread divergence. Establish applicability to the exact SM70 toolchain and any supported 12.x correction or validated source-level mitigation; assume neither V100 impact nor a backport. Retain a targeted reproducer/regression check when applicable. CUDA 13's fix does not restore SM70 compilation. Source: [NVIDIA compiler issue](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html).

### Shared-change and backend release gates

Classify each change by the concrete profiles/contracts it affects. For shared preparation/API/state/sampling/media/build code, verify all affected selected backends and installed callers; for a local kernel/packing change, verify its full legal shape/numerical/resource region and integrated model workload. Preserve independent oracles and recorded failures. Do not demand a hypothetical all-model/all-device Cartesian test matrix, inherit a pass from another architecture, or waive a real affected target because another backend passes.

For each affected profile, bind exact source/dirty patch, installed wheel/native origins, immutable foundation/application and target/checkpoint identities. Run source/CPU contract checks, installed-artifact/native checks, actual target model/state/quality/API tests and matched resource/performance measurements at the level the change requires. Checkout-injected pytest paths, cross-compilation, fake devices and component oracles retain their limited evidence scope. Unavailable actual-target checks remain pending and constrain release claims; they do not turn into synthetic passes or aggregated historical test totals.

Preserve the current shared contracts: principal-isolated bounded logical records; request/input/output ownership until caller and backend retirement; committed-token/request-domain RNG and exact probability/filter ordering; coherent prefix/recovery boundaries; immutable media replay/pixel leases until DMA/encoder retirement; and host-only status under a valid profile-reader lifetime. Equivalent implementations can replace existing classes directly after the relevant proof and end-to-end gates. Shared semantics do not require identical bit sequences, layouts or compute precision across devices; each selected numerical profile declares its own equivalence and quality boundary.

Complete rebrand/package/deployment/help/recipe/API-capability and canonical-design updates with the actual code cutover. Verify final installed artifacts outside the checkout, their packaged native/model assets and updater/release destination. Advertised capabilities reflect implemented and qualified exact combinations. A stale family README, accepted flag or newer dependency version cannot establish model/device support. Keep SC117’s Section 5 targets and nonregression gates intact; scope existing deployment-specific accepted trades and future changes to their declared profiles.

### Performance implementation audit

Retain a trace-backed inventory for each released regime: H2D/D2H/D2D calls/bytes per committed decode token (per text/visual input row or media unit for prefill/encoding); host waits/implicit synchronization; allocation/initialization; kernel launches; graph replay/breaks/recapture and compiled-cache misses; per-rank idle/communication time; memory throughput/spills; CPU thread/cpuset/NUMA placement, PLE faults/lookup latency and locked-memory peaks. Distinguish useful/tentative work and steady-state/cold preparation. Sample asynchronously so telemetry does not become the bottleneck. Record selected plan/key, useful/padded rows and decision cost through existing receipts; expensive diagnostics remain separate from mandatory scheduler inputs.

Identify the actually executed operator/dispatch, installed native binary and caller signature for each row/shape regime. Configured, compiled, warmed, captured and replayed paths are distinct evidence; dummy capture lengths cannot freeze an invalid route. Validate warm compile/AOT caches and the installed artifact outside its source checkout, with observation-neutral instrumentation. Include shrinking batches and near-context-limit verifier widths in replay tests; unsupported paths must fail explicitly rather than silently skip work or change numerics.

Explain remaining hot-path boundaries/waits and compare the best qualified alternative at every integer activity count, including one active with full retained histories, near-full-history/full concurrency and mixed media/prefill/decode. Check Section 15's capability table and deployed-container transport paths; prove dependency/buffer correctness through cancellation, stream overlap and graph replay. Require numerical/output-semantic gates, complete resource feasibility and repeatable whole-workload benefit within predeclared fairness/latency regression tolerances; kernel-only gains are insufficient. Log rejected/unhelpful candidates rather than exposing gratuitous knobs.

Retain production error checks, ECC and supported power/thermal protections. Record host/GPU clocks, power, throttling and competing work to control comparisons. Tuning cannot disable reliability mechanisms, silently oversubscribe managed memory or page executing/GPU-only-promised histories to host; Section 17's explicit tiered-retention contract is separate. Investigate MPS, persistent kernels, alternate allocators, host huge pages, direct storage or extra frameworks only for a measured bottleneck; qualify required PP2 RDMA separately under Section 20, with independent support, lifecycle and resource qualification; preserve single-target and no-restart contracts.

### Performance gates

Use repeatable prompts, generation settings, target/draft revisions, warmup, stable clocks/thermals, and timing boundaries. Repeat enough to separate gains from noise. Compare adaptive execution with the best qualified static profile under identical context, quality, retention, and workload conditions. Apply Section 5's milestone matrix and performance targets. Compare before/after commits under matched workload/profile and actual history/cache conditions; predeclare repetitions, statistic and noise tolerance, alternate or otherwise balance A/B run order, and rerun ambiguous results. A noisy apparent improvement is not a passed milestone.

Keep history ranges and reasoning/sampling settings comparable across active counts. Test different owner/slot phases at low activity, not only slot zero. Profile indexer and PLE separately to avoid attributing their latency to attention placement or proposal depth.

Report useful text prefill, visual-row processing and committed decode separately, alongside whole-step/end-to-end rates and isolated-kernel measurements. Include CPU media preparation, encoding, transfers, quantization, draft maintenance, rejected work, collectives, recovery, scheduling and amortized migration. Visual rows are not ordinary text tokens/s. Report latency distributions and fairness so throughput cannot hide starvation. Select the best measured eligible profile at each load regime. Retain a per-milestone table for all six required counts: before/after prefill and aggregate committed decode rates, absolute/relative changes, variability and pass/fail. For the 1- and 24-active Q8/native-context decode targets and the bound prefill reference, report target attainment and remaining absolute/percentage gaps. Aggregate throughput uses one common measurement interval across participating requests; include per-request service so aggregate gains cannot hide starvation. Overall averages or weighted scores cannot waive Section 5's decode nonregression requirement.

## 22 Implementation sequence and exit gates

Use Section 2's pinned source/status map and dependency/processor manifests with Section 15's operator/library contracts. Required work below means complete, refactor where justified and qualify the behavior; reuse matching current implementations and controls rather than rebuilding them by default. Parallelize where state/resource contracts permit; every changed execution path passes Section 21's applicable gates. Shared changes preserve the selected existing profiles throughout the V100 delivery.

| Stage | Required work | Exit evidence |
|---:|---|---|
| Source and ownership baseline | Freeze the actual `python-0.6`/ProviderOne commit and installed origins, map current implementations and evidence to this contract, select the canonical design and record carry-forward/replacement owners | Exact source/image/artifact identities, completed versus pending gates, protected selected profiles and no duplicated editable design |
| 1 | Qualify the selected SM70 stack and complete family manifests and bounded mixed-format GGUF import; bind exact SC117 IQ3_S, shared MTP, Inco draft, visual/processor artifacts and recovery contracts | Build/library IDs, complete tensor inventory, source/runtime parity and fresh per-rank resource receipt |
| 2 | Add canonical page/session/media metadata, checked addressing, typed arenas, full-context commitments, accepted-prefix/sampling/PLE/streaming APIs and SM70 recurrent/convolution kernel regimes using Q8 | Target/state parity, bounded prefill/decode work, transcript, cancellation, media identity and page-class guarantees |
| 3 | Compact raw indexer storage to bounded pending/transactional state and remove persistent KV/pooled-indexer replication through owner execution; retain weight TP8 | Exact growing/fixed history bytes, rank balance, group transactions and output/selection equivalence |
| 4 | Qualify one fixed unique KV map, exact partial merge, topology-aware mapping and separate fixed MTP-cache placement; four-way striping is the first control | Fixed-map state/numerical parity, complete worst-rank ledger and serving profiles; no live degree-conversion obligation |
| 5 | Port RotorQuant Q4 and direct SM70 pack/unpack/attention | Layout, quality and long-context gates; no full-history mirror |
| 6 | Implement required image/video processor, FP16 vision path, target positions/PLE inputs, bounded staging/replay and mixed scheduling | Reference parity, media quality, fixed placement and complete maximum-work resource envelope |
| 7 | Implement required shared-Q8 QSA MTP with actual SC117 aliases, selected Q8/Q4 KV, full-prefix text/visual initialization and refresh | Pinned-control/imported-weight parity, useful speculation, recovery and complete capacity receipts |
| 8 | Prepare bounded graph/kernel/batch/width plans and implement the small scheduler with mixed media/prefill/decode quotas inside one maximum envelope | Safe immediate load changes without recapture/state movement; Section 21 audit, integer sweep and measured useful breakpoints |
| 9 | Integrate IncoAI DFlash2, TP8 HC capture, range-qualified SM70 precision/kernels/selector, qualified stochastic verifier and bounded SWA/features with separate draft positions | Independently qualified SC117/Q8-KV and SC117/RotorQ4-KV text/visual correctness, finite activation/KV and proposal probabilities, supported sampling, capacity, acceptance and performance |
| 10 | Complete in-process administration, observability, sustained-load tests and operations; implement native/YaRN positional profiles and extended Q8/Q4 capacity/quality qualification with required MTP and separately qualified DFlash | No tuning restart; immutable position identity, bounded phase resources, native/extended parity and quality, lifetime commitments, readiness/failure contracts and full-load receipts/benchmarks |
| FIFO, prefix reuse and retained-session tiers | Integrate bounded FIFO admission, completion/idle slot handoff and fair GPU scheduling with required compatible prefix reuse, one page/state catalog and complete RAM/optional-storage suspension; qualify standard OpenAI-compatible Chat/Completions/Responses lifecycle and independently qualified standard Conversations handles using the shared logical store | Section 9/17/21 cold/hit/miss, handoff/restore, ownership/reference, state/quality, failure and backlog performance receipts; no restart or routine full-history round-trips |
| Parallelism profiles | Implement the required TP8×PP1/PP2 with ETP8/4/2: three one-server cases and three two-server cases; one TP8 stage/server and selected ETP for MTP | All required ETP degrees, rank/layer/expert coverage, fixed-map state/graph parity, stage-local capacity and full benchmark matrix. TP8×PP2 remains pending actual second-server/RDMA qualification; TP4×PP2 is optional future work, not a release gate. |
| Optional migration study | Only after the best fixed placement and cheap dispatch show a measured residual opportunity, evaluate live degree changes/migration under Section 11 | Complete net payback, intermediate-map/resource/state correctness and burst/oscillation evidence; no initial development or release gate |
| Shared and existing profiles | Preserve selected common implementations and existing device/model contracts during each affected milestone; installed package/rebrand, metadata, API and backend changes use Section 1/21’s scoped cross-profile checks | Installed-origin/native identity and affected profile numerical/state/resource/API/performance evidence; no inherited pass totals or unsupported support claim |
| Follow-on models/formats/hardware | After current SC117 gates, complete missing mainstream family/format and new hardware pairings, including concrete NVFP4/MXFP8 recipes. Existing Blackwell support is preserved throughout; Ada/Ampere additions use exact capability/operator admission | Exact model/container/recipe/backend support, state/API/quality, build and peak-ledger proof, per-count and actual-maximum performance; no inherited SC117 geometry or absolute speed promise |
| Later | Evaluate SAW and other architectural alternatives | Separate quality/capacity/performance evidence before inclusion |

MTP is a required deliverable independent of DFlash availability or speed. The initial default uses MTP; DFlash is an independently selected method profile. Both need their own geometry and multimodal gates, and neither is silently disabled to advertise more capacity.

Retain each stage's commit/build ID, workload, resource receipt, results and limits. Qualification does not extend implicitly to larger contexts, different media envelopes or higher concurrency.

## 23 Integration inputs and unresolved measurements

The current SC117 model choices are settled; Section 1 records the follow-on roadmap. Implementation supplies these remaining current-profile bindings and measurements:

1. **Artifacts/build (Sections 1–2, 12–15):** refresh the pinned `python-0.6` starting commit against the actual local checkout/dirty patch and installed host/native/image identities; bind ProviderOne ownership, canonical design, carry-forward/replacement plan and exact device/model capability map; SC117 binary census; tokenizer/template; target/MTP/DFlash/vision tensor maps/transforms and allocations; ordinary Q8, RotorQuant Q4 and explicit MTP QSA bindings. Section 2’s current source audit supplies implementation starting points; historical snapshots and recorded tests retain their original scope, and neither establishes the installed ProviderOne behavior.
2. **Server/stack (Section 2):** installed CPU/DIMMs, GPU UUID/ranks, links, PCIe/NUMA, firmware and newest compatible build/library IDs; SM70 and compiler/library qualification.
3. **Multimodal (Sections 2, 16–17):** standard endpoint/content-type-to-decoder/processor wire mapping, native video versus frame-sequence semantics, service media bounds, fixed encoder placement, operator/position parity, CPU/GPU/pinned/replay peaks and task quality.
4. **Speculation (Sections 12, 14):** MTP import/norm/HC, lookahead-dependent cache identity and text/visual refresh; Inco DFlash2 TP8 features, weight conversion and activation/output/KV range-qualified representation, selector, SWA, separate positions and useful blocks for SC117 with each KV codec. Recalculate the ledger if FP16 draft storage fails.
5. **Recovery/tiering (Sections 12, 17):** complete checkpoint schema, RAM/storage quotas and paths, retained media/replay, replacement/transfer overlap, bandwidth, failure/durability policy and proven shared lifetimes.
6. **Execution (Sections 10–18):** prepared row/prefill/media bounds, fixed history map, recurrent/convolution kernels/chunks, graph/tactic breakpoints and draft-width policy across actual activity/workloads. Live history conversion, online cost fits and more elaborate scheduling policies remain optional gated work, not missing initial bindings. Graph-set replacement belongs only to qualified preparation or explicit supported administration.
7. **Service/API (Sections 5–7, 19):** context/position selection and quality thresholds; maximum capacity then full-load decode with fairness/latency/queue limits; lifetimes/multiplicity, fresh/resume FIFO order, per-principal/queue limits, input commit/cancellation, output backpressure, drains and readiness/failure behavior. Bind Section 5's user-verified reference benchmark receipts and current implementation rates/gaps without inventing missing workload or speculative settings.

Publish the qualified integer maximum from the complete post-import ledger for the selected context/method/media envelope. Historical weights establish no SC117 limit; 20-Q8/32-Q4 remain reference points, never promises or threshold ceilings. Section 5's 24-active Q8 benchmark is a separate explicit performance/capacity objective.

## 24 Acceptance checklist

All detailed requirements and applicable Section 21 gates remain mandatory. Release also requires this cross-section checklist:

- Sections 1–2 bind the actual `python-0.6`/ProviderOne source and installed origins, selected existing profiles, minimal shared/device ownership and canonical repository design. Rebrand/package/updater and affected cross-profile gates pass Section 21; no historical suite or other device is substituted for actual-target evidence.
- Section 15's hardware/hot-path rules and Section 21's trace-backed performance implementation audit pass, including NUMA placement, inference-mode, transfer/synchronization, graph, numerical and resource checks.
- Explicit context/RoPE, numerical/method/media profiles, capacity and limiting-resource explanation; extended windows pass every Section 6 gate.
- Exact SM70 build/loaded-library identities and resolved compiler/library issues; combined-stack qualification and bounded workspace/communication (Sections 2, 21).
- ProviderOne’s independent ownership/removal policy and phased roadmap are explicit; no inherited internal API, framework or module must be retained for upstream compatibility. One manifest-driven serving path bound to the actual build; independent source/runtime weight, activation and target/draft KV choices; capacity from imported allocations (Sections 2, 7).
- Every sequence can reach its promised expanded input-plus-reply window with selected speculation/recovery. Image/video → text passes declared media bounds; native audio/media generation stays outside scope (Sections 6–7, 16).
- Account for independent sequences; retire request-scoped state at completion and preserve retained/input-commit semantics through prefill cancellation/resumption (Sections 6, 16).
- Section 20’s exact six-case TP8×PP1/PP2×ETP8/4/2 matrix has explicit implementation/qualification status; PP1 requires one eight-GPU server and PP2 two. ETP1 is excluded. TP4×PP2 with ETP4/2 only is retained as an optional future study; it is not exposed or required now. Accepted duplication satisfies Section 1’s byte accounting, tiered evidence and explicit capacity/performance decision. Unique canonical KV/pooled histories, qualified fixed ownership, any accepted read copies and bounded raw state pass chunk/speculation/rollback/prefix/restore tests (Sections 9–11); live migration is not an initial gate.
- Q8/RotorQuant Q4 layouts and SM70 kernels pass Section 13 gates; Signed128 results do not qualify RotorQuant.
- Complete worst-rank and host/pinned/replay ledgers support useful prefill/verification and the largest promised media unit at full residency (Sections 4, 7).
- Maximize qualified capacity, then near-full-history/full-active committed decode; preserve commitments at low load. Measure every integer activity count, thresholds/endpoints/breakpoints, without a 32-session ceiling (Sections 5, 17, 21). Section 5's >250 single-request / >600 aggregate-at-24 Q8 decode and >3,500 prefill objectives, six-point milestone A/B matrix, 1<2<4<8<16<24 priority and decode nonregression gate are explicitly evaluated; unmet objectives remain visible implementation gaps.
- Sudden load increases execute immediately from fixed ownership/prepared plans; cancellations, slot reuse and required state transfers are correct and bounded. Partial migration/union-map tests are mandatory only for an adopted Section 11 experiment.
- Sections 9/17 implement compatible prefix reuse and bounded FIFO slot handoff end to end through one page/state catalog. Standard OpenAI API identity/continuation and internal retry/concurrency checks, fresh/prefix-hit/queued-checkpoint paths and complete RAM/optional-storage restore, required-state protection, dirty-page reuse, cancellation, profile changes and cold/warm resource/performance gates pass Section 21.
- Every Section 12 state type commits/recovers correctly; padded rows cannot mutate sessions. Unsupported verification rejects before mutation; publication/cancellation follows canonical commit.
- No tuning container/worker/CUDA-context restart; normal load response selects prepared work/width/kernel/graph plans inside a fixed maximum envelope and ownership map. Graph-set/max-arena replacement remains preparation/explicit administration; unsupported live settings stay fixed or pending under Section 19.
- Structural changes preserve lifetime, quality, context, media fidelity and selected-method commitments (Section 19).
- All-rank readiness precedes replacement publication. Recoverable failure exposes pending/unavailable state and in-process recovery without duplicate full models (Section 19).
- Section 21 receipts distinguish observations, allocations, commitments, estimates and qualification; report useful text/visual prefill, committed decode, media work, latency, fairness and overhead separately.
- Required shared-Q8 QSA MTP passes Section 12's import/alias/norm, full-prefix text/visual conditioning and selected-KV gates. IncoAI DFlash2 independently passes Section 14's SC117/V100/TP8 feature, finite-range/precision, bounded-window and position gates with its actual ledger. Additional pairings require their own complete qualification; SAW follows core implementation.

## 25 Evidence and source register

### User-supplied deployment evidence

Baseline values and historical qualification descriptions come from the supplied 2026-10-25 04:16 UTC accounting receipt. Its files on the user's TensorFold machine were not directly inspected:

```text
/home/kklouzal/TensorFold/task-artifacts/v100-indexer-archive/production-gpu-history-smoke/stats-initial/result.json
/home/kklouzal/TensorFold/task-artifacts/v100-indexer-archive/report-facts/weights.json
/home/kklouzal/TensorFold/task-artifacts/v100-indexer-archive/report-facts/frontier-all-layouts.json
```

Unique-history capacity arithmetic derives from that receipt's geometry and Section 9's pinned compact-indexer construction. Proposed algorithms, policies, placement, and gates are implementation design, not observations of a modified deployment. Section 5 separately records the user's subsequent verification of >600 aggregate decode tokens/s at 24 active requests and >250 at one active request, both at 262K context/Q8 KV, plus >3,500 prefill tokens/s on this exact hardware. These supplied performance observations were not independently executed here and do not replace the current fork's matched benchmark receipts or full resource proof.

### Exact artifact inspection

SC117 IQ3_S shard and visual-file sizes/SHA256 values were verified from immutable public file pointers. Both selected SC117 binary tensor directories were subsequently inspected with bounded range reads, yielding the 1,224-tensor source census and payload/offset reconciliation summarized in Section 2 and the numerical companion. Full-file hashes, weight values and tokenizer/template behavior were not independently validated; numerical import, transformed allocations, executed kernels, target-specific memory and throughput remain installation/qualification gates. Source-header arithmetic is not a runtime receipt.

The 2026-10-29 review rechecked official/shared-Q8 MTP and Inco metadata/tensor headers, their stated counts/payload arithmetic and file-pointer identities; no full large target/MTP/draft payload was downloaded or executed. Both official processor JSON files were fully retrieved and hash-checked again, and the pinned processor/model/speculative source paths were inspected. Source contracts do not establish SC117 acceptance, multimodal quality or V100 performance. Full-file integrity, imported-value parity and runtime qualification remain installation gates.

### Pinned owned-fork implementation evidence

Only the `python-0.6` lineage at [`c83187878423388715571d194536eba836a64804`](https://github.com/kklouzal/TensorFold/tree/c83187878423388715571d194536eba836a64804) supplies current implementation evidence in this review. All other branches are excluded. Relevant current-source anchors are:

| Evidence surface | Pinned implementation or contract and claim boundary |
|---|---|
| SM70 foundation/status | [Source pins](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/deploy/v100/source-pins.json), [owned source foundation](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/deploy/v100/source-foundation.md) and [V100 validation](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/docs/v100-adaptive-validation.md): exact build/CPU milestones and explicitly unqualified physical/model/performance gates |
| Unique history and prepared execution | [Compact indexer](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/docs/compact-indexer.md), [profile manifest](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/docs/execution-profile-manifest.md), current family `history_placement.py`, `state.py`, `engine.py`, `multi.py`, `owner_execution.py` and `owner_graphs.py`: preserve actual authority/lifetime, preparation and evidence gates; align degree/reserve wiring to Section 10 |
| Shared serving/preparation | [Logical record store](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/src/tensorfold/server/api_records.py), [Conversations dispatch](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/src/tensorfold/server/conversations.py), [runtime status](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/src/tensorfold/server/runtime.py), [model identity](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/src/tensorfold/engine/model_identity.py), metadata/closure, request limits, exact sampling, media replay/work and retention owners: existing contracts to reuse/refactor, not blanket native qualification |
| Existing modern-device profiles | [README](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/README.md), [Gemma qualification](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/docs/gemma4-cuda-validation.md), [kernel contracts](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/docs/cuda-kernel-contracts.md), [modern foundation guide](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/deploy/gb10/README.md), `cuda/build.py`, `precision.py` and native NVFP4 providers: selected policies, explicit nonselection and source/image-specific evidence, separate from refreshed inputs |
| Direct source/package cutover | [Packaging](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/pyproject.toml), [native extension build](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/setup.py), [updater](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/src/tensorfold/update.py) and [maintainer workflow](https://github.com/kklouzal/TensorFold/blob/c83187878423388715571d194536eba836a64804/CONTRIBUTING.md): actual names/inputs/origins/callers to update coherently; historical notices/evidence remain unchanged |

The source cache was reconciled to the pinned Git tree. Recorded external/private evidence is used only at its published scope; private task packets and installed machines were not independently accessed. Source and exact-byte inspection do not establish execution, physical memory or global optimality. Outstanding hardware/compiler/installed-model/service gates remain visible in Sections 2,21–24.

### Primary external references

Sources are cited beside their requirements: platform, model, visual/processor, tokenizer, NVFP4 and dependencies in Section 2; compact QSA storage/selection in Section 9; MTP in Section 12; RotorQuant/SAW in Section 13; DFlash2 in Section 14; SM70 libraries in Section 15; multimodal execution in Section 16; FIFO/tiered-retention policy in Section 17; graph/collective contracts in Section 18. Preserve each source's stated scope; modern-GPU evidence does not qualify this V100 implementation.

Additional pinned implementation and background references:

| Reference | Use and boundary |
|---|---|
| [Transformers model implementation](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py) | Geometry, PLE indexing and normalization; preserve selected GGUF tokenizer/template and converted values |
| [SGLang QSA parser](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/attention/qsa/config.py) | Section 12's explicit QSA MTP policy; companion zero-ratio metadata does not establish dense/sparse equivalence |
| [SGLang GemmaRMSNorm](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/layernorm.py) and [GGUF norm conversion](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/conversion/qwen.py) | Exact MTP normalization axis/weights and single application of the converted Gemma offset; matching shapes alone do not establish numerical parity |
| [Original DFlash paper](https://arxiv.org/html/2602.06036v2) and [repository](https://github.com/z-lab/dflash) | Method background; distinguish DFlash 2 model changes from a serving engine named Spec V2 |
| [vLLM context parallelism](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/), [dynamic speculation](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/), and [NanoCP](https://arxiv.org/abs/2605.21100) | Related concepts, not drop-in qualification for ProviderOne, stock MTP, or this topology |

Pin the recorded artifact/reference revisions, final ProviderOne implementation, and full runtime stack in each receipt. Changes to sources, codecs, kernels, feature contracts, or numerical settings invalidate affected qualification until rechecked. Label every capacity/performance claim as supplied observation, arithmetic estimate, header plan, or live-qualified result.

This review inspects the owned `python-0.6` source at `c83187878423388715571d194536eba836a64804`, task-relevant callers/contracts/tests and build/deployment evidence, and checks arithmetic and cross-document consistency. It does not inspect a private dirty checkout, execute an installed image or run hardware/model/performance tests. Section 2’s source/status map and Section 23’s installation/resource/execution gates remain explicit. Source-reported historical measurements are not independently reproduced or transferred to this HEAD. The companion backlog supplies additional pinned evidence and conditional experiments without overriding those gates.
