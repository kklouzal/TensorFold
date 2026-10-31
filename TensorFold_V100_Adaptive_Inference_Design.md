# TensorFold V100 Adaptive Inference Design

**Platform:** Supermicro SYS-4029GP-TVRT; eight 32 GiB V100 SXM2 GPUs.  
**Initial target:** non-Swift `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF`, `IQ3_S`.  
**Required:** QSA MTP, separately selectable IncoAI DFlash2, image/video → text.  
**Scope:** implementation design for qualified family variants, GGUF/NVFP4 and compatible drafts.  
**Basis:** selected platform/model requirements and whole-document primary-source review dated 2026-10-26 UTC.  
**Evidence:** primary-source facts, artifact/header measurements, arithmetic estimates and V100 workload qualification are distinct (Section 25).

**Status:** finalized source-reviewed implementation specification, independent of implementation progress. Section 25 distinguishes verified source facts from installation and workload gates; no unmeasured concurrency, quality or throughput result is certified.

## 1 Governing operating contract

Use one engine for qualified Qwen3.8-Flash-Next variants. Administrators select target/runtime weights, context/RoPE, KV codec, speculative artifact/method and service limits; TensorFold computes full-context capacity and selects qualified workload plans. Initial profiles use SC117 IQ3_S, SC117 Q8 MTP with target embedding/head aliases, and IncoAI DFlash2. Additional artifacts use the same contracts.

**Mission:** maximize qualified GPU-resident sessions at the selected context, weight/KV quantizations, speculative method and media envelope, then aggregate committed decode tokens/s with all sessions active. Adapt execution to GPU-admitted activity while preserving context, quality, fairness and no-restart commitments. Bounded FIFO admission handles excess requests; qualified RAM/storage suspension preserves idle retained conversations (Section 17). Report the measured context/capacity/throughput tradeoff (Section 5); never silently sacrifice one objective for another.

### Settled requirements

| Area | Governing requirement |
|---|---|
| Hardware | Optimize for this server's measured NVLink/PCIe/NUMA and SM70 behavior (Section 2). |
| Target | Non-Swift SC117 IQ3_S; qualified mixed-format import; TP8 baseline with all eight ranks available at low load. Section 20 governs alternatives. |
| Residency | One distributed target, at most one speculative model; qualified no-speculation transition releases both speculative systems. No extra full-model or persistent weight/vision copies for throughput. Preserve unique histories; charge required rank-local parameters and bounded activation/communication/recovery/migration storage. |
| Context | Equal administrator-selected windows shared by expanded text/media, retained conversation and reasoning/reply. Native 262,144 rows; required RoPE/YaRN extension needs separate row/coordinate qualification (Section 6). |
| Capacity | Compute the complete worst-rank integer maximum; 20/24/32 and power-of-two counts are not fixed limits (Sections 7–8). |
| Speculation | Required/default SC117 Q8 QSA MTP with target embedding/head aliases; separately selectable IncoAI DFlash2; explicit no-speculation reference/admin profile cannot replace MTP implementation (Sections 12, 14). |
| KV | Specified H32/midpoint INT8 (Q8) or RotorQuant `iso4` K/V. No unspecified native RotorQuant Q8. RotorQuant first; SAW later (Section 13). |
| Histories | Unique attention KV/pooled indexer and bounded raw pending state in VRAM during execution. Qualified idle retained sessions may suspend to RAM/storage (Section 17); raw-history archiving stays off. |
| Multimodal | Required text/images/video → text, pinned vision/processor, bounded execution and separate MTP/DFlash qualification. Native audio/media generation excluded (Sections 2, 16). |
| Adaptation | Workload-driven batches, kernels, graphs, prefill budgets, proposals and justified attention placement; FIFO admission and qualified idle-session tier transfers (Sections 10–18). |
| Restarts | No container/worker/CUDA-context restart. Non-live settings stay at qualified maximum-concurrency values during adaptation or await a supported administrator in-process transition (Section 19). |
| Migration | Batch-boundary changes and bounded state-preserving migration retain every admitted sequence's lifetime/context commitment (Section 11). |
| Simplicity | Refactor one serving path directly; initially one homogeneous numerical/speculative resident profile. No simultaneous speculative systems, persistent GPU history replication or unnecessary framework layers; Section 17 bounds cross-tier checkpoint copies. |
| Evidence | Complete accounting and representative execution support capacity/performance claims; header arithmetic, startup allocation and live qualification remain distinct (Sections 21, 25). |

Section 7 reconciles the complete post-import allocation plan. Section 8 supplies geometry cross-checks; 20-Q8/32-Q4 are reference points, not admission promises or ceilings.

### Boundaries of the design

Preserve the target's sparse/indexed attention, recurrent equations, masks, pooling, positions, and sampling semantics while changing storage ownership and scheduling. Qualify quality-changing codecs separately.

No-restart applies to serving adaptation and supported administrator changes. Hardware/driver faults and software deployment have separate lifecycles; universal GPU/process fault recovery without restart is not promised.

Preserve the selected tokenizer and operators; declare every weight conversion and numerical choice in the effective profile. Pin the implementation revision and exact runtime build in each qualification receipt.

## 2 Fixed server and model profiles

### Physical platform

Eight V100 SXM2 GPUs use the X10DGO-SXMV-P carrier and X11DGO-T host board, a cube-style NVLink mesh, and switched PCIe 3.0 CPU-GPU paths. There is no NVSwitch. The advertised 300 GB/s NVLink bandwidth is aggregate bidirectional per GPU, not per pair or host transfer. Sources: [Supermicro specification](https://www.supermicro.com/en/products/system/4U/4029/SYS-4029GP-TVRT.php), [manual MNL-2115](https://www.supermicro.com/manuals/superserver/4U/MNL-2115.pdf), and [Red Hat/Supermicro reference architecture](https://www.redhat.com/rhdc/managed-files/pa-supermicro-deep-learning-openshift-reference-architecture-f19703-201910-en_0.pdf).

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

Select **R580.196.07 with the proprietary NVIDIA kernel module** as the verified Linux driver candidate on the audit date; choose and pin the supported host OS/kernel during deployment. NVIDIA's open kernel modules require Turing or newer and cannot drive V100. Record the actual host driver/kernel and container-injected driver libraries independently of the CUDA toolkit image. Sources: [R580.196.07 release](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-196-07/index.html) and [kernel-module support](https://download.nvidia.com/XFree86/Linux-x86_64/580.196.07/README/kernel_open.html).

Volta Tensor Cores use FP16 matrix operands with FP16 or FP32 accumulation. Packed Q4/Q8 needs an explicitly supported compute path; do not assume newer GPUs' native INT4, INT8, FP8, or BF16 Tensor Core paths. Source: [Volta tuning guide](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html).

Use Volta-capable profilers and verify every dependency/kernel backend. FlashAttention or modern FA3/FA4/Triton recipes do not establish SM70 support for this sparse, rotated, quantized attention path.

Release targets rechecked on 2026-10-26 follow. Deployment requires Section 21's combined-stack qualification. Resolve newer eligible patches for subsequent builds.

| Component | Selected release target | SM70 condition |
|---|---|---|
| CUDA Toolkit | 12.9.2, Update 2 | Newest published 12.x toolkit; build native `sm_70` objects with its supported host compiler |
| cuBLAS and cuBLASLt | 12.9.2.10 | Newest published 12.x library candidate; qualify actual FP16/FP32 algorithms, layouts, and workspace bounds |
| CUTLASS C++ | 4.8.1 | Use retained SM70 C++ TensorOp/WMMA components; the name of its 2.x kernel API does not require a 2.x package. |
| NCCL | 2.32.3, release tag `v2.32.3-1` | Released CUDA 12.x candidate; verify the exact release build's SM70 objects and V100 collective/graph execution |
| CCCL, CUB, Thrust, libcu++ | 3.5.0 required-primitives SM70 build candidate | CUDA 12.x compilation remains supported; SM70 is outside regular upstream validation, so qualify every required primitive and the coherent framework build |
| cuDNN, when required by the framework or a selected operator | 9.10.2 for CUDA 12.x | Its matrix supports Volta with CUDA 12.9; 9.11.0 removes Volta |
| Nsight Compute | 2025.2.1, build 2025.2.1.3 | 2025.3 removes Volta |
| Nsight Systems | 2025.3.2, build 2025.3.2.474 | 2025.4 and later remove Volta |
| Compute Sanitizer | 2025.2.1 release, CUDA package 12.9.79 | 2025.3 removes Volta |
| PyTorch execution framework | 2.14.1 source-build candidate against the selected CUDA 12.9 stack | Its published Volta wheel uses CUDA 12.6; inspect source-build and transitive dependency compatibility instead of assuming the wheel supplies this stack |

Toolkit/library versions: [CUDA 12.9.2 component manifest](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-toolkit-release-notes/index.html), [cuBLAS patch releases](https://docs.nvidia.com/cuda/cublas-patch-release-notes/), [CUTLASS releases and compatibility](https://github.com/NVIDIA/cutlass), [NCCL release](https://github.com/NVIDIA/nccl/releases/tag/v2.32.3-1), [NCCL build rules](https://github.com/NVIDIA/nccl/blob/12df1a11afad322be5a204a2db890161cbf8131d/makefiles/common.mk), [CCCL 3.5.0 release](https://github.com/NVIDIA/cccl/releases/tag/v3.5.0) and [exact-release architecture policy](https://github.com/NVIDIA/cccl/blob/v3.5.0/README.md), [cuDNN support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html) and [9.11 removal](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.11.0/release-notes.html). The selected NCCL release's build rules include SM70 under CUDA 12.x; source configuration does not replace inspection of the actual built binaries.

CCCL 3.x omits older-toolkit-only architectures from regular testing without disabling every path. Qualify 3.5.0 scan/sort/compaction/reduction and warp/block primitives for numerics/selection, graphs/streams, bounded scratch and whole-step performance; resolve vendored headers coherently. If it fails, select the newest earlier passing release. CCCL 2.8.5 is a comparison baseline, not a ceiling or LTS claim. Pin one deployment release; no serving-time library fallback. Sources: [CCCL 2.8.5](https://github.com/NVIDIA/cccl/releases/tag/v2.8.5); [issue 6226](https://github.com/NVIDIA/cccl/issues/6226); [clarification 6229](https://github.com/NVIDIA/cccl/pull/6229).

Tools/framework evidence: [Nsight Compute history](https://developer.nvidia.com/nsight-compute-history) and [2025.3 support boundary](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_3), [Nsight Systems boundary](https://docs.nvidia.com/nsight-systems/ReleaseNotes/), [Sanitizer history](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html), [NVIDIA published package builds](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/), [PyTorch 2.14.1](https://github.com/pytorch/pytorch/releases/tag/v2.14.1), and [PyTorch CUDA 12 packaging policy](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432). A package index identifies a build; it does not select the server's OS. Source builds must resolve framework-vendored headers and transitive libraries coherently, rather than overriding individual libraries inside an incompatible wheel.

Dependencies are immutable during serving. Replacements follow Section 1’s deployment lifecycle and invalidate affected algorithm, graph, numerical, resource and performance qualification; adaptation selects qualified plans.

Audit actual loaded builds and direct consumers, including framework-private or independently vendored CUTLASS, CCCL and NCCL. Resolve them coherently and retain their exact identities. A framework’s Triton dependency does not establish an SM70 execution path. Software-only review cannot clear device, model, resource, quality or performance gates.

### CPU, NUMA and transport placement

Initially retain one worker per GPU and one distributed coordinator; other process models require qualification. Record physical cores/SMT siblings, allowed CPU/memory cpusets, GPU PCI bus/NUMA affinity, memory channels and active NIC/NVMe paths. Bind launch/progress, lookup and media workers deliberately; reserve CPU service for communication, tokenization and output instead of letting eight workers each consume every core. Set framework/OpenMP/BLAS/media thread budgets before worker initialization; measure SMT, spinning and blocking policies rather than assuming more threads help.

Set NUMA task policy in each allocating thread or before creating workers/threads; existing sibling threads retain their policy. Apply it before allocating/first-touching anonymous staging/work arenas and bounded pinned pools near the intended GPU path. Verify placement with `numastat` and `/proc/PID/numa_maps`; CPU binding does not relocate pages. Regular file-backed PLE page-cache placement differs from anonymous VMA policy. Initially retain one shared mapped table; measure cold/warm placement, remote traffic, lookup ownership and common-result distribution without extra GPU weight copies. Measure automatic NUMA balancing and huge-page policies. Pin an entire evictable mapping only with demonstrated benefit and sufficient host budget.

Verify peer access and measured transfer/collective paths inside the actual container, not only on the host. Check cpuset/NUMA permissions, shared-memory/locked-memory limits and topology visibility. Investigate unexpected host staging or poor P2P, including relevant ACS/IOMMU settings, against NVIDIA's platform requirements; topology labels alone do not prove correct direct transfers. Record approved host configuration changes and requalify, rather than blindly applying system-wide tuning. Multi-node GPUDirect RDMA remains a separately qualified extension.

Use ordinary bounded host I/O and explicit host/device staging for storage on this V100 baseline. GDS 1.7.2 and later do not support the proprietary kernel module required by V100; GDS 1.15 additionally removes Volta explicitly. Installing cuFile from the selected toolkit therefore does not establish a supported direct storage-to-V100 DMA path. Any compatibility-mode CPU path must be identified and measured as such. Source: [NVIDIA GDS release/support history](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html).

Sources: [Linux NUMA memory policy](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html), [PyTorch CPU/NUMA tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html), [CUDA NUMA guidance](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html#numa-best-practices), and [NCCL transport diagnostics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html).

### Model profile contract

Validate each target/speculative manifest using family-specific operators and one scheduler, allocator, planner and serving path across publishers/encodings. Initial artifact values do not establish another artifact's geometry, numerics, tokenizer or compatibility.

| Manifest field | Required content |
|---|---|
| Source identity | Target/MTP/DFlash and visual repositories, immutable revisions, files/hashes, actual encodings/quantization metadata, tokenizer/template, and matching processor/decoder identity |
| Model geometry | Layer types, HC streams, attention/indexer dimensions, QSA, recurrence, experts, PLE, vocabulary/context, visual geometry/token expansion, and target/draft position rules |
| Runtime representation | Per-tensor retained format, scale layout, sharding/replication, conversion or repacking recipe, compute and accumulation precision, activation-quantization policy, and exact allocation sizes |
| Speculative pairing | Embedded or separately supplied MTP; or draft architecture, feature taps/transforms, embedding/head ownership, tokenizer alignment, masks, position/window rules, selector/proposal probabilities, and commit/recovery convention |
| Qualification identity | Source/conversion hashes, kernel/runtime build, selected target/draft KV formats, capacity ledger, and measured execution profiles for this numerical pairing |

Before device allocation, validate tensor names/shapes/transforms/operators and derive planner dimensions from the manifest. Specialize kernels without publisher/filename hardcoding. Unsupported geometry/operators require implementation before admission; the two-head placement proof and 12/13-cache formulas apply only to matching geometry.

Keep container/source encoding, runtime weights, compute precision and target/draft KV independent. GGUF is a container, NVFP4 a numerical representation; inspect per-tensor formats, not repository suffixes. Weight quantization does not select KV.

### Model assets and configuration resolution

Before allocation or admission, resolve each target, MTP, DFlash and visual role completely through the existing manifest/preparation path. Discover supported companion files automatically; equivalent embedded GGUF metadata is valid. Missing filenames are acceptable only when their required semantics resolve elsewhere. An MTP-only file’s logical block count must not instantiate another target or imply absent trunk tensors.

| Asset or embedded equivalent | Required information |
|---|---|
| `config.json` and applicable quantization configuration | Registered model family, actual layer/operator geometry, norms/activations, positions, quantization/transforms, shared tensors, and role-specific MTP/draft/vision parameters |
| `tokenizer.json`, `tokenizer.model`, or native vocabulary/merge files | Complete token-to-ID mapping and encoding/decoding pipeline: normalization, pretokenization, model rules, added tokens and postprocessing |
| `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json` where applicable | Backend/class settings, special-token roles, insertion flags, cleanup/padding behavior and any supported embedded template; unused model-vocabulary rows remain distinct from lexical tokens |
| `chat_template.jinja`, supported named templates, or GGUF `tokenizer.chat_template` | Exact message/tool/reasoning/media serialization and generation-prefix behavior; select one explicit target template |
| `generation_config.json` or supported embedded generation fields | Source generation defaults and terminators, kept separate from explicit service defaults and request overrides |
| `processor_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json` and pinned processor/decoder implementation | Actual image/video normalization, resizing, patch/merge expansion, frame sampling, timestamps, padding and coordinate rules |
| Tensor index, complete container directories and pinned conversion/operator definitions | Actual tensor availability, shapes/types, shard coverage, representation transforms and executable operator semantics; descriptive configuration alone cannot establish these |

The target owns one text frontend; shared-target MTP/DFlash consume its IDs without separate tokenizer/template files. Vision resolves its processor/coordinates, while companions retain their own operator, feature, mask, position and window contracts. Family names or media tokens do not prove required tensors/operators exist.

Resolve metadata once during profile preparation:

1. Bind each component to exact repository/revision/file identities, inspect its complete tensor inventory, and read its available metadata and companion files. Cross-check configuration against physical tensor facts and supported family operators. Selectors, aliases and source-to-runtime transforms are part of this inventory.
2. Apply the selected format and adapter's field- and role-specific precedence. The target's selected template/token IDs govern its frontend; a companion's or base model's template, EOS or sampling fields cannot overwrite them. Resolve nested text/vision/generation fields by their documented roles, rather than flattening all dictionaries or taking whichever value loads last. Equivalent representations may be reconciled only through the declared conversion rules.
3. Fill missing information from an explicitly bound, compatible source revision or an unambiguous supported derivation. Tensor dimensions can establish dimensions; they cannot establish an absent normalization epsilon, tokenizer normalization program, chat template, RoPE rule, feature boundary or video policy. Borrow only validated fields from an upstream model and preserve derivative-specific values.
4. Expose role-qualified metadata asset locations and typed field overrides through the same administrator model profile. Accept local files or pinned repository/revision/file references for config, tokenizer, template, generation and processor assets. Validate every explicit value against the field schema and tensor/operator/token constraints; report unknown fields, missing fields and unresolved same-role conflicts before allocation. An override cannot change a file's physical geometry or supply an unimplemented operator. A missing vocabulary or tokenizer program requires the actual assets or a complete supported specification; a vocabulary-size integer is insufficient. If a required semantic remains unresolved, the profile is unavailable with a precise diagnostic.
5. Freeze the effective values together with each source field, revision/hash, adapter identity, derivation and explicit override in the existing manifest. Prepare required assets locally before serving. Extract embedded templates verbatim when a consumer needs a Jinja file; generate other sidecars only when their complete schema and behavior are representable, and identify them as derived outputs. Do not fabricate an original publisher file or require a conversion merely to satisfy a filename convention.

Token, rendering, position, processor or operator changes invalidate dependent prefix/checkpoint/pairing identities under Section 19. Verify encoding/decoding, token roles, tools/reasoning/media rendering and preprocessing against pinned references. A large tokenizer `model_max_length` sentinel grants no context. Source defaults do not choose kernels, TP placement, batches or capacity; Sections 5–7 and 17 select measured execution without changing model semantics.

Sources: [GGUF metadata specification](https://github.com/ggml-org/ggml/blob/da8e2a54b919557b816b1c2beb432ccb766f7ce0/docs/gguf.md), [configuration API](https://huggingface.co/docs/transformers/main/en/main_classes/configuration), [tokenizer pipeline](https://huggingface.co/docs/tokenizers/main/en/pipeline), [tokenizer API](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer), [template storage](https://huggingface.co/docs/transformers/en/chat_templating_writing), [generation configuration](https://huggingface.co/docs/transformers/en/main_classes/text_generation), and [processor API](https://huggingface.co/docs/transformers/main/en/main_classes/processors). These describe metadata responsibilities; execution still uses the profile's pinned, qualified family/backend implementation.

### Initial metadata bindings

The pinned SC117 and Unsloth GGUF repositories do not supply standalone `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, generation or processor sidecars. Resolve their embedded metadata and bind the following supplementary sources. The official source supplies these files individually; its absence of `processor_config.json` is harmless when the processor class and complete component settings resolve through the other pinned files/code.

| Component | Authoritative input and completion rule |
|---|---|
| SC117 target | Inspect the selected GGUF geometry, formats, tokenizer rows and embedded template. Use the exact target template recorded below. Bind missing tokenizer/operator semantics to the verified official source fields and qualified adapter; never replace the target template with the base template. |
| SC117 Q8 MTP | Inspect Section 12's companion directory and its explicit QSA metadata; import only the 32 MTP-core tensors and bind the actual target embedding/head. Reuse the target frontend. The companion template is descriptive source metadata and is not executed. |
| Official Qwen semantic/control source | At `de4b8e4d43b917e7706784d8bb445c9af86a3540`, `config.json` is 4,745 bytes, `generation_config.json` 202, `chat_template.jinja` 8,952, `tokenizer.json` 12,809,320 and `tokenizer_config.json` 17,928; `vocab.json`, `merges.txt`, the tensor index and processor files are also present. Use validated fields and operator definitions for completeness and BF16 control. The base template is not the selected target template. |
| Inco DFlash2 | Its pinned repository supplies `config.json` (1,236 bytes), weights and capture patch, without separate tokenizer/template/processor files. Use its own draft configuration and feature contract, plus the target frontend and shared tensors. Its draft EOS/theta/block fields do not replace target generation/position settings or the qualified serving block size. |
| Visual encoder/projector | Combine the SC117 visual GGUF with the official 390-byte image and 385-byte video processor configurations, pinned processor/decoder code, and the target's media-token/position contract. Resolve these inputs before advertising image/video capability. |

Sources: exact [SC117 file inventory](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/tree/90a3350cd800ec1fe0d1660a38414ad9e985782f), [official Qwen assets](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540), [Unsloth companion inventory](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6/MTP), and [Inco assets](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/tree/a22a1744e85c17364dcc54a6a420af7135d4c7f6).

### NVFP4 sources on V100

Import NVFP4 through an explicit SM70 representation. E2M1 values use two-level scaling, commonly an E4M3 scale per 16 values and an FP32 tensor scale. Exporter layouts/recipes vary: decode the actual scale tensors, orientations, transforms, padding, and mixed-precision exclusions. Sources: [NVIDIA NVFP4 format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) and [scaling/recipe documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html).

| Qualified runtime representation | Implementation and accounting |
|---|---|
| Retained packed NVFP4 values/scales with software tile decode | Custom SM70 kernels decode bounded weight tiles into supported FP16 operands, with sensitive scale arithmetic/accumulation as required. Charge packing, scales, tiles, conversion work and actual per-rank retention. This preserves compact storage without claiming native NVFP4 matrix throughput. |
| Converted supported packed weight format | Stream through bounded import buffers and materialize the chosen SM70 format. Repacking that preserves decoded values is distinct from requantization; requantization changes the numerical profile and needs separate quality, acceptance, memory and performance qualification. |
| Resident FP16 for selected tensors | Permit only where the full retained and peak allocation plan fits and measurement justifies it. Do not silently expand an entire target because its source encoding lacks a native instruction. |

Initially retain packed NVFP4 weights and decode bounded tiles into the qualified FP16 path. Select another qualified representation only when measured throughput and quality/resource constraints justify it, using the same engine's loader/packing and kernel choices. Source file size is not a VRAM budget; native-Blackwell benchmarks do not predict V100 performance.

Inspect activation quantization, calibration transforms, and QAT computation. Weight decoding alone does not reproduce native W4A4. Declare activation/accumulation policy and separately qualify departures from the source recipe. Conversion cannot restore lost quantization information; validate finite range and reconstruction error before serving.

Prepare/hash converted or repacked artifacts once per source/recipe with bounded host/device memory; reuse them, never convert on admission. Budget concurrent preparation. Record source/derived identities and reject unsupported/infeasible choices without substituting model, precision or offload policy.

### Initial target artifact

Use non-Swift `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF` revision `90a3350cd800ec1fe0d1660a38414ad9e985782f`, directory `IQ3_S`. Load both files prefixed `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-`. Sources: [pinned directory](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/tree/90a3350cd800ec1fe0d1660a38414ad9e985782f/IQ3_S) and immutable [first](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/raw/90a3350cd800ec1fe0d1660a38414ad9e985782f/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf)/[second](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/raw/90a3350cd800ec1fe0d1660a38414ad9e985782f/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf) file pointers.

| Shard suffix | Exact file bytes | SHA256 |
|---|---:|---|
| `00001-of-00002.gguf` | 55,063,448,064 | `40f369eca301ec6ea318dbb671b3132c26a2f0b89aa5458a7051b33215ee296c` |
| `00002-of-00002.gguf` | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| Total | 83,863,586,496 | 78.104051 GiB of files |

The complete binary directories confirm 1,223 transformer tensors in the first shard and one IQ4_NL PLE/ngram tensor in the second. Their file sizes are 51.281832 and 26.822219 GiB, respectively; separate vision and speculative artifacts are excluded. File size is not GPU residency. Identify PLE by tensor name/geometry, never a general shard-number rule.

Bounded binary-prefix inspection parsed both complete GGUF v3 directories and verified split indices/counts, 1,224 unique tensor names, block/shape arithmetic, aligned offsets, non-overlapping extents and exact final file boundaries. Shard 1's tensor payload starts at byte 11,024,384 and occupies 55,052,423,680 bytes. Shard 2 starts its payload at byte 192 and contains `per_layer_token_embd.weight`, GGML dimensions `[160, 320001536]`, occupying 28,800,138,240 bytes. These are header-derived source sizes. The SHA256 values above are the publisher's immutable full-file identities; this audit did not download and independently hash the complete large payloads.

At installation, repeat directory validation, verify complete-file hashes, and generate the exact source-to-runtime inventory before allocation. Validate codebooks/scales and decoded numerical values while importing; correct directory extents alone do not prove correct payload contents or conversion.

### Required target geometry

The initial implementation targets the following base-family geometry from `Qwen/Qwen3.8-Flash-Next` revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`; corresponding SC117 binary metadata and tensor dimensions were independently cross-checked. Assert the complete manifest again during import, including fields resolved from pinned operator code; publisher lineage alone cannot validate every field. Source: [pinned base configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json).

| Property | Required value or interpretation |
|---|---|
| Architecture | GGUF `qwen4exp`; base configuration `qwen4_exp` |
| Target depth | 48 blocks: 36 Gated DeltaNet/recurrent blocks and 12 QSA blocks, every fourth block |
| Hidden and residual geometry | Hidden width 2,560; four hyper-connection streams; low-rank width 320 |
| Target attention | 24 query heads; two global KV heads; head dimension 256 |
| QSA indexer | Four query heads; one history head of dimension 128; compression ratio 4; 512 selected complete four-token groups plus up to three visible tail tokens |
| Recurrent geometry | 16 key heads and 48 value heads, each dimension 128; convolution width 4; FP32 state |
| MoE | 512 routed experts, ten selected per token, plus one sigmoid-gated shared expert; routed and shared intermediate width 640 |
| Context | Native configured allowance 262,144; extended allowances use Section 6's qualified RoPE/YaRN profiles. Prompt and generated output share the selected allowance. |
| Positional encoding | Target theta 10,000,000; 64 rotary dimensions out of each 256-dimensional head; preserve the supplied mRoPE conventions |
| Vocabulary | 248,320 entries; target embedding and output head are distinct tensors |
| PLE | N-gram size 3; depthwise causal convolution width 4, dilation 3, nine prior activation rows; base config layer identifier 2 is one-based and corresponds to GGUF `blk.1` |

Preserve all four residual streams within target computation; do not average them or treat them as independent requests. DFlash's defined reduction applies separately at specified completed blocks.

### Mixed weight import

`IQ3_S` is a learned mixed allocation. All thirteen formats in the publisher's list occur in the inspected binary directories. The complete source census across both shards is:

| Source format | Tensor count | Tensor payload bytes |
|---|---:|---:|
| BF16 | 484 | 1,527,889,920 |
| F32 | 292 | 10,063,360 |
| F16 | 1 | 81,920 |
| Q2_0 | 9 | 2,123,366,400 |
| Q4_K | 43 | 163,307,520 |
| Q5_K | 31 | 303,677,440 |
| Q6_K | 89 | 1,361,740,800 |
| Q8_0 | 96 | 885,719,040 |
| IQ2_S | 40 | 10,747,904,000 |
| IQ3_XXS | 34 | 10,918,297,600 |
| IQ3_S | 20 | 7,208,960,000 |
| IQ4_NL | 40 | 47,202,647,040 |
| IQ4_XS | 45 | 1,398,906,880 |
| **Total** | **1,224** | **83,852,561,920** |

IQ4_NL includes the single host-mapped PLE tensor. These source payloads exclude headers and are not a GPU allocation plan. Repeat the census for any changed artifact; the quantization directory name or model-card histogram cannot substitute for it. Sources: the pinned binary artifacts above, [SC117's method description](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/blob/90a3350cd800ec1fe0d1660a38414ad9e985782f/README.md), and the [pinned GGUF type/block schema](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/gguf-py/gguf/constants.py).

Preserve the selected learned values, scales and codebooks through layout repacking. Use artifact-specific packed offsets and hashes. An unsupported format must not trigger uniform requantization, a different publisher's pack, or whole-model FP16 expansion. Any changed numerical representation is a separately selected and qualified profile.

Inventory every retained/decoded/replicated/sharded/aliased tensor per rank, including speculation and vision. Bind Section 15's decay/head-order transforms; identify the host-mapped PLE tensor explicitly. Qualify BF16→FP16 range, sensitive accumulation and indexer selection stability. The source payload census is not a runtime allocation plan.

### Tokenization and positions

Render/tokenize canonical input once through the target’s tokenizer, template and media path. Target, MTP and DFlash consume that stream with aligned positions, features and text/visual embeddings. Companion templates are unused provenance: no second rendering, translation, retokenization or equal-template-hash requirement. Hash the effective frontend and preserve reasoning/tool/media formatting. Count rendered/expanded input, retained conversation, special tokens and generated reasoning/output against context/reply limits.

The inspected target has tokenizer model/pretokenizer `gpt2`/`qwen35`, 248,320 token entries and 247,587 merges. Its 9,993-byte UTF-8 chat template has SHA256 `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`. The selected SC117 Q8 MTP and the inspected Unsloth shared companions have identical token/type/merge arrays and the same mapped BOS/EOS/padding IDs. Their packaged template is 8,952 bytes with SHA256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`, identical to the official base template. Preserve the SC117 target template's developer/system handling and reasoning-effort mapping; MTP loading, enabling or disabling cannot change the rendered target input, generation prefix or stop policy. Normalize standard API tool arguments to the structured mapping expected by this one target template.

Bind the target tokenizer's full pipeline explicitly: the pinned official `tokenizer.json` supplies NFC normalization, the Unicode-mark-aware Qwen35 splitting expression, ByteLevel processing and decoder, and no automatic BOS/EOS insertion. Disable additional tokenization-space cleanup. `qwen35` alone is insufficient to establish these semantics. The selected target GGUF omits the NFC key; the selected SC117 Q8 companion explicitly carries `tokenizer.ggml.normalizer.nfc=true` and `add_eos_token=false`. Use the verified official normalization/pipeline to complete the target frontend, rather than inheriting a generic loader default. In the pinned llama.cpp reference, the BPE path does not apply NFC merely because the pretokenizer is Qwen35; an alternative backend must pass explicit normalization fixtures.

The official tokenizer supplies 248,044 ordinary BPE tokens, 33 added-token entries and 247,587 ordered merges matching the target's corresponding strings/IDs. The initial frontend recognizes all 33 added-token literals at their exact IDs in canonical rendered input. Preserve the target's CONTROL classification, including FIM/repository IDs 248060–248065, although those six have `special=false` in the official tokenizer. Full internal decoding retains CONTROL spellings; skip-special presentation omits them. Keep the IDs in transcript/state, accounting and applicable protocol parsing; presentation filtering cannot delete model bookkeeping. Preserve USER_DEFINED tool/reasoning markers for their parser. The official tokenizer's null BOS declaration does not erase the target's BOS identity or enable automatic insertion.

The target's 243 UNUSED rows, IDs 248077–248319, remain nonlexical matrix rows: never register their synthetic `[PADnnn]` labels as input literals. They decode to zero visible bytes in full and skip-special modes; DFlash supplies mask ID 248077 internally by ID. A generated UNUSED row still counts toward token/state and reply/context budgets. Recognition, visibility and generation termination are separate policies: token type alone introduces no stop rule or logit suppression. Any suppression is an explicit, separately qualified target sampling setting.

Sources: pinned official [tokenizer pipeline](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer.json), [tokenizer configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer_config.json), the inspected target/companion metadata, [pinned GGUF tokenizer implementation](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/src/llama-vocab.cpp), and [MTP's token/feature input interface](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/models/qwen4_exp_mtp.py). Host fixtures confirmed the official pipeline and a target-GGUF reconstruction with these explicit settings agree for the inspected text/Unicode/chat/tool/media inputs; Section 21 still qualifies the actual serving backend.

Bind the initial token roles independently:

| Role | Initial policy |
|---|---|
| Generation termination | Explicit set `{248046, 248044}`, adopting the pinned official generation configuration as the serving default; the target GGUF's scalar EOS identity remains 248046 (`im_end`). |
| BOS/padding identity | 248044 (`endoftext`), with automatic BOS/EOS insertion disabled. |
| PLE n-gram segmentation and token-history padding | 248044, from `qwen4exp.ple.eos_token_id` and the pinned text configuration. ID 248046 alone does not reset n-gram history; generation terminators do not redefine this role or add recurrent/convolution resets. |
| Internal DFlash mask | 248077, supplied by ID; not a lexical input token, visible output or generation terminator. |

Do not inherit backend-added FIM pad/repository/file-separator stops at IDs 248063–248065. Any different generation-terminator set requires an explicit qualified profile; enabling MTP or DFlash cannot change it. QSA's 2,048-token complete-group budget can require 2,051 positions with its visible incomplete tail. Sources: inspected target metadata, [base generation configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/generation_config.json), pinned base configuration/model operators, and the tokenizer implementation above.

### Multimodal profile

Image and video understanding with text output is required. At the same pinned SC117 revision, use `mmproj-Qwen3.8-Flash-Next-BF16.gguf`: **907,542,944 bytes**, SHA256 `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355`. It contains the encoder/projector; its 0.845215 GiB file size is not a per-GPU allocation. Source: [immutable visual file pointer](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/raw/90a3350cd800ec1fe0d1660a38414ad9e985782f/mmproj-Qwen3.8-Flash-Next-BF16.gguf).

Pin its tensor inventory, matching official processor configuration/code, selected template, and FP16 SM70 conversion recipe. Validate base geometry: 27 vision layers, width 1,152, 16 heads, projected width 2,560, spatial patch 16, merge 2, temporal patch 2; image/video IDs 248056/248057 and vision delimiters 248053/248054. Preserve interleaved M-RoPE sections `[11,11,10]` and continuation deltas. The pinned [base configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) and [target/vision implementation](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py) define reference semantics; Section 16 specifies bounded execution and scheduling. Exact SC117 multimodal quality and V100 performance require qualification.

The inspected visual directory contains 334 tensors: 224 F32 tensors (19,321,792 payload bytes) and 110 BF16 tensors (888,201,216 bytes), totaling 907,523,008 payload bytes from aligned offset 19,936. The `BF16` filename does not describe every tensor. Preserve the 48×48 learned position grid with bilinear `align_corners=True` interpolation, LayerNorm epsilon 1e-6, tanh-approximate GELU in transformer MLPs and exact GELU in the final merger. Transformer MLP intermediate width is 4,304. The merger normalizes each 1,152-wide row before 2×2 concatenation, then applies 4,608→4,608→2,560. The two distinct F32 temporal patch kernels join on Conv3D's temporal axis with one bias. Vision axial RoPE is independent of language-model YaRN; no deepstack layers are configured. Use these pinned operator semantics for import and encoding, subject to numerical and full-pipeline qualification.

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

Use Section 12’s SC117 Q8 companion. The target has 48 blocks and no `nextn`/MTP metadata or MTP/`blk.48` tensors. Validate actual coverage for every artifact; declarations alone cannot establish tensors or authorize duplicate imports. MTP and Section 14’s DFlash share actual target vocabulary tensors. SC117 describes stock Qwen MTP and abliteration confined to the 48 trunk layers, not an MTP refit.

### License and redistribution provenance

Keep a pinned license/notice manifest for TensorFold, imported implementation code, and target/MTP/draft/vision assets. The exact [Qwen base LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE) is Qwen Community License 1.0 (3,235 bytes; SHA256 `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08`). The selected MTP and vision GGUF metadata also name that license. SC117's pinned model card claims Apache-2.0 inherited from the base, which conflicts with this primary license evidence; Inco's pinned draft card separately declares Apache-2.0. Resolve the derivative metadata conflict before redistributing bundled/converted model assets or describing them as a uniformly Apache-2.0 distribution. Preserve upstream notices and provenance through conversion; a software image or TensorFold code license does not relicense weights.

For imported ISO4 code, retain the notices from the actual [pinned implementation LICENSE](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/LICENSE), which supplies an MIT grant. Record other copied sources independently in the implementation's third-party notices. These are provenance requirements; source metadata inspection does not determine every deployment's license obligations.

## 3 Evidence and accounting baseline

The platform provides eight nominal 32-GiB GPUs, or 256 GiB in aggregate. Capacity is constrained by the limiting rank, actual device availability and complete overlapping obligations; aggregate nominal VRAM is not a shared allocation pool. Record measured totals instead of rounding a device's usable memory up to its marketed capacity. Throughout this document, GiB and MiB mean `2^30` and `2^20` bytes; advertised interconnect rates retain their stated decimal GB/s units.

### Evidence classes

| Class | What it establishes | What it does not establish |
|---|---|---|
| Primary specification/source | Hardware/software support boundaries, model/operator contracts and declared artifact identity | The installed machine or executable actually follows the contract |
| Complete tensor-directory inspection | Tensor names, shapes, encodings, offsets and header-derived source payload sizes | Correct complete-file payloads, decoded values or GPU allocation peaks |
| Analytical derivation | Memory and transfer counts under explicitly stated geometry/layout assumptions | Runtime capacity, latency, model quality or throughput |
| Installation verification | Checked full hashes, imported representations, loaded libraries and measured allocations | Every workload's correctness, resource peak or useful performance |
| Workload qualification | The declared build/profile satisfies the tested correctness, quality, lifecycle and performance envelope | Larger contexts, other codecs/artifacts/media bounds or untested execution regimes |

### Required physical baseline

Before solving capacity, establish a timestamped per-rank baseline with device totals/free memory, process and allocator residency, initialized CUDA/library/communicator costs, and a manifest of project allocations already materialized. Distinguish allocator-allocated bytes, allocator-reserved bytes, cached unused blocks, external allocations and physical free memory. Allocated bytes are included in reserved bytes; never add the two.

Take the baseline at a defined loading/preparation phase and record which future costs remain. Section 4 defines the incremental serving budget and protective headroom. Section 7 charges the remaining complete-window promises and execution envelope. Validate loading/conversion peaks separately from steady serving, and initialize or conservatively bound lazy library/graph residency before capacity publication.

One history copy means one physical copy of each required logical KV head and completed pooled indexer entry under Sections 9–10. A replicated comparison implementation has a different ledger. Section 8 derives the intended unique-history rates from the selected codec geometry; those rates become an admission basis only after compact pending-state storage and the exact ownership/stride plan are implemented and verified.

Keep arithmetic and measurements separate. Qualification does not transfer across checkpoints, codecs, layouts, toolchains or media envelopes; deployment maxima require Section 21’s evidence.

## 4 Accounting rules and complete resource ledger

Require at least 4 GiB/GPU of physical free memory and a separate serving-transient allowance of at least 2 GiB/GPU. Derive recovery, replacement and graph obligations from actual state and permitted overlap; smaller checkpoints cannot weaken these protections.

Bind each budget to one timestamped allocation baseline. Start from physical free bytes reported by `cudaMemGetInfo`, subtract the required device floor, then apply any administrator grant cap. Reject an effective floor below 4 GiB in this profile. For incremental planning:

```text
usable_serving_budget[r] = min(
    max(0, baseline_physical_free[r] - required_device_floor[r]),
    remaining_absolute_grant_cap[r])
```

Omit the cap term when unset. Subtract already granted project allocations from the remaining cap as required; do not subtract them again from a physical-free sample that already excludes them. Section 7's left side includes only costs not already accounted for in that baseline. Report materialized project bytes separately to reconstruct total occupancy, and record baseline external/runtime residency without charging it twice.

The full-window geometry separately reserves serving-transient headroom of **at least 2 GiB/rank after its charged fixed execution arenas**, increasing it to the actual unmaterialized concurrent transient bound when larger. Use `max(transient_requirement, minimum_workspace_bytes)`; the 2-GiB promise cannot be credited merely because fixed expert or graph arenas occupy 2 GiB. Account for genuine arena reuse once and for independent overlapping work separately. Import/loading and steady-serving peaks are distinct lifecycle maxima unless their overlap is permitted. Sources: [capacity baseline and cap](https://github.com/kklouzal/TensorFold/blob/600b0eaa0c59d35154c1be8e0c0a8570a9d8291e/src/tensorfold/cuda/capacity.py), [full-window geometry](https://github.com/kklouzal/TensorFold/blob/600b0eaa0c59d35154c1be8e0c0a8570a9d8291e/src/tensorfold/cuda/geometry.py), and [remaining-service promises](https://github.com/kklouzal/TensorFold/blob/600b0eaa0c59d35154c1be8e0c0a8570a9d8291e/src/tensorfold/families/qwen4_exp/cuda/service_memory.py).

Charge each allocation/promise once. Unmaterialized promises are not startup occupancy; included floors are not deducted again.

Use checked 64-bit arithmetic for source-file offsets, allocation sizes, byte strides and page/arena address products; widen operands before multiplication or addition. Narrow local indices remain appropriate where bounds are proven for the effective profile. Reject overflow or out-of-range descriptors before allocation or launch. The selected shard exceeds 4 GiB; full-context arena addressing also requires checked wide products.

Every retained or peak allocation must have:

- An owner rank and allocation size including real alignment and padding.
- A persistent, session, batch, graph, replacement, or migration lifetime.
- A reason it must overlap another allocation, or a proven reason it can reuse an arena.
- A maximum derived from the effective profile and implementation limits.
- A measurement or construction proof for allocations made by libraries outside the explicit allocator.

Per-rank budgets cover target/method/vision weights and histories; recurrent/convolution/raw-pending/pooling/draft/sampling/constraint/output/media-position state; bounded language/visual execution/features; verification/recovery; graphs/communication; replacement/migration overlap; metadata/padding. Host media/replay/pinned staging and Section 17's queued inputs/retained checkpoints are separate bounded resources. Tiered retention increases logical session capacity, not GPU-resident KV capacity; charge transfer metadata/staging and permitted overlap in the GPU ledger.

Take the maximum for proven mutually exclusive workspaces and sum overlapping occupancy. Use one bounded execution arena where it can replace per-slot maximum slabs; reuse only after all asynchronous readers retire.

Use fixed-size typed history-page arenas and bounded execution/graph arenas. Track free classes/counts per rank, not aggregate bytes; reserve matching layer bundles before execution. Full admitted windows must remain allocatable despite fragmentation, graph replacement and migration, without fresh general-purpose device allocation per page at maximum residency.

Before publishing capacity, initialize or conservatively budget every eligible kernel/communicator/library family's lazy residency. Every selected plan must fit admitted commitments and the overlap envelope. Deleting a graph does not prove its pool was released. Enforce these invariants in one canonical allocator/planner.

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

Administrators choose context using the measured context/capacity/throughput frontier. Load adaptation must not resize windows, downgrade Q8/Q4 or disable the selected method. It may tune safe batches, proposals, prefill/media quotas, kernels/graphs and ownership within the selected profile.

Measure text prompt tokens/s, expanded visual rows/s, and committed output tokens/s separately, including per-request output rate, TTFT, TPOT percentiles, queue delay and fairness. Report media pixels/frames and preprocessing/encoding time; visual rows are not ordinary text-token throughput. Rejected candidates are not delivered output.

Count each target prompt row once; report prefix/media hits, replay and MTP/other conditioning separately. Include media preparation/encoding, transfers and maintenance in complete-operation timing. Track successful-task latency and reasoning/output counts so shorter answers are not mistaken for higher token/s.

Avoid batch-filling delay at low activity; maximize aggregate committed throughput within administrator latency limits at high activity. Mixed load must advance both prefill and decode. Optimize whole iterations, including routing, features, draft maintenance/drafting, verification, rejection/replay, communication, sampling, and commit.

Establish the selected TensorFold implementation's measured performance frontier at relevant operating points. Neither modern-GPU benchmarks nor peak arithmetic throughput supplies an established tokens/s target or universally winning plan.

## 6 Administrator configuration and effective profiles

### Inputs

Expose only implemented, qualified choices:

- A qualified Qwen3.8-Flash-Next target manifest and runtime weight representation, initially non-Swift SC117 IQ3_S.
- Role-qualified config/tokenizer/template/generation/processor asset locations and typed metadata overrides when preparation cannot resolve required values automatically; Section 2 defines validation and provenance. Preserve one target frontend for all shared-target speculative methods.
- Equal context allowance and a qualified native or extended RoPE/YaRN profile, with explicit target/MTP/draft coordinate bounds; 262,144 is the native baseline, not a permanent engine ceiling.
- Target KV format and its pinned implementation.
- Speculative method: none, stock MTP, or DFlash.
- Compatible MTP/DFlash artifact and supported method-weight/KV choices; initial MTP KV matches target KV, while initial DFlash KV is FP16.
- Explicit history-residency policy and implemented media limits/processing policy; the effective profile includes fixed vision placement and bounded visual execution.
- Optional session ceiling.
- Performance objective and any requested latency or waiting-time limits.
- Explicit retained-session lifetime, bounded FIFO admission/output queues and per-principal limits; qualified GPU-only or RAM/optional-storage retention with separate session/input/backing budgets (Section 17).

A draft sliding-window override is exposed only if the selected checkpoint and storage implementation support it. Target and draft weight formats, target KV, and draft KV are independent dimensions. Validate each requested sampling/constraint policy against the effective verifier's capabilities before admission or session mutation; reject unsupported combinations with a specific reason.

### Outputs

The engine resolves full-context resident capacity, admission and media-resource limits, language/visual execution envelopes, row/patch budgets, graph/kernel families, fixed vision placement and per-batch scheduling. It reports the binding rank/resource and qualification status.

Illustrative logical schema; concrete CLI/API names must implement these semantics:

```yaml
target:
  repository: SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF
  revision: 90a3350cd800ec1fe0d1660a38414ad9e985782f
  weight_directory: IQ3_S
context_tokens: 262144
position_profile: native  # immutable resolved RoPE/MTP/draft manifest; qualified YaRN profiles may extend this
kv_format: q8
speculation:
  method: mtp
  artifact: sc117_q8_mtp  # pinned source; actual target embedding/head aliases, Section 12
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

Define and preserve the declared lifetimes. Request-scoped slots release state after completion/cancellation and retirement of readers, transfers and callbacks. Only intentionally retained sessions survive requests under close/expiry rules; missing expiry does not make stateless requests persistent. Inactivity cannot delete retained state; Section 17 permits qualified idle-state suspension while preserving its lifetime/context. Completion, cancellation, disconnection, GPU demotion and retained-session destruction are distinct.

Use Section 17's bounded FIFO admission queue, explicit queue-full response and eligibility/aging rules. Bound media preparation before slot assignment, then validate exact expanded prompt/positions, output budget, media limits, and sequence count. Pending media work must retain its declared resource claim. Resumption preserves the retained logical/backing claim and acquires a GPU claim if suspended. Serialize or reject concurrent state mutations per sequence under the API contract. Each GPU-admitted request-scoped sequence retains its complete-window reservation until slot release; waiting input alone holds no GPU claim.

## 7 Automatic capacity planner

For the fixed context/position/numerical/method/media profile, compute the largest resident count supporting a useful qualified execution envelope. At that count, select the best measured aggregate committed decode under service limits, with prefill/media progress and fairness. Report any smaller-count throughput advantage without changing the requested capacity objective:

```text
Nmax = max N such that, for every rank r:
    target_weights[r] + method_weights[r] + vision_weights[r]
  + target_histories[r, N, context, codec, ownership]
  + method_histories[r, N, method_policy]
  + retained_state_and_media_metadata[r, N]
  + retained_recovery_checkpoints[r, N, method]
  + fixed_language_visual_execution_and_recovery_arenas[r, envelope]
  + unmaterialized_serving_transient_allowance[r, envelope]
  + graph_and_communication_residency[r]
  + positional_tables_and_scratch[r, position_profile, envelope]
  + replacement_obligations[r, N]
  + migration_overlap_limit[r]
  + metadata_and_padding[r]
  <= usable_serving_budget[r]
```

Execution envelopes bound target/expanded-prompt rows, proposals, the largest media unit, visual microbatches and overlap. Enforce separate host/pinned/replay bounds. Full residency must support useful selected speculation and promised media work; visual encoding may queue instead of multiplying its workspace by all sessions.

Ledger terms are disjoint allocation/lifetime categories. Charge exclusive positional tables/scratch there; reference shared language/graph arenas without double-counting. Target weights exclude separately charged speculation/vision; actual embedding/head aliases add no payload. Use the selected profile's actual SC117 allocations and MTP/DFlash/vision placement.

Apply Section 4’s incremental baseline and separate 2-GiB transient minimum beyond fixed arenas. Keep retained checkpoints, replacement and migration obligations distinct. After allocation, compare remaining promises with remaining budget; do not recharge materialized weights/history.

For the compact indexer, growing histories contain attention KV and completed pooled keys. Raw pending rings and saved tail versions belong to retained state/recovery; raw prefill/verification spans belong to bounded execution.

Charge retained checkpoint versions/future occupancy separately from peak recovery execution. Reconcile shared objects once while preserving private future-window/branch claims. Sections 9 and 17 define catalog/lease/dirty metadata, reclaimable cache, and snapshot/COW overlap.

Implementation sequence:

1. Resolve target/method/vision/processor manifests, actual imported allocations, codecs, media bounds and speculative compatibility; include conversion, activation policy and media replay/staging costs.
2. Measure runtime baseline and per-rank usable budget while preserving device-floor, workspace and recovery protections; establish host/pinned/media-store bounds.
3. Evaluate a small set of qualified execution/placement envelopes for the selected objective.
4. Compute exact per-rank persistent and peak costs for candidate resident counts.
5. Apply implementation limits and an optional administrator ceiling.
6. Initialize the chosen arenas and bounded graph resources.
7. Verify measured allocation against the plan and publish the effective profile.

Admission uses exact allocated strides, page/tail rounding, rank ownership, and library residency. Logical token rates are cross-checks.

Any binary search for maximum resident capacity requires monotone per-rank costs. For each candidate envelope, keep all earlier slot phases/owners, arenas and obligations fixed as N increases; adding a slot must not remap or reclaim an earlier commitment. Evaluate different envelopes independently, then compare their feasible maxima and measured throughput. If a candidate construction cannot establish monotonicity, use an exact bounded enumeration instead of applying binary search to it. Source: [capacity search](https://github.com/kklouzal/TensorFold/blob/600b0eaa0c59d35154c1be8e0c0a8570a9d8291e/src/tensorfold/cuda/capacity.py).

Preserve full-window GPU entitlements and tiered backing/reacquisition claims under Sections 6 and 17. Short prompts, presumed sharing or unimplemented offload cannot increase resident capacity.

Capacity may fall between policy thresholds: retain a computed 18. Buckets aid execution; they neither round down admission nor impose a permanent 32-session ceiling.

### Method-specific resource description

Each method declares weights/state, history geometry, proposals/verification rows, target/draft tentative and feature rows, recovery, supported shapes, validity maintenance and stop/commit rules.

Guard rows follow the method's tail policy and page/pooling rounding. A fixed +4 guard cannot be generalized to every MTP depth or DFlash block.

Clamp proposals to remaining context/reply budget, supported positions, and stops. Physical guard rows neither extend advertised context nor permit unsupported positions.

## 8 Capacity derivations and profile cross-checks

### Unique-history rates

For the required geometry, one head's K+V costs 544 B/row/layer in Q8 and 272 B in the pinned compact Q4 layout. Completed FP16 indexer keys cost 256 B per four-token group, amortized to 64 B/token/layer. Raw pending state is bounded rather than growing with context (Section 9). Rows include visual expansion; verify imported geometry and physical strides before applying these rates.

With two global KV heads stored once, balanced over eight ranks, and one copy of pooled indexer history:

```text
13-cache Q8 rate = 13 × (544 / 4 + 64 / 8) = 1,872 B/token/GPU
13-cache Q4 rate = 13 × (272 / 4 + 64 / 8) =   988 B/token/GPU

12-cache Q8 rate = 12 × (544 / 4 + 64 / 8) = 1,728 B/token/GPU
12-cache Q4 rate = 12 × (272 / 4 + 64 / 8) =   912 B/token/GPU
```

Twelve caches cover the target only; immediately available stock MTP needs its additional resources. DFlash adds its actual draft resources to twelve target caches.

### Full-context history arithmetic

These are **ideal average growing-history costs**, excluding raw rings/staging/recovery versions, guards, page rounding and owner imbalance. MTP assumes thirteen QSA caches whose K/V use the selected target codec; pooled indexer keys and raw pending state remain FP16. For a different MTP cache format, sum twelve target caches and the actual MTP cache independently.

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

Add the selected profile's actual target/MTP/vision weights, all retained state, recovery, execution, graph/communication residency and overlap obligations. A count in this table does not establish SC117 capacity. Fixed/session costs also prevent a rule that halving context doubles capacity.

Under Section 9's indexer mapping, sixteen and twenty-four MTP sessions balance to 26 and 39 indexed caches per rank. Twenty sessions produce 32 or 33 per rank; twenty-eight produce 45 or 46. Use actual worst-rank ownership, weight placement and visual residency, not averages or rounding to policy thresholds.

### Selected DFlash history cross-check

For the initial TP8/FP16 Inco draft and context C of at least 2,048 tokens, the ideal unique growing-history cost per resident sequence per GPU is:

```text
target Q8 KV + pooled indexer: (1,728 × C) bytes + 5 MiB of draft KV
target Q4 KV + pooled indexer:   (912 × C) bytes + 5 MiB of draft KV
```

At C=262,144, this is 0.426758 GiB/session/GPU for Q8 target KV or 0.227539 GiB for Q4. Add target-only weights, approximately 0.429022 GiB/rank draft payload, and the remaining ledger, including bounded raw indexer state. Shorter contexts use draft retention up to `min(C,2048)` plus tentative/rollback rows. These cross-checks exclude page rounding and do not establish final session counts.

## 9 Canonical paged history and state ownership

During GPU residency, store each target KV head once across its four-rank group and each completed pooled indexer history once across the box, with bounded raw pending state at the same owner. Section 17 governs complete idle-session suspension/restoration; executing attention never reads a partially restored or host-paged history. Initially compute each session/layer indexer at one canonical owner. Retain target TP8 projections and output layout. Its four indexer query heads and aggregation are distinct from the 24 target query heads and two KV heads; rank-local or per-KV-head approximations are not equivalent.

Assign the indexer owner as `(indexed_cache_count × slot_id + cache_ordinal) mod 8`, with profile-fixed cache ordinals and slot IDs. Charge actual ownership across every configured slot and preserve its phase on reuse. Use thirteen indexed caches for QSA-MTP and twelve for target-only/DFlash. Total-count divisibility does not replace this ownership calculation.

### Page and session metadata

Each logical page records its session generation, attention layer, global head, absolute sequence-row range, valid rows, codec/schema, immutable rotation identity, physical owner/offset, ownership generation, and decoding metadata. Keep sequence-row indexing distinct from target multimodal coordinates. Device tables and arena addresses remain stable for captured execution.

Session state distinguishes:

- Canonical committed transcript, externally delivered-output position, and per-sequence sampling/constraint, incremental detokenization and stop-matching state under Section 12.
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

Bind initial target/MTP QSA selection to descending FP32 score with ascending logical pooled-group ID for equal scores. This selected numerical policy follows the pinned helper's deterministic small-index option; ordinary [`torch.topk`](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html) does not guarantee stable tied indices. Use the same comparator in controls, chunked/verification paths and any distributed selection merge. Physical owner/page/ring indices cannot break ties. Include the comparator in profile/cache identity and test cutoff ties, including all-zero scores, across batching, migration and restore; do not demand agreement with an unstable reference's arbitrary tied IDs.

For valid cache length `n` measured from that cache's fixed grouping origin, retain `floor(n/4)` completed pooled keys and `n mod 4` pending raw rows. Use a four-row FP16 raw ring per indexed cache/sequence: 1,024 payload bytes before metadata, placement, padding and recovery versions. Preserve required member coordinates and logical positions. Release completed raw members once pooling and all required tail/checkpoint captures have finished and readers retire. Source: [pinned ring metadata](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/attention/qsa/metadata.py).

Protect the pre-step pending tail and bounded new raw span until compression and accepted-prefix commit resolve; writing an entire verification span into the same four-slot ring can overwrite required carry-in members. Form crossing groups from the saved tail plus new rows. If `a` new cache rows become valid, retain `floor((n+a)/4)` complete pools and the ending `(n+a) mod 4` raw tail; invalidate groups containing rejected members. Each query sees only groups complete at its own causal frontier, even when later pools are already computed. Support arbitrary chunk splits; padded rows do not advance grouping.

Checkpoint each recoverable boundary's pooled-valid frontier, pending raw values and coordinates with its other state. Whole-append rollback restores this checkpoint and discards later pools. A long pending append uses protected checkpoint tails plus bounded current-span storage. Use each cache's own validity frontier, especially MTP's shifted/deferred rows; its target-conditioned refresh rebuilds affected groups. Physical migration preserves grouping origin and membership.

### Initial owner execution

For each target attention layer:

1. Execute the target TP8 projections.
2. Route each query head to the owner of its corresponding global KV head history.
3. Append incoming KV once at its owner, in the committed or tentative range required by the execution method; promote only the accepted prefix.
4. Construct exact indexer queries and update the bounded raw pending state and pooled keys once at their canonical owner under the same committed/tentative transaction.
5. Select visible complete pooled groups, expand all their members, and append the visible incomplete tail.
6. Dispatch selected logical positions to the KV owners.
7. Compute attention and return outputs to their original TP positions.
8. Continue the target output projection and distributed execution.

Batch routing and collectives by layer, not request. Preserve per-row causal boundaries, masking, pooling, ties, deterministic logical positions, and exact sparse selection. Later tentative appends must remain invisible to earlier queries. Replicated histories do not establish replicated indexer query inputs.

Transient projection duplication is initially permitted; persistent histories remain unique. Optimize it only for measured benefit.

### Prefix reuse

Compatible prompt-prefix reuse is a required implementation/qualification workstream. It avoids repeated model work; physical cross-session page sharing remains optional and full-context capacity assumes cache misses and private growth. Use one page/state manager for active histories, reusable prefixes and Section 17's queued checkpoints, with explicit ownership classes rather than duplicate cache stores.

Within an explicitly allowed cache-sharing domain, match the exact rendered/expanded token prefix and complete dependent identity: target/runtime weights and arithmetic, tokenizer/template, codec/rotation, RoPE/M-RoPE/positions, model/feature contracts and required companion state. Include immutable media content, processor/settings, grids/order and target continuation state; equal placeholder IDs with different pixels cannot hit. Use canonical serialized metadata and collision-resistant prefix/page keys, validating identity/lengths rather than trusting a filename or short hash. Bound tokenization/render/media-result caches separately and key each by its own complete dependencies. Reuse model state, never a previous response as the new answer.

Select the deepest stored boundary with every required component: target KV/pooled keys/raw group tail, GDN and PLE recurrence/convolution/token-hash state, positions, and method conditioning or a qualified bounded reconstruction path. A later recurrent checkpoint cannot be cropped backward, and taking the minimum of component lengths does not create a missing checkpoint. Reuse inside a group requires the matching raw-tail snapshot or replay from an earlier complete boundary; pooled means cannot reconstruct discarded keys. Retain a bounded set of useful message/chunk/end boundaries and budget their actual state versions.

An exact whole-prompt hit still needs valid next-logit/hidden state under the method's cached-versus-committed convention, or a qualified earlier boundary from which to execute required rows. Never consume the final prompt row twice into recurrence or pretend KV alone supplies its logits. Honor requested prompt log-probabilities: skip work only when the cache contains the exact required values. Canonical rendering changes require longest-prefix matching again, not reuse by chat/session label alone.

For a new request or branch, initialize its own sampling/RNG, constraints, penalties, stops and output state from its API contract and prompt; do not inherit another request's generation bookkeeping. An interrupted same-request continuation preserves generation bookkeeping; a new turn in a retained conversation initializes new-request bookkeeping while reusing compatible model state. Read-only shared prefix pages require reference counts and copy-on-write of mutable/partial pages plus private recurrent/draft/sampling state; otherwise use a budgeted independent copy or move exclusively owned idle state. No write may invalidate a longer retained branch or another reader.

Retain useful prefixes only within a bounded reclaimable budget after guaranteed GPU/host/backing and execution obligations. Reclaim unleased reusable entries before suspending useful work; cache residency cannot strand promised full-window page classes. Publish entries only at validated state boundaries, and acquire a lease before selecting/restoring them. A reusable-cache miss/corrupt entry permits ordinary qualified prefill; loss of the sole required queued checkpoint is a session failure. Section 17 defines shared tier transfer and backing-object lifetimes.

References: [vLLM prefix identity/block ownership](https://docs.vllm.ai/en/latest/design/prefix_caching/) and [SGLang hybrid reuse boundaries](https://www.sglang.io/blog/unified-radix-cache). These support the identity/ownership principles; the exact SC117/SM70/TP8 hybrid-state snapshots need independent qualification.

## 10 Attention degree and balanced placement

Support target attention degrees 1, 2, and 4 within each global KV-head group: that many ranks hold disjoint portions of a session/layer/head history. Target weights remain TP8.

Degree 1 computes attention at the owner. Degrees 2 and 4 distribute selected positions, route queries, and merge partial attention. Balanced bytes can still yield unbalanced compute when sparse selections concentrate on particular pages.

### Proposed target-layer mapping

For local rank indices 0 through 3:

```text
base = (slot_phase + layer_phase) mod 4
owner(page, degree) = base XOR (page_index mod degree)
degree is one of 1, 2, or 4
```

Cycle layer phase uniformly over the twelve target attention caches. Each matching quartet must use a common degree for a given page index and identical fixed-size physical footprints. XOR then places that logical page equally across the four ranks. Partly valid tails retain full page allocations; independent per-layer degrees invalidate the balance proof.

Verify these assumptions against the manifest. Reserve complete matching page bundles before prefill/verification advances different layers, and charge residual transient layer skew. Map physical ranks so nested degree-2 pairs use appropriate measured links.

The MTP cache is outside this twelve-layer proof. Initial MTP KV may be striped across all four ranks per head group; qualify that layout and place/charge its indexer separately. DFlash requires its own geometry/placement plan.

### Exact partial attention merge

For the same query and disjoint selected positions, each participating rank computes local maximum m_i, exponential denominator l_i, and weighted value numerator z_i. Merge:

```text
m = max_i(m_i)
l = sum_i(exp(m_i - m) × l_i)
z = sum_i(exp(m_i - m) × z_i)
output = z / l
```

Do not average independently normalized outputs. Empty shards contribute an empty softmax state, with correct handling of the all-empty case according to the original attention semantics.

Keep the indexer canonical during KV placement changes and route selected positions through the current owner table. Independent local top-k operations require a separately implemented exact global procedure.

### Placement policy

Measure degrees 1, 2, and 4 across activity, context, selected-work, and prefill/speculation shapes; no threshold has a predetermined winner. All eight GPUs continue target-model work even when degree-1 attention uses fewer owners.

Compare fixed four-way context sharding without load-driven movement. Adopt adaptive placement only when its gain exceeds communication and migration costs.

## 11 Bounded live cache migration

Migration changes physical ownership while preserving compressed bytes, without a second complete history, precision conversion, or model restart.

### Resource and scheduling rules

Qualify or adjust an initial additional-occupancy cap of 64 MiB/GPU. Count all temporary excess: uncommitted destination copies, staging/bounce space, metadata, and source-retirement lag.

Before moving, verify current occupancy, future full-window reservations, final placement, and every intermediate overlap. Inactive sessions retain their pages. If the move cannot fit, execute from the valid current placement.

Migrate target-layer degree changes as balanced bundles of the same token-page index across matching layer quartets, sharing old/new degrees and equal footprints. Publish each bundle together. Reserve destination page classes and all overlap before copying. Bound transfers by time and occupancy; never move an entire layer first and repair temporary imbalance later.

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

Before moving, assess workload persistence, remaining generation, measured bandwidth, interruption/interference and uncertainty.

Under the twelve-layer geometry, full owner-to-degree-4 redistribution moves about three quarters of unique target KV: approximately 1.20 GiB Q4 or 2.39 GiB Q8 across the box per full session, excluding indexer movement. These are transfer volumes, not timing predictions.

On demand spikes, stop unnecessary new chunks and use the valid actual placement. Coalesce pending changes and apply hysteresis against repeated threshold movement.

## 12 Speculative execution and accepted-prefix correctness

### One method interface

None, stock MTP, and DFlash supply resource/state requirements to one planner and scheduler through a small execution contract. MTP implementation and qualification are mandatory for the initial SC117 release; DFlash is the separately selected companion workstream. Do not load both speculative systems in one initial resident profile.

Each method defines proposal construction, target row count, target/draft tentative storage, valid position offsets, recovery, feature/state maintenance, and commits, with limits matching actual kernels and buffers.

Resolve MTP per target: use compatible embedded tensors or an explicitly paired companion, without loading both. Family identity does not transfer the initial companion's tensor count, QSA semantics, or footprint. DFlash eligibility requires the Section 14 feature and probability contract.

### Stock MTP

Use `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF`, revision `90a3350cd800ec1fe0d1660a38414ad9e985782f`, file `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` as the canonical packed MTP source. Its file size is **4,137,429,312 bytes** and SHA256 is **`713b16d7e97246858ebd9a6dec403b1cfc3407d8a539f65401948e20b38b0f91`**. The complete directory contains 34 tensors: 21 Q8_0, eleven F32 and two BF16, with 4,126,482,432 payload bytes beginning at aligned byte 10,946,880. Its published full-file identity matches the ggml-org source copied by SC117. Sources: [immutable file pointer](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/raw/90a3350cd800ec1fe0d1660a38414ad9e985782f/mtp-Qwen3.8-Flash-Next-Q8_0.gguf) and [publisher replacement provenance](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/commit/de0c658ad462f748e6278060a3f8d57b5b3724b4).

Import only the **32 MTP-core tensors**, totaling **2,775,621,632 source bytes** (2.584999 GiB): nineteen Q8_0, eleven F32 and two BF16. Skip the companion's `token_embd.weight` and `output.weight` copies, which together occupy 1,350,860,800 Q8_0 bytes. Bind the actual selected target tensors, whose source representations are IQ4_XS embedding and Q6_K output, through the same qualified target kernels. Never convert, retain or execute the companion's duplicate vocabulary matrices merely because they are present. The source file is larger than its imported core; actual per-rank import/sharding/alias allocations still determine capacity. Read/hash the full source at installation and prepare/import through bounded memory.

This stock base-family companion supplies explicit QSA metadata and shares the target repository; it is not a demonstrated SC117 refit or V100 speedup. Section 25 records the Unsloth core comparison and its partial-payload evidence. No automatic alternate-companion path is permitted. Refit, lower-bit or engine-specific heads require complete pairing/import/numerical/acceptance qualification.

The official control source is the 31 `mtp.*` tensors in `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d43b917e7706784d8bb445c9af86a3540`: 5,214,301,696 bytes of BF16 payload across the source shards. Extract only the required tensors with bounded preparation memory, retain a checksummed import manifest, and convert to the qualified SM70 representation. The inspected Unsloth shared-BF16 GGUF from the [pinned comparison directory](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6/MTP) is conversion evidence; it does not select a second serving path or override the declared QSA policy. Source: [official tensor index](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/model.safetensors.index.json).

The official 31 tensors span 28 shards. Conversion produces 32 core tensors by splitting `gate_up` and `index_qk`, then concatenating `fc_embedding`/`fc_hidden` into `eh_proj`. Section 25 records the shared-BF16 census. Validate converted numerical values and actual target embedding/head aliases; tensor-count reconciliation is not parity.

Packed and BF16-derived controls use the actual SC117 vocabulary aliases and Section 2’s target frontend. Reconcile fused/split projections, transformed RMS weights, native fusion and MTP hyper-connections. Capture completed block-47 HC residuals before the final target HC mixer: all four streams form the 10,240-wide input. Preserve the selected SGLang reference’s RMS normalization across that entire width, its 10,240 learned weights/order, epsilon 1e-6 and one `1 + weight` application; normalize 2,560-wide input embeddings separately. GGUF conversion already folds the Gemma offset into gamma: use it directly or convert once to zero-centered weights. Only then reshape HC into four streams for the shared hidden projection. Per-stream normalization, the target’s mixed output, stream means and DFlash features are not substitutes. MTP disables PLE; target PLE still advances/rolls back during verification. Qualify each representation separately without permanently loading both.

### Explicit MTP attention policy

The initial TensorFold MTP profile uses **QSA with compression four and complete-group token budget 2,048**, matching the selected SC117 Q8 companion's final compression entry, indexer metadata, and the inspected model-config/SGLang path. Validate this declared policy and the final nonrecurrent MTP layer against the tensor inventory and executed operator path. Older shared-Q8/BF16 comparison artifacts encode zero in that final compression entry; they cannot silently select dense attention for this profile or replace its authoritative metadata. Dense and QSA MTP are different algorithms, even when indexer tensors are present. Sources: [SGLang MTP reference](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/models/qwen4_exp_mtp.py) and [llama.cpp target/MTP implementation](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/src/models/qwen4exp.cpp).

Recompute MTP QSA selection for each row/step. Disable `index_share_for_mtp_iteration` in [pinned SGLang parity controls](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/speculative/eagle_worker_v2.py); later selection reuse requires separate qualification.

Verify QSA-MTP import/execution parity using identical decoded weights, QSA policy, numerical settings, inputs, masking, and selected-tail semantics. Compare packed-Q8 and BF16-derived control profiles for acceptance, useful throughput, and quality; different weight encodings need not yield identical proposals. Dense MTP requires a separately selected profile with its own attention-work, allocation, and throughput proof. A thirteenth indexer allocation alone does not establish which attention path executed.

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

Zero proposals retains weights, valid conditioning and execution prerequisites for immediate reactivation; charge maintenance. Reclaiming resources requires a configuration transition and any necessary reconstruction.

### Accepted-prefix transaction

Verification must commit only the accepted and permitted prefix across every state type:

- Target KV, pooled indexer history and bounded raw pending state.
- Recurrent state and convolution tails.
- PLE rolling token/hash state and PLE convolution tail.
- Pooling origin, completed-group frontier and pending members' coordinates.
- Draft KV and conditioning state.
- Logical/emitted/target-cached/draft-valid positions, multimodal continuation metadata and any media/replay dependencies.
- Enabled grammar/constraint cursors, repetition/frequency/presence counts, incremental detokenization and stop-matching state.

Isolate tentative updates to this bookkeeping and commit only the permitted prefix, or reconstruct it from the canonical transcript before reuse. Rejected candidates must not alter later penalties, constraints, stops or published text. Charge retained state and bounded tentative work to their actual host/device owners.

Follow the method's proven sampling algorithm, preserving its RNG advancement through per-sequence proposal, acceptance, and target/correction streams or draw identifiers. Rejected proposals, acceptance tests, and correction sampling may consume draws; rejection does not authorize rewinding RNG to the accepted-token count.

The 36 recurrent layers require accepted-prefix state selection or proven checkpoint/replay; KV cropping cannot undo recurrence. Budget and measure the strategy, including longer blocks' additional intermediates or recomputation.

EOS, cancellation, and context/reply-budget exhaustion can further shorten the committed prefix. Later evaluated tokens must not become visible or permanent state. Preserve the service's token-versus-text stop boundary; an intra-token byte stop does not permit KV cropping at a character offset.

### Commit and output publication

One coordinator resolves verification, limits, stops, and method-defined target/draft state before publishing the accepted prefix to the output queue. Never stream tentative tokens. The server's committed transcript governs retained-session resumption, including after disconnection; network delivery is not model-state authority.

Apply cancellation at a coordinated scheduling boundary. Finish or safely retire the issued plan and prescribed communication before releasing pages/slots. Cancellation cannot retract published output. Generation/lifetime checks must prevent late completions writing into reused slots.

Use bounded asynchronous per-request output queues. Backpressure pauses that request or applies its timeout/cancellation policy without blocking unrelated GPU progress. Detect stop strings across token, speculative-block, and stream-chunk boundaries, withholding text that could expose a partial stop marker. Test against the non-speculative path.

### Sampling and numerical guarantees

Verify speculation against the configured target computation and sampling policy. Greedy matching does not establish stochastic acceptance/rejection. Apply temperature, truncation, penalties, constraints, and stops at the correct conditional prefix. Candidate-selector DFlash requires the corresponding proposal probabilities.

For a sampled proposal `x_i`, let `p_i` be the normalized effective target distribution and `q_i` the actual conditional proposal law given preceding proposed tokens. Accept with `min(1, p_i(x_i) / q_i(x_i))`. At the first rejection, sample the correction from `max(p_i - q_i, 0)` normalized over the entire target vocabulary, then discard later proposals. If all proposals are accepted and output room remains, draw the bonus from the next target distribution. For sparse selector support, `q_i` is zero outside that support; correction sampling must still cover the full target support. Greedy target sampling is the explicitly defined argmax point mass. Source: [speculative sampling algorithm and proof](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).

The proposal law may differ from target truncation/penalty settings, but the verifier must retain the law that actually generated each proposal. Validate finite normalization, nonzero probability of sampled proposals, and numerical residual handling; never replace an invalid residual with an undeclared distribution. Bound and account for distributed probability/correction work. Apply reply/stop cuts through the same accepted-prefix transaction, without publishing an otherwise valid bonus beyond the permitted boundary.

Lossless speculation is relative to the configured numerical target, not Q4/Q8 equivalence. Declare tolerances and sampling guarantees for reduction-order differences; universal bit-identical text is not promised.

Keep per-sequence RNG ownership independent of batch permutation and migration. Padded rows must not advance another sequence's RNG or state.

## 13 RotorQuant and future codec work

### Initial Q8 contract

Define the baseline TensorFold INT8 codec exactly as follows; it uses a normalized Walsh–Hadamard H32 transform independently within each 32-value group. For transformed values `y = H32(x)`, the pinned writer computes FP32 `s = max(abs(y)) + 1e-10`, stores `s` in FP16, and constructs codes using the unrounded FP32 scale:

```text
q = clamp(floor(y * (1 / s) * 128) + 128, 0, 255)
stored_signed_code = q - 128
decoded_in_H32_basis = (stored_signed_code + 0.5)
                      * fp32(stored_fp16_scale) / 128
```

Use the matching query/value basis handling and preserve each native transform/cast point; the SM70 attention path uses FP16 working operands after reconstruction. Code construction uses the FP32 scale while decoding uses its stored FP16 rounding. An all-zero group stores a scale that rounds to zero and reconstructs zero. Each group occupies 32 code bytes plus a two-byte scale, giving `2 × 8 × 34 = 544` bytes per dimension-256 target-head K+V row. This H32/midpoint codec is **not GGML Q8_0**; replacing it with conventional nearest-integer symmetric quantization changes the numerical target.

Pin writer, reader, attention consumer and receipt-build revisions. Verify signed-code/scale layout, exact boundary rounding/clamping, underflow/overflow, zero/nonfinite handling and byte compatibility; qualify execution-induced numerical differences. Migration copies bytes and metadata without requantization. A different codec needs a separately selected numerical profile and cannot reinterpret retained pages. Sources: [native writer](https://github.com/kklouzal/TensorFold/blob/600b0eaa0c59d35154c1be8e0c0a8570a9d8291e/src/tensorfold/cuda/sm70/family.cu), [Python mathematical control](https://github.com/kklouzal/TensorFold/blob/600b0eaa0c59d35154c1be8e0c0a8570a9d8291e/src/tensorfold/cuda/sm70/family_ops.py), and [packed attention reader](https://github.com/kklouzal/TensorFold/blob/600b0eaa0c59d35154c1be8e0c0a8570a9d8291e/src/tensorfold/cuda/sm70/packed_kv_decode.cuh).

### Format requirements

The initial Q4 profile is **K=iso4, V=iso4**, specifically `GGML_TYPE_ISO4_0` / `block_iso4_0` at reference commit `08e025c06ab521e4fa9e5c08b80af57614543e53`. This starting profile is neither a measured V100 winner nor a universal upstream default. The author's limited four-bit comparison favors IsoQuant; qualify actual SC117 long-context quality and throughput.

For this profile, the pinned **CUDA writer, CUDA quaternion/centroid tables and CUDA attention reader** define the format authority. The same reference commit's CPU ISO4 quaternion tables differ from its CUDA tables; its CPU Planar4 rotation tables differ as well. The upstream CPU conversion therefore cannot serve as a byte-compatible oracle unchanged. Construct an independent control using the selected CUDA constants and explicitly matched representation/math before asserting cross-backend compatibility. Sources: [CPU ISO4 implementation](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-iso4-quant.c), [CPU Planar4 implementation](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-planar4-quant.c), and [CUDA tables](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-cuda/planar-iso-constants.cuh).

Each iso4 block represents 128 values with 32 quaternion rotations and occupies 68 bytes: 64 bytes of codes plus two FP16 metadata fields. Preserve the pinned centroid/quaternion tables, reconstruction-norm correction in `norm`, and reference zero initialization of the reserved `rnorm` field. A dimension-256 K or V head uses two blocks; K and V together occupy 272 bytes per head/token/layer before allocation padding. Other four-bit layouts can have the same nominal footprint while using different metadata and numerical semantics.

A compact arena may carry the two FP16 fields as one opaque four-byte word. A `float32` carrier tensor does not make that word an FP32 numerical scale: preserve its bits during copy, checkpoint and restore, without numeric casts. Source: [compact-codec representation](https://github.com/kklouzal/TensorFold/blob/600b0eaa0c59d35154c1be8e0c0a8570a9d8291e/src/tensorfold/cuda/sm70/rotorquant_compact.py).

RotorQuant requires independent qualification of format roundtrips, zero vectors, scales/norms, rotations/centroids, quality, peak memory, prefill, and decode; qualification of another codec does not transfer.

Sources: [RotorQuant project](https://github.com/scrya-com/rotorquant) and [audited reference schema](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-common.h).

Use planar4/planar4 as the first equal-memory comparison after iso4; its simpler two-dimensional rotation does not establish a faster V100 path. Do not start with unsupported mixed variants. Administrators select qualified codec profiles; activity thresholds adjust execution within that codec, without silently requantizing retained pages. Exact implementation references: [GPU packing and norm correction](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-cuda/set-rows-planar-iso.cuh), [centroids and rotations](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-cuda/planar-iso-constants.cuh), and [attention transforms](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-cuda/fattn-common.cuh).

### Reference integration boundaries

The audited llama.cpp commit allocates deferred FP16 K plus a compressed destination and provides CPU conversion. Use direct GPU packing and bounded progressive-prefill storage in TensorFold; a full FP16 history shadow or full-history conversion peak invalidates compact-only capacity. Source: [pinned cache allocation and conversion code](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/src/llama-kv-cache.cpp).

The reference dispatcher uses vector Planar/Iso attention, which does not establish optimized SM70 prefill or large verification. Implement and compare small-row and tiled TensorFold paths. Source: [pinned attention dispatch](https://github.com/johndpope/llama-cpp-turboquant/blob/08e025c06ab521e4fa9e5c08b80af57614543e53/ggml/src/ggml-cuda/fattn.cu).

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

Derive physical retention per layer from its actual full-prefix, sliding-window, or other supported policy. Full-prefix storage grows with context; windows save memory only through bounded allocation. Do not transfer the initial Inco profile's geometry, state, block convention, or memory totals to another draft. Budget actual persistent recurrence/convolution state, feature buffers and selector tables. Reused target embeddings/heads require real storage sharing or qualified owner-served access with bounded activation transfer; do not add persistent copies for throughput. Independently learned draft tensors are distinct parameters and require their own numerical and residency inventory.

Qualify acceptance and useful throughput against exact target weights, effective arithmetic, KV codec, and generation policy. Matching dimensions do not preserve profitable proposal length, capacity, or state compatibility. Cache identity includes the numerical target/draft and feature contract; model changes cannot reuse KV, recurrence, or draft conditioning based only on shapes.

### Selected artifact and alternatives

Use **incoai/Qwen3.8-Flash-Next-DFlash2**, revision **a22a1744e85c17364dcc54a6a420af7135d4c7f6**, as the initial DFlash pairing to complete and qualify against the pinned SC117 target. Import its source `config.json`, `model.safetensors`, and feature-capture contract. The initial draft profile is **fixed TP8, FP16 draft weights, FP16 physically bounded draft KV, and block size four**. The following manifest, calculations, and operations apply to this pairing.

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

The reference capture patch asserts TP1/CP1. Implement TP8 capture for TensorFold's actual residual sharding/replication and verify parity across ranks, chunk sizes, prefill, and verification. Removing the assertion cannot establish equivalent features. The author's GGUF card identifies missing HC-mean capture as a llama.cpp blocker; downloadable weights alone are insufficient.

Keep native four-stream target computation; take the mean only for draft features. Qualify exact SC117 IQ3_S features and acceptance for both target KV codecs, including visual prefixes, chunking, corrections and continuation positions.

### Physical draft memory plan

Reference placement shards attention/MLP projections and replicates feature projection, dynamic-convolution parameters, selector codebooks/projection, and normalization arrays. The FP16 TP8 payload is:

| Weight group | Source payload bytes | Bytes per TP8 rank |
|---|---:|---:|
| Attention and MLP projections | 851,968,000 | 106,496,000 |
| Remaining replicated parameters | 354,163,200 | 354,163,200 |
| Total | 1,206,131,200 | **460,659,200 = 0.429022 GiB** |

The two selector codebooks account for 254,279,680 bytes per replicated rank. Dividing the whole checkpoint by eight would undercount this placement. Source: [pinned SGLang DFlash model implementation](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/models/dflash.py).

Account for padding, qualified replacement-format residency, activations, graphs, communication, feature buffers and sampling scratch. KV-materialization helpers must use source-weight aliases or bounded tile decoding; target embedding/head access must share actual storage or use the qualified owner service. A replacement FP32 or fused runtime representation releases its superseded GPU copy after bounded conversion/validation. Do not retain additional full stacked or decoded weight copies for throughput; charge all permitted conversion overlap and tile workspace.

At TP8, each draft rank can own one of its eight KV heads. An unreplicated FP16 window costs:

```text
2 × 5 layers × 8 KV heads × 128 values × 2 bytes × 2,048 positions
= 41,943,040 bytes = 40 MiB/session across the draft
= 5 MiB/session/GPU at TP8
```

Enforce bounded physical allocation for the committed window plus tentative rows, page alignment, and rejection/rollback protection. Future writes must not overwrite committed entries needed after rejection. Retire committed entries only at the accepted-prefix boundary through bounded tentative pages or an equivalent proven ring transaction.

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

Qualify genuine block-two/block-three execution separately: it changes non-causal interactions, grouped-convolution boundaries, masks, scratch, and graphs. Configured block eight is not qualification for target verification. Near the position limit, skip fixed-block drafting that would evaluate unsupported positions unless a smaller valid block is qualified.

Longer blocks require per-sequence QSA/recurrence support and aggregate row capacity; a larger R32 envelope alone is insufficient.

### Selector, masks, and sampling

Implement the selector and block-local two-tap dynamic convolutions, respecting block membership and padding. These convolutions create no persistent draft recurrent tail.

The selector defines conditional candidate paths; rejection sampling must use their actual conditional proposal probabilities, not pre-selector logits. Preserve algorithm-defined RNG advancement. Source: [author's DFlash 2 method description](https://inco.ai/blog/dflash2/).

Implement and qualify selector-aware stochastic verification on SM70 for every exposed sampling policy. The pinned worker's `_validate_phase1_sampling_support` can fall back to greedy argmax when non-greedy verification is unavailable, explicitly changing the requested distribution. TensorFold must reject unsupported settings before admission or state mutation; this reference fallback is not permitted. Source: [pinned DFlash worker](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/speculative/dflash_worker_v2.py).

Use the checkpoint's non-causal draft mask while preserving per-row target causality across KV, QSA pooled/tail selection, recurrence, and PLE. Future candidates must not influence earlier target logits.

Keep vocabulary projection sharded where possible and merge local top-16 candidate sets without gathering all draft logits per rank. Bind codebook rows to actual target token IDs and mask token.

Budget target verification/correction sampling separately: one FP32 [128,248320] matrix costs 121.25 MiB, with potentially overlapping transformed logits, probabilities, masks, and correction intermediates. Qualify exact temperature, top-k/top-p, penalties, constraints, and correction sampling; greedy top-k establishes neither general sampling correctness nor peak memory. Source: [pinned speculative worker](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/speculative/dflash_worker_v2.py).

### Performance and capacity qualification

Start with FP16 draft weights/KV, appropriate FP32 accumulation/reductions, and explicit BF16 conversion checks. GB300, BF16, TRT-LLM, FlashInfer, and Triton reference paths do not establish SM70 support. Qualify kernels, HC capture, selector, masks, window wraparound, every rejection position, skipped-proposal maintenance, and high absolute positions.

Compute capacity from twelve target attention/indexer caches plus actual DFlash weights, bounded draft KV, features, target-state recovery, execution resources, graphs, and overlaps. The MTP thirteenth-cache formula does not apply. A DFlash-only profile may release MTP resources through the permitted in-process transition.

Compare committed throughput against no speculation and imported MTP with identical target numerics, reasoning/sampling modes, histories, and workloads. Select the best qualified execution within the enabled profile, including zero proposals when beneficial.

Inco is a SC117 candidate, not guaranteed useful speculation. Poor acceptance may justify exact-target fine-tuning using verified SC117 numerical-target HC features/continuations/distributions and its intended embedding/head, consistently training/validating backbone, block convolutions and selector. Training cost and V100 speedup remain unestablished.

## 15 SM70 kernel and model execution work

Retain each profile's qualified runtime weight inventory, including explicit NVFP4 decode/conversion choices. All routed experts in the selected SC117 retained plan remain resident. Do not introduce unbudgeted host expert paging or full-checkpoint FP16 expansion; each artifact must fit its complete serving plan.

Implement qwen4exp's four HC streams, target per-stream normalization, gated query projection, PLE injection and all selected mixed weight formats. Its 12,288-wide query/gate projection contains 24 query heads plus their gate. Preserve MTP's distinct flattened-HC norm axes (Section 12); reconcile HF `1 + weight` conventions with converted GGUF values exactly once.

Profile indexer-query assembly, pooled scan/top-k, selected-group expansion and KV gather, selected-position routing, and KV attention separately. Reuse history tiles across query rows only when causal masks permit. Bound selected-index and unpacking work by tiles and row budgets; never expand selected KV for an entire long prefill batch. Increasing KV attention degree does not remove a canonical indexer bottleneck. Further indexer parallelism requires measured whole-step benefit and exact global selection.

Every kernel family requires native SM70 compilation and supported math/library paths. Respect Volta's independent thread scheduling and explicit synchronization requirements ([Volta tuning guide](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html)). Select paths using complete packing, communication, state, and recovery costs, not Tensor Core utilization or isolated kernel rate alone.

### Performance engineering contract

Apply these rules to target, MTP, DFlash and vision under Sections 4–7's capacity/quality commitments. Justify hot-path compute, traffic, launch and synchronization costs; exceptions require measured justification and bounded resources/lifetimes.

| Area | Required implementation rule |
|---|---|
| Device residency | Keep reusable KV/state, routing/indexer results and speculative features on GPU until consumers retire. Pack KV directly there and unpack only required tiles. CPU tokenization, bounded media preparation, the host/NVMe-backed PLE table, committed output and Section 17's controlled session checkpoints are intentional boundaries; GPU residency never authorizes extra weight copies or reduced session commitments. |
| Host feedback | Audit explicit/implicit `.item()`, `.cpu()`, `.tolist()`, NumPy conversion, host branches on device values and CPU-created indexing. Avoid per-layer/row readbacks. Prefer device-resident counters/indices and fixed-capacity outputs; dynamic `nonzero`/masked selection or tensor slice bounds can introduce synchronization. Preserve the exact operator, ordering and masks when changing representation. |
| Transfers | Coalesce necessary control/input/output transfers; normally return committed token IDs/compact status rather than full logits/features. Budget explicit log-probability requests separately. Use bounded preallocated pinned staging and useful asynchronous copies; protect buffers through DMA/consumer retirement and await D2H completion before CPU reads. No on-demand pin/register/unregister cycle per token. |
| Sampling/output | Prefer qualified GPU sampling, penalties/constraints and verification to avoid host round-trips of vocabulary logits. Preserve selector probabilities, grammar/stop semantics and RNG. CPU text assembly/decisions consume bounded compact committed results/status; reject unsupported sampling without changing policy. |
| Inference runtime | Set evaluation mode and disable autograd in every inference worker/thread; qualify thread-local `inference_mode` compatibility. Keep Python orchestration, serialization, logging and locks off the per-layer critical path; move measured hotspots into the canonical native path. Bound queues and diagnostics; retain launch/error and asynchronous completion checks. |
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

cuBLASLt provides reusable algorithms and workspace constraints; CUTLASS supports SM70 FP16 TensorOp/WMMA, and its grouped GEMM can be specialized for Volta. Sources: [cuBLAS API](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html), [CUTLASS functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html), and [grouped-GEMM specialization](https://github.com/NVIDIA/cutlass/issues/888). Inventory existing calls and traces before replacing work already using an appropriate library; these options still require measured TensorFold benefit.

Keep handles/descriptors and bounded per-stream scratch persistent; tune outside serving. Key choices by shape, layout/strides, alignment, numerics, workspace, architecture and exact library build. Concurrent scratch must be separate or provably serialized. Set cuBLAS stream before rebinding workspace; stream changes reset it. Budget lazy residency/scratch before admission; larger workspaces require the resource-envelope protocol.

Compare fused packed-weight tile decode with bounded FP16 staging/library GEMM where row reuse amortizes staging. Count reads/writes/launches and peak memory; no full-model/KV expansion for BLAS. Specify FP32 accumulation/reduction to prevent silent numerical changes. Select expert kernels from actual routed rows without per-layer host readback merely to populate library calls.

Use cuDNN or ordinary cuSPARSE only for a matching operator with measured complete-step benefit. cuBLASXt's competing multi-GPU distribution policy is outside fixed TP8. cuSPARSELt requires SM80 or newer structured-sparsity hardware; cuBLASDx 0.6.0 removes Volta. Neither is a baseline dependency. Sources: [cuSPARSELt support](https://docs.nvidia.com/cuda/cusparselt/index.html) and [cuBLASDx release notes](https://docs.nvidia.com/cuda/cublasdx/0.6.0/release_notes.html).

## 16 Prefill and mixed-work scheduling

Use chunked prefill, bounded workspaces, and continuous admission into scheduled work. Preserve causal order and all 36 recurrent layers' transitions within each sequence. Independent sequences may share an execution batch; parallel recurrent scans within one sequence require a semantics-preserving algorithm, not ordinary token batching.

Preserve the API's input-commit and prefill-cancellation contract. Track the admitted input range, its committed/pending boundary, target-consumed position, and draft-valid position through every chunk. Request-scoped cancellation discards the slot after work retires. Retained-session cancellation must retain the specified input progress or restore its checkpoint consistently across transcript, KV/indexer, recurrent/PLE, draft, sampling/output bookkeeping and multimodal state. Whole-append rollback requires protected, budgeted pre-append recovery state, including overwritten draft-window entries; a KV-length reset is insufficient.

Resumption validates the canonical expanded token prefix, immutable media identity, processor and position state, then continues from the recorded boundary; submitted input is not automatically fully cached. Test cancellation/resumption at chunk, pooling, page and media boundaries. Implement the selected retention and input-commit contract explicitly; scheduling changes cannot alter it.

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

Initially qualify one resident FP16 encoder/projector copy on one fixed GPU, choosing the rank by topology, worst-rank headroom and mixed-work latency. Qualified sharding of that same encoder may replace this placement after implementation and measured benefit; do not introduce additional encoder replicas for throughput. Publish its complete placement/resource plan before admission. Placement stays fixed during serving. Route projected embeddings through the target TP8 input contract; eight target ranks do not require eight visual copies.

Adapt bounded CPU preprocessing workers, legal image/frame microbatches, attention tiles and language-prefill quotas at coordinated boundaries. Independent images or valid temporal/attention segments may batch together; arbitrary spatial splitting must not remove noncausal attention edges. Use exact bounded attention tiling, preserving segment boundaries, positional interpolation and merger order, rather than an unbounded score tensor or an approximation.

Charge placed vision weights, encoder activations, projected embeddings, staging, graph pools and communication for permitted overlap. Encoding concurrency may be lower than resident session capacity. Queue visual work fairly while preserving every full-context commitment; the largest supported media work unit must fit at full residency. Idle visual execution does not release promised reserve for additional guaranteed contexts.

Use bounded host/pinned buffers and immutable bounded replay sources or reproducible equivalents. Release embeddings after target/speculative conditioning unless explicitly retained/budgeted; a `[262144,2560]` FP16 archive costs 1.25 GiB/copy and is not default. Bound media caches by content, processor/settings, vision weights/numerics, grids/order; capacity must hold on misses. Target-prefix reuse also requires Section 9's complete state/profile identity.

#### Multimodal speculation

Apply Section 12's actual next-position embedding contract for MTP, including visual lookahead and tentative chunk-tail state. Qualify splits inside visual spans, timestamp/delimiter transitions, multiple media items, cancellation/resumption and zero-proposal maintenance; placeholder-token lookup is insufficient.

DFlash consumes the target's visually conditioned completed-layer HC means. Its draft RoPE uses scalar absolute sequence-row positions, separately from target M-RoPE coordinates/deltas. Visual rows count toward the bounded conditioning window. Keep draft materialization, correction, rollback and warm maintenance aligned while the verifier retains target positions. This feature contract needs no separate draft vision encoder. Source: pinned [DFlash model](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/models/dflash.py) and [worker position materialization](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/speculative/dflash_worker_v2.py). Text-only qualification cannot establish multimodal correctness, capacity or acceptance.

## 17 Global adaptive controller

Use a small deterministic controller with offline-qualified benchmark profiles and lightweight runtime observations; no complex online learning system is required.

### Two control timescales

| Timescale | Decisions | Safe application point |
|---|---|---|
| Every scheduled step | Eligible requests, proposal length, verification rows, prefill/visual work quotas, legal media microbatch, kernel family and existing graph bucket | Globally agreed batch boundary |
| Sustained workload change | Attention degree, page placement, bounded graph replacement, resource-envelope changes | Qualified migration or quiescence protocol respecting admission commitments |

FIFO admissions and qualified idle-state transfers follow the policy below; they do not trigger routine generation-time swapping. The coordinator issues one plan and epoch for all TP ranks. Ranks must not independently change proposal lengths, graph shapes, communicators, or collective participation.

Observe active sequences, request multiplicity, history lengths, selected indexer work, expert group sizes, prefill/decode mix, acceptance, queued visual patches/frames, encoder/transfer costs, pending arrivals, and memory pressure. Active count selects a profile family, but a nearly full history can cost more than a short history in the same count bucket.

### Required concurrency coverage

| Actual active sequences, one per initial request | Required behavior |
|---|---|
| 0 | Retain promised state and a ready execution profile; perform permitted quiescent maintenance |
| 1, 2, 4, 8 | Measure low-load latency/throughput, ownership degrees, and useful speculation widths |
| 12, 16, 20, 24, 28, 32 | Measure saturation, verification splitting, prefill interference, fairness, and speculation profitability wherever capacity permits |
| Other counts | Use a correctly masked qualified covering profile; measure every integer count and investigate performance breakpoints |
| Above 32 | Extend the same policy when shorter contexts or other qualified settings permit |

For each released context/precision/method, measure every integer active count through capacity. Detailed long-context/transition tests cover named thresholds, endpoints and observed breakpoints; test 18 when capacity is 18 and extend above 32 when feasible. Masked covering graphs prove safety, not optimal intermediate throughput; add buckets only for measured end-to-end gains.

Handle an immediate jump from one active sequence to the configured maximum through the canonical path, using bounded smaller batches when necessary. Hysteresis must never permit oversized batches on undersized buffers or unsupported graphs.

Costly placement changes require hysteresis and a minimum expected benefit over realistic residence time in the new regime. When evidence is insufficient, keep current placement and a safe batch plan. Avoid repeated large-history migrations around oscillating thresholds.

Keep statistics asynchronous where possible; do not synchronize the whole device each step merely for controller input. Select profiles using complete-step elapsed time, committed tokens, and latency, including draft maintenance, communication, packing, recovery, and amortized migration.

### FIFO admission and tiered retained sessions

**Default policy.** Use one bounded FIFO admission queue with continuous batching. Admit the oldest eligible request when a qualified GPU slot/execution envelope is available. Hand off released request-scoped slots or successfully suspended idle state; GPU-only retained guarantees remain reserved. Runnable generation is not routinely preempted for tier rotation. FIFO orders admission, not completion; up to the qualified GPU-resident maximum can execute together. Fresh queued work has no computed KV to offload: retain only bounded input/media sources and metadata.

FIFO avoids repeated full-history transfers but guarantees neither equal wall-clock active time nor bounded waiting behind arbitrarily long work. Preserve administrator context/reply/work limits and cancellation policies; never truncate accepted generation to advertise fairness. GPU-only operation supports bounded new-request queuing; qualified tiering additionally lets idle retained conversations release GPU slots while preserving state in RAM or administrator-selected local storage.

**Integration contract.** Fresh work, prefix hits and retained resumes use one admission/ownership path. Background-priority or replay-preemption changes require a separately qualified policy. Reusable-cache eviction may trigger ordinary prefill; required retained checkpoints preserve their sole authoritative state.

Related primary references inform the design without supplying drop-in SM70/qwen4exp support: [vLLM V1](https://docs.vllm.ai/en/latest/usage/v1_guide/) and its [optimization guide](https://docs.vllm.ai/en/latest/configuration/optimization/) describe recomputation-based preemption; [SGLang HiCache](https://docs.sglang.ai/advanced_features/hicache_design.html) and [hybrid-state work](https://www.sglang.io/blog/hicache) cover hierarchical state recovery. [Fair-serving research](https://www.usenix.org/conference/osdi24/presentation/sheng) distinguishes cost-aware service fairness from request ordering; its bounds do not automatically apply to this FIFO policy.

#### OpenAI-compatible API and internal session identity

Use standard OpenAI-compatible endpoints, schemas and continuation fields; no TensorFold session/revision headers, private client session protocol or fixed GPU-lane identifier is required. Clients use the local base URL, normal authentication and model IDs from `GET /v1/models`. Pin the supported API/SDK schema and publish/test its endpoint/feature matrix, including each media content type's exact decoder/processor mapping. Required video understanding stays a release gate: distinguish native video/file transport from a standard image-frame/timestamp workflow, and separately qualify the latter if used. Neither invent a private video field nor silently replace Section 2's pinned video semantics to claim API compatibility. Compatibility means correct wire and lifecycle behavior for supported operations, not a claim that this model/server implements every OpenAI modality, hosted tool or product feature.

| Client interface | Conversation and cache behavior |
|---|---|
| `POST /v1/chat/completions` | Client sends `messages`; automatically match compatible rendered prefixes. A response `id` identifies that completion, not a stable chat or hidden append target. |
| `POST /v1/completions` | Implement the standard full `prompt` contract and automatic prefix reuse; do not infer conversation identity from equal prompts. |
| `POST /v1/responses` | Support the standard `previous_response_id` chain; a new response returns a new `id`. Resolve retained logical items and render the actual next input before selecting model state. |
| Standard Conversations API | For a stable conversation handle, implement/qualify `/v1/conversations` and its standard item lifecycle, then pass its `id` through Responses' `conversation` field. Do not substitute a proprietary field. |
| Internal identities | Owner-scoped logical sequence IDs, mutation/checkpoint generations and GPU slots protect state and placement; prefix/page keys identify reusable computation. These remain distinct. |

OpenAI's [conversation-state guide](https://developers.openai.com/api/docs/guides/conversation-state) defines manual Chat history, response chains and durable Conversations IDs. Reject simultaneous `conversation`/`previous_response_id`. Responses `instructions` is request-scoped and never inherited through `previous_response_id` ([instruction scope](https://developers.openai.com/api/docs/guides/text)); distinguish it from historical input items. Rerender changed/omitted instructions before prefix selection; obsolete instruction state cannot be reused. Repeated `user`, `metadata` or `prompt_cache_key` never authorizes hidden Chat history. A cache hint aids lookup only within its allowed domain; validate content/profile and authorization.

Implement and qualify Responses create/retrieve/delete and standard `previous_response_id` continuation through the shared generation engine. Standard Conversations support requires its own complete API contract before advertising stable client conversation IDs: durable logical-item storage, ownership, create/retrieve/update/delete/items behavior and bounded capacity. Reject `conversation` explicitly until that contract is implemented. Storing a response or conversation retains logical API items, not an automatic permanent GPU slot or full KV checkpoint. Budget those records and parent/media dependencies separately; an explicit computed-state retention promise additionally needs Section 17's backing claim. A valid item history can rebuild a reusable-prefix miss, while required queued/retained inference checkpoints remain protected. Logical API identity can survive computed-cache eviction; each rebuilt execution still validates the requested profile and actual rendered prefix.

Serialize mutations of a shared conversation under an internal owner/generation transaction, with bounded FIFO waiting or a documented standard-shaped conflict; no external revision handshake. Response-chain branches require independent sequence/state claims and must not mutate their parent. Distinguish a new turn from resumption of the same operation: initialize new-request sampling/constraints/output state for the former and preserve it for the latter. Validate principal access on every ID lookup. Unknown/deleted/expired references return explicit API errors, never a new unrelated conversation. Honor `store` and declared retention/deletion semantics; reserve response/item/manifest/media bytes and dependency lifetimes before promising retention. Apply the documented deletion semantics to descendants and independently retained items: preserve needed dependencies only where allowed; otherwise mark affected references unavailable explicitly. Never leave a dangling continuation or override standard deletion behavior.

Standard `DELETE /v1/conversations/{conversation_id}` deletes the conversation object **without deleting its items**. Item/media references therefore have separate retention and deletion lifetimes; do not cascade that operation into item or sole required-checkpoint destruction. Source: [official Conversations deletion contract](https://developers.openai.com/api/reference/resources/conversations/methods/delete).

Bound logical response/item storage by count, serialized bytes, physical-memory overhead and declared durability/retention. An in-memory response-chain cache cannot satisfy durable Conversations merely by returning standard-shaped IDs. Bind an authenticated principal to every lookup/storage operation before claiming multi-principal isolation; a shared API key defines a shared trust domain. Generation limits, logical API retention and retained-compute guarantees remain independently accounted contracts.

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

Among GPU-admitted runnable sequences, rotate bounded prefill/decode opportunities fairly using the canonical scheduler, including long prompts, heterogeneous acceptance and media bursts. Give every ready sequence progress; bound prefill chunks/visual units and consecutive decode rounds so one cannot monopolize execution. Equal turns do not imply equal committed tokens/s or latency. Measure service and waiting lag, and bound outstanding requests/bytes per principal so queue flooding cannot consume the entire service. Use no complex new scheduling framework. If future requirements demand preemption of long-running jobs for a tighter waiting bound, separately qualify that policy and its throughput/transfer tradeoff; do not silently introduce it into FIFO.

#### Capacity and commitments

Publish separate limits for GPU-resident sequences, retained logical sessions across all tiers, and accepted waiting requests/input bytes. Tiering increases retention capacity, not simultaneously executing GPU capacity or compute throughput. The Section 7 planner still acquires each GPU-admitted/restored sequence's full-window worst-rank commitment and required execution envelope. Queue length must not select an oversized GPU batch.

Fresh request-scoped waiting work reserves bounded input/media/queue resources, not hypothetical KV it has not computed. For every tiered retained session, additionally reserve a RAM or storage destination sufficient for its complete future context, continuation/recovery state and metadata, even while GPU-resident. No reliance on a short current history, unreserved future SSD spill or presumed shared prefix. Release its GPU entitlement only after complete checkpoint publication and reader retirement; preserve the logical context/lifetime and backing claim. GPU-only retained sessions keep their GPU guarantee. Before promising FIFO progress, prove that eligible work can obtain a recyclable slot within the declared waiting policy: tiered idle state can demote, active work can finish, or a known close/expiry can release capacity. If indefinite GPU-only reservations occupy every slot, do not admit more work with a progress guarantee; reject it or explicitly report blocked/bounded waiting under the API policy. Never force eviction/deletion or quietly overcommit.

Bound host RAM, storage, pinned staging, transfer descriptors, open files, queued I/O, media replay, old checkpoint generations/replacement and suspend/restore overlap. Subtract OS, PLE page cache, decoder workers and ordinary host obligations from RAM budgets; do not rely on OS swap. RAM need not all be pinned. Preserve the required 4 GiB/2 GiB GPU protections. Reserve progress headroom: saving/releasing an idle victim must not require first restoring another session. Initial save-release-restore can be serialized when overlap cannot fit; no second full GPU history is required. Section 11's ownership-migration cap does not implicitly cover tier transfers; cap/account for their combined permitted overlap.

#### Complete state and transfer protocol

At request completion/cancellation, retire issued work and detach bounded host output/recovery obligations from the GPU slot. Release request-scoped GPU commitments promptly; no mandatory offload is needed if the state is disposable. If the session is explicitly retained, preserve its complete required checkpoint before freeing the last GPU copy. A queued same-session continuation may take its valid resident state directly when FIFO selects it; otherwise suspend idle retained state as needed to admit older work. Keep idle state warm only while capacity is spare; a reuse prediction or grace period cannot indefinitely delay an eligible queued request. Use write-back at suspension, reusing verified backing and writing only changed state under the integration contract below. All attention still executes with required materialized history on GPU; this is whole-session suspension, not CPU attention, expert paging, raw-indexer archiving or managed-memory oversubscription.

Use one typed inventory covering every continuation/recovery field in Sections 9, 12 and 16, classified as required retained state, qualified bounded rebuild input or disposable scratch. Include deferred HC/next-embedding dependencies, RNG/constraint/output bookkeeping, undelivered committed output, media/replay references and protected pre-append rollback versions. Necessary host state may remain resident with explicit ownership. Add no unnecessary full feature/embedding archive. Shared model/vision weights, PLE tables and execution scratch are not per-session payload.

1. The coordinator selects an idle victim/destination, reserves backing/staging and issues a transfer epoch. Serialize suspension with mutation, cancellation and ownership migration. Resolve/retire issued work at its method-defined consistent boundary and drain affected readers/writers/transfers through stream/events.
2. Preserve legal pending anchor/bonus and distinct committed/target-cached/draft-valid lengths; do not generate or consume RNG just to equalize them. Copy only valid packed rows and required state/tails in bounded coalesced chunks without requantization; exclude or initialize serialized padding and use qualified NUMA-local pinned staging and asynchronous host I/O. No GDS dependency.
3. Validate sizes, page ranges, checksums and a complete all-rank/component manifest. Publish one checkpoint generation only after every required piece succeeds; storage commit ordering must satisfy any explicitly promised durability. Until then GPU state remains authoritative. A failed save retains it and aborts eviction.
4. Retire readers/callbacks, detach or transfer reusable-page references, invalidate the GPU slot generation, then release its state/commitment and wake FIFO admission without waiting for a polling tick. Keep any leased shared pages alive under their remaining owners. The suspended checkpoint is required retained state, not an evictable best-effort prefix entry.
5. At FIFO admission of a resume, validate profile/schema and checked descriptor/size bounds before allocation. Reserve destination page classes/state, its complete future GPU window and bounded I/O/execution resources; verify checksums and restore every required materialized component before execution. Translate logical pages to the selected GPU slot/qualified owner mapping, validating worst-rank indexer/KV balance for arbitrary cohorts. Never deserialize device pointers or reinterpret ring slots as positions.
6. All ranks acknowledge readiness before GPU-ready publication. Partial/failed restoration releases only tentative destination resources after readers retire; the lower-tier checkpoint stays authoritative. Track GPU-authoritative mutable versions after resumption; verified immutable backing remains reusable, while stale mutable versions cannot satisfy a current checkpoint. Charge backing/old generations until safely retired. A new turn resets request-specific generation state; continuing an interrupted request preserves it.

Use byte-preserving continuation initially; KV-only restore, silent method changes or replay from an incomplete recurrent/media boundary are invalid. Recomputing required continuation needs independent complete-state, media, numerical and sampling qualification. A reusable-prefix miss uses Section 9's ordinary qualified prefill, not protected-checkpoint loss recovery. Shared prefixes require reference counts/lifetime guarantees; capacity assumes none.

#### Queue and cache integration

Use one canonical catalog entry per page/state object with content/profile identity, generation, validity, owner/references, leases, dirty status and per-tier residency/backing. Account separately for permitted physical/staging copies. Share the allocator, lookup catalog and bounded transfer executor between prefix reuse and queued sessions; implement them together within the same serving path. Logical session/request manifests reference these objects and their small private mutable state. Distinguish protected active/queued continuation from reclaimable reusable-prefix residency: LRU eviction or failed best-effort spill must never delete a required queued object's last authoritative copy. On request completion, only explicitly retained sessions stay protected; released state may become bounded reusable cache without retaining an accidental session guarantee.

Reuse verified immutable backing on repeated suspension: transfer only newly appended/dirty pages, partial-page replacements and changed recurrent/draft/request state. A mutable GPU tail never overwrites a backing object referenced by an older checkpoint or fork. Publish a new manifest over immutable objects only after all referenced pieces validate; retire superseded objects after manifests, readers and transfers release their references. First writes and cache misses still require complete transfers; budget/measure that cold path and report actual dirty versus reused bytes rather than assuming delta savings. Metadata, reference maps, hashes, old checkpoint generations and copy-on-write peaks belong in the ledger.

Queue admission resolves the dependency path once: fresh work takes a compatible leased GPU/RAM/storage prefix hit or performs ordinary prefill; an exact queued continuation restores its authoritative manifest; an already GPU-resident valid continuation needs no transfer. A GPU-ready sequence omitted from a microbatch remains GPU-resident and does not become a paging victim. Stage bounded lookahead reads for the oldest eligible requests into host buffers when they can hide I/O; acquire the full GPU claim before restoring GPU pages. Prefetch must neither leapfrog FIFO nor evict active state, consume reserved progress buffers or expand unboundedly with queue depth.

Coalesce duplicate reads of the same immutable object and share completed buffers under reader leases; cancellation releases only that consumer's references. Do not hold the scheduler/catalog lock across disk I/O, DMA waits or serialization. Separate bounded I/O progress from launch/communication service, and rate-limit copies so PLE lookups, media processing and full-load decode retain their qualified service envelope. Do not build a second whole-session host image merely to send it to disk when bounded streaming is sufficient.

Fresh prefill, prefix hits and authoritative resumes follow the same transaction above. Record transition reason/ticket and prevent new work before quiescing. Hits change preparation cost, not FIFO priority, reservations, publication, cancellation or numerical/lifecycle gates; all paths use the same target/speculative execution.

#### Transfer cost, API and failure behavior

Measure simultaneous eight-rank D2H/H2D/storage paths under shared PCIe/NUMA and PLE/media contention; NVLink bandwidth is not host bandwidth. Section 8's native-context MTP growing history alone is 3.656250 GiB/session across the box for Q8 or 1.929688 GiB for iso4. Out-and-back therefore transfers at least 7.312500/3.859375 GiB before state, rounding and storage I/O. Use sustained measured rates and actual per-rank bytes, never a fixed millisecond restore promise; these costs motivate FIFO rather than routine rotation.

Administrators set bounded accepted/waiting counts/input bytes, per-principal limits, retention tier and RAM/storage budgets/path, and service/queue timeout limits. Resolve transfer chunks/overlap from qualified measurements. Begin with RAM suspension, then qualify optional storage. Quota reductions preserve accepted claims; disabling/changing a tier or numerical/model profile drains or safely relocates affected checkpoints, including suspended sessions. Queue saturation returns a specific response; deadlines/cancellation retire queued inputs without allocating GPU state. A hard wait SLO requires admission/workload bounds and measured completion/restore limits; FIFO alone provides none.

Expose queue order/age, lifecycle counts, backing bytes, GPU residency, restore delay, traffic and blocking reasons through the standard request/cancellation and administrative status surfaces. Preserve standard API semantics while waiting; implement missing lifecycle operations directly. Resumption publishes canonical output without duplicates/retractions; define pause/heartbeat/idle timeouts without fake tokens. Bounded slow-client backpressure/cancellation must prevent indefinite GPU monopolization. Retire transfers/generations before cancellation/expiry reclamation.

On RAM/storage exhaustion, partial writes, corruption or I/O timeout, preserve an authoritative source, bound retries, stop affected admissions and expose failure without discarding context or blocking unrelated GPU work. A missing/corrupt sole checkpoint is an explicit session failure, never silent recomputation. RAM snapshots are not process-crash durable; local storage alone promises no restart recovery. Enforce ownership, file access and close/expiry cleanup. Normal queue/tier operations require no container/worker/CUDA-context restart.

## 18 CUDA graphs and communication

Initialize communication groups, kernel families, descriptor bounds, and maximum buffer capacities for the validated effective-profile envelope before use. A larger profile or ceiling requires new bound validation even if its bytes fit. Keep weight TP8 fixed. Build a bounded graph set and retain uncaptured execution of the same plan, kernels, and allocation contracts within the canonical serving path.

Prefer replay when saved submission time outweighs padding, graph memory and update costs; graphs do not accelerate kernel bodies. Use bounded instances, stable arenas and device-updated inputs/counters/masks. Remove CPU dependencies from capture or keep that logic outside; preserve Section 12's graph-safe RNG/state ownership. Warmup must not mutate live sessions or their RNG. Key validity by the complete effective profile and captured assumptions, not batch count alone; record breaks, recapture causes and replay coverage. Capture/recompile follows Section 19's coordinated resource protocol, never unbounded per-token cache-miss handling.

For PyTorch capture, use a non-default CUDA stream; the graph context manager may manage that stream, while the raw `CUDAGraph` API requires the caller to do so. Perform eager warmup on a side stream and join the required readiness dependencies before capture/replay. Keep warmup storage and module/library initialization inside the declared preparation budget. Source: [PyTorch CUDA graph constraints](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).

Use a small measured set of masked row/token and legal visual-shape/segment buckets with stable indirection; do not capture every combination of session count, context, proposal width, prefill size, media shape, codec, and placement. Charge padding's actual compute and storage.

### Graph validity

Stable page-table pointers alone do not establish reuse across layout changes. Validate captured direct pointers, scalar arguments, launch dimensions, collective counts, participating ranks, stream dependencies, allocator pools, and object lifetimes. All captured assumptions must remain valid.

Mask inactive/padded rows from KV/indexer writes, recurrent/convolution state, draft state, RNG, stop processing, and committed output. Validate session generation to prevent stale writes into reused slots; asynchronous media completions obey the same generation and reader-retirement rules.

Charge graph-private pools, persistent inputs, and live instances in the ledger. Share scratch only when replay ordering proves nonoverlapping lifetimes. Source: [PyTorch CUDA graph memory/lifetime documentation](https://docs.pytorch.org/docs/2.14/notes/cuda.html); qualify the actual framework build under Section 2.

### Collective ordering

Use NCCL for normal TP8 collectives. Initialize full-rank and bounded attention subgroups before serving with topology-aware rank mapping. Use AllReduce when every participant needs the result, ReduceScatter when the next operation consumes shards, and AllGather where replication is needed. Immediate ReduceScatter-plus-AllGather does not itself eliminate communication. Bounded migration uses matched NCCL P2P or qualified peer copies. Group mutually dependent two-sided sends/receives with `ncclGroupStart`/`ncclGroupEnd`, matching endpoint counts and datatypes within the globally agreed communication phase. Sources: [collective semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) and [P2P operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html).

Start with NCCL's topology-aware selection. Measure small decode-message latency, larger prefill traffic, and GPU contention during communication/computation overlap. Tune counts/bytes, scheduling, and graph integration before adding frameworks. Do not force one algorithm/protocol across message sizes without evidence, recreate communicators on active-count changes, or use changing process environment variables as runtime controls. Device-initiated NCCL, MSCCL++, or NVSHMEM needs a measured benefit justifying its additional contracts. NVLS requires newer NVSwitch/Hopper hardware and is unavailable on this V100 mesh ([NCCL controls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)).

All ranks agree on captured/uncaptured execution and compatible collective order. The issued plan fixes groups, ordering, counts, and empty contributions: cancellation, empty shards, or partial migration cannot independently suppress a worker's collective. Coordinate migration, visual-input distribution and serving with explicit streams/events and a globally ordered plan; do not inject rank-local copies or visual broadcasts into captured collective schedules. Source: [NCCL CUDA graph requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html).

Keep default NCCL graph stream ordering and mixing support. `graphUsageMode` is 0 for no graphs, 1 for one graph, and 2 for multiple graphs or captured/uncaptured mixing. A separately qualified `NCCL_GRAPH_STREAM_ORDERING=0`/`graphStreamOrdering=0` experiment must use compatible mode 0/1 and prevent environment overrides from forcing mode 2. Serialize all NCCL device operations sharing a GPU across communicators, graphs and captured/uncaptured paths using stream/event dependencies. With mixing disabled, same/split-shared communicators cannot participate in parallel graphs; a noncaptured call must wait for their outstanding graph work to complete on device, irrespective of stream ordering. Host-call serialization alone is insufficient. Sources: [ordering and mixing requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html), [configuration types](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html), and [graph contract](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html).

Separately serialize host NCCL issuance for the same communicator and for communicators sharing a GPU; one host thread issues every operation within a group. Device ordering does not establish [host-thread safety](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html).

Capture/replacement follows the actual framework's quiescence rules. PyTorch 2.14 permits one capture at a time per process and prohibits noncaptured CUDA work on any thread in that process during capture. Quiesce visual, transfer, checkpoint and other CUDA-issuing threads in that process too; another host thread is not an exemption. Use uncaptured execution between capture operations and pause the process's affected GPU work while capture is active. Source: [PyTorch CUDA graph constraints](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).

## 19 Configuration changes without restart

Expose configured, pending, and effective values separately. Normal adaptation and supported administrator changes must not restart containers, workers, or CUDA contexts; worker replacement is not a live update.

Drains cover every affected sequence, including retained RAM/storage checkpoints: request-scoped work must finish/cancel and retire; retained sessions follow close/expiry/reconstruction rules. Zero GPU residency is insufficient while suspended sessions retain incompatible state. Zero activity leaves retained history, draft, recurrent/position/media and recovery obligations intact. No retained sessions does not authorize unloading request-scoped or in-flight state.

| Adjustment | Implementation policy |
|---|---|
| Scheduled batch, prefill/visual quota, legal media microbatch, bounded CPU worker quota, supported proposal length, existing graph/kernel choice | Apply at a coordinated batch boundary within allocated limits |
| KV ownership and attention degree | Use bounded byte-preserving migration and a published mapping epoch |
| Resident-sequence ceiling decrease | Restrict new admissions; preserve existing sequences and context commitments |
| Resident-sequence ceiling increase | Require the complete maximum-context and execution-envelope proof |
| Larger context with unchanged effective positional transform | Activate for all affected GPU/tier-retained sequences together only after enlarging GPU/backing claims as applicable and validating qualified positions, state and execution bounds; otherwise keep pending until the affected profile drains |
| RoPE/YaRN algorithm, frequencies, factor, original-reference length or amplitude change | Initially drain affected GPU and tier-retained state and retire dependent graphs/caches; validate and activate the prepared profile in process. Never change from active-count adaptation or reuse incompatible state (Section 6). |
| Smaller context | Preserve existing windows; keep pending until the homogeneous profile, including suspended retained state, drains |
| Codec, target/visual weights, processor/numerics, structural encoder placement, draft checkpoint or speculative method | Use a qualified in-process transition, or drain all affected GPU/tier-retained sequences and reinitialize in process |
| Retention tier/path or lower queue/backing budgets | Preserve accepted claims; drain or safely relocate required checkpoints before retiring a tier/path (Section 17) |
| No supported in-process transition | Fix the setting at its qualified maximum-concurrency value; expose no live control |

Use one homogeneous context/quality resident profile; simultaneous profiles require separate budgeting.

Structural drains stop admissions that could indefinitely extend the old profile. Define resume/closure behavior while existing sessions finish or close under their retention contract. An idle retained session, including a suspended checkpoint, can keep the change pending: report its blocking condition or resource without forced deletion, restart, or a promise to switch at the next zero-active instant.

GPU page moves preserve bytes. Q4-to-Q8 cannot recover discarded information, and a new speculative method may need unretained features/state. Use qualified reconstruction or drain affected sessions. Warm zero-proposal operation follows Section 12; activity adaptation cannot change method, checkpoint or KV precision.

Target/runtime-weight changes initially drain affected GPU/tier-retained sequences, retire graphs/readers/state and load a prepared replacement. A saved checkpoint cannot be restored under incompatible replacement weights or positions. Saved API items alone do not pin a GPU profile: keep their source identity and use them with another requested profile only through a qualified render/rebuild path, without treating old computed state as compatible. Protected retained compute claims still require the drain/reconstruction contract. If old/new layouts cannot coexist in bounded memory, use an empty-resident in-process transition; uninterrupted service is not promised. Queue or report pending per service policy. Bind queued work to its complete requested model/context/RoPE/KV/method/tokenizer/processor/media profile; reject or retain incompatible work pending. Retained media cannot be reprocessed under changed settings without a qualified transition.

Publish a replacement/capacity only after all ranks finish loading, allocation, target/draft binding, bound initialization and required library/kernel residency. Retire/rebuild affected graphs and tuning plans. On recoverable preparation/load failure, coherently retire partial state and expose failed/pending, never unready capacity. Releasing the old model ends its service guarantee. Retry/reload a prepared profile in process without two full resident models.

## 20 Host memory and deferred architecture alternatives

The selected PLE tensor has 28,800,138,240 IQ4_NL payload bytes in the 26.822219-GiB second shard; validate the pinned tensor identity/offset at import. Rank file mappings can share physical pages; eight mappings do not imply eight physical copies. Residency is evictable; measure actual NUMA placement.

Profile cold/warm lookups, page faults, CPU execution, NUMA access, and bounded host-to-device transfers during prefill/decode. Preserve all sixteen bigram/trigram hashes, offsets, resets, IQ4_NL decoding, and PLE convolution. Batch lookups and eliminate proven duplicate TP-rank work, routing common results where beneficial. Shared file pages do not eliminate duplicate lookup/transfer work; PLE can dominate low-load latency after KV deduplication.

Budget host media decode, normalized pixels, optional retained projected embeddings, immutable replay sources/caches and pinned transfer stages separately from PLE and history. Bound CPU worker concurrency, queued decoded media and every buffer lifetime; do not allocate maximum video working sets for all resident sessions. These media resources do not provide attention-KV capacity.

Keep raw-indexer RAM archiving outside this profile. Compact indexing has no growing raw array; charge its bounded pending ring, protected tails and execution staging explicitly. A full-raw archive's ring/staging budget does not describe the compact construction.

Section 17 defines qualified idle-session RAM/optional-storage checkpoints with FIFO restoration. Per-step host paging of executing attention KV and retained expert weights remains outside scope and cannot supply GPU-resident capacity or silently rescue overcommitment. The eligible structural comparison is topology-aware TP8 versus TP4 × PP2 with one distributed target. Independent TP4 model replicas and the proposed TP4 attention/dense plus EP2 layout that duplicates shardable nonexpert weights are excluded. A nonduplicating expert-placement change remains eligible only with an exact inventory and measured benefit. The TP8-specific contracts in this document govern the implementation baseline until an alternative receives separate qualification.

The pipeline candidate requires stage-specific weight/state plans, full HC transfer, transactional scheduling and a worst-rank resource proof. Keep the selected target embedding at its owning stage and serve MTP/DFlash lookups through bounded vector transfers; the final-stage target head can remain shared with its local draft. Include lookup latency in complete speculative rounds and keep the owner service schedulable during feedback waits to prevent a dependency deadlock. Add no persistent target embedding or vision copies. Changing TP/PP placement is a separate in-process weight-and-state transition, not a cache-only active-count control; retain the selected maximum-concurrency layout if that transition cannot meet the no-restart contract.

## 21 Observability and qualification

### Runtime receipt

Publish a compact machine-readable effective-profile/capacity receipt and user-facing limiting-resource explanation. Include:

- Exact target/MTP/DFlash/visual/processor/codec IDs, hardware/runtime manifest, immutable resolved RoPE/YaRN and companion position manifests/epochs, native/reference versus configured/qualified maxima, context, media envelope/placement, configured ceiling, computed resident capacity, and qualification status.
- Standard response/conversation references, internal request/session generations and lifecycle status; separate request-scoped/retained, GPU-admitted active, FIFO-waiting, idle, suspending/suspended/restoring and scheduled sequences; queue order/age, per-principal use, tier budgets/bytes, checkpoint generations, slot-release-to-admission delay, transfer/restore time, I/O failures and blocking reasons; text/visual prefill rows, frames/pixels, cache hits, maintenance/replay and verification work.
- Proposal policy, draft-valid state, acceptance distribution, recovery/replay work, and committed generated tokens.
- Timestamped per-rank weights, history pages, state, graph pools, language/visual workspaces, external allocations, physical free memory and protected commitments; bounded host/pinned/media-store use and queue occupancy.
- Placement degree, mapping epoch, outstanding migration bytes, transient occupancy, migration time, and admission restrictions.
- Useful text/visual prefill and committed decode rates, media decode/encode/transfer time, TTFT, inter-token latency, per-request service/fairness and error/cancellation counts.

Separate startup/current readings, measured allocations, logical reservations, and planned maxima. Do not add allocated to reserved bytes or count tentative/draft tokens as generated throughput.

### Qualification matrix

| Axis | Required cases |
|---|---|
| Artifacts, precision and method | Every released target/runtime-weight pairing and supported target KV with none/MTP; every DFlash pairing/draft format; selected NVFP4 import/conversion/activation semantics |
| Metadata and target frontend | Complete per-role resolution from sidecars/embedded/pinned-source/explicit inputs; missing and conflicting fields, invalid typed overrides and unsupported capabilities reject before allocation. Verify exact token IDs, NFC/composed/decomposed Unicode, pretokenization/ByteLevel, disabled automatic BOS/EOS and cleanup, CONTROL/UNUSED handling, full/skip-special decoding, tools/reasoning/media rendering and generation defaults. MTP loading/enabling/disabling must preserve target-rendered bytes, token IDs, media rows, generation prefix and stop policy; actual target embedding/head aliases replace the companion's duplicates. |
| Context and RoPE/YaRN | Native and every released extended positional profile; short/intermediate/near-full histories, native/new-limit boundaries, coordinate-versus-row bounds, invalid/nonfinite profile rejection, partial-RoPE amplitude parity, Q8/Q4 quality, required MTP and separately qualified DFlash, media, profile identity and in-process changes; input plus output fits the allowance |
| Concurrency | Integer sweep from 1 through capacity; detailed thresholds, exact endpoints and measured breakpoints; burst from 1 to maximum |
| Retention | Request-scoped release and intentional retention; one resident/one active, all resident/one active, all resident/all active; partial/near-full histories; supported multi-sequence accounting |
| FIFO and tiered retention | Fresh/resume order, bounded eligibility, no warm-resume jump, fair admitted progress, completion/idle slot handoff and immediate wakeup, no routine runnable preemption; above-GPU-capacity backlog, idle demotion and RAM/optional-storage restore; complete-state parity for Q8/iso4, MTP/DFlash, native/extended positions and media; arbitrary owner/slot phases, cancellation/expiry/load-change races, quota reduction, no transfer-progress deadlock, all-slot idle GPU-only blockage versus guaranteed FIFO progress, exhausted/unhealthy tiers, corrupt/partial checkpoints and bounded retries |
| OpenAI API/session identity | Unmodified SDK Chat/Completions/Responses calls, full-history automatic reuse, stored response chains/branches, request-scoped instructions changed/omitted across turns and resulting prefix misses, owner isolation, reference expiry/deletion, store/usage/errors and streamed/nonstreamed lifecycle; no custom session headers. Separately qualify standard Conversations/item APIs and conversation/previous-response exclusivity, concurrent mutations, durability/quotas and cache-independent logical retention, record/compute-claim separation and deleted-parent descendant lifetimes before advertising stable conversation handles |
| Prefix/cache integration | Exact rendered/media/profile identities; no-hit/partial/full-prompt hits; new-turn versus interrupted-request RNG/constraint state; earlier forks and immutable longer branches; all hybrid/draft components at a common valid cut; COW/reference/lease races, concurrent load coalescing, dirty-page reuse, cache eviction during save/restore, protected-checkpoint survival, reclaimed page-class guarantees and bounded cold-path execution |
| Workload mix | Prefill only, decode only, mixed long prompts/decode, heterogeneous acceptance, routed expert skew; visual bursts with decode fairness |
| Media | Standard wire-to-processor mapping and explicit native-video/frame-sequence distinction; single/multiple images, aspect ratios/resolutions, interleaved text/media, video temporal order/timestamps, exact grids/rounding/temporal padding, boundary limits, explicit frame-count paths and invalid metadata |
| Multimodal lifecycle | Cancellation/resumption inside visual spans and at media boundaries; same-token/different-media prefix misses; retained replay identity, stale media completions and bounded cache misses |
| Transitions | Every supported ownership change both ways, partial migration, threshold oscillation, and continuing arrivals |
| Lifecycle | Completion, all profile-defined generation terminators, PLE resets, cross-token/block stop strings, context exhaustion, partial-prefill cancellation/resumption, cancellation around commit/publication, slow/disconnected clients, slot reuse, bounded admissions and pending drains |
| Speculation | Text/visual MTP aligned prefill/target-conditioned refresh and DFlash HC/position alignment; all-accept, early/late rejection, zero accepted candidates, warm zero-proposal operation, final bonus/anchor handling and recovery |
| Sampling policy | Every exposed greedy/stochastic policy; supported constraints/penalties; rejection, cancellation and resumption preserve canonical bookkeeping; unavailable verifier capability rejects before state mutation |
| Profile changes | A-to-B-to-A without worker/context restart; enlarged execution/descriptor bounds; recoverable load failure after old-profile retirement and in-process recovery; repeated switches without uncharged residual allocations |
| Long-lived service | Repeated fill/drain and slot permutations, typed page-class availability, graph-pool replacement, lazy-library residency, bounded migration and full resident commitments plus the largest permitted media work unit |

A 260,096-row expanded prompt plus up to 2,048 generated tokens fits 262,144; a 262,144-row prompt plus a reply does not.

### Correctness and quality gates

Exercise import and device-address calculations across the `2^31` and `2^32` byte boundaries and the largest supported offset, including intermediate products and descriptor serialization. Prove bounds for narrower local indices.

Compare chunkwise prefill and recurrent/verification outputs, convolution tails and final states against the pinned reference with fresh/nonzero initial states, partial chunks, mixed sequence lengths, masked rows, resets, text/visual rows and every accepted speculative prefix. Include PLE's dilation-three, nine-row history across chunk and rejection boundaries. Import parity must cover the stored decay representation and complete coupled head ordering before TP partitioning. Measure recurrence/convolution time and bounded workspace separately in prefill, decode and mixed workloads.

Compare unsharded/sharded attention with identical configured codec and target semantics. Cover empty shards, extreme scores, pooled-history boundaries, partial pages, indexer ties, long absolute positions, masks, and codec metadata. Separately compare quantization quality against the chosen higher-precision reference. Verify all four pooling phases, spans crossing multiple groups, every accepted prefix, whole-append rollback and reuse cuts inside groups. Compare pooled values, selected positions, ending raw tails and group-first coordinates against the pinned semantics for text and visual rows.

Test accepted-prefix recovery across attention KV, pooled indexer history and raw pending state, recurrent state, Gated DeltaNet/PLE convolution tails, PLE token/hash history, draft state, target/draft positions, media identity, sampling/constraint/output bookkeeping and RNG streams, including cancellation and slot-generation races. Compare greedy outputs where exactness is expected; validate stochastic acceptance and resulting distributions without requiring identical sampled strings across schedules.

Qualify RotorQuant's actual SM70 kernels through progressive long-prompt prefill and long decode, beyond isolated-vector error. Verify MTP import fusion, flattened 10,240-wide hidden normalization, complete learned norm vectors, single Gemma-offset conversion, HC transforms, actual next-position visual embeddings and explicit QSA execution against identical decoded weights/numerics. Qualify MTP with both target-selected KV codecs; compare weight formats by quality, acceptance and throughput without requiring identical proposals. For DFlash 2, verify all five completed-layer HC means across TP8/chunking, block-local convolutions, selector probabilities, 2,048-window wrap/rollback, separate RoPE, and conditioning after rejected/skipped proposals. Detect future-token leakage into earlier target logits or indexer decisions, including the up-to-three-token QSA tail.

Compare visual preprocessing, patch/grid order, projected embeddings, target positions and representative logits with the pinned reference. Evaluate SC117 image/video understanding on OCR, charts, screenshots, documents, spatial relationships, multi-image comparisons and temporal video tasks. Verify MTP/DFlash conditional acceptance and target-equivalent output with visual prefixes; family compatibility and unchanged vision weights do not establish quality after language-model modification.

For Section 17, compare suspended/resumed continuation with uninterrupted execution under identical profiles and sampling ownership, including legal pending anchors, stopped output and partial-prefill retention. Measure FIFO waiting/TTFT, slot handoff/idle-reclaim latency, admitted fairness, full-load committed decode, tail restore latency, PCIe/NUMA/storage contention and bytes per completed request above GPU capacity. Separate requested logical-input rate from executed prefill; report reusable hit rows by GPU/RAM/storage tier, avoided work, dirty versus reused backing bytes, load coalescing and cache-management overhead. Include cold/miss paths, repeated cycles and slow clients; claim neither a hard wait bound nor faster decode from added storage alone.

Use focused synchronization/memory diagnostics for kernel risks and lifecycle races. Each gate must address a concrete correctness, resource, or performance risk; avoid tests that merely repeat implementation details.

Qualify Section 2's combined dependency manifest: inspect SM70 coverage and loaded libraries, exercise selected GEMM algorithms and every communicator at actual dtypes/message sizes, and test captured/uncaptured serving. Framework/extensions must not silently introduce unsupported kernels/backends. Use NVTX/Nsight Systems for host gaps, collective critical paths, transfers, and overlap; Nsight Compute for bandwidth, Tensor Core/SIMT activity, occupancy, and spills; focused Compute Sanitizer checks and NCCL correctness/performance tests. Separate instrumented runs from throughput receipts.

Resolve applicable compiler issues before release. NVIDIA issue 6111901 concerns failed reconvergence under nested thread divergence, introduced in CUDA 12.8 and fixed in 13.2.2. Keep an explicit disposition gate for the selected SM70 compilation paths: record a supported correction, documented non-applicability or validated source-level mitigation with actual coverage; retain applicable targeted reproducers/regression checks. Assume neither V100 impact nor a 12.x backport; CUDA 13’s fix does not restore SM70 compilation. A successful build, SM70 object inventory, smoke test or synthetic/illustrative divergence probe alone cannot close the gate. Source: [NVIDIA compiler issue](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html).

### Performance implementation audit

Retain a trace-backed inventory for each released regime: H2D/D2H/D2D calls/bytes per committed decode token (per text/visual input row or media unit for prefill/encoding); host waits/implicit synchronization; allocation/initialization; kernel launches; graph replay/breaks/recapture and compiled-cache misses; per-rank idle/communication time; memory throughput/spills; CPU thread/cpuset/NUMA placement, PLE faults/lookup latency and locked-memory peaks. Distinguish useful/tentative work and steady-state/cold preparation. Sample asynchronously so telemetry does not become the bottleneck.

Explain remaining hot-path boundaries/waits and compare the best qualified alternative at every integer activity count, including one active with full retained histories, near-full-history/full concurrency and mixed media/prefill/decode. Check Section 15's capability table and deployed-container transport paths; prove dependency/buffer correctness through cancellation, stream overlap and graph replay. Require numerical/output-semantic gates, complete resource feasibility and repeatable whole-workload benefit within predeclared fairness/latency regression tolerances; kernel-only gains are insufficient. Log rejected/unhelpful candidates rather than exposing gratuitous knobs.

Retain production error checks, ECC and supported power/thermal protections. Record host/GPU clocks, power, throttling and competing work to control comparisons. Tuning cannot disable reliability mechanisms, silently oversubscribe managed memory or page executing/GPU-only-promised histories to host; Section 17's explicit tiered-retention contract is separate. Investigate MPS, persistent kernels, alternate allocators, host huge pages, separately supported RDMA or extra frameworks only for a measured bottleneck, with independent support, lifecycle and resource qualification; preserve single-target and no-restart contracts. Storage work must respect Section 2's explicit V100/GDS support boundary.

### Performance gates

Control prompts, generation settings, target/draft revisions, warmup, clocks/thermals and timing boundaries. Repeat enough to distinguish gains from noise. Compare adaptive execution with the best qualified static plan under identical context, quality, retention and workload conditions.

Keep history ranges and reasoning/sampling settings comparable across active counts. Test different owner/slot phases at low activity, not only slot zero. Profile indexer and PLE separately to avoid attributing their latency to attention placement or proposal depth.

Report useful text prefill, visual-row processing and committed decode separately, alongside whole-step/end-to-end rates and isolated-kernel measurements. Include CPU media preparation, encoding, transfers, quantization, draft maintenance, rejected work, collectives, recovery, scheduling and amortized migration. Visual rows are not ordinary text tokens/s. Report latency distributions and fairness so throughput cannot hide starvation. Select the best measured eligible profile at each load regime.

## 22 Implementation sequence and exit gates

Use Section 2's dependency/processor manifests and Section 15's operator/library contracts throughout. Parallelize work where state/resource contracts permit; every new execution path must pass Section 21's applicable gates.

The stages define required outcomes and exit evidence. Reuse suitable implementation components after checking their contracts; source/host checks cannot mark device or integrated workload gates complete.

| Stage | Required work | Exit evidence |
|---:|---|---|
| 1 | Qualify SM70 stack; implement family manifests, complete per-role metadata resolution and bounded mixed-format GGUF import; bind exact SC117 IQ3_S/Q8 MTP, Inco draft, target frontend and visual/processor artifacts | Build/library IDs, effective metadata provenance, tokenizer/template parity, complete tensor/import inventory and fresh per-rank resource receipt |
| 2 | Add canonical page/session/media metadata, checked addressing, typed arenas, full-context commitments, accepted-prefix/sampling/PLE/streaming APIs and SM70 recurrent/convolution kernel regimes using Q8 | Target/state parity, bounded prefill/decode work, transcript, cancellation, media identity and page-class guarantees |
| 3 | Compact raw indexer storage to bounded pending/transactional state and remove persistent KV/pooled-indexer replication through owner execution; retain weight TP8 | Exact growing/fixed history bytes, rank balance, group transactions and output/selection equivalence |
| 4 | Add attention degrees, exact merge, topology-aware mapping and separate MTP-cache placement | Correct degree 1/2/4 operation; placement/communication profiles |
| 5 | Port RotorQuant Q4 and direct SM70 pack/unpack/attention | Layout, quality and long-context gates; no full-history mirror |
| 6 | Implement required image/video processor, FP16 vision path, target positions/PLE inputs, bounded staging/replay and mixed scheduling | Reference parity, media quality, fixed placement and complete maximum-work resource envelope |
| 7 | Implement required SC117 Q8 QSA MTP with actual target aliases, selected Q8/Q4 KV, full-prefix text/visual initialization and refresh | Pinned-control/imported-weight parity, useful speculation, recovery and complete capacity receipts |
| 8 | Add bounded graphs, global controller and mixed media/prefill/decode scheduling | Safe immediate load changes; Section 21 performance implementation audit; integer sweep and detailed threshold/breakpoint evidence |
| 9 | Add bounded migration, epoch publication and hysteresis | Intermediate states fit; serving remains correct; adopted migrations yield net gains |
| 10 | Integrate IncoAI DFlash2, TP8 HC capture, FP16 kernels/selector, qualified stochastic verifier and bounded SWA/features with separate draft positions | Independently qualified SC117/Q8-KV and SC117/RotorQ4-KV text/visual correctness, supported sampling, capacity, acceptance and performance |
| 11 | Complete in-process administration, observability, sustained-load tests and operations; implement native/YaRN positional profiles and extended Q8/Q4 capacity/quality qualification with required MTP and separately qualified DFlash | No tuning restart; immutable position identity, bounded phase resources, native/extended parity and quality, lifetime commitments, readiness/failure contracts and full-load receipts/benchmarks |
| FIFO, prefix reuse and retained-session tiers | Integrate bounded FIFO admission, completion/idle slot handoff and fair GPU scheduling with required compatible prefix reuse, one page/state catalog and complete RAM/optional-storage suspension; qualify standard OpenAI-compatible Chat/Completions/Responses lifecycle and separately implemented standard Conversations handles | Section 9/17/21 cold/hit/miss, handoff/restore, ownership/reference, state/quality, failure and backlog performance receipts; no restart or routine full-history round-trips |
| Additional profiles | Qualify other target/draft artifacts, including the NVFP4 source/runtime contract | Conversion parity, quality/acceptance, actual allocations and recomputed activity/capacity profiles in one serving path |
| Later | Evaluate SAW and other architectural alternatives | Separate quality/capacity/performance evidence before inclusion |

MTP remains required/default; DFlash is independently selected. Each needs its own geometry and multimodal gates; neither may be silently disabled to increase capacity.

Retain each stage's commit/build ID, workload, resource receipt, results and limits. Qualification does not extend implicitly to larger contexts, different media envelopes or higher concurrency.

## 23 Integration inputs and unresolved measurements

The model choices are settled; implementation supplies these remaining bindings and measurements:

1. **Artifact integrity/numerics (Sections 2, 12–15):** Repeat the SC117/vision/MTP/draft source-directory census at installation; verify complete payload hashes and decoded values, resolved metadata provenance, the single target tokenizer/template including NFC and token roles, target/MTP/DFlash/vision transforms, aliases and per-rank allocations. Import only the selected MTP's 32 core tensors and use actual target vocabulary matrices. Preserve the source-bound H32 Q8, CUDA-authority RotorQuant Q4 and explicit MTP QSA contracts. Resolve model-license provenance before any weight redistribution.
2. **Server/stack (Section 2):** installed CPU/DIMMs, GPU UUID/ranks, links, PCIe/NUMA, firmware and newest compatible build/library IDs; SM70 and compiler/library qualification.
3. **Multimodal (Sections 2, 16–17):** standard endpoint/content-type-to-decoder/processor wire mapping, native video versus frame-sequence semantics, service media bounds, fixed encoder placement, operator/position parity, CPU/GPU/pinned/replay peaks and task quality.
4. **Speculation (Sections 12, 14):** MTP import/norm/HC and text/visual refresh; Inco DFlash2 TP8 features, FP16 conversion, selector, SWA, separate positions and useful blocks for SC117 with each KV codec.
5. **Recovery/tiering (Sections 12, 17):** complete checkpoint schema, RAM/storage quotas and paths, retained media/replay, replacement/transfer overlap, bandwidth, failure/durability policy and proven shared lifetimes.
6. **Execution (Sections 10–18):** row/prefill/media budgets, recurrent/convolution kernels/chunks, graphs, ownership degree, draft width, migration cap and hysteresis across all activity/workload regimes.
7. **Service/API (Sections 5–7, 19):** context/position selection and quality thresholds; maximum capacity then full-load decode with fairness/latency/queue limits; lifetimes/multiplicity, fresh/resume FIFO order, per-principal/queue limits, input commit/cancellation, output backpressure, drains and readiness/failure behavior.

Publish the qualified integer maximum from Section 7’s complete post-import ledger for the selected context/method/media envelope; Section 8’s geometry examples are not capacity promises.

## 24 Acceptance checklist

Every detailed requirement and applicable Section 21 gate remains mandatory. Release sign-off must establish:

| Area | Required outcome |
|---|---|
| Platform and execution | Exact coherent SM70 builds/loaded libraries, resolved compiler issues, bounded workspace/communication, and Section 15/21’s trace-backed hardware, NUMA, numerical and hot-path audit. |
| Models and frontend | One manifest-driven serving path; independent source/runtime weight, activation and target/draft KV choices; complete metadata provenance and validated gap inputs; one qualified target frontend, with no MTP template translation/remapping (Sections 2, 6). |
| Required capabilities | Every sequence reaches its promised input-plus-reply window. Native/extended positions, image/video → text, required/default SC117 Q8 QSA MTP and separately selected Inco DFlash2 pass their own geometry, alias/import, text/visual conditioning, KV, position, quality and sampling gates (Sections 6, 12–16). Native audio/media generation is excluded; additional pairings require independent qualification; SAW follows core implementation. |
| Memory and state | Complete worst-rank and host/pinned/replay ledgers fit useful prefill/verification and the largest media unit at full residency. Stable TP8 or a qualified single-target alternative; no extra persistent target/vision copies. Unique balanced histories, bounded raw state, source-bound Q8/iso4, accepted-prefix recovery and padded-row isolation pass Sections 4, 7, 9–13 and 20. |
| Service and capacity | Maximum qualified capacity, then full-active/near-full-history committed decode; preserved low-load commitments; every integer activity count, endpoints and breakpoints without a 32-session ceiling. Independent sequence accounting, bounded FIFO/prefix/tier integration, standard API identity, complete protected checkpoints, cold/warm failure gates, cancellation, slow clients and slot reuse pass Sections 5–7, 9, 16–17 and 21. |
| Live changes | Sudden load changes and partial migration stay bounded/correct. Structural changes preserve lifetime, context, quality, media fidelity and selected-method commitments. No tuning container/worker/CUDA-context restart; unsupported controls stay fixed/pending. All-rank readiness precedes publication; recoverable failure supports in-process recovery without duplicate full models (Sections 11, 18–19). |
| Evidence | Receipts separate observations, allocations, commitments, estimates and qualification. Report useful text/visual prefill, committed decode, media work, latency, fairness and overhead separately; reject unsupported verifier policies before state mutation and publish only canonical commits (Sections 12, 21–23). |

## 25 Evidence and source register

### Verification scope

Primary documentation and exact pinned implementations were inspected across all 25 sections for platform/SM70 support, model operators, quantization, speculation, preprocessing, positions, state/accounting and service/API contracts. Cited sources were retrieved and their relevant contents checked; link reachability alone is not evidence for a claim. Repeated whole-document reviews cover cross-section consistency as well as individual source facts.

Complete directories were parsed from bounded prefixes of both target shards, vision GGUF, selected/comparison MTP companions, Inco safetensors and 28 official shards containing MTP tensors. Applicable shape/type/split/offset/alignment/non-overlap/file-boundary checks passed. Ancillary inventories, official tokenizer data, MTP alternatives and companion interfaces were also inspected. Complete large weight payloads were not downloaded, hashed or executed.

| Inspected artifact | Directory-derived result |
|---|---|
| SC117 IQ3_S target, both shards | 1,224 unique tensors; thirteen source formats; 83,852,561,920 tensor payload bytes; no embedded MTP |
| SC117 visual GGUF | 334 tensors; 224 F32 and 110 BF16; 907,523,008 payload bytes |
| Selected SC117 Q8 MTP | 34 source tensors; 21 Q8_0, eleven F32, two BF16; 4,126,482,432 payload bytes. Import the 32-core subset of 2,775,621,632 bytes and alias the actual target embedding/head. |
| Unsloth shared-Q8 comparison | 32 tensors; nineteen Q8_0, eleven F32, two BF16; 2,775,621,632 payload bytes; matching core descriptors do not prove all quantized values equal. |
| Unsloth shared-BF16 conversion control | 32 tensors; 21 BF16 and eleven F32; 5,217,016,832 payload bytes |
| Official BF16 MTP source | 31 `mtp.*` tensors across 28 shards; 5,214,301,696 payload bytes |
| Inco DFlash2 | 81 BF16 tensors; 603,065,600 parameters; 1,206,131,200 payload bytes |
| Official image/video processor JSON | Full files retrieved; 390/385-byte lengths and both stated SHA256 hashes independently checked |
| Selected tokenizer/templates | Complete GGUF arrays inspected; target/MTP token/type/merge arrays match. Official 248,044 BPE entries, 33 added-token IDs and 247,587 merges match the target's corresponding entries; target CONTROL/UNUSED roles and the selected target template remain explicit. |

All thirteen nonquantized core tensors in the selected SC117 Q8 and Unsloth shared-Q8 companions were fetched through bounded exact ranges: 8,707,072 bytes per artifact, independently compared byte-for-byte and equal. The nineteen Q8_0 core tensors were not fully compared; structural equality and source lineage do not certify their complete numerical identity.

Bounded host checks used Tokenizers 0.22.2 and Jinja 3.1.6. The official tokenizer and a GGUF-derived reconstruction with explicit official NFC/Qwen35/ByteLevel settings agreed on 35 fixture encodings and full decodes; all 33 added-token literals matched their IDs. Ten template fixtures checked common rendering and the selected target's instruction/effort/tool behavior. The review separately identified the target's six CONTROL-versus-base-special-flag differences and 243 UNUSED rows, which require Section 2's explicit target policy. Disabling NFC changed token IDs for decomposed accented text and Hangul Jamo. These host checks establish the tested metadata/pipeline behavior, not qualification of the eventual serving backend or any V100 inference workload.

Large-artifact SHA256 values are expected installation hashes checked against immutable public pointers, not independently recomputed payload hashes. Source sizes and operator agreement do not establish runtime residency, imported-value correctness, task quality or V100 execution.

All capacity tables, history rates, migration volumes, indexer-owner counts and large-buffer examples were independently recomputed. They remain geometry/stride/rounding/placement-dependent estimates, excluding other ledger terms; they establish no final capacity, speculative speedup, extended/media quality or throughput.

### Remaining installation and workload evidence

Release requires full-file integrity and numerical import checks; actual GPU/CPU/NUMA/link inventory; a coherent supported SM70 build and compiler-issue disposition; measured loading/serving/graph/transfer peaks; complete state/sampling/media/API lifecycle tests; and representative quality/performance qualification at each released context/codec/method/media profile. Source-level facts and design invariants cannot replace these empirical gates. Sections 21–24 specify the required evidence and acceptance criteria.

### Primary external references

Sources appear beside their requirements. Preserve each source’s scope; modern-GPU evidence does not qualify this V100 implementation.

Additional pinned implementation and background references:

| Reference | Use and boundary |
|---|---|
| [Transformers model implementation](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py) | Geometry, PLE indexing and normalization; preserve selected GGUF tokenizer/template and converted values |
| [SGLang QSA parser](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/attention/qsa/config.py) | Section 12's explicit QSA MTP policy agrees with the selected SC117 Q8 metadata; comparison artifacts' zero-ratio metadata does not establish dense/sparse equivalence |
| [SGLang GemmaRMSNorm](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/layernorm.py) and [GGUF norm conversion](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/conversion/qwen.py) | Exact MTP normalization axis/weights and single application of the converted Gemma offset; matching shapes alone do not establish numerical parity |
| [Original DFlash paper](https://arxiv.org/html/2602.06036v2) and [repository](https://github.com/z-lab/dflash) | Method background; distinguish DFlash 2 model changes from a serving engine named Spec V2 |
| [vLLM context parallelism](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/), [dynamic speculation](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/), and [NanoCP](https://arxiv.org/abs/2605.21100) | Related concepts, not drop-in qualification for TensorFold, stock MTP, or this topology |

Pin the recorded artifact/reference revisions, final TensorFold implementation, and full runtime stack in each receipt. Changes to sources, codecs, kernels, feature contracts, or numerical settings invalidate affected qualification until rechecked. Label every capacity/performance claim as primary-source fact, arithmetic estimate, header plan, or live-qualified result.
