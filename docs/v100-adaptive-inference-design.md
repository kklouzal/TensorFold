# TensorFold V100 Adaptive Inference Design

**Platform:** Supermicro SYS-4029GP-TVRT; eight 32 GiB V100 SXM2 GPUs.\
**Initial target:** non-Swift `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF`, `IQ3_S`.\
**Required:** QSA MTP, separately selectable IncoAI DFlash2, image/video → text.\
**Scope:** implementation design for qualified family variants, GGUF/NVFP4 and compatible drafts.\
**Basis:** discussion and startup receipt dated 2026-10-25 04:16 UTC.\
**Evidence:** supplied observations, artifact/header measurements, arithmetic estimates and V100 workload qualification are distinct (Section 25).

## 1 Governing operating contract

Use one engine for qualified Qwen3.8-Flash-Next variants. Administrators select target/runtime weights, context/RoPE, KV codec, speculative artifact/method and service limits; TensorFold computes full-context capacity and selects qualified workload plans. Initial profiles use SC117 IQ3_S, shared-Q8 MTP and IncoAI DFlash2. Additional artifacts use the same contracts.

**Mission:** maximize qualified sessions at the selected context, weight/KV quantizations, speculative method and media envelope, then maximize aggregate committed decode tokens/s with all sessions active. Adapt safe execution to actual activity for useful prefill/decode throughput, preserving context, quality, fairness and no-restart commitments. Report the measured context/capacity/throughput tradeoff; never silently sacrifice one to improve another (Section 5).

### Settled requirements

| Area | Governing requirement |
|---|---|
| Hardware | Optimize for this server's measured NVLink/PCIe/NUMA and SM70 behavior (Section 2). |
| Target | Non-Swift SC117 IQ3_S; qualified mixed-format import; TP8 baseline with all eight ranks available at low load. Section 20 governs alternatives. |
| Residency | One distributed target, at most one speculative model; qualified no-speculation transition releases both speculative systems. No extra full-model or persistent weight/vision copies for throughput. Preserve unique histories; charge required rank-local parameters and bounded activation/communication/recovery/migration storage. |
| Context | Equal administrator-selected windows shared by expanded text/media, retained conversation and reasoning/reply. Native 262,144 rows; required RoPE/YaRN extension needs separate row/coordinate qualification (Section 6). |
| Capacity | Compute the complete worst-rank integer maximum; 20/24/32 and power-of-two counts are not fixed limits (Sections 7–8). |
| Speculation | Required/default shared-Q8 QSA MTP; separately selectable IncoAI DFlash2; explicit no-speculation reference/admin profile cannot replace MTP implementation (Sections 12, 14). |
| KV | Ordinary precisely specified Q8 or RotorQuant `iso4` K/V. No unspecified native RotorQuant Q8. RotorQuant first; SAW later (Section 13). |
| Histories | Unique attention KV and completed pooled indexer history in VRAM; bounded raw pending state; RAM raw-history archive off (Section 9). |
| Multimodal | Required text/images/video → text, pinned vision/processor, bounded execution and separate MTP/DFlash qualification. Native audio/media generation excluded (Sections 2, 16). |
| Adaptation | Workload-driven batches, kernels, graphs, prefill budgets, proposals and justified attention placement (Sections 10–18). |
| Restarts | No container/worker/CUDA-context restart. Non-live settings stay at qualified maximum-concurrency values during adaptation or await a supported administrator in-process transition (Section 19). |
| Migration | Batch-boundary changes and bounded state-preserving migration retain every admitted sequence's lifetime/context commitment (Section 11). |
| Simplicity | Refactor one serving path directly; initially one homogeneous numerical/speculative resident profile. No simultaneous speculative systems, duplicate full histories or unnecessary framework layers. |
| Evidence | Complete accounting and representative execution support capacity/performance claims; header arithmetic, startup allocation and live qualification remain distinct (Sections 21, 25). |

SC117 capacity requires reconciled post-import target/method/vision, state, recovery, execution and placement allocations. Section 8 gives geometry-only cross-checks; 20-Q8/32-Q4 are reference points, not admission promises or ceilings.

### Boundaries of the design

Preserve the target's sparse/indexed attention, recurrent equations, masks, pooling, positions, and sampling semantics while changing storage ownership and scheduling. Qualify quality-changing codecs separately.

No-restart applies to serving adaptation and supported administrator changes. Hardware/driver faults and software deployment have separate lifecycles; universal GPU/process fault recovery without restart is not promised.

Preserve the selected tokenizer and operators; declare every weight conversion and numerical choice in the effective profile. Pin the custom Torch-based fork's revision rather than assuming public TensorFold interfaces match it.

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

Volta Tensor Cores use FP16 matrix operands with FP16 or FP32 accumulation. Packed Q4/Q8 needs an explicitly supported compute path; do not assume newer GPUs' native INT4, INT8, FP8, or BF16 Tensor Core paths. Source: [Volta tuning guide](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html).

Use Volta-capable profilers and verify every dependency/kernel backend. FlashAttention or modern FA3/FA4/Triton recipes do not establish SM70 support for this sparse, rotated, quantized attention path.

Release targets checked on 2026-10-25 follow. Deployment requires Section 21's combined-stack qualification. Resolve newer eligible patches for subsequent builds.

| Component | Selected release target | SM70 condition |
|---|---|---|
| CUDA Toolkit | 12.9.2, Update 2 | Newest published 12.x toolkit; build native `sm_70` objects with its supported host compiler |
| cuBLAS and cuBLASLt | 12.9.2.10 | Newest published 12.x library candidate; qualify actual FP16/FP32 algorithms, layouts, and workspace bounds |
| CUTLASS C++ | 4.8.1 | Use retained SM70 C++ TensorOp/WMMA components; the name of its 2.x kernel API does not require a 2.x package |
| NCCL | 2.32.3, release tag `v2.32.3-1` | Released CUDA 12.x candidate; verify the exact release build's SM70 objects and V100 collective/graph execution |
| CCCL, CUB, Thrust, libcu++ | 3.5.0 required-primitives SM70 build candidate | CUDA 12.x compilation remains supported; SM70 is outside regular upstream validation, so qualify every required primitive and the coherent framework build |
| cuDNN, when required by the framework or a selected operator | 9.10.2 for CUDA 12.x | Its matrix supports Volta with CUDA 12.9; 9.11.0 removes Volta |
| Nsight Compute | 2025.2.1, build 2025.2.1.3 | 2025.3 removes Volta |
| Nsight Systems | 2025.3.2, build 2025.3.2.474 | 2025.4 and later remove Volta |
| Compute Sanitizer | 2025.2.1 release, CUDA package 12.9.79 | 2025.3 removes Volta |
| PyTorch, for the existing Torch-based fork | 2.14.1 source-build candidate against the selected CUDA 12.9 stack | Its published Volta wheel uses CUDA 12.6; inspect source-build and transitive dependency compatibility instead of assuming the wheel supplies this stack |

Toolkit/library versions: [CUDA 12.9.2 component manifest](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-toolkit-release-notes/index.html), [cuBLAS patch releases](https://docs.nvidia.com/cuda/cublas-patch-release-notes/), [CUTLASS releases and compatibility](https://github.com/NVIDIA/cutlass), [NCCL release](https://github.com/NVIDIA/nccl/releases/tag/v2.32.3-1), [NCCL build rules](https://github.com/NVIDIA/nccl/blob/master/makefiles/common.mk), [CCCL 3.5.0 release](https://github.com/NVIDIA/cccl/releases/tag/v3.5.0) and [exact-release architecture policy](https://github.com/NVIDIA/cccl/blob/v3.5.0/README.md), [cuDNN support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html) and [9.11 removal](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.11.0/release-notes.html). Current NCCL source includes SM70 under CUDA 12.x; this does not replace exact-release binary inspection.

CCCL 3.x omits older-toolkit-only architectures from regular testing without disabling every path. Qualify 3.5.0 scan/sort/compaction/reduction and warp/block primitives for numerics/selection, graphs/streams, bounded scratch and whole-step performance; resolve vendored headers coherently. If it fails, select the newest earlier passing release. CCCL 2.8.5 is a comparison baseline, not a ceiling or LTS claim. Pin one deployment release; no serving-time library fallback. Sources: [CCCL 2.8.5](https://github.com/NVIDIA/cccl/releases/tag/v2.8.5); [issue 6226](https://github.com/NVIDIA/cccl/issues/6226); [clarification 6229](https://github.com/NVIDIA/cccl/pull/6229).

Tools/framework evidence: [Nsight Compute history](https://developer.nvidia.com/nsight-compute-history) and [2025.3 support boundary](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_3), [Nsight Systems boundary](https://docs.nvidia.com/nsight-systems/ReleaseNotes/), [Sanitizer history](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html), [NVIDIA published package builds](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/), [PyTorch 2.14.1](https://github.com/pytorch/pytorch/releases/tag/v2.14.1), and [PyTorch CUDA 12 packaging policy](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432). A package index identifies a build; it does not select the server's OS. Source builds must resolve framework-vendored headers and transitive libraries coherently, rather than overriding individual libraries inside an incompatible wheel.

Keep dependencies immutable during serving. Library/compiler/driver replacement follows Section 1's deployment lifecycle and invalidates affected algorithm, graph, numerical, resource and performance qualification. Adaptation selects qualified plans; it never upgrades dependencies.

Use NUMA-aware scheduling/transfer workers and bounded pinned staging. Pin an evictable file mapping only with measured need and a host-memory budget. Initially use one worker per GPU and one distributed coordinator; another process model needs synchronization and performance qualification.

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

### NVFP4 sources on V100

Import NVFP4 through an explicit SM70 representation. E2M1 values use two-level scaling, commonly an E4M3 scale per 16 values and an FP32 tensor scale. Exporter layouts/recipes vary: decode the actual scale tensors, orientations, transforms, padding, and mixed-precision exclusions. Sources: [NVIDIA NVFP4 format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) and [scaling/recipe documentation](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html).

| Qualified runtime representation | Implementation and accounting |
|---|---|
| Retained packed NVFP4 values/scales with software tile decode | Custom SM70 kernels decode bounded weight tiles into supported FP16 operands, with sensitive scale arithmetic/accumulation as required. Charge packing, scales, tiles, conversion work and actual per-rank retention. This preserves compact storage without claiming native NVFP4 matrix throughput. |
| Converted supported packed weight format | Stream through bounded import buffers and materialize the chosen SM70 format. Repacking that preserves decoded values is distinct from requantization; requantization changes the numerical profile and needs separate quality, acceptance, memory and performance qualification. |
| Resident FP16 for selected tensors | Permit only where the full retained and peak allocation plan fits and measurement justifies it. Do not silently expand an entire target because its source encoding lacks a native instruction. |

Initially retain packed NVFP4 weights and decode bounded tiles into the existing FP16 path. Select another qualified representation only when measured throughput and quality/resource constraints justify it, using the same engine's loader/packing and kernel choices. Source file size is not a VRAM budget; native-Blackwell benchmarks do not predict V100 performance.

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

Before allocation, inspect both binary directories and validate complete non-overlapping tensor coverage, shapes, block sizes, split metadata, offsets, alignment, scales/codebooks, and full-file hashes. Generate a per-tensor source-to-runtime inventory. The published pointer identities above are verified; the selected SC117 binary headers and full payloads have not been independently inspected here.

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

`IQ3_S` is a learned mixed allocation. Initial decoders must support the publisher's BF16, F32, F16, Q2_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS list. Obtain exact counts/payloads from both shard headers; the card histogram is not a complete census. Sources: [pinned SC117 inventory and method](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/blob/90a3350cd800ec1fe0d1660a38414ad9e985782f/README.md).

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

Retain the existing 4 GiB/GPU device-floor and 2 GiB/GPU serving-workspace protections. The historical planner observed a 27.086 GiB/GPU serving budget after its initial runtime allocations; remeasure the selected profile's baseline rather than treating that value as universal. Its three-slot 0.273 GiB snapshot and 0.211 GiB replacement promises must be recomputed for actual state/lifetimes, with separately bounded graphs. A smaller checkpoint does not authorize weakening these protections or recovery.

Reconcile each allocation and resource promise once. Promises need not be allocated slabs; do not add them to the startup snapshot or subtract an included floor twice.

Use checked 64-bit arithmetic for source-file offsets, allocation sizes, byte strides and page/arena address products; widen operands before multiplication or addition. Narrow local indices remain appropriate where bounds are proven for the effective profile. Reject overflow or out-of-range descriptors before allocation or launch. The selected shard exceeds 4 GiB; full-context arena addressing also requires checked wide products.

Every retained or peak allocation must have:

- An owner rank and allocation size including real alignment and padding.
- A persistent, session, batch, graph, replacement, or migration lifetime.
- A reason it must overlap another allocation, or a proven reason it can reuse an arena.
- A maximum derived from the effective profile and implementation limits.
- A measurement or construction proof for allocations made by libraries outside the explicit allocator.

Per-rank budgets cover target/method/vision weights and histories; recurrent/convolution/raw-pending/pooling/draft/sampling/constraint/output/media-position state; bounded language/visual execution/features; verification/recovery; graphs/communication; replacement/migration overlap; metadata/padding. Host media, replay and pinned staging are separate admission constraints, not KV capacity.

Take the maximum for proven mutually exclusive workspaces and sum overlapping occupancy. Use one bounded execution arena where it can replace per-slot maximum slabs; reuse only after all asynchronous readers retire.

Use fixed-size typed history-page arenas and bounded execution/graph arenas. Track free classes/counts per rank, not aggregate bytes; reserve matching layer bundles before execution. Full admitted windows must remain allocatable despite fragmentation, graph replacement and migration, without fresh general-purpose device allocation per page at maximum residency.

Before publishing capacity, initialize or conservatively budget every eligible kernel/communicator/library family's lazy residency. Every selected plan must fit admitted commitments and the overlap envelope. Deleting a graph does not prove its pool was released. Enforce these invariants in the existing allocator/planner.

Reconcile the workspace protection with actual language/visual arena lifetimes. The historical 0.640 GiB workspace plan and 27.798 MiB/session subset cannot be added wholesale to a newly itemized ledger. Recalculate snapshots/replacement for the selected profile without weakening recovery.

## 5 Performance objectives and workload definitions

### Separate quantities

| Quantity | Definition |
|---|---|
| Configured context allowance | Maximum prompt-plus-reply length promised to each session under the effective profile |
| Resident capacity | Maximum independent logical sequences whose complete context and necessary state can be held under that promise |
| Resident sequences | Request-scoped or intentionally retained sequences currently holding history/state |
| Active requests | Admitted requests with pending or in-flight inference work |
| Active sequences | Independently advancing logical sequences with pending or in-flight inference work |
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

Test one active request with all permitted histories resident. Idle retained history remains committed memory, not spare workspace.

### Optimization objective

Hold context/position, quantizations, method and media bounds fixed. Establish the largest qualified resident count with useful execution resources; measure sustained committed decode with all sessions active and near-full histories. Tune every active count through capacity, including full idle histories elsewhere. Report per-session rate/latency; empty-context batch rates do not establish full-load performance.

Administrators choose context using the measured context/capacity/throughput frontier. Load adaptation must not resize windows, downgrade Q8/Q4 or disable the selected method. It may tune safe batches, proposals, prefill/media quotas, kernels/graphs and ownership within the selected profile.

Measure text prompt tokens/s, expanded visual rows/s, and committed output tokens/s separately, including per-request output rate, TTFT, TPOT percentiles, queue delay and fairness. Report media pixels/frames and preprocessing/encoding time; visual rows are not ordinary text-token throughput. Rejected candidates are not delivered output.

Count each target prompt row once; report prefix/media hits, replay and MTP/other conditioning separately. Include media preparation/encoding, transfers and maintenance in complete-operation timing. Track successful-task latency and reasoning/output counts so shorter answers are not mistaken for higher token/s.

Avoid batch-filling delay at low activity; maximize aggregate committed throughput within administrator latency limits at high activity. Mixed load must advance both prefill and decode. Optimize whole iterations, including routing, features, draft maintenance/drafting, verification, rejection/replay, communication, sampling, and commit.

Establish this fork's measured performance frontier at relevant operating points. Neither modern-GPU benchmarks nor peak arithmetic throughput supplies an established tokens/s target or universally winning plan.

## 6 Administrator configuration and effective profiles

### Inputs

Expose only implemented, qualified choices:

- A qualified Qwen3.8-Flash-Next target manifest and runtime weight representation, initially non-Swift SC117 IQ3_S.
- Equal context allowance and a qualified native or extended RoPE/YaRN profile, with explicit target/MTP/draft coordinate bounds; 262,144 is the native baseline, not a permanent engine ceiling.
- Target KV format and its pinned implementation.
- Speculative method: none, stock MTP, or DFlash.
- Compatible MTP/DFlash artifact and supported method-weight/KV choices; initial MTP KV matches target KV, while initial DFlash KV is FP16.
- Explicit history-residency policy and implemented media limits/processing policy; the effective profile includes fixed vision placement and bounded visual execution.
- Optional session ceiling.
- Performance objective and any requested latency or waiting-time limits.
- Explicit retained-session lifetime and bounded admission/output queue policy, preserving the existing service contract.

A draft sliding-window override is exposed only if the selected checkpoint and storage implementation support it. Target and draft weight formats, target KV, and draft KV are independent dimensions. Validate each requested sampling/constraint policy against the effective verifier's capabilities before admission or session mutation; reject unsupported combinations with a specific reason.

### Outputs

The engine resolves full-context resident capacity, admission and media-resource limits, language/visual execution envelopes, row/patch budgets, graph/kernel families, fixed vision placement and per-batch scheduling. It reports the binding rank/resource and qualification status.

Illustrative schema, not existing TensorFold flags:

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
  artifact: shared_q8_mtp  # pinned manifest in Section 12
  kv_format: target
  proposal_length: auto
multimodal:
  enabled: true
  artifact: sc117_vision  # pinned manifest in Section 2
  limits: qualified_media_envelope  # concrete bounds required before admission
max_sessions: auto
history_residency: gpu
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

**Transitions/identity.** Fix frequencies, amplitude and algorithm for resident-state lifetime; active-count adaptation never retunes them. An allowance increase with unchanged transform needs Section 19's all-resident reservation/compatibility proof. A transform/reference-policy change initially drains affected sequences and retires state/graphs before in-process activation. Updating constants or re-rotating keys alone cannot convert quantized KV/pooled keys, recurrence, features and draft conditioning. Future replay conversion needs a bounded full-state proof. Include the complete positional manifest/epoch in session/prefix/checkpoint/draft/graph/tuning identities; forbid incompatible reuse. Expose configured/pending/effective state without container/worker/CUDA-context restart.

**Release evidence.** Test native/extended windows at native/new-limit boundaries, arbitrary chunk/group/ring splits, long generation, rejection/rollback and media continuation. Compare reference/one-shot/chunked logits and state under identical settings. Predeclare quality thresholds for short-context regression, retrieval across depths, multi-needle and order-sensitive reasoning plus representative user tasks; one needle test is insufficient. Cover Q8/Q4, required MTP, separate DFlash and image/video cases. For every released profile, run Section 21's integer activity sweep and near-full-history/full-concurrency tests; report acceptance, prefill/committed decode, TTFT/TPOT, fairness and resources. Keep unsupported candidates visibly unqualified; never silently clamp.

Reference algorithm background: [YaRN paper](https://arxiv.org/abs/2309.00071) and [Transformers RoPE parameters](https://huggingface.co/docs/transformers/main/en/internal/rope_utils). The initial frequency/amplitude construction reference is [`_compute_yarn_parameters` at the pinned Transformers revision](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/modeling_rope_utils.py); resolve and record its factor, original length, beta values, truncation and mscale/attention-factor conventions explicitly. This routine is independent of current sequence length; generic dynamic-NTK behavior is not a substitute. Pin the exact algorithm implementation/hash used for release; these references do not establish V100 performance or SC117 extended-context quality.

### Retention and admission

Preserve existing lifetimes. Request-scoped slots release state after completion/cancellation and retirement of readers, transfers, and callbacks. Only intentionally retained sessions survive requests under their close/expiry contract; missing expiry does not make stateless requests persistent, and inactivity does not evict retained state. Completion, cancellation, disconnection, and retained-session destruction are distinct events.

Bound the fair admission queue, define its queue-full response, and use aging where needed to prevent starvation. Bound media preparation before slot assignment, then validate exact expanded prompt/positions, output budget, media limits, and sequence count. Pending media work must retain its declared resource claim. Resumption uses the retained session's existing claim. Serialize or reject concurrent state mutations per sequence under the API contract. Each request-scoped sequence retains the same complete-window reservation until slot release.

## 7 Automatic capacity planner

For the selected context and positional/numerical/method/media profile, the planner calculates the largest supported resident count for a useful qualified execution envelope. At that count, select the measured envelope with the best aggregate committed decode throughput under service limits; retain useful prefill/media progress and fairness. Report the measured capacity/throughput tradeoff when a smaller count is faster, without silently lowering the requested capacity objective:

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

Ledger terms are disjoint allocation/lifetime categories. Charge exclusive positional tables/scratch there; reference shared language/graph arenas without double-counting. Target weights exclude separately charged speculation/vision; actual embedding/head aliases add no payload. Use fresh SC117 allocations, not the historical 13.421 GiB/GPU aggregate with another MTP plan.

For the compact indexer, growing histories contain attention KV and completed pooled keys. Raw pending rings and saved tail versions belong to retained state/recovery; raw prefill/verification spans belong to bounded execution.

Retained recovery includes mandatory per-session checkpoint/snapshot versions and future occupancy; peak recovery covers additional execution work. Reconcile shared storage once without omitting retained snapshots or charging a checkpoint twice.

Implementation sequence:

1. Resolve target/method/vision/processor manifests, actual imported allocations, codecs, media bounds and speculative compatibility; include conversion, activation policy and media replay/staging costs.
2. Measure runtime baseline and per-rank usable budget while preserving device-floor, workspace and recovery protections; establish host/pinned/media-store bounds.
3. Evaluate a small set of qualified execution/placement envelopes for the selected objective.
4. Compute exact per-rank persistent and peak costs for candidate resident counts.
5. Apply implementation limits and an optional administrator ceiling.
6. Initialize the chosen arenas and bounded graph resources.
7. Verify measured allocation against the plan and publish the effective profile.

Admission uses exact allocated strides, page/tail rounding, rank ownership, and library residency. Logical token rates are cross-checks.

Reserve every admitted sequence's complete future window for its declared lifetime. Short prompts, expected inactivity, presumed prefix sharing, or unimplemented offload cannot increase guaranteed capacity.

Capacity may fall between policy thresholds: retain a computed 18. Buckets aid execution; they neither round down admission nor impose a permanent 32-session ceiling.

### Method-specific resource description

Each method declares weights/state, history geometry, proposals/verification rows, target/draft tentative and feature rows, recovery, supported shapes, validity maintenance and stop/commit rules.

Guard rows follow the method's tail policy and page/pooling rounding. The reported +4 belongs to the historical receipt's verification geometry, not every MTP depth or DFlash block.

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

Add the selected profile's actual target/MTP/vision weights, all retained state, recovery, execution, graph/communication residency and overlap obligations. Neither a count in this table nor an earlier loader receipt establishes SC117 capacity. Fixed/session costs also prevent a rule that halving context doubles capacity.

Under Section 9's indexer mapping, sixteen and twenty-four MTP sessions balance to 26 and 39 indexed caches per rank. Twenty sessions produce 32 or 33 per rank; twenty-eight produce 45 or 46. Use actual worst-rank ownership, weight placement and visual residency, not averages or rounding to policy thresholds.

### Selected DFlash history cross-check

For the initial TP8/FP16 Inco draft and context C of at least 2,048 tokens, the ideal unique growing-history cost per resident sequence per GPU is:

```text
target Q8 KV + pooled indexer: (1,728 × C) bytes + 5 MiB of draft KV
target Q4 KV + pooled indexer:   (912 × C) bytes + 5 MiB of draft KV
```

At C=262,144, this is 0.426758 GiB/session/GPU for Q8 target KV or 0.227539 GiB for Q4. Add target-only weights, approximately 0.429022 GiB/rank draft payload, and the remaining ledger, including bounded raw indexer state. Shorter contexts use draft retention up to `min(C,2048)` plus tentative/rollback rows. These cross-checks exclude page rounding and do not establish final session counts.

## 9 Canonical paged history and state ownership

Store each target KV head once across its four-rank group and each completed pooled indexer history once across the box, with bounded raw pending state at the same owner. Initially compute each session/layer indexer at one canonical owner. Retain target TP8 projections and output layout. Its four indexer query heads and aggregation are distinct from the 24 target query heads and two KV heads; rank-local or per-KV-head approximations are not equivalent.

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

For valid cache length `n` measured from that cache's fixed grouping origin, retain `floor(n/4)` completed pooled keys and `n mod 4` pending raw rows. Use a four-row FP16 raw ring per indexed cache/sequence: 1,024 payload bytes before metadata, placement, padding and recovery versions. Preserve required member coordinates and logical positions. Release completed raw members once pooling and all required tail/checkpoint captures have finished and readers retire. Source: [pinned ring metadata](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/attention/qsa/metadata.py).

Protect the pre-step pending tail and bounded new raw span until compression and accepted-prefix commit resolve; writing an entire verification span into the same four-slot ring can overwrite required carry-in members. Form crossing groups from the saved tail plus new rows. If `a` new cache rows become valid, retain `floor((n+a)/4)` complete pools and the ending `(n+a) mod 4` raw tail; invalidate groups containing rejected members. Each query sees only groups complete at its own causal frontier, even when later pools are already computed. Support arbitrary chunk splits; padded rows do not advance grouping.

Checkpoint each recoverable boundary's pooled-valid frontier, pending raw values and coordinates with its other state. Whole-append rollback restores this checkpoint and discards later pools. A long pending append uses protected checkpoint tails plus bounded current-span storage. Use each cache's own validity frontier, especially MTP's shifted/deferred rows; its target-conditioned refresh rebuilds affected groups. Physical migration preserves grouping origin and membership.

### Initial owner execution

For each target attention layer:

1. Execute the existing TP8 projections.
2. Route each query head to the owner of its corresponding global KV head history.
3. Append incoming KV once at its owner, in the committed or tentative range required by the execution method; promote only the accepted prefix.
4. Construct exact indexer queries and update the bounded raw pending state and pooled keys once at their canonical owner under the same committed/tentative transaction.
5. Select visible complete pooled groups, expand all their members, and append the visible incomplete tail.
6. Dispatch selected logical positions to the KV owners.
7. Compute attention and return outputs to their original TP positions.
8. Continue the existing target output projection and distributed execution.

Batch routing and collectives by layer, not request. Preserve per-row causal boundaries, masking, pooling, ties, deterministic logical positions, and exact sparse selection. Later tentative appends must remain invisible to earlier queries. Replicated histories do not establish replicated indexer query inputs.

Transient projection duplication is initially permitted; persistent histories remain unique. Optimize it only for measured benefit.

### Prefix reuse

Prefix sharing is optional; full-context capacity must not depend on it. Reuse requires compatible target/profile identity, recurrent and Gated DeltaNet/PLE tails, PLE token/hash history, indexer pooling/selection, codec and speculative conditioning. Include immutable media content, processor/settings, grids/order and target positional state: matching text and image/video placeholder IDs with different pixels is a different prefix.

Reuse inside a group requires a matching raw-tail snapshot at that cut or replay from a valid prior target/method state boundary. A compatible complete-group boundary needs no pending members. Pooled means cannot reconstruct discarded individual raw keys.

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

The first packed MTP candidate is `unsloth/Qwen3.8-Flash-Next-GGUF`, revision `766911a6b7369840a91dbcd95f9f997acaab6cd6`, file `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`. Its file size is 2,786,568,256 bytes and SHA256 is `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6`. The inspected tensor payload is 2,775,621,632 bytes, or 2.584999 GiB: nineteen Q8_0 tensors, eleven F32 tensors, and two BF16 tensors. Source: [pinned MTP companion directory and README](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6/MTP).

The official control source is the 31 `mtp.*` tensors in `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d43b917e7706784d8bb445c9af86a3540`: 5,214,301,696 bytes of BF16 payload across the source shards. Extract only the required tensors with bounded preparation memory, retain a checksummed import manifest, and convert to the qualified SM70 representation. The shared-BF16 GGUF companion is also available for import parity checks. Source: [official tensor index](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/model.safetensors.index.json).

Both paths share the actual SC117 embedding/output tensors without replacing the target trunk. Reconcile fused/split projections, transformed RMS weights, native fusion and MTP hyper-connections. Capture completed block-47 HC residuals before the final target HC mixer: all four streams form the 10,240-wide input. Preserve the selected SGLang reference's hidden RMS normalization over that entire flattened width, its 10,240 learned weights and order, epsilon 1e-6, and one application of the `1 + weight` convention; normalize input embeddings over 2,560 separately. The pinned GGUF conversion already folds the Gemma offset into gamma: use that gamma directly, or convert once to zero-centered weights; never add one twice. Only then reshape HC into four streams for the shared hidden projection. Another engine's per-stream normalization is not equivalent merely because tensor shapes match. Never substitute the target's 2,560-wide mixed output, stream mean or DFlash features. MTP disables PLE; target PLE still advances/rolls back during verification. Qualify packed and BF16-derived control profiles separately without permanently loading both.

### Explicit MTP attention policy

The initial TensorFold MTP profile uses **QSA with compression four and complete-group token budget 2,048**, matching the inspected model-config/SGLang path. The inspected shared-Q8/BF16 GGUF companions have a final MTP compression-ratio metadata entry of zero, which selects dense attention in the inspected llama.cpp path despite the presence of indexer tensors. Therefore the importer must bind and validate the declared QSA policy against the reference; blindly interpreting that metadata would produce a different algorithm. Sources: [SGLang MTP reference](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/models/qwen4_exp_mtp.py) and [llama.cpp target/MTP implementation](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/src/models/qwen4exp.cpp).

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

Zero proposals retains weights, valid history/state and execution prerequisites for immediate reactivation; include maintenance time. Reclaiming method resources requires a configuration transition and may require reconstruction.

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

Lossless speculation is relative to the configured numerical target, not Q4/Q8 equivalence. Declare tolerances and sampling guarantees for reduction-order differences; universal bit-identical text is not promised.

Keep per-sequence RNG ownership independent of batch permutation and migration. Padded rows must not advance another sequence's RNG or state.

## 13 RotorQuant and future codec work

### Initial Q8 contract

Preserve TensorFold's current INT8 KV codec. Pin writer, reader, attention-consumption source and receipt-build revisions. Before kernel or ownership changes, record signed-code interpretation/range; grouping/layout; scale calculation, FP16 storage and reconstruction; rounding/ties; saturation; zero blocks; and nonfinite handling. Its 544 bytes per target-head K+V token establish capacity, not external-format compatibility.

Paged storage, packing, and attention must reproduce the pinned semantics. Verify byte compatibility for identical representations and qualify execution-induced numerical differences. Migration copies bytes/metadata without requantization. Require source-confirmed equivalence before calling it GGML Q8_0. A different codec requires a separately selected, qualified numerical profile; never reinterpret retained pages silently.

### Format requirements

The initial Q4 profile is **K=iso4, V=iso4**, specifically `GGML_TYPE_ISO4_0` / `block_iso4_0` at reference commit `08e025c06ab521e4fa9e5c08b80af57614543e53`. This starting profile is neither a measured V100 winner nor a universal upstream default. The author's limited four-bit comparison favors IsoQuant; qualify actual SC117 long-context quality and throughput.

Each iso4 block represents 128 values with 32 quaternion rotations and occupies 68 bytes: 64 bytes of codes plus two FP16 metadata fields. Preserve the pinned centroid/quaternion tables, reconstruction-norm correction in `norm`, and reference zero initialization of the reserved `rnorm` field. A dimension-256 K or V head uses two blocks; K and V together occupy 272 bytes per head/token/layer before allocation padding. The existing Signed128 format has different metadata and numerical semantics despite the same nominal footprint.

RotorQuant requires independent qualification of format roundtrips, zero vectors, scales/norms, rotations/centroids, quality, peak memory, prefill, and decode; Signed128's live qualification does not transfer.

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

Use **incoai/Qwen3.8-Flash-Next-DFlash2**, revision **a22a1744e85c17364dcc54a6a420af7135d4c7f6**, as the first DFlash candidate to implement and qualify against the pinned SC117 target. Import its source `config.json`, `model.safetensors`, and feature-capture contract. The initial draft profile is **fixed TP8, FP16 draft weights, FP16 physically bounded draft KV, and block size four**. The following manifest, calculations, and operations apply to this pairing.

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

Required execution work:

- Implement qwen4exp's four HC streams, target per-stream normalization, gated query projection, PLE injection and all selected mixed weight formats. Its 12,288-wide query/gate projection contains 24 query heads plus their gate. Preserve MTP's distinct flattened-HC norm axes (Section 12); reconcile HF `1 + weight` conventions with converted GGUF values exactly once.
- Implement direct GPU KV packing and tiled unpacking with exact codec metadata.
- Select vector/small-M or tiled attention and projection paths by measured complete-step behavior.
- Group routed rows by expert; use actual group sizes, not global batch size, to select GEMV, small-M, or grouped-GEMM execution.
- Reuse decoded weight tiles and activation buffers where proven useful, with explicit lifetimes.
- Preserve required FP32 recurrent state and sensitive accumulation unless separately qualified.
- Compute generation/verification logits as required; requested prompt log-probabilities need a separate workload budget.
- Reduce launch, routing, and host/device synchronization overhead where profiling identifies it.
- Audit warp masks, barriers, register/shared-memory pressure, occupancy, spills, coalescing, and actual bandwidth.

Profile indexer-query assembly, pooled scan/top-k, selected-group expansion and KV gather, selected-position routing, and KV attention separately. Reuse history tiles across query rows only when causal masks permit. Bound selected-index and unpacking work by tiles and row budgets; never expand selected KV for an entire long prefill batch. Increasing KV attention degree does not remove a canonical indexer bottleneck. Further indexer parallelism requires measured whole-step benefit and exact global selection.

Every kernel family requires native SM70 compilation and supported math/library paths. Respect Volta's independent thread scheduling and explicit synchronization requirements ([Volta tuning guide](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html)). Select paths using complete packing, communication, state, and recovery costs, not Tensor Core utilization or isolated kernel rate alone.

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

The coordinator issues one plan and epoch for all TP ranks. Ranks must not independently change proposal lengths, graph shapes, communicators, or collective participation.

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

## 18 CUDA graphs and communication

Initialize communication groups, kernel families, descriptor bounds, and maximum buffer capacities for the validated effective-profile envelope before use. A larger profile or ceiling requires new bound validation even if its bytes fit. Keep weight TP8 fixed. Build a bounded graph set and retain uncaptured execution of the same plan, kernels, and allocation contracts within the canonical serving path.

Use a small measured set of masked row/token and legal visual-shape/segment buckets with stable indirection; do not capture every combination of session count, context, proposal width, prefill size, media shape, codec, and placement. Charge padding's actual compute and storage.

### Graph validity

Stable page-table pointers alone do not establish reuse across layout changes. Validate captured direct pointers, scalar arguments, launch dimensions, collective counts, participating ranks, stream dependencies, allocator pools, and object lifetimes. All captured assumptions must remain valid.

Mask inactive/padded rows from KV/indexer writes, recurrent/convolution state, draft state, RNG, stop processing, and committed output. Validate session generation to prevent stale writes into reused slots; asynchronous media completions obey the same generation and reader-retirement rules.

Charge graph-private pools, persistent inputs, and live instances in the ledger. Share scratch only when replay ordering proves nonoverlapping lifetimes. Source: [PyTorch CUDA graph memory/lifetime documentation](https://docs.pytorch.org/docs/2.14/notes/cuda.html); qualify the actual framework build under Section 2.

### Collective ordering

Use NCCL for normal TP8 collectives. Initialize full-rank and bounded attention subgroups before serving with topology-aware rank mapping. Use AllReduce when every participant needs the result, ReduceScatter when the next operation consumes shards, and AllGather where replication is needed. Immediate ReduceScatter-plus-AllGather does not itself eliminate communication. Bounded migration uses matched NCCL P2P or qualified peer copies. Group mutually dependent two-sided sends/receives with `ncclGroupStart`/`ncclGroupEnd`, matching endpoint counts and datatypes within the globally agreed communication phase. Sources: [collective semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) and [P2P operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html).

Start with NCCL's topology-aware selection. Measure small decode-message latency, larger prefill traffic, and GPU contention during communication/computation overlap. Tune counts/bytes, scheduling, and graph integration before adding frameworks. Do not force one algorithm/protocol across message sizes without evidence, recreate communicators on active-count changes, or use changing process environment variables as runtime controls. Device-initiated NCCL, MSCCL++, or NVSHMEM needs a measured benefit justifying its additional contracts. NVLS requires newer NVSwitch/Hopper hardware and is unavailable on this V100 mesh ([NCCL controls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)).

All ranks agree on captured/uncaptured execution and compatible collective order. The issued plan fixes groups, ordering, counts, and empty contributions: cancellation, empty shards, or partial migration cannot independently suppress a worker's collective. Coordinate migration, visual-input distribution and serving with explicit streams/events and a globally ordered plan; do not inject rank-local copies or visual broadcasts into captured collective schedules. Source: [NCCL CUDA graph requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html).

Capture/replacement follows framework quiescence rules. Another host thread does not make unrelated concurrent CUDA work safe. Use uncaptured execution around recapture where needed and pause affected device work as required.

## 19 Configuration changes without restart

Expose configured, pending, and effective values separately. Normal adaptation and supported administrator changes must not restart containers, workers, or CUDA contexts; worker replacement is not a live update.

Drains cover every affected resident sequence: request-scoped work must finish/cancel and retire; retained sessions follow close/expiry/reconstruction rules. Zero activity leaves retained history, draft, recurrent/position/media and recovery obligations intact. No retained sessions does not authorize unloading request-scoped or in-flight state.

| Adjustment | Implementation policy |
|---|---|
| Scheduled batch, prefill/visual quota, legal media microbatch, bounded CPU worker quota, supported proposal length, existing graph/kernel choice | Apply at a coordinated batch boundary within allocated limits |
| KV ownership and attention degree | Use bounded byte-preserving migration and a published mapping epoch |
| Resident-sequence ceiling decrease | Restrict new admissions; preserve existing sequences and context commitments |
| Resident-sequence ceiling increase | Require the complete maximum-context and execution-envelope proof |
| Larger context with unchanged effective positional transform | Activate for all resident sequences together only after reserving enlarged commitments and validating qualified positions, state and execution bounds; otherwise keep pending until the affected profile drains |
| RoPE/YaRN algorithm, frequencies, factor, original-reference length or amplitude change | Initially drain affected resident state and retire dependent graphs/caches; validate and activate the prepared profile in process. Never change from active-count adaptation or reuse incompatible state (Section 6). |
| Smaller context | Preserve existing windows; keep pending until the homogeneous resident profile drains |
| Codec, target/visual weights, processor/numerics, structural encoder placement, draft checkpoint or speculative method | Use a qualified in-process transition, or drain all affected resident sequences and reinitialize in process |
| No supported in-process transition | Fix the setting at its qualified maximum-concurrency value; expose no live control |

Initially use one homogeneous context/quality profile for recovery, capacity and graph ownership; simultaneous profiles need separate budgeting.

Structural drains stop admissions that could indefinitely extend the old profile. Define resume/closure behavior while existing sessions finish or close under their retention contract. An idle retained session can keep the change pending: report its blocking condition or resource without forced deletion, restart, or a promise to switch at the next zero-active instant.

Byte-preserving GPU page moves retain the quantized representation. Q4-to-Q8 conversion cannot recover discarded information; enabling another speculative method may require historical features/state never retained. Neither is a lossless migration. Use qualified reconstruction where supported; otherwise drain affected sessions. Within an enabled method, zero-proposal operation is a scheduling choice only while conditioning remains valid. The active-count controller does not change MTP/DFlash, draft checkpoints, or KV precision.

Target/runtime-weight changes initially drain affected sequences, retire graphs/readers/state and load a prepared replacement. If old/new layouts cannot coexist in bounded memory, use an empty-resident in-process transition; uninterrupted service is not promised. Queue or report pending per service policy. Bind queued work to its complete requested model/context/RoPE/KV/method/tokenizer/processor/media profile; reject or retain incompatible work pending. Retained media cannot be reprocessed under changed settings without a qualified transition.

Publish a replacement/capacity only after all ranks finish loading, allocation, target/draft binding, bound initialization and required library/kernel residency. Retire/rebuild affected graphs and tuning plans. On recoverable preparation/load failure, coherently retire partial state and expose failed/pending, never unready capacity. Releasing the old model ends its service guarantee. Retry/reload a prepared profile in process without two full resident models.

## 20 Host memory and deferred architecture alternatives

The historical PLE/ngram mapping is reported as 26.822 GiB. The SC117 publisher identifies the 26.822219 GiB second shard as its IQ4_NL PLE table; confirm exact tensor payload/offsets at import. Rank file mappings can share physical pages; eight mappings do not imply eight physical copies. Residency is evictable; measure actual NUMA placement.

Profile cold/warm lookups, page faults, CPU execution, NUMA access, and bounded host-to-device transfers during prefill/decode. Preserve all sixteen bigram/trigram hashes, offsets, resets, IQ4_NL decoding, and PLE convolution. Batch lookups and eliminate proven duplicate TP-rank work, routing common results where beneficial. Shared file pages do not eliminate duplicate lookup/transfer work; PLE can dominate low-load latency after KV deduplication.

Budget host media decode, normalized pixels, optional retained projected embeddings, immutable replay sources/caches and pinned transfer stages separately from PLE and history. Bound CPU worker concurrency, queued decoded media and every buffer lifetime; do not allocate maximum video working sets for all resident sessions. These media resources do not provide attention-KV capacity.

Keep the raw-indexer RAM archive disabled. Compact indexing has no growing raw array. The historical 13.038 MiB/session/GPU ring/staging cost belongs to the full-raw layout, not an initial compact allocation or capacity-saving option.

Host paging of attention KV or retained expert weights is unimplemented in the supplied receipt and cannot supply advertised capacity or silently rescue overcommitment. The eligible structural comparison is topology-aware TP8 versus TP4 × PP2 with one distributed target. Independent TP4 model replicas and the proposed TP4 attention/dense plus EP2 layout that duplicates shardable nonexpert weights are excluded. A nonduplicating expert-placement change remains eligible only with an exact inventory and measured benefit. The TP8-specific contracts in this document govern the implementation baseline until an alternative receives separate qualification.

The pipeline candidate requires stage-specific weight/state plans, full HC transfer, transactional scheduling and a worst-rank resource proof. Keep the selected target embedding at its owning stage and serve MTP/DFlash lookups through bounded vector transfers; the final-stage target head can remain shared with its local draft. Include lookup latency in complete speculative rounds and keep the owner service schedulable during feedback waits to prevent a dependency deadlock. Add no persistent target embedding or vision copies. Changing TP/PP placement is a separate in-process weight-and-state transition, not a cache-only active-count control; retain the selected maximum-concurrency layout if that transition cannot meet the no-restart contract.

## 21 Observability and qualification

### Runtime receipt

Publish a compact machine-readable effective-profile/capacity receipt and user-facing limiting-resource explanation. Include:

- Exact target/MTP/DFlash/visual/processor/codec IDs, hardware/runtime manifest, immutable resolved RoPE/YaRN and companion position manifests/epochs, native/reference versus configured/qualified maxima, context, media envelope/placement, configured ceiling, computed resident capacity, and qualification status.
- API request counts; separate request-scoped/retained, active, queued and scheduled sequences; text/visual prefill rows, frames/pixels, prefix/media-cache hits, maintenance/replay work and target verification rows.
- Proposal policy, draft-valid state, acceptance distribution, recovery/replay work, and committed generated tokens.
- Timestamped per-rank weights, history pages, state, graph pools, language/visual workspaces, external allocations, physical free memory and protected commitments; bounded host/pinned/media-store use and queue occupancy.
- Placement degree, mapping epoch, outstanding migration bytes, transient occupancy, migration time, and admission restrictions.
- Useful text/visual prefill and committed decode rates, media decode/encode/transfer time, TTFT, inter-token latency, per-request service/fairness and error/cancellation counts.

Separate startup/current readings, measured allocations, logical reservations, and planned maxima. Do not add allocated to reserved bytes or count tentative/draft tokens as generated throughput.

### Qualification matrix

| Axis | Required cases |
|---|---|
| Artifacts, precision and method | Every released target/runtime-weight pairing and supported target KV with none/MTP; every DFlash pairing/draft format; selected NVFP4 import/conversion/activation semantics |
| Context and RoPE/YaRN | Native and every released extended positional profile; short/intermediate/near-full histories, native/new-limit boundaries, coordinate-versus-row bounds, invalid/nonfinite profile rejection, partial-RoPE amplitude parity, Q8/Q4 quality, required MTP and separately qualified DFlash, media, profile identity and in-process changes; input plus output fits the allowance |
| Concurrency | Integer sweep from 1 through capacity; detailed thresholds, exact endpoints and measured breakpoints; burst from 1 to maximum |
| Retention | Request-scoped release and intentional retention; one resident/one active, all resident/one active, all resident/all active; partial/near-full histories; supported multi-sequence accounting |
| Workload mix | Prefill only, decode only, mixed long prompts/decode, heterogeneous acceptance, routed expert skew; visual bursts with decode fairness |
| Media | Single/multiple images, aspect ratios/resolutions, interleaved text/media, video temporal order/timestamps, exact grids/rounding/temporal padding, boundary limits, explicit frame-count paths and invalid metadata |
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

Use focused synchronization/memory diagnostics for kernel risks and lifecycle races. Each gate must address a concrete correctness, resource, or performance risk; avoid tests that merely repeat implementation details.

Qualify Section 2's combined dependency manifest: inspect SM70 coverage and loaded libraries, exercise selected GEMM algorithms and every communicator at actual dtypes/message sizes, and test captured/uncaptured serving. Framework/extensions must not silently introduce unsupported kernels/backends. Use NVTX/Nsight Systems for host gaps, collective critical paths, transfers, and overlap; Nsight Compute for bandwidth, Tensor Core/SIMT activity, occupancy, and spills; focused Compute Sanitizer checks and NCCL correctness/performance tests. Separate instrumented runs from throughput receipts.

Resolve applicable compiler issues before release. NVIDIA issue 6111901, introduced in CUDA 12.8 and fixed in 13.2.2, concerns failed reconvergence under nested thread divergence. Establish applicability to the exact SM70 toolchain and any supported 12.x correction or validated source-level mitigation; assume neither V100 impact nor a backport. Retain a targeted reproducer/regression check when applicable. CUDA 13's fix does not restore SM70 compilation. Source: [NVIDIA compiler issue](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html).

### Performance gates

Use repeatable prompts, generation settings, target/draft revisions, warmup, stable clocks/thermals, and timing boundaries. Repeat enough to separate gains from noise. Compare adaptive execution with the best qualified static profile under identical context, quality, retention, and workload conditions.

Keep history ranges and reasoning/sampling settings comparable across active counts. Test different owner/slot phases at low activity, not only slot zero. Profile indexer and PLE separately to avoid attributing their latency to attention placement or proposal depth.

Report useful text prefill, visual-row processing and committed decode separately, alongside whole-step/end-to-end rates and isolated-kernel measurements. Include CPU media preparation, encoding, transfers, quantization, draft maintenance, rejected work, collectives, recovery, scheduling and amortized migration. Visual rows are not ordinary text tokens/s. Report latency distributions and fairness so throughput cannot hide starvation. Select the best measured eligible profile at each load regime.

## 22 Implementation sequence and exit gates

Use Section 2's dependency/processor manifests and Section 15's operator/library contracts throughout. Parallelize work where state/resource contracts permit; every new execution path must pass Section 21's applicable gates.

| Stage | Required work | Exit evidence |
|---:|---|---|
| 1 | Qualify SM70 stack; implement family manifests and bounded mixed-format GGUF import; bind exact SC117 IQ3_S, shared MTP, Inco draft, visual/processor artifacts and recovery contracts | Build/library IDs, complete tensor inventory, source/runtime parity and fresh per-rank resource receipt |
| 2 | Add canonical page/session/media metadata, checked addressing, typed arenas, full-context commitments, accepted-prefix/sampling/PLE/streaming APIs and SM70 recurrent/convolution kernel regimes using Q8 | Target/state parity, bounded prefill/decode work, transcript, cancellation, media identity and page-class guarantees |
| 3 | Compact raw indexer storage to bounded pending/transactional state and remove persistent KV/pooled-indexer replication through owner execution; retain weight TP8 | Exact growing/fixed history bytes, rank balance, group transactions and output/selection equivalence |
| 4 | Add attention degrees, exact merge, topology-aware mapping and separate MTP-cache placement | Correct degree 1/2/4 operation; placement/communication profiles |
| 5 | Port RotorQuant Q4 and direct SM70 pack/unpack/attention | Layout, quality and long-context gates; no full-history mirror |
| 6 | Implement required image/video processor, FP16 vision path, target positions/PLE inputs, bounded staging/replay and mixed scheduling | Reference parity, media quality, fixed placement and complete maximum-work resource envelope |
| 7 | Implement required shared-Q8 QSA MTP with actual SC117 aliases, selected Q8/Q4 KV, full-prefix text/visual initialization and refresh | Pinned-control/imported-weight parity, useful speculation, recovery and complete capacity receipts |
| 8 | Add bounded graphs, global controller and mixed media/prefill/decode scheduling | Safe immediate load changes; integer sweep and detailed threshold/breakpoint evidence |
| 9 | Add bounded migration, epoch publication and hysteresis | Intermediate states fit; serving remains correct; adopted migrations yield net gains |
| 10 | Integrate IncoAI DFlash2, TP8 HC capture, FP16 kernels/selector, qualified stochastic verifier and bounded SWA/features with separate draft positions | Independently qualified SC117/Q8-KV and SC117/RotorQ4-KV text/visual correctness, supported sampling, capacity, acceptance and performance |
| 11 | Complete in-process administration, observability, sustained-load tests and operations; implement native/YaRN positional profiles and extended Q8/Q4 capacity/quality qualification with required MTP and separately qualified DFlash | No tuning restart; immutable position identity, bounded phase resources, native/extended parity and quality, lifetime commitments, readiness/failure contracts and full-load receipts/benchmarks |
| Additional profiles | Qualify other target/draft artifacts, including the NVFP4 source/runtime contract | Conversion parity, quality/acceptance, actual allocations and recomputed activity/capacity profiles in one serving path |
| Later | Evaluate SAW and other architectural alternatives | Separate quality/capacity/performance evidence before inclusion |

MTP is a required deliverable independent of DFlash availability or speed. The initial default uses MTP; DFlash is an independently selected method profile. Both need their own geometry and multimodal gates, and neither is silently disabled to advertise more capacity.

Retain each stage's commit/build ID, workload, resource receipt, results and limits. Qualification does not extend implicitly to larger contexts, different media envelopes or higher concurrency.

## 23 Integration inputs and unresolved measurements

The model choices are settled; implementation supplies these remaining bindings and measurements:

1. **Artifacts/fork (Sections 2, 12–15):** SC117 binary census; tokenizer/template; target/MTP/DFlash/vision tensor maps/transforms and allocations; ordinary Q8, RotorQuant Q4 and explicit MTP QSA bindings. Historical receipts remain comparisons.
2. **Server/stack (Section 2):** installed CPU/DIMMs, GPU UUID/ranks, links, PCIe/NUMA, firmware and newest compatible build/library IDs; SM70 and compiler/library qualification.
3. **Multimodal (Sections 2, 16):** pinned processor/decoder, service media bounds, fixed encoder placement, operator/position parity, CPU/GPU/pinned/replay peaks and task quality.
4. **Speculation (Sections 12, 14):** MTP import/norm/HC and text/visual refresh; Inco DFlash2 TP8 features, FP16 conversion, selector, SWA, separate positions and useful blocks for SC117 with each KV codec.
5. **Recovery:** checkpoint counts/cadence, retained media/replay requirements, replacement overlap, failure behavior and proven shared lifetimes.
6. **Execution (Sections 10–18):** row/prefill/media budgets, recurrent/convolution kernels/chunks, graphs, ownership degree, draft width, migration cap and hysteresis across all activity/workload regimes.
7. **Service/API (Sections 5–7, 19):** context/position selection and quality thresholds; maximum capacity then full-load decode with fairness/latency/queue limits; lifetimes/multiplicity, input commit/cancellation, output backpressure, drains and readiness/failure behavior.

Publish the qualified integer maximum from the complete post-import ledger for the selected context/method/media envelope. Historical weights establish no SC117 limit; 20-Q8/32-Q4 remain reference points, never promises or threshold ceilings.

## 24 Acceptance checklist

All detailed requirements and applicable Section 21 gates remain mandatory. Release also requires this cross-section checklist:

- Explicit context/RoPE, numerical/method/media profiles, capacity and limiting-resource explanation; extended windows pass every Section 6 gate.
- Exact SM70 build/loaded-library identities and resolved compiler/library issues; combined-stack qualification and bounded workspace/communication (Sections 2, 21).
- One manifest-driven serving path; independent source/runtime weight, activation and target/draft KV choices; capacity from imported allocations (Sections 2, 7).
- Every sequence can reach its promised expanded input-plus-reply window with selected speculation/recovery. Image/video → text passes declared media bounds; native audio/media generation stays outside scope (Sections 6–7, 16).
- Account for independent sequences; retire request-scoped state at completion and preserve retained/input-commit semantics through prefill cancellation/resumption (Sections 6, 16).
- Stable TP8 baseline or Section 20-qualified single-target alternative; no additional persistent target/vision copies for throughput. Unique, balanced KV/pooled histories and bounded raw state pass chunk/speculation/rollback/prefix tests (Sections 9–11).
- Q8/RotorQuant Q4 layouts and SM70 kernels pass Section 13 gates; Signed128 results do not qualify RotorQuant.
- Complete worst-rank and host/pinned/replay ledgers support useful prefill/verification and the largest promised media unit at full residency (Sections 4, 7).
- Maximize qualified capacity, then near-full-history/full-active committed decode; preserve commitments at low load. Measure every integer activity count, thresholds/endpoints/breakpoints, without a 32-session ceiling (Sections 5, 17, 21).
- Sudden load increases, partial migrations, cancellations, and slot reuse are correct and bounded.
- Every Section 12 state type commits/recovers correctly; padded rows cannot mutate sessions. Unsupported verification rejects before mutation; publication/cancellation follows canonical commit.
- No tuning container/worker/CUDA-context restart; unsupported live settings remain fixed or pending under Section 19.
- Structural changes preserve lifetime, quality, context, media fidelity and selected-method commitments (Section 19).
- All-rank readiness precedes replacement publication. Recoverable failure exposes pending/unavailable state and in-process recovery without duplicate full models (Section 19).
- Section 21 receipts distinguish observations, allocations, commitments, estimates and qualification; report useful text/visual prefill, committed decode, media work, latency, fairness and overhead separately.
- Required shared-Q8 QSA MTP passes Section 12's import/alias/norm, full-prefix text/visual conditioning and selected-KV gates. IncoAI DFlash2 independently passes Section 14's SC117/V100/TP8 feature, bounded-window and position gates. Additional pairings require their own complete qualification; SAW follows core implementation.

## 25 Evidence and source register

### User-supplied deployment evidence

Baseline values and historical qualification descriptions come from the supplied 2026-10-25 04:16 UTC accounting receipt. Its files on the user's TensorFold machine were not directly inspected:

```text
/home/kklouzal/TensorFold/task-artifacts/v100-indexer-archive/production-gpu-history-smoke/stats-initial/result.json
/home/kklouzal/TensorFold/task-artifacts/v100-indexer-archive/report-facts/weights.json
/home/kklouzal/TensorFold/task-artifacts/v100-indexer-archive/report-facts/frontier-all-layouts.json
```

Unique-history capacity arithmetic derives from that receipt's geometry and Section 9's pinned compact-indexer construction. Proposed algorithms, policies, placement, and gates are implementation design, not observations of a modified deployment.

### Exact artifact inspection

SC117 IQ3_S shard and visual-file sizes/SHA256 values were verified from immutable public file pointers. The selected SC117 binary headers, tokenizer and payloads were not independently inspected; its complete tensor census, numerical import and target-specific memory plan remain explicit importer gates. The publisher's mixed-format list is evidence of required support, not a verified per-format payload census.

The prior official/shared MTP and Inco draft metadata/header inspections supply their stated configurations and payload arithmetic; no full large target/MTP/draft payload was downloaded or executed. Both official processor JSON files were fully retrieved and hash-checked, and the pinned processor/model/speculative source paths were inspected. Source contracts do not establish SC117 acceptance, multimodal quality or V100 performance. Full-file integrity, imported-value parity and runtime qualification remain installation gates.

### Primary external references

Sources are cited beside their requirements: platform, model, visual/processor, tokenizer, NVFP4 and dependencies in Section 2; compact QSA storage/selection in Section 9; MTP in Section 12; RotorQuant/SAW in Section 13; DFlash2 in Section 14; SM70 libraries in Section 15; multimodal execution in Section 16; graph/collective contracts in Section 18. Preserve each source's stated scope; modern-GPU evidence does not qualify this V100 implementation.

Additional pinned implementation and background references:

| Reference | Use and boundary |
|---|---|
| [Transformers model implementation](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py) | Geometry, PLE indexing and normalization; preserve selected GGUF tokenizer/template and converted values |
| [SGLang QSA parser](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/attention/qsa/config.py) | Section 12's explicit QSA MTP policy; companion zero-ratio metadata does not establish dense/sparse equivalence |
| [SGLang GemmaRMSNorm](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/layernorm.py) and [GGUF norm conversion](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/conversion/qwen.py) | Exact MTP normalization axis/weights and single application of the converted Gemma offset; matching shapes alone do not establish numerical parity |
| [Original DFlash paper](https://arxiv.org/html/2602.06036v2) and [repository](https://github.com/z-lab/dflash) | Method background; distinguish DFlash 2 model changes from a serving engine named Spec V2 |
| [vLLM context parallelism](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/), [dynamic speculation](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/), and [NanoCP](https://arxiv.org/abs/2605.21100) | Related concepts, not drop-in qualification for TensorFold, stock MTP, or this topology |

Pin the recorded artifact/reference revisions, final TensorFold implementation, and full runtime stack in each receipt. Changes to sources, codecs, kernels, feature contracts, or numerical settings invalidate affected qualification until rechecked. Label every capacity/performance claim as supplied observation, arithmetic estimate, header plan, or live-qualified result.
