# TensorFold / iCat V100 integration handoff — final draft

Finalized 2026-10-24 for [`kklouzal/TensorFold`, branch `python-0.6`](https://github.com/kklouzal/TensorFold/tree/python-0.6), pinned to commit **`94d84f510f87ee27950d821626a3510bff9c5625`**. This document combines the eight-V100 experiments on `ow-ai-01` with a read-only review of that consolidated source.

**Document authority:** this is the implementation handoff for the consolidated branch. The [first draft](TensorFold-iCat-V100-integration-handoff.md) remains preserved as historical context. All TensorFold implementation references in this final draft resolve to the single pinned baseline.

**Implementation plan:** extend the consolidated TensorFold branch with a dedicated SM70 execution provider, the exact GGUF checkpoint path already verified on this host, and TP8 concurrent inference. Retain TensorFold's model, request, sampling, cache-identity, and ownership infrastructure. Establish FP16 execution with FP32 recurrent state, then decode graphs, native MTP, and calibrated/lower-bit KV. Changing a compute-capability check alone will not make the existing CUDA implementation work on V100.

**CUDA target policy:** target the **latest released CUDA 12.x toolkit and patch level that explicitly supports Volta/SM70**, and recheck the newest SM70-supported CUDA release whenever selecting new foundation pins. As verified on 2026-10-24, that target is **CUDA Toolkit 12.9.2**. The CUDA 12.8.1 / PyTorch `2.10.0+cu128` environment recorded below is the historical measured iCat baseline, not the target version for the new TensorFold integration. [NVIDIA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive), [CUDA 12.9.2 SM70 compiler targets](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-compiler-driver-nvcc/index.html#gpu-name-gpuname-arch).

**Evidence boundary:** the performance and working configurations below are from **1Cat-vLLM**, not a TensorFold V100 implementation. TensorFold source was reviewed; TensorFold kernels were not built or run on this host during preparation of this document. Proposed improvements are explicitly unmeasured. The current iCat service was not stopped or reconfigured for this documentation task.

## Contents

1. [What to preserve first](#1-what-to-preserve-first)
2. [Authoritative baseline and implementation scope](#2-authoritative-baseline-and-implementation-scope)
3. [Host, toolchain, and model identity](#3-host-toolchain-and-model-identity)
4. [Results actually measured](#4-results-actually-measured)
5. [Working iCat configuration](#5-working-icat-configuration)
6. [Source map and port boundaries](#6-source-map-and-port-boundaries)
7. [FP16 execution is a substantive port](#7-fp16-execution-is-a-substantive-port)
8. [Checkpoint and GGUF integration](#8-checkpoint-and-gguf-integration)
9. [Eight-GPU parallelism](#9-eight-gpu-parallelism)
10. [Attention, indexer, and cache contracts](#10-attention-indexer-and-cache-contracts)
11. [FP8 calibration and local patches](#11-fp8-calibration-and-local-patches)
12. [Exact six-context calculation](#12-exact-six-context-calculation)
13. [Lower-bit KV: an existing TensorFold advantage](#13-lower-bit-kv-an-existing-tensorfold-advantage)
14. [PLE / n-gram memory](#14-ple--n-gram-memory)
15. [MTP, graphs, and sampling](#15-mtp-graphs-and-sampling)
16. [Scheduling and benchmark lessons](#16-scheduling-and-benchmark-lessons)
17. [Failed paths and limitations](#17-failed-paths-and-limitations)
18. [Integration sequence and acceptance gates](#18-integration-sequence-and-acceptance-gates)
19. [Optimization opportunities](#19-optimization-opportunities)
20. [Build, disk, and operational practices](#20-build-disk-and-operational-practices)
21. [Local evidence and primary sources](#21-local-evidence-and-primary-sources)

## 1. What to preserve first

The highest-value findings from this host are:

- **CUDA graphs were the largest early decode improvement.** The eager TP8 baseline was around 6 tokens/s; graphs without MTP reached 47.76 tokens/s. Native MTP then improved single-stream decode further. Establish graph replay before spending time on minor fusions.
- **TP8 works in iCat; its fastest specialized kernels do not all support TP8.** Generic/correct fallback paths are essential. A TP4-only fast path must not become an unconditional SM70 default.
- **Use FP16 model execution with FP32 recurrent state.** FP8 is a cache storage format here, decoded in software. This is not native V100 FP8 Tensor Core inference.
- **Keep compressed weights compressed.** The demonstrated checkpoint is a mixed-format GGUF, not TensorFold's MLX affine or NVFP4 layout. A nominal bit width does not identify a compatible kernel.
- **PLE host residency and KV-history host offload are different features.** The working service retains the large packed n-gram table in system RAM and attention history on GPUs.
- **FP8 target and MTP caches need checkpoint-specific scales.** We collected and loaded scales for both. The local GGUF scale-overlay loader patch is part of the reproducible configuration.
- **Full-context capacity is not twice the FP16 capacity just because the main KV payload halves.** Indexer history, recurrent states, graphs, workspaces, allocation granularity, and fixed per-request state remain.
- **The final six-request result is 233.55 aggregate decode tokens/s, or 38.93 per stream on average.** Including prompt processing, aggregate output throughput was 64.05 tokens/s. These are different metrics.
- **Six independent full contexts were not load-tested.** The 237,598-token cap is an exact bound under the profiled allocator model, not an observed six-full-context stress-test result.

The useful deliverable is a small, understandable operator/provider boundary with shape and dtype admission, followed by correct TP and state management. Copying all of vLLM would import substantial scheduler, loader, cache, and distributed assumptions that TensorFold already handles differently.

## 2. Authoritative baseline and implementation scope

### Single source baseline

| Field | Authority |
| --- | --- |
| Repository | `https://github.com/kklouzal/TensorFold` |
| Working branch | **`python-0.6`** |
| Reviewed commit | **`94d84f510f87ee27950d821626a3510bff9c5625`** |
| Reviewed tree | `3324100cf9d150ee6248472c9dec5464895640fa` |
| Snapshot date | 2026-10-24; reviewed head's commit timestamp is 04:21:23 UTC |
| iCat source reference | `6a143197e6e5f6fa77bd483df0c5a09dcea6aa20`, plus the two recorded local Python overlays |

Begin implementation from this commit on the maintained branch and record every resulting implementation/build revision. All **31 previously inspected CUDA/family/contract files have identical Git blob identities at this head**. The technical findings carry forward; consolidation does not itself implement SM70, FP16 family execution, or TP8 concurrency.

The consolidated source adds native build-preparation IO tools and their documentation/tests: `tools/gb10_capture.py`, `gb10_controller.py`, `gb10_io.py`, and `gb10_progress.py`. Its README reports **58 source/fault controls passed**; a new actual capture and ARM build remain **UNRUN** there. These tools provide preparation/ownership infrastructure, with their stated evidence limits, rather than V100 kernel or performance qualification.

### Existing infrastructure versus required integration

| Area | Present in the pinned baseline | Work required for this V100 implementation |
| --- | --- | --- |
| CUDA build selection | Architecture-qualified extension names/locks; default minimum capability 8.9 | Add a provider-specific SM70 route using the latest SM70-supported CUDA 12.x release (currently 12.9.2), while preserving modern-provider admission. |
| Execution precision | Explicit BF16 buffers, rounding, projections and attention | Implement FP16 execution throughout the family, retaining FP32 recurrent state and required reductions. |
| Flash Next loading | MLX affine, ModelOpt/NVFP4 and EXL3 | Add the exact mixed-format GGUF reader, descriptors, model mapping and legal TP shard plans. |
| Tensor parallelism | Flash Next TP1/TP2; TP2 serves one stream | Extend complete model/communication/state handling to TP8 with concurrent requests. |
| Expert parallelism | Distributed expert banks with a rank-zero backbone | Reuse applicable ownership/communication work while implementing attention/backbone/cache sharding separately. |
| KV formats | BF16, INT8, INT4, RotorQuant variants and ordered K/V pairs | Add raw-half/FP8 identities as needed; adapt readers/execution to SM70 and qualify each selected format. |
| Graphs and state lifetime | Single/multi-stream graphs, context/parity keys, owner retention and memory planning | Extend reachable shapes, provider readiness, TP collectives and growth/retirement behavior. |
| PLE tables | Host/SSD tables, read-ahead and lifetime infrastructure | Add packed IQ4_NL rows and validate hash/index/history parity. |
| MTP and sampling | Family MTP and exact sampling/state machinery | Integrate the stock matching draft and target IO, then validate accepted-prefix state and sampling. |
| Build/evidence tooling | Immutable source/context capture, bounded controller IO, progress/retirement contracts | Prepare separate V100 foundation/input identities and obtain native installed-image/model receipts. |

### Decisions for the first implementation

1. **Reproduction target:** the verified Huihui mixed-GGUF target and extracted stock MTP checkpoint in section 3. Reuse the existing files and calibration provenance.
2. **Hardware/execution:** all eight V100s through TP8/PP1, FP16 model execution, FP32 recurrent state, native RoPE, text first, packed PLE in system RAM, GPU KV history.
3. **Bring-up sequence:** provider and loader → one-stream TP8 serial reference → decode graphs → concurrent TP8 → MTP depth 2 → calibrated FP8 and lower-bit-cache experiments.
4. **Performance reference:** reproduce the iCat workload conditions, including a genuine six-way decode interval; compare steady decode and end-to-end performance separately.
5. **Capacity authority:** compute TensorFold's memory budget. The iCat 237,598-token six-context cap is a reference calculation, not a TensorFold default or completed full-context stress test.
6. **Optimization order:** address measured prefill, packed GEMM/expert, GDN 2/6-head, NUMA and cache-codec costs after correctness. Pipeline parallelism, full compilation and alternate drafters follow the established baseline.

Preserve repository requirements: read `AGENTS.md`, `README.md`, `CONTRIBUTING.md` and affected family/deployment guides; update public capabilities, flags and qualification status with implementation changes. The retained `TF_FOUNDATION_STATE` reader fix concerns toolkit-state **file IO and cleanup**, not CUDA Unified Memory.

Final review manifest: [tensorfold-integration-final-review.json](qwen-flash-next/research/tensorfold-integration-final-review.json). Immutable source: [TensorFold at 94d84f5](https://github.com/kklouzal/TensorFold/tree/94d84f510f87ee27950d821626a3510bff9c5625).

## 3. Host, toolchain, and model identity

### Hardware

| Component | Observed configuration |
| --- | --- |
| Host | `ow-ai-01`, Linux x86-64 |
| GPUs | Eight Tesla V100-SXM2-32GB, compute capability 7.0 |
| CPU | Two Xeon Gold 6130 sockets; 32 physical cores / 64 logical CPUs |
| NUMA | Two nodes; node 0 CPUs `0-15,32-47`; node 1 CPUs `16-31,48-63` |
| RAM | About 376 GiB, with 8 GiB swap |
| GPU power policy | Existing 200 W caps retained during the experiments |
| Root filesystem | About 223 GiB, Btrfs, on SATA SSD |
| Free disk after cleanup | About 39.6 GiB; the earlier near-full state was real |
| Other disk | Another approximately 223 GB SSD was not formatted or mounted for this work |

GPUs 0–3 are local to NUMA node 0; GPUs 4–7 to node 1. Each group of four has NVLink connections. Cross-group pairs include NVLink bridges `0↔4`, `1↔5`, `2↔6`, and `3↔7`; many other cross-group paths are `SYS`. This is not an all-pairs uniform NVLink fabric. Capture fresh `nvidia-smi topo -m` output before choosing ranks or making communication claims.

Volta's Tensor Cores consume FP16 matrix inputs and can accumulate into FP32. INT8 packing/arithmetic support does not imply INT8 Tensor Core support or an automatic INT8-cache speed advantage. FP8/INT4 storage still requires a compatible decode-and-compute path. [NVIDIA Volta tuning guide](https://docs.nvidia.com/cuda/volta-tuning-guide/).

### Historical iCat software stack — measurement baseline

- Runtime source: `1CatAI/1Cat-vLLM`, commit **`6a143197e6e5f6fa77bd483df0c5a09dcea6aa20`**.
- Local checkout: `/home/kklouzal/qwen-flash-next/1cat-vllm`.
- Image tag: `local/1cat-flash-next:6a143197`.
- Image ID: `sha256:6dd9632766cb2e66044efb3bc9461c5a6c9e99ca83f9408194b4fd99f56554a3`.
- Base: `nvidia/cuda:12.8.1-devel-ubuntu24.04`.
- Python 3.12; PyTorch **`2.10.0+cu128`**; Triton **`3.6.0`**; TileLang **`0.1.10`**.
- Runtime identifies as `1.5.1+g6a143197.cu128`.
- Host driver observed during setup: `580.178.04`.
- Native architecture settings: `TORCH_CUDA_ARCH_LIST=7.0`, `CMAKE_CUDA_ARCHITECTURES=70`.

**The image ID alone is insufficient to reproduce the final runtime:** two Python source files are bind-mounted over the image. They are described in section 11 and preserved in the local patch artifact.

The consolidated branch's deployment guide selects CUDA **13.4.3**, AMD64 Torch **`2.16.0.dev20261017+cu134`**, and Triton **`3.9.0+gitfc8d31a5`** for its existing foundation. Preserve that modern-GPU foundation and add a dedicated V100 profile with its own lock/artifact identities.

For the V100 profile, select the **newest released CUDA version that still supports SM70**, including the latest applicable CUDA 12.x update. The verified selection as of 2026-10-24 is **12.9.2**: NVIDIA's archive lists it as the newest 12.x release, and its compiler documents `compute_70` / `sm_70` targets. CUDA 13 removed offline compilation and library support for Volta. [Toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive), [CUDA 12.9.2 compiler targets](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-compiler-driver-nvcc/index.html#gpu-name-gpuname-arch), [CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html).

Resolve the newest compatible PyTorch, Triton, TileLang and CUDA-library combination that retains actual SM70 support; build compatible components where necessary. Verify both compiler targeting and installed framework/library execution on V100. A toolkit supporting SM70 does not establish that an arbitrary framework wheel contains SM70 kernels. Record exact versions, native architecture coverage and artifact hashes. Recheck release/support information before future updates instead of treating either 12.9.2 or the historical 12.8.1 pin as permanent. The 12.9.2 selection is an integration target awaiting native qualification; all existing benchmark numbers retain their original 12.8.1 conditions.

### Target checkpoint

```text
Repository: huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF
Revision:   411f1f7349c9711e6f75aa8d20e1612eb3a4d55d
Preset:     unsloth-UD-Q4_K_XL-V2
Local:      /home/kklouzal/qwen-flash-next/models/huihui-v2/
            unsloth-UD-Q4_K_XL-V2/
Entrypoint: Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
Total:      111334654784 bytes across four verified shards
```

The first shard is legitimately metadata-only: its zero tensor count is not a corrupted download. All shard SHA256 checks completed. Do not redownload the checkpoint to fix that observation.

Relevant model metadata:

- Architecture `qwen4exp`; 48 decoder layers; hidden width 2,560.
- 512 routed experts, top 10 selected; routed and shared intermediate widths 640.
- HC has four residual streams, making the wide residual representation 10,240 values per row.
- Native context 262,144; native RoPE theta 10,000,000; rotary dimension 64; sections `[11,11,10,0]` in GGUF metadata.
- PLE n-gram size 3, eight heads per n-gram, per-layer input width 160.
- Under TP8, DeltaNet local key/value head counts are **2 / 6**.

The preset is a **mixed GGUF quantization policy**, not a promise that every tensor is uniformly four bits. The packed PLE table uses IQ4_NL in this checkpoint. Preserve actual tensor format metadata throughout loading and dispatch.

### MTP checkpoint

```text
Source:   RadixArk/Qwen3.8-Flash-Next-NVFP4
Revision: 7b719225242aacd3dbd3f9407468c2ee9a9d2594
Extract:  31 mtp.* BF16 tensors
File:     models/mtp-stock/model.safetensors
Bytes:    5214305568
SHA256:   8e14b1d2f2286e7421f68cde1d5f4028c50bd1c704f25e267d430fc46e4c6e23
```

The draft is loaded for FP16 execution. Its copied configuration has the target checkpoint's quantization declaration removed because these extracted draft tensors are BF16, not NVFP4 expert banks. It shares appropriate target IO weights in the working TP8/PP1 route. This is a stock draft paired with the abliterated target, not an extracted abliterated draft; target verification remains authoritative, and acceptance must be measured. A separate calibrated overlay directory, `models/mtp-fp8`, hard-links the large file and adds small KV-scale tensors; it does not duplicate the draft's 5.21 GB payload.

## 4. Results actually measured

Rates below are decode rates unless explicitly labeled end-to-end. Different serving configurations, prompt nonces, warmup state, output lengths, and speculative acceptance mean this is a progression of observations, not a single controlled experiment proving every individual change's isolated effect.

| Experiment | Observed result | Evidence / qualification |
| --- | --- | --- |
| TP8 eager baseline | Around 6 tokens/s, one stream | Early functional baseline; useful indication of launch/host overhead. |
| FP16 KV, CUDA graphs, no MTP | 47.757 tokens/s, one stream | Two measured 8,183-input / 512-output rounds. |
| FP16 KV, MTP depth 4 | 76.509 tokens/s, one stream | Same initial benchmark family. |
| FP16 KV, MTP depth 2 | 77.148 tokens/s initially | Depth 2 selected over 4 as the practical baseline. |
| Later warm FP16/MTP2 single-stream runs | Approximately 83–88 tokens/s | Configuration and prompt/acceptance variation; not one universal fixed rate. |
| FP16 KV, two concurrent streams | 136.397 aggregate tokens/s | Original two-slot matched run; another three-slot-config C2 run averaged 133.506. |
| FP8 KV, six concurrent streams | **233.552 aggregate; 38.925 mean per stream** | Final two measured rounds, after one warmup; all six overlapped in decode. |
| Same final six-stream run, including prefill | **64.048 aggregate output tokens/s** | Average batch duration about 95.9 s; mean TTFT about 74.0 s. |

Final six-stream measured rounds:

| Round | Aggregate decode tokens/s | Mean per stream | Shared decode interval |
| --- | ---: | ---: | ---: |
| 1 | 233.966 | 38.994 | 19.836 s |
| 2 | 233.137 | 38.856 | 20.992 s |

Each measured request had **8,187 input tokens and 1,024 output tokens**, temperature 0, forced generation with EOS ignored for the throughput test, and an independent early prompt nonce. Both rounds had six active streams, no cache preemptions, and no OOMs. Each stream contributed well over 128 tokens within the shared interval. The warmup result, 227.863 aggregate tokens/s, is excluded from the reported average.

During that benchmark, per-GPU peak memory was 29,472 MiB, minimum free memory 3,023 MiB, and peak utilization 98–100%. These are observed peaks, not sustained-bandwidth or Tensor Core occupancy measurements.

Separate functional observations:

- Arithmetic `37 × 41 = 1517`, automatic tool call and result continuation, reasoning/answer separation, and natural EOS passed.
- A model-card-sampling reasoning request calculated `7^2026 mod 1000 = 649`. Its 4,096-token allowance cut off the final answer; rerunning with an 8,192-token allowance completed naturally at **4,108 generated tokens**. This was a test output-budget issue, not a demonstrated wrong arithmetic answer.
- A prior **single-request, FP16-KV** retrieval probe used 261,625 input tokens and 11 output tokens, recovered `ORBIT-7391-MAPLE`, and ended naturally after about 378.7 s total. Eleven output tokens are insufficient for a useful decode-throughput claim.
- That single needle check is not a general long-context quality benchmark and does not qualify full-context FP8 or six-full-context operation.
- The requested 1–6 concurrency sweep was later shortened by the user to the six-request test. Do not invent missing FP8 C1–C5 results or call the interrupted FP16 C3 experiment a completed baseline.

Primary local result: [fp8-c6-results.json](qwen-flash-next/research/fp8-c6-results.json). Raw streamed requests and timing events: [fp8-c6-bench-6.json](qwen-flash-next/logs/fp8-c6-bench-6.json).

## 5. Working iCat configuration

The current authority is [runtime.json](qwen-flash-next/runtime.json), launched through [serve.py](qwen-flash-next/serve.py) and [compose.yaml](qwen-flash-next/compose.yaml). The following is a reference configuration, **not TensorFold CLI syntax**:

```json
{
  "tensor_parallel_size": 8,
  "pipeline_parallel_size": 1,
  "max_model_len": 237598,
  "max_num_seqs": 6,
  "max_num_batched_tokens": 8192,
  "gpu_memory_utilization": 0.9,
  "enforce_eager": false,
  "speculative_tokens": 2,
  "kv_cache_dtype": "fp8_e4m3",
  "long_prefill_token_threshold": 1024,
  "long_prefill_token_threshold_adaptive": true,
  "mixed_prefill_step_latency_ms": 0,
  "compilation_config": {
    "mode": 0,
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [3, 6, 9, 12, 15, 18],
    "max_cudagraph_capture_size": 18
  }
}
```

Other consequential settings:

- Model dtype `half`; recurrent/SSM cache dtype **`float32`**.
- Chunked prefill and prefix caching with Mamba cache mode `align`.
- Main attention KV on GPU: `qsa_host_kv=false`.
- `sm70_gdn_verify=false`, `sm70_gdn_spec_replay=false` for this TP8 geometry.
- `sm70_greedy_verify=true`, `sm70_draft_single_graph=true`.
- GGUF: original expert/embedding storage, expert arena enabled, canonical dense storage, device transcode enabled, one expert load-prefetch bank setting.
- MTP method `mtp`, two speculative tokens, greedy draft proposals, safetensors draft loading.
- Language-model-only/text operation; no vision qualification on these V100 experiments.
- Automatic tool choice with `qwen3_coder` parsing; `qwen3` reasoning parser.
- Default thinking enabled with preserved thinking and `reasoning_effort=xhigh`.
- Model-card defaults T=1, top-p=.95, top-k=20, min-p=0, presence penalty=0, repetition penalty=1. Throughput probes explicitly override sampling and thinking as recorded in their payloads.

Environment choices that mattered include `VLLM_USE_V2_MODEL_RUNNER=1`, `NCCL_GRAPH_REGISTER=0`, `PYTORCH_ALLOC_CONF=expandable_segments:True`, strict QSA FP8 scales, persistent GEMM LUT files, and persistent Triton/TileLang/Inductor caches. These names are iCat/vLLM-specific; port their purpose where needed rather than adding unused variables to TensorFold.

Local API: `http://127.0.0.1:8000/v1`, model alias `qwen3.8-flash-next-abliterated`.

## 6. Source map and port boundaries

All iCat paths in this table refer to the pinned checkout. Abbreviated `layers/`, `kernels/`, and `model_loader/` entries are beneath `vllm/model_executor/`; "family" means `vllm/models/qwen4_exp/nvidia/`; bare native operation filenames are beneath `csrc/sm70_turbomind/ops/`. TensorFold's shared CUDA paths are beneath `src/tensorfold/`, and its family CUDA paths beneath `src/tensorfold/families/qwen4_exp/`. The destination column identifies an integration seam, not files already containing a V100 implementation.

| Function | iCat source to inspect | TensorFold integration seam |
| --- | --- | --- |
| SM70 build and native dependency closure | `CMakeLists.txt`, `setup.py`; `csrc/sm70_turbomind/ops/`; bundled `lmdeploy/src/turbomind/`; `csrc/torch_bindings.cpp` | `src/tensorfold/cuda/build.py`; a separate SM70 extension/provider and its packaging |
| Bundled attention/GDN libraries | `flash-attention-v100/`, `flash_qla/`; `setup.py` bundle helpers | Separate, versioned SM70 dependencies or extracted operators; installed-wheel ABI and packaging checks |
| Packed GEMM/provider dispatch | `vllm/model_executor/kernels/linear/sm70_provider.py`, `sm70_dense.py`, `mixed_precision/sm70_gguf*.py`; TurboMind `gemm_runtime.*` | `cuda/precision.py`, CUDA QMM faces, family `cuda/qmm.py` and `cuda/bf16.py` |
| Native GGUF tensors and sharding | `model_loader/gguf_adapters/qwen4exp.py`; `model_loader/gguf_loader.py`; `layers/quantization/gguf_*.py` | Family `cuda/reader.py`, `weights.py`, `weight_types.py`; new format-specific loader/weight descriptors |
| Routed experts, arenas, and reductions | `layers/quantization/gguf_expert_storage.py`, `gguf_moe.py`; `layers/fused_moe/sm70/`; `ops/gguf_moe_gu_sm70.cu` | `cuda/experts*`, expert authority/cache interfaces, family expert and host-memory providers |
| FP16 narrow and batched projections | Family `sm70_fp16_gemv.py`, `sm70_fp16_hc.py`; native HC/GDN input/row reduction sources | Family `cuda/bf16.py`, `qmm.py`, `hc_fused.py`, `hc_upmix.py`, `glue.py` |
| QSA indexer and sparse attention | Family `qsa.py`, `indexer_qsa.py`, `ops/qsa*.py`; native `qsa_indexer_sm70.cu`, `qsa_prep_sm70.cu`, `qsa_dense_sm70.cu` | Family `cuda/attention.py`, `attn_multi.py`, `forward.py`, `kvcache.py` |
| Software FP8 conversion | `vllm/models/deepseek_v4/common/ops/fp8_software.py`; QSA cache write/read call sites | Dedicated SM70 KV codec and reader; format descriptors / cache identities |
| DeltaNet prefill/decode/replay | `layers/fla/ops/gdn_*`; `layers/fla/ops/sm70/`; native `gdn_verify_sm70.cu` | Family `cuda/gdn.py`, `gdn.cu`, `gdn_io.*`, `gdn_multi.py`, `state.py` |
| Packed PLE lookups | `model_executor/kernels/ple/gguf_pinned.py`, `gguf_packed_result.py`; family `gguf_embedding.py`, `ple_layer.py`, `sm70_ngram_input.py` | Family `host_table.py`, `table_file.py`, `ple_lifetime.py`, `cuda/ngram.py`, host/SSD table providers |
| MTP and shared IO | Family `mtp.py`, `sm70_mtp_fc.py`, `sm70_mtp_head.py`, `mtp_*experts.py` | Family `cuda/mtp.py`, `decode.py`, `multi.py`, shared weights/state interfaces |
| Graph preparation and warmup | `model_executor/warmup/sm70_*`; V2 GPU model runner; native preparation/acceleration reporting | Family `cuda/graphs.py`, `multi_graphs.py`, `service_memory.py`; engine lifetime and geometry policies |
| Calibration and reproducibility | Family `ops/qsa_kv_calibration.py`; `tools/qwen4_exp/qsa_kv_calibration.py`; local `calibrate_kv.py` / `prepare_fp8.py` | Offline calibration/export tools plus startup scale admission |
| Topology and collectives | `layers/sm70_collectives.py`, calibration helpers, `csrc/sm70_ring_collectives.cu` | `cuda/comm.py`, distributed family plumbing and rank protocol |

Suggested new provider location such as `src/tensorfold/cuda/sm70/` is **a design proposal**, not an existing TensorFold interface. Keep storage readers, layout conversion, and execution dispatch explicit. A useful operator contract records:

```text
operation + storage format/version + execution dtype + accumulation dtype
+ M/N/K and local-head geometry + strides/alignment + sharding plan
+ stream/workspace ownership + graph-capture readiness + fallback reason
```

Start with direct native entry points and thin TensorFold-owned wrappers. iCat Python wrappers frequently depend on vLLM configurations, global execution policy, parallel groups, and model-runner metadata. Those dependencies must be replaced or isolated, not copied into TensorFold imports accidentally.

Do not assume copying the `ops/*.cu` directory is sufficient. TurboMind sources require the bundled GEMM runtime, layouts, registry, conversion code, headers, and compile definitions. QSA's FP8 path imports software conversion helpers from another model directory. Derive the actual dependency closure from the pinned build and wrapper imports.

Two easy-to-miss root-level dependencies are `flash-attention-v100/` and `flash_qla/`. `setup.py` bundles the Flash-V100 package, including `flash_attn_v100_cuda` and `paged_kv_utils`, and precompiles `flash_qla_sm70_gdn_strided` from `flash_qla/ops/gated_delta_rule/chunk/sm70/csrc/gdn_forward.cu`. The GDN provider can also select the original SM70 TileLang implementation; that route appeared in our runtime logs. Ordinary attention/FlashInfer installations are not proof that these Volta-specific routes or ABIs exist. Keep only the libraries/operators your selected TensorFold path actually needs, but test their installed imports and native entry points explicitly.

Use TensorFold-owned operator names, for example a proposed `tensorfold_sm70` namespace, instead of assuming `torch.ops._C` or `torch.ops.vllm` registrations can be imported without vLLM. Preserve schema mutability/output contracts, current CUDA stream/device guards, workspace ownership, and any fake/meta implementations needed by compilation. Distinguish an operator being registered, a policy advertising it, and an actual measured route hit. The iCat acceleration endpoint is useful evidence but some fields describe configured capabilities rather than executed kernels.

## 7. FP16 execution is a substantive port

The reviewed TensorFold CUDA build gate defaults to `MIN_CAPABILITY=(8,9)` in `cuda/build.py`; its architecture-qualified extension names and lock identities are useful and should be retained. Add a provider-specific SM70 capability requirement rather than lowering every existing kernel's gate.

The Flash Next path has explicit BF16 assumptions throughout:

- `cuda/state.py`: residual, activation, projection, and output buffers are allocated as BF16.
- `cuda/attention.py`: probability tiles, dot operands, scratch outputs, and final writes use BF16.
- `cuda/glue.py`: intermediate BF16 rounding is part of the specified operation sequence.
- `cuda/bf16.py`: weight conversion, Triton operands, split-K results, and outputs target BF16.
- `cuda/qmm.py`: MLX affine packed data dequantizes to BF16 under a defined group/reduction order.
- `cuda/gdn.cu`, `gdn_io.*`, and `gdn_multi.py`: BF16 stored inputs/outputs surround FP32 state and reductions.
- Expert-parallel buffers currently use BF16 activations.
- KV descriptors identify raw `bf16` cache storage explicitly.

V100 does not supply native BF16 Tensor Core matmul. Some BF16 storage/conversion operations may be representable through software, but that does not make the current BF16 dot paths supported or efficient.

Create an explicit execution-precision policy, with at least FP16 model tensors and FP32 state/reductions for SM70. Audit every producer/consumer boundary: embedding, HC streams, Q/K/V, indexer, normalization, shared/routed experts, PLE injection, MTP fusion, and final logits. Keep exp/softmax/statistical reductions and recurrence stability requirements explicit.

Do not do a global BF16-to-FP16 text replacement. TensorFold deliberately preserves rounding and summation order in several kernels. FP16 also has a narrower exponent range than BF16; inspect activation ranges, overflow, normalization, and scaling under real prompts. Establish appropriate numerical references and tolerances for the new precision policy.

Raw FP16 cache bytes must not be labeled `bf16`. TensorFold's `KVFormat`/`KVPairFormat` identities and peer handshakes encode storage meaning. Register a distinct raw-half format if using FP16 storage, and ensure snapshots, prefix copies, cache readers, and graphs recognize it. Execution precision and compressed-cache format should be separate fields.

For native kernels, gate architecture-specific instructions individually. Volta-compatible FP16 WMMA/HMMA is different from newer `mma.sync` shapes, BF16/FP8 MMA, `cp.async`, TMA, thread-block clusters, and programmatic dependent launch. Keep modern-GPU paths and their current precision contracts intact.

Keep four independent precision axes in configuration and evidence: **weight storage**, **activation/matmul execution**, **attention KV storage**, and **recurrent-state arithmetic**. A model labeled FP8 or NVFP4 does not identify all four. For broader iCat-provider reuse, FP8/NVFP4/AWQ weight readers can feed software decode or weight-only FP16 kernels on SM70 where implemented; this does not create native FP8/FP4 Tensor Core math. In particular, replacing a checkpoint's calibrated FP4 activation arithmetic with W4A16 is a different execution policy unless the required activation quantization is deliberately reproduced. Preserve TensorFold's checkpoint/full-math distinction and qualify the selected policy rather than presenting a fallback as identical arithmetic.

## 8. Checkpoint and GGUF integration

TensorFold's inspected Flash Next loader supports MLX affine, ModelOpt/NVFP4, and EXL3 routes. It does not contain an implemented GGUF route merely because its model family matches iCat's.

For this implementation, make the **existing, verified Huihui mixed-GGUF checkpoint** the first full-model target. That gives the closest practical comparison with iCat and avoids another approximately 100 GB download or a lossy checkpoint conversion. Add GGUF as a format-specific route alongside the maintained MLX affine, ModelOpt/NVFP4 and EXL3 implementations.

An initial operator-only bring-up can use small fixtures or decoded tensor slices before the complete loader is ready. The full-model parity milestone in section 18 uses the exact GGUF and stock MTP identities in section 3. Additional checkpoint formats can use the SM70 provider after their own layout and numerical qualification.

Reuse the verified files read-only. Preserve per-tensor quant type, block sizes, shape interpretation, scale/codebook interpretation, row order, and sharding metadata. Q4_K, IQ4_NL, MLX affine4, NVFP4 and EXL3 are different encodings. A four-bit native kernel for one cannot consume another's bytes.

Map all nontrivial model semantics before enabling fast paths:

- Qwen4Exp GGUF names to TensorFold's family weight descriptors.
- HC down/injection packing and the four residual streams.
- Merged vs separate Q/K/V and GDN projections.
- Routed and shared experts, including their gates and reductions.
- The checkpoint's normalization conventions; do not blindly add or omit a `+1` transformation.
- Embedding and vocabulary-head ownership, including MTP sharing.
- PLE metadata and global n-gram indexing.
- Native interleaved rotary policy and the configured context limit.

The working iCat path keeps expert and embedding storage in their original compressed form, uses an expert arena, prepares canonical dense layouts, and enables device transcoding. These choices avoided expanding the full model into FP16. Port the useful representation changes as explicit load-time plans with memory accounting.

Whole-bank reads and bounded pinned staging worked. Avoid Python per-scalar extraction over large expert tensors, repeated quantize/dequantize cycles, or retaining several complete temporary layouts. Load-time CPU transcodes can become an enormous startup cost; GPU transcodes can consume peak VRAM if workspace lifetimes are not bounded.

Use the consolidated branch's existing `src/tensorfold/cuda/direct_read.py`, `read_budget.py`, `nvfp4/host_experts.py`, and family `cuda/reader.py` as the loader/ownership integration seams. Their bounded reads, reader backing, GPU-packing peaks and publication lifetimes are useful contracts; their checkpoint encodings are not interchangeable with GGUF. All required changes belong on the maintained baseline and must be validated against the selected packed representation.

## 9. Eight-GPU parallelism

The reviewed TensorFold `FlashNextEngine` currently:

- Admits TP only in `(1,2)`.
- Rejects concurrent `streams>1` when TP is greater than one.
- Restricts Flash Next NVFP4 and EXL3 checkpoint routes to one-GPU TP operation.
- Creates two-rank communication explicitly in the existing TP branch.

The loader already uses `world` in several sharding calculations, but this does **not** establish complete TP8 support. Admission, projection layouts, collective protocols, MTP, graph keys, vocabulary sampling, shared-state ownership, and memory planning all need review.

TensorFold also has **expert parallelism**: follower ranks own expert banks while rank zero retains the backbone and request execution. This is valuable, but it is not equivalent to iCat TP8. Merely setting expert ranks to eight will not shard all attention history or backbone state across eight GPUs. Rank-zero memory/compute can remain the limiting factor for long-context concurrency.

For a true TP8 port, validate at least:

1. Device/rank assignment and symmetric collective ordering on all ranks.
2. Shards for heads, expert intermediates, shared experts, embeddings, and output vocabulary.
3. HC replicated vs sharded representations and each all-reduce/all-gather boundary.
4. Identical top-k routing/weights where required; deterministic tie behavior.
5. Per-request state tables and row mapping across all six streams.
6. Draft and target sharing, output-vocabulary offsets, sampled-token broadcasts, and acceptance decisions.
7. Capture-safe NCCL operations and communicator lifetime through graph destruction.
8. Startup failure, peer failure, cancellation, and shutdown without orphaned communicators or freed live buffers.

**Critical geometry:** `640/8=80`. A local expert intermediate width of 80 is not divisible by a 32-value quantization block. iCat's optional Q8_0 MTP expert route therefore does not fit this TP8 geometry. TensorFold's group-32 affine column sharding needs the same careful scrutiny: splitting logical dimensions is not necessarily splitting packed blocks legally. Use an explicit aligned/padded/repacked shard plan, another decomposition, or a correct fallback; do not reinterpret a partial block.

The TP8 DeltaNet geometry is local NK=2, NV=6. Several attractive iCat native verify/replay routes were qualified for TP4 local NK=4, NV=12. Keep those gates until the TP8 route and its state-commit tests are qualified.

**Potential shortcut identified in the source review:** the mixed-QKV verification caller in `qwen_gdn_linear_attn.py` explicitly gates on local heads 4/12, while the native `sm70_gdn_verify_out` entry point accepts runtime `H`/`HV` parameters. This does not prove the underlying kernel is correct for 2/6, but it means the first investigation should be a small direct 2/6 kernel/state test plus an audit of the caller/replay buffers, rather than assuming a complete kernel rewrite is necessary. Other preprocessing, norm, and launch-tuning paths have their own independent TP4 shape gates.

A future topology experiment could compare TP8 against alternatives using each local four-GPU group. Two full TP4 replicas duplicate model weights and leave substantially less KV room; they are a possible decode-throughput experiment, not a promised six-full-context solution. PP2×TP4 was unsuccessful in our iCat trials, as recorded below.

## 10. Attention, indexer, and cache contracts

Qwen4Exp combines sparse QSA attention, a compressed-key indexer, DeltaNet recurrent state, PLE convolution/history, and MTP state. Treat these as distinct memory/precision contracts.

In iCat's QSA path, attention pages have the ABI:

```text
[num_blocks, 2, block_size, num_kv_heads, head_dim]
                 K/V axis
```

FP8 K/V pages are stored as `uint8`. TensorFold's family cache uses separately owned K and V arrays, logically `[capacity, kv_heads, payload_width]`, with independent scale arrays and immutable format descriptors. Neither address mapping nor ownership is interchangeable.

An attention port must handle page tables or provide a deliberate contiguous-cache adapter, correct row-to-request mappings, absolute positions, causal masking, padded graph rows, context lengths, KV heads, strides, and scale application. The indexer must retain the correct compression ratio and visibility bounds; top-k scratch and partition selection are part of the dependency closure.

Keep score, softmax, maximum/sum, and accumulation precision explicit. For recurrent state, verify updates and rollback through every accepted-prefix length, not just final text output. Restarting from a prefix checkpoint must reproduce the intended state under the new FP16 policy.

TensorFold already has valuable protections in `tensor_contracts.cuh`, kernel-contract documentation, lifecycle handling, and `KVPairFormat`: preserve active-byte overlap checks, immutable format identity, pointer-owner references, and failure ownership. Native bindings should validate dtype, shape, alignment, written prefixes, and disallowed aliasing before launching CUDA.

Cache growth or format changes must invalidate/rebuild graphs that captured old pointers. Do not free pointer tables, KV spans, or their backing buffers until all dependent CUDA work and graph replays are complete. TensorFold's VMM arena checks actual device support and allocation granularity; probe that on this V100/driver combination rather than assuming or bypassing it.

## 11. FP8 calibration and local patches

The selected QSA backend supports FP16/BF16 and E4M3 main KV cache. Its supported list does not make the global runtime's INT8 or TurboQuant options available to this specific model/backend.

The converter contract is:

```text
stored       = E4M3FN(x / scale)
reconstructed = decode_E4M3FN(stored) * scale
maximum finite E4M3FN magnitude = 448
```

Use the exact E4M3FN encoding/scale convention of the reader and writer, including zero/subnormal, saturation, and nonfinite handling; a similarly named FNUZ or other FP8 variant is not a compatible byte format.

The main QSA path can otherwise fall back to unit scales in some explicit-FP8 configurations. The MTP draft is stricter because uncalibrated proposals were found invalid in upstream qualification. We enabled strict target/draft scale validation and supplied actual scales rather than depending on defaults.

### What was collected

- Built-in QSA trace hooks while serving this checkpoint with FP16 KV and eager execution.
- Eight small mixed-domain prompts, 64 output tokens each, plus a 19,522-input-token synthetic audit prompt with 16 output tokens.
- The temporary `COLLECTING` marker controls tracing; it is absent in normal serving.
- Finite activation distributions across all eight TP ranks.
- Thirteen attention owners: main layer IDs `3,7,11,...,47`, plus MTP logical layer 48.
- Twenty-four main K/V scales and two MTP K/V scales.

Deployed scale policy:

```text
scale = 2 ** ceil(log2(1.25 * observed_max_abs / 448))
```

This is a power-of-two scale with at least 25% observed-range headroom. It is our local deployment policy, not a universal model-card requirement. The corpus is small and synthetic/mixed; it does not prove zero clipping or unchanged quality for all future prompts. Recalibrate/qualify when checkpoint, projection arithmetic, execution precision, or cache conversion changes.

Artifacts:

- [calibrate_kv.py](qwen-flash-next/calibrate_kv.py): collection requests and marker management.
- [kv-calibration-requests.json](qwen-flash-next/research/kv-calibration-requests.json): exact prompts/payloads/responses.
- [kv-calibration-summary.json](qwen-flash-next/research/kv-calibration-summary.json): aggregated observed ranges.
- [prepare_fp8.py](qwen-flash-next/prepare_fp8.py): deployed scale generation, hard-linked draft overlay, provenance.
- [fp8-scale-provenance.json](qwen-flash-next/research/fp8-scale-provenance.json): values, counts, policy, and hashes.

Raw rank traces remain under `qwen-flash-next/cache/qsa-calibration/`, named `qsa-kv-rank*-pid*.jsonl`. The pinned `tools/qwen4_exp/qsa_kv_calibration.py summarize` command was run with `--expected-layers 13`; the extra attention owner is the MTP draft. Capture is conditional on the marker and should not be left enabled for performance runs.

Main scale names are `model.layers.{layer}.self_attn.k_scale` / `v_scale`. Serialized draft names are `mtp.layers.0.self_attn.k_scale` / `v_scale`; the draft's logical runtime layer 48 must not be confused with its checkpoint layer index zero.

### Local Python changes beyond the image

1. **`gguf_loader.py`:** accepts `model_loader_extra_config.kv_scales_safetensors`, yields scale tensors after GGUF weights and before model scale finalization, and rejects non-scale names, non-scalar, non-floating, nonfinite, or nonpositive values. This allows scales to accompany the unchanged GGUF checkpoint.
2. **`kv_cache_utils.py`:** an opt-in local startup diagnostic under `LOCAL_KV_LAYOUT_REPORT` exports allocator geometry and searches the six-request context bound. This is observability code, not a new cache implementation.

Both files are read-only bind mounts in Compose. The patch is [fp8-local-runtime-overlay.patch](qwen-flash-next/research/fp8-local-runtime-overlay.patch). Per-file hashes and the image ID are in the final result artifact.

Strict startup logged **24/24 main** and **2/2 draft** scales loaded. Do not copy an iCat scale file into a TensorFold reader without checking its cache format, scale granularity, layer mapping, and execution contract. TensorFold's rotated/groupwise integer codecs are not this per-layer E4M3 format.

## 12. Exact six-context calculation

At FP16 KV and a 262,144 cap, a prior profile reported roughly **777,473 effective tokens / 2.97 contexts**. A three-slot FP16 configuration at 258,048 reported **776,868 / 3.01 contexts**. These effective-token values derive from a hybrid allocator, not a simple uniform bytes-per-token pool.

The initial FP8 six-slot launch at **258,048** reported **1,429,889 effective tokens / 5.54 contexts**, not six. The difference from a naive 2× prediction was consequential.

Final profile:

| Allocator item | Value |
| --- | ---: |
| Shared physical block count | 1,885 |
| Shared null block | 1 |
| History group's token block size | 800 |
| History demand at the final cap | 297 blocks per request |
| Additional fixed group demands | `1 + 4 + 4 + 4 + 4 = 17` per request |
| Total per full request | `297 + 17 = 314` blocks |
| Six requests plus null | `6 × 314 + 1 = 1,885` |
| Reported maximum concurrency | `1885/314 = 6.003184713...` before null reservation |

Reserve the two speculative lookahead positions as well:

```text
ceil((context_limit + 2) / 800) <= 297
context_limit <= 297 * 800 - 2
context_limit <= 237598
```

Thus the largest context admitted by this **specific profiled layout and reservation model** is **237,598 tokens per request**, totaling **1,425,588 logical context tokens across six requests**. The limit covers input plus generated output, including reasoning and template tokens; it is not an allowance for 237,598 input tokens plus an arbitrary answer.

The profile is preserved in [kv-layout-fp8-c6.json](qwen-flash-next/research/kv-layout-fp8-c6.json). This is not a portable TensorFold constant. Different workspace planning, graph sets, model precision, cache codecs, tensor placement, memory utilization, or allocator granularity will change it. Recompute from TensorFold's own allocation plan, including fixed state, both target/draft caches, lookahead, prefix retention/eviction, and graph pools.

**Still unverified:** six independently filled contexts at this exact cap, simultaneous final-token decode, zero recomputation/preemption, and FP8 long-context answer quality. The completed benchmark used 8K prompts. A proper future capacity test must prevent shared-prefix reuse, keep all six contexts resident together, validate their separate retrieval answers, and require no preemptions/OOMs.

## 13. Lower-bit KV: an existing TensorFold advantage

The reviewed fork already registers raw BF16, ExLlama-style `int8` and `int4`, and several RotorQuant 3/4/6/7/8-bit variants. It also supports ordered, independently selected key/value formats. This is a substantial head start over the particular iCat QSA backend we ran, whose lowest supported main KV format was FP8 E4M3.

These are **existing source capabilities on the fork's supported paths**, not V100 qualification. The current integer dequantizers and attention code round into BF16; they still need SM70-safe execution and end-to-end testing.

Useful payload-plus-scale arithmetic, ignoring allocator padding and fixed metadata:

| Format | Scale convention | Effective bits/value |
| --- | --- | ---: |
| Raw 16-bit | No per-group scale | 16 |
| Local iCat E4M3 | Per-layer K/V scales | Approximately 8, plus small layer metadata |
| TensorFold native INT8 | 16-bit scale / 32 values | 8.5 |
| TensorFold native INT4 | 16-bit scale / 32 values | 4.5 |
| RotorQuant 4-bit | 32-bit scale / 128 values | 4.25 |
| RotorQuant 6-bit | 32-bit scale / 128 values | 6.25 |
| RotorQuant 3-bit | 32-bit scale / 128 values | 3.25 |

TensorFold native INT4/INT8 use H32 rotation and midpoint reconstruction; RotorQuant variants pin their own basis, norm, codebook, and codec identities. Equal bit counts do not mean compatible bytes or equal numerical behavior. Keep those descriptor and snapshot identities intact.

Suggested qualification order is raw FP16 → one well-defined 8-bit format → INT4 or a selected RotorQuant variant → asymmetric K/V experiments. An 8-bit-key / 4-bit-value pair is a useful candidate to investigate, not an established winner. Measure both attention error and whole-model behavior; preserve the intended QSA indexer and recurrent precision.

Do not predict total concurrent-context capacity by scaling only K/V payload bits. Indexer history and recurrent state can become dominant as main KV shrinks. Also, a lower-bit reader can increase decode cost through rotations, unpacking, register pressure, or repeated dequantization. INT8 storage does not automatically invoke DP4A, and DP4A availability alone does not prove it faster than software E4M3 attention.

Reviewed format authority: [kv_formats.py at 94d84f5](https://github.com/kklouzal/TensorFold/blob/94d84f510f87ee27950d821626a3510bff9c5625/src/tensorfold/families/qwen4_exp/kv_formats.py).

## 14. PLE / n-gram memory

The target's packed IQ4_NL PLE table has 320,001,536 rows, width 160, and 90 packed bytes per row: about **28.8 GB / 26.82 GiB**. Keeping it packed in host RAM is very different from expanding it into a large FP16 table or leaving random reads on SATA SSD.

The current iCat environment includes:

```text
VLLM_QWEN4EXP_PLE_HOST_GIB=16
VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB=64
VLLM_PLE_DISK_OFFLOAD=0
VLLM_PLE_CPU_OFFLOAD=0
VLLM_SM70_QWEN38_HYBRID_PLE=0
```

The named host-share/admission setting is not itself the measured physical table size. Interpret it through the provider implementation and actual shared/pinned allocations. The GGUF pinned lookup route retains compressed table storage in host RAM; generic offload flags alone do not describe that route.

TensorFold has substantial host/SSD table infrastructure already. Adapt its lifetime and IO ownership to the exact GGUF table encoding rather than replacing it with another ad hoc offload subsystem. Do not assume the MLX affine n-gram table and IQ4_NL table share the same row layout or dequantization scales.

Validate exact n-gram IDs: seed, prime/head sizes, offsets, EOS-filled initial history, overflow arithmetic, token history length, layer indexing, speculative append/rollback, and chunk boundaries. A fast wrong hash can produce fluent but degraded outputs, so text smoke tests alone will not find every error.

For performance, investigate NUMA placement and shared ownership of the table, bounded pinned staging, batched row gather/decode, and overlapped transfer with actual streams/events. Keep host table inserts outside captured GPU regions where required. Benchmark lookup/cache hit behavior before adding another GPU hot cache that consumes the memory needed for KV.

The current setup **does not offload KV history to system RAM**. Host-KV code exists in iCat, but it is not part of the measured final configuration and should be a separate later feature.

## 15. MTP, graphs, and sampling

### Native MTP

Native MTP depth 2 is the practical reference from this host. Depth 4 did not provide a useful gain in the initial single-stream comparison. This is not proof that depth 2 is optimal at every concurrency, prompt distribution, or context length.

Keep target verification authoritative and retain all speculative state: HC streams, DeltaNet recurrent/conv state, PLE history/conv state, target attention/indexer cache, MTP cache, accepted-prefix lengths, and pending tokens. Test every rejection/acceptance prefix and EOS boundary.

The four-stream HC residual representation means the draft interface is not simply a conventional final hidden vector. Use the model's actual MTP fusion/normalization and embedding conventions. Do not copy a draft head from a similarly named model without checking the family configuration and tensor names/shapes.

The iCat run uses greedy draft proposals with the runtime's own verifier. TensorFold's exact sampling/rejection infrastructure is a separate contract. Preserve normalized proposal/target probabilities, rejection residuals, keyed uniforms, temperature/top-k/top-p/penalty behavior, and per-request seed ownership.

For greedy mode, serial/drafted target token identity under the same execution/cache policy is an appropriate gate. For sampled mode, do **not** require sampled serial and drafted token sequences to be identical merely because the seed matches: a correct rejection sampler can consume randomness differently while preserving the target distribution. Require repeatability for fixed configuration and validate distributions separately. The fork already documents this distinction in `docs/rejection-sampling-validation.md`.

### CUDA graphs

With depth 2, each verification batch can contain `concurrency × (1+2)` rows. For concurrency 1–6, the iCat graph sizes are therefore **3, 6, 9, 12, 15, 18**. Actual acceptance and padding still require valid per-row metadata and masked unused rows.

The successful setting is **Torch compilation mode 0 with FULL decode CUDA graphs**. Disabling TorchDynamo compilation is not the same as disabling CUDA graph replay. Do not lose the graph speedup while working around a compiler tracing issue.

TensorFold already owns graph infrastructure with persistent buffers, context buckets, DeltaNet parity keys, and strong owner references. Extend it for the new precision/provider, TP8, and concurrent state; reuse those ownership mechanisms. Its single-decoder `Graphs.warm()` explicitly dirties sequence state: reset/prefill after capture. Check both recurrent parities and every reachable context/row bucket.

Avoid lazy layout conversions or memory allocations during capture/replay. Prepare packed weights, codec workspaces, LUTs, and kernels before capture. In TensorFold's multi-graph path, host operations are deliberate inserts between captured pieces; do not silently drag Python/Python-list work into a supposedly capturable expert provider.

### DSpark / DFlash

DSpark was researched but **not integrated or benchmarked** in the working iCat setup. A Flash Next DSpark checkpoint and SGLang-oriented integration material were found; the local research artifacts are listed below. Adding the SGLang package by itself would not provide the required auxiliary hidden-state capture, draft interfaces, exact verification, quantization support, and state rollback in another engine.

Treat a TensorFold DSpark path as a later adapter project after native MTP parity and performance. Reuse any compatible existing family/drafter machinery, but verify trained target/checkpoint compatibility, HC intermediate semantics, precision, and all cache/state transitions. There is no measured V100 DSpark uplift from this session to use as a planning assumption.

## 16. Scheduling and benchmark lessons

The first six-stream attempt used 512 output tokens. Earlier requests finished while later prompts were still prefilling; there was **no common six-stream decode interval**. The benchmark correctly refused to report a six-stream decode number from that run.

Longer output alone was not the whole solution. iCat's adaptive mixed-prefill latency controller substantially throttled prompt progress while earlier requests decoded. For the final experiment, it was disabled using `mixed_prefill_step_latency_ms=0`, with a long-prefill threshold of 1,024 and its adaptive fair-share option enabled. The final workload used 1,024 output tokens and produced about 20 seconds of true shared decode per measured round.

Transfer the lesson to TensorFold's scheduler, not these literal flag names:

- Separate admission count, resident decode count, partial-prefill progress, and actual simultaneous generation.
- Reserve decode capacity without starving pending/partial prompts indefinitely.
- Bound per-step prefill work and compare aggregate throughput against streaming latency.
- Record per-request prefill progress and first-token time; six "running" requests can include five still prefilling.
- Exercise arrivals, cancellations, cache hits, partial prefill, and context growth, not just simultaneous identical short requests.

A reproducible decode benchmark should record token IDs/counts and token-arrival events with small SSE read buffering. The local probe uses `iter_lines(chunk_size=1)` to avoid client buffering dominating short measurements. For simultaneous streams, define the shared interval as latest first-token arrival through earliest final-token arrival, then count each stream's generated tokens in that interval.

Retain warmup separately. Use early unique nonces for independent-cache tests, fixed request sampling, recorded prompt hashes, and enough generated tokens. Do not reuse a common prefix accidentally when measuring independent full-context capacity. Report aggregate decode, individual rates, TTFT, and end-to-end throughput separately.

The final 233.55 decode vs 64.05 end-to-end aggregate result makes prefill an important optimization target, even though the immediate request prioritized decode speed.

## 17. Failed paths and limitations

| Path | What happened | Integration lesson |
| --- | --- | --- |
| TP4 × PP2 | Successive failures involved missing pipeline-layer expert ownership, an HC output-shape assertion, then a CUDA device-side assertion during inference. | Keep TP8/PP1 as the established reference. Pipeline stages need complete ownership and broadcast semantics, not just a rank-count change. |
| Torch compilation mode 3 | A Dynamo/FLA tracing path failed around device-index handling. | Mode 0 plus CUDA graphs works. Treat full compilation as optional future work. |
| TP8 native GDN verify/spec replay | Fast-path geometry expects the qualified local-head contract rather than TP8's 2/6. | Keep the generic path until a tested TP8 specialization exists. |
| Q8_0 MTP experts under TP8 | The 80-wide local intermediate cannot satisfy 32-value block alignment. | Validate layout/sharding constraints before enabling a memory-saving route. |
| Explicit FP8 without scale overlays | Main target could fall back to unit scales; draft requires calibrated scales. | Use strict scale admission; format support alone is insufficient. |
| FP8 assumed to provide six 258,048 contexts | Actual profile was 5.54×. | Profile complete hybrid memory, then calculate rounded block demands. |
| Six short streams submitted together | Did not guarantee simultaneous six-way decode. | Check actual overlap and scheduler behavior before publishing throughput. |
| Full six-context capacity | Not run after user shortened the benchmark scope. | Keep it explicitly pending as future qualification, not a claimed successful test. |
| 4-bit KV through this iCat QSA backend | Not supported by the selected backend despite other global runtime codecs existing. | Backend-specific capability tables must govern admission. |
| 4,096-token reasoning test | Correct arithmetic but truncated final answer; 4,108 tokens were needed for natural EOS. | Separate output-budget failure from numerical correctness and runtime failure. |

Pipeline failure logs are `logs/startup-pipeline4-attempt1.log`, `attempt2.log`, and `attempt3.log`. Once a CUDA device assertion fires, retire that process/context and its peers before another attempt. Disabling core dumps prevented repeated worker crashes from filling the system disk; hiding assertions would not fix the underlying ownership error.

No completed vision qualification, comprehensive model-quality evaluation, TensorFold SM70 run, full FP8 concurrency sweep, or full-context FP8 throughput measurement should be inferred from this handoff.

## 18. Integration sequence and acceptance gates

### Milestone A: reproducible SM70 foundation

Create a dedicated V100 image/dependency lock targeting the **latest SM70-supported CUDA release and latest applicable 12.x patch**, currently **CUDA 12.9.2**. Resolve the newest compatible framework/compiler/library stack that retains SM70 execution; use the recorded 12.8.1 / cu128 environment only to reproduce the historical reference. Preserve TensorFold's foundation validation rather than weakening its trust/toolkit checks globally. Audit TensorFold's required PyTorch/Triton APIs against the new stack and verify its actual compiled GPU targets. Build a small extension with an SM70 cubin, run it on a V100, and exercise the required framework/library operations before loading the full model. Record the source commit, wheel/image hash, compiler, PyTorch/Triton versions and compiled operator inventory. Reuse the consolidated immutable-input tools with the V100-specific foundation contexts described in section 20.

Keep architecture-qualified build/lock identities. Include the source/header dependency hashes and relevant build flags in cache keys; a modern-GPU extension must not be loaded by a V100 rank merely because a module name matches.

### Milestone B: provider and numerical primitives

Add explicit FP16 execution and FP32 accumulation/state policies. Port the minimal projection, normalization, dequantization, rotary, attention, and recurrence primitives required by Flash Next. Test representative actual dimensions, including HC width 10,240, intermediate width 640 and local TP8 width 80, local GDN 2/6 heads, multiple rows, and boundary/padded rows.

Validate layout conversion independently from arithmetic. Use decoded scalar or trustworthy high-precision references for packed tensors; use the intended FP16-policy serial reference for fused vs unfused equality. Specify tolerances where changing precision legitimately changes results.

### Milestone C: exact checkpoint and TP8 serial reference

Bring up one stream across all eight GPUs with no MTP, native RoPE, FP16 attention cache, FP32 recurrent state, host packed PLE, and a modest context. The full model need not fit one GPU to test individual components there. Qualify complete answers, token accounting, EOS, reasoning parsing, tools, and rank cleanup. Then increase context in steps and verify prefix/chunk equivalence.

Do not demand equality with a different quantized checkpoint as the same baseline. Record both checkpoint identity and execution/cache precision for every comparison.

### Milestone D: graphs and concurrent TP

Capture the serial reference, then add true concurrent streams under TP8. Cover all row shapes, recurrent parities, context buckets, cache growth, and pointer identities. Compare eager vs replay outputs/state under the same policy, including warm/cold prefix reuse. Require coherent per-rank collective sequences and clean cancellation/retirement.

### Milestone E: native MTP

Start with depth 2 and the stock matching draft. Verify accepted/rejected-prefix state and greedy serial identity, then sampled rejection/distribution behavior. Test changing concurrency and draft depth; do not assume every native fused path supports the resulting rows or local heads.

### Milestone F: cache quantization and capacity

Add calibrated E4M3 or adapt an existing TensorFold 8-bit format. Use explicit format identities and correct conversion scales. Establish numerical/answer-quality gates before moving to INT4/RotorQuant or asymmetric pairs. Recompute memory capacity after each policy change.

For the future six-full-context test, require six independent prompts, input+output budgets at the served cap, all six fully resident/overlapping before any retires, separate retrieval evidence, no cross-request identifier leakage, no preemptions or recomputation, no OOM, and completion/cleanup of every request. Reserve output space explicitly; a prompt filling the entire context cannot also generate an answer.

### Validation matrix to retain

| Dimension | Required distinctions |
| --- | --- |
| Execution | Serial/eager, captured replay, fused/unfused; actual installed build |
| Requests | One and multiple streams, arrivals, cancellation, partial replies, EOS, forced-length throughput |
| Context/cache | Short, 8K, intermediate, near-native; warm/cold prefixes; separate histories; growth/eviction |
| Precision | FP16 model / FP32 state; each KV codec/pair independently; calibration/checkpoint identity |
| Speculation | Disabled/enabled; all acceptance lengths; draft state replay/rollback; greedy vs sampled semantics |
| Distribution | TP ranks, collectives, vocabulary tails, routed/shared expert reduction, rank failures |
| Product behavior | Complete answer, reasoning separation, tool call/result loop, token counts, context admission |
| Performance | Warmup excluded, TTFT, shared decode, per-stream rates, end-to-end throughput, memory peaks, preemptions |

The fork already contains relevant test seams: `tests/cuda/test_flashnext_tp.py`, `test_flashnext_multi.py`, `test_flashnext_multi_graphs.py`, `test_flashnext_kvquant.py`, `test_flashnext_rotorquant.py`, `test_flashnext_prompt_cache.py`, and family state/ownership host tests. Extend those contracts with SM70-specific cases; source-only or mocked CPU tests do not establish native V100 correctness.

## 19. Optimization opportunities

The order below reflects the evidence from this host and likely integration value. Gains and effort are hypotheses until profiled and qualified.

| Priority | Candidate | Why it could help / required evidence |
| --- | --- | --- |
| First | Stable decode graphs with prepared weights/workspaces | Largest demonstrated early gain. Prove replay uses identical precision and persistent state owners. |
| First | Efficient packed-weight GEMV/GEMM and routed/shared experts | Decode repeatedly touches large compressed banks; select kernels by real M/N/K and quant layout, not model name. |
| First | Qualify/generalize native GDN verify/replay for local 2/6 heads | Caller gates are narrower than the native entry point's parameter list. Start with direct kernel/state tests; specialize only where necessary. Preserve recurrent/conv rollback and every accepted-prefix length. |
| High | Better prefill path and fair mixed scheduling | Final six-stream TTFT was ~74 s. End-to-end throughput is far below steady decode. Profile QSA scoring/top-k, GDN prefill, host PLE gathers, and dispatch overhead separately. |
| High | Lower-bit KV using TensorFold's existing codecs | Potential long-context concurrency advantage. Test software FP16 readers, attention error, quality, and actual total memory rather than payload bits alone. |
| High | NUMA-aware PLE staging and rank placement | Host table accesses plus a nonuniform eight-GPU fabric can limit overlap. Measure socket-local placement, pinning, shared table copies vs replication, and transfer costs. |
| High | Batch/concurrency-aware MTP depth and draft preparation | Depth 2 beat 4 as the initial practical choice, but six-stream optimum is unknown. Compare accepted tokens per unit of total target+draft time. |
| Medium | QSA FP8 conversion/scorer fusion and head-aware sparse readers | Reduce unpacking, temporary traffic, and redundant work. Preserve per-layer scales and FP32 score/softmax accumulation. |
| Medium | Reuse buffers across serial target/draft phases | iCat shares eligible QSA top-k and RoPE storage. TensorFold already has owner-aware graph pools; quantify live/peak memory before aliasing more buffers. |
| Medium | Persistent GEMM tuning and warmed kernel inventory | Reduce startup/JIT noise and choose appropriate skinny/batched kernels. Key caches by architecture, source, format, dtype, and shape. |
| Medium | Communication plan tailored to the NVLink topology | Benchmark NCCL versus any compatible specialized collective using real message sizes. Custom ring collectives are not automatically better for all operations. |
| Medium | Raise admitted VRAM budget carefully | The reference uses 90% utilization and retains a few GiB free per GPU. Additional KV capacity may be available, but dynamic prefill/graph peaks must be measured; six native-262K contexts are not an established result. |
| Later | TP/EP hybrids or local-group replicas | Potential throughput/layout advantages; account for replicated backbone/cache or duplicated full model memory. EP alone does not solve rank-zero full-context KV pressure. |
| Later | DSpark adapter / alternate draft architectures | Possible further acceleration, but no session evidence of benefit. Requires target auxiliary-state capture and exact verifier/state integration. |
| Later | Full Torch compilation or pipeline parallelism | Both had concrete problems in this setup; revisit after the simpler provider/TP8 graph baseline is sound. |

Profile GPU timelines and CPU launch/lookup work before choosing fusions. GPU utilization percentage is not proof of useful Tensor Core utilization, optimal memory bandwidth, or absence of synchronization stalls. Keep power caps, clocks, thermals, background jobs, checkpoint, prompt lengths, sampling, and warmup conditions comparable.

A good benchmark scoreboard should maximize accepted target tokens/s while also showing TTFT, complete-request latency, quality gates, VRAM/RAM peaks, startup cost, and disk growth. A faster draft or a smaller cache can lose overall if it reduces acceptance or makes prefill disproportionately expensive.

## 20. Build, disk, and operational practices

### Consolidated build-preparation tooling

The pinned branch already contains `tools/gb10_capture.py`, `gb10_controller.py`, `gb10_io.py`, and `gb10_progress.py`. Reuse their source/context identity, bounded IO, durable publication, deadline and retirement contracts where applicable to the V100 build workflow.

The capture tool freezes a clean commit plus the `foundation_pins` and `foundation_artifacts` named contexts. It records source/archive/context identities and declares a native `amd64` or `arm64` target. This host uses `linux/amd64`. Stage separate V100 foundation pins/artifacts for the **newest SM70-supported CUDA 12.x update, currently 12.9.2**, with its compatible framework/library stack. Keep those inputs distinct from the CUDA 13.4 GB10 bundle and the historical 12.8.1 benchmark bundle. Bind the selected provider architecture, precision, toolchain and source/header identities into the build/cache evidence. CPU architecture alone does not distinguish SM70 from newer NVIDIA GPUs.

Keep its new-output and failure semantics: inputs remain immutable through capture/build/audit; a partial attempt is preserved for diagnosis and is not silently retried or overwritten. Controller IO success, command completion, source CPU controls, installed-image imports and native/model qualification remain separate evidence. The README's 58 passing source/fault controls do not qualify a new V100 build. Preserve the shared helper and its project-owned cleanup import closure rather than copying only the top-level capture/controller script.

The reviewed `docs/gb10-controller-io.md` and `docs/gb10-progress-io.md` describe the retained contracts. Adapt any controller profile/lifecycle action explicitly to the V100 deployment; these preparation tools themselves do not authorize or perform this handoff's future model rollout.

### Compilation and resource policy

The host has enough CPU parallelism for a large build, but compilation concurrency and runtime thread counts are different decisions.

- The source build used a 64-worker Ninja build with `MAX_JOBS=128`, `CMAKE_BUILD_PARALLEL_LEVEL=64`, and `NVCC_THREADS=2` in its later build environment. Record the actual generated Ninja command rather than assuming an environment variable took effect.
- Runtime image defaults were subsequently reduced for worker startup to avoid eight workers each launching excessive build jobs. Do not turn on 64 compiler threads per rank during serving.
- Use `uv` and a virtual-environment Python for the iCat tree; its local `AGENTS.md` explicitly prohibits system Python and bare pip.
- Preserve native-object/wheel caches when useful. Local `native-cache/` is roughly 1.9 GiB and `wheels/` roughly 2.8 GiB; they are not proof that another PyTorch/CUDA ABI can reuse the binaries.
- Persist `TRITON_CACHE_DIR`, `TILELANG_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, and the equivalent provider GEMM LUT paths. Missing TileLang persistence caused repeated startup compilation before it was fixed.
- Build SM70-only artifacts initially; avoid compiling every architecture or bundling CUDA 13-only code into the first V100 target.
- Include new `.cu`, `.cpp`, and `.cuh` files in TensorFold wheel package data. The reviewed family package-data list does not automatically include an arbitrary new SM70 subpackage/header tree. Qualify an installed wheel/container, not only a source checkout.

Disk incidents and protections:

- Failed pipeline workers produced sparse core files that consumed about 11 GB physically. Compose now sets soft/hard core limits to zero, and the launcher also applies `RLIMIT_CORE=(0,0)`.
- Docker JSON logs rotate at 20 MB, three files per container.
- Only experiment-specific build caches/core files were removed; Portainer and unrelated services were preserved. Avoid global Docker pruning as a routine integration step.
- Model and extracted draft downloads are already verified; hard-link or mount where appropriate rather than duplicating them for TensorFold.
- Loading the active iCat service already occupies nearly all useful GPU memory. A second full-model TensorFold server cannot be assumed to fit alongside it. Plan later A/B runs as explicit sequential deployments, with the old image/config retained for rollback.
- Long startup needs a realistic healthcheck start period. In our setup the container uses a 30-minute startup allowance; a process being alive is not evidence the API is ready.
- After a failed native run, release its workers/communicators and confirm memory is returned before interpreting the next run's capacity profile.

Useful first read-only commands for a future implementation session:

```bash
cd /home/kklouzal/qwen-flash-next
cat runtime.json
cat compose.yaml
git -C 1cat-vllm rev-parse HEAD
git -C 1cat-vllm diff --stat
df -h /
nvidia-smi topo -m
sudo docker compose ps
curl --fail --silent http://127.0.0.1:8000/health
```

Capture the baseline before making changes. Do not assume a fresh checkout of the pinned iCat commit contains the local calibration loader overlay or the current launcher settings.

## 21. Local evidence and primary sources

### Local reproduction and measurements

All relative paths below resolve from this handoff's location in `/home/kklouzal`.

| Artifact | Purpose |
| --- | --- |
| [runtime.json](qwen-flash-next/runtime.json), [serve.py](qwen-flash-next/serve.py), [compose.yaml](qwen-flash-next/compose.yaml) | Active launch configuration and source mounts |
| [Dockerfile](qwen-flash-next/Dockerfile) | Native build and environment; distinguish image settings from live Compose overrides |
| [fp8-c6-results.json](qwen-flash-next/research/fp8-c6-results.json), [CSV](qwen-flash-next/research/fp8-c6-results.csv) | Final six-stream metrics, image/source hashes, memory, and qualification boundaries |
| [fp8-c6-bench-6.json](qwen-flash-next/logs/fp8-c6-bench-6.json) | Per-request token IDs, payload conditions, timing events, and shared decode windows |
| [GPU trace](qwen-flash-next/logs/fp8-c6-bench-6.gpu.csv), [metrics trace](qwen-flash-next/logs/fp8-c6-bench-6.metrics.json) | GPU observations, cache use, concurrency, and preemptions |
| [suite-fp8-six-final.log](qwen-flash-next/logs/suite-fp8-six-final.log) | Benchmark rounds; includes the original 4,096-output-budget reasoning test failure |
| [card-fp8-c6-8192.json](qwen-flash-next/logs/card-fp8-c6-8192.json) | Successful reasoning follow-up with natural EOS at 4,108 tokens |
| [smoke-fp8-c6.json](qwen-flash-next/logs/smoke-fp8-c6.json) | Arithmetic, reasoning separation, EOS, and tool call/result checks |
| [benchmark-summary.json](qwen-flash-next/research/benchmark-summary.json) | Earlier graphs/MTP progression; configuration conditions differ from final six-stream run |
| [concurrency2-mtp2.json](qwen-flash-next/logs/concurrency2-mtp2.json) | Completed earlier FP16 two-stream reference |
| [long-final.json](qwen-flash-next/logs/long-final.json) | Earlier single-request FP16 near-native-context retrieval check |
| [kv-layout-fp8-c6.json](qwen-flash-next/research/kv-layout-fp8-c6.json) | Exact profiled hybrid allocator geometry and six-context bound |
| [fp8-local-runtime-overlay.patch](qwen-flash-next/research/fp8-local-runtime-overlay.patch) | Two local Python changes beyond the image |
| [fp8-scale-provenance.json](qwen-flash-next/research/fp8-scale-provenance.json) | Deployed main/draft scales and hashes |
| [model-integrity.json](qwen-flash-next/research/model-integrity.json), [mtp-integrity.json](qwen-flash-next/research/mtp-integrity.json) | Model and extracted-draft provenance |
| [gguf-metadata.json](qwen-flash-next/research/gguf-metadata.json) | Entrypoint shard metadata; first shard has zero tensors intentionally |
| [bench_concurrent.py](qwen-flash-next/bench_concurrent.py), [probe.py](qwen-flash-next/probe.py) | Request generation, streaming measurement, correctness checks; inspect arguments rather than assuming defaults |
| [dspark-model-card.txt](qwen-flash-next/research/dspark-model-card.txt), [SGLang capture research](qwen-flash-next/research/qwen38_dspark_sglang_qwen4_exp_aux_capture.py.txt) | Research inputs, not a working DSpark integration or benchmark |
| [tensorfold-integration-final-review.json](qwen-flash-next/research/tensorfold-integration-final-review.json) | Single authoritative branch/commit, current source identities, and carried-forward blob verification |

The current source tree and artifacts are more useful than conversational recollection if a detail differs. Historical files such as `research/final-report.json` describe earlier one-stream configurations; they must not silently override the newer FP8 six-slot report.

### Immutable upstream/source links

- [1Cat-vLLM source at 6a143197](https://github.com/1CatAI/1Cat-vLLM/tree/6a143197e6e5f6fa77bd483df0c5a09dcea6aa20).
- [1Cat native build manifest](https://github.com/1CatAI/1Cat-vLLM/blob/6a143197e6e5f6fa77bd483df0c5a09dcea6aa20/CMakeLists.txt).
- [Authoritative TensorFold source snapshot](https://github.com/kklouzal/TensorFold/tree/94d84f510f87ee27950d821626a3510bff9c5625).
- [Consolidated source/context capture tool](https://github.com/kklouzal/TensorFold/blob/94d84f510f87ee27950d821626a3510bff9c5625/tools/gb10_capture.py).
- [Selected foundation and build-preparation guide](https://github.com/kklouzal/TensorFold/blob/94d84f510f87ee27950d821626a3510bff9c5625/deploy/gb10/README.md).
- [TensorFold build gate and extension identity](https://github.com/kklouzal/TensorFold/blob/94d84f510f87ee27950d821626a3510bff9c5625/src/tensorfold/cuda/build.py).
- [TensorFold Flash Next engine admission](https://github.com/kklouzal/TensorFold/blob/94d84f510f87ee27950d821626a3510bff9c5625/src/tensorfold/families/qwen4_exp/cuda/engine.py).
- [TensorFold Flash Next weight loader](https://github.com/kklouzal/TensorFold/blob/94d84f510f87ee27950d821626a3510bff9c5625/src/tensorfold/families/qwen4_exp/cuda/weights.py).
- [TensorFold KV format/identity authority](https://github.com/kklouzal/TensorFold/blob/94d84f510f87ee27950d821626a3510bff9c5625/src/tensorfold/families/qwen4_exp/kv_formats.py).
- [TensorFold CUDA kernel contracts](https://github.com/kklouzal/TensorFold/blob/94d84f510f87ee27950d821626a3510bff9c5625/docs/cuda-kernel-contracts.md).
- [TensorFold Flash Next ownership](https://github.com/kklouzal/TensorFold/blob/94d84f510f87ee27950d821626a3510bff9c5625/docs/cuda-flashnext-lifecycle.md).
- [TensorFold rejection-sampling evidence contract](https://github.com/kklouzal/TensorFold/blob/94d84f510f87ee27950d821626a3510bff9c5625/docs/rejection-sampling-validation.md).
- [TensorFold Flash Next recipe](https://github.com/kklouzal/TensorFold/blob/94d84f510f87ee27950d821626a3510bff9c5625/docs/recipes/qwen3.8-flash-next.md).

### Attribution and vendored dependencies

Both reviewed top-level projects use Apache-2.0. Preserve SPDX/copyright headers, applicable notices, and provenance when moving code. Audit the **actual files selected**: iCat bundles TurboMind/LMDeploy and other dependencies; `csrc/sm70_turbomind/ops/LICENSE.v100-skinny` and `flash_qla/LICENSE` contain MIT notices; `flash-attention-v100/LICENSE` is BSD-3-Clause. Model checkpoints have their own licenses/provenance, separate from runtime source. TensorFold already maintains `NOTICE`, `THIRD_PARTY_NOTICES.md`, and `LICENSES/`; extend those records for imported material and ship the required license/header files in built artifacts.

The desired first success is a reproducible TensorFold FP16/FP32-state SM70 path with correct state, compressed weights, and TP8 graph replay. Lower-bit KV and better prefill are strong follow-on opportunities because the fork already has much of their surrounding infrastructure; their V100 quality and performance still need to be established.

