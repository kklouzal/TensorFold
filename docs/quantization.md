# Quantized checkpoints

TensorFold reads the precision stored in a checkpoint, including its per-layer settings.
Serve a local or Hugging Face checkpoint through the usual CLI:

```bash
tensorfold info ./model-8bit
tensorfold serve ./model-8bit
```

## Offline NVFP4 planning

Inspect a local original BF16 dense Gemma4 E4B or Google E4B assistant without
loading a model, importing a producer, downloading files or changing weights:

```bash
tensorfold quantize plan ./gemma4-e4b-bf16
tensorfold quantize plan ./gemma4-e4b-assistant-bf16
```

The JSON result is explicitly `plan-only` with `execution_implemented=false`.
Its versioned Local-Hessian W4A4 recipe is a quality candidate, not a measured
winner. It specifies E2M1 weight codes, E4M3 scales per 16 values, FP32 global
scales and calibrated per-projection activation globals. Batch-wide activation
amax fallback is forbidden because it can change serial versus speculative-row
results. The producer source is pinned; compatible container installation and
execution remain separate gates.

The initial mixed recipe selects active text attention/MLP projections in E4B,
or the assistant's 22 ordinary core projections. PLE/token tables, tied output
heads, norms, native scalar arithmetic, KV and multimodal components retain
their original precision. Stored shared-KV receiver tensors that the native
architecture does not execute are identified separately. Raw payload estimates
include these choices; they exclude runtime layout padding, allocator reserve,
calibration peaks and metadata. The selected E4B source estimates about 10.4 GB,
not a uniform four-bit conversion of its approximately 16 GB checkpoint.

Source header/configuration identities are recorded; full weight-payload
verification, observed calibration coverage, export validation, a compatible
packed loader and held-out quality/runtime checks are still required. Seeing
multimodal prefixes alone cannot prove complete modality coverage. The
assistant additionally requires actual target-conditioned calibration and an
explicit adapter before its custom ordered vocabulary head can be quantized.

FP8 and BF16 sensitivity-based component assignments, packed embedding lookup,
and quantization-aware distillation are further pipeline work; this command
does not execute those methods. Keep the original BF16 checkpoint as the
reference and export future derived checkpoints to a separate destination.

### E4B mixed-precision candidates

Precision assignments are separate decisions for storage, arithmetic and
quality. A smaller table does not establish faster inference, and a BF16
exception may protect numerical behavior or avoid an expensive conversion path.
The initial planner intentionally chooses the conservative column below.

| Component | Initial assignment | Candidate requiring qualification |
| --- | --- | --- |
| Active text attention/MLP projections | Calibrated NVFP4 W4A4 | FP8 or BF16 for projections whose held-out quality requires it |
| Token embeddings and tied output head | BF16, one shared owner | FP8 table with shared lookup/head storage and separately verified arithmetic |
| Per-layer embedding tables and associated PLE computation | BF16 | FP8 tables; retain gates, projections and normalization until separately assessed |
| Vision/audio encoders and modality projectors | Original precision | Explicit modality-aware calibration and quality evaluation before changing precision |
| Norms, scalars, rotary computation and KV | Original precision | No change in the initial recipe |
| Assistant core | Planned NVFP4 projections | Target-conditioned calibration, with explicit BF16 exceptions for paths not exercised by ordinary Linear hooks |

The experimental source includes packed projection materialization, a native
dense loader, FP8 row64 embedding lookup, assistant calibration-frame ownership
and export hashing. These components have CPU contract tests; they are not yet
connected to a calibrated E4B export or enabled by `serve`. FP8 row64 storage
is a TensorFold-specific format, not MXFP8 or a declaration that GB10 executes
every operation with FP8 tensor cores. Lookup rounds decoded rows to BF16;
the shared packed GEMM uses its own scale application arithmetic. Both need
full-model quality and serial/speculative verification.

Candidate selection must retain the same model features and evaluate decode,
prefill and memory separately on the actual GB10 runtime. Quantization changes
the target distribution: speculative verification can preserve that quantized
target's distribution, but does not recover the original BF16 distribution.

The `quants` branch extends the Qwen3.5/3.8 dense family to MLX affine weights at 2, 3, 4, 5, 6 and 8 bits, with group sizes 32, 64 and 128.
It keeps packed words, scales and biases at their stored precision, including mixed layer formats.
It does not convert a checkpoint during loading.
Choose an 8-bit checkpoint made from the original model when you want its 8-bit quality.

## Backend paths

| Path | Formats |
| --- | --- |
| Qwen dense, M5 native lane kernels | Affine 2/3/4/5/6/8-bit, group 64; affine 4-bit also group 32 |
| Qwen dense, packed row kernels on Apple Silicon | Affine 2/3/4/5/6/8-bit, groups 32/64/128 |
| Qwen dense, CUDA | Affine 2/3/4/5/6/8-bit, groups 32/64/128 |
| Nemotron | Existing affine 4-bit recipe; broader expert formats are not enabled here |
| Flash Next | Existing uniform affine 4-bit/group-32 recipe |
| GLM and EXL3 | Existing family recipes and separate EXL3 integration |

With `--lane-kernels auto`, M5 uses its native kernels when they support the checkpoint's formats and otherwise uses the packed row decoder.
M1 through M4 use the packed row decoder.
`--lane-kernels on` requires an M5-generation GPU and formats supported by its native kernels; an incompatible format is refused with guidance.
The language path retains BF16 activation requirements.
The new Metal affine projection reader accepts BF16, FP16 or FP32 affine scales and biases with matching metadata dtypes.

All paths still verify draft rows through the lane engine.
The existing specialized 4-bit routes remain available; the general readers make different formats executable without expanding the entire model into dense weights.
Their performance can differ substantially from the specialized routes.

## Mixed precision

The loader reads `quantization` before the legacy `quantization_config`, matching the current MLX loader.
A nonempty per-module mapping uses that mapping's own defaults, so `{"bits": 8}` means affine 8-bit with group 64 even when the global group is 128.
Boolean `true` inherits global settings; `false` and an empty mapping disable quantization for that module where the family supports its resulting unquantized projection.
Conflicting aliases and packed shapes that disagree with the declared bit width or group size are refused.

Fused stacks combine projections only when bit width, group size, input width and metadata dtypes agree.
Otherwise the model computes those projections separately with their declared formats.
CUDA tensor-parallel splits retain complete groups and packed-word boundaries, and memory admission counts the actual packed precision.
Changing a checkpoint, precision or kernel implementation also changes the conditions under which cached states remain valid.

## Scope and qualification

These additions cover MLX affine safetensors checkpoints for the dense Qwen family.
GGUF, AWQ, GPTQ, NVFP, MXFP and arbitrary 8-bit MoE expert formats require their own readers and kernels.
Unsupported formats remain explicit errors.

The new paths are experimental until their hardware qualification is complete.
Each backend must pass packed-value checks, one-row versus batched output equality, full-model drafted versus serial equality, prefix resume, concurrent-stream equality and memory admission before a release claim.
The Metal kernel has a dedicated hardware test in `tests/test_affine_rows_metal.py`; model and GPU tests must run under the machine's existing resource controls.
No speed or quality improvement is implied by a higher bit width.
