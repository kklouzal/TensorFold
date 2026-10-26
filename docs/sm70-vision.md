# Bounded SM70 vision component

This component implements the GGUF vision import and fixed-device encoder
boundary for the [adaptive design](v100-adaptive-inference-design.md). It does
not yet connect a complete image/video request to TP8 serving. The bounded CPU
frontend, replay, admission/resource envelope, target-input transport and final
State/MTP integration remain separate work.

`mmproj_weights.plan_mmproj` validates the complete named tensor inventory and
declared `VisionLayout` before allocating parameters. The initial profile uses
RGB, temporal patch width two, no deepstack, the final merger and the pinned
LayerNorm epsilon. F32, F16 and BF16 source parameters convert to FP16 in bounded
source-row reads; nonfinite values and FP16 overflow fail. The source-read bound
does not bound the full import peak: account for retained parameters, temporal
kernel assembly, Reader storage, FP32 decoding and FP16 conversion separately.

The mapper follows the original converter at
`ggml-org/llama.cpp` commit `b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9`:
QKV order is unchanged, and the two temporal patch weights join on Conv3D's
temporal axis with one bias. The meta constructor receives one strictly assigned
FP16 parameter set. The unchanged pinned Transformers axial-frequency builder
runs explicitly on CPU, independent of the caller's default device; its two
nonpersistent frequency buffers remain FP32. Target context/YaRN transforms do
not alter the vision transform.

`SM70VisionEncoder.create` requires the enclosing loader to register the owner
before fallible model/device work. It verifies the process-issued
[combined source/media authority](sm70-software-binding.md), a framework built
for SM70, the actual device and the initial H16/D72 geometry. It uses the exact
Qwen4Exp model from Transformers commit
`50e35cc0ca18ec19b80653690de553e6decf74f5`, with only Torch's memory-efficient
SDPA backend enabled. Unsupported dispatch fails. Public library helpers retain
complete per-frame noncausal attention segments, learned-position interpolation,
rotary order and merger order; the implementation does not spatially truncate
attention or allocate an explicit score matrix.

The caller supplies a positive, signed-int32-safe patch bound and complete
CPU-int64 grids plus CPU-FP32 normalized patches. One encoder working unit is
live at a time. Source and derived CPU metadata remain owned through transfer
and completion; only required pixel/interpolation/position tensors move to the
fixed GPU. Output ownership passes to the caller until target/speculative
conditioning and transfer retire. This is a bounded unit, not a projected-prefix
archive or another persistent encoder copy.

Vision GEMMs require FP32 accumulation and reductions. The component validates
the actual framework flags and records the independent FP32 split-K setting;
it does not change process globals. The enclosing engine must own and bind the
numeric scope before loading/warming shared target/vision operators and restore
it only after all consumers retire. Software reissuance occurs explicitly after
warmup at adoption boundaries, never per request. Successful close fences the
fixed device before releasing owners. A failed fence retains them. Coordinated
construction-error recovery must additionally retire captured exception frames
through the existing engine error-transport protocol.

## Evidence and remaining gates

Source checks in the selected CPU media environment compare every tiny-model
parameter and complete outputs with the pinned HF implementation, including
temporal order, differing grids, interpolation, merger, enclosing-meta CPU
preparation, supplied-metadata-only forward, malformed inventories and overflow.
They also cover explicit arithmetic settings and failed-retirement ownership.
The actual selected 907,542,944-byte mmproj has been imported and executed on a
small synthetic CPU input; this establishes neither supported-image/task quality
nor a device memory envelope.

The selected source-built Torch attention dispatch includes SM70 FP16/D72
memory-efficient kernels. Actual V100 eligibility, numerical accuracy, temporary
storage, maximum legal unit, library warm residency, stream/collective behavior,
full-model visual quality, complete media lifecycle and throughput are unrun.
`tests/cuda/test_sm70_vision_sdpa.py` is a device-only constant-value attention
control, not a replacement for those model/integration gates. No performance or
release qualification is claimed.
