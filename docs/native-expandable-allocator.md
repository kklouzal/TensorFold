# Explicit native allocation and FP8 packing

`tensorfold serve ... --native-expandable-segments` maps to the dense CUDA
factory/constructor keyword `native_allocator="expandable"`. This positive
loading policy admits Qwen dense ModelOpt/compressed-tensors checkpoint math on
SM12.1, `tp=1`, `rank=0`, one stream and bf16 prompts. It sets native expandable
allocation and selects an exact FP8 byte transform for that target load.
Precision, numerical kernels, quantized values, context, cache and draft rules
are unchanged. Other families, affine/EXL3, full-precision mode, FP8 prompts,
vision and other topologies/devices refuse the explicit request before the
setter or weight payloads.

Omission preserves existing behavior, without importing or inspecting the new
allocator helper or selecting the new byte transform. Preconfigured, embedded
and multiple-engine callers keep their existing policy, including the existing
multi-stream setting. The optional byte packer reaches only this target's two
original checkpoint-FP8 constructor branches. BF16-derived FP8, MXFP8, block-FP8,
generic requantization and DFlash2 loads retain the accepted packing. Selection
is a per-load callable, never a mutable process/module setting, and loaded
projections do not retain the callable. Existing conservative resource bounds
are retained; the new transform's smaller scratch is not used to lower them.

The opt-in caller owns a newly exec'd process with a reviewed import/startup
path, no prior allocator setting calls or CUDA allocator Tensor consumers, and
exclusive initialization. Calling an importable CLI/factory inside an arbitrary
existing process does not prove this precondition. Empty segments and default
last-setting text corroborate it but cannot reveal prior freed Tensor storage
or every persistent native field. Foreign hooks/wrappers and dependency import
history need separate ownership evidence.

Before configuration the implementation requires the native backend, exact
pinned exposed default settings, zero allocator segments, and absence of every
declared allocator/cache/IPC environment override, even empty values. Existing
operator settings are not removed, normalized or reset. After metadata, device
and precision admission, it calls the already adopted native allocator setter
exactly once, before capacity allocations or payloads, and verifies effective
expansion with unchanged unrelated settings. The byte-packing module is loaded
only after successful configuration and passed to this target load.

The independently versioned allocator API/settings schema is tied to the pinned
Torch runtime and must be qualified there. Missing capabilities, warnings and
partial parse failures terminate startup. There is no retry, restore, fallback
or proof Tensor. After an attempted setting, the owner must retire that startup
process on failure; successful settings stay fixed through every consumer's
lifetime and engine close does not reset them.

The packer preserves every original U8 byte and zero padded row using bounded
4096-row chunks and the existing fragment ABI. It adds no floating arithmetic
or quantization. Ordinary current-stream Torch copies own their temporary and
output lifetimes. The legacy gather transform remains an independent default
and reference. Source-level layout/origin controls, actual native consumers,
allocator/graph lifetime/capacity, complete output/cache/speculation controls,
and paired complete-model startup/request evidence are required for selection.

The isolated composition passed 250 CPU controls in pinned AMD64 image `509b4135`,
Python3.12.3 and Torch `9026ad48`/CUDA13.4, with 34 CUDA-only cases deferred. The
same source then passed all 34 direct byte-layout CUDA cases on local SM12.0.
All 1,314 source/test entries matched before and after; actual imported project,
test and operational helper origins were checked, the protected native
foundation stayed identical, CUDA device0 settled, and both exact containers retired.
The CUDA cases call the packing helper directly: they do not activate the
optional SM12.1 allocator policy or build/force unused owned CUDA extensions.

Actual SM12.1 policy, new installed-image and full-model gates remain unrun.
Earlier source or image comparisons keep their recorded identities and do not
qualify this composition. The final
comparison must include policy checks, imports, mapping, callback/argument costs,
startup/loading, uncached and cached prefill, and decode under unchanged quality
and resource budgets. No performance winner is claimed.
