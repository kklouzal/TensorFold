# CUDA B16 linear and prompt contracts

The Qwen B16 entry points borrow contiguous CUDA fp16 or bf16 matrices on one
device. The linear bias, when nonempty, has the input dtype, device and N
elements. An empty bias is unused. Paired projections may have different output
widths and both weights have the input dtype and K. Input and weight reads may
share storage. The returned outputs own separate allocations.

Linear vector loads and prompt staging require 16-byte input and weight
alignment. K is divisible by eight for linear and by 64 for prompt. Zero K
preserves the original zero-dot-product or bias result without input loads.
Rows are positive. A paired linear projection may have one empty output width;
its other weight defines the grid. A paired prompt may do so when K is zero.
For nonzero K, prompt weights have positive width: Torch's untyped data pointer
is null for empty tensors, and zero-fill staging still forms K-row pointers.
The host checks signed-int padded row,
column and linear loop bounds before narrowing; it checks the actual device's
grid limits before allocating outputs. The prompt policy's paired width stays
int64 while choosing the tile. Arithmetic and rounding inside both CUDA kernels
remain unchanged.

Each compiled prompt tile owns one configuration flag per actual runtime CUDA
device through the shared `KernelConfiguration` contract. The device guard
selects the tensor's Torch primary context. A successful shared-memory attribute
setter commits that device's flag; a setter failure throws before launch and
leaves the flag uncommitted. Runtime-visible devices and primary contexts remain
stable for the loaded module's lifetime. Resetting contexts requires process
restart. This imposes no additional device-count ceiling.

Callers publish inputs before the consumer stream uses them and retain their
storage through asynchronous completion, or register that consumer with the
allocator. Graph callers warm the selected tiles before capture and retain
inputs and outputs through every replay and outstanding use. Configuration
checks do not establish producer readiness. Native qualification must cover
both dtypes, scalar and paired outputs, linear row buckets, prompt tiles,
layout and failure paths. One-device execution does not qualify switching
between devices; that gate requires two compatible CUDA devices.

## Private Gemma CG composition

The private composed candidate preserves the ordinary entry points above.
Only the eligible prepared Gemma BF16 head binds the existing separate
`gemma_b16_head_cg` entry, with R1/R4/R8/R16, the original shape/grid/alignment
checks, current stream and fresh output ownership. Its weight-only CG load
retains the same authoritative FMA/XOR/BF16 chain; input loads remain LDG.
The generic Python factory and binder are unchanged, and normal compilation
excludes CG instantiations. Changed shared source bytes can still cause normal
cache/build work and must not be called startup-neutral.

Bits8 DFlash owns a private FP8 Head built from the same original BF16 weight;
it does not invoke this BF16 target callable. The current FP8 provider
preparation, public FP32-logit contract and default profile remain unchanged.
Inherited Plain.prefill and embedding access remain available, while eligible
target project_logits calls use the prepared head wherever requested. Policy
binding precedes graph preparation and storage/provider owners outlive all
consumers. Four prepared partials, a separately cached provider/DSO and its
compile/code/cache residency are counted; no tensor/weight copy is added by
this policy. Singleton wrapper and adaptive-row interactions prohibit pure
cache-attribution claims. All new-composition execution/ARM/model/P3 gates
remain pending; see the Gemma validation guide.
