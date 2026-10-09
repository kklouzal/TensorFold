# Qwen3.5 FP8 prompt admission

The shared prompt path selects FP8 glue only when `prompt_precision.fp8()` is
enabled before loading and every consumed projection satisfies
`Weights.fast_prefill`. MLX affine projections require four-bit words, groups
of64, and bf16 scales and biases. EXL3 and checkpoint-specific NVFP4/FP8 math
select their existing bf16 glue. NV full math exposes FP8 prompt projections
when the option is enabled.

FP8 attention gating creates one Triton block of
`next_power_of_2(local_query_heads) * head_dim` cells. GDN gated normalization
creates `next_power_of_2(local_value_heads) * value_dim` cells. Both blocks must
fit the installed provider's `TRITON_MAX_TENSOR_NUMEL`, as enforced by Triton's
[block-shape validator](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/_utils.py).
The existing general
geometry admission covers different blocks and does not prove these products.
This is a compiler tensor limit, without an added device-memory or GPU model
requirement. Layer types absent from the configuration impose no such check.

The MLX loader proves selection from its already-read SafeTensors headers and
affine metadata, then checks these products before reading the first weight
payload. The NV loader checks its selected full-math route at the same boundary.
A rejected load drains the existing reader through its ordinary failure
cleanup. Successful tensor reads, packing, floating-point operations and prompt
kernels retain their original order and precision.

Two-rank MLX startup passes `prefill_world=2` before loading full weights. The
admission uses the eventual per-rank heads; the existing shard routine still
validates all its other divisibility constraints. No two-device runtime pass is
inferred from these integer checks.

The public custom MLP callback can declare its consumed gate/up/down projection
names with `prefill_mlp`, an immutable tuple of unique names. The existing routed
MoE loader declares an empty tuple because its dense projection fields are
absent. An undeclared custom callback retains its existing construction; the
loader checks its resulting selected FP8 geometry before returning weights.
Early payload admission for a custom callback therefore requires its accurate
projection declaration.
