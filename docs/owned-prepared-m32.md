# Private prepared M32 buffer candidate

Select `preparation.prepared_linears: owned-prepared-m32-v1` in an explicit
`trial: true` private RN owner manifest. The campaign `inputs` command exposes
`--prepared-linears owned-prepared-m32-v1`; its `model` command consumes the
resulting manifest through `settings.profile_manifest`. This candidate is omitted
by default and cannot be combined with `dense_projections` or reference arithmetic
islands. It can be compared independently or combined with the separate native
family selection.

The private TP8/ETP4, HC4/D2560, bound NONE/MTP region preserves RN attention,
expert banks, full session/context promises, GPU floor and runtime workspace.
Decode/MTP Buffers through120 logical rows own physical extents rounded up to a
multiple of32. A C1 prefix can use an existing120-row owner; panel offsets32/64/96
are bounded against its actual initialized128-row storage. Source/target overlap
is declined before writes. Nonowned inputs may use the paid M32 input staging
area. Unowned output layouts, including MTP `fc_e`/`fc_h` output storage, retain
the existing provider. Prefill and ordinary/global `_ENABLED` policy are unchanged.

The operator remains the existing `GGUFDense.matmul_prepared`/`dense32` or
`GGUFConcat.matmul_prepared`/`concat32` helper. Each original full-N M32 matrix
product and Concat part order is retained; this adds no GEMM algorithm or vendor
patch. Existing512-byte alignment eligibility remains in those helpers: e.g.
FP16 projection width2060 at panel offsets32/96 can retain result-copy fallback. FP16 operands, FP16/FP32 outputs and FP32 HC partials retain their selected
contracts. Numerical identity is explicit; no reference bit-identity, model
quality or performance equivalence is claimed.

Before State creation, real target and MTP projection faces warm on initialized
M32 operands, with temporary storage retired after device completion and opaque
residency reconciled through the existing preparation owner. The ordinary Buffers
geometry pays padding and scratch exactly once. Buffer construction reconciles
its actual tensor payload with the forecast before State allocation. GDN's cold
registration accepts a strided logical projection only when it is the exact paid
`PaddedLinears.proj` view. It registers the full physical allocation span, checks
logical shape/pointer/strides, and retains existing per-layer native pointer/stride
checks; nonowner contiguous validation is unchanged.

At the current configuration, both decode and MTP have120 rows. Each role adds
2,989,824 bytes of padding,655,360 bytes of shared M32 input storage and329,728
bytes of FP32 HC partial arenas:3,974,912 bytes per role,7,949,824 bytes per rank
for both. These are raw tensor payload forecasts, not allocator/private-graph
residency or a live-fit qualification. Entry-point/library warmup, actual full24
session capacity, NUMA, partial commits, MTP4 and captured/eager model execution,
output/unchanged teacher quality and uninstrumented matched throughput remain
required before selection. Compare against both the original provider and the
existing paid `dense_projections` candidate; neither implementation's superiority
is established by source size or eliminated copies.
