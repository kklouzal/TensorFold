# Private prepared M32 buffer candidate

Select `preparation.prepared_linears: owned-prepared-m32-v1` in an explicit
`trial: true` private RN owner manifest. The campaign `inputs` command exposes
`--prepared-linears owned-prepared-m32-v1`; its `model` command consumes the
resulting manifest through `settings.profile_manifest`. This candidate is omitted
by default and can compose only with `dense_projections` roles `["prefill"]` for the supported
`cublas`, `cublaslt` and `cutlass` backends. Overlapping decode roles, BitBLAS
combinations and reference arithmetic islands remain refused. It can be compared independently or combined with the separate native
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


## Vendor prefill composition

Use the existing `dense_projections` selection with explicit `roles: ["prefill"]`
and `prepared_linears: owned-prepared-m32-v1`. No new backend or GEMM is added.
Dense/Concat prefill uses the existing vendor-owner face; `GGUFConcat.fused`
already owns the joined FP16 storage and its parts alias that allocation. This
selection creates no second weight bank. A fused whole-N GEMM may differ from
original per-part products, so the complete profile remains an explicit
unqualified numerical trial requiring native/model/quality and matched P3 gates.

The vendor owner prepares and pays its persistent prefill arenas before physical
admission. Prepared M32 pays only its existing decode/MTP Buffer forecast through
ordinary geometry. Both actual-weight bindings run serially before State creation;
their independently settled opaque commitment deltas are summed once, while
serial temporary peak is the maximum. No final commitment is published if either
owner cannot prove its endpoint. Adoption attaches vendor frames only to pbuf and
prepared buffers only to decode/MTP. The existing vendor thread handoff and
post-graph retirement contract remain required.

MTP prompt absorption also borrows pbuf. Cold prefill adoption records only its
actual distinct MTP FC, HC and layer faces as reference exclusions. Unknown
matrices still fail closed; target aliases remain eligible target faces. This
preserves the supported shared-GGUF method path without applying vendor prefill
to an unprepared MTP face. These exclusions release their tensor references on
positive frame retirement.

For a complete32-row contiguous prefill input, the existing fixed-M32 plan can
read its live source directly when its stride and alignment satisfy the already
sealed scratch-input contract. It uses the same plan, weight, full N and output
selection. No runtime plan discovery or new arena is introduced. Partial tails,
other strides and insufficient alignment retain the paid initialized input
staging and original output fallback. The ordinary provider, decode vendor path
and optional direct-CUTLASS geometry keep their established dispatch.
