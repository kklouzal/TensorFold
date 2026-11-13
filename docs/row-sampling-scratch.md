# Private canonical row sampling scratch

Set `preparation.row_sampling_scratch` to `canonical-row-scratch-v1` in an
explicit private `trial: true` manifest. The candidate requires the existing
bound TP8/ETP4 HC4 reference-owner NONE/stock-MTP region and evenly sharded
target vocabulary. Omission retains the original packet-producing path.

Cold owner preparation allocates and accounts one scratch set per actual
decode/MTP/prefill Buffers role before weights/States. Buffer construction
binds the original model, communicator, role, logit/packet owners and original
draft ID map. Captured exact-row replays validate those bindings before GPU
submission. Tensor and storage owners remain retained through graph/model
retirement; this is the existing serialized current-stream contract, not a
concurrent-stream backend. Each queued row's reads/copies precede reuse on the
same stream. Unsupported storage/map/autograd/layout states reject explicitly.

The original FP32 score row is copied into a retained512-byte-aligned row with
the same compact1×width stride. `torch.topk(sorted=False)` writes values into
compact packet-row views and indices into retained int64 storage. Original ID
addition/index selection uses retained int64 output, then converts directly
into the packet's int32-bit field. Standard `torch.logsumexp` writes its1×1
FP32 result directly into the packet. Padding, collective count/order, full
vocabulary LSE, downstream tie/truncation repair, seeded sampling and codec
status/error checks are unchanged. The score-row copy remains necessary; this
candidate removes output allocation/copy-backs rather than that copy.

Raw persistent GPU bytes per role are
`4 * (width + 127) + 16 * min(32, width)`. At local width31,040, that is125,180
bytes per role,375,540 per rank for three roles, and3,004,320 across eight ranks.
Actual Python metadata headers are reported separately; these are not complete
host/native allocator RSS measurements. No new host readback or pinned payload
is added. Torch's internal top-k/logsumexp workspace and temporary allocations
remain uncontrolled, so this is not an allocation-free claim.

The unchanged original `forward.candidates` path is the packet oracle. The
focused193-control CPU gate passes on the protected SDK, including complete
packet bits and candidate/receive mutation-version deltas. A separate installed
Source28 SM70 component gate passed all13 complete-bit, mapped-ID and
changed-input graph cases with protected application/SDK provenance and NUMA0
binding; see [the exact component scope](v100-adaptive-validation.md#source28-prepared-router-and-sampling-component-result).
Those cases use a synthetic collective fixture. Actual eight-rank NCCL,
full24 capacity, model/teacher quality and repeated matched throughput remain
pending, as does the separate legacy six-lane graph mismatch. Source-level
copy/allocation removal is not a speed result.
