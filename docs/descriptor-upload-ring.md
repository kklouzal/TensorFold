# Private descriptor upload ring trial

Set `preparation.descriptor_upload` to `pinned-descriptor-ring2-v1` in an
explicit `trial: true` owner preparation manifest. Omission keeps the existing
single pinned mirror and its upload event. No environment override or serving
redispatch selects this candidate.

The trial allocates two pinned host descriptor slots and one completion event
per slot during the existing rank-local NUMA-bound cold owner preparation,
before States and admission. The first slot is the existing host mirror. Each
attention-layer upload rotates slots; the host waits only when it must reuse
that slot's outstanding DMA source. Both events are materialized cold. The
GPU descriptor slab, query/history owners, native reader event ordering,
descriptor fields, generation checks, inactive-row zeroing, GPU copies and
attention/capture arithmetic remain unchanged.

Every slot retains its tensor, NumPy view and actual stream through upload
completion. A copy or completion-record failure poisons the ring; retirement
attempts both slots, using the actual stream fence if its event could not be
recorded, and preserves the original failure and cleanup errors. A failed
device fence still attempts independent host-upload retirement. Owners release
their host slots only after successful retirement; failed owners remain held.
The helper inherits the GPU descriptor owner's serialized coordinator contract;
its busy flag detects serial reentry and does not make shared workspace calls
safe for concurrent host threads.

Each additional slot pays `128 * owner.rows` pinned bytes. At decode/MTP/prefill
owner rows120/120/256, this adds63,488 pinned bytes per rank (507,904 across
eight ranks), with actual Python/view/event metadata reported separately.
There is no additional GPU tensor payload. The cold CUDA-event residency is
included in existing physical preparation observations, without assuming it
is free. These are construction extents, not allocator/NUMA or full-model fit
measurements; the new pages require the ordinary native NUMA receipts.

CPU ownership, payload/backpressure, failed-record/fence, partial-construction,
unchanged descriptor-address/generation/tail and host-credit controls accompany
the implementation. Actual SM70 upload ordering, eager/captured model execution,
full24 capacity, unchanged output/teacher quality, NUMA and matched repeated
single-slot versus two-slot throughput remain required. A removed host wait
site is not a measured speedup; the event may already have completed.
