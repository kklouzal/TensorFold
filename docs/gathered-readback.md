# Private gathered sampling readback

Set `preparation.gathered_readback` to
`{"policy": "owned-gathered-readback-v1", "host_budget_bytes": N}` only in an
explicit private `trial: true` owner-preparation manifest. Omission keeps the
original allocating readback. The candidate uses the existing TP8/ETP4 HC4
Half reference-owner region and allocates one owner for each actual decode/MTP
Buffers role after complete host-capacity admission and before weights/States/capture. Prefill has no selected gathered-readback
caller and receives no extra host owner.

For maximum rows R and world8, each owner retains a pinned FP32 packet of
`8 * R * 65 * 4 = 2080R` bytes and pageable int64 ID storage of
`8 * R * 32 * 8 = 2048R` bytes. Aggregate controlled host payload is `4128R`
bytes per role; the explicit per-rank budget must cover the sum over selected roles.
The same-host capacity plan charges world8 times that per-rank total in its
existing steady host workspace obligation before any readback owner allocates.
The role forecast and actual retained payload remain in the capacity receipt;
no already-materialized credit or extra grant is added.
At R256 this is532,480 pinned plus524,288 pageable bytes per role. NumPy view,
Tensor, lock and binding metadata are reported separately; native allocator/RSS
retention is not inferred from these payload figures.

The original packet and collective remain unchanged. Readback copies only the
used contiguous prefix into the retained pinned packet, with blocking
`copy_(non_blocking=False)` on the caller's serialized current stream. Host
inspection follows completion. It retains the original FP32 value domain,
widens int32 graph ID words into retained int64 storage, and uses the existing
padding, full-vocabulary LSE, score-domain, global-ID tie, truncation/fallback
and probability rules. All existing codec status waits/checks remain. Returned
tokens and probabilities are independent Python values. A borrow ends before
calling the full-source fallback sampler; concurrent/nested borrowing is
refused. The exact weights, communicator, Buffers role, packet and host storage
must remain bound. Storage replacement/resize, capture and autograd transforms
are refused. A failed copy poisons the owner; teardown retains pinned storage
unless device retirement succeeds.

The candidate removes the original fresh host packet and int64 ID allocations,
the transient contiguous int32 ID copy, and the FP32 value `astype` allocation
(which copies even when the dtype is already FP32). It does not remove D2H
traffic or synchronization. NumPy per-row reshape/partition/masks, FP64 LSE
merging, Python result construction and Torch internal copy work remain. No
allocation-free or performance claim is made. The original readback path is
retained for independent packet/selection comparisons. Native DMA/stream,
actual eight-rank collectives, model/teacher quality and matched end-to-end
performance qualification remain pending; the default stays unchanged.
