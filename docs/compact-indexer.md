# Compact unique indexer state

The adaptive unique GPU-history path stores each completed pooled indexer key
once at its canonical owner. Only pooled keys grow with context. Every indexed
cache owns a four-row FP16 unnormalized, unrotated ring and its int64 THW
coordinates; zero to three members are valid at a committed frontier. Protected
append carry and bounded new-span storage cover speculative and checkpoint cuts.
No full raw GPU history or host archive is a fallback for this path. Existing
legacy archive/other-hardware paths keep their separate contracts.

Canonical ownership remains `(cache_count * slot_id + ordinal) % 8`. The count
is fixed by the selected capability: twelve target caches, plus stock MTP's
own cache when enabled. A zero-proposal setting does not change that phase.
Target execution staging uses one channel per eight ordinals per execution
row; only the canonical owner writes it. All layer spans survive until target
verification resolves. MTP staging has an independent epoch and its proposal
base/span can outlive a target round or another request's buffer use.

Pooling forms complete four-row groups from protected carry and new raw rows.
It sums in FP32, stores the FP16 mean, applies the existing key RMS transform,
then rotates at the first member's actual coordinates. Each query reads only
pools complete at its own causal frontier and appends visible incomplete-tail
KV positions. Grouping rows, ring slots and rotary coordinates are distinct.
The compact core consumes the resolved immutable frequency/amplitude policy;
it neither infers MTP coordinates from a shifted embedding nor retunes context
extension from live load.

Each recoverable cut owns its pooled-valid frontier, grouping origin and
pending raw values/coordinates. Accepted-prefix commit promotes only those rows;
rollback invalidates later pools and restores its protected tail. Prefix forks
copy completed pools and relocate the captured pending tail to the new slot's
canonical owner. Pooled means never reconstruct discarded raw members. Bounded
copy staging is agreed across all ranks before transfer; uncertain transfer
failure poisons the affected cohort before ordinary acknowledgements.
Each source and destination State also owns a transfer fence covering all KV,
pooled-key and tail reads/writes, including ranks without an indexer cache.
Subsequent forward, restore, reset or slot reuse waits that fence when the
coordinator changes CUDA streams. Partial operations still retire issued uses;
failed retirement retains the borrowers and prevents either State's reuse.

Guarded bitwise publication checks live slot/ownership generations before
writing persistent compact state. Snapshot/tail storage and execution arenas
remain owned through their asynchronous consumers. Capacity geometry includes
their fixed extents, recovery versions and role staging separately from growing
pooled/KV history. The selected context remains a profile input; these source
changes establish no new resident-session count.

Source/host controls and source-stack compilation are separate from actual
SM70 execution. Eight-V100 address, capture, numerical, resource, extended
position, media, lifecycle and performance gates remain required before a
native/extended SC117 profile is released. Historical full-raw or other-target
receipts do not qualify this representation.
