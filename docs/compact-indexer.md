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

New positional profiles initially support the unique-history layout. Legacy
replicated execution keeps its existing unbound native/YaRN policy; an explicit
new profile is refused there. A bound State requires the Engine's full numerical,
model and profile coordinate identity before allocation. This persistent identity
is separate from an active media-span identity and survives ordinary text
requests. Snapshots and prefix reuse bind it together with the positional epoch.

Explicit coordinate staging accepts ordered `CoordinateWindow` records from the
shared issuer, including separate deferred-boundary and current-span pieces.
Every piece belongs to one State's own cache rows. The candidate frame revokes
old authorization before copying and receives all segment proofs after the
batch's final mutation counter is known. Storage remains version-tracked under
inference mode; raw native writes increment that counter before submission.
All bound frames are checked before target or MTP persistent computation, not
only when the first attention layer runs. Known text coordinates are issued
from bounded host row arithmetic and the admitted continuation delta; media
coordinates require their actual producer records without GPU scalar readback.
One `InputPreparationContext` belongs to the bound Weights and is shared by
warmup, target and MTP callers. It uses the existing same-host rank store and
phase/retirement protocol, with one checked ordinal across all roles. Preparation
must enclose local leases, windows and staging before any forward collective;
same-thread nested staging belongs to the outer phase. Device preparation
defaults to fatal cohort containment on failure. Only explicitly pure preflight
may agree a validation refusal. This source policy adds control work whose
V100 end-to-end cost remains unmeasured.

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
