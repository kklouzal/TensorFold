# Private observed graph demand

Source27 recorded 190 not-captured misses among 252 multi-participant C8
attempts. Common keys included MTP (3,3), (5,5), (6,6), (7,7), and verifier
(28,8), (25,8). These are host attempt/replay API observations, not GPU occupancy
or performance/quality acceptance.

The explicit `observed-demand-v1` policy uses the same `keys` and
`evidence_sha256` fields as the historical subset policy. Its digest identifies
the immutable observed-demand/source evidence. It does not forecast memory.
Only this policy permits compact ragged totals and previously unlisted State
counts. Every key must fit the current State count, role buffers and depth+1
per-State row bound, before any reset, staging or capture.

The cold registry is the original catalogue plus the explicitly admitted keys,
not all possible integer shapes. A balanced real-State segment pattern seeds
each distinct (total rows, participants) program; live attention/PLE/State
handoffs and MTP last-row maps continue to stage the actual runtime layout.
No padded State envelopes, live capture or new kernel math are introduced.
Default preparation and `paid-subset-v1` retain their original order and shape
restrictions.

All requested keys must actually capture on all eight ranks for a requested-fit
claim. Exclusions of unrequested legacy keys are policy declines; actual
pre-capture no-room or post-capture budget failures remain failures of the
experiment. Keep the actual AUTO graph grant, service workspace, device floor
and full-context session commitments unchanged. Source support alone does not
qualify arbitrary ragged replay: native controlled unequal-length and reordered
State cases, serial/MTP outputs, conditional quality and repeated matched
end-to-end regional measurements remain required.

The controlled native component oracle is
`tools/v100_ragged_replay_oracle.py:qualify(decoder, snapshot_payload_bytes=...)`. Invoke it on every TP8
rank of the same loaded idle decoder after cold capture and before request
admission. It requires actual `ExactStateGraphs` with both `(28,8)` role keys
captured under `observed-demand-v1`; it never captures or substitutes an eager
fallback. It reuses ten existing empty slots, ordinary generation/reset and
staging, and compares two reordered eight-slot subsets with different unequal
row distributions against eager execution of the same fixed tokens. Real
commits establish unequal contexts and both GDN banks. It checks exact target
and MTP outputs, rank candidate publications, HC rows, MTP last-row maps,
partial commits, valid local KV spans, compact indexer tails and pools. Only
short valid prefixes and committed recurrence are copied to CPU; there are no
full-window State clones or additional weights. A failure propagates to the
existing caller's cohort/process containment; successful slots reset before
admission. This gate is implemented but native execution remains pending.
It is a short-context component check, not qualification of all 24 windows,
conditional model quality or P3 performance. The caller must retain its actual
image/source/native identities and all eight returned rank receipts.

Use the existing supervised `tools/v100_owned_qualification.py` parent with
`--ragged-replay-input /absolute/input.json` and the ordinary model arguments.
The input is a closed object (maximum 4096 bytes):

```json
{"schema":"tensorfold.native-ragged-replay-input-v1","key":[28,8],"layouts":[[5,5,5,5,5,1,1,1],[1,5,2,4,5,3,4,4]],"slots":10,"maximum_context_rows":10,"comparison":"bit-exact"}
```

This operation requires explicit TP8/ETP4, MTP depth4, 24 resident full
262,144-token Q8 States and graphs; it excludes benchmark, teacher and other
observer captures. The authoritative model header computes the bounded CPU
reference obligation before Engine/State construction. Existing immutable host
admission pays both reference trees and transfer peak; actual tensor dimensions
are checked before probe mutation. The pre-listener callback writes
`ragged-replay-rank-N.json` on every rank and uses the existing fatal observer
path on any failure. All 24 States must remain clean and idle afterward. No
new serving option or model/context reduction is introduced.
