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
admission. The Source30 native execution below passed this gate on all eight ranks.
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

Each rank's existing oracle receipt also exports bounded before/after accounting:
all 32 original-plus-requested captures and their admission observations, original
settled graph/pool records, unchanged AUTO budget and full-window/service
commitments, and all 24 actual empty State frontiers/generations. A separately
labelled fresh quiescent CUDA sample records free/allocated/reserved bytes and
checks the unchanged default device floor. Capture-settlement allocator values
retain their original startup timing. Both metadata trees and encoding bounds
are admitted before Engine construction. Missing/declined keys, inconsistent
charges, changed commitments or an insufficient fresh floor fail before the
listener. The Source30 result below includes actual all-eight-rank execution.

`tools/v100_ragged_replay_gate.py` is the standalone host owner. Its closed
`tensorfold.native-ragged-gate-operation-v1` DATA binds immutable source/image,
actual installed-check file identities, read-only profile/oracle/demand assets,
ordinary NUMA authority, host resources, GPU/link/Xid baseline and its separately
frozen host-helper closure. A separate `tensorfold.native-ragged-gate-root-go-v1`
record binds that operation digest, controller digest, image, source and current
boot before container effects. Invoke only with reviewed final DATA:

```text
python3 -B tools/v100_ragged_replay_gate.py --operation /absolute/operation.json \
  --operation-sha256 EXACT_SHA --root-go /absolute/root-go.json \
  --root-go-sha256 EXACT_GO_SHA --out /absolute/fresh-attempt
```

The runtime uses the original frozen owned qualifier and installed application;
the host controller may be frozen separately when those runtime bytes are
unchanged. It captures all eight oracle/accounting/startup receipts, validates
settled full-window READY and actual ordinary rank UID/GID/CapEff0, then retires
the owned CID through normal SIGINT and bounded wait. Source/input/NUMA/health
changes, missing rank evidence, acquisition/publication errors and removal
failures remain nonzero outcomes. No benchmark, teacher, profiler, smoke or
HTTP inference request is issued. Kernel-Xid collection uses the existing
noninteractive host journal permission; an unavailable query fails visibly.

## Source30 actual eight-rank component result

The loaded SC117 component gate completed on all eight V100s with TP8/ETP4,
Q8 KV, MTP4, 24 resident logical 262,144-token windows (262,149 physical rows
including method guards), P256/E120, the default 4 GiB device floor and 2 GiB
workspace. It used source `13e1ab6f`, immutable image
`sha256:0ad2d11d26cc921f72c675d0e787b590c9da9de2ae3504bdd21c98ed290d6426`,
frozen source UID `adcb9120776f501629560257c6b39b2d33fd3a880bcf2e5387af8711fc2edd1a`,
and the separately frozen, reviewed standalone host controller. The image's
887 CPU tests plus 38 subtests, all 848 installed members, FD owner, protected
SDK and actual Triton 3.2 checks passed before this native run.

Each rank passed both exact eager-versus-captured `(28,8)` layouts, reversing
slots 7..0 and then using the overlapping reversed subset 9..2. This is 16
case comparisons and 32 target/MTP role comparisons across the eight ranks.
The source oracle checks logits, candidate packets, HC, committed recurrent
state and valid cache/indexer prefixes, partial commits, MTP last-row maps and
ordinary reset/generation reuse. All 24 State frontiers were clean afterward.
The histories exercised here were short; allocating all full windows does
not qualify near-limit history execution or full-context throughput.

All ranks captured the original 30 keys plus verify/MTP `(28,8)`: 16 verify
and 16 MTP keys, 272 pieces per rank, with no declines. The recorded graph
charge was 1,182,793,728 bytes (1128 MiB), settled charge 1,180,696,576 bytes
(1126 MiB), within the unchanged 1,503,076,308-byte AUTO grant. These are
capture-settlement observations, not additive predictions for other catalogues.
The oracle's global host obligation was 4,479,619,888 bytes, admitted before
construction; it requested no additional device payload.

Fresh post-probe CUDA samples on every rank reported 8,466,726,912 free bytes
(8074.5 MiB), above the 4 GiB floor. Relative to the before-probe sample,
allocated bytes were unchanged while reserved bytes increased by 20 MiB;
the historical graph-settlement records remained unchanged. A separate host
NVML observation, bounded by live-container checks after READY, reported
8075 MiB free on each GPU. NVML's displayed MiB precision is separate from
the CUDA byte samples; neither is an allocator or peak-memory timeline.

The service reached idle READY with zero inference requests. All 24 NUMA
receipts and eight distinct UID/GID1000, CapEff0 rank processes reconciled.
The controller exited 0 in 308.4 seconds, with no OOM or cleanup error, removed
its owned container and retained source/input identities. All eight GPUs'
ECC/retired-page baseline and 48 links were unchanged, with no new Xids,
link errors or remaining compute contexts. IPv6 localhost fallback warnings
were retained; the host has IPv6 disabled and startup proceeded.

Raw evidence is retained in the private
`onecat-integration-20261113-v30/ragged-native-gate-model-v1` packet, with its
separate post-READY NVML record and raw-output seal. Independent review
reconciled the per-rank cases, capture records, samples and retirement.
This functional run overlapped a CPU dependency build and supplies no P3
or model-accuracy result. The separate legacy six-lane exact-output failure
remains unresolved. Non-bucket MTP participant counts 3/5/6/7, other requested
shapes, full-history behavior, conditional quality and matched regional
performance still require their own evidence.
