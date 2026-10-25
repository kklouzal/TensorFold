# Measured SM70 execution profiles

`--execution-profile-manifest /packet/manifest.json` opts the resident GGUF
GPU-history engine into exact input verification and a measured profile table.
Ordinary startup does no checkpoint/source/library hashing for this feature.
An empty table enables identity capture; it does not qualify any execution plan.

This implementation's distributed verification contract is one Linux host and
container, with immutable shared read-only checkpoint and qualification mounts.
Every rank must observe the same host identity and the same opened checkpoint
file device, inode, size, modification time and change time. Rank zero streams
the complete content once, using a bounded 1 MiB read buffer. All ranks verify
their opened identities again before publication. A different mount, changed
file, incomplete inventory or mismatched checksum rejects the manifest.
Normal serving on other hardware does not acquire this optional restriction.

The administrator owns the qualification packet and the external experiments
that produced it. Hash verification establishes the packet's exact bytes and
scope; it cannot establish that an administrator's reported experiment occurred.
The loader requires substantive raw comparisons and measurements in addition
to a passed verdict. HTTP callers cannot supply new evidence through booleans.
No eligible V100 profile or synthetic qualification packet ships by default.

## Startup packet

The JSON object has schema `tensorfold.execution-profile-manifest`, integer
version `1`, and these fields:

| Field | Meaning |
| --- | --- |
| `artifacts` | Exact logical filename to `{size, sha256}` map |
| `evidence` | Optional list of `{path, sha256}` receipt references relative to the packet |
| `profiles` | Optional `tensorfold.execution-profiles` version 1 table |
| `preparation` | Optional fixed native attention workspace/partial-graph preparation; see below |

The artifact inventory is derived from the actual loader. `target/` contains
every selected original GGUF shard. `metadata/` contains the prepared config,
tokenizer, tokenizer configuration and any supported present generation,
special-token, added-token or chat-template assets. `draft/` contains the
selected stock MTP or DSpark `config.json` and `model.safetensors`. Additional
undeclared inputs and omitted inputs both fail. The engine captures file
identities before weight loading and verifies they remain unchanged afterward.

The effective identity includes SHA256 digests for target, draft, tokenizer,
codec, numerical settings, installed TensorFold source, actually loaded native
and runtime libraries, runtime dependencies/settings, hardware topology,
retention contract, target weight encodings and actual stationary placement.
It also includes context tokens, resident sessions, history layout and method.
`draft: "none"` means there is no separate selected draft artifact; target
bytes and the method still bind any target-contained capability.
The tokenizer digest also includes any startup-captured tokenizer override and
both compiled chat-template routes, with padding/truncation disabled as in the
actual CUDA tokenizer. An override filename alone is never its identity.
The runtime digest binds the decoder's actual decode-share setting. The numerical
digest binds every rank's complete validated CPU draft-ID map with a full SHA256
over canonical little-endian signed 64-bit IDs. Full-vocabulary execution has an
explicit sentinel and actual vocabulary count. No selection filename or truncated
rank-agreement digest substitutes for the map, and registration reads no IDs back
from CUDA.

Placement uses actual owner ranks, relative page offsets, degree and geometry.
It excludes raw GPU addresses and reusable slot generations. A pending or
partial migration refuses binding. Degree 1 evidence cannot qualify degree 4.
The runtime manifest includes GPU UUID/rank mapping, CPU affinity, GPU/PCIe
topology and NVLink status. Host identifiers are hashed. It uses an explicit
allowlist of execution environment controls, including CUDA launch blocking and
relevant NCCL algorithm, protocol, P2P, networking and thread settings.
Registration freshly reads each GPU's configured power limit, application
graphics/memory clocks, compute mode and current ECC mode. Unsupported `N/A`
values remain explicit. Instantaneous clock, temperature and P-state samples
belong to measurement telemetry and are not immutable profile keys.

`GET /v1/runtime` exposes the bound identity through the controller/profile
receipt even when the profile table is empty. The header capacity estimate,
actual allocated windows and measured execution evidence remain distinct.

## Opt-in native preparation and qualification trials

No preparation section means no additional native workspace, compilation or
capture for this feature. The following source candidate is not a qualified
V100 performance configuration:

```json
"preparation": {
  "kernel_family": "sm70-paged-owner-q8-compact-v1",
  "graph_family": "owner-attention-v1",
  "graph_budget_bytes": 67108864,
  "trial": true
}
```

Use `graph_family: "uncaptured"` and zero graph allowance for the same native
pipeline without captures. The allowance is an administrator ceiling across
all prepared roles, not a measured optimum. Optional `row_buckets` maps
`decode`, `mtp` and/or `prefill` to strictly increasing row lists ending at the
actual owner workspace bound; omitted roles use powers of two plus that exact
endpoint. A role absent from the selected method is rejected.

The engine validates the section before CUDA setup, compiles/loads the selected
native module, and prepares zero-active header-only workspaces before capacity
planning and weight loading. All TP8 ranks agree on role shapes and codecs.
The serving buffers adopt those exact objects. Their persistent bytes are
already excluded from physical free, so geometry retains only future temporary
work. An absolute CUDA cap deducts the proven preallocation once from its
remaining grant. Private graph-pool spare is never credited without a settled
allocator proof; unsupported accounting retains physical-only admission and
cannot establish an absolute-cap commitment.

These are partial attention graphs: native statistics, normalization, packed
PV and their ordered collectives. Query preparation, canonical indexer
selection, model-state updates and sampling remain outside capture. Captured
and uncaptured modes execute the same native pipeline and padded row policy.
Capture is startup-only and quiescent; a new request, session identity or layer
does not create another graph key. Startup source-built framework/library and
eight-V100 component, model, numerical, resource and performance gates remain
pending until actually executed.

`trial: true` explicitly selects the candidate for qualification and reports
performance qualification as pending. Preparation with `trial: false` leaves
reference execution selected until verified correctness/resource/performance
receipts select both `kernel_family` and `graph_family` in a profile's execution
scope. The trial marker does not change qualification identity; the actual
family, codecs, resources and bucket policy do. Preparation is fixed for the
loaded profile and cannot be changed by a later `profile_manifest` registration.

The source-built foundation audit and actual loaded binaries govern release
eligibility. Running source-level CPU controls with an older reference image
does not qualify that image for the new native stack.


## Evidence receipts

An evidence receipt has schema `tensorfold.execution-profile-evidence`, integer
version `1`, a `kind` of `correctness`, `resources` or `performance`, the exact
`identity`, `workload` and `execution` scope, `qualification: "live"`, and
`verdict: "passed"`. Its `proof` contains:

- `protocol`: `tensorfold.execution-profile-<kind>-v1`.
- `record_sha256`: the verified artifact containing the role's raw measurement record.
- `artifacts`: nonempty `{path, size, sha256}` references to actual raw files.

Receipt references hash the exact JSON file bytes. Raw artifacts are separately
streamed and verified before registration. The named raw JSON record must
use schema `tensorfold.execution-profile-<kind>`, integer version `1`, and the
same identity/workload/execution scope. Its `measurements` must contain the
following role data:

| Role | Required raw measurements |
| --- | --- |
| Correctness | Nonempty `comparisons`, each naming a contract and matching `reference_sha256`/`candidate_sha256` values backed by verified raw artifact bytes |
| Resources | All actual `ranks`, each with `rank`, `budget_bytes`, positive `measured_peak_bytes`, `protected_future_bytes`, `materialized_sessions` and `context_tokens`; every complete window must be materialized and peak plus future obligations must fit |
| Performance | At least two `pairs` of `reference` and `candidate` samples, plus a predeclared `analysis` |

Each performance sample reports `elapsed_seconds`, `prompt_tokens`,
`committed_tokens`, `offered`, `completed`, `rejected`, `timeouts`,
`ttft_seconds` and `tpot_seconds`. Failed or partial outcomes cannot disappear
from the record. Registered positive evidence requires complete outcomes and
positive finite useful-work rates and latency values.

The analysis names `committed_decode_tokens_per_second` or
`prompt_tokens_per_second`, its `confidence`, positive `lower_gain` and
`upper_gain`, and `protocol_source_sha256`. The protocol source must itself be
a verified raw artifact. Observed useful throughput must agree with the claimed
gain. The administrator's independent protocol owns appropriate uncertainty,
fairness, latency and full-region qualification; this loader does not invent a
statistical margin or treat a short-prompt observation as long-context proof.

Documents are bounded at 4 MiB each and 16 MiB of receipt content per manifest.
A receipt has at most 64 raw artifacts, each at most 256 MiB. The packet's raw
artifact verification is bounded at 4,096 files and 1 GiB. Paths cannot escape
the administrator's packet root. These are parsing/verification bounds, not
model-capacity or performance limits.

## Installing later measurements without restart

The startup packet directory establishes the allowed evidence root.
`POST /v1/runtime` with `profile_manifest` names a relative manifest under that
root. The request queues an operation; model workers execute the matched rank
operation at a coordinator boundary. No GPU collective runs in the HTTP thread.
Pending, applied and failed results remain observable.

A rebind checks unchanged process-owned input identities and reuses issued
content authority. It does not stream the full checkpoint again. It re-reads
actual numerical globals, relevant environment/dependency settings, CPU
affinity and loaded-library maps. Newly loaded libraries receive their own
bounded content capture. Physical GPU/link identity stays fixed during normal
tuning; hardware replacement or link failure is a separate operational event.
Numerical registration reads the underlying runtime-policy authority directly,
so a cached dispatch descriptor cannot hide a changed FP16/BLAS/workspace flag.
The engine gathers a fresh stationary placement digest and atomically replaces
verified evidence and the profile table. Invalid replacements preserve a prior
table when its actual facts remain valid. Changed actual runtime facts invalidate
old measurements and restore the canonical static envelope.
Migration invalidates prior placement evidence; newly measured stationary
placement evidence can be registered through the same worker operation.

The rank store holds operation-owned inventories only until every consumer
acknowledges completion. Inventory, outcome, readiness and retirement keys are
then removed. A store-retirement failure preserves one bounded cohort diagnostic
and contains further profile registration before publishing more inventory.
Other runtime failure/containment rules continue to apply.

Every rank publishes a phase outcome before a dependent wait. A missing packet
or a rank-local read refusal becomes an agreed diagnostic and retains any still
valid table. Uncertain publication, interruption or commit failure immediately
notifies the parent-supervised fatal-cohort channel; it never enters a different
retirement wait while peers are still in the failed phase. Migration requests
arriving during registration are coalesced and deferred until all ranks finish
their commit/refusal and outcome retirement. Thus an HTTP request cannot change
only the leader's placement generation halfway through profile publication.

The default bounded static envelope remains available when no measured profile
covers the actual work. A table never widens its evidence scope implicitly and
never releases retained sessions' full-context commitments.
