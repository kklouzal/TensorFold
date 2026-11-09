# V100 one-session phase matrix

Use the maintained `tools/bench_concurrent.py` matrix mode with the private
`tools/v100_owned_qualification.py` observer. This attaches to one loaded
SC117 service for all warmups, levels and repetitions; it never launches or
restarts the server itself. The existing finite launch supervisor owns container,
image, weights, provider identity and retirement. Tool/plan hashes are separate
from that runtime provenance. The older generic OpenAI decode-arrival mode remains
available and keeps its explicitly estimated SSE timing scope.

The standard levels are **1, 2, 4, 8, 16, 24, 32**. These personal deployment
profiles retain context 262,144 and MTP depth 4:

| Profile | Administrator ceiling | Eligible standard levels |
| --- | ---: | --- |
| INT8/Q8 KV | 24 | 1,2,4,8,16,24;32 excluded by administrator cap |
| `rotorquant-iso4-sm70-v1` KV | 32 | All seven, subject to actual safe residency |

These are administrator choices, not generic engine limits or measured hardware
maxima. Use the maintained [V100 personal profile recipe](../deploy/v100/README.md#candidate-launch)
with explicit parallel/full-window24 or32 and the corresponding admitted frontend
budgets. The tool checks actual selected cache format, full context, configured
ceiling and residency. Insufficient actual residency is unsupported, never
relabelled as successful concurrency. Q4 component tests alone do not qualify
32-way model capacity, output quality or performance.

## Prepare once, load once, run the complete short matrix

Run preparation in the existing application environment containing tokenizers,
against the exact read-only frontend directory that the service will use. The
plan contains pretokenized natural-text completion prompts, not random special
IDs. A recorded fresh run identity and distinct natural prefixes prevent reuse
across warmups, levels, repetitions and streams. Native tokenization establishes
exact input length; no character-to-token ratio or prompt truncation is guessed.
The worker verifies the actual tokenizer file and full prompt hashes.

```bash
python tools/bench_concurrent.py "$URL" "$MODEL_NAME" \
  --prepare-matrix "$MODEL_FRONTEND_DIR" --profile q4 \
  --input-tokens 64 --tokens 256 --reps 1 --output matrix-plan.json
```

The default standard levels are already selected. Preparation does not contact
or load the GPU model. Use fresh plan/results paths. Stage the canonical tools
and their existing protocol/lifetime/output helpers together read-only, keeping
their commit/byte identities. Add the plan to the existing private supervised
launch, preserving the profile's ordinary model/context/method/cache arguments:

```bash
python tools/v100_owned_qualification.py --gpus "$TF_RANK_GPUS" \
  --out /qualification-output/new-run --benchmark-plan matrix-plan.json \
  -- "${serve_args[@]}" --expert-tensor-parallel-size 4
```

This observer currently uses the existing private owned-native SC117 capability;
it does not promote that policy to a public/default serving mode. Its bounded
metadata obligation is passed through the existing private preparation/admission
owner. Four CUDA event objects are reused sequentially between cohorts. Ordinary shipping class definitions/default instances and installed libraries
remain unchanged; only the selected private instance uses source-owned wrappers.

For a controlled dense-operator comparison, the private launcher accepts
`--dense-implementation reference` or `--dense-implementation native32` before
the `--` serve-argument separator. Keep `TENSORFOLD_SM70_PROFILE=reference`.
Each worker selects Dense before model preparation and warmup; Concat remains
reference. Every rank receipt records requested/actual Dense and Concat, and
a mismatch prevents readiness. Omitting the option preserves prior selection.
Use an immutable application image containing the exact candidate bytes and
record its digest/source identity; the option cannot add code to an older image.
This is a private trial selector, not an optimized-profile switch, production
promotion or evidence of a speedup. Preserve matched inputs and all other
settings, and run comparisons without the scheduler profiler.

After normal readiness, perform the ordinary arithmetic MTP/cold-serial smoke
checks in the same loaded container. The observer has an explicit unobserved
pre-matrix phase. After the first planned cohort starts, it rejects unplanned
model work or cross-cohort overlap. Then run:

```bash
python tools/bench_concurrent.py "$URL" "$MODEL_NAME" \
  --matrix-plan matrix-plan.json --output matrix-results.json \
  --timeout 1800 --response-mib 16 --sse-line-kib 64
```

A barrier releases each cohort's callers together. Every response must complete
its declared fixed work with `ignore_eos`, complete IDs, matching independent
ID SHA, `length` finish and zero reported cached prompt tokens. Transport,
protocol, stale-service and control failures remain in partial results/raw
exchanges; they do not become successful measurements. Socket inactivity and
post-cohort readiness polling are bounded by `--timeout`; this is not a claim
of a whole-suite wall deadline. All child callers drain before return.

One short warmup per input size precedes measured repetitions in the same
session. With one repetition, Q4 sends 1 warmup + 87 measured requests; Q8 sends
1 + 55 and records 32 as administrator-excluded. The token budget must be long enough
for actual backend overlap; offered simultaneous requests alone do not qualify a
level. No forced model scheduling or hidden server reconfiguration creates that
proof. If overlap is not observed, keep the row unqualified and its raw data.

## Private grouped-down and reference-tile trials

These source candidates remain unselected. Use an immutable application image
containing their exact bytes; flags cannot add missing code to an older image.
Keep the profile's ordinary TP8/ETP4/PP1, full context, precision, MTP, caps and
serve arguments unchanged. Preserve `TENSORFOLD_SM70_PROFILE=reference` for the
reference-tile trial.

Inside the candidate container, set
`TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN=grouped-v1` before starting the existing
private supervised launcher to request grouped expert down projection. Unset or
`inherit` retains the incumbent. Separately, add `--reference-tiles` to
`tools/v100_owned_qualification.py` **before** its existing `--` separator to
request reference Tile128 preparation. The trusted capability reaches the
ordinary factory/Engine; it is not a public serving/admin flag or a switch to
native-owner WMMA attention. The default omits the capability.

Reference preparation loads/checks both CUDA entry points in the voted startup
phase before the physical baseline and admission. Decode/prompt/MTP owners are
journalled and adopted before warmup; retain all-rank source/ELF/slab/role
receipts. The reference preparer borrows a paid 67,072-byte receive prefix;
module residency is separate. Grouping physically adds 46,148 bytes per rank in
the recorded 32/256/32-row arenas inside the paid 292,864-byte reservation, rather
than allocating zero bytes. Original arithmetic folds remain unchanged;
component comparisons use no tolerance.

[Component and source-integration evidence](v100-adaptive-validation.md#private-grouped-down-and-reference-tile-preparation)
are separate from model and P3 selection. The final combined source passed
408 selected-x86 CPU and 17 updated GPU checks, including real entry-point
preparation before owner-buffer construction. No new speedup or deployment-default claim follows
from source availability or strict component passes. Compare only after normal
readiness/resource admission, with matched recorded inputs and controls;
retain failures and existing baseline reproducibility limits.

## Private verification-row envelope trials

The supervised qualification launcher accepts `--execution-rows N` before its
`--` separator. This experiment requires an explicit MTP method, draft depth
and parallel count in the serve arguments. It validates the complete-window
bound before construction and forwards the requested envelope through the
trusted Python capability into the ordinary factory, planner and allocator.
No administrator/public serve option, context or concurrency default changes.

For example, a 40-row envelope can hold eight complete four-proposal MTP
windows; the current 32-row default holds six. Actual draft lengths and fair
ready-work selection still determine each round. Every rank records and checks
the requested rows against the engine, target/MTP buffers and controller
maximum/configured budgets before publication. This cannot hot-raise a loaded
32-row controller or bypass physical admission. Omission preserves the prior
constructor and default envelope; an older CLI without capability support
rejects an explicit request.

Source integration passed 308 selected x86 CPU checks with zero skips. Installed
successor, physical allocation, active-work/quality and throughput comparisons
remain pending; planner estimates are not grants or performance results.
Review/evidence: `private-execution-rows-v1/reviewer-final-source-CPU-v3.json`,
SHA256 `0c483856461b8d77b7800a6b94dc4e31b9aec48c1b226bc696963fd2e873fc16`.

## Denominators and actual overlap

The private worker samples the full non-done `streams`+`filling` set at successful
worker boundaries, checks distinct actual GPU States/wires and planned prompt
identity, and reports actual owner high-water. Forward microbatch State count is
reported separately: MTP rows/windows are not additional requests. A requested
level satisfies the measured overlap criterion only when its real backend owner count reaches that level.
A large HTTP queue or a 32-row matrix cannot substitute for 32 owners.

| Metric | Numerator | Denominator/scope |
| --- | --- | --- |
| Prefill envelope TPS | Uncached original prompt tokens, once per request | First `_prepare_pass` entry to final original-prompt `_joined` entry on rank0's actual current CUDA stream |
| Decode envelope TPS | Sum of `max(completion_ID_count-1,0)` | First positive first-token `_take` commit to last cohort final `_take` commit on that same stream |
| Per-request reported host rates | Prompt tokens or output IDs after first | Existing `prefill_s`/`decode_s`; prefill includes admission setup and apportioned shared-pass host time |
| Cohort HTTP-wall TPS | Completed output tokens | Earliest declared barrier-released client start to latest complete response, including local protocol recording/I/O, serialization, transport, queueing and server work |

GPU-stream envelopes are **inclusive**. They may overlap each other and include
interleaved prefill/decode, communication and CPU submission gaps. They are not
kernel-only busy times, mutually exclusive phase partitions or summed per-request
service durations. Rank0 is the declared critical-stream clock domain; tokens
are never multiplied by TP8. TTFT is not used as a pure prefill denominator.
The client constructs its already-tokenized request before the barrier, so the
API-wall measure includes the canonical client's request-directory creation and durable request recording before network acquisition; it does not represent ordinary text-tokenization latency or a network-send-only interval.

Events are recorded only at the four cohort boundaries; there is no per-step
synchronize or disk flush. Timer query/elapsed-time collection is deferred until
the cohort has completed and the decoder is idle. Immutable published snapshots
bind the plan identity and actual serving instance. Missing/zero/invalid timing
or failed identity/overlap checks produce an unmeasured/failed row, not fabricated
TPS. The observer's instrumentation is part of this measured profile; absolute
figures are not a matched speedup or general quality/capacity claim.

## Bounded scheduler diagnostics

Active capture windows additionally label `attention.partial_statistics`,
`attention.global_statistics` (the global merge) and
`attention.normalized_partial`, plus `HC.writeback`, `HC.normed`, `HC.act` and
`HC.mix`. These ranges enclose the regular functions that actually execute the
tile/body work; they do not wrap only generator creation or enable Python stack
profiling. Outside an active capture, the wrappers call the original directly.
The five-window limit, scheduler-thread guard and synchronization behavior are
unchanged. Nested/inclusive ranges and correlated GPU activity aid attribution;
annotation overhead is diagnostic, not a clean throughput measurement.

Tool source `5e412dfd2154623ecf048d57bae07d7081475d0e` passed 205 CPU controls
with zero skips on the selected x86 runtime in the same `597c7435` application
image; CUDA remained uninitialized. The earlier local ARM resource-guard refusal
is retained separately and is not a new target failure. This validates the
expanded hook's controls, not an actual new profiling result. Gate:
`combined-core-residual-trace-preparation-v1/CPU205-actual-gate-v1.json`, SHA256
`3535996d18a73bac1c86f6a1146674c1bdf94680fa23118a4c4b76852cd2387d`.

The private qualification launcher additionally accepts
`--scheduler-trace /qualification-output/fresh-trace-directory` together with
`--benchmark-plan`. Use a fresh directory and a separate supervised diagnostic
run, preserving the application/model/profile settings. The hook traces at
most five rank0 scheduler calls: the first eligible C1 uncached-prefill-containing
round, then two eligible warmed decode rounds each at C1 and C8. It skips plan
warmups. Other ranks are untraced; a captured round can contain multiple target
and MTP forwards, communication and host waits.

Each profiler context starts, stops and exports on the actual scheduler thread
within one call. It records CPU/CUDA activity, preparation/control/staging labels,
process I/O/page-fault snapshots and operator metadata; Python stack recording
is disabled. Receipts record the exported byte count. The existing diagnostic
operator's resource and filesystem containment still applies; the tool does
not impose an additional post-write file-size refusal. Export and stat errors
remain failures and retain any primary work exception.
Profiler start/stop/export changes execution timing: rates from that diagnostic
run must not enter clean performance comparisons. The ordinary launcher path
without this option installs no trace hook.

## Longer inputs and reproducibility

Future runs may prepare `--input-tokens 16384,32768,65536,131072`, with the full
reply allowance still fitting the unchanged 262,144 context. This support does not
mean that long matrix has been executed or qualified. Each input size is a
separate workload; retain every repetition and error, warmup policy, request/plan
hash, actual owner peaks, cache counters, complete outputs and timing scopes.
Do not aggregate different input lengths or cache formats into one speed claim.
Keep image/weights/source/driver/GPU clocks, method, KV format, context, frontend
caps, caches, background load and execution settings fixed within each curve.
A different cache-format profile is a separately loaded curve, never a silent
fallback or a restart per level.


## Complete grouped and reference-tile short curve

One immutable application, source `899934ecc00ac027f290ebd6c230ca47b2821231`,
image `sha256:597c743531a328aeaf5d711d2e86c5a25ff0d1bd652b95275bf45ff4f3073a42`,
completed all seven levels in one loaded container. The SC117 IQ3_S target with
shared-Q8 MTP retained TP8/ETP4/PP1, ISO4 (`rotorquant-iso4-sm70-v1`) KV, context
262,144 with 32 materialized and
reserved full-window States on every rank, MTP depth 4, eager prefill rows 256
and an execution-row envelope of 160. Grouped down `grouped-v1` and reference
Tile128 preparation were enabled; Dense/Concat stayed reference and the native
owner-attention trial was off. Half/Router/EMPTY environment overrides were
absent. This is the recorded combination, not an all-candidate selection.

Each request used 64 uncached input IDs and 128 output IDs. There was one
repetition per level: 87 measured requests. One separate 16-output-token warmup
and two arithmetic MTP/cold-serial smoke requests are excluded from this table.
All measured replies retained zero cached tokens, full token-ID/hash/length
checks and actual worker-boundary owner peaks matching the requested levels.
No profiler or concurrent target job ran during measurement.

| Concurrent requests | Prefill envelope token/s | Aggregate decode envelope token/s | HTTP-wall output token/s | Active-owner peak |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 23.5970 | 3.2606 | 3.0504 | 1 |
| 2 | 23.2190 | 4.7486 | 4.5323 | 2 |
| 4 | 27.0960 | 5.7081 | 5.5776 | 4 |
| 8 | 24.5767 | 7.0107 | 6.9022 | 8 |
| 16 | 22.4508 | 7.8726 | 7.8442 | 16 |
| 24 | 21.3734 | 8.5347 | 8.5293 | 24 |
| 32 | 19.9860 | 8.6503 | 8.6590 | 32 |

The [declared denominators](#denominators-and-actual-overlap) apply: rank0 CUDA
stream envelopes are inclusive, can overlap and contain submission/communication
gaps. They are neither kernel-only nor additive phase times. Decode is aggregate
committed output after each request's first token, without a TP8 multiplier.
HTTP-wall uses all completed output tokens and a separate client interval,
including recording/I/O and transport; it need not equal the decode rate.
The event/owner observer is part of this measured procedure, so these are not
uninstrumented shipping-TPS measurements.

This is a **descriptive single short curve**. It has no matched repeats,
variance estimate, gain/winner selection, output-quality acceptance or
deployment-default release. Full-window allocation and owner peaks do not prove
populated near-full histories, sustained 32-owner occupancy throughout every
interval, or the longer-input matrix. The observed rates remain far below the
4,500 prefill / 450 aggregate decode token/s objectives. The earlier partial
reference below remains historical; different output budgets and execution
settings prevent interpreting these tables as a controlled speedup comparison.

Client, launcher, container and controller exited successfully. The container
retired with PID 0, no OOM or cleanup errors, and unchanged recorded source and
GPU health. Full-precision rows and raw exchanges
are retained in `combined-core-model-899934-v1/live-matrix.json` and
`combined-core-model-899934-v1/actual-model`. Sealed gate:
`combined-core-model-899934-v1/combined-core-full7-actual-gate-v1.json`, SHA256
`5507eaf035687779d093fe716c9d544a2e3fc386d4e0cb02fbda81393400f3ea`.


## Batched global-normalization trial: selection held

A later candidate used source `09b2f0fbdddb3a6b775dbc02a677240f3d271561`, image
`sha256:f1269e70cc4b95a2149555644eea09c77b7912764fd7cf84774a741c03629921`.
Its genuine installed CPU controls passed 191 checks with zero skips; the
separate source-component evidence remains 169 CPU / 15 GPU cases, not model
qualification. The actual model run reused the **exact** `e18f4ca3` matrix bytes
and the `597c7435` baseline profile above: TP8/PP1/ETP4, ISO4, full 262,144
context with 32 full-window States/rank, shared-Q8 MTP4, eager prefill rows 256,
execution rows 160, Grouped + reference Tiles, reference Dense/Concat and no
native-owner trial or Half/Router/EMPTY environment overrides.

All seven protocol, requested-token-count and owner-peak checks passed, with zero cached tokens,
one 64-input/128-output repetition per level, and warmup/smoke excluded as in
the baseline. All eight ranks reported the prepared batched-normalization policy
on decode/prefill/MTP roles 160/256/160 before the client ran. Observed rates
were higher in every cell:

| Requests | Baseline prefill token/s | Candidate prefill token/s | Baseline aggregate decode token/s | Candidate aggregate decode token/s | Exact outputs | Same proposal work |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23.5970 | 29.0620 | 3.2606 | 3.7157 | 1/1 | 1/1 |
| 2 | 23.2190 | 29.1380 | 4.7486 | 5.6331 | 2/2 | 2/2 |
| 4 | 27.0960 | 33.1131 | 5.7081 | 6.8692 | 4/4 | 4/4 |
| 8 | 24.5767 | 33.3702 | 7.0107 | 8.6092 | 6/8 | 6/8 |
| 16 | 22.4508 | 29.2216 | 7.8726 | 9.3672 | 14/16 | 13/16 |
| 24 | 21.3734 | 26.3286 | 8.5347 | 10.0862 | 20/24 | 19/24 |
| 32 | 19.9860 | 25.0366 | 8.6503 | 10.2865 | 27/32 | 27/32 |

**Selection is held.** Only 74/87 replies match full token IDs and choice output;
13 differ. The `rounds`/`drafted`/`accepted` work counters match for 72/87,
with 15 differing records. The
cause is unresolved. Prior baseline variability does not exonerate this new
signal, and identical requests do not imply identical executed work. This
single baseline-then-candidate sequence, without matched repeats or controlled
run-order drift, establishes neither a repeatable gain nor equivalence. The
[inclusive, overlapping envelope definitions](#denominators-and-actual-overlap)
still apply; these are not kernel-only rates. No general output-quality,
populated long-history, deployment-default or target-achievement claim follows.
The 4,500/450 objectives remain unmet.

Both runs retired normally without OOM or cleanup errors and with unchanged
recorded source/health. Full-precision rates, separate HTTP/client latency and
all differences remain in `global-normalization-model-v1/final-matched-comparison-v1.json`,
SHA256 `3c0f3f0d52e3ba89924c3b33efcf6227f934d5d190c9ecd141e046c8c0c8e3ee`.
Actual candidate gate:
`global-normalization-model-v1/global-normalization-full7-actual-gate-v1.json`,
SHA256 `95fb86c8b4627fa8a25ab7d6946115751c06a3d34c3f28d9144207a82d5c39b3`.
The preceding `597c7435` curve remains the historical baseline, not a rewritten
candidate result.


### Private same-input normalization parity witness

Add `--normalization-parity` **before** the existing `--` serve separator in
`tools/v100_owned_qualification.py`. The CLI requires an admitted
`--benchmark-plan` and `--reference-tiles`, and excludes scheduler tracing,
router-row and history capture. This investigation reuses the full original
`e18f4ca3` seven-level/87-request plan as its protocol, rather than a hardcoded
tool restriction. Retain the ordinary profile and immutable `f126` application;
no application-image rebuild is required. The diagnostic adds execution and
fences, while leaving weights, precision, tolerance, public API and defaults
unchanged.

Every rank invokes the actual candidate once, preserving its arguments, and
compares against the unchanged scalar oracle using the same pre-call input
snapshot and shadow statuses seeded from the pre-call state. Raw gathered-input,
FP32 maximum/denominator and status bits are checked. It borrows an unused paid tail of `owner.partial`, the local send/PV slab,
rather than adding a persistent GPU buffer; host obligations
are funded before admission. **One comparison fence per helper invalidates all
event, HTTP and other timings from this run for performance claims.** This is
neither another performance sample nor proof of the cause of earlier differences.

Live CPU metadata is published at `work_observations.normalization_parity`,
including role/layer calls, row extents, mismatches and source/resource identity.
A first mismatch attempts nine bounded NPY snapshots plus `mismatch.json`
before fatal cohort notification; recording failures remain failures. On proven
Engine retirement the hook restores its bindings and writes
`normalization-parity-rank-0/summary.json` through
`normalization-parity-rank-7/summary.json` under the fresh qualification output.
All eight successful summaries must report `closed: true`, `failed: false`,
nonzero comparison coverage for the protocol's required roles, and zero
mismatches; client/retirement outcomes remain required. These artifacts concern one helper,
not general model quality or numerical acceptance.

Frozen tool controls passed 17 focused CPU cases plus Ruff/AST checks, then
222 selected-x86 CPU and four SM70 GPU controls with zero skips. The injected
GPU mismatch produced and verified all nine NPY files; it is fault-control
evidence, not an observed live-model mismatch. Gate:
`normalization-parity-tool-v1/target-v2/normalization-parity-source-CPU222-GPU4-gate-v1.json`,
SHA256 `5d97592fc4093158cc6da6f84f07be1850ecfffadaff8962f789a67e15bdd4ac`.
The first full original pinned-model witness **executed but failed retirement
verification**. All 87 requests across all seven levels passed the HTTP client,
and no mismatch files were found, but **0/8 mandatory retirement summaries**
were produced. Final pre-stop rank0 stats alone report 6,086 helper calls,
247,765 summed per-layer rows and zero mismatches, with `closed: false` and
`failed: false`. This is neither all-eight-rank evidence nor a completed parity
proof; absence of mismatch files does not supply that proof.

The inner launcher exited zero, while the outer/container outcome was exit 1,
PID 0, no OOM and unchanged recorded source/health. This is a retained failure,
not normal witness retirement. All diagnostic timings remain invalid for P3;
selection is held and the earlier 13 output differences remain unresolved.
No normalization numerical failure or cause waiver is inferred from the
missing summaries.

The audited shutdown path gives rank0 a TERM handler in `cuda/http.serve`, while
followers skip that server path. The failed launcher signaled all worker groups;
early stop-on-departure killing and a forced parent-zero outcome can bypass
follower cleanup and summary publication. The source correction now sends TERM
only to rank0 for an operator stop and waits for every rank to exit zero within
the existing 30-second graceful-stop deadline. Owned sessions are then retired
through the existing bounded escalation path. Nonzero rank
exit, a fatal notification or an expired shutdown deadline cannot be forced to
a zero result. Its 26 dependency-free CPU controls passed, including real
two-/eight-rank subprocess fixtures and failure cleanup. The full cohort,
expert-control and qualification modules also passed all 286 pinned-source CPU
checks with zero skips on f126; four-file Ruff and syntax checks passed, with
CUDA uninitialized. These are source controls, separate from the subsequent
live rerun below. The 26 controls are a subset of the 286, not an additional total. Source
gate: `tp-graceful-shutdown-v1/target-v1/tp-graceful-shutdown-source-CPU286-gate-v1.json`,
SHA256 `aa44782e61dfe43aeaf66c75d0d815c241ef41a8abb128edecc4c451df68d116`.
The genuine lifecycle-only image `cd52d838` from commit `1d6c5809` also passed
all 286 installed CPU checks with zero skips (18.840 seconds). Its 785 installed
runtime files match the expected inventory; only the shutdown supervisor differs
from `f126`. Gate: `lifecycle-app-1d6c580-v1/genuine-lifecycle-installed-gate-v1.json`,
SHA256 `02b54981fcb8074d539f117eb3a52d14ca3c6f79264f4702a5a42f9f36b2529d`.
Those installed checks alone did not prove live eight-rank shutdown or complete
the witness.
The lifecycle finding is separate from numerical attribution. Failure gate:
`normalization-parity-model-v1/normalization-parity-live-failure-gate-v1.json`,
SHA256 `b49952e6b672fc2a9d7372d3373cf3f44768e7a985de459f54baa4db0f8e77d2`.
Raw HTTP, pre-stop observations and failed retirement outcomes are preserved.

The separate **successful lifecycle rerun** used source `1d6c5809` and image
`cd52d838` with the same full original e18f plan and profile. All seven levels
and 87 HTTP requests passed. All eight mandatory summaries report
`closed: true`, `failed: false` and zero actual-input, FP32-result or status-bit
mismatches: each rank observed 5,984 calls and 247,863 summed per-layer query
rows, totaling **47,872 calls / 1,982,904 rows** across ranks. Per rank, coverage
was prefill 415 calls / 74,427 rows, MTP 1,477 / 17,892 and decode 4,092 / 155,544;
observed helper row ranges were respectively 51–256, 1–144 and 1–148.
These row totals count layer/helper work, not distinct tokens.

Client, launcher and container exited zero; PID was zero, no OOM occurred, and
recorded source/health remained unchanged. This proves the observed same-input
bit/status comparisons and coordinated eight-rank retirement for this
instrumented workload. It does **not** explain the prior 13 output differences,
prove universal quality or 64K sparse-QSA behavior, or select normalization.
Selection remains held; **all diagnostic P3 timings are invalid**. The failed
b499 run above remains separate historical evidence. Success gate:
`normalization-parity-model-lifecycle-v1/normalization-all8-live-parity-success-gate-v1.json`,
SHA256 `b536f85e3f1bee2a47d0a153e3775fb306491e94b9cc14106cd02bc41b97e4a5`.

## Reference-query packing source candidate

This separate partial A21 increment applies only inside private reference-tile
selection. After the first nonempty job in a chunk, it packs the active quartet's
queries once into the unused opposite half of the already-paid query slab.
Per-row contiguous views serve both statistics and PV. One-row chunks retain
the original zero-copy view; all-EMPTY chunks add no copy. The active half stays
unchanged. Same-stream ordering keeps packed rows valid through their consumers
before the next gather overwrites the slab. Floating-point arithmetic, matrix
shapes/strides/alignment and ordered reductions are unchanged; no new device
buffer is allocated. Default/legacy and native/graph branches retain their old
paths; these controls do not qualify a new whole-forward graph.

The immutable source component passed 40 CPU controls and 15 strict SM70 GPU
cases with zero skips, plus Ruff/syntax checks. Tests compare raw Half bits to
the old per-row reshape and an integer-layout oracle at R1/2/3/5/40/160/256,
both quartets, including all Half encodings, CUDA allocation peaks, row alignment
and consumer/next-gather stream lifetimes. Gate:
`reference-query-pack-v1/target-v1/reference-query-pack-source-CPU40-GPU15-gate-v1.json`,
SHA256 `06724aa6017cfe07ae03dc87612d0a654a9a7783b178eb8303e93098c0a75872`.

This is source-component evidence only: no new application build, full-model
output, P3 gain or default release is qualified. The lifecycle-only successor
bound to commit `1d6c5809` excludes this code; its witness and the earlier
`f126`/`597c` results are not query-packing evidence.

## First SC117 Q4 reference

The first descriptive reference used eight V100-SXM2-32GB GPUs, SC117 IQ3_S
revision `90a3350cd800ec1fe0d1660a38414ad9e985782f`, shared-Q8 MTP depth 4,
TP8/PP1/ETP4, ISO4 KV, configured context 262,144, 32 full-window slots and
eager prefill rows 256. Application source was `7ea3d8ce`, immutable image
`sha256:0a13de3c01a4171139a0fa120b7a859ddc8b8f9c224bd191eacbdbb9df94ff75`,
with benchmark tools at `d68d3573`. Each measured request had 64 uncached input
tokens and 256 output IDs; each level ran once after the separate warmup.

| Concurrent requests | Prefill envelope token/s | Decode envelope token/s | HTTP-wall output token/s | Active-owner peak |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 17.1962 | 2.6097 | 2.5200 | 1 |
| 2 | 17.1927 | 3.1797 | 3.1177 | 2 |
| 4 | 17.7008 | 4.6617 | 4.5986 | 4 |
| 8 | 17.0966 | 5.4382 | 5.4038 | 8 |
| 16 | 15.3982 | 5.3763 | 5.3682 | 16 |

These are the inclusive envelopes defined above, including submission gaps;
they are not kernel-busy rates. There is no repeated-sample uncertainty estimate
or matched optimization claim. Before the matrix, all eight ranks reported 32
materialized full-window States and the arithmetic MTP/cold-serial smoke matched.
That allocation does not prove 32 active requests or populated full histories.

The operator intentionally stopped during level 24 to prioritize shared-path
profiling. Level 24 is cancelled/unmeasured and 32 is unrun; the full-sweep result
remains false. Normal launcher retirement completed, GPU workers exited, and
health/source checks matched. Raw cancellation and incomplete-matrix errors are
retained. Private provenance is
`tp8-etp4-standard-iso4-c32-v1/intentional-partial-baseline-gate-v1.json`, SHA256
`e557283e1862927794c5e0e96cd2b4972c906075d13fb224ed71fc84e47cb4ca`.
Q8 and longer-input performance, broader output quality and full-window activity
remain unqualified. The current objectives are 4,500 prefill and 450 aggregate
decode token/s, with shared implementation work preceding topology tuning;
these objectives are not measured results or hardware ceilings.

## Matched owner-attention pilot

A subsequent paired pilot kept the same SC117 target/MTP weights, eight V100s,
ISO4, configured context 262,144, 32 resident slots, MTP depth 4, eager rows 256
and unchanged application image `0a13de3c`. Both arms prepared the native owner
workspaces and verified the same complete input closure. Only the explicit
startup trial selected reference versus native owner attention; graphs remained
uncaptured and Dense/Concat stayed reference. No profiler or build overlapped
the measured requests.

Each request used 64 uncached input tokens and 128 output IDs. Prompt IDs matched
across fresh service/run identities; each arm ran one repetition after its
separate warmup. Native ran before the corrected reference. The inclusive
envelope definitions above still apply:

| Active requests | Reference prefill token/s | Native prefill token/s | Reference decode token/s | Native decode token/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 17.4576 | 64.6492 | 2.8001 | 3.4745 |
| 8 | 17.0336 | 56.0141 | 5.5198 | 10.3801 |

Actual backend owner peaks were 1 and 8 in both arms. All 128 output IDs matched
in the single-request case. Every eight-request continuation differed between
arms, with first differences at output indices 6–94. That row therefore includes
changed token, routing and draft trajectories; it is not an isolated kernel
speedup. Even C1 used different speculative work: native ran 45 rounds with
109 drafted/82 accepted tokens, versus reference's 43 rounds and 101/84.
Neither row isolates kernel-only cost. The samples establish neither
repeatability nor model-quality or
numerical acceptance. Native attention remains an unqualified trial.

An initial reference attempt stopped at an incorrect supervisor stats-field
lookup before measuring any row. Its failure is retained separately and excluded
from this pair. Both measured arms retired normally with matching source/health
checks. Private provenance:
`native-owner-clean-pilot-v1/matched-owner-pilot-gate-v1.json`, SHA256
`d8751c4a4dd76a3e05d3ff0da06ce2b91f1138913fd751b369520826e492fa15`.


## Balanced ordered Half partial pilot

Four clean runs used A/B/B/A order on image `60dcb0e2` (committed `69c3ce2c`),
with TP8/ETP4/PP1, ISO4, cap32, context262144, MTP4, row budget32 and prefill256.
Dense/Concat and owner attention retained their reference selections; Router
remained `inherit`. A used inherited Half partials; B alone set
`TENSORFOLD_PRIVATE_SM70_MOE_PARTIAL=native-half-v1`. All eight workers attested
those selections. Original measured request bodies, 64 input/128 output tokens,
one repetition per run and the untimed smoke/warmup procedures were unchanged.
No build, profiler or other target job ran during timing.

Prefill/decode are the established inclusive CUDA-stream envelopes; HTTP is
separate wall throughput. All nine measured requests per run were uncached,
with coherent active-owner peaks1/8 and clean lifecycle/health/source checks.

| Run | Concurrency | Prefill tok/s | Decode tok/s | HTTP tok/s |
| --- | ---: | ---: | ---: | ---: |
| A1 inherit | 1 | 17.179579 | 3.005478 | 2.775337 |
| A1 inherit | 8 | 16.878539 | 5.557588 | 5.477276 |
| B1 native Half | 1 | 15.987849 | 3.067251 | 2.808309 |
| B1 native Half | 8 | 17.881589 | 5.692447 | 5.614675 |
| B2 native Half | 1 | 17.303511 | 3.150825 | 2.898253 |
| B2 native Half | 8 | 16.536820 | 5.613603 | 5.537426 |
| A2 inherit | 1 | 16.054689 | 3.029498 | 2.779220 |
| A2 inherit | 8 | 17.419297 | 5.596376 | 5.519570 |

Using full-precision records, observed mean decode was 3.03396% higher at C1
and 1.36351% higher at C8 for native Half; mean HTTP was 2.73660%/1.41181%
higher. Prefill means were 0.17179%/0.35154% higher, with wide overlapping
ranges. Two runs per mode do not establish formal confidence or equivalence.
C1 IDs and proposal work matched across all four runs; C8 trajectories and
proposal work varied, including between the two controls, and remain retained.
No outlier was discarded. This partial pilot does not qualify the full
1/2/4/8/16/24/32 matrix, long context, new output quality or production/default
promotion. Private gate: `half-clean-pilot-v1/balanced-Half-pilot-final-gate-v1.json`,
SHA256 `b723c656d5a13d9bb83eefade944e1b07d23b7bf1c88efbf45dff5dffba48d55`.

## Preserved raw-completion reproducibility diagnostic

Use the existing client without a private benchmark observer to investigate
repeatability before attributing a model difference to an operator candidate:

```bash
python tools/bench_concurrent.py http://127.0.0.1:8000 SERVED_MODEL \
  --repro-input preserved-repro-input.json --output fresh-repro-result.json \
  --timeout 1800 --response-mib 16 --sse-line-kib 64
```

DATA uses schema `tensorfold.raw-completion-repro-input-v1`, the recorded
`model_configuration` from `/stats.model`, and eight ordered `cases`. Each case
contains `index`, the original `body`, its `body_sha256` (SHA256 of JSON encoded
with sorted keys and separators `(',', ':')`), and the original normalized
`reference` completion including message, usage, token IDs and independent
ID hash. The source DATA file hash and client source hash are recorded.
The selected diagnostic preserves 64 prompt IDs, 128 output tokens, greedy
sampling, streaming and all original fields; only serial variants change
`draft` to false. It requires the recorded ISO4/MTP4/full262144/cap32,
TP8/PP1/ETP4 native-owned service and actual eight-rank full-window receipts.
It never changes model settings or starts/reloads the service.

One loaded instance runs an eight-request MTP repeat, prompts 2 and 7 alone
with MTP and serial decoding, then the eight-request serial batch: 20 requests.
If the first batch differs from its saved baseline, one further eight-request
MTP repeat makes 28 requests. The start barrier describes offered batches;
this diagnostic does not certify strict simultaneous backend occupancy.
Every response must retain complete ID/hash/finish evidence, actual mode and
proposal counters, and zero cached tokens in both native and usage metadata.
Compiler/frontend caches are independent of prompt-cache reuse. Full output
comparisons are exact; there is no tolerance or performance result. A completed
procedure can exit zero with differences (`all_exact: false`); exchange,
identity, cache or protocol failures remain errors with partial results and
raw exchanges retained. Initial, per-stage and final stats identify the
service and lifecycle. `--timeout` retains the existing client socket/read
semantics; it does not promise a hard whole-suite wall deadline. Output and
its `.raw` directory must both be fresh.

Prefill router FP32 GEMM can use a different batch shape from decode's
independent-row router arithmetic. History-slot assignment also selects page
and indexer owners, with ordered per-owner attention reductions. These are
investigation axes, not established causes of any observed divergence;
request bodies and actual state mapping must resolve attribution.

### Private router-row capture

The supervised private qualification tool can install a default-off diagnostic
with `--router-row-input capture.json`. This is separate from its history and
benchmark modes. DATA schema `tensorfold.router-row-input-v1` contains a fresh
32-character lowercase hexadecimal `run_identity` and two ordered
`prompt_token_ids_sha256` strings for original requests 2 and 7. These hashes
use compact JSON integer arrays, matching the private history helper.

The repro client takes the same run identity and the qualification directory's
fixed marker path:

```bash
python tools/bench_concurrent.py BASE MODEL --repro-input original-repro.json \
  --output fresh-repro.json --router-row-stage QUALIFICATION_DIR/router-row-stage.json \
  --router-row-run-id RUN_ID --timeout 1800
```

The marker is atomically published before each offered batch; request bodies
remain unchanged. Only `baseline-c8-mtp`, `alone-2-mtp` and `alone-7-mtp` can
capture their selected prompt tails. Conditional repeated C8 and serial phases
do not consume the four snapshots. Worker metadata records actual frame size,
row offset, stable wire, slot/generations, sampling settings, first attention
owner mapping and the marker's path/identity/hash/stat. Actual submitted State
segments must match. Successful worker-round records do not replace client
outputs or distributed operation ACKs.

Rank 0 saves real layer-0 router weights once and at most four actual Half
input rows with their already computed FP32 router logits. Contiguous views
are copied directly to CPU; no GPU clone or device scratch is introduced.
CPU tensor payload, wrappers, NPY/file/hash workspace and bounded metadata are
funded through the existing private host obligation before model admission.
NPY files are created without overwriting, flushed/fsynced, made read-only and
recorded with shape/dtype/stat/SHA. Failed rounds never publish successful
capture records. This observer adds blocking copies and file I/O: its runs are
numerical diagnostics, never performance measurements or numerical acceptance.
