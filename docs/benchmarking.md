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

## Governing performance targets

[The governing design, Section 5](../ProviderOne%20Implementation%20Docs/ProviderOne_V100_Adaptive_Inference_Design.md#verified-hardware-targets-and-milestone-policy)
binds the following qualification objectives on the eight-V100 server:

| Workload | Required objective | Profile binding |
| --- | --- | --- |
| One active request | Greater than 250 committed decode tokens/s | Q8 KV, native 262,144-row context |
| 24 simultaneously active requests | Greater than 600 aggregate committed decode tokens/s | Q8 KV, native 262,144-row context per request |
| Prefill | Greater than 3,500 processed prompt tokens/s | Bind reference workload, concurrency and cache conditions before comparing |

These are unachieved qualification gates, not measured rates or capacity caps.
Preserve the C1/2/4/8/16/24 milestone comparisons, the actual integer capacity
sweep, one active request with idle residents, and near-full-history/full-active
gates. Allocated windows and queued requests do not prove these workloads.
The earlier interim 4,500/450 goals remain identified in the historical records
below; their unspecified workload bindings do not equate short Q4 curves with
the governing Q8/native-context objectives.

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

## Private grouped gate/up source component

An explicit private typed Plan selector requests two or four rows only in the
already-grouped SC117 TP8/ETP4 path. The gate/up incumbent remains the default;
there is no public serving flag or automatic crossover. Both candidates reuse
existing expert-group metadata and activation/output scratch, with no new bank
or application device buffer. A separate bounded kernel preserves foreign,
shared and invalid activation Half-NaN values; its launch and traffic must be
included in complete-chain measurements.

Pinned source composition passed **388 CPU controls and 49 strict paired SM70
GPU cases, with zero skips**, plus Ruff and syntax checks. The GPU cases used
original-format synthetic fixtures on one SM70 device, comparing raw local-ID,
activation and slot-output bits across admitted reader combinations, ragged
rows, changed-route replay and finite Half extremes through ±65504. The frozen
grouped incumbent ELF `71327e0d` was loaded before candidate compilation; the
candidate ELF is `e88563f0`. Source and health checks matched, with normal
zero-status retirement and no OOM.

Static resource reports for both variants show `REG56/STACK64/LOCAL0`; SASS
contains `LDL` and `STL` instructions. These reports do not establish zero stack
traffic, spill-free execution, runtime occupancy or a performance gain. The
source results do not qualify a genuine candidate application, an SC117
full-model run, resident capacity, affected-region P3 or a new default. Gate:
`grouped-gate-up-assessment-v1/grouped-gate-up-source-CPU388-GPU49-gate-v1.json`,
SHA256 `e5005541cad0ebbd3fd160064fc7c79942dfc9e3a8768447f9a8f97959ab559a`.

### Private qualification selector bridge

In a new immutable application containing the bridge, pass
`--grouped-gate-up-rows 2` or `--grouped-gate-up-rows 4` to
`tools/v100_owned_qualification.py` **before** its existing `--` separator.
This private qualifier option requires the already-grouped SC117 TP8/ETP4
preparation above; it does not enable grouped down, add a public serve or
environment flag, or change the incumbent default.

The typed capability passes through CLI, family factory and Engine before
construction, joins early eight-rank mode agreement, and binds the prepared
Plan. Each rank's `grouped_gate_up` receipt marker checks requested versus
actual rows before publication/readiness. An application missing the private
CLI field or Engine parameter is refused before construction; use a newly
built application rather than an installed-code patch.

This separate bridge source composition passed **459 pinned CPU controls with
zero skips**, plus seven-file Ruff/syntax checks, with CUDA uninitialized.
It does not qualify a new installed application, live eight-rank activation,
SC117 model quality/capacity, affected-region P3 or a new default. The earlier
388 CPU / 49 GPU component checkpoint remains separate evidence. Source gate:
`grouped-gate-up-assessment-v1/private-selector-bridge-v1/target-v1/private-gup-bridge-source-CPU459-gate-v1.json`,
SHA256 `577bea0cc87facf9a359334363aae7987cb145c15a85ff61915f1d313f7b8d2c`.

### Prepared feature retirement receipts

Add `--prepared-feature-receipts` to the private qualifier **before** `--`, with
`--benchmark-plan`. It excludes normalization parity, scheduler tracing, router
capture and history capture. Select Graph arithmetic and/or grouped gate/up
through their existing controls; this receipt mode changes no arithmetic,
backend or kernel selection.

Before listener publication, each rank writes
`prepared-features-rank-N/prepared.json` and a startup marker binding source IDs,
requested/actual gate/up rows, preparation charges, budget, before/after memory
and pool accounting. The fixed JSON bound is **36,864 bytes per receipt**; the
pinned policy charges **86,389,272 host bytes across eight ranks** for the observer.
Graph device preparation retains its separate explicit admission budget.
After original Engine close proves normal, nonaborted retirement, the observer
restores the close attribute and writes `summary.json`. Every configured Graph
role must be closed, nonfailed, with positive statistics/probability/PV counters
and unchanged preparation/source/budget facts. Graph-off/gate-up-on records
explicit null Graph roles with the typed row marker. The observer adds no HOT
hooks, CUDA probes or additional CUDA synchronization/profiler work.

Runtime image `30fd5e0b`, source `66b86f7e`, passed **690 installed CPU controls
across 15 modules and 84 GPU cases across three modules, zero skips**, with
explicit `/canonical` pytest root and module-identity guards. The installed
runtime inventory remained exact; only the owned qualifier/test sources were
overlaid. Final gate:
`grouped-gate-up-assessment-v1/prepared-feature-receipts-v1/combined-installed-final-CPU690-GPU84-gate-v1.json`,
SHA256 `00f7a3c17a980b93375de4e0fb246784c569867cd550673f9e1ca83fac27c903`.
The separate tool-source checkpoint passed 280 CPU controls; these are separate
verification scopes, not an added total. Its gate is
`grouped-gate-up-assessment-v1/prepared-feature-receipts-v1/prepared-feature-receipts-source-CPU280-gate-v1.json`,
SHA256 `c9152b5164ef7887d119271071191a203c911025f69dba8cd7604880bd5ef968`.

The older 658-CPU `/dev`-root coverage is historical and superseded by the
correct-root 690/84 gate. The earlier 200-pass/80-failure attempt remains
preserved: collapsed module identities leaked a module-local fixture into
other modules. Correcting pytest root and guarding collection restored fixture
scope; runtime behavior and test assertions were unchanged. These gates do not
prove live eight-rank summaries or deployment quality by themselves. The later
[combined short curve](#combined-graph-islands-and-grouped-gateup-short-curve) records
actual all-role trial charges and retirement separately; full quality, matched
resource/P3 selection and a new default remain pending.

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
earlier interim 4,500 prefill / 450 aggregate decode token/s objectives. The earlier partial
reference below remains historical; different output budgets and execution
settings prevent interpreting these tables as a controlled speedup comparison.

Client, launcher, container and controller exited successfully. The container
retired with PID 0, no OOM or cleanup errors, and unchanged recorded source and
GPU health. Full-precision rows and raw exchanges
are retained in `combined-core-model-899934-v1/live-matrix.json` and
`combined-core-model-899934-v1/actual-model`. Sealed gate:
`combined-core-model-899934-v1/combined-core-full7-actual-gate-v1.json`, SHA256
`5507eaf035687779d093fe716c9d544a2e3fc386d4e0cb02fbda81393400f3ea`.


## Combined Graph islands and grouped gate/up short curve

Runtime image `30fd5e0b`, source `66b86f7e`, with qualifier/tools `0b31aed5`,
completed one full seven-level trial. The SC117 TP8/PP1/ETP4 profile retained
262,144 context, 32 materialized full-window States per rank, ISO4 KV, shared-Q8
MTP depth 4, prefill rows 256 and execution rows 160. Grouped down and gate/up
rows 2, reference Tile128 and the three-island reference ATen Graph policy were
enabled. Dense/Concat remained reference; native-owner attention was off and
Half/Router/EMPTY overrides were absent. This is the recorded private profile,
not a deployment-default selection or NativeFP32 qualification.

All **87 measured requests** completed with 64 uncached input IDs, 128 output
IDs and zero cached tokens. Each requested active-owner peak from 1 through 32
was observed. There was **one repetition**, without a matched control. Plan
`d70b92e8` preserves the original measured request bodies but changes the excluded
warmup to 4,096 input / 16 output IDs; that warmup took **219.129 seconds**.
The two separate smoke requests are also excluded.

| Concurrent requests | Prefill envelope token/s | Aggregate decode envelope token/s |
| ---: | ---: | ---: |
| 1 | 27.7748876 | 3.7499138 |
| 2 | 28.2364306 | 5.7710176 |
| 4 | 33.7870883 | 6.8524109 |
| 8 | 33.5397453 | 8.8983587 |
| 16 | 29.7983826 | 9.9919026 |
| 24 | 27.0128457 | 10.7278308 |
| 32 | 26.0115412 | 10.8023203 |

The [declared denominators](#denominators-and-actual-overlap) apply: rank0's
inclusive CUDA-stream envelopes include submission/communication gaps and carry
no TP8 multiplier. These instrumented short-workload observations have no
matched repetitions, uncertainty estimate or accepted speedup. They remain far
below the earlier interim 4,500 prefill / 450 aggregate decode token/s objectives; allocated
full windows do not establish populated near-full-history quality or sustained
occupancy throughout every interval.

All eight ranks published final prepared-feature summaries after normal close.
Canonical-object digests and unchanged source, preparation and budget facts
matched the startup markers; every configured role was closed and nonfailed.
For **each rank and each of statistics/probability/PV**, post-readiness replay
increases were decode **82,896**, prefill **625,856** and MTP **9,131**.
Counter deltas cover the post-readiness window, including excluded warmup/smokes,
rather than only the 87 measured requests; they are Graph calls, not distinct tokens.
Gate/up rows 2 was bound by the requested/actual Plan marker; no GPU gate/up call
count is inferred. The three Graph capture charges totaled **113,246,208 bytes
per rank / 905,969,664 across eight ranks**, under the explicit **352,321,536-byte
per-rank trial allowance**. These are gross capture-preparation charges, not a
matched net VRAM increase or the whole Engine footprint. The allowance is a
provisional test-profile setting, not a production default. Per-role and whole
preparation snapshots/pool accounting remain in the receipts.

**Historical output differences remain unresolved.** Prompt SHA values align,
but warmups do not: the historical `597c`/`f126` curves used 64-input warmup,
versus 4,096 here. Exact token-ID plus choice equality held for **73/87** requests
against `597c` and **69/87** against `f126`; equality of `(rounds, drafted,
accepted)` held for **72/87** and **68/87**, respectively. These are unmatched
historical diagnostics, not a causal attribution, quality acceptance or gain.
Baseline variability and strict component passes do not waive the differences.
Comparison: `combined-graph-gup-model-v1/reviewer-actual-full7-all8-v1.json`,
SHA256 `253aa5c24a1ccaba014ed808de926612b2142b9231d9bf0338ac457712d18e71`.

Client, launcher and container exited zero; the container PID was zero, with no OOM, unchanged
source/health and retired GPU contexts. This proves the observed request,
owner-overlap and prepared-feature retirement protocol for this trial. Full
quality, matched resource/P3 comparisons, NativeFP32 work and default selection
remain pending; older failures and component gates retain their own scope.
Actual gate: `combined-graph-gup-model-v1/combined-graph-gup-full7-all8-actual-gate-v1.json`,
SHA256 `ac084bc1373602510cc432c484379bd11d0b41e071446c92ce63e5811eaf0793`.

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
The earlier interim 4,500/450 objectives remain unmet; this curve also does not
qualify the separately bound [governing workloads](#governing-performance-targets).

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

## Private reference ATen arithmetic islands: source component

This default-off partial A21 / Design §10,15 candidate requires the existing
private reference-tile capability and an explicit positive per-rank graph budget
across prepared roles. Three prepared graphs retain the original ATen operation, cast and
reduction order: statistics; normalized QK to Half probabilities; PV plus ordered
result. Only full 128-entry tiles replay; ragged tails retain eager shapes.
Native K decode precedes probability replay, then native V decode overwrites the
same Half slab before PV replay. There is no new V slab, bank/history mirror or
whole-forward serving capture. New persistent arithmetic tensors and graph pools
are additional memory, measured before admission. A nonblocking operation lease
covers staging through copies into caller-paid statistics/PV destinations;
retirement fences graphs before releasing borrowed views.

The exact source component passed **330 pinned CPU controls and 35 SM70 GPU
cases, zero skips** (18 new cases including one-byte budget refusal, 17 existing
reference-tile controls; GPU 1.948 s). Synthetic Q8/ISO4 cases compare raw FP32
statistics/PV and status bits to unchanged eager arithmetic, including full/tail
lengths, changed inputs, stale generations, empty/nonfinite inputs and signed
zero. Each graph replayed 94 times, and owners retired closed. These counts are
this milestone only, not a sum with earlier results. Gate:
`reference-arithmetic-islands-v1/reference-arithmetic-source-CPU330-GPU35-gate-v1.json`,
SHA256 `4ff3225e2c6ee72fc397bf599b7a8f34439d2f1c051e0929d2606d8a72333dc2`.

The cold one-byte refusal measured **117,440,512 B** charged preparation; the
warmed nominal fixture measured **27,262,976 B**, with **34,080 B** retained tensor
payload and **6,275,072 B** private-pool spare. These are distinct one-owner
fixture observations, not a complete three-role/eight-rank deployment budget.
The nominal allowance, **33,624,162,304 B**, came from actual test-device free
physical memory and is **test only**, not a production default. The earlier
64 MiB fixture refused before numerical comparisons; its setup refusal and
separate observer publication failure remain in
`reference-arithmetic-islands-v1/target-v2/GPU-actual/console-v1.log` and
`dense-result.json`. That attempt supports no numerical conclusion.

No new application image, full-model quality/capacity result, P3 gain or default
promotion is qualified. All-role application budgeting and model/performance
selection remain open; historical installed-image curves are separate evidence.

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
remain unqualified. This historical run used interim objectives of 4,500 prefill
and 450 aggregate decode token/s. Current qualification uses the separately
bound [governing workloads](#governing-performance-targets); neither set of
objectives is a measured result or hardware ceiling.

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

## Private reference-first teacher diagnostics

`tools/v100_owned_qualification.py --teacher-plan PLAN.json` is a separate,
default-off diagnostic over the existing owned Engine and ordinary completions
route. It does not enable a public serving mode or run the old 714-step quality
worker. The initial four-file implementation passed 284 pinned x86 CPU controls
with zero skips. Actual short reference capture/replay and full-vector control
results are recorded below; broader residency, quality and performance gates
remain open.

The first region is SC117 TP8/ETP4/PP1, reference SM70 profile, explicit
`--draft-method none`, no loaded MTP, `--no-graphs` and fixed parallelism of at
least nine. Admit one exact cold greedy text request, without constraints,
penalties, media, retention, background work or other diagnostic procedures.
MTP4 and other regions reject explicitly. GET readiness is allowed; do not run
an unrelated arithmetic POST before the single planned request.

The optional private `--teacher-client CLIENT.json` entry now drives that one
request and requests normal leader-first shutdown only after validated client
completion. Failure uses the existing rank supervisor; client work is cancelled
and drained before authority publication. The shared closed
`tensorfold.owned-teacher-client-v1` schema binds the plan SHA, literal loopback
base/model, actual expected service profile, readiness/whole/socket deadlines
and response bound. This is not an arbitrary warmup or benchmark client.
The initial client/host integration passed 358 x86 unit controls. Its first
actual reference capture is recorded below; those unit counts alone are not
model evidence.

The [closed host operation](../deploy/v100/source-foundation.md#closed-owned-teacher-model-operation)
is the maintained model launch boundary. Existing CPU/CUDA pytest specs remain
component suites; do not substitute model argv or infer a model pass from a
JUnit count. Each model operation requires actual DATA and assets/cache
binding, physical C24 admission, its client request and complete all-rank/host
retirement. The bounded completed controls below do not replace those gates
for other workloads or profiles.

```bash
python tools/v100_owned_qualification.py --gpus "$TF_RANK_GPUS" \
  --out "$TF_TEACHER_FRESH_OUTPUT" --teacher-plan "$TF_TEACHER_PLAN" \
  -- "${serve_args[@]}"
```

Bind `serve_args` to the actual immutable model, frontend, numerical profile,
context, method and complete resource envelope before launch. The plan is the
closed `tensorfold.owned-teacher-plan-v1` schema defined by
`owned_teacher_plan`: capture/replay mode, actual full vocabulary and context,
frozen rendered input IDs, maximum frame count, declared case/run/model/frontend/
profile identities, reference descriptor and explicit `read_timeout_s`. Derive
that operational I/O budget from the outer preparation deadline. Chunk checks
are cooperative; blocking storage calls and whole-job cancellation remain with
the outer owner. Model/frontend/profile SHA declarations alone do not prove the
actual loaded pairing.

Capture copies raw Float32 vocabulary shards before the original sampler runs
once. It borrows an idle, disjoint, already-paid target-logit tail; collective
and CPU-copy completion precede reuse. This adds no project CUDA data buffer,
but native-library/allocator residency still requires actual measurement. Host
parse/tree/frame/writer obligations use the existing selected resource policy;
full vector input/output is bounded and streamed rather than materialized again.

After the existing launcher returns successfully, the parent verifies eight
matching positive worker-retirement summaries, eight normal startup bindings,
actual State/profile/source identities and rank0 vector hashes before publishing
`teacher-authority.json`. A published file alone is insufficient: require the
owning parent/CLI **and outer host run** to complete successfully before using
it. Publication failures can leave a partial or complete file. Replay accepts
a sealed successful capture authority, not a single rank's local summary, and
verifies all referenced receipts/vectors before deriving teacher IDs.

Replay advances the same frozen teacher prefixes after measuring candidate
logits. Its returned text is forced reference text, not candidate free-generation
quality. Transfer/fence/writer overhead invalidates P3 timings. Preserve separate
capture/replay cohorts and actual output-quality, state, sampling and resource
gates; do not borrow the old 8192-context/714-step profile or its margins.

### Private model-operation time funding

The qualification client's socket interval is explicit DATA and may exceed the
initial 300-second pilot limit. It remains positive and finite, no greater than
the whole client interval; readiness is at most 3,600 seconds and whole time at
most 7,500. Existing short-control values remain supported. This is private
qualification timing support, not a service SLA or a latency result.

After parent content verification, fixed argv preparation and synchronous phase
publication, the driver checks the actual clock immediately before entering the
owned model operation. Remaining work time must fund **whole client interval +
one configured in-flight socket interval**. The 7,800-second outer envelope and
90-second retirement reservation remain unchanged. Parent hashing, imports and
phase IO consume that envelope; a numerically valid client declaration may
therefore be refused before workers start if its combined allowance no longer
fits. No timeout is silently widened after a failed run.

The existing `ReadControl` bounds body work cooperatively, and an in-flight
read can outlast that cooperative deadline by one socket interval. DNS/status/
header acquisition uses socket inactivity limits; this admission check does
not manufacture a whole-wall acquisition guarantee. Existing outer containment
and response ownership remain required. There is no new launcher, timeout
fallback profile or model numerical policy.

For the preselected 65,472-input reference probe, the planned private values are
600 seconds readiness, 3,900 whole and 3,000 socket: 6,900 seconds of combined
client-plus-tail funding. The old 3,900/300 short control needs 4,200. These are
predeclared operational allowances; the observed 52-token control supplies no
64K completion guarantee. The updated driver passed 76 x86 controls with no
skips, including refusal after phase publication consumes the remaining
allowance. The first long model operation timed out as recorded below.

### Reproducing independent quality inputs

`tools/v100_quality_data.py` preserves the original 27 C27 bodies, adds five
independently labelled near-full histories, three fixed 64K multi-key/order
cases and the seven short cases. The dependent-history short case stays in
its original contract rather than an invented standalone request. Its 42 x86
CPU controls passed with no skips. Genuine current identity capture and DATA
generation subsequently passed as recorded below. This is not inference or a
model-quality pass. The archived original
generator is provenance input, not executable code required by this producer.

Capture a distinct identity in the intended owned image and normal frontend
namespace before generation. The explicit first selector has only four
arguments; its output is a fresh identity **file** with an existing owned parent:

```bash
python -m tools.v100_quality_data --capture-identity \
  --input-identity "$TF_QUALITY_BASE_IDENTITY" \
  --input-identity-sha "$TF_QUALITY_BASE_IDENTITY_SHA" \
  --frontend "$TF_MODEL_FRONTEND_DIR" --output "$TF_QUALITY_IDENTITY_FILE"
```

This preserves the pinned base target/MTP declarations as declared history; it
does not read, load or physically qualify checkpoint weights. It records the
actual five normal frontend assets, template, native tokenizer, dependency
versions, machine and loaded Python closure after the same normal
frontend/publisher initialization used by generation. No historical module
list is forced into the process. Publication refuses existing output and
fsyncs the file and directory. Use the new identity only after its CLI exits
successfully; a complete-looking file can survive a later failure.

Bind `TF_INPUT_IDENTITY` and its SHA below to that new file. Normal mode keeps
its existing six required arguments and writes a fresh corpus **directory**:

```bash
python -m tools.v100_quality_data \
  --corpus "$TF_C27_CORPUS" --short-contract "$TF_SHORT_CONTRACT" \
  --input-identity "$TF_INPUT_IDENTITY" --input-identity-sha "$TF_INPUT_IDENTITY_SHA" \
  --frontend "$TF_MODEL_FRONTEND_DIR" --output "$TF_QUALITY_FRESH_OUTPUT"
```

Supply the immutable original corpus/short inputs and a distinct, matching
current frontend identity. Asset bytes, actual loaded source closure and its
aggregate hash, native tokenizer binary, dependency versions and machine must
match before publication and sealing. A stale historical identity is refused;
do not rewrite it or force-import its old modules to manufacture a match.
Use a fresh output directory. Before creating it, the producer reopens each
original C27 request through its pinned hash, renders/encodes the full template,
and requires the original 262,080-token count and compact-JSON token-ID hash.
Only one case's token list is retained at a time. A mismatch refuses publication;
the producer does not silently rewrite old token metadata or gold. Frontend
assets and the loaded closure are checked again through generation and sealing.
Historical `5fadb028` DATA and its 82 case files remain unchanged.
Tokenization equivalence, sparse-QSA execution, teacher
vectors, maximum live capacity, media gold and full-model quality keep their
separate actual-target gates; no older codec-profile margins are inherited.


### Actual current frontend DATA reproduction

The maintained producer executed in two fresh x86 CPU processes on application
`d736ab78`, using immutable operator source `47bdb515` plus the exact two-file
hold whose producer bytes are committed at `04bc3861`. The first captured the
current frontend identity; after its positive CLI/observer/container retirement,
an independent read-only copy and its actual hash bound the second generation
process. The fixed audit delegate invokes only the canonical producer. It
records actual NoTorch and installed TensorFold origin without importing or
replacing either module. Both processes exited successfully, with no OOM,
Torch absent before/after, and the expected installed raw/resolved origin.

Current identity SHA256 is
`95d713b14069390e2129ef871b7c3e0469f177d9237dcae0db723023cb861d9c`;
current manifest SHA256 is
`b94ff3c3c99188e77752daa2eb5c664c8d86c4b05ba199f5ef7f8a8109667b1d`.
Its 38 TensorFold modules were actually loaded by normal frontend/publisher
initialization; no historical module list was forced. Both processes matched
the native tokenizer, dependency/platform, source-closure and five-file frontend
bindings. Target/MTP checkpoint fields remain declared-only base authority;
this CPU operation mounted or loaded no weights.

All 27 preserved long requests matched 262,080 whole-template tokens and their
original compact token-ID hashes before publication. The fresh output has 85
files: 82 request/oracle payloads plus three controls. All 82 payloads are
byte-equal to their historical counterparts, and all 85 historical files remain
unchanged. The catalogue retains 24 Q8 cases, 32 ISO4 cases, three 64K cases and
seven short cases. The six static short bodies/gold are preserved, but their
ID/count export is not provided by this manifest; no new alignment result for
all six is inferred. The dependent-history case still requires its actual
prior assistant response. Codec labels do not establish codec execution.

All source, input, frontend and output hashes were stable after execution; the
new identity and corpus have independent read-only copies. The actual gate is
`217d8ef8ac33c102b9547c99ee37bcd22e922ccacd2a5ea1e99f7a47832ab8ca`,
with independent actual review
`4a6fd359dc1c9bc45df4af41d6f7c315851b307a28c816134ff7fbfbbc740f7c`.
This closes the current CPU DATA reproduction/alignment gate. It does not
qualify current MTP availability, sparse-QSA execution, model answers, full
activity/capacity, media, physical worst-case resources or performance.

### First actual reference teacher capture

The first closed operation used committed operator `30dd2f19`, genuine App
`d736ab78` and isolated verification image `efb4cb8f`. All eight V100 ranks
materialized 24 Q8 (`int8`) States with a 262,144-row logical window and
262,145-row physical capacity/limit, including the anchor. The method was NONE,
with no MTP banks, Torch-reference attention, no graphs and 256-row prefill
pieces. This is one active request with idle full-window residents, not a
24-concurrent-request or populated-full-history qualification.

Exactly one cold `/v1/completions` POST returned HTTP200 and text `372`, equal
to the frozen first arithmetic oracle. It used 52 input IDs, zero cached tokens
and a maximum 64-frame budget; valid EOS ended after four actual frames/IDs:
`[18,22,17,248046]`. Four raw finite Float32 vectors each cover all 248,320
vocabulary entries. Their raw argmax agrees with each original selected ID.
This was capture through the original sampler, not forced-prefix replay.

The genuine current frontend key is `61ea346c`; normal CPU preparation and an
independent read-only cache-hit proof preserved all four semantic files,
template and 52 IDs. The earlier copied `77634fe5` cache metadata referred to
old file identities and remains preserved as an unlaunched failed binding.
No ready/key rewrite or extra model-weight copy was used.

All eight startup/profile and closed/nonfailed summaries, eight child VMA-FD
records and four full-vector hashes were verified. The parent remained CUDA-
uninitialized; nine held asset checks completed, all rank/CLI/typed operation
and outer Host gates returned0, and retirement left no GPU contexts, OOM,
cleanup errors or source/health drift. HTTP instance identity matches rank0;
followers correctly have their own process instances. Nine c10d socket
EAFNOSUPPORT97 warnings were retained and were nonfatal in this run; no direct
transport-family claim or warning suppression follows.

The reference authority SHA256 is
`4a7197c49e803ddf371c8e9d571d12feb88c605e2853dcb1dd4ef1773b7e97f4`;
the complete actual receipt is
`e22054dbdb11ff24699a628065ff95edfc292f48c44c8226a8d9f539f9541c84`.
This makes bounded reference DATA available. Broader replay/task/media and MTP
quality, worst-case resources, NativeRN/C06 adoption and matched per-count
performance keep separate gates. Copying, synchronization and recording make
all timings diagnostic; no token/s result is published from this operation.


### Actual matched reference replay

One matched replay subsequently passed using clean operator `3e582361` and
its committed teacher helper, excluding the newer offline comparator. The
application `d736ab78`, verification image `efb4cb8f`, current frontend
`61ea346c`, installed math sources and actual four native-module hash sets
match the capture. The same Q8/NONE/reference/no-graphs profile materialized
24 logical 262,144-row windows per rank with the extra physical anchor.

One cold 52-ID POST returned HTTP200 with four frozen IDs
`[18,22,17,248046]`, EOS/stop and text `372`. The replay plan's frame count
was exactly four; the CLI's separate ceiling remained 64. All eight ranks
closed positively with four finite full-vocabulary FP32 vectors. The held
32-file content pass, normal read-only frontend cache hit, parent no-CUDA,
source/health, native-origin, CLI/Host and retirement gates passed. The original
external CLI exit 0 is recorded separately from attach/container exit 0;
post-retirement GPU compute contexts were empty.

The actual replay gate SHA256 is
`4ee2cd9b5a6ccf0cc33531b3cfc9c57c728715966837a32c34afa0d84902c4c9`,
its replay authority is
`5c22ec0812c1317d0be74f850c483bc19408f8136ce3f1ab290fea54bfe2dfb5`,
and independent actual review is
`fc4b86e598e7d9e33531b4147d03909f88035d848fc9eade846ec3a31667af4c`.
A forced replay answer establishes no numerical agreement by itself. The
separate full-vector comparison below passed exactly. This short control does not
qualify MTP/media/full histories, 24 active requests, physical worst-case
bounds, candidate adoption or P3. All timings remain diagnostic.


### Offline matched teacher comparison

The maintained quality tool has a separate `--compare-owned-teacher` selector.
It accepts a SHA-bound closed comparison input containing `reference`, `replay`
and an explicit read deadline. Each side binds its raw Host result and external
completion receipt, then validates the original authority, rank/startup and
vector references. This path does not use the historical 714-model corpus,
summary groups or quality margins.

Pairing requires the same immutable image and recorded native-module identities,
model/frontend/execution declaration, input IDs, vocabulary, frozen output
sequence and installed math-source identities.
Capture and replay retain their distinct mode, operator and observer source
identities. Coordinate profile IDs are resident-activation namespaces issued
from a fresh nonce, not content certificates. The comparator validates each
run's own activation linkage across its authority, startup, rank and frame
records, preserves both IDs, and normalizes only the single well-typed
`coordinate-profile-v1` activation marker when comparing stable KV semantics.
All other positional and numerical markers remain exact. Appending this offline comparator changes the quality tool's whole
file hash; carry-forward of existing teacher behavior requires structural
verification rather than relabeling the old hash as current.

The report includes uint32 bit equality (including signed zero), each full
vocabulary vector's raw argmax and ties, original selection versus the frozen
teacher ID, and diagnostic KL, NLL change and maximum absolute logit difference.
A forced matching answer is not evidence that its underlying logits agree.
Nonexact control repeats remain observed variability requiring analysis, not an
automatic quality pass or an invented numerical tolerance. The comparator
passed 132 x86 CPU controls with no skips. The 27 original helpers and classes
retain their ASTs; the legacy main body follows the new explicit offline
selector unchanged. Actual replay passed the operation gates above; its paired
comparison result follows.


The maintained comparator subsequently executed on the owned x86 application
image with immutable held source `4ea8c2c2`, after all 132 current controls
passed. It consumed the unchanged pinned capture/replay pair and reported
`EXACT_MATCHED_CONTROL`: four full Float32 vectors, 248,320 entries each, all
993,280 uint32 representations identical. Every frame had zero differing bits,
zero KL, zero teacher NLL change and zero maximum absolute logit difference.
Raw argmax and original sampler selections agreed at all four steps, with one
unique maximum per frame. Both distinct resident-activation IDs are retained
and each run's complete internal linkage passed; no numerical tolerance or
runtime guard change was used.

The comparison CLI exited 0; its container retired with exit 0, PID 0 and no
OOM. Its report SHA256 is
`6f87902e2bcab325139ac08853d5f2a1304ec303a1a35ec7966c96baea902e50`. The
report explicitly retains `qualification=false`, `quality_acceptance=false`
and `P3_valid=false`: this is one short matched reference control. Broader
quality/capacity/history/method/media and candidate/performance gates remain
open. The earlier pre-metrics activation-identity refusal remains preserved in
the audit evidence; the corrected comparator did not rerun either model.


### First 64K reference capture: preserved timeout

The first current long-input reference attempt used clean operator `cd2d0251`,
application `d736ab78` and verification image `efb4cb8f`. Its independently
preselected `sparse64k-000000` case supplied 65,472 validated prompt IDs, a
64-frame reply ceiling and three fixed retrieval keys in order 2,0,1. The
profile retained Q8 KV, 24 full native 262,144-row windows, NONE speculation,
Torch-reference attention and no decode graphs. This was one active request;
other materialized windows were idle.

All eight startup/setup bindings and readiness checks passed before exactly
one completion-request attempt. The canonical client then raised `TimeoutError`
while acquiring the nonstream HTTP response status/header under the predeclared
3,000-second socket inactivity interval. No response body, teacher logit frame,
closed-rank summary, native-child record or usable teacher authority was
published. Gold is **ungradable**, not a wrong-answer observation. No completed
prefill or decode rate is inferred from this failed, instrumented operation.

The original CLI and container exited 1; the container retired with PID 0 and
`OOMKilled=false`. All owned GPU compute contexts were gone afterward. Source, operator
inputs, model stamps and recorded health remained unchanged. The run was not
restarted and its deadline, prompt, precision, method and runtime were not
altered. The actual failure gate SHA256 is
`bc8622e3350804feff6e160ae129cdc9e00ef2e08094c4becd2fdfd94783996a`,
with independent failure/retirement review
`a46c3af5de8a9f6b797a453dac617f3a0260191cf4d9dfb46842f0c17ead4f6a`.
Its raw logs, client failure and partial evidence remain preserved.

A source-only assessment identifies per-query two-pass tiled attention,
repeated query materialization, full pooled-domain QSA scoring/sorting and
chunk-boundary status synchronization as hypotheses. It also confirms that
teacher frames need not exist until the final prompt piece. Neither fact
attributes the live delay or proves a hang. A separate bounded late-prefill
CPU/CUDA diagnostic is needed; the existing first-prefill trace cannot be
represented as a trace of this operation. This timeout closes no long-input,
sparse-QSA, method/media, full-activity, quality, performance or candidate
adoption gate, and grants no waiver for the four open Q03 numerical failures.


### Bounded late-prefill trace selector

The existing private benchmark scheduler trace accepts
`--scheduler-trace-prefill-start N` alongside `--scheduler-trace` and
`--benchmark-plan`, before the serve-argument separator. Omitting the selector
retains the first-prefill behavior. It binds the first non-warmup C1 cohort and
its prompt hash, requires `0 <= N < input_tokens`, and preserves C8-only default
plans. Teacher capture still rejects benchmark/trace combinations; appending
these flags to a teacher operation is not an admitted diagnostic.

Selection reads the decoder's committed host prefill cursor at the existing
rank0 adapter-round boundary. It performs no GPU readback, repeated `_pieces`
or `freeze_execution`, or per-round trace IO. Once selected, the existing
prepare callback supplies actual start/count, wire, slot/generation and state
position. The prepare callback must return after one positive prepared pass;
the receipt also records the prefix operation. The receipt
retains requested threshold versus actual prepared geometry; a cursor jump
need not land exactly on N. Maintenance, cancellation, missing/foreign owners,
no positive pass or an unreached trigger cannot claim an observed prefill.
The initial selection receipt is explicitly `UNSUPPORTED_NOT_OBSERVED`;
`SELECTED_UNVERIFIED`, `OBSERVED` and `FAILED` remain distinct outcomes.

The single prefill window replaces the initial prefill window rather than
adding a sixth capture. The existing maximum is five finite rank0 calls:
one prefill, two warmed C1 decode and two warmed C8 decode. Other ranks remain
untraced. Original callbacks are called once, on the scheduler thread, and
source-owned profiler retirement preserves the original failure even when
stop, export, geometry verification, clock or receipt publication also fail.
Profiler ownership is journaled before entry; cleanup status is prepaid and
keeps chronological failure priority.

The exact two-file source hold `2f7b9fb4` passed all 275 pinned x86 CPU controls
with zero skips, no outer diagnostic errors and positive CLI/Host/container
retirement. The earlier held version passed its pytest assertions but failed
the outer route because a test patched the shared clock; that failure remains
preserved. The correction scopes the fake clock to the tested module and does
not suppress or relax the outer diagnostic.

These are source/CPU collector results, not actual profiler execution or model
qualification. Capture count alone does not bound Kineto/CUPTI allocator or
export-file costs; operator RAM, device headroom, disk and deadline admission
remain required. All trace timing is `P3_valid=false`. A closed benchmark
client/shutdown/native-origin/result workflow must be admitted before an actual
late-prefill diagnostic is released; no new GPU run follows from this selector.


### Closed benchmark-diagnostic operation

The new `tensorfold.v100-owned-benchmark-operation-v1` schema with mode
`owned-benchmark-model-v1` uses the existing Host, canonical driver, owned
qualifier and rank supervisor. Its fixed operation DATA binds the model and
five prepared frontend assets, canonical matrix plan, client declaration and
four trace fields: committed prefill threshold, host obligation, minimum free
device headroom and disk allowance. It accepts no embedded program or arbitrary
argv. Teacher and component-suite operations keep their separate schemas and
constraints.

The initial diagnostic region is Q8/int8, C24 full native 262,144-row windows,
TP8/ETP4/PP1, NONE with no MTP, reference attention, no graphs, one measured C1
cohort and an optional preceding C1 warmup. Reply length is two. This prevents
the unchanged warmed-decode trace guard from selecting another window once a
stream reaches its limit, but actual receipts must still prove exactly one
observed prefill trace. The first proposed plan selects a 64-token warmup and
16,384-token measured prompt, with threshold 8,192. It is a diagnostic subset,
not the complete standard matrix or the failed golden 64K retrieval task.

The qualifier's private `--benchmark-client` input is admitted only with the
matching benchmark/trace operation supplied by the validated driver. Explicit
trace grants travel through a trusted Python-only operation argument and a
bounded internal worker DATA argument. Ordinary manual benchmark behavior and
teacher exclusions remain intact; appending unsupported flags to a teacher
operation is not an admission shortcut.

The client waits for eight actual startup bindings and same-instance full-State
readiness, then delegates to the canonical `bench_concurrent.run_matrix`.
Its optional caller-owned `ReadControl` reaches stats, SSE requests and event
polling; default `None` preserves the ordinary CLI request/work semantics.
Controlled barriers have finite waits, failure aborts pending rendezvous, and
accepted tasks are drained through their completion journals. A cancellation
flag or timeout alone does not release the client or its consumers. Byte grants
must be exactly representable in the canonical integer MiB/KiB arguments;
unaligned extents are rejected, never rounded or truncated.

Only complete matrix protocol/work/hash checks, the same idle service instance
and one valid `OBSERVED` trace permit the normal leader-first stop. The shared
owned-model observer records real worker PID/source/profile/full-State bindings,
positive resource retirement and child VMA-FD native origins. The parent and
Host consume typed client, matrix, trace and ordered all-eight completion
records. No teacher authority or parent-only native-origin proof is fabricated.

The host trace obligation joins the existing immutable pre-constructor host
admission once; it is not a later rebind or a second charge for settled
residents. Each worker observes physical free CUDA memory after full States,
and rank0 checks the declared floor again only at selected profile entry.
Those observations are headroom checks, not reserved memory or a proven
profiler peak. Disk extents and outer memory/time containment remain operator
obligations; an exported file must satisfy the declared bound before acceptance.
All instrumented client/trace results retain `P3_valid=false` and
`quality_acceptance=false`, regardless of rates printed by the matrix tool.

The integrated implementation passed five pinned x86 CPU suites: 84 Driver
and 57 Host tests in separate fresh standalone interpreters before real App
preload, plus 60 contract, 71 matrix and 299 owned-qualification tests through
the genuine qualified CPU route. All 571 passed with zero skips, no primary,
post or outer diagnostic errors, and successful CLI/container retirement.
Source and recorded health stayed stable. The final test-only revision added
real eight-rank aggregate and rebound semantic/producer-drift rejection
controls; implementation bytes did not change, and the four earlier suites
were not repeated.

The source snapshot is committed `43cb5549` plus exactly ten held source/test
paths, including the new contract test. These CPU checks exercise real
protocol code and synthetic model/transport fixtures; they do not execute a
model or profiler, prove physical allocator peaks, or establish P3. The old
component failures remain preserved rather than relabelled as passes. No new
model or profiler operation is released by this documentation. Accepted CPU
plan DATA and source/CPU controls do not replace actual runtime, resource,
trace, native-origin and all-rank closure gates.

## Grouped expert and larger-prefill C1 comparison

An eight-V100 SXM2 32GB comparison used the SC117 IQ3_S target shards at
revision `90a3350cd800ec1fe0d1660a38414ad9e985782f`, source-built
Torch `04cf2557`/CUDA12.9, and App
`sha256:32d79bf177aefca987aaf09a81c215f1627b9263dd294dfd7a7a2787f0cd883b`.
Its curated source UID is
`ce5eca6e5640dbb4b34f303cf969468703ea69e5e7ec323777f37a6cb352ea52`;
it is the recorded pre-commit snapshot, not a rebuilt clean `91ef9427` image.
Both arms retained TP8/ETP4/PP1, RN-H12 uncaptured attention, Q8 KV,
24 materialized full 262,144-token States, NONE, FP16 activations and no graphs.

The control used 256-row prefill and inherited expert dispatch. The candidate
used 4,096-row prefill, grouped-down and gate/up grouping of four. This is a
comparison of the complete three-setting change, not attribution to one kernel.
The canonical frozen plan SHA256 was
`be16ff1c554a13ec765d3a53f2a6a444827cfaf85675b82247352c07cb7a8a6a`:
one excluded 1,024/16 warmup followed by two C1 1,024/32 requests per process.
Four fresh processes ran candidate–control–control–candidate, producing four
measured repetitions per arm. All requests completed fixed work with zero
cached tokens; no profiler or forced-token witness ran. Each process first
passed a separate unforced 52-input arithmetic smoke and retired successfully
with unchanged source/device health and no remaining CUDA compute contexts.

Rates below divide total completed work by the corresponding total interval;
they are not arithmetic means of rates. Decode excludes each request's first
token and uses the maintained inclusive phase envelopes.

| Arm | Prefill tokens/s | Decode tokens/s | HTTP-wall output tokens/s |
| --- | ---: | ---: | ---: |
| Control | 87.7630 | 1.95446 | 1.15742 |
| Grouped/R4096 | 268.5302 | 1.96368 | 1.62241 |

Observed prefill and complete-request ratios were 3.0597 and 1.4017. The small
decode difference remains uncertain; these dependent within-process samples
provide no invented equivalence margin or confidence claim. Both measured
32-token trajectories matched across all four processes. The excluded warmup
trajectory differed between profiles and remains preserved. This does not
establish full-model quality, other request sizes, maximum-active throughput,
long-history performance or deployment selection. The governing performance
targets remain unmet. The local evidence ledger retains the four raw runs and
`grouped-balanced-comparison-v1/paired-receipt.json` under the
`providerone-alignment-20261106` audit root.
