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
