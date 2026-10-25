# Adaptive V100 source implementation and qualification

The [implementation specification](v100-adaptive-inference-design.md) governs
the eight-V100 TP8 path. Historical results in
[the V100 qualification record](v100-flashnext-validation.md) identify their
original sources and layouts. They do not qualify the adaptive implementation,
unique histories, a different codec, or a new execution envelope.

This work is developed without the eight-V100 server or its checkpoint and
source-accounting files. Source controls, pinned-stack compilation, installed
CPU checks, physical allocation, model correctness, quality, and measured
performance are separate gates. No new session maximum or throughput result is
claimed. The specification's 16 Q8 and 24 Q4 full-context MTP cases remain first
measurement candidates.

## Configuration and capacity

For Flash Next GGUF with GPU indexer history, `--parallel auto` computes a
resident count at the selected context. An integer is an administrator ceiling,
including counts above 32; it does not force an unsafe allocation. Every chosen
slot reserves its complete prompt-plus-reply window. The solver uses the
limiting rank and separate loader/serving/host costs, with a useful shared
execution envelope, retained recovery, replacement overlap, and bounded
migration occupancy. It keeps the selected context rather than shortening it to
fit a requested session count.

`--kv-full-windows auto` selects that full-context policy on the GGUF GPU-history
path. An explicit integer constrains the same capacity solver. Other providers
and the legacy RAM-indexer path retain their own explicit contracts; they do not
inherit the unique-history capacity calculation. The V100 Compose defaults use
automatic session and full-window selection. Existing numeric environment
overrides remain ceilings.

`capacity_plan.session_capacity` reports the computed count, configured ceiling,
limiting resource/rank, and rank costs. `capacity_plan.execution_envelope`
separates verification rows, prompt work, and proposal depth from resident
capacity. Splitting verification into fair complete-request microbatches does
not reduce the number of allocated full-context slots. Planning and materialized
allocation fields must be read with their evidence labels.

Target weights stay TP8. Unique GPU histories apply only to the supported
two-global-KV-head SM70 geometry. Existing Blackwell numerical and storage
paths keep their contracts. A different placement's performance cannot be
inferred from byte balance.

## Numerical formats and speculation

Ordinary INT8 retains the H32 transformation, signed byte payload, and FP16
scales per 32 values. Existing `rotorquant-signed128-4` and
`rotorquant-signed64-norm4` remain distinct formats with their original evidence.

The new `rotorquant-planar4-sm70-v1` format pins the CUDA Planar4 implementation
at `llama-cpp-turboquant` commit
`08e025c06ab521e4fa9e5c08b80af57614543e53`. A 128-value block has 64 code bytes,
an FP16 norm, and a reserved FP16 field. TensorFold stores code bytes and the
opaque four-byte metadata word separately; wire controls verify their combined
68-byte representation. The pinned reference's CPU and CUDA rotation constants
differ, so the format explicitly selects its CUDA authority. Native long-prompt
packing, attention, quality, and performance require their own V100 runs.

Stock MTP and `--draft-method none` have explicit resource and tail conventions.
Temporarily selecting zero proposals keeps enabled MTP conditioning current;
it does not free resident method resources. Reply/context budgets and stop cuts
bound the prefix committed to recurrent state, convolution, KV, indexer state,
and the drafter. Algorithm-defined random draws are not rewound on rejection.
Startup `--mtp-drafts 0` under `auto` is the legacy structural-disable choice;
explicit stock MTP requires a positive allocated maximum. Use the runtime zero
proposal setting for warm capability.

DSpark is a separate existing method. A DFlash checkpoint is not inferred from
a model name or classified as stock MTP. The exact compatible Flash Next
DFlash pairing remains an integration input. Its masks, feature conditioning,
draft histories, and resource envelope need independent implementation and
qualification before it is advertised as a serving choice.

Reference research identified a concrete candidate,
[`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/tree/a22a1744e85c17364dcc54a6a420af7135d4c7f6),
pinned at `a22a1744e85c17364dcc54a6a420af7135d4c7f6`. Only public metadata and
the tensor header were inspected. Its five-layer, 2,048-token sliding draft
stack, block-8 convention, and completed-target-layer HC averaging require a
separate provider/resource implementation. The author's GB300/TP1 evidence
does not qualify this fork's V100/TP8/Huihui target. Artifact discovery is not
the blocker; exact local target identity and implementation/qualification
remain outstanding.

## Runtime control

Adaptive engines expose `GET /v1/runtime` and `POST /v1/runtime`. Both use the
listener's existing API-key policy; opening metrics does not open this route.
The POST body contains one or more integer execution settings:

```json
{"row_budget": 32, "proposal_limit": 0, "prefill_tokens": 512, "session_ceiling": 16}
```

Values must fit that engine's reported `allocated_maximum`. The example values
are illustrative, not a V100 performance recommendation. The response reports
configured, pending, and effective values. The coordinator applies changes
between batches and shares a single ordered execution plan with TP peers.
Lowering the admission ceiling preserves running requests; raising it cannot
exceed the capacity already allocated. Active requests keep their full-context
reservation until completion/cancellation and retirement of their readers.

Structural changes to context, codec, model, or method are not writable through
this endpoint. They require the specification's state-preserving in-process
transition; a new setting must not silently evict sessions or be presented as a
live control before that transition exists. The endpoint never replaces
workers or restarts the CUDA context.

The runtime receipt is an advisory host snapshot, including active requests,
allocated slots, bounded derived prefix-cache slots/entries, the current
execution epoch, useful committed tokens,
and the capacity receipt. `/stats` includes this section for adaptive engines.
Reading it does not synchronize GPUs or turn startup planning into a current
physical-free-memory measurement.
Step elapsed time is host wall time, including waits actually observed by that
thread. Device-completion time is unmeasured and reported as null; asynchronous
maintenance must be included by a complete workload measurement before using a
timing sample as performance evidence.

The execution receipt includes `work_observations`, published by the serialized
worker at operation boundaries. It separates fresh logical prompt admissions,
request/replay prefix hits, submitted target prefill/decode and replay rows, and
MTP prefill/target-conditioned refresh/provisional proposal rows, including the
existing cached-prefix boundary refresh at its saved position. Warm runtime
zero-proposal refresh has its own count. Synthetic startup warmup is cleared
before serving. None of these row counts is generated output or device time.

Indexer counts are derived from actual submitted query ranges: visible complete
pooled groups, visible sparse group/query-head score pairs, and selected key rows
including incomplete tails. They count the canonical logical work once across
TP ranks, with no factor for replicated scratch or masked physical launches.
They are not measured memory traffic or a claim that every implementation
executes only visible arithmetic. Partial failed forward submissions are marked
unknown; complete totals are not fabricated. The prospective profile-table
`indexer_work` axis remains unknown until its separate covering contract is
established. `expert_group_max` remains null; batch size is not a substitute for
actual routed expert occupancy.

Acceptance histograms are fixed to the allocated proposal maximum, including
maxima above32. They count the existing verifier's accepted prefix after its
reply/EOS rules and before the text-stop prefix callback. Offered widths and
accepted totals by offered width are separate. The existing useful-output count
continues to exclude owed replay and failed grammar publication. No per-stage
durations are guessed. A failed admission observation retires or quarantines its borrowed slot and rolls
back the attempted counters. A separate failure latch invalidates completeness
even when publishing another snapshot cannot allocate. Counter exhaustion is
visible and invalidates completeness instead of silently wrapping; these host snapshots are diagnostic readings, not
new capacity or performance proofs.

Unique-history placement requests use standalone `{"attention_degree": 2}`
updates. Migration copies packed bytes through balanced bundles and publishes
each committed owner-map generation. An expected staging-capacity refusal
keeps the valid current map, reports blocking and waits for an explicit retry.
Ambiguous copy/publication failures retain poisoned ownership and staging
resources for cohort containment. The prototype's 64 MiB allowance charges
explicit additional tensor storage; allocator slab and driver peak occupancy
still require V100 measurement before claiming a physical migration bound.
Managed TP launches signal their supervisor on a fatal cohort operation so a
peer blocked in a collective cannot strand process retirement. Unmanaged
launches retain their external containment responsibility.

The CUDA generation and Responses APIs create request-scoped sequences. They
have no GPU retained-session close/expiry contract; storing or deleting a
Response affects message/response JSON rather than GPU ownership. Completion
or cancellation retires the request claim after its readers, transfers and
callbacks finish. Full-window allocations can remain reusable, and the legacy
bounded prompt-prefix cache can retain evictable derived bytes. Neither is an
intentional retained session: `sessions.retained` reports zero, while `active`,
`allocated_slots`, `derived_prefix_slots` and `derived_prefix_entries` report
those separate quantities. New admissions increment the destination slot's
logical generation before prefix copy/restore, including qualified prefix hits;
rollback within the same request preserves its generation.

A zero proposal setting keeps enabled MTP conditioning current only while its
request is active. A per-request `draft: false` switch disables speculative
work for that call. Request completion does not maintain an idle endpoint in
anticipation of an unspecified future resumption.

Execution profile registration uses either startup-bound proof references or a
standalone `{"profile_manifest": "new/manifest.json"}` request. Verification is
queued to the TP worker; paths cannot escape the configured evidence directory.
See [the exact manifest contract](execution-profile-manifest.md). Qualified
covering regions are explicit and nonoverlapping; token-by-token changes inside
one proven region can retain its selected execution plan. No measurements are
fabricated when the table has no eligible region.

## Reproducible hardware and workload evidence

After building the V100 verification image as described in
[the deployment guide](../deploy/v100/README.md), record the actual server:

```bash
python tools/v100_adaptive_qualification.py inventory \
  --rank-devices 0,1,2,3,4,5,6,7 --out platform.json
python tools/v100_adaptive_qualification.py transfers --out transfers.json
```

Inventory is read-only and bounds subprocess output and time. Missing tools,
failed commands, and absent measurements are retained explicitly. Run isolated
transfer probes outside serving performance measurements. GPU indexes in the
example are explicit logical-rank order, not a claim about the best NVLink
mapping. Record CPU/RAM/NUMA placement, UUIDs, ECC, links, clocks, power,
temperature, errors, and exact software versions.

Save the running `/stats` response, then derive the qualification matrix:

```bash
python tools/v100_adaptive_qualification.py matrix \
  --stats stats.json --out matrix.json
python tools/v100_adaptive_qualification.py benchmark \
  --url http://127.0.0.1:8000/v1 --out benchmark.json
python tools/v100_adaptive_exercise.py --url http://127.0.0.1:8000 \
  --out exercise --request short=/path/to/short.json \
  --request intermediate=/path/to/intermediate.json --request long=/path/to/long.json
```

The matrix includes each integer concurrency through the computed endpoint,
including capacities above 32. The benchmark uses the existing HTTP SSE
workload implementation and preserves raw responses/failures. Its short-prompt
measurements do not automatically qualify every matrix cell. Near-full
retention, mixed load, migration, output/quality, long positions, cancellation,
slot reuse, and sustained fill/drain remain distinct gates. Tool resource caps
are explicit workload-generator limits and must not become serving ceilings.

Compare identical target/draft revisions, codec, context, retention, output
quality, arrivals, and runtime state. Count useful prompt work and committed
output separately from tentative proposals. Include queueing, TTFT, inter-token
latency, fairness, recovery, communication, and migration. Select performance
profiles only from applicable completed measurements, never from a generated
matrix or startup allocation alone.

The exercise runner uses supplied JSON requests and actual token counts. Add
distinct long requests to establish multiple retained contexts, plus named
`stop`, `eos`, `exact` and `overrun` requests for their corresponding gates.
Missing input roles, early EOS in a context-exhaustion case, unobserved slot
reuse/retention and incomplete migrations remain explicitly unsupported or
pending. It drains response owners before restoring changed controls. Its
cooperative body deadline has the documented one-socket-read bound; it does not
claim instantaneous DNS/header cancellation. HTTP observations alone do not
authorize a qualified execution profile.
