# Adaptive V100 source implementation and qualification

The [implementation specification](v100-adaptive-inference-design.md) governs
the eight-V100 TP8 path. Historical results in
[the V100 qualification record](v100-flashnext-validation.md) identify their
original sources and layouts. They do not qualify the adaptive implementation,
unique histories, a different codec, or a new execution envelope.

This work is developed without the eight-V100 server. The selected SC117 GGUF
payloads and mmproj have been downloaded and verified; source/header/import
controls do not establish device execution. Source controls, pinned-stack compilation, installed
CPU checks, physical allocation, model correctness, quality, and measured
performance are separate gates. No new session maximum or throughput result is
claimed. The specification's 20-Q8/32-Q4 figures are geometry reference points,
not admission promises or capacity ceilings.

## Current v6 source and CPU evidence

Current source implements the standard Responses and durable Conversations
lifecycle, bounded FIFO admission and live input/output credits, the common
prefix/retained-state catalog, RAM save/restore, and blocked-request requeue.
Logical API records, reusable computation and guaranteed retained checkpoints
have separate ownership and resource claims. See the [API contract](api.md#durable-conversations-and-quotas)
and [fixed serving policy](serving-profiles.md#fixed-endpoint-fifo-and-retention-policy).
These implemented paths have CPU software controls; they are not a qualified
V100 serving configuration.

The latest consolidated installed CPU checkpoint is immutable input tree
`413e132bf977625f485fffedf2011cc4bfc02f07`, image
`sha256:05eca41e61a9410660d2a38c2c4d9ced06124efec64d635cb21580bcbb399323`.
It passed **5,214 tests and 144 subtests across 156 modules**, with nine
documented skips. All 706 installed code files matched the frozen source;
there was no source overlay or within-run source drift. The skips cover two
MLX modules, an optional Qwen tokenizer, optional exact DFlash/MTP artifacts,
and four host Docker Compose controls. This candidate checkpoint is separate
from the published 4,365-test media milestone below.

Later source changes have their own evidence: the actual prepared-copy resource
correction passed **50 CPU checks**, retaining the enabled projection and
removing only structurally absent CopyIndex/list obligations when disabled.
The post-checkpoint prefix/cache/media caller changes passed **138 CPU checks**,
followed by **three** focused required-reservation/optional-prefix reclamation
controls. These counts overlap other suites and are not added to the installed
total. Their bounded cleanup and policy reviews are closed; they do not make
the earlier image evidence for later bytes.

The complete native transfer/backend lifetime and allocated host-progress
envelope, combined resource admission, final composed installed application and
publication remain pending. The selected key/value wake integration still
awaits final consolidation and verification. A borrowed client/clone's backend
property alone does not prove the connected daemon's backend identity.
Actual V100/TP8 CUDA and NCCL execution, model output/quality,
physical capacity, compiler and
whole-workload performance remain unrun for this source. No native/RSS or
physical-memory qualification is inferred from Python payload projections.

## Historical text/core installed CPU checkpoint

The installed application built from `0f656e9ab9642af6fa19210d721de13d375ea51d`
passed 3,277 tests and 105 subtests across the 78 affected test files, with four
documented optional/environment skips. The only later fixture change at
`b56298202ebe14b7fb1533ccb9f3871009f94d4c` declares the owner-layer reference's
existing unbound context explicitly; numerical assertions remain unchanged.
The image is
`sha256:9dc7ada61e4c1fdf272d218936ec8a08f3e372f40bec1dab53cae9237970bbc9`.
The source-built dependency environment was unchanged; this was an application
wheel rebuild and ordinary installed-package test, with no source-code overlay.

Coverage includes compact raw/pool transactions, issued coordinate/frame
identity, MTP conditioning/recovery, source authority, native/YaRN preparation,
Engine/CLI and serving-profile lifecycle, bounded admission/output, cancellation,
and the retained legacy CPU references. Corrections preserve primary failures,
guard complete model/profile identity, snapshot overlapping MTP inputs before
writes and poison a bound State after failed restore mutations. Separate cleanup
and policy source reviews are closed for this checkpoint.

This is a software/CPU result, not GPU kernel, collective, model-output,
quantization-quality, physical-residency or performance qualification. Native
SM70 compilation retains its separately recorded exact-source scope. That
snapshot preceded the installed media/frontend milestone below; it does not
replace the current source and qualification boundary above.

## Installed media and exact frontend integration

The application at `caf1eeda`, merged to `python-0.6` as `64b24dd5`, passed
4,365 installed CPU tests and 124 subtests across 111 affected test modules.
Four documented optional/environment cases were skipped. The ordinary wheel
image is `sha256:30e7036aa0d23ed4bb203f68a2d6e5f34cda617d06bf5cc2a8297d60080ef2bc`;
all 694 installed code files matched immutable source tree
`72aa4e585f2747ae02da083bb7a0d5958358212a`, with no source overlay.
[The maintained receipt](verification-sm70-media-integration-20261026.json)
records provenance and limitations.

Coverage includes the actual bounded CPU media work/TP input/App ownership path,
request-local decoder failure retirement, target/MTP shared input spans, complete
model token-ID domains, CONTROL presentation, nonlexical UNUSED rows, canonical
32-core MTP import/shared target vocabulary weights, frame/recovery containment,
principal request ownership and side-stream graph warmup ordering. CPU cohorts
and import controls do not qualify CUDA collectives or model outputs.

Current v6 FIFO admission, complete RAM retention/prefix reuse, standard durable
API lifecycle, host/device resource admission and automatic blocked-request
requeue are later integration work. This checkpoint is neither V100 release
qualification nor a capacity or throughput claim.

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

The initial `rotorquant-iso4-sm70-v1` format pins the CUDA ISO4 implementation
at `llama-cpp-turboquant` commit
`08e025c06ab521e4fa9e5c08b80af57614543e53`. A 128-value block has 64 code bytes,
an FP16 norm, and a reserved FP16 field. TensorFold stores code bytes and the
opaque four-byte metadata word separately; wire controls verify their combined
68-byte representation. ISO4 uses the pinned 32 quaternion rotations.
The distinct `rotorquant-planar4-sm70-v1` remains the first later equal-memory
comparison; it is not an alias or a substitute for initial ISO4 qualification.
The pinned reference's CPU and CUDA rotation constants differ, so these formats
explicitly select their CUDA authority. Native long-prompt
packing, attention, quality, and performance require their own V100 runs.

Stock MTP and `--draft-method none` have explicit resource and tail conventions.
Temporarily selecting zero proposals keeps enabled MTP conditioning current;
it does not free resident method resources. Reply/context budgets and stop cuts
bound the prefix committed to recurrent state, convolution, KV, indexer state,
and the drafter. Algorithm-defined random draws are not rewound on rejection.
Startup `--mtp-drafts 0` under `auto` is the legacy structural-disable choice;
explicit stock MTP requires a positive allocated maximum. Use the runtime zero
proposal setting for warm capability.

DSpark and DFlash2 are separately implemented methods. A DFlash checkpoint is
not inferred from a model name or classified as stock MTP. The current
`--draft-method dflash2`/`--drafter` path uses the same Engine and prepared
serving-profile lifecycle, with exact `draft_pairing` content authority before
CUDA allocation. Its [provider/runtime](../src/tensorfold/families/qwen4_exp/cuda/dflash2_runtime.py)
implements completed-HC feature conditioning, target embedding/head aliases,
bounded FP16 draft windows, accepted-prefix recovery and warm maintenance.
The implemented block is four rows, with one to three proposals selected
before the draw; a source block-eight declaration does not enable block-eight
serving. Source/CPU checks do not qualify this pairing's V100 execution,
acceptance, quality, physical envelope or performance.

Reference research identified a concrete candidate,
[`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/tree/a22a1744e85c17364dcc54a6a420af7135d4c7f6),
pinned at `a22a1744e85c17364dcc54a6a420af7135d4c7f6`. The initial research used
public metadata and the tensor header; the later source implementation now
handles its five-layer, 2,048-token sliding draft stack and completed-target-HC
averaging under the manifest-bound contract. The optional exact large DFlash
artifact control was skipped in the 5,214-test checkpoint. The author's
GB300/TP1 evidence does not qualify this fork's V100/TP8/SC117 target; exact
pairing and complete target execution/quality/resource qualification remain
required.

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

Opt-in [serving profiles](serving-profiles.md) add the structural transition for
context, codec, model and method through a prepared local manifest. The same
worker and communicator drain model-bound prepared/queued requests, retire old
resources before replacement allocation, and publish after all-rank readiness.
Host fake-loader protocol/lifetime tests cover A-to-B-to-A and unavailable-state
load refusal/retry. Actual model/CUDA-context loading, resident-byte recovery,
quality and integrated V100 performance remain unrun qualification gates.

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

Without an explicit computed-retention policy, CUDA generation remains
request-scoped: completion or cancellation retires the claim after its readers,
transfers and callbacks finish. Reusable full-window allocations and bounded
derived prefixes do not create an intentional retained-session promise.
Stored Responses/Conversations retain logical records independently of GPU
ownership.

An explicit typed GPU/RAM retention policy uses the implemented common catalog
for owner/generation claims, protected idle checkpoints, prefix leases,
suspension/restoration, close/expiry and incompatible-profile drains. FIFO
waiting, input credits and pending/in-flight output credits retain their logical
owner through blocked-request requeue. New-turn generation bookkeeping remains
separate from same-request continuation. These source contracts and CPU checks
are subject to the pending native resource/backend and complete target gates
above. New admissions increment the destination slot's generation before
copy/restore; rollback within the same request preserves that generation.

A zero proposal setting keeps enabled method conditioning current during active
execution. A per-request `draft: false` switch disables speculative work for
that call. Without computed retention, request completion does not maintain an
idle endpoint in anticipation of an unspecified future resumption.

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
