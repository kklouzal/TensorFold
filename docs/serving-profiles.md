# In-process Flash Next serving profiles

`--serving-profile-root /profiles` enables prepared structural profile selection
for resident adaptive Flash Next GGUF GPU histories on SM70. This is an opt-in
implementation for qualification. Host protocol tests do not qualify actual
model loading, CUDA recovery, output quality, capacity or performance on V100.
The current design requires the SC117 IQ3_S source, image/video-to-text with
pinned visual/processor/decoder inputs, and compact indexer transactions. The
earlier text/fake-loader checks cover only their unchanged lifetime scope.
Those new source, media-envelope, position/replay, capacity and quality gates
remain open; this document does not establish their implementation or eligibility.

The existing `/v1/runtime` integer controls keep their meanings. A separate
standalone request selects model/context/KV/method resources:

```json
{"serving_profile":"flashnext/B.json"}
```

This selector is relative to the startup directory; traversal and symlink
escapes are refused. It uses the endpoint's existing authentication policy.
Operators control the local directory and checkpoint files. The HTTP body
cannot provide arbitrary constructor options, remote repositories or downloads.
`profile_manifest` remains the distinct execution-proof registration API.

A profile is a `tensorfold.serving-profile`, version1 JSON object with exactly
`schema`, `version`, `id`, `model_dir`, `options`, `api` and `artifacts`.
`options` uses supported `FlashNextEngine._initialize` arguments, requires an
explicit `max_len` prompt-plus-reply context, and can select `kv_dtype`, ordered
K/V formats, maximum `depth`, draft artifact directories and execution bounds.
Depth0 disables speculative ownership; a positive maximum keeps that method's
allocated capability available for runtime proposal_limit0. Processes, TP ranks,
communication settings, expert deployment and indexer-history mode stay fixed.
The selected model must use the maintained resident GGUF SM70 path.

`position_profile_manifest` optionally selects the position authority's bounded
JSON manifest, with its path relative to the serving manifest. The real family
resolver checks requested rows against the declared native/default or static
YaRN implementation and independent target/MTP/DFlash coordinate domains. New
bound positional execution currently requires the actual adaptive TP8 unique
GPU-history layout with two global KV heads and an eligible SM70 source provider.
Other maintained layouts retain their existing unbound source/CLI behavior;
an explicit new position manifest is refused there.
`max_position_embeddings` is the native reference, not a permanent ceiling on
prepared extended candidates. For example, `max_len:524288` with `yarn_factor:2`
requests a target candidate against a 262144 native reference; this is not a
released context or quality guarantee. Non-power-of-two allowances are subject
to the same resolved bounds.

A target YaRN override does not change MTP settings. An extended MTP recipe must
bind the actual companion source and an explicit compatible MTP policy. DFlash2
keeps its own scalar-position numerical algorithm and native absolute limit;
its bounded ring never extends those positions. Unsupported combinations are
refused before structural loading, without substituting another method.
The existing native DSpark path keeps its source-policy context limit; this
position authority does not implement an explicit DSpark positional manifest.

`api` optionally changes `served`, `aliases`, `default_thinking`, `sampling`,
`max_tokens`, `reasoning_effort` and `thinking_budget`. Sampling defaults accept
`temperature`, `top_k`, `top_p` and `min_p`; seeds remain request-scoped. Omitted API defaults come
from startup, so A-to-B-to-A cannot accidentally inherit B's tokenizer settings
or defaults. Each new decoder starts with its own allocated execution settings;
old derived prefix caches, execution proofs and pending controls do not transfer.
Execution-control updates are refused while a structural transition is pending
or unavailable; retry them against the new ready profile.

Every `artifacts` entry contains a local `path`, exact byte `size` and SHA256.
The inventory covers selected target shards, metadata, tokenizer/template and
draft inputs. Additional selected files appearing after preparation invalidate
the binding. The coordinator hashes actual content before queueing a transition;
peer ranks independently check the same manifest, metadata and file identities.
Files must remain unchanged through final publication. This identifies inputs;
it does not certify quality or measured execution profiles.
Hash work is bounded by the opened file size plus one EOF probe; growing,
truncated or changed inputs are refused. Draft inventory follows the actual
loader's source closure, including the DFlash2 pairing document and selected
tokenizer override. Explicit family manifests supply their actual source
closure through the provider helper. The complete resolved position receipt,
algorithm hashes, epoch and method/KV/lookahead binding participate in the
serving identity and `position_profile` record. All resolver-required source and
manifest files are inventoried; the receipt is resolved again before loading
and publication. These checks grant input/implementation identity only, and
report `unqualified-candidate` separately from artifact preparation status.
Before issuing a bound positional profile, the shared numerical authority constructs
and checks the actual small CPU frequency vectors, including combined FP32
intermediates. Its preparation receipt and byte counts are recorded; the
consumer then releases those CPU operands. Engine loading independently prepares
and adopts its own version-checked operands before model/device retention.

Prepare these files outside request execution, using the maintained helper:

```bash
python tools/prepare_serving_profile.py \
  --tp 8 --adaptive-sessions \
  --model /models/flashnext-prepared \
  --id flashnext-B \
  --options B-options.json \
  --api B-api.json \
  --out /profiles/flashnext/B.json
```

For example, an options object can contain
`{"max_len":8192,"kv_dtype":"int8","depth":0,"graphs":false}`.
`--tp` and `--adaptive-sessions` describe the actual intended deployment for
preflight only. They are never stored as process overrides in profile `options`.
The tool keeps its existing single-rank, nonadaptive default for legacy source
profiles; explicit new position manifests need the real compatible deployment
flags, source provider and layout. Programmatic preparation passes the same
startup settings through `create(..., fixed_options=...)`. The running service
independently checks its own actual settings before selecting that profile.

Choose method/artifact/quality settings from the applicable qualification; this
example supplies no V100 recommendation. The helper performs bounded streaming
hashes, real metadata preflight and atomic file publication. Model input paths
must already exist locally. Startup must use the CLI's required finite endpoint
limits for adaptive V100 serving.

Selection first validates every rank while the current model remains available.
Then new model-bound admissions stop. Already prepared and queued requests stay
bound to their original model epoch and finish or cancel before retirement.
A standalone prepared Python request owns its binding until `run`, `close()` or
garbage collection. Standalone preparations reserve the existing finite
`max_pending_requests` budget before tokenization; `run` adopts that same owner
without double counting. A callback creating another preparation needs an
independent slot. Profile-enabled Apps require that finite pending bound.
This API uses request-scoped GPU state; stored Responses JSON
keeps canonical transcript data without keeping GPU model/session allocations.
Interrupted close remains retryable until required release/notification succeeds.
Pruning/coalesced notification gets one bounded retry for allocation or
cancellation failure, preserving the original error. Repeated failure closes
admission and invokes terminal containment rather than stranding an idle drain.

Once those borrowers drain, the same worker fences graph/readers, retires the old
model and derived resources, and runs the actual engine load/admission/allocation/
warm pipeline using the existing communicator and TP control sequence. It does
not keep two full GPU models. All ranks must report readiness before the new
tokenizer/App binding and model epoch admit requests. CUDA graph launches are
collective where their captured NCCL operations are collective; construction
frontiers cover both ordinary communicator calls and maintained round replays.
Those CPU-store readiness wrappers are removed after construction.

`GET /v1/runtime` reports `serving_profile` configured/pending/effective inputs,
stage, model epoch, availability, borrower count and bounded errors. During a
drain, new model-bound requests receive HTTP503 with Retry-After. Models,
tokenization, health/metrics/stats reads also borrow their model binding. Runtime
status/selection and stored Responses GET/DELETE remain available independently.
Ready runtime projections borrow their model while reading host fields; during
a drain or unavailable state the endpoint returns host-only profile status.
Preserved notification errors never capture the old decoder in selector frames.

Recoverable request failures retain their original exception, causes, contexts
and groups. After successful decoder cleanup, an actor fence and method-specific
reader retirement, worker stacks become frame-free diagnostic notes before the
error reaches a caller. Unknown payload/reader ownership or failed cleanup stays
retained under fatal containment. Caller-held exceptions cannot invisibly pin an
old model while a replacement loads. This preserves request-error recovery and
does not rewind generic RNG or canonical output state.

An expected local bounds/file/allocation refusal before an unissued collective
can leave the old model running if preflight fails. After retirement it leaves
the service unavailable, with no effective model or advertised capacity. Select
a prepared profile again to retry in the same process/worker/CUDA context.
An already committed collective remains an obligation for every participant.
Ambiguous device, transport, capture or cleanup failures retain the existing
fatal cohort/supervisor containment policy.

Required target gates remain actual eight-V100 A-to-B-to-A loading, rank-local
load refusal/retry, cancellation/slow clients, repeated switches without residual
model allocations, selected contexts/methods/formats, numerical/quality and
integrated latency/performance on the selected source-built runtime stack.
The effective profile must also bind the real visual weights, processor and
decoder code/configuration, media fidelity policy, placement and declared work/
staging/replay bounds. Those helpers and the compact raw-ring/pool-frontier/
cut-tail state APIs must be integrated and qualified before the revised initial
profile is eligible; a full raw indexer archive is not a reconstruction contract.
