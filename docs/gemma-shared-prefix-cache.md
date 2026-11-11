# Gemma shared-prefix cache implementation ledger

Status: implementation, installed CPU and bounded CUDA transport gates passed;
full-model cache, quality, capacity and performance qualification is in progress. Production retains its prior four-entry snapshot cache. The
committed 256-entry cost-aware snapshot mitigation is held and unselected.
This task refines the shared-state contracts of the canonical ProviderOne
design; it is not a second full engine design or a new serving stack.

## Required behavior

With sufficient charged byte capacity and compatible exact prepared tokens and
media, A with 100,000 input tokens creates a reusable branch. B with 50,000 input
tokens sharing A's first 20,000 reuses those 20,000 and computes 30,000. A's next
turn reuses its own input and canonically rendered completed response. Common
prefix storage is shared, not cloned per conversation. Arbitrary text matches
after different preceding context are not reusable KV.

No fixed prompt-entry ceiling governs eviction. Tensor payload, storage-owner
overhang, metadata, live pins and publication scratch must all be funded. Cache
pressure may evict inactive state; it must never invalidate an active reader or
report an unavailable sliding-window cut as a hit. Telemetry must distinguish
matched/materialized tokens, physical retained bytes and eviction causes without
exposing prompt text. The 500K example concerns aggregate retained context, not
a change to the 262,144-token per-request limit or a proven memory capacity.

## Implementation increment and boundaries

First increment: immutable shared GPU block archive with independent full,
sliding-window and endpoint-hidden residency; exact prefix lookup and import
into the existing target and DFlash State buffers. This preserves the selected
attention/graph implementations and has a paid GPU import cost. Direct paged
readers/writers remain a subsequent performance candidate; no zero-copy or
optimality claim follows from the archive design.

Generated speculative rows are not automatically canonical prompt state. Cache
publication must include the final accepted token, exclude rejected proposals,
and use the existing prompt-role arithmetic. Chat publication uses the actual
completed wire-visible assistant fields and the same renderer/tokenizer/media
preparation authority as the incoming request. This matters when the template
omits reasoning or normalizes tools. Raw SDK and frontend publication are
mutually exclusive, request-owned routes, not two postpasses.

Frontend finalization retains its turn ownership, then reacquires the engine
lane and rematches/pins state before publication. No borrowed live State crosses
that gap. Cancellation, partial tool output, failed retirement and context
exhaustion must produce honest nonpublication/frontier outcomes. Existing
model weights, KV precision, 40 GiB CUDA cap and 16 GiB host reserve are retained.
No absolute 500K all-cut BF16 capacity is promised: keeping every historical
sliding-window row has a substantially different cost from keeping only recent
restorable conversation frontiers.

## Work ownership and evidence gates

Canonical source: `/home/kklouzal/TensorFold` on `sm-v100`; retained branch
`python-0.6`. Baseline is commit `9d8587abecec70726bb74b41280d894f173bcf1e`.
The GB10 checkout is only a mirror/push relay and immutable packaging source.
One combined dedicated reviewer follows the user's current Lite D13 instruction,
which supersedes the repository's older separate-reviewer wording for this task.

- Core owner: shared store, GPU adapters and engine publication.
- Frontend owner: immutable input capture, completed-turn rendering and lifecycle.
- Primary: integration, documentation, source/image provenance and Git.
- Dedicated reviewer: task-wide cleanup, correctness and policy audit.

Required gates remain open: component and failure controls; installed-origin
checks; actual GB10 target/DFlash state and output parity; image/video reuse;
context and physical-resource admission; cancellation/close/reader retirement;
complete serving measurements including imports, canonicalization, finalization
and next-request wait. Exercise interior cuts, ring wraps, partial blocks,
A/B/A branches and long-conversation retention among short requests. CPU tests
or another image's receipts do not close these target gates.

## Research basis

Reviewed 10 November 2026: [vLLM prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
provides the shared-block and contextual identity reference;
[vLLM hybrid cache manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
identifies distinct full/sliding-window residency requirements;
[SGLang radix cache](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
is a structural comparison candidate. These are architectural references, not
proof that the current TensorFold kernels support their layouts unchanged.

## Maintenance integration checkpoint

The three Gemma launch/proposal/head qualification tools now use the engine's
synchronous `clear_prefix_cache()` ownership boundary. Exact head-pair checks use
`prefix_cache_identity(tokens, media=())`, an immutable position/backing-owner
fingerprint, rather than a borrowed snapshot or processed logits. The agreed
runtime implementation must return no identity unless target, canonical drafter
and endpoint hidden state are all resident. Generations may not be reused.

The six affected utility/test files passed 118 source CPU controls in the owned
AMD64 source-unit image `11c89e2c` with zero skips, after final fixture cleanup.
A dedicated review found no component blocker. These controls exercise the
maintenance protocol through doubles; installed/native cache implementation and
full-model qualification remain pending. The exact command, six file hashes and
full log are retained in the remote task artifact `maintenance-tools-cpu.json`
and `maintenance-tools-cpu.log` under `audit-artifacts/gemma-shared-prefix-design-v1`.

## State transport checkpoint

The target State and DFlash State now expose typed range byte counts, detached
row export and in-place range import. The target stays poisoned until pair-level
restore publication. Their original snapshot, restore and inference arithmetic
method bodies are unchanged; DFlash construction additionally retains immutable
layer-type metadata for these range maps.

Six source CPU cases passed with explicit canonical module origins, checking
unique BF16/FP32 words, ring wrap, interior offsets, source detachment, guard
regions, fixed addresses, byte counts, invalid frontiers and partial-copy failure.
Dedicated component review is clear. The first attempt loaded old installed
classes before pytest configured the source path; its six failures remain in
`state-ranges-cpu-v1.log` and are excluded. The corrected source run and three
file hashes are in `state-ranges-cpu-v2.json`; the frozen source packet is
`state-range-component-v1/hold-v1.json`. These are source transport checks, not
GPU/full-model/archive coverage or permission to publish an incomplete restore.

A subsequent range-export increment accepts explicit full/window group selection
so an existing resident component need not be cloned when another component is
missing. Full-only export can recover older committed global rows without
claiming that overwritten sliding rows exist. All ten source CPU controls passed
with unchanged canonical origins; four added cases prohibit unrequested gather,
index and clone work and validate selectors. Dedicated V3 review is clear.
The test-only semicolon formatting cleanup is AST-identical, recorded separately;
Ruff also passes. This is a resource-work-elimination contract, not a measured
serving gain or a complete partial-hit qualification.

## Host admission classification checkpoint

The existing shared physical-host commitment check now distinguishes a valid
but insufficient observed allowance with `HostCommitmentCapacityError`, a
`ValueError` subclass. The sole refusal branch changes exception classification;
its calculations, budgets and diagnostic text remain unchanged. Invalid inputs,
missing or inconsistent physical authority, unreadable cgroups and unexpected
allocation failures retain their original failure contracts. This lets optional
cache metadata decline honestly without hiding a failed physical observation.

All 22 affected source CPU controls and both-file Ruff checks passed in the
owned CPython 3.12 container. Dedicated component review verified the narrow
change. These controls mock the observation boundary: they do not establish an
actual host allocation reservation or full-model resource qualification. The
metadata adapter must still supply the complete outstanding host commitment and
catch only the expected typed decline. Receipts are
`host-admission-type-cpu-v2.json` and `reviewer-host-admission-type-clear.json`.

## Shared index checkpoint

The Torch-free shared-prefix index is implemented with immutable, shared payload
spans, arbitrary interior cuts, independently resident required groups and
explicit read/write/temporary leases. Token and media identity determine the
prefix; actual group coverage determines the reusable cut. Capacity is byte
credit, including index/lease metadata, with no prompt-entry ceiling. Eviction
uses dependent-prefix work plus aging; freed bytes are a tie-break, not a
work-per-byte ratio. Component history is bounded to 32 events.

Actual CPython 3.12 container controls passed 26 tests plus 13 subtests with no
Torch import, alongside Ruff and syntax checks. Dedicated review's writer
registration, borrowed-planning and cleanup-bookkeeping failure findings were
remediated with preserved reproducers. The final unused helper and duplicate
public count were removed. The last removal is independently AST-equivalent;
its final hold is `cf6c7616` and source SHA starts `d4301c9d`. These controls prove
only the source index/ownership contracts; GPU copies, application integration,
physical resources, full-model quality and serving performance remain open.

## Byte admission and endpoint-retention successor

An integrated 40K retention control exposed that all KV groups could remain
resident while the final hidden row was evicted. The preserved 73-pass/2-failure
run was remedied by valuing complete-prefix availability for that component.
The exact original byte allowance and strict 40,000-token endpoint assertions
remain unchanged. The new policy is
`complete-prefix-availability-aging-bytes-tiebreak-v2`; it is an availability
proxy with aging and a bytes tie-break, not measured marginal recompute time.
The full 75-case Engine/archive cohort then passed without skips.

Explicit archive byte credit now replaces the old allowance exactly once in
capacity planning. None retains the existing default algebra; zero disables.
An explicit allowance pins the requested or native context rather than silently
shortening it to fit. Core, capacity and Engine preflight share the same minimum
constructible index-budget check, before Torch/header effects for explicit
invalid credit. All 78 source CPU tests plus 22 subtests passed in CPython 3.12;
review verified unchanged workspace/math and the original constructor boundary.
Eleven style diagnostics in an unmodified old test helper are baseline-proven;
there are no new static findings. These are source admission/retention gates,
not installed-image, GPU/model, physical capacity or performance qualification.

## Integrated production-backed source checkpoint, 11 November 2026

All task implementation components are source-reviewed, including the independent
native-retirement latch when archive credit is zero. A failed native fence stays
latched; a graph-capture refusal before any fence remains a separate contract.
The source package was assembled over the exact deployed `9e475` parent using
24 reviewed runtime path changes. CLI and typed host-admission patches reverse
exactly to their parent files; 102 native source files and the FD C source remain
unchanged. The derived source identity is
`e9701a8f432d9f2339bed869b1bbe46d763efbc52414d7126fdaa95e954da11e`.

The immutable production-backed source ran in the existing owned ARM64 SDK
verification image with Transformers 5.17, qualified Torch and the exact FD addon:
512 tests passed, five optional OpenAI client-SDK cases were skipped because that
client is absent, and no tests failed. All four required xgrammar cases and five
qualified-Torch GQA cases executed successfully. SDK, source, installed-origin
and NoCUDA guards passed; the CPU container retired without OOM. The optional
Python client pairing remains explicitly unqualified, not counted as a pass.

This closes a source/API/dependency integration checkpoint, not deployment,
installed-candidate, GPU or full-model qualification. Genuine whole-wheel
installed-origin checks, GB10 state/graph/media checks, full-model A/B/A cache and
canonical-history checks, physical-resource admission and complete serving
measurements remain required before replacing production. Production is still
on its previous four-entry snapshot image.


## Installed and actual CUDA transport checkpoints, 11 November 2026

The production-backed assembly was built as a genuine offline application wheel
with the existing qualified SDK and native inputs. Runtime image `bcb8d744` and
verification image `9bf137ec` contain the unchanged `e9701a8f` application source.
The wheel SHA256 is
`3faf3210d701931b1d583d674baf3bd6570b80dbb5c0c88d99bbb0df731214c3`.
Installed-origin controls passed 644 tests with five explicitly unavailable
optional OpenAI client-SDK cases skipped. All four required xgrammar cases and
five qualified-Torch GQA cases passed. Exact 783-member application, FD, SDK,
source and NoCUDA guards passed before and after execution. The installed
production tokenizer configuration also qualified the incremental encoder.
A first fixture-only collection failure remains preserved; adding its exact
read-only tool dependencies changed neither package nor assertions.

Twelve actual GB10 SM121 controls then passed against the installed candidate:
interior and wrapped paired state rows, fixed graph addresses and replay,
delayed-event pins, partial-copy failure retention, and native-262144 tail
transport. No checkpoint or model was loaded. All source/origin/resource guards
and retirement checks passed, with zero skips, no OOM and final allocator usage
zero. Observed maximum allocated/reserved bytes were 154941952/167772160 under
the separate 2 GiB allocator allowance; these observations are not physical
worst-case or full-model admission bounds. Production remained running during
the successful twelve-case run.

Earlier operator-only failures occurred before pytest: unsupported Docker
runtime binding, an invalid fraction-order allocator check, and a diagnostic
NameError. They remain recoverable and were not counted as runtime failures or
passes. Each maintenance attempt restored the original production container.
The corrected whole operator passed static checks and a complete no-array
preflight before the successful target suite. Its allocator guard derives the
unique enforced integer-byte bound from the exact pinned SDK set/get rounding
contract, without an epsilon or increased allowance.

Reviewed receipts on GB10 are `built-installedCPU-receipt.json` and
`reviewer-built-installedCPU-binding-clear.json` under
`gemma-shared-prefix-production-assembly-20261111-v1`, and
`actual-tiny12-receipt.json` plus
`reviewer-actual-tiny12-and-whole-preflight-binding-clear.json` under
`gemma-shared-prefix-tiny-GPU-20261111-v1` in the audit-artifacts directory.
These gates establish installed integration and synthetic state ownership;
full-model quality, media, context, serving latency and deployment remain open.

## Bounded serving qualification procedure

`tools/gemma_shared_prefix_qualify.py` supplies raw 24K/12K branches sharing 4K,
actual accepted-output continuation, canonical Chat/tool/Responses histories,
and strict 40K retention across six and ten short requests. It uses the actual
served tokenizer, public HTTP/SSE data and frozen input identities. Baseline
arms record unavailable new cache features; candidate arms require them.
Common quality, usage, protocol and resource assertions apply to all arms.
External timing includes complete HTTP settlement; it does not silently treat
a server decode interval as complete request latency or establish a P3 gain.

The client and focused boundary controls passed 24 CPU tests and static checks;
its historical EOF failure and corrected cleanup-failure controls are retained.
The exact client source and 24 frozen input references were independently
reviewed. The authorized model order is baseline A1, candidate B, baseline A2,
with one model/context at a time and unchanged native context, winning DFlash
profile, byte allowance and host/GPU limits. Actual results must close these
model gates before public candidate promotion.


## Target findings and complete-frontier retention

The `15893f8a` cancellation revision passed actual GB10 decode and prefill
cancellation, consumed recovery and strict zero-obligation checks. After six
decode tokens, and after 1024 of 65536 prefill tokens, native pins, pending
batches, temporary credit and metadata owners reached zero. Cancellation alone
now retires through a real fence after the producer stops; ordinary requests do
not acquire that fence. Failed native retirement remains sticky with its pins
retained. The original failed receipt remains recoverable.

That revision completed one native-capacity cell: 262128 input tokens, `READY`,
two accepted output tokens and ordinary EOS. It took 775.02 seconds: 385.80
seconds of prefill and 388.61 seconds of publication. Publication reported no
matched prefix and recomputed all 262130 canonical rows. Only one of eight
capacity cells completed; the repeat was halted without a result, and timing
and A2 remained unrun. This is capacity evidence for that cell, not a completed
native gate or acceptable performance.

The retained cache contained about 12.59 GB and had evicted required full-history
groups while retaining later sliding-history groups. Independent physical-owner
geometry showed that the latest complete frontier required about 4.45 GB under
the unchanged 12.67 GB allowance; all historical cuts required about 63.22 GB.
The missing current frontier was therefore a dependency-retention defect, not
proof that its necessary state could not fit.

The next revision configures the actual full/window/endpoint requirements and
funds frontier descriptors and group memberships from byte credit. Soft
retention is separate from active reader pins. Pressure removes surplus first,
then relinquishes whole inactive frontiers; shared dependencies are retained
once. A provisional frontier protects data through an atomic media span without
making an interior cut reusable. Interrupted provisional frontiers yield before
valuable complete ones after real pins retire. Branch, abort, failed cleanup,
container growth and early configured-budget checks remain mandatory.

Final source controls passed 191 tests and 22 subtests. Bounded CPU controls also
preserved both the latest frontier and the original 262128-token endpoint after
separate additions of two and sixteen rows. The original endpoint has some
resident groups without soft retention: its immediate hit is verified, not a
promise against arbitrary future pressure.

Two paired valid CPU metadata comparisons measured incremental maintenance at
10.93/11.08 seconds versus optimized reconstruction at 16.10/14.73 seconds with
identical coverage, owner oracle and accounting. Final-source sanity measured
11.53 seconds. These are CPU metadata results; they do not establish GPU,
complete-serving or general performance superiority. Immutable reference modules
are test evidence, not alternate shipping modes.

The derived source is `2f63c731`; only the four held runtime files and declared
test closure change over `15893f8a`. App cancellation cleanup, all 102 native
files, SDK, checkpoint, drafter, profile, context and byte allowance remain
unchanged. The exact runtime image starts `6619cadc` and verification image
`0eda4b55`. Installed checks and a fresh full affected model sweep must qualify
this revision. Earlier core/protocol/media/cancellation passes remain bound to
their own images. Production still uses the original four-entry snapshot image.

## Completed serving comparison and input checkpoints

Later target execution exposed two additional defects, both preserved with their
original failures. A partially matched radix label was incorrectly treated as a
fully covered ancestor, allowing the original conversation endpoint to lose its
hidden row. Requiring matched-depth ancestry restored the unchanged 945/945
conversation-reuse check after the full core pressure sequence. Structural
metadata pricing also omitted some new child dictionaries, separately rounded
insertion chains and two splits within one label. Tight public-API controls now
price those allocations before mutation; the reservation guard and byte budget
remain unchanged.

Revision `dd06c137`, runtime `f3e502b0`, completed all seven stages on candidate B
and the original-image A2 control: core, protocol, media, both cancellation
phases, fixed-output timing and native capacity. All 36 measured primary request
and output-ID tapes matched across arms; all eight native output-ID tapes also
matched. Publication reused 262128 already computed prompt rows and computed
only the two canonical output rows, rather than replaying the entire prompt.
Both private models retired with positive context closure and the original
service was restored healthy. This completes the exercised functional suite for
that revision, not performance acceptance or public promotion.

Every one of the 18 two-observation HTTP median groups was slower on B, with
ratios from 1.00738 to 1.13657. Replayed 64K Chat seed requests frequently missed
on B and took about 33 seconds, while A2 usually reused them in about two
seconds. These seed costs remain included in complete-cell results. Repeated
input hashes are labelled as replays within the timing chronology; they are not
all novel-context growth. Raw completion wire first/last-visible timestamps are
unavailable and remain null; server timing is not substituted for client wire
TTFT or token pacing. No equivalence margin or retention benefit cancels these
regional losses.

The next source increment retains actual request-input and canonical endpoints
as separately funded, soft checkpoints. Intermediate capture progress remains
replaceable. The seed input is a true token ancestor of its completed history,
but later template rendering clears thinking history; token ancestry alone does
not preserve the older window and endpoint state needed to replay that input.
Checkpoint touches use radix path and media identity, including fresh equivalent
list tapes, without duplicating owners or permanently pinning entries.

For an already complete canonical entry, publication checks metadata, performs
the existing completion fence and positive reap, retains the boundary, then
rechecks complete identity. Only that final authoritative check permits a cached
publication result without importing payload into private working state. An
entry lost during funding falls through to the existing prefill path. Fatal
cleanup and cancellation behavior remain intact.

The combined source controls cover 216 unique passing tests and 22 subtests:
215 unchanged passes from the combined run plus a targeted passing correction
to one new test's expected error wording. This is carried coverage, not a claim
that a final 216-test cohort was rerun. Final Core/Archive and Engine source holds
are immutable; installed-image and fresh paired serving checks remain required.

This release targets the fixed 12,667,846,656-byte cache and 40 GiB GPU/16 GiB
host-reserve profile. Very small manually configured budgets are unqualified:
for example, 20 MiB can pass metadata admission but cannot hold a planned
1024-row window group, which the capture caller currently misclassifies as a
fatal value error. That separate caller-classification follow-up is deferred;
this ledger makes no claim that all positive configured budgets are qualified.

The checkpoint/publication revision is source `ef30337d`, runtime `d992ac68`,
verification image `b452563c`. Its installed cohort passed 292 tests with zero
skips or failures, including the actual installed media source-contract check.
Fresh B and A2 execution then completed all seven stages per arm, including all
36 measured fixed-output cells and all eight native-capacity cells. Exact
primary inputs and output tapes matched across arms. Both timing windows used
the same prospective timing-before-native order.

All repeated 64K seed inputs on B now fully reused 65536 tokens in 1.63–2.23
seconds; the prior roughly 33-second replay misses were removed. First-seen code
and prose seeds still cost 32.96/33.36 seconds and remain in the record. However,
17 of 18 descriptive two-observation primary HTTP median groups remained slower
on B, up to a ratio of 1.15936; short T0 warm was about 0.9924. The code T1
complete-cell gain reflects an A2 replay-seed miss, not novel growth. All costs,
including new canonical output materialization, remain visible. The original
snapshot cache lacks that new output-publication contract, so this comparison
is not proof of a same-contract whole-feature winner or equivalence.

The user explicitly prioritizes the complete new cache and considers an added
0.3–0.5 seconds on cold requests likely acceptable. This clarifies feature and
performance acceptance; it does not erase the cached-region costs, establish
statistical equivalence, or prove whole-feature superiority. The exact public
cutover and rollback plan still requires final bound inspection/configuration
review and Root's concrete execution release. The original container
and persistent configuration remain preserved for rollback.
