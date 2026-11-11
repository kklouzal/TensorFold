# Gemma shared-prefix cache implementation ledger

Status: implementation in progress; no new cache deployment or performance
qualification. Production retains its prior four-entry snapshot cache. The
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
