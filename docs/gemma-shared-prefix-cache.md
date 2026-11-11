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
