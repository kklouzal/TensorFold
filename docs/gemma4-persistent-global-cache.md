# Experimental persistent CT global cache

This constructor investigation is default off and unselected, with no public CLI
flag. The immutable source/image below has passed numerical and scoped service
quality gates. Matched performance, confirmation and production selection remain
incomplete. Historical results and failed/partial attempts retain their original
identities; documentation-only checkpoint changes are not a new tested image.

`Gemma4Engine(..., global_cache="k")` or `global_cache="kv"` enables a native CT
candidate; the default is `"none"`. `Gemma4Target` exposes the same policy. Other
numerical policies are refused for this option without changing their default
paths. Original raw BF16 projections and FP32 inverse RMS remain authoritative.

Each State owns fixed per-global-layer BF16 derived K, and optionally V. The
initial policy prepares every append, including prompt and visual rows, then uses
derived attention only for decode/verification. Prompt attention retains its
existing raw/trig or shared prepared-prompt choice. Persistent prompt consumption
and deferred end-of-prompt preparation remain separate open candidates. Existing
trig tables and shared prompt scratch are retained and charged separately, with
no implicit aliasing/reuse credit.

Preparation retains the original two-half FP32 reciprocal-RMS reduction, CT
FP32 normalization and learned multiply before BF16 rounding, original partial
RoPE columns/trigonometry, nonfused scalar arithmetic, dot accumulation, key-tile,
softmax and chunk-fold order. K-only computes V from original raw/inverse values.
No BF16 normalized-V reconstruction of K or fused producer is introduced.

Snapshots remain detached raw authority plus sliding caches. Restore marks the
derived prefix invalid; the target regenerates it on the exclusive owner stream
before eager use or graph replay. Regeneration inside capture is refused. Fixed
addresses survive restore, reset and rollback. Reset makes an empty prefix valid;
rollback changes visibility and later appends overwrite rejected suffixes.
Interrupted restore/forward/regeneration leaves the State poisoned; a completed
restore or reset is required before further use. A State's derived preparation
owner must match its target. Raw-only snapshots can still be restored into an
otherwise matching State, whose local derived owner regenerates its own buffers.

At131072 with five global layers, two KV heads and D512, live K adds
1,342,177,280 bytes (1.25GiB); live K/V adds2,684,354,560 bytes (2.5GiB).
The three-snapshot reservation covers one prefix entry's two checkpoints plus
the next prompt boundary, not three independent conversations. Derived arrays
are not multiplied by this snapshot count. Existing graph outputs/pool/partials,
raw snapshots, companion, workspace and host floor remain reserved. Optional
prepared-prompt scratch adds512MiB and trig tables64MiB independently.

The source adds mode checks to the default eager/staged/layer paths and one to
`Graphs.forward`; source inspection does not prove execution neutrality. Caching
more prepared data can increase cache footprint and physical memory traffic.
Required comparison includes original raw, K-only, K/V, existing trig/prompt
candidates, request/restore costs, generated-code spills/cache traffic, complete
model numerical gates and controlled end-to-end distributions. No speed,
quality, memory-capacity qualification or equivalence follows from source counts.

Maintained controls are `tests/test_gemma_persistent_global_cache.py` and
`tests/cuda/test_gemma_persistent_global_cache_cuda.py`. The GPU component target
uses fixed BF16 projection lookups to exercise actual State/eager/graph entries;
it is not an NVFP4 full-model replacement or qualification.


## Qualification checkpoint, 2026-11-03

The tested GB10 capture is source
`6a1a9716a91ffb80984ab0df876760a2543380c9937a8960dc2d3881810dd1cd`, runtime image
`sha256:38670db40e4ba10a09b397425329666d2cfd3bfb225141a525bd68d0fc2170d7`,
and verification image
`sha256:0bdd410ec3f40e7ca8669d2b9b50112d77c57840aeb1de941758fec3a6c564d4`.
It used the pinned CUDA 13.4.3 runtime on SM 12.1, AEON target revision
`90c8601b1a439497bc15d8fba9e52990b45ea0cb` and DFlash revision
`77d4202772dfe50b2396ec7bac9cfffc7b9e7057`. This checkpoint's later documentation
edits preserve the six tested runtime files and two maintained test files byte
for byte; they do not relabel a rebuilt or newly tested application image.

Source CPU checks passed 459 tests with 15 documented CUDA skips. Installed CPU
checks passed 128 tests, and actual GPU component checks passed all 34 tests
without skips, including 18 persistent-cache word/State/eager/graph/handoff
cases. These suites overlap and are not summed. Fresh independent NONE, K and
KV model processes all used **Delta4 R16**, raw prompt/attention, FP8 DFlash
at depth four and context 131,072. Code/prose 64K and native-short cases matched
full unscaled vocabulary lookup, complete private FP8 code/scale fields,
serial/eager/graph teacher hidden/logit/tap words and effective target p, actual
immutable q packets, greedy serial/drafted IDs, seeded own replay and non-time
counters/cache behavior. K/KV deliberately poisoned valid derived prefixes,
restored raw-only snapshots and regenerated exact safe outputs before eager
and already-captured graph use. No extra State copy supplied that proof.

Private KV service checks exercised aliases, sampling/cache/SSE, thinking,
reasoning/backticks, strict schema/tools and tool history, bias and whitespace,
Responses history/storage, image/video Chat and Responses, and unsupported
valid-WAV audio refusal. K separately exercised its K-only/V-raw consumer with
cache transitions and serial-ID controls, full-context replay, active
cancellation/recovery and the exact recorded KV image/video request bodies,
complete deterministic response semantics and usage, stored Responses GET and
audio 400. The unchanged App/Harness protocol evidence is explicitly reused;
the entire original API helper was not rerun on K.

Both modes passed native-framed 131,056-token prompts with 16 usable reply
slots at the unchanged 131,072 context. Greedy replies reached ordinary EOS and
replayed from the complete prefix. Seeded T1 checks emitted all 16 requested
ignore-EOS tokens, separately reached ordinary EOS, and reproduced IDs,
non-time counters and cache hits on their own replay. This is a synthetic
capacity/usable-answer task, not general long-context retrieval quality.
Image/video execution and parity are not general vision-semantic accuracy.

The original short-SSE cancellation-counter assertion remains failed/unproven;
its summary was not rewritten. Separate probes closed after visible generated
output while still running, or after interior native prefill progress, and
required cancellation-counter advance, idle and a successful consumed recovery
request. These are wire/server observations, not independent GPU-inflight or
physical-retirement proofs. All private quality shadows retired through the
reviewed healthy shutdown path; production was not replaced.

At the loaded-state checkpoint, K and KV increased recorded PyTorch allocated,
reserved and peak counters by 1.25 GiB and 2.5 GiB respectively versus the
same-source, same-head NONE reference, matching their derived arrays. CPU
RSS/high-water and shared-host availability were recorded separately. These
are process allocator observations, not a net physical UMA RAM saving or a
startup improvement. Earlier Delta4 raw-table replacement observations retain
their separate `f913d5f0` source scope.

### Performance remains open

Four explicit mode producers completed all compiler/full workloads, retired
safely, and supplied hashed, durably published private compiler-cache snapshots.
Those producers are not timing references. The prospective Engine screen uses
six separate T0/T1 × short/code64K/prose64K regions, fresh processes in
A B D C | C D B A order, and full same-process warmup before strict measured
requests. A is raw R16/NONE; B/C/D are Delta4 R16 with NONE/K/KV respectively.

V2 stopped after an online comparator rejected in-memory tuple graph keys
against JSON list keys. Retained A/B JSON projections were exactly equal, and
a CPU A-self reconstruction reproduced that representation defect. The original
failed B outcome remains rejected and separate. A new reviewed V3 operator
normalizes only the declared graph-key tuple/list representation while
preserving exact integer/integer/boolean values and all ID/counter/config/script
checks. Its fresh matched epoch completed raw A, then production activity
invalidated B during warmup. No complete eight-process series or final raw-A
comparison is available. No candidate rates, integrated gain, equivalence or
winner follow from these incomplete attempts; no automatic retry runs under
active client traffic.

A meaningful quiet measurement opportunity, both raw-endpoint semantic checks,
and independent whole-process confirmation remain required. The conditional
memory-versus-small-prefill tradeoff supplies no numeric slowdown tolerance or
decode-regression permission. Persistent prompt consumption and deferred
preparation remain open candidates, and the final package still needs its
bridge against actual production before promotion.

Detailed source/build/control/quality and interrupted-run receipts are local
audit evidence under `gemma-aeon26b-optimization-20261102`; they are not bundled
as a public performance-validation receipt. The maintained validation summary
links here so old named-checkpoint pending labels do not imply a later image
was tested by inheritance.
