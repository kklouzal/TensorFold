# Gemma 4 26B-A4B ModelOpt NVFP4 on GB10

## Installed deployment: 2026-11-01

`gemma4-26b-a4b-tf` serves port `8888` using
[llmfan46/gemma-4-26B-A4B-it-qat-q4_0-uncensored-heretic-NVFP4](https://huggingface.co/llmfan46/gemma-4-26B-A4B-it-qat-q4_0-uncensored-heretic-NVFP4),
revision `a9e81e26ed81f541ec1d7314e72c61237a24566f`.
It runs alone, without a drafter, at the explicitly configured **131,072-token**
window, with BF16 KV and native image/video input. Client IDs `gemma4-26b-a4b`,
`gemma4-26b`, `gemma4-e4b` and `gemma4-e2b` all resolve to this service.
The preceding E4B container and image remain stopped and preserved for rollback;
unrelated services were not replaced.

The immutable production image is
`sha256:1d53fa836eb9e98c0c6dc04a2ba9e7295cab0f9a035fe0c4be14a587309ebe29`,
with source UID
`2f9d0c1ef280377b0976dcc3e23dbb6c20cb078247e99de6d94d14f194099948`.
It contains the installed application and owned allocator adapter, with no
serving source overlay or installed vendor patch. Source capture includes
uncommitted integration work; a Git commit alone is not its source identity.
The actual CUDA allocator ceiling is 40 GiB. Keep the protected bootstrap,
ordinary capacity admission and host reserve active.

```sh
serve /models/hf/hub/models--llmfan46--gemma-4-26B-A4B-it-qat-q4_0-uncensored-heretic-NVFP4/snapshots/a9e81e26ed81f541ec1d7314e72c61237a24566f \
  --backend cuda --precision checkpoint --name gemma4-26b-a4b \
  --alias gemma4-26b --alias gemma4-e4b --alias gemma4-e2b \
  --host 0.0.0.0 --port 8888 --no-drafts --parallel 1 --context 131072 \
  --kv-dtype bf16 --vision --thinking --thinking-budget 0
```

The template defaults thinking off, while this deployment intentionally defaults
it on; both explicit API settings work. Native sampling defaults remain enabled,
with temperature 1, top-k 64, top-p 0.95 and EOS IDs `[1, 106, 50]`.
The model has no audio encoder; audio input is rejected rather than routed to
another model.

## Stored arithmetic and architecture

Only routed experts use original ModelOpt NVFP4 W4A4 with direct calibration
scales. Attention, shared MLP, routers, tied embeddings/head, normalizations,
vision/projector tensors and KV remain BF16. Native FP32 RMS normalization
followed by BF16 rounding, BF16 GeGLU and native routing preserve the selected
contract. This format is distinct from compressed-tensors NVFP4 and does not
reinterpret its scale conventions or invent a dequantized BF16 weight mirror.

The 30 layers have hidden size 2,816, 128 experts with eight selected per token,
expert width 704 and shared MLP width 2,112. Attention uses 16 query heads,
eight local/two global KV heads, head dimensions 256/512 and a 1,024-token
sliding window. The native vision tower has width 1,152 and 27 layers.

## Qualification and remaining limits

Both shards (18,788,485,588 bytes total) matched their pinned publisher SHA256.
All 11,520 FP8 block-scale tensors, 23,040 FP32 calibration globals and 953 BF16
exception tensors passed the payload checks. CPU processor preparation matched
native image/video inputs and both thinking settings. Six actual GPU vision
feature comparisons matched native outputs bitwise.

The installed candidate passed 14 sequential API checks covering normal text,
thinking on/off, context recall and cached replay, tokenization, SSE, strict
schema output, tools, Responses history, image/video execution, unsupported
audio and cancellation/recovery. A separate configured-window capacity exercise
completed 131,056 prompt tokens plus 16 output tokens with EOS ignored. This
proves execution capacity, not retrieval quality or real-EOS output quality.
A nonstream disconnect check observed a request accepted/running before output,
then a server cancellation-counter increment, idle state and a reply of `4` to
a recovery request. Those observations alone do not prove a GPU chunk executed
before disconnect or independently establish physical CUDA retirement.
Post-cutover checks passed all four aliases, ordinary chat/EOS/no-draft behavior,
Responses and healthy idle state.

Native checkpoint metadata allows 262,144 tokens. An earlier raw-attention
262,128-plus-16 capacity request did not finish within the 600-second HTTP
inactivity timeout and the candidate was subsequently stopped cleanly. That
larger window is **not qualified or deployed**; improving its prefill remains
ongoing. This initial cutover preserves the preceding production window.

Earlier controlled 64-token greedy checks at a different, 262,144-token
configured context observed approximately 33–38 output tokens/s. They are
limited diagnostic measurements, not production throughput guarantees or a
universal fastest-model claim. The prepared-global-KV prompt candidate remains
**disabled and unselected**: 63 GPU operator checks and 144 full-model
output/logit comparisons were exact, and long-prompt TTFT improved, but the
highest-priority decode comparison remained inconclusive. Production uses the
raw-attention path.

B01 pairs equal-width BF16 local K/V and shared gate/up projections using the
existing B16 pair operators, original weights and arithmetic. It passed 23 CPU
checks, four actual GPU checks and 128 exact whole-Engine output comparisons.
A four-process screen at the configured 131,072-token window covered greedy
and seeded temperature-1/top-k64/top-p0.95 requests. All eight decode point
estimates improved by 0.458–1.649%; only the short-code production-like setting
had a conditional throughput-ratio interval above one: ratio 1.004576,
interval [1.000053133, 1.009119458]. The other seven remained inconclusive.
This limited screen establishes neither equivalence nor inferiority; B01
remains **unselected** and has not changed production image `1d53fa83`.
The [screen receipt](../../deploy/gb10/gemma26b-b16-pair-screen-20261101.json)
retains per-process medians, image/source identities and marginal 95% paired-log
Student-t intervals with only two process pairs (one degree of freedom).
The three cycles within a process are correlated; these conditional intervals
are not simultaneous confidence or a general performance guarantee.

The separate **unconnected** Sym8 prototype (`tensorfold.bf16.sym8.v1`)
passed 20 installed CPU checks, its owned native build, and four actual GPU
checks. These covered full original-head CPU bit reconstruction, direct BF16
GEMV at rows 1–8 against the independent raw B16 operator, raw/packed row lookup,
and changed-input/ID CUDA graph replay. The actual tied head occupies
830,472,192 logical packed bytes versus 1,476,395,008 original BF16 bytes, with
zero raw-fallback groups. This is format/operator evidence, not a serving,
allocator-residency or tokens/s improvement. The
[qualification receipt](../../deploy/gb10/gemma26b-sym8-qualification-20261101.json)
records the source/image, seeded inputs and validation results.

Sym8 retains exact BF16 words through an eight-magnitude palette, packed
indices and versioned metadata; ineligible groups retain original raw words.
Its fields must remain immutable and owned through stream retirement and the
entire lifetime of a captured graph and every future replay. Transfer status
retains owners on failure; invalid device IDs/metadata are process-fatal,
not clamped or retried. Separate fresh-worker probes produced the required
`CUDA_ERROR_ASSERT` failures. The subsequent installed integration passed four warmups and eight whole-Engine
text comparisons exactly, plus phased image, cancellation and cache checks.
The table saved 645,922,816 logical bytes; measured peak allocated memory fell
by 626,524,160 bytes. Observed decode was 1.03–2.87% slower, not faster, in a
limited process comparison. The owner accepted this small measured tradeoff for
head-memory savings and requested lossless heads for both drafter evaluations
and final deployment. This is not blanket acceptance of larger regressions.
The [integration receipt](../../deploy/gb10/gemma26b-sym8-integration-20261101.json)
retains image/source identities, conditions and comparison limits.

The source CLI now exposes `--lossless-head` for Gemma ModelOpt MoE CUDA
(default off). It supplies one packed table for embedding and projection while
preserving separate scaling and softcap; no `.weight` facade or decoded GPU
mirror is created. Header admission uses generic valid-format bounds, not the
measured all-compressed size. CPU raw/encoded preparation owners are charged;
unsafe nested transfer prevents outer retirement retries. The R16 primitive
gates passed; full speculative execution remains a separate gate. This prepares
runtime storage without modifying or quantizing the checkpoint. The production image remains
unchanged until the replacement is qualified.

The latest installed SDK capture is source UID `4bf0ba0a…` (runtime `b2a…`,
verification `fd7…`): 474 CPU checks passed in 5.64 seconds without CUDA
initialization. This is installed source/API evidence, separate from actual
model and optional-candidate gates below.

A separate source increment permits explicit BF16 z-lab DFlash opt-in for the
ModelOpt target after complete companion metadata, matching dimensions/taps and
normal admission. The existing five-layer decoder, six target layer outputs,
committed append/cache snapshots and actual-q verification are reused. No
companion is selected automatically, and the standalone production command
above is unchanged. FP8 proposal copies with a lossless target head remain
a separate prototype: the existing FP8 backend builds a private proposal head
from bounded, unscaled Sym8 lookup chunks using the existing quantizer. Original
target storage/probabilities remain unchanged; no full decoded GPU main-head
mirror is retained. New chunk owners and the intentional FP8 proposal copy are
charged. Two actual GPU tests passed on the new `4bf0ba0a…` SDK in 62.80 seconds
(82.146 seconds including the outer worker): FP8 payload/scales, FP32 logits
for rows 1–16 and raw lookup matched the original BF16-source reference bitwise.
Operator peak allocated memory was 4.733 GB; this is not whole-model residency
or a throughput result. Actual model-q cache/replay and performance gates
remain pending; BF16 remains the baseline. Lossless target verification is
explicitly bounded to 16 rows.

Baseline `04f9…` completed 128 actual target row-law comparisons, including
64k history: widths 1–16 in eager and graph modes matched the serial oracle
for hidden/logits, six taps, logical KV/state and complete selected probability
bytes. Rejection rollback, complete drafter cache/replay and fair 64k
end-to-end comparisons remain separate pending gates. No draft performance or
quality claim follows from removing the previous startup restriction.

Sparkulator is selected separately by its native assistant metadata. It retains
its own trained BF16 assistant table/head and exactly decoded CT INT4 baseline
projections, rather than borrowing the main table. The engine supplies scaled
main embeddings and accepted final normalized hidden rows, binds the actual
main state after allocation, and suppresses unused target tap collection.
Admission includes live derived donor KV, hidden-only snapshots and conservative
native SDPA repeat/FP32 workspace. A recorded cold-64k native-head oracle passed
108 bitwise stage comparisons
(pre-projection, native hidden/logits and post-projection), with proposal-prefix
IDs matching at counts 2/4/6/15. This does not establish live bridge/history,
restore/cache or media behavior; those and the fair 64k comparison remain
pending. No configured window is reduced merely to make the baseline fit;
resource refusal is recorded.

Video execution is not blanket semantic-quality qualification. Directed
red-object/rightward-motion and fixed-blue-object questions passed; a generic
caption was inaccurate, and a thinking-on diagnostic exhausted its 768-token
budget without a final answer. These limits remain recorded.

Two additional source prototypes remain default off. `--spark-global-gqa`
selects only the global Spark attention wrapper, with actual ModelOpt26B target
geometry and an explicit context no greater than 131,072; other heads/pairs are
refused before weights. The selected profile charges its separate bounded
workspace while the native default retains its full MATH repeat/upcast bound
(16 GiB at 131k plus the declared small owners). Forced efficient-SDPA provider
checks at 64k and 131k passed all 16 bitwise observations. The new `4bf0ba0a…`
SDK's recorded-64k whole-head GQA gate passed 108 bitwise field comparisons in
20.134 seconds. A subsequent Engine startup refused before weights because the
planner probed beyond the selected 131k profile. The source correction bounds
that search without changing the model native window or workspace formula;
installed verification is pending. Live Engine/cache/media, actual-q and model-performance gates
remain pending.

`--spark-sampled-q` selects the existing recurrent proposal sampler, with
verification width at most 16. It uses proposal temperature factor 1 and caps
q top-k at 256 (or the smaller positive target top-k), without changing target
p or its original sampling identity. Admission adds one-step effective-sampler
workspace and immutable q probabilities/IDs/headers beside target extraction;
no phase-reuse credit is assumed. Greedy requests retain the native delta-q
path. Actual cache/replay, law and end-to-end performance gates remain pending;
neither source option changes current production.

The earlier AEON dense 31B attempt is not the current deployment target. Its
publisher checkpoint contained a nonfinite scale; no NaN repair or weight-block
scrubbing was enabled. See the [historical 31B investigation](gemma-4-31b.md)
and [preserved E4B rollback deployment](gemma-4-e4b.md).

## AEON compressed-tensors replacement under validation

The 2026-11-02 replacement is `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`,
with the same z-lab DFlash companion. Its pinned headers use the existing CT
NVFP4 layout, original BF16 tied embedding/head, routers and vision tower;
attention and shared MLP projections use calibrated CT NVFP4 rather than the
ModelOpt checkpoint's BF16 exceptions. Native context remains 262,144. Actual
publisher generation EOS IDs are `[1, 106, 50]`; the card's historical extra
channel-stop workaround is not applied.

The AEON baseline uses its original exact BF16 tied embedding/output head;
`--lossless-head` remains a ModelOpt-only option. Actual AEON preparation rejected
the Sym8 24-bit raw-pool limit. Even without that limit its fallback count proves
at least 408 MiB expansion over the native table, so the CT codec extension was
withdrawn and its failure evidence retained. This Sym8 result is distinct from
the subsequently qualified lossless Delta4 path below.
DFlash retains its raw embedding callback and one target softcap. The model's
actual publisher EOS IDs are retained without the card's extra channel-stop
workaround. Loader retirement improvements preserve pending native owners on
unknown CUDA failure and report primary failures without a cleanup retry.

Native `tensorfold serve ... --delta4-head` now enables the existing CT-only
lossless Delta4 tied table at fixed reuse16 (default off). Keep the same model,
`--drafter-bits 8 --draft-rows 5 --context 131072 --vision` and owned Google
template arguments; append only `--delta4-head` to activate the table. The flag
is not supported by ModelOpt/dense Gemma or MLX and conflicts with the separate
ModelOpt Sym8 `--lossless-head`. It does not enable persistent caches, shared
prompt/trig tables, DFlash graphs or clipping. The owner accepted the observed
2-4% decode versus CPU-peak/startup tradeoff and requested deployment; independent
confirmation was not a cutover prerequisite. Production now runs this native
CLI selection in container `d1436c99`, image `98f6b664`, source UID `eecb4bbf`,
source commit `93ab8ba2`. These image/source identities precede this docs-only
update; runtime/test bytes remain unchanged. `production-current` points to
`deploy-delta4-cli-20261103-v1`; old container `099563af`/image `8da9dff3` and
its volumes remain stopped for rollback. No confirmed shipping-image speed gain
or global fastest/exhaustion claim follows from the earlier 07d/f9 pilot.

The new capture passed 408 source and 214 installed CPU checks, 14 probe controls and
actual typed Delta4/R16/load/drain verification. Ordinary native CLI shadow
testing passed 16 individual API/media checks, 14 directed regressions and strong
observed-active decode/prefill cancellation and recovery. Its original weak
short-SSE counter assertion remains failed/incomplete, so the overall API packet
is not labeled fully passed. Post-cutover 14 directed tools/bias/reasoning and
exact plain/SSE math checks passed. Both speech services remain healthy.

This checkpoint's full payload matched its publisher SHA, and the actual header
inventory passed CT/vision validation. The preceding 2026-11-02 deployment used
image `8da9dff3…`, source UID `6f3e809f…`. Current `98f6b664` serves port `8888`
with BF16 KV, the configured 131,072-token
window and private FP8 DFlash at depth four. The deployment explicitly binds the
owned Google template `gemma4-chat-template-google-4d7ae498.jinja`; its SHA256 is
`ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`.
The native AEON template and checkpoint remain unchanged. This override preserves
the existing OpenAI tool-history and current-turn reasoning replay contract.

The following checks retain the preceding 8da9/6f3 capture's scope, separate from
the current Delta4 cutover above. Installed-source checks passed 2,356 CPU cases with five MLX skips and twelve
explicit off-target SM70 deselections. The actual candidate passed text, SSE,
tools, structured output, thinking controls, image/video execution, Responses,
greedy serial/speculative equality and sampled cache replay. Four consumed
production alias generations and an observed-running nonstream cancellation,
counter/idle/recovery check passed after cutover. A synthetic ignore-EOS request
completed 131,056 prompt tokens plus sixteen outputs in 142.93 seconds; this is
capacity evidence, not retrieval quality or representative throughput. The short
SSE cancellation diagnostic completed generation before cancellation-counter
publication and remains inconclusive. The independent installed-image target-row
probe passed all seventy eager/graph versus serial comparisons at widths one
through five, with prefixes spanning the 1,024-token main window and 2,048-token
drafter window. Hidden states, logits, taps, logical KV and selected probability
bytes matched. Five bounded greedy/sample/cache requests also passed for the
actual private FP8 drafter at depth four; this is not a performance comparison.

Fourteen directed production regressions passed for native channel-close token
101 after backticks, plain/SSE identity, never-opened thought content with plain,
tool and strict-schema requests, active logit bias/cache restoration, and strict
or ordinary required tools. The tools/backtick fixture needed 839 generated
tokens before its natural close: its legacy 700-token run remains recorded as
budget-limited, while the same assertions passed in the normal 4,096-token reply
envelope. No server thinking setting or checkpoint was changed to force a close.

The [operational receipt](../../deploy/gb10/deployment-gemma4-aeon26b-20261102.json)
records exact images, source/configuration identities, checks and remaining
limits. The prior ModelOpt container, image, model and cache remain stopped and
preserved for rollback. Earlier results above retain their original checkpoint
and image scope; native 262,144-token execution and general media quality are not
qualified by this replacement.


## Separate persistent CT global-cache investigation

The native CT AEON target has default-off constructor-only K and K/V cache
candidates, with original raw authority and raw-only snapshots. They are
unselected and have no public CLI flag. The immutable `6a1a9716` / `38670db4`
capture passed same-head NONE/K/KV numerical/full-model gates and scoped private
K/KV service quality, including usable 131,072-context boundary/replay checks.
The original short-SSE cancellation assertion remains failed, with separate
observed-active cancellation/recovery evidence. Quiet matched performance is
incomplete after traffic invalidation; no speed or total physical-RAM gain and
no winner is claimed. The [candidate contract and qualification scope](../gemma4-persistent-global-cache.md)
retain exact identities, limits and pending P3/confirmation. ModelOpt defaults
and the existing trig/shared prepared-prompt investigations remain distinct.


The subsequent default-off `global_cache_prompt=True` constructor candidate
consumes those same active K/KV buffers during prefill, with no added persistent
bytes. It rejects redundant shared prepared-prompt/trig combinations. The fresh
`07d1664f` / `f9e04667` capture passed same-source five-variant numerical/prefill-state
qualification and scoped prompt-enabled K/KV service/media/cancel/131k checks.
The original short-SSE counter assertion remains failed; separate observed-active
checks passed. P3 remains unmeasured/traffic-blocked, no winner/default/deployment
changes follow, and earlier checkpoints retain their own identities. See the
[exact prompt qualification](../gemma4-persistent-global-cache.md#prompt-consumption-qualification-07d1664f).


### Optional shared global-prompt workspace

Gemma 4 native CT NVFP4 and ModelOpt MoE CUDA serving accepts
`--prepare-global-prompt`, default off. It forwards the existing admitted shared
BF16 full-attention prompt K/V scratch path; affine and dense Gemma are
unsupported. Workspace scales with the admitted context, not layers. The
accepted AEON131072/Delta4R16/FP8-DFlash-D4 GB10 profile uses one512 MiB pair and
retains unchanged target arithmetic; its prefill/retained-memory tradeoff was
accepted separately. Other context/model shapes must use their actual header
admission. This option is independent of ModelOpt's `--lossless-head` and does
not turn on other experimental graph/cache/precision choices. The accepted AEON Shared512 +Delta4/R16 +DF8/D4 profile is now deployed via
ordinary native CLI with both `--prepare-global-prompt --delta4-head`, on source
`7ac9307b…`/image `58a280bb…`, container `ffd9c363`. Existing model/drafter,
131072 context, thinking, vision and Google template/settings are preserved;
old `d1436c99` remains rollback. Source441 and installed115/777-source checks,
all14 directed production checks and16 individual API/cache/media functional
checks passed. The known seventeenth short-SSE cancellation-counter test failed
again, leaving aggregatefalse/exit1; no blanket API or new strong HTTP
cancellation qualification is claimed. See the [exact deployment scope and
planner estimates](../gemma4-cuda-validation.md#shared-global-prompt-scratch-cli).
The512 MiB pair is inferred from source/CLI/context/NativeReady, not an HTTP
physical-VRAM measurement. This documents an accepted prefill/retained-memory
tradeoff, not a new P3 result. The CLI/image artifacts predate this docs-only
status refresh. Other launches remain default off.

### Accepted private W4A16 head option

After qualifying this launch's original BF16 tied-head metadata, add
`--dflash-head-w4a16 --drafter-bits 8` to an enabled Gemma4 DFlash CUDA launch.
The flag defaults off; it changes only the private proposal head. Keep the
selected model/drafter revisions, context, BF16 KV, vision/thinking/template,
Shared512 and Delta4/R16 settings unchanged when reproducing the accepted GB10
D4 policy. It does not activate clipping, A8 or proposal graphs. The user
explicitly accepted the mixed measured trade, including the2.146% cached T1
code loss; it is not a universal speed claim. The new public frontend's
installed build/readback is complete: production `4b15d567`, image `2dff55cd`,
source `723221fd`, config `d95ecbff`, NativeReady/math4/14 directed checks.
The pointer selects `deploy-w4a16-head-cli-20261104-v1`; prior Shared/FP8
`ffd9c363` is stopped and retained as rollback with speech/settings preserved.
Source441-existing+31-new and installed462/exact777 checks passed. Full API
verification awaits natural idle: its first attempt refused active user traffic
before tests, preserving exit1 as procedure-only evidence, not a model failure
or aggregate pass/fresh strong HTTP cancellation qualification. Planner
estimates are not physical RAM; see the [validation scope](../gemma4-cuda-validation.md#w4a16-private-head-cli).
