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
20.134 seconds. Live Engine/cache/media, actual-q and model-performance gates
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
