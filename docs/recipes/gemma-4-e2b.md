# Gemma 4 E2B and DFlash2 on CUDA

This integration reads the complete original BF16 Gemma 4 E2B conditional
generation checkpoint. It uses the native Transformers 5.17 model, including
per-layer embeddings, shared KV, clipped projections, vision and audio encoders.
The 26B MoE implementation and companion are separate architectures.

The current checkpoint pair is:

| Role | Repository | Revision |
| --- | --- | --- |
| Target | `DavidAU/gemma-4-E2B-it-The-DECKARD-Expresso-ONE-Universe-HERETIC-UNCENSORED-Thinking` | `a91cc4c5eaa277338bff3fd2889725bb6e07c2e0` |
| Companion | `rasyosef/gemma-4-E2B-it-dflash2` | `6eb3ade2e4ae841974d692e9edc177dcaa7e20ff` |

Both repositories must be complete local snapshots. Mount the complete
Hugging Face cache when snapshots contain links to shared blobs. No checkpoint
Python is executed. Preserve the target's own tokenizer, chat template, EOS and
generation metadata; the 26B deployment's template override does not apply.

The implementation requires one GPU, one request lane (`--parallel 1`) and BF16
target weights and KV. The native maximum context is 131,072 tokens. Admission
checks the requested window against the actual memory envelope; a small tested
window does not establish full-context qualification. The companion provides
one through seven proposals (`--draft-rows 2` through `8`). It borrows the raw
target embedding and vocabulary head, with four checkpoint-specified target
features. Target logits retain Gemma's soft cap; draft logits do not apply it.

`--drafter-bits 16` keeps the companion in BF16. The explicit `8` candidate uses
private FP8 copies of proposal projections and its borrowed vocabulary head.
The target and KV remain BF16. Proposal precision may change proposals; standard
rejection sampling must still preserve the represented target distribution.
FP8 and draft depth require actual checkpoint performance and correctness
qualification before choosing deployment defaults.

With `--vision`, the same checkpoint accepts images, timestamped video and user
`input_audio` parts. Audio is decoded to finite mono 16 kHz samples and is limited
to the model's native 30 seconds per segment. Oversized clips are rejected.
Images, video frames and audio use native processors and encoders. The reply is
text; audio input support does not add audio generation. See the
[API](../api.md) for transport shapes. Encoded byte, decoded media, visual-token,
context and request admission limits remain active. URL fetching requires the
separate `--vision-urls` opt-in and existing public URL policy.

## Deployed correctness baseline

The GB10 service `gemma4-e2b-tf` serves port `8888` and retains the API model alias
`gemma4-26b` for existing clients. The loaded weights are the E2B pair above.
The baseline uses BF16 target weights, BF16 KV and a BF16 DFlash2 companion with
eight verification rows (seven proposals), one request lane, the native
131,072-token context and this checkpoint's unchanged chat template. Image,
timestamped video and audio input are enabled; output is text.

The installed immutable runtime passed **1,423 CPU checks** with **50 documented
skips**, followed by **38 native processor checks with no skips**. The latter
exercise the 17 opt-in processor cases omitted in the earlier CPU run. All
**23 HTTP/Harness checks** passed, including stored audio/video, thinking,
tools/schema, SSE and cancellation. Production readiness and the retained alias
passed four additional checks.

Managed source-mounted execution through the same protected SDK passed complete
target/companion loading, exact target logits at verification widths 1–8,
greedy identity and seeded cache replay. It also exercised the native context
with 131,040 prompt tokens plus 32 generated continuation tokens in serial and
drafted greedy calls and sampled new/replay calls. EOS stopping was disabled
for this capacity diagnostic. A separate teacher-forced comparison of the first
eight generated IDs found hidden states, logits and all four draft taps identical
at verification widths one and eight. That native-context test preceded the
wheel build and used a source mount: it is **not** an installed-wheel 131K test.
The installed package's source identity supports carrying the result within
that recorded scope. This capacity diagnostic does not prove long-context
retrieval quality or performance. Earlier bare-entrypoint GPU runs are debugging
evidence only. See the [validation record](../gemma4-e2b-validation.md) for immutable
identities and the distinction between these checks.

Comparative prefill/decode performance selection remains pending. BF16/eight-row
settings are a validated correctness baseline, not a measured optimum. The
explicit FP8 companion candidate remains outside this baseline. Historical 26B
results do not establish E2B throughput.

## Deployment and API boundaries

The [Compose profile](../../deploy/gb10/compose.gemma4-e2b.yaml) and
[environment template](../../deploy/gb10/gemma4-e2b.env.example) retain the complete
read-only HF cache and protected Harness entrypoint. Use the qualified immutable
image and its resource envelope; do not substitute the 26B model paths, template
override or context. The former 26B container is stopped and retained for rollback.

Video input processes visual frames by default. The optional native HF
`load_audio_from_video=True` convenience is not exposed by this API; submit the
soundtrack separately as `input_audio` when both modalities are required. Audio
segments retain the native 30-second limit. Existing `logprobs` and decision
requests return HTTP 400; this baseline does not add those capabilities.

The [deployment receipt](../../deploy/gb10/deployment-gemma4-e2b-baseline-20261031.json)
records the installed baseline. Preserve the managed CUDA entrypoint and its
provider verification. Application-only rebuilds use the guarded
`Dockerfile.application`; changed SDK dependencies require the foundation build
workflow and fresh qualification.
