# Gemma 4 E4B on CUDA

The E4B candidate loads the complete original BF16 conditional-generation
checkpoint, including per-layer embeddings, shared KV and all vision/audio
tensors. Its text backbone has hidden width 2560, 42 layers and two KV heads;
its native context is 131,072 tokens. It retains BF16 target weights and KV,
one request lane and the checkpoint's own tokenizer and template. The live
service remains the [E2B correctness baseline](gemma-4-e2b.md) until a qualified
cutover; this recipe does not identify an installed E4B deployment or a
performance winner.

## Pinned model and distinct companion candidates

| Role | Public repository | Revision | Architecture |
| --- | --- | --- | --- |
| Target | [DavidAU E4B DECKARD V2](https://huggingface.co/DavidAU/gemma-4-E4B-it-The-DECKARD-V2-Strong-HERETIC-UNCENSORED-Thinking) | `2c91d3253385fe92a898799a93b94ce806b8c840` | Full BF16 Gemma4 conditional generation |
| Native assistant | [Google E4B assistant](https://huggingface.co/google/gemma-4-E4B-it-assistant) | `8d0031ea8c2109e2b1e86bb9368a4539b537f80a` | Approximately 159 MB, public HF Gemma4 assistant with borrowed target hidden/shared KV |
| EAGLE3 | [slinhug EAGLE3 E4B 32K drafter](https://huggingface.co/slinhug/eagle3-e4b-drafter-32k) | `ba0c76216c11f03da3d668d66673e7b2acfc75ca` | Approximately 1.745 GB, own embedding/head and independent one-layer sliding KV |

These two heads are distinct architectures; neither is DFlash or an E2B DFlash2
companion. The native assistant uses the pinned HF implementation and the
verifier's committed hidden/shared-KV state. EAGLE3 consumes three auxiliary
states (embedding-inclusive indices 2/21/39, decoder blocks 1/20/38), uses its
own full embedding and reduced 32K head, and maps draft IDs into target IDs
using the checkpoint's `draft_id + d2t[draft_id]` offsets. Its `t2d` tensor is a
membership mask. The EAGLE checkpoint does not declare a weight license; these
notes do not grant redistribution rights. Model files remain external to the
source tree.

Both proposal paths use their actual deterministic proposal law with standard
rejection sampling when the target samples. Candidate size, architectural labels
and correctness tests do not select a throughput winner. Compare complete
prefill/decode workloads and memory use on the same target/runtime before
choosing a head or proposal depth.

## Thinking and sampling contract

The shipped `chat_template.jinja` starts by setting `enable_thinking=true`.
This overrides an ordinary template argument; TensorFold preserves the shipped
policy and returns HTTP 400 for explicit thinking-off requests. The repository
also contains `chat_template-instruct.jinja`, but it is not selected. Do not
silently substitute it or apply the 26B deployment's template override.
Thinking output consumes the reply budget, so allow enough tokens for both
reasoning and the final answer when assessing quality.

This snapshot ships no `generation_config.json`. Omitted API sampling controls
therefore retain the application's greedy defaults. Benchmark preparation with
`--generation-policy hf-defaults` records the pinned library's conditional
sampling settings in the explicit `sampled_hf_defaults` region. Those values are
an experiment configuration, not a publisher recommendation or a change to
omitted API settings. Use thinking on for that checkpoint.

HF 5.17 emitted a Mistral-regex warning because the tokenizer metadata lacks a
root `transformers_version`, triggering its heuristic. The complete tokenizer's
structural identity with the official Gemma4 base was verified; the warning was
dispositioned as a false positive. No regex, tokenizer or checkpoint mutation
was applied in response.

## Media and evidence scope

With `--vision`, the same frontend supports images, timestamped video frames
and mono 16 kHz audio input. Audio is limited to 30 seconds per segment, and
replies are text. Video ingestion supplies visual frames by default; send the
soundtrack as a separate `input_audio` part. Optional native HF automatic
soundtrack extraction is not exposed. Encoded/decoded media, context and request
admission limits remain in force. See the [API contract](../api.md).

All 2,130 target tensors are BF16. Actual metadata and header checks confirmed
complete audio and vision towers, the audio projector from 1536 encoder features
to text width 2560, and the vision projector from 768 to 2560. Native processor
checks passed 38 cases with no skips against the actual checkpoint metadata.

Managed source-mounted GB10 checks through the protected SDK passed short
native-model logits, verification widths and cache replay. The native Google
assistant separately passed native proposal/shared-KV/hidden comparisons at
prompt lengths 511/512/513, serial/drafted 64-token identity and sampled cache
replay. EAGLE3 actual-weight GPU qualification and comparative performance are
still pending; CPU contract checks alone do not establish either result.

Selected image/audio/video fixtures completed through the protected managed
source-overlay runtime. Image and audio output IDs matched native HF generation;
the image answer identified the correct shapes and positions, while ASR had
transcription errors. Video final answers differed: the adapter identified the
known motion on this fixture, while the native batched reference did not. The
adapter's reasoning also invented an overlap absent from the fixture. These
observations establish neither general quality superiority nor numerical
identity between different encoder batching shapes. Cold/cache replay IDs agreed.
No incidental token rate is presented as an optimization winner.

These are source-mounted/overlay results, not installed E4B wheel or live-service
qualification. An installed-image receipt, full serving checks and a measured
head/setting selection must be recorded before describing an E4B cutover as
qualified. Preserve the protected CUDA entrypoint/provider checks and the E2B
rollback service while evaluating this candidate.

## Offline mixed-precision quantization work

`tensorfold quantize plan ./original-e4b-bf16` prints the versioned, local
header-only NVFP4 plan. Its initial quality recipe retains the embedding tables,
multimodal components, norms, output head and KV at their existing precision.
The assistant plan initially selects only 22 ordinary projections; its custom
ordered vocabulary head remains BF16. These are explicit mixed-precision plans,
not uniform four-bit exports. FP8 component alternatives require their own
calibration, packed-consumer and quality gates.

The [quantization guide](../quantization.md#offline-nvfp4-planning) separates the
implemented planner from pending conversion, producer-container compatibility,
packed E4B execution and quantized-model qualification. Original BF16 weights
remain the quality reference.
