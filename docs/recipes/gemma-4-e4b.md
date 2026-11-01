# Gemma 4 E4B on CUDA

## Historical GB10 deployment / retained rollback: 2026-11-01

This deployment has been replaced by the standalone
[26B-A4B ModelOpt service](gemma-4-26b-modelopt.md). `gemma4-e4b-tf` and its
image remain stopped and preserved for rollback. The following configuration
and results describe that earlier installed image, not the current service.

Previously, `gemma4-e4b-tf` served port `8888` using the 1,024-sample mixed NVFP4/BF16 E4B
export and the original BF16 Google assistant. Three proposals/four verification
rows are the balanced deployment setting. BF16 KV, one request lane, the native
131,072-token context and image/video/audio input remain enabled. Thinking is
fixed on by the checkpoint. Both previous client IDs, `gemma4-26b` and
`gemma4-e2b`, remain aliases. This is a deployed setting, not a universal
performance winner; further optimization continues.

The canonical derived model is
`/models/local/gemma4-e4b-deckard-v2-strong-nvfp4-nlh1024-v3`, mounted read-only
as `/models/e4b`. It was moved atomically without recopying or changing file
inodes, sizes or bytes; the original BF16 source remains separate. Its manifest
SHA256 is `7ba166232ab502830beed25fccd255c4975aa6dc767c20a792367dde20effdb6`.
Image
`sha256:2326daf164b0cf0598dcbdde987f07bd499dc1206cec61d0568aa3a76bb78927`
contains the complete installed wheel and an owned allocator adapter. Exact
source UID and adapter hash are in the
[deployment record](../../deploy/gb10/deployment-gemma4-e4b-20261101.json).
The source capture includes uncommitted integration code, so commit `d0ad6c24`
alone does not reproduce it. P01 scalar preparation was archived/reverted
before capture and is not part of this deployment. No serving source overlay or
installed vendor patch is used.

The historical command supplied to the image's protected entrypoint was:

```sh
serve /models/e4b --backend cuda --precision checkpoint \
  --name gemma4-e4b --alias gemma4-26b --alias gemma4-e2b \
  --host 0.0.0.0 --port 8888 \
  --drafter /models/hf/hub/models--google--gemma-4-E4B-it-assistant/snapshots/8d0031ea8c2109e2b1e86bb9368a4539b537f80a \
  --drafter-bits 16 --draft-rows 4 --parallel 1 --context 131072 \
  --kv-dtype bf16 --vision --thinking --thinking-budget 0
```

Retain the protected CUDA bootstrap and baked allocator adapter; invoking raw
Python bypasses their contracts. The configured Torch CUDA ceiling is 32 GiB,
the CPU cgroup is 20 GiB with no extra swap, and the normal host reserve is
16 GiB. Ordinary admission remains active; no blanket 76 GiB free-memory
requirement is imposed. Actual installed-candidate execution consumed 131,056
prompt tokens plus 16 generated tokens with 13 drafted and 10 accepted proposals,
while E2B was still running. Minimum observed host availability was 25.09 GiB.
EOS was ignored only for that capacity exercise; this is neither retrieval
quality nor a throughput benchmark.

Installed API checks passed schema output, tools, Responses, SSE, tokenization,
cancellation/recovery and explicit thinking-off HTTP 400. Image/audio/video
requests executed and serial/drafted reply hashes matched. The video fixture
answered the motion question incorrectly: parity and pipeline completion do
not establish video quality. Post-cutover health and a legacy-alias request
returning `4` passed. The old `gemma4-e2b-tf` container, image, cache and launch
configuration are stopped and preserved for rollback. Other services were not
changed. Broader quality acceptance and optimization are ongoing.

## Pinned model and distinct companion candidates

| Role | Public repository | Revision | Architecture |
| --- | --- | --- | --- |
| BF16 source / quality reference | [DavidAU E4B DECKARD V2](https://huggingface.co/DavidAU/gemma-4-E4B-it-The-DECKARD-V2-Strong-HERETIC-UNCENSORED-Thinking) | `2c91d3253385fe92a898799a93b94ce806b8c840` | Full BF16 Gemma4 conditional generation |
| Historical deployed BF16 assistant | [Google E4B assistant](https://huggingface.co/google/gemma-4-E4B-it-assistant) | `8d0031ea8c2109e2b1e86bb9368a4539b537f80a` | Approximately 159 MB, public HF Gemma4 assistant with borrowed target hidden/shared KV |
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

## Media and earlier BF16 development evidence

With `--vision`, the same frontend supports images, timestamped video frames
and mono 16 kHz audio input. Audio is limited to 30 seconds per segment, and
replies are text. Video ingestion supplies visual frames by default; send the
soundtrack as a separate `input_audio` part. Optional native HF automatic
soundtrack extraction is not exposed. Encoded/decoded media, context and request
admission limits remain in force. See the [API contract](../api.md).

The original source checkpoint has 2,130 BF16 tensors; this paragraph describes
that source, not the mixed-precision deployed artifact. Actual metadata and header checks confirmed
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

Those earlier results were source-mounted/overlay checks, not installed-wheel
or live-service qualification. The subsequent installed checks and retained
limitations are recorded in the deployment section above. Preserve the
protected CUDA entrypoint/provider checks and the E2B rollback artifacts.

## Offline mixed-precision quantization work

`tensorfold quantize plan ./original-e4b-bf16` prints the versioned, local
header-only NVFP4 plan. Its initial quality recipe retains the embedding tables,
multimodal components, norms, output head and KV at their existing precision.
The assistant plan initially selects only 22 ordinary projections; its custom
ordered vocabulary head remains BF16. These are explicit mixed-precision plans,
not uniform four-bit exports. FP8 component alternatives require their own
calibration, packed-consumer and quality gates.

The [quantization guide](../quantization.md#offline-nvfp4-planning) describes the
public planning interface. The deployed artifact was converted through the
separate experimental producer; this does not introduce a public conversion
CLI. Original BF16 weights remain the quality reference, and broader quantized
quality acceptance is still open.

## Packed ModelOpt NVFP4 storage contract

The same dense CUDA engine accepts a complete local E4B export with
`quant_method=modelopt`, `quant_algo=NVFP4`, native static W4A4 calibration,
all 258 active text attention/MLP projections packed, and the schema1
`tensorfold_quantization_manifest.json`. All exported tensor and consumed
metadata hashes are checked by the existing dense loader before publication.
The 54 unused shared-tail tensors must follow the manifest's explicit omitted
or retained-original-BF16 policy. Embeddings, PLE, tied head, media encoders and
projectors, learned norms, RoPE and KV retain the original BF16/native contract.
FP8 projection/table variants and transformed/quantized-KV exports are refused.
Packed preparation is restricted to the original E4B vision/audio allocation
profile. It checks the complete native META topology against headers, then
binds actual checkpoint payloads into the parameter skeleton without a full
CPU BF16 parameter or tied-head mirror. Nonpersistent native buffers use real
native initialization; META values are never used as semantic model data.

Use `--precision checkpoint`; this path does not support BF16 projection
arithmetic overrides or whole-weight dequantization. It shares the original
single-lane engine, native processor/template, assistant interface and prefix
lifetime. A target `native_reference()` context preserves the packed weights
while using native HF attention/norm routing; it is not a BF16-weight oracle.

The installed deployment above passed bounded speculative-law, capacity and
API checks; these do not establish every workload's quality or a performance
winner. Header capacity estimates include prepared packed buffers, simultaneous
packing and validation owners, bounded read/metadata scratch, and real native
nonpersistent buffers. The current META loader allocates no full CPU parameter
or tied-head placeholder. These bounds grant no allocator-reuse credit;
allocator/driver/runtime overhead still requires live host/CUDA admission and
the configured allocation ceiling. Dated deployment results above retain their
original image and loading-evidence scope.
