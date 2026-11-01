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

Video execution is not blanket semantic-quality qualification. Directed
red-object/rightward-motion and fixed-blue-object questions passed; a generic
caption was inaccurate, and a thinking-on diagnostic exhausted its 768-token
budget without a final answer. These limits remain recorded.

The earlier AEON dense 31B attempt is not the current deployment target. Its
publisher checkpoint contained a nonfinite scale; no NaN repair or weight-block
scrubbing was enabled. See the [historical 31B investigation](gemma-4-31b.md)
and [preserved E4B rollback deployment](gemma-4-e4b.md).
