# DeepSeek-V4.1-Flash on eight V100s (TP8/ETP4)

TensorFold's CUDA lane for DeepSeek-V4.1-Flash-1017 on eight V100 (SM70) GPUs, without a vLLM runtime: attention and
dense layers TP8, routed experts ETP4 x EP2. It serves the OpenAI-compatible server of the other V100 lanes.
Status: development lane, qualified on one host only (SC117: 8x V100 32 GB, 180 W cap). Image
`tensorfold-v100:py315-app-48c51356-pillow13` (`67b43e682057`).

## The model

- Backbone: GSQ-RCO IQ2_XXS GGUF (four shards, routed experts IQ2_XXS / IQ2_XS / IQ1_M, dense Q8_0 / Q5_K).
- Engram: the separate IQ4_NL Engram GGUF (103 GiB). It is mapped into **host RAM only** and registered read-only
  with every GPU. Only the rows a step uses reach the GPU, in a 96 MiB per-step working buffer. VRAM residency is 0 B.
- Model directory: the base `config.json` plus `quantization_config: {"quant_method": "gguf"}`,
  `gguf_file` (first backbone shard) and `engram_file`, with `tokenizer.json`, `tokenizer_config.json` and
  `chat_template.jinja` copied (not symlinked) into the directory. Paths outside the model root are refused.

## Launch

```sh
python tools/launch_cuda_tp.py --gpus 0,1,2,3,4,5,6,7 -- /models/ds41-tf --backend cuda \
  --name deepseek-v4.1-flash --parallel 32 --context 65536 --prefix-host-bytes 17179869184 --port 8000
```

The container needs `--ipc host --ulimit memlock=-1:-1 --cap-add IPC_LOCK` for the host-mapped Engram table and the
pinned prefix tier. Startup with warm extension caches takes about 105-111 s, including about 47 s to map the Engram
table and 17 s to capture the decode graphs. A first start that has to JIT-build the native extensions took up to
about 520 s.

| Flag | DeepSeek-V4.1 SM70 |
| --- | --- |
| `--parallel N` | 1-32 concurrent streams. Requests beyond N wait in the queue. |
| `--context N` | Admitted context per stream (default: the model's maximum). The KV pool is shared, so `--context 65536 --parallel 32` gives a 7.78M-token pool. |
| `--prefix-host-bytes B` | Pinned host tier for evicted prefix points, shared across ranks (Gemma 4 SM70 sizing; `0` disables). |
| `--queue-timeout-s S` | Bounds the queue wait (503 after). |
| `--no-graphs` | Eager decode, for A/B checks only. |
| `--vision` | Image input (needs `mmproj_file` and `vision_bias_file` in `config.json`; see "Image input"). |

## What runs

- Decode: startup CUDA graphs for batches 1-16, 24 and 32. Capture happens at startup, never lazily. Each batch has
  index-score classes 8192 / 32768 / `--context`, sized by the batch's live maximum length. Decode also uses small
  all-gathers and an LL row all-reduce for small sums, a compact device greedy argmax with batched round sampling
  (one host wait per round), and expert-major routed experts (the friend's EM3 kernels, bit-exact with the per-6-row
  op) below 128 rows.
- Prefill: the friend's TurboMind lattice grouped GEMMs for routed experts at 128 or more rows. They use TensorFold's
  exact weight decode with FP32 gate/up, which is bitwise equal to the earlier grouped op. The friend's FP16 lattice
  arithmetic was removed because it had 2.6x the operator error. Prefill also uses fused sparse attention (the
  friend's patch 181 at 8 heads per rank) and dmv13 dense GEMMs with exact FP32 Q5_K restore coefficients.
- Scheduling: decode runs during prefill in automatic layer groups under the 175 ms policy, with half-layer prompt
  units.
- Prefix cache: points are kept only at V4.1 message boundaries; 8192-token chunk ends are not kept. The newest
  conversation's last-message point is protected. Evicted points spill to the host tier and are restored
  byte-exactly. On a 13-turn agent session, total time was 16.3 s cached against 81.0 s cold.

## Measurements

10-cell matrix (`tools/matrix_benchmark.py`, plan `plan-ds41-v1`, greedy, one rep, 180 W, `--parallel 32 --context
65536 --prefix-host-bytes 16G`; source ds41-int3 e68d06a3). The friend's 1Cat-vLLM container numbers are his frozen
runs on the same host at 180 W. Each cell shows TensorFold vs 1Cat.

| Cell | Prefill tok/s | Decode tok/s per user | Output tok/s, whole run |
| --- | --- | --- | --- |
| C1 4K | 2925 vs 2383 | 63.0 vs 60.6 | 54 vs 50 |
| C1 16K | 3044 vs 3305 | 62.0 vs 59.0 | 38 vs 38 |
| C8 4K | 2606 vs 1484 | 28.2 vs 35.3 | 159 vs 127 |
| C8 16K | 2823 vs 1460 | 21.6 vs 32.0 | 68 vs 42 |
| C16 4K | 2453 vs 1719 | 18.2 vs 24.0 | 189 vs 164 |
| C16 16K | 2695 vs 1616 | 13.9 vs 27.1 | 72 vs 48 |
| C24 4K | 2734 vs 1839 | 13.3 vs 17.3 | 212 vs 183 |
| C24 16K | 2772 vs 1943 | 9.6 vs 16.4 | 75 vs 58 |
| C32 4K | 2215 vs 1936 | 11.0 vs 14.5 | 205 vs 196 |
| C32 16K | 2551 vs 1636 | 8.1 vs 27.2 | 73 vs 50 |

TensorFold streams every admitted user at once, so per-user decode at 8 or more concurrent requests is lower while
whole-run output is higher in every cell. C1 16K prefill is the one cell below the reference (-8%).

## Quality gate

Every numerics-affecting change is judged with `tools/quality_eval` on `corpus-v1.ds41` against the baseline of the
first build that passed, using the **confidence interval of the NLL delta**, not KL. Any bit-level change shows
KL of about 1.8e-2 and per-token |d| up to about 0.015 on this corpus (a chaotic floor). For the final tree, prefill
d was -0.0013 (CI [-0.0091, +0.0055]) and decode d +0.0022 (CI [-0.0008, +0.0058]), both within the floor. As a cheap
smoke test, known answers scored 9/10 with thinking off and 10/10 with thinking on.

## Image input (`--vision`)

`--vision` serves OpenAI `image_url` content parts (data URLs; public HTTPS URLs also with `--vision-urls`) in user
messages and tool results. The model directory's `config.json` names the mmproj and the vision router bias:

```json
"mmproj_file": "/models/ds41-gsq-rco/mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf",
"vision_bias_file": "/models/ds41-gsq-rco/exp_probs_b_vl.f32"
```

- Prompts: DeepSeek's V4.1 encoder renders each image part as one `<｜deepseek_image｜>` (id 129264) in block order,
  the message's parts joined with `\n\n`. Request text that spells the placeholder is refused. Each placeholder
  expands to the image's span, `[START] ([IMAGE] * w + [NEWLINE]) * h + [END]`, at most 1024 tokens per image
  (upstream's resize plan, Pillow `ImageOps.pad` with grey 127). Unlike the friend's server, EXIF orientation is
  applied before the plan. `--vision-max-images` (default 4) and `--vision-image-tokens` (default 4096) bound a
  request.
- Tower: the mmproj ViT (32 x 1024) and 3 x 3 aligner on GPU 0 of rank 0, the friend's patch-150 numerics (BF16
  weights widened exactly to FP16, FP16 GEMMs with FP32 accumulation, FP32 residual, norms and RoPE). It is loaded
  before the KV pool is sized and warmed at startup on the plan's largest image. Rank 0 encodes and broadcasts
  each prompt chunk's image rows to the other ranks. While other streams decode, an encode runs in groups of
  tower blocks under the 175 ms policy, as prompt chunks do.
- Backbone: image rows replace the image token's embedding. They route their experts with the vision bias
  (`exp_probs_b_vl`), and they are dead Engram n-gram members (`valid & ~image`), as upstream does. Text-only
  steps run exactly the text model's kernels.
- Prompt cache: an image span's positions carry ids derived from the image's pixels (sha256), so a kept point is
  reused only by a prompt with the same images. Points stay at message boundaries. Prompt scoring
  (`/v1/tensorfold/score`) stays text only.

## Diagnostics (never for serving or benchmarks)

- `TENSORFOLD_DS41_ENGRAM_TRACE=FILE`: rank 0 appends each forward's Engram row ids to FILE. **It disables the decode
  graphs**, so it is for offline hit-rate analysis only.
- `TENSORFOLD_DS41_SCHED_TRACE=1`: prints the Decoder's layer-group plans and costs.
- `TENSORFOLD_DS41_SMOKE_LAYERS=N`: loads only the first N layers (N=1 skips the Engram mapping) for plumbing
  checks. Outputs are meaningless.

## Not yet

DSpark drafting is not merged into the serving lane. Startup time and
the C1 16K prefill gap are open optimization items.
