# DeepSeek-V4.1-Flash on eight V100s (TP8)

TensorFold's CUDA lane for DeepSeek-V4.1-Flash-1017 on eight V100 (SM70) GPUs uses TP8 for attention,
dense layers, and shared experts. Routed experts default to ETP4 × EP2. It serves the
OpenAI-compatible server of the other V100 lanes.
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

This launch serves exact chat counting, the active renderer's capabilities, and telemetry through
the standard [CUDA API](../api.md). Select the **TensorFold** provider in Harness.

Absent request, launch or checkpoint overrides, this lane uses temperature 1.0
and top-p 1.0 with or without tools. [1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/README.md)

Set `temperature: 1.0` and `top_p: 0.95` explicitly, or launch with
`--temperature 1.0 --top-p 0.95`, to use the reported coding-agent settings. [V4.1 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)

A shared runtime image can set `TENSORFOLD_DS41_EXPERIMENT_W576_FP32` or
`TENSORFOLD_DS41_EXPERIMENT_SP_FP32` for the compact 128K RotorQuant profile
and still launch the native FP8-window/FP4-compressed cache. Those settings
apply only on RotorQuant, where the existing compact profile checks still run.

Add `--expert-tensor-parallel-size 1` to choose ETP1 × EP8 at startup.
Each GPU then holds 48 whole routed experts. Use `--expert-tensor-parallel-size 2`
to choose ETP2 × EP4, with 96 routed experts per GPU sharded across each pair.
Use `--expert-tensor-parallel-size 4` or omit the option to choose the default.
All eight ranks must use the same degree. Check the effective value in
`model.expert_tensor_parallel_size` from `/v1/stats`.
ETP1 and ETP2 are experimental while native quality, memory, and throughput qualification is in progress.

The container needs `--ipc host --ulimit memlock=-1:-1 --cap-add IPC_LOCK` for the host-mapped Engram table and the
pinned prefix tier. With warm extension and page caches, the C24/C32 matrix server became ready in about 45 s.
The C1/C8/C16 server took about 85 s with disk reads during startup. The earlier startup study measured 47-48 s
warm, down from 144-375 s, and 126 s with model pages dropped, down from 166 s. Every rank maps the Engram table
read-only straight from the GGUF file (its page-cache pages are the one
host copy; no second 103 GiB copy) and registers it on a thread while the weights load through pinned staging
buffers; the host tier's slab pins beside the warm-up; the decode graphs take about 9 s. Each rank prints one
`startup (...)` line with every phase's start, wall time, CPU time and storage reads. A first start that has to
JIT-build the native extensions builds all six at once (about 210 s of compilation, measured on the CPU, against
713 s one after another). The model file must not be rewritten in place while a server runs: its pages are the
registered Engram table.

| Flag | DeepSeek-V4.1 SM70 |
| --- | --- |
| `--parallel N` | 1-32 concurrent streams. Requests beyond N wait in the queue. |
| `--context N` | Admitted context per stream (default: the model's maximum). The text-only matrix at `--context 65536 --parallel 32` reported 7,775,360 tokens in the shared KV pool. |
| `--prefix-host-bytes B` | Pinned host tier for evicted prefix points, shared across ranks (Gemma 4 SM70 sizing; `0` disables). |
| `--kv-dtype rotorquant-iso64-norm4-h64` | RotorQuant Q4 for both sliding-window and compressed attention latents. DeepSeek stores one 272-byte row for shared K and V. The learned index rows retain their separate format. |
| `--kv-dtype fp8-window+fp4-compressed` | DeepSeek's native cache path uses FP8-quantized sliding-window values and packed FP4 compressed attention latents. The historical `bf16` value selects this same path. |
| `--yarn-factor F` | Overrides the checkpoint's static YaRN factor on compressed layers. Window-only layers retain plain RoPE. |
| `--queue-timeout-s S` | Bounds the queue wait (503 after). |
| `--no-graphs` | Eager decode, for A/B checks only. |
| `--vision` | Image input (needs `mmproj_file` and `vision_bias_file` in `config.json`; see "Image input"). |

The 128K text profile uses `--kv-dtype rotorquant-iso64-norm4-h64 --context 131072` with the
checkpoint's native YaRN factor. With CUDA graphs enabled and the standard prompt chunk size,
ETP4 H64 text with an admitted context of at most 128K reserves 2.5 GiB on each GPU for prompt work.
ETP1 and ETP2 use a 3 GiB prompt reserve.
The KV allocator budgets for decode graphs and uses the available memory on the tightest GPU
to size the shared pool. `/v1/stats` reports the effective codec, YaRN factor, and KV budget.

### Attention FP8 grouping

DeepSeek applies E4M3 FP8 quantization with one power-of-two UE8M0 scale per
32 channels at two boundaries. The model-native sliding-window path quantizes
the full row after RoPE, including the rotary tail. The attention output is
quantized after inverse RoPE, before the first output projection. These
boundaries follow the [official 1017 inference reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py).
The V100 provider simulates them in FP16 with software conversions. The
reference uses BF16 around these operations.

H64 cold prefill uses the same grouping when it computes an FP8 window
operand. The H64 ring and main cache encode the selected RotorQuant format
directly. Learned index rows and compressed latent encoding retain their
existing paths. Full-model throughput and quality of the grouping correction
remain to be verified.

### Experimental decode graph memory allowance

The integrated candidate reserves 1,536 MiB per GPU for the exact ordered
48-key catalogue in the full compact ETP4 text profile. It uses 32 lanes,
128K admission, 8,192-row prompt chunks, native YaRN 16 and default kept
slots. Request the 16 GiB host prefix tier explicitly so all ranks agree
on 2 GiB each and any pin refusal aborts startup. The prompt reserve
stays at 2.5 GiB per GPU.

```sh
python tools/launch_cuda_tp.py --gpus 0,1,2,3,4,5,6,7 -- /models/ds41-tf --backend cuda \
  --name deepseek-v4.1-flash --parallel 32 --context 131072 --prefix-host-bytes 17179869184 \
  --kv-dtype rotorquant-iso64-norm4-h64 --expert-tensor-parallel-size 4 \
  --expert-balance-state /state/expert-balance-etp4.json --port 8000
```

`shape_keys` in `src/tensorfold/families/deepseek_v4/cuda/graphs.py` owns
the capture catalogue. Pool sizing deducts the allowance before startup
and uses it to check actual graph memory after startup capture and live
recapture. Other captured profiles use `GRAPH_BYTES` plus
`GRAPH_EACH_BYTES` for each capture. The generic allowance also applies
when the keys change or move to a different order, including changes that
keep 48 captures.

The standalone `f9606970` pressure run observed 11,554,240 KV tokens
with all 48 graphs. The `dcce7d0c` live ETP4 transition passed at the
generic 2,048 MiB allowance with a measured pool of 10,922,624 tokens.
The integrated candidate's capacity, live recapture and performance
remain to be measured.

To expand or replace captures, requalify `H64_128K_GRAPH_KEYS` and
`H64_128K_GRAPH_BYTES` in `src/tensorfold/families/deepseek_v4/cuda/engine.py`.
Record CUDA driver free bytes and allocator allocated and reserved bytes
on all eight ranks through startup, old-graph retirement, bank transfer,
capture and recapture. Verify the actual KV pool and worst-rank memory
during late 128K prefill overlapping decode and during full scoring.
Keep the prompt reserve separate. Recheck paired perplexity and serving
across C1, C8, C16 and C32 at 4K and 16K before selecting a new allowance.

## Routed expert placement

`--expert-placement PLAN` loads a version 1 JSON placement at startup on TP8
with ETP1, ETP2, or ETP4. Each layer lists all 384 logical expert IDs exactly
once in physical slot order. Each consecutive group of 48, 96, or 192 IDs
belongs to one expert-parallel group for ETP1, ETP2, or ETP4, respectively.
The ranks in a group keep their original tensor-parallel cuts of those experts.

The plan carries `version`, `checkpoint`, `etp`, and `placement` fields.
`checkpoint` binds the local GGUF files and model configuration. Rank 0 reads
the plan once and every rank verifies the same plan before loading weights or
capturing graphs. Configured placement starts with a fresh worker process and
fresh KV state. Native qualification is pending for this development option.

Live expert balancing samples executed routed experts in prefill and decode
separately. The history decays with the number of observed routes. The
development defaults sample one in 64 forwards on average and use 262,144
observed routes per phase and layer as the half-life. Use the corresponding
`--expert-balance-sample-every` and `--expert-balance-half-life` options for
controlled qualification. Startup warm-up and graph padding are excluded.
When both phases have enough fresh observations, the scheduler can apply
eligible proposals in the running server. It stores the committed placement and rolling history in
`expert-balance-etpN.json` in the model directory, where `N` is the expert
tensor-parallel degree. A restart loads that placement before the workers
load the expert weights or capture graphs. For a read-only model mount, use
`--expert-balance-state FILE` with a writable directory for the sidecar.
In a container, mount that directory on a persistent volume so atomic file
replacement and restart recovery use the same storage.
Use `--expert-placement` to seed a new state file. For later restarts, omit
that startup option to load the committed sidecar placement. If supplied,
the explicit plan must agree with the saved placement.
Use `--no-expert-balance` to disable balancing. This also disables history
collection and loading of a saved placement.

The scheduler holds new admissions while
active requests finish. The workers retire the previous placement's cached
prefixes, move the packed expert banks, and rebuild the enabled decode graph
catalogue. Admission resumes after all eight ranks confirm the placement and
the new epoch is saved in the state file. A failed post-mutation transaction
retires the cohort. ETP4 completed the native transition with the generic
allowance. Validation with the 1,536 MiB allowance and performance
qualification remain pending.

## What runs

- Decode: CUDA graphs for batches 1-16, 24 and 32, captured at startup and recaptured at a live placement
  boundary. Each batch has
  index-score classes 8192 / 32768 / `--context`, sized by the batch's live maximum length. Decode also uses small
  all-gathers and an LL row all-reduce for small sums, a compact device greedy argmax with batched round sampling
  (one host wait per round), and expert-major routed experts (the friend's EM3 kernels, bit-exact with the per-6-row
  op) below 128 rows.
- Prefill: the friend's TurboMind lattice grouped GEMMs for routed experts at 128 or more rows. EM3 is faster up to
  about 320 rows, but its Q8_1 activations are less accurate, so the lattice keeps those chunks. They use TensorFold's
  exact weight decode with FP32 gate/up, which is bitwise equal to the earlier grouped op. The friend's FP16 lattice
  arithmetic was removed because it had 2.6x the operator error. Prefill also uses fused sparse attention (the
  friend's patch 181 at 8 heads per rank) and dmv13 dense GEMMs with exact FP32 Q5_K restore coefficients.
- Scheduling: decode runs during prefill in automatic layer groups under the 175 ms policy, with half-layer prompt
  units.
- Sequence-parallel prefill: at 1,024 forward rows and above, ranks exchange global expert IDs and their
  probabilities before translating the routes to their own expert banks. The routing correction in `1d763a3d`
  applies to text and mixed image prompts. A rank that owns a sequence shard can belong to either expert group.
- Prefix cache: points are kept only at V4.1 message boundaries; 8192-token chunk ends are not kept. The newest
  conversation's last-message point is protected. Evicted points spill to the host tier and are restored
  byte-exactly. On a 13-turn agent session, total time was 16.3 s cached against 81.0 s cold.

## Measurements

10-cell matrix on SC117 (`tools/matrix_benchmark.py`, frozen `plan-ds41-v1`, 512 output tokens per request, greedy,
one server session with one repeat per cell, 180 W, image `67b43e682057`, `--parallel 32 --context 65536
--prefix-host-bytes 16G`, corrected source `aef6e42a`). Ten requests across five cells rendered 2 to 27 tokens above the
planned length. The reference is the frozen 1Cat-vLLM deployment on the same host at 180 W with its configured
1,048,576-token context and the same benchmark plan. The first three measurement columns show TensorFold vs 1Cat.
These measurements describe the tested deployments and workloads. One repeat per cell leaves between-session
uncertainty unresolved.

| Cell | Prefill tok/s | Decode tok/s per user | Output tok/s, whole run | TensorFold p99 token gap (ms) |
| --- | --- | --- | --- | ---: |
| C1 4K | 3884 vs 2383 | 71.1 vs 60.6 | 62 vs 50 | 15.6 |
| C1 16K | 4293 vs 3305 | 68.2 vs 59.0 | 45 vs 38 | 16.1 |
| C8 4K | 3537 vs 1484 | 36.3 vs 35.3 | 208 vs 127 | 168.1 |
| C8 16K | 4051 vs 1460 | 29.5 vs 32.0 | 95 vs 42 | 164.2 |
| C16 4K | 3368 vs 1719 | 22.1 vs 24.0 | 240 vs 164 | 170.3 |
| C16 16K | 3979 vs 1616 | 21.2 vs 27.1 | 101 vs 48 | 173.7 |
| C24 4K | 3188 vs 1839 | 16.1 vs 17.3 | 251 vs 183 | 172.2 |
| C24 16K | 3817 vs 1943 | 12.2 vs 16.4 | 103 vs 58 | 174.3 |
| C32 4K | 3062 vs 1936 | 13.2 vs 14.5 | 263 vs 196 | 173.4 |
| C32 16K | 3730 vs 1636 | 10.1 vs 27.2 | 104 vs 50 | 177.5 |

All admitted TensorFold replies streamed concurrently. Whole-run output rose in all ten cells, while median
per-user decode was lower in seven. The largest observed TensorFold token gap was 308.5 ms at C24 4K.

### Fixed-order prefill RMS

The fused prefill normalization preserves the FP32 reduction order of ATen `05adb434` and FP16 rounding. Against
`aef6e42a`, source `3e86f455` had bitwise-identical target log probabilities across all twelve scoring checks,
including the 268-document cap64 corpus and the 32-document uncapped long-prompt corpus. On the latter, prefill perplexity
was 5.361628 and full-decode perplexity was 3.849869 in both arms. All 35 native normalization tests passed.
Fixed-schedule FP32 logits also matched on all 15 tested paths.

Four independent SC117 sessions ran in reference, candidate, candidate, reference order, with four additional
reference sessions for uncertainty calibration. The load used the frozen ten-cell greedy plan, 512 output tokens
per request, 180 W per V100, image `67b43e682057`, 32 lanes, a 65,536-token context and a 16 GiB host prefix tier.
The usable KV pool was 7,775,360 tokens in each run. Each row shows the candidate's change from the reference.

| Cell | Aggregate prefill | Median decode per user |
| --- | ---: | ---: |
| C1 4K | +1.11% | −0.03% |
| C1 16K | +1.33% | −0.13% |
| C8 4K | +1.39% | +0.15% |
| C8 16K | +1.28% | +0.35% |
| C16 4K | +1.20% | +0.40% |
| C16 16K | +1.19% | +2.49% |
| C24 4K | +1.45% | +0.43% |
| C24 16K | +1.25% | +0.56% |
| C32 4K | +1.27% | +0.57% |
| C32 16K | +0.63% | −8.33% |

The geometric mean prefill gain was 1.21% (95% interval +0.92% to +1.51%). Six prefill cells and C32/4K
decode gained under the familywise comparison. The original C32/16K decode estimate had an adjusted interval
from −22.29% to +8.14%. A fresh six-cell repeat measured −0.23% at C32/16K, with an adjusted two-endpoint
interval from −18.56% to +22.23%. That result remains uncertain. The largest text-cell p99 gap across the four
full sessions was 177.64 ms.

The original and fresh sessions shared the same C1 startup prefix and produced identical replies. The fresh C1/16K
estimate was −0.125% user-visible decode throughput, with an adjusted interval from −0.423% to +0.175%.
The combined estimate was −0.129%, with an adjusted interval from −0.239% to −0.019%.
The project accepted this specific tradeoff. In four instrumented repeats, the measured difference was inside
GPU round events. The trace did not identify a practical RMS-specific correction.

Four separate vision sessions passed all seven image checks, cached image reuse, and both 8192×1 and 1×8192
inputs. All four retained 7,697,600 usable KV tokens and a maximum rank warm tower peak of 235 MiB. The two
mixed-load prefill results and their p99 gap comparisons remain inconclusive.

## Quality gate

Two `tools/quality_eval` repeats on `corpus-v1.ds41` were bitwise identical to each other and to the earlier
`a7ba45a4` result. This corpus has a maximum prompt of 496 tokens, below the sharding threshold. Against the `e68d06a3` measurement
for the previous published lane, the 95% intervals for the NLL delta were [0, +0.00388] prefill and [0, +0.00112]
decode, both inside the predeclared 0.005 nats/token equivalence margin. Against the initial `p1` baseline,
prefill d was -0.00015
(CI [-0.00782, +0.00726]) and decode d was +0.00271 (CI [-0.00032, +0.00622]). Both comparisons with `p1` were
inconclusive at that margin. The known-answer check scored 9/10 with thinking off and 10/10 with thinking on.

A larger paired teacher-forced comparison used the frozen 268-document v2 corpus, tokenized with the DeepSeek
V4.1 renderer. It compared historical `fa3ca06e` with corrected `aef6e42a` on SC117, image `67b43e682057`,
`--parallel 8 --context 65536`, and the same GSQ-RCO checkpoint and tokenizer. Each continuation was capped at
64 tokens. Its first continuation token, scored by the prompt-end head, is counted under prefill. Intervals
below use document-blocked resampling and report corrected minus historical NLL, in nats/token.

| Prompt band | Prefill rows | Prefill NLL delta (95% interval) | Early decode rows | Early decode NLL delta (95% interval) |
| --- | ---: | ---: | ---: | ---: |
| All 268 documents | 221,428 | +0.00190 [+0.00040, +0.00375] | 16,884 | +0.00040 [−0.00201, +0.00283] |
| Below 1,024 tokens | 75,129 | +0.00022 [−0.00130, +0.00172] | 14,868 | −0.00021 [−0.00276, +0.00236] |
| 1,024 tokens and above | 146,299 | +0.00276 [+0.00070, +0.00555] | 2,016 | +0.00495 [−0.00127, +0.01109] |

Overall prefill perplexity was 5.7228 in `p1` and 5.7337 in corrected source. The prefill loss is established
on the full corpus, within the overall 0.005 nats/token margin. The long-prompt interval extends above that
margin. Overall early-decode perplexity was 3.8711 and 3.8727, with the paired difference meeting the
equivalence criterion. Early decode in the long-prompt band is inconclusive. The long-prefill loss is under
investigation.

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
- Tower. The mmproj ViT (32 x 1024) and 3 x 3 aligner split their projection outputs across eight ranks. Gathers
  assemble FP16 features in input order, each GEMM accumulates its full input dimension in FP32, and the residual,
  norms and RoPE stay FP32. The deployment keeps the mmproj and vision bias files stable during loading. The tower
  is loaded before the KV pool is sized and warmed at startup on the plan's 9,189-patch bound. Each rank computes
  the image span. While other streams decode, the leader schedules groups of tower blocks under the 175 ms policy.
- Backbone: image rows replace the image token's embedding. They route their experts with the vision bias
  (`exp_probs_b_vl`), and they are dead Engram n-gram members (`valid & ~image`), as upstream does. Text-only
  steps run exactly the text model's kernels.
- Prompt cache: an image span's positions carry ids derived from the image's pixels (sha256), so a kept point is
  reused only by a prompt with the same images. Points stay at message boundaries. Prompt scoring
  (`/v1/tensorfold/score`) stays text only.

### TP8 serving qualification

The corrected source `aef6e42a` passed the seven image checks, the 456-token image-cache repeat, and both extreme
image orientations. It retained 7,697,600 shared KV tokens and a 235 MiB worst-rank tower warm-up transient.
The first 16K prefill while 23 other requests were decoding had a 211.6 ms p99 gap across 903 visible text gaps.

The tower-only build `24a4636d` passed the original per-image gate on all seven images. It produced bitwise-identical
spans across all eight ranks. Against the FP32 oracle, each TP8 relative L2 error was 0.999972–1.000009 times the
rank-0 tower's error, and the change in minimum token cosine was at most 1.2e-7 in magnitude. The largest image's
span also matched the rank-0 tower bit for bit, and the other six differed in low bits. The independent upstream
FP32 and patch-150 comparisons passed on one device.

Before the sequence-parallel routing correction, on SC117 at 180 W with image `67b43e682057`,
`--vision --parallel 32 --context 65536 --prefix-host-bytes 16G`,
the rank-0 build `1c093114` and integrated TP8 build `51acced7` ran in A/B then B/A order. Each arm used a fresh
server. All four sessions passed seven image checks, exact-image prefix reuse, and the two 8192 × 1 and 1 × 8192
orientations at the 1,024-token image limit. The same 20 greedy chats and 4,522 prefill and 1,008 decode rows from
16 teacher-forced text documents matched exactly across all four sessions.

| Cold image workload | Rank-0 session medians | TP8 session medians | Rank-0 / TP8 TTFT and family-corrected interval |
| --- | ---: | ---: | ---: |
| One 994-token image | 1.158 / 1.148 s | 0.673 / 0.680 s | 1.705 [1.637, 1.776] |
| Four 994-token images | 3.861 / 3.862 s | 1.984 / 1.971 s | 1.953 [1.914, 1.993] |

The shared KV pool held 6,677,248 tokens with the rank-0 tower and 7,697,600 with TP8. The worst-rank transient
peak at startup's 9,189-patch bound fell from 613 to 235 MiB within the 3 GiB prompt reserve. The tower tensor
footprint was 972,189,696 bytes on rank 0 or 123,921,408 bytes per TP8 rank, an increase of 19,181,568 bytes
across the machine. During the mixed four-image request, the background text stream finished all 400 chunks.
The overlap p99 gaps in A/B/B/A order were 157.33 / 157.89 / 156.94 / 156.81 ms. The automatic 175 ms grouping
policy stayed in effect.

The ten-cell text matrix with `--vision` classified all prefill and decode cells as uncertain at its familywise
comparison level. The aggregate TP8/rank-0 prefill and per-user decode ratios were 1.001 [0.997, 1.004] and
1.010 [1.000, 1.020], respectively, with 95% intervals rounded for display. At C32 16K the median per-user decode
point estimate was 0.938 [0.594, 1.481] with its familywise interval. The median user received a first token
2.6–4.7 s earlier in the paired TP8 runs and finished 0.6–0.7 s earlier. The slower per-user decode estimate included
a larger share of time overlapping other prompts' prefill. At the end of prefill while all 32 replies still
streamed, delivery rates were 668.0 / 660.5 / 657.7 / 657.2 tok/s in A/B/B/A order, and whole-run rates were
104.48 / 104.67 / 104.43 / 104.60 tok/s. Those windows characterize the tested schedule and arrival order.

### Earlier rank-0 measurements

Measured on SC117 (180 W, image `67b43e682057`,
`--parallel 32 --context 65536 --prefix-host-bytes 16G --vision`, source ds41-vision2 1921ae48):

- Tower parity (one V100, real mmproj, 7 images including tall/wide/largest): the FP16 tower is bitwise equal to the
  friend's patch-150 tower, and the FP32 oracle is within relative L2 1e-4 of upstream's FP32 math. Group-wise
  encodes are bitwise equal to whole ones.
- Answers (greedy, thinking off): two photos described correctly, OCR of a rendered "TENSORFOLD 2026", counts of 5
  and 3 circles, and a two-image question: 7/7. An exact repeat reused all 456 prompt tokens (TTFT 1.49 s cold,
  0.05 s cached). The other photo at the same size reused none and was described as itself.
- TTFT: about 1.14 s for one 994-token image (the tower takes about 0.63 s of it). For four such images (3,989
  prompt tokens) it is 4.25 s, about 940 prompt tok/s, against about 2,960 tok/s for a 4K text prompt.
- Cost: the tower takes 0.91 GiB on rank 0, and the KV pool is sized by the smallest rank, so the pool drops from
  7,780,032 to 6,681,984 tokens. The 994-token square warm-up used 577 MiB of the 3 GiB prompt reserve. Startup
  takes about 20 s longer.
- Text requests: with `--vision`, the 20 greedy known-answer replies (thinking off and on) were identical to the
  base build's, and the C1/C8 4K/16K matrix cells matched within 0.3% on the same hold.
- Decoding streams: while a four-image prompt encoded and filled, the largest gap in a decoding stream was 0.26 s.
  A text prompt of the same size gave 0.27 to 0.40 s (the existing prompt-group policy).

The reviewed source (`01c5109d`, same launch settings and image) passed all seven vision checks and answered
requests with 8192 x 1 and 1 x 8192 images. The 1024-token image warm-up used a 613 MiB transient peak within
the 3 GiB prompt reserve. This was a functional run with a CPU suite in flight.

## Diagnostics (never for serving or benchmarks)

- `TENSORFOLD_DS41_ENGRAM_TRACE=FILE`: rank 0 appends each forward's Engram row ids to FILE. **It disables the decode
  graphs**, so it is for offline hit-rate analysis only.
- `TENSORFOLD_DS41_SCHED_TRACE=1`: prints the Decoder's layer-group plans and costs.
- `TENSORFOLD_DS41_SMOKE_LAYERS=N`: loads only the first N layers (N=1 skips the Engram mapping) for plumbing
  checks. Outputs are meaningless.

## Not yet

DSpark drafting is not merged into the serving lane.
