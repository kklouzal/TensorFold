# Gemma 4 26B-A4B (`gemma4`)

CUDA DFlash sampled requests use standard rejection sampling; greedy serial identity remains required. Fixed seed/configuration and cache reuse reproduce the drafted request, while sampled serial/drafted or different proposal-precision tokens may differ. Apple paths retain their separate sampling contract. Historical measurements below retain their source and scope. See [migration validation](../rejection-sampling-validation.md) for outstanding qualification gates.

## Native CUDA trial

The CUDA implementation reads original MLX affine 4/6-bit projections in groups of 64, including the original 6-bit tied embedding/head and expert down projections. It does not requantize the checkpoint. The separate `z-lab/gemma-4-26B-A4B-it-DFlash` companion retains BF16 weights by default (`--drafter-bits 16`). With a BF16 tied target head, as in the native CT profile below, `--drafter-bits 8` runs it on private FP8 copies (E4M3, one scale an output row) of its projections and head, changing proposals only. The original affine checkpoint's 6-bit tied head requires `--drafter-bits 16`. Proposals are verified by the target before publication, so replies retain their target semantics. CUDA currently requires `--parallel 1` and BF16 KV. Tensor/expert parallelism is outside this initial implementation. The separate native CT NVFP4 path below requires SM12.x.

For a local complete mixed4/6 snapshot and matching companion:

```sh
tensorfold serve /models/gemma4 --backend cuda --drafter /models/gemma4-dflash \
  --drafter-bits 16 --draft-rows 16 --parallel 1 --context 0 --vision \
  --host 0.0.0.0 --port 8888 --name gemma4-26b
```

For Harness, keep the standard container entrypoint (`cuda_entrypoint.py` → `tensorfold_harness.py`) and add `--thinking-budget 0`; this delegates budgets to individual requests. The CLI otherwise enables thinking by default, subject to a checkpoint-fixed template. Overriding the entrypoint with `python -m tensorfold` omits Harness's required `harness_tokenizer_version`, `model` and `context_length` response fields. Count-only image requests may exceed the context window; normal generation retains its context check and all media limits remain active.

The checkpoint's chat template and generation metadata remain authoritative. The zecanard mixed4/6 template fixes thinking on; an explicit conflicting request is refused. Use a template override to select different behavior. Image/video inputs use the local standard Gemma processor, without executing checkpoint Python. Multiple clips in one request currently require equal sampled frame counts; unequal counts are refused before preprocessing. On GB10, configure `TENSORFOLD_MEMORY_RESERVE_GIB=16`; system `MemAvailable` governs unified-memory admission. Full-model CUDA startup, output and throughput checks for this mixed4/6 path remain pending; the measurements below describe the earlier Apple implementation only. The separate NVFP4 release has its own evidence below.

## Original AEON NVFP4 on CUDA

The current integration candidate captures text decode/verification graphs over persistent KV. Startup prepares the smallest attention grid for each width; larger grids capture on first use. Prefill and visual-block forwards remain eager. On GB10, runtime source `d19d807a` passed 460 installed CPU checks, 23 native DFlash checks and eight 512-token identity cases. Remaining numerical, full-context, media and comparative performance runs were waived for the accepted deployment; no new performance winner is claimed. Retain the source/image scope of the historical results below.

`AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4` stores individually calibrated compressed-tensors NVFP4 projections. TensorFold preserves the original FP4 code/block-scale bytes and input/weight global calibrations. It executes the declared FP4 activation math on SM12.x, with bounded GPU packing during load. The original BF16 tied embedding/head, custom BF16 router weights, vision tower and projector remain unquantized. `--precision full` is refused for this path rather than silently changing the declared activation math.

Use a complete local NVFP4 snapshot and the same BF16 DFlash companion:

```sh
tensorfold serve /models/gemma4-nvfp4 --backend cuda --precision checkpoint \
  --drafter /models/gemma4-dflash --drafter-bits 16 --draft-rows 5 \
  --parallel 1 --context 0 --vision --host 0.0.0.0 --port 8888 \
  --name gemma4-26b --thinking --thinking-budget 0 \
  --chat-template deploy/gb10/gemma4-chat-template-google-4d7ae498.jinja
```

The checkpoint ships Google's launch template, which renders OpenAI tool results as a `<|turn>tool` turn and drops the thought of tool-call turns. `--chat-template` selects Google's current template (`google/gemma-4-26B-A4B-it@4d7ae498`, SHA256 `ae53464b…`). It renders tool results in the trained `<|tool_response>` format and keeps current-turn `reasoning_content` in tool-call turns. [Provenance and prompt differences](../gemma4-cuda-validation.md#chat-template-googles-current-template) list what changes. For Harness, retain the standard compatibility container entrypoint described above. This template accepts the thinking switch. This profile enables thinking by default with no server-side thinking-token limit (`--thinking-budget 0`). Harness controls thinking and budgets per request; `chat_template_kwargs.enable_thinking=false` disables it for an individual request. EOS and sampling defaults come from its generation metadata. Gemma has no Flash Next n-gram tables to offload. The initial native path uses fixed row-independent FP4 K reductions, the publisher's BF16 router input order with FP32 logits/softmax, single-cast learned RMS norms, BF16 GeGLU activation/product boundaries, and ascending-expert BF16 weighted accumulation. Serial/DFlash equality is scoped to this TensorFold arithmetic; cross-engine bit identity is not claimed.

The GB10 [Compose profile](../../deploy/gb10/compose.gemma4.yaml) preserves the Harness entrypoint, LAN port 8888, model ID `gemma4-26b`, resident vision and BF16 KV without Docker RAM/CPU quotas. Its current defaults are a 16 GiB host reserve, ten compilation/OMP/MKL workers, cpuset `5-9,15-19`, BF16 DFlash, five verification rows, one request lane and the native 262,144-token context. These resource and context settings have explicit `TENSORFOLD_*` parameters in [gemma4.env.example](../../deploy/gb10/gemma4.env.example); set them for the actual host and retain the selected region's limits. The profile mounts the shipped Google template from its own directory read-only and passes it with `--chat-template`; keep the template beside a copied compose file. Copy the environment example to a host-owned file and set the qualified immutable image, complete pinned model/drafter directories, cache and GPU UUID:

```sh
docker compose --env-file /path/to/gemma4.env -f deploy/gb10/compose.gemma4.yaml config --quiet
docker compose --env-file /path/to/gemma4.env -f deploy/gb10/compose.gemma4.yaml up -d
```

The persistent service uses Docker's `unless-stopped` restart policy and its existing health check. A finite qualification worker closes its container at completion; do not use that worker or a short detached launch command as the lifetime owner of production serving. Preserve the host's VBoost and clock policy, acquire the maintained GPU admission for the deployment transition, and keep any faulty-image production hold until the fixed image's required gates close. A rendered configuration or successful startup does not establish full-context, output-quality or performance qualification.

The example's five rows permit four DFlash proposals. A bounded GB10 comparison of 3/5/11/16 rows selected five for the tested short code/chat workloads; longer prompts and other workloads require their own measurements. The final installed image passed 461 CPU and 50 CUDA cases, full-token serial/DFlash identity, Harness counting, streaming, image/video and enforced strict tools. Four 512-token coding requests observed 64.49–74.41 decode tok/s with DFlash versus 41.16–41.37 serially. A separate 200-request source comparison retained every reference token and observed about 76 code and 55 chat tokens/s through the warmed engine at context 4096. R16 head reuse clearly improved both source workloads over the original head; its ranking against R8 remains inconclusive, rather than equivalent. [Validation and immutable receipt](../gemma4-cuda-validation.md) record the exact checkpoint, image, workload and limits. Preserve the host's Docker affinity policy and verify actual process/cgroup affinity before measuring, as described in the [deployment guide](../../deploy/gb10/nvfp4.md). Current qualification uses ten CPUs; historical twenty-CPU results retain their original scope. Earlier affine smoke results and Apple measurements do not qualify this NVFP4 path.

Measured on an M4 Pro with 64 GB (273 GB/s) and MLX 0.31.2, with `mlx-community/gemma-4-26b-a4b-it-4bit`.
Packages: `src/tensorfold/families/gemma4/` and `src/tensorfold/kernels/gemma/v1/`.

## What decides the speed

- 30 layers: 25 sliding-window attention (window 1,024; 16 query heads over 8 KV heads, head dim 256) and 5
  full attention (16 query heads over 2 KV heads, head dim 512, keys reused as values, proportional RoPE).
  Hidden size 2,816.
- Every layer has a dense GeGLU MLP (width 2,112) and a MoE block beside it: 128 experts, top 8, expert width
  704, GeGLU. Three post-feedforward norms and a per-layer scalar join them.
- Vocabulary 262,144, tied to the embedding, logits soft-capped at 30. 4-bit affine weights in groups of 64.
- A token reads about 2.16 GB of weights: 7.9 ms at 273 GB/s. The vocabulary head alone is 415 MB (1.5 ms).
- mlx_lm's step runs about 40 small kernels a layer around the matmuls and takes 14.6 ms synchronous, 13.3
  ms decoded one step ahead.

## Drafts

The checkpoint has no draft head, so a stream drafts copies of its context. With `--drafter
z-lab/gemma-4-26B-A4B-it-DFlash --drafter-bits 8` (Apache-2.0; `tensorfold pull` it first), z-lab's DFlash model
drafts a chain of up to 15 tokens a round. It reads the target's hidden rows after layers 1, 6, 11, 17, 22 and 27,
and the engine sets each round's depth from its costs. Replies stay equal to `"draft": false` ones.

Measured on an M3 Ultra (MLX 0.32.2, quick cells, 64 and 256 tokens, thinking off, gemma/mlx_lm's server):

| Cell | No draft model | DFlash |
| --- | --- | --- |
| Chat, greedy | 1.31-1.35 | 1.52-1.53 |
| Chat, sampled | 1.12-1.16 | 1.29-1.31 |
| Code, greedy | 1.31-1.35 | 2.03-2.12 |
| Code, sampled | 1.10-1.16 | 1.62-1.84 |

Rounds average 3-5 rows at 13-20 ms, and the draft takes about 4 ms of that. The draft model runs once per stream
in a round, so four streams sharing rounds make 173-179 tok/s together against 196 without it.

## What worked

One-row steps, greedy, median ms a step on a chat prompt (`tools/gemma4_decode_bench.py` for synchronous,
one step ahead as the server's pipelined engine runs it):

| Step | Synchronous | One step ahead |
| --- | --- | --- |
| mlx_lm model | 14.62 (68.4 tok/s) | 13.34 (75.0 tok/s) |
| fused glue only (norms, residuals, routing, stacked q/k/v and dense gate/up) | about 14.0 | |
| plus the expert kernels | 12.65 (79.1 tok/s) | 12.34 (81.0 tok/s) |
| plus the expert down loads issued before the reductions | 12.7 | 12.3-12.5 (80-81 tok/s) |
| plus every load of `attn_tail` and `moe_tail` issued before their first reduction | | 11.7-11.9 (84-85 tok/s) |

Through the server (`TF_GEMMA4_PIPELINE=1`, `tools/bench_openai.py`, 512 tokens, 5 reps, back to back on the
same machine), median tok/s:

| Prompt | Temperature | mlx_lm forward (`TF_GEMMA4_FUSED=0`) | fused, first cut | fused, tails fixed |
| --- | --- | --- | --- | --- |
| fibonacci-raw | 0 | 70.0 | 76.3 | 81.3 |
| gpu-chat-no-think | 0 | 70.0 | 76.2 | 80.2 |
| fibonacci-raw | 1.0 | 61.8 | 77.5 | |
| gpu-chat-no-think | 1.0 | 69.3 | 75.2 | |

The server runs about 0.6 ms a token slower than a short in-process loop: back-to-back requests keep the GPU
hot (12.05 ms rested against 12.64 ms after a 2,500-token run, same short context), and the step grows with
context (12.6 to 14.3 ms over 2,500 tokens, the sliding-window caches filling to 1,024 keys).

The pieces (`kernels/gemma/v1/kernels.py`), per layer:

- q, k (and v on sliding layers) stacked into one quantized matmul; `qkv_norm` applies the three head norms
  in one kernel. RoPE, the cache update and attention stay mlx_lm's.
- `attn_tail`: post-attention norm, residual add, and the three norms that read the result (dense MLP input,
  expert input, router input) in one kernel.
- The dense MLP's gate and up stacked into one matmul.
- `route`: router logits to the top 8 of 128 (ties to the lower id), softmax over the 8, per-expert scale.
- `expert_gateup`: the 8 experts' gate and up projections and GeGLU in one pass over the input. MLX's gathered
  matmuls read the input once a projection and launch the activation apart.
- `expert_down`: the 8 experts' down projections and their weighted sum.
- `moe_tail`: post-feedforward norms 1 and 2, the joint norm, residual add, layer scalar, and the next layer's
  input norm.
- The work before and after attention compiled with `mx.compile` per layer; the graph goes to the GPU every 8
  layers.

The glue alone saved about 4%: mlx_lm's glue kernels were cheap next to its gathered expert matmuls, which
read the input once a projection. Stubbing parts of the step showed the experts at 4.2 ms of 13.1 before
the expert kernels.

## Where the time goes now

Each op timed over all 30 layers in one chain (each layer's weights read once, as a real step reads them),
best of five:

| Op | ms a token | MB | GB/s |
| --- | --- | --- | --- |
| q/k/v (MLX quantized matmul) | 1.72 | 397 | 220-234 |
| o_proj | 1.0 | 227 | 228 |
| dense gate/up + down | 1.45 | 301 | 195-215 |
| expert gate/up + GeGLU | 2.5 | 535 | 210-215 |
| expert down + sum | 1.37 | 268 | 196 |
| head (in the step, with soft-cap and sampling input) | 2.0 | 415 | |

The most this machine read in our runs was 254 GB/s (the head), so the practical floor is about 8.5 ms.

Those rates are for independent calls, which overlap. The step is a dependent chain of about 10 kernels a
layer, so what counts is each kernel's latency in that chain. Removing parts from the real step (medians of
three interleaved rounds, 11.86 ms whole): the head is 1.74 ms, RoPE, the cache update and attention together
0.2 ms, so the 30 layers take about 10.1 ms for 1.74 GB, 173 GB/s. Each matmul reads only 7 to 27 MB a layer,
and its ramp-up and drain sit on the critical path. Dependent-chain latency per call, 30 layers of weights:

| Op | us a call |
| --- | --- |
| trivial elementwise kernel | 9 |
| q/k/v matmul + `qkv_norm` | 93 |
| o_proj | 50 |
| `attn_tail` (before / after issuing loads first) | 54 / 25 |
| router matmul + `route` | 41 to 62 |
| `expert_gateup` | 107 |
| `expert_down` | 80 |
| `moe_tail` (before / after) | 39 / 27 |
| dense MLP (off the critical path, beside routing and experts) | 86 |

## Tried and rejected

- The expert down kernel with every lane busy (704 inputs are 44 chunks of 16, so the committed layout idles
  20 of 32 lanes on the second chunk): 4, 8, 16 or 32 lanes a row. All slower than issuing the 8 rows' loads
  before the reductions, and they change the sum order.
- `expert_gateup` launch shapes (1 to 8 simdgroups, 1 to 8 rows each) and an unrolled K loop: within noise.
- Handing the graph to the GPU every 0, 4, 15 or 30 layers instead of 8: within noise.
- Folding RoPE into `qkv_norm`: removing RoPE from the step saves 0.07 ms, so not attempted.
- The dense MLP as three more expert slots (its width 2,112 is three expert widths; Nemotron folds its shared
  expert the same way): two launches instead of five, correct to bf16, but 12.39 to 12.56 ms. The dense MLP
  used to run beside routing; merged, it waits for the router.
- The 8-bit router matvec and the top-8 selection in one threadgroup: same experts and bit-identical weights
  in 600 of 600 checks, 62 to 46 us in a dependent chain, but 11.69 to 12.13 ms in the step. MLX's matvec
  spreads over the GPU beside the dense MLP; one threadgroup does not.
- `qkv_norm` with its norm weights loaded before the reduction: no change (many threadgroups hide the latency).
- `MLX_MAX_OPS_PER_BUFFER=200` and `MLX_MAX_MB_PER_BUFFER=100000` (Nemotron's and Flash Next's settings):
  within noise.

## Exactness

- One-token steps run the fused kernels; prompts run mlx_lm's forward and fill the same caches. The fused
  arithmetic follows mlx_lm's (fp32 sums, bf16 where mlx_lm stores bf16) with its own sum orders (strategy C
  in the [recipe book](README.md)), so serial decoding through it is its own reference.
- The family does not draft: without `multi_row_exact` the engine runs one row a round. The same limit as
  Flash Next and Nemotron applies to prompts: a one-token suffix after a cache hit goes through the fused
  step, a longer one through mlx_lm's forward.
- Quality, judged against the same 4-bit weights run with fp32 activations (`tools/gemma4_truth_eval.py`), on
  5,742 tokens of the model's own sampled answers to 12 chat prompts:

| Path | KL mean | KL trimmed | KL median | tokens over 1 nat | top-1 = truth |
| --- | --- | --- | --- | --- | --- |
| mlx_lm whole-sequence forward | 0.00297 | 0.00228 | 3.6e-6 | 0 | 98.89% |
| mlx_lm decode | 0.00329 | 0.00251 | 4.1e-6 | 0 | 98.61% |
| fused decode | 0.00893 | 0.00245 | 4.0e-6 | 2 | 98.66% |

  The fused mean comes from 2 tokens of 5,742, knife-edge positions where any bf16 rounding flips the answer.
  At one of them every bf16 path disagrees with the fp32 one. At the other, each of the 354 x 30 expert-down
  calls matches mlx_lm within bf16 on identical inputs.
- Tests (`tests/test_gemma4_kernels.py`): every kernel against an MLX reference, each layer against mlx_lm's
  layer on identical inputs with a small quantized model, and the caches advancing as mlx_lm's do.

## Traps

- KL or NLL against mlx_lm's own samples charges any difference from mlx_lm as damage: it called this path 4x
  worse. Compare every path against an fp32 truth, and read the trimmed and median figures beside the mean.
- A chat model's NLL on raw text or on chat templates is 7 to 10 nats a token. The distribution there is too
  flat to rank kernels. Score the model's own answers.
- Timing a kernel by calling it repeatedly on one layer's weights reads them from cache: 402 GB/s on a 273
  GB/s machine. Chain through all layers.
- Single timing runs of the step drifted up to 7% between runs (12.3 to 13.2 ms). Decide with interleaved A/B
  runs in one process and identical output hashes.
- A kernel's latency in an isolated dependent chain did not predict the step twice: the merged dense MLP and
  the one-threadgroup router both won alone and lost in the step, where they gave up overlap with other work.
- The server rebuilds saved system blocks after a kernel change. A request that arrives during that rebuild
  waits (11.8 s behind a 25k-token block here), which looks like a slow first token.

## Next

- The layers read 1.74 GB at 173 GB/s against about 245 possible, 2.9 ms a token. The loss is latency in the
  per-layer chain, so it needs fewer dependent kernels a layer (one kernel per matmul and its tail, or a
  persistent kernel over the layer), not faster matvecs.
- `expert_gateup` from about 215 toward 240 GB/s is worth about 0.3 ms a token.
- A multi-row fused step with a load-time row check would allow copy windows (drafting).
- Prefill through the fused step would remove the cache dependence of prompt bits.
