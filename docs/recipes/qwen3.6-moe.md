# Qwen3.6-35B-A3B

The `qwen3_5_moe` family serves Qwen3.6-35B-A3B on Apple Silicon and on one NVIDIA GPU. Its layers are the 27B's
(Gated DeltaNet and gated full attention, every fourth layer attention) with routed experts in place of the dense
MLP. Affine CUDA checkpoints draft with their MTP layer; original CT NVFP4 uses a separate DFlash v1 model.
MLX preserves this fork's DFlash default;
`--mtp-drafts N` explicitly selects the newly integrated MTP capability. Its actual Apple numerical,
exactness and performance qualification remains pending.

## Checkpoint

```bash
tensorfold pull TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP
tensorfold serve TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP --name bench
```

Tested revision: `81169a9bc511a27c1b4eedb77a2cd98ced431847` (20.9 GB). Its weights are
`mlx-community/Qwen3.6-35B-A3B-4bit` (MLX affine 4-bit in groups of 64, routers and the shared-expert gate at
8 bits), converted from `Qwen/Qwen3.6-35B-A3B`; that conversion drops the MTP layer, which ships beside it as
`mtp-4bit.safetensors` (converted by the same rules: experts split into gate and up projections, norms
shifted by one, projections 4-bit, gates 8-bit). The mlx-community folder serves the same way once that file
is placed in it. After the weights, one Spark keeps about 75 GB for caches: attention holds 20 KB a token and
DeltaNet 63 MB a stream.

`--no-drafts` or request field `"draft": false` selects serial decoding, the reference drafted output equals.

## Heretic NVFP4 host check

The GB10 held `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` at revision
`3d05c6063e82cdae8dd1b9362116ea71be8d1439`. Its 11 files, including the unchanged
23,354,242,448-byte safetensors file and ancillary metadata, were copied to the
RTX PRO 2000 Blackwell host and fully verified against the publisher's pinned
hashes. This is the 35B-A3B Heretic model, not the dense Qwen3.8-27B.

The ordinary installed runtime `a44085dc…`, application source `8cf97219`,
correctly refused `compressed-tensors nvfp4-pack-quantized` at capability
preflight. That recorded image read MLX affine 4-bit/group-64 weights on CUDA;
the refusal is retained as historical evidence before the new reader below.

Headers show **18,119,639,040 bytes (16.88 GiB)** of routed expert payload,
before shared experts, backbone, state or KV. Fully resident execution therefore
cannot fit the 16 GB card. The historical image lacked this family's NVFP4 MoE
reader and RAM-expert path. The installed CLI test exposed no GPU and stopped before weight
loading. No decoder, GPU admission, OOM, model-quality or throughput pass is
claimed by that historical run. The [receipt](../verification-heretic-a3b-20261017.json) preserves the
input identities, installed refusal, size evidence and verification limits.

## Original CT NVFP4 and DFlash v1

The new loader reads original `compressed-tensors` `nvfp4-pack-quantized` Qwen MoE
exports. Routed and shared gate/up/down projections retain their packed weights,
block scales and separate activation/global calibrations. Excluded BF16 routers,
shared gate and backbone projections remain BF16. Native FP4 activations require
SM 12.x and `--precision checkpoint`; this path explicitly refuses other math
modes. It does not convert the model to affine weights.

Activation quantization consumes each original CT FP32 input-global multiplier
directly; reconstructing it from a rounded reciprocal can change FP8 block-scale
bytes. Output calibration uses separately rounded FP32 input/weight reciprocals
and their FP32 product. Exactness means the same target math/build and sampling
rule across serial, drafted, resident and cached execution; it does not promise
bit identity with another runtime's reduction or epilogue ordering.

The secondary `z-lab/Qwen3.6-35B-A3B-DFlash` at
`42d3b34d588423cdae7ba8f53a8cf7789346a719` is an eight-layer DFlash v1 block-diffusion
model, with target residual taps `[1, 10, 19, 28, 37]`, block size 16 and its declared
YaRN rotary configuration. It is separate from a native MTP head. TensorFold reads
its metadata and weights through owned code; checkpoint Python is never imported.
Default draft linears keep BF16 storage. Explicit `--drafter-bits 4` changes only
proposals; the target verifies every committed token. `0` or `16` selects BF16 on
this CUDA path. With a secondary, `--draft-rows N` overrides the metadata block
width, from 2 through 128 on one stream and at most 16 with `--parallel`.
Without a secondary, width 1 is also valid. `--mtp-drafts` selects the
affine model's native MTP path and does not apply to this separate model.

```bash
tensorfold serve /models/heretic-nvfp4 --backend cuda --name heretic \
  --drafter /models/qwen36-dflash --context 4096 --vram-experts auto \
  --draft-rows 8 --drafter-bits 16 --vram-experts-pin on
```

The example uses the measured single-request profile on the 16 GB Blackwell;
registration must pass host admission. The secondary's metadata default remains
16 rows, and registration defaults to `off`. Context and eight draft rows are
explicit operator choices. Without a context override,
normal capacity admission applies; a larger admitted context can leave a smaller
expert pool. `--no-drafts` and request field `"draft": false` use the serial target
reference. An automatically resolved drafter is used only when the family's
separate checkpoint is present locally; explicit `--drafter` selects its path.

When resident experts cannot fit the minimum admitted state, omitted
`--vram-experts` selects RAM-backed experts automatically. A numeric GiB budget or
`auto` explicitly selects the bounded pool; `--vram-experts-pin off|auto|on` controls
authority registration. The pool reserves future KV, retained states, draft
contexts and runtime scratch before choosing its final size. It requires discrete
VRAM, a fresh native CUDA allocator and stable allocator configuration through
startup. Reused/unproved allocator state fails explicitly. Resident execution may
use unified memory. Resident banks group each layer's routes on the GPU (no host
read of the picks) and run staged native FP4 activation kernels that also compute
the layer's shared expert as the bank's last cell when it has the routed experts'
A4 storage (otherwise it keeps its own projections); cached banks plan host windows
from the picks. Tests compare both paths bit for bit. With one resident stream,
verification rounds that form a chain (serial steps, DFlash v1 and copy proposals)
replay CUDA graphs over a fixed copy of the request's state, captured at startup
for the smallest context span (larger spans on first use); grammar-pruned trees,
image prompts, concurrent streams and RAM-backed experts run eagerly. Admission
charges the fixed state, the retained graph outputs and the device-plan scratch.
The affine native-MTP graph path remains separate. Cached authority stores one
16-byte-aligned U8 record per expert, including its three packed projections and
four calibration fields. A miss
uploads that record once. Native field views are prepared before the pool's
first borrow and rebuilt transactionally when startup sizes the pool. For this
checkpoint, each record is 1,769,536 bytes, including 40 bytes of initialized
padding; admission charges the padding and host registration geometry.

Loading validates original block-scale bytes and scalar calibration on CPU and
retains the original one-projection CPU byte permutation. GPU loading candidates
did not establish an improvement in the controlled component/startup region;
they are not a production backend or fallback.

The shared NVFP4 estimator owns the Reader's physical host-ring reservation.
The offload plan adds CPU authority and projection workspace without charging
that same ring again. Ring accounting does not change the Reader's span size or
the GPU scratch allowance.

The cache owns the CPU authority and GPU pool. Cached banks retain geometry and
cache identity, with no separate CPU calibration copies. A lease fences native
consumers before reuse or close. Failed allocation or view preparation before
replacement commit preserves the existing pool; interrupted publication aborts
startup and retires owned resources. Failed retirement retains its owners for
explicit cleanup. Calls after a
successful close fail before native execution or output mutation.

Installed source `099eb4d0` passed native component checks and full-model serial,
drafted and seeded token identity on the RTX PRO 2000 Blackwell. The eight-row
BF16 profile reached about 37 tok/s with a 7.29 GiB expert pool. A balanced
54-reply comparison retained identical 512-token outputs and work counts across
the original and prepared vocabulary heads. The conditional paired analysis and
its assumptions, source/image identities, API coverage and limitations are in
the [qualification](../qwen36-ct-nvfp4-validation.md). These results establish
same-target execution identity, rather than general answer quality or parity
with another runtime.

Concurrent prompt prefill runs the routed MoE over the aggregate batch with the
same BF16 post-attention boundary as separate prefills. Each stream retains its
own convolution, recurrent, attention and DFlash context. The scheduler bounds
the aggregate prompt rows by the admitted forward limit, including while other
streams decode. Routed batched prefill supports one GPU. Fresh installed checks cover the aggregate row bound, independent
per-stream state, and four concurrent 512-token replies equal to the serial
reference. That separate four-slot/four-row profile completed 2,048 tokens at
57.70 aggregate client tok/s; it is not the single-request profile above.

## Mac execution

Macs verify on the dense family's row decoder, with every verification width checked at load against
one-row steps. The implicit drafter remains `z-lab/Qwen3.6-35B-A3B-DFlash`, as in fork
`2f8089a7b59588875d639a7efa860117168a29d0`; upstream's MTP speed results have not qualified replacing it.

Select the checkpoint's optional MTP head explicitly:

```bash
tensorfold serve TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP --mtp-drafts 4 --name bench
```

An explicit separate `--drafter` combined with active `--mtp-drafts` is refused before model downloads.
`--no-drafts` disables drafting. The optional MTP path absorbs kept target rows and uses the target's keyed
sampling rule for its proposals; the target still verifies committed tokens against its full vocabulary.
The fork's existing measured-depth policy remains in place. Upstream's new plain-round guard, 5% margin
and probing policy are isolated candidates, with actual Apple comparisons pending.

`mlx-community/Qwen3.6-35B-A3B-4bit` has the same weights without the MTP file. For explicit MTP, download
`mtp-4bit.safetensors` from the named TensorFold checkpoint, put it beside the model or use
`TF_QWEN36_MTP=<file>`. Without the optional file, the selected MTP path reports the absence and decodes
one token per round. `--mtp-drafts 0` explicitly selects a zero-draft path.

The following table is historical upstream evidence from
[`python-0.6` at ed78d6fc](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/docs/recipes/qwen3.6-moe.md).
Its MTP row includes upstream policies that remain isolated here; it does not measure this integration or
establish the fastest drafter for the current fork.

On an M3 Ultra (60-core GPU, MLX 0.32.3), greedy, median of three after a warm-up. mlx-vlm 0.7.4 serves the same
weights with `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` (two drafts a round). The short prompts are about 85 tokens
with 384 generated; the long one is 28,400 tokens of Python's `typing.py` with 128 generated.

| tok/s | Short prose | Short code | Short, thinking on | 28,400-token prompt | First token at 28,400 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Upstream TensorFold, MTP | 173.4 | 244.7 | 196.4 | 124.0 | 15.9 s |
| TensorFold, `--no-drafts` | 118.5 | 119.2 | 119.3 | 92.7 | 15.9 s |
| TensorFold, `--drafter z-lab/Qwen3.6-35B-A3B-DFlash` | 156.1 | 288.2 | 183.1 | 87.5 | 15.7 s |
| mlx-vlm 0.7.4, MTP | 111.9 | 141.8 | 120.7 | 85.6 | 14.8 s |
| mlx-vlm 0.7.4 | 106.4 | 105.7 | 107.5 | 88.0 | 14.7 s |

Upstream reports the same token SHA-256 in all three TensorFold modes in this trial. MTP rounds verify 4 to 5 rows and keep about
three tokens. DFlash's 16-row blocks keep more on short code, and fall behind plain decoding at the long prompt.

The chat template thinks by default, as on vLLM: replies reason in `reasoning_content` before the answer in
`content`, and `max_tokens` counts both. The server says so at startup, and prints a warning when a reply reaches
`max_tokens` before it leaves its think block (its `content` is empty). `--no-thinking`, or
`"chat_template_kwargs": {"enable_thinking": false}` in a request, turns thinking off.

## CUDA execution

Private copies of a retained MTP attention cache keep its model-owned arena origin as well as its
committed rows and position. The copied K/V buffers remain private. Resumed and concurrent requests
must complete without contained stream errors and retain the fresh serial token sequence.


The CUDA figures below are retained upstream historical measurements at the source above, not a new
qualification of this fork or its pinned deployment environment.

The historical measurements below use the MLX 4-bit checkpoint; the CT NVFP4 path
is described separately above, and EXL3 exports are not read. Affine prompts take bf16
activations by default: 0.90-0.96x the FP8 prompt path from 2k to 128k, and 1.23-1.89x vLLM on NVIDIA's NVFP4 export
from 2k to 64k. `--prefill-fp8` restores the FP8 path ([prompt precision](cuda.md#prompt-precision)).

Verify windows run the 27B's shared kernels (4-bit matmul, DeltaNet tree and replay, tree attention) with
routed experts from `tensorfold/cuda/experts.py`: the router's top 8 of 256 by fp32 logit (ties to the lower
id), weights renormalized over the eight, the shared expert as expert 256 with a sigmoid gate, and the slots
summed in slot order. Each (row, expert) pair gets the same bits in any window, so a drafted row equals the
serial step.

Each round first verifies a copied continuation when the context repeats eight or more tokens, and otherwise
a chain of up to three MTP drafts: the head reads the target's final normed state and the next token's
embedding, drafts with the target's keyed sampling rule over a draft vocabulary, and a chain stops after a
draft it gives under 30%. Decoding runs in buffers the engine keeps between requests, so verify chains and
head steps replay CUDA graphs captured once per width and context bucket.
Prompts prefill in chunks; the head absorbs every prompt row but the last. States are kept at the second
message's start, the last assistant turn's start and prompt ends, so a prompt sharing a system block or
extending a conversation resumes there with a fresh prefill's bits.

### Concurrent requests

```bash
tensorfold serve TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP --parallel 8 --name bench
```

`--parallel N` decodes up to N requests in shared rounds, and every reply equals the same request served alone
and its `"draft": false` run. A round verifies every stream's MTP chain or copied continuation in one forward;
the head absorbs every stream's kept rows in one call, then the chains advance a step at a time for all streams
by the one-stream rule, so a stream drafts what it drafts alone. A new prompt prefills 1,024 tokens a round
while the others decode, states are kept at message starts and prompt ends, and each stream's caches are sized
once, at admission. Startup admits N full prompt/reply windows, three kept prompt ends and the graph buffers
before loading; an explicit `--context` that does not fit is refused with the window that does. Rounds over
several streams run eagerly, as the 27B's do; a stream decoding alone replays the one-stream CUDA graphs, so a
lone request runs as fast as without `--parallel`.

On one RTX PRO 6000 Blackwell Max-Q (NGC 26.07, torch 2.13, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
checkpoint revision 81169a9), with 8 client threads over HTTP:

| | `--parallel 1` | `--parallel 4` | `--parallel 8` |
| --- | ---: | ---: | ---: |
| Label JSON: 32 requests, public-domain passages, up to 1,410 tokens, default sampling | 904 tok/s | 1,255 tok/s | 1,476 tok/s |
| p50 / p95 latency | 8.6 / 9.1 s | 5.9 / 8.1 s | 5.1 / 7.7 s |
| Chat: 48 requests, up to 256 tokens, half greedy, half seeded | 427 tok/s | 859 tok/s | 1,094 tok/s |
| p50 / p95 latency | 3.3 / 3.9 s | 1.6 / 2.5 s | 1.2 / 2.0 s |
| Peak memory (nvidia-smi) | 22.3 GiB | 22.1 GiB | 22.9 GiB |

Every reply's token SHA-256 is the same at each N and with `"draft": false`. The label replies repeat their JSON,
so copied continuations keep 9.3 tokens a round per stream; chats keep 2.9. One client at a time gets 912 tok/s
on the label requests at `--parallel 8`, as a lone stream replays the graphs.

## Measurements

One DGX Spark (GB10) in NVIDIA's `pytorch:26.07-py3` container, checkpoint revision 81169a9, against vLLM serving
`nvidia/Qwen3.6-35B-A3B-NVFP4` with MTP=3 on the same Spark (`vllm/vllm-openai`, prefix caching, chunked prefill,
`--max-num-batched-tokens 8192`, `--gpu-memory-utilization 0.60`).

Decode with the [public benchmark command](README.md#measurements); drafted replies equal `"draft": false` ones:

| | Code sampled | Chat sampled | Code greedy | Chat greedy |
| --- | ---: | ---: | ---: | ---: |
| TensorFold | 179.4 tok/s | 141.3 tok/s | 166.6 tok/s | 162.8 tok/s |
| vLLM, MTP=3 | 120.6 tok/s | 100.6 tok/s | 122.1 tok/s | 117.0 tok/s |

Serial decoding (`--no-drafts`) runs at 86-88 tok/s. A round verifies up to four rows, and each row brings its
own eight experts, so a round reads about twice the bytes of one serial step; longer drafts pay off only when
most of their rows are kept.

Cold prefill with `tools/prefill_cold.py` (chat prompts from the Python standard library at exact rendered
lengths, a unique first line each so nothing resumes; median time to first token of three):

| Prompt tokens | 2,048 | 8,192 | 16,384 | 32,768 | 65,536 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorFold | 7,161 tok/s | 7,353 tok/s | 6,541 tok/s | 5,257 tok/s | 3,688 tok/s |
| vLLM, MTP=3 | 5,907 tok/s | 5,881 tok/s | 5,090 tok/s | 3,951 tok/s | 2,693 tok/s |

The server process peaks at 31.3 GiB (nvidia-smi) during the 65,536-token prompts; vLLM holds its memory
reservation, 72.9 GB at 0.60.
