# Smaller pretrained NVFP4 validation

The installed TensorFold container ran **surogate/Qwen3.5-9B-NVFP4**, revision
`3410fbc4f4075661f58f6b222d05b6dde646b8c1`, on the single RTX PRO 2000
Blackwell/SM120 host. Its complete 9,361,048,680-byte weight file matched the
publisher's pinned LFS SHA256. This is a real ModelOpt export of a dense model.
It exercises the dense NVFP4 implementation; Flash Next's routed-expert and
SSD n-gram paths require their own model inputs.

The [receipt](verification-small-nvfp4-20261017.json) identifies the ordinary
installed image `c74633ad…`, runtime source `b551e68f`, wheel, model hashes,
request captures, resource observations and corrected validation. No application,
checkpoint or installed dependency was modified for these runs.

## Executed configuration and observed rates

Both modes used a 16,384-token configured window, four slots, BF16 KV, drafting
off, FP8 prefill off, thinking off and the documented 2 GiB startup-reserve
override. No absolute CUDA-memory cap was set. Each benchmark sent the same
greedy LRU-code prompt with seed 7, then repeated it twice. All six replies
actually reached 512 tokens with finish reason `length`; each mode reproduced
its own token sequence exactly.

| Precision | Activation math against stored NVFP4 weights | First request after startup | Warm 1 | Warm 2 |
| --- | --- | ---: | ---: | ---: |
| `checkpoint` | Native FP4 × FP4 on this SM120 GPU | 24.06 tok/s | 24.27 tok/s | 24.13 tok/s |
| `full` | BF16 activations | 34.41 tok/s | 34.73 tok/s | 34.73 tok/s |

These are the engine's observed `decode_tps` readings from three sequential
requests per mode, after ordinary constructor warmup. Source/compiler caches
were reused and clocks/thermals were not locked. The modes produced different
token sequences and have separate numerical contracts; these readings establish
no quality-preserving speedup, general model quality, or full-program correctness.
The token cap truncated the generated code. They are single-request readings
from a four-slot server, rather than four-request aggregate throughput.

The user required at least 8 GB of system RAM to remain available. Validation
used a stricter 8 GiB floor, a per-run container RAM cap derived from initial
MemAvailable minus 10 GiB, no additional swap, and a fail-closed guard. The
lowest recorded availability across the selected runs was **53,324,820,480 bytes
(about 49.7 GiB)**. The highest sampled framebuffer reading was **8,851 MB as
labeled by `nvidia-smi dmon`**, including other host GPU users. These are sampled
extrema, not continuous reservations or Torch allocator peaks. Every selected
server retired with SIGINT status 0 and no OOM kill. The embedding container
remains stopped.

## Request policy and capabilities

This pinned export declares generation EOS `248044` (`<|endoftext|>`), while its
tokenizer calls `248046` (`<|im_end|>`) the chat end. A default request therefore
returned `323<|im_end|>` followed by a newline before its configured EOS. That
failed smoke test is retained. The successful qualification explicitly supplied
the standard API option:

```json
{"stop": ["<|im_end|>"]}
```

This is a request policy for this checkpoint's metadata. The original files
remain unchanged, and default behavior is not relabeled as passing. With the
explicit policy, exact integer output, seeded repeat/serial-request identity,
ordinary SSE framing and usage, strict JSON-schema output, and four concurrent
HTTP requests versus their solo token sequences were verified.

This checkpoint uses JSON tool-call markup. TensorFold's current strict
function-tool compiler requires Qwen XML markup, so it correctly refused a
required strict tool and the unchanged 31-tool Harness catalog with HTTP 400.
Their strict flags were never relaxed. Positive strict function-call/Harness
compatibility is not qualified for this template; strict JSON `response_format`
was separately enforced and an unsupported strict schema was refused.

Dense prompt-progress events are outside the backend contract. Its KV-memory
reading is also unknown (`kv: null`), so the corresponding percentage metric
is correctly omitted. Request/build/queue metrics and zero final logical
request owners were verified. Vision, actual drafting, filled-window capacity,
other GPUs and the full Flash Next ModelOpt checkpoint remain separate gates.

## Evidence scope

The result combines six completed benchmark generations from the v2 runs with
all 26 real HTTP operations per mode captured by v4, followed by an independent
corrected offline validator. Early helpers incorrectly borrowed Flash Next's
progress, XML-tool and KV-telemetry expectations. Their original nonzero exits
and raw responses remain retained. The final offline validator checks exact
request/status sequences, unchanged image/model/quality inputs, supported output
and refusal contracts, all remaining metrics and actual resource retirement.
This is composite qualification, not a single clean monolithic harness run.
No further GPU execution was needed to correct an assertion over a complete
saved metrics response. The receipt records the exact validator and captures.
