# Gemma 4 SM70 throughput checkpoint, 2026-11-15

This checkpoint reduces prefill work and streaming-parser overhead without
changing target quantization, KV precision, sampling, routing or tool/channel
semantics. Optimization work continues; this is not an exhaustion claim or a
new installed-image qualification.

## Implementation

- Decode eight original FP4 coefficients per thread into one aligned 128-bit
  FP16 store. The original scalar store remains for unaligned output views.
  Exact FP4/E4M3 conversion and half rounding are unchanged. Sixteen coefficients
  per thread was measured and superseded by the eight-coefficient layout.
- Gather prompt expert outputs directly into the slot-ordered FP32 sum, avoiding
  the `[rows,8,2816]` temporary (88 MiB at 1,024 rows per rank). Apply each expert
  row multiplier there rather than scaling the whole padded output. Separate
  round-to-nearest multiplications and additions preserve original operation
  order, absent-slot behavior, signed zero and strided output.
- Skip ordinary text spans in the shared markup lexer and Gemma channel splitter.
  Existing recognition, quoting, withholding and tool-authorization boundaries
  remain in place. No incremental parser state is introduced.

## Measured source and environment

Reference: `98c535f60d66d260bc2d802f886fbb7204733d1f`. The measured checkout also
contained pre-existing changes to shared communication code; these were held
constant across references and candidates and are not included in this task's
commit. Therefore these results do not qualify a clean image built from the
commit alone. The private source archives and manifests retain the exact trees.

The runtime source digest is
`5a897588e55ab8bd1f8be60831bd356cb33dbfde9920674481005bda8436d7b1`:
SHA256 of each path, a NUL byte, and file bytes, in this order:
`src/tensorfold/cuda/sm70/gemma4.cu`,
`src/tensorfold/cuda/sm70/gemma4_bindings.cpp`,
`src/tensorfold/families/gemma4/sm70/target.py`,
`src/tensorfold/markup_lex.py`,
`src/tensorfold/server/gemma_values.py`.

- Eight V100-SXM2-32GB GPUs, driver 580.178.04, 150 W cap per GPU.
- Runtime image `sha256:cf8c0ff6baf353ce6a2a548330855631dee1c09906885968b77154b50296e11b`,
  source mounted read-only; Torch `2.16.0.dev20261111+tfcu129sm70`, CUDA 12.9.
- AEON NVFP4 target revision `90c8601b`, DFlash revision `77d42027`;
  numerical profile `gemma4-sm70-w4a16-v1`.
- TP8/ETP4, one request lane, 32,768-token allocated context, FP16 KV,
  `TF_SMALL_GATHER_BYTES=262144`, `OMP_NUM_THREADS=1`, CUDA graphs;
  DFlash uses `--drafter-bits 8 --draft-rows 5`.
- Google chat template SHA256
  `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`.

## Results and limits

Three matched requests per case used fixed prompts and request order. Each long
prompt was unique within a run, identical across variants, with zero cached
tokens. The model was loaded and warmed; these are fresh-prompt results, not
cold model or cold hardware-cache results. Long-prompt requests generated one
token; their complete-request latency is a prefill proxy including API overhead.

| DFlash HTTP prompt tokens | Reference median request seconds | Checkpoint median request seconds | Throughput gain |
| --- | --- | --- | --- |
| 4,010 | 1.650819 | 1.065797 | 54.9% |
| 16,010 | 6.711888 | 4.371614 | 53.5% |

All 15 DFlash requests (three generations each of code, prose and lists at 512 tokens plus
the six long-prompt requests) matched reference reply text, reasoning and token
counts. Draft acceptance counts can vary; exact proposal identity is not claimed.
Short-request complete-throughput medians improved by 1.0–3.9%; the small sample
does not establish a general decode gain or equivalence bound.

An earlier exact vector8/parser increment, before row-scaling fusion, also
passed all 15 serial requests with identical outputs. Prose/list streaming
100-delta block means stayed approximately 5.0–5.22 ms through 500 deltas, versus
earlier runs whose late blocks grew to approximately 8 ms. Complete-request
medians improved 21.2% for prose and 18.1% for lists in that comparison. This is
evidence for removing the observed parser stall, not a universal token-rate
claim. DFlash SSE deltas may contain multiple tokens and are not per-token clocks.

The final checkpoint's forward-only 4K/16K medians were 3,998/3,843 tokens/s;
all six final-logit hashes matched the original reference. HTTP measurements
above are the relevant complete-request result. All samples, including slower
first requests, are retained; no regional loss is dismissed by averaging cases.

## Verification and continuing work

- Native Gemma kernel suite: 16 passed, including exact scale/code combinations,
  aligned/unaligned stores, ordered expert scaling/combination, exceptional
  values, signed zero, subnormals and adjacent-output canaries.
- Shared parser, tool and grammar suites: 634 passed, six MLX-dependent tests
  skipped on this Linux host. This does not claim native MLX execution.
- Isolated baseline/candidate parser processes matched 73,050 streaming-prefix,
  flag, trusted-boundary and strict-header records exactly.
- Static task-wide review found no unresolved correctness finding in these
  changes. Existing unrelated edits remain preserved.

Private evidence is under `task-artifacts/gemma4-optimization-20261115/` in the
development checkout: the ledger, immutable source manifests, commands, complete
logs and request JSONL files bind each result. These artifacts are not production
dependencies. The first diagnostic's teardown stall and its correction are
retained; only later runs with successful teardown are clean passes.

The final row-scaling increment's serial HTTP comparison, broader workload and
full 262,144-token qualification remain pending. Expert-padding reduction and
other measured bottlenecks remain under investigation. A proposal-only FP16
prefill alternative remains unadopted because its numerical-domain proof is
unfinished. No capacity, media, concurrency or new-model qualification is added.
