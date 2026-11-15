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
- Pack experts in contiguous groups of sixteen for prompt chunks of at least
  64 rows, reducing padding to each group's largest population. Shorter chunks
  retain the original single-group geometry and scalar count read. Both use the
  same BMM and ordered-combination implementation; no precision flag changes.
- Capture integer position/token copies with their decode consumers. Target and
  drafter share their read-only start position; previous-position copies run only
  when needed. Replay submits the existing pinned host-to-device input copy.

## Measured source and environment

Reference: `98c535f60d66d260bc2d802f886fbb7204733d1f`. The measured checkout also
contained pre-existing changes to shared communication code; these were held
constant across references and candidates and are not included in this task's
commit. Therefore these results do not qualify a clean image built from the
commit alone. The private source archives and manifests retain the exact trees.

The `ef97c31d` checkpoint's six-file runtime source digest is
`88d4dea83a5ce68a7b9bbb7d40f4a92692219a1293fe8ac1e2f12ca49f28797d`:
SHA256 of each path, a NUL byte, and file bytes, in this order:
`src/tensorfold/cuda/sm70/gemma4.cu`,
`src/tensorfold/cuda/sm70/gemma4_bindings.cpp`,
`src/tensorfold/families/gemma4/sm70/target.py`,
`src/tensorfold/families/gemma4/sm70/graphs.py`,
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

## Results and limits at ef97c31d

Three matched requests per case used fixed prompts and request order. Each long
prompt was unique within a run, identical across variants, with zero cached
tokens. The model was loaded and warmed; these are fresh-prompt results, not
cold model or cold hardware-cache results. Long-prompt requests generated one
token; their complete-request latency is a prefill proxy including API overhead.

| HTTP mode | Prompt tokens | Reference median request seconds | Checkpoint median request seconds | Throughput gain |
| --- | --- | --- | --- | --- |
| DFlash | 4,010 | 1.650819 | 0.923567 | 78.7% |
| DFlash | 16,010 | 6.711888 | 3.830779 | 75.2% |
| Serial | 4,010 | 1.647839 | 0.884093 | 86.4% |
| Serial | 16,010 | 6.690121 | 3.733353 | 79.2% |

All 15 requests in each mode (three generations each of code, prose and lists at 512 tokens plus
the six long-prompt requests) matched reference reply text, reasoning and token
counts. Draft acceptance counts can vary; exact proposal identity is not claimed.
Short-request complete-throughput medians improved by 1.7–5.2% with DFlash and
5.8–26.8% serially; these include prefill and parser costs. The small sample does
not establish a general decode gain or equivalence bound.

An earlier exact vector8/parser increment, before row-scaling fusion, also
passed all 15 serial requests with identical outputs. Prose/list streaming
100-delta block means stayed approximately 5.0–5.22 ms through 500 deltas, versus
earlier runs whose late blocks grew to approximately 8 ms. Complete-request
medians improved 21.2% for prose and 18.1% for lists in that comparison. This is
evidence for removing the observed parser stall, not a universal token-rate
claim. DFlash SSE deltas may contain multiple tokens and are not per-token clocks.

The preceding `c08ff79b` checkpoint (before grouped packing and captured input
copies) measured DFlash gains of 54.9%/53.5% and forward-only medians of
3,998/3,843 tokens/s at 4K/16K; its six final-logit hashes matched the original.
Those results retain that earlier source scope. All samples, including slower
first requests, are retained; no regional loss is dismissed by averaging cases.

A separate synchronization-safe ABAB graph diagnostic compared input staging
alone. The two reference/candidate run means were 4.590/4.492 and 4.588/4.466 ms
for serial steps, 9.226/9.196 and 9.235/9.202 ms for one-committed-row rounds,
and 9.462/9.381 and 9.423/9.396 ms for five-committed-row rounds. Each replay
completed before pinned input reuse; recorded output probes matched. This is a
synthetic diagnostic, not attribution of the complete HTTP gain. Earlier tight
replay loops without safe pinned-buffer reuse are excluded from qualification.

## DFlash context-KV follow-up

The next increment packs each layer's original context K/V rows into an
immutable bank after the selected BF16 or INT8 preparation. Context append
projects the common input once, then uses strided layer views for the existing
cache kernels. Proposal-layer QKV weights and target arithmetic are unchanged.
INT8-to-FP32 cached preparation also avoids a redundant conversion temporary.
Larger GEMM shapes can change arithmetic order; acceptance uses measured quality,
not an assumption of bitwise equivalence for every projection.

Its seven-file runtime digest is
`bc06443852d9e8717bc8b1550af332a4a2a94f5acfc9fa953d93e7bd22df5add`,
using the six paths and hashing method above with
`src/tensorfold/families/gemma4/sm70/dflash.py` inserted immediately after
`src/tensorfold/families/gemma4/sm70/graphs.py`. The same runtime and dirty-source
qualification limits apply. Comparisons below use the preserved pre-bank source,
not the original `98c535f6` baseline, with 32K allocated context.

| Drafter configuration | Code | Prose | Lists | 4K prompt | 16K prompt |
| --- | --- | --- | --- | --- | --- |
| INT8, five rows | +0.77% | +2.17% | +0.08% | +2.64% | +0.36% |
| BF16, sixteen rows | +0.84% | +0.59% | +0.55% | +3.70% | +4.28% |

These are complete-request median throughput differences from three matched
requests per case. All 15 replies and token counts matched in each configuration;
all paired BF16 request latencies improved. Small samples do not establish
universal gains. Repeated INT8 graph diagnostics showed approximately 0.6–1.2%
round gains; wide BF16 one-committed-row diagnostic results were mixed and are
not claimed as a win. Selection uses the complete-request results, not a claim
that every internal timing improved.

Thirteen new native tests passed for packing, strided cache writes, canaries and
dtype promotion, covering BF16/INT8, two/five layers and 1–1,024 input rows.
The original and BOS-sensitivity quality corpora produced identical prefill and
decode scores with five-row drafting for both formats; repeated scores matched.
The sixteen-row BF16 checks above are native and serving comparisons, not a
separate teacher-forced corpus qualification at that width. INT8 also matched the
two long-context stress cases (20,812 prefill and 524 decode positions). Earlier
24 native tests and parser coverage apply to unchanged code. BF16 long-context
quality and full 262K qualification are not newly claimed.

Summed post-initialization Torch allocations increased by 30,973,952 bytes
across eight ranks for the default INT8 configuration; summed reserved and peak
reserved memory were unchanged. In the wide BF16 comparison, summed allocated
bytes changed from 37,804,859,392 to 37,866,201,088, reserved from 39,871,053,824
to 39,845,888,000, peak allocated from 43,297,484,800 to 43,350,962,176, and peak
reserved from 47,135,588,352 to 47,152,365,568. Sums of per-rank peaks are not
simultaneous physical peaks. These measurements do not increase a capacity claim.

## Verification and continuing work

- Native Gemma kernel suite: 24 passed, including exact scale/code combinations,
  aligned/unaligned stores, ordered expert scaling/combination, exceptional
  values, signed zero, subnormals and adjacent-output canaries.
- Quality acceptance permits different outputs when quality is unchanged or
  improved. The existing nats/perplexity harness uses a predeclared 0.005
  nats/token margin and 4,000 document-blocked bootstrap samples, seed zero;
  established losses, including `worse-within-margin`, are not accepted.
  G16 scored identically to the original on the 16-document frozen corpus,
  its separately frozen raw-text BOS sensitivity condition, and two templated
  4K/16K stress documents. These are separate matched comparisons, not pooled
  corpora or proof that adding BOS improves calibration. Regional packing plus
  input staging also matched scores on four short-boundary cases (32, 33, 45
  and 1,055 prompt tokens); repeated scores matched. Larger-chunk G16 evidence
  is reused where dispatch is unchanged, with structural integer-copy review
  for staging. Identity is an observed result, not the general quality gate.
- Shared parser, tool and grammar suites: 634 passed, six MLX-dependent tests
  skipped on this Linux host. This does not claim native MLX execution.
- Isolated baseline/candidate parser processes matched 73,050 streaming-prefix,
  flag, trusted-boundary and strict-header records exactly.
- Three cancellation/recovery cycles cancelled different 33-, 45- and
  16,518-token prompts after two content deltas, then issued a fresh 32-token
  request. Its 64-token reply and usage matched the clean reference each time;
  the test server stopped successfully.
- Static task-wide review found no unresolved correctness finding in these
  changes. Existing unrelated edits remain preserved.

Private evidence is under `task-artifacts/gemma4-optimization-20261115/` in the
development checkout: the ledger, immutable source manifests, commands, complete
logs and request JSONL files bind each result. These artifacts are not production
dependencies. The first diagnostic's teardown stall and its correction are
retained; only later runs with successful teardown are clean passes.

Grouping alternatives G8 and G4 were rejected for measured quality losses;
G2 and the tested G8 cuBLASLt configurations remained inconclusive and were not
adopted, without claiming they are slower or universally inferior. Native exact
checks remain appropriate for transformations promising exact arithmetic.

Broader workloads and full 262,144-token qualification remain pending. Other
measured bottlenecks remain under investigation. A proposal-only FP16
prefill alternative remains unadopted because its numerical-domain proof is
unfinished. No capacity, media, concurrency or new-model qualification is added.
