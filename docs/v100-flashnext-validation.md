# Flash Next on eight V100 GPUs

Implementation base: `kklouzal/TensorFold`, branch `python-0.6`,
`94d84f510f87ee27950d821626a3510bff9c5625`, 2026-10-24. Native model runs use the
existing TensorFold CUDA family engine, scheduler, tokenizer/chat preparation
and HTTP server. No vLLM inference or vendor operators are used.

## Current repaired target checkpoint (C21)

C21 source archive`f781094322c6083bbb1986475d1db84aadd45b462401d3100aefce4de8c54e0b`,
installed image`28061391762e292cf85ad37a865dc47c13a914f0fdc5d461a8767c3699f26f98`,
passed both fixed teacher suites independently. No threshold changed:

| Suite | Teacher tokens | Mean KL | Perplexity ratio | Top-1 agreement |
| --- | --- | --- | --- | --- |
| Original | 325 | 0 | 1 | 1 |
| Broader | 389 | 0.0000254873671 | 1.00004296853 | 1 |

The combined714-step mean KL was0.0000138859745 and perplexity ratio1.00002340980.
All691tokens outside the three long retrieval cases matched distributions
exactly. Those three approximately6400-token cases have nonzero distribution
variation: mean KL0.0004504/0.0002729/0.0005897, with full top-1 agreement.
The fixed broader-suite aggregate gates passed. Actual native attention counters
were positive in each long case (84/84/72calls, maximum context6443/6444/6441).

The authority uses original family/attention/reference-bank arithmetic with
common prepared-linear/router mechanical paths previously proven byte exact.
Candidate execution uses canonical packed experts and native family/attention.
Both share unchanged weights, complete vocabulary, int8 KV and fixed teacher IDs;
MTP is absent, graphs disabled, and every case resets before prefill. This is a
relative same-checkpoint distribution gate, not BF16/HF equivalence or full-window
quality coverage.714original Float32 vectors with file hashes and exact input/
teacher IDs were retained for the separate hybrid EP comparison.

Installed checks passed2149CPU cases (2skips,35subtests); native integration281,
full-forward short-window fallback4, and legacy attention compatibility331 passed.
The fallback regression covers R1/R3 and raw/int8 without changing State positions.
Out-of-scope former native/GQA cases explicitly refuse or use original Torch;
retired experimental sources remain archived. No tolerance was widened.

`TENSORFOLD_SM70_PROFILE=optimized` is an explicit normal-engine selection;
`reference` remains the general default. The optimized profile requires the
pinned runtime and TP8 Flash Next geometry, agrees its identity before payload
loading and prepares ownership before States/graphs. Native attention is limited
to real first-query ends65–8192, localH3/KV1/D256 and at most24rows per State.
Unsupported frames decline graph binding and use original per-query Torch math.
KV capacity remains the ordinary admitted limit. The C22 capacity patch caps
native scratch at8192 without changing CUDA arithmetic; 25CPU/7GPU controls
passed with an actual262144-token KV owner and original per-query oracles.
The conservative original-Torch fallback bound is1.25GiB, below the unchanged
2GiB serving workspace. Full-service allocation and performance remain pending.

The actual hybrid expert-parallel run completed all714teacher steps and failed:
baseline325 meanKL0.000204397147; broader389 meanKL0.001012661232, perplexity
ratio1.003518535 and top-1 agreement0.984575835. It remains unselected; no
throughput result is claimed.

C21 live baseline12/12 passed, but draft8/9 and concurrent2/6 failed exact-ID
checks. Exclusive cold/cached controls isolated a sampled cached-suffix issue;
concurrent failures followed joint cold-prefill shapes. C22 declines only
positive-temperature prefix borrowing in the optimized SM70 profile and
schedules independent prompts in separate prefill passes, preserving drafting
and decode batching. C22 actual serving acceptance below resolves both failures, including
complete R24 concurrent response identity after the correction.

## C22 serving and capacity closeout

C22 source archive`58d321629b528624d1a70471fcd0b9fdf64ab06724eba75b728f71e7775ccf0a`,
installed image`ea40a2b74843614fbc49e790087ad83e0998d21384b6b7db773de0de169aa68b`,
passed2679CPU checks (2skips,41subtests);607installed runtime source hashes match.
The runtime-identical initial build passed338GPU checks, including large262144
KV ownership with exact per-query native outputs/replay through8192 and331
compatibility cases. The final revision repairs eight incomplete CPU fixtures;
its runtime bytes are unchanged. All fixture assertions were preserved.

Actual TP8/DSpark depth3/int8/parallel6/context8192/full-windows6 serving passed
all9serial/drafted128-token pairs and12baseline cases, with exact IDs/messages.
All six simultaneous registered128-token R24 replies matched the serial answer.
Full SSE, cancellation cleanup and cached127 prefix recovery passed. Actual
R24 replay51/capture96 entries were observed; retained graph owner877093376bytes
fit the8327917052-byte allowance. Proposal330calls/readbacks and context61frames/
readbacks agreed. All eight ranks reported six materialized8196-slot caches
(context8192 plus speculation reserve),635747328owned cache bytes per rank.
These are bounded serving checks, not throughput or full-window quality claims.

Normal CLI `--parallel N --kv-full-windows N` allocates equal complete target
and drafter windows for resident GGUF. Admission protects all selected windows,
replacement overlap and2GiB runtime workspace, retaining the4GiB device floor.
Normal idle retirement preserves these allocations; error recovery restores the
full window before publishing a healthy slot. Ordinary `/stats.capacity_plan`
reports the admitted policy and startup allocation receipts for all TP ranks.

With the exact capped attention byte helper, plans fit DSpark2×262140 with
1464294436bytes of graph room, or stock MTP3×262144 with692445232bytes.
Actual startup and full-window MTP serving acceptance are recorded below;
no262K generation/quality workload was run.

## Full-window method selection and final acceptance

All runs use installed C22 image`ea40a2b74843614fbc49e790087ad83e0998d21384b6b7db773de0de169aa68b`,
TP8 on eight V100-SXM2-32GB/200W, INT8 K/V, confidence0, the pinned target/drafter
and explicit optimized profile. Every selected full window was physically
materialized on every rank. Quiet single-client screen:128prompt tokens,
three512-token greedy replies, thinking disabled, no profiler or other GPU job.

| Method/depth | Equal windows × context | Three decode tokens/s |
| --- | --- | --- |
| DSpark2 | 2 ×262141 | 48.3462 /49.6908 /53.4866 |
| DSpark3 | 2 ×262140 | 50.4079 /53.9212 /54.6653 |
| DSpark7 | 2 ×262136 | 46.7392 /48.4696 /52.9825 |
| Stock MTP2 | 3 ×262144 | 51.1568 /50.4731 /53.3553 |
| Stock MTP3 (selected) | 3 ×262144 | 52.8635 /53.2930 /58.9348 |

All four complete choices/ID sequences, including the initial128-output prime,
matched across these C22 runs. MTP had cached127tokens for its final repeated
prompt; DSpark's two-slot runs evicted that prefix and were cold. This is best
measured configuration selection, not an isolated-method causal gain or a
randomized small-effect significance claim. DSpark depth3 beat2/7 on all three
matched screens and remains the documented DSpark choice; both methods remain
in the maintained codebase. Earlier C20 first three benchmark outputs differ
from C22 (first differences46/125/429); C20 target arithmetic failed the broader
quality gate and is not a matching-quality authority. Its final repeat matched.

MTP3 cold128-token prefill was2.4306/2.5245/2.5710s (52.66/50.70/49.79tokens/s).
The cached127 repeat prefilled in0.1691s, TTFT0.1720s. DSpark3 cold128-token
prefill on512-output replies was2.1798/2.0499/2.1492s (58.72/62.44/59.56tokens/s).
Configured maximum context is not actual prompt length in these measurements.
100single-request decode tokens/s remains unmet.

Actual MTP3/all8rank startup allocated three262148-slot caches each
(262144context plus speculation reserve),8833339008owned cache bytes/rank.
Full-context reservation acceptance passed: all three simultaneous128-token
replies matched the serial reference completely; fullSSE, cancellation cleanup
and healthy cached127prefix recovery passed. Fresh9/9serial/drafted128-token
pairs and12/12baseline cases matched IDs/messages. ActualR12 replay52 and
exactR12capture96 were observed. Common graph ownership653821400bytes fit
692445232bytes (13verify/24MTPkeys), with no refusal or reserve/context change.
The authoritative full-window functional summary is
`task-artifacts/v100-optimize/candidate22-final-mtp-d3-functional/summary.json`
(SHA1da7efbd8cd823e6e33832d22809519fa9c569bb10b6b856ac41761a75ca0c64).

The DSparkD7 full-window sampled reason19 serial/draft pair also matched all128
IDs/messages, bothcold0, accepted92/245; it matched the short-window D3 reference.
Full DSpark2-window startup owned6777995264cache bytes/rank. The six-slot8192
DSpark functional suite above establishes concurrent serving; no complete262K
prompt or two simultaneous262K workloads were generated for DSpark.

R32 canonical component/full-bank protocols and actual R24 serving passed.
The separate hybrid EP2 full714 quality trial failed and remains private and
unselected. The registered TP4 graph-memory experiment refused budget with no
throughput sample; its authorized fallback did not qualify a gain. These finite
experiments are closed. Final plain port8000 restoration/verification passed as recorded below.

## Shipping service verification

Ordinary Compose project`tensorfold-closeouts` runs healthy container
`tensorfold-flash-next` on the preserved port8000 binding. Immutable image
`1443fc46f8df311d72024b8444b97840302064b24ce18a2005fa9aa0745a2b8e`
comes from shipping archive
`174c6bcd216e19652b18529bca932577d785d09b915e5509a71c0dd4dbca8eb7`.
Its607installed runtime hashes match C22; only documentation changed from the
qualified image. The normal `tools/launch_cuda_tp.py` starts all eight ranks.
No private experiment observer, startup wrapper or runtime patch is installed.
Read-only model mounts, external bind, alias and cache volume were preserved.

Plain service configuration: MTPdepth3/confidence0, INT8K/INT8V, context262144,
parallel3/full-windows3. Ordinary `/stats` confirmed all8rank full allocations.
Three simultaneous128-token replies matched the qualified serial IDs/messages
completely. All four benchmark outputs also matched the private C22 authority.
Quiet single-client128-input/512-output decode was54.0358/54.4202/59.7780tokens/s;
cold128-token prefill2.1836/2.1994/2.1978s (58.62/58.20/58.24tokens/s).
The127-token cached repeat prefilled in0.1365s, TTFT0.1393s. These shipping
measurements are recorded independently from private-observer screens and
are not a causal estimate of observer removal. No other GPU job was active.

Authoritative receipt: `task-artifacts/v100-optimize/FINAL-CLOSEOUT.json`;
raw benchmark`final-service-live/`, plain concurrent proof`final-service-functional/`.
The original C3 container/image is preserved stopped under
`tensorfold-flash-next-c3-rollback-20261024`; original deployment files and
private environment retain their permissions in the host rollback directory.
Only the shipping service and existing Portainer remain running.

All four finite closeouts are complete.100single-request decode tokens/s remains
unmet; optimization pauses at the user's agreed stopping point. No full262K
prompt/generation quality workload is claimed.

The C22 shipping image admitted rawFP16 or nativeINT8 independently for K/V.
CurrentINT8 uses H32 groups of32 and FP16 scales; it is lossy and distinct from
RotorQuant. That image rejected the other fork codecs before payload loading;
the Q4 extension below adds two existing four-bit descriptors. The fixed numerical authority uses the same
INT8 cache baseline, so these gates do not establish FP16-KV equivalence.
Changing cache precision needs fresh memory admission and configuration fit.

## Model and foundation

The read-only local Huihui UD-Q4_K_XL V2 checkpoint contains 1,224 tensors in
four GGUF shards; its first shard contains metadata only. Encoding counts are
557 F32, 503 Q8_0, 94 Q4_K, 43 Q5_1, 24 BF16, two Q5_K and one IQ4_NL. Original
packed tensor extents total 111,323,630,080 bytes. The IQ4_NL PLE table has shape
`[320001536, 160]`, with 90 bytes per row. Inventory validates metadata and extents,
not a full payload checksum. The matching separate stock MTP has 31 BF16 tensors;
the loader creates explicit FP16 execution arrays and preserves the target embedding/head.

Host `ow-ai-01` has eight Tesla V100-SXM2-32GB GPUs (SM70) and 376 GiB RAM. The AMD64
CPython 3.12 image pins CUDA 12.9.2 compilation (NVCC 12.9.86), Torch 2.14.1+cu126,
TorchVision 0.29.1+cu126 and Triton 3.8.0. Native SM70 paths use CUDA/ATen; they
never invoke unsupported Triton kernels. The modern CUDA 13/GB10 foundation and
BF16 model providers retain their existing contracts.

## Installed evidence

| Candidate | Check | Result and scope |
| --- | --- | --- |
| 1 | CPU runtime | 1,392 passed, 1 skipped; GGUF/tokenizer/weight mapping, TP protocol and existing CLI/capacity/family checks |
| 1 | Deployment fixtures | 194 passed, using `/src/tests` with the installed-wheel pytest configuration |
| 1 | Single V100 | 49 passed; native scalar/FP64 oracles, all encodings, full tiny family, raw/INT8 KV, prefix state and graph replay |
| 1 | Native collectives | 2 passed; 4/8 ranks, FP16/FP32/int64 transport, EOS and captured all-gather replay |
| 1 | Stock MTP components | 4 passed against the installed wheel; FP16 mm/bmm, dynamic routing graph and shared MTP forward/cache |
| 2 | CPU runtime/deployment | 1,623 passed, 1 skipped, 2 subtests; includes source-relative deployment/probe fixtures while importing the installed package |
| 2 | Single V100 | 53 passed in 34.18 s, including stock MTP components |
| 2 | Native NCCL/scheduler | 4 passed in 77.33 s; 4/8 ranks, collective replay and two requests through the actual family/MultiDecoder/Scheduler with global sharded sampling |
| 3 | CPU runtime/deployment | 1,680 passed, 3 skipped, 2 subtests in 9.68 s; includes cache boundaries, recovery/MTP digest, PLE admission/lifetime and masked attention cases |
| 3 | Single V100 | 58 passed in 63.93 s; includes masked V regressions and 2,049-row GDN kept-prefix parity |
| 3 | Native NCCL/scheduler | 4 passed in 67.68 s at 4/8 ranks; actual concurrent family execution matches sequential decoding |
| 3 | Native receipts | RMSNorm, GGUF decode and GDN chain/replay smoke passed; all three binary hashes, compiler identities and SM70 cubins recorded |

Independent real-checkpoint slices for F32, BF16, Q8_0, IQ4_NL, Q5_1, Q4_K and
Q5_K matched the external GGUF decoder exactly. Source-only audit tests cover
PLE lock lifetime/staging accounting, TP recovery ownership/MTP control digest,
masked future/sparse V rows, and 2,049-row GDN 2/6 state with kept cuts at 511/1025/2048.
Candidate 3 includes these audit fixes and passes the installed matrix above.

## Full-model results

Candidate 1 ran TP8, raw FP16 KV, no drafting, one request at a time and a
4,096-token window. Completion and chat produced correct answers. The sampled
France chat generated 37 tokens, including reasoning, and matched the existing
vLLM baseline's response; EOS stopped normally. Repeated and streamed requests
preserved token hash `7b89c1c412ee`; prefix reuse read 63 of 64 prompt tokens. A
request with thinking disabled returned the arithmetic answer 42. A 2,695-token
prompt crossed the 2,048-row prefill boundary and correctly retrieved its secret word; measured
prefill was 62.30 s while component qualification also ran on the host.

Candidate 2 ran TP8, raw FP16 KV, stock MTP depth 2, two slots and an 8,192-token
window. Drafting-disabled and MTP France requests produced the same 37 tokens
and token hash. MTP accepted 22 of 24 proposed tokens in that sample. Concurrent
France/arithmetic requests completed correctly, with prefix reuse and distinct
EOS timing. These are real model checks, not an exhaustive quality benchmark.

On this single short greedy request, candidate 2 recorded 3.23 decode tokens/s
without drafting and 6.71 with MTP, 5.36 s MTP decode time and 1.63 s prefill.
Comparison order was reference then MTP; prompts were 64 tokens and replies
37 tokens, with eager execution, warm weights and initially uncached prompts.
There are no repeated trials or
confidence bounds. These observations are not a general throughput claim or a
matched comparison against the original vLLM's FP8 cache and graph policies.

Candidate 1 startup took 418.1 s, including 306 s waiting on PLE reads during actual
`/dev/sda` SATA timeouts/link resets and Btrfs checksum errors. After kernel
recovery/NCQ disablement, candidate 2 startup took 126.1 s, including 46.9 s warmup.
The storage fault remains an independent host reliability concern; successful
loading does not establish disk health.

Candidate 3 is deployed at port 8000 with TP8, INT8 H32 KV on both sides,
stock MTP depth 2, six request slots and a 237,598-token per-request limit.
Header admission estimated 17.90 GiB per rank within a 27.09 GiB budget.
Startup took 135.5 s, including 57.8 s warmup. The decoder reported 12.4 GiB
available for opportunistic cache growth and 2.49 GiB for one full window;
several slots share this room.

In that deployed configuration, reference and MTP France requests preserved
hash `7b89c1c412ee`; MTP accepted 21 of 25 proposals. Six simultaneously submitted
requests returned the expected France and arithmetic answers with normal EOS.
The France request reused 63 prompt tokens. Streaming and a lenient weather
function call passed, including parsed JSON arguments and `tool_calls` finish
reason. The 2,695-token secret-word request also passed with INT8 and MTP,
using 75.89 s prefill. A streaming client was disconnected after its first content chunk; the next
request completed correctly, exercising cancellation and state reuse.

The final short reference/MTP sample recorded 2.60/5.04 decode tokens/s, with
INT8 and six configured slots. As above, this is one ordered warm sample,
not a repeated benchmark. The raw FP16 candidate 2 observation uses different
cache and slot settings and must not be presented as the deployed throughput.

## Repeated live requests, 2026-10-24

The unchanged candidate 3 service was tested through `/v1/chat/completions` on
eight V100s with existing 200 W per-GPU power limits. The reported application
clock setting was 1,290 MHz; actual SM clocks reached 1,530 MHz under load, so
1,290 MHz must not be interpreted as a fixed SM clock cap.
TP8, INT8 H32 KV, stock MTP depth 2/confidence 0.7, six configured slots, a
237,598-token limit and eager execution were retained. Weights, native kernels
and filesystem caches were already warm. Here **uncached** means a KV prefix
miss, not model startup or a cold disk.

Three distinct prompts at each exact rendered size (128/1,024/2,048 tokens)
were interleaved. Each prompt requested a sustained transformer explanation;
temperature was 0, top-p 1 and thinking disabled. Every timed reply generated
64 tokens. Each cold request was immediately repeated with the same prompt.
All nine pairs produced matching generated-token hashes. The client harness
paused after one repeat miss to inspect the existing cache policy, then resumed
only the unfinished requests; the server was neither restarted nor modified.

Rates below use `(prompt_tokens - cached_tokens) / prefill_s` and
`(completion_tokens - 1) / decode_s`, matching the CUDA HTTP decode definition.
Prefill is the server-attributed prompt processing/admission time; TTFT includes
additional request work and is reported separately. Values are medians with
observed min–max ranges across three requests, not confidence intervals or peak
hardware throughput estimates.

| Prompt tokens | Uncached prefill tokens/s | Decode tokens/s | TTFT, seconds |
| ---: | ---: | ---: | ---: |
| 128 | 32.57 (32.56–32.63) | 3.94 (3.94–4.43) | 3.934 |
| 1,024 | 35.45 (35.30–38.02) | 4.28 (4.28–4.44) | 28.896 |
| 2,048 | 35.66 (35.63–35.76) | 4.21 (4.21–4.22) | 57.436 |

Actual cache-hit results are separated from the one repeat miss. Cached hits
reused all but the final prompt token, so dividing the complete prompt length
by resume time would misstate prefill throughput. With only one newly processed token, this measures resume plus one-token
processing latency rather than comparable bulk-prefill throughput; TTFT is
more useful here.

| Prompt tokens | Repeat hits | Reused tokens per hit | Cache-hit TTFT, seconds | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 2/3 | 127 | 0.402 | 4.18 |
| 1,024 | 3/3 | 1,023 | 0.402 | 4.30 |
| 2,048 | 3/3 | 2,047 | 0.406 | 4.20 |

The 128-token miss is permitted by the existing GreedyDual-Size prefix policy:
with eight valuable long-prefix entries retained, a newly created short prefix
can be evicted immediately. It returned the same output hash and performed a
full prefill. No cache/runtime policy was changed for these tests.

One batch of six fresh 128-token chats produced 64 tokens each. All six returned
HTTP 200 with zero cached tokens, nonempty content and `length` finish reasons.
Aggregate completion throughput was **6.15 tokens/s over 62.48 s**, including
admission, shared prefill, decode and HTTP. Per-request decode rates were
**1.51–1.57 tokens/s**; these rates are not summed. This is one batch, not a
repeated concurrency estimate.

The host remained healthy with no new matching GPU/storage/OOM kernel errors
between the pre-test and post-load snapshots. GPU temperatures were 37–52°C
during functional requests, with no observed throttle events. The pre-existing
Btrfs corruption counters were unchanged (sda2: 397, sdb: 0); no repair or runtime
configuration change was performed.

All seven live functional cases passed: expected HTTP 404/400 and healthy
recovery, MTP/reference chat hash agreement, plain JSON output/natural EOS,
SSE content/reasoning/usage/EOS agreement, streamed function call plus tool-result
follow-up, real client-disconnect cancellation/ownership recovery, and six
simultaneous mixed MTP/reference requests. The mixed run observed six in-flight
requests and preserved their reference output hashes. An asymmetric whitespace
comparison in the private SSE harness was corrected; the preserved raw reasoning
already matched, and the targeted rerun passed. The original packet and corrected
rerun are both retained; no server/runtime code changed during live testing.

Raw prompts, HTTP responses, timings, host snapshot, corrected harness and CSV
are retained privately in `task-artifacts/v100-live-tests/`. The excluded raw
completion pilot ended at EOS before any decode interval; its artifacts remain
under `pilot-raw-completions/`. This request matrix does not qualify a full
237,598-token prompt, exhaustive model quality, sampled generation, optional
grammar enforcement or a performance comparison against another runtime.

## Image identities and reproduction

```text
candidate1 source archive SHA256: 8edec91e578c4dfc68381ae84d055a0897efb62a04c9b91391bd21f368f9b8fd
candidate1 image: sha256:b05eccc4e758d4944bf6e6dbd363d7803dd7c1421ea0524a93fe6f8306f6716d
candidate2 source archive SHA256: 63134816455eaf59159dfce8d5758e5b9568ad038f8fe2876e2c63fd7eeb4e97
candidate2 image: sha256:48957dfca8b1cc44c43ce0524b78d08f3537bc143771de5b3cdaadd55234725f
candidate3 source archive SHA256: 3c87703c9577baaeb35df9e53285268cb7a5577753a90392ba0fee3ce3fc9f8f
candidate3 image: sha256:b223074bfe57d6a4c9a706a49b0eb97ecc0ed73d37ff93170ba419470f655d8d
candidate3 wheel SHA256: c6de9f71ba10adda0049c9419e5b5eda06e5eca08a4979b0781630dfbd96a2c4
```

[Build, component checks and serving commands](../deploy/v100/README.md).
Private reproducibility packets, actual request/response JSON, source archives,
image/build logs and task decisions are retained under
`task-artifacts/v100-full/`; native binary/compiler/cubin receipts are in
`task-artifacts/v100-full/candidate3-component-receipt.json` (earlier receipts
remain under `v100-port/`). All 576 installed Python/native source and data files
matched the final checkout byte-for-byte. Documentation updates and a test-only
lint annotation followed the candidate archive without changing runtime bytes.
Maintained tests have no dependency on these private artifacts.

Full-model CUDA graphs remain disabled: concurrent TP requires a
qualified pointer-routing adaptation. SM70 cache admission supports raw FP16 and
the existing INT8 H32 codec, including mixed sides; FP8/RotorQuant formats are
not claimed. Grammar extra packages are optional and outside this foundation.
Long-window configuration/admission is separate from a long-window quality or
six-full-window concurrency benchmark.

## Native optimization checkpoint

The ongoing target is **100+ single-request decode tokens/s**, holding original
weights, INT8 H32 KV, FP16 activations and FP32 recurrent state fixed. Aggregate
concurrency is measured separately. The target remains unmet; candidate 3 is
preserved for rollback while private candidates are evaluated.

The isolated packed-expert candidate, image
`sha256:df20ddef2ce718a12f22944c15ba4ca75e8fc3591dd4db10afd5cfc7542a31bd`,
matched all token IDs and complete messages in 12 HTTP cases. Paired scoring
on 325 fixed teacher steps measured top-1 agreement 1.0, mean KL
`5.921239837e-5`, mean NLL delta `-0.000341674882` and perplexity ratio
`0.9996583835`. Predeclared gates were perplexity ratio at most 1.001, mean KL
at most `1e-4`, and top-1 agreement at least 0.995. These gates passed. A quiet
128-input/128-output request measured **5.23 decode tokens/s**, with 24.2935
decode seconds; this single observation has no confidence interval.

Combined candidate 5 adds fused family operations, fixed-position WMMA
attention and bound-state graph integration. Its source archive SHA256 is
`98a96e1f3599efaff26cb8bd66cd078419d4a59875c95d432f7f9e9bae4b89de`;
image digest is
`sha256:3142fb98cd596437fb24df13281c181b07e5b13ef97362b5306951e4d743aa96`.
Installed CPU boundary checks passed 84 tests. The corresponding native family
suite passed 84 component tests; the bound-state module passed seven tests
with one TP8-not-visible skip on four GPUs. Strict comparisons cover actual
context versus padded graph buckets, main/MTP logits, residuals, accepted
recurrent/convolution/PLE/cache state, compact sampling packets, mixed parities,
live 1/2/3-row prefixes and growth across 8,192 tokens. These component results
do not qualify whole-model graph performance.

Candidate 5's 325-step full-model target scoring retained every top-1 choice
and measured perplexity ratio `0.9996347218`, but mean KL was
`0.0003770707511`: **the predeclared gate failed**. Both native-family selectors
and default packed-expert selection remain off. Operator-group isolation and
broader reasoning, code and longer-context tests are pending before promotion.

Nsight Compute 2025.2.1 collected actual checkpoint component counters on V100
with clock control disabled, kernel replay and cache flush. Dense HC/QKV/head
kernels execute Volta HMMA instructions. Expert gate/up kernels used about
17% achieved occupancy and 4.8–5.2% DRAM throughput, identifying insufficient
parallel work and instruction cost rather than saturated HBM. Component kernel
times and hardware counters are not serving throughput. No claim that every
hardware feature is optimally used follows from tensor-core instructions alone.

Nsight Systems 2025.3.2 initially selected its bundled CUPTI 13.1 and reported
`CUPTI_ERROR_INVALID_DEVICE`, producing no GPU trace. Selecting its own bundled
CUPTI 12.9 through a profiling-only search directory passed a subprocess smoke
with 20 CUDA kernel events. The full-model capture must use that selection;
Python emission timestamps from the failed initial capture remain separate
latency observations. Local raw packets and reports are retained under
`task-artifacts/v100-optimize/`; maintained tests require none of them.

The corrected full-model Systems capture completed on candidate 5. Six verify
and six MTP graphs occupied 877 MiB of a 3,983 MiB automatic graph allowance.
Five profiled rounds contained 3,950 GPU kernels each, roughly 38 ms of recorded
GPU activity and 10 ms without recorded activity. The trace exposed two bound
MTP defects: the singleton fast path headed the padded last row, and MTP buffers
inherited the target's FP16 expert-slot policy instead of their original FP32
policy. These were private graph-candidate bugs; candidate 3 remained eager.
Profiler start/stop introduced 0.5/18.4-second gaps, so its reported gross decode
time is excluded from performance comparisons.

Candidate 6 retains candidate 5 family arithmetic, fixes both MTP defects and
adds exact stable routing indices with the existing Torch softmax unchanged.
Its archive SHA256 is
`1a6cf21ff8f2798e931297270253f678836ccd2d418025177850a649d740cb83`;
image digest is
`sha256:8b535900f38f68ee3438e710052aa03f7e98ad0d706d0cc452c2cd197aba9baf`.
Eleven strict native graph/protocol tests passed with one TP8-not-visible skip;
54 installed CPU boundary checks passed. Six native router tests cover exact
indices/weights, ties, NaNs, signed zero, empty rows and dynamic graph replay.

On the same eight V100s, with TP8, INT8 H32 KV, MTP2/0.7 confidence, six slots,
237,598 configured context and 200 W limits, its unprofiled private API returned:

| Input tokens | Output tokens | Cached tokens | Decode tokens/s | Prefill seconds |
| --- | --- | --- | --- | --- |
| 128 | 128 | 0 | 45.35 | 2.8950 |
| 128 | 512 | 0 | 44.25 | 2.9772 |
| 128 | 512 | 0 | 44.59 | 2.9127 |
| 128 | 512 | 127 | 45.66 | 0.1312 |

Requests ran sequentially with other GPU work held, greedy sampling, thinking
disabled, actual EOS stopping and complete output IDs retained. Decode rate is
`(output_tokens - 1) / decode_seconds`; these are individual observations across
different prompts/cache states, without confidence intervals. The cached
512-token reply's first 128 IDs exactly matched its earlier cold 128-token reply.
MTP accepted 268–270 of 367–368 proposed tokens on the 512-token replies.
This remains a **private performance investigation**: candidate 6 retains the
candidate 5 arithmetic whose mean-KL gate failed, and 100+ tokens/s remains unmet.

Candidate 6 subsequently matched all 12 baseline greedy HTTP responses, including
complete token IDs. A broader nine-pair serial/drafted self-comparison passed
all three greedy pairs and two of six sampled pairs; four sampled pairs diverged
at output indices 12, 37, 38 and 69. Settings were temperature 0.7, top-k 40,
top-p 0.9 and fixed seeds, with cold serial versus cached drafted prefixes.
Reference Torch reduction trees also depend on row geometry, so the selected
four failures are being replayed against immutable candidate 3 to distinguish
inherited cold/cache drift from new behavior. This remains an unresolved
validation failure and does not change the predeclared numerical gates.


Candidate 7 incorporates exact Torch rounding for native normalization, rotary,
PLE and HC operations, width-one MTP graphs and collective status batching.
Its image is `sha256:3712519dca68f791f379e85e803d59ff566635cb416802fa841472c8fb1d3cb1`;
archive SHA256 is `62b9892648a2adabdbfe2bfd74ff5971481af829c4f370929de3d3a09ba3d103`.
All-native prompt attention still failed the fixed gate (mean KL `0.0002030325`).
Using the original Torch attention for prompts, while retaining native decode
attention, passed 325 steps: mean KL `8.817068396e-5`, perplexity ratio
`0.9990703056` and top-1 agreement 1.0. This paired probe uses the single-row
engine; scoring the actual padded graph path and broader teacher cases remains
pending.

Under the same quiet API conditions as candidate 6, candidate 7 measured
46.71 decode tokens/s for 128 input/128 output tokens, then 46.73, 45.01 and
47.67 for 128 input/512 output tokens. The last request reused 127 prompt tokens
and reached its first token in 0.125 seconds. Its first 128 output IDs matched
the earlier cold request. These individual observations do not establish a
confidence interval or qualify a later image.

Candidate 8 explicitly routes every prompt expert call, including a cached
one-token suffix, through the original prompt provider. Candidate 7 had instead
selected packed decode arithmetic for that suffix. A controlled candidate 7
replay reproduced the sampled code mismatch only with the cached prefix.
Candidate 8 fixes that case and retains candidate 7 decode arithmetic.
Its archive SHA256 is `a4ab365bfcd422e8148c87077070604361961dafc8818825db6bcf8a311e9718`;
image is `sha256:53a42559abde9bc82edd20fb9595cc7eb7f49777a756bbb0987e1c5e6864d04d`.
Installed CPU checks passed **1,960 tests**, with two skips and 94 passing
subtests. All 12 baseline greedy HTTP responses matched complete IDs/messages.
Eight of nine serial/drafted pairs matched complete IDs/messages: all six
sampled pairs and two greedy pairs. The remaining cold/cold greedy hash-table
case diverged at token index 102. This failure is preserved and under
investigation; no candidate 8 throughput or promotion result is claimed.

NCCL 2.29.3 transport tuning on the actual eight-rank communicator found the
default protocol faster than forced LL128 or a single channel for the tested
payloads. Those alternatives remain unselected. Exact packed-block expert
instruction reductions and same-arithmetic concatenation staging are component
qualified but still await whole-model integration. The 100+ target remains open.


Candidate 9 selects byte-exact block decoding for packed experts and native
concatenation staging, with the same separate part GEMMs. Dense execution,
prompt expert policy and attention are unchanged from candidate 8. Archive
SHA256 is `32df4b89ad31cbca1758f451de397e3d4a5074521efa5bffbd9b6d97b45869bf`;
image is `sha256:2934352b204d66c780308b8a79dbd3740485bc66a516b68e6c2afbee5640d689`.
All 18 serial/drafted comparison responses matched candidate 8's complete IDs
and messages, including its unresolved greedy divergence. All 12 baseline HTTP
cases matched. Selected installed CPU checks passed 255 tests in 3.90 seconds.

With other GPU work held and the same benchmark/configuration used above,
candidate 9 returned 49.12 decode tokens/s for 128 input/128 output tokens.
Three 128-input/512-output requests returned **54.06, 52.08 and 53.53 tokens/s**;
respective prefill times were 3.5698, 3.4685 and 0.1389 seconds. The last reused
127 tokens, reached its first token in 0.1418 seconds, and retained its earlier
cold request's first 128 output IDs. Candidate 8 had no quiet timing run, so
comparison with candidate 7 does not isolate the expert/concatenation changes.
These are individual private measurements, with the target and qualification
status still open.

A corrected-path Nsight capture on candidate 8 recorded 12 rounds averaging
45.67 ms, with 35.75 ms of GPU activity and 9.93 ms without recorded GPU activity,
and 24 emitted tokens. Roughly 3,848 kernels per round include many small GEMMs,
fills and copies. The graph's numeric-status wait includes GPU forward execution
and must not be labeled pure CPU overhead. This observer-affected trace guides
optimization; quiet HTTP requests above supply throughput measurements.


Candidate 11 incorporates decode-only serial PLE and router reduction geometry,
plus stable 32-bit radix keys with original IDs as values. CUB's stable ordering
preserves the original tie behavior while avoiding 32 redundant key bits.
Its archive SHA256 is `a27a497313111f04cdd9fac16dfbb5f7340e876f387b66b80df6b89738afc54b`;
image is `sha256:e1a78c0aca113afb52d8f86216fd00bf7931bde45dd800cb3eda69a6e015e5bf`.
Candidate 10 was an intermediate PLE-only build and had no live qualification.

Two full-model candidate 8 probes located the speculative discrepancy. Identical
prompt splitting and isolated graph/eager windows were exact, but accumulated
multirow verification changed a rounded router weight at layer 6 by one FP16
step, identically on all eight ranks. Normalized inputs, expert outputs and
selected IDs were still exact there; weighted partials then diverged, followed
by HC, attention and recurrent state. This was actual-row GEMM arithmetic,
rather than graph scheduling. Candidate 11 keeps each decoded row's original
serial router reduction; prompt routing remains unchanged.

The installed candidate 11 passed **1,960 CPU tests**, with two skips and 94
passing subtests, and **135 GPU checks** without a source override. These
include an independent router counterexample, every-row prep/GDN comparisons
and canonical PLE controls. All 12 baseline responses matched full IDs/messages,
and all nine serial/drafted comparisons passed, including the formerly failing
greedy hash-table case.

The quiet 128-input/512-output API observations were **52.53, 52.00 and 54.06
decode tokens/s**, with prefill 3.6850, 3.7314 and 0.1399 seconds. The last reused
127 tokens and reached its first token in 0.1430 seconds. A preceding cold
128-output request measured 44.95 tokens/s. Its first 128 IDs did **not** match
the cached 512-output continuation: divergence began at index 46. This separate
prompt-chunk/cache result remains unresolved, so no promotion or complete quality
qualification is claimed. Full graph-path state scoring and a controlled MTP
depth/confidence study remain pending; the 100+ target is unmet.


Candidate 12 combines exact fused routing, static FP32 storage of the same
FP16-rounded router coefficients, two-warp packed gate/up, decode normalization
that retains serial geometry at deeper windows, and batched TCPStore operations.
It also honors the existing GGUF MTP draft-vocabulary selection: the default
79,591-token proposal head is assembled from original global target-head rows;
The Python engine option `draft_vocab=0` retains the full head. There is no
public CLI `--draft-vocab` flag. The full target head is unchanged.
Admission includes the added proposal head, ID maps, and 126,074,880 additional
resident router bytes per rank. The optimized path is still selected by private
qualification policy; default family selectors remain off.

Archive SHA256 is `8ead2f3d2e5156017b6f65be4b7dd09568fa19679d8c2e4c71863007b17b9e8d`;
image is `sha256:49f15f2ba6851395d718848f1c6669ad6f64fe9a13d2937e5b48a801d4b9d5ea`.
CPU checks passed 2,029 tests, with two skips and 94 passing subtests. Its
installed runtime passed 228 family/routing GPU checks and 209 GGUF GPU checks.
The family cohort's first run exposed 12 capture-fixture allocations; the
corrected test preallocates its device inputs outside capture. The rerun kept
the same installed runtime and mounted only that corrected test, passing every
assertion. A future image must bake the corrected fixture. Two native four-rank
sampling/concurrent-scheduler tests passed, with one TP8-not-visible skip.

The private CLI used depth 3, confidence 0 and the full MTP vocabulary,
with TP8, six slots, INT8 H32 KV, configured context 237,598, original model and
200 W limits. All nine serial/drafted comparisons and 12 baseline responses
matched complete IDs/messages. Three quiet 128-input/512-output requests measured
**59.74, 61.54 and 64.19 decode tokens/s**, with prefill 3.6629, 3.6735 and
0.1301 seconds. The last reused 127 prompt tokens and reached its first token
in 0.1329 seconds. A preceding cold 128-output request measured 57.36 tokens/s.
The same cold/cached prefix identity limitation remains; no full quality or
100+ result is claimed.

Quiet captured component checks compared identical outputs: fused routing took
about 7.3 microseconds versus 29.1–31.2 for native indices plus Torch weights.
At three rows, storing already-rounded router coefficients avoided their
repeated conversion and reduced the canonical router from 52.0 to 32.2
microseconds. These are GPU component measurements, separate from API rates.

The preceding candidate 11 depth/confidence study completed all 19 screens and
five three-trial 512-token confirmations with complete reference IDs and no
measured graph misses. On its one cached 128-token performance prompt, median
rates were 33.70 without MTP, 55.00 for depth 2/confidence 0.7, 57.60 for depth
2/confidence 0, and 58.46 for depth 3/confidence 0. Depths 4–7 were slower in the
128-token screen. This supports selecting by time per accepted token instead of
assuming deeper speculation is faster. Candidate 12 has not repeated that matrix.

Candidate 12's completed Nsight capture retained twelve rounds, averaging
43.407 ms per round, 32.676 ms with recorded GPU activity and 10.731 ms without
recorded activity. Kernel launches averaged 2,827.5 per round, versus roughly
3,848 in the earlier candidate 8 trace. Different accepted-token counts and
profiler overhead prevent treating this as an isolated throughput comparison.
GPU activity includes NCCL waiting, and the nested KV check range waits for GPU
work; neither is a direct measurement of useful arithmetic or pure CPU overhead.

Candidate 13 archive is
`a6757c5ead47e690122c0dbc1d38ef216538f23bdee3cf4760b88bcfbe1519c2`;
image is `sha256:dc8b5d3c0ae49cd93d0aafc1702d72e9a8b5370f02ef15a05df0bdc858574f50`.
It canonicalizes SM70 candidate-packet reductions to serial row geometry and
avoids a redundant KV stream wait after TP has already validated all statuses.
Eight-rank forced-history checks passed for deep physical eight-row plans,
partial acceptance, both state parities, complete logits and all 65 candidate
words; the older depth-two history regression also passed. Against the original
reference, its 325 teacher-forced steps passed the unchanged limits: mean KL
0.0000881706839583, perplexity ratio 0.9990703056 and top-1 agreement 1.0.
A separate packed-expert prompt experiment failed the mean-KL limit
(0.000257976) and is excluded. Cold versus split-prompt logits still differ
(KL 0.000001775); restoring a copy of the split state is exact. The prompt
geometry limitation therefore remains open, rather than being attributed to
state ownership or silently waived.

Candidate 14 archive is
`ee6fa846849827a33610139c5de596e360a93f1f9211cf13cec3a3b211083b6b`;
image is `sha256:8a3fa8e09d58e2d788b2b72598b58d28c607c79422b39d5427151127c1662728`.
It adds explicitly owned 32-row matrix storage that avoids repeated padding
and eligible copies, and a single-stream greedy MTP provider that stages draft
handoffs on device with one final status readback per chain. Other sampling and
unsupported shapes retain existing paths. The private policy enables both;
ordinary provider defaults remain unchanged. Its installed image passed 2,090
CPU tests (two skips, 94 subtests), 332 GPU checks, and a four-rank combined
state/proposal/tie/error check. Eight-rank deep cases passed all logits, streams,
PLE, committed state and 65 candidate words. The older 120 decode-history rows
also passed on all ranks; the two prompt-geometry diagnostics still differ.

All nine **sequential** serial/drafted 128-token pairs passed complete IDs and
messages, as did all 12 baseline HTTP responses. An initial private stats observer
looked up the TP wrapper instead of its underlying decoder and returned null;
it was corrected and regression-tested before throughput measurements. The
same installed runtime restarted with that CPU-only observer. A preceding small
priming request initialized counter readings; it is excluded from timing.

With the original eight V100s, 200 W limits, checkpoint, INT8 H32 cache, six
slots, context 237,598, depth 3/confidence 0 and full MTP head, the
three quiet 128-input/512-output requests measured **62.15, 64.02 and 69.27 decode
tokens/s**. Cold prefill took 4.6394 and 4.6176 seconds, slower than candidate
12's approximately 3.67 seconds. The last request reused 127 prompt tokens,
taking 0.1607 seconds prefill and 0.1635 seconds to first token. A preceding
cold 128-output request measured 54.21 decode tokens/s. All four full token-ID
sequences matched candidate 12's corresponding requests.

Each measured request used GPU handoffs, with one readback per completed chain
and no recorded error. The 512-output requests completed 197/191/189 chains and
589/573/567 steps, crossing the 256/512/1,024 context buckets with both prepared
plans present. Decode is `(completion_tokens - 1) / decode_s`; raw API requests,
responses, stats and summary are retained in the private `candidate14-live`
artifacts. These are three workload/cache observations without a confidence
interval. Cold/cached prefix identity still fails; 100+ tokens/s and promotion
remain unachieved. A new Nsight profile is collecting the handoff ranges.

That Nsight capture completed with the same 35 emitted tokens across twelve
profiled rounds as candidate 12: average round time 39.245 ms, recorded GPU
activity union 31.898 ms, no recorded GPU activity 7.347 ms and 2,563.5 kernel
launches per round. The draft range averaged 6.753 ms per round versus 9.348 ms
previously. Its gather calls averaged 0.535 ms each. These nested ranges include
GPU waits and profiler overhead. The largest dense kernel is the qualified
Torch/cuBLAS CUTLASS WMMA kernel, not the optional custom WMMA provider. Hardware
samples reached 1,530 MHz under work without active clock-event flags; power
limits and clock configuration were unchanged.

A subsequent control ran the original installed candidate 3 on the same 128
explicit prompt IDs, comparing cold 128-row prefill with 127+1. Only synthetic
startup warm was skipped; state was reset before each scored prompt. It used
TP8, INT8 KV, one slot and an 8,192-token allocation, compared with candidate
14's six-slot full allocation. The original baseline reproduced the variation:
KL 0.00000177524721358, maximum logit delta 0.02094268799, both top-1 IDs 32.
More decisively, **all 248,320 logits matched exactly** between original and
candidate for the cold geometry and, separately, the split geometry. This
qualifies that known difference as inherited prompt arithmetic rather than a
new optimization regression. It does not prove universal cache identity or
broader long-context quality; historical study verdicts remain unchanged.

Candidate 15 combines an exact batched implementation of the pinned serial
FP32 router reduction (decoded rows 2–32 only) with two native draft-packet
kernels around the existing NCCL gather. It retains the same full-shard maximum,
minimum global ID for ties, original log-sum-exp/status words, one readback per
chain, state owners and fallback policies. Both new selections default off.
Its archive is `ce7a34224708403101bde9822c7493e5b611c3257220bff329abd808f453aefc`;
image is `sha256:ac866119d0c58808effa626b3b2f7d8120e785466b6bcc42b002b645acff8a4a`.
The installed wheel passed 1,942 selected CPU tests (two skips, 11 subtests)
plus 176 separate common CUDA/build/communication/precision CPU controls,
a different selection from candidate 14's broad cohort, and 201 GPU tests.
Separate source-overlay proofs passed 119 router/state/routing controls and
82 gather controls; a four-rank comparison preserved every historical packet
word, token and committed state, including ties and a remote numeric error.
Independent review added 110 GPU IEEE edge cases against the original Torch
oracle without a blocking finding. Its full eight-rank deep state/logit/packet
and older 120-row history checks subsequently passed on every rank; the two
inherited prompt-layout diagnostics remain unchanged. Every rank recorded
installed source and loaded native-module identities. A separate installed
four-rank protocol check passed in 15.26 seconds with no source overlay.

Candidate 15's private API passed all nine sequential serial/drafted pairs and
all 12 original baseline responses, comparing complete IDs/messages. Three
quiet 128-input/512-output requests measured **69.46, 71.23 and 74.15 decode
tokens/s**, under the same TP8/six-slot/237,598-context/INT8/depth3/confidence0/
full-draft-head configuration and original 200 W limits. Cold prefill took
3.5562 and 3.5587 seconds; the final request reused 127 prompt tokens, with
0.1308 seconds prefill and 0.1337 seconds to first token. A preceding cold
128-output request measured 66.16 decode tokens/s. The prior candidate 14
prefill slowdown was not observed here; its cause has not been established.
All four complete benchmark ID sequences and messages matched candidate 12.

Per-request telemetry verified the native gather, positive bound-graph replay
counts, qualified router captures, GPU handoff chains, one readback per chain,
no recorded errors and all three 256/512/1,024 buckets on the 512-output cases.
Those cases completed 198/191/189 handoff chains and 592/573/567 draft steps;
accepted/verified drafts were 310/601, 315/588 and 319/576. Raw requests,
responses, observations and the complete-ID comparison are retained in
`candidate15-live`. These remain three workload/cache observations without a
confidence interval. The 100+ target, broader quality gates and ordinary
provider/service promotion remain pending.

**Draft-head label correction:** a subsequent audit found that the family CUDA
serving factory forced `draft_vocab=0` for GGUF with a separate drafter. Thus
the candidate 12/14/15 HTTP rates above used the full 248,320-token draft head,
although earlier records labeled the default pruned head. Their timings,
complete response identities and observed native handoff use remain valid.
The direct-engine numerical/state studies and Nsight probes explicitly select
their own vocabulary and are separate evidence; the matched candidate 12/14
Nsight comparison used the 79,591-token pruned head. HTTP and profile head sizes
must not be conflated. A source fix now preserves the engine default and
explicit Python selections; 66 serving-factory/vocabulary/capability CPU controls
passed. The installed candidate 16 qualification below supersedes the image
portion of that pending work; live pruned-head qualification remains pending.

### Candidate 16 and closeout inventory

Candidate 16 is image
`sha256:1cd47a606770a74ab5ed6b4eab09d1f5e8f9e22ef172289a6480f041e5b3634d`,
source archive
`aaec2566d8097052d39851d780c8d6d82bd540291c1d4b23a188300cf52627b2`.
It adds admitted adaptive reference-math expert prefill, prioritizes the existing
greedy workspace before optional graph capture, and preserves the engine's
default draft-vocabulary selection through the serving factory. Private
qualification enables a 1 GiB prefill-workspace budget. The installed wheel
passed 2,151 CPU checks (two skips, 11 subtests) and 235 GPU checks. The original
325-step gate passed: mean KL 0.0000881706839583, perplexity ratio 0.999070305644,
top-1 agreement 1.0. These are the unchanged gates and aggregate metrics, not a
broader quality certification.

The eight-rank deep-state/proposal gate and older 120-row forced-history gate
passed exactly. Two separately retained cold/split prompt diagnostics remain
nonexact to each other; their full logit vectors each match candidate 15 exactly.
All ranks verified 48 installed source identities. Actual draft shards were
9,949 rows on seven ranks and 9,948 on the last, mapped to 79,591 tokens;
target-head shards had 31,040 rows. Admitted and loaded prefill metadata agreed
on 128 pairs and 657,937,024 workspace bytes. These direct-engine checks do not
establish serving-factory API behavior or throughput.

Subsequent private API checks passed all nine sequential serial/drafted pairs
and all 12 original baseline responses. A quiet four-request screen observed
the mapped local 9,949-row draft head through the serving factory, native GPU
handoff/gather, graph replays and 256/512/1,024 context buckets. Its three
512-token replies measured 69.39, 71.25 and 73.99 decode tokens/s; all four
complete benchmark ID sequences/messages match candidate 15. Cold 128-token
prefill took 2.1371 and 2.1227 seconds, compared with candidate 15's earlier
3.5562 and 3.5587 seconds. This small unpaired screen has no confidence
interval and establishes no decode gain. The final cached127 request took
0.1305 seconds prefill and 0.1333 seconds to first token. Broader quality and
ordinary provider/service promotion remain pending.

The subsequently completed additional teacher suite covers nine fixed reasoning,
code, translation and retrieval cases (389 steps), including three approximately
6,436-token prompts. It **failed** the unchanged gates: mean KL
0.0003104520713505378 exceeds 0.0001, and top-1 agreement 0.9897172236503856
falls below 0.995 (four differences). Perplexity ratio 1.0000343681019446 passes
1.001. The original 325-step pass remains a narrower result; the combined
candidate is not promoted. First-case prompt logits match the reference exactly;
decode operator isolation is pending. These are distribution comparisons against
the same target weights/history, not a separate human reasoning evaluation.

The separate expanded CPU-affinity comparison completed 12 balanced randomized
paired blocks (24 cached 512-token replies), preserving every ID/message.
Its paired geometric mean change was -0.1153%, paired-bootstrap 95% interval
[-0.3409%, +0.0480%], exact randomization p=0.3486. Original masks were restored;
there is no verified gain to select. The raw summary's inherited `no CI` scope
suffix is stale; its explicit paired-bootstrap fields contain the interval.

Observed two-replica TP4 runs measured single-request medians of 32.18 and
34.01 decode tokens/s and median concurrent aggregate end-to-end throughput of
62.84 tokens/s. These runs lacked the native greedy workspace and fell back
outside retained graph keys; they are not equivalent optimized-path comparisons
or evidence of a 100-token/s single request. A matched TP8/context4,096/two-slot
study encountered a separate mixed-prompt/cache-history response discrepancy.
That joint-prompt case has not been classified against original candidate 3;
the older cold/split control does not explain it. Preserve it as an open
correctness issue before promotion.

The bounded closeout work is: resolve that correctness issue; finish candidate
16 API, broader-quality and quiet-performance checks; qualify or decline the
private 32-row expert-decode and graph-lane memory proposals; finish the
prepared DSpark component/collective proofs and decide separately on its full
engine integration; qualify or explicitly defer the private hybrid expert/TP
prototype; then capture the selected provider profile, integrate accepted
changes into normal provider selection, document and validate the final service.
No new optimization search is needed to make that checkpoint reviewable.


## Experimental DSpark target graphs (C18–C19)

Both runs used the actual pinned DSpark checkpoint
`shadowpa0327/Qwen3.8-Flash-Next-DSpark` at revision
`3f74bdfa4a1b9deaaa56cc7c1377440ec5095fad`, checkpoint SHA256
`fb41acdeaeaa6e434099edc2cb04ad40e0e052ec168b6509cfd858d186f2c67a`.
TP8, six scheduler slots, int8 KV, context237598, depth3, confidence0, full
248320-token target head, single client and 200W limits were unchanged. The
resident drafter used334864388bytes/rank. Stock MTP was absent in these runs.

| Candidate | Installed image SHA256 | 512-token decode tokens/s |
| --- | --- | --- |
| C18 eager DSpark | `5359ddcf8b591685d9ba9138ff5bb9ba6d2998b799b458990b29c5f892339760` | 26.8594 / 28.0967 / 27.2260 |
| C19 target graphs, eager proposal | `0fbf4a0190c051acf648d3da5dd5f9ae79ae3b29f4f8961c00b7ec7bfe628f3c` | 43.1573 / 45.1948 / 44.2696 |

All9 serial/draft and12 baseline live cases passed on both images. All four
benchmark response choices (including complete token IDs/messages) matched C16
and each other. Accepted/drafted counts remained294/649,303/624,296/645 on the
three512-token requests. Actual C19 bound graph hits were215,206,214; MTP graph
keys were0. DSpark proposal/commit counters showed0errors. These few timing
requests demonstrate a large graph-path gain, not a confidence interval for small
gains. C19 cold128-token prefill took1.9364–2.0045s; the cached127-token case
took.1334s with.1366s TTFT. Installed C19 checks passed1943CPU tests (2skips,
7subtests) and14GPU tests. Startup capture used an exclusive allocator window.

The C18 Nsight12-round window emitted32tokens:122.7935ms wall,51.1712ms GPU
union and71.6223ms without GPU work. CPU ranges were nested and included
waiting: DSpark draft355.665ms, hidden232.845ms, context91.751ms across12rounds.
These profile figures include observer overhead and are separate from quiet API
measurements. C19 Nsight also completed the same12-round/32-emission window: mean76.2837ms
wall,53.1744ms GPU union and23.1093ms without GPU work, with actual target replay
hits. Context CPU ranges include backpressure from target graphs; their duration
is not wholly removable compute. Proposal-owner/full-checkpoint graph validation
remains pending.

Neither candidate includes the expert or native-attention arithmetic repair.
The broader389-step C16 numerical failure therefore remains a promotion blocker.
A separate repaired-expert81-step full-model case matched the original reference
exactly (KL/NLL0, perplexity ratio1, all top1), using original attention and
shared Dense math. Full325/389-step repaired-backend gates are still pending.
DSpark will remain supported as a selectable method regardless of which drafter
wins this host's eventual qualified performance comparison.


### C20 captured DSpark proposals

C20 image`b7cac25e49e10d7db059d9547744ab5a52453d63a7493c172b8b7d6165d5f3e1`
keeps the C19 target arithmetic and adds full-block greedy/confidence0 proposal
graphs in the same admitted RoundGraphs pool. Each proposal performs one final
status/token readback. Stochastic requests, partial final blocks, confidence
stopping and unavailable keys retain the eager provider. Exact installed source
comparison covered602runtime files. Final CPU checks passed2094tests (2skips,
14subtests); unchanged runtime bytes passed19GPU tests and171CLI tests
(80subtests). Earlier assembly/import/fixture failures are retained in evidence.

The same live9+12gates passed. Three quiet512-token requests measured
**58.1184 / 60.7408 / 59.9322decode tokens/s**, versusC19's43.1573/45.1948/44.2696.
All four benchmark complete token IDs and response choices matchedC16/C18/C19.
Accepted/drafted counts stayed294/649,303/624,296/645. Actual proposal graph
hits/calls/readbacks were216/208/215, with zero errors; target hits215/206/214.
Cold prefill took2.0034/2.0078s and cached prefill.1324s (TTFT.1356s).
This remains experimental: C16's broader389-step numerical failure is unresolved
in this image. The native expert/attention repair and normal optimized-provider
selection are separate pending qualification.

The maintained code now retains DSpark, explicit launch method selection and
shared State/graph ownership. Default kernel flags and the optional proposal
graph flag are still unpromoted. Compose accepts`TF_DRAFT_METHOD`,
`DRAFTER_DIRECTORY`, `TF_DRAFTS` and`TF_DRAFT_CONFIDENCE`; the legacy
`MTP_DIRECTORY` path remains a fallback.


### Bounded C17 TP4 graph-memory closeout

The registered C16/C17 comparison used TP4 on both NVLink cliques, context4096,
two slots, int8 KV, depth3/confidence0, default pruned draft head, socket CPU
affinity and unchanged200W/reserves. Both primary admitted1GiB-prefill arms
refused startup capture: retained216006656bytes exceeded193921024bytes of
graph allowance. The registered original-prefill fallback reduced the startup
estimate from26.04 to25.26GiB, but its C16 baseline hit the same graph-budget
refusal. The fallback C17 peer was cancelled normally; no failure is claimed
for that cancelled peer. No HTTP requests, throughput samples or crossover
were possible under these settings. This experiment is closed unqualified,
with no gain selected and no reserve changes. The separate bounded component
proof of C17's lane-admission behavior remains valid.

## RotorQuant Q4 on V100

The SM70 implementation admits the existing `rotorquant-signed128-4` (CODEC4)
and `rotorquant-signed64-norm4` (CODEC5) descriptors, independently on K/V.
Raw FP16/INT8 keep their existing working policy. Q4 uses uint8 packed nibbles,
one FP32 scale per128components, staged signed-basis rotation, FP16 reconstructed
Q/K/V operands, FP32 accumulation and FP32 inverse value rotation before final
Half output. These are existing pinned formats, not an invented INT4 wire codec.
Other RotorQuant/INT4 formats remain unadmitted on this provider.

### Component and same-format evidence

Canonical installed R3 image`e82b37d8bd17d81904cdf0ac5945b40ead5f5d3e990248c865fb41c4e3e11d85`
matched614runtime source hashes and passed2983CPU cases (2skips,41subtests).
Native cohorts passed59codec/copy,177fused-prep/legacy,653attention/legacy/capacity,
and8full-forward fallback GPU cases. All16ordered raw/INT8/CODEC4/CODEC5 pairs
have component coverage. Encoder comparisons use existing analytic index/scale
rounding envelopes; rotation, decode and stored-basis attention use exact output
oracles. No tolerance was widened. Original prep kernel/launcher/binding stayed
byte-identical; new malformed scalar conversions are range-checked before
narrowing. No persistent attention workspace was added.

An additional forced-hybrid32-key INT8 MTP control failed one of12288Half
intermediate values by3.0517578125e-5. The SAMEindex/value/failure reproduced on
old shipping image1443fc46 with only the canonical test mounted read-only, so it
predates Q4. It forcesNativeFamily=True/NativeAttention=False, outside both
service profiles and the real-query-end65 native qualification. Reproducer,
strict assertions and full failure logs remain; no whole-legacy-suite pass is
claimed. Applicable new Q4 actual-model/profile gates are reported separately.

Native attention retains localH3/KV1/D256, at most24rows per State, and real
query ends65–8192. Other frames retain their selected codec through the original
per-query Torch path and bounded working decode. Component/native cache capacity
coverage includes a physical262144-slot owner. Full configured262K windows below
are allocation evidence, not a complete262K prompt/quality workload.

### Fixed target-distribution comparison

Both variants completed all21cases/714teacher steps and723prefill/decodeframes
with the ordinary target-only optimized engine, context8192, fixed2048prompt
pieces, depth0/noMTP/graphsFalse, and literal frozen input/teacher IDs. The
reference is the original INT8 C21 authority manifest
`9cf873c58c295c2649e9fb89a9c15a79977ec1689c41528fc52276bd3c9469d1`.
All complete Float32 vocabulary vectors and checksums were verified; no prompts
were retokenized. These cross-format measurements are distinct from native-Q4
implementation correctness.

| Format / suite | Mean KL | Perplexity ratio vsINT8 | Top-1 agreement |
| --- | --- | --- | --- |
| signed128 Q4 /325 | 0.00303164314 | 0.99605870067 | 0.99692307692 |
| signed128 Q4 /389 | 0.00533592937 | 1.00652224383 | 0.97429305913 |
| signed128 Q4 /714 | 0.00428705959 | 1.00174586609 | 0.98459383754 |
| signed64-norm Q4 /325 | 0.00229689506 | 1.00064709104 | 0.99384615385 |
| signed64-norm Q4 /389 | 0.00466476362 | 1.00804118734 | 0.97686375321 |
| signed64-norm Q4 /714 | 0.00358695230 | 1.00466877664 | 0.98459383754 |

The unchanged historical thresholds KL≤0.0001/PPL≤1.001/top1≥0.995 failed for
both formats. Quantization is an explicitly lossy configuration choice; these
results do not establish INT8/FP16 equivalence or justify silently changing the
user's earlier quality requirement. CODEC4 quality image384083f2... and CODEC5
imagee82b37d8... each observed8676teacher fusedQ4preps and5376nativeQ4attention
calls; CODEC5 setup/last fields were explicitly5/5. Full raw per-step/per-case/
rank/source/foundation/native-binary records remain in
`task-artifacts/v100-rotorquant-q4/quality-q4-r2/` and `quality-norm-q4-r3/`.

### Prompt reuse qualification and live checks

A real signed128 Q4 cold serial/cold MTP pair matched all128IDs/messages.
Reusing the same127-token prompt suffix first changed output at61. Four short
filler prompts restored cold state and exact identity. This isolates cached
preparation, not a specific floating-point operator. The invalid warm result is
preserved and excluded as an admissible performance reference.

R4 image`f1c1c3e1c881e29edf25d4ec2a101fc00db5994dc0b9783d7bc078219621701f`
therefore qualifies Q4 for cold prompt preparation in serial/twin and concurrent
paths. Drafting and request-local KV stay active. Raw/INT8 reuse policies stay
unchanged; malformed-sampling precedence is preserved. Registered immutable
pairs and a versioned policy enter worker agreement before payload loading.
Both profiles are covered by old-peer-refusal and actual-generate CPU controls.
Prefix snapshots/cuts remain because their complete in-request role was not
proved removable. The cost is cold TTFT on repeated Q4 prompts.

Both Q4 variants passed28actual live checks:12original-budget plus9seeded
serial/MTP pairs,4concurrent complete128-token replies, fullSSE, cancellation,
and recovery; all IDs/messages matched fresh SAME-format references. Every
completed drafted reply reportedcached0 under explicit `--prefix-reuse cold`.
Public peak in-flight count was4, withall8rank materialization receipts. Raw
HTTP exchanges, usage, counters, source/tool hashes and summaries are in
`live-functional-c4-r4/` and `live-functional-c5-r4/`. Old INT8 reply IDs were not
substituted as same-Q4 correctness oracles.

R4 selectedCPU initially reported3044PASS/onefixturefailure: a hand-constructed
oldprefix-fork decoder lacked the newly owned immutable pair. The fixture was
corrected from its actual State;101focused cases then passed with no assertion
relaxation. Final canonical R5 installed-image checks passed3046CPU cases
(2skips,41subtests,43.97s) with614runtime source hashes matching, plus295
source-relative tool/deployment/permission/probe cases (4.07s). All affected
Python files and the named pre-push Ruff checks passed.
New canonical source-relative quality/live tools have separate CPU/fault tests;
they reuse the existing protocol/output helpers. A C6tool-only retirement fix
preserves primary+cleanup failures and prior native cause/context, verified by
46CPU adversarial tests and a same-target structural proof that the normal
model body/constants/gates stayed unchanged. It does not require reinterpreting
or rerunning the earlier arithmetic results.

### Capacity and quiet configuration screens

All8ranks physically allocated four262148-slot caches (262144context plus
speculation),8069964032ownedcache bytes/rank, preserving4GiBdevice floor and
2GiBworkspace. Automatic graph allowance1432787312bytes remained protected.
Header-only canonical plans also predict DSpark three262140-token Q4 windows
with551069852bytesgraphroom; actual DSpark compatibility checks are separate.

Same pinned target/host/TP8/optimized profile, MTPdepth3/confidence0,128prompt
and512greedy output tokens, thinking disabled, no profiler/otherGPUjob:

| Configuration | Full windows | Three decode tokens/s | Repeated prompt |
| --- | --- | --- | --- |
| INT8 C22 shipping reference | 3×262144 | 54.0358 /54.4202 /59.7780 | cached127 |
| signed128 Q4 R4 | 4×262144 | 51.2476 /50.2409 /50.2750 | cold0 |
| signed64-norm Q4 R4 | 4×262144 | 51.5636 /52.5942 /50.4676 | cold0 |

Signed128 Q4 cold128-token prefill was1.6512/1.6418/1.6255s (77.52/77.96/78.75
prefill tokens/s); repeated-prompt TTFT1.6291s. The INT8 reference's prefix-hit
TTFT was0.1393s. Format/output/cache state/concurrency differ, so these are
configuration tradeoffs, not isolated kernel gains, equivalence proofs or a
global optimum. No100single-request decode tokens/s result is claimed.

The user was asked which format should remain the production default after
these concrete quality/latency tradeoffs. No answer or elapsed-time approval is
assumed; the previous INT8 configuration remains the known admissible fallback.

### DSpark compatibility closeout

The Q4/DSpark3-window262140 startup estimate fit, but the longer live corpus
exhausted its551069852-byte shared graph allowance. All ranks retired under
the declared poisoned-cohort policy. The startup and failed live request/logs
are preserved; this is not a qualified three-full-maximum DSpark service.

One registered fallback used the canonical header calculation for1GiBgraphs:
three equal245713-token windows, unchanged4GiBfloor/2GiBworkspace. All8ranks
materialized the caches (7123834176bytes/rank), with1073769276graphbytes.
All27same-format checks passed:21serial/DSpark pairs,3concurrent replies,
SSE,cancellation and recovery, withcached0. This establishes actualDSpark
compatibility for that configuration. MTP's four262144windows remain separate.
No further layout/depth/memory experiment was opened.
