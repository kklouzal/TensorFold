# Optimization validation

Each optimization has a declared workload, platform, precision and sampling
mode. A result applies to that region while its checkpoint, source, compiler,
runtime, configuration and relevant cache state match. Rebuild or revalidate
when those inputs change.

Keep the README's measured-results summary and this validation document current
in the same change whenever a retained optimization, configuration, supported
region, or qualification result changes. Identify the tested checkpoint,
hardware, runtime, workload and uncertainty alongside each result. Historical
generation evidence keeps its original build identity; it does not qualify a
later release automatically. Task artifacts referenced during development are
local evidence unless published with an accessible validation receipt.

Preserve functionality, authorization, data integrity, memory ownership and
the numerical contract before comparing speed. Quantization options have their
own explicit quality tradeoffs; changing an optimization must not silently
change the user's selected precision or sampling policy.

## Python-line integration, 2026-10-11

The integration joins fork `2f8089a7b59588875d639a7efa860117168a29d0` with
upstream Python `ed78d6fc204d89d90b045bf033d6551e7714f3a1`. This records Git
lineage, not a new performance result. The selected fork cache geometry, sampler,
attention ordering, eager shared rounds, CUDA support floor and relevant memory
policies are retained. New source corrections include defined EXL3 no-bias loads,
bounded HTTP framing, authentication file/counter ownership, and whole-request
decision admission. Applicable gates must run on the combined source.

Reviewed source `dfb4669d6b98f26de9ecfd75d3ccf7e90ec747bf` also aligns the
Harness with installed TensorFold 0.6.6, binds its two HTTP factory exports, and
validates label IDs before CUDA indexing. The ordinary probability arithmetic
and kernels are retained. Native test corrections preserve collective startup
and the fork's single shared EXL3 reader, with independent failure retirement.

The following locally built images use that exact committed archive:

| Native target | Verification image | Serving image |
| --- | --- | --- |
| ARM64 / GB10 | `sha256:605b5b0282d39bbb1608f5dea48312d89dd0cc82f0ffb6e52e82fde1b4baf94d` | `sha256:fa86511d30356f5386b49c83c198403f924fd7a978a2a36edf67176a4837359d` |
| AMD64 / RTX PRO 2000 Blackwell | `sha256:c8894f478938ef08ab105692b6d3b0d5f85a6bb55a9ca8961ad7e60f77bcb044` | `sha256:56a7fc80b0851b37847b8ed3703d5dbe0d092b78679aed9919587c31f1c3dae4` |

Each fresh installed CPU suite passed **7,320 checks**, with 177 explicit skips,
zero failures/errors and two inherited pinned-Torch allocator warnings. Each
run used CPython 3.12, the architecture's hash-locked dependencies, UID 1000,
no GPU/network, and 3 GiB with no extra swap. The 490 wheel files, 484 required
package sources, 1,133 qualified source/baked-script records and 261 loaded
module origins passed before/after checks. Imports resolved to the normal
installation. Each serving image passed 13 CLI/Harness checks, including its
actual baked launcher; the ordinary OCI entrypoint with its GPU also passed
`serve --help`. This is startup qualification, not loaded-model generation.
The no-GPU collection policy omits CUDA modules from both pass and skip counts.

Follow-up source `2edfaa515d85ca48a57365552f38a97a17123c3c` changes four
documents and two family-error diagnostic URL constants. All other family AST
nodes, runtime sources, build/pin/launcher inputs and maintained tests match
`dfb4669d…`. Its own normal installed images passed 36 affected maintained
checks per host, 13 standard CLI/Harness checks and one owned-link diagnostic,
without failures or skips. Both ordinary GPU-bearing OCI launchers passed
`serve --help`. The full CPU suite was not rerun at this revision; the recorded
structural proof bounds reuse of the earlier full-suite/native evidence.

| Native target | `2edfaa51…` verification image | `2edfaa51…` serving image |
| --- | --- | --- |
| ARM64 / GB10 | `sha256:3bf2b74adf623afa61fd496aa63157fc5a0c9b2c9bec03ecb091610b04c576cd` | `sha256:8e1fe707502d99e8565af2900bd862847141a9da30779b208c239cbb21ee44be` |
| AMD64 / RTX PRO 2000 Blackwell | `sha256:fb50d549a8758cf109e2b0fbc5ab48649428325d7900b7f28feb55772b67f835` | `sha256:aaa93b1ee89b0faff53d4d6c99dea18399bf08b3f8ddeaa44d572faddf9721a7` |

The owned Python branch and `archive/main-before-python-20261011` are published;
the archive preserves `cb2ebf0540f42604e2759b2ddef497861e928248`. GitHub still
uses `main` as default because the settings API returned HTTP 403. Removing the
old branch remains pending that settings change and guarded ref verification.

Each architecture separately passed the following native gates in its
`dfb4669d…` verification image. Source `2edfaa51…` reuses this evidence only
for the unchanged native/runtime contracts within their recorded region:

| Native module | Passed | Scope |
| --- | --- | --- |
| Tensor parallel | 22 | Two threads on one actual GPU; collective initialization, tokens and retirement. |
| Probability capture | 38 | Existing probability controls and independent label/LSE, range and callback checks. |
| Public label scoring | 2 | Synthetic model, retained prefix and following generation. |
| Explicit memory limit | 2 | Derived real allocation bound, refusal below the bound, positive boundary and recovery. |
| FP8 n-gram table | 7 | Admission, exact BF16 staging and full logits/serial/MTP tokens against converted BF16 weights. |
| RAM EXL3 engine | 12 | Serialized synthetic checkpoint, cache behavior and two independent failed-load retirements. |

The exhaustive composed KV gate at `dfb4669d…` passed **all 256 ordered pairs
on each host**, with exact coverage and completed-run retirement reinspection. The maintained test compares each
pair's sparse multistream execution with its same-pair solo contract, including
YaRN and vision inputs. It retains all 16 formats, assertions, seeds and quality
settings. The GB10 run uses 64 processes with four cases each. Both hosts retain
UID 1000, 4 GiB/no extra swap, two CPUs and one compiler job. Original combined
processes were killed by the 4 GiB cgroup OOM limit and remain failed evidence.
The AMD64 four-case run completed 184 cases before a later process exceeded its
900-second diagnostic client bound. Independent Docker logs, wait and daemon
events proved all four cases passed (pytest reported 1,040.28 seconds), followed
by actual exit 0 and retirement; its original timed-out producer remains failed.
The reviewed recovery reused only the 184 completed producer successes and
passed the remaining 72 cases in 36 two-case processes with the same 900-second
bound. Exact mixed coverage, producer receipts and actual retirement passed
completed-only reinspection, without rerunning GPU work. Each case reconstructs
its model with a local seed-72 generator, vision with a separate seed-41
generator and stateless seed-59 sampling keys; grouping preserves its inputs.
Partitioning changes process lifetime and cache/resource state, not the cases
or their numerical contracts. This synthetic
comparison does not establish BF16 equivalence, pretrained-model quality,
filled-context capacity or performance.

### Isolated GDN candidates on GB10

The shared-GDN representation candidate `a2fe672cec14653eee1c55aca2808accbf44da7e`
and Flash history/read-ahead candidate `2905dfde32b834f03a253443115348a92bc5dbcd`
each use `2edfaa51…` as parent and change one CUDA source file. They remain on
separate evaluation branches and are not selected in the maintained runtime.
Their normal ARM64 package/launcher audits and ordinary GPU OCI startup passed;
each passed **56 fresh synthetic native checks**, with 46 unrelated cases
deselected. The verification images are respectively
`sha256:659d40ab9dc9b1263be9c0c880cb24ea7c04dd25151e63f00cf95d17fee29c6e`
and `sha256:0ea65811f9c8fafd427d7082d1ee5ef29e45530afefe58ec1023bf49a8c4444c`.
The corresponding serving images are
`sha256:6e265f880754377dcd38971153d1514b1b734cff9412928122d2639f55594cf6`
and `sha256:281578a6539bd943dd0b8066cf8fa5bd0151df7565f76542b917a2e998bacd90`.

At the actual automatic SM 121 target, pinned CUDA 13.4 compiler and supported
`-O3 --fmad=false` flags, all six compiled providers match their earlier
candidate or baseline references in 58 kernels' GPU instruction words, resources
and recognized argument layouts.
The Flash candidate's monolithic family provider is byte-identical. The other
five providers differ only
in two or three nonallocated temporary-filename metadata bytes; executable and
loaded bytes match. The CUDA 13.1 inspection tool leaves newer metadata fields
undecoded, although the opaque GPU ELF bytes also match. An older IO cache lacks
an original recorded binary hash; its current sealed bytes are compared without
retroactively claiming that identity for earlier runs. Fresh native checks
independently qualify the current IO path.

Earlier donor packets retain their own sources: 428 finite tensors and 5,294
exceptional tensors matched their reference under the independent controls.
Source/header/compiler closure and compiled-provider comparisons support scoped
reuse; they do not establish whole-model or runtime performance equivalence.
The deployed four-slot/vision decode uses shared GDN. Flash read-ahead applies
to a separate supported one-slot, text-only region; the engine requires at
least two slots for vision. The wide upstream specialization needs SM 12.0
and at least 96 SMs: neither GB10 SM 12.1 nor the available SM 12.0/34-SM RTX
PRO 2000 meets that gate. Full-model quality and performance selection for
applicable candidates remain pending.

### Remaining qualification and retained policies

Earlier integrated native packets also passed 173 numeric cases with one
second-device skip per architecture; Flash forward 54, multistream 28,
NVFP4 loader 40, YaRN 11, vision 23 and RAM affine 9 passed per architecture.
Their source/image receipts remain separate; these counts are not combined
with overlapping packets. All use synthetic weights/buffers and bounded
resources. They do not establish pretrained-model quality or performance.
Pinned Triton interpreter teardown diagnostics are retained and triaged using
type names and matching dependency binaries. Printed function lists are
truncated, so equal counts do not prove complete identity or absence of leaks.

An earlier optional-TUI frozen-source snapshot passed 7,332 checks with 172
skips; its source manifests matched before/after. It remains a distinct result.
Maintained tests of unselected GPU optimizations are preserved with their
evaluation packets. Actual pretrained-model/API and controlled performance
comparisons remain pending. Apple execution, true multi-GPU NCCL and additional
GPU architectures remain unrun where hardware is unavailable.

Grouped attention is rejected under the existing numerical contract: a finite
BF16 input produces different output under its changed reduction association.
Other unverified CUDA/MLX kernel, scheduling, prebuild, pinning and distributed
Flash Next alternatives remain isolated evaluation packets. Source-only
operation-order reasoning does not establish their integrated performance.

Flash Next's concurrent rounds read a step's codec status words in its sampling
packet's host wait, read a window's n-gram rows behind the layers before their
layer, run on static round tables at a fixed launch geometry, replay CUDA graphs
captured within `TENSORFOLD_MULTI_GRAPH_MIB`, launch their decode kernels with
programmatic dependent launch on GB10, and score and select every stream's
sparse-attention blocks in one launch, long contexts' with an exact multi-CTA
select. A GB10 A/B selected them on 2026-10-13: image `arm64-runtime-263d7203`
served the deployed Flash Next snapshot (`2b170fa6…`) with the production flags
(`rotorquant6-norm`, `--parallel 4`, MTP depth 4 at confidence 0.5, copy drafts)
except `TENSORFOLD_MULTI_GRAPH_MIB=1024` (keys captured with and without PDL;
the default is 512 MiB), each alternative behind a benchmark-only switch in that
one container. A cell ran every arm in mirrored order on one resumed
(prefix-cached; at 300k each cell's first arm prefilled ~43k tokens first)
public-text prompt of ~30k, ~150k, ~300k or ~650k tokens (explain and rewrite
tasks, 650k explain only; seeds 11/22/33; up to 1,536 tokens at reasoning effort
low), one stream at a time (6 cells a context, 3 at 650k) or in two four-stream
mixes (12 streams in 3 groups each); a 650k rerun (seeds 44/55/66, 10 GiB memory
reserve) repeated the host-wait, n-gram, packet, PDL and suffix arms. Every
arm's tokens were identical. Each figure below is the removed path's decode
tok/s against the kept one (mean paired log-ratio, 95% bootstrap CI; negative:
the removed path was slower):

- one-launch and multi-CTA select: one stream +0.48% [-0.26, +1.24] at 30k,
  below the multi-CTA threshold, -1.54% [-2.01, -1.04] at 150k, -4.16%
  [-4.76, -3.47] at 300k and -7.99% [-8.77, -6.57] at 650k; per stream in the
  mixes -3.77% [-4.96, -2.79] and -4.76% [-5.44, -4.04].
- captured rounds on static tables: one stream -2.15% [-2.62, -1.54], -1.73%
  [-2.39, -1.01] and -2.30% [-3.02, -1.10] at 150k, 300k and 650k, and -1.59%
  [-2.18, -0.92] per stream in the second mix; at 30k (-12.20%) and in the
  first mix (-16.70%) the removed path's first requests also compiled its
  kernels within their decode.
- late n-gram reads: one stream pooled over 30k to 300k -0.35% [-0.79, +0.10];
  650k -0.36% [-1.17, +0.85], in the rerun -1.00% [-1.71, -0.27].
- one host wait a step: one stream pooled -0.02% [-0.43, +0.43]; the second
  mix's aggregate +0.99% [+0.12, +1.63] (3 groups; per stream +0.08%
  [-0.54, +0.68]).
- PDL: one stream pooled -0.01% [-0.62, +0.60]; per stream in the mixes -0.59%
  [-1.54, +0.24] and +0.64% [-0.01, +1.29] (the second mix's 150k streams
  +0.88% [+0.05, +1.76]).

The host-wait and PDL intervals above zero are single unreplicated exclusions
among many comparisons; the predeclared independent rerun of such a bucket did
not run before testing closed. A percentile bootstrap of three groups or six
streams under-covers: paired-t 95% intervals of the same rows are -0.95 to
+2.92% (host wait, the mix's 3 aggregates) and -0.36 to +2.12% (PDL, the 6
streams), so neither is a measured regression, and both stages are kept under
the owner's rule (keep unless a measured regression), without an established
gain. Lean sampling packets were rejected: the original path was faster, one
stream pooled +0.41% [+0.07, +0.77] (paired-t [+0.01, +0.80]) and per stream
in the second mix +0.74% [+0.07, +1.45], with the same sign at every
one-stream context (30k to 650k) and in that mix's aggregate. The A/B's base
arm also ran a cost-aware draft rule and a session suffix drafter; both were rejected, so drafting keeps the confidence
rule, and this combination of kept stages is not itself measured. Prompt
passes also launch with the decoder's PDL policy; prompt throughput under the
kept round stages was not A/B-measured. Captured keys' memory is measured from
host available memory, which moves with page-cache state on unified memory, so
the captured set can differ between starts; which keys the default 512 MiB
captures on GB10 is unmeasured. RotorQuant 6 prompt passes decode centroids by
register gather: a separate GB10 container A/B (`5e81802c` against
`32b4ab99`, two starts each, ABBA, cold ~30k and ~150k prompts) measured
prompt throughput +33.8 to +37.7% (95% bootstrap CIs within +32.1 and +38.8%)
with identical tokens; decode kernels are unchanged. The switches are removed
from this source, whose kernel sources equal the A/B image's. Its
synthetic-model CUDA suites pass on an RTX PRO 2000 Blackwell (sm_120, 894
tests: each stream's tokens equal its solo run, and the static tables and graph
replays equal the per-stream forward bit for bit) except five prompt-attention
bit tests that fail identically on the production source, and a synthetic
two-stream RotorQuant 6 decode hashes its tokens and caches as before. Its
installed image (`fd0205a0`) qualified on GB10 and serves there since
2026-10-14 ([deployment](../deploy/gb10/README.md#performance-build-deployment-2026-10-14)):
reference tokens at ~30k and ~150k, four concurrent streams equal to their
one-stream tokens, and only machine code already run on the GB10. Its
full-model quality comparison and four-stream throughput at the default
512 MiB graph budget (which captures every one- to three-stream key and the
first four-stream keys) remain unmeasured.

Decode-bandwidth build (`52b8f229`, deployed 2026-10-14 with RotorQuant 8-norm K/V, the canonical format): exact
changes, every one keeping each output's arithmetic, selected on an RTX PRO 2000 Blackwell (sm_120, 34 SMs) with a
real-shape synthetic four-layer Flash Next round (`tools/bench_decode_round.py`) and qualified on GB10.
Six- and eight-bit RotorQuant decode chunks decode 64-column quarters with table centroids
([contract](cuda-kernel-contracts.md)); decode hyper-connection read-outs run the separate kernels with scales and
biases loaded ahead and the stored padding rows masked; DeltaNet chain trees run 4 value rows a warp. On sm_120
(600 rounds, interleaved ABBA against `18a7c448`, identical tokens in every run) the round median fell 11.3% at one
stream and 150k (pairs 0.894, 0.876, 0.892), 9.4% at four streams (0.907, 0.904) and 10.6% at 400k (0.895, 0.894);
150k prompt fill rose 1.65x. On GB10 (container A/B against the `fd0205a0` image, two starts each, ABCCBA, prompts
of ~30k to ~300k, 512-token replies) every reply equalled the previous build's tokens, decode rose 5.4 to 6.5%
(mean +5.9%) and prompt passes 1.44 to 1.59x; the same build reproduced the previous build's reply tokens at
~30k, ~150k, ~300k and ~650k. On GB10 the trees' 8 x 4 shape decoded 0.1 to 0.5% faster than 4 x 4 in each of the
five requests (mean +0.3%); this build runs 4 x 4 on every chip, and later sources pick the chain shape by chip
(`CHAIN_ROWS` in `src/tensorfold/cuda/kernels/gdn.py`: 4 on sm_120 with 34 SMs, else 8, the shape before this build).
Rejected: one pooled-key tile load per stream in concurrent index scoring (no measured gain: a tile's row programs
already share it through L2). Full-model quality is unchanged by construction (identical tokens); four-stream
throughput on GB10 was not measured.

Decode-bandwidth round 2 (`a05c3219`, deployed 2026-10-14 over `52b8f229`;
[receipt](../deploy/gb10/deployment-bw2-20261014.json)): exact changes, selected with the same sm_120 round bench
(rotorquant8-norm, 150k, 600 rounds, interleaved ABBA, identical tokens in every run) and qualified on GB10. The
attention chain preps each head's value side separately, encodes eight-bit K/V against a threshold table
(`AttnScratch.thresholds`), skips RoPE angles past the rotary dims and merges one query head a program: round 0.986x
at one stream (pairs 0.985, 0.987, 0.986), 0.998x at four, 0.990x with rotorquant6-norm (four pairs); in-round
profiles show the prep launch 33.8 to 7.1 us. The lane matmul's block and cluster choice and the decode expert window
tiles come from per-chip tables (`qmm.LANE`, `experts.WINDOW_TILES`) whose only entries are sm_120's: round 0.970x at
one stream (three pairs), 0.960x at four (two pairs); the LM head's 17- to 32-row entry (block 5) measured 0.987,
0.986 and 0.994 in an isolated four-stream A/B and is kept. DeltaNet chain trees take their rows a warp from
`gdn.CHAIN_ROWS` (4 on sm_120 with 34 SMs, else 8). Against the deployed `52b8f229` source the combined build ran the
round 0.954x at one stream (0.954, 0.952, 0.957) and 0.946x at four (0.944, 0.947), prompt fill unchanged. On GB10
(container A/B against the `52b8f229` image, ABBA, prompts of ~30k to ~300k, 512-token replies) decode rose 1.2 to
2.0% (geometric mean +1.7%, every new-build start above every old one per request), prompt passes 0.2 to 0.7%, with
identical tokens; the build reproduced the reference reply tokens at ~30k, ~150k, ~300k and ~650k. GB10 lane and
expert-tile table candidates (ABCDDCBA, one and four streams) were inconclusive (base pass-to-pass spread 1.5% at one
stream, 2.8% at four), so the GB10 keeps its previous launches. The 2026-10-15 follow-up closed the lane/tile
candidates without selection: full lane+tile regressed one-stream verify time, tile alone remained inconclusive, and
the lane table restricted to at most four rows established no complete-round gain across both regions. The original
launches remain selected. Rejected on sm_120: a router decode tile of 16 experts (no in-round gain; reverted),
prefetching shared-expert weights into L2 during routing (1.006 to 1.018x slower), and a fused top-k plus plan kernel
(1.7 to 3.5x slower). Left open below round A/B resolution (< ~0.1% of a round each): alternative lane blocks for the
attention output, DeltaNet/attention 17- to 20-row and per-layer-embedding projections. Full-model quality is
unchanged by construction; four-stream GB10 throughput was measured only in the synthetic table bench.

Current deployed RQ8-norm/YaRN-4 functional receipts and historical INT8/YaRN-2,
EXL3, startup and transport comparisons keep their original image, target and
workload scope. Actual integrated native/device/model/performance qualification
is reported separately; CPU or fake controls do not certify missing targets.

## GB10 Python deployment, 2026-10-12

The user authorized permanent replacement based on the completed source/native
qualification. ARM64 runtime `8e1fe707…`, source `2edfaa51…`, now serves the
unchanged pinned Flash Next model and 1M/YaRN-4/RQ8-norm/four-slot configuration.
It uses a new derived cache host directory; the old image's full configuration
and old cache are retained in an independent stopped rollback container.

Actual local/LAN health, model identity, package/module 0.6.6, Unicode
`/v1/tokenize`, independent arithmetic, full HTTP/SSE consumption and complete
returned token arrays passed. Target-only/MTP/warmed MTP match for all 64 IDs;
four concurrent replies and their warmed repeats match at their fixed seeds.
All 12 requests across six batches completed, yielding 718 output tokens. The
process remained healthy, with zero restarts and observed `OOMKilled=false`.
First-use MTP/shared delays and repeated requests remain separate observations;
there is no controlled baseline or population performance conclusion. Filled
1M quality/capacity and live image/video/tool/grammar queries remain outside this
finite deployment batch. Exact source/image/settings and rollout scope are in
[the deployment receipt](../deploy/gb10/deployment-python-20261012.json).

## GB10 deployment selection, 2026-10-10

The [earlier GB10 measurements](../README.md#earlier-gb10-deployment-with-yarn-2)
use runtime image `5b6627ba…`, built from `5fe0e89d…`, with the unchanged
Flash Next affine 4-bit/group-32 checkpoint at revision `2b170fa6…`, INT8 K/V,
524,288-token YaRN-2 context, four slots, vision and thinking. The selected
profile retains MTP-4/confidence 0.50/copy mode and increases prefill rows from
512 to 2,048. Three fresh logical 1,886-token prompts with 64 generated tokens,
target-only decoding and zero cached tokens measured 0.9786/0.9848/0.9795
seconds of engine prefill with 512 rows versus 0.2413/0.2341/0.2355 with 2,048.
All three same-format output token hashes matched. This is integrated
medium-prompt evidence, not a general long-context throughput claim.

Selected warm solo MTP/target-only medians were 57.40/36.21 client tokens/s;
four simultaneous 128-token replies yielded 115.16/115.98/114.63 aggregate
tokens/s over three measured batches. Counts include reasoning and timing
includes whole HTTP/SSE consumption; one warmup was excluded. The original
four-request profile and its repeat measured medians of 114.23 and 116.10.
Six drafts and disabling copy mode did not demonstrate a repeatable short-decode
gain. Retain the original draft/copy policy and attribute the selected
improvement to the measured medium-prompt prefill region. Small samples, warm
kernel caches and one selected process limit uncertainty claims. Greedy
sample comparisons and concurrent references passed; they do not establish
BF16 equivalence or general accuracy for this quantized model.

The 114 GiB cgroup limit and equal memory-plus-swap limit remain unchanged.
Unified GPU memory is not fully charged to the cgroup, so host MemAvailable is
required for accounting. Four configured slots do not imply four full context
windows fit. Latest-image full-524k stress was not run; the earlier retrieval
retains its original image scope. Deployment restart and feature receipts are
recorded with the [GB10 profile](../deploy/gb10/README.md).

## Acceptance

Record complete requests, generated token IDs and the required intermediate
state oracle. Use the same accepted requests, prompt/context sizes, sampling
keys, cache lifetime and minimal instrumentation in both arms. A microbenchmark
supports investigation; adoption requires evidence in the affected complete
workload. Diagnostic or profiler timings do not establish serving superiority.

Compare balanced run orders and retain each region's distribution and
uncertainty. Evaluate throughput, first-token latency, decode latency, tails,
startup and resource peaks according to the region's ordered objective. Never
hide a required-path regression in an aggregate or subtract estimated overhead.
There is no minimum gain threshold. Keep a path only where repeated evidence
shows an improvement while all required semantics and budgets hold.

Four simultaneous requests are a separate region from serial generation.
Expert weights are shared, while KV state, scratch buffers and captured plans
consume additional memory. Compare natural capacity and the actual resulting
cache/copy traffic. A captured path that loses in four-request service is not
promoted in that region on the strength of a serial microbenchmark.

Record GPU allocated and reserved peaks, observed CPU/cgroup memory, swap,
OOM events, copy counters and teardown. A memory estimate is admission planning;
it does not replace observed whole-operation peaks. All accepted tasks must
retire before releasing the model, streams, graphs or borrowed file state.

## File and snapshot evidence

The POSIX descriptor owner uses a valid closed slot published before native
acquisition. See [descriptor ownership](fd-ownership.md). Root's current native
qualification passed 15 primitive controls and 13 common-file/publisher controls
with zero skips. The consumer result binds the exact source and private binary;
it does not qualify a later installed wheel or another platform automatically.

Snapshot source controls exercise explicit initialized cache schemas, bounded
metadata, expected token prefixes, content identity, private copies and failure
retirement. Disk class strings cannot select executable code. Successful cleanup
preserves ordinary exception suppression, so malformed cache data can remain a
cache miss. Real cleanup failures remain operation failures with their original
primary error and secondary context.
Disk indexing, restoration, eviction, spilling and shutdown apply the same
failure classification. An outer recovery handler cannot convert failed
resource retirement into a successful cache miss.

FORMAT2's masked whole-file SHA detects accidental byte corruption. It is not
authentication of an attacker-controlled snapshot. Restore verifies bytes while
making its private copy. A bounded unchanged-file memo may suppress a redundant
write verification; it cannot suppress restore verification.

The complete-byte cost job measures 8, 64 and 256 MiB payloads with balanced
raw-copy and integrity-copy orders. Input generation and initial sealing are
reported separately. These inputs have observed warm cache state; the job makes
no cold-disk claim. Both arms consume complete bytes and include private-copy
retirement. Results describe byte-copy/hash cost, not model token throughput.
The recorded Linux run completed all 24 copies and three seals. Its independent
oracle binds generated bytes before sealing; it does not reuse the sealer's
digest algorithm as its expected result.

The real NumPy restore job uses a declared synthetic cache schema and F32, F16
and I32 words through the canonical capture, publisher, seal, registry and restore
code. It checks bitwise storage, metadata refusal before the loader, corruption,
selected-prefix binding and private-file retirement. It does not emulate MLX or
qualify built-in family cache layouts, lazy arrays, resumed model state or output
quality. Those require the corresponding runtime and model.
The recorded run passed five checks across three tensors, four schema refusals
and two actual array-loader calls. Its short duration produced no host polling
distribution. These are source-bound results; a later installed wheel must pass
its own current-source and native-library verification.

## Generated table and fault evidence

The generated table fixture passed actual Linux execution across four layouts
and 32 files totaling 702,569,794 bytes: eight PLE lifecycle controls, Python and
native C++ SSD gather oracles, and four matched IO comparisons with 16 records
each. Separate native fault controls passed for mapping, production prefetch,
Python SSD and native SSD ownership. These results establish their bounded
byte and lifetime contracts; they do not select an IO performance winner or
qualify full-model serving, an installed wheel, or Apple execution. The
per-worker file-reuse alternative remains an unselected task artifact. Its
correctly armed native fault control also passed: the injected close failure
retired its published owner without retrying or closing the replacement
descriptor. This proves the tested fault contract, not a performance win.

## Historical Flash Next generation evidence

The compact-cache comparison below used prefetch disabled in both arms. Its original image was
`sha256:afed8f340bba01d2333d2408faeb2daf9d3209c7411cc0a87482bccf7bed0387`;
the compact image was `sha256:fb20eb4c7c55018c93f889c1705bdc4b8d56a3dd2845f3dc4e509b45eedc5fdc`.
An alternating baseline/candidate/candidate/baseline comparison on the RTX PRO
2000 Blackwell used the pinned EXL3 checkpoint described in the
[system RAM expert recipe](recipes/ram-experts.md), four slots, a 2,048-token
context, INT8 KV, and YaRN factor 2. Each region had 20 matched trials. Across
240 timed requests, all 64-token outputs matched the unchanged teacher exactly.

| Region | Baseline median tokens/s | Candidate median tokens/s | Paired geometric ratio |
| --- | ---: | ---: | ---: |
| Serial greedy | 8.09 | 11.64 | 1.439 |
| Serial greedy with MTP | 12.76 | 21.95 | 1.717 |
| Four simultaneous greedy requests, aggregate | 48.82 | 81.71 | 1.677 |

These rates include complete timed request execution; they are distinct from
decode-only rates. Observed first-token and emission-stall tails improved in
all three regions. Expert bytes copied per output token fell by 52.3%, 60.9%,
and 61.4%, respectively. Constructor time increased from 225–228 seconds to
280 seconds, a 52–55-second startup cost. The declared objective prioritizes
sustained serving throughput over startup while retaining the memory and
startup limits.

The observations are nested within four fresh processes, with only two starts
per arm. Their paired uncertainty is conditional on those observations; it is
not a population tail bound or a guarantee for other prompts. The candidate
was built from an intermediate source snapshot. Later builds require the
relevant source/runtime provenance and final installed API checks before
reusing this evidence.

A separate 82-generation run checked seven sampling policies, grammar and
log-probability outputs, four concurrent mixed-policy requests, mapped draft
heads, and exact draft confidence. It passed 83 scalar, 754 stream, 16 mapped,
and 280 draft-confidence comparisons, with 1,117 source-byte preservation
checks. The draft head used the model's actual 79,591-entry nonidentity map.
An earlier failure came from a test oracle that normalized concurrent rows one
at a time instead of preserving the production batched operation. The corrected
oracle kept exact bit comparisons and passed a fresh run; production arithmetic
was unchanged. These checks cover the declared INT8 region, not every optional
KV format or an unrelated model family.

## Current default-prefetch Flash Next reference

A separate completed qualification used normal Source06 image
`sha256:cf99e0cf0ab8007f0cfcad962c7df3aac8a845f743619fa9e3616239b8847a08`,
source UID `0b6aa865d82a70e5b2633e90c6ad3e8c7670579f45a078b235f904e81ca48c2e`
and wheel `80f1c23f390da93ead7a88928f3b19b0448508567c5e95464c7b9940d8c9842e`.
The same pinned EXL3 model ran with automatic compact expert caching,
configured context 2,048 with four slots, INT8 KV, YaRN-2, eager execution and full default
n-gram prefetch. MTP requests used depth 4 and confidence 0.5. Stock-policy
median complete generation rates were 11.74 tokens/s for serial greedy,
21.94 for one MTP request, and 82.39 aggregate across four concurrent MTP
requests. These are 20 warmed short-prompt trials from two stock-policy
starts per region, with 64 output tokens per request. They measure consumed
public `Engine.generate` work with synchronized GPU completion and callback
first-token/stall timings, not HTTP client performance.

All four starts, including the two private notification-alternative starts,
completed full prefetch, refused optional table pins, reproduced the original
64-token teacher for all 308 requests, and retired cleanly. The jobs held
the 56 decimal GB/no-swap RAM budget and 15,868,952,576-byte GPU grant;
observed sampled cgroup RAM and Torch allocated/reserved peaks were 46.595, 14.002 and
14.321 GB. The original notification policy remains selected in all three
regions. The guarded alternative's positive MTP estimate did not establish
repeatable superiority across process starts; this is an uncertain result,
not an equivalence or regression claim. The current reference does not imply
a gain over the historical comparison or qualify other policies,
HTTP serving, future images, general answer quality or sustained full-context
capacity. See the [README reference](../README.md#current-default-prefetch-reference)
for observed first-token and emission-stall summaries.

The current notification comparison is closed with the original policy retained. A separate full-request optional expert-arena comparison also retained the fixed expert-cache policy in serial, MTP and four-request workloads: it did not establish a repeatable improvement, and no successfully active arena loan was observed in the timed trials. The separate aggregate-room-clipping diagnostic completed 1,320 observations: one allocation success, 1,319 classified native OOM refusals, no aggregate precheck refusal and no positive strict-partial-room opportunity. Its formula selected the original requested size in every case, so clipping was inapplicable in this measured region and no arena-lending/clipping implementation is retained. No smaller arena was allocated. This single-process diagnostic does not identify the underlying refusal cause, hypothetical smaller-allocation success or other-region behavior, and it carries no throughput credit. These decisions do not change the current stock observations above.

The maintained source integrates the selected `rotorquant7` lookup decoder only within its existing controlled Blackwell/recorded-runtime, plain symmetric codec-4/head-256, one-request ordinary foreground, unconstrained/no-logprob greedy serial/MTP region, with no waiting/copy/background/owed-token/carry states. Native and composed source/image provenance passed; all 128 quality tokens and 1,352 ordered same-format score/state/attention/pending/confidence records matched exactly. Four balanced performance starts with 20 matched 64-token pairs per workload showed gains in both run-order halves. Only two starts per arm were measured, and independent two-process-contrast intervals include 1; selection does not assert population confidence or universal superiority. The [README controlled comparison](../README.md#controlled-rotorquant7-decoder-comparison) identifies the exact images, settings, medians and paired ratios. Full default-prefetch, concurrent requests, other formats/settings/hardware and BF16 quality remain outside this selection, and no new public flag is added. Frozen selected-source shipping audits, baked packaging and fresh API qualification are complete; the controlled performance comparison retains its original image and region scope.

## Completed installed-origin Linux CPU qualification

The normal verification image `sha256:aeef439e98eb2c885e5c131105e2d9b612410e81336d6fa63c36078eb6522b64`
used the same Source06 wheel `80f1c23f390da93ead7a88928f3b19b0448508567c5e95464c7b9940d8c9842e`,
Python 3.12 and pytest 8.4.2. A qualified read-only fixture supplied maintained tests/tools;
package and native-provider imports remained the normal installed copies. The qualified Source06 manifest
contained 387 modules. Its complete 386-path Linux inventory finished in 25 successful chunks:
6,369 cases, with 6,214 non-skipped passes, zero failures/errors, and 155 explicit skips.
The separate Apple-only module contains two off-target tests. Skips remain unrun for their targets;
source/fake CPU controls do not qualify GPU/model math, Apple or true two-GPU execution.

The combined result reuses 16 original successful chunks and eight corrected-fixture reruns,
then adds the final Dense EXL3 cleanup-fixture rerun. Each retains its actual source and fixture
identity; earlier failed receipts are preserved. Native UID-1000 checks verified all 1,033
qualified fixture files before and after the final rerun. This closes the scoped Linux CPU lane;
current baseline model generation and CLI/API results are recorded separately. The selected Rotor7 decoder and new maintained native GPU test are not blanket-certified by this earlier CPU snapshot; their native/model evidence and completed frozen shipping-image/packaging qualification are separate.
See [verification and development](../README.md#verification-and-development).

## Completed current installed CLI/API qualification

The ordinary installed PID-1 CLI and HTTP server in the normal Source06 `cf99…` image above
completed seven functional groups. The configuration used full default n-gram prefetch,
automatic expert caching, four slots, context 2,048, INT8 K/V, YaRN-2, eager execution,
BF16 prompts and MTP-4/confidence 0.5. Its positive limits were five unfinished requests
and 32 HTTP connections, rather than new defaults.

Repeated serial/drafted token-ID and exact log-probability results, the legacy endpoint,
xgrammar `TFOK` followed by actual EOS, a 16-token `ignore_eos` length result, and invalid
requests refused as JSON400 before SSE passed. Four occupied lanes and a FIFO fifth request
were observed; a sixth streaming POST received JSON503 before SSE with the original owners
preserved. Cancellation admitted the queued request and recovery completed. Final request ownership
was zero; the live engine was idle with no running jobs or active streams. The health socket still counted toward HTTP ownership.
SIGINT returned zero; the server/client/container retired cleanly. The `ignore_eos` case
does not prove suppression of an unobserved EOS.

The 360.63-second whole operation included startup and first native compilation. Sampled
cgroup RAM peaked at 47.386 GB within 56 GB with no swap. This is the completed current
baseline API gate, not an API throughput/tail comparison, all-format/model quality proof
or 524k-context qualification. Future changed code/images require their relevant source,
model/API and resource qualification. See the [README API qualification](../README.md#current-installed-cliapi-qualification).

The final frozen-source shipping gates are also complete for commit `07f4294306aba4cc6e1851b9c774fbf89d234900`, runtime UID `e7a15d7331560e25bc31bcd5caedf27c6c81c7dbc6d19b2f74d281c78534daba`: serving image `sha256:b091954b6556a540123d2872d30fe86905344f4b24a6ff47e8361269916f7117`, verification image `sha256:46f16ad6498378a3d9a186dcbfc671c1459e86765a2ee21b950291517968455f`, and shared wheel `4d79339aa56286d6d5863e1d37c6584abacb79938ca97c75386649f62f32fbd2`. Both normal Python 3.12 installed native/source audits passed 15 controls without skips. The separate baked CPU24 packaging batch passed 343 cases, with eight explicit skips, over 16 modules as UID 1000 without a checkout mount; genuine CHANGELOG/update/public-permissions checks passed. These counts do not replace or extend the historical Source06 full CPU snapshot.

The initial same-serving-image GPU probe reported 15,868,952,576 free bytes/16,584,343,552 total and zero Torch allocated/reserved bytes, not loaded-model occupancy. Fresh ordinary installed PID-1 CLI/API qualification then passed all seven functional groups under the same recorded INT8/default-prefetch/four-slot/context-2048/YaRN-2/auto-expert settings and test caps 5/32, with FIFO/pre-SSE refusals/cancel recovery/zero final logical owners/SIGINT zero and clean process/container retirement. Its 359.99 s includes startup/warmup/first request compilation; 349 running samples observed a cgroup peak of 46.036 GB under 56 decimal GB/no swap with no sampled swap/OOM event, ending before final retirement. No new throughput/tail/full-context/all-codec/quality conclusion follows. This final documentation-only result update leaves qualified runtime code unchanged; frozen artifact IDs are not aliases for later HEADs or rebuilds. See [the shipping results](../README.md#frozen-source-shipping-qualification).

## Final release checks

Serving can optionally bound CPU request ownership with three positive integer
flags. Their default is unset, preserving the existing admission and dispatch.
These controls change capacity policy; they do not alter model arithmetic or
discard generated tokens.

- `--max-http-connections` caps retained HTTP handler/socket owners. At capacity,
  the server pauses acceptance and leaves unaccepted connections in the existing
  listen backlog of 128. Completed owners wake acceptance; shutdown wakes a
  saturated wait. Idle keepalive and monitoring connections count toward this
  cap, so allow room for health and metrics clients as well as generation.
- `--max-pending-requests` caps unfinished logical generation requests, including
  queued, running, and canceled work whose engine ownership has not retired.
  Background continuations and replay retain their request's admission. A new
  request at capacity receives HTTP 503 before SSE begins. Configured streaming
  replies open their headers after admission, when output or completion is known.
- `--max-engine-calls` caps queued or running calls on the MLX engine thread,
  including calls whose waiting caller timed out. A timeout does not release a
  callback still owned by the engine. Unsupported backends refuse this flag.

Choose counts for the deployment's request sizes and RAM budget; model slot
count does not determine an HTTP worker or waiting-request cap. `/health`
reports configured limits and their owned counts under `admission`. `/metrics`
reports HTTP connection ownership and configured request/RPC limit gauges.
Source controls exercise real owned sockets and threads, including saturation,
retirement, shutdown, and pre-SSE refusal. Installed runtime and complete model
generation checks remain separate release gates. On Linux, a reused HTTP
connection enables TCP_NODELAY once; its first operation retains the stock socket
setting. Paired complete `/v1/models` requests on `.250` reduced reused-connection
medians from 40.8–41.1 ms to 0.25–0.52 ms across both codecs, with 20 balanced pairs
per region and equal complete responses. Fresh-connection effects remained within
measurement uncertainty; other platforms retain stock parsing. These measurements
cover HTTP metadata handling and do not establish LLM generation speeds.

Final release requires current installed source/wheel/native provenance,
affected model and API generations, cancellation/shutdown/resource checks,
controlled performance comparisons, and a post-change bottleneck review.
Linux native file composition does not qualify Apple execution. Record unavailable
hardware gates explicitly; source controls and substituted runtimes are not
runtime passes. Retain rejected candidates and failed controls with their exact
inputs until final reconciliation establishes which evidence remains applicable.
