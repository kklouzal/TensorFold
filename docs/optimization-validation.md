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

## Qwen3.8-27B AEON mixed NVFP4, 2026-10-19

Source: the per-channel FP8 loader and kernels, `--tokenizer` and the Qwen3.8 dense static YaRN policy (branch
`campaign-qwen38-20261018`, measured from `35b61ae4`; the rebased commits are patch-identical). Checkpoint
`AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED` revision `282e6775`, drafter `z-lab/Qwen3.8-27B-DFlash2`
revision `50307d4c`, and the `tokenizer.json` of `Qwen/Qwen3.8-27B` revision `1d4bf0f2` through `--tokenizer`; one
DGX Spark (GB10, SM 12.1), `--precision checkpoint`, bf16 KV, one stream. Its 232 FP8 projections are ModelOpt
`FP8_PER_CHANNEL_PER_TOKEN`; see the [27B recipe](recipes/qwen3.8-27b.md#aeon-mixed-nvfp4-export).

- Numerics: the per-token quantizer matches a PyTorch reference byte for byte, scales included; the FP8 GEMMs (decode
  lanes, split-K, prompt GEMMs, W8A16 under `full`) match float64 references of the declared math within the retained
  empirical product-magnitude regression envelopes; those thresholds are not deterministic error bounds. A separate
  integer cancellation oracle has zero rounding error in fp32 for any accumulation order. Rows, chunks and split-K
  parts are bit-invariant. 205 GPU checks passed on the GB10,
  the checkpoint's own included; the kernel and synthetic-model suites also pass on an RTX PRO 2000 Blackwell (SM
  12.0). A 60 s FP8 GEMM stress at the export's shapes stayed within 0.0018-0.0027 relative error of an fp32
  reference.
- Exactness on the checkpoint (context 4,096): drafted replies equal serial ones token for token, greedy and two
  seeds, under `checkpoint`, `full` and the FP8-only math SM 8.9-10.x GPUs run. Through `tensorfold serve` (512
  tokens, both thinking modes, short and 4,096-token prompts), each greedy reply resumed from a cached prompt equals its
  cold run (seven cells).
- Decode, 512 tokens: on the engine path at an 8,192-token context, 76.4 (code), 30.5 (chat), 81.1 (JSON) and 11.1
  (Hindi) tok/s greedy and 70.7 / 26.6 / 79.9 / 11.0 sampled (T 0.6, top-k 20, top-p 0.95), against 11.2-12.2
  serially; a 16-row verify round took 96-97 ms and a draft 8-10 ms. Through `tensorfold serve` (one run a cell):
  greedy 31.0-48.3 tok/s with thinking and 32.8-52.6 without, at short and 4,096-token prompts. Hindi drafts almost
  nothing: the drafter scores only the vocabulary's first 98,304 ids and its control tokens.
- Prefill on the engine path, each prompt cold: 1,745-1,753 / 2,574-2,581 / 2,221-2,327 tok/s at 4,096 / 16,384 /
  32,768 tokens; through the server a 4,096-token prompt took 1.56-1.57 s.
- YaRN code on the default path: one interleaved pair against the build before it, at the default 262,144-token
  window, gave the same tokens (serial and drafted, greedy and sampled) and prefill replies at 4,096 and 16,384 tokens,
  at 0.999-1.005x the speed. The second pair is pending.
- GB10 observations: a 32,768-token prompt draws 87-95 W with SM clocks at 2,288-2,350 MHz and no clock-event reason
  (boost backing off on power), and takes the hottest thermal zone to about 94 C within 15 s. After minutes of decode
  near 52 W, prompt bursts reached the firmware's SW thermal slowdown, and the GB10 stopped responding twice at the
  start of such a burst (2026-10-18 23:33Z and 2026-10-19 10:20Z, memory flat). The cause is under investigation, and
  further runs on the GB10 are on hold.
- Pending: the strict and non-strict tool suite, needle recall, 32k and longer server cells, KL and agreement against
  the BF16 export, and the extended YaRN window (1,048,576 positions).

## Qwen MoE CT NVFP4, 2026-10-17

Installed source `099eb4d0` ran the original pinned Heretic 35B-A3B compressed-tensors
export and its separate BF16 DFlash v1 model on the 16 GB RTX PRO 2000 Blackwell.
The [qualification](qwen36-ct-nvfp4-validation.md) and
[receipt](verification-qwen36-ct-nvfp4-20261017.json) identify actual immutable
images, native numerical checks, serial/drafted/seeded identity, strict tools,
streaming, cancellation and four concurrent complete replies. Tests used no
container CPU/RAM/PID quota, zero optional VRAM reserve and an 8 GiB available
host-RAM floor. Required future state and loading reservations remain admitted.

At context 4,096 with eight BF16 draft rows and registered authority, the selected
profile reached about 37 decode tok/s. Nine rotating symmetric blocks compared
54 complete 512-token replies with identical tokens, work and fixed expert pool;
prepared eight-row vocabulary projection was consistently faster than the
four-row and original singleton policies in that experiment. Reported median
intervals are conditional on independent representative block effects; no
unconditional population-confidence or global-optimum claim follows. The earlier
four-row policy ordering and BF16-versus-four-bit drafter speed ordering remain
unresolved. Four concurrent requests in a separate four-row profile completed at
57.70 aggregate client tok/s, which excludes final device retirement and is not
an isolated speculative-decoding speedup comparison.

## Smaller pretrained NVFP4, 2026-10-17

The ordinary installed desktop image `c74633ad…`/runtime `b551e68f` ran
`surogate/Qwen3.5-9B-NVFP4` revision `3410fbc4…`. Its full weight SHA256 matched
the publisher. Native FP4 activations and BF16 activations against the same
NVFP4 weights were qualified separately: three identical 512-token replies
per mode, supported HTTP/SSE/strict-JSON checks and concurrent/solo identity.
The observed single-request rates were 24.06–24.27 and 34.41–34.73 tok/s;
different output contracts preclude a quality-preserving speedup claim.

The [record](small-nvfp4-validation.md) and
[receipt](verification-small-nvfp4-20261017.json) retain the explicit stop policy,
original default-EOS failure, family-specific unsupported feature refusals,
and composite live-capture/offline validation. The recorded host minimum was
53,324,820,480 available bytes against an 8 GiB floor. This dense checkpoint
has no routed experts or n-gram tables; it cannot qualify the reported full
Flash Next ModelOpt offload path. No application/model/dependency changes or
new deployment were needed.

## Desktop report remediation, 2026-10-17

Runtime `b551e68f` implements admitted optional registered expert authority,
asymmetric same-host NCCL expert ranks, explicit shared KV reservation, bounded
affine persistence and original-arithmetic NVFP4 offload/SSD reads. The
[desktop contracts and performance record](desktop-report-remediation.md)
identify each selected design and measured region. Registered DMA, packed
warm starts and common NVFP4 grouping improved their controlled workloads.
Complete advisory-prefetch and device-residency designs regressed required
regions and do not ship; the established serial-only prefetch gain remains
explicitly blocked on full-scale input and dispatch/retirement qualification.

The [installed receipt](verification-desktop-report-20261017.json) identifies
the serving image, unchanged application layers and separate corrected test
captures. The full CPU rerun passed 7,994 cases. The full non-matrix CUDA run's 2,046 passes
plus the complete 12-case accounting-fixture rerun cover 2,049 distinct passing
cases; the original three failures and nine overlapping passes are accounted
explicitly. This is not a single clean full-suite rerun. No runtime defect or
numerical tolerance was hidden by those fixture corrections.

Both trained local EXL3 registration policies passed the fixed strict-tool,
serial/MTP/concurrent and streaming API checks and clean shutdown. Their token
sequences and the three-format long-context token/KV hashes match the prior
qualified runtime. Native one-rank NCCL/model and the installed Harness child
passed; this one-GPU host cannot qualify actual multi-GPU or SM89 execution.
The reported full affine checkpoint, full pretrained ModelOpt/vision, fresh
exhaustive KV matrix and new ARM64/Apple execution remain outside this record.
No desktop-report speed estimate or additive combined gain is claimed.

## CUDA failure remediation, 2026-10-17

The remediation follows source `0240cefea90fcd83a593384259b7346875e3603c` and preserves the previous
41 failing and 45 error IDs and their logs. Focused checks identify separate causes: Qwen MoE's missing
private-cache arena origin; compiler-dependent Triton/native operation order; a one-ULP defect in the Torch
PLE sum comparator; generic weighted BF16 expert mode differing from GLM's declared combine; stale GLM
vocabulary/token-type fixtures; and an obsolete FileBaton expectation under pinned Torch advisory locks.
No production validator, numerical tolerance, exact-token assertion or optional-check prerequisite is weakened.

Frozen runtime/test/build source `a7b9feed34a814d62bf944de646ac3ed7bf3149e` produced runtime image
`sha256:1528158f426991935c05adfc61a949734a2ccb2b1dba56cd53349b80627b3328`, verification image
`sha256:3d6a57cf578884d60ad360d632dc6be98a082aaceb8a6f869fd39a74752a9cb6`, and wheel
`c50a39c6735dbd546317d7f5bbb1319c25b5b5612b0381e91619bb4837afe8fa`.
The ordinary installed-package gates passed **7,642 CPU tests** (168 explicit skips), **2,012 CUDA tests**
(97 explicit skips), and **231 unprivileged public controls** (six explicit skips), with zero failures/errors.
All 41 earlier failures and 45 setup errors are remediated. The killed-build case now verifies the actual
lock protocol under its updated name. Two CPU and 28 CUDA warnings are existing Torch allocator API
deprecations, retained without suppression. The installed audit matched all 501 owned files; three omitted
README files contain no runtime code. No source overlay supplied the qualified package.

The same RTX PRO 2000 Blackwell/SM120 host and pinned Python 3.12/Torch 2.16-dev/CUDA 13.4/Triton 3.9
stack passed fresh three-format exactness: two streams, 64 outputs each, RotorQuant8-norm/8/6-norm,
aggregate hash `67a762f3ac80e419`. The actual Flash Next EXL3 checkpoint passed the installed HTTP gate
at context 16,384/four slots, RQ8-norm, BF16 prompts and MTP 4/confidence 0.5 under the recorded 56 GB/no-swap
profile. Strict tools, the actual 31-tool catalog, unsupported-schema refusal, repeat/serial/drafted tokens,
SSE usage/progress, four concurrent versus solo replies, final zero logical owners and SIGINT exit 0/no OOM
passed. Nine fixed sequential replies match the previous build by request position; four concurrent replies
match with each prompt's solo oracle, and the text SSE choices/token IDs also match exactly. No failure
capture was produced. These fixed inputs do not establish general answer quality or all-configuration bits.

The whole CUDA nonmatrix run excludes the unchanged KV-pair matrix module. Its earlier 1,666-pass subset retains
its original `2dd77a9a`/image identity: an independent source/dependency proof covers exactly seven unchanged
preparation/encoder/oracle families, not the 3,430 deselected cases or a fresh exhaustive matrix run.
An initial cold CUDA run was deliberately interrupted near 54% and preserved with exit 2; only the complete
restarted run qualifies. Standard compiler-cache reuse kept original keys/metadata and the same pinned ABI;
it makes no cold-compilation or startup timing claim.

The [machine-readable receipt](verification-cuda-remediation-20261017.json) records source, images,
counts, baseline provenance, limits and evidence hashes. Raw development evidence remains in
`/home/kklouzal/tensorfold-remediate-20261017`; preceding failed logs and receipts remain unchanged.
This closing documentation update preserves qualified runtime/test/build bytes; a future rebuild has its
own wheel metadata and image identity.

The numerical correction preserves selected native CUDA arithmetic. Legacy Triton/untiled results can
change where implicit contraction differed. GLM's current model forward already used its family combine;
the corrected generic weighted BF16 API now honors the same rounding. Default Flash Next FP32 expert
arithmetic remains unchanged. The PLE production implementation was correct for the independently checked
fixtures; its old Torch sum oracle was off by one ULP. See [contracts](cuda-kernel-contracts.md).

Pinned Torch uses OS advisory build locks: killing the owner releases the lock even when its marker file
remains. Do not delete that persistent marker to recover a build. Legacy FileBaton recovery instructions
apply only to a runtime actually using FileBaton, after its owners have stopped. The Linux-native test
keeps child identity until all session members retire; its pidfd and waitid/WNOWAIT prerequisites were
verified on the recorded Linux 7.0.0-31 host. This changes test supervision, not runtime platform support.

Component EXL3 timings retain exact outputs but do not qualify model-wide throughput or equivalence.
Existing API/statistics readout-performance limitations below remain separate; this remediation makes no
new overall monitoring-overhead claim. Apple, real two-GPU execution and fresh GB10 qualification remain
outside the owner-selected local-host run.

## API/statistics and strict tools: local qualification, 2026-10-17

Runtime source `2dd77a9a794a9d2349c43adf487b12f21a0a5d0f` integrates the interrupted cleanup,
API/statistics and strict-function work. Subsequent qualification documentation changes leave its runtime,
test and build files unchanged. The immutable AMD64 runtime is
`sha256:0768e447d8b135089a1f2a185398d1c9aa0a9fcfcc14711478d990c74cb3a1c3`;
verification is `sha256:9f1004f96f51ee9e21c39d661dc6bf64fc44856224a9c033ac4a4303dd61abe0`.
README changes affect future wheel metadata; these image IDs identify the tested artifacts, not a later rebuild.
The [machine-readable receipt](verification-api-stats-20261017.json) records inputs, gate scope and evidence hashes.
Raw local evidence is retained in `/home/kklouzal/tensorfold-resume-20261017`; it is not part of the ordinary build.

The host is Linux AMD64, RTX PRO 2000 Blackwell 16 GB (SM120), approximately 62 GiB host RAM,
70 W GPU power cap. The normal installed Python 3.12.3 image uses the existing CUDA 13.4.1 pins:
Torch `2.16.0.dev20261006+cu134`, Triton `3.9.0+gitaad2a60d`, xgrammar 0.2.8,
tokenizers 0.23.1, Jinja 3.1.6 and NumPy 2.1.0. The NVIDIA compatibility userspace driver is
615.71.09 over host kernel driver 595.91.07. Other GPU workloads were stopped during qualification;
GPU tasks used one shared lock. This local scope was explicitly selected by the owner; no GB10 service was changed.

| Gate | Result and scope |
|---|---|
| Final installed affected CPU | 476 passed, one MLX-only skip; API/error/schema/reader, statistics, metrics, deployment and packaging contracts |
| Earlier full installed CPU at `8583d43b` | 7,596 passed, 168 skipped, two failures; both defects corrected and covered by the final affected gate (optional build-label oracle and packaged API reference) |
| Unprivileged installed controls at `8583d43b` | 231 passed, six explicit platform/control skips; final packaging changes also covered above |
| Final installed GPU | 29 passed; statistics/progress, mask and staging-lifetime oracles, strict synthetic lanes/two-rank/MoE and real-checkpoint four-layer serial/MTP equality |
| Final installed KV preparation/encoder/oracles | 1,666 passed, 3,430 deselected; one kernel-cache write warning retained in the full log |
| Final installed synthetic three-format exactness | RQ8-norm/RQ8/RQ6-norm reference `ALL 67a762f3ac80e419` unchanged |
| Broad CUDA non-matrix at `05d053c1` | 1,871 passed, 97 skipped, 41 failed, 45 errors; exact comparison with pristine `1a2836b0` has zero new failure/error IDs and five resolved cases |
| Installed runtime audit | 501 owned source files hash-identical; zero mismatches; three README-only omissions and native file-owner extension recorded |
| Harness at `410bbbf` | 323 affected tests passed, zero failures/skips; build and scoped formatting passed |

The broad CUDA gate's retained failures belong to existing numerical/family tests and a build-lock fixture; its 45 errors are old
invalid-vocabulary TP fixtures. This remains a failing broad suite, not an all-GPU pass. Detailed identities and
baseline comparisons are retained; no tests were disabled to change those results. Five old cases resolve with
isolated memory-limit execution and exclusive GPU capacity. Earlier unchanged passing coverage is reused only
with its actual source/configuration identity; the final affected checks qualify later fixes. The KV gate warned that its configured Torch disk-kernel cache directory could not be created. It completed
all selected correctness cases with disk caching disabled for that kernel path; it supplies no timing claim.

The ordinary PID-1 CLI served the actual local EXL3 Flash Next checkpoint with automatic RAM experts,
RQ8-norm KV, BF16 prompts, context 16,384, four slots, MTP-4/confidence 0.5, requests explicitly disabling thinking,
HTTP/pending caps 32/8, six CPUs, 56 decimal GB RAM/no swap and 2 GiB shared memory. It loaded in 142.4 s
and reserved all four slots and 24 snapshots within its startup plan. Models/list/retrieve/unknown-model 404,
native tokenization, Responses input counting, JSON/SSE usage and request IDs, progress, exact repeated/serial/drafted
token IDs, a forced strict call, unsupported-bound HTTP 400, the actual 31-tool Harness prompt (8,225 tokens),
four concurrent replies matching solo references, statistics and Prometheus all passed. Final logical owners were
zero and normal requests produced no failure captures. SIGINT/container retirement passed with exit zero and no OOM.
These bounded requests do not establish filled-context capacity or broad model quality.

TensorFold never weakens an active strict request. Unsupported strict schemas and active grammar plus
`ignore_eos: true` are refused before generation/SSE. The consumer change
[Harness `410bbbf`](https://github.com/kklouzal/Harness/commit/410bbbf8222c8f75e469b5a019dc88c712e54b66)
keeps every schema and execution validator, explicitly chooses non-strict decoding for 22 audited executable tools,
and retains strict decoding for nine foreground tools, all five private function submissions and four memory JSON
requests. Generation, counting and runtime probes share that choice; other providers retain their prior payloads.
The curator's `anyOf` replaces disjoint `oneOf` branches with identical accepted values (required, distinct action
constants). All actual exported requests pass the server validator; the full vocabulary/template and mixed-tool
stream checks pass. Deploy this consumer before the enforcing server: older Harness marks unsupported schemas strict.
No Harness deployment or provider-database change was performed here.

String-length candidates remain rejected: native fast JSON bounds do not preserve the required escape/control-character
domain; an exact escape-aware grammar took approximately 0.4 s per token and a raw-string DFA approximately 3.1 s per
mask in the recorded prototypes. Those candidates remain private evidence, not production paths. Finite uniqueness
for arrays over at most ten distinct string enum values uses the exact selected grammar. General unsupported
constraints still fail explicitly. No new decoder/kernel optimization is selected by this qualification.

The statistics readout experiment compares no background scraping against `/v1/stats` at 2 Hz plus `/metrics` at 1 Hz
on the same warm installed service, with the unchanged actual tool catalog and equal output token IDs.
One-request and four-request regions have separate, nonoverlapping paired ABBA/BAAB blocks, fixed prompts/seeds and a 16-token
maximum with normal EOS. The objective measured here is the geometric full-batch wall-latency ratio; it does not
measure the entire instrumentation change against an older release, nor claim arithmetic mean decode-round equivalence.
The initial 20-block pilots are retained separately. Confirmation uses 80 fresh blocks per region, fixed in advance,
97.5% Student-t intervals for a two-look alpha budget, and the larger of IID variance and a fixed Bartlett HAC
bandwidth-four variance (finite-sample correction included). The unchanged acceptance interval is [0.995, 1.005].
Order parity and lag correlations are reported; no observations are removed. The completed fresh-connection
confirmation is **unqualified**: n1 ratio 1.002389, 97.5% interval [0.999304, 1.005484]; n4 ratio 1.006104,
interval [1.001793, 1.010435]. Positive overhead is established in n4; neither interval proves the declared bound.
ABBA/BAAB point ratios are 1.002702/1.002076 (n1) and 1.008774/1.003442 (n4). Fixed lag-four HAC widens n4's
interval; it does not establish absence of longer-range dependence. Immediate polling at every short B-batch start
made realized n1 rates approximately 2.67/1.78 Hz (stats/metrics) and n4 2.16/1.08 Hz, above nominal timer rates.
All 640 measured batches (1,600 generation requests) completed with equal tokens; retirement passed exit zero/no OOM.

A separate, predeclared consumer-shaped steady-state region follows authoritative Harness source: shared stats-only
reads, at least two seconds after the preceding fetch completes, with one warmed reusable HTTP/1.1 connection
retained idle in A. The same fixed 80 blocks, margin, conservative interval method and quality settings apply.
This region excludes first connection, factory-handler rotation and native .NET reader execution. It cannot qualify
or erase the failed fresh-connection stress or the unrun source-instrumentation comparison. Its results are recorded
below. The closing validator independently checks sample order/counts, token identity, interval arithmetic,
connection count, completion spacing and the actual reader's two-second timeout.

The consumer-shaped n1 ratio is 0.9997535875, with 97.5% interval [0.9980131859, 1.0014970240]: it meets the
unchanged criterion. The n4 ratio is 0.9981978676, with interval [0.9947211719, 1.0016867149]: it remains
**inconclusive for two-sided equivalence**, since its lower edge is below 0.995. The upper edge is +0.169%, but
that is not a pass of the declared two-sided criterion. Neither region establishes a speedup. Order-parity point
ratios are 1.0002331189/0.9992742860 (n1) and 0.9989578556/0.9974384578 (n4); lag correlations remain in the
receipt. Exactly one monitoring connection recorded 446 post-setup reads (440 measured and six workload-warmup reads),
after two unrecorded setup reads. The recorded reads have minimum completion-to-next-start spacing
2.000028673 s and maximum duration 0.002016093 s. All 640 measured batches/1,600 generation requests
preserved their token sequences; the server retired exit zero/no OOM.

Local functional integration is verified; the general overhead qualification remains incomplete. Fresh-connection
stress shows positive n4 overhead, while the narrow consumer-shaped n4 interval lacks the precision required by
the symmetric margin. No additional samples, exclusions, changed margins, production optimization or rollout are
used to declare a pass. A future performance qualification needs a new predeclared study with enough power and
its required hardware/workload, preserving these outcomes.

Apple/MLX native execution, new ARM64/GB10 qualification, actual multi-GPU execution, exhaustive model quality and
large/full-context performance remain unrun for this source. Historical deployment and performance records below
retain their original source, images, hardware and workloads.

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
model/API and resource qualification. See the [README API qualification](../README.md#earlier-installed-cliapi-qualification).

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


## FP8 fragment packing candidate (2026-10-21)

The source candidate replaces `_fragment_order`'s expanded integer indices and
advanced-index gather with an exact mixed-radix view copied directly to the
final byte output. It retains 4,096-row chunks, output ownership, padding and
the earlier conservative admission bound. The inverse `fragment_index` remains
required by MXFP8 reconstruction and independent tests.

This is component evidence, not a complete-model loading claim. The comparison
used the original `d9bc2bc2` function and candidate module SHA256
`a19888976e496dc2f782b08e4d870f9485b05681c58e72001492092c0c1e913c`,
a local RTX PRO 2000 Blackwell SM120, pinned CUDA13.4.3/Torch
2.16.0.dev20261017+cu134/Triton3.9, two CPU workers and an8GiB container limit.
Twelve alternating paired blocks per shape measured synchronized complete calls
on128x64,128x5120,5120x5120,13696x5120 and248320x5120 byte matrices. Each pair
retained exact bytes; every block favored the candidate. Geometric component
latency ratios ranged5.50–7.93. At248320x5120, mean call time changed from
135.650 to24.584ms; additional peak allocated storage changed from1,649,344,512
to1,313,341,440bytes, including the1,271,398,400-byte output, with zero retained
allocation after release. An untimed5120x5120 profile observed74 versus2 GPU
kernels. Before/after source hashes were identical during measurement.

The focused layout/accounting gate passed76 cases. A frozen integrated source
consumer gate subsequently passed204 cases with9 explicit missing-checkpoint
skips, including the new exact-bit prompt controls; this is local source-mounted
execution, not a new installed ARM64 result. An earlier serial cold compile
timed out, and a concurrent-edit run mixed two prefill revisions; both failed
attempts remain excluded and preserved. Complete GB10 startup, final installed
images and same-region integrated performance selection remain pending. This
candidate has not established a universal or whole-model speedup.
