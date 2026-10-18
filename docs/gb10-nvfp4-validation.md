# GB10 Flash Next NVFP4 validation, 2026-10-18

Installed source `bd1f0e9d` passed four full-checkpoint protocols and a separate
public CLI/strict-tool check on one NVIDIA GB10 (SM12.1, native ARM64). The model
is [`dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4`](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4)
at revision `be794b990578ef3031eccf9f28e675a289a09ee9`. The
[machine-readable receipt](verification-gb10-nvfp4-20261018.json) binds the exact
source, images, model metadata, raw evidence hashes and remaining gates.

The later `297e8487` composition also passed two installed AUTO graph model
protocols and 580 focused CPU checks, recorded [separately below](#source297-auto-graph-protocols).
Its nine-case native fixture follow-up also passed. The user subsequently selected
[profile A for measured four-stream throughput](#controlled-eight-start-comparison-and-user-selection).
Each result qualifies its recorded source and workload. This record
does not replace a deployed service or establish a desktop, FreeToken, two-GPU
or broad quality comparison.

## Source and protected foundation

| Artifact | Identity |
| --- | --- |
| Source | `bd1f0e9d624183b341bc615adefc0c308ad88aa5` |
| Source UID | `c56ce6b8860af67d8f37d02e85328f52400457d3bd6a6e9c7a0e726b43c4b645` |
| Build input UID | `a5a6aa47e3a0e7c2b56f25c0d602e8e3ac49c572a34cbbb3de731d4381fd19d6` |
| Runtime image | `sha256:0ee2b4628e9b2eba7f1458fff80105a7b658d3f3ee7752f91f5592a55f1472d6` |
| Verification image used for protocols | `sha256:82f58da8dadf265a6a753537b78c9d6504613dfabfd616fd835c3e88b6d4b7d5` |

The protected ARM foundation uses official NGC CUDA 13.4.3 cuDNN development,
CPython 3.12.3, official Torch `2.16.0.dev20261014+cu134` (git `ef46daf1`),
TorchVision `0.30.0.dev20261006+cu134` and Triton `3.9.0+gitfc8d31a5`.
Native NCCL reports 23203; NVCC is 13.4.92. The model runs admitted the sealed
forward-compatibility driver `libcuda.so.615.71.09`, API 13040, on this host.
That is actual device admission, not eligibility inferred from a CUDA label.

The [managed foundation](../deploy/gb10/foundation.md) protects the Python closure,
SDK, compiler-owned C++ runtimes and selected native providers. Supported Torch
RTLD/preloads and Triton tool/header/libdevice controls select the core SDK.
Exact protected official TorchVision CUDART/NVJPEG members and cuda-bindings'
static CUDART are authorized bundled-runtime exceptions. Upstream dependency
metadata is intact; no vendor patch, Torch source build or runtime alias shim is
used. Each model protocol's same-process post-consumer proof passed with
22 native-foundation and two official-Vision-bundle physical objects recorded.

## Workload and observed rates

All four profiles used 48 resident ModelOpt routed banks, `--precision checkpoint`,
`--kv-dtype rotorquant-iso4`, `--parallel 4`, `--ple-on-ssd` and `--vision`.
Native MTP was disabled or set to three/six proposals as shown below. The export's
declared static group-16 W4A4 uses TensorFold's explicit FP32 round-to-nearest
[native arithmetic](cuda-kernel-contracts.md); shared/backbone/MTP tensors keep
their separately declared representations. Equality below is within this math
and KV format, not against full precision or another runtime's quantizer.

The fixed LRU-code prompt used temperature 0, seed 7, thinking off and exactly 512
generated tokens. Each profile ran a first request, two warm requests and a
barrier-started quartet of 512-token replies (2048 total). MTP profiles also ran
their own draft-disabled reference and compared against the independent MTP0
run. Every complete token array matched; the recorded SHA256 prefix is
`144dec81bb0f`.

| Native MTP proposals | Graph cap (MiB, bd1) | Captured verify/MTP keys | Warm single-request HTTP tok/s | Four-request aggregate HTTP tok/s |
| --- | --- | --- | --- | --- |
| 0 | 512 | 8 / 0 | 17.91–17.92 | 55.76 |
| 3 | 512 | 39 / 19 | 41.22–41.31 | 124.20 |
| 3 | 2048 | 68 / 34 | 41.30–41.31 | 141.50 |
| 6 | 4096 | 128 / 64 | 43.27–43.50 | 143.80 |

These are completed HTTP-body observations: client preparation/open through
body EOF; the quartet interval includes barrier/executor joins. Final quiet,
provider proof and engine retirement follow outside that timer. The 512 MiB MTP3
profile captured 58 of 102 possible keys; 2048 MiB captured all 102. The 4096 MiB MTP6
profile captured all 192. The first two profiles used bd1's default 512 MiB cap;
2048/4096 MiB were explicit `TENSORFOLD_MULTI_GRAPH_MIB` overrides. These are
graph-memory caps, not context lengths or selection of the later AUTO default.

MTP6 had the highest observed rates among these tested configurations. One
sequential observation per quartet and two warm requests do not establish a
balanced repeated performance comparison, uncertainty bounds or sustained
capacity. Constructor observations ranged 234.9–278.0 s under their recorded cache
conditions; they are not a controlled loading-speed comparison. The admitted
262,144-token ceiling does not qualify a filled window or four simultaneous full
windows: these checks used a short prompt and 512-token outputs.

## Model, API and resource checks

Each profile passed image semantics (red), video semantics (first red, last blue)
and exact media token comparison with its serial path. Cancellation was observed,
then recovery completed with the same pre-cancel reference tokens. The actual
FP8 n-gram `EncodedSSDTable` matched all 256 code values and sampled real rows at
the nonunit BF16 scale `0x3951` (`0.00019931793212890625`). Reads were bounded
buffered positional I/O; OS page-cache hits are possible. This is no physical-SSD
bandwidth or `O_DIRECT` claim.

The separate installed public CLI run used literal `serve`, `--backend cuda`,
`--precision checkpoint`, `--ple-on-ssd`, `--vision`, `--parallel 4`, `--context 0`,
`--kv-dtype rotorquant-iso4`, `--mtp-drafts 6`, `--mtp-confidence 0.70`,
`--drafter none` and `--thinking-budget 0`. It reproduced the 512-token LRU output
and media results. Required strict function calls preserved typed `true`, `false`
and `null` through ordinary and SSE responses, completed with `tool_calls`, and
drained the streaming body. Parent and serving-child post-consumer provider
proofs passed; the child exited 0. This is enforcement/output validation, not
execution of a generated tool or a speed comparison.

The owned containers used all 20 eligible logical CPUs without Docker
RAM/swap/CPU/PID quotas, with actual ancestor limits inspected. A sampled 8 GiB
host-MemAvailable floor was enforced. The lowest model-protocol sample was
23,300,030,464B; the CLI minimum was 23,752,450,048B. Sampling is not a continuous
reservation. All protocol handlers drained, loops stopped, engines closed and
outer ownership/observer/cleanup checks passed.

## Composite checks and remaining scope

The bd1 A4/large-bank extent selection passed 38 native cases in 115.37 s. The
predecessor `1e04aa92` passed 29 A4 cases under Compute Sanitizer memcheck with
zero errors. These overlap and are not added into a total. Its separate
219-case CPU selection passed with one expected CUDA-only skip, including the
five fixtures that failed the earlier broad `97018237` run (9398 passed,
183 skipped, five failed). That original broad result remains failed; the
focused repair is not a fresh clean full-suite run. The mixed predecessor
native fixture failures and initial full-model calibration extent failure are
also retained in the receipt.

The later composition's results are recorded separately below rather than
inferred from these earlier gates. A controlled same-format performance
comparison is recorded below. The entire fresh
RotorQuant/KV matrix, long-context quality/capacity, actual two-GPU or
heterogeneous expert ranks and deployment rollout are not established here.
Earlier [Python integration](optimization-validation.md#python-line-integration-2026-10-11),
[desktop/NCCL](desktop-report-remediation.md) and
[NVFP4 loader](nvfp4-loading-validation.md) records retain their own sources,
hardware and limitations.

## Source297 AUTO graph protocols

Source `297e8487bc44e941d0fffeac753ac465c72165ce` includes the loading read plan
and automatic graph sizing. Its native ARM runtime/verification builds and
installed audits passed; both audits matched 536 package files and the project
wheel SHA256 `7277e7ef3dbd0b53b021dae6f4efd6e0ad9de5c0fa933c4a92c695a70ce4d3d4`.
The distinct source UID is
`7f2fd04f3576e75595d2fa0f745e77b6a3eea0074fc233725dab83c957460f16`;
build input UID is
`6a91d772d7d87fe5ae0d038ab6cb0a57613361a5070a6a1ac3ea5350fbdadf90`.

| Artifact | Image |
| --- | --- |
| Runtime | `sha256:3645d51c532ed42606facc09d86e3b808eec9e0b400f615f9ed23329cb6bc366` |
| Verification used for model protocols | `sha256:310abf1b44f4589b22b985e86eaf41f47c8b2ec4249a1745961da321a3ae0377` |

Two fresh installed model starts used the same checkpoint, precision, four
slots, RotorQuant-iso4, SSD tables, vision, short LRU prompt and 512-token protocol
as above. `TENSORFOLD_MULTI_GRAPH_MIB` was unset: actual records report
`automatic=true`, using native-pool inactive-memory accounting. The MTP0 run
captured eight verify keys; MTP6 captured all 128 verify plus 64 MTP keys.
The MTP6 record reports a 17,988,594,142 B allowance derived by runtime admission
and an observed 3,109,662,720 B capture charge, after preserving its future
serving plan. These run-specific values are neither a fixed default nor a
reservation against other processes.

| Native MTP proposals | Warm single-request HTTP tok/s | Four-request aggregate HTTP tok/s | Lowest sampled host MemAvailable |
| --- | --- | --- | --- |
| 0, AUTO | 18.0356–18.0614 | 59.2309 | 40,239,493,120 B |
| 6, AUTO | 43.2464–43.3762 | 143.4139 | 27,453,063,168 B |

The HTTP interval has the same body-EOF and final-retirement exclusions as the
bd1 table. Cache state is explicit: the MTP6 warm requests and quartet report
45 cached tokens of a 46-token prompt; MTP0 reports zero. The earlier bd1
fixed-budget observations retain their own cache conditions. These are not
uniform-prefix comparisons or a performance selection. Constructor observations
were 197.374 s (MTP0) and 191.381 s (MTP6); the separate starts/cache conditions
do not establish a controlled loading improvement over bd1 or each other.

All complete 512-token arrays matched within and between these serial/MTP runs
(hash prefix `144dec81bb0f`). Image/video semantic and serial-token checks,
nonunit-scale SSD decoding, cancellation/recovery, same-process post-consumer
provider checks and complete handler/loop/engine retirement passed. Both runs
remained above the sampled 8 GiB host-free floor. These are actual model and
AUTO/default correctness checks for this source, not inference from CPU mocks.

The installed 20-module focused CPU gate passed 580 cases with one CUDA-only
skip in 5.28 s. The original eight-module native gate remains **failed**:
151 passed and one failed in 292.57 s. Its copy-draft test compared a fixed-count
serial path with EOS disabled against a concurrent path using the EOS default;
their prefix agreed through EOS. A separate read-only test overlay at
`9a4b0e49a4f003599106810ddd141a56e67b3557`, running the installed 297 application,
then passed nine cases in 28.390 s with no failures, errors, skips or warnings.
It covers the repaired node, three graph parameter cases, four unchanged
ignore-EOS controls and the scalar-authority case skipped by the CPU gate.
Only the fixed-count graph test module changed: all 55 existing assertions
remain, with three length/error checks added. The other two overlay files are
unchanged, and no application/build bytes changed. Provider/retirement checks
passed, with minimum sampled MemAvailable of 122,127,585,280 B. This is a
composite follow-up, not a clean rerun of all 152 native cases; overlapping
counts are not added. The original failed run remains failed in the receipt.

The [receipt's `source297_auto` entry](verification-gb10-nvfp4-20261018.json)
binds these actual source/images, graph accounting, cache usage, raw logs,
provider/outer receipts and remaining scope. The clean full-project CPU rerun,
complete fresh RotorQuant/KV matrix,
long-context capacity/quality, two GPUs and rollout remain separate.

## Controlled eight-start comparison and user selection

The user selected **A** for the best measured four-stream throughput among the
four tested configurations, and stopped further candidate work. This is a scoped
operating choice, not an all-region SUPERIOR or equivalence claim. E/F were not
promoted; their partial work supplies no completed model/performance result here.
Deployment and its live receipt remain separate.

Eight fresh installed-container starts ran in predeclared order **A B C D D C B A**,
with two starts per configuration. All used the same source297 verification
image, pinned checkpoint, exact math/KV, MTP6, four slots, AUTO graphs, SSD tables,
vision and full 512-token protocol. A/B/C prepared common-input banks; D used the
general-input path. B used four Torch intraop threads during construction;
all serving restored 20 threads. C changed only the original read-ahead queue
policy on the current source; early metadata discovery remained common.

| Profile | Loading / constructor threads / reads / input policy | Constructor s, first / second | Warm R1 mean HTTP tok/s, first / second | R4 aggregate HTTP tok/s, first / second |
| --- | --- | --- | --- | --- |
| **A, selected** | GPU / 20 / split / common | 188.636 / 187.611 | 43.3203 / 43.2399 | **144.2591 / 144.1688** |
| B | CPU / 4 / split / common | 284.294 / 280.050 | 43.3377 / 43.5270 | 144.0446 / 143.4509 |
| C | GPU / 20 / original queue / common | 236.307 / 235.656 | 43.2160 / 43.2185 | 143.3265 / 142.8198 |
| D | GPU / 20 / split / general | 187.289 / 188.186 | 43.3472 / 43.4327 | 143.4303 / 143.1909 |

Each R1 number is the arithmetic mean of two warm requests in that start;
those requests are not independent process replicates. Each R4 observation is
one completed 2048-token quartet. The raw individual values, first-request rates
and paired differences are retained in the receipt's `controlled_abcd` entry.

A's two R4 observations exceeded every other measured R4 observation. Its
constructor was shorter than B by 95.659/92.439 s (33.648%/33.008%) and C by
47.671/48.045 s (20.173%/20.388%) in the respective halves. A/D startup ordering
reversed between halves. D's warm R1 mean exceeded A by 0.0268/0.1929 tok/s;
B also exceeded A in both halves. A exceeded D in R4 by 0.8288/0.9779 tok/s.
The user chose the four-stream result while retaining these single-request
observations; startup or R4 wins do not erase them.

All 64 complete 512-token arrays matched the unwrapped source297 MTP0 reference,
including each run's own draft-disabled control. Every warm/quartet response
reported 45 cached tokens of 46; first and serial-control requests reported 0.
Persistent native/compiler caches were retained and OS file caches were not
globally dropped. These are fresh model/container starts, not cold-host or
cold-storage measurements. Constructor time includes model preparation and graph
capture, while HTTP rates retain the body-EOF interval and exclude final
quiet/provider/engine retirement. All separate quality, media, SSD-scale,
cancellation/recovery, provider, floor and retirement checks passed.

Every start captured 128 verify and 64 MTP keys. All had 20 eligible CPUs and
unlimited live ancestor CPU/RAM/PID limits; the lowest sampled MemAvailable was
23,539,601,408 B, above the 8 GiB floor. Whole-run GPU monitoring is retained,
without phase-specific thermal/clock attribution. Two starts per configuration
and a mirrored nonrandom order support these repeated observations, not a
population confidence bound, sustained-capacity guarantee or formal equivalence.
No new desktop, FreeToken, filled-context, two-GPU or broad-quality result follows.

The release-integration proof at `e82aeae1` matches runtime/build/lock/tool source
bytes to source297; changes are docs and the recorded test-only EOS fixture.
It does not claim identical rebuilt wheels or final image IDs. The final
release/deployment receipts must record those actual artifacts.
