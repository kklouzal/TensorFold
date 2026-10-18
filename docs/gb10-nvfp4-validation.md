# GB10 Flash Next NVFP4 validation, 2026-10-18

Installed source `bd1f0e9d` passed four full-checkpoint protocols and a separate
public CLI/strict-tool check on one NVIDIA GB10 (SM12.1, native ARM64). The model
is [`dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4`](https://huggingface.co/dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4)
at revision `be794b990578ef3031eccf9f28e675a289a09ee9`. The
[machine-readable receipt](verification-gb10-nvfp4-20261018.json) binds the exact
source, images, model metadata, raw evidence hashes and remaining gates.

These results qualify that installed source and workload. The later composition
adds loading read plans and AUTO graph budgeting; its fresh installed/model gates
and performance selection remain pending. This record does not replace a deployed
service or establish a desktop, FreeToken, two-GPU or broad quality comparison.

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

Final composed-source installed CPU/native/model gates and a controlled
same-format performance comparison remain required. The entire fresh
RotorQuant/KV matrix, long-context quality/capacity, actual two-GPU or
heterogeneous expert ranks and deployment rollout are not established here.
Earlier [Python integration](optimization-validation.md#python-line-integration-2026-10-11),
[desktop/NCCL](desktop-report-remediation.md) and
[NVFP4 loader](nvfp4-loading-validation.md) records retain their own sources,
hardware and limitations.
