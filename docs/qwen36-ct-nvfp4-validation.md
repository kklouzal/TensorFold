# Qwen3.6 CT NVFP4 qualification, 2026-10-17

The original `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` export and its separate
DFlash v1 model completed installed-container qualification on the single
RTX PRO 2000 Blackwell host. This record applies to application source
`099eb4d0233a6de883d8f2de76b96cec6accd578`. It combines source-scoped reuse of
earlier checks with fresh affected CPU/native, full-model API and completed
54-request performance checks. It does not claim a complete rerun of every
test at this source.

The [machine-readable receipt](verification-qwen36-ct-nvfp4-20261017.json)
retains exact source, model, image, request, resource and capture identities.
Later loader changes have a separate qualification boundary described below;
they do not inherit these measurements automatically.

## Inputs and installed artifacts

| Input | Pinned identity |
| --- | --- |
| Target | `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4@3d05c6063e82cdae8dd1b9362116ea71be8d1439` |
| Target weight file | 23,354,242,448 bytes; SHA-256 `c9301dfd497b6d2bd11045897bb56aadc5e5de72ee8fbdad1a4be28dfeafc797` |
| Secondary | `z-lab/Qwen3.6-35B-A3B-DFlash@42d3b34d588423cdae7ba8f53a8cf7789346a719` |
| Secondary weight file | 948,000,184 bytes; SHA-256 `6db5c712b4f3d924026162ad1aedf7fd1fef32437690451137f967d9b7160144` |
| Published chat template | SHA-256 `55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c` |
| EOS | `[248044, 248046]`, read through normal checkpoint metadata |

All target and secondary file hashes come from prior publisher-matched,
independent post-transfer verification. The secondary also passed independent
normal and optimized-Python verification before atomic publication. Files were
mounted read-only; neither checkpoint was rewritten. These receipts were reused
without rehashing the large weights during evidence assembly. TensorFold loads
the secondary through its own DFlash v1 block-diffusion implementation; it does
not import checkpoint Python or reinterpret it as native MTP/dense DFlash2.

The immutable local images are:

- Runtime: `sha256:4337064296a99de5eadfc82ff2029112efc9268fe1f26cfb7399191582d82657`.
- Verification: `sha256:907dddcad1b934fc66da2b22114fa701879a699679f888278985aed180deda97`.
- Wheel SHA-256: `c3dc9449018b7597bdf28baae7c8aa30561427919302135f78b481794a16f15b`.
- Source UID: `a9f5b9dee32129ec444ee09900503908e3f08ecad29b320786c6abb3fa7ee025`.

Both installed audits matched 530 package files and the same native file-owner
extension. The verification image adds test inputs; its source-bound checks
used the installed wheel, without a source overlay. Pytest explicitly used
`-o pythonpath=tests` to override the repository's `src,tests` default. The
completed formal comparison ran in the runtime image. These IDs identify the
tested local artifacts, rather than a floating image tag or a new-build pass.

The host GPU was an NVIDIA RTX PRO 2000 Blackwell, SM 12.0, with
16,584,343,552 bytes reported by CUDA. The stack was Python 3.12.3, TensorFold
0.6.6, PyTorch `2.16.0.dev20261006+cu134`, Triton distribution
`3.9.0+gitaad2a60d`, CUDA `13.4.1.012`, xgrammar 0.2.8, tokenizers 0.23.1,
Transformers 5.17.0 and NCCL 2.30.7. CUDA forward compatibility used driver
615.71.09 over kernel driver 595.91.07. The receipt includes the pinned NGC
base digest and the complete recorded dependency versions.

## Numerical and serving contract

The target retains original CT packed FP4 weights, block-scale bytes and
projection-specific FP32 calibrations. Activation quantization consumes the
original input-global multiplier directly: a rounded reciprocal round trip
can change FP8 scale bytes. Output calibration preserves the separately
rounded FP32 reciprocals and their product. Excluded BF16 projections remain
BF16; routed/shared SwiGLU retains the BF16 SiLU result and subsequent BF16
product before the down projection.

Exactness here means the same target math/build and keyed sampling rule across
the tested serial, drafted and batched full-model runs, all using RAM-backed
experts. Resident-versus-cached comparisons are native component-bank checks;
the actual 35B model cannot fit fully resident on this card. Native checks
include independent checkpoint/activation-scale oracles, odd strides,
eviction, direct pinned uploads, replacement failures and retirement. This is
neither a cross-runtime bit-identity promise nor a general answer-quality
benchmark. The 512-token cap truncates the coding response; it does not prove
that the generated module is complete or correct Python.

The single-request comparison used CUDA checkpoint precision, context 4,096,
one stream, eight draft rows, BF16 secondary linears, `--vram-experts auto`
and `--vram-experts-pin on`. Host-backed A4 execution was eager, not CUDA-graph
capturable. The expert pool stayed fixed at 4,424 cells / 7,828,427,264 bytes
(7.290791 GiB), with 18,120,048,640 bytes of host authority. Its future KV,
retained states, target/draft scratch and allocator bounds were admitted before
pool sizing. No alternate KV codec was requested.

Equivalent public serving settings are:

```bash
TENSORFOLD_MEMORY_RESERVE_GIB=8 TENSORFOLD_CUDA_MEMORY_RESERVE_GIB=0 \
tensorfold serve /models/heretic-nvfp4 --backend cuda --name heretic \
  --precision checkpoint --drafter /models/qwen36-dflash --parallel 1 \
  --context 4096 --draft-rows 8 --drafter-bits 16 \
  --vram-experts auto --vram-experts-pin on
```

Requests explicitly disabled thinking. Context/eight rows/pinning on are
recorded operator choices, not changed defaults: draft width still defaults
to the secondary's metadata block size, and registration defaults to `off`.
The CLI example alone does not reproduce the external qualification guard or
comparison instrumentation. Normal resource admission still applies.

Qualification used no Docker RAM, swap, CPU or PID quota. The complete live
ancestor cgroup memory/CPU/PID limits and recorded Docker memlock/stack limits
were unlimited. These configuration and identity-bound cgroup checks preceded
release of a readiness barrier; unrelated OS/global limits are not claimed.
GPU model runs used the recorded 8 GiB host-admission reserve and zero explicit
CUDA reserve, with no absolute CUDA-memory cap. An external guard sampled
MemAvailable every 250 ms against an 8 GiB floor. Shared memory was only a
per-run ceiling derived from initial availability. These are sampled bounds,
not continuous reservations against unrelated host allocations. CPU checks
omitted inference reserve overrides and used one-thread OMP/BLAS settings,
without an OS CPU quota.

All named completed runs closed their owned containers without OOM or cleanup
errors. The lowest current-source sampled MemAvailable was 34,331,910,144
bytes in the API run; the formal run's minimum was 34,752,368,640 bytes. Jobs
were serialized; compiler caches were reused and GPU clocks were recorded,
not locked.

## Completed test and API coverage

| Source/image scope | Result | Qualification boundary |
| --- | --- | --- |
| Earlier source `240c72eb`, verification `074a5f6a…` | 8,989 CPU passes; 187 skips; 2 inherited warnings | Historical full installed CPU run |
| Same earlier artifact | 238 selected native/coupled passes; 2 skips; 28 inherited warnings | Includes 13 CPU head-policy controls; not the full CUDA suite |
| Fresh 099 / verification 907 | 1,014 + 43 + 118 = **1,175 application CPU passes** | Three disjoint groups of 40, 3 and 4 modules |
| Same coupled run | **18 private portable comparison controls** | Separate from application counts |
| Fresh 099 / verification 907 | **113 selected native/coupled passes**, 1 skip | Includes the same 13 head CPU controls; do not add them again |
| Pretrained dense admission selection | **0 passes, 2 skips** | Missing dense 27B/DFlash2 inputs; no coverage from the outer successful exit |

The source-scoped composite reuse was reviewed independently. Between 240 and
099 only three application Python files changed: shared batched-prefill
dispatch/bounds and the MoE head table. CUDA/C++, build and dependency inputs
were unchanged. Fresh installed checks cover the affected/coupled behavior.
Counts across sources or overlapping gates are never summed into a supposed
full 099 suite. Fresh static checks compiled 1,148 Python files and introduced
no relevant Ruff diagnostic relative to 240; existing advisory dispositions
remain recorded.

The actual model API passed tokenization/detokenization, prefix reuse, strict
JSON object/schema output, strict typed function calls, ordinary/SSE completion,
intentional cancellation/recovery and synchronized clients. Scalar booleans/null
and literal strings `True`/`None` were checked in ordinary and streamed calls
using the unchanged published XML template. Prefix reuse reported 1,605 cached
tokens. Expected cancellation is retained in statistics; failed requests stayed
zero at the end.

A separate four-slot, four-draft-row, BF16/pinned profile used a 2,447-cell
(4.032678 GiB) expert pool. Its serial 512-token reply and four barrier-started
512-token replies matched complete token arrays. Actual successful DFlash
activity increased by 160 batches / 640 contexts for the quartet, versus zero
for its serial reference. Quartet statistics reconciled four requests, 2,048
tokens, 640 rounds, 1,856 drafts and 1,404 accepted drafts. Handlers drained,
the loop stopped and the engine closed.

Those four clients observed **57.6985074033 aggregate client tok/s** over
**35.4948523310 seconds**, including HTTP/executor setup and completion. This
interval was not explicitly CUDA-drained and is not the single-request region
below. Memory-wait/stop counters were null, so no zero-wait/zero-stop claim is
made.

## Completed single-request comparison

Every timed request used the public LRU Python-module prompt, temperature 0,
seed 7, top-k 0, top-p 1, thinking off and 512 actual completion tokens with
finish reason `length`. Seeded 32-token checks used temperature 0.7, top-k 20,
top-p 0.95 and seed 47. Greedy and seeded serial/drafted/repeated arrays matched.
The engine's 12-character SHA prefixes were `ffb65cfd59e8` and `b45d2d117ddf`;
the receipt retains full SHA-256 values and their comma-joined integer encoding.

The completed predeclared design ran **54 timed requests in nine symmetric,
rotating ABC-CBA blocks**, two requests per policy per block, after one warmup
per policy. All policies used the same engine, admission, expert pool and BF16
head weight storage. Plain used the incumbent vocabulary head; prepared R4/R8
changed only reuse at head row counts seven/eight. Policy changes occurred only
at proved HTTP/engine quiescence with a device fence, and original ownership and
tables were restored before shutdown.

| Head policy | Median decode tok/s, 18 requests | Median complete-request tok/s, 18 requests | Median block-geometric decode tok/s |
| --- | ---: | ---: | ---: |
| Plain R1 | 28.51960905 | 28.39775161 | 28.49554770 |
| Prepared R4 | 35.70928019 | 35.51650759 | 35.71974234 |
| Prepared R8 | **37.45293971** | **37.25144125** | **37.42209214** |

Decode rates count all 512 accepted tokens over the engine's decode interval,
rather than the rounded server `(n-1)` rate. Complete-request rates include
ordinary preparation, HTTP response consumption and an explicit CUDA drain.
The predeclared objective orders the median within-block geometric decode rate
first, then the corresponding complete-request rate. Individual-request medians
above are descriptive summaries, not the paired estimator.

| Paired comparison | Median block decode ratio | Observed nine-block range / conditional median interval | Median block complete-request ratio |
| --- | ---: | --- | ---: |
| R4 / Plain | 1.25237376 | 1.24809875–1.25822816 | 1.25096830 |
| R8 / Plain | 1.31098691 | 1.30932430–1.32220726 | 1.30973378 |
| R8 / R4 | 1.04708274 | 1.04346276–1.05535683 | 1.04697661 |

R8 exceeded R4, and both exceeded Plain, in every recorded block on both
objectives. Round/work signatures stayed fixed at 119 rounds, 392 accepted and
833 drafted tokens per timed drafted request. The source retained R8 as the best
observed repeated local ordering in this width-eight region.

Superseded (2026-10-18, NVFP4 campaign): the head-only, SM 12.0-only prepared
reuse is removed. The shared `b16.cu` row-block policy now gives every BF16
projection on `b16.matmul` (GDN projections, vocabulary head, DFlash drafter
linears; every family) row reuse at 2 through 511 rows on any GPU. Row reuse
never changes a row's bits, so replies are unchanged.

The exact order-statistic median intervals allocate nominal family error across
three pairs and two metrics. Their interpretation assumes independent,
representative block effects. Those assumptions were not proved by balancing,
lag/drift checks or clock observations. For nine blocks these interval bounds
equal the observed minima/maxima; they are not a promise that future ratios
stay within them. The helper's literal classification remains
`inconclusive for selection; independence/model assumptions require external validation`.
Separately, retaining R8 rests on the observed finite-region
ordering above. No unconditional confidence, global optimum, equivalence
or outside-region superiority is asserted. In particular, this experiment
does not resolve the earlier four-row R4/R8 ordering or BF16-versus-four-bit
secondary speed ordering.

The fresh no-secondary/copy-only 512-token check used a larger 5,714-cell
(9.417 GiB) pool. Its two warm complete-token decode observations were
17.04801073 and 17.27463086 tok/s, with a same-engine strict-serial observation
of 17.13218289 tok/s. All seven greedy/seeded replies passed. This separate
constructor is a complete-policy reference, not a fixed-pool paired head
comparison; no uncertainty or causal speedup is inferred from its two warm
samples. Earlier complete width-two/four/eight/sixteen and four-bit proposal
screens retain their own source/image identities in the receipt.

A separate completed 099 CUPTI diagnostic captured the third actual generate
call: 32 seeded tokens, eight draft rows and pinning on. It contained 24,830
CUDA kernel events and 12 actual DFlash forwards. Mixed GDN/draft R1 BF16 work
accounted for 43.00% of summed CUDA activity durations, pinned H2D copies for
18.94%, and R8 vocabulary heads for 10.65%. These activity sums can overlap;
they are not wall-time shares or throughput. The R1 family mixes several
projection roles, and aggregate H2D cannot be assigned entirely to experts.
Different block/seed/cache work prevents a causal comparison with earlier
traces. The profile identifies further candidates without qualifying them.

## Limits and later increments

The missing pretrained dense admission cases and second compatible GPU remain
unrun. This record establishes no two-GPU NCCL or mixed-architecture result,
fresh Apple/ARM64 execution, vision/video quality, filled 262,144-token capacity,
or other-hardware performance. Compute Sanitizer was unavailable in the recorded
image capability checks; no memcheck pass is claimed. Existing MLX and absent
dense-tokenizer skips are listed in the receipt. Historical failed attempts,
including the earlier batched-MoE prefill refusal, remain preserved separately
from the successful 099 qualification.

Any subsequent GPU-loader preparation increment is **separate and unqualified
by this record**. New source/image, numerical, startup, resource and performance
results must be recorded under their own identities. Retain the completed 099
measurements above when adding later results; they do not qualify a later
artifact merely because its public options are unchanged.
