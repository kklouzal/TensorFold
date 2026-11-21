# CPython 3.15.2 runtime (V100)

CPython 3.15.2 is the project's only supported runtime. The audited
[source foundation](source-foundation.md) builds the V100 image on it: the builder
installs the pinned standalone archive at `/opt/python3.15`, every closure is
resolved on CPython 3.15, and packages without a usable cp315 wheel are owned
source builds. This page records why each workaround exists and the evidence; the
first measurements below used an earlier experimental candidate image, and the
pipeline-built image is verified separately (see the ledger and the README).

## Runtime and dependencies

| Component | previous 3.12 image (`cf8c0ff6`) | CPython 3.15.2 image |
| --- | --- | --- |
| CPython | Ubuntu 24.04 apt 3.12.3 | python-build-standalone `20261110` `cpython-3.15.2` `x86_64-unknown-linux-gnu-install_only_stripped` (sha256 `ad23d1b73fac26440b0306ae06911031344c49a90ec19716c8362e7fa40cae43`), installed unchanged at `/opt/python3.15` |
| PyTorch | owned source `05adb434` for cp312 | same owned source and `source_build.sh` framework flags, rebuilt for cp315 |
| Triton | PyPI `3.2.0` cp312 wheel | `3.2.0` source build for cp315 with [the owned patch](framework-patches/triton-3.2.0/0001-python315-compatibility.patch) |
| pyyaml 6.0.3, xgrammar 0.2.9 | PyPI wheels | built from their PyPI sdists (no cp315 wheels) |
| Other runtime packages | locked versions | same versions, cp315/abi3/py3 PyPI wheels |
| Pillow (image loader) | not in the application image | `13.0.0` official cp315 wheel, hash-pinned, added by `Dockerfile.pillow` (see the [source foundation](source-foundation.md#whole-application-wheel-updates)) |

Why each workaround exists:

- Triton 3.2.0 is the SM70 (MMAv1) compiler profile and publishes wheels only
  through cp313. Its code generator uses `ast.Num` (removed in 3.14), builds
  `ast.AnnAssign` without the required `simple` field (a `TypeError` in 3.15)
  and reads module annotations from globals (PEP 649 stores `__annotate__`).
  Its pybind translation units include LLVM/libstdc++ headers before `Python.h`;
  3.15's `pyconfig.h` selects POSIX.1-2024 feature macros, so the redefinition is
  fatal under Triton's `-Werror`. The patch fixes the three Python sites and puts
  `Python.h` first. A CPU compile of range loops, a constexpr default and an
  annotated module constexpr for SM70 produced PTX and cubin identical to the
  official cp312 wheel.
- xgrammar's tvm-ffi bindings compile as C++20 and its core as C++17; GCC 13
  LTO then reports `-Wodr` for `std::variant`, which upstream's `-Werror` makes
  fatal. The candidate builds it with `CXXFLAGS=-Wno-error=odr` (warning kept).
- The standalone interpreter is built against old kernel headers, so `fcntl`
  lacks the memfd seal names; the media frontend uses the Linux seal ABI values.
  It is also the only 3.15 build here with the experimental JIT available
  (`PYTHON_JIT=1`), the tail-calling interpreter, PGO, LTO and BOLT.

## Source contract changes

- `requires-python` is `>=3.15` and the FD owner targets the 3.15 Limited API.
- `retention_resources.py` binds its payload facts to the standalone 3.15.2 build
  (re-derived from its sources and installed files); other versions and
  free-threaded builds are refused.
- CPython 3.14 replaced the compiled C recursion counter (10000) with C-stack
  checks. `tensorfold.json_nesting.loads` keeps the 3.12-era refusal of JSON
  nested deeper than 10000 at HTTP and TP-frame boundaries, and
  `json_wire_bound` refuses integers beyond the 4300-digit limit before any
  lease. With these, 3.15 host-payload projections equal 3.12's except
  +2.62 GB of fixed TP RPC obligation (streamed RPC publisher failure notes).
- The stable-ABI FD owner is named `_fd_owner.abi3-x86_64-linux-gnu.so` on 3.15.

## Building the image

Use the source foundation's build stages; the owned builds run in its
`owned-wheels` phase. The serving image chain is SDK, `sm70-triton32-v1` runtime
profile, `Dockerfile.pillow`, then `Dockerfile.application-wheel`. The first measurements below came from an experimental
image assembled outside the pipeline (FROM the 3.12 image, separate caches); that
recipe is superseded and not maintained.

## Measured evidence

8×V100, SC117 IQ3_S + Q8_0 MTP drafter, TP8/ETP4, MTP4 confidence 0.7, INT8 KV,
`--context 262144 --parallel 32 --kv-full-windows 32`, manifest `m-profile7h32`, greedy,
512 output tokens, source `53d6da37` overlaid on both images, one GPU-lock hold per
session in A B B A order (A = 3.12.3 image `cf8c0ff6`, B = candidate
`sha256:27588e4931dbaf243ebdf3782f7a89381bc63a1aa7a31a49e7b503b4d67b380b`).

| Cell | Prefill tok/s A → B | Δ | Decode tok/s (median stream) A → B | Δ |
| --- | --- | --- | --- | --- |
| C1 4K | 2261.9/2190.7 → 2277.8/2216.0 | +0.9% | 63.6/64.3 → 67.4/64.7 | +3.2% |
| C1 16K | 2354.8/2343.5 → 2377.6/2388.4 | +1.4% | 55.1/56.6 → 59.5/55.2 | +2.6% |
| C8 4K | 2063.6/2043.3 → 2045.1/2040.4 | −0.5% | 16.5/16.0 → 18.1/15.5 | +3.3% |
| C8 16K | 2091.9/2092.1 → 2111.1/2088.0 | +0.4% | 13.9/12.7 → 13.3/13.9 | +2.4% |
| C16 4K | 1920.2/1952.0 → 1974.4/2007.0 | +2.8% | 6.7/7.1 → 7.2/7.4 | +6.5% |
| C16 16K | 2135.5/2120.9 → 2141.4/2134.9 | +0.5% | 7.0/6.8 → 7.0/7.5 | +4.9% |
| C24 4K | 1951.1/1873.5 → 1905.5/1928.7 | +0.3% | 5.4/5.2 → 5.4/5.5 | +2.8% |
| C24 16K | 2089.7/2108.4 → 2097.7/2100.0 | −0.0% | 5.0/5.0 → 5.2/5.3 | +4.8% |
| C32 4K | 1861.5/1840.4 → 1865.5/1885.5 | +1.3% | 4.2/4.3 → 4.4/4.7 | +7.4% |
| C32 16K | 2098.2/2081.8 → 2104.8/2103.4 | +0.7% | 4.1/4.1 → 4.1/4.3 | +3.5% |

`tools/matrix_benchmark.py compare` (2 vs 2 sessions): prefill geomean 1.008
(0.989–1.027), decode geomean 1.041 (1.019–1.063), no regression; individual cells
remain uncertain at two sessions. Interpreter version and build flags are confounded
(no 3.12 standalone arm). Quality: golden-vg2 identical; the ten golden-long round-0
prompts produced identical text on both runtimes. Warm startup to ready: 3.12 93 s,
3.15 83–88 s; the first 3.15 start builds its separate extension cache (683 s, once).
CPU suite (tests/ without tests/cuda): no 3.15-only failures beyond the Gemma
qualification tool's 3.12 AST contract, the GB10 calibration environment and the
wip image's stale protected install manifest.

## 3.15 interpreter features

- Experimental JIT (`PYTHON_JIT=1`, available in this build, off by default): one
  session against the two default 3.15 sessions was mixed (C1 decode −3% at 4K and
  −9% at 16K; C8 16K +9%; prefill within ±2.5%). Not enabled.
- Free-threading: not applicable to this stack. The SM70 Triton 3.2 build, xgrammar,
  the owned PyTorch build and the 3.11 limited-API FD owner are GIL builds, and
  serving already uses one process per tensor-parallel rank.

## Remaining items

- The media image's closures must be regenerated on CPython 3.15 before a media
  image can be rebuilt (in progress separately).
- GB10 (`deploy/gb10`) is ported in source; its ARM64 locks are generated and its
  image is built natively on GB10.
