# NVFP4 loading and runtime compatibility validation

This is a **qualification draft**. Historical source `abb67d6231903b67da2dd3fbca208c9e8b667180` passed its installed-provenance audits and affected installed345-case CPU/28-case native gates, including NCCLworld1. The selected refresh uses the latest audited NGC CUDA13.4.3 base, genuine official nightly Torch/Triton and a verified native-core loader with the exact user-authorized TorchVision bundled components. CPU provider validation and both architecture dependency-generation gates are closed. Final production guard/pin promotion, immutable images and fresh GPU/model qualification remain pending. OldABB results below retain their original stack identities and do not qualify the new foundation. This document does not announce release completion.

The [receipt](verification-nvfp4-loading-20261017.json) records exact source/image identities, closed evidence hashes, reuse boundaries and outstanding gates. Source099's [CT NVFP4 qualification](qwen36-ct-nvfp4-validation.md) and [receipt](verification-qwen36-ct-nvfp4-20261017.json) remain unchanged.

## Implementation and numerical contract

ModelOpt original routed codes and FP8 block-scale bytes are packed on the selected CUDA device, using the existing exact packed representation. The final up/down payloads and global-scale fields are transferred synchronously into owned CPU authority before publication. Per-tensor global conversion remains on CPU; original F32 global bits are checked independently. This changes placement of loading work, not checkpoint calibration, serving arithmetic, or the existing draft-only BF16/FP8 conversion. Only owned expert IDs are read for a compact rank shard; the leader retains the separate shared/backbone owner.

CT native-A4 keeps its original CPU byte permutation. The proposed Torch GPU permutation and private `checkpoint.pack4` alternative were not selected. The selected-source proof preserves the relevant099 numerical source bytes and CT loader body; it does not assert whole-runtime or startup equivalence.

Host admission now counts physical pinned Reader backing separately from GPU packing scratch and temporary CPU publication payloads. On the pinned allocator, each64MiB+3page Reader block uses a128MiB bucket, so its two-block ring is256MiB. Linux read and later D2H publication phases add the larger temporary allowance once. Authority, metadata and read-ahead are accounted separately. An expert-parallel leader with distinct backbone and expert Readers must account for each live ring; the follower accounts for its own Reader. Pure staged-platform resource controls do not constitute Windows execution qualification.

RotorQuant replaces the sole actual Triton-JIT chained range comparison with `CODEC >= 1 and CODEC <= 8`. The inclusive1–8 check, codec formats and numerical function bodies remain unchanged. The historical ordinary container used Torch2.16/Triton3.9 pins; the reported2.14/Triton3.8 combination is a separate compatibility region, now intentionally stopped before native completion.

Outside a managed foundation, NCCL library resolution honors `TF_NCCL_LIB` first, then the active interpreter's NVIDIA namespace package, Torch's bundle, and system discovery. The managed foundation binds its selected physical NCCL provider and refuses a conflicting override. A broken selected override or environment library fails explicitly. The loaded library remains owned through native communicator retirement. POSIX rendezvous sets `SO_REUSEADDR` before binding to reuse retired sessions while preserving active-listener exclusivity; it does not enable `SO_REUSEPORT`. Older sessions created without reuse may still need their kernel timeout. The collective/numerical/retirement bodies are unchanged by these discovery and bind edits.

## Historical ABB67 artifacts — audited and affected gates closed

| Artifact | Immutable identity |
| --- | --- |
| Source | `abb67d6231903b67da2dd3fbca208c9e8b667180` |
| Source UID | `a7b31df85b7c03f041a23ec46f41daf71f7debbb6548b3bb4636108a02a15d41` |
| Archive SHA256 | `bf0e8b1e228081351f2cfcb0a88b39eb3f49a273ccabb886bc6ed35b530f67e4` |
| Runtime image | `sha256:ef32b6bf8c459ae945eb00628a02bae40461e88829a63b7c62b7a920068be3df` |
| Verification image | `sha256:8a7765e9655fa10f60956c26c21442f64164d45bf431b0490e74cd157b39ca0c` |
| Installed project wheel SHA256 | `943ff1a2dc099c31c25a2db692dbebf95b74f65ab9abe1d4bf5e5d4a30288b96` |

Historical capture `final-capture-abb67d62/attempts/final-03` completed on native LinuxAMD64. Both audits matched531 installed package files and the source/native-owner identities; runtime matched542 baked files and verification1261. The native file-owner extension is `e8199a960dd762f77555016a004308310e0bdba7a76ce04205be29237c868ff1`. NCCL2.30.7/native23007 loaded from the active `/opt/tensorfold-venv` package directory. These audits did not initialize CUDA or run a model.

The stack is CPython3.12.3, Torch2.16.0.dev20261006+cu134, CUDA13.4, Triton3.9.0+gitaad2a60d, xgrammar0.2.8, tokenizers0.23.1, Jinja3.1.6 and NumPy2.1.0. Build identity and dependency pins remain source-specific; existing099 timing numbers are not relabeled as results for these new images.

## Selected foundation — CPU and dependency gates closed

The route is selected: the latest audited `nvcr.io/nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04` index is `sha256:f2d23b75391766a02175908d3fa1c2afee9849aa13d09326d71649de2b33cfe9`, with separate immutable native child images and genuine official wheel hashes. All six root wheels were fetched and independently verified through the same regular descriptor for whole SHA256 and ZIP metadata; metadata hashes were never substituted for whole-wheel hashes.

| Target | Native base child digest | Torch | TorchVision | Triton |
| --- | --- | --- | --- | --- |
| AMD64 | `b3f64e4fec7d0825e05bf685e628e7e112afd7d63bc5906e2f71fd1f56cce336` | `2.16.0.dev20261017+cu134` | `0.30.0.dev20261007+cu134` | `3.9.0+gitfc8d31a5` |
| ARM64 | `e57a21d3cf1398462f74efe1b57603cb968ee750088c6160ea89458b4f64dcfc` | `2.16.0.dev20261014+cu134` | `0.30.0.dev20261006+cu134` | `3.9.0+gitfc8d31a5` |

The complete original upstream dependency metadata, including Torch's `cuda-toolkit==13.4.1` requirement and transitive extras, remains intact. Package presence and actual native provider selection are different contracts: the chosen loader prepares the native13.4.3 core, uses supported compiler/tool/header/libdevice controls, and permits only exact protected component-bundled files authorized by the user. No vendor patch, source-built Torch, invented wheel metadata, alias shim or older matching-base fallback is selected. Native addons are the two runtime packages cuSPARSELt0.8.1.1 and NVSHMEM3.7.2; their dev/static, MAGMA and gfortran source-build dependencies are not selected.

The closed AMD64 CPU proof uses image `sha256:f09a4185f8ce6c0e35295565e43f7c27b9f74e50b7cf9d78390dc4d105a76adb` and helper `ae87f5b8…`. Actual Torch/core and TorchVision native/schema imports, Triton import, CPU integer arithmetic, provider maps and `pip check` passed without CUDA initialization. Its18 native physical rows comprise15 SDK providers, two C++ runtimes and one explicit forward-compatibility driver. Two additional exact official TorchVision members are authorized: `torchvision.libs/libcudart.a6a4ec3c.so.13` and `torchvision.libs/libnvjpeg.be6456a4.so.13`, with content/RECORD identities in the receipt. This is a CPU provider proof, not a device-eligibility, GPU JPEG, model, final-production-image or ARM execution pass.

Source `0a00cb181426aeb50e83c7ee97bd73c08157901e` contains the managed foundation/origin interfaces and successfully generated both real dependency closures: **34 foundation,31 runtime and3 test distributions per architecture**. AMD64 uses a native CPython3.12 report. ARM64 uses the supported cross-report path with the real x86_64 resolver environment retained; target wheel tags, markers, extras and dependencies were validated. No native ARM report or execution is invented. Both reports cover68 distinct distributions; counts across architectures are not combined into a test total. The0a00 origin change narrowly accepts HTTPS `pypi.nvidia.com` on its normal TLS port for `cuda-toolkit`/`nvidia-*` transitive packages; the three audited roots keep their exact official PyTorch URLs. No local-file or fabricated dependency report is substituted.

The initial resolver build commands exited0, but their output-reading wrappers failed with preserved `PermissionError` records. Independent acceptance sealed readable byte-identical output copies and their original inner provenance. The maintained generator then passed at0a00 for both reports, emitting five artifacts per target; all five exactly match the earlier1c32 outputs. A copied schema2 pin file is an input, not a sixth generated lock. This is completed resolution/generation evidence, not package installation or GPU qualification.

The [managed foundation contract](../deploy/gb10/foundation.md) describes protected prefixes, native-provider selection, immutable SDK ownership and generated artifacts. Final production guard/pin promotion and final sourceUID/runtime/verification image capture remain pending. HistoricalABB345CPU/28native and488RQ/KV observations are not promoted to the new stack.

## Closed selection evidence

Four source5f9 actual-model starts used CPU/GPU/GPU/CPU order after a separate warmup. CPU engine construction took65.713419/65.337612s (mean65.525515s); GPU permutation took67.334924/67.568535s (mean67.451730s). Complete HTTP-ready means were66.008059s and67.935910s. Every GPU observation was slower than every CPU observation; the observed mean engine difference was+1.926215s/+2.93964%. Two starts per arm give descriptive local ordering, without a confidence interval, equivalence or universal ranking. Wrapper time includes requests and is not the startup objective.

The separate warmup took91.048715s and logged two native builds, one during the later first request. The four timed logs had no native extension-build messages; this does not prove every possible silent compilation absent. All25 full32-token outputs, serial/draft/seeded identities, metadata, work signatures, pool sizes and owner retirement were verified. Context4096/one stream/eight draft rows/BF16 secondary/pinon were fixed, with the same4424-cell pool. This is source5f9 diagnostic evidence; it is not a new final-image512-token or full-ModelOpt result.

Inclusive default4-thread components also favored CT CPU:1.212763ms versus1.362959ms for Torch GPU and1.344498ms for checkpoint.pack4. ModelOpt width512/1024 with hidden2560 showed9.606×/16.630× ratios of CPU to GPU medians, including transfers and final CPU authority copy. These deterministic periodic synthetic components do not measure a full exported model's startup. Explicit8-thread results reverse CT component ordering and remain separate sensitivity evidence; they do not select a new CPU thread default.

The selected ModelOpt/host-budget source `b9d370166da39a5e5d0df19d7d3921bd7d1631b2` passed266 CPU controls with12 CUDA skips in4.87s in a diagnostic source overlay. New final-image gates are still required. The later640 geometry test uses nonperiodic independent projection bytes, all256 block-scale encodings, distinct original F32 globals, exact CPU authority comparison and failure-lifetime recovery.

## Closed pinned3.9 and NCCL/store gates

Source `4c0d3279` passed **488 selected cases,1 skip in791.70s**, covering the full selected native RotorQuant and KV test modules plus six new ModelOpt loading cases. Width640/hidden2560 had39,937,536B peak Torch allocation against231,833,856B admitted device loading, with zero live allocation delta on return. All five printed ModelOpt shape records met their admission bounds. These are synthetic payloads, including the user's actual geometry.

The raw summary did not name its skip. Source inspection identifies the unique in-run optional `exllamav3_ext` import-skip; that is a source inference, not a raw `-rs` report or a pass of the external-extension oracle. All selected test/numerical/dependency/build bytes match the finalabb67 capture. The only runtime/test/build changes from4c0 are `comm.py`, `local_store.py` and their two affected CPU test files. This supports scoped numerical-source reuse; the488-case result remains a4c0 diagnostic gate.

NCCL/store source `4755ac0f490b66f680de01492fbf85ef13f1d6b3` passed139 coupled CPU controls on the pinned stack (4.39s) and139 on the owned2.14 compatibility image (3.07s). Its two production-file hashes matchabb67. The compatibility CPU native `ncclGetVersion` returned23007 from `/opt/triton38-venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2` without CUDA initialization. These are library/store/boundary checks, not GPU-communicator or two-GPU qualification. Preserved initial missing-native-FD preparation failures remain historical evidence; the proper compatibility gate closes that prerequisite gap.

## Closed historical ABB67 installed gates

Verification image `8a7765e9…` ran without a source overlay. The affected17-module installed CPU gate passed **345 cases in8.24s**. It covers loader/admission/host and EP accounting, range-source, NCCL/store and lifetime controls; this is not a new full-project CPU run. The installed native gate passed **28 cases in281.42s**, selecting ModelOpt loading/host-expert tests and two NCCLworld1 checks: native broadcast/deferred-error checking, and an unsharded-bank session that loads, runs and retires. Full tiny ModelOpt fixtures remain synthetic; these tests do not represent the user's full export or a two-GPU run.

Both containers exited0 with ownership resolved and no OOM, observer errors or cleanup errors. Minimum sampled MemAvailable was54,115,098,624B for CPU and41,164,124,160B for native, above the8GiB floor. The native log preserves two c10d IPv6 localhost-probe `EAFNOSUPPORT` warnings before successful IPv4 operation; it is not IPv6 qualification. These results qualify only the originalABB67/pinned stack.

Tests-only increment `7621d853cb11594a527830f7e6f4e19130492de1` integrates origin `8c0728c4` and adds early optional-Triton collection admission plus an isolated absence/no-CUDA regression. The preserved predecessor control, with Torch present and Triton explicitly blocked, raised `ModuleNotFoundError` through fixture/backend imports without initializing CUDA. The correction places the guard before those imports; source proof retains all existing test-function AST and nine assertions. Four diagnostic cases passed in4.70s on image `9115f1a8…`, separately from the historical345. Mechanical comparison toABB67 changes exactly two test files and no production numerical/build bytes; both coding reviews are clean. Neither the four diagnostic cases nor the earlier345-case count is relabeled as a new-foundation gate.

## Owned Torch2.14/Triton3.8 compatibility region

Image `sha256:db68e206cea81db95c1a8f228ca1f30eee5cf9bd97256c1da4db151a513b234a` passed CPU readiness:50 packages,23 mapped owned backend DSOs, clean `pip check`, isolated CPython3.12.3 virtualenv and a successfully created/closed native file owner. It contains Torch2.14.1+cu130/Triton3.8.0/NumPy2.5.3. NVCC is13.4.59 and Triton's bundled ptxas12.9.86; this mixed compiler identity is explicit.

Preparation used audited installed-wheel RECORD relocation plus missing project dependencies and a normal offline install of the cached099 project wheel `c3dc9449018b7597bdf28baae7c8aa30561427919302135f78b481794a16f15b`. It is not a fresh official-wheel install or a full reproduction of the friend's environment. The fixed source must be supplied as a read-only diagnostic overlay; installed-only old Python kernels do not qualify the fix. Commands use the owned interpreter with `-m`, not copied old-prefix console entrypoints. The reference environment and stopped embedding container were not changed or started.

The full two-module RQ/KV plus six ModelOpt native compatibility run was **intentionally stopped at the latest user request**. Its terminal exit was137 with `OOMKilled=false`; no test summary was reached. The raw runner correctly retains a nonzero-payload result and terminal-status cleanup diagnostic. This incomplete run is neither a completed compatibility pass nor evidence of a product-test failure; partial progress is not coverage. Owners resolved and the observer retired. The separate CPU-readiness pass remains historical evidence.

## Reported geometry and remaining gates

The user's width640/hidden2560 is authoritative geometry. Their48.9ms CPU/4.9ms RTX4080 component observations and6.5-minute GPU startup are **user-reported**. The old CPU attempt never completed;30 minutes was a projection. None is a local full-startup measurement. No full reported ModelOpt checkpoint is available here.

Pending before final qualification: final production guard/pin promotion and new immutable capture/audits, then fresh installed CPU/native/NCCLworld1, actual Heretic512 and the exact640×2560 inclusive component benchmark on the selected foundation. The route and dependency generation are closed; final code/image identities and execution gates remain pending. The older3.8 native attempt is intentionally stopped, not pending success. Results will be added only after their owned receipts close. Historical099 andABB67 qualification stays source/image-specific and unchanged.

The audited owned-container execution gates retained the external sampled8GiB host-free floor and complete live ancestor cgroup resource proof; Docker RAM/swap/CPU/PID quotas were absent, with unlimited memlock/stack. Resolver preparation instead has its recorded sampled host-floor and direct-client retirement evidence; those build receipts do not establish the live BuildKit daemon/executor ancestor limits. Sampling is not a continuous reservation. No unrelated OS/global-limit, two-GPU/mixed-architecture, Windows, ARM64, Apple or sanitizer qualification is claimed for this increment. Counts from overlapping gates or different sources are not added together.
