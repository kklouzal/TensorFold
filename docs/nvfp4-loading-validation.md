# NVFP4 loading and runtime compatibility validation

This is a source-specific composite record. Historical source `abb67d6231903b67da2dd3fbca208c9e8b667180` passed its installed-provenance audits and affected installed 345-case CPU/28-case native gates, including NCCL world1. The selected refresh uses NGC CUDA 13.4.3, genuine official nightly Torch/Triton and protected native-core providers with exact authorized bundled components. Pin/guard promotion, native ARM image audits and the [GB10 bd1 full-checkpoint protocols](gb10-nvfp4-validation.md) are now closed at their recorded identities. The later source297 AUTO model protocols and focused 580-case CPU gate are also closed; its nine-case native fixture follow-up also passed. The [controlled eight-start record](gb10-nvfp4-validation.md#controlled-eight-start-comparison-and-user-selection) documents the user-selected four-stream profile and its observed single-request tradeoffs; no all-region superiority or equivalence is claimed. Old ABB results below retain their original stack identities and do not qualify the new foundation; this document does not announce release completion.

The [receipt](verification-nvfp4-loading-20261017.json) records exact source/image identities, closed evidence hashes, reuse boundaries and outstanding gates. Source099's [CT NVFP4 qualification](qwen36-ct-nvfp4-validation.md) and [receipt](verification-qwen36-ct-nvfp4-20261017.json) remain unchanged.

## Implementation and numerical contract

ModelOpt original routed codes and FP8 block-scale bytes are packed on the selected CUDA device, using the existing exact packed representation. The final up/down payloads and global-scale fields are transferred synchronously into owned CPU authority before publication. Per-tensor global conversion remains on CPU; original F32 global bits are checked independently. This changes placement of loading work, not checkpoint calibration, serving arithmetic, or the existing draft-only BF16/FP8 conversion. Only owned expert IDs are read for a compact rank shard; the leader retains the separate shared/backbone owner.

CT native-A4 keeps its original CPU byte permutation. The proposed Torch GPU permutation and private `checkpoint.pack4` alternative were not selected. The selected-source proof preserves the relevant099 numerical source bytes and CT loader body; it does not assert whole-runtime or startup equivalence.

Host admission now counts physical pinned Reader backing separately from GPU packing scratch and temporary CPU publication payloads. On the pinned allocator, each64MiB+3page Reader block uses a128MiB bucket, so its two-block ring is256MiB. Linux read and later D2H publication phases add the larger temporary allowance once. Authority, metadata and read-ahead are accounted separately. An expert-parallel leader with distinct backbone and expert Readers must account for each live ring; the follower accounts for its own Reader. Pure staged-platform resource controls do not constitute Windows execution qualification.

RotorQuant replaces the sole actual Triton-JIT chained range comparison with `CODEC >= 1 and CODEC <= 8`. The inclusive1–8 check, codec formats and numerical function bodies remain unchanged. The historical ordinary container used Torch2.16/Triton3.9 pins; the reported2.14/Triton3.8 combination is a separate compatibility region, now intentionally stopped before native completion.

Outside a managed foundation, NCCL library resolution honors `TF_NCCL_LIB` first, then the active interpreter's NVIDIA namespace package, Torch's bundle, and system discovery. The managed foundation binds its selected physical NCCL provider and refuses a conflicting override. A broken selected override or environment library fails explicitly. The loaded library remains owned through native communicator retirement. POSIX rendezvous sets `SO_REUSEADDR` before binding to reuse retired sessions while preserving active-listener exclusivity; it does not enable `SO_REUSEPORT`. Older sessions created without reuse may still need their kernel timeout. The collective/numerical/retirement bodies are unchanged by these discovery and bind edits.

## Native CT bank loading candidate — qualification pending

The current optional-error increment starts from frozen metadata source
`430936c8`. Both production consumers of the previous private factor result
discarded its successful values: public `native_calibration` returns its
original activation, while bank preflight needs only admission. Their private
helper now returns an error string or None, avoiding the discarded inner factor
tuple and outer result tuple. The existing `_calibration_result` value/error
contract is unchanged where its value is consumed. No compatibility alias or
copied arithmetic is introduced.

The actual checkpoint `_inv` and `alpha` helpers execute in their original
order. An explicit original CT multiplier still passes through its original
calibration check without reciprocal reconstruction. Invalid explicit input
returns before alpha; an invalid implicit inverse still executes alpha before
the joint finite/positive refusal. Exact public ValueError text, unexpected
helper/resource/cancellation propagation and public activation return remain.
The ordered and bulk loaders, ModelOpt, Qwen and Gemma callers therefore remain
in the affected validation/startup scope. The current source passed56 unique
CPU checks in8.36s on pinned AMD64/Torch9026/NumPy2.1 with GPUs hidden:
22 new helper-order/optional-error/public-return/exact-error/root-propagation
cases,6 existing actual FP32 admission/exhaustion cases,23 bulk byte/error
controls, one original CT maximum-multiplier boundary,3 CT loader propagation
cases and one real FD-owner control. No cases were skipped. All1324 source/test
files, the four runtime/test candidate paths and29 actual project origins
matched the sealed cohort. The unchanged CPP compiled to the existing native
provider under the original protected C++17/O3 recipe; raw mapped presence is
not a protected VMA-FD proof. The owned actor completed and retired without
reported cleanup failures.

The GPU-hidden four-CPU/16GiB gate used one-thread correctness settings. It does
not establish production thread-region performance, ARM64 ten/ten or installed
execution. Same-region integrated performance, other affected loading regions
and complete cold/warm model/resource/quality gates remain pending. No gain or
performance equivalence is inferred from removing tuples; historical bank
results below retain their identities and do not qualify this increment.

This checkout adds a separate metadata candidate: immediate validation may
borrow the nested declaration of the current sole default policy instead of
copying it. The public resolver still returns independent mutable results.
Exclusions, default-format inheritance, live nested policy mutations and public
resolver overrides remain authoritative; other policy capabilities retain
their original resolution/merge and precise errors. The borrowed view is never
mutated or returned from validation, whose result remains primitive values.
No policy cache or concurrent-mutation guarantee is introduced. It passed152
focused source CPU controls in1.71s on pinned AMD64/Torch9026/NumPy2.1, including
independent public resolver unions/errors, live mutation/replacement, resolver
dispatch overrides, primitive format outputs and existing host admission/error
and actual FD-owner integration. Native CPP, IO and scalar arithmetic are
unchanged; these are source-checkout checks, not installed or ARM64 qualification.

A separate local AMD64/SM120 warm bank comparison used the original Qwen3.6 CT
checkpoint, resident257-cell width512/dims2048 bank,8 intra-op/1 inter-op threads,
fresh production Reader contexts,24 measured positions plus4 warmups and6
mirrored HMMH/MHHM blocks. H is frozen `e87214fb`; M is the metadata increment.
Mean complete bank transaction time was0.469614653250s for H and0.445129550667s
for M. M minus H was−0.024485102583s, with the predeclared10,000-resample paired
block bootstrap95% pointwise interval[−0.027481021000,−0.021509427167]s; all6 block differences
were negative. Six blocks do not establish the assumed independence or
representativeness, and no whole-model saving is extrapolated.

Full six-field authority and native decode/prefill outputs matched before and
after. All1323 source/test files and39 actual project origins stayed unchanged.
Authenticated private H policy/format bodies and distinct real metadata owners
were verified before timing, with their method/default/source bindings unchanged
afterward. The preparation-only private import binding was restored to ordinary
builtins before any measured call. One unchanged CPU module/factory served both
arms and recorded728 production cache hits without misses or binding changes.
Six actual native proof aliases identified three verified mappings: that shared
CPU provider, the grouped CUDA provider and the FD provider. Inputs/foundation,
CUDA settlement and owned actor retirement passed their gates.

The metric includes loading/publication and attempted bank/cache/FD retirement.
Metadata/context preparation and initial compilation are excluded. Nonqualifying
policies retain extra dispatch costs requiring their own performance controls.
Required owned-resource admission, observed library/stack/module/compilation
effects, ARM64 ten-intra/ten-inter-op policy, installed packaging and full cold/warm model
startup/request/quality qualification remain pending. This is a local warm
component observation, not production promotion or a gain against accepted source.
The historical native-loop results below retain their separate source scope.

That reference folds the eight fixed byte transfers into the tile-row body of
frozen paired-tile reference `ab3c33ba`. It passed the same55 independent CPU
controls with an actual source-owned CPP build, and the separate local warm
bank comparison below. No ISA flag or compiler-specific inlining promise is
added. Required ARM64/installed, total resource admission and whole-model gates
remain pending; historical reference results keep their separate source scope.

The paired-tile reference is a separate loop variant of native reference `9734f82a`.
It copies each four-byte logical word directly to its final interleaved location
and consumes two adjacent source tiles per row, with a singleton final tile
when needed. It retains the reference's ABI, validation, owner lifetime and
publication behavior. It passed55 focused CPU controls and a separate warmed
bank comparison described below. Historical reference results retain their
source-specific scope. Required ARM64/installed, total resource admission and
whole-model qualification remain pending for both sources.

Native A4 loading can read a bank's validated F32 calibration fields into one
owned CPU vector, prepare the unchanged FP32 factors, and read an eligible
canonical Reader-owned contiguous SF run plus bounded code chunks. Physical
lexical source order maps explicitly to the requested local expert order.
One CPU native byte permutation preserves each projection's word prefix and
SF tail directly in final authority, including strided host-authority cells.
It replaces four high-rank indexed copies without an intermediate packed chunk.
Other supported topologies
retain the projection-ordered path with direct packing into final CPU rows.

Only explicit calibration-invalid results select ordered validation. Unexpected
conversion, helper, I/O and resource errors propagate. Invalid SF runs preserve
the original logical projection error order and code-read sequence. Raw owners
and temporary plans retire before synchronous device publication completes;
serving retains none of them. Original checkpoint calibrations, native kernels,
BF16 exclusions and serving arithmetic are unchanged.

The raw-owner/alignment envelope uses the existing64MiB reader window. Resident
admission includes simultaneous private CPU authority and raw owners; offload
keeps its separately charged authority and loading window. Both charge bounded
scalar/bank planning metadata, and preliminary pin admission removes only the
known resident-only increment. Fixed Python/native-record accounting remains an
open admission gate. The fused primitive additionally retains five borrowed
ndarray/managed-DLPack owner chains and their buffer descriptors, two whole-storage
Tensor views, a fixed1023-entry int64 stack map (8184 bytes), and bounded
byte-copy call state. Its byte loop is serial and synchronous; portable compiler
optimization is enabled with `-O3`, without architecture-specific intrinsics or floating math.
Parallel packing remains a separate candidate. No fixed-record dominance or total
admission result is inferred from removing indexed copies. Dense loading does not
enable or scan these bank plans.

The primitive follows the existing CPU native SSD reader's Torch C++ extension
facility, with `with_cuda=False`, an owned source-hash module name and C++17.
First-use compilation, code residency and failure are part of startup; a missing
or failing supported compiler is an explicit preparation failure, never a silent
fallback. CPU-only correctness must also run with GPUs hidden.

NumPy borrows newly imported DLPack aliases. Each alias owns a separate from_blob
Storage and retains its source Tensor, so NumPy's non-resizable base flag does
not change the original Storage. All original owners must remain exclusively
owned and unmodified by other threads through the GIL-released native interval.
Whole-storage byte exports and explicit offsets/strides bound physical cache-row
access. The kernel snapshots and validates IDs and all byte ranges before its
first write. Public `torch.autograd.graph.increment_version` marks original
targets before raw mutation. Any failed call leaves authority unpublished; exact
historical version-counter increments are not promised. Checkpoint bytes, input
versions, storage capacities, unselected cells and padding remain unchanged.

The frozen fused reference builds on the indexed loading reference `dcfc448a`
(7359 source composition), which is itself composed on accepted source
`3b8cee82`. Its first six-file source passed68 CPU controls in8.64s, including
actual source-owned CPU compilation with `-O3 -std=c++17 -Wall -Wextra -Werror`,
independent nonperiodic/all-byte packing oracles, strided authority, capacity,
version, lifetime and malformed native-buffer checks. Review then corrected
the source-hash read to use the existing owned regular-file stream; five
additional actual-function/owned-stream controls passed. The CPP byte algorithm,
adapter and independent oracle were unchanged by that correction. These are
checkout checks on pinned Linux AMD64 Torch2.16.0.dev20261017 (`9026ad48`),
and NumPy2.1 with GPUs hidden. The complete bank controls below used the
RTX PRO 2000 Blackwell SM120. Neither is installed ARM64 or whole-model
qualification.

The corrected fused seven-file source (manifest3a0742cf) completed the local warmed
resident257-cell bank comparison against the indexed reference. It used the
original AEON-7 Qwen3.6 CT export, width512/dims2048, checkpoint math,8 intra-op
and1 inter-op threads,6 mirrored AFFA/FAAF blocks (24 measured positions plus4
warmups), fresh production Reader contexts and the same complete bank consumer.
Mean load/publication/retirement time was1.154287824s for the indexed reference
and0.550330685s for the fused implementation. The fused-minus-reference mean
difference was−0.603957139s; the predeclared10,000-resample paired-block bootstrap
gave a95% pointwise interval of[−0.667603003,−0.568140370]s. Every paired block
was faster. The six-block independence/representativeness assumption remains
an exploratory limit; no whole-model effect is extrapolated from these values.

Complete six-field bank bytes and native prefill/decode controls matched before
and after the cohort. All1323 source/test files were stable,39 actual project
origins were coherent, and actual mapped-file FD proofs covered the CPU kernel,
grouped consumer and file-owner provider. The original cached CPU factory had
exactly364 production hits (14 fused banks×26 chunks), with unchanged misses,
binding and cache size; no timed factory wrapper or observer was introduced.
Foundation/input identities, CUDA settlement and exact actor retirement passed.
The original CPU shared object was253960 bytes; its loaded-file hash matched the retained
CPU build product. Both independent reviews closed this bounded result.

The reference metric excludes header/context preparation and first-use compilation,
which initial correctness warmed before timing. A subsequent diagnostic profile
used one unprofiled warm bank and two fresh instrumented bank passes with the
same source and provider checks. Its cProfile attributes roughly0.217s to26
native pack calls,0.159s to28 reads,0.066s to six transfers and0.075s to771
metadata resolutions. Instrumented timings are attribution only. ATen CPU
events do not observe the raw CPP byte loop or total NumPy/native/Python memory.

The separate direct-copy/paired-tile variant passed55 CPU checks in8.88s on
the same pinned AMD64 checkout runtime, including an actual source-owned CPP
build, independent byte oracles over singleton, paired and odd-tail tiles,
multiple N tiles, strided authority, unaligned input and output margins, plus
the existing native-buffer, ownership, source-read and FD controls. Its generated
CPU shared object was254008 bytes. These checks do not establish ARM64 or
installed-package behavior.

Its separate local warm resident257-cell comparison retained the same Qwen3.6
checkpoint, width512/dims2048,8 intra-op and1 inter-op threads, full-bank
transaction boundary,24 measured positions plus4 warmups, and6 mirrored
FGGF/GFFG blocks. Here F is frozen native reference `9734f82a` and G is the
direct-copy/paired-tile variant. Mean transaction time was0.5541440385s for F
and0.49185965075s for G. G minus F was−0.06228438775s, with the predeclared
10,000-resample paired-block95% pointwise interval
[−0.068787578583,−0.055258490583]s. All6 paired blocks were faster. The same
six-block independence and representativeness limits apply; these values are
not whole-model startup estimates.

Complete six-field authority and native decode/prefill outputs matched before
and after that cohort. All1323 source/test files and39 actual project origins
were coherent. Seven actual mapped-file FD records covered both exact CPU
build products, the grouped consumer and FD owner, including their mapped
aliases. Each unchanged production CPU factory had exactly364 cohort hits,
with no new misses or binding changes. Seven private/shared reference module
bindings, foundation/input identities, CUDA settlement and exact actor
retirement also passed. Both independent reviews closed this local result.
Preparation and cold compilation remain excluded from the metric.

Actual AMD64 disassembly shows direct scalar four-byte transfers without the
reference's interleaved row-array staging. The copy helper remains an actual
per-row call; SIMD and helper-call elimination remain separate candidates.
Direct stores and paired traversal changed together, so the measured gain is
not attributed separately to either. No reduced stack or total-admission claim
follows from removing the source arrays.

The current tile-row-body increment passed55 CPU checks in8.92s on the pinned
AMD64 checkout, including the unchanged independent byte, paired/tail,
unaligned-input, strided-authority, buffer/ownership, error and FD controls. Its
actual shared object was258056 bytes. Read-only disassembly shows no separate
copy helper and no calls between the owned GIL-release/reacquisition boundaries;
it also shows compiler-generated unaligned SSE2 word-unpack operations. This
does not establish which compiled version every input selects. The observed
pack frame reserves9544 bytes,224 more than the paired-tile reference; no stack
or total-admission reduction is claimed.

Its separate G/H comparison used the same pinned AMD64/SM120 runtime and original
Qwen3.6 checkpoint, resident257-cell width512/dims2048 bank,8 intra-op and1
inter-op threads, fresh production Reader contexts and complete transaction
boundary. Six mirrored GHHG/HGGH blocks retained all24 measured positions and4
warmups. G is frozen `ab3c33ba`; H is this tile-row-body increment. Mean
transaction time was0.488834094917s for G and0.464444748167s for H. H minus G
was−0.024389346750s, with the predeclared10,000-resample paired-block95% pointwise
interval[−0.027546247750,−0.021127616167]s. All6 block differences were negative.
The six-block independence/representativeness assumption remains exploratory;
these observations are not a whole-model startup estimate.

Complete six-field bank bytes and native decode/prefill controls matched before
and after. All1323 source/test files,39 actual project origins, the authenticated
private reference bindings and input file stamps stayed unchanged. Both original
CPU factories recorded exactly364 production cache hits without a miss or binding
change, and actual mapped-file FD proofs identified both CPU shared objects plus
the grouped CUDA and FD providers. CUDA settlement and owned container retirement
completed. The metric includes bank loading/publication and attempted bank/cache/FD
retirement; metadata/context preparation and initial compilation are excluded.
The sealed plan's schedule description contains a retained transcription typo;
its executable operator, frozen analyzer and all actual records agree on the
GHHG/HGGH schedule stated here. No positions or raw assets were changed.

The earlier fused-reference packet failure before container launch and a separate
native-manifest schema failure before test collection remain retained; neither
produced performance samples. The complete24-position result uses only the
successful fresh actors. The original fused result compares against the indexed
candidate; the paired-tile result compares against the frozen fused reference,
and the tile-row-body result compares against the paired-tile reference.
Full fixed/stack/module/compiler admission, required
ARM64 thread-policy observations, installed execution, whole-model cold/warm
startup and request/quality qualification, and comparison against accepted
source remain pending. No production gain or release promotion is claimed.

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

Source `0a00cb181426aeb50e83c7ee97bd73c08157901e` contains the managed foundation/origin interfaces and successfully generated both real dependency closures: **34 foundation,31 runtime and3 test distributions per architecture**. AMD64 uses a native CPython3.12 report. ARM64 uses the supported cross-report path with the real x86_64 resolver environment retained; target wheel tags, markers, extras and dependencies were validated. That cross report is not native execution. A later independent acceptance of the actual native ARM CPython 3.12.3/aarch64 report confirmed the same dependency payload; its original output-reading wrapper failure remains preserved. Both reports cover68 distinct distributions; counts across architectures are not combined into a test total. The0a00 origin change narrowly accepts HTTPS `pypi.nvidia.com` on its normal TLS port for `cuda-toolkit`/`nvidia-*` transitive packages; the three audited roots keep their exact official PyTorch URLs. No local-file or fabricated dependency report is substituted.

The initial resolver build commands exited0, but their output-reading wrappers failed with preserved `PermissionError` records. Independent acceptance sealed readable byte-identical output copies and their original inner provenance. The maintained generator then passed at0a00 for both reports, emitting five artifacts per target; all five exactly match the earlier1c32 outputs. A copied schema2 pin file is an input, not a sixth generated lock. This is completed resolution/generation evidence, not package installation or GPU qualification.

The [managed foundation contract](../deploy/gb10/foundation.md) describes protected prefixes, native-provider selection, immutable SDK ownership and generated artifacts. Guard/pin promotion and native ARM image capture/audits are complete for the [GB10 bd1 qualification](gb10-nvfp4-validation.md), which keeps full-checkpoint results separate from pending final-composition gates. Historical ABB 345 CPU/28 native and 488 RQ/KV observations are not promoted to the new stack.

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

The user's width640/hidden2560 is authoritative geometry. Their48.9ms CPU/4.9ms RTX4080 component observations and6.5-minute GPU startup are **user-reported**. The old CPU attempt never completed;30 minutes was a projection. None is a local full-startup measurement. The separate GB10 Flash Next ModelOpt checkpoint now has [source-specific full-model evidence](gb10-nvfp4-validation.md); it is not the reported desktop geometry or a controlled startup comparison.

Source297 installed builds/audits, focused CPU and AUTO model protocols are separately closed in the [GB10 record](gb10-nvfp4-validation.md#source297-auto-graph-protocols). Its nine-case native fixture follow-up also passed. The controlled same-format A-D record and user four-stream selection are linked above; E/F were not promoted after the user stopped further candidate work. ARM bd1 foundation/model checks retain their identities; they do not refresh the AMD64 Heretic 512-token or exact 640×2560 component comparisons on this foundation. The older3.8 native attempt is intentionally stopped, not pending success. Results will be added only after their owned receipts close. Historical099 andABB67 qualification stays source/image-specific and unchanged.

The audited owned-container execution gates retained the external sampled8GiB host-free floor and complete live ancestor cgroup resource proof; Docker RAM/swap/CPU/PID quotas were absent, with unlimited memlock/stack. Resolver preparation instead has its recorded sampled host-floor and direct-client retirement evidence; those build receipts do not establish the live BuildKit daemon/executor ancestor limits. Sampling is not a continuous reservation. No unrelated OS/global-limit, two-GPU/mixed-architecture, Windows, ARM64, Apple or sanitizer qualification is claimed by the historical ABB increment. The newer ARM model and focused memcheck results have their separate [GB10 receipt](verification-gb10-nvfp4-20261018.json). Counts from overlapping gates or different sources are not added together.
