# ProviderOne implementation status and authority handoff

This is a source-to-contract map, not another design or a release certificate.
The [canonical governing design](../../ProviderOne%20Implementation%20Docs/ProviderOne_V100_Adaptive_Inference_Design.md),
particularly Sections1,2,21,22 and24, selects the required behavior and exit
gates. The four companions in that directory provide research/review and a
conditional experimental backlog; they do not independently select a serving
policy. All five authoritative input files are adopted in place without byte
changes. The [legacy design path](../v100-adaptive-inference-design.md) points
there. Untracked `!` drafts remain user material, not maintained authority.

## Baseline, dirty state and evidence identities

The starting implementation is `python-0.6` commit
`8d69e49c0adc0192a3849169cccff9303cee51d0` in the owned
`https://github.com/kklouzal/TensorFold.git` lineage. The governing document's
c8318787 audit is a historical source snapshot; it is not an audit/pass claim
for this later baseline. Current package/import/native/CLI identities still
use TensorFold. Product rebrand, packaging/updater/remote changes and an
installed ProviderOne release remain requested work, not completed by this
handoff. Historical source, image, namespace and benchmark names are not renamed.

At the starting checkpoint the index was empty and local/remote commits matched.
The old tracked root `TensorFold_V100_Adaptive_Inference_Design.md` was already
deleted by the user; this checkpoint adopts that deletion without recreating
or editing it. Otherwise a fresh checkout would retain a stale full contract.
Its old HEAD blob `a289270fed60f61d5d471c9b39cd8c1c2ac57c5d` is preserved outside
the checkout. The five canonical files were untracked author inputs and are
now the explicitly authorized additions. Four untracked `!` drafts and six
unrelated B16 source/test files remain untouched and unstaged. This checkpoint
changes no runtime code, model, dependency, installed service or hardware setting.

Executor baseline receipt (private audit root
`/home/kklouzal/audit-artifacts/gemma-aeon26b-optimization-20261102`):
`query-tile8-cleanup-v1/git-closeout-v1.json`, SHA256
`e4feb74b676bfb9b155644328d1d2cf01b70c2e97442b7812b67a408b918e84d`.
The preserved old-root reference is `providerone-authority-old-root-8d69-reference.json`,
SHA256 `2af5e7fc2377ef5767d07af48af1b1a127d2f08e38373a9e5422c960dd12bd94`;
its bytes hash is `7c9f724346866cfb89382ee2e30d60d9d827222c347610783b37683ba04e5a8c`.
Private receipts are provenance, not dependencies for fresh-checkout tests.

## Requested, implemented, qualified and effective

The [branch consolidation scope](../branch-consolidation.md) records the
independently ported boundary/tool/fixture changes, the Dense wrapper correctness
repair and the prepared native-owner host-word component. Tail/Pool preparation
remains outside integration for target qualification. Those source checks do not
qualify an installed image, change the selected profiles below or complete the
ProviderOne release.

The unselected [NativeRN component trial](../v100-adaptive-validation.md#nativern-current-source-component-qualification)
has 52 passing SM70 operator cases. After a structurally verified view-return
cleanup, a fresh genuine image passed 255 installed CPU controls and 42
owner-fixture cases, including four eight-rank cases and 32 child origins,
without skips. These are bounded API/ownership, numerical-enclosure and
RN-H1/toy-Q8 graph/transport results, not full-model inference or a selected
serving profile. Full model, context/capacity, resource and performance gates
remain open; the
existing four Q03 strict numerical failures remain open.

Ordinary fixed-profile HTTP now has source-verified global host phase credits,
including worker/error tails and actual metadata/ID conversions. The recorded C27
joint-reservation control and 591 distinct CPU nodes plus 21 subtests passed.
The installed successor has since admitted 32 full-window Q4 States per rank
and completed the [first partial throughput reference](../benchmarking.md#first-sc117-q4-reference)
through 16 active requests. Live C27 activity, the complete sweep, managed
profile publication and native allocator/RSS qualification keep their separate
gates. See [the integrated scope](../v100-adaptive-validation.md#ordinary-http-phase-credit-integration).

“Implemented” below identifies existing starting source and bounded software
controls, not completion of the full governing subsystem. Qualification and
effective deployment require their own exact artifacts, target and receipts.

A5 restores the 37 inherited fixture cases in the scoped regression set. Source
and installed batches now each pass 1,498 tests with one optional companion
skip; all 115 original assertion nodes remain and three new non-GGUF provider
negatives pass. Runtime source and the installed A4 images are unchanged. Earlier
failure counts remain historical evidence; no actual hardware, model, capacity
or performance gate is inferred from these repaired CPU controls. See
[the fixture and artifact scope](../v100-adaptive-validation.md#restored-fixture-contracts-a5).

| Contract surface | Requested | Existing implementation starting point | Qualified evidence | Effective status |
| --- | --- | --- | --- | --- |
| Shared API/admission/lifecycle | One shared logical/API owner, bounded FIFO/credits, cancellation/publication and complete computed-state preservation | [admission](../../src/tensorfold/server/admission.py), [Responses](../../src/tensorfold/server/responses.py), [Conversations](../../src/tensorfold/server/conversations.py), [logical records](../../src/tensorfold/server/api_records.py); Qwen prefix/retained catalog and RAM restore | CPU software milestones exist. Logical records alone do not qualify GPU continuation. The concrete ProviderOne integration and actual SC117 workload/failure gates remain open. | Existing named API profiles remain unchanged; no SC117 ProviderOne release is asserted. |
| Artifact/container boundary | Bounded neutral container decoding before family selection, exact content/model/recipe identity | [neutral GGUF inventory](../../src/tensorfold/gguf_file.py), [neutral file checks](../../src/tensorfold/tensor_file.py), [checkpoint metadata](../../src/tensorfold/checkpoint_metadata.py), [Flash Next semantics](../../src/tensorfold/families/qwen4_exp/gguf.py) | A1 source separates the existing container/file algorithms from family admission and passes one validated inventory into family configuration. Exact file-helper bytes and token/decode definitions are retained. Focused boundary and corrected resolver cases passed; the current CUDA-hidden source batch recorded 1228 passed/48 baseline failures/4 skipped, with clean static checks on 115 Python files. The fresh installed image recorded 1264 passed/12 pre-existing unqualified failures/4 skipped, with 778 installed runtime files unchanged before/after, active purelib origins and no source overlay. Fresh native construction resolved 36 stale-FD failures; no test or guard was relaxed. [Exact identities and limits](../v100-adaptive-validation.md#neutral-artifact-io-boundary-a1). | Affected installed I/O boundary qualified; no new model/format/device support or production deployment is selected by the refactor. |
| SC117 target/SM70 stack | Non-Swift SC117 IQ3_S, exact source-built SM70 stack, native/extended context and complete state/quality/resource gates | [V100 build procedures](../../deploy/v100/README.md), [source foundation](../../deploy/v100/source-foundation.md), Qwen GGUF import and CUDA engine | Exact source/header/import and historical CPU/build milestones exist; selected payload/mmproj verification is recorded. Current physical V100 allocation, model/state/quality, useful capacity and throughput are not established by this checkpoint. | No selected SC117/V100 release established by this map. |
| Full-window capacity and unique state | Imported-allocation worst-rank capacity, fixed unique KV/indexer placement, typed recovery; no silent context reduction | [history owner](../../src/tensorfold/families/qwen4_exp/cuda/history_owner.py), [placement](../../src/tensorfold/families/qwen4_exp/cuda/history_placement.py), [indexer history](../../src/tensorfold/families/qwen4_exp/cuda/indexer_history.py) | A2 source selects explicit fixed degree4 at initial TP8 unique-history preparation and publishes that map to State. The ledger retains a zero dedicated migration-overlap term; ordinary serving does not activate or accept live-degree updates. Prefix/COW/restore/recovery/replacement reserves and optional research oracles are preserved. A2 affected source and installed CPU controls passed: each broader batch recorded 367 passed/4 existing dedicated ABI/version failures/12 frontend-version skips; required deployment/public-permission 194 and static 7 passed. Fresh installed 778-file inventories and changed-module purelib origins matched with no source overlay or CUDA initialization. The four target gaps remain unqualified. Physical fixed-map numerical/resource/long-history qualification and the maximum admitted integer count remain pending. [Exact A2 identities and unit-test scope](../v100-adaptive-validation.md#fixed-initial-unique-history-a2). | No SC117 integer capacity or new live-redistribution policy is effective here. |
| Required QSA MTP | Default shared-Q8 MTP, exact aliases/norms, selected Q8/iso4 KV and full-prefix text/visual conditioning | [GGUF MTP import](../../src/tensorfold/families/qwen4_exp/cuda/gguf_mtp_weights.py), [SM70 MTP weights](../../src/tensorfold/families/qwen4_exp/cuda/sm70_mtp_weights.py), [media conditioning](../../src/tensorfold/families/qwen4_exp/cuda/media_mtp.py) | Import/alias/recovery software controls exist. Required SC117/V100 MTP execution, full-prefix quality, acceptance, memory and performance remain unqualified. DFlash availability or explicit no-speculation cannot substitute for MTP delivery. | No newly qualified SC117/V100 MTP profile is effective here. |
| Separately selected Inco DFlash2 | Exact trained pairing, TP8 HC conditioning, finite-range precision, rollback/sampling and text/visual gates | [manifest](../../src/tensorfold/families/qwen4_exp/cuda/dflash2_manifest.py), [runtime](../../src/tensorfold/families/qwen4_exp/cuda/dflash2_runtime.py), [verification](../../src/tensorfold/families/qwen4_exp/cuda/dflash2_verify.py) | A bounded four-row source runtime and controls exist; block-eight serving is not implied. Actual SC117/V100/TP8 quality, finite ranges, capacity and useful performance remain required under Section14. | No newly qualified SC117/V100 DFlash2 profile is effective here. |
| Image/video → text | Pinned processor/vision/positions/PLE and bounded mixed-work staging with separate method gates | [media runtime](../../src/tensorfold/families/qwen4_exp/cuda/media_runtime.py), [media MTP](../../src/tensorfold/families/qwen4_exp/cuda/media_mtp.py), V100 media foundation | Source/installed CPU/media authority milestones do not qualify actual SC117/mmproj/V100 fidelity, maximum work or sustained mixed-load service. Audio/media generation is outside this contract. | No new SC117/V100 media envelope is effective here. |
| Prepared plans and no-restart control | Fixed maximum envelope, ready-work/prepared-plan selection, FIFO/prefix/tiered state and supported in-process administration | [adaptive controller](../../src/tensorfold/families/qwen4_exp/cuda/adaptive_controller.py), [serving profiles](../../src/tensorfold/families/qwen4_exp/cuda/serving_profiles.py), [profile runtime](../../src/tensorfold/families/qwen4_exp/cuda/serving_runtime.py), retained catalog/host storage | Software tests and bounded source mechanisms exist. Actual burst/load/fairness/backpressure/recovery/resource and no-restart target gates remain required. Online fits, graph replacement during ordinary adaptation and live migration are not assumed requirements. | Only previously admitted profiles retain their effective controls; no new target transition is certified. |
| Follow-on models/formats/devices | Complete concrete family/recipe/backend/API/state qualification after the SC117 priority, while preserving selected profiles | Existing family implementations and [recipes](../recipes/README.md) are starting points | User-requested labels are not verified artifact identities. No universal model/format/quantization support, inherited SC117 geometry or new-device qualification is claimed. | No new model/format/device profile is activated by this handoff. |

## Reference teacher capture and replay

The canonical owned-model operation now has one actual SC117/V100 reference
capture: all eight ranks materialized 24 Q8 States with logical capacity 262,144 and the
physical anchor, NONE/reference/no graphs. One cold 52-ID completion returned
`372` against fixed independent gold and sealed four complete finite FP32
vocabulary frames. All eight startup/native/positive closure plus CLI/Host,
asset/cache, parent no-CUDA and retirement checks passed. The [actual receipt](../benchmarking.md#first-actual-reference-teacher-capture)
retains exact source/image/input/native identities and limitations. One matched
four-frame replay also passed all operation and closure gates on clean source
`3e582361`; the separate maintained comparison matched all 993,280 logits
bit-for-bit and all original token selections across the four steps. The
comparison tool has 132 passing current x86 CPU controls.
MTP/media/full-history quality, all-active capacity, resources and P3 remain
open; no production or candidate-family default is selected by this capture.

## Ordinary HTTP accounting checkpoint

The source binds a narrowly eligible plain-ByteLevel/FIFO HTTP payload region
to actual scheduler, logical-request and resident-model owners. Completed
handler roots, cache copies and pinned CPython C-encoder storage remain distinct
from logical generation credit. Physical/cgroup admission, full model context,
MTP, 96 MiB generation bodies and 256 MiB default cache capacity are preserved.
A certified encoder allocation failure poisons the owning Server and uses the
existing cohort retirement notification; subsequent encoding is refused.
Profile-managed Apps retain conservative accounting across initial admission and
profile staging; administrative functionality is unchanged. Other regions also
retain conservative accounting. The installed `d48e96a1`/`f4cbd945` successor
then served one controlled HTTP chat request successfully with configured
262,144-token context, TP8/PP1/ETP4 and shared-Q8 MTP depth 6, and retired normally.
This establishes that diagnostic's request and ownership lifecycle; full quality,
capacity, performance and private owned-native profile qualification remain open.
The subsequent source default is depth 4, preserving explicit overrides; that new
default is not relabeled as the successful depth 6 run. See [scope, failure lifetime
and remaining gates](../v100-adaptive-validation.md#ordinary-http-host-envelope).

The canonical one-loaded-session pilot subsequently completed all 15 generation
exchanges with baseline-identical output IDs/messages/finishes, including two
known wrong answers. Five of seven semantic cases pass; cold serial/MTP parity
and SSE parity do not turn that into a quality pass. Three independent same-GGUF
CPU llama references match full IDs in both TensorFold modes and reproduce both
failures. The optional `TF_PREFILL_ROWS=256` setting is selected for the measured
eager development workflow after A-B-B-A startup controls; mean readiness fell
from 657.139 s to 172.268 s. Backend defaults, full logical context and all guards
are preserved; long-prompt inference performance and quality remain unqualified. See [the precise
pilot scope](../v100-adaptive-validation.md#prefill-row-development-pilot).

Fixed per-rank startup phase diagnostics now have one installed probe.
Existing `load_seconds` still ends before MTP
payload attachment; separate attachment/vocabulary intervals and surrounding
engine setup use monotonic host clocks. Diagnostics stay outside prepared
constructor options and numerical/capacity identities. Existing loading and
warmup milestones retain their meanings. The recorded rank 0 host intervals were
59.700 s before weights, 57.308 s enclosing weights call, 46.390 s target load,
10.862 s stock-MTP attach, 0.000310 s vocabulary bind and 6.815 s subsequent setup.
Loader subintervals are contained in the enclosing call, not additional costs
or GPU-completion timings. The actual gate is
`sm-v100-startup-timing-app-v1/execution-v1/startup-phase-timing-actual-gate-v1.json`,
SHA256 `26d0b0cfdd1baad0678867bfabf733888e27f6696dbb11c71b660f0e61edc0d7`.

A CPU profile of the actual prepared-directory caller measured 27.357 s including
profiling overhead: two target header opens plus one MTP open. Config JSON
dispatch itself took about 0.002 s. The minimal source candidate reuses the first
validated inventory for target-derived metadata; `Config.read` and the neutral
configuration API remain unchanged. This removes one duplicate target parse;
prepared JSON/sidecar authority and original mapping/RoPE/EOS logic are preserved.
Raw-entry profiling does not describe the normal JSON branch.

The inventory-only prepared CPU comparison produced six byte-identical
3,150,610-byte normalized plan/byteplan graphs (including owner aliases), SHA256
`c3e98d9e2e80744cc04d566ddc0a33a21e37aea0f057cd35fc3b30bad16cac08`.
Three samples per implementation observed model-plan means 16.0236→14.6616 s
and byteplan means 15.9619→14.3811 s. These are CPU observations, not end-to-end
startup gains. The gate is
`single-inventory-reuse-execution-v1/prepared-inventory-CPU-equality-gate-v1.json`,
SHA256 `45d0f7e7470b6ef2d9096fd56a76cf6493e521936e51a639ed49492e32443aee`.

The separately measured row-planning candidate replaces expanded contiguous
expert down-row lists with ranges and gives unit-step ranges an exact run
representation. Arbitrary permutations, duplicates, descending/strided inputs,
gate/up selection and matrix bounds retain their existing behavior. Its own
three prepared CPU descriptor/byteplan graphs match that same full SHA against
the inventory-only reference. Observed means were:

| CPU operation | Original `4c351a70` | Inventory-only | Combined compact-row |
| --- | ---: | ---: | ---: |
| Model plan | 16.0236 s | 14.6616 s | 6.1140 s |
| Byteplan | 15.9619 s | 14.3811 s | 5.5880 s |

The combined gate is
`combined-inventory-range-execution-v1/combined-prepared-CPU-gate-v1.json`, SHA256
`7b4f7f3af5c1bde221a216e57fc8e7d22e9322156b0ae91e3148700a168d6998`.
Only three new combined trials were needed; the captured inventory-only
reference was reused. User steering now ends this startup-optimization phase;
no dedicated startup benchmark is scheduled. The private native functional run
below exercised this planner checkpoint. No integrated startup gain is claimed, and
no packing, payload I/O, cache or numerical-kernel change is included.

## Required SC117 parallelism matrix

The source private owned-routed procedure now uses the maintained TP supervisor
and ordinary CLI/frontend/API, with a typed Python-only capability and no public
flag or hot toggle. Its post-construction boundary requires actual SC117 banks,
native policy, reconciled arenas and coordinate/State numerical identity before
all ranks may publish readiness. Receipt/image/tool provenance and qualification
scope stay separate. The actual SC117 private run now preserved all 15 selected
reference outputs/IDs/messages/finishes/SHAs, including native cold-serial/MTP and
SSE parity. Known quality remains five of seven; broader quality/capacity and
performance gates remain open. The default reference command/policy is unchanged.
See [the exact functional scope](../v100-adaptive-validation.md#private-native-owned-sc117-functional-milestone).
See [the guarded procedure](../../deploy/v100/README.md#private-owned-routed-qualification-procedure).

With the same installed private image, automatic session/full-window selection
then computed 27 resident slots and all eight ranks materialized their full
262,149 cache extents for configured context 262,144 and default MTP depth 4.
Both arithmetic MTP/cold-serial smoke responses matched and retirement was clean.
H1/P1 remained in effect: this is resident allocation evidence, not simultaneous
27-request, populated near-full-history or throughput qualification. The solver's
next-count refusal is a resource-model result rather than a physical C28 test.
See [the allocation scope](../v100-adaptive-validation.md#automatic-resident-allocation-on-eight-v100s).

The ordinary unmanaged TP `NoRoom` path now retires completed operation-frame
payloads after acknowledgments while preserving the authentic error, queued
Stream children and retry semantics. Source CPU controls pass 302 cases with
zero skips, including GC-disabled leader/seven-peer lifetime checks and both
FIFO and legacy admission. Managed profiles keep their existing State-retirement
certifier. This source fix does not qualify the pending host-credit integration
or an installed/model run. See
[the scoped evidence](../v100-adaptive-validation.md#ordinary-tp-admission-error-lifetimes).

The installed SDK/native-component checkpoint uses source `e90caf58` and
immutable image `sha256:a24a579e3ffd2a1ead0def6b0d68947d94796135c00c946e8440ebe1f47c5015`.
The genuine final SM70 source build/export, protected installed software audit
and two runtime CPU tests passed. Normal guarded `gguf_ops` JIT compilation on
`sm-v100` GPU0 then passed 49 native ETP4 component cases with zero skips,
failures or errors. This advances the earlier offline-only evidence below;
it does not qualify distributed ETP4 or a complete SC117 profile. The subsequent
`104e6987` media producer change is a separate source checkpoint and is not
installed in that immutable image. Full-model quality, worst-rank capacity,
matched performance, media and compiler issue6111901 gates remain pending.
See [exact artifacts and scope](../v100-adaptive-validation.md#installed-sm70-sdk-and-native-etp4-component).

Current delivery priority is the one-server TP8/ETP4 profile, ahead of further
PP2 implementation. The existing reference path remains selected. An explicit
owned-routed native component now separates ETP4 width160 from shared-expert
TP8 width80, with device route mapping and caller-owned fixed scratch. It is
unselected: target/method peak admission, selected-SDK integrated build and
actual SM70 operator/model quality,
capacity and matched performance remain pending. The maintained V100 Compose
entrypoint forwards an explicitly set `TF_EXPERT_TENSOR_PARALLEL_SIZE`, leaving
unset/empty behavior intact. At the typed-component checkpoint, source and fresh installed CPU controls each passed
1,692 tests with one optional fixture skip; 782 installed runtime files match
the frozen source. These controls do not qualify native CUDA or the complete
TP8/ETP4 profile. See [the current component and qualification limits](../v100-adaptive-validation.md#tp8etp4-one-server-native-component).

Physical target access is now available on `sm-v100` (`192.168.1.140`). At
source `c0b11f0e` with the pinned `934497` builder image, standalone primitive,
library and synthetic reconvergence probes passed 40 single-device runs across
all eight V100s and 12 sanitizer runs on GPU0. These 52 jobs are component-fixture
evidence only; they do not execute the native expert operator, multi-rank NCCL
or SC117/MTP/DFlash models, and do not qualify the final SDK or resolve vendor
issue6111901. The full TP8/ETP4 matrix status remains unqualified. See
[physical component evidence](../v100-adaptive-validation.md#physical-standalone-component-campaign).

The subsequent communication component gate passed 42 CPU status controls and
both corrected physical four/eight-rank cases with no skips, using editable
Torch `04cf2557` and pinned NCCL 2.32.3 on `sm-v100`. The source UID captures held
`8db10092` plus fixture/documentation WIP, with unchanged application runtime.
Store provenance, word/status/error agreement and side-stream graph replay are
covered; the first fixture failure remains preserved. Native ETP4/model,
throughput/performance and final SDK qualification remain open. See
[communication evidence](../v100-adaptive-validation.md#collective-fixture-store-backend).

The private prepared-engine path now owns one native route arena per existing
target/prompt/MTP buffer, reuses their activation/output storage and binds
exclusive dispatch. Paid routing workspace contains these arenas; no extra
reservation or measured capacity gain is asserted. Rank agreement, state/cache
arithmetic identity and independently observed execution/coordinate facts bind
the native policy. The reference default remains unchanged. This private mode
passed scoped 49-module source and fresh installed CPU batches of 1,735 tests
each, with one optional fixture skip; all 783 installed runtime files match the
frozen source at the MTP/none checkpoint. The native candidate also has a
separate accepted offline SM70 compile/load result; no kernel was invoked.
Its DFlash composition binds validated native preparation inside the existing
voted pairing preflight and uses only target/prompt native arenas, retaining
independent dense TP8 draft resources. The expanded source CPU batch passed
1,850 tests with two optional large-artifact skips and 194 required controls.
Installed-package and native AMD64 checks are separate evidence. It still needs
actual SM70/model/resource/performance and native administrator-transition
qualification; it is not a released one-server
profile. See [prepared execution](../v100-adaptive-validation.md#prepared-tp8etp4-execution-and-workspace).

A4 adds mandatory basic physical-stage admission for canonical SM70 TP8
GPU-history startup: one kernel host, ordered complete rank coverage, distinct
canonical GPU UUIDs and the specified V100 SXM2 32GB/SM70 observations before
NCCL and model admission. All SM70 TP8 ranks use the common CPU Store envelope;
uniform legacy participation skips eligibility and gets no physical-stage
receipt. Source and fresh installed CPU batches each recorded 1,458 passes,
37 inherited fixture failures and one optional-fixture skip. Strict per-case
comparisons retain every failure as a failure. This does not establish the actual
mesh, full model, resident capacity or performance gates below. See the
[A4 artifact and proof scope](../v100-adaptive-validation.md#pp1-physical-stage-admission-a4).

The first full SC117 attempt stopped before weight loading because pinned
Torch exposes `_CUuuid` instead of a string/stdlib UUID. A source-only adapter
now converts only that exact trusted property type before strict canonical
validation. Existing str/stdlib UUID inputs, duplicate/rank/host/SKU guards and
the voted preflight remain unchanged. The original failure and separate GPU0
property diagnosis are retained. A later target attempt reached MTP warm-up but
failed when FP32 MoE slots entered generic Triton and its read-only default cache.
Triton 3.8's broad hardware matrix starts at CC8.0, but maintainers explicitly
retain best-effort pre-Ampere FMA support. Actual real-shape FP32 partial checks
pass the original fused kernel's exact oracle. The original dispatcher and
arithmetic are retained for a writable-cache-only readiness retry on the same
application. The separate-rounding Torch reference remains an independent
comparison, not a numerically equivalent replacement; its isolated probe latency
was higher, without an integrated model-performance claim. The unselected dispatch
patch is preserved as read-only evidence and removed from canonical runtime.
Persist `/cache/triton`, the native CPP cache and their metadata on the host;
3.2 is deferred as a separately qualified future version profile. No blanket
3.8 V100 or complete-model claim follows. Normal readiness and full-model
quality/capacity/performance remain pending. See [dispatch evidence](../v100-adaptive-validation.md#sm70-fp32-moe-partial-dispatch).

These are six required cases, not six qualified implementations. TP flags,
smaller-world tests, KV placement degrees or manifest metadata do not establish
expert ETP/pipeline execution. Each needs explicit target/MTP partition,
rank/layer/expert coverage, fixed-map state/graph parity, complete worst-rank
ledger and the governing benchmark matrix.

| Profile | Requested execution | Implemented starting point | Qualified | Effective |
| --- | --- | --- | --- | --- |
| TP8 × PP1 × ETP8 | One eight-GPU server | Existing all-expert/width80 arithmetic retained; prepared selection and all-rank agreement have A3 source/installed CPU controls. | Actual complete SC117/V100 profile unqualified here. | No new released profile established. |
| TP8 × PP1 × ETP4 | One eight-GPU server | A3 source/installed CPU controls cover EP2 banks of256 experts, width160, target/shared-GGUF MTP and owned-route reference execution. | Actual complete SC117/V100 profile unqualified here. | Pending this case's gates. |
| TP8 × PP1 × ETP2 | One eight-GPU server | A3 source/installed CPU controls cover EP4 banks of128 experts, width320, target/shared-GGUF MTP and owned-route reference execution. | Actual complete SC117/V100 profile unqualified here. | Pending this case's gates. |
| TP8 × PP2 × ETP8 | Two eight-GPU servers, one TP8 stage/server | A6 private 24/24 import/local execution and A7 private bounded two-leader HC transfer components exist; TP8 fanout, endpoint/method services, transactions and full admission remain missing. | Actual two-server profile unqualified here. | No PP2 serving path is activated. |
| TP8 × PP2 × ETP4 | Two eight-GPU servers | Stage-local EP2 import is exercised by A6 components; the complete pipeline and its resource proof remain required. | Actual two-server profile unqualified here. | No PP2 serving path is activated. |
| TP8 × PP2 × ETP2 | Two eight-GPU servers | Stage-local EP4 import is exercised by A6 components; the complete pipeline and its resource proof remain required. | Actual two-server profile unqualified here. | No PP2 serving path is activated. |

A3's source and fresh installed batches each recorded 1,217 passed, 29 inherited
fixture failures and one optional-fixture skip. All 29 failed cases have matching
nonempty diagnostics on the prior source reference; they remain failures, not
equivalence or target qualification. Physical stage binding, full-model numerical
quality, maximum capacity and performance gates remain open. See the
[A3 artifact identities and limits](../v100-adaptive-validation.md#pp1-expert-partitions-a3).

The private [A6 component boundary](../v100-adaptive-validation.md#private-pp2-stage-components-a6)
does not implement PP2 serving. It preserves full HC instead of a stream mean,
global block/PLE identity, local cache ordinals and actual stage coefficient
ownership. Required MTP/DFlash endpoint/features, one distributed round commit,
bounded progressing services and complete two-server admission are still open.
The scoped 42-module source and fresh installed batches each pass 1,600 tests
with one optional pinned-companion/target-metadata skip. These are CPU component
controls, including real stage-owned imports; they do not qualify actual
two-server transport, model quality, resident capacity or performance.

The private [A7 transfer component](../v100-adaptive-validation.md#private-two-leader-hc-transfer-a7)
owns a separate two-rank communicator/Store, bounded receive slots and exact
full-HC/position/token payloads. Generation-bound leases retain producer storage
through send completion and receiver storage through consumer completion.
Identity agreement and two alternating Store key banks bound rendezvous state;
fatal failures quarantine resources and notify an available local supervisor.
This component does not supply cross-server supervision, stage1 TP8 fanout,
progressing owner services or distributed speculative commit. CUDA/NCCL transfer,
resource overhead, actual hardware/model quality and performance are unqualified.
Source and fresh installed-image batches each pass 1,670 CPU tests with one
optional pinned-companion/target-metadata skip; 782 installed runtime files match
the frozen source. This does not qualify CUDA transfer or activate PP2 serving.

One-server release may proceed only after all three PP1 profiles and required
SC117 feature gates pass; PP2 remains required work pending its own hardware
qualification. ETP1 is excluded; TP4×PP2 is optional future study, not a current
release gate. Native262144 and extended-position commitments, every admissible
activity count and actual maximum, C1/2/4/8/16/24 milestones, >250 single-request
and >600 aggregate-at24 Q8 decode plus >3500 prefill objectives remain requested
measurement gates, not achieved figures or admission caps.

## Selected existing profiles and historical evidence

The selected GB10 Gemma profile remains source
`ae8d40bef288d9528aed2b45b06cffecb955e392788c240e89e8ebe45d148e6a`, image
`b579387bc49a092cd2a0b6f48a4c8f17a23995ce7191bd4d359e42c811c12684`,
N64/W4A16-head DFlash8/depth4, Shared512, Delta4/R16 and clipping ON, context131072,
BF16 KV, original model/drafter/template, vision, aliases and port. A8/proposal
graphs/trig/persistent cache remain off. It is not a SC117/V100 numerical profile.
The existing disclosed owner-accepted GB10 trade does not become a general
regression allowance or waive SC117 decode/capacity gates.

Private evidence anchors: `clip-sampled-cli-v1/production-closeout-v1.json`
SHA `050a0698a5d2b62853c22c273354612712fd3dbeb9db53cd25d7fb9460756301`;
installed650/exact777 receipt `installed-gate-v2.json`
SHA `d0d3c276613309d4146dc0bf7b9dc0044b0dea5cd3d9dab51c873dcc28db56cf`.
Latest recorded restoration is `query-tile8-operator-v1/production-restore-collect-v1.json`
SHA `7d735f9894eccf06abb23ef94746a4f626e8634213c7384f1566ed6db7539c76`,
Started2026-11-06T05:35:53.382963622Z, with API/room-one/math/aliases/pointer/speech
checks. This map reads those receipts; it does not claim a new live probe.
The full API suite's16 functional passes/failed17th cancellation-counter and
aggregate false/exit1 remain distinct from the separate observed-active
cancellation witness. See [GB10 qualification](../gemma4-cuda-validation.md).

QT8 is deferred/unqualified after the goal change: source/installed435 and
metadata/cache-admission controls are not GPU-word/lowering/timing evidence.
Its prototype/images/failures remain archived; default16 attention and original
tests were restored at the baseline. N128, fused-tail and other investigations
retain their recorded scoped dispositions, gains and costs. No benchmark is
renamed ProviderOne evidence merely because the project direction changed.
Other existing model/backend checkpoints in [README](../../README.md) and
[optimization validation](../optimization-validation.md) retain their own scope;
this initial map is not an exhaustive new qualification of every existing profile.

The historical5582-check/144-subtest milestones belong to ff188313/21a12df3 and
707 installed files, not to8d69 or a new ProviderOne build. Keep the
[V100 qualification record](../v100-adaptive-validation.md), its historical
[CPU receipt](../verification-v100-adaptive-cpu-20261026.json) and
[corrected-stack receipt](../verification-v100-sdk-rebind-cpu-20261026.json)
at their actual identities. Source existence, compilation, installed CPU proof,
physical allocation, readiness, exact model/state output, quality and full-load
performance remain separate statuses.

## Verification and next ownership boundary

Existing affected controls include `tests/test_v100_source_inventory.py`,
`test_v100_source_runtime.py`, `test_v100_source_install.py`, `test_v100_compose.py`,
`test_flashnext_gguf_mtp.py`, `test_flashnext_media_mtp.py`,
`test_flashnext_dflash2_manifest.py`, `test_flashnext_dflash2_verify.py`,
`test_flashnext_history_placement.py`, `test_flashnext_adaptive_controller.py`,
`test_serving_profile_manifest.py`, `test_retained_catalog.py`,
`test_responses_api.py` and `test_conversations_api.py`. They are procedure
pointers, not new pass claims. Follow [CONTRIBUTING](../../CONTRIBUTING.md) for
required deployment/source/static checks and the actual-target gates for a
changed runtime. No duplicate prose-hash test framework is introduced here;
authority uniqueness, unchanged input bytes, relative links and the exact staged
allowlist are checked directly at this checkpoint.

A1 implements the neutral artifact boundary under a bounded canonical source
lease following authority checkpoint `ff47150525dc55d8c791965beb8a894cc611541a`;
its verification remains separate from the earlier documentation checkpoint.
Coherent package/native/CLI/cache/deployment/updater rebrand remains a required
release boundary. No new framework, backend emulation or parallel legacy serving
stack is authorized by the handoff alone.
