# Adaptive V100 source implementation and qualification

The [implementation specification](v100-adaptive-inference-design.md) governs
the eight-V100 TP8 path. Historical results in
[the V100 qualification record](v100-flashnext-validation.md) identify their
original sources and layouts. They do not qualify the adaptive implementation,
unique histories, a different codec, or a new execution envelope.

This work is developed without the eight-V100 server. The selected SC117 GGUF
payloads and mmproj have been downloaded and verified; source/header/import
controls do not establish device execution. Source controls, pinned-stack compilation, installed
CPU checks, physical allocation, model correctness, quality, and measured
performance are separate gates. No new session maximum or throughput result is
claimed. The specification's 20-Q8/32-Q4 figures are geometry reference points,
not admission promises or capacity ceilings.

## Neutral artifact I/O boundary (A1)

The source now separates bounded container/file validation from model and device
selection. `tensorfold.tensor_file` is the unchanged shared stdlib file-boundary
implementation moved out of the CUDA package, with its owned callers updated.
`tensorfold.gguf_file` owns the existing GGUF formats, immutable inventories,
shard/extent validation, packed-byte spans and path resolution. Neither neutral
module imports a model family or accelerator runtime. Flash Next retains its
architecture guard, token-ID identity domain, semantic configuration and CPU
decode oracle. An opened inventory can pass directly to family configuration;
this avoids reading headers twice and preserves an explicitly selected shard
as the entrypoint.

Focused controls cover neutral inventory versus family admission, backend-blocked
imports, split-header read counts and the existing MTP inventory handoff. The current CUDA-hidden source batch recorded 1,228 passed,48 failed and four
skipped; all48 failures also occurred in the immutable baseline, and all new
boundary and corrected resolver cases passed. Static checks passed on 115 Python
files. This is partial source evidence, not an all-green suite: the existing
source failures remain historical evidence rather than an all-green result.

A fresh built-image check then recorded **1,264 passed, 12 failed and four
skipped** in 17.17 seconds. All five new boundary cases passed. The same 778
installed runtime-file hashes were present before and after the check, with no
runtime source overlay, neutral-module origins under the active purelib, and the
old `tensorfold.cuda.tensor_file` module absent. Native construction used the
current owned C source, exposing `memfd`; rebuilding the native helper resolved
36 stale-FD failures from the source-test SDK without changing source or tests.
CUDA remained uninitialized, and the source-foundation check exited zero.

The remaining 12 failures match the baseline causes: four fake-engine fixtures
lack `_sm70_import_plan`, one fake import blocker rejects the required
`request_limits` import, and seven cases reject the actual ARM64 SDK where their
native retained-payload contract requires the x86-64 Torch04cf2557/GCC13 ABI.
These remain **unqualified failures**, not passes or an equivalence claim. No
native ABI guard, test assertion or runtime check was weakened.

Exact installed evidence scope:

- Source checkpoint: `7be4925fec5eed762fe7b4b110be9289a8e29d0b`.
- Built source UID: `23eb2c5a3058ce73e709e348bde358ec06b5d6c49f5fc6d218db12b3c439da89`.
- Runtime image: `sha256:fce3dc00acc4c5aab8de1a1d26bd57c8be120705f097c4d3ce04169723dac53b`.
- Verification image: `sha256:c2d6a9ccd4acbdbb92a119e2f4d1f4527033567690d6923b95a8e141badd2421`.
- Active package origin: `/opt/tensorfold-venv/lib/python3.12/site-packages/tensorfold/__init__.py`.
- Native C source SHA256: `f85d2c923adfaa14e819250e8def3c3166b8c8624295a37572916e62ad20c1f2`.
- Installed gate SHA256: `b77dcc5a5fc3cfbd7918e1cecd116f271687210dc381de29f3a02e5f34c8566b`; per-cause triage SHA256: `0f6f5fbe95015f6c3fb95c7c4d8115c5ecf8a510c73c0818e86ea0ca8fd745e1`.

The gate and triage are retained in the private
`providerone-alignment-20261106` audit packet. This qualifies the affected
installed I/O boundary only. It adds no format, supported model, device
execution, capacity, performance or physical V100/SM70 qualification claim.
Selected GB10 and other existing profiles retain their original evidence scope;
production was unchanged. Earlier source runs and failures remain preserved.

## Current v6 source and CPU evidence

Current source implements the standard Responses and durable Conversations
lifecycle, bounded FIFO admission and live input/output credits, the common
prefix/retained-state catalog, RAM save/restore, and blocked-request requeue.
Logical API records, reusable computation and guaranteed retained checkpoints
have separate ownership and resource claims. See the [API contract](api.md#durable-conversations-and-quotas)
and [fixed serving policy](serving-profiles.md#fixed-endpoint-fifo-and-retention-policy).
These implemented paths have CPU software controls; they are not a qualified
V100 serving configuration.

The published composed installed CPU checkpoint uses frozen source tree
`ff1883131180963a1de18bb00c74d6777466dcb9` and local image ID `sha256:fb0ac861483f0bc14bef1f73535d07d358c196c7e9358f76da3390f1434813a8`. It passed
**5582 tests and 144 subtests across 183 modules**, with
9 explicitly documented skips. All **707 installed code files** matched
the frozen source; the 2,012 captured inputs and external frontend identities
were verified without a source overlay or within-run drift. The final receipt
records the exact result, skip disposition and immutable image identity:
[CPU software receipt](verification-v100-adaptive-cpu-20261026.json),
application commit `7d3ddbb9d066fe7451769ad9a36739619ce2d618`. The installed pytest run took
551.56 seconds.

The composed source includes the canonical key/value wake relay, actual owned
synchronous Store provenance for every new unique profile (no explicit
retention, GPU tier or RAM tier), and process-fixed resource highwaters.
That checkpoint's payload contract binds CPython 3.12.3, owned Torch
`5ab14acbced0c33e6b6e8fdbc5b541b57487f7e1`, GCC 13 build/ABI and the recorded
GCC 14 runtime content. State and hard file-descriptor lifetime reservations,
native/Python transfer and control owners, source field guards and host
prepayment are implemented and covered by the source/CPU evidence. Generic
legacy provider defaults retain their separate contracts.

The final resource producer passed **236 CPU checks**, with zero skips and
Ruff clean. Earlier scoped evidence includes 50 copy-applicability checks,
138 caller checks plus three reclamation controls, and Python/RPC last-use
and resource controls. These suites
overlap the composed gate and are not summed into its installed total.

The generic per-byte graph projection at the fixed 96-MiB RPC cap is
conservative and can refuse a host configuration. That is a refusal of the
declared projected envelope, not a measured physical capacity limit or proof
that a practical workload requires that much RAM. Tighter schema-specific
bounds remain a separately assessed candidate; no fitted margin was selected.

The previous native SDK has a retained failure-path P1 finding.
On a supported `getnameinfo` failure, its endpoint formatter casts the
address of its pointer variable instead of the pointed-to socket address;
the IPv6 path accesses beyond that pointer object. The same failure path has
an unsynchronized shared flag. Native CPU fault injection reproduced corrupted
endpoint diagnostics while the isolated data round trip succeeded; no crash or
instrumented race was claimed. Owned-FD/nonce authority does not use those
formatted strings. Passing that application's CPU gate did not repair the
native P1 violation.

Current source selects corrected owned Torch
`04cf2557f315a350e7eea7ab064288def65db75a` and refuses the older 5ab revision
for adaptive native-payload promises. The [source foundation record](../deploy/v100/source-foundation.md)
identifies its immutable SDK/wheel and native CPU controls. Direct IPv4/IPv6
formatter/concurrency controls, actual-wheel IPv4 normal/forced endpoint and
Store controls, and the immutable SDK's actual ABI/runtime-content checks pass.
Actual IPv6 Store execution is unrun on the IPv6-disabled development host.
The only source changes since 5ab are the internal socket correction and native
test wiring; public headers, Store layouts, allocation equations and transport
defaults are unchanged. This structural reuse preserves the exact CPython,
compiler/ABI, loaded-runtime and finite hard-FD guards.

The fresh corrected-stack application is commit
`2c0b4c4a51fbf2f8370fcccd2980c3e3a77ede2b`, frozen tree
`21a12df376313d5e95bbae5e269b32ae31ad2d14`, local image ID
`sha256:e99c7284234376f43634c7103fcd6fe313fee4b61c073ad79965a100b2b1baef`.
It independently passed **5,582 installed tests and 144 subtests across 183
modules**, with nine documented skips in 559.91 seconds. All 707 normal
site-packages code files and all 2,024 archive inputs (2,013 base scoped inputs)
matched before/after, with no source overlay. Its client consumer supplies
OpenAI SDK 3.37 and xgrammar 0.2.8 on the actual corrected source/media stack.
The [corrected CPU receipt](verification-v100-sdk-rebind-cpu-20261026.json)
records full identities, raw-result hashes and skip disposition. The earlier
5,582 result remains separate historical evidence; it was not transferred.

The e99 checkpoint's target source audit exposed a project identity policy
issue: the framework's configuration display includes runtime CPU
dispatch (development AVX2 versus target AVX512) and conditional CUDA sections.
The owned immutable portable-policy successor separates stable build identity
from fresh host observations without rewriting the original sealed image.

That application's commit `fc5e6dd4`, tree `7bdd8392`, image `f8f66a9d`, independently
passed **5,707 installed tests and 144 subtests across 187 modules**, with nine
documented skips in 555.98 seconds. All 707 code files, 714 package source/data
files plus the compiled FD extension, and 2,028 fixture inputs matched before
and after, without a runtime source overlay. The mapping-policy successor at
`a39646d1` changes only the audit producer/policy, tests and documentation. Image
`6daae245` retains the exact application wheel SHA256
`805246a27273b16948d984ecf0072fe71705ecad605b02a1e952eb8810718e60`.
It passed 66 affected controls and actual immutable-image source/header/native/
combined-media checks around a native 2-MiB shared mapping. The
[portable-policy receipt](verification-v100-portable-policy-cpu-20261027.json)
preserves both result scopes; the full 187-module gate was run on f8, not 6da.
Actual successor target acceptance after GPU work and all broader model/resource/
quality/performance gates remain separate and pending.

The attention descriptor successor is held commit `e3abd47a`, integrated while
preserving the published policy documentation. Its new whole application wheel
`6fac3438` and immutable image `96f69617` change one of 707 code files; 706 are
byte-identical to the full-187 baseline. Normal-installed checks passed 116
attention/capacity/policy/execution-identity cases and 24 subtests in 11.88 s,
plus 33 actual new-FD ABI/I/O/snapshot/listener cases and five subtests in 3.24 s.
All 715 package records and 46 protected metadata files matched before/after.
The new FD binary has its own identity and actual tests; it is not labelled the
old binary. The [affected-image receipt](verification-v100-attention-runtime-cpu-20261027.json)
records these scoped results separately from the older full 187-module run.
The fix avoids legacy wheel metadata only when that private attention candidate
is inapplicable; it preserves source04cf ineligibility and real math/workspace/
device settings. Actual model/output/numerical/quality/resource/performance
qualification remains separately owned.

The logical/physical State successor, held commit `64e3ed7f`, keeps the admitted
logical context separate from physical speculative guard rows. The initial
target remains logically 262,144 rows with 262,147 physical cache rows; physical
allocations, resource accounting and position guards are preserved. Whole wheel
`b9839482` and image `a19233f2` passed three affected normal-installed suites:
158 constructor/logical/retention/indexer/prefix checks in 10.63 s, 1,177 State/
storage checks in 32.24 s, and 231 engine/Multi/profile/position/media caller
checks in 21.85 s. All 715 package records and 46 protected files remained
unchanged. The [logical-context receipt](verification-v100-logical-context-cpu-20261027.json)
retains the original 72 fixture failures and their justified three-fixture
correction; the application wheel/image did not change for that fixture repair.
This is a new affected-source result, with no full 187-module rerun claim.
The target passed the original State row-domain failure point but later
startup still failed before readiness. That failure remains under primary
investigation; this CPU checkpoint supplies no OOM or model-readiness diagnosis.

The subsequent startup-order source `a1523f66` defers public Scheduler/FIFO
configuration until constructor synthetic warmup completes. Early TP/control
ownership, managed/follower transitions, rollback and real public principal/
ticket/admission validation remain unchanged; no synthetic credentials are added.
Whole wheel `86a04dee` and image `30fda46a` passed 68 direct startup/FIFO/TP
controls. A lifetime AST fixture required the actual `context_window` property
and logical 2,048 plan while retaining physical 2,053/depth four and all original
assertions. Its 15-test/two-subtest leaf reran successfully; 23 nonoverlapping
additional cases retain their separate earlier pass. The original five fixture
failures remain preserved. The [startup-order receipt](verification-v100-startup-order-cpu-20261027.json)
binds all 715 package records and protected/configuration checks without a new
full 187-module claim. A live whole-wheel diagnostic passed the previous rank-0
failure boundary, with all eight ranks joining target input preparation; the
reported model state remained loading, with no health or complete-output
acceptance claimed.

NVIDIA compiler issue **6111901** remains a separate release gate; see the
[source-foundation contract](../deploy/v100/source-foundation.md#compiler-issue-6111901-remains-a-release-gate).
Actual V100/TP8 CUDA/NCCL execution, model output/numerics/quality, physical
capacity, allocator/RSS/NUMA and whole-workload performance remain unqualified
for this source. Native IO wall deadlines are not inferred from Store wait
timeouts. No new session maximum, throughput result or V100 release is claimed.

The previous consolidated candidate was tree
`413e132bf977625f485fffedf2011cc4bfc02f07`, image
`sha256:05eca41e61a9410660d2a38c2c4d9ced06124efec64d635cb21580bcbb399323`:
5,214 tests, 144 subtests, nine documented skips, 156 modules and 706 matched
installed code files. Its skips covered two MLX modules, an optional Qwen
tokenizer, optional exact DFlash/MTP artifacts and four Docker Compose controls.
It remains distinct from the published media/frontend milestone below.

An intermediate installed attempt used tree
`045446b572d9be7a8aa08ef9f1fde125e77b38f8`, image
`sha256:c903e72e3edb5e50bf26bac22acdd697fa444844b90dd639a56ddf48a69a5fe3`.
It retained 16 failures, 5,564 passes, nine skips and 144 subtests: a shared test
Store returned strings where the actual native contract returns UTF8 bytes.
Only that fixture producer and its regression changed; shipping strict ACK
validation and all original lifecycle/error assertions were preserved. The
negative attempt remains separate from the successful successor receipt.

## Historical text/core installed CPU checkpoint

The installed application built from `0f656e9ab9642af6fa19210d721de13d375ea51d`
passed 3,277 tests and 105 subtests across the 78 affected test files, with four
documented optional/environment skips. The only later fixture change at
`b56298202ebe14b7fb1533ccb9f3871009f94d4c` declares the owner-layer reference's
existing unbound context explicitly; numerical assertions remain unchanged.
The image is
`sha256:9dc7ada61e4c1fdf272d218936ec8a08f3e372f40bec1dab53cae9237970bbc9`.
The source-built dependency environment was unchanged; this was an application
wheel rebuild and ordinary installed-package test, with no source-code overlay.

Coverage includes compact raw/pool transactions, issued coordinate/frame
identity, MTP conditioning/recovery, source authority, native/YaRN preparation,
Engine/CLI and serving-profile lifecycle, bounded admission/output, cancellation,
and the retained legacy CPU references. Corrections preserve primary failures,
guard complete model/profile identity, snapshot overlapping MTP inputs before
writes and poison a bound State after failed restore mutations. Separate cleanup
and policy source reviews are closed for this checkpoint.

This is a software/CPU result, not GPU kernel, collective, model-output,
quantization-quality, physical-residency or performance qualification. Native
SM70 compilation retains its separately recorded exact-source scope. That
snapshot preceded the installed media/frontend milestone below; it does not
replace the current source and qualification boundary above.

## Installed media and exact frontend integration

The application at `caf1eeda`, merged to `python-0.6` as `64b24dd5`, passed
4,365 installed CPU tests and 124 subtests across 111 affected test modules.
Four documented optional/environment cases were skipped. The ordinary wheel
image is `sha256:30e7036aa0d23ed4bb203f68a2d6e5f34cda617d06bf5cc2a8297d60080ef2bc`;
all 694 installed code files matched immutable source tree
`72aa4e585f2747ae02da083bb7a0d5958358212a`, with no source overlay.
[The maintained receipt](verification-sm70-media-integration-20261026.json)
records provenance and limitations.

Coverage includes the actual bounded CPU media work/TP input/App ownership path,
request-local decoder failure retirement, target/MTP shared input spans, complete
model token-ID domains, CONTROL presentation, nonlexical UNUSED rows, canonical
32-core MTP import/shared target vocabulary weights, frame/recovery containment,
principal request ownership and side-stream graph warmup ordering. CPU cohorts
and import controls do not qualify CUDA collectives or model outputs.

Current v6 FIFO admission, complete RAM retention/prefix reuse, standard durable
API lifecycle, host/device resource admission and automatic blocked-request
requeue are later integration work. This checkpoint is neither V100 release
qualification nor a capacity or throughput claim.

## Fixed initial unique history (A2)

The selected adaptive Flash Next TP8/two-KV-head preparation now constructs an
explicit degree-four `HistoryPlacement`. That same selected map drives the
planner, the published weight metadata and each State's ownership. Generic
degree-one/two/four helpers remain available to isolated research/oracles; their
internal default is not the serving selection. MTP retains its separately fixed
four-way striping, and six-layer stages retain their actual worst-rank/tail skew
rather than inheriting a twelve-layer balance assumption.

The resource ledger retains `migration_overlap_limit` with value zero. This
removes only the dedicated live-migration liability; prefix/COW, idle recovery,
restore, replacement, media and other workspace charges remain independent.
Ordinary Multi initialization does not enable migration. Initial runtime
`attention_degree` requests refuse before publication or scheduler wake without
changing active/retained state; bounded execution controls remain available.
The optional transfer algorithms and independent failure/partial-progress oracles
are preserved, without an opt-in serving flag.

The affected source and fresh installed-image controls passed. Each broader
batch recorded **367 passed, four failed and 12 skipped**; the installed batch
took 9.97 seconds and exited 1. All eleven affected test modules and the new A2 cases
had no failed nodes. Required deployment/public-permission checks passed 194
cases, and static checks passed on the seven affected Python files. The four
failures are unchanged dedicated target checks: three require the selected
x86-64 Torch04cf2557/GCC13 ABI, and one requires tokenizers 0.23.3. The 12 skips
are frontend-version gates. These remain unqualified target gaps, not an
all-green suite or an equivalence claim; their guards and tests were not relaxed.

The twelve synthetic constructor geometry cases isolate only
`retention_resources.require_native_payload_runtime`, supplying an explicitly
synthetic finite `hard_nofile=1024`. All real headers, native payload equations,
other ledger terms and device-floor assertions remain exercised. The shipping
ABI guard and dedicated actual-ABI tests are byte-unchanged; these unit tests
do not certify an ABI. Actual CPU State constructors exercise prepared fixed
metadata, while the real constructor observer checks the selected planner.
Independent rank/page/tail controls cover both six- and twelve-target-layer
maps and separate MTP ownership. Ordinary Multi initialization and rejected
runtime requests are exercised without activating migration or publishing a wake.

Fresh installed evidence uses source checkpoint
`d4469a679cdd7d9286298ec282c7d6405c06b6b9` and built source UID
`47e568407b78fbcb718da872ccd86f53252c157c301c5d9bd99c38f97873445d`, runtime
`sha256:645aa9af855129ec459a3676ff2a74e27496c7311c3a8ac0de929a5cc5165ce2`
and verification image
`sha256:f5503d631aeb101112bc5f3ed719a22c88dc1a48bf0685f13cf4adb8b7926fcf`.
All 778 installed runtime-file hashes matched before and after the check, with
no runtime source overlay. Changed Engine, Multi and server-runtime module
origins were under `/opt/tensorfold-venv/lib/python3.12/site-packages/tensorfold/`;
the current native helper was freshly built from C-source SHA256
`f85d2c923adfaa14e819250e8def3c3166b8c8624295a37572916e62ad20c1f2`.
The installed gate SHA256 is
`88603b0a4714c2462cfb1becd2b017f59b9626471211389d7f75b04244d23320`.
CUDA stayed uninitialized and the
source-foundation check exited zero. The exact logs and earlier baseline
failures remain in the private `providerone-alignment-20261106/fixed-history-v1`
audit packet; historical passes are not relabeled as current target evidence.

This does not qualify physical V100/SM70 allocations, full-model state/quality,
capacity, latency or throughput. Selected existing GB10 profiles and production
are unchanged.

## PP1 expert partitions (A3)

Source and fresh installed CPU controls cover the one-server expert-partition implementation. Preparation selects
ETP8/4/2 inside TP8, deriving EP1/2/4. Fixed contiguous groups own 512/256/128
SC117 experts, with intermediate shards of 80/160/320 values. Target and
shared-GGUF MoE MTP use the same partition; nonexpert operators and the shared
expert remain TP8. Packed source coefficients retain their original encodings
and partial-block offsets. A rank stores only its group's routed banks.

New ETP4/2 reference execution filters global routes before local bank lookup,
initializes nonowner outputs to zero, and uses the existing full-TP rank-ordered
FP32 gather/consumer. Empty routed groups retain collective participation.
The shared expert remains independent of routed EP ownership. Owned-row
compaction and scalar validation include device synchronization and dynamic
temporary allocations; these are reference implementation costs, not measured
speedups. The resident, decode/prefill, remapping and companion ledgers must
cover the selected geometry. Degree-specific floating-point reductions require
their own numerical and full-model quality qualification.

The public startup option is `--expert-tensor-parallel-size 8|4|2` for local
Flash Next GGUF, CUDA TP8, resident banks and GPU histories. The Python family
and engine option is `expert_tensor_parallel_size`. Omission preserves existing
profiles. Explicit ETP1 is refused; the separate rank-zero/RAM expert-follower
mode does not implement this profile. No PP2 flag or pipeline support is added.
PP2 remains required work with separate stage, transport and transaction gates.

The existing startup cohort agreement now includes effective ETP, global expert
count and full routed width, so default/explicit ETP8 agree and mixed degrees
refuse before model admission. Computed-state identities distinguish the
degree-specific reduction policy from physical rank placement. Prepared manifests
validate the new selector and expert coverage before retiring the current model;
ordinary runtime execution updates cannot change this structural choice.

Source and fresh installed batches each recorded **1,217 passed, 29 failed and
one skipped** across 26 modules. The source batch took 22.73 seconds; the installed
batch took 22.95 seconds. Both exited 1. Every failed case has matching, nonempty
exception diagnostics on the unchanged `0db3f178` reference: seven synthetic
provider fixtures are refused by the existing FP16-GGUF guard, ten by the compact
Planar/Iso4 provider guard, and twelve older parallel fixtures lack a `positions`
attribute. These are inherited fixture failures, not passes, equivalence or
hardware failures. The skip requires optional pinned companion/actual target
metadata that was not supplied. Required deployment/public-permission controls
passed 194 cases; static checks passed on all 22 affected Python files.

The bounded resource repair passed 182 tests with one optional-fixture skip.
Prefill attachment, admission and loading now bind the provider's shared-width
contract: a partitioned in-place plan cannot understate the legacy provider's
full-output-copy workspace merely because routed widths match. Prepared-profile
controls passed 120 cases, including early refusal of invalid ETP degrees and
expert coverage. The existing AST wiring unit reads the active Engine module;
its final four cases passed separately after that test-only origin correction.
These overlapping batches are not summed into a unique-test total.

Fresh artifacts use source checkpoint
`7e4e8cdc0a8e01ec03ab5c43afcc7a8844c85681`, source UID
`4362794f2ebaa3f62ea79e0eb410394fc97b9223bd20aa67d472937edb10d651`, runtime
`sha256:5cea184aeb7c298c767bdabba5dfaf7d8826b0fd9badbd8014ba090a797006b3`
and verification image
`sha256:8d47780d6f48aa7f4ec9e41aafae8d2407411e6a371398c3a47faa517ac1996e`.
All 779 installed runtime-file hashes matched before and after, and all sixteen
changed module origins were under installed `site-packages`. There was no runtime
source overlay. The current native helper's C source identity and memfd interface
were checked; CUDA remained uninitialized and the source-foundation audit exited
zero. The installed gate and per-case source/reference/installed comparisons are
retained in `providerone-alignment-20261106/pp1-etp-v1`. The earlier empty-array
diagnostic-parser artifact is explicitly invalid and preserved as failure
evidence; the corrected comparisons require nonempty diagnostics for every case.
The installed gate SHA256 is
`46147f1f326b7bcce4eaecc8e8e9c1bacaae64d1f2532a83452e4a62ae48cb18`;
the strict source/reference/installed cause comparison SHA256 is
`435a8e7369faecfb97f121c7958149f729497a768bad30d461c7717b430cfdb1`.

Actual eight-V100 topology, imported allocations, full-model numerical/quality,
long-context capacity, complete-serving performance and all PP2 gates remain
unqualified. No gain or released target/MTP profile is claimed. Production is
unchanged.

## PP1 physical stage admission (A4)

Canonical SM70 TP8 GPU-history startup now checks the physical assignment before
initializing NCCL or admitting/loading model weights. The default gate observes
device zero in each rank process and requires one host kernel, ordered complete
eight-rank coverage, canonical distinct GPU UUIDs, the specified V100 SXM2 32GB
SKU and SM70 capability. Reported positive memory bytes are observed directly;
the nominal SKU is not turned into an arbitrary usable-memory floor.

All SM70 TP8 ranks enter the same bounded CPU Store preflight, independently of
their local canonical/legacy admission choice. Mixed participation refuses
before ranks can split into CPU and NCCL waits. Uniform legacy participation
skips physical eligibility and returns no stage receipt. Smaller-world and
selected non-SM70 implementations keep their existing paths. The GPU-history
target has no new flag to disable the guard or silently fall back to legacy.

The existing startup-owned Store, agreed refusal/retirement phases and managed
cohort containment are reused. Capture errors are published before dependent
waits; malformed/oversized transport and uncertain cleanup are fatal. A semantic
refusal whose key retirement fails now preserves its primary failure, cleanup
failure and retained owner while marking the stage/draft/media preparation
cohort unsafe. No new listener, communicator or persistent key registry is
introduced. Managed replacements borrow their existing Store ownership.

Two phases plus retirement have at most 32 keys; even the existing bootstrap
allowance fits the current native key budget. Frames are bounded to 8 KiB before
decode, below the existing constructor value allowance. Expensive checkpoint
hashing, runtime inventory and topology commands remain opt-in. The optional
execution-profile path shares the canonical UUID/rank-map validator while
retaining its existing shared-file/container identity scope.

The default physical-host fingerprint uses the Linux kernel boot ID, not a
container hostname or machine-id. Two actual Docker containers on this GB10
host produced the same fingerprint with different machine-id bindings. Runtimes
that virtualize that kernel input need a separately qualified binding. The
receipt exposes fingerprints rather than raw boot/GPU IDs and remains separate
from the numerical identity used for prefix/copy/restore compatibility.

Source and fresh installed batches each recorded **1,458 passed, 37 failed and
one skipped** across 36 modules. Source took 28.90 seconds and installed took
29.61 seconds; both exited 1. All 37 failed cases have matching, nonempty
per-case diagnostics on pre-change references: the 29 previously disclosed
provider/`positions` fixtures, seven older graph fixtures missing `comm`, and
one native-media fixture that omits architecture metadata and patches an old
resolver owner. That media failure does not demonstrate a runtime mmproj
selection bug. These remain failures, not passes or equivalence. The optional
skip requires pinned companion/actual target metadata that was not supplied.

Focused controls passed 221 cases, including three actual CPU Store exchanges
for canonical, legacy and mixed participation, and six cleanup/refusal cases.
The Store exercise uses this host's SDK and test-only wait deadlines; it does
not certify the selected x86 native layout. Required deployment controls passed
194 cases and static checks passed on eight affected Python files. The focused
and integrated batches overlap and are not added into a unique-test total.

Fresh artifacts use source checkpoint
`14c2d24f929523c2e96eb5bb25e18f2a11b4e50c`, source UID
`fde67a33cde7625f6b2e4571a77b4ca878140937ce9a6eed48baee83ad5b2034`, runtime
`sha256:ad2e0f366a89bff6f8d7dfa3479773d1fb15f4fdb3b370a5f305a32199ae7c91`
and verification image
`sha256:0d658a6a412710958722a8495509ef7c3aea317f069083e22df1dae9b8c2e184`.
All 780 installed runtime-file hashes matched before and after; all four changed
module origins were installed `site-packages`. There was no runtime source
overlay. Current native C identity/memfd checks passed, CUDA stayed uninitialized
and the source-foundation audit exited zero. The exact logs, references and
scope are retained in `providerone-alignment-20261106/pp1-stage-binding-v1`.
Installed gate SHA256:
`18f321f8116df6c898e55249e692b3596df709cbacf43f4db1e3badac59c6ed7`;
strict source/reference/installed failure comparison SHA256:
`bb0703a89c02d82fa232aff35e76df3337e4eb8d64527c64dd768dc175edb0c7`.

Physical-target qualification remains pending. CPU metadata/protocol controls
and current-host CPU Store exercises do not
qualify the selected x86 ABI, eight actual V100s, the NVLink/P2P/PCIe/NUMA mesh,
complete model numerics, capacity or throughput. Unmanaged launches retain their
existing external failure-containment responsibility. Production is unchanged.

## Restored fixture contracts (A5)

Four stale fixture definitions accounted for the 37 inherited failures above.
The repair changes test inputs, not runtime behavior or guards:

- Provider-format controls now use a real tiny GGUF target instead of patching
  an obsolete family resolver alias. Their existing early template sentinel is
  retained; three added negatives confirm SM70-only formats still reject a
  non-GGUF provider before reaching that sentinel.
- Allocation-only weights explicitly declare `positions=None`, matching the
  actual unbound-position interface. All allocation, ownership and budget
  assertions remain exercised.
- Single-rank graph fixtures explicitly declare `comm=None`. The real replay
  wrapper, event ordering, primary-error and no-successful-submission checks stay.
- The media fixture patches the actual neutral resolver and supplies the
  required `qwen4exp` architecture metadata. The real family architecture guard
  runs, the selected entrypoint is asserted, and vocabulary/role plus stale
  tokenizer-override checks remain intact.

All 115 original assertion nodes are retained. The affected four-module batch
passed 563 cases in 6.27 seconds. The complete scoped 36-module source batch
passed **1,498 tests with one optional skip** in 27.47 seconds; the installed
batch passed the same cases in 27.32 seconds. Both exited zero. The skip still
requires optional pinned companion/actual target metadata that was not supplied.
Required deployment/public-permission checks passed 194 cases, and static
checks passed on all four changed Python files. Overlapping batches are not
summed into a unique-test total.

All 783 runtime-source files remain byte-identical to the A4 built capture.
No rebuild or redeployment was performed. Installed verification reused runtime
`sha256:ad2e0f366a89bff6f8d7dfa3479773d1fb15f4fdb3b370a5f305a32199ae7c91`
and verification image
`sha256:0d658a6a412710958722a8495509ef7c3aea317f069083e22df1dae9b8c2e184`,
source UID `fde67a33cde7625f6b2e4571a77b4ca878140937ce9a6eed48baee83ad5b2034`.
All 780 runtime hashes matched before and after; module origins stayed installed,
with no runtime source overlay. Current native identity/memfd checks and the
foundation audit passed; CUDA remained uninitialized. Test hashes, commands and
logs are retained in `providerone-alignment-20261106/fixture-contract-repair-v1`;
the installed gate SHA256 is
`853a06bd183ab56f247285a8378a2b7a461bc5fd04beb06a158e102783b59b6f`.

The historical 37 failures and their nonempty reference diagnostics remain
preserved at their original checkpoints. This restores meaningful CPU software
coverage; it does not retrospectively turn those failures into passes or
establish actual V100 topology, full-model quality, capacity or throughput.
Production is unchanged and the full ProviderOne delivery gates remain open.

## Private PP2 stage components (A6)

The GGUF loader has a private typed `_pipeline_stage=PipelineStage(index)`
component interface. The fixed global 48-block target is split into blocks
0–23 and 24–47, with TP8 and the same selected ETP8/4/2 degree, with groups
confined to each stage.
Planning and materialization retain only the selected stage's coefficients;
the explicit complementary inventory remains checked for missing/unknown
tensors. Packed encodings, row/column offsets and global `LayerW.index` values
are preserved. Global Config layer identities are not replaced by 24 local IDs.

The input stage alone owns the target embedding and PLE lookup/table; the output
stage alone owns final mixing and the target head. The current SC117 PLE is
`blk.1`. Later-stage PLE placement is refused until an owner-row service exists,
rather than duplicated or dropped. A real sparse IQ4_NL table control checks
input-stage lookup/source words and the unchanged global ngram ordinal; it is
not a full-table hash or model-quality proof. Partial import refuses speculative
method attachment until the required cross-stage endpoint/service contract is
implemented. Stage-local expert-partition metadata is explicitly scoped apart
from the global PP2 stage identity.

Private `compute_stage` executes the owned local layers with the existing
`layer_forward`, writeback and final readout operators. The first stage applies
its last pending MoE update and exports borrowed contiguous `[rows,4,hidden]`
FP16 HC, without a mixer/head. The caller must retain that storage until its
handoff completes. The second stage consumes all four streams in the exact
staged dtype/device/layout, then performs final readout. A single hidden vector
or stream mean is refused. Invalid post-preparation inputs notify the existing
input-preparation owner before cache/operator work. Captured execution and
draft-feature transport are not enabled by this component.

State uses 18 local recurrent and six target attention caches while retaining
absolute global layer IDs. Numerical cache identity includes the stage's global
block span, so equal local shapes cannot authorize cross-stage copy/restore.
The input stage allocates no target-logit or candidate storage. Optional padded
projection allocation and its forecast use 18 local layers without changing
the global 48-layer Config. Actual coefficient and host-table ledgers come from
the owned plans, not a division of the full model's runtime buckets by sixteen.

The complete scoped 42-module source and fresh installed batches each passed
**1,600 tests with one optional skip**. Source took 31.74 seconds and installed
took 32.84 seconds; both exited zero. The skip requires optional pinned
companion/actual target metadata that was not supplied. Required deployment
controls passed 194 cases, and static checks passed on nine affected Python
files. Component checks passed 165 cases with the same optional skip, while
44 existing padding/HC/cache controls pass, including a repaired stale AST
lookup that now examines the active Engine's actual geometry wrapper. These
overlapping batches are not added into a unique-test total.

CPU composition uses an independent NumPy Half-storage oracle and deterministic
stand-in layer operators while retaining actual writeback; it does not qualify
CUDA operators or full-model numerics. Sensitive writeback/storage tests cover
the branch, routed-slot and rank/K-slice modes. Actual CPU State/buffer controls
exercise local cache maps, cross-stage restore refusal and padded projection
allocation versus forecast. The legacy `compute` body remains unchanged.

Fresh artifacts use source checkpoint
`8391b599aebc5f6c9af9779e4df0fa8b25d1f663`, source UID
`298eccd1f28f69a7627d799566503b997f480d5ea4d954cdeefa3c24b29a728c`, runtime
`sha256:aad3c1c212c78c097a97b51cffba53b10434df97d64ba9db9ce7080ad9fc4ecd`
and verification image
`sha256:7508c491bcba9dedab06bd9a89ea9131d8a3a11c86081ae0101bae35e319917c`.
All 781 installed runtime-file hashes matched before and after; all five
changed module origins were installed `site-packages`, with no runtime source
overlay. Current native identity/memfd and foundation checks passed; CUDA stayed
uninitialized. Exact commands, test inputs and failure/remediation history are
retained in `providerone-alignment-20261106/pp2-stage-boundary-v1` and the linked
loader packet `pp2-stage-execution-v1`. Installed gate SHA256:
`8c872b7c18feebd41e188340ed0eb6e110f43356281176649db656248266c0f5`.

This is necessary executable groundwork for the required PP2 path, not complete
PP2 inference support. There is no public PP2 Engine/CLI flag. Bounded cross-node
HC/metadata transfer, progressing embedding/lookup/feature services, final-stage
MTP/DFlash, one unresolved speculative round and coordinated commit/rollback,
RNG/frontiers/cancellation/publication remain required. Complete stage resource
admission and actual two-server mesh/RDMA, model quality, capacity and performance
qualification also remain open. Production is unchanged.

## Private two-leader HC transfer (A7)

The private `pipeline_transfer.HCTransport` component exclusively owns a dedicated
two-rank NCCL communicator and its Store. It does not borrow the stage TP8
communicator or expose a public PP2 Engine/CLI flag. Both endpoints submit each
attempt, including refusals, and agree on the complete bounded ordinal,
profile/epoch, sequence generation, round, row count and token/media/position
identity references before any native payload operation. Two alternating key
banks avoid an ever-growing native Store key namespace; coordinated retirement
prevents premature bank reuse.

The payload is exact contiguous FP16 `[rows,4,2560]` HC, followed by int32
`[rows,3]` positions and int32 `[rows]` token IDs. No HC mean or conversion is
introduced. The receiver reserves `slots * max_rows * 20496` tensor bytes;
NCCL internals, streams/events, allocator overhead and transport registration
remain separately unqualified. Sender storage is borrowed and must remain
read-only until the send-completion lease permits reuse. Receiver storage cannot
be consumed before transfer completion and a subsequent asynchronous-error
check, and cannot be reused before consumer-stream completion or a completed
discard. The ring permits only one unresolved round per sequence generation.
Counter exhaustion refuses new work while allowing existing leases to retire.

Unknown native/event/consumer failures quarantine the owner. Primary errors,
cleanup failures and previously retained owners remain available for diagnosis;
resources are retained until successful communicator abort and device retirement.
The existing launcher notification is entered once on poison when configured;
it is not a cross-server supervisor. Stream/device restoration occurs only after
successful scoped work, avoiding additional CUDA calls after an unknown fault.
Partial constructors publish acquired buffers into retained ownership slots.

Deterministic CPU boundary doubles exercise identity/refusal, finite ring and
key reuse, producer/consumer ordering, sentinel HC/metadata routing, cancellation,
stale leases, partial construction and failure retention. These tests are
component controls, not actual CUDA/NCCL transmission or model numerics.
The actual local verification SDK stream implementation was inspected for the
error-restoration contract; this does not qualify the separately pinned V100 SDK.

The final scoped 45-module source batch passed **1,670 tests with one optional
pinned-companion/target-metadata skip**, in 33.28 seconds. Required deployment and
public-permission controls passed 194 cases; affected F/py312 and configured Ruff
checks passed. The batch uses canonical Python source with the freshly built,
unchanged installed native extension and is distinct from installed-image
qualification. CUDA remained uninitialized. Earlier focused controls passed 68
cases before the final two capture-precheck regressions were added. Saved failing
controls, remediation history and exact commands remain in
`providerone-alignment-20261106/pp2-hc-transport-root-v1` and
`pp2-hc-transport-execution-v1`; they are not fresh-checkout test dependencies.

The fresh installed-image batch passed the same **1,670 tests with one optional
skip**, in 33.48 seconds, with CUDA uninitialized. All 782 installed runtime
files matched the frozen source before and after; the new transfer module and
native extension resolved from installed `site-packages`, with no runtime source
overlay. Foundation audit and current native source/memfd checks passed.
The source checkpoint is `01751a062ea4449eb1c2d035d4cbf0061a2bf5bf`, source UID
`6f39b2b73884a2422fc1078c20262c896555814a28a37d3adf09c960946cc106`.
Runtime image:
`sha256:a565853c46f50b776ba7110cf4d855e1909a3e82762453adb23c2112f95c1bdc`;
verification image:
`sha256:d46a83e54ad789c5a5086176d3ea07f86dd87ca491404182f943c5977f77a459`.
The final documentation-only checkpoint does not change those runtime inputs.

TP8 fanout, progressing embedding/lookup/features, method endpoint ownership,
distributed round commit/RNG/frontiers/cancellation/publication and complete
per-stage resource admission remain required. Actual two-server mesh/RDMA,
CUDA transfer, SC117 quality, capacity and performance remain unqualified.
Production is unchanged. No further candidate investigation is part of this
closeout, following the owner's instruction to finish current work and stop.

## TP8/ETP4 one-server native component

The owner prioritized TP8/PP1/ETP4 before further PP2 work. A3's reference
target/shared-GGUF MTP path already partitions experts into two groups of four
ranks. Each group owns 256 experts, with local routed width160; shared experts
retain TP8 width80. Global router IDs, one representation per expert channel
and the existing ordered full-TP8 FP32 partial gather are preserved. Selected
target/prompt buffers store FP16 expert slots before FP32 weighted reduction;
the dedicated MTP buffer uses FP32 slots. Those storage boundaries are distinct
from the subsequent reduction dtype.

An explicit `GGUFExperts.forward_partitioned_native_candidate` component now
uses device-side global-to-resident route mapping and existing packed native
arithmetic, rather than host scalar checks or dynamic `nonzero` compaction.
It publishes literal zero for foreign slots and the independently evaluated
TP8 Dense shared result for global sentinel512. Invalid global IDs produce the
existing native diagnostic NaN contract without an out-of-range bank read;
they are not repaired into a valid expert. This is a component candidate, not
selected serving dispatch or an established speed/quality equivalence result.

The native component supports direct FP16 and FP32 caller outputs. Shared
FP32 results round once to FP16 when the caller owns FP16 slot storage, using
the existing typed down-kernel paths and a typed publication store. No extra
full-slot FP32 temporary or copy is required merely to bridge output dtypes.
Native arithmetic/model equivalence still requires its separate device gates.

Callers supply reusable int64 routes and FP16 activations, separately owned by
target and MTP roles. The native scratch forecast is
`rows * slots * (8 + 2 * 160)` bytes, with at most 65,535 slots per call. Caller
slot output and shared Dense temporaries remain separately charged. No per-bank
resident scratch is introduced. Integrated target/MTP worst-rank peak admission
and a legal prefill policy beyond that call envelope remain selection gates.
The component does not enable whole-model graphs: unique-history live-row
execution has its own independent capture restriction. Existing MoE activation
storage is a possible integration reuse, not an extra allocation promise.
Reference remap reserves cannot automatically fund persistent native arenas
while a reference fallback remains reachable; prove an exclusive prepared
provider or charge the overlapping lifetimes before reusing those reserves.

The V100 Compose profile now forwards an explicitly set
`TF_EXPERT_TENSOR_PARALLEL_SIZE=4` through its existing Bash argument array to
the server's existing option. Unset/empty behavior is preserved, including legacy
profiles; the CLI remains the authority for supported degrees and ownership
compatibility. Ten actual Compose-render/argv controls pass, including degree
8/4/2 forwarding and literal malformed-input rejection without shell execution.

New SM70 component controls cover global boundary IDs, both owner groups,
int32/int64 routing, foreign-only groups, tails, mixed source formats, poisoned
outputs/scratch, changed-route graph replay, alias refusal and separate role
scratch. Their exact native arithmetic comparison reuses the existing packed
provider to test the new mapping/publication boundary; it is not an independent
full-model numerical or distributed-collective oracle.

The typed-output four-module focused CPU batch passed 129 cases in 1.57 seconds.
The scoped 47-module source batch passed 1,692 cases with one optional pinned-
companion/target-metadata skip in 32.96 seconds, including all A7 regression modules.
Required deployment/public-permission controls passed 194 cases. Configured
Ruff, affected F/py312 checks and Python compilation passed. CUDA stayed
uninitialized; Python compilation does not compile or qualify the native CUDA
entrypoint. These are canonical-source checks using the freshly built unchanged
native FD extension, distinct from a fresh installed application check.

The earlier FP32-only component at `033a0509` and its 1,691-case source/installed
results remain historical evidence in `tp8-etp4-execution-v1`. They do not
qualify the subsequent typed-output revision. Current commands and results are
recorded separately in `tp8-etp4-execution-v2`.

The fresh typed-output installed application passed the same 47-module batch:
1,692 passed with one optional skip in 38.65 seconds. All 782 installed runtime
files matched the 785-source-file frozen capture before and after, excluding
three source README files from runtime installation. Both changed Python module
origins and the native FD extension resolved from installed `site-packages`,
without a runtime source overlay. Foundation audit and native FD source/memfd
checks passed; CUDA stayed uninitialized. This uses the GB10 SDK for CPU/package
proof and does not compile the changed SM70 CUDA operator.

The typed source checkpoint is `6b6f7756d8e1d9a093f1f3a39c9306174374d511`,
source UID `6cce8b14fd79f544f0bc64205c85e0780d80504880dab05c40eedaed375cbcba`.
Runtime image:
`sha256:5e8f4b85c8e934c5161a2003b271b0c62b9ac3e500f3badcb15a0d1611aa0b79`;
verification image:
`sha256:972e911a4ce282f61f7a05e874a6204dc0da81c9799d30edec4307c89e60b249`.
Subsequent evidence-only documentation changes do not alter those runtime inputs.

Actual SM70 compilation/execution, canonical SC117/MTP model and media quality,
complete resource/capacity and matched end-to-end performance remain unrun.
The current host is GB10; the known dev-01 connection has an RTX PRO2000
Blackwell. Neither has the selected CUDA12.9.2/Torch04cf V100 SDK installed.
Target-server access remains unresolved. No guard is relaxed to substitute those
devices, no native candidate is promoted and production is unchanged.

## Configuration and capacity

For Flash Next GGUF with GPU indexer history, `--parallel auto` computes a
resident count at the selected context. An integer is an administrator ceiling,
including counts above 32; it does not force an unsafe allocation. Every chosen
slot reserves its complete prompt-plus-reply window. The solver uses the
limiting rank and separate loader/serving/host costs, with a useful shared
execution envelope, retained recovery and replacement overlap. The fixed initial
unique-history profile has zero dedicated live-migration overlap; it keeps COW,
restore, recovery and supported replacement liabilities in their own categories. It keeps the selected context rather than shortening it to
fit a requested session count.

`--kv-full-windows auto` selects that full-context policy on the GGUF GPU-history
path. An explicit integer constrains the same capacity solver. Other providers
and the legacy RAM-indexer path retain their own explicit contracts; they do not
inherit the unique-history capacity calculation. The V100 Compose defaults use
automatic session and full-window selection. Existing numeric environment
overrides remain ceilings.

`capacity_plan.session_capacity` reports the computed count, configured ceiling,
limiting resource/rank, and rank costs. `capacity_plan.execution_envelope`
separates verification rows, prompt work, and proposal depth from resident
capacity. Splitting verification into fair complete-request microbatches does
not reduce the number of allocated full-context slots. Planning and materialized
allocation fields must be read with their evidence labels.

Target weights stay TP8. Unique GPU histories apply only to the supported
two-global-KV-head SM70 geometry. Existing Blackwell numerical and storage
paths keep their contracts. A different placement's performance cannot be
inferred from byte balance.

## Numerical formats and speculation

Ordinary INT8 retains the H32 transformation, signed byte payload, and FP16
scales per 32 values. Existing `rotorquant-signed128-4` and
`rotorquant-signed64-norm4` remain distinct formats with their original evidence.

The initial `rotorquant-iso4-sm70-v1` format pins the CUDA ISO4 implementation
at `llama-cpp-turboquant` commit
`08e025c06ab521e4fa9e5c08b80af57614543e53`. A 128-value block has 64 code bytes,
an FP16 norm, and a reserved FP16 field. TensorFold stores code bytes and the
opaque four-byte metadata word separately; wire controls verify their combined
68-byte representation. ISO4 uses the pinned 32 quaternion rotations.
The distinct `rotorquant-planar4-sm70-v1` remains the first later equal-memory
comparison; it is not an alias or a substitute for initial ISO4 qualification.
The pinned reference's CPU and CUDA rotation constants differ, so these formats
explicitly select their CUDA authority. Native long-prompt
packing, attention, quality, and performance require their own V100 runs.

Stock MTP and `--draft-method none` have explicit resource and tail conventions.
Temporarily selecting zero proposals keeps enabled MTP conditioning current;
it does not free resident method resources. Reply/context budgets and stop cuts
bound the prefix committed to recurrent state, convolution, KV, indexer state,
and the drafter. Algorithm-defined random draws are not rewound on rejection.
Startup `--mtp-drafts 0` under `auto` is the legacy structural-disable choice;
explicit stock MTP requires a positive allocated maximum. Use the runtime zero
proposal setting for warm capability.

DSpark and DFlash2 are separately implemented methods. A DFlash checkpoint is
not inferred from a model name or classified as stock MTP. The current
`--draft-method dflash2`/`--drafter` path uses the same Engine and prepared
serving-profile lifecycle, with exact `draft_pairing` content authority before
CUDA allocation. Its [provider/runtime](../src/tensorfold/families/qwen4_exp/cuda/dflash2_runtime.py)
implements completed-HC feature conditioning, target embedding/head aliases,
bounded FP16 draft windows, accepted-prefix recovery and warm maintenance.
The implemented block is four rows, with one to three proposals selected
before the draw; a source block-eight declaration does not enable block-eight
serving. Source/CPU checks do not qualify this pairing's V100 execution,
acceptance, quality, physical envelope or performance.

Reference research identified a concrete candidate,
[`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/tree/a22a1744e85c17364dcc54a6a420af7135d4c7f6),
pinned at `a22a1744e85c17364dcc54a6a420af7135d4c7f6`. The initial research used
public metadata and the tensor header; the later source implementation now
handles its five-layer, 2,048-token sliding draft stack and completed-target-HC
averaging under the manifest-bound contract. The optional exact large DFlash
artifact control was skipped in the 5,214-test checkpoint. The author's
GB300/TP1 evidence does not qualify this fork's V100/TP8/SC117 target; exact
pairing and complete target execution/quality/resource qualification remain
required.

## Runtime control

Adaptive engines expose `GET /v1/runtime` and `POST /v1/runtime`. Both use the
listener's existing API-key policy; opening metrics does not open this route.
The POST body contains one or more integer execution settings:

```json
{"row_budget": 32, "proposal_limit": 0, "prefill_tokens": 512, "session_ceiling": 16}
```

Values must fit that engine's reported `allocated_maximum`. The example values
are illustrative, not a V100 performance recommendation. The response reports
configured, pending, and effective values. The coordinator applies changes
between batches and shares a single ordered execution plan with TP peers.
Lowering the admission ceiling preserves running requests; raising it cannot
exceed the capacity already allocated. Active requests keep their full-context
reservation until completion/cancellation and retirement of their readers.

Opt-in [serving profiles](serving-profiles.md) add the structural transition for
context, codec, model and method through a prepared local manifest. The same
worker and communicator drain model-bound prepared/queued requests, retire old
resources before replacement allocation, and publish after all-rank readiness.
Host fake-loader protocol/lifetime tests cover A-to-B-to-A and unavailable-state
load refusal/retry. Actual model/CUDA-context loading, resident-byte recovery,
quality and integrated V100 performance remain unrun qualification gates.

The runtime receipt is an advisory host snapshot, including active requests,
allocated slots, bounded derived prefix-cache slots/entries, the current
execution epoch, useful committed tokens,
and the capacity receipt. `/stats` includes this section for adaptive engines.
Reading it does not synchronize GPUs or turn startup planning into a current
physical-free-memory measurement.
Step elapsed time is host wall time, including waits actually observed by that
thread. Device-completion time is unmeasured and reported as null; asynchronous
maintenance must be included by a complete workload measurement before using a
timing sample as performance evidence.

The execution receipt includes `work_observations`, published by the serialized
worker at operation boundaries. It separates fresh logical prompt admissions,
request/replay prefix hits, submitted target prefill/decode and replay rows, and
MTP prefill/target-conditioned refresh/provisional proposal rows, including the
existing cached-prefix boundary refresh at its saved position. Warm runtime
zero-proposal refresh has its own count. Synthetic startup warmup is cleared
before serving. None of these row counts is generated output or device time.

Indexer counts are derived from actual submitted query ranges: visible complete
pooled groups, visible sparse group/query-head score pairs, and selected key rows
including incomplete tails. They count the canonical logical work once across
TP ranks, with no factor for replicated scratch or masked physical launches.
They are not measured memory traffic or a claim that every implementation
executes only visible arithmetic. Partial failed forward submissions are marked
unknown; complete totals are not fabricated. The prospective profile-table
`indexer_work` axis remains unknown until its separate covering contract is
established. `expert_group_max` remains null; batch size is not a substitute for
actual routed expert occupancy.

Acceptance histograms are fixed to the allocated proposal maximum, including
maxima above32. They count the existing verifier's accepted prefix after its
reply/EOS rules and before the text-stop prefix callback. Offered widths and
accepted totals by offered width are separate. The existing useful-output count
continues to exclude owed replay and failed grammar publication. No per-stage
durations are guessed. A failed admission observation retires or quarantines its borrowed slot and rolls
back the attempted counters. A separate failure latch invalidates completeness
even when publishing another snapshot cannot allocate. Counter exhaustion is
visible and invalidates completeness instead of silently wrapping; these host snapshots are diagnostic readings, not
new capacity or performance proofs.

The historical optional migration prototype accepted standalone
`{"attention_degree": 2}` updates, copied packed bytes through balanced bundles
and published committed owner-map generations. Its blocking, partial-map and
containment oracles remain research/reference evidence. Its 64 MiB/rank candidate
allowance is not charged or enabled by initial serving; it was never a measured
physical migration bound. Initial fixed-map serving does not expose these live
updates. Any future migration adoption must restore its complete intermediate
resource and lifecycle obligations and pass the governing qualification gates.
Managed TP launches signal their supervisor on a fatal cohort operation so a
peer blocked in a collective cannot strand process retirement. Unmanaged
launches retain their external containment responsibility.

Without an explicit computed-retention policy, CUDA generation remains
request-scoped: completion or cancellation retires the claim after its readers,
transfers and callbacks finish. Reusable full-window allocations and bounded
derived prefixes do not create an intentional retained-session promise.
Stored Responses/Conversations retain logical records independently of GPU
ownership.

An explicit typed GPU/RAM retention policy uses the implemented common catalog
for owner/generation claims, protected idle checkpoints, prefix leases,
suspension/restoration, close/expiry and incompatible-profile drains. FIFO
waiting, input credits and pending/in-flight output credits retain their logical
owner through blocked-request requeue. New-turn generation bookkeeping remains
separate from same-request continuation. These source contracts and CPU checks
are subject to the pending native resource/backend and complete target gates
above. New admissions increment the destination slot's generation before
copy/restore; rollback within the same request preserves that generation.

A zero proposal setting keeps enabled method conditioning current during active
execution. A per-request `draft: false` switch disables speculative work for
that call. Without computed retention, request completion does not maintain an
idle endpoint in anticipation of an unspecified future resumption.

Execution profile registration uses either startup-bound proof references or a
standalone `{"profile_manifest": "new/manifest.json"}` request. Verification is
queued to the TP worker; paths cannot escape the configured evidence directory.
See [the exact manifest contract](execution-profile-manifest.md). Qualified
covering regions are explicit and nonoverlapping; token-by-token changes inside
one proven region can retain its selected execution plan. No measurements are
fabricated when the table has no eligible region.

## Reproducible hardware and workload evidence

After building the V100 verification image as described in
[the deployment guide](../deploy/v100/README.md), record the actual server:

```bash
python tools/v100_adaptive_qualification.py inventory \
  --rank-devices 0,1,2,3,4,5,6,7 --out platform.json
python tools/v100_adaptive_qualification.py transfers --out transfers.json
```

Inventory is read-only and bounds subprocess output and time. Missing tools,
failed commands, and absent measurements are retained explicitly. Run isolated
transfer probes outside serving performance measurements. GPU indexes in the
example are explicit logical-rank order, not a claim about the best NVLink
mapping. Record CPU/RAM/NUMA placement, UUIDs, ECC, links, clocks, power,
temperature, errors, and exact software versions.

Save the running `/stats` response, then derive the qualification matrix:

```bash
python tools/v100_adaptive_qualification.py matrix \
  --stats stats.json --out matrix.json
python tools/v100_adaptive_qualification.py benchmark \
  --url http://127.0.0.1:8000/v1 --out benchmark.json
python tools/v100_adaptive_exercise.py --url http://127.0.0.1:8000 \
  --out exercise --request short=/path/to/short.json \
  --request intermediate=/path/to/intermediate.json --request long=/path/to/long.json
```

The matrix includes each integer concurrency through the computed endpoint,
including capacities above 32. The benchmark uses the existing HTTP SSE
workload implementation and preserves raw responses/failures. Its short-prompt
measurements do not automatically qualify every matrix cell. Near-full
retention, mixed load, migration, output/quality, long positions, cancellation,
slot reuse, and sustained fill/drain remain distinct gates. Tool resource caps
are explicit workload-generator limits and must not become serving ceilings.

Compare identical target/draft revisions, codec, context, retention, output
quality, arrivals, and runtime state. Count useful prompt work and committed
output separately from tentative proposals. Include queueing, TTFT, inter-token
latency, fairness, recovery, communication, and migration. Select performance
profiles only from applicable completed measurements, never from a generated
matrix or startup allocation alone.

The exercise runner uses supplied JSON requests and actual token counts. Add
distinct long requests to establish multiple retained contexts, plus named
`stop`, `eos`, `exact` and `overrun` requests for their corresponding gates.
Missing input roles, early EOS in a context-exhaustion case, unobserved slot
reuse/retention and incomplete migrations remain explicitly unsupported or
pending. It drains response owners before restoring changed controls. Its
cooperative body deadline has the documented one-socket-read bound; it does not
claim instantaneous DNS/header cancellation. HTTP observations alone do not
authorize a qualified execution profile.
