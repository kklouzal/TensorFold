# G3 private target-prefill integration contract

The standalone component is an intermediate numerical trial. Actual native
verification precedes the serving hook; this design does not claim a selected
profile, numerical equivalence or qualified performance. Source39's standalone
seven-case State/span/copy controls passed and retired on its actual protected
SM70 image; this does not qualify the new accepted-target group or a model.
`cuda.sm70.gdn_chunk64_half` and the default-off group remain numerical trials.
Current native chain/Value8 paths remain independently usable.

## Guarded frontend and epilogue

The insertion point is the Half branch of
`families/qwen4_exp/cuda/forward.py::_prefill_chain`; target decode, MTP and
verification are excluded initially. The candidate must be selected explicitly
in a prepared per-engine policy, not an environment-global family switch.
The caller must prove TP8 Hq2/HV6/D128, Half projection/conv/output, FP32 state,
current State ownership, funded row capacity and admitted numerical mode before
any launch. Unsupported modes retain the existing required capability path;
no silent tolerance or dtype conversion is allowed.

The current fused SM70 chain performs conv, SiLU, normalization, gate generation,
recurrence, gated RMSNorm and output group sums. The imported core replaces
only recurrence. The new owned frontend should reuse the coefficient/cast
sequence in `cuda/sm70/gdn.cu::chain_body` and
`gdn_value8.cu::front_body`, with the following explicit changes:

- Preserve ordered FP32 four-tap convolution and the current Half SiLU result,
  existing epsilon inside L2 sum and grouped Hq-to-HV mapping.
- Compute unscaled FP32 normalized q/k, then round to Half for this component.
  Let the core apply D^-0.5 exactly once. Current TF scales q in FP32 before its
  scalar readout, so this remains an explicitly different numerical mode; do
  not unscale a Half/scaled intermediate and call it the original producer.
- Store Half v with the incumbent convolution/SiLU rounding.
- Produce raw FP32 `rate*softplus(a+dt_bias)` directly, with rate from GGUF
  stored A_neg or explicit HF `-exp(A_log)` according to the existing typed
  weight contract. Do not derive it from saved exp gates or rename A_neg as
  A_log. Preserve the current softplus threshold and FP32 operation order.
- Initially retain current TF `Half(sigmoid(b))` rounding, widened to FP32
  storage for the core. FP32 storage does not require recomputing beta in a
  different precision. Upstream's unrounded FP32 beta is a separate potential
  numerical variation; it must not enter implicitly.

The existing generic `gdn_io.front/back` are BF16-oriented and do not supply
this raw-log/Half frontend. Calling them unchanged is not a valid adapter.
After the core, an owned Half epilogue must reproduce the incumbent Half readout
storage, RMSNorm casts, sigmoid(z), final Half output and FP32 32-channel sums.
Any in-place reuse of b.gout for raw/core and final output needs a proof that
all per-head source values are read before overwrite; do not infer this from
matching shape. Allocate an explicit funded raw output until that proof exists.

The new frontend's q/k/v/raw-log/beta tensor payload is2608 bytes/row. The
current core scratch is792 bytes/row+21 bytes metadata, plus raw output1536
bytes/row if kept separate. Thus a straightforward separate allocation needs
4936 bytes/row+21, excluding existing final output, initial/final state and
compiler/runtime retention. Reusing old scratch/output needs actual lifetime
and alias evidence and an updated admission receipt; no savings are asserted.

## Implemented fixed-capacity Half boundaries

`gdn_chunk64_half.layout(capacity)` admits1..65535 rows (the owned native
frontend/epilogue use grid-y). `allocate_workspace` requires4144 bytes/row
explicit device credit before allocation:2608 for the frontend plus1536 for a
separate raw Half core output. Existing final output/sums, FP32 state and the
core's792 bytes/row+21 are independently funded. No storage savings are claimed.
The owner cold-binds exact contiguous plain Half/F32 tensors, original conv
history, exactly one typed `a_log` or `a_neg`, positive representable FP32 epsilon
and disjoint outputs. It retains full-capacity owners and active contiguous
views. Binding and submission require inference mode, the current prepared
CUDA device, one CPU thread and one stream; capture is refused. Shape/storage
replacement is rejected and a submission failure latches the bound owner.
All borrowers, kernels and operands survive positive stream completion; failure
requires State containment and retained ownership, with no retry/publication.

The native specialization shares `gdn_value8.cu`'s existing convolution,
normalization, gate and epilogue bodies. Its explicit compile-time variant emits
unscaled Half q/k and direct FP32 raw gates. The old variant retains its scaled
FP32 q and exp gate. Both use Half-rounded beta widened to FP32. The epilogue
shares the incumbent arithmetic and reads the G3 core's separate Half raw
buffer; final Half output and FP32 group32 sums do not alias any input/workspace.
The independently bound core must consume the frontend and produce raw output
on this same ordered stream before `back()`. Uninitialized raw output is never
a valid epilogue operand. Typed weights/producers must establish finite inputs
and finite nonpositive raw gates; this owner introduces no synchronizing GPU
value discovery or boolean numerical attestation.

The standalone fixed-capacity boundary does not select engine behavior; the
separate private group policy below is the only opt-in target hook.
It prepares kernels once, binds outside requests and performs no device
allocation/compilation during submission. Independent CPU boundary diagnostics
retain cast points and direct-gate underflow cases without a tolerance or CUDA
bit-equivalence claim. Actual native frontend/epilogue/call ordering, full model
continuation/output/PPL/MTP, fit and end-to-end P3 gates remain required. A separate explicit segmented adapter described below is implemented without
serving dispatch; its native and model gates still block selection.

## Prefix cuts and alternate banks

`_prefill_chain` currently splits the accepted prompt at every retained cut,
passes a shifted conv clone per span, alternates two recurrent banks, copies
cut states and advances the convolution tail. The core keeps interior
cut policy outside its primitive API; `gdn_chunk64_prefill` now binds a
segmented candidate explicitly; the private group policy below adds the
accepted-target hook. Its contract is:

1. Validate all cut rows (strictly ascending, strictly inside the prompt) and
   funded cut-rec/conv destinations before any frontend/core writes. Prepare
   metadata from already-owned host positions; no device `.item()` discovery.
2. Generate frontend values once for the complete prompt against the original
   immutable conv history. Convolution indexing uses full prompt offsets, not
   a zero-based reset at each cut.
3. For each span, pass contiguous logical q/k/v/g/beta slices, current FP32 VLK
   bank as readonly input and the distinct alternate bank as final destination.
   Core scaling/state layout are unchanged. Swap the local bank index only
   after that span is submitted under the owning actor/stream contract.
4. At each cut, copy that prefix's final rec state and the exact convolution
   tail derived from original history plus projection prefix. Current conv
   state must not be advanced early and reused as if it were original history.
5. Publish the layer's final bank index and complete conv tail through the
   existing accepted-prefill lifecycle. This is not MTP tentative publication.
   Multiple spans reuse two banks and can overwrite an earlier bank after its
   final consumer; they do not provide rollback to the original prompt state.
   A partial asynchronous CUDA failure therefore requires the existing fatal
   State containment/poisoning contract; no retry against a half-mutated State
   or false transactional rollback claim is permitted.

Row/cut variation must not compile new kernels on a request. The original
TileLang generators have dynamic row dimensions; extend the owned component
with a funded maximum-row span binding that supplies workspace views and
logical chunk counts for actual spans. Validate `0<R<=capacity` and aliases at
the owning binding boundary. The maximum-row owner now implements that binding with active scratch views,
positive logical spans and per-span chunk counts. The three original generators
remain unchanged. Native validation is still required before a serving hook.
All per-span operands/workspace/kernel owners must survive completion. Initial
integration remains outside graphs until an explicit capture and replay/version
contract is verified; retain graph execution paths in the incumbent policy.

## Explicit segmented component owner

`families.qwen4_exp.cuda.gdn_chunk64_prefill.prepare` requires complete device
payload credit4936 bytes/capacity-row+21, the protected G3 compiler and an
already prepared same-device native-family shift helper. This resolves no
helper or compiler during submission. The single completion event is also
resolved cold; runtime/compiler retention remains separately accountable.
`bind` borrows an actual Half `State`, one linear layer, its original conv
history, a complete projection piece and independently funded output/sums.
Every cut must already own full-layer FP32 recurrence and Half conv snapshot
destinations. Their explicit additional grant is400896 bytes*layers*cuts;
no lazy cut allocation or incomplete snapshot publication occurs in this owner.
`Cut.at` must match the declared prompt-buffer origin.

All storage/view signatures, State bank/slot generation and owner identities,
cut rows/destinations and prepared workspace/helper/event identities are sealed
before any frontend write. Frontend generation runs once for the whole prompt.
Each core binding borrows contiguous full-prompt slices, active scratch views
and distinct current/alternate banks; live rows and chunk counts are shape/scalar
arguments, so arbitrary non-chunk-aligned spans compile no new variants. The
nonvarlen/non-CP/nonindexed generator configuration does not consume cu/offset
values: capacity metadata remains immutable, avoiding per-span H2D rewrites.
The pinned FFI host wrappers ignore the singleton batch stride; row strides
remain contiguous. Owned Half row strides preserve the generated 16-byte copy
alignment across cuts. Retained view mutations and prepared tensor replacement
are rejected before dispatch.

The adapter copies each prefix recurrence and conv tail from original history,
then consumes the complete raw output, enqueues the complete final conv tail,
records completion and publishes the final bank index. It does not update
accepted token positions or publish all-layer cut snapshots: existing accepted
prefill/commit/snapshot ownership still does that. State clone/restore methods
are unchanged. One active borrower journal is registered in the existing State
native-reader set. Submission does not wait for GPU completion. Reuse is an
explicit actor retirement action through the captured event; an event-record
failure requires positive owning-stream retirement. Primary/cleanup errors are
preserved, the State is poisoned and borrower ownership is retained on failure.
This isolated component does not install a per-layer host wait or a serving
selection policy; any future model integration must prove its scheduling,
lifetime, capture and end-to-end resource/performance contract independently.

## Remaining integration and selection

Component controls now cover the protected compiler/stream path, Half boundaries
and segmented State adapter; the native receipts below retain their exact scope.
The serving hook remains to be implemented. It must support packed State
segments using actual `pbuf.rows`, an explicit accepted-target-prefill phase,
bounded same-stream workspace/journal ownership and existing cut-snapshot
retention. It must not introduce per-layer completion waits or require a
single-prompt scheduler. Compilation and all persistent storage are admitted
before requests and graph preparation. The current MTP importer sets
`linear=False` for the MTP layer; sharing `pbuf` does not authorize selecting G3 during MTP
staging or absorb.
Full model continuation/output/PPL/MTP acceptance, graph behavior and all
required-region capacity and end-to-end performance gates precede selection.

## Standalone native span/State diagnostics

The maintained `tools/v100_gdn_chunk64_trial.py --phase spans-state` is a
separate bounded native diagnostic phase. It leaves the historical18 main
numerical cases explicitly `unrun`; existing `full` and `nondefault-stream`
phases retain their procedures. The phase requires a matching
application/GDN artifact, protected Compiler/OwnedArtifact
closure, the maintained native envelope and a separate Root GPU/resource
and health authorization. Source38's installed old core wrapper does not qualify
this phase. No GPU, profiler or build is triggered by authoring or CPU controls.
The new group path requires its own matching artifact, native group controls
and separate GPU/resource authorization; the standalone result does not qualify it.

The finite census is `(rows,cut)`:63/31,64/3,65/1,129/63,129/64,129/65,
131/65. Both independent owners fund maximum capacity131 once. Two real
`State` linear layers with per-rank Hq2/Hv6/D128 geometry, nonzero FP32 state
and Half conv history (including signed zero), cover stored GGUF A_neg and
HF A_log in separate layers and share one serialized nondefault
actor stream. This uses the actual State constructor with an unweighted,
GDN-only diagnostic descriptor, no attention or MTP caches; it is not a model
or TP8 execution claim. Both layer frames submit before explicit retirement;
no State exclusivity guard is bypassed.

Each case has exactly one cut and two spans. The physical cut bank is therefore
not overwritten by the final span: the snapshot is compared directly with that
actual produced bank in bits, including signed zero. Original input banks and
conv are legitimately updated by later spans/final publication; no permanent
readonly assertion is made for them. Projection/weights/parameters remain
immutable; cut/final conv tails, capacity sentinels, bank parity and storage
ownership/refusal are exact assertions. CPU Half boundary and float64 serial
references report raw/output/final/cut errors without an invented tolerance or
quality acceptance. Finite-value failures remain failures.

A diagnostic observer forwards to the genuine CUDA event and counts only the
adapter's `record`/`synchronize` API calls. The report asserts no call to that
observer's synchronize before the specified retirement; it does not infer the
absence of all CUDA/runtime/FFI waits, timing accuracy or performance. A final
controlled host observer exception occurs only after genuine CUDA event
completion and checks primary preservation, State poisoning and retained
borrowers. That final single-span control also asserts the opposite-bank
publication and readonly initial-bank bits (with signed zero), within that
consumer borrow only; the earlier two-span cases intentionally overwrite
their original input bank. It is explicitly not a CUDA fault or substitute native math case.
This phase requests no profiler, so its required reserved `--stream-trace-output`
path is unused and no stream trace is claimed.

### Actual component results, 2026-11-13

Source38's native run passed 61 cases: 42 Half-boundary diagnostics and 19 existing
Value8 compatibility controls, including capture/replay. All 42 Half properties
and 16 property-emitting Value8 cases matched the actual kernel-VMA-backed ELF
paths and SHA256 identities; the remaining three Value8 cases are controls.
The candidate and incumbent were distinct loaded modules. The completed run
retained 48 cache entries/6,725,389 bytes, retired the container and unmounted
its owned cache. All eight GPUs' health and empty-context checks matched the
baseline. The failed, never-started first attempt with an incomplete ulimit
declaration remains separate evidence.

Source39's `spans-state` run completed all seven cases and the controlled host
exception probe using image `59d619d0...`, literal application source
`361cb6b1`, and procedure SHA256
`81193141cf7cd3daf7fb832c6ccf4351c29532721c2c415a3a6039d651d3636d`.
All copy, bank, storage, refusal, reader-journal and retirement predicates
passed. All generated CUDA sources and 82 cache entries/10,868,790 bytes were
retained and rehashed, including 14 ELF artifacts. The controller and container
exited 0, the container was removed, and all eight GPUs' health/context checks
matched the pre-run baseline.

The observed maxima against the CPU references were approximately 1.075e-5
for raw output, 7.778e-6 for final recurrent State, 2.480e-5 for cut State,
0.001953125 for output and 0.004531861 for group sums. These are measurements
over the declared finite cases, not accepted error bounds or model-quality
results. Main 18 cases and the stream profiler remain explicitly unrun in this
new phase; their historical evidence is not relabeled.

Owning receipts are `Root-native61-actual-milestone-v1.json` under the Source38
`half-native61-data-v1` packet and `Root-native-spans-state-actual-v1.json`
under Source39's `gdn-spans-state-proposal-v1` packet in the build cache.
These component results do not select G3 or Value8 for serving and provide no
prefill/decode token/s, full-context capacity or model/PPL acceptance claim.

## Private accepted-target group integration (implemented, unqualified)

The optional `preparation.g3_prefill` object selects policy
`g3-target-prefill-half-raw-v1`; absence leaves the incumbent numerical and
preparation policy unchanged. It requires `trial:true`, the original PP1
TP8/ETP4 adaptive full24-Q8/full32-Q4 NONE/MTP text profile and the protected
owned GDN image. Managed profile switching, media and separate speculative
providers remain outside this first private mode; requested absent capability
fails during cold preparation, with no package installation or library fallback.
The exact object fields are `policy`, `tensor_budget_bytes`,
`journal_host_budget_bytes`, `cold_host_budget_bytes`,
`warmup_device_budget_bytes`, `cuda_home`, `nvcc_sha256`, and `artifact`.
`artifact` is the existing closed OwnedArtifact descriptor: attestation
path/bytes/SHA, owned wheel path, aggregate hash-byte credit and optional retained
native base wheel. These are actual operator-issued identities, never booleans.
No public serving flag is added and existing closed RN12/ATen teacher versions
do not admit this new numerical family as transport.

The owner prepares one actual compiled core and one Half frontend/raw slab at
`actual_pbuf.rows`, not attention's min(P,256) role envelope. Tensor payload is
4936P+21 bytes once per rank, independent of layer/segment count. The selected
mode omits the old shared prefill replay allocation2608P whose sole consumer is
the incumbent Half chain; decode State scratch and MTP absorb stay unchanged.
AtP256 the new payload is1,263,637 bytes and the net tensor increase595,989;
36 independent owners would instead allocate45,490,932 bytes before subtracting
that one old replay slab. These are structural byte counts, not a measured
throughput/startup gain. Source39 observed duplicate KKT/core compilation in its
two-owner diagnostic; production uses one prepare call and no per-layer compile.

The bounded accepted-target group is created before `_kv_begin`; it validates
packed unique States and cuts, owns a fixed layer/segment journal and registers
one group reader on every participating State. It preserves the original
projection/weight/conv/state owners and per-segment cut destinations. Same-actor
stream order allows each complete frontend/core/epilogue to finish consuming
scratch before the next queued segment/layer overwrites it. No CPU wait occurs
inside those submissions. The group never calls `kv_check` per layer: pending
Q8/Q4 status is preserved until the unchanged final whole-frame check. Only
after that authoritative check may normal group retirement release the journal.
Token-position commit and retained snapshot publication remain with the existing
caller. The pbuf target-frame marker is cleared before its ordinary MTP absorb.

Cut tensors use the existing POINTS/recovery snapshot category, are allocated
before layer submission and transfer through Cut to the existing snapshot and
retention owners. They are not recycled when the group event retires. Failure
poisons all affected States and retains the group; it propagates into the
existing cohort fatal/drain path before any local stream wait can deadlock a
peer collective. Primary and retirement failures remain owning outcomes.

Cold preparation checks explicit tensor/warmup/host credits, warms the complete
native path before graph admission, and joins the existing settled GPU physical
commitment. Journal host credit is reserved in aggregate acrossTP8; already
allocated root-list bytes are excluded from future host credit once. The Python
header charge is scoped explicitly; Torch/FFI/compiler private retention still
requires actual peak/RSS/model-resource qualification. No exact RSS claim is
inferred from Python `sizeof`. Cold positive completion and an explicit
Scheduler worker handoff preserve the compiled owner thread/stream contract.
Decode/MTP graph selection and all existing expert selectors remain unchanged.

The group path, real packed-State pending-status lifecycle, worker handoff,
full24/full32 fit, closed G3 model/quality contract and all end-to-end P3 regions
remain unqualified. The single-frame API and seven-case native oracle remain
available independently. No source implementation alone selects this mode for
ordinary serving or provides numerical tolerance/quality acceptance.

## Packed-group native diagnostics

`tests/cuda/test_sm70_g3_target_group.py` is a separately authorized component
procedure for the matching installed Source41 GDN image. It uses real Q8/ISO4
State allocations with two GDN layers, one cold compiled kernel set and one
P256 slab. It runs packed63/64/65 pieces and24/32-piece P256 shapes twice with
changed frame ordering, nonzero recurrent/conv history, interior cuts and a
real admitted actor handoff. It exercises epoch, alias, budget and actual
codec-status failure boundaries. It does not substitute a fake TP8 communicator
or claim actual model/policy adoption. The standalone seven-case native oracle
and runtime algorithms remain unchanged.

The diagnostic observes actual cut copies using the protected Torch dispatch
mode: it forwards each copy once, then clones that copy's source on the same
stream before reuse. Explicit clone/header credits and positive completion
retain those owners. It compares cut copies and conv-tail storage bits,
including signed zero, against the actually produced source. No CPU numerical
approximation is used as an exact-copy oracle. CPU serial and actual incumbent
outputs/State errors are recorded without a new acceptance tolerance.

The finite fixture tensor upper bound is216,639,189 bytes at32 pieces, below
the256MiB fixture grant. This includes diagnostic source clones and the
serialized incumbent temporary. Allocator, compiler, native metadata and process
RSS are separate measured envelope obligations. The proposed native envelope
is4CPU/8GiB on Node0,1200 seconds,2GiB scratch and1GiB retained compiler cache;
these are proposal DATA, never execution authority. An external held input
binds the actual source UID, immutable image, compiler and owned artifact.
Native collection/execution, TP8 full24/full32 adoption, model/quality and P3
remain separate required gates.

The CPU fixture/real redispatch controls pass11 tests on the Source41 protected
SDK without CUDA visibility or application overlay. The new CUDA test has not
been collected or executed. A first combined gate preserved Ruff's read-only
cache failure after the11 passes; a subsequent static-only gate fixed that
procedure and passed scoped Ruff. Formatting retained exact ASTs. No numerical
or native result follows from these CPU checks.

The component observation additionally binds the three retained actual TVM
callee modules with `get_function(..., query_imports=False)` and their unique
`__tvm_ffi_` dynamic exports to the existing VMA-backed FD observer. It checks
compiler-library bytes/SHA against the owned admission report. The pinned public
`TempDirectory.set_keep_for_debug()` context keeps actual generated libraries
pathname-backed through that proof and restores its prior flag in `finally`.
Owned compiler inspection precedes the TileLang/TVM import; the extra cold hash
pass is diagnostic overhead. There is no SDK patch or deleted-path relaxation.
The proposal sets ordinary `TMPDIR=/cache`, retaining the debug objects and
actual generated DSOs inside the same paid1GiB compiler cache before unmount;
`/tmp` remains a separate2GiB process scratch grant.

Every completed State cohort, its final chosen-order borrower, projections and
parameters are released before the next allocation RHS. Each independent
incumbent State/cut owner is likewise released after its CPU diagnostic copies
complete. This prevents an uncharged two-cohort/two-baseline overlap. Retained
Cut tensors remain owned until their exact-copy proof and positive completion.
The seven new CPU evidence-boundary controls use explicit mock records; they
validate refusal/protocol behavior and grant no native provider authority.

An additional real installed CPU control invokes the actual image's `nm` through
an inherited held FD for its genuine `_fd_owner` ELF, finds `PyInit__fd_owner`,
checks unchanged file metadata and proves descriptor closure. It passes without
CUDA visibility. This verifies that evidence-tool boundary; generated G3 native
DSOs remain unobserved until the separately authorized native run.

The export probe uses the existing bounded `v100_adaptive_qualification.capture`
process-group owner with an optional caller-owned `pass_fds` tuple. Its old
empty default is unchanged. The probe bounds output while reading, rejects
truncation, timeout and nonzero status, and retains the input FD until positive
process-group retirement and the final metadata check. Three refusal controls,
two actual passed-FD output/timeout controls and the affected actual installed
`nm` control pass; earlier CPU scopes are not repeated for this repair.
