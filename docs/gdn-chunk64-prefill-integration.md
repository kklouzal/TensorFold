# G3 prefill integration contract (private boundaries; no engine hook)

The standalone component is an intermediate numerical trial. Actual native
verification precedes the serving hook; this design does not claim a selected
profile, numerical equivalence or qualified performance. The explicit
`cuda.sm70.gdn_chunk64_half` frontend/epilogue is implemented as an unqualified
component; its native build/math and full-model gates remain pending.
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

This fixed-capacity boundary component has no engine selector or forward hook.
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
segmented candidate explicitly, with no engine hook. Its contract is:

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

## Native span/State diagnostics

The maintained `tools/v100_gdn_chunk64_trial.py --phase spans-state` is a
separate bounded native diagnostic phase. It leaves the historical18 main
numerical cases explicitly `unrun`; existing `full` and `nondefault-stream`
phases retain their procedures. The phase requires a matching
application/GDN artifact, protected Compiler/OwnedArtifact
closure, the maintained native envelope and a separate Root GPU/resource
and health authorization. Source38's installed old core wrapper does not qualify
this phase. No GPU, profiler or build is triggered by authoring or CPU controls.

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
