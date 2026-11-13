# G3 prefill integration contract (design; no engine hook)

The standalone component is an intermediate numerical trial. Actual native
verification precedes the serving hook; this design does not claim a selected
profile, numerical equivalence, implemented frontend or qualified performance.
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

## Prefix cuts and alternate banks

`_prefill_chain` currently splits the accepted prompt at every retained cut,
passes a shifted conv clone per span, alternates two recurrent banks, copies
cut states and advances the convolution tail. G3 standalone refuses all interior
cuts before allocation/mutation. The engine hook must either keep that explicit
refusal before mutating State or implement the complete segmented adapter:

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
the owning binding boundary. The current fixed-row standalone API has not yet
implemented that extension, and is not ready for an ordinary serving hook.
All per-span operands/workspace/kernel owners must survive completion. Initial
integration remains outside graphs until an explicit capture and replay/version
contract is verified; retain graph execution paths in the incumbent policy.

## Finite next increments

First verify the existing component on protected CUDA12.9/CUTLASS4.8.1/CCCL3.5,
Torch2.16/FFI0.1.16/Triton3.2, including actual nondefault stream activity and
producer-to-core-to-consumer ordering. Preserve its recorded errors, underflow
failures and initial/final bank observations without inventing acceptance
bounds. Then implement the guarded frontend/epilogue plus capacity/span binding,
and compare each boundary with incumbent conv/gate/beta arithmetic and an
independent serial-state oracle. Cover63/64/65 tails, arbitrary initial state,
raw-gate underflow, non-chunk-aligned cuts, clone/restore and continuations.
Full model continuation/output/PPL/MTP acceptance, graph behavior and all
required-region capacity and end-to-end performance gates precede selection.
