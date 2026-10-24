# Gemma target greedy max/index candidate

This private candidate starts from corrected selected437. It changes only
Gemma's greedy target verification-loop sampler. CPU, native CUDA,
installed-image, full-model and performance gates are **UNRUN**. Existing FP8
provider-preparation results and private G1/G2 drafter results do not qualify
this change or establish a gain.

## Scope and semantics

`Gemma4Engine.generate` keeps its initial `sample_rows` call unchanged. After
that draw validates the request, it binds `greedy_rows` only for a None policy
or temperature at most zero. Positive-temperature targets retain the public
sampler, including serial requests with drafting disabled. The existing
`stochastic` flag also depends on drafting, so it does not select target policy.
The public shared sampler, other model families, TP/global-ID rules, sampled
probabilities and rejection law are unchanged.

The new Gemma-owned helper keeps the shared row/position/policy validator. It
reduces scores in their original FP16, BF16, FP32 or FP64 dtype. Only the row
maxima widen to FP64; their raw int64 words and int64 selected columns share one
synchronous CPU transfer. Columns are identity token IDs and never pass through
floating-point storage. Scores, graph outputs and packet storage remain owned
and immutable on the caller's CUDA stream until readback completes. No packet,
token memo, new graph key or persistent scratch survives the call.

The actual target head can produce BF16 logits; penalties produce FP32 and
grammar masking preserves dtype. The helper does not assume FP32 input or
narrow FP64 extrema. No head, softcap or penalty arithmetic changes. The adopted
max/index reduction selects the first tied column, propagates any NaN, returns
positive infinity when present without NaN, and leaves an all-negative-infinity
row nonfinite. Thus returned maxima suffice for the complete-row score domain.
Both pinned Torch9026 and ef46 source chains support this integration. The
paired `max_launch_kernel` in `ReduceMaxValuesKernel.cu` uses `MaxOps` and
`GreaterOrNan` from `SharedReduceOps.h`; it is distinct from the preceding
value-only helper in that file.

Before any IDs return, a NaN or positive-infinity maximum anywhere in the packet
raises exactly `ValueError("sampling scores contain NaN or positive infinity")`.
Only then does an all-masked row raise exactly
`ValueError("a complete vocabulary row needs at least one finite score")`.
Partial negative-infinity masks remain valid. Unrelated resource or copy errors
propagate unchanged. Constraints/history, acceptance, EOS/room handling,
rollback, taps and callbacks keep their existing engine paths.

## Maintained checks prepared, not executed

`tests/test_gemma_cuda_engine.py` retains its independent CPU request fixture and
existing assertions. The fixture explicitly supplies both sampler boundaries;
it does not add a production CPU fallback. New controls compare complete replies
with retained legacy sampling, check greedy/temperature-zero/negative/sampled
dispatch with and without drafting, preserve first-draw failure precedence, and
prevent a second sampling call on already finished requests.

`tests/cuda/test_gemma_cuda_target_sampling.py` compares the real helper with the
unchanged public sampler and independent NumPy column choices. It covers four
dtypes, noncontiguous rows, FP64 values outside FP32 range and adjacent maxima,
ties/signed zeros/masks, exact packet-wide error precedence in both row orders,
empty-row validation, current/nondefault stream completion and ownership,
unchanged copy-failure identity, and fresh borrowed CUDA-graph outputs. These are
project packet/lifetime checks, not a replacement library reduction suite.

Root owns execution in the pinned supported environments. Relevant prepared
commands, to be incorporated into the existing guarded source/owner workflow:

```bash
python -m pytest tests/test_gemma_cuda_engine.py -q
python -m pytest tests/cuda/test_gemma_cuda_target_sampling.py -q
python -m ruff check src/tensorfold/families/gemma4/cuda/sampling.py src/tensorfold/families/gemma4/cuda/engine.py tests/test_gemma_cuda_engine.py tests/cuda/test_gemma_cuda_target_sampling.py
git diff --check
```

Required integration retains original full output IDs, greedy serial/drafted
behavior, sampled-law controls, penalties/constraints, cached turns and graph
states. Actual collection and outcomes must be retained without guessed counts;
native CUDA checks permit no CUDA skips. Installed ARM/SM121 and complete model
qualification remain separate from source or local synthetic checks.

## Costs and candidate disposition

The prototype removes whole-vocabulary invalid flags and a selected-value gather
after argmax; paired max itself still scans every score. It adds a small maximum
output, FP64 widening/packet materialization and one request policy binding/helper
import, including short or stopped requests. Those first-use, dispatch and
allocation costs belong to complete request measurements.

Two-word FP64 transport is the initial complete candidate, not a selected
performance policy. A three-word packet with validity flags derived only from
row maxima remains a credible comparator if actual costs justify it. A global
shared-sampler replacement is outside this Gemma-only investigation. Reusing
G2's FP32-only, single-error adapter would violate this target's dtype/error
contract. No cold-start, memory-capacity, throughput or globally optimal result
is claimed.
