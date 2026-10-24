# Existing G1 alternative with current prepared FP8

This private source composes the original G1 implementation from
`e4277966619a6e7ab1b76a0401d11945a9b8c281` into corrected selected437. It is an
existing alternative on the frozen candidate board, not a new algorithm or
variant search. CPU, native CUDA, installed ARM/SM121, full-model, resource and
registered comparative performance qualification are **UNRUN** for this exact
combination. Historical G1/G2 and selected FP8 results keep their original scope;
no additive gain or new deployment selection is claimed.

## Composition boundary

Only G1's existing `sample_rows` import, early-bound native exception-argument
descriptor, complete `DFlash.propose_tokens` method and engine route are ported.
The method consumes the original FP32 `DFlash.logits` producer and existing
shared greedy no-probability sampler. It does not use G2's paired-max packet or
change target sampling. Every produced row is validated before IDs return.
Only the sampler's two exact builtin score-domain errors normalize to
`ValueError("draft confidence scores must be finite")`, retaining the original
cause. Shape/resource errors, subclasses and foreign arguments propagate as
before; exact tuple/string guards avoid invoking arbitrary equality.

The engine binds the method only for enabled, unpenalized greedy drafting with
no confidence cutoff. Sampled proposals, histories/penalties and public
`propose`/Chains retain their original contracts. Logical parent-prefix
acceptance, target verification, constraints, EOS/room, rollback, taps and
callbacks remain unchanged. Source scores and all temporary packet owners stay
live on the engine's current CUDA stream through synchronous readback. No new
cache, graph key, persistent scratch, runtime flag or fallback is introduced.

The current selected constructor's before-ready FP8 preparation is retained
verbatim: after validated private projection/head copies, `_prompt_ext()` then
`_ext()` prepare the selected providers. Managed-state9f89 and the bits8 GB10
Compose profile remain inherited from selected437. No old tree, managed-state
file, profile or constructor was copied over them. The general CLI default16 and
required BF16/affine, sampled and history paths remain unchanged.

## Retained checks and remaining gates

The existing G1 engine fixture/route checks and DFlash boundary/error/native
oracles are adapted into the current maintained test files. Original request,
cache/constraint/history/sampled and numerical assertions remain. Current FP8
constructor ordering/failure controls are retained, including BF16 no-preparation
and failures in reader close, copies, private head and either provider. The
existing native FP8/BF16 comparison adds an exact comparison of this G1 method
with the retained public scored result and the actual unchanged FP8 logits;
its original numerical tolerance and memory-accounting assertions are unchanged.

Root owns all execution through the qualified source/installed workflows. The
relevant maintained modules are:

```bash
python -m pytest tests/test_gemma_cuda_engine.py tests/test_gemma_cuda_dflash.py -q
python -m ruff check src/tensorfold/families/gemma4/cuda/dflash.py src/tensorfold/families/gemma4/cuda/engine.py tests/test_gemma_cuda_dflash.py tests/test_gemma_cuda_engine.py
git diff --check
```

These are required future checks, not executed results. Root must retain actual
collection/outcomes and source/origin/provider/owner/retirement proofs; CUDA-only
cases skipped in a CPU role do not qualify native behavior. Source/native checks
do not replace installed ARM, complete original model-ID/counter quality and
resource/registered P3 gates. The active independent G2/F8R cohort's sources,
outputs and timing data remain frozen and are not reused as this arm's results.

## Costs and selection

G1 omits unconsumed confidence normalization and returned score/parent-list
materialization, while retaining the required logical accepted-prefix walk. Its
three-word sampler packet, full-row invalid flags, range validation and request
method binding have real costs. Selected FP8 preparation still belongs before
ready and counts in startup. First-use, cold/warm, complete request, memory and
copy costs must be compared on the same qualified source/image/model/region.
Root will qualify and rank this existing alternative; no win, equivalence,
unique optimum or completed scope is asserted here.
