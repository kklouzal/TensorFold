# Campaign evidence boundaries

`campaign_gates.py needle --needle-evidence-out NEW_DIRECTORY` optionally retains
one full parsed request/reply record for each existing inference. The directory
must be new. `index.json` records corpus identity, seed, fractions, request files
and held work; each request file contains exact streamed request fields, expected
codes, target/mode, complete content/reasoning, token IDs, usage, finish and stats.
It also retains the existing final character bracket and one upper-end token count,
allowing a consumer to distinguish target coverage from a corpus-limited shortfall
without another sizing search or inference.
Publication uses the existing atomic benchmark writer before and after inference.
A failed operation retains its request/error and propagates; publication failure
does not replace an inference failure or its nested cause chain. When both fail,
a supported exception group retains the original inference exception and the
publication exception independently. No second request or new sizing loop runs.
The callable `gate_needle(..., evidence_out=PATH)` exposes the same bounded output.

This evidence does not change the diagnostic needle metric: `found` still means
the expected code appears anywhere in the answer, and its summary answer remains
limited to600 characters. It is not an exact requested JSON map or completed-answer
quality pass. A qualification consumer can inspect the complete record under its
declared finite known-answer contract, retaining native baseline misses and any
candidate miss of a code the matching native reference retrieved. No new recall
threshold or numerical margin is implied.

Determinism's extra chat next-turn probe now checks the actual rendered input
plus reply budget against the served context before submission. If it cannot fit,
the reported-only witness is held with explicit counts/reason and the gate is
incomplete. Its required exact-token-ID extension and warm/cold comparisons keep
their existing budgets, equality and90%/10% exercise rules. Neither a long-prefill
override nor available memory permits an over-context request. A separate near-full
cache witness must leave space for both its first reply and extension; it cannot
replace the larger full-window benchmark/identity workload.

These are preparation and retention changes. They add no model/native/long-context
qualification or performance claim. Actual installed source/image/config, tokenizer,
templates, corpus, CPU affinity, host reserve and outer guard/retirement evidence
remain required for any executed stage.
