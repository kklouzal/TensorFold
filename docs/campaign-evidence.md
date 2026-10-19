# Campaign evidence boundaries

`campaign_gates.py` identity rows include `response_tokens` for `drafted` and
`serial`. Determinism rows include counts for `cold`, `warm`, `recold`, `extended`
and `extended_cold`. Each count is the length of the already validated typed token
IDs, which must match that response's `usage.completion_tokens`. Existing `tokens`
fields retain their drafted/cold meaning. These fields add no request, budget,
filter or new gate verdict: sampled serial differences remain reported, and two
equally short extension replies can still satisfy determinism. A fixed-length
qualification consumer must separately require its declared count for every
required peer, rather than interpreting a gate PASS or the single `tokens` field
as proof of complete output. Reported-only next-turn probes keep their existing
status and applicability. Installed/model qualification remains separate.

`campaign_kl.py dump` propagates the production CLI's nonzero result even after
the scoring hook completed, and always restores its temporary serving hook.
A written dump is not success if later CLI cleanup failed. Existing teacher-forced
token sequences, statistics and numerical gate rules are unchanged.

`campaign_bench.py run` retains the existing excluded warmup's `client` timings:
`headers`, `first`, `last` and `done`, each in seconds from that request's
submission, or null when unavailable. These use the existing monotonic timestamps
and match measured reply timing fields. They expose first-use request cost
separately from the measured schedule; no extra request, warmup, synchronization
or server-startup timing is introduced. Warmup remains excluded from benchmark
records and aggregate comparisons.

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
