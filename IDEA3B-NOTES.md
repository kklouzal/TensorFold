# Idea 3 phase 2 working note (not committed)

Branch perf-idea3-phase2 @ ade71aa4, worktree /Warehouse/TensorFold-wt/idea3b. Local commits only; no push, no GB10.
Policy: /home/kklouzal/CLAUDE.md. Design: scratchpad design-3-drafting.md sections 3-6 (authority).
Phase 1 note: /Warehouse/TensorFold-wt/idea3/IDEA3-NOTES.md (read-only).
Logs: /tmp/claude-1000/-home-kklouzal/052a3fc3-7937-4d3e-a805-81d8238e45a9/scratchpad/idea3b/logs/

## Scope
- "draft_rule" (ab_switch): new = cost-aware stop rule (design 3.4) + round-start trim + joined drafts through the
  same batched rule; old = today's confidence rule (first draft always kept, per-step p >= confidence).
- "suffix" (ab_switch): new = session-only drafter (S1 index over this request's prompt + reply, owed replay c=1);
  old = CopyIndex(8) copy drafts (TENSORFOLD_MTP_COPY=1). Read at admission (a request's drafter is built then).
- Predictor v1 tables (policy x source x depth x score bins), JSON calibration with provenance, loader+validation,
  prior generated from the configured confidence (reproduces today's decisions at production settings).
- Online cost model: prior-regularized exponentially-weighted least squares (RLS with forgetting) per mixedness,
  lambda* ratio-of-EMAs per region (live, mixed, ctx bucket), clamps.
- tools/fit_draft_rule.py (log + traces -> tables, held-out log-loss/Brier/reliability, cost block);
  tools/draft_replay.py extensions (rule counterfactuals on probe rounds; session drafter = runtime index).
- Corpus (design 4.3): NOT built (owner: deferred until GB10 trace replay shows value) -> D18 disposition below.

## Invariants
- Emitted tokens/logits/KV identical for every switch value: only s.drafts change (design 2.3); drafts per stream
  <= depth (room), so windows stay within the captured graph keys (n <= R <= n*(depth+1)) and static tables
  (width depth+1); admission clamp (X9) unchanged; _make_room margins unchanged.
- No new host syncs: predictor/cost model/trim are host math on values already read back; suffix candidates under
  the new rule join the existing pick-0 readback (design 4.4: d0 always available after the absorb).

## Decisions / dispositions (D18)
- Cost-aware stop rule + round-start trim + joined drafts through the batched rule (draft_rule new): IMPLEMENTED;
  selection pending GB10 A/B (and a fitted table: tools/fit_draft_rule.py on GB10 logs).
- Predictor v1 (monotone tables, feature p or m selected by held-out log-loss in the fit tool): IMPLEMENTED.
- Predictor v2 (logistic on m, e, log p, EMA, ctx): BLOCKED (no real logs); the fit tool's held-out report is where
  it would be compared (design 3.3: selected only on held-out log-loss/Brier gain).
- Hybrid agree0 feature: LOGGED ("a0" on text drafts under the new rule), not in v1 tables: BLOCKED on GB10 logs.
- allocate (per-round ratio, engine/allocate.py): evaluator comparison only (draft_replay rules section "allocate").
- S1 session index (anchor 4, recent 32, longest 64, packed-key bulk build off-thread): IMPLEMENTED (suffix new).
- S2 native suffix automaton: NOT SELECTED now (P5: native code + ~40-100 B/token vs numpy S1); its coverage oracle is
  in the replay (Automaton, sam-old / sam-new-w{W}); reopen if GB10 traces show S1 outside an equivalence margin
  (D15) of S2 on acceptance-weighted tokens/round.
- Observed-request corpus (design 4.3) + exact-replay index: DEFERRED by the owner until GB10 trace replay shows value
  (replay corpus-k{K}-w{W} rows measure it from traces; nothing built in the server).
- Owed-replay drafts (c = 1, design 4.2 "other session sources"): IMPLEMENTED under suffix new.
- Grammar-forced single-token drafts: DEFERRED (constrained requests only; needs a measurement of the matcher's
  single-token detection cost and the constrained-traffic share).
- Deeper windows (W 8/16 beyond depth+1): NOT IMPLEMENTED (static tables' width, captured keys n..n*(depth+1),
  buffers and graph memory budget); replay what-ifs suffix-new-w8/w16, oracle-w* quantify the value first.
- Deferred absorb during copy streaks; MTP extension of short suffix chains: DEFERRED (need Tier-2 MTP oracle).
- alpha*L cap: no effect while windows <= depth+1 (L >= anchor 4 >= room 4 at alpha >= 1): not implemented.
- Startup cost curve: NOT IMPLEMENTED (not cheap: dedicated timed forwards at startup, synthetic routing bias); the
  fit tool's cost block (least squares over logged rounds) seeds the online model instead (TENSORFOLD_DRAFT_RULE).
- Drafting-failure containment (design 7.1): unchanged (owner decision pending); a failed background build raises
  where the index is adopted (inside the round), as a CopyIndex failure on the engine thread would.

## Evidence / checks
(appended as run)
- 2026-10-13 dev-01: implemented drafters/session_index.py (S1), drafters/draft_rule.py (predictor/prior/cost model/
  decisions/trim), multi.py integration (switches draft_rule, suffix), draft_log schema (c, L, a0, greedy, cost, ab,
  trace text + per-round rule), replay extensions (suffix-old parity, suffix-new-w{W}, copy8-new, --log rules).
- Smoke (gpu-pytest): test_flashnext_multi.py 36 passed (logs/smoke-multi-1.log); test_draft_log.py (cuda) +
  test_flashnext_round_switches.py + graphs copy test: 28 passed (logs/smoke-2.log) incl. suffix-old parity 2/2 on
  scripted quotes (differential replay check, design 5.5).

## Decisions (SELECT)
- S1 bulk build: packed key (hash high bits | end position) + np.sort: 11 ms vs stable argsort 60 ms at 800k
  (container numpy 2.1.0, dev-01 Xeon E3-1285 v6; bench/sortopts.py) -> packed sort selected; 12 B/token.
  Hash: polynomial with a final multiply (without it the last token only reaches low bits, which the packed key drops:
  caught by the brute-force check).
- Prompt -> int32 conversion on the builder thread in 64k chunks (GIL held ~2 ms a chunk): 25 ms at 800k.
- Prior: generated from the configured confidence (p-binned, edge = float below confidence): chances
  0.9 (p >= conf), 0.35 (first, p < conf), 0.2 (later, p < conf); suffix L >= 8 -> 0.94, L < 8 -> 0.
  Reproduces today's decisions for a lone stream while lambda*·row_ms in (0.253, 0.35]; shared steps (another
  stream forces) may chain a low first draft on (design intent); zero-draft rounds when thresholds > 0.35.
- Window cap: drafts per stream <= depth (room) in every path: R <= slots*(depth+1), per-stream <= depth+1
  (static tables' width and captured keys). Deeper suffix windows (W 8/16) = what-if in the replay only.
- "suffix" read at admission (ab_switch.refresh in admit: the bench flips the control file then sends a request).
- Cost model observed only under draft_rule new (old arm: no added host work); warm() round's costs discarded.
- Broad CUDA run (all test_flashnext_*.py) was 6+ hours (kernel parametrizations): stopped after 544 passed/0 failed
  (logs/cuda-broad-1.log, killed). Targeted affected set: 284 passed (logs/cuda-set-1.log): multi, multi_graphs,
  round_switches, draft_log, round_profile, fill_arrivals, fork_lanes, large_pieces, prompt_cache, drafting,
  memory_limit, pdl, logprobs, vision, labels.

## Finding: online cost-model pathology on the synthetic model (fixed)
- probe/hostcost.py (4 streams, depth 4, copy on, synthetic model; rounds ~6 ms, launch-bound): first version
  new/new 8.7 ms vs old/old 6.0 ms median a round with equal tokens/round (3.67): +1.3 MTP steps a round. Cause: the
  regression's step coefficient went NEGATIVE under the rule's own feedback (steps collinear with drafted rows),
  clamped to 0 -> continuation looked free (logs/hostcost-2.log).
- Fix (SELECT): measure each chain step directly (its launch to its pick's readback, perf_counter only; the readback
  exists anyway) -> step EMA; the regression fits the round less its chain steps (t0, row, ctx, pass). Measured step
  2.1-2.2 ms on synthetic (logs/hostcost-3.log). CPU test reproduces the collinear feedback case.
- Second effect: unseen regions (live 1-3 as streams finish/join) fell back to the prior lambda (0.0466 vs ~0.5
  synthetic) -> extra steps. Fix: nearest seen region (same mixedness/bucket, nearest live count), then every round's,
  then prior; new regions seeded from the nearest; first rounds plain averages then EMA (bias-corrected start).
- After fixes (logs/hostcost-5.log): new/new 6.24 ms median (6.81 mean) vs old/old 6.12 (6.68): +0.12 ms (+2%) on a
  6 ms launch-bound synthetic round; rows/round 4.1 vs 7.3 (the prior drops low first drafts once lambda*·row_ms >
  0.35; synthetic acceptance ~0 so tokens/round equal 3.67 both). In-situ host time a round (mean us): observe 53,
  continuations (4 session indexes) 70 (CopyIndex mean 143 incl. its first build), thresholds 10, mtp 7, lam 6,
  trim 4, abar 3, chain_step 2 -> ~0.16 ms a round of new host work. Not a GB10 prediction.
- Session index per round: optimized (Python-side last k tokens, one searchsorted call): 23 -> 11 us a stream-round
  (no-match case, host python).

## Measurements (dev-01, container amd64-verification-2edfaa51: Python 3.12.3, numpy 2.1.0; Xeon E3-1285 v6)
tools/bench_draft_host.py --sizes 50000,200000,800000 --rounds 2000 --repeats 5 (logs/bench-host.json):
| tokens | S1 build (min/median) | background build wall | engine wake-late p99 (idle p99) | round sync+propose p50/p99/max | repeated-gram query | memory |
| 50k  | 2.8 / 3.2 ms  | 4.3 ms  | 0.19 ms (0.09) | 12.8 / 68 / 324 us | 91 us  | 12 B/tok, 0.6 MB |
| 200k | 12.1 / 13.1 ms | 14.7 ms | 0.19 ms (0.09) | 10.6 / 69 / 218 us | 118 us | 12 B/tok, 2.3 MB |
| 800k | 50.3 / 50.6 ms | 54.5 ms | 0.74 ms (0.10) | 9.8 / 54 / 89 us   | 120 us | 12 B/tok, 9.2 MB |
CopyIndex (A/B baseline): first propose (engine thread) 5.0 / 21.8 / 101 ms; per round p50 30 us.
Rule host math, 4 streams x depth 4 (thresholds, chain steps, text-vs-chain, trim, cost update): p50 69 us, p99 126.
In situ on the synthetic model (probe/hostcost.py, logs/hostcost-5.log): ~0.16 ms a round of new host work.
Full CPU suite (container, --ignore=tests/cuda): branch 143 failed / 7427 passed / 177 skipped; base ade71aa4 export
142 failed / 7384 passed (logs/cpu-full-base.log): identical failure set (env memory-limit TENSORFOLD_CUDA_MEMORY_LIMIT_GB=10
from the wrapper, CUDA-initialized ordering) except test_flashnext_prefix_fork's copy-reservation test: my admit read
self.copy before the slot was taken (its fake lacks .copy) -> FIXED by reading the switch inside admit's cleanup block
(also frees the slot when a bad switch value raises; CUDA test added). Rerun: 19 passed (logs/admit-fix.log).

## GB10 plan (for the primary; nothing run on GB10 by me)
1. Collect (one container of the integration build, benchmark prompts only):
   TENSORFOLD_AB_CONTROL=/cache/perf/<s>/ab.json TENSORFOLD_DRAFT_LOG=/cache/perf/<s>/drafts.jsonl
   TENSORFOLD_DRAFT_PROBE=0.0625 TENSORFOLD_DRAFT_TRACE=/cache/perf/<s>/trace TENSORFOLD_MTP_COPY=1 (prod).
   Arms {"draft_rule","suffix"} in {old,new}^2 interleaved per request (bench/ab_bench.py --control-cmd);
   ctx30k/150k/300k/650k x explain/rewrite, n=1 and n=4, seeds 11/22/33.
2. Validate the evaluator: python3 tools/draft_replay.py <s>/trace --log <s>/drafts.jsonl --json replay.json:
   "copy8 parity" must be N/N on traces of arm (old, old); "suffix-old parity" N/N on arm (old, new).
3. Fit: python3 tools/fit_draft_rule.py --log <s>/drafts.jsonl --traces <s>/trace --output draft_rule.json
   --source provenance.json (model rev, image, commands) --confidence 0.5; read held_out (log-loss/Brier/
   reliability per table vs the prior; selected feature p|m), prior_tables (too few samples), cost block.
4. Evaluate offline: tools/draft_replay.py <s>/trace --log <s>/drafts.jsonl --tables draft_rule.json
   --windows 5,8,16 --anchors 4,8: rules section (confidence vs prior vs fit vs allocate on probe rounds:
   tokens/round, modeled ms, tok/s by bucket); policies copy8 vs suffix-old vs suffix-new-w5 vs sam-old/sam-new
   (S2 coverage oracle) vs corpus-* (deferred corpus) vs oracle-w*; w8/w16 = what-if beyond captured keys.
5. Install the fit for the A/B: mount draft_rule.json, restart with TENSORFOLD_DRAFT_RULE=<path> (validated at
   startup; a bad file fails startup). Rerun the interleaved A/B with the fit.
6. Gates per bucket and n: identical token hashes across all 4 arms (exactness); decode tok/s (paired, bootstrap
   CI); tokens/round; acceptance; rows/round; MTP steps/round; ms/round (round profile); TTFT/prefill (joined
   drafting batched under the new rule); host: tools/bench_draft_host.py in the arm64 runtime image (Grace numbers,
   engine wake-late during an 800k background build); peak host memory (session index ~12 B/prompt token/stream);
   startup with TENSORFOLD_DRAFT_RULE.

## Commits (local, perf-idea3-phase2)
- 137e7e6d Drafting host math: a session suffix index and the cost-aware draft rule
- d8da49a9 Drafting host math: measured MTP steps, nearest-region lambda*, a leaner index
- 3e0549f8 Flash Next rounds: cost-aware drafting and session drafts behind switches
- bb3fef4b Draft rule calibration and host-cost tools; documentation
- Final affected set on HEAD bb3fef4b (container started ~14:02Z after waiting on the GPU lock): 400 passed, 0 failed
  (logs/final-set-1.log): CUDA multi, multi_graphs, round_switches, draft_log, round_profile, fill_arrivals,
  fork_lanes, large_pieces, prompt_cache, drafting (22 incl. int8/int4 and bad-switch admission), memory_limit, pdl,
  logprobs, vision, labels; CPU prefix_fork, prefix_copy, session_index, draft_rule, fit_draft_rule, draft_replay,
  draft_log, draft_margins, copy_drafts, ab_switch, engine_lifetime.
