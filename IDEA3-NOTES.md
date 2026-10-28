# Idea 3 phase 1 working note (not committed)

Branch perf-idea3-drafting @ b536f2c3, worktree /Warehouse/TensorFold-wt/idea3. No push, no GB10.
Policy: /home/kklouzal/CLAUDE.md. Ledger: /home/kklouzal/.tensorfold-perf-ledger-20261013.md.
Design: scratchpad design-3-drafting.md (sections 1, 2, 3.2, 3.6, 5).

## Scope (phase 1: instrumentation only, no drafting-policy change)
1. Keyed margins in MultiDecoder._picks -> (draft, p, margins|None); shared helper keyed_scores/keyed_margins in
   engine/exact_sampling.py; greedy gap from top-2 in the same single readback. Drafts bit-identical.
2. TENSORFOLD_DRAFT_LOG (JSONL, no token ids), labels resolved as s.out grows, TENSORFOLD_DRAFT_PROBE probe rounds
   (shadow drafts labelled, never verified). Background bounded drop-oldest writer (shared with round_profile).
3. TENSORFOLD_DRAFT_TRACE=<dir>: per finished request JSON (prompt ids, out ids, sampling, rounds), mode 0600.
4. tools/draft_replay.py offline evaluator.

## Invariants
- Emitted tokens unchanged (drafts only change rows); drafts unchanged by margins/log/trace; probes add only shadow
  chain steps (trimmed by mtp_drafted next round).
- Off: watch None, one `is not None` test per hook; _picks margins=False path = today's readback/ops.
- No added host syncs: margin mode adds topk(2) into the same torch.cat readback.

## Decisions
- Margin mode only when the draft log is on (zero cost off). Greedy margin = top1-top2 raw FP32 gap.
- Edge margin over the policy's own top_k+MARGIN sorted candidates (whole row when top_k=0); +inf when none cut.
- JSON: +inf encoded as the string "inf" (strict JSON).
- Join drafts (single-stream draft()) logged with p/m null (join=true); probes only in _draft_all.
- Probe without log -> ValueError at decoder init.
- CopyIndex for the evaluator loaded by AST from src (torch-free), as tests/test_copy_drafts.py does.

## Checks / evidence
(see below, appended as run)
- Baseline (wrapper, keyed_scores refactor already in): test_exact_sampling, greedy_picks_control, copy_drafts,
  ab_switch, cuda/test_round_profile, cuda/test_flashnext_multi: 56 passed, 2 skipped (logs: scratchpad idea3-logs/baseline.log)
- Host CPU: test_draft_margins/test_draft_log/test_draft_replay/test_exact_sampling/test_copy_drafts 33 passed 2 skipped;
  greedy_picks_control 10 passed (3 new margin-mode controls: single readback, topk(2) in same cat).

## Files
- engine/exact_sampling.py: keyed_scores (shared), choose_rows = argmax(keyed_scores), keyed_margins, value_gap
- cuda/record_writer.py: RecordWriter/JsonLines/JsonFiles (bounded drop-oldest queue, daemon thread, close())
- cuda/round_profile.py: now on JsonLines (test uses writer.close())
- cuda/draft_log.py: DraftWatch (log, probe, trace)
- families/qwen4_exp/cuda/multi.py: hooks; _draft_all returns MTP steps; _chain (probe); _picks margins
- tools/draft_replay.py; tests/test_draft_{margins,log,replay}.py; tests/cuda/test_draft_log.py

## Review dispositions (D13 cleanup agent, D14 policy agent; read-only, owner applied)
- D13 B1 / D14 #6 joined count lost by second begin(): FIXED (joins reset only when a line is written) + CPU test.
- D13 1 _chain loop simplification: APPLIED (reviewer brute-forced equivalence; CUDA drafts-equal tests).
- D13 2 _picks scores building: APPLIED. D13 3 dead RoundProfile.path: REMOVED. D13 4 steps vs self.chained: APPLIED.
- D13 5 pass-row sum x3: APPLIED. D13 6 unreachable _rid fallback: REMOVED.
- D13 7 duplicated keyed formula: APPLIED as _keyed/_nucleus shared by choose_rows and keyed_margins (also removes
  double noise for top_k=0). Host timing: choose_rows ~76 us, keyed_margins ~88 us per 28-cand row; full 79,591 row
  10.8 vs 10.6 ms. repaired_choose(margins=list) scores once (log on): ~+12 us per sampled draft row.
- D13 8 / D14 2 trace doc on engine thread: list() copies + json default=int. D13 9 quadratic corpus: segments.
- D14 1 writers never closed: MultiDecoder.close_telemetry() from engine._close_resources (last); close raises on
  failure/timeout. Engine lifetime fixture extended (telemetry-close event).
- D14 2 dead-writer memory / silent trace drops: put() drops+counts after failure; real evictions only; trace docs
  carry "dropped"; evaluator reports it.
- D14 3 JsonFiles: makedirs at init (0700), fsync, unlink .part on failure; JsonLines opens at init.
- D14 4 evaluator defaults/BASE_MATCH: required fields validated in load(); copy_match in trace decoder dict.
- D14 5 corpus: per decoder process, causal by finish order (order/after), background excluded, segment FIFO.
- D14 7 NaN: value_gap(-inf,-inf) -> inf; _num(NaN) -> "nan".
- Advisories: probe RNG fixed seed 0 (APPLIED); constrained per-stream field (APPLIED); hook-error behaviour
  documented; failed-round CPU test added; warm() suppression covered by CUDA test (warm in conf 0.9 case).
- NOT FIXED (pre-existing, out of scope, report): drop() skips streams already done after a round raises -> never
  finish()ed (slot leak). Probe-batch-composition MTP invariance verified only on the synthetic model.
- Off-path cost: structural; adds one _chain call per drafting round and two cheap tests per pick (~us/round),
  no GPU work, allocation, threads or syncs.

## Synthetic-model note
The synthetic model never repeats an 8-gram, so copy drafts never trigger naturally: the scripted CUDA test patches the
target draws (by seed) to quote the prompt; decoder copy/accept/MTP machinery stays real; copy8 parity 2/2.
test_flashnext_exl3_fixture asserts CUDA uninitialized: fails only when run after CUDA tests in one process (ordering).

## D14 re-check (after remediation): 1-7 fixed; new A (close stops at first failing writer; engine close fails
forever) FIXED via record_writer.close_all + close_telemetry releasing writers (report once) + tests; B (keyed_margins
array width vs choose_rows) FIXED: kept scores computed on choose_rows' exact arrays, tail separately; dropped summed
per process.

## Final evidence (wrapper, image amd64-verification-2edfaa51)
- CPU broad set: 1510 passed, 3 skipped (mlx) -> idea3-logs/final-cpu.log
- tests/test_flashnext_exl3_fixture.py alone: 2 passed (combined-run failure reproduced on base b536f2c3 export:
  pre-existing ordering) -> final-exl3-fixture.log, base-cpu.log
- CUDA set (draft_log, round_profile, flashnext_multi, fill_arrivals, fork_lanes, large_pieces, prompt_cache,
  sampling): 116 passed -> final-cuda.log; earlier run incl. test_draft_margins: 120 passed (cuda-5.log)
- Commit 32b4ab99 on perf-idea3-drafting.
