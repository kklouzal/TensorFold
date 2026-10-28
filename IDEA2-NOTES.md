# Idea 2 working notes (uncommitted; X1/X6 ledger for perf-idea2-round)

Worktree /Warehouse/TensorFold-wt/idea2, branch perf-idea2-round, base 328489c5 (perf-ideas-20261013 tip).
Authority: design-2-graphs.md (scratchpad), CLAUDE.md policy, perf ledger (read-only).
Logs: /tmp/claude-1000/-home-kklouzal/052a3fc3-7937-4d3e-a805-81d8238e45a9/scratchpad/idea2/logs/
CPU tests: scratchpad/idea2/bin/cpu-pytest.sh (docker image, no GPU, no lock). GPU: gpu-pytest.sh (flock).

## Facts checked (code)
- Production experts are resident grouped `moe_mod.moe` (no `--vram-experts`): no host plan read in decode.
  The ledger's lead (forward.py `_ExpertWindow.read` ready.synchronize per MoE layer) applies only to
  `CachedExperts` (--vram-experts) -> NOT production. The only drain inside verify `compute()` is the trailing
  `_kv_check` (RotorQuant status readback).
- Production KV rotorquant6-norm -> every State has kv_status (2 int32); bf16/int8/int4 have none.

## Stage 1: B1 (switch "b1": new|old)
Decisions:
- compute()/mtp_compute() take check= (default True: every other caller unchanged). MultiDecoder passes False.
- KVWords (forward.py): pending states' 2-word status copied (non_blocking) into decoder-owned pinned
  kv_host[slots,2] behind the sampler's / draft pick's packet D2H; one stream synchronize; latch in stream
  order (State.kv_check_words == kv_check's latch without the readback); `after` callback (MTP: set_mtp_len /
  mtp_drafted) runs only once the frame is valid; once-only (checked flag).
- Precedence preserved: any failure between the forward and the readback (grammar masks, launch, picks
  validation, copy-draft proposals) calls settle() first (queue+wait+latch) so the KV error wins, as today's
  check right after the forward did.
- sampling.py: sample_streams refactored into shared helpers (_stream_groups/_group_packet/_group_choose) with
  today's exact order (nucleus drawn inline, one .cpu() per group); new launch_streams/collect_streams (nucleus
  streams drawn after the single wait; packets via .to(cpu, non_blocking) + one stream sync).
- Profile split (both arms): verify_enq = host enqueue (+ GPU event). Baseline: kv_wait (the drain),
  mtp_absorb_wait, chain{j}_wait. New: sample_enq, sample_wait (the drain), pick{j}_wait.
- Baseline arm = today's ops: compute(check=False) immediately followed by _kv_check(segs) (nothing between in a
  pure round), _mtp(check=False) immediately followed by _kv_check.
Open: mixed rounds keep compute_mixed's own check (unchanged; KVWords then finds nothing pending).

## Commits so far
- 4f149ba1 B1, f786c35c B3, 0b402840 B2, 71631fb8 A (static tables + graphs), eb80732e fix (B2 profile timer
  `ple_s` clobbered Buffers.ple_s = n-gram scales: wrong logits with b2 new; caught by the B2 SSD test), 45e3da16
  test fix (growth check over kept slots).

## Test runs (GPU, dev-01; logs/)
- b123-gpu-1.log (snap b2 = 0b402840): 74 pass / 4 fail -> B2 new tokens wrong (ple_s collision, fixed eb80732e),
  profile test assumption (chained rounds; fixed), test import (fixed), draft-log spy signature (fixed).
- a1-gpu-1.log (snap a1 = A working tree): 23 pass, 1 fail (test assumption: dec.free empty when prompt ends are
  kept; fixed 45e3da16). Bit-equality per key (verify both parities, MTP) eager-static vs old vs replay: PASS.
- CPU full suite on B1 tree: 7489 pass / 177 skip (logs/b1-cpu-full.log).

## Stage A decisions
- Piecewise capture with eager inserts (select per attention layer, n-gram conv per stream, late n-gram stage);
  idea 1's T/B selection stays eager between pieces (its grid follows the longest context): exact and keeps idea 1's
  policy; stage D (context-independent select geometry) not taken (D18: deferred, needs design-1 coordination;
  persistent-loop scores + _select_tiles would regress long contexts vs T).
- Keys: verify (PDL, n, R, parity), MTP (PDL, n, R). Vision rounds, lookup-decoder rounds, mixed rounds: eager.
- Capture at warm() on the idle slots, eager warm-up per key, slots reset after; capture errors propagate.
- Memory: TENSORFOLD_MULTI_GRAPH_MIB (default 512) planned in indexed_stream_geometry (truthful admission);
  measured bytes = drop of free memory (MemAvailable on unified, card free otherwise; uncapped) taken off
  memory_gate.room; capacity_plan["multi_graphs"] records keys/bytes/time. Budget overflow drops the key and stops.
- Static tables also counted in the geometry (KBs).

## Stage C (PDL) design
- tensorfold/cuda/pdl.py: select() once per decoder (GB10 12.1 on; others off; TENSORFOLD_PDL=1/0 forces on sm_90+);
  ON global set each round = chip policy and ab "pdl". Triton kernels: constexpr PDL + pdl.wait(PDL) as the first
  statement (gdc_wait then gdc_launch_dependents) + launch_pdl=pdl.ON; PDL=False compiles today's code.
- CUDA: kernels/pdl.cuh (pdl_wait first statement, tensorfold::launch with the attribute); qmm.cu C1: weights of
  the prologue stages before grid_wait, x/xs after (qmm_group's proven pattern), reduce_kernel waits; gdn tree/
  replay, gdn_io front/back, experts plan(4)/run/prefill wait first. Extension names bumped (qmm v6, gdn v3,
  gdn_io v2, experts v8) so the shared build cache does not ping-pong with other worktrees.
- Graphs keyed by PDL; both variants captured only when an A/B control file is configured (ab_switch.active()).

## dev-01 measurements (diagnostic only; RTX PRO 2000 sm_120 shared; synthetic weights D=1024, 64 experts)
- logs/bench-a2-l48.{log,json} (snap a2 = 45e3da16): 48 layers (36 GDN + 12 attention, PLE bf16 table on layer 2),
  4 streams (one 2,300-token sparse prompt + 3 short), depth 4, conf 0.5, RQ6, 40 tokens, 48 rounds, x2 repeats,
  round wall median (p10) ms:
  today's path 48.95 (45.9) / 55.53 (52.4); B1+B2+B3 49.95 (47.7) / 56.00 (54.9); + static tables eager
  50.49 (47.0) / 47.85 (46.2); + captured graphs 22.78 (21.1) / 21.97 (20.8). Tokens identical across all 8 runs.
  Repeat-to-repeat noise ~6 ms: B1-B3 and static-table effects not resolvable on dev-01; graphs ~2.3x (this synthetic
  is launch-bound: small weights). GB10 decides.
- Graph capture, 48 layers, slots 4 depth 4: 88 verify + 44 MTP keys, 1,408 pieces, 8.3 s, 363 MB measured
  (~2.75 MB a key, ~3.5 KB a captured node; ~16 pieces a verify key: 1 + PLE stage + PLE conv + 12 selects + ...).
  2-layer model: 264 pieces, 10 MiB, 0.7 s. A second capture in the same process measured 2 MiB: the driver reuses
  memory of destroyed graph execs, so the measure is valid for the first capture of a fresh process (production).
- Implication: GB10 (same node counts) ~350 MiB per PDL variant; an A/B session capturing both PDL variants needs
  TENSORFOLD_MULTI_GRAPH_MIB >= ~800 or the pdl-off keys are partly dropped (biased A/B).

## Stage C results
- c2-gpu-1.log (snap c2): test_flashnext_pdl (selection, lane matmul under PDL x 20 shapes incl. clustered/staged
  K slices, decode eager+graphs flipping "pdl" each round, 1,000 PDL-graph replays bit-equal), shared kernel
  contracts, test_gdn, gdn_io, experts, test_qmm: 133 pass, 4 fail = test_qmm::test_27b_triton_bits (qmm_group
  gs-64 vs qwen3_5 Triton on sm_120): PRE-EXISTING (fails 5/5 on snap a2 without any QMM change; logs/a2-qmm27b.log).
- Bug found by the PDL lane-matmul test: helper `reduce<F32>(...)` inside qmm_cuda parsed as comparisons against its
  bool parameter `reduce` (staged K-slice sum never ran); renamed reduce_slices before commit.
- Build with -Wall -Wextra (host) and nvcc for qmm/gdn/gdn_io/experts: no warnings (logs/c2-warnings.log).
- CPU contract test tests/test_pdl_kernel_contract.py: wait-first over all converted Triton/CUDA kernels.

## Dispositions (D18)
- B2b row-level read-ahead: not implemented; needs a row-granular cache in ReadAhead (whole-array keys, intricate
  close protocol). Decide after GB10 ple_ms vs layers 0-1 GPU time: if the late read already hides behind them,
  B2b has nothing left to hide.
- B3 draft packet fusion (max+logsumexp+topk one pass): ruled out (exactness: torch logsumexp reduction order).
- A4 (packet ops inside the last piece): deferred, measure eager packet cost first.
- Stage D (context-independent select geometry, one graph a key): deferred (design-1 coordination; T would regress).
- O10 static VISION: not taken; vision rounds eager (exact). Candidate if GB10 shows image streams matter.
- Per-State pointer-block cache by version: rejected (tensors can be replaced without a version bump:
  test_streams_keep_their_bits_when_their_caches_move); GDN pointers now by stride arithmetic instead.
- C4 (replace in-graph torch ops with Triton to keep PDL chains): not done; measure first.
- Pre-bound Triton runners for eager mixed rounds: open (mixed rounds stay eager).
- Device acceptance: ruled out (P0). B4: excluded (owner).

## Final verification (snap final = 6b82e3d9; 6e7e4c27 adds a CPU test fix only)
- CPU full suite: 7499 pass / 177 skip / 1 fail (test_cuda_kv_dtype admission: plan lacked the graph budget; fixed
  6e7e4c27, rerun 339/339 of the admission files) (logs/final-cpu-full.log).
- GPU (dev-01): final-gpu-1.log (stopped after 578 pass / 0 fail: too broad, container stopped by me),
  final-gpu-2.log 456 pass (yarn/vision concurrent image paths, logprobs, fork lanes, fill arrivals, prompt cache,
  pdl, rotorquant multi/graph/failure/capacity), final-gpu-pdl.log (TENSORFOLD_PDL=1): 87 pass (multi, multi_graphs,
  round switches, profile, engine --parallel 2 with graphs).
- dev-01 PDL bench (logs/bench-final-pdl-l48.*, TENSORFOLD_PDL=1, both variants captured, 1 GiB budget): graphs
  22.97->22.40 / 21.51->21.34 ms, eager 48.51->48.02 / 47.53->47.55 ms (pdl old->new): within noise on sm_120.
  Both variants: 176+88 keys, 2,816 pieces, 742 MB first capture; capture 240 s first run (Triton JIT of the PDL
  specializations), 15.8 s once compiled. Tokens identical across all variants.
