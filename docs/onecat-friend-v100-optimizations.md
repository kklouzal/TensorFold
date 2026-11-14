# Friend's 1Cat-vLLM V100 deployment: optimization catalogue and TensorFold adoption map

Status: research record (read-only survey, 2026-11-14). Maintained document: update the ranked list when an item is
adopted, rejected or measured on our side. Sources are mirrored (text only) at `/home/kklouzal/build-cache/friend-1cat/`
(`MANIFEST.md` lists source paths and sha256 values). Paths below of the form `R/...` are under
`build-cache/friend-1cat/bench/results/`, `SNAP/` is the production snapshot
`build-cache/friend-1cat/snap/lp-snap-c1es8-.../`, `top/` and `misc/` are the deploy-root copies.

Evidence labels: **VERIFIED** = number found in a result file (path given); **claim** = README/STATUS text without a result
file; **contradiction** = the friend's own gate disagrees with what was adopted. "Ours" = TensorFold SC117 on our host.

## 1. Summary

- Host: the same Supermicro 8x V100-SXM2-32GB server, same NVLink hybrid cube mesh (GPUs 0-3 on NUMA node 0, 4-7 on node 1),
  same 2x Xeon Gold 6130. Software: upstream 1Cat-vLLM at commit 4d57473d (2026-10-29), checkout left clean; every change is a
  patch file (md5/text-anchored, default off, env-gated, live JSON knobs) applied as a thin Docker overlay.
- Two models were optimized:
  1. **Qwen3.8-Flash-Next** (huihui abliterated, unsloth UD-Q4_K_XL GGUF; NOT our IQ3_S) on **PP2 x TP4** (one TP4 group per
     NVLink quad, one pipeline stage per NUMA node), **no MTP in production**, KV `rotor4_planar` (their 4-bit rotated KV),
     max-num-seqs 32, 512K ctx (YaRN). Gate 2026-10-31 (VERIFIED `R/seq49-gate.json`): decode **82.1 / 460.9 / 721.4 / 981.7**
     tok/s aggregate at 1/8/16/32 users, prefill **6853 / 8262** tok/s at 8K/32K. Re-measured 2026-11-10 (`R/seqqcycle2-1.log`):
     c1 83.1, c2 161.0, c4 280.5, c8 474.6, c32 987.4, prefill 32K 7804. Qwen patches: 01-79, 100-106, q* overlays, 400.
     With MTP tiers they reached c1 81.5-91.5 but lost at >= 16 users (section 3).
  2. **DeepSeek-V4.1-Flash** GSQ-RCO IQ2_XXS (+ Engram IQ4_NL tables in host RAM) = **current production** since ~2026-11-01,
     also PP2 x TP4 (20/20 layers), no speculation, model-native compressed KV, 1M ctx. Decode (VERIFIED `R/seqcycle2rs-2.log`
     arm A, 512 in/512 out): **C1 58.7, C2 115.7, C4 203.7, C8 357.6, C32 818.3**; C16 572.8 from an older image
     (`R/seqdsgraph-1.log`). Patches 80-99, 110-503.
- Their 1Cat "82.9" style C1 on Flash-Next was reached **without speculation**: ~12.2 ms/token, almost pure GPU time
  (stage-0 graph 5.64 + eager pre-graph 0.53 + stage-1 graph 5.37 ms; stage-1 wait 0.003 ms; `R/qovl1/t1-off1-probe-c1.txt`).
  Ours: ~46 ms round / ~2.3 tokens = ~20 ms/token with ~16 ms host-serial work. The gap is structure (host off the critical
  path, two batches in flight, graphs everywhere), not individual kernels.
- **Power cap (new, needs user decision):** our GPUs report a **150 W** power limit (`nvidia-smi`, 2026-11-14); theirs
  225 W now (nvidia-power-limit.service; every DS41 window logged 225 W) and 300 W at the Oct 31 Qwen gate (STATUS.md);
  V100-SXM2 default is 300 W. Their Qwen prefill dropped ~5% between Oct 31 (300 W) and November (225-230 W) at unchanged
  decode (hypothesis, not isolated). Every number in this document was measured at 225-300 W; our 150 W cap is a confound
  for every comparison, and it most affects compute-bound phases (prefill, verify at C8).
- Their CPU governor is `performance` with C-states on (prefill +1-3%, decode neutral for them because their host is off the
  critical path; C-states off was worse). Ours is `powersave` by user decision; since our decode is host-bound, an A/B would
  be informative but needs user approval.

## 2. How they structured the deployment (architecture we have not tried)

| Aspect | Friend | TensorFold today | Note |
|---|---|---|---|
| Parallel layout | PP2 x TP4: TP inside each NVLink quad, PP across NUMA nodes; experts TP-sharded, all GPU-resident | TP8 non-expert / ETP4 experts across both quads | TP4-in-quad already ledger item 13 (user decision). Their evidence: PP2xTP4 = 97-99% of DP2 (2x TP4) throughput with 2.56x the KV pool (8.85M vs 1.73M tokens/engine; `bench/results/*-kv.txt`, seq21/seq26) |
| Process model | vLLM V1: API proc + EngineCore (scheduler) + 8 spawned workers, NUMA-bound (`--numa-bind-nodes [0,0,0,0,1,1,1,1]`); SchedulerOutput via shm broadcast queue; DS41 Engram 110 GB held by a sidecar holder process (memfd, mlocked, SCM_RIGHTS) | rank processes + coordinator rank; TCPStore/FIFO control plane | Holder process = restart-time trick (Engram reattach 252 s -> ~9 s) |
| Host hiding | Two micro-batches in flight across two stages; host prepares k+1 while GPUs run k; balanced decodes per micro-batch (18); last-stage sampled-token D2H async with an output thread (262) | one round at a time per server; host prep serial with GPU (~16 ms of ~46 ms) | Already ledger item 1; their probe: stage idle 7-10% at C8 with 2.2 ms host prep fully hidden; 55% idle at C1 (one micro-batch) |
| Graphs | FULL graphs for uniform decode at **every** batch size 1..16, 24, 32 (DS41) / 1-8,10,12,14,16 (Qwen); mixed/prefill eager | round graphs keyed (streams, rows); attention owner `row_buckets` default powers of two (m-profile7 `"row_buckets": {}`) | Holes in the capture grid cost them 4.5-20% (section 4) |
| Speculation | none in production for either model | MTP4, confidence 0.7 | Their MTP crossover: wins C1, roughly ties C8, loses >= 16 users |
| Scheduler | sync V1 + PP balance; prefill caps per request (185), per pass (269), time share 0.5 (271) with trains (277), small-first (273), lookup skip (275) | tp_scheduler decode_share credit, fewest-left fill | multi-user mixed-load items, not C1 |
| KV / prefix | Qwen: rotor4_planar (3.1x tokens vs FP16); DS41: GPU anchor snapshots (230) + /dev/shm host tier surviving restarts (231), ladder anchors (390), tail anchors (502), superseded-first eviction (470) | int8 KV; prefixes.py LADDER + GreedyDual; in-process RAM tier | Agent-workload TTFT items |
| Patch discipline | every change: anchored replacement, default off, engagement log line, live knob, verify-N mode against the original path, gated A/B/A2 with predeclared bars | ledger + golden oracle | Their "flag on != feature ran" lessons (73, 105, 111) |
| Host system | governor performance, C-states on, GPU power 225 W (300 W earlier), OMP_NUM_THREADS 1, MALLOC_ARENA_MAX 2, NCCL_BUFFSIZE 4 MiB, NCCL_MAX_CTAS 8 (Qwen prefill +1.3-1.5%) | governor powersave (user), **150 W** | see section 7 |

## 3. Like-for-like Flash-Next numbers (aggregate decode tok/s; prefill tok/s at ~8K/32K)

All friend rows: UD-Q4_K_XL, same server model. Full list with labels in Appendix A.

| Config (result label) | Layout | Spec | KV | c1 | c8 | c16 | c32 | Prefill | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| tp4-baseline | TP4 | MTP4 | fp16 | 99.1 | 201.0 | 124.3 | - | 713 / 714 | R/tp4-* (best C1 ever; "original" expert storage, batching collapses) |
| tp4-canonical-mtp3 / -mtp2 / -nomtp | TP4 | 3 / 2 / none | fp16 | 88.2 / 82.5 / 64.9 | 344.9 / 372.8 / 283.7 | 318.3 / 324.5 / 440.3 | - | - | R/tp4-canonical-* |
| tp8-nomtp | TP8 | none | fp8 | 31.7 | 153.7 | 54.3 | - | 664 / 686 | not like-for-like (eager, compile 0); dropped; TP8+MTP impossible with stock draft (Q8_0 640/8 not divisible by 32) |
| h-pp-tiers / h-pp-nomtp | PP2xTP4 | tiers 4:2,3:5,2:11,0 / none | planar4 | 81.5 / 61.0 | - | 413.2 / 456.5 | 665.1 / 695.9 | - | seq21 |
| m-d0 (tiers + depth-0 patches 50-52) | PP2xTP4 | tiers | planar4 | 91.5 | - | 600.9 | 843.3 | - | seq28 (candidate only; NaN bug) |
| cand6 (Oct 31 production) | PP2xTP4 | none | planar4 | **82.1** | **460.9** | **721.4** | **981.7** | 6853 / 8262 | R/seq49-gate.json |
| Nov 10 production | PP2xTP4 | none | planar4 | 83.1 | 474.6 | - | 987.4 | - / 7804 | R/seqqcycle2-1.log |
| **TensorFold SC117** | TP8 / ETP4 | MTP4 conf 0.7 | int8 | ~48.5-53 | ~114-123 | n/m | n/m | 4K ~1770, 16K ~2520 | our ledger (different prompt set; 150 W) |

Readings: (a) at C8 they are 3-4x ours **with or without MTP**, so speculation policy alone does not explain our C8 gap;
(b) their MTP tables show fixed depth 4 collapsing with concurrency (TP4 MTP4 c16 124 vs no-MTP 440), MTP2 best at c8, and
load-dependent tiers (patch 05) +11.6% c8 over fixed depth 4 (pp-mtp4 310.6 vs tiers 346.5); fixed depth 2 cliffs at c12
when rows exceed captured graphs; (c) their best no-spec C1 (82-83) equals our 1Cat target, reached through structure + MoE
kernels, not speculation.

DS41 trajectory (VERIFIED, Appendix C section 2): base multi-user port c1 57.3 / c8 273.3 / c32 455.8 -> +110 expert-major MoE
c32 617.4 (+35%) -> +89 row-wise kernels to 64 rows 669.7 -> +112 dense one-weight-pass 697.8 -> ... -> +262 async PP output
c8 +9.2%, c32 819.6 -> production 58.7 / 357.6 / 818.3. Prefill 8K 1422 -> 3605 (+140 prefill MoE on grouped HMMA +79%,
+181 fused sparse attention +23%, +184 cuBLAS for dense >= 2048 rows +11%, +193 per-bucket GEMM tuning +5%), 32K up to 4217.

## 4. Why they scale with concurrency (mechanisms, model-agnostic)

1. **Forward time nearly flat in rows**: DS41 stage forward 8.5 ms at 1 row vs 9.6 ms at 4 rows (patch-255 probe). Achieved by
   reading each distinct expert's bytes once per step (expert-major MoE 110; distinct experts D(M)=E(1-(1-k/E)^M)), dense
   kernels that dequantize a weight fragment once for up to 32-40 rows (112/122), and row-wise fast paths extended from 6-8 to
   64 rows (89). For Qwen: per-route compact MoE grid at M<=16 (72/74/77, c1 +12%, c8 +15%, +11%), one slot per distinct
   expert from M>=16 (75, c32 +7.4%; it lost 6% at c1/c8 when enabled below 16 rows).
2. **Two batches in flight hide host work** (PP micro-batches + balance + no idle sleep + async last-stage output). Three in
   flight lost 30% (137: each extra batch re-reads weights).
3. **Every batch size has a graph**: filling capture holes gave +4.5-9.6% (DS41 seqdsgraph-1 / seqdstune-2) and +19.9% at a
   10-user load on Qwen (qtrim GRAPHS: 451.8 -> 540.5, `R/seqqtrim-1.log`).
4. **No hidden host syncs**: patch 27 (sampler `.any()` sync removed) +32/27/21% at c8/c16/c32 because the sync stopped the
   last stage's host run-ahead (GPU busy 73% -> 94-99%).
5. **No host work proportional to context or requests per step** (173 history shipping, 183 input cache, 186, 480 tail copy):
   decisive at long context (c16 x 200K 64 -> 413 tok/s with indexer fixes).
6. **No speculation at concurrency** (DS41 spec analysis projected c8 x0.748; Qwen MTP loses >= 16 users).

## 5. Ranked adoption list for TensorFold

Estimated effects are reasoning from their evidence plus our ledger round breakdown (C1 round ~46 ms: verify GPU ~25 ms,
host ~16 ms, MTP steps); none is measured on our side. "Known" = already in our ledger (1Cat/Strata surveys or agent
sections); the friend's data adds magnitude and a recipe.

| Rank | Item (friend source) | TensorFold mapping | Expected effect | Status vs ours |
|---|---|---|---|---|
| 1 | **Host off the critical path: device token handoff + next round prepared while the current one runs; async readback via output thread** (22, 52, 262, PP micro-batches) | `families/qwen4_exp/cuda/multi.py` round loop, `tp_scheduler.py`, `control_pipes`, post-verify sample/commit | C1: up to ~10-14 ms/round (their C1 token is pure GPU); C8: +4-9% from output thread alone | Known (ledger survey item 1). Friend hit our blocker too: async depth only when no drafts are in flight (52) |
| 2 | **Speculation depth policy by active stream count** (05 tiers + hysteresis + graphs per tier; DS41 spec analysis) | `cuda/adaptive_controller.py`, `multi.py _proposal_limit`, `server/runtime.py proposal_limit` | C4-C8: possibly the largest no-kernel lever (their TP4: MTP2 c8 372.8 vs MTP4 ~201-288; tiers +11.6% c8 vs fixed 4). C1: none | New. First step: measure C4/C8 at proposal_limit 0/1/2/4 live (no code) |
| 3 | **Graph/bucket coverage for every live row count** (qtrim GRAPHS +19.9% at c10; DS41 +4.5-9.6%) | `owner_execution.row_buckets` (m-profile7 empty -> powers of two), `multi_graphs.py` keys | C2-C8 at non-pow2 verify rows (C*5); C1 0 | New. Check which keys/buckets serve C2..C8 first |
| 4 | **Hidden host-sync audit per round** (27: one `.any()` cost 21-32%; 350: pageable H2D) | `multi.py`, `input_preparation.py`, `position_coordinates.py` (`issue_coordinates` pageable `torch.tensor` H2D syncs - ledger dec4), `row_sampling.py` | enables 1; alone small at C1 | Partly known (coordinate_staging trial neutral because nothing runs ahead yet) |
| 5 | **Packed pinned input staging** (350: one pinned rectangle -> one H2D into persistent buffers; reused metadata objects) | `input_preparation.py`, `position_coordinates.py`, `descriptor_upload.py` | 1-3 ms/round of the 3.1 ms verify staging + ~2.5 ms/MTP step; their +1.8-3% decode | Same family as survey item 1; concrete recipe |
| 6 | **Expert-major (distinct-expert) MoE for multi-row verify** (110, 75) | `cuda/sm70/onecat_moe_gu.cu` / `onecat_moe_down.cu` are route-major (`grid(5, routes)`); `routing.cu` | C8 (400 routes, ~279 distinct of 512): ~30% fewer MoE bytes; C4 smaller; C1 none (keep per-route below ~16 rows: friend lost 6% there) | New |
| 7 | **Dense one-weight-pass tiles for 17-40 rows + merged same-input projections** (112, 122) | `cuda/sm70/dense_small.{cu,py}` (MAX_ROWS 32, splits > 16), `dense_backends`, `dense_projections.py` | C4-C8 verify dense (our dense small GEMMs ~12 ms/round at C1, more at C8); their +3% each | New |
| 8 | **Router batch kernel band 5..40 rows** (61: +2.9% c16; slower >= 12 rows in their shape) + **top-16 select** (78) | `cuda/sm70/routing_native.py` (native logits opt-in via `TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS`), `routing.cu` | ~0.1-0.2 ms/round | Partially have |
| 9 | **Padded-row MoE masking** (19; neutral for them with dense graph sizes) | `multi.py:2301-2311` pads verify windows to depth+1 rows | measure expert-union widening first; matters only with coarse buckets | New, measure-first |
| 10 | **Shared-memory bank-conflict audit (ncu) of SM70 prefill/MoE mma kernels** (340: pad A rows, warp 32x64, CTA_K 64; bit-identical) | `cuda/sm70/gguf_ops.cu`, `onecat_moe_*.cu`, `prefill_dense.cu`, `prompt_attention.cu` | prefill +5-6% if same conflict class | New (method) |
| 11 | **Prefill MoE route/unroute and elementwise launch counts** (70: +18% prefill from one unroute kernel; 101 counting-sort route +1.2%; 140 device route sort, no host sync, +79%) | `gguf_prefill.py`, `onecat_experts.py` prefill path | count kernels in a prefill trace first | Check |
| 12 | **Per-shape split/tile tuning persisted to a table, with an engagement assertion** (73/105/400, 193) | `cuda/sm70/dense_small.py` plans, MoE plans | +1.8% c1, +3.2% c8 for them | Method |
| 13 | **GC pause measurement per TP rank** (483 stats: worker gen-2 pauses 300-460 ms) | rank processes; `gc.callbacks` stats | tail latency (lockstep TP8: one rank's pause stalls all); gc.freeze itself gave nothing for them | New, measure-first |
| 14 | **Remove O(context)/O(streams) per-round host work** (173, 183, 186, 480, 481) | state_retention, ngram context, retained_catalog victim sort, tp_scheduler | long context (262K x 24 windows) | Check at 128K vs 4K |
| 15 | **Agent prefix retention for GDN state**: tail anchors at P-L (502: re-prefilled tokens/turn 2517 -> 21), superseded-first eviction (470: returning-agent TTFT 12.1 -> 0.36 s), miss-reason log (500); tokenizer prefix reuse (310: -0.34 s TTFT at 118K, -1.1 s at 402K) | `prefixes.py`, `retained_catalog.py`, server segment cache | TTFT / prefill work on agent turns, not decode | New concepts |
| 16 | Small kernel items: residual add fused into custom all-reduce (256, +0.5-1%), row-shared weight loads in HC/router small kernels (257, 192), remove concat between producer/consumer (79, +2.8% c1), barrier protocol: epoch counters, issued-together peer loads, no reset (103 vs 100: +1% vs -3%) | `hc_fused.py`, custom gather/all-reduce | <1-3% each | Partly known (HCX = survey item 3) |
| 17 | Measurement tools: one-server A/B by live knob with graph variants captured up front (102/162), per-stage CUDA-event probe of wait/prep/forward (255, 265) | diagnostics | method | Our interleaved A/B note covers part |

Already have (no action): TC prompt attention on quantized KV (their 40: prefill +60-66%), per-route compact fused gate+up MoE
(72/74), fused SiLU*mul (28), sync-free shortlist sampler (27 design), quantized LM head (60, survey item 8), HF-exact YaRN (16),
NUMA rank binding (01/02), rotor/planar KV formats (10-13), live runtime knobs (26), decode-share prefill credit (271),
fewest-left fill (273), GreedyDual retention and LADDER anchors (261/390), recurrent-state prefix checkpoints (230),
StreamDecoder generated-ids-only (490), no per-pass `empty_cache` (321).

## 6. Rejected / failed paths (do not repeat without new evidence)

| Path | Their result | Evidence |
|---|---|---|
| TP8 for Flash-Next | 31.7 / 153.7 / 54.3 (c1/c8/c16), prefill 664/686; dropped (config not like-for-like) | R/tp8-* |
| saw4 / k4v3 KV codecs | 54.7/138.1/187.0 and 73.1/318.1/347.9 vs planar4 76.8/309.1/349.5 | R/final-* |
| rotor4 KV without a TC prefill path | prefill -37/-39% until patch 40 | R/final-planar4-tiers |
| Fused rotor rotations (104) | c1 -3%, c32 -1.1% | R/seq47.log |
| HC prefill split-K bmm (76) | 258 vs 228 us at M=3088 | R/p76-bench.log |
| Generic HCX batch kernel with heavy grid barrier (100) | c32 -3.1% | R/seq42c.log |
| Distinct-expert dedup at small M (75 without MIN_M) | c1 -6%, c8 -4% | R/seq42c.log |
| Per-route compact grid at M=32 (72) | 421 vs 281 us | R/seq42b.log |
| Router batch kernel above its band (61 at M>=12; DS41 Triton router 121) | slower; DS41 c1 -8% | R/p6061-bench.log, seqds5 |
| Depth-0 host trim alone (51) | c1 -12% | R/seq28.log |
| Async depth with drafts in flight (22 + MTP) | unsafe (optimistic computed-token counts); needed 52 (depth 0 only) | README 22/52 |
| MTP in production at >= 16 users | 413 vs 456 (c16), 665 vs 696 (c32); depth-0 start NaN bug | seq21, seq28/29 |
| Three batches in flight (137) | -30% (each re-reads weights) | seqds6c |
| Async scheduling on PP2 (266) | c8 -12%, c32 -7% (c1 +10%) | seqasched-t1c |
| Cold experts in pinned host RAM via PCIe (460) | decode c8 -13 to -32%, c32 -16 to -43% | seqcoldx-1c/2b |
| Side-stream transcode/GEMM overlap (320) | no gain (x0.97-0.99) | seqdsprefill-1 arm O |
| Launch fusion inside graphs (263, -83 launches/step) | -1.05% | seqopt screen |
| GC freeze after capture (483/510/330) | no gain to -3.2% | seqcycle2rs-2 C, seqqcycle2 |
| Pinning hot CPU threads to dedicated cores | c8 -12.3% (aborted) | hostclk-1 |
| Performance governor with C-states off | c32 -3.8%, prefill down | R/seq31.log |
| NCCL_PROTO LL128/Simple, NCCL_MAX_CTAS/BUFFSIZE (DS41), numa_balancing=0, EngineCore on node 1 | noise or worse | seq39/41/48, seqdsutil-1 |
| Larger prefill chunks (6208) on Qwen; chunk caps below the Mamba/GDN state block | pf32K -3.3%; caps silently ineffective (align block 3088) | seqqtrim-2, seqqcap |
| Sync host saves of KV/state inside the step (231 rev 3) | c4 x 128K -11% (async rev 4 -2.2%) | seqram4-t1 |
| Large fused attention with 82 KB smem, 1 CTA/SM (138) | prefill -13% (26 KB / 3 CTAs/SM version +23%) | seqds5 / seqdsnext1b |
| Expert cache on disk (23) | load 335 s -> 52 s but 77 GB; owner later wanted it off | seq22 (for us: user decision, ledger) |

## 7. Items needing a user decision

1. **GPU power limit**: ours 150 W vs their 225 W (300 W at the Oct 31 Qwen gate). Raising it is a system power change.
   Without it, compute-bound comparisons (prefill, C8 verify) against their numbers are confounded.
2. **CPU governor** (`powersave` by user decision): their data shows prefill +1-3% and neutral decode, but their host was off
   the critical path; ours is host-bound, so an A/B would be informative.
3. **Layout**: PP2 x TP4 (or TP4-in-quad) instead of TP8/ETP4 (ledger item 13). Their evidence favours TP4 groups inside
   NVLink quads; the change is architectural.
4. **Speculation default at concurrency**: lowering/turning off MTP above some stream count changes per-request latency
   characteristics; measuring is free (live `proposal_limit`), adopting a policy is a product decision.
5. Restart-durable host prefix tier / holder processes / on-disk prepared caches (231, 160, 23): durability and disk
   contracts (ledger already lists disk caches of prepared data as a user decision).

## 8. Methodology lessons worth copying

- "Flag on" is not "feature ran": 73 never ran for a week ("no active runtime owner"), 105 exported 0 records on stage 0,
  111 never engaged (head-count mismatch), 106's log line never appeared. Gate on engagement log-line counts per arm.
- Micro-bench wins often vanish integrated (73 stock-grid 1.3-1.7x, 76, 104, 240 1.06x -> +0.95%, 263). Only integrated A/B
  decides. Use realistic (zipf) routing in MoE micro-benches: uniform routing hid duplicate-expert streaming (75).
- Never trim measured categories per arm: decode 1/8/16/32 + prefill 8K/32K (+128K, long-context c4 x 128K for DS41), PPL bar,
  tasks, needles to 500K, greedy equivalence; regressions hide in loads a patch "should not" touch.
- Noise: no-MTP c1 <0.3%; MTP c1 +-7-10% across runs; restart-to-restart greedy identity only 4-7/8 (c8) for the same config,
  so cross-restart output diffs are not evidence of a numerics change; same-server controls and kernel-level bit checks are.
- Profile GPU busy with CUDA events/profiler, not nvidia-smi utilization (counts NCCL spin); torch profiler inflates host time
  2-3x (they used a CUDA-event probe).
- Band-limit every specialized kernel to the row range where it measured faster; expose bands as live knobs.
- Owner overrides are not evidence: 491 (failed exactness), 410 (two NOT POSITIVE windows), 273/274 (negative), 277+321r4
  (failed worst-gap gate), 191 (NO-GO) are in their production anyway.

## 9. Catalogue (per-patch detail)

The appendices below are the per-scope survey notes, unabridged. Each has a per-patch table (mechanism, model scope,
measured effect with evidence path, status, TensorFold relevance), per-item notes, rejected paths and open questions. Where an
appendix's own "top adoptable" ranking differs from section 5, section 5 is the consolidated ranking.

- Appendix A: patches 01-28 (NUMA, PP host path, MTP tiers, rotor KV, sampler) + Qwen measurement history and like-for-like table.
- Appendix B: patches 40-106 (rotor prefill, MoE decode compact/dedup/tune, router, HCX, LUT persistence) + Qwen production
  timeline after Oct 31 + q* overlays (qcap, qovl, qpfx, qtrim, qtune, qshare, qcycle2, qsched).
- Appendix C: DS41 patches 80-90, 110-197, 220-276 + multi-user architecture + concurrency trajectory.
- Appendix D: DS41 patches 277, 300-503 (cycle 2) + verification of the production numbers + production configuration.

Patch numbers that do not exist: 08, 09, 107-109. Category index (patch numbers):
- Host path / scheduler: 17, 18, 20, 21, 22, 26, 27, 52, 135, 137, 173, 183, 185, 186, 187, 262, 266, 269, 271, 273-275, 277, 300-302, 350, 351, 480, 481.
- CUDA graphs: 07, 102, qtrim GRAPHS, seqdsgraph-1 capture grid, 197, 263.
- Collectives / NVLink: 03, 20, 21, 79, 100, 103, 135, 194, 195, 196, 256, NCCL env arms.
- MoE kernels: 19, 28, 70, 72-75, 77, 78, 86, 88, 110, 120, 140, 193, 240, 320, 340, 450, 460.
- Dense / router / head kernels: 60, 61, 76, 112, 121-123, 184, 192, 257.
- Attention / KV: 10-15, 40, 71, 104, 111, 138, 170-172, 180, 181, 190, 191.
- KV tiers / prefix cache: 90, 230, 231, 260, 261, 390, 410, 470, 500-503.
- Speculation: 04-07, 50-52, DS41 spec analysis.
- NUMA / host system: 01, 02, 83, 160, 272, seq31 governor, hostclk-1, seqdsutil-1.
- Startup / caches: 23, 24, 25, 105, 161, 400.
- API / vision / tokenization: 16, 220, 270, 276, 310, 482, 490-492.


## Appendix A: friend 1Cat patches 01-28 + Qwen measurement history (read-only survey, 2026-11-14)

Sources (all under /home/kklouzal/build-cache/friend-1cat/): snap/lp-snap-*/README.md rows 22-47, snap/.../patches/0*-28*.py
(+ pp_files/, sampler_files/, moe_files/local_silu_mul.py, cache_files/, live_files/), top/runtime*.json, top/TODO.md s.1,
top/TONIGHT.md, top/STATUS.md, bench/results/{seq19..seq29.log, ab-*, c-*, n-202122*, h-*, t-*, u-*, d-276061*, m-*, e-*,
tp8-*, tp4-*, pp-*, final-*, g-*, cand*, p23-findings.md}. Numbers below were re-extracted from the result JSON summaries
(aggregate_decode_tps, 3 runs each; workload: 512-token prompt, 1024 out, ignore_eos, T=0.6/top_p .95/top_k 20, fixed seeds).
Note: 08/09 do not exist. Patch 05's module (local_spec_depth.py) is embedded in the patch script.

### How they restructured 1Cat (what this scope reveals)
- Process model: one vLLM engine, PP2 x TP4 (stage 0 = GPUs 0-3 = 24 layers on NUMA node 0, stage 1 = GPUs 4-7 = 24 layers
  + lm_head/sampler/MTP drafter on node 1; `--numa-bind-nodes [0,0,0,0,1,1,1,1]`, needs 01/02 + cap SYS_NICE + numactl).
  Layer split 24/24 kept (25/23 arm t-2728-p2523: c32 851.4 vs 834.8 but c8/c16 352.2/568.9 vs 366.7/581.3 -> rejected;
  22/26 arm pp2tp4-hcx-split22-26 worse on all).
- Async scheduling, Model Runner V2, batch queue of pp_size (+1 with 22) micro-batches; tokens returned stage1->stage0
  on GPU via an NCCL slot ring (event-ordered), so the host does not sit between stages.
- Why PP2xTP4 and not DP2 (2 x TP4) / TP4 / TP8 (VERIFIED, bench/results/*-kv.txt + seq21/seq26):
  KV pool PP 8.85M tokens vs DP2 1.73M per engine (3.46M total; 2.56x more for PP, one 512K-ctx pool);
  throughput PP = 99% / 97% / 98% of DP2 at 1 / 16 / 32 users with the same fixes (u-276061 62.3/605.5/870.3 vs
  d-276061 63.1 / 8+8=621.4 / 16+16=885.7). TP8 dropped (see table). C1 of PP is serial across stages
  (p23: c1 8.65 ms/step per stage, ~4 ms of non-NCCL kernels each, rest = waiting on the other stage).
- Production (top/runtime.production.json, Nov 7): no speculative-config (MTP off), kv rotor4_planar, max-num-seqs 32,
  max-num-batched-tokens 4096, 512K ctx with YaRN factor 2 (hf-overrides), cudagraph FULL sizes 1-8,10,12,14,16,
  compilation mode 3, NCCL_BUFFSIZE 4 MiB, NCCL_MAX_CTAS 8, OMP_NUM_THREADS 1, MALLOC_ARENA_MAX 2.
  Patch env set: 03, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28 (+ 40, 60-79, 100-105 outside this scope).
  NOT set: 04/05/07 (MTP off), 14 (saw4), 19. Always-on/implicit: 01/02 (numa-bind), 06, 10-13 (rotor4_planar), 15, 16 (yarn).
- Host: performance governor (seq31) + nvidia-power-limit.service 300 W (STATUS.md). Ours: powersave by USER DECISION.

### Patch table (01-28)
| # name | purpose / mechanism | scope | measured effect (EVIDENCE) | status | TensorFold relevance |
|---|---|---|---|---|---|
| 01 numa_wrapper | install missing vllm/utils/numa_wrapper.sh so `--numa-bind` workers exec | both models | functional only | prod (numa-bind on) | already-have: src/tensorfold/cuda/rank_placement.py (strict NUMA rank bootstrap, libnuma) |
| 02 numa_nested_spawn | EngineCore re-exec loop fix (100% CPU) + serialize configure_subprocess global swap across spawning threads | both | functional; no numa on/off speed A/B for Qwen found (seqdsnuma-1 DS41 arm failed to start) | prod | already-have (rank_placement.py); lesson: bind rank->GPU-local node + membind (they later hit node-0 OOM, patch 272) |
| 03 pp_hcx_stage_end | last layer of non-final PP stage keeps normal reduction (HCX packed MoE payload would be consumed on next stage); plain hc_combine at stage end | Qwen PP | correctness (assert in _hc_combine) | prod | N/A unless we activate PP (families/qwen4_exp/pipeline_stage.py is a non-serving scaffold); note for any PP port: HC/HCX fusion must not cross a stage boundary |
| 04 pp_mtp | final-stage MTP drafter shares target embed_tokens + lm_head (was uninitialised table under PP -> garbage drafts) | Qwen PP+MTP | correctness | used only in MTP arms | N/A (no PP); lesson: assert draft-head weight sharing at startup |
| 05 spec_depth_tiers | scheduler picks ONE draft depth per step from running count, table `4:2,3:5,2:11,0`, hysteresis H (enter heavier at once, lighter only at bound-H); grant <= drafts produced last step; drafter stops after k steps (depth 0 still runs draft step 0 to keep draft KV); verify graphs captured per tier q_len | Qwen MTP | VERIFIED: pp-mtp4 (fixed 4) c1 89.0 c8 310.6 vs pp-mtp-tiers c1 82.5 c8 346.5 (+11.6% c8); fixed depth 2 cliffs at c12 (pp-mtp2 c10 332.6, c12 217.1, c16 260.6 vs no-MTP pp-nomtp-x 272.7/312.5/397.2) | tested; MTP not in prod | PARTIAL: we have live `proposal_limit` (server/runtime.py SETTINGS, adaptive_controller.py) + per-stream confidence stop (decode.py:371), no automatic active-count tier table w/ hysteresis. Adoptable (see Top list #1) |
| 06 pp_spec_replay_skip | idx_mapping=-1 rows in PP sampled ring crashed SpecReplayState.record_accepted index_select; kernel skips negative rows | Qwen PP+MTP | correctness (pp-mtp-tiers-06 c16 366.3 now runs) | always on | N/A; lesson: masked rows (-1) must be skipped by every consumer |
| 07 spec_depth0_fastpath | at depth 0: no MTP forward/gather/draft writes (except draft step 0 on prompt steps), PP packet width 1, q_len=1 graphs get QSA short variants | Qwen MTP | VERIFIED neutral/no-op: pp-tiers-fast-qsashort 328.9/368.0 vs pp-tiers-fast 339.7/369.7 (c8/c16); TODO: "07's QSA short graphs are a no-op (rotor4 forces dense short off)"; depth-0 KV-refresh skip: pp-tiers-tuned-norefresh 88.0/330.5/368.6 vs tuned 86.2/324.2/362.2 (+2%) | MTP arms only | lesson: a "spec server at depth 0" still cost 7-10% vs no-spec until 50-52 (out of scope) -> our depth-0/low-depth rounds must not pay draft upkeep |
| 10 rotor4_module | 4-bit rotated KV codec (rotation + Lloyd-Max codebook), Triton store + split-K sparse reader | Qwen | inert alone | prod (via dtype) | already-have formats: families/qwen4_exp/kv_formats.py rotorquant-planar4/iso4/... (we run int8 on SC117) |
| 11 rotor4_dtypes | registers rotor4/8/3 + k4v3 dtypes | Qwen | - | prod | already-have |
| 12 rotor4_qsa | QSA K,V = FP16 norm + 4-bit codes of R x/‖x‖ (130 B/vec at D=256 vs 512); all reads via Triton split-K in rotated domain (Q rotated once, out rotated back once); XQA/page4/dense-short/host-KV/DCP routes off | Qwen | VERIFIED (FP16 vs planar4, MTP tiers, 10-29): fp16 83.0/329.5/373.4, p 4638/5763 vs planar4 76.8/309.1/349.5, p 2904/3527 (c1 -7.5%, prefill -37/-39%); KV pool 2.33M -> 7.26M tokens (3.1x); PPL fp16 3.5099 vs rotor 3.5217 (n-202122/t-2728), bar 3.5393; prefill gap later closed by patch 40 (5471/6681 > fp16 4773/6153, seq25) | prod | already-have codec family; lesson: rotor KV only pays when KV capacity binds; decode reader had no TC path (prefill fixed by FP16 scratch dequant -> page4 kernel, patch 40) |
| 13 rotor4_hybrid_page | sizes hybrid attention/GDN page from rotor spec (260 B/token/layer/head) so block holds ~3.9x tokens and GDN state page still matches | Qwen | capacity (block 3168 tokens, seq25 log) | prod | already-have analog (our capacity planner); N/A to vLLM page geometry |
| 14 saw4 | saw-int4/BDR codecs (block-Hadamard K + per-token asym INT4) on rotor page | Qwen | VERIFIED BAD: final-saw4k-tiers 54.7/138.1/187.0, prefill 493/595 | rejected | skip |
| 15 kv_zeroer_bounds | startup bounds check + clamp of KV block zeroer segments | both | safety | default on | N/A (vLLM-specific) |
| 16 yarn_mrope | HF-exact `_compute_yarn_parameters` for partial-rotary interleaved MRoPE (upstream used 4x video table length for the ramp: cos off by 1.4 at pos 200000; ignored attention_factor/mscale); MTP draft uses same table | Qwen 512K | correctness; yarn-tiers 76.8/302.0/341.6 p 3164/3850 ~= non-yarn | prod | already-have: families/qwen4_exp/rope.py follows Transformers `_compute_yarn_parameters` incl. attention_factor/mscale |
| 17 pp_no_idle_sleep | skip engine-core 1 ms `time.sleep` after a batch-queue step that blocked on a real micro-batch (1 ms on every token at 1 user) | PP (batch queue) | ab-17 vs ab-b0 (MTP tiers): c1 77.3 vs 73.8 (+4.7%, inside the ±7% run spread 68.6-78.8), c8 308.5 vs 306.8, c16 340.4 vs 341.6 -> README-claim, data consistent but not significant | prod | checked: no sleep on our round path (control_pipes uses blocking TCPStore.wait; server/app.py chunk get is blocking with timeout; expert_parallel.py sleep(.001)/(.01) polls are the EP-only mode, TP path not used). Lesson: grep for poll-sleeps on the critical path |
| 18 pp_balance | cap decodes per step at ceil(running decodes / pp_size) so micro-batches stay balanced (otherwise co-eligible requests share one micro-batch forever, other stage idles) | PP | VERIFIED: ab-1718 vs ab-17: c8 353.2 vs 308.5 (+14.5%), c16 405.9 vs 340.4 (+19%); live toggle one server (seq22): c16 on 400.7 / off 305.4 / on 396.7 | prod | N/A without PP; principle applies to any two-group overlap scheme we build (balance groups by count) |
| 19 moe_pad_mask | padded graph rows (>= real count, from 1-elem device int buffer) get row 0's expert ids with weight 0 after select_experts (Triton, no sync) so padding does not widen the expert union | Qwen | VERIFIED neutral: ab-171819 vs ab-1718 c1 74.0 vs 82.3 (noise), c8 353.6 vs 353.2, c16 404.0 vs 405.9 | not in prod | CANDIDATE for us: multi.py:2301-2311 pads every multi-stream verify window to depth+1 rows with repeated last token; padded rows go through MoE routing with their own (position-shifted) hidden states. At C8 (8 x 5 = 40 rows vs ~18 real) union widening may cost MoE GU/down time. Their graph sizes were dense (1..8,10,12..16) so padding was small; ours can be ~2x. Measure expert-union size first |
| 20 pp_static_xfer | per FULL uniform-decode step: no pickled gloo metadata (2 blocking CPU sends/step) and no 1/TP slice+all-gather; both stages derive padded rows from the identical scheduler output; rank r sends whole tensor to rank r directly into next stage's persistent graph input buffer; deferred recv resolved after dispatch | PP (both models; DS41 version 135) | VERIFIED (MTP tiers, same image -local9, seq20): c-20 vs c-base c1 82.4 vs 79.0, c8 355.7 vs 345.7, c16 424.8 vs 391.5 (+8.5%) - measured WITH the debug CHECK header (one D2H sync/step) on; c1 greedy 8/8 identical | prod | Already known class (ledger: rank-0 control plane 2.6 ms/round, shm ring item 1). Transferable technique: replicate deterministic per-round decisions on every rank from one small fixed-size packet instead of pickled/store metadata; receive straight into captured-graph input buffers |
| 21 pp_token_return | one int64 [N, W+1] NCCL broadcast (counts packed in last column) instead of two (tokens, then int32 counts) | PP | VERIFIED neutral: c-2021 vs c-20 80.5/361.0/426.0 vs 82.4/355.7/424.8 | prod | ledger-known class (merge small collectives); expect small |
| 22 pp_async_depth | batch queue pp_size+1; after queuing an empty micro-batch, schedule the next step at once; stage 0 consumes the token from the GPU slot ring behind a CUDA event, so last-stage D2H -> update -> schedule -> RPC -> stage-0 prep leaves the per-token path. Only PP>1, async sched, V2 runner, NO spec decode (optimistic num_computed_tokens breaks with drafts in flight) | PP no-MTP | README/TODO claim "+18% at 1 user"; files: n-202122 (no-MTP 17/18/20/21/22) 59.5/262.2/445.2, p 3292/4176; its seq20 control n-base was OVERWRITTEN by a later n-base (10-30 22:55) -> no same-session baseline in files. Older no-MTP planar4 runs (p300-planar4-nomtp 47.2/235.8/375.1, pp-rotor-planar 46.8/236.0/370.5) are consistent with +20-26% c1, but different sessions/images | prod | ALREADY KNOWN (ledger item 1: async N+1 prep / device token handoff). Friend data point: worth ~+18-26% C1 when host round trip (~2-3 ms) is exposed; it required depth-0-only gating with spec (patch 52 out of scope) — same obstacle as our rejected pipelined MTP chain |
| 23 expert_cache | after full load each rank writes finalized expert banks per (model key, ppP-tpT); next start reads each bank as one file, validated by GGUF identity hash + config/torch/transcode hashes + bit-exact transcode of first expert | Qwen (DS41 later 161) | VERIFIED seq22: model load 418 s cold / 161 s warm page cache -> 335 s (write, 77 GB disk) -> 52 s (hit, 9.6 GiB/rank in 4.2 s); healthy 599/341 -> 193 s | prod (owner later wanted it off: TODO s.2) | constraint-ruled-out for us (ledger: disk caches of prepared data = user decision) |
| 24 aot_save_fix | log AOT save errors with traceback, remove partial .tmp, resolve Triton kernels by alias | Qwen | functional | prod | N/A (torch.compile AOT) |
| 25 compile_hash_runtime | drop runtime-only VLLM_LOCAL_* flags from compile cache hash (each A/B arm recompiled ~77 s); test fails when a new flag is unclassified | both | seq22: torch.compile 24.4 s first -> 2.3 s | prod | lesson only: our extension/graph cache keys should exclude non-graph flags (ledger startup: extension builds are already flag-hash named) |
| 26 live_knobs | engine re-reads JSON control file between steps (1 stat/interval); scheduler + worker knobs (sent in SchedulerOutput); rejects unsafe sets (tiers only if graphs captured); writes .ack | both | VERIFIED working (seq22 live balance on/off/on: 400.7/305.4/396.7); used for one-server A/B | prod | already-have: /v1/runtime (server/runtime.py: prefill_tokens, proposal_limit, row_budget, session_ceiling). Matches our ledger "interleave two paths in one server" method |
| 27 sampler_shortlist | sync-free sampler: when all rows top_k<=64, min_p 0, no logprobs: torch.topk(128) + one Triton program/row (temperature on 128 values, stable cutoff incl. ties via in-kernel row scan, top-p, Gumbel argmax w/ upstream noise). Removes `if reference_rows.any()` cudaStreamSynchronize (27 syncs/25 steps, median 16.4 ms in p23-pp-c32 rank 4) that stopped stage-1 host run-ahead (GPU busy 73% vs 99%) | both | VERIFIED seq24: t-27 vs t-base 62.3/362.3/576.2/831.6 vs 60.9/274.5/454.0/689.2 (+2/+32/+27/+21%); reproduced seq26 u-27 vs u-base 62.2/362.5/576.8/831.3 vs 60.8/273.5/453.0/694.4; GPU sampler check 0 mismatches in ~45k trials (seq25); prefill unchanged | prod | ALREADY-HAVE design (row_sampling.py: per-row topk(CAND) + logsumexp packets, no host sync). Possible micro: pack() loops rows with ~5 launches/row (copy, topk, add/index_select, copy, logsumexp) - a one-launch-per-batch Triton version could save launches at C8+ (only if not graph-captured; unmeasured). Generalized lesson = ANY hidden host sync (`.any()`, `.item()`, pageable H2D: ledger issue_coordinates) kills run-ahead and exposes host prep |
| 28 moe_silu_mul | one Triton launch for silu(gate)*up with identical rounding points (bit-exact) for Q4_K experts outside dp4a path | Qwen | VERIFIED small: t-2728 vs t-27 63.3/366.7/581.3/834.8 vs 62.3/362.3/576.2/831.6 (+0.4-1.6%), bit-exact R=1..400 (seq24); PPL unchanged 3.5217 | prod | already-have: fused activation in cuda/sm70/onecat_moe_gu.cu |

### Like-for-like Qwen3.8-Flash-Next table (decode = aggregate tok/s; prefill = tok/s at 8164/32764-token prompts)
All friend rows: UD-Q4_K_XL GGUF, 8x V100-SXM2 32 GB (same server model). Image tag in parentheses.
| config (results label) | layout | spec | KV | c1 | c8 | c16 | c32 | p8K | p32K | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| tp4-baseline | TP4 (4 GPUs) | MTP4 | fp16 | 99.1 | 201.0 | 124.3 | - | 713 | 714 | expert_storage "original": best C1 ever measured, batching/prefill collapse |
| tp4-canonical | TP4 | MTP (depth not recorded in result; likely 4) | fp16 | 93.8 | 288.4 | 326.6 | - | 3452 | 3524 | canonical expert storage |
| tp4-canonical-mtp3 / -mtp2 / -nomtp | TP4 | 3 / 2 / none | fp16 | 88.2 / 82.5 / 64.9 | 344.9 / 372.8 / 283.7 | 318.3 / 324.5 / 440.3 | - | - | - | MTP: +45% c1, -26% c16 |
| tp8-nomtp | TP8 | none | fp8_e4m3 | 31.7 | 153.7 | 54.3 | - | 664 | 686 | NOT like-for-like: compile mode 0, FULL_DECODE_ONLY, original storage, arena; seq19 TP8 retest cancelled ("TP8 dropped"); TP8+MTP impossible with stock draft (Q8_0 640/8=80 not /32) |
| pp2tp4-devkv-nomtp-nohcx / -hcx | PP2xTP4 | none | fp16 | 46.6 / 49.2 | 244.5 / 247.2 | 397.1 / 394.5 | - | 4823 / 4822 | 6249 / 6233 | early PP (pre 17-22); PP prefill 1.4-1.8x TP4 |
| final-fp16-nomtp | PP2xTP4 | none | fp16 | 48.5 | 256.3 | 396.6 | - | 4773 | 6153 | PPL 3.5099 |
| final-fp16-tiers | PP2xTP4 | tiers 4:2,3:5,2:11,0 | fp16 | 83.0 | 329.5 | 373.4 | - | 4638 | 5763 | |
| final-planar4-tiers | PP2xTP4 | tiers | rotor4_planar | 76.8 | 309.1 | 349.5 | - | 2904 | 3527 | KV 7.26M tok vs 2.33M |
| final-k4v3-tiers / final-saw4k-tiers | PP2xTP4 | tiers | k4v3 / saw4_k | 73.1 / 54.7 | 318.1 / 138.1 | 347.9 / 187.0 | - | 2694 / 493 | 3410 / 595 | rejected |
| ab-b0 -> ab-17 -> ab-1718 -> ab-171819 | PP2xTP4 | tiers | planar4 | 73.8 / 77.3 / 82.3 / 74.0 | 306.8 / 308.5 / 353.2 / 353.6 | 341.6 / 340.4 / 405.9 / 404.0 | - | ~3163 | ~3853 | patches 17, +18, +19 (-local7) |
| c-base -> c-20 -> c-2021 | PP2xTP4 | tiers | planar4 | 79.0 / 82.4 / 80.5 | 345.7 / 355.7 / 361.0 | 391.5 / 424.8 / 426.0 | - | 3121-3138 | ~3827 | seq20 (-local9) |
| n-202122 | PP2xTP4 | none | planar4 | 59.5 | 262.2 | 445.2 | - | 3292 | 4176 | 17/18/20/21/22, mns 16 |
| h-pp-nomtp (TODO table) | PP2xTP4 | none | planar4 | 61.0 | - | 456.5 | 695.9 | - | - | seq21 (-local9), mns 32 |
| h-pp-tiers (TODO table "81.5") | PP2xTP4 | tiers | planar4 | **81.5** | - | 413.2 | 665.1 | - | - | seq21 |
| h-dp2-nomtp | 2 x TP4 engines | none | planar4 | 61.5 | 8+8: 503.7 | 16+16: 767.5 | - | - | - | DP2 "c16" = 8 users/engine; KV 1.73M/engine |
| h-dp2-tiers | 2 x TP4 | tiers | planar4 | 85.0 | 8+8: 639.2 | 16+16: 753.4 | - | - | - | |
| t-base / t-27 / t-2728 | PP2xTP4 | none | planar4 | 60.9 / 62.3 / 63.3 | 274.5 / 362.3 / 366.7 | 454.0 / 576.2 / 581.3 | 689.2 / 831.6 / 834.8 | ~3296 | ~4190 | seq24 (-local10) |
| u-276061 / d-276061 (DP2) | PP vs DP2, same fixes 27/60/61 | none | planar4 | 62.3 / 63.1 | 362.6 / - | 605.5 / 8+8: 621.4 | 870.3 / 16+16: 885.7 | - | - | PP = 99/97/98% of DP2 |
| cand (seq27 adopt, -local13) | PP2xTP4 | none | planar4 | 63.2 | 366.4 | 611.4 | 876.6 (prod-cap32, mns 32) | 5429 | 6722 | +27/28/40/60/61 |
| m-tiers (TODO "88.5") | PP2xTP4 | tiers | planar4 | **88.5** | - | 524.7 | 757.3 | - | - | MTP + 27/28/40/60/61 |
| m-d0 (TODO "91.5") | PP2xTP4 | tiers + 50/51/52 (+22 at depth 0) | planar4 | **91.5** | - | 600.9 | 843.3 | - | - | ramp c10..16 363-659; candidate only (NaN bug) |
| m-52 / m-51 | PP2xTP4 | tiers + 52 / + 51 | planar4 | 87.1 / 77.9 | - | 601.3 / 514.7 | 841.6 / 756.0 | - | - | |
| cand6 = Oct 31 gate | PP2xTP4 | none | planar4 | 82.1 | 460.9 | 721.4 | 981.7 | 6853 | 8262 | + patches 70-79, 100-105 (other scopes) |
| q-final (11-01) | PP2xTP4 | none | planar4 | 82.0 | 460.8 | 721.5 | 982.5 | 6483 | 7800 | later Qwen state; prefill lower than cand6 (not in scope) |
| **TensorFold SC117** | TP8 non-expert / ETP4 experts | MTP4 conf 0.7 | int8 | ~48.5-53 (501/601/602/611 prompt sets) | ~114-123 | not measured | not measured | 4K ~1770 | 16K ~2520 | IQ3_S GGUF; different prompt set/lengths; no c16/c32 numbers on our side |
Takeaways: (a) their best C1 configs are speculative (TP4 MTP4 99.1 original storage; PP tiers 81.5 -> 91.5 with depth-0
work); their production C1 82.1 is no-MTP after kernel patches 60-79/100-105. (b) MTP crossover on PP: wins c1 (+34%),
roughly ties c8 (346.5 vs 262.2 before patch 27; no post-27 MTP c8 measured), loses c16 (413.2 vs 456.5) and c32
(665.1 vs 695.9); even with 50-52 it stays slightly behind no-MTP at c16/c32 (600.9/843.3 vs 611.4/876.6).
(c) Our C8 (~117) is ~3-4x below their c8 (no-MTP 362-461, MTP tiers 346) — the gap at C8 is not explained by speculation
policy; it is our per-round cost (host ~16 ms + verify 25 ms) vs their ~22 ms/step at 16 rows.

### Per-item notes (performance-relevant)
- p23-findings.md (VERIFIED profile, -local9 no-MTP): c32 step 27.55 ms; stage 0 99.1% busy (18.45 ms kernels), stage 1
  73.2% busy -> ~7 ms/step idle = stage-1 HOST starving its GPU; gaps: graph end->eager 22.9 ms/25 steps, broadcast->
  pageable HtoD 19.5, QSA metadata ~12, mamba align 7.2, item+sync before gumbel 3.1. After 27+28: c32 stage 1 94% busy,
  20.7 ms/step; after 60/61 stages balanced 18.3/18.2 ms kernels, 19.95 ms/step. "nvidia-smi 95%+ util counts NCCL spin;
  real busy from the profiler." TP4 single engine c16: 77.6% busy, 46.75 ms/step.
- Their per-token cost at c1 no-MTP: ~16.4 ms/token for 48 layers on PP (2 x ~4 ms kernels + NCCL/handoff/wait) -> for
  comparison our C1 round ~46 ms carries ~2.3 tokens.
- Speculation tiers table detail (TODO s.1): verify graphs captured for {q_len: max_reqs} = {1:16, 3:16, 4:8, 5:2};
  depth chosen from TOTAL running count though each PP micro-batch holds half; median MTP acceptance ~2.5 (README/TODO
  claim). Owner idea (not built): per-request adaptive depth from recent acceptance under a per-step verify-token budget.
  TensorFold already has the building blocks (engine/family_depth.py `_draft_budgets` allocates by per-depth acceptance
  vs shared costs for some families; qwen4_exp cuda uses fixed depth + confidence stop + manual proposal_limit).
- MTP depth-0 NaN bug (seq28/29, VERIFIED in logs): server STARTED with TIERS=0 (only q_len-1 graphs) gives NaN logprobs
  (HTTP 400) on 2nd c1 greedy+logprobs request; started with tiers and switched live to 0 is clean. Lesson: start-time
  sizing paths keyed on max depth need their own coverage.
- CPU policy seq31 (VERIFIED g-*): performance governor: decode identical (63.1/367.6/611.4/874.7 vs 63.1/367.3/609.8/
  875.2), prefill +2.9%/+1.3% (5560/6726 vs 5403/6643); performance + C1E/C6 off WORSE (c32 842.2, p 5401/6512).
  Their decode is insensitive because host work is off the GPU critical path (run-ahead); ours is host-bound (~16 ms/round),
  so a governor A/B could matter more for us -> needs USER DECISION (governor is powersave by user decision).

### Top adoptable for TensorFold (ranked by expected effect on OUR bottlenecks)
1. Hidden host-sync audit of the round path (from 27; VERIFIED +27-32% c8/c16 for them, mechanism = sync kills run-ahead).
   Map: families/qwen4_exp/cuda/multi.py round loop, coordinate/round-table staging (ledger: issue_coordinates pageable
   torch.tensor H2D syncs the stream), row_sampling.py. Use CUDA_LAUNCH_BLOCKING-free nsys/torch-profiler search for
   cudaStreamSynchronize/cudaMemcpy (pageable) per round. Partly already known (coordinate_staging trial was neutral
   because no run-ahead exists yet) -> value materializes together with item 2.
2. Device-side token handoff + schedule one round ahead (22; claim +18-26% c1 when the host round trip is exposed).
   ALREADY KNOWN (ledger item 1). Friend confirms the blocker we hit: it is only safe/profitable when no drafts are in
   flight (they gated it to depth 0, patch 52). Our analog: overlap round N+1 host prep with round N verify GPU time.
3. Load-dependent speculation depth with hysteresis (05) + captured graphs per tier width. Map: adaptive_controller.py
   (`proposal_limit` effective value; server/runtime.py already exposes it live) + multi.py `_proposal_limit()`;
   tier from active stream count, enter-heavier immediately / lighter after bound-H. Expected: no effect at C1; gains at
   C16/C32 where verifying 1+k rows/stream is compute-bound (their data: MTP -10% c16, -4.5% c32 vs none; tiers +11.6% c8
   vs fixed depth 4; fixed depth 2 cliffs at c12 when rows exceed captured graphs). At our C8 (host-bound) uncertain.
4. Padded-row MoE masking (19) for multi-stream padded verify windows (multi.py:2301-2311). Friend: neutral on dense graph
   sizes; ours pads to depth+1 per stream (up to ~2x rows at C8). Measure expert-union widening (rows 40 vs ~18) before
   building; cost if positive: one tiny kernel after routing, weight 0 for pad rows (outputs unchanged since pad rows
   are never kept - must keep GDN/attention state writes of pad rows tentative as today).
5. Replicated deterministic per-round decisions + fixed-size packet (20) instead of rank-0 store metadata: ledger-known
   control-plane item (2.6 ms/round follower wait). Friend: +8.5% c16, +4% c1 even with a debug D2H check enabled.
6. Strategic (user/architecture decision, ledger-known item 13 TP4-in-quad): PP2xTP4 gave them 2.56x KV vs DP2 at 97-99%
   of DP2 throughput, and every collective spans 4 GPUs (one NVLink quad) not 8. Our pipeline_stage.py is a scaffold only.
   Expected effect on our NCCL small all-gathers (6.4 ms/round at TP8) is large but it trades C1 latency (serial stages:
   their PP no-MTP c1 61-63 before kernel work vs TP4 64.9) — not a quick adoption.
7. CPU performance governor: friend +1-3% prefill, decode neutral (their host off-path); ours host-bound -> worth an A/B
   only with USER approval.

### Rejected / failed paths (with numbers)
- TP8 (tp8-nomtp): 31.7 / 153.7 / 54.3 (c16 collapse), prefill 664/686; non-like-for-like config; retest cancelled; dropped.
  TP8+MTP impossible with stock draft (Q8_0 expert block 32 vs 640/8=80).
- saw4_k KV: 54.7/138.1/187.0, prefill 493/595 -> rejected. rotor k4v3: 73.1/318.1/347.9, p 2694/3410 vs planar4 76.8/309.1/
  349.5, 2904/3527 -> not adopted. rotor4 planar adopted for capacity despite -7.5% c1 / -38% prefill until patch 40.
- Patch 19 neutral (not in prod); patch 21 neutral (kept); patch 07 QSA-short graphs no-op; patch 17 within noise.
- MTP in production: loses c16/c32 (413.2 vs 456.5; 665.1 vs 695.9); with 50-52 still behind at c16/c32 (600.9/843.3 vs
  611.4/876.6) and NaN bug at TIERS=0 start -> deferred (owner decision 2026-10-30).
- Fixed MTP depth 2: c12 217.1 / c16 260.6 vs no-MTP 312.5 / 397.2 (graph-size cliff). Fixed depth 4 c8 310.6 vs tiers 346.5.
- Layer partition 25/23 (mixed: c32 +2%, c8 -4%, c16 -2%) and 22/26 (worse) -> keep 24/24.
- Performance governor + C-states off: c32 842 vs 875, prefill down.
- DP2 instead of PP: +1-3% throughput but 2.56x less KV and no single 512K pool -> PP chosen.
- Patch 22 with speculation: unsafe (optimistic num_computed_tokens, grants near max_tokens) -> needed 52, depth-0 only.
- Expert cache (23): 77 GB disk; owner later wanted it removed from production default.
- TP4 "original" expert storage: best c1 (99.1) but c16 124.3, prefill 713 -> canonical storage chosen.

### Methodology lessons
- Noise floors (VERIFIED from run arrays): MTP c1 spread ±7-10% across 3 runs (ab-b0: 68.6/74.1/78.8; ab-1718 77.8-89.1);
  no-MTP c1 spread <0.3% (n-base 70.62/70.77/70.7); c8-c32 ~±1-2%. Small c1 MTP deltas (<8%) were never significant.
- Output equivalence: same server re-run 8/8 identical; restart same config greedy c1 7/8, c8 4/8; sampled c8 1/8 ->
  cross-restart output diffs are not evidence of a numerics change; use same-server controls + kernel-level bit checks
  (patch 28 torch.equal R=1..400; patch 27 0/45k mismatches) instead.
- Gate per adoption: decode 1/8/16/32 + prefill 8K/32K, PPL bar <= 3.5393 (fp16 3.5099), tasks 8/4, needles to 500K,
  smoke 3/3, rollback config kept; one-server live-knob A/Bs for scheduler/host knobs (26).
- Profile-led: GPU busy from profiler not nvidia-smi; idle-gap attribution between kernels located the host sync (27).
- Keep a result file per arm with unique labels: the seq20 n-base control was overwritten by a later n-base run.

### Open questions
- Patch 22 "+18% c1" has no same-session baseline file (seq20 n-base overwritten); magnitude unverified.
- tp4-canonical result's MTP depth not recorded (runtime.tp4-canonical-mtp4.json suggests 4).
- No post-27 MTP-tiers c8 number exists; MTP vs no-MTP at c8 after the sampler fix is unknown.
- Our SC117 has no C16/C32 measurements; like-for-like comparison above stops at C8.
- Whether our padded multi-stream verify rows widen the routed expert union materially (needs a routing count probe).
- NUMA bind on/off was never A/B'd for Qwen (only functional); its value for us (rank_placement.py already binds) unknown.


## Appendix B: friend 1Cat Qwen patches 40, 50-52, 60-61, 70-79, 100-106 + Qwen q* overlays (read-only survey 2026-11-14)

Paths below are relative to /home/kklouzal/build-cache/friend-1cat/ (SNAP = snap/lp-snap-c1es8-.../, R = bench/results/).
Model scope of every patch in this file: Qwen3.8-Flash-Next UD-Q4_K_XL, PP2 x TP4, NO MTP (except 50-52), KV rotor4_planar,
max-num-seqs 32, FULL CUDA graphs. Bench: decode tok/s aggregate at 1/8/16/32 users (512 in, 512/1024 out), prefill 8K/32K.
Noise floor seen in their own A/A arms: c1 +-0.5..2 tok/s (x-base 80.4 vs x-71r2 78.2 same code), c32 +-3, prefill +-0.5-1%.
Greedy c1 identical across RESTARTS of the same config only 0-7/8 (TurboMind re-measures split-K each start) -> patch 105.
107-109: do not exist (README has no rows; patch dir stops at 106).

### 1. Patch table

| # | name | mechanism (short) | measured (EVIDENCE) | status (runtime.production.json) | TensorFold relevance |
|---|---|---|---|---|---|
| 40 | rotor4_prefill_page4 | per layer, dequant the request's rotor4 KV once into FP16 scratch (rotated domain), rotate Q once, run stock FP16 mma884 page4 attention, rotate output back; replaces Triton split-K reader (Triton 3.x tl.dot on SM70 = FMA, each row re-reads 2051 tokens) | VERIFIED R/p40-bench.log: per QSA layer call 2048 rows 35.7->4.76 ms (7.5x), 8192 rows 129.7->11.5 ms (11.3x), within 1.03-1.09x of FP16 KV; serving R/seq25.log l11-off->l11-p40 prefill 8K 3293->5471 (+66%), 32K 4187->6681 (+60%), decode same, PPL 3.5273 | PRODUCTION (MAX_TOKENS 524288) | ALREADY-HAVE: cuda/sm70/prompt_attention.cu = TC mma884 on Q8 codes decoded in-register. Lesson: never let SM70 Triton dot (FMA) carry prefill attention |
| 50 | spec_m1_parity | under MTP, target M=1 decode uses no-spec kernels (HC down GEMV, fused PLE short-conv) | part of m-d0 arm only | not in prod (MTP off) | n/a (our verify is never M=1) |
| 51 | spec_depth0_host_trim | skip SpecReplayState.prepare / record_accepted / GDN common metadata (.item()) at depth 0 when provably no device change | VERIFIED R/seq28.log: m-51 alone 77.9/514.7/756.0 vs m-tiers 88.5/524.7/757.3 (c1 -12%: NEGATIVE alone) | not prod | lesson only |
| 52 | pp_async_depth_spec0 | PP batch queue depth pp_size+1 with MTP while every queued step is depth 0 (exact optimistic num_computed_tokens) | R/seq28.log: m-52 87.1/601.3/841.6 vs m-tiers 88.5/524.7/757.3 (c16 +14.6%, c32 +11%); m-d0 (50+51+52) 91.5/600.9/843.3 | not prod: owner kept no-MTP (TODO.md s.1); depth-0 NaN-logprob bug at start with TIERS=0 (TONIGHT 12:20-12:35) | model-agnostic idea: keep pipeline (async) depth full when speculation degenerates; maps to our MTP/adaptive_controller.py + engine round pipelining |
| 60 | lmhead_q8_canonical | Q8_0 lm_head (62080x2560/rank) -> canonical TurboMind FP16xint8 GEMM for M 2..32 instead of llama.cpp MMQ | VERIFIED R/p6061-bench.log: M=8 573.8->238.6 us, M=16 1146.7->244.7, M=32 2290->308; serving R/seq26.log u-27 -> u-2760 c32 831.3->871.0 (+4.8%), c16 576.8->588.2 | PRODUCTION | ALREADY KNOWN (LEDGER 1Cat survey #8 quantized LM head). For us: head over verify rows (C*5) -> M 5..40 band matters |
| 61 | router_batch_decode | router [512,2560] FP16 for decode M in band -> SM70 batch router kernel (4 ordered FP32 K640 partitions) instead of cuBLAS wmma 32 one-warp CTAs | R/p6061-bench.log: 9.9->6.3 us M=5, 10.0->7.4 M=8, SLOWER at M>=12 (10.3 vs 9.6 at 16) -> band 5,16 in prod; serving u-2760 -> u-276061 c16 588.2->605.5 (+2.9%), c32 flat | PRODUCTION band 5,16 | CHECK: our router logits default path (cuda/sm70/routing_native.py: _NATIVE_LOGITS False, serial_logits only via TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS) -> qualify for verify rows 5..40; ~3-4 us/layer x 40 layers |
| 70 | moe_prefill_unroute | prefill MoE unroute torch chunk loop (2097 of 4428 kernels, 51 of 315 ms per 32K step) -> one Triton launch/layer, bit-identical (same reduction order) | VERIFIED R/p70-gpu.log M=3088: 2978.8->276.5 us (10.8x), 71.5->6.6 ms per stage-step; serving R/seq33.log v-base->v-70 prefill 5404->6392 (+18.3%), 6716->7874 (+17.2%), decode same | PRODUCTION | CHECK our prefill MoE route/unroute for torch-op chains at M>32 (gguf_prefill.py / onecat_experts.py); lesson: count kernels in a prefill trace |
| 71 | rotor4_decode_profile | per-row-count (splits, BLOCK_N, warps) override for rotor4 split-K decode reader | R/p71-sweep.log: rows 32 695->579 us, 16 374->314, 8 -12 us, 1/2 stock best; serving R/seq33.log v-71 c32 874.5->884.1 (+1.1%) c1-16 flat; seq47 alt profiles x-71r1..3 = wash | PRODUCTION profile 4:32:16:2,8:64:16:2,16:64:16:2,32:64:16:2 | ADOPTABLE (small): sweep our paged_attention decode split plan per rows (cuda/sm70/paged_attention*.py) - helps C>=4 only |
| 72 | moe_decode_compact | small-M decode MoE: instead of TurboMind grouped GEMM over 512 experts (10,260 CTAs, 200 used) gather the selected experts' StridedPtr records -> grid of M*top_k single-row slots, same operators | VERIFIED R/seq42b.log (zipf): per MoE layer M1 167.9->75.8 us (2.22x), M4 237.6->118.8, M8 1.34x, M16 1.02x, M32 slower (421 vs 281); serving R/seq34.log w-base->w-7072 c1 63.1->70.7 (+12%), c8 367.1->423.0 (+15%) (with 70) | PRODUCTION band 1,16 | ALREADY-HAVE: our onecat_moe_gu.cu grid (5, routes) and onecat_moe_down are per-route compact kernels |
| 73 | moe_grouped_tune | at warmup time every TurboMind grouped-GEMM descriptor timed (heuristic vs registered kernels x split-K) under CUDA-graph replay, kept if >=3% faster after confirm; LUT per rank persisted | bench R/seq42b.log per stage-step M1 3.17->2.25 ms, M8 6.19->3.98 (stock grid). BUT in every server Oct 31-Nov 6 it logged "tuning skipped (no active runtime owner)" (qtune readme-row-400): seq42c "u-73 no gain" = it never ran. Fixed by 400 (below): real gain c1 +1.8%, c8 +3.2% | PRODUCTION (works only since 400, Nov 7) | ADOPTABLE as method: offline/once per-shape sweep of dense_small / MoE split plans with persisted table (cuda/sm70/dense_small.py plans). Lesson: VERIFY the tuner engaged (log line count), not just the flag |
| 74 | moe_compact_gateup | gate+up of compact grid in one grouped launch over 2R slots (2x CTAs, 1 launch less/layer) | R/seq42b.log M8: 72 174.6 -> 72+74 141.6 us; 75 133.9 -> 75+74 107.0 | PRODUCTION | ALREADY-HAVE (our moe_gu does gate+up in one kernel) |
| 75 | moe_compact_dedup | compact grid with ONE slot per distinct expert (stock expert-sorted route + 1 Triton launch compacting offsets/records; grid = distinct experts) | R/seq42b.log zipf M16: stock 233.2, 72 244.0, 75 182.8, 75+74 156.9 us; M8 distinct 53/80. Serving R/seq42c.log u-737475 c32 899.4->966.5 (+7.4%) but c1 78.7->74.0, c8 453->435 (dedup costs at small M) -> DEDUP_MIN_M=16: R/seq45 cand4 79.0/453.3/717.9/973.0 | PRODUCTION (MIN_M 16) | ADOPTABLE for C>=4: our verify rows = C*5 (C4=20, C8=40 rows -> 200/400 routes, heavy expert reuse); per-route slots re-stream the same expert weights. Map: cuda/sm70/onecat_moe_gu.cu/onecat_moe_down.cu + routing.cu (route compaction). C1 (5 rows): keep per-route (their data: dedup loses below M=16) |
| 76 | hc_prefill_splitk | HC down prefill as strided-batched split-K bmm | R/p76-bench.log M=3088 mm 228 us vs S4 258 (SLOWER), M=1884 132 vs 155 | rejected ("not recommended") | rejected |
| 77 | moe_compact_gather | Triton gather of 6 x 16-B records + token row (replaces index_select 9.6 us 1-CTA + repeat_interleave 4.9 us per layer) | R/seq42b.log M8 72 174.6 -> 72+77 157.2; 72+74+77 125.7 | PRODUCTION | n/a (we have no per-layer index_select staging); generic lesson: hunt 1-CTA torch index ops in graphs |
| 78 | router_top16 | decode top-k: tl.topk of 16 instead of bitonic sort of all 512 keys | R/p78-bench.log 5.0->3.7 us (w8); serving R/seq41.log t-78 71.1/424.6/650.4/900.7 vs t-base 70.7/422.2/648.0/898.9 (+0.2-0.6%) | PRODUCTION | low: our routing.cu select512 (128 thr/row); measure us/call vs ~3.7 |
| 79 | hcx_moe_pair | hand (routed, shared) MoE outputs to HCX consumer as a tuple: drops a [2M,2560] cat copy (2.9 us + gap x24/stage) | R/seq44.log v-79 80.6/462.1/727.7/906.1 vs v-base 78.4/454.9/717.5/901.8 (c1 +2.8%); R/seq46 cand5 80.3/459.5/725.5/977.2 | PRODUCTION | generic: remove concat/copy kernels between producer/consumer (our HC fused path hc_fused.py) |
| 100 | hcx_batch_above | rows > N use the fork's older generic hcx_batch_kernel (heavier sync: fence+atomic+reset, serial peer polls) | R/seq42c.log u-100 c32 899.7->872.0 (-3.1%); u-1000 (all calls) 69.2/409.6/632.3/871.4 worse everywhere | rejected | lesson: grid-barrier protocol cost dominates at small M |
| 101 | moe_route_count | prefill expert route: radix sort+searchsorted+gather+argsort (302 sort kernels 2.57 ms + gather 6.9 ms per 32K step) -> 3 Triton launches (histogram, scan, scatter+gather), exact | R/seq42.log M=3088 617.5->445.4 us; serving R/seq42c u-101 prefill 6652->6735 (+1.2%), 8033->8127 | PRODUCTION | CHECK our prefill route (see 70) |
| 102 | live_graph_knobs | capture N graph sets (one per knob variant) at start, swap the runner's graph dict live on a knob change; per-call knob reads = one int compare | infra; R/seq47.log x-71live "24 captured" then 3 live profile arms in one start | infra (LIVE_KNOBS on, no variants in prod) | METHOD adoptable: one-start A/B of kernel choices inside graphs (complements our interleaved A/B note in LEDGER) |
| 103 | hcx_ls_batch | C++ local-schedule BATCH HCX kernel: all 8-row groups per phase, exchanges paid once per launch, epoch barriers (2/launch, no reset), 3 peer all-reduce loads issued together, up-weight prefetch after combine, warp per output row (122 regs, 44 KB smem) | R/seq47.log x-103 (rows>8) 80.2/460.8/725.8/980.7; x-1030 (all rows) 81.9/462.2/721.4/983.8 vs x-base 80.4/460.0/725.7/978.3; prefill 6805->6850/8189->8270 | PRODUCTION (ABOVE=0 = every call) | ALREADY KNOWN (LEDGER #3 HCX); detail worth copying: pay cross-GPU exchange once per launch for all row groups, epoch-counter barriers (no reset) |
| 104 | rotor4_fused_rot | KV rotations fused into store; split merge fused with output rotation+round+gate (3 of 4 rotor launches/QSA layer) | R/seq47.log x-104 78.0/464.2/717.2/967.7 vs base 80.4/460.0/725.7/978.3 (c1 -3%, c32 -1.1%), PPL 3.52826 vs 3.52798 | rejected ("slower") | n/a (int8 KV); lesson: fusion that lengthens the critical-path kernel can lose at c1 |
| 105 | tm_lut_persist | export TurboMind dispatch cache after capture, import next start (restart determinism of split-K) | R/seq47.log x-105b greedy vs x-105a 7/8 identical; speed same (80.1/460.9/725.6/976.5). PP0 exported 0 records until 400 (Nov 7: 25/39 records) | PRODUCTION | ALREADY-HAVE in spirit: our dense_small plans are static per shape (no runtime measurement) - confirm no runtime autotune feeds numerics |
| 106 | pp_prefill_split | cap a prompt's prefill chunk at max(MIN, ceil(N/K)) so short prompts span >=K chunks that pipeline across PP stages | R/seq48.log q-pf8 6862/8264, q-pf16 6853/8261 vs q-base 6848/8241 = NEUTRAL; the "q-pf8 106:" log line is EMPTY (patch's start-up log never seen -> engagement unproven) | not prod (shared w/ DS141) | relevant only if we pipeline stages; we are TP8, no PP |

Other config-only arms in scope: NCCL_MAX_CTAS 6->8 prefill +1.5% (R/seq39 n-ctas 6745/8150 vs 6643/8030; R/seq41 repeat +1.3%) PRODUCTION; NCCL_PROTO LL128/Simple worse/neutral (seq39 n-proto, seq41 t-nccl2, seq48 q-ar/q-arag); capture size 32 neutral (seq48 q-graphs).
Host: seq31 (R/seq31.log) performance governor: decode same, prefill 5403->5560 (+2.9%)/6643->6726 (+1.2%); performance + C1E/C6 off WORSE (c32 875->842, prefill down). They adopted performance governor with C-states ON. [For us: USER DECISION - powersave by user choice; flag only.]
GPU power: STATUS (Oct 31) says nvidia-power-limit.service 300 W; later scripts note "GPUs at 230 W" (top/seqshare2-t1.sh, seqqcap-2.sh). Nov Qwen prefill is ~5% below Oct 31 (pf32K ~7800 vs 8262) at identical decode -> probably the power cap (hypothesis, not verified).

### 2. Qwen production timeline (what it became after Oct 31)

Qwen has NOT been serving since ~Nov 1: DeepSeek V4.1 holds all 8 GPUs; every Qwen "adoption" after that is pointer-only
(top/adoptqtune.sh header: "Pointer-only: Qwen is not running"), measured in short test windows that stop DeepSeek.

| when | image / change | numbers (decode 1/8/16/32; prefill 8K/32K) | evidence |
|---|---|---|---|
| Oct 30 11:09 | -local13: 27/28/60/61(5,10)/40 | 63.2/366.4/611.4/-; 5429/6722 | R/seq27.log |
| Oct 30 22:41 | -local15: +70, 72 band 1,8, 71 profile, 61 band 5,16 | 70.7/422.3/647.8/898.2; 6625/8009 | R/seq38, seq37 x-all8 |
| Oct 31 02:43 | -local17 cand3: +73(inert!),74,77,101,78, NCCL_MAX_CTAS 8 | 78.8/454.3/718.4/901.4; 6839/8246 | R/seq43-gate.json |
| Oct 31 03:44 | -local18 cand4: +75 dedup band 1,16 MIN_M 16 | 79.0/453.3/717.9/973.0; 6826/8242 | R/seq45-gate.json |
| Oct 31 03:58 | cand5: +79 | 80.3/459.5/725.5/977.2; 6857/8277 | R/seq46-gate.json |
| Oct 31 09:45 | -local19-hcx cand6: +103 (ABOVE=0), +105 | 82.1/460.9/721.4/981.7; 6853/8262; PPL 3.5280 | R/seq49-gate.json |
| Nov 3 | -local19-hcx-img (qimgfix): image trim 270 (keep 32/max 64, token frac 0.6), downscale, MM cache guard, mm-processor-cache 4 GB | speed x1.00: 82.0/461.0/723.0/995.6; 6500/7796; 120-turn screenshot session 0 errors, hit 0.94 | R/seqqimgfix-1.log |
| Nov 6 | -qshare: DS41 scheduler patches 141/185/187/269/271 ported; live file prefill_share 0.5, depth 2 | c8/c32 x0.996; mixed16 (16 decoders + one 128K prompt): decode/user during prefill 3.26->22.69 tok/s (x7), 128K TTFT 19.0->49.4 s (x2.6) -> its own gate said NEGATIVE (TTFT>1.5x); owner adopted anyway | R/seqqshare-1nr.log, R/adoptqshare.log, R/seqqlive-1.log |
| Nov 7 19:46 | -qshare-qtune1 (CURRENT production.image): patch 400 q_tune_owner (73+105 run under borrowed runtime owner) + dense capture sizes [1..8,10,12,14,16] + new LUT dirs | qtune arm: 83.2-83.7 / 474.6 / - / 985-987 (B: 82.0/459.8/990); pf32K ~7780 (flat); qtrim GRAPHS arm c10 451.8->540.5 (+19.9%), others same | R/seqqtune-1.log, R/seqqtrim-1.log, top/adoptqtune.sh |
| Nov 10 | production re-measured (qcycle2 B1) | c1 83.09, c2 160.96, c4 280.46, c8 474.58, c10 555.32, c32 987.42; pf32K 7804; mixed16 during 22.9 tok/s/user, 128K TTFT 51.4 s | R/seqqcycle2-1.log |
| Nov 11 | qsched (DS 273/274/275-as-331/277/310/490 on Qwen) | BUILT + CPU-tested only; GPU window seqqsched-1 written, dry-run, NOT queued | overlays/qsched-overlay/readme-rows-qsched.md, R/qsched/ |

Current runtime.production.json env (Qwen): BALANCED_PREFILL_CHUNKS, NCCL_BUFFSIZE 4M, NCCL_MAX_CTAS 8, PP_HCX_STAGE_END, PP_STATIC_XFER,
PP_TOKEN_RETURN, PP_ASYNC_DEPTH, EXPERT_CACHE, PP_NO_IDLE_SLEEP, PP_BALANCE, SAMPLER_SHORTLIST, MOE_SILU_MUL, LMHEAD_Q8_CANONICAL(60),
ROUTER_BATCH band 5,16 (61), ROTOR4_PREFILL_PAGE4 (40), MOE_UNROUTE_LARGE (70), MOE_COMPACT band 1,16 (72), ROTOR4_DECODE_PROFILE (71),
MOE_TUNE (73), COMPACT_GATHER (77), GATEUP_MERGE (74), ROUTE_COUNT (101), ROUTER_TOP16 (78), COMPACT_DEDUP MIN_M 16 (75), HCX_MOE_PAIR (79),
HCX_LS_BATCH_ABOVE 0 (103), TM_LUT_DIR (105), IMAGE_TRIM*/DOWNSCALE/MM_CACHE_GUARD (270), DS41_DECODE_CAP_FILE (qshare 271 share 0.5),
LOCAL_Q_TUNE_OWNER/DIR (400). Args: TP4 x PP2, numa-bind nodes [0,0,0,0,1,1,1,1], max-num-batched-tokens 4096, mixed-prefill-min 2048,
prefix caching + mamba align + tail checkpoint, FULL graphs sizes [1,2,3,4,5,6,7,8,10,12,14,16], hcx local schedule, hc_ll_shard.
NOT set: 50/51/52 (MTP off), 76, 100, 102 variants, 104, 106.

### 3. q* overlays (Qwen re-ports, Nov 1-11)

| overlay | content | result | status |
|---|---|---|---|
| qcap (seqqcap, -1, -2) | DS 185 decode cap / 187 decode-first / 200-202 decode-paced sub-block prefill on Qwen | seqqcap: caps 2048/1024/512 never took effect (mamba align block = 3088 tokens; cap never cut below a block) -> 1.09-1.12x = noise, NO-GO; df (decode-first) TTFT 1.70x. qcap-2 (202 split each block into k pieces while others decode): k2 mid-window decode/user 1.09->2.22 (2.04x), TTFT 1.03x, GO (R/qcap2-final-gate.json, speed 81.98/460.8/721.5/982.5) | GO but NOT in production env (no DECODE_PACE vars); superseded by qshare |
| qovl (264/265) | 264 fused Mamba/GDN block-id kernel (7 eager ops -> 1, 95->44.4 us micro); 265 CUDA-event chain probe | T1: +1.82% c1 vs threshold 2.00% (noise 0.39%), greedy c8 worse than off/off control -> NO-GO (R/qovl1/t1-verdict.txt) | rejected |
| qpfx1 | partial prefix cache, model-name fallback, decode pace file | agent bench warm TTFT 3.45->2.64 s but warm decode/user x0.92; image-session arm 95/120 errors -> NO-GO (R/seqqpfx-1.log) | rejected |
| imgtrim/qimgfix | 270 image_trim (oldest images -> text placeholder in batches of MAX-KEEP so prefix stays byte-stable; oversize downscale) | GO, speed x1.00 | PRODUCTION |
| qshare | ports of DS 141/185/187/269/271 (prefill time share) | see timeline; x7 decode during long prefill, x2.6 TTFT | PRODUCTION (share 0.5) |
| qtrim | 330 GC freeze (V2 runner), 331 lookup skip, 332 step timer, + arms EAGERTOK, NCCL (drop BUFFSIZE/MAX_CTAS), GRAPHS, CHUNK (6208) | GRAPHS c10 +19.9% (ADOPTED via qtune); GC/EAGERTOK/NCCL same; CHUNK 6208: pf32K x0.967 worse, turn1 x0.90 better, mixed16 x0.96 worse (R/seqqtrim-2.log) | GRAPHS adopted, rest rejected |
| qtune | 400: run 73/105 inside set_current_vllm_config + activate_runtime_resources (site-packages .pth hook, vllm tree byte-identical) | 18 descriptors tuned (5 faster, 75.7 us/call summed) in 4.7 s; c1 +1.8%, c8 +3.2%, c32 flat; PPL identical | PRODUCTION |
| qcycle2 | DS cycle-2 (480-483, 490-492, 500/501) applicability; new 510 (GC freeze on V2 capture_model), 511 tool-call stats | 490 detok tail prime exact, -51 ms warm first text at 130K (within spread: no gain); 510 FRZ c8 x0.968 NEGATIVE; 511: streamed tool call gap 20 s before a 3049-char chunk (parser holds args) | nothing adopted |
| qsched | DS model-agnostic scheduler set on Qwen | not GPU-run | pending |

### 4. Per-item notes (performance-relevant)

- Their C1 decode critical path (R/qovl1/t1-off1-probe-c1.txt, CUDA events, c1, 82 tok/s): cycle 12.18 ms/token = stage0 graph 5.64 ms +
  stage0 eager pre-graph chain 0.53 ms + stage1 graph 5.37 ms + transfers/sample (~0.6). Stage-1 "wait" 0.003 ms: HOST IS FULLY OFF THE
  CRITICAL PATH (host 6-50 ms ahead). At c16 micro-batches: graph 14.5 ms, cycle 15.5 ms (stage ~94% busy). Whole 48-layer no-MTP token
  ~11 ms of GPU time on 4 GPUs/stage. Compare ours: ~46 ms round / ~2.3 tok = ~20 ms/token with ~16 ms host-serial -> their structure
  (graphs replayed back-to-back, all metadata device-resident, one token handoff) is the gap, not kernels.
- MoE small-M lesson set (72/74/75/77/73): per-route compact grid beats a 512-expert grouped grid by 2.2x at M1..4, ties at M16, loses
  at M32; dedup (one slot per distinct expert) wins at M>=16 (+7.4% c32) but costs ~6% at c1/c8 (extra route launch); so they gate
  per M (band 1,16 compact; dedup from M16; stock grouped above). Isolated benches with UNIFORM routing hid the dup-stream cost: they
  re-benched with zipf routing (seq42b). For TF verify rows C*5: C1=5 rows (stay per-route), C4=20/C8=40 rows -> dedup candidate.
- 73 never ran in production for a week (needed an active runtime owner outside forward context) and the A/B "u-73 alone: no gain"
  was misread as "tuning gives nothing". Real tuned gain +1.8% c1 / +3.2% c8. Lesson: assert engagement via log counts per arm.
- Graph bucket density (qtrim GRAPHS): with captures [1,2,4,8,16], c10 ran padded to 16 rows: 451.8 tok/s; adding 3,5,6,7,10,12,14:
  540.5 (+19.9%), nothing else changed; capture cost +6 s. TF: owner_execution.row_buckets() default = powers of two + endpoint ->
  verify rows C*5 (5,10,15,20...,40) pad up to next power of two (e.g. 40 -> 64 if 64 is the bound, 20 -> 32). Check m-profile7's
  explicit row_buckets; if pow2, add the exact C*5 multiples. Expected: C2..C8 aggregate gains up to ~10-25% at non-bucket loads; C1 0.
- 61/60: band-limited kernel selection - each specialized kernel only inside the M band where it measured faster (61 lost at M>=12).
- 103 vs 100: same batching idea, opposite results (+0.5..2% vs -3.1%) purely from barrier protocol (epoch counters, issued-together
  peer loads, no end-of-kernel reset) - relevant to our in-graph collectives / small_gather barriers.
- 104/76: fusions/split-K that the model predicted to win lost in serving (104 c1 -3%, 76 slower than cuBLAS mm at M 1-3K).
- qshare (271) is the model-agnostic multi-user scheduler lever: under PP, every decode micro-batch waits for prefill chunks; time-share
  0.5 keeps decoders at ~22.7 tok/s/user during a 128K prefill instead of ~3. Not a C1 lever; maps to our tp_scheduler.py /
  serving_runtime.py admission once we serve mixed long-prompt loads. Their qcap finding: chunk caps below the Mamba/GDN state block
  are silently ineffective (align mode) -> 202 splits a block into k equal sub-chunks ending on the boundary.
- 52 (MTP + PP async depth pp+1): +11-15% at c16/c32 for MTP; owner still chose no-MTP because no-MTP + PP async depth beat MTP
  tiers at >=16 users (TODO.md: 61.0 vs 81.5 c1 but 456.5 vs 413.2 c16, 695.9 vs 665.1 c32 on -local9) and later kernel work brought
  no-MTP c1 to 82-83. MTP depth-0 path had a NaN-logprob bug at start with TIERS=0.

### 5. Top adoptable for TensorFold (ranked by expected effect on OUR bottlenecks)

1. Dense graph row buckets for verify rows (qtrim GRAPHS, +19.9% at a padded load). Map: families/qwen4_exp/cuda/owner_execution.py
   row_buckets + m-profile7 manifest. Effect: C2-C8 throughput (non-pow2 verify rows); zero C1. Cheap; costs capture memory/time.
2. Host-off-critical-path structure evidence (12.18 ms/token with 0.003 ms stage wait): confirms LEDGER 1Cat survey #1 (already known)
   as the dominant gap; their eager pre-graph chain is only 0.53 ms/step and they still chased it (264).
3. Distinct-expert dedup for verify batches >= ~16 rows (75; +7.4% c32 for them). Map: onecat_moe_gu.cu / onecat_moe_down.cu /
   routing.cu. Effect: C4/C8 MoE GU+down (our MoE ~9.3 ms/round at C1; larger at C8); measure distinct/route ratio on real traces first.
4. Router logits batch kernel band (61) - our native serial_logits is opt-in; qualify for rows 5..40. ~0.1-0.2 ms/round.
5. Per-shape split tuning with persisted table + engagement assertion (73/105/400) for dense_small and MoE plans; ~2-3% at C1/C8 for them.
6. Decode attention split profile per row count (71): +1.1% c32 only; low priority for C1.
7. Scheduler prefill time-share / sub-block decode pacing (271/202) for mixed long-prompt multi-user serving (not C1).
8. Live graph variants (102) as a measurement method: capture alternatives once, swap per knob -> same-server A/B.
Already have / known: 40 (TC prompt attention on Q8), 72/74 (per-route compact GU with fused gate+up), 60 (LEDGER #8), 103/79 HCX (#3).

### 6. Rejected / failed paths (with numbers)
- 76 HC prefill split-K bmm: 258 vs 228 us (M=3088), 155 vs 132 (M=1884).
- 100 generic HCX batch kernel: c32 -3.1% (872.0 vs 899.7); all-rows variant worse everywhere (69.2/409.6/632.3/871.4).
- 104 fused rotor rotations: c1 78.0 vs 80.4, c32 967.7 vs 978.3.
- 75 dedup at all M: c1 74.0 vs 78.7, c8 435.5 vs 453.4 (only c32 gained) -> gated to M>=16.
- 72 at M=32: 421 vs 281 us stock (band capped at 16). 61 batch router at M>=12 slower than cuBLAS.
- 51 alone (depth-0 host trim): c1 77.9 vs 88.5.
- 71 alternative profiles (seq47 x-71r1..3): wash. Capture size 32: neutral. 106 PP prefill split: neutral (engagement unproven).
- NCCL LL128/Simple proto, allreduce/allgather simple: neutral/slightly worse. Performance governor + C-states off: c32 -3.8%.
- qovl 264 fused block-id: +1.82% < 2% threshold, greedy c8 control fail. qpfx1 partial prefix/model-name fallback: NO-GO.
- qcap caps (2048/1024/512): ineffective under Mamba align blocks; decode-first: TTFT 1.70x. 202 k>=4: TTFT 2.5-6x.
- GC freeze on Qwen V2 runner: neutral (qtrim GC) / c8 -3.2% (qcycle2 FRZ). Bigger chunks (6208): pf32K -3.3%, mixed -4%.
- 73 (inert for a week) and 105 (PP0 exported 0 records) show "flag on" != "feature engaged".

### 7. Generalizable lessons (methodology)
- Gate per adoption: 3-repeat speed >= 99% of base at EVERY load (1/8/16/32 + prefill 8K/32K) + PPL bar + tasks + needles 32K-500K +
  vision smoke + greedy-equivalence vs the previous production; automatic rollback on smoke failure. Owner rule: never trim measured
  categories per arm ("regressions hide in loads a patch shouldn't touch").
- Re-run the base arm twice (x-base/x-base2) to read noise; c1 swings ~2 tok/s across restarts.
- Use realistic (zipf) routing in MoE micro-benches; uniform routing hid duplicate-expert streaming.
- Micro-bench wins (73 stock grid 1.26-1.68x; 76; 104) frequently did not reproduce in serving; only integrated A/B decides.
- Band-limit every specialized kernel to the M range where it measured faster; expose bands as live knobs.

### 8. Open questions
- Did 106 engage at all in seq48 (empty "106:" log line)?
- Is the Nov prefill drop (~-5% vs Oct 31) the 230 W power cap? (decode unchanged).
- Their per-layer counts per decode token under TP4 (graph 5.6 ms / 24 layers ~ 0.23 ms/layer/stage) vs our verify program 25 ms for
  5 rows x 48 layers at TP8: worth a per-layer side-by-side (from LEDGER 1Cat survey data) to separate kernel vs structure gaps.
- TF: what are m-profile7's actual row_buckets for verify/MTP roles at C2..C8? (decides item 5.1's size).


## Appendix C: friend's DS41 patches 80-90, 110-199, 220-276 + MULTIUSER_DESIGN/BRINGUP (read-only survey, 2026-11-14)

Paths: M=/home/kklouzal/build-cache/friend-1cat ; README = M/snap/lp-snap-*/README.md (row `| NN `) ; patches M/snap/lp-snap-*/patches/ ;
R = M/bench/results ; prod config = M/misc/ds41/runtime.ds41.production.json (+ live file M/misc/ds41-knobs/decode_cap:
`{"cap":2048,"decode_first":0,"mixed_ms":0,"step_total":2048,"prefill_share":0.5,"prefill_depth":2,"small_first":16384,"share_boost":0.8,...,"prefill_train":4}`).
VERIFIED = a results file supports the number. TF = TensorFold.

### 1. Architecture / how they restructured 1Cat for DS41 (MULTIUSER_DESIGN.md, BRINGUP.md, prod config)

- Process model: stock vLLM V1 multiproc: API server proc + EngineCore proc (scheduler, sync scheduling, `async-scheduling false`)
  + 8 worker procs (spawn, `numa-bind-nodes [0,0,0,0,1,1,1,1]`, OMP 1, MALLOC_ARENA_MAX 2). SchedulerOutput goes to all workers
  through vLLM's shm broadcast MessageQueue; PP hand-off by NCCL isend/irecv (135). Engram tables (110.6 GB) in a mlocked memfd
  owned by a sidecar "holder" container on node 0 (160), passed by SCM_RIGHTS.
- Layout: PP2 x TP4 (`VLLM_PP_LAYER_PARTITION 20,20`; stage 0 = GPUs 0-3 node 0 incl. embedding + Engram, stage 1 = GPUs 4-7
  node 1 + head). All routed experts GPU-resident (16.3/17.7 GiB per GPU). Upstream TP4 (config A) needed host expert offload
  (~27 cold-expert misses/round); PP2 removes the offload. TP4 groups = the NVLink-fully-connected quads -> 1Cat's one-stage
  push custom all-reduce (`sm70_cross_device_reduce_1stage_push<4>`) for decode.
- Batching = vLLM continuous batching with micro-batches: batch queue depth = pp_size (2 micro-batches in flight), patch 18
  balances decodes ceil(running/2) per micro-batch, patch 17 removes the 1 ms idle sleep. Each stage processes micro-batch k
  while the host prepares k+1 -> host prep is hidden. 255 probe (R/container-logs, "LOCAL-PATCH 255 probe"): steady decode at
  4 rows/micro-batch: stage-0 cycle 10.4 ms = forward 9.6-9.7 + prep 0.67-0.79, GPU wait 0.06 ms, idle 7-8 %, while host prep
  is 2.2-2.3 ms (hidden); stage 1 wait 0.84 ms. At C1 (one micro-batch) the round trip is exposed: cycle 19.0 ms, idle 55 %.
- Per-request state model (85): position-addressed data = paged KV through vLLM block tables; finite recurrences (Engram 3-id
  history, C2 partial group) = slot tensors [S+1,...], lowest free slot (deterministic across stages), released on finish/
  preempt, per-token request map `token_req` built ON DEVICE from query_start_loc (one compiled kernel, graph padding = -1),
  commit inline after the kernel (no spec decoding). Dummy/capture batches are device no-ops. Window latents -> per-slot rings
  [S+1,128,512] (87) so vLLM builds ONE KV group (6 groups/128 KiB blocks before). Packed FP4 rows 534 B/token stage 0.
- Graphs: FULL_DECODE_ONLY (uniform decode only; mixed/prefill steps eager; FULL rejected because mixed steps carry host spans),
  capture sizes 1..16,24,32. Persistent graph buffers: token_req rebuilt per replay; req_slot via a 4-deep pinned ring so a
  non-blocking H2D never races a host rewrite; runner gets an input-prep event (137/262) so a host running ahead never
  rewrites pinned inputs whose copies are still queued.
- Batch invariance as a requirement: a row's kernel path must not depend on M (row-wise kernels to 64 rows: 86/89/110/112/121/
  257); the equivalence harness (ds41mu_equivalence.py) demands contamination-exact outputs; owner accepted near-tie (top-2
  margin <= 0.2) differences for remaining M-dependent ops. Every patch: anchored text replacement, default OFF, env flag,
  CPU/interpreter/sm_70-ptxas compile tests before GPU, then a "seq" GPU script with JUDGE/VERDICT gates.

### 2. Where C1 58.7 -> C8 357.6 -> C32 818 came from (decode 512/512, VERIFIED)

| step (date) | c1 | c8 | c16 | c32 | evidence |
|---|---|---|---|---|---|
| 85-87 base MU port (10-31) | 57.3 | 273.3 | 363.7 | 455.8 | R/seqds3.log d-base |
| +110 EM2 expert-major MoE | 60.7 | 309.4 | 448.7 | 617.4 (+35 %) | seqds3 d-110 (greedy c1/c8 8/8 identical) |
| +89 row-wise kernels to 64 rows (d-all = 89+110) | 60.8 | 299.5 | 504.9 | 669.7 | seqds3 d-all |
| +112 dense rows one weight pass | 60.9 | - | - | 697.8 | seqds6b r-112 vs r-base 677.7 |
| s-base (10-31 night: +120/122 etc.) | 61.6 | 325.2 | 514.0 | 715.6 | seqds7r |
| dsnext (183 input cache, 186, 173) | 60.0-61.7 | 319-328 | 499-531 | 726-763 | seqdsnext1/2/4 |
| +195 decode-indexer split | 59.0 | 325.3 | 532.3 | 772.1 (x1.020) | seqdsp4-7 t-final |
| +262 PP async output | 58.7 | 354.7 (x1.092) | 573.3 (x1.063) | 819.6 (x1.043) | seqppovl-t2 JUDGE x-cand |
| +240/256/257 combo (owner adopt, hostpath off) | x0.986 | x1.021 | x1.030 | x1.038 | seqopt-t2 x-cand; top/adoptopt.sh |
| dense capture-size grid 1..16,24,32 | - | - | c13/14 +4.6 % | newly covered sizes +4.5 %; off-grid +9.6 % | seqdsgraph-1, seqdstune-2 |
| production 11-10 arm A | 58.7 | 357.6 | - | 818.3 | seqcycle2rs-2 (SE 0.08 %) |

Mechanisms that make throughput scale (model-agnostic reading):
1. Forward time nearly flat in rows: 4 rows/micro-batch = 9.6 ms vs 1 row 8.5 ms per stage (255 probe). Achieved by
   expert-major MoE (each distinct expert's bytes read once per step, not once per row/piece), dense GEMV that reconstructs
   weights once for up to 32-40 rows, row-wise kernels with fast paths extended from 6/8 rows to 64.
2. Two micro-batches in flight across two stages hide all host work (prep 2.2 ms, sampling/readback) behind the other
   micro-batch's GPU time; 262 also removed the last stage's host serialization on the sampled-token D2H.
3. Every batch size has a graph (holes in the capture grid cost 4-10 %).
4. Host costs proportional to context/requests removed (173, 183, 186) - decisive at long context, ~2 % at short.
5. No speculation (DS41 spec analyzer: projected all-segment speedup c1 x1.108, c4 x0.954, c8 x0.748; SPEC ACCEPT NO-GO,
   R/seqdstune-1p.log). Qwen production also runs no MTP (C8 460.9).

Prefill 8K/32K (idle): 1422/2194 (10-31) -> 140 PF_MOE 2576/2734 (seqds7c) -> 181 PF_ATTN 3113/3607 (seqdsnext1b) -> 184
DENSE_BLAS 3357/4069 (seqdsnext2) -> 193 buckets 3605/4190 (seqdsp4-4) -> combo-1 32K 4216.8, 128K 4778.7 (ds41/adoptcombo-1-gates.txt).
Production idle pf8k ~3200 (seqdsgraph-1; share/cap knobs on).

### 3. Per-patch table

Status: P = production (env in prod config / live file), R = rejected/not adopted, S = superseded, H = hold/off, N = never active.

| # | name | mechanism (short) | scope | measured (evidence) | status | TF relevance |
|---|---|---|---|---|---|---|
| 80 | llamacpp_layout | GGUF tensor-index remap for llama.cpp DS41 layout | DS41 | functional | P | n/a (loader) |
| 81 | ds41_pp | PP2 stages own [a,b) layers; FP32 HC residual (T,4,5120)+hc_pre cross stages; stage starts at KV source layers | DS41 | enables all-GPU experts | P | agnostic: PP stage split of an HC model; TF has pipeline_transfer.py (private, not a serving profile) |
| 82 | reasoning_effort | tokenizer arg mapping | DS41 | - | P | n/a |
| 83 | engram_ram | 110.6 GB tables in mlocked memfd, mbind node, verified rows, preads | DS41 | 252-262 s load SATA (seqds3) | P | agnostic: big host table pinned+NUMA-bound; TF ram_backing/numa_memory already |
| 84 | prefill_moe | admit MMQ/grouped dense for >=64-token prefill | DS41 | superseded by 140 | S | - |
| 85 | multiuser | slot state + device token_req map + paged KV; batched paged indexer; dummy-batch no-op | DS41 | base 455.8 c32 | P | agnostic design = TF multi.py already per-stream; device-built token map idea maps to TF round tables |
| 86 | mu_moe | dp4a op in pieces of 6 rows (batch invariant) | DS41 | 1.35x distinct bytes at M=32 | S (by 110) | - |
| 87 | mu_packed | paged FP4 rows 534/356 B/token; per-slot window rings; one KV group | DS41 | 17.2M tokens KV (seqds3) | P | MLA-specific; agnostic: keep sliding state outside pager -> one group |
| 88 | mu_em | expert-major dp4a 7-64 rows v1 | DS41 | c32 562 vs 456 (seqds3 d-88) | S (by 110) | - |
| 89 | mu_invariant | row-wise norm/proj/HC/dmv13 fast paths to 64 rows | DS41 | c32 487 vs 456 alone (+7 %), with 110 669.7; greedy differs (near-tie) | P | agnostic: extend small-M fast paths to the real row band |
| 90 | mu_prefix | slot restore at prefix hit (Engram history from ids, C2 at block) | DS41 | (230) | P | TF prefixes.py already |
| 110 | mu_em2 | expert-major routed MoE 1..64 rows, bit-exact: codebooks in smem via 16B loads, SWAR sign, plan fused w/ Q8_1 quantizer, SwiGLU in GU epilogue, 4 launches/layer (was ~12), per-(type,M) schedule JSON from bench | DS41 IQ2_XXS/XS/IQ1_M | c32 +35 %, c16 +23 %, c8 +13 %, c1 +6 %, ~1.9x byte floor (seqds3) | P | ADOPTABLE idea for TF MoE at multi-row verify (see Top list) |
| 111 | mu_idx | WMMA decode indexer, 16 heads | DS41 | never engaged (8 heads/rank) | N (fixed by 172) | lesson: verify engagement logs |
| 112 | dense_rows | dmv13 with 2-4 8-row tiles per reconstructed weight fragment (9..32 rows, one weight pass, bit-identical per row) | DS41 Q5_K/Q8_0 | c32 677.7->697.8 (+3 %) (seqds6b) | P | ADOPTABLE: TF dense_small MAX_ROWS 32, split only <=16; check 17-40-row verify path |
| 120 | moe_swiglu | prefill SwiGLU 9 torch ops -> 1 Triton | DS41 prefill | pf8k 1434 vs 1418 (+1 %) (seqds5) | P | TF prefill elementwise fusion (check gguf_prefill) |
| 121 | router_batch | Triton gate projection 1..64 rows | DS41 | c1 52.7 vs 57.5 (-8 %) (seqds5 d-121) | R | lesson |
| 122 | dense_merge | merge same-input projections into one segment launch 9..40 rows; wo_a own plane slice (half bytes); head one launch | DS41 | c32 585.6 vs 567.5 (+3 %) (seqds5) | P | agnostic: merge qkv/gate-up launches at all M (TF dense_projections) |
| 123 | dense_wm | weight-major dense kernel for DS shapes 9..40 rows | DS41 | c32 569 vs 567 neutral (seqds5) | R | - |
| 129 | kv_dump | diagnostics for rotor KV study | DS41 | - | diag | - |
| 135 | pp_xfer | metadata-free PP hand-off: both stages derive T from scheduler output; 3-slot send ring, stream-wait reuse, irecv into persistent buffers; no gloo pickles (was 2x8.9 ms/step blocking) | PP | c32 679.6 vs 677.7, c8 318.8 vs 314.7 (seqds6 r-135) | P | agnostic: TF pipeline_transfer.py already NCCL+slots; lesson = never pickle per step |
| 137 | pp_depth | 3 micro-batches in flight (sync sched) | PP | crash (pinned-input race), then c8 219.5 / c32 461 vs 315/678 (-30 %) (seqds6c) | R | lesson: more micro-batches = more weight/expert reads per token |
| 138 | mu_attn | fused sparse attention v1 (WMMA, 82 KB smem, 1 CTA/SM) | DS41 prefill | pf8k 1241 vs 1420 (-13 %) | R (by 181) | lesson: occupancy |
| 140 | pf_moe | prefill MoE: device route sort (no host sync), transcode tile of 128 experts -> TurboMind grouped HMMA GEMM; ~20 launches/layer vs ~66K/step | DS41 prefill | pf8k 1440->2576 (+79 %), 32K 2217->2734 (seqds7c/7r) | P | TF prefill MoE: check host syncs / per-expert loops in gguf_prefill.py |
| 141 | pp_split | cap prompt chunk to ceil(N/K) (min 2048) so chunks overlap across stages | PP prefill | pf8k 1440->1543/1564 (+8 %) | P | PP-only |
| 150 | vision | image input path | DS41 | - | P | n/a |
| 160 | engram_holder | sidecar holds memfd across restarts (SCM_RIGHTS), child-process load, OOM-safe fallback | DS41 | restart Engram 252 s -> ~9 s attach (seqdstune-2 "Engram holder 8.8 s") | P | agnostic: keep big host tables in a holder process across restarts (TF startup) |
| 161 | expert_read | threaded whole-bank expert reads | load | cold 117 s vs 87.5 s upstream (worse) | R | lesson |
| 162 | live_knobs | HTTP knob -> collective_rpc on workers: env + module constants refreshed, graphs re-captured (~2-3 s) | tool | enables ABBA in one server | P | ADOPTABLE methodology (TF has interleaved A/B note) |
| 170 | mu_reindex | upstream hierarchical indexer (Full/Reindex/Reuse) | DS41 long | with 171-173: c4x128K 67->173, c16x200K 64->413, c1x480K 14.6->54.1 tok/s (seqdslong2) | P | model-specific (DSA) |
| 171 | mu_topk_seg | segmented top-k above 16K scores | DS41 long | part of above | P | TF compact_indexer top-k analog |
| 172 | mu_idx8 | WMMA indexer for 8 heads/rank (+candidate mode) | DS41 long | part of above | P | model-specific |
| 173 | pp_no_history | stop shipping every request's full token history in SchedulerOutput each PP step (pickle+unpickle x8 workers: 7.3 ms/128K req) | host | +12 ms GPU idle/step removed at 4x128K (seqdsprof7 per README); part of dslong2 gain | P | ADOPTABLE check: any O(context) per-round host work in TF control plane |
| 180 | pf_index | prefill indexer split by query rows over TP (all_to_all queries, no [32,N] score all-reduce) | DS41 prefill | pf128K 2334->2560 (+10 %) (seqdsnext1) | P | agnostic: split by rows instead of all-reducing big score tiles |
| 181 | pf_attn | fused sparse attn v2: raw mma.m8n8k4, keys streamed L2->fragments, 26 KB smem 3 CTA/SM | DS41 prefill | pf8k 2524->3113 (+23 %), 128K 2977->3731 (seqdsnext1b); decode-split variant c1 -4.5 % (not used) | P (prefill) | V100 lesson: HMMA.884 raw PTX, <48 KB smem for 3 CTAs |
| 183 | input_cache | InputBatch re-add without per-step list->numpy copies; prompt cached as ndarray, outputs incremental; block row cached; penalties tensor built on device once | host | c32 741.8 vs 726.1 (+2 %); local CPU 0.93->0.10 ms per re-added 32K req (seqdsnext1) | P | ADOPTABLE check: TF per-round per-stream host staging |
| 184 | dense_blas | dequant+cuBLAS HGEMM for dense prefill >= 2048 rows | DS41 prefill | pf8k 3226->3357, 32K 3668->4069 (+11 %) (seqdsnext2) | P | TF prefill dense: check large-M route |
| 185 | decode_cap | cap prefill chunk while anyone decodes (live file) | sched | DSNEXT3 GO at 1024 | P (cap 2048) | TF prompt_fill decode_share already |
| 186 | pp_nosample | skip sampling-metadata rebuild on non-last PP ranks | host | c1 60.8->61.7, c32 = (seqdsnext4 y-186 GO) | P | agnostic: only the sampling rank builds sampling state |
| 187 | pp_decode_first | <=1 in-flight micro-batch with prefill; mixed_ms budget | sched | all settings trade TTFT ~1:1 (JUDGE w-s*/v-s*) | H (file 0) | - |
| 190 | pf_chunk | Reindex prefill via candidate scorer; one host sync/step; src topk | DS41 prefill | w-rx NO-GO (pf32K -1 %) | R (env unset) | PF_SYNC idea: one host copy per step |
| 191 | pf_fused | WMMA fused prefill index scores + combine epilogue | DS41 prefill | pf128K 4078->4804 (+18 %) but JUDGE NO-GO on decode-during rule (seqdsp4-1) | P (later) | - |
| 192 | hc_exact | prefill HC transition: TOK tokens per program share fn loads (same words) | HC models | pf8k 3378->3427 (+1.5 %) (seqdsp4-2) | P | ADOPTABLE: TF HC prefill (hc_fused.py) weight reuse across tokens |
| 193 | pf_moe_buckets | tune TurboMind grouped GEMM per chunk-size bucket at startup | prefill | pf8k 3438->3605 (+5 %), 128K -1.8 % (seqdsp4-4 y-bk) | P | agnostic: per-M-bucket kernel config table |
| 194 | ar_split | >8 MiB eager all-reduce as pieces on custom AR | prefill | pf8k 3411 vs 3438 NO-GO (seqdsp4-5 y-ar) | R | lesson |
| 195 | dec_split | decode indexer rows split over TP + all_gather, no score all-reduce; min 4 rows | DS41 | c32 x1.020, c1 x0.976 before MIN (seqdsp4-6/7) | P | agnostic: below TP rows keep all-reduce |
| 196 | pp_fused | one NCCL message for hand-off | PP | c32 x0.977 (REPS=1) NO-GO | R | expected 0.4 % < noise |
| 197 | vis_graph | text steps take token-id path so embedding runs inside graph | DS41 vision | c1 x1.030 but greedy c8 0/8 [DIFF] (t-vis) | R | lesson: embedding outside graph costs ~3 % c1 |
| 220 | model_name_fallback | unknown model name -> served | API | - | P | n/a |
| 221 | engram_shard | separate Engram file registration | DS41 | - | P | n/a |
| 230 | mu_pfx | prefix cache on packed KV + window-ring snapshots at anchors (scheduler AnchorStore, worker capture/restore at bind) | DS41 + any recurrent/window state | agent warm TTFT 164.5 -> 6.06 s (x0.04), decode/user x1.65 (seqpfx-1 p-pfx) | P | TF prefixes.py/state checkpoints = already-have |
| 231 | ram_tier | /dev/shm host store of KV pages + ring snapshots surviving restarts; identity hash; crc+gen commit; async side-stream gathers, writer thread | host tier | agent-resume TTFT 140.6 -> 8.46 s (x0.06); sync saves c4x128K x0.890 (rev3) -> async rev4 x0.978 (seqram4-t1) | P | TF retained_catalog/ram_backing = in-process; restart durability = user decision |
| 240 | mu_em3 | EM3: replicated IQ2 tables per lane, window activation layout, persistent CTAs | DS41 | alone c32 +0.95 % (DMOE T1 NOT POSITIVE); micro 1.06x | P (combo) | V100 lesson: codebook LDS bank conflicts dominate |
| 255 | host_probe | CUDA-event probe of wait/prep/forward per stage; bind() numpy trim | tool/host | trim +0.21 % (HOSTPATH T1) | probe P(off), trim R | ADOPTABLE tool: per-rank idle/prep/forward line |
| 256 | moe_sum2 | shared+routed add fused into TP all-reduce push kernel in graphs; stale-slot reset | collective | +0.76 % (noise 2.1 %) screen | P (combo) | ADOPTABLE small: fuse residual add into custom AR |
| 257 | hc_dec | gate projection + HC parts share weight loads across rows (RB rows/program) | HC models | gate 19.7->12.9 us (8 rows, 4080); +1.05 % c32 screen | P (combo) | ADOPTABLE: TF HC/router at multi-row verify |
| 260 | pfx_mm | prefix hits for image requests | DS41 | - | P | n/a |
| 261 | pfx_retain | value-tiered anchor retention per conversation | sched | 8322 captures/8194 evictions before | P | TF prefixes.py GreedyDual already |
| 262 | pp_async_out | last stage: enqueue token D2H+event, output thread replies; defer top-k/p `.any()` fences, recompute exactly in thread; input-prep event | host/PP | c8 +9.2 %, c16 +6.3 %, c32 +4.3 %, c4x128K +12 % (seqppovl-t2); ASYNCOUT T1 +7.34 % noise 0.44 % | P | ADOPTABLE (see Top list) |
| 263 | dec_fuse | fuse norm/RoPE/QDQ glue: -83 launches/stage-step in graph | DS41 | SCREEN -1.05 % NEGATIVE | R | lesson: launch-count cuts inside graphs ~0 |
| 267 | ctx_1m | 1M max-model-len | DS41 | CTX1M GO, short-ctx x0.99 | P | n/a |
| 268 | long_prefill_mem | bounded transient allocations at 1M (granules, tile caps, empty_cache policy) | DS41 | MEMFIX FAIL (min free 1089 MiB < 1100) then adopted | P | agnostic: allocator fragmentation at long prefill |
| 269 | step_prefill | per-pass total prefill tokens while decoding | sched | burst7 gap max 3.96->0.94 s (STEPBUD T1) | P (2048) | TF prompt_fill: one chunk per gap already |
| 270 | image_trim | drop oldest images | API | - | P | n/a |
| 271 | prefill_share | prefill passes owe decode-only cool-down C(1-S)/S | sched | mixed8 decode/user during 128K 2.43 -> 21.9 tok/s, 128K TTFT x2.78 (SHARE2 T1) | P (0.5) | TF decode_share credit = already-have |
| 272 | ram_numa | mbind tmpfs store files to node 1 (stage 0 on node 0 was OOM-killed with Engram) | NUMA | c32 x1.00, placement only (DSNUMA) | P (stage0:1) | TF numa_memory placement; lesson below |
| 273 | small_first | small prefills first, large yield w/ ageing | sched | DSADMIT-2 GO=False (warm p50 x0.05 but cold x1.13) | P (16384, owner) | TF _next_fill fewest-left already |
| 274 | share_boost | raise share when queue long | sched | time-to-clear +35 % but decoder gap 3.8 s > 3 s limit (shareboost-1) | P (owner) | - |
| 275 | lookup_skip | no prefix lookup in held passes | sched | lookups x0.006, dec x1.25 (DSADMIT 275) | P | agnostic: don't redo cache lookup for refused requests |
| 276 | image_text | allow literal `<image>` text | API | - | P | n/a |

### 4. Notes on performance-relevant items (model-agnostic part -> TF mapping)

- 110/240 expert-major MoE. Their calc (MULTIUSER_DESIGN s4): distinct experts D(M)=E(1-(1-k/E)^M); expert-major reads D(M)
  experts once vs pieces/route-major ~1.35x at M=32. For TF Qwen (E=512, top-10): C1 verify 5 rows -> 50 routes, D=~48;
  C8 x 5 rows = 40 rows -> 400 routes, D=~279: route-major MoE reads ~1.43x the distinct bytes. If TF onecat_moe_gu/down
  (cuda/sm70/onecat_moe_gu.cu, onecat_moe_down.cu) iterate per route row, an expert-major plan (sort routes by expert on device,
  decode each expert block once per 8-16 routes, fixed FP32 order) cuts MoE bytes ~30 % at C8. Expected: C1 ~0 (MoE GU 6.3 +
  down 3.0 ms, already ~1 route/expert), C8 MoE share of round -25-30 %. Their extra V100 lessons: codebook lookups from one
  smem copy are bank-conflict bound (~6 wavefronts; per-lane replicated tables -34 %), fold scales into activation scale,
  fuse quantizer into plan and SwiGLU into GU epilogue (4 launches/layer), schedule variants per (type, M) from a bench JSON.
- 112/122/89 dense at 9-40 rows: reconstruct quantized weight fragment once and run 2-4 8-row MMA tiles on it (bytes x1 instead of
  x M/8); merge same-input projections at all M. TF: dense_small.py MAX_ROWS 32 and splits >16; C8xMTP4 verify = 40 rows ->
  likely dense_backends path; worth a microbench of 24/40-row dense with one-weight-pass tiling.
- 262 async output: mechanism = sampled-token D2H + event enqueued, a worker output thread waits/replies in RPC order, main thread
  immediately prepares/launches next micro-batch; host-fence ops in the sampler (`.any()` on top-p reference rows) deferred and
  resolved exactly in the thread. TF analog: post-verify sample+commit 1.5 ms and MTP picks readback on the rank-0 critical
  path; already-known as "one async D2H/round" (1Cat survey item 1) - friend's numbers confirm magnitude: +4-9 % at c8-c32 when
  2 micro-batches exist; c1 x0.995 (no overlap possible at one micro-batch). Without a second in-flight batch the gain at C1 is
  ~0 -> TF C1 needs the device-side chain instead.
- Two micro-batches in flight (PP + balance 18 + no idle sleep 17) is THE host-hiding mechanism; 137 (three) lost 30 % because
  each extra micro-batch re-reads weights/experts. TF TP8 single stage: the equivalent is splitting streams into 2 groups and
  alternating rounds so group B's host prep overlaps group A's GPU program. Cost: weights read twice per token-pair. At C8 TF host
  is ~16 ms of a ~46 ms round -> potential large gain, but needs measurement (their stage forward 1->4 rows +13 % only).
- 173/183/186 host O(context)/O(requests) work: history shipped every step (31 ms per 480K request), per-request list->numpy
  re-copies on every PP re-add (5.3 ms per 200K request), sampling metadata rebuilt on non-sampling ranks. Combined with
  indexer fixes: c16x200K 64 -> 413 tok/s. TF check: per-round per-stream host staging (coordinate issuance 1.0 ms, PLE ids
  0.45, stage_round 0.57) - measure at 128K vs 4K context to see if any grows with context.
- Graph capture grid: filling holes (5,6,7,9-15,24) +4.5-9.6 % at those concurrencies, 0.33 GiB graphs; TF multi_graphs.py keys
  (streams, rows): confirm every live stream count has a key (else padded/eager).
- 256 sum2: one launch + gap less per MoE layer by doing shared+routed FP16 add inside the AR push kernel (same rounding) -> TF
  custom all-reduce / small_gather could fuse the residual add. Small (+0.5-1 %).
- Collectives seen in their c32 decode profile (R/prof-ds41-ref-c32-summary.txt): custom 1-stage push AR = 12.7 % of stage-0
  kernel time (63 us mean rank 0 vs 35.7 us rank 1 -> rank skew absorbed in the AR), NCCL AllGather RING_LL 293 us x80.
  NCCL tuning: NCCL_PROTO=LL128,Simple NO-GO (seqdsnext2), NCCL_MAX_CTAS=8 + BUFFSIZE 4M no gain (seqdstune-2 "NCCL env alone").
- KV/host tiers: GPU prefix cache with recurrent/window snapshots (230) gives the agentic win (x0.04 TTFT); RAM tier (231) only
  matters across restarts/GPU eviction; synchronous GPU->host saves inside the step cost 11 % at c4x128K -> must be side-stream
  + pinned pool + writer thread, skip (never wait) when pool empty, consumer waits on an event at the next bind.
- NUMA/system (friend's host, same server model): workers numa-bound to their GPU's node; Engram (110 GB mlocked) on node 0 with
  stage 0; tmpfs RAM tier pages were allocated on node 0 by NUMA-bound writer threads and OOM-killed production
  (CONSTRAINT_MEMORY_POLICY) -> mbind the shared file mapping (inode policy) to node 1 (272). Neutral for speed. DSUTIL
  (R/ds41/seqdsutil-1.log): numa_balancing=0 x1.004 c32, EngineCore on node 1 x1.0065, no logging x1.001 -> all noise.
  hostclk-1: friend runs governor PERFORMANCE (hot CPUs ~2.9 GHz); thread "herding" (pin hot threads to dedicated CPUs, rest to
  housekeeping) c8 -12.3 % -> aborted. TF uses powersave by USER DECISION: governor effect on our 16 ms host path is a user
  decision item, not a recommendation.

### 5. Top adoptable for TensorFold (ranked by expected effect on OUR bottlenecks)

1. MTP off / depth-limited at concurrency (agnostic lesson from DS41 spec analysis + Qwen production no-MTP at 32 users).
   Their projection: drafts at c8 x0.748. TF C8 117 aggregate with MTP4 verify of 40 rows. Action: measure C4/C8 with MTP0/1/2
   (serving_profiles / speculation.py policy keyed on active stream count). Potentially the largest C8 lever; zero kernel work.
2. Host-work overlap via a second in-flight group (their PP2 two-micro-batch mechanism; 262 output thread). TF tp_scheduler.py /
   multi.py: at C>=2 alternate two stream groups so rank-0 control plane + staging of group B runs during group A's GPU program;
   move post-verify readback/sample to an output thread. Already-known as 1Cat survey item 1; friend evidence: idle 7-10 % with
   2.2 ms host prep hidden; +4-9 % c8-c32 from the output thread alone. C1 unaffected.
3. Expert-major MoE for multi-row verify (110): ~-30 % MoE bytes at C8 (D(400 routes)=~279 of 512); C1 ~0. Map:
   cuda/sm70/onecat_moe_gu.cu, onecat_moe_down.cu (+ onecat_experts.py plan). Gate: bit-exact vs route-major per row.
4. Dense one-weight-pass for 17-40 rows (112/122): map cuda/sm70/dense_small.cu/.py (MAX_ROWS 32, splits >16) and
   dense_backends; C8 verify only.
5. Graph-key coverage for every stream count (seqdsgraph-1/tune-2: +4.5-9.6 % at uncovered sizes). Map multi_graphs.py.
6. Audit O(context)/O(streams) host work per round (173/183/186). Map input_preparation.py, tp_scheduler.py, compact indexer
   span issuance. Relevant to long-context and C8.
7. Row-shared weight loads in HC/router small kernels (257, 192): TF hc_fused.py / router at R=5..40 (C1 small, C8 more).
8. Residual add fused into custom all-reduce (256, +0.5-1 %).
9. Methodology tools: live knob switch with graph re-capture in one server (162) + per-stage wait/prep/forward event probe (255).

Already-have in TF (no action): decode-share prefill credit (271), fewest-left-first fill + guard ageing (273), prefix retention
with GreedyDual value (261), state checkpoints for recurrent prefixes (230), in-process RAM retention tier (231 minus restart
durability), NUMA host allocator (272), pinned staging.

### 6. Rejected / failed paths (with numbers)

- 137 three micro-batches: c8 219.5 / c32 461.2 vs 314.7/677.7 (-30 %), first crashed on pinned-input race (seqds6/6c).
- 138 fused attention v1 (82 KB smem, 1 CTA/SM): prefill -13 % (1241 vs 1420). Replaced by 181 (26 KB, 3 CTAs/SM).
- 121 Triton router 1..64 rows: c1 -8 % (52.7 vs 57.5). 123 weight-major dense: neutral.
- 111 never engaged (head-count mismatch; "neutral" result was the fallback) - check engagement lines.
- 161 threaded expert reads: cold start 117 s vs 87.5 s.
- 181 decode split (attention over 5 CTAs/token at decode): c1 -4.5 %.
- 187 decode-first / mixed_ms, 185 caps < 1024: TTFT traded ~1:1 for decode (JUDGE w-s*, v-s*).
- 190 PF_RX: pf32K -1 % NO-GO. 194 AR split to custom kernel: -0.8 % prefill. 196 fused PP message: below noise.
- 197 graph embedding for vision: c1 +3 % but greedy diverged 0/8 -> rejected on exactness.
- 255 host trims: +0.21 % (noise 0.16 %). 263 launch fusion inside graphs (-83 launches/stage-step): -1.05 %.
- 240 EM3 alone: +0.95 % c32 (noise 1.04 %) despite 1.06x microbench.
- NCCL_PROTO=LL128,Simple; NCCL_MAX_CTAS=8/BUFFSIZE 4 MiB: no gain. numa_balancing=0, EngineCore on node 1: noise.
- CPU thread herding/pinning: c8 -12.3 % (hostclk-1, aborted).
- 231 rev 3 synchronous host saves in the step: c4x128K x0.890; rev 4 async still x0.978 (failed its 0.98 gate; adopted by owner).
- Speculation for DS41 (DSpark): SPEC ACCEPT NO-GO (1.48 acc/step on content+tool; c8 projected x0.748).

### 7. Generalizable lessons / methodology

- Gate every candidate on: decode c1/c8/c16/c32, prefill 8K/32K/128K, long-context c4x128K, KA, PPL bar <= base x1.005, needles,
  greedy c1/c8 identity, and engagement log lines present on cand / absent on base. Two-stage: cheap screen (noise-relative
  threshold 2x noise or 2 %) then full stage-2 server A/B. Noise: decode c32 run-to-run ~0.2-1 %, c8 ~0.3 %, prefill ~2-3 %.
- Base-vs-base greedy at c8 is 6/8 identical (batching placement + accepted near-tie ops): a c8 diff alone is not a bug.
- Microbench wins frequently vanish integrated (240 1.06x -> +0.95 %; 263 -1 %; 196 expected 0.4 %).
- Profile host idle with CUDA events per stage (torch profiler inflated host time 2-3x).
- Per-step host costs that scale with context or request count dominate long-context serving; look for them first.

### 8. Open questions

- Whether TF's onecat MoE kernels are already expert-major (not verified here; grep found no expert grouping).
- TF C8 round composition (host vs GPU, MTP acceptance at C8) - needed to rank items 1 vs 2.
- Exact contribution of 240/256/257 individually in production (only combo + screens measured).
- 191 PF_FUSED and 274 share_boost are in production despite NO-GO/NEGATIVE verdicts (owner overrides); adoption evidence beyond
  the verdicts not found.


## Appendix D: DS41 patches 300-503 (+277, cycle 2) and DS41 production config/number verification

Mirror root M = /home/kklouzal/build-cache/friend-1cat. Snapshot S = M/snap/lp-snap-c1es8-.../ (README.md 502 lines; rows
300-351 at lines 129-134, 320/321/340/410/450/460 at 376-381, cycle 2 480-492 at 403-409, 500/501 at 415-416, 277 at 451,
502 at 459, 321r4 at 482; 310 row in S/tokcache310r2/readme-row-310.md; 470 in S/ringkeep470/patches/ringkeep_files/
local_ds41_ring_keep.py docstring; 503 in M/overlays/sesspin-overlay/readme-row-503.md). R = M/bench/results.
Mechanisms below come from README rows + module docstrings (code spot-checked for 470/stagea.json/knobs, not line-audited).

### 1. Production numbers: VERIFIED, with caveats

| claim | file | exact source | verdict |
|---|---|---|---|
| C1 58.7, C2 115.7, C4 203.7, C8 357.6, C32 818.3 | R/seqcycle2rs-2.log lines 20-24, ROW 121-145 | arm A, 2026-11-10 13:25-13:32 UTC; 512 in/512 out, bench_speed2.py; 3 repeats c1-c4 (SE 0.08-0.37 %), 2 repeats c8/c32 (SE 0.28/0.11 %) | VERIFIED |
| C16 572.8 | R/seqdsgraph-1.log ("base-c16 ... 572.8", 2026-11-07 10:44) | base arm of a DIFFERENT, older image (4d57473d-numa-admit2-...-pfxfix4, pre-c1es), graphs [1,2,3,4,5,6,8,10,12,16,24,32], single run. Confirmed again by R/seqdsprefill-1.log "dec-c16 572.6" (same image) | VERIFIED (mixed source: not the c1es6 image of the other numbers) |
| live-shaped 154.7 (162.7 with 480+481) | seqcycle2rs-2 ROW "live-shaped decode tok/s (8 decoders + 1 prefiller, 220K ctx, 768 suffix)" A 154.7 (4 windows cv 1.3 %), B 162.7 (x1.052, cv 0.9 %); noise F/A x1.006 | B = 480+481 env 1 + live files on | VERIFIED numbers, BUT the script's own VERDICT for 480+481 = "INVALID (arm B env not as planned: LOCAL_DS41_EXPERT_STATS=None)" (an env-check artefact: 450 lines 30 were present). Independent seqcycle2ab-2 (verify mode): 480 live decode x1.031 POSITIVE, 481 x0.996 (gap MAX x1.044 better) |
| idle prefill 8K 3199->3310, 32K 3845->3969, 128K 4380->4537 | R/seqdsprefill-1.log arms B1 vs F (2026-11-07) | B1 = 320 overlap 0 + 321 rev1 "upstream" (empty_cache every pass); F = 320 overlap ON + 321 "off". Noise pf8k 0.2 %, pf32k 0.34 %, pf128k 0.32 %. Greedy+PPL identical all arms | VERIFIED numbers, but F is NOT the production setting. Closest to production (overlap 0, empty gated) is arm E: 3344/3958/4517 (x1.047/1.031/1.030). 320 alone (arm O) x0.972/0.997/0.993 = no gain. The gain is 321, not 320. |

Exact config of seqcycle2rs-2 arm A: image 4d57473d-c1es6-mmfast430-imgkeep420-numa-admit2-imgtext-share2-fix5-ram4-stepbud-
ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4 (= production 2026-11-10..11, BEFORE 277/502/321r4), production runtime of
that day + test copies of every knob file (-rs2), env READD_TAIL=2 RAM_PLAN_FAST=2 (verify mode, live files OFF), WORKER_GC=2,
EXPERT_STATS=1, RAM_RINGS 2048. GPU power limits read 225 W on all 8 GPUs (V100-SXM2 default is 300 W: their box is
power-capped - same server model as ours; check our cap before comparing). Idle sampler confirmed no foreign load.
No c1es8 (current production) decode table exists; the only c1es8-shaped measurement is seqtrain-2 arm E (own driver:
decode steady 85.3 tok/s, burst 25.7 tok/s/aggregate of its decoders) - not comparable to the 512/512 table.
Within the arm A table, 450 expert stats cost: arm E (450 off) C1 59.9 (+2.0 %), C2 +1.6 %, C4 +0.7 %, C8 +0.8 %, C32 -0.6 %;
production keeps 450 ON (so production C1 pays ~2 %).

### 2. Production DS41 configuration (c1es8, since 2026-11-11T18:22Z)

- Pointers: M/top/production.ds41.image = `4d57473d-c1es8-mmfast430-imgkeep420-numa-admit2-imgtext-share2-fix5-ram4-stepbud-
  ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4`; production.ds41.snap = lp-snap-c1es8-... (= S).
- Compose: M/top/compose.ds41prod.yaml (container ds41-main, ipc host, cap SYS_NICE for numactl --membind, memlock -1,
  ports 8000/8001, mounts runtime.ds41.json -> /app/runtime.json, snapshot -> /lp, serve.main.py entrypoint). compose.ds41.yaml
  is the same file with the old image (local-ds41-em) and lp-snap19ds: not production.
- Runtime: M/top/runtime.ds41.json == M/misc/ds41/runtime.ds41.production.json (byte-identical JSON, 9806 B).
- Model: /models/ds41-uncensored-gsq-rco/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS (4 backbone shards, old/ISTA layout via 221)
  + Engram table ...-00006-of-00006.gguf (IQ4_NL, held in RAM by a separate holder process, ~110.6 GB mlocked on node 0,
  holder.sock) + mmproj BF16 vision. dtype half, quantization/load-format gguf, tokenizer-mode deepseek_v41.
- Layout: PP2 x TP4 (pipeline-parallel-size 2, tensor-parallel-size 4, VLLM_PP_LAYER_PARTITION 20,20); stage 0 = GPUs 0-3 on
  NUMA node 0, stage 1 = GPUs 4-7 on node 1 (numa-bind, numa-bind-nodes [0,0,0,0,1,1,1,1]). No EP (experts sharded by TP).
  V1 model runner, multiproc spawn workers (8) + EngineCore + API server; synchronous scheduling (async-scheduling false,
  V41_ASYNC 0; 266 async scheduler tested NOT POSITIVE); PP balance (micro-batches alternate, VLLM_LOCAL_PP_BALANCE 1).
- Speculation: none (no speculative-config; DSpark5 integration never finished: MULTIUSER_DESIGN.md s.20).
- KV: V4.1 native compressed attention KV (FP8/128 QDQ window + indexer FP4 paths per 263's description), MU packed rows
  (VLLM_LOCAL_DS41_MU_PACKED 1, V41_PACKED_KV 0), block-size 64; GPU KV pool 13,968,704 tokens (6.95 GiB available, log).
  No --kv-cache-dtype flag. 230 GPU window-ring snapshots (PFX_SNAPS 256) + 231 host-RAM tier in /dev/shm (196,608 pages,
  RAM_RINGS 2048, stage-0 pages mbind to node 1).
- Limits: max-num-seqs 32, max-num-batched-tokens 8192, max-model-len 1,048,576, chunked prefill, prefix caching on,
  gpu-memory-utilization 0.9, enforce-eager false, cudagraph_mode FULL_DECODE_ONLY, capture sizes 1..16,24,32
  (dense list adopted after seqdsgraph-1: newly covered sizes c13/c14 +4.6 %), compilation mode 0, custom all-reduce on,
  sm70_ring off, gguf expert_storage original + expert_arena, no expert cache.
- Full env (runtime.ds41.production.json; 140 keys): VLLM_WORKER_MULTIPROC_METHOD spawn, OMP_NUM_THREADS 1,
  MALLOC_ARENA_MAX 2, TOKENIZERS_PARALLELISM false, HF/TRANSFORMERS offline; DS41 bring-up: LLAMACPP_LAYOUT, ENGRAM_RAM,
  ENGRAM_LOAD_THREADS 8, EFFORT, V41_INDEX_TILE_MB 32, PP_NO_IDLE_SLEEP, DS41_PP, PREFILL_MOE, ENGRAM_NUMA 0, ENGRAM_HOLDER,
  ENGRAM_NOCACHE, ENGRAM_SHARD; multi-user MU 1, MU_MOE, MU_INDEX paged, MU_PACKED, MU_EM2 (+schedule json), MU_INVARIANT
  (rows 64), MU_IDX, MU_REINDEX, MU_TOPK_SEG, MU_IDX8, MU_PREFIX, MU_EM3 (+@dmoe-schedule.json); kernels MOE_SWIGLU,
  DENSE_MERGE, HEAD_SEGMENTS, DENSE_ROWS, DENSE_BLAS 2048, MOE_SUM2, MOE_SLOT_RESET, HC_DEC (RB 1,1,4,4,4 TOK 1,1,2,4,4),
  HC_EXACT (TOK 2), DEC_SPLIT; PP: PP_XFER, PP_SPLIT 4, PP_SPLIT_MIN 2048, PP_NO_HISTORY, PP_NOSAMPLE, PP_ASYNC_OUT;
  prefill: PF_MOE (TILE 128, TUNE, MIN 128, BUCKETS 256..8192, EXTRA 64), PF_INDEX, PF_ATTN (MIN 64), PF_FUSED,
  LONG_PREFILL_MEM, CTX_1M; scheduler: DECODE_CAP 1024 (+_FILE decode_cap), STEP_PREFILL 1024, PREFILL_SHARE 0.5,
  LOOKUP_SKIP; prefix: PFX, PFX_RETAIN (+LOG, +FILE), PFX_SNAPS 256, PFX_MM, RAM (+FILE), RAM_RINGS 2048, RAM_MM,
  RAM_RING_FIRST, RAM_NUMA stage0:1; vision: VISION (+MMPROJ, BIAS_VL), IMAGE_TRIM (MAX 32 KEEP 16 +FILE), IMAGE_DOWNSCALE,
  MM_CACHE_GUARD, LOCAL_IMAGE_TRIM_CACHE_ENTRIES 4096 / _MB 2048, LOCAL_MM_FAST, LOCAL_DS41_IMAGE_TEXT_OK;
  misc INPUT_CACHE, LIVE_KNOBS, MODEL_NAME_FALLBACK, HOST_TRIM 0; site-packages (LOCAL_*, outside vLLM so the 231 store
  identity survives): KERN340_RULE, BIND_TRIM, VOCAB_MASK_EAGER, INPUT_FAST (+VERIFY 200, STATS 2000), PFX_LADDER
  (32768 + FILE), CACHE_OUTPUT (+FILE), EXPERT_STATS (+DIR), COLD_EXPERTS 0 (+FILE), PF_TRIM_FILE, RING_KEEP (+FILE),
  PROMPT_TOK_CACHE (+FILE), MISS_REASON (+FILE), TOOLCALL_STATS, WORKER_GC 2 (stats only), READD_TAIL 1, RAM_PLAN_FAST 1,
  MM_ONCE 1, DETOK_TAIL_PRIME 2 (file 1 = use), TOOLCALL_STREAM 1, PREFILL_TRAIN 1, TAIL_ANCHOR 1, PF_EMPTY4 1.
- Live knobs (M/misc/ds41-knobs): decode_cap {cap 2048, step_total 2048, prefill_share 0.5, prefill_depth 2, small_first 16384
  (age 120 s, gap 20 s), share_boost 0.8 (waiting 4, decoders 2, min_left 1024), share_solo_free 0, prefill_train 4};
  pf_trim {overlap 0, empty "probe", floor_mb 1024, growth_mb 256, probe_mb 256}; kern340-rule {enabled 1, verify 0, 24 rules};
  tail_anchor/ring_keep/pfx_ladder/cache_output/miss_reason/readd_tail/ram_plan_fast/mm_once {"enabled": true};
  toolcall_stream {enabled, buffered true, resync false}; detok_tail_prime 1; prompt_tok_cache 1; image_trim {max 1024, keep 16};
  cold_experts.json present but 460 off. NO session_pin.json, NO turn-anchor env (501/503 not in production).
  The cycle-2 files were flipped from stage-A OFF to ON on 2026-11-10 (*.bak-pre-enable-20261110).

### 3. Patch table (300-503, 277)

Status key: PROD = env + live knob on in c1es8; OFF = installed but off; NOT = not in image/env.

| # | name | mechanism (agnostic part) | model scope | evidence (R=bench/results) | status | TF relevance |
|---|---|---|---|---|---|---|
| 300 | ds41_step_timer | per-phase host timer at TP rank 0 of each PP stage (gap/update_states/bind/prepare_inputs/slot_mappings/attn_metadata/preprocess/launch/forward/sample) | any V1 runner | seqdsstep-1: stage-0 p50 ms gap 10.86, worker 2.58, runner 2.20, update_states 0.24, bind 0.18, prepare_inputs 0.65, attn_metadata 0.19, preprocess 0.50, launch 2.09 (host; GPU starts 0.14 ms in). seqdsfloor-2 arm T: timer+prod c8 x0.982 (cost) | NOT (measurement tool) | already-have (dec4 diag timers) |
| 301 | bind_trim | per-row Python loop -> one NumPy slice assignment; md5-guarded equivalence | DS41 MuContext | seqdsstep-1 T(301+302): c1 x1.001, c8 x1.002, c32 x1.003 (noise 0.2-0.6 %) = neutral; bind -0.054 ms | PROD | idea only (vectorize per-row host loops) |
| 302 | vocab_mask_eager | replace torch.compile'd mask on an EAGER path (25 us dynamo guard + 0.24 ms) with 6 aten ops | any eager embed | same window; -0.02 ms preprocess | PROD | check TF host path for torch.compile/dynamo guards on eager calls (probably none) |
| 310 | prompt_tok_cache r2 | API server encode reuses the token ids of the longest previously-encoded text prefix at an added-token boundary (exact by construction; 600/600 fuzz) | any (plain-text prompts) | seqdstok-2 POSITIVE: TTFT -1.12 s at 402K, -0.34 s at 118K, burst round-max -1.22 s, API CPU -0.69 s/req, 0 verify mismatch (rev1 seqdstok-1 had 1 mismatch / did not engage; seqdsengine-1 rev1 never ran) | PROD | TF has a template segment cache (cuda/server.py:290 segment_cache, server/prompt_literals.py credit path): verify it reuses token ids across turns of a growing agent conversation; if it re-encodes the full text, adopt (~0.3-1 s TTFT at 100-400K; not C1 decode) |
| 320 | pf_overlap | per 128-expert tile: dequant/transcode on a side stream overlapped with grouped GEMM of the previous tile (events), bit-identical | DS41 140 prefill MoE | seqdsprefill-1 arm O (320 alone): pf8k x0.972, pf32k x0.997, pf128k x0.993 = NO GAIN (verify 0 mismatches) | OFF (overlap 0) | rejected-pattern: side-stream overlap of DRAM-bound dequant vs smem-bound GEMM did not pay on V100 |
| 321 r1-3 | pf_empty | gate upstream per-pass torch.cuda.empty_cache() (cudaFree = device sync + re-malloc: ~33 ms GPU idle + 7-12 ms per 2048-token pass, ~8 %) by floor/adaptive | DS41 upstream model code | seqdsprefill-1 arm E (floor): pf8k x1.047, pf32k x1.031, pf128k x1.030, load x1.058, min free 690 vs 1294 MiB; seqdsfloor-2 arm A (adaptive r3): pf32k x1.032, pf128k x1.028 (verdict "gain < 2 % beyond noise: keep off", adopted anyway 2026-11-09) | PROD (rev 3 superseded by rev 4) | already-have: TF qwen4_exp forward has no per-pass empty_cache (only reference/exl3/hc_readout setup paths); lesson = never release allocator per pass |
| 321 r4 | pf_empty4 | probe gate: skip release only if measured growth bound fits above floor (rev 3 was inert: 1786/1788 passes released) | DS41 | seqtrain-2 arm D alone NEUTRAL (+4.1 % clear, +4.3 % cold prefill, < 8 % bar); arm E with 277 "NEGATIVE" (worst gap +0.48 s > +0.3 s limit) but owner adopted (steady 85.3 vs 85.1 tok/s) | PROD (probe_mb 256) | n/a (no per-pass release in TF) |
| 340 | kern340 | TurboMind lattice MoE prefill GEMM re-instantiated as 9 tile variants (padded A rows +8 halves, warp 32x64, CTA_K 64, 16x128...) to remove smem bank conflicts (ncu: 5.75 wavefronts/LDS, L1/shared pipe 87-92 %, tensor 29-34 %); rule table per (n,k,routes) from a micro bench; same K order = bit-identical | kernel shapes DS41-specific; method generic | seqdskern-1: micro x1.308 (234.6 -> 180.5 ms at 2K), server K/B pf8k x1.056, pf32k x1.064, load x1.045 (noise 0.3-1.6 %), greedy+PPL identical, verify 0 mismatch | PROD (rule enabled, 24 rules, speedups 1.15-1.42) | ADOPTABLE method: ncu our sm70 prefill/MoE mma kernels (cuda/sm70/gguf_ops.cu, onecat_moe_gu.cu/onecat_moe_down.cu, prefill_dense.cu) for shared-load wavefronts/request > ~4 and bank-conflict share; pad A rows / widen warp N tile. Expected prefill +5-6 % if same conflict class; decode untouched |
| 350 r2 | input_fast | pure-decode steps: NumPy-computed inputs written into the SAME persistent buffers via one torch._foreach_copy_; block table = only [rows, used blocks] rectangle packed into pinned memory -> 1 H2D + 1 D2D (was 1 MiB full table + 11 H2Ds); logits idx = view of device arange; memoised batch descriptor; reused metadata objects; md5 guards + verify-N on GPU | generic V1 runner | seqdsinput-2 (rev 2) vs base: c1 x1.0225, c8 x1.030, c13 x1.017, c14 x1.024, c32 x1.018 (noise 0.1-0.9 %); stage-0 prepare_inputs -0.31 ms, attn_metadata -0.15, launch -0.57, runner -0.57 ms; CPU 1.70 -> 0.41 ms at 16 rows; rev 1 broke (import-hook recursion with 300) | PROD (+VERIFY 200) | ADOPTABLE: maps to TF verify staging 3.1 ms/round (issue_coordinates pageable torch.tensor H2D syncs the stream - LEDGER dec4). Pack all per-round small inputs into one pinned staging buffer -> one async H2D into persistent device buffers. Expected 1-2 ms of the ~16 ms host/round (+2-4 % C1). Overlaps LEDGER survey item 1 (persistent device metadata) - same family, concrete recipe |
| 351 | input_parity | twin persistent InputBatch per PP parity to avoid remove/re-add of 16 rows per step | PP2 only | never measured (seqdsinput-1/2: "P not run") | NOT | n/a (no PP) |
| 390 | pfx_ladder | anchor (state snapshot) every 32768 tokens inside long prefills, ladder rungs protected from STALE eviction, host-ring refresh | any model whose prefix reuse needs a state snapshot (DS41 window rings; = GDN state for Qwen) | seqdsladder-1 POSITIVE: fork TTFT 17.7 -> 4.0 s (x0.226), cached 0 -> 65536; host-tier fork 8.3 vs 21.4 s; exact identical | PROD | already-have concept: TF prefixes.py LADDER (doubling-distance points per chain, GreedyDual-Size) |
| 410 | cache_output | reasoning-mode requests also cache OUTPUT blocks (fork cached prompt blocks only) so the next agent turn reuses the reply | any agent/thinking workload | seqdsreply-1 NOT POSITIVE (uncached/turn x0.723, TTFT med x0.449 but mean x0.906); seqdsreply-2 NOT POSITIVE (uncached x0.797, TTFT med x1.37 worse); adopted anyway (cache_output.json enabled) | PROD (contradiction: both windows not positive) | check TF retention captures at reply end (decode anchor), not just prompt end; evidence weak |
| 450 r3 | expert_stats | 1 graph-safe Triton atomic histogram per MoE call (+ exact distinct experts per <=32-row call) | MoE generic | seqcycle2rs-2 arm E (off) vs A: C1 +2.0 %, C2 +1.6 %, C8 +0.8 %, C32 -0.6 % = it COSTS ~2 % at low concurrency | PROD (observation) | lesson: even "one atomic launch per MoE layer" costs ~1 %/launch-class at C1; keep diagnostics out of C1 graphs |
| 460 | cold_experts | cold routed experts in pinned host RAM, in-graph PCIe copy into pool slots, VRAM -> KV | MoE generic | seqcoldx-1c: th0.10 KV x1.118, decode c8 x0.685, c32 x0.571; th0.05 KV x1.055, c8 x0.850, c32 x0.820; seqcoldx-2b r2 (10.7 GB/s copy) c8 x0.873 c32 x0.835 | OFF (rejected) | REJECTED path: on-demand PCIe expert fetch at 2-11 GB/s costs 13-43 % decode; 1-6 % miss rate already too much |
| 470 | ring_keep | host ring table: superseded-first victim (ancestor rings of a continuing conversation evicted before LRU), 390 rungs exempt | any prefix-state host tier | seqcburst-1 POSITIVE: return after churn cached 0 -> 49600, TTFT 12.09 -> 0.36 s, cold prime x0.962 (no clear cost); replay of 36 h opencode: LRU lost 6/110 returns, superseded-first 0 even at 512 rings | PROD | ADOPTABLE concept for TF retained_catalog.py / prefixes.py: mark an entry superseded when a descendant of its chain is saved, evict those first (no "superseded" notion found in TF). Effect: TTFT of returning agents, not decode |
| 277 | prefill_train | 271 admitted prefill only in PAIRS (owed cool-down holds the gate) -> both PP stages idle 0.3-0.5 s per pair in anti-phase; trains of N passes, owed cool-downs accumulate, time share unchanged | PP2 + 271 | seqtrain-2 arm B (train 4) POSITIVE: cold clear +13.9 %, cold prefill +16.1 %, decoders +3.7 %, worst gap -0.06 s, GPU<50 % util 26 -> 15 % (stage 0); train 6 NEGATIVE (gap +0.37 s) | PROD (train 4) | lesson only: TF has no PP; analogous check = any admission rule that idles the GPU between prefill chunks (tp_scheduler.py / lane_admission.py) |
| 480 | readd_tail | when a decoder is removed/re-added to the persistent batch (PP micro-batch balancing), copy only the row tail from prompt end instead of the whole prompt; condense moves only tail | V1 runner generic | seqcycle2ab-2 (verify mode): live decode x1.031 POSITIVE; seqcycle2rs-2 B: live x1.052 (verdict INVALID-by-env-check), 8.6e9 prompt tokens not copied | PROD | check TF for O(context) per-round host copies (canonical_tokens / ngram context / stream.context in state_retention.py, ngram.py); only matters at long ctx |
| 481 | ram_plan_fast | host-save planner converts only the saved pages' ids (not the whole prompt) and lazy chunked stable argsort for victims (was full 196,608-slot argsort per saving step, 30-70 ms) | any host tier | seqcycle2ab-2: gap MAX x1.044 better, decode x0.996 | PROD | check retained_catalog.py victim selection for full sorts per step |
| 482 | mm_once | ship image pixel payloads to workers only with the step that encodes them | vision | seqcycle2ab-2 NEUTRAL | PROD (enabled anyway) | low |
| 483 | worker_gc (288) | gc.freeze after graph capture in workers + 60 s GC pause stats everywhere | Python engine | seqcycle2rs-2 arm C (freeze): no gain (c1 x0.997, live x1.003; worker gen-2 MAX 404 -> 422 ms, frozen objects only 375). Stats: worker gen-2 152 x/2 h, MAX 405-463 ms, ~320 ms per process-minute; engine 115-130; API 95-105 | PROD as stats only (=2) | ADOPTABLE measurement: TF runs 8 lockstep TP rank processes; any rank's 300-450 ms gen-2 pause stalls every rank at the next collective. Measure gc.callbacks pause stats per rank; if similar, gc.freeze after warmup + gen-2 threshold/idle collection. Effect on p99 token gap, ~0.5 % mean |
| 490 | detok_tail_prime | streaming detokenizer primed with the last 24 prompt tokens instead of whole prompt (whole-prompt decode cost 66-194 ms at 140-460K on the event loop) | any HF DecodeStream server | seqcycle2ab-2: gap MAX x1.251 better (0.73 -> 0.58 s); 3030 synthetic streams identical | PROD (1) | already-have: TF cuda/reply_text.py StreamDecoder decodes only generated ids |
| 491 | toolcall_stream | DSML tool-call streaming parser no longer buffers whole scalar params; skips unmatched text | DS41 parser | seqcycle2ab-2 VERDICT FAILED-EXACTNESS (greedy c8 on differs from off, 2 real); tool delta 2534 -> 265 chars, gap 21.2 -> 2.5 s | PROD (CONTRADICTION: enabled despite failed exactness gate) | TF own parser; low |
| 492 | toolcall_stats | observation counters | DS41 | n/a | PROD | n/a |
| 500 | miss_reason | one log line per request whose prefix hit is >16K below the block-cache hit, per tier why (dropped when/age, slot reused, pages missing...) | any multi-tier prefix cache | no GPU gate (observation) | PROD | ADOPTABLE cheap diagnostic for TF retention misses |
| 501 | turn_anchor | pin per conversation the anchor its latest burst restored from (<=48 snapshots/rings) | prefix tier | CPU only (32 checks), never GPU-tested | NOT (deliberately absent) | idea |
| 502 | tail_anchor | learn per conversation a constant "moving tail" L (client re-renders last L tokens); place an extra anchor at P-L in the prefill step containing it | any state-snapshot prefix cache | CPU replay -71 % trimmed; seqtail-2 POSITIVE: trimmed/turn t850 2517 -> 21, t5100 1664 -> 21, switch 587 -> 11; TTFT p50 1.69 -> 0.84 s (t850); clean unchanged | PROD | ADOPTABLE for TF hybrid-GDN prefix reuse (GDN state must be snapshotted at an exact position, like DS41 rings): agent clients re-render the tail -> learn L and snapshot at P-L. Effect: TTFT/prefill work on agent turns (17 % of their prefill work was this pattern) |
| 503 | session_pin | one host checkpoint per conversation pinned from LRU for 2 h (<=128 pins, <=40 % pages), sidecar survives restart | prefix tier | CPU only (64 checks), store-gate OK; never GPU-tested | NOT | idea (owner policy) |

Overlays with earlier numbers (2xx, 420/430) in my overlay list - one line each (status from production env/knobs):
262 pp_async_out PROD: last PP stage returns sampled tokens via async D2H + event instead of a blocking _to_list
(cudaEventSynchronize 16 ms at c32) -> seqppovl-t2 GO c8 x1.092, c16 x1.063, c32 x1.043, c1 x0.995 (PP-generic; TF
equivalent = LEDGER survey item 1 "one async D2H/round", already known). 263 dec_fuse NOT (DEC_FUSE absent). 266 async_sched
NOT POSITIVE (seqasched-t1c sampled c8 x0.878, c32 x0.927; c1 x1.10). 255 host_probe (CUDA events at execute_model entry to
measure true GPU idle; torch profiler inflates host 2-3x) - methodology, not prod. 256 moe_sum2 PROD (shared+routed add fused
into TP all-reduce push kernel; -1 launch/MoE layer). 257 hc_dec PROD. 240 EM3 PROD. 250 dec_attn / 251 NOT. 245 iq2mma NOT.
210 dsls rev 5 removed from production 2026-11-03 (runtime.dsls5-removed). 261 pfx_retain PROD. 231 ram tier rev 5 PROD
(ram4-t1 NEGATIVE c4x128k x0.978; stepbud-t2 RAM tier NO-GO c4x128k x0.890 earlier - later revs adopted). 267 ctx_1m PROD
(GO: c1 x0.991..c32 x0.993; 1M needle prefill 264 s, decode 52.3 tok/s). 268 long_prefill_mem PROD (seqmemfix FAIL on min-free
bar 1089 MiB < 1100 but adopted). 269 step_prefill PROD (STEP 1024; stepbud-t2 NO-GO). 271 prefill_share 0.5 PROD
(share2-t1 POSITIVE: mixed8 decode/user during 128K 2.43 -> 21.9 tok/s, but 128K TTFT x2.78). 272 ram_numa PROD (speed
neutral c8 x1.002 c32 x1.002; fixes node-0 OOM). 273 small_first / 274 share_boost: env absent but ON via decode_cap live
file (small_first 16384, share_boost 0.8) although seqdsadmit-2 GO=False and seqshareboost-1 all arms NEGATIVE (decoder max
gap 3.3-3.8 s > 3 s limit; clear time +35-43 %) - owner override. 275 lookup_skip PROD (seqdsadmit-1c GO: dec x1.25,
lookups x0.006). 276 image_text PROD. 287 share_floor NOT. 420 image cache / 430 mm_fast PROD (seqdsimgkeep-2: positive only
with 430). chunksweep (seqchunk-1): chunk 1024 qualifies (p95 gap +38 %, cold prefill -19 %), 512/256 trade-off; prod cap 2048.
hookfix overlay = import-hook rev 2 for all site-packages modules (rev 1 hooks recursed when two hooked the same module).
iq2s/dattn/dmoe/hcar/engshard/numa/ram*/share*/admit*/memfix/ctx1m/ppovl/pfxr/imgkeep*/imgtext belong to patches 2xx/4xx
covered above only at status level.

### 4. How they restructured 1Cat for DS41 (visible from this scope)
- Process model: vLLM V1 multiproc (API server process, EngineCore process, 8 GPU worker processes), plus an external Engram
  holder process that owns ~110 GB mlocked RAM and hands it out over a unix socket (restarts keep it). PP2 x TP4 with NUMA
  membind per stage; host prefix store in /dev/shm (persistent across restarts, identity = vLLM code digest + VLLM_* env
  digest, so later features are written as site-packages modules + .pth import hooks with LOCAL_* (non-VLLM_) env names to
  keep the store valid). Every patch: md5-anchored "needles" on the code it wraps, DRIFT = install nothing, wrapper error =
  self-disable, live JSON knob re-read <= 1 s, verify-N mode comparing against the original path on the GPU.
- Scheduling: synchronous V1 scheduler + PP micro-batch balance; prefill limited per request (185 cap), per pass (269
  step_total 2048), by time share (271 share 0.5, depth 2) with trains (277), queue-aware boost (274) and small-first (273);
  prefix lookups skipped in held passes (275).
- Decode: FULL_DECODE_ONLY CUDA graphs at every size 1..16,24,32; async last-stage output (262).

### 5. Top adoptable for TensorFold (ranked by expected effect on OUR bottlenecks)
1. 350-style packed pinned input staging (one async H2D of a packed rectangle into persistent device buffers; no pageable
   torch.tensor H2D in the round). Target: verify staging 3.1 ms + MTP per-step staging ~2.5 ms/step (LEDGER dec4).
   Their result: -0.57 ms runner/step, +1.8-3.0 % decode at c1-c32. Ours plausibly 1-3 ms/round (+2-6 % C1). Files:
   families/qwen4_exp/cuda/input_preparation.py, position_coordinates.py, descriptor_upload.py. (Same family as survey item 1.)
2. 340-style ncu bank-conflict audit of sm70 prefill/MoE mma kernels (gguf_ops.cu, onecat_moe_gu/down.cu, prefill_dense.cu):
   pad A rows, warp 32x64 so one A fragment feeds two N atoms, CTA_K 64; per-shape rule table from a micro bench, bit-identical
   K order. Their prefill +5.6-6.4 %. Prefill only.
3. GC pause measurement per TP rank (483 stats) then gc.freeze/threshold policy: their workers see 300-460 ms gen-2 pauses;
   in our lockstep TP8 one rank's pause stalls all. Tail-latency + ~0.5 % mean. Cheap to measure (gc.callbacks).
4. Agent-workload prefix retention for hybrid GDN state (the DS41 window ring is the same problem): 502 tail anchors
   (seqtail-2 trimmed/turn 2517 -> 21), 470 superseded-first eviction (returning agent TTFT 12.1 -> 0.36 s), 500 miss-reason log.
   TF already has the 390-like ladder (prefixes.py LADDER) and GreedyDual retention. Effect: TTFT / prefill work, not decode.
5. 310 prompt-tokenization prefix reuse (verify TF segment cache covers growing conversations): -0.34 s TTFT at 118K,
   -1.1 s at 402K.
6. Lesson-level: keep observation kernels out of decode graphs (450 costs 2 % at C1); 480/481 = remove O(context) host work
   from per-step paths at long context.

### 6. Rejected / failed paths (with numbers)
- 460 cold experts in host RAM: decode c8 x0.68-0.87, c32 x0.57-0.84 (PCIe 2.7-10.7 GB/s, 200-900 us per expert copy).
- 320 transcode/GEMM side-stream overlap: pf8k x0.972, pf32k x0.997 (no gain).
- 266 async scheduling on PP2: c8 x0.878, c32 x0.927 (c1 +10 %).
- 483/288 GC freeze after capture: no gain (frozen 375 objects; pauses come from the live heap).
- 277 train 6 and 277+321r4 failed their own worst-gap gate (+0.37/+0.48 s) - 4+probe adopted by owner anyway.
- 491 toolcall stream: FAILED-EXACTNESS (greedy c8 2 real diffs) yet enabled in production.
- 410 cache_output: two NOT POSITIVE windows, adopted. 274 share_boost: NEGATIVE, live knob on.
- 351 parity batch / 300 timer: never measured / costs 1.8 % c8.
- seqdsutil-1: numa_balancing=0 (c8 x0.993), EngineCore on node 1 (x0.999), logging off (x0.995) - all neutral.
- dsls (layer-sliced long prefill) rev 1-3 failed exactness/screening; rev 5 GO then removed from production 2026-11-03.

### 7. Methodology lessons
- Their noise floors: decode c1 SE 0.05-0.2 % (3 repeats), restart-to-restart (arm F vs A) 0.0-0.3 % for 512/512 decode;
  live-shaped 8+1 at 220K: cv 0.8-1.3 %, restart noise 0.6-2.7 %; greedy restart-to-restart control shows "REAL diffs" too
  (near-tie classification with logprob gap) - they gate exactness against a restarted control arm, not bit-identity.
- Every window: idle sampler proving no foreign load, A/B/A2 with the repeat of the base as noise, verdict scripts with
  predeclared bars (e.g. decoder -5 %, worst gap +0.3 s, >= 8 % gain), install-line counts as a gate, GPU min free per GPU.
- Live-shaped metric (8 decoders + 1 prefiller at 220K ctx, 768-token suffix) as the production-relevant decode number;
  idle 512/512 tables barely move for host-path patches (480: c1 x1.001 but live x1.03-1.05).
- Tail-pass profile (seqcycle2rsg-2): DS41 decode-only step per stage at 220K ctx = GPU span 12.3-13.7 ms with compute
  9.2-9.8 ms (~3-4 ms host/NCCL gap per stage step even in production); prefill pass fixed part 242/254 ms per stage of
  which host gaps ~103 ms (profiler-inflated 2-3x).
- Power cap: their GPUs report 225 W limits during every window.

### 8. Open questions
- Is our V100 power limit also 225 W (their numbers are at 225 W)? (read-only nvidia-smi query; user decision to change.)
- Does TF segment_cache reuse token ids across turns of a growing conversation (310 equivalent) or only template segments?
- Does TF per-round host path contain O(context) copies at 262K ctx (480 analogue)? Not measured in dec4 (C1 at short ctx).
- No c1es8 512/512 decode table exists; production numbers quoted are c1es6 (+older image for C16).
- Why production enabled 491/410/274 despite failing/negative gates (owner decisions, not evidence).
