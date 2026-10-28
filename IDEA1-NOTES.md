# Idea 1 working notes (not committed): exact multi-CTA QSA select (T) + batched streams (B)

Worktree /Warehouse/TensorFold-wt/idea1, branch perf-idea1-select (base b536f2c3). Design:
scratchpad/design-1-select.md. Policy: /home/kklouzal/CLAUDE.md. Task ledger: ~/.tensorfold-perf-ledger-20261013.md.

## Objective / contract
- P0: ids/nk/sparse bit-identical to `_select`/`_select_tiles` (whole buffers incl. untouched positions); scores
  unchanged (same `_scores` arithmetic, tile [64,128], 4 warps). Determinism: integer-only after scores.
- P3 (decode REGION, dev-01 first, GB10 later): per-call device time of select (and scores+select) at blocks
  32768/75000/150000/262144, rows 1/3/5/20; crossover 8k/16k/32k. Predeclared acceptance: design §6.4.
- No new host syncs; CUDA-graph capturable; bounded scratch counted in the memory plan.

## Environment
- dev-01 GPU: NVIDIA RTX PRO 2000 Blackwell, sm_120, 34 SMs, L2 32 MiB, 16 GB (shared; ~4.5-5 GB used by another
  process). Image tensorfold-python-integration:amd64-verification-2edfaa51: torch 2.16.0.dev20261006+cu134,
  triton 3.9.0, CUDA 13.4.

## Ledger
| item | state | evidence |
|---|---|---|
| baseline `-k qsa` kernel tests | pass 4/4 | logs/baseline-kernels-qsa.log |
| T v1 (count kernel, while-emit, TB 4096) | exact (smoke vs tiles+oracle) | scratch/idea1/probe2.py: dev 30-36 us @1 row, host ~135 us |
| host launch cost | measured | Triton 3.9 nop launch 12.7 us host, 5-arg kernel ~17 us (Xeon E3-1285v6); slice zero_ 4.4 us vs full 1.6 us |
| O2 (count folded into pass 3 via per-slice suffix sums [rows,CW,257]) | adopted | host 135 -> 110 us; one fewer data pass |
| emit restructure (for-loop + if placed<last; one packed reduction; rank scan only in the straddling tile) | adopted, exact | ablate2.py: 5x150k emit 34 -> 22 us; 20x262k 182 -> 163 |
| TB 2048 (4 warps), pass-3 vector accumulator | adopted | tune.py sweep: TB 2048 best/near-best for rows 1-20 |

### dev-01 probe (graph replay = device us; host = enqueue us; median of 50) after the above
| blocks | rows | old kernel | old dev | old host | T dev | T host |
|---|---|---|---|---|---|---|
| 16384 | 1/5/20 | _select | 41.9/42.2/42.5 | 25 | 23.5/25.5/37.7 | 110-112 |
| 32768 | 1/5/20 | _select | 122.3/132.6/134.6 | 25 | 23.6/26.4/49.9 | 109-111 |
| 75000 | 1/5/20 | tiles | 232.9/239.1/239.1 | 22 | 23.5/36.3/89.6 | 108-112 |
| 150000 | 1/5/20 | tiles | 443.8/456.2/456.2 | 23 | 32.1/48.6/167.4 | 109-114 |
| 262144 | 1/5/20 | tiles | 753.1/775.6/775.6 | 23 | 33.8/71.1/275.9 | 109-115 |
Decision pending bench: MULTI_MIN_BLOCKS stays 16384 (T loses eager-isolated below it: +85 us host vs <=20 us device gain).

## Tests (pre-review revision)
- tests/cuda/test_flashnext_qsa_select.py (G1-G6, G8): 17 passed (logs/qsa-select-2.log); mutation check: 3 seeded
  kernel defects (equal keys all chosen; histogram bins of count 1 dropped; prefix ignoring need) -> 7/13/7 failures.
- G7 in tests/cuda/test_flashnext_multi.py: sparse streams x {new, new-T, old} x {bf16, int8} + long stream T forced
  eager/graphs: 8 passed (logs/multi-g7-3.log); each variant's path asserted by launch counters.
- Suites kernels+qsa_select+multi+forward: 145 passed (logs/cuda-suites-1.log).
- CPU: geometry/memory-plan/admission/kv-dtype/capacity-dispatch/ab-switch/oracle: 149 passed, 1 failed =
  test_unified_available_memory_uses_reclaimable_host_pages, which fails identically on a pristine export of b536f2c3
  (environment-dependent; unrelated).
- Tooling gap: no ruff/pyflakes in the image or host (no network to install); 120-col checked by awk.

## Design C (cooperative CUDA kernel, tools/qsa_select_coop.cu, bench-only)
- 11/11/10-bit passes, smem hist with __match_any_sync aggregation, 4 grid syncs, hist zeroed at kernel end.
- Exact: equals T and the oracle on 24 cases incl. bits/tied/flat, 600..262144 blocks, rows 1-20 (coop_check.py).
- CUDA-graph capture of the cooperative launch works (CUDA 13.4, torch 2.16 dev).
- coop_time.py (dev-01; graph = amortized 20/graph; eager = isolated call from idle GPU; host = enqueue):
| blocks | rows | T graph | T eager | T host | C graph | C eager | C host |
|---|---|---|---|---|---|---|---|
| 16384 | 1/5/20 | 20.6/21.3/33.4 | 131-188 | 110 | 25.0/27.2/40.7 | 37.8/35.6/48.6 | 5.9 |
| 75000 | 1/5/20 | 22.5/33.1/86.1 | 133-163 | 112 | 27.6/40.5/87.4 | 38.0/48.3/95.3 | 5.9 |
| 150000 | 1/5/20 | 30.8/45.8/163.9 | 131-216 | 112 | 30.9/53.1/144.9 | 39.4/60.9/153.3 | 6.0 |
| 262144 | 1/5/20 | 32.7/66.9/275.1 | 131-312 | 113 | 37.2/72.2/232.3 | 45.2/80.8/240.3 | 6.0 |
- Disposition: NOT selected (mixed: device-only slower at rows<=5 by 0-10 us, faster at 20 rows; host -104 us/call).
  Needs owner decision + GB10 e2e evidence (is the eager round host-bound?); adds a compiled extension and cooperative
  launch (co-residency cap: 136 CTAs dev-01).

## Preliminary bench (bench/prelim-T.json; pre-review source; old single-stream path sliced the scratch -> its host
time is ~16 us high, fixed later) and slice tuning (bench/tune2.log, graph-amortized device us)
- T device (graph) vs old: 32768: 28.7/30.4/33.1/56.3 vs 129.8/135.9/139.1/142.2 (rows 1/3/5/20); 150000: 38.3/46.0/
  56.1/175.1 vs 451.5/465.8/465.7/466.0; 262144: 39.2/56.1/78.6/288.6 vs 762.8/787.3/790.1/788.0.
- Isolated eager (device + exposed enqueue): T ~155-180 us (rows 1-5) vs old 173-846 us from 32768 blocks up; at
  8192/16384 blocks old (_select) 60-80 us (policy) beats T ~156-174 -> MULTI_MIN_BLOCKS = 16384 kept.
- tune2: TB 2048 best everywhere. FILL 4 best at rows 1-5; rows 20 best at FILL 8-16 (150k: 137.5 vs 162.6;
  262k: 199.2 vs 255.4). Pattern: best slices are 3-5 tiles (6144-10240 keys); FILL 4 at 20 rows gives 11-19 tiles.
  -> candidate: cap a slice at 5 tiles (keeps rows 1-5 unchanged, 20 rows -> ~10240-key slices).
