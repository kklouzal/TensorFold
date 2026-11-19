# Gemma 4 SM70 concurrent serving (`--parallel` 1-32)

The eight-V100 Gemma 4 26B-A4B provider (`src/tensorfold/families/gemma4/sm70/`) serves up to 32 requests at once.
This note records the design, the memory budget, numerical contracts and validation status. Measurements are
listed with their source, image and configuration; anything not listed is unmeasured.

## Scheduling

Rank 0 runs the shared CUDA `Scheduler` (`tensorfold/cuda/scheduler.py`, also used by the Qwen CUDA engines):
the admission queue, background priority, cancellation of waiting/filling requests, and per-request token
delivery on the request's own thread. The provider's `Decoder` (`engine.py`) implements its decoder interface.
Each scheduler round is pipelined:

1. wait for the batched decode round in flight (one CUDA event wait that releases the GIL, so the requests'
   own threads detokenize and stream while the GPU runs) and decide each stream's tokens;
2. launch the next batched decode round at once for every stream that continues;
3. while it runs, deliver the decided tokens and issue the next prompt chunk. Prompts fill round-robin (each
   waiting prompt's next fixed 1,024-row chunk in turn, foreground first). Chunk boundaries are fixed per
   prompt, so a prompt's arithmetic never depends on load. Concurrent prompts finish together and their
   decode batches stay full, and every prompt chunk is followed by one batched decode round.

Requests without grammar or penalties draft with DFlash, greedy and sampled alike. Every verify row is sampled
from the target distribution by the request's own position-keyed sampler (`tensorfold.cuda.sampling.sample_rows`)
and a proposal survives only when it equals the token sampled at its row (`engine.draft_acceptance.accepted_prefix`).
Each emitted token is therefore exactly the token a serial step would have sampled at that position, so a sampled
reply is the serial reply (rejection sampling with a point-mass proposal: acceptance is the target probability of
the proposal) and drafted and `"draft": false` replies are equal. Latency-bound batches (padded
verify rows within one 16-row tile) always draft, as the C1 engine does. Wider batches compare the measured
cost of a DFlash round and of a serial step at that batch size, scaled by the observed tokens per round, and
re-probe DFlash every 32 serial rounds. Output does not depend on that choice. Requests with grammar or
penalties take one row per round. Plain serial batches of up to 16 rows
gather complete vocabulary rows inside their CUDA graph, as the C1 engine did. Other batches exchange
per-shard maxima (value, index, NaN/+inf flag) and gather complete rows only for the requests that need them
(sampled, grammar, penalty and scored requests).

Followers mirror the control frames (attach, detach, prompt chunk, batch, stop and the prefix-cache operations below) over the existing same-host
control pipes and run identical kernels and collectives.

DFlash rounds carry no pending state. The round's drafter block writes K/V at positions `start..start+block-1`.
The target then verifies those rows, and the drafter appends every verified row's taps at the same positions.
Rows beyond the accepted prefix are rewritten by the next block, which starts inside that range, before any
read. Serial steps of drafting requests also append their taps. A stream can therefore switch between DFlash
and serial rounds in any round.

## Prompt-prefix reuse

`prefix.py` keeps prompt states and resumes later requests from the longest kept point that prefixes their prompt,
with Flash Next's retention policy (`families/qwen4_exp/cuda/prefixes.py`: resume in place, fork a copy into a free
slot, GreedyDual-Size value over prompt tokens, a chain ladder of spare points). Every request without media and with
drafting allowed reuses, at any temperature: the sampler does not change prompt state. A `"draft": false` request
neither borrows nor keeps. Usage reports the reused tokens as `prompt_tokens_details.cached_tokens`; `/stats`
`prefix_cache` and the Prometheus `prefix_cache_*` families report entries, lookups and hits/tokens by tier.

- Points: 1,024-row prefill chunk boundaries at or before the chat template's message starts
  (`cuda.markers.resume_points`: the second message, the last assistant start and the last other message) and
  before the prompt end, plus each prompt's exact end (with its last-row logits) for exact repeats, kept as the
  point one token before the end (`prefixes`' convention). Prefills are chunked at multiples of 1,024 from
  position 0, and every resume starts on such a boundary (a request that merely extends an exact end resumes at the
  boundary below it), so a resumed request computes exactly the chunks a cold prefill of its prompt computes and
  replies bitwise like the cold request; an exact repeat computes no prompt row. The price is recomputing up to
  1,023 rows of the shared prefix (measured below).
- State of a point: full-attention caches are linear, so the rows below a point stay valid while later rows are
  rewritten before any read. Sliding-window rings are overwritten as a slot moves on, so the ring rows a point needs
  (its last window + 1 positions: 25 target layers and the drafter's sliding layers, 29.7 MB per rank) are copied
  out when the prefill reaches the point (before the next chunk). The drafter ring is 3,072 rows (window 2,048 +
  one chunk) so a chunk never overwrites the window behind the prompt end. Every prompt row feeds the drafter, so a
  kept point serves later drafting requests.
- Slots: `--parallel` live streams plus `max(8, --parallel)` slots for idle kept conversations (plus one padding
  slot); each slot's rings exist from startup. A fork copies the full-cache rows below the point and restores the
  snapshot; a resume in place restores the snapshot. A slot whose capacity is too small grows by copying its rows.
- Memory: attached full caches and snapshots are charged to the KV budget. A request that needs room first releases
  free slots' caches, then evicts the least valuable idle chains; it waits only for memory live streams hold.
- Host tier (`--prefix-host-bytes`, off by default): points `prefixes` drops are copied (full-cache rows below the
  chain's longest point and every point's snapshot) into one pinned slab per rank allocated at startup, in
  contiguous runs of 2 MiB blocks (least valuable entries evicted first), and restored byte-exactly into a free
  slot when a prompt extends a host point by at least 64 tokens beyond its best VRAM point. Copies are ordered on
  the device stream; nothing is written to disk. An exact end on the host serves only exact repeats.

## Memory

Per rank, every slot (`--parallel` + 1 padding slot) has sliding-window rings from startup:

- target: 25 layers × 2,048 rows × 256 × FP16 × K/V = 50 MiB;
- drafter sliding layers: 4 × 2,304 rows × 128 × FP16 × K/V + positions = 4.5 MiB (rings replace the earlier
  full-context linear caches).

A stream's full-attention caches are allocated (zeroed) at admission with capacity
`min(context, prompt + max_tokens + block)` rounded to 256 rows, and freed when it finishes:

- target: 5 global layers × 512 × FP16 × K/V = 10,240 B per token;
- drafter full layer: 128 × FP16 × K/V = 512 B per token.

Total 10,752 B/token/rank. Admission takes bytes from a ledger. Its budget is the smallest, over all ranks, of
free device memory after loading, warm-up and graph capture, minus the measured prompt-chunk transient and
1 GiB. A request that does not fit waits (`NoRoom`, held first in line) until a stream finishes. A request
larger than the whole budget is refused.

| Per-stream tokens (prompt + output) | KV per stream (MiB, rank) | Streams in a 24.5 GiB budget |
| ---: | ---: | ---: |
| 4,096 + 1,024 | 58 + 55 rings | ≥ 32 |
| 16,384 + 1,024 | 184 + 55 rings | ≥ 32 |
| 32,768 + 1,024 | 352 + 55 rings | ≥ 32 (61 by bytes) |
| 65,536 + 4,096 | 720 + 55 rings | 32 (34 by bytes) |
| 131,072 + 4,096 | 1,392 + 55 rings | 17 |
| 262,144 | 2,694 + 55 rings | 9 |

The 24.5 GiB budget is the measured startup value at `--parallel 8` and `--parallel 32` with `--context 32768`
(source in this change, image `cf8c0ff6`); it shrinks slightly with the graph catalogue and context. The
server prints its own budget at startup.

## Numerical contracts

- Decode forwards use the C1 arithmetic for any number of rows by default. The W4/BF16/INT8 GEMVs and router run row
  tiles chosen by row count (1 to 6 rows; 16 for wide BF16 heads), with per-row arithmetic independent of the row count
  and the tile. Each stream's attention is its own segment, with its own KV slot, the C1
  chunk size and the C1 split rule. Per-stream results are therefore bitwise equal to the same stream alone.
  Batches beyond one row tile are compute-bound on CUDA cores (FP32 GEMVs at about 2 TFLOP/s).
- Optional, off by default (`TF_GEMMA_WIDE_ROWS=N`): decode forwards of at least N rows use tensor cores. Activations are split into FP16 hi + lo (about 22 significant bits),
  weights decode exactly to FP16, and accumulation is FP32. Routed experts are grouped per local expert at 96
  or more rows (per-pair GEMVs below). The head is an exact per-row power-of-two-scaled FP16 copy of the BF16
  shard. Partials are all-reduced instead of all-gathered and summed in rank order. These forwards are not
  bitwise equal to C1. A stream's greedy reply can differ from its C1 reply once its padded batch is wide
  (observed: 1 of 8 replies at C8 and 3 of 32 at C32 diverged late in otherwise coherent text): a
  load-dependent numerical result. Quality: teacher-forced through the wide arithmetic at C1
  (`TF_GEMMA_WIDE_ROWS=1`) against the default arithmetic, `tools/quality_eval`, draft rows 3, `--repeat 2`:
  - corpus v1: prefill equivalent; decode d = −0.0049 nats/token, 95% interval [−0.0132, +0.0022],
    inconclusive;
  - BOS-sensitivity corpus: prefill equivalent; decode −0.0026 [−0.0154, +0.0076], inconclusive.

  Top-k MoE routing amplifies the small arithmetic differences: median |Δ log-probability| was 6e-4 nats, with
  rare outliers of 2–3.5 nats late in documents. Neither interval is within the predeclared 0.005 margin, so
  the wide arithmetic is not enabled. It awaits a larger frozen corpus. The default arithmetic's teacher-forced
  scores were identical to the previous engine's for every prefill and decode position of corpus v1.
- Prompt chunks keep the C1 prompt arithmetic and fixed boundaries.
- The drafter's sliding layers read rings instead of linear caches, which reorders their attention sums.
  Proposals can differ from the earlier implementation; verified output cannot.

## Validation (2026-11-19, image `4b2661d8`, eight V100-SXM2-32GB, `--context 262144`, INT8 DFlash 5 rows)

Evidence in the claude-perf ledger section "Gemma 4 V100 parity (g4p)"; one server session per row.

- Replies: 5 greedy and 9 seeded sampled chat replies (plus the 9 with `"draft": false`) equal the previous
  one-lane engine (`3f49a0b2`) at `--parallel 1` and `--parallel 8`, with and without the host tier.
- Prefix reuse: replies resumed in place, forked, from an exact repeat and from an extension (resumed at the chunk
  boundary below the kept end) equal the cold `"draft": false` reply bitwise (4 documents of quality corpus v2);
  a conversation evicted to the host tier and extended replies identically. Teacher-forced exact repeats score
  identically to cold (80 documents). Resuming an extension one token before its end without the boundary
  alignment was measured inconclusive against cold (decode d +0.0043 nats, 95% CI [-0.0165, +0.0229]) and is not
  used.
- Chat (two interleaved conversations with 7-11K-token prompts, thinking on, T 1.0): later-turn time to first
  token 1.59 s cold, 0.19 s reused; exact repeat 0.013 s.
- C1 decode, 4K/16K prompts: greedy 268/238 tok/s, sampled 235/219 (previous engine 263/234 and 202/184);
  `--parallel 8` serves C1 at the same speed. Per-request decode at `--parallel 8`, greedy 4K: C2 139, C4 114,
  C8 62 tok/s (sampled 128, 86, 53): DFlash verify rows are compute-bound FP32 GEMV rows (about 0.85 ms per extra
  row), so batches gain less than serial batching would suggest.
- Eight and twelve concurrent thinking-on requests with `max_tokens` 200,000 at `--parallel 8`: all 200; eight
  ran at once, the rest queued.
- Robustness: three queued requests at `--parallel 1`, client disconnects during a 30K-token prefill and during
  decode, then a greedy reply equal to the reference; `/stats`, `/metrics` and `/v1/tokenize` respond.

## Validation (source of the 2026-11-15 change, image `cf8c0ff6`, eight V100-SXM2-32GB)

- Native kernel tests (`tests/cuda/test_sm70_gemma4_kernels.py`, 55 passed): row tiles beyond 16 rows equal
  16-row calls bitwise (W4, BF16, INT8); several attention segments in one launch equal per-stream launches
  bitwise (linear, ring, 1/5/300 rows, D 128/256/512); the wide tensor-core projections and grouped experts
  match FP64 references.
- Server, `--parallel 8`/`32`, context 32,768, greedy chat (5 short prompts, 2 × 3,026-token prompts, 200 tokens):
  C1 replies identical to the `0913d721` engine; C3 (bitwise path) identical; C7/C8/C20 identical at 8
  streams (`--parallel 8`); C8/C32 at `--parallel 32` diverge as described above. Sampled (seeded) C7 and C16
  replies equal their sampled C1 replies. Eight streaming requests (half cancelled during prefill, half after
  three deltas) left no live lanes, and a following greedy C7 matched.
