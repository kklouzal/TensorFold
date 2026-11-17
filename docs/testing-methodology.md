# Testing methodology

This is the authoritative guide for testing a change to this fork. It applies
to every agent and contributor. It summarizes the procedures and links to the
detailed references; it does not replace the governing policy in `CLAUDE.md`
and `AGENTS.md`, the [contribution guide](../CONTRIBUTING.md) or a model's
[recipe](recipes/README.md).

A change is judged on three independent questions, each with its own tool:

| Question | Tool | Decides on |
| --- | --- | --- |
| Is it faster? | [Serving matrix](#1-performance-the-serving-matrix) (`tools/matrix_benchmark.py`) | Prefill and decode tok/s over the whole matrix, server session as the unit |
| Is it at least as accurate? | [Teacher-forced quality harness](#2-quality-outputs-may-change-accuracy-may-not-regress) (`tools/quality_eval/quality_eval.py`) and kernel error against FP64 | Per-token NLL in nats/token with a predeclared margin |
| Is it correct? | [Unit, native GPU and server checks](#3-correctness-and-determinism) | Contracts, determinism, failure paths |

Record every command, source/image identity, result and unrun check in the
task's ledger (see [Evidence](#5-evidence-and-ledgers)). Unrun checks stay
pending; a result for one image, source tree or host does not qualify another.

## 1. Performance: the serving matrix

Every throughput optimization is judged on the **whole serving matrix**, never
on one prompt length, one concurrency level or one microbenchmark. The matrix is
concurrency **C = 1, 2, 4, 8, 16, 24, 32** by rendered prompt length **4K, 8K,
16K, 32K and 64K** tokens (4,096 to 65,536 exactly), each in two modes:

- `greedy` (`temperature: 0`; on Gemma 4 SM70 the DFlash speculative path);
- `sampled` (the server's default sampling with a fixed per-request seed; on
  Gemma 4 SM70 serial decoding).

Two metrics are scored per cell: `prefill_agg_tps` (all prompt tokens / time to
the last request's first token) and `decode_median_tps` (median request decode
rate). Every request is a streamed chat completion with thinking off,
`ignore_eos` and 512 output tokens; every prompt is unique, so no prefix-cache
hit can enter a measurement. Cells the server cannot run are recorded as
`pending` (C above the served lanes) or `infeasible` (context or KV budget),
never silently dropped. Full details:
[Serving matrix](benchmarking.md#serving-matrix-prefill-and-decode-across-concurrency-and-prompt-length).

### Procedure

```bash
# once per corpus/plan version (plan needs transformers: run it in the serving image)
python3 tools/matrix_benchmark.py corpus --commit COMMIT --tokenizer MODEL_DIR --out corpus.txt
python tools/matrix_benchmark.py plan --corpus corpus.txt --tokenizer MODEL_DIR \
  --template CHAT_TEMPLATE.jinja --out plan.json --jobs 16
# one server process per session; restart the server between sessions
python3 tools/matrix_benchmark.py run --model NAME --plan plan.json --corpus corpus.txt \
  --label SOURCE-AND-CONFIG --source-dir SERVED_SRC --out SESSION.json --reps 5
python3 tools/matrix_benchmark.py summary A1.json,A2.json
python3 tools/matrix_benchmark.py compare A1.json,A2.json B1.json,B2.json \
  --calibration AA1.json,AA2.json --json comparison.json
```

The Gemma 4 SM70 matrix uses frozen inputs in `/home/kklouzal/build-cache/matrix`
(`corpus-0913d721.txt`, `plan-v1.json`) and the session script
`tf-session.sh` there (server at `--context 66048`, the smallest context that
admits a 65,536-token prompt plus 512 output tokens; the context is part of the
REGION, so compare only equal contexts).

### Required design

- **Unit = one server session (process).** Between-process variation (warm
  state, allocation layout, clocks) is about 1% on prefill, larger than the
  within-session repeat noise. Run **at least two sessions per arm**, in
  alternating and then reversed order (A B B A, or A B then B A), each with its
  own warmup and the **same repeats** (default five).
- Pass any A/A sessions of an identical configuration (for example two earlier
  sessions of the reference) as `--calibration` groups. They add degrees of
  freedom to the between-session variance estimate. A single session per arm
  without calibration has 0 df and is always `uncertain`.
- Same host state for both arms: never change power cap, clocks or CPU governor
  (the V100 host stays at 150 W and `powersave`); no other GPU process.
- Identical quality settings in both arms; a change that alters outputs also
  needs the [quality harness](#2-quality-outputs-may-change-accuracy-may-not-regress).

### Scoring rule (predeclared)

`compare` implements the rule; it is fixed before any candidate is scored.

1. Per cell and session: the mean log value over the repeats all scored
   sessions ran. Cell ratio = candidate / reference geometric mean of session
   means.
2. Interval: t with se² = τ²(1/m_ref + 1/m_cand), τ² the pooled within-group
   variance of session means over both arms and every calibration group, df =
   Σ(sessions − 1) per group.
3. Per cell the deciding interval is **familywise**: level 1 − 0.05/k over the k
   compared (cell, metric) pairs (Bonferroni). **gain** if its lower bound > 1,
   **regression** if its upper bound < 1, else **uncertain**. The per-cell 95%
   interval is only a screen (`!`: more sessions needed).
4. **Aggregate** per metric: geometric mean of the cell ratios, with a 95%
   interval formed from per-session scores (the mean of a session's cell means),
   so a process that is slow everywhere counts once. The worst cell is reported.
5. **Verdict**: any cell or aggregate regression **rejects** the candidate,
   whatever else improved (gains never offset a regression; a REGION-specific
   selection is a separate, explicit decision). **superior** needs at least one
   cell or aggregate gain, no regression and every reference-measured cell
   present. Otherwise **incomplete** (missing cells) or **uncertain** (neither a
   win nor a tie: add sessions, or keep the reference).

No gain threshold is applied: every established gain counts. "Uncertain" is
never reported as "equivalent".

### Speculative depth sweeps (DFlash rows, MTP depth)

Draft depth is chosen on the matrix, per REGION:

1. Sweep ascending from depth 1 (DFlash `--draft-rows` = depth + 1) up to 5,
   scoring each depth against its predecessor with the rule above.
2. Stop at the first depth whose aggregate is **slower** (a regression) than its
   predecessor; the predecessor wins. Uncertain is not slower: continue.
3. Persist the winner with its evidence (Gemma 4 SM70:
   `/home/kklouzal/build-cache/matrix/drafter-depth-winner.json`). Later rounds
   re-test the winner against winner ± 1.

Greedy DFlash replies must remain token-identical to serial decoding where the
contract promises it (2–8 rows on Gemma 4 SM70); otherwise the quality harness
decides. SC117 MTP depth may be chosen per active-request band from a fixed table
fixed by profiling (user policy); its 2026-11-14 profile, swept live with
`POST /v1/runtime {"proposal_limit": d}` in descending order 4..0 per
concurrency (64-token prompts, 256 greedy output tokens; not yet repeated on the
serving matrix), found depth 4 best or tied at C1–C24, so the table is 4 everywhere
(MTP4, confidence stop 0.7). Re-profile when graph keys or round costs change.

### Profiling

- Nsight Systems 2025.3 with its default CUPTI 13.1 **cannot trace Volta
  (SM70) kernels** (`CUPTI_ERROR_INVALID_DEVICE`); selecting its bundled CUPTI
  12.9 through a profiling-only search directory does work
  ([V100 validation](v100-flashnext-validation.md)), and Nsight Compute 2025.2.1
  works on V100. The routine tool is `torch.profiler`: inject it into the server process
  (`/home/kklouzal/build-cache/cc-perf-20261114/inject/sitecustomize.py`,
  started and stopped by real-time signals) or drive the engine directly as
  `task-artifacts/claude-perf-20261114/matrix_profile.py` does (all ranks, real
  plan prompts). A signal-started profiler only reaches ranks currently running
  Python. `py-spy` needs `SYS_PTRACE` for host-side stacks. Profiles are
  diagnostics, never acceptance evidence.
- Rank kernel time by cost share per matrix region (prefill vs decode, short vs
  long prompts) and treat the ranking as a list of hypotheses: only the matrix
  decides.

## 2. Quality: outputs may change, accuracy may not regress

Bitwise reproduction of reference outputs is **not** an acceptance gate. A
faster change that alters outputs is acceptable, and may be better, if it is not
less accurate. Semantic contracts (sampling, stop, confidence and acceptance
rules) remain binding.

### Teacher-forced NLL harness

`tools/quality_eval/quality_eval.py` scores a frozen token corpus through the
served model by teacher forcing: every document's prompt tokens through
prompt-pass (prefill) rows and its continuation through teacher-forced
decode/verify rows. The server must run with `TENSORFOLD_PRIVATE_SCORE=1`
(endpoint `POST /v1/tensorfold/score`); `--backend vllm` scores a vLLM
reference through `prompt_logprobs`.

```bash
# once per corpus version and tokenizer (needs transformers + tensorfold: serving image)
python tools/quality_eval/quality_eval.py tokenize --corpus tools/quality_eval/corpus-v1.json \
  --tokenizer MODEL_DIR [--chat-template SERVED_TEMPLATE] --out corpus-v1.MODEL.tokens.json
python3 tools/quality_eval/quality_eval.py score --tokens TOKENS.json --out A.json --label REF --repeat 2
python3 tools/quality_eval/quality_eval.py score --tokens TOKENS.json --out B.json --label CAND --repeat 2
python3 tools/quality_eval/quality_eval.py compare A.json B.json --out quality.json
```

Documents with `messages` are chat documents: the conversation before the final
assistant message is rendered by the chat template with thinking off and the
final reply is scored; `--chat-template` uses the served template file instead of
the checkpoint's (its SHA-256 is recorded in the token file). Documents with
`text` are tokenized raw (no BOS); their last min(192, n/3) tokens are the
continuation. A document's `kind` is its stratum in `compare`'s per-kind means.
Both runs must score the same token file (checked by SHA-256).

Frozen corpora:

- `tools/quality_eval/corpus-v1.json` (16 documents, chat and prose), tokenized
  as `corpus-v1.tokens.json` (SC117 tokenizer) and `corpus-v1.gemma4.tokens.json`
  (Gemma 4 tokenizer). About 2,600 decode positions: too few to resolve a
  numerics change near the margin (a Gemma 4 SM70 candidate scored decode
  d −0.0049, CI [−0.0132, +0.0022], `inconclusive`).
- `tools/quality_eval/corpus-v2.gemma4.json` (Gemma 4; sha256 `b38fa75f…`),
  tokenized as `corpus-v2.gemma4.tokens.json` (sha256 `94f299f4…`; AEON Gemma 4
  26B-A4B tokenizer, served Google template `gemma4-chat-template-google-4d7ae498.jinja`).
  268 documents, 239,608 prompt-pass and 101,168 decode positions:

  | Stratum (`kind`) | Docs | Decode positions | Source |
  | --- | ---: | ---: | --- |
  | `chat-general` / `chat-math` / `chat-code` | 32 / 30 / 40 | 12,543 / 16,867 / 26,588 | self-written; some system prompts and two-turn chats |
  | `chat-multilingual` | 34 | 13,905 | self-written, 20 languages |
  | `chat-repo` / `chat-repo-long` | 26 / 10 | 9,125 / 3,870 | "continue this file" over repository excerpts; long prompts 4K–15K tokens |
  | `text-prose` / `text-math` / `text-multilingual` | 26 / 14 / 22 | 4,962 / 2,674 / 4,156 | self-written raw text |
  | `text-repo-doc(-long)` / `text-repo-code(-long)` | 16 / 18 | 3,040 / 3,438 | raw repository excerpts, five of 12K–15K tokens |

  Self-written documents were written for this repository; repository excerpts
  come from TensorFold's own Apache-2.0 files at commit `5a0d38b6`
  (third-party and vendored files excluded). Tokenizing takes about 45 CPU
  minutes (each chat render costs ~8 s); do it once per corpus version.
- `tools/quality_eval/corpus-v2.sc117.tokens.json`: the same v2 documents
  (`corpus-v2.gemma4.json`, recorded as `corpus-v2.json`, sha256 `b38fa75f…`)
  tokenized with the SC117 tokenizer (sha256 `7b6b3e0e…`) and its checkpoint chat
  template; 268 documents, 226,235 prompt-pass and 99,605 continuation positions
  (18 CPU minutes in image `cf8c0ff6`).

**SC117 prompt rows: judge the prefill path on raw-text documents.** Inside a
chat user turn SC117 puts ~99% of each row's probability on `<|im_end|>`, so a
chat document's prompt-row NLL (5–10 nats/token on the `chat-repo(-long)`
strata) measures only the remaining ~1% tail and swings by ±0.1–0.4
nats/token per document under any numerics change; the 96 raw-text documents
carry the prefill evidence (`compare --kinds text`; v2 SC117: 107,932 rows,
block SE ~0.0005–0.0007). The decode path is judged on all documents (chat
continuations are assistant replies). A prefill-path run may cap the decode rows
with `score --continuation-limit 32` (8,308 decode rows; ~6.5 minutes per run
including the 268 prompt passes); `compare` refuses runs with different caps.

Size rationale for v2: in the v1 A/B the document-blocked SE equalled the naive
per-token SE (0.0041 vs 0.0040), so decode differences behave as nearly
independent tokens with rare 1–3.5 nat outliers (MoE routing flips); the 95%
half-width is about 1.96·sd/√N. With the observed per-token sd of d (0.21 overall;
by kind 0.03 chat, 0.04 multilingual, 0.16 math, 0.27 prose, 0.31 code), v2's
101K decode positions give an expected half-width of about **±0.0013–0.0014
nats/token** for a change of that kind (v1: ±0.008), well inside M = 0.005.
A change with larger per-token differences widens the interval in proportion
to its sd; the bootstrap interval, not this estimate, decides.

Run time (TP8 Gemma 4 SM70, `--parallel 1`, DFlash draft rows 3): about 13
minutes per repeat, estimated from v1's rates (~160 decode positions/s, ~1,900
prompt tokens/s), so `--repeat 2` fits in under 30 minutes plus server start.

`score` reports mean NLL (nats/token) and perplexity separately for the
**prefill** and **decode** paths; `--repeat N` also checks that repeats are
bitwise identical. `compare` computes paired per-token differences
d = NLL(B) − NLL(A) per path, with a 95% interval from a **4,000-resample
document-block bootstrap** (tokens inside a document are correlated), top-1
agreement and a KL lower bound. The margin **M = 0.005 nats/token** (about 0.5%
perplexity) is predeclared; never move it after seeing results.

| Verdict | Condition on the 95% interval [lo, hi] of d | Acceptable? |
| --- | --- | --- |
| `identical` | every token scored bitwise equal | yes |
| `better` | hi < 0 | yes |
| `equivalent` | −M < lo ≤ 0 and hi < M | yes |
| `worse-within-margin` | lo > 0 and hi ≤ M | **no**: an established loss, however small |
| `worse` | lo > 0 and hi > M | **no** |
| `inconclusive` | otherwise | **no**: score more tokens or keep the reference |

### Kernel error against FP64

A numerics-changing kernel (new GEMM/attention/reduction order, precision,
quantization path) is checked against a float64 reference of the same
mathematical operation on representative and adverse inputs, comparing its error
to the error of the kernel it replaces: the new kernel must not be less accurate.
Tests under `tests/cuda/` follow this pattern (for example teacher-score shard
combination within 1e-5 of a float64 `log_softmax`). Arbitrary tolerances fitted
to the result are not accepted.

### Golden outputs and determinism are change detectors

Bitwise golden checks tell you **whether** outputs changed, not whether a
numerics-changing change is acceptable: a difference sends the change to the
quality harness; it does not reject it by itself. A change that claims to be
execution-neutral (host-only, refactoring, scheduling) must stay bitwise
`IDENTICAL`; a difference there is a defect. The SC117 greedy oracle is
`/home/kklouzal/build-cache/cc-perf-20261114/golden.py`
(`golden.py record|check FILE`, `checkcN` serves N concurrent copies of every
prompt; six short prompts, 160 greedy tokens; references `golden-*.json` beside
it). When a numerics change is accepted, record a new reference. The GB10
campaign uses a teacher-forced FP32 reference instead
(`tools/campaign_golden.py`, gated by `tools/campaign_kl.py`).

Determinism contracts are binding where promised:

- **repeat**: the same request twice gives identical tokens/scores
  (`quality_eval.py score --repeat 2` prints `repeats bitwise identical`);
- **slot**: a request's result does not depend on which lane/slot serves it or
  what runs beside it (golden `checkcN` runs N concurrent copies);
- **cold vs cached**: a prefix-cache hit should give the same result as a cold
  prefill. On SC117 an exact repeat of a prompt joins from the kept end of its
  first request, so it replies identically (2026-11-16) while that end is kept
  (drafting prompts; not DFlash2/DSpark or a retention policy; dropped by eviction
  or when another prompt resumes its point in place). The long-context oracle
  `/home/kklouzal/build-cache/cc-perf-20261114/golden_long.py` (reference
  `golden-long3.json` beside it, recorded after the 4605f899 NCCL prompt sums and
  round 0 reproduced by three fresh server processes at a8d0a03b; `golden-long1/2.json` are
  superseded) sends each of
  10 prompts (336–6,190 tokens, text and chat) cold and then as its repeat and
  requires cold == repeat == reference; each round has its own prompt salts, so a
  round is checked once per server process (`check FILE --round K`, K = 0 or 1).
  The same source, configuration and requests must reply identically in a fresh
  server process; keep the oracle's full output (a truncated `DIFFER` list once
  read as cross-process nondeterminism).
  Still open: a prompt that extends a kept point (a next turn, a shared system
  block) is cut into different prompt pieces than its cold prefill, so it may
  differ from a cold prefill of the same prompt, and concurrent replies differ from
  C1 replies (batch numerics; golden `checkcN`). Serving-matrix prompts are unique,
  so no cache hit enters a measurement.

A failure of a promised determinism contract is a defect, not noise.

## 3. Correctness and determinism

- **CPU suites**: `python -m pytest tests/<affected files> -q`,
  `python -m ruff check <changed files>` and `git diff --check`, in the pinned
  image with no GPU (on the V100 host: image `cf8c0ff6…`, `/opt/venv/bin/python`,
  `PYTHONPATH=src`, read-only repository mount, `NVIDIA_VISIBLE_DEVICES=void`).
  Run the affected maintained tests; run the full `tests` tree for cross-cutting
  changes. Container-side ruff/pytest are the reference (the host may lack them).
- **Native GPU tests** (`tests/cuda/`; collected only when CUDA is available):
  run in the pinned CUDA image on the target GPUs, holding the GPU lock (below),
  with the changed source overlaid read-only on the image's installed
  `tensorfold` package and `tests/` mounted (a working runner:
  `/home/kklouzal/build-cache/cc-perf-20261114/pause/slotfix/gpu.sh ROOT OVLFILE
  GPU pytest-args…`, which runs `pytest -q -p no:cacheprovider` in image
  `cf8c0ff6…` on one GPU). SM70 kernel tests
  compare against FP64 references (`test_sm70_gemma4_kernels.py`,
  `test_sm70_prompt_attention.py`, `test_sm70_shared_ffn.py`, …).
- **Known pre-existing failures**: the CPU tree and some GPU files fail in this
  image for environmental reasons (for example missing PIL, retained-catalog
  fixtures). Run the same files on a `git archive` snapshot of the base commit
  and compare outcome sets per test: pre-existing failures are recorded in the
  ledger, never counted as passes; any new failure blocks.
- Test boundaries and failure paths, not only the normal path. New tests assert
  contracts, not implementation internals.

### GPU sharing

The eight V100s are shared by several agents. Every GPU use (servers, GPU
tests, profiling, benchmarks) runs under

```bash
flock /home/kklouzal/build-cache/cc-perf-20261114/gpu.lock <command>
```

with each hold under about **45 minutes** (split long work into sessions; one
matrix session fits). Scripts meant to run under the lock say so in their
header ("caller holds gpu.lock"). Remove your containers before releasing the lock. Another
process on a serving GPU changes memory admission and invalidates timings.
Never change GPU power caps, clocks or the CPU governor.

### Server launch recipes

- **Gemma 4 26B-A4B** on eight V100s (TP8/ETP4, DFlash, `TF_SMALL_GATHER_BYTES=262144`):
  [Gemma 4 recipe](recipes/gemma-4.md#eight-v100s-tp8etp4); the matrix session
  script `/home/kklouzal/build-cache/matrix/tf-session.sh` launches the same
  configuration from a mounted source tree at `--context 66048`.
- **SC117** (Qwen3.8 Flash Next IQ3_S target with the shared Q8_0 MTP drafter)
  on eight V100s: [V100 candidate launch](../deploy/v100/README.md#candidate-launch)
  (`TF_EXPERT_TENSOR_PARALLEL_SIZE=4` selects ETP4; `TF_DRAFTS=4`), the
  [Flash Next recipe](recipes/qwen3.8-flash-next.md#cuda)
  (`--expert-tensor-parallel-size 4`) and the
  [V100 one-session phase matrix](benchmarking.md#v100-one-session-phase-matrix).
  The performance push's REGION is TP8/ETP4, `--mtp-drafts 4 --mtp-confidence
  0.7`, INT8 KV, 24 × 262,144-token windows (launch script
  `/home/kklouzal/build-cache/cc-perf-20261114/run.sh`).
- Host policy for both: TP8 for non-expert weights and ETP4 for experts, 150 W
  GPU cap, CPU governor `powersave`. Do not change them to win a benchmark.

## 4. Before you call a change done

1. Correctness: affected CPU and GPU tests pass; no new failures against base.
2. Quality: outputs unchanged (golden), or the quality harness verdict is
   `identical`, `better` or `equivalent` on every path.
3. Performance (optimizations): the serving-matrix verdict is `superior`, with
   at least two sessions per arm, against the named current reference.
4. Docs: README, recipe, CLI help and this file updated when behavior, defaults,
   flags or qualified results change.
5. Ledger: commands, identities, results, unrun checks and blockers recorded.

## 5. Evidence and ledgers

- **Task ledgers**: one append-only ledger per task under `task-artifacts/<task>/`
  (for the V100 throughput work,
  `task-artifacts/claude-perf-20261114/LEDGER.md`; others beside it). Record objectives,
  REGIONs, predeclared rules, commands, source/image identities, results,
  dispositions, unrun checks and blockers. `task-artifacts/` is private
  experiment state: maintained tests must not depend on it.
- **Raw measurements**: `/home/kklouzal/build-cache/` (for example
  `matrix/` for serving-matrix sessions, comparisons and the depth winner;
  `cc-perf-20261114/` for SC117 performance work and the GPU lock). Keep raw
  result JSON; never edit measurements (record any metadata rewrite in the
  ledger).
- **Maintained evidence**: qualified results with their source, image,
  hardware, workload and quality scope go to the README, the affected recipe and
  [optimization validation](optimization-validation.md); method references live
  in [benchmarking](benchmarking.md).
