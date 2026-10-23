# Gemma4 native NVFP4 on GB10, 2026-10-18

## Adaptive vocabulary-head candidate, unqualified

This source candidate prepares the existing R4 kernel for two through four
rows, R8 for five through eight, and R16 for larger calls. Singleton calls
retain the original head; zero rows retain the fixed-R16 native refusal path.
The original contiguous BF16[262144,2816] weight and GB10SM12.1 eligibility are
unchanged, with no capability discovery during projection. Other shapes,
representations and architectures keep their original implementation.

Each active row retains the same ordered FP32 FMA chain, XOR reduction and
BF16 output rounding. Target and BF16 DFlash borrowers still apply the existing
softcap exactly once. Prompt methods, the separate FP8 companion head, sampler,
native CUDA/C++ sources and fixed `prepare_head` comparators are unchanged.
The retained R8/R16 qualification tool recognizes the exact new policy only to
recover its original weight owner; its benchmark arms remain fixed R8/R16.

The prepared policy adds two callable objects relative to the fixed head and a
row-count dispatch on each projection. Weight/output tensor geometry, graph
keys and native recipes are unchanged. Preparation, capture, dispatch and
compiler-cold effects remain part of the required complete-model comparison.
The policy is bound before target/DFlash graph preparation and belongs to that
engine lifetime; it is not a mutable serving flag.

Pinned source checks passed 70 cases on the protected AMD64 Torch9026/CUDA13.4
SDK with GPU hidden: 24 policy controls, 45 retained qualifier controls and one
actual FD metadata case. Complete source and imported tool identities were
verified. A separate local SM120/8+1-thread functional gate passed eight cases:
seven direct-policy tests plus the FD control. Complete raw bits matched fixed
R16 and original singleton projections, including partition/permutation,
softcap, signed-zero/extreme inputs, CUDA replay and the zero-row refusal.
Actual B16/FD mapped-object proofs, foundation stability, device settlement and
owner retirement passed. The original host attempt with 17 missing-Torch
failures remains recorded separately; it is not relabeled as a pass.

Both gates used the pinned509 source checkout. They do not qualify installed
ARM64/public SM121 selection. Serial/drafted greedy and sampled full-model
identity, other request/cache regions, resource/cold-start/capture budgets and
comparative throughput remain pending. No benchmark result is claimed.
The historical fixed-R16 gain and unresolved R8/R16 ranking below remain their
own evidence; no adaptive-policy gain or production selection is claimed.

## Sampled proposal CPU repair, 2026-10-21

The original installed image `5be0e0fa` spent substantial sampled decoding time repairing tied BF16 draft
scores on the CPU. A profile of that immutable image observed 591 full-vocabulary proposal repairs out of
644 possible rows; canonical candidate ordering consumed 6.75 seconds of CPU self time across warmup and
the profiled request. Profiler timings diagnose execution and do not establish throughput.

Candidate source `cda5d48e`, verification image `sha256:16bf28a03ea568c63334b551696a9d4707a1727a6d17e58a5b7b48e74bf44599`,
passed 495 installed ARM64/CUDA tests without skips. Qualification tool source `e2abb4eb` then compared twelve
combinations of canonical proposal support (256 or request top-k64), extra candidate guard (1/8/32), and
complete-row repair (full lexsort or exact partial selection). The selected source policy uses bounded
support, 32 extra scores, and partial repair. Every combination retained the same normalized q bytes,
complete generated sequences, and round/acceptance counts. A plateau wider than the guard still repairs
from the complete immutable row; validation and probability normalization remain intact.

The screen used the original AEON NVFP4 and BF16 DFlash revisions recorded below, CUDA13.4.3/Torch nightly cu134,
ten fast GB10 CPU cores, VBoost0, context262144, BF16 target head/KV, vision loaded, five verification rows,
one lane, and the pinned Google template. Global thinking stayed enabled; these requests explicitly disabled
thinking and used T1/top-k64/top-p0.95/seed7, 64 output tokens, and ignored EOS. Prompt lengths were48 and8451.
Both voice containers remained resident. All146 cold/proof/warm/timed requests passed. Each of two mirrored
timing positions per policy followed a full warm request; proof observers were removed during timing.

| Workload | Original q256/G1/full-sort decode tok/s | q64/G32/partial decode tok/s |
| --- | --- | --- |
| Code, 48-token prompt | 34.84 | 70.23 |
| Chat history, 8,451-token prompt | 20.42 | 44.85 |

These are arithmetic means of two observations in a finite screen, not confidence bounds or a universal
hardware rate. The selected arm required no draft full-row repairs across248 proposal rows, versus217
repairs with the incumbent; two
target repairs remained in the history request. Partial selection's synthetic all-tied contiguous-ID rows
can cost more than lexsort; this screen qualifies the stated model/workloads and makes no universal repair
speed claim.

The selected source `259b4238` and verification image
`sha256:97d6defcbdd69f3e38a9a142d9514c23f6cf46d7176ee56d35f8d2785578a0f7`
passed the subsequent 256-token holdout: two cold, eight proof, two warm and 48 timed requests, with no failures.
Each workload used six seeds (13/31/47/83/101/127), two observations per policy per seed, and alternating
ABBA/BAAB orders. Prompt prefixes and compiled kernels were warm; identical completions were not prewarmed
between timed positions. Both policies retained the same full output IDs and acceptance counts in each
block. Probability-byte proof remained the separate seed7 check. The original container was restored and
the owned test container retired; both voice containers were preserved.

| Workload | Original mean complete wall, seconds | Selected mean complete wall, seconds | Pooled decode tok/s, original → selected |
| --- | --- | --- | --- |
| Code, 48-token prompt | 6.622005 | 2.912488 | 38.515 → 87.596 |
| Chat history, 8,451-token prompt | 10.939156 | 4.743466 | 23.325 → 53.833 |

All six paired-block mean latencies improved in each region. The predeclared sign gate's familywise bound
is 2/64, conditional on each regional null assigning at most one-half win probability given prior blocks;
sequential measurements do not establish that assumption, and correlated drift can invalidate the bound.
Diagnostic geometric selected/original wall ratios were 0.439975 and 0.433653. Unadjusted 95% paired-log
t intervals were [0.430454, 0.449707] and [0.431091, 0.436231], assuming independent identically normal block
log effects. This reporting method was specified after collection but before the analyzer owner accessed
the results; it did not alter the predeclared selection gate. Rates use 255 decode steps per request;
the objective uses all 256 tokens and settled complete-request wall time. No universal rate or 200 tok/s
claim follows. Runtime image
`sha256:244312373625cacc8bd82af1372b058eec8af5d19ee217a6bc628b62b027896d`
is deployed and passed all 12 live checks: nine complete 256-token cold/warm SSE requests with native-reference
IDs, counting/template/context parity, default reasoning and answer separation, and a named tool call with
synthetic-result continuation. Prefill progress is validated when emitted; a fully cached prompt needs no
prefill pass or progress frame. Global thinking remains enabled with unlimited server thinking budget,
both speech containers are unchanged, and old model containers were retired. Two earlier rollout attempts
were restored after defective deployment-check assumptions; their failures and the independently justified
metadata/full-cache corrections remain preserved in the
[deployment receipt](../deploy/gb10/deployment-gemma4-throughput-20261021.json).
The user canceled further launch tuning;
the existing five-row, BF16 drafter and R16 head launch remains selected. Prepared launch-comparison
packets were not executed and establish no additional performance or correctness result.

Raw traces, complete reports, source freezes and failures remain in the task evidence ledger.

## Original deployment, 2026-10-18

The installed TensorFold runtime serves the original `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4` revision `90c8601b1a439497bc15d8fba9e52990b45ea0cb`, with BF16 `z-lab/gemma-4-26B-A4B-it-DFlash` revision `77d4202772dfe50b2396ec7bac9cfffc7b9e7057`. All fourteen target files passed their publisher hash checks. Checkpoint Python was not executed. Original FP4 codes, block scales and global calibrations remain authoritative; BF16 embedding/head, router and vision weights are retained. CUDA loading packs projections on the GPU.

Runtime image `sha256:2b6a6d7cc496d518553d824eec7fb5cd955e20f8b87a8f2861952de79dc072b6` and verification image `sha256:ff43aee142f4d885395a69f4213ffa0945c2fddea61ef83ee5ea6abfe3069e6b` share application source capture `0752d5a6+aeon.22d1450991182980` and wheel SHA256 `863d7a5e48e7e91d31566012c377f72a3ff7246f5f5de5acaf4867dd28372f28`. All 27 affected installed package files matched that capture. Later README/recipe and receipt publication updates leave execution unchanged. An excess EOF blank line in Gemma value parsing was also removed; module bytecode matched the image capture at all three Python optimization levels. The [machine-readable receipt](../deploy/gb10/deployment-gemma4-nvfp4-20261018.json) retains the source hashes, pinned foundation, raw evidence identities and limitations.

The protected ARM64 foundation uses NVIDIA CUDA 13.4.3, official Torch `2.16.0.dev20261014+cu134` and Triton 3.9.0. Foundation audits and dependency checks passed; no Torch build or vendor patch was used. The profile serves one request lane, native context 262,144, BF16 KV, vision, five verification rows and four proposals. Thinking is explicitly off by default and remains request-selectable. The Harness compatibility entrypoint and model ID `gemma4-26b` remain on `http://192.168.1.50:8888/v1`. Docker RAM/CPU quotas are absent, all 20 CPUs were granted before Torch initialized, the host reserve is 8 GiB, and the optional VRAM reserve is zero. The cached-extension start reported a 19.1-second model load.

Installed-package checks passed 461 CPU cases and all 50 CUDA cases. Two Apple-only thinking-note tests were skipped on this host. Full-model checks passed greedy 512-token and sampled 64-token serial/DFlash identity, exact SSE greeting/usage/DONE, generation-free Harness/native count parity, image/video color fixtures and a schema-enforced function call preserving boolean/null types. The short greeting counts 19 tokens; a 6,013-token history is retained without the checkpoint's saved 4,096-token truncation. Counting and generation use an unpadded, untruncated tokenizer and the server retains context admission.

The installed HTTP workload was the report's LRU-cache coding prompt, maximum 512 output tokens, temperature 0, seed 7, thinking off, sequential serial/DFlash/DFlash/serial. All four replies contained the same 512 token IDs. The first prompt was uncached; later requests reused 48/48 prompt tokens. Engine decode rates exclude the initial token:

| Mode | Decode tok/s, observed runs |
| --- | --- |
| Serial | 41.16, 41.37 |
| DFlash, five verification rows | 64.49, 74.41 |

Both drafted observations are retained. This four-request check is not a confidence bound or a general throughput guarantee.

A separate source campaign at context 4,096 compared original and exact R2/4/8/16 BF16 head weight reuse. It used 128-token interval-merging code and 32-token sky-explanation replies, greedy/thinking-off, complete warmed prefixes, ten balanced forward/reverse blocks per workload and 200 complete matching outputs. R16 observed median full-engine request rates 75.64 code and 54.63 chat tok/s, versus 39.78 and 30.02 with the original head. Its block-paired latency improved in all ten blocks of each workload; the predeclared conditional sign analysis has familywise p upper 0.0390625 across 20 comparisons. No other GPU workload or build ran concurrently; 330 memory samples retained at least 100.27 GiB MemAvailable. These source observations differ from the installed HTTP workload above.

R8 versus R16 remains **ranking verification-blocked**, rather than equivalent or inferior: median paired R8/R16 latency ratios were 1.00134 code and 1.00922 chat, with R16 faster in 6/10 and 7/10 blocks. R16 is selected as a verified improvement over the original head, without a unique-optimum claim. The exploratory 3/5/11/16-row comparison favored five on both tested workloads; it is not independent proof of universal depth superiority. Different GPUs, concurrent lanes, larger/longer prompts, other models and cross-engine numerical identity require separate qualification. Native CT arithmetic and retained MLX arithmetic remain separate contracts.

## Chat template: Google's current template

The GB10 profile now renders prompts with Google's current Gemma 4 template rather than the checkpoint's own copy. AEON's `chat_template.jinja` (SHA256 `2dfbfc7d538912f4ea11d29d85b4e25d7bc26386e53f57529f1d707c28b5828c`) is byte-identical to Google's launch template, commit `47b6801b`. It renders an OpenAI `role: "tool"` message as a `<|turn>tool` turn instead of the trained `<|tool_response>response:NAME{…}<tool_response|>` block, and never renders `reasoning`/`reasoning_content`, so a tool-call turn loses its thought. The card requires that thought to be kept.

[`deploy/gb10/gemma4-chat-template-google-4d7ae498.jinja`](../deploy/gb10/gemma4-chat-template-google-4d7ae498.jinja) holds the unmodified `chat_template.jinja` of `google/gemma-4-26B-A4B-it` at revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (Apache-2.0). Its git blob is `4741bf6e4132ba23a5537f9d6e74e9a6d613d7cd` and its size 18,683 bytes. It was last changed by commit `35b4173cf6211bf5ee1f4c3c8d97cf2a0d89c122` on 2026-07-15 ("null handling, reasoning preservation, turn-tag balance, input validation"). The shipped bytes, SHA256 `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`, were fetched from that pinned revision on 2026-10-18 and matched both its `main` revision and Hugging Face's recorded blob ID. [`compose.gemma4.yaml`](../deploy/gb10/compose.gemma4.yaml) mounts the file read-only and passes it with `--chat-template`. That flag replaces checkpoint templates for rendering, strict-call discovery and Harness counting together; there is no fallback to the checkpoint template. Copy the template beside the compose file when deploying from another directory, because Compose resolves the relative mount from the compose file's directory. Startup prints the selected template's source and SHA256, and the Harness bridge reports the same fingerprint.

Serving differences that change prompts, and therefore outputs:

- A tool result continues the model turn as `<|tool_response>response:NAME{value:<|"|>…<|"|>}<tool_response|>`. `tool_call_id` names it from the matching call, falling back to the message `name`. String results are rendered as `value` strings exactly as Hugging Face renders them; JSON text is not reparsed. After tool results, the generation prompt opens no new turn. With thinking on, it ends with `<|channel>thought\n`, so the reply begins inside the thought channel. With thinking off, it appends nothing.
- `reasoning` or `reasoning_content` of assistant messages after the last user message renders as `<|channel>thought\n…\n<channel|>`. Earlier thoughts are dropped, unless `chat_template_kwargs.preserve_thinking` keeps those of tool-call turns.
- Thinking on renders `<|think|>\n` at the start of the system turn. Thinking off after a user message still ends with the empty `<|channel>thought\n<channel|>` channel.
- Call arguments render JSON `null` instead of Python `None`. String arguments are refused by the template; the server already decodes OpenAI's JSON-string arguments to objects.

Two server changes keep replies correct with this template. When the rendered prompt ends inside an open thought channel, the CUDA server reads the reply from inside that channel: reasoning ends at the first `<channel|>`, and calls after it are parsed. Strict Gemma tools discover the template's null word, `None` or `null`, and constrain and read only that spelling. Checkpoints served with the launch template keep `None`.

Verification status: `tests/test_gemma4_chat_template.py` pins the shipped bytes and the profile's selection. It compares the server's renders with Hugging Face's `render_jinja_template` for ten conversation/thinking cases, asserts the trained tool formats, and checks the CUDA server's parsing of prompt-opened channels and strict `null` calls with a scripted engine. Live GB10 HTTP check (2026-10-18, production flags plus `--chat-template`, source 63415861 lineage, greedy): a thinking-off tool call returned `get_weather{city: Paris, unit: celsius}` and, after the `role: "tool"` result, the answer from it; with thinking on the call carried 83 reasoning tokens, the follow-up rendered the preserved thought before the call and a `<|tool_response>` block (no `<|turn>tool`), reused the 92-token prompt prefix and answered with separated reasoning; a strict call rendered `unit: null`; a plain thinking-on reply kept reasoning and content apart. This is a functional check, not a quality benchmark.

## Integration candidate, 2026-10-19

The current source combines persistent Gemma KV, sliding/global attention and prompt/glue changes, shared device-planned native A4 experts, shared FP8 proposal requantization, and target-forward CUDA graphs. Text decode and verification replay on the lane's persistent state; prefill and visual-block forwards stay eager. Startup captures each width's smallest attention grid only. A larger grid captures on its first request, so first-use capture time remains part of that request's latency. Graph outputs and the private pool remain live with the engine and are included in admission.

The shared Qwen CT chain runner uses its fixed attention buffers for prefill and decoding. Switching conversations moves still-kept prefixes to private buffers only when the stream's memory budget admits the copy beside a resumed conversation's buffers; otherwise those other prefixes are evicted. Resumed prefixes remain live through copy-in, and the runner refuses inconsistent sharing before changing buffers. This eliminates a duplicate full attention cache without weakening prefix-cache independence.

Final integration qualification is pending. Earlier native tests, golden v1s comparisons and engine timings apply to their recorded source revisions and host configuration; they do not qualify this composition or the GB10's new VBoost 0 setting. Required remaining gates include native ARM64/SM12.1 tests, multi-chunk calibrated quality comparisons, deterministic/cache-independent model checks, greedy drafted/serial identity, full-context prefill and decode with admission/peak-memory traces, and repeated HTTP comparisons at the declared sampling settings. The existing production image's known long-prompt KV-growth failure is retained as historical evidence and does not qualify long-context serving.
