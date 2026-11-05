# Gemma4 native NVFP4 on GB10, 2026-10-18

## Private W4A16 head D4 screen

The default-off internal `dflash_head_w4a16` prototype passed source and
installed-wheel430 CPU checks (17 explicit CUDA-only skips in each suite),
all5 native cases, and the actual paired OFF/ON model-law qualification.
The head uses original BF16 checkpoint bytes, existing whole-tensor NVFP4
quantization and `Fp4Linear(act=None)` with FP32 logits/single softcap;
OFF retains the selected private FP8 head. No public CLI or default changes.

The exploratory OFF–ON | ON–OFF screen used NVIDIA GB10 (SM121), source UID
`5d2c930b5e483c93d92f2d30d2dcd53e808b2a1357a1dc339228d5642dfaa485`,
runtime image
`sha256:8e4757e14a4a7c31775e74275f4765767533029136432749a87c0b454b93b0b2`
and verification image
`sha256:e983dda7c32ec10a2101240a960225fa9380cafa79e36ceb96ecc47a70c4a12f`.
Target `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4` revision
`90c8601b1a439497bc15d8fba9e52990b45ea0cb` and drafter
`z-lab/gemma-4-26B-A4B-it-DFlash` revision
`77d4202772dfe50b2396ec7bac9cfffc7b9e7057` were fixed.
Both arms used Shared512 MiB prompt scratch, Delta4/R16, N64, private FP8
transformer, DFlash8/depth4, target graphs, BF16 KV, context131072, vision,
thinking and the owned Google template. Draft graphs, clipping, persistent
cache, trig and native-A8 were off. Each fresh process had its own compatible
compiler-cache copy, six32-token compiler warmups and42 full256-token untimed
warm requests before42 measured requests. The six regions were T0 and T1
(temperature1/top-k64/top-p0.95), each on short-native/code64k/prose64k;
one logical-cold prefix plus six seeded cache hits per region consumed all256
outputs with EOS ignored. Logical cold prefix does not mean cold compiler
cache. Production was stopped, both speech services remained resident, and
all workers used CPU quota0/cpuset10,20 GiB CPU memory,40 GiB CUDA cap and
16 GiB resource floor. All four processes retired safely with valid stopped
receipts; two same-policy and four cross-policy T0-ID checks passed.

Primary throughput is255 post-first outputs divided by the drained interval
from first callback, not an Engine rounded decode timer. The following are
descriptive geometric means of the two paired-half ON/OFF ratios for six-hit
median throughput; the two processes per policy are the replication units,
with requests/seeds correlated (T0 seeds repeat deterministic work).

| Region | Cached primary throughput change |
| --- | ---: |
| T0 code64k | +6.915% |
| T0 prose64k | +2.475% |
| T0 short-native | +2.514% |
| T1 code64k | −2.146% |
| T1 prose64k | +2.780% |
| T1 short-native | +9.182% |

Cold T1 short primary throughput also regressed1.570%, repeated in both
halves, as did the cached T1 code regression. Gains in other regions do not
offset these losses under the initial constrained objective: broad all-request
D4 ON was initially **not selected**. The user subsequently accepted this
specific disclosed trade and directed promotion; see the [current CLI policy](#w4a16-private-head-cli).
This does not establish all-region superiority or erase the losses. Observed
Torch allocated/peak allocated fell309 MiB and reserved/peak reserved636 MiB;
these are allocator observations, not total physical memory savings. The two
ON constructor-and-drain intervals increased3.671s and4.375s against their
paired OFF processes, and CPU memory was higher. Cold TTFT was mixed. The
report retains all cold/hit, work, startup, memory, callback-batch tails and
endpoint drift evidence without correction or pooled regional offsets.
No CI, equivalence, noise dismissal, winner or global-inferiority claim is
made. Narrower request policies and other depths remain credible; prototype,
oracles and all failures are preserved rather than treating this screen as
exhaustive selection.

Private audit evidence under
`/home/kklouzal/audit-artifacts/gemma-aeon26b-optimization-20261102`:
`df-w4a16-report-v1/summary.json` SHA256
`38f2bb292e054164ae92682c20b75ca52028cee572da0c95825283e4af2b1c31`;
`df-w4a16-pilot-final-packet-v1.json` SHA256
`5235ebe506dafbdb1df3d3323a471bdde86f83c71d5aff573721061febf6db50`;
`df-w4a16-actual-report-d13-review-v1.json` SHA256
`77bb26dcfa2296ed6f168e4356f7f6e3ed5335f03f168c3dbeaae548ddd1c248`.
These are local retained receipts, not public benchmark downloads. The accepted
production remains FFD9/58a/7ac/config928d, restored healthy at
2026-11-05T00:21:58.133243316Z; the candidate was not deployed.

## Experimental Delta4 tied-table integration

The CT-only path is default-off (`delta4_head=False`), with bounded internal
`delta4_reuse` choice1/2/4/8/16. Public `--delta4-head` activates the existing path
at fixed reuse16 on Gemma CT NVFP4 MoE CUDA; it does not select an alternative
activation precision. The owner accepted the observed decode/CPU-startup tradeoff
and authorized deployment without making independent confirmation a prerequisite.
CLI wiring leaves backend math unchanged. The 2026-11-03 native CLI cutover is
complete and healthy on port 8888: container
`d1436c99ede93c41cd3b403895c6957b73be7c4717c1427daa781604e6bb8714`,
image `sha256:98f6b664df48775bd92df3e615387904538f4ee8efa7daa1933202d369e6f807`,
source UID `eecb4bbf61d1a8ba9b0d648eb7776a20f844309574ed88b33f4bcd92b992d0de`,
source commit `93ab8ba241da2ce195626332832d135b1552cf46`.
These deployed build identities precede this documentation-only update;
runtime/test bytes remain unchanged. The existing raw R16 CT path and ModelOpt/Sym8 investigation remain
separate and unchanged by default.

The ordinary native command explicitly adds only `--delta4-head`/reuse16,
preserving DFlash8/depth4, context131072, BF16 KV, vision, thinking/aliases and
the owned Google template. DFlash graphs, clipping, shared/persistent caches and
trig remain off. `production-current` selects `deploy-delta4-cli-20261103-v1`;
old container `099563af`/image `8da9dff3` and its volumes remain stopped for rollback.
Both speech services retained their original running instances.

The new source passed 408 CPU checks and 214 installed checks, plus 14 probe CPU
controls and actual native typed Delta4/R16/no-raw-facade/loader-retirement,
capacity and generation/drain checks. Ordinary native CLI shadow testing passed
16 individual API/media actions, 14 directed regressions and independent
observed-active decode/prefill cancellation and recovery. The original weak
short-SSE cancellation-counter assertion remains failed/incomplete; the API
packet's overall flag remains false. After cutover, 14 directed production
tools/bias/reasoning checks and exact plain/SSE math output passed.

The final receipt `delta4-cli-cutover-v1/deployment-final.json` has SHA256
`97f6f6c09ba759d1d1d5cd854b1cbb86f37794f5bea278776b3fef82c38167be`.
The user accepted CPU peak/startup costs and explicitly authorized deployment
before independent confirmation. Performance evidence remains the stopped 07d/f9
exploratory screen; no confirmed shipping-image gain, global fastest result or
exhaustion follows. Earlier source/image results below retain their own scope.

One owned `tensorfold.bf16.delta4.v1` descriptor stores64-word groups in100 base
bytes plus128 raw bytes when the exponent span exceeds15. Encoding preserves
every BF16 bit, including signed zeros and exceptional payloads. The maintained
path imports no protected untracked prototype module. Those original files
remain an independent historical oracle, not a release dependency.

CT loading reads the original tied table on CPU, packs it in bounded512-row
chunks, and uploads only the encoded fields. The caller-owned transfer status
retains every source/destination and original failure until successful retirement
or terminal process teardown. A failed copy/fence forbids any outer CUDA fence,
reader-close or alias-release retry. The successful model owns one packed GPU
table for both lookup and projection, without a `.weight` facade or incumbent
raw allocation. Native lookup returns unscaled BF16 rows; existing target
sqrt(hidden) scaling, ordered FP32 FMA/XOR/final-BF16 projection and single
softcap stay at their original boundaries. Private FP8 DFlash construction uses
the same row quantizer on bounded original-word lookup chunks.

Ordinary engine prefill uses `output_logits=False` and projects only the final
hidden row. Its1024-row chunks still need large embedding lookup. The direct
head's17+ row contract uses bounded native ordered tiles, without a new prompt
TensorCore arithmetic. Packed addresses remain fixed throughout graph lifetime;
State/KV/prefix snapshot formats and cache identity are unchanged.

Admission reserves worst-case valid escapes and the complete CPU encoding/
reader/upload overlap, including UMA accounting. It never assumes the favorable
AEON census from headers. For the pinned90c8601b original tied table, the earlier
standalone census found1,476,395,008 raw bytes and1,158,125,312 packed bytes, a
potential318,269,696-byte payload reduction. That is not a measured integrated
VRAM saving: allocator granularity, graph pools, private FP8 copies, constructor
transients and steady/peak retained memory require actual measurement.

### Qualification checkpoint, 2026-11-03 UTC

These results belong to an immutable dirty application capture, not to a later
documentation-only tree or an unbuilt image:

| Identity | Value |
| --- | --- |
| Application source UID | `f913d5f0974bc5701492361d7e4962d2c84ee3e69cafc049c28f82a978234c71` |
| Runtime image | `43bc1cc12cbf2ed83efe9d0f09563b211013858ba88c49437e7642202d565872` |
| Verification image | `73d349f84f7be5d1dbbd19957a94b41498586d5d3d44b797fafae1321faf8b8e` |
| Hardware/runtime | GB10 SM12.1, CUDA13.4.3 / nvcc13.4.92, PyTorch2.16.0.dev20261014+cu134 |
| Main/draft revisions | AEON `90c8601b1a439497bc15d8fba9e52990b45ea0cb` / DFlash `77d4202772dfe50b2396ec7bac9cfffc7b9e7057` |

The final source checks passed508 CPU tests with15 existing CUDA-only skips;
the final test-only alias-release correction passed45 affected tests. The
installed wheel passed102 CPU tests with immutable test-support provenance.
All16 actual GPU positives passed without skips: literal BF16 words, lookup,
projection reuse and long-row tiles, alignment/refusals, full pinned head,
FP8-copy and graph replay. Eight fresh poisoned contexts independently proved
the native malformed-metadata and lower/upper-ID assertions, retaining actual
owners until exact-container SIGKILL; no CUDA recovery was attempted.

Fresh RAW and explicit Delta4R16 processes then passed full-model equivalence
with raw attention, BF16 target/KV, FP8 DFlash depth4, vision enabled and the
same Google template. Context was configured at131072; the exercised long
prefixes contained65536 tokens, with fixed256-output ignore-EOS generation and
a native short case. Every original unscaled vocabulary word, full private FP8
code and scale fields, teacher hidden/logit/tap/effective-probability manifests,
actual proposal distributions, complete output IDs, counters and cache fields
matched. The candidate had one typed packed authority, no raw-weight facade,
and retired loader owners.

At each recorded stage, process Torch allocated and peak-allocated counters
were319,526,912 bytes (304.724609MiB) lower than RAW; reserved and peak-reserved
counters were314,572,800 bytes (300MiB) lower. The logical table difference was
318,269,696 bytes (303.525635MiB). CPU resident high-water usage increased, while
current RSS varied by stage. These are separate observations, not a net
physical GPU/RAM saving claim. RAW included cold native compilation; the
candidate used a verified private copy of its completed compatible cache.
Concurrent client traffic and a production restart of unknown cause also
confound host-memory and startup/timing comparisons. No speed conclusion is
drawn from these correctness runs.

Those original f913 numerical/resource runs did not establish API/media or shipping
performance. The later eecb/98f6 native CLI/API qualification and user-authorized
cutover are recorded above; earlier counts and image identities are not relabeled.
The parser remains default off, with production explicitly selecting Delta4R16.
Independent shipping-image performance confirmation and further optimization
remain open; no numeric prefill tolerance or decode-regression margin is invented.

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


## Shared global-prompt scratch CLI

Accepted Shared512 is deployed through the maintained native CLI on2026-11-04:
container `ffd9c363af4f59a46139dc7b8d8e6fff900ce233ab1562d3a6d3f743ad59cda5`, source UID `7ac9307b39430169cf60a15db6d1d3ce47f455dd9758f70ae7cabd8bc20309d1`,
runtime image `58a280bb171adefcabdbeabc4fc9f6fc96dea33447a38ea0d4c730b3f00c5315`.
Production became NativeReady/healthy after starting at
`2026-11-04T14:37:39.75804538Z` (observed PID1843019, configuration fingerprint928d1572…);
previous `d1436c99` is retained stopped for rollback. `production-current`
points to `deploy-shared-prompt-cli-20261104-v1`; both speech services remain healthy.

The explicit production options include `--prepare-global-prompt --delta4-head`
with existing reuse16/private FP8 DFlash/depth4, BF16 KV,131072 context,
AEON90c860/DFlash77d420, thinking, vision and owned Google template/settings.
The new flag forwards existing `prepare_global_prompt=True` through native
CLI/server/family validation to Gemma4Engine. It defaults off for other
launches; other families/backends, dense Gemma and affine formats are refused.
CT NVFP4 and ModelOpt MoE retain their existing header/admission/native math.
No additional hardware/context/precision requirement or candidate activation
is introduced by this frontend exposure.

Source tests passed441 without skips/warnings; all-F/diff checks and combined
review passed. Owned runtime/verification builds preserved the SDK. The runtime
passed115 installed checks and its777 installed source files matched the
capture, without a source overlay. All14 directed new-production checks and16
individual full API/cache/media functional checks passed. The seventeenth
known short-SSE cancellation-counter assertion failed again; aggregate is
false/exit1. Preserve that failure: it is neither dismissed as flaky/false
positive nor represented as complete API/cancellation qualification. Strong
older observed-active shadow checks and the P0 cooperative-stop/recovery proof
remain evidence only for their recorded sources/workloads; no new strong HTTP
cancellation pass is inferred here.

Actual `/v1/stats` planner values (bytes, estimates rather than physical memory):

| Planner field | Prior Delta4/R16 | Shared512 + Delta4/R16 | Difference |
|---|---:|---:|---:|
| Cache workspace |13,102,320,516|13,639,191,428|536,870,912|
| Serving peak |31,842,231,432|32,379,102,344|536,870,912|
| Startup peak |27,514,210,564|27,514,210,564|0|

The source/CLI/admitted-context/NativeReady proof implies one BF16 pair of
shape131072×2×512 for keys and values:536870912 bytes total, once across
layers. This is a structural allocation inference; HTTP has no dedicated
scratch-byte readback and planner values are not measured allocator residency,
physical RAM/VRAM saving or runtime peaks. It is prompt workspace, not a
persistent derived KV cache. Prior source-bound model/integrated Shared evidence
supports unchanged target words/output/cache semantics; the owner accepted its
prefill/retained-memory tradeoff. No new performance confirmation, statistical
equivalence or global-optimum claim follows from deployment.

The tested/captured CLI source and image identities precede this docs-only
status update. Runtime/CLI/test bytes remain those verified before cutover;
these later prose edits do not imply a rebuild or change the running image.

Private immutable evidence manifest: `shared-prompt-cli-deployment-final-v1.json`,
SHA256 `4b6eb49960dc8debdb986eab96a79836c41bc8aca223b180bbeb061990675a07`.
It links the source/installed/source-transfer/build and actual deployment checks.
The directed14 summary `shared-prompt-cli-production-directed-v1/summary.json`
is SHA256 `3a4b3fc33b7dc2cd6769ad94142a7119ed76e7230bf51a095fd7d559dfcc3631`.
The complete API packet summary is SHA256
`dd337443771055fb578c0f0f0f62e5e35cec210f8c52de5c31a59c0707bb91d7`;
its aggregatefalse/exit1 and failed seventeenth probe are retained alongside
all16 passing individual functional results. These are bounded deployment
records, not a new benchmark or a claim that every API/cancellation gate passed.

## W4A16 private-head CLI

The native `--dflash-head-w4a16` option forwards explicit true to the existing
head-only Engine policy. It requires CUDA Gemma4 CT NVFP4 or ModelOpt MoE,
an enabled DFlash companion, `--drafter-bits 8` and an admitted original BF16
checkpoint head. Dense/affine/other families, disabled drafting and wrong
precision fail before model loading; existing header admission owns source
dtype/shape/index checks. DefaultFalse is omitted when forwarding. No target
quantization, native SDK/kernel, activation calibration or new dispatch policy
is introduced. Other production flags/settings remain separately explicit.

The user accepted the audited broad D4 trade: cached T0 code/prose/short
+6.915/+2.475/+2.514% (both halves positive), with real cached T1 code−2.146%
and cold T1 short−1.570%. Report38f2/final5235/review77bb describe the complete
prior internal-policy scope; they do not establish all-region superiority or
equivalence. This specific accepted trade is not permission to hide later
regressions. The CLI bridge passed441 existing source and31 new flag checks;
the same-SDK installed image passed462 checks and exact777-file inventory,
with static/diff and combined reviews clear. Actual production now runs
container `4b15d567`, image `2dff55cd`, source `723221fd`, configuration
`d95ecbff`, started2026-11-05T02:17:39Z, with NativeReady, math4 and all14
directed production checks passing. Original model/drafter revisions,
context131072/BF16KV, aliases/port, vision/thinking/Google template, Shared512,
Delta4R16/N64/DF8D4 and disabled clipping/A8/proposalgraph remain preserved.
The active pointer selects `deploy-w4a16-head-cli-20261104-v1`; the previous
Shared/FP8 container `ffd9c363` is stopped and retained as rollback, with speech
services unchanged. New launches still default off unless the flag is explicit.

The initial full API attempt refused its idle precondition before testing.
A later single suite started at natural idle and passed16 functional/API/cache/
media checks, then failed `cancel-retirement`: aggregate passed/complete are
false, exit1 is retained. Its raw stream already contained the normal256-token
`length` completion before the client stopped parsing; it therefore did not
exercise an observed-active disconnect. This establishes neither a server
cancellation defect nor a cancellation pass. A separate stronger active witness then passed: visible thinking output and
running health at six generated tokens, well below4096, before disconnect;
cancellation counter0→1 and idle within the bounded cleanup contract; consumed
recovery4 with normal EOS and final idle. Its summary SHA256 is
`c92917b9bb2d3aa78a88b636bbddd4c6a8ef86534c8276a8508ad34a904d3d34`.
This is wire/server evidence, not independent GPU-inflight or physical-retirement
proof. The original full-suite assertion and aggregate remain failed and unchanged.
Fresh suite summary SHA256 is
`9a13d4f3fd629bfeec2b4e345616ea67ffd7e0f813334579788d8a178e378537`.
The earlier checkpoint `d5401c73`, actual review `f0f40a84` and closeout
`1cc74288` remain dated records of the preceding state. Capacity deltas returned
by stats are planner estimates, not physical RAM proof. The image/source/test
artifacts predate this docs-only status update; no new build is implied.


## W4A16 depth 2 and 3 screen

The deployed723221fd runtime/image2dff55cd on GB10 SM121 was compared with
all heads W4A16: accepted D4, then D2, D3, D3, D2, D4. Three genuine current-
runtime cache producers each completed6 compiler and42 full256-token workloads;
each of the six fresh CPU-unrestricted measured processes rebuilt its own
CUDA graphs through6 compiler and42 full warmups before42 measurements.
The actual current69f4 compiler context was copied with its original keys;
old5d2 model-law evidence retained its original identity and transferred only
through the audited four-frontend source delta. No old warm producer was relabeled.

All three same-policy and12 cross-policy T0-ID checks passed. Same-policy
q/p, outputs, work and cache witnesses were exact; cross-depth T1 q/IDs/work
were allowed to differ and all actual work remained charged. Accepted W4D4
production was restored and checked before report analysis.

The following cached primary rates are geometric means of two process-level
six-hit medians (tokens/s), using unchanged e51 metrics. Seeds within a process
are correlated; there are two fresh processes/config, not12 independent replicates.

| Region | D4 | D2 | D3 | D2/D4 | D3/D4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0/code64k | 71.952 | 68.877 | 73.699 | -4.274% | +2.429% |
| T0/prose64k | 86.967 | 79.564 | 88.732 | -8.513% | +2.030% |
| T0/short-native | 121.824 | 100.712 | 115.873 | -17.330% | -4.885% |
| T1/code64k | 68.766 | 68.894 | 74.372 | +0.187% | +8.152% |
| T1/prose64k | 83.916 | 74.485 | 84.133 | -11.238% | +0.259% |
| T1/short-native | 106.493 | 97.519 | 96.978 | -8.427% | -8.935% |

Neither D2 nor D3 is selected for broad production. D3's cached T0-short and
T1-short losses repeated in both halves (geometric deltas−4.885%/−8.935%);
its cold T1-prose primary rate fell17.004% in both halves and whole-request
duration rose1.947%. Cached T1 prose was mixed (−0.075%/+0.595%), despite its
+0.259% geometric point. D2's small repeated cached T1-code gain (+0.187%)
coexisted with cold-code loss and five other cached-cell losses. Gains do not
offset the required-region conflicts under this screen's objective; they are
preserved as scoped evidence, without a generic prompt/length classifier.

Cold D3 T1-short decode improved8.454% and whole-request duration fell7.584%,
while TTFT rose1.164%; those distinct costs are retained. All other cold/whole/
TTFT, callback tails, raw work and end/start drift remain in the full report.
D3/D2 cached T1-short was lower in both halves (−0.696490%/−0.412556%,
geometric−0.554624%): the zero-margin adjacent rule stops higher-depth
exploration. D4 was the incumbent reference, not a new upward exploration;
no D5, equivalence, noise or global depth-inferiority claim follows.

Observed Torch allocated maxima were lower than D4 by16,200,704 bytes for D2
and8,709,120 for D3; peak-reserved maxima were20,971,520 bytes lower for both.
These allocator observations do not prove total physical-memory savings and
cannot offset the regional losses. Constructor-plus-drain seconds were
D4[34.434,35.096], D2[32.314,34.026], D3[32.214,32.828]; CPU high-water marks
were mixed. Per-request graph keys/dispatch and48 warm q/p witnesses/process
were retained, with no measured main-graph growth or causal timing attribution.

Report SHA256 `c5f36920466baf29a0f6e0b52763c62ab931ebd8a658acb648c2cd75ba9b6af4`,
63-link final packet `b90a942b`, actual review `38ebe8dc`, final15-check result
`5a1a37d1` and exact production restoration `3aecc53e` record this finite scope.
The depth-screen disposition itself changes no runtime/test behavior or depth
deployment. The separate N128 screen below completed without production selection.


## Fused expert-up N128 candidate

The archived four-warp candidate specialized CT (`direct_input=False`) GEGLU,
width704/dims2816, rows1..5 decode/verification, retaining SK1/BF16 arithmetic
and explicit prompt-role separation. Source385 checks (one documented existing
GPU-assisted loader skip), installed277 checks/exact777 inventory, native57
independent cases and corrected full-model teacher/q/p/IDs/work/cache parity
passed. Initial metadata/setup failures and the original full-model B1
live-tuple versus JSON-list penalty-bias rejection remain preserved; lossless
container normalization fixed that harness boundary without changing kernel
math, numerical tolerances or the failed records. Fresh B2 passed.

Four fresh processes ran BASE/CAND | CAND/BASE, each with six compiler32,
42 full256 warm requests and42 measured requests. All retired safely, and
same-role/cross-role semantic checks passed before reporting. Primary rates
measure255 post-first outputs through explicit drain. Two processes per arm
are exploratory evidence; requests/seeds are correlated, and drift/tails remain
in the raw report.

| Region | Cached N128/N64 change, halves | Cached geometric mean | Cold decode geometric mean |
| --- | ---: | ---: | ---: |
| T0 code64K | -0.338260% / -0.720464% | -0.529546% | -0.683235% |
| T0 prose64K | -0.219341% / -0.223367% | -0.221354% | +0.347168% |
| T0 short | -0.633563% / -0.407337% | -0.520515% | -0.130997% |
| T1 code64K | -0.213179% / -0.404491% | -0.308881% | -0.685053% |
| T1 prose64K | -0.184264% / +0.083235% | -0.050604% | -0.237648% |
| T1 short | -0.708961% / -0.443528% | -0.576333% | -0.438038% |

Long-prefix prefill times improved0.09–0.27%; constructor+drain observations in
cohort order were39.488447/34.624802/33.781945/35.048457s. The four Torch
allocated/reserved/peak counters and logical capacity maxima were identical;
these counters do not establish total physical VRAM residency. Cold whole,
TTFT, callback tails, CPU/cgroup memory and drift remain separately recorded.

**Broad four-warp N128 is not selected.** Five repeated cached losses and the
sixth negative mixed aggregate do not justify its retained cold/prefill/startup
benefits under decode > prefill > VRAM > cache > other. This is not global
inferiority, equivalence or a noise claim. The exact six runtime files and26
independent N64 oracles are restored to accepted preprototype `c8318787` bytes;
only N128-specific selector controls are removed. Prototype Git `d0d44a50`,
build capturea538, immutable fixture and private audit history remain intact.
No performance gain is claimed from source cleanup, and no rebuild/restart is
needed for exact runtime restoration. Accepted723/2dff W4D4/N64 production
was restored healthy at2026-11-05T08:43:20.49196889Z with unchanged configd95.

Audit artifacts: `expert-n128-p3-v1/report-v1/summary.json` SHA256
`f2caa0b85d7f035ef5965f088440a39da937828559236736a8f454e8817036b8`,
final packet `55d58aaf`, actual D13 `afb6d61c`, disposition `a00b13d8`;
cleanup hashes are recorded separately in `expert-n128-cleanup-v1`.
