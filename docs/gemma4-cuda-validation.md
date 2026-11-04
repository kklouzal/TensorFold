# Gemma4 native NVFP4 on GB10, 2026-10-18

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
