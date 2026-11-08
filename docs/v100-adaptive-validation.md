# Adaptive V100 source implementation and qualification

The [implementation specification](v100-adaptive-inference-design.md) governs
the eight-V100 TP8 path. Historical results in
[the V100 qualification record](v100-flashnext-validation.md) identify their
original sources and layouts. They do not qualify the adaptive implementation,
unique histories, a different codec, or a new execution envelope.

Initial source work preceded access to the eight-V100 server. The selected SC117 GGUF
payloads and mmproj have been downloaded and verified; source/header/import
controls do not establish device execution. Source controls, pinned-stack compilation, installed
CPU checks, physical allocation, model correctness, quality, and measured
performance are separate gates. No new session maximum or throughput result is
claimed. The specification's 20-Q8/32-Q4 figures are geometry reference points,
not admission promises or capacity ceilings.

## Ordinary HTTP host envelope

An earlier persistent-cache SC117 attempt reached seven ready followers and completed prompt
warmup, then correctly failed the unchanged host-commitment guard. Its failure
and the original conservative component totals remain retained. Completing
warmup does not establish an HTTP-ready service.

The source now closes a bounded ordinary HTTP region: actual Scheduler and
RequestLimit owners, the resident model catalog, unpublished FIFO admission,
plain tokenizers 0.23.3 ByteLevel, and the exact pinned CPython JSON runtime.
Profile-managed Apps retain conservative accounting at construction and stage
re-finalization because their HTTP class outlives profile publication; admin
functionality remains available. Other tokenizer, retained/media and configured SQL regions keep their
conservative accounting. Credit insufficient for one full-context generation
is refused. The generation body remains 96 MiB, metadata body 32 MiB and default
Responses cache 256 MiB; model context, shared MTP and physical/cgroup admission
are unchanged.

The envelope independently pays logical model work, completed accepted-handler
continuations, pinned/evicted cache generations, deepcopy/memo work, transport
and RPC ownership. Completed slow sockets cannot borrow a retired logical
request's credit. C one-shot encoding includes writer resize overlap, markers,
numeric conversion objects, controlled SSE framing and overlapping strings and
wire bytes. The selected object-layout/source proof is distinct from native
allocator, dtoa workspace, C stack and RSS qualification.

Actual pinned `_json` disassembly identified a writer leak if its terminal
one-element-tuple allocation fails. Source-owned CUDA and shared HTTP encoders
therefore publish a fatal failure in the existing thread-safe Server journal,
close keepalive, refuse later encoding and notify the existing CUDA cohort
failure channel. Calls admitted before publication may coexist and are paid by
the HTTP connection bound. The real Server cap must be positive and no larger
than the admitted cap, checked before socket creation; supplied CUDA servers
must carry the same App handler and satisfy that cap before serving. Responses, SSE, statistics and authentication errors
use that same owning boundary. This changes no installed library and adds no
JSON monkeypatch or serialization lock.

The pinned-SDK CPU source composition recorded 302 passed, two failed and
zero skipped, with CUDA uninitialized. Both failures are unchanged counting
fixtures (expecting grammar compilation on count and a compiled grammar on a
count-prepared request); they reproduce on the pre-increment `1b765` baseline and
old installed runtime. At that checkpoint their assertions were unchanged;
failure evidence remains preserved. The later phase-credit integration below
reconciles the count fixtures against the established count-only contract.
This is a partially passing source gate, not an all-green or installed-runtime
qualification. The original 22 focused containment controls passed independently. The final
actual pinned-SDK four-module check then recorded 75 passed and only the same
two baselined failures, with zero skips/errors and CUDA uninitialized. All new
managed-profile, accepted-cap, supplied-server and serve-lifetime controls passed.
The final local boundary composition separately passed 38 controls. These source
compositions alone do not qualify an installed successor or model request.

Injected source controls cover poison publication, authentic primary and
secondary failures, concurrent in-flight writers, actual CUDA/shared handlers,
authentication, Responses and statistics. These are not native OOM experiments.
Pinned-stack CPU source composition, an installed successor and an actual normal
HTTP/model request remain separate evidence gates. A lower explicit diagnostic
connection cap may bound accepted-handler ownership; it does not reduce context,
MTP or route body limits. Startup still binds the real TP identity and schema
and must pass its original physical guard before serving.

## First controlled SC117 HTTP success

The subsequent normal installed diagnostic on `sm-v100` passed one HTTP chat
request using source `d48e96a1fe244335972242dd0902dab8530fc797` and image
`sha256:f4cbd945b8f2c958522cd49bac29d419614d5b606fe507d8a94524ac4d0213af`.
Its protected software manifest was
`491f93dab1f592da675221d60552e19f0ed1aa6b4e803f2eacd082fad19c0e53`.
The retained gate is
`sm-v100-http-app-successor-plan-v1/first-controlled-model-HTTP-success-gate-v5.json`,
SHA256 `0cba3a78d64dd34790fb3ce176c71114da4bed0bf7b775b205cd4a15c42b4a0e`.
Earlier UUID, cache and host-admission failures remain preserved.

The actual profile was TP8/PP1/ETP4 with reference routed experts, shared-Q8
MTP configured to maximum depth 6 and QSA 4/2048, INT8 KV, one stream/full window,
eager execution, one pending request, one accepted HTTP connection and 16 MiB
waiting-input credit. Logical context remained 262,144; the short request does
not exercise a full-context workload. Generation/metadata body caps and default
Responses cache capacity remained unchanged. The normal mandatory final HTTP
host-admission guard returned before serving, without an override. Its numeric
catalog envelope was not exported by `/stats`; no header-only or counterfactual
total is substituted for the unrecorded final authority.

Rank 0 reported serving after 622.2 s, including 491.0 s prompt warmup. The wrapper
observed readiness after 632.585 s. POST chat returned HTTP 200 with content `OK`,
15 prompt tokens and two completion tokens, finishing with `stop`. The recorded
round drafted four tokens and accepted zero; these facts do not establish MTP
quality or a speed gain. Graceful retirement completed with exit 0, PID 0, no
cleanup errors and no remaining compute processes; total diagnostic flow was
667.213 s. These are functional elapsed times, not a matched performance result.

Host-backed `/cache/triton` and `/cache/torch-extensions` remained writable and
retained across the run. All nine Triton cache files had identical before/after
hashes; worker mappings identified retained native/helper libraries in those
cache roots. This records retained/reused artifacts, not cache-hit telemetry or
a measured startup improvement. Ordinary restart cleanup must preserve the
cache roots, as described in the [persistent-cache contract](../deploy/v100/README.md#tritonv100-compatibility-and-persistent-caches).

This closes one controlled installed HTTP request and normal retirement on the
recorded profile. Full-range/model quality, same-weight/BF16 and MTP quality,
full-context/multi-session capacity, sustained/performance comparisons, media,
and the private owned-native expert profile remain unqualified. The source
now defaults Flash Next CUDA and V100 Compose to MTP depth 4, retaining explicit
overrides; the successful diagnostic above remains depth 6. MLX and other family
defaults are unchanged. The subsequent installed depth 4 batch and canonical
pilot are recorded below.

## Prefill-row development pilot

The canonical correctness pilot ran all 15 generation exchanges on immutable
image `sha256:05f24dc72f14b8221095f438fe36e8f1105737be78a7d2724dc19fc841b7acc1`
with tools commit `6d5dba660dc93fbb4719e4493429a957fee175a2`. With
`TENSORFOLD_PREFILL_ROWS=256`, all generation bodies and output token IDs,
messages, finishes and token SHAs matched the retained default 4 baseline. All seven
serial/MTP pairs were cold-serial/exact-parity checks and structured SSE matched
its nonstream result. The canonical client correctly exited 1: five of seven
known-answer cases pass, while both modes return `31` for the loop whose oracle
is `43`, and `fig,kiwi` for the filter whose oracle is `kiwi,pear`. The oracles,
original failures and full output IDs are preserved.

Independent CPU llama.cpp commit `f2918cabbffe8abf8e2a90c1c089085fa116c2cf`
using the same acquired GGUF and exact input IDs matched all output IDs for
three cases in both TensorFold modes, including both wrong answers. This
supports checkpoint/model behavior for these samples; different arithmetic and
three short cases do not establish full numerical, model-quality or import
qualification, or exclude every shared converter/architecture problem.
The reference gate is
`sm-v100-sc117-cpu-reference-v1/same-GGUF-cpu-reference-result-v1.json`, SHA256
`957760acbc3bda1150d8f6dd6282747f2b1b72d407f2f06608822acd3635f4da`.

The recorded dev profile uses `--no-graphs --no-prefill-fp8 --no-thinking`,
reference routed experts, INT8 KV, configured context 262,144, TP8/PP1/ETP4,
maximum MTP depth 4, one full window,
H1/P1, 16 MiB waiting-input credit and all ordinary device/host guards. Only
prompt-piece rows changed; no image rebuild or reduced context was used.
The recorded backend plan is 2048 when the override is unset. Fresh-process
A-B-B-A observations on that same eager profile were:

| Order | Prompt-piece rows | Readiness seconds | Warmup seconds | Validation |
| --- | ---: | ---: | ---: | --- |
| A1 | 2048, unset | 660.660 | 520.1 | Original seven-case/15-generation batch |
| B1 | 256 | 173.279 | 30.8 | Canonical 15-generation batch; exact baseline outputs |
| B2 | 256 | 171.257 | 30.9 | Frozen arithmetic probe exact |
| A2 | 2048, unset | 653.617 | 512.4 | Frozen arithmetic probe exact |

Mean readiness was 657.138593432 s for 2048 and 172.267940721 s for 256, a 73.785%
reduction (ratio of means 3.815). This selects 256 as an opt-in measured eager
development configuration for the recorded startup/short-validation workload.
It does not change the global backend default. There are two startup samples
per geometry; this is not a general latency guarantee. Compiler/frontend caches
were retained; model-page cache was not flushed or controlled.

The pilot's canonical client took 333.925 s including seven tokenizer reads;
that duration is not a matched inference benchmark. Its quality-failure exit
remains primary and its launcher retired normally. The repeat/control probes
and normal retirement passed. Longer-prompt throughput, full-context/multi-session
capacity, other codecs/methods/graphs and broader quality remain unqualified.

Evidence:

| Retained gate | SHA256 |
| --- | --- |
| `sm-v100-prefill256-candidate-v1/prefill256-pilot-actual-gate-v1.json` | `0617be2f5a1754134a6720d293dc7f42dc3bf35b213d6031d51dbeeaa92131d3` |
| `sm-v100-startup-repeat-v1/startup-256-repeat-actual-gate-v1.json` | `89c24c34cae9e9969da64efcebc72b36212c5c2a6bdc52a88f1da6eb18d80c30` |
| `sm-v100-startup-repeat-v1/startup-comparison-actual-gate-v2.json` | `80c49245087901ba444debc736149d59ab402347c4703559120f87d4f6bcaaea` |

The maintained Compose and candidate Bash recipe expose this existing backend
knob as optional `TF_PREFILL_ROWS`. Unset/empty preserves the engine's own plan;
there is no new global backend default, runtime flag, entrypoint or hardcoded
serving budget. See [the opt-in configuration](../deploy/v100/README.md#optional-development-prefill-rows).

## Private native-owned SC117 functional milestone

The guarded private run on `sm-v100` used source
`bd0885e1c46f27ba19ab366d9934fe636702b12a` and immutable image
`sha256:540c54b7fe2add056c34656d948be9c6ffa99a81250b4401cbeb3e8e47a6eeb3`.
Its retained gate, with 383 bound artifacts, is
`tp8-etp4-private-model-execution-v1/actual-native-functional-gate-v1.json`,
SHA256 `8c27d7669a1412dbb999dccddaf9a5ba0c150995ffcaea30f727388f86b5cade`.

The profile retained configured context 262,144, one full window, TP8/PP1/ETP4,
INT8 (Q8) KV, default MTP maximum 4 with shared-Q8 companion, eager execution,
prefill 256 and H1/P1. All eight genuine pre-API receipts verified the actual
`owned-etp4-native-slots-v1` plan, 48 target banks plus one MTP bank, coordinate
and State numerical bindings, and materialized local arenas. Each rank owned
23,408 bytes in arenas with rows `[5,256,5]` (target/prompt/MTP), contained within
its previously paid 292,864-byte routing reserve. This is positive actual scratch
storage, not a claim of zero allocation or additional capacity.

All 15 generation requests matched the selected reference's emitted IDs,
messages, finishes and independently checked token SHAs. All seven within-native
MTP/cold-serial pairs and structured SSE/nonstream parity passed. The known
semantic score remains five of seven: `31` still fails the `43` loop oracle,
and `fig,kiwi` still fails `kiwi,pear`. The canonical client correctly exited 1;
those failures are preserved without a waiver. Final stats were captured, the
normal launcher exited 0, no OOM/cleanup errors occurred, container PID was 0 and
health/input identities were unchanged after retirement.

This also completes the planner checkpoint `b88ad1dd` target functional check
in the same main-goal run. Combined new-path timings are descriptive only;
there was no matched integrated startup or native/reference performance benchmark.
The corpus's largest prompt was 666 tokens, so the configured full window is
not a full-context capacity test. Attention and other kernel families retain
their distinct implementations; this is native routed-expert behavior evidence,
not a fully native model or general numerical/quality/capacity/performance release.
Normal serving still selects the reference policy. No PP2 claim is made.

## Automatic resident allocation on eight V100s

The same `bd0885e1` source and immutable `540c54b7` image above were tested with
only `--parallel 1` and `--kv-full-windows 1` changed to `auto`. Context 262,144,
TP8/PP1/ETP4, INT8 KV, default shared-Q8 MTP depth 4, eager execution, prefill
256, H1/P1 and 16 MiB input credit were retained. No SDK or application rebuild
was involved.

The solver selected C27 without an administrator ceiling. All eight physical
ranks materialized 27 cache extents of 262,149 tokens, including the MTP tail,
and their actual cache bytes were covered by Torch allocated/reserved readings.
Each native receipt verified 27 States, 48 target banks plus one MTP bank,
coordinate/numerical bindings and arena rows `[32,256,32]`. The arenas used
28,160 bytes per rank within the existing 292,864-byte routing reserve.

The limiting resource was rank device memory. The header model's C28 weight
plus execution obligation exceeded the 29,073,342,464-byte serving budget on
every rank. C28 was not physically attempted; this is not a measured OOM limit
or a claim of maximum qualified useful concurrency.

Both arithmetic smoke requests returned `372`, `finish_reason: stop`, exact IDs
`[18,22,17,248046]` and independently checked SHA `33bcc7da87c4`; the cold serial
request used no draft or cached prefix. Launcher/container exit status was 0,
PID was 0, error/cleanup fields were empty, and health/input identities were
unchanged. Raw results are retained under
`tp8-etp4-auto-capacity-v1/actual-auto-capacity-v1/`, particularly
`stats-before-request.json`, `auto-capacity-result.json`,
`capacity-smoke-result.json`, `firstload-result.json` and `result-v1.json`.

This establishes resident allocation only. H1/P1 cannot establish C27 simultaneous
activity, the short prompts do not populate near-full histories, and these two
requests are not a throughput benchmark. General quality, all-active capacity,
full-history behavior and performance remain open; prior checkpoint quality
failures and the unchanged default reference policy remain explicit.

## Supplied-long client and host admission

The maintained opt-in [supplied-long client](../deploy/v100/README.md#supplied-long-correctness)
records independent oracle/request provenance, actual tokenizer histories,
explicit N/R/L, cold serial/MTP/SSE parity and positive draft work. Source CPU
controls passed 322 tests with no skips and no CUDA initialization. These are
client/fixture checks, not an actual long-context model run. Valid EOS can pass
semantic retrieval without proving exact exhaustion. Advisory slot readings
retain their reading contract; strict simultaneous populated-C ownership and
the overall requested gate remain Unsupported until a separate coherent
ownership attestation exists.

A source-only [private worker attestation procedure](../deploy/v100/README.md#private-owned-routed-qualification-procedure)
now observes the actual sole-owner round boundary instead of the advisory API.
It binds all-eight common TP prefix sequences and real State/wire/generation/
prompt/coordinate/native facts to a fresh run identity and the client service
instance. Its separate offline acceptor rechecks completed client gold, cold
serial/MTP/SSE parity and actual proposal work. There is no new GPU fence or
public mode, no change to ordinary round methods, and no catalog rebinding;
bounded private metadata is paid in loading and serving admission. The scope is
logical owner/device-stream ordering, not GPU wall-clock simultaneity. Source
checks and the pending installed long-context/host-successor run remain separate.

After the resident C27 allocation above, the H28/P27 frontend attempt failed
the mandatory host guard: projected commitment 4,403,538,763,721 bytes exceeded
351,202,996,224 usable bytes. Physical GPU allocation passed, but no API listener,
concurrent burst or benchmark was reached. All processes retired; retained raw
evidence includes the inner readiness/terminal-1 markers and outer-stop sentinel,
not an all-active success. The reviewed receipt is
`tp8-etp4-C27-allactive-v1/C27-H28-host-refusal-gate-v1.json`, SHA-256
`fe8ab915343de9c0aa2b4ba4c8305a416137d765b252a2125a9413bd225af9ad`.
Host-accounting successor work requires its own implementation and verification;
this client checkpoint neither bypasses admission nor qualifies that successor.

## Ordinary TP admission error lifetimes

The CPU diagnostic using the actual TP invocation/execution methods and a
two-rank Store/backend stand-in found eight operation-parent graphs retained
after eight recoverable `NoRoom` attempts and acknowledgment/key reclamation
with cyclic GC disabled. Explicit diagnostic GC released those parents while
all eight deliberately leased Stream children remained alive. The retained gate
is `tp-noroom-parent-diagnostic-v1/diagnostic-gate-v1.json`, SHA256
`cf31262804c05c22a99ea90b4d4854f9eeec6f26fda42b396569d0b1c5b6644e`.
Operation completion does not retire a queued Stream or its input claim.

The source fix returns the authentic `NoRoom` through a private admission
outcome after peer acknowledgments, drops the packed parent before rethrow,
and clears only finished owned RPC-frame locals. Traceback code/lines,
cause/context/notes, foreign/backend locals and live Stream children remain
intact. The Scheduler applies the same cleanup at its FIFO and legacy retry
boundaries. Peer cleanup relinquishes operation bindings after acknowledgment.
CPython's cached locals snapshots are explicitly refreshed after frame clearing;
the preserved seven-peer regression exposed that additional parent reference.
Managed serving profiles keep the original propagation and their existing
certified State-retirement/fatal-owner policy.

The final seven-module source batch passed 302 tests with zero skips and no
CUDA initialization. It covers leader and all seven follower parent lifetimes
without GC, still-owned Stream children, original diagnostic identity, managed
certifier behavior, and real Scheduler FIFO/legacy queue exercises with a CPU
backend stand-in. The source seal is
`tp-noroom-lifecycle-v1/author-source-seal-v1.json`, SHA256
`e94c3ee84ac25056c878035bb6f97df32369e784c60896b483151fcc4cc9745f`.
Earlier failing fixture and peer-lifetime logs remain retained. These are source
ownership checks, not transport, device/model, RSS or throughput qualification.
The host-credit implementation and its installed full-model checks remain pending.

Separately, the committed private-history tool and exact C27 corpus input passed
a CPU-only helper check on `sm-v100` using installed image `540c54b7`. The selected
JSON policy computed a global obligation of 45,889,656 bytes (eight times
5,736,207 bytes), with zero materialized credit. Torch was not imported and the
container exited 0 with PID 0. The helper gate is
`private-history-sdk-control-0e100ce5-v1/actual-SDK-history-helper-control-gate-v1.json`,
SHA256 `31f5da0a70f677eb4e4d33e61980de769923f07418012e71d945cc0e4aebb45e`.
This checks input/resource-helper applicability only; the new CLI/application,
worker attestation and long-context model execution are still unqualified.

## Prompt encoding allocation boundaries

The owned chat-template, protected-literal and segment encoders now accept
optional request-local allocation callbacks. They expose native Encoding counts
before Python ID getters, known batch-wrapper counts before batch encoding,
replacement/output lengths before list growth, cache obligations before base-str
conversion/splitting/lookup, and rendered-part counts before wrapping or joining.
Counts include simultaneously retained fresh arrays and cached output, without
per-input-token pool locking. A captured metadata extent can be credited before
tokenizer serialization. Normalization, literal/origin decisions, tokenizer IDs,
cache publication and LRU ordering remain unchanged; omitted callbacks retain
the existing interfaces.

The isolated prerequisite check used immutable `b7fe8e2b` source plus only the
three encoder files and their two focused test modules. It ran directly on
`sm-v100` in existing media image
`sha256:8f5c4dddd460bad82f584db3cf44e739454b317c3dd452a5ca8de76b96dab5fd`,
whose Transformers/PIL dependencies cover the shared frontend paths absent from
the text-only image `540c54b7`. All 78 tests passed with zero skips; configured
Ruff and syntax checks passed. Native FD/source identity matched, CUDA remained
uninitialized, and the container retired with exit 0, PID 0 and no cleanup errors.
The retained gate is
`http-host-admission-source-cpu-v1/foundation-isolated-v1/isolated-foundation-CPU-gate-v1.json`,
SHA256 `5fdfda8bc6dc9fbacfc7ef53b7495629811eee92b24ad866fb0ab2378c7f50b1`.

This is source-level allocation-boundary and output-preservation evidence.
The moving host-pool/App implementation was excluded from the snapshot. The
media image has a different installed application/manifest; this check does not
qualify an installed successor, the complete host-credit integration, C27
admission, native encoder/Jinja heap usage, model execution or performance.

## Progress fixture contract reconciliation

Three old stats/progress assertions failed on unchanged installed image
`540c54b7`: internal engine progress callbacks remained active for disconnect
polling even without requested wire progress, and the Scheduler delivered the
latest coalesced status rather than every intermediate publication. The retained
baseline gate is
`http-stats-installed-baseline3-v1/installed540-stats-three-node-baseline-gate-v1.json`,
SHA256 `1f3b4b996bd3e838c5e918a767a5892b8bc126f837c7ea31ab1f1c8a88897c47`.
The source and bounded-cell contract introduced by `0a7a7509` independently
establish the intended behavior; no runtime change was made for these fixtures.

The corrected fake producer emits valid nondecreasing progress bounded by its
unchanged prompt length. Chat and completion tests retain numeric bounds,
pre-token order, usage, refusal and no-unsolicited-frame checks. Scheduler tests
still require publication at rows 1, 4, 7 and 8; delivery must be a monotone valid
subset ending at 8 before the token. A deterministic stalled-consumer control
requires admission row 1 followed by latest row 8 before token 7. Documentation
now distinguishes publication from potentially coalesced delivery.

With only the held test overlay and committed test/documentation inputs, the
whole stats module passed 33 tests with zero skips against unchanged installed
image `540c54b7`. Ruff/syntax passed, CUDA remained uninitialized and the container
exited 0 with PID 0 and no cleanup errors. Gate:
`http-progress-fixture-installed540-v1/corrected-progress-installed540-CPU-gate-v1.json`,
SHA256 `849f6ee82d80be4e4afbdf185229d7147c702353cafe23d188d969dacb0d82ca`.
Baseline failures and the initial missing-documentation test-input error remain
retained. This closes fixture drift, not the moving host-credit integration,
model admission or performance qualification.

## Ordinary HTTP phase-credit integration

The ordinary fixed-profile FIFO/ByteLevel source path now funds fixed model,
cache, header and RPC obligations once, then derives one App-owned global
variable-payload allowance from the existing physical/cgroup sample. This is
an application payload bound, not an exclusive OS reservation. It preserves
the selected context, MTP, H/P limits, body limits, output queue and cache caps.
The maintained CLI initializes one App exclusively before listener publication;
concurrent factories on one engine are outside this qualified source scope.

Grants precede body reads/parse, generation preparation/publication and future
encoding, actual metadata copies, token-ID/lexical conversion, segment-cache
pins/render assembly, and RPC publication. Stream/mailbox shared loans and
authentic error loans survive socket/logical-request completion. Unknown cleanup
failures fence the owner; live-owner growth does not wait while holding payloads.
Store HTTP reads borrow immutable snapshots while public copy semantics and
cache/LRU behavior stay intact. Metadata credits accumulate both capacity-plan
copies and controller observations before copying, without reducing route caps.
The earlier generic RPC future allowance remains funded.

The immutable V4 source check passed the C27 joint-arithmetic node first, then
576 suite nodes plus 21 subtests on image `540c54b7`. Its 13 missing-Transformers
failures and two PIL skips were replayed as the exact 15 nodes on the same
snapshot in compatible media image `8f5c4ddd`; all 15 passed with zero skips.
Thus 591 distinct suite nodes were exercised passing across explicitly separate
source-test environments. Configured Ruff and syntax passed, native FD/source
identity matched, and both containers retired with CUDA uninitialized. Gate:
`http-host-admission-source-cpu-v1/increment-v4/coherent-V4-source-CPU-gate-v1.json`,
SHA256 `76f0462c0079517e8cfd309cd8cd5c5cf6d4744512fe7e2ef2d876a96dfb4417`.
The C27 node uses recorded geometry/baseline, 27 conservative 1 MiB/R64 owners
and monitoring, with mocked physical allowance; it is not a fresh physical grant.

Final V5 cleanup removed two verified write-only fields, including an unnecessary
closed lease/body cycle. All other V4 files remained unchanged. The targeted
selected-image rerun passed the C27 node and 41 core tests with no skips, plus
Ruff/syntax; the new lifetime control runs with GC disabled. Gate:
`http-host-admission-source-cpu-v1/increment-v5-targeted/V5-targeted-cleanup-CPU-gate-v1.json`,
SHA256 `bd6b98a3252112720fb1a9b7667bbc2a6c5e5666128497687dce13127da6fc91`.

Count routes retain validation without native generation-grammar compilation.
Count-produced preparation is explicitly refused if reused for generation,
before generation-host/model/matcher work; outer logical-limit bookkeeping is
unchanged. The old counting assertions were corrected against `2173fa21` and
independent installed-baseline evidence while preserving the generation native
retention barrier. Public stats fields remain unchanged; constructor refusal
diagnostics are bounded and retained in operator tracebacks.

The installed `7ea3d8ce` successor subsequently passed 214 focused CPU checks
with no skips, then admitted the actual model with 32 Q4 full-window States on
each of eight V100 ranks. Arithmetic MTP/cold-serial smoke passed and the
[partial phase reference](benchmarking.md#first-sc117-q4-reference) completed
levels 1–16 before intentional cancellation during 24. This advances installed
admission and bounded execution evidence; it does not qualify actual C27
activity, 32 active requests, complete model quality/performance, native
encoder/Jinja heap, allocator/RSS, C-stack or broader managed/media profiles.

## Private Dense M32 component checkpoint

The existing private `native32` Dense selector now reuses prepared operands
for complete, contiguous, 512-byte-aligned M32 tiles with disjoint output.
It retains the incumbent M32 products, dtype and output leading dimension.
In this new multi-tile path, tails, unsupported views, gradients and aliases
keep the original implementation.
There is no new kernel or persistent storage. Reference remains the default.

On the current V100 SDK/image `0a13de3c`, a read-only canonical-source candidate
passed all 117 Dense GPU checks plus 39 existing prepared controls, with no
skips. Frozen incumbent oracles cover FP16/FP32 bits, actual projection widths,
tails, guard regions, changed-input replay, signed zeros and subnormals.
The source CPU module passed 127 tests. This is component evidence, not an
installed candidate image or full-model performance/quality qualification.

The broader legacy bound-graph checks remain incomplete. After correcting
invalid no-method proposal geometry and an unqualified native-attention fixture
selection, a strict stock-MTP/INT8 graph comparison still differed by up to
`4.041939973831177e-6`; its failure is retained without widening equality.
The graph investigation and distributed follow-up remain open. The controlled
Dense benchmark will use uncaptured execution and a genuine successor image
containing the tested candidate bytes before any promotion decision.

Private source hold: `sm70-shared-operator-audit-v1/dense-prototype-hold-v1.json`,
SHA256 `fe4ae5b91bd9cc84e1e57730bbc71f63533e110954789b15113b6439138aa4db`.
Initial GPU results and subsequent graph failures remain under the same audit
directory in `dense-GPU-source-v1/` and `graph-GPU-v3/`.

## Ordered Half MoE partial component checkpoint

The process-start private selector
`TENSORFOLD_PRIVATE_SM70_MOE_PARTIAL=native-half-v1` reuses the existing SM70
CUDA kernel for ordered Half expert partials. It retains the original separate
FP32 multiply/add sequence (`--fmad=false`), output storage and row bounds.
There is no new kernel or tensor allocation. The selector applies only to CUDA
Half on SM70; CPU/other-device paths, the distinct F32 Triton FMA path and other
family operators retain their prior policy.

Unset or explicit `inherit` preserves the existing family policy, including
legacy optimized-profile behavior. The value is fixed at module import;
changing the environment after import does not change the selected policy.
Unknown values fail closed. Use only a candidate image containing this source,
record its exact identity and verify the selected policy before an integrated
trial; an environment variable cannot add support to an older image.

The held source passed 51 CPU and 13 actual SM70 GPU checks with zero skips on
the pinned `0a13de3c` SDK. Strict independent ordered oracles cover cancellation,
subnormals, signed zeros, row/guard bounds, malformed inputs and selector
isolation. Source-only verification and the existing native-kernel ELF are
recorded in `native-half-partial-prototype-v1/source-CPU51-GPU13-gate-v1.json`,
SHA256 `844dbef4aaa55384f388ca32ec444f8d1f25d0ad327a0027d8ed4ef4afc127e2`.
Complete model output and performance checks remain pending; this checkpoint
selects no new deployment default.

The private owned launcher additionally validates a captured startup request
before model construction and again before publication. Each rank records
`half_partial_policy`: requested/configured values, selector support and the
inherited native-family flag. An older image with no selector is accepted only
when no explicit request was supplied, and is labeled `legacy-inherited`.
Present invalid metadata, unsupported requests and mismatches fail closed;
the launcher never mutates the policy or treats a later environment edit as
a mode change. The attestation passed 111 local ARM CPU protocol controls,
Ruff and syntax checks; that is separate from V100 numerical qualification.
Gate: `private-half-attestation-CPU-v1/half-attestation-final-gate.json`, SHA256
`b8c70391b469f7a48cca431ee2340b9cc3efba3009e29870cfe242f876b670f7`.

An application-only image built directly on V100 from committed `5fccdc2c`
is `sha256:8e08ef95fd2e30b9229e6b569f17d3389827b143fca97a36e14f352bd9c02cb8`.
It passed 276 installed CPU controls with zero skips. Its 782-member runtime
inventory differs from `0a13de3c` only in `family_ops.py` and the inherited
private Dense implementation in `gguf_weights.py`; the generic expert CUDA
source, SDK, Torch and native FD remain unchanged. The image retains `inherit`
without an environment override. This is packaging/control evidence, not an
enabled-Half model trial, numerical requalification or deployment promotion.
Gate: `half-app-5fcc-v1/genuine-half-only-installed-gate-v1.json`, SHA256
`25fb37f12c80042d4f0bce9bdfab947d564f9b48aea749ffddf548ced769236d`.

## Unchanged baseline reproducibility

One unchanged `0a13de3c` instance completed all 28 preserved-request exchanges
at TP8/ETP4/PP1, ISO4, MTP4, context 262144 and cap32. All responses retained
128 output IDs, length termination and zero cached tokens. Across the saved
reference, repeated concurrent batches and isolated/serial variants, five
prompts produced two distinct full output sequences. Both sequences changed
by the coefficient candidate were also observed in unchanged-baseline runs.
This prevents uniquely attributing those differences to that candidate;
it does not establish candidate equivalence, quality acceptance or causality.
Actual router inputs and state mapping remain investigation requirements.

The client, normal launcher and container exited zero with no OOM or cleanup
errors and unchanged source/health. The outer supervisor exited one because
it still required a matrix-only `allactive_observation_completed` marker from
the diagnostic result. That operator schema failure remains recorded; this
is not a whole-operator pass or an active-owner capacity certificate. No
performance result is inferred from the diagnostic.
Gate: `coefficient-baseline-repro-v1/baseline-repro-actual-gate-v1.json`, SHA256
`cca46c70e3fcade464ecc9a0e2e9dad822b54bb16709b76ff83f33e9e768a469`.

The private [router-row observer](benchmarking.md#private-router-row-capture)
passed 186 CPU controls with zero skips on the pinned V100 runtime. Exact
Torch `04cf2557` source inspection confirms its same-dtype contiguous D2H
copies do not allocate device temporaries. A subsequent removal of one
redundant piece-count check is structurally neutral: the pre-loop bound and
one-append-per-piece invariant remain. Actual capture and numerical attribution
are pending; the observer is excluded from performance measurements.
CPU gate: `router-row-capture-v1/CPU-v2/capture-source-CPU-gate-v2.json`, SHA256
`2debc35014d6440b9da656154683778b41a453430f8c6395cb2487775dad0a8a`.

## Independent serial-router trial

`TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS=native-serial-v1` selects the existing
native router-logit operator independently of the broader execution profile.
The import-time default `inherit` preserves that profile's existing flag;
invalid values fail closed and later environment edits do not change the
selection. This does not enable native top-k/route, padded linears, attention
or other family kernels. Existing decode-only dispatch and storage/device
guards remain unchanged: eligible Half inputs have 2–32 rows of width2560,
with aligned contiguous FP32 router weights and disjoint FP32 output on SM70.
Prefill, one-row and unsupported operands retain their existing paths.

The selector passed 49 local ARM CPU controls with zero skips, plus Ruff and
syntax checks. These establish policy isolation and unchanged guard behavior,
not current-V100 arithmetic, model quality or performance qualification.
The kernel's strict serial-cuBLAS comparisons on the current V100 SDK are
separate from selector and complete-model qualification. In those tests the
private selector must stay unset or `inherit`: their independent reference arm
toggles only the original profile flag. Enabling the private override for both
arms would invalidate the oracle.
No new default or production deployment is selected by this checkpoint.

On one physical V100, the unchanged installed `8e08ef95` router kernel and
the two maintained GPU test modules from `5fccdc2c` collected 64 tests:
59 direct arithmetic/dispatch/routing checks and four full-forward/state/prefix
checks passed, with zero skips. The remaining six-lane graph fixture failed
before capture or router comparison: it requested two proposals for a model
with speculation method `none`. The overall suite remains failed; no runtime
guard was relaxed. The private selector was unset, preserving independent
reference/native arms. These results do not qualify the new selector in an
installed candidate or establish full-model performance or quality.
Gate: `current-SDK-router-GPU-v1/current-SDK-router-component-gate-v1.json`,
SHA256 `b378912e41947a1277e43b577751493c98b0b25673c6f5890dbe27ba35b964f4`.

## Shared owner-attention diagnostics

On the existing V100 SDK/application image `0a13de3c` (application source
`7ea3d8ce`, Torch `2.14.1+tfcu129sm70`), all 42 maintained paged-owner GPU
controls passed with no skips. These cover Q8, ISO4 and Planar4 ownership,
mask/epoch/status, exact structured normalization/merge and capture behavior.
They do not establish general numerical equivalence or model quality.

The separate seeded diagnostic uses the original owner Torch operations as its
oracle, with actual encoded Q8/ISO4 caches and matching query/selection inputs.
Seven CPU metric controls passed and all 48 GPU diagnostic cases executed.
There is no acceptance threshold or quality verdict:

| KV format | Cases with any output-bit difference | Largest final absolute difference | Largest final finite ULP distance |
| --- | ---: | ---: | ---: |
| Q8 | 23/24 | 0.015625 | 127 |
| ISO4 | 24/24 | 0.015625 | 44 |

Final outputs were finite. Empty-shard local maxima retain their expected
negative-infinity sentinels. Phase differences are cumulative through the
native pipeline; they do not isolate an individual kernel's error. These are
component frames, not language-model logits, perplexity or output-quality
measurements. The native WMMA/TILE32 path and incumbent ATen/TILE128 path
therefore remain separate candidates pending full-model qualification; no
existing INT8/8192 quality tolerance is borrowed for this decision.

Reproduce the diagnostic on actual SM70 with the pinned application environment:

```bash
python -m pytest -q tests/test_paged_attention_diagnostic.py
python -m pytest -q -s tests/cuda/test_sm70_paged_attention_diagnostic.py \
  -o junit_family=xunit1 --junitxml=numerical-diagnostic.xml
```

The original run's xunit2 property-format advisories are retained: all 48
JSON properties were present and parsed. The command above selects the format
that supports those diagnostic properties; it changes no numerical checks.
Raw provenance is retained under `native-owner-numerical-diagnostic-v1/GPU-v1/`,
including `numerical-records-v1.json` and `numerical-summary-v1.json`.
The source/CPU seal is `source-checks-final-v2.json`, SHA256
`8e5ac9e235dd76e92cde3eb0549f1eedc85c0685dd2e9ddf466552d871828021`.

## Neutral artifact I/O boundary (A1)

The source now separates bounded container/file validation from model and device
selection. `tensorfold.tensor_file` is the unchanged shared stdlib file-boundary
implementation moved out of the CUDA package, with its owned callers updated.
`tensorfold.gguf_file` owns the existing GGUF formats, immutable inventories,
shard/extent validation, packed-byte spans and path resolution. Neither neutral
module imports a model family or accelerator runtime. Flash Next retains its
architecture guard, token-ID identity domain, semantic configuration and CPU
decode oracle. An opened inventory can pass directly to family configuration;
this avoids reading headers twice and preserves an explicitly selected shard
as the entrypoint.

Focused controls cover neutral inventory versus family admission, backend-blocked
imports, split-header read counts and the existing MTP inventory handoff. The current CUDA-hidden source batch recorded 1,228 passed,48 failed and four
skipped; all48 failures also occurred in the immutable baseline, and all new
boundary and corrected resolver cases passed. Static checks passed on 115 Python
files. This is partial source evidence, not an all-green suite: the existing
source failures remain historical evidence rather than an all-green result.

A fresh built-image check then recorded **1,264 passed, 12 failed and four
skipped** in 17.17 seconds. All five new boundary cases passed. The same 778
installed runtime-file hashes were present before and after the check, with no
runtime source overlay, neutral-module origins under the active purelib, and the
old `tensorfold.cuda.tensor_file` module absent. Native construction used the
current owned C source, exposing `memfd`; rebuilding the native helper resolved
36 stale-FD failures from the source-test SDK without changing source or tests.
CUDA remained uninitialized, and the source-foundation check exited zero.

The remaining 12 failures match the baseline causes: four fake-engine fixtures
lack `_sm70_import_plan`, one fake import blocker rejects the required
`request_limits` import, and seven cases reject the actual ARM64 SDK where their
native retained-payload contract requires the x86-64 Torch04cf2557/GCC13 ABI.
These remain **unqualified failures**, not passes or an equivalence claim. No
native ABI guard, test assertion or runtime check was weakened.

Exact installed evidence scope:

- Source checkpoint: `7be4925fec5eed762fe7b4b110be9289a8e29d0b`.
- Built source UID: `23eb2c5a3058ce73e709e348bde358ec06b5d6c49f5fc6d218db12b3c439da89`.
- Runtime image: `sha256:fce3dc00acc4c5aab8de1a1d26bd57c8be120705f097c4d3ce04169723dac53b`.
- Verification image: `sha256:c2d6a9ccd4acbdbb92a119e2f4d1f4527033567690d6923b95a8e141badd2421`.
- Active package origin: `/opt/tensorfold-venv/lib/python3.12/site-packages/tensorfold/__init__.py`.
- Native C source SHA256: `f85d2c923adfaa14e819250e8def3c3166b8c8624295a37572916e62ad20c1f2`.
- Installed gate SHA256: `b77dcc5a5fc3cfbd7918e1cecd116f271687210dc381de29f3a02e5f34c8566b`; per-cause triage SHA256: `0f6f5fbe95015f6c3fb95c7c4d8115c5ecf8a510c73c0818e86ea0ca8fd745e1`.

The gate and triage are retained in the private
`providerone-alignment-20261106` audit packet. This qualifies the affected
installed I/O boundary only. It adds no format, supported model, device
execution, capacity, performance or physical V100/SM70 qualification claim.
Selected GB10 and other existing profiles retain their original evidence scope;
production was unchanged. Earlier source runs and failures remain preserved.

## Current v6 source and CPU evidence

Current source implements the standard Responses and durable Conversations
lifecycle, bounded FIFO admission and live input/output credits, the common
prefix/retained-state catalog, RAM save/restore, and blocked-request requeue.
Logical API records, reusable computation and guaranteed retained checkpoints
have separate ownership and resource claims. See the [API contract](api.md#durable-conversations-and-quotas)
and [fixed serving policy](serving-profiles.md#fixed-endpoint-fifo-and-retention-policy).
These implemented paths have CPU software controls; they are not a qualified
V100 serving configuration.

The published composed installed CPU checkpoint uses frozen source tree
`ff1883131180963a1de18bb00c74d6777466dcb9` and local image ID `sha256:fb0ac861483f0bc14bef1f73535d07d358c196c7e9358f76da3390f1434813a8`. It passed
**5582 tests and 144 subtests across 183 modules**, with
9 explicitly documented skips. All **707 installed code files** matched
the frozen source; the 2,012 captured inputs and external frontend identities
were verified without a source overlay or within-run drift. The final receipt
records the exact result, skip disposition and immutable image identity:
[CPU software receipt](verification-v100-adaptive-cpu-20261026.json),
application commit `7d3ddbb9d066fe7451769ad9a36739619ce2d618`. The installed pytest run took
551.56 seconds.

The composed source includes the canonical key/value wake relay, actual owned
synchronous Store provenance for every new unique profile (no explicit
retention, GPU tier or RAM tier), and process-fixed resource highwaters.
That checkpoint's payload contract binds CPython 3.12.3, owned Torch
`5ab14acbced0c33e6b6e8fdbc5b541b57487f7e1`, GCC 13 build/ABI and the recorded
GCC 14 runtime content. State and hard file-descriptor lifetime reservations,
native/Python transfer and control owners, source field guards and host
prepayment are implemented and covered by the source/CPU evidence. Generic
legacy provider defaults retain their separate contracts.

The final resource producer passed **236 CPU checks**, with zero skips and
Ruff clean. Earlier scoped evidence includes 50 copy-applicability checks,
138 caller checks plus three reclamation controls, and Python/RPC last-use
and resource controls. These suites
overlap the composed gate and are not summed into its installed total.

The generic per-byte graph projection at the fixed 96-MiB RPC cap is
conservative and can refuse a host configuration. That is a refusal of the
declared projected envelope, not a measured physical capacity limit or proof
that a practical workload requires that much RAM. Tighter schema-specific
bounds remain a separately assessed candidate; no fitted margin was selected.

The previous native SDK has a retained failure-path P1 finding.
On a supported `getnameinfo` failure, its endpoint formatter casts the
address of its pointer variable instead of the pointed-to socket address;
the IPv6 path accesses beyond that pointer object. The same failure path has
an unsynchronized shared flag. Native CPU fault injection reproduced corrupted
endpoint diagnostics while the isolated data round trip succeeded; no crash or
instrumented race was claimed. Owned-FD/nonce authority does not use those
formatted strings. Passing that application's CPU gate did not repair the
native P1 violation.

Current source selects corrected owned Torch
`04cf2557f315a350e7eea7ab064288def65db75a` and refuses the older 5ab revision
for adaptive native-payload promises. The [source foundation record](../deploy/v100/source-foundation.md)
identifies its immutable SDK/wheel and native CPU controls. Direct IPv4/IPv6
formatter/concurrency controls, actual-wheel IPv4 normal/forced endpoint and
Store controls, and the immutable SDK's actual ABI/runtime-content checks pass.
Actual IPv6 Store execution is unrun on the IPv6-disabled development host.
The only source changes since 5ab are the internal socket correction and native
test wiring; public headers, Store layouts, allocation equations and transport
defaults are unchanged. This structural reuse preserves the exact CPython,
compiler/ABI, loaded-runtime and finite hard-FD guards.

The fresh corrected-stack application is commit
`2c0b4c4a51fbf2f8370fcccd2980c3e3a77ede2b`, frozen tree
`21a12df376313d5e95bbae5e269b32ae31ad2d14`, local image ID
`sha256:e99c7284234376f43634c7103fcd6fe313fee4b61c073ad79965a100b2b1baef`.
It independently passed **5,582 installed tests and 144 subtests across 183
modules**, with nine documented skips in 559.91 seconds. All 707 normal
site-packages code files and all 2,024 archive inputs (2,013 base scoped inputs)
matched before/after, with no source overlay. Its client consumer supplies
OpenAI SDK 3.37 and xgrammar 0.2.8 on the actual corrected source/media stack.
The [corrected CPU receipt](verification-v100-sdk-rebind-cpu-20261026.json)
records full identities, raw-result hashes and skip disposition. The earlier
5,582 result remains separate historical evidence; it was not transferred.

The e99 checkpoint's target source audit exposed a project identity policy
issue: the framework's configuration display includes runtime CPU
dispatch (development AVX2 versus target AVX512) and conditional CUDA sections.
The owned immutable portable-policy successor separates stable build identity
from fresh host observations without rewriting the original sealed image.

That application's commit `fc5e6dd4`, tree `7bdd8392`, image `f8f66a9d`, independently
passed **5,707 installed tests and 144 subtests across 187 modules**, with nine
documented skips in 555.98 seconds. All 707 code files, 714 package source/data
files plus the compiled FD extension, and 2,028 fixture inputs matched before
and after, without a runtime source overlay. The mapping-policy successor at
`a39646d1` changes only the audit producer/policy, tests and documentation. Image
`6daae245` retains the exact application wheel SHA256
`805246a27273b16948d984ecf0072fe71705ecad605b02a1e952eb8810718e60`.
It passed 66 affected controls and actual immutable-image source/header/native/
combined-media checks around a native 2-MiB shared mapping. The
[portable-policy receipt](verification-v100-portable-policy-cpu-20261027.json)
preserves both result scopes; the full 187-module gate was run on f8, not 6da.
Actual successor target acceptance after GPU work and all broader model/resource/
quality/performance gates remain separate and pending.

The attention descriptor successor is held commit `e3abd47a`, integrated while
preserving the published policy documentation. Its new whole application wheel
`6fac3438` and immutable image `96f69617` change one of 707 code files; 706 are
byte-identical to the full-187 baseline. Normal-installed checks passed 116
attention/capacity/policy/execution-identity cases and 24 subtests in 11.88 s,
plus 33 actual new-FD ABI/I/O/snapshot/listener cases and five subtests in 3.24 s.
All 715 package records and 46 protected metadata files matched before/after.
The new FD binary has its own identity and actual tests; it is not labelled the
old binary. The [affected-image receipt](verification-v100-attention-runtime-cpu-20261027.json)
records these scoped results separately from the older full 187-module run.
The fix avoids legacy wheel metadata only when that private attention candidate
is inapplicable; it preserves source04cf ineligibility and real math/workspace/
device settings. Actual model/output/numerical/quality/resource/performance
qualification remains separately owned.

The logical/physical State successor, held commit `64e3ed7f`, keeps the admitted
logical context separate from physical speculative guard rows. The initial
target remains logically 262,144 rows with 262,147 physical cache rows; physical
allocations, resource accounting and position guards are preserved. Whole wheel
`b9839482` and image `a19233f2` passed three affected normal-installed suites:
158 constructor/logical/retention/indexer/prefix checks in 10.63 s, 1,177 State/
storage checks in 32.24 s, and 231 engine/Multi/profile/position/media caller
checks in 21.85 s. All 715 package records and 46 protected files remained
unchanged. The [logical-context receipt](verification-v100-logical-context-cpu-20261027.json)
retains the original 72 fixture failures and their justified three-fixture
correction; the application wheel/image did not change for that fixture repair.
This is a new affected-source result, with no full 187-module rerun claim.
The target passed the original State row-domain failure point but later
startup still failed before readiness. That failure remains under primary
investigation; this CPU checkpoint supplies no OOM or model-readiness diagnosis.

The subsequent startup-order source `a1523f66` defers public Scheduler/FIFO
configuration until constructor synthetic warmup completes. Early TP/control
ownership, managed/follower transitions, rollback and real public principal/
ticket/admission validation remain unchanged; no synthetic credentials are added.
Whole wheel `86a04dee` and image `30fda46a` passed 68 direct startup/FIFO/TP
controls. A lifetime AST fixture required the actual `context_window` property
and logical 2,048 plan while retaining physical 2,053/depth four and all original
assertions. Its 15-test/two-subtest leaf reran successfully; 23 nonoverlapping
additional cases retain their separate earlier pass. The original five fixture
failures remain preserved. The [startup-order receipt](verification-v100-startup-order-cpu-20261027.json)
binds all 715 package records and protected/configuration checks without a new
full 187-module claim. A live whole-wheel diagnostic passed the previous rank-0
failure boundary, with all eight ranks joining target input preparation; the
reported model state remained loading, with no health or complete-output
acceptance claimed.

NVIDIA compiler issue **6111901** remains a separate release gate; see the
[source-foundation contract](../deploy/v100/source-foundation.md#compiler-issue-6111901-remains-a-release-gate).
Actual V100/TP8 CUDA/NCCL execution, model output/numerics/quality, physical
capacity, allocator/RSS/NUMA and whole-workload performance remain unqualified
for this source. Native IO wall deadlines are not inferred from Store wait
timeouts. No new session maximum, throughput result or V100 release is claimed.

The previous consolidated candidate was tree
`413e132bf977625f485fffedf2011cc4bfc02f07`, image
`sha256:05eca41e61a9410660d2a38c2c4d9ced06124efec64d635cb21580bcbb399323`:
5,214 tests, 144 subtests, nine documented skips, 156 modules and 706 matched
installed code files. Its skips covered two MLX modules, an optional Qwen
tokenizer, optional exact DFlash/MTP artifacts and four Docker Compose controls.
It remains distinct from the published media/frontend milestone below.

An intermediate installed attempt used tree
`045446b572d9be7a8aa08ef9f1fde125e77b38f8`, image
`sha256:c903e72e3edb5e50bf26bac22acdd697fa444844b90dd639a56ddf48a69a5fe3`.
It retained 16 failures, 5,564 passes, nine skips and 144 subtests: a shared test
Store returned strings where the actual native contract returns UTF8 bytes.
Only that fixture producer and its regression changed; shipping strict ACK
validation and all original lifecycle/error assertions were preserved. The
negative attempt remains separate from the successful successor receipt.

## Historical text/core installed CPU checkpoint

The installed application built from `0f656e9ab9642af6fa19210d721de13d375ea51d`
passed 3,277 tests and 105 subtests across the 78 affected test files, with four
documented optional/environment skips. The only later fixture change at
`b56298202ebe14b7fb1533ccb9f3871009f94d4c` declares the owner-layer reference's
existing unbound context explicitly; numerical assertions remain unchanged.
The image is
`sha256:9dc7ada61e4c1fdf272d218936ec8a08f3e372f40bec1dab53cae9237970bbc9`.
The source-built dependency environment was unchanged; this was an application
wheel rebuild and ordinary installed-package test, with no source-code overlay.

Coverage includes compact raw/pool transactions, issued coordinate/frame
identity, MTP conditioning/recovery, source authority, native/YaRN preparation,
Engine/CLI and serving-profile lifecycle, bounded admission/output, cancellation,
and the retained legacy CPU references. Corrections preserve primary failures,
guard complete model/profile identity, snapshot overlapping MTP inputs before
writes and poison a bound State after failed restore mutations. Separate cleanup
and policy source reviews are closed for this checkpoint.

This is a software/CPU result, not GPU kernel, collective, model-output,
quantization-quality, physical-residency or performance qualification. Native
SM70 compilation retains its separately recorded exact-source scope. That
snapshot preceded the installed media/frontend milestone below; it does not
replace the current source and qualification boundary above.

## Installed media and exact frontend integration

The application at `caf1eeda`, merged to `python-0.6` as `64b24dd5`, passed
4,365 installed CPU tests and 124 subtests across 111 affected test modules.
Four documented optional/environment cases were skipped. The ordinary wheel
image is `sha256:30e7036aa0d23ed4bb203f68a2d6e5f34cda617d06bf5cc2a8297d60080ef2bc`;
all 694 installed code files matched immutable source tree
`72aa4e585f2747ae02da083bb7a0d5958358212a`, with no source overlay.
[The maintained receipt](verification-sm70-media-integration-20261026.json)
records provenance and limitations.

Coverage includes the actual bounded CPU media work/TP input/App ownership path,
request-local decoder failure retirement, target/MTP shared input spans, complete
model token-ID domains, CONTROL presentation, nonlexical UNUSED rows, canonical
32-core MTP import/shared target vocabulary weights, frame/recovery containment,
principal request ownership and side-stream graph warmup ordering. CPU cohorts
and import controls do not qualify CUDA collectives or model outputs.

Current v6 FIFO admission, complete RAM retention/prefix reuse, standard durable
API lifecycle, host/device resource admission and automatic blocked-request
requeue are later integration work. This checkpoint is neither V100 release
qualification nor a capacity or throughput claim.

## Fixed initial unique history (A2)

The selected adaptive Flash Next TP8/two-KV-head preparation now constructs an
explicit degree-four `HistoryPlacement`. That same selected map drives the
planner, the published weight metadata and each State's ownership. Generic
degree-one/two/four helpers remain available to isolated research/oracles; their
internal default is not the serving selection. MTP retains its separately fixed
four-way striping, and six-layer stages retain their actual worst-rank/tail skew
rather than inheriting a twelve-layer balance assumption.

The resource ledger retains `migration_overlap_limit` with value zero. This
removes only the dedicated live-migration liability; prefix/COW, idle recovery,
restore, replacement, media and other workspace charges remain independent.
Ordinary Multi initialization does not enable migration. Initial runtime
`attention_degree` requests refuse before publication or scheduler wake without
changing active/retained state; bounded execution controls remain available.
The optional transfer algorithms and independent failure/partial-progress oracles
are preserved, without an opt-in serving flag.

The affected source and fresh installed-image controls passed. Each broader
batch recorded **367 passed, four failed and 12 skipped**; the installed batch
took 9.97 seconds and exited 1. All eleven affected test modules and the new A2 cases
had no failed nodes. Required deployment/public-permission checks passed 194
cases, and static checks passed on the seven affected Python files. The four
failures are unchanged dedicated target checks: three require the selected
x86-64 Torch04cf2557/GCC13 ABI, and one requires tokenizers 0.23.3. The 12 skips
are frontend-version gates. These remain unqualified target gaps, not an
all-green suite or an equivalence claim; their guards and tests were not relaxed.

The twelve synthetic constructor geometry cases isolate only
`retention_resources.require_native_payload_runtime`, supplying an explicitly
synthetic finite `hard_nofile=1024`. All real headers, native payload equations,
other ledger terms and device-floor assertions remain exercised. The shipping
ABI guard and dedicated actual-ABI tests are byte-unchanged; these unit tests
do not certify an ABI. Actual CPU State constructors exercise prepared fixed
metadata, while the real constructor observer checks the selected planner.
Independent rank/page/tail controls cover both six- and twelve-target-layer
maps and separate MTP ownership. Ordinary Multi initialization and rejected
runtime requests are exercised without activating migration or publishing a wake.

Fresh installed evidence uses source checkpoint
`d4469a679cdd7d9286298ec282c7d6405c06b6b9` and built source UID
`47e568407b78fbcb718da872ccd86f53252c157c301c5d9bd99c38f97873445d`, runtime
`sha256:645aa9af855129ec459a3676ff2a74e27496c7311c3a8ac0de929a5cc5165ce2`
and verification image
`sha256:f5503d631aeb101112bc5f3ed719a22c88dc1a48bf0685f13cf4adb8b7926fcf`.
All 778 installed runtime-file hashes matched before and after the check, with
no runtime source overlay. Changed Engine, Multi and server-runtime module
origins were under `/opt/tensorfold-venv/lib/python3.12/site-packages/tensorfold/`;
the current native helper was freshly built from C-source SHA256
`f85d2c923adfaa14e819250e8def3c3166b8c8624295a37572916e62ad20c1f2`.
The installed gate SHA256 is
`88603b0a4714c2462cfb1becd2b017f59b9626471211389d7f75b04244d23320`.
CUDA stayed uninitialized and the
source-foundation check exited zero. The exact logs and earlier baseline
failures remain in the private `providerone-alignment-20261106/fixed-history-v1`
audit packet; historical passes are not relabeled as current target evidence.

This does not qualify physical V100/SM70 allocations, full-model state/quality,
capacity, latency or throughput. Selected existing GB10 profiles and production
are unchanged.

## PP1 expert partitions (A3)

Source and fresh installed CPU controls cover the one-server expert-partition implementation. Preparation selects
ETP8/4/2 inside TP8, deriving EP1/2/4. Fixed contiguous groups own 512/256/128
SC117 experts, with intermediate shards of 80/160/320 values. Target and
shared-GGUF MoE MTP use the same partition; nonexpert operators and the shared
expert remain TP8. Packed source coefficients retain their original encodings
and partial-block offsets. A rank stores only its group's routed banks.

New ETP4/2 reference execution filters global routes before local bank lookup,
initializes nonowner outputs to zero, and uses the existing full-TP rank-ordered
FP32 gather/consumer. Empty routed groups retain collective participation.
The shared expert remains independent of routed EP ownership. Owned-row
compaction and scalar validation include device synchronization and dynamic
temporary allocations; these are reference implementation costs, not measured
speedups. The resident, decode/prefill, remapping and companion ledgers must
cover the selected geometry. Degree-specific floating-point reductions require
their own numerical and full-model quality qualification.

The public startup option is `--expert-tensor-parallel-size 8|4|2` for local
Flash Next GGUF, CUDA TP8, resident banks and GPU histories. The Python family
and engine option is `expert_tensor_parallel_size`. Omission preserves existing
profiles. Explicit ETP1 is refused; the separate rank-zero/RAM expert-follower
mode does not implement this profile. No PP2 flag or pipeline support is added.
PP2 remains required work with separate stage, transport and transaction gates.

The existing startup cohort agreement now includes effective ETP, global expert
count and full routed width, so default/explicit ETP8 agree and mixed degrees
refuse before model admission. Computed-state identities distinguish the
degree-specific reduction policy from physical rank placement. Prepared manifests
validate the new selector and expert coverage before retiring the current model;
ordinary runtime execution updates cannot change this structural choice.

Source and fresh installed batches each recorded **1,217 passed, 29 failed and
one skipped** across 26 modules. The source batch took 22.73 seconds; the installed
batch took 22.95 seconds. Both exited 1. Every failed case has matching, nonempty
exception diagnostics on the unchanged `0db3f178` reference: seven synthetic
provider fixtures are refused by the existing FP16-GGUF guard, ten by the compact
Planar/Iso4 provider guard, and twelve older parallel fixtures lack a `positions`
attribute. These are inherited fixture failures, not passes, equivalence or
hardware failures. The skip requires optional pinned companion/actual target
metadata that was not supplied. Required deployment/public-permission controls
passed 194 cases; static checks passed on all 22 affected Python files.

The bounded resource repair passed 182 tests with one optional-fixture skip.
Prefill attachment, admission and loading now bind the provider's shared-width
contract: a partitioned in-place plan cannot understate the legacy provider's
full-output-copy workspace merely because routed widths match. Prepared-profile
controls passed 120 cases, including early refusal of invalid ETP degrees and
expert coverage. The existing AST wiring unit reads the active Engine module;
its final four cases passed separately after that test-only origin correction.
These overlapping batches are not summed into a unique-test total.

Fresh artifacts use source checkpoint
`7e4e8cdc0a8e01ec03ab5c43afcc7a8844c85681`, source UID
`4362794f2ebaa3f62ea79e0eb410394fc97b9223bd20aa67d472937edb10d651`, runtime
`sha256:5cea184aeb7c298c767bdabba5dfaf7d8826b0fd9badbd8014ba090a797006b3`
and verification image
`sha256:8d47780d6f48aa7f4ec9e41aafae8d2407411e6a371398c3a47faa517ac1996e`.
All 779 installed runtime-file hashes matched before and after, and all sixteen
changed module origins were under installed `site-packages`. There was no runtime
source overlay. The current native helper's C source identity and memfd interface
were checked; CUDA remained uninitialized and the source-foundation audit exited
zero. The installed gate and per-case source/reference/installed comparisons are
retained in `providerone-alignment-20261106/pp1-etp-v1`. The earlier empty-array
diagnostic-parser artifact is explicitly invalid and preserved as failure
evidence; the corrected comparisons require nonempty diagnostics for every case.
The installed gate SHA256 is
`46147f1f326b7bcce4eaecc8e8e9c1bacaae64d1f2532a83452e4a62ae48cb18`;
the strict source/reference/installed cause comparison SHA256 is
`435a8e7369faecfb97f121c7958149f729497a768bad30d461c7717b430cfdb1`.

Actual eight-V100 topology, imported allocations, full-model numerical/quality,
long-context capacity, complete-serving performance and all PP2 gates remain
unqualified. No gain or released target/MTP profile is claimed. Production is
unchanged.

## PP1 physical stage admission (A4)

Canonical SM70 TP8 GPU-history startup now checks the physical assignment before
initializing NCCL or admitting/loading model weights. The default gate observes
device zero in each rank process and requires one host kernel, ordered complete
eight-rank coverage, canonical distinct GPU UUIDs, the specified V100 SXM2 32GB
SKU and SM70 capability. Reported positive memory bytes are observed directly;
the nominal SKU is not turned into an arbitrary usable-memory floor.

A later source correction handles the pinned Torch 04cf CUDA property binding
`torch._C._CUuuid` at capture only. The first normal eight-rank SC117 attempt
refused before weights with `a bounded actual GPU UUID is required`; an actual
GPU0 property-only probe on image `b9585046` confirmed the exact exported native
type and its canonical bare UUID string. The adapter uses that formatter only
when type identity matches the trusted Torch export, then applies the unchanged
scalar validator. Arbitrary objects, type-name spoofs and native subclasses do
not gain string coercion. String/stdlib UUID forms and ordered/distinct physical
rank validation remain intact. These source/CPU controls do not establish a
successful installed successor admission or model load. The original failure
and probe are retained in the private `sm-v100-sc117-firstload-plan-v1` and
`sm-v100-uuid-property-probe-v1` packets. A subsequent target attempt reached MTP
warm-up but still stopped before readiness, as recorded below.

All SM70 TP8 ranks enter the same bounded CPU Store preflight, independently of
their local canonical/legacy admission choice. Mixed participation refuses
before ranks can split into CPU and NCCL waits. Uniform legacy participation
skips physical eligibility and returns no stage receipt. Smaller-world and
selected non-SM70 implementations keep their existing paths. The GPU-history
target has no new flag to disable the guard or silently fall back to legacy.

The existing startup-owned Store, agreed refusal/retirement phases and managed
cohort containment are reused. Capture errors are published before dependent
waits; malformed/oversized transport and uncertain cleanup are fatal. A semantic
refusal whose key retirement fails now preserves its primary failure, cleanup
failure and retained owner while marking the stage/draft/media preparation
cohort unsafe. No new listener, communicator or persistent key registry is
introduced. Managed replacements borrow their existing Store ownership.

Two phases plus retirement have at most 32 keys; even the existing bootstrap
allowance fits the current native key budget. Frames are bounded to 8 KiB before
decode, below the existing constructor value allowance. Expensive checkpoint
hashing, runtime inventory and topology commands remain opt-in. The optional
execution-profile path shares the canonical UUID/rank-map validator while
retaining its existing shared-file/container identity scope.

The default physical-host fingerprint uses the Linux kernel boot ID, not a
container hostname or machine-id. Two actual Docker containers on this GB10
host produced the same fingerprint with different machine-id bindings. Runtimes
that virtualize that kernel input need a separately qualified binding. The
receipt exposes fingerprints rather than raw boot/GPU IDs and remains separate
from the numerical identity used for prefix/copy/restore compatibility.

Source and fresh installed batches each recorded **1,458 passed, 37 failed and
one skipped** across 36 modules. Source took 28.90 seconds and installed took
29.61 seconds; both exited 1. All 37 failed cases have matching, nonempty
per-case diagnostics on pre-change references: the 29 previously disclosed
provider/`positions` fixtures, seven older graph fixtures missing `comm`, and
one native-media fixture that omits architecture metadata and patches an old
resolver owner. That media failure does not demonstrate a runtime mmproj
selection bug. These remain failures, not passes or equivalence. The optional
skip requires pinned companion/actual target metadata that was not supplied.

Focused controls passed 221 cases, including three actual CPU Store exchanges
for canonical, legacy and mixed participation, and six cleanup/refusal cases.
The Store exercise uses this host's SDK and test-only wait deadlines; it does
not certify the selected x86 native layout. Required deployment controls passed
194 cases and static checks passed on eight affected Python files. The focused
and integrated batches overlap and are not added into a unique-test total.

Fresh artifacts use source checkpoint
`14c2d24f929523c2e96eb5bb25e18f2a11b4e50c`, source UID
`fde67a33cde7625f6b2e4571a77b4ca878140937ce9a6eed48baee83ad5b2034`, runtime
`sha256:ad2e0f366a89bff6f8d7dfa3479773d1fb15f4fdb3b370a5f305a32199ae7c91`
and verification image
`sha256:0d658a6a412710958722a8495509ef7c3aea317f069083e22df1dae9b8c2e184`.
All 780 installed runtime-file hashes matched before and after; all four changed
module origins were installed `site-packages`. There was no runtime source
overlay. Current native C identity/memfd checks passed, CUDA stayed uninitialized
and the source-foundation audit exited zero. The exact logs, references and
scope are retained in `providerone-alignment-20261106/pp1-stage-binding-v1`.
Installed gate SHA256:
`18f321f8116df6c898e55249e692b3596df709cbacf43f4db1e3badac59c6ed7`;
strict source/reference/installed failure comparison SHA256:
`bb0703a89c02d82fa232aff35e76df3337e4eb8d64527c64dd768dc175edb0c7`.

Physical-target qualification remains pending. CPU metadata/protocol controls
and current-host CPU Store exercises do not
qualify the selected x86 ABI, eight actual V100s, the NVLink/P2P/PCIe/NUMA mesh,
complete model numerics, capacity or throughput. Unmanaged launches retain their
existing external failure-containment responsibility. Production is unchanged.

### SM70 FP32 MoE partial dispatch

A subsequent SC117 TP8/ETP4 shared-Q8 MTP attempt reached constructor warm-up,
then failed along `absorb → mtp_compute → layer_forward → moe_block →
glue.moe_partial`: its intentional FP32 slot storage selected the original
Triton partial, whose default `/home/ubuntu/.triton` cache was read-only.
The normal failure trace and operator retirement remain preserved. This is not
a completed readiness or model-quality result.

The original dispatcher and kernel remain unchanged: Half uses the existing
family implementation; FP32 uses the original Triton scalar kernel. The
unselected SM70-to-Torch dispatch candidate is archived as read-only evidence,
with its checks, and removed from canonical runtime. It would change operation
boundaries as well as launch count. The existing family reference/native
selector, its default, FP32 slot/partial storage, ordered TP8 gather, admission,
context, method and graph policy are unchanged. No Half cast is introduced.

The three-hop MTP trace confirms its input embeddings, HC streams, norms,
activations, projection/mix storage and attention operands are Half. Their glue
wrappers already select by those Half owners, including activation/write-back
operations that consume FP32 branches. Only non-prefill MoE slot storage is
intentionally FP32, including the method buffer. Existing target-verifier and
prompt constructors select Half MoE storage. MTP staging bounds the active row
count to its admitted buffer; the partial writes only that prefix. No additional
same-path dispatch patch was indicated.

An actual isolated GPU0 probe on installed image `d64f55ea` verified the existing
reference FP32 partial against an independent ordered oracle at R1/R7,
D2560/S11. It also executed a tiny generic Triton SM70 dyadic partial exactly
with a private writable cache and the real bundled CUDA12.9.86 assembler.
That genuine successful experiment is retained in `sm-v100-f32-moe-probe-v1`
(gate SHA256 `386b7e3c2d83d0da2bc8bcf0da84149193a85a2d6d5a31da53dab0e9664a512b`).
The broad [Triton 3.8 hardware matrix](https://github.com/triton-lang/triton/blob/v3.8.0/README.md#compatibility)
starts at compute capability 8.0; the
[3.2 matrix](https://github.com/triton-lang/triton/blob/v3.2.0/README.md#compatibility)
listed 7.0+. However, the official
[pre-Ampere MMA change and maintainer clarification](https://github.com/triton-lang/triton/pull/5066)
distinguishes removal of MMA-v1 from the FMA path, which remains maintained
with best-effort older-platform support and no upstream testing promise.
The earlier inference that the 3.8 README excludes every SM70 kernel was too
categorical. This loads/multiply/add/store kernel does not need tensor cores;
its actual successful compilation/execution must be assessed within that
specific maintained subset. It does not qualify unrelated dot/MMA kernels or
the complete model, and it does not justify a full-stack downgrade.

The generated original PTX contains `fma.rn.f32`: its fused rounding differs
from separate Torch multiplication/addition on sensitive inputs. Extended actual
R1/R7, D2560/S11 finite, cancellation, shared-slot and inactive-row cases pass
each implementation's own exact oracle. The original fused route is retained
for the unchanged-application cache-only retry. This preserves its numerical
contract and avoids an unnecessary application or full-stack rebuild. The
separate-rounding Torch path remains a valid reference, but is not equivalent:
for the sensitive case, fused gives `-2**-46` and separate operations give zero.
No tolerance is invented to hide that difference. The extended receipt is
`sm-v100-f32-moe-extended-v1/extended-f32-actual-gate-v1.json`, SHA256
`3568e218036c8daa67d8196ed16883774bdb1d7c6ecbfa4e047eb6f8c6b72653`.
Isolated latency results do not establish an integrated model-performance gain.

Maintained CPU controls retain independent scalar separate-operation and C99
`fmaf` fused oracles, order-sensitive cancellation and shared-slot cases, and
inactive-row bit checks. Dispatch controls preserve the original FP32 Triton
call and existing Half/native/reference roles. New canonical CUDA controls
exercise both selected operation policies; those new test nodes are unrun here,
separately from the actual private extended probe. No native-family promotion,
compiler flag forcing or architecture override is selected. Triton 3.2 remains
an optional future version profile, requiring its own compatible stack and
qualification; it is deferred rather than swapped into the current pinned SDK.

Use an owned host-backed `/cache` for `TRITON_CACHE_DIR=/cache/triton` and the
selected `TORCH_EXTENSIONS_DIR`, retaining entire trees and compiler/cache
metadata across restarts. In-image HOME is read-only and is not the cache
owner. Retire workers before clearing/replacing their loaded compiler artifacts.
See the [persistent-cache launch contract](../deploy/v100/README.md#tritonv100-compatibility-and-persistent-caches).
Normal readiness and later model gates remain required. No whole-model quality,
useful capacity or performance qualification follows from these operator checks.

## Restored fixture contracts (A5)

Four stale fixture definitions accounted for the 37 inherited failures above.
The repair changes test inputs, not runtime behavior or guards:

- Provider-format controls now use a real tiny GGUF target instead of patching
  an obsolete family resolver alias. Their existing early template sentinel is
  retained; three added negatives confirm SM70-only formats still reject a
  non-GGUF provider before reaching that sentinel.
- Allocation-only weights explicitly declare `positions=None`, matching the
  actual unbound-position interface. All allocation, ownership and budget
  assertions remain exercised.
- Single-rank graph fixtures explicitly declare `comm=None`. The real replay
  wrapper, event ordering, primary-error and no-successful-submission checks stay.
- The media fixture patches the actual neutral resolver and supplies the
  required `qwen4exp` architecture metadata. The real family architecture guard
  runs, the selected entrypoint is asserted, and vocabulary/role plus stale
  tokenizer-override checks remain intact.

All 115 original assertion nodes are retained. The affected four-module batch
passed 563 cases in 6.27 seconds. The complete scoped 36-module source batch
passed **1,498 tests with one optional skip** in 27.47 seconds; the installed
batch passed the same cases in 27.32 seconds. Both exited zero. The skip still
requires optional pinned companion/actual target metadata that was not supplied.
Required deployment/public-permission checks passed 194 cases, and static
checks passed on all four changed Python files. Overlapping batches are not
summed into a unique-test total.

All 783 runtime-source files remain byte-identical to the A4 built capture.
No rebuild or redeployment was performed. Installed verification reused runtime
`sha256:ad2e0f366a89bff6f8d7dfa3479773d1fb15f4fdb3b370a5f305a32199ae7c91`
and verification image
`sha256:0d658a6a412710958722a8495509ef7c3aea317f069083e22df1dae9b8c2e184`,
source UID `fde67a33cde7625f6b2e4571a77b4ca878140937ce9a6eed48baee83ad5b2034`.
All 780 runtime hashes matched before and after; module origins stayed installed,
with no runtime source overlay. Current native identity/memfd checks and the
foundation audit passed; CUDA remained uninitialized. Test hashes, commands and
logs are retained in `providerone-alignment-20261106/fixture-contract-repair-v1`;
the installed gate SHA256 is
`853a06bd183ab56f247285a8378a2b7a461bc5fd04beb06a158e102783b59b6f`.

The historical 37 failures and their nonempty reference diagnostics remain
preserved at their original checkpoints. This restores meaningful CPU software
coverage; it does not retrospectively turn those failures into passes or
establish actual V100 topology, full-model quality, capacity or throughput.
Production is unchanged and the full ProviderOne delivery gates remain open.

## Private PP2 stage components (A6)

The GGUF loader has a private typed `_pipeline_stage=PipelineStage(index)`
component interface. The fixed global 48-block target is split into blocks
0–23 and 24–47, with TP8 and the same selected ETP8/4/2 degree, with groups
confined to each stage.
Planning and materialization retain only the selected stage's coefficients;
the explicit complementary inventory remains checked for missing/unknown
tensors. Packed encodings, row/column offsets and global `LayerW.index` values
are preserved. Global Config layer identities are not replaced by 24 local IDs.

The input stage alone owns the target embedding and PLE lookup/table; the output
stage alone owns final mixing and the target head. The current SC117 PLE is
`blk.1`. Later-stage PLE placement is refused until an owner-row service exists,
rather than duplicated or dropped. A real sparse IQ4_NL table control checks
input-stage lookup/source words and the unchanged global ngram ordinal; it is
not a full-table hash or model-quality proof. Partial import refuses speculative
method attachment until the required cross-stage endpoint/service contract is
implemented. Stage-local expert-partition metadata is explicitly scoped apart
from the global PP2 stage identity.

Private `compute_stage` executes the owned local layers with the existing
`layer_forward`, writeback and final readout operators. The first stage applies
its last pending MoE update and exports borrowed contiguous `[rows,4,hidden]`
FP16 HC, without a mixer/head. The caller must retain that storage until its
handoff completes. The second stage consumes all four streams in the exact
staged dtype/device/layout, then performs final readout. A single hidden vector
or stream mean is refused. Invalid post-preparation inputs notify the existing
input-preparation owner before cache/operator work. Captured execution and
draft-feature transport are not enabled by this component.

State uses 18 local recurrent and six target attention caches while retaining
absolute global layer IDs. Numerical cache identity includes the stage's global
block span, so equal local shapes cannot authorize cross-stage copy/restore.
The input stage allocates no target-logit or candidate storage. Optional padded
projection allocation and its forecast use 18 local layers without changing
the global 48-layer Config. Actual coefficient and host-table ledgers come from
the owned plans, not a division of the full model's runtime buckets by sixteen.

The complete scoped 42-module source and fresh installed batches each passed
**1,600 tests with one optional skip**. Source took 31.74 seconds and installed
took 32.84 seconds; both exited zero. The skip requires optional pinned
companion/actual target metadata that was not supplied. Required deployment
controls passed 194 cases, and static checks passed on nine affected Python
files. Component checks passed 165 cases with the same optional skip, while
44 existing padding/HC/cache controls pass, including a repaired stale AST
lookup that now examines the active Engine's actual geometry wrapper. These
overlapping batches are not added into a unique-test total.

CPU composition uses an independent NumPy Half-storage oracle and deterministic
stand-in layer operators while retaining actual writeback; it does not qualify
CUDA operators or full-model numerics. Sensitive writeback/storage tests cover
the branch, routed-slot and rank/K-slice modes. Actual CPU State/buffer controls
exercise local cache maps, cross-stage restore refusal and padded projection
allocation versus forecast. The legacy `compute` body remains unchanged.

Fresh artifacts use source checkpoint
`8391b599aebc5f6c9af9779e4df0fa8b25d1f663`, source UID
`298eccd1f28f69a7627d799566503b997f480d5ea4d954cdeefa3c24b29a728c`, runtime
`sha256:aad3c1c212c78c097a97b51cffba53b10434df97d64ba9db9ce7080ad9fc4ecd`
and verification image
`sha256:7508c491bcba9dedab06bd9a89ea9131d8a3a11c86081ae0101bae35e319917c`.
All 781 installed runtime-file hashes matched before and after; all five
changed module origins were installed `site-packages`, with no runtime source
overlay. Current native identity/memfd and foundation checks passed; CUDA stayed
uninitialized. Exact commands, test inputs and failure/remediation history are
retained in `providerone-alignment-20261106/pp2-stage-boundary-v1` and the linked
loader packet `pp2-stage-execution-v1`. Installed gate SHA256:
`8c872b7c18feebd41e188340ed0eb6e110f43356281176649db656248266c0f5`.

This is necessary executable groundwork for the required PP2 path, not complete
PP2 inference support. There is no public PP2 Engine/CLI flag. Bounded cross-node
HC/metadata transfer, progressing embedding/lookup/feature services, final-stage
MTP/DFlash, one unresolved speculative round and coordinated commit/rollback,
RNG/frontiers/cancellation/publication remain required. Complete stage resource
admission and actual two-server mesh/RDMA, model quality, capacity and performance
qualification also remain open. Production is unchanged.

## Private two-leader HC transfer (A7)

The private `pipeline_transfer.HCTransport` component exclusively owns a dedicated
two-rank NCCL communicator and its Store. It does not borrow the stage TP8
communicator or expose a public PP2 Engine/CLI flag. Both endpoints submit each
attempt, including refusals, and agree on the complete bounded ordinal,
profile/epoch, sequence generation, round, row count and token/media/position
identity references before any native payload operation. Two alternating key
banks avoid an ever-growing native Store key namespace; coordinated retirement
prevents premature bank reuse.

The payload is exact contiguous FP16 `[rows,4,2560]` HC, followed by int32
`[rows,3]` positions and int32 `[rows]` token IDs. No HC mean or conversion is
introduced. The receiver reserves `slots * max_rows * 20496` tensor bytes;
NCCL internals, streams/events, allocator overhead and transport registration
remain separately unqualified. Sender storage is borrowed and must remain
read-only until the send-completion lease permits reuse. Receiver storage cannot
be consumed before transfer completion and a subsequent asynchronous-error
check, and cannot be reused before consumer-stream completion or a completed
discard. The ring permits only one unresolved round per sequence generation.
Counter exhaustion refuses new work while allowing existing leases to retire.

Unknown native/event/consumer failures quarantine the owner. Primary errors,
cleanup failures and previously retained owners remain available for diagnosis;
resources are retained until successful communicator abort and device retirement.
The existing launcher notification is entered once on poison when configured;
it is not a cross-server supervisor. Stream/device restoration occurs only after
successful scoped work, avoiding additional CUDA calls after an unknown fault.
Partial constructors publish acquired buffers into retained ownership slots.

Deterministic CPU boundary doubles exercise identity/refusal, finite ring and
key reuse, producer/consumer ordering, sentinel HC/metadata routing, cancellation,
stale leases, partial construction and failure retention. These tests are
component controls, not actual CUDA/NCCL transmission or model numerics.
The actual local verification SDK stream implementation was inspected for the
error-restoration contract; this does not qualify the separately pinned V100 SDK.

The final scoped 45-module source batch passed **1,670 tests with one optional
pinned-companion/target-metadata skip**, in 33.28 seconds. Required deployment and
public-permission controls passed 194 cases; affected F/py312 and configured Ruff
checks passed. The batch uses canonical Python source with the freshly built,
unchanged installed native extension and is distinct from installed-image
qualification. CUDA remained uninitialized. Earlier focused controls passed 68
cases before the final two capture-precheck regressions were added. Saved failing
controls, remediation history and exact commands remain in
`providerone-alignment-20261106/pp2-hc-transport-root-v1` and
`pp2-hc-transport-execution-v1`; they are not fresh-checkout test dependencies.

The fresh installed-image batch passed the same **1,670 tests with one optional
skip**, in 33.48 seconds, with CUDA uninitialized. All 782 installed runtime
files matched the frozen source before and after; the new transfer module and
native extension resolved from installed `site-packages`, with no runtime source
overlay. Foundation audit and current native source/memfd checks passed.
The source checkpoint is `01751a062ea4449eb1c2d035d4cbf0061a2bf5bf`, source UID
`6f39b2b73884a2422fc1078c20262c896555814a28a37d3adf09c960946cc106`.
Runtime image:
`sha256:a565853c46f50b776ba7110cf4d855e1909a3e82762453adb23c2112f95c1bdc`;
verification image:
`sha256:d46a83e54ad789c5a5086176d3ea07f86dd87ca491404182f943c5977f77a459`.
The final documentation-only checkpoint does not change those runtime inputs.

TP8 fanout, progressing embedding/lookup/features, method endpoint ownership,
distributed round commit/RNG/frontiers/cancellation/publication and complete
per-stage resource admission remain required. Actual two-server mesh/RDMA,
CUDA transfer, SC117 quality, capacity and performance remain unqualified.
Production is unchanged. No further candidate investigation is part of this
closeout, following the owner's instruction to finish current work and stop.

## TP8/ETP4 one-server native component

The owner prioritized TP8/PP1/ETP4 before further PP2 work. A3's reference
target/shared-GGUF MTP path already partitions experts into two groups of four
ranks. Each group owns 256 experts, with local routed width160; shared experts
retain TP8 width80. Global router IDs, one representation per expert channel
and the existing ordered full-TP8 FP32 partial gather are preserved. Selected
target/prompt buffers store FP16 expert slots before FP32 weighted reduction;
the dedicated MTP buffer uses FP32 slots. Those storage boundaries are distinct
from the subsequent reduction dtype.

An explicit `GGUFExperts.forward_partitioned_native_candidate` component now
uses device-side global-to-resident route mapping and existing packed native
arithmetic, rather than host scalar checks or dynamic `nonzero` compaction.
It publishes literal zero for foreign slots and the independently evaluated
TP8 Dense shared result for global sentinel512. Invalid global IDs produce the
existing native diagnostic NaN contract without an out-of-range bank read;
they are not repaired into a valid expert. This is a component candidate, not
selected serving dispatch or an established speed/quality equivalence result.

The native component supports direct FP16 and FP32 caller outputs. Shared
FP32 results round once to FP16 when the caller owns FP16 slot storage, using
the existing typed down-kernel paths and a typed publication store. No extra
full-slot FP32 temporary or copy is required merely to bridge output dtypes.
Native arithmetic/model equivalence still requires its separate device gates.

Callers supply reusable int64 routes and FP16 activations, separately owned by
target and MTP roles. The native scratch forecast is
`rows * slots * (8 + 2 * 160)` bytes, with at most 65,535 slots per call. Caller
slot output and shared Dense temporaries remain separately charged. No per-bank
resident scratch is introduced. Integrated target/MTP worst-rank peak admission
and a legal prefill policy beyond that call envelope remain selection gates.
The component does not enable whole-model graphs: unique-history live-row
execution has its own independent capture restriction. Existing MoE activation
storage is a possible integration reuse, not an extra allocation promise.
Reference remap reserves cannot automatically fund persistent native arenas
while a reference fallback remains reachable; prove an exclusive prepared
provider or charge the overlapping lifetimes before reusing those reserves.

The V100 Compose profile now forwards an explicitly set
`TF_EXPERT_TENSOR_PARALLEL_SIZE=4` through its existing Bash argument array to
the server's existing option. Unset/empty behavior is preserved, including legacy
profiles; the CLI remains the authority for supported degrees and ownership
compatibility. Ten actual Compose-render/argv controls pass, including degree
8/4/2 forwarding and literal malformed-input rejection without shell execution.

New SM70 component controls cover global boundary IDs, both owner groups,
int32/int64 routing, foreign-only groups, tails, mixed source formats, poisoned
outputs/scratch, changed-route graph replay, alias refusal and separate role
scratch. Their exact native arithmetic comparison reuses the existing packed
provider to test the new mapping/publication boundary; it is not an independent
full-model numerical or distributed-collective oracle.

The typed-output four-module focused CPU batch passed 129 cases in 1.57 seconds.
The scoped 47-module source batch passed 1,692 cases with one optional pinned-
companion/target-metadata skip in 32.96 seconds, including all A7 regression modules.
Required deployment/public-permission controls passed 194 cases. Configured
Ruff, affected F/py312 checks and Python compilation passed. CUDA stayed
uninitialized; Python compilation does not compile or qualify the native CUDA
entrypoint. These are canonical-source checks using the freshly built unchanged
native FD extension, distinct from a fresh installed application check.

The earlier FP32-only component at `033a0509` and its 1,691-case source/installed
results remain historical evidence in `tp8-etp4-execution-v1`. They do not
qualify the subsequent typed-output revision. Current commands and results are
recorded separately in `tp8-etp4-execution-v2`.

The fresh typed-output installed application passed the same 47-module batch:
1,692 passed with one optional skip in 38.65 seconds. All 782 installed runtime
files matched the 785-source-file frozen capture before and after, excluding
three source README files from runtime installation. Both changed Python module
origins and the native FD extension resolved from installed `site-packages`,
without a runtime source overlay. Foundation audit and native FD source/memfd
checks passed; CUDA stayed uninitialized. This uses the GB10 SDK for CPU/package
proof and does not compile the changed SM70 CUDA operator.

The typed source checkpoint is `6b6f7756d8e1d9a093f1f3a39c9306174374d511`,
source UID `6cce8b14fd79f544f0bc64205c85e0780d80504880dab05c40eedaed375cbcba`.
Runtime image:
`sha256:5e8f4b85c8e934c5161a2003b271b0c62b9ac3e500f3badcb15a0d1611aa0b79`;
verification image:
`sha256:972e911a4ce282f61f7a05e874a6204dc0da81c9799d30edec4307c89e60b249`.
Subsequent evidence-only documentation changes do not alter those runtime inputs.

At that component checkpoint, actual SM70 compilation/execution, canonical
SC117/MTP model and media quality, complete resource/capacity and matched
end-to-end performance were unrun. Subsequent native AMD64 editable-build,
CPU-load and offline SM70 component compile/load evidence is recorded in the
[fresh-bootstrap report](../deploy/v100/source-foundation.md#executed-fresh-bootstrap).
Those earlier GB10/dev-01 results do not provide actual V100 execution or a
selected final V100 SDK. Access to `sm-v100` is now available; the subsequent
standalone component campaign below records its own physical evidence. It does
not execute the native expert operator or promote a serving candidate.

### Installed SM70 SDK and native ETP4 component

On 2026-11-07, the genuine final source build/export from
`e90caf5879f0eac5566c9d9aae85e4595be51e00` produced immutable AMD64 image
`sha256:a24a579e3ffd2a1ead0def6b0d68947d94796135c00c946e8440ebe1f47c5015`.
The installed Torch is `2.14.1+tfcu129sm70`, source
`04cf2557f315a350e7eea7ab064288def65db75a`, CUDA 12.9/native SM70 and CXX11 ABI 1.
The genuine wheel SHA256 is
`d2e7eb3ffb8120544e2d39b7b031e37ba322268cb0d58f54b532a568c2595a86`;
the protected read-only installed manifest SHA256 is
`4854c6356624834838cc945ff1c534af8fb0220ad7c018ad673dc9ecdad78484`.
The installed software audit and both named runtime CPU tests passed, including
a separate read-only UID1000 run. The earlier interrupted Ninja generation
remains quarantined and contributes no final-build qualification.

The same installed image then ran
`tests/cuda/test_sm70_gguf_etp.py` on `sm-v100` GPU0, driver 580.178.04,
through the normal guarded `gguf_ops` extension loader and protected selected
headers. There was no editable source overlay or diagnostic loader bypass.
Actual Ninja 1.13.2 used `-j24`; NVCC 12.9.86/GCC 13.3 built native SM70 code
with the recorded production flags. All **49 cases passed**, with zero skips,
failures or errors, in 4.061 pytest seconds after preparation. They cover the
component's packed arithmetic, FP16/FP32 outputs, route mapping/publication,
foreign and invalid routes, graph replay, scratch contracts and D2560 shapes.
This is component correctness evidence, not model throughput.

Normal extension preparation took 47.745 seconds. The pre-fixture cumulative
host cgroup memory peak was 3,445,723,136 bytes, including Torch/CUDA discovery,
header auditing, compilation, linking and loading. It is neither GPU memory nor
a phase-exclusive peak, minimum RAM requirement or worst-case bound. The
container had a 24-CPU quota, 256 GiB memory limit and no swap; runtime OMP was 1.
Source/header identities and protected inputs matched before/after; the process
audit passed, GPU health counters were unchanged and all owned consumers retired
with exit 0/PID 0 and no remaining compute applications.

Private receipts under `providerone-alignment-20261106` retain raw outputs,
binary/SASS/tool/header identities and retirement evidence:

| Receipt | SHA256 |
| --- | --- |
| `sm-v100-final-sdk-plan-v1/canonical-e90-installed-sdk-cpu-pass-gate-v1.json` | `68e75f62241163ba20f68ac251ce9761b1c7c105b5ab1c3fc44bad2a032a9268` |
| `sm-v100-native-etp4-plan-v1/native-etp49-actual-pass-gate-v1.json` | `d99d670b2496877f667a663694d03d4cf167f33289a2d808892ecaa9f7529fac` |

The same immutable installed image subsequently passed the unchanged installed
four- and eight-rank collective fixture: two cases, zero skips/errors/failures,
14.956 pytest seconds. All 12 actual worker PID/rank tuples matched their raw
loader traces and NCCL initialization records. The 64 initialized provider paths
retain separate immutable-image and host-driver provenance. Protected software
audits, headers, source/image identities and GPU health stayed unchanged; no
application JIT was produced. Both the test and provider-readback containers
retired with exit 0/PID 0 and no remaining compute applications. The private
`sm-v100-protected-collectives-plan-v1/protected-collective-actual-pass-gate-v1.json`
receipt, SHA256
`0a2075c465ae74c4d68cba88003384bee3b9eb20478afed7fe8f1c3b14de8f30`,
binds 46 independently rehashed artifacts. Optional IB/RMA/GDRCopy/GIN/NVLS
advisories remain recorded; successful Socket bootstrap/P2P-CUMEM execution
does not qualify those unavailable capabilities or NCCL throughput. Two caught
prelaunch operator defects were repaired before execution and remain historical
evidence; they were not GPU test failures. The tiny scheduler is a separate gate.

The native gate binds 30 direct artifacts, independently rehashed during review.
`release_qualified` remains false. Distributed ETP4, SC117/MTP/DFlash and media
quality, complete model/state allocations, full-window capacity and matched
end-to-end performance remain separate gates. Synthetic success does not close
vendor issue6111901 applicability. The reference serving path stays selected.
Later source commits do not retroactively change this image's identity.

### Physical standalone component campaign

On 2026-11-07, the maintained primitive/library/reconvergence probes were staged,
compiled and executed on `sm-v100` (`192.168.1.140`), Ubuntu 26.04.1 with driver
580.178.04 and eight V100 SXM2 devices. Source checkpoint
`c0b11f0e3c0db4f9292150f58c4d117db62ee84b` and image
`sha256:9344978748e6eac9b9323b332b2feadb71e151ed217e0984466161f1a24dc451`
bind this campaign. The unmodified `source_build.sh probes` stage used
NVCC 12.9.86/GCC 13.3 and produced these three standalone binaries:

| Binary | SHA256 |
| --- | --- |
| `cccl-primitives` | `a13aa016faf3d199f15392b4918d5ba5b7dff4a4f7250d010267efe7396d698e` |
| `libraries-probe` | `186d8a204d919c82d0ee0691b8eb1abde59f92c0e682243503e4e92ddf287c6c` |
| `reconvergence-6111901` | `77af74f98098189264ed6e228d8205c1bb4d5b011f7edadc9ed11009ba15db6c` |

Each of the eight devices passed five ordinary runs: CCCL at 1, 33 and 1,048,576
elements, the library probe and the synthetic reconvergence probe. The CCCL
runs exercised eager and captured execution with caller scratch of 1,023 bytes
for the two smaller sizes and 8,485,375 bytes for the largest. The library probe
passed eager/captured exact integer comparisons with cuBLASLt algorithm 0 and a
32 MiB workspace ceiling. The reconvergence probe passed 3,072 synthetic cases
per device. These are fixture correctness observations, with no performance
claim or vendor-issue applicability inference.

GPU0 additionally passed memcheck, racecheck, initcheck and synccheck on each
of the three binaries: 12 sanitizer jobs. All 40 ordinary and 12 sanitizer jobs
exited 0 and retired with PID 0; final checks found no running containers or
compute applications. Recorded GPU health counters remained unchanged,
including GPU5's pre-existing single-bit retired-page count of 1. Source/prefix
identities matched before/after, every job checked the exact binary SHA256,
and canonical/protected inputs remained unchanged. Actual library initialization
paths matched GPU0 across all eight library runs; host-driver and container
provider identities retain their distinct provenance.

The private `sm-v100-component-plan-v1/all8-component-gate-v1.json` receipt has
SHA256 `3082ccfc0b728bd2db511fcece386af3ebdcb765b78cdb58e5c444c093ebe04e`.
Its 195 direct artifacts retain commands, raw outputs, before/after identities,
provider observations, health and retirement evidence. These 52 jobs are serial
single-device fixtures, not a multi-GPU/NCCL run or native `gguf_ops` execution.
SC117/MTP/DFlash model/state/quality, resident capacity, performance, final SDK
qualification and vendor issue6111901 disposition remain open. No runtime
selection, guard or production setting changed. See the
[source-foundation gates](../deploy/v100/source-foundation.md#required-primitive-and-library-gates).

### Collective fixture Store backend

The existing four/eight-rank `test_flashnext_sm70_collectives.py` fixture now
selects `use_libuv=False` explicitly, matching the adaptive SM70 engine. Exact
pinned Torch `04cf2557` source/configuration evidence shows
`USE_LIBUV:BOOL=OFF` in the actual CMake cache and `use_libuv=True` in the native
`TCPStore` constructor binding. Omitting the argument would therefore select an
unavailable backend before any collective oracle. That backend mismatch was
established statically; no failed backend execution is claimed.

After opening the communicator, every rank uses the supported
`verify_store_cohort()` and `validate_store_origin(..., verified=True)` APIs.
The fixture asserts actual legacy-backend provenance, guarded rank0 daemon
ownership and completed cohort agreement. This follows the existing Python
socket/`master_listen_fd` handoff; it does not attribute that handoff to the
native FD extension. FP16/FP32/int64 exchanges, status/error and EOS agreement,
side-stream graph replay, world sizes 4/8 and communicator/process retirement
checks are retained. No fallback, environment default, skip or production guard
was added or changed.

The private `sm-v100-collectives-plan-v1/store-backend-prerequisite-v1.json`
receipt has SHA256
`197be099e618d77b362b71c3d1a3c42a9ceab6c79614ce02bc93b22520187f6c`.
Its versioned source/cache excerpts and original implicit-backend diagnosis
remain preserved. That diagnosis alone was not a collective execution pass.
The first subsequent physical four-rank attempt failed when the synthetic
`object.__new__(State)` status fixture omitted the constructor's noncompact
`compact_indexer=None` default. The eight-rank case correctly remained unrun
under `-x`; no complete four-rank pass was claimed. Raw console, XML and all four
rank traces are preserved by `collectives-gpu-failure-gate-v1.json`, SHA256
`6cf4e09b25f4ad41a5fbb7d903a903a5bcf4a88607e892b8e491a9732e1b9aba`.

The GPU fixture and its independent CPU per-sequence/status-cohort oracle now
each initialize that actual noncompact default. Their assertions and production
methods remain unchanged; no shared helper or error masking was introduced.
On the staged native editable SDK, all 42 CPU controls passed with no skips,
followed by both corrected physical GPU cases at world sizes 4 and 8, also with
no skips. Across those two cases, 12 worker processes exercised actual guarded
legacy Store/cohort provenance, FP16/FP32/int64 all-gathers, codec error/status
and EOS agreement, nondefault streams, graph capture/replay and communicator/
process retirement. Health counters and recorded inputs remained unchanged.

The accepted diagnostic source UID is
`dc4a9dd3d11381635d0a23421edae6290353d783936afd66488e06dd30de6dd9`:
base `8db10092` plus held fixture/documentation WIP, not a clean source commit.
Application runtime bytes were unchanged. Execution used editable Torch
`04cf2557f315a350e7eea7ab064288def65db75a` (`2.14.1+tfcu129sm70`, CUDA 12.9),
NCCL 2.32.3, image
`sha256:9344978748e6eac9b9323b332b2feadb71e151ed217e0984466161f1a24dc451`
and `sm-v100` driver 580.178.04. All 12 worker rank/PID NCCL and loader
initialization traces are retained. Provider files were hashed separately from
the same immutable application/SDK/prefix; these are not worker memory hashes.
Host-driver injection sources retain separate identities. No TensorFold native
JIT test was selected, and loading a driver PTX-JIT library does not establish
application JIT execution.

| Receipt under `sm-v100-collectives-plan-v1` | SHA256 |
| --- | --- |
| `incremental-v2/cpu42-gate-v2.json` | `ca8e504f407752ba6350d4bcf2447dd144eb02d5311bdca2a970a9ffbb842320` |
| `incremental-v2/collectives-gpu-pass-gate-v2.json` | `7f78a75c26e5816044062ce6acc60b83948c19d1cdf12cdae311e04533ffb51f` |

Retained NCCL logs report unavailable optional plugins, libibverbs/IB, RMA,
GDRCopy, GIN and NVLS paths. The exercised single-node configuration initialized
Socket bootstrap and P2P/CUMEM channels, and its component oracles passed.
These advisories are retained, not suppressed; they establish no IB/RMA/GIN/NVLS
capability or performance result. No transport override or dependency change
was made to obtain the pass.

The passing GPU gate binds 49 artifacts, with full input preservation and
retirement records. This is source/editable communication component evidence,
not an installed final SDK, NCCL throughput or performance result.
Native ETP4/model/state/quality/capacity and vendor issue6111901 qualification
remain separate open gates; no production selection changed.

## Prepared TP8/ETP4 execution and workspace

The existing engine has a private constructor-only `_owned_expert_native=True`
qualification path; normal CLI, Compose and administrator JSON do not expose
or select it. It requires adaptive resident GGUF TP8/ETP4 on actual SM70 with
stock shared-GGUF MTP, dense TP8 DFlash or no method. Private native mode with
administrator profile transitions or DSpark remains explicitly unsupported;
ordinary reference profiles retain their existing contracts. DFlash composition
has separate pairing, resource and feature-transaction controls below. This
staged qualification scope does not remove unimplemented or unqualified
requirements from the governing delivery goal.

None/MTP requests join common rank-mode agreement before private eligibility
checks. DFlash prepares private facts and votes failures inside its existing
matched CPU pairing preflight before loading model payloads.
The immutable plan validates the actual routed/shared geometry and maximum
decode/prompt row-slot extents before loading. A native call remains bounded
at 65,535 pairs; an unsupported private envelope refuses preparation instead of
truncating work or silently changing providers. Whole-model graphs remain off.

Each existing `Buffers` owns one int64 `[rows,slots]` local-route arena only in
this private mode. Native activation storage is a contiguous view of the
already-owned FP16 `moe.act`; output uses the existing FP16 target/prompt `y`
or FP32 dedicated MTP `y`. The same prompt buffer also serves MTP absorption.
No per-layer arena, duplicate activation bank or full-slot FP32 copy is added.
Actual target, prompt and method arena sizes are reconciled before warmup, with
shared twin/slot storage remaining shared.

Let `S=top_k+1`, `R` be the actual decode-buffer capacity, `P` the actual prompt
capacity and `m` be one when stock MTP is resident. Local arenas occupy
`8*S*((1+m)*R+P)` bytes simultaneously. The unchanged target/method reference
routing reservation is `(1+m)*52*S*max(P,R)`, which contains these allocations
for stock MTP and target-only native configurations (no method or dense DFlash).
This reuse is valid because every prepared target,
prompt and MTP call uses the native owner and bound banks reject reference
compaction. A reachable fallback would require a new overlap proof or charge.
Existing decoded/shared Dense, state, recovery, history and floor reservations
are retained. The resource receipt reports positive materialized local bytes;
zero additional device bytes means no increment to the already-paid reservation,
not zero physical allocation. No extra resident session or performance gain is
inferred from this containment proof.

Rank settings and cache/recovery identities bind the native arithmetic policy.
Independently observed execution-manifest numerics and the coordinate activation
contract also record the actual weight-bound policy, so reference qualification
facts cannot authorize native arithmetic. Default reference facts and identities
remain unchanged when the private policy is absent. Different prepared native
envelopes remain separate identities until cross-envelope reuse is qualified.

At the MTP/none checkpoint, source controls passed 213 focused tests in 2.48
seconds and the complete scoped 49-module batch passed 1,735 tests with one optional pinned-companion/
target-metadata skip in 33.47 seconds. Required deployment/public-permission
controls passed 194 cases; affected F/py312, configured Ruff and Python compilation
passed. Canonical Python source used the freshly built unchanged native FD
extension; CUDA stayed uninitialized. These overlapping batches are not summed
into a unique-test total and do not establish installed-application or CUDA proof.

Controls cover actual buffer allocations/views, arena reconciliation and default
identities, explicit mode/rank refusals, exclusive MoE call dispatch, rejection of
old execution qualification facts and eight-participant coordinate contracts.
Earlier ABI/store prerequisites in the single-participant positional unit fixture
are explicit CPU stand-ins; its assertions remain intact. The real selected ABI
guard and startup transport are not qualified by that fixture. Raw initial
failures, bounded hang diagnosis, corrections and final command/results remain
in `tp8-etp4-prepared-v1` and `tp8-etp4-prepared-execution-v1` under the private
`providerone-alignment-20261106` audit root.

A fresh installed application passed the same 49-module batch: 1,735 tests with
one optional skip in 34.87 seconds. All 783 installed runtime files matched the
786-source-file capture before and after, excluding three source README files.
All eight changed module origins and the native FD extension resolved from
installed `site-packages`; no runtime source overlay was used. Foundation audit
and native FD source/memfd checks passed; CUDA stayed uninitialized. This is
GB10-SDK CPU/package evidence and does not compile the SM70 CUDA operator or
qualify the selected V100 SDK ABI.

The source checkpoint is `fcbfe518094ca09217eda219c1384b10c6b59d5d`, source UID
`a19cd0c14c3eac67e25b7a372dae2a0856f1025fc1e939edc611a3f6f948710d`.
Runtime image:
`sha256:a9baaea79550be9eff52c66d95a215fd25083dbfd2660cf84a9030fbf734f0d6`;
verification image:
`sha256:527336c3b958177e189a6ceb4c2190137fd7020efdd2be76d17dca5e5bfd8d74`.
The installed gate is `tp8-etp4-prepared-execution-v1/installed-gate-v1.json`,
SHA256 `c5741f6f3dbdfdf07869a2fd2369a8a8e2089afe5007f974471252bd2bf6d3c3`.
The final evidence-only documentation checkpoint leaves those runtime inputs
unchanged.

Source-level integration and CPU controls do not qualify actual SM70 execution,
numerical quality for SC117 and its selected method, visual inputs, worst-rank
physical peaks, full-context
capacity or matched useful performance. V100 access and the selected SDK remain
unresolved. No native production promotion or other service change is made here.

### Private native target with dense TP8 DFlash

The existing reference ETP4+DFlash path is preserved. The private native
candidate now composes with that dense TP8 draft implementation. Native plan
construction runs inside the existing voted CPU pairing preflight, validating
the local rank/world, hidden geometry, bounded envelope and target-only method
role before hashing model payloads or CUDA allocation. Its immutable arithmetic
facts shared by all ranks enter the pairing numerical contract. Reference pairings and
mixed/invalid rank facts refuse through the existing matched retirement path;
the prepared native plan is published only after pairing and retirement succeed.
Later loaded-bank and arena bindings must still match those prepared facts.

Dense DFlash has no expert axis. The target and prompt buffers own the two
native route arenas with `m=0`; no MTP arena is added. Draft windows, HC feature
frames, selector state and their reservations remain separate. CPU integration
controls exercise actual dense TP8 draft shards, windows, target state and
feature commit/cancel/restore, with explicit native-expert and collective
boundaries. These boundary spies provide no CUDA arithmetic proof.

The sealed composition passed 168 focused CPU tests, the expanded 54-module
batch with 1,850 passes and two optional large-artifact skips, and 194 required
deployment/public-permission controls. Affected F/py312, configured Ruff and
Python compilation passed. Source hashes stayed unchanged and CUDA remained
uninitialized. This run used the existing ARM64/GB10 Torch 2.16/cu134 verifier;
native AMD64/Torch 2.14 and fresh installed-package results are separate gates.
The initial AST fixture import-context failure and read-only-cache failure are
retained. Their corrections supplied real module context and a private bounded
writable cache without weakening assertions or changing library code.

The source gate is `tp8-etp4-dflash-execution-v1/source-gate-v3.json`, SHA256
`d3f09fab338aa87376d5f5b927784730b6b086189f152f9ea3a45b31514c3923`.
No public opt-in, device admission, model pairing quality, capacity or useful
performance is qualified by these CPU controls.

The subsequent native AMD64 checks used the actual finalized read-only
Torch `2.14.1+tfcu129sm70` stack: six focused modules passed 168 tests from
source and 168 from the freshly installed project wheel, in 5.05 and 5.12
seconds respectively. Python, pytest, isolated dependency, Torch/NumPy and fresh
native-FD origins were checked; CUDA remained uninitialized. Twelve existing
hash-locked wheels supplied the isolated test dependencies without replacing
SDK packages. Full native media coverage remains unfinished: its maintained
source-pinned media stack cannot be replaced by the older GB10 Transformers
closure, whose HF Hub requirement conflicts with the selected SDK dependency.

Fresh guarded GB10 runtime and verification images were built. The verification
image passed the installed 54-module batch: 1,850 tests and the same two optional
artifact skips in 35.92 seconds. All 779 tracked runtime-file hashes matched before and after, the
changed modules and native FD loaded from installed `site-packages`, normal
foundation audits passed, and no runtime source overlay was used. All 2,277
captured inputs and the fifteen protected baseline files remained unchanged.
The optional missing large model/companion cases remain unqualified. A retained
native pytest advisory names the preimported AnyIO provider only; optimization
was zero and test assertions were not disabled or suppressed.

The historical 783-file package included four untracked `b16_lossless` prototype
files. The current tracked capture has 782 source runtime files, with the same
three source README exclusions yielding 779 installed files. Actual membership
comparison found only those four prototype omissions. They remain unchanged
in the canonical workspace as protected independent oracles; all four tracked
Delta4 backend files are included in the new package. No production image was
replaced.

The immutable runtime input is
`ae71cbe6+tracked-wip.b7c33f42cf78264a`, source UID
`b7c33f42cf78264a53e56c916575fdfde2ffb99669f595c09169b8703f65b296`.
It records the held working-tree bytes rather than claiming a clean commit.
Runtime image:
`sha256:fe87f4d5e5ce210991a1780f1601abf46c0c8a0a98f169864b8fbe2cb7f6ee10`;
verification image:
`sha256:f3a65e6f33edc806b583a1fbfb82c81c012a29bb7ee6614be6c9a916172da3ce`.
Inherited `org.tensorfold.*` image labels describe the qualified base lineage;
the current application uses the recorded deployment source UID/revision.
The final evidence-only validation-record update changes no packaged runtime
or root README input.

| Receipt in `tp8-etp4-dflash-native-v1` | SHA256 |
| --- | --- |
| `native-focused-gate-v3.json` | `971a2706cdc905d131bf37b726b94aac4884457a5bfa02ef0a52737b3ae09727` |
| `gb10-application-build-gate-v1.json` | `f090e811fb50f00cf425d34290b9b7197328ba5874d8f78c8ac61354c3ef44ec` |
| `gb10-installed54-gate-v1.json` | `fa030718fcff2a1a545ae862e28e5b58a333b1d76a62bd4ca49bf97aac35ec40` |
| `runtime-membership-delta-783-to-779-v1.json` | `fe47f1a09a8d4da99faead39aefb84e829d266e718da381de5a341dd41761e6b` |

These gates establish the stated CPU/source/package contracts, not actual
SM70 kernel execution, SC117 pairing/visual quality, full-context capacity or
matched throughput. The final V100 SDK and physical target qualification remain
separate work; the private native candidate is still unselected.

## Configuration and capacity

For Flash Next GGUF with GPU indexer history, `--parallel auto` computes a
resident count at the selected context. An integer is an administrator ceiling,
including counts above 32; it does not force an unsafe allocation. Every chosen
slot reserves its complete prompt-plus-reply window. The solver uses the
limiting rank and separate loader/serving/host costs, with a useful shared
execution envelope, retained recovery and replacement overlap. The fixed initial
unique-history profile has zero dedicated live-migration overlap; it keeps COW,
restore, recovery and supported replacement liabilities in their own categories. It keeps the selected context rather than shortening it to
fit a requested session count.

`--kv-full-windows auto` selects that full-context policy on the GGUF GPU-history
path. An explicit integer constrains the same capacity solver. Other providers
and the legacy RAM-indexer path retain their own explicit contracts; they do not
inherit the unique-history capacity calculation. The V100 Compose defaults use
automatic session and full-window selection. Existing numeric environment
overrides remain ceilings.

`capacity_plan.session_capacity` reports the computed count, configured ceiling,
limiting resource/rank, and rank costs. `capacity_plan.execution_envelope`
separates verification rows, prompt work, and proposal depth from resident
capacity. Splitting verification into fair complete-request microbatches does
not reduce the number of allocated full-context slots. Planning and materialized
allocation fields must be read with their evidence labels.

Target weights stay TP8. Unique GPU histories apply only to the supported
two-global-KV-head SM70 geometry. Existing Blackwell numerical and storage
paths keep their contracts. A different placement's performance cannot be
inferred from byte balance.

## Numerical formats and speculation

Ordinary INT8 retains the H32 transformation, signed byte payload, and FP16
scales per 32 values. Existing `rotorquant-signed128-4` and
`rotorquant-signed64-norm4` remain distinct formats with their original evidence.

The initial `rotorquant-iso4-sm70-v1` format pins the CUDA ISO4 implementation
at `llama-cpp-turboquant` commit
`08e025c06ab521e4fa9e5c08b80af57614543e53`. A 128-value block has 64 code bytes,
an FP16 norm, and a reserved FP16 field. TensorFold stores code bytes and the
opaque four-byte metadata word separately; wire controls verify their combined
68-byte representation. ISO4 uses the pinned 32 quaternion rotations.
The distinct `rotorquant-planar4-sm70-v1` remains the first later equal-memory
comparison; it is not an alias or a substitute for initial ISO4 qualification.
The pinned reference's CPU and CUDA rotation constants differ, so these formats
explicitly select their CUDA authority. Native long-prompt
packing, attention, quality, and performance require their own V100 runs.

Stock MTP and `--draft-method none` have explicit resource and tail conventions.
Temporarily selecting zero proposals keeps enabled MTP conditioning current;
it does not free resident method resources. Reply/context budgets and stop cuts
bound the prefix committed to recurrent state, convolution, KV, indexer state,
and the drafter. Algorithm-defined random draws are not rewound on rejection.
Startup `--mtp-drafts 0` under `auto` is the legacy structural-disable choice;
explicit stock MTP requires a positive allocated maximum. Use the runtime zero
proposal setting for warm capability.

DSpark and DFlash2 are separately implemented methods. A DFlash checkpoint is
not inferred from a model name or classified as stock MTP. The current
`--draft-method dflash2`/`--drafter` path uses the same Engine and prepared
serving-profile lifecycle, with exact `draft_pairing` content authority before
CUDA allocation. Its [provider/runtime](../src/tensorfold/families/qwen4_exp/cuda/dflash2_runtime.py)
implements completed-HC feature conditioning, target embedding/head aliases,
bounded FP16 draft windows, accepted-prefix recovery and warm maintenance.
The implemented block is four rows, with one to three proposals selected
before the draw; a source block-eight declaration does not enable block-eight
serving. Source/CPU checks do not qualify this pairing's V100 execution,
acceptance, quality, physical envelope or performance.

Reference research identified a concrete candidate,
[`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/tree/a22a1744e85c17364dcc54a6a420af7135d4c7f6),
pinned at `a22a1744e85c17364dcc54a6a420af7135d4c7f6`. The initial research used
public metadata and the tensor header; the later source implementation now
handles its five-layer, 2,048-token sliding draft stack and completed-target-HC
averaging under the manifest-bound contract. The optional exact large DFlash
artifact control was skipped in the 5,214-test checkpoint. The author's
GB300/TP1 evidence does not qualify this fork's V100/TP8/SC117 target; exact
pairing and complete target execution/quality/resource qualification remain
required.

## Runtime control

Adaptive engines expose `GET /v1/runtime` and `POST /v1/runtime`. Both use the
listener's existing API-key policy; opening metrics does not open this route.
The POST body contains one or more integer execution settings:

```json
{"row_budget": 32, "proposal_limit": 0, "prefill_tokens": 512, "session_ceiling": 16}
```

Values must fit that engine's reported `allocated_maximum`. The example values
are illustrative, not a V100 performance recommendation. The response reports
configured, pending, and effective values. The coordinator applies changes
between batches and shares a single ordered execution plan with TP peers.
Lowering the admission ceiling preserves running requests; raising it cannot
exceed the capacity already allocated. Active requests keep their full-context
reservation until completion/cancellation and retirement of their readers.

Opt-in [serving profiles](serving-profiles.md) add the structural transition for
context, codec, model and method through a prepared local manifest. The same
worker and communicator drain model-bound prepared/queued requests, retire old
resources before replacement allocation, and publish after all-rank readiness.
Host fake-loader protocol/lifetime tests cover A-to-B-to-A and unavailable-state
load refusal/retry. Actual model/CUDA-context loading, resident-byte recovery,
quality and integrated V100 performance remain unrun qualification gates.

The runtime receipt is an advisory host snapshot, including active requests,
allocated slots, bounded derived prefix-cache slots/entries, the current
execution epoch, useful committed tokens,
and the capacity receipt. `/stats` includes this section for adaptive engines.
Reading it does not synchronize GPUs or turn startup planning into a current
physical-free-memory measurement.
Step elapsed time is host wall time, including waits actually observed by that
thread. Device-completion time is unmeasured and reported as null; asynchronous
maintenance must be included by a complete workload measurement before using a
timing sample as performance evidence.

The execution receipt includes `work_observations`, published by the serialized
worker at operation boundaries. It separates fresh logical prompt admissions,
request/replay prefix hits, submitted target prefill/decode and replay rows, and
MTP prefill/target-conditioned refresh/provisional proposal rows, including the
existing cached-prefix boundary refresh at its saved position. Warm runtime
zero-proposal refresh has its own count. Synthetic startup warmup is cleared
before serving. None of these row counts is generated output or device time.

Indexer counts are derived from actual submitted query ranges: visible complete
pooled groups, visible sparse group/query-head score pairs, and selected key rows
including incomplete tails. They count the canonical logical work once across
TP ranks, with no factor for replicated scratch or masked physical launches.
They are not measured memory traffic or a claim that every implementation
executes only visible arithmetic. Partial failed forward submissions are marked
unknown; complete totals are not fabricated. The prospective profile-table
`indexer_work` axis remains unknown until its separate covering contract is
established. `expert_group_max` remains null; batch size is not a substitute for
actual routed expert occupancy.

Acceptance histograms are fixed to the allocated proposal maximum, including
maxima above32. They count the existing verifier's accepted prefix after its
reply/EOS rules and before the text-stop prefix callback. Offered widths and
accepted totals by offered width are separate. The existing useful-output count
continues to exclude owed replay and failed grammar publication. No per-stage
durations are guessed. A failed admission observation retires or quarantines its borrowed slot and rolls
back the attempted counters. A separate failure latch invalidates completeness
even when publishing another snapshot cannot allocate. Counter exhaustion is
visible and invalidates completeness instead of silently wrapping; these host snapshots are diagnostic readings, not
new capacity or performance proofs.

The historical optional migration prototype accepted standalone
`{"attention_degree": 2}` updates, copied packed bytes through balanced bundles
and published committed owner-map generations. Its blocking, partial-map and
containment oracles remain research/reference evidence. Its 64 MiB/rank candidate
allowance is not charged or enabled by initial serving; it was never a measured
physical migration bound. Initial fixed-map serving does not expose these live
updates. Any future migration adoption must restore its complete intermediate
resource and lifecycle obligations and pass the governing qualification gates.
Managed TP launches signal their supervisor on a fatal cohort operation so a
peer blocked in a collective cannot strand process retirement. Unmanaged
launches retain their external containment responsibility.

Without an explicit computed-retention policy, CUDA generation remains
request-scoped: completion or cancellation retires the claim after its readers,
transfers and callbacks finish. Reusable full-window allocations and bounded
derived prefixes do not create an intentional retained-session promise.
Stored Responses/Conversations retain logical records independently of GPU
ownership.

An explicit typed GPU/RAM retention policy uses the implemented common catalog
for owner/generation claims, protected idle checkpoints, prefix leases,
suspension/restoration, close/expiry and incompatible-profile drains. FIFO
waiting, input credits and pending/in-flight output credits retain their logical
owner through blocked-request requeue. New-turn generation bookkeeping remains
separate from same-request continuation. These source contracts and CPU checks
are subject to the pending native resource/backend and complete target gates
above. New admissions increment the destination slot's generation before
copy/restore; rollback within the same request preserves that generation.

A zero proposal setting keeps enabled method conditioning current during active
execution. A per-request `draft: false` switch disables speculative work for
that call. Without computed retention, request completion does not maintain an
idle endpoint in anticipation of an unspecified future resumption.

Execution profile registration uses either startup-bound proof references or a
standalone `{"profile_manifest": "new/manifest.json"}` request. Verification is
queued to the TP worker; paths cannot escape the configured evidence directory.
See [the exact manifest contract](execution-profile-manifest.md). Qualified
covering regions are explicit and nonoverlapping; token-by-token changes inside
one proven region can retain its selected execution plan. No measurements are
fabricated when the table has no eligible region.

## Reproducible hardware and workload evidence

After building the V100 verification image as described in
[the deployment guide](../deploy/v100/README.md), record the actual server:

```bash
python tools/v100_adaptive_qualification.py inventory \
  --rank-devices 0,1,2,3,4,5,6,7 --out platform.json
python tools/v100_adaptive_qualification.py transfers --out transfers.json
```

Inventory is read-only and bounds subprocess output and time. Missing tools,
failed commands, and absent measurements are retained explicitly. Run isolated
transfer probes outside serving performance measurements. GPU indexes in the
example are explicit logical-rank order, not a claim about the best NVLink
mapping. Record CPU/RAM/NUMA placement, UUIDs, ECC, links, clocks, power,
temperature, errors, and exact software versions.

Save the running `/stats` response, then derive the qualification matrix:

```bash
python tools/v100_adaptive_qualification.py matrix \
  --stats stats.json --out matrix.json
python tools/v100_adaptive_qualification.py benchmark \
  --url http://127.0.0.1:8000/v1 --out benchmark.json
python tools/v100_adaptive_exercise.py --url http://127.0.0.1:8000 \
  --out exercise --request short=/path/to/short.json \
  --request intermediate=/path/to/intermediate.json --request long=/path/to/long.json
```

The matrix includes each integer concurrency through the computed endpoint,
including capacities above 32. The benchmark uses the existing HTTP SSE
workload implementation and preserves raw responses/failures. Its short-prompt
measurements do not automatically qualify every matrix cell. Near-full
retention, mixed load, migration, output/quality, long positions, cancellation,
slot reuse, and sustained fill/drain remain distinct gates. Tool resource caps
are explicit workload-generator limits and must not become serving ceilings.

Compare identical target/draft revisions, codec, context, retention, output
quality, arrivals, and runtime state. Count useful prompt work and committed
output separately from tentative proposals. Include queueing, TTFT, inter-token
latency, fairness, recovery, communication, and migration. Select performance
profiles only from applicable completed measurements, never from a generated
matrix or startup allocation alone.

The exercise runner uses supplied JSON requests and actual token counts. Add
distinct long requests to establish multiple retained contexts, plus named
`stop`, `eos`, `exact` and `overrun` requests for their corresponding gates.
Missing input roles, early EOS in a context-exhaustion case, unobserved slot
reuse/retention and incomplete migrations remain explicitly unsupported or
pending. It drains response owners before restoring changed controls. Its
cooperative body deadline has the documented one-socket-read bound; it does not
claim instantaneous DNS/header cancellation. HTTP observations alone do not
authorize a qualified execution profile.
