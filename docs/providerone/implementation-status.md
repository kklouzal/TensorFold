# ProviderOne implementation status and authority handoff

This is a source-to-contract map, not another design or a release certificate.
The [canonical governing design](../../ProviderOne%20Implementation%20Docs/ProviderOne_V100_Adaptive_Inference_Design.md),
particularly Sections1,2,21,22 and24, selects the required behavior and exit
gates. The four companions in that directory provide research/review and a
conditional experimental backlog; they do not independently select a serving
policy. All five authoritative input files are adopted in place without byte
changes. The [legacy design path](../v100-adaptive-inference-design.md) points
there. Untracked `!` drafts remain user material, not maintained authority.

## Baseline, dirty state and evidence identities

The starting implementation is `python-0.6` commit
`8d69e49c0adc0192a3849169cccff9303cee51d0` in the owned
`https://github.com/kklouzal/TensorFold.git` lineage. The governing document's
c8318787 audit is a historical source snapshot; it is not an audit/pass claim
for this later baseline. Current package/import/native/CLI identities still
use TensorFold. Product rebrand, packaging/updater/remote changes and an
installed ProviderOne release remain requested work, not completed by this
handoff. Historical source, image, namespace and benchmark names are not renamed.

At the starting checkpoint the index was empty and local/remote commits matched.
The old tracked root `TensorFold_V100_Adaptive_Inference_Design.md` was already
deleted by the user; this checkpoint adopts that deletion without recreating
or editing it. Otherwise a fresh checkout would retain a stale full contract.
Its old HEAD blob `a289270fed60f61d5d471c9b39cd8c1c2ac57c5d` is preserved outside
the checkout. The five canonical files were untracked author inputs and are
now the explicitly authorized additions. Four untracked `!` drafts and six
unrelated B16 source/test files remain untouched and unstaged. This checkpoint
changes no runtime code, model, dependency, installed service or hardware setting.

Executor baseline receipt (private audit root
`/home/kklouzal/audit-artifacts/gemma-aeon26b-optimization-20261102`):
`query-tile8-cleanup-v1/git-closeout-v1.json`, SHA256
`e4feb74b676bfb9b155644328d1d2cf01b70c2e97442b7812b67a408b918e84d`.
The preserved old-root reference is `providerone-authority-old-root-8d69-reference.json`,
SHA256 `2af5e7fc2377ef5767d07af48af1b1a127d2f08e38373a9e5422c960dd12bd94`;
its bytes hash is `7c9f724346866cfb89382ee2e30d60d9d827222c347610783b37683ba04e5a8c`.
Private receipts are provenance, not dependencies for fresh-checkout tests.

## Requested, implemented, qualified and effective

“Implemented” below identifies existing starting source and bounded software
controls, not completion of the full governing subsystem. Qualification and
effective deployment require their own exact artifacts, target and receipts.

A5 restores the 37 inherited fixture cases in the scoped regression set. Source
and installed batches now each pass 1,498 tests with one optional companion
skip; all 115 original assertion nodes remain and three new non-GGUF provider
negatives pass. Runtime source and the installed A4 images are unchanged. Earlier
failure counts remain historical evidence; no actual hardware, model, capacity
or performance gate is inferred from these repaired CPU controls. See
[the fixture and artifact scope](../v100-adaptive-validation.md#restored-fixture-contracts-a5).

| Contract surface | Requested | Existing implementation starting point | Qualified evidence | Effective status |
| --- | --- | --- | --- | --- |
| Shared API/admission/lifecycle | One shared logical/API owner, bounded FIFO/credits, cancellation/publication and complete computed-state preservation | [admission](../../src/tensorfold/server/admission.py), [Responses](../../src/tensorfold/server/responses.py), [Conversations](../../src/tensorfold/server/conversations.py), [logical records](../../src/tensorfold/server/api_records.py); Qwen prefix/retained catalog and RAM restore | CPU software milestones exist. Logical records alone do not qualify GPU continuation. The concrete ProviderOne integration and actual SC117 workload/failure gates remain open. | Existing named API profiles remain unchanged; no SC117 ProviderOne release is asserted. |
| Artifact/container boundary | Bounded neutral container decoding before family selection, exact content/model/recipe identity | [neutral GGUF inventory](../../src/tensorfold/gguf_file.py), [neutral file checks](../../src/tensorfold/tensor_file.py), [checkpoint metadata](../../src/tensorfold/checkpoint_metadata.py), [Flash Next semantics](../../src/tensorfold/families/qwen4_exp/gguf.py) | A1 source separates the existing container/file algorithms from family admission and passes one validated inventory into family configuration. Exact file-helper bytes and token/decode definitions are retained. Focused boundary and corrected resolver cases passed; the current CUDA-hidden source batch recorded 1228 passed/48 baseline failures/4 skipped, with clean static checks on 115 Python files. The fresh installed image recorded 1264 passed/12 pre-existing unqualified failures/4 skipped, with 778 installed runtime files unchanged before/after, active purelib origins and no source overlay. Fresh native construction resolved 36 stale-FD failures; no test or guard was relaxed. [Exact identities and limits](../v100-adaptive-validation.md#neutral-artifact-io-boundary-a1). | Affected installed I/O boundary qualified; no new model/format/device support or production deployment is selected by the refactor. |
| SC117 target/SM70 stack | Non-Swift SC117 IQ3_S, exact source-built SM70 stack, native/extended context and complete state/quality/resource gates | [V100 build procedures](../../deploy/v100/README.md), [source foundation](../../deploy/v100/source-foundation.md), Qwen GGUF import and CUDA engine | Exact source/header/import and historical CPU/build milestones exist; selected payload/mmproj verification is recorded. Current physical V100 allocation, model/state/quality, useful capacity and throughput are not established by this checkpoint. | No selected SC117/V100 release established by this map. |
| Full-window capacity and unique state | Imported-allocation worst-rank capacity, fixed unique KV/indexer placement, typed recovery; no silent context reduction | [history owner](../../src/tensorfold/families/qwen4_exp/cuda/history_owner.py), [placement](../../src/tensorfold/families/qwen4_exp/cuda/history_placement.py), [indexer history](../../src/tensorfold/families/qwen4_exp/cuda/indexer_history.py) | A2 source selects explicit fixed degree4 at initial TP8 unique-history preparation and publishes that map to State. The ledger retains a zero dedicated migration-overlap term; ordinary serving does not activate or accept live-degree updates. Prefix/COW/restore/recovery/replacement reserves and optional research oracles are preserved. A2 affected source and installed CPU controls passed: each broader batch recorded 367 passed/4 existing dedicated ABI/version failures/12 frontend-version skips; required deployment/public-permission 194 and static 7 passed. Fresh installed 778-file inventories and changed-module purelib origins matched with no source overlay or CUDA initialization. The four target gaps remain unqualified. Physical fixed-map numerical/resource/long-history qualification and the maximum admitted integer count remain pending. [Exact A2 identities and unit-test scope](../v100-adaptive-validation.md#fixed-initial-unique-history-a2). | No SC117 integer capacity or new live-redistribution policy is effective here. |
| Required QSA MTP | Default shared-Q8 MTP, exact aliases/norms, selected Q8/iso4 KV and full-prefix text/visual conditioning | [GGUF MTP import](../../src/tensorfold/families/qwen4_exp/cuda/gguf_mtp_weights.py), [SM70 MTP weights](../../src/tensorfold/families/qwen4_exp/cuda/sm70_mtp_weights.py), [media conditioning](../../src/tensorfold/families/qwen4_exp/cuda/media_mtp.py) | Import/alias/recovery software controls exist. Required SC117/V100 MTP execution, full-prefix quality, acceptance, memory and performance remain unqualified. DFlash availability or explicit no-speculation cannot substitute for MTP delivery. | No newly qualified SC117/V100 MTP profile is effective here. |
| Separately selected Inco DFlash2 | Exact trained pairing, TP8 HC conditioning, finite-range precision, rollback/sampling and text/visual gates | [manifest](../../src/tensorfold/families/qwen4_exp/cuda/dflash2_manifest.py), [runtime](../../src/tensorfold/families/qwen4_exp/cuda/dflash2_runtime.py), [verification](../../src/tensorfold/families/qwen4_exp/cuda/dflash2_verify.py) | A bounded four-row source runtime and controls exist; block-eight serving is not implied. Actual SC117/V100/TP8 quality, finite ranges, capacity and useful performance remain required under Section14. | No newly qualified SC117/V100 DFlash2 profile is effective here. |
| Image/video → text | Pinned processor/vision/positions/PLE and bounded mixed-work staging with separate method gates | [media runtime](../../src/tensorfold/families/qwen4_exp/cuda/media_runtime.py), [media MTP](../../src/tensorfold/families/qwen4_exp/cuda/media_mtp.py), V100 media foundation | Source/installed CPU/media authority milestones do not qualify actual SC117/mmproj/V100 fidelity, maximum work or sustained mixed-load service. Audio/media generation is outside this contract. | No new SC117/V100 media envelope is effective here. |
| Prepared plans and no-restart control | Fixed maximum envelope, ready-work/prepared-plan selection, FIFO/prefix/tiered state and supported in-process administration | [adaptive controller](../../src/tensorfold/families/qwen4_exp/cuda/adaptive_controller.py), [serving profiles](../../src/tensorfold/families/qwen4_exp/cuda/serving_profiles.py), [profile runtime](../../src/tensorfold/families/qwen4_exp/cuda/serving_runtime.py), retained catalog/host storage | Software tests and bounded source mechanisms exist. Actual burst/load/fairness/backpressure/recovery/resource and no-restart target gates remain required. Online fits, graph replacement during ordinary adaptation and live migration are not assumed requirements. | Only previously admitted profiles retain their effective controls; no new target transition is certified. |
| Follow-on models/formats/devices | Complete concrete family/recipe/backend/API/state qualification after the SC117 priority, while preserving selected profiles | Existing family implementations and [recipes](../recipes/README.md) are starting points | User-requested labels are not verified artifact identities. No universal model/format/quantization support, inherited SC117 geometry or new-device qualification is claimed. | No new model/format/device profile is activated by this handoff. |

## Required SC117 parallelism matrix

A4 adds mandatory basic physical-stage admission for canonical SM70 TP8
GPU-history startup: one kernel host, ordered complete rank coverage, distinct
canonical GPU UUIDs and the specified V100 SXM2 32GB/SM70 observations before
NCCL and model admission. All SM70 TP8 ranks use the common CPU Store envelope;
uniform legacy participation skips eligibility and gets no physical-stage
receipt. Source and fresh installed CPU batches each recorded 1,458 passes,
37 inherited fixture failures and one optional-fixture skip. Strict per-case
comparisons retain every failure as a failure. This does not establish the actual
mesh, full model, resident capacity or performance gates below. See the
[A4 artifact and proof scope](../v100-adaptive-validation.md#pp1-physical-stage-admission-a4).

These are six required cases, not six qualified implementations. TP flags,
smaller-world tests, KV placement degrees or manifest metadata do not establish
expert ETP/pipeline execution. Each needs explicit target/MTP partition,
rank/layer/expert coverage, fixed-map state/graph parity, complete worst-rank
ledger and the governing benchmark matrix.

| Profile | Requested execution | Implemented starting point | Qualified | Effective |
| --- | --- | --- | --- | --- |
| TP8 × PP1 × ETP8 | One eight-GPU server | Existing all-expert/width80 arithmetic retained; prepared selection and all-rank agreement have A3 source/installed CPU controls. | Actual complete SC117/V100 profile unqualified here. | No new released profile established. |
| TP8 × PP1 × ETP4 | One eight-GPU server | A3 source/installed CPU controls cover EP2 banks of256 experts, width160, target/shared-GGUF MTP and owned-route reference execution. | Actual complete SC117/V100 profile unqualified here. | Pending this case's gates. |
| TP8 × PP1 × ETP2 | One eight-GPU server | A3 source/installed CPU controls cover EP4 banks of128 experts, width320, target/shared-GGUF MTP and owned-route reference execution. | Actual complete SC117/V100 profile unqualified here. | Pending this case's gates. |
| TP8 × PP2 × ETP8 | Two eight-GPU servers, one TP8 stage/server | A6 private 24/24 import and local execution components exist; transport, endpoint/method services, transactions and full admission remain missing. | Actual two-server profile unqualified here. | No PP2 serving path is activated. |
| TP8 × PP2 × ETP4 | Two eight-GPU servers | Stage-local EP2 import is exercised by A6 components; the complete pipeline and its resource proof remain required. | Actual two-server profile unqualified here. | No PP2 serving path is activated. |
| TP8 × PP2 × ETP2 | Two eight-GPU servers | Stage-local EP4 import is exercised by A6 components; the complete pipeline and its resource proof remain required. | Actual two-server profile unqualified here. | No PP2 serving path is activated. |

A3's source and fresh installed batches each recorded 1,217 passed, 29 inherited
fixture failures and one optional-fixture skip. All 29 failed cases have matching
nonempty diagnostics on the prior source reference; they remain failures, not
equivalence or target qualification. Physical stage binding, full-model numerical
quality, maximum capacity and performance gates remain open. See the
[A3 artifact identities and limits](../v100-adaptive-validation.md#pp1-expert-partitions-a3).

The private [A6 component boundary](../v100-adaptive-validation.md#private-pp2-stage-components-a6)
does not implement PP2 serving. It preserves full HC instead of a stream mean,
global block/PLE identity, local cache ordinals and actual stage coefficient
ownership. Required MTP/DFlash endpoint/features, one distributed round commit,
bounded progressing services and complete two-server admission are still open.

One-server release may proceed only after all three PP1 profiles and required
SC117 feature gates pass; PP2 remains required work pending its own hardware
qualification. ETP1 is excluded; TP4×PP2 is optional future study, not a current
release gate. Native262144 and extended-position commitments, every admissible
activity count and actual maximum, C1/2/4/8/16/24 milestones, >250 single-request
and >600 aggregate-at24 Q8 decode plus >3500 prefill objectives remain requested
measurement gates, not achieved figures or admission caps.

## Selected existing profiles and historical evidence

The selected GB10 Gemma profile remains source
`ae8d40bef288d9528aed2b45b06cffecb955e392788c240e89e8ebe45d148e6a`, image
`b579387bc49a092cd2a0b6f48a4c8f17a23995ce7191bd4d359e42c811c12684`,
N64/W4A16-head DFlash8/depth4, Shared512, Delta4/R16 and clipping ON, context131072,
BF16 KV, original model/drafter/template, vision, aliases and port. A8/proposal
graphs/trig/persistent cache remain off. It is not a SC117/V100 numerical profile.
The existing disclosed owner-accepted GB10 trade does not become a general
regression allowance or waive SC117 decode/capacity gates.

Private evidence anchors: `clip-sampled-cli-v1/production-closeout-v1.json`
SHA `050a0698a5d2b62853c22c273354612712fd3dbeb9db53cd25d7fb9460756301`;
installed650/exact777 receipt `installed-gate-v2.json`
SHA `d0d3c276613309d4146dc0bf7b9dc0044b0dea5cd3d9dab51c873dcc28db56cf`.
Latest recorded restoration is `query-tile8-operator-v1/production-restore-collect-v1.json`
SHA `7d735f9894eccf06abb23ef94746a4f626e8634213c7384f1566ed6db7539c76`,
Started2026-11-06T05:35:53.382963622Z, with API/room-one/math/aliases/pointer/speech
checks. This map reads those receipts; it does not claim a new live probe.
The full API suite's16 functional passes/failed17th cancellation-counter and
aggregate false/exit1 remain distinct from the separate observed-active
cancellation witness. See [GB10 qualification](../gemma4-cuda-validation.md).

QT8 is deferred/unqualified after the goal change: source/installed435 and
metadata/cache-admission controls are not GPU-word/lowering/timing evidence.
Its prototype/images/failures remain archived; default16 attention and original
tests were restored at the baseline. N128, fused-tail and other investigations
retain their recorded scoped dispositions, gains and costs. No benchmark is
renamed ProviderOne evidence merely because the project direction changed.
Other existing model/backend checkpoints in [README](../../README.md) and
[optimization validation](../optimization-validation.md) retain their own scope;
this initial map is not an exhaustive new qualification of every existing profile.

The historical5582-check/144-subtest milestones belong to ff188313/21a12df3 and
707 installed files, not to8d69 or a new ProviderOne build. Keep the
[V100 qualification record](../v100-adaptive-validation.md), its historical
[CPU receipt](../verification-v100-adaptive-cpu-20261026.json) and
[corrected-stack receipt](../verification-v100-sdk-rebind-cpu-20261026.json)
at their actual identities. Source existence, compilation, installed CPU proof,
physical allocation, readiness, exact model/state output, quality and full-load
performance remain separate statuses.

## Verification and next ownership boundary

Existing affected controls include `tests/test_v100_source_inventory.py`,
`test_v100_source_runtime.py`, `test_v100_source_install.py`, `test_v100_compose.py`,
`test_flashnext_gguf_mtp.py`, `test_flashnext_media_mtp.py`,
`test_flashnext_dflash2_manifest.py`, `test_flashnext_dflash2_verify.py`,
`test_flashnext_history_placement.py`, `test_flashnext_adaptive_controller.py`,
`test_serving_profile_manifest.py`, `test_retained_catalog.py`,
`test_responses_api.py` and `test_conversations_api.py`. They are procedure
pointers, not new pass claims. Follow [CONTRIBUTING](../../CONTRIBUTING.md) for
required deployment/source/static checks and the actual-target gates for a
changed runtime. No duplicate prose-hash test framework is introduced here;
authority uniqueness, unchanged input bytes, relative links and the exact staged
allowlist are checked directly at this checkpoint.

A1 implements the neutral artifact boundary under a bounded canonical source
lease following authority checkpoint `ff47150525dc55d8c791965beb8a894cc611541a`;
its verification remains separate from the earlier documentation checkpoint.
Coherent package/native/CLI/cache/deployment/updater rebrand remains a required
release boundary. No new framework, backend emulation or parallel legacy serving
stack is authorized by the handoff alone.
