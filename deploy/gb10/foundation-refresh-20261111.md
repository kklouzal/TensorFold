# GB10 foundation refresh — 2026-11-11

Status: discovery and implementation in progress; no new image, native result, performance result or promotion is qualified by this ledger.

## Authority, scope and ownership

The user authorized a thorough GB10 foundation/dependency refresh using an official NGC **CUDA**, not PyTorch, development base, with the newest mutually compatible official PyTorch nightly cu134 and other compatible dependencies. Preserve base-owned SDK packages and genuine upstream wheel metadata. The user-supplied Lite policy and parent/repository AGENTS.md apply; README, CONTRIBUTING.md, GB10 foundation/build procedures and governing ProviderOne preservation contracts govern affected work. Source existence, CPU checks, image integrity, native execution, model quality and performance are separate evidence states.

Canonical development source is /home/kklouzal/TensorFold on 192.168.1.140, branch python-0.6. Initial clean HEAD was b64eb2330c43154c0934504b355f7e9365baec5d (verified over strict noninteractive SSH). The local TensorFold-python-integration checkout is a read-only source mirror and authorized push relay, not an implementation copy. No force push or credential copying is authorized.

Task ownership: root owns final integration acceptance and production promotion; coordinator owns Git, this single durable ledger and affected documentation; source author owns implementation/generators/pins; builder owns acquisition, immutable source capture, container builds and installed/native evidence; discovery owns official version/candidate assessment; the primary assigns one distinct task-wide D13 reviewer. Coordinate canonical mutation and benchmark/device leases. Reviewer scope follows affected foundation interfaces, generators, deployment selection and actual consumers, up to three relationship hops; expand only for concrete task risk.

## Preserved baseline and invariants

The supplied current-production baseline is image sha256:d992ac681ac33afb9f29c8712d8a373f11cdb4d10a61a1163960d089a4caf402, source UID ef30337dbbf5fb5734b4cafe9230346fa25921b4c48ee60e247a6a994ce92642, container gemma4-26b-a4b-tf / 7ea6807342843b6dac2545d1c94dd63b3f62673ffe56badb63e6607d47090854. Root verifies live identity before any cutover. The production Compose SHA256 is f9ee65ee57b9df46060d090f086f37e751917c0a742f509619afed1c0b4dce76; its retained path is /home/kklouzal/services/gemma4-26b-a4b/deploy-w4-clipping-20261105-v1/compose-production.json, project tensorfold-shared-prefix-production-20261111-v3.

Preserve AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4 revision 90c8601b1a439497bc15d8fba9e52990b45ea0cb; z-lab DFlash revision 77d4202772dfe50b2396ec7bac9cfffc7b9e7057; Google template revision 4d7ae498 / SHA256 ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4. Preserve context262144, BF16 KV, vision, thinking-budget0, draft-bits8, rows5, Delta4 head, global prompt preparation, DFlash-head W4A16 and sampled clipping. Preserve the 40GiB CUDA allocator cap, 16GiB host reserve and 12,667,846,656-byte full cache without an entry ceiling. Prior user acceptance of approximately 0.3–0.5s cold cache cost is specific to that feature, not a foundation regression waiver.

Initial selected foundation: NGC CUDA13.4.3 cuDNN development Ubuntu24.04, index sha256:f2d23b75391766a02175908d3fa1c2afee9849aa13d09326d71649de2b33cfe9 and ARM64 child sha256:e57a21d3cf1398462f74efe1b57603cb968ee750088c6160ea89458b4f64dcfc; ARM64 Torch2.16.0.dev20261014+cu134, TorchVision0.30.0.dev20261006+cu134, Triton3.9.0+gitfc8d31a5. Wheel cu134 declares a CUDA minor line, not independently a toolkit patch. Confirm actual nvcc and loaded native providers separately.

SDK-native holds and full file/symlink audits remain mandatory. Protected Python roots (including existing NumPy2.1 closure) may change only through fresh foundation resolution and qualification, never by application pip overwrite or guard waiver. Keep V100 CUDA12.9 inputs untouched. Keep AMD64 maintained selectors/metadata coherent while distinguishing unrun native coverage. Preserve all prior rollback images/containers/cache directories, including the original a13 rollback and failed-created 4a0 artifacts.

## Regions, objective and candidates

Required region: native ARM64 GB10 SM121, actual production model/profile above, fresh foundation-specific caches, cold startup/first request, warm serial and concurrent serving, model/media/tools/context/cache/cancellation behavior. Objective order is preservation of P0–P2 and existing memory/context/quality commitments, then complete-request latency and throughput (including cold and warm costs), then additional authorized quality, then maintainability. No arbitrary numerical/performance tolerance or new support target is introduced. Discovery and qualification must retain uncertainty and regional losses rather than aggregate them away.

Candidate disposition (pending evidence):
- Latest compatible immutable NGC CUDA13.4.3 base and newest coherent official Torch/TorchVision/Triton cu134 set: discovery pending.
- Latest compatible application/verification package closure: discovery and resolver evidence pending; explicitly record constraints that exclude newer releases.
- Existing protected SDK/foundation isolation and generated locks: retained required contract.
- V100 refresh, model/kernel optimization and deferred low-20MiB-cache issue: outside this task.

## Evidence plan and stop conditions

Use maintained deploy/gb10/lock_dependencies.py on actual matching-platform or explicitly validated cross-report metadata. Generated lock/hash/provenance outputs must be regenerated, never hand edited. Record whole-wheel identities, official HTTPS origins, complete resolver environment and SDK add-on identities. Freeze an explicit clean canonical commit with tools/gb10_capture.py and named contexts; retain capture/build completion and image identity receipts.

A changed foundation invalidates historical ABI-cache, native, model and performance admission. Required current-image gates include SDK file/symlink/provider/tool audits, pip metadata consistency, affected installed CPU/deployment checks, actual ARM64 native/math/model/media/tools/cache/cancellation/context/quality checks and matched end-to-end cold/warm performance observations. Reuse unchanged valid evidence and run bounded checks proportional to affected risks; do not repeat broad suites per edit. Root controls production/device test leases and final promotion. Failed checks and unrun coverage remain explicit. Canonical final-source checks, D13 findings/dispositions, coherent commits and non-force push/synchronization proof remain required.

## Checkpoints and blockers

2026-11-11: coordinator verified clean canonical python-0.6 at b64eb233. Read repository AGENTS.md, CONTRIBUTING.md and managed-foundation procedures. Remote rg is absent; git ls-files, grep and bounded reads provide navigation. An overly broad parent find encountered unrelated permission-denied configuration; no task dependency is blocked and no unrelated files were changed.

Open: exact newest compatible candidate set, generated input identities, source/application composition preserving qualified production runtime bytes, acquisition/build logs, current-image gates, D13 review, final documentation, production decision and push verification. Canonical remote has no write credential; authorized relay must fetch canonical SSH, fast-forward, push origin, then canonical fetch and verify clean status plus zero divergence. Do not treat commit/push as native qualification.
