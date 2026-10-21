# Performance development paused 2026-10-21

Optimization development is paused at the user's request. No new candidate from
this campaign completed all model/performance gates; no new performance gain is
selected. The accepted source reference is `3b8cee82`. Its runtime sources are
identical to the deployed `259b4238` image, and the Dockerfile, foundation, launcher
and Compose launch configuration are unchanged.

The GB10 container was verified healthy before and after a live HTTP generation
that returned `OK` with typed token accounting. Its exact image/CID/configuration
and both voice peers match the accepted deployment. It already runs the latest
validated runtime, so no container recreation or experimental image rollout was
needed. See [the current verification receipt](../deploy/gb10/deployment-performance-pause-20261021.json)
and [the full deployment qualification](../deploy/gb10/deployment-gemma4-throughput-20261021.json).

The direct FP8 packing candidate remains rejected: its loading improvement came
with an established required request-region regression. The allocator extension,
Qwen bulk loader, native AOT preparation, GDN scheduling and other candidates
remain experiments. CPU/native component checks do not establish a model-level
performance gain. No claim of exhausted optimization possibilities is made.

All experimental source is retained in local Git checkpoints, separate from this
validated release. Raw measurements, failed attempts, immutable captures, review
records and next steps remain under
`task-artifacts/performance-resume-20261021/`; `ledger.md` is the main index and
`pause-closeout/PAUSED.md` is the resume entry point. Those local artifacts are
intentionally not part of the shipping package. Resume from these records rather
than treating old branch divergence as a qualified backlog.
