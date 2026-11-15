# SC117 pending optimization closeout — 2026-11-15

The previously uncommitted quantized-dense, fused hyper-connection, and device draft-choice experiments were evaluated and retired from the runtime. The selected implementation remains the committed reference at `be7c355bd55b39d307452b45ba350d664e5b46fd`. This closeout makes no new default-performance claim and does not declare further optimization exhausted.

## Decisions

| Experiment | Evidence | Disposition |
| --- | --- | --- |
| Quantized dense projections | After repairing a C++ API error and admission arithmetic in the experimental copy, sampled real-weight decode and matmul results matched the FP16 reference. Every selected component sample was slower: five-row output projection 25.5 versus 9.7 microseconds; head 1047.4 versus 206.1 microseconds. | Not adopted. There was no serving integration or full-model gain; retain the verified FP16 implementation. These component results do not establish full-model performance or complete format qualification. |
| Fused hyper-connection collective | No serving caller or dedicated native qualification existed. Review found geometry/admission gaps, a V100 residency concern, missing downstream group sums, and unresolved synchronization/numerical obligations. Earlier HC-sharding timings concern a different implementation. | Not deployable as written. Retain the established collective/read-out path; preserve the prototype as evidence rather than enable it. |
| Device draft chooser | 129 native differential cases and 19 CPU dispatch/validation cases passed. Full-model quality and capacity checks passed within the scope below. End-to-end performance was mixed across workloads. | Unproven as a global improvement; not adopted under the no-regional-regression selection rule. No verified region selector resolves the observed tradeoff. |

The implementations, corrective experiments, raw logs and source manifests are preserved in the local `task-artifacts/optimization-closeout-20261115` packet, including `unselected-prototypes.tar.gz` and `draft-pick-unselected.tar.gz`. They are historical evidence, not alternate production implementations.

## Draft-choice comparison

Four fresh processes ran in reference/candidate/candidate/reference order. The source manifests differ only in `MultiDecoder`'s draft-choice integration. Both arms used the same requests, warmup and cache lifecycle; no profiling injection was active. The candidate added no target-model approximation.

- Eight Tesla V100-SXM2-32GB GPUs, driver 580.178.04, 150 W limits.
- Image `sha256:cf8c0ff6baf353ce6a2a548330855631dee1c09906885968b77154b50296e11b`.
- SC117 IQ3_S target snapshot `90a3350cd800ec1fe0d1660a38414ad9e985782f`, shared Q8_0 MTP snapshot `766911a6b7369840a91dbcd95f9f997acaab6cd6`.
- TP8/ETP4, INT8 KV, 24 full 262144-token windows, execution rows 120, prefill chunks 1024, MTP4 at confidence 0.7, existing `m-profile7h` policy.
- The queued experiment's `TF_SMALL_GATHER_BYTES=131072` was retained. This is **not** the separately recommended 262144-byte configuration; these results do not qualify that configuration or justify extrapolation to it.
- C1: four 400-token requests; C4/C8: two batches of 256-token requests each; C24: one 128-token batch; long-prompt cases: one 128-token request after approximately 4K/16K words. Actual token counts and request hashes are retained. Prefixes were fresh within each process; no filesystem-cold claim.

The primary metric is completed tokens divided by full client makespan, including time to first content. Values below are candidate/reference changes in each matched pair:

| Workload | Reference then candidate | Candidate then reference |
| --- | ---: | ---: |
| C1 | +1.580% | +6.186% |
| C4 | +1.530% | +0.495% |
| C8 | +3.651% | +1.507% |
| C24 | +4.285% | -2.831% |
| prefill4096 | -0.114% | -0.182% |
| prefill16384 | -1.670% | +0.963% |

C1/C4/C8 gains are recorded, not rounded away. They do not offset mixed C24/long-prompt results or establish equivalence for uncertain cases. Two processes per arm do not support a precise regional equivalence claim. A context cutoff chosen after seeing these results would require a new cost rationale and qualification.

The historical concurrent “decode aggregate” calculation counts all tokens over an interval beginning at the latest first-content event; early requests can produce tokens before that interval. It is a common-frontier proxy, not a conservation-correct aggregate decode rate. Selection here uses full makespan; individual-request timings are also retained.

## Quality, capacity and lifecycle

The maintained quality harness scored the frozen SC117 corpus: 16 documents, 4696 prefill and 2626 decode positions, two repeats per arm. Tokenizer/config hashes match the selected model metadata. Both repeats were identical. The standard comparison used margin 0.005 nats/token, 4000 document-block bootstrap samples and seed 0; no real loss was accepted merely because it fell within the margin.

| Path | Reference and candidate NLL, nats/token | Reference and candidate perplexity | Comparison |
| --- | ---: | ---: | --- |
| Prefill | 0.96137 | 2.6153 | Identical; delta and CI endpoints exactly zero |
| Decode | 0.74245 | 2.1011 | Identical; delta and CI endpoints exactly zero |

Top-1 agreement was 1.0 and measured coarse KL was zero. All six short golden replies matched. C1/C4/long-prompt benchmark replies matched; concurrent C8/C24 text variation was retained, including C24 differences between the reference's own repeats. Quality findings apply to the scored corpus and do not assert universal output identity.

Candidate startup records show all eight ranks materializing 24 windows of 262149 cache slots for a logical context of 262144. The 2 GiB serving workspace and 1,522,253,824 bytes of retained-snapshot allowance remained protected, with zero future cache growth. All 79 requested graph keys and their round variants were admitted. The worst-rank pre-capture graph grant was 5,784,748,032 bytes. Post-capture numerical allocator details were coordinator-only; no fabricated worst-rank post-capture margin is claimed. This is reservation/admission evidence, not a full-context traffic benchmark.

Sampled-policy bypass and a successful request after cancelling a stream were checked in each process. All four servers shut down with exit status zero. Native and CPU checks preserve error precedence and ambiguous-tie fallback; no assertions or quality gates were weakened.

## Scope and reproducibility

The local packet contains commands, hardware identity, source hashes, benchmark requests/replies/usage, startup statistics, quality results, independent review and archived prototypes. The current runtime is the verified reference; these were private source-mounted experiments, not qualification of a new installed image. Completing these leftover experiments does not satisfy or replace the separate broader throughput targets in `docs/benchmarking.md`.
