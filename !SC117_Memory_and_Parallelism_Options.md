# SC117 memory and parallelism options

30 October 2026, America/Phoenix. One or two servers, each with eight V100 SXM2 32-GiB GPUs. This is a compact numerical companion to the ProviderOne design and parallelism research. The revised policy permits duplication experiments of any size, with acceptance based on verified complete-serving gains weighed carefully against added VRAM. **≥1 GB net extra across the whole selected deployment** receives exceptional scrutiny; it is not a per-GPU allowance or a hard cap. The one-copy layout remains the control.

The useful distinction is **one expert bank versus multiple expert banks**. Splitting experts differently costs no additional full expert copy. Replicating only the comparatively small nonexpert portion is much cheaper than replicating the complete target.

These figures govern the current SC117/V100 profile of independent ProviderOne. Mainstream-model, GGUF/safetensors/quantization and later Blackwell support follow the governing design’s phased roadmap. Replacing inherited TensorFold internals is permitted; numerical/state/capacity obligations remain. Do not apply SC117 tensor prices, cache formulas, MTP/ETP assumptions or eight-GPU topology to another model/device without its own census and qualification.

## Selected artifact: inspected headers

Source: non-Swift SC117 IQ3_S, revision `90a3350cd800ec1fe0d1660a38414ad9e985782f`, [immutable directory](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/tree/90a3350cd800ec1fe0d1660a38414ad9e985782f/IQ3_S).

Both GGUF tensor directories were read using bounded HTTP range requests. The first contains 1,223 tensors; the second contains one PLE table. All declared tensor payload sizes reconcile exactly with consecutive offsets and file lengths, with no inter-tensor padding in these files. This establishes source layout/payload arithmetic, not full-file hashes, tensor-value integrity, post-import allocations, executed kernels or measured performance.

| Source component | Exact payload bytes | GiB per complete source set |
|---|---:|---:|
| Routed experts, 48 layers × 512 experts/layer | 50,292,326,400 | 46.838379 |
| All other transformer tensors | 4,760,097,280 | 4.433186 |
| Transformer tensor payload total | 55,052,423,680 | 51.271565 |
| First-shard metadata/header/alignment | 11,024,384 | 0.010267 |
| First-shard file total | 55,063,448,064 | 51.281832 |
| PLE/ngram table payload, held outside GPU memory in this profile | 28,800,138,240 | 26.822219 |
| Second-shard header/alignment | 192 | negligible |

Experts are **91.35%** of transformer payload. The remaining **8.65%** includes attention/GDN projections, HC, routers, shared experts, embedding/head, PLE-side projections and small parameters. The huge PLE table is separate and host-mapped in the control, so it is not part of these GPU-weight prices. A deliberate GPU table/cache variant needs its own measured memory/performance case under the revised policy.

Source tensor counts: Q6_K: 89; BF16: 484; F32: 292; IQ4_XS: 45; Q4_K: 43; IQ4_NL: 39; Q8_0: 96; IQ3_XXS: 34; Q5_K: 31; Q2_0: 9; IQ2_S: 40; F16: 1; IQ3_S: 20. Type 42 is Q2_0: 64 values/18 bytes, verified against source block definitions. Routed expert products have total 120,795,955,200 parameters and average 3.330729 packed bits/parameter including block metadata.

These are **source-format payloads**. They exclude actual required rank-local copies, runtime transformations, padding, codebooks outside the source tensor records, MTP, vision, histories, recurrent/recovery state, graphs and workspace. Dividing 51.272 by 8 gives 6.409 GiB/GPU of ideal balanced unique payload, not a certified runtime weight allocation. Do not substitute the older 13.421 GiB/GPU receipt for this model.

## Supported configuration matrix

The required target matrix is **TP8×PP1/PP2 with ETP8/4/2**. ETP1 is excluded from development and qualification. PP1 uses one eight-V100 server; PP2 uses two physical servers with one eight-GPU stage each. Nonexpert TP remains 8 and stage-local EP=8/ETP. The selected ETP also applies to the single TP8 MoE MTP group; dense DFlash remains TP8 without an expert axis. No other target layout or independent method degree is required.

| Servers | GPUs | TP | PP | Supported ETP | Corresponding EP within each server |
|---:|---:|---:|---:|---|---|
| 1 | 8 | 8 | 1 | 8 / 4 / 2 | 1 / 2 / 4 |
| 2 | 16 | 8 | 2 | 8 / 4 / 2 | 1 / 2 / 4 |

These are six implementation/qualification cases, all retaining one routed-expert representation across the deployment. TP8×PP2’s three cases remain required work pending actual second-server and RDMA execution evidence; all three qualified PP1 cases and SC117 feature gates can support a one-server release meanwhile. ETP8/4/2 gives intermediate widths 80/160/320 and groups owning 512/256/128 experts per layer in the stage. Source encodings and padding still determine actual allocation; degrees do not predict a fixed speedup.

The ideal evenly distributed transformer payload is 6.408946 GiB/GPU for PP1 and 3.204473 for PP2, before rank-local copies, actual layer-cut imbalance, transforms, method, vision and other resources. The source total remains 51.271565 GiB across the deployment. PP2 does not add a second complete target; do not apply the old one-server four-GPU-stage costs to it.

## Optional future TP4×PP2

TP4×PP2 with ETP4/2 only remains a low-priority future implementation/benchmark choice, not part of the six-case support matrix or a current release gate. On one eight-GPU server, four-rank stages allow ETP4×EP1 or ETP2×EP2. ETP8 cannot fit that layout and is excluded from the future study. No TP4 or ETP1 development work is required now. Current TP8×PP2 history/companion costs below cannot certify the smaller stages; use Design Section 20’s future-promotion gates.

## Stage-specific history and capacity

At 262,144 total positions, one QSA cache’s unique growing history is 0.28125 GiB Q8 or 0.1484375 GiB iso4 across its stage. PP1 has twelve target caches plus one with QSA-MTP. PP2’s initial 24/24 split has six target caches on each stage, with MTP on the final stage. At 24 active requests:

| Profile / stage | Q8 history GiB/GPU | iso4 history GiB/GPU |
|---|---:|---:|
| TP8×PP1, thirteen caches | 10.968750 | 5.789063 |
| TP8×PP2, first server: six caches | 5.062500 | 2.671875 |
| TP8×PP2, final server: seven caches | 5.906250 | 3.117188 |

These are ideal history-only stage means. Actual rank balance, guards, raw tails, graphs, workspaces, recurrence/recovery, positional state, fixed vision service and the entire final-stage companion determine capacity. Current FP32 GDN matrix state is 13.5 MiB/session/GPU at PP1 and 6.75 MiB at PP2’s eighteen recurrent layers/stage; versions and convolution/PLE tails are additional. DFlash uses twelve target caches and adds its range-qualified bounded window/features only at the final stage.

For PP1, each additional 1 GiB on every GPU corresponds to approximately 2.188 Q8 or 4.146 iso4 full-context growing-history equivalents. This is memory currency, not an integer admission claim. PP2 needs its actual first/last-stage ledger; spare bytes on one server cannot offset another GPU’s shortfall.

## Selective copies within the supported matrix

The ≥1-GB escalation threshold applies across **the entire selected deployment**: eight GPUs for PP1 or sixteen for PP2. Decimal 1 GB is 1,000,000,000 bytes ≈0.931323 GiB. Include individual and cumulative copies, actual encodings, released allocations and induced steady/peak resources. Existing replicas are not charged again. All accepted copies need verified complete-serving gains; big-ticket additions require exceptional scrutiny. No copy budget authorizes another target/expert routing topology.

| Example | Source-payload increment across deployment | Review implication |
|---|---:|---|
| Full target indexer Q/K projections on all eight ranks of the owning stage | 0.256348 GiB | Below 1 GB alone; charge actual pre-existing placement and cumulative additions |
| Full target routers on all eight ranks of the owning stage | 0.821915 GiB | Below 1 GB alone; may already exist in the control |
| Full packed target embedding on all eight PP1 ranks | 2.201653 GiB | Exceptional scrutiny; compare owner-served vectors |
| Full packed target head on all eight PP1 ranks | 3.399611 GiB | Exceptional scrutiny; repeated compute can erase communication savings |
| One additional packed embedding set, sharded across PP2’s final eight ranks | 0.314522 GiB | Candidate to remove repeated cross-server MTP/DFlash lookup dependencies; about 40.26 MiB/final-stage GPU |

These comparisons use the selected packed source formats, not certified runtime increments. The embedding is IQ4_XS, 0.314522 GiB total; the distinct Q6_K head is 0.485659 GiB. A full FP16 representation of either is 1.184082 GiB, changing both the budget and review tier. A representation copy must reproduce the selected qualified values/casts and have bounded graph/consumer lifetimes. The selected MTP source payload is 2.584999 GiB and remains separately charged; vision is also separate.

## Implementation and measurements

Use PP1/TP8/ETP8 as the first control and implement ETP4/2 in the same execution path. Qualify the three required TP8×PP2 cases when the second server is available; one-server TP4×PP2 is not a substitute. Compare ETP at fixed PP/hardware using the existing C1/2/4/8/16/24 prefill/decode matrix and actual maximum resident/full-active population. Preserve context, codec, selected method, quality, fairness and idle retained histories. Report one-versus-two-server results as scaling, without inventing linear gains or a single-stream speedup. Stage-local cache/companion/endpoint maps, RDMA progress, complete rollback/restore and no-restart boundaries are specified in Design Section 20.
