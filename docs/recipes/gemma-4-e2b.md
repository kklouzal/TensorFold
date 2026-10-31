# Gemma 4 E2B and DFlash2 on CUDA

This integration reads the complete original BF16 Gemma 4 E2B conditional
generation checkpoint. It uses the native Transformers 5.17 model, including
per-layer embeddings, shared KV, clipped projections, vision and audio encoders.
The 26B MoE implementation and companion are separate architectures.

The current checkpoint pair is:

| Role | Repository | Revision |
| --- | --- | --- |
| Target | `DavidAU/gemma-4-E2B-it-The-DECKARD-Expresso-ONE-Universe-HERETIC-UNCENSORED-Thinking` | `a91cc4c5eaa277338bff3fd2889725bb6e07c2e0` |
| Companion | `rasyosef/gemma-4-E2B-it-dflash2` | `6eb3ade2e4ae841974d692e9edc177dcaa7e20ff` |

Both repositories must be complete local snapshots. Mount the complete
Hugging Face cache when snapshots contain links to shared blobs. No checkpoint
Python is executed. Preserve the target's own tokenizer, chat template, EOS and
generation metadata; the 26B deployment's template override does not apply.

The implementation requires one GPU, one request lane (`--parallel 1`) and BF16
target weights and KV. The native maximum context is 131,072 tokens. Admission
checks the requested window against the actual memory envelope; a small tested
window does not establish full-context qualification. The companion provides
one through seven proposals (`--draft-rows 2` through `8`). It borrows the raw
target embedding and vocabulary head, with four checkpoint-specified target
features. Target logits retain Gemma's soft cap; draft logits do not apply it.

`--drafter-bits 16` keeps the companion in BF16. The explicit `8` candidate uses
private FP8 copies of proposal projections and its borrowed vocabulary head.
The target and KV remain BF16. Proposal precision may change proposals; standard
rejection sampling must still preserve the represented target distribution.
FP8 and draft depth require actual checkpoint performance and correctness
qualification before choosing deployment defaults.

With `--vision`, the same checkpoint accepts images, timestamped video and user
`input_audio` parts. Audio is decoded to finite mono 16 kHz samples and is limited
to the model's native 30 seconds per segment. Oversized clips are rejected.
Images, video frames and audio use native processors and encoders. The reply is
text; audio input support does not add audio generation. See the
[API](../api.md) for transport shapes. Encoded byte, decoded media, visual-token,
context and request admission limits remain active. URL fetching requires the
separate `--vision-urls` opt-in and existing public URL policy.

## Verification status

Source-mounted diagnostic execution on GB10 with Torch
`2.16.0.dev20261014+cu134` and Transformers `5.17.0` loaded all target and
companion weights. Initial greedy serial/drafted output agrees at prompt lengths
22, 511, 512 and 513; a fixed-seed drafted prefix replay also agrees. These
checks do not establish a sampled target distribution or performance winner.
The stricter checks initially exposed batch-dependent normalization and attention
reductions. The owned decode adapters now reuse fixed-row FP32 RMS reduction
and a uniform attention mask dispatch, retaining BF16 checkpoint operands and
output rounding.

Those initial direct-entrypoint GPU runs bypassed the owned CUDA bootstrap and
mapped a different cuBLAS/driver combination. They are debugging evidence only.
All target correctness, quality and performance gates must be repeated through
`cuda_entrypoint.py` and its protected provider launch. The corrected managed
launch subsequently passed full target/companion loading, bit-exact target
logits for every verification width 1–8, greedy identity at prompt lengths
22/511/512/513 and fixed-seed drafted prefix replay. The 24 normalization
primitive cases also pass all widths. These are source-mounted results;
installed-image qualification remains required. Keep the standard Harness
entrypoint in deployment.

Installed final-image checks, full multimodal and native-context execution,
HTTP/Harness behavior, cancellation/grammar integration and comparative prefill
and decode measurements remain pending. There is no qualified E2B deployment
command or throughput claim yet. Existing 26B measurements retain their original
checkpoint and runtime scope.

The [Compose integration example](../../deploy/gb10/compose.gemma4-e2b.yaml)
and [environment template](../../deploy/gb10/gemma4-e2b.env.example) preserve the
complete read-only HF cache and the Harness entrypoint. Their BF16/eight-row
values are initial correctness settings, not a measured performance selection.
Use a separate container name and port for qualification. Replace the image
placeholder only after qualifying the actual immutable image and resource
envelope. The example intentionally uses this checkpoint's own template.
