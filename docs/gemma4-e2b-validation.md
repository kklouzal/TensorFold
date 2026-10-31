# Gemma 4 E2B deployment validation

The 2026-10-31 GB10 deployment is a correctness baseline for the complete BF16
DavidAU Gemma 4 E2B checkpoint and its BF16 rasyosef DFlash2 companion. The
service is `gemma4-e2b-tf`, port `8888`, with the retained client alias
`gemma4-26b`. The alias identifies the endpoint compatibility name; the loaded
architecture is E2B. The previous 26B container is stopped and retained for
rollback, and the central model inventory records the new service.

## Immutable installed identity

| Artifact | Identity |
| --- | --- |
| Application source revision | `4d19bdd1546cd2417f16af09fd71de37647920e6` |
| Runtime image ID | `sha256:290c8f7f132e6d98ed521a39c4616f63385becb55553009aaf9bb5d666d60d72` |
| Installed wheel SHA-256 | `1cf6613d6a1260e72f0c632856a8731431b57911cf12f6f07b2f00d6bb19ac8b` |
| Rebuilt FD-owner C source SHA-256 | `f85d2c923adfaa14e819250e8def3c3166b8c8624295a37572916e62ad20c1f2` |

All **726 installed application files** matched the captured source exactly,
without an application source overlay. The FD-owner extension was rebuilt from
the recorded C source. The application-only image build used the guarded
`deploy/gb10/Dockerfile.application` workflow, preserving the qualified SDK and
protected CUDA entrypoint/provider verification. It did not replace the SDK
with an unqualified dependency installation.

The [compact receipt](../deploy/gb10/deployment-gemma4-e2b-baseline-20261031.json)
records these public identities. These documents describe the qualified wheel;
subsequent README edits alter wheel metadata and require a new source capture
and image identity when rebuilt.

## Model and launch contract

Target and companion revisions are pinned in the
[E2B recipe](recipes/gemma-4-e2b.md). The baseline retains all target tensors,
BF16 target weights, BF16 KV, BF16 companion projections, eight verification
rows (seven proposals), one request lane and native context `131072`. The
checkpoint's own tokenizer, chat template and generation metadata remain in
use. The 26B template override does not apply.

The model uses per-layer embeddings, shared KV and its complete vision and
audio encoders. Image, timestamped visual video and audio inputs are enabled;
output is text. Video uses frames by default. The optional native HF
`load_audio_from_video=True` loader convenience is not exposed: send a separate
`input_audio` part for the soundtrack. Audio segments are limited to the native
30 seconds; oversized clips are rejected rather than silently truncated.
Existing `logprobs` and decision requests return HTTP 400.

## Evidence and its scope

| Check | Result | Execution scope |
| --- | --- | --- |
| Installed CPU suite | 1,423 passed; 50 documented skips | Installed application; no source overlay |
| Native processor suite | 38 passed; zero skips | Actual processor metadata, including 17 opt-in cases skipped earlier |
| HTTP/Harness qualification | 23/23 passed | Installed service; stored audio/video, thinking, tools/schema, SSE and cancellation |
| Production readiness and retained alias | 4 passed | Deployed endpoint and compatibility alias |
| Verification widths 1–8 | Target logits bit-exact | Protected managed SDK, source-mounted implementation |
| Greedy and seeded prefix replay | Passed | Protected managed SDK, source-mounted implementation |
| Native context capacity | 131,040 prompt tokens + 32 generated continuation tokens in each of four calls: serial greedy, cached drafted greedy, sampled new and sampled replay | Protected managed SDK, source-mounted run before wheel installation; EOS stopping disabled for capacity diagnostic |
| Native-context verifier identity | First eight generated IDs teacher-forced at widths one and eight; hidden states, logits and four draft taps identical | Separate source-mounted comparison at the native-context snapshot |

The native-context result is **not an installed-wheel 131K exercise**. Exact
installed application/source identity and the same protected SDK support
carrying the source-mounted evidence within its recorded scope; they do not
change which execution was measured. Installed HTTP and native processor
checks establish their own coverage, rather than substituting for that context
run. The forced-length capacity diagnostic does not establish long-context
retrieval quality or performance. Earlier GPU runs that bypassed the managed
entrypoint are debugging artifacts only and do not qualify the deployment.

## Performance remains open

BF16/eight-row settings are the verified baseline. Comparative end-to-end
prefill/decode performance selection has not finished; incidental token rates
are not published as a winner. The FP8 companion candidate is not selected by
this receipt. Historical 26B measurements retain their original model, image,
precision and workload scope. Preserve the baseline and rollback artifacts
while further tuning is evaluated under matched conditions.
