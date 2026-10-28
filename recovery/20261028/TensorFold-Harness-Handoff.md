# TensorFold launch configuration for Harness

Prepared 2026-10-06 UTC. This is a launch handoff; saving it does not deploy Harness or restart TensorFold.

## Target and current state

- TensorFold host: GB10, `192.168.1.50` (SSH alias `gb10`).
- OpenAI-compatible endpoint: `http://192.168.1.50:8888/v1`.
- Existing container: `qwen38-flash-next-tf`.
- Served model ID: `Qwen3.8-Flash-Next`.
- Audited runtime: TensorFold 0.6.1, CUDA, with the site's maintained native SSD reader.
- Harness project: `/Warehouse/Harness`. TensorFold provider integration is implemented in the source, with provider type `tensorfold`.
- The running TensorFold container remains on its original image. Harness deployment and provider/default records have not been changed.

## Required image and launcher

Use the image already built on GB10:

```text
harness-tensorfold:20261005-candidate
```

Verified image configuration digest:

```text
sha256:67d8a210ff82ffdd1263e6d0a63577ec37d189ee145ff7bd5e3aff53588b19e1
```

This derivative retains the original TensorFold/native SSD/CUDA stack and includes:

1. The Harness compatibility launcher at `/opt/harness/tensorfold_harness.py`, which supplies exact, generation-free `POST /v1/tokenize` counting, including tools, thinking options, images, video, and oversized history awaiting compaction.
2. `xgrammar==0.2.8`, needed for schema-constrained JSON used by Harness memory workflows. Existing base dependencies were preserved with `--no-deps`.

Stock TensorFold 0.6.1 has no launch flag that supplies the missing counting endpoint or installs the grammar dependency. Flags alone on the original image are insufficient for full Harness compatibility.

The derivative image's entrypoint is:

```text
/opt/nvidia/nvidia_entrypoint.sh python3 /opt/harness/tensorfold_harness.py
```

## Complete container command arguments

Pass the following arguments to that image's entrypoint:

```text
serve /root/.cache/huggingface/hub/models--TensorFold--Qwen3.8-Flash-Next-MLX-4bit-MTP/snapshots/2b170fa6309d5d1ee380b35636075fac7945f286 \
  --backend cuda \
  --host 0.0.0.0 \
  --port 8888 \
  --name Qwen3.8-Flash-Next \
  --parallel 4 \
  --context 262144 \
  --kv-dtype int8 \
  --mtp-drafts 4 \
  --mtp-confidence 0.50 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --max-tokens 32768 \
  --decode-share 0.5 \
  --ple-on-ssd \
  --vision \
  --vision-max-images 50 \
  --thinking \
  --thinking-budget 0
```

This block is the container command argument list, not a standalone shell executable. When invoking the launcher manually inside the derivative image, prepend `python3 /opt/harness/tensorfold_harness.py` before `serve`.

These settings preserve the current deployment's tuning and precision. `--backend cuda` makes its existing backend explicit. Runtime context `262144` includes both prompt and reply tokens; `--max-tokens 32768` is the default total generated-token budget, including reasoning.

`--vision` enables image and video input for this supported model, and `--thinking` enables the reasoning default. Global `--thinking-budget 0` is required by the compatibility launcher: native zero means unlimited, and Harness sets each request's thinking policy/budget independently. A zero thinking budget requested by Harness explicitly disables thinking.

Tools, streaming, required/named private-function calls, cache usage and native timing reports need no additional feature flags.

## Optional flags and feature limits

- `--vision-urls` enables supported public remote media URL retrieval. Harness's data URL inputs do not require it.
- No global `--reasoning-effort` is required; Harness discovers distinct supported levels and sends the selected effort per request.
- Additional cache/checkpoint/disk settings are deployment tuning choices, not missing feature-enablement flags. Preserve existing values until separately measured.
- The parallel CUDA path reports prefill/decode durations and cached-token usage but omits `decode_tps`; missing throughput remains unknown.
- Embeddings, speech and image generation require their existing separate Harness providers. This TensorFold server does not supply those APIs.

## Activation notes for the receiving agent

Inspect the current container before replacing it. Preserve its GPU access, networking, IPC, resource limits, environment, model/cache mounts, restart policy and existing operational settings. Retain the original image and launch definition as rollback inputs.

The site's recipe is `/home/kklouzal/services/qwen3.8-flash-next/recipe/start.sh` on GB10. It constructs `tensorfold serve ...`. With the derivative image's entrypoint, the command must begin with `serve ...`; changing the image name alone would produce incorrect arguments. Use an owned launch override or deployment definition and keep customization outside vendor source.

The original serving image configuration digest is:

```text
sha256:c701ad0b594e129dfb5b295b2945aaf723c614cc6a9e3bd76b698564e601bef8
```

## Verification after activation

```bash
curl --fail http://192.168.1.50:8888/health
curl --fail http://192.168.1.50:8888/v1/models
curl --fail http://192.168.1.50:8888/v1/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Harness compatibility probe."}]}'
```

The tokenizer response must contain `harness_tokenizer_version: 1`, model `Qwen3.8-Flash-Next`, a positive runtime `context_length` and a nonnegative integer `count`. Verify schema output, thinking controls, image/video counting and generation, and actual Harness admission after activating the candidate image and deploying the Harness change.

Completed verification includes 444 targeted backend tests with isolated PostgreSQL, 523 frontend tests, seven bridge tests, exact native preparation for text/tools/images/video/schema, strict grammar acceptance/rejection, and two real production-adapter chat/private-function probes against the original server. Candidate-image GPU generation and full deployed Harness admission have not been certified. Seven broader-suite failures reproduce on the original Harness source.

## Source and evidence locations

- Deployment runbook: `/Warehouse/Harness/deploy/tensorfold/README.md`.
- Derivative Dockerfile: `/Warehouse/Harness/deploy/tensorfold/Dockerfile`.
- Compatibility launcher: `/Warehouse/Harness/scripts/tensorfold_harness.py`.
- Provider documentation: `/Warehouse/Harness/docs/providers.md`, TensorFold section.
- Integration report and logs: `/Warehouse/Harness/review/tensorfold-20261005/`.
