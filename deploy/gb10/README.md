This integration reproduces the GB10 Flash Next deployment from repository
source. It includes the native SSD reader, multi-image/video and copy-draft
changes, xgrammar 0.2.8, and the Harness `/v1/tokenize` endpoint. The native
deployment remains 262,144 prompt-plus-output tokens with four request slots.
`provenance.json` records the upstream commits, integrated patch hashes, model
revision, and container stack.

Build from the repository root:

```bash
docker buildx build --load --target runtime \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  --tag tensorfold-gb10-fork:local --file deploy/gb10/Dockerfile .
docker compose -f deploy/gb10/compose.yaml config --quiet
```

The Dockerfile builds and installs a wheel from this fork. It pins the existing
registry base by digest and installs xgrammar without dependency resolution,
preserving Torch, Triton, Transformers, and the CUDA stack. The native reader's
C++ source is included in the wheel. The verification image is a separate build
target with pinned pytest tools; it has no model weights or GPU access by default.

`compose.yaml` preserves the current serving arguments, model/cache mounts,
request slots, memory/swap settings, CPU set, health checks, and restart policy.
Its container name and port are the existing production service's. Rendering the
configuration is read-only; applying it is a service replacement and belongs to
the deployment step after validation.

The Harness endpoint counts the native rendered prompt and image expansion
without generation. It permits counting a history beyond the served context,
while retaining generation admission and independent image safety checks.
It requires `--thinking-budget 0`; callers select budgets per request. Returned
token/prompt inspection is capped at 16,384 tokens, and count-only responses
support larger histories. HTTP preparation has a bounded queue and cancellation.

YaRN is an optional CUDA text-RoPE policy for Flash Next. The CLI option
`--yarn-factor 2 --context 524288` requests a 2× window without modifying the
shared model snapshot. Native operation omits that option. A full 512k validation
must measure memory and long-context quality; four full 512k streams are not
assumed to fit. Start a candidate with fewer slots while retaining vision's
minimum of two slots. The normal startup and runtime memory gates remain active.

Qwen's [official model guidance](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices)
documents factor 2 for 524,288 tokens and factor 4 for approximately one million
tokens. Static YaRN can affect shorter-context quality. A frequency/kernel oracle
does not establish full-model quality at the expanded limit; long-context
retrieval and representative Harness tasks must be checked separately.

CPU verification, from the repository root:

```bash
docker buildx build --load --target verification \
  --tag tensorfold-gb10-fork:verification --file deploy/gb10/Dockerfile .
docker run --rm --network none --memory 3g --memory-swap 3g --pids-limit 512 \
  --cpus 2 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  tensorfold-gb10-fork:verification \
  tests/test_native_ssd.py tests/test_harness_tokenizer.py \
  tests/test_qwen4_exp_yarn.py -q
```

The native SSD tests use synthetic, isolated files and an independent byte
oracle. Harness tests use the native request preparation with a small tokenizer,
including oversized counts, image expansion, validation, framing, and bounded
preparation. YaRN tests compare frequencies against the pinned Transformers
reference and test context/memory admission. GPU and full-model checks require
explicit GPU access and their documented fixtures.
