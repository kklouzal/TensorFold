# GB10 Flash Next NVFP4 profile

This guide records the qualified Flash Next profile. The later
[Gemma4 NVFP4 deployment](../../docs/gemma4-cuda-validation.md) now uses the GB10's
LAN port 8888; stop that service before launching this alternative profile.

[compose.nvfp4.yaml](compose.nvfp4.yaml) serves
`dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4`, revision
`be794b990578ef3031eccf9f28e675a289a09ee9`. It uses the measured four-stream
profile: resident checkpoint-native W4A4 experts, six native MTP drafts at
0.70 confidence, RotorQuant iso4 KV, SSD n-grams, image/video input and automatic
CUDA-graph sizing. The configured context is the checkpoint's native 262,144
tokens. Quantized KV changes output; this is not a BF16-equivalence claim.

The earlier [2026-10-18 receipt](deployment-nvfp4-20261018.json) records the then-healthy
`qwen38-flash-next-nvfp4` container at `http://192.168.1.50:8888/v1`, on the old
service's LAN port. Serial/MTP full-token identity, four parallel submitted
requests, known image/video input and normal/streamed strict tool calls passed.
Forty other retired TensorFold containers, including `qwen38-flash-next-tf`,
were removed after preserving their inspect records and logs. Images, model
files, volumes, caches and unrelated containers were retained.

The [qualification](../../docs/gb10-nvfp4-validation.md) records the selected
four-stream throughput and the small observed single-request tradeoff. Model
sampling, EOS, chat template and default reply length come from the checkpoint;
the API can override them. The throughput requests explicitly disabled thinking.
This service retains the template's thinking default.

Copy [nvfp4.env.example](nvfp4.env.example) to a host-owned environment file and
fill in the immutable runtime image, complete model directory, writable cache
and GB10 GPU UUID. NVIDIA CDI must be installed. On the qualified GB10, the local
runtime image is
`sha256:3645d51c532ed42606facc09d86e3b808eec9e0b400f615f9ed23329cb6bc366`,
built from source `297e8487`; later documentation, deployment files and the
test-only EOS fixture correction do not change its application, dependency or
build source. This is a local image ID, not a public registry reference. For a
fresh machine, [build the pinned runtime](foundation.md#build-and-audit) and use
its actual image ID.

```bash
docker compose --env-file /path/to/nvfp4.env -f deploy/gb10/compose.nvfp4.yaml config --quiet
docker compose --env-file /path/to/nvfp4.env -f deploy/gb10/compose.nvfp4.yaml up -d
docker compose --env-file /path/to/nvfp4.env -f deploy/gb10/compose.nvfp4.yaml ps
```

The profile has no Docker RAM, swap, PID or CPU quota. Admission and cache growth
retain an 8 GiB host-MemAvailable reserve; external host allocations can still
change available memory. Contexts share the growing cache pool, so four full
windows are not a fit guarantee. Model files are read-only. Kernel caches,
bounded logs, health checks and `unless-stopped` restart policy are persistent.

This GB10 has a separate Docker event service that overrides container affinity
to ten CPUs on create/start. After starting or restarting this profile, restore
its measured twenty-CPU affinity and inspect it:

```bash
TF_CONTAINER_ID=$(docker compose --env-file /path/to/nvfp4.env -f deploy/gb10/compose.nvfp4.yaml ps -q tensorfold)
docker update --cpuset-cpus 0-19 "$TF_CONTAINER_ID"
docker inspect --format '{{.HostConfig.CpusetCpus}}' "$TF_CONTAINER_ID"
```

The deployment check records actual process/cgroup affinity before accepting
the service. The endpoint is `http://GB10:8888/v1`; its primary model ID remains
`Qwen3.8-Flash-Next`, with `flash-next-gb10` as an alias. The older affine/YaRN
[profile](compose.yaml) remains a separate historical configuration.
