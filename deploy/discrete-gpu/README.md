# Discrete GPU deployment

This profile targets a native AMD64 Linux host with a separate NVIDIA GPU,
using Flash Next's RAM-backed expert cache. It is separate from the
[GB10 profile](../gb10/README.md), whose unified-memory limits and CPU affinity
do not apply here. Build this fork's `runtime` image with the pinned NVIDIA
stack in the [root guide](../../README.md#build-and-run-the-nvidia-container).
The image's default entrypoint runs the CUDA compatibility launcher and Harness.

The configured deployment has two request slots, a **262,144-token native
context**, `rotorquant-iso3` for both K and V, MTP disabled (`--mtp-drafts 0`), automatic
expert caching, BF16 prompt activations and full default n-gram prefetch. It
preserves checkpoint RoPE settings by omitting a YaRN override. Thinking and
checkpoint sampling defaults remain enabled; the Harness requires
`--thinking-budget 0`. The default reply limit is 32,768 tokens, included in the
total context alongside the prompt. The HTTP and unfinished-request caps are
32 and eight; accepted requests beyond the two active slots can queue.

Three-bit KV is an explicit accuracy/memory tradeoff and can change output.
The 56,000,000,000-byte RAM limit is about 52.2 GiB; the equal memory/swap limit
adds no swap allowance. `--vram-experts auto` reserves required weights, slot
state and scratch before sizing the shared expert pool, including its safety
margin. It does not allocate every reported free GPU byte. See the
[RAM expert contract](../../docs/recipes/ram-experts.md) and
[KV formats](../../README.md#kv-cache-formats).

The 2026-10-10 deployment on the RTX PRO 2000 Blackwell host admitted the
full configured window and completed ordinary and streaming generations.
Same-container restart and subsequent complete generations also passed, with
`unless-stopped` restart behavior configured.
The selected image and measured settings are recorded in the
[root README](../../README.md#full-native-context-deployment).
Startup reserves the full configured KV capacity and service workspace; short
successful generations do not establish performance or quality at a filled
262,144-token window. Earlier shipping API results used a different context,
KV format and concurrency and retain their original scope.

## Configure storage and launch

Use current Docker Compose with [GPU support](https://docs.docker.com/reference/compose-file/services/#gpus)
(`gpus` requires Compose 2.30.0 or newer), Docker Engine, an appropriate NVIDIA host driver and the NVIDIA
Container Toolkit. Follow the [official Toolkit setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
This profile uses rootful Docker with a container UID/GID of **1000:1000**;
generic rootless Docker is outside its qualification.

Run these commands from the repository root:

```bash
cp deploy/discrete-gpu/.env.example deploy/discrete-gpu/.env
# Edit .env: select the built fork image and replace all four host paths.
docker compose --env-file deploy/discrete-gpu/.env \
  -f deploy/discrete-gpu/compose.yaml config
```

Set `TENSORFOLD_IMAGE` to the actual image ID/digest or a recorded local build
tag. Set all four directory variables to absolute paths. Acquire the entire
checkpoint at a recorded revision before starting: the model mount is read-only
and Hugging Face access is offline. Model files must remain immutable while the
service runs. The previously tested 2.05-bpw EXL3 snapshot occupied about 62.8 GB
on disk, including its mapped n-gram table; that is different from resident RAM.
Provide room for compiler caches, temporary files and logs as well.

Create the cache, temporary and work directories before launch. UID/GID
1000:1000 needs write and traverse access to those directories, plus read and
traverse access to the model directory. Choose dedicated service directories
outside the source checkout. Bind mounts refuse missing host directories;
Compose will not silently create root-owned paths. Do not use a shared system
temporary directory for `TF_TMP_DIR`.

The root filesystem is read-only, with all capabilities dropped and
`no-new-privileges` enabled. Writable storage is limited to `/cache`, `/tmp` and
`/work`. Persist `/cache` across restarts; the image's CUDA/Torch/Triton and
architecture cache namespaces remain in effect. Keep cache files while a live
service or compiler uses them. Model/native cache contents are not source code
to edit inside the container.

`TF_BIND_IP` defaults to `127.0.0.1`, and `TF_PORT` defaults to `8080`. For remote
clients, set a specific trusted host interface and apply the service's access
controls. TensorFold's HTTP listener does not provide authentication itself.
The default bridge networking needs no host IPC, host network or Docker socket.

```bash
docker compose --env-file deploy/discrete-gpu/.env \
  -f deploy/discrete-gpu/compose.yaml up -d
docker compose --env-file deploy/discrete-gpu/.env \
  -f deploy/discrete-gpu/compose.yaml ps
docker compose --env-file deploy/discrete-gpu/.env \
  -f deploy/discrete-gpu/compose.yaml logs --tail 100 -f tensorfold
```

The service uses eight compiler jobs, a 256-process limit, and rotated local
logs of 20 MB each with three files. Docker's `unless-stopped` policy restarts
an exited service and starts it after a daemon reboot unless manually stopped.
Ensure Docker starts at boot. Use Docker as the single restart owner rather
than adding a competing host process manager. See
[Docker restart policies](https://docs.docker.com/engine/containers/start-containers-automatically/).

## Readiness, stop and update

Loading and initial compilation can take several minutes. The health check
allows a 15-minute startup period, then polls every 30 seconds with a 10-second
timeout and three retries. Health status does not automatically restart a live
unhealthy process. Inspect startup logs for actual context, cache allocation
and memory admission; a rejected configuration needs a deliberate settings
change.

For the default publication, verify both readiness and a complete generation:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Explain why the sky is blue."}],"max_tokens":256}'
```

Change these URLs when selecting another host interface or port. The health
response should advertise the admitted context. A short successful generation
establishes operation, not full-window stress or answer-quality equivalence.
Check ordinary and streaming requests, complete response consumption, cache
telemetry/resource limits and same-container restart before treating the host
deployment as qualified.

Stop new clients and let owned work retire before stopping or replacing the
service. Compose sends SIGINT and allows 60 seconds before forced termination.
This grace period does not establish that an in-flight full-context operation
can always retire within 60 seconds.

```bash
docker compose --env-file deploy/discrete-gpu/.env \
  -f deploy/discrete-gpu/compose.yaml stop tensorfold
docker compose --env-file deploy/discrete-gpu/.env \
  -f deploy/discrete-gpu/compose.yaml start tensorfold
```

To update, retain the old image/configuration, change `TENSORFOLD_IMAGE` in the
environment file to the new recorded build, and run `up -d` again. Recheck
health, admitted context, complete generations and restart. Restore the old
image/configuration if the replacement fails. Do not update installed packages
inside the running container. Follow [CONTRIBUTING](../../CONTRIBUTING.md) and
update the root README and affected deployment notes when settings, capabilities
or qualified results change.
