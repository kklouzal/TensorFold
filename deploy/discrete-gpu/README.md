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
`TF_RAM_BYTES` defaults to **56,000,000,000 bytes** (about 52.2 GiB), the
recorded EXL3 profile. The same value sets RAM and RAM-plus-swap limits, adding
no swap allowance. Select a checkpoint-specific limit before launch. `--vram-experts auto` reserves required weights, slot
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

A Hugging Face snapshot can contain links into repository-local or cache-wide
blob stores. Binding only that snapshot at `/model` can leave dangling links
and removes the original cache ancestry used for path authorization. For this
profile, materialize a complete model tree at a fresh destination first:

```bash
TF_SNAPSHOT=/absolute/path/to/hub/models--owner--model/snapshots/recorded-revision
TF_MODEL_DIR=/absolute/path/to/materialized-model
test ! -e "$TF_MODEL_DIR" && cp -aL -- "$TF_SNAPSHOT" "$TF_MODEL_DIR"
```

Allow disk space for the copied payloads, check that the copy completed, and
verify the checkpoint revision/content before using `TF_MODEL_DIR` in `.env`.
A complete owned cache may instead be mounted read-only with its directory
layout preserved and the actual snapshot path supplied to `serve`; a flat
snapshot-only bind does not supply that layout. Do not make the entire host
cache writable to solve broken links. The source loader supports managed HF
shared blobs only through the snapshot's own repository blob link, a real
shared `blobs` directory with regular `.huggingface-shared-blobs` marker `1\n`,
and canonical regular `<2-hex-prefix>/<64-hex-hash>` payloads. Other cache paths
remain outside its authority. See the [HF layout source](https://github.com/huggingface/huggingface_hub/blob/v2.2.0/src/huggingface_hub/utils/_shared_blobs.py).
Keep the selected model files, their path ancestry and links, the shared marker
and admitted payloads immutable while loading or serving. Unrelated cache
entries may change without altering those inputs. Stop their readers before
modifying or collecting selected files.

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

### Select the intended NVIDIA devices

The base profile retains `gpus: all`. On a mixed-vendor host, select the NVIDIA
UUID explicitly through the optional device-request override. Set
`TF_GPU_UUID` in `.env` from `nvidia-smi -L`, then include the same files for
`config`, `up`, `ps`, `logs`, `stop` and later updates:

```bash
docker compose --env-file deploy/discrete-gpu/.env \
  -f deploy/discrete-gpu/compose.yaml \
  -f deploy/discrete-gpu/compose.gpus.yaml config
```

Alternatively, use `compose.cdi.yaml` with `TF_CDI_DEVICE` set to an exact
NVIDIA name from `nvidia-ctk cdi list`, such as `nvidia.com/gpu=GPU-...`.
Native CDI needs Linux Docker Engine with CDI enabled and its NVIDIA
specification installed; Docker enables CDI by default from Engine 28.3.0.
The override clears the base GPU request with Compose `!reset` and suppresses
the image's legacy `NVIDIA_VISIBLE_DEVICES=all` injection. Select one override
mechanism. See [Docker CDI](https://docs.docker.com/reference/cli/docker/container/run/#cdi-devices),
[NVIDIA CDI setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
and [Compose merge rules](https://docs.docker.com/reference/compose-file/merge/#reset-value).

```bash
nvidia-ctk cdi list
docker compose --env-file deploy/discrete-gpu/.env \
  -f deploy/discrete-gpu/compose.yaml \
  -f deploy/discrete-gpu/compose.cdi.yaml config
```

For a supported multi-GPU model configuration, copy the chosen override and
use an explicit device list. For device requests, the replacement is:

```yaml
services:
  tensorfold:
    gpus: !override
      - driver: nvidia
        device_ids: [GPU-first-uuid, GPU-second-uuid]
```

For CDI, keep `gpus: !reset []` and `NVIDIA_VISIBLE_DEVICES: void`, and replace
`devices` with `!override [nvidia.com/gpu=GPU-first-uuid, nvidia.com/gpu=GPU-second-uuid]`.
Include that file last. This selects visibility only: choose the corresponding
supported model parallelism and device arguments separately. Check actual
visible UUIDs and their CUDA indices inside the container before assigning
ranks or loading a model. Request-list order alone does not establish that
mapping. Successful Compose parsing does not establish GPU availability,
execution or multi-GPU correctness.

### Set host capacity and restart behavior

Set `TF_RAM_BYTES` to a positive decimal byte count that fits the checkpoint's
actual resident authority, loading/staging peak, pinned buffers and mapped-file
working set as well as serving state. The default is the measured EXL3 profile,
not a budget for every affine or NVFP4 checkpoint. Raising a cgroup ceiling
does not create host RAM or bypass TensorFold admission. Leave measured host
headroom for the OS, other services, compiler/transient allocations and page
cache. Confirm startup estimates, host `MemAvailable` and cgroup
`memory.current`/`memory.events`; a mapped table's file size alone does not
establish its resident peak. Stop or reduce competing allocations before retrying
an admission refusal. No universal host-headroom byte count is prescribed.

`TF_RESTART_POLICY` retains `unless-stopped` by default. During setup, choose
`no` for a single attempt or `on-failure:3` to limit Docker's failure retries.
Docker already increases restart delays and resets that delay after a run of
at least ten seconds. A retry cannot fix an unchanged capacity refusal; inspect
the first failure and correct its settings. Use Docker as the single restart
owner. See [Docker restart behavior](https://docs.docker.com/reference/cli/docker/container/run/#restart-policies---restart).

`TF_BIND_IP` defaults to `127.0.0.1`, and `TF_PORT` defaults to `8080`. For remote
clients, set a specific trusted host interface and apply the service's access
controls. API-key authentication is available through the serving flags; configure it when exposing the endpoint.
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
Ensure Docker starts at boot if using that policy. The optional `on-failure`
policy does not provide the same daemon-restart behavior. See
[Docker restart policies](https://docs.docker.com/engine/containers/start-containers-automatically/).

## Optional expert ranks and memory policies

The checked-in two-slot EXL3 profile keeps its existing command. Affine4 and
ModelOpt NVFP4 Flash Next models can instead use the same-host NCCL expert
launcher: rank zero owns the backbone, KV, HTTP and supported vision, while
followers load their assigned routed-expert banks. It supports rank-zero shared
request rounds; tensor-parallel `--tp 2` restrictions remain separate. EXL3
expert partitioning is refused.

[Desktop remediation contracts and launch examples](../../docs/desktop-report-remediation.md)
show both explicit UUID and CDI selection, the compatibility launcher before
rank creation, the trusted Harness rank entrypoint, and an explicit larger
shared-memory allocation for NCCL. Check each GPU's UUID-to-CUDA-index mapping,
shard counts, host authority and per-rank memory requirements before choosing
an asymmetric split. Full two-GPU/SM89 qualification remains separate from the
single-device visibility checks.

`--vram-experts-pin off|auto|on` defaults to `off`. Registration is opt-in,
capacity checked and retained through all transfers/consumers; `on` refuses
startup if it cannot be provided. `--kv-full-windows N` is an explicit reduction
in simultaneous long-request capacity for automatic RAM experts and at least
two request slots; it keeps normal precision and can return capacity errors.
Leaving it unset retains reservation for all slots. Full-context capacity and
service workspace must still fit on rank zero.

The [packed affine expert cache](../../docs/packed-expert-cache.md) is also
opt-in. Add `TENSORFOLD_PACKED_EXPERT_CACHE` and
`TENSORFOLD_PACKED_EXPERT_CACHE_BYTES` explicitly to the container environment
or a local Compose override; `.env` interpolation variables are not forwarded
to the container automatically. Put it in service-owned private writable cache
storage and select a disk quota from the actual checkpoint. NVFP4 and EXL3
retain their own layouts; format-aware `--ple-on-ssd` supports the documented
non-EXL3 n-gram formats.

## Readiness, stop and update

Loading and initial compilation can take several minutes. The health check
allows a 15-minute startup period, then polls every 30 seconds with a 10-second
timeout and three retries. Health status does not automatically restart a live
unhealthy process. Inspect startup logs for actual context, cache allocation
and memory admission; a rejected configuration needs a deliberate settings
change. With the pinned Torch runtime, an interrupted builder releases its OS
advisory lock automatically. A persistent `lock` file is not evidence of a live
or stranded owner: do not unlink it. Inspect the actual runtime, cache path,
compiler processes and first failure before diagnosing a build hang.

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
