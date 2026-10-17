# Flash Next desktop memory and expert ranks

These contracts describe the desktop integration added after the CUDA
correctness qualification at `5daa5911`. They cover Flash Next CUDA RAM experts,
optional host registration and affine persistence, shared KV-growth reservation,
format-aware n-gram SSD reads and same-host NCCL expert ranks. The earlier
[CUDA receipt](verification-cuda-remediation-20261017.json) retains its own source
and image identity. Qualification of this integration must name its final
source, image, workload and hardware separately.

The [discrete-GPU guide](../deploy/discrete-gpu/README.md) supplies the storage,
permissions, UUID/CDI selection, RAM limits and Docker restart policy. Its
checked-in EXL3/two-slot profile remains unchanged unless the operator selects
another command or policy. These features do not establish a throughput gain
on the reported 5090/4080 desktop or make a checkpoint fit its available RAM.

## RAM experts, host registration and storage

`--vram-experts GIB|auto` keeps immutable packed expert authority in system RAM
and caches unchanged entries in a bounded VRAM pool. Affine4/group-32, native
EXL3 and supported ModelOpt NVFP4 layouts have their own loaders and byte
accounting. GPU computation follows the selected model's existing arithmetic;
cache residency does not choose different experts or precision. Flash Next's
NVFP4 grouped experts retain W4A16 execution, including on SM120; RAM offload
adds no activation-quantized FP4-MMA path.

`--vram-experts-pin off|auto|on` requires supported CUDA RAM experts:

| Mode | Contract |
| --- | --- |
| `off` | Default. Keep pageable authority and the existing bounded staging path. |
| `auto` | Admit complete authority backing and future host allocations before loading. Explicit capacity/capability refusal can retain valid pageable bytes; unexpected driver, I/O or cleanup failures remain errors. |
| `on` | Require admitted host registration. A refusal stops startup explicitly. |

Registration belongs to newly allocated model authority. Eligible live
registered ranges may upload directly into the existing GPU expert pool;
pageable ranges keep bounded pinned staging. Source allocations and successful
registrations outlive their accepted copies and kernel consumers. Resource
snapshots are admission observations, not reservations against other processes.
Include loading, staging, mapped pages, page rounding and other host users when
setting the container limit. A large pageable authority fitting RAM does not
by itself prove that registering all its pages is safe. The default remains
`off` pending representative end-to-end selection for an enabled policy.

With `--ple-on-ssd`, supported affine, BF16, FP8 and NVFP4 n-gram tables use
validated format-specific rows and bounded positional reads. Their decoded
BF16 bits and scale handling match the mapped path. The reader drains accepted
lookups before retiring file descriptors. Linux read-ahead advice does not
promise zero page-cache residency. EXL3 retains its native mapped n-gram table
and refuses this flag. `--ssd-experts` remains an MLX checkpoint-streaming
option; CUDA expert caching uses `--vram-experts`.

### Packed affine persistence

The [packed expert cache](packed-expert-cache.md) is disabled by default.
Enabling it requires both environment variables, with a positive decimal disk
quota selected from the checkpoint's packed authority and available storage:

```bash
: "${TF_PACKED_CACHE_DIR:?Set an absolute private service-owned cache directory}"
: "${TF_PACKED_CACHE_BYTES:?Set a positive checkpoint-specific byte quota}"
export TENSORFOLD_PACKED_EXPERT_CACHE="$TF_PACKED_CACHE_DIR"
export TENSORFOLD_PACKED_EXPERT_CACHE_BYTES="$TF_PACKED_CACHE_BYTES"
```

For Docker, pass these two variables with `-e`, or add them to the Compose
service's `environment`. A Compose `.env` file alone does not forward arbitrary
variables. Keep the cache outside the read-only checkpoint mount, owned by the
service UID and inaccessible for mutation by other users. The quota covers
managed payloads and publication workspace, with the documented entry-count
bound; filesystem metadata and unrelated files need additional disk room.

Hits verify the contributing checkpoint bytes, typed geometry, owned expert
order, layout version and cached payload integrity. Final authority owns its
bytes; it is not a memory map into a mutable cache artifact. Missing, stale or
corrupt entries are reported and canonically repacked. Source/access/resource,
I/O and retirement errors preserve their failure context and stop startup.
There is no reuse based only on mtimes. This persistence path covers affine4;
EXL3 and NVFP4 keep their own source representations.

## Shared KV reservation

`--kv-full-windows N` is opt-in. It requires a positive integer,
`--vram-experts auto`, `--parallel 2` or more, and `--tp 1` on Flash Next CUDA.
It also applies to rank zero in expert-rank mode. Leaving it unset reserves
full configured KV growth for every slot. Explicit N reserves the largest N
slot-growth requirements, capped at the slot count; each slot's initial state,
snapshot storage, resize overlap, MTP and service workspace remain accounted.

This is an operator-selected reduction in simultaneous long-request capacity.
The bytes released from full-window reservation can increase the fixed expert
pool. Precision, context limits and successful-request token semantics remain
the same. It does not promise N fully occupied windows under every additional
allocation or workload, and never permits growth past an explicit absolute
CUDA cap.

Under pressure the scheduler can evict idle prefixes and wait for other work
to progress. If all decoders are blocked, it can terminate a newer decoder with
an explicit capacity error; an isolated request that cannot grow is also
refused. A lone decoder cannot escape the selected shared budget while prompts
are filling. Shorten the prompt/reply, reduce concurrent long requests or
context, or increase N and recheck admission. The default all-slot reservation
retains its existing behavior.

## Same-host expert ranks

`--expert-ranks` defaults to 1. Two through four ranks may partition supported
affine4 or ModelOpt NVFP4 routed experts across distinct GPUs on one Linux host.
EXL3 expert partitioning is refused. Expert parallelism requires `--tp 1` and
`--vram-experts GIB|auto`; it is separate from tensor parallelism. In particular,
the existing `--tp 2` restriction on Flash Next shared request rounds still
applies to that tensor-parallel mode.

Rank zero owns HTTP, tokenization, backbone, router, attention, KV state,
shared-expert work and supported vision. Followers own only their disjoint
routed-expert banks, including corresponding MTP banks when drafting is enabled.
Rank-zero `--parallel` and supported `--vision` with at least two slots remain
available. Each device and the aggregate host authority/workspace must pass
admission; the leader's backbone and request state still need to fit one GPU.
`auto` sizes each rank from its available room. A numeric expert-pool budget
is passed to every rank.

`--expert-shards COUNTS` gives one positive routed-expert count per rank. Counts
must sum exactly to checkpoint `num_experts`, with at most 1023 routed experts.
Ownership is contiguous in logical expert-ID order, permitting unequal counts;
there is no inferred split ratio for a particular pair of cards. Measure the
complete workload before choosing a split for throughput.

NCCL transports disjoint **unweighted per-slot outputs**. Rank zero places each
returned output into its original route slot, then the normal model consumer
applies weights and combines slots in the existing order. No floating-point
cross-rank reduction replaces that arithmetic. This preserves the selected
same-format contract; it does not promise equality across different model,
weight/KV precision or runtime settings. This implementation retains explicit
host routing preparation and does not claim device-only routing or decode
CUDA-graph capture for distributed expert work.

Use `python3 -m tensorfold.cuda.expert_launch` to own all rank sessions. It takes
one through four distinct visible GPU UUIDs or CUDA ordinals with `--devices`,
the shared `--expert-shards`, and an optional `--master-port` (default 29551).
It supplies rank/topology flags and loopback rendezvous itself; do not repeat
`--tp`, `--rank`, `--expert-ranks`, `--expert-shards`, `--master` or
`--master-port` after the `serve` separator. The group checks distinct devices,
a common host epoch and matching checkpoint/configuration before expert work.
The launcher requires Linux pidfd/waitid session ownership. Failure or
interruption retires owned rank/compiler sessions with bounded cleanup and
preserves errors.

The launcher uses the package CLI by default outside the container. In the
fork container, `TENSORFOLD_EXPERT_ENTRYPOINT=/opt/harness/tensorfold_harness.py`
selects the trusted installed Harness for each rank, retaining `/v1/tokenize`
and its metadata on rank zero. An explicit `--rank-entrypoint FILE` overrides
that setting. This is an operator-controlled trusted Python program, never a
checkpoint-provided script. Keep `--thinking-budget 0` for the Harness.

### Container example with explicit devices

Use a newly built fork image, a complete immutable affine or ModelOpt NVFP4
checkpoint and service-owned cache storage. Set checkpoint-specific host RAM
and expert counts from actual headers; the available local EXL3 checkpoint is
unsuitable for this expert-rank example. The following commands use Bash.

```bash
: "${TENSORFOLD_IMAGE:?Set the built fork image ID or recorded tag}"
: "${TF_MODEL_DIR:?Set the complete immutable model directory}"
: "${TF_CACHE_DIR:?Set the writable service cache directory}"
: "${TF_RAM_BYTES:?Set a checkpoint-specific positive RAM byte limit}"
: "${TF_EP_GPU0:?Set the first visible NVIDIA GPU UUID}"
: "${TF_EP_GPU1:?Set the second visible NVIDIA GPU UUID}"
: "${TF_EP_SHARDS:?Set two positive counts summing to num_experts}"
TF_EP_DEVICES="$TF_EP_GPU0,$TF_EP_GPU1"
```

Select the UUID device-request mechanism:

```bash
TF_GPU_ARGS=(--gpus "\"device=$TF_EP_DEVICES\"")
```

Or select native CDI using exact registered NVIDIA names:

```bash
TF_GPU_ARGS=(--device "nvidia.com/gpu=$TF_EP_GPU0" \
             --device "nvidia.com/gpu=$TF_EP_GPU1" \
             -e NVIDIA_VISIBLE_DEVICES=void)
```

Check actual UUID-to-CUDA-index mapping inside the container before assigning
ranks. The explicit UUIDs used by the launcher identify the intended devices
without relying on request-list enumeration order. Use the chosen GPU array
in the common command:

```bash
docker run --rm "${TF_GPU_ARGS[@]}" \
  --shm-size 1g --memory "$TF_RAM_BYTES" --memory-swap "$TF_RAM_BYTES" \
  --pids-limit 512 --ulimit memlock=-1 --ulimit stack=67108864 \
  -e TENSORFOLD_EXPERT_ENTRYPOINT=/opt/harness/tensorfold_harness.py \
  -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 \
  --mount "type=bind,src=$TF_MODEL_DIR,dst=/model,readonly" \
  --mount "type=bind,src=$TF_CACHE_DIR,dst=/cache" \
  -p 127.0.0.1:8080:8080 --entrypoint python3 "$TENSORFOLD_IMAGE" \
  /opt/harness/cuda_entrypoint.py python3 -m tensorfold.cuda.expert_launch \
  --devices "$TF_EP_DEVICES" --expert-shards "$TF_EP_SHARDS" --master-port 29551 \
  -- serve /model --backend cuda --host 0.0.0.0 --port 8080 \
  --name Flash-Next-EP --parallel 4 --context 32768 --max-tokens 256 \
  --vram-experts auto --vram-experts-pin off --kv-full-windows 1 \
  --ple-on-ssd --thinking-budget 0 --no-update-check \
  --max-http-connections 32 --max-pending-requests 8
```

The CUDA compatibility entrypoint runs **before** the expert launcher spawns
ranks. Shared memory must also be provisioned at container creation for NCCL;
the 1 GiB ceiling follows the pinned [NCCL 2.30.7 Docker example](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/troubleshooting/runtime_and_mpi_issues.html#shared-memory).
Validate it against the actual topology and host budget. Shared-memory use
counts against the container's RAM budget; that ceiling adds no host capacity. The context, four slots and one shared KV
window are example policy choices, not fit or performance guarantees. Add
`--vision` only for a supported vision checkpoint after its rank-zero workspace
has been admitted. Preserve explicit limits and inspect complete startup and
request outcomes; do not treat a running container as model qualification.

## Checkpoint and validation scope

A flat snapshot-only container bind can break HF links and remove their cache
ancestry. Materialize a complete model tree or mount the complete owned cache
read-only with its original layout and use the actual snapshot path. Managed
shared-HF support authorizes only the selected snapshot's repository blob link,
regular supported marker and canonical shared payload. Keep selected model
files, their ancestor/link identities, admitted shared payloads and the marker
immutable while in use. Unrelated cache entries can change without altering
those selected inputs. Stop their readers before replacing or collecting them.

The local validation host has one RTX PRO 2000 Blackwell/SM120 and the trained
Flash Next EXL3 checkpoint. Small complete affine/ModelOpt fixtures and native
oracles can exercise their component contracts there. Real two-GPU NCCL
expert execution, SM89 execution, the reported full affine checkpoint and a
full pretrained ModelOpt/vision workload require their actual inputs and
hardware; those regions remain unqualified by single-GPU fixtures. Apple and
new ARM64 execution are separate as well. No reported desktop speed target,
blanket latency gain or unchanged-capacity claim follows from these features.
