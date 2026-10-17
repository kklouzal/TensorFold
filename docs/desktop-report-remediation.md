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

The supplied desktop report measured **71.92 GiB of packed affine expert
authority** for its checkpoint. Its ModelOpt inventory listed **63.3 GiB of
routed experts** and **47.7 GiB of n-gram tables**. These components already
exceed the profile's 56-decimal-GB EXL3 limit before loading workspace; use the
checkpoint-specific `TF_RAM_BYTES` setting and host admission. SSD table reads
change residency, while file pages and the rest of the host still consume RAM.

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
interruption attempts bounded retirement of owned rank/compiler sessions and
preserves errors. An unresolved child or native descriptor handoff remains an
explicit retained-owner failure requiring rank-process containment; it is never
reported as successful cleanup. Rendezvous binds loopback explicitly, which
limits listening interfaces and supplies no peer authentication. Local ranks
under the service account and the immutable checkpoint authority are trusted.
The POSIX listener permits reuse after properly retired sessions, including new
sessions that leave accepted-side `TIME_WAIT`, while an active listener remains
exclusive. Older non-reuse sockets can still require their timeout to expire;
there is no automatic port change. [NCCL discovery and recovery](../RUNBOOK.md#nvidia-gpus)
also distinguish the active Python wheel from system libraries and explicit overrides.

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


## Installed local qualification

The later [smaller pretrained NVFP4 record](small-nvfp4-validation.md) qualifies
a dense ModelOpt 9B checkpoint on this same installed image. The full ModelOpt
and offload gaps in this desktop-report section concern Flash Next's separate
routed-expert/SSD-table workload.

Frozen runtime `b551e68f` was built as the ordinary installed wheel in the
maintained Dockerfile. All 512 packaged owned source files matched the captured
source. Corrected verification images retain the serving image's exact
application layers; the [machine-readable receipt](verification-desktop-report-20261017.json)
names runtime, test revisions, image digests, wheel hash and retained evidence.
One final empty EOF line was removed from the shared ExpertWindow module;
the pinned Python 3.12 check proved its complete compiled code and AST location
attributes identical. The receipt records both file hashes and this narrow
post-qualification cleanup.

The complete CPU rerun passed **7,994 cases** with 177 explicit skips. The full
non-matrix CUDA run passed 2,046 cases with 97 explicit skips and found three stale
EXL3 accounting assertions. The complete corrected module passed all 12 cases,
including the formerly blocked assertion tails. This yields **2,049 distinct
passing CUDA cases**, with no unresolved failure; it is a full initial run plus
an affected-module rerun. Original failed logs remain available. Seven earlier
CPU fixture failures were similarly repaired without weakening their assertions.
The two CPU and 28 CUDA warnings came from unchanged private allocator API calls.

The trained EXL3 service passed the same bounded HTTP checks with registration
off/default KV reservation and registration on/two windows shared by four slots:
strict tool enforcement and explicit unsupported-schema refusal, the actual
31-tool catalog, repeated/serial/MTP and solo/concurrent token identity,
streaming, statistics, metrics and clean shutdown. Fixed reply tokens and SSE
choices match the earlier qualified runtime; three long-context KV-format token
and cache hashes also match. The observed expert pool grew from 3.05 to 3.53 GiB
under the changed capacity policy. Those two starts establish functionality,
not an isolated speed comparison.

The actual native library was NCCL **2.30.7**. Native one-rank Session/model
execution, serial/MTP/concurrent identity, resource release, and the installed
one-child Harness launch/tokenization/chat/retirement gate passed. This host
has one GPU: actual distinct-device NCCL execution, SM89, full reported affine
and pretrained ModelOpt/vision workloads remain unqualified. The full KV-pair
matrix was not freshly run; only the exact historical unchanged subset retains
its original qualification. No GB10, Kubernetes or Harness rollout occurred.

## Controlled performance comparisons and selection

These component comparisons used one Linux AMD64 RTX PRO 2000 Blackwell/SM120
host, approximately 62 GiB host RAM, CUDA 13.4.1, PyTorch
`2.16.0.dev20261006+cu134`, and the pinned verification dependency image
`sha256:3d6a57cf578884d60ad360d632dc6be98a082aaceb8a6f869fd39a74752a9cb6`
with recorded source overlays. GPU jobs were serialized; typical limits were
four CPUs, 12 GiB host RAM without additional swap and an 8 GiB device-allocation
cap. WI6 used a 12-decimal-GB host cap. Clocks and thermals were not locked;
the routing record observed a 70 W power limit and Gen3 x8 link.
Each study retained exact output/byte checks and its own preparation, ordering
and uncertainty; gains from separate studies cannot be added together. These
measurements identify their component sources below. Installed qualification
of frozen runtime `b551e68f` is a separate record.

### WI1: registered authority and direct DMA

Source `dbe143e7` compared registered direct uploads with the retained pageable,
two-stage path in a complete generated affine Flash Next workload: two main
layers plus MTP, hidden width 2560, expert width 640, 176 routed experts per
bank, 531 logical expert entries and a fixed 32-cell cache. Each entry was
3,072,000 bytes. Replies contained 32 greedy tokens, context capacity was 128,
graphs were off, and MTP used depth 2/confidence 0. The measured objective was
wall time through complete replies with synchronized GPU completion.

| Workload | Registered/pageable latency ratio | Conditional interval |
| --- | ---: | --- |
| One request, serial | 0.883706 | 0.879684–0.887747 |
| One request, MTP | 0.931315 | 0.928525–0.934114 |
| Four requests, serial | 0.942278 | 0.936870–0.947718 |
| Four requests, MTP | 0.958868 | 0.954882–0.962869 |

Ratios below 1 indicate lower elapsed time. Single-request rows used 24 balanced
four-arm blocks and 99.1667% paired log-ratio intervals for a six-comparison
family. Four-request rows used 12 balanced Williams-order blocks and 99.6429%
intervals for a 14-comparison family. Both used the larger of ordinary and
predeclared Bartlett lag-4 variance. Kernels were warm; every complete token
sequence matched its reference. Single-request copied-byte counts also matched.
Parallel copied-byte counts differed between the pageable and registered arms
in six of twelve serial blocks and nine of twelve MTP blocks; all counters were
retained. These complete wall-time comparisons include the observed cache-work
differences and do not isolate a pure DMA transfer effect. The four-request test
used the maintained **MultiDecoder/Stream combined prompt/decode/MTP path**;
HTTP and outer scheduler-queue costs were excluded. These finite-run intervals
are conditional on this host and workload, not population or tail guarantees.

Direct registered uploads are selected for this controlled region.
`--vram-experts-pin` stays opt-in/default `off`: the trained/full-scale affine
authority and other hardware still require qualification. The existing
pageable copy/lease body is retained; comparisons whose intervals cross 1 are
not treated as proof of equivalence. The records are
`transfer/synthetic.json`, `pageable-parallel.json` and `disposition.json`.

### WI6: verified packed-affine cache hits

At source `c11fb828`, a complete generated affine model with two main layers,
shared experts and MTP used 64 experts, hidden width 1024, expert width 128,
a 108,431,424-byte checkpoint and 47,923,200 bytes of packed authority. Fresh
engines read warm source pages and compiler caches, with MTP2 and eight
completed sampled reply tokens. After two warmups per arm, seven fixed AB/BA
pairs compared canonical packing with verified owned-copy hits.

Constructor medians were **562.817 → 384.236 ms**; the paired median saving was
178.300 ms, conditional 95% bootstrap interval **170.363–203.924 ms**.
Constructor plus completed first reply was **759.455 → 568.361 ms**, with a
paired saving of 190.670 ms **[169.740, 206.695]**. Both improved in all seven
pairs. First-reply-only evidence remained inconclusive: its saving interval
was **−0.622–19.951 ms**. The separate first cache-publication constructor took
818.066 ms; no cold-disk or first-start gain is claimed. All authority bytes
matched native packing and every complete reply matched the resident teacher.

Persistence is selected for this measured warm-start region and remains
explicitly configured with a disk quota. The source, payload hashes, raw pairs
and resource observations are recorded in `packed/receipt.json` and
`startup.json`. This does not qualify the reported 72-GiB affine checkpoint or
predict its startup time.

### NVFP4: common owned-item grouping

Delta `a380e599`, measured in source `8b9bb6a8`, replaced duplicate grouping
with the common owned-item helper. The reference was the exact `run_picks`
method from `2d7c5dd5` in the same dependency/runtime context. A generated
64-expert ModelOpt model, hidden width 512, two main layers plus MTP, used a
two-cell expert pool, context 128, SSD n-gram reads, graphs/prefetch off and
16-token greedy replies; MTP used depth 2/confidence 0. Native NCCL world size
was **one**, exercising the real local Session/engine path without qualifying
two GPUs.

| Workload | Common/reference latency ratio | Conditional 95% interval |
| --- | ---: | --- |
| One request, serial | 0.837470 | 0.833888–0.841068 |
| One request, MTP | 0.845145 | 0.840705–0.849609 |
| Four requests, MTP | 0.896214 | 0.890276–0.902193 |
| Four requests, first cohort in fresh process | 0.935020 | 0.925032–0.945116 |
| Four requests, warm process mean | 0.900611 | 0.883091–0.918479 |

The first three rows used eight ABBA blocks/32 timed operations after warmup.
Initial four-request serial evidence was inconclusive
(**0.956762 [0.823250, 1.111927]**); its 485.286-ms candidate cohort was retained.
One predeclared follow-up used **12 alternating-order fresh-process pairs**:
one first cohort, two untimed preparation cohorts and four timed warm cohorts
per process, starting from the same frozen compiler-cache snapshot. Every one
of the 168 measured/preparation cohorts matched the resident teacher; none
was excluded. First and warm intervals use paired log-ratio Student-t analysis
with 11 degrees of freedom.

A separately labeled prior-inclusive sensitivity retained all eight earlier
block ratios, including that slow cohort: **0.922665 [0.874035, 0.974000]**.
Its mixed block/process units are supplemental to the separately reported
first/warm strata. All three upper bounds met the predeclared selection rule;
the common helper is selected within these distributions. Records:
`nvfp4/common-benchmark.json`, `common-modes.json` and
`serial-four-followup/{predeclare,analysis}.json`. Full pretrained ModelOpt,
SM89 and actual second-GPU performance remain unqualified.

### WI2: preserve the serial gain; do not ship the complete policies

A dedicated copy stream alone was slower than registered current-stream DMA:
serial ratio **1.003639 [1.001547, 1.005737]**, MTP
**1.001853 [1.000775, 1.002933]**, at 99.1667% interval coverage. Stronger
advisory-prefetch variants then retained current-stream foreground DMA and
predicted at most four or eight cells on a separate stream. Fresh seed-23
holdouts used 24 balanced single-request blocks and 12 actual four-request
blocks, with 99.5833% family-adjusted intervals and the same lag-4 sensitivity.

| Predicted-cell limit | Serial latency ratio | MTP ratio | Four serial ratio | Four MTP ratio |
| --- | ---: | ---: | ---: | ---: |
| 4 | 0.982619 | 1.005116 | 1.008468 | 1.008205 |
| 8 | 0.992126 | 1.005250 | 1.013304 | 1.013179 |

Both serial gains are established; all three other required-region intervals
lie above 1. Those regressions prevent selection of either complete policy.
The stronger four-cell serial gain is **1.74%**, interval
**0.980336–0.984908** for its latency ratio. It remains a positive result for a
possible serial-only dispatch, **blocked on trained/full-scale affine input
qualification and required dispatch/retirement verification**. The reported
71.92-GiB authority exceeds this host's RAM; the available trained checkpoint
is EXL3. No serial-only dispatch, prefetch backend or experimental toggle ships.

Eight pageable staging buffers were also inconclusive in all four tested
regions; no equivalence or replacement is inferred. Four CPU copy workers
regressed all four. Exact active-prediction, output, copy and lifetime controls
are retained in `transfer/disposition.json`; copy-byte-per-wall-time counters
are diagnostic and do not establish physical PCIe bandwidth or its causal
contribution. The bounded exploration is closed without a global-optimum claim.

### WI3: retain CPU-owned residency

The private GPU-residency/mapped-fill prototype was corrected to preserve the
incumbent touch-based aging policy; earlier wins with different aging semantics
are excluded from selection. The corrected model repeated seeded layer payloads under distinct logical keys:
32 main layers plus MTP, 2,145 keys, 6,325,862,400 authority bytes and a 128-cell
pool, with eight decoded tokens. It compared complete synchronized functions against CPU-owned
residency/direct DMA: real serial/MTP graph execution and an eager two-stream
concurrent region, six alternating-order paired blocks each.

| Region | GPU-residency/CPU-DMA latency ratio | Conditional 95% bootstrap interval |
| --- | ---: | --- |
| serial | 1.022667 | 1.015350–1.029602 |
| mtp | 1.043388 | 1.038664–1.047446 |
| concurrent | 1.053673 | 1.051622–1.055568 |

All token sequences, tested logical plans and exact policy checks passed, but
every measured required region regressed. Even these optimistic prototype
timings excluded a final status oracle and did not qualify production boundary,
lifetime or capture validation. No mapped-residency backend is selected.
Independent registered-memory mapping controls establish capability on this
host; they do not qualify the rejected prototype for the new allocator.
This is a decision about the measured complete designs, not every possible
device-residency algorithm or another interconnect. Records and limitations:
`routing/disposition.json`, `experiment-policy/model-results.json` and the
sealed `routing/evidence-manifest.json`.

Raw records referenced above are members of the retained
`tensorfold-report-remediation-20261017` evidence packet; transfer/routing
manifests and the packed receipt identify their content hashes. These controlled
comparisons establish no 5090/4080, PCIe5, full trained affine/ModelOpt,
long-context, HTTP tail-latency or general combined speedup prediction.
