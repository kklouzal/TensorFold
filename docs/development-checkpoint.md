# Development handoff checkpoint

The retained development branch is `python-0.6`. This checkpoint preserves the
complete pending source, tests and research drafts so development can continue
from a standalone checkout on the V100 host. It is a development checkpoint;
production deployments retain their separately qualified image identities.

## Preserved work

- SM70 owner preparation, Q8 preparation, GDN, routing, HC graph and model
  identity work, including their independent references and controls.
- The private 1Cat expert integration, active-frame preparation, down-projection
  trials and direct raw-IQ2 gate/up prototype.
- Dense prefill alternatives and private B16 codec experiments with their tests.
- The four user-authored root `!` design/research documents. These remain drafts;
  the maintained design authority is the bundle in `ProviderOne Implementation Docs/`.
- Qualification tools and the unfinished offline candidate comparison adapter.

Component receipts belong to their recorded source/image and workload. They do
not qualify this combined checkpoint. Native compilation, full-model quality,
capacity and performance gates remain applicable to each trial before adoption.

## Known unfinished comparison work

The offline candidate adapter in `tools/v100_rotorquant_quality.py` historically
stopped with **71 passing and two failing CPU controls**. The Docker inspect
array boundary is now repaired without relaxing object-only authority readers,
duplicate-key, finite-number, immutable-file or hash checks. The focused source
suite passed82 controls in the pinned Python3.12 media image. The subsequent
qualification/combination tool source batch passed120 tests and11 subtests;
these CPU results do not qualify an installed model or performance comparison.
The existing strict teacher comparator retains its independent contract.

The raw-IQ2 gate/up prototype compiled and passed37 actual SM70 component cases
in the repaired frozen image `71ebf375`, among163 passing native cases. The new
Q8 slots API passed12 cases against independent bit/oracle, poison and graph
controls. The same image passed679 installed CPU tests and11 subtests with real
source/SDK binding. The original compilation failures remain evidence; these
passes do not qualify full-model quality or throughput. The private B16 experiments
remain separate development work, without a serving selection claim.

## Current expert comparison implementation

The private qualification selector independently binds TensorFold Grouped4 or
1Cat gate/up, TensorFold or IQ4_NL Q8_1/DP4A down, and packed or direct raw-IQ2
input. Q2_0 down retains its declared TensorFold implementation. The original
combined TensorFold Grouped4 operation remains an independent benchmark control;
a split TensorFold gate/up+down call does not replace that control by assumption.
Both down providers publish unweighted slots and preserve the original shared
expert and outer weighting/reduction boundary. Additional quantization and
epilogue rounding remain explicit numerical choices, not equivalence claims.

Target verification and optional candidate prefill use sequential32-row panels. The
separate MTP expert bank retains the declared TensorFold FP32 implementation;
no Half conversion or approximate down path is installed on it. Candidate Q8
scratch belongs to each selected target/prefill role, with one panel per role
admitted before loading. TensorFold prefill remains an independent control and
is selected unless the candidate prefill is explicitly requested.

The unique-history graph implementation captures State-independent expert,
linear and HC GPU pieces. Attention, recurrent chains and PLE State handoffs
stay live between those pieces. Startup prepares exact row/participant shapes;
unprepared shapes run eagerly. Serving counters distinguish target and MTP
replays from startup replay. RAM indexer and other drafter graph contracts are
outside this candidate. Source CPU graph controls passed100 tests; actual V100
replay, graph/eager model parity, numerical differences, full-context resources
and matched MTP4 performance remain required before selecting winners.

## Additional V100 library candidates

The pinned 1Cat source uses TurboMind/LMDeploy CUDA kernels, CUTLASS 4.4.2,
GGML quantization primitives, FlashAttention-V100 and FlashQLA GDN kernels.
Some consumers also use TileLang. These are separate mechanisms with different
shape, state and numerical contracts; installing the complete 1Cat dependency
set does not establish compatibility with this engine. The existing source SDK
retains CUTLASS 4.8.1. FlashInfer's inspected attention backend requires SM75 or
newer; its separate SM70 primitive probe does not establish V100 attention support.

Private prepared adapters now assess CUDA 12.9 cuBLAS/cuBLASLt and CUTLASS SM70
dense operations. Their native compilation, 15 CPU checks and 48 V100 component
checks pass using the canonical wrapper and explicitly identified native ELF
on the source8 SDK. JIT-provider and integrated selection remain pending.
The xFormers 0.0.30 attention interface can
reach the compatible ATen CUTLASS operator already built into the owned Torch;
the old xFormers wheel is not installed over that SDK. Its adapter passed five
actual V100 component controls, with explicit normalized-output/LSE numerical
representation. Integration and quality acceptance remain open.

BitBLAS 0.1.0.post1 direct-shape pilots exposed unsafe generated tails; those
original failures remain evidence. The corrected bounded physical-padding
exports passed all20 native value, changed-input capture and Compute Sanitizer
memcheck cases. Row-shape-dependent arithmetic differs from TensorFold and
between SIMT/WMMA variants; the fixed-M32 adapter remains a separate required
control before model use. No quality or performance acceptance is claimed.

Requested FlashAttention-V100 26.10 is a distinct source identity from 1Cat's
bundled and externally pinned variants. Its owned prepared forward binding
passed18 actual V100 component cases using the exact source/ELF closure on the
source8 SDK. The separate GDN Value8 candidate passed19 native controls on the
source11 SDK. These are explicit component bindings, not factory-JIT or
whole-model qualification. Attention normalization/mass representation and
GDN owner/admission integration remain separate gates.


The Triton 3.2.0 candidate retains the last inspected upstream MMAv1/SM70
implementation. Its isolated SDK profile passes nonroot binding and installed
CPU checks, but actual compilation exposed its unsupported CUDA 12.9 version
mapping. The owned launch boundary now selects the supported PTX8.3 option;
the real CUDA12.9 assembler accepted an SM70 compilation without initializing
CUDA. The source10 image then passed two actual V100 controls: bit-exact ordered
FP32 FMA with changed-input graph replay, and a 64x64 FP16 dot product checked
against an independent integer oracle with Volta MMA PTX/CUBIN provenance.
The frozen Source19 NUMA TensorFold reference subsequently passed with actual
Triton 3.2.0/PTX8.3, native CUDA12.9 compiler binding, TP8/ETP4, 24 allocated
262144-token Q8 windows, MTP4 and the 10-key paid graph catalogue. All 24 placement receipts
validated before generation and all eight ranks closed with 75 verify/66 MTP
serving hits. All six complete output-ID sequences match the fresh Source18
reference, including the 64-token serial/MTP pair. Source UID
`216adcab1716d9b72f97eb9232604a388433e46f35ccee75c78a9b6cd3756f55` and image
`sha256:bade98770b97f00d1c582c631893bdce81d37ff078dbffd50d7eccabc1ff248e`
bind this bounded result. Both Source19 reference and Triton3.2 images passed
1,052 installed CPU tests and 235 subtests each, with one existing Docker Compose
availability skip. Broader quality and matched performance remain pending;
no vendor patch, fake compiler version or broad SM70 qualification is implied.

The frozen source7 reference eager model completed four requests with full24
GPU-resident sessions admitted, TP8/ETP4, native context262144, Q8 histories and
MTP depth4. Serial and MTP token IDs matched on the two smoke prompts. This is
functional evidence, not a throughput benchmark. Source12 graph startup exceeds its fixed memory allowance before serving
and remains under diagnosis. Graph replay and all matched
performance/quality comparisons remain open gates.

The private campaign's `model --diagnostic-commit-events N` reserves finite
host storage before engine construction and records committed-output batch
timestamps; `--diagnostic-commit-nvtx` adds numeric markers. A smoke diagnostic
uses the existing known-answer requests; the other diagnostic selects the
original C1/C8 prompts and work from the immutable plan. All eight rank traces
must close, with positive leader work. These records are explicitly ineligible
for performance selection; MTP batch timestamps do not imply individual token
spacing. The ordinary timing campaign retains its separate protocol.

The unselected dense integration now binds actual matrix contracts before
State construction and includes its numerical selection in coordinate/cache
identity. Its optional scheduler handoff completes on the decoding worker
before request admission opens; an ambiguous thread start cannot acquire that
owner. The shared scheduler passed105 CPU lifecycle, FIFO and eight-peer Store
controls with the exact source11 Scheduler/Engine implementation. Two corrected
test fixtures retain separate source identities. Actual selected-dense native
handoff, admission and full-model gates remain pending.

## Continue development

Use the current procedures in [the V100 deployment guide](../deploy/v100/README.md),
[source foundation guide](../deploy/v100/source-foundation.md) and
[benchmarking guide](benchmarking.md). Build containers on the V100 host with
the maintained parallel-build arguments and persistent native caches. Keep
runtime dependencies and verification containerized.

Before releasing a new image, bind the exact committed source, installed
application/native origins, SDK, devices, checkpoint and selected profile, then
run the affected checks. A clean Git checkout and successful transfer establish
source integrity, not target qualification. Ignored caches, compiled outputs,
credentials, model payloads and active service configuration are outside the
source handoff.


## Graph allocation diagnostic

The private campaign runner accepts `--diagnostic-graph-pieces` to enable
startup-only per-piece physical and Torch allocation observations. It records
only the handoff kind, layer index and row count; weight objects and State
pointers remain outside receipts. Samples occur after capture ends, without
per-piece synchronization or cache flushing. Their asynchronous allocation
attribution is diagnostic and these runs are explicitly ineligible for P3.
The default capture path performs no additional allocation queries.

The S12 complete catalog still exceeds its fixed allowance at the 21st key
(target40 rows/eight participants). Explicit cuBLAS workspace caching produced
the same physical deltas and did not resolve that gate. Context, full24-State
capacity and the configured memory reserves remain unchanged. Native component
passes do not qualify complete model graph startup, quality or throughput.


The optional private manifest field `preparation.exact_graph_policy` selects
`paid-subset-v1`: an explicit finite list of `(role, total_rows, participants)`
keys and a SHA256 identifying their cost-evidence packet. It is an unqualified
trial and leaves the existing full catalog as the default when omitted. Startup
validates the list against the actual decoder geometry before capture. Keys
outside the chosen set remain supported through the ordinary eager route;
unexpected per-key or settled budget overruns remain fatal. Policy, chosen keys
and declined coverage are recorded, and the selection is fixed for the engine's
lifetime. This is a distinct policy candidate, with model coverage, quality and
performance verification pending.


The separate private boolean `preparation.native_hc_writeback_capture` trial
prepares the existing owned HC4/D2560/TP8 modes0/3 writeback operator before
admission and enables it only within private exact nonprefill captures. Other
writeback modes and eager warmup, serving misses and prefill retain their
original Torch path. The unchanged operator passed20 actual-row graph cases,
40 changed-input replays and five retained special-value controls on V100 with
its frozen source/ELF binding. This is component evidence; the new core hook's
model capacity, outputs, coverage and performance remain unqualified.


The bounded functional smoke keeps its two short known-answer pairs and adds
one identical serial/MTP continuation pair capped at64 tokens. It requires at
least16 generated IDs and exact full-sequence agreement so the fixture reaches
beyond the initial MTP4 window. Length alone is not a graph-use pass: all eight
closed rank receipts must still show positive target and MTP serving replay.
The original short-only paid-catalog run remains a failed MTP-coverage result.


## Bounded graph/MTP serving milestone

Frozen source15 (`344f93e6`, source UID `20bb0dff`, image `a2ed1829`)
passed the six-request functional protocol with the TensorFold expert reference,
TP8/ETP4, MTP4, all24 full262144-token windows and unchanged resource reserves.
Its paid catalog captured seven target and15 MTP keys (1320 MiB within its
1441 MiB allowance); eight other target keys retain eager execution. Every rank
closed successfully with75 target and84 MTP serving hits. The long serial/MTP
pair matched all64 output IDs, with70 MTP drafts and32 accepted. Actual live
serving coverage was C1; the larger keys were startup captures. These are
functional/resource results, not token/s or held-out model-quality results.

The first maximal 1Cat run on that image failed before Engine construction:
the family adapter still admitted only `_GroupedGateUp` while the qualifier and
Engine already supported exact `OneCatGU`. The corrected adapter admits the
same two exact owned types and preserves nonowned/subclass/duck-type refusals.
Its57 CPU adapter/selection/tool controls pass. A new installed model run is
required; the original failure remains evidence.


The next maximal 1Cat trial passed family admission and real native factory
builds, then stopped before serving because its preparation selector was a
tuple under the strict runtime JSON protocol. Explicit preparation, profile,
resource and receipt wire fields now use lists; internal selector and plan
identities remain hashable tuples. The strict serializer and typed validation
remain unchanged, and actual producer/consumer controls cover all20 selectors.
A corrected installed trial remains required before 1Cat qualification.

The private `preparation.dense_projections` backend `bitblas-hc` uses its sealed
manifest/SHA and target roles, with all other linears and MTP on their existing
providers. Campaign settings supply `bitblas_hc_exports`; the exact ten-file
bundle is validated from the frozen stdlib helper, copied under read-only
`/operator/bitblas-hc`, and revalidated before native loading. Inject padding,
row scratch and copy-kernel residency are funded before State construction.
Local HC readout graphs are explicitly incompatible with selected dense
numerics because they clone reference matrices. Whole-round capture and the
local fallback are now mutually exclusive; no numerical provider is silently
substituted. Model, quality and P3 checks remain pending.


## Required NUMA deployment region

The closed prepared-teacher operation and comparison reader now have CPU
coverage for the incumbent plus20 explicit expert selectors, complete frame
specifications and full-vocabulary conditional KL/NLL/PPL diagnostics. Historical
teacher validators and the guarded native-contract Host lifecycle remain
independent controls. The versioned NUMA/CAP40 bridge now binds four sealed
operator files, the source-owned policy renderer and all 24 placement receipts.
The Host checks exact CPU/memory-node and seccomp authority; before generation
and after retirement, the validator links worker PIDs to startup and native
observer records. Replay retains the complete immutable closure. Matching-image
native permission/admission proof and model execution remain pending. No
dataset, accepted error margin, held-out accuracy or P3 result is implied.

Live Source17 ranks were allowed on both CPU/memory nodes and sampled CPU and
memory placements crossed sockets. GPUs0–3 are PCI-local to node0 and GPUs4–7
to node1. The image already contains libnuma2.0.18; Docker's captured default
blocks policy queries and static binding for UID1000/CapEff0. The owned worker
profile preserves that captured default and adds only the static32770 thread
policy operation to the existing narrow NUMA rules. Real node0/node1 binding,
readback, owned-page location and fork/exec inheritance pass without added
capabilities, unconfined seccomp, package installation or daemon changes.

The common launcher now accepts a sealed UUID-ordered placement plan, assigning
four physical cores plus their allowed SMT siblings per rank on this host.
Actual eight-rank CPU/exec tests pass. New model runs require Docker CPU/memory
allowances, the derived policy, read-only plan and all24 bootstrap/post-CUDA/ready
receipts, including GPU UUID, same worker PID, current thread CPU masks and owned
anonymous/pinned-page locality. The actual eight-GPU Torch/CUDA/NCCL test now
passes all24 receipts, local anonymous/pinned pages and helper-thread masks.
The frozen Source18 TensorFold reference now passes the bounded full-model
NUMA gate: TP8/ETP4, 24 full 262,144-token Q8 windows, MTP4, prefill256/execution120
and the 10-key paid graph catalogue. All 24 placement receipts validated before any
generation and remained unchanged through clean retirement. Six requests
include exact 64-token serial/MTP agreement; every rank closed with 75 verify
and 66 MTP serving graph hits. Captures used 848 MiB within the 1,441 MiB allowance.
Source UID `68ad56e92f8debb9e4a40176c50d1a39013ce04ea28ab2b124f84804b824fe9e`
and image `sha256:d40618a6a17b23a2a05f592d9fc0f6253b68bb12aebd7f5ab3cb7a481a6029e4`
bind that finite result. The separately held host controller includes the
reviewed NUMA settings-parser correction; frozen worker tools and the installed
image were unchanged. The installed gate passed 494 tests and 55 subtests with
one existing platform skip under the required worker policy. The first CPU
procedure's read-only cache failure and the first model controller's pre-Docker
settings rejection are retained.

The same image and NUMA authority also passed the maximal 1Cat profile
(half-epilogue GU, Q8-slot down, raw IQ2 and candidate prefill) and the separate
TensorFold capture-only GDN Value8 profile. Each validated all 24 placement
receipts before generation and closed all eight ranks cleanly. All six complete
texts and token-ID arrays match the fresh TensorFold reference, including the
64-token serial/MTP pair. The 1Cat run recorded 75 verify and 62 MTP serving
hits per rank; Value8 recorded 75 and 66. Value8 reuses 122,880 bytes of existing
Q scratch with no additional workspace tensor allocation, while graph storage
is 856 MiB versus the reference's 848 MiB; this is not total-resource
equivalence. Broader model quality, all remaining expert combinations and
matched new-region performance qualification remain pending.
Shared file-cache pages and necessary TP8 traffic are outside a zero-UPI claim.

The initial unbound TensorFold timing arm completed, but the matching 1Cat arm
was cleanly cancelled after C8 when NUMA became a required region. Both remain
diagnostic evidence; incomplete candidate results cannot establish a winner.
The NUMA region now has a fresh TensorFold functional baseline; matched timing
and candidate qualification remain required.
