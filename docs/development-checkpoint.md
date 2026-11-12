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
Full-model gates remain pending; no vendor patch, fake compiler version or
broad SM70 qualification is implied.

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
