# ProviderOne V100 Parallelism Research

Evidence cutoff: 29 October 2026 in America/Phoenix / 30 October UTC. Per-server platform: **Supermicro SYS-4029GP-TVRT, two CPU sockets, eight V100 SXM2 32-GiB GPUs**. Implementation scope refined 4 November 2026: one server for PP1, two for PP2, with fixed unique history placement and prepared execution plans. This report accompanies the [current governing design](https://chatgpt.com/library?file_id=libfile_409ab65a4194819194b7337392258936&account_id=personal) and [Experimental Optimization Candidates](https://chatgpt.com/library?file_id=libfile_f592a45084988191a3072db8ec831c85&account_id=personal). Current-source alignment on 5 November 2026 UTC uses only the [owned `python-0.6` snapshot](https://github.com/kklouzal/TensorFold/tree/c83187878423388715571d194536eba836a64804); the external-research cutoff and historical measurements remain unchanged. The governing design supplies its implemented/pending status map. Source inspection does not qualify an installed build or eight-V100/model/performance execution. The duplication policy and accepted adaptivity scope govern implementation.

ProviderOne is the independently developed successor being detached from TensorFold. The current governing design permits replacing inherited internals and retains SC117/V100 as the immediate new delivery target while preserving selected existing backends, especially owned GB10 deployments. Missing mainstream model/format/device pairings follow their own qualification; owned RTX 4090 and concrete Ampere extensions retain legal native operator/build admission. This report’s topology, counts, six required profiles and speed/memory calculations remain specific to the current SC117/V100 phase; they do not impose TP8, ETP, 262K, MTP or a CUDA12-only stack on every future deployment. Other model/device layouts require their own qualified design and measurements; no current configuration backlog is expanded by that roadmap.

Shared preparation, API/lifecycle, artifact identity and evidence rules are reusable; hot kernels, packed layouts, graph/communication plans and memory ledgers remain device-specific. Do not copy the SM70 foundation onto modern GPUs or count GB10 unified memory as additive host RAM plus VRAM. Preserve same-host rank-shared hashing; required PP2 must verify independently stored node copies and agree content/profile digests without imposing equal cross-node inode/host identities. Current source’s unique TP8 owner path and smaller-world providers do not establish separate ETP/PP implementations. All six required cases retain their explicit development and actual-target gates.

## 1 Supported scope and confidence

**The required supported configuration matrix is TP8×PP1/PP2 with ETP8/4/2.** PP1 uses one eight-V100 server; PP2 uses two physical servers with eight GPUs per stage. EP=8/ETP remains inside each server. All six cases are implementation targets; ETP1 is excluded. The three TP8×PP2 cases remain pending second-server/RDMA execution qualification. No DPA, other nonexpert TP/PP degrees, model/expert-bank replicas or independent method degree is in the implementation backlog.

The selected ETP also applies to MoE MTP in one TP8 group on the sole/final server; dense DFlash has no ETP axis and remains TP8 there. Design Section 20 governs stage/resource maps and qualification. The source research below retains external results and earlier other-layout analyses as **historical comparison evidence only**. Only TP4×PP2 remains an optional, low-priority future implementation/benchmark choice, not a current requirement. Its standard eight-GPU layout supports ETP4/2 only; ETP8 is excluded because each stage has four ranks. Other historical layouts create no implementation/test obligations. The narrowed scope is a development commitment, not proof of an absolute throughput winner.

The initial serving implementation requires one qualified **fixed unique attention-KV layout and fixed canonical pooled-indexer owners**, using four-way attention-KV striping within each stage-local KV-head quartet as the first control. That control is not a measured winner; test fixed alternatives only where a traced bottleneck justifies them. Live attention-degree changes among 1/2/4, history migration, mixed-holder/union-map execution and their transition sweeps are optional benchmark-gated experiments, with no initial release duty. Normal serving selects ready work, legal proposal widths/quotas and prepared kernel/graph tactics inside one prepared maximum execution envelope. Context, numerics, method, structural placement, graph set and maximum allocations remain fixed until a prepared profile or qualified administrator transition is activated.

The user-verified hardware objectives remain **>250 committed decode tokens/s at C1**, **>600 aggregate at C24**, at the stated 262K/Q8 profile, plus **>3,500 processed prompt tokens/s** with the prefill workload/concurrency bound in its receipt. The supplied reference figures do not identify the layout, exact target/method, actual cached lengths, warm-prefix conditions or timer boundaries. Preserve them as objectives; do not assign them to TP8 or PP2. Capacity comes first, then full-active committed decode, with milestone priority **1 < 2 < 4 < 8 < 16 < 24** and no hidden lower-count decode regression.

## 2 Evidence and measurement rules

The following labels apply throughout:

| Evidence | Meaning |
|---|---|
| Reference hardware fact | Vendor wiring/specification, distinct from the installed-unit inventory |
| Published measurement | Primary project report; not independently rerun here |
| Derived arithmetic | Reproducible calculation with stated geometry or idealized assumptions |
| Architectural inference | A conditional mechanism or experiment priority, not measured tokens/s |
| Qualification gap | Information or execution required before promoting a configuration |

**C denotes simultaneously active requests**, not configured slots, clients waiting in a queue, rows, or a cache-capacity ratio. Pure decode excludes prefill and admission. Whole-request output/wall includes them. Cold computed prompt throughput, input/TTFT, prefix-hit speed and RAM/disk restore are different metrics. Use actual token IDs/counts, actual cached history and complete intervals; do not sum overlapping per-request rates as aggregate throughput.

MTP with four proposals ordinarily verifies five target rows/request; the design's illustrative depth-three/R32 profile uses four. Inco block eight ordinarily verifies one anchor plus seven proposals. Accepted output per round is measured separately. A faster verifier, higher acceptance, or more verified rows does not necessarily mean more committed output per second. Configured 262K, a one-token boundary probe and sustained decoding with genuinely near-full histories are distinct evidence.

Search coverage was broad but bounded by public availability: Strata's 50 current branches, 83 PRs through #84, zero non-PR issues and no releases; live report-tree hashes plus relevant upstream community reports; 1Cat's 624 branch heads, latest 100 updated PR records and targeted TP/PP/EP/topology searches; all 48 ReVolta PR records then exposed plus current documentation. This does not claim unpublished/deleted results do not exist, or that every line of unrelated upstream history was audited.

| Repository | Inspected live main | Release / scope |
|---|---|---|
| [Strata-V100](https://github.com/jmnargi/Strata-V100) | `353003069ba1296d7c9706e9d37791da8c71dd58`, 30 October UTC | No release objects; native layer-split PP, principally two 16-GiB PCIe V100s |
| [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) | `4d57473db67eb77a13018d179e4d95169e82044f`, 28 October | Latest release v1.5.1; model-specific SM70 fork, not current upstream vLLM support |
| [ReVolta](https://github.com/Leonccaa/ReVolta) | `f427a87220daede8f64517f1f3ad33e183acdc69`, 29 October UTC | v0.1.4; four-card TP4/DCP2 AWQ reference, TP8/GGUF untested |

Do not translate another project's support guard into a physical hardware limit or the current ProviderOne implementation’s capability. Bind the actual ProviderOne checkout/build before implementation work; inherited module boundaries are replaceable.

## 3 Reference topology and rank placement

The [Supermicro manual MNL-2115](https://www.supermicro.com/manuals/superserver/4U/MNL-2115.pdf), printed pages 17 and 41, Figures 1-8 and 3-4, provides the actual TVRT wiring. Both solid and dashed green connections count for Volta. The generic TRT/TRT2/TRT3 PCIe chart linked from the product family page is not a substitute.

### Weighted adjacency

The following numbers are **manual diagram labels**, not deployment CUDA indices. Each entry counts physical NVLink connections; zero means no directly wired NVLink between that pair.

| GPU | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1 | 1 | 2 | 2 | 0 | 0 | 0 |
| 1 | 1 | 0 | 2 | 1 | 0 | 2 | 0 | 0 |
| 2 | 1 | 2 | 0 | 2 | 0 | 0 | 1 | 0 |
| 3 | 2 | 1 | 2 | 0 | 0 | 0 | 0 | 1 |
| 4 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 2 |
| 5 | 0 | 2 | 0 | 0 | 1 | 0 | 2 | 1 |
| 6 | 0 | 0 | 1 | 0 | 1 | 2 | 0 | 2 |
| 7 | 0 | 0 | 0 | 1 | 2 | 1 | 2 | 0 |

Every GPU has six NVLink ports connecting to four peers. The natural quartets are A={0, 1, 2, 3} and B={4, 5, 6, 7}. Each quartet is fully connected, with mixed single and double connections.

The cross-quartet links are 0–4 and 1–5 with two connections each, and 2–6 and 3–7 with one each. At the V100 link specification of 25 GB/s per direction, this cut has a theoretical **150 GB/s in each direction**, or 300 GB/s summed bidirectionally. These are decimal GB/s wiring limits, not measured application bandwidth. NVIDIA specifies six links and 300 GB/s aggregate bidirectional bandwidth per V100; this is not 300 GB/s per GPU pair. [NVIDIA V100 system architecture](https://images.nvidia.com/content/pdf/dgx1-v100-system-architecture-whitepaper.pdf)

### Derived topology consequences

Enumerating all 35 distinct equal 4+4 partitions of the matrix gives the natural quartets as the unique minimum weighted cut: six connections cross, versus at least ten for another partition. Each quartet retains nine physical connections. Independent quartet collectives leave six of the machine's 24 physical connections unused.

The double-link edges form this complete ring:

**0 → 3 → 2 → 1 → 5 → 6 → 7 → 4 → 0**

Inside the separate quartets, those double edges form open paths, not closed four-GPU rings. A simple ring confined to either quartet must include a single-link edge. This does not prove TP8 faster: fewer participants can reduce latency, while different collective algorithms use several paths. It does disprove the assumption that all TP4 links are wider than all cross-quartet links.

NVIDIA describes decomposing this class of eight-V100 mesh into three interwoven bidirectional rings using all six ports per GPU. Its published large-message collective results are architectural evidence, not small-message LLM timings. [NVIDIA architecture, printed page 26](https://images.nvidia.com/content/pdf/dgx1-v100-system-architecture-whitepaper.pdf)

### PCIe and NUMA

The exact TVRT drawing shows four PEX9765 switches. Each connects two GPUs, a front x16 expansion/NIC slot, and a shared CPU-facing x16 link. One CPU root serves quartet A and the other quartet B. Host traffic from a GPU pair shares its uplink; GPU-to-GPU traffic using a cross-quartet NVLink does not inherently travel through CPU RAM or UPI.

There is no NVSwitch. A path through an intermediate GPU is an explicit communication operation, not a guarantee that arbitrary remote pointer accesses route through that GPU. Verify CUDA peer support, actual transports and NCCL routes. The [official product specification](https://www.supermicro.com/en/products/system/4U/4029/SYS-4029GP-TVRT.php) establishes supported components, not the installed CPU/RAM configuration.

The manual's numbering examples are internally awkward: an illustration caption mentions IPMI while its adjacent table distinguishes IPMI and OS numbering. Use GPU UUIDs and PCI bus IDs to derive the real mapping. Never hard-code the manual's example bus addresses.

The manual is MNL-2115 revision 1.0b, dated 28 February 2020. Its GPU figures were visually checked; the numbering caveat above applies to every suggested rank below. A fully populated eight-card reference fabric is not proof that every link is enabled or healthy in the deployed server.

| Purpose | Initial diagram-label assignment to qualify | Important distinction |
|---|---|---|
| TP8 global KV-head group 0 / group 1 | `[d0,d3,d1,d2]` / `[d4,d7,d5,d6]` | Natural quads; local XOR-1 pairs use double links |
| TP8 collective ring | `d0–d3–d2–d1–d5–d6–d7–d4–d0` exists | NCCL may use different/multiple rings or trees; do not force this as an unmeasured default |
| TP4×PP2 stages | Same two ordered quads | Corresponding boundary pairs d0–d4/d1–d5 are double, d3–d7/d2–d6 single |
| TP2×PP4 stage-order candidate | `[d0,d3] → [d1,d2] → [d5,d6] → [d4,d7]` | Each stage has a double link; corresponding ranks at every boundary have direct links |
| TP1×PP8 stage-order candidate | `d0 → d3 → d2 → d1 → d5 → d6 → d7 → d4` | All consecutive target boundaries use double links; stage balance/feedback still require qualification |
| PCIe-switch/host staging pairs | `{d0,d1}`, `{d2,d3}`, `{d4,d5}`, `{d6,d7}` | Each pair shares one CPU-facing Gen3 x16 uplink and an optional NIC slot |

Diagram CPU0 serves d0–d3, CPU1 d4–d7; motherboard socket names and OS NUMA IDs need not match. The two GPU-switch uplinks per socket do not establish eight independent host links. Gen3 x16 is about 15.75 GB/s per direction before transaction effects. Cross-socket host pages/copies can consume UPI; direct cross-quartet NVLink does not inherently do so.

The exhaustive 4+4 cut counts, in physical links, are **6:1, 10:3, 12:9, 14:11, 16:9, 18:1, 20:1**. These sum to 35. Stronger cross cuts leave fewer internal links, so maximum bisection alone is the wrong TP4 partition objective. An NVLink path through a third GPU also does not establish transparent hardware forwarding for arbitrary peer loads.

The product supports Skylake/Cascade Lake Xeon Scalable CPUs and 24 DIMM slots. Neither CPU model/core count nor populated memory channels, SNC policy, number of NUMA domains, active UPI rate, RAM capacity, storage/NIC paths or available pinned memory has been supplied. Choose thread counts, SIMD dispatch and page placement from live inventory, not reseller configurations or the DGX-1's CPUs.

Initialize persistent bounded worker pools, NUMA/CPU affinity, SIMD paths and framework/OpenMP/BLAS/media thread budgets before allocation. Queues issue ready tasks within bounded task/byte quotas; reserve launch, PLE and communication progress. Extra feedback-based quotas require measured contention benefit. Pool size, affinity, memory policy and library thread settings remain fixed during ordinary serving; a change in request count does not require host reconfiguration.

## 4 Hardware acceleration: what can actually execute

V100 SXM2 product peaks are **80 SMs, 5,120 CUDA cores, 640 Tensor Cores, 125 FP16 tensor TFLOP/s, 15.7 FP32 TFLOP/s, 7.8 FP64 TFLOP/s, 900 GB/s HBM2 and 300 W maximum**. Eight modules have a theoretical 7.2-TB/s sum of HBM bandwidths for balanced concurrent accesses, not for a remote load from one rank. Clocks, power limits, thermals, ECC and useful operation shapes determine sustained performance. V100S PCIe specifications are not this SXM2 product. [NVIDIA V100 data sheet](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf)

| Facility | Actual SM70 capability | Use / constraint for ProviderOne |
|---|---|---|
| Tensor math | FP16 operands, FP16 or FP32 accumulation | Native SM70 TensorOp/WMMA or qualified cuBLAS/CUTLASS paths; preserve required accumulation and finite outputs |
| Packed integer dot product | `dp4a` byte products with INT32 accumulation | Real INT8 hardware, but **not INT8 Tensor Cores**; activation quantization changes require a declared numerical profile |
| IQ/Q2–Q8/NVFP4/FP8 storage | No native low-bit tensor execution for these formats | Decode bounded tiles or use a qualified SIMT/integer route; include scales/codebooks, unpacking, tails and scratch |
| BF16 / TF32 / integer INT8-INT4 tensor MMA / FP8-FP4 tensor math | Absent in the required native forms on SM70 | Reading a source encoding or installing a package does not create these arithmetic paths |
| FP32/FP64/INT32 execution | Supported; separate FP32/INT32 execution resources | Sensitive state/sums and indexing/unpacking can use the appropriate units; mixed-kernel utilization still needs measurement |
| Shared memory and registers | 96 KiB shared memory/SM; >48 KiB per block needs opt-in; 64K 32-bit registers/SM | Head-256 attention and dequant/GEMM fusion must fit without harmful spills or occupancy loss |
| Warp primitives | Independent thread scheduling | Correct `_sync` masks/barriers are required; old implicit warp-lockstep assumptions can corrupt codecs/reductions |
| Global-to-shared asynchronous copy | No SM80 `cp.async` | Use a Volta-compatible staged load/synchronize pipeline; newer source-level async APIs do not guarantee hardware overlap |
| Hopper/Blackwell mechanisms | No TMA, WGMMA, block clusters/distributed shared memory or NVLS | Port the algorithm, not unsupported instructions or modern-GPU throughput estimates |
| NVLink/P2P and copy engines | Compatible peer transfers and stream/event overlap | Verify pair access, transport, correctness and actual overlap; no uniform all-to-all fabric or automatic pooled VRAM |
| CUDA Graphs | Usable with qualified SM70 stack/operators | Cache bounded shape/layout/method families; retain pointer/communicator lifetimes, legal dynamic descriptors and safe buffer reuse |
| MPS / multiple streams | Volta MPS and independent work scheduling exist | Possible measured overlap tools; they do not add memory or replace efficient batching and process-lifetime accounting |

Hardware/instruction references: [Volta tuning guide](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html), [PTX ISA 8.8](https://docs.nvidia.com/cuda/archive/12.9.2/parallel-thread-execution/index.html), [CUDA 12.9.2 programming guide](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-c-programming-guide/index.html), and [Volta architecture](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf).

Keep the governing design's compatible native-SM70 stack and exact loaded binaries. CUDA 13 removes offline compilation below CC7.5; NVIDIA directs Volta development to CUDA 12.9/R580. Newer framework or kernel-library availability alone proves neither source-build compatibility nor execution of the selected packed/QSA/GDN path. Current upstream DeepEP's requirements are not satisfied by V100; an old-source SM70 port is a separate experiment. [NVIDIA architecture-support guidance](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/)

For video-to-text, NVIDIA's current [encode/decode support matrix](https://developer.nvidia.com/video-encode-decode-support-matrix) lists Tesla V100 with one third-generation NVDEC engine and three sixth-generation NVENC engines. Supported input decoding can offload CPU work and keep frames device-local; SDK/driver/codec and processor-equivalence qualification remain necessary. AV1 hardware decode is unavailable. NVENC is not a text-inference accelerator and need not be added to this output service. Do not apply a blanket claim that all data-center GPUs lack video engines to this product.

Both supported CPU generations provide AVX512F/CD/DQ/BW/VL; VNNI requires Cascade Lake, and neither provides AVX512_BF16 or AMX. Bind actual CPU/OS vector support, execution resources and sustained frequency before selecting tokenizer/hash/packing/media routines. CPU SIMD remains useful for host PLE, input preparation and bounded retention; it does not make active-expert offload part of this profile. [Intel family feature table](https://www.intel.com/content/www/us/en/support/articles/000058341/processors/intel-xeon-processors.html)

UVA simplifies addressing; it does not pool eight HBM allocations, enable every peer path or establish coherent CPU/GPU memory. Query peer read/write and native-atomic support per pair (`cudaDevP2PAttrNativeAtomicSupported`) and host atomics separately (`hostNativeAtomicSupported`). This Intel/PCIe machine does not inherit POWER9's NVLink-to-CPU ATS/coherence. Ordinary mapped pinned memory also does not make GPU atomics atomic from the CPU's viewpoint. Managed-memory migration cannot supply uncharged active KV/expert capacity. [CUDA memory model](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-c-programming-guide/index.html)

## 5 Configuration space and exact duplication

TP splits matrices; PP splits target blocks; ETP splits one expert; EP assigns experts to groups; DPA assigns different requests to nonexpert/attention groups; true DP instantiates complete target replicas. SP/HC-SP partitions suitable activation rows, while CP/DCP/PCP refer to context/query/history work with implementation-specific copy semantics. The labels are insufficient: publish the physical owners and actual bytes.

The binding catalog is **Design Section 20’s six-case TP8×PP1/PP2×ETP8/4/2 matrix**. Each server/stage has 8 GPUs; world size is 8×PP. Expert ETP×EP=8 on each stage, with one distributed target and one selected TP8 companion group on the sole/final stage. ETP1 is excluded; required TP8×PP2 needs a second physical server. The following other-layout table and eight-GPU PP/copy calculations are historical analyses, not current launch modes. TP4×PP2 with ETP4/2 only remains an optional future study under Section 20’s qualification constraints.

| Eight-GPU configuration | Target/expert copies | History/state placement | Main advantage | Main cost / status |
|---|---|---|---|---|
| **TP8** | One target representation plus required existing rank-local arrays | Unique KV/indexer; sharded recurrent state | C1 latency, full eight-GPU batches, strong global ring | Small expert K80 and eight-rank latency; primary baseline |
| **TP4×PP2** | One target split by depth | Each layer's unique state on its stage, or separately served distributed owners | Quartet-local TP, K160, stage overlap | C1 serial traversal, endpoint imbalance, new state/queue/graph integration; optional future benchmark, not a current requirement |
| TP2×PP4 | One target | Three target QSA caches/stage at equal cuts; terminal MTP extra | Double-link TP pairs, K320 | Four-stage feedback and smaller useful groups; historical comparison, excluded from the backlog |
| TP1×PP8 | One target | Unequal 1/2 QSA caches per six-block stage; terminal MTP extra | K640, no target TP collective within a stage | Eight serial stages, strong terminal memory skew, many boundaries; historical comparison, excluded from the backlog |
| **Dense TP8 + ETP4×EP2** | One dense set, one expert set | Target histories unchanged | Expert K160 without duplicating dense/attention weights | Group load skew; expert/activation interface required; useful first folded-expert test |
| Dense TP8 + ETP2×EP4 | Same one-copy principle | Histories unchanged | K320; less fragmented expert kernels | Expert-owner skew and possible dispatch/combine cost; required degree, unqualified performance |
| TP8 + HC row ownership / attention-only context work | Same weights | Histories remain unique | Reduce repeated pointwise/HC work; distribute heavy QSA/indexer work | Layout transfers and exact score/state semantics; operator-level candidates |
| Attention TP4×DPA2 + ETP4×EP2 | One expert set; **two nonexpert sets** | Each request can have unique history in its home group | Smaller dense/attention groups with shared experts | Extra shardable nonexpert weights; coupled expert service; tiered duplication qualification |
| Attention TP2×DPA4 / TP1×DPA8 + shared experts | One expert set; **four/eight nonexpert sets** | Unique request state possible | Fewer attention TP participants | Larger weight tax and smaller local batches; tiered duplication qualification |
| **True DP2×TP4** | **Two complete targets**, usually two method plans | Different requests unique; shared prefixes may repeat | Independent scheduling/isolation, no global expert service | Duplicate target/method bytes, partitioned capacity pools, four GPUs for C1; big-ticket qualification |
| True DP4×TP2 / DP8×TP1 | Four/eight complete targets | Per-replica unique request state | More independent engines | Census may fail before histories; no feasibility claim without actual imports |
| Classical TP4×CP2 / TP2×CP4 / TP1×CP8 | Usually two/four/eight model representations | History may shard or repeat depending on implementation | Large single-prompt parallel work | Weight copies and GDN/PLE dependencies; not a free DCP setting |
| Conventional separate TP4 prefill + TP4 decode | **Two complete targets** | Full hybrid-state handoff, bounded overlap | Mixed-arrival latency isolation | Weight tax, cross-phase state transfer; big-ticket qualification |
| Unequal PP cuts / mixed TP degrees / interleaved virtual stages | One copy possible with exclusive ownership | Exact per-stage/per-operator state map | Correct measured stage imbalance | Resharding, extra crossings/queues and divisibility work; later experiment |
| Smaller single draft group / sole vision or head owner | One target and one selected companion/encoder | State concentrated or explicitly served remotely | Less existing companion replication or repeated work | Owner bottleneck, remote endpoint service and worst-rank headroom; conditional secondary option |
| Hot-expert/draft/vision/head replicas | Additional named tensor copies | Histories separately inventoried | Less owner contention or local lookup | Persistent copy cost; qualify named copies under the aggregate threshold |

Framework degrees do not always multiply: dense TP8 and ETP4×EP2 can use the **same eight ranks**. vLLM's EP spans nominal TP×DP when PCP=1; its current configuration includes TP×PCP×DP when PCP participates. SGLang distinguishes full replicas from attention-DP. A nominal MoE DP engine may still synchronize a shared expert bank. [vLLM EP](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/), [SGLang DP/DPA](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide), [Megatron parallel folding](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)

Let **E** be the sole routed-expert payload, **S** the shardable nonexpert payload and **R** the actual already-required rank-local parameters. Keep packing, companion, vision, embedding/head and graphs explicit. For ideal balanced placement:

| Placement | Per-rank shardable target bytes before separate R/packing | Extra box-wide copies versus one-copy TP8 |
|---|---|---|
| One-copy TP8 or folded expert ownership | `E/8 + S/8` | None |
| TP4×DPA2 with one expert bank | `E/8 + S/4` | `S` |
| TP2×DPA4 with one expert bank | `E/8 + S/2` | `3S` |
| TP1×DPA8 with one expert bank | `E/8 + S` | `7S` |
| True DP2×TP4 | `(E+S)/4` | `E+S`, before method/vision differences |

Already replicated R is not doubled again by this formula. Conversely, immutable CPU file mappings sharing physical page-cache pages do not make two GPU target representations one copy. Different requests' KV/state are not duplicates of each other. Bounded transient activation/query/send copies are permitted when charged. The one-copy control still has communication buffers and existing small replicated parameters; additional optional copies follow Design Section 1, with gross added bytes, actual released bytes and net whole-system/per-rank deltas reported separately.

Two-dimensional tensor grids, layer-weight all-gather/FSDP-style inference, and fine-grained operator services are possible mathematical designs. They require new packed layouts and frequent data movement; training optimizer/gradient sharding offers no saving when those tensors are absent. A dedicated draft or vision GPU is also not a free ninth device: it removes a GPU from the existing eight-rank target plan. These are lower priority than fixing measured bottlenecks in the layouts above.

## 6 Model and service contracts retained

Use the non-Swift SC117 IQ3_S target, the selected shared-Q8 MTP companion, and Inco DFlash2 as separate speculative profiles. Preserve multimodal image/video-to-text support, original tokenizer/positions, mixed packed weight values and the full numerical contract. Q8 and RotorQuant iso4 Q4 refer to the KV profiles already defined in the implementation design; changing TP does not change model-weight quantization.

| Geometry controlling this comparison | Governing profile |
|---|---|
| Target depth | 48 blocks: 36 GDN, 12 QSA at zero-based 3, 7,…, 47 |
| Hidden/residual | Width 2,560; four HC streams, 10,240 residual values/row |
| QSA | 24 Q heads, two KV heads, head dimension 256 |
| Indexer | Four query heads/one key head, dimension 128; completed groups of four, top 512 groups plus visible tail |
| Recurrence | 16 key/48 value heads, dimension 128; FP32 canonical state |
| Routed experts | 512, top 10, intermediate 640; actual mixed-format tensors must be inventoried |
| Context | Native 262,144 total rows; administrator-selected RoPE/YaRN extensions require separate positional, method, quality and resource qualification |

These are the design’s initial geometry contracts. The selected GGUF source tensor directories have since been inspected as summarized in the numerical companion; this does not establish a verified runtime import or allocation plan. Planner dimensions come from the actual manifest; another variant requires its own geometry and layout proof.

Eligible comparisons implement unique attention KV and completed pooled indexer history with bounded raw pending state. Comparing optimized TP4 against the historical wastefully replicated TP8 caches would assign a cache redesign benefit to the wrong architectural change. Raw-history RAM archive and host expert paging remain outside the selected baseline. The control adds no target, expert-set, endpoint or vision copies; separately qualified duplication arms must inventory and justify each added copy under Design Section 1.

Each resident sequence keeps its complete equal context promise. At 262,144 positions, prompt, expanded media, reasoning and generated reply share that limit. Idle retained history still occupies committed capacity. Preserve the device floor, serving workspace, snapshots and prepared graph bounds, and charge actual supported profile-replacement overlaps. The fixed-placement implementation needs no migration-only reserve; charge optional migration overlaps if that experiment is adopted, releasing only actual dedicated liability without weakening shared protections. The old 27.086 GiB serving budget and old 13.421 GiB/rank weights belong to a different qualified receipt; remeasure them for the selected target and each layout.

Optimize **committed output tokens per wall-clock second** at the qualified maximum active population, with prefill throughput, per-request latency and fairness reported alongside it. Also measure all lower active counts while the maximum resident population retains history. The historical 20-Q8/32-Q4 points and the required C24/Q8 objective are not certified capacity limits. Capacity and throughput are separate dimensions of the result.

## 7 Memory, cache uniqueness and stage balance

Historical comparison analysis: other target layouts in this section are outside the current supported matrix. TP4×PP2 with ETP4/2 only remains an optional future study; none of these measurements or calculations creates a current development obligation.

### Weight placement

Reuse Section 5's expert E and nonexpert S subtotals. Let W=E+S be all shardable target payload, R the actual required rank-local payload and P the packing/padding/derived-storage overhead. For an illustrative balanced placement:

```text
TP8 rank:                 W/8 + R8 + P8
Independent TP4 rank:     W/4 + R4 + P4
PP2 stage j, TP4 rank:    Wj/4 + Rj + Pj
```

Wj includes that stage's target endpoints exactly once; separately add actual method/vision placements and all execution resources. A single-owner or differently sharded endpoint uses its actual allocation, not Wj/4 by assumption. Do not double every TP8 bucket for TP4: already replicated arrays can remain the same size per rank while sharded arrays grow. PP2 can reduce block-local replicated residency because each GPU hosts only its stage's blocks.

The reference contains meaningful replicated HC mixing matrices, routers and indexer/PLE projections. As an illustration, the two low-rank matrices in each of 97 HC mixers would occupy about 1.184 GiB/rank if retained in FP16. **This is not the selected IQ3_S resident inventory.** The selected binary tensor formats, block padding, aliases, conversion products and actual executor placement determine that inventory. [Pinned HC implementation](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/hyperconnection.py)

The selected transformer GGUF file's approximately 51.282 GiB is an artifact size, not a measured GPU weight budget. The 26.822 GiB PLE shard is a host mapping. Neither whole-checkpoint size divided by eight nor the historical loader payload can certify new capacity.

Packed shapes matter. Under conventional matrix-sharded experts, the intermediate width 640 becomes 80 at TP8 and 160 at TP4. Source quantization blocks are not automatically divisible at either shard boundary. Preserve decoded values and account for partial blocks, shared scales, padding or value-preserving repacking; do not silently requantize to simplify a shard. First record whether the actual ProviderOne expert loader uses matrix sharding, expert placement, or a hybrid before applying those width assumptions. [Pinned GGML block definitions](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/ggml/src/ggml-common.h)

### Unique history formulas

One global KV head costs 544 bytes per token/cache in Q8, or 272 in iso4. Two heads plus one completed pooled indexer history therefore cost:

```text
Q8:  2 × 544 + 64 = 1,152 B/token/cache across the model
Q4:  2 × 272 + 64 =   608 B/token/cache across the model
```

With 12 target caches and one MTP cache:

| Layout/rank | Q8 B/token | Q4 B/token | Sequences charged to that rank |
|---|---:|---:|---|
| TP8 | 1,872 | 988 | All global sequences |
| Independent TP4 replica | 3,744 | 1,976 | Only that replica's sequences |
| PP2 first stage, six target caches | 1,728 | 912 | All global sequences |
| PP2 last stage, six target plus MTP | 2,016 | 1,064 | All global sequences |

For balanced replicas, half as many sessions times twice the local history cost gives the same mean history/rank as TP8. A second target copy does not create extra global VRAM.

| Full-context example | TP8 history/rank | Balanced independent TP4 history/rank | PP2 first/last stage history/rank |
|---|---:|---:|---:|
| 20 global Q8 sessions | 9.140625 GiB | 9.140625 GiB | 8.437500 / 9.843750 GiB |
| 32 global Q4 sessions | 7.718750 GiB | 7.718750 GiB | 7.125000 / 8.312500 GiB |

These are ideal balanced growing histories, not total memory or feasibility claims. Stage means are not proven worst-rank physical allocations. Charge guard rows, page rounding, raw rings, state, snapshots, execution and all fixed resources. PP2's last stage has 7.69% more history/rank than the TP8 MTP average; reduced replicated target weights can offset some of that, while terminal speculative weights and bounded lookup/feature buffers add to it.

Canonical FP32 recurrent matrix state is 108 MiB per sequence across all 36 recurrent layers. TP8 holds 13.5 MiB/rank; a TP4 replica holds 27 MiB per local sequence; a balanced 18-recurrent-layer TP4 pipeline stage again holds 13.5 MiB per global sequence. Versions, convolution tails and recovery are additional existing ledger items. There is no independent recurrent-state capacity multiplier from splitting replicas.

### Capacity gate

For every rank r, require:

```text
weights_r
+ unique histories_r(N, context, codec)
+ canonical and transactional states_r(N, method)
+ execution/media/graphs/communication/snapshot/transition obligations_r
<= validated usable budget_r
```

Solve for the largest N satisfying every rank of the single distributed target. Pipeline stages may have unequal weights, auxiliary work and headroom; keep the session context equal despite that placement. Do not reserve global-average bytes and hope the fullest GPU fits. Replica arithmetic shows its additional residency cost; eligibility to experiment does not establish feasibility or acceptance.

Each extra 1 GiB/rank consumed by fixed resources replaces approximately 2.188 global full-context Q8 MTP histories or 4.146 Q4 histories under the TP8 average rates. This sensitivity illustrates why weight duplication matters; it does not establish an integer capacity before the complete ledger is measured.

### C24/native-context comparison and deeper pipelines

For `a_i` target QSA caches on stage i, `m_i=1` on the MTP stage and zero elsewhere, stage TP degree p, total window L and N sessions:

```text
ideal stage-mean history/rank = N × L × (a_i + m_i) × B_cache / p
B_cache = 1,152 B/token for Q8, 608 for iso4
```

At L=262,144, one unique cache is 288 MiB/session Q8 or 152 MiB iso4 across its stage. The following assumes equal target block counts and **stage-local MTP**, before guards, raw tails, page rounding, state, execution, graphs, media or ownership skew:

| Layout | Target+MTP cache counts by stage | C24 Q8 GiB/rank, largest stage mean | C24 iso4 GiB/rank, largest stage mean |
|---|---|---:|---:|
| TP8×PP1 | 13 across eight ranks | 10.968750 | 5.7890625 |
| TP4×PP2 | 6 / 7 across four each | 11.812500 | 6.234375 |
| TP2×PP4 | 3 / 3 / 3 / 4 across two each | 13.500000 | 7.125000 |
| TP1×PP8 | 1 / 2 / 1 / 2 / 1 / 2 / 1 / 3 | 20.250000 | 10.687500 |

TP4×PP2 specifically needs **10.125 / 11.8125 GiB** first/last rank Q8, or **5.34375 / 6.234375 GiB** iso4 at C24. Its late-stage Q8 history is 0.84375 GiB/rank above TP8 before concentrated MTP/head weights. If the 2.584999-GiB MTP payload were wholly shardable, TP4 instead of TP8 adds another approximately 0.323125 GiB/late rank; actual replicated/packed allocations require an import receipt. A positive whole-box saving can therefore coexist with worse limiting-rank capacity.

The design's twelve-cache TP8 phase/XOR balance proof uses four ranks/global KV head. It does **not** automatically prove balance for six/seven-cache TP4 stages with two ranks/head. Define paired/striped KV placement and pooled-indexer ownership, including incomplete pages and MTP, then compute actual max-rank allocations. A global eight-rank history service can remove local-stage skew in principle, but adds cross-stage query/append/selection/attention services and scheduling contention. It is another measured placement, not a free equality assumption.

For contrast, conventional TP8 with one full KV head and one full pooled-indexer history on every rank costs **7,904 Q8 or 4,368 iso4 B/token/rank** for 13 caches. At C24/native context that is **46.3125 GiB Q8/rank for history alone**, exceeding a 32-GiB device. Unique ownership is necessary for the stated Q8 goal; DCP cannot divide the already-unique budget again. Its scope is ownership/compute distribution, not a second fourfold saving.

### A pipeline cut is a time-and-memory choice

For a cut after n blocks, first-stage QSA count is `floor(n/4)` and GDN count `n-floor(n/4)`. Include final-stage MTP separately. Geometry-only C24 history means:

| Target split | First GDN/QSA | Last GDN/QSA | Q8 GiB first/last rank | iso4 GiB first/last rank |
|---|---|---|---|---|
| 20/28 | 15/5 | 21/7 | 8.4375 / 13.5 | 4.453125 / 7.125 |
| 23/25 | 18/5 | 18/7 | 8.4375 / 13.5 | 4.453125 / 7.125 |
| 24/24 | 18/6 | 18/6 | 10.125 / 11.8125 | 5.34375 / 6.234375 |
| 25/23 | 19/6 | 17/6 | 10.125 / 11.8125 | 5.34375 / 6.234375 |
| 28/20 | 21/7 | 15/5 | 11.8125 / 10.125 | 6.234375 / 5.34375 |

23/25 balances recurrent counts but puts two more target QSA layers plus MTP late. 25/23 moves one GDN block away from the terminal method/head stage without increasing its cache count; 28/20 also moves one QSA cache early. Neither is automatically optimal: actual weights, prefill/decode costs, vocabulary work, PLE, vision, draft and headroom must fit at both stages. The total target history is unchanged. Stage-local MTP need not be the only possible placement, but a remote placement must pay its actual dataflow/progress cost.

For the separate DFlash profile, use **12 target caches**, not 13, plus its actual draft window. Unique target-only native histories are 3.375 GiB/session Q8 or 1.78125 GiB iso4 across the box. The reference draft window is 40 MiB/session at safely qualified FP16, 80 MiB at FP32. The full 108-MiB recurrent state remains additional in both cases; verification/recovery versions and convolution/PLE tails are not included in that number.

## 8 Throughput: compute, communication and reuse

Historical comparison analysis: other target layouts in this section are outside the current supported matrix. TP4×PP2 with ETP4/2 only remains an optional future study; none of these measurements or calculations creates a current development obligation.

### Why a pipeline supplies no automatic compute speedup

Let `T8(b,k)` be a full TP8 pass for b sequences and k verification rows/sequence. For perfectly shardable arithmetic, a TP4 half-model stage does twice the per-layer local work but half the layers: its service **S=T8**, not T8/2. C1 then traverses two such stages. A genuine gain must come from more efficient kernels, less exposed communication or repeated work, and must exceed boundary, endpoint and scheduling costs.

For M initially ready identical microbatches and balanced stages, ignoring additional overhead:

```text
TP8 wave time = M × T8
PP(P) wave time = (M + P − 1) × S
PP wins only if S/T8 < M/(M + P − 1)
```

At R32/depth three, C24 gives three eight-sequence microbatches: PP2 needs S/T8 < 0.75 for that finite wave. C32 gives four and requires <0.80. These are ideal fill/drain thresholds, not measured gains. Continuous scheduling can amortize fill/drain; it still needs a better bottleneck useful-microbatch service than TP8 to improve saturated throughput on the same hardware.

With stage services `s_i`, minimum no-queue group feedback cycle τ, b sequences/group, N active chains and measured mean g committed tokens/sequence/round:

```text
aggregate committed tokens/s ≤ g × min(N/τ, b/max_i(s_i))
```

Add independent link, embedding lookup, sampler/coordinator or draft service limits. Saturation roughly needs `N/b ≥ τ/s_max`, approximately P only for balanced stages with little feedback. At C24 that gives optimistic population-based group ceilings 12/6/3 for PP2/4/8, also intersected with `b ≤ floor(Rmax/k)`. Under R32/depth three, PP2 therefore permits at most min(12, 8)=8 sequences/group. These bounds do not prove the scheduler saturates stages. A whole-wave barrier reintroduces bubbles; continuous ready queues can avoid it while preserving per-sequence causality.

One ordinary autoregressive chain cannot advance its next unknown token while waiting for the final sample. A known long prompt can pipeline causal chunks; legitimate speculation can provide candidate rows, but does not guarantee accepted work or remove feedback/rollback. Compare complete rounds and committed output, not target-pass time alone.

For C1 single-chain latency, let F denote local compute time, K exposed intrastage communication, B PP boundary cost and J endpoint/speculation/host service. PP beats TP8 only when `K8−Kp > (Fp−F8)+B+(Jp−J8)`. Measure every term for identical useful work; the inequality supplies a break-even condition without inventing token/s.

A normalized scenario makes the break-even concrete. Assume balanced PP2, identical useful batches, unchanged acceptance and no additional boundary/endpoint overhead; q=S/T8 is supplied by a future measurement, not estimated here:

| Hypothetical q | One-group target-pass latency / TP8 | Saturated microbatch throughput / TP8 | Three-group finite-wave throughput / TP8 |
|---:|---:|---:|---:|
| 0.80 | 1.60× | 1.25× | 0.9375× |
| 0.60 | 1.20× | 1.6667× | 1.25× |
| 0.45 | 0.90× | 2.2222× | 1.6667× |

The entries are `2q`, `1/q`, and `3/(4q)` for the same b/k/history shape. One-group latency represents C1 only when b=1; q measured with eight sequences cannot predict C1. Even a sizable saturated gain can accompany worse group latency or a slower finite wave. Real overhead, stage imbalance, row limits and acceptance make this an optimistic sensitivity example, not a committed-token/s prediction.

### Communication is not determined by GPU count alone

A simple ring AllReduce transfers 2(p−1)/p times the message size per rank: 1.75 times at eight ranks versus 1.5 at four. TP8 volume is 16.7% higher than TP4; TP4 is 14.3% lower than TP8, not twofold. Ring phase count, latency, message size, channels, reduction work and available links still matter. Tree and custom algorithms have other tradeoffs. [NCCL collective semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

Begin with NCCL's discovered topology and automatic selection. Trace the actual graph before changing rank ordering or forcing algorithms. A hierarchy that funnels everything through one leader per quartet can waste the other cross links. If traces justify a hierarchical TP8 variant, distribute the crossing work and include its extra local operations. Select tactics by complete-step benefit, including communication's interference with GPU compute. A custom tactic is a bounded offline-qualified operation/payload/dtype/topology choice with startup identity/readiness checks. Keep communicator families and rail/rank maps fixed; request-count changes select prepared tactics without environment-variable mutation, communicator recreation or schedule search. [NCCL configuration guidance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

Keep the current CUDA 12.x/SM70-compatible stack pinned consistently across comparisons. Use native SM70 FP16 compute, FP32 accumulation/state where required, packed weight/KV kernels, cuBLAS/cuBLASLt and suitable CUTLASS paths. A V100 cannot gain native BF16/FP8/NVFP4 Tensor Core execution through a topology change. Build and qualify any alternative kernel; do not infer support from a modern framework's install instructions.

### Local kernels and repeated work

With conventional matrix sharding, TP4 has six Q heads instead of three per rank, 12 recurrent V heads instead of six, and expert width 160 instead of 80. These can improve tile utilization or reduce awkward packed tails. They also change work per rank and useful row grouping.

HC/router and other replicated operators may execute on four ranks per logical layer rather than eight. PP2 additionally places only part of their weights on each rank. Measure the actual ProviderOne operator plan: the planned canonical indexer already avoids redundant indexer execution, so that saving must not be credited to TP4 a second time. An optimized TP8 owner/reduction path may capture part of the same benefit.

### When smaller batches reread more weights

If TP8 executes one batch of B rows while two TP4 replicas each execute B/2, dense weight reads per GPU can approach twice the TP8 amount in a weight-bandwidth-bound regime. Similarly, splitting one TP8 batch into two smaller PP microbatches can make each stage reread its weights. Total arithmetic alone does not reveal this penalty.

For sparse experts, the penalty depends on routing. Under an illustrative independent uniform top-ten selection from 512 experts:

```text
U(B) = 512 × [1 − (1 − 10/512)^B]
two-batch expert-read ratio = 2 × U(B/2) / U(B)
```

| Rows in the hypothetical unsplit batch | Expected distinct experts | Split-batch expert-read ratio |
|---:|---:|---:|
| 8 | 74.74 | 1.039 |
| 32 | 239.64 | 1.156 |
| 64 | 367.11 | 1.306 |
| 80 | 406.33 | 1.375 |
| 128 | 471.00 | 1.559 |

This is a synthetic traffic model. Real hot experts, shared weights and correlated speculative rows change reuse; it is not a tokens/s forecast. Record expert unions and actual rows/expert.

### The R32 constraint changes the comparison

At MTP depth three, one sequence contributes an anchor plus three proposals: four target verification rows. An R32 microbatch therefore serves eight sequences. The resident capacity may be much larger.

For 32 active sequences, TP8 already requires four R32 microbatches. Two TP4 replicas can each run two R32 microbatches; PP2 can pipeline the same four R32 microbatches. **Neither alternative must halve an existing 32-row batch.** The preceding split-batch penalty cannot be applied automatically.

For ideal dense shardable weight size D, ignoring routed experts, replicated and auxiliary tensors, the total per-rank reads in one such sweep are:

| Layout at 32 active sequences | Ideal per-rank dense reads |
|---|---:|
| TP8, four R32 passes | 4 × D/8 = D/2 |
| Two TP4 replicas, two passes per replica | 2 × D/4 = D/2 |
| PP2, four passes through each half-model stage | 4 × (D/2)/4 = D/2 |

This isolates why PP2 is a credible high-concurrency contender even before a larger verification kernel exists. It does not predict time: stage fill/drain, speculation, unequal expert unions, replicated work, wider kernels and communication remain.

Twenty active sequences give TP8 batches such as 8+8+4, whereas balanced TP4 replicas give 8+2 each. Tail batches can change reuse and idle time. For PP2, use the actual stage queues and round sizes. Report every real microbatch, rather than derive performance from configured session count.

Those partitions describe a synchronized sweep, not an unavoidable steady-state penalty. Continuous scheduling can fill a microbatch with ready sequences from different speculative rounds, subject to causality and fairness. Ten sessions on a replica need not perpetually run an eight-session batch followed by a two-session tail. Measure actual row histograms, proposal readiness, queueing and per-session progress.

Run both a matched R32 comparison and independently optimized layouts. Qualify R64/R128 or other useful row envelopes during preparation/profile activation if measurements justify them; do not make R32 a permanent architectural ceiling or pretend larger shapes are already supported. Ordinary load changes vary used rows/quotas inside the selected prepared maximum envelope.

### Full context means substantial indexer work

QSA scores all visible completed pooled keys before selecting 512 groups. At 262,144 positions, 65,536 groups × 128 FP16 values is 16 MiB/cache. Twelve target caches contain 192 MiB of pooled scan input per sequence; including MTP gives 208 MiB. These are unique working-set bytes for one full scan of each cache, not unavoidable traffic per committed token. Tiling can reuse keys across causally valid query rows. [Pinned indexer](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/attention/qsa/qsa_indexer.py) and [scoring kernel](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/layers/attention/qsa/mqa.py)

Attention then uses approximately 2,048 selected rows plus the incomplete-group tail; this is not dense full-history attention. Q4 reduces selected KV traffic and storage requirements, but leaves the FP16 pooled scan unchanged. Profile pooled scan/top-k, selected-index expansion, KV gather/attention, recurrent state and speculation separately. More KV attention ranks alone do not remove a canonical indexer bottleneck.

### Native shape and boundary cross-checks

| Uniform TP / PP | Q heads/rank | GDN key/value heads per hosted layer/rank | Expert intermediate/rank | Target blocks/stage |
|---|---:|---|---:|---:|
| TP8 / PP1 | 3 | 2 / 6 | 80 | 48 |
| TP4 / PP2 | 6 | 4 / 12 | 160 | 24 |
| TP2 / PP4 | 12 | 8 / 24 | 320 | 12 |
| TP1 / PP8 | 24 | 16 / 48 | 640 | 6 |

K80 is not intrinsically invalid for Volta tensor math. A *particular* K64-tiled kernel might pad 80→128 (+60%) and 160→192 (+20%); another kernel can handle these tails differently. Original quantization-block boundaries still need value-preserving handling. Measure the actual packed route, not a universal padding penalty. Routed-expert matrix products alone are `2×3×2560×640×10 = 98,304,000 FLOPs/row/block`; peak TFLOP/s cannot price small-M decoding, unpacking or communication.

At FP16 a full HC boundary is **20 KiB/row**; R32 is 640 KiB, R96 1.875 MiB, R8192 160 MiB. PP2/4/8 have 1/3/7 target boundaries, respectively. If four ranks each send a replicated full tensor, physical crossing volume is four times the logical boundary; if disjoint slices are sent, receiver redistribution may still be necessary. At R96 an ideal 50-GB/s double-link transfer alone has a 39.3-µs serialization floor, not a complete handoff latency. Do not divide every transfer by the 150-GB/s cut unless a qualified striped transfer actually uses it.

A contracted hidden/embedding vector is only 5 KiB/FP16 row; it is not the 20-KiB HC residual. A simple ring all-reduce's 1.75/1.5 volume factors do not specify this fork's operation count, datatype or exposed latency. Trace actual HC, recurrent, expert, indexer, head and draft collectives. A dense-model assumption of two reductions/block cannot be multiplied into a claimed ProviderOne speedup.

If an R96 C24 batch is split into equal PP-saturating groups, the uniform-routing toy predicts ideal expert-read factors of approximately **1.441/1.776/1.985** for PP2/4/8 relative to the unsplit four-row/request batch. For ordinary one-row decode they are **1.118/1.184/1.219**. These are explanatory union counts before caching and correlations, not actual HBM traffic or rate predictions. They must not be applied to an R32 control that already executes separate microbatches.

For perspective, the >250 C1 objective means <4 ms per committed output token. If g=3, the complete average C1 round must be <12 ms. At C24, >600 means >25 tokens/s average per active request; an illustrative synchronized round emitting three/request must finish in <120 ms. These are target-derived timing budgets, not measured acceptance, deadlines for every individual request, or a parallel-layout forecast.

## 9 Implementing TP4×PP2 without weakening the model

Historical comparison analysis: other target layouts in this section are outside the current supported matrix. TP4×PP2 with ETP4/2 only remains an optional future study; none of these measurements or calculations creates a current development obligation.

### Stage ownership and balancing

Start with blocks 0–23 on quartet A and 24–47 on B: six QSA and 18 recurrent layers per stage. This is a convenient first measurement, not a final cut. Put input embedding and PLE at the first stage; final HC mixing, vocabulary head and sampling naturally belong to the last.

Balance **measured stage service time and worst-rank memory**, including speculation and media. MTP can make the last stage slower even when target block counts match. Nearby complete-block cuts may help; changing the cut can also change QSA-layer counts, cache ownership and feature transport. Choose the cut for the service profile rather than moving layers at every active-count threshold.

### Correct boundary state

The target has four HC streams, each width 2,560. Preserve all 10,240 values: 20 KiB/row at FP16 or 40 KiB at FP32. An averaged 2,560-wide state is not an equivalent pipeline boundary. Specify dtype, state representation, row/position identifiers and whether transfer buffers are sharded or replicated.

The source already uses pipeline proxy tensors and relays multimodal embeddings needed for MTP prefill. This is a useful correctness precedent, not a ready-made V100 path. [Pinned target pipeline implementation](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/models/qwen4_exp.py)

Compare matched-rank sends against a logical single-copy transfer followed by local distribution, or weighted striping over the 2:2:1:1 cross connections. Sending four redundant full states may waste bandwidth, but eliminating them can add latency; measure complete boundaries. Use supported NCCL/P2P ordering and graph-stable bounded buffers.

### Native MTP

For this optional future TP4×PP2 study, initially locate the MTP drafter on the final quartet. Its final HC inputs and target vocabulary head are local. The one-copy control keeps the **selected SC117 embedding** at its owning stage and provides owner-served lookup with bounded transfer of required vectors. An additional persistent embedding representation is a separately priced duplication experiment under Design Section 1, as in the required TP8×PP2 profile. Relay the actual multimodal row embeddings where required; token-only lookup is not a substitute for those values.

Budget the lookup requests, returned vectors and buffers in the complete speculative-round path, particularly across sequential MTP proposal steps. The owning stage must service lookups while target work or feedback waits are pending; avoid a cycle in which stage 0 waits for the draft while the draft waits for stage-0 embedding service. This extra latency can determine whether PP2 wins over TP8.

The pinned MTP source exposes this head-versus-embedding distinction and implements last-stage drafting of the next chain before earlier stages start verification. Its separate embedding placement is not the one-copy control: compare it as an explicitly charged copy arm against the owner-served endpoint path, preserving the exact flattened-HC norm, QSA policy and target-conditioned text/visual refresh. [MTP sharing](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/models/qwen4_exp_mtp.py) and [tail-stage speculative worker](https://github.com/sgl-project/sglang/blob/dba22e9b68dda90345ac17f5146e7ac812a09ed6/python/sglang/srt/speculative/eagle_worker_v2.py)

Each verification microbatch traverses both stages. MTP work on the last quartet competes with verification of another microbatch. Include full-prefix initialization, accepted-prefix refresh, proposal passes and recovery in timing and stage balance. MTP capability and its reserved histories stay resident when a controller chooses a shorter proposal length within the authorized method policy.

### Inco DFlash2

Treat DFlash as a separately qualified profile. A final-stage TP4 drafter is a candidate, not a consequence of selecting PP2. For the conditional FP16 weight representation, the existing reference-placement draft payload becomes 0.528204 GiB per TP4 rank versus 0.429022 at TP8: the replicated 354,163,200 bytes remain, while 851,968,000 shardable bytes are divided by four instead of eight.

If FP16 KV passes the design's range gate, its 2,048-position window becomes 10 MiB per global sequence on final-stage ranks, versus 5 MiB/rank when spread over TP8; FP32 KV doubles both. FP32 accumulation does not prevent an overflowing FP16 output cast: qualify actual feature/output/KV ranges, with no silent clamping. Add tentative rows, selector/scratch, borrowed target endpoints and graph obligations. Target history in a 24/24 split remains six caches per stage; only the final stage carries the draft window.

As with MTP, a final-stage draft uses the qualified owner-served target embedding lookup and bounded vector transfers; a persistent target embedding copy is a separate duplication experiment under Design Section 1. The final target head remains a real storage alias for the local draft. Include lookup scheduling and latency in DFlash qualification.

The target feature taps are completed blocks 5, 15, 25, 35, 45. For a 24/24 cut, stage 0 supplies the first two HC means: 10 KiB/retained row at FP16 in addition to the HC boundary. Forward/materialize only bounded features needed by the actual draft window and current work. Preserve all five taps, their order, absolute draft positions, separate target M-RoPE coordinates and speculative rollback. Do not construct a full-context feature archive.

The selected draft's source dimensions permit TP4, but existing TP1-only helpers and graph restrictions are not an SM70 implementation. A TP8 draft spanning both four-rank pipeline stages is a historical alternative outside the backlog; it would need its own ledger and timing. The required two-server profile instead keeps the complete TP8 draft on the final server. [Pinned DFlash model](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/models/dflash.py), [worker](https://github.com/sgl-project/sglang/blob/3c560089e1818a5217a73f2168273485a5350341/python/sglang/srt/speculative/dflash_worker_v2.py), and [author's model card](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)

### Continuous pipeline scheduling and recovery

Use stage-ready queues with per-sequence causal ordering and bounded in-flight groups. Initially overlap independent sequence groups, with at most one uncommitted speculative round per sequence. Do not speculate through an unresolved commit merely to keep stages busy.

Rebalance microbatch membership as requests arrive or finish, proposal readiness changes, or measured stage costs diverge. Tune prefill chunk sizes by measured stage duration so a long prefill does not monopolize one stage while decode waits. Avoid waiting to fill a batch at low activity. More microbatches can improve overlap but reduce useful row reuse; optimize both together.

The final stage's acceptance decision must commit the same prefix on every stage. Recurrent matrices, convolution/PLE tails, KV/indexer pages, draft state, RNG and output publication must follow one transaction. Cancellation retires remote and queued work before resource reuse. Bound feedback messages, queues and snapshots; keep deterministic collective ordering even for empty stages and changing proposal lengths.

The OSDI 2026 paper on pipeline serving supports adaptive prefill chunking and decode-group rebalancing as useful mechanisms. Its Qwen2.5 experiments use four A100 40GB GPUs with PCIe 4.0 host connections; those results cannot supply V100/NVLink speedups. [USENIX primary paper](https://www.usenix.org/system/files/osdi26-hwang.pdf)

### Multimodal and host placement

Retain one fixed vision encoder/projector or a qualified sharding as the control; additional replicas require Design Section 1’s duplication qualification. Charge its temporary buffers and worst-case work with all session histories resident; forward projected embeddings and original token/position metadata to the correct stage.

Keep the existing shared PLE file-backed mapping. Shared physical pages are not simultaneously NUMA-local to both sockets; measure access and route lookups/results appropriately. Neither pageable host mapping nor spare PCIe bandwidth is unbudgeted KV or expert capacity.

With a single draft group, the reference Inco placement `851,968,000/t + 354,163,200` bytes per participating rank gives:

| Draft TP t | Conditional FP16 weight GiB/rank | Total draft weight GiB/box | FP16 window MiB/session/rank |
|---:|---:|---:|---:|
| 8 | 0.429022 | 3.432178 | 5 |
| 4 | 0.528204 | 2.112818 | 10 |
| 2 | 0.726569 | 1.453137 | 20 |
| 1 | 1.123297 | 1.123297 | 40 |

Fewer draft ranks reduce existing repeated draft arrays but concentrate work/history; at C24, TP1 adds 840 MiB of FP16 draft KV to its owner versus TP8's per-rank window, before other buffers. It can lower whole-box weight bytes and still reduce target capacity. A full FP16 `[262144,12800]` DFlash feature archive would be 6.25 GiB/replica: retain only bounded chunks and required ending-window features. Smaller-rank draft/head/embedding services require their own progress, numerical and memory proof.

## 10 Expert folding, context work and replication alternatives

Historical comparison analysis: other target layouts in this section are outside the current supported matrix. TP4×PP2 with ETP4/2 only remains an optional future study; none of these measurements or calculations creates a current development obligation.

### One-copy expert folding within the required matrix

Keep dense/GDN/attention execution on TP8 while choosing expert ETP t independently:

| ETP × EP over eight ranks | Experts/group | Expert K/rank | Ideal expert bytes/rank | Possible co-located top-ten group load / mean |
|---|---:|---:|---|---:|
| 8 × 1 | 512 | 80 | E/8 | 1× |
| 4 × 2 | 256 | 160 | E/8 | 2× |
| 2 × 4 | 128 | 320 | E/8 | 4× |

The inspected SC117 source has 144 routed tensors: 48 layers × gate/up/down, each with 512 experts of one shape and encoding. Expert source bytes are equal within each layer/projection; mixed formats across layers/projections do not establish per-expert byte heterogeneity inside a layer. Equal-count per-layer placement is the source-byte-balanced control. Qualify transformed allocations, packed-block tails and actual service separately; a byte-weighted expert optimizer is unnecessary without runtime evidence of unequal footprints. With **complete, identical FFN input rows and authoritative routes already present on all ranks**, owners can filter their experts locally, form weighted partial outputs, and use a qualified output sum. There need not be a new input-dispatch all-to-all, or separate mandatory ETP-then-EP reductions. Derive the actual bias/shared-expert/gate/cast semantics before combining sums. If HC/SP makes rows exclusive instead, gather or dispatch them; do not claim free replicated inputs and row-exclusive storage simultaneously.

For uniformly selected ten distinct experts, expected owner-group fanout is `e×[1−choose(512−512/e,10)/choose(512,10)]`: approximately 1.9982/3.7813 groups at EP2/4. Most EP2/EP4 rows already touch nearly every group. A universally hot expert needs R rows on its owner; if all ten coactive experts sit in one group, its load can be e times the mean. One hot expert alone is only one tenth of route assignments and does not establish that worst case. Measure coactivation, per-expert rows, byte costs, critical-owner service and scratch under legal worst routing. Never drop routes, change gates or truncate overflow assignments.

Use ETP8×EP1 as the control and qualify ETP4×EP2 and ETP2×EP4 under the required matrix; measure whether wider kernels outweigh skew and dataflow cost. ETP1×EP8 is excluded, with no implementation or test obligation. Keep the selected ETP and qualified stage-local placement fixed during ordinary serving. Device-side actual-row compaction, grouping and prepared kernel selection remain required efficient execution. A local work-centric schedule or coactivation-informed static map needs a measured service opportunity. ETP8 has EP1 and no competing owner group to rebalance; ETP4/2 need sustained owner-critical service skew before any live weight reshuffling merits investigation. Hot-expert migration/EPLB is a separate optional structural experiment with transfer/repacking, staging, planning, mapping epochs, graph/readers and reversal charged; it creates no initial implementation duty. [Parallel folding](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md), [DeepSpeed MoE inference](https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/)

Ordinary GPU collectives can implement EP on Volta. Current [DeepEP](https://github.com/deepseek-ai/DeepEP) is not an SM70 dependency; the experimental old-source port in Section 11 is a distinct negative benchmark. [HAP](https://arxiv.org/html/2508.19373v1) supports module-specific placement as an idea, but its V100 PCIe/Mixtral/2048-input/64-output experiments and optional CPU INT4 weight backup do not transfer to this resident IQ3_S/NVLink profile. Training-only Tutel/Lancet overlap results likewise supply mechanisms, not this target's inference rates.

### Context parallelism must preserve QSA, GDN and PLE

The required fixed unique attention-KV layout supplies a one-copy context axis within TP8's four-rank KV-head groups. Begin with four-way striping per quartet and fixed canonical pooled-indexer owners; validate exact page/slot balance for every stage, including PP2's six target QSA caches and final-stage MTP. Four-way striping is the first control, not a measured winner. Fixed degree-one/two or other legal layouts are optional comparisons only when a bottleneck justifies them. Live degree changes among 1/2/4 and their migration/union-map machinery are separately benchmark-gated and do not block initial release. TP4 stages conventionally have two ranks/head, TP2 one, TP1 both heads on the same rank. Bytes and selected-work balance differ: a balanced page layout can still concentrate nearly all selected positions on one rank. Measure query/append/selection/partial-merge costs at real rows/history lengths.

Fixed storage ownership still permits changing local CTA/query decomposition or selected-work lists without moving retained history. A bounded selected-KV gather to fewer compute ranks is an optional execution tactic that must pay its per-step reads, transfers and staging. Adopt live history movement only after it beats the best qualified fixed layout with observation, decision, quiescence, transfer/interference, metadata/graph activation and any reverse move included over realistic residence time. A sudden C1-to-maximum jump must make safe progress immediately from the valid fixed map. Fixed placement retains exact partial-softmax merges, page descriptors, generations, prefix/restore ownership and reader retirement; those obligations do not depend on implementing migration.

Classical CP copies a smaller TP group while splitting token work; it often repeats weights. Current vLLM PCP-only configuration explicitly repeats KV in its own implementation and does not increase KV shard count merely by increasing PCP. That is not a theorem that all custom query-row CP must repeat histories. DCP reuses participating TP ranks and removes conventional KV repetition; it is not another capacity multiplier over ProviderOne's unique-history baseline. Some source-project DCP paths still repeat indexer keys, recurrence or MTP KV. [vLLM CP](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/), [configuration source](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py)

A narrower experiment distributes QSA query rows or the indexer scan while keeping one weight/history representation. A distributed indexer starts as a separately qualified fixed-placement research profile; it requires no live repartitioner. The baseline canonical indexer remains fixed, including during any separately adopted KV-migration experiment. For history-sharded exact selection, compute each group's **complete four-query-head score**, keep local top-512 with a common total tie order, then globally select 512 from their union. Splitting query heads and merging independent top-k lists is incorrect: sum head contributions before selection, and reduce a split dot product before ReLU. Preserve pooling origin, causal frontiers and up to three raw tail positions. Partial attention merges use exact max/denominator/numerator or consistent LSE states, never normalized-output averages.

Ulysses-style sequence/head all-to-all is another prefill dataflow to compare. The 24 target Q heads divide by 2/4/8, but two KV heads do not head-shard at degree 4/8 without explicit GQA sharing, repetition or combined context sharding. The four-head indexer and GDN/PLE state require independent correctness proofs. Count any repeated KV/weights and full-history gathers. The [DeepSpeed training tutorial](https://www.deepspeed.ai/tutorials/ds-sequence/) is an algorithm reference, not a ready SM70 QSA/GDN serving backend.

Full-model CP is harder: the 36 GDN blocks and PLE convolutions are not token-independent. Every remote chunk needs its correct incoming FP32 recurrent state and tails, or a separately derived and numerically qualified scan. One complete GDN layer's state is 3 MiB/session globally; all 36 total 108 MiB before versions. Arbitrary zigzag/interleaved token assignment to ordinary recurrent decode is invalid. Preserve head order, decay/beta/casts, nonzero initial state, PLE token/hash resets, nine-row tails and speculative recovery. First parallelize independent prompts and optimize native GDN chunk prefill. Dense/H100 [CP papers](https://arxiv.org/html/2411.01783v1) do not establish a hybrid-QSA/GDN V100 speedup.

### HC sequence parallelism and owner services

HC row ownership can reduce repeated row-local mixing/norm/projection work and that specific replicated activation buffer. It cannot split four HC streams into independent sequences or replace them with their mean. MTP needs full flattened terminal HC; only DFlash uses its five specified means. Include the gather/reduce-scatter interface to the next producer/consumer and keep recurrence order intact. At R1 there is only one useful row; padding and extra transfers can cost more than the work saved. Current [vLLM HC-SP scope](https://docs.vllm.ai/en/latest/api/vllm/config/parallel/) is TP>1, PP1 and explicitly does not change MoE placement; its existence is not SM70 qualification.

Common PLE lookups, routing/indexer projection and vocabulary services may also avoid redundant work without weight shadows. The selected embedding and output head are distinct tensors; companion aliasing does not tie them together. Per-operator ownership is valuable only after charging vector transfers, progress, queue service, temporary buffers and actual worst-rank memory.

### What duplication can buy, and its exact price

Independent TP4 replicas can isolate batches and tail latency and avoid global expert coordination. A low-load request uses four target GPUs; an idle second replica does not accelerate it. With balanced distinct requests, halving the sessions per replica and doubling unique history/session/rank leaves the same **mean history/rank** as TP8. Extra target/method weights and indivisible pools are the tax. Prefix routing may repeat common prefixes or require transfer; CPU file sharing does not erase GPU copies.

DPA with one expert bank pays only repeated shardable nonexpert S rather than all E+S, but remains coupled at expert steps and can need dummy/progress work for idle request groups. It is not independent full-model DP. Hot-expert copies can split a bottleneck, costing `sum((copies_e−1)×weight_bytes_e)`; routing each original assignment to one equal copy must preserve every gate and accumulation contract. These are deliberate memory-versus-service alternatives, not the present one-copy baseline.

Conventional prefill/decode disaggregation can protect TTFT/TPOT under sustained arrivals. Two TP4 phase groups require two full targets on this box, unlike TP4×PP2. A handoff must include KV, pooled indexer/tails/origin, GDN/conv/PLE state, chosen method's history/features, media/position mapping, RNG/constraints/stops and a consistent commit/owner generation, with sender/receiver overlap budgeted. A generic KV connector does not establish that hybrid transaction. Shared sole experts reduce some copy tax but also couple both phases at the same expert service. [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)

Retain one vision encoder/projector or a qualified sharding of it. Video decoding/CPU preparation can overlap independently where resource-safe. Replicated encoders or draft/head shadows are separate duplication choices. Accepted immutable copies belong to the prepared effective profile; creating/releasing copies as request count changes is not required normal adaptation. Reserving a GPU just for them changes target rank geometry and must improve the complete mixed workload and capacity ledger; low mean occupancy is insufficient.

## 11 Published V100 measurements, with comparison boundaries

### 11.1 Flash-Next: 1Cat's strongest matched concurrent decode

The [repaired no-MTP study](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_qwen38_concurrency40.md) uses **four V100 SXM2 32-GiB, TP4, different NVFP4 target weights, FP16 KV, 8,192 actual input/256 forced output**, fixed KV bytes and complete engine intervals. It is not SC117/Q8/MTP or full-history decoding.

| Actual C | Control ms/step | Batch-on ms/step | Batch-on aggregate tokens/s | Per-stream tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 10.198848 | 10.272228 | 97.350 | 97.350 |
| 2 | 17.316287 | 13.401774 | 149.234 | 74.617 |
| 4 | 17.715166 | 15.040512 | 265.948 | 66.487 |
| 8 | 21.398201 | 18.847988 | 424.448 | 53.056 |
| 16 | 28.662908 | 26.464495 | 604.584 | 37.786 |

Paired fixed outputs match across 62 sequences. C1 regresses 0.71%; extra packed copies cost **1,055.625 MiB/rank**. Separate repeated synthetic 8K prefill is about **6.83K input tokens/s**, not natural-chat prefill. This proves batching/kernel potential and a real memory cost, not a default ProviderOne promotion or C24/262K result.

### 11.2 Eight-card TP4×PP2, closest topology class

[Issue #852](https://github.com/1CatAI/1Cat-vLLM/issues/852) reports eight V100 SXM2 32-GiB, DGX-1 topology, Flash-Next NVFP4, TP4×PP2, build `1.5.1.dev919+g2c8b9baa7`, configured 262144/max 16. KV codec and measured speculation are unspecified.

| Measurement | Async enabled | Async disabled |
|---|---|---|
| C1 window-median decode | 31 tokens/s | 48 tokens/s |
| C16 aggregate, three runs | 336.5 / 413.6 / 410.1 | 405.3 / 418.8 / 408.1 |
| 250K×3 wall time | 253.9 s | 254.4 s |

There is **no TP8 control**: that Flash-Next build restricts TP to 1/2/4; its 126-GiB weight plan did not fit TP4×PP1. Profiler evidence localizes the difference between steps, with unequal profiler distortion; C8 was bistable. This is a scheduler A/B within PP2, not proof of PP2 superiority or an instruction to disable async universally. The three-stream wall time is not an isolated prefill rate.

### 11.3 Closest SC117 derivative and required method costs

| Primary source / contract | Published result | Transfer limit |
|---|---|---|
| [SC117 deployment](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/flashnext_abliterated_local_20261014.md): four 16-GiB SXM2, TP4, different SC117 revision, 80/20 GPU/host experts, host E4M3 target KV, MTP4 | Actual ~34.6K input: **5,088.8 input/first-choice-latency tokens/s**; 32,774-input warm requests: **154.9 pure decode tokens/s** | Different cache/residency and original-block Q8_1 activation/`dp4a` routes; Q8_0 MTP. Does not qualify ProviderOne FP16-activation/iso4 math. [Execution contract](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/flashnext_16gb_host_kv.md) |
| [PR #1211 closed/unmerged campaign](https://github.com/1CatAI/1Cat-vLLM/blob/99efc51b0eb2356e4a24a211ae4519f108510e5b/docs/design/flashnext_mtp_batch_16gb_20261015.md): similar four-16-GiB host-history profile, actual 32,774 input, MTP4 | C1/2/4/6/8 **fenced, aligned-TP diagnostic** whole verifier **15.814/23.167/32.618/39.967/48.297 ms**; draft separately **3.466/3.975/4.814/5.857/6.695 ms** | Separate from ordinary-worker throughput below. Complete batch latency, not per-request latency divided by C; C2/4/8 goals unmet |
| Same PR #1211, ordinary common decode windows | C1 **140.55**, C2 **153.27/157.33**, C4 **261.20**, C6 **337.48**, C8 **322.04/310.21 aggregate tokens/s** | Different capacity/row controls from the preceding deployment; scaling is not monotone |
| [Merged PR #1242](https://github.com/1CatAI/1Cat-vLLM/pull/1242), four 16-GiB/host-KV SC117 campaign | C1 full-round p50 at actual 4K/32K/128K/250K: **18.1/18.3/19.5/20.3 ms**; natural prefill 42K **5.2K**, 159K **4.7K input tokens/s** | Target pin/timer and tokens/round not completely specified in body; not a correction of the unmerged PR #1211 receipt |
| Same PR #1242, concurrent at actual 4K/32K | C1 **19.0/19.6–20.2 ms**; C2 **26.2–26.8/27.9–30.9**; C4 **36.3–37.5/38.7–47.0**; C8 at 4K **62.9 ms** | Row/hot-cache envelopes differ; cannot infer committed tokens/s without acceptance |
| [NVFP4 MTP batch defaults](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_flash_next_batch_defaults.md), four 32-GiB TP4 | C1 actual 31,744: **79.832→92.199**; C4 four 4K inputs: **225.400→227.778 aggregate tokens/s** | Estimated, policy-dependent extra packed-copy budget **1,261.5625 MiB/rank**; C4 change within variation; not four 32K histories |

Historical **~6,708 input tokens/s SC117 MTP** is disqualified: expert IDs 256–511 were skipped/unwritten in the old control. That defect is repaired in the inspected pin. It does **not** invalidate the separate no-MTP NVFP4 6.83K synthetic prefill measurement. Keep these distinct. Earlier [PR #415](https://github.com/1CatAI/1Cat-vLLM/pull/415) no-MTP 80.732 and [PR #389](https://github.com/1CatAI/1Cat-vLLM/pull/389) short-code MTP 138.26 were narrower historical points, not the latest matched frontier.

### 11.4 DCP and EP: useful negative controls

[1Cat target-QSA DCP2](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/qwen4exp_qsa_dcp2.md) shards main target KV but repeats indexer keys/rings, GDN/PLE state and MTP QSA inside DCP membership. Selected same-code controls:

| KV / configured allowance | Reported capacity DCP1→DCP2 | C1 decode DCP1→DCP2 |
|---|---:|---:|
| E4M3 / 32K | 396336→493244 | 77.9→75.8 tokens/s |
| E4M3 / 256K | 775096→1178337 | Same reported C1 control; actual decode input length unspecified |
| FP16 / 32K | 284341→404706 | 72.9→70.3 |
| FP16 / 256K | 433401→718015 | Same input-length caveat |

Short repeated 2K TTFT is 16–18% slower; layout/metadata repairs reduced a larger decode penalty. This is a capacity/latency tradeoff from removing some repetition, not a further saving over fully unique ProviderOne histories.

The [true-EP4 branch](https://github.com/1CatAI/1Cat-vLLM/commit/16b4f03395a0dc58f5ebeb50600e6e5b115764b9) uses four SXM2, TP2×DPA2/EP4, NVFP4/no-MTP, actual 8K/256/C16. An experimental old-source SM70 DeepEP port gives **408.609 tokens/s /39.1572 ms**, versus **422.211 /37.8958 ms** for same-source AG/RS EP4: **−3.22%**, rejected, quality unresolved. It is not a matched regular-TP4 result or the one-copy dense-TP8 folding proposed here.

### 11.5 ReVolta: four-card method and long-prefill evidence

The [current performance report](https://github.com/Leonccaa/ReVolta/blob/f427a87220daede8f64517f1f3ad33e183acdc69/docs/performance.md) and [DFlash guide](https://github.com/Leonccaa/ReVolta/blob/f427a87220daede8f64517f1f3ad33e183acdc69/docs/dflash2.md) use AWQ, E4M3 target KV, TP4+DCP2, host PLE, optional retention tiers and a 512K allowance. They are not SC117/iso4/TP8.

| Hardware/workload | MTP4 | Inco DFlash2 | Metric |
|---|---:|---:|---|
| Four PCIe V100 32-GiB, v0.1.0 C1/2/4/8 | 152.5 /167.2 /262.2 /383.6 | Separate later width policies | Total output/wall including TTFT |
| Four SXM2 32-GiB, v0.1.2 community NVLink, 200W; C1 code | 180.4 | 234.3 | Single-prompt decode |
| Same, C1 reasoning / Chinese | 184.9 /117.1 | 228.2 /114.3 | Category points, not a workload average |
| Same, C2/4/8 | 231.8 /337.6 /516.3 | 260.1 /367.8 /513.0 | Concurrent total output/wall |

Cold NVLink MTP input/TTFT at actual 8K/32K/131K/262K/460K is **4,680/4,505/4,037/3,618/3,007 tokens/s**. Its [PCIe reference](https://github.com/Leonccaa/ReVolta/blob/f427a87220daede8f64517f1f3ad33e183acdc69/docs/getting-started.md) is a different power/host setup, not an isolated NVLink A/B. Real-agent DFlash/MTP speed ratio is **0.97×, 95% interval 0.90–1.01**; MTP remains the default.

State replay illustrates a separate capacity/speed choice: [v0.1.4](https://github.com/Leonccaa/ReVolta/releases/tag/v0.1.4) graph replay averages 30.49 versus 30.05 ms/step with replay off in 30 agent turns at 200W, **greedy drafting with sampled target**, with bounded output checks. Recommended probabilistic drafting instead reports about +0.8 ms/step in six sessions at 250W. Optional replay changes capacity. CPU/disk restore is a cache-hit path, not cold prefill compute. New prefix/replay changes are mechanisms, not layout evidence.

The recommended DFlash width policy verifies seven proposals at C1–3, three at C4 and four at C5+, while still drafting all seven. It requires synchronous scheduling; async conflicts and PP is unsupported. Those active-count policies change useful row/acceptance curves and must be named in layout comparisons. TP8/GGUF remain untested. [Limits](https://github.com/Leonccaa/ReVolta/blob/f427a87220daede8f64517f1f3ad33e183acdc69/docs/known-limits.md)

### 11.6 Strata: actual concurrency versus headline PP speed

Strata's layer split is **PP, not TP**. Its expert-cache helper GPUs are also not TP. Most fork measurements use two 16-GiB V100 PCIe cards, asymmetric host links, Ryzen 5, 175 W and host expert service, unlike this resident eight-SXM2 target. [Current multi-GPU semantics](https://github.com/jmnargi/Strata-V100/blob/353003069ba1296d7c9706e9d37791da8c71dd58/docs/MULTI_GPU.md)

[PR #46, closed/unmerged](https://github.com/jmnargi/Strata-V100/blob/2a63d29848dea55738b663412532d605b409a866/benchmarks/v100-iq3_s-concurrency-2026-10-07.md) configured four slots in every arm, INT8 KV, 32K allowance, split16. All 80 requests returned 128 tokens. Two, three and four simultaneous decoding slots were observed; one client uses the solo path. Columns identify submitted clients, **not constant decode C throughout the wave**. Effective batch-MTP settings were not published; solo speculation does not establish concurrent speculation. These are aggregate output/whole-wave-wall rates, including prefill/admission/polling:

| Actual input band | 1 client | 2 clients | 3 clients | 4 clients |
|---|---:|---:|---:|---:|
| 912–916 | 22.08 | 19.52 | 21.76 | 22.07 |
| 4085–4089 | 15.87 | 13.73 | 14.10 | 15.39 |
| 16275–16280 | 6.68 | 6.31 | 6.48 | 6.49 |
| 31974–31979 | 4.23 | 3.80 | 3.73 | 3.83 |

This demonstrates four active slots, not improved decode throughput. A separate [Q2_0 one/two-slot pair](https://github.com/jmnargi/Strata-V100/blob/353003069ba1296d7c9706e9d37791da8c71dd58/benchmarks/v100-q2_0-upstream-v0139-2026-10-04.md) shortened second-request TTFT but made two/four-client wave completion about 23% slower. Its concurrent batch path offered zero drafts, versus roughly 75–78 accepted of 100–103 offered in solo MTP. This includes a method change, not an isolated batching-speed comparison; queue latency and useful throughput can move in opposite directions.

The closest [all-resident two 32-GiB SM70 PP2 flag study](https://github.com/Niko1221/Strata/blob/93b172590266f1f375a6b055f2951b29ae81e1cc/bench/results/2026-10-14-community-v100-pg500-pipeline-windows/README.md) uses IQ3_XXS, INT8 KV, 24/24, short 37/36/50-token English/Japanese/code inputs; all output texts match across its ABBA studies:

| Power | English engine decode tokens/s, off→pw2 | Japanese, off→pw2 | Code, off→pw2 |
|---|---|---|---|
| 170W | 100.60→106.10 | 111.85→120.65 | 122.85→144.80 |
| 250W | 100.75→106.10 | 112.10→120.75 | 123.30→145.25 |

Prompt performance was essentially neutral; moving MTP from last to first stage reduced decode 2.9–6.7%. Strata's pw2 **optimistically launches another guessed window before the current verdict**, with an extra verifier and recurrent snapshots/rollback. The initial independent-group controller in Section 9 excludes that cross-round speculation. These gains therefore do not directly validate that controller or rank TP8 versus PP2. No full-history C24 result.

### 11.7 Closest four-SXM2 Strata long-prefill report

[Upstream report PR #2525](https://github.com/Niko1221/Strata/blob/93d121eb77d36e360c834e6f7bf67b7720d3622e/bench/results/2026-10-14-community-4x-tesla-v100-iq3-s-262k-code/README.md): four **16-GiB SXM2** behind one shared Gen3 x8 host link, different IQ3_S target, 98% experts resident, INT8 KV, PP auto, MTP4. These measured **upstream 0.1.44.1** prefills followed a discarded warm-up sweep: zero prefix reuse, warm expert/page caches, not cold process starts.

| Actual input | Engine prompt tokens/s | TTFT s |
|---:|---:|---:|
| 3,971 | 881.6 | 4.53 |
| 31,307 | 1,985.3 | 15.81 |
| 63,539 | 2,345.3 | 27.16 |
| 125,011 | 2,490.7 | 50.29 |
| 185,531 | 2,560.5 | 72.59 |
| 260,331 | 2,536.9 | 102.78 |

Reported 94.6 decode at 260,331 inputs has output lengths 98/164/93; it is not a reliable fixed-work comparison. No batching sweep. Versus same-session 0.1.43, long prefill regressed 0.3–0.9%, against about ±0.3% same-binary control variation. P2P probes reached ~38.8 GB/s/pair versus ~1.56 GB/s/card simultaneous host transfer. The measured helper was unset; peer-device+layer-split was refused, and the helper constructor forced host staging. No eligible layer-split configuration exercised that P2P path; direct NVLink byte counters were unavailable. Experimental control vectors/vocabulary also prevent target equivalence.

Additional useful Strata controls, without confusing their gains with this layout decision:

| Primary evidence | Measurement / mechanism | Applicability |
|---|---|---|
| [Same exact-input single/dual prefill study](https://github.com/jmnargi/Strata-V100/blob/353003069ba1296d7c9706e9d37791da8c71dd58/benchmarks/v100-iq3_s-prefill-decode-2026-10-23.md) | Candidate actual 32K prompt times 17.74855 s single, 15.54370 s dual | Different placement/memory envelopes; not isolated GPU-count scaling |
| [Current prefill changes, PR #80](https://github.com/jmnargi/Strata-V100/pull/80) | Exact 32K prompt 14.3409→14.0664→14.3638 s off/on/off | Smaller loans 3.24→2.73 GiB/card change expert placement; decode nonregression not established |
| [V100 batch-group control, upstream PR #3411](https://github.com/Niko1221/Strata/blob/58a56194ac1f8de37ace6ac386ba13bf5b587df5/bench/results/2026-10-20-v100-sm70-defaults/extended/README.md) | Two 32-GiB PCIe V100, regular IQ3_S, 32K/two slots; two 32000-input+512-output clients: two groups 43.55/43.58 s versus one group 59.06/58.80 s | All sentinels pass but only 2/4 complete outputs match; long-input whole-request observation, not decode-only fixed-work proof |
| [Fork PP2 service study](https://github.com/jmnargi/Strata-V100/blob/353003069ba1296d7c9706e9d37791da8c71dd58/benchmarks/v100-upstream-pr-evaluation-2026-10-15.md) | Median short-turn decode ratios+11.31%; exact-input prefill through 32K neutral | Different executable provenance and unequal continuations; not isolated flag/fixed-work proof |
| [DFlash2 PR #68, closed/unmerged](https://github.com/jmnargi/Strata-V100/blob/baede60aa1d8ccf5a7b8ed83822da4f1229f92ef/benchmarks/v100-iq3_s-dflash2-2026-10-20.md) | Sampled GSM8K/MATH/HumanEval request-throughput 52.06/58.98/55.36 MTP versus 51.31/55.78/51.53 DFlash | Higher acceptance but 1.45–6.92% slower; FP32 draft outputs/KV after overflow; no concurrent DFlash, batch/pw2 rejected |
| [Current helper repair PR #84](https://github.com/jmnargi/Strata-V100/pull/84) | Old “no gain” helper had an empty eligibility window; actual execution exposed H-buffer overrun | Main now refuses unsafe helper; no demonstrated repaired-helper acceleration. Prefill host/PLE counters now request deltas, formerly lifetime sums |
| [Verify expert cap PR #82](https://github.com/jmnargi/Strata-V100/pull/82) / [PR #83](https://github.com/jmnargi/Strata-V100/pull/83) | +17.7%/+~11% reported screens relax target expert verification | Changes accepted distribution; excluded as exact-target acceleration |

Historical four 16-GiB Gen3 C1/2/4/8 aggregate 120/113/205/360 in Strata's batching docs is **explicitly not V100**. Other R9700/3080/A4000 positive and negative pipeline/group results are scheduling precedents, not this machine's rates. Source host-expert/active-KV offload gains do not satisfy ProviderOne's selected residency contract.

### 11.8 Other-model V100 records: capability, not Flash-Next predictions

| Model / primary record | Actual contract / measurement | Result | Interpretation |
|---|---|---|---|
| Dense27B [v1.5.1](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.1) | Four 32-GiB SXM2, TP4, NVFP4/E4M3, DFlash q7; C1 actual 4K/32K/128K | **229.34/201.18/171.43 pure tokens/s**; C4 four 32K forced-output common decode 445.45 | Different model/precision; no C24/full-history transfer |
| Dense27B [natural code fixture](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_dflash2_acceptance_20260909.md) | C1; 135 actual input tokens, one MBPP fixture, 5 warmups+5 runs, q7, E4M3 KV, FP32 logits, natural EOS | **306.098 pure tokens/s** | Bounded fixture, not a broad average or matched 4K release workload |
| Dense27B [lookup-assisted draft](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_dflash2_lookup_augmented.md) | Repeated-context continuation, optional lookup contract | **316.27 tokens/s** | Separate speculative method/fixture |
| Dense27B [matched 32K/256 source pair](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_dflash2_long_decode_followup_20260927.md) | C1/2/4/8 common decode | **161.621/252.465/436.004/709.714** final | Fixed fixture matches; separate public-data C2 acceptance guard 70.417→53.940% failed, retained in report |
| Dense27B [later C4/6/8](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_dflash2_m48_scaling_20260927.md) | 32K/256, ordinary service, three waves, matched outputs/counters | **447.421/666.955/730.373** | No C16/C24 claim; larger campaign targets unmet |
| Dense27B [FP8 prefill](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_dflash2_prefill_closure.md) | Exact 8,000 input, target only | **5,170.96 pure /5,121.44 request-wall input tokens/s** | 5,500 was a goal; separate 256K 2,971.51 screen rejected zero outputs |
| Dense27B [NVFP4 DFlash prefill](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_dflash2_nvfp4_prefill_promotion.md) | Actual 32K/64K, E5M2 target KV | **4,069.25/3,566.94 pure input tokens/s** | Different codec and model |
| DeepSeek V4 [TP8](https://github.com/1CatAI/1Cat-vLLM/pull/181) / [later TP4×PP2](https://github.com/1CatAI/1Cat-vLLM/pull/344) | Eight V100, different campaigns/kernels, short/no-spec | **65.1 versus 73.613–73.646 tokens/s** | Not a matched layout A/B; do not derive a PP speedup |
| GLM5.3 [TP8](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_glm53_flash_nvfp4.md) / [strict PP2](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_v100_migration_control.md) | NVFP4/E4M3, DFlash; distinct audits | TP8 ~30.8 ms/round, 180.008 weighted pure; PP2 ~43.4–44.2 ms/round, 130.6–132.1 mean steady | Different average/receipt definitions; model-specific latency/capacity tradeoff, not Flash-Next control |

GLM's same packed-cache placement reports PP2 capacity 333796 versus TP8 309806 tokens, while its TP8 round was 31.093 ms. This illustrates why a slower layout can fit more history. Its actual 262136-input+8-output PP2 boundary took 1160.21 s prefill and 1.825 decode tokens/s; configured context alone concealed that cost.

Excluded peaks remain visible: [dense27B issue #1269](https://github.com/1CatAI/1Cat-vLLM/issues/1269) claims 436–451 C1 at 1K but reports stochastic corruption/two valid repeats; [907.292 C8](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_dflash2_batch_latency_20260926.md) is a rejected narrow eight-prompt screen; [784.79 C8](https://github.com/1CatAI/1Cat-vLLM/blob/4d57473db67eb77a13018d179e4d95169e82044f/docs/design/sm70_dflash2_batch_attention_gdn.md) uses estimated re-tokenized counts/one sample. None establishes the requested exact-model winner.

### 11.9 Collectives and NUMA: improvements that can change the ranking

[Open 1Cat PR #1245](https://github.com/1CatAI/1Cat-vLLM/pull/1245), head `205ac6f550a5870b7ac06dc9286ea1a16069559e`, measures complete all-reduce calls on eight 16-GiB SXM2 with mixed NVLink/PCIe peers. Selected median µs, extension/NCCL:

| Payload | FP16 sums | FP32 sums/partials |
|---|---|---|
| 5 KiB | 6.341 /21.819 | 9.083 /29.678 |
| 40 KiB | 12.567 /36.338 | 29.193 /38.985 |
| 128 KiB | 26.126 /54.318 | 70.807 /95.611 |
| 512 KiB | 90.373 /165.070 | 273.733 /251.049 |

Small-message gains are promising, but FP32 loses at the largest listed point. Pure eight-GPU NVLink mesh is untested; claimed model+6–22% is a projection, not measured model output. Preserve ProviderOne's dtype/reduction contract and actual message histogram before importing a threshold.

Both columns retain **FP16 inputs/outputs**. The optional FP32 mode changes sums/partials; its NCCL comparison includes conversion to/from FP32. These are not direct timings for FP32 activation payloads. [ReVolta PR #49](https://github.com/Leonccaa/ReVolta/pull/49), merged 29 October, also removed a fixed 27-token collective cutoff after another NVLink layout crossed near 60: calls of 30–56 tokens regressed 1–4 µs. Whole-model gains were small or inconsistent, reinforcing topology-and-shape-specific qualification.

[NUMA issue #678](https://github.com/1CatAI/1Cat-vLLM/issues/678) on a Dell C4140/two Gold 6148/four V100s shows locality 72.6→98.2%, decode 63.66→64.88, prefill 2820→2847; whole-container binding did not help. Different machine, small measured gain. PP2 issue #852's larger scheduler/CPU findings likewise require local attribution, not generic affinity/thread-count presets.

[Issue #1285](https://github.com/1CatAI/1Cat-vLLM/issues/1285) corrects its own “four 262K” headline: 336,934 KV tokens is 1.29×262K, **four mixed-length users**, not four full histories; its eight clients had four active slots. Its faster real-agent comparison to Strata also used 92% prefix hits and different output volume. Such workload reports are not isolated inference or layout A/B tests.

## 12 Runtime adaptation without restart

The administrator selects context/positions, KV codec, target/draft profile, speculative method and latency/resource policy. During preparation or qualified profile activation, the engine computes the full-window resident ceiling and one maximum execution envelope for a **qualified fixed structural and numerical layout**. Normal serving uses a small scheduler and prepared plans: issue legal ready work and vary used rows/quotas without changing retained-state placement, maximum allocations or profile identity.

| Adjustment | Policy |
|---|---|
| Scheduled sequences, target row budget, prefill chunk and legal media work | Between batches inside the prepared maximum envelope; issue ready work without waiting to fill a batch |
| Prepared MTP depth or legal proposal prefix from the prepared DFlash block | Choose within method-specific correctness, position and resource bounds before fresh draws; preserve conditioning and charge warm-zero/full-block costs. A genuine different DFlash block is a separate prepared implementation/qualification choice |
| Kernel tactic and existing graph bucket | Select a compatible prepared plan by actual work/geometry; no search, compilation or activity-triggered graph replacement |
| Communication tactic | NCCL's normal selection or a prepared measured operation/payload/dtype/topology choice; fixed communicators/rail map, without environment-variable reload or runtime schedule search |
| PP microbatch count/membership | Between safe scheduling boundaries; preserve per-sequence causal/transaction state |
| CPU/media task and byte quotas | Queue-driven work in persistent bounded NUMA-bound pools; feedback-based quotas only for measured contention. Pool size, affinity and library thread settings remain fixed |
| KV ownership / attention degree | Fixed qualified unique placement; four-way per-head-quartet striping is the first control. Fixed alternatives need evidence; live 1/2/4 changes, migration and union maps are optional benchmark-gated work |
| Canonical indexer ownership | Fixed for the baseline; optional distributed selection starts with its own qualified fixed placement |
| Graph set, maximum execution/arena envelope or resident ceiling increase | Prepared profile or explicit qualified administrator activation with complete bounds, lifetime and capacity proof; ordinary activity selects existing plans/quotas |
| Context, positions, codecs, target/draft numerics, method, copies or structural owners | Fixed during ordinary requests; use Design Section 19's compatible prepared in-process transition or affected-state drain |
| Supported ETP or PP/server assignment | Select at initialization; later use a qualified in-process complete-state transition or drain. TP remains 8, EP=8/ETP and no other target layout is exposed |

Initialize only the selected profile's required communication/descriptor families, maximum buffers and bounded prepared graph/kernel set before admission. Record effective method, fixed owner map, row limits, qualified plans and the worst-rank memory receipt. Several activity counts may share one best plan; add distinct tactics/buckets only for measured complete-serving gains. Test every integer population through capacity, with detailed points at 1, 2, 4, 8, 12, 16, 20, 24, 28, 32 and any additional measured breakpoints.

Select one of the six supported profiles at initialization and keep its ETP, PP/server assignment, layer/weight owners, history placement and maximum envelope fixed during ordinary serving. Choose ready batches/chunks, legal widths, work quotas and existing graph/kernel plans. C1-to-maximum activity must execute immediately inside the prepared bounds, using smaller legal groups where needed; migration, recapture and allocation-envelope growth cannot be prerequisites for progress. Live history redistribution is optional only after net benefit over the best fixed layout pays all transition and reversal costs while preserving capacity and service gates. No TP4 or automatic server expansion/contraction path is required.

Prepared administrator changes retain the no-restart contract without requiring arbitrary occupied-service migration. A supported structural transition needs persistent-worker repartition/reload, state compatibility/mapping, affected graph/tuning retirement and atomic activation. An affected-state drain followed by bounded empty-resident unload/reload in the same workers can avoid holding two full models at once. It may pause admission without restarting a container or CUDA context. Zero active requests is insufficient when GPU- or tier-retained histories exist. A setting without a supported transition remains fixed and exposes no live control.

For an occupied service, changing PP/server assignment also moves target/MTP/DFlash weights, recurrent states, caches, feature ownership, snapshots and aliases. Prove intermediate executable states, temporary headroom, transfer/repacking cost, rollback and response continuity. Pre-created NCCL groups do not move missing weights. Until that work is qualified, keep structural placement fixed for the configured maximum profile, as required by the no-restart contract.

## 13 Decisive qualification and selection

### A Inventory and communication

Capture the actual server, without changing its settings:

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total --format=csv
nvidia-smi topo -m
nvidia-smi topo -mp
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
nvidia-smi topo -p2p n
nvidia-smi topo -p2p a
nvidia-smi nvlink -s
nvidia-smi nvlink -p
nvidia-smi nvlink -e
lspci -tv
numactl --hardware
```

Use flags supported by the installed driver, recording unsupported fields. Save CUDA peer-access tests, CPU/DIMM topology, ECC, clocks, power, temperatures, link errors and the exact native SM70 stack. [NVIDIA system-management documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

Measure single/double links, concurrent traffic, TP8 and the required ETP4/2 subgroups on each participating server. For PP2, measure actual GPU Direct RDMA small-message, multi-rail and contended pipeline traffic across the two servers. Include real operator sizes, from small decode reductions through prefill messages, graph replay and concurrent compute. Record algorithm bandwidth, bus bandwidth, latency and GPU occupancy as distinct metrics. Use [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests) plus operator-level instrumentation; do not compare one idle quartet against a fully loaded eight-GPU run.

### B Import and memory

Create exact rank-by-rank tensor and auxiliary inventories for TP8×PP1/PP2 with each ETP8/4/2, one eight-GPU stage/server and the matching MoE MTP degree. No other target layout is required. Keep the one-copy control; accept optional weight/vision or other copies only through Design Section 1’s measured tradeoff decision. Label and charge each arm’s gross/net aggregate and per-rank steady/peak increments, cumulative threshold tier and capacity loss. Big-ticket items require compelling repeatable gains and feasible lower-copy comparisons. Warm actual kernels, the prepared graph set, owner-served embeddings, speculation and maximum media. Test full future-context commitments, fixed-placement balance and worst-rank peaks, including recovery and supported prepared profile transitions; charge migration-only overlap if that optional experiment is adopted. Retain the design's 4-GiB device-floor and 2-GiB serving-workspace policies with disjoint accounting. Eliminate infeasible layouts before throughput work. Do not raise capacity by changing codec, disabling required speculation or reducing reserves.

### C Matched performance

Compare ETP degrees at fixed PP/hardware with the same concurrency, context, model values, KV representation, speculative method, sampling policy and input set. Report PP1-versus-PP2 separately as one-versus-two-server scaling; the multi-node measurements require actual hardware. Include R32 with identical row semantics. Then optimize each layout's supported microbatch/row sizes and nearby pipeline cut where traces justify it.

Separately compare every layout at its own qualified maximum concurrency. Publish the capacity-throughput-latency frontier. A layout can be fastest at the same N while admitting fewer full-context sessions; those are different results. If neither candidate dominates both capacity and throughput, retain the administrator's explicit objective rather than hide the tradeoff in a single score.

### D Workload coverage

Exercise every feasible requested threshold and neighboring breakpoints. Include short, intermediate and genuinely near-full histories; one active request with all slots resident; all slots decoding; arrivals/completions; mixed prefill/decode; representative images/video; weak and strong speculative acceptance; and realistic expert skew. Session fairness and tail latency remain gates.

A full-window timing case can use 261,120 valid expanded prompt positions plus up to 1,024 generated positions. A 262,144-position prompt plus a reply would exceed the window. Retain representative routing and task behavior instead of creating a repetitive synthetic prompt that makes the model artificially easy.

### E Result and selection gate

Record committed output tokens/s, per-session rates, TTFT, text prefill tokens/s, visual rows/s and encoder time, tail inter-token latency, queue delay, acceptance, complete speculative-round time, collective time on the critical path, stage idle time, expert group sizes, indexer timings, memory high-water marks and remaining headroom.

Use repeated sustained runs and confidence intervals; keep clocks, temperature and workload conditions comparable. Validate outputs, long-context quality and exact selected semantics, including rollback and multimodal positions. A changed reduction order needs the numerical qualification already required by the project; a speedup cannot excuse missing computation.

Promote an alternative only when its complete profile meets capacity, correctness, media, latency and no-restart requirements and improves the chosen useful-throughput objective by more than run variation. Stop expanding experiments once a material uncertainty is resolved; traces should determine the next optimization.

### F Required experiment set

| Round | Required work | Evidence |
|---|---|---|
| 1 | Actual eight-GPU topology/native SM70 stack and SC117 source/runtime census | Correct transport, values, state and resource control |
| 2 | TP8×PP1 with ETP8/4/2 | All three degrees pass numerical, cache, graph, capacity and complete-serving gates |
| 3 | TP8×PP2 with ETP8/4/2 on two physical servers | One eight-GPU stage/server; real RDMA/rail measurements, HC/features/endpoints, transactions, stage balance and worst-rank ledger |

PP2 is pending until hardware and actual execution evidence exist. The same ETP selection governs target and MoE MTP; no independent target×method degree sweep or other-layout test program is required. Compare ETP at fixed PP/hardware and label PP1-versus-PP2 as scaling.

For each primary method separately, publish this matrix at **C1, 2, 4, 8, 16, 24**, plus every supported integer for release qualification and the actual maximum beyond 24 where feasible:

| Case | History/cache conditions | Required reporting |
|---|---|---|
| Single and multiple cold prefills | Actual 8K, 32K, 128K and near-full valid expanded prompts; independent noncached prefixes | Isolated prefill compute, input/TTFT, aggregate union-wall processed tokens, media rows/encoder time |
| Pure concurrent decode | Same occupied histories/input set, enough output for sustained all-live measurement | Actual committed counts, aggregate and per-stream rates, p50/p99 inter-token latency, method/acceptance and complete round |
| C1 with idle residents | Other admitted sessions retain full promised histories | Low-load speed under real capacity pressure, not an empty cache pool |
| Mixed arrivals/prefill/decode/media | Same arrival trace and admitted resource envelope | TTFT, decode goodput/ITL, FIFO delay/fairness, prefill progress and worst-rank peaks |
| Prefix resume / RAM / optional disk restore | Same compatibility key and declared hit extent | Restored/cached/computed tokens and separate transfer/recompute timing; never cold-prefill credit |

Use interleaved/reversed A/B order, independent repetitions and sustained thermals. Keep the detailed per-count result visible. The design supplies an ordinal priority, not numerical scoring weights: declare any score weights and noise tolerance before selecting. Both equal-C performance and each layout's own maximum qualified capacity are necessary; a layout can win one and lose the other.

Decode counting must exclude rejected drafts, tentative output, padded rows and requests no longer active in the common window. Whole verifier latency is the maximum aligned critical-rank interval for a batch, not batch latency divided by C. Include full-prefix MTP initialization/refresh, draft/head/embedding service, accepted-state recovery, graphs, output and coordinator gaps in the complete round.

An alternative wins only when it passes all mandatory numerical/media/lifecycle/resource/fairness gates and improves useful performance beyond variation without a prohibited lower-count decode regression. If the high-C frontier and C1 nonregression cannot both be met by one fixed structural plan, report that limitation. Do not promise an automatic active-count TP8↔PP2 switch to hide it.

## 14 Implementation commitment

Implement the required TP8×PP1/PP2 with ETP8/4/2 through one parameterized serving path; ETP1 is excluded. PP1 has one physical server; PP2 has two, each an eight-GPU TP/ETP/EP domain. Initial PP2 target layers split 24/24, with the selected TP8 MTP/DFlash group on the final server and exact owner-served endpoints/features. The selected ETP applies to MoE MTP; DFlash remains dense. Hold the structural profile fixed during ordinary adaptation.

Require a qualified fixed unique KV layout, fixed canonical indexer owners and one prepared maximum envelope. Four-way attention-KV striping per head quartet is the first control, without a winner claim. Preserve exact merges, slot generations, prefix/restore ownership and reader retirement. Initial release requires cheap ready-work scheduling and prepared width/kernel/graph selection. Live degree conversion/migration and mixed-holder/union maps remain optional research rather than release gates; promote them only after complete-serving gains over the best fixed implementation pay measured steady and transition costs under the existing capacity/nonregression gates. Graph-set replacement and maximum-envelope resizing belong to prepared profile or qualified administrator activation; ordinary activity selects existing plans and quotas.

The three PP1 cases can be qualified and released on the current server once all required SC117 feature gates pass; unavailable PP2 hardware does not certify or block that one-server release. Record the three TP8×PP2 cases as pending until the second server, V100-compatible GPU Direct RDMA path and complete multi-node state/resource behavior are measured. Preserve the existing correctness, full-context capacity, fairness, prefill/decode and no-restart gates. TP4×PP2 is retained only as low-priority optional future work with ETP4/2 in a conventional four-rank stage; ETP8 is excluded from this study. No DPA/deeper-pipeline/replica discussion expands current scope. No extra method-degree sweep or ETP1 work is required. Selective tensor-copy optimizations remain subject to the ≥1-GB aggregate deployment review without altering target group/routing topology.

## 15 Reproducibility, sources and remaining unknowns

The arithmetic in Sections 3, 5, 7–10 is derived from the stated graph/model contracts, including exhaustive equal bisections, physical link directions, ideal history means, stage cuts, R32 grouping and uniform-routing examples. It does not certify selected-file GPU allocations, actual ownership balance, sustained interconnect service, capacity or output speed.

Source links are pinned where practical. Section 2 of the governing design binds the owned `python-0.6` implementation snapshot and its source/status evidence; live official framework documentation establishes terminology or requirements, not installed-code identity. ProviderOne integration must bind the real checkout/dirty patch, source/derived weights, native libraries, flags, loaded objects and effective owner/kernel/graph plans. Repository merge status is not a numerical or endpoint acceptance result.

Remaining binding unknowns are the installed CPU/DIMM/NUMA topology and link health; exact imported mixed-format target/MTP/vision allocations and aliases; executed expert/HC/indexer/head layout; measured packed-kernel/collective service curves; actual acceptance and complete round/host timing; maximum full-context resident capacity; and matched throughput at the required active counts. These are concrete measurements needed to choose the winner, not reasons to abandon the best-supported architecture or weaken its contracts.

Exact source identities retained from the governing implementation design:

| Artifact / reference | Revision |
|---|---|
| SC117 non-Swift target GGUF | `90a3350cd800ec1fe0d1660a38414ad9e985782f` |
| Qwen base config/source | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Shared-Q8 MTP companion | `766911a6b7369840a91dbcd95f9f997acaab6cd6` |
| Inco DFlash2 | `a22a1744e85c17364dcc54a6a420af7135d4c7f6` |
| Target/MTP SGLang reference | `dba22e9b68dda90345ac17f5146e7ac812a09ed6` |
| DFlash SGLang reference | `3c560089e1818a5217a73f2168273485a5350341` |
| GGUF conversion/format reference | `b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9` |

Additional primary method references: [NCCL point-to-point ordering/progress](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html), [NVIDIA nvbandwidth](https://github.com/NVIDIA/nvbandwidth), [Sarathi-Serve](https://arxiv.org/html/2403.02310v3), [GPipe](https://arxiv.org/html/1811.06965v5), [SGLang prefill CP](https://docs.sglang.io/docs/advanced_features/prefill_cp), [SGLang DCP](https://docs.sglang.io/docs/advanced_features/dcp), [Megatron CP assumptions](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html). Training, dense-model or modern-GPU evidence supplies principles only. Every benchmark-specific link in Section 11 includes its model/hardware/timer/copy limitations beside the claim it supports.
