# Shared CUDA kernel contracts

The shared GDN and lane-matmul extensions enqueue work on PyTorch's current
CUDA stream. Their native wrappers check tensor metadata, device identity,
kernel dimension ranges, and known write/read overlap before launch. These
checks do not synchronize device data or change floating-point operations.

## SM70 FP16 execution policy

The `cuda.sm70` provider stores activations as FP16, with FP32 reductions and
recurrent state. Native RMSNorm, text split-half RoPE, mixed-GGUF decode and GDN
chain/replay enqueue on PyTorch's current stream. Native wrappers validate
layout, dtype, shape, same-device residency, launch ranges and applicable
read/write overlap; empty row batches do not launch kernels. GDN retains its
existing State/GDNScratch ownership, including a separate reusable prefill
packet sized to admitted prompt rows. Convolution, normalization, decay, beta,
state transition and output gating are part of the family execution path.

GGUF tensors retain their encoding identity. Dense projections are decoded in
bounded chunks at load; expert banks retain original blocks and decode bounded
routed tiles. Partial column shards duplicate complete intersecting blocks and
crop the logical view without requantization. Fixed dense row tiles preserve a
row's reduction policy across prompt/decode grouping. Expert slots retain the
shared family's lower-precision prefill and FP32 decode storage policy.
Checkpoint gamma, GDN head tiling and PLE layer/hash conventions are restored
by the loader, without changing other checkpoint readers.

The same family interfaces dispatch FP16 HC, PLE, routing, indexed causal
attention and KV operations. Attention masks both scores and V operands so
stale nonfinite rows outside the causal/QSA selection cannot contaminate a
retained prefix. SM70 admits raw FP16 and the existing INT8 H32 KV descriptor,
including mixed sides; modern BF16 providers reject raw FP16. Storage identity
and execution policy are part of state reuse and rank agreement. INT8 changes
cache arithmetic; its scales and codes retain the existing format identity.

Preparation/compilation happens before graph capture. Captured tensors retain
their owners through every replay under the stream rules below. Component and
tiny-family graph replay checks do not enable full-model capture: the SM70
serving profile uses an explicit eager fallback, and concurrent TP still needs
qualified pointer routing for captured rounds. TP uses the ordinary NCCL owner,
per-peer acknowledgements and shared scheduler; numeric cache failures reach
rank consensus before publishing tokens or entering later sampling collectives.
See [the V100 qualification](v100-flashnext-validation.md) for actual installed
and full-model results; BF16 bit identity is not claimed.

## Streams and ownership

Every input producer must complete before its consumer reads that input. Same
stream enqueue order supplies that dependency; a consumer on another stream
must wait for the producer's event or stream. `record_stream` protects allocation
lifetime and does not establish producer completion. This follows PyTorch's
[stream semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-streams)
and [allocation lifetime contract](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.record_stream.html).

GDN `pointers` and `replay_table` return list-compatible values retaining their
tensor owners. Passing those values directly to `gdn.to_device` transfers the
owner references to the device pointer table. `tree` and `replay` record these
owners on the consuming stream, so dropping the table after enqueue cannot
recycle pointees while that stream still uses them. Pointer tables are immutable
between construction and consumer completion. Copying their integer values,
replacing pointers, taking a table view, or invoking the native extension
directly requires the caller to retain the actual pointees and register their
consumer stream until completion. A device address alone cannot prove allocation
size, dtype, device residency, initialization, or lifetime.

A captured CUDA graph may consume these addresses again on later replays. Keep
its pointer tables and pointees alive and unchanged for the graph's entire
usable lifetime, and wait for outstanding replays before releasing them.
Recording one capture or launch stream does not retain ownership for arbitrary
future graph replays.

QMM's automatically allocated split-K scratch is created and used on the same
current stream; the caching allocator's stream ordering protects its release
after enqueue. Caller-supplied inputs, scratch and outputs follow the same
producer, consumer and lifetime rules. Do not reuse or mutate scratch until its
producer and any reduction consumer complete. Programmatic dependent launch in
the grouped kernel retains its device dependency wait before reading inputs;
it does not replace cross-stream waits.

## Block-drafter confidence and readback

`cuda.speculative.Chains` owns one stream-major launch packet. Its borrowed
kernel-source owners remain live until the synchronous host readback completes;
a failed readback does not clear them. Ownership does not establish cross-stream
readiness: the caller must provide a dependency if the producer and readback use
different CUDA streams. Newly decoded cumulative confidences must
all be finite before the host packet is published, including rows outside a
caller-specific node cap. Penalized confidences are checked after computation
from the current request history, before publication or cutoff.

The owning drafter validates a fixed optional stop threshold during
configuration. `Chains.finish(stop=...)` then applies the existing node cap and
stops before the first score strictly above that threshold, retaining one
proposal when the cap permits it. Equality stays in the chain. A cached packet
does not revalidate its unchanged confidence data or configuration; history
identity remains part of the launch contract. This changes neither target math
nor the rule for committing proposals.

## GDN schedules and pointer tables

`schedule` accepts integer parent indices for one nonempty tree: row zero has
parent -1, every other parent precedes its child, and no other root is allowed.
`plan_host` and `plan` produce one nonempty range per stream, signed-int32 row
offsets, a complete visiting order and initialized state slots. A tree requiring
slots has at most 1024 rows per stream and at most 32 live slots. Chains use zero
slots, read rows in their natural order, and need no shared schedule storage.

A caller providing its own `Plan`, device row/count arrays, or raw native table
must establish these same invariants before upload and preserve them until
consumer completion:

- Starts begin at zero, end at W, increase by nonempty stream ranges and never
  exceed W. `max_rows` covers each range.
- Each non-chain plan visits every row of its stream exactly once. Its first
  source is -1; later sources are -2 or initialized slots within `slots`; each
  destination is -1 or a slot within `slots`. The visiting order follows parents.
- Every replay or pending count is in `[0, P]`, where P is the corresponding
  path width; every referenced row is within the pointee row tensors' W range.
  Zero counts need no row read and preserve the starting state.
- Tree tables have one fp32 `(Hv, Dv, 128)` state pointer per stream. Replay
  tables contain k, v, g and beta per layer, followed by states in stream/layer
  order. Layers share row shapes, head geometry and key dtype.
- Keys have Hk positive 128-element heads, values have Hv positive Dv-element
  heads, and Hv is divisible by Hk. Queries/keys are bf16 or fp32; values are
  bf16, gates and states fp32. All tensors and pointees use the same CUDA device.
  Key four-element loads and state float4 loads require their checked alignment.
- Read-only states may be shared. Pending and in-place replay states must be
  exclusive, mutually disjoint and disjoint from all rows and schedule inputs.
  Final states must be disjoint from inputs and other final states. Use
  `replay_table(..., in_place=True)` to validate its intended write contract.

The native ABI intentionally does not copy device schedules back to the host:
those arrays are trusted products of validated host builders. Arbitrary or
concurrently mutated device pointer/row contents violate that ABI contract.
Qwen's accepted-path builder validates integer, increasing, in-window rows and
stream cardinality before constructing its packed replay rows/counts. Supplied
commit indices must be unchanged results of that builder for those same paths.

`gdn.chain` keeps its input state read-only and writes a disjoint final state.
An empty chain returns empty outputs and leaves the final state untouched. Its
prefill recurrence and verify recurrence retain their distinct rounding orders;
each recurrence remains invariant to its supported chunking or tree grouping.

## Affine accumulation and prompt attention

CUDA's four-bit lane epilogue adds each group's dot product and affine bias correction as
`acc = fma_rn(group_sum, bias, fma_rn(dot, scale, acc))` in FP32. Groups and split-K partials
retain their specified order. The stored-layout Triton lane and row-parallel partial use the same explicit
FMA chain as the packed native kernels. Implicit `acc + dot * scale + group_sum * bias` is insufficient:
a compiler may round each product separately. This correction can change the older Triton/untiled results;
it preserves the selected native CUDA order and promises no cross-backend Metal bit identity.

Prompt attention uses absolute 64-key partitions and the native online-softmax denominator recurrence
`l = fma_rn(l, alpha, tile_sum)`. The Triton path explicitly follows that recurrence. Its Q/K/V/output
head-row offsets widen to int64 before multiplying row, head and dimension strides. Positions still obey
the signed-int kernel range, including the last padded query block; the adapter rejects out-of-range
metadata before narrowing. Numerical comparisons retain bit identity and independent cancellation checks.

Queries and caches are contiguous BF16 on one CUDA device, with positive head counts and HK dividing H.
Native dimensions 128/256 require query alignment of 4/16 bytes, cache alignment of 16 bytes and output
alignment of 4 bytes. Triton's dimension-64 path keeps its naturally aligned views. Explicit output is
disjoint from queries and each cache's filled `[0, p0 + W)` prefix; an unfilled tail may hold output.
The head grid must fit the selected device. Empty query windows return empty output without a launch.

Flash Next's precomputed n-gram group sums must agree with the consuming group-sum kernel for the stored
BF16 rows. Test fixtures use independent exact arithmetic to distinguish a faulty reduction comparator
from a production-kernel defect; a generic PyTorch reduction tree does not define those bits by itself.

GLM's BF16 expert mode rounds each weighted FP32 slot product before adding it in slot order. The
shared weighted expert path chooses that native specialization before launch. FP32 expert mode retains
its existing FMA chain; a device loop does not switch between the two policies.

## QMM buffers and aliases

Row-strided x, scales and biases remain supported where the native kernel
declares contiguous rows. Overlap checks compare the bytes actually accessed,
so separate regions of one allocation and outputs in an unused row gap remain
valid. Sharing read-only weight or scale tensors is allowed.

For a direct or clustered matmul, output writes must be disjoint from every
input read in that launch. A reduced split K sums in a cluster on sm_90 and
newer unless the host's launch choice declines clusters; then, as on older GPUs
and past 8 slices, it is staged. With non-cluster split-K staging, the producer
writes only the used scratch prefix `(SK, M, n)`; that prefix must be disjoint
from its inputs. A subsequent reduction reads scratch and writes an output
disjoint from scratch. Because input reads have already completed, this staged
output may alias an input. `reduce=False` with SK greater than one does not
write `out`; SK equal to one writes `out` and does not use `part`. Buffers
unused by the chosen path impose no alias restriction. Scratch capacity beyond
its used prefix is not accessed.

Grouped QMM outputs must be mutually disjoint and disjoint from every group's
actual input reads. Prefill and FP8 prefill outputs must be disjoint from their
actual input reads. The
grouped tile selector is 0 through 12; the signed `early` flag uses a negative
value for device policy, zero to disable and a positive value to enable it.
These metadata checks preserve K-slice addition order and existing bf16/fp32
rounding.

Outputs and split-K scratch may occupy packed padding that the selected kernel
never reads. The direct lane kernel loads columns through n rounded up to its
selected block's columns, 64 or 128.
Grouped launches load through n rounded up to their selected column tile, 64 or
128. BF16 prefill loads through its selected 64, 128 or 256-column tile, stopping
at the allocation's 128-column padded width. FP8 prefill uses 128-column tiles
and reads that entire padded width. Scale/bias projections preserve each row's
leading stride; packed weight reads form a contiguous prefix. Normal disjoint
operands use a constant-time logical-range check; projected checks run only
when those logical ranges overlap. An output crossing even one loaded cell is
rejected. Merely discarding a loaded padding column's computed result does not
make concurrent writes to that column safe.

Staged 16-byte copies require aligned packed weights, scale/bias bases and each
actually read scale/bias row. A strided single-row parameter ignores its unused
leading stride. FP8 inputs require 16-byte alignment. L64 FP8 weight staging
uses 8-byte copies and permits 8-byte-aligned views; other weight staging uses
16-byte copies. FP8 prefill's scalar bias reads need only bf16 alignment.
Paired bf16 output stores require 4-byte alignment; scalar stores in a separate
split-K reducer, odd-width outputs or grouped tiles 6 and 7 preserve their bf16
alignment contract. These requirements follow the native pointer types and
[PTX memory-access alignment rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands).

## Native specialization configuration

Each loaded QMM or grouped-expert prefill-form kernel specialization owns
configuration flags for the actual CUDA device count initialized by Torch
(gate/up on 16-pair items needs more than the default 48 KB of dynamic shared
memory). The flags use `std::call_once` per
device, with no additional GPU-count ceiling. A device guard establishes the
target primary context before configuration. A successful
`cudaFuncSetAttribute` publishes that device's flag; failure aborts the operation
before launch and leaves the flag uncommitted. CUDA's
[shared-memory attribute contract](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__HIGHLEVEL.html)
allows device-dependent limits and reports invalid requests as errors.

The public native wrapper validates tiled grid products and reduction grids in
int64 before scratch allocation or signed-int narrowing. Selected 256-column
prefill tiles also validate their full N-padding arithmetic. Grouped exports
require their declared sm_12x device capability; other CUDA devices use the
existing public grouped-matmul fallback to individual supported projections.

These caches have the loaded module's process lifetime, matching the existing
Torch primary-context callers. Visible device identity and primary contexts
remain stable during that lifetime; resetting CUDA contexts requires restarting
the process and rebuilding its Torch/model state. The project does not use
custom driver contexts or reset/reuse live model allocations. Switching among
compatible devices configures each independently. A two-device execution gate
is required to claim that path validated; one-device execution proves only that
device's configuration and output contracts.

## Grouped-expert prefill items

A prefill-form expert plan's items hold at most `tile` pairs, the size
`experts.route` planned them with: 64 (`experts_prefill.cu`, 64-row tiles) or 16
(`experts_window.cu`, 16-row tiles). The native `prefill` launch takes that tile
and runs one item a tile, so it must equal the plan's: a 16-row launch on items
of 64 would skip rows. `experts.gate_up` and `experts.down` pass `plan.tile` and
size the grid with `experts.max_items` for it. Like GDN schedules, items and
counts are trusted products of the plan kernel; the launch does not read them
back to check them. A pair's bits do not depend on its item's size.

## Flash Next concurrent rounds: static tables, captured rounds and PDL

A concurrent decoder (`MultiDecoder`) owns its rounds' device tables at fixed addresses
(`gdn_multi.RoundTables`; `attn_multi.StepTables` for the verify and the MTP steps). A step writes them through a
pinned mirror with one copy each on the current stream before its forward. The mirror is rewritten only after an
event shows that copy complete, and the device tables only by later copies on the same stream, so every earlier
consumer is stream-ordered before the rewrite. Pointees are the decoder's States: a resize, shrink or drop frees
storage through the same stream's caching allocator. Rows past a stream's window and chunks past a row's keys are
launched (fixed geometry) and write nothing; zero pending counts are the documented no-op.

`multi_graphs.RoundGraphs` captures each pure decode round's verify forward and each MTP step in pieces that share
one private pool, keyed by (PDL, streams, rows) and the DeltaNet parity. What a step's host data shapes runs eagerly
between the pieces on that step's data: block selection, each stream's n-gram conv tail and the late n-gram reads.
A round replays one key's pieces in order and reads only static buffers afterwards, so no piece's pool memory is
an output. A numeric `TENSORFOLD_MULTI_GRAPH_MIB` is an exact ceiling; unset/blank selects an allowance from
remaining memory after eager warmup. Initial header admission chooses its context without graph memory; the
resolved graph ceiling is bounded by that plan's remaining serving allowance and current live room after required
service growth. Resident graphs protect one complete native/configured window, one layer's resize overlap, bounded
kept/in-flight snapshots and the larger of the gate's workspace reserve or vision plus serving transients. Extra
concurrent KV growth remains opportunistic under the memory gate. The live provider charges the memory floor and
explicit device cap once. Host expert pools keep their separate fixed-pool window reservation policy.
Keys are captured only while their measured memory stays within this allowance: a key's
memory is the drop of free memory across its capture alone, after its eager warm-up's allocator blocks are in
place, and the stream caches' room loses the larger of the kept keys' sum and the whole capture's settled drop.
Startup admission plans the budget only where capture can happen: experts that read their plan on the host
(`--vram-experts`) decline capture and plan none, and the decoder's resolved budget must equal the plan. The original
context/loading admission remains in the receipt. Fresh live-room checks precede warmup and capture; an attempted
key that consumes required service room is rejected and recorded. After synchronization, slot reset, collection
and allocator-cache release, insufficient remaining service room raises `MemoryError`. These observations do not
guarantee a transient memory floor or reserve memory against other processes. Settled retained bytes must also fit
the admitted ceiling. Capture safety uses physical available bytes after its floor/cap, without crediting graph
private-pool cache. A quiescent public `memory_snapshot(mempool_id=..., include_traces=False)` records this pool's
inactive bytes (`total_size - allocated_size`, with no awaiting-free blocks); the ordinary runtime gate excludes
those bytes from reusable allocator spare. This constant is valid because replays allocate no tensors into the
pool and the captured set is immutable until decoder retirement; the native live callable retains its pool owner.
The native allocator supplies this segment proof. Other allocator backends use explicitly recorded physical-only
service admission instead of interpreting an empty/unsupported native snapshot as zero private storage; ordinary
cached bytes are not credited in that capability mode. The memory floor and device cap remain unchanged.
Concurrent requests, including `draft:false`, stay in the scheduler and use its graph-free slot engines, so they
cannot lazily create a serial graph pool after this snapshot. A failed capture also retains its accounted pool:
after successful drain/reset and admissible settlement, eager recovery debits its retained bytes and excludes its
private spare, preserving the original exception. Recapture is refused so pools cannot accumulate. A failed
pool independently retains its original primary, traceback/native links and a strong journal of original/causal
frame-local values before public propagation and allocator measurement. `traceback.clear_frames` and clearing the
caller exception's traceback cannot release those graph-private Tensor locals and invalidate its inactive-byte
receipt. The journal is operation-owned/read-only; failed journal allocation poisons recovery. A failed
drain/reset/accounting, unadmitted retained storage or failed publication poisons the decoder until retirement.
The capture failure stays the raised error; failed cleanup/accounting is attached to it.

`tensorfold.cuda.pdl.ON`, the launch policy every converted kernel's launch reads, is a process global with an
explicit contract in `pdl.py`: only the concurrent decoder writes it (construction, each round and capture, each
time its own policy), on the one thread launching the decode kernels and never while a launch is being built; every
launch in the process follows its current value, which never changes results.

Programmatic dependent launch (`tensorfold/cuda/pdl.py`, `kernels/pdl.cuh`): a kernel launched with the attribute
may be scheduled while its predecessor finishes. Every such kernel calls `griddepcontrol.wait` as its first
operation on every block and path, before any global access, then `griddepcontrol.launch_dependents`; only the
lane matmul loads its immutable weights before the wait. A completed PDL kernel therefore implies its predecessors
completed, writes stay ordered, and arithmetic is unchanged. Torch operations and copies between kernels keep
full dependencies. Kernels compiled or launched without PDL keep their original code (Triton's `PDL` constexpr,
the lane matmul's template flag); the other extension kernels execute the two instructions as no-ops.
`tests/test_pdl_kernel_contract.py` checks the wait-first rule over the converted sources.

## CT native FP4 routed experts

Qwen MoE and Gemma4 native-A4 experts use one storage and arithmetic contract for
resident six-field banks and (Qwen) cached interleaved records. Per-expert fields have
contiguous inner payloads and an explicit outer expert stride. Native boundary
checks prove positive shapes, complete inner spans and byte-range arithmetic
before multiplication or launch; unused singleton strides remain legal.
Cache pool replacement prepares all typed views before committing its new pool.
Consumers enqueue inside the base cache lease, which owns copy readiness and
retirement; a failed retirement retains all owners.

Original CT input-global FP32 multipliers drive group-16 activation quantization
directly. Alpha separately rounds the reciprocal input calibration and its
product with the prepared weight reciprocal. Each gate/up result rounds to
BF16, the checkpoint's activation (Qwen SiLU, Gemma tanh GELU) rounds to BF16, and
its product with up rounds again before independent down-input quantization.
Decode-mode down outputs are FP32 and prompt-mode outputs BF16; Gemma runs every
call in prompt mode. In host windows (cached banks, expert-parallel local picks)
out-of-bank picks are inactive slots and write zero without reading authority.
Stored native block scales admit only unsigned finite E4M3 byte encodings
`0x00..0x7e`, once at load; the general packing permutation preserves all bits.

Resident banks (`ExpertsA4.run_plan`, also ModelOpt's direct-scale banks) read no
picks on the host: the device plan groups each layer's pairs into items of 16 or
64 same-expert pairs (`plan_tile`: 64 once experts average 32 pairs) and traps on
expert IDs outside [0, plan experts). Three launches follow: activation codes (one
quantization per row, for up to 65,535 rows, when the loader's complete CPU bit
proof shows every expert shares the gate/up input global), the gate/up GEMMs with
the activation and the down quantization in one epilogue, and the down GEMM; each
CTA stages an item's weights once per 64-column block. Every output element
keeps the host-window chain (same fragments, input groups ascending, shape-only
K slices added in order), so a pair's bits never depend on its item size, the
other pairs or the row count; tests compare both paths bit for bit. An item whose cell equals the
bank count (a shared expert computed separately) is skipped and its slot left to
the caller; a Qwen bank that loaded the shared expert as its last cell computes it
in the same plan. Retained per-buffer scratch is allocated by the first eager call;
a first call under CUDA graph capture is refused. The resident CT routers are
row-invariant split-K BF16 tensor-core GEMVs (`route.cu`: eight K slices fixed by
K, added in slice order); Gemma's fuses the CT RMS/root/scale input and is followed
by one softmax/top-k/renormalize/per-expert-scale kernel (stable descending order,
lower ID on ties).

The maintained independent oracle packs original checkpoint projections through
the individual linear implementation and computes each routed pair separately.
An additional scalar CPU activation quantizer checks encoded bytes, including
the raw-multiplier reciprocal-roundtrip boundary. These references live in
`tests/nvfp4_a4_inputs.py`; they are not production fallback implementations.
Flash Next's existing W4A16 expert representation and arithmetic are separate.

## Flash Next block selection and RotorQuant centroid decoding

A sparse-attention row keeps its `TOP` best-scoring complete blocks, in block order, ties to the lower block id,
followed by its unfinished tail. `attention._launch_select` chooses the kernel that lists them: `_select` while a
row's scores fit its registers, `_select_tiles` past them, and for windows of fewer than `MULTI_ROWS` rows past
`MULTI_MIN_BLOCKS` blocks the multi-CTA select (`_select_multi`), which splits each row into slices: four radix
histogram passes add each slice's counts into the row's histograms with integer atomics (any order gives the same
sums), and placement seeds each slice with the counts of the slices before it. Every kernel writes the same ids,
lengths and flags bit for bit. A concurrent step scores every row from its own stream's pooled keys
(`attn_multi._scores_multi`, sharing `_score_rows` with `_scores`) and selects all rows in one launch, each row ending
at its own position. The select scratch (four 256-bin histograms a row, 257 placement counts a slice; shapes from
`geometry.QSA_SELECT_*`) is counted in the startup memory plan. `tests/cuda/test_flashnext_qsa_select.py` compares
the whole buffers with `_select_tiles`, `_select` and an independent sort (`tests/cuda/qsa_select_oracle.py`).

Prompt passes (`attention(prompt=True)`, the forward's prompt branch) decode six-bit RotorQuant sides through
`rotorquant_kernel._centroids_lookup`, a register gather of the same canonical constants, so every element has
the selection chain's bits. Every other chunk whose K and V are both six- or eight-bit RotorQuant at head width
256 (`attention._chunk_quarters`, both `_chunks` and `_chunks_multi`) decodes 64-column quarters
(`kv_pair.load_quarter`) whose centroids load from the attention scratch's float32 tables, `CENTROIDS_6` then
`CENTROIDS_8` (`rotorquant_kernel.dequant_half_6` / `dequant_quarter_8`; `geometry.CENTROID_TABLE_BYTES` in the
memory plan): the four quarters' score MMAs accumulate in column order into one accumulator (the whole-tile dot's
MMA sequence as the pinned Triton lowers it), the value update is per output column and the softmax runs on the
same score tile, so the partials keep the whole-tile bits of the six-bit selection chain and the eight-bit register
gather. The table loads are unmasked: RotorQuant payloads are unsigned bytes and masked rows load code 0, so every
index stays inside its table (0..63 six-bit, 64..319 eight-bit). Other pairs keep the whole-tile decode (a six-bit
register gather in the sparse decode kernels spills its table and was slower on GB10). The folded six-bit selection
hard-hung GB10 and is not used; re-qualify on GB10 before changing six- or eight-bit decode or encode kernel code.
`tests/cuda/test_flashnext_kv_pairs.py` checks against captured-operand partials the gather in prompt passes for
every key and value format pair with a six-bit side, and the quarter path (through `_chunks`) in decode passes for
every pair of six- or eight-bit RotorQuant sides (an eight-bit pair's prompt pass launches the same kernel; a
prompt pass with a six-bit side takes the gather). `tests/cuda/test_flashnext_multi_graphs.py` covers
`_chunks_multi`'s quarters through concurrent verify windows and MTP steps, whose logits, streams and cache rows
equal the per-stream forward's bit for bit with `rotorquant6-norm` and `rotorquant8-norm` caches.

Eight-bit RotorQuant encoders (`rotorquant_kernel.quant_groups_8`, through `kv_pair.store_side` from `glue._prep_row` in
`_attn_prep` and `attn_multi._prep_multi`) read the same scratch's tables: the binary-lifting thresholds `LIFTING_8`
(`AttnScratch.thresholds`, 255 float32; `geometry.THRESHOLD_TABLE_BYTES` in the memory plan) and `CENTROIDS_8` at offset
64 of `AttnScratch.centroids` (`attention.codebook` makes both). `_indices_8` makes the eight-bit binary-lifting chain's
eight comparisons and steps with each level's boundary loaded instead of selected: level L's 2^L cells sit at [2^L - 1,
2^(L + 1) - 1) and the cell is q >> (8 - L), so every load stays within 0..254 for any input, NaN included, and every
constant is an exact float32: the codes are the selection chain's. The norm8 scale gathers its centroids from the loaded
table broadcast to a register tile, as `_centroids(q, 8)` gathers from its selected table: the FP32 energy sum's order
follows that register layout, and a plain indexed load would lay the centroids out differently (another summation order
can change the stored scale). The other encoders keep their register selection chains. `glue.attn_prep` refuses an
eight-bit RotorQuant side whose tables are missing or differ in shape, dtype, device or contiguity
(`kv_pair.validate_tables`); the concurrent decoder's `attn_multi.layer` passes its `AttnScratch`'s tables, made at
construction and never written, without checking them again. `tests/cuda/test_flashnext_rotorquant.py` keeps the
selecting eight-bit encoder (its selection chain `_indices8_selected`) as the oracle whose codes and scales the table
encoder equals bit for bit, and checks `_indices_8` at every midpoint and its FP32 neighbours against independently
expected indices.
