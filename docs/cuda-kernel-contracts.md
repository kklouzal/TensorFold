# Shared CUDA kernel contracts

The shared GDN and lane-matmul extensions enqueue work on PyTorch's current
CUDA stream. Their native wrappers check tensor metadata, device identity,
kernel dimension ranges, and known write/read overlap before launch. These
checks do not synchronize device data or change floating-point operations.

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

## QMM buffers and aliases

Row-strided x, scales and biases remain supported where the native kernel
declares contiguous rows. Overlap checks compare the bytes actually accessed,
so separate regions of one allocation and outputs in an unused row gap remain
valid. Sharing read-only weight or scale tensors is allowed.

For a direct or clustered matmul, output writes must be disjoint from every
input read in that launch. With non-cluster split-K staging, the producer writes
only the used scratch prefix `(SK, M, n)`; that prefix must be disjoint from its
inputs. A subsequent reduction reads scratch and writes an output disjoint from
scratch. Because input reads have already completed, this staged output may
alias an input. `reduce=False` with SK greater than one does not write `out`;
SK equal to one writes `out` and does not use `part`. Buffers unused by the chosen
path impose no alias restriction. Scratch capacity beyond its used prefix is
not accessed.

Grouped QMM outputs must be mutually disjoint and disjoint from every group's
actual input reads. Prefill and FP8 prefill outputs must be disjoint from their
actual input reads. The
grouped tile selector is 0 through 12; the signed `early` flag uses a negative
value for device policy, zero to disable and a positive value to enable it.
These metadata checks preserve K-slice addition order and existing bf16/fp32
rounding.

Outputs and split-K scratch may occupy packed padding that the selected kernel
never reads. The direct lane kernel loads columns through n rounded up to 64.
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

Each loaded QMM kernel specialization owns configuration flags for the actual
CUDA device count initialized by Torch. The flags use `std::call_once` per
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
