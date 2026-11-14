// Low-latency all-gather for small per-rank payloads over CUDA IPC peer mailboxes.
//
// Protocol (the "LL" line format of NCCL, and the generation-tagged packet idea
// of 1Cat-vLLM's Apache-2.0 csrc/sm70_ring_collectives.cu; no code is copied):
// every 8 payload bytes travel as one 16-byte line {d0, flag, d1, flag}. Each
// 8-byte half is written by a single store and so is observed whole; a reader
// accepts a line only when both flags equal the call's generation. Data and
// its validity therefore arrive together: no separate flag, fence or host step.
//
// Mailbox (one cudaMalloc per rank, exported once through cudaIpc):
//   line[(parity * world + source) * capacity + index], parity = generation & 1.
// Exactly one writer exists per mailbox line: the source itself (NVLink
// neighbour) or the one relay assigned to that line, which is an NVLink
// neighbour of both ends and forwards the line as soon as it receives it.
//
// Generations: state[0] holds the last completed generation (0 = none). The
// next is state[0] + 1, except that 0xFFFFFFFF is followed by 2: generations
// are never 0 (the cleared-slot value) and consecutive calls always alternate
// parity (odd 0xFFFFFFFF -> even 2; wrapping to 1 would repeat odd parity).
//
// Reuse ordering: consecutive calls use disjoint parity slots, so call g+1
// never writes a slot that call g may still read. A writer reaches the next
// call of g's parity (g+2 in sequence) only after its own call g+1 received every
// rank's g+1 data, so each destination had already started call g+1 and (stream
// order) finished every read of call g. A reader clears each consumed line to
// zero, so a slot holds either zero or the line of its pending call; a stale
// line therefore never matches, whatever generation values wrap reuses.
//
// Requirement: on each rank, calls on one communicator are stream-ordered (each
// starts after the previous completes). Consecutive eager calls, or calls in the
// same capture, on different streams are joined here with an event. Every rank
// issues the same sequence of calls on a communicator.
//
// Progress and residency: a receiving block spins until its peers publish, so
// every launch must eventually have all its blocks running. One launch is
// bounded to half of the device's resident blocks (occupancy x SMs / 2): on an
// otherwise idle device its whole grid is co-resident, and two communicators'
// spinning launches on concurrent streams (e.g. tensor- and expert-parallel
// groups, dispatched in any order) also fit together. When other kernels hold
// SMs, the row order below (sender, NVLink sources, relayed sources) still
// progresses provided blocks of a launch are dispatched in blockIdx order;
// that is observed hardware behaviour, not a documented CUDA guarantee. More
// than two concurrently spinning communicators per device are unsupported (as
// for NCCL kernels on concurrent streams).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kMaxWorld = 8;
constexpr int kThreads = 256;
constexpr int kSplit = 4;  // relayed lines are assigned to relays by line index % kSplit
// Per-rank payload ceiling: at 256 KiB the measured in-graph latency already
// equals NCCL's on 8x V100-SXM2 and NCCL is faster beyond (README), so larger
// mailboxes would only cost memory (4 * world * capacity bytes per rank).
constexpr long long kMaxCapacityBytes = 256 * 1024;

struct Params {
  const unsigned char* send;
  unsigned char* recv;
  long long bytes;
  long long lines;
  long long capacity;  // lines per (parity, source) slot
  uint4* mailbox[kMaxWorld];  // mapped base of every destination written or read
  uint32_t* state;            // [0] last generation, [1] block completion ticket
  const volatile int* abort;  // mapped host flag; nonzero ends spinning on teardown
  uint32_t forward[kSplit][kMaxWorld];  // [line % kSplit][source]: destinations this rank relays to
  uint32_t direct;              // destinations of this rank's own lines
  int order[kMaxWorld];         // source handled by each block row
  int rank;
  int world;
};

__device__ __forceinline__ void store_line(uint4* address, uint4 value) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(address), "r"(value.x),
               "r"(value.y), "r"(value.z), "r"(value.w)
               : "memory");
}

__device__ __forceinline__ uint4 load_line(const uint4* address) {
  uint4 value;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ uint2 read_payload(const unsigned char* p, long long remaining) {
  if (remaining >= 8 && (reinterpret_cast<uintptr_t>(p) & 7u) == 0) return *reinterpret_cast<const uint2*>(p);
  uint32_t word[2] = {0u, 0u};
  const int count = remaining < 8 ? int(remaining) : 8;
  for (int k = 0; k < count; ++k) word[k >> 2] |= uint32_t(p[k]) << ((k & 3) * 8);
  return make_uint2(word[0], word[1]);
}

__device__ __forceinline__ void write_payload(unsigned char* p, long long remaining, uint32_t lo, uint32_t hi) {
  if (remaining >= 8 && (reinterpret_cast<uintptr_t>(p) & 7u) == 0) {
    *reinterpret_cast<uint2*>(p) = make_uint2(lo, hi);
    return;
  }
  const int count = remaining < 8 ? int(remaining) : 8;
  for (int k = 0; k < count; ++k) p[k] = (unsigned char)(((k < 4 ? lo : hi) >> ((k & 3) * 8)) & 255u);
}

__global__ void __launch_bounds__(kThreads) small_gather_kernel(Params p) {
  __shared__ uint32_t shared_generation;
  if (threadIdx.x == 0) {
    uint32_t next = *reinterpret_cast<volatile uint32_t*>(p.state) + 1u;
    shared_generation = next == 0u ? 2u : next;  // Skip 0 and keep parity alternating (header).
  }
  __syncthreads();
  const uint32_t generation = shared_generation;
  const long long parity_base = (long long)(generation & 1u) * p.world;
  // Row order[0] == rank publishes this rank's lines; every other row receives
  // (and relays) one source's lines. Rows are ordered sender, NVLink sources,
  // relayed sources: a row waits only on earlier classes of rows elsewhere, so
  // blockIdx-order dispatch progresses even when the grid is not resident (header).
  const int source = p.order[blockIdx.y];
  const long long slot_base = (parity_base + source) * p.capacity;
  const long long stride = (long long)gridDim.x * blockDim.x;
  bool aborted = false;
  if (source == p.rank) {
    const bool in_place = p.send == p.recv + (long long)p.rank * p.bytes;
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x; index < p.lines; index += stride) {
      const long long offset = index * 8, remaining = p.bytes - offset;
      const uint2 mine = read_payload(p.send + offset, remaining);
      const uint4 line = make_uint4(mine.x, generation, mine.y, generation);
      for (uint32_t mask = p.direct; mask; mask &= mask - 1)
        store_line(p.mailbox[__ffs(mask) - 1] + slot_base + index, line);
      if (!in_place) write_payload(p.recv + (long long)p.rank * p.bytes + offset, remaining, mine.x, mine.y);
    }
  } else {
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x; index < p.lines && !aborted;
         index += stride) {
      uint4* local = p.mailbox[p.rank] + slot_base + index;
      uint4 value = load_line(local);
      for (uint32_t spins = 1; value.y != generation || value.w != generation; ++spins) {
        if ((spins & 1023u) == 0u && *p.abort) {
          aborted = true;
          break;
        }
        value = load_line(local);
      }
      if (aborted) break;
      for (uint32_t out = p.forward[index % kSplit][source]; out; out &= out - 1)
        store_line(p.mailbox[__ffs(out) - 1] + slot_base + index, value);
      store_line(local, make_uint4(0u, 0u, 0u, 0u));
      const long long offset = index * 8;
      write_payload(p.recv + (long long)source * p.bytes + offset, p.bytes - offset, value.x, value.z);
    }
  }
  // On abort the communicator is being torn down and its outputs are undefined.
  if (__syncthreads_or(aborted) == 0 && threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(p.state + 1, 1u) == gridDim.x * gridDim.y - 1) {
      p.state[1] = 0u;
      p.state[0] = generation;  // Every block read state[0] before taking its ticket.
    }
  }
}


// Row all-reduce of FP32 partials [rows, dims] into FP16 sums, on the same mailboxes and generation
// sequence as the gather (each call is one generation; the reuse-ordering argument above holds: every
// rank receives every other rank's phase-2 lines, so each destination has started the call).
// Phase 1 (reduce-scatter): rank s sends columns [j * slice, (j + 1) * slice) of every row to their
// owner j; a non-neighbour owner is reached through relay via[i % kSplit][j], which holds the line in
// its transit region and forwards it. The owner sums the world partials in rank order 0..world-1 (the
// exact FP32 sequence the gathered consumer used), rounds once to FP16 and (phase 2) publishes four
// sums per line to every rank, relayed as in the gather. Slot (parity, source) of a mailbox holds:
//   [0, L)                    phase-1 lines from `source` for this mailbox's own slice
//   [L + o * L, L + (o+1)*L)  transit lines from `source` to its o-th non-neighbour (relays only)
//   [(1 + nb) * L, + L / 2)   phase-2 lines of `source`'s summed slice
// with L = rows * slice / 2. Rows (blockIdx.y): 0 sender, 1 transit relay, 2 reducer, then one row per
// phase-2 source (NVLink sources first): each waits only on earlier rows elsewhere (header argument).
struct ReduceParams {
  Params base;
  const float* send;
  __half* recv;
  int rows, dims, slice;
  long long lines;               // L
  int neighbours_max;            // nb: transit regions per slot
  int via[kSplit][kMaxWorld];    // relay of this rank's lines to non-neighbour owner j (-1: direct)
  int ordinal[kMaxWorld][kMaxWorld];  // [source][destination] transit region index, -1 for neighbours
  int far[kMaxWorld][kMaxWorld];      // [source][region] destination of that transit region, -1 unused
};

__device__ __forceinline__ bool wait_line(const uint4* address, uint32_t generation, const volatile int* abort,
                                          uint4& value) {
  value = load_line(address);
  for (uint32_t spins = 1; value.y != generation || value.w != generation; ++spins) {
    if ((spins & 1023u) == 0u && *abort) return false;
    value = load_line(address);
  }
  return true;
}

__global__ void __launch_bounds__(kThreads) small_allreduce_kernel(ReduceParams q) {
  const Params& p = q.base;
  __shared__ uint32_t shared_generation;
  if (threadIdx.x == 0) {
    uint32_t next = *reinterpret_cast<volatile uint32_t*>(p.state) + 1u;
    shared_generation = next == 0u ? 2u : next;
  }
  __syncthreads();
  const uint32_t generation = shared_generation;
  const long long parity_base = (long long)(generation & 1u) * p.world;
  const long long L = q.lines, half_lines = L / 2, per_row = q.slice / 2;
  const long long stride = (long long)gridDim.x * blockDim.x;
  const long long first = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long phase2 = (1 + q.neighbours_max) * L;
  bool aborted = false;
  const int role = blockIdx.y;
  if (role == 0) {  // sender: this rank's slice of every other owner
    for (int owner = 0; owner < p.world; ++owner) {
      if (owner == p.rank) continue;
      const long long slot = (parity_base + p.rank) * p.capacity;
      for (long long i = first; i < L; i += stride) {
        const long long r = i / per_row, c = owner * q.slice + (i % per_row) * 2;
        const float2 v = *reinterpret_cast<const float2*>(q.send + r * q.dims + c);
        const uint4 line = make_uint4(__float_as_uint(v.x), generation, __float_as_uint(v.y), generation);
        const int relay = q.via[i % kSplit][owner];
        if (relay < 0) store_line(p.mailbox[owner] + slot + i, line);
        else store_line(p.mailbox[relay] + slot + L + q.ordinal[p.rank][owner] * L + i, line);
      }
    }
  } else if (role == 1) {  // transit relay: each thread forwards one (source, destination, line) item
    const long long items = (long long)p.world * q.neighbours_max * L;
    for (long long item = first; item < items && !aborted; item += stride) {
      const int source = int(item / (q.neighbours_max * L)), o = int(item / L % q.neighbours_max);
      const long long i = item % L;
      const int destination = q.far[source][o];
      if (source == p.rank || destination < 0 || destination == p.rank
          || !(p.forward[i % kSplit][source] >> destination & 1u))
        continue;
      const long long slot = (parity_base + source) * p.capacity;
      uint4* local = p.mailbox[p.rank] + slot + L + o * L + i;
      uint4 value;
      if (!wait_line(local, generation, p.abort, value)) { aborted = true; break; }
      store_line(p.mailbox[destination] + slot + i, value);
      store_line(local, make_uint4(0u, 0u, 0u, 0u));
    }
  } else if (role == 2) {  // reducer: one phase-1 line per thread; lane pairs form one phase-2 line
    const int lane = threadIdx.x & 31;
    const long long padded = (L + 31) / 32 * 32;  // whole warps take part in the pairing shuffle
    for (long long i = first; i < padded; i += stride) {
      const bool live = i < L && !aborted;
      float a = 0.f, b = 0.f;
      if (live) {
        const long long r = i / per_row, c = p.rank * q.slice + (i % per_row) * 2;
        uint4 value[kMaxWorld];
        #pragma unroll
        for (int source = 0; source < kMaxWorld; ++source)  // every peer's line requested at once
          if (source < p.world && source != p.rank)
            value[source] = load_line(p.mailbox[p.rank] + (parity_base + source) * p.capacity + i);
        #pragma unroll
        for (int source = 0; source < kMaxWorld; ++source) {
          if (source >= p.world || aborted) continue;
          float x, y;
          if (source == p.rank) {
            const float2 v = *reinterpret_cast<const float2*>(q.send + r * q.dims + c);
            x = v.x; y = v.y;
          } else {
            uint4* local = p.mailbox[p.rank] + (parity_base + source) * p.capacity + i;
            if (value[source].y != generation || value[source].w != generation)
              if (!wait_line(local, generation, p.abort, value[source])) { aborted = true; continue; }
            store_line(local, make_uint4(0u, 0u, 0u, 0u));
            x = __uint_as_float(value[source].x); y = __uint_as_float(value[source].z);
          }
          if (source == 0) { a = x; b = y; } else { a = a + x; b = b + y; }  // rank order, as the consumer
        }
      }
      const __half2 pair = __floats2half2_rn(a, b);
      const uint32_t word = *reinterpret_cast<const uint32_t*>(&pair);
      const uint32_t next = __shfl_down_sync(0xffffffffu, word, 1);
      if (live && !aborted && (lane & 1) == 0) {  // i even: lines i and i + 1 (same row) form phase-2 line i/2
        const long long h = i / 2;
        const uint4 line = make_uint4(word, generation, next, generation);
        const long long slot = (parity_base + p.rank) * p.capacity + phase2;
        for (uint32_t mask = p.direct; mask; mask &= mask - 1)
          store_line(p.mailbox[__ffs(mask) - 1] + slot + h, line);
        const long long r = i / per_row, c = p.rank * q.slice + (i % per_row) * 2;
        *reinterpret_cast<uint2*>(q.recv + r * q.dims + c) = make_uint2(word, next);
      }
    }
  } else {  // phase-2 receiver of one source's summed slice
    const int source = p.order[role - 2];  // order[1..] are the other ranks, NVLink sources first
    const long long slot = (parity_base + source) * p.capacity + phase2;
    for (long long h = first; h < half_lines && !aborted; h += stride) {
      uint4* local = p.mailbox[p.rank] + slot + h;
      uint4 value;
      if (!wait_line(local, generation, p.abort, value)) { aborted = true; break; }
      for (uint32_t out = p.forward[h % kSplit][source]; out; out &= out - 1)
        store_line(p.mailbox[__ffs(out) - 1] + slot + h, value);
      store_line(local, make_uint4(0u, 0u, 0u, 0u));
      const long long r = (2 * h) / per_row, c = source * q.slice + ((2 * h) % per_row) * 2;
      *reinterpret_cast<uint2*>(q.recv + r * q.dims + c) = make_uint2(value.x, value.z);
    }
  }
  if (__syncthreads_or(aborted) == 0 && threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(p.state + 1, 1u) == gridDim.x * gridDim.y - 1) {
      p.state[1] = 0u;
      p.state[0] = generation;
    }
  }
}

void check(cudaError_t status, const char* what) {
  TORCH_CHECK(status == cudaSuccess, "small all-gather ", what, ": ", cudaGetErrorString(status));
}

class SmallGather {
 public:
  SmallGather(int64_t rank, int64_t world, int64_t capacity_bytes) {
    TORCH_CHECK(world >= 2 && world <= kMaxWorld && rank >= 0 && rank < world, "small all-gather requires 2..8 ranks");
    TORCH_CHECK(capacity_bytes > 0 && capacity_bytes % 8 == 0 && capacity_bytes <= kMaxCapacityBytes,
                "small all-gather capacity must be a positive multiple of 8 bytes up to ", kMaxCapacityBytes);
    rank_ = int(rank);
    world_ = int(world);
    capacity_lines_ = capacity_bytes / 8;
    for (auto& peer : mailbox_) peer = nullptr;
    try {
      allocate();
    } catch (...) {
      try {
        release();
      } catch (...) {
      }
      throw;  // The allocation failure is primary.
    }
    mailbox_[rank_] = reinterpret_cast<uint4*>(own_);
  }

  void allocate() {
    check(cudaGetDevice(&device_), "device query");
    mailbox_bytes_ = size_t(2) * world_ * capacity_lines_ * sizeof(uint4);
    check(cudaMalloc(&own_, mailbox_bytes_), "mailbox allocation");
    check(cudaMalloc(&state_, 2 * sizeof(uint32_t)), "state allocation");
    check(cudaHostAlloc(&abort_host_, sizeof(int), cudaHostAllocMapped), "abort flag allocation");
    *abort_host_ = 0;
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&abort_device_), abort_host_, 0), "abort flag mapping");
    check(cudaMemset(own_, 0, mailbox_bytes_), "mailbox clear");
    check(cudaMemset(state_, 0, 2 * sizeof(uint32_t)), "state clear");
    check(cudaEventCreateWithFlags(&tail_, cudaEventDisableTiming), "event creation");
    check(cudaDeviceSynchronize(), "initial clear");  // Zeroed before the handle can be published.
    int per_sm = 0, sms = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, small_gather_kernel, kThreads, 0), "occupancy");
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_), "SM count");
    // At most half the device's resident blocks per launch (header: two
    // communicators stay co-resident). At the 256 KiB ceiling a row needs 128
    // blocks; fewer only loop more lines per thread.
    row_blocks_ = std::max(1, per_sm * sms / (2 * world_));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, small_allreduce_kernel, kThreads, 0), "occupancy");
    reduce_row_blocks_ = std::max(1, per_sm * sms / (2 * (world_ + 2)));
  }
  ~SmallGather() {
    try {
      release();  // The Python owner calls release() explicitly and reports its failure.
    } catch (...) {
    }
  }

  py::bytes handle() const {
    cudaIpcMemHandle_t value;
    check(cudaIpcGetMemHandle(&value, own_), "IPC export");
    return py::bytes(reinterpret_cast<const char*>(&value), sizeof(value));
  }

  std::string pci_bus_id() const {
    char text[64] = {0};
    check(cudaDeviceGetPCIBusId(text, sizeof(text), device_), "PCI bus query");
    return text;
  }

  // handles[r] for every destination this rank writes; forward[k * world + s] relays
  // lines of source s whose index % kSplit == k.
  // incoming: sources whose own lines this rank receives directly (row order only).
  void connect(const std::vector<py::bytes>& handles, int64_t direct, int64_t incoming,
               const std::vector<int64_t>& forward) {
    TORCH_CHECK(!connected_ && int(handles.size()) == world_ && int(forward.size()) == kSplit * world_,
                "small all-gather connects once with one handle and route per rank");
    const uint32_t all = (1u << world_) - 1u, self = 1u << rank_;
    uint32_t targets = uint32_t(direct);
    TORCH_CHECK(direct >= 0 && (targets & ~all) == 0 && (targets & self) == 0, "invalid direct route");
    for (int k = 0; k < kSplit; ++k) {
      for (int source = 0; source < world_; ++source) {
        const int64_t raw = forward[k * world_ + source];
        const uint32_t out = uint32_t(raw);
        TORCH_CHECK(raw >= 0 && (out & ~all) == 0 && (out & self) == 0 && (out >> source & 1u) == 0 &&
                        (source != rank_ || out == 0),
                    "invalid relay route");
        forward_[k][source] = out;
        targets |= out;
      }
    }
    direct_ = uint32_t(direct);
    int row = 0;
    order_[row++] = rank_;
    for (int pass = 0; pass < 2; ++pass)
      for (int source = 0; source < world_; ++source)
        if (source != rank_ && bool(incoming >> source & 1) == (pass == 0)) order_[row++] = source;
    for (int peer = 0; peer < world_; ++peer) {
      if (!(targets >> peer & 1u)) continue;
      const std::string raw = handles[peer];
      TORCH_CHECK(raw.size() == sizeof(cudaIpcMemHandle_t), "IPC handle size mismatch");
      cudaIpcMemHandle_t value;
      std::memcpy(&value, raw.data(), sizeof(value));
      void* mapped = nullptr;
      check(cudaIpcOpenMemHandle(&mapped, value, cudaIpcMemLazyEnablePeerAccess), "IPC open");
      mailbox_[peer] = reinterpret_cast<uint4*>(mapped);
      opened_.push_back(mapped);
    }
    connected_ = true;
  }

  void gather(int64_t send, int64_t recv, int64_t bytes, int64_t stream_value) {
    TORCH_CHECK(connected_ && !released_, "small all-gather is not connected");
    TORCH_CHECK(bytes > 0 && (bytes + 7) / 8 <= capacity_lines_, "small all-gather payload exceeds its capacity");
    auto stream = reinterpret_cast<cudaStream_t>(stream_value);
    cudaStreamCaptureStatus status;
    unsigned long long capture = 0;
    check(cudaStreamGetCaptureInfo(stream, &status, &capture), "capture query");
    if (status != cudaStreamCaptureStatusActive) capture = 0;
    if (has_tail_ && tail_stream_ != stream && tail_capture_ == capture)
      check(cudaStreamWaitEvent(stream, tail_, 0), "stream join");
    Params p{};
    p.send = reinterpret_cast<const unsigned char*>(send);
    p.recv = reinterpret_cast<unsigned char*>(recv);
    p.bytes = bytes;
    p.lines = (bytes + 7) / 8;
    p.capacity = capacity_lines_;
    for (int r = 0; r < kMaxWorld; ++r) p.mailbox[r] = mailbox_[r];
    p.state = state_;
    p.abort = abort_device_;
    for (int k = 0; k < kSplit; ++k)
      for (int r = 0; r < kMaxWorld; ++r) p.forward[k][r] = forward_[k][r];
    p.direct = direct_;
    p.rank = rank_;
    p.world = world_;
    for (int r = 0; r < kMaxWorld; ++r) p.order[r] = order_[r];
    const long long blocks = std::min<long long>((p.lines + kThreads - 1) / kThreads, row_blocks_);
    small_gather_kernel<<<dim3(unsigned(blocks), unsigned(world_)), kThreads, 0, stream>>>(p);
    check(cudaGetLastError(), "launch");
    check(cudaEventRecord(tail_, stream), "stream tail");
    has_tail_ = true;
    tail_stream_ = stream;
    tail_capture_ = capture;
  }


  // via[k * world + j]: relay of this rank's residue-k phase-1 lines to owner j (-1: NVLink neighbour or self);
  // ordinal[s * world + j]: transit region of the pair (s, j) on its relays (-1: neighbours). Identical on
  // every rank (computed from the gather routes); `regions` = transit regions per slot.
  void connect_reduce(const std::vector<int64_t>& via, const std::vector<int64_t>& ordinal, int64_t regions) {
    TORCH_CHECK(connected_ && !reduce_connected_ && int(via.size()) == kSplit * world_
                    && int(ordinal.size()) == world_ * world_ && regions >= 0 && regions < world_,
                "small all-reduce connects once after the gather with one route table per rank");
    for (int k = 0; k < kSplit; ++k)
      for (int j = 0; j < world_; ++j) {
        const int64_t relay = via[k * world_ + j];
        TORCH_CHECK(relay >= -1 && relay < world_ && relay != rank_ && (relay < 0 || mailbox_[relay] != nullptr)
                        && (relay >= 0 || j == rank_ || mailbox_[j] != nullptr),
                    "invalid all-reduce relay");
        via_[k][j] = int(relay);
      }
    for (int a = 0; a < world_ * world_; ++a) {
      TORCH_CHECK(ordinal[a] >= -1 && ordinal[a] < regions, "invalid all-reduce transit region");
      ordinal_[a / world_][a % world_] = int(ordinal[a]);
    }
    regions_ = int(regions);
    reduce_connected_ = true;
  }

  // Lines one slot needs for a [rows, dims] all-reduce (0 when the shape is outside this communicator).
  long long reduce_lines(int64_t rows, int64_t dims) const {
    if (!reduce_connected_ || rows <= 0 || dims <= 0 || dims % (4 * world_) != 0) return 0;
    const long long lines = rows * (dims / world_) / 2;
    const long long need = (1 + regions_) * lines + lines / 2;
    return need <= capacity_lines_ ? need : 0;
  }

  // send: FP32 [rows, dims] partials; recv: FP16 [rows, dims] rank-ordered sums. Caller checked reduce_lines.
  void allreduce(int64_t send, int64_t recv, int64_t rows, int64_t dims, int64_t stream_value) {
    TORCH_CHECK(connected_ && !released_ && reduce_lines(rows, dims) > 0, "small all-reduce shape is not admitted");
    auto stream = reinterpret_cast<cudaStream_t>(stream_value);
    cudaStreamCaptureStatus status;
    unsigned long long capture = 0;
    check(cudaStreamGetCaptureInfo(stream, &status, &capture), "capture query");
    if (status != cudaStreamCaptureStatusActive) capture = 0;
    if (has_tail_ && tail_stream_ != stream && tail_capture_ == capture)
      check(cudaStreamWaitEvent(stream, tail_, 0), "stream join");
    ReduceParams q{};
    Params& p = q.base;
    p.capacity = capacity_lines_;
    for (int r = 0; r < kMaxWorld; ++r) p.mailbox[r] = mailbox_[r];
    p.state = state_;
    p.abort = abort_device_;
    for (int k = 0; k < kSplit; ++k)
      for (int r = 0; r < kMaxWorld; ++r) p.forward[k][r] = forward_[k][r];
    p.direct = direct_;
    p.rank = rank_;
    p.world = world_;
    for (int r = 0; r < kMaxWorld; ++r) p.order[r] = order_[r];
    q.send = reinterpret_cast<const float*>(send);
    q.recv = reinterpret_cast<__half*>(recv);
    q.rows = int(rows);
    q.dims = int(dims);
    q.slice = int(dims / world_);
    q.lines = rows * (dims / world_) / 2;
    q.neighbours_max = regions_;
    for (int k = 0; k < kSplit; ++k)
      for (int j = 0; j < kMaxWorld; ++j) q.via[k][j] = j < world_ ? via_[k][j] : -1;
    for (int a = 0; a < kMaxWorld; ++a)
      for (int b = 0; b < kMaxWorld; ++b) {
        q.ordinal[a][b] = a < world_ && b < world_ ? ordinal_[a][b] : -1;
        q.far[a][b] = -1;
      }
    for (int a = 0; a < world_; ++a)
      for (int b = 0; b < world_; ++b)
        if (ordinal_[a][b] >= 0) q.far[a][ordinal_[a][b]] = b;
    // The widest role: the transit relay scans world x regions x lines items (about a third its own).
    const long long work = std::max<long long>(q.lines, (long long)world_ * regions_ * q.lines / 4);
    const long long blocks = std::min<long long>((work + kThreads - 1) / kThreads, reduce_row_blocks_);
    small_allreduce_kernel<<<dim3(unsigned(blocks), unsigned(world_ + 2)), kThreads, 0, stream>>>(q);
    check(cudaGetLastError(), "launch");
    check(cudaEventRecord(tail_, stream), "stream tail");
    has_tail_ = true;
    tail_stream_ = stream;
    tail_capture_ = capture;
  }

  // Test-only: start the generation sequence after `last` (every rank seeds the
  // same value) so wrap handling is exercised. Only before the first gather.
  void seed_generation(int64_t last) {
    TORCH_CHECK(!has_tail_ && !released_ && last >= 0 && last <= 0xFFFFFFFFll,
                "small all-gather generation is seeded once before any gather");
    const uint32_t value = uint32_t(last);
    check(cudaMemcpy(state_, &value, sizeof(value), cudaMemcpyHostToDevice), "generation seed");
  }

  // Ends spinning kernels; their outputs are undefined. Teardown only.
  void abort() {
    if (abort_host_ != nullptr) *reinterpret_cast<volatile int*>(abort_host_) = 1;
    aborted_ = true;
  }

  // Callers fence the device first, so no local kernel uses any mapping. In an
  // ordinary close every rank issued the same calls, so every write into this
  // mailbox was consumed by a completed local call and no peer still writes it.
  // After abort() peers may still be inside (or not yet past) a gather that
  // writes this mailbox through their IPC mapping. The CUDA documentation does
  // not define freeing an allocation that another process still has mapped, so
  // an aborted communicator deliberately retains its own mailbox (bounded:
  // 4 * world * capacity bytes per aborted communicator) until process exit.
  void release() {
    if (released_) return;
    released_ = true;
    cudaError_t first = cudaSuccess;
    auto keep = [&](cudaError_t status) { if (first == cudaSuccess && status != cudaSuccess) first = status; };
    for (void* mapped : opened_) keep(cudaIpcCloseMemHandle(mapped));
    opened_.clear();
    if (tail_ != nullptr) keep(cudaEventDestroy(tail_));
    if (own_ != nullptr && !aborted_) keep(cudaFree(own_));  // Retained after abort (above).
    if (state_ != nullptr) keep(cudaFree(state_));
    if (abort_host_ != nullptr) keep(cudaFreeHost(abort_host_));
    tail_ = nullptr;
    own_ = nullptr;
    state_ = nullptr;
    abort_host_ = nullptr;
    check(first, "release");
  }

 private:
  int rank_ = 0, world_ = 0, device_ = 0;
  long long capacity_lines_ = 0;
  size_t mailbox_bytes_ = 0;
  void* own_ = nullptr;
  uint32_t* state_ = nullptr;
  int* abort_host_ = nullptr;
  int* abort_device_ = nullptr;
  uint4* mailbox_[kMaxWorld];
  uint32_t forward_[kSplit][kMaxWorld] = {};
  uint32_t direct_ = 0;
  int order_[kMaxWorld] = {};
  long long row_blocks_ = 1, reduce_row_blocks_ = 1;
  int via_[kSplit][kMaxWorld] = {};
  int ordinal_[kMaxWorld][kMaxWorld] = {};
  int regions_ = 0;
  bool reduce_connected_ = false;
  std::vector<void*> opened_;
  cudaEvent_t tail_ = nullptr;
  cudaStream_t tail_stream_ = nullptr;
  unsigned long long tail_capture_ = 0;
  bool has_tail_ = false, connected_ = false, released_ = false, aborted_ = false;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<SmallGather>(m, "SmallGather")
      .def(py::init<int64_t, int64_t, int64_t>())
      .def("handle", &SmallGather::handle)
      .def("pci_bus_id", &SmallGather::pci_bus_id)
      .def_property_readonly_static("split", [](py::object) { return kSplit; })
      .def("connect", &SmallGather::connect)
      .def("gather", &SmallGather::gather)
      .def("connect_reduce", &SmallGather::connect_reduce)
      .def("reduce_lines", &SmallGather::reduce_lines)
      .def("allreduce", &SmallGather::allreduce)
      .def("seed_generation", &SmallGather::seed_generation)
      .def_property_readonly_static("max_capacity", [](py::object) { return kMaxCapacityBytes; })
      .def("abort", &SmallGather::abort)
      .def("release", &SmallGather::release);
}
