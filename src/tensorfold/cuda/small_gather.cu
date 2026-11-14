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
// Reuse ordering: a writer reaches call g+2 only after its call g+1 received
// every rank's g+1 data, so each destination had already started call g+1 and
// (stream order) finished every read of call g on the same parity. A reader
// clears each consumed line to zero, so a slot holds either zero or the line of
// its pending call; generation wrap cannot match stale data. Generations skip 0.
//
// Requirement: on each rank, calls on one communicator are stream-ordered (each
// starts after the previous completes). Consecutive eager calls, or calls in the
// same capture, on different streams are joined here with an event.
#include <torch/extension.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kMaxWorld = 8;
constexpr int kThreads = 256;
constexpr int kSplit = 4;  // relayed lines are assigned to relays by line index % kSplit

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
    shared_generation = next == 0u ? 1u : next;
  }
  __syncthreads();
  const uint32_t generation = shared_generation;
  const long long parity_base = (long long)(generation & 1u) * p.world;
  // Row order[0] == rank publishes this rank's lines; every other row receives
  // (and relays) one source's lines. Rows are ordered sender, NVLink sources,
  // relayed sources: a row waits only on earlier classes of rows elsewhere, so
  // in-order block dispatch progresses even when the grid is not resident.
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

void check(cudaError_t status, const char* what) {
  TORCH_CHECK(status == cudaSuccess, "small all-gather ", what, ": ", cudaGetErrorString(status));
}

class SmallGather {
 public:
  SmallGather(int64_t rank, int64_t world, int64_t capacity_bytes) {
    TORCH_CHECK(world >= 2 && world <= kMaxWorld && rank >= 0 && rank < world, "small all-gather requires 2..8 ranks");
    TORCH_CHECK(capacity_bytes > 0 && capacity_bytes % 8 == 0 && capacity_bytes <= (1ll << 30),
                "small all-gather capacity must be a positive multiple of 8 bytes up to 1 GiB");
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
    row_blocks_ = std::max(1, per_sm * sms / world_);  // whole grid resident on an otherwise idle device
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

  // Ends spinning kernels; their outputs are undefined. Teardown only.
  void abort() {
    if (abort_host_ != nullptr) *reinterpret_cast<volatile int*>(abort_host_) = 1;
  }

  // Callers fence the device first. No rank reads a peer mailbox, and every
  // write into this mailbox was consumed by a completed call here, so no
  // cross-process barrier is needed before release.
  void release() {
    if (released_) return;
    released_ = true;
    cudaError_t first = cudaSuccess;
    auto keep = [&](cudaError_t status) { if (first == cudaSuccess && status != cudaSuccess) first = status; };
    for (void* mapped : opened_) keep(cudaIpcCloseMemHandle(mapped));
    opened_.clear();
    if (tail_ != nullptr) keep(cudaEventDestroy(tail_));
    if (own_ != nullptr) keep(cudaFree(own_));
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
  long long row_blocks_ = 1;
  std::vector<void*> opened_;
  cudaEvent_t tail_ = nullptr;
  cudaStream_t tail_stream_ = nullptr;
  unsigned long long tail_capture_ = 0;
  bool has_tail_ = false, connected_ = false, released_ = false;
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
      .def("abort", &SmallGather::abort)
      .def("release", &SmallGather::release);
}
