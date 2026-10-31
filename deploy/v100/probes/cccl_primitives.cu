// Required-primitives integration gate, not an upstream CUB replacement suite.
#include <cub/cub.cuh>
#include <thrust/version.h>
#include <cuda/std/array>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static_assert(CCCL_VERSION == 3005000 && CUB_VERSION == 300500 && THRUST_VERSION == 300500,
              "this gate must compile against the complete selected CCCL3.5.0 headers");

#include "cuda_probe_resources.cuh"

__global__ void block_primitives(int* output) {
  using Scan = cub::BlockScan<int, 128>;
  using Reduce = cub::BlockReduce<int, 128>;
  using Warp = cub::WarpReduce<int, 32>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ typename Reduce::TempStorage reduce;
  __shared__ typename Warp::TempStorage warp[4];
  cuda::std::array<int, 1> value{{int(threadIdx.x % 7) - 3}};
  int prefix = 0;
  Scan(scan).ExclusiveSum(value[0], prefix);
  output[threadIdx.x] = prefix;
  int total = Reduce(reduce).Sum(value[0]);
  int local = Warp(warp[threadIdx.x / 32]).Sum(value[0]);
  if (threadIdx.x % 32 == 0) output[128 + threadIdx.x / 32] = local;
  if (threadIdx.x == 0) output[132] = total;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--metadata-only") == 0) {
    std::printf("{\"schema\":\"tensorfold.cccl-required-primitives\",\"version\":1,\"cccl\":%d,\"cub\":%d,\"thrust\":%d,\"cuda_headers\":%d,\"device_execution\":\"unrun\"}\n",
                CCCL_VERSION, CUB_VERSION, THRUST_VERSION, CUDART_VERSION);
    return 0;
  }
  Resources resources;
  char primary[1024]{};
  size_t scratch_bytes = 0;
  int n = 0;
  // Every asynchronous host borrower outlives catch and resource retirement.
  std::vector<int> input, flags, keys, values, scan, compact, sorted_keys, sorted_values;
  std::array<int,133> block{};
  int count=0, total=0;
  try {
    if (argc != 3 || std::strcmp(argv[1], "--run") != 0)
      throw std::runtime_error("use --metadata-only or --run ELEMENTS (1..1048576)");
    size_t consumed = 0;
    n = std::stoi(argv[2], &consumed);
    if (consumed != std::strlen(argv[2]) || n < 1 || n > 1048576)
      throw std::runtime_error("probe elements must be an integer in1..1048576");
    int devices = 0;
    cudaError_t available = cudaGetDeviceCount(&devices);
    if (available == cudaErrorInsufficientDriver || available == cudaErrorNoDevice || (available == cudaSuccess && !devices)) {
      std::puts("{\"device_execution\":\"unrun\",\"reason\":\"V100 device unavailable\"}");
      return 77;
    }
    check(available, "cudaGetDeviceCount");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    if (properties.major != 7 || properties.minor != 0) throw std::runtime_error("this device gate requires actual SM70");
    for(auto pointer:{&input,&flags,&keys,&values,&scan,&compact,&sorted_keys,&sorted_values}) pointer->resize(n);
    std::vector<std::pair<int,int>> expected_sort(n);
    std::vector<int> expected_compact;
    for (int i = 0; i < n; ++i) {
      input[i] = i % 17 - 8; flags[i] = i % 3 != 0; keys[i] = i % 29 - 14; values[i] = i;
      expected_sort[i] = {keys[i], i};
      if (flags[i]) expected_compact.push_back(input[i]);
    }
    std::stable_sort(expected_sort.begin(), expected_sort.end(), [](auto a, auto b) { return a.first < b.first; });
    auto alloc = [&]() { return static_cast<int*>(resources.allocate(size_t(n) * sizeof(int))); };
    int *di = alloc(), *df = alloc(), *dk = alloc(), *dv = alloc(), *ds = alloc(), *dc = alloc(),
        *dko = alloc(), *dvo = alloc();
    int *dcount = static_cast<int*>(resources.allocate(sizeof(int))),
        *dsum = static_cast<int*>(resources.allocate(sizeof(int))),
        *db = static_cast<int*>(resources.allocate(133 * sizeof(int)));
    resources.create_stream();
    auto stream = resources.stream;
    for (auto pair : {std::make_pair(di, input.data()), std::make_pair(df, flags.data()),
                      std::make_pair(dk, keys.data()), std::make_pair(dv, values.data())})
      check(cudaMemcpyAsync(pair.first, pair.second, size_t(n)*sizeof(int), cudaMemcpyHostToDevice, stream), "input copy");
    size_t scan_bytes=0, sort_bytes=0, select_bytes=0, reduce_bytes=0;
    check(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, di, ds, n, stream), "scan storage query");
    check(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, dk, dko, dv, dvo, n, 0, 32, stream), "sort storage query");
    check(cub::DeviceSelect::Flagged(nullptr, select_bytes, di, df, dc, dcount, n, stream), "compact storage query");
    check(cub::DeviceReduce::Sum(nullptr, reduce_bytes, di, dsum, n, stream), "reduce storage query");
    scratch_bytes = std::max({scan_bytes, sort_bytes, select_bytes, reduce_bytes});
    void* scratch = resources.allocate(scratch_bytes);
    auto run = [&]() {
      check(cub::DeviceScan::ExclusiveSum(scratch, scan_bytes, di, ds, n, stream), "scan");
      check(cub::DeviceRadixSort::SortPairs(scratch, sort_bytes, dk, dko, dv, dvo, n, 0, 32, stream), "sort");
      check(cub::DeviceSelect::Flagged(scratch, select_bytes, di, df, dc, dcount, n, stream), "compact");
      check(cub::DeviceReduce::Sum(scratch, reduce_bytes, di, dsum, n, stream), "reduce");
      block_primitives<<<1,128,0,stream>>>(db);
      check(cudaGetLastError(), "warp/block primitives");
    };
    for (int mode = 0; mode < 2; ++mode) {
      // 0x5a5a5a5a exceeds every proved integer result range. Poison outside
      // capture so a graph or kernel that writes nothing cannot reuse eager data.
      for(auto pointer:{ds,dc,dko,dvo}) check(cudaMemsetAsync(pointer,0x5a,size_t(n)*4,stream),"poison outputs");
      for(auto pointer:{dcount,dsum}) check(cudaMemsetAsync(pointer,0x5a,4,stream),"poison scalar outputs");
      check(cudaMemsetAsync(db,0x5a,133*4,stream),"poison block outputs");
      if (!mode) { run(); }
      else {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture begin");
        run();
        resources.end_capture();
        resources.instantiate();
        check(cudaGraphLaunch(resources.replay, stream), "graph replay");
      }
      count=0; total=0; block.fill(0);
      auto copy = [&](void* out, const void* in, size_t bytes) {
        if (bytes) check(cudaMemcpyAsync(out, in, bytes, cudaMemcpyDeviceToHost, stream), "output copy");
      };
      copy(scan.data(), ds, size_t(n)*4);
      copy(sorted_keys.data(), dko, size_t(n)*4); copy(sorted_values.data(), dvo, size_t(n)*4);
      copy(&count, dcount, 4); copy(&total, dsum, 4); copy(block.data(), db, block.size()*4);
      check(cudaStreamSynchronize(stream), "execution completion");
      if (count != int(expected_compact.size())) throw std::runtime_error("compaction count mismatch");
      // DeviceSelect initializes only its retained prefix. Never transfer the
      // unused tail merely because the output allocation has full capacity.
      copy(compact.data(), dc, size_t(count)*4);
      check(cudaStreamSynchronize(stream), "compaction prefix completion");
      int prefix=0;
      for (int i=0; i<n; ++i) {
        if (scan[i] != prefix || sorted_keys[i] != expected_sort[i].first || sorted_values[i] != expected_sort[i].second)
          throw std::runtime_error("scan/sort independent oracle mismatch");
        prefix += input[i];
      }
      if (total != prefix ||
          !std::equal(expected_compact.begin(), expected_compact.end(), compact.begin()))
        throw std::runtime_error("reduce/compaction independent oracle mismatch");
      prefix=0;
      for (int i=0; i<128; ++i) {
        if (block[i] != prefix) throw std::runtime_error("block scan mismatch");
        prefix += i%7-3;
      }
      if (block[132] != prefix) throw std::runtime_error("block reduce mismatch");
      for (int warp=0; warp<4; ++warp) {
        int sum=0; for (int lane=0; lane<32; ++lane) sum += (32*warp+lane)%7-3;
        if (block[128+warp] != sum) throw std::runtime_error("warp reduction mismatch");
      }
    }
  } catch (const std::exception& error) { std::snprintf(primary, sizeof(primary), "%s", error.what()); }
  if (int outcome=probe_outcome(resources.cleanup(),primary)) return outcome;
  std::printf("{\"elements\":%d,\"temporary_bytes\":%zu,\"scratch_allocation_bytes\":%zu,\"device_execution\":\"passed\",\"eager_and_graph\":true}\n",
              n, scratch_bytes, std::max<size_t>(scratch_bytes,1));
  return 0;
}
