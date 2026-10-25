#pragma once
// Explicitly retired standalone probe resources; cleanup failures remain failures.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

static void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct Resources {
  std::array<void*, 16> pointers{};
  size_t count = 0;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t replay = nullptr;
  void create_stream() {
    cudaStream_t candidate = nullptr;
    check(cudaStreamCreateWithFlags(&candidate, cudaStreamNonBlocking), "create stream");
    stream = candidate;
  }
  void end_capture() {
    cudaGraph_t candidate = nullptr;
    check(cudaStreamEndCapture(stream, &candidate), "capture end");
    graph = candidate;
  }
  void instantiate() {
    cudaGraphExec_t candidate = nullptr;
    check(cudaGraphInstantiate(&candidate, graph, nullptr, nullptr, 0), "graph instantiate");
    replay = candidate;
  }
  void* allocate(size_t bytes) {
    if (count == pointers.size()) throw std::runtime_error("probe allocation inventory exhausted");
    void* pointer = nullptr;
    check(cudaMalloc(&pointer, std::max<size_t>(bytes, 1)), "cudaMalloc");
    pointers[count++] = pointer;
    return pointer;
  }
  static bool retired(cudaError_t status, const char* operation) noexcept {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "cleanup %s: %s\n", operation, cudaGetErrorString(status));
    return false;
  }
  bool complete() noexcept {
    return !stream || retired(cudaStreamSynchronize(stream), "stream completion");
  }
  bool retire_graphs() noexcept {
    if (replay && !retired(cudaGraphExecDestroy(replay), "graph executable")) return false;
    replay = nullptr;
    if (graph && !retired(cudaGraphDestroy(graph), "graph")) return false;
    graph = nullptr;
    return true;
  }
  bool cleanup(bool completed=false) noexcept {
    // Uncertain retirement retains all remaining borrowers until process exit.
    if ((!completed && !complete()) || !retire_graphs()) return false;
    while (count) {
      if (!retired(cudaFree(pointers[count-1]), "device allocation")) return false;
      --count;
    }
    if (stream && !retired(cudaStreamDestroy(stream), "stream")) return false;
    stream = nullptr;
    return true;
  }
};

static int probe_outcome(bool clean, const char* primary) noexcept {
  if (primary[0]) std::fprintf(stderr, "%s\n", primary);
  if (!clean) {
    std::fputs("probe retirement is uncertain; terminal process teardown retains host/native borrowers\n", stderr);
    std::fflush(stderr);
    std::_Exit(1); // Do not destroy host DMA buffers before the owning process retires.
  }
  return primary[0] ? 1 : 0;
}
