#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime_api.h>
struct DenseCutlassPlan;
DenseCutlassPlan* dense_cutlass_prepare(const at::Tensor&, const at::Tensor&, at::Tensor, cudaStream_t);
void dense_cutlass_run(DenseCutlassPlan*, const at::Tensor&, const at::Tensor&, at::Tensor, cudaStream_t);
void dense_cutlass_destroy(DenseCutlassPlan*) noexcept;
