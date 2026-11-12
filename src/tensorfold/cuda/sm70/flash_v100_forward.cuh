#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime_api.h>
struct FlashV100Info {
    int shared_bytes;
    int threads;
    cudaFuncAttributes attributes;
};
void flash_v100_prepare_kernel(FlashV100Info&);
void flash_v100_launch(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      at::Tensor, at::Tensor, float, cudaStream_t);
void flash_v100_prepare_varlen_kernel(FlashV100Info&);
void flash_v100_init_compact(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, cudaStream_t);
void flash_v100_compact(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       at::Tensor, at::Tensor, at::Tensor, cudaStream_t);
void flash_v100_launch_varlen(const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor, at::Tensor, cudaStream_t);
