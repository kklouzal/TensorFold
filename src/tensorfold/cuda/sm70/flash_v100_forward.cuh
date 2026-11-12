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
