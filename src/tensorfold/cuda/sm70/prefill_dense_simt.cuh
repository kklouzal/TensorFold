#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime_api.h>
#include <cstddef>

struct PrefillSimtPlan;
struct PrefillSimtInfo {
    size_t workspace_bytes;
    size_t shared_storage_bytes;
    int threads;
    cudaFuncAttributes attributes;
};
PrefillSimtPlan* prefill_simt_prepare(const at::Tensor&, const at::Tensor&, at::Tensor,
                                    cudaStream_t, PrefillSimtInfo&);
void prefill_simt_run(PrefillSimtPlan*, cudaStream_t);
void prefill_simt_destroy(PrefillSimtPlan*) noexcept;
