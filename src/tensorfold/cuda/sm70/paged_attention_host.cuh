// Shared host admission for paged and tensor-core prompt attention.
#pragma once
#include <torch/extension.h>
#include <cstdint>
#include "paged_attention_descriptor.cuh"

namespace tf_sm70_paged_host {
using T = at::Tensor;
constexpr int WORLD = 8, WORDS = tf_sm70_paged::WORD_COUNT;

inline void check(const T& value, const T& owner, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(value.is_cuda() && value.device() == owner.device() && value.scalar_type() == dtype
                && value.is_contiguous() && !value.is_neg() && !value.is_conj(),
                name, " needs resolved contiguous same-device storage with declared dtype");
}

inline void controls(const T& table, const T& owner, int rows, int group, int page_tokens, const T& layer) {
    check(table, owner, at::kLong, "owner row descriptors");
    TORCH_CHECK(table.dim() == 2 && table.size(0) == rows && table.size(1) == WORDS,
                "one WORD_COUNT-word descriptor per query row required");
    TORCH_CHECK(group == 0 || group == 4, "KV-head group starts at rank zero or four");
    TORCH_CHECK(page_tokens > 0 && page_tokens % 4 == 0, "history pages require positive complete pooling groups");
    check(layer, owner, at::kInt, "current layer");
    TORCH_CHECK(layer.dim() == 1 && layer.numel() == 1, "current layer requires a stable int32[1] owner");
}

inline void selections(const T& selected, const T& owner, int rows) {
    check(selected, owner, at::kInt, "canonical selections");
    TORCH_CHECK(selected.dim() == 3 && selected.size(0) == WORLD && selected.size(1) == rows
                && selected.size(2) >= 2 && selected.size(2) <= INT32_MAX,
                "canonical selections require [8,rows,id_width+2]");
}
}  // namespace tf_sm70_paged_host
