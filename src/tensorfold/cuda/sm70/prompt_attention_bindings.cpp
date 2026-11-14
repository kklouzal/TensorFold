// Bounded caller-owned buffers for the SM70 tensor-core prompt attention local phases.
#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include "paged_attention_descriptor.cuh"

using T = at::Tensor;

void sm70_prompt_attention_cuda(int, const T&, const T&, const T&, const T&, T&, T&, int, int, const T&);
void sm70_prompt_attention_weight_cuda(const T&, const T&, const T&, const T&, T&, const T&);

namespace {
constexpr int WORLD = 8, HEADS = 12, DIM = 256, WORDS = tf_sm70_paged::WORD_COUNT;

void check(const T& value, const T& owner, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(value.is_cuda() && value.device() == owner.device() && value.scalar_type() == dtype
                && value.is_contiguous() && !value.is_neg() && !value.is_conj(),
                name, " needs resolved contiguous same-device storage with declared dtype");
}

void sm70(const T& owner) {
    const auto* properties = at::cuda::getDeviceProperties(owner.get_device());
    TORCH_CHECK(properties->major == 7 && properties->minor == 0, "tensor-core prompt attention requires SM70");
}

int rows_of(const T& queries) {
    check(queries, queries, at::kHalf, "gathered queries");
    sm70(queries);
    TORCH_CHECK(queries.dim() == 4 && queries.size(0) == WORLD && queries.size(1) <= 65535
                && queries.size(2) == 3 && queries.size(3) == DIM, "queries require [8,rows<=65535,3,256]");
    return int(queries.size(1));
}

void controls(const T& table, const T& owner, int rows, int group, int page_tokens, const T& layer) {
    check(table, owner, at::kLong, "owner row descriptors");
    TORCH_CHECK(table.dim() == 2 && table.size(0) == rows && table.size(1) == WORDS,
                "one 16-word descriptor per query row required");
    TORCH_CHECK(group == 0 || group == 4, "KV-head group starts at rank zero or four");
    TORCH_CHECK(page_tokens > 0 && page_tokens % 4 == 0, "history pages require positive complete pooling groups");
    check(layer, owner, at::kInt, "current layer");
    TORCH_CHECK(layer.dim() == 1 && layer.numel() == 1, "current layer requires a stable int32[1] owner");
}

void selections(const T& selected, const T& owner, int rows) {
    check(selected, owner, at::kInt, "canonical selections");
    TORCH_CHECK(selected.dim() == 3 && selected.size(0) == WORLD && selected.size(1) == rows
                && selected.size(2) >= 2 && selected.size(2) <= INT32_MAX,
                "canonical selections require [8,rows,id_width+2]");
}

void statistics(const T& value, const T& owner, int rows) {
    check(value, owner, at::kFloat, "softmax statistics");
    TORCH_CHECK(value.dim() == 3 && value.size(0) == rows && value.size(1) == HEADS && value.size(2) == 2,
                "statistics require [rows,12,2]");
}

void values(const T& value, const T& owner, int rows, const char* name) {
    check(value, owner, at::kFloat, name);
    TORCH_CHECK(value.dim() == 3 && value.size(0) == rows && value.size(1) == HEADS && value.size(2) == DIM,
                name, " require [rows,12,256]");
}

void q8(int bits, int codec) {
    TORCH_CHECK(bits == 8 && codec == 0, "tensor-core prompt attention supports only ordinary Q8 (bits 8, codec 0)");
}

void distinct(const T& output, std::initializer_list<const T*> inputs) {
    for (const auto* input : inputs) at::assert_no_overlap(output, *input);
}
}  // namespace

void local_statistics(const T& queries, const T& selected, const T& table, T output,
                      int group, int page_tokens, const T& layer, int bits, int codec) {
    const int rows = rows_of(queries);
    controls(table, queries, rows, group, page_tokens, layer);
    selections(selected, queries, rows);
    statistics(output, queries, rows);
    q8(bits, codec);
    distinct(output, {&queries, &selected, &table, &layer});
    c10::cuda::CUDAGuard guard(queries.device());
    T unused;
    sm70_prompt_attention_cuda(0, queries, selected, table, unused, output, unused, group, page_tokens, layer);
}

void partial_values(const T& queries, const T& selected, const T& table, const T& global, T output,
                    int group, int page_tokens, const T& layer, int bits, int codec) {
    const int rows = rows_of(queries);
    controls(table, queries, rows, group, page_tokens, layer);
    selections(selected, queries, rows);
    statistics(global, queries, rows);
    values(output, queries, rows, "partial values");
    q8(bits, codec);
    distinct(output, {&queries, &selected, &table, &global, &layer});
    c10::cuda::CUDAGuard guard(queries.device());
    T unused;
    sm70_prompt_attention_cuda(1, queries, selected, table, global, unused, output, group, page_tokens, layer);
}

void fused_local(const T& queries, const T& selected, const T& table, T stats, T unnormalized,
                 int group, int page_tokens, const T& layer, int bits, int codec) {
    const int rows = rows_of(queries);
    controls(table, queries, rows, group, page_tokens, layer);
    selections(selected, queries, rows);
    statistics(stats, queries, rows);
    values(unnormalized, queries, rows, "unnormalized local values");
    q8(bits, codec);
    distinct(stats, {&queries, &selected, &table, &layer, &unnormalized});
    distinct(unnormalized, {&queries, &selected, &table, &layer});
    c10::cuda::CUDAGuard guard(queries.device());
    T unused;
    sm70_prompt_attention_cuda(2, queries, selected, table, unused, stats, unnormalized, group, page_tokens, layer);
}

void weight_values(const T& unnormalized, const T& local, const T& global, const T& table, T output, const T& layer) {
    check(unnormalized, unnormalized, at::kFloat, "unnormalized local values");
    sm70(unnormalized);
    TORCH_CHECK(unnormalized.dim() == 3 && unnormalized.size(0) <= 65535, "unnormalized local values geometry");
    const int rows = int(unnormalized.size(0));
    values(unnormalized, unnormalized, rows, "unnormalized local values");
    controls(table, unnormalized, rows, 0, 4, layer);
    statistics(local, unnormalized, rows);
    statistics(global, unnormalized, rows);
    values(output, unnormalized, rows, "weighted local values");
    // Elementwise: exact in-place (output == unnormalized) is permitted; partial overlap is not.
    if (!output.is_same(unnormalized)) distinct(output, {&unnormalized});
    distinct(output, {&local, &global, &table, &layer});
    c10::cuda::CUDAGuard guard(unnormalized.device());
    sm70_prompt_attention_weight_cuda(unnormalized, local, global, table, output, layer);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("family") = "sm70-paged-owner-tc884-q8-exact-code-v1";
    module.attr("max_selected") = 2051;
    module.attr("log2e_bits") = uint32_t(0x3fb8aa3b);
    module.attr("descriptor_abi") = tf_sm70_paged::ABI_VERSION;
    module.attr("descriptor_words") = int(tf_sm70_paged::WORD_COUNT);
    module.attr("descriptor_schema_id") = tf_sm70_paged::SCHEMA_ID;
    module.def("local_statistics", &local_statistics);
    module.def("partial_values", &partial_values);
    module.def("fused_local", &fused_local);
    module.def("weight_values", &weight_values);
}
