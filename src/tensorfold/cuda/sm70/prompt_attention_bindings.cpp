// Bounded caller-owned buffers for the SM70 tensor-core prompt attention fused local phase and weighting.
#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include "paged_attention_host.cuh"

using T = at::Tensor;

void sm70_prompt_attention_fused_cuda(const T&, const T&, const T&, T&, T&, int, int, const T&, int);
void sm70_prompt_attention_weight_cuda(const T&, const T&, const T&, const T&, T&, const T&);

namespace {
constexpr int WORLD = 8, HEADS = 12, DIM = 256;
using tf_sm70_paged_host::check;
using tf_sm70_paged_host::controls;
using tf_sm70_paged_host::selections;

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

// Ordinary Q8 (8, 0) and the owned RotorQuant kernel codecs: eight-bit (8, 8/12) and four-bit (4, 1..5/9/10).
void kv_format(int bits, int codec) {
    TORCH_CHECK((bits == 8 && (codec == 0 || codec == 8 || codec == 12))
                || (bits == 4 && ((codec >= 1 && codec <= 6) || codec == 9 || codec == 10)),
                "tensor-core prompt attention supports ordinary Q8 or an owned RotorQuant kernel codec");
}

void distinct(const T& output, std::initializer_list<const T*> inputs) {
    for (const auto* input : inputs) at::assert_no_overlap(output, *input);
}
}  // namespace

void fused_local(const T& queries, const T& selected, const T& table, T stats, T unnormalized,
                 int group, int page_tokens, const T& layer, int bits, int codec) {
    const int rows = rows_of(queries);
    controls(table, queries, rows, group, page_tokens, layer);
    selections(selected, queries, rows);
    statistics(stats, queries, rows);
    values(unnormalized, queries, rows, "unnormalized local values");
    kv_format(bits, codec);
    distinct(stats, {&queries, &selected, &table, &layer, &unnormalized});
    distinct(unnormalized, {&queries, &selected, &table, &layer});
    c10::cuda::CUDAGuard guard(queries.device());
    sm70_prompt_attention_fused_cuda(queries, selected, table, stats, unnormalized, group, page_tokens, layer, codec);
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
    module.def("fused_local", &fused_local);
    module.def("weight_values", &weight_values);
}
