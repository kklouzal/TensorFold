#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <limits>
#include "tensor_contracts.cuh"
#include "qmm_policy.cuh"
#include "kernel_configuration.cuh"

// Kernel dimensions and row leading dimensions are signed int. Check them
// before padding arithmetic, allocations or narrowing at the launch boundary.
static void check_dimensions(const at::Tensor& x, int64_t n) {
    constexpr int64_t maximum = std::numeric_limits<int>::max();
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= maximum - 127 &&
                x.size(1) >= 1 && x.size(1) <= maximum && n >= 1 && n <= maximum - 127,
                "positive M, K and n must fit the kernel's signed-int dimensions and 128-row/column padding");
    TORCH_CHECK(x.size(0) == 1 || x.stride(0) <= maximum, "x row stride exceeds signed-int addressing");
}

static void check_devices(const at::Tensor& x, std::initializer_list<const at::Tensor*> tensors) {
    for (const auto* tensor : tensors)
        TORCH_CHECK(tensor->device() == x.device(), "all lane-matmul tensors must be on the same CUDA device");
}

static void check_alignment(const at::Tensor& tensor, uintptr_t alignment, const char* message) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0, message);
}

// cp16 stages weight bytes and complete scale/bias rows. One-row views need
// no leading-dimension alignment because their unused stride is never read.
static void check_packed_alignment(const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases) {
    check_alignment(w, 16, "packed weight requires 16-byte staging alignment");
    check_alignment(scales, 16, "scales require 16-byte staging alignment");
    check_alignment(biases, 16, "biases require 16-byte staging alignment");
    TORCH_CHECK(scales.size(0) <= 1 || scales.stride(0) % 8 == 0, "scale/bias rows require 16-byte staging alignment");
}

// Logical disjointness is the normal O(1) path. Only an actual logical overlap
// needs the selected kernel's packed prefix or scale/bias column projection.
static void check_packed_reads(const at::Tensor& write, const at::Tensor& w, const at::Tensor& scales,
                               const at::Tensor& biases, int64_t columns, int64_t k, int64_t elements = -1) {
    TORCH_CHECK(!tensorfold::overlaps(write, w, elements) ||
                !tensorfold::overlaps(write, w, elements, columns * k / 8), "writes must not overlap loaded packed weights");
    for (const auto* tensor : {&scales, &biases})
        TORCH_CHECK(!tensorfold::overlaps(write, *tensor, elements) ||
                    !tensorfold::overlaps(write, *tensor, elements, -1, columns), "writes must not overlap loaded scale/bias columns");
}

bool qmm_clusters(int, bool);
void qmm_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
              at::Tensor&, const at::Tensor&, int, int, int, int, bool, bool);
void qmm_group_cuda(const at::Tensor&, const at::Tensor&, const std::vector<at::Tensor>&,
                    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, std::vector<at::Tensor>&,
                    const std::vector<int64_t>&, const std::vector<int64_t>&, bool, int, int);
void qmm_prefill_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int,
                      int, bool, int);
void qmm_prefill8w_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int,
                        int, bool, int, bool);
void qmm_prefill8_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       const at::Tensor&, at::Tensor&, int, int, bool, int);

// x (M, K) bf16 times a packed 4-bit weight; ``reduce`` false leaves the K slices unadded in ``part`` (SK, M, n).
void qmm(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& scales,
         const at::Tensor& biases, at::Tensor& out, const c10::optional<at::Tensor>& part, int64_t n, int64_t sk,
         int64_t gs, int64_t bm, bool f32, bool reduce) {
    TORCH_CHECK(gs == 32 || gs == 64, "groups of 32 or 64");
    TORCH_CHECK(bm == 16 || bm == 32 || bm == 64, "row tile 16, 32 or 64");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) >= 1 &&
                x.stride(1) == 1 && x.stride(0) >= x.size(1), "x: (M, K) bf16 with contiguous rows");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.size(0) == 1 || x.stride(0) % 8 == 0),
                "x rows must start on 16-byte boundaries");
    check_dimensions(x, n);
    check_devices(x, {&xs, &w, &scales, &biases, &out});
    const int64_t m = x.size(0), k = x.size(1), kg = k / gs, npad = (n + 127) / 128 * 128;
    TORCH_CHECK(k % gs == 0 && sk >= 1 && sk <= kg && kg % sk == 0, "K splits into whole groups a slice");
    TORCH_CHECK(xs.is_cuda() && xs.is_contiguous() && xs.scalar_type() == at::kFloat && xs.size(0) == m &&
                xs.size(1) == kg, "xs: (M, K / gs) fp32");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kInt && w.numel() == npad * k / 8,
                "packed weight does not match n and K");
    TORCH_CHECK(scales.stride(1) == 1 && biases.stride(1) == 1 && scales.stride(0) >= npad &&
                scales.stride(0) <= std::numeric_limits<int>::max() &&
                biases.stride(0) == scales.stride(0) && scales.scalar_type() == at::kBFloat16 &&
                biases.scalar_type() == at::kBFloat16 && scales.size(0) == kg && scales.size(1) == npad &&
                biases.sizes() == scales.sizes(), "scales and biases: (K / gs, n padded to 128) bf16, rows may be strided");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    check_packed_alignment(w, scales, biases);
    c10::cuda::CUDAGuard guard(x.device());
    const auto* properties = at::cuda::getDeviceProperties(x.get_device());
    tensorfold::check_grid((m + bm - 1) / bm * ((n + 63) / 64), static_cast<int>(sk), properties);
    at::Tensor p;
    const bool staged = sk > 1 && !qmm_clusters(static_cast<int>(sk), reduce);
    if (staged && reduce) tensorfold::check_grid((m * n + 255) / 256, 1, properties);
    const int64_t columns = (n + 63) / 64 * 64;
    if (staged) {
        TORCH_CHECK(m <= std::numeric_limits<int64_t>::max() / sk / n,
                    "K-slice scratch dimensions exceed signed-int64 storage");
        p = part.has_value() ? *part : at::empty({sk, m, n}, x.options().dtype(at::kFloat));
        check_devices(x, {&p});
        TORCH_CHECK(p.is_cuda() && p.is_contiguous() && p.scalar_type() == at::kFloat && p.numel() >= sk * m * n,
                    "part: at least (SK, M, n) fp32");
        tensorfold::check_disjoint(p, {&x, &xs},
                                  "written K-slice scratch must not overlap inputs", sk * m * n);
        check_packed_reads(p, w, scales, biases, columns, k, sk * m * n);
        if (reduce)
            TORCH_CHECK(!tensorfold::overlaps(p, out, sk * m * n), "reduction output must not overlap K-slice scratch");
    }
    // The staged producer reads every input before the separate reducer writes
    // out. That path permits output/input aliasing; reduce=false never writes out.
    if (!staged) {
        if (!f32 && n % 2 == 0) check_alignment(out, 4, "paired bf16 output stores require 4-byte alignment");
        tensorfold::check_disjoint(out, {&x, &xs}, "written output must not overlap inputs");
        check_packed_reads(out, w, scales, biases, columns, k);
    }
    qmm_cuda(x, xs, w, scales, biases, out, p, static_cast<int>(n), static_cast<int>(sk), static_cast<int>(gs),
             static_cast<int>(bm), f32, reduce);
}

// Up to four packed 4-bit weights against one x in one sm_12x launch, each with its own K split (its own bits).
void qmm_group(const at::Tensor& x, const at::Tensor& xs, const std::vector<at::Tensor>& ws,
               const std::vector<at::Tensor>& scales, const std::vector<at::Tensor>& biases,
               std::vector<at::Tensor> outs, const std::vector<int64_t>& ns, const std::vector<int64_t>& sks, bool f32,
               int64_t tile, int64_t pdl) {
    const size_t parts = ws.size();
    TORCH_CHECK(parts >= 1 && parts <= 4 && scales.size() == parts && biases.size() == parts && outs.size() == parts &&
                ns.size() == parts && sks.size() == parts, "one to four parts, each with weights, scales, biases, out");
    TORCH_CHECK(tile >= 0 && tile <= 12, "tile 0-12");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) >= 1 &&
                x.stride(1) == 1 && x.stride(0) >= x.size(1), "x: (M, K) bf16 with contiguous rows");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.size(0) == 1 || x.stride(0) % 8 == 0),
                "x rows must start on 16-byte boundaries");
    check_dimensions(x, ns[0]);
    const int64_t m = x.size(0), k = x.size(1), kg = k / 64;
    const auto* properties = at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(properties->major == 12, "grouped QMM requires an sm_12x CUDA device");
    const int selected = tensorfold::resolve_qmm_group_tile(static_cast<int>(tile), static_cast<int>(m),
                                                           properties->major == 12 && properties->minor == 1);
    const auto geometry = tensorfold::qmm_group_tile(selected);
    int c = 1;
    for (const auto sk : sks) {
        TORCH_CHECK(sk == 1 || sk == 2 || sk == 4 || sk == 8, "K split 1, 2, 4 or 8");
        c = std::max(c, static_cast<int>(sk));
    }
    int64_t clusters = 0;
    TORCH_CHECK(k % 64 == 0 && xs.is_cuda() && xs.is_contiguous() && xs.scalar_type() == at::kFloat &&
                xs.size(0) == m && xs.size(1) == kg, "groups of 64; xs: (M, K / 64) fp32");
    for (size_t i = 0; i < parts; ++i) {
        check_dimensions(x, ns[i]);
        check_devices(x, {&xs, &ws[i], &scales[i], &biases[i], &outs[i]});
        const int64_t n = ns[i], npad = (n + 127) / 128 * 128, sk = sks[i];
        TORCH_CHECK(sk == 1 || sk == 2 || sk == 4 || sk == 8, "K split 1, 2, 4 or 8");
        TORCH_CHECK(kg % sk == 0, "K splits into whole groups a slice");
        TORCH_CHECK(ws[i].is_cuda() && ws[i].is_contiguous() && ws[i].scalar_type() == at::kInt &&
                    ws[i].numel() == npad * k / 8, "packed weight does not match n and K");
        TORCH_CHECK(scales[i].stride(1) == 1 && biases[i].stride(1) == 1 && scales[i].stride(0) >= npad &&
                    scales[i].stride(0) <= std::numeric_limits<int>::max() &&
                    biases[i].stride(0) == scales[i].stride(0) && scales[i].scalar_type() == at::kBFloat16 &&
                    biases[i].scalar_type() == at::kBFloat16 && scales[i].size(0) == kg && scales[i].size(1) == npad &&
                    biases[i].sizes() == scales[i].sizes(), "scales and biases: (K / 64, n padded to 128) bf16");
        TORCH_CHECK(outs[i].is_cuda() && outs[i].is_contiguous() && outs[i].size(0) == m && outs[i].size(1) == n &&
                    outs[i].scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
        check_packed_alignment(ws[i], scales[i], biases[i]);
        if (!f32 && n % 2 == 0 && geometry.pairs)
            check_alignment(outs[i], 4, "paired bf16 grouped output stores require 4-byte alignment");
        const int64_t tiles = (n + geometry.columns - 1) / geometry.columns;
        clusters += (m + geometry.rows - 1) / geometry.rows * ((tiles + c / sk - 1) / (c / sk));
    }
    tensorfold::check_grid(clusters * c, 1, properties);
    for (size_t i = 0; i < parts; ++i) {
        tensorfold::check_disjoint(outs[i], {&x, &xs}, "grouped outputs must not overlap inputs");
        for (size_t j = 0; j < parts; ++j) {
            const int64_t bn = geometry.columns;
            check_packed_reads(outs[i], ws[j], scales[j], biases[j], (ns[j] + bn - 1) / bn * bn, k);
            if (j < i) TORCH_CHECK(!tensorfold::overlaps(outs[i], outs[j]), "grouped outputs must not overlap");
        }
    }
    c10::cuda::CUDAGuard guard(x.device());
    const int early = pdl < 0 ? -1 : pdl > 0 ? 1 : 0;
    qmm_group_cuda(x, xs, ws, scales, biases, outs, ns, sks, f32, selected, early);
}

// Prefill: x (M, K) bf16 times a packed 4-bit weight with each weight rounded once to bf16, one fp32 chain over K.
void qmm_prefill(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
                 at::Tensor& out, int64_t n, int64_t gs, bool f32, int64_t tile) {
    TORCH_CHECK(gs == 32 || gs == 64, "groups of 32 or 64");
    TORCH_CHECK(tile >= 0 && tile <= 11, "tile 0-11");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) >= 1 &&
                x.stride(1) == 1 && x.stride(0) >= x.size(1), "x: (M, K) bf16 with contiguous rows");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.size(0) == 1 || x.stride(0) % 8 == 0),
                "x rows must start on 16-byte boundaries");
    check_dimensions(x, n);
    const auto geometry = tensorfold::qmm_prefill_tile(static_cast<int>(tile));
    const int64_t bn = geometry.columns;
    TORCH_CHECK(n <= std::numeric_limits<int>::max() - (bn - 1), "prefill n padding exceeds signed-int column dimensions");
    check_devices(x, {&w, &scales, &biases, &out});
    const int64_t m = x.size(0), k = x.size(1), kg = k / gs, npad = (n + 127) / 128 * 128;
    TORCH_CHECK(k % gs == 0, "K splits into whole groups");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kInt && w.numel() == npad * k / 8,
                "packed weight does not match n and K");
    TORCH_CHECK(scales.is_contiguous() && biases.is_contiguous() && scales.scalar_type() == at::kBFloat16 &&
                biases.scalar_type() == at::kBFloat16 && scales.size(0) == kg && scales.size(1) == npad &&
                biases.sizes() == scales.sizes(), "scales and biases: (K / gs, n padded to 128) bf16");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    check_packed_alignment(w, scales, biases);
    tensorfold::check_grid((m + geometry.rows - 1) / geometry.rows * ((n + bn - 1) / bn), 1,
                          at::cuda::getDeviceProperties(x.get_device()));
    if (!f32 && n % 2 == 0) check_alignment(out, 4, "paired bf16 prefill stores require 4-byte alignment");
    tensorfold::check_disjoint(out, {&x}, "prefill output must not overlap inputs");
    const int64_t columns = std::min(npad, (n + bn - 1) / bn * bn);
    check_packed_reads(out, w, scales, biases, columns, k);
    c10::cuda::CUDAGuard guard(x.device());
    qmm_prefill_cuda(x, w, scales, biases, out, static_cast<int>(n), static_cast<int>(gs), f32, static_cast<int>(tile));
}

// Prefill in FP8: x8, xs and a from ``quantize_rows`` (e4m3 bytes, group sums / a, row scales) times a 4-bit weight.
void qmm_prefill8(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& w,
                  const at::Tensor& scales, const at::Tensor& biases, at::Tensor& out, int64_t n, int64_t gs, bool f32,
                  int64_t tile) {
    TORCH_CHECK(gs == 32 || gs == 64, "groups of 32 or 64");
    TORCH_CHECK(tile >= 0 && tile <= 2, "tile 0-2");
    TORCH_CHECK(x8.is_cuda() && x8.scalar_type() == at::kByte && x8.dim() == 2 && x8.is_contiguous() &&
                x8.size(0) >= 1, "x8: (M, K) e4m3 bytes, contiguous");
    check_dimensions(x8, n);
    check_devices(x8, {&xs, &a, &w, &scales, &biases, &out});
    const int64_t m = x8.size(0), k = x8.size(1), kg = k / gs, npad = (n + 127) / 128 * 128;
    TORCH_CHECK(k % gs == 0 && k % 32 == 0, "K splits into whole groups of 32 or 64 inputs");
    TORCH_CHECK(xs.is_cuda() && xs.is_contiguous() && xs.scalar_type() == at::kBFloat16 && xs.size(0) == m &&
                xs.size(1) == kg, "xs: (M, K / gs) bf16");
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == at::kFloat && a.numel() == m, "a: (M,) fp32");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kInt && w.numel() == npad * k / 8,
                "packed weight does not match n and K");
    TORCH_CHECK(scales.is_contiguous() && biases.is_contiguous() && scales.scalar_type() == at::kBFloat16 &&
                biases.scalar_type() == at::kBFloat16 && scales.size(0) == kg && scales.size(1) == npad &&
                biases.sizes() == scales.sizes(), "scales and biases: (K / gs, n padded to 128) bf16");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    check_alignment(x8, 16, "FP8 input requires 16-byte staging alignment");
    check_alignment(w, 16, "packed weight requires 16-byte staging alignment");
    check_alignment(scales, 16, "scales require 16-byte staging alignment");
    const int64_t bm = tile == 1 ? 64 : 128;
    tensorfold::check_grid((m + bm - 1) / bm * ((n + 127) / 128), 1,
                          at::cuda::getDeviceProperties(x8.get_device()));
    if (!f32 && n % 2 == 0) check_alignment(out, 4, "paired bf16 FP8 prefill stores require 4-byte alignment");
    tensorfold::check_disjoint(out, {&x8, &xs, &a, &w, &scales, &biases}, "FP8 prefill output must not overlap inputs");
    c10::cuda::CUDAGuard guard(x8.device());
    qmm_prefill8_cuda(x8, xs, a, w, scales, biases, out, static_cast<int>(n), static_cast<int>(gs), f32,
                      static_cast<int>(tile));
}

// Prefill in FP8 over e4m3 weight bytes (fragment order): x8 and a from ``quantize_rows``, a bf16 scale per (gs, column).
void qmm_prefill8w(const at::Tensor& x8, const at::Tensor& a, const at::Tensor& w8, const at::Tensor& scales,
                   at::Tensor& out, int64_t n, int64_t gs, bool f32, int64_t tile, bool l64) {
    TORCH_CHECK(tile >= 0 && tile <= 2, "tile 0-2");
    TORCH_CHECK(gs == 32 || gs == 64, "groups of 32 or 64 inputs");
    TORCH_CHECK(!l64 || gs == 32, "the 64-input byte order serves groups of 32");
    TORCH_CHECK(x8.is_cuda() && x8.scalar_type() == at::kByte && x8.dim() == 2 && x8.is_contiguous() &&
                x8.size(0) >= 1, "x8: (M, K) e4m3 bytes, contiguous");
    check_dimensions(x8, n);
    check_devices(x8, {&a, &w8, &scales, &out});
    const int64_t m = x8.size(0), k = x8.size(1);
    TORCH_CHECK(k % 64 == 0 && n % 128 == 0, "K a multiple of 64 and n of 128");
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == at::kFloat && a.numel() == m, "a: (M,) fp32");
    TORCH_CHECK(w8.is_cuda() && w8.is_contiguous() && w8.scalar_type() == at::kByte && w8.numel() == n * k,
                "staged weight: n * K e4m3 bytes");
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.scalar_type() == at::kBFloat16 &&
                scales.size(0) == k / gs && scales.size(1) == n, "scales: (K / gs, n) bf16");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    check_alignment(x8, 16, "FP8 input requires 16-byte staging alignment");
    check_alignment(w8, l64 ? 8 : 16, "FP8 weight must align to its 8/16-byte staging copy");
    check_alignment(scales, 16, "scales require 16-byte staging alignment");
    const int64_t bm = tile == 1 ? 64 : 128;
    tensorfold::check_grid((m + bm - 1) / bm * ((n + 127) / 128), 1,
                          at::cuda::getDeviceProperties(x8.get_device()));
    if (!f32) check_alignment(out, 4, "paired bf16 FP8 weight-prefill stores require 4-byte alignment");
    tensorfold::check_disjoint(out, {&x8, &a, &w8, &scales}, "FP8 weight-prefill output must not overlap inputs");
    c10::cuda::CUDAGuard guard(x8.device());
    qmm_prefill8w_cuda(x8, a, w8, scales, out, static_cast<int>(n), static_cast<int>(gs), f32, static_cast<int>(tile),
                       l64);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qmm", &qmm);
    m.def("qmm_group", &qmm_group);
    m.def("qmm_prefill", &qmm_prefill);
    m.def("qmm_prefill8", &qmm_prefill8);
    m.def("qmm_prefill8w", &qmm_prefill8w);
}
