// MoE routing on the GPU: router logits as a row-invariant split-K BF16 tensor-core GEMV, optionally with the
// compressed-tensors Gemma router preprocessing fused in, and the softmax / top-k / renormalize / per-expert-scale
// selection that Gemma's router applies. A row's bits never depend on the other rows of the call, on the row count or
// on its row tile: every row has the same K partition, the same mma chain per slice and the same slice-sum order.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <torch/extension.h>

#include <type_traits>

namespace {

constexpr int ROWS = 16;        // a CTA's row tile (the mma's M)
constexpr int EXPERTS = 8;      // a CTA's experts (the mma's N)
constexpr int SLICES = 8;       // warps; each owns one contiguous K slice
constexpr int THREADS = SLICES * 32;
// The fragment mapping below is m16n8k16's (rows g and g + 8, expert g), the pre-norm lets warp w reduce rows w and
// w + 8, and the slice sum takes one thread per output.
static_assert(ROWS == 16 && EXPERTS == 8 && ROWS == 2 * SLICES && ROWS * EXPERTS <= THREADS, "route tile geometry");

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Eight consecutive BF16 inputs of one row, as the router consumes them. PRENORM applies the compressed-tensors Gemma
// router input: bf16(x * inv) (the unweighted RMS norm), times the BF16 root scalar, times the learned BF16 scale, each
// product rounded to BF16 as the reference's separate BF16 tensor operations do.
template <bool PRENORM>
__device__ __forceinline__ uint4 inputs(const __nv_bfloat16* x, bool live, float inv, float root,
                                        const __nv_bfloat16* scale, int k) {
    if (!live) return make_uint4(0u, 0u, 0u, 0u);
    uint4 raw = *reinterpret_cast<const uint4*>(x + k);
    if constexpr (PRENORM) {
        const uint4 sc = *reinterpret_cast<const uint4*>(scale + k);
        __nv_bfloat16* v = reinterpret_cast<__nv_bfloat16*>(&raw);
        const __nv_bfloat16* s = reinterpret_cast<const __nv_bfloat16*>(&sc);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const __nv_bfloat16 normed = __float2bfloat16_rn(__fmul_rn(__bfloat162float(v[i]), inv));
            const __nv_bfloat16 rooted = __float2bfloat16_rn(__fmul_rn(__bfloat162float(normed), root));
            v[i] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(rooted), __bfloat162float(s[i])));
        }
    }
    return raw;
}

// Grid (row tiles, expert tiles); warp w owns K slice w: [w * per, (w + 1) * per) of 32-input steps, per a function of
// K alone (trailing warps may own nothing). Lane (g, t) loads 16 bytes of input row g and g + 8 and of expert g at
// k = base + 8t: elements {0,1},{2,3} feed one m16n8k16 and {4,5},{6,7} the next, the same K positions for A and B, so
// each mma sums one fixed set of 16 inputs. Slices add in slice order through shared memory.
template <bool PRENORM>
__global__ void __launch_bounds__(THREADS) logits_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, int rows,
    const __nv_bfloat16* __restrict__ w, int experts, int k, float* __restrict__ out, int64_t ldo, float eps,
    float root, const __nv_bfloat16* __restrict__ scale) {
    __shared__ float inv_rows[ROWS];
    __shared__ float partial[SLICES][ROWS][EXPERTS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int r0 = blockIdx.x * ROWS, e0 = blockIdx.y * EXPERTS;
    if constexpr (PRENORM) {
        // Warp w reduces rows w and w + 8: lane-strided FP32 squares in K order, then a fixed xor tree.
        for (int h = 0; h < 2; ++h) {
            const int r = r0 + warp + 8 * h;
            float ss = 0.f;
            if (r < rows) {
                const __nv_bfloat16* row = x + static_cast<int64_t>(r) * ldx;
                for (int at = lane * 8; at < k; at += 256) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(row + at);
                    const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float f = __bfloat162float(v[i]);
                        ss = __fmaf_rn(f, f, ss);
                    }
                }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
            if (lane == 0) inv_rows[warp + 8 * h] = rsqrtf(__fadd_rn(__fdiv_rn(ss, static_cast<float>(k)), eps));
        }
        __syncthreads();
    }
    const int ra = r0 + g, rb = r0 + g + 8, e = e0 + g;
    const bool la = ra < rows, lb = rb < rows, le = e < experts;
    const float ia = PRENORM ? inv_rows[g] : 0.f, ib = PRENORM ? inv_rows[g + 8] : 0.f;
    const __nv_bfloat16* xa = x + static_cast<int64_t>(la ? ra : 0) * ldx;
    const __nv_bfloat16* xb = x + static_cast<int64_t>(lb ? rb : 0) * ldx;
    const __nv_bfloat16* we = w + static_cast<int64_t>(le ? e : 0) * k;
    const int per = (k / 32 + SLICES - 1) / SLICES * 32;
    const int end = min(k, (warp + 1) * per);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int base = warp * per; base < end; base += 32) {
        const int at = base + 8 * t;
        const uint4 a = inputs<PRENORM>(xa, la, ia, root, scale, at);
        const uint4 b = inputs<PRENORM>(xb, lb, ib, root, scale, at);
        const uint4 wv = le ? *reinterpret_cast<const uint4*>(we + at) : make_uint4(0u, 0u, 0u, 0u);
        mma_bf16(acc, a.x, b.x, a.y, b.y, wv.x, wv.y);
        mma_bf16(acc, a.z, b.z, a.w, b.w, wv.z, wv.w);
    }
    partial[warp][g][2 * t] = acc[0];
    partial[warp][g][2 * t + 1] = acc[1];
    partial[warp][g + 8][2 * t] = acc[2];
    partial[warp][g + 8][2 * t + 1] = acc[3];
    __syncthreads();
    if (threadIdx.x < ROWS * EXPERTS) {
        const int r = threadIdx.x / EXPERTS, c = threadIdx.x % EXPERTS;
        float sum = partial[0][r][c];
#pragma unroll
        for (int s = 1; s < SLICES; ++s) sum = __fadd_rn(sum, partial[s][r][c]);
        if (r0 + r < rows && e0 + c < experts) out[static_cast<int64_t>(r0 + r) * ldo + e0 + c] = sum;
    }
}

// Selection order: a larger logit first, NaN above every number (a stable descending argsort's order), the lower id
// among equal logits; retired candidates hold -inf and id BLOCK, so a live -inf logit still outranks them.
__device__ __forceinline__ bool before(float v1, int i1, float v0, int i0) {
    const bool n1 = v1 != v1, n0 = v0 != v0;
    if (n1 != n0) return n1;
    return (!n1 && v1 > v0) || ((n1 || v1 == v0) && i1 < i0);
}

// One CTA per row (blockDim = a power of two >= experts, <= 1024). FP32 softmax over every expert, then the TOPK
// first experts in selection order, each weight its probability over the selected probabilities' sum (added in
// pick order), times its BF16 per-expert scale.
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK) select_kernel(const float* __restrict__ logits, int64_t ldl, int experts,
    const __nv_bfloat16* __restrict__ expert_scale, int top_k, int* __restrict__ picks, float* __restrict__ weights) {
    __shared__ float vals[BLOCK];
    __shared__ int ids[BLOCK];
    __shared__ int chosen[32];
    __shared__ float picked[32];
    const int r = blockIdx.x, e = threadIdx.x;
    const bool live = e < experts;
    const float l = live ? logits[static_cast<int64_t>(r) * ldl + e] : -INFINITY;
    // Maximum, then the exponential sum, each over a fixed tree.
    vals[e] = l;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (e < s) vals[e] = fmaxf(vals[e], vals[e + s]);
        __syncthreads();
    }
    const float top = vals[0];
    __syncthreads();
    const float p = live ? expf(__fsub_rn(l, top)) : 0.f;
    vals[e] = p;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (e < s) vals[e] = __fadd_rn(vals[e], vals[e + s]);
        __syncthreads();
    }
    const float total = vals[0];
    const float prob = live ? __fdiv_rn(p, total) : 0.f;
    bool alive = live;
    for (int j = 0; j < top_k; ++j) {
        __syncthreads();
        vals[e] = alive ? l : -INFINITY;
        ids[e] = alive ? e : BLOCK;
        __syncthreads();
        for (int s = BLOCK / 2; s > 0; s >>= 1) {
            if (e < s && before(vals[e + s], ids[e + s], vals[e], ids[e])) {
                vals[e] = vals[e + s];
                ids[e] = ids[e + s];
            }
            __syncthreads();
        }
        if (e == 0) chosen[j] = ids[0];
        __syncthreads();
        if (e == chosen[j]) {
            alive = false;
            picked[j] = prob;
        }
    }
    __syncthreads();
    if (e == 0) {
        float sum = 0.f;
        for (int j = 0; j < top_k; ++j) {
            if (chosen[j] >= experts) __trap();      // top_k <= experts live candidates always remain
            sum = __fadd_rn(sum, picked[j]);
        }
        for (int j = 0; j < top_k; ++j) {
            const int id = chosen[j];
            picks[static_cast<int64_t>(r) * top_k + j] = id;
            weights[static_cast<int64_t>(r) * top_k + j] =
                __fmul_rn(__fdiv_rn(picked[j], sum), __bfloat162float(expert_scale[id]));
        }
    }
}

}  // namespace

void route_logits(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, double eps, double root,
                  const c10::optional<at::Tensor>& scale) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 &&
                x.size(0) > 0 && x.size(0) <= INT32_MAX && x.stride(0) % 8 == 0 &&
                reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "route: BF16 rows, 16-byte aligned");
    const int64_t rows = x.size(0), k = x.size(1), experts = w.size(0);
    // The kernel takes K as int and forms (warp + 1) * per <= K + 8 * 31 in int.
    TORCH_CHECK(k > 0 && k % 32 == 0 && k <= (int64_t{1} << 30), "route: input width must be a multiple of 32, <= 2^30");
    TORCH_CHECK(w.is_cuda() && w.device() == x.device() && w.scalar_type() == at::kBFloat16 && w.dim() == 2 &&
                w.size(1) == k && w.is_contiguous() && experts > 0 && experts <= INT32_MAX &&
                reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "route: contiguous BF16 [experts, K] rows");
    TORCH_CHECK(out.is_cuda() && out.device() == x.device() && out.scalar_type() == at::kFloat && out.dim() == 2 &&
                out.size(0) == rows && out.size(1) == experts && out.stride(1) == 1 && out.stride(0) >= experts,
                "route: FP32 [rows, experts] logits");
    const bool prenorm = scale.has_value();
    if (prenorm) {
        const auto& s = *scale;
        TORCH_CHECK(s.is_cuda() && s.device() == x.device() && s.scalar_type() == at::kBFloat16 && s.dim() == 1 &&
                    s.size(0) == k && s.is_contiguous() && reinterpret_cast<uintptr_t>(s.data_ptr()) % 16 == 0,
                    "route: BF16 [K] router scale");
    }
    const c10::cuda::CUDAGuard guard(x.device());
    const dim3 grid((rows + ROWS - 1) / ROWS, (experts + EXPERTS - 1) / EXPERTS);
    TORCH_CHECK(grid.y <= 65535, "route: too many experts");
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
    if (prenorm)
        logits_kernel<true><<<grid, THREADS, 0, stream>>>(xp, x.stride(0), rows, wp, experts, k,
            out.data_ptr<float>(), out.stride(0), static_cast<float>(eps), static_cast<float>(root),
            reinterpret_cast<const __nv_bfloat16*>(scale->data_ptr()));
    else
        logits_kernel<false><<<grid, THREADS, 0, stream>>>(xp, x.stride(0), rows, wp, experts, k,
            out.data_ptr<float>(), out.stride(0), 0.f, 0.f, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void route_select(const at::Tensor& logits, const at::Tensor& expert_scale, at::Tensor& picks, at::Tensor& weights) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 2 && logits.stride(1) == 1 &&
                logits.size(0) > 0 && logits.size(0) <= INT32_MAX, "select: FP32 [rows, experts] logits");
    const int64_t rows = logits.size(0), experts = logits.size(1), top_k = picks.size(1);
    TORCH_CHECK(experts > 0 && experts <= 1024 && top_k > 0 && top_k <= 32 && top_k <= experts,
                "select: 1..1024 experts and 1..min(32, experts) picks");
    TORCH_CHECK(expert_scale.is_cuda() && expert_scale.device() == logits.device() &&
                expert_scale.scalar_type() == at::kBFloat16 && expert_scale.dim() == 1 &&
                expert_scale.size(0) == experts && expert_scale.is_contiguous(), "select: BF16 [experts] scales");
    TORCH_CHECK(picks.is_cuda() && picks.device() == logits.device() && picks.scalar_type() == at::kInt &&
                picks.dim() == 2 && picks.size(0) == rows && picks.is_contiguous() &&
                weights.is_cuda() && weights.device() == logits.device() && weights.scalar_type() == at::kFloat &&
                weights.sizes() == picks.sizes() && weights.is_contiguous(),
                "select: contiguous INT32 picks and FP32 weights [rows, top_k]");
    const c10::cuda::CUDAGuard guard(logits.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto launch = [&](auto block) {
        constexpr int BLOCK = decltype(block)::value;
        select_kernel<BLOCK><<<rows, BLOCK, 0, stream>>>(logits.data_ptr<float>(), logits.stride(0), experts,
            reinterpret_cast<const __nv_bfloat16*>(expert_scale.data_ptr()), top_k, picks.data_ptr<int>(),
            weights.data_ptr<float>());
    };
    if (experts <= 128) launch(std::integral_constant<int, 128>{});
    else if (experts <= 256) launch(std::integral_constant<int, 256>{});
    else if (experts <= 512) launch(std::integral_constant<int, 512>{});
    else launch(std::integral_constant<int, 1024>{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("logits", &route_logits, pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("out"),
          pybind11::arg("eps"), pybind11::arg("root"), pybind11::arg("scale"));
    m.def("select", &route_select, pybind11::arg("logits"), pybind11::arg("expert_scale"), pybind11::arg("picks"),
          pybind11::arg("weights"));
}
