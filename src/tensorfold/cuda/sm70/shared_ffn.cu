// SM70 decode shared-expert SwiGLU FFN for 1..32 FP16 rows (shared_ffn.py).
//
//   act[r, c] = fp16(silu(sum_k x[r,k] gate[c,k]) * sum_k x[r,k] up[c,k])     (FP32 sums, FP32 SiLU/product)
//   out[r, n] = sum_c act[r,c] down[n,c]                                      (FP32 sums, FP32 output)
//
// The same operation boundaries as the reference dense program (two FP32-output
// GEMMs, FP32 SiLU and product, one FP16 rounding, FP32-output down GEMM); only
// the summation order differs. Every row's sums use one fixed order (per-lane
// sequential FMA over its K chunks, then a xor butterfly) that does not depend
// on the call's row count. Two launches, no workspace, no allocation: capturable.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {
constexpr int MAX_ROWS = 32;
constexpr int MAX_K_CHUNKS = 10;   // K <= 2560 (8 halves per lane per chunk)

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, mask));
    return value;
}

__device__ __forceinline__ float dot8(const uint4& a, const uint4& b, float acc) {
    const __half2* x = reinterpret_cast<const __half2*>(&a);
    const __half2* w = reinterpret_cast<const __half2*>(&b);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 xf = __half22float2(x[i]), wf = __half22float2(w[i]);
        acc = __fmaf_rn(xf.x, wf.x, acc);
        acc = __fmaf_rn(xf.y, wf.y, acc);
    }
    return acc;
}

// One warp per shared column c: its gate and up rows stay in registers for all rows.
template <int CHUNKS>
__global__ void __launch_bounds__(128) gate_up_act(const __half* __restrict__ x, int64_t x_stride,
                                                   const __half* __restrict__ gate, const __half* __restrict__ up,
                                                   __half* __restrict__ act, int rows, int width, int k) {
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (c >= width) return;
    uint4 g[CHUNKS], u[CHUNKS];
    const uint4* gr = reinterpret_cast<const uint4*>(gate + int64_t(c) * k);
    const uint4* ur = reinterpret_cast<const uint4*>(up + int64_t(c) * k);
    #pragma unroll
    for (int i = 0; i < CHUNKS; ++i) { g[i] = __ldg(gr + i * 32 + lane); u[i] = __ldg(ur + i * 32 + lane); }
    for (int r = 0; r < rows; ++r) {
        const uint4* xr = reinterpret_cast<const uint4*>(x + int64_t(r) * x_stride);
        float sg = 0.0f, su = 0.0f;
        #pragma unroll
        for (int i = 0; i < CHUNKS; ++i) {
            const uint4 v = __ldg(xr + i * 32 + lane);
            sg = dot8(v, g[i], sg);
            su = dot8(v, u[i], su);
        }
        sg = warp_sum(sg);
        su = warp_sum(su);
        if (lane == 0) {
            const float silu = __fdiv_rn(sg, __fadd_rn(1.0f, expf(-sg)));
            act[int64_t(r) * width + c] = __float2half_rn(__fmul_rn(silu, su));
        }
    }
}

// One warp per eight output columns; the rows' activations sit in shared memory. Lane l holds the down
// row's half2 pairs l, l + 32, ... (J of them, width = 2 * pairs), summed in that order, then a butterfly.
constexpr int MAX_WIDTH = 256;

template <int J>
__global__ void __launch_bounds__(128) down_rows(const __half* __restrict__ act, const __half* __restrict__ down,
                                                 float* __restrict__ out, int64_t out_stride, int rows, int n,
                                                 int width) {
    __shared__ __align__(16) __half2 a[MAX_ROWS * MAX_WIDTH / 2];
    const int pairs = width / 2;
    for (int i = threadIdx.x; i < rows * pairs; i += blockDim.x)
        a[i] = reinterpret_cast<const __half2*>(act)[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * 4 + (threadIdx.x >> 5)) * 8;
    if (first >= n) return;
    __half2 w[8][J];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __half2* src = reinterpret_cast<const __half2*>(down + int64_t(first + j) * width);
        #pragma unroll
        for (int e = 0; e < J; ++e)
            w[j][e] = lane + 32 * e < pairs ? __ldg(src + lane + 32 * e) : __float2half2_rn(0.f);
    }
    for (int r = 0; r < rows; ++r) {
        float2 av[J];
        #pragma unroll
        for (int e = 0; e < J; ++e)
            av[e] = lane + 32 * e < pairs ? __half22float2(a[r * pairs + lane + 32 * e]) : make_float2(0.f, 0.f);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float s = 0.0f;
            #pragma unroll
            for (int e = 0; e < J; ++e) {
                if (lane + 32 * e < pairs) {
                    const float2 wf = __half22float2(w[j][e]);
                    s = __fmaf_rn(av[e].x, wf.x, s);
                    s = __fmaf_rn(av[e].y, wf.y, s);
                }
            }
            s = warp_sum(s);
            if (lane == j) out[int64_t(r) * out_stride + first + j] = s;
        }
    }
}

template <int J>
void launch_down(const at::Tensor& act, const at::Tensor& down, at::Tensor& out, cudaStream_t stream) {
    const int rows = int(act.size(0)), n = int(down.size(0)), width = int(down.size(1));
    down_rows<J><<<(n / 8 + 3) / 4, 128, 0, stream>>>(
        reinterpret_cast<const __half*>(act.data_ptr<at::Half>()), reinterpret_cast<const __half*>(down.data_ptr<at::Half>()),
        out.data_ptr<float>(), out.stride(0), rows, n, width);
}
}  // namespace

void shared_ffn(const at::Tensor& x, const at::Tensor& gate, const at::Tensor& up, const at::Tensor& down,
                at::Tensor act, at::Tensor out) {
    const auto half = at::kHalf;
    const int64_t rows = x.size(0), k = x.size(1), width = gate.size(0), n = down.size(0);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == half && x.dim() == 2 && 1 <= rows && rows <= MAX_ROWS
                && x.stride(1) == 1 && x.stride(0) % 8 == 0 && x.data_ptr<at::Half>() != nullptr
                && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "shared FFN input must be 1..32 16-byte aligned channel-contiguous FP16 rows");
    TORCH_CHECK(k % 256 == 0 && k / 256 <= MAX_K_CHUNKS, "shared FFN requires K a multiple of 256, at most 2560");
    for (const at::Tensor* w : {&gate, &up})
        TORCH_CHECK(w->device() == x.device() && w->scalar_type() == half && w->is_contiguous()
                    && w->dim() == 2 && w->size(0) == width && w->size(1) == k
                    && reinterpret_cast<uintptr_t>(w->data_ptr()) % 16 == 0,
                    "shared FFN gate/up must be aligned contiguous FP16 [width, K]");
    TORCH_CHECK(width >= 2 && width <= MAX_WIDTH && width % 2 == 0, "shared FFN supports even widths up to 256");
    TORCH_CHECK(down.device() == x.device() && down.scalar_type() == half && down.is_contiguous() && down.dim() == 2
                && down.size(1) == width && n % 32 == 0 && reinterpret_cast<uintptr_t>(down.data_ptr()) % 4 == 0,
                "shared FFN down must be aligned contiguous FP16 [N, width], N a multiple of 32");
    TORCH_CHECK(act.device() == x.device() && act.scalar_type() == half && act.is_contiguous()
                && act.dim() == 2 && act.size(0) == rows && act.size(1) == width
                && reinterpret_cast<uintptr_t>(act.data_ptr()) % 4 == 0,
                "shared FFN activation scratch must be aligned contiguous FP16 [rows, width]");
    TORCH_CHECK(out.device() == x.device() && out.scalar_type() == at::kFloat && out.dim() == 2
                && out.size(0) == rows && out.size(1) == n && out.stride(1) == 1 && out.stride(0) >= n,
                "shared FFN output must be FP32 [rows, N] with contiguous columns");
    const auto* prop = at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "shared FFN requires SM70");
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    const auto* gp = reinterpret_cast<const __half*>(gate.data_ptr<at::Half>());
    const auto* upp = reinterpret_cast<const __half*>(up.data_ptr<at::Half>());
    auto* ap = reinterpret_cast<__half*>(act.data_ptr<at::Half>());
    const int blocks = int((width + 3) / 4);
    switch (k / 256) {
#define TF_SHARED_FFN_CASE(C) case C: gate_up_act<C><<<blocks, 128, 0, stream>>>(xp, x.stride(0), gp, upp, ap, int(rows), int(width), int(k)); break;
        TF_SHARED_FFN_CASE(1) TF_SHARED_FFN_CASE(2) TF_SHARED_FFN_CASE(3) TF_SHARED_FFN_CASE(4) TF_SHARED_FFN_CASE(5)
        TF_SHARED_FFN_CASE(6) TF_SHARED_FFN_CASE(7) TF_SHARED_FFN_CASE(8) TF_SHARED_FFN_CASE(9) TF_SHARED_FFN_CASE(10)
#undef TF_SHARED_FFN_CASE
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int j = int((width / 2 + 31) / 32);
    if (j == 1) launch_down<1>(act, down, out, stream);
    else if (j == 2) launch_down<2>(act, down, out, stream);
    else if (j == 3) launch_down<3>(act, down, out, stream);
    else launch_down<4>(act, down, out, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
