// One launch replays every recurrent layer of one State: grid (value head, layer).
//
// Per (layer, value head) the work is gdn.cu's replay_kernel unchanged: the same
// ``update`` delta-rule step (copied verbatim; compiled with --fmad=false as gdn.cu),
// the same row order and the same FP32 state layout, so the results are bitwise
// those of the per-layer replay calls. Layer l reads State.rec[cur_l, l] and writes
// State.rec[1 - cur_l, l]; its saved rows come from a per-State device table of the
// layer's GDNScratch k/v/g/b pointers.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int DK = 128, DV = 128;

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

// Identical to gdn.cu ``update``.
__device__ __forceinline__ void update(float (&s)[4][4], const float (&kk)[4], const float* vrow, int warp,
                                       float g, float beta) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * g;
            kv = kv + s[j][i] * kk[i];
        }
        kv = warp_sum(kv);
        const float delta = (vrow[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }
}

template <int NK, int NV>
__global__ void __launch_bounds__(1024) replay_many_kernel(
        float* __restrict__ rec, int64_t layers, uint64_t parity, const int64_t* __restrict__ saved, int rows) {
    const int hv = blockIdx.x, layer = blockIdx.y, hk = hv / (NV / NK);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const size_t bank = (size_t)NV * DV * DK;
    const int cur = (int)((parity >> layer) & 1u);
    const float* __restrict__ state_in = rec + ((size_t)cur * layers + layer) * bank;
    float* __restrict__ state_out = rec + ((size_t)(1 - cur) * layers + layer) * bank;
    const float* __restrict__ k_save = reinterpret_cast<const float*>(saved[4 * layer + 0]);
    const __half* __restrict__ v_save = reinterpret_cast<const __half*>(saved[4 * layer + 1]);
    const float* __restrict__ g_save = reinterpret_cast<const float*>(saved[4 * layer + 2]);
    const float* __restrict__ b_save = reinterpret_cast<const float*>(saved[4 * layer + 3]);
    __shared__ float vs[DV];
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r = 0; r < rows; ++r) {
        if (t < DV) vs[t] = __half2float(v_save[((size_t)r * NV + hv) * DV + t]);
        __syncthreads();
        float kk[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) kk[i] = k_save[((size_t)r * NK + hk) * DK + lane * 4 + i];
        update(s, kk, vs, warp, g_save[r * NV + hv], b_save[r * NV + hv]);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
}

// rec: FP32 [2, layers, NV, 128, 128]; saved: int64 [layers, 4] device pointers whose
// geometry the Python owner validated when it built the table (see gdn_replay_many.py).
void replay_many(at::Tensor rec, int64_t parity, const at::Tensor& saved, int64_t rows) {
    TORCH_CHECK(rec.is_cuda() && rec.scalar_type() == at::kFloat && rec.is_contiguous() && rec.dim() == 5
                && rec.size(0) == 2 && rec.size(3) == DV && rec.size(4) == DK, "replay_many: FP32 [2,L,NV,128,128] state");
    const int64_t layers = rec.size(1), nv = rec.size(2);
    TORCH_CHECK(layers >= 1 && layers <= 64 && layers <= 65535, "replay_many: 1..64 layers");
    TORCH_CHECK(saved.is_cuda() && saved.device() == rec.device() && saved.scalar_type() == at::kLong
                && saved.is_contiguous() && saved.dim() == 2 && saved.size(0) == layers && saved.size(1) == 4,
                "replay_many: int64 [layers,4] saved-row pointer table on the state device");
    TORCH_CHECK(rows >= 1 && rows <= 1024, "replay_many: 1..1024 rows");
    TORCH_CHECK(layers == 64 || ((uint64_t)parity >> layers) == 0, "replay_many: parity bits beyond layers");
    const auto* prop = at::cuda::getDeviceProperties(rec.get_device());
    TORCH_CHECK(prop->major == 7 && prop->minor == 0, "replay_many requires SM70");
    c10::cuda::CUDAGuard guard(rec.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((unsigned)nv, (unsigned)layers);
    auto args = [&](auto kernel) {
        kernel<<<grid, 1024, 0, stream>>>(rec.data_ptr<float>(), layers, (uint64_t)parity,
                                          saved.data_ptr<int64_t>(), (int)rows);
    };
    if (nv == 48) args(replay_many_kernel<16, 48>);
    else if (nv == 24) args(replay_many_kernel<8, 24>);
    else if (nv == 12) args(replay_many_kernel<4, 12>);
    else if (nv == 6) args(replay_many_kernel<2, 6>);
    else TORCH_CHECK(false, "replay_many: 48, 24, 12 or 6 value heads");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("replay_many", &replay_many);
}
