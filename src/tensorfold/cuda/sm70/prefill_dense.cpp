#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/core/InferenceMode.h>
#include <cublas_v2.h>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <array>
#include <memory>
#include <utility>
#include "prefill_dense_contracts.cuh"

void prefill_dense_round_half(const at::Tensor&, at::Tensor);
namespace {
using tensorfold::prefill::operand;
using tensorfold::prefill::writable;
using tensorfold::prefill::overlaps;
void blas(cublasStatus_t status) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "prefill Dense cuBLAS status ", int(status));
}

struct State {
    cublasHandle_t handle_ = nullptr;
    at::Tensor workspace_;
    c10::Device device_;
    cudaStream_t stream_;
    bool configured_ = false;
    bool terminal_ = false;
    bool pending_ = false;
    bool failed_ = false;
    const void* workspace_data_ = nullptr;
    int64_t workspace_bytes_ = 0;
    std::array<at::Tensor, 4> loans_;
    explicit State(at::Tensor workspace): workspace_(std::move(workspace)), device_(workspace_.device()) {}
};

class Prepared {
    State* state_ = nullptr;
    State& live() const {
        TORCH_CHECK(state_ && state_->handle_ && !state_->terminal_,
            "prefill Dense capability is closed or retirement is uncertain");
        return *state_;
    }
public:
    explicit Prepared(at::Tensor workspace) {
        auto pending = std::make_unique<State>(std::move(workspace));
        auto& s = *pending;
        TORCH_CHECK(s.device_.is_cuda(), "prefill Dense workspace must be CUDA");
        operand(s.workspace_, at::kByte, s.device_);
        TORCH_CHECK(s.workspace_.dim() == 1 && s.workspace_.is_contiguous()
            && uintptr_t(s.workspace_.data_ptr()) % 256 == 0, "prefill Dense workspace requires aligned byte vector");
        c10::cuda::CUDAGuard guard(s.device_);
        const auto* prop = at::cuda::getDeviceProperties(s.device_.index());
        TORCH_CHECK(prop->major == 7 && prop->minor == 0, "prefill Dense requires SM70");
        s.stream_ = at::cuda::getCurrentCUDAStream(s.device_.index()).stream();
        s.workspace_data_ = s.workspace_.data_ptr();
        s.workspace_bytes_ = s.workspace_.numel();
        const auto status = cublasCreate(&s.handle_);
        if (status != CUBLAS_STATUS_SUCCESS && s.handle_) {
            // A failed foreign acquisition with a published token is ambiguous.
            // Preserve its storage; never attempt an unproved destroy/retry.
            std::fputs("prefill Dense: retaining uncertain acquisition\n", stderr);
            pending.release();
        }
        blas(status);
        state_ = pending.release();
    }
    Prepared(const Prepared&) = delete;
    Prepared& operator=(const Prepared&) = delete;
    ~Prepared() noexcept {
        if (!state_) return;
        if (state_->handle_) {
            // No CUDA API/retry in destruction. Retain the preallocated state,
            // including workspace and in-flight operands, until process exit.
            std::fputs("prefill Dense: retaining unretired handle and borrowers\n", stderr);
        } else {
            delete state_;
        }
    }
    void configure() {
        auto& s = live();
        TORCH_CHECK(s.handle_ && !s.configured_, "prefill Dense configure state");
        c10::cuda::CUDAGuard guard(s.device_);
        blas(cublasSetStream(s.handle_, s.stream_));
        // SetStream resets workspace; the explicit owner is installed AFTER it.
        blas(cublasSetWorkspace(s.handle_, s.workspace_.data_ptr(), s.workspace_.numel()));
        blas(cublasSetPointerMode(s.handle_, CUBLAS_POINTER_MODE_HOST));
        blas(cublasSetAtomicsMode(s.handle_, CUBLAS_ATOMICS_NOT_ALLOWED));
        blas(cublasSetMathMode(s.handle_, static_cast<cublasMath_t>(
            CUBLAS_PEDANTIC_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
        s.configured_ = true;
    }
    void run(const at::Tensor& x, const at::Tensor& weight, at::Tensor accumulation, at::Tensor out) {
        auto& s = live();
        TORCH_CHECK(s.handle_ && s.configured_, "prefill Dense capability is not open/configured");
        TORCH_CHECK(!s.pending_ && !s.failed_, "prefill Dense work is pending or failed");
        TORCH_CHECK(s.workspace_.data_ptr() == s.workspace_data_ && s.workspace_.numel() == s.workspace_bytes_,
            "prefill Dense workspace lease changed");
        c10::cuda::CUDAGuard guard(s.device_);
        TORCH_CHECK(at::cuda::getCurrentCUDAStream(s.device_.index()).stream() == s.stream_, "prefill Dense stream mismatch");
        cudaStreamCaptureStatus capture;
        C10_CUDA_CHECK(cudaStreamIsCapturing(s.stream_, &capture));
        TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "prefill Dense component requires uncaptured execution");
        operand(x, at::kHalf, s.device_); operand(weight, at::kHalf, s.device_);
        operand(accumulation, at::kFloat, s.device_); operand(out, at::kHalf, s.device_);
        writable(accumulation); writable(out);
        TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && accumulation.dim() == 2 && out.dim() == 2
            && x.size(0) > 0 && x.size(0) <= INT_MAX && x.size(1) > 0 && x.size(1) <= INT_MAX
            && weight.size(0) > 0 && weight.size(0) <= INT_MAX && weight.size(1) == x.size(1)
            && accumulation.size(0) == x.size(0) && accumulation.size(1) == weight.size(0)
            && out.sizes() == accumulation.sizes(), "prefill Dense geometry");
        TORCH_CHECK(x.stride(1) == 1 && x.stride(0) >= x.size(1) && x.stride(0) <= INT_MAX
            && weight.is_contiguous() && accumulation.stride(1) == 1
            && accumulation.stride(0) >= accumulation.size(1) && accumulation.stride(0) <= INT_MAX
            && out.stride(1) == 1 && out.stride(0) >= out.size(1), "prefill Dense row strides");
        TORCH_CHECK(!overlaps(accumulation, x) && !overlaps(accumulation, weight)
            && !overlaps(out, x) && !overlaps(out, weight)
            && !overlaps(out, accumulation) && !overlaps(s.workspace_, x)
            && !overlaps(s.workspace_, weight) && !overlaps(s.workspace_, accumulation)
            && !overlaps(s.workspace_, out), "prefill Dense writable spans must be disjoint");
        s.loans_ = {x, weight, accumulation, out};
        s.pending_ = true;
        const float alpha = 1.f, beta = 0.f;
        // Row-major X*W^T is column-major W*X^T: preserve original N/K and part order.
        try {
            blas(cublasGemmEx(s.handle_, CUBLAS_OP_T, CUBLAS_OP_N, int(weight.size(0)), int(x.size(0)), int(x.size(1)),
            &alpha, weight.data_ptr(), CUDA_R_16F, int(weight.size(1)),
            x.data_ptr(), CUDA_R_16F, int(x.stride(0)), &beta, accumulation.data_ptr(), CUDA_R_32F,
            int(accumulation.stride(0)), CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
            prefill_dense_round_half(accumulation, out);
        } catch (...) {
            s.failed_ = true;
            throw;
        }
    }
    void finish() {
        auto& s = live();
        c10::cuda::CUDAGuard guard(s.device_);
        const auto status = cudaStreamSynchronize(s.stream_);
        if (status != cudaSuccess) s.terminal_ = true;
        C10_CUDA_CHECK(status);
        s.loans_ = {};
        s.pending_ = false;
    }
    pybind11::dict settings() const {
        const auto& s = live();
        TORCH_CHECK(s.handle_ && s.configured_, "prefill Dense capability is not open/configured");
        c10::cuda::CUDAGuard guard(s.device_);
        cublasMath_t math;
        cublasPointerMode_t pointer;
        cublasAtomicsMode_t atomics;
        int version;
        blas(cublasGetMathMode(s.handle_, &math));
        blas(cublasGetPointerMode(s.handle_, &pointer));
        blas(cublasGetAtomicsMode(s.handle_, &atomics));
        blas(cublasGetVersion(s.handle_, &version));
        pybind11::dict result;
        result["math_mode"] = int(math);
        result["pointer_mode"] = int(pointer);
        result["atomics_mode"] = int(atomics);
        result["cublas_version"] = version;
        result["device"] = s.device_.index();
        result["workspace_bytes"] = s.workspace_bytes_;
        return result;
    }
    void close() {
        if (!state_ || !state_->handle_) return;
        auto& s = live();  // An uncertain previous cleanup is never retried.
        c10::cuda::CUDAGuard guard(s.device_);
        const auto fence = cudaStreamSynchronize(s.stream_);
        if (fence != cudaSuccess) s.terminal_ = true;
        C10_CUDA_CHECK(fence);
        const auto status = cublasDestroy(s.handle_);
        if (status != CUBLAS_STATUS_SUCCESS) s.terminal_ = true;
        blas(status);
        s.handle_ = nullptr;
        s.configured_ = false;
        s.loans_ = {};
        s.workspace_ = at::Tensor();
    }

};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_overlaps", &overlaps);  // Actual ABI boundary helper, exercised independently of GEMM.
    pybind11::class_<Prepared>(m, "Prepared")
        .def(pybind11::init<at::Tensor>()).def("configure", &Prepared::configure)
        .def("run", &Prepared::run).def("finish", &Prepared::finish).def("settings", &Prepared::settings)
        .def("close", &Prepared::close);
    m.attr("math_mode") = int(CUBLAS_PEDANTIC_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    m.attr("pointer_mode") = int(CUBLAS_POINTER_MODE_HOST);
    m.attr("atomics_mode") = int(CUBLAS_ATOMICS_NOT_ALLOWED);
}
