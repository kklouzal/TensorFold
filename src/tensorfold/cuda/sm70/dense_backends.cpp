#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <array>
#include <climits>
#include <cstdio>
#include <memory>
#include <limits>
#include <thread>
#include "dense_backends.cuh"
#include "prefill_dense_contracts.cuh"

namespace {
void blas(cublasStatus_t s) { TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "dense cuBLAS status ", int(s)); }
struct ContextState {
    c10::Device device;
    std::string backend;
    cublasHandle_t handle = nullptr;
    cublasLtHandle_t lt = nullptr;
    std::thread::id thread = std::this_thread::get_id();
    bool closed = false, terminal = false, configured = false, retirement_fenced = false;
    ContextState(int index, std::string backend_): device(c10::kCUDA, index), backend(std::move(backend_)) {}
};
class Context {
    std::shared_ptr<ContextState> state_;
public:
    Context(int index, std::string backend): state_(new ContextState(index, std::move(backend)), [](ContextState* s) {
        if (!s->closed && (s->handle || s->lt)) std::fputs("dense: retaining unretired context\n", stderr);
        else delete s;
    }) {
        TORCH_CHECK(index >= 0 && index <= std::numeric_limits<c10::DeviceIndex>::max(),
            "dense context device index outside supported range");
        TORCH_CHECK(state_->backend == "cublas" || state_->backend == "cublaslt" || state_->backend == "cutlass",
            "unknown dense backend");
    }
    std::shared_ptr<ContextState> state() const { return state_; }
    bool retirement_fenced() const { return state_->retirement_fenced; }
    void configure() {
        auto& s = *state_;
        TORCH_CHECK(s.thread == std::this_thread::get_id(), "dense context thread mismatch");
        c10::cuda::CUDAGuard guard(s.device);
        TORCH_CHECK(!s.closed && !s.terminal && !s.configured, "dense context configure state");
        const auto* prop = at::cuda::getDeviceProperties(s.device.index());
        TORCH_CHECK(prop->major == 7 && prop->minor == 0, "dense candidates require SM70");
        cudaStreamCaptureStatus capture;
        C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(s.device.index()).stream(), &capture));
        TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "dense context preparation requires uncaptured stream");
        if (s.backend == "cublas") {
            const auto status = cublasCreate(&s.handle);
            if (status != CUBLAS_STATUS_SUCCESS && s.handle) s.terminal = true;
            blas(status);
            blas(cublasSetPointerMode(s.handle, CUBLAS_POINTER_MODE_HOST));
            blas(cublasSetAtomicsMode(s.handle, CUBLAS_ATOMICS_NOT_ALLOWED));
            blas(cublasSetMathMode(s.handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        } else if (s.backend == "cublaslt") {
            // One explicit handle survives startup->scheduler thread handoff;
            // no borrowed thread-local handle can outlive its pool window.
            const auto status = cublasLtCreate(&s.lt);
            if (status != CUBLAS_STATUS_SUCCESS && s.lt) s.terminal = true;
            blas(status);
        }
        s.configured = true;
    }
    void handoff_to_current_thread() {
        auto& s = *state_;
        TORCH_CHECK(s.configured && !s.closed && !s.terminal, "dense context closed or retirement uncertain");
        c10::cuda::CUDAGuard guard(s.device);
        cudaStreamCaptureStatus capture;
        C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(s.device.index()).stream(), &capture));
        TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "dense context handoff requires uncaptured execution");
        // Callers drain/join the previous host owner before this explicit cold
        // handoff. Pybind retains the GIL for all context/plan operations.
        const auto fence = cudaDeviceSynchronize();
        if (fence != cudaSuccess) s.terminal = true;
        C10_CUDA_CHECK(fence);
        s.thread = std::this_thread::get_id();
    }
    void close() {
        auto& s = *state_;
        if (s.closed) return;
        TORCH_CHECK(!s.terminal, "dense context retirement uncertain");
        c10::cuda::CUDAGuard guard(s.device);
        cudaStreamCaptureStatus capture;
        C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(s.device.index()).stream(), &capture));
        TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "dense context retirement requires uncaptured stream");
        const auto fence = cudaDeviceSynchronize();
        if (fence != cudaSuccess) s.terminal = true;
        C10_CUDA_CHECK(fence);
        s.retirement_fenced = true;
        cublasStatus_t classic = CUBLAS_STATUS_SUCCESS, lightweight = CUBLAS_STATUS_SUCCESS;
        if (s.handle) {
            const auto result = cublasDestroy(s.handle);
            if (result != CUBLAS_STATUS_SUCCESS) s.terminal = true;
            else s.handle = nullptr;
            classic = result;
        }
        if (s.lt) {
            const auto result = cublasLtDestroy(s.lt);
            if (result != CUBLAS_STATUS_SUCCESS) s.terminal = true;
            else s.lt = nullptr;
            lightweight = result;
        }
        TORCH_CHECK(!s.terminal, "dense context retirement status ", int(classic), ",", int(lightweight));
        s.closed = true;
    }
};
struct Binding {
    at::Tensor tensor;
    c10::Device device;
    at::ScalarType dtype;
    std::array<int64_t, 2> sizes, strides;
    explicit Binding(at::Tensor t): tensor(std::move(t)), device(tensor.device()), dtype(tensor.scalar_type()),
        sizes{tensor.size(0), tensor.size(1)}, strides{tensor.stride(0), tensor.stride(1)} {}
    void check(const at::Tensor& current) const {
        TORCH_CHECK(current.device() == device && current.scalar_type() == dtype
            && current.dim() == 2 && current.size(0) == sizes[0]
            && current.size(1) == sizes[1] && current.stride(0) == strides[0] && current.stride(1) == strides[1],
            "dense prepared geometry/dtype/device changed");
        tensorfold::prefill::operand(current, dtype, device);
    }
};
struct State {
    std::shared_ptr<ContextState> context;
    Binding x, w, out;
    std::string backend;
    c10::Device device;
    cudaStream_t stream;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    std::array<uint32_t, 3> alignment{};
    DenseCutlassPlan* cutlass = nullptr;
    bool configured = false, failed = false, terminal = false, ran = false, warmed = false;
    State(std::shared_ptr<ContextState> context_, at::Tensor x_, at::Tensor w_, at::Tensor out_):
        context(std::move(context_)), x(std::move(x_)), w(std::move(w_)), out(std::move(out_)),
        backend(context->backend), device(x.tensor.device()) {}
};

class Prepared {
    State* s_ = nullptr;
    State& live() const {
        TORCH_CHECK(s_ && !s_->terminal, "dense owner closed or retirement uncertain");
        TORCH_CHECK(s_->context->retirement_fenced || s_->context->thread == std::this_thread::get_id(),
            "dense owner thread mismatch");
        return *s_;
    }
    static void uncaptured(cudaStream_t stream) {
        cudaStreamCaptureStatus status;
        C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
        TORCH_CHECK(status == cudaStreamCaptureStatusNone, "dense preparation/retirement requires uncaptured stream");
    }
public:
    Prepared(Context& context, at::Tensor x, at::Tensor w, at::Tensor out) {
        using namespace tensorfold::prefill;
        const auto owner = context.state();
        TORCH_CHECK(owner->configured && !owner->closed && !owner->terminal && owner->device == x.device()
            && owner->thread == std::this_thread::get_id(), "dense context mismatch/closed");
        TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 2, "dense operands require matrices");
        operand(x, at::kHalf, x.device()); operand(w, at::kHalf, x.device());
        TORCH_CHECK(out.scalar_type() == at::kHalf || out.scalar_type() == at::kFloat, "dense output requires Half/Float");
        operand(out, out.scalar_type(), x.device()); writable(out);
        TORCH_CHECK(x.size(0) > 0 && x.size(0) <= INT_MAX && x.size(1) > 0 && x.size(1) <= INT_MAX
            && w.size(0) > 0 && w.size(0) <= INT_MAX && w.size(1) == x.size(1)
            && out.size(0) == x.size(0) && out.size(1) == w.size(0), "dense geometry");
        TORCH_CHECK(w.is_contiguous() && x.stride(1) == 1 && x.stride(0) >= x.size(1)
            && x.stride(0) <= INT_MAX && out.stride(1) == 1 && out.stride(0) >= out.size(1)
            && out.stride(0) <= INT_MAX, "dense row strides");
        TORCH_CHECK(!overlaps(out, x) && !overlaps(out, w), "dense writable span overlaps input");
        c10::cuda::CUDAGuard guard(x.device());
        const auto* prop = at::cuda::getDeviceProperties(x.device().index());
        TORCH_CHECK(prop->major == 7 && prop->minor == 0, "dense candidates require SM70");
        const auto stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();
        uncaptured(stream);
        auto state = std::make_unique<State>(owner, std::move(x), std::move(w), std::move(out));
        state->stream = stream;
        const std::array<const void*, 3> pointers{state->x.tensor.data_ptr(), state->w.tensor.data_ptr(),
            state->out.tensor.data_ptr()};
        for (size_t i = 0; i < pointers.size(); ++i) {
            const auto address = reinterpret_cast<uintptr_t>(pointers[i]);
            uint32_t alignment = 1;
            while (alignment < 256 && address % (alignment * 2) == 0) alignment *= 2;
            state->alignment[i] = alignment;
        }
        s_ = state.release();
    }
    ~Prepared() noexcept {
        if (s_) std::fputs("dense: retaining unretired owner and bound operands\n", stderr);
    }
    Prepared(const Prepared&) = delete;
    Prepared& operator=(const Prepared&) = delete;
    void configure() {
        auto& s = live();
        TORCH_CHECK(!s.configured && !s.failed, "dense configure state");
        c10::cuda::CUDAGuard guard(s.device);
        uncaptured(s.stream);
        try {
            if (s.backend == "cutlass") {
                s.cutlass = dense_cutlass_prepare(s.x.tensor, s.w.tensor, s.out.tensor, s.stream);
            } else if (s.backend == "cublas") {
                TORCH_CHECK(s.context->handle, "dense cuBLAS context not configured");
            } else {
                TORCH_CHECK(s.context->lt, "dense Lt context not configured");
                auto acquire = [&](cublasStatus_t status, auto token) {
                    // A failed acquisition publishing a token provides no
                    // valid-destroy guarantee. Retain the journal terminally.
                    if (status != CUBLAS_STATUS_SUCCESS && token) s.terminal = true;
                    blas(status);
                };
                // Calls precede token inspection in separate full expressions.
                auto status = cublasLtMatmulDescCreate(&s.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
                acquire(status, s.operation);
                const cublasOperation_t transpose = CUBLAS_OP_T;
                blas(cublasLtMatmulDescSetAttribute(s.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
                // Column-major W^T * X^T produces row-major X * W^T.
                status = cublasLtMatrixLayoutCreate(&s.a, CUDA_R_16F, s.w.sizes[1], s.w.sizes[0], s.w.strides[0]);
                acquire(status, s.a);
                status = cublasLtMatrixLayoutCreate(&s.b, CUDA_R_16F, s.x.sizes[1], s.x.sizes[0], s.x.strides[0]);
                acquire(status, s.b);
                const auto dtype = s.out.dtype == at::kHalf ? CUDA_R_16F : CUDA_R_32F;
                status = cublasLtMatrixLayoutCreate(&s.d, dtype, s.out.sizes[1], s.out.sizes[0], s.out.strides[0]);
                acquire(status, s.d);
                status = cublasLtMatmulPreferenceCreate(&s.preference);
                acquire(status, s.preference);
                const uint64_t bytes = 0;
                blas(cublasLtMatmulPreferenceSetAttribute(s.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)));
                const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
                blas(cublasLtMatmulPreferenceSetAttribute(s.preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                    &reduction, sizeof(reduction)));
                const std::array<cublasLtMatmulPreferenceAttributes_t, 4> attributes{
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
                for (size_t i = 0; i < attributes.size(); ++i) {
                    // The prepared alignment class is part of the plan key.
                    const uint32_t alignment = s.alignment[i == 0 ? 1 : i == 1 ? 0 : 2];
                    blas(cublasLtMatmulPreferenceSetAttribute(s.preference, attributes[i], &alignment, sizeof(alignment)));
                }
                cublasLtMatmulHeuristicResult_t result{};
                int count = 0;
                blas(cublasLtMatmulAlgoGetHeuristic(s.context->lt, s.operation, s.a, s.b, s.d, s.d, s.preference, 1, &result, &count));
                TORCH_CHECK(count == 1 && result.state == CUBLAS_STATUS_SUCCESS && result.workspaceSize == 0,
                    "dense Lt has no zero-workspace algorithm");
                s.algorithm = result.algo;
            }
            s.configured = true;
        } catch (...) { s.failed = true; throw; }
    }
    void run(const at::Tensor& x, const at::Tensor& w, at::Tensor out) {
        auto& s = live();
        TORCH_CHECK(s.configured && !s.failed, "dense owner not configured or failed");
        TORCH_CHECK(!s.context->closed && !s.context->terminal, "dense context closed or retirement uncertain");
        c10::cuda::CUDAGuard guard(s.device);
        const auto stream = at::cuda::getCurrentCUDAStream(s.device.index()).stream();
        s.x.check(x); s.w.check(w); s.out.check(out);
        if (s.backend == "cublaslt") {
            TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % s.alignment[0] == 0
                && reinterpret_cast<uintptr_t>(w.data_ptr()) % s.alignment[1] == 0
                && reinterpret_cast<uintptr_t>(out.data_ptr()) % s.alignment[2] == 0, "dense Lt pointer alignment class changed");
        }
        tensorfold::prefill::writable(out);
        TORCH_CHECK(!tensorfold::prefill::overlaps(out, x) && !tensorfold::prefill::overlaps(out, w),
            "dense writable span overlaps input");
        const float alpha = 1.f, beta = 0.f;
        s.ran = true;
        try {
            if (s.backend == "cutlass") dense_cutlass_run(s.cutlass, x, w, out, stream);
            else if (s.backend == "cublas") {
                // This handle is private: SetStream resets workspace, so bind
                // the declared zero workspace after each stream handoff.
                blas(cublasSetStream(s.context->handle, stream));
                blas(cublasSetWorkspace(s.context->handle, nullptr, 0));
                const auto dtype = s.out.dtype == at::kHalf ? CUDA_R_16F : CUDA_R_32F;
                blas(cublasGemmEx(s.context->handle, CUBLAS_OP_T, CUBLAS_OP_N, int(s.w.sizes[0]), int(s.x.sizes[0]),
                    int(s.x.sizes[1]), &alpha, w.data_ptr(), CUDA_R_16F, int(s.w.strides[0]),
                    x.data_ptr(), CUDA_R_16F, int(s.x.strides[0]), &beta, out.data_ptr(), dtype,
                    int(s.out.strides[0]), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            } else blas(cublasLtMatmul(s.context->lt, s.operation, &alpha, w.data_ptr(), s.a, x.data_ptr(), s.b,
                &beta, out.data_ptr(), s.d, out.data_ptr(), s.d, &s.algorithm, nullptr, 0, stream));
        } catch (...) { s.failed = true; throw; }
    }
    void warm(const at::Tensor& x, const at::Tensor& w, at::Tensor out) {
        auto& s = live();
        c10::cuda::CUDAGuard guard(s.device);
        const auto stream = at::cuda::getCurrentCUDAStream(s.device.index()).stream();
        uncaptured(stream);
        run(x, w, out);
        const auto status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) s.terminal = true;
        C10_CUDA_CHECK(status);
        s.warmed = true;
        s.x.tensor = at::Tensor(); s.w.tensor = at::Tensor(); s.out.tensor = at::Tensor();
    }
    pybind11::dict settings() const {
        const auto& s = live();
        pybind11::dict d;
        d["backend"] = s.backend; d["workspace_bytes"] = 0; d["warmed"] = s.warmed;
        d["numeric_profile"] = "half-input-float-accumulator-unqualified-order";
        d["rows"] = s.x.sizes[0]; d["columns"] = s.w.sizes[0]; d["reduction"] = s.x.sizes[1];
        d["input_alignment"] = s.alignment[0]; d["weight_alignment"] = s.alignment[1];
        d["output_alignment"] = s.alignment[2];
        return d;
    }
    void close() {
        if (!s_) return;
        auto& s = live();
        c10::cuda::CUDAGuard guard(s.device);
        // The preparation stream may have been retired after positive warmup.
        // Context fences the whole device; only the current capture matters.
        uncaptured(at::cuda::getCurrentCUDAStream(s.device.index()).stream());
        TORCH_CHECK(!s.ran || s.context->retirement_fenced, "dense plans retire after context fence/close");
        // Attempt each independent destroy once. Failed tokens remain retained;
        // successful tokens are cleared. No subsequent retirement retry occurs.
        std::array<cublasStatus_t, 7> statuses{};
        size_t count = 0;
        auto retire = [&](auto& token, auto destroy) {
            if (token) {
                const auto status = destroy(token);
                statuses[count++] = status;
                if (status == CUBLAS_STATUS_SUCCESS) token = nullptr;
            }
        };
        retire(s.preference, cublasLtMatmulPreferenceDestroy);
        retire(s.a, cublasLtMatrixLayoutDestroy); retire(s.b, cublasLtMatrixLayoutDestroy);
        retire(s.d, cublasLtMatrixLayoutDestroy); retire(s.operation, cublasLtMatmulDescDestroy);
        dense_cutlass_destroy(s.cutlass); s.cutlass = nullptr;
        for (size_t i = 0; i < count; ++i) if (statuses[i] != CUBLAS_STATUS_SUCCESS) s.terminal = true;
        TORCH_CHECK(!s.terminal, "dense retirement failed; attempted statuses ",
            int(statuses[0]), ",", int(statuses[1]), ",", int(statuses[2]), ",", int(statuses[3]),
            ",", int(statuses[4]), ",", int(statuses[5]), ",", int(statuses[6]));
        delete s_;
        s_ = nullptr;
    }
};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Context>(m, "Context")
        .def(pybind11::init<int, std::string>()).def("configure", &Context::configure).def("close", &Context::close)
        .def("retirement_fenced", &Context::retirement_fenced)
        .def("handoff_to_current_thread", &Context::handoff_to_current_thread);
    pybind11::class_<Prepared>(m, "Prepared")
        .def(pybind11::init<Context&, at::Tensor, at::Tensor, at::Tensor>())
        .def("configure", &Prepared::configure).def("run", &Prepared::run)
        .def("warm", &Prepared::warm)
        .def("settings", &Prepared::settings).def("close", &Prepared::close);
}
