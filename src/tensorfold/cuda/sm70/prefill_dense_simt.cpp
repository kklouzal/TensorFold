#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <array>
#include <climits>
#include <cstdio>
#include <memory>
#include "prefill_dense_contracts.cuh"
#include "prefill_dense_simt.cuh"

namespace {
using tensorfold::prefill::operand;
using tensorfold::prefill::writable;
using tensorfold::prefill::overlaps;
struct Binding {
    at::Tensor tensor;
    const void* pointer;
    int64_t rows, columns, stride;
    explicit Binding(const at::Tensor& value): tensor(value), pointer(value.data_ptr()),
        rows(value.size(0)), columns(value.size(1)), stride(value.stride(0)) {}
    bool matches(const at::Tensor& value) const {
        return value.dim() == 2 && value.data_ptr() == pointer && value.size(0) == rows
            && value.size(1) == columns && value.stride(0) == stride && value.stride(1) == 1;
    }
};
void validate(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& out) {
    const auto device = x.device();
    operand(x, at::kHalf, device); operand(weight, at::kHalf, device); operand(out, at::kHalf, device);
    writable(out);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && out.dim() == 2
        && x.size(0) > 0 && x.size(0) <= 4096 && x.size(1) > 0 && x.size(1) <= 10240
        && weight.size(0) > 0 && weight.size(0) <= 10240 && weight.size(1) == x.size(1)
        && out.size(0) == x.size(0) && out.size(1) == weight.size(0), "prefill SIMT geometry/finite-proof region");
    TORCH_CHECK(x.stride(1) == 1 && x.stride(0) >= x.size(1) && x.stride(0) <= INT_MAX
        && weight.is_contiguous() && out.stride(1) == 1 && out.stride(0) >= out.size(1)
        && out.stride(0) <= INT_MAX, "prefill SIMT row strides");
    TORCH_CHECK(!overlaps(out, x) && !overlaps(out, weight), "prefill SIMT output aliases a read operand");
}
struct State {
    Binding x, weight, out;
    c10::Device device;
    cudaStream_t stream;
    PrefillSimtPlan* plan = nullptr;
    PrefillSimtInfo info{};
    bool configured = false, pending = false, failed = false, terminal = false, closed = false;
    State(const at::Tensor& a, const at::Tensor& b, const at::Tensor& d):
        x(a), weight(b), out(d), device(a.device()) {}
};
class Prepared {
    State* state_ = nullptr;
    State& live() const {
        TORCH_CHECK(state_ && !state_->closed && !state_->terminal,
                    "prefill SIMT is closed or retirement is uncertain");
        return *state_;
    }
public:
    Prepared(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& out) {
        validate(x, weight, out);
        auto pending = std::make_unique<State>(x, weight, out);
        c10::cuda::CUDAGuard guard(pending->device);
        const auto* prop = at::cuda::getDeviceProperties(pending->device.index());
        TORCH_CHECK(prop->major == 7 && prop->minor == 0, "prefill SIMT requires SM70");
        pending->stream = at::cuda::getCurrentCUDAStream(pending->device.index()).stream();
        state_ = pending.release();
    }
    Prepared(const Prepared&) = delete;
    Prepared& operator=(const Prepared&) = delete;
    ~Prepared() noexcept {
        if (!state_) return;
        if (!state_->closed) {
            // Do not drop bound GPU storage or retry uncertain CUDA cleanup.
            std::fputs("prefill SIMT: retaining unretired plan and bound operands\n", stderr);
        } else delete state_;
    }
    void configure() {
        auto& s = live();
        TORCH_CHECK(!s.configured, "prefill SIMT already configured");
        c10::cuda::CUDAGuard guard(s.device);
        s.plan = prefill_simt_prepare(s.x.tensor, s.weight.tensor, s.out.tensor, s.stream, s.info);
        s.configured = true;
    }
    void run(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& out) {
        auto& s = live();
        TORCH_CHECK(s.configured && !s.pending && !s.failed, "prefill SIMT unconfigured/pending/failed");
        validate(x, weight, out);
        TORCH_CHECK(s.x.matches(x) && s.weight.matches(weight) && s.out.matches(out), "prefill SIMT binding changed");
        c10::cuda::CUDAGuard guard(s.device);
        TORCH_CHECK(at::cuda::getCurrentCUDAStream(s.device.index()).stream() == s.stream,
                    "prefill SIMT stream mismatch");
        cudaStreamCaptureStatus capture;
        C10_CUDA_CHECK(cudaStreamIsCapturing(s.stream, &capture));
        TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "prefill SIMT component requires uncaptured execution");
        s.pending = true;
        try { prefill_simt_run(s.plan, s.stream); }
        catch (...) { s.failed = true; throw; }
    }
    void finish() {
        auto& s = live();
        c10::cuda::CUDAGuard guard(s.device);
        const auto status = cudaStreamSynchronize(s.stream);
        if (status != cudaSuccess) s.terminal = true;
        C10_CUDA_CHECK(status);
        s.pending = false;
    }
    void close() {
        if (!state_ || state_->closed) return;
        auto& s = live();
        c10::cuda::CUDAGuard guard(s.device);
        const auto status = cudaStreamSynchronize(s.stream);
        if (status != cudaSuccess) s.terminal = true;
        C10_CUDA_CHECK(status);
        prefill_simt_destroy(s.plan);
        s.plan = nullptr;
        s.x.tensor = s.weight.tensor = s.out.tensor = at::Tensor();
        s.closed = true;
    }
    pybind11::dict settings() const {
        const auto& s = live();
        TORCH_CHECK(s.configured, "prefill SIMT unconfigured");
        pybind11::dict d;
        d["operation"] = "cutlass-simt-sm70-f32-scalar-rn16-v1";
        d["cutlass_source_commit"] = "2c85f497547eb441f0fe80bfb0fb2e6f9add1719";
        d["epilogue"] = "ScaleTypeNothing-RN16";
        d["workspace_bytes"] = s.info.workspace_bytes;
        d["additional_device_tensor_bytes"] = 0;
        d["shared_storage_bytes"] = s.info.shared_storage_bytes;
        d["threads"] = s.info.threads;
        d["registers"] = s.info.attributes.numRegs;
        d["local_bytes_per_thread"] = s.info.attributes.localSizeBytes;
        d["constant_bytes"] = s.info.attributes.constSizeBytes;
        d["device"] = s.device.index();
        d["split_k_slices"] = 1;
        d["warp_k_partitions"] = 1;
        d["instruction_shape"] = pybind11::make_tuple(1, 1, 1);
        d["threadblock_shape"] = pybind11::make_tuple(128, 128, 8);
        d["warp_shape"] = pybind11::make_tuple(32, 64, 8);
        d["emitted_RN32_DAG_verified"] = false; // Source selection is not binary verification.
        return d;
    }
};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Prepared>(m, "Prepared").def(pybind11::init<const at::Tensor&, const at::Tensor&, const at::Tensor&>())
        .def("configure", &Prepared::configure).def("run", &Prepared::run).def("finish", &Prepared::finish)
        .def("close", &Prepared::close).def("settings", &Prepared::settings);
}
