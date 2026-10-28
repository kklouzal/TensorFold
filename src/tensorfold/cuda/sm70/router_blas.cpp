#include <torch/extension.h>
#include <ATen/Context.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <optional>
#include <memory>

namespace {
constexpr int K = 2560, E = 513, Y_STRIDE = 516, MAX_ROWS = 32;
static_assert(sizeof(void*) == sizeof(int64_t), "private CUDA pointer-table target is64-bit");

void status(cublasStatus_t value, const char* operation) {
    TORCH_CHECK(value == CUBLAS_STATUS_SUCCESS, operation, " cuBLAS status ", int(value));
}

class RouterPlan {
    at::Tensor weight_, weight_storage_, input_, output_, a_, x_, y_;
    int rows_;
    uintptr_t weight_address_;
    std::optional<c10::cuda::CUDAStream> stream_;

    static at::Tensor pointers(const at::Tensor& tensor, int rows, int stride, bool shared) {
        auto host = at::empty({rows}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
        for (int row = 0; row < rows; ++row) {
            const float* value = tensor.data_ptr<float>() + (shared ? 0 : int64_t(row) * stride);
            // Transfer the CUDA pointer representation, not a signed-address cast.
            std::memcpy(host.data_ptr<int64_t>() + row, &value, sizeof(value));
        }
        // Preparation is cold and blocking: host pointer bytes retire before
        // their local owner is released. No pointer copy occurs in run().
        return host.to(tensor.device(), at::kLong, false, true);
    }

public:
    RouterPlan(const at::Tensor& weight, int rows, std::shared_ptr<RouterPlan> shared = nullptr)
        : weight_(weight), rows_(rows), weight_address_(0) {
        TORCH_CHECK(weight.is_cuda() && weight.layout() == at::kStrided
                    && weight.scalar_type() == at::kFloat && weight.dim() == 2
                    && weight.size(0) == E && weight.size(1) == K && weight.is_contiguous(),
                    "router batch requires contiguous CUDA FP32[513,2560] weight");
        TORCH_CHECK(rows > 0 && rows <= MAX_ROWS, "router batch capacity must be1..32");
        TORCH_CHECK(!weight.requires_grad(), "private router batch weight is inference-only");
        weight_address_ = reinterpret_cast<uintptr_t>(weight.data_ptr());
        TORCH_CHECK(weight_address_ % 16 == 0, "router batch weight requires16-byte alignment");
        c10::cuda::CUDAGuard device(weight.device());
        const auto* properties = at::cuda::getDeviceProperties(weight.get_device());
        TORCH_CHECK(properties->major == 7 && properties->minor == 0, "private router batch requires SM70");
        stream_ = at::cuda::getCurrentCUDAStream(weight.get_device());
        weight_storage_ = weight.view_as(weight);  // Retain original storage if caller later rebinds weight_.
        if (shared) {
            TORCH_CHECK(shared->rows_ == rows_ && shared->weight_.device() == weight.device()
                        && shared->stream_ == stream_, "router shared workspace geometry/stream differs");
            input_ = shared->input_; output_ = shared->output_;
            x_ = shared->x_; y_ = shared->y_;
        } else {
            input_ = at::empty({rows_, K}, weight.options());
            output_ = at::empty({rows_, Y_STRIDE}, weight.options());
            TORCH_CHECK(reinterpret_cast<uintptr_t>(input_.data_ptr()) % 16 == 0
                        && reinterpret_cast<uintptr_t>(output_.data_ptr()) % 16 == 0, "router vector alignment");
            x_ = pointers(input_, rows_, K, false);
            y_ = pointers(output_, rows_, Y_STRIDE, false);
        }
        a_ = pointers(weight_, rows_, 0, true);
        at::NoTF32Guard no_tf32;
        (void)at::cuda::getCurrentCUDABlasHandle();
    }

    at::Tensor run(const at::Tensor& half, at::Tensor out) {
        TORCH_CHECK(half.is_cuda() && half.layout() == at::kStrided && half.device() == weight_.device() && half.scalar_type() == at::kHalf
                    && half.dim() == 2 && half.size(1) == K && half.size(0) > 0 && half.size(0) <= rows_,
                    "router batch input requires1..capacity FP16 rows of2560");
        TORCH_CHECK(out.layout() == at::kStrided && out.device() == half.device() && out.scalar_type() == at::kFloat && out.dim() == 2
                    && out.size(0) == half.size(0) && out.size(1) == E && out.is_contiguous(),
                    "router batch output requires contiguous matching FP32 rows of513");
        TORCH_CHECK(!half.requires_grad() && !weight_.requires_grad() && !out.requires_grad(),
                    "private router batch is inference-only");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(weight_.data_ptr()) == weight_address_
                    && weight_.sizes() == at::IntArrayRef({E,K}) && weight_.is_contiguous(),
                    "router batch weight storage/geometry changed after preparation");
        at::assert_no_overlap(out, half); at::assert_no_overlap(out, weight_storage_);
        c10::cuda::CUDAGuard device(half.device());
        TORCH_CHECK(at::cuda::getCurrentCUDAStream(half.get_device()) == *stream_,
                    "router batch scratch belongs to its construction stream");
        at::NoTF32Guard no_tf32;
        auto handle = at::cuda::getCurrentCUDABlasHandle();
        cublasPointerMode_t pointer_mode;
        status(cublasGetPointerMode(handle, &pointer_mode), "get pointer mode");
        TORCH_CHECK(pointer_mode == CUBLAS_POINTER_MODE_HOST, "router batch requires existing host pointer mode");
        cudaStream_t handle_stream;
        status(cublasGetStream(handle, &handle_stream), "get stream");
        TORCH_CHECK(handle_stream == stream_->stream(), "cuBLAS handle stream differs from prepared router stream");
        const int count = int(half.size(0));
        input_.narrow(0, 0, count).copy_(half);  // Exact FP16→FP32 conversion, into prepaid storage.
        const float alpha = 1.f, beta = 0.f;
        // Row-major[E,K] is column-major[K,E]; transpose reproduces a row GEMV.
        status(cublasSgemvBatched(handle, CUBLAS_OP_T, K, E, &alpha,
            reinterpret_cast<const float* const*>(a_.data_ptr<int64_t>()), K,
            reinterpret_cast<const float* const*>(x_.data_ptr<int64_t>()), 1, &beta,
            reinterpret_cast<float* const*>(y_.data_ptr<int64_t>()), 1, count), "SgemvBatched");
        out.copy_(output_.narrow(0,0,count).narrow(1,0,E));
        return out;
    }

    int64_t gpu_bytes() const {
        return (int64_t(rows_) * (K + Y_STRIDE) * sizeof(float)
                + int64_t(3) * rows_ * sizeof(void*));
    }
};
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    pybind11::class_<RouterPlan, std::shared_ptr<RouterPlan>>(module, "RouterPlan")
        .def(pybind11::init<const at::Tensor&, int, std::shared_ptr<RouterPlan>>(),
             pybind11::arg("weight"), pybind11::arg("rows"), pybind11::arg("shared") = nullptr)
        .def("run", &RouterPlan::run)
        .def("gpu_bytes", &RouterPlan::gpu_bytes);
}
