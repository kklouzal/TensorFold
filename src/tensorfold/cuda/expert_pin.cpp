#include <torch/extension.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>

// Explicit model-owned registration. The Python policy publishes this owner
// before calling pin(), so interrupts cannot strand a successful registration.
class Registration {
    at::Tensor storage_;
    bool active_ = false;
    int close_error_ = 0;
public:
    explicit Registration(at::Tensor storage) : storage_(std::move(storage)) {
        const long page = sysconf(_SC_PAGESIZE);
        TORCH_CHECK(page > 0 && storage_.device().is_cpu() && storage_.scalar_type() == at::kByte &&
                    storage_.dim() == 1 && storage_.is_contiguous() && storage_.numel() > 0,
                    "host registration requires a positive contiguous CPU byte vector");
        const auto address = reinterpret_cast<uintptr_t>(storage_.data_ptr());
        TORCH_CHECK(address % page == 0 && storage_.numel() % page == 0 &&
                    static_cast<uint64_t>(storage_.numel()) <= std::numeric_limits<uintptr_t>::max() - address,
                    "host registration must own complete aligned pages");
    }
    int pin() {
        TORCH_CHECK(!active_ && close_error_ == 0, "host registration already active or poisoned");
        const auto status = cudaHostRegister(storage_.data_ptr(), storage_.numel(), cudaHostRegisterPortable);
        active_ = status == cudaSuccess;
        return static_cast<int>(status);
    }
    int close() {
        if (close_error_) return close_error_; // an ambiguous failed cleanup is not retried
        if (!active_) return 0;
        const auto status = cudaHostUnregister(storage_.data_ptr());
        if (status == cudaSuccess) {
            active_ = false;
            storage_ = at::Tensor();
        } else {
            close_error_ = static_cast<int>(status);
        }
        return static_cast<int>(status);
    }
    bool active() const { return active_; }
    ~Registration() noexcept {
        // Reclaiming still-registered CPU storage could race a DMA consumer.
        // Explicit close (after quiescence) is mandatory; abandoning a failed
        // owner is a fatal lifecycle violation rather than unsafe reclamation.
        if (active_) {
            std::fputs("TensorFold: unretired expert host registration\n", stderr);
            std::terminate();
        }
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Registration>(m, "Registration")
        .def(pybind11::init<at::Tensor>()).def("pin", &Registration::pin)
        .def("close", &Registration::close).def_property_readonly("active", &Registration::active);
    m.attr("memory_allocation") = static_cast<int>(cudaErrorMemoryAllocation);
    m.attr("not_supported") = static_cast<int>(cudaErrorNotSupported);
    m.def("error_text", [](int code) { return cudaGetErrorString(static_cast<cudaError_t>(code)); });
}
