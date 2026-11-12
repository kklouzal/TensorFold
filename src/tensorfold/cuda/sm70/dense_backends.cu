#include "dense_backends.cuh"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/version.h>
#include <memory>
#include <cstdint>
static_assert(CUTLASS_MAJOR == 4 && CUTLASS_MINOR == 8 && CUTLASS_PATCH == 1,
              "dense candidate requires protected CUTLASS4.8.1");
static_assert(sizeof(cutlass::half_t) == sizeof(at::Half)
              && alignof(at::Half) >= alignof(cutlass::half_t), "binary16 ABI");

namespace {
template<class T> using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    T, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<T, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 8, 8>;
void check(cutlass::Status s) {
    TORCH_CHECK(s == cutlass::Status::kSuccess, "dense CUTLASS status ", int(s));
}
}
struct DenseCutlassPlan {
    virtual ~DenseCutlassPlan() = default;
    virtual void run(const at::Tensor&, const at::Tensor&, at::Tensor, cudaStream_t) = 0;
};
namespace {
template<class T> struct Plan final : DenseCutlassPlan {
    Gemm<T> operation;
    static typename Gemm<T>::Arguments arguments(const at::Tensor& x, const at::Tensor& w, at::Tensor out) {
        using Kernel = typename Gemm<T>::GemmKernel;
        using AccessA = typename Kernel::Mma::IteratorA::AccessType;
        using AccessB = typename Kernel::Mma::IteratorB::AccessType;
        // CUTLASS4.8.1 TensorRef_aligned checks pointer bytes modulo the
        // element count (8), not the AlignedArray's byte alignment (16).
        // Validate the actual typed-access precondition before constructing refs.
        TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % alignof(AccessA) == 0
            && reinterpret_cast<uintptr_t>(w.data_ptr()) % alignof(AccessB) == 0,
            "dense CUTLASS input pointer alignment");
        auto* d = reinterpret_cast<T*>(out.data_ptr());
        return typename Gemm<T>::Arguments({int(x.size(0)), int(w.size(0)), int(x.size(1))},
            {reinterpret_cast<const cutlass::half_t*>(x.data_ptr()), int(x.stride(0))},
            {reinterpret_cast<const cutlass::half_t*>(w.data_ptr()), int(w.stride(0))},
            {d, int(out.stride(0))}, {d, int(out.stride(0))}, {1.f, 0.f}, 1);
    }
    Plan(const at::Tensor& x, const at::Tensor& w, at::Tensor out, cudaStream_t stream) {
        auto args = arguments(x, w, out);
        check(Gemm<T>::can_implement(args));
        TORCH_CHECK(Gemm<T>::get_workspace_size(args) == 0, "dense CUTLASS requires unadmitted workspace");
        check(operation.initialize(args, nullptr, stream));
    }
    void run(const at::Tensor& x, const at::Tensor& w, at::Tensor out, cudaStream_t stream) override {
        auto args = arguments(x, w, out);
        check(Gemm<T>::can_implement(args));
        check(operation.update(args, nullptr));
        check(operation.run(stream));
    }
};
}
DenseCutlassPlan* dense_cutlass_prepare(const at::Tensor& x, const at::Tensor& w, at::Tensor out,
                                      cudaStream_t stream) {
    if (out.scalar_type() == at::kHalf) return new Plan<cutlass::half_t>(x, w, out, stream);
    return new Plan<float>(x, w, out, stream);
}
void dense_cutlass_run(DenseCutlassPlan* p, const at::Tensor& x, const at::Tensor& w, at::Tensor out,
                       cudaStream_t stream) { p->run(x, w, out, stream); }
void dense_cutlass_destroy(DenseCutlassPlan* p) noexcept { delete p; }
