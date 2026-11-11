#include "prefill_dense_simt.cuh"
#include <c10/cuda/CUDAException.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <memory>

using Output = cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float,
    cutlass::epilogue::thread::ScaleType::Nothing, cutlass::FloatRoundStyle::round_to_nearest>;
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>, Output,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false,
    cutlass::arch::OpMultiplyAdd>;
using Kernel = Gemm::GemmKernel;
static_assert(!Gemm::kSplitKSerial && Gemm::ThreadblockShape::kK == Gemm::WarpShape::kK,
              "the certified candidate has no block/warp split-K reduction");
static_assert(sizeof(cutlass::half_t) == sizeof(at::Half), "binary16 ABI");
static_assert(alignof(at::Half) >= alignof(cutlass::half_t), "binary16 alignment ABI");

struct PrefillSimtPlan { Gemm operation; };
namespace {
void check(cutlass::Status status) {
    TORCH_CHECK(status == cutlass::Status::kSuccess, "prefill SIMT CUTLASS status ", int(status));
}
}

PrefillSimtPlan* prefill_simt_prepare(const at::Tensor& x, const at::Tensor& weight, at::Tensor out,
                                    cudaStream_t stream, PrefillSimtInfo& info) {
    const auto* a = reinterpret_cast<const cutlass::half_t*>(x.data_ptr<at::Half>());
    const auto* b = reinterpret_cast<const cutlass::half_t*>(weight.data_ptr<at::Half>());
    auto* d = reinterpret_cast<cutlass::half_t*>(out.data_ptr<at::Half>());
    Gemm::Arguments args({int(x.size(0)), int(weight.size(0)), int(x.size(1))},
        {a, int(x.stride(0))}, {b, int(weight.stride(0))},
        {d, int(out.stride(0))}, {d, int(out.stride(0))}, Output::Params(1.f, 0.f), 1);
    check(Gemm::can_implement(args));
    info.workspace_bytes = Gemm::get_workspace_size(args);
    TORCH_CHECK(info.workspace_bytes == 0, "non-splitK SIMT unexpectedly requires workspace");
    info.shared_storage_bytes = sizeof(Kernel::SharedStorage);
    info.threads = Kernel::kThreadCount;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&info.attributes, cutlass::Kernel<Kernel>));
    if (info.shared_storage_bytes >= (48u << 10)) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(info.shared_storage_bytes)));
    }
    auto plan = std::make_unique<PrefillSimtPlan>();
    // SplitK=false/slices1 makes initialize host-only; there is no semaphore memset.
    check(plan->operation.initialize(args, nullptr, stream));
    return plan.release();
}
void prefill_simt_run(PrefillSimtPlan* plan, cudaStream_t stream) {
    check(plan->operation.run(stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void prefill_simt_destroy(PrefillSimtPlan* plan) noexcept { delete plan; }
