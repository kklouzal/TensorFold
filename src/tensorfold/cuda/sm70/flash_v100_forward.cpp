#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <memory>
#include <limits>
#include <thread>
#include "flash_v100_forward.cuh"
#include "prefill_dense_contracts.cuh"

namespace {
using T = at::Tensor;
struct State {
    c10::Device device;
    std::array<int64_t, 5> geometry;
    std::array<T, 6> initial;
    std::array<T, 5> workspace; // Compacted K/V, immutable cuQ/cuK, producer-only used counts.
    bool compacted = false;
    int64_t credit = 0;
    float scale;
    std::thread::id thread = std::this_thread::get_id();
    FlashV100Info info{};
    bool configured = false, failed = false, terminal = false, closed = false;
    State(T q, T k, T v, T out, T lse, float scale_): device(q.device()),
        geometry{q.size(0), q.size(1), q.size(2), k.size(1), k.size(2)},
        initial{q, k, v, out, lse, T()}, scale(scale_) {}
};
void uncaptured(cudaStream_t stream) {
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "FA26.10 cold lifecycle requires uncaptured execution");
}
void geometry(int64_t batch, int64_t rows, int64_t heads, int64_t keys, int64_t kvheads) {
    TORCH_CHECK(batch > 0 && batch <= INT_MAX && rows > 0 && rows <= INT_MAX - 32
        && heads > 0 && heads <= INT_MAX && keys > 0 && keys <= INT_MAX - 64 && kvheads > 0
        && kvheads <= heads && heads % kvheads == 0 && batch * heads <= 65535,
        "FA26.10 head256 positive geometry/grid contract (empty shards handled outside this component)");
    // Pinned forward.cu advances K/V as start_kv*(H_K*256) in signed int.
    // H_K*256 is safe after B*Hq<=65535; bound the largest actual tile start
    // before any foreign pointer expression. No giant tensor is needed to
    // exercise this same pre-allocation validator.
    const int64_t last_tile = ((keys - 1) / 64) * 64;
    TORCH_CHECK(last_tile <= INT_MAX / (kvheads * 256), "FA26.10 signed-int KV tile offset overflow");
}
void operands(const T& q, const T& k, const T& v, const T& out, const T& lse) {
    using namespace tensorfold::prefill;
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && out.dim() == 4 && lse.dim() == 3,
        "FA26.10 expects Q/Out[B,M,Hq,256], KV[B,N,Hkv,256], LSE[B,Hq,M]");
    for (const auto* t : {&q, &k, &v, &out}) {
        operand(*t, at::kHalf, q.device());
        TORCH_CHECK(t->is_contiguous() && uintptr_t(t->data_ptr()) % 16 == 0,
            "FA26.10 Half operands require full contiguity and16-byte pointer alignment");
    }
    operand(lse, at::kFloat, q.device());
    TORCH_CHECK(lse.is_contiguous() && uintptr_t(lse.data_ptr()) % alignof(float) == 0, "FA26.10 LSE layout");
    writable(out); writable(lse);
    const int64_t batch = q.size(0), rows = q.size(1), heads = q.size(2), keys = k.size(1), kvheads = k.size(2);
    geometry(batch, rows, heads, keys, kvheads);
    TORCH_CHECK(q.size(3) == 256 && k.size(0) == batch && k.size(3) == 256 && v.sizes() == k.sizes()
        && out.sizes() == q.sizes() && lse.size(0) == batch && lse.size(1) == heads && lse.size(2) == rows,
        "FA26.10 Q/K/V/Out/LSE tensor geometry");
    TORCH_CHECK(!overlaps(out, q) && !overlaps(out, k) && !overlaps(out, v)
        && !overlaps(lse, q) && !overlaps(lse, k) && !overlaps(lse, v) && !overlaps(lse, out),
        "FA26.10 writable spans must be disjoint from inputs and each other");
}
int64_t compacted_bytes(int64_t batch) {
    TORCH_CHECK(batch > 0 && batch <= 256, "FA26.10 compacted rows require1..256");
    // All foreign int intermediates are bounded: TQ<=3072, k_base+start_kv
    // <=256*2051=525056; Hq=Hk=1. Byte arithmetic stays int64_t.
    return 2 * batch * 2051 * 256 * 2 + (3 * batch + 2) * 4;
}
int64_t compacted_operands(const T& q, const T& k, const T& v, const T& valid,
                           const T& out, const T& lse) {
    using namespace tensorfold::prefill;
    TORCH_CHECK(q.dim() == 3 && q.size(0) % 12 == 0 && q.size(1) == 1 && q.size(2) == 256,
        "FA26.10 compacted Q requires[12*B,1,256] fixed head-as-row geometry");
    const int64_t batch = q.size(0) / 12;
    compacted_bytes(batch);
    for (const auto* t : {&q, &k, &v, &out}) {
        operand(*t, at::kHalf, q.device());
        TORCH_CHECK(t->is_contiguous() && uintptr_t(t->data_ptr()) % 16 == 0,
            "FA26.10 compacted Half operands require contiguous16-byte alignment");
    }
    operand(valid, at::kBool, q.device()); operand(lse, at::kFloat, q.device());
    TORCH_CHECK(k.sizes() == at::IntArrayRef({batch, 2051, 1, 256}) && v.sizes() == k.sizes()
        && valid.sizes() == at::IntArrayRef({batch, 1, 2051}) && valid.is_contiguous()
        && out.sizes() == q.sizes() && lse.sizes() == at::IntArrayRef({1, batch * 12})
        && lse.is_contiguous() && uintptr_t(lse.data_ptr()) % alignof(float) == 0,
        "FA26.10 compacted KV/valid/Out/LSE geometry differs");
    writable(out); writable(lse);
    for (const auto* output : {&out, &lse})
        for (const auto* input : {&q, &k, &v, &valid})
            TORCH_CHECK(!overlaps(*output, *input), "FA26.10 compacted output/input overlap");
    TORCH_CHECK(!overlaps(out, lse), "FA26.10 compacted writable spans overlap");
    return batch;
}
class Prepared {
    State* state_ = nullptr;
    State& live(bool control_close = false) const {
        TORCH_CHECK(state_ && !state_->terminal && !state_->closed, "FA26.10 owner closed or retirement uncertain");
        TORCH_CHECK(control_close || state_->thread == std::this_thread::get_id(), "FA26.10 owner thread mismatch");
        return *state_;
    }
    void acquire(T q, T k, T v, T out, T lse, double scale, T valid = T(), int64_t credit = 0) {
        const bool compacted = valid.defined();
        int64_t batch = 0;
        if (compacted) {
            batch = compacted_operands(q, k, v, valid, out, lse);
            TORCH_CHECK(credit >= compacted_bytes(batch), "FA26.10 compacted tensor byte credit exhausted");
        } else operands(q, k, v, out, lse);
        TORCH_CHECK(std::isfinite(scale) && scale > 0 && scale <= std::numeric_limits<float>::max()
            && float(scale) > 0 && (!compacted || scale == .0625), "FA26.10 explicit score scale contract");
        auto state = std::make_unique<State>(q, k, v, out, lse, float(scale));
        state->compacted = compacted; state->credit = credit;
        if (compacted) { state->geometry = {batch, 12, 1, 2051, 1}; state->initial[5] = valid; }
        c10::cuda::CUDAGuard guard(state->device);
        const auto* prop = at::cuda::getDeviceProperties(state->device.index());
        TORCH_CHECK(prop->major == 7 && prop->minor == 0, "FA26.10 component requires actual SM70");
        uncaptured(at::cuda::getCurrentCUDAStream(state->device.index()).stream());
        state_ = state.release(); // Journal exists before configure acquires device resources.
    }
public:
    Prepared(T q, T k, T v, T out, T lse, double scale) {
        acquire(q, k, v, out, lse, scale);
    }
    Prepared(T q, T k, T v, T out, T lse, double scale, T valid, int64_t credit) {
        acquire(q, k, v, out, lse, scale, valid, credit);
    }
    Prepared(const Prepared&) = delete;
    Prepared& operator=(const Prepared&) = delete;
    ~Prepared() noexcept {
        if (!state_) return;
        if (state_->closed) delete state_;
        else std::fputs("FA26.10: retaining unretired owner and initial loans\n", stderr);
    }
    void configure() {
        auto& s = live();
        TORCH_CHECK(!s.configured && !s.failed, "FA26.10 configure state");
        c10::cuda::CUDAGuard guard(s.device);
        uncaptured(at::cuda::getCurrentCUDAStream(s.device.index()).stream());
        try {
            if (s.compacted) {
                const auto batch = s.geometry[0];
                s.workspace[0] = at::empty({batch * 2051, 1, 256}, s.initial[0].options());
                s.workspace[1] = at::empty_like(s.workspace[0]);
                s.workspace[2] = at::empty({batch + 1}, s.initial[0].options().dtype(at::kInt));
                s.workspace[3] = at::empty_like(s.workspace[2]);
                s.workspace[4] = at::empty({batch}, s.initial[0].options().dtype(at::kInt));
                flash_v100_prepare_varlen_kernel(s.info);
                flash_v100_init_compact(s.workspace[0], s.workspace[1], s.workspace[2], s.workspace[3],
                    s.workspace[4], at::cuda::getCurrentCUDAStream(s.device.index()).stream());
            } else flash_v100_prepare_kernel(s.info);
            s.configured = true;
        }
        catch (...) { s.failed = true; throw; }
    }
    void run(const T& q, const T& k, const T& v, T out, T lse) {
        auto& s = live();
        TORCH_CHECK(s.configured && !s.failed && !s.compacted, "FA26.10 unconfigured/failed or wrong-mode owner");
        operands(q, k, v, out, lse);
        TORCH_CHECK(q.device() == s.device && s.geometry == (std::array<int64_t, 5>{
            q.size(0), q.size(1), q.size(2), k.size(1), k.size(2)}), "FA26.10 prepared geometry changed");
        c10::cuda::CUDAGuard guard(s.device);
        try { flash_v100_launch(q, k, v, out, lse, s.scale,
            at::cuda::getCurrentCUDAStream(s.device.index()).stream()); }
        catch (...) { s.failed = true; throw; }
    }
    void run_compacted(const T& q, const T& k, const T& v, const T& valid, T out, T lse) {
        auto& s = live();
        TORCH_CHECK(s.configured && !s.failed && s.compacted, "FA26.10 compacted owner state");
        const auto batch = compacted_operands(q, k, v, valid, out, lse);
        TORCH_CHECK(q.device() == s.device && batch == s.geometry[0], "FA26.10 compacted prepared geometry changed");
        c10::cuda::CUDAGuard guard(s.device);
        const auto stream = at::cuda::getCurrentCUDAStream(s.device.index()).stream();
        try {
            flash_v100_compact(k, v, valid, s.workspace[0], s.workspace[1], s.workspace[4], stream);
            flash_v100_launch_varlen(q, s.workspace[0], s.workspace[1], s.workspace[2], s.workspace[3],
                s.workspace[4], out, lse, stream);
        } catch (...) { s.failed = true; throw; }
    }
    void warm() {
        auto& s = live();
        c10::cuda::CUDAGuard guard(s.device);
        const auto stream = at::cuda::getCurrentCUDAStream(s.device.index()).stream();
        uncaptured(stream);
        if (s.compacted) {
            TORCH_CHECK(s.configured && !s.failed, "FA26.10 compacted cold warm state");
            // Warm every owned producer too, then restore initialized private
            // one-key data to exercise WMMA even when the initial mask is empty.
            flash_v100_compact(s.initial[1], s.initial[2], s.initial[5],
                s.workspace[0], s.workspace[1], s.workspace[4], stream);
            flash_v100_init_compact(s.workspace[0], s.workspace[1], s.workspace[2], s.workspace[3],
                s.workspace[4], stream);
            flash_v100_launch_varlen(s.initial[0], s.workspace[0], s.workspace[1], s.workspace[2],
                s.workspace[3], s.workspace[4], s.initial[3], s.initial[4], stream);
        } else run(s.initial[0], s.initial[1], s.initial[2], s.initial[3], s.initial[4]);
        const auto status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) s.terminal = true;
        C10_CUDA_CHECK(status);
        s.initial = {};
    }
    void handoff_to_current_thread() {
        auto& s = live(true);
        c10::cuda::CUDAGuard guard(s.device);
        uncaptured(at::cuda::getCurrentCUDAStream(s.device.index()).stream());
        const auto status = cudaDeviceSynchronize();
        if (status != cudaSuccess) s.terminal = true;
        C10_CUDA_CHECK(status);
        s.thread = std::this_thread::get_id();
    }
    void close() {
        if (!state_ || state_->closed) return;
        auto& s = live(true);
        c10::cuda::CUDAGuard guard(s.device);
        uncaptured(at::cuda::getCurrentCUDAStream(s.device.index()).stream());
        const auto status = cudaDeviceSynchronize();
        if (status != cudaSuccess) s.terminal = true;
        C10_CUDA_CHECK(status);
        s.initial = {};
        s.workspace = {};
        s.closed = true;
    }
    pybind11::dict settings() const {
        const auto& s = live();
        pybind11::dict d;
        d["source_commit"] = "0ababf82e3836c1786d7937c6002670ca4ecda62";
        d["shared_bytes"] = s.info.shared_bytes; d["threads"] = s.info.threads;
        d["registers_per_thread"] = s.info.attributes.numRegs;
        d["binary_version"] = s.info.attributes.binaryVersion;
        d["compacted"] = s.compacted;
        d["private_tensor_bytes"] = s.compacted ? compacted_bytes(s.geometry[0]) : 0;
        d["tensor_credit_bytes"] = s.credit;
        d["compactor_shared_bytes"] = s.compacted ? (256 + 2051) * 4 : 0;
        d["numeric_profile"] = s.compacted ?
            "fa-v10026.10-varlen-stable-compact2051-fixed12-query-rows-fastmath-half-output-float-lse-unqualified" : "fa-v10026.10-custom-mma161616-fastmath-local-half-output-float-lse-unqualified";
        return d;
    }
};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compacted_workspace_bytes", &compacted_bytes);
    m.def("validate_geometry", &geometry); // Actual pre-allocation/operand boundary, no CUDA calls.
    pybind11::class_<Prepared>(m, "Prepared")
        .def(pybind11::init<T, T, T, T, T, double>())
        .def(pybind11::init<T, T, T, T, T, double, T, int64_t>()).def("configure", &Prepared::configure)
        .def("warm", &Prepared::warm).def("run", &Prepared::run)
        .def("run_compacted", &Prepared::run_compacted).def("close", &Prepared::close)
        .def("handoff_to_current_thread", &Prepared::handoff_to_current_thread).def("settings", &Prepared::settings);
}
