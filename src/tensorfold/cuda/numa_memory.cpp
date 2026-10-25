// Owned anonymous host ranges. No GPU API, process policy or page migration.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cerrno>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <vector>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace py = pybind11;

class Budget {
    size_t limit_;
    std::atomic<size_t> held_{0};
public:
    explicit Budget(size_t limit) : limit_(limit) {}
    size_t held() const { return held_.load(std::memory_order_acquire); }
    void take(size_t bytes) {
        size_t old = held();
        do {
            if (old > limit_ || bytes > limit_-old) throw std::bad_alloc();
        } while (!held_.compare_exchange_weak(old, old+bytes, std::memory_order_acq_rel));
    }
    void give(size_t bytes) noexcept {
        if (held_.fetch_sub(bytes, std::memory_order_acq_rel) < bytes) std::terminate();
    }
};

class Mapping {
    void* reservation_ = MAP_FAILED;
    unsigned char* data_ = nullptr;
    size_t extent_ = 0, reserved_ = 0;
    std::shared_ptr<Budget> budget_;
    bool charged_ = false;
public:
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Mapping(size_t bytes, int node, std::shared_ptr<Budget> budget) : budget_(std::move(budget)) {
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0 || !bytes || node < 0 || node >= int(sizeof(unsigned long)*8) ||
            bytes > size_t(std::numeric_limits<py::ssize_t>::max()) - 3*size_t(page))
            throw std::invalid_argument("NUMA mapping requires bounded positive bytes and a node within the native mask");
        extent_ = (bytes + size_t(page)-1)/size_t(page)*size_t(page);
        reserved_ = extent_ + 2*size_t(page);
        if (budget_) { budget_->take(reserved_); charged_ = true; }
        reservation_ = mmap(nullptr, reserved_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (reservation_ == MAP_FAILED) {
            const int primary = errno;
            release();
            throw std::system_error(primary, std::generic_category(), "NUMA mmap");
        }
        data_ = static_cast<unsigned char*>(reservation_) + page;
        unsigned long mask = 1ul << node;
        if (mprotect(data_, extent_, PROT_READ | PROT_WRITE) ||
            syscall(SYS_mbind, data_, extent_, long(MPOL_BIND | MPOL_F_STATIC_NODES),
                    &mask, sizeof(mask)*8ul, 0ul)) {
            const int primary = errno;
            release();
            throw std::system_error(primary, std::generic_category(), "NUMA mmap/mbind before first touch");
        }
        // Writing every byte both initializes the exported data and faults
        // private physical pages only after the range policy is installed.
        std::memset(data_, 0, extent_);
    }
    ~Mapping() noexcept { release(); }
    void release() noexcept {
        if (reservation_ != MAP_FAILED) {
            if (munmap(reservation_, reserved_)) {
                std::fprintf(stderr, "TensorFold: owned NUMA munmap failed (%d); stopping unsafe retirement\n", errno);
                std::terminate();
            }
            reservation_ = MAP_FAILED;
            data_ = nullptr;
        }
        if (charged_) { budget_->give(reserved_); charged_ = false; }
    }
    uintptr_t address() const { return reinterpret_cast<uintptr_t>(data_); }
    size_t extent() const { return extent_; }
    size_t reserved() const { return reserved_; }
    py::buffer_info buffer() {
        return py::buffer_info(data_, sizeof(unsigned char), py::format_descriptor<unsigned char>::format(),
                               1, {py::ssize_t(extent_)}, {1});
    }
    std::map<int, size_t> nodes() const {
        const long page = sysconf(_SC_PAGESIZE);
        std::vector<void*> addresses(4096);
        std::vector<int> status(4096);
        std::map<int, size_t> result;
        const size_t count = extent_/size_t(page);
        for (size_t first = 0; first < count; first += addresses.size()) {
            const size_t number = std::min(addresses.size(), count-first);
            for (size_t i = 0; i < number; ++i) addresses[i] = data_+(first+i)*size_t(page);
            const long queried = syscall(SYS_move_pages, 0l, number, addresses.data(),
                                         static_cast<const int*>(nullptr), status.data(), 0ul);
            if (queried < 0)
                throw std::system_error(errno, std::generic_category(), "NUMA move_pages self-query");
            for (size_t i = 0; i < number; ++i) {
                if (status[i] < 0) throw std::system_error(-status[i], std::generic_category(), "NUMA page placement");
                ++result[status[i]];
            }
            if (queried) throw std::system_error(EIO, std::generic_category(), "NUMA incomplete page query");
        }
        return result;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::register_local_exception_translator([](std::exception_ptr error) {
        try { if (error) std::rethrow_exception(error); }
        catch (const std::system_error& failure) {
            py::tuple args = py::make_tuple(failure.code().value(), failure.what());
            PyErr_SetObject(PyExc_OSError, args.ptr());
        }
    });
    py::class_<Budget, std::shared_ptr<Budget>>(m, "Budget")
        .def(py::init<size_t>()).def_property_readonly("held", &Budget::held);
    py::class_<Mapping, std::shared_ptr<Mapping>>(m, "Mapping", py::buffer_protocol())
        .def(py::init<size_t, int, std::shared_ptr<Budget>>(), py::arg("bytes"), py::arg("node"),
             py::arg("budget") = nullptr, py::call_guard<py::gil_scoped_release>())
        .def_buffer(&Mapping::buffer)
        .def_property_readonly("address", &Mapping::address)
        .def_property_readonly("extent", &Mapping::extent)
        .def_property_readonly("reserved", &Mapping::reserved)
        .def("nodes", &Mapping::nodes, py::call_guard<py::gil_scoped_release>());
}
