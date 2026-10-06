// Linux ARM64 buffered pread executor. Python owns and keeps FDs open until close drains.
// Caller must exclusively own the writable output throughout read(); descriptors are copied.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
namespace py = pybind11;
struct Read { int fd; off_t offset; size_t size, at; };
struct Failure { int code=0, fd=-1; off_t offset=0; bool eof=false, closed=false; };
class Reader {
    struct Worker { std::condition_variable cv; bool ready=false; std::thread thread; };
    const size_t limit;
    std::vector<std::unique_ptr<Worker>> workers;
    std::mutex batch, state, error;
    std::condition_variable drained;
    bool stopping=false, closed=false;
    size_t remaining=0;
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    const std::vector<Read>* jobs=nullptr;
    unsigned char* output=nullptr;
    Failure failure;
    void consume() noexcept {
        while (!failed.load(std::memory_order_relaxed)) {
            size_t i=next.fetch_add(1, std::memory_order_relaxed);
            if (i>=jobs->size()) return;
            const auto& r=(*jobs)[i];
            size_t done=0;
            while (done<r.size) {
                ssize_t n=::pread(r.fd, output+r.at+done, r.size-done, r.offset+done);
                if (n>0) { done+=static_cast<size_t>(n); continue; }
                int code=(n<0) ? errno : 0;
                if (n<0 && code==EINTR) continue;
                std::lock_guard<std::mutex> guard(error);
                if (!failed.load(std::memory_order_relaxed)) {
                    failure={code,r.fd,r.offset+static_cast<off_t>(done),n==0,false};
                    failed.store(true,std::memory_order_relaxed);
                }
                return;
            }
        }
    }
    void worker(Worker& w) noexcept {
        std::unique_lock<std::mutex> guard(state);
        for (;;) {
            w.cv.wait(guard,[&]{return stopping || w.ready;});
            if (stopping) return;
            w.ready=false;
            guard.unlock(); consume(); guard.lock();
            if (--remaining==0) drained.notify_one();
        }
    }
    Failure execute(const std::vector<Read>& reads, unsigned char* out) {
        std::lock_guard<std::mutex> call(batch); // also orders close against every accepted batch
        if (closed) { Failure f; f.closed=true; return f; }
        if (reads.empty()) return {};
        jobs=&reads; output=out; next.store(0); failed.store(false); failure={};
        size_t participants=std::min(limit,reads.size());
        {
            std::lock_guard<std::mutex> guard(state);
            remaining=participants-1;
            for (size_t i=0;i<remaining;++i) {
                workers[i]->ready=true;
                workers[i]->cv.notify_one(); // inactive helpers remain asleep
            }
        }
        consume();
        {
            std::unique_lock<std::mutex> guard(state);
            drained.wait(guard,[&]{return remaining==0;});
        }
        jobs=nullptr; output=nullptr;
        return failure; // every dispatched read has stopped before return, including errors
    }
public:
    explicit Reader(size_t count) : limit(count) {
        if (count<1 || count>64) throw py::value_error("workers must lie in [1, 64]");
        try {
            for (size_t i=1;i<count;++i) {
                workers.push_back(std::make_unique<Worker>());
                Worker* w=workers.back().get();
                w->thread=std::thread([this,w]{worker(*w);});
            }
        } catch (...) { close(); throw; }
    }
    ~Reader() { close(); }
    Reader(const Reader&)=delete;
    Reader& operator=(const Reader&)=delete;
    void close() noexcept {
        std::lock_guard<std::mutex> call(batch);
        if (closed) return;
        closed=true;
        {
            std::lock_guard<std::mutex> guard(state);
            stopping=true;
            for (auto& w:workers) w->cv.notify_one();
        }
        for (auto& w:workers) if (w->thread.joinable()) w->thread.join();
    }
    void read(py::array descriptors, py::array out) {
        if (!descriptors.dtype().is(py::dtype::of<int64_t>()) || descriptors.ndim()!=2 ||
            descriptors.shape(1)!=4 || !(descriptors.flags() & py::array::c_style))
            throw py::value_error("descriptors must be native int64 C-contiguous [n,4]");
        if (!out.dtype().is(py::dtype::of<uint8_t>()) || out.ndim()!=1 ||
            !(out.flags() & py::array::c_style) || !out.writeable())
            throw py::value_error("output must be writable uint8 C-contiguous [bytes]");
        auto d=descriptors.request(), o=out.request();
        const size_t count=static_cast<size_t>(d.shape[0]), capacity=static_cast<size_t>(o.size);
        std::vector<Read> reads;
        std::vector<std::pair<size_t,size_t>> spans;
        reads.reserve(count); spans.reserve(count);
        const char* data=static_cast<const char*>(d.ptr);
        for (size_t i=0;i<count;++i) {
            int64_t r[4]; std::memcpy(r,data+i*4*sizeof(int64_t),sizeof(r));
            if (r[0]<0 || r[0]>std::numeric_limits<int>::max() || r[1]<0 || r[2]<=0 || r[3]<0)
                throw py::value_error("invalid fd, file offset, size or destination offset");
            uint64_t offset=r[1], size=r[2], at=r[3];
            if (offset>static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
                size>static_cast<uint64_t>(std::numeric_limits<ssize_t>::max()) ||
                size>static_cast<uint64_t>(std::numeric_limits<off_t>::max())-offset ||
                at>capacity || size>capacity-at)
                throw py::value_error("read range overflows file offsets or output bounds");
            reads.push_back({static_cast<int>(r[0]),static_cast<off_t>(offset),static_cast<size_t>(size),static_cast<size_t>(at)});
            spans.emplace_back(static_cast<size_t>(at),static_cast<size_t>(at+size));
        }
        std::sort(spans.begin(),spans.end());
        for (size_t i=1;i<spans.size();++i)
            if (spans[i].first<spans[i-1].second) throw py::value_error("read destinations overlap");
        Failure f;
        { py::gil_scoped_release release; f=execute(reads,static_cast<unsigned char*>(o.ptr)); }
        if (f.closed) throw py::value_error("the native reader is closed");
        if (f.fd>=0) {
            std::string message="n-gram pread fd="+std::to_string(f.fd)+" offset="+std::to_string(f.offset)+
                (f.eof ? ": unexpected EOF (checkpoint changed)" : ": "+std::string(std::strerror(f.code)));
            py::object args=py::make_tuple(f.code ? f.code : EIO,message);
            PyErr_SetObject(PyExc_OSError,args.ptr()); throw py::error_already_set();
        }
    }
};
// GIL-held, completion-known ownership transfer for Linux read-only local FDs.
// No Python callback or interrupt boundary occurs between close and marking its
// identity consumed. Linux releases an FD even on EINTR: never retry close(2).
void close_fds(py::list fds) {
    if (!PyList_CheckExact(fds.ptr())) throw py::value_error("FD owner must be an exact list");
    std::vector<int> values;
    for (py::handle value:fds) {
        if (!PyLong_CheckExact(value.ptr())) throw py::value_error("FD identities must be exact integers");
        long fd=PyLong_AsLong(value.ptr());
        if (PyErr_Occurred()) throw py::error_already_set();
        if (fd<0 || fd>std::numeric_limits<int>::max()) throw py::value_error("invalid owned FD");
        values.push_back(static_cast<int>(fd));
    }
    auto sorted=values; std::sort(sorted.begin(),sorted.end());
    if (std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())
        throw py::value_error("owned FD identities must be distinct");
    int code=0, bad=-1;
    py::int_ consumed(-1);
    for (size_t i=0;i<values.size();++i) {
        int result=::close(values[i]); int current=errno;
        Py_INCREF(consumed.ptr());
        if (PyList_SetItem(fds.ptr(),static_cast<Py_ssize_t>(i),consumed.ptr())<0)
            std::terminate(); // exact validated list and index: unreachable C-API invariant failure
        if (result<0 && !code) { code=current; bad=values[i]; }
    }
    if (PyList_SetSlice(fds.ptr(),0,PyList_GET_SIZE(fds.ptr()),nullptr)<0)
        throw py::error_already_set();
    if (code) {
        py::object args=py::make_tuple(code,"n-gram close fd="+std::to_string(bad)+": "+std::string(std::strerror(code)));
        PyErr_SetObject(PyExc_OSError,args.ptr());throw py::error_already_set();
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("close_fds",&close_fds);
    py::class_<Reader>(m,"Reader")
        .def(py::init<size_t>())
        .def("read",&Reader::read)
        .def("close",&Reader::close,py::call_guard<py::gil_scoped_release>());
}
