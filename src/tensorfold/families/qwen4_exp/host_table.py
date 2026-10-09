"""Host n-gram shards for CUDA and for Metal past GPU memory: memory-mapped here, or read from disk by SSDTable."""

from __future__ import annotations

import os
import stat
import threading
import weakref
from contextlib import contextmanager
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tensorfold.families.qwen4_exp.ssd_table import SSDTable, _span
from tensorfold.families.qwen4_exp.read_ahead import ReadAhead, row_ids
from tensorfold.families.qwen4_exp.table_lifetime import PythonPool, TableLifetime, release_files
from tensorfold.cuda.capacity import SIZES
from tensorfold.cuda.tensor_file import checkpoint_path, read_header as _header, read_header_stream
from .table_file import TableFile as _HostFile

_PARTS = ("weight", "scales", "biases")
# a prompt chunk's gather copies big row runs on worker threads (GIL released); bytes stay the same as single-threaded
GATHER_THREADS = 16
GATHER_SPLIT = 512


def ngrams_on_host(model_dir: Path, ssd: bool = False) -> bool:
    """Host n-gram tables when read from SSD, else past the GPU working-set threshold (TF_NGRAM_HOST=0/1 overrides)."""

    flag = os.environ.get("TF_NGRAM_HOST", "")
    if ssd:
        if flag == "0":
            raise ValueError("--ple-on-ssd reads the n-gram tables on the host: unset TF_NGRAM_HOST=0")
        return True
    if flag in ("0", "1"):
        return flag == "1"
    import mlx.core as mx

    size = sum(p.stat().st_size for p in Path(model_dir).glob("model*.safetensors"))
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return size > 0.75 * int(info["max_recommended_working_set_size"])


def windows_lock_pages(arrays, kernel32=None):
    """Best effort page pinning on Windows: VirtualLock answers False when it refuses, and nothing stays pinned."""

    import ctypes

    try:
        api = kernel32 if kernel32 is not None else ctypes.WinDLL("kernel32", use_last_error=True)
        pinned = []
        for array in arrays:
            address, size = ctypes.c_void_p(array.ctypes.data), ctypes.c_size_t(array.nbytes)
            if not api.VirtualLock(address, size):
                for past, past_size in pinned:
                    api.VirtualUnlock(past, past_size)
                return False
            pinned.append((address, size))
        return True
    except (AttributeError, OSError):      # no kernel32 here either: the tables simply stay unpinned
        return False


def _note(primary, message):
    """A failed diagnostic cannot replace an established operation failure."""
    try:
        BaseException.add_note(primary, message)
    except BaseException as annotation:
        if annotation is primary:
            raise primary
        raise primary from annotation


def _retain_prefetch_lifetime(primary, cleanup, life) -> None:
    """Publish retry authority through BaseException's native instance dictionary."""
    for error in (primary, cleanup):
        try:
            dictionary = BaseException.__dict__["__dict__"].__get__(error)
            dict.__setitem__(dictionary, "prefetch_lifetime", life)
        except BaseException as publication:
            if publication is primary:
                raise primary
            raise primary from publication




def _release_unpublished_file(slot) -> None:
    """Consume one exact FileIO owner, retaining an unfinished close for retry."""
    file = slot[0]
    if file is not None:
        try:
            if not file.closed:
                file.close()
        finally:
            if file.closed:
                slot[0] = None


def _retire_local_file(file, slot) -> None:
    """Adopt the acquired local even when interruption preceded slot publication."""
    interrupted = None
    while True:
        try:
            slot[0] = file
            break
        except KeyboardInterrupt as error:
            if interrupted is None:
                interrupted = error
    try:
        _release_unpublished_file(slot)
    except BaseException as cleanup:
        if interrupted is not None:
            _note(interrupted, "unpublished file close also failed; owner retained")
            raise interrupted from cleanup
        raise
    if interrupted is not None:
        raise interrupted


def _release_prefetch_files(fds, *, files, slots) -> None:
    for slot in slots:
        _release_unpublished_file(slot)
    release_files(fds, files)
    slots.clear()


def _release_mapped(fds, *, files, lists, pins, unpublished) -> None:
    """Release only table-owned references; borrowed NumPy views keep their maps alive."""
    _release_unpublished_file(unpublished)
    _unlock_pins(pins)
    release_files(fds, files)
    for arrays in lists:
        arrays.clear()
    lists.clear()


def _unlock_pins(pins) -> None:
    while pins:
        _, unlock, address, size, windows = pins[-1]
        result = unlock(address, size)
        if (not result if windows else result != 0):
            raise OSError("could not unlock the n-gram table's owned pages")
        pins.pop()


def _joined_copy(life, pool, copy, jobs, workers) -> None:
    """Bound dispatch and observe every copy before surfacing recoverable read errors.

    Copy errors leave a fully drained pool usable. Dispatch loss or caller
    interruption uses TableLifetime's terminal journal and fail-closed drain.
    No mapping can retire while an accepted callback is still using it.
    """
    failures = [None] * len(jobs)
    indexed = list(enumerate(jobs))
    batches = min(workers, len(jobs))

    def run(batch):
        for index, job in batch:
            try:
                copy(job)
            except BaseException as error:
                failures[index] = error
                break

    primary = None
    try:
        life.parallel(pool, run, [indexed[i::batches] for i in range(batches)])
    except BaseException as error:
        primary = error
    for error in failures:
        if error is not None:
            if primary is None:
                primary = error
            elif error is not primary:
                _note(primary, "additional mapped table copy failed (" + type(error).__name__ + ")")
    if primary is not None:
        raise primary


class _MappedTable:
    """A table owns its pool, mappings and pins through every accepted operation.

    Close rejects new work, drains all borrowers and accepted copies, unpins
    owned regions, then drops owned array references. It never forcibly closes
    NumPy's private mmap: an externally borrowed array/view remains valid.
    The checkpoint must remain immutable while its mappings are in use.
    """

    @contextmanager
    def _construction(self):
        self._owned_lists, self._files, self._fds, self._pins = [], [], [], []
        self._sources = {}
        self._unpublished_file = [None]
        self._pin_guard = threading.Lock()
        self._pool = self._life = self._closer = None
        try:
            self._pool = ThreadPoolExecutor(GATHER_THREADS, thread_name_prefix="ngram-gather")
            release = partial(_release_mapped, files=self._files, lists=self._owned_lists, pins=self._pins,
                              unpublished=self._unpublished_file)
            self._life = TableLifetime(PythonPool(self._pool), self._fds, release)
            self._closer = weakref.finalize(self, self._life.close)
            yield
            release_files(self._fds, self._files)
            self._sources.clear()
        except BaseException as primary:
            try:
                if self._life is not None:
                    self.close()
                else:
                    if self._pool is not None:
                        self._pool.shutdown(wait=True)
                    _release_mapped(self._fds, files=self._files, lists=self._owned_lists, pins=self._pins,
                                    unpublished=self._unpublished_file)
            except BaseException as cleanup:
                _note(primary, "mapped table construction cleanup also failed; owner retained")
                raise primary from cleanup
            raise

    def _arrays(self):
        arrays = []
        self._owned_lists.append(arrays)
        return arrays

    def _map(self, path, entry, dtype, kind):
        path = Path(path)
        if path not in self._sources:
            file = None
            try:
                file = _HostFile(path)
                self._unpublished_file[0] = file
                self._files.append(file)
                self._unpublished_file[0] = None
                file.open()
                fd = file.fileno()
                self._fds.append(fd)
            except BaseException as primary:
                if file is not None:
                    try:
                        _retire_local_file(file, self._unpublished_file)
                    except BaseException as cleanup:
                        _note(primary, "unpublished mapped file close failed; owner retained")
                        raise primary from cleanup
                raise
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(f"{path.name}: the n-gram checkpoint must be a regular file")
            data, header = read_header_stream(file, SIZES, label=path)
            descriptors = {(info["dtype"], tuple(info["shape"]), tuple(info["data_offsets"]))
                           for name, info in header.items() if name != "__metadata__"}
            self._sources[path] = (file, data, os.fstat(fd).st_size, descriptors)
        file, data, size, descriptors = self._sources[path]
        rows, width, start = _span(entry, kind, data, size, path.name)
        if (entry["dtype"], tuple(entry["shape"]), tuple(entry["data_offsets"])) not in descriptors:
            raise ValueError(f"{path.name}: the n-gram tensor metadata disagrees with its opened checkpoint header")
        array = np.memmap(file, dtype=dtype, mode="r", offset=start, shape=(rows, width // kind[2]))
        _random_access(array)
        return array

    def _gather_owned(self, flat):
        return self._life.call(self._gather_flat, flat)

    def close(self) -> None:
        self._life.close()
        self._sources.clear()
        if self._closer is not None:
            self._closer.detach()

    def _lock_arrays(self, arrays) -> bool:
        with self._pin_guard:
            return self._pin_arrays(arrays)

    def _pin_arrays(self, arrays) -> bool:
        import ctypes

        if self._pins:
            return True
        windows = os.name == "nt"
        try:
            api = ctypes.WinDLL("kernel32", use_last_error=True) if windows else ctypes.CDLL(None, use_errno=True)
            lock, unlock = (api.VirtualLock, api.VirtualUnlock) if windows else (api.mlock, api.munlock)
        except (AttributeError, OSError):
            if windows:
                return False
            raise
        lock.argtypes = unlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        try:
            for array in arrays:
                address, size = array.ctypes.data, array.nbytes
                # Publish before the syscall: an interrupted return may have
                # pinned pages; the retained array keeps rollback authority.
                self._pins.append((array, unlock, address, size, windows))
                result = lock(address, size)
                if (not result if windows else result != 0):
                    self._pins.pop()
                    _unlock_pins(self._pins)
                    return False
        except BaseException as primary:
            try:
                _unlock_pins(self._pins)
            except BaseException as cleanup:
                _note(primary, "mapped table pin rollback also failed; pins retained")
                raise primary from cleanup
            raise
        return True


class HostTable(_MappedTable):
    """Keep n-gram shards memory-mapped on the host; gather copies only requested rows, never whole tables to the GPU."""

    def __init__(self, files: list[tuple[Path, dict, dict, dict]]) -> None:
        with self._construction():
            if not files:
                raise ValueError("the n-gram table has no shards")
            self.words, self.scales, self.biases, starts = self._arrays(), self._arrays(), self._arrays(), [0]
            maps: dict = {}
            fidx, wbase, sbase, bbase = [], [], [], []
            for path, hw, hs, hb in files:
                path = Path(path)
                self.words.append(self._map(path, hw, np.uint32, ("weight", "U32", 4)))
                self.scales.append(self._map(path, hs, np.uint16, ("scales", "BF16", 2)))
                self.biases.append(self._map(path, hb, np.uint16, ("biases", "BF16", 2)))
                if (self.words[-1].shape[0] != self.scales[-1].shape[0]
                        or self.scales[-1].shape != self.biases[-1].shape
                        or self.words[-1].shape[1] != 4 * self.scales[-1].shape[1]):
                    raise ValueError(f"{Path(path).name}: an n-gram shard is not 4-bit rows with a scale and bias every 32")
                if self.words[-1].shape[1] != self.words[0].shape[1]:
                    raise ValueError(f"{Path(path).name}: the n-gram shards differ in row width")
                starts.append(starts[-1] + self.words[-1].shape[0])
                if path not in maps:
                    source, data, _, _ = self._sources[Path(path)]
                    view = np.memmap(source, dtype=np.uint8, mode="r")
                    _random_access(view)          # gathers read through this view: a fault reads its page, not those around
                    maps[path] = (len(maps), view, data)
                index, _, data = maps[path]
                fidx.append(index)
                wbase.append(data + hw["data_offsets"][0])
                sbase.append(data + hs["data_offsets"][0])
                bbase.append(data + hb["data_offsets"][0])
            self.starts = np.array(starts, dtype=np.int64)
            self.rows = int(self.starts[-1])
            # byte views of the files, so a gather is one fancy index per file and component, not per shard
            self.files = self._arrays()
            self.files.extend(m for _, m, _ in sorted(maps.values(), key=lambda t: t[0]))
            self.fidx = np.array(fidx, dtype=np.int64)
            self.wbase, self.sbase, self.bbase = (np.array(x, dtype=np.int64) for x in (wbase, sbase, bbase))
            self.wrow = self.words[0].shape[1] * 4
            self.grow = self.scales[0].shape[1] * 2
            self.nbytes = sum(a.nbytes for a in self.words + self.scales + self.biases)

    def gather(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows ``ids`` (global) -> words [n, W] uint32, scales and biases [n, G] (bf16 bits as uint16)."""

        return self._life.call(self._gather_flat, row_ids(ids, self.rows))

    def _gather_flat(self, flat):
        n = len(flat)
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        local = flat - self.starts[shard]
        where = self.fidx[shard]
        wo = self.wbase[shard] + local * self.wrow
        so = self.sbase[shard] + local * self.grow
        bo = self.bbase[shard] + local * self.grow
        w = np.empty((n, self.wrow), dtype=np.uint8)
        sc = np.empty((n, self.grow), dtype=np.uint8)
        bi = np.empty((n, self.grow), dtype=np.uint8)
        aw, ag = np.arange(self.wrow), np.arange(self.grow)

        def copy(job) -> None:
            mm, at = job
            w[at] = mm[wo[at, None] + aw]
            sc[at] = mm[so[at, None] + ag]
            bi[at] = mm[bo[at, None] + ag]

        if GATHER_THREADS > 1 and n >= 2 * GATHER_SPLIT:          # a prompt chunk: copy on threads
            jobs = []
            for f in np.unique(where):
                at = np.nonzero(where == f)[0]
                parts = max(1, min(GATHER_THREADS, len(at) // GATHER_SPLIT))
                jobs += [(self.files[f], piece) for piece in np.array_split(at, parts)]
            _joined_copy(self._life, self._pool, copy, jobs, GATHER_THREADS)
        else:
            for f in np.unique(where):
                copy((self.files[f], np.nonzero(where == f)[0]))
        return w.view(np.uint32), sc.view(np.uint16), bi.view(np.uint16)

    def lock(self) -> bool:
        """Pin owned pages, rolling back a refused lock; close releases owned pins."""
        return self._life.call(self._lock_arrays, self.words + self.scales + self.biases)

    def prefetch(self, workers: int = 8) -> float:
        """Read every shard once so the lookups hit the page cache (seconds taken); the pages stay evictable."""

        return self._life.call(_prefetch, self.words + self.scales + self.biases, workers)


class BF16Table(_MappedTable):
    """bf16 n-gram shards (the NVFP4 checkpoint's): memory-mapped, gathered a lookup at a time as bf16 bits."""

    bits = 16

    def __init__(self, files: list[tuple[Path, dict]]) -> None:
        with self._construction():
            if not files:
                raise ValueError("the n-gram table has no shards")
            self.values, starts = self._arrays(), [0]
            for path, weight in files:
                if not isinstance(weight, dict) or weight.get("dtype") != "BF16":
                    raise ValueError(f"{Path(path).name}: the n-gram weights must be BF16 tensors")
                self.values.append(self._map(path, weight, np.uint16, ("weight", "BF16", 2)))
                if self.values[-1].shape[1] != self.values[0].shape[1]:
                    raise ValueError(f"{Path(path).name}: the n-gram shards differ in row width")
                starts.append(starts[-1] + self.values[-1].shape[0])
            self.starts = np.array(starts, dtype=np.int64)
            self.rows = int(self.starts[-1])
            self.width = int(self.values[0].shape[1])       # bf16 values a row (the engine's ``dh``)
            self.wrow = self.width * 2                      # bytes a row
            self.nbytes = sum(a.nbytes for a in self.values)

    def _where(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each id's shard and row in it, after checking the ids lie in the table."""

        flat = ids      # validated public IDs, or immutable IDs owned by ReadAhead
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        return shard, flat - self.starts[shard]

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """Rows ``ids`` (global) -> [n, W] uint16 (the bf16 bits of W values)."""

        return self._life.call(self._gather_flat, row_ids(ids, self.rows))

    def _gather_flat(self, flat):
        shard, local = self._where(flat)
        out = np.empty((shard.size, self.width), dtype=np.uint16)

        def copy(f: int, at: np.ndarray) -> None:
            out[at] = self.values[f][local[at]]

        _copy_rows(self._life, self._pool, shard, copy)
        return out

    def lock(self) -> bool:
        """Pin owned pages, rolling back a refused lock; close releases owned pins."""
        return self._life.call(self._lock_arrays, self.values)

    def prefetch(self, workers: int = 8) -> float:
        """Read every shard once so the lookups hit the page cache (seconds taken); the pages stay evictable."""

        return self._life.call(_prefetch, self.values, workers)


class FP8Table(BF16Table):
    """e4m3 n-gram shards with one table scale: lookups give bf16(e4m3 x scale) through a 256-entry table."""

    def __init__(self, files: list[tuple[Path, dict]], scale: float) -> None:
        with self._construction():
            if not files:
                raise ValueError("the n-gram table has no shards")
            from tensorfold.cuda.nvfp4.format import e4m3

            self.values, starts = self._arrays(), [0]
            for path, weight in files:
                if not isinstance(weight, dict) or weight.get("dtype") != "F8_E4M3":
                    raise ValueError(f"{Path(path).name}: the n-gram weights must be F8_E4M3 tensors")
                self.values.append(self._map(path, weight, np.uint8, ("weight", "F8_E4M3", 1)))
                if self.values[-1].shape[1] != self.values[0].shape[1]:
                    raise ValueError(f"{Path(path).name}: the n-gram shards differ in row width")
                starts.append(starts[-1] + self.values[-1].shape[0])
            self.starts = np.array(starts, dtype=np.int64)
            self.rows = int(self.starts[-1])
            self.width = int(self.values[0].shape[1])
            self.wrow = self.width
            self.nbytes = sum(a.nbytes for a in self.values)
            f32 = (e4m3(np.arange(256)) * np.float32(scale)).astype(np.float32).view(np.uint32).astype(np.uint64)
            self.lut = ((f32 + 0x7FFF + ((f32 >> 16) & 1)) >> 16).astype(np.uint16)   # round to nearest even
            self.lut[(np.arange(256) & 0x7F) == 0x7F] = 0x7FC0                         # e4m3's NaN codes stay NaN

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """Rows ``ids`` (global) -> [n, W] uint16 (bf16 bits of e4m3 x scale)."""

        return self._life.call(self._gather_flat, row_ids(ids, self.rows))

    def _gather_flat(self, flat):
        return self.lut[super()._gather_flat(flat)]


class NVFP4Table(BF16Table):
    """NVFP4 n-gram shards (e2m1 codes, e4m3 a 16 values, one fp32 table scale): lookups give bf16(code x scale x g)."""

    def __init__(self, files: list[tuple[Path, dict, dict]], scale: float) -> None:
        with self._construction():
            if not files:
                raise ValueError("the n-gram table has no shards")
            from tensorfold.cuda.nvfp4.format import E2M1, e4m3

            self.values, self.scales, starts = self._arrays(), self._arrays(), [0]
            for path, weight, block in files:
                if not isinstance(weight, dict) or not isinstance(block, dict) or weight.get("dtype") != "U8" or block.get("dtype") != "F8_E4M3":
                    raise ValueError(f"{Path(path).name}: NVFP4 n-gram shards are U8 codes with F8_E4M3 scales")
                self.values.append(self._map(path, weight, np.uint8, ("weight", "U8", 1)))
                self.scales.append(self._map(path, block, np.uint8, ("block scales", "F8_E4M3", 1)))
                if self.values[-1].shape[1] != self.values[0].shape[1] or \
                        self.scales[-1].shape[1] * 8 != self.values[-1].shape[1] or \
                        self.scales[-1].shape[0] != self.values[-1].shape[0]:
                    raise ValueError(f"{Path(path).name}: the n-gram shards differ in row width")
                starts.append(starts[-1] + self.values[-1].shape[0])
            self.starts = np.array(starts, dtype=np.int64)
            self.rows = int(self.starts[-1])
            self.width = int(self.values[0].shape[1]) * 2
            self.wrow = self.width // 2
            self.nbytes = sum(a.nbytes for a in self.values + self.scales)
            self.e2m1, self.e4m3, self.g = E2M1, e4m3(np.arange(256)), np.float32(scale)

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """Rows ``ids`` (global) -> [n, W] uint16: bf16 bits of the fp32 code x block scale x table scale, rounded once."""

        return self._life.call(self._gather_flat, row_ids(ids, self.rows))

    def _gather_flat(self, flat):
        shard, local = self._where(flat)
        codes = np.empty((shard.size, self.width // 2), dtype=np.uint8)
        blocks = np.empty((shard.size, self.width // 16), dtype=np.uint8)

        def copy(f: int, at: np.ndarray) -> None:
            codes[at], blocks[at] = self.values[f][local[at]], self.scales[f][local[at]]

        _copy_rows(self._life, self._pool, shard, copy)
        nib = np.stack([codes & 0xF, codes >> 4], -1).reshape(shard.size, self.width)
        v = (self.e2m1[nib] * np.repeat(self.e4m3[blocks], 16, axis=1)).astype(np.float32) * self.g
        f32 = v.astype(np.float32).view(np.uint32).astype(np.uint64)
        return ((f32 + 0x7FFF + ((f32 >> 16) & 1)) >> 16).astype(np.uint16)

    def lock(self) -> bool:
        return False

    def prefetch(self, workers: int = 8) -> float:
        return self._life.call(_prefetch, self.values + self.scales, workers)


def shard_keys(name: str, count: int, names) -> list[str]:
    """Resolve the flat and nested shard spellings used by MLX checkpoints."""

    return [next((key for key in (f"{name}.shard_{i}", f"{name}.shards.{i}")
                  if key + ".weight" in names), f"{name}.shard_{i}") for i in range(count)]


def open_table(model_dir: Path, shards: list[tuple[str, str]], scale, *, ssd: bool = False):
    """The n-gram table in its shards' layout (MLX 4-bit, bf16, FP8, NVFP4); ``scale(name)`` reads a table scale."""

    headers: dict[str, dict] = {}
    kinds: dict[str, list] = {"mlx": [], "bf16": [], "fp8": [], "nvfp4": []}
    for shard, key in shards:
        if shard not in headers:
            headers[shard] = read_header(checkpoint_path(model_dir, shard))
        h, path = headers[shard], checkpoint_path(model_dir, shard)
        if key + ".scales" in h:
            kinds["mlx"].append((path, h[key + ".weight"], h[key + ".scales"], h[key + ".biases"]))
        elif h[key + ".weight"].get("dtype") == "F8_E4M3":
            kinds["fp8"].append((path, h[key + ".weight"]))
        elif key + ".weight_scale" in h:
            kinds["nvfp4"].append((path, h[key + ".weight"], h[key + ".weight_scale"]))
        else:
            kinds["bf16"].append((path, h[key + ".weight"]))
    used = [k for k, v in kinds.items() if v]
    if len(used) != 1:
        raise ValueError(f"the n-gram shards mix layouts: {', '.join(used)}")
    files = kinds[used[0]]
    if used[0] == "nvfp4":
        return NVFP4Table(files, scale("weight_scale_2"))
    if used[0] == "fp8":
        return FP8Table(files, scale("weight_scale"))
    table = BF16Table(files) if used[0] == "bf16" else _ssd_table(files) if ssd else HostTable(files)
    try:
        table.weight_scale = float(scale("weight_scale"))
        return table
    except BaseException as primary:
        try:
            table.close()
        except BaseException as cleanup:
            _note(primary, "n-gram scale setup cleanup also failed")
            raise primary from cleanup
        raise




def _copy_rows(life, pool: ThreadPoolExecutor, shard: np.ndarray, copy) -> None:
    """``copy(f, at)`` for each shard's rows; a prompt chunk's (2 GATHER_SPLIT rows or more) split over the pool."""

    if GATHER_THREADS > 1 and shard.size >= 2 * GATHER_SPLIT:
        jobs = []
        for f in np.unique(shard):
            at = np.nonzero(shard == f)[0]
            parts = max(1, min(GATHER_THREADS, len(at) // GATHER_SPLIT))
            jobs += [(f, piece) for piece in np.array_split(at, parts)]
        _joined_copy(life, pool, lambda job: copy(*job), jobs, GATHER_THREADS)
    else:
        for f in np.unique(shard):
            copy(f, np.nonzero(shard == f)[0])




def _random_access(array: np.ndarray) -> None:
    """Advise random access on a table's mapping (read-ahead only evicts useful pages); best effort."""

    try:
        import mmap as _mmap

        array._mmap.madvise(_mmap.MADV_RANDOM)          # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        pass


PREFETCH_READ = 16 << 20        # bytes a prefetch read: a page fault under MADV_RANDOM reads one page, a read the span


@contextmanager
def _prefetch_file(path, files, guard, slot):
    """Retain a FileIO owner across fallible close; never retry a raw descriptor."""
    file = None
    primary = None
    published = False
    try:
        file = _HostFile(path)
        slot[0] = file
        with guard:
            files.append(file)
            published = True
        slot[0] = None
        file.open()
        yield file
    except BaseException as error:
        primary = error
        raise
    finally:
        if file is not None:
            try:
                if published:
                    file.close()
                else:
                    _retire_local_file(file, slot)
            except BaseException as cleanup:
                if primary is not None:
                    _note(primary, "prefetch span file close also failed; owner retained")
                    raise primary from cleanup
                raise
            finally:
                if published and file.closed:
                    with guard:
                        files.remove(file)


def _prefetch(arrays: list[np.ndarray], workers: int = 8) -> float:
    """Read every selected byte, completing short reads, with bounded accepted work.

    Each worker owns one reusable buffer. A per-span file owner is retained
    until its read completes. File reuse is a separate measured candidate.
    Table callers hold their outer lifetime lease throughout this operation.
    """
    import time

    if type(workers) is not int or workers <= 0:
        raise ValueError("n-gram prefetch workers must be a positive integer")
    spans = [(arr, at) for arr in arrays for at in range(0, arr.nbytes, PREFETCH_READ)]
    local = threading.local()
    files, slots, guard = [], [], threading.Lock()

    def read(span) -> None:
        arr, at = span
        n = min(PREFETCH_READ, arr.nbytes - at)
        path, offset = getattr(arr, "filename", None), getattr(arr, "offset", None)
        if path is None or offset is None:
            np.asarray(arr.reshape(-1).view(np.uint8)[at:at + n]).sum(dtype=np.uint64)
            return
        if getattr(local, "buf", None) is None:
            buffer, slot = memoryview(bytearray(PREFETCH_READ)), [None]
            with guard:
                slots.append(slot)       # registered before any file acquisition
            local.buf, local.slot = buffer, slot
        with _prefetch_file(path, files, guard, local.slot) as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode) or offset + at + n > info.st_size:
                raise OSError("the n-gram checkpoint changed before prefetch")
            file.seek(offset + at)
            done = 0
            while done < n:
                got = file.readinto(local.buf[done:n])
                if type(got) is not int or not 0 < got <= n - done:
                    raise OSError(f"short read of the n-gram tables at byte {offset + at + done}: checkpoint changed")
                done += got

    pool = life = None
    primary = None
    t0 = time.perf_counter()
    try:
        pool = ThreadPoolExecutor(workers, thread_name_prefix="ngram-prefetch")
        life = TableLifetime(PythonPool(pool), [], partial(_release_prefetch_files, files=files, slots=slots))
        life.call(_joined_copy, life, pool, read, spans, workers)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            if life is not None:
                life.close()
            elif pool is not None:
                pool.shutdown(wait=True)
                _release_prefetch_files([], files=files, slots=slots)
        except BaseException as cleanup:
            primary = primary if primary is not None else cleanup
            _retain_prefetch_lifetime(primary, cleanup, life)
            _note(primary, "n-gram prefetch cleanup also failed; owner retained")
            if cleanup is primary:
                raise primary
            raise primary from cleanup
    return time.perf_counter() - t0


def read_header(path: Path) -> dict:
    """The shared strict safetensors geometry/UTF-8/JSON contract, without a numeric runtime."""
    return _header(path, SIZES)[1]


def from_checkpoint(model_dir: Path, name: str, count: int, *, ssd: bool = False) -> HostTable | SSDTable:
    """Shards ``{name}.shard_{i}``, i < count, each in one file: memory-mapped, or with ``ssd`` read at each lookup."""

    if type(count) is not int or count <= 0:
        raise ValueError("the n-gram shard count must be a positive integer")
    paths = [checkpoint_path(model_dir, path.name) for path in sorted(Path(model_dir).glob("model*.safetensors"))]
    headers = {path: read_header(path) for path in paths}
    files = []
    for key in shard_keys(name, count, {key for h in headers.values() for key in h}):
        found = [(path, h) for path, h in headers.items() if any(f"{key}.{part}" in h for part in _PARTS)]
        if len(found) != 1 or not all(f"{key}.{part}" in found[0][1] for part in _PARTS):
            raise ValueError(f"{key}: expected its weight, scales and biases together in one checkpoint file")
        path, h = found[0]
        files.append((path, *(h[f"{key}.{part}"] for part in _PARTS)))
    return _ssd_table(files, read_ahead=False) if ssd else HostTable(files)


def _ssd_table(files, *, read_ahead=True):
    """Opt-in native executor; startup selection and wrapper acquisition fail closed."""
    workers = os.environ.get("TENSORFOLD_SSD_NATIVE_THREADS", "")
    if not workers:
        table, wrap = SSDTable(files), ReadAhead
    else:
        if workers not in ("16", "32", "64"):
            raise ValueError("TENSORFOLD_SSD_NATIVE_THREADS must be 16, 32 or 64")
        from tensorfold.families.qwen4_exp.native_ssd import NativeSSDTable, NativeReadAhead
        table, wrap = NativeSSDTable(files, workers=int(workers)), NativeReadAhead
    if not read_ahead:
        return table
    try:
        return wrap(table)
    except BaseException as primary:
        try:
            table.close()
        except BaseException as cleanup:
            _note(primary, "n-gram read-ahead wrapper construction cleanup failed; table retained")
            raise primary from cleanup
        raise
