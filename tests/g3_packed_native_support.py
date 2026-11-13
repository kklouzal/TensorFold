"""Finite native G3 fixture/observation helpers; no model adoption authority.

The copy observer delegates every real ATen operation. Captures are additional
D2D diagnostic work, never reference outputs or a performance measurement.
"""

import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tensorfold.families.qwen4_exp.cuda.state import State
from tools.v100_gdn_chunk64_trial import ObservedCUDAEvent, same_storage_bits

CAPACITY = 256
LAYERS = 2
SHAPES = {"tails": (63, 64, 65), "packed24": (63, 64, 65, *([3] * 20), 4), "packed32": (63, 64, 65, *([2] * 28), 8)}
INPUT_SCHEMA = "tensorfold.g3-packed-native-input-v1"


def load_input():
    path = Path(os.environ["TENSORFOLD_G3_NATIVE_INPUT"])
    expected = os.environ["TENSORFOLD_G3_NATIVE_INPUT_SHA256"]
    size = int(os.environ["TENSORFOLD_G3_NATIVE_INPUT_BYTES"])
    if not 0 < size <= 65536:
        raise ValueError("G3 native input exceeds its fixed metadata bound")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise ValueError("G3 native input size differs from held DATA")
        raw = stream.read(size + 1)
        after = os.fstat(stream.fileno())
    stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if len(raw) != size or stamp(before) != stamp(after) or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("G3 native input changed or differs from external identity")
    data = json.loads(raw)
    required = {
        "schema",
        "source_uid",
        "image_id",
        "cuda_home",
        "nvcc_sha256",
        "artifact",
        "fixture_budget_bytes",
        "journal_host_budget_bytes",
        "observer_host_budget_bytes",
    }
    if (
        type(data) is not dict
        or set(data) != required
        or data["schema"] != INPUT_SCHEMA
        or type(data["source_uid"]) is not str
        or re.fullmatch("[0-9a-f]{64}", data["source_uid"]) is None
        or type(data["image_id"]) is not str
        or re.fullmatch("sha256:[0-9a-f]{64}", data["image_id"]) is None
        or type(data["nvcc_sha256"]) is not str
        or re.fullmatch("[0-9a-f]{64}", data["nvcc_sha256"]) is None
        or type(data["cuda_home"]) is not str
        or not Path(data["cuda_home"]).is_absolute()
        or type(data["artifact"]) is not dict
        or type(data["fixture_budget_bytes"]) is not int
        or not 0 < data["fixture_budget_bytes"] <= 256 * (1 << 20)
        or type(data["journal_host_budget_bytes"]) is not int
        or data["journal_host_budget_bytes"] <= 0
        or type(data["observer_host_budget_bytes"]) is not int
        or data["observer_host_budget_bytes"] <= 0
    ):
        raise ValueError("G3 native test requires its complete externally bound Source41 DATA")
    return data


def diagnostic_weights(device, layers=LAYERS):
    cfg = SimpleNamespace(
        hidden=2560,
        streams=4,
        nk=2,
        nv=6,
        dk=128,
        dv=128,
        conv_kernel=4,
        conv_dim=1280,
        index_ratio=4,
        index_dim=128,
        kv_heads=2,
        head_dim=256,
        ple_kernel=1,
        ngram_size=1,
        ple_layers=[],
    )
    return SimpleNamespace(
        cfg=cfg,
        device=device,
        mtp=None,
        comm=None,
        layers=[SimpleNamespace(index=index, linear=True) for index in range(layers)],
        meta={"execution_dtype": "float16", "execution_provider": "sm70"},
    )


def make_state(device, codec, slot, generator):
    # Actual allocator/codec/status methods; max_rows is the ordinary five-row
    # replay extent, independent of the larger accepted prompt piece.
    state = State(diagnostic_weights(device), CAPACITY, 5, kv_dtype=codec, slot=slot)
    state.rec.normal_(std=0.125, generator=generator)
    state.rec[..., 0, 0] = -0.0
    state.conv.normal_(std=0.05, generator=generator)
    state.conv[..., 0] = -0.0
    state.cur = [slot % 2] * LAYERS
    assert state.kv_status is not None
    return state


def cut_rows(rows):
    if rows <= 1:
        return ()
    if rows == 2:
        return (1,)
    if rows == 3:
        return (1, 2)
    if rows == 63:
        return (1, 31, 62)
    if rows == 64:
        return (3, 32, 63)
    if rows == 65:
        return (1, 63, 64)
    return (1, rows // 2, rows - 1)


def payload_bound(count, *, rows=CAPACITY, layers=LAYERS):
    """Explicit fixture logical tensor upper bound; opaque JIT/runtime separate.

    Deliberately reserves three cuts per State even when a two-row piece can
    only have one. CPU oracle inputs/snapshots reside on CPU, not a GPU clone.
    """
    state = count * (2 * layers * 6 * 128 * 128 * 4 + layers * 3 * 1280 * 2 + layers * 5 * 2608 + 20)
    cuts = count * 3 * layers * (6 * 128 * 128 * 4 + 3 * 1280 * 2)
    captured_cut_sources = count * 3 * layers * 6 * 128 * 128 * 4
    frontend_capture = layers * 4144 * rows
    slab = 4936 * rows + 21
    projections = (layers + 1) * rows * 2060 * 2
    outputs = (layers + 1) * rows * (768 * 2 + 24 * 4)
    weights = layers * (1280 * 4 * 2 + 6 * 4 * 2 + 128 * 2)
    # One serialized incumbent temporary, source bank, saved rows and conv.
    incumbent = (
        2 * layers * 6 * 128 * 128 * 4
        + layers * 3 * 1280 * 2
        + layers * 5 * 2608
        + 3 * layers * (6 * 128 * 128 * 4 + 3 * 1280 * 2)
        + rows * 2608
    )
    parts = {
        "state": state,
        "cuts": cuts,
        "captured_cut_sources": captured_cut_sources,
        "frontend_capture": frontend_capture,
        "slab": slab,
        "projections": projections,
        "outputs": outputs,
        "weights": weights,
        "incumbent": incumbent,
    }
    return {
        **parts,
        "total": sum(parts.values()),
        "scope": "fixture tensor payload only; no allocator/JIT/runtime bound",
    }


class CutCopies(TorchDispatchMode):
    """Snapshot the ACTUAL bank passed to each real Cut.rec.copy_ invocation.

    Within a mode handler this mode is disabled by PyTorch, so src.clone()
    redispatches normally. The original func is always invoked exactly once.
    """

    def __init__(self, cuts, *, gpu_budget_bytes, host_budget_bytes):
        super().__init__()
        self.cuts = cuts
        self.sources = {}
        self.logical_clone_bytes = 0
        self.gpu_budget_bytes = gpu_budget_bytes
        maximum = len(cuts) * LAYERS
        # General-key dict capacity and Python tuple/Tensor headers on the
        # protected interpreter. Native TensorImpl/allocator RSS is measured
        # separately by the process envelope, not inferred from Python sizeof.
        header = 0 if not cuts else sys.getsizeof(cuts[0].rec)
        self.python_header_bound = (
            sys.getsizeof(dict.fromkeys(range(maximum)))
            + maximum * (sys.getsizeof((0, 0)) + 2 * sys.getsizeof(0) + header)
            + sys.getsizeof(self)
            + sys.getsizeof(vars(self))
        )
        if type(host_budget_bytes) is not int or self.python_header_bound > host_budget_bytes:
            raise MemoryError("copy observer lacks its explicit bounded Python-header credit")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        key = source_bank = None
        if func is torch.ops.aten.copy_.default and len(args) >= 2:
            destination, source = args[:2]
            if destination.dtype == torch.float32 and tuple(destination.shape) == (6, 128, 128):
                for index, cut in enumerate(self.cuts):
                    if cut.rec is None:
                        continue
                    width = cut.rec.stride(0) * cut.rec.element_size()
                    offset = destination.data_ptr() - cut.rec.data_ptr()
                    if 0 <= offset < cut.rec.numel() * cut.rec.element_size() and offset % width == 0:
                        key, source_bank = (index, offset // width), source
                        if key in self.sources:
                            raise RuntimeError("a G3 cut layer was copied twice")
        result = func(*args, **({} if kwargs is None else kwargs))  # Exactly one real copy.
        if key is not None:
            # Enqueue after the real copy, before Python can submit any bank
            # overwrite. This mode is disabled inside its handler by PyTorch.
            charge = source_bank.numel() * source_bank.element_size()
            if self.logical_clone_bytes + charge > self.gpu_budget_bytes:
                raise MemoryError("copy observer clone GPU credit exhausted")
            self.sources[key] = source_bank.clone()
            self.logical_clone_bytes += charge
        return result

    def assert_bits(self):
        assert len(self.sources) == len(self.cuts) * LAYERS
        for (index, layer), source in self.sources.items():
            assert same_storage_bits(self.cuts[index].rec[layer], source)


def observed_event(prepared):
    event = ObservedCUDAEvent(prepared.event)
    prepared.event = event
    return event
