"""SDK-free execution of the cache's host admission and ownership protocol.

Arrays below carry shapes and opaque expression labels. They do not emulate
MLX numerical results or establish native Metal correctness/performance.
"""
from __future__ import annotations

import ast
from copy import copy
from itertools import islice
from operator import index
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest


ROOT = Path(__file__).resolve().parents[1]


def broadcast(*shapes):
    result = []
    width = max(map(len, shapes), default=0)
    for axis in range(1, width + 1):
        values = {s[-axis] for s in shapes if len(s) >= axis} - {1}
        if len(values) > 1:
            raise ValueError("incompatible broadcast")
        result.append(next(iter(values), 1))
    return tuple(reversed(result))


class Array:
    def __init__(self, value, dtype="float32", *, shape=None):
        self.dtype = dtype
        self.values = value if isinstance(value, list) else None
        self.payload = tuple(value) if isinstance(value, list) else value
        self.shape = tuple(shape) if shape is not None else (len(value),)
        self.fail_write = False

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        value = 1
        for dimension in self.shape:
            value *= dimension
        return value

    @property
    def itemsize(self):
        return 2 if self.dtype in ("bfloat16", "float16") else 8 if self.dtype in ("float64", "int64") else 4

    @property
    def nbytes(self):
        return self.size * self.itemsize

    def tolist(self):
        return list(self.values)

    def __copy__(self):
        value = Array(self.payload, self.dtype, shape=self.shape)
        value.values = list(self.values) if self.values is not None else None
        value.fail_write = self.fail_write
        return value

    def astype(self, dtype):
        return Array(("cast", self.payload, dtype), dtype, shape=self.shape)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        shape = list(shape)
        if -1 in shape:
            known = 1
            for value in shape:
                if value != -1:
                    known *= value
            shape[shape.index(-1)] = self.size // known
        return Array(("reshape", self.payload), self.dtype, shape=shape)

    def transpose(self, *axes):
        return Array(("transpose", self.payload, axes), self.dtype, shape=tuple(self.shape[a] for a in axes))

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        if Ellipsis in keys:
            at = keys.index(Ellipsis)
            used = sum(part is not None and part is not Ellipsis for part in keys)
            keys = keys[:at] + (slice(None),) * (self.ndim - used) + keys[at + 1:]
        output, axis = [], 0
        for part in keys:
            if part is None:
                output.append(1)
            else:
                dimension = self.shape[axis]
                axis += 1
                if isinstance(part, Array):
                    output.extend(part.shape)
                elif isinstance(part, slice):
                    output.append(len(range(*part.indices(dimension))))
        output.extend(self.shape[axis:])
        return Array(("get", self.payload, repr(key)), self.dtype, shape=output)

    def __setitem__(self, key, value):
        if self.fail_write:
            raise OSError("history descriptor update refused")
        self.payload = ("write", self.payload, value.payload)

    def _binary(self, value, operation):
        other = value if isinstance(value, Array) else Array(value, shape=())
        dtype = "float64" if "float64" in (self.dtype, other.dtype) else self.dtype
        result = Array((operation, self.payload, other.payload), dtype, shape=broadcast(self.shape, other.shape))
        if operation == "sub" and self.values is not None and not isinstance(value, Array):
            result.values = [item - value for item in self.values]
        return result

    def __add__(self, value):
        return self._binary(value, "add")

    def __sub__(self, value):
        return self._binary(value, "sub")

    def __mul__(self, value):
        return self._binary(value, "mul")

    __rmul__ = __mul__

    def __lt__(self, value):
        result = self._binary(value, "lt")
        result.dtype = "bool"
        return result

    def __matmul__(self, value):
        shape = broadcast(self.shape[:-2], value.shape[:-2]) + (self.shape[-2], value.shape[-1])
        return Array(("matmul", self.payload, value.payload), self.dtype, shape=shape)

    def sum(self, axis=-1, keepdims=False):
        shape = list(self.shape)
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]
        return Array(("sum", self.payload), self.dtype, shape=shape)


class DType(str):
    @property
    def size(self):
        return 2 if self in ("float16", "bfloat16") else 8 if self == "float64" else 4


class Core:
    array, float32, float16, bfloat16, int32 = Array, "float32", "float16", "bfloat16", "int32"
    uint32, uint8 = "uint32", "uint8"
    floating, integer, gpu = "floating", "integer", "gpu"

    def __init__(self):
        self.allocations = 0
        self.fail_allocate_at = None
        self.native = []
        self.metal = SimpleNamespace(is_available=lambda: True)
        self.fast = SimpleNamespace(rms_norm=lambda a, *_: a)

    @staticmethod
    def issubdtype(dtype, category):
        return dtype in (("float16", "bfloat16", "float32", "float64") if category == "floating" else ("int32", "int64", "uint32"))

    @staticmethod
    def result_type(*dtypes):
        types = {str(dtype) for dtype in dtypes}
        return DType("float64" if "float64" in types else "float32" if "float32" in types
                     or {"float16", "bfloat16"} <= types else next(iter(types)))

    def zeros(self, shape, dtype):
        self.allocations += 1
        if self.allocations == self.fail_allocate_at:
            raise MemoryError("bounded fixture allocation")
        return Array("zero", dtype, shape=shape)

    @staticmethod
    def contiguous(value):
        return copy(value)

    @staticmethod
    def concatenate(values, axis=0):
        shape = list(values[0].shape)
        shape[axis] = sum(value.shape[axis] for value in values)
        return Array(("concatenate", tuple(value.payload for value in values)), values[0].dtype, shape=shape)

    @staticmethod
    def repeat(value, repeat, axis):
        shape = list(value.shape)
        shape[axis] *= repeat
        return Array(("repeat", value.payload, repeat), value.dtype, shape=shape)

    @staticmethod
    def exp(value):
        return Array(("exp", value.payload), value.dtype, shape=value.shape)

    @staticmethod
    def arange(count):
        return Array(list(range(count)), "int32")

    broadcast_shapes = staticmethod(broadcast)

    @staticmethod
    def split(value, cuts, axis):
        bounds = [0, *cuts, value.shape[axis]]
        result = []
        for a, b in zip(bounds, bounds[1:]):
            shape = list(value.shape)
            shape[axis] = b - a
            result.append(Array(("split", value.payload, a, b), value.dtype, shape=shape))
        return result

    @staticmethod
    def stack(values, axis):
        shape = list(values[0].shape)
        shape.insert(axis, len(values))
        return Array(("stack", tuple(v.payload for v in values)), values[0].dtype, shape=shape)

    def native_kernel(self, **kwargs):
        self.native.append(kwargs)
        return [Array("native result", dtype, shape=shape)
                for dtype, shape in zip(kwargs["output_dtypes"], kwargs["output_shapes"])]


def scope():
    core = Core()
    path = ROOT / "src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py"
    names = {"_count", "_storage", "_array", "_projection_storage", "_convolution_storage", "_tn_array", "_t_array", "LaneGDNCache", "_step_kh", "lane_gdn_call"}
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    values = {"Any": Any, "copy": copy, "index": index, "islice": islice, "mx": core, "F32": "float32", "HIST": "bfloat16",
              "_INT32_MAX": (1 << 31) - 1, "_UINT32_MAX": (1 << 32) - 1, "_UINT64_MAX": (1 << 64) - 1,
              "_step_kernel_kh": lambda: core.native_kernel, "_step_kernel": lambda: core.native_kernel,
              "nn": SimpleNamespace(silu=lambda a: a)}
    exec(compile(ast.Module(nodes, []), str(path), "exec"), values)
    values["LaneGDNCache"]._step_kh = values["_step_kh"]
    values["_gates"] = lambda alog, bias, a, b: (a.astype("float32"), b.astype("float32"))
    return values, core


def cache_case(n=2, hk=2, hv=6, dk=128, dv=128, taps=3):
    values, core = scope()
    cache = values["LaneGDNCache"](Array("conv", shape=(n, taps, 8)), Array("s0", shape=(hv, dv, dk)),
                                   key_heads=hk, capacity=4)
    args = (Array("q", shape=(n, hk, dk)), Array("k", shape=(n, hk, dk)), Array("v", shape=(n, hv, dv)),
            Array("gate", shape=(n, hv)), Array("beta", shape=(n, hv)))
    return values, core, cache, args


class Admission(unittest.TestCase):
    def test_bad_constructor_heads_shapes_counts_and_types_precede_allocation(self):
        for hk, hv, bad in ((0, 6, None), (4, 6, None), (True, 6, None), (1.5, 6, None),
                            (2, 0, None), (2, 6, "s0 rank"), (2, 6, "conv rank"), (2, 6, "dtype")):
            values, core = scope()
            conv = Array("conv", shape=(2, 3, 8) if bad != "conv rank" else (2, 8))
            initial = Array("s0", "int32" if bad == "dtype" else "float32",
                            shape=(hv, 128, 128) if bad != "s0 rank" else (hv, 128))
            with self.assertRaises(ValueError):
                values["LaneGDNCache"](conv, initial, key_heads=hk)
            self.assertEqual(core.allocations, 0)

    def test_storage_products_precede_constructor_growth_selection_and_derived_casts(self):
        limit = (1 << 31) - 1
        values, core = scope()
        with self.assertRaises(ValueError):
            values["LaneGDNCache"](Array("zero conv", shape=(limit, 0, 1)),
                                   Array("s0", shape=(1, 128, 128)), key_heads=1, capacity=limit)
        self.assertEqual(core.allocations, 0)
        with self.assertRaises(ValueError):
            values["LaneGDNCache"](Array("zero conv", shape=(0, 0, 1)),
                                   Array("bf16 s0", "bfloat16", shape=(65536, 65536, 1 << 30)),
                                   key_heads=1, capacity=1)
        self.assertEqual(core.allocations, 0)  # FP32 materialization overflows although bf16 metadata fits
        cache = values["LaneGDNCache"](Array("conv", shape=(1 << 30, 0, 1)),
                                       Array("s0", shape=(1, 8, 8)), key_heads=1, capacity=(1 << 30) - 32)
        cache.growth = 64
        allocated, previous = core.allocations, tuple(cache.state)
        with self.assertRaises(ValueError):
            cache._grow()
        self.assertEqual(core.allocations, allocated)
        self.assertEqual(tuple(cache.state), previous)
        cache = values["LaneGDNCache"](Array("conv", shape=(1, 0, 1)),
                                       Array("s0", shape=(1, 8, 8)), key_heads=1, capacity=1 << 30)
        keep = Array([], "int32", shape=(limit,))
        keep.tolist = lambda: self.fail("selection expanded before its storage proof")
        with self.assertRaises(ValueError):
            cache.filter(keep)
        # Stored bf16 histories fit uint64; widening their active prefix to FP32 does not.
        cache = values["LaneGDNCache"](Array("conv", shape=(limit, 0, 1)),
                                       Array("s0", shape=(1, 1, 2)), key_heads=1, capacity=limit)
        cache.t = limit
        q = Array("query", shape=(limit, 1, 2))
        with self.assertRaises(ValueError):
            cache._history(q, limit)
        args = (q, q, Array("value", shape=(limit, 1, 1)),
                Array("gate", shape=(limit, 1)), Array("beta", shape=(limit, 1)))
        previous = tuple(cache.state)
        with self.assertRaises(ValueError):
            cache.step(*args)
        self.assertEqual(tuple(cache.state), previous)
        self.assertEqual(core.native, [])

    def test_prepare_bounds_iterables_and_array_shape_before_materialization(self):
        _, _, cache, _ = cache_case()
        consumed = []
        def producer():
            for value in range(1000):
                consumed.append(value)
                yield 1
            raise AssertionError("unbounded input producer reached")
        with self.assertRaises(ValueError):
            cache.prepare(producer())
        self.assertEqual(consumed, [0, 1, 2])
        wrong = Array([1] * 1000, "int32")
        wrong.tolist = lambda: self.fail("bad MLX lengths shape materialized")
        with self.assertRaises(ValueError):
            cache.prepare(wrong)
        self.assertIsNone(cache.lengths)
        cache.prepare(iter((1, 2)))
        self.assertEqual(cache.lengths.values, [1, 2])

    def test_bad_current_arrays_and_histories_precede_flush_and_native_launch(self):
        changes = (("key_heads", 0), ("key_heads", 4), ("repeat", 2), ("conv", Array("bad", shape=(2, 8))),
                   ("k_hist", Array("bad", "float32", shape=(2, 2, 4, 128))),
                   ("d_hist", Array("bad", "bfloat16", shape=(2, 6, 5, 128))),
                   ("lg_hist", Array("bad", shape=(2, 6, 3))), ("t", -1), ("t", 5),
                   ("_pending", (Array("one", shape=(2, 2, 128)),)))
        for name, replacement in changes:
            _, core, cache, args = cache_case()
            cache._pending = (copy(args[1]), copy(args[2]), copy(args[3]))
            setattr(cache, name, replacement)
            pending = cache._pending
            with self.assertRaises(ValueError, msg=name):
                cache.step(*args)
            self.assertIs(cache._pending, pending)
            self.assertEqual(core.native, [])
        for position in range(5):
            _, core, cache, args = cache_case()
            items = list(args)
            items[position] = Array("bad", shape=(1,))
            with self.assertRaises(ValueError):
                cache.step(*items)
            self.assertEqual(cache.t, 0)
            self.assertEqual(core.native, [])

    def test_native_heads_cover_all_value_heads_and_capture_previous_descriptor(self):
        for hk, hv in ((1, 1), (1, 3), (2, 6), (4, 32)):
            _, core, cache, args = cache_case(hk=hk, hv=hv)
            cache.step(*args)
            self.assertEqual(core.native[0]["output_shapes"][0], (2, hv, 128))
            self.assertEqual(core.native[0]["grid"][0], 2 * hv * 128)
            key_snapshot = cache._pending[0]
            args[1].payload = "later descriptor"
            self.assertNotEqual(key_snapshot.payload, args[1].payload)
            state_snapshot = cache._pending[2]
            cache.log_g.payload = "later log descriptor"
            self.assertNotEqual(state_snapshot.payload, cache.log_g.payload)

    def test_invalid_kernel_policy_precedes_pending_history_flush(self):
        for attribute, value in (("use_kernel", 1), ("kernel_version", True), ("kernel_version", 3)):
            _, core, cache, args = cache_case()
            cache._pending = (copy(args[1]), copy(args[2]), copy(args[3]))
            original = tuple(cache.state)
            pending = cache._pending
            setattr(cache, attribute, value)
            with self.assertRaises(ValueError):
                cache.step(*args)
            self.assertEqual(tuple(cache.state), original)
            self.assertIs(cache._pending, pending)
            self.assertEqual(cache.t, 0)
            self.assertEqual(core.native, [])

    def test_current_s0_transpose_and_no_metal_existing_recurrence(self):
        _, core, cache, args = cache_case(dk=64, dv=96)
        before = cache.s0_t.payload
        cache.s0.payload = "current state"
        self.assertNotEqual(cache.s0_t.payload, before)
        out = cache.step(*args)
        self.assertEqual(out.shape, (2, 6, 96))
        self.assertEqual(core.native, [])
        _, core, cache, args = cache_case()
        core.metal.is_available = lambda: False
        self.assertEqual(cache.step(*args).shape, (2, 6, 128))
        self.assertEqual(core.native, [])

    def test_history_update_and_growth_failures_preserve_previous_state(self):
        _, core, cache, args = cache_case()
        cache._pending = (copy(args[1]), copy(args[2]), copy(args[3]))
        previous = tuple(cache.state)
        pending = cache._pending
        cache.d_hist.fail_write = True
        with self.assertRaises(OSError):
            cache._flush()
        self.assertEqual(tuple(cache.state), previous)
        self.assertIs(cache._pending, pending)
        self.assertEqual(cache.t, 0)
        cache.d_hist.fail_write = False
        cache.t = 4
        core.fail_allocate_at = core.allocations + 2
        with self.assertRaises(MemoryError):
            cache._grow()
        self.assertEqual(tuple(cache.state), previous)
        self.assertIs(cache._pending, pending)

    def test_filter_prepare_and_advance_validate_before_replacing_state(self):
        _, _, cache, _ = cache_case()
        previous = tuple(cache.state)
        for keep in (Array([2], "int32"), Array([0.0], "float32"), Array([True], "bool")):
            with self.assertRaises(ValueError):
                cache.filter(keep)
            self.assertEqual(tuple(cache.state), previous)
        cache.filter(Array([-1, 1, 1], "int32"))
        self.assertEqual(cache.lanes, 3)
        cache.prepare([3, 1, 0])
        cache.advance(4)
        self.assertEqual(cache.lengths.values, [-1, -3, -4])
        lengths = cache.lengths
        for invalid in ([2, 1], [True, 1, 0], [1.5, 1, 0]):
            with self.assertRaises(ValueError):
                cache.prepare(invalid)
            self.assertIs(cache.lengths, lengths)
        with self.assertRaises(ValueError):
            cache.advance(-1)
        cache.lengths = Array([-(1 << 31)] * cache.lanes, "int32")
        unchanged = cache.lengths
        with self.assertRaises(ValueError):
            cache.advance(1)
        self.assertIs(cache.lengths, unchanged)


class LayerBoundary(unittest.TestCase):
    def test_dynamic_producers_are_read_once_in_original_call_order(self):
        values, _, cache, base, inputs = self.layer_case()
        events = []
        class Final(Array):
            def reshape(value, *shape):
                self.assertEqual(events[-1], "get-output")
                events.append("reshape-output")
                return super().reshape(*shape)
        class Layer:
            def __init__(owner):
                for name, value in vars(base).items():
                    if name not in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "conv1d", "norm"):
                        setattr(owner, name, value)
                owner.q_reads, owner.conv_reads = 0, 0
                owner.z_value = "before-q"
            @property
            def in_proj_qkv(owner):
                owner.q_reads += 1
                events.append("get-q")
                def project(value):
                    events.append("call-q")
                    owner.z_value = "after-q"
                    return base.in_proj_qkv(value)
                return project if owner.q_reads == 1 else lambda x: self.fail("second q getter selected")
            @property
            def in_proj_z(owner):
                self.assertEqual(owner.z_value, "after-q")
                events.append("get-z")
                def project(value):
                    events.append("call-z")
                    return base.in_proj_z(value)
                return project
            @property
            def in_proj_b(owner):
                events.append("get-b")
                return lambda value: events.append("call-b") or base.in_proj_b(value)
            @property
            def in_proj_a(owner):
                events.append("get-a")
                return lambda value: events.append("call-a") or base.in_proj_a(value)
            @property
            def conv1d(owner):
                owner.conv_reads += 1
                return base.conv1d
            def norm(owner, out, z):
                return Final(out.payload, out.dtype, shape=out.shape)
            @property
            def out_proj(owner):
                events.append("get-output")
                return lambda value: events.append("call-output") or value
        layer = Layer()
        values["lane_gdn_call"](layer, inputs, None, cache)
        self.assertEqual(events, ["get-q", "call-q", "get-z", "call-z", "get-b", "call-b", "get-a", "call-a",
                                  "get-output", "reshape-output", "call-output"])
        self.assertEqual((layer.q_reads, layer.conv_reads), (1, 2))

    def test_projection_gate_and_output_stack_sizes_precede_every_producer(self):
        limit = (1 << 31) - 1
        for rows, state_type, bias_type in ((limit, "float32", "bfloat16"),
                                            (limit * 3 // 5, "float64", "bfloat16"),
                                            (limit * 3 // 5, "float32", "float64")):
            values, core = scope()
            cache = values["LaneGDNCache"](Array("conv", "bfloat16", shape=(limit, 0, 3)),
                                            Array("s0", shape=(1, 1, 1)), key_heads=1, capacity=1)
            cache.log_g = Array("current log", state_type, shape=(limit, 1))
            def producer(*a):
                self.fail("oversized output must refuse before any projection")
            layer = SimpleNamespace(num_k_heads=1, num_v_heads=1, head_k_dim=1, head_v_dim=1,
                 conv_kernel_size=1, key_dim=1, conv_dim=3,
                 conv1d=SimpleNamespace(weight=Array("conv weight", "bfloat16", shape=(3, 1, 1))),
                 A_log=Array("A", shape=(1,)), dt_bias=Array("dt", bias_type, shape=(1,)),
                 in_proj_qkv=producer, in_proj_z=producer, in_proj_a=producer, in_proj_b=producer,
                 out_proj=producer, norm=producer)
            previous = tuple(cache.state)
            with self.assertRaises(ValueError):
                values["lane_gdn_call"](layer, Array("input", "bfloat16", shape=(limit, rows, 1)), None, cache)
            self.assertEqual(tuple(cache.state), previous)
            self.assertFalse(cache._failed)
            self.assertEqual(core.native, [])

    def test_declared_mlx_projection_dtype_bias_cast_and_convolution_storage(self):
        values, _ = scope()
        class Linear(dict):
            def __getattr__(self, field):
                return self[field]
        class Quantized(Linear):
            pass
        class Conv(Linear):
            pass
        values["nn"] = SimpleNamespace(Linear=Linear, QuantizedLinear=Quantized, Conv1d=Conv)
        admit = values["_projection_storage"]
        for kind in (Linear, Quantized):
            for source_dtype in ("float16", "bfloat16", "float32", "float64"):
                for parameter_dtype in ("float16", "bfloat16", "float32", "float64"):
                    if kind is Linear:
                        module = kind(weight=Array("weight", parameter_dtype, shape=(3, 64)))
                    else:
                        module = kind(weight=Array("weight", "uint32", shape=(3, 8)), bits=4, group_size=64,
                                      mode="affine", scales=Array("scale", parameter_dtype, shape=(3, 1)),
                                      biases=Array("quant bias", parameter_dtype, shape=(3, 1)))
                    dtype = admit(module, (2, 2, 64), DType(source_dtype), 3, "projection")
                    expected = "float64" if "float64" in (source_dtype, parameter_dtype) else "float32" \
                        if "float32" in (source_dtype, parameter_dtype) or source_dtype != parameter_dtype else source_dtype
                    self.assertEqual(dtype, expected)
        limit = (1 << 31) - 1
        module = Quantized(weight=Array("packed", "uint32", shape=(64, 8)), bits=4, group_size=64, mode="affine",
                           scales=Array("scale", "bfloat16", shape=(64, 1)),
                           biases=Array("quant bias", "bfloat16", shape=(64, 1)), bias=Array("bias", "float64", shape=(64,)))
        with self.assertRaisesRegex(ValueError, "output bias output"):
            admit(module, (limit, limit // 64, 64), DType("bfloat16"), 64, "output bias")
        # Unknown/custom implementations retain their current-value path; no
        # presumed FP64 output floor is imposed on their legal BF16 metadata.
        self.assertIsNone(admit(object(), (limit, limit, 1), DType("bfloat16"), 1, "custom"))
        conv = Conv(weight=Array("weight", "float64", shape=(1, 1, 1)))
        with self.assertRaises(ValueError):
            values["_convolution_storage"](conv, (limit, limit, 1), DType("bfloat16"), (limit, limit, 1))

    @staticmethod
    def layer_case(kernel=1):
        values, core, cache, _ = cache_case(hk=1, hv=1, taps=kernel - 1)
        width = 3 * 128
        cache.conv = Array("conv", shape=(2, kernel - 1, width))
        class Conv:
            weight = Array("weight", shape=(width, kernel, 1))

            def __call__(self, inputs):
                return Array("conv output", shape=(inputs.shape[0], inputs.shape[1] - kernel + 1, width))

        layer = SimpleNamespace(num_k_heads=1, num_v_heads=1, head_k_dim=128, head_v_dim=128,
                                conv_kernel_size=kernel, key_dim=128, conv_dim=width, conv1d=Conv(),
                                A_log=Array("A", shape=(1,)), dt_bias=Array("dt", shape=(1,)),
                                in_proj_qkv=lambda x: Array("qkv", shape=(2, 2, width)),
                                in_proj_z=lambda x: Array("z", shape=(2, 2, 128)),
                                in_proj_a=lambda x: Array("a", shape=(2, 2, 1)),
                                in_proj_b=lambda x: Array("b", shape=(2, 2, 1)),
                                norm=lambda x, z: x, out_proj=lambda x: x)
        return values, core, cache, layer, Array("inputs", shape=(2, 2, 128))

    def test_kernel_one_retains_empty_convolution_tail(self):
        values, _, cache, layer, inputs = self.layer_case()
        output = values["lane_gdn_call"](layer, inputs, None, cache)
        self.assertEqual(cache.conv.shape, (2, 0, 384))
        self.assertEqual(output.shape, (2, 2, 128))

    def test_bad_layer_geometry_or_mask_precedes_projection_and_mutation(self):
        for field in ("num_k_heads", "head_k_dim", "conv_dim"):
            values, _, cache, layer, inputs = self.layer_case()
            setattr(layer, field, 5)
            with self.assertRaises(ValueError):
                values["lane_gdn_call"](layer, inputs, None, cache)
            self.assertFalse(cache._failed)
        values, _, cache, layer, inputs = self.layer_case()
        with self.assertRaises(ValueError):
            values["lane_gdn_call"](layer, inputs, Array("mask", shape=(3, 3)), cache)
        self.assertFalse(cache._failed)

    def test_layer_execution_failure_keeps_primary_and_marks_cache_unusable(self):
        values, _, cache, layer, inputs = self.layer_case()
        primary = OSError("projection failed")
        def fail(value):
            raise primary
        layer.in_proj_qkv = fail
        try:
            values["lane_gdn_call"](layer, inputs, None, cache)
        except OSError as error:
            self.assertIs(error, primary)
        else:
            self.fail("missing primary error")
        with self.assertRaises(RuntimeError):
            cache._admit_state()


if __name__ == "__main__":
    unittest.main()
