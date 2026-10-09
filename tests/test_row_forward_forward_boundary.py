"""Host forward admission controls; opaque arrays never perform SDK math."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from itertools import islice
import math
from operator import index
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Sequence
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/tensorfold/kernels/qwen/dense/v1/row_forward.py"
NATIVE_SOURCE = SOURCE.parent


class Array:
    def __init__(self, values=(), dtype=None, *, shape=None):
        self.values, self.dtype = tuple(values), dtype
        self.shape = tuple(shape) if shape is not None else (len(self.values),)
        self.ndim = len(self.shape)
        self.size = math.prod(self.shape)

    def __iter__(self):
        raise AssertionError("lazy array contents were read")


def selected(source, names):
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    return nodes


def execute(nodes, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
            "<actual-forward-admission>",
            "exec",
        ),
        namespace,
    )


@contextmanager
def fixture(*, raw=True):
    namespace = {
        "mx": SimpleNamespace(array=Array, int32="I32", bfloat16="BF16"),
        "Sequence": Sequence,
        "Any": object,
        "integer_index": index,
        "islice": islice,
        "math": math,
        "ROW_ATTENTION": raw,
        "row_matmul": SimpleNamespace(BACKEND=SimpleNamespace(max_rows=128, name="opaque")),
        "_CHAINS": {width: tuple(range(-1, width - 1)) for width in range(1, 257)},
        "_chain": lambda parents: tuple(parents) == tuple(range(-1, len(parents) - 1)),
    }
    execute(
        selected(
            SOURCE.read_bytes(),
            {
                "Record",
                "_Rows",
                "_stream_inputs",
                "_row_integer",
                "_array_shape",
                "_attention_metadata",
                "_attention_inputs",
                "_forward_cache_inputs",
                "_forward_windows",
                "_implicit_stream_inputs",
                "_attend",
                "_rows_forward",
                "hidden_rows",
            },
        ),
        namespace,
    )
    module = ModuleType("tensorfold.kernels.qwen.dense.v1")
    parents = {"islice": islice, "index": index, "Sequence": Sequence}
    execute(selected((NATIVE_SOURCE / "row_attention.py").read_bytes(), {"_parents"}), parents)
    tree = {}
    execute(selected((NATIVE_SOURCE / "lane_tree.py").read_bytes(), {"tree_paths"}), tree)
    module.row_attention = SimpleNamespace(
        SPLIT=4, CK=128, _parents=parents["_parents"], row_sdpa=lambda *args: ("raw", args)
    )
    module.exact_attention = SimpleNamespace(exact_sdpa=lambda *args: ("exact", args))
    module.lane_tree = SimpleNamespace(tree_paths=tree["tree_paths"], MAX_DEPTH=128, MAX_TREE=32)
    module.row_streams = SimpleNamespace(GROUP=8)
    with patch.dict(sys.modules, {"tensorfold.kernels.qwen.dense.v1": module}):
        yield namespace, module


def cache(*, step=256, offset=5, capacity=16):
    item = SimpleNamespace(
        keys=Array(dtype="BF16", shape=(1, 2, capacity, 32)),
        values=Array(dtype="BF16", shape=(1, 2, capacity, 32)),
        offset=offset,
    )
    if step is not None:
        item.step = step
    item.trim = lambda count: None

    def update(keys, values):
        previous = item.offset
        item.offset += keys.shape[2]
        capacity = item.keys.shape[2]
        if step is None:
            capacity += keys.shape[2]
        elif item.offset > capacity:
            capacity = (previous if previous % step else capacity) + ((keys.shape[2] + step - 1) // step) * step
        item.keys = Array(dtype=keys.dtype, shape=(1, 2, capacity, 32))
        item.values = Array(dtype=values.dtype, shape=item.keys.shape)
        return keys, values

    item.update_and_fetch = update
    return item


def core_fixture(api):
    calls = []
    norm = SimpleNamespace(weight="weight", eps="eps")
    attn = SimpleNamespace(o_proj="out", num_attention_heads=4, num_key_value_heads=2, head_dim=32, scale=0.125)
    layers = [
        SimpleNamespace(
            input_layernorm=norm, post_attention_layernorm=norm, self_attn=attn, mlp=SimpleNamespace(down_proj="down")
        )
        for _ in range(2)
    ]

    class Embedding:
        weight = Array(dtype="BF16", shape=(64, 128))

        def __call__(self, tokens):
            calls.append(("embed", tokens))
            return "hidden"

    core = SimpleNamespace(layers=layers, norm=norm, embed_tokens=Embedding())
    api["_token_ids"] = lambda windows: calls.append(("tokens", windows)) or "ids"
    api["add_norm"] = lambda hidden, pending, *args: (hidden, "normed")
    api["_attention"] = lambda attn, x, items, rows: calls.append(("attention", tuple(items))) or "attention"
    api["project"] = lambda module, x: x
    api["mlp_act"] = lambda x: x
    api["_gate_up"] = lambda mlp, x: x
    api["mx"].async_eval = lambda *args: calls.append(("async", args))
    return core, calls


class ForwardBoundary(unittest.TestCase):
    def test_invalid_attention_inputs_refuse_before_record_or_cache_update(self):
        for bad in ("rank", "value_width", "group", "dtype", "simd_width", "threads", "scale", "offset"):
            with self.subTest(bad=bad), fixture() as (api, _):
                item, record = cache(), []
                q = Array(dtype="F32", shape=(1, 4, 3, 32))
                k = Array(dtype="BF16", shape=(1, 2, 3, 32))
                v = Array(dtype="BF16", shape=k.shape)
                scale = 0.125
                if bad == "rank":
                    q.ndim = 3
                elif bad == "value_width":
                    v.shape = (1, 2, 3, 64)
                elif bad == "group":
                    q.shape = (1, 3, 3, 32)
                elif bad == "dtype":
                    k.dtype = "F16"
                elif bad == "simd_width":
                    q.shape = k.shape = v.shape = (1, 2, 3, 33)
                elif bad == "threads":
                    q.shape = (1, 18, 3, 32)
                elif bad == "scale":
                    scale = math.inf
                else:
                    item.offset = -0.5
                item.update_and_fetch = lambda *args: self.fail("invalid metadata reached update")
                with self.assertRaises(ValueError):
                    api["_attend"](SimpleNamespace(scale=scale), q, k, v, item, (-1, 0, 1), True, record)
                self.assertEqual(record, [])

    def test_exact_delegation_preserves_packed_rotating_and_value_head_width(self):
        for kind in ("packed", "rotating"):
            with self.subTest(kind=kind), fixture(raw=False) as (api, module):
                item = cache()
                item.keys = item.values = (object(), object(), object())
                if kind == "packed":
                    item.bits = 4
                else:
                    item.max_size = 4
                q = Array(dtype="F32", shape=(1, 4, 3, 32))
                k = Array(dtype="F16", shape=(1, 2, 3, 32))
                v = Array(dtype="F16", shape=(1, 2, 3, 64))
                returned = (object(), object())
                writes = []

                def update(*args):
                    writes.append(args)
                    item.offset += 3
                    return returned

                item.update_and_fetch = update
                record = []
                actual = api["_attend"](SimpleNamespace(scale=0.125), q, k, v, item, (-1, 0, 1), True, record)
                self.assertEqual(actual[0], "exact")
                self.assertIs(actual[1][1], returned[0])
                self.assertIs(actual[1][2], returned[1])
                self.assertEqual(writes, [(k, v)])
                self.assertEqual(record, [("kv", k, v)])

    def test_concat_and_step_growth_valid_metadata_preserves_raw_call_order(self):
        for step, capacity in ((None, 5), (256, 5), (256, 16)):
            with self.subTest(step=step, capacity=capacity), fixture() as (api, _):
                item = cache(step=step, capacity=capacity)
                q = Array(dtype="F32", shape=(1, 4, 3, 32))
                k = Array(dtype="BF16", shape=(1, 2, 3, 32))
                v = Array(dtype="BF16", shape=k.shape)
                record = []
                actual = api["_attend"](SimpleNamespace(scale=0.125), q, k, v, item, (-1, 0, 1), True, record)
                self.assertEqual(actual[0], "raw")
                self.assertEqual(actual[1][-2:], (5, (-1, 0, 1)))
                self.assertEqual(record, [("kv", k, v)])

    def test_complete_late_stream_cache_preflight_precedes_embedding(self):
        for bad in ("shape", "dtype", "offset", "alias", "growth", "prepared"):
            with self.subTest(bad=bad), fixture() as (api, _):
                core, calls = core_fixture(api)
                caches = [[cache(), cache()], [cache(), cache()]]
                item = caches[1][1]
                if bad == "shape":
                    item.keys.shape = (1, 3, 16, 32)
                elif bad == "dtype":
                    item.values.dtype = "F16"
                elif bad == "offset":
                    item.offset = 6
                elif bad == "alias":
                    caches[1][1] = caches[0][0]
                elif bad == "growth":
                    item.keys.shape = item.values.shape = (1, 2, 5, 32)
                    item.step = -1
                else:
                    core.layers[1].is_linear = True
                    core.layers[1].linear_attn = SimpleNamespace(
                        num_k_heads=2, num_v_heads=4, head_k_dim=32, head_v_dim=32, conv_kernel_size=4, conv_dim=256
                    )

                    class State(list):
                        lengths = Array(dtype="I32", shape=(1,))
                        def advance(self, count):
                            return None

                    caches[0][1], caches[1][1] = State([None, None]), State([None, None])
                with self.assertRaises(ValueError):
                    api["_rows_forward"](core, [[1], [2]], [[-1], [-1]], caches, [5, 5])
                self.assertEqual(calls, [])

    def test_host_token_range_and_bounded_producers_precede_embedding(self):
        for token in (-1, True, 0.5, 64, 1 << 32):
            with self.subTest(token=token), fixture() as (api, _):
                core, calls = core_fixture(api)
                with self.assertRaises(ValueError):
                    api["_rows_forward"](core, [[token]], [[-1]], [[cache(), cache()]], [5])
                self.assertEqual(calls, [])

        class Changed(Sequence):
            def __init__(self):
                self.reads = 0

            def __len__(self):
                return 1

            def __getitem__(self, at):
                self.reads += 1
                if self.reads > 2:
                    raise AssertionError("over-consumed token producer")
                return 1

        with fixture() as (api, _):
            core, calls = core_fixture(api)
            window = Changed()
            with self.assertRaises(ValueError):
                api["_rows_forward"](core, [window], [[-1]], [[cache(), cache()]], [5])
            self.assertEqual(window.reads, 2)
            self.assertEqual(calls, [])

    def test_vision_signed_spans_are_checked_without_clamping_negative_positions(self):
        for shift, admitted in (
            (-10, True),
            (-(1 << 31), True),
            ((1 << 31) - 6, True),
            ((1 << 31) - 5, False),
            (True, False),
            (0.25, False),
        ):
            with self.subTest(shift=shift), fixture(raw=False) as (api, _):
                core, calls = core_fixture(api)
                items = [cache(), cache()]
                for item in items:
                    item.vision_rope_delta = shift
                if admitted:
                    result, _ = api["_rows_forward"](core, [[1]], [[-1]], [items], [5])
                    self.assertEqual(result, "normed")
                else:
                    with self.assertRaises(ValueError):
                        api["_rows_forward"](core, [[1]], [[-1]], [items], [5])
                    self.assertEqual(calls, [])

    def test_valid_forward_retains_original_lazy_identity_and_layer_order(self):
        with fixture(raw=False) as (api, _):
            core, calls = core_fixture(api)
            lazy = Array(dtype="F32", shape=(2,))
            items = [cache(), cache()]
            actual, rows = api["_rows_forward"](core, [lazy], [[-1, 0]], [items], [5])
            self.assertEqual(actual, "normed")
            self.assertIs(calls[0][1][0], lazy)
            self.assertEqual(
                [call for call in calls if call[0] == "attention"],
                [("attention", (items[0],)), ("attention", (items[1],))],
            )
            self.assertEqual(rows.host_positions, (5, 6))
            self.assertEqual(rows.positions.values, (5, 6))

    def test_implicit_head_dim_preserves_original_projection_row_geometry(self):
        with fixture() as (api, _):
            core, calls = core_fixture(api)
            attn = core.layers[0].self_attn
            del attn.head_dim
            attn.q_proj = SimpleNamespace(weight=Array(shape=(256, 128)))
            api["_rows_forward"](core, [[1]], [[-1]], [[cache(), cache()]], [5])
            self.assertEqual(len([call for call in calls if call[0] == "attention"]), 2)

    def test_implicit_facade_producers_are_bounded_before_default_tree_creation(self):
        class Changed(Sequence):
            def __init__(self, value, count=1):
                self.value, self.count, self.reads = value, count, 0

            def __len__(self):
                return self.count

            def __getitem__(self, at):
                self.reads += 1
                if self.reads > self.count + 1:
                    raise AssertionError("unbounded implicit input producer")
                return self.value

        for producer in ("windows", "caches", "layers"):
            with self.subTest(producer=producer), fixture() as (api, _):
                core, calls = core_fixture(api)
                windows, caches = [[1]], [[cache(), cache()]]
                if producer == "windows":
                    changing = windows = Changed([1])
                elif producer == "caches":
                    changing = caches = Changed([cache(), cache()])
                else:
                    changing = Changed(cache(), count=2)
                    caches = [changing]
                with self.assertRaises(ValueError):
                    api["hidden_rows"](core, windows, caches)
                self.assertEqual(changing.reads, changing.count + 1)
                self.assertEqual(calls, [])
        with fixture() as (api, _):
            core, calls = core_fixture(api)
            with self.assertRaises(ValueError):
                api["hidden_rows"](core, [[1] * 129], [[cache(), cache()]])
            self.assertEqual(calls, [])

    def test_valid_implicit_facade_preserves_lazy_window_and_offset_for_admission(self):
        with fixture(raw=False) as (api, _):
            core, _ = core_fixture(api)
            lazy = Array(dtype="I32", shape=(2,))
            items = [cache(), cache()]
            actual, records = api["hidden_rows"](core, [lazy], [items])
            self.assertEqual(actual, "normed")
            self.assertEqual((records[0].start, records[0].parents), (5, (-1, 0)))
            items[0].offset = 0.75
            with self.assertRaises(ValueError):
                api["hidden_rows"](core, [lazy], [items])

    def recurrent_fixture(self, api, *, widths=(1,), trees=False, dk=32, dv=32, nv=4):
        core, calls = core_fixture(api)
        channels = 4 * dk + nv * dv
        gdn = SimpleNamespace(num_k_heads=2, num_v_heads=nv, head_k_dim=dk, head_v_dim=dv,
            conv_kernel_size=4, conv_dim=channels, conv1d=SimpleNamespace(weight=Array(shape=(channels, 4, 1))),
            A_log=Array(shape=(nv,)), dt_bias=Array(shape=(nv,)))
        core.layers[1] = SimpleNamespace(is_linear=True, linear_attn=gdn)

        class State(list):
            left_padding = lengths = None
            def advance(self, count):
                self.advanced = count

        parents = [tuple(range(-1, width - 1)) for width in widths]
        if trees:
            parents[-1] = (-1, *([0] * (widths[-1] - 1)))
        items = [[cache(), State([None, None])] for _ in widths]
        return core, calls, gdn, items, parents

    def test_recurrent_callee_geometry_and_spans_refuse_before_prior_attention_write(self):
        for bad in ("unequal", "taps", "parameters", "state_span"):
            with self.subTest(bad=bad), fixture() as (api, _):
                core, calls, gdn, items, parents = self.recurrent_fixture(api,
                    dk=1024 if bad == "state_span" else 32,
                    dv=1024 if bad == "state_span" else (64 if bad == "unequal" else 32),
                    nv=4098 if bad == "state_span" else 4)
                if bad == "taps":
                    gdn.conv1d.weight.shape = (gdn.conv_dim, 3, 1)
                    gdn.conv1d.weight.size = gdn.conv_dim * 3
                elif bad == "parameters":
                    gdn.A_log.size = 3
                with self.assertRaises(ValueError):
                    api["_rows_forward"](core, [[1]], parents, items, [5])
                self.assertEqual(calls, [])

    def test_recurrent_width_limit_uses_same_selected_group_chain_as_callee(self):
        for widths in ((33,), (40, 3)):
            with self.subTest(widths=widths), fixture() as (api, _):
                core, calls, _, items, parents = self.recurrent_fixture(api, widths=widths, trees=True)
                with self.assertRaises(ValueError):
                    api["_rows_forward"](core, [[1] * width for width in widths], parents, items, [5] * len(widths))
                self.assertEqual(calls, [])
        with fixture() as (api, _):
            core, _, _, items, parents = self.recurrent_fixture(api, widths=(40, 3))
            rows = api["_Rows"](parents, [5, 5])
            admitted = api["_forward_cache_inputs"](core.layers, items, rows)
            self.assertEqual(admitted, tuple(tuple(item) for item in items))

    def test_valid_recurrent_empty_and_current_state_metadata_is_borrowed_without_read(self):
        with fixture() as (api, _):
            core, _, gdn, items, parents = self.recurrent_fixture(api)
            rows = api["_Rows"](parents, [5])
            api["_forward_cache_inputs"](core.layers, items, rows)
            conv = Array(dtype="BF16", shape=(1, 3, gdn.conv_dim))
            state = Array(dtype="F32", shape=(1, 4, 32, 32))
            items[0][1][:] = [conv, state]
            admitted = api["_forward_cache_inputs"](core.layers, items, rows)
            self.assertIs(admitted[0][1][0], conv)
            self.assertIs(admitted[0][1][1], state)

    def test_grouped_recurrent_input_and_tail_spans_use_selected_launch_rows(self):
        with fixture() as (api, _):
            core, calls, _, items, parents = self.recurrent_fixture(api,
                widths=(16,) * 8, nv=1 << 20)
            with self.assertRaises(ValueError):
                api["_rows_forward"](core, [[1] * 16 for _ in items], parents, items, [5] * 8)
            self.assertEqual(calls, [])
        with fixture() as (api, _):
            core, _, _, items, parents = self.recurrent_fixture(api,
                widths=(16,) * 8, nv=65536)
            rows = api["_Rows"](parents, [5] * 8)
            admitted = api["_forward_cache_inputs"](core.layers, items, rows)
            self.assertEqual(admitted, tuple(tuple(item) for item in items))
        with fixture() as (api, module):
            # Separate launches and separate state owners are not combined
            # into a nonexistent all-group tensor span.
            module.row_streams.GROUP = 1
            core, _, _, items, parents = self.recurrent_fixture(api,
                widths=(16,) * 8, nv=1 << 20)
            rows = api["_Rows"](parents, [5] * 8)
            api["_forward_cache_inputs"](core.layers, items, rows)

    def test_convolution_weight_full_tap_span_is_proved_before_cache_write(self):
        with fixture() as (api, _):
            core, calls, gdn, items, parents = self.recurrent_fixture(api, nv=2097151)
            gdn.num_k_heads = gdn.num_v_heads
            gdn.conv_dim = 3 * gdn.num_v_heads * 32
            gdn.conv_kernel_size = 11
            gdn.conv1d.weight = Array(shape=(gdn.conv_dim, 11, 1))
            # Per-owner state and ten-row history fit; signed CW[c*TAPS+r]
            # must also cover the eleventh weight row without wrapping.
            self.assertLess(gdn.num_v_heads * 32 * 32, 1 << 31)
            self.assertLess(10 * gdn.conv_dim, 1 << 31)
            self.assertGreater(gdn.conv1d.weight.size, (1 << 31) - 1)
            with self.assertRaises(ValueError):
                api["_rows_forward"](core, [[1]], parents, items, [5])
            self.assertEqual(calls, [])

    def test_unsigned_stacked_total_with_valid_signed_qkv_prefix_is_preserved(self):
        with fixture() as (api, _):
            heads, width = 2064889, 8
            core, _, gdn, items, parents = self.recurrent_fixture(api, widths=(width,), nv=heads)
            gdn.num_k_heads = heads
            gdn.conv_dim = 3 * heads * 32
            gdn.conv_kernel_size = 1
            gdn.conv1d.weight = Array(shape=(gdn.conv_dim, 1, 1))
            stacked = gdn.conv_dim + heads * 32 + 2 * heads
            self.assertGreater(width * stacked, (1 << 31) - 1)
            self.assertLessEqual((width - 1) * stacked + gdn.conv_dim - 1, (1 << 31) - 1)
            rows = api["_Rows"](parents, [5])
            admitted = api["_forward_cache_inputs"](core.layers, items, rows)
            self.assertIs(admitted[0][1], items[0][1])
        with fixture() as (api, _):
            core, _, _, items, parents = self.recurrent_fixture(api, dk=1024, dv=1024, nv=4096)
            rows = api["_Rows"](parents, [5])
            # A uint maximum last state element permits 2**32 elements.
            api["_forward_cache_inputs"](core.layers, items, rows)

    def test_exact_sdk_last_signed_position_preserves_its_python_end_offset(self):
        with fixture(raw=False) as (api, _):
            item = cache(offset=(1 << 31) - 1)
            item.bits = 4
            item.keys = item.values = (object(), object(), object())
            q = Array(dtype="F32", shape=(1, 4, 1, 32))
            k = Array(dtype="F16", shape=(1, 2, 1, 32))
            v = Array(dtype="F16", shape=(1, 2, 1, 32))
            def update(keys, values):
                item.offset += 1
                return keys, values
            item.update_and_fetch = update
            actual = api["_attend"](SimpleNamespace(scale=.125), q, k, v, item, (-1,), True, [])
            self.assertEqual(actual[0], "exact")
            self.assertEqual(item.offset, 1 << 31)


if __name__ == "__main__":
    unittest.main()
