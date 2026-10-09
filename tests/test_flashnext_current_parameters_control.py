"""Current-value descriptor authority controls; stdlib only, no MLX/numerical run."""
from __future__ import annotations

import __future__
import ast
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
import builtins
import unittest

ROOT = Path(__file__).resolve().parents[1]
DECODE = ROOT / "src/tensorfold/families/qwen4_exp/decode.py"
EMBED = ROOT / "src/tensorfold/kernels/qwen/flash_next/v1/embed.py"


class Descriptor:
    """Opaque mutable metadata: operations record values, never numerical math."""
    def __init__(self, value, shape=(2, 8), dtype="uint32"):
        self.value, self.shape, self.dtype = value, shape, dtype
        self.size = 1
        for dimension in shape:
            self.size *= dimension

    def astype(self, dtype):
        return Descriptor(("astype", dtype, self.value), self.shape, dtype)

    def __getitem__(self, key):
        return Descriptor(("slice", self.value, repr(key)), self.shape)

    def __radd__(self, value):
        return Descriptor(("add", value, self.value), self.shape)

    def __add__(self, value):
        return self.__radd__(value)


class Module:
    def __init__(self):
        self.initialized = True

    def freeze(self):
        self.frozen = True

    def __getitem__(self, name):
        return getattr(self, name)


class Linear(Module):
    def __init__(self, value, *, bits=4, group=32, shape=(2, 8)):
        super().__init__()
        self.bits, self.group_size, self.mode = bits, group, "affine"
        self.weight = Descriptor((value, "weight"), shape)
        self.scales = Descriptor((value, "scale"), (shape[0], 2), "bfloat16")
        self.biases = Descriptor((value, "bias"), (shape[0], 2), "bfloat16")


def concat(parts, axis=0):
    parts = list(parts)
    shape = list(parts[0].shape)
    shape[axis] = sum(part.shape[axis] for part in parts)
    return Descriptor(tuple(part.value for part in parts), tuple(shape))


def repeat(value, copies, axis):
    shape = list(value.shape)
    shape[axis] *= copies
    return Descriptor(("repeat", copies, value.value), tuple(shape))


MX = SimpleNamespace(concatenate=concat, repeat=repeat, eval=lambda *a: None,
                     contiguous=lambda a: Descriptor(("contiguous", a.value), a.shape),
                     array=lambda a, dtype=None: Descriptor(tuple(a)),
                     float32="float32", bfloat16="bfloat16", uint32="uint32")


def definitions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def api():
    base = definitions(ROOT / "src/tensorfold/kernels/qwen/flash_next/v1/base.py", {"QWeights"},
                       {"mx": MX, "AFFINE_BITS": (2, 3, 4, 5, 6, 8)})
    embed = definitions(EMBED, {"PleTables", "_selected_shard_rows"}, {"mx": MX, "AFFINE_BITS": (2, 3, 4, 5, 6, 8)})
    namespace = {"mx": MX, "nn": SimpleNamespace(Module=Module, QuantizedLinear=Linear),
                 "base": SimpleNamespace(QWeights=base["QWeights"]), "embed": SimpleNamespace(**embed),
                 "DENSE": "lane", "_NONE": ("none", (), None)}
    definitions(DECODE, {"_PreparedLinear", "_StackPlan", "_stacked", "_Split", "_dense_rows",
                         "_HC", "FusedDecode", "_lane_project", "project"}, namespace)
    return namespace


def connection(label):
    return SimpleNamespace(input_mix_weight_down=Linear(label + "down"),
                           block_inject_weight=Linear(label + "inject"),
                           input_mix_weight_up=Linear(label + "up"),
                           hc_norm=SimpleNamespace(weight=Descriptor(label + "norm")))


def gdn(label):
    return SimpleNamespace(**{name: Linear(label + name) for name in
                              ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")},
                           conv1d=SimpleNamespace(weight=Descriptor(label + "conv", (8, 4, 1))))


class Layer(SimpleNamespace):
    def __contains__(self, key):
        return hasattr(self, key)


def model():
    layer = Layer(is_linear=True, linear_attn=gdn("g"),
                  attn_hyper_connection=connection("a"), mlp_hyper_connection=connection("m"),
                  mlp=SimpleNamespace(gate=SimpleNamespace(weight=Descriptor("router")),
                                      shared_expert_gate=SimpleNamespace(weight=Descriptor("shared"))))
    return SimpleNamespace(layers=[layer], args=SimpleNamespace(hc_count=4, rms_norm_eps=.01),
                           model=SimpleNamespace(hyper_connection_mixer=connection("final")))


class IDs:
    def __init__(self, values, kind="u"):
        self.values, self.ndim, self.size = tuple(values), 1, len(values)
        self.dtype = SimpleNamespace(kind=kind)

    def min(self):
        return min(self.values)

    def max(self):
        return max(self.values)

    def astype(self, dtype):
        return self


@dataclass
class HeadArgs:
    num_hidden_layers: int = 4
    layer_types: object = None
    ple_layer_ids: object = None


def runtime_api(n):
    path = ROOT / "src/tensorfold/families/qwen4_exp/runtime.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlashNext")
    names = {"_mtp_scales", "_draft_head", "_warm_sparse", "draft"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    view = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_MTPModelView")
    cut_path = ROOT / "src/tensorfold/families/qwen4_exp/draft_head.py"
    cut = next(node for node in ast.parse(cut_path.read_text()).body
               if isinstance(node, ast.FunctionDef) and node.name == "cut_head")
    mx = SimpleNamespace(**vars(MX))
    mx.take = lambda value, ids, axis: Descriptor(("take", value.value, ids.value),
                                                 (len(ids.value), value.shape[1]), value.dtype)
    mx.array = lambda ids, dtype=None: Descriptor(ids.values if isinstance(ids, IDs) else tuple(ids))
    attention = SimpleNamespace(warm_decode=lambda **kwargs: None)
    def imports(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tensorfold.families.qwen4_exp.decode":
            return SimpleNamespace(_PreparedLinear=n["_PreparedLinear"])
        if name == "tensorfold.kernels.qwen.flash_next.v1.base":
            return n["base"]
        if name == "tensorfold.families.qwen4_exp.draft_head":
            return SimpleNamespace(cut_head=scope["cut_head"])
        if name == "tensorfold.kernels.qwen.flash_next.v1":
            return SimpleNamespace(attention=attention)
        return builtins.__import__(name, globals, locals, fromlist, level)
    scope = {"__builtins__": {**vars(builtins), "__import__": imports}, "mx": mx,
             "np": SimpleNamespace(asarray=lambda x: x, uint32="uint32"), "nn": n["nn"]}
    owner = ast.ClassDef(name="CurrentRuntime", bases=[], keywords=[],
                        body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cut, view, owner], type_ignores=[]))
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), scope)
    return scope, attention


class CurrentParameterControls(unittest.TestCase):
    def test_stack_preserves_standard_sources_and_observes_same_descriptor_mutation(self):
        n = api()
        source = [Linear("a"), Linear("b", group=64)]
        original = [(x.weight, x.scales, x.biases) for x in source]
        first, cuts = n["_stacked"](source)
        self.assertTrue(first.initialized)
        self.assertEqual(cuts, [2])
        self.assertEqual(first.group_size, 32)
        self.assertEqual(original, [(x.weight, x.scales, x.biases) for x in source])
        source[0].weight.value = "same-object new contents"
        source[1].scales = Descriptor("replacement scale", (2, 2))
        second, _ = n["_stacked"](source)
        self.assertNotEqual(first.weight.value, second.weight.value)
        self.assertNotEqual(first.scales.value, second.scales.value)
        self.assertEqual(original[0][0], source[0].weight)

    def test_hc_keeps_sources_and_reads_all_current_parameters(self):
        n, source = api(), connection("hc")
        original = source.input_mix_weight_down.weight
        first = n["_HC"](source, inject=True)
        self.assertIs(source.input_mix_weight_down.weight, original)
        original.value = "new down"
        source.hc_norm.weight.value = "new norm"
        source.input_mix_weight_up = Linear("replacement up")
        second = n["_HC"](source, inject=True)
        self.assertNotEqual(first.down.weight.value, second.down.weight.value)
        self.assertNotEqual(first.scale.value, second.scale.value)
        self.assertNotEqual(first.up.weight.value, second.up.weight.value)

    def test_fused_entries_read_current_modules_conv_router_hc_and_config(self):
        n, source = api(), model()
        fused = n["FusedDecode"](source)
        first = fused._entry(0)
        source.layers[0].linear_attn = gdn("replacement")
        source.layers[0].mlp.gate.weight.value = "new router"
        source.layers[0].attn_hyper_connection.hc_norm.weight.value = "new HC norm"
        source.args = SimpleNamespace(hc_count=8, rms_norm_eps=.02)
        second = fused._entry(0)
        self.assertIs(second["gdn"][-1], source.layers[0].linear_attn)
        self.assertNotEqual(first["gdn"][1].value, second["gdn"][1].value)
        self.assertNotEqual(first["moe"][1].value, second["moe"][1].value)
        self.assertNotEqual(first["attn_hc"].scale.value, second["attn_hc"].scale.value)
        self.assertIs(fused.cfg, source.args)
        self.assertEqual(fused.streams, 8)
        self.assertEqual(fused.eps.value, (.02,))
        self.assertFalse({"cfg", "layers", "mixer", "ple_parts", "ple_tables"} & vars(fused).keys())

    def test_prefill_plan_resolves_replacement_and_current_bit_partition(self):
        n, owner = api(), gdn("g")
        names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        plan = n["_StackPlan"](owner, names)
        first = plan.current()
        owner.in_proj_qkv = Linear("new")
        second = plan.current()
        self.assertNotEqual(first.weight.value, second.weight.value)
        owner.in_proj_a.bits = 8
        self.assertIsNone(plan.current())

    def test_lane_binding_reads_weight_scale_bias_format_and_descriptor_current_values(self):
        n, source = api(), Linear("l", shape=(64, 8))
        calls = []
        lane = SimpleNamespace(MAX_ROWS=8, NT=32,
             tile_weight=lambda w, nt, group, bits: Descriptor((w.value, nt, group, bits), w.shape),
             pack_scales=lambda s, b: Descriptor((s.value, b.value)),
             lane_matmul=lambda x, w, sbt, **kw: calls.append((w.value, sbt.value, kw)) or Descriptor("out"))
        n["lane_qmm"] = lane
        x = Descriptor("input", (1, 64))
        n["_lane_project"](x, source)
        source.weight.value = "new weight"
        source.scales.value = "new scale"
        source.biases = Descriptor("new bias")
        source.bits, source.group_size = 8, 128
        n["_lane_project"](x, source)
        self.assertNotEqual(calls[0], calls[1])
        self.assertEqual(calls[1][2]["group"], 64)
        self.assertEqual(calls[1][0][-1], 8)
        self.assertNotIn("_lane", n)

    def test_ple_grouping_is_current_and_never_rewrites_shards(self):
        n = api()
        shards = [Linear(str(index)) for index in range(8)]
        emb = SimpleNamespace(dims=64, shards=shards, quant_bits=4, quant_group=32, table_scale=1.0, host=None)
        originals = [(s.weight, s.scales, s.biases) for s in shards]
        plan = n["embed"].PleTables(emb)
        first = plan.current()
        first_value = first.shards[0].weight.value
        shards[0].weight.value = "new first shard"
        shards[1].biases = Descriptor("new second bias", (2, 2), "bfloat16")
        second = plan.current()
        self.assertNotEqual(first_value, second.shards[0].weight.value)
        self.assertIs(second.shards[1].biases, shards[1].biases)
        self.assertEqual(second.counts, (2,) * 8)
        self.assertFalse(hasattr(second, "weights"))
        self.assertIs(shards[0].weight, originals[0][0])
        self.assertEqual(set(vars(plan)), {"embedding"})
        emb.host, emb.table_scale = object(), .5
        host = plan.current()
        self.assertIs(host.host, emb.host)
        self.assertEqual(host.scale, .5)
        self.assertFalse(hasattr(host, "weights"))

    def test_selected_shard_plan_matches_independent_flat_indexing(self):
        plan = api()["embed"]._selected_shard_rows
        for counts in ((2,) * 8, (0, 2, 0, 3, 0), (1, 0, 0), (3, 1, 2)):
            full = [(shard, local) for shard, count in enumerate(counts) for local in range(count)]
            for requested in (list(range(len(full))), list(reversed(range(len(full)))), [0] * 7, []):
                groups, inverse = plan(counts, requested)
                grouped = [(shard, local) for shard, ids in enumerate(groups) for local in ids]
                self.assertEqual([grouped[index] for index in inverse], [full[index] for index in requested])
        for invalid in (-1, 6, True, 1.5, "1"):
            with self.assertRaises((TypeError, ValueError)):
                plan((3, 1, 2), [invalid])
        with self.assertRaises(ValueError):
            plan((1, -1), [])

    def test_attention_entry_reads_all_replaced_norms_and_indexer(self):
        n, source = api(), model()
        attn = SimpleNamespace(q_proj=Linear("q"), k_proj=Linear("k"), v_proj=Linear("v"),
                               q_norm=SimpleNamespace(weight=Descriptor("qn")),
                               k_norm=SimpleNamespace(weight=Descriptor("kn")),
                               indexer=SimpleNamespace(index_qk_proj=Linear("iq"),
                                                       q_layernorm=SimpleNamespace(weight=Descriptor("iqn")),
                                                       k_layernorm=SimpleNamespace(weight=Descriptor("ikn"))))
        source.layers[0].is_linear, source.layers[0].self_attn = False, attn
        fused = n["FusedDecode"](source)
        first = fused._entry(0)["attn"]
        for norm in (attn.q_norm, attn.k_norm, attn.indexer.q_layernorm, attn.indexer.k_layernorm):
            norm.weight.value = (norm.weight.value, "new")
        second = fused._entry(0)["attn"]
        self.assertTrue(all(a.value != b.value for a, b in zip(first[1:5], second[1:5])))

    def test_simd_current_weight_check_does_not_authorize_another_value_of_same_shape(self):
        n, source = api(), Linear("simd")
        checked, kinds = [], []
        def check(weight, scales, biases, *, group_size):
            checked.append((weight.value, scales.value, biases.value, group_size))
            return len(checked) == 1
        n["DENSE"] = "simd"
        n["simd_qmm"] = SimpleNamespace(fits=lambda linear: True, check=check,
             qmm=lambda x, w, sc, bi, group, *, kind: kinds.append(kind) or Descriptor("out"))
        x = Descriptor("input", (1, 64))
        n["project"](x, source)
        source.weight.value = "same identity new SIMD contents"
        n["project"](x, source)
        self.assertEqual(kinds, [None, "mma"])
        self.assertEqual(len(checked), 2)
        self.assertNotEqual(checked[0], checked[1])

    def test_current_ple_gate_and_conv_metadata_follow_replaced_values(self):
        n, source = api(), model()
        n["DENSE"] = "rows"
        fused = n["FusedDecode"](source)
        ple = SimpleNamespace(key_proj=Linear("key"), value_proj=Linear("value"),
            norm_key=SimpleNamespace(weight=Descriptor("keynorm")),
            norm_query=SimpleNamespace(weight=Descriptor("querynorm")),
            norm_conv=SimpleNamespace(weight=Descriptor("convnorm")),
            conv1d=SimpleNamespace(weight=Descriptor("pleconv", (8, 4, 1))))
        first = fused._ple_parts(ple)
        ple.key_proj = Linear("replacement key")
        for norm in (ple.norm_key, ple.norm_query, ple.norm_conv):
            norm.weight.value = (norm.weight.value, "new")
        ple.conv1d.weight.value = "new convolution"
        second = fused._ple_parts(ple)
        self.assertNotEqual(first[0].weight.value, second[0].weight.value)
        self.assertTrue(all(a.value != b.value for a, b in zip(first[1:], second[1:])))

    def test_prefill_derives_only_the_current_layer_hyperconnections(self):
        n, source = api(), model()
        fused = n["FusedDecode"](source)
        def forbidden(*args):
            raise AssertionError("prefill must not derive unused stacked/router weights")
        n["_stacked"], n["_dense_rows"] = forbidden, forbidden
        entry = fused._entry(0, connections_only=True)
        self.assertEqual(set(entry), {"attn_hc", "mlp_hc"})
        path = ROOT / "src/tensorfold/kernels/qwen/flash_next/v1/prefill_hc.py"
        tree = ast.parse(path.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "_entry"]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(any(k.arg == "connections_only" and isinstance(k.value, ast.Constant)
                                 and k.value.value is True for k in call.keywords) for call in calls))
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr == "layers"
                             and isinstance(node.value, ast.Name) and node.value.id == "fused"
                             for node in ast.walk(tree)))

    def test_mtp_norm_and_draft_head_getters_bind_current_values_without_retention(self):
        n = api()
        scope, _ = runtime_api(n)
        caller = scope["CurrentRuntime"]()
        caller.mtp = SimpleNamespace(pre_fc_norm_embedding=SimpleNamespace(weight=Descriptor("embedding norm")),
                                     pre_fc_norm_hidden=SimpleNamespace(weight=Descriptor("hidden norm")))
        caller.model = SimpleNamespace(lm_head=Linear("head", shape=(4, 8)))
        caller._draft_cut_ids = IDs([0, 2, 3])
        first_norm, first_head = caller._mtp_scales, caller._draft_head
        caller.mtp.pre_fc_norm_hidden.weight.value = "same descriptor new norm"
        caller.model.lm_head.weight.value = "same descriptor new vocabulary weight"
        second_norm, second_head = caller._mtp_scales, caller._draft_head
        self.assertNotEqual(first_norm[1].value, second_norm[1].value)
        self.assertNotEqual(first_head.weight.value, second_head.weight.value)
        self.assertEqual(second_head.bits, caller.model.lm_head.bits)
        self.assertFalse({"_mtp_scales", "_draft_head"} & vars(caller).keys())
        for ids in (IDs([-1]), IDs([4]), IDs([]), IDs([1], kind="f")):
            caller._draft_cut_ids = ids
            with self.assertRaises(ValueError):
                caller._draft_head
        caller.model.lm_head = Linear("large metadata only", shape=(2**32, 8))
        caller._draft_cut_ids = IDs([2**31, 2**32 - 1])
        self.assertEqual(caller._draft_head.weight.shape[0], 2)
        caller._draft_cut_ids = IDs([2**32])
        with self.assertRaises(ValueError):
            caller._draft_head
        caller._draft_cut_ids = None
        self.assertIsNone(caller._draft_head)

    def test_mtp_model_view_tracks_replaced_layer_list_mixer_and_embedding(self):
        scope, _ = runtime_api(api())
        backbone = SimpleNamespace(args=HeadArgs(), model=SimpleNamespace(embed_tokens=object()))
        head = SimpleNamespace(layers=[object()], hyper_connection_mixer=object())
        view = scope["_MTPModelView"](backbone, head)
        first = view.model
        head.layers = [object(), object()]
        head.hyper_connection_mixer = object()
        backbone.model.embed_tokens = object()
        self.assertIs(view.layers, head.layers)
        self.assertIs(view.model.hyper_connection_mixer, head.hyper_connection_mixer)
        self.assertIs(view.model.embed_tokens, backbone.model.embed_tokens)
        self.assertIsNot(first.hyper_connection_mixer, view.model.hyper_connection_mixer)
        self.assertEqual(view.args.num_hidden_layers, 1)
        self.assertEqual(view.args.layer_types, ["sparse_attention"])
        self.assertEqual(view.args.ple_layer_ids, [])

    def test_sparse_warmer_prepares_only_first_matching_attention_layer(self):
        scope, attention = runtime_api(api())
        caller = scope["CurrentRuntime"]()
        caller.model = SimpleNamespace(layers=[SimpleNamespace(is_linear=True),
                                               SimpleNamespace(is_linear=False), SimpleNamespace(is_linear=False)])
        prepared, warmed = [], []
        block = SimpleNamespace(indexer=SimpleNamespace(top_blocks=2), scale=.125)
        def entry(index):
            prepared.append(index)
            return {"attn": (None, None, None, None, "current pool norm", block)}
        caller.fused = SimpleNamespace(_entry=entry, eps="current eps")
        caller.args = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, head_dim=128,
                                      indexer_n_heads=2, indexer_head_dim=128, rotary_dim=32, rope_theta=10000)
        attention.warm_decode = lambda **kwargs: warmed.append(kwargs)
        caller._warm_sparse()
        self.assertEqual(prepared, [1])
        self.assertEqual(len(warmed), 1)
        self.assertEqual(warmed[0]["norm"], "current pool norm")

    def test_draft_head_plan_is_call_local_reused_then_refreshed_next_call(self):
        scope, _ = runtime_api(api())
        caller = scope["CurrentRuntime"]()
        caller.model = SimpleNamespace(lm_head=Linear("head", shape=(4, 8)))
        caller._draft_cut_ids, caller._draft_ids = IDs([0, 2, 3]), "owned IDs"
        caller.drafts = 3
        caller._absorb = lambda *args: ("mixed", "streams")
        caller._mtp_step = lambda *args: ("mixed", "streams")
        plans = []
        def draw(mixed, sampling, positions, *, prepared=None):
            plans.append(prepared)
            return SimpleNamespace(item=lambda: 2)
        caller._draft_draw = draw
        cache = [SimpleNamespace(drafted=0)]
        self.assertEqual(caller.draft(cache, "streams", [1], 0, None), [2, 2, 2])
        first = plans[0][0]
        self.assertTrue(all(plan[0] is first for plan in plans))
        caller.model.lm_head.weight.value = "new current head"
        plans.clear()
        caller.draft(cache, "streams", [1], 0, None)
        self.assertIsNot(first, plans[0][0])
        self.assertNotEqual(first.weight.value, plans[0][0].weight.value)
        self.assertNotIn("prepared", vars(caller))
        tree = ast.parse((ROOT / "src/tensorfold/families/qwen4_exp/runtime.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlashNext")
        for name in ("draft", "settle", "draft_streams"):
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            gets = [n for n in ast.walk(method) if isinstance(n, ast.Attribute) and n.attr == "_draft_head"]
            self.assertEqual(len(gets), 1)
            calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "_draft_draw"]
            self.assertTrue(calls)
            self.assertTrue(all(any(k.arg == "prepared" for k in call.keywords) for call in calls))

    def test_operation_entries_are_shared_without_retained_derived_global_cache(self):
        tree = ast.parse(DECODE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecode")
        for name in ("run", "run_multi"):
            node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            entry_calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Attribute) and n.func.attr == "_entry"]
            self.assertEqual(len(entry_calls), 1)
            moe_call = next(n for n in ast.walk(node) if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute) and n.func.attr == "_moe")
            self.assertIsInstance(moe_call.args[-1], ast.Name)
            self.assertEqual(moe_call.args[-1].id, "entry")
        self.assertNotIn("_lane", {n.targets[0].id for n in tree.body if isinstance(n, ast.Assign)
                                  and isinstance(n.targets[0], ast.Name)})


if __name__ == "__main__":
    unittest.main()
