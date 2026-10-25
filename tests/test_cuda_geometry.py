"""Compare startup estimates with actual cache constructors using allocation-only tensors."""

import importlib
import math
import sys
from types import SimpleNamespace, ModuleType

import pytest

from tensorfold.cuda import geometry, capacity


class Allocation:
    def __init__(self, shape, dtype, device):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
    def numel(self):
        return math.prod(self.shape)
    def element_size(self):
        return getattr(self.dtype, "itemsize", None) or {"bf16": 2, "fp16": 2, "int16": 2, "int8": 1, "uint8": 1}.get(
            self.dtype, 4)
    def __getitem__(self, index):
        return self
    def __add__(self, other):            # arange(rows)[:, None] + arange(k): the [rows, k] tap table
        return Allocation(self.shape + other.shape, self.dtype, self.device)
    def contiguous(self):
        return self
    def view(self, *shape):
        assert math.prod(shape) == self.numel()
        return Allocation(shape, self.dtype, self.device)


@pytest.fixture
def allocations(monkeypatch):
    before = set(sys.modules)
    lang = ModuleType("triton.language")
    class Constexpr:
        def __init__(self, value):
            self.value = value

        def __mul__(self, other):
            return Constexpr(self.value * (other.value if isinstance(other, Constexpr) else other))

        def __rmul__(self, other):
            return Constexpr((other.value if isinstance(other, Constexpr) else other) * self.value)
    lang.constexpr = Constexpr
    triton = ModuleType("triton")
    triton.language = lang
    triton.jit = lambda fn=None, **kwargs: fn if fn is not None else (lambda function: function)
    triton.cdiv = lambda a, b: (a + b - 1) // b
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", lang)
    recorded = []
    def allocate(shape, **kw):
        tensor = Allocation(shape, kw.get("dtype", "fp32"), kw.get("device", "cpu"))
        recorded.append(tensor)
        return tensor
    fake = SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32", int8="int8", uint8="uint8", int16="int16",
                           int32="int32", int64="int64",
                           zeros=allocate, empty=allocate, full=lambda shape, fill, **kw: allocate(shape, **kw),
                           tensor=lambda values, **kw: allocate((len(values),), **kw),
                           zeros_like=lambda x: allocate(x.shape, dtype=x.dtype, device=x.device),
                           arange=lambda *a, **kw: allocate((len(range(*a)),), **kw),
                           cuda=SimpleNamespace(is_available=lambda: False,
                                                get_device_properties=lambda device: SimpleNamespace(
                                                    multi_processor_count=48)))
    # Imports use real torch annotations; only the allocation sites are replaced.
    try:
        yield recorded, fake
    finally:
        prefix = "tensorfold.families."
        added = [(name, module) for name, module in list(sys.modules.items())
                 if name not in before and name.startswith(prefix) and ".cuda" in name]
        for name, module in sorted(added, key=lambda item: len(item[0]), reverse=True):
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, child, None) is module:
                delattr(parent, child)
            sys.modules.pop(name, None)


def bytes_in(arrays):
    return sum(t.numel() * t.element_size() for t in arrays)


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("mtp", [False, True])
def test_indexed_stream_status_and_independent_kv_sides_remain_budgeted_per_rank(world, mtp):
    text = {"hidden_size": 512, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64,
            "num_hidden_layers": 4, "layer_types": ["linear_attention", "full_attention"] * 2,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 1024, "hc_count": 4}
    options = dict(streams=3, each=4, keep=8, mtp=mtp, world=world)
    bf16 = geometry.indexed_stream_geometry(text, **options)
    pair = ((16, 32, 2), (8, 32, 2))
    mixed = geometry.indexed_stream_geometry(text, **options, kv_pair=pair)
    checked = geometry.indexed_stream_geometry(text, **options, kv_pair=pair, kv_status=True)
    for slots in (256, 4096, 65536):
        rows = slots + 2 * 256  # the growing slot and the other slots' first allocation
        heads = 2 // world
        # One BF16 side replaces its 128-byte row and 2-byte dummy scale
        # with a 68-byte INT8 row, independently of K's unchanged BF16 side.
        difference = (2 + int(mtp)) * (rows * heads * (128 - 68) + 3 * 2)
        assert bf16.bytes_at(slots) - mixed.bytes_at(slots) == difference
        assert checked.bytes_at(slots) - mixed.bytes_at(slots) == 8 * 3 * (1 + (world if world > 1 else 0))


MULTI_TEXT = {"hidden_size": 512, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64,
              "num_hidden_layers": 6, "layer_types": ["linear_attention"] * 2 + ["full_attention"] * 2
              + ["linear_attention", "full_attention"], "linear_num_key_heads": 2, "linear_num_value_heads": 4,
              "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
              "vocab_size": 1024, "hc_count": 4}


@pytest.mark.parametrize("value,mib", [(None, 0), ("", 0), (" 64 ", 64), ("0", 0), ("0064", 64),
                                       ("65536", 65536)])
def test_multi_graph_budget_reads_whole_mebibytes(monkeypatch, value, mib):
    if value is None:
        monkeypatch.delenv(geometry.MULTI_GRAPH_ENV, raising=False)
    else:
        monkeypatch.setenv(geometry.MULTI_GRAPH_ENV, value)
    assert geometry.multi_graph_bytes() == mib << 20


@pytest.mark.parametrize("value", ["65537", "-1", "1.5", "0x10", "1e3", "+5", "512MiB", "abc", "\u0661\u0662",
                                   pytest.param("1" * 5000, id="5000-digits")])
def test_multi_graph_budget_refuses_anything_else_by_name(monkeypatch, value):
    monkeypatch.setenv(geometry.MULTI_GRAPH_ENV, value)
    with pytest.raises(ValueError, match=geometry.MULTI_GRAPH_ENV):
        geometry.multi_graph_bytes()


@pytest.mark.parametrize("streams,each,mtp", [(2, 1, False), (3, 4, True), (4, 5, True)])
def test_indexed_stream_geometry_plans_the_static_tables_and_the_graph_budget(monkeypatch, streams, each, mtp):
    """The concurrent decoder's plan includes its static round tables (``multi_static_bytes``) and the captured
    rounds' budget, each byte for byte, at every window."""

    plan = geometry.indexed_stream_geometry(MULTI_TEXT, streams, each, 8, mtp=mtp)
    graphs = geometry.indexed_stream_geometry(MULTI_TEXT, streams, each, 8, mtp=mtp, graph_bytes=512 << 20)
    tables = geometry.multi_static_bytes(MULTI_TEXT, streams, each, mtp)
    monkeypatch.setattr(geometry, "multi_static_bytes", lambda *args: 0)
    bare = geometry.indexed_stream_geometry(MULTI_TEXT, streams, each, 8, mtp=mtp)
    for slots in (256, 4096, 65536):
        assert graphs.bytes_at(slots) - plan.bytes_at(slots) == 512 << 20
        assert plan.bytes_at(slots) - bare.bytes_at(slots) == tables > 0


@pytest.mark.torch
@pytest.mark.parametrize("streams,each,mtp", [(2, 1, False), (3, 4, True), (4, 5, True)])
def test_static_round_table_bytes_are_the_tables_allocations(streams, each, mtp):
    """``multi_static_bytes`` is what the decoder's static tables allocate (device arrays and their host mirrors:
    ``gdn_multi.RoundTables``, ``attn_multi.StepTables`` for the verify and the MTP steps) plus its codec status
    words (``MultiDecoder.kv_host``: two int32 a slot)."""

    import torch

    from tensorfold.families.qwen4_exp.cuda import attn_multi, gdn_multi

    layers = [SimpleNamespace(linear=kind == "linear_attention") for kind in MULTI_TEXT["layer_types"]]
    w = SimpleNamespace(device=torch.device("cpu"), layers=layers)
    tables = [gdn_multi.RoundTables(w, SimpleNamespace(lin=sum(layer.linear for layer in layers)), streams, each),
              attn_multi.StepTables(w, streams, each, mtp=False)]
    if mtp:
        tables.append(attn_multi.StepTables(w, streams, each, mtp=True))
    arrays = [t for table in tables for t in (table.i32, table.h32, table.i64, table.h64)]
    arrays.append(torch.zeros((streams, 2), dtype=torch.int32))
    assert geometry.multi_static_bytes(MULTI_TEXT, streams, each, mtp) == bytes_in(arrays)


@pytest.mark.torch
@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("prefill_rows", [2048, 4096])
@pytest.mark.parametrize("kv_dtype,bits", [("bf16", 16), ("int8", 8), ("int4", 4)])
def test_indexed_state_actual_kv_and_serial_twin_are_budgeted(monkeypatch, allocations, world, mtp, kv_dtype, bits,
                                                            prefill_rows):
    arrays, fake = allocations
    mod = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    gdn = importlib.import_module("tensorfold.families.qwen4_exp.cuda.gdn")
    monkeypatch.setattr(mod, "torch", fake)
    monkeypatch.setattr(gdn, "torch", fake)
    monkeypatch.setattr(mod.kvcache, "torch", fake)
    text = {"hidden_size": 512, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64,
            "num_hidden_layers": 4, "layer_types": ["linear_attention", "full_attention"] * 2,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 1024, "hc_count": 4}
    cfg = SimpleNamespace(hidden=512, streams=4, conv_kernel=4, conv_dim=1024 // world,
                          nk=2 // world, nv=4 // world, dk=128, dv=128, kv_heads=2 // world,
                          head_dim=64, index_dim=128, index_ratio=4, ple_kernel=4,
                          ngram_size=3, ple_layers=[], heads=8 // world, index_heads=4,
                          index_budget=2048, low=320, experts=8, top_k=2, moe_width=512 // world,
                          shared_width=512 // world, heads_per_ngram=8, ple_dim=512)
    weights = SimpleNamespace(cfg=cfg, device="cpu", layers=[SimpleNamespace(index=i, linear=i % 2 == 0) for i in range(4)],
                              mtp=SimpleNamespace() if mtp else None, meta={"world": world}, head=SimpleNamespace(n=1024 // world))
    slots = 65536
    for child in (mod.attn_mod, mod.moe_mod):
        monkeypatch.setattr(child, "torch", fake)
    mod.Buffers(weights, 64, slots)
    if mtp:
        mod.Buffers(weights, 64, slots)
    mod.Buffers(weights, prefill_rows, slots, prefill=True)
    mod.State(weights, slots, 64, kv_dtype)
    mod.State(weights, slots, 64, kv_dtype)  # the actual serial-reference twin constructor
    estimated = geometry.gdn_geometry(text, world, 7, indexed=True, mtp=mtp, kv_bits=bits,
                                      prefill_rows=prefill_rows).bytes_at(slots)
    kv = [t for t in arrays if len(t.shape) == 3 and t.shape[:2] == (slots, cfg.kv_heads)]
    caches = 2 * (2 + int(mtp))                                          # two states: two attention layers, the MTP's
    assert bytes_in(kv) == caches * 2 * slots * cfg.kv_heads * geometry.kv_bytes(cfg.head_dim, bits)
    assert bytes_in(arrays) <= estimated


@pytest.mark.torch
@pytest.mark.parametrize("capacity", [4096, 65536, 262151, 1048584])
@pytest.mark.parametrize("rows", [1, 20, 31, 32, 256])
def test_qsa_select_scratch_is_the_planned_bytes(monkeypatch, allocations, capacity, rows):
    """Flash Next's multi-CTA select scratch (histograms and placement counts) is ``qsa_select_bytes`` exactly."""

    _, fake = allocations
    mod = importlib.import_module("tensorfold.families.qwen4_exp.cuda.attention")
    monkeypatch.setattr(mod, "torch", fake)
    scratch = mod.AttnScratch(rows, 8, 64, capacity, "cpu")
    assert geometry.QSA_SELECT_ROWS == mod.MULTI_ROWS - 1
    assert (geometry.QSA_SELECT_TILE, geometry.QSA_SELECT_SLICES) == (mod.SELECT_TB, mod.SELECT_SLICES)
    assert bytes_in([scratch.hist, scratch.suf]) == geometry.qsa_select_bytes(rows, scratch.nb)


@pytest.mark.torch
def test_centroid_tables_are_the_planned_bytes(monkeypatch, allocations):
    """Each Flash Next attention scratch's centroid tables (``AttnScratch.centroids``: 64 six-bit then 256 eight-bit
    float32 centroids) and eight-bit thresholds (``AttnScratch.thresholds``: 255 float32) take
    ``CENTROID_TABLE_BYTES`` and ``THRESHOLD_TABLE_BYTES``: their bytes in the caching allocator's 512-byte blocks."""

    _, fake = allocations
    mod = importlib.import_module("tensorfold.families.qwen4_exp.cuda.attention")
    monkeypatch.setattr(mod, "torch", fake)
    scratch = mod.AttnScratch(1, 8, 64, 4096, "cpu")
    assert scratch.centroids.shape == (64 + 256,) and scratch.centroids.dtype == fake.float32
    assert scratch.thresholds.shape == (255,) and scratch.thresholds.dtype == fake.float32
    assert geometry.CENTROID_TABLE_BYTES == -(-bytes_in([scratch.centroids]) // 512) * 512
    assert geometry.THRESHOLD_TABLE_BYTES == -(-bytes_in([scratch.thresholds]) // 512) * 512


def test_mla_latent_estimate_grows_by_the_cache_and_counts_one_prompt_chunk_scratch(monkeypatch):
    """Per token, the latent estimate grows by the latent and indexer caches of every attention layer (the MTP's
    too) plus one fp32 pool score for each of the (at most 512) prompt-chunk rows scored at once; the MTP head adds its
    caches and decode buffers, never a second set of the prompt chunk's latent partials (it absorbs through the same
    prefill buffers)."""

    text = {"hidden_size": 512, "num_attention_heads": 8, "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2, "linear_num_heads": 8,
            "qk_nope_head_dim": 256, "v_head_dim": 256, "vocab_size": 1024, "kv_lora_rank": 512,
            "moe_intermediate_size": 512, "num_experts_per_tok": 2}
    rows, heads, lw, index = geometry.PREFILL_ROWS, 4, 512, 128
    a, b = 1 << 18, (1 << 18) + 4096
    const = {}
    for block in (256, 512):                     # rows of the dense pass's partials
        monkeypatch.setattr(geometry, "MLA_PROMPT_ATT_ROWS", block)
        for mtp in (0, 1):
            g = geometry.mla_geometry({**text, "num_nextn_predict_layers": mtp}, 2, 8, latent=True)
            count = 2 + mtp
            slope = count * lw * 2 + count * index * 2 * 9 // 4 + min(rows, 512)
            assert g.bytes_at(b) - g.bytes_at(a) == (b - a) * slope
            const[block, mtp] = g.bytes_at(a) - a * slope
    # one set of partials, with or without the MTP head: 256 more rows of them cost the same in both
    more = ((2560 + rows + 511) // 512) * 256 * heads * (lw + 2) * 4
    assert const[512, 0] - const[256, 0] == const[512, 1] - const[256, 1] == more
    assert const[512, 1] > const[512, 0]


@pytest.mark.torch
@pytest.mark.parametrize("latent", [True, False], ids=["latent", "per-head"])
@pytest.mark.parametrize("mtp", [False, True])
def test_mla_actual_cache_and_replay_state_are_budgeted(monkeypatch, allocations, mtp, latent):
    arrays, fake = allocations
    mod = importlib.import_module("tensorfold.families.glm5_next.cuda.forward")
    kda = importlib.import_module("tensorfold.families.glm5_next.cuda.kda")
    cache = importlib.import_module("tensorfold.families.glm5_next.cuda.latent")
    monkeypatch.setattr(mod, "torch", fake)
    monkeypatch.setattr(kda, "torch", fake)
    monkeypatch.setattr(cache, "torch", fake)
    monkeypatch.setattr(cache, "ENABLED", latent)
    text = {"hidden_size": 512, "num_attention_heads": 8, "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2, "linear_num_heads": 8,
            "qk_nope_head_dim": 256, "v_head_dim": 256, "vocab_size": 1024,
            "moe_intermediate_size": 512, "num_experts_per_tok": 2,
            "num_nextn_predict_layers": int(mtp)}
    cfg = SimpleNamespace(heads=8, lin_heads=8, conv=4, qk_dim=256, v_dim=256, index_dim=128,
                          hidden=512, streams=4, q_lora=512, kv_lora=512, index_heads=32,
                          dense_width=1024, top_k=2, moe_width=512, experts=8, quant="mlx")
    layers = [SimpleNamespace(index=i, kind="kda" if i % 2 == 0 else "dsa",
                              kda=SimpleNamespace(proj=SimpleNamespace(n=3 * 4 * 128 + 256 + 4))) for i in range(4)]
    weights = SimpleNamespace(cfg=cfg, world=2, device="cpu", layers=layers, meta={"long_context": True},
                              mtp=SimpleNamespace() if mtp else None, head=SimpleNamespace(n=512))
    slots = 65536
    attention = importlib.import_module("tensorfold.families.glm5_next.cuda.attention")
    monkeypatch.setattr(attention, "torch", fake)
    mod.Buffers(weights, 64, slots)
    if mtp:
        mod.Buffers(weights, 64, slots)
    mod.Buffers(weights, 2048, slots, prefill=True)     # the prompt chunks' buffers, as ``decode.Engine`` makes them
    mod.State(weights, slots, 64)
    estimated = geometry.mla_geometry(text, 2, 8, latent=latent).bytes_at(slots)
    assert any(slots in t.shape for t in arrays)             # the constructors ran on the fake allocator
    assert bytes_in(arrays) <= estimated


@pytest.mark.torch
def test_mla_prompt_chunk_latent_partials_hold_one_row_block(monkeypatch, allocations):
    """A prompt chunk's dense partials and pool scores hold one row block each, as mla_geometry counts."""
    arrays, fake = allocations
    mod = importlib.import_module("tensorfold.families.glm5_next.cuda.forward")
    kda = importlib.import_module("tensorfold.families.glm5_next.cuda.kda")
    cache = importlib.import_module("tensorfold.families.glm5_next.cuda.latent")
    attention = importlib.import_module("tensorfold.families.glm5_next.cuda.attention")
    sparse = importlib.import_module("tensorfold.families.glm5_next.cuda.sparse")
    for m in (mod, kda, cache, attention):
        monkeypatch.setattr(m, "torch", fake)
    monkeypatch.setattr(cache, "ENABLED", True)
    cfg = SimpleNamespace(heads=64, lin_heads=8, conv=4, qk_dim=256, v_dim=256, index_dim=128, hidden=512, streams=4,
                          q_lora=512, kv_lora=512, index_heads=32, dense_width=1024, top_k=2, moe_width=512, experts=8,
                          quant="mlx")
    layers = [SimpleNamespace(index=i, kind="kda" if i % 2 == 0 else "dsa",
                              kda=SimpleNamespace(proj=SimpleNamespace(n=3 * 4 * 128 + 256 + 4))) for i in range(4)]
    weights = SimpleNamespace(cfg=cfg, world=2, device="cpu", layers=layers, meta={}, mtp=None,
                              head=SimpleNamespace(n=512))
    assert mod.PROMPT_ATT_ROWS == geometry.MLA_PROMPT_ATT_ROWS == 512
    assert sparse.SELECT_ROWS == geometry.MLA_SELECT_ROWS == 512
    for rows, prefill, part in ((2048, True, 512), (4096, True, 512), (256, True, 256), (16, False, 16)):
        s = mod.Buffers(weights, rows, 1 << 20, prefill=prefill).lat_s
        assert s.rows == rows and s.part_rows == part
        assert s.po.shape == (s.nch * part * 32 * 512,) and s.pm.shape == s.pl.shape == (s.nch * part * 32,)
        assert s.qa.shape == s.ol.shape == (rows, 32, 512)


def test_mla_chunk_scratch_counts_one_block_of_pool_scores():
    text = {"num_attention_heads": 64, "kv_lora_rank": 512, "index_topk": 2048, "qk_nope_head_dim": 256}
    for cap in (4096, 262152, 1 << 20):
        partials = 5 * 2048 * 32 * 514 * 4                            # sparse attention's 5 chunk partials a row
        assert geometry.mla_chunk_scratch(text, 2, cap, latent=True) == \
            512 * 4 * ((cap + 3) // 4) + 2048 * 16 * 2051 + partials


def test_weight_partition_rounding_and_float_casts():
    from tensorfold.families.glm5_next.cuda.split import rule
    transform = geometry.split_weights(rule)
    info = {"dtype": "U32", "shape": [192, 128], "split": False}
    assert transform("model.language_model.layers.0.mlp.experts.0.gate_proj.weight", info) == (128 * 128 * 4, 0)
    assert transform("model.language_model.layers.0.mlp.experts.0.down_proj.weight", info) == (192 * 64 * 4, 0)
    split = {**info, "shape": [96, 128], "split": True}
    assert transform("model.language_model.layers.0.mlp.experts.0.gate_proj.weight", split) == (128 * 128 * 4, 0)
    vector = {"dtype": "BF16", "shape": [64], "split": False}
    assert transform("model.language_model.layers.0.self_attn.A_log", vector) == (32 * 4, 0)
    assert transform("model.visual.weight", info) == (0, 0)


@pytest.mark.usefixtures("default_memory_policy")
def test_unified_available_memory_uses_reclaimable_host_pages(monkeypatch):
    from pathlib import Path
    fake = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (100 * capacity.GIB, 128 * capacity.GIB)))
    monkeypatch.setattr(Path, "read_text", lambda *a: "MemTotal: 134217728 kB\nMemAvailable: 62914560 kB\n")
    monkeypatch.setattr(capacity, "unified", lambda torch: True)
    # the grant is the reclaimable pool less the floor the host keeps free: a tenth of its 128 GiB of RAM
    assert capacity.available_bytes(fake) == 60 * capacity.GIB - 128 * capacity.GIB // 10


@pytest.mark.torch
@pytest.mark.parametrize("mtp", [False, True])
def test_mla_exl3_scratch_and_buffers_are_budgeted(monkeypatch, allocations, mtp):
    """GLM-5.3-Flash EXL3 at one rank's shapes: the scratch and buffers allocated stay within mla_geometry."""
    arrays, fake = allocations
    mod = importlib.import_module("tensorfold.families.glm5_next.cuda.forward")
    names = ("kda", "latent", "attention", "exl3_mm")
    mods = [mod] + [importlib.import_module(f"tensorfold.families.glm5_next.cuda.{n}") for n in names]
    for m in mods + [importlib.import_module("tensorfold.cuda.experts"),
                     importlib.import_module("tensorfold.cuda.exl3.experts")]:
        monkeypatch.setattr(m, "torch", fake)
    monkeypatch.setattr(mods[2], "ENABLED", True)
    exl3_mm = mods[-1]
    text = {"hidden_size": 4096, "num_attention_heads": 64, "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "full_attention"] * 2, "linear_attn_config": {"num_heads": 64},
            "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256, "vocab_size": 154880,
            "q_lora_rank": 1536, "kv_lora_rank": 512, "intermediate_size": 12288, "moe_intermediate_size": 2048,
            "num_experts_per_tok": 8, "n_routed_experts": 288, "index_n_heads": 32, "index_head_dim": 128,
            "num_nextn_predict_layers": int(mtp), "_quantization": {"quant_method": "exl3"}}
    cfg = SimpleNamespace(heads=64, lin_heads=64, conv=4, qk_dim=256, v_dim=256, index_dim=128, hidden=4096,
                          streams=4, q_lora=1536, kv_lora=512, index_heads=32, dense_width=12288, top_k=8,
                          moe_width=2048, shared_width=2048, experts=288, quant="exl3")
    layers = [SimpleNamespace(index=i, kind="kda" if i % 2 == 0 else "dsa",
                              kda=SimpleNamespace(proj=SimpleNamespace(n=3 * 32 * 128 + 256 + 32))) for i in range(4)]
    weights = SimpleNamespace(cfg=cfg, world=2, device="cpu", layers=layers, meta={"long_context": True},
                              mtp=SimpleNamespace() if mtp else None, head=SimpleNamespace(n=154880 // 2))
    slots, d, width = 9, 4096, 1024
    for rows in (8, 64, geometry.PREFILL_ROWS):
        arrays.clear()
        exl3_mm.Scratch(rows, slots, d, width, "cpu")
        assert bytes_in(arrays) == geometry.exl3_expert_scratch(rows, slots, d, width), rows
    arrays.clear()
    cap = 1 << 20
    mod.Buffers(weights, 64, cap)
    if mtp:
        mod.Buffers(weights, 64, cap)
    pbuf = mod.Buffers(weights, geometry.PREFILL_ROWS, cap, prefill=True)
    assert pbuf.sk.numel() == 8 * geometry.PREFILL_ROWS * 16384          # BF16 prompt projections' partials
    mod.State(weights, cap, 64)
    estimated = geometry.mla_geometry(text, 2, 16, latent=True).bytes_at(cap)
    assert bytes_in(arrays) <= estimated - geometry.mla_chunk_scratch(text, 2, cap, latent=True)


@pytest.mark.torch
@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("mtp", [False, True])
def test_flash_message_snapshot_budget_counts_actual_saved_tensors(world, mtp):
    import torch
    from tensorfold.families.qwen4_exp.cuda.state import State

    text = {"hidden_size": 512, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64,
            "layer_types": ["linear_attention", "full_attention"] * 2,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 1024, "hc_count": 4}
    cfg = SimpleNamespace(hidden=512, streams=4, conv_kernel=4, conv_dim=1024 // world,
                          nk=2 // world, nv=4 // world, dk=128, dv=128, kv_heads=2 // world,
                          head_dim=64, index_dim=128, index_ratio=4, ple_kernel=4, ngram_size=3, ple_layers=[])
    weights = SimpleNamespace(cfg=cfg, device="cpu", layers=[SimpleNamespace(index=i, linear=i % 2 == 0)
                                                           for i in range(4)], mtp=object() if mtp else None,
                              meta={"world": world})
    state = State(weights, 32, 8)
    snapshot = state.snapshot()
    tensors = [value for value in snapshot.values() if isinstance(value, torch.Tensor)]
    if mtp:
        tensors.append(torch.zeros((1, cfg.hidden * cfg.streams), dtype=torch.bfloat16))
    saved = sum(t.numel() * t.element_size() for t in tensors)
    low = geometry.gdn_geometry(text, world, 8, indexed=True, mtp=mtp, kept=2).bytes_at(32)
    high = geometry.gdn_geometry(text, world, 8, indexed=True, mtp=mtp, kept=5).bytes_at(32)
    assert high - low == 3 * saved
    if world == 1:
        low = geometry.indexed_stream_geometry(text, 2, 4, 2, mtp=mtp).bytes_at(32)
        high = geometry.indexed_stream_geometry(text, 2, 4, 8, mtp=mtp).bytes_at(32)
        assert high - low == 6 * saved
