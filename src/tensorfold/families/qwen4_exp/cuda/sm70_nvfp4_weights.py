"""Manifest-bound NVFP4 import into the shared Flash Next weight graph.

Planning is pure metadata work. Every retained tensor has one construction
plan, used by both admission and bounded import; the original Blackwell loader
is not modified or used as an implicit SM70 conversion route.
"""

from dataclasses import dataclass, fields, replace
import math
from pathlib import Path
from functools import partial

import numpy as np
import torch

from tensorfold.cuda.nvfp4.format import E2M1, calibration, e4m3, scheme
from tensorfold.cleanup import raise_failures

from ..host_table import BF16Table, PREFETCH_READ, _copy_rows, open_table, shard_keys
from ..read_ahead import ReadAhead, row_ids
from ..ssd_table import EncodedSSDTable
from ..ple_lifetime import PLETables, _retain_failure
from . import sm70_nvfp4 as runtime
from .gguf_weights import GGUFConcat, STAGING_BYTES, _ranges
from .reader import _Reader
from .sm70_weight_manifest import load as load_manifest
from .weight_types import AttnW, Config, GDNW, HC, LayerW, MoEW, PLEW, Weights
from tensorfold.cuda.sm70.nvfp4_layout import EXPERT_DEVICE, EXPERT_WORDS


@dataclass(frozen=True)
class Matrix:
    name: str
    kind: str
    shape: tuple
    rows: tuple
    column: int
    k: int
    factor_index: int = -1

    @property
    def n(self):
        return sum(count for _, count in self.rows)

    @property
    def resident_bytes(self):
        return self.n * self.k * 2 if self.kind == "bf16" else self.n * (self.k // 2 + self.k // 16) + 4


@dataclass(frozen=True)
class Array:
    name: str
    shape: tuple
    indices: tuple | None = None
    dtype: str = "float32"
    transform: str = "identity"

    @property
    def resident_bytes(self):
        return math.prod(self.shape) * (4 if self.dtype == "float32" else 2)


@dataclass(frozen=True)
class Concat:
    parts: tuple


@dataclass(frozen=True)
class Experts:
    parts: tuple
    width: int
    dims: int


@dataclass(frozen=True)
class Table:
    shards: tuple
    base: str
    rows: int
    dims: int
    kind: str
    scale_name: str | None


class Planner:
    def __init__(self, manifest, directory):
        self.manifest = manifest
        self.directory = Path(directory).resolve()
        self.infos = manifest.infos
        self.used = set()
        self.matrices, self.arrays, self.tables = [], [], []
        self.factor_count = 0

    def matrix(self, base, shape, *, rows=None, columns=None):
        selected = {field: (self.infos[base + "." + field]["dtype"], self.infos[base + "." + field]["shape"])
                    for field in ("weight", "weight_packed", "weight_scale", "weight_scale_2", "input_scale")
                    if base + "." + field in self.infos}
        kind = scheme(selected)
        weight = selected.get("weight", selected.get("weight_packed"))
        actual = (weight[1][0], weight[1][1] * (2 if kind == "nvfp4" else 1))
        if kind not in ("nvfp4", "bf16") or tuple(shape) != actual:
            raise ValueError(f"{base}: unsupported encoding or shape; expected {shape}, got {kind}/{actual}")
        if "weight_packed" in selected:
            raise ValueError(f"{base}: this ModelOpt direct-scale recipe requires weight; CT weight_packed is distinct")
        for field in selected:
            self.used.add(base + "." + field)
        n, k = shape
        row_ranges = ((0, n),) if rows is None else _ranges(rows)
        column, count = (0, k) if columns is None else columns
        if (any(lo < 0 or size <= 0 or lo > n - size for lo, size in row_ranges)
                or not row_ranges or column < 0 or count <= 0 or column > k - count
                or kind == "nvfp4" and (column % 16 or count % 16)):
            raise ValueError(f"{base}: TP shard cannot preserve the selected NVFP4 group16 layout")
        factor_index = self.factor_count if kind == "nvfp4" else -1
        self.factor_count += kind == "nvfp4"
        result = Matrix(base, kind, tuple(shape), row_ranges, column, count, factor_index)
        self.matrices.append(result)
        return result

    def array(self, name, shape, *, indices=None, dtype="float32", transform="identity"):
        info = self.infos[name]
        if info["dtype"] not in ("F16", "BF16", "F32"):
            raise ValueError(f"{name}: expected an original floating parameter, not unscaled packed bytes")
        actual = tuple(info["shape"])
        if indices is not None:
            indices = tuple(indices)
            if not actual or any(type(i) is not int or not 0 <= i < actual[0] for i in indices):
                raise ValueError(f"{name}: invalid parameter TP slice")
            actual = (len(indices), *actual[1:])
        if math.prod(shape) != math.prod(actual):
            raise ValueError(f"{name}: parameter shape differs from admitted family geometry")
        item_size = {"F16": 2, "BF16": 2, "F32": 4}[info["dtype"]]
        if (math.prod(info["shape"]) * item_size > STAGING_BYTES
                or math.prod(shape) * (4 if dtype == "float32" else 2) > STAGING_BYTES):
            raise ValueError(f"{name}: parameter exceeds the bounded source/runtime array import span")
        self.used.add(name)
        result = Array(name, tuple(shape), indices, dtype, transform)
        self.arrays.append(result)
        return result


def model_plan(directory, manifest_path, tp=None, *, rope=None, eos_token_ids=None):
    manifest = load_manifest(manifest_path, directory)
    result = _model_plan(directory, manifest, tp, rope=rope, eos_token_ids=eos_token_ids)
    manifest.check_versions(directory)
    return result


def check_mtp_positions(target, directory, *, context):
    """This recipe implements matching native positions, never implicit YaRN propagation."""
    from tensorfold.checkpoint_metadata import read_config
    from ..rope import RopeParameters

    supplied = RopeParameters.from_config(read_config(directory))
    selected = target.cfg.rope
    if (selected.rope_type != "default" or supplied.rope_type != "default"
            or selected.theta != supplied.theta or selected.rotary_dim != supplied.rotary_dim
            or selected.mrope_section != supplied.mrope_section
            or selected.attention_factor != supplied.attention_factor
            or type(context) is not int or not 0 < context <= min(selected.context_limit, supplied.context_limit)):
        raise ValueError("SM70 NVFP4 MTP position pairing is unsupported; require matching native transforms/bounds or an implemented separately qualified extension")


def _model_plan(directory, manifest, tp=None, *, rope=None, eos_token_ids=None):
    """Shared post-validation construction, also used by the manifest generator."""
    full = replace(Config.read(directory, rope=rope, eos_token_ids=eos_token_ids),
                   execution_dtype="float16", quant="sm70_nvfp4")
    rank, world = (0, 1) if tp is None else tp
    if (type(world) is not int or world not in (1, 2, 4, 8) or type(rank) is not int or rank not in range(world)
            or any(value <= 0 or value % world for value in
                   (full.heads, full.nk, full.nv, full.moe_width, full.shared_width, full.vocab))
            or full.kv_heads <= 0 or full.heads % full.kv_heads or full.nv % full.nk
            or not 0 < full.top_k <= full.experts or len(full.layer_types) != full.layers
            or full.streams <= 0 or full.low <= 0 or full.index_heads <= 0 or full.index_dim <= 0
            or full.index_ratio <= 0 or full.index_budget <= 0 or full.ngram_size < 2
            or full.heads_per_ngram <= 0 or full.ngram_divisor <= 0 or full.ngram_shards <= 0
            or full.ple_dim <= 0 or full.ple_dim % ((full.ngram_size - 1) * full.heads_per_ngram)
            or any(index < 0 or index >= full.layers for index in full.ple_layers)):
        raise ValueError("SM70 NVFP4 family geometry cannot legally partition the selected TP topology")
    if full.kv_heads >= world:
        if full.kv_heads % world:
            raise ValueError("SM70 NVFP4 KV heads must partition into complete TP groups")
        kv, first = full.kv_heads // world, rank * (full.kv_heads // world)
    else:
        if world % full.kv_heads:
            raise ValueError("SM70 NVFP4 KV heads must replicate in complete TP groups")
        kv, first = 1, rank // (world // full.kv_heads)
    cfg = replace(full, heads=full.heads // world, kv_heads=kv, nk=full.nk // world, nv=full.nv // world,
                  moe_width=full.moe_width // world, shared_width=full.shared_width // world)
    if (full.hidden % 256 or full.low % 32 or cfg.heads * full.head_dim % 32
            or full.index_dim < full.rotary_dim):
        raise ValueError("SM70 family requires complete HC256/activation32 groups and supported indexer rotary width")
    if "linear" in full.layer_types and (full.dk != 128 or full.dv != 128 or full.conv_kernel != 4
            or full.nv != 3 * full.nk or cfg.nv not in (6, 12, 24, 48)):
        raise ValueError("SM70 native GDN requires128 dimensions,4conv taps and local2/6,4/12,8/24 or16/48 heads")
    planner = Planner(manifest, directory)
    d, wide = full.hidden, full.hidden * full.streams
    roots = [base for base in ("model.", "model.language_model.", "language_model.model.")
             if base + "embed_tokens.weight" in planner.infos]
    if len(roots) != 1:
        raise ValueError("SM70 NVFP4 needs one unambiguous implemented family tensor namespace")
    root = roots[0]

    def norm(name, width):
        return planner.array(name, (width,), transform="norm_offset")

    def hc(base, inject=True):
        down = planner.matrix(base + ".input_mix_weight_down", (full.low, wide))
        if inject:
            down = Concat((down, planner.matrix(base + ".block_inject_weight", (full.streams, wide))))
        return HC(down, planner.matrix(base + ".input_mix_weight_up", (wide, full.low)),
                  norm(base + ".hc_norm.weight", wide), inject)

    def gdn(base):
        q = list(range(rank * cfg.nk * full.dk, (rank + 1) * cfg.nk * full.dk))
        v = list(range(rank * cfg.nv * full.dv, (rank + 1) * cfg.nv * full.dv))
        channels = q + [full.nk * full.dk + i for i in q] + [2 * full.nk * full.dk + i for i in v]
        heads = range(rank * cfg.nv, (rank + 1) * cfg.nv)
        projected = Concat((planner.matrix(base + ".in_proj_qkv", (full.conv_dim, d), rows=channels),
                            planner.matrix(base + ".in_proj_z", (full.nv * full.dv, d), rows=v),
                            planner.matrix(base + ".in_proj_b", (full.nv, d), rows=heads),
                            planner.matrix(base + ".in_proj_a", (full.nv, d), rows=heads)))
        return GDNW(projected, planner.array(base + ".conv1d.weight", (cfg.conv_dim, cfg.conv_kernel),
                    indices=channels, dtype="float16"),
                    planner.array(base + ".A_log", (cfg.nv,), indices=heads),
                    planner.array(base + ".dt_bias", (cfg.nv,), indices=heads),
                    planner.array(base + ".norm.weight", (full.dv,), dtype="float16"),
                    planner.matrix(base + ".out_proj", (d, full.nv * full.dv),
                                   columns=(rank * cfg.nv * full.dv, cfg.nv * full.dv)))

    def attention(base):
        hd = full.head_dim
        q = range(rank * cfg.heads * 2 * hd, (rank + 1) * cfg.heads * 2 * hd)
        kr = range(first * hd, (first + kv) * hd)
        projection = Concat((planner.matrix(base + ".q_proj", (2 * full.heads * hd, d), rows=q),
                            planner.matrix(base + ".k_proj", (full.kv_heads * hd, d), rows=kr),
                            planner.matrix(base + ".v_proj", (full.kv_heads * hd, d), rows=kr),
                            planner.matrix(base + ".indexer.index_qk_proj", ((full.index_heads + 1) * full.index_dim, d))))
        return AttnW(projection, norm(base + ".q_norm.weight", hd), norm(base + ".k_norm.weight", hd),
                     norm(base + ".indexer.q_layernorm.weight", full.index_dim),
                     norm(base + ".indexer.k_layernorm.weight", full.index_dim),
                     planner.matrix(base + ".o_proj", (d, full.heads * hd),
                                    columns=(rank * cfg.heads * hd, cfg.heads * hd)))

    def moe(base):
        runs = []
        for expert in range(full.experts + 1):
            shared = expert == full.experts
            width = full.shared_width if shared else full.moe_width
            local = cfg.shared_width if shared else cfg.moe_width
            prefix = base + (".shared_expert" if shared else f".experts.{expert}")
            rows = range(rank * local, (rank + 1) * local)
            runs.append((planner.matrix(prefix + ".gate_proj", (width, d), rows=rows),
                         planner.matrix(prefix + ".up_proj", (width, d), rows=rows),
                         planner.matrix(prefix + ".down_proj", (d, width), columns=(rank * local, local))))
        router = Concat((planner.array(base + ".gate.weight", (full.experts, d), transform="half_to_float"),
                         planner.array(base + ".shared_expert_gate.weight", (1, d), transform="half_to_float")))
        return MoEW(router, Experts(tuple(runs), max(cfg.moe_width, cfg.shared_width), d))

    layers = []
    for index, kind in enumerate(full.layer_types):
        base = f"{root}layers.{index}"
        linear = kind == "linear"
        layer = LayerW(index, linear, hc(base + ".attn_hyper_connection"), hc(base + ".mlp_hyper_connection"),
                       gdn(base + ".linear_attn") if linear else None,
                       None if linear else attention(base + ".self_attn"), moe(base + ".mlp"))
        if index in full.ple_layers:
            p = base + ".ple"
            embed = p + ".ple_embedding."
            ngram = cfg.ngram(full.ple_layers.index(index))
            names = shard_keys(embed + "ngram_embedding", cfg.ngram_shards, planner.infos)
            if any(name + ".weight" not in planner.infos for name in names):
                raise ValueError("SM70 NVFP4 PLE manifest is missing a declared hash-table shard")
            table = Table(tuple((manifest.where[name + ".weight"], name) for name in names),
                          embed, ngram.rows, ngram.dims, "", None)
            kinds, total = set(), 0
            for name in names:
                info = planner.infos[name + ".weight"]
                if len(info["shape"]) != 2:
                    raise ValueError("SM70 PLE shards must be original row-major matrices")
                rows, columns = info["shape"]
                kind = {"BF16": "bf16", "F8_E4M3": "fp8", "U8": "nvfp4"}.get(info["dtype"])
                if kind is None or columns * (2 if kind == "nvfp4" else 1) != ngram.dims:
                    raise ValueError(f"{name}: unsupported PLE encoding/width for the declared host row reader")
                if kind == "nvfp4":
                    block = planner.infos.get(name + ".weight_scale")
                    if (ngram.dims % 16 or block is None or block["dtype"] != "F8_E4M3"
                            or block["shape"] != [rows, ngram.dims // 16]):
                        raise ValueError(f"{name}: PLE NVFP4 requires original E4M3/group16 block scales")
                if any(name + "." + field in planner.infos for field in ("weight_scale_2", "input_scale")):
                    raise ValueError(f"{name}: per-shard PLE calibration needs a distinct implemented source recipe")
                kinds.add(kind)
                total += rows
            if len(kinds) != 1 or total != ngram.rows:
                raise ValueError("SM70 PLE shards mix encodings or differ from the complete hash space")
            kind = kinds.pop()
            field = "weight_scale_2" if kind == "nvfp4" else "weight_scale"
            scale_name = embed + "ngram_embedding." + field
            if kind != "bf16" or scale_name in planner.infos:
                info = planner.infos.get(scale_name)
                if (info is None or info["dtype"] not in ("F16", "BF16", "F32")
                        or math.prod(info["shape"]) != 1):
                    raise ValueError(f"{scale_name}: PLE source requires one actual floating global calibration")
            else:
                scale_name = None
            table = replace(table, kind=kind, scale_name=scale_name)
            required = {embed + field for field in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")}
            required.update(name + ".weight" for name in names)
            if kind == "nvfp4":
                required.update(name + ".weight_scale" for name in names)
            if scale_name is not None:
                required.add(scale_name)
            if any(name not in planner.infos for name in required):
                raise ValueError("SM70 NVFP4 PLE source lacks its required hash constants")
            for field, count in (("layer_multipliers", ngram.n), ("ngram_heads_offsets", ngram.heads),
                                 ("ngram_heads_vocab_sizes", ngram.heads)):
                info = planner.infos[embed + field]
                if info["dtype"] != "I64" or math.prod(info["shape"]) != count:
                    raise ValueError(f"{embed + field}: PLE hash constants require the exact signed64-bit layout")
            from safetensors import safe_open
            from tensorfold.cuda.tensor_file import checkpoint_path

            constants = []
            for field in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
                name = embed + field
                with safe_open(str(checkpoint_path(directory, manifest.where[name])), framework="pt", device="cpu") as source:
                    constants.append(source.get_tensor(name).numpy().copy())
            ngram.check(*constants)
            planner.used.update(required)
            planner.tables.append(table)
            layer.ple = PLEW(table, planner.matrix(p + ".key_proj", (wide, full.ple_dim)),
                             planner.matrix(p + ".value_proj", (d, full.ple_dim)),
                             norm(p + ".norm_key.weight", wide), norm(p + ".norm_query.weight", wide),
                             norm(p + ".norm_conv.weight", wide),
                             planner.array(p + ".conv1d.weight", (wide, full.ple_kernel), dtype="float16"), ngram)
        layers.append(layer)
    embedding = planner.matrix(root + "embed_tokens", (full.vocab, d))
    if embedding.kind != "bf16":
        raise ValueError("this SM70 recipe requires an explicitly excluded floating embedding; packed gather needs a separate implemented operator")
    head_base = "language_model.lm_head" if "language_model.lm_head.weight" in planner.infos else "lm_head"
    head = planner.matrix(head_base, (full.vocab, d), rows=range(rank * (full.vocab // world),
                                                              (rank + 1) * (full.vocab // world)))
    prototype = Weights(cfg, (embedding,), layers, hc(root + "hyper_connection_mixer", False), head, None,
                        around_one=True)
    planner.used.update(manifest.document["inactive_source_tensors"])
    unused = set(planner.infos) - planner.used
    if unused:
        raise ValueError("unsupported/unmapped tensors in selected SM70 NVFP4 recipe: " + ", ".join(sorted(unused)))
    if not planner.factor_count:
        raise ValueError("the selected numerical manifest contains no retained NVFP4 target projection")
    prototype.meta.update(full=full, rank=rank, world=world, vocab_offset=rank * (full.vocab // world),
                          execution_provider="sm70", execution_dtype="float16", family_manifest=manifest.identity,
                          numerical_profile=runtime.NUMERICAL_PROFILE, source_activation_departure="unqualified",
                          weight_source="modelopt-nvfp4-row-major-direct-v1")
    return prototype, planner


def byte_plan(prototype, planner, *, prefill_rows=1024, execution_rows=32):
    """Construction-derived logical bytes, separate from allocator/library peaks."""
    width = max(prototype.cfg.moe_width, prototype.cfg.shared_width)
    kernel_family = planner.manifest.document["numerical"]["linear_kernel_family"]
    expert_family = planner.manifest.document["numerical"]["expert_kernel_family"]
    selection_rows = max(prefill_rows, execution_rows)
    workspace = runtime.workspace_bytes(width=width, dims=prototype.cfg.hidden, kernel_family=kernel_family,
        expert_family=expert_family, selection_rows=selection_rows, slots=prototype.cfg.top_k + 1,
        experts=prototype.cfg.experts + 1)
    workspace += max(prefill_rows, execution_rows) * prototype.cfg.hidden * 2  # Bounded embedding gather.
    plain_temporary = max((runtime.ROWS * (part.k * 2 + part.n * 4)
                           for part in planner.matrices if part.kind == "bf16"), default=0)
    workspace += plain_temporary
    host_workspace = runtime.host_workspace_bytes(selection_rows, prototype.cfg.top_k + 1, prototype.cfg.experts + 1,
                                                 expert_family=expert_family)
    table_names = {base + "." + field for table in planner.tables for _, base in table.shards
                   for field in (("weight", "weight_scale") if table.kind == "nvfp4" else ("weight",))}
    mapped = sum(math.prod(planner.infos[name]["shape"]) * {"U8": 1, "F8_E4M3": 1, "BF16": 2,
                 "F16": 2, "F32": 4, "I64": 8, "I32": 4}[planner.infos[name]["dtype"]]
                 for name in table_names if "ngram_embedding" in name)
    resident = sum(part.resident_bytes for part in (*planner.matrices, *planner.arrays))
    descriptor_bytes = (len(prototype.layers) * (prototype.cfg.experts + 1) * 3 * EXPERT_WORDS * 8
                        if expert_family == EXPERT_DEVICE else 0)
    resident += descriptor_bytes
    # Matrix row uploads never expand a whole expert bank. Only router arrays
    # are concatenated; their sources and joined allocation briefly coexist.
    router_overlap = max((sum(part.resident_bytes for part in layer.moe.router.parts)
                          for layer in prototype.layers), default=0)
    staging = max(STAGING_BYTES, router_overlap)
    # At most two underlying gathers run (one coordinator plus one read-ahead
    # worker).32bytes/value bounds the sum of direct decoder array payloads;
    # at most4 owned FP16 result planes cover active/cached/retired futures.
    # Row metadata and Python/runtime allocator effects remain separately
    # observable target-qualification requirements, never a claimed zero.
    ple_values = max((selection_rows * layer.ple.ngram.heads * layer.ple.ngram.dims
                      for layer in prototype.layers if layer.ple is not None), default=0)
    ple_workspace = (2 * 32 + 4 * 2) * ple_values
    prefetch = len(planner.tables) * 8 * PREFETCH_READ
    return {"resident_bytes": resident, "staging_bytes": staging,
            "workspace_bytes": workspace + prototype.cfg.rotary_dim // 2 * 4,
            "host_resident_bytes": mapped,
            "host_staging_bytes": 3 * STAGING_BYTES + prefetch,
            "host_workspace_bytes": host_workspace, "selection_rows": selection_rows,
            "selection_slots": prototype.cfg.top_k + 1, "selection_experts": prototype.cfg.experts + 1,
            "host_ple_workspace_bytes": ple_workspace,
            "host_runtime_managed_bytes": None,
            "host_runtime_managed_scope": "row sorting/thread/future/object/allocator peaks require target qualification",
            "execution_dtype": "float16", "numerical_profile": runtime.NUMERICAL_PROFILE,
            "linear_kernel_family": kernel_family, "plain_linear_temporary_bytes": plain_temporary,
            "expert_kernel_family": expert_family, "expert_descriptor_bytes": descriptor_bytes,
            "qualification": "unqualified",
            "tensors": [{"source": part.name, "runtime": "retained-e2m1-e4m3-global" if part.kind == "nvfp4"
                          else "excluded-float16", "rows": part.n, "columns": part.k,
                          "row_ranges": part.rows, "column_offset": part.column,
                          "resident_bytes": part.resident_bytes} for part in planner.matrices]
                       + [{"source": part.name, "runtime": part.dtype, "transform": part.transform,
                           "shape": part.shape, "resident_bytes": part.resident_bytes} for part in planner.arrays]}


def _validate_float(value, name, *, half=False):
    if value.device.type != "cpu":
        raise ValueError("SM70 payload validation belongs to CPU import preparation")
    if value.ndim != 2:
        value = value.reshape(1, -1)
    for row in range(0, value.shape[0], runtime.OUTPUTS):
        for column in range(0, value.shape[1], runtime.INPUTS):
            tile = value[row:row + runtime.OUTPUTS, column:column + runtime.INPUTS].float().numpy()
            #65520 is the FP16 RNE overflow boundary, not65504 (the largest
            #representable value). Values below the midpoint still round finite.
            if not np.isfinite(tile).all() or half and np.any(np.abs(tile) >= 65520):
                raise ValueError(f"{name}: values do not fit the manifest's finite {'FP16' if half else 'FP32'} profile")


def _finite_half(value, name):
    _validate_float(value, name, half=True)
    return value.half()


def _validate_nvfp4(codes, scales, factor, name):
    """Exact finite-FP16 range proof without an expanded weight mirror.

    E2M1 magnitude codes are monotone. The maximum magnitude in each16-value
    block therefore determines its range under the admitted FP32 scale order.
    Sum of array payloads created by the owned vector/E4M3 operations is below
    256bytes/group; derive the group count from the existing32MiB import cap.
    """
    packed = codes.numpy().reshape(-1, 8)
    blocks = scales.numpy().reshape(-1)
    step = STAGING_BYTES // 256
    for start in range(0, len(blocks), step):
        data = packed[start:start + step]
        magnitude = np.maximum(data & 7, (data >> 4) & 7).max(axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            maximum = E2M1[magnitude] * np.abs(e4m3(blocks[start:start + step]))
            maximum *= np.float32(factor)
        if not np.isfinite(maximum).all() or np.any(maximum >= 65520):
            raise ValueError(f"{name}: decoded NVFP4 values do not fit the finite FP16 operand profile")


def _half_rows(values):
    with np.errstate(over="ignore", invalid="ignore"):
        result = values.astype(np.float16)
    if not np.isfinite(values).all() or not np.isfinite(result).all():
        raise ValueError("PLE source values do not fit the declared finite FP16 profile")
    return result


def _nvfp4_half(codes, blocks, *, factor, table):
    nibbles = np.stack((codes & 15, codes >> 4), axis=-1).reshape(codes.shape[0], codes.shape[1] * 2)
    with np.errstate(over="ignore", invalid="ignore"):
        values = E2M1[nibbles] * np.repeat(table[blocks], 16, axis=1)
        values *= np.float32(factor)
    return _half_rows(values)


class HalfPLETable:
    """Typed FP16 row adapter over the existing mapped/positional lifetime.

    BF16 source bits are widened by representation, never numerically cast as
    uint16. Encoded tables reconstruct directly to FP16, avoiding an extra
    BF16 rounding boundary. ReadAhead owns this adapter and its source close.
    """
    bits, dtype, weight_scale = 16, "float16", 1.0

    def __init__(self, table, kind, factor):
        self.table, self.kind, self.factor = table, kind, factor
        self.rows, self.width, self.nbytes, self.wrow = table.rows, table.width, table.nbytes, table.wrow
        self.e4 = e4m3(np.arange(256, dtype=np.uint8))
        self.lut = None
        if kind == "fp8":
            with np.errstate(over="ignore", invalid="ignore"):
                self.lut = (self.e4 * np.float32(factor)).astype(np.float16)
        if isinstance(table, EncodedSSDTable):
            # Owned, unpublished constructor callback; no reader can observe
            # the source's legacy BF16 decoder before this typed binding.
            if kind == "nvfp4":
                table.decode = partial(_nvfp4_half, factor=factor, table=self.e4)
            elif kind == "fp8":
                table.decode = partial(np.take, self.lut)

    def gather(self, ids):
        return self._gather_owned(row_ids(ids, self.rows))

    def _gather_owned(self, flat):
        return self.table._life.call(self._gather_flat, flat)

    def _gather_flat(self, flat):
        if isinstance(self.table, EncodedSSDTable) and self.kind != "bf16":
            return self.table._gather_flat(flat)
        if self.kind == "bf16":
            bits = self.table._gather_flat(flat)
            values = (bits.astype(np.uint32) << 16).view(np.float32)
            with np.errstate(over="ignore", invalid="ignore"):
                values *= np.float32(self.factor)
            return _half_rows(values)
        if self.kind == "fp8":
            codes = BF16Table._gather_flat(self.table, flat)
            result = self.lut[codes]
            if not np.isfinite(result).all():
                raise ValueError("PLE FP8 values violate the imported finite FP16 invariant")
            return result
        shard, local = self.table._where(flat)
        codes = np.empty((flat.size, self.width // 2), dtype=np.uint8)
        blocks = np.empty((flat.size, self.width // 16), dtype=np.uint8)
        def copy(index, at):
            codes[at], blocks[at] = self.table.values[index][local[at]], self.table.scales[index][local[at]]
        _copy_rows(self.table._life, self.table._pool, shard, copy)
        return _nvfp4_half(codes, blocks, factor=self.factor, table=self.e4)

    def close(self):
        return self.table.close()

    def prefetch(self, workers=8):
        return self.table.prefetch(workers)

    def lock(self):
        return self.table.lock()


def _validate_ple(prototype, planner, reader):
    factors = {}
    for layer in prototype.layers:
        if layer.ple is None:
            continue
        table = layer.ple.table
        layer.ple.ngram.check(*(reader.get(table.base + name).numpy() for name in
            ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")))
        factor = (1.0 if table.scale_name is None else
                  calibration(float(reader.get(table.scale_name).float().reshape(-1)[0]), name=table.scale_name))
        factors[id(table)] = factor
        bad = None
        if table.kind != "nvfp4":
            if table.kind == "bf16":
                values = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
            else:
                values = e4m3(np.arange(256, dtype=np.uint8))
            with np.errstate(over="ignore", invalid="ignore"):
                values *= np.float32(factor)
                converted = values.astype(np.float16)
            bad = ~np.isfinite(values) | ~np.isfinite(converted)
        for _, name in table.shards:
            shape = planner.infos[name + ".weight"]["shape"]
            row_bytes = shape[1] * (2 if table.kind == "bf16" else 1)
            step = max(1, (STAGING_BYTES // 2) // row_bytes)
            for start in range(0, shape[0], step):
                end = min(shape[0], start + step)
                values = reader.get_rows(name + ".weight", start, end, device="cpu")
                if table.kind == "nvfp4":
                    scale = reader.get_rows(name + ".weight_scale", start, end, device="cpu").view(torch.uint8)
                    _validate_nvfp4(values, scale, factor, name)
                else:
                    encoded = values.view(torch.uint16 if table.kind == "bf16" else torch.uint8).numpy()
                    if np.any(bad[encoded]):
                        raise ValueError(f"{name}: PLE payload violates its finite FP16 profile")
    return factors


def _matrix(plan, reader, device, workspace, factors):
    n, k = plan.n, plan.k
    info = reader.info(plan.name + ".weight")
    source_width = math.prod(info["shape"][1:])
    source_row_bytes = source_width * (1 if plan.kind == "nvfp4" else
                        {"BF16": 2, "F16": 2, "F32": 4}[info["dtype"]])
    if source_row_bytes > STAGING_BYTES:
        raise ValueError(f"{plan.name}: one source row exceeds the bounded SM70 import span")
    rows_at_once = max(1, (STAGING_BYTES // 2) // source_row_bytes)
    factor = None
    if plan.kind == "nvfp4":
        global_scale = reader.get(plan.name + ".weight_scale_2")
        factor = calibration(float(global_scale.float().reshape(-1)[0]), name=plan.name + ".weight_scale_2")
        # Original activation calibration is identity evidence, never executed
        # as W4A4 by this explicit full-activation numerical profile.
        if reader.has(plan.name + ".input_scale"):
            calibration(float(reader.get(plan.name + ".input_scale").float().reshape(-1)[0]),
                        name=plan.name + ".input_scale")
        codes = torch.empty((n, k // 2), dtype=torch.uint8, device=device)
        scales = torch.empty((n, k // 16), dtype=torch.uint8, device=device)
    else:
        values = torch.empty((n, k), dtype=torch.float16, device=device)
    destination = 0
    for first, count in plan.rows:
        for begin in range(first, first + count, rows_at_once):
            end = min(first + count, begin + rows_at_once)
            value = reader.get_rows(plan.name + ".weight", begin, end, device="cpu")
            if plan.kind == "nvfp4":
                raw = value[:, plan.column // 2:(plan.column + k) // 2].contiguous()
                scale = reader.get_rows(plan.name + ".weight_scale", begin, end, device="cpu").view(torch.uint8)
                scale = scale[:, plan.column // 16:(plan.column + k) // 16].contiguous()
                _validate_nvfp4(raw, scale, factor, plan.name)
                codes[destination:destination + end - begin].copy_(raw)
                scales[destination:destination + end - begin].copy_(scale)
            else:
                raw = value[:, plan.column:plan.column + k]
                values[destination:destination + end - begin].copy_(_finite_half(raw, plan.name))
            destination += end - begin
    if plan.kind == "nvfp4":
        global_factor = factors[plan.factor_index:plan.factor_index + 1]
        global_factor.fill_(factor)
        return runtime.PackedLinear(codes, scales, global_factor, workspace)
    return runtime.FloatLinear(values)


def _array(plan, reader, device, norm_offset):
    info = reader.info(plan.name)
    if math.prod(info["shape"]) * {"BF16": 2, "F16": 2, "F32": 4}[info["dtype"]] > STAGING_BYTES:
        raise ValueError(f"{plan.name}: array exceeds the bounded parameter import span")
    value = reader.get(plan.name).cpu()
    if plan.indices is not None:
        value = value[list(plan.indices)]
    value = value.reshape(plan.shape).float()
    _validate_float(value, plan.name)
    if plan.transform == "norm_offset":
        value.add_(norm_offset)
    elif plan.transform == "half_to_float":
        value = _finite_half(value, plan.name).float()
    elif plan.transform != "identity":
        raise ValueError(f"{plan.name}: unimplemented parameter transform")
    if plan.dtype == "float16":
        value = _finite_half(value, plan.name)
    return value.to(device).contiguous()


def _close_reader(reader):
    # Construction is journaled before __init__. Only acquired fields exist
    # when a constructor fails; do not invent a normal initialized reader.
    namespace = vars(reader)
    if "reads" in namespace:
        reader.close()
    elif "io" in namespace:
        reader.io.close()


def load(directory, manifest_path, device="cuda", *, tp=None, rope=None, eos_token_ids=None,
         mtp=False, mtp_model_dir=None, _plan=None, _workspace=None, _weight_plan=None, **options):
    """Materialize one declared shared-family plan; never import through BF16 banks."""
    from tensorfold.cuda import precision
    from .sm70_mtp_weights import attach

    if precision.mode() != precision.FULL:
        raise ValueError("SM70 NVFP4 requires explicit --precision full and its unqualified W4A16 numerical manifest")
    if options.get("vram_experts") is not None or options.get("expert_loader") is not None:
        raise ValueError("this SM70 NVFP4 recipe requires admitted resident TP weight shards; host expert pools are distinct")
    if mtp and mtp_model_dir is None:
        raise ValueError("this SM70 NVFP4 recipe has no implemented embedded MTP layout; select none or a validated stock companion")
    prototype, planner = model_plan(directory, manifest_path, tp, rope=rope, eos_token_ids=eos_token_ids) if _plan is None else _plan
    if not isinstance(planner, Planner) or planner.directory != Path(directory).resolve():
        raise ValueError("SM70 import plan belongs to another source directory")
    planner.manifest.check_versions(directory)
    weight_plan = byte_plan(prototype, planner) if _weight_plan is None else _weight_plan
    width = max(prototype.cfg.moe_width, prototype.cfg.shared_width)
    reader, tables, cache = object.__new__(_Reader), PLETables(), {}
    workspace = None
    factors = None
    table_factors = {}

    def materialize(value):
        if isinstance(value, (Matrix, Array, Concat, Experts, Table)):
            key = id(value)
            if key not in cache:
                cache[key] = owned(value)
            return cache[key]
        return owned(value)

    def owned(value):
        if isinstance(value, Matrix):
            return _matrix(value, reader, device, workspace, factors)
        if isinstance(value, Array):
            return _array(value, reader, device, planner.manifest.norm_offset)
        if isinstance(value, Concat):
            if all(isinstance(part, Array) for part in value.parts):
                return torch.cat(tuple(_array(part, reader, device, planner.manifest.norm_offset) for part in value.parts))
            return GGUFConcat(tuple(materialize(part) for part in value.parts))
        if isinstance(value, Experts):
            return runtime.ExpertBank(tuple(tuple(materialize(part) for part in run) for run in value.parts),
                                      workspace, width=value.width, dims=value.dims)
        if isinstance(value, Table):
            def scale(field):
                name = value.base + "ngram_embedding." + field
                if name != value.scale_name and not (value.scale_name is None and field == "weight_scale"):
                    raise ValueError(f"{name}: PLE callback differs from its admitted calibration convention")
                return table_factors[id(value)]
            raw = tables.acquire(open_table(Path(directory), list(value.shards), scale,
                                            ssd=bool(options.get("ple_on_ssd", False))))
            converted = HalfPLETable(raw, value.kind, table_factors[id(value)])
            table = ReadAhead(converted)
            tables.unpublished = table
            tables.tables[-1] = table  # The outer owner now drains reads before closing the same source.
            tables.unpublished = None
            if table.rows != value.rows or table.width != value.dims:
                raise ValueError("SM70 NVFP4 PLE table geometry differs from its admitted hash space")
            external = options.get("table_reads")
            if external is not None and not options.get("ple_on_ssd", False):
                from tensorfold.cuda.direct_read import in_background

                in_background(table.prefetch, tables.reads)
                external.append(tables.reads[-1])
            return table
        if isinstance(value, (HC, GDNW, AttnW, MoEW, PLEW, LayerW)):
            return type(value)(**{field.name: materialize(getattr(value, field.name)) for field in fields(value)})
        return value

    try:
        reader.__init__(Path(directory), "cpu")
        # Whole PLE range checks stream packed/bit encodings through bounded
        # CPU spans, before retaining any model parameter on the device.
        table_factors = _validate_ple(prototype, planner, reader)
        workspace = (runtime.Workspace(device, width=width, dims=prototype.cfg.hidden,
                     selection_rows=weight_plan["selection_rows"], slots=weight_plan["selection_slots"],
                     experts=weight_plan["selection_experts"], kernel_family=weight_plan["linear_kernel_family"],
                     expert_family=weight_plan["expert_kernel_family"])
                     if _workspace is None else _workspace)
        if (workspace.kernel_family != weight_plan["linear_kernel_family"]
                or workspace.expert_family != weight_plan["expert_kernel_family"]):
            raise ValueError("prepared SM70 workspace differs from the manifest's selected kernel family")
        factors = torch.empty(planner.factor_count, dtype=torch.float32, device=device)
        embed = runtime.FloatEmbedding(materialize(prototype.embed[0]).weight)
        result = Weights(prototype.cfg, (embed,), [materialize(layer) for layer in prototype.layers],
                         materialize(prototype.mixer), materialize(prototype.head),
                         prototype.cfg.rope.inverse_frequencies(torch).to(device), around_one=True)
        result.meta.update(prototype.meta, ple_tables=tables, sm70_weight_workspace=workspace,
                           weight_plan=weight_plan,
                           family_source=planner.manifest.document["source"],
                           expert_execution=weight_plan["expert_kernel_family"])
        if mtp:
            attach(result, mtp_model_dir, tp=(result.meta["rank"], result.meta["world"]))
        _close_reader(reader)
        return result
    except BaseException as primary:
        _retain_failure(primary, "ple_tables", tables)
        failures = []
        def fence():
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
        for cleanup in (lambda: _close_reader(reader), lambda: tables.close(fence)):
            try:
                cleanup()
            except BaseException as error:
                failures.append(error)
        if failures:
            _retain_failure(primary, "_tensorfold_retained_owners", (reader, tables, workspace, factors, cache))
        raise_failures(primary, failures)


def capacity_options(prototype, planner, *, mtp=False, mtp_model_dir=None,
                     prefill_rows=1024, execution_rows=32, workspace_preallocated=False):
    from tensorfold.cuda.capacity import Weights as WeightBytes
    from .sm70_mtp_weights import byte_plan as mtp_bytes

    plan = byte_plan(prototype, planner, prefill_rows=prefill_rows, execution_rows=execution_rows)
    if mtp:
        if mtp_model_dir is None:
            raise ValueError("SM70 NVFP4 MTP requires an explicitly validated companion")
        extra = mtp_bytes(prototype, mtp_model_dir, tp=(prototype.meta["rank"], prototype.meta["world"]))
        plan["resident_bytes"] += extra["resident_bytes"]
        plan["workspace_bytes"] += extra["workspace_bytes"]
        for name in ("staging_bytes", "host_staging_bytes"):
            plan[name] = max(plan[name], extra[name])
        plan["stock_mtp"] = extra

    def estimate(_directory):
        return WeightBytes(plan["resident_bytes"], plan["staging_bytes"], 0), plan["host_staging_bytes"]

    workspace = plan["workspace_bytes"]
    if workspace_preallocated:
        workspace -= runtime.workspace_bytes(width=max(prototype.cfg.moe_width, prototype.cfg.shared_width),
            dims=prototype.cfg.hidden, kernel_family=plan["linear_kernel_family"],
            expert_family=plan["expert_kernel_family"], selection_rows=plan["selection_rows"],
            slots=plan["selection_slots"], experts=plan["selection_experts"])
    return {"weight_estimator": estimate, "host_resident": plan["host_resident_bytes"],
            "host_workspace_bytes": prototype.meta["world"] * (plan["host_ple_workspace_bytes"]
                + (0 if workspace_preallocated else plan["host_workspace_bytes"])),
            "device_workspace_bytes": workspace}, plan
