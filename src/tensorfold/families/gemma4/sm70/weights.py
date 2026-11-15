"""Rank shards of the AEON compressed-tensors NVFP4 Gemma 4 checkpoint for the SM70 TP8/ETP4 provider.

Original FP4 code bytes and E4M3 block-scale bytes are sliced, never re-encoded. Each projection keeps its
own 1/weight_global_scale as an FP32 per-output-row multiplier so projections sharing an input concatenate
into one launch without rescaling. Layout (world 8):
  sliding attention: q heads 2r, 2r+1 and KV head r; full attention: q heads 2r, 2r+1 and KV head r // 4;
  o_proj: the matching input columns; dense MLP: 272 (ranks 0-3) / 256 (ranks 4-7) intermediate rows;
  experts (ETP4 x EP2): experts [64 * (r // 4), +64), intermediate slice [176 * (r % 4), +176).
BF16 tensors (tied embedding/head, router, norms, scalars) stay original BF16; the embedding is replicated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..cuda.weights import Config
from ..cuda.nvfp4_weights import CT_PREFIX, validate_headers

WORLD = 8
ETP = 4


def dense_split(intermediate: int, world: int = WORLD) -> list[tuple[int, int]]:
    """16-aligned (offset, width) slices covering the dense intermediate width."""
    groups = intermediate // 16
    if intermediate % 16:
        raise ValueError("Gemma 4 dense intermediate width must be a multiple of the NVFP4 group")
    base, extra = divmod(groups, world)
    out, offset = [], 0
    for r in range(world):
        width = 16 * (base + (1 if r < extra else 0))
        out.append((offset, width))
        offset += width
    return out


@dataclass
class Packed:
    """FP4 codes [.., N, K/2] u8, E4M3 bytes [.., N, K/16] u8 and FP32 row multipliers [.., N]."""
    codes: Any
    scales: Any
    rowmul: Any

    @property
    def n(self):
        return self.codes.shape[-2]

    @property
    def k(self):
        return self.codes.shape[-1] * 2

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.codes, self.scales, self.rowmul))


@dataclass
class Layer:
    sliding: bool
    qkv: Packed           # q rows | k rows (| v rows on sliding layers)
    o: Packed
    gate_up: Packed       # gate rows | up rows (dense, this rank's width)
    down: Packed
    experts_gate_up: Packed   # [64, 2*176, K/2]
    experts_down: Packed      # [64, 2816, 176/2]
    router: Any
    norms: dict = field(default_factory=dict)


@dataclass
class RankWeights:
    config: Config
    rank: int
    world: int
    embed: Any            # BF16 [vocab, hidden], replicated
    head: Any             # view: this rank's vocabulary rows
    norm: Any
    layers: list
    dense_width: int
    expert_first: int
    expert_count: int
    expert_width: int
    raw_config: dict

    def nbytes(self):
        total = self.embed.numel() * 2
        for layer in self.layers:
            for p in (layer.qkv, layer.o, layer.gate_up, layer.down, layer.experts_gate_up, layer.experts_down):
                total += p.nbytes()
            total += layer.router.numel() * 2
        return total


def _check_scales(tensor, name):
    import torch
    raw = tensor.view(torch.uint8)
    if bool(((raw & 0x7F) == 0x7F).any()):
        raise ValueError(f"{name}: E4M3 block scale is NaN")
    return raw


def load(model_dir: str | Path, rank: int, world: int = WORLD, device="cuda") -> RankWeights:
    import torch
    from tensorfold.checkpoint_metadata import read_config, read_json
    from tensorfold.cuda.direct_read import SafeTensors
    from tensorfold.cuda.nvfp4.format import calibration
    from tensorfold.tensor_file import checkpoint_path

    if world != WORLD or not 0 <= rank < world:
        raise ValueError("Gemma 4 SM70 provider requires TP8 ranks 0..7")
    root = Path(model_dir)
    raw = read_config(root)
    index = read_json(root, "model.safetensors.index.json")
    files = ([checkpoint_path(root, "model.safetensors")] if index is None else
             [checkpoint_path(root, name) for name in sorted(set(index["weight_map"].values()))])
    reader = SafeTensors([])
    for file in files:
        shard = SafeTensors([file], reader=reader.reader)
        reader.where.update(shard.where)
    headers = {name: {"dtype": v[3], "shape": v[4]} for name, v in reader.where.items()}
    c = validate_headers(raw, headers, vision=False)
    if (c.hidden, c.heads, c.kv_heads, c.global_kv_heads, c.head_dim, c.global_head_dim, c.experts, c.top_k,
            c.moe_width) != (2816, 16, 8, 2, 256, 512, 128, 8, 704):
        raise ValueError("Gemma 4 SM70 TP8 layout is prepared for the 26B-A4B geometry")
    dev = torch.device(device)
    scalars = {}

    def cpu(name):
        return reader.get(name, "cpu")

    def gscale(name):
        if name not in scalars:
            scalars[name] = calibration(float(cpu(name + ".weight_global_scale").item()), name=name,
                                        reciprocal=True)
        return scalars[name]

    def part(name, rows=None, cols=None):
        """(codes, scales, rowmul) CPU slices; rows/cols are element ranges of the logical [N, K] matrix."""
        codes = cpu(name + ".weight_packed")
        scales = _check_scales(cpu(name + ".weight_scale"), name)
        if rows is not None:
            codes, scales = codes[rows[0]:rows[1]], scales[rows[0]:rows[1]]
        if cols is not None:
            if cols[0] % 16 or cols[1] % 16:
                raise ValueError(f"{name}: input slice must align to the NVFP4 group")
            codes, scales = codes[:, cols[0] // 2:cols[1] // 2], scales[:, cols[0] // 16:cols[1] // 16]
        mul = torch.full((codes.shape[0],), gscale(name), dtype=torch.float32)
        return codes, scales, mul

    def pack(parts, stack=False):
        join = torch.stack if stack else torch.cat
        codes = join([p[0] for p in parts]).contiguous()
        scales = join([p[1] for p in parts]).contiguous()
        mul = join([p[2] for p in parts]).contiguous()
        return Packed(codes.to(dev), scales.to(dev), mul.to(dev))

    split = dense_split(c.intermediate, world)
    d_off, d_width = split[rank]
    groups = world // ETP
    count = c.experts // groups
    first = (rank // ETP) * count
    ew = c.moe_width // ETP
    e_off = (rank % ETP) * ew
    layers = []
    for i in range(c.layers):
        p = f"{CT_PREFIX}layers.{i}."
        sliding = c.sliding(i)
        if sliding:
            d = c.head_dim
            qkv = pack([part(p + "self_attn.q_proj", rows=(2 * rank * d, 2 * rank * d + 2 * d)),
                        part(p + "self_attn.k_proj", rows=(rank * d, rank * d + d)),
                        part(p + "self_attn.v_proj", rows=(rank * d, rank * d + d))])
        else:
            d = c.global_head_dim
            kv = rank // (world // c.global_kv_heads)
            qkv = pack([part(p + "self_attn.q_proj", rows=(2 * rank * d, 2 * rank * d + 2 * d)),
                        part(p + "self_attn.k_proj", rows=(kv * d, kv * d + d))])
        o = pack([part(p + "self_attn.o_proj", cols=(2 * rank * d, 2 * rank * d + 2 * d))])
        gate_up = pack([part(p + "mlp.gate_proj", rows=(d_off, d_off + d_width)),
                        part(p + "mlp.up_proj", rows=(d_off, d_off + d_width))])
        down = pack([part(p + "mlp.down_proj", cols=(d_off, d_off + d_width))])
        egu, edown = [], []
        for e in range(first, first + count):
            q = f"{p}experts.{e}."
            g = part(q + "gate_proj", rows=(e_off, e_off + ew))
            u = part(q + "up_proj", rows=(e_off, e_off + ew))
            egu.append(tuple(torch.cat([a, b]) for a, b in zip(g, u)))
            edown.append(part(q + "down_proj", cols=(e_off, e_off + ew)))
        norms = {}
        for key, name in (("input", "input_layernorm.weight"), ("post_attn", "post_attention_layernorm.weight"),
                          ("pre_ff", "pre_feedforward_layernorm.weight"),
                          ("pre_ff2", "pre_feedforward_layernorm_2.weight"),
                          ("post_ff1", "post_feedforward_layernorm_1.weight"),
                          ("post_ff2", "post_feedforward_layernorm_2.weight"),
                          ("post_ff", "post_feedforward_layernorm.weight"), ("q", "self_attn.q_norm.weight"),
                          ("k", "self_attn.k_norm.weight"), ("router_scale", "router.scale"),
                          ("expert_scale", "router.per_expert_scale"), ("scalar", "layer_scalar")):
            norms[key] = cpu(p + name).to(dev).contiguous()
        layers.append(Layer(sliding, qkv, o, gate_up, down, pack(egu, stack=True), pack(edown, stack=True),
                            cpu(p + "router.proj.weight").to(dev).contiguous(), norms))
    embed = cpu(CT_PREFIX + "embed_tokens.weight").to(dev).contiguous()
    norm = cpu(CT_PREFIX + "norm.weight").to(dev).contiguous()
    reader.close()
    shard = c.vocab // world
    torch.cuda.synchronize(dev)
    return RankWeights(c, rank, world, embed, embed[rank * shard:(rank + 1) * shard], norm, layers, d_width,
                       first, count, ew, raw)
