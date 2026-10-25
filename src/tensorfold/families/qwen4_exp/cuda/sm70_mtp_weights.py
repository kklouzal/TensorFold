"""Matching stock floating or shared-GGUF MTP in the FP16 Flash Next runtime.

MTP shares target embeddings/head, uses the existing MTPW forward protocol and
retains tensor-parallel expert/intermediate slices. No checkpoint is rewritten.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tensorfold.checkpoint_metadata import read_config
from .weight_types import AttnW, HC, LayerW, MoEW, MTPW


class Dense:
    kernel = "sm70_dense"

    def __init__(self, weight):
        if weight.ndim != 2 or weight.dtype != torch.float16:
            raise ValueError("SM70 dense weights require FP16 [output,input]")
        self.weight = weight.contiguous()
        self.n, self.k = map(int, weight.shape)

    def matmul(self, x, out=None, f32=False):
        if x.ndim != 2 or x.shape[1] != self.k or x.dtype != torch.float16 or x.device != self.weight.device:
            raise ValueError("SM70 dense input differs from its weight geometry/device/dtype")
        result = torch.empty((x.shape[0], self.n), dtype=torch.float32, device=x.device)
        # Fixed row tiles retain the same arithmetic when prompt/decode rows regroup.
        for start in range(0, x.shape[0], 32):
            count = min(32, x.shape[0] - start)
            padded = torch.zeros((32, self.k), dtype=x.dtype, device=x.device)
            padded[:count].copy_(x[start:start + count])
            value = (torch.mm(padded, self.weight.T, out_dtype=torch.float32) if x.is_cuda
                     else padded.float() @ self.weight.float().T)
            result[start:start + count].copy_(value[:count])
        if out is not None:
            if out.shape != result.shape or out.device != x.device or out.dtype != (torch.float32 if f32 else x.dtype):
                raise ValueError("SM70 dense output differs from its requested shape/device/dtype")
            out.copy_(result)
            return out
        return result if f32 else result.half()

    def __call__(self, x, out=None):
        return self.matmul(x, out=out, f32=out is not None and out.dtype == torch.float32)

    prefill = __call__

    def partials(self, x):
        return self.matmul(x, f32=True).unsqueeze(0)

    def nbytes(self):
        return self.weight.numel() * self.weight.element_size()

    def device_tensors(self):
        return (self.weight,)


class DenseExperts:
    """Bounded gather/BMM expert execution, with unchanged per-slot HC reduction."""
    kernel = "sm70_dense"
    limit = 0.0
    swiglu = True

    def __init__(self, gate, up, down, *, tile_pairs=16):
        if (gate.ndim != 3 or gate.shape != up.shape or down.shape != (gate.shape[0], gate.shape[2], gate.shape[1])
                or any(t.dtype != torch.float16 or t.device != gate.device for t in (gate, up, down))):
            raise ValueError("SM70 MTP expert bank geometry/dtype/device disagrees")
        self.gate, self.up, self.down = gate, up, down
        self.count, self.width, self.dims = gate.shape
        self.tile_pairs = int(tile_pairs)
        if self.tile_pairs <= 0:
            raise ValueError("SM70 MTP expert staging tile must be positive")

    def forward(self, x, picks, out=None):
        if (x.ndim != 2 or x.shape[1] != self.dims or x.dtype != torch.float16
                or picks.ndim != 2 or picks.shape[0] != x.shape[0] or picks.dtype not in (torch.int32, torch.int64)
                or picks.device != x.device or x.device != self.gate.device):
            raise ValueError("SM70 MTP expert inputs differ from their admitted bank")
        rows, slots = picks.shape
        if out is None:
            out = torch.empty((rows, slots, self.dims), dtype=torch.float32, device=x.device)
        if out.shape != (rows, slots, self.dims) or out.device != x.device or out.dtype not in (torch.float16, torch.float32):
            raise ValueError("SM70 MTP expert output differs from its admitted shape/device/dtype")
        flat_ids = picks.reshape(-1).long()
        flat_x = x[:, None, :].expand(rows, slots, self.dims).reshape(-1, self.dims)
        flat_out = out.view(-1, self.dims)
        for start in range(0, rows * slots, self.tile_pairs):
            count = min(self.tile_pairs, rows * slots - start)
            ids = torch.zeros(self.tile_pairs, dtype=torch.int64, device=x.device)
            values = torch.zeros((self.tile_pairs, self.dims, 1), dtype=x.dtype, device=x.device)
            ids[:count].copy_(flat_ids[start:start + count])
            values[:count].copy_(flat_x[start:start + count, :, None])
            gate = self.gate.index_select(0, ids)
            up = self.up.index_select(0, ids)
            if x.is_cuda:
                g = torch.bmm(gate, values, out_dtype=torch.float32).squeeze(-1).half()
                u = torch.bmm(up, values, out_dtype=torch.float32).squeeze(-1).half()
            else:
                g = torch.bmm(gate.float(), values.float()).squeeze(-1).half()
                u = torch.bmm(up.float(), values.float()).squeeze(-1).half()
            activation = torch.nn.functional.silu(g) * u
            down = self.down.index_select(0, ids)
            result = (torch.bmm(down, activation[..., None], out_dtype=torch.float32) if x.is_cuda else
                      torch.bmm(down.float(), activation.float()[..., None]))
            flat_out[start:start + count].copy_(result[:count].squeeze(-1))
        return out

    def device_tensors(self):
        return self.gate, self.up, self.down

    def bytes_per_expert(self):
        return sum(t[0].numel() * t.element_size() for t in self.device_tensors())


def _validate(target, directory):
    cfg = read_config(directory)
    text = cfg.get("text_config") or cfg
    main = target.cfg if hasattr(target, "cfg") else target
    pairs = (("hidden_size", "hidden"), ("head_dim", "head_dim"), ("num_experts", "experts"),
             ("hc_count", "streams"), ("hc_lowrank", "low"))
    for field, attr in pairs:
        if int(text.get(field, -1)) != int(getattr(main, attr)):
            raise ValueError(f"stock MTP {field} does not match its target")
    mtp = text.get("mtp") or {}
    if mtp.get("num_hidden_layers") != 1 or mtp.get("layer_types") != ["full_attention"]:
        raise ValueError("stock MTP requires its declared single full-attention layer")
    file = Path(directory) / "model.safetensors"
    if not file.is_file():
        raise ValueError("stock MTP requires the original model.safetensors")
    return file


def byte_plan(target, directory, *, tp=(0, 1)):
    """Exact retained FP16/F32 arrays; shared target embedding/head are excluded."""
    from ..gguf import resolve_gguf

    if resolve_gguf(directory) is not None:
        from .gguf_mtp_weights import byte_plan as gguf_plan

        return gguf_plan(target, directory, tp=tp)
    _validate(target, directory)
    c = target.cfg if hasattr(target, "cfg") else target
    _, world = tp
    d, wide, low = c.hidden, c.streams * c.hidden, c.low
    width = c.moe_width  # loader's cfg is already local for this rank
    # three packed-by-expert FP16 banks (gate/up/down), router, fusion linears.
    resident = (c.experts + 1) * width * d * 6 + (c.experts + 1) * d * 2 + 4 * d * d
    resident += 2 * ((low + c.streams) * wide * 2 + wide * low * 2 + wide * 4)
    resident += low * wide * 4 + wide * 4
    projection = 2 * c.heads * c.head_dim + 2 * c.kv_heads * c.head_dim + (c.index_heads + 1) * c.index_dim
    resident += projection * d * 2 + d * c.heads * c.head_dim * 2
    resident += (2 * c.head_dim + 2 * c.index_dim + d + wide) * 4
    workspace = 16 * width * d * 6 + 16 * (width + d) * 8
    return {"resident_bytes": int(resident), "staging_bytes": 32 << 20,
            "host_staging_bytes": 32 << 20, "workspace_bytes": int(workspace), "world": world}


def attach(w, directory, *, tp=(0, 1)):
    """Attach the matching declared stock MTP using existing family weight types."""
    from ..gguf import resolve_gguf

    if resolve_gguf(directory) is not None:
        from .gguf_mtp_weights import attach as attach_gguf

        return attach_gguf(w, directory, tp=tp)
    from safetensors import safe_open

    rank, world = tp
    c, device = w.cfg, w.device
    file = _validate(w, directory)
    full_cfg = read_config(directory)["text_config"]
    global_heads, global_kv = int(full_cfg["num_attention_heads"]), int(full_cfg["num_key_value_heads"])
    if global_heads // world != c.heads or max(1, global_kv // world) != c.kv_heads:
        raise ValueError("stock MTP attention partition differs from target")
    width, d = c.moe_width, c.hidden
    full_width = int(full_cfg["moe_intermediate_size"])
    if width * world != full_width:
        raise ValueError("stock MTP expert width does not partition its target")
    lo, hi = rank * width, (rank + 1) * width
    base = "mtp.layers.0."
    with safe_open(file, framework="pt", device="cpu") as source:
        def raw(name, slices=None, *, dtype=torch.float16):
            item = source.get_slice(name)
            if item.get_dtype() not in ("BF16", "F16", "F32"):
                raise ValueError(f"stock MTP {name} is not an original floating tensor")
            value = item[:] if slices is None else item[slices]
            return value.to(dtype=dtype).contiguous().to(device)

        def gamma(name):
            # This source declares GemmaRMSNorm: gamma = 1 + trained parameter.
            return (raw(name, dtype=torch.float32) + 1.0).contiguous()

        def hc(prefix, inject):
            down = raw(prefix + ".input_mix_weight_down.weight")
            if inject:
                down = torch.cat((down, raw(prefix + ".block_inject_weight.weight")))
            return HC(Dense(down), Dense(raw(prefix + ".input_mix_weight_up.weight")),
                      gamma(prefix + ".hc_norm.weight"), inject)

        qrows = c.heads * 2 * c.head_dim
        q = raw(base + "self_attn.q_proj.weight", (slice(rank * qrows, (rank + 1) * qrows), slice(None)))
        kv_head = rank // (world // global_kv) if world > global_kv else rank * c.kv_heads
        kv_rows = slice(kv_head * c.head_dim, (kv_head + c.kv_heads) * c.head_dim)
        k = raw(base + "self_attn.k_proj.weight", (kv_rows, slice(None)))
        v = raw(base + "self_attn.v_proj.weight", (kv_rows, slice(None)))
        index = raw(base + "self_attn.indexer.index_qk_proj.weight")
        out_columns = slice(rank * c.heads * c.head_dim, (rank + 1) * c.heads * c.head_dim)
        attention = AttnW(Dense(torch.cat((q, k, v, index))), gamma(base + "self_attn.q_norm.weight"),
                          gamma(base + "self_attn.k_norm.weight"), gamma(base + "self_attn.indexer.q_layernorm.weight"),
                          gamma(base + "self_attn.indexer.k_layernorm.weight"),
                          Dense(raw(base + "self_attn.o_proj.weight", (slice(None), out_columns))))
        shape = (c.experts + 1, width, d)
        gate, up = [torch.empty(shape, dtype=torch.float16, device=device) for _ in range(2)]
        down = torch.empty((c.experts + 1, d, width), dtype=torch.float16, device=device)
        for expert in range(c.experts):
            gate[expert].copy_(raw(base + "mlp.experts.gate_up_proj", (expert, slice(lo, hi), slice(None))))
            up[expert].copy_(raw(base + "mlp.experts.gate_up_proj",
                                 (expert, slice(full_width + lo, full_width + hi), slice(None))))
            down[expert].copy_(raw(base + "mlp.experts.down_proj", (expert, slice(None), slice(lo, hi))))
        for tensor, name, slices in ((gate, "gate_proj", (slice(lo, hi), slice(None))),
                                     (up, "up_proj", (slice(lo, hi), slice(None))),
                                     (down, "down_proj", (slice(None), slice(lo, hi)))):
            tensor[-1].copy_(raw(base + f"mlp.shared_expert.{name}.weight", slices))
        router = torch.cat((raw(base + "mlp.gate.weight"), raw(base + "mlp.shared_expert_gate.weight")))
        layer = LayerW(c.layers, False, hc(base + "attn_hyper_connection", True),
                       hc(base + "mlp_hyper_connection", True), None, attention,
                       MoEW(router, DenseExperts(gate, up, down)))
        w.mtp = MTPW(gamma("mtp.pre_fc_norm_embedding.weight"), gamma("mtp.pre_fc_norm_hidden.weight"),
                    Dense(raw("mtp.fc_embedding.weight")), Dense(raw("mtp.fc_hidden.weight")), layer,
                    hc("mtp.hyper_connection_mixer", False))
        w.draft_head, w.draft_ids = None, None  # shared full-vocabulary head, preserving original stock math
    w.meta["mtp_checkpoint"] = str(file.resolve())
    w.meta["mtp_storage"] = "stock-bf16-execution-fp16"
    return w
