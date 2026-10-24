"""Optional owned M32 projection storage; logical TensorFold row views stay unchanged."""
from __future__ import annotations

from types import SimpleNamespace

import torch

# Private qualification policy; ordinary Buffers and generic matmul remain selected.
_ENABLED = False
ROWS = 32


def field_layout(c, rows, head_width, world):
    """(logical shape, element bytes) shared by allocation and admission."""
    wide = c.streams * c.hidden
    attention = c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim + (c.index_heads + 1) * c.index_dim
    half = {"normed": wide, "act": c.low, "up": wide, "mixed": c.hidden,
            "pa": attention, "gated": c.heads * c.head_dim, "gout": c.nv * 128,
            "ple_emb": c.ple_dim, "ple_keys": wide, "ple_vals": c.hidden}
    half["branch"] = c.hidden
    result = {name: ((rows, width), 2) for name, width in half.items()}
    result["logits"] = ((rows, head_width), 4)
    if world > 1:
        result["part_branch"] = ((rows, c.hidden), 4)
    linear = sum(kind == "linear" for kind in c.layer_types)
    projection = 2 * c.nk * 128 + 2 * c.nv * 128 + 2 * c.nv
    result["proj"] = ((linear, rows, projection), 2)
    return result


def workspace_bytes(c, rows, head_width, world, *, enabled=None, prefill=False):
    """Exact additional persistent bytes; original logical Buffers are already charged."""
    enabled = _ENABLED if enabled is None else enabled
    if not enabled or prefill or not 1 <= rows <= ROWS:
        return 0
    extra = 0
    for shape, size in field_layout(c, rows, head_width, world).values():
        if len(shape) == 2:
            extra += (ROWS - rows) * shape[1] * size
        else:
            extra += shape[0] * (ROWS - rows) * shape[2] * size
    maximum_input = max(c.streams * c.hidden, c.ple_dim, c.nv * 128, c.heads * c.head_dim, c.low)
    extra += ROWS * maximum_input * 2
    extra += ROWS * (c.low + c.low + c.streams) * 4
    return extra


def workspace_bytes_from_text(text, rows, world, *, enabled=None, prefill=False):
    """The same forecast before weights load, including replicated local KV heads."""
    enabled = _ENABLED if enabled is None else enabled
    if not enabled or prefill or not 1 <= rows <= ROWS:
        return 0
    text = text.get("text_config") or text
    c = SimpleNamespace(
        streams=int(text.get("hc_count", 4)), hidden=int(text["hidden_size"]),
        low=int(text.get("hc_lowrank", 320)), heads=int(text["num_attention_heads"]) // world,
        kv_heads=max(1, int(text["num_key_value_heads"]) // world),
        head_dim=int(text.get("head_dim") or int(text["hidden_size"]) // int(text["num_attention_heads"])),
        index_heads=int(text.get("indexer_n_heads", 4)), index_dim=int(text.get("indexer_head_dim", 128)),
        nk=int(text["linear_num_key_heads"]) // world, nv=int(text["linear_num_value_heads"]) // world,
        ple_dim=int(text.get("ple_embed_dim") or text["hidden_size"]),
        layer_types=["linear" if kind == "linear_attention" else "attention" for kind in text["layer_types"]],
    )
    return workspace_bytes(c, rows, int(text["vocab_size"]) // world, world, enabled=True)


class PaddedLinears:
    """All storage is owned before capture; only bounded reshape/slice views escape."""

    def __init__(self, w, rows):
        self.rows, self.device = rows, w.device
        c = w.cfg
        self.layout = field_layout(c, rows, w.head.n, int(w.meta.get("world", 1)))
        self.owners = {}
        self.inputs = []
        self.outputs = []
        self.maximum_input = max(c.streams * c.hidden, c.ple_dim, c.nv * 128, c.heads * c.head_dim, c.low)
        self.input_storage = torch.zeros((ROWS * self.maximum_input,), dtype=torch.float16, device=w.device)
        self.partial = {width: torch.zeros((ROWS, width), dtype=torch.float32, device=w.device)
                        for width in (c.low, c.low + c.streams)}
        self.outputs.extend(self.partial.values())

    def tensor(self, name, shape, dtype, *, zero=False):
        expected = self.layout.get(name)
        if expected is None:
            factory = torch.zeros if zero else torch.empty
            return factory(shape, dtype=dtype, device=self.device)
        if tuple(shape) != expected[0] or torch.empty((), dtype=dtype).element_size() != expected[1]:
            raise ValueError("padded projection allocation disagrees with its forecast")
        if len(shape) == 2:
            owner = torch.zeros((ROWS, shape[1]), dtype=dtype, device=self.device)
            matrices = [owner]
            view = owner[:shape[0]]
        else:
            owner = torch.zeros((shape[0], ROWS, shape[2]), dtype=dtype, device=self.device)
            matrices = list(owner.unbind(0))
            view = owner[:, :shape[1]]
        self.owners[name] = owner
        self.outputs.extend(matrices)
        if dtype == torch.float16:
            self.inputs.extend(matrices)
        return view

    @staticmethod
    def _matrix_view(value, owners):
        if value.ndim != 2 or not value.is_contiguous() or value.stride(0) != value.shape[1]:
            return None
        need = ROWS * value.shape[1]
        for owner in owners:
            if (owner.dtype == value.dtype and owner.device == value.device
                    and owner.data_ptr() == value.data_ptr() and owner.numel() >= need):
                return owner.reshape(-1)[:need].view(ROWS, value.shape[1])
        return None

    def matmul(self, x, matrix, out):
        eligible = getattr(matrix, "can_matmul_prepared", None)
        if (eligible is None or not eligible(out.dtype) or x.ndim != 2 or out.ndim != 2
                or not 1 <= x.shape[0] <= ROWS or out.shape != (x.shape[0], matrix.n)
                or x.shape[1] != matrix.k or x.dtype != torch.float16):
            return None
        target = self._matrix_view(out, self.outputs)
        if target is None:
            return None
        source = self._matrix_view(x, self.inputs)
        if source is None:
            if matrix.k > self.maximum_input:
                return None
            source = self.input_storage[:ROWS * matrix.k].view(ROWS, matrix.k)
            source[:x.shape[0]].copy_(x)
        matrix.matmul_prepared(source, target)
        return out

    def partials(self, matrix, x):
        owner = self.partial.get(matrix.n)
        if owner is None or not 1 <= x.shape[0] <= self.rows:
            return None
        out = owner[:x.shape[0]]
        result = self.matmul(x, matrix, out)
        return None if result is None else result.unsqueeze(0)


def create(w, rows, *, prefill, dtype):
    if (not _ENABLED or prefill or dtype != torch.float16 or not 1 <= rows <= ROWS
            or w.meta.get("execution_provider") != "sm70" or w.device.type != "cuda"):
        return None
    return PaddedLinears(w, rows)
