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


def workspace_bytes(c, rows, head_width, world, *, enabled=None, prefill=False, large_profile=None):
    """Exact additional persistent bytes; original logical Buffers are already charged."""
    enabled = _ENABLED if enabled is None else enabled
    if not prefill and rows > ROWS and large_profile in LARGE_PROFILES:
        maximum_input, maximum_output = large_limits(c, rows, head_width, world)
        return (ROWS * maximum_input * 2 + ROWS * maximum_output * 6 + 3 * 512
                + rows * (2 * c.low + c.streams) * 4)
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


def workspace_bytes_from_text(text, rows, world, *, enabled=None, prefill=False, large_profile=None):
    """The same forecast before weights load, including replicated local KV heads."""
    enabled = _ENABLED if enabled is None else enabled
    if prefill or not ((enabled and 1 <= rows <= ROWS)
                       or (rows > ROWS and large_profile in LARGE_PROFILES)):
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
    return workspace_bytes(c, rows, int(text["vocab_size"]) // world, world,
                           enabled=enabled, large_profile=large_profile)


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


LARGE_PROFILES = frozenset(("source-owner-target-dense-trial", "source-dense-owner-batch-prelude-trial"))
LARGE_HOST_BYTES_PER_ROLE = 64 << 10


def host_workspace_bytes(rows, profile):
    # Eight persistent Tensor headers plus bounded tile/view/tuple metadata.
    # No chunk-result list or per-matrix cache grows with calls. Native ATen
    # provider metadata remains within its existing resource contract.
    return LARGE_HOST_BYTES_PER_ROLE if rows > ROWS and profile in LARGE_PROFILES else 0


def large_limits(c, rows, head_width, world):
    maximum_input = max(c.streams * c.hidden, c.ple_dim, c.nv * 128, c.heads * c.head_dim, c.low)
    maximum_output = max(shape[-1] for shape, _size in field_layout(c, rows, head_width, world).values())
    return maximum_input, maximum_output


class TiledLinears:
    """Private large-row scratch owned by one serial Buffers writer/stream.

    Each tile is consumed/copied on the construction stream before scratch is
    reused. Final outputs remain caller-owned. `partials` has the existing
    immediate HC activation consumer; its view must not survive another HC use.
    Other streams/layouts retain the original matrix operation.
    """

    def __init__(self, w, rows):
        from .gguf_weights import GGUFDense, GGUFConcat

        self.rows, self.device = rows, w.device
        self.submitted_calls = self.submitted_rows = 0
        self.counter_overflowed = False
        self.dense_type, self.concat_type = GGUFDense, GGUFConcat
        self.maximum_input, self.maximum_output = large_limits(w.cfg, rows, w.head.n, int(w.meta.get("world", 1)))
        self.stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        self.input_owner, self.input_storage = self._aligned(ROWS * self.maximum_input, torch.float16)
        self.half_owner, self.half_storage = self._aligned(ROWS * self.maximum_output, torch.float16)
        self.float_owner, self.float_storage = self._aligned(ROWS * self.maximum_output, torch.float32)
        self.partial = {width: torch.empty((rows, width), dtype=torch.float32, device=self.device)
                        for width in (w.cfg.low, w.cfg.low + w.cfg.streams)}

    def _aligned(self, elements, dtype):
        size = 2 if dtype == torch.float16 else 4
        owner = torch.empty((elements + 512 // size,), dtype=dtype, device=self.device)
        offset = ((-owner.data_ptr()) % 512) // size
        return owner, owner[offset:offset + elements]

    def device_tensors(self):
        return (self.input_owner, self.half_owner, self.float_owner, *self.partial.values())

    def tensor(self, _name, shape, dtype, *, zero=False):
        # Logical Buffers keep their original shapes/owners; only scratch is new.
        return (torch.zeros if zero else torch.empty)(shape, dtype=dtype, device=self.device)

    def _parts(self, matrix):
        if type(matrix) is self.dense_type:
            return (matrix,) if matrix.implementation == "native32" else None
        if (type(matrix) is self.concat_type and matrix.implementation == "native32"
                and 1 <= len(matrix.parts) <= 8 and all(type(part) is self.dense_type for part in matrix.parts)):
            return matrix.parts
        return None

    def matmul(self, x, matrix, out):
        # Small-row projections preserve the existing Dense/Concat bodies.
        if type(x) is not torch.Tensor or x.ndim != 2 or not ROWS < x.shape[0] <= self.rows:
            return None
        parts = self._parts(matrix)
        if (parts is None or type(out) is not torch.Tensor or x.layout != torch.strided
                or out.layout != torch.strided or not x.is_cuda or x.device != self.device
                or x.dtype != torch.float16 or x.shape[1] != matrix.k or not 0 < matrix.k <= self.maximum_input
                or out.shape != (x.shape[0], matrix.n) or not 0 < matrix.n <= self.maximum_output
                or out.device != self.device or out.dtype not in (torch.float16, torch.float32)
                or out.stride(1) != 1 or out.stride(0) < matrix.n or x.stride(0) < 0 or x.stride(1) < 0
                or x.is_neg() or x.is_conj() or out.is_neg() or out.is_conj()
                or out.is_inference() and not torch.is_inference_mode_enabled()):
            return None
        weights = tuple(part.weight for part in parts)
        if (sum(part.n for part in parts) != matrix.n
                or any(part.implementation not in ("reference", "native32") or part.k != matrix.k
                       or not 0 < part.n <= self.maximum_output or type(weight) is not torch.Tensor or weight.layout != torch.strided
                       or weight.dtype != torch.float16 or weight.device != self.device
                       or weight.shape != (part.n, matrix.k) or not weight.is_contiguous()
                       or weight.is_neg() or weight.is_conj() for part, weight in zip(parts, weights))
                or torch.is_grad_enabled() and any(value.requires_grad for value in (x, out, *weights))
                or torch._C._overlaps(x, out) or any(torch._C._overlaps(out, weight) for weight in weights)):
            return None
        scratch = (self.input_owner, self.half_owner, self.float_owner)
        if any(torch._C._overlaps(value, owner) for value in (x, out, *weights) for owner in scratch):
            return None
        if torch.cuda.current_stream(self.device) != self.stream:
            return None
        # Preserve original part order, including which completed prefix exists
        # if a later projection fails. Each part keeps the original M32 GEMM.
        column = 0
        for part in parts:
            for start in range(0, x.shape[0], ROWS):
                count = min(ROWS, x.shape[0] - start)
                source = x[start:start + count]
                if not (count == ROWS and source.is_contiguous() and source.data_ptr() % 512 == 0):
                    padded = self.input_storage[:ROWS * matrix.k].view(ROWS, matrix.k)
                    padded[:count].copy_(source)
                    if count < ROWS:
                        padded[count:].zero_()
                    source = padded
                destination = out[start:start + count, column:column + part.n]
                direct = count == ROWS and destination.is_contiguous() and destination.data_ptr() % 512 == 0
                if direct:
                    target = destination
                else:
                    storage = self.float_storage if out.dtype == torch.float32 else self.half_storage
                    target = storage[:ROWS * part.n].view(ROWS, part.n)
                version = None if target.is_inference() else target._version
                part.matmul_prepared(source, target)
                if direct:
                    if version is not None and target._version == version:
                        torch.autograd.graph.increment_version(target)
                else:
                    destination.copy_(target[:count])
            column += part.n
        if not self.counter_overflowed:
            limit = (1 << 63) - 1
            if self.submitted_calls == limit or self.submitted_rows > limit - x.shape[0]:
                self.counter_overflowed = True
            else:
                self.submitted_calls += 1
                self.submitted_rows += x.shape[0]
        return out

    def partials(self, matrix, x):
        owner = self.partial.get(matrix.n)
        if owner is None or not ROWS < x.shape[0] <= self.rows:
            return None
        result = self.matmul(x, matrix, owner[:x.shape[0]])
        return None if result is None else result.unsqueeze(0)


def create(w, rows, *, prefill, dtype):
    if (not prefill and rows > ROWS and dtype == torch.float16
            and w.meta.get("sm70_profile") in LARGE_PROFILES
            and w.meta.get("execution_provider") == "sm70" and w.device.type == "cuda"):
        return TiledLinears(w, rows)
    if (not _ENABLED or prefill or dtype != torch.float16 or not 1 <= rows <= ROWS
            or w.meta.get("execution_provider") != "sm70" or w.device.type != "cuda"):
        return None
    return PaddedLinears(w, rows)
