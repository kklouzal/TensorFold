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


HC_PROFILE = "source-batch-router-blas-trial"
HC_HOST_BYTES = 64 << 10


def mtp_hc_workspace_bytes(text, rows, profile):
    if profile != HC_PROFILE or not 0 < rows <= 72:
        return 0
    text = text.get("text_config") or text
    hidden, streams = int(text["hidden_size"]), int(text.get("hc_count", 4))
    return ROWS * hidden * 8 + 3 * 512 if rows * streams > ROWS else 0


def mtp_hc_host_bytes(text, rows, profile):
    return HC_HOST_BYTES if mtp_hc_workspace_bytes(text, rows, profile) else 0


class MTPHCTiles:
    """Named MTP fc_h only; original M32 tile math and caller-owned output.

    This does not adopt generic large-row Dense/Concat eligibility. Its three
    fixed scratch owners serve one serialized construction stream; other
    weights, small physical rows, prefill and foreign streams retain reference.
    """
    def __init__(self, w, rows):
        from .gguf_weights import GGUFDense

        self.rows, self.device = rows, w.device
        self.hc_projection = w.mtp.fc_h
        self.hc_streams, self.hc_hidden = w.cfg.streams, w.cfg.hidden
        if (type(self.hc_projection) is not GGUFDense or not 0 < rows <= 72
                or not ROWS < rows * self.hc_streams or self.hc_projection.k != self.hc_hidden
                or self.hc_projection.n != self.hc_hidden):
            raise ValueError("MTP HC owner requires its actual named hidden projection and complete row span")
        self.maximum_input = self.maximum_output = self.hc_hidden
        self.stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        self.input_owner, self.input_storage = self._aligned(ROWS * self.maximum_input, torch.float16)
        self.half_owner, self.half_storage = self._aligned(ROWS * self.maximum_output, torch.float16)
        self.float_owner, self.float_storage = self._aligned(ROWS * self.maximum_output, torch.float32)
        self.partial = {}
        self.submitted_calls = self.submitted_rows = 0
        self.counter_overflowed = False

    def _parts(self, matrix):
        if matrix is self.hc_projection and matrix.implementation == "native32":
            return (matrix,)
        return None

    def partials(self, _matrix, _x):
        return None  # No generic HC low-rank path or partial scratch is added.


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

    def _hc_rows_allowed(self, rows, matrix):
            return (matrix is self.hc_projection and self.hc_projection is not None
                    and ROWS < rows <= self.rows * self.hc_streams and rows % self.hc_streams == 0
                    and matrix.k == self.hc_hidden and matrix.n == self.hc_hidden)

    def matmul(self, x, matrix, out):
            # Small-row projections preserve the existing Dense/Concat bodies.
            if type(x) is not torch.Tensor or x.ndim != 2 or not ROWS < x.shape[0]:
                return None
            if not self._hc_rows_allowed(x.shape[0], matrix):
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

def create(w, rows, *, prefill, dtype, role=None):
    if (role == "mtp" and not prefill and dtype == torch.float16 and 0 < rows <= 72
            and w.meta.get("sm70_profile") == HC_PROFILE and w.meta.get("execution_provider") == "sm70"
            and w.device.type == "cuda" and rows * w.cfg.streams > ROWS):
        from .gguf_weights import GGUFDense

        projection = getattr(getattr(w, "mtp", None), "fc_h", None)
        if (type(projection) is GGUFDense and projection.implementation == "native32"
                and projection.k == w.cfg.hidden and projection.n == w.cfg.hidden):
            return MTPHCTiles(w, rows)
    if (not _ENABLED or prefill or dtype != torch.float16 or not 1 <= rows <= ROWS
            or w.meta.get("execution_provider") != "sm70" or w.device.type != "cuda"):
        return None
    return PaddedLinears(w, rows)
