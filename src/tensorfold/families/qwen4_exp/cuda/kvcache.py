"""Flash Next's CUDA key/value caches: bf16, or ExLlamaV3's -cq 8 / -cq 4 codes (H32-rotated groups of 32, fp16 absmax scales, midpoint grid) kept rotated."""

from __future__ import annotations

import math

import torch

from ..kv_formats import BITS_OF as BITS_OF, DTYPES as DTYPES, KVPairFormat, get, get_pair

GROUP = 32                      # values per scale (ExLlamaV3's cache-quant group)
SCALE_DTYPE = torch.float16     # ExLlamaV3 stores the group absmax as a half (__float2half_rn)
R32 = 1.0 / math.sqrt(32)


def check(dtype: str) -> str:
    """Refuse a cache dtype this engine has no storage for."""

    return get(dtype).name


def _select_pair(dtype: str, pair: KVPairFormat | None, key_dtype: str | None,
                 value_dtype: str | None) -> KVPairFormat:
    base = get(dtype)
    if pair is None:
        return get_pair(base.name, key_dtype, value_dtype)
    if type(pair) is not KVPairFormat:
        raise ValueError("pair must be a registered KVPairFormat")
    # The default positional shorthand remains backward compatible. An explicit
    # owned plan has no second set of side overrides or conflicting shorthand.
    if key_dtype is not None or value_dtype is not None:
        raise ValueError("explicit KV pair cannot be combined with side dtype overrides")
    if base.name != "bf16" and pair != get_pair(base.name):
        raise ValueError("KV dtype shorthand conflicts with the explicit pair")
    return pair


def row_bytes(kv_heads: int, head_dim: int, dtype: str = "bf16", *, pair: KVPairFormat | None = None,
              key_dtype: str | None = None, value_dtype: str | None = None) -> int:
    """Per-position K+V bytes; BF16 dummy scales are once per cache, not per row."""

    return _select_pair(dtype, pair, key_dtype, value_dtype).row_bytes(kv_heads, head_dim)


# -- storage -------------------------------------------------------------------------------------------
class KVCache:
    """Owned K/V storage with one immutable ordered format/shape plan.

    Each side has its own payload type, packed width, scale group/type and BF16
    dummy. Payload and metadata contents are mutable; formats and dimensions
    require a new cache. Legacy one-format accessors refuse asymmetric plans.
    """

    __slots__ = ("_pair", "_capacity", "_kv_heads", "_head_dim", "k", "v", "ks", "vs")

    def __init__(self, capacity: int, kv_heads: int, head_dim: int, device, dtype: str = "bf16", *,
                 pair: KVPairFormat | None = None, key_dtype: str | None = None,
                 value_dtype: str | None = None) -> None:
        selected = _select_pair(dtype, pair, key_dtype, value_dtype)
        selected.nbytes(capacity, kv_heads, head_dim)  # Validate both sides before the first allocation.
        self._pair = selected
        self._capacity, self._kv_heads, self._head_dim = capacity, kv_heads, head_dim
        self.k, self.ks = self._side(selected.key, device)
        self.v, self.vs = self._side(selected.value, device)

    def _side(self, format, device):
        width = self.head_dim if not format.quantized else self.head_dim * format.bits // 8
        shape = (self.capacity, self.kv_heads, width)
        groups = ((self.capacity, self.kv_heads, self.head_dim // format.group)
                  if format.quantized else (1,))
        return (torch.zeros(shape, dtype=getattr(torch, format.payload_dtype), device=device),
                torch.zeros(groups, dtype=getattr(torch, format.scale_dtype), device=device))

    @property
    def pair(self) -> KVPairFormat:
        return self._pair

    @property
    def key_format(self):
        return self.pair.key

    @property
    def value_format(self):
        return self.pair.value

    @property
    def key_dtype(self) -> str:
        return self.pair.key_dtype

    @property
    def value_dtype(self) -> str:
        return self.pair.value_dtype

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def kv_heads(self) -> int:
        return self._kv_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def format(self):
        if not self.pair.symmetric:
            raise ValueError("mixed KV cache has no single format; use key_format/value_format")
        return self.key_format

    @property
    def dtype(self) -> str:
        return self.format.name

    @property
    def bits(self) -> int:
        return self.format.bits

    @property
    def codec(self) -> int:
        return self.format.codec

    @property
    def quantized(self) -> bool:
        """Whether either side is quantized; per-side work uses its own format."""
        return self.key_format.quantized or self.value_format.quantized

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes + self.ks.nbytes + self.vs.nbytes

    def resized(self, capacity: int, keep: int) -> "KVCache":
        """Independent copy with the same pair, including both dummy scales.

        Nonnegative integer ``keep`` is bounded by old/new capacity, preserving
        existing truncation semantics. New rows retain their zero initialization.
        """

        if type(keep) is not int or keep < 0:
            raise ValueError("KV rows to keep must be a nonnegative integer")
        self.pair.nbytes(capacity, self.kv_heads, self.head_dim)
        other = KVCache(capacity, self.kv_heads, self.head_dim, self.k.device, pair=self.pair)
        keep = min(keep, self.capacity, capacity)
        other.k[:keep].copy_(self.k[:keep])
        other.v[:keep].copy_(self.v[:keep])
        for original, copied, format in ((self.ks, other.ks, self.key_format),
                                          (self.vs, other.vs, self.value_format)):
            if format.quantized:
                copied[:keep].copy_(original[:keep])
            else:
                copied.copy_(original)
        return other

    def clone(self) -> "KVCache":
        # The validated immutable pair/shape is shared; all four tensors own
        # fresh initialized storage before the clone can become observable.
        other = object.__new__(KVCache)
        other._pair = self.pair
        other._capacity, other._kv_heads, other._head_dim = self.capacity, self.kv_heads, self.head_dim
        other.k, other.v = self.k.clone(), self.v.clone()
        other.ks, other.vs = self.ks.clone(), self.vs.clone()
        return other


# -- the reference quantizer (tests, and what the kernels are checked against) --------------------------
def h32_ref(x: torch.Tensor) -> torch.Tensor:
    """H32 over the last axis of a (..., 32k) fp32 tensor, in the operations ``h32`` runs: -> (..., k, 32)."""

    x = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    for lo in (1, 2, 4, 8, 16):
        hi = 32 // (2 * lo)
        t = x.reshape(*x.shape[:-1], hi, 2, lo)
        a, b = t[..., 0, :], t[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape[:-1], 32)
    return x * R32


def pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    """Unsigned codes (..., even) in 0..15 -> uint8, low nibble = even index, high nibble = odd index."""

    pair = q.to(torch.int32).reshape(*q.shape[:-1], q.shape[-1] // 2, 2)
    return (pair[..., 0] | (pair[..., 1] << 4)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """The inverse of ``pack_nibbles``: uint8 -> int32 codes, low nibble first."""

    p = packed.to(torch.int32)
    lo, hi = p & 15, (p >> 4) & 15
    return torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def quantize_ref(k: torch.Tensor, v: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, ...]:
    """bf16 or fp32 ``[N, HK, D]`` keys and values -> (k codes, k scales, v codes, v scales) on ExLlamaV3's midpoint grid in fp32: 8 bits store q - 128, 4 bits two codes a byte."""

    if bits not in (8, 4):
        raise ValueError(f"quantize_ref bits must be 8 or 4, not {bits}")
    m = 1 << (bits - 1)
    qmax = float((1 << bits) - 1)
    out: list[torch.Tensor] = []
    for x in (k, v):
        rot = h32_ref(x.float())
        s = rot.abs().amax(dim=-1) + 1e-10
        q = torch.floor(rot * (1.0 / s)[..., None] * m) + m
        q = q.clamp(0.0, qmax)
        if bits == 8:
            out.append((q - 128.0).to(torch.int8).reshape(x.shape))
        else:
            out.append(pack_nibbles(q).reshape(*x.shape[:-1], x.shape[-1] // 2))
        out.append(s.to(SCALE_DTYPE))
    return out[0], out[1], out[2], out[3]


def dequant_ref(code: torch.Tensor, scale: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Codes and fp16 scales -> bf16 still in the cache's rotation: (code + 0.5) * s / 128 at 8 bits, (q - 7.5) * s / 8 at 4."""

    if bits == 8:
        width = code.shape[-1]
        c = code.float().reshape(*code.shape[:-1], width // GROUP, GROUP)
        s = scale.float().reshape(*scale.shape, 1)
        return ((c + 0.5) * s * 0.0078125).to(torch.bfloat16).reshape(code.shape)
    q = unpack_nibbles(code).float()
    width = q.shape[-1]
    c = q.reshape(*q.shape[:-1], width // GROUP, GROUP)
    s = scale.float().reshape(*scale.shape, 1)
    return ((c - 7.5) * s * 0.125).to(torch.bfloat16).reshape(q.shape)
