"""Flash Next cache formats shared by configuration, allocation and admission.

The descriptor fixes storage and rotation meaning; equal bit counts do not imply
compatible cache bytes. This module imports no accelerator runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType

from .cuda.rotorquant_ref import FormatDescriptor, descriptor, variant_id


@dataclass(frozen=True, slots=True)
class KVFormat:
    name: str
    bits: int
    codec: int
    group: int
    scale_bytes: int
    rotor: FormatDescriptor | None = None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or any(type(value) is not int for value in
                                                                 (self.bits, self.codec, self.group, self.scale_bytes)):
            raise ValueError("KV format needs a name and exact integer storage fields")
        if self.bits not in (3, 4, 6, 7, 8, 16) or self.codec < 0 or self.group <= 0 or self.scale_bytes <= 0:
            raise ValueError("KV format has invalid storage fields")
        if self.rotor is not None and (type(self.rotor) is not FormatDescriptor or self.rotor.bits != self.bits
                                      or self.rotor.group != self.group or self.rotor.scale_dtype != "float32"
                                      or self.scale_bytes != 4 or not self.codec):
            raise ValueError("Rotor KV format differs from its pinned descriptor")
        if bool(self.codec) != (self.rotor is not None):
            raise ValueError("KV codec and Rotor descriptor must agree")
        if self.rotor is not None and self.codec != variant_id(self.rotor.variant):
            raise ValueError("KV codec differs from its Rotor transform/norm policy")

    @property
    def payload_dtype(self) -> str:
        return "bfloat16" if self.bits == 16 else "int8" if self.name == "int8" else "uint8"

    @property
    def scale_dtype(self) -> str:
        return "float32" if self.rotor is not None else "float16"

    @property
    def quantized(self) -> bool:
        return self.bits != 16

    @property
    def dummy_bytes(self) -> int:
        return 0 if self.quantized else 2

    @property
    def storage_uid(self) -> str:
        if self.rotor is not None:
            return self.rotor.codec_id
        return f"kv-native-v1-{self.name}-h32-midpoint-fp16-absmax" if self.quantized else "kv-native-v1-bf16"

    def value_bytes(self, head_dim: int) -> int:
        """One position/head's K or V payload and metadata; reject partial groups."""

        if type(head_dim) is not int or head_dim <= 0:
            raise ValueError(f"{self.name} KV cache needs a positive integer head dim, not {head_dim!r}")
        if self.bits == 16:
            return head_dim * 2
        if head_dim % self.group:
            raise ValueError(f"{self.name} KV cache needs a positive head dim that is a multiple of "
                             f"{self.group}, not {head_dim}")
        if self.codec and head_dim & (head_dim - 1):
            raise ValueError(f"{self.name} KV cache needs a power-of-two head dim, not {head_dim}")
        return head_dim * self.bits // 8 + head_dim // self.group * self.scale_bytes

    @property
    def handshake(self) -> int:
        """Legacy protocol values stay fixed; new formats identify all pinned tables."""

        if self.rotor is None:
            return self.bits
        return int.from_bytes(hashlib.sha256(self.rotor.codec_id.encode("ascii")).digest()[:8], "big") & (2**63 - 1)


FORMATS = (
    KVFormat("bf16", 16, 0, 32, 2),
    KVFormat("int8", 8, 0, 32, 2),
    KVFormat("int4", 4, 0, 32, 2),
    # Users select their quality/compression tradeoff; distinct norm/basis
    # descriptors remain distinct even when their payload byte counts match.
    KVFormat("rotorquant-planar4", 4, 1, 128, 4, descriptor(4, "planar")),
    KVFormat("rotorquant-iso4", 4, 2, 128, 4, descriptor(4, "isofast")),
    KVFormat("rotorquant-iso64-norm4", 4, 3, 128, 4, descriptor(4, "iso64-norm")),
    KVFormat("rotorquant-signed128-4", 4, 4, 128, 4, descriptor(4, "signed-iso128")),
    KVFormat("rotorquant-signed64-norm4", 4, 5, 128, 4, descriptor(4, "signed-iso64-norm")),
    KVFormat("rotorquant6", 6, 4, 128, 4, descriptor(6, "signed-iso128")),
    KVFormat("rotorquant6-norm", 6, 6, 128, 4, descriptor(6, "signed-iso128-norm")),
    KVFormat("rotorquant6-outlier-norm", 6, 7, 128, 4, descriptor(6, "signed-iso128-outlier-norm")),
    KVFormat("rotorquant7", 7, 4, 128, 4, descriptor(7, "signed-iso128")),
    KVFormat("rotorquant8", 8, 4, 128, 4, descriptor(8, "signed-iso128")),
    KVFormat("rotorquant8-norm", 8, 8, 128, 4, descriptor(8, "signed-iso128-norm8")),
    KVFormat("rotorquant-planar3", 3, 1, 128, 4, descriptor(3, "planar")),
    KVFormat("rotorquant-iso3", 3, 2, 128, 4, descriptor(3, "isofast")),
)
DTYPES = tuple(item.name for item in FORMATS)
BITS_OF = MappingProxyType({item.name: item.bits for item in FORMATS})
_BY_NAME = MappingProxyType({item.name: item for item in FORMATS})


def get(dtype: str) -> KVFormat:
    """Configuration boundary: no unknown format silently becomes a legacy dtype."""

    if type(dtype) is not str:
        raise ValueError(f"kv-dtype {dtype!r}: expected one of {' or '.join(DTYPES)}")
    try:
        return _BY_NAME[dtype]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"kv-dtype {dtype!r}: this engine serves {' or '.join(DTYPES)}") from exc


@dataclass(frozen=True, slots=True)
class KVPairFormat:
    """Ordered storage authority; execution arithmetic is a separate plan.

    Both sides must be registered immutable descriptors. Payload, scales and
    dummy metadata are independent. A new pair requires new cache/state/graphs;
    swapping two equal-width sides is not a compatible copy or peer handshake.
    """

    key: KVFormat
    value: KVFormat
    _identity: str = field(init=False, repr=False, compare=False)
    _handshake: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for side in (self.key, self.value):
            if type(side) is not KVFormat or get(side.name) != side:
                raise ValueError("KV pair must own registered, unchanged side descriptors")
        header = json.dumps(["tensorfold-kv-pair-v1", self.key.storage_uid, self.value.storage_uid],
                            separators=(",", ":")).encode("ascii")
        identity = "kv-pair-v1-" + hashlib.sha256(header).hexdigest()
        handshake = (self.key.handshake if self.symmetric else
                     int.from_bytes(hashlib.sha256(identity.encode("ascii")).digest()[:8], "big") & (2**63 - 1))
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_handshake", handshake)

    @property
    def key_dtype(self) -> str:
        return self.key.name

    @property
    def value_dtype(self) -> str:
        return self.value.name

    @property
    def symmetric(self) -> bool:
        return self.key == self.value

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def storage_uid(self) -> str:
        return self.identity

    @property
    def handshake(self) -> int:
        return self._handshake

    @property
    def dummy_bytes(self) -> int:
        return self.key.dummy_bytes + self.value.dummy_bytes

    def row_bytes(self, kv_heads: int, head_dim: int) -> int:
        if type(kv_heads) is not int or kv_heads <= 0:
            raise ValueError("KV pair needs a positive integer KV head count")
        return kv_heads * (self.key.value_bytes(head_dim) + self.value.value_bytes(head_dim))

    def nbytes(self, capacity: int, kv_heads: int, head_dim: int) -> int:
        if type(capacity) is not int or capacity < 0:
            raise ValueError("KV pair capacity must be a nonnegative integer")
        return capacity * self.row_bytes(kv_heads, head_dim) + self.dummy_bytes


def get_pair(shorthand: str = "bf16", key_override: str | None = None,
             value_override: str | None = None) -> KVPairFormat:
    """Each explicit side overrides the shorthand; only None means inherit."""

    base = get(shorthand)
    key = base if key_override is None else get(key_override)
    value = base if value_override is None else get(value_override)
    return KVPairFormat(key, value)
