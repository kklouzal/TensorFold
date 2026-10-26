"""Actual tensor views and immutable descriptors for complete retained state.

These parts describe required payload, never a second model or snapshot store.
The common catalog owns transfer/generation publication. Method adapters and
the State adapter use the same descriptor and actual destination-view contract.
"""

from dataclasses import asdict, dataclass
import json
import sys


_I64 = (1 << 63) - 1
_NAME_BYTES = max(map(len, (f"history.kv.{_I64}.{_I64}.{_I64}.vs",
    f"recovery.{_I64}.indexer.{_I64}.coordinates", f"state.indexer.{_I64}.proposal_coordinates",
    f"method.recovery.{_I64}.mtp.coordinates")))
_DTYPES = frozenset(("float16", "bfloat16", "float32", "int64", "int32", "uint8", "int8", "bool"))


@dataclass(frozen=True)
class PartSpec:
    name: str
    category: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    semantic: tuple


@dataclass(frozen=True)
class TensorPart:
    name: str
    tensor: object
    category: str = "method"
    semantic: tuple = ()

    def spec(self):
        import torch

        if (type(self.name) is not str or not self.name or self.category not in {"history", "state", "recovery", "method"}
                or type(self.semantic) is not tuple or not isinstance(self.tensor, torch.Tensor)
                or self.tensor.requires_grad or not self.tensor.is_contiguous()
                or not self.name.isascii() or len(self.name) > _NAME_BYTES
                or self.tensor.ndim > 5 or len(self.semantic) > 7
                or self.tensor.nbytes > sys.maxsize or any(dim > _I64 for dim in self.tensor.shape)
                or str(self.tensor.dtype).removeprefix("torch.") not in _DTYPES):
            raise ValueError("retained parts require named contiguous owned tensor views and semantic descriptors")
        if any(type(value) not in (str, int, bool, type(None)) for value in self.semantic):
            raise ValueError("retained tensor semantics must be canonical scalar values")
        if any(type(value) is str and (not value.isascii() or len(value) > 64)
               or type(value) is int and not -_I64 - 1 <= value <= _I64 for value in self.semantic):
            raise ValueError("retained component semantic exceeds its bounded family wire schema")
        return PartSpec(self.name, self.category, str(self.tensor.dtype).removeprefix("torch."),
                        tuple(self.tensor.shape), self.tensor.numel() * self.tensor.element_size(), self.semantic)


def part_specs(parts):
    specs = tuple(part.spec() for part in parts)
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("complete retained inventory repeats a tensor component name")
    return specs


def manifest_bound(component_count):
    """Encoded control payload bound from the actual family part wire schema.

    All current State/method sources have at most five axes and seven scalar
    semantic fields. The longest generated names use at most three signed64
    indices; positional identities are SHA256. These shipping checks prevent a
    future source from silently invalidating the admitted descriptor bound.
    Python allocator/graph overhead is separately qualified host residency.
    """
    if type(component_count) is not int or component_count < 0:
        raise ValueError("retained manifest component bound is invalid")
    spec = PartSpec("n" * _NAME_BYTES, "recovery", max(_DTYPES, key=len), (_I64,) * 5, sys.maxsize, ("f" * 64,) * 7)
    object_record = {"identity": "f" * 64, "checksum": "f" * 64, "bytes": sys.maxsize}
    def encoded(value):
        return len(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    # Three parallel arrays: descriptors, immutable object records, and keys.
    return (16 << 20) + component_count * (encoded(asdict(spec)) + encoded(object_record) + encoded("f" * 64) + 3)
