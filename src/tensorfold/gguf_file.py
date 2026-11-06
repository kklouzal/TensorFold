"""Bounded backend-neutral GGUF inventory and packed-byte addressing.

Container admission does not establish model-family or execution support.
GGUF/ggml format: https://github.com/ggml-org/ggml (MIT).
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import stat
import struct
from types import MappingProxyType

FORMATS = MappingProxyType({
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    18: ("IQ3_XXS", 256, 98), 20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136), 30: ("BF16", 1, 2), 42: ("Q2_0", 64, 18),
})


_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


_SPLIT = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf")


_HEADER_LIMIT = 64 << 20


class _Header:
    def __init__(self, stream):
        self.stream = stream
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("GGUF inputs must be regular files")
        self.size = info.st_size

    def read(self, length):
        if length < 0 or self.stream.tell() + length > min(self.size, _HEADER_LIMIT):
            raise ValueError("truncated or oversized GGUF header")
        data = self.stream.read(length)
        if len(data) != length:
            raise ValueError("short GGUF header read")
        return data

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self):
        return self.read(self.number("Q")).decode("utf-8")

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 9:
            element, count = self.number("I"), self.number("Q")
            if element == 9 or element not in (*_SCALARS, 8) or count > 1_000_000:
                raise ValueError("unsupported or oversized GGUF metadata array")
            # Check the minimum encoded span before allocating the Python list.
            minimum = 8 if element == 8 else struct.calcsize("<" + _SCALARS[element])
            if count * minimum > min(self.size, _HEADER_LIMIT) - self.stream.tell():
                raise ValueError("truncated GGUF metadata array")
            return tuple(self.value(element) for _ in range(count))
        if kind not in _SCALARS:
            raise ValueError(f"unknown GGUF metadata type {kind}")
        return self.number(_SCALARS[kind])


@dataclass(frozen=True)
class PackedTensor:
    name: str
    shape: tuple[int, ...]  # NumPy/Torch order; GGUF stores dimensions reversed.
    ggml_type: int
    path: Path
    offset: int
    nbytes: int

    @property
    def encoding(self):
        return FORMATS[self.ggml_type][0]

    @property
    def rows(self):
        return math.prod(self.shape[:-1])

    @property
    def row_bytes(self):
        _, block, size = FORMATS[self.ggml_type]
        return self.shape[-1] // block * size

    def row_span(self, start, count):
        if (type(start) is not int or type(count) is not int
                or start < 0 or count < 0 or start > self.rows or count > self.rows - start):
            raise ValueError("packed GGUF row range is outside the tensor")
        return self.offset + start * self.row_bytes, count * self.row_bytes

    def column_span(self, start, count):
        """A per-row byte slice only if both logical boundaries preserve blocks.

        TP8's 640/8=80 is deliberately rejected for block32/block256 storage;
        the later loader must decode/repack or choose another decomposition.
        """
        _, block, size = FORMATS[self.ggml_type]
        if (type(start) is not int or type(count) is not int or start < 0 or count <= 0
                or start + count > self.shape[-1] or start % block or count % block):
            raise ValueError(f"{self.encoding} column shard must preserve complete {block}-value blocks")
        return start // block * size, count // block * size

    def read_rows(self, reader, start, count, *, device="cpu", max_bytes=64 << 20):
        """Borrow the supplied owned byte reader; never read a whole expert/PLE bank implicitly."""
        offset, length = self.row_span(start, count)
        if type(max_bytes) is not int or max_bytes <= 0 or length > max_bytes:
            raise ValueError("packed GGUF row read exceeds its staging budget")
        return reader.read(self.path, offset, length, device).reshape(count, self.row_bytes)


def _shard(path):
    # Reject a FIFO/device by its opened descriptor without first blocking on it.
    with open(path, "rb", opener=lambda name, flags: os.open(name, flags | getattr(os, "O_NONBLOCK", 0))) as stream:
        header = _Header(stream)
        if header.read(4) != b"GGUF" or header.number("I") != 3:
            raise ValueError("GGUF inventory requires little-endian GGUF version 3")
        tensors, fields = header.number("Q"), header.number("Q")
        if tensors > 100_000 or fields > 100_000:
            raise ValueError("GGUF directory count exceeds its bound")
        metadata = {}
        for _ in range(fields):
            key = header.string()
            if key in metadata:
                raise ValueError(f"duplicate GGUF metadata key {key}")
            metadata[key] = header.value(header.number("I"))
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
            raise ValueError("GGUF alignment must be a power of two through 4096")
        entries = []
        for _ in range(tensors):
            name, dims = header.string(), header.number("I")
            if not name or not 1 <= dims <= 4:
                raise ValueError("GGUF tensor must have a name and 1 to 4 dimensions")
            shape = tuple(reversed([header.number("Q") for _ in range(dims)]))
            kind, offset = header.number("I"), header.number("Q")
            if kind not in FORMATS:
                raise ValueError(f"{name}: unsupported ggml tensor type {kind}")
            _, block, size = FORMATS[kind]
            if any(d <= 0 for d in shape) or shape[-1] % block or offset % alignment:
                raise ValueError(f"{name}: invalid shape, quantization block, or aligned offset")
            entries.append((name, shape, kind, offset, math.prod(shape) // block * size))
        data = (stream.tell() + alignment - 1) // alignment * alignment
        result, end = [], data
        for name, shape, kind, offset, length in sorted(entries, key=lambda t: t[3]):
            begin = data + offset
            if begin < end or begin > header.size or length > header.size - begin:
                raise ValueError(f"{name}: overlapping or truncated GGUF tensor payload")
            result.append(PackedTensor(name, shape, kind, path, begin, length))
            end = begin + length
        return metadata, result


@dataclass(frozen=True)
class GGUFCheckpoint:
    metadata: object
    tensors: object
    paths: tuple[Path, ...]

    @classmethod
    def open(cls, entrypoint):
        path = Path(entrypoint).resolve(strict=True)
        match = _SPLIT.fullmatch(path.name)
        if match:
            stem, number, count = match.groups()
            count = int(count)
            if not 1 <= int(number) <= count <= 1024:
                raise ValueError("invalid GGUF split filename")
            paths = tuple(path.with_name(f"{stem}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1))
        else:
            paths = (path,)
        metadata, tensors = {}, {}
        for index, shard in enumerate(paths):
            fields, entries = _shard(shard)
            if len(paths) > 1 and (fields.get("split.no") != index or fields.get("split.count") != len(paths)):
                raise ValueError("GGUF split metadata does not match the complete ordered shard set")
            for key, value in fields.items():
                if key.startswith("split."):
                    if key == "split.tensors.count" and key in metadata and metadata[key] != value:
                        raise ValueError("GGUF shards disagree about total tensor count")
                    if key != "split.tensors.count":
                        continue
                if key in metadata and metadata[key] != value:
                    raise ValueError(f"GGUF shards disagree about {key}")
                metadata[key] = value
            for tensor in entries:
                if tensor.name in tensors:
                    raise ValueError(f"duplicate GGUF tensor {tensor.name}")
                tensors[tensor.name] = tensor
        if metadata.get("split.tensors.count", len(tensors)) != len(tensors) or not tensors:
            raise ValueError("GGUF checkpoint tensor count is incomplete")
        return cls(MappingProxyType(metadata), MappingProxyType(tensors), paths)


def resolve_gguf(path):
    """Resolve a local GGUF file or an unambiguous split checkpoint directory.

    Return None for ordinary checkpoint directories, so existing family discovery
    and safetensors routes keep their established behavior.
    """
    path = Path(path)
    if path.is_file():
        return path.resolve() if path.suffix.lower() == ".gguf" else None
    if not path.is_dir():
        return None
    config = path / "config.json"
    if config.is_file():
        from tensorfold.tensor_file import read_metadata_json
        raw = read_metadata_json(config)
        selected = raw.get("gguf_file") or raw.get("gguf_entrypoint")
        if selected:
            target = Path(selected)
            if not target.is_absolute():
                target = path / target
            if not target.is_file() or target.suffix.lower() != ".gguf":
                raise ValueError("GGUF sidecar entrypoint must name an existing GGUF file")
            return target.resolve()
    candidates = sorted(path.glob("*.gguf"))
    entries = [p for p in candidates if not _SPLIT.fullmatch(p.name)
               or int(_SPLIT.fullmatch(p.name).group(2)) == 1]
    if len(entries) > 1:
        raise ValueError("GGUF directory contains multiple checkpoints; name one entrypoint file")
    return entries[0].resolve() if entries else None
