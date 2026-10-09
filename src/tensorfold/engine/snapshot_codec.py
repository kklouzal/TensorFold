"""Current MLX safetensors representation, resolved once by its policy owner.

MLX 0.32.2/0.32.3's native safetensors writer defines these exact dtype codes.
Arrays keep their existing dtype; NumPy state is converted only after complete
registered payload validation, with the corresponding explicit native dtype.
This adapter imports no SDK: its owner passes the already loaded runtimes.
"""

from __future__ import annotations


# MLX's native save/load contract includes C64; F64 is not supported by its
# safetensors writer. This mirrors that representation, not an LLM dtype floor.
DTYPES = (
    ("bool_", "BOOL", 1),
    ("int8", "I8", 1),
    ("uint8", "U8", 1),
    ("int16", "I16", 2),
    ("uint16", "U16", 2),
    ("int32", "I32", 4),
    ("uint32", "U32", 4),
    ("int64", "I64", 8),
    ("uint64", "U64", 8),
    ("float16", "F16", 2),
    ("bfloat16", "BF16", 2),
    ("float32", "F32", 4),
    ("complex64", "C64", 8),
)
SIZES = {code: size for _, code, size in DTYPES}


class Codec:
    """Operation/startup-owned runtime types, without global mutable lookup."""

    def __init__(self, runtime, numpy):
        self.runtime, self.numpy = runtime, numpy
        self.native = {getattr(runtime, name): code for name, code, _ in DTYPES}
        self.native_codes = {code: dtype for dtype, code in self.native.items()}
        self.host = {numpy.dtype(name): code for name, code, _ in DTYPES if name != "bfloat16"}

    def kind(self, value):
        if isinstance(value, self.runtime.array):
            return "array"
        if isinstance(value, self.numpy.ndarray):
            return "numpy"
        return None

    def describe(self, value):
        kind = self.kind(value)
        if kind is None:
            return None
        mapping = self.native if kind == "array" else self.host
        code = mapping.get(value.dtype)
        if code is None:
            raise ValueError("cache dtype is unsupported by the current MLX safetensors representation")
        return {"dtype": code, "shape": list(value.shape)}

    def native_array(self, value):
        code = self.describe(value)["dtype"]
        converted = self.runtime.array(value, dtype=self.native_codes[code])
        if self.describe(converted) != self.describe(value):
            raise ValueError("host cache conversion changed dtype or geometry")
        return converted

    def host_array(self, value):
        converted = self.numpy.array(value)
        if self.describe(converted) != self.describe(value):
            raise ValueError("restored host cache conversion changed dtype or geometry")
        return converted

    def load(self, path):
        arrays, metadata = self.runtime.load(str(path), return_metadata=True)
        # Complete lazy reads while the immutable private file remains owned.
        self.runtime.eval(list(arrays.values()))
        return arrays, metadata

    def write(self, path, payload):
        arrays = {
            name: self.native_array(value) if name in payload.numpy_keys else value
            for name, value in payload.arrays.items()
        }
        self.runtime.save_safetensors(str(path), arrays, metadata=payload.metadata)
