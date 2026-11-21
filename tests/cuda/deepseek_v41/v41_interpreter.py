"""Triton-interpreter (CPU) harness fixes shared by every DeepSeek-V4.1 test entry point (conftest and the MU
check scripts). They correct gaps of the Triton 3.2 interpreter only; compiled kernels never see them.
"""

import os


def install():
    """Harness-only fixes; kernels are unchanged. NumPy >= 2.5 refuses int() of a one-element array, which the
    Triton interpreter uses for loaded scalars (range(.., loaded)): same value via a 0-d read."""
    if os.environ.get("TRITON_INTERPRET", "0") != "1":
        return
    from triton.runtime import interpreter

    if getattr(interpreter, "_tf_v41_index_fix", False):
        return
    original = interpreter._patch_lang_tensor

    def index(self):
        return int(self.handle.data.reshape(-1)[0])

    def patched(tensor, *scope):
        # Triton 3.2: _patch_lang_tensor(tensor) assigns attributes; 3.6: (tensor, scope) with scope.set_attr.
        original(tensor, *scope)
        if scope:
            scope[0].set_attr(tensor, "__index__", index)
        else:
            tensor.__index__ = index

    interpreter._patch_lang_tensor = patched

    # Triton 3.2 interpreter: np.sum / np.cumsum promote int32 data to int64 while the handle keeps the int32
    # Triton type; the native masked store then writes 8 bytes per int32 lane (clobbering the next element).
    # Keep integer reductions in their own width, which is also the compiled kernels' arithmetic.
    import numpy as np

    def _same_width(data):
        return {"dtype": data.dtype} if np.issubdtype(data.dtype, np.integer) else {}

    def _sum(self, input):
        data = input.handle.data
        return self.to_tensor(np.sum(data, axis=self.axis, keepdims=self.keep_dims, **_same_width(data)), input.dtype)

    interpreter.ReduceOps.sum = _sum
    _cumsum = interpreter.ScanOps.cumsum if hasattr(interpreter.ScanOps, "cumsum") else None
    if _cumsum is not None:

        def cumsum(self, input):
            data = input.handle.data
            return [self.to_tensor(np.cumsum(data, axis=self.axis, **_same_width(data)), dtype=input.dtype)]

        interpreter.ScanOps.cumsum = cumsum
    # Triton 3.2 interpreter: tl.full(..., tl.int1) needs the builder's int1 type, which only the compiled
    # builder provides (the interpreter already maps tl.int1 to NumPy bool).
    if not hasattr(interpreter.InterpreterBuilder, "get_int1_ty"):
        interpreter.InterpreterBuilder.get_int1_ty = lambda self: interpreter.tl.int1
    # ... and `~mask` on int1 needs an int1 all-ones value, missing from the 3.2 interpreter builder.
    _all_ones = interpreter.InterpreterBuilder.get_all_ones_value

    def get_all_ones_value(self, type):
        if type == interpreter.tl.int1:
            return interpreter.TensorHandle(np.array([True], dtype=np.bool_), interpreter.tl.int1)
        return _all_ones(self, type)

    interpreter.InterpreterBuilder.get_all_ones_value = get_all_ones_value
    interpreter._tf_v41_index_fix = True
