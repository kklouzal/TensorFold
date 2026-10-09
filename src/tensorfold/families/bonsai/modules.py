"""Bonsai's layers: projections that rotate their rows first, the rotated embedding and the unrotated fp32 gates."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.kernels.qwen.prism.v1 import rotate


def rotated(x: mx.array, signs: mx.array) -> mx.array:
    """x in the rotated basis, observing current input and signs values."""

    return rotate.rotate_rows(x, signs)


class RotationCache:
    """A sibling transform identity; derived arrays live only in a forward operation.

    Public MLX arrays may overwrite their descriptors between calls. Generic
    calls therefore compute current values. A synchronous projection operation
    may reuse a result while its owner borrows unchanged input/signs; the scope
    bounds entries and retires references before another operation starts.
    """

    __slots__ = ()

    def __call__(self, x: mx.array, signs: mx.array) -> mx.array:
        from tensorfold.kernels.qwen.dense.v1 import projection_operation

        hit = projection_operation.rotation_of(self, x, signs)
        if hit is not None:
            return hit
        y = rotate.rotate_rows(x, signs)
        projection_operation.remember_rotation(self, x, signs, y)
        return y


class RotatedLinear(nn.Module):
    """A projection stored in the rotated basis: rows are rotated, then ``inner`` (the lane or row matmul) runs."""

    def __init__(self, inner: nn.QuantizedLinear, signs: mx.array, rotation: RotationCache | None = None) -> None:
        super().__init__()
        self.inner = inner
        self.signs = signs
        self.rotation = rotation if rotation is not None else RotationCache()

    def rotate(self, x: mx.array) -> mx.array:
        return self.rotation(x, self.signs)

    def __call__(self, x: mx.array) -> mx.array:
        # MLX promotes bf16 rows against the pack's fp16 scales to fp32; caches and row kernels keep bf16
        return self.inner(self.rotate(x)).astype(x.dtype)

    def project_rows(self, x: mx.array) -> mx.array:
        """The row decoder's projection (``row_matmul.project``) of the rotated rows."""

        from tensorfold.kernels.qwen.dense.v1 import row_matmul

        return row_matmul.project(self.inner, self.rotate(x))


class RotatedEmbedding(nn.Module):
    """A 2-bit embedding stored in the rotated basis: a lookup dequantizes and rotates back, row by row."""

    def __init__(self, weight: mx.array, scales: mx.array, biases: mx.array, signs: mx.array, group: int) -> None:
        super().__init__()
        self.weight, self.scales, self.biases, self.signs = weight, scales, biases, signs
        self.group = int(group)

    def __call__(self, ids: mx.array) -> mx.array:
        return rotate.embed_rows(ids, self.weight, self.scales, self.biases, self.signs, self.group)


# the widest call any decode window makes (the lane kernels' 128 rows); prompt chunks past it take MLX's matmul
ROW_EXACT_ROWS = 128


class RowDense(nn.Module):
    """An unquantized fp32 projection (the recurrent layers' a and b): decode windows' rows independent of the count."""

    def __init__(self, weight: mx.array) -> None:
        super().__init__()
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        if x.size // int(x.shape[-1]) > ROW_EXACT_ROWS:
            return (x.astype(mx.float32) @ self.weight.T).astype(mx.bfloat16)
        return rotate.dense_rows(x, self.weight)

    def project_rows(self, x: mx.array) -> mx.array:
        return rotate.dense_rows(x, self.weight)


def inner_of(module: Any) -> Any:
    """The matmul module under a rotated projection, else the module itself."""

    return module.inner if isinstance(module, RotatedLinear) else module


__all__ = ["RotatedEmbedding", "RotatedLinear", "RotationCache", "RowDense", "inner_of", "rotated"]
