"""Prepared row reuse for the measured Blackwell BF16 vocabulary head."""

from __future__ import annotations

import torch

from tensorfold.families.qwen3_5.cuda.weights import Plain

# Only these measured row counts use a different launch geometry. Every row
# keeps the incumbent ordered FP32 FMA chain, XOR reduction and BF16 rounding.
_ROWS = (0, 0, 2, 4, 4, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 16)


class _RowsHead(Plain):
    def __init__(self, weight, multiply):
        super().__init__(weight)
        self.multiply = multiply

    def __call__(self, x):
        rows = _ROWS[x.shape[0]] if x.shape[0] < len(_ROWS) else 0
        if not rows:
            return super().__call__(x)
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        return self.multiply(x.contiguous(), self.weight, rows)


def prepare_head(head):
    """Bind once before DFlash borrows the head; other heads retain Plain.

    This changes neither resident weight storage nor precision. Device/weight
    geometry is immutable for the loaded model's lifetime; no capability query
    occurs during generation. Prompt projections retain Plain.prefill.
    """
    if (type(head) is not Plain or head.weight.device.type != "cuda" or head.weight.dtype != torch.bfloat16
            or tuple(head.weight.shape) != (248320, 2048) or not head.weight.is_contiguous()
            or torch.cuda.get_device_capability(head.weight.device) != (12, 0)):
        return head
    from tensorfold.families.qwen3_5.cuda.b16 import _ext

    return _RowsHead(head.weight, _ext().b16_linear_rows)
