"""Canonical token-row RMS reduction, preserving native HF FP32 stage order.

Pinned Torch's pow(x,-0.5) CUDA implementation dispatches to rsqrt. TensorFold's
existing CT RMS primitive has the same square/mean/epsilon/rsqrt/scale order and
one BF16 output rounding, with a fixed reduction per flattened row. Prompts
remain native by default; explicit reference scopes retain the HF baseline
even when the prompt fusion candidate is enabled.
"""
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm


class E2BRowRMSNorm(Gemma4RMSNorm):
    def __init__(self, native, policy):
        torch.nn.Module.__init__(self)
        self.eps, self.with_scale = native.eps, native.with_scale
        if self.with_scale:
            self.weight = native.weight
        self.policy = policy
        self.train(native.training)

    def forward(self, value):
        if self.policy.prompt:
            if not self.policy.fuse_prompt_norms or self.policy.reference_depth:
                return super().forward(value)
        if value.dtype != torch.bfloat16:
            raise ValueError("Gemma E2B canonical RMS requires BF16 activations")
        from .forward import ct_rms_norm

        return ct_rms_norm(value, self.weight if self.with_scale else None, self.eps)


def install_row_norms(module, policy):
    for name, child in tuple(module.named_children()):
        if isinstance(child, Gemma4RMSNorm):
            setattr(module, name, E2BRowRMSNorm(child, policy))
        else:
            install_row_norms(child, policy)
