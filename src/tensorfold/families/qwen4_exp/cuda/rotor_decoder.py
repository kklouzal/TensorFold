"""Exact seven-bit lookup policy for the measured serial/MTP envelope.

Startup selects a qualified exact format and runtime envelope. A serialized
decoder boundary then requires one complete logical request owner, including
filling/waiting requests. The returned flag is a Triton constexpr; it changes
only centroid lookup, preserving original FP32 multiplication/BF16 rounding.
"""
from __future__ import annotations

from dataclasses import dataclass

# Normal installed native, exact raw-quality and balanced generation gates
# qualify this startup/request envelope; see docs/optimization-validation.md.
SERIAL_FORMATS: tuple[str, ...] = ("rotorquant7",)
MTP_FORMATS: tuple[str, ...] = ("rotorquant7",)
FORMATS = ("rotorquant6", "rotorquant6-norm", "rotorquant6-outlier-norm", "rotorquant7")
RUNTIME = (
    ("torch", "2.16.0.dev20261006+cu134"),
    ("torch_git", "1a66f68165eb1685c14874588d9a2e0cd475e116"),
    ("CUDA", "13.4"),
    ("triton_distribution", "3.9.0+gitaad2a60d"),
    ("device", "NVIDIA RTX PRO 2000 Blackwell"),
    ("capability", (12, 0)),
    ("kernel_release", "7.0.0-31-generic"),
    ("driver_proc_sha256", "ee50bb4d3c6062b42c45be2e1e2dbce5b51f47096748a71d9aaa8e5089de5874"),
)


@dataclass(frozen=True, slots=True)
class RotorDecoderPolicy:
    serial: bool = False
    mtp: bool = False

    def __post_init__(self):
        if type(self.serial) is not bool or type(self.mtp) is not bool:
            raise TypeError("lookup Region selections must be booleans")

    def request(self, request, *, owned_requests: int, participants: int, copy: bool) -> bool:
        """Called outside layer loops with the decoder's serialized ownership."""
        if not self.serial and not self.mtp:
            return False
        if (owned_requests != 1 or participants != 1 or copy or request.done
                or request.waiting or request.vision is not None
                or request.constraint is not None or request.background
                or request.probabilities is not None or request.owed or request.carry is not None):
            return False
        # The admitted production-equivalent comparisons use the canonical
        # None greedy policy. Non-None policies keep their original validation
        # and kernel ordering; this hook never reads unvalidated policy fields.
        if request.sampling is not None:
            return False
        return self.mtp if request.draft else self.serial


def select(*, mode, symmetric, head_dim, context, slots, depth, confidence, exl3,
           graphs, prefetch, automatic_experts, vision, tp, yarn_factor, runtime):
    """Originals remain selected outside the measured startup envelope."""
    if (mode not in FORMATS or symmetric is not True or exl3 is not True or head_dim != 256 or context != 2048
            or slots != 4 or depth != 4 or confidence != .5 or graphs or prefetch
            or not automatic_experts or vision or tp != 1 or yarn_factor != 2
            or runtime != dict(RUNTIME)
            or any(type(value) is not int for value in (head_dim, context, slots, depth, tp))
            or any(type(value) is not bool for value in (graphs, prefetch, automatic_experts, vision))
            or type(confidence) is not float or type(yarn_factor) not in (int, float)):
        return RotorDecoderPolicy()
    return RotorDecoderPolicy(mode in SERIAL_FORMATS, mode in MTP_FORMATS)
