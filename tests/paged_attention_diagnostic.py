"""Tolerance-free numerical observations; no acceptance policy is implied."""

import hashlib

import torch


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def differences(reference, candidate):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("diagnostic operands require identical shape and dtype")
    if reference.dtype not in (torch.float16, torch.float32):
        raise ValueError("diagnostic operands require FP16 or FP32")
    reference, candidate = reference.detach().cpu().contiguous(), candidate.detach().cpu().contiguous()
    width = reference.element_size() * 8
    integer = torch.int16 if width == 16 else torch.int32
    a, b = reference.view(integer).to(torch.int64), candidate.view(integer).to(torch.int64)
    a, b = a & ((1 << width) - 1), b & ((1 << width) - 1)
    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    delta = (reference.double() - candidate.double()).abs()
    nonzero = finite & (reference != 0)
    zero_mismatch = finite & (reference == 0) & (candidate != 0)
    sign = 1 << (width - 1)

    def ordered(bits):
        # Adjacent signed zeros have distance one; NaN/Inf excluded below.
        return torch.where(bits & sign != 0, sign - 1 - (bits & (sign - 1)), sign + bits)

    return {
        "elements": reference.numel(), "dtype": str(reference.dtype),
        "exact_bits": bool(torch.equal(a, b)), "different_bits": int((a != b).sum()),
        "reference_sha256": tensor_sha(reference), "candidate_sha256": tensor_sha(candidate),
        "reference_nonfinite": int((~torch.isfinite(reference)).sum()),
        "candidate_nonfinite": int((~torch.isfinite(candidate)).sum()),
        "finite_pairs": int(finite.sum()),
        "max_absolute_finite": float(delta[finite].max()) if bool(finite.any()) else None,
        "max_relative_nonzero_reference": float((delta[nonzero] / reference.double()[nonzero].abs()).max())
        if bool(nonzero.any()) else None,
        "zero_reference_nonzero_candidate": int(zero_mismatch.sum()),
        "max_ulp_finite": int((ordered(a) - ordered(b)).abs()[finite].max()) if bool(finite.any()) else None,
        "ulp_definition": "ordered IEEE bit distance, signed zeros adjacent; finite pairs only",
    }
