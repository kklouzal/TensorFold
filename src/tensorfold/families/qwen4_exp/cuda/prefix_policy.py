"""Reuse only qualified prompt state for SM70 request paths."""

from ..qsa_selection import POLICY as QSA_SELECTION

IDENTITY = "sm70-stochastic-or-non-owned-rotor-prefix-cold-v3:" + QSA_SELECTION


def identity(profile):
    """Sampler and ordered cache descriptors already share the worker wire."""
    return IDENTITY if profile in ("optimized", "reference") else None


def _cold_rotor(w, kv_pair):
    """Rotor sides whose prompt state is not borrowed: all but the homogeneous owned-writer formats in the
    unique-history layout, whose rows are written by the same per-row history owner code as ordinary Q8."""
    if kv_pair is None or not (kv_pair.key.codec or kv_pair.value.codec):
        return False
    from tensorfold.cuda.sm70.family_ops import OWNED_FORMATS

    return (w.meta.get("history_layout") != "unique"
            or (kv_pair.key.bits, kv_pair.key.codec, kv_pair.value.bits, kv_pair.value.codec) not in OWNED_FORMATS)


def reuse_allowed(w, sampling, *, draft, vision, position, kv_pair=None):
    """Borrow only qualified SM70 prompt state; drafting remains independent.

    Cold serial/drafted Q4 frames match, but the count-dependent legacy Q4
    preparation never met cached-suffix output identity, so those Rotor sides
    stay cold. In the unique-history layout, homogeneous Rotor pairs of
    family_ops.OWNED_FORMATS are written by the history owner exactly as
    ordinary Q8 rows are, and keep the FP16/INT8 policy. Malformed sampling retains the original error
    path so this decision cannot overtake a producing KV failure.
    """
    ordinary = draft and vision is None
    if not ordinary or w.meta.get("execution_provider") != "sm70":
        return ordinary
    rotor = _cold_rotor(w, kv_pair)
    if sampling is None:
        return ordinary and not rotor
    if not rotor and w.meta.get("sm70_profile") != "optimized":
        return ordinary
    from tensorfold.cuda.sampling import validate_policy

    try:
        validate_policy([position], sampling)
        stochastic = sampling.temperature > 0
    except (ValueError, TypeError, OverflowError):
        return ordinary
    return ordinary and not rotor and not stochastic


def serial_reuse_allowed(w, sampling, *, draft, position, kv_pair):
    """Serial/twin Q4 has the same qualification; legacy serial policy stays intact."""
    if not draft or w.meta.get("execution_provider") != "sm70" or not _cold_rotor(w, kv_pair):
        return draft
    return reuse_allowed(w, sampling, draft=draft, vision=None, position=position, kv_pair=kv_pair)
