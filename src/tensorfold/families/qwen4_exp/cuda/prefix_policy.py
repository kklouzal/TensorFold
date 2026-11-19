"""Reuse only qualified prompt state for SM70 request paths."""

from ..qsa_selection import POLICY as QSA_SELECTION

IDENTITY = "sm70-non-owned-rotor-prefix-cold-v4:" + QSA_SELECTION


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
    ordinary Q8 rows are, and keep the FP16/INT8 policy. The sampler does not
    change prompt state: positive-temperature requests borrow like greedy ones
    (exact repeats resume bitwise; an extension's suffix rounds as its own
    prompt pieces, measured NLL-equivalent to cold in the ledger/docs/api.md).
    Malformed sampling on a cold Rotor side retains the original error path
    (borrowing), so this decision cannot overtake a producing KV failure.
    """
    ordinary = draft and vision is None
    if not ordinary or w.meta.get("execution_provider") != "sm70" or not _cold_rotor(w, kv_pair):
        return ordinary
    if sampling is None:
        return False
    from tensorfold.cuda.sampling import validate_policy

    try:
        validate_policy([position], sampling)
    except (ValueError, TypeError, OverflowError):
        return ordinary
    return False


def serial_reuse_allowed(w, sampling, *, draft, position, kv_pair):
    """Serial/twin Q4 has the same qualification; legacy serial policy stays intact."""
    if not draft or w.meta.get("execution_provider") != "sm70" or not _cold_rotor(w, kv_pair):
        return draft
    return reuse_allowed(w, sampling, draft=draft, vision=None, position=position, kv_pair=kv_pair)
