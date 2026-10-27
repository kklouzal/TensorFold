"""Reuse only qualified prompt state for SM70 request paths."""

from ..qsa_selection import POLICY as QSA_SELECTION

IDENTITY = "sm70-stochastic-or-unqualified-rotor-prefix-cold-v2:" + QSA_SELECTION
FAMILY_TRIAL_IDENTITY = "sm70-source-family-trial-all-prefix-cold-v1:" + QSA_SELECTION


def identity(profile):
    """Sampler and ordered cache descriptors already share the worker wire."""
    if profile == "source-family-trial":
        return FAMILY_TRIAL_IDENTITY
    return IDENTITY if profile in ("optimized", "reference") else None


def reuse_allowed(w, sampling, *, draft, vision, position, kv_pair=None):
    """Borrow only qualified SM70 prompt state; drafting remains independent.

    Cold serial/drafted Q4 frames match, but cached-suffix preparation has not
    met same-format output identity. Its preparation and quantization history
    require separate qualification before borrowing. FP16/INT8
    keep their existing policy. Malformed sampling retains the original error
    path so this decision cannot overtake a producing KV failure.
    """
    ordinary = draft and vision is None
    if not ordinary or w.meta.get("execution_provider") != "sm70":
        return ordinary
    if w.meta.get("sm70_profile") == "source-family-trial":
        return False  # Cold qualification has no same-format cached-suffix receipt yet.
    rotor = kv_pair is not None and (kv_pair.key.codec or kv_pair.value.codec)
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
    if (w.meta.get("execution_provider") == "sm70"
            and w.meta.get("sm70_profile") == "source-family-trial"):
        return False
    if (not draft or w.meta.get("execution_provider") != "sm70"
            or not (kv_pair.key.codec or kv_pair.value.codec)):
        return draft
    return reuse_allowed(w, sampling, draft=draft, vision=None, position=position, kv_pair=kv_pair)
