"""Reuse only qualified prompt state for the optimized SM70 sampling path."""

IDENTITY = "sm70-optimized-stochastic-prefix-cold-v1"


def identity(profile):
    """Immutable startup identity; the request's sampling fields already share the worker wire."""
    return IDENTITY if profile == "optimized" else None


def reuse_allowed(w, sampling, *, draft, vision, position):
    """Decline cached suffixes only for validated positive-temperature optimized SM70 requests.

    Drafting and prefix creation remain independent. Malformed policies retain
    their original path so validation does not overtake a producing KV error.
    """
    ordinary = draft and vision is None
    if (not ordinary or w.meta.get("execution_provider") != "sm70"
            or w.meta.get("sm70_profile") != "optimized" or sampling is None):
        return ordinary
    from tensorfold.cuda.sampling import validate_policy

    try:
        validate_policy([position], sampling)
        stochastic = sampling.temperature > 0
    except (ValueError, TypeError, OverflowError):
        return ordinary
    return ordinary and not stochastic
