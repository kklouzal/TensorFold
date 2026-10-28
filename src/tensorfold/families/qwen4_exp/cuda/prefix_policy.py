"""Reuse only qualified prompt state for SM70 request paths."""

import hashlib

from ..qsa_selection import POLICY as QSA_SELECTION

IDENTITY = "sm70-stochastic-or-unqualified-rotor-prefix-cold-v2:" + QSA_SELECTION
FAMILY_TRIAL_IDENTITY = "sm70-source-family-trial-all-prefix-cold-v1:" + QSA_SELECTION
FAMILY_PREFIX_PROFILE = "source-family-prefix-trial"
FAMILY_PREFIX_IDENTITY = "sm70-source-family-int8-greedy-text-prefix-trial-v1:" + QSA_SELECTION
# A fixed semantic component extends the existing codec/layout/position identity.
# It is a policy epoch, not a model-content or qualification certificate.
FAMILY_PREFIX_STATE = ("sm70-prefix-epoch-v1", hashlib.sha256(FAMILY_PREFIX_IDENTITY.encode()).hexdigest())
PACKED_PROFILE = "source-packed-trial"
PACKED_IDENTITY = "sm70-packed-exact-native32-int8-greedy-text-prefix-trial-v1:" + QSA_SELECTION
PACKED_STATE = ("sm70-prefix-epoch-v1", hashlib.sha256(PACKED_IDENTITY.encode()).hexdigest())
PACKED_GRAPH_PROFILE = "source-packed-graphs-trial"
PACKED_GRAPH_IDENTITY = "sm70-packed-exact-native32-n1-piece-graphs-int8-greedy-prefix-trial-v1:" + QSA_SELECTION
PACKED_GRAPH_STATE = ("sm70-prefix-epoch-v1", hashlib.sha256(PACKED_GRAPH_IDENTITY.encode()).hexdigest())
OWNER_GRAPH_PROFILE = "source-owner-pieces-trial"
OWNER_GRAPH_IDENTITY = "sm70-native-owner-full8-eager-packed-exact-native32-n1-pieces-int8-greedy-v1:" + QSA_SELECTION
OWNER_GRAPH_STATE = ("sm70-prefix-epoch-v1", hashlib.sha256(OWNER_GRAPH_IDENTITY.encode()).hexdigest())
TARGET_GREEDY_PROFILE = "source-target-greedy-trial"
TARGET_GREEDY_IDENTITY = "sm70-target-greedy-int-key-chain-int8-text-prefix-trial-v1:" + QSA_SELECTION
TARGET_GREEDY_STATE = ("sm70-prefix-epoch-v1", hashlib.sha256(TARGET_GREEDY_IDENTITY.encode()).hexdigest())
OWNER_TARGET_GREEDY_PROFILE = "source-owner-target-greedy-trial"
OWNER_TARGET_GREEDY_IDENTITY = "sm70-native-owner-n1-pieces-target-greedy-int-key-chain-v1:" + QSA_SELECTION
OWNER_TARGET_GREEDY_STATE = ("sm70-prefix-epoch-v1", hashlib.sha256(OWNER_TARGET_GREEDY_IDENTITY.encode()).hexdigest())
OWNER_TARGET_GUARD_PROFILE = "source-owner-target-guard-trial"
OWNER_TARGET_GUARD_IDENTITY = "sm70-native-owner-n1-pieces-target-greedy-valueguard-v1:" + QSA_SELECTION
OWNER_TARGET_GUARD_STATE = ("sm70-prefix-epoch-v1", hashlib.sha256(OWNER_TARGET_GUARD_IDENTITY.encode()).hexdigest())
OWNER_TARGET_PRELUDE_PROFILE = "source-owner-target-prelude-trial"
OWNER_TARGET_PRELUDE_IDENTITY = "sm70-native-owner-frame-prelude-existing-prep-zero-delta-v1:" + QSA_SELECTION
OWNER_TARGET_PRELUDE_STATE = ("sm70-prefix-epoch-v1", hashlib.sha256(OWNER_TARGET_PRELUDE_IDENTITY.encode()).hexdigest())
GATEUP_INDEPENDENT_PROFILE = "source-owner-gateup-independent-trial"
GATEUP_INDEPENDENT_IDENTITY = "sm70-owner-prelude-exact-independent-gate-up-warps-v1:" + QSA_SELECTION
GATEUP_INDEPENDENT_STATE = ("sm70-prefix-epoch-v1", hashlib.sha256(GATEUP_INDEPENDENT_IDENTITY.encode()).hexdigest())
EARLY_IDENTITY_STATES = (FAMILY_PREFIX_STATE, PACKED_GRAPH_STATE, OWNER_GRAPH_STATE, TARGET_GREEDY_STATE, OWNER_TARGET_GREEDY_STATE, OWNER_TARGET_GUARD_STATE, OWNER_TARGET_PRELUDE_STATE, GATEUP_INDEPENDENT_STATE)


def identity(profile):
    """Sampler and ordered cache descriptors already share the worker wire."""
    if profile == "source-family-trial":
        return FAMILY_TRIAL_IDENTITY
    if profile == FAMILY_PREFIX_PROFILE:
        return FAMILY_PREFIX_IDENTITY
    if profile == PACKED_PROFILE:
        return PACKED_IDENTITY
    if profile == PACKED_GRAPH_PROFILE:
        return PACKED_GRAPH_IDENTITY
    if profile == OWNER_GRAPH_PROFILE:
        return OWNER_GRAPH_IDENTITY
    if profile == TARGET_GREEDY_PROFILE:
        return TARGET_GREEDY_IDENTITY
    if profile == OWNER_TARGET_GREEDY_PROFILE:
        return OWNER_TARGET_GREEDY_IDENTITY
    if profile == OWNER_TARGET_GUARD_PROFILE:
        return OWNER_TARGET_GUARD_IDENTITY
    if profile == GATEUP_INDEPENDENT_PROFILE:
        return GATEUP_INDEPENDENT_IDENTITY
    if profile == OWNER_TARGET_PRELUDE_PROFILE:
        return OWNER_TARGET_PRELUDE_IDENTITY
    return IDENTITY if profile in ("optimized", "reference") else None


def state_epoch(profile):
    """Only the explicit cached trial adds an immutable State compatibility epoch."""
    if profile == PACKED_PROFILE:
        return PACKED_STATE
    if profile == PACKED_GRAPH_PROFILE:
        return PACKED_GRAPH_STATE
    if profile == OWNER_GRAPH_PROFILE:
        return OWNER_GRAPH_STATE
    if profile == TARGET_GREEDY_PROFILE:
        return TARGET_GREEDY_STATE
    if profile == OWNER_TARGET_GREEDY_PROFILE:
        return OWNER_TARGET_GREEDY_STATE
    if profile == OWNER_TARGET_GUARD_PROFILE:
        return OWNER_TARGET_GUARD_STATE
    if profile == GATEUP_INDEPENDENT_PROFILE:
        return GATEUP_INDEPENDENT_STATE
    if profile == OWNER_TARGET_PRELUDE_PROFILE:
        return OWNER_TARGET_PRELUDE_STATE
    return FAMILY_PREFIX_STATE if profile == FAMILY_PREFIX_PROFILE else None


def _family_prefix_allowed(sampling, position, kv_pair):
    if kv_pair is None or kv_pair.key_dtype != "int8" or kv_pair.value_dtype != "int8":
        return False
    from tensorfold.cuda.sampling import validate_policy

    try:
        validate_policy([position], sampling)
        return sampling is None or bool(sampling.temperature <= 0)
    except (ValueError, TypeError, OverflowError):
        # Leave the original producing validation/error ordering intact; a
        # malformed request gains neither a cache lease nor a default policy.
        return False


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
    if w.meta.get("sm70_profile") in (FAMILY_PREFIX_PROFILE, PACKED_PROFILE, PACKED_GRAPH_PROFILE, OWNER_GRAPH_PROFILE, TARGET_GREEDY_PROFILE, OWNER_TARGET_GREEDY_PROFILE, OWNER_TARGET_GUARD_PROFILE, OWNER_TARGET_PRELUDE_PROFILE, GATEUP_INDEPENDENT_PROFILE):
        return _family_prefix_allowed(sampling, position, kv_pair)
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


def serial_reuse_allowed(w, sampling, *, draft, position, kv_pair, vision=None):
    """Serial/twin Q4 has the same qualification; legacy serial policy stays intact."""
    if (w.meta.get("execution_provider") == "sm70"
            and w.meta.get("sm70_profile") in (FAMILY_PREFIX_PROFILE, PACKED_PROFILE, PACKED_GRAPH_PROFILE, OWNER_GRAPH_PROFILE, TARGET_GREEDY_PROFILE, OWNER_TARGET_GREEDY_PROFILE, OWNER_TARGET_GUARD_PROFILE, OWNER_TARGET_PRELUDE_PROFILE, GATEUP_INDEPENDENT_PROFILE)):
        return draft and vision is None and _family_prefix_allowed(sampling, position, kv_pair)
    if (w.meta.get("execution_provider") == "sm70"
            and w.meta.get("sm70_profile") == "source-family-trial"):
        return False
    if (not draft or w.meta.get("execution_provider") != "sm70"
            or not (kv_pair.key.codec or kv_pair.value.codec)):
        return draft
    return reuse_allowed(w, sampling, draft=draft, vision=None, position=position, kv_pair=kv_pair)
