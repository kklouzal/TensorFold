"""Select the qualified Flash Next Volta operators before admission and weight loading."""
from __future__ import annotations

import os

PROFILE_ENV = "TENSORFOLD_SM70_PROFILE"
_SELECTED = "reference"


def configure(text, *, tp, torch):
    """Explicit profiles share the existing engine, admission and worker handshake."""
    global _SELECTED
    profile = os.environ.get(PROFILE_ENV, "reference")
    if profile not in ("reference", "optimized"):
        raise ValueError(f"{PROFILE_ENV} must be reference or optimized")
    from . import family_ops, routing_native
    from tensorfold.families.qwen4_exp.cuda import (
        dspark_context_graphs, dspark_proposal_graphs, gguf_prefill, gguf_weights, padded_linears,
    )
    enabled = profile == "optimized"
    if enabled:
        from .attention_qualification import REQUIRED_IDENTITY, runtime_identity
        identity = runtime_identity(torch.device("cuda", torch.cuda.current_device()))
        compatible = (tp == 8 and int(text["hidden_size"]) == 2560
                      and int(text["num_hidden_layers"]) == 48
                      and int(text["num_attention_heads"]) == 24
                      and int(text["num_key_value_heads"]) == 2
                      and tuple(torch.cuda.get_device_capability()) == (7, 0)
                      and all(identity.get(key) == value for key, value in REQUIRED_IDENTITY.items()))
        if not compatible:
            raise ValueError("optimized SM70 profile requires the qualified TP8 Flash Next geometry and pinned Volta foundation")
    family_ops._NATIVE_FAMILY = family_ops._NATIVE_ATTENTION = enabled
    routing_native._NATIVE_LOGITS = padded_linears._ENABLED = enabled
    gguf_weights.GGUFExperts.implementation = "packed_exact" if enabled else "reference"
    gguf_weights.GGUFConcat.implementation = "native32" if enabled else "reference"
    gguf_prefill.WORKSPACE_BUDGET_BYTES = (1 << 30) if enabled else None
    dspark_proposal_graphs._ENABLED = dspark_context_graphs._ENABLED = enabled
    _SELECTED = profile
    return profile


def prepare_weights(w):
    """Mark native owners only after the same profile was admitted on every rank."""
    w.meta["sm70_profile"] = _SELECTED
    if _SELECTED == "optimized":
        from .attention_qualification import metadata_prepare
        metadata_prepare(w)
        w.meta.update(capture_ready=True, device_greedy_mtp_ready=True, device_greedy_native_ready=True)
