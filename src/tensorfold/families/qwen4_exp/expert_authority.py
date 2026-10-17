"""Exact allocation geometry for immutable affine/NVFP4 host expert banks."""
from __future__ import annotations


def allocations(text, layout, *, owned=None, shared=True):
    """Actual tensor bytes, in loader order, including MTP banks already selected by layout."""
    experts = int(text["num_experts"])
    count = experts if owned is None else owned
    if type(count) is not int or not 1 <= count <= experts or type(shared) is not bool:
        raise ValueError("invalid owned expert authority geometry")
    full = experts + (layout.format == "affine4")
    count += shared and layout.format == "affine4"
    fields = layout.authority_allocations
    if not fields or any(size % full for size in fields):
        raise ValueError("expert authority layout does not describe whole per-expert fields")
    return tuple((size // full) * count for size in fields)


def pin_policy(text, layout, mode, *, owned=None, shared=True, future_bytes=0):
    from tensorfold.cuda.expert_pin import ExpertPinPolicy
    geometry = allocations(text, layout, owned=owned, shared=shared)
    return ExpertPinPolicy(sum(geometry), mode=mode, allocation_bytes=geometry,
                           future_bytes=layout.loading_bytes + layout.staging_bytes + layout.metadata_host_bytes
                           + layout.control_host_bytes + future_bytes)
