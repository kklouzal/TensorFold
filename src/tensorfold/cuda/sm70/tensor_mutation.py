"""Mutation budget for the selected source Torch's uint32 version counters."""

MAX_MUTATION_VERSION = (1 << 32) - 1


def require_mutation_budget(tensor, writes):
    """Refuse untracked tensors or counter wrap before any owned write."""
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.is_inference():
        raise ValueError("coordinate/native storage must track Tensor mutations; normal tracked storage is required")
    if type(writes) is not int or not 0 < writes <= MAX_MUTATION_VERSION:
        raise ValueError("coordinate/native mutation requires a positive bounded write count")
    version = tensor._version
    if version > MAX_MUTATION_VERSION - writes:
        raise ValueError("coordinate mutation version exhausted; retire and replace its storage")
    return version
