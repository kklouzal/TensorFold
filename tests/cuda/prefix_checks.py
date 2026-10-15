"""Token SHA and bitwise tensor checks shared by the repeated-prefix and prompt-cache regressions."""

import hashlib
import json

import torch


def same_tokens(actual, expected):
    digest = lambda tokens: hashlib.sha256(json.dumps([int(t) for t in tokens]).encode()).hexdigest()
    assert digest(actual) == digest(expected)
    return digest(actual)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))
