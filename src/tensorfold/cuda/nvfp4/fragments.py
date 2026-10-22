"""Bounded FP8 byte packing selected only by the positive dense loading policy.

The caller supplies U8 [N,K], K divisible by64, and npad >= N divisible by128,
as validated/constructed by Fp8Linear.from_checkpoint. No floating arithmetic,
new quantization, device synchronization or persistent scratch is introduced.
"""

import torch

from .linear import FRAGMENT_ROWS


def fragment_order(codes: torch.Tensor, npad: int) -> torch.Tensor:
    """Pack the original e4m3 bytes into the shared GEMM/W8A16 fragment ABI.

    Split rows into [tile,8,8] and K into [tile,2,2,2,4,2]. Their permutation is
    the exact original fragment_index lane/byte map. Each bounded chunk copies
    that view into final owned output, without gather indices or a packed copy.
    Current-stream copy ordering and source/temporary ownership are ordinary
    Torch operations, matching the original packer's execution boundary.
    """
    n, k = codes.shape
    out = torch.empty((npad * k,), dtype=torch.uint8, device=codes.device)
    for r0 in range(0, npad, FRAGMENT_ROWS):
        r1 = min(r0 + FRAGMENT_ROWS, npad)
        padded = torch.zeros((r1 - r0, k), dtype=torch.uint8, device=codes.device)
        padded[:max(0, min(r1, n) - r0)] = codes[r0:min(r1, n)]
        tiles = padded.view((r1 - r0) // 64, 8, 8, k // 64, 2, 2, 2, 4, 2)
        fragment = tiles.permute(0, 3, 1, 2, 7, 4, 5, 6, 8)
        out[r0 * k:r1 * k].view((r1 - r0) // 64, k // 64, 8, 8, 4, 2, 2, 2, 2).copy_(fragment)
    return out
