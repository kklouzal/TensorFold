"""Independent CPU bit decoder for tests; production never imports this oracle."""
import numpy as np
import torch


def decode_cpu_chunks(packed, *, chunk_groups=4096):
    """Yield owned flat BF16 chunks from the documented byte/nibble layout."""
    metadata = packed.metadata.numpy().view(np.uint32)
    raw = packed.raw.numpy().view(np.uint16)
    for begin in range(0, len(metadata), chunk_groups):
        info = metadata[begin:begin + chunk_groups]
        indices = info >> 8
        low = packed.low[begin:begin + len(info)].numpy().astype(np.uint16)
        delta = packed.delta[begin:begin + len(info)].numpy()
        differences = np.empty((len(info), 64), dtype=np.int32)
        differences[:, 0::2], differences[:, 1::2] = delta & 15, delta >> 4
        exponent = (info & 255).astype(np.int32)[:, None] - differences
        compressed = indices == 0xffffff
        if np.any(exponent[compressed] < 0):
            raise ValueError('invalid compressed exponent delta')
        words = ((low & 128) << 8) | (exponent.astype(np.uint16) << 7) | (low & 127)
        if np.any(~compressed):
            if np.any(indices[~compressed] >= len(raw)):
                raise ValueError('raw escape index outside owned pool')
            words[~compressed] = raw[indices[~compressed]]
        yield begin * 64, torch.from_numpy(words.reshape(-1).view(np.int16).copy()).view(torch.bfloat16)
