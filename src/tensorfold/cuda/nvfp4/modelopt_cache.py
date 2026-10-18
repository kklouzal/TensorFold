"""Exactly two admitted ModelOpt expert layouts over one fenced hot pool."""

from math import prod

import torch

from tensorfold.cuda.expert_cache import HostExpertCache, _Field

from .host_layout import authority_fields_modelopt


def _formats(width, dims):
    """Prepare exact field signatures without allocating authority tensors."""
    formats = []
    for native in (False, True):
        fields, offset = [], 0
        for shape, dtype_name in authority_fields_modelopt(1, width, dims, native_a4=native):
            dtype, itemsize = {"U8": (torch.uint8, 1), "I32": (torch.int32, 4), "F32": (torch.float32, 4)}[dtype_name]
            offset = (offset + 15) // 16 * 16
            size = prod(shape[1:]) * itemsize
            fields.append(_Field(shape[1:], dtype, offset, size))
            offset += size
        formats.append(tuple(fields))
    return frozenset(formats)


class ModelOptExpertCache(HostExpertCache):
    """A4 main and unchanged W4A16 MTP share physical cells, never live leases.

    Every layer has complete typed views and immutable authority. Base leasing
    retires the previous consumer before a new layout overwrites any cell;
    copies initialize every field of the newly installed logical expert.
    """

    def __init__(self, *args, width, dims, **kwargs):
        self._formats = _formats(width, dims)
        super().__init__(*args, **kwargs)

    def _validate_layout(self, size, fields):
        if size > self.entry_bytes or fields not in self._formats:
            raise ValueError("ModelOpt cache accepts only its admitted W4A16 and native-A4 field layouts")
