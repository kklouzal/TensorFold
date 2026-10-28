"""Frozen c928 indexed tail oracle; deliberately uses original tensor gathers/masks."""
import torch


def indexed_tail(entry, length):
    if type(length) is not int or not 0 <= length <= entry.capacity:
        raise ValueError("compact tail length must fit the selected cache")
    if length == entry.n:
        with entry.reading((entry.raw, entry.coordinates)):
            return entry.raw.clone(), entry.coordinates.clone()
    if length % 4 == 0 and length < entry.n:
        return torch.zeros_like(entry.raw), torch.zeros_like(entry.coordinates)
    span = entry.span
    active_span = None
    if entry.base is not None and entry.base <= length <= entry.base + entry.proposal_count:
        start, raw, coordinates = entry.base, entry.proposal_raw[:entry.proposal_count], entry.proposal_coordinates[:entry.proposal_count]
        protected, protected_coordinates = entry.base_raw, entry.base_coordinates
    elif span is not None and span.start <= length <= span.end:
        span.check()
        if span.resolved is not None and length > span.resolved:
            raise ValueError("compact cut contains rejected cache rows")
        start, raw, coordinates = span.start, span.raw, span.coordinates
        protected, protected_coordinates = entry.protected, entry.protected_coordinates
        active_span = span
    else:
        raise ValueError("compact cut needs its captured raw tail; pooled means cannot reconstruct it")
    count = length % 4
    result, coords = torch.zeros_like(entry.raw), torch.zeros_like(entry.coordinates)
    if count:
        with entry.reading((protected, protected_coordinates, raw, coordinates), span=active_span):
            positions = torch.arange(length - count, length, device=entry.raw.device, dtype=torch.int64)
            old = positions < start
            source = (positions - start).clamp(0, max(0, len(raw) - 1))
            values = torch.where(old[:, None], protected.index_select(0, positions % 4), raw.index_select(0, source))
            axes = torch.where(old[:, None], protected_coordinates.index_select(0, positions % 4), coordinates.index_select(0, source))
            result.index_copy_(0, positions % 4, values)
            coords.index_copy_(0, positions % 4, axes)
    return result, coords
