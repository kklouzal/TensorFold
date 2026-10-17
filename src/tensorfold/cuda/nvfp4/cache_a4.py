"""One-source native A4 expert cache with prepared strided GPU field views."""

from contextlib import contextmanager

import torch

from tensorfold.cleanup import rollback
from tensorfold.cuda.expert_cache import HostExpertCache

from .host_layout import entry_layout_a4


def native_views(blob, geometry):
    """Prepare six strided expert-row views; each individual expert is contiguous."""
    if (blob.dtype != torch.uint8 or blob.ndim != 2 or not blob.is_contiguous()
            or blob.shape[0] <= 0 or blob.shape[1] != geometry.entry_bytes):
        raise ValueError("native A4 blob must be contiguous U8[count,aligned_entry]")
    result = []
    for field in geometry.fields:
        typed = blob.narrow(1, field.offset, field.size).view(
            torch.uint8 if field.dtype == "U8" else torch.float32)
        inner, stride = [], 1
        for size in reversed(field.shape):
            inner.append(stride)
            stride *= size
        # view() can canonicalize an unused singleton expert stride. Explicit
        # strides preserve the entry format at capacity1 as well as after growth.
        result.append(typed.as_strided((blob.shape[0], *field.shape),
                      (geometry.entry_bytes // typed.element_size(), *reversed(inner))))
    return tuple(result)


class A4ExpertCache(HostExpertCache):
    """Base source/copies stay oneblob; leases expose prepared six-field views.

    Pool replacement prepares every derived view before commit. The base lease
    owns readiness and retirement; all native consumers enqueue inside it.
    """

    def __init__(self, gpu_bytes, entry_bytes, device="cuda", *, width, dims, **kwargs):
        self.field_layout = entry_layout_a4(width, dims)
        if entry_bytes != self.field_layout.entry_bytes:
            raise ValueError("native A4 blob entry must preserve the six-field cache budget")
        self._native_views, self._native_down = (), None
        super().__init__(gpu_bytes, entry_bytes, device, **kwargs)
        try:
            with torch.inference_mode(False):
                self._native_views = native_views(self._pool.view(self.capacity, self.entry_bytes), self.field_layout)
                self._native_down = self._native_views[1].unsqueeze(1)
        except BaseException as primary:  # noqa: BLE001 - rollback rethrows after owned retirement.
            rollback(self, primary, self.close)

    def register(self, layer_id, payloads):
        if (not isinstance(payloads, tuple) or len(payloads) != 1 or not isinstance(payloads[0], torch.Tensor)
                or payloads[0].dtype != torch.uint8 or payloads[0].ndim != 2
                or payloads[0].shape[1] != self.entry_bytes):
            raise ValueError("native A4 cache registers exactly one complete U8 expert blob")
        super().register(layer_id, payloads)

    def source_views(self, blob):
        return native_views(blob, self.field_layout)

    def zero_padding(self, blob):
        for lo, hi in self.field_layout.padding:
            blob[:, lo:hi].zero_()

    def _replacement_views(self, pool, capacity):
        views, _ = super()._replacement_views(pool, capacity)
        native = native_views(pool.view(capacity, self.entry_bytes), self.field_layout)
        return views, (native, native[1].unsqueeze(1))

    def _commit_replacement(self, controls):
        self._native_views, self._native_down = controls

    @contextmanager
    def lease(self, layer_id, expert_ids):
        with super().lease(layer_id, expert_ids) as (_, mapping):
            yield self._native_views, mapping

    def close(self):
        super().close()
        self._native_views, self._native_down = (), None

