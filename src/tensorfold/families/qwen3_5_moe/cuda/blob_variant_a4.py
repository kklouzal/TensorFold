"""PRIVATE planner variant for the cached one-blob experiment, not a CLI mode."""

from dataclasses import dataclass, fields

from ..ram_experts import Layout
from ..ram_experts import layout as six_field_layout


@dataclass(frozen=True)
class BlobLayout(Layout):
    source_padding_bytes: int = 0
    compact_cpu_calibration_bytes: int = 0

    def receipt(self):
        result = super().receipt()
        result.update(experimental_variant="single-blob-grouped",
                      source_padding_bytes=self.source_padding_bytes,
                      compact_cpu_calibration_bytes=self.compact_cpu_calibration_bytes,
                      authority_allocation_count=len(self.authority_allocations),
                      h2d_submissions_per_expert_miss=1)
        return result


def layout(model_dir, gib, *, tp=1):
    """Preserve GPU slots/budget while charging padding and ordinary CPU controls."""
    base = six_field_layout(model_dir, gib, tp=tp)
    host_bytes = base.expert_count * base.entry_bytes
    padding = host_bytes - base.host_bytes
    if padding < 0:
        raise RuntimeError("private one-blob authority disagrees with the admitted six-field byte geometry")
    compact = 24 * base.expert_count
    values = {field.name: getattr(base, field.name) for field in fields(Layout)}
    values.update(host_bytes=host_bytes,
                  authority_allocations=(base.experts_per_bank * base.entry_bytes,) * len(base.banks),
                  control_host_bytes=base.control_host_bytes + compact,
                  format="ct-nvfp4-a4-single-blob-private")
    return BlobLayout(**values, source_padding_bytes=padding, compact_cpu_calibration_bytes=compact)
