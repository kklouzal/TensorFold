"""Private blob geometry/planner oracles without Torch or accelerator imports."""

from test_qwen_moe_ram_experts import checkpoint

from tensorfold.cuda.nvfp4.blob_layout_a4 import blob_geometry_a4
from tensorfold.families.qwen3_5_moe import ram_experts
from tensorfold.families.qwen3_5_moe.cuda.blob_variant_a4 import layout


def test_blob_preserves_exact_six_field_cell_offsets_and_budget():
    geometry = blob_geometry_a4(512, 2048)
    b = 512 * 2048 * 9 // 16
    assert geometry.entry_bytes == 3 * b + 64 == 1_769_536
    assert [f.offset for f in geometry.fields] == [0, 2 * b, 3 * b, 3 * b + 16, 3 * b + 32, 3 * b + 48]
    assert [f.size for f in geometry.fields] == [2 * b, b, 8, 4, 8, 4]
    assert sum(hi - lo for lo, hi in geometry.padding) == 40
    intervals = [(f.offset, f.offset + f.size) for f in geometry.fields] + list(geometry.padding)
    at = 0
    for lo, hi in sorted(intervals):
        assert lo == at and lo < hi
        at = hi
    assert at == geometry.entry_bytes


def test_blob_plan_charges_source_padding_and_compact_cpu_metadata_without_gpu_growth(tmp_path):
    checkpoint(tmp_path)
    reference = ram_experts.layout(tmp_path, "auto")
    candidate = layout(tmp_path, "auto")
    assert candidate.entry_bytes == reference.entry_bytes
    assert candidate.slots == reference.slots and candidate.gpu_bytes == reference.gpu_bytes
    assert candidate.host_bytes == reference.host_bytes + 40 * reference.expert_count
    assert candidate.authority_allocations == (3 * reference.entry_bytes,) * 2
    assert sum(candidate.authority_allocations) == candidate.host_bytes
    assert candidate.control_host_bytes == reference.control_host_bytes + 24 * reference.expert_count
    assert candidate.staging_bytes == reference.staging_bytes
    assert candidate.loading_bytes == reference.loading_bytes
    assert candidate.runtime_workspace_bytes == reference.runtime_workspace_bytes
    receipt = candidate.receipt()
    assert receipt["experimental_variant"] == "single-blob-grouped"
    assert receipt["source_padding_bytes"] == 40 * reference.expert_count
    assert receipt["compact_cpu_calibration_bytes"] == 24 * reference.expert_count
    assert receipt["prepared_activation_calibration_bytes"] == 12 * reference.expert_count
    assert receipt["h2d_submissions_per_expert_miss"] == 1
