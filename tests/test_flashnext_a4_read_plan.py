"""Metadata-selected native reads retain every other format's queue contract."""

import pytest
from test_modelopt_a4_control import PREFIX, block, infos

from tensorfold.cuda.nvfp4.modelopt_layout import a4_bank, a4_read_plan, scalar_prefetch_host_bytes


def test_plan_omits_only_native_matrix_reads_and_keeps_original_scalars_on_cpu():
    header = infos()
    assert a4_bank(header, PREFIX, 3, block(), native=True)
    names = list(header)
    device, host = a4_read_plan(names, {PREFIX}, 3)
    matrices = [name for name in names if name.endswith((".weight", ".weight_scale"))]
    calibrations = [name for name in names if name.endswith((".weight_scale_2", ".input_scale"))]
    assert len(matrices) == len(calibrations) == 18
    assert not device and host == calibrations
    assert set(names) == set(matrices) | set(host)
    assert not set(host) & set(matrices)


@pytest.mark.parametrize("native,activation,checkpoint", [(False, True, True), (True, False, True),
                                                         (True, True, False)])
def test_capability_weight_only_and_full_regions_preserve_original_prefetch(native, activation, checkpoint):
    header = infos()
    selected = {PREFIX} if a4_bank(header, PREFIX, 3, block(activation), native=native, checkpoint=checkpoint) else set()
    names = list(header)
    device, host = a4_read_plan(names, selected, 3)
    assert device == names and not host


def test_unknown_shared_backbone_other_bank_and_mtp_names_retain_order_and_destination():
    names = [PREFIX + ".experts.0.gate_proj.weight", "model.language_model.layers.0.self_attn.q_proj.weight",
             PREFIX + ".shared_expert.gate_proj.weight", "mtp.layers.0.mlp.experts.gate_up_proj",
             "mtp.layers.0.mlp.experts.0.gate_proj.input_scale", PREFIX + ".experts.3.gate_proj.weight",
             PREFIX + ".experts.00.gate_proj.input_scale", PREFIX + ".experts.٠.gate_proj.input_scale",
             PREFIX + ".experts.0.other.weight", PREFIX + ".experts.0.gate_proj.other",
             PREFIX + ".experts." + "9" * 10000 + ".gate_proj.weight",
             PREFIX + ".experts.2.down_proj.input_scale"]
    device, host = a4_read_plan(names, {PREFIX}, 3)
    assert device == names[1:-1] and host == names[-1:]
    assert len(names) == 12  # input inventory remains caller-owned/unchanged


@pytest.mark.parametrize("count", [0, -1, True, 1.0])
def test_bad_plan_extent_fails_before_names_are_consumed(count):
    def forbidden():
        pytest.fail("consumed invalid input")
        yield "unused"

    with pytest.raises(ValueError, match="positive expert count"):
        a4_read_plan(forbidden(), {PREFIX}, count)


def test_scalar_prefetch_payload_bound_counts_both_future_layers_and_raw_alignment():
    assert scalar_prefetch_host_bytes(512, workers=8, alignment=4096) == 147456
    assert scalar_prefetch_host_bytes(3, workers=1, alignment=4096) == 288 + 12288
    for count, workers, alignment in ((0, 8, 4096), (1, 0, 4096), (True, 8, 4096), (1, 8, -1)):
        with pytest.raises(ValueError, match="positive integers"):
            scalar_prefetch_host_bytes(count, workers=workers, alignment=alignment)
