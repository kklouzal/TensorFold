"""The larger decode domain remains inside the existing admitted native scratch."""

import pytest


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_r32_native_launch_and_optional_splitk_scratch_fit_existing_allowance(world):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import DECODE_BYTES, GGUFExperts

    width, rows, slots = 640 // world, GGUFExperts.packed_rows, 11
    assert rows == 32 and rows * slots <= 65535
    activation = rows * slots * width * 2
    splitk = rows * slots * width * 4 * 2 * 4
    assert activation + splitk <= DECODE_BYTES
    assert activation <= 450560 and activation + splitk <= 7659520
    # The caller's output is separately allocated/accounted in MoEBuffers.
    assert rows * slots * 2560 * 4 == 3604480
