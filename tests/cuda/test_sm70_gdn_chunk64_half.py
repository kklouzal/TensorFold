"""Native Half boundary diagnostics; error measurements are not acceptance bounds."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from tensorfold.cuda.sm70 import gdn_chunk64_half as half
from tools.v100_gdn_chunk64_trial import errors, half_epilogue_reference, half_frontend_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires an actual coordinator-owned SM70 component lease")


@pytest.fixture(scope="module")
def prepared():
    return half.prepare(torch.device("cuda", torch.cuda.current_device()))


def operands(rows, pattern, negative):
    generator = torch.Generator(device="cuda").manual_seed(20261113 + rows)

    def random(shape, dtype=torch.float16):
        return torch.randn(shape, generator=generator, device="cuda", dtype=dtype) * .125

    projection = random((rows, 2060))
    history, weight = random((3, 1280)), random((1280, 4))
    parameter = torch.tensor([.001, .01, .1, 1., 17., 127.], device="cuda")
    parameter = -parameter if negative else parameter.log()
    bias = random((6,), torch.float32)
    norm = random((128,)) + 1
    raw = random((1, rows, 6, 128))
    if pattern == "zero":
        projection.zero_(); history.zero_(); raw.zero_()
    elif pattern == "extreme-gate":
        projection[:, 2048:2054] = torch.tensor([-30., 30., -1., 1., -0., 0.], device="cuda")
        projection[:, 2054:2060] = torch.tensor([-1000., 1000., -20., 20., -1., 1.], device="cuda")
        raw[0, 0, 0, 0] = -0.
    return projection, history, weight, parameter, bias, norm, raw


@pytest.mark.parametrize("rows", [1, 63, 64, 65, 128, 129, 256])
@pytest.mark.parametrize("negative", [False, True], ids=["A_log", "A_neg"])
@pytest.mark.parametrize("pattern", ["random", "zero", "extreme-gate"])
@torch.inference_mode()
def test_native_boundaries_measure_error_preserve_inputs_and_inactive_rows(
        prepared, rows, negative, pattern, record_property):
    p, history, weight, parameter, bias, norm, raw = operands(rows, pattern, negative)
    originals = [value.clone() for value in (p, history, weight, parameter, bias, norm, raw)]
    capacity = rows + 2
    workspace = half.allocate_workspace(capacity, p.device, device_budget_bytes=capacity * 4144)
    for value in (workspace.q, workspace.k, workspace.v, workspace.raw_gate, workspace.beta, workspace.raw_output):
        value.fill_(-77)
    output = torch.full((rows, 768), float("nan"), device=p.device, dtype=torch.float16)
    sums = torch.full((rows, 24), float("nan"), device=p.device)
    bound = prepared.bind(workspace, p, history, weight, bias, norm, 1e-6, output, sums,
                          **{"a_neg" if negative else "a_log": parameter})
    core = bound.front()
    # Isolate the epilogue contract from the separately qualified recurrent core.
    core[-1].copy_(raw)
    bound.back()
    torch.cuda.synchronize(p.device)
    expected = half_frontend_reference(p, history, weight, parameter, bias, negative_decay=negative)
    expected_back = half_epilogue_reference(raw, p, norm, 1e-6)
    measurements = {name: errors(actual, reference) for name, actual, reference in
                    zip(("q", "k", "v", "raw_gate", "beta", "output", "sums"),
                        (*core[:5], output, sums), (*expected, *expected_back))}
    assert all(item["finite"] and item["nonfinite_elements"] == 0
               and item["error_elements"] == actual.numel()
               for item, actual in zip(measurements.values(), (*core[:5], output, sums)))
    assert bool((core[3] <= 0).all())
    assert torch.equal(core[-1].view(torch.uint8), raw.view(torch.uint8))
    for value, original in zip((p, history, weight, parameter, bias, norm, raw), originals):
        assert torch.equal(value.view(torch.uint8), original.view(torch.uint8))
    for value in (workspace.q, workspace.k, workspace.v, workspace.raw_gate, workspace.beta, workspace.raw_output):
        assert bool((value[:, rows:] == -77).all())
    if pattern == "zero":
        assert all(bool((value == 0).all()) for value in (*core[:3], output, sums))
    module = Path(prepared.module.__file__)
    record_property("g3_half_boundaries", json.dumps({
        "rows": rows, "capacity": capacity, "pattern": pattern, "negative_decay": negative,
        "numerical_mode": half.NUMERICAL_MODE, "errors": measurements,
        "native_module": str(module), "native_elf_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "qualification": "boundary controls only; no accepted error bound, core/model quality or P3 claim"}))
