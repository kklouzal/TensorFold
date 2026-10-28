"""CPU checks for finite expert diagnostic dispatch and report contracts."""
import csv
import io
from pathlib import Path
import pytest
from tools import ncu_experts_actual as probe
from tools import ncu_counter_smoke as smoke


def report(names=("expert_gate_up<int>", "expert_down80_exact<int>")):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Kernel Name", *smoke.METRICS])
    writer.writerow(["", "", "us", "cycle", "byte"])
    for index, name in enumerate(names):
        writer.writerow([str(index), name, "2.88", "3,723.825", "640"])
    return output.getvalue()


def test_exact_two_kernels_and_host_policy_preserved():
    argv = probe.command(Path('/source'), Path('/fresh'), Path('/model.gguf'), Path('/wheel.whl'), 'a7b1661f', 32)
    def option(name):
        return argv[argv.index(name) + 1]
    assert option('--launch-count') == '2'
    assert option('--profile-from-start') == 'off'
    assert option('--clock-control') == option('--cache-control') == 'none'
    assert option('--replay-mode') == 'kernel'
    assert [argv[i + 1] for i, value in enumerate(argv) if value == '--section'] == list(probe.SECTIONS)
    assert '--run-operator' in argv and option('--rows') == '32'
    assert probe.ROWS == (1, 8, 32) and probe.EXPERTS == 512


def test_raw_wide_report_retains_kernel_identity_counter_values_and_units():
    result = probe.parse_report(report())
    assert result['units'] == dict(zip(smoke.METRICS, ['us', 'cycle', 'byte']))
    assert result['kernels'][1]['counters']['sm__cycles_elapsed.avg'] == 3723.825


@pytest.mark.parametrize('names', [(), ('expert_gate_up',), ('unrelated', 'expert_down80_exact'),
                                  ('expert_down80_exact', 'expert_gate_up'),
                                  ('expert_gate_up', 'expert_down80_exact', 'other')])
def test_missing_extra_or_wrong_kernel_never_grants_pass(names):
    with pytest.raises(ValueError):
        probe.parse_report(report(names))


def test_nonfinite_hardware_measurement_never_grants_pass():
    with pytest.raises(ValueError):
        probe.parse_report(report().replace('2.88', 'nan'))


def test_fixture_loader_expands_owned_fixture_count_without_gpu(monkeypatch, tmp_path):
    directory = tmp_path / 'tests/cuda'
    directory.mkdir(parents=True)
    (directory / 'test_sm70_gguf_exact_actual.py').write_text('count = 10\ndef make(layer, world):\n    return count, layer, world\n')
    fixture = probe.fixture(tmp_path)
    assert fixture.make(0, 8) == (512, 0, 8)


def test_actual_owned_fixture_imports_with_explicit_full_bank_count():
    pytest.importorskip("torch")
    module = probe.fixture(Path(__file__).resolve().parents[1])
    assert module.count == 512 and module.dims == 2560
    assert callable(module.make) and callable(module.checkpoint)


@pytest.mark.parametrize("rows", probe.ROWS)
def test_shipping_int32_picks_preserve_seeded_legacy_routing_values(rows):
    torch = pytest.importorskip("torch")
    generator = torch.Generator(device="cpu").manual_seed(9201 + rows)
    torch.randn(rows, 2560, generator=generator)  # Same input draws precede routing.
    prior_state = generator.get_state().clone()
    actual = probe.synthetic_ids(rows, generator)
    legacy = torch.Generator(device="cpu")
    legacy.set_state(prior_state)
    expected = torch.stack([torch.randperm(512, generator=legacy)[:10] for _ in range(rows)])
    expected = torch.cat([expected, torch.full((rows, 1), 512)], dim=1)
    assert actual.dtype == torch.int32 and actual.device.type == "cpu"
    assert actual.shape == (rows, 11) and actual.is_contiguous()
    assert torch.equal(actual.long(), expected)
    assert torch.equal(generator.get_state(), legacy.get_state())
