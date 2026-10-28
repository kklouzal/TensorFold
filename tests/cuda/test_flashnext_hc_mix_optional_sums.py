"""Unused XS stays untouched; mixed bits match the independently gated default HC path."""

import pytest

torch = pytest.importorskip('torch')
if not torch.cuda.is_available():
    pytest.skip('CUDA only', allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import glue


@pytest.mark.parametrize('rows', [1, 16, 28])
@pytest.mark.parametrize('dims', [256, 2560])
@pytest.mark.parametrize('adversarial', [False, True])
def test_omitted_group_sums_preserve_mixed_bits_inputs_versions_and_canaries(rows, dims, adversarial):
    streams, device = 4, 'cuda'
    generator = torch.Generator().manual_seed(20261023)
    if adversarial:
        values = torch.tensor([0., -0., 2.**-120, -2.**-120, 2.**-12, -2.**-12,
                               .5, -.5, 8., -8., 2.**120, -2.**120], dtype=torch.bfloat16)
        index = torch.arange(rows * streams * dims)
        up = values[index % len(values)].view(rows, streams * dims).to(device)
        normed = values[(index * 7 + 3) % len(values)].view_as(up).to(device)
    else:
        up = torch.randn((rows, streams * dims), generator=generator).to(device, torch.bfloat16)
        normed = torch.randn((rows, streams * dims), generator=generator).to(device, torch.bfloat16)
    input_bits = (up.view(torch.int16).clone(), normed.view(torch.int16).clone())
    expected = torch.full((rows + 1, dims), 3.25, dtype=torch.bfloat16, device=device)
    actual = expected.clone()
    sums = torch.full((rows + 1, dims // 32), 12345.25, dtype=torch.float32, device=device)
    ignored = sums.clone()
    untouched_bits, version = ignored.view(torch.uint8).clone(), ignored._version
    glue.hc_mix(up, normed, expected[:rows], sums[:rows], streams)
    glue.hc_mix(up, normed, actual[:rows], ignored[:rows], streams, sums=False)
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(ignored.view(torch.uint8), untouched_bits) and ignored._version == version
    assert torch.equal(up.view(torch.int16), input_bits[0]) and torch.equal(normed.view(torch.int16), input_bits[1])
    assert torch.equal(actual[rows], torch.full_like(actual[rows], 3.25))
