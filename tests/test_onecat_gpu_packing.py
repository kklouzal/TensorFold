"""GPU producer boundary/resource controls; independent CPU producer unchanged."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import onecat_gpu_packing as gpu
from tensorfold.cuda.sm70 import onecat_moe_packing as cpu


@pytest.mark.parametrize('kind', [18, 21, 22, 23])
def test_wire_extents_match_independent_committed_CPU_producer(kind):
    actual, expected = gpu.layout(kind), cpu.requirements(kind)
    assert actual['codes_shape'] == expected['codes_shape']
    assert actual['scales_shape'] == expected['scales_shape']
    assert actual['pair_plane_bytes'] == 2*expected['output_bytes']
    assert actual['workspace_bytes'] == 0
    assert actual['pair_plane_bytes'] + 11264 <= 117976064


@pytest.mark.parametrize('kind', [True, False, None, 0, 19, '18'])
def test_unknown_type_refused_without_native_access(kind):
    with pytest.raises(ValueError):
        gpu.layout(kind)


def test_wrapper_preserves_outputs_versions_and_explicit_metadata():
    calls = []
    module = NS(pack_active=lambda *args: calls.append(args))
    prepared = gpu.Prepared(module, torch.device('cpu'))
    raw = torch.zeros(1, dtype=torch.uint8)
    picks = torch.zeros(1, dtype=torch.int64)
    outputs = tuple(torch.zeros(1, dtype=torch.uint8) for _ in range(4))
    versions = [x._version for x in outputs]
    result = prepared.pack_active(raw, raw, picks, *outputs, first=256, ggml_type=18,
                                  gate_column_offset=256, up_column_offset=512)
    assert all(a is b for a, b in zip(result, outputs))
    assert [x._version for x in outputs] == [v+1 for v in versions]
    assert calls[0][-4:] == (256, 18, 256, 512)


def test_inference_output_refusal_precedes_native_mutation():
    calls = []
    prepared = gpu.Prepared(NS(pack_active=lambda *args: calls.append(args)), torch.device('cpu'))
    raw = torch.zeros(1, dtype=torch.uint8)
    picks = torch.zeros(1, dtype=torch.int32)
    with torch.inference_mode():
        outputs = tuple(torch.zeros(1, dtype=torch.uint8) for _ in range(4))
    with pytest.raises(RuntimeError, match='inference output'):
        prepared.pack_active(raw, raw, picks, *outputs, first=0, ggml_type=23)
    assert not calls
    with torch.inference_mode():
        assert prepared.pack_active(raw, raw, picks, *outputs, first=0, ggml_type=23) == outputs
    assert len(calls) == 1


@pytest.mark.parametrize('keyword,value', [('first', True), ('first', 1), ('ggml_type', 19),
                                         ('gate_column_offset', 1), ('up_column_offset', -256)])
def test_python_refusal_preserves_every_output(keyword, value):
    prepared = gpu.Prepared(NS(pack_active=lambda *args: pytest.fail('native call')), torch.device('cpu'))
    raw = torch.zeros(1, dtype=torch.uint8)
    picks = torch.zeros(1, dtype=torch.int32)
    outputs = tuple(torch.full((1,), 177, dtype=torch.uint8) for _ in range(4))
    args = dict(first=0, ggml_type=18)
    args[keyword] = value
    with pytest.raises(ValueError):
        prepared.pack_active(raw, raw, picks, *outputs, **args)
    assert all(int(t[0]) == 177 for t in outputs)
