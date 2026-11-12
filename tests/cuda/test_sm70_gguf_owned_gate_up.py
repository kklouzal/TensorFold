"""Split Grouped4 GU+down must preserve the incumbent slots bit for bit."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import gguf_ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='SM70 owned GU component')


@pytest.fixture(scope='module')
def prepared():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip('SM70 owned GU component')
    device = torch.device('cuda', torch.cuda.current_device())
    return gguf_ops.prepare_owned_gate_up(device), gguf_ops.prepare_owned_from_activation(device)


@pytest.mark.parametrize('kind', [18, 21, 22, 23])
@pytest.mark.parametrize('first', [0, 256])
def test_standalone_gu_and_original_down_match_full_grouped4(prepared, kind, first):
    gu, down_call = prepared
    with np.load(Path(__file__).parents[1] / 'fixtures/gguf_mixed_ggml.npz', allow_pickle=False) as archive:
        raw = np.tile(archive[f'raw_{kind}'], (80, 2))[:, :10*{18:98,21:110,22:82,23:136}[kind]].copy()
    raw.reshape(640, 10, -1)[:, :, :2] = np.array([2**-7], dtype='<f2').view(np.uint8)
    raw_tensor = torch.from_numpy(np.tile(raw, (64, 1))).cuda()
    gate = SimpleNamespace(data=raw_tensor, ggml_type=kind, column_offset=0)
    down = SimpleNamespace(data=torch.zeros((256*2560, 180), dtype=torch.uint8, device='cuda'),
                           ggml_type=42, column_offset=0)
    rows = 9
    generator = torch.Generator(device='cuda').manual_seed(78117+kind)
    x = torch.randn((rows, 2560), generator=generator, dtype=torch.float16, device='cuda')
    picks = torch.arange(first, first+11, dtype=torch.int64, device='cuda').repeat(rows, 1)
    picks[:, 5:10] = torch.arange(256-first,261-first,device='cuda')
    picks[:, 10] = 512
    shared = torch.randn((rows, 2560), generator=generator, dtype=torch.float32, device='cuda')
    local = torch.empty_like(picks)
    act = torch.empty((rows*11,160),dtype=torch.float16,device='cuda')
    group = torch.empty(gguf_ops.grouped_workspace_words(rows*11,256),dtype=torch.int32,device='cuda')
    expected = torch.empty((rows,11,2560),dtype=torch.float32,device='cuda')
    gguf_ops.experts_owned(x,picks,gate,gate,down,first=first,experts=512,width=160,
                          shared=shared,local=local,activation=act,out=expected,grouping=group,
                          _gate_up=gguf_ops._GroupedGateUp(4))
    expected_act = act.view(rows,11,160)[:,:5].clone()
    gu.run(x,picks,gate,gate,first=first,local=local,activation=act,grouping=group)
    assert torch.equal(act.view(rows,11,160)[:,:5].view(torch.int16), expected_act.view(torch.int16))
    output = torch.empty_like(expected)
    down_call.run(x,picks,gate,gate,down,first=first,shared=shared,local=local,activation=act,out=output,grouping=group)
    assert torch.equal(output.view(torch.int32),expected.view(torch.int32))
    picks[0, 1] = picks[0, 0]
    gu.run(x,picks,gate,gate,first=first,local=local,activation=act,grouping=group)
    down_call.run(x,picks,gate,gate,down,first=first,shared=shared,local=local,activation=act,out=output,grouping=group)
    assert torch.isnan(output[0]).all()
