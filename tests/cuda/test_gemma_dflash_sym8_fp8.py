"""Root-only exact privateFP8-copy equivalence; target codec/p unchanged.

Actual head optional via TENSORFOLD_SYM8_MODEL_DIR. Raw full GPU BF16 matrix is
explicit test oracle only; production factory materializes bounded chunks.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from tensorfold.cuda.b16_sym8 import pack, to_cuda, TransferStatus, Sym8Table
from tensorfold.cuda.requant import Head, requantize
from tensorfold.families.gemma4.cuda.dflash import _sym8_fp8_head


def check(original, seed):
    status = TransferStatus()
    packed = to_cuda(pack(original), status=status)
    # A small lookup_limit also forces multiple chunks for the synthetic case;
    # actual native table uses its configured262144 limit and16k source chunks.
    table = Sym8Table(packed, min(262144, max(2, original.shape[0] // 3)))
    actual = _sym8_fp8_head(table, softcap=30.)
    raw_oracle = original.to('cuda')
    expected = Head(requantize(raw_oracle, 'fp8'), 'fp8', softcap=30.)
    assert torch.equal(actual.linear.w8, expected.linear.w8)
    assert torch.equal(actual.linear.cols.view(torch.int32), expected.linear.cols.view(torch.int32))
    generator = torch.Generator(device='cuda').manual_seed(seed)
    x = torch.empty((16, original.shape[1]), device='cuda', dtype=torch.bfloat16).uniform_(-.5, .5, generator=generator)
    for rows in range(1, 17):
        result = actual(x[:rows])
        reference = expected(x[:rows])
        assert result.dtype == reference.dtype == torch.float32
        assert torch.equal(result.view(torch.int32), reference.view(torch.int32))
    ids = torch.tensor([0, original.shape[0]-1], device='cuda', dtype=torch.long)
    looked_up = table.embedding(ids)
    expected_rows = raw_oracle[ids]
    assert torch.equal(looked_up.view(torch.int16), expected_rows.view(torch.int16))
    torch.cuda.synchronize()
    assert status.safe_to_release and not hasattr(table, 'weight')


def test_small_raw_equivalent_fp8_copy_including_zero_rows():
    generator = torch.Generator().manual_seed(62811)
    weight = torch.empty((257, 64), dtype=torch.bfloat16, device='cpu').uniform_(-.5, .5, generator=generator)
    weight[0].zero_()
    check(weight, 62812)


def test_actual_pinned_head_fp8_payload_and_logits(record_property):
    directory = os.environ.get('TENSORFOLD_SYM8_MODEL_DIR')
    if directory is None:
        pytest.skip('explicit actual pinned head oracle required')
    root = Path(directory)
    assert hashlib.sha256((root/'config.json').read_bytes()).hexdigest() == '56ed14f6dd662fcefcc6b99b7f25c68cb4ed3978af9abf2ee179fa859877d37e'
    index = json.loads((root/'model.safetensors.index.json').read_bytes())['weight_map']
    name = 'model.language_model.embed_tokens.weight'
    with safe_open(str(root/index[name]), framework='pt', device='cpu') as source:
        weight = source.get_tensor(name)
        assert weight.shape == (262144, 2816) and weight.dtype == torch.bfloat16
        check(weight, 62813)
    record_property('actual_private_fp8_input_seed', 62813)
