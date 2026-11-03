"""Isolated actual CUDA word/projection oracles; no performance claim.

Root runs under the owned quota/host-floor/terminal-retaining supervisor.
--fatal probes each need a FRESH worker; the expected device failure must be
retained for exact-CID SIGKILL, never followed by CUDA cleanup/retry. Ordinary
pytest excludes those probes. Actual-head coverage additionally requires the
explicit pinned TENSORFOLD_DELTA4_MODEL_DIR; a skip is not qualification.
"""
import argparse
from dataclasses import replace
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from tensorfold.cuda import b16_delta4_vector as vector
from tensorfold.cuda import b16_delta4 as codec
from tensorfold.families.qwen3_5.cuda.b16 import matmul as raw_reference
from tests.delta4_oracle import decode_cpu_chunks


def equal_words(actual, expected):
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def upload(original):
    """Every returned destination is owned before the next transfer begins."""
    cpu = codec.pack(original, chunk_rows=512)
    fields = [None, None, None, None]
    for index, field in enumerate((cpu.low, cpu.delta, cpu.metadata, cpu.raw)):
        fields[index] = field.to(device='cuda')
    packed = codec.PackedDelta4(*fields, cpu.shape)
    torch.cuda.synchronize()
    return cpu, packed, fields


@pytest.fixture(scope='module', autouse=True)
def actual_cuda_required():
    if not torch.cuda.is_available():
        pytest.skip('actual CUDA qualification requires an isolated supported GPU worker')


def test_full_word_domain_compressed_and_raw_escape_decode_independent_cpu_oracle():
    sequential = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(1024, 64)
    # A permutation of the same literal word domain forces wide exponent spans.
    shuffled = sequential.reshape(64, 1024).T.contiguous().reshape(1024, 64)
    original = torch.cat((sequential, shuffled))
    cpu, packed, fields = upload(original)
    assert 0 < len(cpu.raw) < original.numel() // 64
    for at, chunk in decode_cpu_chunks(cpu):
        equal_words(chunk, original.reshape(-1)[at:at + chunk.numel()])
    decoded = vector.unpack(packed)
    torch.cuda.synchronize()
    equal_words(decoded.cpu(), original)
    for ids in (torch.arange(len(original), device='cuda', dtype=torch.int64),
                torch.tensor([[0, 1024], [2047, 1024]], device='cuda', dtype=torch.int32),
                torch.tensor(7, device='cuda', dtype=torch.int64),
                torch.empty((2, 0), device='cuda', dtype=torch.int64)):
        looked_up = vector.lookup(ids, packed, max_rows=max(1, ids.numel()))
        torch.cuda.synchronize()
        equal_words(looked_up.cpu(), original[ids.cpu().long()])
    assert all(field is not None for field in fields)


@pytest.mark.parametrize('k', [64, 192, 2816])
def test_ordered_fma_xor_matches_independent_raw_backend_across_rows_reuse_and_tail_columns(k):
    generator = torch.Generator().manual_seed(812344 + k)
    original = torch.randn((17, k), generator=generator).to(torch.bfloat16)
    # Finite signed-zero/subnormal escape inputs supplement ordinary blocks.
    original.view(torch.int16)[0, 0] = -32768
    original.view(torch.int16)[0, 1] = 1
    cpu, packed, fields = upload(original)
    raw = original.to(device='cuda')
    x_cpu = torch.randn((16, k), generator=generator).to(torch.bfloat16)
    x_cpu.view(torch.int16)[0, 0] = -32768
    x_cpu.view(torch.int16)[0, 1] = 1
    x = x_cpu.to(device='cuda')
    for rows in (1, 2, 3, 4, 5, 8, 16):
        expected = raw_reference(x[:rows], raw)
        for reuse in (1, 2, 4, 8, 16):
            actual = vector.matmul_rows(x[:rows], packed, rows=reuse)
            torch.cuda.synchronize()
            equal_words(actual, expected)
    assert cpu.shape == (17, k) and len(fields) == 4


def test_long_row_head_uses_same_bounded_ordered_tiles_and_partition_law():
    generator = torch.Generator().manual_seed(812346)
    original = torch.randn((17, 64), generator=generator).to(torch.bfloat16)
    _cpu, packed, fields = upload(original)
    raw = original.to('cuda')
    x = torch.randn((1024, 64), generator=generator).to(device='cuda', dtype=torch.bfloat16)
    for rows in (17, 32, 64, 128, 1024):
        expected = raw_reference(x[:rows], raw)
        for reuse in (8, 16):
            actual = vector.matmul_rows(x[:rows], packed, rows=reuse)
            torch.cuda.synchronize()
            equal_words(actual, expected)
    partition = torch.cat([vector.matmul_rows(x[:7], packed, rows=16),
                           vector.matmul_rows(x[7:17], packed, rows=16)])
    torch.cuda.synchronize()
    equal_words(partition, raw_reference(x[:17], raw))
    assert len(fields) == 4


def test_native_lookup_id_type_count_shape_refusals_and_prompt_sized_rows():
    original = torch.arange(17 * 64, dtype=torch.float32).reshape(17, 64).to(torch.bfloat16)
    _cpu, packed, fields = upload(original)
    ids = torch.arange(16384, device='cuda', dtype=torch.int64) % 17
    result = vector.lookup(ids, packed, max_rows=16384)
    torch.cuda.synchronize()
    equal_words(result.cpu(), original[ids.cpu()])
    for maximum in (0, True, -1):
        with pytest.raises(ValueError):
            vector.lookup(ids, packed, max_rows=maximum)
    with pytest.raises(RuntimeError, match='count'):
        vector.lookup(ids, packed, max_rows=16383)
    for invalid in (torch.tensor([0.], device='cuda'), torch.tensor([0]),
                    torch.arange(4, device='cuda').reshape(2, 2).T):
        with pytest.raises(RuntimeError, match='contiguous same-device INT32/INT64'):
            vector.lookup(invalid, packed, max_rows=4)
    assert len(fields) == 4


@pytest.mark.parametrize('field,offset', [('low', 1), ('delta', 1), ('raw', 1)])
def test_misaligned_contiguous_encoded_views_refused_before_output_allocation(field, offset):
    original = torch.ones((2, 64), dtype=torch.bfloat16)
    original[0, 0] = 0  # ensure a nonempty raw pool to test raw16 alignment
    _cpu, packed, fields = upload(original)
    warm = vector.unpack(packed)
    torch.cuda.synchronize()
    value = getattr(packed, field)
    storage = torch.empty(value.numel() + offset, dtype=value.dtype, device='cuda')
    misaligned = storage[offset:].view(value.shape)
    misaligned.copy_(value)
    malformed = replace(packed, **{field: misaligned})
    x = torch.ones((1, 64), dtype=torch.bfloat16, device='cuda')
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    with pytest.raises(RuntimeError, match='alignment'):
        vector.unpack(malformed)
    with pytest.raises(RuntimeError, match='alignment'):
        vector.matmul_rows(x, malformed)
    assert torch.cuda.memory_allocated() == before
    assert warm is not None and all(tensor is not None for tensor in fields)


def test_native_input_dtype_rows_shape_alignment_and_encoded_schema_refusals():
    _cpu, packed, fields = upload(torch.ones((2, 64), dtype=torch.bfloat16))
    warm = vector.unpack(packed)
    backing = torch.empty(1 + 64, device='cuda', dtype=torch.bfloat16)
    bad_alignment = backing[1:].view(1, 64)
    for x in (torch.empty((0, 64), device='cuda', dtype=torch.bfloat16),
              torch.empty((1, 64), device='cuda', dtype=torch.float32),
              torch.empty((1, 64), device='cpu', dtype=torch.bfloat16),
              torch.empty((1, 128), device='cuda', dtype=torch.bfloat16), bad_alignment):
        with pytest.raises(RuntimeError):
            vector.matmul_rows(x, packed, rows=16)
    native = vector._ext()
    with pytest.raises(RuntimeError, match='dimensions'):
        native.unpack(packed.low, packed.delta, packed.metadata, packed.raw, 2**31, 64)
    with pytest.raises(RuntimeError, match='metadata payload'):
        native.unpack(packed.low, packed.delta, packed.metadata[:-1], packed.raw, 2, 64)
    bad_delta = packed.delta[:, :-1].contiguous()
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match='delta payload'):
        native.unpack(packed.low, bad_delta, packed.metadata, packed.raw, 2, 64)
    torch.cuda.synchronize()
    assert warm is not None and len(fields) == 4


def test_actual_pinned_head_all_words_and_raw_projection_rows1_to5(record_property):
    directory = os.environ.get('TENSORFOLD_DELTA4_MODEL_DIR')
    if directory is None:
        pytest.skip('actual-head gate requires explicit pinned AEON snapshot; small tests do not qualify it')
    root = Path(directory)
    assert root.name == '90c8601b1a439497bc15d8fba9e52990b45ea0cb'
    with safe_open(str(root / 'model.safetensors'), framework='pt', device='cpu') as source:
        original = source.get_tensor('model.language_model.embed_tokens.weight')
        assert original.dtype == torch.bfloat16 and original.shape == (262144, 2816)
        cpu, packed, fields = upload(original)
        for at, chunk in decode_cpu_chunks(cpu):
            equal_words(chunk, original.reshape(-1)[at:at + chunk.numel()])
        raw = original.to(device='cuda')
        decoded = vector.unpack(packed)
        torch.cuda.synchronize()
        for start in range(0, 262144, 512):
            equal_words(decoded[start:start + 512], raw[start:start + 512])
        torch.cuda.synchronize()
        decoded = None
        generator = torch.Generator().manual_seed(812345)
        x = torch.randn((5, 2816), generator=generator).to(device='cuda', dtype=torch.bfloat16)
        for rows in range(1, 6):
            expected = raw_reference(x[:rows], raw)
            for reuse in (8, 16):
                actual = vector.matmul_rows(x[:rows], packed, rows=reuse)
                torch.cuda.synchronize()
                equal_words(actual, expected)
        record_property('original_bytes', original.numel() * original.element_size())
        record_property('packed_bytes', packed.nbytes)
        record_property('raw_escape_blocks', len(packed.raw))
        assert packed.nbytes == 1158125312 and len(fields) == 4
        torch.cuda.synchronize()


def fatal_probe(kind, operation):
    """Expected fatal metadata checks; Root supplies terminal-retaining boundary."""
    if kind in ('id-negative', 'id-upper') and operation != 'lookup':
        raise ValueError('fatal ID probes require lookup operation')
    original = torch.ones((2, 64), dtype=torch.bfloat16)
    if kind == 'raw-index':
        original[0, 0] = 0
    _cpu, packed, fields = upload(original)
    x = torch.ones((1, 64), device='cuda', dtype=torch.bfloat16)
    if kind == 'raw-index':
        packed.metadata.fill_(len(packed.raw) << 8)
    elif kind == 'delta-underflow':
        packed.metadata[0] = -256  # compressed sentinel, maximum exponent0
        packed.delta[0, 3] = 0x10  # invalid final nibble of the first eight-word group
    if operation == 'lookup':
        ids = torch.tensor([-1 if kind == 'id-negative' else 2 if kind == 'id-upper' else 0],
                           device='cuda', dtype=torch.int64)
        result = vector.lookup(ids, packed, max_rows=1)
    else:
        result = vector.unpack(packed) if operation == 'unpack' else vector.matmul_rows(x, packed, rows=16)
    torch.cuda.synchronize()  # must raise; retain result/fields until exact-CID KILL
    assert result is not None and fields
    raise AssertionError('malformed metadata did not terminate affected CUDA work')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fatal', choices=('raw-index', 'delta-underflow', 'id-negative', 'id-upper'), required=True)
    parser.add_argument('--operation', choices=('unpack', 'project', 'lookup'), required=True)
    options = parser.parse_args()
    fatal_probe(options.fatal, options.operation)
