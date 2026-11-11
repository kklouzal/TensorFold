"""Exact representation and CPU-boundary tests; GPU arithmetic has separate gates."""
from dataclasses import replace

import pytest
import torch

from tensorfold.cuda.b16_lossless import pack, unpack_cpu


def bits(words):
    return torch.tensor(words, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)


def equal_bits(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_all_65536_bf16_words_including_nan_payloads_round_trip():
    original = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(1024, 64)
    encoded = pack(original, chunk_rows=17)
    assert equal_bits(unpack_cpu(encoded, chunk_blocks=29), original)
    # Sequential bit patterns cover only one or two nearby exponents per block.
    assert encoded.raw.numel() == 0
    assert encoded.nbytes == 1024 * 100


def test_mixed_raw_escape_indices_and_payload_charge_are_exact():
    compressed = [0x3f80 + (i % 64) for i in range(64)]
    raw = [0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0xffc2, 0x0001, 0x807f] * 8
    original = bits(compressed + raw + compressed + raw).reshape(2, 128)
    encoded = pack(original, chunk_rows=1)
    assert equal_bits(unpack_cpu(encoded), original)
    assert encoded.raw.shape == (2, 64)
    assert ((encoded.metadata.to(torch.int64) & 0xffffffff) >> 8).tolist() == [0xffffff, 0, 0xffffff, 1]
    assert encoded.nbytes == 4 * 100 + 2 * 128  # Includes unused96-byte raw payload slots.
    assert torch.count_nonzero(encoded.low[[1, 3]]) == 0
    assert torch.count_nonzero(encoded.delta[[1, 3]]) == 0


@pytest.mark.parametrize('span', [0, 1, 15, 16, 255])
def test_exact_exponent_span_boundary(span):
    words = [((255 - span) << 7) | 73] + [(255 << 7) | 17] * 63
    original = bits(words).reshape(1, 64)
    encoded = pack(original)
    assert equal_bits(unpack_cpu(encoded), original)
    assert len(encoded.raw) == (1 if span > 15 else 0)


def test_encoding_is_independent_of_cpu_row_chunking_and_owns_payload():
    generator = torch.Generator().manual_seed(62941)
    original = torch.randn(19, 192, generator=generator).to(torch.bfloat16)
    first, second = pack(original, chunk_rows=1), pack(original, chunk_rows=7)
    for a, b in zip((first.low, first.delta, first.metadata, first.raw),
                    (second.low, second.delta, second.metadata, second.raw)):
        assert torch.equal(a, b)
    saved = original.clone()
    original.zero_()
    assert equal_bits(unpack_cpu(first), saved)


@pytest.mark.parametrize('value', [torch.zeros(2, 64), torch.zeros(2, 63, dtype=torch.bfloat16),
                                   torch.zeros(2, 128, dtype=torch.bfloat16)[:, ::2],
                                   torch.empty(0, 64, dtype=torch.bfloat16)])
def test_invalid_pack_boundary_fails_before_encoding(value):
    with pytest.raises(ValueError):
        pack(value)


def test_corrupt_raw_index_and_compressed_delta_fail_in_cpu_decoder():
    original = bits([0x0000, 0x7f80] * 32).reshape(1, 64)
    encoded = pack(original)
    with pytest.raises(ValueError, match='raw escape index'):
        unpack_cpu(replace(encoded, metadata=torch.tensor([256], dtype=torch.int32)))
    compressed = pack(torch.zeros(1, 64, dtype=torch.bfloat16))
    wrong = compressed.delta.clone()
    wrong[0, 0] = 1
    with pytest.raises(ValueError, match='exponent delta'):
        unpack_cpu(replace(compressed, delta=wrong))
