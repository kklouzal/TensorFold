"""Owned CPU layout/oracle/transfer controls; no physical CUDA work here."""
from dataclasses import replace
import gc
import weakref

import pytest
import torch

from tensorfold.cuda import b16_delta4 as codec
from tests.delta4_oracle import decode_cpu_chunks


def decoded(packed, groups=7):
    return torch.cat([chunk for _, chunk in decode_cpu_chunks(packed, chunk_groups=groups)]).reshape(packed.shape)


def equal_words(left, right):
    assert left.dtype == right.dtype == torch.bfloat16 and left.shape == right.shape
    assert torch.equal(left.view(torch.int16), right.view(torch.int16))


def test_all65536_literal_words_and_shuffled_raw_blocks_preserve_bits():
    sequential = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(1024, 64)
    mixed = torch.cat((sequential, sequential.reshape(64, 1024).T.contiguous().reshape(1024, 64)))
    packed = codec.pack(mixed, chunk_rows=13)
    equal_words(decoded(packed), mixed)
    assert 0 < len(packed.raw) < mixed.numel() // 64
    assert packed.nbytes == 100 * (mixed.numel() // 64) + 128 * len(packed.raw)


def test_chunking_preserves_exact_fields_and_descriptor_does_not_borrow_original():
    generator = torch.Generator().manual_seed(614244)
    original = torch.randn((17, 192), generator=generator).to(torch.bfloat16)
    first, second = codec.pack(original, chunk_rows=1), codec.pack(original, chunk_rows=7)
    for name in ('low', 'delta', 'metadata', 'raw'):
        assert torch.equal(getattr(first, name), getattr(second, name))
    reference = original.clone()
    original.zero_()
    equal_words(decoded(first), reference)


@pytest.mark.parametrize('span', [0, 15, 16, 255])
def test_span_boundary_uses_raw_only_beyond15(span):
    words = [((255 - span) << 7) | 73] + [(255 << 7) | 17] * 63
    original = torch.tensor(words, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(1, 64)
    packed = codec.pack(original)
    equal_words(decoded(packed), original)
    assert len(packed.raw) == (1 if span > 15 else 0)


def test_header_bounds_never_assume_measured_escape_count_or_compression_gain():
    bounds = codec.header_bounds((262144, 2816))
    assert bounds['groups'] == 11534336 and bounds['raw_source_bytes'] == 1476395008
    assert bounds['max_valid_encoded_bytes'] == 2629828608
    assert bounds['max_valid_encoded_bytes'] > bounds['raw_source_bytes']
    assert bounds['pack_chunk_rows'] == 512
    assert bounds['pack_temporary_bytes'] == sum(bounds['pack_temporary_inventory'].values())
    assert bounds['pack_temporary_inventory']['previous_encode_chunk_owners'] > 0


@pytest.mark.parametrize('shape', [(0, 64), (1, 63), (True, 64), (2**31, 64), (1, 2**31), [1, 64]])
def test_invalid_geometry_is_refused_before_allocation(shape):
    with pytest.raises(ValueError):
        codec.geometry(shape)


@pytest.mark.parametrize('groups,raw_count', [(4, 5), (1, -1), (True, 0),
    (codec.SENTINEL + 1, codec.SENTINEL + 1), (codec.MAX_I64, 0)])
def test_encoded_byte_and24bit_pool_bounds_are_checked(groups, raw_count):
    with pytest.raises(ValueError):
        codec.allocation_bytes(groups, raw_count)


@pytest.mark.parametrize('original', [torch.zeros((2, 64)), torch.zeros((2, 63), dtype=torch.bfloat16),
    torch.zeros((0, 64), dtype=torch.bfloat16), torch.zeros((2, 128), dtype=torch.bfloat16)[:, ::2]])
def test_original_source_refusals_precede_encoding(original):
    with pytest.raises(ValueError):
        codec.pack(original)


def test_schema_and_corrupt_shape_refusals():
    packed = codec.pack(torch.ones((2, 64), dtype=torch.bfloat16))
    for invalid in (replace(packed, schema='different'), replace(packed, delta=packed.delta[:, :-1]),
                    replace(packed, raw=packed.raw.to(torch.int32))):
        with pytest.raises(ValueError):
            codec._validate(invalid, cuda=False)


def test_failed_block_conversion_releases_source_view_without_erasing_native_primary(monkeypatch):
    original = torch.ones((2, 64), dtype=torch.bfloat16)
    borrowed = weakref.ref(original)
    primary = MemoryError('block conversion allocation')
    native = OSError('original native root')
    primary.__cause__ = native

    def fail(source, *args, **kwargs):
        try:
            raise primary
        finally:
            del source  # the injected failure must not own its borrowed view

    monkeypatch.setattr(torch.Tensor, 'to', fail)
    caught = None
    try:
        codec.pack(original)
    except MemoryError as error:
        caught = error
    del original
    gc.collect()
    assert caught is primary and primary.__cause__ is native and primary.__traceback__ is not None
    assert borrowed() is None


def test_partial_pack_allocation_releases_large_cpu_aliases(monkeypatch):
    original_empty = torch.empty
    references = []
    primary = MemoryError('second field allocation')
    original = torch.ones((2, 64), dtype=torch.bfloat16)
    source = weakref.ref(original)

    def allocate(*args, **kwargs):
        if references:
            raise primary
        value = original_empty(*args, **kwargs)
        references.append(weakref.ref(value))
        return value

    monkeypatch.setattr(torch, 'empty', allocate)
    caught = None
    try:
        codec.pack(original)
    except MemoryError as error:
        caught = error
    del original
    gc.collect()
    assert caught is primary and references[0]() is None and source() is None


def test_late_descriptor_publication_clears_partialself_formals_and_pack_temporaries(monkeypatch):
    original_empty = torch.empty
    references, partial = [], []
    primary = MemoryError('late descriptor publication')
    native = OSError('native descriptor root')
    primary.__cause__ = native
    original = torch.ones((2, 64), dtype=torch.bfloat16)
    source = weakref.ref(original)

    def allocate(*args, **kwargs):
        value = original_empty(*args, **kwargs)
        references.append(weakref.ref(value))
        return value

    def fail(owner, value):
        try:
            partial.append(weakref.ref(owner))
            raise primary
        finally:
            del owner, value  # isolate constructor ownership, not the fault injector

    monkeypatch.setattr(torch, 'empty', allocate)
    monkeypatch.setattr(codec.PackedDelta4, 'raw', property(fset=fail), raising=False)
    caught = None
    try:
        codec.pack(original)
    except MemoryError as error:
        caught = error
    del original
    gc.collect()
    assert caught is primary and primary.__cause__ is native
    assert source() is None and len(references) == 4 and all(reference() is None for reference in references)
    assert partial[0]() is not None and object.__getattribute__(partial[0](), '__dict__') == {}


@pytest.mark.parametrize('failure_at', ['copy', 'fence', 'descriptor'])
def test_transfer_failure_retains_actual_status_owners_and_never_retries(monkeypatch, failure_at):
    original = torch.ones((2, 64), dtype=torch.bfloat16)
    packed = codec.pack(original)
    status = codec.TransferStatus()
    primary = RuntimeError('native transfer failure')
    primary.__cause__ = ValueError('original native root')
    copies, fences = [], []

    def copy(source, *args, **kwargs):
        if failure_at == 'copy' and len(copies) == 1:
            raise primary
        destination = object()
        copies.append(destination)
        return destination

    def fence(device):
        fences.append(device)
        if failure_at == 'fence':
            raise primary

    monkeypatch.setattr(torch.Tensor, 'to', copy)
    monkeypatch.setattr(torch.cuda, 'synchronize', fence)
    if failure_at == 'descriptor':
        def fail_descriptor(self, *args, **kwargs):
            raise primary
        monkeypatch.setattr(codec.PackedDelta4, '__init__', fail_descriptor)
    with pytest.raises(RuntimeError) as caught:
        codec.to_cuda(packed, status=status)
    assert caught.value is primary and isinstance(primary.__cause__, ValueError)
    assert status.started and not status.transferred and status.source is packed
    assert status.result is None and status.fields[:len(copies)] == copies
    assert len(fences) == (0 if failure_at == 'copy' else 1)
    assert status.safe_to_release == (failure_at == 'descriptor')


def test_successful_transfer_publishes_one_owned_result_after_fence(monkeypatch):
    packed = codec.pack(torch.ones((2, 64), dtype=torch.bfloat16))
    status = codec.TransferStatus()
    sources, destinations = [], []

    def copy(source, *args, **kwargs):
        sources.append(source)
        destination = object()
        destinations.append(destination)
        return destination

    def fence(device):
        assert status.source is packed and status.result is None
        assert status.fields == destinations and len(destinations) == 4

    monkeypatch.setattr(torch.Tensor, 'to', copy)
    monkeypatch.setattr(torch.cuda, 'synchronize', fence)
    result = codec.to_cuda(packed, status=status)
    assert status.safe_to_release and status.transferred and status.result is result and status.source is None
    assert (result.low, result.delta, result.metadata, result.raw) == tuple(destinations)
    assert sources == [packed.low, packed.delta, packed.metadata, packed.raw]
    with pytest.raises(ValueError, match='fresh caller-owned'):
        codec.to_cuda(packed, status=status)
