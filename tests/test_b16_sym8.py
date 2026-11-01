"""Unconnected Sym8 CPU bit/format/ownership controls; no CUDA work."""
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from tensorfold.cuda import b16_sym8 as sym


def bf16(words, width=32):
    return torch.from_numpy(np.asarray(words, dtype=np.uint16).view(np.int16).copy()).view(torch.bfloat16).reshape(-1, width)


def decoded(packed):
    return torch.cat([chunk for _, chunk in sym.decode_cpu_chunks(packed, chunk_groups=3)]).view(torch.int16)


def test_all65536_word_bits_roundtrip_including_signedzero_subnormals_nan_payloads():
    original = bf16(range(65536))
    packed = sym.pack(original, chunk_groups=19)
    assert torch.equal(decoded(packed), original.view(torch.int16).reshape(-1))
    assert packed.nbytes == 36 * (original.numel() // 32) + 64 * len(packed.raw)


def test_literal_mode_codes_signs_and_raw_fallback_boundary():
    pairs8 = [word for mag in range(1, 9) for word in (mag, mag | 0x8000)]
    pairs7 = [word for mag in range(1, 8) for word in (mag, mag | 0x8000)]
    cases = [([0], False), (pairs8, False), ([0] + pairs7 + [8], False),
             ([0] + pairs7 + [0x8008], False), ([0] + pairs8, True), ([0x8000], True),
             ([0x7f80, 0xff80, 0x7fc1, 0xffc1, 0x7fff, 0xffff], False)]
    for words, escape in cases:
        row = (words * ((32 + len(words) - 1) // len(words)))[:32]
        packed = sym.pack(bf16(row), chunk_groups=1)
        assert len(packed.raw) == int(escape)
        assert torch.equal(decoded(packed), bf16(row).view(torch.int16).reshape(-1))
        assert packed.nbytes == 36 + 64 * int(escape)
        if words == [0] + pairs7 + [0x8008]:
            assert (int(packed.metadata[0]) & 255) == 1 | 2 | (8 << 2)
            assert int(packed.palette[0, 7]) == 8


def test_pack_is_chunk_invariant_and_never_mutates_original():
    words = np.tile(np.array([0, 1, 0x8001, 2, 0x8002, 0x7fc1, 0xffc1, 0x8000], dtype=np.uint16), 128)
    weight = bf16(words, 64)
    original = weight.view(torch.int16).clone()
    a, b = sym.pack(weight, chunk_groups=1), sym.pack(weight, chunk_groups=11)
    for left, right in zip((a.palette, a.codes, a.metadata, a.raw), (b.palette, b.codes, b.metadata, b.raw)):
        assert torch.equal(left, right)
    assert torch.equal(weight.view(torch.int16), original)


@pytest.mark.parametrize('weight,chunk', [(torch.ones(2, 32), 2),
    (torch.ones(32, 2, dtype=torch.bfloat16).T, 2), (torch.ones(2, 31, dtype=torch.bfloat16), 2),
    (torch.ones(2, 32, dtype=torch.bfloat16), 0), (torch.ones(2, 32, dtype=torch.bfloat16), True)])
def test_pack_invalid_metadata_refused(weight, chunk):
    with pytest.raises(ValueError):
        sym.pack(weight, chunk_groups=chunk)


def test_decoder_rejects_malformed_flags_unused_codes_and_raw_index():
    for mutate in ('flags', 'code', 'raw'):
        packed = sym.pack(bf16([0] * 32))
        if mutate == 'flags':
            packed.metadata[0] = -1  # reserved flags/count invalid
        elif mutate == 'code':
            packed.codes[0, 0] = 8  # allzero count0 cannot address mag7
        else:
            packed.metadata[0] = 0  # no raw pool exists
        with pytest.raises(ValueError):
            list(sym.decode_cpu_chunks(packed))


def test_pack_partial_allocation_error_releases_owned_large_alias(monkeypatch):
    original = torch.empty
    refs = []
    primary = MemoryError('synthetic second allocation')
    def allocate(*args, **kwargs):
        if refs:
            raise primary
        result = original(*args, **kwargs)
        refs.append(weakref.ref(result))
        return result
    weight = bf16([0] * 64)
    monkeypatch.setattr(torch, 'empty', allocate)
    with pytest.raises(MemoryError) as caught:
        sym.pack(weight)
    gc.collect()
    assert caught.value is primary
    assert refs[0]() is None  # retained primary is not an owner of the partial palette



def test_raw_pool_capacity_and_byte_product_refuse_before_allocation():
    cap = (1 << 24) - 1
    assert sym._allocation_bytes(cap, cap) == cap * 100
    for groups, raw in ((cap + 1, cap + 1), (1, 2), (1 << 63, 0), (True, 0)):
        with pytest.raises(ValueError):
            sym._allocation_bytes(groups, raw)



def test_descriptor_publication_failure_releases_partial_fields_and_formals(monkeypatch):
    original_empty = torch.empty
    refs = []
    primary = MemoryError('synthetic descriptor publication failure')
    primary.__cause__ = ValueError('original native root')
    def allocate(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        refs.append(weakref.ref(result))
        return result
    def fail_codes(owner, value):
        try:
            raise primary
        finally:
            del owner, value  # isolate the actual constructor's ownership
    monkeypatch.setattr(torch, 'empty', allocate)
    monkeypatch.setattr(sym.PackedSym8, 'codes', property(fset=fail_codes), raising=False)
    with pytest.raises(MemoryError) as caught:
        sym.pack(bf16([0] * 64))
    gc.collect()
    assert caught.value is primary and isinstance(caught.value.__cause__, ValueError)
    assert len(refs) == 4 and all(ref() is None for ref in refs)


def test_cpu_decoded_chunk_owners_remain_independent_after_iterator_advances():
    original = bf16([0] * 32 + [1, 0x8001] * 16)
    packed = sym.pack(original)
    iterator = sym.decode_cpu_chunks(packed, chunk_groups=1)
    _, first = next(iterator)
    _, second = next(iterator)
    iterator.close()
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first.view(torch.int16), original[0].view(torch.int16))
    assert torch.equal(second.view(torch.int16), original[1].view(torch.int16))
    first.fill_(1)
    assert torch.equal(second.view(torch.int16), original[1].view(torch.int16))
    assert torch.equal(decoded(packed), original.view(torch.int16).reshape(-1))


def transfer_fixture(monkeypatch, *, fail_copy=None, fail_sync=False):
    calls = []
    primary = RuntimeError('native failure')
    primary.__cause__ = ValueError('native root')
    outputs = [object() for _ in range(4)]
    class Source:
        def __init__(self, slot):
            self.slot = slot
        def to(self, *, device, non_blocking):
            assert non_blocking is True
            calls.append(('copy', self.slot))
            if fail_copy == self.slot:
                raise primary
            return outputs[self.slot]
    def sync(device):
        calls.append(('sync', str(device)))
        if fail_sync:
            raise primary
    packed = sym.PackedSym8(*(Source(i) for i in range(4)), (1, 32))
    monkeypatch.setattr(sym, '_validate', lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', sync)
    return SimpleNamespace(packed=packed, calls=calls, primary=primary, outputs=outputs)


@pytest.mark.parametrize('failure', ['copy', 'sync'])
def test_transfer_failure_retains_preallocated_owners_without_cuda_retry(monkeypatch, failure):
    fixture = transfer_fixture(monkeypatch, fail_copy=1 if failure == 'copy' else None, fail_sync=failure == 'sync')
    status = sym.TransferStatus()
    with pytest.raises(RuntimeError) as caught:
        sym.to_cuda(fixture.packed, status=status)
    assert caught.value is fixture.primary and isinstance(caught.value.__cause__, ValueError)
    assert status.started and not status.safe_to_release and not status.transferred
    assert status.source is fixture.packed and status.fields[0] is fixture.outputs[0]
    assert fixture.calls == ([('copy', 0), ('copy', 1)] if failure == 'copy'
                             else [('copy', i) for i in range(4)] + [('sync', 'cuda')])


def test_successful_transfer_syncs_once_and_publishes_original_field_owners(monkeypatch):
    fixture = transfer_fixture(monkeypatch)
    status = sym.TransferStatus()
    result = sym.to_cuda(fixture.packed, status=status)
    assert status.safe_to_release and status.transferred and status.source is None
    assert status.fields == [None] * 4 and status.result is result
    assert [result.palette, result.codes, result.metadata, result.raw] == fixture.outputs
    assert fixture.calls == [('copy', i) for i in range(4)] + [('sync', 'cuda')]
    with pytest.raises(ValueError):
        sym.to_cuda(fixture.packed, status=status)
