"""Root-only Sym8 direct operators. Run --noconftest -x after owned build.

Actual-head mode is explicit via TENSORFOLD_SYM8_MODEL_DIR; original CPU/GPU
BF16 weight owners exist only as qualified independent test oracles. No speed
claim. Fatal bad-ID probe is a SEPARATE fresh guarded process, never pytest.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open

from tensorfold.cuda import b16_sym8 as sym
from tensorfold.families.qwen3_5.cuda.b16 import matmul as reference

SMALL_GEMV_SEED = 81631
ACTUAL_HEAD_SEED = 81632


def bits_equal(actual, expected):
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def upload(weight):
    packed = sym.pack(weight, chunk_groups=512)
    status = sym.TransferStatus()
    result = sym.to_cuda(packed, status=status)
    assert status.transferred and status.safe_to_release
    return result, status



def graph_replay_bits(packed, raw_oracle, x):
    """Retain every packed/static owner until the graph is retired/destroyed.

    On any native/capture failure Root's --noconftest -x process stops without
    a cleanup CUDA retry. Only the fully successful path drains/destroys graph.
    """
    static_x = x[:1].clone()
    static_ids = torch.tensor([0, min(3, len(raw_oracle) - 1)], device='cuda', dtype=torch.int64)
    changed_ids = torch.tensor([1, min(5, len(raw_oracle) - 1)], device='cuda', dtype=torch.int64)
    # Both actual operator paths are warmed before capture on the same owners.
    warm_gemv = sym.matmul(static_x, packed)
    warm_lookup = sym.lookup(static_ids, packed, max_rows=2)
    torch.cuda.synchronize()
    del warm_gemv, warm_lookup
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        held_gemv = sym.matmul(static_x, packed)
        held_lookup = sym.lookup(static_ids, packed, max_rows=2)
    graph.replay()
    bits_equal(held_gemv, reference(static_x, raw_oracle))
    bits_equal(held_lookup, raw_oracle[static_ids])
    old_lookup = raw_oracle[static_ids].clone()
    old_gemv = reference(static_x, raw_oracle)
    static_x.mul_(-.75)
    static_ids.copy_(changed_ids)
    # Ensure this fixture is sensitive to stale captured x and ID values.
    assert not torch.equal(old_lookup.view(torch.int16), raw_oracle[static_ids].view(torch.int16))
    assert not torch.equal(old_gemv.view(torch.int16), reference(static_x, raw_oracle).view(torch.int16))
    graph.replay()
    bits_equal(held_gemv, reference(static_x, raw_oracle))
    bits_equal(held_lookup, raw_oracle[static_ids])
    torch.cuda.synchronize()
    # Packed fields, raw reference, static inputs and held outputs remain live
    # until after this explicit successful graph destruction; no future replay.
    del graph


def test_lookup_all_word_patterns_shape_scalar_empty_and_raw_fallback():
    pairs8 = [word for mag in range(1, 9) for word in (mag, mag | 0x8000)]
    pairs7 = [word for mag in range(1, 8) for word in (mag, mag | 0x8000)]
    special = [[0], pairs8, [0] + pairs7 + [8], [0] + pairs7 + [0x8008],
               [0, 1, 0x8001, 2, 0x8002], [0x7f80, 0xff80, 0x7fc1, 0xffc1, 0x7fff, 0xffff]]
    extra = [word for row in special for word in (row * ((32 + len(row) - 1) // len(row)))[:32]]
    words = np.concatenate((np.arange(65536, dtype=np.uint16), np.array(extra, dtype=np.uint16)))
    original = torch.from_numpy(words.view(np.int16).copy()).view(torch.bfloat16).reshape(-1, 32)
    cpu_packed = sym.pack(original, chunk_groups=512)
    indices = cpu_packed.metadata.numpy().view(np.uint32) >> 8
    assert np.any(indices == sym.SENTINEL) and np.any(indices != sym.SENTINEL)
    assert np.all(indices[-len(special):] == sym.SENTINEL)
    status = sym.TransferStatus()
    packed = sym.to_cuda(cpu_packed, status=status)
    raw = original.to('cuda')  # explicit oracle only
    for ids in (torch.arange(len(original), device='cuda', dtype=torch.int64),
                torch.tensor([[3, 9], [100, 2047]], device='cuda', dtype=torch.int32),
                torch.tensor(5, device='cuda', dtype=torch.int64),
                torch.empty((2, 0), device='cuda', dtype=torch.int64)):
        actual = sym.lookup(ids, packed, max_rows=ids.numel())
        bits_equal(actual, raw[ids.long()])
    torch.cuda.synchronize()
    assert status.result is packed


def test_small_gemv_preserves_native_fma_xor_bits_and_rowlaw():
    # Finite exact words with mixed signs, positive/negative zeros, compressed
    # and raw groups; exceptional payloads are checked by lookup, not GEMV.
    row = [word for magnitude in (0x3c00, 0x3c10, 0x3c20, 0x3c30, 0x3c40, 0x3c50, 0x3c60, 0x3c70)
           for word in (magnitude, magnitude | 0x8000)]
    pattern = row * 3 + [0x8000] * 16
    # Row0 retains every original fixture bit; further rows add distinct
    # mantissas so changed lookup IDs cannot pass by selecting identical rows.
    words = np.array([((word & 0x8000) | ((word & 0x7fff) + r)) if word & 0x7fff else word
                      for r in range(17) for word in pattern], dtype=np.uint16)
    original = torch.from_numpy(words.view(np.int16).copy()).view(torch.bfloat16).reshape(17, 64)
    packed, status = upload(original)
    raw = original.to('cuda')
    generator = torch.Generator(device='cuda').manual_seed(SMALL_GEMV_SEED)
    x = torch.empty((8, 64), device='cuda', dtype=torch.bfloat16).uniform_(-.5, .5, generator=generator)
    for rows in range(1, 9):
        actual = sym.matmul(x[:rows], packed)
        bits_equal(actual, reference(x[:rows], raw))
        bits_equal(actual, torch.cat([sym.matmul(x[r:r+1], packed) for r in range(rows)]))
    graph_replay_bits(packed, raw, x)
    torch.cuda.synchronize()
    assert status.safe_to_release


def test_actual_hf_head_gemv_lookup_and_cpu_streaming_bit_oracles(record_property):
    directory = os.environ.get('TENSORFOLD_SYM8_MODEL_DIR')
    if directory is None:
        pytest.skip('explicit actual pinned head oracle required')
    root = Path(directory)
    assert hashlib.sha256((root / 'config.json').read_bytes()).hexdigest() == '56ed14f6dd662fcefcc6b99b7f25c68cb4ed3978af9abf2ee179fa859877d37e'
    name = 'model.language_model.embed_tokens.weight'
    index = json.loads((root / 'model.safetensors.index.json').read_bytes())['weight_map']
    with safe_open(str(root / index[name]), framework='pt', device='cpu') as source:
        original = source.get_tensor(name)
        assert original.dtype == torch.bfloat16 and original.shape == (262144, 2816)
        cpu_packed = sym.pack(original, chunk_groups=4096)
        assert cpu_packed.nbytes == 830472192 and len(cpu_packed.raw) == 0
        for at, chunk in sym.decode_cpu_chunks(cpu_packed, chunk_groups=4096):
            bits_equal(chunk, original.reshape(-1)[at:at + chunk.numel()])
        status = sym.TransferStatus()
        packed = sym.to_cuda(cpu_packed, status=status)
        raw_oracle = original.to('cuda')
        generator = torch.Generator(device='cuda').manual_seed(ACTUAL_HEAD_SEED)
        record_property('actual_head_input_seed', ACTUAL_HEAD_SEED)
        x = torch.empty((8, 2816), device='cuda', dtype=torch.bfloat16).uniform_(-.5, .5, generator=generator)
        for rows in range(1, 9):
            bits_equal(sym.matmul(x[:rows], packed), reference(x[:rows], raw_oracle))
        ids = torch.tensor([[0, 1, 50, 106], [131071, 131072, 262142, 262143]], device='cuda')
        bits_equal(sym.lookup(ids, packed, max_rows=8), raw_oracle[ids])
        graph_replay_bits(packed, raw_oracle, x)  # actual full-head R1 + lookup capture
        record_property('actual_head_original_bytes', original.numel() * original.element_size())
        record_property('actual_sym8_allocated_bytes', packed.nbytes)
        record_property('actual_raw_fallback_groups', len(packed.raw))
        torch.cuda.synchronize()



def test_native_metadata_refusals_before_output_allocation():
    packed, status = upload(torch.ones((2, 32), dtype=torch.bfloat16, device='cpu'))
    for x in (torch.empty((0, 32), device='cuda', dtype=torch.bfloat16),
              torch.empty((9, 32), device='cuda', dtype=torch.bfloat16),
              torch.empty((1, 32), device='cuda', dtype=torch.float32),
              torch.empty((1, 32), device='cpu', dtype=torch.bfloat16)):
        with pytest.raises(ValueError):
            sym.matmul(x, packed)
    for ids, maximum in ((torch.tensor([0], device='cpu'), 1),
                         (torch.tensor([0.], device='cuda'), 1),
                         (torch.tensor([0, 1], device='cuda'), 1),
                         (torch.arange(4, device='cuda').reshape(2, 2).T, 4)):
        with pytest.raises(ValueError):
            sym.lookup(ids, packed, max_rows=maximum)
    torch.cuda.synchronize()
    assert status.safe_to_release


def fatal_probe(*, bad_metadata=False):
    # Root must launch this entry in a fresh quota/host-guarded process.
    # Device assertion is expected fatal; do not catch it or retry CUDA.
    original = torch.zeros((1, 32), dtype=torch.bfloat16, device='cpu')
    if bad_metadata:
        cpu = sym.pack(original)
        cpu.metadata[0] = 0  # raw index0 while the owned raw pool is empty
        status = sym.TransferStatus()
        packed = sym.to_cuda(cpu, status=status)
    else:
        packed, status = upload(original)
    ids = torch.tensor([0 if bad_metadata else 1], device='cuda', dtype=torch.int64)
    output = sym.lookup(ids, packed, max_rows=1)
    torch.cuda.synchronize()  # first error observation; ids/output stay frame-owned
    del output, ids  # reached only if the required fatal check failed to fire
    raise AssertionError('invalid ID/metadata did not trigger required fatal device validation')


if __name__ == '__main__':
    import sys
    if sys.argv[1:] not in (['--fatal-bad-id'], ['--fatal-bad-metadata']):
        raise SystemExit('only --fatal-bad-id or --fatal-bad-metadata is supported outside pytest')
    fatal_probe(bad_metadata=sys.argv[1] == '--fatal-bad-metadata')
