"""Pinned compact Planar4 metadata and bounded canonical execution contracts."""
from types import SimpleNamespace
import json
from pathlib import Path
import struct

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.cuda.sm70 import rotorquant_compact as compact
from tensorfold.families.qwen4_exp.kv_formats import get_pair
from tensorfold.families.qwen4_exp.cuda.sm70_kv import check_pair

GOLDENS = json.loads((Path(__file__).parent / 'fixtures/rotorquant_planar4_sm70.json').read_text())


def test_compact_descriptor_is_distinct_and_charges_the_exact_two_half_metadata():
    pair = get_pair(compact.NAME)
    assert pair.row_bytes(1, 256) == 272
    assert pair.key.compact.vector_bytes_per_group == 68
    assert pair.key.compact.norm_policy == 'fp16-corrected-l2-reserved-zero'
    assert pair.key.storage_uid != get_pair('rotorquant-signed128-4').key.storage_uid
    assert pair.key.storage_uid != get_pair('rotorquant-planar4').key.storage_uid
    assert check_pair(pair, head_dim=256)


def _source_wire(payload, metadata):
    codes = payload.reshape(-1, 64)
    words = metadata.contiguous().view(torch.uint8).reshape(-1, 4)
    return b''.join(bytes(word.tolist()) + bytes(code.tolist()) for code, word in zip(codes, words))


@pytest.mark.parametrize('case', GOLDENS['fixtures'], ids=lambda case: case['name'])
def test_compact_split_storage_matches_the_independent_source_block_oracle(case):
    raw = bytes.fromhex(case['input_fp16_le_hex'])
    values = struct.unpack('<' + 'e' * (len(raw)//2), raw)
    value = torch.tensor(values).reshape(1, 1, -1)
    payload, metadata = compact.quantize_ref(value)
    wire = _source_wire(payload, metadata)
    assert wire == bytes.fromhex(case['output_blocks_hex'])
    assert wire == compact.encode_oracle(values)
    decoded = compact.dequant_ref(payload, metadata, inverse=True).flatten()
    scalar = torch.tensor(compact.decode_oracle(wire))
    # Both decoders have explicit FP32 staging; scale placement differs between
    # scalar inverse-then-scale and vector scale-then-inverse. FP16 output is the
    # actual SM70 operand contract, so check that rounding here directly.
    assert torch.equal(decoded.half(), scalar.half())
    assert len(wire) == len(values) // 128 * 68
    assert all(struct.unpack('<H', wire[i+2:i+4])[0] == 0 for i in range(0, len(wire), 68))


def test_independent_original_source_oracle_has_the_selected_pinned_provenance():
    assert GOLDENS['reference_revision'] == compact.SOURCE_REVISION
    assert GOLDENS['reference_cuda_sha256'] == compact.SOURCE_CUDA_SHA256
    assert GOLDENS['reference_constants_sha256'] == compact.SOURCE_CONSTANTS_SHA256


def test_compact_zero_and_invalid_metadata_have_explicit_meanings():
    payload, metadata = compact.quantize_ref(torch.zeros(2, 1, 128))
    assert payload.eq(0x77).all()
    assert metadata.view(torch.int32).eq(0).all()
    assert compact.dequant_ref(payload, metadata).eq(0).all()
    metadata.view(torch.int32)[0, 0, 0] = 1 << 16
    with pytest.raises(ValueError, match='reserved zero'):
        compact.dequant_ref(payload, metadata)
    with pytest.raises(ValueError, match='metadata'):
        compact.decode_oracle(struct.pack('<ee', 1., 1.) + bytes(64))
    with pytest.raises(ValueError, match='metadata'):
        compact.decode_oracle(struct.pack('<ee', 1., -0.) + bytes(64))
    with pytest.raises(ValueError, match='finite'):
        compact.quantize_ref(torch.full((1,1,128), float('nan')))
    with pytest.raises(ValueError, match='FP16'):
        compact.quantize_ref(torch.full((1,1,128), 60000.))


def test_compact_family_encode_preserves_idle_rows_and_poisoned_overflow():
    codes = torch.full((5, 1, 64), 91, dtype=torch.uint8)
    scales = torch.zeros(5, 1, 1)
    status = torch.tensor([0, -2], dtype=torch.int32)
    value = torch.ones(3, 1, 128).half()
    ops._rotor_encode(value, codes, scales, torch.tensor([2]), codec=9,
                      valid_rows=torch.tensor([1]), status=status, layer=7)
    assert codes[:2].eq(91).all() and codes[3:].eq(91).all()
    expected, words = compact.quantize_ref(value[:1])
    assert torch.equal(codes[2:3], expected)
    assert torch.equal(scales[2:3].view(torch.int32), words.view(torch.int32))
    ops._rotor_encode(torch.full((1,1,128), 60000.), codes, scales, torch.tensor([0]),
                      codec=9, status=status, layer=7)
    assert status.tolist() == [8,7]


def test_compact_attention_is_bounded_and_causal_with_partial_last_tile(monkeypatch):
    torch.manual_seed(17)
    rows, keys, heads, dim = 3, 2300, 3, 128
    q = torch.randn(rows, heads, dim).half()
    key = torch.randn(keys,1,dim).half()
    value = torch.randn_like(key)
    kc, ks = compact.quantize_ref(key)
    vc, vs = compact.quantize_ref(value)
    selected = torch.tensor([[0],[1024],[2048]], dtype=torch.int32)
    scratch = SimpleNamespace(qsa=True, sparse=torch.ones(rows,dtype=torch.int32),
                              nk=torch.ones(rows,dtype=torch.int32), ids=selected,
                              out=torch.empty(rows,heads,dim).half())
    original = ops._rotor_decode
    decoded_rows = []
    def bounded(data, metadata, width, codec):
        decoded_rows.append(data.shape[0])
        return original(data, metadata, width, codec)
    monkeypatch.setattr(ops, '_rotor_decode', bounded)
    # Rows beyond every causal end contain deliberately invalid stale bytes.
    ks[2203:].view(torch.int32).fill_(0x7fffffff)
    vs[2203:].view(torch.int32).fill_(0x7fffffff)
    output = ops.attention(q, kc, vc, torch.tensor([2200]), scratch, rows, dim**-.5,
                           ks=ks, vs=vs, bits=4, codec=9, _native_allowed=False)
    expected = compact.dequant_ref(vc[selected.flatten()], vs[selected.flatten()]).half()
    expected = compact.rotate_ref(expected.float(), inverse=True).half().expand(rows,heads,dim)
    assert torch.equal(output, expected)
    assert max(decoded_rows) <= 1024 and 252 in decoded_rows


@pytest.mark.parametrize('side', ['key', 'value', 'both'])
def test_compact_factory_refuses_non_sm70_before_engine_or_optional_drafter(monkeypatch, side):
    from tensorfold.families import qwen4_exp as family
    from tensorfold.families.qwen4_exp import gguf
    from tensorfold.families.qwen4_exp.cuda import engine
    import tensorfold.cuda.exl3.format as exl3

    monkeypatch.setattr(exl3, 'is_exl3', lambda _: False)
    monkeypatch.setattr(gguf, 'resolve_gguf', lambda _: None)
    def forbidden(*args, **kwargs):
        pytest.fail('compact factory reached modern engine ownership or optional drafter inspection')
    monkeypatch.setattr(family, 'has_mtp', forbidden)
    monkeypatch.setattr(engine, 'FlashNextEngine', forbidden)
    options = {'kv_dtype': compact.NAME} if side == 'both' else {
        'kv_dtype': 'int8', f'kv_{side}_dtype': compact.NAME}
    with pytest.raises(ValueError, match='compact Planar4.*GGUF SM70'):
        family.cuda_engine('target', **options)
