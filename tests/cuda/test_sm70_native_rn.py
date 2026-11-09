"""NativeRN discrete/exact-uniform and independent whole-phase certificates.

The original uniform fixtures compare unchanged native controls bitwise.
Nonuniform fixtures import the separately owned Fraction/interval oracle;
model quality, actual binary adherence and integrated P3 remain separate gates.
"""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import paged_attention as operator
from test_sm70_paged_attention import shard, expected_value, statistics

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='NativeRN requires actual SM70 component execution')


def exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


@pytest.mark.parametrize('heads', [1, 4, 12])
@pytest.mark.parametrize('format', [(8, 0), (4, 9), (4, 10)])
@pytest.mark.parametrize('group', [0, 4])
def test_uniform_exact_operator_keeps_half_global_p_and_ordered_four_peer_finish(heads, format, group):
    states = [shard(4, rank, group=group, key=format, value=format) for rank in range(4)]
    rows = 2
    receive = torch.zeros((8, rows, 12, 256), dtype=torch.float32, device='cuda')
    all_stats = receive.reshape(-1)[:8 * rows * 12 * 2].view(8, rows, 12, 2)
    all_stats[..., 0] = -torch.inf
    # Original WMMA path remains an independent same-input control here only:
    # its uniform exact fixture is not a tolerance for general NativeRN math.
    for rank, state in enumerate(states):
        all_stats[group + rank].copy_(statistics(state))
    for state in states:
        operator.merge_statistics(all_stats, state.table, state.global_stats,
                                  group_start=group, page_tokens=128, layer=state.layer)
    for state in states:
        operator.normalized_partial(state.query, state.selection, state.table, state.global_stats, state.values,
                                    group_start=group, page_tokens=128, layer=state.layer,
                                    key_bits=format[0], key_codec=format[1], value_bits=format[0], value_codec=format[1],
                                    scale=1/16)
    for rank, state in enumerate(states):
        receive[group + rank].copy_(state.values)
    for state in states:
        operator.merge_values(receive, state.table, state.out, rank=group, page_tokens=128, layer=state.layer,
                              value_bits=format[0], value_codec=format[1])
    old = [NS(local=s.values.clone(), out=s.out.clone()) for s in states]
    all_stats[..., 0] = -torch.inf
    all_stats[..., 1] = 0
    for rank, state in enumerate(states):
        operator.rn_local_statistics(state.query, state.selection, state.table, state.stats,
                                     group_start=group, page_tokens=128, layer=state.layer,
                                     key_bits=format[0], key_codec=format[1], heads_per_block=heads)
        all_stats[group + rank].copy_(state.stats)
    # ALL gathered-stat readers precede the first P write to SAME receive slab.
    for state in states:
        operator.rn_merge_statistics(all_stats, state.table, state.global_stats,
                                     group_start=group, page_tokens=128, layer=state.layer)
        exact(state.global_stats[..., 0], torch.zeros_like(state.global_stats[..., 0]))
        exact(state.global_stats[..., 1], torch.full_like(state.global_stats[..., 1], 8))
    probability = receive.view(torch.float16).reshape(-1)[:rows * 12 * 2051].view(rows, 12, 2051)
    for rank, state in enumerate(states):
        operator.rn_probabilities(state.query, state.selection, state.table, state.global_stats, probability,
                                  group_start=group, page_tokens=128, layer=state.layer,
                                  key_bits=format[0], key_codec=format[1], heads_per_block=heads)
        ids = state.selection[0, :, :10]
        owned = (ids >= 0) & (ids < 511 + torch.arange(rows, device='cuda')[:, None]) & ((ids // 128) % 4 == rank)
        expected = torch.zeros_like(probability)
        expected[:, :, :10] = torch.where(owned[:, None], .125, 0).half()
        exact(probability, expected)
        operator.rn_partial_values(probability, state.selection, state.table, state.values,
                                   page_tokens=128, layer=state.layer, value_bits=format[0], value_codec=format[1])
        exact(state.values, old[rank].local)
    # Every P reader enqueued before overwrite by gathered Float PV.
    for rank, state in enumerate(states):
        receive[group + rank].copy_(state.values)
    for rank, state in enumerate(states):
        operator.merge_values(receive, state.table, state.out, rank=group, page_tokens=128,
                              layer=state.layer, value_bits=format[0], value_codec=format[1])
        exact(state.out, old[rank].out)
        exact(state.out, expected_value(format).cuda().view(1, 1, 256).expand(rows, 3, 256))
        assert state.status.cpu().tolist() == [0, -2]


@pytest.mark.parametrize('end,count,sparse,expected_status', [
    (2048, 2048, False, 0), (2049, 2049, False, 0), (2050, 2050, False, 0), (2051, 2051, False, 0),
    (2052, 2052, False, 32), (2055, 2051, True, 0), (2055, 2052, True, 32), (2052, 2048, True, 0),
])
def test_domain_is_checked_before_half_probability_addressing_without_truncation(end, count, sparse, expected_status):
    state = shard(4, 0, rows=1)
    offsets = torch.full((32,), -1, dtype=torch.int64, device='cuda')
    state.table[:, 4] = offsets.data_ptr()
    state.table[:, 11] = 4096
    state.table[:, 12] = 0
    state.table[:, 13] = 32
    state.position.fill_(end - 1)
    state.selection = torch.zeros((8, 1, 2054), dtype=torch.int32, device='cuda')
    state.selection[0, 0, :count] = torch.arange(count, dtype=torch.int32, device='cuda')
    state.selection[0, 0, -2:] = torch.tensor([count, int(sparse)], dtype=torch.int32, device='cuda')
    receive = torch.full((8, 1, 12, 256), 37., dtype=torch.float32, device='cuda')
    p = receive.view(torch.float16).reshape(-1)[:12 * 2051].view(1, 12, 2051)
    untouched_tail = receive.view(torch.float16).reshape(-1)[12 * 2051:].clone()
    state.global_stats[..., 0] = 0
    state.global_stats[..., 1] = 1  # Another peer owns the global contribution.
    operator.rn_local_statistics(state.query, state.selection, state.table, state.stats,
                                 group_start=0, page_tokens=128, layer=state.layer,
                                 key_bits=8, key_codec=0, heads_per_block=1)
    operator.rn_probabilities(state.query, state.selection, state.table, state.global_stats, p,
                              group_start=0, page_tokens=128, layer=state.layer,
                              key_bits=8, key_codec=0, heads_per_block=1)
    operator.rn_partial_values(p, state.selection, state.table, state.values,
                               page_tokens=128, layer=state.layer, value_bits=8, value_codec=0)
    exact(p, torch.zeros_like(p))
    exact(state.values, torch.zeros_like(state.values))
    assert state.status.cpu().tolist() == [expected_status, 3 if expected_status else -2]
    # The fixed P pitch's unused receive tail is not touched by P/PV.
    exact(receive.view(torch.float16).reshape(-1)[12 * 2051:], untouched_tail)


def test_probability_schema_and_alias_refuse_before_any_launch_or_status_mutation():
    state = shard(4, 0, rows=1)
    invalid = torch.full((1, 12, 2052), -13., dtype=torch.float16, device='cuda')
    snapshot = invalid.clone()
    with pytest.raises(RuntimeError, match='2051'):
        operator.rn_probabilities(state.query, state.selection, state.table, state.global_stats, invalid,
                                  group_start=0, page_tokens=128, layer=state.layer,
                                  key_bits=8, key_codec=0, heads_per_block=1)
    exact(invalid, snapshot)
    probability = invalid.reshape(-1)[:12 * 2051].view(1, 12, 2051)
    overlapping = probability.reshape(-1).view(torch.float32)[:12 * 256].view(1, 12, 256)
    with pytest.raises(RuntimeError):
        operator.rn_partial_values(probability, state.selection, state.table, overlapping,
                                   page_tokens=128, layer=state.layer, value_bits=8, value_codec=0)
    exact(invalid, snapshot)
    assert state.status.cpu().tolist() == [0, -2]


def test_stale_generation_initializes_bounded_scratch_and_keeps_first_layer_status():
    state = shard(4, 0, rows=1)
    state.generations[1] = 1
    probability = torch.full((1, 12, 2051), 7., dtype=torch.float16, device='cuda')
    operator.rn_local_statistics(state.query, state.selection, state.table, state.stats,
                                 group_start=0, page_tokens=128, layer=state.layer,
                                 key_bits=8, key_codec=0, heads_per_block=12)
    operator.rn_probabilities(state.query, state.selection, state.table, state.global_stats, probability,
                              group_start=0, page_tokens=128, layer=state.layer,
                              key_bits=8, key_codec=0, heads_per_block=12)
    operator.rn_partial_values(probability, state.selection, state.table, state.values,
                               page_tokens=128, layer=state.layer, value_bits=8, value_codec=0)
    exact(probability, torch.zeros_like(probability))
    exact(state.values, torch.zeros_like(state.values))
    exact(state.stats[..., 0], torch.full_like(state.stats[..., 0], -torch.inf))
    exact(state.stats[..., 1], torch.zeros_like(state.stats[..., 1]))
    assert state.status.cpu().tolist() == [32, 3]


# Fixed before NativeRN target results. These are source-component fixtures,
# never model-derived gold, tuned margins, or an end-to-end quality verdict.
ENCLOSURE_CASES = (
    ('zero', 1, 1, 0, False, 'zero'),
    ('cancel31', 31, 4, 0, False, 'cancel'),
    ('full32', 32, 12, 10, False, 'rescale'),
    ('tail33', 33, 1, 0, False, 'rescale'),
    ('peer127', 127, 4, 10, True, 'peer'),
    ('full128', 128, 12, 0, False, 'cancel'),
    ('tail129', 129, 1, 10, False, 'rescale'),
    ('dense2048', 2048, 4, 0, False, 'rescale'),
    ('dense2049', 2049, 12, 10, False, 'rescale'),
    ('dense2050', 2050, 1, 0, False, 'cancel'),
    ('dense2051', 2051, 4, 10, False, 'rescale'),
    ('sparse2051-tail', 2051, 12, 0, True, 'tail'),
    ('tiny-q8', 33, 1, 0, True, 'tiny'),
    ('tiny-iso4', 33, 4, 10, True, 'tiny'),
    ('small-probability', 33, 12, 10, True, 'sharp'),
    ('planar-masks', 33, 4, 9, True, 'masks'),
    ('extreme-q-tiny-k', 33, 4, 0, False, 'extreme-q'),
    ('extreme-k-tiny-q', 33, 12, 10, False, 'extreme-k'),
    ('extreme-v', 33, 1, 0, False, 'extreme-v'),
)

# Frozen source/authority identities; self-updating hashes would authenticate an
# arbitrary changed algorithm instead of the reviewed numerical family.
ENCLOSURE_SOURCES = {
    'src/tensorfold/cuda/sm70/paged_attention.cu': 'f219a70b7e62ee73163cb1658dd8e51911db3b9c910e89d7966f7de7de29cbc4',
    'src/tensorfold/cuda/sm70/paged_attention_bindings.cpp': 'be8de6c81fb0d16628d0b756a0ac359f9bc3652437dae795d3e3179898b630ce',
    'src/tensorfold/cuda/sm70/paged_attention.py': '130013a5ea7a735ff2e4792ab6519887c8eb36f8557df96d5f80d79d1c002fd7',
    'src/tensorfold/families/qwen4_exp/cuda/owner_execution.py': '8aef86b9cc783283c68611412f922ee48e8ab79040a088bf334feb467e06ab9c',
    'src/tensorfold/families/qwen4_exp/cuda/owner_preparation.py': 'cee76e2a5fc8e53ae9509ea4f07e6d60e2ae14e91ea8cd6a2411ad974e077330',
    'src/tensorfold/families/qwen4_exp/cuda/history_owner.py': '895b294d3b0ee93750d08645694beda263edda11acb76089bd5429381b69495f',
    'src/tensorfold/families/qwen4_exp/cuda/owner_graphs.py': 'f378f20b4feb64667057be90143473f4a718b88cf2271989cad373469037920f',
    'src/tensorfold/families/qwen4_exp/cuda/adaptive_controller.py': '86121c67ba41bd162e251c0798e54799345f841bf3ad696b21fd07418d050673',
    'src/tensorfold/cuda/sm70/packed_kv_decode.cuh': 'c8657a401137cf1de3fbb80cc11373976ad2f732c55130c4ace60f54736b0d71',
    'src/tensorfold/cuda/sm70/paged_attention_descriptor.cuh': '932de7e9d521a9e123ad2133c2e97fc0104c041c1091d11a871bf8037aad1be7',
    'src/tensorfold/cuda/sm70/paged_attention_descriptor.py': '1b14e4f2743ed259924b66150baf8c178bc2811648ecb3ebe8641a6ec1c5e045',
    'tests/sm70_native_rn_oracle.py': '0867af65eb158f09170e75e4a355761dd1de275f2a6735b819cbc22648965811',
    'tests/test_sm70_native_rn_oracle.py': '7a35612ff817f69b9e931465e4044d543bb98040e75723cafd369272357619e6',
    'src/tensorfold/cuda/sm70/rotorquant_compact.py': '6757ebf92092f1f63b090a50a891e7299c643ce0baecb5046f932f0e0f0e5199',
    'src/tensorfold/cuda/sm70/rotorquant_q4_tables.py': '8cfa74d9349e4fcc2bc2947a8428ce384df4e585ce9b9498bca9bfef0fd6259d',
    'src/tensorfold/cuda/sm70/rotorquant_q4_tables.cuh': 'b1e1feb8efb035ffa7b7aefcde742705f3c1bb4ae59865afbf763ff7806667a4',
    'src/tensorfold/cuda/sm70/rotorquant_q4.cuh': '55cf44442ba425e52a9244ae559cb61937c5812e2d5e0d315e2c81334909dbf5',
}


def enclosure_authority():
    import ast
    from pathlib import Path
    import struct
    from tests import sm70_native_rn_oracle as oracle
    root = Path(__file__).resolve().parents[2]
    snapshots = {name: (root / name).read_bytes() for name in ENCLOSURE_SOURCES}
    receipt = oracle.source_binding(snapshots, ENCLOSURE_SOURCES)
    constants = ast.parse(snapshots['src/tensorfold/cuda/sm70/rotorquant_compact.py'])
    literal = next(ast.literal_eval(node.value) for node in constants.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'CENTROIDS' for target in node.targets))
    centroids = tuple(struct.unpack('<I', struct.pack('<f', value))[0] for value in literal)
    finishes = {codec: oracle.compact_finish_authority(codec, snapshots, ENCLOSURE_SOURCES) for codec in (9, 10)}
    return NS(oracle=oracle, receipt=receipt, centroids=centroids, finishes=finishes)


def _signed16(bits):
    return bits if bits < 32768 else bits - 65536


def _half_tensor(bits):
    return torch.tensor([_signed16(value) for value in bits], dtype=torch.int16).view(torch.float16)


def packed_oracle_row(authority, codec, side, mode, variant):
    """Independent authority-data decode to WORKING Half, not candidate decode.

    Q8's dyadic expression has exact Float products for these Half scales.
    Compact centroid constants are authenticated FP32 DATA; explicitly round
    the FP32 coefficient product before the final Half boundary. No native
    decoder, candidate exponential or device result supplies these operands.
    """
    from fractions import Fraction
    oracle = authority.oracle
    tiny = mode == 'tiny' or side == 'k' and mode in ('extreme-q', 'extreme-v')
    zero = mode == 'zero'
    scale_bits = (1 if codec == 0 else 8) if tiny else (0x4C00 if codec == 0 else 0x4800)  #16 or8
    if zero:
        scale_bits = 0
    if mode == 'extreme-k' and side == 'k' or mode == 'extreme-v' and side == 'v':
        scale_bits = 0x7BFF
    if side == 'v' and not tiny and not zero and mode != 'extreme-v':
        scale_bits = 0x3800 if codec == 0 else 0x4400  #1/2 or4
    scale = oracle.bits_to_fraction(scale_bits, 16)
    codes, working = [], []
    for col in range(256):
        if codec == 0:
            code = ((variant * 5 + col * 3) % 17) - 8
            if tiny:
                code = 127 if (col + variant) % 2 else -128
            if side == 'v':
                magnitude = 127 if tiny else 16 + col % 7 + 8 * (variant // 2)
                code = magnitude if variant % 2 else -magnitude - 1
                if mode == 'extreme-v':
                    magnitude = 127 if col in (0, 128) else 0
                    code = magnitude if variant % 2 else -magnitude - 1
            value = Fraction(2 * code + 1, 256) * scale
            word = oracle.round_fraction(value, 16, negative_zero=(not value and code < 0))
        else:
            code = (variant * 5 + col * 5) % 16
            if tiny:
                code = 15 if (col + variant) % 2 else 0
            if side == 'v':
                code = (15 if variant % 2 else 0) if tiny else (8 + (col + 2 * (variant // 2)) % 8 if variant % 2 else 7 - (col + 2 * (variant // 2)) % 8)
            product = oracle.bits_to_fraction(authority.centroids[code]) * scale
            product_bits = oracle.round_fraction(product, negative_zero=(not product and bool(authority.centroids[code] >> 31)))
            value = oracle.bits_to_fraction(product_bits)
            word = oracle.round_fraction(value, 16, negative_zero=(product_bits == 0x80000000))
        codes.append(code)
        working.append(word)
    packed = (tuple(codes) if codec == 0 else
              tuple(codes[col] | codes[col + 1] << 4 for col in range(0, 256, 2)))
    return packed, scale_bits, tuple(working)


def enclosure_fixture(authority, case):
    """Pure CPU frozen inputs; no observed kernel value chooses an oracle bound."""
    name, count, heads, codec, sparse, mode = case
    index = next(i for i, item in enumerate(ENCLOSURE_CASES) if item[0] == name)
    phase, group = index % 4, (index % 2) * 4
    if mode in ('peer', 'tiny', 'sharp', 'masks'):
        selected = tuple((i // 4 % 4) * 128 + (i // 16) * 4 + i % 4 for i in range(count))
        end = 512
        if mode == 'masks':
            selected = (-1, *selected[1:-2], end, end + 1)
    elif mode == 'tail':
        selected = (*range(count - 3), 4092, 4093, 4094)
        end = 4095  # Canonical2048selected+3rawtail, not a multiple-of-four dense end.
    else:
        selected, end = tuple(range(count)), count
    pages = (max(end, max(selected) + 1) + 127) // 128
    logical = pages * 128
    owners = tuple(None if not 0 <= value < end else phase ^ (value // 128 % 4) for value in selected)
    q_a = tuple((1 if col % 2 else 0x8001) if mode == 'tiny' else
                ((0x8000 if col % 2 else 0) if mode == 'zero' else
                 ((0x5400 if col == 0 else 0) if mode == 'sharp' else (0x3C00, 0xB800, 0x3400, 0xB000)[col % 4]))
                for col in range(256))
    if mode in ('extreme-q', 'extreme-k'):
        magnitude = 0x7BFF if mode == 'extreme-q' else 1
        q_a = tuple(magnitude | (0x8000 if col % 2 else 0) for col in range(256))
    q_b = tuple(value ^ 0x8000 for value in q_a)
    row_cache = {}
    def row(value, side):
        page_tag = value // 512 % 3
        variant = (((value // 128) % 4 if mode in ('peer', 'sharp') else value // 32 % 4)
                   + 4 * page_tag) if side == 'k' else value % 2 + 2 * page_tag
        key = side, variant
        if key not in row_cache:
            row_cache[key] = packed_oracle_row(authority, codec, side, mode, variant)
        return row_cache[key]
    keys = tuple((0x7E00,) * 256 if owner is None else row(value, 'k')[2] for value, owner in zip(selected, owners))
    values = tuple((0x7E00,) * 256 if owner is None else row(value, 'v')[2] for value, owner in zip(selected, owners))
    packed = []
    for peer in range(4):
        # Reverse owned-page physical placement, preserving the independent
        # canonical logical ownership map and testing relocatable descriptors.
        owned_pages = list(reversed([page for page in range(pages) if phase ^ (page % 4) == peer]))
        offsets = tuple(owned_pages.index(page) * 128 if page in owned_pages else -1 for page in range(pages))
        physical = len(owned_pages) * 128
        columns = 256 if codec == 0 else 128
        data_k = torch.full((physical, 1, columns), 127, dtype=torch.int8 if codec == 0 else torch.uint8)
        data_v = torch.full_like(data_k, 127)
        scales_k = torch.empty((physical, 1, 8 if codec == 0 else 2), dtype=torch.float16 if codec == 0 else torch.float32)
        scales_v = torch.empty_like(scales_k)
        scales_k.view(torch.int16 if codec == 0 else torch.int32).fill_(0x7E00)  #poison unused/future payload
        scales_v.view(torch.int16 if codec == 0 else torch.int32).fill_(0x7E00)
        for value, owner in zip(selected, owners):
            if owner != peer:
                continue
            at = offsets[value // 128] + value % 128
            for side, data, scales in (('k', data_k, scales_k), ('v', data_v, scales_v)):
                raw, scale, _ = row(value, side)
                data[at, 0].copy_(torch.tensor(raw, dtype=data.dtype))
                scales.view(torch.int16 if codec == 0 else torch.int32)[at].fill_(_signed16(scale) if codec == 0 else scale)
        packed.append(NS(k=data_k, v=data_v, ks=scales_k, vs=scales_v, offsets=offsets, physical=physical))
    return NS(name=name, count=count, heads=heads, codec=codec, sparse=sparse, mode=mode, group=group,
              phase=phase, selected=selected, end=end, logical=logical, owners=owners,
              queries=(q_a, q_b), keys=keys, values=values, packed=packed)


def _phase_value(bits, width, bound, label):
    from tests.sm70_native_rn_oracle import bits_to_fraction
    value = bits_to_fraction(bits, width)  #nonfinite observations are failures
    assert bound.lo <= value <= bound.hi, (label, hex(bits), str(value), str(bound.lo), str(bound.hi))
    return bound.hi - bound.lo


def enclosure_input_metrics(authority, fixture, reports):
    from fractions import Fraction
    oracle = authority.oracle
    maximum = Fraction()
    infinite = False
    unique = set(key for key, owner in zip(fixture.keys, fixture.owners) if owner is not None)
    for query in fixture.queries:
        q = tuple(oracle.bits_to_fraction(value, 16) for value in query)
        for key in unique:
            products = tuple(a * oracle.bits_to_fraction(b, 16) for a, b in zip(q, key))
            total, absolute = sum(products, Fraction()), sum(map(abs, products), Fraction())
            if not total:
                infinite = infinite or bool(absolute)
            else:
                maximum = max(maximum, absolute / abs(total))
    return {'qk_max_finite_condition': str(maximum), 'qk_exact_cancellation': infinite,
            'ideal_PV_coordinates_including_zero': sum(bound.lo <= 0 <= bound.hi for report in reports
                                                       for bound in report['ideal']['exact_pv']),
            'interpretation': 'input conditioning only; no bound/tolerance or model acceptance chosen from results'}


@pytest.fixture(scope='module')
def certified_enclosure_authority():
    return enclosure_authority()


@pytest.mark.parametrize('case', ENCLOSURE_CASES, ids=[case[0] for case in ENCLOSURE_CASES])
def test_whole_native_phases_fit_independent_decoded_operator_enclosures(case, record_property, certified_enclosure_authority):
    import hashlib
    import json
    authority = certified_enclosure_authority
    fixture = enclosure_fixture(authority, case)
    oracle = authority.oracle
    finish = authority.finishes.get(fixture.codec)
    # Two immutable queries alternate over all twelve REAL heads. No output or
    # observed error participates in certificate construction. UNRESOLVED raises.
    reports = tuple(oracle.attention_head_enclosure(q, fixture.keys, fixture.values, fixture.owners,
                    sparse=fixture.sparse, finish='q8-h32' if fixture.codec == 0 else 'compact-linear',
                    codec_rows=None if finish is None else finish['coefficient_rows'],
                    source_receipt=authority.receipt) for q in fixture.queries)
    if fixture.mode == 'sharp':
        from fractions import Fraction
        assert any(0 < bound.lo <= bound.hi < Fraction(1, 2**14)
                   for report in reports for bound in report['half_probabilities'])
    if fixture.mode == 'tiny':
        assert any(0 < (value & 0x7FFF) < 0x400 for q in fixture.queries for value in q)
        assert any(0 < (value & 0x7FFF) < 0x400 for key, owner in zip(fixture.keys, fixture.owners)
                   if owner is not None for value in key)
        assert any(0 < (value & 0x7FFF) < 0x400 for v, owner in zip(fixture.values, fixture.owners)
                   if owner is not None for value in v)
    input_metrics = enclosure_input_metrics(authority, fixture, reports)
    states = []
    query = torch.full((8, 1, 3, 256), torch.nan, dtype=torch.float16, device='cuda')
    for head in range(12):
        query[fixture.group + head // 3, 0, head % 3].copy_(_half_tensor(fixture.queries[head % 2]).cuda())
    selected = torch.zeros((8, 1, 2054), dtype=torch.int32, device='cuda')
    selected[0, 0, :fixture.count] = torch.tensor(fixture.selected, dtype=torch.int32, device='cuda')
    selected[0, 0, -2:] = torch.tensor([fixture.count, int(fixture.sparse)], dtype=torch.int32, device='cuda')
    for packed in fixture.packed:
        state = NS(**{name: getattr(packed, name).cuda() for name in ('k', 'v', 'ks', 'vs')})
        state.offsets = torch.tensor(packed.offsets, dtype=torch.int64, device='cuda')
        state.generations = torch.zeros(2, dtype=torch.int64, device='cuda')
        state.status = torch.tensor([0, -2], dtype=torch.int32, device='cuda')
        state.position = torch.tensor([fixture.end - 1], dtype=torch.int32, device='cuda')
        state.live = torch.tensor([1], dtype=torch.int32, device='cuda')
        state.layer = torch.tensor([3], dtype=torch.int32, device='cuda')
        state.table = torch.tensor([[state.k.data_ptr(), state.v.data_ptr(), state.ks.data_ptr(), state.vs.data_ptr(),
            state.offsets.data_ptr(), state.generations.data_ptr(), state.status.data_ptr(), state.position.data_ptr(),
            state.live.data_ptr(), 0, 0, fixture.logical, packed.physical, len(packed.offsets), 0, 0]], dtype=torch.int64, device='cuda')
        state.stats = torch.empty((1, 12, 2), dtype=torch.float32, device='cuda')
        state.global_stats = torch.empty_like(state.stats)
        state.values = torch.empty((1, 12, 256), dtype=torch.float32, device='cuda')
        state.out = torch.empty((1, 3, 256), dtype=torch.float16, device='cuda')
        states.append(state)
    receive = torch.empty((8, 1, 12, 256), dtype=torch.float32, device='cuda')
    gathered = receive.reshape(-1)[:8 * 12 * 2].view(8, 1, 12, 2)
    gathered[..., 0] = -torch.inf
    gathered[..., 1] = 0
    widths = {name: 0 for name in ('local_ML', 'global_ML', 'P', 'PV', 'fold', 'finish')}
    checks = {name: 0 for name in widths}
    def check(bits, width, bound, phase, label):
        widths[phase] = max(widths[phase], _phase_value(bits, width, bound, (fixture.name, phase, label)))
        checks[phase] += 1
    for peer, state in enumerate(states):
        operator.rn_local_statistics(query, selected, state.table, state.stats,
            group_start=fixture.group, page_tokens=128, layer=state.layer,
            key_bits=8 if fixture.codec == 0 else 4, key_codec=fixture.codec, heads_per_block=fixture.heads)
        observed = state.stats.cpu().view(torch.int32)[0].tolist()
        for head in range(12):
            maximum, mass = reports[head % 2]['local_statistics'][peer]
            if maximum is None:
                assert observed[head] == [-8388608, 0]  #raw(-inf,+0)
            else:
                check(observed[head][0] & 0xFFFFFFFF, 32, maximum, 'local_ML', (peer, head, 'M'))
                check(observed[head][1] & 0xFFFFFFFF, 32, mass, 'local_ML', (peer, head, 'L'))
        gathered[fixture.group + peer].copy_(state.stats)
    # ALL statistics readers finish on this stream before P reuses receive.
    for peer, state in enumerate(states):
        operator.rn_merge_statistics(gathered, state.table, state.global_stats,
                                     group_start=fixture.group, page_tokens=128, layer=state.layer)
        observed = state.global_stats.cpu().view(torch.int32)[0].tolist()
        for head in range(12):
            for component in (0, 1):
                check(observed[head][component] & 0xFFFFFFFF, 32, reports[head % 2]['global_statistics'][component],
                      'global_ML', (peer, head, component))
    probability = receive.view(torch.float16).reshape(-1)[:12 * 2051].view(1, 12, 2051)
    for peer, state in enumerate(states):
        operator.rn_probabilities(query, selected, state.table, state.global_stats, probability,
            group_start=fixture.group, page_tokens=128, layer=state.layer,
            key_bits=8 if fixture.codec == 0 else 4, key_codec=fixture.codec, heads_per_block=fixture.heads)
        observed = probability.cpu().view(torch.int16)[0].tolist()
        for head in range(12):
            for at, owner in enumerate(fixture.owners):
                if owner == peer:
                    check(observed[head][at] & 0xFFFF, 16, reports[head % 2]['half_probabilities'][at], 'P', (peer, head, at))
                else:
                    assert observed[head][at] == 0  #mask/foreign zeros are a RAW discrete contract
            assert all(value == 0 for value in observed[head][fixture.count:])
        operator.rn_partial_values(probability, selected, state.table, state.values, page_tokens=128, layer=state.layer,
                                   value_bits=8 if fixture.codec == 0 else 4, value_codec=fixture.codec)
        observed = state.values.cpu().view(torch.int32)[0].tolist()
        for head in range(12):
            for col, value in enumerate(observed[head]):
                check(value & 0xFFFFFFFF, 32, reports[head % 2]['local_values'][peer][col], 'PV', (peer, head, col))
    # Observation of the specified three RN pointwise peer additions, not an
    # oracle: every bound above/below came from original working-Half inputs.
    folded = states[0].values.clone()
    for peer in (1, 2, 3):
        folded.add_(states[peer].values)
    observed = folded.cpu().view(torch.int32)[0].tolist()
    for head in range(12):
        for col, value in enumerate(observed[head]):
            check(value & 0xFFFFFFFF, 32, reports[head % 2]['peer_fold'][col], 'fold', (head, col))
    # Every P reader precedes the actual gathered-Float overwrite.
    for peer, state in enumerate(states):
        receive[fixture.group + peer].copy_(state.values)
    for peer, state in enumerate(states):
        operator.merge_values(receive, state.table, state.out, rank=fixture.group + peer,
            page_tokens=128, layer=state.layer, value_bits=8 if fixture.codec == 0 else 4, value_codec=fixture.codec)
        observed = state.out.cpu().view(torch.int16)[0].tolist()
        for local_head in range(3):
            head = peer * 3 + local_head
            for col, value in enumerate(observed[local_head]):
                check(value & 0xFFFF, 16, reports[head % 2]['finished_half_values'][col], 'finish', (head, col))
        assert state.status.cpu().tolist() == [0, -2]
    # The exactdecoded mathematical ideal is INDEPENDENT of the RN candidate.
    # Report finite analytic widths, never fit a margin to these observations.
    assert all(report['global_statistics'][1].lo > 0 for report in reports)
    encoded = json.dumps({'case': fixture.name, 'N': fixture.count, 'heads_per_block': fixture.heads,
        'codec': fixture.codec, 'group': fixture.group, 'phase': fixture.phase, 'checks': checks,
        'max_analytic_widths': {name: str(value) for name, value in widths.items()},
        'oracle_work_bound_per_head': [report['work_bound'] for report in reports],
        'input_conditioning': input_metrics,
        'source_receipt': authority.receipt, 'codec_descriptor': None if finish is None else finish['descriptor'],
        'input_sha256': hashlib.sha256(repr((fixture.queries, fixture.keys, fixture.values, fixture.owners,
                                           fixture.selected, fixture.end)).encode()).hexdigest(),
        'fold_observation': 'three GPU Float adds of measured localPV; native internal fold is covered by finish enclosure',
        'scope': 'source component phase-value certificate; no signedzero/model/P3 qualification'}, sort_keys=True)
    record_property('native_rn_phase_enclosures', encoded)



def test_global_empty_is_fatal_not_an_enclosure_pass(certified_enclosure_authority):
    authority = certified_enclosure_authority
    with pytest.raises(ValueError, match='globally empty'):
        authority.oracle.attention_head_enclosure([0] * 256, [[0x7E00] * 256] * 10,
                                                 [[0x7E00] * 256] * 10, [None] * 10)
    states = [shard(4, peer, rows=1) for peer in range(4)]
    receive = torch.zeros((8, 1, 12, 256), dtype=torch.float32, device='cuda')
    gathered = receive.reshape(-1)[:8 * 12 * 2].view(8, 1, 12, 2)
    gathered[..., 0] = -torch.inf
    for peer, state in enumerate(states):
        state.page_offsets.fill_(-1)
        operator.rn_local_statistics(state.query, state.selection, state.table, state.stats,
            group_start=0, page_tokens=128, layer=state.layer, key_bits=8, key_codec=0, heads_per_block=4)
        gathered[peer].copy_(state.stats)
    for state in states:
        operator.rn_merge_statistics(gathered, state.table, state.global_stats,
                                     group_start=0, page_tokens=128, layer=state.layer)
        assert state.status.cpu().tolist() == [16, 3]
    probability = receive.view(torch.float16).reshape(-1)[:12 * 2051].view(1, 12, 2051)
    for state in states:
        operator.rn_probabilities(state.query, state.selection, state.table, state.global_stats, probability,
            group_start=0, page_tokens=128, layer=state.layer, key_bits=8, key_codec=0, heads_per_block=4)
        operator.rn_partial_values(probability, state.selection, state.table, state.values,
                                   page_tokens=128, layer=state.layer, value_bits=8, value_codec=0)
        exact(probability, torch.zeros_like(probability))
        exact(state.values, torch.zeros_like(state.values))
        assert state.status.cpu().tolist() == [16, 3]



@pytest.mark.parametrize('maximum,denominator', [(-1., 1.), (float('nan'), 1.),
                                                (float('inf'), 1.), (0., 0.)])
def test_bad_global_normalization_latches_failure_without_accepting_repaired_output(maximum, denominator):
    state = shard(4, 0, rows=1)
    state.global_stats[..., 0] = maximum
    state.global_stats[..., 1] = denominator
    probability = torch.full((1, 12, 2051), 17., dtype=torch.float16, device='cuda')
    operator.rn_probabilities(state.query, state.selection, state.table, state.global_stats, probability,
        group_start=0, page_tokens=128, layer=state.layer, key_bits=8, key_codec=0, heads_per_block=1)
    operator.rn_partial_values(probability, state.selection, state.table, state.values,
                               page_tokens=128, layer=state.layer, value_bits=8, value_codec=0)
    exact(probability, torch.zeros_like(probability))
    exact(state.values, torch.zeros_like(state.values))
    assert state.status.cpu().tolist() == [16, 3]  # Defined FAILED scratch, never an accepted answer.
