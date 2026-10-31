"""Packed-word decoding and boundary guards without importing CUDA runtimes."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gemma_cuda_quant_reference import decode_bytes, pack_codes

ROOT = Path(__file__).resolve().parents[1] / 'src/tensorfold/families/gemma4/cuda'


class Array(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype)


def _array(value, dtype=None):
    return np.asarray(value, dtype=dtype).view(Array)


class Pointer:
    def __init__(self, data, offset=0):
        self.data, self.offset = data, offset

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset)


def _load(pointer, mask, other):
    offsets, mask = np.broadcast_arrays(pointer.offset, mask)
    assert np.all(offsets[mask] >= 0)
    assert np.all(offsets[mask] < len(pointer.data)), 'kernel tried to read beyond its packed allocation'
    out = np.full(offsets.shape, other, dtype=np.uint32)
    out[mask] = pointer.data[offsets[mask]]
    return out.view(Array)


def _definitions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    compiled = compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), filename, 'exec')
    exec(compiled, namespace)
    return namespace


def _codes_namespace():
    return _definitions('linear.py', {'_codes'}, {'tl': SimpleNamespace(uint32=np.uint32, load=_load)})


@pytest.mark.parametrize('bits', [4, 6])
@pytest.mark.parametrize('k', [64, 704, 2816])
def test_actual_kernel_decode_matches_independent_byte_oracle_and_guards_tail(bits, k):
    rng = np.random.default_rng(311 + bits + k)
    codes = rng.integers(0, 1 << bits, (3, k), dtype=np.int64)
    # Include all values at every bit offset, with unsigned high words and a
    # nonzero final code. Reading a nonexistent next word fails in _load.
    codes[0] = np.arange(k) % (1 << bits)
    codes[-1, -1] = (1 << bits) - 1
    words = pack_codes(codes, bits)
    ns = _codes_namespace()
    got = ns['_codes'](Pointer(words.reshape(-1)), _array(np.arange(3)[:, None]),
                       _array(np.arange(k)[None, :]), _array(np.ones((3, k), dtype=bool)), words.shape[1], bits)
    expected = np.array([decode_bytes(row, bits, k) for row in words])
    assert np.array_equal(got, expected)
    assert np.array_equal(got, codes)


def test_decoder_masks_incomplete_output_tiles_before_access():
    words = pack_codes(np.full((1, 64), 63, dtype=np.int32), 6)
    ns = _codes_namespace()
    got = ns['_codes'](Pointer(words.reshape(-1)), _array([[0], [1], [-1]]), _array(np.arange(64)[None]),
                       _array([[True], [False], [False]]), words.shape[1], 6)
    assert np.array_equal(got[0], np.full(64, 63))
    assert np.array_equal(got[1:], np.zeros((2, 64)))


class Tensor:
    def __init__(self, shape, dtype='bf16', device='cuda:0', contiguous=True, cuda=True):
        self.shape = tuple(shape)
        self.dtype, self.device, self.is_cuda = dtype, device, cuda
        self._contiguous = contiguous

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return 4 if self.dtype in ('int32', 'uint32', 'fp32') else 2

    def reshape(self, *shape):
        shape = tuple(self.numel() // math.prod(d for d in shape if d != -1) if d == -1 else d for d in shape)
        return Tensor(shape, self.dtype, self.device, self._contiguous, self.is_cuda)

    def contiguous(self):
        return Tensor(self.shape, self.dtype, self.device, True, self.is_cuda)


def _namespace():
    calls = []

    def allocate(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('invalid boundary data reached a CUDA allocation')

    torch = SimpleNamespace(Tensor=Tensor, int32='int32', uint32='uint32', int64='int64', bfloat16='bf16',
                            empty=allocate)
    ns = _definitions('linear.py', {'_geometry', 'Linear'}, {'torch': torch})
    ns = _definitions('experts.py', {'ExpertLinear', 'Experts'}, ns)
    return ns, calls


def _linear(ns, n=19, k=704, e=None):
    shape = (n, k * 6 // 32) if e is None else (e, n, k * 6 // 32)
    scalar = (n, k // 64) if e is None else (e, n, k // 64)
    cls = ns['Linear'] if e is None else ns['ExpertLinear']
    return cls(Tensor(shape, 'uint32'), Tensor(scalar), Tensor(scalar), bits=6)


@pytest.mark.parametrize('bits,group', [(0, 64), (3, 64), (8, 64), (True, 64), (6.0, 64), (6, 32), (6, True)])
def test_unsupported_quantization_is_rejected_before_allocation(bits, group):
    ns, calls = _namespace()
    with pytest.raises(ValueError):
        ns['Linear'](Tensor((19, 132), 'uint32'), Tensor((19, 11)), Tensor((19, 11)), bits, group)
    assert not calls


@pytest.mark.parametrize('shape,scale_shape,rank', [((19, 131), (19, 11), 2), ((19, 132), (18, 11), 2),
                                                  ((0, 132), (0, 11), 2), ((19, 0), (19, 0), 2),
                                                  ((19, 132), (19, 11), 3),
                                                  ((9, 19, 132), (8, 19, 11), 3)])
def test_packed_geometry_must_match_declared_affine_abi(shape, scale_shape, rank):
    ns, calls = _namespace()
    with pytest.raises(ValueError):
        ns['_geometry'](shape, scale_shape, 6, 64, rank)
    assert not calls


@pytest.mark.parametrize('which,change', [('weight', {'dtype': 'bf16'}), ('scales', {'dtype': 'fp32'}),
                                         ('biases', {'dtype': 'fp32'}), ('biases', {'shape': (19, 10)}),
                                         ('weight', {'cuda': False}), ('biases', {'device': 'cuda:1'}),
                                         ('scales', {'contiguous': False})])
def test_original_tensor_representation_is_required_without_conversion(which, change):
    ns, calls = _namespace()
    values = {'weight': {'shape': (19, 132), 'dtype': 'uint32'},
              'scales': {'shape': (19, 11)}, 'biases': {'shape': (19, 11)}}
    values[which].update(change)
    with pytest.raises(ValueError):
        ns['Linear'](*(Tensor(**value) for value in values.values()), bits=6)
    assert not calls


def test_six_bit_storage_remains_original_and_counts_exact_payload_bytes():
    ns, calls = _namespace()
    values = Tensor((19, 132), 'uint32'), Tensor((19, 11)), Tensor((19, 11))
    linear = ns['Linear'](*values, bits=6)
    assert (linear.n, linear.k, linear.bits, linear.group) == (19, 704, 6, 64)
    assert all(a is b for a, b in zip(values, (linear.weight, linear.scales, linear.biases)))
    assert linear.nbytes() == 19 * 132 * 4 + 19 * 11 * 4
    assert not calls


@pytest.mark.parametrize('x', [None, Tensor(()), Tensor((0, 704)), Tensor((2, 703)),
                              Tensor((2, 704), dtype='fp32'), Tensor((2, 704), cuda=False),
                              Tensor((2, 704), device='cuda:1')])
def test_linear_input_guards_precede_gpu_allocation(x):
    ns, calls = _namespace()
    with pytest.raises(ValueError):
        _linear(ns)(x)
    assert not calls


@pytest.mark.parametrize('picks', [None, Tensor((2, 3), dtype='bf16'), Tensor((1, 3), dtype='int32'),
                                  Tensor((2, 10), dtype='int32'), Tensor((2, 0), dtype='int32'),
                                  Tensor((2, 3, 1), dtype='int32'), Tensor((2, 3), dtype='int32', cuda=False)])
def test_expert_pick_boundary_guards_precede_allocation(picks):
    ns, calls = _namespace()
    with pytest.raises(ValueError):
        _linear(ns, e=9)(Tensor((2, 704)), picks)
    assert not calls


def test_incompatible_expert_banks_and_slot_shapes_fail_before_launch():
    ns, calls = _namespace()
    gate = _linear(ns, n=128, k=192, e=9)
    up = _linear(ns, n=128, k=192, e=9)
    with pytest.raises(ValueError):
        ns['Experts'](gate, up, _linear(ns, n=192, k=192, e=9))
    down = _linear(ns, n=192, k=128, e=9)
    experts = ns['Experts'](gate, up, down)
    with pytest.raises(ValueError):
        experts.forward(Tensor((2, 192)), Tensor((2, 3), dtype='int32'), Tensor((2, 3), dtype='fp32'))
    with pytest.raises(ValueError):
        gate(Tensor((2, 4, 192)), Tensor((2, 3), dtype='int32'))
    assert not calls


def test_expert_bank_cannot_accidentally_use_the_tied_embedding_api():
    ns, calls = _namespace()
    bank = _linear(ns, e=9)
    with pytest.raises(ValueError, match='explicit expert picks'):
        bank.embedding(Tensor((2,), dtype='int64'))
    assert not calls
