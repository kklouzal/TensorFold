"""Host GDN schedules, pointer ownership and replay boundaries without numerical imports."""
from __future__ import annotations

import ast
import gc
from operator import index
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
import weakref

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/tensorfold/cuda/kernels/gdn.py'
FORWARD = ROOT / 'src/tensorfold/families/qwen3_5/cuda/forward.py'


class Tensor:
    def __init__(self, shape, *, dtype='fp32', device='cuda:0', address=4096, contiguous=True, stride=None):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.address, self.contiguous, self.outer_stride = address, contiguous, stride
        self.records = []

    @property
    def is_cuda(self):
        return self.device.startswith('cuda:')

    def dim(self):
        return len(self.shape)

    def numel(self):
        n = 1
        for dimension in self.shape:
            n *= dimension
        return n

    def element_size(self):
        return {'bf16': 2, 'fp32': 4, 'i32': 4, 'i64': 8}[self.dtype]

    def data_ptr(self):
        return self.address

    def is_contiguous(self):
        return self.contiguous

    def stride(self, dimension):
        if dimension == 0 and self.outer_stride is not None:
            return self.outer_stride
        return 1 if dimension == self.dim() - 1 else self.shape[1]

    def pin_memory(self):
        return self

    def to(self, device, non_blocking=False):
        self.device = device
        return self

    def record_stream(self, stream):
        self.records.append(stream)

    def view(self, *shape):
        view = Tensor(shape, dtype=self.dtype, device=self.device, address=self.address)
        view.base = self
        return view

    def unbind(self, dimension):
        assert dimension == 0
        row_bytes = self.numel() // self.shape[0] * self.element_size()
        out = []
        for row in range(self.shape[0]):
            view = Tensor(self.shape[1:], dtype=self.dtype, device=self.device, address=self.address + row * row_bytes)
            view.base = self
            out.append(view)
        return tuple(out)


def namespace():
    calls = []

    def allocation(values, dtype):
        calls.append(('copy', list(values), dtype))
        return Tensor((len(values),), dtype=dtype, device='cpu')

    names = {'_order_slots', 'schedule', 'plan_host', '_PointerValues', 'to_device', '_record_pointers', '_overlaps',
             '_check_state_writes', 'pointers', 'pointer_tables', 'replay_table'}
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    ns = {'torch': SimpleNamespace(Tensor=Tensor, float32='fp32', bfloat16='bf16', int64='i64', tensor=allocation,
                                  cuda=SimpleNamespace(current_stream=lambda device: ('consumer', device))),
          'index': index}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns, calls


def replay_inputs(*, layers=2, streams=2):
    address = 4096

    def make(shape, dtype):
        nonlocal address
        tensor = Tensor(shape, dtype=dtype, address=address)
        address += ((tensor.numel() * tensor.element_size() + 255) // 256) * 256
        return tensor

    keys = [make((7, 2, 128), 'bf16') for _ in range(layers)]
    values = [make((7, 6, 8), 'bf16') for _ in range(layers)]
    gates = [make((7, 6), 'fp32') for _ in range(layers)]
    beta = [make((7, 6), 'fp32') for _ in range(layers)]
    states = [[make((6, 8, 128), 'fp32') for _ in range(layers)] for _ in range(streams)]
    return keys, values, gates, beta, states


class GDNBoundary(unittest.TestCase):
    def test_schedule_rejects_lossy_parent_coercions_and_invalid_trees(self):
        ns, _ = namespace()
        for parents in ([], [0], [-1, -1], [-1, 1], [-1, .9], [-1, '0'], [-1, False]):
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                ns['schedule'](parents)
        for streams in ([], [[]], [[-1], []]):
            with self.subTest(streams=streams), self.assertRaises(ValueError):
                ns['plan_host'](streams)

    def test_independent_parent_state_oracle_and_plan_row_ranges(self):
        ns, _ = namespace()
        rng = random.Random(713)
        for _ in range(500):
            parents = [-1] + [rng.randrange(row) for row in range(1, rng.randrange(2, 80))]
            entries, slots = ns['schedule'](parents)
            previous, saved, visited = None, {}, []
            for offset in range(0, len(entries), 3):
                row, source, destination = entries[offset:offset + 3]
                actual = -1 if source == -1 else previous if source == -2 else saved[source]
                self.assertEqual(actual, parents[row])
                self.assertLess(max(source, destination), slots)
                if destination >= 0:
                    saved[destination] = row
                previous = row
                visited.append(row)
            self.assertEqual(sorted(visited), list(range(len(parents))))
        entries, starts, slots, maximum = ns['plan_host']([[-1, 0, 0], [-1, 0]])
        self.assertEqual(starts, [0, 3, 5])
        self.assertEqual(maximum, 3)
        self.assertEqual(sorted(entries[::3]), list(range(5)))
        self.assertGreaterEqual(slots, 1)

    def test_pointer_values_keep_owners_and_consumers_record_their_stream(self):
        ns, calls = namespace()
        owner = Tensor((8,), address=8192)
        reference = weakref.ref(owner)
        values = ns['pointers']([owner])
        self.assertEqual(values, [8192])
        self.assertIsInstance(values, list)
        del owner
        gc.collect()
        self.assertIsNotNone(reference())
        table = ns['to_device'](values, 'i64', 'cuda:0')
        del values
        gc.collect()
        self.assertIsNotNone(reference())
        owners = ns['_record_pointers'](table)
        self.assertEqual(owners[0].records, [('consumer', 'cuda:0')])
        self.assertEqual(calls, [('copy', [8192], 'i64')])
        del owners, table
        gc.collect()
        self.assertIsNone(reference())

    def test_mutated_host_pointer_values_and_bad_dtype_fail_before_upload(self):
        ns, calls = namespace()
        values = ns['pointers']([Tensor((8,))])
        with self.assertRaises(ValueError):
            ns['to_device'](values, 'i32', 'cuda:0')
        values[0] += 16
        with self.assertRaises(ValueError):
            ns['to_device'](values, 'i64', 'cuda:0')
        self.assertEqual(calls, [])

    def test_pointer_table_views_preserve_one_copy_and_group_owners(self):
        ns, calls = namespace()
        groups = [[Tensor((8,), address=4096), Tensor((8,), address=8192)],
                  [Tensor((8,), address=16384), Tensor((8,), address=32768)]]
        tables = ns['pointer_tables'](groups, 'cuda:0')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ('copy', [4096, 8192, 16384, 32768], 'i64'))
        self.assertEqual([table.shape for table in tables], [(2,), (2,)])
        for table, group in zip(tables, groups):
            self.assertEqual(table._tensorfold_gdn_owners, tuple(group))
        ns, calls = namespace()
        with self.assertRaises(ValueError):
            ns['pointer_tables']([groups[0], groups[1][:1]], 'cuda:0')
        self.assertEqual(calls, [])

    def test_exact_strided_overlap_matches_independent_byte_sets(self):
        ns, _ = namespace()
        rng = random.Random(912)
        for _ in range(2000):
            rows, width, stride = rng.randrange(1, 8), rng.randrange(1, 8), rng.randrange(0, 15)
            base, output, count = rng.randrange(80, 140), rng.randrange(60, 220), rng.randrange(1, 20)
            read = Tensor((rows, width), dtype='bf16', address=base, contiguous=False, stride=stride)
            write = Tensor((count,), dtype='fp32', address=output)
            read_bytes = {base + (row * stride + col) * 2 + byte
                          for row in range(rows) for col in range(width) for byte in range(2)}
            write_bytes = set(range(output, output + count * 4))
            self.assertEqual(ns['_overlaps'](write, read), bool(read_bytes & write_bytes))

    def test_replay_table_validates_cardinality_geometry_device_alignment_and_alias(self):
        ns, _ = namespace()
        for change in ('empty_layers', 'empty_streams', 'missing_layer', 'wrong_state_shape', 'zero_hk', 'other_device',
                       'unaligned_keys', 'wrong_gate_dtype', 'mismatched_rows'):
            args = replay_inputs()
            if change == 'empty_layers':
                args = ([], [], [], [], [[]])
            elif change == 'empty_streams':
                args = (*args[:4], [])
            elif change == 'missing_layer':
                args[-1][0].pop()
            elif change == 'wrong_state_shape':
                args[-1][0][0].shape = (6, 8)
            elif change == 'zero_hk':
                args[0][0].shape = (7, 0, 128)
            elif change == 'other_device':
                args[1][0].device = 'cuda:1'
            elif change == 'unaligned_keys':
                args[0][0].address += 2
            elif change == 'wrong_gate_dtype':
                args[2][1].dtype = 'bf16'
            else:
                args[0][1].shape = (8, 2, 128)
            with self.subTest(change=change), self.assertRaises(ValueError):
                ns['replay_table'](*args)
        args = replay_inputs()
        values = ns['replay_table'](*args)
        self.assertEqual(len(values), 12)
        self.assertEqual(values[:4], [args[i][0].data_ptr() for i in range(4)])
        args[-1][1][0] = args[-1][0][0]
        self.assertEqual(len(ns['replay_table'](*args)), 12)  # shared read-only states remain legal
        with self.assertRaises(ValueError):
            ns['replay_table'](*args, in_place=True)

    def test_forward_paths_reject_bad_input_before_pinned_copy(self):
        nodes = [node for node in ast.parse(FORWARD.read_text()).body if isinstance(node, ast.FunctionDef)
                 and node.name == '_validated_paths']
        ns = {'index': index}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(FORWARD), 'exec'), ns)
        record = [SimpleNamespace(k=Tensor((7, 2, 128)))]
        for paths in ([], [[]], [[-1]], [[7]], [[0, 2, 1]], [[0, 0]], [[0, 1.5]], [[False]],
                      [list(range(8))]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                ns['_validated_paths'](record, paths)
        self.assertEqual(ns['_validated_paths'](record, [[0, 2, 6], [1]]), [[0, 2, 6], [1]])


if __name__ == '__main__':
    unittest.main()
