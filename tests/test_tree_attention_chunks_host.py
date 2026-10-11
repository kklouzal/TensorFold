"""Tree plans preserve one ordered partial per absolute-position key chunk."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest


def _policy():
    path = Path(__file__).resolve().parents[1] / "src/tensorfold/cuda/kernels/attention.py"
    source = ast.parse(path.read_text())
    constants = {"CHUNK", "MAX_NODES", "QUERY_TILE"}
    methods = {"slots", "plan_host", "padded_host"}
    body = [node for node in source.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in constants for target in node.targets)]
    body.extend(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in methods)
    namespace = {"Sequence": Sequence}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def _count_chunks(length, chunk):
    whole, tail = divmod(length, chunk)
    return whole + bool(tail)


def _writers(policy, flat, width, rows, p, items, group):
    count = _count_chunks(p + rows, policy.CHUNK)
    assert flat[width:width + 4] == [0, rows, p, count]
    programs = [tuple(flat[width + 4 + item * 3:width + 7 + item * 3]) for item in range(items)]
    assert len(programs) == len(set(programs))
    shared = {(stream, first, chunk) for stream, first, chunk in programs
              if (chunk + 1) * policy.CHUNK <= p}
    expected = {(0, first, chunk) for chunk in range(p // policy.CHUNK)
                for first in range(0, rows * group, policy.QUERY_TILE)}
    assert shared == expected
    committed = {chunk for _, _, chunk in shared}
    tail = set(range(p // policy.CHUNK, count))
    assert committed.isdisjoint(tail)
    assert committed | tail == set(range(count))
    return shared, tail


@pytest.mark.parametrize("group", [1, 6, 16])
@pytest.mark.parametrize("rows", [1, 3, 16, 64, 128])
@pytest.mark.parametrize("p", [0, 1, 511, 512, 513, 2047, 2048, 4095, 4096, 8192, 9000, 32768])
def test_exact_and_padded_plans_write_each_chunk_and_query_tile_once(group, rows, p):
    policy = _policy()
    parents = list(range(-1, rows - 1))
    flat, items, capacity = policy.plan_host([parents], [p], group)
    exact = _writers(policy, flat, rows, rows, p, items, group)
    assert capacity == policy.slots(p, rows) == _count_chunks(p + rows, policy.CHUNK)
    context = _count_chunks(p + rows, 2048) * 2048
    padded, items, capacity = policy.padded_host(parents, context, group)
    padded[rows + 2], padded[rows + 3] = p, policy.slots(p, rows)
    assert _writers(policy, padded, rows, rows, p, items, group) == exact
    assert capacity == _count_chunks(context + rows, policy.CHUNK)
    assert capacity >= policy.slots(p, rows)


def test_multi_stream_plan_keeps_local_chunks_and_global_parent_offsets():
    policy = _policy()
    widths, lengths, group = [16, 1, 128], [32768, 9000, 2047], 6
    trees = [list(range(-1, width - 1)) for width in widths]
    flat, items, most = policy.plan_host(trees, lengths, group)
    width = sum(widths)
    starts = [0, 16, 17]
    assert flat[:width] == [stream for stream, rows in enumerate(widths) for _ in range(rows)]
    streams = [flat[width + 4 * index:width + 4 * (index + 1)] for index in range(3)]
    counts = [_count_chunks(p + rows, policy.CHUNK) for p, rows in zip(lengths, widths)]
    assert streams == [[start, rows, p, count]
                       for start, rows, p, count in zip(starts, widths, lengths, counts)]
    assert most == max(counts)
    programs = [tuple(flat[width + 12 + item * 3:width + 15 + item * 3]) for item in range(items)]
    expected = [(stream, first, chunk) for stream, (rows, p) in enumerate(zip(widths, lengths))
                for chunk in range(p // policy.CHUNK)
                for first in range(0, rows * group, policy.QUERY_TILE)]
    assert programs == expected
    assert flat[width + 12 + items * 3:] == [-1 if parent < 0 else start + parent
                                           for start, tree in zip(starts, trees) for parent in tree]
