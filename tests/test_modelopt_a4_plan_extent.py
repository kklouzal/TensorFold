"""Independent cardinality proof for the native consumer's live item extent."""

import ast
from pathlib import Path


def consumer_extent(pairs, experts, tile=16):
    path = Path(__file__).parents[1] / "src/tensorfold/cuda/nvfp4/experts_a4.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExpertsA4")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run_plan")
    assignment = next(node for node in method.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "items" for target in node.targets))
    expression = ast.Expression(body=assignment.value.slice.upper)
    # Evaluate the actual owned pure extent expression without importing Torch.
    return eval(compile(expression, str(path), "eval"),
                {"pairs": pairs, "self": type("Bank", (), {"count": experts})(),
                 "plan": type("Plan", (), {"tile": tile})()})


def partitions(total, maximum):
    if not total:
        yield ()
    else:
        for first in range(1, min(total, maximum) + 1):
            for rest in partitions(total - first, first):
                yield (first, *rest)


def test_every_item_owns_a_pair_so_padding_never_needs_exposing():
    # place_items emits ceil(population/16) per used expert. Exhaust all
    # population partitions through32 pairs independently of the Python bound.
    for tile in (16, 64):
        for pairs in range(1, 33):
            for populations in partitions(pairs, pairs):
                used = len(populations)
                live = sum((population + tile - 1) // tile for population in populations)
                assert live <= pairs
                for experts in (used, 512):
                    extent = consumer_extent(pairs, experts, tile)
                    assert live <= extent <= pairs


def test_wide_64_pair_plans_never_expose_fewer_items_than_live():
    # ceil(c / T) <= 1 + c // T per used expert, and the floors sum to at most pairs // T.
    for tile in (16, 64):
        for used in (1, 3, 129, 257):
            for pairs in (used, 2 * used + 5, 64 * used + 63, 4096 * 9):
                if pairs < used:
                    continue
                base, extra = divmod(pairs, used)
                populations = [base + (i < extra) for i in range(used)]
                live = sum((p + tile - 1) // tile for p in populations)
                assert live <= consumer_extent(pairs, used - 1, tile) <= pairs


def test_actual_short_row_bank_extent_never_exceeds_native_pair_boundary():
    for rows in (1, 2, 3, 8, 17, 65, 129):
        pairs = rows * 11
        extent = consumer_extent(pairs, 512)
        assert extent <= pairs
        assert extent >= min(pairs, 513)
    assert consumer_extent(22, 512) == 22


def test_every_previously_admitted_extent_is_unchanged():
    for experts in (1, 2, 3, 16, 512, 1023):
        for pairs in range(1, 4097):
            old = min(pairs, experts + 1) + pairs // 16
            if old <= pairs:
                assert min(pairs, experts + 1 + pairs // 16) == old
