"""Flash Next's lane matmul launch table (host): measured chips only, blocks that exist, fixed K slices untouched."""

import re
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda import qmm  # noqa: E402

POLICY = Path(qmm.__file__).parents[3] / "cuda" / "kernels" / "qmm_policy.cuh"


def _lane_blocks() -> set[int]:
    text = POLICY.read_text()
    body = text[text.index("#define TENSORFOLD_QMM_LANE_TILES(X)"):]
    body = body[:body.index("\n\n")]
    return {int(m) for m in re.findall(r"X\((\d+),", body)}


def test_other_chips_keep_the_row_buckets_with_clusters():
    shapes = [(n, k) for n, k in qmm.SPLITS] + [shape for table in qmm.LANE.values() for shape in table]
    for chip in ((12, 1, 48), (12, 0, 188), (12, 0, 36), (9, 0, 132), (8, 9, 128)):
        assert chip not in qmm.LANE
        assert {qmm.lane_launch(n, k, m, chip) for n, k in shapes for m in range(1, 129)} == {(0, True)}, chip


def test_the_table_names_existing_blocks_for_ascending_row_ranges():
    blocks = _lane_blocks()
    assert {1, 2, 3} <= blocks
    for chip, table in qmm.LANE.items():
        for (n, k), entries in table.items():
            rows = [r for r, _, _ in entries]
            assert rows == sorted(set(rows)) and rows[0] >= 1, (chip, n, k)
            assert all(tile in blocks and isinstance(clusters, bool) for _, tile, clusters in entries), (chip, n, k)
            assert qmm.lane_launch(n, k, rows[-1] + 1, chip) == (0, True)
