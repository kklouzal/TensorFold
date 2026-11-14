"""Staged packed GGUF loads (pinned pieces, on-device row/segment selection) equal the host-selected bytes."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import direct_read  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import gguf_weights  # noqa: E402
from tensorfold.gguf_file import PackedTensor  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="staged reads upload to a CUDA device")


@pytest.mark.parametrize("columns", [((0, 128),), ((64, 64), (0, 32))], ids=["contiguous", "permuted-segments"])
@pytest.mark.parametrize("row_ranges", [((0, 50),), ((40, 7), (3, 20), (60, 4), (23, 1))], ids=["one-run", "runs"])
def test_staged_packed_rows_match_host_selection(tmp_path, monkeypatch, columns, row_ranges):
    rows, row_bytes, lead = 64, 4 * 34, 4096 + 13             # Q8_0 [64, 128]; unaligned tensor start
    payload = torch.randint(0, 256, (lead + rows * row_bytes + 77,), dtype=torch.uint8,
                            generator=torch.Generator().manual_seed(11))
    path = tmp_path / "packed.bin"
    path.write_bytes(payload.numpy().tobytes())
    source = PackedTensor("w", (rows, 128), 8, path, lead, rows * row_bytes)
    plan = gguf_weights.MatrixPlan(source, row_ranges, columns, packed=True)
    monkeypatch.setattr(direct_read, "PIECE", 1000)           # spans cross several reused pinned pieces
    monkeypatch.setattr(gguf_weights, "STAGING_BYTES", 9 * row_bytes)   # several read groups per plan

    reader = direct_read.Reader()
    staged = gguf_weights._packed(plan, reader, "cuda").data
    host_reader = SimpleNamespace(read=reader.read)            # no read_staged: host selection path
    hosted = gguf_weights._packed(plan, host_reader, "cuda").data
    torch.cuda.synchronize()
    reader.close()

    table = payload[lead:lead + rows * row_bytes].view(rows, row_bytes)
    order = torch.tensor([r for first, count in row_ranges for r in range(first, first + count)])
    expected = torch.cat([table[order, lo:lo + n] for lo, n in plan.byte_segments], dim=1)
    assert torch.equal(staged.cpu(), expected) and torch.equal(hosted.cpu(), expected)
