"""DeepSeek-V4.1 weight-load staging (one GPU): pinned-buffer uploads equal the file's bytes for ranges that split
across buffers, pack several per buffer, and reuse buffers across calls while earlier copies may still run."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def test_staging_copies_exact_bytes(tmp_path):
    from tensorfold.families.deepseek_v4.cuda.staging import Staging
    rng = np.random.default_rng(7)
    data = rng.integers(0, 256, 24 << 20, dtype=np.uint8)
    path = tmp_path / "blob.bin"
    data.tofile(path)
    staging = Staging(chunk=1 << 20, buffers=3)
    try:
        cases = [
            [(0, 5 << 20)],                                            # one range over five buffers
            [(int(o), 4096 + 17 * i) for i, o in enumerate(rng.integers(0, 20 << 20, 300))],  # many per buffer
            [(123, (1 << 20) - 1), (7 << 20, (3 << 20) + 5), (1, 1)],   # odd sizes across boundaries
        ]
        outs = []
        for ranges in cases * 2:                                       # buffers reused across calls
            out = torch.empty(sum(n for _, n in ranges), dtype=torch.uint8, device="cuda")
            outs.append((ranges, staging.copy(path, ranges, out)))
        torch.cuda.synchronize()
        for ranges, out in outs:
            want = np.concatenate([data[o:o + n] for o, n in ranges])
            assert np.array_equal(out.cpu().numpy(), want)
        with pytest.raises(ValueError):
            staging.copy(path, [(0, 10)], torch.empty(9, dtype=torch.uint8, device="cuda"))
        with pytest.raises(OSError):                                   # past the end of the file
            staging.copy(path, [(os.path.getsize(path) - 4, 8)], torch.empty(8, dtype=torch.uint8, device="cuda"))
    finally:
        staging.close()
