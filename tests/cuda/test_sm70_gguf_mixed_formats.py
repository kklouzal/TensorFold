"""Real-device controls; CPU fixture success cannot replace these SM70 gates."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.sm70.gguf_ops import decode  # noqa: E402 - optional backend guard precedes CUDA imports

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="mixed GGUF requires an actual SM70 device")


@pytest.mark.parametrize("kind", (14, 18, 21, 22, 23, 42))
def test_native_full_and_partial_tp_columns_match_original_ggml_bits(kind):
    fixture = Path(__file__).parents[1] / "fixtures/gguf_mixed_ggml.npz"
    with np.load(fixture, allow_pickle=False) as wire:
        raw = torch.tensor(wire[f"raw_{kind}"], device="cuda")
        expected = wire[f"bits_{kind}"].view(np.float32)
    for dtype in (torch.float32, torch.float16):
        with np.errstate(over="ignore"):
            wanted = expected.astype(np.float32 if dtype == torch.float32 else np.float16)
        actual = decode(raw, kind, expected.shape[1], dtype=dtype).cpu().numpy()
        np.testing.assert_array_equal(actual.view(np.uint8), wanted.view(np.uint8))
        ids = torch.tensor([7, 0, 7, 3], dtype=torch.int64, device="cuda")
        actual = decode(raw, kind, 80, column_offset=16, dtype=dtype, ids=ids).cpu().numpy()
        np.testing.assert_array_equal(actual.view(np.uint8), wanted[[7, 0, 7, 3], 16:96].copy().view(np.uint8))
    for offset, columns in ((2**63 - 1, 80), (16, 2**63 - 1), (0, 0), (-1, 80)):
        with pytest.raises(RuntimeError, match="visible columns"):
            decode(raw, kind, columns, column_offset=offset)
