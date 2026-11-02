"""Tiny numeric regression for additive bias; no checkpoint/model allocation."""
import numpy as np
import pytest

from tensorfold.engine.penalties import History, Penalties

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("rows", [1, 4, 16])
def test_bias_uses_ordinary_fp32_addition_including_subnormals_and_hard_masks(rows):
    if not torch.cuda.is_available():
        pytest.skip("CUDA numeric qualification requires an admitted GPU process")
    words = np.array([[0, 1, 0xFF800000]], np.uint32)
    bias_words = np.array([[1, 1, np.float32(100).view(np.uint32)]], np.uint32)
    raw = torch.from_numpy(np.repeat(words, rows, axis=0)).to("cuda").view(torch.float32)
    delta = torch.from_numpy(bias_words).to("cuda").view(torch.float32)
    expected = raw + delta
    history = History(Penalties(bias=((0, float.fromhex("0x1p-149")),
                                     (1, float.fromhex("0x1p-149")), (2, 100.0))), [])
    np.testing.assert_array_equal(history._bias_values.view(np.uint32), bias_words[0])
    actual = history.apply(raw, copy=True)
    torch.cuda.synchronize()
    np.testing.assert_array_equal(actual.cpu().numpy().view(np.uint32),
                                  expected.cpu().numpy().view(np.uint32))
    np.testing.assert_array_equal(actual.cpu().numpy().view(np.uint32),
                                  np.repeat(np.array([[1, 2, 0xFF800000]], np.uint32), rows, axis=0))
    np.testing.assert_array_equal(raw.cpu().numpy().view(np.uint32), np.repeat(words, rows, axis=0))
