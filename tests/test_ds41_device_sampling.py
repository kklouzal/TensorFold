"""DeepSeek-V4.1 device decode sampling: policy eligibility and control words (CPU)."""

import pytest

pytest.importorskip("torch")

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v4.cuda import device_sampling as ds


def test_eligibility_and_words():
    assert not ds.eligible(None)
    assert not ds.eligible(Sampling(1, temperature=0.0, top_k=0))
    assert not ds.eligible(Sampling(1, temperature=1.0, top_k=20))
    assert not ds.eligible(Sampling(1, temperature=float("nan"), top_k=0))
    assert not ds.eligible(Sampling(1, temperature=1.0, top_k=0, top_p=3 * 2.0 ** -110))
    assert ds.eligible(Sampling(1, temperature=1.0, top_k=0, top_p=0.95))
    assert ds.eligible(Sampling(1, temperature=1.0, top_k=0, top_p=2.0 ** -60))
    assert ds.row_words(None) == [0] * ds.WORDS
    with pytest.raises(ValueError):
        ds.row_words(Sampling(1, temperature=1.0, top_k=5))


def test_words_round_trip_seed_and_temperature():
    import struct

    for seed in (0, 7, 2 ** 63 + 17, -5, 2 ** 64 - 1):
        policy = Sampling(seed, temperature=0.7, top_k=0, top_p=0.95, min_p=0.05)
        words = ds.row_words(policy)
        assert all(-(1 << 31) <= w < 1 << 31 for w in words)
        got_seed, inverse = ds.lane_noise(words)
        assert got_seed & (2 ** 64 - 1) == seed & (2 ** 64 - 1)
        assert struct.unpack("<d", struct.pack("<q", inverse))[0] == 1.0 / 0.7
        num, den = (0.95).as_integer_ratio()
        assert (words[4] & 0xFFFFFFFF) | (words[5] << 32) == num and 1 << words[6] == den
