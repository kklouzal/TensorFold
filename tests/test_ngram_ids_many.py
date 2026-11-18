"""NGram.ids_many: one batched pass over a round's windows equals one ``ids`` call per window."""

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.cuda.ngram import NGram

EOS = 13


@pytest.fixture(scope="module")
def ngram():
    return NGram(vocab=1024, ngram_size=3, heads_per_ngram=8, vocab_base=20_000_000,
                 divisor=128, seed=1234, eos=EOS, embed_dim=512, ple_index=0)


@pytest.mark.parametrize("seed", range(40))
def test_batched_windows_equal_per_window_ids(ngram, seed):
    rng = np.random.default_rng(seed)
    count = int(rng.integers(1, 25))                       # one window up to a C24 round
    histories, windows = [], []
    for _ in range(count):
        # Frequent EOS in histories and tokens exercises the resets; mixed lengths 1-5 exercise the padding.
        history = rng.integers(0, 1024, ngram.context)
        history[rng.random(ngram.context) < 0.3] = EOS
        tokens = rng.integers(0, 1024, int(rng.integers(1, 6)))
        tokens[rng.random(len(tokens)) < 0.2] = EOS
        histories.append(history.astype(np.int64))
        windows.append(tokens.astype(np.int64))
    expected = np.concatenate([ngram.ids(h, t) for h, t in zip(histories, windows)])
    got = ngram.ids_many(histories, windows)
    assert got.dtype == expected.dtype and got.shape == (sum(map(len, windows)), ngram.heads)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("histories, windows", [([], []), ([np.full(2, EOS)], [np.array([], dtype=np.int64)]),
                                                ([np.full(2, EOS)], [np.array([1]), np.array([2])])])
def test_refuses_empty_or_mismatched_windows(ngram, histories, windows):
    with pytest.raises(ValueError, match="n-gram windows"):
        ngram.ids_many(histories, windows)
