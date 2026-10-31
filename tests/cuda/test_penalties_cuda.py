"""CUDA penalty rows equal the float32 host rule (vLLM apply_penalties) bit for bit, by path."""

import numpy as np
import pytest
import torch

from tensorfold.engine.penalties import History, Penalties
from tests.test_penalties import _reference_row

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


@pytest.mark.parametrize("p", [Penalties(presence=1.5), Penalties(presence=-0.5, frequency=0.25, repetition=0.8),
                               Penalties(presence=2.0, frequency=-2.0, repetition=1.05)])
def test_cuda_tree_rows_match_the_host_rule(p):
    rng = np.random.default_rng(11)
    vocab = 262144
    prompt = rng.integers(0, vocab, size=5000).tolist()
    reply = rng.integers(0, vocab, size=300).tolist() + [7, 7, 7]
    history = History(p, prompt)
    history.extend(reply)
    tokens = [reply[-1], 7, 123, 7, prompt[0], 99999, reply[3]]
    parents = [-1, 0, 1, 0, 3, 3, 2]
    logits = torch.tensor(rng.normal(size=(len(tokens), vocab)) * 6, dtype=torch.float32, device="cuda")
    got = history.apply(logits.bfloat16(), tokens, parents).cpu().numpy()
    widened = logits.bfloat16().float().cpu().numpy()
    for row in range(len(tokens)):
        path, node = [], row
        while node > 0:
            path.append(tokens[node])
            node = parents[node]
        np.testing.assert_array_equal(got[row], _reference_row(widened[row], reply + path, prompt, p))
    assert got.shape == widened.shape
