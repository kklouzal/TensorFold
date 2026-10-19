"""Block-drafter chains on CUDA: argmax proposals, and penalized ones for streams with penalties, in one readback."""

import numpy as np
import pytest
import torch

from tensorfold.cuda import speculative
from tensorfold.engine.penalties import History, Penalties

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


def test_mixed_streams_propose_their_argmax_or_penalized_argmax():
    vocab, rows = 4099, 5
    generator = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn((2 * rows, vocab), generator=generator, device="cuda") * 3
    top = logits.argmax(-1).cpu().tolist()
    history = History(Penalties(presence=2.0, frequency=1.0), [top[rows], 11])
    history.extend(top[rows:rows + 2])
    chains = speculative.Chains(logits, [None, history], rows)
    assert speculative.Chains(logits, [None, None], rows).words.shape == (2 * rows, 2)
    ids, parents, scores = chains.finish(0, rows)
    assert ids == top[:rows] and parents == list(range(-1, rows - 1)) and all(a <= b for a, b in zip(scores, scores[1:]))
    host = logits[rows:].cpu().numpy()
    want = []
    for row in host:
        want.append(int(np.argmax(history.candidates(row[None], np.arange(vocab)[None], want)[0])))
    assert chains.finish(1, rows, history)[0] == want != top[rows:]
    with pytest.raises(ValueError):
        chains.finish(1, rows)
    with pytest.raises(ValueError):
        speculative.Chains(logits, [None], rows)
