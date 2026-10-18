"""The CUDA candidate packet of coupled drafting: one readback carries every sampled stream's exact candidates."""

import numpy as np
import pytest
import torch

from tensorfold.cuda import speculative
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.speculative import coupled

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


def test_mixed_streams_read_their_own_candidates_and_nucleus_norms():
    vocab, rows = 4099, 5
    generator = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn((3 * rows, vocab), generator=generator, device="cuda") * 3
    policies = [Sampling(seed=1, temperature=1.0, top_k=64, top_p=0.95), None,
                Sampling(seed=2, temperature=0.6, top_k=0, top_p=0.9)]
    packet = speculative.launch(logits, policies, rows)
    assert packet.shape == (3 * rows, 2 * speculative.NUCLEUS_DRAFT + 1) and packet.is_cuda
    host = speculative.read(packet)
    full = logits.cpu().numpy()
    ids = np.broadcast_to(np.arange(vocab, dtype=np.int64), full.shape)
    for stream in (0, 2):
        s = policies[stream]
        part = slice(stream * rows, (stream + 1) * rows)
        norms = torch.logsumexp(logits[part].double() / s.temperature, -1).cpu().numpy()
        want = coupled(full[part], ids[part].copy(), 50 + np.arange(rows), s, norms=norms)[0][:, 0]
        got, scores = speculative.chain(host, stream * rows, rows, 50, s)
        assert got == want.tolist() and all(a <= b for a, b in zip(scores, scores[1:]))
    assert speculative.launch(logits, [None, None, None], rows) is None
    with pytest.raises(ValueError):
        speculative.launch(logits, policies[:2], rows)
