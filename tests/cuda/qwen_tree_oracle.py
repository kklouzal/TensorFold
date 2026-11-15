"""Shared independent proposal-tree fixture for Qwen scheduler integration tests."""

import random

from tensorfold.families.qwen3_5.cuda.multi import ONE, TREE, MultiDecoder

V = 256


class _Oracle(MultiDecoder):
    """Proposes trees mixing each prompt's true continuation with wrong tokens, so rounds keep runs of drafts."""

    def __init__(self, w, truth, seed=0, **kw):
        super().__init__(w, None, max_rows=6, **kw)
        self.truth, self.rng = truth, random.Random(seed)

    def _mode(self, s, copied):
        if s.sampling is not None and s.sampling.temperature > 0:
            # This independent scheduler oracle deliberately supplies arbitrary q-less trees and checks target
            # choices. Real stochastic DFlash/copy producers use explicit CHAIN/COPY_CHAIN modes instead.
            copied[s.sid] = []
            return TREE if s.draft else ONE
        mode = super()._mode(s, copied)
        return TREE if s.draft and not copied.get(s.sid) else mode

    def _trees(self, plan, blocks):
        out = {}
        for sid, mode, _, _ in plan:
            if mode != TREE:
                continue
            s = self.streams[sid]
            ref, n = self.truth[tuple(s.prompt)], len(s.out)
            guesses, parents = [], []
            for d in range(self.rng.randint(1, 4)):
                guesses.append(ref[n + d] if n + d < len(ref) and self.rng.random() < 0.8 else self.rng.randrange(1, V))
                parents.append(d - 1)
            if self.rng.random() < 0.5:                   # a sibling of the first draft
                guesses.append(self.rng.randrange(1, V))
                parents.append(-1)
            out[sid] = (guesses, parents, [0.3 * (k + 1) for k in range(len(guesses))])
        return out

