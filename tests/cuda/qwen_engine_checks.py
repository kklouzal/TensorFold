"""Small checks shared by the Qwen 27B / 36B CUDA engine tests."""

import torch


def same_state(a, b):
    """Two States hold the same committed position, recurrent and conv states, and K/V rows, bit for bit."""
    assert a.pos == b.pos
    for x, y in zip(a.rec, b.rec):
        assert (x is None) == (y is None) and (x is None or torch.equal(x, y))
    for x, y in zip(a.conv, b.conv):
        assert (x is None) == (y is None) and (x is None or torch.equal(x, y))
    for x, y in zip(a.kv, b.kv):
        if x is not None:
            assert torch.equal(x[0][:a.pos], y[0][:b.pos]) and torch.equal(x[1][:a.pos], y[1][:b.pos])


def generated_ids(engine, prompt, sampling, draft, tokens=48):
    """The engine's reply tokens to ``prompt`` and its generation stats."""
    out: list[int] = []
    stats = engine.generate(list(prompt), tokens, sampling, lambda new: out.extend(new) and False, draft=draft)
    return out, stats


def points_after(k):
    """Snapshot points for the test prompts: position ``k`` when a prompt is longer."""

    return lambda ids: [k] if len(ids) > k + 1 else []
