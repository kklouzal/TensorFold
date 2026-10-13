"""A prompt whose first token cannot be sampled fails alone; the prompts that joined with it go on decoding.

Sampling a joining prompt's first token ran outside any per-request guard, so one request the sampler refused
(an out-of-range ``top_p`` reached it before the HTTP layer checked the range) raised out of the round: every
live stream failed with it, and the refused request was never handed back to ``finish`` to free its slot.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.cuda.sampling import validate_policy
from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda import multi
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

VOCAB = 8


class Engine:
    """The slot engine's sampler: the real policy check, then the likeliest token."""

    def __init__(self):
        self.buf, self.mbuf, self.last_streams = SimpleNamespace(), None, None

    def sample(self, logits, positions, sampling):
        validate_policy(positions, sampling)
        return [int(logits[0].argmax())]


def joining(sid, sampling, got):
    st = SimpleNamespace(pos=4, image_rows=(1,), image_features=torch.zeros(1, 2))
    return Stream([1, 2, 3, 4], 5, sampling, draft=False, emit=got.append, sid=sid, st=st)


def decoder(streams):
    dec = object.__new__(MultiDecoder)
    dec.w = SimpleNamespace(meta={})
    dec.eos, dec.depth, dec.confidence = (VOCAB - 1,), 0, 0.0
    dec.streams, dec.kept, dec.free, dec._shrink = {}, [], [], lambda st: None
    dec.used, dec.clock = {}, 0
    dec.filling = list(streams)
    dec.fills = {s.sid: [Engine(), False, 0, [], None] for s in streams}   # [engine, drafts?, next row, kept, tail]
    dec._rotor_lookup = lambda *args: False
    dec.watch = None                                                       # draft telemetry off
    return dec


def test_a_refused_first_token_fails_only_its_own_request(monkeypatch):
    monkeypatch.setattr(multi, "_kv_check", lambda segs: None)
    good_tokens, bad_tokens = [], []
    good = joining(1, Sampling(7, 1.0, 20, 0.95), good_tokens)
    bad = joining(2, Sampling(7, 1.0, 20, 1.5), bad_tokens)             # out of range: the sampler refuses it
    after = joining(3, None, [])
    dec = decoder([good, bad, after])
    heads = torch.zeros(3, VOCAB)
    heads[:, 3] = 1.0
    pieces = [(s, 0, len(s.prompt)) for s in (good, bad, after)]

    joined = dec._joined(pieces, heads, [None] * 3, 0.3)

    assert joined == [bad] and bad.done and isinstance(bad.error, ValueError) and bad_tokens == []
    assert bad.sid not in dec.streams and dec.filling == [] and dec.fills == {}
    for s in (good, after):                                    # the others joined with their first token
        assert dec.streams[s.sid] is s and s.out == [3] and not s.done and s.error is None
    assert good_tokens == [[3]]
    for s in (good, bad, after):                               # prefill ended for all three: image rows released
        assert s.st.image_rows == () and s.st.image_features is None
    dec.finish(joined)
    assert dec.free == [bad.st] and set(dec.streams) == {good.sid, after.sid}
