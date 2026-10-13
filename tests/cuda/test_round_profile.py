"""The opt-in round profile: off creates nothing; on writes one record a round and never changes the tokens."""

import json
import time

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import _model  # noqa: E402

from tensorfold.cuda import round_profile  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12]]
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95)]


def _decode(w):
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3)
    outs = []
    for prompt, sampling in zip(PROMPTS, SAMPLINGS):
        got: list[int] = []
        dec.admit(Stream(prompt, 24, sampling, emit=lambda new, got=got: got.extend(new)))
        outs.append(got)
    while dec.live():
        dec.finish(dec.round())
    return dec, outs


def test_profile_off_creates_nothing(monkeypatch):
    monkeypatch.delenv(round_profile.ENV, raising=False)
    created = []
    monkeypatch.setattr(round_profile.RoundProfile, "__init__", lambda *a, **k: created.append(1))
    dec, outs = _decode(_model())
    assert dec.prof is None and not created and all(outs)


def test_profile_records_rounds_without_changing_tokens(monkeypatch, tmp_path):
    w = _model()
    monkeypatch.delenv(round_profile.ENV, raising=False)
    _, plain = _decode(w)
    path = tmp_path / "rounds.jsonl"
    monkeypatch.setenv(round_profile.ENV, str(path))
    dec, profiled = _decode(w)
    assert profiled == plain
    deadline = time.monotonic() + 10
    while dec.prof.records.qsize() or not path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    time.sleep(0.2)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records and all(r["n"] >= 1 and r["rows"] >= r["n"] and r["wall_ms"] > 0 for r in records)
    names = [p[0] for p in records[-1]["phases"]]
    for phase in ("room", "stage", "tables", "verify_enq", "kv_wait", "sample", "commit", "take", "end"):
        assert phase in names, (phase, names)
    timed = [p for r in records for p in r["phases"] if p[2] is not None]
    assert timed and all(p[2] >= 0 for p in timed)          # GPU progress between event marks
    assert {"window", "kept", "chained", "source", "ctx"} <= set(records[-1])
