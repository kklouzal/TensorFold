"""The draft rule's fit (``tools/fit_draft_rule.py``) on synthetic telemetry with a known truth: monotone tables the
server accepts, the better score feature selected on held-out requests, suffix tables from traces, the cost prior, and
no token ids in the file."""

import importlib.util
import json
import math
from pathlib import Path
import random
import sys

import pytest

from tensorfold.drafters import draft_rule as dr

TOOL = Path(__file__).resolve().parents[1] / "tools/fit_draft_rule.py"
_spec = importlib.util.spec_from_file_location("fit_draft_rule", TOOL)
fit = importlib.util.module_from_spec(_spec)
sys.modules["fit_draft_rule"] = fit
_spec.loader.exec_module(fit)

BASE = 910_000                                           # distinctive token ids: none may reach the file


def _lines(rng, rounds=3000):
    """Rounds whose MTP drafts land with chance sigmoid(margin) (the margin informs, the probability does not), and
    whose wall time is 30 + 5 (R - 1) + 0.3 sum(rows x context) / 1e5 ms plus 1.5 ms a chain step."""

    out = []
    for r in range(rounds):
        rid = f"r{r % 200}"
        greedy = r % 3 == 0
        drafts, hit_all = [], True
        for j in range(4):
            m = rng.uniform(-3, 4)
            hit = rng.random() < 1 / (1 + math.exp(-m))
            label = int(hit and hit_all)
            drafts.append({"j": j, "p": rng.random(), "m": m, "e": None, "g": None, "z": None, "c": None, "L": None,
                           "a0": None, "inc": False, "hit": int(hit), "label": label})
            hit_all = hit_all and hit
        verified = rng.randint(0, 4)
        for d in drafts[:verified]:
            d["inc"] = True
        accepted = next((i for i, d in enumerate(drafts[:verified]) if d["label"] != 1), verified)
        steps, ctx = 1 + rng.randint(0, 3), rng.randint(1_000, 400_000)
        chain_ms = sum(1.5 + rng.gauss(0, 0.05) for _ in range(steps - 1))
        ms = 30 + 5 * verified + chain_ms + 0.3 * (verified + 1) * ctx / 1e5 + rng.gauss(0, 0.1)
        out.append({"t": 0, "live": 1, "rows": verified + 1, "mixed": False, "pass_rows": 0, "steps": steps,
                    "chain_steps": steps - 1, "chain_ms": chain_ms, "probed": True, "joined": 0, "ms": ms,
                    "cost": None, "ab": {}, "dropped": 0, "forced": 0,
                    "streams": [{"rid": rid, "ctx": ctx, "n": 5, "greedy": greedy, "src": "mtp", "probe": True,
                                 "join": False, "constrained": False, "verified": verified, "accepted": accepted,
                                 "done": False, "drafts": drafts}]})
    return out


def _trace(rid, prompt, out, sampling=None):
    return {"v": 1, "rid": rid, "start": 0.0, "end": 1.0, "prompt": prompt, "out": out, "count": len(out),
            "stop_eos": True, "sampling": sampling, "draft": True, "constrained": False, "cached": 0,
            "background": False, "error": None, "process": "p", "order": 1, "after": 0, "dropped": 0,
            "decoder": {"depth": 4, "confidence": 0.5, "copy": True, "copy_match": 8, "anchor": 4, "recent": 32,
                        "longest": 64}, "text": "suffix", "rounds": []}


def test_a_fit_selects_the_informative_feature_and_writes_a_valid_contentless_file(tmp_path):
    rng = random.Random(4)
    log = tmp_path / "drafts.jsonl"
    log.write_text("\n".join(json.dumps(line) for line in _lines(rng)) + "\n")
    traces = tmp_path / "trace"
    traces.mkdir()
    for i in range(40):                                  # replies that quote their prompts, then wander off
        prompt = [BASE + rng.randrange(5000) for _ in range(300)]
        out = []
        while len(out) < 200:
            at = rng.randrange(250)
            out += prompt[at:at + rng.randint(5, 40)] + [BASE + 5000 + rng.randrange(5000)]
        sampling = None if i % 2 else {"seed": i, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}
        (traces / f"{i}.json").write_text(json.dumps(_trace(f"t{i}", prompt, out[:200], sampling)))
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"model": "synthetic"}))
    output = tmp_path / "rule.json"
    assert fit.main(["--log", str(log), "--traces", str(traces), "--output", str(output),
                     "--source", str(source)]) == 0
    data = json.loads(output.read_text())
    pred = dr.Predictor.load(output)                     # what the server accepts
    report = data["source"]["held_out"]
    for policy in ("greedy", "sampled"):
        mtp = report["mtp"][policy]
        assert mtp["selected"] == "m" and mtp["m"]["held_out"]["log_loss"] < mtp["p"]["held_out"]["log_loss"]
        assert mtp["m"]["held_out"]["log_loss"] < mtp["prior"]["log_loss"]
        table = pred.tables["mtp", policy]
        assert table.feature == "m" and pred.needs_margins
        assert pred.mtp(policy == "greedy", 0, 0.5, 3.5) > 0.8 > 0.2 > pred.mtp(policy == "greedy", 0, 0.5, -2.5)
        assert pred.mtp(policy == "greedy", 0, 0.5, math.inf) > 0.8     # beyond every bin: the last
        assert all(sum(row) > 0 for row in table.count[:3])
        suffix = pred.tables["suffix", policy]
        assert report["suffix"][policy]["fit"]["samples"] > 1000 and suffix.feature == "L"
        assert report["suffix"][policy]["fit"]["held_out"]["log_loss"] < report["suffix"][policy]["prior"]["log_loss"]
        assert 0.5 < suffix.chance(1, 20) < 1.0                    # these replies keep quoting for a while
    assert data["source"]["model"] == "synthetic" and data["source"]["suffix_from"] == "traces"
    assert data["source"]["logs"][0]["sha256"] and data["source"]["traces"] == 40 and not data["source"]["prior_tables"]
    cost = data["cost"]["unmixed"]
    assert (cost["t0"], cost["row"], cost["ctx"], cost["step"]) == pytest.approx((30, 5, 0.3, 1.5), abs=0.1)
    assert data["cost"]["mixed"] == pytest.approx(dr.COST_PRIOR["mixed"])     # no prompt-pass rounds: the prior's
    assert not any(isinstance(x, int) and BASE <= x < BASE + 10_000 for x in _values(data))     # no token ids


def _values(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _values(v)
    elif isinstance(x, list):
        for v in x:
            yield from _values(v)
    else:
        yield x


def test_too_few_samples_keep_the_prior_and_say_so(tmp_path):
    log = tmp_path / "drafts.jsonl"
    log.write_text("\n".join(json.dumps(line) for line in _lines(random.Random(1), rounds=5)) + "\n")
    data = fit.run(type("Args", (), dict(log=[str(log)], traces=[], output="", source=None, confidence=0.5,
                                         suffix_depth=8))())
    prior = dr.Predictor.prior(0.5).as_dict()["tables"]
    assert data["tables"] == prior and sorted(data["source"]["prior_tables"]) == \
        ["mtp.greedy", "mtp.sampled", "suffix.greedy", "suffix.sampled"]
    dr.Predictor.from_dict(data)
