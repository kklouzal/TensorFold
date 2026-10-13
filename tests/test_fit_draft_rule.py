"""The draft rule's fit (``tools/fit_draft_rule.py``) on synthetic telemetry with a known truth: monotone tables the
server accepts, the better score feature selected on held-out requests, suffix tables from traces, the cost prior, and
no token ids in the file."""

import importlib.util
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import pytest

from tensorfold.drafters import draft_rule as dr

TOOL = Path(__file__).resolve().parents[1] / "tools/fit_draft_rule.py"
_spec = importlib.util.spec_from_file_location("fit_draft_rule", TOOL)
fit = importlib.util.module_from_spec(_spec)
sys.modules["fit_draft_rule"] = fit
_spec.loader.exec_module(fit)

BASE = 910_000                                           # distinctive token ids: none may reach the file


TRUTH = {"t0": 30.0, "stream": 8.0, "row": 5.0, "ctx": 0.3, "step": 1.5, "step_stream": 0.4}


def _stream(rng, rid, greedy, ctx):
    """A stream's entry: four MTP drafts landing with chance sigmoid(margin) (the margin informs, the probability
    does not), some verified."""

    drafts, hit_all = [], True
    for j in range(4):
        m = rng.uniform(-3, 4)
        hit = rng.random() < 1 / (1 + math.exp(-m))
        drafts.append({"j": j, "p": rng.random(), "m": m, "e": None, "g": None, "z": None, "c": None, "L": None,
                       "a0": None, "inc": False, "hit": int(hit), "label": int(hit and hit_all)})
        hit_all = hit_all and hit
    verified = rng.randint(0, 4)
    for d in drafts[:verified]:
        d["inc"] = True
    accepted = next((i for i, d in enumerate(drafts[:verified]) if d["label"] != 1), verified)
    return {"rid": rid, "ctx": ctx, "n": 5, "greedy": greedy, "src": "mtp", "probe": True, "join": False,
            "constrained": False, "verified": verified, "accepted": accepted, "done": False, "drafts": drafts}


def _lines(rng, rounds=3000):
    """Rounds of one to four streams (``_stream``) whose wall time is TRUTH's: 30 + 8 (live - 1) + 5 (R - live) +
    0.3 sum(rows x context) / 1e5 ms, plus 1.5 ms a chain step and 0.4 for each further stream in it."""

    out = []
    for r in range(rounds):
        live = rng.randint(1, 4)
        streams = [_stream(rng, f"r{(r * 4 + i) % 200}", r % 3 == 0, rng.randint(1_000, 400_000))
                   for i in range(live)]
        rows = sum(1 + st["verified"] for st in streams)
        joined = [rng.randint(1, live) for _ in range(rng.randint(0, 3))]
        chain_ms = sum(TRUTH["step"] + TRUTH["step_stream"] * (n - 1) + rng.gauss(0, 0.05) for n in joined)
        ms = TRUTH["t0"] + TRUTH["stream"] * (live - 1) + TRUTH["row"] * (rows - live) + chain_ms + \
            TRUTH["ctx"] * sum((1 + st["verified"]) * st["ctx"] for st in streams) / 1e5 + rng.gauss(0, 0.1)
        out.append({"t": float(r), "live": live, "rows": rows, "mixed": False, "pass_rows": 0, "steps": 1 + len(joined),
                    "chain_steps": len(joined), "chain_ms": chain_ms, "chain_rows": sum(joined), "probed": True,
                    "joined": 0, "ms": ms, "cost": None, "ab": {}, "dropped": 0, "forced": 0, "streams": streams})
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
        assert table.feature == "m" and pred.needs_margins and pred.mtp_tables[policy == "greedy"] is table
        assert table.chance(0, 3.5) > 0.8 > 0.2 > table.chance(0, -2.5)
        assert table.chance(0, math.inf) > 0.8                     # beyond every bin: the last
        assert all(sum(row) > 0 for row in table.count[:3])
        suffix = pred.tables["suffix", policy]
        assert report["suffix"][policy]["fit"]["samples"] > 1000 and suffix.feature == "L"
        assert report["suffix"][policy]["fit"]["held_out"]["log_loss"] < report["suffix"][policy]["prior"]["log_loss"]
        assert 0.5 < suffix.chance(1, 20) < 1.0                    # these replies keep quoting for a while
    assert data["source"]["model"] == "synthetic" and data["source"]["suffix_from"] == "traces"
    assert data["source"]["logs"][0]["sha256"] and data["source"]["traces"] == 40 and not data["source"]["prior_tables"]
    cost = data["cost"]["unmixed"]
    assert {t: cost[t] for t in TRUTH} == pytest.approx(TRUTH, abs=0.1)
    assert data["cost"]["mixed"] == pytest.approx(dr.COST_PRIOR["mixed"])     # no prompt-pass rounds: the prior's
    assert data["source"]["prior_costs"] == ["mixed", "mixed.steps"] and data["source"]["fit_half"] is None
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


def _args(log, **kw):
    return type("Args", (), {"log": [str(log)], "traces": [], "output": "", "source": None, "confidence": 0.5,
                             "suffix_depth": 8, "fit_half": None, **kw})()


def test_too_few_samples_keep_the_prior_and_say_so(tmp_path):
    log = tmp_path / "drafts.jsonl"
    log.write_text("\n".join(json.dumps(line) for line in _lines(random.Random(1), rounds=5)) + "\n")
    data = fit.run(_args(log))
    prior = dr.Predictor.prior(0.5).as_dict()["tables"]
    assert data["tables"] == prior and sorted(data["source"]["prior_tables"]) == \
        ["mtp.greedy", "mtp.sampled", "suffix.greedy", "suffix.sampled"]
    # fewer rounds than the online fit's memory: the server's cost prior stays, said so
    for name in ("unmixed", "mixed"):
        assert data["cost"][name] == pytest.approx(dr.COST_PRIOR[name])
    assert data["source"]["prior_costs"] == ["unmixed", "unmixed.steps", "mixed", "mixed.steps"]
    assert fit.MIN_COST_ROUNDS == 200
    dr.Predictor.from_dict(data)
    for bad in ([1, 2], {"held_out": {}}):                # added provenance: an object that keeps the fit's own fields
        source = tmp_path / "source.json"
        source.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="--source"):
            fit.run(_args(log, source=str(source)))


def test_the_cost_fit_bounds_paused_rounds_and_steps_as_the_online_model_does():
    """Rounds paused by seconds (a host stall, a collection) in 3% of the log, and a few paused chain steps: the fit
    holds each within the online outlier bound and still recovers the true costs, where plain least squares and a
    plain step mean would not; the report counts the rounds held to the bound."""

    rng = random.Random(9)
    lines = _lines(rng)
    for line in rng.sample(lines, 90):
        line["ms"] += rng.uniform(2000, 5000)
    for line in rng.sample([x for x in lines if x["chain_steps"]], 15):
        line["chain_ms"] += 3000.0
        line["ms"] += 3000.0
    cost, report = fit.fit_cost(lines)
    got = cost["unmixed"]                            # a held round still counts its bound (~27 ms): t0 a little high
    assert got["t0"] == pytest.approx(TRUTH["t0"], abs=1.5)
    assert {t: got[t] for t in ("stream", "row", "ctx", "step", "step_stream")} == pytest.approx(
        {t: TRUTH[t] for t in ("stream", "row", "ctx", "step", "step_stream")}, abs=0.3)
    assert report["unmixed"]["bounded"] >= 90 and report["unmixed"]["iterations"] < fit.FIT_ITERATIONS
    assert report["unmixed"]["steps"]["bounded"] >= 15 and report["unmixed"]["steps"]["underived"] == 0
    prior = dr.CostFit([dr.COST_PRIOR["unmixed"][t] for t in dr.COST_TERMS], dr.COST_SD)
    a = np.asarray([dr.features(x["rows"], x["live"], sum((s["verified"] + 1) * s["ctx"] for s in x["streams"]), 0)
                    for x in lines])                 # the same data without the bound: far off (the test's power)
    b = np.asarray([x["ms"] - x["chain_ms"] for x in lines])
    p0 = np.diag(prior.precision0)
    plain = np.linalg.solve(a.T @ a + p0, a.T @ b + p0 @ np.asarray(prior.theta0))
    assert abs(plain[0] - TRUTH["t0"]) > 50
    steps = [x for x in lines if x["chain_steps"]]
    assert sum(x["chain_ms"] for x in steps) / sum(x["chain_steps"] for x in steps) > 3.0
    clean, _ = fit.fit_cost(_lines(random.Random(9)))            # no outliers: the bound changes nothing material
    assert (clean["unmixed"]["step"], clean["unmixed"]["step_stream"]) == pytest.approx((1.5, 0.4), abs=0.02)


def test_a_logs_step_streams_come_from_the_round_that_verified_its_drafts():
    """A log written before ``chain_rows``: a round's steps' streams, summed, are what the next round (by ``t``, not
    file order) verified from them, each drafting call apart (prompts that joined); a next round that does not
    account for the steps leaves the round out of the step fit."""

    def stream(drafts, join=False, src="mtp"):
        return {"src": src, "join": join, "drafts": [{}] * drafts}

    lines = [{"t": 3.0, "chain_steps": 0, "streams": [stream(1)]},
             {"t": 1.0, "chain_steps": 3, "streams": []},
             {"t": 2.0, "chain_steps": 0, "streams": [stream(4), stream(2), stream(1), stream(3, src="suffix"),
                                                      stream(2, join=True)]},
             {"t": 4.0, "chain_steps": 2, "chain_rows": 5, "streams": []},
             {"t": 5.0, "chain_steps": 1, "streams": []}]
    ordered = sorted(lines, key=lambda x: x["t"])
    # t=1: its call drafted 4, 2 and 1 (3 + 1 stream steps over 3 steps) and a joined prompt 2 (1 more step: 4 in all)
    assert fit.stream_steps(ordered, 0) is None          # 3 steps logged, 4 accounted for: left out
    lines[1]["chain_steps"] = 4
    assert fit.stream_steps(ordered, 0) == 3 + 1 + 0 + 1
    assert fit.stream_steps(ordered, 3) == 5             # logged
    assert fit.stream_steps(ordered, 4) is None          # no next round


def test_a_fit_on_one_half_holds_out_the_other(tmp_path):
    """``--fit-half``: the written tables come from that half's requests alone, and the report's held-out samples
    are the other half's (two such fits cross-fit ``tools/draft_replay.py``'s evaluation)."""

    rng = random.Random(4)
    log = tmp_path / "drafts.jsonl"
    lines = _lines(rng)
    log.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    samples = fit.mtp_samples(lines)["sampled"]
    halves = {h: sum(1 for line in lines for st in line["streams"] if not st["greedy"]
                     for j in range(len(st["drafts"])) if dr.half(st["rid"]) == h and
                     (j == 0 or st["drafts"][j - 1]["label"] == 1)) for h in (0, 1)}
    assert sum(halves.values()) == len(samples)
    tables = {}
    for h in (0, 1):
        data = fit.run(_args(log, fit_half=h))
        held = data["source"]["held_out"]["mtp"]["sampled"]
        assert data["source"]["fit_half"] == h and held[held["selected"]]["held_out"]["n"] == halves[1 - h]
        tables[h] = data["tables"]["mtp"]["sampled"]
        assert sum(map(sum, tables[h]["count"])) == halves[h]               # fitted on its half alone
    whole = fit.run(_args(log))["tables"]["mtp"]["sampled"]
    assert sum(map(sum, whole["count"])) == len(samples) and tables[0]["table"] != tables[1]["table"]


def test_the_bounded_fits_settle_or_say_so(monkeypatch):
    fitted = dr.CostFit([dr.COST_PRIOR["unmixed"][t] for t in dr.COST_TERMS], dr.COST_SD)
    a = np.asarray([dr.features(r + live, live, (r + live) * c, 0) for r in range(0, 12) for live in (1, 2, 4)
                    for c in (1_000, 150_000, 600_000)])
    b = a @ np.asarray([300.0, 90.0, 40.0, 3.0, 0.0])           # ten times the prior's costs: still settles
    theta, iterations, held = fit.bounded_fit(fitted, a, b)
    assert theta[:4] == pytest.approx([300.0, 90.0, 40.0, 3.0], rel=1e-3) and held == 0 and iterations > 2
    monkeypatch.setattr(fit, "FIT_ITERATIONS", 2)
    with pytest.raises(ValueError, match="did not settle"):
        fit.bounded_fit(fitted, a, b)
