"""The cost-aware draft rule's host math (``tensorfold.drafters.draft_rule``, design 3): the prior against today's
confidence rule, the stop rule's thresholds and shared steps, text-versus-chain values, the round-start trim against
brute force, calibration files' validation and the online cost model."""

import itertools
import json
import math
import random

import numpy as np
import pytest

from tensorfold.drafters import draft_rule as dr

LAM, ROW, STEP = dr.LAMBDA_PRIOR, 6.5, 1.6


def th(lam=LAM, row=ROW, step=STEP) -> dr.Thresholds:
    r, s = lam * row, lam * step
    return dr.Thresholds(r, r + s, r + dr.STEP_SHARED * s)


def confidence_rule(ps, confidence, room):
    """Today's ``_chain`` for one stream: (drafts verified, picks computed)."""

    kept = picks = 0
    for j in range(room):
        picks = j + 1
        low = confidence > 0 and ps[j] < confidence
        if not low or j == 0:
            kept += 1
        if low:
            break
    return kept, picks


def cost_rule(ps, pred, greedy, room, t, shared=False):
    """The cost-aware rule for one stream (``shared``: another stream forces every step)."""

    c, kept, picks = 1.0, 0, 0
    for j in range(room):
        picks = j + 1
        c, include, full, share = dr.chain_step(c, pred.mtp(greedy, j, ps[j], None), pred.abar("mtp", greedy, j + 1),
                                                j + 1 < room, t)
        kept += include
        if not (full or (shared and share)):
            break
    return kept, picks


@pytest.mark.parametrize("confidence", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("lam", [0.0395, LAM, 0.0526, 0.0538])
def test_the_prior_makes_todays_decisions_for_a_lone_stream(confidence, lam):
    """Every pattern of draft probabilities below/at/above the confidence, rooms 1 to 6, both policies: the prior's
    rule verifies the same drafts and computes the same picks as today's, wherever lambda* x row_ms is in (0.253,
    0.35] (lambda* 0.039 .. 0.054 at 6.5 ms a row)."""

    pred = dr.Predictor.prior(confidence)
    for room in range(1, 7):
        for pattern in itertools.product((confidence / 2, confidence, min(1.0, confidence * 1.5)), repeat=room):
            for greedy in (True, False):
                assert cost_rule(pattern, pred, greedy, room, th(lam)) == confidence_rule(pattern, confidence, room), \
                    (room, pattern)


def test_the_prior_without_a_confidence_verifies_every_draft_to_nine_and_departs_where_the_design_says():
    pred = dr.Predictor.prior(0.0)
    for room in range(1, 10):
        assert cost_rule([1e-9] * room, pred, True, room, th()) == (room, room)
    assert cost_rule([1e-9] * 12, pred, True, 12, th()) == (9, 9)        # the 9th's .9^9 x .9 under the step's .38
    low = dr.Predictor.prior(0.5)
    assert cost_rule([0.1, 0.9, 0.9], low, False, 3, th(0.06)) == (0, 1)   # dear rows: a low first draft is dropped
    assert cost_rule([0.1, 0.9, 0.9], low, False, 3, th(0.04), shared=True) == (3, 3)   # it rides forced steps
    assert cost_rule([0.1, 0.9, 0.9], low, False, 3, th(0.04)) == (1, 1)                # alone it stops, as today
    assert cost_rule([0.9] * 6, low, False, 6, th(0.005)) == (6, 6)        # cheap rows: deeper chains, as ever


def test_chain_steps_go_on_alone_or_in_a_step_another_stream_forces():
    t = dr.Thresholds(0.3, 0.4, 0.32)
    assert dr.chain_step(1.0, 0.5, 0.9, True, t) == (0.5, True, True, True)          # .45 >= .4
    assert dr.chain_step(1.0, 0.4, 0.9, True, t) == (0.4, True, False, True)         # .36: only a forced step
    assert dr.chain_step(1.0, 0.4, 0.9, False, t) == (0.4, True, False, False)       # no room for another
    assert dr.chain_step(1.0, 0.29, 1.0, True, t) == (0.29, False, False, False)     # not verified: the chain ends
    assert dr.chain_step(0.5, 0.5, 1.0, True, t)[1] is False                          # .25 < .3: cumulative


def test_text_and_chain_values():
    t = dr.Thresholds(0.3, 0.4, 0.32)
    kept, value = dr.suffix_plan([0.9, 0.9, 0.3, 1.0], t)
    assert kept == pytest.approx([0.9, 0.81]) and value == pytest.approx(0.6 + 0.51)  # then .243 < .3: the end
    assert dr.suffix_plan([0.2, 1.0], t) == ([], 0.0)
    # .9 - .3, then each deeper draft pays a row and a step (.4): .45 - .4 and .405 - .4
    assert dr.mtp_value(0.9, [0.9, 0.5, 0.9], 3, t) == pytest.approx(0.6 + 0.05 + 0.005)
    assert dr.mtp_value(0.9, [0.9, 0.4, 0.9], 3, t) == pytest.approx(0.6)               # .36 < .4: no deeper value
    assert dr.mtp_value(0.25, [0.9] * 3, 3, t) == 0.0
    prior = dr.Predictor.prior(0.5)                      # a copy (8+ tokens) beats any MTP chain; a shorter one none
    for room in range(1, 6):
        given = [prior.suffix(False, k, 8) for k in range(room)]
        assert dr.suffix_plan(given, th())[1] > dr.mtp_value(0.9, [prior.abar("mtp", False, j) for j in range(room)],
                                                             room, th())
        assert dr.suffix_plan([prior.suffix(False, k, 7) for k in range(room)], th()) == ([], 0.0)


def _best(chances, thresholds, budget):
    """Brute force: the largest sum of chances over each stream's prefixes of drafts above their thresholds, at most
    ``budget`` drafts in all."""

    best = None
    for counts in itertools.product(*[range(len(chain) + 1) for chain in chances]):
        if sum(counts) > budget or any(any(chain[j] < t for j in range(n)) for chain, t, n in
                                       zip(chances, thresholds, counts)):
            continue
        value = sum(sum(chain[:n]) for chain, n in zip(chances, counts))
        if best is None or value > best[0] + 1e-12:
            best = (value, counts)
    return best[0]


def test_the_trim_keeps_the_highest_prefixes_that_pay_within_the_rows():
    rng = random.Random(7)
    for _ in range(300):
        chances = []
        for _ in range(rng.randint(1, 4)):
            c, chain = 1.0, []
            for _ in range(rng.randint(0, 4)):
                c *= rng.choice([1.0, 0.95, 0.9, 0.6, 0.35])
                chain.append(c)
            chances.append(chain)
        thresholds = [rng.choice([0.0, 0.3, 0.5]) for _ in chances]
        budget = rng.randint(0, 10)
        kept = dr.trim(chances, thresholds, budget)
        assert sum(kept) <= budget and all(0 <= n <= len(c) for n, c in zip(kept, chances))
        assert all(c[j] >= t for c, t, n in zip(chances, thresholds, kept) for j in range(n))
        assert sum(sum(c[:n]) for c, n in zip(chances, kept)) == pytest.approx(_best(chances, thresholds, budget))
        if budget:
            fewer = dr.trim(chances, thresholds, budget - 1)
            assert all(a <= b for a, b in zip(fewer, kept))          # a smaller budget takes, never adds
    assert dr.trim([[0.9, 0.9], [0.9]], [0.0, 0.0], 2) == [1, 1]       # ties: the shallower first, then stream order
    assert dr.trim([[0.2], [1.0, 1.0]], [0.3, 0.3], 9) == [0, 2]       # a zero-draft stream


def _file(**over):
    data = dr.Predictor.prior(0.5).as_dict()
    data["source"] = {"tool": "test"}
    for key, value in over.items():
        data[key] = value
    return data


def test_a_calibration_file_round_trips_and_is_validated(tmp_path, monkeypatch):
    data = _file(cost={"unmixed": dict(dr.COST_PRIOR["unmixed"]), "mixed": dict(dr.COST_PRIOR["mixed"]),
                       "lambda": 0.05})
    path = tmp_path / "rule.json"
    path.write_text(json.dumps(data))
    pred = dr.Predictor.load(path)
    assert pred.as_dict() == data and pred.cost["lambda"] == 0.05 and not pred.needs_margins
    monkeypatch.setenv(dr.ENV, str(path))
    assert dr.Predictor.from_env(0.9).as_dict() == data                # the file, not the confidence's prior
    monkeypatch.delenv(dr.ENV)
    assert dr.Predictor.from_env(0.9).source == {"prior": "confidence rule", "confidence": 0.9}
    margins = json.loads(json.dumps(data))
    margins["tables"]["mtp"]["sampled"].update(feature="m", score_edges=[0.5, 2.0], table=[[0.2, 0.5, 0.9]] * 2)
    pred = dr.Predictor.from_dict(margins)
    assert pred.needs_margins and pred.mtp(False, 0, 0.01, math.inf) == 0.9 and pred.mtp(False, 1, 0.9, 0.1) == 0.2
    with pytest.raises(ValueError, match="needs its 'm'"):
        pred.mtp(False, 0, 0.5, None)

    def bad(change, match):
        broken = json.loads(json.dumps(data))
        change(broken)
        with pytest.raises(ValueError, match=match):
            dr.Predictor.from_dict(broken)
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match=dr.ENV):
            dr.Predictor.load(path)

    table = lambda d: d["tables"]["mtp"]["greedy"]  # noqa: E731
    bad(lambda d: d.update(v=2), "version 1")
    bad(lambda d: d.update(extra=1), "unknown fields")
    bad(lambda d: d.pop("source"), "provenance")
    bad(lambda d: d["tables"].pop("suffix"), "one object a source")
    bad(lambda d: d["tables"]["mtp"].pop("greedy"), "one table a policy")
    bad(lambda d: table(d).update(feature="L"), "feature")
    bad(lambda d: table(d).update(depth_edges=[1, 2]), "depth_edges")
    bad(lambda d: table(d).update(score_edges=[0.5, 0.5]), "strictly ascending")
    bad(lambda d: table(d).update(score_edges=["x"]), "finite number")
    bad(lambda d: table(d).update(table=[[0.9, 0.35], [0.2, 0.9]]), "non-decreasing")
    bad(lambda d: table(d).update(table=[[0.35, 1.5], [0.2, 0.9]]), "from 0 to 1")
    bad(lambda d: table(d).update(table=[[0.35, float("nan")], [0.2, 0.9]]), "finite number")
    bad(lambda d: table(d).update(table=[[0.35, 0.9]]), "a row a depth bin")
    bad(lambda d: table(d).update(abar=[0.9]), "abar")
    bad(lambda d: table(d).update(count=[[1, -1], [0, 0]]), "count")
    bad(lambda d: d["cost"].update(lam=1), "cost")
    bad(lambda d: d["cost"].update(**{"lambda": 9.0}), "tokens a ms")
    bad(lambda d: d["cost"]["mixed"].pop("pass"), "cost.mixed")
    with pytest.raises(ValueError, match="probability"):
        dr.Predictor.prior(1.5)


def _rounds(rng, truth, step, mixed, count=400):
    """Rounds of a known linear cost (``truth`` over ``dr.features``) plus chain steps of ``step`` ms each, with
    noise: (rows, sum of rows' contexts, prompt rows, ms, chain ms, chain steps)."""

    out = []
    for _ in range(count):
        rows, steps = rng.randint(1, 12), rng.randint(0, 4)
        ctx = rng.randint(1_000, 600_000)
        passed = rng.randint(64, 512) if mixed else 0
        chain = sum(step + rng.gauss(0, 0.05) for _ in range(steps))
        x = dr.features(rows, rows * ctx, passed)
        out.append((rows, rows * ctx, passed, float(x @ truth) + chain + rng.gauss(0, 0.2), chain, steps))
    return out


def _observe(model, rounds, mixed, tokens=3, live=2, context=100_000):
    for rows, ctx_rows, passed, ms, chain, steps in rounds:
        model.observe(rows=rows, ctx_rows=ctx_rows, pass_rows=passed, mixed=mixed, ms=ms, chain_ms=chain,
                      chain_steps=steps, tokens=tokens, live=live, context=context)


def test_the_cost_model_learns_each_regimes_costs_and_lambda_per_region():
    rng = random.Random(1)
    model = dr.CostModel()
    truths = {False: (np.asarray([20.0, 3.0, 0.4, 0.0]), 0.9), True: (np.asarray([25.0, 0.5, 0.4, 1.2]), 1.3)}
    for mixed, (truth, step) in truths.items():
        _observe(model, _rounds(rng, truth, step, mixed), mixed)
    unmixed = model.fits[False].theta
    assert unmixed[:3] == pytest.approx(truths[False][0][:3], abs=0.15)
    assert unmixed[3] == pytest.approx(dr.COST_PRIOR["unmixed"]["pass"], abs=1e-6)   # never excited: the prior's
    assert model.fits[True].theta == pytest.approx(truths[True][0], abs=0.15)
    assert model.row_ms(100_000, False) == pytest.approx(3.0 + 0.4, abs=0.2)
    assert (model.step_ms(False), model.step_ms(True)) == pytest.approx((0.9, 1.3), abs=0.05)   # measured steps
    region = model.regions[model.region(2, False, 100_000)]
    assert model.lam(2, False, 100_000) == pytest.approx(region[0] / region[1])
    assert model.lam(3, False, 100_000) == model.lam(2, False, 100_000)    # unseen: the nearest live count's
    assert model.lam(2, False, 700_000) == pytest.approx(model.overall[0] / model.overall[1])   # else every round's
    assert dr.CostModel().lam(2, False, 100_000) == dr.LAMBDA_PRIOR       # nothing seen: the prior
    _observe(model, _rounds(rng, truths[False][0], 0.9, False, 1), False, live=4)
    assert model.regions[model.region(4, False, 100_000)][0] / dr.PRIOR_MS == pytest.approx(
        model.lam(2, False, 100_000), rel=0.2)                              # a new region starts from its nearest
    t = model.thresholds(0.05, 100_000, False)
    assert (t.row, t.full) == pytest.approx((0.05 * model.row_ms(100_000, False),
                                             0.05 * (model.row_ms(100_000, False) + model.step_ms(False))))


def test_steps_measured_apart_keep_their_cost_when_decisions_tie_rows_to_steps():
    """Under the rule's feedback a round's steps follow its rows (every stream that drafts deep also chains): a
    regression on both cannot tell them apart, a measured step can."""

    rng = random.Random(4)
    model = dr.CostModel()
    rounds = []
    for _ in range(600):
        drafts = rng.randint(0, 4)
        chain = sum(2.0 + rng.gauss(0, 0.05) for _ in range(drafts))
        rows = 4 * (1 + drafts)
        rounds.append((rows, rows * 2000, 0, 6.0 + 0.1 * (rows - 1) + chain + rng.gauss(0, 0.3), chain, drafts))
    _observe(model, rounds, False, live=4, context=2000)
    assert model.step_ms(False) == pytest.approx(2.0, abs=0.05)
    assert model.row_ms(2000, False) == pytest.approx(0.1 + 0.22 * 0.02, abs=0.1)


def test_the_cost_model_forgets_bounds_outliers_and_clamps():
    rng = random.Random(2)
    model = dr.CostModel()
    _observe(model, _rounds(rng, np.asarray([20.0, 3.0, 0.0, 0.0]), 1.0, False, 600), False, 2, 1, 1000)
    _observe(model, _rounds(rng, np.asarray([20.0, 6.0, 0.0, 0.0]), 1.0, False, 1500), False, 2, 1, 1000)
    assert model.fits[False].theta[1] == pytest.approx(6.0, abs=0.2)      # the newer cost wins
    before = model.fits[False].theta.copy()
    model.observe(rows=2, ctx_rows=2000, pass_rows=0, mixed=False, ms=5000.0, chain_ms=0.0, chain_steps=0, tokens=2,
                  live=1, context=1000)
    assert np.abs(model.fits[False].theta - before).max() < 1.0           # a 5 s pause moves it boundedly
    for _ in range(200):                                 # absurd measurements: estimates stay within the clamps
        model.observe(rows=12, ctx_rows=0, pass_rows=0, mixed=True, ms=1e-6, chain_ms=1e-6, chain_steps=3,
                      tokens=900, live=1, context=1)
    assert model.row_ms(0, True) >= dr.ROW_MS[0] and model.lam(1, True, 1) <= dr.LAMBDA[1]
    assert model.step_ms(True) >= dr.STEP_MS[0]
    for bad, chain in ((0.0, 0.0), (-1.0, 0.0), (math.nan, 0.0), (math.inf, 0.0), (5.0, 6.0), (5.0, -1.0)):
        with pytest.raises(ValueError, match="observation"):
            model.observe(rows=1, ctx_rows=0, pass_rows=0, mixed=False, ms=bad, chain_ms=chain, chain_steps=1,
                          tokens=1, live=1, context=1)
    seeded = dr.CostModel({"unmixed": {**dr.COST_PRIOR["unmixed"], "row": 2.0, "step": 0.7},
                           "mixed": dict(dr.COST_PRIOR["mixed"]), "lambda": 0.1})
    assert (seeded.row_ms(0, False), seeded.step_ms(False), seeded.lam(1, False, 0)) == (2.0, 0.7, 0.1)
