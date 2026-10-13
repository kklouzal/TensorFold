"""The cost-aware draft rule's host math (``tensorfold.drafters.draft_rule``, design 3): the prior against today's
confidence rule, the joint chain step against brute force and its one-stream reduction, a step's expected value,
text-versus-chain values, the round-start trim against brute force, calibration files' validation, the small solver
against numpy, and the online cost model: its split row and step costs, outlier bounds, clamps, and that it is total
over valid observations (a decoder's drafting never raises from it)."""

import itertools
import json
import math
import random

import numpy as np
import pytest

from tensorfold.drafters import draft_rule as dr

PRIOR = dr.COST_PRIOR["unmixed"]
ROW = PRIOR["row"] + PRIOR["ctx"] * 0.3            # a draft row's ms at 30k tokens
STEP, STEP_STREAM = PRIOR["step"], PRIOR["step_stream"]
BAND = (dr.PRIOR_LOW_FIRST * dr.PRIOR_HIGH / (ROW + STEP), dr.PRIOR_LOW_FIRST / ROW)   # Predictor.prior's lambda*


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


def cost_rule(streams, pred, lam, row=ROW, step=STEP, extra=STEP_STREAM):
    """The cost-aware rule as ``MultiDecoder._chain`` runs it over streams ``[(ps, greedy, room)]`` (ms costs at
    ``lam`` tokens a ms): each stream's (drafts verified, picks computed)."""

    th, s, x = lam * row, lam * step, lam * extra
    c, kept, picks = [1.0] * len(streams), [0] * len(streams), [0] * len(streams)
    alive = list(range(len(streams)))
    for j in range(max(room for _, _, room in streams)):
        values = []
        for i in alive:
            ps, greedy, room = streams[i]
            picks[i] = j + 1
            table = pred.mtp_tables[greedy]
            c[i], include, value = dr.chain_step(c[i], table.chance(j, ps[j]), table, j, j + 1 < room, th)
            kept[i] += include
            if value is not None:
                values.append((i, value))
        alive = [i for (i, _), on in zip(values, dr.joint_step([v for _, v in values], s, x)) if on]
        if not alive:
            break
    return list(zip(kept, picks))


@pytest.mark.parametrize("confidence", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("lam", [BAND[0] * 1.001, 0.056, dr.LAMBDA_PRIOR, BAND[1]])
def test_the_prior_makes_todays_decisions_for_a_lone_stream(confidence, lam):
    """Every pattern of draft probabilities below/at/above the confidence, rooms 1 to 6, both policies: the prior's
    rule verifies the same drafts and computes the same picks as today's, across the band of lambda* its docstring
    states at the cost prior (the calibration log's lone streams' 0.056, and the server's prior, inside)."""

    assert BAND[0] < 0.056 < dr.LAMBDA_PRIOR < BAND[1]
    assert BAND == pytest.approx((0.0494, 0.0832), abs=1e-3)
    pred = dr.Predictor.prior(confidence)
    for room in range(1, 7):
        for pattern in itertools.product((confidence / 2, confidence, min(1.0, confidence * 1.5)), repeat=room):
            for greedy in (True, False):
                assert cost_rule([(pattern, greedy, room)], pred, lam) == [confidence_rule(pattern, confidence,
                                                                                           room)], (room, pattern)


def test_the_prior_without_a_confidence_verifies_every_draft_and_departs_where_the_design_says():
    pred = dr.Predictor.prior(0.0)
    for room in range(1, 9):
        assert cost_rule([([1e-9] * room, True, room)], pred, 0.056) == [(room, room)]
    low = dr.Predictor.prior(0.5)
    one = ([0.1, 0.9, 0.9], False, 3)
    assert cost_rule([one], low, 0.09) == [(0, 1)]                      # dear rows: a low first draft is dropped
    assert cost_rule([one], low, 0.056) == [(1, 1)]                     # alone it stops, as today
    assert cost_rule([one], low, 0.04) == [(3, 3)]                      # cheap: .35 x .9 pays a row and a step
    assert cost_rule([([0.9] * 6, False, 6)], low, 0.005) == [(6, 6)]   # cheap rows: deeper chains, as ever
    # at a lambda* where a stream alone stops after two drafts (.729 - .12 x 4.2 < .12 x 2.18), four such streams
    # chain on together to their rooms: each step's values add up to its cost (the joint step)
    four = [([0.9] * 4, False, 4)] * 4
    assert cost_rule(four[:1], low, 0.12) == [(2, 2)]
    assert cost_rule(four, low, 0.12) == [(4, 4)] * 4


def test_a_chain_step_reports_its_draft_and_the_next_steps_value():
    t = dr.Predictor.prior(0.5).mtp_tables[True]        # every depth's chance .9 at p >= .5, no samples
    assert dr.chain_step(1.0, 0.5, t, 0, True, 0.3) == (0.5, True, pytest.approx(0.5 * 0.9 - 0.3))
    assert dr.chain_step(1.0, 0.5, t, 0, False, 0.3) == (0.5, True, None)       # no room for another
    assert dr.chain_step(1.0, 0.29, t, 0, True, 0.3) == (0.29, False, None)     # not verified: the chain ends
    assert dr.chain_step(0.5, 0.5, t, 2, True, 0.3)[1] is False                 # .25 < .3: cumulative
    assert dr.chain_step(1.0, 0.3, t, 0, True, 0.3)[2] == 0.0                   # .27 < .3: the next never pays


def _best_set(values, step, extra):
    """Brute force: the largest gain of a step over every set of candidates (their values less the step's cost,
    ``step`` for the first and ``extra`` each further one, in tokens; 0: no step)."""

    best = 0.0
    for n in range(1, len(values) + 1):
        for subset in itertools.combinations(range(len(values)), n):
            best = max(best, sum(values[i] for i in subset) - step - extra * (n - 1))
    return best


def test_the_joint_step_takes_the_set_that_adds_most_and_runs_when_it_pays():
    rng = random.Random(11)
    for _ in range(4000):
        n = rng.randint(1, 5)
        step = rng.choice([0.05, 0.2, 0.4])
        extra = step * rng.choice([0.0, 0.1, 0.12, 0.5, 1.0])
        values = [rng.choice([0.0, 0.01, rng.uniform(0, 0.5), rng.uniform(0, 1.2)]) for _ in range(n)]
        joins = dr.joint_step(values, step, extra)
        gain = sum(v for v, on in zip(values, joins) if on) - (step + extra * (sum(joins) - 1) if any(joins) else 0)
        assert gain == pytest.approx(_best_set(values, step, extra), abs=1e-9), (values, step, extra, joins)
        assert not any(on and v <= 0 for v, on in zip(values, joins))      # a member is worth something
    assert dr.joint_step([], 0.2, 0.02) == []


def test_alone_the_joint_step_is_the_step_paying_for_itself():
    """One candidate goes on exactly when its value pays a whole step: with the prior's tables, today's ``c * abar >=
    row + step``."""

    rng = random.Random(5)
    table = dr.Predictor.prior(0.5).mtp_tables[False]
    for _ in range(20000):
        row, step = rng.uniform(0, 0.6), rng.uniform(1e-3, 0.3)
        extra, c = step * rng.uniform(0, 1), rng.uniform(0, 1)
        if abs(c * dr.PRIOR_HIGH - (row + step)) < 1e-12:
            continue
        assert dr.joint_step([table.value(1, c, row)], step, extra) == [c * dr.PRIOR_HIGH >= row + step]


def test_streams_that_only_pay_together_share_a_step_and_cheap_riders_join_one():
    step, extra = 0.27, 0.03                             # four streams' lambda* x (step, step_stream) ms
    assert dr.joint_step([0.2, 0.2, 0.2, 0.2], step, extra) == [True] * 4     # .2 < .27 each; together they pay
    assert dr.joint_step([0.2], step, extra) == [False]
    assert dr.joint_step([0.3, 0.02, 0.05], step, extra) == [True, False, True]   # .02 < its .03 share: left out
    assert dr.joint_step([0.1, 0.1, 0.0], step, extra) == [False] * 3        # .14 < .24, the rest of the step


def test_a_steps_value_is_the_next_drafts_expected_value():
    """``Table.value`` against brute force over the depth row's chances weighted by the fit's samples; without
    samples (the prior's tables) all weight sits on the row's optimistic chance."""

    rng = random.Random(3)
    for _ in range(300):
        bins = rng.randint(1, 6)
        rows = [sorted(round(rng.uniform(0, 1), 3) for _ in range(bins)) for _ in range(2)]
        counts = [[rng.choice([0, 0, 3, 50, 900]) for _ in range(bins)] for _ in range(2)]
        abar = [rng.uniform(0, 1), rng.uniform(0, 1)]
        edges = [e / 100 for e in sorted(rng.sample(range(1, 100), bins - 1))]
        table = dr.Table("p", dr.Calibration(rows, [0, 2], edges), abar, counts)
        for depth in (0, 1, 2, 3, 9):
            r = 0 if depth < 2 else 1
            c, row = rng.uniform(0, 1), rng.uniform(0, 0.8)
            total = sum(counts[r])
            spread = [(a, n / total) for a, n in zip(rows[r], counts[r])] if total else [(abar[r], 1.0)]
            want = sum(w * max(0.0, c * a - row) for a, w in spread)
            assert table.value(depth, c, row) == pytest.approx(want, abs=1e-12)
    prior = dr.Predictor.prior(0.5).mtp_tables[False]
    assert prior.value(1, 0.9, 0.3) == pytest.approx(0.9 * 0.9 - 0.3) and prior.value(1, 0.3, 0.3) == 0.0
    assert prior.value(1, 0.0, 0.0) == 0.0


def test_text_and_chain_values():
    kept, value = dr.suffix_plan([0.9, 0.9, 0.3, 1.0], 0.3)
    assert kept == pytest.approx([0.9, 0.81]) and value == pytest.approx(0.6 + 0.51)  # then .243 < .3: the end
    assert dr.suffix_plan([0.2, 1.0], 0.3) == ([], 0.0)
    # .9 - .3, then each deeper draft pays a row and a step (.4): .45 - .4 and .405 - .4
    assert dr.mtp_value(0.9, [0.9, 0.5, 0.9], 3, 0.3, 0.1) == pytest.approx(0.6 + 0.05 + 0.005)
    assert dr.mtp_value(0.9, [0.9, 0.4, 0.9], 3, 0.3, 0.1) == pytest.approx(0.6)    # .36 < .4: no deeper value
    assert dr.mtp_value(0.25, [0.9] * 3, 3, 0.3, 0.1) == 0.0
    prior = dr.Predictor.prior(0.5)                      # a copy (8+ tokens) beats any MTP chain; a shorter one none
    row, step = 0.056 * ROW, 0.056 * STEP
    for room in range(1, 6):
        given = [prior.suffix(False, k, 8) for k in range(room)]
        assert dr.suffix_plan(given, row)[1] > dr.mtp_value(0.9, [prior.abar("mtp", False, j) for j in range(room)],
                                                            room, row, step)
        assert dr.suffix_plan([prior.suffix(False, k, 7) for k in range(room)], row) == ([], 0.0)


def _best(chances, thresholds):
    """Brute force: the largest sum of chances over each stream's prefixes of drafts at or above its threshold."""

    best = None
    for counts in itertools.product(*[range(len(chain) + 1) for chain in chances]):
        if any(any(chain[j] < t for j in range(n)) for chain, t, n in zip(chances, thresholds, counts)):
            continue
        value = sum(sum(chain[:n]) for chain, n in zip(chances, counts))
        if best is None or value > best[0] + 1e-12:
            best = (value, counts)
    return best


def test_the_trim_keeps_each_streams_drafts_that_pay_at_the_rounds_threshold():
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
        kept = dr.trim(chances, thresholds)
        assert all(0 <= n <= len(c) for n, c in zip(kept, chances))
        assert all(c[j] >= t for c, t, n in zip(chances, thresholds, kept) for j in range(n))
        assert tuple(kept) == _best(chances, thresholds)[1]           # every draft that pays, nothing else
    assert dr.trim([[0.2], [1.0, 1.0]], [0.3, 0.3]) == [0, 2]          # a zero-draft stream
    assert dr.trim([[0.9, 0.5, 0.4]], [0.45]) == [2]


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
    bad(lambda d: d.update(v=1), "version 2")           # version 1's cost block had one row slope
    bad(lambda d: d.update(extra=1), "unknown fields")
    bad(lambda d: d.pop("source"), "provenance")
    bad(lambda d: d["tables"].pop("suffix"), "one object a source")
    bad(lambda d: d["tables"]["mtp"].pop("greedy"), "one table a policy")
    bad(lambda d: table(d).update(feature="L"), "feature")
    bad(lambda d: table(d).update(depth_edges=[1, 2]), "depth_edges")
    bad(lambda d: table(d).update(depth_edges=[0, 65]), "at most 64")          # lookups hold a row a depth
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
    bad(lambda d: d["cost"]["mixed"].pop("step_stream"), "cost.mixed")
    bad(lambda d: d["cost"]["unmixed"].update(row=2e6), "at most 1e\\+06")
    bad(lambda d: d["cost"]["mixed"].update(step=-0.1), "a step's from 0")
    bad(lambda d: d["cost"]["unmixed"].update(step_stream=-0.1), "a step's from 0")
    with pytest.raises(ValueError, match="probability"):
        dr.Predictor.prior(1.5)


def test_a_calibration_files_read_failures_name_the_file_and_keep_their_cause(tmp_path, monkeypatch):
    """Startup fails with the variable and path for a missing file, one past MAX_FILE bytes, text that is not UTF-8
    JSON, and JSON nested past the parser's depth; the underlying error stays the cause."""

    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match=f"{dr.ENV}={missing}") as info:
        dr.Predictor.load(missing)
    assert isinstance(info.value.__cause__, FileNotFoundError)
    monkeypatch.setenv(dr.ENV, str(missing))
    with pytest.raises(ValueError, match=dr.ENV):
        dr.Predictor.from_env(0.5)
    path = tmp_path / "rule.json"
    path.write_text(json.dumps(_file()))
    monkeypatch.setattr(dr, "MAX_FILE", len(path.read_bytes()) - 1)
    with pytest.raises(ValueError, match="larger than"):
        dr.Predictor.load(path)
    monkeypatch.setattr(dr, "MAX_FILE", 16 << 20)
    assert dr.Predictor.load(path).as_dict() == _file()
    path.write_bytes(b"\xff\xfe{")
    with pytest.raises(ValueError, match=dr.ENV):
        dr.Predictor.load(path)
    path.write_text("[" * 200_000 + "]" * 200_000)
    with pytest.raises(ValueError, match=dr.ENV):
        dr.Predictor.load(path)


def test_the_small_solver_matches_numpy_and_says_when_a_system_is_singular():
    """``solve`` (Cholesky, scaled) against numpy's general solver on random anchored normal equations, badly scaled
    ones included; a system singular to working precision, or not finite, returns None."""

    rng = np.random.default_rng(1)
    for k in (1, 2, 5):
        for _ in range(300):
            x = rng.normal(size=(int(rng.integers(0, 40)), k)) * rng.choice([1.0, 1e3, 1e-3], size=k)
            m = x.T @ x
            diag = rng.uniform(1e-4, 1.0, size=k)
            v = rng.normal(size=k) * 100
            want = np.linalg.solve(m + np.diag(diag), v)
            got = dr.solve(m.tolist(), diag.tolist(), v.tolist())
            assert got is not None and np.allclose(got, want, rtol=1e-7, atol=1e-9 * np.abs(want).max())
    assert dr.solve([[1e300, 0.0], [1e300, 1e300]], [1.0, 1.0], [1.0, 1.0]) is None       # rank one to rounding
    assert dr.solve([[math.inf]], [1.0], [1.0]) is None
    assert dr.solve([[math.nan]], [1.0], [1.0]) is None


def _rounds(rng, truth, step, mixed, count=400, lives=(1, 2, 3, 4)):
    """Rounds of a known cost (``truth`` over ``dr.features``: t0, stream, row, ctx, pass) plus chain steps of
    ``step`` = (first stream, each further) ms, with noise: (rows, live, sum of rows' contexts, prompt rows, ms, chain
    ms, chain steps, their streams)."""

    out = []
    for _ in range(count):
        live = rng.choice(lives)
        rows = live + rng.randint(0, 4 * live)
        joined = [rng.randint(1, live) for _ in range(rng.randint(0, 4))]
        ctx = rng.randint(1_000, 600_000)
        passed = rng.randint(64, 512) if mixed else 0
        chain = sum(step[0] + step[1] * (n - 1) + rng.gauss(0, 0.05) for n in joined)
        ms = float(np.dot(dr.features(rows, live, rows * ctx, passed), truth)) + chain + rng.gauss(0, 0.2)
        out.append((rows, live, rows * ctx, passed, ms, chain, len(joined), sum(joined)))
    return out


def _observe(model, rounds, mixed, tokens=3, context=100_000):
    for rows, live, ctx_rows, passed, ms, chain, steps, stream_steps in rounds:
        model.observe(rows=rows, live=live, ctx_rows=ctx_rows, pass_rows=passed, mixed=mixed, ms=ms, chain_ms=chain,
                      chain_steps=steps, chain_rows=stream_steps, tokens=tokens, context=context)


def test_the_cost_model_learns_each_regimes_costs_and_lambda_per_region():
    """A stream's own cost apart from its draft rows', and a step's first stream apart from each further one."""

    rng = random.Random(1)
    model = dr.CostModel()
    truths = {False: ([20.0, 7.0, 3.0, 0.4, 0.0], (0.9, 0.2)), True: ([25.0, 4.0, 0.5, 0.4, 1.2], (1.3, 0.3))}
    for mixed, (truth, step) in truths.items():
        _observe(model, _rounds(rng, np.asarray(truth), step, mixed), mixed)
    unmixed = model.fits[False].coef
    assert unmixed[:4] == pytest.approx(truths[False][0][:4], abs=0.15)
    assert unmixed[4] == pytest.approx(dr.COST_PRIOR["unmixed"]["pass"], abs=1e-6)   # never excited: the prior's
    assert model.fits[True].coef == pytest.approx(truths[True][0], abs=0.15)
    assert model.row_ms(100_000, False) == pytest.approx(3.0 + 0.4, abs=0.2)          # a draft row's, not 7 + 3
    for mixed, (_, step) in truths.items():
        assert (model.step_ms(mixed), model.step_stream_ms(mixed)) == pytest.approx(step, abs=0.03)
    regions = dr.CostModel()                             # lambda* by region: its own, else the nearest live count's
    _observe(regions, _rounds(rng, np.asarray(truths[False][0]), (0.9, 0.2), False, 50, lives=(1,)), False, 2)
    _observe(regions, _rounds(rng, np.asarray(truths[False][0]), (0.9, 0.2), False, 50, lives=(4,)), False, 9)
    own = {n: regions.regions[regions.region(n, False, 100_000)] for n in (1, 4)}
    assert regions.lam(1, False, 100_000) == pytest.approx(own[1][0] / own[1][1])
    assert regions.lam(4, False, 100_000) == pytest.approx(own[4][0] / own[4][1])
    assert regions.lam(2, False, 100_000) == regions.lam(1, False, 100_000)          # nearest (fewer on ties)
    assert regions.lam(3, False, 100_000) == regions.lam(4, False, 100_000)
    assert regions.lam(2, False, 700_000) == pytest.approx(regions.overall[0] / regions.overall[1])   # every round's
    assert dr.CostModel().lam(2, False, 100_000) == dr.LAMBDA_PRIOR       # nothing seen: the prior


def test_a_folded_row_slope_would_overprice_draft_rows_where_streams_share_rounds():
    """The defect the split form removes: rounds of one to four streams whose own cost (9 ms) is far above a draft
    row's (4 ms). The split fit recovers the draft row; one slope over every row (version 1's) does not."""

    rng = random.Random(8)
    rounds = _rounds(rng, np.asarray([38.0, 9.0, 4.0, 0.12, 0.0]), (2.2, 0.25), False, 1200)
    model = dr.CostModel()
    _observe(model, rounds, False)
    assert model.row_ms(0, False) == pytest.approx(4.0, abs=0.1)
    a = np.asarray([[1.0, r[0] - 1.0, r[2] / 1e5] for r in rounds])
    folded = np.linalg.lstsq(a, np.asarray([r[4] - r[5] for r in rounds]), rcond=None)[0]
    assert folded[1] > 4.4               # a stream's own cost folded into every row's price (the log: 5.42 vs 4.17)


def test_steps_measured_apart_keep_their_cost_when_decisions_tie_rows_to_steps():
    """Under the rule's feedback a round's steps follow its rows (every stream that drafts deep also chains): a
    regression on both cannot tell them apart, measured steps can."""

    rng = random.Random(4)
    model = dr.CostModel()
    rounds = []
    for _ in range(600):
        drafts = rng.randint(0, 4)
        joined = [rng.randint(1, 2) for _ in range(drafts)]                        # one or two streams a step
        chain = sum(2.0 + 0.3 * (n - 1) + rng.gauss(0, 0.05) for n in joined)
        rows = 2 + sum(joined)
        rounds.append((rows, 2, rows * 2000, 0, 6.0 + 0.1 * (rows - 2) + chain + rng.gauss(0, 0.3), chain, drafts,
                       sum(joined)))
    _observe(model, rounds, False, context=2000)
    assert (model.step_ms(False), model.step_stream_ms(False)) == pytest.approx((2.0, 0.3), abs=0.05)
    assert model.row_ms(2000, False) == pytest.approx(0.1 + dr.COST_PRIOR["unmixed"]["ctx"] * 0.02, abs=0.1)


def test_the_cost_model_forgets_bounds_outliers_and_clamps():
    rng = random.Random(2)
    model = dr.CostModel()
    _observe(model, _rounds(rng, np.asarray([20.0, 5.0, 3.0, 0.0, 0.0]), (1.0, 0.1), False, 600), False, 2, 1000)
    _observe(model, _rounds(rng, np.asarray([20.0, 5.0, 6.0, 0.0, 0.0]), (1.0, 0.1), False, 1500), False, 2, 1000)
    assert model.fits[False].coef[2] == pytest.approx(6.0, abs=0.2)      # the newer cost wins
    before = list(model.fits[False].coef)
    model.observe(rows=2, live=1, ctx_rows=2000, pass_rows=0, mixed=False, ms=5000.0, chain_ms=0.0, chain_steps=0,
                  chain_rows=0, tokens=2, context=1000)
    assert max(abs(a - b) for a, b in zip(model.fits[False].coef, before)) < 1.0    # a 5 s pause moves it boundedly
    for _ in range(200):                                 # absurd measurements: estimates stay within the clamps
        model.observe(rows=12, live=1, ctx_rows=0, pass_rows=0, mixed=True, ms=1e-6, chain_ms=1e-6, chain_steps=3,
                      chain_rows=3, tokens=900, context=1)
    assert model.row_ms(0, True) >= dr.ROW_MS[0] and model.lam(1, True, 1) <= dr.LAMBDA[1]
    assert dr.STEP_MS[0] <= model.step_stream_ms(True) <= model.step_ms(True)
    for bad in (dict(ms=0.0), dict(ms=-1.0), dict(ms=math.nan), dict(ms=math.inf), dict(chain_ms=6.0),
                dict(chain_ms=-1.0), dict(live=0), dict(rows=1, live=2), dict(chain_rows=0), dict(chain_steps=0),
                dict(ctx_rows=-1.0), dict(ctx_rows=math.nan), dict(pass_rows=-1), dict(tokens=-1)):
        kw = dict(rows=2, live=1, ctx_rows=0, pass_rows=0, mixed=False, ms=5.0, chain_ms=1.0, chain_steps=1,
                  chain_rows=1, tokens=1, context=1)
        kw.update(bad)
        with pytest.raises(ValueError, match="observation"):
            model.observe(**kw)
    seeded = dr.CostModel({"unmixed": {**dr.COST_PRIOR["unmixed"], "row": 2.0, "ctx": 0.0, "step": 0.7,
                                       "step_stream": 0.9},
                           "mixed": dict(dr.COST_PRIOR["mixed"]), "lambda": 0.1})
    assert (seeded.row_ms(0, False), seeded.step_ms(False), seeded.lam(1, False, 0)) == (2.0, 0.7, 0.1)
    assert seeded.step_stream_ms(False) == 0.7          # a further stream's share is at most a step's


def test_a_paused_chain_step_moves_the_step_estimate_boundedly():
    """The steps' fit holds each round's steps within the outlier bound of their prediction, as the round fit does:
    a 5 s pause in one step moves it boundedly; a slower true step still comes through, a bound a round."""

    model = dr.CostModel()
    for _ in range(100):
        model.observe(rows=4, live=1, ctx_rows=4000, pass_rows=0, mixed=False, ms=40.0, chain_ms=4.0, chain_steps=2,
                      chain_rows=2, tokens=4, context=1000)
    assert model.step_ms(False) == pytest.approx(2.0, abs=1e-2)
    model.observe(rows=4, live=1, ctx_rows=4000, pass_rows=0, mixed=False, ms=5040.0, chain_ms=5002.0, chain_steps=2,
                  chain_rows=2, tokens=4, context=1000)
    assert model.step_ms(False) - 2.0 < 0.1
    slow = dr.CostModel()
    for _ in range(1000):
        slow.observe(rows=2, live=1, ctx_rows=2000, pass_rows=0, mixed=True, ms=80.0, chain_ms=40.0, chain_steps=2,
                     chain_rows=2, tokens=2, context=1000)
    assert slow.step_ms(True) == pytest.approx(20.0, rel=1e-3)


def test_the_cost_model_is_total_over_valid_observations():
    """Any valid observation (extreme or degenerate rows, streams, contexts, prompt rows, times and tokens) leaves
    every estimate finite and within its clamps, and the decisions built on them never raise: a decoder's drafting
    cannot fail from its cost model."""

    rng = random.Random(23)
    model = dr.CostModel()
    table = dr.Predictor.prior(0.5).mtp_tables[False]
    for _ in range(3000):
        live = rng.choice([1, 2, 4, 9])
        rows = live + rng.choice([0, 1, 4, 40, 2048])
        ms = rng.choice([1e-6, 0.01, 5.0, 50.0, 4e3, 1e7]) * rng.uniform(0.5, 1.0)
        chain_steps = rng.choice([0, 1, 3, 15])
        chain_ms = ms * rng.choice([0.0, 0.5, 1.0]) if chain_steps else 0.0
        model.observe(rows=rows, live=live, ctx_rows=rows * rng.choice([0, 1, 10**6, 4 * 10**7]),
                      pass_rows=rng.choice([0, 128, 8192]), mixed=rng.random() < 0.3, ms=ms, chain_ms=chain_ms,
                      chain_steps=chain_steps, chain_rows=chain_steps * rng.choice([1, 2, 4]),
                      tokens=rng.choice([0, 1, 4, 4096]), context=rng.choice([0, 1, 10**6, 10**8]))
        for mixed in (False, True):
            assert all(math.isfinite(c) for fit in (model.fits[mixed], model.steps[mixed]) for c in fit.coef)
            context, live = rng.choice([0, 10**5, 10**8]), rng.randint(0, 9)
            lam = model.lam(live, mixed, context)
            assert dr.LAMBDA[0] <= lam <= dr.LAMBDA[1]
            assert dr.ROW_MS[0] <= model.row_ms(context, mixed) <= dr.ROW_MS[1]
            assert dr.STEP_MS[0] <= model.step_stream_ms(mixed) <= model.step_ms(mixed) <= dr.STEP_MS[1]
            row, step, extra = (lam * model.row_ms(context, mixed), lam * model.step_ms(mixed),
                                lam * model.step_stream_ms(mixed))
            _, _, value = dr.chain_step(1.0, rng.random(), table, rng.randint(0, 3), True, row)
            dr.joint_step([value or 0.0, rng.random()], step, extra)
    assert model.fits[False].singular == model.fits[True].singular == 0


def test_a_singular_update_keeps_the_estimate_and_is_counted():
    fit = dr.CostFit([1.0, 2.0], [1.0, 1.0])
    fit.observe([1.0, 1.0], 3.0)
    before = list(fit.coef)
    fit.observe([1e300, 1e300], 1e300)                   # past any round's: singular to working precision
    assert fit.coef == before and fit.singular == 1 and fit.rounds == 2


def test_requests_split_into_stable_halves():
    rids = [f"{i:016x}" for i in range(2000)]
    halves = [dr.half(rid) for rid in rids]
    assert halves == [dr.half(rid) for rid in rids] and set(halves) == {0, 1}
    assert 900 < sum(halves) < 1100
