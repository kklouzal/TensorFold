"""Dense adapter controls on CPU: real probability/rejection core, stand-in target/companion math only.

These qualify packet/routing/history/lifetime/schedule integration, not native CUDA numerics or performance.
"""

import importlib
import queue
import threading
import weakref
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tensorfold.cuda import sampling as target_sampling
from tensorfold.engine.call_gate import generate_gated
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.rejection import ChainPacket, ProbabilityRow
from tests import test_cuda_geometry

allocations = test_cuda_geometry.allocations


@pytest.fixture
def dense(allocations, monkeypatch):
    decode = importlib.import_module("tensorfold.families.qwen3_5.cuda.decode")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    # The CPU stand-in preserves all policy validation; only the actual CUDA device admission is replaced.
    monkeypatch.setattr(target_sampling, "validate_rows",
                        lambda logits, positions, sampling: target_sampling.validate_policy(positions, sampling))

    def forward(w, tokens, parents, state, **kwargs):
        logits = torch.tensor([[np.log(.3), np.log(.7)]] * len(parents), dtype=torch.float32)
        return logits, None, torch.zeros((len(parents), 1))

    monkeypatch.setattr(decode, "tree_forward", forward)
    monkeypatch.setattr(decode, "commit", lambda state, record, path, **kw: setattr(state, "pos", state.pos + len(path)))
    return decode


class Draft:
    trained = 4

    def __init__(self):
        self.calls = []
        self.taps = []

    def propose_chain(self, pending, position, nodes, sampling, history=None):
        self.calls.append((pending, position, nodes))
        q = ProbabilityRow([0], [1.0])
        return ChainPacket((0,) * nodes, (q,) * nodes, position)

    def propose_tree(self, *args, **kwargs):
        pytest.fail("sampled decoding used the old coupled tree")

    def add_taps(self, taps):
        self.taps.append(len(taps))


def run(decode, count, seed=6, on_tokens=None, constraint=None):
    draft = Draft()
    w = NS(config=NS(eos=(99,)), norm=torch.ones(1))
    got = [1]

    def emit(tokens):
        got.extend(tokens)
        return bool(on_tokens(tokens)) if on_tokens is not None else False

    result = decode.draft_decode(w, NS(pos=10), list(range(10)), 1, count, Sampling(seed, 1.0, 2, 1.0), draft,
                                 max_rows=12, allow_copy=False, stop_eos=False, on_tokens=emit, inplace=True,
                                 constraint=constraint)
    return got, result, draft


@pytest.mark.parametrize("seed", [1, 6, 1234, 1237])
def test_output_limit_preserves_the_same_rejection_prefix_and_canonical_q(dense, seed):
    expected, _, _ = run(dense, 17, seed)
    for limit in (2, 3, 4, 8, 12, 17):
        got, stats, draft = run(dense, limit, seed)
        assert got == expected[:limit] and len(got) == limit
        assert stats.drafted_rows >= 0 and draft.calls
        assert all(nodes == 3 for _, _, nodes in draft.calls)
        assert len(draft.taps) == len(draft.calls)


def test_yield_replays_the_original_request_without_resampling_another_logical_schedule(dense):
    expected, _, _ = run(dense, 17)

    class Once:
        replay = True

        def __init__(self):
            self.calls = 0

        def cut(self, tokens):
            self.calls += 1
            return (len(tokens), []) if self.calls == 3 else None

        def observe(self, token):
            pass

    starts, got = [], []

    def generate(prompt, count, emit):
        starts.append((list(prompt), count))
        if emit([1]):
            return {}
        run(dense, count, on_tokens=emit)
        return {}

    generate_gated(generate, list(range(10)), 17, [Once()], lambda tokens: got.extend(tokens) and False)
    assert got == expected
    assert len(starts) == 2 and starts[0] == starts[1] == (list(range(10)), 17)


def test_invalid_grammar_child_keeps_original_q_and_rejects_before_pruned_descendants(dense):
    class Constraint:
        def window(self, tokens, parents):
            return NS(tokens=tokens[:1], parents=[-1])

        def mask(self, logits, window):
            logits[:, 0] = -torch.inf

        def advance(self, tokens):
            assert all(t == 1 for t in tokens)

    got, _, draft = run(dense, 9, constraint=Constraint())
    assert got == [1] * 9 and len(draft.calls) == 8
    assert all(nodes == 3 for _, _, nodes in draft.calls)


def test_stop_callback_keeps_the_completed_rejection_prefix(dense):
    expected, _, _ = run(dense, 31)
    emitted = []

    def stop(tokens):
        emitted.extend(tokens)
        return len(emitted) >= 5

    got, _, draft = run(dense, 31, on_tokens=stop)
    assert got == expected[:len(got)] and 5 <= len(emitted) < 30 and draft.calls


@pytest.mark.parametrize("sampled,copied", [(False, False), (False, True), (True, False), (True, True)])
def test_real_multi_mode_selection_never_routes_sampled_producers_to_q_less_tree(allocations, sampled, copied):
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.max_rows, dec.draft = 16, NS(trained=8)
    stream = NS(draft=True, sid=3, sampling=Sampling(6) if sampled else None,
                copies=NS(propose=lambda context, nodes: [4, 5] if copied else []), context=[1, 2, 3])
    expected = multi.COPY_CHAIN if sampled and copied else multi.CHAIN if sampled else \
        multi.COPY if copied else multi.TREE
    assert dec._mode(stream, {}) == expected


def test_dflash_chain_q_uses_actual_predecessor_and_history_without_target_noise(allocations):
    module = importlib.import_module("tensorfold.families.qwen3_5.cuda.dflash2")
    d = module.DFlash2.__new__(module.DFlash2)
    pred = np.arange(32, dtype=np.float64)[:, None] / 8
    succ = np.zeros((32, 1), dtype=np.float64)
    succ[1] = 1
    d.weights = {"candidate_selector.predecessor_codebook": pred,
                 "candidate_selector.successor_codebook": succ}
    ids = np.tile(np.arange(16, dtype=np.int64), (3, 1))
    values = np.full((3, 17), -100.0)
    values[:, :2], values[:, 16] = [0.0, 1.0], 1.0

    class History:
        def __init__(self):
            self.paths = []

        def candidates(self, scores, tokens, path):
            self.paths.append(tuple(path))
            result = scores.copy()
            for token in path:
                result[tokens == token] -= .25
            return result

    history = History()
    packet = d.finish_chain(([None, None, (ids, values)], 0, 3, 2), 10, 3,
                            Sampling(6, 1.0, 2, 1.0), history)
    assert history.paths == [packet.tokens[:i] for i in range(3)]
    for i, q in enumerate(packet.q_rows):
        previous = 2 if i == 0 else packet.tokens[i - 1]
        z = np.array([-.25 * packet.tokens[:i].count(0),
                      1 + .6 * previous / 8 - .25 * packet.tokens[:i].count(1)])
        expected = np.exp(z - max(z)); expected /= expected.sum()
        actual = dict(zip(q.ids.tolist(), q.probs.tolist()))
        # The two-way FP64 softmax is followed by ProbabilityRow's second normalization: <=4 rounding ULPs.
        assert all(abs(actual[t] - expected[t]) <= 4 * np.spacing(expected[t]) for t in (0, 1))
        assert not q.ids.flags.writeable and not q.probs.flags.writeable
    with pytest.raises(ValueError, match="cannot be truncated"):
        d.finish_chain(([None, None, (ids, values)], 0, 3, 2), 10, 2, Sampling(6))


@pytest.mark.parametrize("tied", [False, True])
def test_compact_candidate_repair_matches_full_sort_and_retires_launch_owner_once(allocations, tied):
    module = importlib.import_module("tensorfold.families.qwen3_5.cuda.dflash2")
    d = module.DFlash2.__new__(module.DFlash2)
    d.weights = {"candidate_selector.predecessor_codebook": np.zeros((32, 1)),
                 "candidate_selector.successor_codebook": np.zeros((32, 1))}
    logits = torch.zeros((3, 32)) if tied else torch.arange(32, 0, -1).float().repeat(3, 1)
    values, ids, flags = target_sampling.candidate_fields(logits, 16)
    canonical, order = torch.sort(logits, dim=1, descending=True, stable=True)
    baseline = [order[:, :16], torch.cat((canonical[:, :16], torch.zeros((3, 1))), dim=1), None]
    repairs = []

    class Owner:
        def __init__(self, tensor):
            self.tensor = tensor

        def __getitem__(self, row):
            repairs.append(row)
            return self.tensor[row]

    owner = Owner(logits)
    ref = weakref.ref(owner)
    packet = [ids, torch.cat((values, torch.zeros((3, 1)), flags[:, None].float()), dim=1), None,
              {"logits": owner, "ids": np.arange(32, dtype=np.int64), "width": 32,
               "columns": values.shape[1], "canonical": False}]
    del owner, logits
    policy = Sampling(6)
    expected = d.finish_chain((baseline, 0, 3, 2), 10, 3, policy)
    got = d.finish_chain((packet, 0, 3, 2), 10, 3, policy)
    assert ref() is None and repairs == (list(range(3)) if tied else [])
    assert got.tokens == expected.tokens
    for a, b in zip(got.q_rows, expected.q_rows):
        assert a.ids.tobytes() == b.ids.tobytes() and a.probs.tobytes() == b.probs.tobytes()
    again = d.finish_chain((packet, 0, 3, 2), 10, 3, policy)
    assert again.tokens == got.tokens and repairs == (list(range(3)) if tied else [])


def test_logical_masked_feature_width_depends_only_on_config_and_declared_context_position(allocations):
    module = importlib.import_module("tensorfold.families.qwen3_5.cuda.dflash2")
    d = module.DFlash2.__new__(module.DFlash2)
    d.trained, d.max_position = 8, 100
    d._launch_blocks = lambda snaps, pendings, length, **kw: [(p, snap[3], length) for snap, p in zip(snaps, pendings)]
    snaps = [(None, None, 1, 20), (None, None, 1, 98), (None, None, 1, 99)]
    together = d.launch_chains(snaps, [7, 8, 9], 15)
    assert together == [(7, 20, 8), (8, 98, 2), None]
    assert together[0] == d.launch_chains(snaps[:1], [7], 127)[0]
    assert together[1] == d.launch_chains(snaps[1:2], [8], 15)[0]
    assert d.launch_blocks(snaps, [7, 8, 9], 127, block=16) == [(7, 20, 16), (8, 98, 2), None]


def test_sampled_multi_packet_survives_batch_deepest_and_greedy_allocation(dense, monkeypatch):
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    from tensorfold.cuda.streams import Stream
    from tests.test_qwen27_stream_depth import decoder

    outcomes = []
    for companion in (False, True):
        dec = decoder(multi, monkeypatch, [3, 3], [(1, 1.0), (32, 1.1)])
        dec.eos, dec.block = (99,), 15
        s = Stream([2, 3], 20, Sampling(6, 1.0, 2, 1.0), stop_eos=False)
        s.st, s.snap, s.out = NS(pos=10), "sampled", [1]
        dec.streams = {0: s}
        if companion:
            g = Stream([4, 5], 20, None, stop_eos=False)
            g.st, g.snap, g.out = NS(pos=10), "greedy", [1]
            dec.streams[1] = g
        launches = []

        def launch(snaps, pendings, nodes, *, events=launches):
            events.append((list(snaps), list(pendings), nodes))
            return list(snaps)

        def finish(raw, position, nodes, sampling, history=None):
            q = ProbabilityRow([0], [1.0])
            return ChainPacket((0, 0, 0), (q, q, q), position)

        dec.draft.launch_chains, dec.draft.finish_chain = launch, finish
        monkeypatch.setattr(multi, "allocate", lambda *args: [1])

        def forward(w, windows, **kwargs):
            starts = [0]
            for tokens, _, _ in windows:
                starts.append(starts[-1] + len(tokens))
            logits = torch.tensor([[np.log(.3), np.log(.7)]] * starts[-1], dtype=torch.float32)
            return logits, None, None, starts

        monkeypatch.setattr(multi, "multi_tree_forward", forward)
        plan = [(sid, multi.CHAIN if sid == 0 else multi.TREE, 1, 11) for sid in dec.streams]
        wins, _, _, _, results = dec._verify(plan, {})
        assert len(wins[0][0]) == 4 and launches == [(["sampled"], [1], 15)]
        if companion:
            assert len(wins[1][0]) == 2                  # greedy budget shrinks, stochastic q never does
        outcomes.append(results[0])
    assert outcomes[0] == outcomes[1]


def test_two_rank_split_head_agrees_on_rejection_when_only_rank_zero_has_a_companion(dense, monkeypatch):
    tp = importlib.import_module("tensorfold.families.qwen3_5.cuda.decode_tp")
    policy = importlib.import_module("tensorfold.families.qwen3_5.cuda.draft_tree")
    broadcasts, sent, extraction = queue.Queue(), [], []
    monkeypatch.setattr(tp.dist, "barrier", lambda: None)
    monkeypatch.setattr(tp, "split_sampling_plan", lambda width, device, rank: NS(rank=rank))
    monkeypatch.setattr(tp, "_sample_split", lambda *a, **k: pytest.fail("a sampled rank chose coupled collectives"))

    def share(values, rank, device):
        if rank == 0:
            value = list(values)
            sent.append(value)
            broadcasts.put(value)
            return value
        return broadcasts.get(timeout=5)

    monkeypatch.setattr(tp, "_share", share)

    def probabilities(logits, sampling, rows, *, plan=None):
        extraction.append((plan.rank, logits.shape[0]))
        p = ProbabilityRow([100, 101], [.3, .7])
        return [p] * logits.shape[0] + [None] * (rows - logits.shape[0])

    monkeypatch.setattr(policy, "chain_probabilities", probabilities)

    def forward(w, tokens, parents, state, **kwargs):
        assert kwargs["full_logits"] and kwargs["tp"]
        logits = torch.zeros((len(parents), 101))
        return (logits, None, torch.zeros((len(parents), 1))) if kwargs["capture_taps"] else (logits, None)

    monkeypatch.setattr(tp, "tree_forward", forward)
    monkeypatch.setattr(tp, "commit", lambda state, record, path: setattr(state, "pos", state.pos + len(path)))

    class Companion(Draft):
        def propose_chain(self, pending, position, nodes, sampling, history=None):
            self.calls.append((pending, position, nodes))
            q = ProbabilityRow([100], [1.0])
            return ChainPacket((100,) * nodes, (q,) * nodes, position)

    states, results, errors, companion = [NS(pos=10), NS(pos=10)], {}, [], Companion()

    def worker(rank):
        try:
            w = NS(norm=torch.ones(1), head=NS(n=101), config=NS(vocab=202, eos=(99,)))
            results[rank] = tp.decode_tp(w, states[rank], list(range(10)), 100 if rank == 0 else -1, 11,
                                         Sampling(6, 1.0, 2, 1.0), rank,
                                         companion if rank == 0 else None, max_rows=12, stop_eos=False,
                                         inplace=True)
        except BaseException as error:  # noqa: BLE001 - surface every worker failure after both thread owners join
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(rank,)) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors, errors
    assert sent[0] == [1] and broadcasts.empty() and companion.calls
    assert len(results[0].tokens) == 11 and set(results[0].tokens) <= {100, 101}
    assert states[0].pos == states[1].pos
    assert sorted(n for rank, n in extraction if rank == 0) == sorted(n for rank, n in extraction if rank == 1)
