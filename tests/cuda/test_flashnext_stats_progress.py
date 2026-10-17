"""Flash Next's concurrent decoder reports each prefill pass to a request that asked for progress, and its /stats
memory and prefix-cache readings, without changing any reply's tokens."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import FIRST, MultiDecoder  # noqa: E402

PROMPTS = ((700, None), (900, Sampling(seed=9, top_k=20, top_p=0.95)))


@pytest.fixture(scope="module")
def w():
    return _model()


def _prompt(n: int, seed: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (n,), generator=gen).tolist()


def _run(w, report: bool):
    dec = MultiDecoder(w, slots=2, capacity=4096, depth=3, confidence=0.3, prefill_rows=256)
    outs, rows, usages = [[], []], [[], []], []
    for i, (n, sampling) in enumerate(PROMPTS):
        s = Stream(_prompt(n, i), 24, sampling, emit=lambda new, got=outs[i]: got.extend(new))
        if report:
            s.on_fill = rows[i].append
        dec.admit(s)
    while dec.live():
        usages.append((dec.usage(), [s.st for s in [*dec.streams.values(), *dec.filling]]))
        dec.finish(dec.round())
    usages.append((dec.usage(), []))
    torch.cuda.synchronize()
    return outs, rows, usages, dec


def test_prefill_progress_reports_every_pass_and_changes_no_token(w):
    plain = _run(w, False)[0]
    outs, rows, usages, dec = _run(w, True)
    assert outs == plain and all(len(o) == 24 for o in outs)
    for reported, (n, _) in zip(rows, PROMPTS):
        assert len(reported) >= 3 and reported == sorted(reported) and reported[-1] == n


def test_usage_reads_the_gate_the_live_slots_and_the_kept_points(w):
    _, _, usages, dec = _run(w, False)
    gate = dec.memory_gate
    for usage, live in usages:
        kv, prefix = usage["kv"], usage["prefix_cache"]
        # docs/api.md's /stats fields (tests/test_server_stats.py's Decoder reports the same keys)
        assert set(usage) == {"kv", "prefix_cache", "memory_waits", "memory_stops"}
        assert set(kv) == {"budget_bytes", "held_bytes", "active_bytes", "usage_ratio", "bytes_per_token",
                           "free_tokens_estimate"}
        assert set(prefix) == {"entries", "capacity", "slots", "tokens"}
        assert kv["budget_bytes"] == gate.room - gate.reserve
        first = min(dec.capacity, FIRST)
        assert kv["active_bytes"] == sum(st.cache_bytes() - st.cache_bytes(first) for st in live)
        assert kv["usage_ratio"] == round(kv["active_bytes"] / kv["budget_bytes"], 6)
        assert kv["bytes_per_token"] > 0 and kv["free_tokens_estimate"] >= 0
        assert prefix["capacity"] == dec.keep and 0 <= prefix["entries"] <= dec.keep
        assert prefix["slots"] <= prefix["entries"] and prefix["tokens"] >= 0
    final = usages[-1][0]
    assert final["kv"]["active_bytes"] == 0 and final["kv"]["held_bytes"] == gate.held
    assert final["prefix_cache"]["entries"] == len(dec.kept)
    assert (final["memory_waits"], final["memory_stops"]) == (gate.waits, gate.ends)
