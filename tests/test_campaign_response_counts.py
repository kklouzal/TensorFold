"""Output-only peer counts expose equal partial replies without changing gate laws or requests."""
import copy
from types import SimpleNamespace

import pytest
from test_campaign_tools import PROFILE, PROMPTS, gates


@pytest.mark.parametrize("short_peer", [None, "serial", "extensions"])
def test_real_gates_retain_each_typed_response_count_without_new_requests(monkeypatch, short_peer):
    seen, evictions = [], []
    previous = []
    item = copy.deepcopy(PROMPTS["items"][0])
    monkeypatch.setattr(gates, "_items", lambda *_a: ([item], []))
    monkeypatch.setattr(gates, "_prompt_ids", lambda *_a: [7, 8, 9])

    def evict(*_a):
        evictions.append(True)
        previous.clear()

    def provider(_url, body, **kwargs):
        seen.append((copy.deepcopy(body), kwargs))
        prompt = body.get("prompt", [7, 8, 9])
        cached = min(len(previous), len(prompt)) if previous[:3] == prompt[:3] else 0
        previous[:] = prompt
        short = ((short_peer == "serial" and body.get("draft") is False and body["temperature"] > 0)
                 or (short_peer == "extensions" and "prompt" in body))
        count = body["max_tokens"] - int(short)
        return {"content": "reply", "reasoning": "", "usage": {"completion_tokens": count},
                "stats": {"token_ids": list(range(count)), "token_sha": str(count), "cached": cached,
                          "rounds": 2, "drafted": 8, "accepted": 0}}

    monkeypatch.setattr(gates, "_evict", evict)
    monkeypatch.setattr(gates, "stream_chat", provider)
    args = SimpleNamespace(max_tokens=512, seed=2024, thinking="off", allow_long_prefill=True,
                           allow_greedy_identity_failure=False, allow_nondeterminism=False, evict_requests=6)
    profile = {**PROFILE, "context": 4096}
    identity = gates.gate_identity("http://provider", profile, args)
    assert identity["status"] == "pass" and identity["greedy_identical"] == 1
    assert len(seen) == 4 and not evictions
    assert [body.get("draft") for body, _ in seen] == [None, False, None, False]
    assert [body.get("seed") for body, _ in seen] == [None, None, 2024, 2024]
    for row in identity["rows"]:
        serial = 511 if short_peer == "serial" and row["sampling"] == "card" else 512
        assert row["response_tokens"] == {"drafted": 512, "serial": serial}
        assert row["tokens"] == 512
    seen.clear()
    determinism = gates.gate_determinism("http://provider", profile, args)
    assert determinism["status"] == "pass" and determinism["identical"] == 2
    assert determinism["exercised"] == {"warm": 2, "recold": 2, "extended": 2}
    assert len(seen) == 16 and len(evictions) == 8  # original 8 requests/4 evictions per sampling mode
    assert all(body["max_tokens"] == 512 for body, _ in seen)
    assert sum("prompt" in body for body, _ in seen) == 4
    for row in determinism["rows"]:
        extension = 511 if short_peer == "extensions" else 512
        assert row["response_tokens"] == {"cold": 512, "warm": 512, "recold": 512,
                                          "extended": extension, "extended_cold": extension}
        assert row["tokens"] == 512  # the representative field alone cannot prove peer completeness
