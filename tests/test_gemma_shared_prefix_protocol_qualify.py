"""Historical protocol gold is independent of archive entry-count policy."""
import copy

import pytest

from tools import gemma_shared_prefix_protocol_qualify as q


def fixture(arm="candidate", bad_literal=False, lose_b=False):
    requests = {name: {"model": "fixture", "messages": [{"role": "user", "content": name}]}
                for name in q.NAMES}
    requests["clock-followup-stream"]["messages"].append(
        {"role": "tool", "tool_call_id": "old-frozen-id", "content": "The UTC time is 12:34."})
    class Client:
        def __init__(self):
            self.arm, self.called = arm, []
        def feature(self, passed, message):
            if self.arm == "candidate":
                q.core.require(passed, message)
            return bool(passed)
        def stats(self, name):
            return {"prefix_cache": {"charged_bytes": 20, "byte_budget": 100,
                                    "entries": None if arm == "candidate" else 4}}
        def generate(self, name, path, body):
            self.called.append((name, copy.deepcopy(body)))
            calls, content = [], "reply"
            if name == "markdown-nonstream":
                content = "` starts inline code."
            elif name == "strict-arithmetic-stream":
                content = "4"
            elif name == "literal-tool-stream":
                literal = "bad" if bad_literal else "<|channel>thought <|channel>synthesis <channel|>"
                calls, content = [{"id": "literal", "function": {"arguments": q.json.dumps({"text": literal})}}], ""
            elif name == "clock-call-stream":
                calls, content = [{"id": "actual-current-call-id", "type": "function",
                                  "function": {"name": "get_time", "arguments": '{"zone":"UTC"}'}}], ""
            elif name == "clock-followup-stream":
                assert body["messages"][-1]["tool_call_id"] == "actual-current-call-id"
                assert body["messages"][-2]["tool_calls"][0]["id"] == "actual-current-call-id"
                content = "It is 12:34."
            elif name == "json-nonstream":
                content = '{"ok":true}'
            elif name.startswith("cache-A") and "extended" not in name:
                content = "CACHE_A"
            elif "extended" in name:
                content = "CACHE_NEXT"
            elif name.startswith("four-cache-") or name.startswith("five-cache-"):
                content = "CACHE_" + ("E" if name == "four-cache-insert-E" else name[-1])
            cached = 0 if name == "four-cache-evicted-B" and (arm == "baseline" or lose_b) else 100
            return {"message": {"role": "assistant", "content": content, "reasoning_content": "", "tool_calls": calls},
                    "prompt": 100, "cached": cached}
    return Client(), requests


def test_all_historical_quality_checks_and_five_byte_funded_branches():
    client, requests = fixture()
    original = copy.deepcopy(requests)
    result = q.run(client, requests)
    assert result["passed"] and len(client.called) == 26
    assert result["original_protocol_requests"] == 23
    assert requests == original
    assert [name for name, _ in client.called][-3:] == ["five-cache-final-C", "five-cache-final-D", "five-cache-final-E"]


def test_baseline_old_four_entry_eviction_is_observed_not_a_candidate_waiver():
    baseline, requests = fixture("baseline")
    assert q.run(baseline, requests)["B_cached"] == 0
    candidate, requests = fixture(lose_b=True)
    with pytest.raises(ValueError, match="B branch"):
        q.run(candidate, requests)


@pytest.mark.parametrize("arm", ["baseline", "candidate"])
def test_quoted_channel_tool_payload_gold_remains_required(arm):
    client, requests = fixture(arm, bad_literal=True)
    with pytest.raises(ValueError, match="quoted tool literal"):
        q.run(client, requests)
    assert len(client.called) == 3


def test_visible_output_equality_oracle_is_not_replaced_by_hit_counts():
    with pytest.raises(ValueError, match="content or reasoning"):
        q.equal_visible({"message": {"content": "gold", "reasoning_content": ""}},
                        {"message": {"content": "changed", "reasoning_content": ""}})


def test_protocol_module_import_is_inert(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network at import")
    monkeypatch.setattr(q.core.http.client, "HTTPConnection", forbidden)
    assert callable(q.main) and len(q.NAMES) == 23
