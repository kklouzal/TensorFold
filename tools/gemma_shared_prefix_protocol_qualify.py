"""Private-arm binding of the preserved public Gemma protocol regression.

Original: run_live_smoke.py SHA d8dfe5218efc51bd4898cb89059ceadccff14ad23b7d33b9beada5b7841a189f.
Channel/content/tool/numerical/equality oracles are retained. Only the obsolete
entry-count/LRU replacement assumptions become byte-funded archive checks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json

if __package__:
    from . import gemma_shared_prefix_qualify as core
else:
    import gemma_shared_prefix_qualify as core

NAMES = ("markdown-nonstream", "strict-arithmetic-stream", "literal-tool-stream", "clock-call-stream",
         "clock-followup-stream", "json-nonstream", "cache-A-first", "cache-A-repeat-stream",
         "cache-A-extended", "cache-B-intervening", "cache-A-after-B",
         *("four-cache-fill-" + name for name in "ABCD"),
         *("four-cache-repeat-" + name for name in "ABCD"),
         "four-cache-touch-A", "four-cache-insert-E", "four-cache-preserved-A", "four-cache-evicted-B")


def load_requests(spec):
    requests = {}
    for name in NAMES:
        ref = spec["requests"][name]
        data = core.read_bound(ref["path"], ref["bytes"])
        core.require(len(data) == ref["bytes"] and hashlib.sha256(data).hexdigest() == ref["sha256"],
                     "frozen protocol request identity differs")
        body = json.loads(data)
        core.require(type(body) is dict and body.get("model") == spec["model"], "explicit model/body required")
        requests[name] = body
    return requests


def fields(result):
    message = result["message"]
    return message.get("content") or "", message.get("reasoning_content") or "", message.get("tool_calls") or []


def equal_visible(first, second):
    core.require(fields(first)[:2] == fields(second)[:2], "cold/hit visible content or reasoning differs")


def full_hit(result):
    return result["cached"] == result["prompt"] and result["prompt"] > 0


def run(client, requests):
    def request(name, body=None):
        return client.generate(name, "/v1/chat/completions",
                               copy.deepcopy(requests[name] if body is None else body))
    result = request("markdown-nonstream")
    content, _, _ = fields(result)
    core.require(content.strip() and "`" in content, "inline-code character gold failed")
    result = request("strict-arithmetic-stream")
    content, _, calls = fields(result)
    core.require(content.strip() or calls, "strict arithmetic reply is empty")
    result = request("literal-tool-stream")
    content, _, calls = fields(result)
    literal = "<|channel>thought <|channel>synthesis <channel|>"
    core.require(len(calls) == 1 and not content.strip(), "literal tool call/content contract failed")
    core.require(json.loads(calls[0]["function"]["arguments"]) == {"text": literal}, "quoted tool literal changed")
    called = request("clock-call-stream")
    _, _, calls = fields(called)
    core.require(len(calls) == 1 and json.loads(calls[0]["function"]["arguments"])["zone"] == "UTC",
                 "clock call gold failed")
    follow = copy.deepcopy(requests["clock-followup-stream"])
    tool = {**follow["messages"][-1], "tool_call_id": calls[0]["id"]}
    follow["messages"] = requests["clock-call-stream"]["messages"] + [core.assistant(called), tool]
    result = request("clock-followup-stream", follow)
    core.require("12:34" in fields(result)[0] and fields(result)[0].strip(), "actual clock-result history gold failed")
    core.require(json.loads(fields(request("json-nonstream"))[0]) == {"ok": True}, "JSON object gold failed")

    first = request("cache-A-first")
    second = request("cache-A-repeat-stream")
    core.require(full_hit(second), "immediate exact prefix repeat did not fully reuse")
    equal_visible(first, second)
    extension = copy.deepcopy(requests["cache-A-extended"])
    extension["messages"] = requests["cache-A-first"]["messages"] + [core.assistant(first), extension["messages"][-1]]
    extended = request("cache-A-extended", extension)
    core.require(extended["cached"] > 0, "growing history did not reuse any prefix")
    request("cache-B-intervening")
    after = request("cache-A-after-B")
    core.require(full_hit(after), "A did not survive one intervening B")
    equal_visible(first, after)

    saved = {}
    for name in "ABCD":
        saved[name] = request("four-cache-fill-" + name)
    for name in "ABCD":
        hit = request("four-cache-repeat-" + name)
        core.require(full_hit(hit), "four existing conversations did not fully reuse")
        equal_visible(saved[name], hit)
    request("four-cache-touch-A")
    saved["E"] = request("four-cache-insert-E")
    recent = request("four-cache-preserved-A")
    core.require(full_hit(recent), "recent A did not survive E")
    equal_visible(saved["A"], recent)
    b = request("four-cache-evicted-B")
    equal_visible(saved["B"], b)
    client.feature(full_hit(b), "byte-funded B branch was lost after E")
    # The replacement byte contract covers all five branches, rather than
    # manufacturing a new entry-count ceiling. Baseline misses remain recorded.
    for name in "CDE":
        body = requests["four-cache-insert-E"] if name == "E" else requests["four-cache-fill-" + name]
        hit = client.generate("five-cache-final-" + name, "/v1/chat/completions", copy.deepcopy(body))
        equal_visible(saved[name], hit)
        client.feature(full_hit(hit), "byte-funded five-conversation retention failed")
    final = client.stats("protocol-final")
    return {"passed": True, "original_protocol_requests": 23, "additional_byte_retention_requests": 3,
            "B_cached": b["cached"], "B_prompt": b["prompt"], "prefix_cache": final["prefix_cache"],
            "P3_acceptance": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    spec, _, _ = core.load_inputs(args.inputs, args.inputs_sha256)
    requests = load_requests(spec)  # Entire body authority checked before HTTP.
    evidence = core.Evidence(args.out)
    client = core.Client(args.base_url, evidence, args.arm)
    result, primary, errors = {"passed": False}, None, [None, None]
    try:
        client.stats("protocol-initial")
        result = run(client, requests)
    except BaseException as error:
        primary = error
        try:
            result["failure"] = {"type": type(error).__name__, "message": str(error)[:1024]}
        except BaseException as error:
            errors[0] = error
    try:
        evidence.save("protocol-result.json", {**result, "requests": client.results})
    except BaseException as error:
        errors[1] = error
    core.finish_errors(primary, errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
