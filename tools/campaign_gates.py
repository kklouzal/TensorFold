"""Campaign quality gates over a TensorFold OpenAI server (the production HTTP path). Standard library only.

  tools        strict and non-strict tool-call suite (OpenAI tools; the server parses each model's own call format)
  needle       multi-key needle-in-a-haystack recall at several depths up to the full advertised context
  determinism  same request twice -> same tokens; prefix-cache warm vs cold (and after eviction) -> same tokens
  identity     drafted vs serial ("draft": false): greedy must be token-identical; seeded sampling reported
  all          every gate above in one session

Every gate needs --url, --profile (the bench profile: model, card sampling per thinking mode, reply budget, end ids)
and --out. needle sizes its haystacks from --corpus with the server's /v1/tokenize; determinism and identity reuse
prompts from a bench prompt set (--prompts). Thresholds are parameters with no built-in values except the contract
gates (determinism and greedy drafted == serial must be exact unless --allow-* says otherwise); a gate without a
threshold reports its value and a null verdict. The exit status is 1 when any gate with a threshold fails.
A determinism or identity verdict covers only the paths a request exercised: determinism counts the cases whose warm
request actually resumed cached tokens (warm_exercised, extended_exercised) and identity the greedy cases that drafted
at all (greedy_drafted); token ids are required from every reply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]}); "
             "TensorFold's tools use PEP 604 unions. Try: python3.12 tools/campaign_gates.py ...")

if __package__:
    from .benchmark_output import write_json
    from .campaign_bench import (chat_request, corpus_text, file_sha256, first_difference, get_json, post_json,
                                 reply_token_ids, stream_chat)
else:
    from benchmark_output import write_json
    from campaign_bench import (chat_request, corpus_text, file_sha256, first_difference, get_json, post_json,
                                reply_token_ids, stream_chat)

# ---------------------------------------------------------------------------------------------------------------
# tool-call suite: strict-compatible schemas (every property required, no additional properties)


def _fn(name: str, description: str, properties: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}}}


WEATHER = _fn("get_weather", "Current weather for a city.",
              {"city": {"type": "string", "description": "City name, e.g. Paris"},
               "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}})
CALCULATE = _fn("calculate", "Evaluate an arithmetic expression exactly.",
                {"expression": {"type": "string", "description": "e.g. (2 + 3) * 4"}})
EVENT = _fn("create_event", "Create a calendar event.",
            {"title": {"type": "string"}, "start": {"type": "string", "description": "ISO 8601 local date-time"},
             "duration_minutes": {"type": "integer"},
             "attendees": {"type": "array", "items": {
                 "type": "object", "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                 "required": ["name", "email"], "additionalProperties": False}}})
SEARCH = _fn("search_files", "Search the repository for files.",
             {"query": {"type": "string"}, "max_results": {"type": "integer"},
              "extensions": {"type": "array", "items": {"type": "string"}}})
THERMOSTAT = _fn("set_thermostat", "Set a room thermostat.",
                 {"room": {"type": "string"}, "temperature": {"type": "number"},
                  "mode": {"type": "string", "enum": ["heat", "cool", "auto"]}})


WRITE_FILE = _fn("write_file", "Write a text file.", {"path": {"type": "string"}, "content": {"type": "string"}})
SEND_EMAIL = _fn("send_email", "Send an email.", {"to": {"type": "string"}, "subject": {"type": "string"},
                                                  "body": {"type": "string"}})
FLIGHT = _fn("book_flight", "Book a one-way flight.", {"origin": {"type": "string"}, "destination": {"type": "string"},
                                                       "date": {"type": "string", "description": "YYYY-MM-DD"}})
CONVERT = _fn("convert_currency", "Convert money between currencies.",
              {"amount": {"type": "number"}, "from_currency": {"type": "string", "description": "ISO 4217 code"},
               "to_currency": {"type": "string", "description": "ISO 4217 code"}})
DISTRACTORS = [_fn(name, description, {arg: {"type": "string"}}) for name, description, arg in (
    ("get_stock_price", "Latest price of a stock.", "ticker"), ("translate", "Translate text to English.", "text"),
    ("get_time", "Current time in a time zone.", "timezone"), ("define_word", "Dictionary definition.", "word"),
    ("lookup_package", "Track a parcel.", "tracking_number"), ("get_news", "Headlines on a topic.", "topic"),
    ("play_song", "Play a song.", "title"), ("find_recipe", "Find a recipe.", "dish"))]


def _directory(seed: int = 11, count: int = 300) -> tuple[str, str, str]:
    """A deterministic employee directory (about 6k tokens) and one entry's name and email to retrieve."""

    rng = random.Random(seed)
    first = ("Ada", "Bram", "Chen", "Dalia", "Emre", "Farah", "Goran", "Hana", "Ivo", "Jun", "Kira", "Lars", "Mei",
             "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara", "Ugo", "Vera", "Wen", "Ximena", "Yusuf", "Zoe")
    last = ("Abara", "Bianchi", "Castro", "Dubois", "Eriksen", "Fujita", "Garcia", "Hoffman", "Ivanova", "Jensen",
            "Kowalski", "Lindqvist", "Moreau", "Nakamura", "Okafor", "Petrov", "Quispe", "Rossi", "Silva", "Tanaka")
    rows, seen = [], set()
    while len(rows) < count:
        name = f"{rng.choice(first)} {rng.choice(last)}"
        if name in seen:
            continue
        seen.add(name)
        email = f"{name.split()[0].lower()}.{name.split()[1].lower()}{rng.randrange(100)}@example.org"
        rows.append((name, email, rng.choice(("Finance", "Research", "Support", "Legal", "Platform")),
                     rng.randrange(1000, 9999)))
    target = rows[len(rows) * 2 // 3]
    table = "\n".join(f"- {n} | {e} | {d} | extension {x}" for n, e, d, x in rows)
    return table, target[0], target[1]


DIRECTORY, DIRECTORY_NAME, DIRECTORY_EMAIL = _directory()


def _user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def _has(value, *needles) -> bool:
    text = json.dumps(value).lower()
    return all(n.lower() in text for n in needles)


# name -> (messages, tools, extra request fields, expectation(calls, content) -> bool)
TOOL_CASES = {
    "weather": (_user("What's the weather in Paris right now? Use celsius."), [WEATHER, CALCULATE], {},
                lambda calls, content: len(calls) >= 1 and calls[0][0] == "get_weather"
                and _has(calls[0][1].get("city"), "paris") and calls[0][1].get("unit") == "celsius"),
    "weather-fahrenheit": (_user("Tell me the current temperature in Austin, Texas, in Fahrenheit."), [WEATHER], {},
                           lambda calls, content: len(calls) >= 1 and calls[0][0] == "get_weather"
                           and _has(calls[0][1].get("city"), "austin") and calls[0][1].get("unit") == "fahrenheit"),
    "calculate": (_user("Use the calculator to compute 1234 * 5678 + 91."), [CALCULATE, WEATHER], {},
                  lambda calls, content: len(calls) >= 1 and calls[0][0] == "calculate"
                  and _has(calls[0][1].get("expression"), "1234", "5678", "91")),
    "nested-event": (_user("Schedule a 45-minute meeting titled 'Design review' on 2026-11-03 at 14:00 with Alice "
                           "(alice@example.com) and Bob (bob@example.com)."), [EVENT], {},
                     lambda calls, content: len(calls) >= 1 and calls[0][0] == "create_event"
                     and calls[0][1].get("duration_minutes") == 45 and len(calls[0][1].get("attendees") or []) == 2
                     and _has(calls[0][1].get("attendees"), "alice@example.com", "bob@example.com")
                     and _has(calls[0][1].get("start"), "2026-11-03")),
    "choose-tool": (_user("Find Python files about tokenizers in the repository; at most 5 results."),
                    [WEATHER, CALCULATE, SEARCH], {},
                    lambda calls, content: len(calls) >= 1 and calls[0][0] == "search_files"
                    and _has(calls[0][1].get("query"), "tokeniz") and calls[0][1].get("max_results") == 5),
    "enum-number": (_user("Set the living room thermostat to 21.5 degrees and keep it cooling."), [THERMOSTAT], {},
                    lambda calls, content: len(calls) >= 1 and calls[0][0] == "set_thermostat"
                    and _has(calls[0][1].get("room"), "living") and calls[0][1].get("temperature") == 21.5
                    and calls[0][1].get("mode") == "cool"),
    "no-call": (_user("Write a haiku about autumn leaves."), [WEATHER], {},
                lambda calls, content: not calls and len(content.strip()) > 0),
    "required": (_user("Hi there!"), [WEATHER, CALCULATE], {"tool_choice": "required"},
                 lambda calls, content: len(calls) >= 1),
    "named": (_user("What is two plus two?"), [WEATHER, CALCULATE],
              {"tool_choice": {"type": "function", "function": {"name": "calculate"}}},
              lambda calls, content: len(calls) >= 1 and calls[0][0] == "calculate"),
    "after-result": ([{"role": "user", "content": "What's the weather in Paris? Use celsius."},
                      {"role": "assistant", "content": "", "tool_calls": [
                          {"id": "call_1", "type": "function", "function": {
                              "name": "get_weather", "arguments": json.dumps({"city": "Paris", "unit": "celsius"})}}]},
                      {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(
                          {"temperature": 18, "condition": "cloudy"})}], [WEATHER], {},
                     lambda calls, content: not calls and "18" in content),
    "parallel": (_user("What's the weather in Paris and in Tokyo? Use celsius for both."), [WEATHER], {},
                 lambda calls, content: {c[0] for c in calls} == {"get_weather"}
                 and _has([c[1] for c in calls], "paris", "tokyo")),
    "unicode": (_user("What's the weather in São Paulo and Zürich? Use celsius."), [WEATHER], {},
                lambda calls, content: len(calls) >= 1 and all(c[0] == "get_weather" for c in calls)
                and any("São Paulo" in str(c[1].get("city")) or "Zürich" in str(c[1].get("city")) for c in calls)),
    "escaped-content": (_user('Create the file hello.py containing exactly these three lines:\n'
                              'print("Hello, \\"world\\"")\nfor i in range(3):\n    print(i, end="\\t")'),
                        [WRITE_FILE], {},
                        lambda calls, content: len(calls) >= 1 and calls[0][0] == "write_file"
                        and calls[0][1].get("path", "").endswith("hello.py")
                        and "range(3)" in calls[0][1].get("content", "") and "\n" in calls[0][1].get("content", "")
                        and 'print("Hello' in calls[0][1].get("content", "")),
    "many-tools": (_user("Convert 250 euros to Japanese yen."), [*DISTRACTORS[:4], CONVERT, *DISTRACTORS[4:], FLIGHT],
                   {}, lambda calls, content: len(calls) >= 1 and calls[0][0] == "convert_currency"
                   and calls[0][1].get("amount") == 250 and _has(calls[0][1].get("from_currency"), "eur")
                   and _has(calls[0][1].get("to_currency"), "jpy")),
    "dates": (_user("Book me a one-way flight from Boston to Denver next Friday, which is 2026-11-06."),
              [FLIGHT, WEATHER], {},
              lambda calls, content: len(calls) >= 1 and calls[0][0] == "book_flight"
              and _has(calls[0][1].get("origin"), "boston") and _has(calls[0][1].get("destination"), "denver")
              and "2026-11-06" in str(calls[0][1].get("date"))),
    "long-context": (_user(f"Employee directory (name | email | department | extension):\n{DIRECTORY}\n\nSend "
                           f"{DIRECTORY_NAME} an email with the subject 'Welcome' and a one-sentence welcome body."),
                     [SEND_EMAIL, WEATHER, CALCULATE], {},
                     lambda calls, content: len(calls) >= 1 and calls[0][0] == "send_email"
                     and calls[0][1].get("to") == DIRECTORY_EMAIL and _has(calls[0][1].get("subject"), "welcome")),
}


def schema_errors(value, schema: dict, at: str = "$") -> list[str]:
    """Violations of the suite's JSON-schema subset (type, enum, properties, required, additionalProperties false,
    items); an independent check of the server's strict decoding and of non-strict replies."""

    kind = schema.get("type")
    checks = {"string": lambda v: isinstance(v, str), "integer": lambda v: type(v) is int,
              "number": lambda v: type(v) in (int, float), "boolean": lambda v: type(v) is bool,
              "object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
              "null": lambda v: v is None}
    if kind is not None and not checks[kind](value):
        return [f"{at}: expected {kind}, got {type(value).__name__}"]
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{at}: {value!r} not in {schema['enum']}")
    if kind == "object":
        props = schema.get("properties", {})
        errors += [f"{at}: missing {k}" for k in schema.get("required", ()) if k not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{at}: unexpected {k}" for k in value if k not in props]
        for key, sub in props.items():
            if key in value:
                errors += schema_errors(value[key], sub, f"{at}.{key}")
    if kind == "array" and "items" in schema:
        for i, item in enumerate(value):
            errors += schema_errors(item, schema["items"], f"{at}[{i}]")
    return errors


def tool_case(url: str, profile: dict, name: str, strict: bool, sampling: dict, thinking: bool) -> dict:
    messages, tools, extra, expect = TOOL_CASES[name]
    tools = [{**t, "function": {**t["function"], "strict": True}} if strict else t for t in tools]
    body = {"model": profile["model"], "messages": messages, "tools": tools,
            "max_tokens": int(profile.get("gate_tool_tokens", 4096 if thinking else 1024)),
            "chat_template_kwargs": {"enable_thinking": thinking}, "return_token_ids": True, **sampling, **extra}
    started = time.perf_counter()
    reply = post_json(url + "/v1/chat/completions", body, timeout=3600)
    choice = reply["choices"][0]
    message = choice.get("message") or {}
    calls, problems = [], []
    by_name = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError as error:
            problems.append(f"{fn.get('name')}: arguments are not JSON ({error.msg})")
            continue
        if fn.get("name") not in by_name:
            problems.append(f"unknown function {fn.get('name')!r}")
            continue
        problems += [f"{fn['name']}: {e}" for e in schema_errors(arguments, by_name[fn["name"]])]
        calls.append((fn["name"], arguments))
    content = message.get("content") or ""
    ok = not problems and bool(expect(calls, content))
    if calls and choice.get("finish_reason") != "tool_calls":
        problems.append(f"finish_reason {choice.get('finish_reason')!r} with calls")
        ok = False
    return {"case": name, "strict": strict, "thinking": thinking, "pass": ok, "problems": problems,
            "calls": calls, "content": content[:400], "finish": choice.get("finish_reason"),
            "seconds": round(time.perf_counter() - started, 2),
            "token_sha": (reply.get("tensorfold") or {}).get("token_sha")}


def sampling_modes(profile: dict, thinking: bool, seed: int) -> dict:
    card = profile["card_sampling"]["think" if thinking else "nothink"]
    return {"greedy": {"temperature": 0}, "card": {**card, "seed": seed}}


def _thinking(profile: dict, args) -> list[bool]:
    modes = [bool(t) for t in profile["thinking"]]
    return modes if args.thinking == "both" else [t for t in modes if t == (args.thinking == "on")]


def gate_tools(url: str, profile: dict, args) -> dict:
    rows = []
    for thinking in _thinking(profile, args):
        for mode, sampling in sampling_modes(profile, thinking, args.seed).items():
            for strict in (True, False):
                for name in TOOL_CASES:
                    row = tool_case(url, profile, name, strict, sampling, thinking)
                    row["sampling"] = mode
                    rows.append(row)
                    print(json.dumps({k: row[k] for k in ("case", "strict", "thinking", "sampling", "pass",
                                                          "problems")}), flush=True)
    groups = {}
    for row in rows:
        key = f"{'think' if row['thinking'] else 'nothink'}/{row['sampling']}/{'strict' if row['strict'] else 'loose'}"
        groups.setdefault(key, []).append(row["pass"])
    rates = {k: sum(v) / len(v) for k, v in groups.items()}
    overall = sum(r["pass"] for r in rows) / len(rows)
    verdict = None if args.min_tool_pass is None else all(v >= args.min_tool_pass for v in rates.values())
    return {"gate": "tools", "pass_rate": overall, "pass_rate_by_group": rates, "threshold": args.min_tool_pass,
            "verdict": verdict, "rows": rows}


# ---------------------------------------------------------------------------------------------------------------
# needle-in-a-haystack (multi-key: one prefill per depth)

VAULTS = ("AMBER", "BIRCH", "CEDAR", "DELTA", "EMBER", "FJORD", "GROVE", "HARBOR")


def needle_prompt(text: str, chars: int, fractions: list[float], codes: dict[str, str]) -> list[dict]:
    hay = text[:chars]
    pieces, last = [], 0
    for fraction, (vault, code) in zip(fractions, codes.items()):
        at = hay.rfind("\n", 0, max(1, int(len(hay) * fraction))) + 1
        pieces += [hay[last:at], f"# NOTE: the access code for vault {vault} is {code}.\n"]
        last = at
    pieces.append(hay[last:])
    names = ", ".join(codes)
    ask = (f"The document above contains notes giving the access codes of the vaults {names}. Reply with a JSON "
           f"object mapping each vault name to its access code, and nothing else.")
    return [{"role": "user", "content": "<document>\n" + "".join(pieces) + "</document>\n\n" + ask}]


def gate_needle(url: str, profile: dict, args) -> dict:
    text, digest = corpus_text(args.corpus)
    rng = random.Random(args.seed)
    fractions = [float(f) for f in args.needle_fractions.split(",")]
    rows = []
    for depth in [d for d in args.needle_depths.split(",") if d]:
        codes = {VAULTS[i]: f"{rng.randrange(10 ** 6):06d}" for i in range(len(fractions))}
        for thinking in ([False, True] if args.needle_thinking == "both" else [args.needle_thinking == "on"]):
            kwargs = {"enable_thinking": thinking}
            budget = int(args.needle_tokens) * (8 if thinking else 1)      # full depth leaves room for this reply
            target = (int(profile["context"]) - budget - int(profile.get("full_margin", 64))
                      if depth == "full" else int(depth))

            def count(chars):
                return post_json(url + "/v1/tokenize", {
                    "model": profile["model"], "chat_template_kwargs": kwargs,
                    "messages": needle_prompt(text, chars, fractions, codes)})["count"]

            lo, hi = 0, min(len(text), target * 8)
            while hi - lo > 64:                      # bisection on characters: the largest haystack within target
                mid = (lo + hi) // 2
                lo, hi = (mid, hi) if count(mid) <= target else (lo, mid)
            messages = needle_prompt(text, lo, fractions, codes)
            body = {"model": profile["model"], "messages": messages, "temperature": 0, "max_tokens": budget,
                    "chat_template_kwargs": kwargs, "return_token_ids": True}
            reply = stream_chat(url, body)
            answer = reply["content"]
            found = {vault: code in answer for vault, code in codes.items()}
            row = {"depth": depth, "target": target, "thinking": thinking,
                   "prompt_tokens": reply["usage"].get("prompt_tokens"), "codes": codes, "found": found,
                   "recall": sum(found.values()) / len(found), "answer": answer[:600],
                   "prefill_s": reply["stats"].get("prefill_s"), "ttft_s": reply["stats"].get("ttft_s")}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("depth", "thinking", "prompt_tokens", "recall", "ttft_s")}),
                  flush=True)
    recall = sum(r["recall"] for r in rows) / len(rows)
    verdict = None if args.min_needle_recall is None else all(r["recall"] >= args.min_needle_recall for r in rows)
    return {"gate": "needle", "recall": recall, "threshold": args.min_needle_recall, "verdict": verdict,
            "corpus_sha256": digest, "fractions": fractions, "rows": rows}


# ---------------------------------------------------------------------------------------------------------------
# determinism and drafted-vs-serial identity over bench prompts


def _items(args) -> list[dict]:
    items = {it["id"]: it for it in json.loads(Path(args.prompts).read_text())["items"]}
    chosen = [items[name] for name in args.gate_prompts.split(",") if name]
    return [it for it in chosen if args.thinking == "both" or it["thinking"] == (args.thinking == "on")]


EVICT = [{"role": "user", "content": "Reply with one word: ok. (cache eviction request)"}]


def gate_determinism(url: str, profile: dict, args) -> dict:
    rows = []
    evict = {"model": profile["model"], "messages": EVICT, "max_tokens": 4, "temperature": 0,
             "chat_template_kwargs": {"enable_thinking": False}}
    for item in _items(args):
        for mode, sampling in sampling_modes(profile, item["thinking"], args.seed).items():
            request = chat_request(profile, item, sampling, args.max_tokens)
            stream_chat(url, evict)                                   # this prompt is not cached
            cold = stream_chat(url, request)
            warm = stream_chat(url, request)                          # full-prompt hit
            stream_chat(url, evict)
            recold = stream_chat(url, request)
            # a next turn extending this prompt: the kept prefix (warm) against the same prompt after eviction
            turn = {**item, "messages": item["messages"] + [
                {"role": "assistant", "content": cold["content"] or cold["reasoning"]},
                {"role": "user", "content": "Continue."}]}
            turn_request = chat_request(profile, turn, sampling, args.max_tokens)
            extended = stream_chat(url, turn_request)
            stream_chat(url, evict)
            extended_cold = stream_chat(url, turn_request)
            row = {"prompt": item["id"], "sampling": mode,
                   "cached": {"cold": cold["stats"].get("cached"), "warm": warm["stats"].get("cached"),
                              "recold": recold["stats"].get("cached"), "extended": extended["stats"].get("cached"),
                              "extended_cold": extended_cold["stats"].get("cached")},
                   "warm_vs_cold": first_difference(reply_token_ids(cold), reply_token_ids(warm)),
                   "recold_vs_cold": first_difference(reply_token_ids(cold), reply_token_ids(recold)),
                   "extended_warm_vs_cold": first_difference(reply_token_ids(extended), reply_token_ids(extended_cold)),
                   "tokens": len(reply_token_ids(cold)), "sha": cold["stats"].get("token_sha")}
            row["identical"] = all(row[k] is None for k in ("warm_vs_cold", "recold_vs_cold", "extended_warm_vs_cold"))
            # whether each warm request resumed more cached tokens than its cold twin: an unexercised comparison is
            # a cold repeat, so the verdict covers prefix-cache independence only where these hold
            cached = row["cached"]
            row["cache_exercised"] = {"warm": (cached["warm"] or 0) > (cached["cold"] or 0),
                                      "extended": (cached["extended"] or 0) > (cached["extended_cold"] or 0)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    identical = sum(r["identical"] for r in rows)
    verdict = None if args.allow_nondeterminism else identical == len(rows)
    return {"gate": "determinism", "identical": identical, "cases": len(rows), "verdict": verdict,
            "warm_exercised": sum(r["cache_exercised"]["warm"] for r in rows),
            "extended_exercised": sum(r["cache_exercised"]["extended"] for r in rows), "rows": rows}


def gate_identity(url: str, profile: dict, args) -> dict:
    rows = []
    for item in _items(args):
        for mode, sampling in sampling_modes(profile, item["thinking"], args.seed).items():
            drafted = stream_chat(url, chat_request(profile, item, sampling, args.max_tokens))
            serial = stream_chat(url, chat_request(profile, item, sampling, args.max_tokens, draft=False))
            row = {"prompt": item["id"], "sampling": mode,
                   "first_difference": first_difference(reply_token_ids(drafted), reply_token_ids(serial)),
                   "tokens": len(reply_token_ids(drafted)), "drafted_sha": drafted["stats"].get("token_sha"),
                   "serial_sha": serial["stats"].get("token_sha"),
                   "drafted": {k: drafted["stats"].get(k) for k in ("rounds", "drafted", "accepted", "decode_tps")},
                   "serial": {k: serial["stats"].get(k) for k in ("rounds", "decode_tps")}}
            row["identical"] = row["first_difference"] is None
            rows.append(row)
            print(json.dumps(row), flush=True)
    greedy = [r for r in rows if r["sampling"] == "greedy"]
    sampled = [r for r in rows if r["sampling"] != "greedy"]
    verdict = None if args.allow_greedy_identity_failure else all(r["identical"] for r in greedy)
    return {"gate": "identity", "greedy_identical": sum(r["identical"] for r in greedy), "greedy_cases": len(greedy),
            "greedy_drafted": sum(bool(r["drafted"]["drafted"]) for r in greedy),   # cases that ran a draft at all
            "sampled_identical": sum(r["identical"] for r in sampled), "sampled_cases": len(sampled),
            "verdict": verdict, "rows": rows}


GATES = {"tools": gate_tools, "needle": gate_needle, "determinism": gate_determinism, "identity": gate_identity}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate", choices=(*GATES, "all"))
    ap.add_argument("--url", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2024, help="card-sampling seed and needle code generator seed")
    ap.add_argument("--prompts", help="bench prompt set (determinism, identity)")
    ap.add_argument("--gate-prompts", default="short-code-nothink,short-prose-think,4k-doc-review-nothink,"
                                              "32k-doc-code-nothink", help="prompt ids from --prompts")
    ap.add_argument("--max-tokens", type=int, default=512, help="reply tokens for determinism and identity")
    ap.add_argument("--thinking", choices=("on", "off", "both"), default="both",
                    help="thinking modes for tools, determinism and identity (needle: --needle-thinking)")
    ap.add_argument("--corpus", help="corpus file (needle haystack)")
    ap.add_argument("--needle-depths", default="4096,32768,131072,full")
    ap.add_argument("--needle-fractions", default="0.05,0.3,0.55,0.8,0.95")
    ap.add_argument("--needle-tokens", type=int, default=128, help="reply budget (x8 with thinking)")
    ap.add_argument("--needle-thinking", choices=("on", "off", "both"), default="off")
    ap.add_argument("--min-tool-pass", type=float, help="minimum pass rate in every tools group (unset: report)")
    ap.add_argument("--min-needle-recall", type=float, help="minimum recall at every depth (unset: report)")
    ap.add_argument("--allow-nondeterminism", action="store_true", help="report determinism without a verdict")
    ap.add_argument("--allow-greedy-identity-failure", action="store_true",
                    help="report drafted-vs-serial identity without a verdict")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    profile = json.loads(Path(args.profile).read_text())
    chosen = list(GATES) if args.gate == "all" else [args.gate]
    if "needle" in chosen and not args.corpus:
        ap.error("needle needs --corpus")
    if any(g in chosen for g in ("determinism", "identity")) and not args.prompts:
        ap.error("determinism and identity need --prompts")
    stats = get_json(url + "/v1/stats")
    result = {"version": 1, "url": url, "model": profile["model"], "profile_sha256": file_sha256(args.profile),
              "tool_sha256": {Path(f).name: file_sha256(f)
                              for f in (__file__, str(Path(__file__).with_name("campaign_bench.py")))},
              "source_revision": stats.get("server", {}).get("source_revision"), "args": vars(args), "gates": {}}
    for name in chosen:
        result["gates"][name] = GATES[name](url, profile, args)
        write_json(args.out, result)
    verdicts = {name: g["verdict"] for name, g in result["gates"].items()}
    print(json.dumps({"verdicts": verdicts}), flush=True)
    return 1 if any(v is False for v in verdicts.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
