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
threshold has status "report" and a null verdict. Determinism and identity are "incomplete" (verdict null) when a
required comparison was not exercised: a warm request that resumed no cached prefix, a cold one that did, a greedy
reply that verified no draft (silence is not coverage). Token ids are required from every reply. The overall status
is fail > incomplete > pass > report; exit status 1 = fail, 3 = incomplete, 4 = error (a refusal or failure of the
tool or server, never a verdict: status "error" with the reason, gates already run kept; when that record cannot be
written either, the printed summary's "record_error" says why), 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
import traceback

if sys.version_info < (3, 11):
    print(f"Python 3.11+ is required (this is {sys.version.split()[0]}); "
          "TensorFold's tools use PEP 604 unions. Try: python3.12 tools/campaign_gates.py ...", file=sys.stderr)
    sys.exit(4)                                     # an error, never a gate verdict

if __package__:
    from .benchmark_output import write_json
    from .campaign_bench import (LONG_PREFILL_TOKENS, Progress, chat_request, corpus_text, expected_tokens,
                                 file_sha256, first_difference, get_json, post_json, reply_token_ids, stream_chat)
else:
    from benchmark_output import write_json
    from campaign_bench import (LONG_PREFILL_TOKENS, Progress, chat_request, corpus_text, expected_tokens,
                                file_sha256, first_difference, get_json, post_json, reply_token_ids, stream_chat)

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
    # a non-streamed reply: the watchdog sizes its wait from the request (prompt estimate and reply budget)
    Progress.update("wait", request=f"tools {name} strict={strict} thinking={thinking}",
                    prompt_tokens=expected_tokens(body), max_tokens=body["max_tokens"])
    try:
        reply = post_json(url + "/v1/chat/completions", body, timeout=3600)
    finally:
        Progress.update("idle", force=True)
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


def _threshold_status(verdict: bool | None) -> str:
    return "report" if verdict is None else ("pass" if verdict else "fail")


def _hold(record: dict) -> dict:
    """Gate coverage the long-prefill hold kept from running: printed, and returned as a held record."""

    print(json.dumps({**record, "held": "long-prefill hold"}), flush=True)
    return {**record, "status": "incomplete (hold)"}


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
    return {"gate": "tools", "status": _threshold_status(verdict), "pass_rate": overall,
            "pass_rate_by_group": rates, "threshold": args.min_tool_pass, "verdict": verdict, "rows": rows}


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
    rows, held = [], []
    for depth in [d for d in args.needle_depths.split(",") if d]:
        codes = {VAULTS[i]: f"{rng.randrange(10 ** 6):06d}" for i in range(len(fractions))}
        for thinking in ([False, True] if args.needle_thinking == "both" else [args.needle_thinking == "on"]):
            kwargs = {"enable_thinking": thinking}
            budget = int(args.needle_tokens) * (8 if thinking else 1)      # full depth leaves room for this reply
            target = (int(profile["context"]) - budget - int(profile.get("full_margin", 64))
                      if depth == "full" else int(depth))
            if target > LONG_PREFILL_TOKENS and not args.allow_long_prefill:
                held.append(_hold({"depth": depth, "target": target, "thinking": thinking}))
                continue

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
            reply = stream_chat(url, body, prompt_tokens=target, label=f"needle {depth} thinking={thinking}")
            answer = reply["content"]
            found = {vault: code in answer for vault, code in codes.items()}
            row = {"depth": depth, "target": target, "thinking": thinking,
                   "prompt_tokens": reply["usage"].get("prompt_tokens"), "codes": codes, "found": found,
                   "recall": sum(found.values()) / len(found), "answer": answer[:600],
                   "prefill_s": reply["stats"].get("prefill_s"), "ttft_s": reply["stats"].get("ttft_s")}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("depth", "thinking", "prompt_tokens", "recall", "ttft_s")}),
                  flush=True)
    recall = sum(r["recall"] for r in rows) / len(rows) if rows else None
    verdict = None if args.min_needle_recall is None else all(r["recall"] >= args.min_needle_recall for r in rows)
    status = _threshold_status(verdict)
    if status != "fail" and (held or not rows):            # held depths are coverage not run, never a pass
        status, verdict = "incomplete", None
    return {"gate": "needle", "status": status, "recall": recall, "threshold": args.min_needle_recall,
            "verdict": verdict, "corpus_sha256": digest, "fractions": fractions, "rows": rows, "held": held}


# ---------------------------------------------------------------------------------------------------------------
# determinism and drafted-vs-serial identity over bench prompts


def _items(args, reply_tokens: int = 0) -> tuple[list[dict], list[dict]]:
    """The gate prompts to run, and those held back by the long-prefill hold (reported as incomplete): a prompt whose
    tokens plus ``reply_tokens`` exceed LONG_PREFILL_TOKENS (determinism re-sends a prompt with its reply as one
    prompt)."""

    items = {it["id"]: it for it in json.loads(Path(args.prompts).read_text())["items"]}
    chosen = [items[name] for name in args.gate_prompts.split(",") if name]
    chosen = [it for it in chosen if args.thinking == "both" or it["thinking"] == (args.thinking == "on")]
    kept, held = [], []
    for it in chosen:
        longest = it["prompt_tokens"] + reply_tokens
        if longest <= LONG_PREFILL_TOKENS or args.allow_long_prefill:
            kept.append(it)
        else:
            held.append(_hold({"prompt": it["id"], "prompt_tokens": it["prompt_tokens"],
                               "longest_prompt_tokens": longest}))
    return kept, held


def _prompt_ids(url: str, profile: dict, item: dict) -> list[int]:
    """A chat item's prompt ids from the native /tokenize route: a production image's Harness bridge owns
    /v1/tokenize and returns ids only up to 16,384 tokens (a 400 beyond), while /tokenize is TensorFold's own on
    every server."""

    return post_json(url + "/tokenize", {"model": profile["model"], "messages": item["messages"],
                                         "chat_template_kwargs": {"enable_thinking": item["thinking"]}})["tokens"]


def _evictions(count: int) -> list[list[dict]]:
    """Distinct ~600-token prompts that displace every kept prefix-cache state of an earlier prompt: one-entry caches
    (Gemma4) need one, multi-state caches (Qwen keeps message-start and prompt-end states) need several prompts long
    enough to be kept."""

    words = ("amber", "basalt", "cobalt", "dune", "ember", "flint", "garnet", "harbor", "indigo", "jasper")
    return [[{"role": "user", "content": f"Cache eviction request {n}. " + " ".join(
        f"{words[(n + i) % len(words)]}-{i}" for i in range(300)) + " Reply with one word: ok."}]
        for n in range(count)]


def _evict(url: str, profile: dict, args) -> None:
    for messages in _evictions(args.evict_requests):
        stream_chat(url, {"model": profile["model"], "messages": messages, "max_tokens": 4, "temperature": 0,
                          "chat_template_kwargs": {"enable_thinking": False}})


def _cached(reply: dict) -> int:
    return int(reply["stats"].get("cached") or 0)


def _status(rows: list[dict], allow_failure: bool, held: list[dict]) -> tuple[str, bool | None]:
    """A gate's status from its rows: fail when any compared pair differs (unless reporting only), incomplete when
    there are no rows, any required comparison was not exercised (silence is not coverage) or items were held back
    (long-prefill hold), else pass. verdict mirrors it (null for incomplete or report-only)."""

    if any(not r["identical"] for r in rows):
        return ("report", None) if allow_failure else ("fail", False)
    if not rows or held or any(not all(r["exercised"].values()) for r in rows):
        return "incomplete", None
    return ("report", None) if allow_failure else ("pass", True)


def gate_determinism(url: str, profile: dict, args) -> dict:
    """Per prompt and sampling mode, every pair must be token-identical:
    - repeat: the prompt cold (after eviction) against the same prompt warm (a full prompt-cache hit) and against it
      cold again after another eviction;
    - extension: the prompt's token ids plus its own reply, sent right after the prompt (the cache holds the prompt:
      a partial hit) against the same ids cold after eviction (/v1/completions with token ids, so the extension is
      exact for any chat template);
    - chat next turn (reported, not required): the conversation continued with the reply and "Continue.", warm
      against cold. A template whose generation prompt is not a prefix of its history rendering (Gemma4 without
      thinking appends an empty thought block) never resumes it, so it is not a required comparison.
    A comparison counts as exercised only when its warm request resumed >= 90% of the shared prefix from the cache
    and its cold twin resumed <= 10%. Without --allow-long-prefill, a prompt whose tokens plus the reply budget exceed
    LONG_PREFILL_TOKENS is held, and so is a next turn whose prompt does; held work makes the gate incomplete."""

    rows = []
    items, held = _items(args, args.max_tokens)          # the extension prompt is a prompt plus its whole reply
    for item in items:
        prompt_ids = _prompt_ids(url, profile, item)
        for mode, sampling in sampling_modes(profile, item["thinking"], args.seed).items():
            request = chat_request(profile, item, sampling, args.max_tokens)
            _evict(url, profile, args)
            cold = stream_chat(url, request)
            warm = stream_chat(url, request)
            _evict(url, profile, args)
            recold = stream_chat(url, request)
            extended_ids = list(prompt_ids) + reply_token_ids(cold)
            extension = {"model": profile["model"], "prompt": extended_ids, "max_tokens": args.max_tokens,
                         "return_token_ids": True, "ignore_eos": bool(profile.get("ignore_eos", True)), **sampling}
            extended = stream_chat(url, extension, route="/v1/completions")
            _evict(url, profile, args)
            extended_cold = stream_chat(url, extension, route="/v1/completions")
            turn = {**item, "messages": item["messages"] + [
                {"role": "assistant", "content": cold["content"] or cold["reasoning"]},
                {"role": "user", "content": "Continue."}]}
            turn_request = chat_request(profile, turn, sampling, args.max_tokens)
            turn_tokens = len(_prompt_ids(url, profile, turn))
            if turn_tokens > LONG_PREFILL_TOKENS and not args.allow_long_prefill:
                # the reported next turn's cold prefill would exceed the hold: not run, recorded as held
                held.append(_hold({"prompt": item["id"], "sampling": mode, "next_turn_prompt_tokens": turn_tokens}))
                next_turn = next_turn_cold = None
            else:
                stream_chat(url, request)                              # the cache holds this prompt again
                next_turn = stream_chat(url, turn_request)
                _evict(url, profile, args)
                next_turn_cold = stream_chat(url, turn_request)
            ran = next_turn is not None
            n, m = len(prompt_ids), len(extended_ids)
            row = {"prompt": item["id"], "sampling": mode, "prompt_tokens": n,
                   "cached": {"cold": _cached(cold), "warm": _cached(warm), "recold": _cached(recold),
                              "extended": _cached(extended), "extended_cold": _cached(extended_cold),
                              "next_turn": _cached(next_turn) if ran else None,
                              "next_turn_cold": _cached(next_turn_cold) if ran else None},
                   "differences": {
                       "warm_vs_cold": first_difference(reply_token_ids(cold), reply_token_ids(warm)),
                       "recold_vs_cold": first_difference(reply_token_ids(cold), reply_token_ids(recold)),
                       "extended_warm_vs_cold": first_difference(reply_token_ids(extended),
                                                                 reply_token_ids(extended_cold)),
                       "next_turn_warm_vs_cold": first_difference(reply_token_ids(next_turn),
                                                                  reply_token_ids(next_turn_cold)) if ran else None},
                   "tokens": len(reply_token_ids(cold)), "sha": cold["stats"].get("token_sha")}
            c = row["cached"]
            row["exercised"] = {"warm": c["cold"] <= 0.1 * n and c["warm"] >= 0.9 * n,
                                "recold": c["recold"] <= 0.1 * n,
                                "extended": c["extended"] >= 0.9 * n and c["extended_cold"] <= 0.1 * m}
            row["next_turn_exercised"] = ran and c["next_turn"] > c["next_turn_cold"]
            row["identical"] = all(v is None for v in row["differences"].values())
            rows.append(row)
            print(json.dumps(row), flush=True)
    status, verdict = _status(rows, args.allow_nondeterminism, held)
    return {"gate": "determinism", "status": status, "verdict": verdict, "cases": len(rows), "held": held,
            "identical": sum(r["identical"] for r in rows),
            "exercised": {k: sum(r["exercised"][k] for r in rows) for k in ("warm", "recold", "extended")},
            "next_turn_exercised": sum(r["next_turn_exercised"] for r in rows), "rows": rows}


def gate_identity(url: str, profile: dict, args) -> dict:
    """Drafted (default) against serial ("draft": false) per prompt: greedy must be token-identical and is exercised
    only when the drafted reply verified at least one draft; seeded sampling is reported."""

    rows = []
    items, held = _items(args)
    for item in items:
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
            row["exercised"] = {"drafts": bool(row["drafted"]["drafted"])}
            rows.append(row)
            print(json.dumps(row), flush=True)
    greedy = [r for r in rows if r["sampling"] == "greedy"]
    sampled = [r for r in rows if r["sampling"] != "greedy"]
    status, verdict = _status(greedy, args.allow_greedy_identity_failure, held)
    return {"gate": "identity", "status": status, "verdict": verdict, "held": held,
            "greedy_identical": sum(r["identical"] for r in greedy), "greedy_cases": len(greedy),
            "greedy_drafted": sum(r["exercised"]["drafts"] for r in greedy),
            "sampled_identical": sum(r["identical"] for r in sampled), "sampled_cases": len(sampled), "rows": rows}


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
    ap.add_argument("--evict-requests", type=int, default=6,
                    help="distinct ~600-token prompts sent to displace kept prefix-cache states before a cold request")
    ap.add_argument("--progress", help="heartbeat file for a watchdog (campaign_bench.Progress)")
    ap.add_argument("--allow-long-prefill", action="store_true",
                    help=f"run prompts above {LONG_PREFILL_TOKENS} tokens (needle depths, gate prompts; determinism "
                         "also counts a prompt plus its reply budget and its next chat turn); without it they are "
                         "held during the campaign's long-prefill hold and reported as incomplete (hold)")
    ap.add_argument("--allow-nondeterminism", action="store_true", help="report determinism without a verdict")
    ap.add_argument("--allow-greedy-identity-failure", action="store_true",
                    help="report drafted-vs-serial identity without a verdict")
    args = ap.parse_args()
    chosen = list(GATES) if args.gate == "all" else [args.gate]
    if "needle" in chosen and not args.corpus:
        ap.error("needle needs --corpus")
    if any(g in chosen for g in ("determinism", "identity")) and not args.prompts:
        ap.error("determinism and identity need --prompts")
    url = args.url.rstrip("/")
    result = {"version": 1, "url": url, "args": vars(args), "gates": {}}
    try:                                # from here on any exception is an error result (exit 4), never a verdict
        Progress.configure(args.progress)
        profile = json.loads(Path(args.profile).read_text())
        result.update(model=profile["model"], profile_sha256=file_sha256(args.profile),
                      tool_sha256={Path(f).name: file_sha256(f)
                                   for f in (__file__, str(Path(__file__).with_name("campaign_bench.py")))},
                      source_revision=get_json(url + "/v1/stats").get("server", {}).get("source_revision"))
        for name in chosen:
            result["gates"][name] = GATES[name](url, profile, args)
            write_json(args.out, result)
        statuses = {name: g["status"] for name, g in result["gates"].items()}
        result["status"] = ("fail" if "fail" in statuses.values() else "incomplete" if "incomplete" in statuses.values()
                            else "pass" if "pass" in statuses.values() else "report")
        write_json(args.out, result)
    except Exception as error:  # noqa: BLE001 - CLI boundary: any failure after parsing is an error (exit 4), never a verdict
        traceback.print_exc()
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        summary = {"status": "error", "error": result["error"]}
        try:                            # the record is best effort; the exit status is the contract
            write_json(args.out, result)
        # --out not an ordinary file, NaN, I/O, a bad value, exhausted memory: an escape would exit 1, the fail code
        except Exception as unwritten:  # noqa: BLE001 - best-effort record: exit 4 whatever its write raised
            traceback.print_exc()
            summary["record_error"] = f"{type(unwritten).__name__}: {unwritten}"
        print(json.dumps(summary), flush=True)
        return 4
    print(json.dumps({"status": result["status"], "gates": statuses}), flush=True)
    return {"fail": 1, "incomplete": 3}.get(result["status"], 0)


if __name__ == "__main__":
    raise SystemExit(main())
