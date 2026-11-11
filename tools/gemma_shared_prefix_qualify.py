"""Bounded public-data shared-prefix qualification client. Importing performs no I/O.

Service/model/container lifecycle belongs to the release coordinator. This client
runs one explicitly selected stage against its supplied loopback endpoint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
import re
import stat
import time
import urllib.parse
from pathlib import Path

if __package__:
    from .benchmark_output import write_json
    from .worker_lifetime import raise_failures
else:
    from benchmark_output import write_json
    from worker_lifetime import raise_failures

CAP = 6 * 1024 * 1024
TIMEOUT = 300
SCHEMA = "tensorfold.gemma-shared-prefix-qualification-v1"
REQUESTS = ("cache-A-first", "cache-A-extended", "clock-call-stream", "clock-followup-stream")


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, name):
    require(type(value) is int and value >= 0, name + " must be a nonnegative integer")
    return value




def finish_errors(primary, errors):
    """Retirements are complete; preserve the first through diagnostic exhaustion.

    On transport failure, original status references and native context remain
    owned by this traceback frame. No new diagnostic allocation is required.
    """
    first = primary
    if first is None:
        for error in errors:
            if error is not None:
                first = error
                break
    if first is None:
        return
    native_cause = BaseException.__cause__.__get__(first)
    native_context = BaseException.__context__.__get__(first)
    native_suppress = BaseException.__suppress_context__.__get__(first)
    try:
        compact = [error for error in errors if error is not None]
        raise_failures(first, compact)
    except BaseException as transport:
        if transport is first:
            raise
        BaseException.__cause__.__set__(first, native_cause)
        BaseException.__context__.__set__(first, native_context)
        BaseException.__suppress_context__.__set__(first, native_suppress)
        raise first
    raise first


def read_bound(path, limit):
    errors, primary, data = [None], None, None
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, "bounded regular input required")
        parts, count = [], 0
        while count <= limit:
            part = os.read(fd, min(65536, limit + 1 - count))
            if not part:
                break
            parts.append(part)
            count += len(part)
        require(count <= limit, "input exceeds capture bound")
        after = os.fstat(fd)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "input changed during read")
        data = b"".join(parts)
    except BaseException as error:
        primary = error
    try:
        os.close(fd)
    except BaseException as error:
        errors[0] = error
    finish_errors(primary, errors)
    return data


def load_inputs(path, digest):
    raw = read_bound(path, CAP)
    require(len(raw) <= CAP and hashlib.sha256(raw).hexdigest() == digest, "input manifest identity/size mismatch")
    spec = json.loads(raw)
    require(type(spec.get("schema")) is int and spec["schema"] == 1 and type(spec.get("model")) is str, "invalid input schema/model")
    def read_ref(ref):
        require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}, "invalid input reference")
        expected = integer(ref["bytes"], "input bytes")
        require(expected <= CAP, "input reference exceeds fixture limit")
        data = read_bound(ref["path"], expected)
        require(len(data) == expected and hashlib.sha256(data).hexdigest() == ref["sha256"], "frozen input differs")
        return json.loads(data)
    cases = read_ref(spec["retention"])
    require(set(cases) == {"six", "ten"}, "six/ten retention scenarios required")
    for key, count in (("six", 6), ("ten", 10)):
        require(len(cases[key]["short"]) == count and len(cases[key]["long"]) == 1, "retention geometry differs")
    requests = {key: read_ref(spec["requests"][key]) for key in REQUESTS}
    require(all(type(value) is dict for value in requests.values()), "request bodies must be objects")
    require(type(spec["raw"]["ordinary_text"]) is str and type(spec["raw"]["suffix_messages"]) is list,
            "raw ordinary/suffix authority missing")
    require(spec["raw"]["seed_scenario"] in cases, "unknown raw seed scenario")
    require(type(spec["responses_body"]) is dict and type(spec["responses_followup"]) is str,
            "explicit Responses bodies required")
    return spec, cases, requests


class Evidence:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(mode=0o700)  # Fresh only: never overwrite prior evidence.
        self.sequence = 0

    def save(self, name, value):
        require(not (self.root / name).exists(), "evidence already exists")
        write_json(self.root / name, value)

    def raw(self, name, data):
        errors = [None]
        primary = None
        fd = os.open(self.root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            view = memoryview(data)
            while view:
                count = os.write(fd, view)
                require(count > 0, "short evidence write")
                view = view[count:]
            os.fsync(fd)
        except BaseException as error:
            primary = error
        try:
            os.close(fd)
        except BaseException as error:
            errors[0] = error
        finish_errors(primary, errors)


def sse_frames(raw):
    text = raw.decode("utf-8")
    require(not text or text.endswith(("\n\n", "\r\n\r\n")), "truncated SSE frame")
    frames, done = [], False
    for block in re.split(r"\r?\n\r?\n", text):
        data = [line[5:].lstrip(" ") for line in block.splitlines() if line.startswith("data:")]
        if not data:
            continue
        payload = "\n".join(data)
        if payload == "[DONE]":
            require(not done, "duplicate DONE")
            done = True
        else:
            require(not done, "event after DONE")
            frames.append(json.loads(payload))
    return frames, done


def normalize(raw, streaming, responses=False):
    events, done = sse_frames(raw) if streaming else ([json.loads(raw)], False)
    if responses:
        finals = [event["response"] for event in events if event.get("type") in
                  ("response.completed", "response.incomplete")]
        reply = finals[-1] if streaming and finals else (events[0] if not streaming else None)
        require(reply is not None and reply.get("status") in ("completed", "incomplete"), "Responses terminal missing")
        message = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}
        for item in reply["output"]:
            if item["type"] in ("message", "reasoning"):
                key = "content" if item["type"] == "message" else "reasoning_content"
                message[key] += "".join(part.get("text", "") for part in item.get("content", []))
            elif item["type"] == "function_call":
                message["tool_calls"].append({"id": item["call_id"], "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]}})
        return {"message": message, "finish": reply["status"], "usage": reply["usage"],
                "tensorfold": reply.get("tensorfold", {}), "response": reply}
    message, finish, usage, stats, calls = {"role": "assistant", "content": "", "reasoning_content": ""}, None, None, {}, {}
    for event in events:
        require("error" not in event, "server reported an error")
        usage = event.get("usage") or usage
        stats = event.get("tensorfold") or stats
        for choice in event.get("choices", []):
            require(choice.get("index", 0) == 0, "one output choice required")
            delta = choice.get("delta") if streaming else choice.get("message", {"content": choice.get("text", "")})
            if delta:
                for key in ("content", "reasoning_content"):
                    message[key] += delta.get(key) or ""
                for part in delta.get("tool_calls") or []:
                    index = part.get("index", 0) if streaming else len(calls)
                    require(type(index) is int and 0 <= index <= len(calls), "invalid tool index")
                    call = calls.setdefault(index, {"id": "", "type": "function",
                                                   "function": {"name": "", "arguments": ""}})
                    call["id"] += part.get("id") or ""
                    for key in ("name", "arguments"):
                        call["function"][key] += (part.get("function") or {}).get(key) or ""
            finish = choice.get("finish_reason") or finish
    require(finish is not None and usage is not None and (not streaming or done), "terminal/usage/DONE missing")
    message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return {"message": message, "finish": finish, "usage": usage, "tensorfold": stats}


class Client:
    def __init__(self, endpoint, evidence, arm, timeout=TIMEOUT, cap=CAP):
        url = urllib.parse.urlsplit(endpoint)
        require(url.scheme == "http" and url.hostname == "127.0.0.1" and not url.path.strip("/")
                and not url.query and not url.fragment and url.username is None, "explicit loopback HTTP endpoint required")
        require(type(timeout) is int and timeout > 0 and type(cap) is int and cap > 0, "positive finite limits required")
        self.host, self.port, self.out, self.arm = url.hostname, url.port or 80, evidence, arm
        self.timeout, self.cap, self.results = timeout, cap, []
        self.instance = None

    def call(self, name, path, body=None):
        self.out.sequence += 1
        tag = f"{self.out.sequence:03d}-{name}"
        self.out.save(tag + "-request.json", {"path": path, "body": body})
        raw, response, primary, errors = bytearray(), None, None, [None, None, None]
        arrival = time.monotonic()
        marks = {"arrival_monotonic": arrival, "first_byte_s": None, "first_sse_data_s": None,
                 "first_visible_s": None, "last_visible_s": None, "terminal_s": None}
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        status, headers, owned_socket = None, None, None
        try:
            data = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            connection.request("GET" if body is None else "POST", path, data,
                               {} if body is None else {"Content-Type": "application/json"})
            owned_socket = connection.sock
            require(time.monotonic() - arrival < self.timeout, "whole request deadline exceeded")
            if owned_socket is not None:
                owned_socket.settimeout(self.timeout - (time.monotonic() - arrival))
            response = connection.getresponse()
            status, headers = response.status, dict(response.getheaders())
            declared = response.getheader("Content-Length")
            if declared is not None:
                require(declared.isascii() and declared.isdigit() and len(declared) <= 20,
                        "invalid Content-Length")
                declared = int(declared)
            partial = bytearray()
            while True:
                if response.isclosed():
                    break
                remaining = self.timeout - (time.monotonic() - arrival)
                require(remaining > 0, "whole request deadline exceeded")
                # Keep the actual socket while HTTPResponse owns its makefile,
                # including Connection: close responses detached from the client.
                if owned_socket is not None:
                    owned_socket.settimeout(remaining)
                chunk = response.read1(min(65536, self.cap + 1 - len(raw)))
                now = time.monotonic() - arrival
                if not chunk:
                    break
                if marks["first_byte_s"] is None:
                    marks["first_byte_s"] = now
                raw.extend(chunk)
                require(len(raw) <= self.cap, "response capture cap exceeded")
                partial.extend(chunk)
                while (boundary := re.search(br"\r?\n\r?\n", partial)) is not None:
                    block = bytes(partial[:boundary.start()])
                    partial = partial[boundary.end():]
                    lines = [line[5:].lstrip() for line in block.splitlines() if line.startswith(b"data:")]
                    if lines:
                        if marks["first_sse_data_s"] is None:
                            marks["first_sse_data_s"] = now
                        payload = b"\n".join(lines)
                        if payload == b"[DONE]":
                            marks["terminal_s"] = now
                        else:
                            event = json.loads(payload)
                            visible = any((choice.get("delta") or {}).get(key) for choice in event.get("choices", [])
                                          for key in ("content", "reasoning_content", "tool_calls"))
                            visible |= event.get("type") in ("response.output_text.delta", "response.reasoning_text.delta",
                                                            "response.function_call_arguments.delta")
                            if visible:
                                marks["first_visible_s"] = now if marks["first_visible_s"] is None else marks["first_visible_s"]
                                marks["last_visible_s"] = now
                            if event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                                marks["terminal_s"] = now
                require(now <= self.timeout, "whole request deadline exceeded")
            require(declared is None or declared == len(raw), "truncated Content-Length response")
            require(status == 200, "HTTP status " + str(status))
        except BaseException as error:
            primary = error
        if response is not None:
            try:
                response.close()
            except BaseException as error:
                errors[0] = error
        try:
            connection.close()
        except BaseException as error:
            errors[1] = error
        marks["full_wall_s"] = time.monotonic() - arrival
        marks["last_visible_to_terminal_s"] = (None if marks["terminal_s"] is None or marks["last_visible_s"] is None
                                               else marks["terminal_s"] - marks["last_visible_s"])
        try:
            self.out.raw(tag + "-response.raw", raw)
            self.out.save(tag + "-transport.json", {"status": status, "headers": headers, "timing": marks,
                          "failure": None if primary is None else type(primary).__name__})
        except BaseException as error:
            errors[2] = error
        finish_errors(primary, errors)
        return bytes(raw), tag, marks

    def json(self, name, path, body=None):
        raw, _, _ = self.call(name, path, body)
        return json.loads(raw)

    def tokens(self, name, body):
        # The preserved Harness owns /v1/tokenize: open chat counting with
        # explicit ID inspection. Raw text and closed-history inspection use
        # the separately declared native public alias, which honors those flags.
        # This is operation binding, never response-triggered retry/fallback.
        route = ("/v1/tokenize" if "messages" in body and body.get("add_generation_prompt", True) is True
                 else "/tokenize")
        inspection = {**body, "return_tokens": True}
        reply = self.json(name, route, inspection)
        require(type(reply) is dict and "tokens" in reply, "served native token IDs missing despite explicit inspection")
        ids = reply["tokens"]
        require(type(ids) is list and all(type(token) is int and token >= 0 for token in ids)
                and integer(reply["count"], "tokenizer count") == len(ids), "tokenizer count/IDs disagree")
        return ids

    def stats(self, name):
        result = self.json(name, "/v1/stats")
        require(result["model"]["context_length"] == 262144, "native context changed")
        instance = result["server"]["instance_id"]
        require(self.instance is None or instance == self.instance, "server instance changed")
        self.instance = instance
        cache = result["prefix_cache"]
        budget, charged = integer(cache["byte_budget"], "cache budget"), integer(cache["charged_bytes"], "charged cache")
        require(charged + integer(cache.get("failed_snapshot_bytes", 0), "failed snapshot bytes") <= budget,
                "cache byte budget exceeded")
        if self.arm == "candidate":
            require(cache.get("store_schema") == "gemma-canonical-block-archive-v1", "candidate archive schema missing")
            require(all(key in cache and cache[key] is None for key in ("entries", "max_entries", "capacity", "tokens")),
                    "candidate reports fictitious entry/token counts")
            require(cache.get("metadata_status") == "ready", "candidate metadata not ready")
            require(cache["eviction_history_capacity"] == 32 and len(cache["eviction_history"]) <= 32,
                    "unbounded component history")
        return result

    def generate(self, name, path, body):
        raw, tag, marks = self.call(name, path, body)
        result = normalize(raw, bool(body.get("stream")), path == "/v1/responses")
        usage = result["usage"]
        responses = path == "/v1/responses"
        prompt = integer(usage["input_tokens" if responses else "prompt_tokens"], "prompt usage")
        output = integer(usage["output_tokens" if responses else "completion_tokens"], "output usage")
        cached = integer(usage.get("input_tokens_details" if responses else "prompt_tokens_details", {}).get("cached_tokens"),
                         "cached usage")
        require(cached <= prompt and output > 0, "invalid completed work/usage")
        ids = result["tensorfold"].get("token_ids")
        if body.get("return_token_ids"):
            require(type(ids) is list and len(ids) == output and all(type(token) is int for token in ids),
                    "actual accepted output IDs missing")
        message = result["message"]
        require(not any(marker in (message.get(key) or "") for marker in ("<|channel>", "<channel|>")
                        for key in ("content", "reasoning_content")), "structural channel leaked")
        record = {"name": name, "prompt_tokens": prompt, "output_tokens": output, "cached_tokens": cached, "uncached_original_input_tokens": prompt - cached,
                  "timing": marks, "server": result["tensorfold"], "message": message, "finish": result["finish"]}
        self.out.save(tag + "-result.json", record)
        self.results.append({key: value for key, value in record.items() if key not in ("message", "server")})
        self.stats(name + "-settled")
        result.update(prompt=prompt, output=output, cached=cached)
        return result

    def feature(self, passed, name):
        if self.arm == "candidate":
            require(passed, name)
        return bool(passed)


def assistant(result):
    message = result["message"]
    out = {"role": "assistant", "content": message.get("content") or None}
    for key in ("reasoning_content", "tool_calls"):
        if message.get(key):
            out[key] = message[key]
    return out


def same(first, second):
    require(first["message"] == second["message"] and first["finish"] == second["finish"], "deterministic output differs")
    if "token_ids" in first["tensorfold"] or "token_ids" in second["tensorfold"]:
        require(first["tensorfold"].get("token_ids") == second["tensorfold"].get("token_ids"),
                "accepted output IDs differ")


def json_ok(result):
    require(json.loads(result["message"].get("content") or "") == {"ok": True}, "independent JSON gold failed")


def run_raw(client, spec, cases):
    seed = {"model": spec["model"], "messages": cases[spec["raw"]["seed_scenario"]]["long"],
            "chat_template_kwargs": {"enable_thinking": False}}
    ids = client.tokens("raw-seed", seed)
    require(len(ids) == 40000, "frozen 40K seed no longer aligns")
    empty = copy.deepcopy(seed)
    empty["messages"][0]["content"] = ""
    blank = client.tokens("raw-empty-frame", empty)
    prefix = 0
    while prefix < min(len(ids), len(blank)) and ids[prefix] == blank[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(ids), len(blank)) - prefix and ids[-suffix-1] == blank[-suffix-1]:
        suffix += 1
    require(0 < prefix < 4000 and 0 < suffix < 7999, "actual tokenizer framing cannot be preserved")
    require(not any(marker in seed["messages"][0]["content"] for marker in ("<|", "<turn|>")),
            "raw public source contains structural marker spellings")
    a = ids[:12000] + ids[-12000:]
    ordinary = client.tokens("raw-ordinary", {"model": spec["model"], "prompt": spec["raw"]["ordinary_text"],
                                             "add_special_tokens": False})
    changed = None
    for token in ordinary:
        text = client.json("raw-ordinary-proof", "/v1/detokenize", {"tokens": [token]})["prompt"]
        if token != a[4000] and re.fullmatch(r"[ A-Za-z]+", text):
            changed = token
            break
    require(changed is not None, "no proven ordinary differing token")
    b = a[:4000] + [changed] + a[-7999:]
    require(len(a) == 24000 and len(b) == 12000 and a[:4000] == b[:4000] and a[4000] != b[4000],
            "raw branch geometry differs")
    client.out.save("raw-tapes.json", {"a": a, "b": b, "source_seed_tokens": len(ids), "shared": 4000,
                                      "verified_frame_prefix": prefix, "verified_frame_suffix": suffix})
    controls = {"model": spec["model"], "seed": 7, "temperature": 0, "max_tokens": 32, "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False}}
    first = client.generate("raw-A-first", "/v1/completions", {**controls, "prompt": a})
    json_ok(first)
    branch = client.generate("raw-B", "/v1/completions", {**controls, "prompt": b})
    json_ok(branch)
    require(branch["prompt"] == 12000, "B prompt usage differs from exact raw IDs")
    client.feature(branch["cached"] == 4000, "B did not reuse exact4000 prefix")
    repeat = client.generate("raw-A-repeat", "/v1/completions", {**controls, "prompt": a})
    same(first, repeat)
    json_ok(repeat)
    client.feature(repeat["cached"] == len(a), "A branch did not survive B")
    suffix = {"model": spec["model"], "messages": spec["raw"]["suffix_messages"],
              "chat_template_kwargs": {"enable_thinking": False}}
    closed = {**suffix, "messages": suffix["messages"][:-1], "add_generation_prompt": False}
    before, after = client.tokens("raw-suffix-closed", closed), client.tokens("raw-suffix-open", suffix)
    require(after[:len(before)] == before, "actual suffix framing is not append-only")
    output = first["tensorfold"]["token_ids"]
    require(first["finish"] == "stop", "raw continuation needs normal completed source turn")
    continuation = a + output + after[len(before):]
    grown = client.generate("raw-A-output-continuation", "/v1/completions", {**controls, "prompt": continuation})
    json_ok(grown)
    client.feature(grown["cached"] >= len(a) + len(output), "final accepted output token was not reusable")
    return {"shared_hit": branch["cached"], "continued_hit": grown["cached"]}


def run_canonical(client, spec, requests):
    body = copy.deepcopy(requests["cache-A-first"])
    first = client.generate("chat-first", "/v1/chat/completions", body)
    closed = {**body, "messages": body["messages"] + [assistant(first)], "add_generation_prompt": False}
    closed_ids = client.tokens("chat-canonical", closed)
    following = copy.deepcopy(requests["cache-A-extended"])
    following["messages"] = closed["messages"] + [following["messages"][-1]]
    next_ids = client.tokens("chat-next-authority", following)
    require(next_ids[:len(closed_ids)] == closed_ids, "Chat canonical history is not exact prefix")
    hit = client.generate("chat-next", "/v1/chat/completions", following)
    client.feature(hit["cached"] >= len(closed_ids), "complete Chat canonical history not reused")
    repeat = client.generate("chat-next-repeat", "/v1/chat/completions", following)
    same(hit, repeat)
    for stream in (False, True):
        call_body = {**copy.deepcopy(requests["clock-call-stream"]), "stream": stream}
        called = client.generate("tool-call-" + str(stream), "/v1/chat/completions", call_body)
        calls = called["message"]["tool_calls"]
        require(len(calls) == 1 and json.loads(calls[0]["function"]["arguments"])["zone"] == "UTC", "tool gold failed")
        follow = copy.deepcopy(requests["clock-followup-stream"])
        follow["stream"] = stream
        tool = {**follow["messages"][-1], "tool_call_id": calls[0]["id"]}
        follow["messages"] = call_body["messages"] + [assistant(called), tool]
        closed_tool = {**call_body, "messages": follow["messages"][:-1], "add_generation_prompt": False}
        wanted = len(client.tokens("tool-canonical-" + str(stream), closed_tool))
        result = client.generate("tool-follow-" + str(stream), "/v1/chat/completions", follow)
        require("12:34" in result["message"]["content"], "tool-result content gold failed")
        client.feature(result["cached"] >= wanted, "actual tool-call history not reused")
    for stream in (False, True):
        response_body = {**copy.deepcopy(spec["responses_body"]), "stream": stream}
        result = client.generate("response-first-" + str(stream), "/v1/responses", response_body)
        publication = result["tensorfold"].get("cache_publication")
        frontier = publication.get("materialized_tokens") if type(publication) is dict else None
        published = (type(publication) is dict and publication.get("mode") == "frontend"
                     and publication.get("complete") is True and type(frontier) is int and frontier > 0)
        client.feature(published, "Responses canonical materialized frontier missing")
        follow = {**response_body, "previous_response_id": result["response"]["id"],
                  "input": spec["responses_followup"]}
        result = client.generate("response-next-" + str(stream), "/v1/responses", follow)
        require(result["message"]["content"].strip(), "Responses followup empty")
        client.feature(published and result["cached"] >= frontier,
                       "Responses complete materialized history was not reused")
    return {"chat_canonical_tokens": len(closed_ids)}


def run_retention(client, spec, cases):
    controls = {"model": spec["model"], "seed": 7, "temperature": 0, "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "cache_check", "strict": True,
                    "schema": {"type": "object", "properties": {"ok": {"const": True}},
                               "required": ["ok"], "additionalProperties": False}}}}
    records = []
    for scenario in ("six", "ten"):
        case = cases[scenario]
        source = {**controls, "messages": case["long"]}
        require(len(client.tokens(scenario + "-40K-authority", source)) == 40000, "original40K token alignment failed")
        first = client.generate(scenario + "-long", "/v1/chat/completions", source)
        json_ok(first)
        for index, messages in enumerate(case["short"]):
            result = client.generate(scenario + "-short-" + str(index), "/v1/chat/completions",
                                     {**controls, "messages": messages})
            json_ok(result)
        following = {**controls, "messages": case["long"] + [assistant(first),
            {"role": "user", "content": 'Continue this conversation. Reply with {"ok":true}.'}]}
        canonical = {**controls, "messages": case["long"] + [assistant(first)], "add_generation_prompt": False}
        expected = len(client.tokens(scenario + "-completed-authority", canonical))
        result = client.generate(scenario + "-next", "/v1/chat/completions", following)
        json_ok(result)
        client.feature(result["cached"] >= max(40000, expected), "strict40K complete history did not survive shorts")
        records.append({"scenario": scenario, "required_canonical": expected, "cached": result["cached"]})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--arm", required=True, choices=("baseline", "candidate"))
    parser.add_argument("--stage", required=True, choices=("raw", "canonical", "retention", "all"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    spec, cases, requests = load_inputs(args.inputs, args.inputs_sha256)
    evidence = Evidence(args.out)
    evidence.save("inputs.json", spec)
    client = Client(args.base_url, evidence, args.arm)
    result = {"schema": SCHEMA, "arm": args.arm, "input_sha256": args.inputs_sha256,
              "passed": False, "stages": {}, "P3_acceptance": False}
    primary, errors = None, [None, None]
    try:
        client.stats("initial")
        for name, function in (("raw", run_raw), ("canonical", run_canonical), ("retention", run_retention)):
            if args.stage in ("all", name):
                result["stages"][name] = function(client, spec, cases if name != "canonical" else requests)
        result["passed"] = True
    except BaseException as error:
        primary = error
        try:
            result["failure"] = {"type": type(error).__name__, "message": str(error)[:1024]}
        except BaseException as metadata:
            errors[0] = metadata
    try:
        result["requests"] = client.results
        evidence.save("result.json", result)
    except BaseException as publication:
        errors[1] = publication
    finish_errors(primary, errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
