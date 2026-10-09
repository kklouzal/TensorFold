"""Cold prefill against any OpenAI server: chat prompts of fixed token lengths from the Python standard library,
a unique first line each so no cached prefix resumes; TTFT and prompt tok/s per length.

build (where tensorfold is installed): python3 tools/prefill_cold.py build MODEL_DIR PROMPTS.json
run (any client):                      python3 tools/prefill_cold.py run URL MODEL PROMPTS.json OUT.json
Response budgets: --response-mib defaults to16MiB, --sse-line-kib to64KiB.
Raise them for larger valid replies or servers that batch a response into one
SSE line. Event storage shares the response budget. Limits concern client
measurement storage and never alter generation parameters. Inactivity timeouts
remain unchanged; these readers do not claim a whole-request wall deadline.
"""

import argparse
import json
import statistics
import time
import urllib.request

if __package__:
    from . import openai_protocol as protocol
    from .benchmark_output import write_json
else:
    import openai_protocol as protocol
    from benchmark_output import write_json

LENGTHS = (2048, 8192, 16384, 32768, 65536)
REPS = 3
ASK = "\nSay in one sentence what the code above does."


def corpus() -> str:
    import sysconfig
    from pathlib import Path

    root = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in ("test", "tests", "idlelib",
                                                                                     "site-packages", "__pycache__")))
    return "".join(f"# {p.relative_to(root)}\n{p.read_text(errors='ignore')}\n" for p in files)


def build(model_dir: str, out: str) -> None:
    from pathlib import Path

    from tokenizers import Tokenizer

    from tensorfold.cuda.server import ChatTemplate

    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    template = ChatTemplate(Path(model_dir))
    text = corpus()

    def messages(nonce: str, start: int, chars: int):
        return [{"role": "user", "content": f"Request {nonce}.\n" + text[start:start + chars] + ASK}]

    def count(m) -> int:
        return len(tok.encode(template.render(m, tools=None, enable_thinking=False), add_special_tokens=False).ids)

    items = []
    for length in (1024,) + LENGTHS:
        for rep in range(1 if length == 1024 else REPS):
            nonce = f"{length}-{rep}" if length != 1024 else "warm"
            start = (rep * 1_000_003 + length * 7) % (len(text) - 20 * length)
            lo, hi = 0, 8 * length
            while lo < hi:                               # the most characters that stay within the length
                mid = (lo + hi + 1) // 2
                if count(messages(nonce, start, mid)) <= length:
                    lo = mid
                else:
                    hi = mid - 1
            m = messages(nonce, start, lo)
            items.append({"length": length, "rep": rep, "tokens": count(m), "messages": m})
            print(json.dumps({"length": length, "rep": rep, "tokens": items[-1]["tokens"]}), flush=True)
    write_json(out, {"items": items})


def one(url: str, model: str, m, *, limits=None) -> dict:
    body = {"model": model, "messages": m, "max_tokens": 2, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    sent, first, usage = time.perf_counter(), None, {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        protocol.response_type(resp, 'text/event-stream')
        for chunk in protocol.sse_objects(resp, limits=limits):
            new_usage, _runtime, pieces = protocol.chunk_fields(chunk)
            if first is None and any(pieces):
                first = time.perf_counter()
            usage = new_usage if new_usage is not None else usage
    protocol.completion_usage(usage, 2, prompt=True)
    if first is None:
        raise ValueError('prefill response emitted no text or reasoning')
    return {"ttft_s": round(first - sent, 4), "prompt_tokens": usage['prompt_tokens']}


def run(url: str, model: str, prompts: str, out: str, *, limits=None) -> None:
    limits = limits or protocol.Limits()
    with open(prompts, encoding="utf-8") as stream:
        items = json.load(stream)["items"]
    rows = []
    for it in items:
        r = one(url, model, it["messages"], limits=limits)
        r.update(length=it["length"], rep=it["rep"])
        rows.append(r)
        print(json.dumps(r), flush=True)
    summary = []
    for length in LENGTHS:
        rs = [r for r in rows if r["length"] == length]
        t = statistics.median(r["ttft_s"] for r in rs)
        n = statistics.median(r["prompt_tokens"] or 0 for r in rs)
        summary.append({"length": length, "prompt_tokens": n, "ttft_s": round(t, 3), "tok_s": round(n / t, 1),
                        "ttft_all": [r["ttft_s"] for r in rs]})
        print(json.dumps(summary[-1]), flush=True)
    write_json(out, {"model": model, "rows": rows, "summary": summary,
                     "response_budget_bytes": limits.response_bytes, "sse_line_budget_bytes": limits.line_bytes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('build')
    prepare.add_argument('model_dir')
    prepare.add_argument('out')
    execute = commands.add_parser('run')
    for name in ('url', 'model', 'prompts', 'out'):
        execute.add_argument(name)
    protocol.add_arguments(execute)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.model_dir, args.out)
    else:
        run(args.url, args.model, args.prompts, args.out, limits=protocol.from_arguments(args))


if __name__ == "__main__":
    main()
