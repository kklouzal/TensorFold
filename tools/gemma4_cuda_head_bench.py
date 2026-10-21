#!/usr/bin/env python3
"""Bounded actual-checkpoint GB10 comparison of exact Gemma BF16 head reuse.

One public engine is constructed per verify width, then closed and released.
The incumbent and R2/4/8/16 candidates share one original BF16 parameter. Real
head inputs are captured only during untimed warmup. A balanced head screen
chooses one candidate to compare with the incumbent through complete serial
and DFlash requests; it does not establish a production winner. Timed requests
all reuse the same warmed prefix and must reproduce complete IDs for the same
proposal geometry and serial/DFlash mode. Every exact head must also match the
incumbent in that mode. Greedy replies retain complete serial identity; sampled
DFlash may differ from serial or between proposal widths.
Each head transition retires the old CUDA graph bank and prepares a new one over
the same persistent state. Graph preparation and full-request warmup are retained
separately from timed requests; only one graph bank is held at a time.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROMPTS = {
    "code": "Write a Python function that merges overlapping intervals. Include type hints and a short example.",
    "chat": "In two sentences, explain why the sky is blue.",
}


def _widths(value: str) -> tuple[int, ...]:
    try:
        widths = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("verify rows must be a comma-separated subset of3,5,11,16") from error
    if not 1 <= len(widths) <= 3 or len(set(widths)) != len(widths) or any(x not in (3, 5, 11, 16) for x in widths):
        raise argparse.ArgumentTypeError("choose1..3 distinct verify widths from3,5,11,16")
    return widths


def _orders(names, pairs):
    """Each pair has the same rotated order and its reverse: balanced positions."""
    for pair in range(pairs):
        at = pair % len(names)
        order = list(names[at:]) + list(names[:at])
        yield order
        yield list(reversed(order))


def _timed_modes(draft_only: bool, index: int) -> tuple[bool, ...]:
    """Choose timed modes; serial warmup/reference capture is always separate."""
    if draft_only:
        return (True,)
    return (False, True) if index % 2 == 0 else (True, False)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _record(path: Path, value: dict) -> None:
    line = json.dumps(value, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


def _run(engine, prompt, tokens, sampling, draft):
    output = []
    engine.drain()
    started = time.perf_counter()
    stats = engine.generate(prompt, tokens, sampling, lambda part: output.extend(part),
                            draft=draft, stop_eos=True)
    engine.drain()
    wall = time.perf_counter() - started
    return output, {**stats, "wall_s": wall, "output_tokens": len(output),
                    "wall_tps": len(output) / wall,
                    "decode_tps": (len(output) - 1) / stats["decode_s"] if stats["decode_s"] else None}


def _capture(engine, prompt, tokens, sampling):
    """Clone eager/head and graph-returned head inputs only during warmup."""
    import torch

    from tensorfold.cleanup import raise_failures

    captured = {}
    graph_calls = 0
    target_call = engine.target.project_logits
    draft_call = engine.head.project_logits
    graph_call = engine.graphs.forward
    hooks = [(owner, name, name in vars(owner), vars(owner).get(name)) for owner, name in
             ((engine.target, "project_logits"), (engine.head, "project_logits"), (engine.graphs, "forward"))]

    def remember(kind, hidden):
        if kind not in captured or hidden.shape[0] > captured[kind].shape[0]:
            captured[kind] = hidden.detach().clone()

    def observer(kind, call):
        def observed(hidden):
            # A first-use graph capture must contain only production kernels.
            # Its returned hidden buffer is observed after capture/replay below.
            if not torch.cuda.is_current_stream_capturing():
                remember(kind, hidden)
            return call(hidden)
        return observed

    def observed_graph(*args, **kwargs):
        nonlocal graph_calls
        result = graph_call(*args, **kwargs)
        remember("target", result.hidden)  # static storage is invalidated by its next replay
        graph_calls += 1
        return result

    primary = None
    try:
        try:
            engine.target.project_logits = observer("target", target_call)
            engine.head.project_logits = observer("dflash", draft_call)
            engine.graphs.forward = observed_graph
            serial, serial_stats = _run(engine, prompt, tokens, sampling, False)
            drafted, draft_stats = _run(engine, prompt, tokens, sampling, True)
        except BaseException as error:  # noqa: BLE001 - retain request/cancellation until hooks and work retire
            primary = error
    finally:
        errors = []
        for owner, name, owned, previous in hooks:
            try:
                if owned:
                    setattr(owner, name, previous)
                elif name in vars(owner):
                    delattr(owner, name)
            except BaseException as error:  # noqa: BLE001 - attempt all restorations and retain every failure
                errors.append(error)
        try:
            engine.drain()
        except BaseException as error:  # noqa: BLE001 - retain drain failure alongside the request primary
            errors.append(error)
        try:
            raise_failures(primary, errors)
        finally:
            primary = errors = None
    if sampling is None and drafted != serial:
        raise RuntimeError("incumbent DFlash warmup changed complete serial token IDs")
    if not graph_calls:
        raise RuntimeError("warmup did not observe graph-returned target head inputs")
    return {"serial": serial, "dflash": drafted}, captured, {
        "serial": serial_stats, "dflash": draft_stats, "target_graph_calls": graph_calls}


def _bind_head(engine, head):
    """Retire the sole old graph bank, then prepare/settle a bank bound to this head."""
    import torch

    from tensorfold.families.gemma4.cuda.graphs import Graphs

    if engine.graphs is None:
        raise RuntimeError("head comparison requires production CUDA graphs")
    if engine.w.embed is head:
        return {"changed": False, "seconds": 0., "prepared_graphs": len(engine.graphs.graphs)}
    started = time.perf_counter()
    engine.drain()  # all consumers and graph outputs have completed before retirement
    span = engine.graphs.span
    engine.graphs = None  # no other bank/pool owner is retained across preparation
    engine.w.embed = head
    with engine._prefix_lock:
        engine._prefix = None  # cached logits belong to the previous head implementation
    # All initialization and serving use this same stream. The final drain
    # settles graph side-stream capture/replay before publishing readiness.
    with torch.cuda.stream(engine._stream), torch.inference_mode():
        engine._state.reset()
        if engine._draft is not None:
            engine._draft.reset()
        engine.graphs = Graphs(engine.target, engine._state, max_rows=engine.max_rows, span=span)
        count = engine.graphs.prepare(taps=(False, True) if engine.head is not None else (False,))
    engine.drain()
    return {"changed": True, "seconds": time.perf_counter() - started, "prepared_graphs": count}


def _warm_request(engine, head, prompt, tokens, sampling, draft, expected):
    """Rebind, warm the full request's grids, and check IDs outside timed evidence."""
    binding = _bind_head(engine, head)
    output, stats = _run(engine, prompt, tokens, sampling, draft)
    if output != expected:
        raise RuntimeError(f"untimed reference changed complete token IDs: expected {expected}, got {output}")
    return {"binding": binding, "warm_ids": output, "warm_stats": stats}


def _references(engine, heads, modes, prompt, tokens, sampling, warm_ids):
    """Capture untimed complete requests for this width/workload/head/mode.

    Every exact head must match the initial incumbent capture in that mode;
    greedy heads additionally match the original serial request. This also
    checks the incumbent capture across cache reuse.
    """
    references = {}
    for draft in modes:
        mode = "dflash" if draft else "serial"
        for name, head in heads.items():
            expected = warm_ids[mode] if sampling is not None else warm_ids["serial"]
            preparation = _warm_request(engine, head, prompt, tokens, sampling, draft, expected)
            output, stats = _run(engine, prompt, tokens, sampling, draft)
            if output != expected:
                raise RuntimeError(f"{name}/{mode} untimed reference changed complete token IDs: "
                                   f"expected {expected}, got {output}")
            if stats["cached"] != len(prompt):
                raise RuntimeError("untimed reference did not use the declared complete warm prefix")
            references[draft, name] = output, stats, preparation
    return references


def _screen(heads, inputs, pairs, loops):
    import torch
    samples = {}
    with torch.inference_mode():
        reference = [heads["reference"](hidden) for hidden in inputs.values()]
        for name, head in heads.items():
            for hidden, expected in zip(inputs.values(), reference):
                if not torch.equal(head(hidden).view(torch.int16), expected.view(torch.int16)):
                    raise RuntimeError(f"{name} changed actual head-input bits")
        raw = []
        for order_index, order in enumerate(_orders(tuple(heads), pairs)):
            for name in order:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(loops):
                    for hidden in inputs.values():
                        output = heads[name](hidden)
                end.record()
                end.synchronize()
                elapsed = begin.elapsed_time(end) / loops
                # Eager native kernels cannot be elided; consume an output only
                # after the recorded timing interval, without extra timed work.
                output.reshape(-1)[0].item()
                samples.setdefault(name, []).append(elapsed)
                raw.append({"order_index": order_index, "order": order, "head": name,
                            "target_plus_draft_head_ms": elapsed})
    medians = {name: statistics.median(times) for name, times in samples.items()}
    candidate = min((name for name in medians if name != "reference"), key=lambda name: medians[name])
    return candidate, {"rows": {kind: list(hidden.shape) for kind, hidden in inputs.items()},
                       "pairs": pairs, "loops": loops, "raw": raw, "median_ms": medians,
                       "screen_choice": candidate, "selection_status": "screen only; integrated comparison required"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--drafter", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True, help="exact installed/test image or derived-source artifact ID")
    parser.add_argument("--out", type=Path, required=True, help="new JSONL evidence file; existing files are refused")
    parser.add_argument("--rows", type=_widths, default=(3, 5, 11), help="verify widths, default3,5,11; at most3 entries")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--code-tokens", type=int, default=128)
    parser.add_argument("--chat-tokens", type=int, default=32)
    parser.add_argument("--head-pairs", type=int, default=2)
    parser.add_argument("--head-loops", type=int, default=2)
    parser.add_argument("--request-pairs", type=int, default=1)
    parser.add_argument("--draft-only", action="store_true",
                        help="skip timed serial comparisons; retain serial warmup, same-config references and bit checks")
    parser.add_argument("--all-heads", action="store_true",
                        help="integrate every exact reuse candidate instead of only the head-screen choice")
    parser.add_argument("--sampled", action="store_true",
                        help="keyed temperature1/top-k64/top-p.95; check same-config replay, not serial/draft identity")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args(argv)
    if not 2 <= args.code_tokens <= 128 or not 2 <= args.chat_tokens <= 128:
        parser.error("bounded replies require2..128 tokens")
    if min(args.context, args.head_pairs, args.head_loops, args.request_pairs) < 1:
        parser.error("context and repeat counts must be positive")
    if max(args.head_pairs, args.head_loops, args.request_pairs) > 10:
        parser.error("bounded runs allow at most10 repeat pairs/loops")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8"):
        pass

    import torch
    import triton

    from tensorfold.cleanup import raise_failures, rollback
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
    from tensorfold.families.gemma4.cuda.head import prepare_head
    from tensorfold.token_metadata import load_tokenizer

    root = Path(__file__).resolve().parents[1]
    source_names = ("tools/gemma4_cuda_head_bench.py", "src/tensorfold/families/gemma4/cuda/head.py",
                    "src/tensorfold/families/gemma4/cuda/forward.py", "src/tensorfold/families/gemma4/cuda/engine.py",
                    "src/tensorfold/families/gemma4/cuda/graphs.py", "src/tensorfold/families/gemma4/cuda/state.py",
                    "src/tensorfold/families/qwen3_5/cuda/b16.py", "src/tensorfold/families/qwen3_5/cuda/b16.cu")
    provenance = {"kind": "provenance", "artifact_id": args.artifact_id,
                  "model": str(args.model.resolve()), "drafter": str(args.drafter.resolve()),
                  "model_config_sha256": _sha(args.model / "config.json"),
                  "drafter_config_sha256": _sha(args.drafter / "config.json"),
                  "tokenizer_sha256": _sha(args.model / "tokenizer.json"),
                  "source_sha256": {name: _sha(root / name) for name in source_names},
                  "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
                  "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                  "command": sys.argv, "verify_rows": args.rows, "thinking": args.thinking,
                  "sampling": "keyed sampled" if args.sampled else "greedy",
                  "seed": args.seed if args.sampled else None,
                  "token_contract": "same seed/width/mode/cache replay across exact heads; serial also across widths; "
                                    "greedy DFlash equals serial",
                  "stop_eos": True, "reply_limits": {"code": args.code_tokens, "chat": args.chat_tokens},
                  "integrated_heads": "all" if args.all_heads else "head-screen choice plus reference",
                  "timed_modes": "DFlash only" if args.draft_only else "serial and DFlash",
                  "serial_reference_warmup": True,
                  "cache_state": "each timed request uses the same warmed complete prompt prefix",
                  "graph_state": "one head-bound bank; rebuild and full-request warmup outside timing on transitions",
                  "precision": "original calibrated native NVFP4; original BF16 tied head; BF16 KV",
                  "objective": "paired complete cached-request wall time; decode time reported separately",
                  "inference_scope": "bounded screen and paired observations; no confidence/equivalence claim"}
    if (root / ".git").exists():
        provenance["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    _record(args.out, provenance)
    tokenizer = load_tokenizer(args.model)
    sampling = Sampling(args.seed, temperature=1., top_k=64, top_p=.95) if args.sampled else None
    across_depths = {}
    for depth_index, rows in enumerate(args.rows):
        engine = None
        try:
            started = time.perf_counter()
            engine = Gemma4Engine(args.model, drafter=str(args.drafter), drafter_bits=16,
                                  depth=rows - 1, context=args.context, context_explicit=True, streams=1)
            engine.drain()
            if engine.w.numerics != "ct_nvfp4":
                raise ValueError("this bounded head comparison requires the original Gemma CT NVFP4 checkpoint")
            if engine.chat_template is None:
                raise ValueError("actual checkpoint chat metadata is required")
            _record(args.out, {"kind": "construction", "verify_rows": rows,
                               "seconds": time.perf_counter() - started,
                               "context": engine.context_window, "weight_bytes": engine.w.nbytes})
            # This GB10/actual-shape tool names the original authority after
            # the production CT loader has selected its prepared exact head.
            incumbent = engine.w.embed.incumbent
            heads = {"reference": incumbent, **{f"reuse{r}": prepare_head(incumbent, reuse_rows=r)
                                                for r in (2, 4, 8, 16)}}
            for workload_index, (workload, text) in enumerate(PROMPTS.items()):
                rendered = engine.chat_template.render([{"role": "user", "content": text}],
                                                       tools=None, enable_thinking=args.thinking)
                prompt = tokenizer.encode(rendered, add_special_tokens=False).ids
                limit = args.code_tokens if workload == "code" else args.chat_tokens
                if len(prompt) + limit > engine.context_window:
                    raise ValueError("rendered prompt plus bounded reply exceeds admitted context")
                binding = _bind_head(engine, incumbent)
                warm_ids, inputs, warm = _capture(engine, prompt, limit, sampling)
                if workload in across_depths and warm_ids["serial"] != across_depths[workload]:
                    raise RuntimeError("incumbent serial token IDs changed across prepared verify widths")
                across_depths[workload] = warm_ids["serial"]
                _record(args.out, {"kind": "warmup", "verify_rows": rows, "workload": workload,
                                   "prompt_ids": prompt, "reference_ids": warm_ids["serial"],
                                   "dflash_reference_ids": warm_ids["dflash"], "binding": binding, "stats": warm})
                selected, screen = _screen(heads, inputs, args.head_pairs, args.head_loops)
                del inputs
                _record(args.out, {"kind": "head_screen", "verify_rows": rows, "workload": workload, **screen})
                modes = _timed_modes(args.draft_only, depth_index + workload_index)
                compared_heads = tuple(heads) if args.all_heads else ("reference", selected)
                references = _references(engine, {name: heads[name] for name in compared_heads},
                                         modes, prompt, limit, sampling, warm_ids)
                for (draft, name), (output, stats, preparation) in references.items():
                    _record(args.out, {"kind": "request_reference", "verify_rows": rows, "workload": workload,
                                       "head": name, "draft": draft, "reference_ids": output, "stats": stats,
                                       "preparation": preparation})
                for draft in modes:
                    for order_index, order in enumerate(_orders(compared_heads, args.request_pairs)):
                        for name in order:
                            reference, _, _ = references[draft, name]
                            preparation = _warm_request(engine, heads[name], prompt, limit, sampling, draft, reference)
                            _record(args.out, {"kind": "request_preparation", "verify_rows": rows, "workload": workload,
                                               "head": name, "draft": draft, "order_index": order_index, **preparation})
                            output, stats = _run(engine, prompt, limit, sampling, draft)
                            equal = output == reference
                            record = {"kind": "integrated", "verify_rows": rows, "workload": workload,
                                      "head": name, "draft": draft, "order_index": order_index, "order": order,
                                      "reference_ids": reference,
                                      "same_reference_ids": equal, "output_ids": output, **stats}
                            _record(args.out, record)
                            if not equal:
                                raise RuntimeError("candidate complete request changed reference token IDs")
                            if stats["cached"] != len(prompt):
                                raise RuntimeError("timed request did not use the declared complete warm prefix")
        except BaseException as primary:
            try:
                _record(args.out, {"kind": "failure", "verify_rows": rows,
                                   "error_type": type(primary).__name__, "error": str(primary)})
            except BaseException as diagnostic:  # noqa: BLE001 - failed evidence publication must retain the primary
                try:
                    raise_failures(primary, [diagnostic])
                except BaseException as retained:  # noqa: BLE001 - transport all native/cancellation/cleanup failures
                    primary = retained
            if engine is not None:
                rollback(engine, primary, engine.close)
            raise
        else:
            engine.close()
        finally:
            # Release every local reference to the single original authority
            # before the next engine is constructed; close alone only drains.
            engine = None
            if "heads" in locals():
                del heads
            if "incumbent" in locals():
                del incumbent
            gc.collect()
            torch.cuda.empty_cache()
    _record(args.out, {"kind": "complete", "verify_rows": args.rows,
                       "status": "bounded comparisons complete; root must assess integrated selection"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
