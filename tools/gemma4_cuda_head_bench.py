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

The explicit --pair-r8-r16 mode instead holds the installed private FP8
drafter fixed, consumes pinned native-thinking64K IDs and compares existing
R16 against R8 in mirrored blocks. It preserves only that exact pair's shared
detached prefix, screens target inputs only, and records transition/memory
costs separately. Fixed256 ignore-EOS evidence does not qualify normal-EOS
throughput or general output quality.
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
PAIR_INPUT_SHA = "fb9aafe715f75ad7b43046ddc59567d025df881c0bcd4119ceba63851e6f2d63"
PAIR_MODEL_REV = "90c8601b1a439497bc15d8fba9e52990b45ea0cb"
PAIR_TEMPLATE_SHA = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
PAIR_TOKENIZER_SHA = "9d0e1c859267d7cded7ad6c23bbb72a838536716fa4c37a3f5671b7918f59de4"


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


def _run(engine, prompt, tokens, sampling, draft, *, stop_eos=True):
    output = []
    engine.drain()
    started = time.perf_counter()
    stats = engine.generate(prompt, tokens, sampling, lambda part: output.extend(part),
                            draft=draft, stop_eos=stop_eos)
    engine.drain()
    wall = time.perf_counter() - started
    return output, {**stats, "wall_s": wall, "output_tokens": len(output),
                    "wall_tps": len(output) / wall,
                    "decode_tps": (len(output) - 1) / stats["decode_s"] if stats["decode_s"] else None}


def _capture(engine, prompt, tokens, sampling, *, stop_eos=True, terminal_on_failure=False):
    """Clone eager/head and graph-returned head inputs only during warmup."""
    import torch

    from tensorfold.cleanup import raise_failures

    captured = {"target": None, "dflash": None}
    pending = None
    graph_calls = 0
    target_call = engine.target.project_logits
    draft_call = engine.head.project_logits
    graph_call = engine.graphs.forward
    hooks = [(owner, name, name in vars(owner), vars(owner).get(name)) for owner, name in
             ((engine.target, "project_logits"), (engine.head, "project_logits"), (engine.graphs, "forward"))]
    restoration_errors = [None] * (len(hooks) + 1)

    def remember(kind, hidden):
        nonlocal pending
        if captured[kind] is None or hidden.shape[0] > captured[kind].shape[0]:
            # The named cell owns a returned clone even if dictionary
            # publication fails. Slots and restoration statuses already exist
            # before any observed GPU work; failed transport retains this frame.
            pending = hidden.detach().clone()
            captured[kind] = pending
            pending = None

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
    native_cause = native_context = None
    try:
        try:
            engine.target.project_logits = observer("target", target_call)
            engine.head.project_logits = observer("dflash", draft_call)
            engine.graphs.forward = observed_graph
            options = {} if stop_eos else {"stop_eos": False}
            serial, serial_stats = _run(engine, prompt, tokens, sampling, False, **options)
            drafted, draft_stats = _run(engine, prompt, tokens, sampling, True, **options)
        except BaseException as error:  # noqa: BLE001 - retain request/cancellation until hooks and work retire
            primary = error
            native_cause = BaseException.__cause__.__get__(primary)
            native_context = BaseException.__context__.__get__(primary)
    finally:
        try:
            for index, (owner, name, owned, previous) in enumerate(hooks):
                try:
                    if owned:
                        setattr(owner, name, previous)
                    elif name in vars(owner):
                        delattr(owner, name)
                except BaseException as error:  # noqa: BLE001 - retain every independent restoration failure
                    restoration_errors[index] = error
                    if primary is None:
                        primary = error
                        native_cause = BaseException.__cause__.__get__(primary)
                        native_context = BaseException.__context__.__get__(primary)
            if primary is None or not terminal_on_failure:
                try:
                    engine.drain()
                except BaseException as error:  # noqa: BLE001 - retain drain failure alongside the request primary
                    restoration_errors[-1] = error
                    if primary is None:
                        primary = error
                        native_cause = BaseException.__cause__.__get__(primary)
                        native_context = BaseException.__context__.__get__(primary)
            errors = [error for error in restoration_errors if error is not None and error is not primary]
            raise_failures(primary, errors)
        except BaseException as transport:
            # Covers status preparation and foreign transport callee ENTRY.
            # Native roots, pending clone, snapshots and actual engine remain
            # frame-owned; unknown completion never triggers a CUDA retry.
            if primary is not None and transport is not primary:
                if native_cause is primary:
                    native_cause = None
                if native_context is primary:
                    native_context = None
                raise primary from transport
            raise
    if sampling is None and drafted != serial:
        raise RuntimeError("incumbent DFlash warmup changed complete serial token IDs")
    if not graph_calls:
        raise RuntimeError("warmup did not observe graph-returned target head inputs")
    return {"serial": serial, "dflash": drafted}, {kind: value for kind, value in captured.items()
                                                   if value is not None}, {
        "serial": serial_stats, "dflash": draft_stats, "target_graph_calls": graph_calls}


def _bind_head(engine, head, *, preserve_exact_prefix=False):
    """Retire the sole old graph bank, then prepare/settle a bank bound to this head."""
    import torch

    from tensorfold.families.gemma4.cuda.graphs import Graphs

    if engine.graphs is None:
        raise RuntimeError("head comparison requires production CUDA graphs")
    if preserve_exact_prefix:
        _check_exact_pair(engine.w.embed, head)
    if engine.w.embed is head:
        return {"changed": False, "seconds": 0., "prepared_graphs": len(engine.graphs.graphs)}
    started = time.perf_counter()
    if preserve_exact_prefix:
        engine.drain()
    else:
        engine.clear_prefix_cache()  # settles consumers before invalidating retained state
    span = engine.graphs.span
    engine.graphs = None  # no other bank/pool owner is retained across preparation
    engine.w.embed = head
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


def _check_exact_pair(previous, replacement):
    """Only original R8/R16 wrappers sharing the same BF16 authority qualify.

    Prompt forwards have output_logits=False; the final next-token projection
    is singleton, and both wrappers call the same incumbent for that singleton.
    Embedding rows, target KV/taps and private FP8 DFlash are unchanged. Prefix
    snapshots own detached target/draft storage and cloned next-token logits;
    resetting/capturing active State cannot mutate them. Ordinary generate then
    restores the very same snapshot, without a benchmark-only copy or re-prime.
    """
    from tensorfold.families.gemma4.cuda.head import _RowsHead

    if (type(previous) is not _RowsHead or type(replacement) is not _RowsHead
            or previous.reuse_rows not in (8, 16) or replacement.reuse_rows not in (8, 16)
            or previous.incumbent is not replacement.incumbent
            or previous.weight is not replacement.weight):
        raise ValueError("prefix preservation requires the exact same-authority R8/R16 pair")


def _pair_inputs(args):
    """Bind the two frozen native-thinking 64K fixtures before engine creation."""
    with args.input_manifest.open("rb") as stream:
        payload = stream.read(2 * 1024 * 1024 + 1)
    if len(payload) > 2 * 1024 * 1024:
        raise ValueError("64K input manifest exceeds2MiB")
    if args.input_manifest_sha256 != PAIR_INPUT_SHA or hashlib.sha256(payload).hexdigest() != PAIR_INPUT_SHA:
        raise ValueError("64K input manifest SHA256 differs from the qualified fixture")
    data = json.loads(payload)
    if (data["schema"] != 1 or data["thinking"] is not True
            or data["actual_prompt_tokens"] != 65536 or data["context_window"] != 131072
            or data["model_revision"] != PAIR_MODEL_REV or data["template_sha256"] != PAIR_TEMPLATE_SHA
            or data["tokenizer_sha256"] != PAIR_TOKENIZER_SHA
            or _sha(args.model / "tokenizer.json") != PAIR_TOKENIZER_SHA
            or _sha(args.chat_template) != PAIR_TEMPLATE_SHA
            or [item["name"] for item in data["cases"]] != ["code64k", "prose64k"]):
        raise ValueError("64K model/tokenizer/template/thinking metadata differs")
    for item in data["cases"]:
        ids = item["ids"]
        digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        if (type(ids) is not list or len(ids) != 65536 or item["prompt_tokens"] != 65536
                or any(type(token) is not int or not 0 <= token < 262144 for token in ids)
                or digest != item["ids_sha256"]):
            raise ValueError("64K native prompt IDs differ")
    return data["cases"]


def _proposal_identity(engine):
    """Read-only identity of the private FP8 head, never a target-head borrower."""
    from tensorfold.cuda.nvfp4.linear import Fp8Linear
    from tensorfold.cuda.requant import Head

    head = engine.head.project_logits
    if engine.head.bits != 8 or type(head) is not Head or type(head.linear) is not Fp8Linear:
        raise ValueError("pair comparison requires the unchanged private FP8 DFlash head")
    linear = head.linear
    fields = {name: None if getattr(linear, name) is None else {
        "object": id(getattr(linear, name)), "data_ptr": getattr(linear, name).data_ptr(),
        "shape": list(getattr(linear, name).shape), "dtype": str(getattr(linear, name).dtype)}
        for name in ("w8", "cols", "groups")}
    return {"head_object": id(head), "linear_object": id(linear), "fields": fields,
            "n": linear.n, "k": linear.k, "softcap": head.softcap, "nbytes": head.nbytes()}


def _memory():
    import torch

    return {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_scope": "process cumulative; includes model construction, screens, transitions and warmups"}


def _check_pair_request(output, stats, expected, reference_stats, prompt_size, tokens):
    if output != expected or len(output) != tokens:
        raise RuntimeError("R8/R16 complete fixed-budget output IDs differ")
    if stats["cached"] != prompt_size:
        raise RuntimeError("R8/R16 request did not restore the same complete warm prefix")
    for name in ("drafts", "rounds", "drafted", "accepted", "min_rows"):
        if stats[name] != reference_stats[name]:
            raise RuntimeError(f"R8/R16 proposal accounting differs: {name}")


def _pair_requests(engine, heads, items, args, sampling):
    """Six or more mirrored paired blocks; graph transitions are never timed requests."""
    proposal = _proposal_identity(engine)
    for item in items:
        prompt = item["ids"]
        limit = args.code_tokens if item["name"] == "code64k" else args.chat_tokens
        initial_binding = _bind_head(engine, heads["reference"], preserve_exact_prefix=True)
        expected, inputs, warm = _capture(engine, prompt, limit, sampling, stop_eos=False,
                                          terminal_on_failure=True)
        prefix = engine.prefix_cache_identity(prompt)
        if (prefix is None or prefix.position != len(prompt)
                or warm["dflash"]["cached"] != len(prompt)):
            raise RuntimeError("initial proof did not create the declared complete prefix")
        if _proposal_identity(engine) != proposal:
            raise RuntimeError("private FP8 proposal authority changed during warmup")
        _record(args.out, {"kind": "pair_warmup", "workload": item["name"], "prompt_ids": prompt,
                           "ids_sha256": item["ids_sha256"], "reference_ids": expected,
                           "stats": warm, "initial_binding": initial_binding,
                           "prefix_identity": {"position": prefix.position, "backing": prefix.backing}, "proposal_identity": proposal, "memory": _memory()})
        # FP8 DFlash does not execute the target BF16 head on its own hidden
        # rows. Screen only the actual target inputs, separately for each tail.
        hidden = inputs["target"]
        for rows in range(1, min(5, hidden.shape[0]) + 1):
            _, screen = _screen(heads, {"target": hidden[:rows]}, args.head_pairs, args.head_loops)
            _record(args.out, {"kind": "pair_target_head_screen", "workload": item["name"],
                               "rows": rows, **screen,
                               "selection_status": "fixed R16/R8 pair; screen does not select a winner"})
        del inputs, hidden
        for order_index, order in enumerate(_orders(tuple(heads), args.request_pairs)):
            for name in order:
                started = time.perf_counter()
                binding = _bind_head(engine, heads[name], preserve_exact_prefix=True)
                if engine.prefix_cache_identity(prompt) != prefix or _proposal_identity(engine) != proposal:
                    raise RuntimeError("head transition replaced the prefix or private FP8 proposal head")
                output, preparation = _run(engine, prompt, limit, sampling, True, stop_eos=False)
                _check_pair_request(output, preparation, expected["dflash"], warm["dflash"], len(prompt), limit)
                _record(args.out, {"kind": "pair_preparation", "workload": item["name"], "head": name,
                                   "block": order_index // 2, "order_index": order_index, "order": order,
                                   "binding": binding, "warm_stats": preparation, "warm_ids": output,
                                   "seconds": time.perf_counter() - started, "prefix_identity": {"position": prefix.position, "backing": prefix.backing},
                                   "proposal_identity": proposal, "memory": _memory()})
                output, stats = _run(engine, prompt, limit, sampling, True, stop_eos=False)
                _check_pair_request(output, stats, expected["dflash"], warm["dflash"], len(prompt), limit)
                _record(args.out, {"kind": "pair_integrated", "workload": item["name"], "head": name,
                                   "block": order_index // 2, "order_index": order_index, "order": order,
                                   "reference_ids": expected["dflash"], "output_ids": output,
                                   "same_reference_ids": True, **stats, "memory": _memory()})
        if engine.prefix_cache_identity(prompt) != prefix or _proposal_identity(engine) != proposal:
            raise RuntimeError("integrated requests changed authoritative prefix or FP8 proposal identity")
        # The fingerprint retains no cache payload; the next workload may evict
        # its backing owners. Each comparison above queries current residency.
        del prefix


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
                            "target_head_ms" if tuple(inputs) == ("target",) else
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
    parser.add_argument("--drafter-bits", type=int, choices=(8, 16), default=16)
    parser.add_argument("--chat-template", type=Path, help="explicit checkpoint-template override")
    parser.add_argument("--artifact-id", required=True, help="exact installed/test image or derived-source artifact ID")
    parser.add_argument("--out", type=Path, required=True, help="new JSONL evidence file; existing files are refused")
    parser.add_argument("--rows", type=_widths, default=(3, 5, 11), help="verify widths, default3,5,11; at most3 entries")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--code-tokens", type=int, default=128)
    parser.add_argument("--chat-tokens", type=int, default=32)
    parser.add_argument("--head-pairs", type=int, default=2)
    parser.add_argument("--head-loops", type=int, default=2)
    parser.add_argument("--request-pairs", type=int, default=None,
                        help="mirrored request blocks: default1 legacy,6 for the fixed64K pair; at most10")
    parser.add_argument("--pair-r8-r16", action="store_true",
                        help="fixed64K FP8/depth4 R16-incumbent versus R8; exact shared-prefix restoration")
    parser.add_argument("--input-manifest", type=Path, help="frozen native-thinking64K token-ID fixture")
    parser.add_argument("--input-manifest-sha256", help="required exact manifest identity for64K pair mode")
    parser.add_argument("--draft-only", action="store_true",
                        help="skip timed serial comparisons; retain serial warmup, same-config references and bit checks")
    parser.add_argument("--all-heads", action="store_true",
                        help="integrate every exact reuse candidate instead of only the head-screen choice")
    parser.add_argument("--sampled", action="store_true",
                        help="keyed temperature1/top-k64/top-p.95; check same-config replay, not serial/draft identity")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args(argv)
    if args.request_pairs is None:
        args.request_pairs = 6 if args.pair_r8_r16 else 1
    maximum = 256 if args.pair_r8_r16 else 128
    if not 2 <= args.code_tokens <= maximum or not 2 <= args.chat_tokens <= maximum:
        parser.error(f"bounded replies require2..{maximum} tokens")
    if min(args.context, args.head_pairs, args.head_loops, args.request_pairs) < 1:
        parser.error("context and repeat counts must be positive")
    if max(args.head_pairs, args.head_loops, args.request_pairs) > 10:
        parser.error("bounded runs allow at most10 repeat pairs/loops")
    items = None
    if args.pair_r8_r16:
        if (args.rows != (5,) or args.context != 131072 or args.drafter_bits != 8
                or args.chat_template is None or args.input_manifest is None
                or args.input_manifest_sha256 is None or not args.thinking
                or args.code_tokens != 256 or args.chat_tokens != 256 or args.request_pairs < 6):
            parser.error("64K pair requires rows5/context131072/bits8/explicit template and manifest SHA/"
                         "thinking/code256/chat256/at least6 paired blocks")
        items = _pair_inputs(args)
    elif args.input_manifest is not None or args.input_manifest_sha256 is not None or args.drafter_bits != 16:
        parser.error("64K fixture and private FP8 options require --pair-r8-r16")
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
                  "drafter_bits": args.drafter_bits,
                  "template_override": None if args.chat_template is None else {
                      "file": str(args.chat_template.resolve()), "sha256": _sha(args.chat_template)},
                  "sampling": "keyed sampled" if args.sampled else "greedy",
                  "seed": args.seed if args.sampled else None,
                  "token_contract": "same seed/width/mode/cache replay across exact heads; serial also across widths; "
                                    "greedy DFlash equals serial",
                  "stop_eos": not args.pair_r8_r16,
                  "reply_limits": {"code": args.code_tokens, "chat": args.chat_tokens},
                  "integrated_heads": ("R16 reference and R8" if args.pair_r8_r16 else
                                       "all" if args.all_heads else "head-screen choice plus reference"),
                  "timed_modes": "DFlash only" if args.draft_only or args.pair_r8_r16 else "serial and DFlash",
                  "serial_reference_warmup": True,
                  "cache_state": "each timed request uses the same warmed complete prompt prefix",
                  "graph_state": "one head-bound bank; rebuild and full-request warmup outside timing on transitions",
                  "precision": "original calibrated native NVFP4; original BF16 tied head; BF16 KV",
                  "objective": ("paired fixed255-step decode_s, then complete cached-request wall_s, memory, cache"
                                if args.pair_r8_r16 else
                                "paired complete cached-request wall time; decode time reported separately"),
                  "inference_scope": "bounded screen and paired observations; no confidence/equivalence claim"}
    if (root / ".git").exists():
        provenance["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    _record(args.out, provenance)
    if args.pair_r8_r16:
        provenance_pair = {"kind": "pair_protocol", "input_manifest_sha256": args.input_manifest_sha256,
                           "reference": "existing R16", "candidate": "R8", "verify_rows": 5,
                           "blocks": args.request_pairs, "orders": list(_orders(("reference", "reuse8"),
                                                                                args.request_pairs)),
                           "stop_eos": False, "tokens": 256, "vision_loaded": True,
                           "cache": "same detached original prefix object across same-authority R8/R16 transitions",
                           "transition": "drain, retire graphs, capture new graphs; untimed warm request restores snapshot",
                           "proposal": "same private FP8 head; target-only BF16 head screen",
                           "objective": ["paired fixed255-step decode_s", "paired complete cached-request wall_s",
                                         "allocated/reserved/peak device bytes", "complete65536 prefix hit"],
                           "scope": "synthetic fixed-budget diagnostic; normal-EOS quality/latency qualification separate"}
        _record(args.out, provenance_pair)
    tokenizer = load_tokenizer(args.model)
    sampling = Sampling(args.seed, temperature=1., top_k=64, top_p=.95) if args.sampled else None
    across_depths = {}
    for depth_index, rows in enumerate(args.rows):
        engine = None
        pair_failed = False
        try:
            started = time.perf_counter()
            options = {} if args.chat_template is None else {"chat_template_file": args.chat_template}
            if args.pair_r8_r16:
                options["vision"] = True
            engine = Gemma4Engine(args.model, drafter=str(args.drafter), drafter_bits=args.drafter_bits,
                                  depth=rows - 1, context=args.context, context_explicit=True, streams=1, **options)
            engine.drain()
            if engine.w.numerics != "ct_nvfp4":
                raise ValueError("this bounded head comparison requires the original Gemma CT NVFP4 checkpoint")
            if engine.chat_template is None:
                raise ValueError("actual checkpoint chat metadata is required")
            _record(args.out, {"kind": "construction", "verify_rows": rows,
                               "seconds": time.perf_counter() - started,
                               "context": engine.context_window, "weight_bytes": engine.w.nbytes})
            if args.pair_r8_r16:
                if engine.chat_template.provenance()["sha256"] != PAIR_TEMPLATE_SHA:
                    raise ValueError("engine did not select the exact explicit Google template")
                reference = engine.w.embed
                if getattr(reference, "reuse_rows", None) != 16:
                    raise ValueError("pair reference must be the actual selected R16 head")
                heads = {"reference": reference, "reuse8": prepare_head(reference.incumbent, reuse_rows=8)}
                _check_exact_pair(heads["reference"], heads["reuse8"])
                _pair_requests(engine, heads, items, args, sampling)
            if not args.pair_r8_r16:
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
            pair_failed = args.pair_r8_r16
            try:
                _record(args.out, {"kind": "failure", "verify_rows": rows,
                                   "error_type": type(primary).__name__, "error": str(primary)})
            except BaseException as diagnostic:  # noqa: BLE001 - failed evidence publication must retain the primary
                try:
                    raise_failures(primary, [diagnostic])
                except BaseException as retained:  # noqa: BLE001 - transport all native/cancellation/cleanup failures
                    primary = retained
            if engine is not None and not pair_failed:
                rollback(engine, primary, engine.close)
            raise
        else:
            try:
                engine.close()
            except BaseException:
                pair_failed = args.pair_r8_r16
                raise
        finally:
            # Release every local reference to the single original authority
            # before the next engine is constructed; close alone only drains.
            if not pair_failed:
                engine = None
                if "heads" in locals():
                    del heads
                if "incumbent" in locals():
                    del incumbent
                if "reference" in locals():
                    del reference
                gc.collect()
                torch.cuda.empty_cache()
            # A pair failure is terminal: the actual exception's main/callee
            # frames retain engine, graphs and pending buffers. No CUDA drain,
            # release or empty_cache retry follows unknown completion.
    _record(args.out, {"kind": "complete", "verify_rows": args.rows,
                       "status": "bounded comparisons complete; root must assess integrated selection"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
