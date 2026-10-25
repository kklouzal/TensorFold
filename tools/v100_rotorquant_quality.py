#!/usr/bin/env python3
"""Compare one installed SM70 target with the frozen 714-step INT8 authority.

This is a quality run, not a throughput benchmark. It uses the ordinary
optimized target, with no MTP, no graphs and no prompt retokenization. A
successful native-codec oracle is separate from this cross-format quality gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import wraps
import hashlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import traceback
from types import MappingProxyType

COUNTS = {"baseline325": 325, "extended389": 389}
GATES = {"mean_kl_max": 1e-4, "ppl_ratio_max": 1.001, "top1_agreement_min": 0.995}
CONTEXT, PROMPT_ROWS, STEPS, CASES = 8192, 2048, 714, 21
REFERENCE_OPERATOR = "originalTorch/referencePackedBank/noMTP"
Q4 = "rotorquant-signed128-4"
Q4_FORMATS = frozenset((Q4, "rotorquant-signed64-norm4"))
SHA = re.compile(r"[0-9a-f]{64}\Z")


def read_regular(path, maximum):
    """Bounded, non-symlink regular input, including a mutation check."""
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError(f"invalid bounded regular quality input: {path.name}")
        data = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if len(data) != before.st_size or any(getattr(before, k) != getattr(after, k) for k in fields):
            raise ValueError(f"quality input changed while reading: {path.name}")
    return data


@dataclass(frozen=True)
class Case:
    index: int
    name: str
    suite: str
    input_ids: tuple[int, ...]
    teacher_ids: tuple[int, ...]


@dataclass(frozen=True)
class Vector:
    path: Path
    sha256: str


@dataclass(frozen=True)
class Authority:
    path: Path
    sha256: str
    image: str
    cases: tuple[Case, ...]
    vectors: object
    vocab: int | None = None


def load_vector(vector, vocab=None):
    import numpy as np

    payload = read_regular(vector.path, 64 << 20)
    if hashlib.sha256(payload).hexdigest() != vector.sha256:
        raise ValueError(f"baseline vector checksum mismatch: {vector.path.name}")
    stream = io.BytesIO(payload)
    array = np.load(stream, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError("baseline payload must be one NPY array")
    if (
        array.dtype.str != "<f4"
        or array.ndim != 1
        or not array.size
        or stream.tell() != len(payload)
        or vocab is not None
        and array.shape != (vocab,)
        or not np.isfinite(array).all()
    ):
        raise ValueError("baseline requires complete finite little-endian Float32 vocabulary vectors")
    return array


def read_authority(path, expected_sha, *, verify_vectors=True):
    """Validate the entire fixed corpus before any supervised rank is started."""
    if not isinstance(expected_sha, str) or not SHA.fullmatch(expected_sha):
        raise ValueError("expected manifest SHA256 must contain 64 lowercase hex digits")
    path = Path(path).absolute()
    content = read_regular(path, 8 << 20)
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError("baseline manifest SHA256 differs from the pinned authority")
    raw = json.loads(content)
    if (
        not isinstance(raw, dict)
        or raw.get("complete") is not True
        or type(raw.get("tokens")) is not int
        or raw["tokens"] != STEPS
        or raw.get("dtype") != "<f4"
        or raw.get("operator") != REFERENCE_OPERATOR
        or not isinstance(raw.get("image"), str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", raw["image"])
    ):
        raise ValueError("incomplete or non-original fixed INT8 authority")
    cases, expected, totals = {}, set(), dict.fromkeys(COUNTS, 0)
    if not isinstance(raw.get("cases"), list) or len(raw["cases"]) != CASES:
        raise ValueError("the authority requires exactly 21 cases")
    names = set()
    for item in raw["cases"]:
        if not isinstance(item, dict):
            raise ValueError("invalid authority case")
        index, name, suite = item.get("index"), item.get("name"), item.get("suite")
        if (
            type(index) is not int
            or index not in range(CASES)
            or index in cases
            or not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(suite, str)
            or suite not in COUNTS
        ):
            raise ValueError("invalid or duplicate case index/name/suite")
        for key in ("input_ids", "teacher_ids"):
            if (
                not isinstance(item.get(key), list)
                or not item[key]
                or any(type(token) is not int or token < 0 for token in item[key])
            ):
                raise ValueError("fixed token IDs must be nonempty nonnegative integer lists")
        if len(item["input_ids"]) + len(item["teacher_ids"]) > CONTEXT:
            raise ValueError("authority case exceeds the fixed context8192")
        case = Case(index, name, suite, tuple(item["input_ids"]), tuple(item["teacher_ids"]))
        cases[index], names = case, names | {name}
        totals[suite] += len(case.teacher_ids)
        expected.update((index, step) for step in range(len(case.teacher_ids)))
    if totals != COUNTS or set(cases) != set(range(CASES)):
        raise ValueError("fixed baseline325/extended389 counts changed")
    directory = path.parent / "baseline-vectors"
    if directory.is_symlink():
        raise ValueError("baseline vector directory must not be a symlink")
    vectors = {}
    if not isinstance(raw.get("vectors"), list) or len(raw["vectors"]) != STEPS:
        raise ValueError("the authority requires all714 vectors")
    for item in raw["vectors"]:
        if not isinstance(item, dict):
            raise ValueError("invalid vector record")
        key = item.get("case_index"), item.get("step")
        if any(type(v) is not int for v in key) or key not in expected or key in vectors:
            raise ValueError("duplicate or out-of-range vector step")
        filename = f"{key[0]:03d}-{key[1]:04d}.npy"
        digest = item.get("sha256")
        if item.get("file") != filename or not isinstance(digest, str) or not SHA.fullmatch(digest):
            raise ValueError("vector filename or SHA256 differs from its fixed step")
        vectors[key] = Vector(directory / filename, digest)
    if set(vectors) != expected:
        raise ValueError("missing fixed teacher vector")
    authority = Authority(
        path, expected_sha, raw["image"], tuple(cases[i] for i in range(CASES)), MappingProxyType(vectors)
    )
    if verify_vectors:
        vocab = None
        for key in sorted(vectors):
            vector = load_vector(vectors[key], vocab)
            vocab = vector.size
        if any(token >= vocab for c in authority.cases for token in (*c.input_ids, *c.teacher_ids)):
            raise ValueError("authority token ID lies outside its complete vocabulary")
        authority = replace(authority, vocab=vocab)
    return authority


def compare_logits(base, candidate, teacher):
    import numpy as np

    if (
        not isinstance(candidate, np.ndarray)
        or base.dtype.str != "<f4"
        or candidate.dtype.str != "<f4"
        or base.ndim != 1
        or base.shape != candidate.shape
        or not np.isfinite(candidate).all()
        or not np.isfinite(base).all()
        or type(teacher) is not int
        or not 0 <= teacher < base.size
    ):
        raise ValueError("comparison requires full finite Float32 logits and a valid teacher ID")
    a, b = base.astype(np.float64), candidate.astype(np.float64)
    la, lb = a - a.max(), b - b.max()
    la -= np.log(np.exp(la).sum())
    lb -= np.log(np.exp(lb).sum())
    return {
        "base_top1": int(np.argmax(a)),
        "candidate_top1": int(np.argmax(b)),
        "kl": float(np.dot(np.exp(la), la - lb)),
        "nll_delta": float(la[teacher] - lb[teacher]),
        "base_nll": float(-la[teacher]),
        "candidate_nll": float(-lb[teacher]),
        "max_abs_logit_delta": float(np.abs(a - b).max()),
        "vocab": base.size,
    }


def group_summary(rows):
    count = len(rows)
    if not count:
        raise ValueError("cannot summarize an empty quality group")
    kl = math.fsum(row["kl"] for row in rows) / count
    nll = math.fsum(row["nll_delta"] for row in rows) / count
    agreement = sum(row["base_top1"] == row["candidate_top1"] for row in rows) / count
    ratio = math.exp(nll) if nll <= math.log(sys.float_info.max) else None
    return {
        "tokens": count,
        "mean_kl": kl,
        "mean_nll_delta": nll,
        "ppl_ratio": ratio,
        "ppl_ratio_overflow": ratio is None,
        "top1_agreement": agreement,
        "passed": kl <= GATES["mean_kl_max"]
        and ratio is not None
        and ratio <= GATES["ppl_ratio_max"]
        and agreement >= GATES["top1_agreement_min"],
    }


def summarize(authority, rows):
    expected = {(c.index, step): c for c in authority.cases for step in range(len(c.teacher_ids))}
    seen = set()
    for row in rows:
        key = row["case_index"], row["step"]
        if (
            key not in expected
            or key in seen
            or row["suite"] != expected[key].suite
            or row["teacher_id"] != expected[key].teacher_ids[key[1]]
        ):
            raise ValueError("quality rows changed fixed case/step/suite/token identities")
        seen.add(key)
    if len(rows) != STEPS or seen != set(expected):
        raise ValueError("quality summary requires all714 steps")
    suites = {name: group_summary([r for r in rows if r["suite"] == name]) for name in COUNTS}
    if any(suites[name]["tokens"] != count for name, count in COUNTS.items()):
        raise ValueError("independent quality suite counts changed")
    result = group_summary(rows)
    return {
        **result,
        "complete": True,
        "gate": GATES,
        "suites": suites,
        "passed": result["passed"] and all(s["passed"] for s in suites.values()),
        "comparison": "selected KV format versus frozen INT8 target; distinct from codec implementation parity",
    }


def teacher_frames(case, first_logits, advance):
    """Consume the frozen previous teacher token, never the candidate's choice."""
    current = first_logits
    for step, token in enumerate(case.teacher_ids):
        if step:
            current = advance(case.teacher_ids[step - 1])
        yield step, token, current


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class NativeObserver:
    """Quality-only call counters; inputs, outputs and selected arithmetic are untouched."""

    def __init__(self):
        self.phase, self.case = "warm", None
        self.counts, self.modules, self.originals, self.proxies = {}, {}, [], {}

    def returned(self, name, arguments=None):
        key = f"{self.phase}:{name}"
        record = self.counts.setdefault(key, {"returned_calls": 0, "cases": {}})
        record["returned_calls"] += 1
        if self.case is not None:
            record["cases"][str(self.case)] = record["cases"].get(str(self.case), 0) + 1
        arguments = arguments or {}
        record["last_integer_fields"] = {
            k: v
            for k, v in arguments.items()
            if k in ("codec", "kcodec", "vcodec", "context", "rows") and type(v) is int
        }
        if arguments.get("kcodec") in (4, 5) or arguments.get("vcodec") in (4, 5):
            record["q4_calls"] = record.get("q4_calls", 0) + 1

    def family_proxy(self, native):
        """Observe the actual fused C++ entry, without replacing any operator."""
        owner = self
        if id(native) not in self.proxies:

            class Proxy:
                def __getattr__(self, name):
                    original = getattr(native, name)
                    if name != "prep_q4":
                        return original

                    def prep(*args, **kwargs):
                        result = original(*args, **kwargs)
                        fields = {"kcodec": args[-2], "vcodec": args[-1]} if len(args) >= 2 else kwargs
                        owner.returned("tensorfold.cuda.sm70.family_native.prep_q4", fields)
                        return result

                    return prep

            self.proxies[id(native)] = Proxy()
        return self.proxies[id(native)]

    def install(self):
        for name, functions in (
            ("tensorfold.cuda.sm70.rotorquant_q4_native", ("encode", "rotate", "decode", "extension")),
            ("tensorfold.cuda.sm70.attention_native", ("attention", "extension")),
            ("tensorfold.cuda.sm70.family_native", ("extension",)),
        ):
            try:
                module = importlib.import_module(name)
            except ModuleNotFoundError as error:
                if error.name != name:
                    raise
                continue
            for function in functions:
                original = getattr(module, function)
                signature = inspect.signature(original)

                def observe(*args, _original=original, _name=name, _function=function, _signature=signature, **kwargs):
                    result = _original(*args, **kwargs)
                    arguments = _signature.bind_partial(*args, **kwargs).arguments
                    self.returned(f"{_name}.{_function}", arguments)
                    if _function == "extension":
                        self.modules[result.__name__] = str(result.__file__)
                        if _name == "tensorfold.cuda.sm70.family_native":
                            return self.family_proxy(result)
                    return result

                setattr(module, function, wraps(original)(observe))
                self.originals.append((module, function, original))

    def close(self):
        for module, name, original in reversed(self.originals):
            setattr(module, name, original)

    def receipt(self):
        return {
            "calls_returned": self.counts,
            "completion": "completed teacher frames commit validated state and synchronously gather full logits",
            "native_binaries": {
                name: {"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                for name, p in self.modules.items()
            },
        }


def native_coverage(counts, *, q4_required):
    standalone = counts.get("teacher:tensorfold.cuda.sm70.rotorquant_q4_native.encode", {}).get("returned_calls", 0)
    fused = counts.get("teacher:tensorfold.cuda.sm70.family_native.prep_q4", {}).get("q4_calls", 0)
    attention = counts.get("teacher:tensorfold.cuda.sm70.attention_native.attention", {}).get("q4_calls", 0)
    return {
        "required": q4_required,
        "standalone_encode_calls": standalone,
        "fused_prep_q4_calls": fused,
        "native_q4_attention_calls": attention,
        "passed": not q4_required or standalone + fused > 0 and attention > 0,
    }


def source_hashes():
    names = (
        "tensorfold.families.qwen4_exp.cuda.engine",
        "tensorfold.families.qwen4_exp.cuda.state",
        "tensorfold.families.qwen4_exp.cuda.forward",
        "tensorfold.families.qwen4_exp.cuda.sm70_kv",
        "tensorfold.cuda.sm70.family_ops",
        "tensorfold.cuda.sm70.attention_native",
        "tensorfold.cuda.sm70.rotorquant_q4_native",
        "tensorfold.cuda.sm70.family_native",
        "tensorfold.cuda.sm70.execution_policy",
    )
    result = {}
    for name in names:
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            path = Path(module.__file__)
            result[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def retire_worker(observer, engine, output, rank, primary, failures):
    """Retire the worker's two owners before transporting any primary failure."""
    from tensorfold.cleanup import raise_failures

    observer_error = engine_error = recording_error = None
    try:
        observer.close()
    except BaseException as error:
        observer_error = error
    try:
        if engine is not None:
            engine.close()
    except BaseException as error:
        engine_error = error
        try:
            write_json(
                output / f"rank-{rank}-close-failure.json",
                {"complete": False, "passed": False, "error": str(error), "traceback": traceback.format_exc()},
            )
        except BaseException as error:
            recording_error = error
    try:
        failures.extend(error for error in (observer_error, engine_error, recording_error) if error is not None)
    except BaseException as transport:
        # Both owners have retired even if cold diagnostic retention exhausts.
        original = primary
        if original is None:
            original = observer_error if observer_error is not None else engine_error
        if original is None:
            original = recording_error
        if original is not None:
            # The new explicit cause replaces native fields; their references
            # remain owned by this traceback frame if diagnostic allocation fails.
            _native_cause = BaseException.__cause__.__get__(original)
            _native_context = BaseException.__context__.__get__(original)
            raise original from transport
        raise
    raise_failures(primary, failures)


def worker(args):
    import numpy as np
    import torch
    import tensorfold
    from tensorfold.families.qwen4_exp.gguf_assets import prepare
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward, read_ahead

    if "site-packages" not in Path(tensorfold.__file__).parts:
        raise RuntimeError("quality workers require the installed TensorFold artifact")
    if os.environ.get("TENSORFOLD_PREFILL_ROWS", "2048") != "2048":
        raise ValueError("fixed authority requires2048-row prompt pieces")
    torch.set_num_threads(4)
    authority = read_authority(args.baseline_manifest, args.expected_manifest_sha, verify_vectors=False)
    vocab = load_vector(authority.vectors[0, 0]).size
    authority = replace(authority, vocab=vocab)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in (f"rank-{args.rank}-setup.json", "summary.json", "teacher-rows.jsonl")):
        raise ValueError("quality worker output already contains an earlier run")
    if any(token >= vocab for c in authority.cases for token in (*c.input_ids, *c.teacher_ids)):
        raise ValueError("authority token ID lies outside the target vocabulary")
    engine, rows, cases, frames, predictions = None, [], [], 0, 0
    primary, failures = None, []
    observer = NativeObserver()
    observer.install()
    try:
        engine = FlashNextEngine(
            prepare(args.model),
            depth=0,
            draft_vocab=0,
            max_len=CONTEXT,
            context_explicit=True,
            tp=args.tp,
            rank=args.rank,
            master=args.master,
            port=args.master_port,
            streams=1,
            kv_dtype=args.kv_dtype,
            kv_key_dtype=args.kv_key_dtype,
            kv_value_dtype=args.kv_value_dtype,
            graphs=False,
        )
        if (
            engine.w.mtp is not None
            or getattr(engine.w, "dspark", None) is not None
            or engine.prefill_rows != PROMPT_ROWS
            or engine.w.cfg.vocab != vocab
            or engine.sm70_profile != "optimized"
        ):
            raise RuntimeError("candidate differs from the fixed target-only optimized quality contract")
        state, weights = engine.e, engine.w
        setup = {
            "image_identity": args.image_identity,
            "baseline_image": authority.image,
            "image_identity_source": "operator --image-identity/QUALIFICATION_IMAGE",
            "baseline_manifest_sha256": authority.sha256,
            "installed_package": str(tensorfold.__file__),
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "settings": {
                "kv_dtype": args.kv_dtype,
                "key": engine.kv_pair.key_dtype,
                "value": engine.kv_pair.value_dtype,
                "depth": 0,
                "graphs": False,
                "context": CONTEXT,
                "tp": args.tp,
                "streams": 1,
                "prompt_rows": PROMPT_ROWS,
                "profile": "optimized",
                "warm_skipped": False,
                "fixed_token_ids": True,
            },
            "foundation": {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "torch_git": torch.version.git_version,
                "gpu": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
                "attention_identity": engine._sm70_attention_identity,
            },
            "kv_identity": state.st.kv_identity,
            "capacity_plan": engine.capacity_plan,
            "source_hashes": source_hashes(),
            "timing_claim": False,
            "diagnostic_wrappers": True,
        }
        write_json(output / f"rank-{args.rank}-setup.json", setup)
        observer.phase = "teacher"
        engine.comm.barrier()

        def gather(logits):
            local = logits[-1].contiguous()
            if local.dtype != torch.float32 or local.numel() * args.tp != vocab:
                raise ValueError("quality target must expose complete original Float32 vocabulary shards")
            complete = torch.empty(vocab, dtype=torch.float32, device=local.device)
            engine.comm.all_gather(local, complete)
            return complete.cpu().numpy() if args.rank == 0 else None

        with (output / f"rank-{args.rank}-frames.jsonl").open("w") as frame_file, torch.no_grad():

            def frame(case, kind, ids, logits):
                nonlocal frames, predictions
                complete = gather(logits) if logits is not None else None
                proof = {
                    "case_index": case.index,
                    "kind": kind,
                    "rows": len(ids),
                    "state_pos": state.st.pos,
                    "state_version": state.st.version,
                    "kv_identity": state.st.kv_identity,
                    "input_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                    "local_logits_shape": list(logits.shape) if logits is not None else None,
                    "local_logits_dtype": str(logits.dtype) if logits is not None else None,
                    "full_vocab": vocab,
                    "commit_returned": True,
                    "logits_gather_completed": logits is not None,
                }
                frame_file.write(json.dumps(proof) + "\n")
                frame_file.flush()
                frames += 1
                predictions += int(logits is not None)
                return complete

            for case in authority.cases:
                observer.case = case.index
                state.reset()
                current = None
                for start in range(0, len(case.input_ids), PROMPT_ROWS):
                    chunk = list(case.input_ids[start : start + PROMPT_ROWS])
                    read_ahead(weights, state.st.ple_history, chunk)
                    last = forward(
                        weights, state.st, state.pbuf, chunk, logits=start + len(chunk) == len(case.input_ids)
                    )
                    commit(weights, state.st, state.pbuf, len(chunk), len(chunk))
                    final = start + len(chunk) == len(case.input_ids)
                    gathered = frame(case, "prefill-last" if final else "prefill-piece", chunk, last if final else None)
                    if final:
                        current = gathered
                case_rows = []

                def advance(previous):
                    read_ahead(weights, state.st.ple_history, [previous])
                    last = state.forward([previous])
                    commit(weights, state.st, state.buf, 1, 1)
                    return frame(case, "teacher-decode", [previous], last)

                for step, token, current in teacher_frames(case, current, advance):
                    if args.rank == 0:
                        base = load_vector(authority.vectors[case.index, step], vocab)
                        row = {
                            "case": case.name,
                            "case_index": case.index,
                            "step": step,
                            "suite": case.suite,
                            "teacher_id": token,
                            **compare_logits(base, current, token),
                        }
                        rows.append(row)
                        case_rows.append(row)
                        with (output / "teacher-rows.jsonl").open("a") as log:
                            log.write(json.dumps(row, allow_nan=False) + "\n")
                if args.rank == 0:
                    cases.append(
                        {
                            "index": case.index,
                            "name": case.name,
                            "suite": case.suite,
                            "prompt_tokens": len(case.input_ids),
                            **group_summary(case_rows),
                        }
                    )
                    write_json(output / "cases.json", cases)
                    write_json(output / "teacher-rows.json", rows)
                    print(json.dumps({"case": case.name, "completed_steps": len(rows)}), flush=True)
        engine.comm.barrier()
        if predictions != STEPS:
            raise RuntimeError("rank did not complete all714 fixed prediction frames")
        write_json(output / f"rank-{args.rank}-native.json", observer.receipt())
        write_json(
            output / f"rank-{args.rank}-complete.json",
            {"complete": True, "frames": frames, "prediction_frames": predictions, "cases": CASES, "steps": STEPS},
        )
        if args.rank == 0:
            summary = summarize(authority, rows)
            summary.update(
                setup=setup,
                native=observer.receipt(),
                frames=frames,
                codec_implementation_parity="separate component gate; not established by target quality",
            )
            summary["quality_passed"] = summary["passed"]
            summary["native_coverage"] = native_coverage(
                observer.counts,
                q4_required=any(name in Q4_FORMATS for name in (engine.kv_pair.key_dtype, engine.kv_pair.value_dtype)),
            )
            summary["passed"] = summary["quality_passed"] and summary["native_coverage"]["passed"]
            write_json(output / "summary.json", summary)
            return 0 if summary["passed"] else 2
        return 0
    except BaseException as error:
        primary = error
        try:
            failure = {
                "complete": False,
                "passed": False,
                "rank": args.rank,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "completed_steps": len(rows),
                "completed_cases": len(cases),
                "frames": frames,
                "expected_steps": STEPS,
            }
            write_json(output / f"rank-{args.rank}-failure.json", failure)
            if args.rank == 0:
                write_json(output / "summary.json", {**failure, "gate": GATES, "cases": cases})
                write_json(output / "teacher-rows.json", rows)
        except BaseException as recording:
            failures.append(recording)
    finally:
        retire_worker(observer, engine, output, args.rank, primary, failures)


def stop_children(children):
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    for child in children:
        try:
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-identity", default=os.environ.get("QUALIFICATION_IMAGE"))
    parser.add_argument("--kv-dtype", default=Q4)
    parser.add_argument("--kv-key-dtype")
    parser.add_argument("--kv-value-dtype")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tp", type=int, default=8, choices=(8,))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--master", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29592)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.image_identity or not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_identity):
        parser.error("--image-identity or QUALIFICATION_IMAGE must identify the candidate image digest")
    if (
        args.rank is not None
        and args.rank not in range(args.tp)
        or not 1 <= args.master_port <= 65535
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
    ):
        parser.error("invalid rank, rendezvous port or positive timeout")
    from tensorfold.families.qwen4_exp.kv_formats import get_pair
    from tensorfold.families.qwen4_exp.cuda.sm70_kv import check_pair

    check_pair(get_pair(args.kv_dtype, args.kv_key_dtype, args.kv_value_dtype), head_dim=256)
    if os.environ.get("TENSORFOLD_SM70_PROFILE", "optimized") != "optimized":
        raise ValueError("quality requires the ordinary optimized SM70 profile")
    os.environ["TENSORFOLD_SM70_PROFILE"] = "optimized"
    if args.rank is not None:
        try:
            return worker(args)
        except BaseException as error:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            path = args.output_dir / f"rank-{args.rank}-failure.json"
            if not path.exists():
                write_json(
                    path,
                    {
                        "complete": False,
                        "passed": False,
                        "rank": args.rank,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
            raise
    devices = [d.strip() for d in args.gpus.split(",")]
    if len(devices) != args.tp or any(not d for d in devices) or len(set(devices)) != args.tp:
        parser.error("--gpus requires eight distinct devices in rank order")
    authority = read_authority(args.baseline_manifest, args.expected_manifest_sha)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("quality output directory must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "authority.json",
        {
            "manifest_sha256": authority.sha256,
            "baseline_image": authority.image,
            "cases": CASES,
            "steps": STEPS,
            "suites": COUNTS,
            "vocab": authority.vocab,
            "all_vector_payloads_verified": True,
            "model_run": False,
        },
    )
    if args.validate_only:
        return 0
    children, logs = [], []
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    old_signals = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    result, reason = 1, "partial"
    try:
        for rank in [*range(1, args.tp), 0]:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                str(args.model),
                "--baseline-manifest",
                str(args.baseline_manifest),
                "--expected-manifest-sha",
                authority.sha256,
                "--output-dir",
                str(args.output_dir),
                "--image-identity",
                args.image_identity,
                "--kv-dtype",
                args.kv_dtype,
                "--rank",
                str(rank),
                "--tp",
                str(args.tp),
                "--master",
                args.master,
                "--master-port",
                str(args.master_port),
            ]
            for flag, value in (("--kv-key-dtype", args.kv_key_dtype), ("--kv-value-dtype", args.kv_value_dtype)):
                if value is not None:
                    command += [flag, value]
            log = (args.output_dir / f"rank-{rank}.log").open("w")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices[rank], OPENBLAS_NUM_THREADS="1"),
                )
            )
        deadline = time.monotonic() + args.timeout_seconds
        while any(child.poll() is None for child in children):
            if stopping:
                result, reason = 130, "interrupted"
                break
            if any(child.poll() not in (None, 0, 2) for child in children):
                result, reason = 1, "rank failure"
                break
            if time.monotonic() >= deadline:
                result, reason = 1, "timeout"
                break
            time.sleep(0.2)
        else:
            codes = [child.returncode for child in children]
            if all(code == 0 for code in codes):
                result, reason = 0, "complete"
            elif all(code in (0, 2) for code in codes) and 2 in codes:
                result, reason = 2, "quality gate failed"
            else:
                result, reason = 1, "rank failure"
            if result in (0, 2):
                try:
                    summary = json.loads((args.output_dir / "summary.json").read_text())
                    complete = [
                        json.loads((args.output_dir / f"rank-{rank}-complete.json").read_text())
                        for rank in range(args.tp)
                    ]
                    if (
                        summary.get("complete") is not True
                        or summary.get("passed") is not (result == 0)
                        or any(
                            row.get("complete") is not True
                            or row.get("steps") != STEPS
                            or row.get("prediction_frames") != STEPS
                            or row.get("cases") != CASES
                            for row in complete
                        )
                    ):
                        raise ValueError("incomplete quality participant receipts")
                except (OSError, ValueError, TypeError):
                    result, reason = 1, "missing or mismatched completion receipts"
    finally:
        stop_children(children)
        for log in logs:
            log.close()
        for signum, previous in old_signals.items():
            signal.signal(signum, previous)
        write_json(
            args.output_dir / "supervisor.json",
            {
                "complete": reason in ("complete", "quality gate failed"),
                "passed": result == 0,
                "reason": reason,
                "rank_exit_codes": [c.returncode for c in children],
                "image_identity": args.image_identity,
                "baseline_manifest_sha256": authority.sha256,
            },
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
