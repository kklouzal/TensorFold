"""Frozen public TRAIN text corpora for offline calibration, never model evaluation.

JSONL is data only: benchmark code/tests are never executed. Tokenization uses
only local checkpoint metadata and its unchanged native chat template. Initial64
is a conversion diagnostic; full1024 follows the example's sample-count default,
not a claim of quality sufficiency. Batch1/maximum512 are explicit corpus policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request
from urllib.parse import urlparse

from tensorfold.cleanup import raise_failures

GSM_REV = "3101c7d5072418e28b9008a6636bde82a006892c"
MBPP_REV = "83f1e7b59e762648c953de751d31f75720683622"
WIKI_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
SOURCE_SHA = {"gsm8k-train.jsonl": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
              "mbpp.jsonl": "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f"}
WIKI_PARQUET_SHA = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
WIKI_JSONL_SHA = "49061c049383fe9da41d3a7dfe21482e3c83e4bd11ebfec78348404d2aa070a2"
WIKI_URL = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{WIKI_REV}/wikitext-2-raw-v1/train-00000-of-00001.parquet"
DOMAINS = ("math", "code", "prose")
MAX_LINE_BYTES = 1024 * 1024
MAX_SOURCE_BYTES = 64 * 1024 * 1024


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _file_sha(path):
    if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("missing/oversized corpus source")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_json(path):
    if path.stat().st_size > MAX_LINE_BYTES:
        raise ValueError("oversized corpus metadata")
    return json.loads(path.read_text())


def _publish_chunks(path, chunks):
    """No-clobber POSIX artifact publication; a cleanup error is terminal/reported.

    The command process owns at most this one output FD. A failed close is never
    retried on an ambiguously reused descriptor; process teardown owns any
    unresolved FD. Post-link errors retain the published artifact for diagnosis.
    """
    primary = close_error = unlink_error = None
    descriptor, name = tempfile.mkstemp(prefix=".corpus-", dir=path.parent)
    try:
        for data in chunks:
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("incomplete corpus artifact write")
                view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.link(name, path)
    except BaseException as error:
        primary = error
    try:
        os.close(descriptor)
    except BaseException as error:
        close_error = error
    try:
        Path(name).unlink()
    except BaseException as error:
        unlink_error = error
    authority = primary if primary is not None else close_error if close_error is not None else unlink_error
    _native_cause = BaseException.__cause__.__get__(authority) if authority is not None else None
    _native_context = BaseException.__context__.__get__(authority) if authority is not None else None
    try:
        raise_failures(primary, [error for error in (close_error, unlink_error) if error is not None])
    except BaseException as failure:
        if authority is None or failure is authority:
            raise
        raise authority from failure


def _write_new(path, data):
    _publish_chunks(path, (data,))


def _rows(path):
    with path.open("rb") as stream:
        number = 0
        while line := stream.readline(MAX_LINE_BYTES + 1):
            if len(line) > MAX_LINE_BYTES:
                raise ValueError("oversized corpus source record")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("corpus source record must be an object")
            yield number, value, _sha(line)
            number += 1


@dataclass(frozen=True)
class Candidate:
    domain: str
    identity: str
    record_sha256: str
    source: dict
    messages: tuple[dict, ...]

    @property
    def rank(self):
        return _sha(("tensorfold-text-corpus-v1:" + self.identity + ":" + self.record_sha256).encode())


def _partition_mbpp(records):
    """Separate official reservations and exclude exact task/whole-solution overlap.

    Only Python str.strip is used, preserving internal whitespace, case and code
    semantics. Empty solutions never establish an overlap. No fuzzy matching.
    """
    records = list(records)
    reserved, prompts, solutions, seen = [], {}, {}, set()
    for _, row, digest in records:
        task = row.get("task_id")
        if type(task) is not int or task in seen or not 1 <= task <= 974:
            raise ValueError("MBPP task identity differs")
        seen.add(task)
        if any(not isinstance(row.get(key), str) for key in ("text", "code")):
            raise ValueError("MBPP task/solution schema differs")
        if task <= 600:
            prompt, solution = row["text"].strip(), row["code"].strip()
            prompts.setdefault(prompt, []).append(task)
            if solution:
                solutions.setdefault(solution, []).append(task)
            reserved.append({"id": f"mbpp/task/{task}", "record_sha256": digest,
                "partition": "test" if 11 <= task <= 510 else "validation" if task >= 511 else "few-shot-only"})
    train, excluded = [], []
    for index, row, digest in records:
        if row["task_id"] <= 600:
            continue
        prompt, solution = row["text"].strip(), row["code"].strip()
        task_matches = sorted(prompts.get(prompt, ()))
        code_matches = sorted(solutions.get(solution, ())) if solution else []
        if task_matches or code_matches:
            excluded.append({"train_id": row["task_id"], "record_sha256": digest,
                "normalized_prompt_sha256": _sha(prompt.encode()),
                "normalized_solution_sha256": _sha(solution.encode()),
                "reserved_prompt_ids": task_matches, "reserved_solution_ids": code_matches})
        else:
            train.append((index, row, digest))
    return train, reserved, excluded, seen


def _wiki_lineage(raw):
    acquired = _load_json(raw / "wikitext-acquisition.json")
    wiki = _load_json(raw / "wikitext-normalization.json")
    if (acquired.get("url") != WIKI_URL or acquired.get("revision") != WIKI_REV
            or acquired.get("split") != "train" or acquired.get("sha256") != WIKI_PARQUET_SHA
            or _file_sha(raw / "wikitext-train.parquet") != WIKI_PARQUET_SHA
            or wiki.get("revision") != WIKI_REV or wiki.get("split") != "train"
            or wiki.get("parquet_sha256") != WIKI_PARQUET_SHA or wiki.get("jsonl_sha256") != WIKI_JSONL_SHA
            or _file_sha(raw / "wikitext-train.jsonl") != WIKI_JSONL_SHA):
        raise ValueError("WikiText frozen TRAIN acquisition/normalization lineage differs")
    return {**wiki, "url": WIKI_URL, "acquisition_sha256": _file_sha(raw / "wikitext-acquisition.json")}


def candidates(raw):
    """Validate exact acquisition identities before reading TRAIN candidates."""
    acquired = _load_json(raw / "acquisition.json")
    records = {item["file"]: item for item in acquired["records"]}
    specs = (("gsm8k-train.jsonl", GSM_REV), ("mbpp.jsonl", MBPP_REV))
    sources = {}
    for name, revision in specs:
        source = records[name]
        if (revision not in source["url"] or source["sha256"] != SOURCE_SHA[name]
                or _file_sha(raw / name) != SOURCE_SHA[name]):
            raise ValueError("frozen source acquisition identity differs")
        sources[name] = dict(source)
    result = {domain: [] for domain in DOMAINS}
    for index, row, digest in _rows(raw / "gsm8k-train.jsonl"):
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("question", "answer")):
            raise ValueError("GSM8K train schema differs")
        result["math"].append(Candidate("math", f"gsm8k/train/{index}", digest,
            {**sources["gsm8k-train.jsonl"], "dataset": "gsm8k", "revision": GSM_REV, "split": "train", "row": index},
            ({"role": "user", "content": row["question"]}, {"role": "assistant", "content": row["answer"]})))
    train, official_holdout, mbpp_excluded, seen = _partition_mbpp(_rows(raw / "mbpp.jsonl"))
    for _, row, digest in train:
        task = row["task_id"]
        identity = f"mbpp/task/{task}"
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("text", "code")):
            raise ValueError("MBPP train schema differs")
        tests = row.get("test_list")
        if not isinstance(tests, list) or any(not isinstance(item, str) for item in tests):
            raise ValueError("MBPP train test descriptions differ")
        prompt = row["text"] + "\n\nThe Python implementation should pass these examples:\n" + "\n".join(tests)
        result["code"].append(Candidate("code", identity, digest,
            {**sources["mbpp.jsonl"], "dataset": "mbpp", "revision": MBPP_REV, "split": "train", "task_id": task},
            ({"role": "user", "content": prompt}, {"role": "assistant", "content": row["code"]})))
    if seen != set(range(1, 975)):
        raise ValueError("MBPP source task inventory is incomplete")
    wiki = _wiki_lineage(raw)
    for index, row, digest in _rows(raw / "wikitext-train.jsonl"):
        if row.get("row_id") != index or not isinstance(row.get("text"), str):
            raise ValueError("WikiText normalized row schema/order differs")
        text = row["text"].strip()
        if not text or text.startswith("="):
            continue  # blank separators/article headings are not prose paragraphs
        result["prose"].append(Candidate("prose", f"wikitext-2-raw-v1/train/{index}", digest,
            {"dataset": "Salesforce/wikitext", "revision": WIKI_REV, "split": "train", "row": index,
             "parquet_sha256": wiki["parquet_sha256"], "jsonl_sha256": wiki["jsonl_sha256"]},
            ({"role": "user", "content": "Read the following passage:\n\n" + text},)))
    return result, official_holdout, sources, wiki, mbpp_excluded


def balanced_counts(capacities, total):
    """Deterministic equal-domain allocation subject to unique eligible TRAIN data."""
    counts = dict.fromkeys(DOMAINS, 0)
    while sum(counts.values()) < total:
        changed = False
        for domain in DOMAINS:
            if counts[domain] < capacities[domain] and sum(counts.values()) < total:
                counts[domain] += 1
                changed = True
        if not changed:
            raise ValueError("insufficient disjoint complete TRAIN records for full corpus")
    return counts


def freeze(raw, model, output, *, initial_counts=None, full_count=1024, holdout_count=8, max_tokens=512):
    if (type(full_count) is not int or full_count < 1 or type(holdout_count) is not int or holdout_count < 1
            or type(max_tokens) is not int or max_tokens < 1):
        raise ValueError("positive bounded corpus policy required")
    initial_counts = {"math": 24, "code": 24, "prose": 16} if initial_counts is None else initial_counts
    if set(initial_counts) != set(DOMAINS) or any(type(n) is not int or n < 1 for n in initial_counts.values()):
        raise ValueError("positive explicit domain counts required")
    pools, official, sources, wiki, mbpp_excluded = candidates(raw)
    # Heavy dependencies only in the isolated CPU tokenizer stage; never weights.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    template = (model / "chat_template.jinja").read_text()
    if tokenizer.chat_template != template:
        raise ValueError("actual selected tokenizer template differs from checkpoint file")
    config = _load_json(model / "config.json")
    media_ids = {config[key] for key in ("image_token_id", "video_token_id", "audio_token_id")}
    selected, heldout, exclusions = {}, {}, {}
    # Tokenize every candidate before selection, preserving deterministic ordering
    # and complete turns rather than truncating a response to fit the cap.
    for domain, records in pools.items():
        eligible, excluded = [], Counter()
        for candidate in sorted(records, key=lambda item: (item.rank, item.identity)):
            messages = list(candidate.messages)
            generation_prompt = messages[-1]["role"] == "user"
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt,
                                                 enable_thinking=True)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) > max_tokens:
                excluded["overlong_complete_record"] += 1
                continue
            if not ids or media_ids.intersection(ids):
                excluded["empty_or_unencoded_media_placeholder"] += 1
                continue
            eligible.append({"id": candidate.identity, "domain": domain,
                "record_sha256": candidate.record_sha256, "selection_rank": candidate.rank,
                "source": candidate.source, "messages": messages,
                "add_generation_prompt": generation_prompt, "rendered_sha256": _sha(text.encode()),
                "input_ids": ids, "attention_mask": [1] * len(ids), "token_count": len(ids)})
        if len(eligible) < holdout_count + initial_counts[domain]:
            raise ValueError("insufficient complete eligible training records for domain: " + domain)
        heldout[domain] = [{key: row[key] for key in ("id", "record_sha256", "selection_rank", "source")}
                           for row in eligible[:holdout_count]]
        selected[domain] = eligible[holdout_count:]
        exclusions[domain] = {"raw_candidates": len(records), "eligible_before_holdout": len(eligible),
                              "excluded": dict(excluded)}
    full_counts = balanced_counts({key: len(value) for key, value in selected.items()}, full_count)
    if any(initial_counts[domain] > full_counts[domain] for domain in DOMAINS):
        raise ValueError("initial corpus must be nested inside the full candidate")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for label, counts in (("initial64", initial_counts), ("full1024", full_counts)):
        # Round-robin domains, preserving hash rank inside each source.
        rows = [selected[domain][index] for index in range(max(counts.values()))
                for domain in DOMAINS if index < counts[domain]]
        encoded = "".join(_json(row) + "\n" for row in rows).encode()
        name = label + ".jsonl"
        _write_new(output / name, encoded)
        artifacts[label] = {"file": name, "sha256": _sha(encoded), "records": len(rows), "domain_counts": counts,
                            "total_tokens": sum(row["token_count"] for row in rows),
                            "maximum_tokens": max(row["token_count"] for row in rows),
                            "purpose": "initial conversion diagnostic" if label == "initial64" else "full TRAIN candidate, NOT quality qualification"}
    heldout_record = {"reserved_train": heldout, "mbpp_official_reserved": official,
        "gsm8k_official_test": {"revision": GSM_REV, "path": "grade_school_math/data/test.jsonl", "status": "UNFETCHED"},
        "wikitext_official_other_splits": {"revision": WIKI_REV, "status": "validation/test UNFETCHED"}}
    _write_new(output / "heldout-identities.json", (_json(heldout_record) + "\n").encode())
    manifest = {"schema": "tensorfold.calibration.text-corpus", "schema_version": 1,
        "scope": "public TRAIN math/code/prose only; no calibration/evaluation/quality claim", "artifacts": artifacts,
        "policy": {"batch_size": 1, "max_sequence_tokens": max_tokens, "truncate": False, "holdout_train_per_domain": holdout_count,
                   "selection": "sha256 rank, reserve first eligible holdout, then nested calibration; full mix equal-domain waterfill",
                   "thinking": True, "assistant_answers": "source-provided ordinary assistant content; no invented thought channel"},
        "tokenizer": {"snapshot": model.name, "class": type(tokenizer).__name__,
            "files": {name: _file_sha(model / name) for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}},
        "sources": sources, "wikitext": wiki, "exclusions": exclusions,
        "mbpp_official_overlap_policy": {"normalization": "Python str.strip ONLY; internal whitespace/case unchanged",
            "reference_ids": "official1–600", "criteria": "exact task prompt OR nonempty complete solution",
            "excluded_train_records": mbpp_excluded},
        "heldout_sha256": _file_sha(output / "heldout-identities.json"),
        "coverage_limits": ["English math/code/prose only", "no image/video/audio/tools/multilingual calibration", "no quality claim"]}
    _write_new(output / "manifest.json", (_json(manifest) + "\n").encode())
    return manifest


def acquire_wikitext(output):
    """Fetch exactly the frozen public TRAIN object; no evaluation payloads."""
    digest, size, start = hashlib.sha256(), 0, time.monotonic()
    with urllib.request.urlopen(WIKI_URL, timeout=30) as response:
        if urlparse(response.geturl()).scheme != "https":
            raise ValueError("corpus redirect must retain HTTPS")
        def chunks():
            nonlocal size
            while chunk := response.read(64 * 1024):
                size += len(chunk)
                if size > MAX_SOURCE_BYTES or time.monotonic() - start > 180:
                    raise ValueError("corpus acquisition exceeds admitted bytes/time")
                digest.update(chunk)
                yield chunk
            if digest.hexdigest() != WIKI_PARQUET_SHA:
                raise ValueError("WikiText download differs from the frozen TRAIN object")
        _publish_chunks(output / "wikitext-train.parquet", chunks())
    record = {"url": WIKI_URL, "revision": WIKI_REV, "split": "train", "bytes": size,
              "sha256": digest.hexdigest(), "deadline": "180s checked between bounded reads;30s socket timeout"}
    _write_new(output / "wikitext-acquisition.json", (_json(record) + "\n").encode())
    return record


def normalize_wikitext(parquet, output, *, revision=WIKI_REV):
    """Use Arrow's actual Parquet reader in the separate data-only environment."""
    if revision != WIKI_REV:
        raise ValueError("this frozen corpus requires the pinned WikiText revision")
    source_sha = _file_sha(parquet)
    acquired = _load_json(parquet.parent / "wikitext-acquisition.json")
    if (source_sha != WIKI_PARQUET_SHA or acquired.get("sha256") != WIKI_PARQUET_SHA
            or acquired.get("url") != WIKI_URL or acquired.get("revision") != WIKI_REV
            or acquired.get("split") != "train"):
        raise ValueError("normalizer requires the exact acquired frozen WikiText TRAIN object")
    import pyarrow.parquet as pq
    table = pq.ParquetFile(parquet)
    if table.metadata.num_rows > 100000 or sum(table.metadata.row_group(i).total_byte_size for i in range(table.num_row_groups)) > 256 * 1024**2:
        raise ValueError("decoded prose metadata exceeds the data-only envelope")
    digest, count = hashlib.sha256(), 0
    def chunks():
        nonlocal count
        for batch in table.iter_batches(batch_size=1024, columns=["text"], use_threads=False):
            for text in batch.column(0).to_pylist():
                if not isinstance(text, str):
                    raise ValueError("WikiText TRAIN text schema differs")
                line = (_json({"row_id": count, "text": text}) + "\n").encode()
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError("WikiText row exceeds bounded record size")
                digest.update(line)
                count += 1
                yield line
    _publish_chunks(output / "wikitext-train.jsonl", chunks())
    record = {"revision": revision, "split": "train", "parquet_sha256": source_sha,
              "jsonl_sha256": digest.hexdigest(), "rows": count, "reader": "pyarrow.parquet"}
    _write_new(output / "wikitext-normalization.json", (_json(record) + "\n").encode())
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser("acquire-wikitext")
    acquire.add_argument("--output", type=Path, required=True)
    normalize = sub.add_parser("normalize-wikitext")
    normalize.add_argument("--parquet", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("freeze")
    prepare.add_argument("--raw", type=Path, required=True)
    prepare.add_argument("--model", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.command == "acquire-wikitext":
        result = acquire_wikitext(args.output)
    elif args.command == "normalize-wikitext":
        result = normalize_wikitext(args.parquet, args.output)
    else:
        result = freeze(args.raw, args.model, args.output, max_tokens=args.max_tokens)
    print(_json(result))


if __name__ == "__main__":
    main()
