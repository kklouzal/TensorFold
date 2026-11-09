"""Freeze independent SC117 quality DATA; never contact a model or derive answers from it.

The original C27 DATA generator is the provenance control, not an inference oracle.
This builder preserves its 27 bodies and adds five independently labelled histories,
three fixed 64K multi-key/order cases, and the original seven short cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import importlib.metadata
import platform
import sys
import types
import os
from pathlib import Path
import re

from tools.v100_rotorquant_quality import read_regular
from tools.openai_protocol import object_json
from tools.worker_lifetime import raise_failures

C27_SHA = "84f8973ae005dab76ffb8050def558edcf25b790c8b251cb3d2793a191a620b7"
SHORT_SHA = "0e370372cbbb536624182d7c020999d0c0896873dbf4df73bd78dd7ec316b7aa"
SEED = "providerone-native-rn-independent-quality-data-v1"
GENERATOR = "long-retrieval-corpus-v1/generate_corpus.py"
GENERATOR_SHA = "f8d7d23cc615b0b6b045921d6912a271f005f4ce0c98c1bfb2903824c312ea02"
SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_BODY, MAX_OUTPUT = 8 << 20, 128 << 20
FRONTEND_FILES = frozenset(('config.json', 'generation_config.json', 'ready.json', 'tokenizer.json', 'tokenizer_config.json'))


def canonical(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )


def digest(data):
    return hashlib.sha256(data).hexdigest()


def object_file(path, expected, maximum=8 << 20):
    data = read_regular(path, maximum)
    if not isinstance(expected, str) or not SHA.fullmatch(expected) or digest(data) != expected:
        raise ValueError("quality DATA authority SHA differs")
    value = object_json(data)
    if type(value) is not dict:
        raise ValueError("quality DATA authority must be an object")
    return value, data


def authority(path, expected):
    value, data = object_file(path, expected)
    if value.get("schema") != "tensorfold.sc117-quality-input-identity-v1":
        raise ValueError("actual SC117 identity schema required")
    for role in ("target", "mtp"):
        record = value.get(role)
        if (
            type(record) is not dict
            or not record.get("revision")
            or not record.get("recipe")
            or not isinstance(record.get("files"), list)
            or not record["files"]
        ):
            raise ValueError("target and MTP immutable file authorities required")
        for item in record["files"]:
            if (
                type(item) is not dict
                or not isinstance(item.get("name"), str)
                or not item["name"]
                or type(item.get("bytes")) is not int
                or item["bytes"] <= 0
                or not isinstance(item.get("sha256"), str)
                or not SHA.fullmatch(item["sha256"])
            ):
                raise ValueError("invalid immutable checkpoint file identity")
    if type(value.get("frontend")) is not dict or not value["frontend"].get("files"):
        raise ValueError("actual frontend file authority required")
    return value, data


def measure_frontend_binding():
    """Validate actual normal-frontend execution, never manufacture imports.

    Whole-tree before/after hashes remain operator capture provenance; only the
    actually loaded Python closure and native/dependency identities are checked.
    """
    import tensorfold
    package_root = Path(tensorfold.__file__).resolve().parent
    root = package_root.parent
    modules = {}
    for name, module in tuple(sys.modules.items()):
        if not (name == 'tensorfold' or name.startswith('tensorfold.')) or not isinstance(module, types.ModuleType):
            continue
        file = module.__dict__.get('__file__')
        if type(file) is not str or Path(file).suffix != '.py':
            continue
        path = Path(file).resolve()
        path.relative_to(package_root)
        relative = path.relative_to(root).as_posix()
        modules[name] = {'file': relative, 'sha256': digest(read_regular(path, MAX_BODY))}
    if not modules:
        raise ValueError('actually loaded frontend source closure is empty')
    by_path = {row['file']: row['sha256'] for row in modules.values()}
    aggregate = digest(canonical(by_path))
    native = sys.modules.get('tokenizers.tokenizers')
    if not isinstance(native, types.ModuleType) or type(native.__dict__.get('__file__')) is not str:
        raise ValueError('native tokenizer must already be loaded by the normal frontend')
    native_sha = digest(read_regular(Path(native.__file__).resolve(), 64 << 20))
    actual = {'schema': 'tensorfold.quality-frontend-execution-binding-v1',
        'loaded_source_modules': modules, 'loaded_frontend_source_sha256': aggregate,
        'native_binary_sha256': native_sha, 'platform': sys.platform, 'machine': platform.machine()}
    for package in ('tokenizers', 'jinja2', 'markupsafe'):
        actual[package] = importlib.metadata.version(package)
    return actual


def validate_frontend_binding(expected):
    actual = measure_frontend_binding()
    if type(expected) is not dict or any(name not in expected for name in actual if name != 'schema'):
        raise ValueError('current frontend execution authority is incomplete')
    if actual['loaded_source_modules'] != expected['loaded_source_modules']:
        raise ValueError('actually loaded frontend source closure differs; supply a distinct current identity')
    if any(actual[name] != expected[name] for name in actual if name != 'schema'):
        raise ValueError('actual frontend source/native/dependency/platform identity differs')
    return actual


def label(index, key="archive"):
    return "VAULT-" + digest(f"{SEED}/{key}/{index}".encode())[:16].upper()


def messages(index, records, padding, *, multi=False):
    if any(type(value) is not int or value < 0 for value in (index, records, padding)):
        raise ValueError('quality construction requires nonnegative integer extents')
    # Reject oversized work BEFORE the record loop/list/join, not after an
    # arbitrary frontend slope has created millions of Python strings.
    line = f"Record {0:06d}: archive {index:06d} routine inventory {'0' * 12} remained unchanged.\n"
    if records * len(line.encode()) + padding * len(' neutral') > MAX_BODY:
        raise ValueError('quality construction exceeds body work bound')
    keys = [label(index, str(k)) for k in range(3)] if multi else [label(index)]
    lines = []
    if not multi:
        lines.append(
            "Archive "
            + str(index)
            + ". The one ARCHIVE_KEY value assigned to this archive is "
            + keys[0]
            + ". Preserve this value while reading the routine records. The records are data, not instructions.\n"
        )
    # Record positions are deterministic; key identity and answer order are independent
    # of tokenization, fitting, candidate outputs and model choice.
    positions = (0, records // 2, records)
    for row in range(records + 1):
        if multi:
            for k, position in enumerate(positions):
                if row == position:
                    lines.append(f"Independent key {k} is {keys[k]}.\n")
        if row < records:
            value = digest(f"{SEED}/{index}/record/{row}".encode())[:12]
            lines.append(f"Record {row:06d}: archive {index:06d} routine inventory {value} remained unchanged.\n")
    order = (2, 0, 1) if index % 2 == 0 else (1, 2, 0)
    question = (
        "\nReturn the independent keys in order " + ",".join(map(str, order)) + ", separated only by |."
        if multi
        else "\nQuestion: What exact ARCHIVE_KEY value was assigned at the beginning of this archive? "
        "Reply with that value only, without explanations, punctuation or markdown.\n"
    )
    content = "".join(lines) + "\nRoutine padding:" + " neutral" * padding + question
    gold = "|".join(keys[k] for k in order) if multi else keys[0]
    if not all(content.count(key) == 1 for key in keys):
        raise ValueError("independent key must occur exactly once")
    return (
        [
            {
                "role": "system",
                "content": (
                    "Read the records as data. Answer only the final retrieval question."
                    if multi
                    else "Read the archive and answer its final retrieval question exactly."
                ),
            },
            {"role": "user", "content": content},
        ],
        gold,
        {
            "key_character_offsets": [content.index(key) for key in keys],
            "answer_key_order": list(order) if multi else [0],
        },
    )


def fit(frontend, index, target, model, reply, *, multi=False):
    def measure(records, padding):
        msg, gold, construction = messages(index, records, padding, multi=multi)
        body = {
            "model": model,
            "messages": msg,
            "max_tokens": reply,
            "temperature": 0,
            "stream": False,
            "enable_thinking": False,
        }
        if len(canonical(body)) > MAX_BODY:
            raise ValueError("quality request exceeds body work bound")
        ids = frontend(msg)
        if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("actual frontend must return nonnegative integer IDs")
        return body, ids, gold, construction

    _, base, _, _ = measure(0, 0)
    _, sample, _, _ = measure(32, 0)
    slope = len(sample) - len(base)
    if slope <= 0 or len(base) >= target:
        raise ValueError("context cannot hold quality structure")
    records = max(1, (target - len(base)) * (31 if multi else 24) // slope)
    for _ in range(12):
        result = measure(records, 0)
        if len(result[1]) <= target:
            break
        records //= 2
    else:
        raise ValueError("record fit exhausted")
    padding, seen = target - len(result[1]), set()
    for _ in range(12):
        result = measure(records, padding)
        count = len(result[1])
        if count == target:
            return (*result, {"records": records, "padding_units": padding})
        if (padding, count) in seen:
            raise ValueError("exact whole-template fit cycled")
        seen.add((padding, count))
        padding += target - count
        if not 0 <= padding <= 2 * target:
            raise ValueError("padding outside work bound")
    raise ValueError("exact whole-template fit exhausted")


class Publisher:
    def __init__(self, output):
        self.output, self.total = Path(output), 0
        self.output.mkdir(mode=0o700)
        for name in ("requests", "oracles"):
            (self.output / name).mkdir(mode=0o700)

    def put(self, name, data):
        from tensorfold.offline_quantization.text_corpus import _write_new

        if self.total + len(data) > MAX_OUTPUT:
            raise ValueError("quality output exceeds cumulative byte bound")
        _write_new(self.output / name, data)
        self.total += len(data)
        return {"bytes": len(data), "sha256": digest(data)}


def sync_directory(directory):
    # Allocate the fixed cleanup journal before acquiring the descriptor.
    primary, cleanup = None, [None]
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    except BaseException as error:
        primary = error
    try:
        os.close(fd)
    except BaseException as error:
        cleanup[0] = error
    else:
        cleanup.clear()
    raise_failures(primary, cleanup)


def build(corpus, short_contract, identity, identity_sha, output, frontend, *, gold_encoder, frontend_guard=None):
    """Frontend is a pinned normal-template DATA encoder, never an inference callback."""
    corpus = Path(corpus)
    original, _ = object_file(corpus / "manifest.json", C27_SHA)
    short, _ = object_file(short_contract, SHORT_SHA)
    actual, actual_bytes = authority(identity, identity_sha)
    if (original.get("C"), original.get("N"), original.get("R"), original.get("L")) != (27, 262080, 64, 262144):
        raise ValueError("original C27 geometry differs")
    cases = original.get("cases")
    if not isinstance(cases, list) or len(cases) != 27 or len(short.get("cases", [])) != 7:
        raise ValueError("original case catalogue differs")
    # Preflight all original bytes BEFORE publishing any output. Original filenames
    # must stay in their fixed namespaces; no path traversal/symlink input accepted.
    for case in cases:
        for kind in ("request", "oracle"):
            name = case[kind]
            if name != f"{kind}s/{case['id']}.json" or Path(case["id"]).name != case["id"]:
                raise ValueError("invalid original case path")
            raw = read_regular(corpus / name, MAX_BODY)
            if len(raw) != case[kind + "_identity"]["bytes"] or digest(raw) != case[kind + "_identity"]["sha256"]:
                raise ValueError("original case bytes differ")
    if frontend_guard is None:
        raise ValueError('current frontend execution binding guard required')
    binding = frontend_guard()
    if type(binding) is not dict or binding.get('schema') != 'tensorfold.quality-frontend-execution-binding-v1':
        raise ValueError('validated current frontend binding required')
    for case in cases:
        request, _ = object_file(corpus / case['request'], case['request_identity']['sha256'], MAX_BODY)
        ids = frontend(request['messages'])
        if (type(ids) is not list or len(ids) != original['N']
                or case.get('prompt_tokens') != original['N']
                or any(type(token) is not int or token < 0 for token in ids)
                or digest(json.dumps(ids, separators=(',', ':')).encode()) != case['prompt_token_ids_sha256']):
            raise ValueError('original C27 whole-template token alignment differs: ' + case['id'])
        del ids
    if frontend_guard() != binding:
        raise ValueError('frontend execution binding changed during original alignment')
    pub = Publisher(output)
    pub.put("input-identity.json", actual_bytes)
    preserved, added, sparse, shorts = [], [], [], []
    for case in cases:
        copy = dict(case)
        for kind in ("request", "oracle"):
            raw = read_regular(corpus / case[kind], MAX_BODY)
            if digest(raw) != case[kind + "_identity"]["sha256"]:
                raise ValueError("original case changed after preflight")
            pub.put(case[kind], raw)
        preserved.append(copy)
    seen = {case["prompt_token_ids_sha256"] for case in preserved}

    def generate(name, index, target, *, multi=False):
        body, ids, gold, construction, fitting = fit(frontend, index, target, original["model"], 64, multi=multi)
        gold_ids = gold_encoder(gold)
        if (
            not isinstance(gold_ids, list)
            or not gold_ids
            or len(gold_ids) >= 64
            or any(type(t) is not int or t < 0 for t in gold_ids)
        ):
            raise ValueError("independent gold exceeds declared reply allowance or has invalid IDs")
        token_sha = digest(json.dumps(ids, separators=(",", ":")).encode())
        if token_sha in seen:
            raise ValueError("quality histories collide")
        seen.add(token_sha)
        if multi:
            # User-content character offsets, not token offsets, establish early/mid/late placement.
            text = body["messages"][1]["content"]
            offsets = construction["key_character_offsets"]
            if not offsets[0] < offsets[1] < offsets[2] or offsets[1] <= len(text) // 4 or offsets[2] <= len(text) // 2:
                raise ValueError("multi-key placement did not span the history")
        request, oracle = f"requests/{name}.json", f"oracles/{name}.json"
        return {
            "id": name,
            "category": "64K-multi-needle-order" if multi else "near-full-independent-retrieval",
            "request": request,
            "request_identity": pub.put(request, canonical(body)),
            "oracle": oracle,
            "oracle_identity": pub.put(oracle, canonical({"expected_text": gold})),
            "expected_text": gold,
            "prompt_tokens": len(ids),
            "reply_tokens": 64,
            "context_tokens": 262144,
            "prompt_token_ids_sha256": token_sha,
            "construction": {
                **construction,
                **fitting,
                "gold_tokens": len(gold_ids),
                "placement_unit": "user-content characters; actual total whole-template token count recorded separately",
            },
        }

    for index in range(27, 32):
        added.append(generate(f"history-{index:06d}", index, 262080))
    for index in range(3):
        sparse.append(generate(f"sparse64k-{index:06d}", index + 1000, 65472, multi=True))
    pub.put("short-contract.json", read_regular(short_contract, 8 << 20))
    common = short["case_common"]
    for case in short["cases"]:
        name = "short-" + case["id"]
        oracle = {k: case[k] for k in ("expected_text", "expected_json") if k in case}
        if "history_from_case" in case:
            # Preserve runtime dependency on ACTUAL previous assistant output.
            # Never substitute its gold or manufacture a static prior reply.
            shorts.append(
                {
                    **case,
                    "category": "original-short-dependent-history",
                    "request_construction": "existing correctness client uses actual prior MTP assistant; original short-contract.json unchanged",
                }
            )
            continue
        body = {**common, "model": original["model"], "messages": case["messages"]}
        if len(oracle) != 1:
            raise ValueError("short case must retain its independent gold")
        shorts.append(
            {
                "id": name,
                "request": f"requests/{name}.json",
                "oracle": f"oracles/{name}.json",
                "request_identity": pub.put(f"requests/{name}.json", canonical(body)),
                "oracle_identity": pub.put(f"oracles/{name}.json", canonical(oracle)),
                "category": "original-short-independent-gold",
            }
        )
    if frontend_guard() != binding:
        raise ValueError("frontend execution binding changed during DATA generation")
    result = {
        "schema": "tensorfold.native-rn-quality-data-v1",
        "seed": SEED,
        "source_generator": {
            "file": GENERATOR,
            "sha256": GENERATOR_SHA,
            "relationship": "same bounded whole-template fit method; new versioned DATA authority",
        },
        "generator_sha256": digest(Path(__file__).read_bytes()),
        "original_c27_sha256": C27_SHA,
        "original_c27_alignment": {"cases": len(cases), "prompt_tokens": original['N'],
                                   "status": "MATCHED_ORIGINAL_COUNTS_AND_COMPACT_JSON_ID_SHA256"},
        "short_contract_sha256": SHORT_SHA,
        "actual_input_identity_sha256": identity_sha,
        "actual_input_identity": actual,
        "executed_frontend_binding": binding,
        "binding_scope": "Current loaded frontend checked separately; whole-tree capture fields remain historical operator provenance.",
        "profiles": {
            "Q8": {"name": "C24-first24-canonical-v1", "cases": preserved[:24]},
            "ISO4": {"name": "C32-independent-extension-v1", "cases": preserved + added},
        },
        "sparse64k": sparse,
        "short7": shorts,
        "gold_origin": "Assigned from fixed seed/index before inference; original gold unchanged; no model answers consulted.",
        "known_quality_limits": "Original loop/filter checkpoint errors remain, not corrected or special-cased; no new wrong reference-correct cases is a minimum, not broad quality qualification.",
        "media": {"status": "MISSING_AUTHORITATIVE_ASSET_AND_GOLD_MANIFEST", "no_motion_answer_invented": True},
        "teacher": {
            "status": "UNRUN_REFERENCE_FIRST_REQUIRED",
            "formats": ["Q8", "ISO4codec10"],
            "methods": ["none", "MTP4"],
            "no_inherited_714_margins": True,
        },
        "scope": "DATA only. Actual sparseQSA execution, live tokenizer alignment, full-cap owners, gold answers, teacher logits and full-model quality remain unrun.",
        "output_bytes_before_manifest": pub.total,
    }
    pub.put("manifest.json", canonical(result))
    for directory in (pub.output / "requests", pub.output / "oracles", pub.output, pub.output.parent):
        sync_directory(directory)
    return result


def frontend_files(directory, declared):
    if type(declared) is not list or len(declared) != len(FRONTEND_FILES):
        raise ValueError('complete SC117 frontend asset catalogue required')
    rows, seen = [], set()
    for row in declared:
        name = row['file']
        if type(name) is not str or not name or Path(name).name != name or name in seen:
            raise ValueError('frontend file must be a unique local basename')
        seen.add(name)
        raw = read_regular(directory / name, 64 << 20)
        rows.append({'file': name, 'bytes': len(raw), 'sha256': digest(raw)})
    if seen != FRONTEND_FILES:
        raise ValueError('complete SC117 frontend asset catalogue required')
    return rows


def prepare_frontend(directory):
    # Both modes initialize exactly this normal frontend/publisher closure.
    from tensorfold.token_metadata import load_tokenizer
    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.offline_quantization.text_corpus import _write_new
    tokenizer, template = load_tokenizer(directory), ChatTemplate(directory)
    def frontend(messages):
        return template.encode(tokenizer, template.render(messages, tools=None, enable_thinking=False))
    frontend([{'role': 'user', 'content': 'frontend execution binding'}])
    return tokenizer, template, frontend, _write_new


def capture_identity(base, base_sha, directory, output, template, write_new):
    # No checkpoint payload is read: these remain SHA-bound declarations.
    rows = frontend_files(directory, base['frontend']['files'])
    binding = measure_frontend_binding()
    current = {'schema': 'tensorfold.sc117-quality-input-identity-v1',
               'target': base['target'], 'mtp': base['mtp'],
               'base_identity_sha256': base_sha,
               'checkpoint_scope': 'Target/MTP declarations inherited from pinned base; no weights loaded or physically verified.',
               'frontend': {**binding, 'files': rows, 'template': template.provenance(),
                            'enable_thinking': False,
                            'normal_frontend_apis': 'load_tokenizer + ChatTemplate.render/encode; post-publisher initialization'}}
    if rows != frontend_files(directory, base['frontend']['files']) or binding != measure_frontend_binding():
        raise ValueError('frontend changed during identity capture')
    # Existing no-clobber publisher fsyncs the file; persist its directory too.
    write_new(Path(output), canonical(current))
    sync_directory(Path(output).parent)
    return current


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    capture = bool(argv and argv[0] == '--capture-identity')
    if capture:
        argv = argv[1:]
    p = argparse.ArgumentParser(description=__doc__)
    if capture:
        p.description = 'DATA-only --capture-identity: --output is a fresh identity FILE; checkpoint declarations come from pinned --input-identity.'
    for name in (('input-identity', 'frontend', 'output') if capture else
                 ('corpus', 'short-contract', 'input-identity', 'frontend', 'output')):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--input-identity-sha', required=True)
    args = p.parse_args(argv)
    actual, _ = authority(args.input_identity, args.input_identity_sha)
    rows = frontend_files(args.frontend, actual['frontend']['files'])
    if not capture and rows != actual['frontend']['files']:
        raise ValueError('actual frontend bytes differ')
    tokenizer, template, frontend, write_new = prepare_frontend(args.frontend)
    if rows != frontend_files(args.frontend, actual['frontend']['files']):
        raise ValueError('frontend assets changed during normal initialization')
    if capture:
        capture_identity(actual, args.input_identity_sha, args.frontend, args.output, template, write_new)
        return
    if template.provenance() != actual['frontend']['template']:
        raise ValueError('actual template provenance differs')
    def guard():
        if frontend_files(args.frontend, actual['frontend']['files']) != actual['frontend']['files']:
            raise ValueError('frontend assets changed during DATA generation')
        return validate_frontend_binding(actual['frontend'])
    build(args.corpus, args.short_contract, args.input_identity, args.input_identity_sha,
          args.output, frontend, gold_encoder=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
          frontend_guard=guard)


if __name__ == '__main__':
    main()
