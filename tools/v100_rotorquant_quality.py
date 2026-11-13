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



def _owned_comparison_contract():
    if __package__:
        from . import v100_owned_model_contract
    else:
        import v100_owned_model_contract
    return v100_owned_model_contract


def _owned_comparison_file(root, reference, control, *, maximum=8 << 20, parsed=True, json_root=dict):
    contract = _owned_comparison_contract()
    contract.identity(reference, key='file', rank='rank' in reference)
    path = root / reference['file']
    if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError('owned comparison reference escapes its evidence root')
    if reference['bytes'] > maximum:
        raise ValueError('owned comparison reference exceeds its bounded extent')
    digest = _owned_teacher_hash(path, reference['bytes'], maximum, control)
    if digest != reference['sha256']:
        raise ValueError('owned comparison referenced SHA differs')
    if not parsed:
        return path
    raw = _owned_teacher_read(path, reference['bytes'])
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('owned comparison reference changed before parsing')
    control.check()
    return _owned_teacher_parse(raw, root_type=json_root)


def _owned_comparison_vector(path, vector, vocab, control):
    """Preflight the actual NPY header on the same held, verified payload.

    NumPy owns the header grammar. Its parser reads the declared header before
    checking max_header_size, so cap each header read before it can allocate.
    No native/vector allocation occurs until shape/dtype/order/extent agree.
    """
    import numpy as np
    raw = _owned_teacher_read(path, vector['bytes'])
    if hashlib.sha256(raw).hexdigest() != vector['sha256']:
        raise ValueError('owned comparison vector changed before header preflight')
    class Header:
        def __init__(self):
            self.stream = io.BytesIO(raw)
        def read(self, size):
            if type(size) is not int or not 0 <= size <= 4096:
                raise ValueError('owned comparison NPY header exceeds its declared header allowance')
            return self.stream.read(size)
        def tell(self):
            return self.stream.tell()
    header = Header()
    version = np.lib.format.read_magic(header)
    readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
    if version not in readers:
        raise ValueError('owned comparison requires supported bounded NPY1/2 header')
    shape, fortran, dtype = readers[version](header, max_header_size=4096)
    if (shape != (vocab,) or dtype.str != '<f4' or fortran is not False
            or header.tell() + 4 * vocab != len(raw)):
        raise ValueError('owned comparison actual NPY header/payload extent differs from full Float32 vocabulary')
    control.check()
    stream = io.BytesIO(raw)
    array = np.load(stream, allow_pickle=False, max_header_size=4096)
    if (not isinstance(array, np.ndarray) or array.dtype.str != '<f4' or array.shape != (vocab,)
            or stream.tell() != len(raw) or not np.isfinite(array).all()):
        raise ValueError('owned comparison requires complete finite Float32 vectors')
    control.check()
    return array



def _owned_coordinate_kv(identity, activation):
    """Normalize only State's typed activation marker, after same-run proof.

    Engine._bind_coordinate_profile issues a nonce-bearing resident namespace,
    explicitly not a content/qualification certificate. Position and all other
    KV markers remain unchanged; malformed/duplicate/missing markers refuse.
    """
    if type(identity) is not list:
        raise ValueError('owned comparison requires the complete State KV identity')
    markers = [index for index, value in enumerate(identity)
               if type(value) is list and value and value[0] == 'coordinate-profile-v1']
    if len(markers) != 1:
        raise ValueError('owned comparison requires exactly one typed coordinate activation marker')
    index = markers[0]
    if len(identity[index]) != 2 or identity[index][1] != activation:
        raise ValueError('owned comparison State coordinate marker differs from its own activation')
    return [*identity[:index], ['coordinate-profile-v1', '<resident-activation>'], *identity[index + 1:]]


def _owned_coordinate_semantics(profile, activation):
    if type(profile) is not dict or profile.get('coordinate_profile_identity') != activation:
        raise ValueError('owned comparison profile coordinate activation differs within its run')
    semantic = profile.get('teacher_semantics')
    if type(semantic) is not dict:
        raise ValueError('owned comparison requires actual teacher State semantics')
    return {**semantic, 'state_kv_identity': _owned_coordinate_kv(semantic.get('state_kv_identity'), activation)}


def _owned_comparison_side(selected, mode, control, *, contract=None):
    """Consume successful owner records; never rerun or reinterpret a launcher."""
    contract = _owned_comparison_contract() if contract is None else contract
    if (type(selected) is not dict or set(selected) != {'root', 'host_result', 'completion'}
            or not contract.absolute(selected['root'])):
        raise ValueError('closed owned comparison side required')
    root = Path(selected['root'])
    host = _owned_comparison_file(root, selected['host_result'], control)
    inner = host.get('inner') or {}
    model = contract.validate_model_result(inner.get('model_result'))
    authority_ref = model['teacher_authority']
    authority = _owned_comparison_file(root, authority_ref, control)
    contract.validate_completed_teacher_host(host, authority_sha256=authority_ref['sha256'],
                                             authority_bytes=authority_ref['bytes'])
    completion_ref = selected['completion']
    contract.identity(completion_ref)
    if completion_ref['bytes'] > 8 << 20 or _owned_teacher_hash(completion_ref['path'], completion_ref['bytes'], 8 << 20, control) != completion_ref['sha256']:
        raise ValueError('owned comparison external completion SHA/extent differs')
    completion_raw = _owned_teacher_read(completion_ref['path'], completion_ref['bytes'])
    if hashlib.sha256(completion_raw).hexdigest() != completion_ref['sha256']:
        raise ValueError('external completion changed before parsing')
    completion = _owned_teacher_object(completion_raw)
    contract.validate_teacher_completion(completion, host_result_sha256=selected['host_result']['sha256'],
                                           authority_sha256=authority_ref['sha256'])
    inner_copy = _owned_teacher_object(_owned_teacher_read(root / 'owned-teacher-operation-result.json', 8 << 20))
    if inner_copy != inner:
        raise ValueError('owned comparison outer/inner operation records differ')
    client = _owned_comparison_file(root, model['client_report'], control)
    if (client.get('complete') is not True or client.get('client_protocol_passed') is not True
            or client.get('post_attempts') != 1 or client.get('mode') != mode
            or client.get('token_ids') != authority.get('teacher_ids')
            or client.get('teacher_plan_sha256') != model['teacher_plan_sha256']
            or client.get('teacher_client_sha256') != model['teacher_client_sha256']
            or client.get('cached_tokens') != 0 or client.get('drafts') is not False
            or client.get('drafted') != 0 or client.get('accepted') != 0):
        raise ValueError('owned comparison client/prefix completion differs')
    if (authority.get('schema') != 'tensorfold.owned-teacher-authority-v1' or authority.get('mode') != mode
            or authority.get('cohort_complete') is not True or authority.get('closed') is not True
            or authority.get('finished') is not True or authority.get('failed') is not False
            or type(authority.get('launch_exit')) is not int or authority['launch_exit'] != 0
            or authority.get('method') != 'none' or authority.get('input_sha256') != model['teacher_plan_sha256']):
        raise ValueError('owned comparison authority is not a completed none cohort')
    profile = authority.get('actual_owned_profile')
    activation = profile.get('coordinate_profile_identity') if type(profile) is dict else None
    if type(activation) is not str or not SHA.fullmatch(activation):
        raise ValueError('owned comparison requires its actual activation SHA identity')
    stable_semantics = _owned_coordinate_semantics(profile, activation)
    state_kv = profile['teacher_semantics']['state_kv_identity']
    frames, ids, vocab = (authority.get(key) for key in ('frames', 'teacher_ids', 'vocab'))
    if (type(ids) is not list or not 0 < len(ids) <= 64 or type(vocab) is not int or not 0 < vocab <= (64 << 20) // 4 - 1024
            or any(type(token) is not int or not 0 <= token < vocab for token in ids)
            or type(frames) is not list or len(frames) != len(ids)):
        raise ValueError('owned comparison requires bounded complete full-vocabulary frames')
    # All metadata/cardinalities are checked before loading any NPY payload.
    for step, frame in enumerate(frames):
        if frame.get('kv_identity') != state_kv:
            raise ValueError('owned comparison authority frame State identity differs within its activation')
        _owned_coordinate_kv(frame['kv_identity'], activation)
        vector = frame.get('vector', {})
        if (frame.get('step') != step or frame.get('position') != len(authority['input_ids']) + step
                or frame.get('teacher_id') != ids[step] or frame.get('full_vocab') != vocab
                or type(frame.get('selected_original')) is not int or not 0 <= frame['selected_original'] < vocab
                or mode == 'capture' and frame['selected_original'] != ids[step]
                or vector.get('file') != f'teacher-rank0/frame-{step:06d}.npy'
                or vector.get('dtype') != '<f4' or vector.get('shape') != [vocab]
                or type(vector.get('bytes')) is not int or not 0 < vector['bytes'] <= 4 * vocab + 4096):
            raise ValueError('owned comparison frame/position/vector identity differs')
    for collection in ('rank_receipts', 'startup_receipts'):
        expected = authority.get(collection)
        if type(expected) is not list or len(expected) != 8:
            raise ValueError('owned comparison authority rank receipt list differs')
        for rank, (saved, bound) in enumerate(zip(expected, model[collection])):
            if bound != {**saved, 'file': 'teacher/' + saved.get('file', '')} or saved.get('rank') != rank:
                raise ValueError('owned comparison authority/model receipt binding differs')
    native_sets = []
    for collection in ('rank_receipts', 'startup_receipts', 'child_native_records'):
        for rank, reference in enumerate(model[collection]):
            record = _owned_comparison_file(root, reference, control)
            if record.get('rank') != rank:
                raise ValueError('owned comparison rank record order differs')
            if collection in ('rank_receipts', 'startup_receipts'):
                own_profile = record.get('actual_owned_profile') if collection == 'rank_receipts' else record.get('teacher_profile')
                if _owned_coordinate_semantics(own_profile, activation) != stable_semantics:
                    raise ValueError('owned comparison rank State semantics differ within its activation')
                if collection == 'startup_receipts' and record.get('coordinate_profile_identity') != activation:
                    raise ValueError('owned comparison startup activation differs from its linked profile')
                if collection == 'rank_receipts':
                    own_frames = record.get('frames')
                    if type(own_frames) is not list or len(own_frames) != len(frames):
                        raise ValueError('owned comparison rank frame list differs')
                    for frame in own_frames:
                        if frame.get('kv_identity') != state_kv:
                            raise ValueError('owned comparison rank frame State identity differs within its activation')
                        _owned_coordinate_kv(frame['kv_identity'], activation)
            if collection == 'child_native_records':
                mapped = record.get('mapped_elfs')
                if (type(mapped) is not list or not 0 < len(mapped) <= 64
                        or any(type(item) is not dict or type(item.get('name')) is not str
                            or not item['name'] or type(item.get('sha256')) is not str
                            or not SHA.fullmatch(item['sha256']) for item in mapped)):
                    raise ValueError('owned comparison lacks actual bounded native name/SHA identity')
                native_sets.append(sorted({(item['name'], item['sha256']) for item in mapped}))
            if collection == 'rank_receipts' and (record.get('closed') is not True or record.get('finished') is not True
                    or record.get('failed') is not False or record.get('mode') != mode
                    or record.get('teacher_ids') != ids or record.get('runtime_sources') != authority.get('runtime_sources')):
                raise ValueError('owned comparison rank did not finish the declared prefix/source')
    content = _owned_comparison_file(root, inner['content_binding'], control)
    if content.get('completed') is not True:
        raise ValueError('owned comparison model content was not completed')
    return {'root': root, 'inner': inner, 'authority': authority, 'authority_ref': authority_ref,
            'native_sets': native_sets, 'activation': activation, 'stable_semantics': stable_semantics,
            'host': host, 'model': model, 'content': content}


def compare_owned_teacher(reference, replay, *, read_timeout_s):
    """Offline matched-control diagnostics; no old714 margin or quality pass."""
    if (type(read_timeout_s) not in (int, float) or not 0 < read_timeout_s <= sys.maxsize
            or not math.isfinite(read_timeout_s)):
        raise ValueError('owned comparison requires positive finite numeric read_timeout_s excluding bool')
    import numpy as np
    control = _teacher_read_control(read_timeout_s)
    first = _owned_comparison_side(reference, 'capture', control)
    second = _owned_comparison_side(replay, 'replay', control)
    a, b = first['authority'], second['authority']
    if first['inner']['image_id_declared'] != second['inner']['image_id_declared']:
        raise ValueError('matched reference control requires the same immutable image')
    if first['native_sets'] != second['native_sets']:
        raise ValueError('matched reference control actual native name/SHA sets differ')
    if b.get('reference_sha256') != first['authority_ref']['sha256']:
        raise ValueError('replay does not name this successful capture authority')
    for key in ('model_assets', 'frontend_assets', 'frontend_root', 'model_entrypoint', 'execution_manifest'):
        if first['inner'].get('operation', {}).get(key) != second['inner'].get('operation', {}).get(key):
            raise ValueError('owned comparison actual bound model/frontend/execution differs: ' + key)
    for key in ('input_ids', 'vocab', 'widths', 'teacher_ids', 'owner_family'):
        if a.get(key) != b.get(key):
            raise ValueError('owned comparison changed frozen ' + key)
    for key in ('case_id', 'model_sha256', 'frontend_sha256', 'profile_sha256'):
        if a.get('declared_identity', {}).get(key) != b.get('declared_identity', {}).get(key):
            raise ValueError('owned comparison declared semantic identity differs: ' + key)
    if first['stable_semantics'] != second['stable_semantics']:
        raise ValueError('owned comparison stable actual teacher State semantics differ')
    for key in ('plan_identity', 'policy', 'target_banks', 'mtp_banks', 'state_count', 'capability'):
        if a.get('actual_owned_profile', {}).get(key) != b.get('actual_owned_profile', {}).get(key):
            raise ValueError('owned comparison actual semantic profile differs: ' + key)
    for key in ('engine', 'multi', 'decode', 'sampling'):
        value = a.get('runtime_sources', {}).get(key)
        if not isinstance(value, str) or not SHA.fullmatch(value) or value != b.get('runtime_sources', {}).get(key):
            raise ValueError('owned comparison installed runtime source differs: ' + key)
    rows = []
    for index, (old, current) in enumerate(zip(a['frames'], b['frames'])):
        for key in ('step', 'position', 'teacher_id', 'full_vocab', 'role', 'execution_family'):
            if old.get(key) != current.get(key):
                raise ValueError('owned comparison frame pairing differs: ' + key)
        if (_owned_coordinate_kv(old['kv_identity'], first['activation'])
                != _owned_coordinate_kv(current['kv_identity'], second['activation'])):
            raise ValueError('owned comparison stable frame KV identities differ')
        vectors = []
        for side, frame in ((first, old), (second, current)):
            vector = frame['vector']
            ref = {key: vector[key] for key in ('file', 'bytes', 'sha256')}
            path = _owned_comparison_file(side['root'] / 'teacher', ref, control, maximum=4*a['vocab']+4096, parsed=False)
            vectors.append(_owned_comparison_vector(path, vector, a['vocab'], control))
            control.check()
        left, right = vectors
        unequal = left.view('<u4') != right.view('<u4')
        metrics = compare_logits(left, right, old['teacher_id'])
        rows.append({'step': index, 'position': old['position'], 'teacher_id': old['teacher_id'],
            'reference_original_selected': old['selected_original'], 'replay_original_selected': current['selected_original'],
            'reference_forced_id': old['teacher_id'], 'replay_forced_id': current['teacher_id'],
            'exact_uint32_bits': not bool(unequal.any()), 'different_uint32_values': int(unequal.sum()),
            'reference_raw_top1_ties': int(np.count_nonzero(left == left.max())),
            'replay_raw_top1_ties': int(np.count_nonzero(right == right.max())),
            'finite': True, **metrics})
        del vectors, left, right, unequal
    exact = all(row['exact_uint32_bits'] for row in rows)
    selections = all(row['reference_original_selected'] == row['replay_original_selected'] for row in rows)
    source = Path(__file__)
    producer_sha = _owned_teacher_hash(source, source.stat().st_size, 8 << 20, control)
    return {'schema': 'tensorfold.owned-teacher-comparison-v1', 'complete': True,
        'comparison_tool_sha256': producer_sha,
        'status': 'EXACT_MATCHED_CONTROL' if exact and selections else 'NONEXACT_MATCHED_CONTROL_VARIABILITY_REQUIRES_ANALYSIS',
        'exact_uint32_all_frames': exact, 'original_selection_agreement_all_frames': selections,
        'qualification': False, 'quality_acceptance': False, 'P3_valid': False,
        'reference': reference, 'replay': replay, 'frames': rows,
        'image_id': first['inner']['image_id_declared'], 'native_sets_by_rank': first['native_sets'],
        'coordinate_activations': [first['activation'], second['activation']],
        'coordinate_scope': 'each run linked completely; only typed resident-activation marker normalized across runs',
        'runtime_sources': {key: a['runtime_sources'][key] for key in ('engine', 'multi', 'decode', 'sampling')},
        'teacher_sources': [a['runtime_sources'].get('teacher'), b['runtime_sources'].get('teacher')],
        'operator_commits': [first['inner'].get('source_commit'), second['inner'].get('source_commit')],
        'scope': 'one matched forced-prefix control; no tolerance, broad quality, free-generation or analytical acceptance'}


def _compare_owned_teacher_main(argv):
    parser = argparse.ArgumentParser(description='Offline owned capture/replay control comparison; no quality margins')
    parser.add_argument('--comparison-input', type=Path, required=True)
    parser.add_argument('--expected-input-sha', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    raw = _owned_teacher_read(args.comparison_input, 8 << 20)
    if not SHA.fullmatch(args.expected_input_sha) or hashlib.sha256(raw).hexdigest() != args.expected_input_sha:
        raise ValueError('owned comparison input SHA differs')
    data = _owned_teacher_object(raw)
    if type(data) is not dict or set(data) != {'schema', 'reference', 'replay', 'read_timeout_s'} or data['schema'] != 'tensorfold.owned-teacher-comparison-input-v1':
        raise ValueError('closed owned comparison input required')
    report = compare_owned_teacher(data['reference'], data['replay'], read_timeout_s=data['read_timeout_s'])
    report['comparison_input_sha256'] = args.expected_input_sha
    args.output_dir.mkdir(mode=0o700)
    _owned_teacher_json(args.output_dir / 'comparison.json', report, 8 << 20, _teacher_read_control(data['read_timeout_s']))
    return 0  # Nonexact results are preserved diagnostics, not acceptance or a hidden tolerance.


def _candidate_teacher_side(selected, mode, control, *, teacher_scope='rn12-p4096-none-v1'):
    """Read the normal RN controller's actual records, not closed-client fiction.

    The caller binds this finite evidence set and the externally observed CLI
    completion. Maps evidence is path/inode/hash observation, not VMA-FD proof.
    """
    if teacher_scope not in ('rn12-p4096-none-v1', 'rn12-p256-none-v1'):
        raise ValueError('explicit finite RN12 NONE teacher scope required')
    prefill_rows = 256 if teacher_scope == 'rn12-p256-none-v1' else 4096
    required = {'authority', 'host', 'inspect', 'seal', 'source_after', 'origins',
                'post', 'exchange', 'before', 'after', 'plan', 'runtime', 'request', 'completion', 'toolmap'}
    contract = _owned_comparison_contract()
    if (type(selected) is not dict or set(selected) != {'root', 'files'}
            or not contract.absolute(selected['root']) or type(selected['files']) is not dict
            or set(selected['files']) != required):
        raise ValueError('closed standalone candidate evidence set required')
    root = Path(selected['root'])
    files = selected['files']
    records = {key: _owned_comparison_file(root, files[key], control,
                                         json_root=list if key == 'inspect' else dict)
               for key in sorted(required)}
    a, host, seal = records['authority'], records['host'], records['seal']
    contract.validate_teacher_completion(records['completion'], host_result_sha256=files['host']['sha256'],
                                       authority_sha256=files['authority']['sha256'])
    def retired(state):
        return (type(state) is dict and state.get('Status') == 'exited' and state.get('Running') is False
                and type(state.get('Pid')) is int and state['Pid'] == 0
                and type(state.get('ExitCode')) is int and state['ExitCode'] == 0
                and state.get('OOMKilled') is False)
    inspect_rows = records['inspect']
    if (host.get('primary') is not None or type(host.get('exit')) is not int or host['exit'] != 0
            or any(type(host.get(k)) is not list or any(e is not None for e in host[k])
                   for k in ('cleanup', 'post_errors', 'capture_errors'))
            or not retired(host.get('state')) or type(inspect_rows) is not list or len(inspect_rows) != 1
            or type(inspect_rows[0]) is not dict
            or not retired(inspect_rows[0].get('State')) or inspect_rows[0].get('Id') != host.get('cid')
            or inspect_rows[0].get('Image') != seal.get('App')):
        raise ValueError('standalone owning Host/CLI retirement is not positive')
    source = records['source_after']
    if (source.get('all_match') is not True or type(source.get('files')) is not dict or not source['files']
            or source['files'] != records['toolmap'].get('files')
            or records['toolmap'].get('files') is None):
        raise ValueError('standalone source-after closure missing')
    for identity in source['files'].values():
        if (type(identity) is not dict or set(identity) != {'bytes', 'sha256'}
                or type(identity['bytes']) is not int or not 0 < identity['bytes'] <= 8 << 20
                or type(identity['sha256']) is not str or not SHA.fullmatch(identity['sha256'])):
            raise ValueError('standalone source-after identity malformed')
    plan = records['plan']
    if (plan.get('schema') != 'tensorfold.owned-teacher-plan-v1' or plan.get('mode') != mode
            or a.get('schema') != 'tensorfold.owned-teacher-authority-v1' or a.get('mode') != mode
            or a.get('method') != 'none' or a.get('cohort_complete') is not True
            or a.get('closed') is not True or a.get('finished') is not True or a.get('failed') is not False
            or type(a.get('launch_exit')) is not int or a['launch_exit'] != 0
            or a.get('input_sha256') != files['plan']['sha256']
            or seal.get('teacher_plan_sha256') != files['plan']['sha256']
            or a.get('input_ids') != plan.get('input_ids') or a.get('declared_identity') != plan.get('identity')):
        raise ValueError('standalone completed authority/actual plan differs')
    ids, frames, vocab = a.get('teacher_ids'), a.get('frames'), a.get('vocab')
    if (type(ids) is not list or not 0 < len(ids) <= 64 or type(frames) is not list or len(frames) != len(ids)
            or type(vocab) is not int or not 0 < vocab <= (64 << 20) // 4 - 1024
            or vocab != plan.get('vocab') or any(type(x) is not int or not 0 <= x < vocab for x in ids)
            or type(plan.get('max_frames')) is not int or not len(ids) <= plan['max_frames'] <= 64
            or mode == 'replay' and plan['max_frames'] != len(ids)):
        raise ValueError('standalone bounded full-vocabulary prefix missing')
    profile = a.get('actual_owned_profile', {})
    activation = profile.get('coordinate_profile_identity')
    if type(activation) is not str or not SHA.fullmatch(activation):
        raise ValueError('standalone activation missing')
    semantics = _owned_coordinate_semantics(profile, activation)
    rn = {'kernel_family': 'sm70-paged-owner-rn-ex2-p2051-v1', 'graph_family': 'uncaptured'}
    prepared = semantics.get('prepared_execution', {})
    runtime = records['runtime']
    family = runtime.get('execution', {}).get('execution_families', {})
    if (semantics.get('context') != 262144 or semantics.get('prefill_rows') != prefill_rows
            or semantics.get('target_rows') != 24 or semantics.get('sm70_profile') != 'reference'
            or any(prepared.get(k) != v for k, v in rn.items()) or prepared.get('trial') is not True
            or prepared.get('rn_heads_per_block') != 12 or family.get('effective') != rn
            or family.get('trial') != rn or family.get('native_prepared') is not True
            or family.get('graph_rows') != 0 or runtime.get('context_tokens') != 262144
            or runtime.get('sessions', {}).get('allocated_slots') != 24):
        raise ValueError('standalone actual RN12/Q8/full24 profile differs')
    instance = records['before'].get('server', {}).get('instance_id')
    for stats in (records['before'], records['after']):
        if (not instance or stats.get('server', {}).get('instance_id') != instance
                or stats.get('model', {}).get('kv_dtype') != 'int8'
                or any(type(stats.get('load', {}).get(k)) is not int or stats['load'][k] != 0
                       for k in ('requests_in_flight', 'requests_running', 'requests_waiting'))):
            raise ValueError('standalone same-instance idle Q8 client evidence differs')
    post, exchange, request = records['post'], records['exchange'], records['request']
    body = request.get('body', {})
    tensor = post.get('tensorfold', {})
    if (exchange.get('complete') is not True or exchange.get('cancelled_by_client') is not False
            or exchange.get('status') != 200 or request.get('method') != 'POST'
            or not request.get('url', '').endswith('/v1/completions')
            or body.get('prompt') != a['input_ids'] or body.get('max_tokens') != plan['max_frames']
            or body.get('temperature') != 0 or body.get('draft') is not False or body.get('stream') is not False
            or tensor.get('token_ids') != ids or tensor.get('drafts') is not False
            or any(type(tensor.get(k)) is not int or tensor[k] != 0 for k in ('drafted', 'accepted', 'cached'))
            or post.get('usage', {}).get('prompt_tokens') != len(a['input_ids'])
            or post.get('usage', {}).get('completion_tokens') != len(ids)):
        raise ValueError('standalone one cold NONE POST/prefix differs')
    origins = records['origins']
    workers, libraries = origins.get('workers'), origins.get('libraries')
    if (type(workers) is not list or len(workers) != 8 or type(libraries) is not dict
            or not 0 < len(libraries) <= 64 or {w.get('rank') for w in workers} != set(range(8))):
        raise ValueError('standalone actual all8 mapped native evidence missing')
    startups = []
    for collection in ('startup_receipts', 'rank_receipts'):
        refs = a.get(collection)
        if type(refs) is not list or len(refs) != 8:
            raise ValueError('standalone all8 receipt closure missing')
        for rank, ref in enumerate(refs):
            if ref.get('rank') != rank:
                raise ValueError('standalone receipt ordering differs')
            record = _owned_comparison_file(root / Path(files['authority']['file']).parent, ref, control)
            p = record.get('teacher_profile') if collection == 'startup_receipts' else record.get('actual_owned_profile')
            if (type(p) is not dict or record.get('rank') != rank or p.get('rank') != rank or p.get('world') != 8
                    or p.get('target_banks') != 48 or p.get('mtp_banks') != 0 or p.get('capability') != [7, 0]
                    or _owned_coordinate_semantics(p, activation) != semantics):
                raise ValueError('standalone all8 same-run profile differs')
            if collection == 'startup_receipts':
                if record.get('teacher_server_instance_id') != instance and rank == 0:
                    raise ValueError('standalone rank0 server instance differs')
                startups.append(record)
            elif (record.get('closed') is not True or record.get('finished') is not True
                    or record.get('failed') is not False or record.get('teacher_ids') != ids
                    or record.get('runtime_sources') != a.get('runtime_sources')
                    or len(record.get('frames', [])) != len(frames)
                    or any(f.get('kv_identity') != profile['teacher_semantics']['state_kv_identity']
                           for f in record['frames'])):
                raise ValueError('standalone all8 summary/source/frame closure differs')
    for index, frame in enumerate(frames):
        vector = frame.get('vector', {})
        if (frame.get('step') != index or frame.get('position') != len(a['input_ids']) + index
                or frame.get('teacher_id') != ids[index] or frame.get('full_vocab') != vocab
                or frame.get('kv_identity') != profile['teacher_semantics']['state_kv_identity']
                or frame.get('execution_family') != a.get('owner_family')
                or mode == 'capture' and frame.get('selected_original') != ids[index]
                or type(frame.get('selected_original')) is not int or not 0 <= frame['selected_original'] < vocab
                or vector.get('dtype') != '<f4' or vector.get('shape') != [vocab]
                or vector.get('file') != f'teacher-rank0/frame-{index:06d}.npy'
                or type(vector.get('bytes')) is not int or not 0 < vector['bytes'] <= 4 * vocab + 4096):
            raise ValueError('standalone frame geometry/identity differs')
    native = []
    for rank in range(8):
        worker = next(w for w in workers if w['rank'] == rank)
        paths = worker.get('family_paths')
        if (type(worker.get('PID')) is not int or worker['PID'] <= 0 or type(paths) is not list
                or not paths or len(paths) > 64 or type(worker.get('maps')) is not str):
            raise ValueError('standalone worker native maps incomplete')
        mapped = {}
        for path in paths:
            item = libraries.get(path, {})
            if (path not in worker['maps'] or type(item.get('bytes')) is not int or item['bytes'] <= 0
                    or type(item.get('sha256')) is not str or not SHA.fullmatch(item['sha256'])):
                raise ValueError('standalone mapped path/hash evidence differs')
            mapped[Path(path).name] = item['sha256']
        native.append(mapped)
    return {'root': root, 'records': records, 'authority': a, 'startups': startups,
            'semantics': semantics, 'activation': activation, 'native': native,
            'teacher_root': root / Path(files['authority']['file']).parent}


def _candidate_plan_identity(plan, method, down_method='tensorfold', iq2_source='active-frame', prefill_method='tensorfold',
                             *, teacher_scope='rn12-p4096-none-v1'):
    if teacher_scope not in ('rn12-p4096-none-v1', 'rn12-p256-none-v1'):
        raise ValueError('explicit finite RN12 NONE teacher scope required')
    # NONE's decoder owns24 rows; its expert preparation retains the existing
    #32-row default because the execution-row override is MTP-only.
    rows, prefill_rows = (32, 256) if teacher_scope == 'rn12-p256-none-v1' else (32, 4096)
    base = ['owned-etp4-native-slots-v1', 8, 4, 512, 640, 2560, 11, rows, prefill_rows, False, 'grouped-v1']
    selector = ['owned-onecat-gu-current-down-v1', method, 117976064, 67108864, 'reference-grouped4']
    if down_method != 'tensorfold':
        selector += ['down', down_method]
    if iq2_source != 'active-frame':
        selector += ['iq2-source', iq2_source]
    if prefill_method != 'tensorfold':
        selector += ['prefill', prefill_method]
    suffix = (['grouped-gate-up', 4] if method is None else
              ['owned-onecat-active-frame-current-down-v1', selector])
    if plan != base + suffix:
        raise ValueError('candidate comparison unexpected owned Plan delta')
    return base + ['<explicit-GU-method>']


def _candidate_semantics(side, method, down_method='tensorfold', iq2_source='active-frame', prefill_method='tensorfold',
                         *, teacher_scope='rn12-p4096-none-v1'):
    semantic = {**side['semantics']}
    kv = list(semantic['state_kv_identity'])
    markers = [i for i, item in enumerate(kv) if type(item) is list and item and item[0] == 'owned-expert-native']
    if len(markers) != 1 or len(kv[markers[0]]) != 2:
        raise ValueError('candidate comparison requires one exact owned expert KV marker')
    identity = side['authority']['actual_owned_profile'].get('plan_identity')
    if kv[markers[0]][1] != identity:
        raise ValueError('candidate State/Plan identity linkage differs')
    kv[markers[0]] = ['owned-expert-native', _candidate_plan_identity(
        identity, method, down_method, iq2_source, prefill_method, teacher_scope=teacher_scope)]
    semantic['state_kv_identity'] = kv
    # AUTO HC funding changes with the explicit 117976064-byte resident frame.
    # Preserve every HC semantic field except that independently recorded grant.
    if 'hc_readout_graphs' in semantic:
        hc = dict(semantic['hc_readout_graphs'])
        if type(hc.get('graph_budget_bytes')) is not int or hc['graph_budget_bytes'] <= 0:
            raise ValueError('candidate positive HC grant missing')
        hc['graph_budget_bytes'] = '<actual-positive-AUTO-grant>'
        semantic['hc_readout_graphs'] = hc
    return semantic


def _candidate_teacher_metrics(left, right, control, *, execution_families=None):
    """One numerical implementation for both closed candidate evidence formats."""
    import numpy as np
    a, b = left['authority'], right['authority']
    rows = []
    for old, new in zip(a['frames'], b['frames']):
        if any(old.get(k) != new.get(k) for k in ('step', 'position', 'teacher_id', 'full_vocab', 'role')):
            raise ValueError('candidate frame pairing differs')
        families = (old.get('execution_family'), new.get('execution_family'))
        if (families[0] != families[1] if execution_families is None else families != execution_families):
            raise ValueError('candidate frame pairing differs')
        arrays = []
        for side, frame in ((left, old), (right, new)):
            vector = frame['vector']
            ref = {k: vector[k] for k in ('file', 'bytes', 'sha256')}
            path = _owned_comparison_file(side['teacher_root'], ref, control, maximum=4*a['vocab']+4096, parsed=False)
            arrays.append(_owned_comparison_vector(path, vector, a['vocab'], control))
        x, y = arrays
        unequal = x.view('<u4') != y.view('<u4')
        rows.append({'step': old['step'], 'position': old['position'], 'teacher_id': old['teacher_id'],
                     'reference_original_selected': old['selected_original'], 'candidate_original_selected': new['selected_original'],
                     'reference_forced_id': old['teacher_id'], 'candidate_forced_id': new['teacher_id'],
                     'exact_uint32_bits': not bool(unequal.any()), 'different_uint32_values': int(unequal.sum()),
                     'reference_raw_top1_ties': int(np.count_nonzero(x == x.max())),
                     'candidate_raw_top1_ties': int(np.count_nonzero(y == y.max())), 'finite': True,
                     **compare_logits(x, y, old['teacher_id'])})
        del arrays, x, y, unequal
    count = len(rows)
    base_nll = math.fsum(row['base_nll'] for row in rows) / count
    candidate_nll = math.fsum(row['candidate_nll'] for row in rows) / count
    nll_delta = math.fsum(row['nll_delta'] for row in rows) / count
    def perplexity(value):
        return math.exp(value) if value <= math.log(sys.float_info.max) else None
    quality = {'tokens': count, 'mean_kl': math.fsum(row['kl'] for row in rows) / count,
        'mean_nll_delta': nll_delta, 'reference_mean_nll': base_nll, 'candidate_mean_nll': candidate_nll,
        'reference_perplexity': perplexity(base_nll), 'candidate_perplexity': perplexity(candidate_nll),
        'ppl_ratio': perplexity(nll_delta),
        'top1_agreement': sum(row['base_top1'] == row['candidate_top1'] for row in rows) / count,
        'exact_frames': sum(row['exact_uint32_bits'] for row in rows),
        'scope': 'conditional forced-prefix teacher-token metrics; not held-out accuracy or free-generation acceptance'}
    return rows, quality


def _prepared_expert_side(selected, mode, control, contract):
    side = _owned_comparison_side(selected, mode, control, contract=contract)
    operation, authority = side['inner']['operation'], side['authority']
    contract.validate_operation(operation)
    if ('reference' in operation) != (mode == 'replay'):
        raise ValueError('prepared teacher capture/replay reference discriminant differs')
    if (operation['teacher_scope'] != contract.SCOPE or authority.get('owner_family') != operation['prepared_execution']
            or type(authority.get('vocab')) is not int or authority['vocab'] != 248320
            or authority.get('widths') != [31040] * 8
            or authority.get('reference_sha256') != (None if mode == 'capture' else operation['reference']['authority']['sha256'])):
        raise ValueError('prepared teacher full authority differs from its closed operation')
    if side['content'].get('assets') is None or [row.get('identity') for row in side['content']['assets']] != contract.operation_assets(operation):
        raise ValueError('prepared teacher actual held content closure differs')
    startups, natives, registrations, pids = [], [], [], set()
    source = side['host'].get('source_after', {}).get('source', {})
    observer = source.get('tools/v100_native_contract_driver.py', {})
    if (not source or type(observer.get('bytes')) is not int or not 0 < observer['bytes'] <= 8 << 20
            or not contract.legacy.sha(observer.get('sha256'))):
        raise ValueError('prepared teacher actual frozen operator source proof missing')
    for rank in range(8):
        startup = _owned_comparison_file(side['root'], side['model']['startup_receipts'][rank], control)
        contract.validate_startup(startup, operation)
        windows = startup.get('teacher_windows', {})
        if (startup.get('state_count') != 24 or windows.get('full_resident') is not True
                or windows.get('slots') != 24
                or any(type(windows.get(key)) is not int for key in
                       ('capacity_min', 'capacity_max', 'limit_min', 'limit_max', 'logical_min', 'logical_max'))
                or windows['capacity_min'] != windows['capacity_max'] or windows['limit_min'] != windows['limit_max']
                or windows['capacity_min'] < 262144 or windows['limit_min'] < 262144
                or windows['logical_min'] != 262144 or windows['logical_max'] != 262144):
            raise ValueError('prepared teacher all-rank full-window materialization differs')
        own_profile = startup['teacher_profile']
        registration = own_profile.get('teacher_execution_registration')
        if type(registration) is not dict or set(registration) != {'binding', 'numerics', 'contents'}:
            raise ValueError('prepared teacher lacks issued registration/numerical contents')
        binding = registration['binding']
        if (type(binding) is not dict or type(binding.get('identity')) is not dict
                or type(registration['numerics']) is not dict or type(registration['contents']) is not dict
                or not registration['contents']
                or any(type(name) is not str or not name.startswith(('source/', 'library/'))
                       or type(item) is not dict or set(item) != {'size', 'sha256'}
                       or type(item['size']) is not int or item['size'] < 0 or not contract.legacy.sha(item['sha256'])
                       for name, item in registration['contents'].items())
                or any(type(binding.get(key)) is not int for key in
                       ('content_passes', 'hashes_reused_this_binding', 'verified_input_files', 'verified_input_bytes'))
                # Scope is initial NONE, no DFlash pairing and no rebind: its
                # actual issuer's _capture branch publishes literal reused=0.
                or binding.get('content_passes') != 1 or binding.get('hashes_reused_this_binding') != 0
                or binding.get('shared_immutable_same_host') is not True
                or type(binding.get('verified_input_files')) is not int or binding['verified_input_files'] <= 0
                or type(binding.get('verified_input_bytes')) is not int or binding['verified_input_bytes'] <= 0):
            raise ValueError('prepared teacher lacks fresh complete profile content admission')
        identity = binding['identity']
        if (type(identity.get('context_tokens')) is not int or identity['context_tokens'] != 262144
                or type(identity.get('resident_sessions')) is not int or identity['resident_sessions'] != 24
                or identity.get('method') != 'none' or identity.get('draft') != 'none'
                or identity.get('history_layout') != 'unique'):
            raise ValueError('prepared teacher profile identity is outside the fixed NONE/full24 region')
        summary = _owned_comparison_file(side['root'], side['model']['rank_receipts'][rank], control)
        if (summary.get('schema') != 'tensorfold.owned-teacher-result-v1'
                or summary.get('actual_owned_profile') != own_profile):
            raise ValueError('prepared teacher registration changed before positive close')
        if any(frame.get('execution_family') != operation['prepared_execution'] for frame in summary['frames']):
            raise ValueError('prepared teacher rank frame dropped frozen numerical authority')
        record = _owned_comparison_file(side['root'], side['model']['child_native_records'][rank], control)
        pid = record.get('PID')
        origin = startup.get('teacher_observer', {})
        if (record.get('schema') != 'tensorfold.native-elf-observation-v3'
                or record.get('identity_method') != 'kernel-vma-backed-fd-v1'
                or type(pid) is not int or pid <= 1 or startup.get('teacher_worker_PID') != pid
                or record.get('rank') != rank or record.get('tag') != 'teacher-closed'
                or record.get('operation_limits') != contract.MODEL_OBSERVER_LIMITS
                or record.get('missing_required_prefixes') != []
                or any(record.get('observer', {}).get(key) != observer[key] for key in ('bytes', 'sha256'))
                or type(origin) is not dict or set(origin) != {'path', 'bytes', 'sha256'}
                or not contract.legacy.absolute(origin.get('path'))
                or not origin['path'].endswith('/tools/v100_native_contract_driver.py')
                or any(origin[key] != observer[key] for key in ('bytes', 'sha256'))
                or record.get('access_precondition') != {'uids': [1000]*4, 'gids': [1000]*4, 'NoNewPrivs': '0',
                    'caps': {key: '0000010000000000' for key in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')}}):
            raise ValueError('prepared teacher actual rank/PID/source/VMA access linkage differs')
        pids.add(pid)
        mapped = record.get('mapped_elfs')
        if (type(mapped) is not list or not 0 < len(mapped) <= 64
                or len({item['name'] for item in mapped}) != len(mapped)
                or not any(item['name'].startswith('tf_gguf_') for item in mapped)
                or any(not contract.legacy.absolute(item.get('path')) or type(item.get('bytes')) is not int
                       or item['bytes'] <= 0 or not contract.legacy.sha(item.get('sha256'))
                       or type(item.get('vmas')) is not list or not 0 < len(item['vmas']) <= 256 for item in mapped)):
            raise ValueError('prepared teacher bounded unique mapped native identities differ')
        for logical, item in registration['contents'].items():
            path = logical.removeprefix('library/')
            if logical.startswith('library/') and Path(path).name.startswith('tf_') and not any(
                    saved['path'] == path and saved['sha256'] == item['sha256'] and saved['bytes'] == item['size']
                    for saved in mapped):
                raise ValueError('prepared issued native fingerprint differs from its actual mapped backing object')
        selector = operation['selector']
        if selector['gu'] == 'incumbent':
            if ('onecat_experts' in startup or startup.get('grouped_gate_up') != {'requested_rows': 4, 'actual_rows': 4}):
                raise ValueError('prepared reference lacks actual Grouped4 ownership')
        else:
            binding_record = startup.get('onecat_experts', {})
            frame = binding_record.get('record', {})
            profile = frame.get('profile', {})
            onecat_gu = selector['gu'] != 'tensorfold'
            frame_bytes, lookup = (117964800, 11264) if onecat_gu else (0, 0)
            if (binding_record.get('schema') != 'tensorfold.private-onecat-frame-binding-v1'
                    or binding_record.get('rank') != rank or binding_record.get('world') != 8
                    or binding_record.get('status') != 'prepared-execution-eligible'
                    or frame.get('schema') != 'tensorfold.onecat-active-frame-owner-v1'
                    or frame.get('enabled') is not True or frame.get('closed') is not False
                    or profile.get('policy') != 'owned-onecat-active-frame-current-down-v1'
                    or any(profile.get(key) != selector[name] for key, name in
                           (('method', 'gu'), ('down_method', 'down'), ('iq2_source', 'iq2_source'), ('prefill_method', 'prefill_method')))
                    or profile.get('banks') != 48 or len(frame.get('raw_bindings', [])) != 48
                    or profile.get('frame_slab_bytes') != frame_bytes or profile.get('shared_lookup_bytes') != lookup
                    or profile.get('additional_device_bytes') != frame_bytes + lookup
                    or profile.get('original_raw_retained') is not True or profile.get('baseline_grouped_gate_up_rows') != 4
                    or profile.get('packer_workspace_bytes') != 0 or profile.get('mtp_provider') != 'tensorfold-fp32-mtp'
                    or profile.get('down_format_policy') != {'20': selector['down'], '42': 'tensorfold'}):
                raise ValueError('prepared expert actual typed frame binding differs')
            fields = ['gate_up', 'down'] + (['packer'] if onecat_gu else [])
            if selector['down'] != 'tensorfold':
                fields.append('q8_down')
            if selector['iq2_source'] == 'raw':
                fields.append('raw_iq2_gate_up')
            for field in fields:
                path = binding_record.get('origins', {}).get(field, {}).get('path')
                if not path or not any(item['path'] == path for item in mapped):
                    raise ValueError('prepared expert origin lacks its actual executable VMA proof')
        startups.append(startup)
        natives.append({item['name']: item for item in mapped})
        registrations.append(registration)
    if len(pids) != 8:
        raise ValueError('prepared teacher requires eight distinct actual worker processes')
    client = _owned_comparison_file(side['root'], side['model']['client_report'], control)
    numa = contract.MODE in (_owned_comparison_contract().NUMA_PREPARED_TEACHER_MODE,
                             _owned_comparison_contract().NUMERICAL_TEACHER_MODE)
    expected_client_schema = contract.CLIENT_REPORT_SCHEMA if numa else 'tensorfold.owned-teacher-client-report-v1'
    if (client.get('schema') != expected_client_schema
            or client.get('quality_acceptance') is not False or client.get('P3_valid') is not False
            or type(client.get('prompt_tokens')) is not int or client['prompt_tokens'] != len(authority['input_ids'])
            or type(client.get('completion_tokens')) is not int or client['completion_tokens'] != len(authority['teacher_ids'])
            or type(client.get('server_instance_id')) is not str or not client['server_instance_id']
            or client['server_instance_id'] != startups[0].get('teacher_server_instance_id')
            or authority.get('actual_owned_profile') != startups[0].get('teacher_profile')):
        raise ValueError('prepared teacher HTTP instance/authority lacks its original rank0 linkage')
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    numerics = [row['numerics'] for row in registrations]
    for registration in registrations:
        identity = registration['binding']['identity']
        contents = registration['contents']
        if (identity != registrations[0]['binding']['identity'] or contents != registrations[0]['contents']
                or identity.get('numerics') != digest({'ranks': numerics})
                or any(identity.get(key) != digest({label: {name: item for name, item in contents.items()
                      if name.startswith(prefix)}}) for key, label, prefix in
                       (('source', 'files', 'source/'), ('kernel_build', 'libraries', 'library/')))):
            raise ValueError('prepared registration fingerprints do not bind the observed raw numerics/contents')
    for frame in authority['frames']:
        if frame.get('execution_family') != operation['prepared_execution']:
            raise ValueError('prepared authority frame lacks its full frozen specification')
    if numa:
        contract.validate_result_files(side['root'], side['model'], operation)
    side.update(startups=startups, native=natives, registrations=registrations,
                semantics=side['stable_semantics'], teacher_root=side['root'] / 'teacher')
    return side


def _compare_prepared_expert_teacher(reference, candidate, *, method, down_method, iq2_source, prefill_method, control,
                                     contract=None):
    contract = _owned_comparison_contract().prepared_teacher_contract() if contract is None else contract
    numa = contract.MODE in (_owned_comparison_contract().NUMA_PREPARED_TEACHER_MODE,
                             _owned_comparison_contract().NUMERICAL_TEACHER_MODE)
    left = _prepared_expert_side(reference, 'capture', control, contract)
    right = _prepared_expert_side(candidate, 'replay', control, contract)
    a, b = left['authority'], right['authority']
    old_op, new_op = left['inner']['operation'], right['inner']['operation']
    selected = {'gu': method, 'down': down_method, 'iq2_source': iq2_source, 'prefill_method': prefill_method}
    if old_op['selector'] != contract.selectors()[0] or new_op['selector'] != selected:
        raise ValueError('prepared comparison declared expert selector differs')
    if b.get('reference_sha256') != left['authority_ref']['sha256']:
        raise ValueError('prepared replay does not name this exact successful incumbent capture')
    imported = new_op['reference']
    expected_reference = {'teacher/teacher-authority.json': {
        key: left['authority_ref'][key] for key in ('bytes', 'sha256')}}
    for collection in ('rank_receipts', 'startup_receipts'):
        for row in a[collection]:
            expected_reference['teacher/' + row['file']] = {key: row[key] for key in ('bytes', 'sha256')}
    for frame in a['frames']:
        vector = frame['vector']
        expected_reference['teacher/' + vector['file']] = {key: vector[key] for key in ('bytes', 'sha256')}
    if numa:
        expected_reference.update(contract.reference_result_files(left['model']))
    if (imported['files'] != expected_reference
            or any(imported['authority'][key] != left['authority_ref'][key] for key in ('bytes', 'sha256'))
            or any(imported['host_result'][key] != reference['host_result'][key] for key in ('bytes', 'sha256'))
            or any(imported['completion'][key] != reference['completion'][key] for key in ('bytes', 'sha256'))):
        raise ValueError('prepared replay imported a different actual capture/CLI/complete reference closure')
    if left['inner']['image_id_declared'] != right['inner']['image_id_declared']:
        raise ValueError('prepared comparison requires the same immutable application image')
    for key in ('model_assets', 'frontend_assets', 'frontend_root', 'model_entrypoint', 'execution_manifest',
                'case_authority', 'prepared_execution', 'teacher_scope', *(['numa'] if numa else [])):
        if old_op[key] != new_op[key]:
            raise ValueError('prepared comparison unrelated closed input differs: ' + key)
    for key in ('source_manifest', 'source_fixture', 'source', 'image_id'):
        old = left['host'].get('source_after', {}).get(key)
        if old is None or old != right['host'].get('source_after', {}).get(key):
            raise ValueError('prepared comparison canonical source/image fingerprint differs: ' + key)
    for key in ('input_ids', 'vocab', 'widths', 'teacher_ids', 'owner_family', 'runtime_sources'):
        if a.get(key) != b.get(key):
            raise ValueError('prepared comparison frozen prefix/installed source differs: ' + key)
    for key in ('case_id', 'model_sha256', 'frontend_sha256', 'profile_sha256'):
        if not a.get('declared_identity', {}).get(key) or a['declared_identity'][key] != b['declared_identity'].get(key):
            raise ValueError('prepared comparison case/model/frontend/profile authority differs')
    selected_method = None if method == 'incumbent' else method
    if (_candidate_semantics(left, None, teacher_scope=contract.SCOPE) != _candidate_semantics(
            right, selected_method, down_method, iq2_source, prefill_method, teacher_scope=contract.SCOPE)):
        raise ValueError('prepared comparison changed unrelated State/row/RN semantics')
    prefixes = ('tf_onecat_gu_', 'tf_onecat_pack_') if method not in ('tensorfold', 'incumbent') else ()
    if down_method != 'tensorfold':
        prefixes += ('tf_onecat_down_',)
    producers = ('runtime_cli_sha256', 'dense_implementation', 'half_partial_policy', 'private_router_policy', 'empty_prefix_policy')
    for rank in range(8):
        old, new = left['native'][rank], right['native'][rank]
        if any(name not in new or new[name]['sha256'] != item['sha256'] for name, item in old.items()):
            raise ValueError('prepared comparison changed an incumbent native fingerprint')
        if any(not name.startswith(prefixes) for name in new.keys() - old.keys()):
            raise ValueError('prepared comparison added an unrelated foreign native library')
        if any(left['startups'][rank].get(key) != right['startups'][rank].get(key) for key in producers):
            raise ValueError('prepared comparison changed an unrelated producer policy')
        old_reg, new_reg = left['registrations'][rank], right['registrations'][rank]
        old_binding, new_binding = old_reg['binding'], new_reg['binding']
        if any(old_binding.get(key) != new_binding.get(key) for key in
               ('manifest_sha256', 'verified_input_files', 'verified_input_bytes', 'content_passes')):
            raise ValueError('prepared comparison actual complete profile content differs')
        if ({k: v for k, v in old_binding['identity'].items() if k not in ('numerics', 'kernel_build')}
                != {k: v for k, v in new_binding['identity'].items() if k not in ('numerics', 'kernel_build')}):
            raise ValueError('prepared comparison unrelated profile identity fingerprint differs')
        old_numbers, new_numbers = dict(old_reg['numerics']), dict(new_reg['numerics'])
        if old_numbers.get('owned_expert_native') != left['startups'][rank]['plan_identity'] or new_numbers.get(
                'owned_expert_native') != right['startups'][rank]['plan_identity']:
            raise ValueError('prepared registration numerical Plan differs from actual typed owner')
        old_numbers['owned_expert_native'] = _candidate_plan_identity(old_numbers['owned_expert_native'], None, teacher_scope=contract.SCOPE)
        new_numbers['owned_expert_native'] = _candidate_plan_identity(new_numbers['owned_expert_native'], selected_method,
            down_method, iq2_source, prefill_method, teacher_scope=contract.SCOPE)
        if old_numbers != new_numbers:
            raise ValueError('prepared comparison changed numerical inputs beyond the explicit expert selector')
        old_files, new_files = old_reg['contents'], new_reg['contents']
        if any(new_files.get(name) != item for name, item in old_files.items()):
            raise ValueError('prepared comparison changed issued incumbent source/library contents')
        for name in new_files.keys() - old_files.keys():
            path = name.removeprefix('library/')
            if (not name.startswith('library/') or not Path(path).name.startswith(prefixes)
                    or not any(item['path'] == path and item['sha256'] == new_files[name]['sha256']
                               and item['bytes'] == new_files[name]['size'] for item in new.values())):
                raise ValueError('prepared profile library delta lacks its exact selected VMA fingerprint')
    rows, metrics = _candidate_teacher_metrics(left, right, control)
    return {'schema': ('tensorfold.prepared-expert-teacher-numa-comparison-v2' if numa else
                       'tensorfold.prepared-expert-teacher-comparison-v1'), 'complete': True,
            'status': 'DIAGNOSTIC_ONLY_REQUIRES_FULL_MODEL_QUALITY_REVIEW', 'qualification': False,
            'quality_acceptance': False, 'P3_valid': False, 'teacher_scope': contract.SCOPE,
            'selector': selected, 'reference': reference, 'candidate': candidate, 'frames': rows,
            'quality_metrics': metrics, 'image_id': left['inner']['image_id_declared'],
            'runtime_sources': [a['runtime_sources'], b['runtime_sources']],
            'actual_profiles': [a['actual_owned_profile'], b['actual_owned_profile']],
            **({'numa_evidence': [left['model']['numa'], right['model']['numa']]} if numa else {}),
            'coverage': {'frames': len(rows), 'vocab': a['vocab'], 'case_id': a['declared_identity']['case_id'],
                         'role_frames': {role: sum(frame.get('role') == role for frame in a['frames'])
                                         for role in ('prefill', 'decode')},
                         'prefill_rows': 256, 'decoder_target_rows': 24, 'expert_prepared_rows': 32,
                         'method': 'none', 'forced_prefix_only': True, 'MTP4_graph_quality': False,
                         'held_out_accuracy': False, 'universal_row_equivalence_proved': False},
            'scope': 'Closed P256/E24/NONE RN12 expert-only forced-prefix diagnostics; no MTP4/graph/P3/held-out acceptance'}


def compare_owned_candidate_teacher(reference, candidate, *, method, read_timeout_s,
                                    down_method='tensorfold', iq2_source='active-frame', prefill_method='tensorfold',
                                    teacher_scope='rn12-p4096-none-v1', numa=False):
    """Explicit expert-method forced-prefix diagnostic, never matched-control acceptance."""
    if method not in ('tensorfold-fp32-epilogue', 'onecat-half-epilogue', 'tensorfold') and not (
            teacher_scope == 'rn12-p256-none-v1' and method == 'incumbent'):
        raise ValueError('explicit OneCat GU comparison method required')
    if (down_method not in ('tensorfold', 'onecat-q8-slots') or iq2_source not in ('active-frame', 'raw')
            or method == 'tensorfold' and iq2_source != 'active-frame'
            or prefill_method not in ('tensorfold', 'candidate')):
        raise ValueError('explicit applicable target down/IQ2 source required')
    if method == 'incumbent' and (down_method, iq2_source, prefill_method) != ('tensorfold', 'active-frame', 'tensorfold'):
        raise ValueError('prepared matched control must retain the exact incumbent selector')
    if type(numa) is not bool or numa and teacher_scope != 'rn12-p256-none-v1':
        raise ValueError('NUMA quality requires its exact explicit prepared NONE region')
    if (type(read_timeout_s) not in (int, float) or not 0 < read_timeout_s <= sys.maxsize
            or not math.isfinite(read_timeout_s)):
        raise ValueError('candidate comparison requires positive finite read_timeout_s excluding bool')
    control = _teacher_read_control(read_timeout_s)
    if teacher_scope == 'rn12-p256-none-v1':
        return _compare_prepared_expert_teacher(reference, candidate, method=method,
            down_method=down_method, iq2_source=iq2_source, prefill_method=prefill_method, control=control,
            contract=_owned_comparison_contract().prepared_teacher_contract(numa=True) if numa else None)
    left = _candidate_teacher_side(reference, 'capture', control, teacher_scope=teacher_scope)
    right = _candidate_teacher_side(candidate, 'replay', control, teacher_scope=teacher_scope)
    a, b = left['authority'], right['authority']
    if b.get('reference_sha256') != reference['files']['authority']['sha256']:
        raise ValueError('candidate replay does not name this capture')
    for key in ('App', 'source_uid'):
        if not left['records']['seal'].get(key) or left['records']['seal'][key] != right['records']['seal'].get(key):
            raise ValueError('candidate immutable App/source differs: ' + key)
    for key in ('input_ids', 'vocab', 'widths', 'teacher_ids', 'owner_family'):
        if a.get(key) != b.get(key):
            raise ValueError('candidate frozen teacher pairing differs: ' + key)
    for key in ('Mounts',):
        def assets(side):
            return sorted((m.get('Source'), m.get('Destination'), m.get('RW'))
                          for m in side['records']['inspect'][0].get(key, [])
                          if m.get('Destination', '').startswith(('/models/', '/frontend')))
        if not assets(left) or assets(left) != assets(right):
            raise ValueError('candidate actual model/frontend mounts differ')
    for field in ('identity', 'manifest_sha256', 'verified_input_files', 'verified_input_bytes'):
        x = left['records']['runtime'].get('capacity_plan', {}).get('execution_profile_binding', {}).get(field)
        y = right['records']['runtime'].get('capacity_plan', {}).get('execution_profile_binding', {}).get(field)
        if x is None or x != y:
            raise ValueError('candidate actual content/profile binding differs: ' + field)
    for side in (left, right):
        bound = side['records']['runtime'].get('capacity_plan', {}).get('execution_profile_binding', {})
        if bound.get('content_passes') != 1 or bound.get('hashes_reused_this_binding') != 0:
            raise ValueError('candidate fresh full content pass missing')
    for key in ('case_id', 'model_sha256', 'frontend_sha256'):
        if not a.get('declared_identity', {}).get(key) or a['declared_identity'][key] != b.get('declared_identity', {}).get(key):
            raise ValueError('candidate declared model/frontend/case differs: ' + key)
    for key in ('engine', 'multi', 'decode', 'sampling'):
        value = a.get('runtime_sources', {}).get(key)
        if type(value) is not str or not SHA.fullmatch(value) or value != b.get('runtime_sources', {}).get(key):
            raise ValueError('candidate installed math source differs: ' + key)
    if (_candidate_semantics(left, None, teacher_scope=teacher_scope)
            != _candidate_semantics(right, method, down_method, iq2_source, prefill_method,
                                    teacher_scope=teacher_scope)):
        raise ValueError('candidate unrelated actual State/RN/HC semantics differ')
    producers = ('runtime_cli_sha256', 'dense_implementation', 'half_partial_policy', 'private_router_policy', 'empty_prefix_policy')
    for side, selected_method in ((left, None), (right, method)):
        first = side['startups'][0]
        for rank, startup in enumerate(side['startups']):
            _candidate_plan_identity(startup.get('plan_identity'), selected_method,
                                     down_method if selected_method is not None else 'tensorfold',
                                     iq2_source if selected_method is not None else 'active-frame',
                                     prefill_method if selected_method is not None else 'tensorfold',
                                     teacher_scope=teacher_scope)
            if any(startup.get(k) != first.get(k) for k in producers):
                raise ValueError('candidate all8 producer selection differs')
            if selected_method is None:
                if 'onecat_experts' in startup or startup.get('grouped_gate_up') != {'requested_rows': 4, 'actual_rows': 4}:
                    raise ValueError('candidate reference is not actual Grouped4')
            else:
                binding = startup.get('onecat_experts', {})
                record = binding.get('record', {})
                p = record.get('profile', {})
                onecat_gu = method != 'tensorfold'
                frame_bytes, lookup_bytes = (117964800, 11264) if onecat_gu else (0, 0)
                if (binding.get('schema') != 'tensorfold.private-onecat-frame-binding-v1'
                        or binding.get('rank') != rank or binding.get('world') != 8
                        or binding.get('status') != 'prepared-execution-eligible'
                        or record.get('schema') != 'tensorfold.onecat-active-frame-owner-v1'
                        or record.get('enabled') is not True or record.get('closed') is not False
                        or p.get('policy') != 'owned-onecat-active-frame-current-down-v1' or p.get('method') != method
                        or p.get('down_method') != down_method or p.get('iq2_source') != iq2_source
                        or p.get('prefill_method') != prefill_method
                        or p.get('down_format_policy') != {'20': down_method, '42': 'tensorfold'}
                        or p.get('mtp_provider') != 'tensorfold-fp32-mtp'
                        or p.get('banks') != 48 or p.get('additional_device_bytes') != frame_bytes + lookup_bytes
                        or p.get('frame_slab_bytes') != frame_bytes or p.get('shared_lookup_bytes') != lookup_bytes
                        or p.get('packer_workspace_bytes') != 0 or p.get('original_raw_retained') is not True
                        or p.get('baseline_grouped_gate_up_rows') != 4 or len(record.get('raw_bindings', [])) != 48):
                    raise ValueError('candidate explicit funded active-frame GU binding missing')
                fields = ['gate_up', 'down']
                if onecat_gu:
                    fields.append('packer')
                if down_method != 'tensorfold':
                    fields.append('q8_down')
                if iq2_source != 'active-frame':
                    fields.append('raw_iq2_gate_up')
                for field in fields:
                    path = binding.get('origins', {}).get(field, {}).get('path', '')
                    if not path or Path(path).name not in side['native'][rank]:
                        raise ValueError('candidate native GU/packer not observed in actual worker maps')
    if any(left['startups'][0].get(k) != right['startups'][0].get(k) for k in producers):
        raise ValueError('candidate unrelated producer selection differs')
    for old, new in zip(left['native'], right['native']):
        if any(new.get(name) != digest for name, digest in old.items()):
            raise ValueError('candidate changed incumbent native dependency')
        prefixes = ('tf_onecat_gu_', 'tf_onecat_pack_') if method != 'tensorfold' else ()
        if down_method != 'tensorfold':
            prefixes += ('tf_onecat_down_',)
        if any(not name.startswith(prefixes) for name in new.keys() - old.keys()):
            raise ValueError('candidate unexpected native dependency delta')
    rows, quality = _candidate_teacher_metrics(left, right, control)
    return {'schema': 'tensorfold.owned-candidate-teacher-comparison-v1', 'complete': True,
            'status': 'DIAGNOSTIC_ONLY_REQUIRES_ANALYTICAL_AND_FULL_MODEL_QUALITY_REVIEW',
            'qualification': False, 'quality_acceptance': False, 'P3_valid': False, 'method': method,
            'down_method': down_method, 'iq2_source': iq2_source, 'prefill_method': prefill_method,
            'teacher_scope': teacher_scope, 'quality_metrics': quality,
            'reference': reference, 'candidate': candidate, 'frames': rows,
            'image_id': left['records']['seal']['App'], 'runtime_sources': [a['runtime_sources'], b['runtime_sources']],
            'coordinate_activations': [left['activation'], right['activation']],
            'actual_profiles': [a['actual_owned_profile'], b['actual_owned_profile']],
            'native_sets_by_rank': [left['native'], right['native']],
            'coverage': {'frames': len(rows), 'vocab': a['vocab'], 'case_id': a['declared_identity']['case_id'],
                         'forced_prefix_only': True, 'free_generation_quality': False, 'analytical_bound_proved': False},
            'scope': 'same-App RN12 control versus explicit GU/down/IQ2 selection; no tolerance, acceptance or timing claim'}


def _compare_owned_candidate_teacher_main(argv):
    parser = argparse.ArgumentParser(description='Offline RN/OneCat GU diagnostics; no acceptance margins')
    parser.add_argument('--comparison-input', type=Path, required=True)
    parser.add_argument('--expected-input-sha', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    raw = _owned_teacher_read(args.comparison_input, 8 << 20)
    if not SHA.fullmatch(args.expected_input_sha) or hashlib.sha256(raw).hexdigest() != args.expected_input_sha:
        raise ValueError('candidate comparison input SHA differs')
    data = _owned_teacher_object(raw)
    if data.get('schema') == 'tensorfold.numerical-teacher-comparison-input-v3':
        if set(data) != {'schema', 'reference', 'candidate', 'read_timeout_s'}:
            raise ValueError('closed numerical teacher comparison input required')
        if __package__:
            from . import v100_numerical_teacher
        else:
            import v100_numerical_teacher
        control = _teacher_read_control(data['read_timeout_s'])
        report = v100_numerical_teacher.compare(data['reference'], data['candidate'], control=control)
        report['comparison_input_sha256'] = args.expected_input_sha
        args.output_dir.mkdir(mode=0o700)
        _owned_teacher_json(args.output_dir / 'comparison.json', report, 8 << 20, control)
        return 0
    required = {'schema', 'reference', 'candidate', 'method', 'read_timeout_s'}
    numa = data.get('schema') == 'tensorfold.prepared-expert-teacher-numa-comparison-input-v2'
    prepared = data.get('schema') in ('tensorfold.prepared-expert-teacher-comparison-input-v1',
                                     'tensorfold.prepared-expert-teacher-numa-comparison-input-v2')
    if prepared:
        required = required | {'teacher_scope'}
    if (type(data) is not dict or not required <= set(data) <= required | {'down_method', 'iq2_source', 'prefill_method'}
            or data['schema'] != ('tensorfold.prepared-expert-teacher-numa-comparison-input-v2' if numa else
                                  'tensorfold.prepared-expert-teacher-comparison-input-v1' if prepared else
                                  'tensorfold.owned-candidate-teacher-comparison-input-v1')
            or prepared and data['teacher_scope'] != 'rn12-p256-none-v1'):
        raise ValueError('closed candidate comparison input required')
    report = compare_owned_candidate_teacher(data['reference'], data['candidate'], method=data['method'],
                                              read_timeout_s=data['read_timeout_s'],
                                              down_method=data.get('down_method', 'tensorfold'),
                                              iq2_source=data.get('iq2_source', 'active-frame'),
                                              prefill_method=data.get('prefill_method', 'tensorfold'),
                                              teacher_scope=data.get('teacher_scope', 'rn12-p4096-none-v1'), numa=numa)
    report['comparison_input_sha256'] = args.expected_input_sha
    args.output_dir.mkdir(mode=0o700)
    _owned_teacher_json(args.output_dir / 'comparison.json', report, 8 << 20,
                        _teacher_read_control(data['read_timeout_s']))
    return 0  # Valid diagnostics, including nonexact results, are not qualification.


def teacher_frames(case, first_logits, advance):
    """Consume the frozen previous teacher token, never the candidate's choice."""
    current = first_logits
    for step, token in enumerate(case.teacher_ids):
        if step:
            current = advance(case.teacher_ids[step - 1])
        yield step, token, current


def _owned_teacher_read(path, maximum):
    if __package__:
        from .gb10_io import owned_descriptor, borrowed_stream, stamp
    else:
        from gb10_io import owned_descriptor, borrowed_stream, stamp
    with owned_descriptor(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC) as fd:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError('bounded immutable teacher input required')
        with borrowed_stream(fd) as stream:
            raw = stream.read(before.st_size)
        if (len(raw) != before.st_size or stamp(os.fstat(fd)) != stamp(before)
                or stamp(os.stat(path, follow_symlinks=False)) != stamp(before)):
            raise ValueError('teacher input changed while reading')
    return raw


def _owned_teacher_hash(path, expected, maximum, control=None):
    """Verify declared immutable raw extent; do not materialize a vocab vector."""
    if type(expected) is not int or not 0 < expected <= maximum:
        raise ValueError('bounded declared reference vector extent required')
    if __package__:
        from .gb10_io import owned_descriptor, borrowed_stream, stamp
    else:
        from gb10_io import owned_descriptor, borrowed_stream, stamp
    digest = hashlib.sha256()
    if control is not None:
        control.check()
    with owned_descriptor(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC) as fd:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected:
            raise ValueError('reference vector extent/type differs')
        with borrowed_stream(fd) as stream:
            remaining = expected
            while remaining:
                if control is not None:
                    control.check()
                chunk = stream.read(min(65536, remaining))
                if control is not None:
                    control.check()
                if not chunk:
                    raise ValueError('truncated reference vector')
                digest.update(chunk)
                remaining -= len(chunk)
        if stamp(os.fstat(fd)) != stamp(before) or stamp(os.stat(path, follow_symlinks=False)) != stamp(before):
            raise ValueError('reference vector changed while hashing')
    return digest.hexdigest()


def _owned_teacher_json(path, value, maximum, control=None):
    """Exclusive bounded JSON publication with existing primary-safe IO owners."""
    if __package__:
        from .gb10_io import opened_file, owned_descriptor
    else:
        from gb10_io import opened_file, owned_descriptor
    written = 0
    with opened_file(path, 'x', encoding='utf-8') as stream:
        for text in json.JSONEncoder(indent=1, ensure_ascii=True, allow_nan=False).iterencode(value):
            if control is not None:
                control.check()
            if len(text) > maximum - written:
                raise ValueError('teacher JSON exceeds its prepaid schema bound')
            if stream.write(text) != len(text):
                raise OSError('short teacher JSON write')
            written += len(text)
            if control is not None:
                control.check()
        stream.flush()
        os.fsync(stream.fileno())
        if control is not None:
            control.check()
    with owned_descriptor(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC) as directory:
        os.fsync(directory)
        if control is not None:
            control.check()


def owned_teacher_resources(input_bytes, reference_bytes, vocab, frames, *, policy=None):
    from tensorfold.families.qwen4_exp.cuda.retention_resources import (
        _json_payload_policy, _json_parse_payload, _json_tree_payload,
        _python_layout, _streamed_json_encoding_payload,
    )
    policy = _json_payload_policy() if policy is None else policy
    # Same selected-parser/tree policy as the existing private qualification
    # input. A second tree pays object_pairs_hook pair-list/tuple plus partial
    # result-dict overlap; the parsed lists/values are borrowed, not deepcopied.
    inputs = sum(_json_parse_payload(size, policy)['total_bytes'] + _json_tree_payload(size, policy)
                 for size in (input_bytes, reference_bytes) if size)
    receipt = 262144 + 2 * input_bytes + 4096 * frames
    authority = receipt + 16384
    if authority > 8 << 20:
        raise ValueError('teacher receipts exceed the existing bounded structured JSON envelope')
    vector = 4 * vocab
    per_rank = (inputs + _json_tree_payload(receipt, policy)
                + _streamed_json_encoding_payload(receipt, _python_layout())['total_bytes']
                + policy.frame_work + 3 * vector + 262144)
    # Parent retains the DATA and first rank while processing one other rank
    # at a time, then publishes its authority after all workers have retired.
    parent = (inputs + _json_parse_payload(receipt, policy)['total_bytes']
              + 3 * _json_tree_payload(receipt, policy)
              + _streamed_json_encoding_payload(authority, _python_layout())['total_bytes']
              + policy.frame_work + 262144)
    if 8 * per_rank + parent > sys.maxsize:
        raise ValueError('teacher host obligation exceeds the selected size domain')
    return {'global_host_obligation_bytes': 8 * per_rank + parent, 'per_rank_payload_bytes': per_rank,
            'parent_payload_bytes': parent, 'authority_bytes_limit': authority,
            'runtime_policy': policy.identity, 'input_bytes': input_bytes, 'reference_bytes': reference_bytes,
            'receipt_bytes_limit': receipt, 'frame_payload_bytes': vector, 'max_frame_file_bytes': vector + 4096,
            'max_total_frame_file_bytes': frames * (vector + 4096), 'additional_project_device_bytes': 0,
            'scope': 'source/object/copy/writer payload; allocator/RSS distinct; diagnostic timings invalid P3'}


def _owned_runtime_sources(engine):
    names = {'engine': type(engine).__module__, 'multi': 'tensorfold.families.qwen4_exp.cuda.multi',
             'decode': 'tensorfold.families.qwen4_exp.cuda.decode', 'sampling': 'tensorfold.cuda.sampling',
             'teacher': __name__}
    result = {}
    for role, name in names.items():
        path = Path(sys.modules[name].__file__)
        size = path.stat().st_size
        result[role] = _owned_teacher_hash(path, size, size)
    return result


def _owned_teacher_object(raw):
    return _owned_teacher_parse(raw, root_type=dict)


def _owned_teacher_parse(raw, *, root_type):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate teacher JSON key')
            result[key] = value
        return result
    def finite(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('nonfinite teacher JSON')
        return result
    result = json.loads(raw, object_pairs_hook=pairs, parse_float=finite,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite teacher JSON')))
    if type(result) is not root_type:
        raise ValueError('teacher JSON must be an array' if root_type is list else 'teacher JSON must be an object')
    return result


def seal_owned_teacher(output, prepared, launch_exit):
    """Only the existing supervisor's positive all-rank retirement permits this seal."""
    if type(launch_exit) is not int or launch_exit != 0:
        raise ValueError('teacher authority requires successful existing cohort retirement')
    root, first, receipts, startup_receipts = Path(output), None, [], []
    data, limits = prepared['data'], prepared['resources']
    control = _teacher_read_control(data['read_timeout_s'])
    for rank in range(8):
        control.check()
        name = f'teacher-rank{rank}/summary.json'
        raw = _owned_teacher_read(root / name, limits['receipt_bytes_limit'])
        record = _owned_teacher_object(raw)
        control.check()
        if (record.get('schema') != 'tensorfold.owned-teacher-result-v1' or type(record.get('rank')) is not int
                or record['rank'] != rank or record.get('mode') != data['mode'] or record.get('method') != 'none'
                or record.get('closed') is not True or record.get('finished') is not True or record.get('failed') is not False
                or record.get('input_sha256') != prepared['input_sha256'] or record.get('declared_identity') != data['identity']
                or record.get('input_ids') != data['input_ids'] or record.get('vocab') != data['vocab']
                or record.get('qualification') is not False or record.get('P3_valid') is not False):
            raise ValueError('teacher rank is incomplete or differs from the declared run')
        profile = record.get('actual_owned_profile', {})
        if (any(type(profile.get(key)) is not int for key in ('rank', 'world', 'target_banks', 'mtp_banks'))
                or profile.get('rank') != rank or profile.get('world') != 8 or profile.get('target_banks') != 48
                or profile.get('mtp_banks') != 0 or profile.get('capability') != [7, 0]):
            raise ValueError('teacher rank lacks its actual SC117 none/TP8 profile binding')
        semantic = profile.get('teacher_semantics')
        semantic_keys = {'state_kv_identity', 'context', 'cache_slots', 'prefill_rows',
                         'target_rows', 'vocab_widths', 'sm70_profile', 'prepared_execution'}
        if (type(semantic) is not dict or set(semantic) not in (semantic_keys, semantic_keys | {'hc_readout_graphs'})
                or type(semantic['context']) is not int or semantic['context'] != data['context']
                or type(semantic['cache_slots']) is not int or semantic['cache_slots'] < data['context']
                or type(semantic['target_rows']) is not int or semantic['target_rows'] < 9
                or type(semantic['prefill_rows']) is not int or semantic['prefill_rows'] <= 0
                or semantic['sm70_profile'] != 'reference' or semantic['vocab_widths'] != record.get('widths')
                or semantic['prepared_execution'] != record.get('owner_family')
                or type(semantic['state_kv_identity']) is not list
                or ['logical-context-v1', data['context']] not in semantic['state_kv_identity']
                or ['coordinate-profile-v1', profile.get('coordinate_profile_identity')] not in semantic['state_kv_identity']):
            raise ValueError('teacher actual KV/context/position/execution semantic binding differs')
        if 'hc_readout_graphs' in semantic:
            hc, hc_record = semantic['hc_readout_graphs'], profile.get('hc_readout_graph_record', {})
            geometry = _hc_graph_geometry(hc, semantic['target_rows'])
            if (set(hc) != {'policy', 'catalogue_keys', 'sources', 'graph_budget_bytes'} | set(geometry)
                    or hc['catalogue_keys'] != 97
                    or type(hc['graph_budget_bytes']) is not int or hc['graph_budget_bytes'] <= 0
                    or type(hc['sources']) is not dict or set(hc['sources']) != {'hc_readout_graphs', 'hc_plans'}
                    or any(type(value) is not str or not SHA.fullmatch(value) for value in hc['sources'].values())
                    or hc_record.get('captured') is not True or hc_record.get('captured_keys') != 97
                    or hc_record.get('policy') != hc['policy']
                    or any(hc_record.get(key) != value for key, value in geometry.items())
                    or hc_record.get('graph_budget_bytes') != hc['graph_budget_bytes']):
                raise ValueError('teacher HC readout graph policy lacks its actual source/admission record')
        ids, frames, widths, sources = (record.get(key) for key in ('teacher_ids', 'frames', 'widths', 'runtime_sources'))
        if (type(ids) is not list or not 0 < len(ids) <= data['max_frames']
                or any(type(token) is not int or not 0 <= token < data['vocab'] for token in ids)
                or type(frames) is not list or len(frames) != len(ids)
                or type(widths) is not list or len(widths) != 8 or any(type(n) is not int or n <= 0 for n in widths)
                or sum(widths) != data['vocab'] or type(sources) is not dict
                or set(sources) != {'engine', 'multi', 'decode', 'sampling', 'teacher'}
                or any(type(value) is not str or not SHA.fullmatch(value) for value in sources.values())):
            raise ValueError('teacher rank vocabulary/frames/source schema differs')
        if data['mode'] == 'replay' and ids != prepared['teacher_ids']:
            raise ValueError('replay cohort did not consume exactly its frozen teacher sequence')
        for index, frame in enumerate(frames):
            if (frame.get('step') != index or frame.get('position') != len(data['input_ids']) + index
                    or frame.get('teacher_id') != ids[index] or frame.get('full_vocab') != data['vocab']):
                raise ValueError('teacher frame does not describe its frozen prefix')
        if first is None:
            first = record
        if (record['teacher_ids'] != first['teacher_ids'] or record['widths'] != first['widths']
                or record['runtime_sources'] != first['runtime_sources'] or record['owner_family'] != first['owner_family']
                or profile.get('teacher_semantics') != first['actual_owned_profile'].get('teacher_semantics')
                or len(record['frames']) != len(first['frames']) or not record['frames']):
            raise ValueError('teacher ranks disagree on prefix/profile/source/frame identities')
        for frame, expected in zip(record['frames'], first['frames']):
            if any(frame[key] != expected[key] for key in ('step', 'position', 'teacher_id', 'selected_original', 'full_vocab',
                                                          'state_pos', 'slot_generation', 'history_slot', 'role', 'execution_family')):
                raise ValueError('teacher rank frame differs from the completed reference')
        receipts.append({'rank': rank, 'file': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        del raw
        startup_name = f'rank-{rank}.json'
        startup_raw = _owned_teacher_read(root / startup_name, limits['receipt_bytes_limit'])
        startup = _owned_teacher_object(startup_raw)
        control.check()
        if (startup.get('schema') != 'tensorfold.private-owned-qualification-v1' or type(startup.get('rank')) is not int
                or startup.get('rank') != rank
                or startup.get('world') != 8 or startup.get('mtp_banks') != 0
                or startup.get('history_host_obligation') != {
                    'global_bytes': limits['global_host_obligation_bytes'], 'input_sha256': prepared['input_sha256']}):
            raise ValueError('teacher startup binding differs from original admitted input/profile')
        if startup.get('teacher_profile') != profile:
            raise ValueError('teacher actual profile differs from its immutable startup binding')
        # These are immutable producer/selection identities, not per-rank
        # allocation/free-memory, timings or local partition observations.
        producer = {key: startup.get(key) for key in ('tool_sha256', 'runtime_cli_sha256', 'dense_implementation',
                    'half_partial_policy', 'private_router_policy', 'empty_prefix_policy', 'grouped_gate_up')}
        producer['reference_tiles_requested'] = startup.get('reference_tile_preparation', {}).get('requested')
        if rank == 0:
            producer_identity = producer
        elif producer != producer_identity:
            raise ValueError('teacher immutable producer/execution selection differs across ranks')
        startup_receipts.append({'rank': rank, 'file': startup_name, 'bytes': len(startup_raw),
                                 'sha256': hashlib.sha256(startup_raw).hexdigest()})
        del record, startup, startup_raw  # Rank0 plus only ONE other rank's records.
    frames = []
    for source in first['frames']:
        vector = source['vector']
        name = f"teacher-rank0/{vector['file']}"
        if (vector['file'] != f"frame-{source['step']:06d}.npy" or vector['dtype'] != '<f4'
                or vector['shape'] != [data['vocab']]
                or _owned_teacher_hash(root / name, vector['bytes'], limits['max_frame_file_bytes'], control) != vector['sha256']):
            raise ValueError('teacher rank0 vocabulary frame differs')
        frames.append({**source, 'vector': {**vector, 'file': name}})
    authority = {**first, 'schema': 'tensorfold.owned-teacher-authority-v1', 'launch_exit': launch_exit,
                 'cohort_complete': True, 'rank_receipts': receipts, 'startup_receipts': startup_receipts, 'frames': frames}
    authority.pop('rank')
    _owned_teacher_json(root / 'teacher-authority.json', authority, limits['authority_bytes_limit'], control)
    return authority


def owned_teacher_plan(path):
    """A separate one-request authority; never inherits the old714 gates.

    The immutable input plan pays its parsed tree before Engine loading. Replay
    consumes IDs from a positively closed reference capture, not candidate IDs.
    Profile/model/frontend hashes are operator declarations, recorded as such;
    actual Engine/State/source evidence remains a separate required binding.
    """
    from tensorfold.families.qwen4_exp.cuda.retention_resources import _json_payload_policy
    policy = _json_payload_policy()  # Prove the parser/object domain before parsing DATA.
    raw = _owned_teacher_read(path, 8 << 20)
    data = _owned_teacher_object(raw)
    fields = {'schema', 'mode', 'vocab', 'context', 'input_ids', 'max_frames', 'identity', 'reference', 'read_timeout_s'}
    if type(data) is not dict or set(data) != fields or data['schema'] != 'tensorfold.owned-teacher-plan-v1':
        raise ValueError('exact owned teacher DATA schema required')
    if (data['mode'] not in ('capture', 'replay') or type(data['vocab']) is not int or not 0 < data['vocab'] < 2**31
            or type(data['context']) is not int or not 0 < data['context'] < 2**31
            or type(data['max_frames']) is not int or data['max_frames'] <= 0
            or type(data['read_timeout_s']) not in (int, float)
            or not 0 < data['read_timeout_s'] <= sys.maxsize or not math.isfinite(data['read_timeout_s'])
            or type(data['input_ids']) is not list or not data['input_ids']
            or any(type(token) is not int or not 0 <= token < data['vocab'] for token in data['input_ids'])
            or len(data['input_ids']) + data['max_frames'] > data['context']):
        raise ValueError('invalid bounded teacher geometry/IDs')
    identity = data['identity']
    if (type(identity) is not dict or set(identity) != {'run_id', 'case_id', 'model_sha256', 'frontend_sha256', 'profile_sha256'}
            or any(type(identity[k]) is not str or not SHA.fullmatch(identity[k]) for k in ('model_sha256', 'frontend_sha256', 'profile_sha256'))
            or any(type(identity[k]) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identity[k]) for k in ('run_id', 'case_id'))):
        raise ValueError('explicit teacher case/run/model/frontend/profile authority required')
    teacher_ids, reference_sha, reference_bytes, reference_file_bytes, resources = [], None, 0, 0, None
    if data['mode'] == 'capture':
        if data['reference'] is not None:
            raise ValueError('capture cannot import a candidate/reference token sequence')
    else:
        ref = data['reference']
        if (type(ref) is not dict or set(ref) != {'path', 'sha256'}
                or type(ref['path']) is not str or not Path(ref['path']).is_absolute()
                or type(ref['sha256']) is not str or not SHA.fullmatch(ref['sha256'])):
            raise ValueError('replay requires a frozen reference capture')
        reference_raw = _owned_teacher_read(ref['path'], 8 << 20)
        reference_bytes = len(reference_raw)
        reference_sha = hashlib.sha256(reference_raw).hexdigest()
        if reference_sha != ref['sha256']:
            raise ValueError('reference capture SHA differs')
        reference = _owned_teacher_object(reference_raw)
        teacher_ids = reference.get('teacher_ids')
        if (reference.get('schema') != 'tensorfold.owned-teacher-authority-v1' or reference.get('mode') != 'capture'
                or reference.get('cohort_complete') is not True or type(reference.get('launch_exit')) is not int
                or reference.get('launch_exit') != 0
                or type(reference.get('rank_receipts')) is not list or len(reference['rank_receipts']) != 8
                or type(reference.get('startup_receipts')) is not list or len(reference['startup_receipts']) != 8
                or reference.get('closed') is not True or reference.get('failed') is not False
                or reference.get('finished') is not True or reference.get('method') != 'none'
                or reference.get('input_ids') != data['input_ids']
                or reference.get('vocab') != data['vocab'] or type(teacher_ids) is not list
                or len(teacher_ids) != data['max_frames']
                or any(type(token) is not int or not 0 <= token < data['vocab'] for token in teacher_ids)):
            raise ValueError('reference is incomplete or differs from the frozen teacher prefix')
        frames = reference.get('frames')
        if type(frames) is not list or len(frames) != len(teacher_ids):
            raise ValueError('reference full-vocabulary frame list is incomplete')
        if any(reference.get('declared_identity', {}).get(key) != identity[key]
               for key in ('case_id', 'model_sha256', 'frontend_sha256')):
            raise ValueError('replay reference case/model/frontend declarations differ')
        for step, frame in enumerate(frames):
            vector = frame.get('vector', {})
            name = vector.get('file')
            if (frame.get('step') != step or frame.get('teacher_id') != teacher_ids[step]
                    or frame.get('position') != len(data['input_ids']) + step
                    or type(name) is not str or name != f'teacher-rank0/frame-{step:06d}.npy'
                    or type(vector.get('bytes')) is not int
                    or not 0 < vector['bytes'] <= 4 * data['vocab'] + 4096
                    or type(vector.get('sha256')) is not str or not SHA.fullmatch(vector['sha256'])
                    or vector.get('dtype') != '<f4' or vector.get('shape') != [data['vocab']]):
                raise ValueError('reference frame/token/vocabulary identity differs')
            reference_file_bytes += vector['bytes']
        # All declared cardinality/extent work is checked BEFORE payload I/O.
        if reference_file_bytes > data['max_frames'] * (4 * data['vocab'] + 4096):
            raise ValueError('reference file aggregate exceeds declared frame obligation')
        resources = owned_teacher_resources(len(raw), reference_bytes, data['vocab'], data['max_frames'], policy=policy)
        control = _teacher_read_control(data['read_timeout_s'])
        rank_file_bytes = 0
        for records, pattern in ((reference['rank_receipts'], 'teacher-rank{rank}/summary.json'),
                                 (reference['startup_receipts'], 'rank-{rank}.json')):
            for rank, receipt in enumerate(records):
                if (type(receipt) is not dict or receipt.get('file') != pattern.format(rank=rank) or receipt.get('rank') != rank
                        or type(receipt.get('sha256')) is not str or not SHA.fullmatch(receipt['sha256'])
                        or type(receipt.get('bytes')) is not int or not 0 < receipt['bytes'] <= resources['receipt_bytes_limit']):
                    raise ValueError('reference rank receipt list differs')
                rank_file_bytes += receipt['bytes']
        if rank_file_bytes + reference_file_bytes > 16 * resources['receipt_bytes_limit'] + resources['max_total_frame_file_bytes']:
            raise ValueError('cumulative reference I/O exceeds the declared obligation')
        for receipt in (*reference['rank_receipts'], *reference['startup_receipts']):
            name = receipt['file']
            digest = _owned_teacher_hash(Path(ref['path']).parent / name, receipt.get('bytes'),
                                         resources['receipt_bytes_limit'], control)
            if digest != receipt['sha256']:
                raise ValueError('reference rank receipt differs from successful cohort authority')
        for frame in frames:
            vector = frame['vector']
            digest = _owned_teacher_hash(Path(ref['path']).parent / vector['file'], vector['bytes'],
                                         4 * data['vocab'] + 4096, control)
            if digest != vector.get('sha256'):
                raise ValueError('frozen reference vocabulary file differs')
    if resources is None:
        resources = owned_teacher_resources(len(raw), reference_bytes, data['vocab'], data['max_frames'], policy=policy)
    resources['reference_file_bytes'] = reference_file_bytes + (rank_file_bytes if data['mode'] == 'replay' else 0)
    resources['reference_file_count'] = len(teacher_ids) + (16 if data['mode'] == 'replay' else 0)
    return {'data': data, 'input_sha256': hashlib.sha256(raw).hexdigest(), 'teacher_ids': teacher_ids,
            'reference_sha256': reference_sha, 'resources': resources}


def _teacher_read_control(seconds, cancelled=None):
    if __package__:
        from .v100_rotorquant_live import ReadControl
    else:
        from v100_rotorquant_live import ReadControl
    # Cooperative before/after chunk checks; an in-flight storage read/write
    # or fsync is not interruptible here. The outer owned controller still owns
    # cancellation and whole-job retirement; this is not a hard wall promise.
    return ReadControl(time.monotonic() + seconds, cancelled)


def _hc_graph_geometry(record, target_rows):
    """Keep historical R1 receipts distinct from owned physical-M32 receipts."""
    if type(record) is not dict or type(target_rows) is not int or target_rows <= 0:
        raise ValueError('teacher HC graph row/workspace geometry differs')
    policy = record.get('policy')
    if policy == 'sm70-gguf-hc-readout-r1-aten-graph-v1':
        geometry = {'rows': [1]}
    elif policy in ('sm70-gguf-hc-readout-m32-aten-graph-v2',
                    'sm70-gguf-hc-readout-m32-weight-templates-v3',
                    'sm70-gguf-hc-readout-r1-direct-m32-templates-v4'):
        workspace, budget = record.get('workspace_tensor_bytes'), record.get('graph_budget_bytes')
        if (type(record.get('physical_rows')) is not int or record['physical_rows'] != 32
                or type(workspace) is not int or type(budget) is not int or not 0 < workspace <= budget):
            raise ValueError('teacher HC graph row/workspace geometry differs')
        geometry = {'rows': list(range(1, min(24, target_rows) + 1)),
                    'physical_rows': 32, 'workspace_tensor_bytes': workspace}
        if policy in ('sm70-gguf-hc-readout-m32-weight-templates-v3',
                      'sm70-gguf-hc-readout-r1-direct-m32-templates-v4'):
            bank, templates = record.get('bank_tensor_bytes'), record.get('template_count')
            if (type(bank) is not int or not 0 < bank or workspace + bank > budget
                    or type(templates) is not int or not 0 < templates <= 97
                    or type(record.get('captured_templates')) is not int
                    or record['captured_templates'] != templates
                    or type(record.get('covered_keys')) is not int or record['covered_keys'] != 97):
                raise ValueError('teacher HC graph template/bank geometry differs')
            geometry.update(bank_tensor_bytes=bank, template_count=templates,
                            captured_templates=templates, covered_keys=97)
            if policy == 'sm70-gguf-hc-readout-r1-direct-m32-templates-v4':
                direct = {'r1_graph_count': 97, 'captured_r1_graphs': 97,
                          'captured_graphs': 97 + templates}
                groups, template_rows = record.get('bank_group_count'), record.get('template_rows')
                if (any(type(record.get(key)) is not int or record[key] != value
                        for key, value in direct.items())
                        or type(groups) is not int or not 0 < groups <= templates
                        or type(template_rows) is not list or any(type(row) is not int for row in template_rows)
                        or template_rows != list(range(2, min(24, target_rows) + 1))):
                    raise ValueError('teacher HC graph direct/template coverage differs')
                geometry.update(direct, bank_group_count=groups, template_rows=template_rows)
    else:
        raise ValueError('teacher HC graph row/workspace geometry differs')
    rows = record.get('rows')
    if (type(rows) is not list or any(type(row) is not int for row in rows)
            or rows != geometry['rows']):
        raise ValueError('teacher HC graph row/workspace geometry differs')
    return geometry


def _owned_teacher_hc_graphs(engine, requested):
    """Verify the local readout-only owner; never waive whole-round graphs."""
    if type(requested) is not bool:
        raise TypeError('teacher HC graph intent must be bool')
    decoder = engine.multi
    plans = getattr(decoder, 'hc_graphs', None)
    if not requested:
        if plans is not None or getattr(decoder, 'hc_graphs_requested', False):
            raise ValueError('no-graphs teacher cannot enroll a local HC graph owner')
        return None
    from tensorfold.families.qwen4_exp.cuda.hc_readout_graphs import HCReadoutGraphs, POLICY
    admission = engine.capacity_plan.get('multi_graphs', {})
    record = decoder.hc_graph_record
    if (type(plans) is not HCReadoutGraphs or plans is not decoder.buf.hc_plans
            or plans._w is not engine.w or plans._b is not decoder.buf
            or getattr(engine, 'graphs_requested', None) is not True
            or getattr(engine, 'graphs_enabled', None) is not False
            or decoder.graphs is not None or getattr(decoder, '_failed_graphs', None) is not None
            or getattr(decoder, '_failed_hc_graphs', None) is not None
            or decoder.hc_graphs_requested is not True or not plans.enabled
            or plans.prepared is not True or plans.closed is not False
            or plans.failure is not None or plans.retirement_failure is not None
            or admission.get('original_requested') is not True
            or admission.get('whole_round_requested') is not False or admission.get('captured') is not False
            or admission.get('local_hc') != record or record.get('captured') is not True
            or not decoder.full_resident or len(decoder.states) != 24 or decoder.full_windows != 24
            or any(st.capacity < engine.context_window or st.limit < engine.context_window
                   or st.logical_context != engine.context_window for st in decoder.states)):
        raise ValueError('teacher HC graphs lack their actual readout-only joint full-State admission')
    live = plans.record()
    geometry = _hc_graph_geometry(live, decoder.buf.rows)
    if (live.get('policy') != POLICY or live.get('prepared') is not True or live.get('closed') is not False
            or live.get('catalogue_keys') != 97 or live.get('captured_keys') != 97
            or record.get('policy') != POLICY or record.get('captured_keys') != 97
            or any(record.get(key) != value for key, value in geometry.items())
            or type(live.get('graph_budget_bytes')) is not int or live['graph_budget_bytes'] <= 0
            or live['graph_budget_bytes'] != admission.get('planned_bytes')
            or live['graph_budget_bytes'] != decoder.graph_budget
            or record.get('materialized_full_windows', {}).get('slots') != len(decoder.states)
            or record.get('materialized_full_windows', {}).get('capacities') != [st.capacity for st in decoder.states]):
        raise ValueError('teacher HC graph catalogue or admitted budget differs')
    votes = record.get('capture_results')
    if (type(votes) is not list or len(votes) != 8
            or any(type(row) is not list or len(row) != 5
                or any(type(n) is not int or n < 0 for n in row)
                or row[:2] != [1, 0] or row[3] < row[4] for row in votes)):
        raise ValueError('teacher HC graph peers did not preserve their future-service grant')
    plans.validate_adoption()
    sources = {}
    for name in ('hc_readout_graphs', 'hc_plans'):
        module = sys.modules['tensorfold.families.qwen4_exp.cuda.' + name]
        path = Path(module.__file__)
        sources[name] = _owned_teacher_hash(path, path.stat().st_size, 8 << 20)
    return {'policy': POLICY, **geometry, 'catalogue_keys': 97, 'sources': sources,
            'graph_budget_bytes': live['graph_budget_bytes']}


class OwnedTeacherFrames:
    """Private none/greedy observer on the real MultiDecoder worker.

    This process handles ONE declared request. Module aliases are replaced only
    during its round and restored before returning. Full logits are copied
    before delegating the original sampler once; replay alone replaces the
    returned token with the frozen reference ID. All GPU data storage is an
    explicitly idle, already-paid target-logit tail, never a new allocation.
    """
    def __init__(self, engine, prepared, directory, *, fail, profile, hc_graphs=False):
        import threading
        import torch
        from tensorfold.cleanup import raise_failures
        from tensorfold.families.qwen4_exp.cuda import decode, multi
        self.raise_failures = raise_failures  # Resolve before observer/hook effects, never on the failure path.
        self.engine, self.decoder, self.prepared = engine, engine.multi, prepared
        self.torch, self.fail = torch, fail
        self.directory, self.profile = Path(directory), profile
        self.runtime_sources = _owned_runtime_sources(engine)
        preparation = getattr(engine, '_owner_preparation', None)
        self.owner_family = ({'kernel_family': 'torch-reference', 'graph_family': 'uncaptured'} if preparation is None
                             else _owned_teacher_object(json.dumps(preparation.specification, allow_nan=False)))
        self.original = {name: getattr(self.decoder, name) for name in ('admit', 'round', 'finish')}
        self.restore_records = tuple((name, self.original[name], name in vars(self.decoder))
                                     for name in ('admit', 'round', 'finish'))
        self.close_status = [None] * 7
        self.alias_status = [None] * 4
        self.install_status = [None] * 3
        self.abort_status = [None] * 2
        alias_records = []
        for module in (decode, multi):
            for name in ('choose_gathered', 'tp_sample_rows'):
                original = getattr(module, name)
                def sampled(*args, _original=original, _gathered=name == 'choose_gathered', **kwargs):
                    try:
                        return self.sample(_original, args, kwargs, gathered=_gathered)
                    except BaseException as primary:
                        self.abort(primary)
                        raise
                alias_records.append((module, name, original, sampled))
        self.alias_records = tuple(alias_records)
        self.old_close = engine.close
        self.stream = self.state = self.thread = None
        self.delegating = self.finished = self.closed = self.close_attempted = False
        self.failed = None
        self.frames, self.tokens = [], []
        self.thread_id = threading.get_ident
        data, buf, plan = prepared['data'], self.decoder.buf, self.decoder._sample_plan
        preparation = getattr(engine, '_owner_preparation', None)
        widths = [int(value) for value in plan.widths]
        width = widths[engine.rank]
        if (getattr(engine, '_lifecycle', None) is None or getattr(engine, '_closed', None) is not False
                or engine.tp != 8 or engine.w.comm is not engine.comm or engine.w.mtp is not None
                or engine.sm70_profile != 'reference'
                or self.decoder.graphs is not None or data['context'] != engine.context_window
                or preparation is not None and (preparation.graphs or preparation.reference_arithmetic)
                or len(widths) != 8 or len(set(widths)) != 1 or sum(widths) != data['vocab']
                or plan.mapping is not None or plan.offset != sum(widths[:engine.rank])
                or engine.w.meta['vocab_offset'] != plan.offset
                or not buf.logits.is_cuda or buf.logits.requires_grad or buf.logits.dtype != torch.float32 or not buf.logits.is_contiguous()
                or tuple(buf.logits.shape)[1:] != (width,) or buf.logits.shape[0] < 9):
            raise ValueError('teacher requires actual TP8 contiguous vocabulary and idle nine-row Float32 target buffer; no graphs')
        self.width, self.widths = width, widths
        states = self.decoder.states
        if (not states or not states[0].kv_identity or any(state.kv_identity != states[0].kv_identity for state in states)
                or states[0].kv_identity[0] != engine.kv_pair.identity
                or ('logical-context-v1', engine.context_window) not in states[0].kv_identity
                or ('position-profile-v1', engine.position_profile.identity, engine.position_profile.epoch)
                   not in states[0].kv_identity
                or ('coordinate-profile-v1', profile['coordinate_profile_identity']) not in states[0].kv_identity):
            raise ValueError('teacher actual State numerical profiles disagree before capture')
        # State.kv_identity is the existing canonical semantic owner: ordered
        # codec/working policy, logical context, position/coordinate identities,
        # unique history/indexer and rank-independent native-expert plan. Keep
        # rank-local device/partition/resource observations outside this key.
        self.semantic_profile = {'state_kv_identity': states[0].kv_identity,
                                 'context': engine.context_window, 'cache_slots': self.decoder.capacity,
                                 'prefill_rows': self.decoder.prefill_rows, 'target_rows': buf.rows,
                                 'vocab_widths': widths, 'sm70_profile': engine.sm70_profile,
                                 'prepared_execution': self.owner_family}
        hc_policy = _owned_teacher_hc_graphs(engine, hc_graphs)
        if hc_policy is not None:
            self.semantic_profile['hc_readout_graphs'] = hc_policy
        self.profile = {**profile, 'teacher_semantics': self.semantic_profile}
        if hc_policy is not None:
            self.profile['hc_readout_graph_record'] = self.decoder.hc_graph_record
        self.gathered = buf.logits[1:9].reshape(-1)
        # The owning before_listener caller journals self before setup allocates
        # this paid host vector or creates the fresh evidence directory.
        self.cpu = None

    def setup(self):
        self.directory.mkdir(mode=0o700)
        if self.engine.rank == 0:
            self.cpu = self.torch.empty((self.prepared['data']['vocab'],), dtype=self.torch.float32, device='cpu')

    def abort(self, primary):
        if self.failed is primary:
            return
        if self.failed is not None and self.failed is not primary:
            self.abort_status[0] = primary
            try:
                self._transport_status(self.failed, self.abort_status)
            except BaseException as failure:
                if failure is self.failed:
                    raise
                raise self.failed from failure
        self.failed = primary
        try:
            self.fail(self.engine, primary)
        except BaseException as error:
            if error is not primary:
                self.abort_status[1] = error
                try:
                    self._transport_status(primary, self.abort_status)
                except BaseException as failure:
                    if failure is primary:
                        raise
                    raise primary from failure
            raise

    def _transport_status(self, primary, status):
        """No list growth/shrink: neutral transport ignores primary duplicates.

        Status and owner records already belong to this operation. Callers
        guard even preparation/entry failure so the authentic primary remains
        the propagated object; every secondary stays rooted in fixed slots.
        """
        if primary is None:
            return
        index = 0
        while index < len(status):
            if status[index] is None:
                status[index] = primary
            index += 1
        self.raise_failures(primary, status)

    def sample(self, original, args, kwargs, *, gathered):
        w = args[0] if args else kwargs['w']
        if self.thread != self.thread_id() or w is not self.engine.w or self.delegating:
            return original(*args, **kwargs)
        logits = kwargs.get('logits') if gathered else args[1]
        positions = args[3] if gathered else args[2]
        sampling = args[4] if gathered else (args[3] if len(args) > 3 else kwargs.get('sampling'))
        data, step = self.prepared['data'], len(self.frames)
        if (self.stream is None or step >= data['max_frames'] or list(positions) != [len(data['input_ids']) + step]
                or self.stream.cancel is not None and self.stream.cancel.is_set()
                or kwargs.get('plan') is not self.decoder._sample_plan or sampling is not self.stream.sampling
                or sampling is not None and sampling.temperature != 0
                or kwargs.get('with_prob', False) or logits is None or logits.dtype != self.torch.float32
                or not logits.is_cuda or logits.requires_grad or tuple(logits.shape) != (1, self.width) or not logits.is_contiguous()
                or logits.device != self.gathered.device):
            raise ValueError('teacher requires exactly one unmodified greedy target row at its frozen prefix')
        state = self.stream.st
        if (state is not self.state or list(self.stream.out) != self.tokens
                or (state.history_slot, state.slot_generation, state.kv_identity) != self.state_identity
                or state.pos != len(data['input_ids']) + max(0, step - 1)):
            raise ValueError('teacher State generation/committed prefix differs from the frozen trajectory')
        role = 'prefill' if step == 0 else 'decode'
        buffers = self.decoder.pbuf if step == 0 else self.decoder.buf
        actual_family = {'kernel_family': buffers.owner_kernel_family,
                         'graph_family': buffers.owner_graph_family}
        if actual_family != {key: self.owner_family.get(key) for key in actual_family}:
            raise ValueError('teacher actual buffer family differs from its frozen preparation authority')
        source = logits[0]
        if step and source.data_ptr() != self.decoder.buf.logits.data_ptr():
            raise ValueError('teacher decode source is not the selected target row0 owner')
        begin, end = source.data_ptr(), source.data_ptr() + source.numel() * source.element_size()
        other, other_end = self.gathered.data_ptr(), self.gathered.data_ptr() + self.gathered.numel() * self.gathered.element_size()
        if begin < other_end and other < end:
            raise ValueError('teacher full-vocabulary destination aliases the borrowed source')
        self.engine.comm.all_gather(source, self.gathered)
        if self.cpu is not None:
            self.cpu.copy_(self.gathered)
        self.torch.cuda.current_stream(self.gathered.device).synchronize()  # All ranks: positive collective/source retirement.
        vector = None
        if self.engine.rank == 0:
            vector = self.write_vector(step)
        self.delegating = True
        try:
            result = original(*args, **kwargs)
        finally:
            self.delegating = False
        if type(result) is not list or len(result) != 1 or type(result[0]) is not int or not 0 <= result[0] < data['vocab']:
            raise ValueError('original teacher sampler returned an unsupported outcome')
        token = result[0] if data['mode'] == 'capture' else self.prepared['teacher_ids'][step]
        frame = {'step': step, 'position': list(positions)[0], 'selected_original': result[0],
                            'teacher_id': token, 'full_vocab': data['vocab'], 'vector': vector,
                            'state_pos': self.stream.st.pos, 'slot_generation': self.stream.st.slot_generation,
                            'history_slot': self.stream.st.history_slot, 'kv_identity': self.stream.st.kv_identity}
        frame['role'] = role
        frame['execution_family'] = dict(self.owner_family)
        if len(json.dumps(frame, allow_nan=False).encode()) > 4096:
            raise ValueError('teacher frame metadata exceeds its prepaid bound')
        self.frames.append(frame)
        self.tokens.append(token)
        return result if data['mode'] == 'capture' else [token]

    def write_vector(self, step):
        import numpy as np
        if __package__:
            from .gb10_io import opened_file, write_all
        else:
            from gb10_io import opened_file, write_all
        values = self.cpu.numpy()
        if values.dtype.str != '<f4' or not np.isfinite(values).all():
            raise ValueError('raw full-vocabulary teacher logits must be finite before sampling')
        target = self.directory / f'frame-{step:06d}.npy'
        control = _teacher_read_control(self.prepared['data']['read_timeout_s'], self.stream.cancel)
        control.check()
        with opened_file(target, 'xb') as stream:
            class BoundedWriter:
                def __init__(self, limit):
                    self.limit, self.written = limit, 0
                def write(self, chunk):
                    control.check()
                    if len(chunk) > self.limit - self.written:
                        raise ValueError('teacher vector serialization exceeds its declared extent')
                    view = memoryview(chunk).cast('B')
                    for start in range(0, len(view), 65536):
                        control.check()
                        write_all(stream, view[start:start + 65536])
                        control.check()
                    self.written += len(view)
                    control.check()
                    return len(view)
            writer = BoundedWriter(self.prepared['resources']['max_frame_file_bytes'])
            # Use NumPy's owned format header, then stream the already-owned
            # contiguous CPU buffer. Avoid np.save's non-file nditer buffer and
            # a whole-array tobytes allocation in an adapter.
            np.lib.format.write_array_header_1_0(writer, np.lib.format.header_data_from_array_1_0(values))
            writer.write(memoryview(values).cast('B'))
            stream.flush()
            os.fsync(stream.fileno())
            control.check()
        actual_bytes = target.stat().st_size
        return {'file': target.name, 'bytes': actual_bytes,
                'sha256': _owned_teacher_hash(target, actual_bytes, self.prepared['resources']['max_frame_file_bytes'], control),
                'dtype': '<f4', 'shape': [self.prepared['data']['vocab']]}

    def install(self):
        from contextlib import contextmanager

        @contextmanager
        def aliases():
            primary, cleanup = None, self.alias_status
            try:
                index = 0
                while index < 4:
                    module, name, original, sampled = self.alias_records[index]
                    if getattr(module, name) is not original:
                        raise ValueError('teacher sampling alias changed outside its owned round')
                    setattr(module, name, sampled)
                    index += 1
                yield
            except BaseException as error:
                primary = error
            index = 3
            while index >= 0:
                module, name, original, _sampled = self.alias_records[index]
                try:
                    setattr(module, name, original)
                except BaseException as error:
                    cleanup[index] = error
                    if primary is None:
                        primary = error
                index -= 1
            try:
                self._transport_status(primary, cleanup)
            except BaseException as failure:
                if primary is None or failure is primary:
                    raise
                raise primary from failure

        def guarded(operation):
            try:
                return operation()
            except BaseException as primary:
                self.abort(primary)  # Before _joined can turn an Exception into a local Stream.error.
                raise

        def admitted(stream):
            data = self.prepared['data']
            if (self.stream is not None or self.decoder.streams or self.decoder.filling
                    or stream.draft or stream.vision is not None or stream.constraint is not None
                    or stream.penalties is not None or stream.probabilities is not None
                    or stream.carry is not None or stream.owed or stream.background
                    or getattr(stream, 'retention_request', None) is not None
                    or getattr(stream, 'retention_window', None) is not None or stream.count != data['max_frames']
                    or len(stream.prompt) != len(data['input_ids'])
                    or any(int(a) != b for a, b in zip(stream.prompt, data['input_ids']))
                    or stream.sampling is not None and stream.sampling.temperature != 0):
                raise ValueError('teacher admits one exact cold none/greedy text request only')
            result = self.original['admit'](stream)
            if (stream.cached != 0 or stream.st is None
                    or stream.st.kv_identity != self.semantic_profile['state_kv_identity']):
                raise ValueError('teacher prefix reuse/retained admission is unsupported')
            self.stream = stream
            self.state = stream.st
            self.state_identity = (stream.st.history_slot, stream.st.slot_generation, stream.st.kv_identity)
            return result

        def rounded():
            if self.failed is not None:
                raise self.failed
            if self.thread is not None:
                raise ValueError('teacher decoder round is reentrant')
            self.thread = self.thread_id()
            try:
                with aliases():
                    return self.original['round']()
            finally:
                self.thread = None

        def finished(done):
            if any(stream is self.stream for stream in done):
                if (self.stream.error is not None or list(self.stream.out) != self.tokens or not self.frames
                        or self.stream.cancel is not None and self.stream.cancel.is_set()
                        or len(self.frames) < self.prepared['data']['max_frames']
                        and self.tokens[-1] not in self.decoder._ends(self.stream)
                        or self.prepared['data']['mode'] == 'replay'
                        and len(self.frames) != len(self.prepared['teacher_ids'])):
                    raise ValueError('teacher request ended without its complete frozen prefix')
                self.finished = True
            return self.original['finish'](done)

        def closed():
            if self.close_attempted:
                raise RuntimeError('teacher Engine.close already attempted; uncertain owners retained')
            self.close_attempted = True
            primary = self.failed
            errors = self.close_status
            retired = False
            try:
                self.old_close()
                with self.engine._lifecycle:
                    if self.engine._closed is not True:
                        raise RuntimeError('teacher Engine.close returned without positive lifecycle retirement')
                retired = True
            except BaseException as error:
                if primary is None:
                    primary = error
                elif error is not primary:
                    errors[0] = error
            if retired:
                self.closed = True
                index = 0
                while index < 3:
                    name, original, owned = self.restore_records[index]
                    try:
                        if owned:
                            setattr(self.decoder, name, original)
                        else:
                            delattr(self.decoder, name)
                    except BaseException as error:
                        errors[index + 1] = error
                        if primary is None:
                            primary = error
                    index += 1
            try:
                if not self.finished or self.failed is not None or not retired:
                    raise RuntimeError('teacher capture lacks positive request/Engine retirement')
            except BaseException as error:
                errors[4] = error
                if primary is None:
                    primary = error
            try:
                _owned_teacher_json(self.directory / 'summary.json', {
                    'schema': 'tensorfold.owned-teacher-result-v1', 'mode': self.prepared['data']['mode'],
                    'method': 'none', 'closed': retired, 'finished': self.finished,
                    'failed': primary is not None, 'qualification': False, 'P3_valid': False,
                    'input_sha256': self.prepared['input_sha256'], 'reference_sha256': self.prepared['reference_sha256'],
                    'declared_identity': self.prepared['data']['identity'], 'actual_owned_profile': self.profile,
                    'runtime_sources': self.runtime_sources, 'owner_family': self.owner_family,
                    'rank': self.engine.rank, 'vocab': self.prepared['data']['vocab'], 'widths': self.widths,
                    'input_ids': self.prepared['data']['input_ids'], 'teacher_ids': self.tokens, 'frames': self.frames,
                    'resources': self.prepared['resources'],
                    'scope': 'raw pre-sampler Float32 teacher data; replay is forced-prefix, not free-generation quality or acceptance'},
                    self.prepared['resources']['receipt_bytes_limit'])
            except BaseException as error:
                errors[5] = error
                if primary is None:
                    primary = error
            if retired:
                # No captured GPU/source owner survives positive Engine.close
                # solely through this observer. Uncertain retirement keeps all.
                try:
                    self.gathered = self.cpu = None
                    self.stream = self.state = self.decoder = self.engine = None
                    self.original.clear()
                    self.restore_records = ()
                    self.old_close = None
                except BaseException as error:
                    errors[6] = error
                    if primary is None:
                        primary = error
            try:
                self._transport_status(primary, errors)
            except BaseException as failure:
                if primary is None or failure is primary:
                    raise
                raise primary from failure
        try:
            self.decoder.admit = lambda stream: guarded(lambda: admitted(stream))
            self.decoder.round = lambda: guarded(rounded)
            self.decoder.finish = lambda done: guarded(lambda: finished(done))
            self.engine.close = closed
        except BaseException as primary:
            errors = self.install_status
            index = 0
            while index < 3:
                name, original, owned = self.restore_records[index]
                try:
                    if owned:
                        setattr(self.decoder, name, original)
                    elif name in vars(self.decoder):
                        delattr(self.decoder, name)
                except BaseException as error:
                    errors[index] = error
                index += 1
            try:
                self._transport_status(primary, errors)
            except BaseException as failure:
                if failure is primary:
                    raise
                raise primary from failure
        return {'mode': self.prepared['data']['mode'], 'method': 'none',
                'actual_profile': self.profile,
                'input_sha256': self.prepared['input_sha256'], 'vocab': self.prepared['data']['vocab'],
                'borrow': 'idle target-logit rows1..8; one active target row; no added project tensor storage; native/allocator reserve separate',
                'resources': self.prepared['resources'], 'qualification': False, 'P3_valid': False}

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
    selected = sys.argv[1:] if argv is None else argv
    if selected[:1] == ['--compare-owned-candidate-teacher']:
        return _compare_owned_candidate_teacher_main(selected[1:])
    if selected[:1] == ['--compare-owned-teacher']:
        return _compare_owned_teacher_main(selected[1:])
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
