"""Process-issued software identity for the administrator-owned SM70 source image.

This is preparation/promotion work, never a per-request check. The protected
producer owns the full audit algorithm. Records are serializable observations;
they cannot recreate an authority or clear compiler/device/model/media gates.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import types

from tensorfold.cuda.tensor_file import _regular_stream

_ROOT = Path("/opt/tensorfold-pins")
_PROTECTED = (_ROOT / "v100-source-foundation.json", _ROOT / "source-pins.json",
              _ROOT / "source_foundation.py")
_MAX_METADATA = 16 << 20
_ADMIN_UID = 0
_ISSUER = object()
_PENDING = ("compiler-6111901", "V100-device-graphs-collectives", "model-quality-performance",
            "image-video-processor-stack")


class SoftwareBindingError(RuntimeError):
    """Protected software input, audit or process identity did not validate."""


class ReferenceSoftwareStatus:
    """Explicit absence of source evidence; never a usable source authority."""

    __slots__ = ()

    def record(self):
        return {"schema": "tensorfold.sm70-software-binding", "version": 1,
                "status": "unverified-reference", "software_only": True, "release_qualified": False,
                "reason": "all protected source foundation artifacts are absent",
                "pending_gates": ["source-foundation-software-audit", *_PENDING]}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_mode, info.st_uid, info.st_gid)


def _state(path, *, protected=False):
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise SoftwareBindingError("software binding input must be a regular non-aliased file: " + str(path))
    if protected and (info.st_uid != _ADMIN_UID or info.st_mode & 0o022):
        raise SoftwareBindingError("protected source software metadata must be administrator-owned and immutable")
    return _signature(info)


def _read_protected(path):
    # Reuse the native checkpoint-file owner and its retained-scope error
    # marker. A promotion caller must never treat unretired ownership as a
    # recoverable source-profile refusal. The immutable protected hierarchy and
    # opened-descriptor identity, not a second mutable path lookup, bind bytes.
    expected = _state(path, protected=True)
    with _regular_stream(path) as stream:
        before = os.fstat(stream.fileno())
        if not 0 < before.st_size <= _MAX_METADATA or _signature(before) != expected:
            raise SoftwareBindingError("protected source software metadata is not a bounded immutable regular file")
        raw = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
        if len(raw) != before.st_size or _signature(before) != _signature(after):
            raise SoftwareBindingError("protected source software metadata changed while reading")
    return raw, _signature(after)


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise SoftwareBindingError("duplicate protected software metadata key")
            result[key] = value
        return result

    def constant(value):
        raise SoftwareBindingError("nonfinite protected software metadata")

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise SoftwareBindingError("protected software metadata must be an object")
    return value


def _inputs(manifest):
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100000:
        raise SoftwareBindingError("protected software artifact inventory exceeds its bound")
    paths = set(_PROTECTED)
    for row in rows:
        name = row.get("path") if isinstance(row, dict) else None
        if not isinstance(name, str) or not Path(name).is_absolute():
            raise SoftwareBindingError("protected software artifact paths must be absolute")
        paths.add(Path(name))
    return paths


def _snapshot(paths):
    return {path: _state(path, protected=path in _PROTECTED) for path in paths}


def _protected_location():
    if _ROOT.resolve(strict=True) != _ROOT:
        raise SoftwareBindingError("protected source software directory must not be aliased")
    for path in (_ROOT, *_ROOT.parents):
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != _ADMIN_UID or info.st_mode & 0o022:
            raise SoftwareBindingError("protected source software directory must be administrator-owned and immutable")


def _load_producer(raw, producer_path):
    # Execute the exact bytes already bound to the sealed image inventory, not
    # a second path lookup or a caller-provided module/attestation.
    module = types.ModuleType("_tensorfold_protected_source_audit")
    module.__file__ = str(producer_path)
    exec(compile(raw, str(producer_path), "exec"), module.__dict__)
    if not callable(getattr(module, "audit_process", None)):
        raise SoftwareBindingError("protected source producer has no current-process audit contract")
    return module


def _audit(module, manifest, pins):
    result = module.audit_process(manifest, pins)
    if (not isinstance(result, dict) or result.get("schema") != "tensorfold.v100-source-foundation-process-audit"
            or type(result.get("version")) is not int or result["version"] != 1
            or result.get("software_integrity") != "passed" or result.get("release_qualified") is not False):
        raise SoftwareBindingError("protected source producer did not issue software-only process evidence")
    return result


class SourceSoftwareBinding:
    """Opaque authority, scoped to this process and the exact issued inputs.

    `record()` is observational. Call `verify_source_binding` at a preparation
    or promotion boundary before using this authority. It reruns the protected
    audit; unchanged authority may be reused, changed state needs explicit
    reissuance after warmup. No filesystem/runtime input is silently rebound.
    """

    __slots__ = ("_issuer", "_pid", "_module", "_manifest", "_pins", "_states", "_audit_digest", "_record")

    def __init__(self, issuer, module, manifest, pins, states, audit, protected):
        if issuer is not _ISSUER:
            raise TypeError("source software authority is process-issued")
        self._issuer, self._pid = issuer, os.getpid()
        self._module, self._manifest, self._pins, self._states = module, manifest, pins, states
        self._audit_digest = hashlib.sha256(_canonical(audit)).hexdigest()
        identity = {"protected": protected, "audit_sha256": self._audit_digest,
                    "runtime_sha256": audit["runtime_sha256"]}
        self._record = {"schema": "tensorfold.sm70-software-binding", "version": 1,
                        "status": "verified-source-software", "software_only": True, "release_qualified": False,
                        "issued_pid": self._pid, **identity,
                        "identity_sha256": hashlib.sha256(_canonical(identity)).hexdigest(),
                        "source_specification_sha256": pins.get("specification_sha256"),
                        "recorded_components": pins.get("targets", {}),
                        "pending_gates": sorted(set(audit.get("pending_gates", ())) | set(_PENDING)),
                        "coverage": "sealed source foundation only; target and media qualification are separate"}

    def record(self):
        _issued(self)
        return json.loads(_canonical(self._record))


def _issued(authority):
    if (type(authority) is not SourceSoftwareBinding or getattr(authority, "_issuer", None) is not _ISSUER
            or getattr(authority, "_pid", None) != os.getpid()):
        raise SoftwareBindingError("source software authority is forged, unverified or from another process")


def issue_source_binding(*, required=False):
    """Issue software-only evidence from the fixed protected image artifacts.

    Only absence of all three files permits explicit reference status. Partial,
    malformed or changed files and audit failures always propagate as failures.
    """
    if type(required) is not bool:
        raise TypeError("required must be a boolean")
    available = []
    for path in _PROTECTED:
        try:
            _state(path, protected=True)
        except FileNotFoundError:
            available.append(False)
        else:
            available.append(True)
    if not any(available):
        if required:
            raise SoftwareBindingError("required source software artifacts are absent")
        return ReferenceSoftwareStatus()
    if not all(available):
        raise SoftwareBindingError("protected source software artifacts are incomplete")
    _protected_location()
    read = [_read_protected(path) for path in _PROTECTED]
    manifest, pins = (_decode(item[0]) for item in read[:2])
    producer_raw = read[2][0]
    producer_path = _PROTECTED[2]
    producer_entry = next((row for row in manifest.get("artifacts", ())
                           if isinstance(row, dict) and row.get("path") == str(producer_path)), None)
    digest = hashlib.sha256(producer_raw).hexdigest()
    if (producer_entry is None or type(producer_entry.get("bytes")) is not int
            or producer_entry["bytes"] != len(producer_raw) or producer_entry.get("sha256") != digest):
        raise SoftwareBindingError("protected source producer differs from its sealed artifact identity")
    states = _snapshot(_inputs(manifest))
    if any(states[path] != item[1] for path, item in zip(_PROTECTED, read)):
        raise SoftwareBindingError("protected source software metadata changed before audit")
    module = _load_producer(producer_raw, producer_path)
    # Discover process-owned inputs through the same protected implementation
    # before its content audit, so a concurrent mapping/file change cannot be
    # silently accepted into the issued lifetime after the audit has returned.
    module.application_nccl_identity()
    runtime = module.runtime_identity()
    process_paths = {Path(path) for name in ("mapped_libraries", "host_driver_libraries")
                     for path in runtime[name]}
    states.update(_snapshot(process_paths))
    audit = _audit(module, manifest, pins)
    _protected_location()
    if _snapshot(states) != states:
        raise SoftwareBindingError("immutable source software inputs changed during audit")
    for kind in ("additional_process_libraries", "host_driver_artifacts"):
        for row in audit.get(kind, ()):
            path = Path(row["path"])
            if path not in states:
                raise SoftwareBindingError("source software runtime changed during issuance; reissue after warmup")
    protected = {path.name: {"path": str(path), "bytes": len(item[0]),
                            "sha256": hashlib.sha256(item[0]).hexdigest()}
                 for path, item in zip(_PROTECTED, read)}
    return SourceSoftwareBinding(_ISSUER, module, manifest, pins, states, audit, protected)


def verify_source_binding(authority):
    """Verify reuse at a coordinated boundary; return a fresh serializable record."""
    _issued(authority)
    _protected_location()
    if _snapshot(authority._states) != authority._states:
        raise SoftwareBindingError("source software inputs changed; authority must not be reused")
    audit = _audit(authority._module, authority._manifest, authority._pins)
    _protected_location()
    if (_snapshot(authority._states) != authority._states
            or hashlib.sha256(_canonical(audit)).hexdigest() != authority._audit_digest):
        raise SoftwareBindingError("source software runtime changed; explicitly reissue after warmup")
    return authority.record()
