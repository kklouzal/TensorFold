"""Process-issued source and CPU media software authority for SM70 adoption.

Only the protected image producers audit their owned artifacts. This adapter
joins their results and immutable lifetimes; it does not qualify vision weights,
media service semantics, devices, numerics, capacity or performance.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
import types

from . import software_binding as source

_ROOT = Path("/opt/tensorfold-media-pins")
_PRODUCER = Path("/opt/tensorfold-pins/media_foundation.py")
_PROTECTED = (_ROOT / "media-install.json", _ROOT / "media-pins.json", _ROOT / "media-build.json",
              _ROOT / "preprocessor_config.json", _ROOT / "video_preprocessor_config.json", _PRODUCER)
_ISSUER = object()
# Source inventory, media inventory, and accumulated process observations each
# fit the existing bounded inventory envelope. Reissue never evicts an anchor.
_MAX_INPUTS = 3 * source._MAX_ARTIFACTS
_PENDING = ("compiler-6111901", "V100-device-graphs-collectives", "SC117-mmproj-model-equivalence",
            "service-input-bounds-and-format-parity", "method-media-position-and-YaRN-quality",
            "V100-resource-quality-performance")


def _locations():
    source._protected_location()
    source._protected_location(_ROOT)


def _snapshot(paths):
    protected = {path for path in paths if path == _PRODUCER or path.is_relative_to(_ROOT)
                 or path.is_relative_to(source._ROOT)}
    return source._snapshot(paths, protected)


def _identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _entry(manifest, path):
    rows = [row for row in manifest["artifacts"] if row["path"] == str(path)]
    if len(rows) != 1:
        raise source.SoftwareBindingError("media metadata must have one exact sealed artifact entry: " + str(path))
    row = rows[0]
    return {"bytes": row.get("bytes"), "sha256": row.get("sha256")}


def _schema(value, name):
    if value.get("schema") != name or type(value.get("version")) is not int or value["version"] != 1:
        raise source.SoftwareBindingError("unsupported protected media metadata: " + name)


def _load_producer(raw, path, source_module):
    # The verified producer imports source_foundation helpers. Bind that one
    # dependency to the privately verified source module without touching the
    # process's shared import cache or trusting a same-name ambient module.
    original_import = builtins.__import__

    def imported(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "source_foundation" and level == 0:
            return source_module
        return original_import(name, globals, locals, fromlist, level)

    namespace = dict(vars(builtins))
    namespace["__import__"] = imported
    module = types.ModuleType("_tensorfold_protected_media_audit")
    module.__file__, module.__dict__["__builtins__"] = str(path), namespace
    exec(compile(raw, str(path), "exec"), module.__dict__)
    if not callable(getattr(module, "audit_install", None)):
        raise source.SoftwareBindingError("protected media producer has no installed software audit")
    return module


class _MediaPreparation:
    __slots__ = ("module", "manifest", "pins", "states", "protected")

    def __init__(self, module, manifest, pins, states, protected):
        self.module, self.manifest, self.pins = module, manifest, pins
        self.states, self.protected = states, protected


def _prepare_media(base):
    # Presence is checked before opening anything. Read/close errors below
    # propagate unchanged, including the native retained-owner failure marker.
    try:
        for path in _PROTECTED:
            source._state(path, protected=True)
    except FileNotFoundError as error:
        raise source.SoftwareBindingError("required protected media artifacts are absent or incomplete") from error
    _locations()
    read = [source._read_protected(path) for path in _PROTECTED]
    manifest, pins, build = (source._decode(item[0]) for item in read[:3])
    _schema(manifest, "tensorfold.v100-media-install")
    _schema(pins, "tensorfold.v100-media-inputs")
    _schema(build, "tensorfold.v100-media-build")
    for item in read[3:5]:
        source._decode(item[0])  # configurations are protected JSON objects, not opaque arbitrary bytes
    paths = source._inputs(manifest, _PROTECTED)
    protected = {path.name: {"path": str(path), **_identity(item[0])}
                 for path, item in zip(_PROTECTED, read)}
    for path, item in zip(_PROTECTED[1:], read[1:]):
        if _entry(manifest, path) != _identity(item[0]):
            raise source.SoftwareBindingError("protected media metadata differs from its sealed artifact: " + str(path))
    producer_identity = _identity(read[-1][0])
    base_identity = base.protected[source._PROTECTED[2].name]
    expected_base = {key: base_identity[key] for key in ("bytes", "sha256")}
    producers = build.get("producer_inputs")
    source_producer_matches = isinstance(producers, dict) and producers.get("source_foundation.py") == expected_base
    if not source_producer_matches and isinstance(producers, dict):
        # A reviewed policy rebind retains the original media build producer
        # under the source authority, separately from its current audit code.
        revision = base.manifest.get("audit_policy_revision")
        original_path = revision.get("original_build_producer_path") if isinstance(revision, dict) else None
        if (isinstance(revision, dict) and base.manifest.get("version") == 2
                and base.manifest.get("runtime_policy") == "torch2141-build-host-v1"
                and revision.get("current_audit_producer") == expected_base
                and revision.get("original_build_producer") == producers.get("source_foundation.py")
                and isinstance(original_path, str)
                and Path(original_path).is_relative_to(source._ROOT / "audit-history")):
            source_producer_matches = _entry(base.manifest, Path(original_path)) == producers["source_foundation.py"]
    if (not isinstance(producers, dict) or producers.get("media_foundation.py") != producer_identity
            or not source_producer_matches
            or build.get("pins_sha256") != hashlib.sha256(source._canonical(pins)).hexdigest()):
        raise source.SoftwareBindingError("media build does not bind the actual source and media audit producers")
    states = _snapshot(paths)
    if any(states[path] != item[1] for path, item in zip(_PROTECTED, read)):
        raise source.SoftwareBindingError("protected media metadata changed before audit")
    module = _load_producer(read[-1][0], _PRODUCER, base.module)
    return _MediaPreparation(module, manifest, pins, states, protected)


def _audit(media):
    result = media.module.audit_install(media.manifest, media.pins)
    if (not isinstance(result, dict) or result.get("schema") != "tensorfold.v100-media-audit"
            or type(result.get("version")) is not int or result["version"] != 1
            or result.get("software_integrity") != "passed" or result.get("release_qualified") is not False
            or not isinstance(result.get("runtime_sha256"), str) or len(result["runtime_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in result["runtime_sha256"])
            or not isinstance(result.get("pending_gates"), list)
            or any(not isinstance(label, str) or not label for label in result["pending_gates"])):
        raise source.SoftwareBindingError("protected media producer did not issue software-only evidence")
    return result


def _issued(authority):
    if (type(authority) is not MediaSoftwareBinding or getattr(authority, "_issuer", None) is not _ISSUER
            or getattr(authority, "_pid", None) != os.getpid()):
        raise source.SoftwareBindingError("media software authority is forged, unverified or from another process")


def _unchanged(authority):
    _issued(authority)
    _locations()
    if _snapshot(authority._states) != authority._states:
        raise source.SoftwareBindingError("immutable source/media inputs changed; software authority cannot be repaired")


class MediaSoftwareBinding:
    """Opaque combined software authority; records are never usable authorities."""

    __slots__ = ("_issuer", "_pid", "_source", "_media", "_states", "_audit_digest", "_record")

    def __init__(self, issuer, base, media, states, audit):
        if issuer is not _ISSUER:
            raise TypeError("media software authority is process-issued")
        self._issuer, self._pid = issuer, os.getpid()
        self._source, self._media, self._states = base, media, states
        self._audit_digest = hashlib.sha256(source._canonical(audit)).hexdigest()
        base_record = base.record()
        identity = {"source_identity_sha256": base_record["identity_sha256"],
                    "protected_media": media.protected, "media_audit_sha256": self._audit_digest,
                    "media_runtime_sha256": audit["runtime_sha256"]}
        runtime = media.manifest["runtime"]
        descriptors = {key: runtime[key] for key in (
            "packages", "processor_classes", "image_backend", "decoder_backend",
            "torchvision_cuda", "loaded_media_providers", "dav1d_version", "pillow") if key in runtime}
        self._record = {"schema": "tensorfold.sm70-media-software-binding", "version": 1,
                        "status": "verified-source-and-media-software", "issued_pid": self._pid,
                        "software_only": True, "release_qualified": False, **identity,
                        "identity_sha256": hashlib.sha256(source._canonical(identity)).hexdigest(),
                        "source_specification_sha256": base_record["source_specification_sha256"],
                        "media_specification_sha256": media.pins.get("specification_sha256"),
                        "verified_software": ["source-foundation", "CPU-image-video-processor-decoder"],
                        "media": descriptors,
                        "pending_gates": sorted(set(_PENDING) | set(base_record["pending_gates"])
                                                | set(audit.get("pending_gates", ()))),
                        "coverage": "software dependencies only; service, model, device and release gates remain separate"}

    def record(self):
        _issued(self)
        return json.loads(source._canonical(self._record))


def _issue(anchor=None):
    if anchor is not None:
        _unchanged(anchor)
    prepared = source._prepare_source(required=True)
    media = _prepare_media(prepared)
    # The protected media audit checks its bytes before importing/materializing
    # declared CPU providers. Final base issuance observes that warm process;
    # no adopted authority is silently rebound during initial construction.
    audit = _audit(media)
    base = source._issue_prepared_source(prepared)
    states = {} if anchor is None else dict(anchor._states)
    for incoming in (base._states, media.states):
        for path, state in incoming.items():
            if path in states:
                if states[path] != state:
                    raise source.SoftwareBindingError("source/media inputs changed during combined audit")
            elif len(states) >= _MAX_INPUTS:
                raise source.SoftwareBindingError("combined software input inventory exceeds its bound")
            states[path] = state
    _locations()
    if _snapshot(states) != states:
        raise source.SoftwareBindingError("immutable source/media inputs changed during combined audit")
    authority = MediaSoftwareBinding(_ISSUER, base, media, states, audit)
    media.states = None  # the complete immutable snapshot is now owned by authority._states
    return authority


def issue_media_binding():
    """Issue required source/media software evidence at a coordinated load boundary.

    Missing, partial, changed or failed audits raise; there is no reference
    downgrade. No model weights or CUDA execution are qualified by this call.
    """
    return _issue()


def verify_media_binding(authority):
    """Verify at adoption/replacement, never per request; runtime drift refuses."""
    _unchanged(authority)
    audit = _audit(authority._media)
    # Media metadata may itself materialize process code. Observe the base
    # runtime afterward, as during issuance, rather than returning a stale base
    # verification when its media dependency added a lazy mapping.
    source.verify_source_binding(authority._source)
    _unchanged(authority)
    if hashlib.sha256(source._canonical(audit)).hexdigest() != authority._audit_digest:
        raise source.SoftwareBindingError("media runtime changed; explicitly reissue after warmup")
    return authority.record()


def reissue_media_binding(authority):
    """Explicitly observe new warm state while retaining the original file anchor."""
    _issued(authority)
    return _issue(authority)
