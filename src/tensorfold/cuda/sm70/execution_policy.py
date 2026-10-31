"""Select the qualified Flash Next Volta operators before admission and weight loading."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

PROFILE_ENV = "TENSORFOLD_SM70_PROFILE"
_SELECTED = "reference"
_GRAPH_KEYS = ("NCCL_GRAPH_MIXING_SUPPORT", "NCCL_GRAPH_STREAM_ORDERING")
_GRAPH_ENV = (*_GRAPH_KEYS, "NCCL_ENV_PLUGIN", "NCCL_CONF_FILE", "HOME")
_SYSTEM_CONF = "/etc/nccl.conf"


def _graph_snapshot():
    values = {key: os.environ.get(key) for key in _GRAPH_ENV}
    user = values["NCCL_CONF_FILE"]
    if not user:
        home = values["HOME"]
        user = None if home is None else home + "/.nccl.conf"
    user = None if user is None else Path(os.fsdecode(os.fsencode(user)[:1023]))
    if user is not None and not user.is_absolute():
        values["cwd"] = os.getcwd()
    return values, user


def _graph_environment(values=None):
    # Record identity without publishing raw configuration/environment values.
    if values is None:
        values, _ = _graph_snapshot()
    consumed = {key: values[key] for key in (*_GRAPH_KEYS, "NCCL_ENV_PLUGIN")}
    user = values["NCCL_CONF_FILE"] or None
    consumed["NCCL_CONF_FILE"] = user
    if user is None:
        consumed["HOME"] = values["HOME"]
    if "cwd" in values:
        consumed["cwd"] = values["cwd"]
    return {key: None if value is None else hashlib.sha256(os.fsencode(value)).hexdigest()
            for key, value in consumed.items()}


def _graph_file(path, values):
    """Bound selected NCCL2.32.3 param.cc file inputs, not arbitrary INI syntax.

    NCCL uses fgets(4096), first '=' and 1023-byte key/value buffers, trims
    CR/LF, and ignores only lines beginning '#'. setenv(overwrite=0) gives
    direct environment, then user file, then system file precedence. Sources
    must stay immutable until native initialization and during serving.
    """
    from tensorfold.cuda.tensor_file import MAX_HEADER_BYTES, _regular_stream

    path = Path(path)
    def stamp(info):
        return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    try:
        before = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "stat": None, "sha256": None, "bytes": 0}
    if before.st_size > MAX_HEADER_BYTES:
        raise ValueError("NCCL configuration exceeds the existing checkpoint metadata read bound")
    digest = hashlib.sha256()
    with _regular_stream(path.resolve(strict=True)) as source:
        if stamp(os.fstat(source.fileno())) != stamp(before):
            raise ValueError("NCCL configuration changed before its opened read")
        remaining = before.st_size
        while remaining:
            raw = source.readline(min(4095, remaining))
            if not raw:
                raise ValueError("NCCL configuration was truncated during validation")
            remaining -= len(raw)
            digest.update(raw)
            line = raw.split(b"\0", 1)[0]
            if line.endswith(b"\n"):
                line = line[:-1]
            if line.endswith(b"\r"):
                line = line[:-1]
            if line.startswith(b"#") or b"=" not in line:
                continue
            key, value = line.split(b"=", 1)
            key = key[:1023]
            for name in (*_GRAPH_KEYS, "NCCL_ENV_PLUGIN"):
                if key == name.encode() and name not in values:
                    values[name] = value[:1023]
        if source.read(1) or stamp(os.fstat(source.fileno())) != stamp(before):
            raise ValueError("NCCL configuration changed during validation")
    if stamp(path.stat()) != stamp(before):
        raise ValueError("NCCL configuration path changed during validation")
    return {"path": str(path), "stat": stamp(before), "sha256": digest.hexdigest(), "bytes": before.st_size}


def graph_inputs():
    """Validate only the initial profile's default graph ordering/mixing inputs.

    This is input provenance, not native/device qualification. An explicit
    environment plugin needs separate qualification. Unrelated config values
    are neither interpreted nor changed. No environment mutation is performed.
    """
    snapshot, user = _graph_snapshot()
    environment = _graph_environment(snapshot)
    values = {name: os.fsencode(snapshot[name]) for name in (*_GRAPH_KEYS, "NCCL_ENV_PLUGIN")
              if snapshot[name] is not None}
    paths = [path for path in (user, Path(_SYSTEM_CONF)) if path is not None]
    files = [_graph_file(path if path.is_absolute() else Path(snapshot["cwd"]) / path, values) for path in paths]
    plugin = values.get("NCCL_ENV_PLUGIN")
    if plugin is not None and plugin.lower() != b"none":
        raise ValueError("an explicit NCCL environment plugin requires separate graph-policy qualification")
    for name in _GRAPH_KEYS:
        raw = values.get(name, b"")
        if not raw:
            continue  # The selected library's unset/empty parameter uses its default.
        # NCCL's strtoll(base=0) accepts an integer prefix. Preserve that
        # grammar directly, including octal/hex, without a foreign/private API.
        match = re.match(rb"[ \t\r\n\v\f]*([+-]?)(0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*)", raw)
        if match is None:
            raise ValueError(f"{name} must resolve to the qualified default value 1")
        number = match[2]
        digits = number[2:] if number.lower().startswith(b"0x") else number
        if match[1] == b"-" or digits.lstrip(b"0") != b"1":
            raise ValueError(f"{name} has an unqualified graph ordering/mixing override")
    record = {"schema": "tensorfold.sm70-nccl-graph-inputs", "version": 1,
              "effective": {"graph_usage_mode": 2, "graph_stream_ordering": 1, "graph_mixing_support": 1},
              "environment": environment, "files": files, "qualification": "validated-inputs-only"}
    verify_graph_inputs(record)
    return record


def verify_graph_inputs(record):
    """Preserve the process-fixed config inputs across prepared model changes."""
    if record is None:
        return
    if record["environment"] != _graph_environment():
        raise ValueError("NCCL graph configuration environment changed after validation")
    for item in record["files"]:
        try:
            info = Path(item["path"]).stat()
        except FileNotFoundError:
            actual = None
        else:
            actual = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
        if actual != item["stat"]:
            raise ValueError("NCCL graph configuration file changed after validation")


def configure(text, *, tp, torch, record_graph_inputs=None):
    """Explicit profiles share the existing engine, admission and worker handshake."""
    global _SELECTED
    profile = os.environ.get(PROFILE_ENV, "reference")
    if profile not in ("reference", "optimized"):
        raise ValueError(f"{PROFILE_ENV} must be reference or optimized")
    if record_graph_inputs is not None and not callable(record_graph_inputs):
        raise ValueError("NCCL graph input journal must be callable")
    graph_policy = graph_inputs() if tp > 1 else None
    if record_graph_inputs is not None:
        record_graph_inputs(graph_policy)
    from . import family_ops, routing_native
    from tensorfold.families.qwen4_exp.cuda import (
        dspark_context_graphs, dspark_proposal_graphs, gguf_prefill, gguf_weights, padded_linears,
    )
    enabled = profile == "optimized"
    if enabled:
        from .attention_qualification import REQUIRED_IDENTITY, runtime_identity
        identity = runtime_identity(torch.device("cuda", torch.cuda.current_device()))
        compatible = (tp == 8 and int(text["hidden_size"]) == 2560
                      and int(text["num_hidden_layers"]) == 48
                      and int(text["num_attention_heads"]) == 24
                      and int(text["num_key_value_heads"]) == 2
                      and tuple(torch.cuda.get_device_capability()) == (7, 0)
                      and all(identity.get(key) == value for key, value in REQUIRED_IDENTITY.items()))
        if not compatible:
            raise ValueError("optimized SM70 profile requires the qualified TP8 Flash Next geometry and pinned Volta foundation")
    family_ops._NATIVE_FAMILY = family_ops._NATIVE_ATTENTION = enabled
    routing_native._NATIVE_LOGITS = padded_linears._ENABLED = enabled
    gguf_weights.GGUFExperts.implementation = "packed_exact" if enabled else "reference"
    gguf_weights.GGUFConcat.implementation = "native32" if enabled else "reference"
    gguf_prefill.WORKSPACE_BUDGET_BYTES = (1 << 30) if enabled else None
    dspark_proposal_graphs._ENABLED = dspark_context_graphs._ENABLED = enabled
    _SELECTED = profile
    return profile


def prepare_weights(w):
    """Mark native owners only after the same profile was admitted on every rank."""
    w.meta["sm70_profile"] = _SELECTED
    if _SELECTED == "optimized":
        from .attention_qualification import metadata_prepare
        metadata_prepare(w)
        w.meta.update(capture_ready=True, device_greedy_mtp_ready=True, device_greedy_native_ready=True)
