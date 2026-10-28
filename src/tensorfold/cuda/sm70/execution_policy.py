"""Select the qualified Flash Next Volta operators before admission and weight loading."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

from tensorfold.families.qwen4_exp.cuda.prefix_policy import (
    PACKED_GRAPH_PROFILE as PACKED_GRAPH_TRIAL, OWNER_GRAPH_PROFILE as OWNER_GRAPH_TRIAL, OWNER_TARGET_GREEDY_PROFILE as OWNER_TARGET_GREEDY_TRIAL,
    OWNER_TARGET_GUARD_PROFILE as OWNER_TARGET_GUARD_TRIAL,
    OWNER_TARGET_PRELUDE_PROFILE as OWNER_TARGET_PRELUDE_TRIAL,
    OWNER_TARGET_DENSE_PROFILE as OWNER_TARGET_DENSE_TRIAL,
    OWNER_BATCH_PRELUDE_PROFILE as OWNER_BATCH_PRELUDE_TRIAL,
    ROUTER_BLAS_PROFILE as ROUTER_BLAS_TRIAL,
)

PROFILE_ENV = "TENSORFOLD_SM70_PROFILE"
FAMILY_TRIAL = "source-family-trial"
FAMILY_PREFIX_TRIAL = "source-family-prefix-trial"
PACKED_TRIAL = "source-packed-trial"
TARGET_GREEDY_TRIAL = "source-target-greedy-trial"
# The existing engine owns this process-wide policy at quiescent startup or
# profile preparation. A successful configure replaces it before weight and
# request publication; serving never mutates these selectors or their record.
_SELECTED = "reference"
_FAMILY_SOFTWARE = None
_FAMILY_RUNTIME = {"torch": "2.14.1+tfcu129sm70",
                   "torch_git": "04cf2557f315a350e7eea7ab064288def65db75a", "cuda": "12.9"}
_FAMILY_COMPONENTS = {"cuda": "12.9.2", "pytorch": "2.14.1", "cccl": "3.5.0",
                      "nccl": "2.32.3", "cutlass": "4.8.1", "cublas": "12.9.2.10"}
_FAMILY_GEOMETRY = {"hidden_size": 2560, "num_hidden_layers": 48, "num_attention_heads": 24,
                    "num_key_value_heads": 2, "head_dim": 256, "linear_num_key_heads": 16,
                    "linear_num_value_heads": 48, "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                    "hc_count": 4, "hc_lowrank": 320, "indexer_n_heads": 4,
                    "indexer_head_dim": 128, "indexer_budget": 2048, "indexer_compress_ratio": 4,
                    "ple_embed_dim": 2560, "ple_conv_kernel_size": 4,
                    "num_experts": 512, "num_experts_per_tok": 10,
                    "moe_intermediate_size": 640, "shared_expert_intermediate_size": 640}
_FAMILY_LOCAL = {"hidden": 2560, "layers": 48, "heads": 3, "kv_heads": 1, "head_dim": 256,
                 "nk": 2, "nv": 6, "dk": 128, "dv": 128, "conv_kernel": 4, "streams": 4,
                 "low": 320, "index_heads": 4, "index_dim": 128, "index_budget": 2048,
                 "index_ratio": 4, "ple_dim": 2560, "ple_kernel": 4, "experts": 512,
                 "top_k": 10, "moe_width": 80, "shared_width": 80}
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


def _family_runtime(torch):
    actual = {"torch": str(torch.__version__), "torch_git": torch.version.git_version,
              "cuda": torch.version.cuda}
    if (actual != _FAMILY_RUNTIME or tuple(torch.cuda.get_device_capability()) != (7, 0)
            or "sm_70" not in torch.cuda.get_arch_list()
            or str(torch.backends.cuda.preferred_blas_library()) != "_BlasBackend.Cublas"
            or torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is not True
            or torch.backends.cuda.matmul.allow_fp16_accumulation is not False
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None):
        raise ValueError("source-family-trial requires the exact source04cf SM70 FP16 numerical foundation")


def _family_software():
    from .software_binding import issue_source_binding

    authority = issue_source_binding(required=True)
    record = authority.record()
    if any(record["recorded_components"].get(key) != value for key, value in _FAMILY_COMPONENTS.items()):
        raise ValueError("source-family-trial requires the selected CUDA12.9.2/NCCL2.32.3 source software")
    return record


def _prepare_family_trial(text, tp, torch, *, packed=False, target_greedy=False, native_guard=False, owner_prelude=False, dense=False, owner_batch=False):
    # This trial changes no weight/KV representation, persistent workspace,
    # graph allowance or capacity rule. Native module residency precedes the
    # existing physical-free/admission baseline and weight loading.
    if (type(tp) is not int or tp != 8
            or any(type(text.get(key)) is not int or text[key] != value
                   for key, value in _FAMILY_GEOMETRY.items())
            or text.get("dtype") != "float16" or text.get("mamba_ssm_dtype") != "float32"
            or text.get("output_gate_type") != "sigmoid"):
        raise ValueError("source-family-trial requires the FP16/FP32-state TP8 Flash Next geometry")
    _family_runtime(torch)
    before = _family_software()
    from .family_native import extension

    module = extension()  # Source/header/compiler guards precede selector changes.
    if native_guard:
        module.prepare_value_guard()  # CUDA code residency precedes initial admission.
    if owner_prelude:
        module.prepare_owner_prelude()
    if owner_batch:
        module.prepare_owner_batch()
    if packed:
        from .gguf_ops import extension as gguf_extension

        module = gguf_extension()
        if dense:
            module.prepare_dense32()  # Materialize lazy kernel code before initial admission.
    if target_greedy:
        from .greedy_native import extension as greedy_extension

        greedy_extension()  # Component compilation is not numerical/model qualification.
    if os.environ.get("TENSORFOLD_SM70_GDN_STATE_BATCH") == "1":
        from .gdn_native import extension as gdn_extension

        gdn_extension().prepare_batch()  # Both decay variants resident before capacity baseline.
    after = _family_software()  # Explicitly issue after native-library warmup.
    _same_family_software(before, after)
    return after


def _same_family_software(before, after):
    # Runtime mappings grow during normal startup; protected source inputs may
    # not change. A new process audit binds those mappings at each preparation.
    for key in ("protected", "source_specification_sha256", "recorded_components"):
        if before[key] != after[key]:
            raise ValueError("source-family-trial protected software changed during preparation")


def configure(text, *, tp, torch, record_graph_inputs=None):
    """Explicit profiles share the existing engine, admission and worker handshake."""
    global _SELECTED, _FAMILY_SOFTWARE
    profile = os.environ.get(PROFILE_ENV, "reference")
    if profile not in ("reference", "optimized", FAMILY_TRIAL, FAMILY_PREFIX_TRIAL, PACKED_TRIAL, PACKED_GRAPH_TRIAL,
                       OWNER_GRAPH_TRIAL, TARGET_GREEDY_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
        raise ValueError(f"{PROFILE_ENV} must be reference, optimized, {FAMILY_TRIAL}, {FAMILY_PREFIX_TRIAL}, "
                         f"{PACKED_TRIAL}, {PACKED_GRAPH_TRIAL}, {OWNER_GRAPH_TRIAL}, {TARGET_GREEDY_TRIAL}, {OWNER_TARGET_GREEDY_TRIAL}, {OWNER_TARGET_GUARD_TRIAL} or {OWNER_TARGET_PRELUDE_TRIAL}, {OWNER_TARGET_DENSE_TRIAL}, {OWNER_BATCH_PRELUDE_TRIAL}, {ROUTER_BLAS_TRIAL}")
    batch_gdn = os.environ.get("TENSORFOLD_SM70_GDN_STATE_BATCH")
    if batch_gdn is not None and (batch_gdn != "1" or profile not in (OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL)):
        raise ValueError("private GDN State batch requires1 and explicit Batch or Router profile")

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
    family_trial = profile in (FAMILY_TRIAL, FAMILY_PREFIX_TRIAL, PACKED_TRIAL, PACKED_GRAPH_TRIAL, OWNER_GRAPH_TRIAL, TARGET_GREEDY_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL)
    packed_trial = profile in (PACKED_TRIAL, PACKED_GRAPH_TRIAL, OWNER_GRAPH_TRIAL, TARGET_GREEDY_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL)
    family_software = (_prepare_family_trial(text, tp, torch, packed=packed_trial,
                                             target_greedy=profile in (TARGET_GREEDY_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL),
                                             native_guard=profile in (OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL),
                                             owner_prelude=profile in (OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL),
                                             dense=profile in (OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL),
                                             owner_batch=profile in (OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL))
                       if family_trial else None)
    if profile == ROUTER_BLAS_TRIAL:
        from .router_blas import extension

        extension()  # Compile/load before physical admission; no model allocation.
    family_ops._NATIVE_FAMILY = enabled or family_trial
    family_ops._NATIVE_ATTENTION = enabled
    routing_native._NATIVE_LOGITS = padded_linears._ENABLED = enabled
    gguf_weights.GGUFExperts.implementation = "packed_exact" if enabled or packed_trial else "reference"
    gguf_weights.GGUFConcat.implementation = "native32" if enabled or packed_trial else "reference"
    gguf_weights.GGUFDense.implementation = "native32" if profile in (OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL) else "reference"
    gguf_prefill.WORKSPACE_BUDGET_BYTES = (1 << 30) if enabled else None
    dspark_proposal_graphs._ENABLED = dspark_context_graphs._ENABLED = enabled
    _FAMILY_SOFTWARE = family_software
    _SELECTED = profile
    return profile


def unique_graph_request(profile, *, method, depth, kv_pair, unique_history, indexer_history, graphs, concurrent, vision,
                         owner_preparation=None):
    """Issue the N1 capture contract only from the real pre-load method request."""
    if profile not in (PACKED_GRAPH_TRIAL, OWNER_GRAPH_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
        return None
    if (method != "mtp" or type(depth) is not int or depth != 2 or unique_history is not True
            or indexer_history != "gpu" or graphs is not True or concurrent is not True or vision
            or kv_pair.key_dtype != "int8" or kv_pair.value_dtype != "int8"):
        raise ValueError("source-packed-graphs-trial requires concurrent unique GPU histories, INT8 KV, stock MTP depth2 and graphs")
    from tensorfold.families.qwen4_exp.cuda.multi_graphs import UNIQUE_CAPTURE_CONTRACT

    if profile in (OWNER_GRAPH_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
        from tensorfold.families.qwen4_exp.cuda.multi_graphs import UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT
        from tensorfold.families.qwen4_exp.cuda.owner_execution import KERNEL_FAMILY, UNCAPTURED, FULL_COLLECTIVE

        if (not isinstance(owner_preparation, dict)
                or owner_preparation.get("kernel_family") != KERNEL_FAMILY
                or owner_preparation.get("graph_family") != UNCAPTURED
                or owner_preparation.get("collective_family", FULL_COLLECTIVE) != FULL_COLLECTIVE
                or type(owner_preparation.get("graph_budget_bytes")) is not int
                or owner_preparation["graph_budget_bytes"] != 0 or owner_preparation.get("trial") is not True):
            raise ValueError("source-owner-pieces-trial requires explicit prepared native eager full8 attention")
        return UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT
    return UNIQUE_CAPTURE_CONTRACT


def prepare_weights(w, *, graph_contract=None):
    """Mark native owners only after the same profile was admitted on every rank."""
    if _SELECTED in (FAMILY_TRIAL, FAMILY_PREFIX_TRIAL, PACKED_TRIAL, PACKED_GRAPH_TRIAL, OWNER_GRAPH_TRIAL, TARGET_GREEDY_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
        import torch

        _family_runtime(torch)
        if (w.cfg.execution_dtype != "float16" or w.cfg.quant != "gguf"
                or any(type(getattr(w.cfg, key, None)) is not int or getattr(w.cfg, key) != value
                       for key, value in _FAMILY_LOCAL.items())
                or w.meta.get("execution_provider") != "sm70"
                or w.meta.get("execution_dtype") != "float16"
                or w.meta.get("world") != 8):
            raise ValueError("source-family-trial requires loaded TP8 GGUF SM70 FP16 weights")
        if any(w.meta.get(key) for key in ("capture_ready", "device_greedy_mtp_ready",
                                          "device_greedy_native_ready", "attention_qualification",
                                          "unique_capture_contract")):
            raise ValueError("source-family-trial cannot inherit graph, greedy or private attention qualification")
        software = _family_software()  # New audit after canonical weight/native preparation.
        _same_family_software(_FAMILY_SOFTWARE, software)
        w.meta["sm70_family_trial"] = {"qualification": "unqualified-candidate", "software": software,
            "attention": "reference", "graphs": "eager",
            "prefix_reuse": "int8-greedy-text-trial" if _SELECTED in (
                FAMILY_PREFIX_TRIAL, PACKED_TRIAL, PACKED_GRAPH_TRIAL, OWNER_GRAPH_TRIAL, TARGET_GREEDY_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL) else "cold"}
        if _SELECTED in (PACKED_TRIAL, PACKED_GRAPH_TRIAL, OWNER_GRAPH_TRIAL, TARGET_GREEDY_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
            w.meta["sm70_family_trial"].update(experts="packed_exact", concat="native32")
        if _SELECTED in (PACKED_GRAPH_TRIAL, OWNER_GRAPH_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
            from tensorfold.families.qwen4_exp.cuda.multi_graphs import (
                UNIQUE_CAPTURE_CONTRACT, UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT,
            )

            expected = UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT if _SELECTED in (OWNER_GRAPH_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL) else UNIQUE_CAPTURE_CONTRACT
            if (graph_contract != expected or w.meta.get("history_layout") != "unique"
                    or w.mtp is None or w.mtp.layer.linear):
                raise ValueError("loaded graph trial requires its issued depth2 contract and unique stock attention MTP")
            if _SELECTED in (OWNER_GRAPH_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL) and set(w.meta.get("owner_workspaces", {})) != {"decode", "mtp", "prefill"}:
                raise ValueError("loaded owner-piece trial requires its canonical prepared role workspaces")
            w.meta.update(capture_ready=True, unique_capture_contract=expected)
            w.meta["sm70_family_trial"]["graphs"] = expected
            if _SELECTED in (OWNER_GRAPH_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
                w.meta["sm70_family_trial"]["attention"] = "prepared-native-owner-eager-trial"
        if _SELECTED in (TARGET_GREEDY_TRIAL, OWNER_TARGET_GREEDY_TRIAL, OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
            from tensorfold.families.qwen4_exp.cuda.target_greedy import POLICY

            w.meta["sm70_target_greedy_trial"] = {"policy": POLICY, "qualification": "unqualified-candidate",
                                                "mtp_confidence": "original-fp32-lse-numpy-fp64"}
            w.meta["target_greedy_owner"] = None
        if _SELECTED in (OWNER_TARGET_GUARD_TRIAL, OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
            w.meta["sm70_native_guard_trial"] = True
        if _SELECTED in (OWNER_TARGET_PRELUDE_TRIAL, OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
            from tensorfold.families.qwen4_exp.cuda.history_owner import _native_prelude_binding

            if not _native_prelude_binding(getattr(w, "positions", None)):
                raise ValueError("native owner prelude requires authoritative native-context text position roles")
            zero = torch.zeros((1,), dtype=torch.int32, device=w.device)
            torch.cuda.current_stream(w.device).synchronize()
            w.meta["sm70_owner_prelude_zero"] = zero  # Immutable; no State reset/media field is borrowed.
            w.meta["sm70_owner_prelude_submitted_frames"] = 0  # Serialized actor diagnostic, not device timing.
            w.meta["sm70_owner_prelude_submitted_mtp_frames"] = 0
            w.meta["sm70_owner_prelude_mtp_counter_overflowed"] = False
    if _SELECTED in (OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
        w.meta["sm70_gdn_state_batch_trial"] = os.environ.get("TENSORFOLD_SM70_GDN_STATE_BATCH") == "1"

        w.meta["sm70_owner_batch_prelude_trial"] = True
        w.meta["sm70_owner_prelude_submitted_batches"] = 0
        w.meta["sm70_owner_prelude_submitted_mtp_batches"] = 0
        w.meta["sm70_owner_prelude_batch_counter_overflowed"] = False
        w.meta["sm70_owner_prelude_max_batch_segments"] = 0
    if _SELECTED in (OWNER_TARGET_DENSE_TRIAL, OWNER_BATCH_PRELUDE_TRIAL, ROUTER_BLAS_TRIAL):
        w.meta["sm70_family_trial"].update(dense="native32", dense_math="original-fixed-m32-aten-v1")
    if _SELECTED == ROUTER_BLAS_TRIAL:
        w.meta["sm70_router_blas_trial"] = {"numeric_implementation": "cublas-pointer-batched-fp32-v1",
            "qualification": "unqualified-experimental-numerics", "coverage": "eager-decode-mtp-only"}
    w.meta["sm70_profile"] = _SELECTED
    if _SELECTED == "optimized":
        from .attention_qualification import metadata_prepare
        metadata_prepare(w)
        w.meta.update(capture_ready=True, device_greedy_mtp_ready=True, device_greedy_native_ready=True)
