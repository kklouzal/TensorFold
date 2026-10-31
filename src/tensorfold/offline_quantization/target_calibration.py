"""Guarded native BF16 E4B calibration; no export, publication or serving changes.

Caller owns an exclusive producer process, immutable read-only source and the
actual external/inner guard. admit_phase(name) MUST check current commitments
and admitted remaining work; a returned dictionary is recorded, not treated as
authorization from untrusted JSON. Independent resource readbacks below never
reset caps or infer reuse. Actual calibration/quality still needs GPU evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata as md
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import torch

from tensorfold.checkpoint_metadata import read_config
from tensorfold.cleanup import rollback
from tensorfold.cuda.capacity import _meminfo, headers, itemsize
from tensorfold.cuda.tensor_file import checkpoint_path, _regular_stream, MAX_HEADER_BYTES
from tensorfold.families.gemma4.cuda.e2b_metadata import E2BConfig
from tensorfold.families.gemma4.dense_tensor_contract import text_tensor_contract
from .calibration_coverage import CalibrationCoverage
from .export_manifest import _discovery
from .plan import plan_nvfp4, PRODUCER_COMMIT
from .producer_scope import scoped_local_hessian_config

GIB = 1024**3
PRODUCER_ROOT = Path("/opt/tensorfold-calibration")
MINIMUM_PROFILE = {"cuda": 20 * GIB, "host": 20 * GIB, "external": 2 * GIB, "margin": 4 * GIB, "reserve": 16 * GIB}
BUDGET_ENV = {
    "cuda": "TF_DIAG_CUDA_BUDGET_BYTES",
    "host": "TF_DIAG_HOST_BUDGET_BYTES",
    "external": "TF_DIAG_EXTERNAL_BUDGET_BYTES",
    "margin": "TF_DIAG_GUARD_MARGIN_BYTES",
    "reserve": "TF_DIAG_RESERVE_BYTES",
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _phase(name, admit_phase):
    # The caller's reviewed guard checks live identity/heartbeat and prospective
    # remaining work. No anonymous success bool or default budget is accepted.
    proof = admit_phase(name)
    if not isinstance(proof, dict) or not proof:
        raise ValueError("actual phase-admission receipt required")
    encoded = json.dumps(proof, allow_nan=False, sort_keys=True).encode()
    if len(encoded) > MAX_HEADER_BYTES:
        raise ValueError("phase admission receipt exceeds the owned metadata bound")
    proof = json.loads(encoded)  # snapshot caller state; retain no mutable/foreign owner
    budgets = {key: int(os.environ.get(env, "")) for key, env in BUDGET_ENV.items()}
    if any(value < MINIMUM_PROFILE[key] for key, value in budgets.items()):
        raise ValueError("target calibration cannot lower the admitted C20/H20/E2/M4/R16 profile")
    current = int(Path("/sys/fs/cgroup/memory.current").read_text())
    if Path("/sys/fs/cgroup/memory.max").read_text().strip() != str(budgets["host"]) or current > budgets["host"]:
        raise MemoryError("actual CPU cgroup differs from the admitted calibration envelope")
    total = int(torch.cuda.get_device_properties(0).total_memory)
    fraction = float(torch.cuda.memory.get_per_process_memory_fraction(0))
    cap = int(total * fraction)
    reserved = int(torch.cuda.memory_reserved(0))
    if not budgets["cuda"] - 4096 <= cap <= budgets["cuda"] or reserved > cap:
        raise MemoryError("preinstalled CUDA reservation quota/readback differs from admitted calibration envelope")
    memory = _meminfo()
    if memory is None:
        raise RuntimeError("actual host memory observation required")
    needed = sum(budgets.values()) if name == "before-native-load" else budgets["reserve"] + budgets["margin"]
    if memory["MemAvailable"] < needed:
        raise MemoryError("host reserve/initial complete calibration envelope is not available")
    # After commitment, do NOT demand another full62GiB free or credit guessed
    # reuse. Caller proof owns explicit remaining-future-work admission.
    return {
        "phase": name,
        "guard": proof,
        "budgets": budgets,
        "host_available": memory["MemAvailable"],
        "host_current": current,
        "cuda_reserved": reserved,
        "cuda_cap_readback": cap,
    }


def _physical_producer_audit(root):
    script = """import importlib.util,json,sys
from pathlib import Path
root=Path(sys.argv[1]); path=root/'tools/role.py'
spec=importlib.util.spec_from_file_location('owned_physical_calibration_role',path)
role=importlib.util.module_from_spec(spec);spec.loader.exec_module(role)
if str((root/'prefix').resolve()) in {str(Path(p).resolve()) for p in sys.path if p}:
    raise RuntimeError('physical producer audit requires original interpreter namespace')
manifest=role.read(root/'manifest.json')
role.audit(root/'prefix',manifest)
if role.read(root/'seal.json')!=role.seal_inputs(root/'prefix',root/'manifest.json'):
    raise ValueError('physical producer seal changed')
print(json.dumps({'manifest_sha256':role.sha(root/'manifest.json'),'seal_sha256':role.sha(root/'seal.json')}))
"""
    result = subprocess.run([sys.executable, "-c", script, str(root)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _runtime_proof(namespace=None):
    from tensorfold.families.gemma4.cuda.e2b_runtime import require_runtime
    import transformers

    require_runtime(torch, transformers.__version__, md.version("triton"))
    root = PRODUCER_ROOT
    spec = importlib.util.spec_from_file_location("_tensorfold_owned_calibration_role", root / "tools/role.py")
    role = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(role)
    manifest = role.read(root / "manifest.json")
    prefix = root / "prefix"
    # role.audit requires the ORIGINAL interpreter namespace, not the active
    # producer union. Reuse it in a clean child; never hide packages in parent.
    physical = _physical_producer_audit(root)
    expected = manifest["baseline"]["packages"] | {
        name: {"version": item["version"], "requires": item["requires"]} for name, item in manifest["additions"].items()
    }
    if namespace is None:
        namespace = {}
    if not namespace:
        namespace["selected_path"] = sys.path.copy()
        namespace["import_contract"] = role.build_import_contract(namespace["selected_path"])
    # Reuse the role's existing public-origin / sealed Setuptools vendor-append
    # policy. No parent sys.path mutation or arbitrary hidden-package filtering.
    role.check_import_context(namespace["selected_path"], namespace["import_contract"])
    if role.installed(namespace["selected_path"]) != expected:
        raise ValueError("actual selected producer namespace differs from sealed baseline/additions union")
    if role.read(root / "seal.json") != role.seal_inputs(prefix, root / "manifest.json"):
        raise ValueError("actual producer role seal differs")
    role.provider_proof()
    origin = manifest["source"]
    addition = manifest["additions"]["nvidia-modelopt"]
    if (
        role.MODEL_OPT_COMMIT != PRODUCER_COMMIT
        or origin["source_commit"] != PRODUCER_COMMIT
        or origin["wheel_sha256"] != addition["sha256"]
        or md.version("nvidia-modelopt") != addition["version"]
    ):
        raise ValueError("actual ModelOpt fork/wheel does not match the pinned producer")
    import modelopt

    role.check_import_context(namespace["selected_path"], namespace["import_contract"])
    if not Path(modelopt.__file__).resolve().is_relative_to(prefix.resolve()):
        raise ValueError("actual ModelOpt import is outside the sealed producer prefix")
    return {
        "producer_commit": PRODUCER_COMMIT,
        "wheel_sha256": addition["sha256"],
        "version": addition["version"],
        "role_manifest_sha256": role.sha(root / "manifest.json"),
        "role_seal_sha256": role.sha(root / "seal.json"),
        "physical_audit": physical,
        "active_namespace_sha256": _digest(expected),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _source_snapshot(root):
    cfg = E2BConfig.from_dict(read_config(root))
    if cfg.profile != "E4B":
        raise ValueError("native target calibration currently supports original BF16 E4B only")
    inventory = headers(root)
    shards, assets = _discovery(root, ())
    records = {}
    for name in (*shards, *assets):
        path = checkpoint_path(root, name)
        if not os.statvfs(path).f_flag & os.ST_RDONLY:
            raise ValueError("calibration source must be mounted read-only")
        with _regular_stream(path) as stream:
            before = os.fstat(stream.fileno())
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = os.fstat(stream.fileno())

        def identity(info):
            return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

        if identity(before) != identity(after):
            raise ValueError("source changed during actual file hashing")
        records[name] = {"sha256": digest, "identity": identity(after)}
    return cfg, inventory, records


def _recheck_source(root, records):
    shards, assets = _discovery(root, ())
    if set((*shards, *assets)) != set(records):
        raise ValueError("immutable source shard/asset namespace changed")
    for name, record in records.items():
        path = checkpoint_path(root, name)
        if not os.statvfs(path).f_flag & os.ST_RDONLY:
            raise ValueError("calibration source is no longer read-only")
        info = path.stat()
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != record["identity"]:
            raise ValueError("immutable source namespace/file identity changed")


@dataclass(frozen=True)
class CalibrationLimits:
    """Frozen logical CPU bytes cap; caller input/backing is already committed.

    Admission before cloning must cover actual caller backing plus this extra
    frozen copy. A single contiguous clone avoids a second materialization.
    """

    max_batches: int
    max_sequence_tokens: int
    max_total_padded_tokens: int
    max_cpu_bytes: int

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError("explicit positive calibration dataset bounds required")


def _batch_digest(batch):
    records = {}
    for name, value in sorted(batch.items()):
        records[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(memoryview(value.reshape(-1).view(torch.uint8).numpy())).hexdigest(),
        }
    return _digest(records)


def _freeze_batches(batches, limits, cfg, raw):
    if not isinstance(batches, (tuple, list)) or not 0 < len(batches) <= limits.max_batches:
        raise ValueError("bounded materialized CPU batches required; one-shot iterators are unsupported")
    tokens = bytes_total = 0
    for batch in batches:
        if not isinstance(batch, dict) or not {"input_ids"} <= set(batch) <= {
            "input_ids",
            "attention_mask",
            "position_ids",
        }:
            raise ValueError("initial calibration schema is text-only native HF tensor kwargs")
        ids = batch["input_ids"]
        if (
            not isinstance(ids, torch.Tensor)
            or ids.device.type != "cpu"
            or ids.dtype != torch.int64
            or ids.ndim != 2
            or min(ids.shape) < 1
            or ids.shape[1] > min(limits.max_sequence_tokens, cfg.max_position)
        ):
            raise ValueError("bounded CPU int64[B,S] input IDs required")
        tokens += ids.numel()
        batch_bytes = sum(
            value.numel() * value.element_size() for value in batch.values() if isinstance(value, torch.Tensor)
        )
        bytes_total += batch_bytes
        if bytes_total > limits.max_cpu_bytes or tokens > limits.max_total_padded_tokens:
            raise ValueError("calibration dataset exceeds admitted CPU bytes/padded tokens")
        if int(ids.min()) < 0 or int(ids.max()) >= cfg.vocab:
            raise ValueError("calibration token IDs outside vocabulary")
        if any(bool((ids == raw[key]).any()) for key in ("image_token_id", "video_token_id", "audio_token_id")):
            raise ValueError("text-only calibration must not contain unencoded media placeholder IDs")
        for name, value in batch.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.shape != ids.shape
                or value.dtype not in ((torch.bool, torch.int64) if name == "attention_mask" else (torch.int64,))
            ):
                raise ValueError("calibration masks/positions must match CPU IDs and retain native integer types")
            if name == "attention_mask" and (
                not bool(((value == 0) | (value == 1)).all()) or not bool(value.any(dim=-1).all())
            ):
                raise ValueError("calibration attention mask must contain valid text rows")
            if name == "position_ids" and (int(value.min()) < 0 or int(value.max()) >= cfg.max_position):
                raise ValueError("calibration positions exceed native context")
    if bytes_total > limits.max_cpu_bytes or tokens > limits.max_total_padded_tokens:
        raise ValueError("calibration dataset exceeds admitted CPU bytes/padded tokens")
    frozen = tuple(
        MappingProxyType(
            {name: value.detach().clone(memory_format=torch.contiguous_format) for name, value in batch.items()}
        )
        for batch in batches
    )
    return frozen, tuple(_batch_digest(batch) for batch in frozen)


def _load_native(root):
    from transformers import Gemma4ForConditionalGeneration

    model = info = None
    try:
        model, info = Gemma4ForConditionalGeneration.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            output_loading_info=True,
            use_safetensors=True,
        )
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("native BF16 source load did not consume its required tensors")
        return model.eval().requires_grad_(False)
    finally:
        model = info = None


def _native_inventory(model, cfg, inventory, device):
    value = storage = None
    native = model.state_dict(keep_vars=True)
    try:
        contract, _, unused = text_tensor_contract(cfg)
        expected = set(native) - {"lm_head.weight"}
        if set(inventory) != expected | unused:
            raise ValueError("source/native inventory differs beyond exact expected unused shared-KV tensors")
        for name, info in inventory.items():
            shape = contract[name] if name in unused else tuple(native[name].shape)
            if info["dtype"] != "BF16" or tuple(info["shape"]) != tuple(shape):
                raise ValueError("original BF16 source/native geometry differs")
        if any(
            value.is_meta or value.dtype != torch.bfloat16 or value.device != torch.device(device)
            for value in model.parameters()
        ):
            raise ValueError("native parameters have wrong BF16 placement or unresolved meta storage")
        if any(value.is_meta or value.device != torch.device(device) for value in model.buffers()):
            raise ValueError("native buffers have unresolved meta storage or wrong placement")
        if model.lm_head.weight is not model.get_input_embeddings().weight:
            raise ValueError("original BF16 token embedding/head must share one Parameter")
        if any(type(module).__module__.startswith("tensorfold.") for module in model.modules()):
            raise ValueError("calibration requires original native HF modules without inference adapters")
        cuda_proof = None
        if torch.device(device).type == "cuda":
            storages = {}
            logical = sum(math.prod(inventory[name]["shape"]) * itemsize(inventory[name], name) for name in expected)
            for name, value in native.items():
                if name == "lm_head.weight":
                    continue
                if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
                    raise ValueError("ordinary native CUDA storage proof unavailable")
                storage = value.untyped_storage()
                key = (value.device.index, storage.data_ptr())
                size = storage.nbytes()
                if key in storages and storages[key] != size:
                    raise ValueError("native CUDA storage extent is inconsistent")
                storages[key] = size
            floor = sum(storages.values())
            allocated = torch.cuda.memory_allocated(device)
            if floor < logical or floor > allocated:
                raise ValueError("unique live native CUDA backing does not prove the logical source payload floor")
            cuda_proof = {
                "logical_persistent_bytes": logical,
                "unique_live_storage_bytes": floor,
                "cuda_allocated_bytes": allocated,
                "unique_storages": len(storages),
                "scope": "ordinary live CUDA parameter and persistent-buffer storages; head alias deduplicated",
            }
        return {
            "native_tensors": len(expected),
            "cuda_storage_proof": cuda_proof,
            "expected_unused_tensors": sorted(unused),
            "tied_head_identity": True,
            "nonpersistent_buffers": [
                {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
                for name, value in model.named_buffers()
                if name not in native
            ],
        }
    finally:
        native.clear()
        native = model = value = storage = None


class CalibratedTarget:
    """Explicit owner; retire all device work before releasing calibrated state."""

    def __init__(self):
        self.model = None
        self.receipt = None
        self.source_root = None
        self._runtime_namespace = None
        self.device_touched = False
        self._terminal = False

    def close(self):
        if self._terminal:
            raise RuntimeError("calibration retirement failed; producer process is terminal")
        try:
            if self.device_touched:
                torch.cuda.synchronize(0)
        except BaseException:
            self._terminal = True
            raise
        self.model = None
        self.source_root = None
        self._runtime_namespace = None
        self.device_touched = False


def calibrate_target(source, batches, *, limits, admit_phase):
    """Calibrate original native E4B only; no files are written or published."""
    if not isinstance(limits, CalibrationLimits) or not callable(admit_phase):
        raise ValueError("explicit dataset limits and trusted live phase admission are required")
    root = Path(source)
    admissions = [_phase("source-verification", admit_phase)]
    namespace = {}
    runtime = _runtime_proof(namespace)
    cfg, inventory, files = _source_snapshot(root)
    raw = read_config(root)
    plan = plan_nvfp4(root)
    admissions.append(_phase("before-data-freeze", admit_phase))
    frozen, digests = _freeze_batches(batches, limits, cfg, raw)
    dataset_memory = {
        "frozen_cpu_bytes": sum(v.numel() * v.element_size() for b in frozen for v in b.values()),
        "caller_backing_bytes_conservative": sum(v.untyped_storage().nbytes() for b in batches for v in b.values()),
        "caller_alias_policy": "per tensor conservative backing count; no inferred reuse credit",
        "copy_policy": "one contiguous clone; original caller owners coexist",
    }
    owner = CalibratedTarget()
    kwargs = None
    try:
        admissions.append(_phase("before-native-load", admit_phase))
        import modelopt.torch.quantization as mtq

        config = scoped_local_hessian_config(plan, mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG)
        # ModelOpt block maps mix integer axes with named attributes. Its public
        # schema owns JSON conversion; finish bookkeeping before loading weights.
        recipe_sha256 = _digest(mtq.QuantizeConfig(**config).model_dump(mode="json"))
        _recheck_source(root, files)
        owner.model = _load_native(root)
        native = _native_inventory(owner.model, cfg, inventory, "cpu")
        admissions.append(_phase("before-native-cuda", admit_phase))
        owner.device_touched = True
        owner.model.to(torch.device("cuda:0"))
        cuda_native = _native_inventory(owner.model, cfg, inventory, "cuda:0")
        native["cuda_storage_proof"] = cuda_native["cuda_storage_proof"]
        max_elements = max(batch["input_ids"].numel() for batch in frozen) * max(
            item["shape"][1] for item in plan["tensors"] if item["action"] == "quantize"
        )
        gate = CalibrationCoverage(plan, max_input_elements=max_elements, max_batches=len(frozen))

        def forward_loop(actual):
            nonlocal kwargs
            try:
                with gate.observe_pass(actual):
                    for batch, digest in zip(frozen, digests):
                        admissions.append(_phase("calibration-batch", admit_phase))
                        if _batch_digest(batch) != digest:
                            raise ValueError("frozen calibration tensors changed between passes")
                        kwargs = {name: value.to("cuda:0") for name, value in batch.items()}
                        with gate.batch(digest), torch.no_grad():
                            actual(
                                **kwargs,
                                use_cache=False,
                                logits_to_keep=1,
                                output_hidden_states=False,
                                output_attentions=False,
                                return_dict=True,
                            )
                        kwargs = None
            finally:
                kwargs = actual = batch = None

        admissions.append(_phase("before-quantize", admit_phase))
        owner.model = mtq.quantize(owner.model, config, forward_loop=forward_loop)
        coverage = gate.verify(owner.model)
        admissions.append(_phase("after-calibration", admit_phase))
        _recheck_source(root, files)
        if owner.model.lm_head.weight is not owner.model.get_input_embeddings().weight:
            raise ValueError("calibration changed the tied BF16 head authority")
        after_runtime = _runtime_proof(namespace)
        if after_runtime != runtime:
            raise ValueError("actual producer runtime identity changed")
        owner.source_root = root.resolve()
        owner._runtime_namespace = namespace
        owner.receipt = {
            "schema_version": 1,
            "source_directory": str(owner.source_root),
            "status": "calibrated-native-target-not-exported-or-quality-qualified",
            "source_file_hashes": files,
            "source_identity_scope": "actual file hashes/read-only snapshot/native-library load; NOT consumed-byte hashes",
            "data_scope": "text-only; no multimodal calibration quality claim",
            "data_sha256": _digest(digests),
            "batch_sha256": digests,
            "dataset_memory": dataset_memory,
            "runtime": runtime,
            "native": native,
            "coverage": coverage,
            "admissions": admissions,
            "recipe_sha256": recipe_sha256,
            "recipe_serialization": "ModelOpt QuantizeConfig.model_dump(mode=json); sorted compact JSON SHA256",
            "output_published": False,
        }
        return owner
    except BaseException as primary:
        rollback(owner, primary, owner.close)
    finally:
        kwargs = None
