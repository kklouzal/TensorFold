"""Mixed ModelOpt Gemma MoE: original BF16 native tensors, direct-scale NVFP4 experts.

Metadata is admitted before allocations. Original immutable files/bytes are
borrowed only during loading; no dequantized expert/whole-model mirror exists.
One synchronous load owns all pending copies and retires them before publication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

from .weights import Config, Gemma4Weights, Layer
from .nvfp4_weights import CT_PREFIX, ct_tensor_contract, _Router

PARTS = ("weight", "weight_scale", "input_scale", "weight_scale_2")
VISION = ("model.vision_tower.", "model.embed_vision.")


@dataclass
class ModelOptLoadStatus:
    """Caller-owned retirement authority, bound before any GPU read/allocation.

    Failed CUDA retirement is process-terminal: retain owners and do not retry
    CUDA/reader close or infer safety from absence of an exception annotation.
    """

    gpu_work_started: bool = False
    safe_to_release: bool = True
    transferred: bool = False
    owners: dict = field(default_factory=dict)


def tensor_contract(c):
    """ModelOpt splits each expert, leaving every non-expert linear in BF16."""
    matrices, plain = ct_tensor_contract(c)
    experts = {name: shape for name, shape in matrices.items() if ".experts." in name}
    plain = {**plain, **{name + ".weight": shape for name, shape in matrices.items() if name not in experts}}
    return experts, plain


def validate_headers(raw, headers, *, vision=False):
    """Exact selected format/inventory, pure CPU metadata before payload allocation."""
    from tensorfold.cuda.nvfp4.format import LinearMetadata, config_block

    c = Config.from_dict(raw)
    block = config_block(raw)
    if (
        not block
        or block.get("quant_method") != "modelopt"
        or block.get("quant_algo") != "NVFP4"
        or block.get("kv_cache_scheme") is not None
        or block.get("pre_quant_scale")
        or block.get("transform_config")
        or block.get("sparsity_config")
        or block.get("lora_rank")
        or block.get("has_zero_point", False) is not False
    ):
        raise ValueError("Gemma ModelOpt MoE requires untransformed direct NVFP4 W4A4 experts and BF16 KV")
    metadata = LinearMetadata(block)
    experts, plain = tensor_contract(c)
    required = set(plain)
    for name, shape in plain.items():
        if headers.get(name, {}).get("dtype") != "BF16" or tuple(headers[name].get("shape", ())) != shape:
            raise ValueError(f"{name}: expected original BF16{shape}")
        if name.endswith(".weight") and len(shape) == 2 and name != CT_PREFIX + "embed_tokens.weight":
            if metadata.validate(name.removesuffix(".weight"), {"weight": ("BF16", list(shape))}, checkpoint=True) != (
                "bf16",
                None,
            ):
                raise ValueError(f"{name}: native projection must remain BF16")
    for name, (n, k) in experts.items():
        expected = {"weight": ("U8", [n, k // 2]), "weight_scale": ("F8_E4M3", [n, k // 16])}
        parts = {}
        if n % 64 or k % 64:
            raise ValueError("ModelOpt experts require complete64-wide tiles")
        for suffix, (dtype, shape) in expected.items():
            key = name + "." + suffix
            required.add(key)
            item = headers.get(key, {})
            if (item.get("dtype"), item.get("shape")) != (dtype, shape):
                raise ValueError(f"{key}: expected {dtype}{shape}")
            parts[suffix] = dtype, shape
        for suffix in ("input_scale", "weight_scale_2"):
            key = name + "." + suffix
            required.add(key)
            item = headers.get(key, {})
            if item.get("dtype") != "F32" or item.get("shape") not in ([], [1]):
                raise ValueError(f"{key}: expected one direct F32 calibration")
            parts[suffix] = item["dtype"], item["shape"]
        if metadata.validate(name, parts, checkpoint=True) != ("nvfp4", 4):
            raise ValueError(f"{name}: declared direct W4A4 expert required")
    vision_names = {name for name in headers if name.startswith(VISION)}
    if vision_names:
        from .vision import vision_shapes, vision_weight_bytes

        expected = {"model." + name: shape for name, shape in vision_shapes(raw).items()}
        expected["model.embed_vision.embedding_projection.weight"] = (c.hidden, raw["vision_config"]["hidden_size"])
        if vision_names != set(expected):
            raise ValueError("ModelOpt Gemma vision inventory differs from native BF16 topology")
        normalized = {name.removeprefix("model."): headers[name] for name in vision_names}
        vision_weight_bytes(raw, normalized)
        for name, shape in expected.items():
            if headers[name]["dtype"] != "BF16" or tuple(headers[name]["shape"]) != shape:
                raise ValueError(f"{name}: expected native BF16 vision payload")
        if metadata.validate(
            "model.embed_vision.embedding_projection",
            {"weight": ("BF16", list(expected["model.embed_vision.embedding_projection.weight"]))},
            checkpoint=True,
        ) != ("bf16", None):
            raise ValueError("ModelOpt vision projector must remain excluded BF16")
        required.update(vision_names)
    elif vision:
        raise ValueError("vision requested without complete native BF16 tower/projector")
    if set(headers) - {"__metadata__"} != required:
        raise ValueError("ModelOpt Gemma has missing or unconsumed tensor payloads")
    return c


def storage_bounds(c, headers, *, vision=False, lossless_head=False):
    """Logical owners plus largest bounded load/validation scratch, not allocator peaks."""
    if type(lossless_head) is not bool:
        raise ValueError("lossless_head must be a bool")
    experts, plain = tensor_contract(c)
    native = set(plain) | ({name for name in headers if name.startswith(VISION)} if vision else set())
    native_bytes = sum(math.prod(headers[name]["shape"]) * 2 for name in native)
    expert_bytes = sum((9 * n * k // 16) + 8 for n, k in experts.values())
    # Original codes/scales, pack4 words, scale permutation, joined byte payload.
    pack_peak = max(4 * (9 * n * k // 16) for n, k in experts.values())
    # Pinned Torch BF16 isfinite is (x==x)*(abs(x)!=inf): its C++
    # full-expression can retain BF16 abs(2N) and three bool(N) owners.
    # Subsequent bool-all scratch (<=2N+5) does not exceed this bound.
    finite_peak = max(5 * math.prod(headers[name]["shape"]) + 5 for name in native)
    largest_source = max(
        math.prod(item["shape"]) * ({"BF16": 2, "F32": 4}.get(item["dtype"], 1))
        for name, item in headers.items()
        if name != "__metadata__"
    )
    bounds = {
        "resident": native_bytes + expert_bytes,
        "device_staging": max(pack_peak, finite_peak),
        "host_staging": largest_source + 2 * (64 << 20) + (1 << 20) + 8 * len(experts) + 3 * 4096,
        "mapped": 0,
    }
    if lossless_head:
        from tensorfold.cuda.b16_sym8_layout import header_bounds
        name = CT_PREFIX + 'embed_tokens.weight'
        head = header_bounds(tuple(headers[name]['shape']))
        bounds['resident'] += head['max_valid_encoded_bytes'] - head['raw_source_bytes']
        # No full head is ever uploaded or validated as a GPU BF16 tensor.
        other_finite = max((5 * math.prod(headers[key]['shape']) + 5 for key in native if key != name), default=0)
        bounds['device_staging'] = max(pack_peak, other_finite)
        # CPU source + encoded source coexist through upload. Sum the finite
        # and pack bounds without inferred phase/allocator reuse credit.
        bounds['host_staging'] += (head['max_valid_encoded_bytes'] + head['pack_temporary_bytes']
                                   + head['cpu_finite_chunk_bytes'])
        bounds['lossless_head'] = head
    return bounds


class _PayloadReader:
    """Operation-local consumed inventory/scalar views for the existing ModelOpt loader."""

    def __init__(self, source, headers, status):
        self.source, self.headers, self.status = source, headers, status
        self.consumed = set()
        self.scalars = {}
        self.scalar_table = None
        # All slots are allocated before GPU work. Updates only replace values
        # under known builtin-string keys; no publication allocation intervenes.
        self.native_owners = {name: None for name, item in headers.items() if item["dtype"] == "BF16"}
        self.native_wrappers = {
            name.removesuffix(".weight"): None
            for name, item in headers.items()
            if name.endswith(".weight") and item["dtype"] == "BF16" and len(item["shape"]) == 2
        }
        self.expert_reads = [None, None]
        status.owners["native"] = self.native_owners
        status.owners["native_wrappers"] = self.native_wrappers
        status.owners["expert_reads"] = self.expert_reads

    def info(self, name):
        return self.headers[name]

    def get(self, name, device="cpu"):
        import torch

        if name in self.scalars:
            return self.scalars[name].to(device)
        if torch.device(device).type == "cuda":
            self.status.gpu_work_started = True
            self.status.safe_to_release = False
        value = None
        try:
            value = self.source.get(name, device)
            if name in self.native_owners:
                self.native_owners[name] = value
            elif torch.device(device).type == "cuda":
                self.expert_reads[0 if name.endswith(".weight") else 1] = value
            self.consumed.add(name)
            return value
        finally:
            value = None

    def get_rows(self, name, start, stop, *, device="cpu"):
        if start != 0 or stop != self.headers[name]["shape"][0]:
            raise ValueError("expert preparation reads one complete validated projection")
        return self.get(name, device)

    def prepare_scalars(self, experts):
        """Coalesce the complete direct scalar table on CPU before GPU allocation."""
        import torch
        from tensorfold.cuda.nvfp4.format import calibration, native_calibration

        if self.scalar_table is not None:
            raise ValueError("direct scalar authority may be prepared once")

        entries = []
        for name in experts:
            for suffix in ("input_scale", "weight_scale_2"):
                key = name + "." + suffix
                file, begin, size, _, _ = self.source.where[key]
                entries.append((file, begin, begin + size, key))
        entries.sort(key=lambda item: (str(item[0]), item[1]))
        self.scalar_table = torch.empty(len(entries), dtype=torch.float32, device="cpu")
        cursor = 0
        batch = []

        def consume(items):
            nonlocal cursor
            raw = value = None
            try:
                first, end = items[0][1], items[-1][2]
                raw = self.source.reader.read(items[0][0], first, end - first, "cpu")
                for _, begin, stop, key in items:
                    value = raw[begin - first : stop - first].view(torch.float32).reshape(self.headers[key]["shape"])
                    calibration(value.item(), name=key)
                    compact = self.scalar_table[cursor : cursor + 1].reshape(self.headers[key]["shape"])
                    compact.copy_(value)
                    self.scalars[key] = compact
                    cursor += 1
                    self.consumed.add(key)
            finally:
                raw = value = None

        for item in entries:
            if batch and (item[0] != batch[0][0] or item[2] - batch[0][1] > 1 << 20 or item[1] - batch[-1][2] > 4096):
                consume(batch)
                batch = []
            batch.append(item)
        if batch:
            consume(batch)
        with torch.device("cpu"):
            for name in experts:
                native_calibration(
                    self.scalars[name + ".input_scale"].item(), self.scalars[name + ".weight_scale_2"].item(), name=name
                )

    def native(self, name, device):
        import torch

        value = None
        try:
            value = self.get(name, device)
            if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name}: native BF16 payload must be finite")
            return value
        finally:
            value = None

    def audit_omitted_vision(self):
        """Disabled frontend owns no GPU tower, but consumes/audits declared bytes."""
        import torch

        raw = None
        try:
            for name in self.headers:
                if not name.startswith(VISION):
                    continue
                file, begin, size, _, _ = self.source.where[name]
                for offset in range(0, size, 1 << 20):
                    raw = self.source.reader.read(file, begin + offset, min(1 << 20, size - offset), "cpu")
                    if not bool(torch.isfinite(raw.view(torch.bfloat16)).all()):
                        raise ValueError(f"{name}: omitted native BF16 vision payload is nonfinite")
                self.consumed.add(name)
        finally:
            raw = None


def _finish(primary, drain_error, close_error, status):
    from tensorfold.cleanup import raise_failures, rollback

    authority = primary if primary is not None else drain_error if drain_error is not None else close_error
    _native_cause = BaseException.__cause__.__get__(authority) if authority is not None else None
    _native_context = BaseException.__context__.__get__(authority) if authority is not None else None
    try:
        if not status.safe_to_release or close_error is not None:

            def incomplete():
                raise_failures(primary, [error for error in (drain_error, close_error) if error is not None])

            rollback(status, authority, incomplete)
        raise_failures(primary, [error for error in (drain_error, close_error) if error is not None])
    except BaseException as transport:
        if authority is not None and transport is not authority:
            raise authority from transport
        raise


def load(model_dir, device="cuda", *, vision=False, status=None, lossless_head=False):
    """Strict resident mixed ModelOpt checkpoint loader with one retirement owner."""
    import hashlib
    import torch
    from tensorfold.checkpoint_metadata import read_config, read_json
    from tensorfold.cuda.direct_read import SafeTensors
    from tensorfold.cuda.nvfp4.checkpoint import available
    from tensorfold.cuda.nvfp4.modelopt_experts import load_modelopt_a4
    from tensorfold.tensor_file import checkpoint_path
    from tensorfold.families.qwen3_5.cuda.weights import Plain
    from .nvfp4_experts import Workspace, geglu
    from .modelopt_experts import ModelOptExperts

    if type(lossless_head) is not bool:
        raise ValueError("lossless_head must be a bool")
    if status is None:
        status = ModelOptLoadStatus()
    if type(status) is not ModelOptLoadStatus or status.transferred or status.owners or not status.safe_to_release:
        raise ValueError("fresh caller-owned ModelOpt load status required")
    root = Path(model_dir)
    raw = read_config(root)
    index = read_json(root, "model.safetensors.index.json")
    mapping = None if index is None else index.get("weight_map")
    if index is not None and (
        not isinstance(mapping, dict)
        or not mapping
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items())
    ):
        raise ValueError("ModelOpt Gemma requires a string safetensors weight_map")
    files = (
        [checkpoint_path(root, "model.safetensors")]
        if mapping is None
        else [checkpoint_path(root, name) for name in sorted(set(mapping.values()))]
    )
    source = payload = workspace = bank = experts = embed = norm = tower = projector = result = pending_layer = None
    layers = []
    status.owners["layers"] = layers
    primary = drain_error = close_error = None
    dev = None
    head_transfer = head_cpu = head_encoded = gpu_head = chunk = None
    try:
        source = SafeTensors([])
        status.owners["reader"] = source
        for file in files:
            shard = SafeTensors([file], reader=source.reader)
            if set(shard.where).intersection(source.where):
                raise ValueError("duplicate ModelOpt tensor across shards")
            source.where.update(shard.where)
        if mapping is not None and (
            set(mapping) != set(source.where)
            or any(source.where[name][0] != checkpoint_path(root, file) for name, file in mapping.items())
        ):
            raise ValueError("ModelOpt index differs from complete actual shard inventory")
        headers = {name: {"dtype": v[3], "shape": v[4]} for name, v in source.where.items()}
        c = validate_headers(raw, headers, vision=vision)
        payload = _PayloadReader(source, headers, status)
        status.owners["payload_reader"] = payload
        quantized, _ = tensor_contract(c)
        payload.prepare_scalars(quantized)
        # Capacity/publication slots are allocated before any CUDA owner exists.
        layers = [None] * c.layers
        pending_fields = [None] * 6
        status.owners.update(
            layers=layers,
            pending_fields=pending_fields,
            pending_bank=None,
            pending_experts=None,
            pending_layer=None,
            workspace=None,
            tower=None,
            projector=None,
            result=None,
            embed=None,
            norm=None,
            head_cpu=None,
            head_encoded=None,
            head_transfer=None,
        )
        dev = torch.device(device)
        if dev.type != "cuda" or not available():
            raise ValueError("ModelOpt Gemma A4 requires SM12.x CUDA")
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(dev)[0] != 12:
            raise ValueError("selected device must support native NVFP4")

        def native(name):
            return payload.native(name, dev)

        def plain(name):
            wrapper = Plain(native(name + ".weight"))
            payload.native_wrappers[name] = wrapper
            return wrapper

        if lossless_head:
            from tensorfold.cuda.b16_sym8 import pack, to_cuda, TransferStatus, Sym8Table
            from tensorfold.cuda.b16_sym8_layout import PACK_CHUNK_GROUPS
            # Existing reader consumes the original BF16 bytes on CPU only.
            # Preserve old finite semantics in bounded chunks before packing.
            head_name = CT_PREFIX + 'embed_tokens.weight'
            head_cpu = payload.get(head_name, 'cpu')
            status.owners['head_cpu'] = head_cpu
            for at in range(0, head_cpu.numel(), PACK_CHUNK_GROUPS * 32):
                chunk = head_cpu.reshape(-1)[at:at + PACK_CHUNK_GROUPS * 32]
                if not bool(torch.isfinite(chunk).all()):
                    raise ValueError(f'{head_name}: native BF16 payload must be finite')
            chunk = None
            head_encoded = pack(head_cpu, chunk_groups=PACK_CHUNK_GROUPS)
            status.owners['head_encoded'] = head_encoded
            head_transfer = TransferStatus()
            status.owners['head_transfer'] = head_transfer
            status.gpu_work_started = True
            status.safe_to_release = False
            gpu_head = to_cuda(head_encoded, device=dev, status=head_transfer)
            embed = Sym8Table(gpu_head, c.max_position)
            status.owners['embed'] = embed
            # Upload's explicit fence retired CPU sources. Only one GPU table
            # remains; neither native_owners nor wrappers retains a raw mirror.
            payload.native_owners[head_name] = None
            payload.native_wrappers[head_name.removesuffix('.weight')] = embed
            status.owners['head_cpu'] = status.owners['head_encoded'] = None
            head_cpu = head_encoded = gpu_head = None
        else:
            embed = plain(CT_PREFIX + "embed_tokens")
            status.owners["embed"] = embed
        norm = native(CT_PREFIX + "norm.weight")
        status.owners["norm"] = norm
        workspace = Workspace()
        status.owners["workspace"] = workspace
        for i in range(c.layers):
            p = f"{CT_PREFIX}layers.{i}."
            # Fields allocation happens inside the shared primitive before its
            # first payload read, so bind unsafe status BEFORE entering it.
            status.gpu_work_started = True
            status.safe_to_release = False
            bank = load_modelopt_a4(
                payload, p.removesuffix("."), c, device=dev, swiglu=geglu, field_owners=pending_fields
            )
            status.owners["pending_bank"] = bank
            experts = ModelOptExperts(bank, workspace)
            status.owners["pending_experts"] = experts
            pending_layer = Layer(
                q=plain(p + "self_attn.q_proj"),
                k=plain(p + "self_attn.k_proj"),
                v=plain(p + "self_attn.v_proj") if c.sliding(i) else None,
                o=plain(p + "self_attn.o_proj"),
                gate=plain(p + "mlp.gate_proj"),
                up=plain(p + "mlp.up_proj"),
                down=plain(p + "mlp.down_proj"),
                router=_Router(native(p + "router.proj.weight")),
                experts=experts,
                input_norm=native(p + "input_layernorm.weight"),
                attention_norm=native(p + "post_attention_layernorm.weight"),
                mlp_norm=native(p + "pre_feedforward_layernorm.weight"),
                expert_norm=native(p + "pre_feedforward_layernorm_2.weight"),
                dense_output_norm=native(p + "post_feedforward_layernorm_1.weight"),
                expert_output_norm=native(p + "post_feedforward_layernorm_2.weight"),
                output_norm=native(p + "post_feedforward_layernorm.weight"),
                q_norm=native(p + "self_attn.q_norm.weight"),
                k_norm=native(p + "self_attn.k_norm.weight"),
                router_scale=native(p + "router.scale"),
                expert_scale=native(p + "router.per_expert_scale"),
                scalar=native(p + "layer_scalar"),
            )
            status.owners["pending_layer"] = pending_layer
            # Existing list slot replacement cannot trigger append growth/OOM.
            layers[i] = pending_layer
            status.owners["pending_layer"] = None
            status.owners["pending_bank"] = None
            status.owners["pending_experts"] = None
            for slot in range(6):
                pending_fields[slot] = None
            bank = experts = pending_layer = None
        tower = (
            {name.removeprefix("model."): native(name) for name in headers if name.startswith("model.vision_tower.")}
            if vision
            else None
        )
        status.owners["tower"] = tower
        projector = plain("model.embed_vision.embedding_projection") if vision else None
        status.owners["projector"] = projector
        if not vision:
            payload.audit_omitted_vision()
        if payload.consumed != set(headers):
            raise ValueError("incomplete actual ModelOpt payload consumption")
        result = Gemma4Weights(c, embed, norm, layers, dev, raw, tower, projector, numerics="modelopt_nvfp4")
        status.owners["result"] = result
        result.precision = "checkpoint"
        result.consumption_receipt = {
            "tensor_count": len(headers),
            "consumed_count": len(payload.consumed),
            "inventory_sha256": hashlib.sha256("\n".join(sorted(payload.consumed)).encode()).hexdigest(),
            "convention": "ModelOpt direct input/weight globals, unchanged packed bytes, BF16 native exceptions",
            "omitted_vision_consumed_on_cpu": not vision,
        }
        if lossless_head:
            result.consumption_receipt['lossless_head'] = {
                'schema': embed.layout, 'packed_bytes': embed.nbytes(),
                'source_dtype': 'BF16', 'source_unchanged': True,
                'compute': 'original BF16 operands / ordered FP32 FMA-XOR / BF16 output',
                'authority': 'one packed table for embedding and output projection',
            }
    except BaseException as error:
        primary = error
        _native_cause = BaseException.__cause__.__get__(primary)
        _native_context = BaseException.__context__.__get__(primary)
    finally:
        unsafe_head = (head_transfer is not None and head_transfer.started
                       and not head_transfer.safe_to_release)
        # A failed nested transfer is terminal. Never retry its CUDA fence or
        # reader retirement from this outer loader boundary.
        if status.gpu_work_started and not unsafe_head:
            try:
                torch.cuda.synchronize(dev)
                status.safe_to_release = True
            except BaseException as error:
                drain_error = error
        if source is not None and status.safe_to_release:
            try:
                source.close()
            except BaseException as error:
                close_error = error
        if status.safe_to_release and close_error is None:
            status.owners.clear()
        if status.safe_to_release:
            if primary is not None or close_error is not None:
                result = embed = norm = tower = projector = bank = experts = workspace = pending_layer = None
                layers = None
            source = payload = None
            head_cpu = head_encoded = head_transfer = gpu_head = chunk = None
        # Unknown retirement retains BOTH named locals and reliable status slots;
        # no original CUDA owner is cleared or retried under terminal state.
    authority = primary if primary is not None else drain_error if drain_error is not None else close_error
    try:
        _finish(primary, drain_error, close_error, status)
    except BaseException as transport:
        if authority is not None and transport is not authority:
            raise authority from transport
        raise
    status.transferred = True
    return result
