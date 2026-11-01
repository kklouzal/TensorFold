"""Native HF5.17 Sparkulator with an exact CT INT4-to-BF16 data adapter.

CT0.15.0 public format authority: compressors/pack_quantized/helpers.py,
base.py and quantization/lifecycle/forward_helpers.py in vllm-project/compressed-tensors.
Nibbles store q+8 (not two's complement); unpack then multiply in BF16 scale
precision. Installed CT is absent; no package/SDK/vendor mutation is performed.
The default baseline keeps native math; optional global GQA forces the
existing public efficient SDPA path and retains a native reference scope.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from tensorfold.checkpoint_metadata import read_json
from tensorfold.cleanup import rollback
from tensorfold.cuda.sampling import sample_rows
from ..sparkulator_metadata import checked_headers, FILE_BYTES, FILE_SHA256, validate_global_gqa_torch
from .assistant import _Chains
from .dense_loader import DenseLoadStatus, _VerifiedTensors, _CONSTRUCTION
from .sparkulator_bridge import SparkulatorState
from .sparkulator_attention import install_spark_global_attention
from .assistant_attention import assistant_native_reference


@dataclass(slots=True)
class LoadStatus(DenseLoadStatus):
    # Current GPU destination, its CPU transfer source, native GPU rotary
    # module, and GPU scalar: allocated before any fallible publication.
    pending: list = field(default_factory=lambda: [None] * 4)


def finite_cpu(value, *, chunk_elements=1 << 20):
    flat = None
    try:
        if value.device.type != "cpu" or not value.is_contiguous() or chunk_elements <= 0:
            raise ValueError("bounded contiguous CPU finite validation required")
        flat = value.view(-1)
        for start in range(0, flat.numel(), chunk_elements):
            if not bool(torch.isfinite(flat[start : start + chunk_elements]).all()):
                raise ValueError("Sparkulator original BF16 parameter is nonfinite")
    finally:
        value = flat = None


def _upload_pending(value, device, status):
    destination = None
    try:
        if status.pending[0] is not None or status.pending[1] is not None:
            raise RuntimeError("Sparkulator pending upload already owns data")
        status.pending[1] = value
        destination = torch.empty_like(value, device=device)
        status.pending[0] = destination
        destination.copy_(value, non_blocking=False)
        return destination
    finally:
        # Actual destination/source remain owned by preallocated caller status.
        value = destination = None


def _publish_parameter(parent, leaf, value, status):
    try:
        if leaf in parent._parameters:
            setattr(parent, leaf, nn.Parameter(value, requires_grad=False))
        else:
            setattr(parent, leaf, value)
        status.pending[0] = status.pending[1] = None
    finally:
        parent = value = None


def decompress_int4_group128(packed, scale, logical_shape, expected_shape):
    """Owned boundary adapter for the pinned symmetric CT format, row64 staging.

    Equivalent to CT0.15 unpack_from_int32(...,4,packed_dim=1) followed by
    dequantize(group128): integer nibble minus8, cast to scale dtype, multiply.
    No requantization, clamp, zero-point guess, dtype promotion or hidden float4.
    """
    n, k = expected_shape
    if (
        packed.device.type != "cpu"
        or packed.dtype != torch.int32
        or tuple(packed.shape) != (n, k // 8)
        or scale.device.type != "cpu"
        or scale.dtype != torch.bfloat16
        or tuple(scale.shape) != (n, k // 128)
        or logical_shape.device.type != "cpu"
        or logical_shape.dtype != torch.int64
        or logical_shape.shape != (2,)
        or logical_shape.tolist() != [n, k]
        or n <= 0
        or k <= 0
        or k % 128
        or not bool(torch.isfinite(scale).all())
        or bool((scale < 0).any())
    ):
        raise ValueError("Sparkulator CT INT4 packed/scale/logical shape differs")
    output = tile = integer = scaled = None
    try:
        output = torch.empty((n, k), dtype=torch.bfloat16, device="cpu")
        shifts = torch.arange(8, dtype=torch.int32, device="cpu") * 4
        for begin in range(0, n, 64):
            end = min(n, begin + 64)
            tile = packed[begin:end].unsqueeze(-1)
            integer = ((tile >> shifts) & 15).reshape(end - begin, k) - 8
            # BF16 multiply is the actual native CT scale-domain operation.
            scaled = integer.to(torch.bfloat16).reshape(end - begin, k // 128, 128) * scale[begin:end, :, None]
            if not bool(torch.isfinite(scaled).all()):
                raise ValueError("Sparkulator decoded BF16 weight is nonfinite")
            output[begin:end].copy_(scaled.reshape(end - begin, k), non_blocking=False)
            tile = integer = scaled = None
        result, output = output, None
        return result
    finally:
        packed = scale = logical_shape = output = tile = integer = scaled = None


def _slot(model, name):
    parent, _, leaf = name.rpartition(".")
    return model.get_submodule(parent), leaf


def _load(directory, cfg, inventory, device, status, *, global_gqa=False):
    import transformers
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM

    if transformers.__version__ != "5.17.0":
        raise ValueError("Sparkulator baseline requires pinned Transformers5.17.0")
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    from transformers.initialization import no_init_weights

    raw = read_json(directory, "config.json", required=True)
    native = deepcopy(raw)
    native.pop("quantization_config", None)  # data were decoded exactly, no CT runtime modules
    model = source = value = cpu_value = packed = scale = logical = parent = rope = None
    gpu_started = False
    drain_failed = False
    close_attempted = False
    closed = False

    def drain():
        nonlocal drain_failed
        try:
            torch.cuda.synchronize(device)
        except BaseException:
            drain_failed = True
            raise

    try:
        with _CONSTRUCTION:
            previous = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                config = Gemma4AssistantConfig.from_dict(native)
                config._attn_implementation = "sdpa"
                config.text_config._attn_implementation = "sdpa"
                with torch.device("meta"), no_init_weights():
                    model = Gemma4AssistantForCausalLM(config)
            finally:
                torch.set_default_dtype(previous)
        status.model = model
        if global_gqa:
            install_spark_global_attention(model, enabled=True)
        source = _VerifiedTensors(
            directory,
            inventory,
            None,
            file_hashes={"model.safetensors": FILE_SHA256},
            file_sizes={"model.safetensors": FILE_BYTES},
        )
        status.source = source
        status.safe_to_release = False
        for name, (dtype, _) in cfg.shapes().items():
            if dtype != "BF16" or name.endswith(".weight_scale"):
                continue
            # CPU verified reads avoid an unowned GPU output inside the generic
            # byte reader, and finite temporaries are bounded before upload.
            cpu_value = source.tensor(name, "cpu")
            finite_cpu(cpu_value)
            gpu_started = True
            value = _upload_pending(cpu_value, device, status)
            parent, leaf = _slot(model, name)
            _publish_parameter(parent, leaf, value, status)
            value = cpu_value = parent = None
        for name, shape in cfg.matrix_shapes().items():
            logical = source.tensor(name + ".weight_shape", "cpu")
            packed = source.tensor(name + ".weight_packed", "cpu")
            scale = source.tensor(name + ".weight_scale", "cpu")
            cpu_value = decompress_int4_group128(packed, scale, logical, shape)
            gpu_started = True
            value = _upload_pending(cpu_value, device, status)
            parent = model.get_submodule(name)
            _publish_parameter(parent, "weight", value, status)
            value = cpu_value = packed = scale = logical = parent = None
        with torch.device(device):
            rope = Gemma4TextRotaryEmbedding(model.config.text_config)
        status.pending[2] = rope
        cpu_value = torch.tensor(cfg.hidden**0.5, dtype=torch.bfloat16, device="cpu")
        value = _upload_pending(cpu_value, device, status)
        status.pending[3] = value
        model.model.embed_tokens.register_buffer("embed_scale", value, persistent=False)
        status.pending[0] = status.pending[1] = status.pending[3] = None
        value = cpu_value = None
        model.model.rotary_emb = rope
        status.pending[2] = None
        rope = None
        model.lm_head.weight = model.model.embed_tokens.weight
        if set(source.consumed) != set(inventory) or model.masked_embedding is not None:
            raise ValueError("Sparkulator full88 consumption/unmasked native head differs")
        if any(x.device.type == "meta" for x in (*model.parameters(), *model.buffers())):
            raise ValueError("Sparkulator native model contains uninitialized META values")
        receipt = {
            "fully_consumed": True,
            "consumed_tensor_sha256": dict(source.consumed),
            "verified_file_sha256": dict(source.verified_files),
            "tensor_count": len(source.consumed),
            "storage": "CT signed-int4(group128) exact BF16 decoded baseline; original own BF16 head",
            "tied_head_identity": model.lm_head.weight is model.model.embed_tokens.weight,
        }
        drain()
        close_attempted = True
        source.close()
        closed = True
        source = None
        model.eval().requires_grad_(False)
        status.transferred = status.safe_to_release = True
        status.model = status.source = None
        result, model = model, None
        return result, receipt
    except BaseException as primary:
        failure = primary

        def retire():
            nonlocal close_attempted, closed
            if drain_failed or (close_attempted and not closed):
                raise failure  # terminal native failure: no drain/close retry
            if gpu_started:
                drain()
            if source is not None and not close_attempted:
                close_attempted = True
                source.close()
                closed = True
            status.pending[0] = status.pending[1] = status.pending[2] = status.pending[3] = None
            status.safe_to_release = True
            status.model = status.source = None

        _native_cause = BaseException.__cause__.__get__(primary)
        _native_context = BaseException.__context__.__get__(primary)
        try:
            rollback((model, source, status), primary, retire)
        except BaseException as transport:
            if transport is not primary:
                raise primary from transport
            raise
        raise
    finally:
        if status.safe_to_release:
            model = source = value = cpu_value = packed = scale = logical = parent = rope = None


class Sparkulator:
    feature_source = "final_hidden"
    tap_layers = ()
    bits = 16

    def __init__(self, directory, target, context_limit, *, block=None, device=None, status=None, global_gqa=False, sampled_q=False):
        if type(global_gqa) is not bool or type(sampled_q) is not bool:
            raise ValueError("Sparkulator GQA and sampled-q selections must be boolean")
        self.global_gqa = global_gqa
        self.sampled_q = sampled_q
        self.attention_policy = None
        self.cfg, inventory = checked_headers(directory)
        block = self.cfg.block if block is None else block
        self.cfg.state_bytes(context_limit)
        if global_gqa:
            validate_global_gqa_torch(torch)
            self.cfg.workspace_bytes(context_limit, global_gqa=True)
        if (target.config.hidden, target.config.vocab) != (self.cfg.backbone_hidden, self.cfg.vocab):
            raise ValueError("Sparkulator main target geometry differs")
        if type(block) is not int or not 2 <= block <= (16 if sampled_q else 128):
            raise ValueError("Sparkulator verification rows exceed admitted proposal mode")
        self.device = torch.device(device) if device is not None else target.device
        if self.device != target.device or self.device.type != "cuda":
            raise ValueError("Sparkulator and main target require the same CUDA device")
        self.target, self.context_limit = target, context_limit
        self.block = self.rows = block
        self._bound = None
        self._closed = False
        self.model = None
        self._load_status = LoadStatus() if status is None else status
        if (
            type(self._load_status) is not LoadStatus
            or not self._load_status.safe_to_release
            or self._load_status.transferred
            or self._load_status.model is not None
            or self._load_status.source is not None
            or type(self._load_status.pending) is not list
            or len(self._load_status.pending) != 4
            or any(value is not None for value in self._load_status.pending)
        ):
            raise ValueError("Sparkulator needs a fresh caller-owned load status")
        if global_gqa:
            self.model, self.consumption_receipt = _load(
                Path(directory), self.cfg, inventory, self.device, self._load_status, global_gqa=True
            )
            self.attention_policy = self.model.model.layers[3].self_attn.policy
        else:
            self.model, self.consumption_receipt = _load(
                Path(directory), self.cfg, inventory, self.device, self._load_status
            )
        self.parameter_bytes = self.cfg.parameter_bytes
        self.resident_bytes, self.loading_peak_bytes = self.cfg.memory_bytes()

    def bind_state(self, state):
        if self._closed or self._bound is not None:
            raise RuntimeError("Sparkulator binds one main state after its allocation")
        self._bound = state

    def make_state(self):
        if self._bound is None or self._closed:
            raise RuntimeError("Sparkulator requires a bound main state")
        return SparkulatorState(self, self._bound)

    def append(self, state, hidden, *, prompt=False):
        if state.owner is not self or self._closed:
            raise ValueError("Sparkulator accepted state belongs to another head")
        state.append(hidden)

    @contextmanager
    def native_reference(self):
        # Caller owns the exclusive/drained lane; candidate bypass is nested-safe.
        if self._closed:
            raise RuntimeError("Sparkulator is closed")
        with assistant_native_reference(self.model, self.attention_policy):
            yield self.model

    @torch.inference_mode()
    def propose(self, state, anchor, count, history=None, *, sampled=False, sampling=None):
        if self._closed or state.owner is not self or self._bound is None:
            raise ValueError("Sparkulator state ownership differs")
        if (
            type(anchor) is not int
            or not 0 <= anchor < self.cfg.vocab
            or type(count) is not int
            or not 0 <= count < self.block
        ):
            raise ValueError("Sparkulator anchor/proposal count outside admitted domain")
        if sampled and (sampling is None or sampling.temperature <= 0):
            raise ValueError("sampled target needs positive temperature")
        count = min(count, self.context_limit - state.position - 1)
        if count <= 0:
            return None
        held = ids = positions = embedding = out = hidden = None
        try:
            held = state.view()
            hidden = held["last_hidden_state"]
            position = held["position"]
            if sampled and self.sampled_q:
                from .assistant_sampled import sampled_assistant_chain

                # Bound actual q support; target p/policy is never modified.
                # K0/unbounded or larger K gets explicit sparse256 proposal q.
                q_top_k = min(sampling.top_k, 256) if sampling.top_k > 0 else 256
                return sampled_assistant_chain(
                    self.model, self.target.embed_tokens, hidden,
                    held["shared_kv_states"], held["attention_mask"], anchor, count, position,
                    sampling, history, enabled=True, temperature_factor=1.0,
                    proposal_top_k=q_top_k, max_count=self.block - 1,
                )
            positions = torch.tensor([[position]], dtype=torch.long, device=self.device)
            token = anchor
            tokens = []
            for _ in range(count):
                ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
                embedding = self.target.embed_tokens(ids)
                if embedding.shape != (1, 1, self.cfg.backbone_hidden) or embedding.dtype != torch.bfloat16:
                    raise ValueError("main scaled embedding callback differs")
                out = self.model(
                    inputs_embeds=torch.cat((embedding, hidden), dim=-1),
                    position_ids=positions,
                    attention_mask=held["attention_mask"],
                    shared_kv_states=held["shared_kv_states"],
                    use_cache=False,
                )
                token = sample_rows(out.logits.reshape(1, -1), [position], None)[0]
                hidden = out.last_hidden_state
                tokens.append(token)
            return _Chains(tokens, history, sampled)
        except BaseException:
            state.poisoned = True
            raise
        finally:
            state = held = ids = positions = embedding = out = hidden = None

    def close(self):
        # Owner must drain all consumers before release; this method does not
        # retry a failed native retirement or mutate main cache/weights.
        self._closed = True
        self.model = self._bound = self.target = self.attention_policy = None
