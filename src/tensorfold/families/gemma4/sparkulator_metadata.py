"""Pinned Sparkulator26B CT INT4/group128 and native unmasked assistant profile."""

from __future__ import annotations

from dataclasses import dataclass
import math

from tensorfold.checkpoint_metadata import read_json

REPO = "sapidlabs/Sparkulator-Gemma-4-26B-A4B"
REVISION = "4eda238ff5511e74edf092ca8500931ee79812c2"
FILE_BYTES = 627751208
FILE_SHA256 = "c6a4c16dc33e7743ce3bf5052e6551408dba766e572f04c92c363c58558e3ef3"


# Conditional pinned efficient-provider inventory. Not a native/AUTO bound.
GLOBAL_GQA_TORCH_GIT = "ef46daf1aa35294a3cb297ef3f59b5a6463063f8"
GLOBAL_GQA_WORKSPACE_BYTES = 65552
LOCAL_NATIVE_WORKSPACE_BYTES = (64 + 8 + 16) << 20


def validate_global_gqa_torch(torch_module):
    """The efficient allocation inventory is pinned to this actual Torch source."""
    if torch_module.version.git_version != GLOBAL_GQA_TORCH_GIT:
        raise ValueError("Spark global GQA workspace requires qualified Torch git " + GLOBAL_GQA_TORCH_GIT)


def global_gqa_workspace_bytes(context_limit):
    if type(context_limit) is not int or not 0 < context_limit <= 131072:
        raise ValueError("Spark efficient global candidate requires context1..131072")
    return GLOBAL_GQA_WORKSPACE_BYTES + LOCAL_NATIVE_WORKSPACE_BYTES


@dataclass(frozen=True)
class SparkulatorConfig:
    hidden: int = 1024
    backbone_hidden: int = 2816
    vocab: int = 262144
    max_position: int = 262144
    window: int = 1024
    block: int = 7  # published six proposals + anchor, not an architectural limit

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or not isinstance(raw.get("text_config"), dict):
            raise ValueError("Sparkulator requires native assistant/text metadata")
        text = raw["text_config"]
        expected_root = {
            "model_type": "gemma4_assistant",
            "architectures": ["Gemma4AssistantForCausalLM"],
            "backbone_hidden_size": 2816,
            "use_ordered_embeddings": False,
            "tie_word_embeddings": True,
        }
        expected_text = {
            "model_type": "gemma4_text",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 2,
            "head_dim": 256,
            "global_head_dim": 512,
            "intermediate_size": 8192,
            "num_kv_shared_layers": 4,
            "sliding_window": 1024,
            "max_position_embeddings": 262144,
            "vocab_size": 262144,
            "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": 0,
            "enable_moe_block": False,
            "use_double_wide_mlp": False,
            "attention_k_eq_v": True,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rms_norm_eps": 1e-6,
            "hidden_activation": "gelu_pytorch_tanh",
            "final_logit_softcapping": None,
            "use_bidirectional_attention": None,
            "tie_word_embeddings": True,
            "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
        }
        for value, expected in ((raw, expected_root), (text, expected_text)):
            for name, want in expected.items():
                got = value.get(name)
                if name not in value or got != want or (type(want) in (int, bool) and type(got) is not type(want)):
                    raise ValueError("Sparkulator native profile differs: " + name)
            if value.get("dtype", value.get("torch_dtype", "bfloat16")) != "bfloat16":
                raise ValueError("Sparkulator execution requires BF16")
        if text.get("rope_parameters") != {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        }:
            raise ValueError("Sparkulator native RoPE differs")
        block = raw.get("quantization_config")
        if not isinstance(block, dict) or any(
            block.get(k) != v
            for k, v in {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "quantization_status": "compressed",
            }.items()
        ):
            raise ValueError("Sparkulator requires declared compressed-tensors packed INT4")
        groups = block.get("config_groups")
        if not isinstance(groups, dict) or set(groups) != {"group_0"}:
            raise ValueError("Sparkulator requires its single CT group")
        group = groups["group_0"]
        weight = group.get("weights", {})
        if (
            group.get("targets") != ["Linear"]
            or group.get("input_activations") is not None
            or group.get("output_activations") is not None
            or any(
                weight.get(k) != v
                for k, v in {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 128,
                    "dynamic": False,
                }.items()
            )
            or set(block.get("ignore", ()))
            != {"re:.*embed_tokens$", "re:.*lm_head$", "re:.*post_projection$", "re:.*pre_projection$"}
        ):
            raise ValueError("Sparkulator CT quantization scope differs")
        return cls()

    def matrix_shapes(self):
        result = {}
        for i in range(4):
            root = f"model.layers.{i}."
            dim = 256 if i < 3 else 512
            result[root + "self_attn.q_proj"] = (16 * dim, 1024)
            result[root + "self_attn.o_proj"] = (1024, 16 * dim)
            result[root + "mlp.gate_proj"] = (8192, 1024)
            result[root + "mlp.up_proj"] = (8192, 1024)
            result[root + "mlp.down_proj"] = (1024, 8192)
        return result

    def shapes(self):
        result = {
            "model.embed_tokens.weight": ("BF16", [self.vocab, self.hidden]),
            "model.norm.weight": ("BF16", [self.hidden]),
            "pre_projection.weight": ("BF16", [self.hidden, 2 * self.backbone_hidden]),
            "post_projection.weight": ("BF16", [self.backbone_hidden, self.hidden]),
        }
        for i in range(4):
            root = f"model.layers.{i}."
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                result[root + name + ".weight"] = ("BF16", [self.hidden])
            result[root + "layer_scalar"] = ("BF16", [1])
            result[root + "self_attn.q_norm.weight"] = ("BF16", [256 if i < 3 else 512])
        for name, (n, k) in self.matrix_shapes().items():
            result[name + ".weight_packed"] = ("I32", [n, k // 8])
            result[name + ".weight_scale"] = ("BF16", [n, k // 128])
            result[name + ".weight_shape"] = ("I64", [2])
        return result

    @property
    def parameter_bytes(self):
        plain = sum(
            2 * math.prod(shape)
            for name, (dtype, shape) in self.shapes().items()
            if dtype == "BF16" and not name.endswith(".weight_scale")
        )
        return plain + sum(2 * n * k for n, k in self.matrix_shapes().values())

    def memory_bytes(self, bits=16):
        if type(bits) is not int or bits != 16:
            raise ValueError("Sparkulator baseline decodes checkpoint INT4 exactly to BF16; bits16 required")
        # GPU decoded model + one destination matrix. Host64MiB read pieces and
        # row64 decode tiles are separate host-loading owners, not a model mirror.
        resident = self.parameter_bytes + 2 * (128 + 256) * 4 + 2
        return resident, resident + max(2 * n * k for n, k in self.matrix_shapes().values())

    @property
    def host_loading_bytes(self):
        # One original BF16 tensor (own536MB table is the largest), bounded
        # reader pieces, finite chunks and row64 decode; never a full CPU model.
        largest_native = max(
            2 * math.prod(shape)
            for name, (dtype, shape) in self.shapes().items()
            if dtype == "BF16" and not name.endswith(".weight_scale")
        )
        return 2 * (64 << 20) + largest_native + 64 * 8192 * 12 + (16 << 20)

    def state_bytes(self, context_limit):
        if type(context_limit) is not int or not 0 < context_limit <= self.max_position:
            raise ValueError("Sparkulator context exceeds native262144")
        return context_limit * 2 * 512 * 2 * 2 + 2 * self.backbone_hidden + 2 * (128 + 256) * 4

    def snapshot_bytes(self, context_limit):
        self.state_bytes(context_limit)
        return 2 * self.backbone_hidden  # derived full KV invalidates, never clones in a snapshot

    def workspace_bytes(self, context_limit, *, global_gqa=False):
        self.state_bytes(context_limit)
        if type(global_gqa) is not bool:
            raise ValueError("Sparkulator global GQA selection must be boolean")
        if global_gqa:
            return global_gqa_workspace_bytes(context_limit)
        # Native HF repeats KV to Q16. Pinned Torch MATH may retain BF16
        # repeated K/V, FP32 K/V upcasts AND an FP32 scaled-key allocation,
        # including scale1. Global/local attention execute sequentially: their
        # temporary peaks do not add. Chronological local gathers remain held
        # by the proposal view across all layers, so add those to either peak.
        math_bytes_per_element = (2 + 4) * 2 + 4
        global_math = context_limit * 16 * 512 * math_bytes_per_element
        local_length = min(context_limit, self.window)
        local_math = local_length * 16 * 256 * math_bytes_per_element
        local_gathers = local_length * 8 * 256 * 2 * 2
        return max(global_math, local_math) + local_gathers + (16 << 20)


def checked_headers(directory):
    from tensorfold.cuda.capacity import headers

    cfg = SparkulatorConfig.from_dict(read_json(directory, "config.json", required=True))
    inventory = headers(directory)
    expected = cfg.shapes()
    if set(inventory) != set(expected) or any(
        (v["dtype"], v["shape"]) != expected[name] for name, v in inventory.items()
    ):
        raise ValueError("Sparkulator requires all88 original packed/BF16 tensors and no extras")
    return cfg, inventory
