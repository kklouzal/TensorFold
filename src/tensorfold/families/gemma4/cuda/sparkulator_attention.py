"""Optional Spark global GQA; native reference remains available.

The caller owns one exclusive head lane. Only the global shared-KV layer is
replaced; local attention retains its native provider and allocation contract.
EFFICIENT-only is intentional: AUTO/MATH is not an admissible small-workspace
fallback. Actual SM121 forced-provider and whole-head gates remain required.
"""
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..sparkulator_metadata import global_gqa_workspace_bytes, validate_global_gqa_torch
from .assistant_attention import AssistantAttentionPolicy
from .e2b_attention import E2BTextAttention


def candidate_workspace_bytes(context_limit):
    """Pinned Torch efficient global inventory + native local/gather/aux owners.

    Excludes original KV, bridge-derived KV and head parameters, charged by their
    owners. This candidate profile MUST NOT replace baseline admission until the
    forced provider is qualified. Allocator rounding/driver pools stay external.
    """
    return global_gqa_workspace_bytes(context_limit)


def _validate_region(hidden, mask, shared, training):
    key, value = shared["full_attention"]
    p = key.shape[2]
    if (
        training or torch.is_grad_enabled() or mask is not None
        or hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (1, 1, 1024)
        or hidden.device.type != "cuda" or not 0 < p <= 131072
        or tuple(key.shape) != (1, 2, p, 512) or tuple(value.shape) != (1, 2, p, 512)
        or any(t.dtype != torch.bfloat16 or t.device != hidden.device
               or t.stride()[-3:] != (512, 1024, 1) for t in (key, value))
    ):
        raise ValueError("Spark efficient global candidate outside qualified-shape region")


class SparkGlobalAttention(E2BTextAttention):
    def forward(self, hidden_states, position_embeddings, attention_mask, shared_kv_states,
                past_key_values=None, **kwargs):
        if self.policy.reference_depth:
            return super().forward(hidden_states, position_embeddings, attention_mask, shared_kv_states,
                                   past_key_values=past_key_values, **kwargs)
        _validate_region(hidden_states, attention_mask, shared_kv_states, self.training)
        if past_key_values is not None or kwargs.get("is_causal", False):
            raise ValueError("Spark efficient global requires frozen shared KV/noncausal singleton query")
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            return super().forward(hidden_states, position_embeddings, attention_mask, shared_kv_states,
                                   past_key_values=None, **kwargs)


def install_spark_global_attention(model, *, enabled=False):
    """Prepare optional global proposal policy before weight publication.

    Wrapping borrows original native parameters. Construct before publication so
    allocation failure leaves the original model intact. Native-reference scopes
    use the returned policy's existing nested/exception-safe context manager.
    """
    if type(enabled) is not bool:
        raise ValueError("Spark global GQA selection must be boolean")
    if not enabled:
        return None
    validate_global_gqa_torch(torch)
    layers = model.model.layers
    if len(layers) != 4:
        raise ValueError("Spark global GQA requires its exact four-layer profile")
    for index, layer in enumerate(layers):
        attention = layer.self_attn
        if (
            type(attention) is not type(layers[0].self_attn)
            or not attention.is_kv_shared_layer
            or attention.config.num_attention_heads != 16
            or attention.layer_type != ("full_attention" if index == 3 else "sliding_attention")
            or attention.head_dim != (512 if index == 3 else 256)
            or attention.num_key_value_groups != (8 if index == 3 else 2)
            or attention.scaling != 1.0 or attention.attention_dropout != 0.0
        ):
            raise ValueError("Spark global GQA requires the native16-query/shared2-global8-local profile")
    policy = AssistantAttentionPolicy()
    wrapper = SparkGlobalAttention(layers[3].self_attn, policy)
    layers[3].self_attn = wrapper
    return policy
