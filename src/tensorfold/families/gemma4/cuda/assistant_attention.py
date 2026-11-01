"""Opt-in native assistant GQA dispatch over immutable target-shared KV.

All four admitted assistant layers are KV-sharing. Their native mask builder
and frozen anchor position remain untouched. Reuse the owned text-attention
projection/RoPE boundary with a permanent prompt policy so the target's custom
eight-query-head decoder kernel is unreachable for this four-head model.
"""
from contextlib import contextmanager

from .e2b_attention import E2BTextAttention
from .e2b_rows import E2BRowPolicy


class AssistantAttentionPolicy(E2BRowPolicy):
    def __init__(self):
        super().__init__(prompt_sdpa_gqa=True)

    @property
    def prompt(self):
        return True

    @prompt.setter
    def prompt(self, value):
        if value is not True:
            raise ValueError("native assistant GQA cannot enter the target decoder attention path")


def install_assistant_attention(model, *, enabled=False):
    """Borrow native children only for the explicitly enabled exact head profile.

    Disabled returns without replacing modules or adding a hot policy check.
    Constructor ownership serializes installation before any model forward.
    """
    if type(enabled) is not bool:
        raise ValueError("native assistant GQA selection must be boolean")
    if not enabled:
        return None
    layers = model.model.layers
    if len(layers) != 4 or any(not layer.self_attn.is_kv_shared_layer for layer in layers):
        raise ValueError("native assistant GQA requires four exclusively shared-KV layers")
    if any(layer.self_attn.config.num_attention_heads != 4
           or layer.self_attn.num_key_value_groups != 2 for layer in layers):
        raise ValueError("native assistant GQA requires its native four-query/two-KV head geometry")
    policy = AssistantAttentionPolicy()
    for layer in layers:
        layer.self_attn = E2BTextAttention(layer.self_attn, policy)
    return policy


@contextmanager
def assistant_native_reference(model, policy):
    """Yield the public HF oracle with native attention even when GQA is enabled.

    The caller owns the exclusive head/target lane and drains CUDA reference
    consumers before handing tensors to another stream. Nesting and exceptions
    restore the candidate selection without mutating weights or shared KV.
    """
    if policy is None:
        yield model
    else:
        with policy.native_reference():
            yield model
