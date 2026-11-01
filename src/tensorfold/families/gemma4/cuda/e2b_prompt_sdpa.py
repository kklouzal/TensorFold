"""Native SDPA GQA prompt candidate without grouped-KV materialization.

The qualified CUDA runtime supports grouped-query SDPA with explicit masks and
D512. HF5.17's selection currently repeats these KV heads first. Preserve its
causal/mask contract while letting public SDPA consume the original KV owners.
No attention math or provider is implemented here.
"""
from torch.nn import functional as F


def prompt_sdpa_gqa(module, query, key, value, attention_mask, *, dropout=0.0, scaling=1.0,
                    is_causal=None):
    """Single Gemma text attention call; masks already own cached alignment.

    A supplied mask remains unchanged (including lower-right causal alignment,
    sliding boundaries and padding). With no mask, mirror native HF5.17's
    upper-left empty-StaticCache slicing and single-query noncausal dispatch.
    """
    causal = module.is_causal if is_causal is None else is_causal
    causal = query.shape[2] > 1 and attention_mask is None and causal
    if causal and key.shape[2] > query.shape[2]:
        key = key[:, :, :query.shape[2], :]
        value = value[:, :, :query.shape[2], :]
    result = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        scale=scaling, is_causal=causal, enable_gqa=True)
    return result.transpose(1, 2).contiguous()
