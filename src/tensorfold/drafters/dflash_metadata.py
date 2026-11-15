"""Tensor inventory for the shared Qwen3-style DFlash companion architecture.

Family owners validate their own metadata before using this shape contract.
This module is header-only and imports no device runtime.
"""

def shapes(self):
    out = {"fc.weight": (self.hidden, len(self.tap_layers) * self.hidden),
           "hidden_norm.weight": (self.hidden,), "norm.weight": (self.hidden,)}
    for i in range(self.layers):
        base = f"layers.{i}."
        out.update({base + key: shape for key, shape in {
            "input_layernorm.weight": (self.hidden,), "post_attention_layernorm.weight": (self.hidden,),
            "self_attn.q_norm.weight": (self.head_dim,), "self_attn.k_norm.weight": (self.head_dim,),
            "self_attn.q_proj.weight": (self.heads * self.head_dim, self.hidden),
            "self_attn.k_proj.weight": (self.kv_heads * self.head_dim, self.hidden),
            "self_attn.v_proj.weight": (self.kv_heads * self.head_dim, self.hidden),
            "self_attn.o_proj.weight": (self.hidden, self.heads * self.head_dim),
            "mlp.gate_proj.weight": (self.width, self.hidden), "mlp.up_proj.weight": (self.width, self.hidden),
            "mlp.down_proj.weight": (self.hidden, self.width)}.items()})
    return out
