"""Shared native dense checkpoint text inventory, without runtime/provider imports."""
TEXT_PREFIX = "model.language_model."


def text_tensor_contract(cfg):
    """Exact text checkpoint storage, including unused shared-tail K/V tensors."""
    shapes = {TEXT_PREFIX+"embed_tokens.weight": (cfg.vocab, cfg.hidden),
              TEXT_PREFIX+"embed_tokens_per_layer.weight": (cfg.vocab, cfg.layers*cfg.per_layer_input),
              TEXT_PREFIX+"norm.weight": (cfg.hidden,),
              TEXT_PREFIX+"per_layer_model_projection.weight": (cfg.layers*cfg.per_layer_input, cfg.hidden),
              TEXT_PREFIX+"per_layer_projection_norm.weight": (cfg.per_layer_input,)}
    active, unused = set(), set()
    for index, kind in enumerate(cfg.layer_types):
        prefix = f"{TEXT_PREFIX}layers.{index}."
        dim = cfg.global_head_dim if kind == "full_attention" else cfg.head_dim
        matrices = {"self_attn.q_proj": (cfg.heads*dim, cfg.hidden),
                    "self_attn.k_proj": (cfg.kv_heads*dim, cfg.hidden),
                    "self_attn.v_proj": (cfg.kv_heads*dim, cfg.hidden),
                    "self_attn.o_proj": (cfg.hidden, cfg.heads*dim),
                    "mlp.gate_proj": (cfg.intermediate, cfg.hidden),
                    "mlp.up_proj": (cfg.intermediate, cfg.hidden),
                    "mlp.down_proj": (cfg.hidden, cfg.intermediate)}
        for suffix, shape in matrices.items():
            name = prefix+suffix+".weight"
            shapes[name] = shape
            if index >= cfg.layers-cfg.shared_layers and suffix in ("self_attn.k_proj", "self_attn.v_proj"):
                unused.add(name)
            else:
                active.add(name)
        shapes[prefix+"per_layer_input_gate.weight"] = (cfg.per_layer_input, cfg.hidden)
        shapes[prefix+"per_layer_projection.weight"] = (cfg.hidden, cfg.per_layer_input)
        for suffix in ("input_layernorm", "post_attention_layernorm", "post_feedforward_layernorm",
                       "pre_feedforward_layernorm", "post_per_layer_input_norm"):
            shapes[prefix+suffix+".weight"] = (cfg.hidden,)
        for suffix in ("q_norm", "k_norm"):
            name = prefix+"self_attn."+suffix+".weight"
            shapes[name] = (dim,)
            if suffix == "k_norm" and index >= cfg.layers-cfg.shared_layers:
                unused.add(name)
        shapes[prefix+"layer_scalar"] = (1,)
    return shapes, active, unused
