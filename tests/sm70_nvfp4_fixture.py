"""Independent wire fixture: literal safetensors metadata and known stored rows.

No production packing/decode/planner routine creates its matrix payloads.
The family hash constants reuse their unchanged authoritative constructor.
"""

import json
from pathlib import Path
import struct

import numpy as np


def write(directory, *, ple=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    d, streams, low, heads, kv, hd, nk, nv, dk, dv, width, experts, vocab = 256, 4, 32, 8, 2, 256, 16, 48, 128, 128, 128, 2, 256
    tensors, quantized = {}, {}

    def add(name, value, dtype):
        tensors[name] = (np.ascontiguousarray(value) if np.asarray(value).ndim else np.asarray(value), dtype)

    def vector(name, size, *, centered=False):
        add(name, np.linspace(-.015625, .015625, size, dtype=np.float32) + (0.0 if centered else 1.0), "F32")

    def linear(name, n, k, *, packed=False):
        if packed:
            add(name + ".weight", np.arange(n * k // 2, dtype=np.uint8).reshape(n, k // 2), "U8")
            add(name + ".weight_scale", np.full((n, k // 16), 0x38, dtype=np.uint8), "F8_E4M3")
            add(name + ".weight_scale_2", np.array(.125, dtype=np.float32), "F32")
            quantized[name] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        else:
            rows, columns = np.arange(n, dtype=np.float32)[:, None], np.arange(k, dtype=np.float32)[None]
            add(name + ".weight", ((rows % 31) / 32 + (columns % 17) / 1024).astype(np.float16), "F16")

    def hc(base, inject=True):
        linear(base + ".input_mix_weight_down", low, streams * d)
        linear(base + ".input_mix_weight_up", streams * d, low)
        if inject:
            linear(base + ".block_inject_weight", streams, streams * d)
        vector(base + ".hc_norm.weight", streams * d, centered=True)

    linear("model.embed_tokens", vocab, d)
    for layer in range(2):
        base = f"model.layers.{layer}"
        hc(base + ".attn_hyper_connection")
        hc(base + ".mlp_hyper_connection")
        linear(base + ".mlp.gate", experts, d)
        linear(base + ".mlp.shared_expert_gate", 1, d)
        for expert in range(experts + 1):
            stem = base + (".mlp.shared_expert" if expert == experts else f".mlp.experts.{expert}")
            linear(stem + ".gate_proj", width, d, packed=expert != experts)
            linear(stem + ".up_proj", width, d, packed=expert != experts)
            linear(stem + ".down_proj", d, width, packed=expert != experts)
        if layer == 0:
            stem = base + ".linear_attn"
            channels = 2 * nk * dk + nv * dv
            for name, n, k in (("in_proj_qkv", channels, d), ("in_proj_z", nv * dv, d),
                               ("in_proj_b", nv, d), ("in_proj_a", nv, d), ("out_proj", d, nv * dv)):
                linear(stem + "." + name, n, k)
            add(stem + ".conv1d.weight", np.arange(channels * 4, dtype=np.float16).reshape(channels, 4) / 32768, "F16")
            add(stem + ".A_log", np.linspace(-4, -1, nv, dtype=np.float32), "F32")
            vector(stem + ".dt_bias", nv, centered=True)
            vector(stem + ".norm.weight", dv)
        else:
            stem = base + ".self_attn"
            for name, n, k in (("q_proj", 2 * heads * hd, d), ("k_proj", kv * hd, d), ("v_proj", kv * hd, d),
                               ("o_proj", d, heads * hd), ("indexer.index_qk_proj", 5 * 128, d)):
                linear(stem + "." + name, n, k)
            for name, n in (("q_norm", hd), ("k_norm", hd), ("indexer.q_layernorm", 128), ("indexer.k_layernorm", 128)):
                vector(stem + "." + name + ".weight", n, centered=True)
    hc("model.hyper_connection_mixer", False)
    linear("lm_head", vocab, d)
    text = {"hidden_size": d, "num_hidden_layers": 2, "vocab_size": vocab, "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
            "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": hd,
            "layer_types": ["linear_attention", "full_attention"], "linear_num_key_heads": nk,
            "linear_num_value_heads": nv, "linear_key_head_dim": dk, "linear_value_head_dim": dv,
            "linear_conv_kernel_dim": 4, "num_experts": experts, "num_experts_per_tok": 2,
            "moe_intermediate_size": width, "shared_expert_intermediate_size": width,
            "hc_count": streams, "hc_lowrank": low, "indexer_n_heads": 4, "indexer_head_dim": 128,
            "indexer_budget": 20, "indexer_compress_ratio": 4, "ple_layer_ids": [], "ple_embed_dim": 128,
            "ple_conv_kernel_size": 4, "ngram_size": 3, "heads_per_ngram": 1,
            "ngram_vocab_size_base": vocab, "make_ngram_vocab_size_divisible_by": 1, "split_ngram_parts": 1,
            "seed": 1234, "eos_token_id": 0, "output_gate_type": "sigmoid", "hidden_act": "silu",
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": .25}}
    if ple:
        from tensorfold.families.qwen4_exp.cuda.ngram import NGram

        ng = NGram(vocab=vocab, ngram_size=3, heads_per_ngram=1, vocab_base=vocab, divisor=1,
                   seed=1234, eos=0, embed_dim=128, ple_index=0)
        text["ple_layer_ids"] = [2]
        stem = "model.layers.1.ple"
        linear(stem + ".key_proj", streams * d, 128)
        linear(stem + ".value_proj", d, 128)
        for name in ("norm_key", "norm_query", "norm_conv"):
            vector(stem + "." + name + ".weight", streams * d, centered=True)
        add(stem + ".conv1d.weight", np.full((streams * d, 4), .125, dtype=np.float16), "F16")
        for name, values in (("layer_multipliers", ng.multipliers), ("ngram_heads_offsets", ng.head_offsets),
                             ("ngram_heads_vocab_sizes", ng.head_sizes)):
            add(stem + ".ple_embedding." + name, values, "I64")
        base = stem + ".ple_embedding.ngram_embedding"
        if ple == "bf16":
            values = np.full((ng.rows, ng.dims), .75, dtype=np.float32)
            add(base + ".shard_0.weight", (values.view(np.uint32) >> 16).astype(np.uint16), "BF16")
        elif ple == "fp8":
            add(base + ".shard_0.weight", np.full((ng.rows, ng.dims), 0x38, dtype=np.uint8), "F8_E4M3")
            add(base + ".weight_scale", np.array(.5, dtype=np.float32), "F32")
        else:
            add(base + ".shard_0.weight", np.full((ng.rows, ng.dims // 2), 0x32, dtype=np.uint8), "U8")
            add(base + ".shard_0.weight_scale", np.full((ng.rows, ng.dims // 16), 0x38, dtype=np.uint8), "F8_E4M3")
            add(base + ".weight_scale_2", np.array(.5, dtype=np.float32), "F32")
    config = {"model_type": "qwen4_exp", "text_config": text,
              "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                      "quantized_layers": quantized}}
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {
        "type": "WordLevel", "vocab": {f"t{i}": i for i in range(vocab)}, "unk_token": "t0"}}))
    (directory / "tokenizer_config.json").write_text(json.dumps({"eos_token": "t0", "bos_token": "t0",
        "chat_template": "{% for m in messages %}{{m.content}}{% endfor %}"}))
    entries, blobs, offset = {}, [], 0
    for name, (value, dtype) in tensors.items():
        data = value.tobytes()
        entries[name] = {"dtype": dtype, "shape": list(value.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    header = json.dumps(entries, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    with (directory / "model.safetensors").open("wb") as target:
        target.write(struct.pack("<Q", len(header)))
        target.write(header)
        for blob in blobs:
            target.write(blob)
    return directory, tensors
