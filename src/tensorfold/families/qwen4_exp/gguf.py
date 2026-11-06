"""Flash Next GGUF semantic configuration, token identity and CPU decode oracle.

GGUF/ggml format: https://github.com/ggml-org/ggml (MIT).
"""

import hashlib
import math
from pathlib import Path

from tensorfold.gguf_file import FORMATS, GGUFCheckpoint, resolve_gguf


def token_id_identity(metadata):
    """Ordered shared token space and roles, independent of frontend rendering.

    Companion normalization/insertion defaults and templates never choose the
    target frontend. Token types, merges and mapped BOS/EOS/padding identities
    still have to agree when supplied by the paired GGUF inventories.
    """
    tokens = metadata.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, tuple) or not tokens:
        raise ValueError("GGUF token-ID identity requires its complete ordered token strings")
    digest = hashlib.sha256(b"tensorfold.gguf-token-ids-v2\0")
    for field in ("tokens", "token_type", "merges", "bos_token_id", "eos_token_id", "padding_token_id"):
        value = metadata.get("tokenizer.ggml." + field)
        digest.update(field.encode() + b"\0")
        if value is None:
            digest.update(b"absent\0")
            continue
        if field.endswith("_id"):
            if type(value) is not int or not 0 <= value < len(tokens):
                raise ValueError("GGUF mapped token role must identify an existing vocabulary row")
            digest.update(value.to_bytes(8, "little"))
            continue
        if (not isinstance(value, tuple)
                or (field == "token_type" and (len(value) != len(tokens)
                    or any(type(item) is not int or not 0 <= item <= 6 for item in value)))
                or (field != "token_type" and any(type(item) is not str for item in value))):
            raise ValueError("GGUF shared token arrays have invalid types or row counts")
        digest.update(len(value).to_bytes(8, "little"))
        for item in value:
            encoded = item.to_bytes(8, "little") if field == "token_type" else item.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def _family_checkpoint(checkpoint):
    if checkpoint.metadata.get("general.architecture") != "qwen4exp":
        raise ValueError("this GGUF boundary accepts the qwen4exp Flash Next architecture only")
    return checkpoint


def open_checkpoint(entrypoint):
    """Admit the Flash Next family before any family-specific allocation."""
    return _family_checkpoint(GGUFCheckpoint.open(entrypoint))


def metadata_config(path, *, role="target"):
    """Resolve a Flash Next GGUF and build its role-specific configuration."""
    if role not in ("target", "mtp"):
        raise ValueError("GGUF metadata role must be target or mtp")
    entry = resolve_gguf(path)
    if entry is None:
        raise ValueError("no GGUF checkpoint was selected")
    return metadata_config_from_checkpoint(GGUFCheckpoint.open(entry), entry, role=role)


def metadata_config_from_checkpoint(checkpoint, entry, *, role="target"):
    """Consume an already validated immutable inventory without reopening headers.

    Keep the selected entrypoint distinct from the ordered shard set: an explicit
    second-shard selection remains the recorded GGUF entrypoint.
    """
    if role not in ("target", "mtp"):
        raise ValueError("GGUF metadata role must be target or mtp")
    entry = Path(entry)
    if entry not in checkpoint.paths:
        raise ValueError("GGUF configuration entrypoint is not a shard of the validated inventory")
    checkpoint = _family_checkpoint(checkpoint)
    fields = checkpoint.metadata

    def positive(key, default=None):
        value = fields.get("qwen4exp." + key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"GGUF qwen4exp.{key} must be a positive integer")
        return value

    hidden = positive("embedding_length")
    layers = positive("block_count")
    if role == "mtp":
        prefix = f"blk.{layers - 1}."
        if (type(fields.get("qwen4exp.nextn_predict_layers")) is not int
                or fields["qwen4exp.nextn_predict_layers"] != 1
                or any(name not in ("token_embd.weight", "output.weight") and not name.startswith(prefix)
                       for name in checkpoint.tensors)
                or not all(prefix + name in checkpoint.tensors for name in (
                    "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.eh_proj.weight"))):
            raise ValueError("GGUF MTP role requires one physical prediction core without target trunk tensors")
        head_dim, rotary = positive("attention.key_length"), positive("rope.dimension_count")
        ratio = fields.get("qwen4exp.attention.compress_ratios")
        if (not isinstance(ratio, tuple) or len(ratio) != layers
                or any(type(n) is not int or n not in (0, 4) for n in ratio)):
            raise ValueError("GGUF MTP compression metadata differs from its declared QSA family")
        epsilon = fields.get("qwen4exp.attention.layer_norm_rms_epsilon")
        theta = fields.get("qwen4exp.rope.freq_base")
        if (type(epsilon) is not float or not math.isfinite(epsilon) or epsilon <= 0
                or type(theta) is not float or not math.isfinite(theta) or theta <= 0):
            raise ValueError("GGUF MTP normalization and RoPE parameters must be positive finite floats")
        sections = fields.get("qwen4exp.rope.dimension_sections")
        if (not isinstance(sections, tuple) or len(sections) < 3
                or any(type(n) is not int for n in sections)):
            raise ValueError("GGUF MTP requires its actual mRoPE section metadata")
        text = {
            "model_type": "qwen4_exp_text", "hidden_size": hidden,
            "num_hidden_layers": 1, "layer_types": ["full_attention"],
            "num_attention_heads": positive("attention.head_count"),
            "num_key_value_heads": positive("attention.head_count_kv"), "head_dim": head_dim,
            "rms_norm_eps": epsilon, "max_position_embeddings": positive("context_length"),
            "rope_parameters": {"rope_type": "default", "rope_theta": theta,
                                "partial_rotary_factor": rotary / head_dim,
                                "mrope_section": list(sections[:3]), "mrope_interleaved": True},
            "mtp": {"num_hidden_layers": 1, "layer_types": ["full_attention"]},
            "indexer_compress_ratio": 4, "indexer_budget": positive("attention.indexer.top_k"),
            "gguf_role": "mtp", "gguf_source_block_count": layers,
            "gguf_source_final_compression_ratio": ratio[-1], "gguf_entrypoint": str(entry),
        }
        return {"model_type": "qwen4_exp", "text_config": text,
                "quantization_config": {"quant_method": "gguf"}, "gguf_entrypoint": str(entry)}
    interval = positive("full_attention_interval")
    head_dim = positive("attention.key_length")
    rotary = positive("rope.dimension_count")
    ple_heads = positive("ple.heads_per_ngram")
    ngram_size = positive("ple.ngram_size")
    dims_per_head = positive("embedding_length_per_layer_input")
    ple_dim = dims_per_head * ple_heads * (ngram_size - 1)
    vocab = len(fields.get("tokenizer.ggml.tokens", ()))
    if vocab <= 0:
        vocab = checkpoint.tensors["token_embd.weight"].shape[0]
    eos = fields.get("tokenizer.ggml.eos_token_id")
    ple_eos = fields.get("qwen4exp.ple.eos_token_id")
    if ((ple_eos is None and fields.get("qwen4exp.ple.layers"))
            or (ple_eos is not None and (type(ple_eos) is not int or not 0 <= ple_eos < vocab))):
        raise ValueError("GGUF PLE layers require an explicit segmentation ID within the vocabulary")
    sections = tuple(fields.get("qwen4exp.rope.dimension_sections", (11, 11, 10)))[:3]
    ratios = [v for v in fields.get("qwen4exp.attention.compress_ratios", ()) if v]
    ratio = ratios[0] if ratios else 4
    if any(v != ratio for v in ratios):
        raise ValueError("Flash Next GGUF requires one common sparse-indexer compression ratio")
    text = {
        "model_type": "qwen4_exp_text", "hidden_size": hidden, "vocab_size": vocab,
        "num_hidden_layers": layers,
        "layer_types": ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(layers)],
        "num_attention_heads": positive("attention.head_count"),
        "num_key_value_heads": positive("attention.head_count_kv"), "head_dim": head_dim,
        "rms_norm_eps": float(fields["qwen4exp.attention.layer_norm_rms_epsilon"]),
        "max_position_embeddings": positive("context_length"), "partial_rotary_factor": rotary / head_dim,
        "rope_parameters": {"rope_type": "default", "rope_theta": float(fields["qwen4exp.rope.freq_base"]),
                            "partial_rotary_factor": rotary / head_dim,
                            "mrope_section": list(sections), "mrope_interleaved": True},
        "mrope_section": list(sections), "mrope_interleaved": True,
        "linear_num_key_heads": positive("ssm.group_count"),
        "linear_num_value_heads": positive("ssm.time_step_rank"),
        "linear_key_head_dim": positive("ssm.state_size"),
        "linear_value_head_dim": positive("ssm.inner_size") // positive("ssm.time_step_rank"),
        "linear_conv_kernel_dim": positive("ssm.conv_kernel"),
        "num_experts": positive("expert_count"), "num_experts_per_tok": positive("expert_used_count"),
        "moe_intermediate_size": positive("expert_feed_forward_length"),
        "shared_expert_intermediate_size": positive("expert_shared_feed_forward_length"),
        "hc_count": positive("hyper_connection.count"), "hc_lowrank": positive("hyper_connection.low_rank"),
        "indexer_n_heads": positive("attention.indexer.head_count"),
        "indexer_head_dim": positive("attention.indexer.key_length"),
        "indexer_budget": positive("attention.indexer.top_k"), "indexer_compress_ratio": ratio,
        "ple_layer_ids": [i + 1 for i in fields.get("qwen4exp.ple.layers", ())], "ple_embed_dim": ple_dim,
        "ple_conv_kernel_size": positive("ple.conv_kernel"), "ngram_size": ngram_size,
        "heads_per_ngram": ple_heads, "ngram_vocab_size_base": 20_000_000,
        "make_ngram_vocab_size_divisible_by": 128, "split_ngram_parts": 1,
        "seed": 1234, "bos_token_id": fields.get("tokenizer.ggml.bos_token_id", eos),
        "eos_token_id": eos, "output_gate_type": "sigmoid", "hidden_act": "silu",
        "dtype": "float16", "mamba_ssm_dtype": "float32", "tie_word_embeddings": False,
        "gguf_architecture": "qwen4exp", "gguf_entrypoint": str(entry),
        "gguf_ple_eos_token_id": ple_eos,
        "gguf_ple_constants": {"layer_multipliers": list(fields.get("qwen4exp.ple.layer_multipliers", ())),
                               "ngram_heads_offsets": list(fields.get("qwen4exp.ple.head_offsets", ())),
                               "ngram_heads_vocab_sizes": list(fields.get("qwen4exp.ple.head_vocab_sizes", ()))},
    }
    from .frontend_metadata import generation_defaults

    generation = generation_defaults(fields)
    return {"model_type": "qwen4_exp", "architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": text, "eos_token_id": eos, "bos_token_id": text["bos_token_id"],
            **({"generation_eos_token_ids": generation["eos_token_id"],
                "generation_eos_token_source": generation["tensorfold_source"]} if generation else {}),
            "quantization_config": {"quant_method": "gguf"}, "gguf_entrypoint": str(entry)}


def decode_rows(tensor, packed):
    """CPU format oracle for the original mixed checkpoint's packed row encodings."""
    import numpy as np

    raw = np.asarray(packed)
    if raw.dtype != np.uint8 or raw.ndim != 2 or raw.shape[1] != tensor.row_bytes:
        raise ValueError("decode expects uint8 [rows, exact packed row bytes]")
    raw = np.ascontiguousarray(raw)
    if tensor.ggml_type in (14, 18, 21, 22, 23, 42):
        from .gguf_iq import decode

        return decode(raw, tensor.ggml_type, tensor.shape[-1], FORMATS[tensor.ggml_type][2])
    if tensor.encoding in ("F32", "F16"):
        return raw.view("<f4" if tensor.encoding == "F32" else "<f2").astype(np.float32)
    if tensor.encoding == "BF16":
        return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
    if tensor.encoding in ("Q8_0", "IQ4_NL"):
        block_bytes = FORMATS[tensor.ggml_type][2]
        blocks = raw.reshape(raw.shape[0], tensor.shape[-1] // 32, block_bytes)
        scale = blocks[..., :2].copy().view("<f2").astype(np.float32)
        payload = blocks[..., 2:]
        if tensor.encoding == "Q8_0":
            values = payload.view(np.int8).astype(np.float32)
        else:
            from .gguf_tables import KVALUES_IQ4NL

            grid = np.array(KVALUES_IQ4NL, np.float32)
            values = grid[np.concatenate((payload & 15, payload >> 4), axis=-1)]
        return (values * scale).reshape(raw.shape[0], tensor.shape[-1])
    if tensor.encoding == "Q5_1":
        blocks = raw.reshape(-1, 24)
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)
        minimum = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
        high = blocks[:, 4:8].copy().view("<u4")
        indices = np.arange(32, dtype=np.uint32)
        low = (blocks[:, 8 + indices % 16] >> (4 * (indices // 16))) & 15
        codes = low | (((high >> indices) & 1) << 4)
        return (d * codes.astype(np.float32) + minimum).reshape(raw.shape[0], tensor.shape[-1])
    if tensor.encoding in ("Q4_K", "Q5_K"):
        size = FORMATS[tensor.ggml_type][2]
        blocks = raw.reshape(-1, size)
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)
        minimum = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
        packed_scales = blocks[:, 4:16]
        scales = np.concatenate((packed_scales[:, :4] & 63,
                                 (packed_scales[:, 8:12] & 15) | ((packed_scales[:, :4] >> 2) & 48)), axis=1)
        minima = np.concatenate((packed_scales[:, 4:8] & 63,
                                 (packed_scales[:, 8:12] >> 4) | ((packed_scales[:, 4:8] >> 2) & 48)), axis=1)
        indices = np.arange(256)
        group = indices // 32
        qstart = 16 if tensor.encoding == "Q4_K" else 48
        codes = ((blocks[:, qstart + (indices // 64) * 32 + indices % 32]
                  >> ((group % 2) * 4)) & 15).astype(np.uint8)
        if tensor.encoding == "Q5_K":
            codes |= (((blocks[:, 16 + indices % 32] >> group) & 1) << 4).astype(np.uint8)
        values = ((d * scales.astype(np.float32))[:, group] * codes.astype(np.float32)
                  - (minimum * minima.astype(np.float32))[:, group])
        return values.reshape(raw.shape[0], tensor.shape[-1])
    raise NotImplementedError(f"{tensor.encoding} decode/operator integration is pending")
