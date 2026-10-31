"""Independent tiny mixed-format GGUF files for loader/operator tests."""

from pathlib import Path
import struct

import numpy as np


def blocks(kind, rows, width, seed=19):
    rng = np.random.default_rng(seed)
    size, block = {0: (4, 1), 1: (2, 1), 30: (2, 1), 7: (24, 32), 8: (34, 32),
                   12: (144, 256), 13: (176, 256), 20: (18, 32)}[kind]
    if width % block:
        raise ValueError("test tensor width must preserve its original blocks")
    if kind == 0:
        return (rng.normal(size=(rows, width)) * .01).astype("<f4").view(np.uint8).reshape(rows, -1)
    if kind in (1, 30):
        values = (rng.normal(size=(rows, width)) * .01).astype("<f4")
        return (values.astype("<f2") if kind == 1 else (values.view("<u4") >> 16).astype("<u2")).view(np.uint8)
    raw = rng.integers(0, 256, size=(rows * width // block, size), dtype=np.uint8)
    raw[:, :2] = np.frombuffer(struct.pack("<e", .001), np.uint8)
    if kind in (7, 12, 13):
        raw[:, 2:4] = np.frombuffer(struct.pack("<e", .0005 if kind != 7 else -.015), np.uint8)
    return raw.reshape(rows, -1)


def scalar_reference(raw, kind, width):
    """Slow scalar FP32 dequantizer, intentionally unlike the vectorized production decoder."""
    raw = np.asarray(raw, np.uint8)
    storage, block = {0: (4, 1), 1: (2, 1), 30: (2, 1), 7: (24, 32), 8: (34, 32),
                      12: (144, 256), 13: (176, 256), 20: (18, 32)}[kind]
    result = np.empty((raw.shape[0], width), np.float32)
    book = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
    for row in range(raw.shape[0]):
        for col in range(width):
            b = bytes(raw[row, col // block * storage:(col // block + 1) * storage])
            i = col % block
            if kind == 0:
                value = struct.unpack("<f", b)[0]
            elif kind == 1:
                value = struct.unpack("<e", b)[0]
            elif kind == 30:
                value = struct.unpack("<f", b"\0\0" + b)[0]
            else:
                d = np.float32(struct.unpack("<e", b[:2])[0])
                if kind == 8:
                    value = d * struct.unpack("<b", b[2 + i:3 + i])[0]
                elif kind == 20:
                    value = d * book[(b[2 + i % 16] >> (4 * (i // 16))) & 15]
                elif kind == 7:
                    high = int.from_bytes(b[4:8], "little")
                    q = ((b[8 + i % 16] >> (4 * (i // 16))) & 15) | (((high >> i) & 1) << 4)
                    value = np.float32(d * q) + np.float32(struct.unpack("<e", b[2:4])[0])
                else:
                    g = i // 32
                    scales = b[4:16]
                    scale = scales[g] & 63 if g < 4 else (scales[g + 4] & 15) | ((scales[g - 4] >> 6) << 4)
                    minimum = scales[g + 4] & 63 if g < 4 else (scales[g + 4] >> 4) | ((scales[g] >> 6) << 4)
                    q = (b[(16 if kind == 12 else 48) + i // 64 * 32 + i % 32] >> (4 * (g % 2))) & 15
                    if kind == 13:
                        q |= ((b[16 + i % 32] >> g) & 1) << 4
                    dm = np.float32(struct.unpack("<e", b[2:4])[0])
                    value = np.float32(np.float32(d * scale) * q) - np.float32(dm * minimum)
            result[row, col] = value
    return result


def write(path, tensors, metadata):
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    def value(item):
        if isinstance(item, bool):
            return struct.pack("<I?", 7, item)
        if isinstance(item, str):
            return struct.pack("<I", 8) + string(item)
        if isinstance(item, float):
            return struct.pack("<If", 6, item)
        if isinstance(item, (tuple, list)):
            if item and all(type(v) is str for v in item):
                return struct.pack("<IIQ", 9, 8, len(item)) + b"".join(string(v) for v in item)
            if item and all(type(v) is bool for v in item):
                return struct.pack("<IIQ", 9, 7, len(item)) + struct.pack("<" + "?" * len(item), *item)
            if item and all(type(v) is float for v in item):
                return struct.pack("<IIQ", 9, 6, len(item)) + struct.pack("<" + "f" * len(item), *item)
            return struct.pack("<IIQ", 9, 5, len(item)) + b"".join(struct.pack("<i", i) for i in item)
        return struct.pack("<Iq", 11, item)

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for name, item in metadata.items():
        header += string(name) + value(item)
    payload = bytearray()
    for name, shape, kind, data in tensors:
        header += string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *reversed(shape))
        header += struct.pack("<IQ", kind, len(payload))
        payload.extend(np.asarray(data, np.uint8).tobytes())
        payload.extend(b"\0" * (-len(payload) % 32))
    header += b"\0" * (-len(header) % 32)
    Path(path).write_bytes(header + payload)
    return Path(path)


def tiny_model(path, *, linear_heads=(8, 16), linear_head_dim=32):
    """Two blocks, with real TP8 width80 and independently packed experts."""
    prefix = "qwen4exp."
    nk, nv = linear_heads
    conv = (2 * nk + nv) * linear_head_dim
    fields = {
        "general.architecture": "qwen4exp", "tokenizer.ggml.eos_token_id": 14,
        "tokenizer.ggml.bos_token_id": 13,
        **{prefix + name: item for name, item in {
            "block_count": 2, "context_length": 256, "embedding_length": 256,
            "attention.head_count": 8, "attention.head_count_kv": 2, "attention.key_length": 32,
            "attention.layer_norm_rms_epsilon": 1e-6, "rope.dimension_count": 8,
            "rope.dimension_sections": (1, 1, 2, 0), "rope.freq_base": 10000.0,
            "expert_count": 2, "expert_used_count": 1,
            "expert_feed_forward_length": 640, "expert_shared_feed_forward_length": 640,
            "ssm.conv_kernel": 4, "ssm.state_size": linear_head_dim, "ssm.group_count": nk,
            "ssm.time_step_rank": nv, "ssm.inner_size": nv * linear_head_dim, "full_attention_interval": 2,
            "hyper_connection.count": 4, "hyper_connection.low_rank": 32,
            "attention.indexer.head_count": 2, "attention.indexer.key_length": 32,
            "attention.indexer.top_k": 8, "attention.compress_ratios": (0, 4),
            "ple.layers": (), "ple.ngram_size": 3, "ple.heads_per_ngram": 8,
            "ple.conv_kernel": 4, "ple.eos_token_id": 13, "embedding_length_per_layer_input": 32,
        }.items()},
    }
    tensors = []

    def add(name, shape, kind=8, constant=None):
        rows, width = int(np.prod(shape[:-1])), shape[-1]
        data = blocks(kind, rows, width)
        if constant is not None:
            data = np.full(shape, constant, "<f4").view(np.uint8).reshape(rows, -1)
        tensors.append((name, shape, kind, data))

    def hc(base, inject=True):
        add(base + "down.weight", (32, 1024))
        add(base + "up.weight", (1024, 32))
        add(base + "norm.weight", (1024,), 0, 1)
        if inject:
            add(base + "inject.weight", (4, 1024), 0)

    add("token_embd.weight", (32, 256))
    add("output.weight", (32, 256))
    hc("output_hc_", False)
    for layer in range(2):
        base = f"blk.{layer}."
        hc(base + "hc_attn_")
        hc(base + "hc_ffn_")
        add(base + "ffn_gate_exps.weight", (2, 640, 256), 12)
        add(base + "ffn_up_exps.weight", (2, 640, 256), 12)
        add(base + "ffn_down_exps.weight", (2, 256, 640), 7)
        add(base + "ffn_gate_shexp.weight", (640, 256))
        add(base + "ffn_up_shexp.weight", (640, 256))
        add(base + "ffn_down_shexp.weight", (256, 640))
        add(base + "ffn_gate_inp.weight", (2, 256), 0)
        add(base + "ffn_gate_inp_shexp.weight", (256,), 0)
        if layer == 0:
            add(base + "attn_qkv.weight", (conv, 256))
            add(base + "attn_gate.weight", (nv * linear_head_dim, 256))
            add(base + "ssm_alpha.weight", (nv, 256), 0)
            add(base + "ssm_beta.weight", (nv, 256), 0)
            add(base + "ssm_a", (nv,), 0, -1)
            add(base + "ssm_dt.bias", (nv,), 0)
            add(base + "ssm_norm.weight", (linear_head_dim,), 0, 1)
            add(base + "ssm_conv1d.weight", (conv, 4), 0)
            add(base + "ssm_out.weight", (256, nv * linear_head_dim))
        else:
            add(base + "attn_q.weight", (512, 256))
            add(base + "attn_k.weight", (64, 256))
            add(base + "attn_v.weight", (64, 256))
            add(base + "attn_output.weight", (256, 256))
            add(base + "indexer.q_proj.weight", (64, 256), 30)
            add(base + "indexer.k_proj.weight", (32, 256), 30)
            for name in ("attn_q_norm", "attn_k_norm", "indexer.q_norm", "indexer.k_norm"):
                add(base + name + ".weight", (32,), 0, 1)
    return write(path, tensors, fields)
