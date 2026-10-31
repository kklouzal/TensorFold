"""Bounded GGUF v3 inventory and packed-row IO for Flash Next bring-up.

This is a checkpoint input boundary, not a conversion to MLX affine/NVFP4.
Shape order, original ggml encoding and file extents remain explicit. No model
weights are read while opening the inventory, including the enormous PLE table.
GGUF/ggml format: https://github.com/ggml-org/ggml (MIT).
"""

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import struct
from types import MappingProxyType

# ggml type -> (name, logical block width, stored block bytes).
# Pinned GGML layouts, including the complete SC117 mixed IQ3_S source set.
FORMATS = MappingProxyType({
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    18: ("IQ3_XXS", 256, 98), 20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136), 30: ("BF16", 1, 2), 42: ("Q2_0", 64, 18),
})
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_SPLIT = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf")
_HEADER_LIMIT = 64 << 20


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


class _Header:
    def __init__(self, stream):
        self.stream = stream
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("GGUF inputs must be regular files")
        self.size = info.st_size

    def read(self, length):
        if length < 0 or self.stream.tell() + length > min(self.size, _HEADER_LIMIT):
            raise ValueError("truncated or oversized GGUF header")
        data = self.stream.read(length)
        if len(data) != length:
            raise ValueError("short GGUF header read")
        return data

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self):
        return self.read(self.number("Q")).decode("utf-8")

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 9:
            element, count = self.number("I"), self.number("Q")
            if element == 9 or element not in (*_SCALARS, 8) or count > 1_000_000:
                raise ValueError("unsupported or oversized GGUF metadata array")
            # Check the minimum encoded span before allocating the Python list.
            minimum = 8 if element == 8 else struct.calcsize("<" + _SCALARS[element])
            if count * minimum > min(self.size, _HEADER_LIMIT) - self.stream.tell():
                raise ValueError("truncated GGUF metadata array")
            return tuple(self.value(element) for _ in range(count))
        if kind not in _SCALARS:
            raise ValueError(f"unknown GGUF metadata type {kind}")
        return self.number(_SCALARS[kind])


@dataclass(frozen=True)
class PackedTensor:
    name: str
    shape: tuple[int, ...]  # NumPy/Torch order; GGUF stores dimensions reversed.
    ggml_type: int
    path: Path
    offset: int
    nbytes: int

    @property
    def encoding(self):
        return FORMATS[self.ggml_type][0]

    @property
    def rows(self):
        return math.prod(self.shape[:-1])

    @property
    def row_bytes(self):
        _, block, size = FORMATS[self.ggml_type]
        return self.shape[-1] // block * size

    def row_span(self, start, count):
        if (type(start) is not int or type(count) is not int
                or start < 0 or count < 0 or start > self.rows or count > self.rows - start):
            raise ValueError("packed GGUF row range is outside the tensor")
        return self.offset + start * self.row_bytes, count * self.row_bytes

    def column_span(self, start, count):
        """A per-row byte slice only if both logical boundaries preserve blocks.

        TP8's 640/8=80 is deliberately rejected for block32/block256 storage;
        the later loader must decode/repack or choose another decomposition.
        """
        _, block, size = FORMATS[self.ggml_type]
        if (type(start) is not int or type(count) is not int or start < 0 or count <= 0
                or start + count > self.shape[-1] or start % block or count % block):
            raise ValueError(f"{self.encoding} column shard must preserve complete {block}-value blocks")
        return start // block * size, count // block * size

    def read_rows(self, reader, start, count, *, device="cpu", max_bytes=64 << 20):
        """Borrow the existing owned CUDA Reader; never read a whole expert/PLE bank implicitly."""
        offset, length = self.row_span(start, count)
        if type(max_bytes) is not int or max_bytes <= 0 or length > max_bytes:
            raise ValueError("packed GGUF row read exceeds its staging budget")
        return reader.read(self.path, offset, length, device).reshape(count, self.row_bytes)


def _shard(path):
    # Reject a FIFO/device by its opened descriptor without first blocking on it.
    with open(path, "rb", opener=lambda name, flags: os.open(name, flags | getattr(os, "O_NONBLOCK", 0))) as stream:
        header = _Header(stream)
        if header.read(4) != b"GGUF" or header.number("I") != 3:
            raise ValueError("Flash Next requires little-endian GGUF version 3")
        tensors, fields = header.number("Q"), header.number("Q")
        if tensors > 100_000 or fields > 100_000:
            raise ValueError("GGUF directory count exceeds its bound")
        metadata = {}
        for _ in range(fields):
            key = header.string()
            if key in metadata:
                raise ValueError(f"duplicate GGUF metadata key {key}")
            metadata[key] = header.value(header.number("I"))
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
            raise ValueError("GGUF alignment must be a power of two through 4096")
        entries = []
        for _ in range(tensors):
            name, dims = header.string(), header.number("I")
            if not name or not 1 <= dims <= 4:
                raise ValueError("GGUF tensor must have a name and 1 to 4 dimensions")
            shape = tuple(reversed([header.number("Q") for _ in range(dims)]))
            kind, offset = header.number("I"), header.number("Q")
            if kind not in FORMATS:
                raise ValueError(f"{name}: unsupported ggml tensor type {kind}")
            _, block, size = FORMATS[kind]
            if any(d <= 0 for d in shape) or shape[-1] % block or offset % alignment:
                raise ValueError(f"{name}: invalid shape, quantization block, or aligned offset")
            entries.append((name, shape, kind, offset, math.prod(shape) // block * size))
        data = (stream.tell() + alignment - 1) // alignment * alignment
        result, end = [], data
        for name, shape, kind, offset, length in sorted(entries, key=lambda t: t[3]):
            begin = data + offset
            if begin < end or begin > header.size or length > header.size - begin:
                raise ValueError(f"{name}: overlapping or truncated GGUF tensor payload")
            result.append(PackedTensor(name, shape, kind, path, begin, length))
            end = begin + length
        return metadata, result


@dataclass(frozen=True)
class GGUFCheckpoint:
    metadata: object
    tensors: object
    paths: tuple[Path, ...]

    @classmethod
    def open(cls, entrypoint):
        path = Path(entrypoint).resolve(strict=True)
        match = _SPLIT.fullmatch(path.name)
        if match:
            stem, number, count = match.groups()
            count = int(count)
            if not 1 <= int(number) <= count <= 1024:
                raise ValueError("invalid GGUF split filename")
            paths = tuple(path.with_name(f"{stem}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1))
        else:
            paths = (path,)
        metadata, tensors = {}, {}
        for index, shard in enumerate(paths):
            fields, entries = _shard(shard)
            if len(paths) > 1 and (fields.get("split.no") != index or fields.get("split.count") != len(paths)):
                raise ValueError("GGUF split metadata does not match the complete ordered shard set")
            for key, value in fields.items():
                if key.startswith("split."):
                    if key == "split.tensors.count" and key in metadata and metadata[key] != value:
                        raise ValueError("GGUF shards disagree about total tensor count")
                    if key != "split.tensors.count":
                        continue
                if key in metadata and metadata[key] != value:
                    raise ValueError(f"GGUF shards disagree about {key}")
                metadata[key] = value
            for tensor in entries:
                if tensor.name in tensors:
                    raise ValueError(f"duplicate GGUF tensor {tensor.name}")
                tensors[tensor.name] = tensor
        if metadata.get("general.architecture") != "qwen4exp":
            raise ValueError("this GGUF boundary accepts the qwen4exp Flash Next architecture only")
        if metadata.get("split.tensors.count", len(tensors)) != len(tensors) or not tensors:
            raise ValueError("GGUF checkpoint tensor count is incomplete")
        return cls(MappingProxyType(metadata), MappingProxyType(tensors), paths)


def resolve_gguf(path):
    """Resolve a local GGUF file or an unambiguous split checkpoint directory.

    Return None for ordinary checkpoint directories, so existing family discovery
    and safetensors routes keep their established behavior.
    """
    path = Path(path)
    if path.is_file():
        return path.resolve() if path.suffix.lower() == ".gguf" else None
    if not path.is_dir():
        return None
    config = path / "config.json"
    if config.is_file():
        from tensorfold.cuda.tensor_file import read_metadata_json
        raw = read_metadata_json(config)
        selected = raw.get("gguf_file") or raw.get("gguf_entrypoint")
        if selected:
            target = Path(selected)
            if not target.is_absolute():
                target = path / target
            if not target.is_file() or target.suffix.lower() != ".gguf":
                raise ValueError("GGUF sidecar entrypoint must name an existing GGUF file")
            return target.resolve()
    candidates = sorted(path.glob("*.gguf"))
    entries = [p for p in candidates if not _SPLIT.fullmatch(p.name)
               or int(_SPLIT.fullmatch(p.name).group(2)) == 1]
    if len(entries) > 1:
        raise ValueError("GGUF directory contains multiple checkpoints; name one entrypoint file")
    return entries[0].resolve() if entries else None


def metadata_config(path, *, role="target"):
    """Build role-specific family configuration from authoritative GGUF fields.

    MTP-only companions retain a logical source block number without providing
    trunk/PLE weights. Their explicit role is never a second target model.
    """
    if role not in ("target", "mtp"):
        raise ValueError("GGUF metadata role must be target or mtp")
    entry = resolve_gguf(path)
    if entry is None:
        raise ValueError("no GGUF checkpoint was selected")
    checkpoint = GGUFCheckpoint.open(entry)
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
