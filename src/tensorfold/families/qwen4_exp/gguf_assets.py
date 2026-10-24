"""Prepare ordinary TensorFold metadata/tokenizer files for immutable GGUF inputs.

The cache contains metadata only. Weight payloads stay in the original files,
and the same Tokenizer/ChatTemplate/server paths consume the prepared assets.
"""

from __future__ import annotations

import hashlib
import errno
import json
import os
from pathlib import Path
import tempfile

from .gguf import GGUFCheckpoint, metadata_config, resolve_gguf

# Qwen byte-level BPE segmentation; the GGUF preprocessing identity determines
# whether Unicode combining marks belong to a word and NFC is applied.
_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
            r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def tokenizer_from_metadata(metadata):
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

    pre = metadata.get("tokenizer.ggml.pre")
    if metadata.get("tokenizer.ggml.model") != "gpt2" or pre not in ("qwen2", "qwen35"):
        raise ValueError("GGUF Flash Next requires declared Qwen2/Qwen35 byte-level BPE preprocessing")
    tokens = metadata.get("tokenizer.ggml.tokens")
    merges = metadata.get("tokenizer.ggml.merges")
    if not tokens or not merges or any(not isinstance(t, str) for t in tokens):
        raise ValueError("GGUF tokenizer requires vocabulary strings and BPE merges")
    vocab = {token: i for i, token in enumerate(tokens)}
    if len(vocab) != len(tokens):
        raise ValueError("GGUF tokenizer contains duplicate vocabulary strings")
    pairs = [merge.split(" ") for merge in merges]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("GGUF BPE merge must identify exactly two pieces")
    tokenizer = Tokenizer(models.BPE(vocab, [tuple(p) for p in pairs], fuse_unk=False, byte_fallback=False))
    pattern = _PATTERN
    if pre == "qwen35":
        pattern = pattern.replace(r"\p{L}", r"[\p{L}\p{M}]")
        # In negated classes, insert the mark category without nested classes.
        pattern = pattern.replace(r"[^\r\n[\p{L}\p{M}]\p{N}]", r"[^\r\n\p{L}\p{M}\p{N}]")
        pattern = pattern.replace(r"[^\s[\p{L}\p{M}]\p{N}]", r"[^\s\p{L}\p{M}\p{N}]")
        tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    types = metadata.get("tokenizer.ggml.token_type", (1,) * len(tokens))
    if len(types) != len(tokens):
        raise ValueError("GGUF token-type inventory differs from vocabulary")
    tokenizer.add_tokens([AddedToken(t, normalized=False) for t, kind in zip(tokens, types) if kind == 4])
    tokenizer.add_special_tokens([AddedToken(t, normalized=False, special=True)
                                  for t, kind in zip(tokens, types) if kind == 3])
    if metadata.get("tokenizer.ggml.add_bos_token", False) or metadata.get("tokenizer.ggml.add_eos_token", False):
        raise ValueError("GGUF tokenizer requests an unsupported automatic BOS/EOS policy")
    if any(tokenizer.token_to_id(token) != i for i, token in enumerate(tokens)):
        raise ValueError("tokenizer reconstruction changed GGUF vocabulary IDs")
    return tokenizer


def prepare(entrypoint: str | Path) -> Path:
    """Return an ordinary metadata directory without altering checkpoint files."""
    source = resolve_gguf(entrypoint)
    if source is None:
        raise ValueError(f"{entrypoint}: no GGUF checkpoint entrypoint")
    checkpoint = GGUFCheckpoint.open(source)
    identity = hashlib.sha256(b"tensorfold-gguf-assets-v1\0")
    for path in checkpoint.paths:
        info = path.stat()
        identity.update(str(path).encode() + repr((info.st_size, info.st_mtime_ns)).encode())
    root = Path(os.environ.get("TF_GGUF_ASSETS_DIR", Path.home() / ".cache/tensorfold/gguf"))
    directory = root / identity.hexdigest()
    if (directory / "ready.json").is_file():
        return directory
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=root))
    try:
        metadata = checkpoint.metadata
        config = metadata_config(source)
        config["gguf_file"] = str(source)
        config["execution_dtype"] = "float16"
        config["quantization_config"] = {"quant_method": "gguf"}
        tokenizer = tokenizer_from_metadata(metadata)
        tokenizer.save(str(temporary / "tokenizer.json"))
        token_config = {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length":
                        int(config.get("text_config", config)["max_position_embeddings"])}
        tokens = metadata["tokenizer.ggml.tokens"]
        for field, target in (("bos", "bos_token"), ("eos", "eos_token"), ("padding", "pad_token")):
            value = metadata.get(f"tokenizer.ggml.{field}_token_id")
            if value is not None:
                if type(value) is not int or not 0 <= value < len(tokens):
                    raise ValueError("GGUF special token ID is outside vocabulary")
                token_config[target] = tokens[value]
        template = metadata.get("tokenizer.chat_template")
        if not isinstance(template, str) or not template:
            raise ValueError("GGUF checkpoint has no chat template")
        token_config["chat_template"] = template
        for filename, data in (("config.json", config), ("tokenizer_config.json", token_config),
                               ("ready.json", {"entrypoint": str(source), "identity": identity.hexdigest()})):
            (temporary / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        try:
            temporary.rename(directory)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not (directory / "ready.json").is_file():
                raise
        return directory
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
