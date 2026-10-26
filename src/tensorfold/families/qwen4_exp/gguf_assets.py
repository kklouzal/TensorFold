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
import shutil
import stat
import tempfile

from tensorfold.cleanup import rollback

from .gguf import GGUFCheckpoint, metadata_config, resolve_gguf
from .frontend_metadata import generation_defaults, semantic_source


def _versions(paths):
    versions = []
    for path in paths:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("GGUF preparation requires regular source files")
        versions.append((str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return tuple(versions)


def _unchanged(paths, versions):
    if _versions(paths) != versions:
        raise ValueError("GGUF source changed during asset preparation")


def _ready(directory, expected):
    from tensorfold.checkpoint_metadata import read_json

    if not (directory / "ready.json").is_file():
        return False
    if read_json(directory, "ready.json", required=True) != expected:
        raise ValueError("prepared GGUF cache identity conflicts with its source version")
    return True

# Qwen2 and pinned Qwen35 byte-level BPE segmentation. The latter's initial
# optional character class intentionally differs from its word/punctuation
# classes; do not mechanically replace every Unicode-letter category.
_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
            r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
_QWEN35_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
                   r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def tokenizer_from_metadata(metadata):
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
    from tensorfold.token_metadata import bind_token_roles

    pre = metadata.get("tokenizer.ggml.pre")
    if metadata.get("tokenizer.ggml.model") != "gpt2" or pre not in ("qwen2", "qwen35"):
        raise ValueError("GGUF Flash Next requires declared Qwen2/Qwen35 byte-level BPE preprocessing")
    tokens = metadata.get("tokenizer.ggml.tokens")
    merges = metadata.get("tokenizer.ggml.merges")
    if (type(tokens) not in (list, tuple) or not tokens or type(merges) not in (list, tuple) or not merges
            or any(type(t) is not str for t in tokens) or any(type(m) is not str for m in merges)):
        raise ValueError("GGUF tokenizer requires vocabulary strings and BPE merges")
    if len(set(tokens)) != len(tokens):
        raise ValueError("GGUF tokenizer contains duplicate vocabulary strings")
    types = metadata.get("tokenizer.ggml.token_type", (1,) * len(tokens))
    if (type(types) not in (list, tuple) or len(types) != len(tokens)
            or any(type(kind) is not int or kind not in range(1, 7) for kind in types)):
        raise ValueError("GGUF token-type inventory differs from vocabulary or contains unknown types")
    # UNUSED is a matrix role, never an input literal. Missing decoder IDs
    # produce zero visible bytes in the native ByteLevel decoder, including
    # when skip_special_tokens=False; lexical IDs are never renumbered.
    vocab = {token: i for i, (token, kind) in enumerate(zip(tokens, types)) if kind != 5}
    pairs = [merge.split(" ") for merge in merges]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("GGUF BPE merge must identify exactly two pieces")
    tokenizer = Tokenizer(models.BPE(vocab, [tuple(p) for p in pairs], fuse_unk=False, byte_fallback=False))
    nfc = metadata.get("tokenizer.ggml.normalizer.nfc")
    if "tokenizer.ggml.normalizer.nfc" in metadata and type(nfc) is not bool:
        raise ValueError("GGUF NFC normalizer policy must be a boolean")
    source = semantic_source(metadata)
    if pre == "qwen35":
        if source is not None:
            if nfc is False:
                raise ValueError("GGUF NFC policy conflicts with the bound target tokenizer pipeline")
            nfc = True
        elif nfc is None:
            raise ValueError("GGUF qwen35 tokenizer requires an explicit NFC policy or a validated semantic source")
    if nfc:
        tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(_QWEN35_PATTERN if pre == "qwen35" else _PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=pre != "qwen35", use_regex=False),
    ])
    if pre == "qwen35":
        tokenizer.post_processor = processors.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_tokens([AddedToken(t, normalized=False) for t, kind in zip(tokens, types) if kind == 4])
    tokenizer.add_special_tokens([AddedToken(t, normalized=False, special=True)
                                  for t, kind in zip(tokens, types) if kind == 3])
    for field in ("tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"):
        if field in metadata and type(metadata[field]) is not bool:
            raise ValueError("GGUF automatic BOS/EOS policies must be booleans")
        if metadata.get(field, False):
            raise ValueError("GGUF tokenizer requests an unsupported automatic BOS/EOS policy")
    if any(tokenizer.token_to_id(token) != (None if types[i] == 5 else i) for i, token in enumerate(tokens)):
        raise ValueError("tokenizer reconstruction changed GGUF vocabulary IDs")
    roles = {"version": 1, "vocab_size": len(tokens),
             "control": [i for i, kind in enumerate(types) if kind == 3],
             "unused": [i for i, kind in enumerate(types) if kind == 5]}
    bind_token_roles(tokenizer, roles)
    tokenizer.prepared_token_roles = roles
    tokenizer.frontend_semantic_source = source
    return tokenizer


def prepare(entrypoint: str | Path) -> Path:
    """Return an ordinary metadata directory without altering checkpoint files."""
    source = resolve_gguf(entrypoint)
    if source is None:
        raise ValueError(f"{entrypoint}: no GGUF checkpoint entrypoint")
    checkpoint = GGUFCheckpoint.open(source)
    # The discovery read establishes the complete shard set. Its metadata is
    # not used until a fresh read is bound to all of these source versions.
    paths = checkpoint.paths
    versions = _versions(paths)
    identity = hashlib.sha256(b"tensorfold-gguf-assets-v4\0" + repr(versions).encode())
    root = Path(os.environ.get("TF_GGUF_ASSETS_DIR", Path.home() / ".cache/tensorfold/gguf"))
    directory = root / identity.hexdigest()
    ready = {"entrypoint": str(source), "identity": identity.hexdigest(),
             "source_versions": [list(version) for version in versions]}
    if _ready(directory, ready):
        _unchanged(paths, versions)
        return directory
    root.mkdir(parents=True, exist_ok=True)
    temporary_name = tempfile.mkdtemp(prefix=".prepare-", dir=root)
    try:
        temporary = Path(temporary_name)
        checkpoint = GGUFCheckpoint.open(source)
        if checkpoint.paths != paths:
            raise ValueError("GGUF source shard set changed during asset preparation")
        metadata = checkpoint.metadata
        config = metadata_config(source)
        config["gguf_file"] = str(source)
        config["execution_dtype"] = "float16"
        config["quantization_config"] = {"quant_method": "gguf"}
        tokenizer = tokenizer_from_metadata(metadata)
        tokenizer.save(str(temporary / "tokenizer.json"))
        token_config = {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length":
                        int(config.get("text_config", config)["max_position_embeddings"]),
                        "clean_up_tokenization_spaces": False,
                        "add_bos_token": False, "add_eos_token": False,
                        "tensorfold_token_roles": tokenizer.prepared_token_roles}
        if tokenizer.frontend_semantic_source is not None:
            token_config["tensorfold_frontend_semantic_source"] = tokenizer.frontend_semantic_source
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
        generation = generation_defaults(metadata)
        if generation:
            # The selected target's scalar EOS identity is independent from
            # this explicitly bound serving terminator set and PLE segmentation.
            (temporary / "generation_config.json").write_text(json.dumps(generation) + "\n")
        for filename, data in (("config.json", config), ("tokenizer_config.json", token_config),
                               ("ready.json", ready)):
            (temporary / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        _unchanged(paths, versions)
        try:
            temporary.rename(directory)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not _ready(directory, ready):
                raise
        else:
            temporary_name = None  # publication transfers ownership to the immutable cache
        _unchanged(paths, versions)
    except BaseException as primary:
        if temporary_name is not None:
            rollback(temporary_name, primary, lambda: shutil.rmtree(temporary_name))
        raise
    if temporary_name is not None:  # another preparer's verified publication won
        shutil.rmtree(temporary_name)
    return directory
