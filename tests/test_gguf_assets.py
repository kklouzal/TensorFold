"""GGUF metadata/tokenizer preparation reuses the ordinary serving assets."""

import json

import pytest

from tensorfold.families.qwen4_exp import gguf_assets
from tensorfold.families.qwen4_exp.kv_formats import get, get_pair


def metadata():
    return {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2",
            "tokenizer.ggml.tokens": ("h", "i", "hi", "!", "<|end|>"),
            "tokenizer.ggml.merges": ("h i",), "tokenizer.ggml.token_type": (1, 1, 1, 1, 3),
            "tokenizer.ggml.eos_token_id": 4, "tokenizer.chat_template": "{{ messages[0]['content'] }}"}


def test_reconstructed_bpe_preserves_training_ids_and_special_markers():
    tokenizer = gguf_assets.tokenizer_from_metadata(metadata())
    assert tokenizer.encode("hi!", add_special_tokens=False).ids == [2, 3]
    assert tokenizer.encode("hi<|end|>", add_special_tokens=False).ids == [2, 4]
    assert tokenizer.decode([2, 3]) == "hi!"
    assert tokenizer.decode([2, 4], skip_special_tokens=True) == "hi"


@pytest.mark.parametrize("change", [{"tokenizer.ggml.pre": "unknown"},
                                    {"tokenizer.ggml.add_bos_token": True},
                                    {"tokenizer.ggml.token_type": (1,)},
                                    {"tokenizer.ggml.tokens": ("h", "h")}])
def test_unsupported_or_inconsistent_tokenizer_metadata_fails(change):
    with pytest.raises(ValueError):
        gguf_assets.tokenizer_from_metadata({**metadata(), **change})


def test_preparation_keeps_weights_immutable_and_reuses_completed_assets(tmp_path, monkeypatch):
    checkpoint = tmp_path / "original.gguf"
    checkpoint.write_bytes(b"original immutable payload")
    cache = tmp_path / "cache"
    monkeypatch.setenv("TF_GGUF_ASSETS_DIR", str(cache))
    monkeypatch.setattr(gguf_assets, "resolve_gguf", lambda _: checkpoint)

    class Inventory:
        paths = (checkpoint,)
        metadata = metadata()

    monkeypatch.setattr(gguf_assets.GGUFCheckpoint, "open", lambda _: Inventory())
    monkeypatch.setattr(gguf_assets, "metadata_config", lambda _: {"model_type": "qwen4_exp",
                        "text_config": {"max_position_embeddings": 4096}})
    prepared = gguf_assets.prepare(checkpoint)
    assert prepared == gguf_assets.prepare(checkpoint)
    assert prepared.parent == cache
    assert checkpoint.read_bytes() == b"original immutable payload"
    assert json.loads((prepared / "config.json").read_text())["gguf_file"] == str(checkpoint)
    assert (prepared / "tokenizer.json").is_file() and (prepared / "ready.json").is_file()
    assert not list(cache.glob(".prepare-*"))


def test_raw_half_cache_has_distinct_payload_identity_and_peer_handshake():
    half, bf16 = get("fp16"), get("bf16")
    assert half.payload_dtype == "float16" and bf16.payload_dtype == "bfloat16"
    assert half.storage_uid != bf16.storage_uid and half.handshake != bf16.handshake
    assert half.value_bytes(256) == bf16.value_bytes(256)
    assert get_pair("fp16").identity != get_pair("bf16").identity
