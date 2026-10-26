"""GGUF metadata/tokenizer preparation reuses the ordinary serving assets."""

import json
import os

import pytest

from tensorfold.families.qwen4_exp import gguf_assets
from tensorfold.families.qwen4_exp.kv_formats import get, get_pair
from tensorfold.token_metadata import decode_presentation, load_tokenizer, tokenizer_identity, tokenizer_vocab_size


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
                                    {"tokenizer.ggml.add_bos_token": 0},
                                    {"tokenizer.ggml.add_eos_token": None},
                                    {"tokenizer.ggml.normalizer.nfc": None},
                                    {"tokenizer.ggml.token_type": (1,)},
                                    {"tokenizer.ggml.token_type": (1, 1, 1, 1, True)},
                                    {"tokenizer.ggml.token_type": (1, 1, 1, 1, 7)},
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
    loaded = load_tokenizer(prepared)
    assert tokenizer_vocab_size(loaded) == 5
    assert decode_presentation(loaded, [2, 4, 3]) == "hi!"
    assert loaded.decode([4], skip_special_tokens=False) == "<|end|>"


def role_metadata():
    return {**metadata(), "tokenizer.ggml.tokens": (*metadata()["tokenizer.ggml.tokens"], "<think>", "[PAD6]"),
            "tokenizer.ggml.token_type": (1, 1, 1, 1, 3, 4, 5)}


def test_control_user_defined_and_unused_have_independent_recognition_and_visibility():
    tok = gguf_assets.tokenizer_from_metadata(role_metadata())
    assert tok.encode("hi<|end|><think>", add_special_tokens=False).ids == [2, 4, 5]
    assert tok.token_to_id("[PAD6]") is None
    assert 6 not in tok.encode("[PAD6]", add_special_tokens=False).ids
    ids = [2, 4, 5, 6, 3]
    assert tok.decode(ids, skip_special_tokens=False) == "hi<|end|><think>!"
    assert tok.decode(ids, skip_special_tokens=True) == "hi<think>!"
    assert decode_presentation(tok, ids) == "hi<think>!"
    assert tok.decode([6], skip_special_tokens=False) == tok.decode([6], skip_special_tokens=True) == ""
    assert tok.get_vocab_size() == 6 and tokenizer_vocab_size(tok) == 7
    assert ids == [2, 4, 5, 6, 3]  # presentation does not rewrite retained state/accounting


def test_prepared_role_metadata_survives_native_serialization(tmp_path):
    tok = gguf_assets.tokenizer_from_metadata(role_metadata())
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tensorfold_token_roles": tok.prepared_token_roles}))
    loaded = load_tokenizer(tmp_path)
    assert tokenizer_identity(loaded) == tokenizer_identity(tok)
    assert loaded.unused_token_ids == frozenset({6})
    assert loaded.control_token_ids == frozenset({4})
    assert tokenizer_vocab_size(loaded) == 7
    assert loaded.decode([6], skip_special_tokens=False) == ""


@pytest.mark.parametrize("change", [{"version": True}, {"vocab_size": True}, {"unused": [5]}, {"unused": [6, 6]},
                                    {"control": []}, {"control": [False]}, {"other": []}, {"unused": "6"}])
def test_prepared_roles_cannot_redefine_native_token_meaning(tmp_path, change):
    tok = gguf_assets.tokenizer_from_metadata(role_metadata())
    tok.save(str(tmp_path / "tokenizer.json"))
    roles = {**tok.prepared_token_roles, **change}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tensorfold_token_roles": roles}))
    with pytest.raises(ValueError):
        load_tokenizer(tmp_path)


def test_streams_stop_matching_and_probabilities_apply_prepared_presentation_only():
    from tensorfold.cuda.reply_text import StopStrings, StreamDecoder, stop_prefix_limit
    from tensorfold.server.probabilities import TokenBytes

    tok = gguf_assets.tokenizer_from_metadata(role_metadata())
    ids = [2, 4, 5, 6, 3]
    stream = StreamDecoder(tok)
    for token in ids:
        stream.add([token])
    assert stream.final() == decode_presentation(tok, ids) == "hi<think>!"
    assert not StopStrings(("<|end|>",), tok).hit(ids)
    assert StopStrings(("<think>",), tok).hit(ids)
    decoder = TokenBytes(tok)
    assert decoder.token(4) == {"token": "", "bytes": None}
    assert decoder.token(5) == {"token": "<think>", "bytes": list(b"<think>")}
    assert decoder.token(6) == {"token": "", "bytes": []}
    assert tok.decode([4], skip_special_tokens=False) == "<|end|>"
    # Invisible model rows still consume the canonical generated-token budget,
    # but cannot evict visible text from a stop string's bounded lookbehind.
    stops = StopStrings(("hi!",), tok)
    hidden = [6, 4] * 100
    assert stops.hit([2, *hidden, 3])
    assert stop_prefix_limit(stops, [2, *hidden], [], [*hidden, 3, 2]) == (len(hidden) + 1, True)


def test_qwen35_pipeline_requires_bound_or_explicit_normalization():
    from tokenizers import pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokens = tuple(alphabet + ["hi"])
    qwen35 = {**metadata(), "tokenizer.ggml.pre": "qwen35", "tokenizer.ggml.tokens": tokens,
              "tokenizer.ggml.token_type": (1,) * len(tokens)}
    with pytest.raises(ValueError, match="explicit NFC policy"):
        gguf_assets.tokenizer_from_metadata(qwen35)
    tok = gguf_assets.tokenizer_from_metadata({**qwen35, "tokenizer.ggml.normalizer.nfc": True})
    assert tok.encode("cafe\u0301 한 कि", add_special_tokens=False).ids == tok.encode("café 한 कि").ids
    assert tok.decode(tok.encode("cafe\u0301 한 कि").ids) == "café 한 कि"
    pipeline = json.loads(tok.to_str())
    assert pipeline["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] == (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
        r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    assert pipeline["pre_tokenizer"]["pretokenizers"][1]["trim_offsets"] is False
    assert pipeline["post_processor"]["trim_offsets"] is False


def test_unknown_token_inventory_never_invents_generation_eos_from_ple_or_padding():
    from tensorfold.families.qwen4_exp.frontend_metadata import generation_defaults

    unknown = {**metadata(), "tokenizer.ggml.pre": "qwen35", "tokenizer.ggml.normalizer.nfc": True,
               "tokenizer.ggml.padding_token_id": 0, "qwen4exp.ple.eos_token_id": 1}
    assert generation_defaults(unknown) == {}


def actual_gguf_frontend(path):
    import numpy as np
    from gguf_fixture import tiny_model, write
    from tensorfold.families.qwen4_exp.gguf import GGUFCheckpoint

    tiny_model(path)
    checkpoint = GGUFCheckpoint.open(path)
    payload = path.read_bytes()
    tensors = [(tensor.name, tensor.shape, tensor.ggml_type,
                np.frombuffer(payload[tensor.offset:tensor.offset + tensor.nbytes], dtype=np.uint8))
               for tensor in checkpoint.tensors.values()]
    tokens = ["h", "i", "hi", "!"] + [f"<token{index}>" for index in range(4, 32)]
    fields = {**checkpoint.metadata, "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2",
              "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.merges": ["h i"],
              "tokenizer.ggml.token_type": [3 if index in (13, 14) else 1 for index in range(32)],
              "tokenizer.chat_template": "old {{ messages[0]['content'] }}"}
    write(path, tensors, fields)
    return path


def test_same_size_source_change_with_restored_mtime_invalidates_prepared_assets(tmp_path, monkeypatch):
    path = actual_gguf_frontend(tmp_path / "model.gguf")
    monkeypatch.setenv("TF_GGUF_ASSETS_DIR", str(tmp_path / "cache"))
    original = path.stat()
    first = gguf_assets.prepare(path)
    old_template = json.loads((first / "tokenizer_config.json").read_text())["chat_template"]
    raw = path.read_bytes()
    assert raw.count(b"old {{") == 1
    path.write_bytes(raw.replace(b"old {{", b"new {{"))
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    changed = path.stat()
    assert changed.st_size == original.st_size and changed.st_mtime_ns == original.st_mtime_ns
    second = gguf_assets.prepare(path)
    assert second != first
    assert old_template == "old {{ messages[0]['content'] }}"
    assert json.loads((second / "tokenizer_config.json").read_text())["chat_template"] == "new {{ messages[0]['content'] }}"


def test_source_change_during_tokenizer_preparation_is_not_published(tmp_path, monkeypatch):
    path = actual_gguf_frontend(tmp_path / "model.gguf")
    cache = tmp_path / "cache"
    monkeypatch.setenv("TF_GGUF_ASSETS_DIR", str(cache))
    construct = gguf_assets.tokenizer_from_metadata

    def mutate(metadata):
        result = construct(metadata)
        original = path.stat()
        path.write_bytes(path.read_bytes().replace(b"old {{", b"new {{"))
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        return result

    monkeypatch.setattr(gguf_assets, "tokenizer_from_metadata", mutate)
    with pytest.raises(ValueError, match="source changed"):
        gguf_assets.prepare(path)
    assert not list(cache.glob("*/ready.json"))
    assert not list(cache.glob(".prepare-*"))


def test_missing_generation_eos_is_not_inferred_from_ple_metadata(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.gguf import GGUFCheckpoint, metadata_config
    from tensorfold.token_metadata import resolve_eos

    path = actual_gguf_frontend(tmp_path / "model.gguf")
    checkpoint = GGUFCheckpoint.open(path)
    fields = dict(checkpoint.metadata)
    fields.pop("tokenizer.ggml.eos_token_id")
    monkeypatch.setattr(GGUFCheckpoint, "open", lambda _: type("Inventory", (), {
        "metadata": fields, "tensors": checkpoint.tensors, "paths": checkpoint.paths})())
    config = metadata_config(path)
    assert config["eos_token_id"] is None
    assert config["text_config"]["gguf_ple_eos_token_id"] == 13
    with pytest.raises(ValueError, match="eos_token_id"):
        resolve_eos(tmp_path, config["eos_token_id"], vocab_size=32)
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 14}))
    assert resolve_eos(tmp_path, config["eos_token_id"], vocab_size=32) == (14,)


def test_required_ple_segmentation_is_not_inferred_from_generation_eos(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.gguf import GGUFCheckpoint, metadata_config

    path = actual_gguf_frontend(tmp_path / "model.gguf")
    checkpoint = GGUFCheckpoint.open(path)
    fields = dict(checkpoint.metadata)
    fields.pop("qwen4exp.ple.eos_token_id")
    fields["qwen4exp.ple.layers"] = (1,)
    monkeypatch.setattr(GGUFCheckpoint, "open", lambda _: type("Inventory", (), {
        "metadata": fields, "tensors": checkpoint.tensors, "paths": checkpoint.paths})())
    with pytest.raises(ValueError, match="explicit segmentation ID"):
        metadata_config(path)
    fields["qwen4exp.ple.layers"] = ()
    assert metadata_config(path)["text_config"]["gguf_ple_eos_token_id"] is None


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_created_temporary_directory_is_owned_before_path_conversion(tmp_path, monkeypatch, cleanup_fails):
    path = actual_gguf_frontend(tmp_path / "model.gguf")
    cache = tmp_path / "cache"
    monkeypatch.setenv("TF_GGUF_ASSETS_DIR", str(cache))
    Path = gguf_assets.Path
    primary, cleanup_error = MemoryError("path conversion"), OSError("temporary retirement")

    def convert(value):
        if type(value) is str and ".prepare-" in value:
            raise primary
        return Path(value)

    convert.home = Path.home
    monkeypatch.setattr(gguf_assets, "Path", convert)
    if cleanup_fails:
        def fail_cleanup(name):
            raise cleanup_error
        monkeypatch.setattr(gguf_assets.shutil, "rmtree", fail_cleanup)
    with pytest.raises(MemoryError) as caught:
        gguf_assets.prepare(path)
    assert caught.value is primary
    pending = list(cache.glob(".prepare-*"))
    if cleanup_fails:
        assert len(pending) == 1
        assert primary.__cause__ is cleanup_error
        assert primary._tensorfold_retained_owners == [str(pending[0])]
    else:
        assert not pending


def test_actual_app_token_strings_preserve_raw_lexemes_and_empty_nonlexical_rows(tmp_path):
    from types import SimpleNamespace
    from tokenizers import processors
    from tensorfold.cuda.server import App

    data = role_metadata()
    data["tokenizer.ggml.tokens"] = (*data["tokenizer.ggml.tokens"], "Ġ", "Ġhi")
    data["tokenizer.ggml.token_type"] = (*data["tokenizer.ggml.token_type"], 1, 1)
    data["tokenizer.ggml.merges"] = ("h i", "Ġ hi")
    tokenizer = gguf_assets.tokenizer_from_metadata(data)
    # Supported tokenizer text handling can insert an internal row by ID. It
    # does not register the UNUSED label as a recognized lexical input token.
    tokenizer.post_processor = processors.TemplateProcessing(single="$A <internal-unused>",
                                                              special_tokens=[("<internal-unused>", 6)])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tensorfold_token_roles": tokenizer.prepared_token_roles}))
    tokenizer = load_tokenizer(tmp_path)
    app = App.__new__(App)
    app.tok, app.engine = tokenizer, SimpleNamespace()
    app.native_context_window = app.context_window = 0
    reply = app.tokenize({"prompt": " hi<|end|><think>!", "return_token_strs": True})
    assert reply["tokens"] == [8, 4, 5, 3, 6] and reply["count"] == 5
    assert reply["token_strs"] == ["Ġhi", "<|end|>", "<think>", "!", ""]
    assert all(type(piece) is str for piece in reply["token_strs"])
    assert tokenizer.token_to_id("[PAD6]") is tokenizer.token_to_id("<internal-unused>") is None
    assert tokenizer.decode(reply["tokens"], skip_special_tokens=False) == " hi<|end|><think>!"


def test_raw_half_cache_has_distinct_payload_identity_and_peer_handshake():
    half, bf16 = get("fp16"), get("bf16")
    assert half.payload_dtype == "float16" and bf16.payload_dtype == "bfloat16"
    assert half.storage_uid != bf16.storage_uid and half.handshake != bf16.handshake
    assert half.value_bytes(256) == bf16.value_bytes(256)
    assert get_pair("fp16").identity != get_pair("bf16").identity
