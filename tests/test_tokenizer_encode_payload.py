"""Selected encode storage/lifetime boundaries, without a model/GPU claim."""

from dataclasses import FrozenInstanceError, replace
import sys

import pytest

from tensorfold.families.qwen4_exp.cuda import tokenizer_encode_payload as payload


def bound(**changes):
    result = payload.EncodeBound("1" * 64, "artifact", 257, 256, 1, 11, 48)
    return replace(result, **changes)


@pytest.fixture
def tokenizer():
    tk = pytest.importorskip("tokenizers")
    from tensorfold.families.qwen4_exp.gguf_assets import _QWEN35_PATTERN

    native = tk.Tokenizer(tk.models.BPE(
        {char: index for index, char in enumerate(sorted(tk.pre_tokenizers.ByteLevel.alphabet()))}, []))
    native.normalizer = tk.normalizers.NFC()
    native.pre_tokenizer = tk.pre_tokenizers.Sequence([
        tk.pre_tokenizers.Split(tk.Regex(_QWEN35_PATTERN), behavior="isolated"),
        tk.pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)])
    native.post_processor = tk.processors.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
    native.decoder = tk.decoders.ByteLevel()
    native.add_tokens([tk.AddedToken("<|literal|>", normalized=False)])
    return native


def test_lifetime_domains_are_separate_and_do_not_multiply_active_work(monkeypatch):
    monkeypatch.setattr(payload, "bytelevel_payload_policy", lambda: pytest.fail("pure math reread runtime files"))
    one = payload.encode_payload(bound(), utf8_bytes=4096, owners=1, cache_generations=2)
    three = payload.encode_payload(bound(), utf8_bytes=4096, owners=3, cache_generations=2)
    assert three["phase_heap_requested_bytes"] == one["phase_heap_requested_bytes"]
    encoding = one["returned_native_encoding_per_owner_bytes"]
    assert three["other_returned_encodings_bytes"] == 2 * encoding
    assert three["native_heap_requested_bytes"] - one["native_heap_requested_bytes"] == 2 * encoding
    assert three["ids_python_output_bytes"] == 3 * one["ids_python_output_bytes"]
    assert three["python_encoding_wrappers_bytes"] == 3 * one["python_encoding_wrappers_bytes"]


def test_startup_cache_generation_does_not_fabricate_another_maximum_encoding():
    a = payload.encode_payload(bound(), utf8_bytes=1000, owners=1, cache_generations=1)
    b = payload.encode_payload(bound(), utf8_bytes=1000, owners=1, cache_generations=2)
    assert a["other_returned_encodings_bytes"] == b["other_returned_encodings_bytes"] == 0
    assert a["phase_heap_requested_bytes"] == b["phase_heap_requested_bytes"]
    assert b["persistent_cache_bytes"] == 2 * a["persistent_cache_bytes"]
    assert b["native_heap_requested_bytes"] - a["native_heap_requested_bytes"] == a["cache_per_generation_bytes"]
    with pytest.raises(TypeError):
        payload.encode_payload(bound(), utf8_bytes=1000, owners=1)


def test_empty_text_keeps_the_native_postprocessor_metadata():
    result = payload.encode_payload(bound(), utf8_bytes=0, owners=1, cache_generations=1)
    assert result["bounds"]["tokens"] == result["ids_getter_native_copy_bytes"] == 0
    # ByteLevel still adds sequence0 to the empty Encoding; its first map
    # needs4 buckets of (usize,Range<usize>) plus the16 control bytes.
    assert result["returned_native_encoding_per_owner_bytes"] == 256 + 4 * 25 + 16
    assert result["ids_python_output_bytes"] == 56
    assert result["known_stack_payload_bytes"]["regex_initial_alloca"] == 0


@pytest.mark.parametrize("name,value", [("utf8_bytes", -1), ("utf8_bytes", True), ("utf8_bytes", 1.0),
                                      ("utf8_bytes", sys.maxsize), ("utf8_bytes", 1 << 31),
                                      ("owners", 0), ("owners", True), ("owners", sys.maxsize),
                                      ("cache_generations", 0), ("cache_generations", False),
                                      ("cache_generations", sys.maxsize)])
def test_invalid_and_overflowing_envelopes_fail_without_runtime_io(monkeypatch, name, value):
    monkeypatch.setattr(payload, "bytelevel_payload_policy", lambda: pytest.fail("invalid math touched runtime"))
    fields = dict(utf8_bytes=128, owners=1, cache_generations=1)
    fields[name] = value
    with pytest.raises(ValueError):
        payload.encode_payload(bound(), **fields)


def test_metadata_is_immutable_and_incompatible_data_is_refused():
    captured = bound()
    with pytest.raises(FrozenInstanceError):
        captured.added_tokens = 2
    with pytest.raises(ValueError, match="captured immutable"):
        payload.encode_payload(object(), utf8_bytes=1, owners=1, cache_generations=1)
    with pytest.raises(ValueError, match="minimum added"):
        payload.encode_payload(bound(minimum_added_utf8=0), utf8_bytes=1, owners=1, cache_generations=1)
    with pytest.raises(ValueError, match="int32"):
        payload.encode_payload(bound(maximum_token_id=1 << 32), utf8_bytes=1, owners=1, cache_generations=1)


@pytest.mark.parametrize("count", [0, 1, 2, 7, 8, 15, 16, 31, 32, 129, 4096, 100_000])
def test_source_byte_and_scalar_roles_have_no_model_context_clamp(count):
    result = payload.encode_payload(bound(), utf8_bytes=count, owners=1, cache_generations=1)
    limits = result["bounds"]
    assert limits["nfc_utf8_bytes"] == 3 * count
    assert 2 * limits["nfc_scalars"] <= 3 * count < 2 * (limits["nfc_scalars"] + 1)
    assert limits["splits"] == count
    assert limits["final_slice_original_bytes"] == 2 * count
    assert limits["all_original_role_peak_bytes"] == 5 * count
    assert result["active_native_heap_requested_bytes"] == max(result["phase_heap_requested_bytes"].values())
    assert result["native_heap_requested_bytes"] >= result["active_native_heap_requested_bytes"]
    if count == 100_000:
        assert limits["tokens"] > 262_144


def test_final_normalized_proof_does_not_shrink_intermediate_nfd_or_old_storage():
    full = payload.encode_payload(bound(), utf8_bytes=1000, owners=1, cache_generations=1)
    refined = payload.encode_payload(bound(), utf8_bytes=1000, owners=1, cache_generations=1,
                                     normalized_utf8_bytes=900, normalized_scalar_bound=300)
    assert refined["bounds"]["nfc_utf8_bytes"] == 900
    assert refined["bounds"]["nfc_scalars"] == refined["bounds"]["splits"] == 300
    assert refined["bounds"]["nfd_scalars"] == full["bounds"]["nfd_scalars"] == 1500
    assert refined["bounds"]["nfc_old_new_extent"] == 1000
    assert refined["native_layout_rows"]["nfc_iterator_heap_bytes"] == full[
        "native_layout_rows"]["nfc_iterator_heap_bytes"]
    assert refined["budget"]["externally_proved_normalized_utf8_bytes"] == 900
    assert refined["budget"]["externally_proved_normalized_scalar_bound"] == 300
    assert refined["native_heap_requested_bytes"] <= full["native_heap_requested_bytes"]


@pytest.mark.parametrize("kwargs", [{"normalized_utf8_bytes": True}, {"normalized_utf8_bytes": -1},
                                  {"normalized_utf8_bytes": 301}, {"normalized_scalar_bound": False},
                                  {"normalized_scalar_bound": 151}, {"normalized_scalar_bound": 1.0}])
def test_normalized_refinements_require_valid_explicit_source_bounds(kwargs):
    with pytest.raises(ValueError):
        payload.encode_payload(bound(), utf8_bytes=100, owners=1, cache_generations=1, **kwargs)


@pytest.mark.parametrize("count", [0, 1, 3, 4, 7, 8, 14, 15, 16, 28, 29, 64, 129, 1000])
def test_character_offset_table_uses_byte_entries_and_source_load_factor(count):
    # Independent capacity construction: the table leaves one empty bucket
    # below8, then1/8 of all buckets empty. Each slot has two usize values.
    buckets = 0
    if count:
        buckets = 4
        while (buckets - 1 if buckets <= 8 else buckets // 8 * 7) < count:
            buckets *= 2
    expected = 17 * buckets + 16 if buckets else 0
    result = payload.encode_payload(bound(), utf8_bytes=count, owners=1, cache_generations=1)
    assert result["native_layout_rows"]["character_offset_map_bytes"] == expected


@pytest.mark.parametrize("tokens", [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65])
def test_encoding_columns_cover_independent_single_column_growth(tokens):
    widths = (4, 4, 24, 8, 16, 4, 4)
    capacities = [0] * len(widths)
    peak = 0
    for length in range(1, tokens + 1):
        for column, width in enumerate(widths):
            if capacities[column] < length:
                previous = capacities[column]
                capacities[column] = max(4, previous * 2, length)
                live = sum(c * w for c, w in zip(capacities, widths))
                peak = max(peak, live + previous * width)
    result = payload.encode_payload(bound(), utf8_bytes=tokens, owners=1, cache_generations=1)
    assert result["phase_heap_requested_bytes"]["encoding_columns"] >= peak
    assert result["native_layout_rows"]["encoding_columns_bytes"] >= sum(
        c * w for c, w in zip(capacities, widths))


def test_capture_binds_the_current_private_pipeline_and_shared_artifact_policy(tokenizer, monkeypatch):
    from tensorfold.token_metadata import tokenizer_identity

    called = []
    monkeypatch.setattr(payload, "bytelevel_payload_policy", lambda: (called.append(True) or {"identity": "shared"}))
    captured = payload.capture_encode_bound(tokenizer)
    assert called == [True]
    assert captured.tokenizer_identity == tokenizer_identity(tokenizer)
    assert captured.artifact_identity == "shared" and captured.minimum_added_utf8 == 11
    assert captured.maximum_token_id == 256
    assert type(captured.python_encoding_wrapper_bytes) is int


def test_actual_selected_artifact_is_consumed_and_failures_are_preserved(tokenizer, monkeypatch):
    captured = payload.capture_encode_bound(tokenizer)
    assert captured.artifact_identity and captured.tokenizer_identity
    failure = ValueError("changed native artifact")

    def refuse():
        raise failure

    monkeypatch.setattr(payload, "bytelevel_payload_policy", refuse)
    with pytest.raises(ValueError) as caught:
        payload.capture_encode_bound(tokenizer)
    assert caught.value is failure


def test_duplicate_native_forward_vocab_ids_refuse_only_the_resource_capability(tokenizer, monkeypatch):
    import tokenizers as tk

    monkeypatch.setattr(payload, "bytelevel_payload_policy", lambda: {"identity": "equations"})
    duplicate = {"a": 0, "unrelated" * 128: 0}
    tokenizer.model = tk.models.BPE(duplicate, [])
    assert tokenizer.get_vocab(with_added_tokens=False) == duplicate
    before = tokenizer.to_str()
    ordinary_ids = tokenizer.encode("a").ids
    assert ordinary_ids == [0]  # The library permits this model.
    with pytest.raises(ValueError, match="injective BPE model vocabulary IDs"):
        payload.capture_encode_bound(tokenizer)
    assert tokenizer.to_str() == before
    assert tokenizer.encode("a").ids == ordinary_ids


def test_normalized_refinement_covers_protected_and_remaining_pieces_jointly(tokenizer, monkeypatch):
    import tokenizers as tk

    monkeypatch.setattr(payload, "bytelevel_payload_policy", lambda: {"identity": "equations"})
    tokenizer.add_tokens([tk.AddedToken("A", normalized=False)])
    text = "A\u030a"
    whole = tokenizer.normalizer.normalize_str(text)
    assert whole == "\u00c5" and len(whole.encode()) == 2
    encoded = tokenizer.encode(text)
    assert len(encoded.ids) == 3  # Protected A plus two ByteLevel ring bytes.
    assert len({word for word in encoded.word_ids if word is not None}) == 2
    # Whole-string NFC(U=2,K=1) would not be a valid caller proof here. The
    # raw added token stays literal; only its remaining piece is normalized.
    protected, remaining = "A", tokenizer.normalizer.normalize_str("\u030a")
    effective_bytes = len(protected.encode()) + len(remaining.encode())
    effective_scalars = len(protected) + len(remaining)
    result = payload.encode_payload(payload.capture_encode_bound(tokenizer), utf8_bytes=len(text.encode()),
        owners=1, cache_generations=1, normalized_utf8_bytes=effective_bytes,
        normalized_scalar_bound=effective_scalars)
    assert result["bounds"]["nfc_utf8_bytes"] == 3
    assert result["bounds"]["nfc_scalars"] == result["bounds"]["splits"] == 2
    assert len(encoded.ids) <= result["bounds"]["tokens"]


@pytest.mark.parametrize("change", ["normalizer", "regex", "prefix", "postprocessor", "padding",
                                    "truncation", "dropout", "normalized_added", "stripping_added", "single_word"])
def test_changed_pipeline_cannot_reuse_the_fixed_proof(tokenizer, monkeypatch, change):
    import tokenizers as tk

    monkeypatch.setattr(payload, "bytelevel_payload_policy", lambda: {"identity": "equations"})
    if change == "normalizer":
        tokenizer.normalizer = tk.normalizers.NFD()
    elif change == "regex":
        tokenizer.pre_tokenizer = tk.pre_tokenizers.Split(tk.Regex("."), behavior="isolated")
    elif change == "prefix":
        from tensorfold.families.qwen4_exp.gguf_assets import _QWEN35_PATTERN

        tokenizer.pre_tokenizer = tk.pre_tokenizers.Sequence([
            tk.pre_tokenizers.Split(tk.Regex(_QWEN35_PATTERN), behavior="isolated"),
            tk.pre_tokenizers.ByteLevel(add_prefix_space=True, trim_offsets=False, use_regex=False)])
    elif change == "postprocessor":
        tokenizer.post_processor = None
    elif change == "padding":
        tokenizer.enable_padding(length=8)
    elif change == "truncation":
        tokenizer.enable_truncation(8)
    elif change == "dropout":
        tokenizer.model.dropout = 0.1
    elif change == "normalized_added":
        tokenizer.add_tokens([tk.AddedToken("<|different|>", normalized=True)])
    elif change == "stripping_added":
        tokenizer.add_tokens([tk.AddedToken("<|different|>", normalized=False, lstrip=True)])
    else:
        tokenizer.add_tokens([tk.AddedToken("<|different|>", normalized=False, single_word=True)])
    with pytest.raises(ValueError):
        payload.capture_encode_bound(tokenizer)


@pytest.mark.parametrize("text", ["", "0123456789" * 20, "a<|literal|>b", "A" + "\u0316" * 100 + "\u030a",
                                  "\u0390" + "\u0323" * 100, "\u0344" * 100,
                                  "\U0001d162" * 100, "\u0f73" * 19 + "\u0f75",
                                  ("1" + "\u0315" * 5 + "\u0300") * 20])
def test_actual_normalization_split_and_encoding_stay_inside_the_bound(tokenizer, monkeypatch, text):
    from tokenizers import NormalizedString, Regex
    from tensorfold.families.qwen4_exp.gguf_assets import _QWEN35_PATTERN

    monkeypatch.setattr(payload, "bytelevel_payload_policy", lambda: {"identity": "equations"})
    captured = payload.capture_encode_bound(tokenizer)
    result = payload.encode_payload(captured, utf8_bytes=len(text.encode()), owners=1, cache_generations=1)
    normalized = NormalizedString(text)
    tokenizer.normalizer.normalize(normalized)
    # PreTokenizedString.split removes empty NormalizedString results.
    parts = [part for part in normalized.split(Regex(_QWEN35_PATTERN), "isolated") if part.normalized]
    assert sum(len(part.original.encode()) for part in parts) <= result["bounds"]["final_slice_original_bytes"]
    assert len(parts) <= result["bounds"]["splits"]
    assert len(normalized.normalized.encode()) <= result["bounds"]["nfc_utf8_bytes"]
    encoded = tokenizer.encode(text)
    assert len(encoded.ids) <= result["bounds"]["tokens"]
    native_columns = sum(len(getattr(encoded, name)) * width for name, width in
        (("ids", 4), ("type_ids", 4), ("tokens", 24), ("word_ids", 8), ("offsets", 16),
         ("special_tokens_mask", 4), ("attention_mask", 4)))
    native_strings = sum(len(token.encode()) for token in encoded.tokens)
    assert native_columns + native_strings <= result["returned_native_encoding_per_owner_bytes"]
