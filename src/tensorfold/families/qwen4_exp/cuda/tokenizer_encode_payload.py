"""Requested storage for the selected synchronous, IDs-only encode caller.

The private tokenizer must come from ``load_tokenizer`` and remain unchanged:
BPE's default cache capacity is not serialized, so the caller must prove no
custom construction/resize/clear and bound generations across native threads.
This is a storage calculation, not a software authority or an RSS grant.

NFC positive-change characters form a composition-closed Letter/Mark alphabet
in the pinned Unicode9/Onig artifacts. An original scalar's monotone alignment
block has at most one arbitrary leading output and that Letter/Mark suffix;
the fixed regex intersects it in at most two parts. Final original copies are
therefore <=2N, current-parent plus new copies <=3N. Binding/PreTokenizedString
whole-input copies add2N. ASCII anchors cannot gain the suffix, giving at most
N nonempty splits when UTF8 byte lengths pay for non-ASCII anchors' two parts.
"""

from dataclasses import dataclass
import hashlib
import json
import sys

from .tokenizer_payload import (
    _vec_capacity as _capacity,
    _vec_reallocation_capacity as _growth,
    bytelevel_payload_policy,
)


# This digest is a proof input, independent of the producer's pattern constant:
# changing that producer cannot silently authorize a new compiled regex.
_SPLIT_SHA256 = "00431aed57e696b747435f734d1e3b9b1bfd931a121fb5cac7129e97c181e9ba"


@dataclass(frozen=True, slots=True)
class EncodeBound:
    tokenizer_identity: str
    artifact_identity: str
    vocabulary_size: int
    maximum_token_id: int
    added_tokens: int
    minimum_added_utf8: int
    python_encoding_wrapper_bytes: int
    codec: str = "encode"


def _size(value, name, *, positive=False):
    if type(value) is not int or not int(positive) <= value <= sys.maxsize:
        raise ValueError(f"encode {name} must be a {'positive' if positive else 'nonnegative'} host-sized integer")
    return value


def _plain_bytelevel(value):
    return (type(value) is dict and value.keys() == {"type", "add_prefix_space", "trim_offsets", "use_regex"}
            and value["type"] == "ByteLevel"
            and all(value[key] is False for key in ("add_prefix_space", "trim_offsets", "use_regex")))


def capture_encode_bound(tokenizer, *, codec="encode", expected_tokenizer_identity=None):
    """Capture one fresh, privately owned tokenizer before concurrent use.

    Artifact verification is shared with the decoder policy. No tokenizer or
    vocabulary is retained. The aggregate owns the loader/cache-lifetime proof;
    neither a version label nor this plain metadata object replaces it.
    """
    import tokenizers
    from tensorfold.token_metadata import tokenizer_identity, tokenizer_vocab_size

    if type(codec) is not str or codec not in ("encode", "encode_batch", "encode_batch_fast"):
        raise ValueError("encode storage requires a proved public encoding API")
    if type(tokenizer) is not tokenizers.Tokenizer:
        raise ValueError("encode storage requires the exact native Tokenizer")
    if not callable(getattr(tokenizer, codec, None)):
        raise ValueError("encode storage requires its selected public API")
    artifact = bytelevel_payload_policy()
    state = json.loads(tokenizer.to_str())
    pre = state.get("pre_tokenizer")
    stages = pre.get("pretokenizers") if type(pre) is dict and pre.get("type") == "Sequence" else None
    split = stages[0] if type(stages) is list and len(stages) == 2 else None
    pattern = split.get("pattern") if type(split) is dict else None
    if (state.get("normalizer") != {"type": "NFC"} or type(pattern) is not dict
            or pattern.keys() != {"Regex"} or type(pattern["Regex"]) is not str
            or hashlib.sha256(pattern["Regex"].encode()).hexdigest() != _SPLIT_SHA256
            or split.get("type") != "Split" or split.get("behavior") != "Isolated"
            or split.get("invert") is not False or not _plain_bytelevel(stages[1])
            or not _plain_bytelevel(state.get("post_processor"))
            or state.get("padding") is not None or state.get("truncation") is not None):
        raise ValueError("encode storage has no proof for this normalization/split/postprocessing pipeline")
    model = state.get("model", {})
    if (type(model) is not dict or model.get("type") != "BPE" or any(model.get(key) is not None for key in
            ("dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix"))
            or any(model.get(key) is not False for key in ("fuse_unk", "byte_fallback", "ignore_merges"))):
        raise ValueError("encode storage requires the selected deterministic BPE settings")
    # Native BPE accepts duplicate IDs and builds its reverse strings by
    # collecting that map. Word output could then clone an unrelated string,
    # invalidating the2U token-string bound. Inspect the actual forward map:
    # serialization may choose an order/canonical form through the reverse.
    vocabulary = tokenizer.get_vocab(with_added_tokens=False)
    if len(set(vocabulary.values())) != len(vocabulary):
        raise ValueError("encode storage requires injective BPE model vocabulary IDs")
    added = state.get("added_tokens", [])
    if (type(added) is not list or any(type(row) is not dict or type(row.get("content")) is not str or not row["content"]
            or any(row.get(key) is not False for key in ("normalized", "lstrip", "rstrip", "single_word")) for row in added)):
        raise ValueError("encode storage requires literal non-normalized, non-stripping, unrestricted added tokens")
    maximum = max(max(vocabulary.values(), default=0), max((row["id"] for row in added), default=0))
    if type(maximum) is not int or not 0 <= maximum < 1 << 32:
        raise ValueError("encode token IDs exceed the native unsigned-int32 domain")
    minimum = min((len(row["content"].encode()) for row in added), default=0)
    # The public empty Encoding constructor allocates no variable columns.
    # Its shallow size includes the 256-byte native Encoding, paid below.
    shallow = sys.getsizeof(tokenizers.Encoding())
    if shallow < 256:
        raise ValueError("Encoding wrapper disagrees with its pinned native layout")
    wrapper = shallow - 256
    identity = tokenizer_identity(tokenizer)
    if expected_tokenizer_identity is not None and identity != expected_tokenizer_identity:
        raise ValueError("IDs-only callback tokenizer changed since profile preparation")
    return EncodeBound(identity, artifact["identity"], tokenizer_vocab_size(tokenizer),
                       maximum, len(added), minimum, wrapper, codec)


def _map_bytes(count, entry_bytes):
    if not count:
        return 0
    if count < 15:
        buckets = 4 if count < 4 else 8 if count < 8 else 16
    else:
        required = 8 * count // 7
        buckets = 1 << (required - 1).bit_length()
    # Rust b940 std hashbrown0.17.1, SSE2 control groups. These entry widths
    # make the control offset already 16-byte aligned for every bucket count.
    return (entry_bytes + 1) * buckets + 16


def _encode_layout(bound, *, utf8_bytes, owners, cache_generations, normalized_utf8_bytes,
                   normalized_scalar_bound, decomposed_scalar_bound, split_bound, protected_match_bound, batch):
    if type(bound) is not EncodeBound:
        raise ValueError("encode storage requires its captured immutable metadata")
    allowed = ("encode_batch", "encode_batch_fast") if batch else ("encode",)
    if bound.codec not in allowed:
        raise ValueError("encoding API requires its matching synchronous or batch resource contract")
    fast_ids = bound.codec == "encode_batch_fast"
    n = _size(utf8_bytes, "UTF8 bytes")
    h = _size(owners, "Encoding owners", positive=True)
    generations = _size(cache_generations, "cache generations", positive=True)
    _size(bound.maximum_token_id, "maximum token ID")
    if bound.maximum_token_id >= 1 << 32:
        raise ValueError("encode token IDs exceed the native unsigned-int32 domain")
    _size(bound.added_tokens, "added token count")
    minimum = _size(bound.minimum_added_utf8, "minimum added-token bytes", positive=bool(bound.added_tokens))
    wrapper = _size(bound.python_encoding_wrapper_bytes, "Encoding wrapper bytes")
    generic_utf8, generic_decomposed = _size(3 * n, "normalized UTF8 bytes"), 3 * n // 2
    decomposed = (generic_decomposed if decomposed_scalar_bound is None else
                  _size(decomposed_scalar_bound, "proved intermediate decomposed scalar count"))
    u = generic_utf8 if normalized_utf8_bytes is None else _size(normalized_utf8_bytes, "proved normalized UTF8 bytes")
    scalars = (generic_decomposed if normalized_scalar_bound is None else
               _size(normalized_scalar_bound, "proved normalized scalar count"))
    if u > generic_utf8 or scalars > generic_decomposed or decomposed > generic_decomposed:
        raise ValueError("proved normalized bounds must fit the selected generic NFC envelope")
    scalars = min(scalars, u, decomposed)  # Final scalars also fit their NFD expansion.
    u = min(u, 4 * scalars)    # Every valid Unicode scalar has at most four.
    splits = n if split_bound is None else _size(split_bound, "proved split count")
    if splits > n:
        raise ValueError("proved split count must fit the input UTF8 envelope")
    m, v = min(splits, scalars), max(n, u)
    t = u
    # Region offsets and stack_n are native int32. Check the actual rounded
    # stack-capacity envelope, not just the number of currently live cells.
    stack_capacity = 160
    while stack_capacity < scalars + 24:
        stack_capacity *= 2
    if u > (1 << 31) - 1 or stack_capacity > (1 << 31) - 1:
        raise ValueError("encode envelope exceeds the selected native regex integer domain")
    a = n // minimum if bound.added_tokens else 0
    if protected_match_bound is not None:
        protected = _size(protected_match_bound, "proved protected match count")
        if protected > a:
            raise ValueError("proved protected matches must fit the literal input envelope")
        a = protected
    a = min(a, m)
    raw_parts = min(m, 2 * a + 1) if n else 0
    c, j = _capacity, _growth
    onig_heap = 96 * max(160, scalars + 24) + 32 + 72 + 112 if u else 0
    # Raw extraction: original-role4N + old/new normalized2N + token valuesN;
    # old/new alignments, offset records24 and two Split-sized104 arrays.
    raw = (7 * n + 16 * (c(n) + n) + 104 + 24 * c(max(1, raw_parts))
           + 208 * c(max(1, raw_parts)) + 48 * a)
    # Before regex there is one original per raw part. All live normalized
    # capacities fit2V+8r and alignment capacities2V+4r, V=max(N,U).
    # NFC's iterator/sort
    # working set and subsequent splices are different lifetime maxima.
    nfc_base = (4 * n + 2 * v + 8 * raw_parts + 16 * (2 * v + 4 * raw_parts)
                + 104 * (c(raw_parts) + raw_parts) + 48 * a + 128)
    nfc_work = max(n + 4 * c(n) + c(u, 8) + 16 * c(v) + 80 * decomposed,
                   c(u, 8) + 16 * c(v) + 16 * j(v) + j(v, 8), n + 17 * u + 128)
    # Original-role5N + protected token valuesN. The82U is normalized copies2U,
    # old/new offset arrays48U, and old/new alignments32U. New normalized80
    # and Split104 arrays move their Strings; they do not clone them again.
    regex = 6 * n + 82 * u + 184 * c(m) + 104 * c(raw_parts) + 48 * a + onig_heap
    # ByteLevel: originals4N + protected valuesN; normalized capacities2U+8m,
    # alignments16*(2U+4m), then current transform/collection/splice working.
    byte_base = (5 * n + 34 * u + 72 * m + 104 * (c(m) + m) + 48 * a + 128)
    byte_work = max(16 * u + 4 * c(u) + 2 * u + 8 + 16 * (2 * u + 4),
                    4 * c(u) + 2 * u + 8 + 16 * (2 * u + 4) + 16 * (3 * u + 4),
                    4 * c(u) + 2 * u + 8 + 3 * u + 8)
    # Model-stage normalized/alignment payload34U+72m, token Strings2U and
    # Token arrays48T survive until each consumed split retires.
    resident = 4 * n + 36 * u + 72 * m + 104 * c(m) + 48 * t
    # OffsetType::None returns before constructing BytesToCharOffsetConverter.
    # It drops each old token value and puts an empty String in the Encoding.
    # Earlier normalization/model/temporary-token ownership stays paid.
    offset_map = 0 if fast_ids else _map_bytes(n, 16)
    sequence_map = _map_bytes(1, 24)
    returned_strings = 0 if fast_ids else 2 * u
    returned = 256 + 64 * c(t) + returned_strings + sequence_map

    # Fresh default BPE: each cached word reserves at most255 Symbols and
    # a255-byte key, even when merging leaves only one live Symbol. The TLS
    # object is80 bytes in the pinned DSO. Outer-map layout is <=308 bytes
    # per generation for every distribution across threads (B(g)<=4g).
    cache_table = _map_bytes(10_000, 48)
    per_cache = 10_000 * 255 * 33 + cache_table + 308 + 80
    cache_bytes = generations * per_cache
    cache_resize = max(cache_table, _map_bytes(generations, 72))
    phases = {
        "raw_added_token_slices": raw,
        "nfc_normalization_and_full_slice": nfc_base + nfc_work,
        "regex_slices_and_search": regex,
        "bytelevel_normalization": byte_base + byte_work,
        # Word starts with byte-sized capacity; its heap starts at symbol
        # count and can double only once (live <=2n-2). Word/heap/skip or
        # Word/token-output working storage is <=128 bytes per input byte.
        # Earlier tokens cost<=50*(U-u_i), so adding128*u_i to the34U
        # model base is<=162U; protected tokens are included in the earlier set.
        "bpe_words_and_prior_tokens": 4 * n + 162 * u + 72 * m + 104 * c(m) + cache_resize,
        # At most one column is reallocating; its largest old element is a
        # String header24, and old capacity is below the required T.
        "encoding_columns": resident + offset_map + 64 * c(t) + 24 * t + 256 + sequence_map,
        # Two64-byte column sets plus a conservative extra16-byte offset
        # map/collect column. Strings move; only their single2U pool survives.
        "postprocessing_move": 144 * c(t) + returned_strings + 2 * sequence_map + 256,
        "ids_getter": returned + 4 * t,
    }
    if not fast_ids:
        phases["character_offset_map"] = resident + 2 * offset_map + 48
    for name, value in phases.items():
        _size(value, name)
    active = max(phases.values())
    retained = _size((h - 1) * returned, "other returned Encodings")
    total = _size(active + retained + cache_bytes, "requested native/Encoding layouts")
    ids = _size(h * (56 + (8 + sys.getsizeof(bound.maximum_token_id)) * t), "Python ID outputs")
    return {"schema": "tensorfold.selected-tokenizer-encode-payload-v1", "codec": bound.codec,
        "tokenizer_identity": bound.tokenizer_identity, "artifact_identity": bound.artifact_identity,
        "budget": {"utf8_bytes": n, "owners": h, "cache_generations": generations,
                   "externally_proved_normalized_utf8_bytes": normalized_utf8_bytes,
                   "externally_proved_normalized_scalar_bound": normalized_scalar_bound,
                   "externally_proved_decomposed_scalar_bound": decomposed_scalar_bound,
                   "externally_proved_split_bound": split_bound,
                   "externally_proved_protected_match_bound": protected_match_bound},
        "bounds": {"nfc_utf8_bytes": u, "nfc_scalars": scalars, "nfd_scalars": decomposed,
                   "nfc_old_new_extent": v, "splits": m, "tokens": t,
                   "final_slice_original_bytes": 2 * n, "slice_construction_original_bytes": 3 * n,
                   "all_original_role_peak_bytes": 5 * n},
        "phase_heap_requested_bytes": phases, "active_native_heap_requested_bytes": active,
        "native_layout_rows": {"nfc_iterator_heap_bytes": 80 * decomposed,
            "character_offset_map_bytes": offset_map, "encoding_columns_bytes": 64 * c(t),
            "returned_token_string_bytes": returned_strings,
            "bpe_one_word_working_bytes": 128 * u, "bpe_cache_table_bytes": cache_table,
            "regex_search_heap_bytes": onig_heap, "regex_stack_capacity_bound": stack_capacity},
        "returned_native_encoding_per_owner_bytes": returned,
        "other_returned_encodings_bytes": retained, "persistent_cache_bytes": cache_bytes,
        "cache_per_generation_bytes": per_cache, "native_heap_requested_bytes": total,
        "python_encoding_wrappers_bytes": _size(h * wrapper, "Encoding Python wrappers"),
        "ids_python_output_bytes": ids, "ids_getter_native_copy_bytes": 4 * t,
        "known_stack_payload_bytes": {"regex_initial_alloca": 5136 if u else 0,
                                      "regex_match_arg": 112 if u else 0, "nfc_inline_elements": 128 if n else 0},
        "required_external_roles": ("Python input and UTF8-cache conversion/error working storage",
            "Python ID/prompt copies and token_strs/JSON/text outputs", "other tokenizer/encoder owners",
            "remaining fixed C/Rust stack, TLS runtime registration and thread overheads",
            "allocator usable capacity/retention, kernel/RSS and device qualification"),
        "caller_preconditions": ("fresh private source loader; no BPE cache resize/clear or tokenizer mutation",
            "explicit retained cache generation count; managed churn is not implicitly qualified",
            "optional final bounds cover protected + independently normalized pieces jointly, not whole-string NFC",
            "optional NFD/split/protected counts need their own source-domain proof, not final-scalar inference",
            "detached batch sequences require their own concurrent/result/container aggregation" if batch else
            "synchronous encode only; encode_batch/SegmentEncoder/Rayon require separate aggregation"),
        "evidence": "source-owned requested layouts and lifetime bounds; no whole HTTP admission/RSS grant"}


def encode_payload(bound, *, utf8_bytes, owners, cache_generations,
                   normalized_utf8_bytes=None, normalized_scalar_bound=None, decomposed_scalar_bound=None,
                   split_bound=None, protected_match_bound=None):
    """One synchronous active peak plus other returned Encoding owners.

    Cache generations are a separate, explicit lifetime bound. Optional final
    bounds cover protected literals + independently normalized pieces jointly,
    never whole-string NFC across protected boundaries or intermediate NFD.
    This contract grants no GIL serialization to batch/SegmentEncoder calls.
    """
    return _encode_layout(bound, utf8_bytes=utf8_bytes, owners=owners, cache_generations=cache_generations,
        normalized_utf8_bytes=normalized_utf8_bytes, normalized_scalar_bound=normalized_scalar_bound,
        decomposed_scalar_bound=decomposed_scalar_bound, split_bound=split_bound,
        protected_match_bound=protected_match_bound, batch=False)


def batch_encode_sequence_payload(bound, *, utf8_bytes, cache_generations,
                                  normalized_utf8_bytes=None, normalized_scalar_bound=None, decomposed_scalar_bound=None,
                                  split_bound=None, protected_match_bound=None):
    """One detached batch sequence, without an implicit batch/process total.

    The caller must aggregate simultaneous sequences and all returned Encodings,
    not assume one GIL-held peak. Global cache storage is charged once. Input
    and result batch vectors, the Python Encoding list, singleton adapter
    containers and callback-live Python sources are separate obligations.
    """
    result = _encode_layout(bound, utf8_bytes=utf8_bytes, owners=1, cache_generations=cache_generations,
        normalized_utf8_bytes=normalized_utf8_bytes, normalized_scalar_bound=normalized_scalar_bound,
        decomposed_scalar_bound=decomposed_scalar_bound, split_bound=split_bound,
        protected_match_bound=protected_match_bound, batch=True)
    result["schema"] = "tensorfold.selected-tokenizer-batch-sequence-payload-v1"
    result["single_sequence_peak_requested_bytes"] = result.pop("active_native_heap_requested_bytes")
    result.pop("native_heap_requested_bytes")
    result.pop("other_returned_encodings_bytes")
    result["budget"].pop("owners")
    result["scheduling"] = "detached batch sequence; no single-active-GIL or whole-batch grant"
    result["cache_accounting"] = "global generation storage, add once across the batch/other callers"
    result["required_external_roles"] += (
        "batch input/EncodeInput and native Encoding/PyEncoding vector backing",
        "Python result Encoding list and singleton input/result containers",
        "all simultaneously active sequences and retained batch Encodings; callback-live source owners",
    )
    return result
