"""Canonical prepared-media rebinding and cache cuts; no weights or GPU work."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from tensorfold.families.gemma4.cuda.vision import (
    GemmaCudaVision, GemmaVisionProcessor, prefix_media_tags, rebind_prepared, retire_cache_metadata,
)
from tensorfold.vision.images import ImageInput
from tensorfold.vision.videos import VideoInput
from test_gemma_cuda_vision_processor import standard_processor, get_aspect_ratio_preserving_size
from test_prompt_literals import tokenizer as _literal_tokenizer_fixture

literal_tokenizer = _literal_tokenizer_fixture  # Expose the existing fixture without redefining an import.


class MetadataLease:
    def __init__(self, nbytes):
        self.nbytes, self.closed, self.closes = nbytes, False, 0

    def close(self):
        if not self.closed:
            self.closed = True
            self.closes += 1


class MetadataRegistration:
    def __init__(self, owner):
        self.owner, self.failed, self.retiring, self.closed = owner, False, False, False
        self.consumer_done, self.close_failed = False, False
        self.primary = None

    def close(self):
        self.consumer_done = True
        if self.failed and not self.retiring:
            return
        self.owner, self.closed = None, True


def metadata_credit(nbytes):
    assert type(nbytes) is int and nbytes > 0
    return MetadataLease(nbytes)


def metadata_register(owner):
    registration = MetadataRegistration(owner)
    owner.registration = registration
    return registration


def metadata_failed(primary, owner, *, close_failed=False):
    owner.registration.failed = True
    owner.registration.close_failed = close_failed
    owner.registration.primary = primary


def prepared_input(text="<|image|> : <|image|>", *, video=None):
    raw, processor = standard_processor()
    front = GemmaVisionProcessor(raw, processor, get_aspect_ratio_preserving_size)
    pixels = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    images = [ImageInput(8, 4, pixels.tobytes()), ImageInput(8, 4, (255 - pixels).tobytes())]
    if video is not None:
        images = []
    prepared = front.prepare(text, images, videos=[] if video is None else [video], max_prompt_tokens=512, metadata_credit=metadata_credit,
        metadata_register=metadata_register, metadata_failed=metadata_failed)
    return raw, front, prepared


def tower(raw, prepared):
    vision = object.__new__(GemmaCudaVision)
    vision.image_token, vision.video_token = raw["image_token_id"], raw["video_token_id"]
    vision.hidden_size = 4
    calls = []
    def features(frame):
        calls.append(frame)
        index = next(i for i, candidate in enumerate(prepared.frames) if candidate is frame)
        return torch.full((frame.soft_tokens, 4), float(index + 1))
    vision._features = features
    return vision, calls


def test_rebind_borrows_frames_and_preserves_per_block_identity_at_new_positions():
    raw, front, prepared = prepared_input()
    tokens = (front.processor.tokenizer.convert_tokens_to_ids(":"),) + prepared.token_ids
    rebound = rebind_prepared(prepared, tokens)
    assert rebound is not prepared and rebound.frames is prepared.frames
    assert rebound.frame_identities is prepared.frame_identities and rebound.processor_owner is prepared.processor_owner
    assert rebound.image_hashes is prepared.image_hashes
    old, new = prefix_media_tags(prepared), prefix_media_tags(rebound)
    assert [(a + 1, b + 1, identity) for a, b, identity in old] == list(new)
    for begin, end, _ in new:
        assert all(t == raw["image_token_id"] for t in tokens[begin:end])


def test_canonical_tokens_reuses_library_expansion_without_processor_or_media_calls(monkeypatch):
    _, front, prepared = prepared_input()
    # Initial prep and canonical text have exactly the same processor-owned media expansion.
    assert front.canonical_tokens(prepared, "<|image|> : <|image|>") == prepared.token_ids
    def forbidden(*args, **kwargs):
        raise AssertionError("canonicalization reprocessed media")
    monkeypatch.setattr(type(front.processor), "__call__", forbidden)
    monkeypatch.setattr(front.processor.image_processor, "preprocess", forbidden)
    monkeypatch.setattr(front.processor.video_processor, "preprocess", forbidden)
    tokens = front.canonical_tokens(prepared, "<|image|> : <|image|> : 123")
    rebound = rebind_prepared(prepared, tokens)
    assert rebound.frames is prepared.frames and len(rebound.blocks) == 2
    assert len(tokens) > len(prepared.token_ids)


def test_different_processor_or_missing_media_fails_closed():
    _, front, prepared = prepared_input()
    _, other, _ = prepared_input()
    with pytest.raises(ValueError, match="authority"):
        other.canonical_tokens(prepared, "<|image|> : <|image|>")
    with pytest.raises(ValueError, match="authority"):
        front.canonical_tokens(prepared, "<|image|>")
    with pytest.raises(ValueError):
        rebind_prepared(prepared, [True, *prepared.token_ids])
    with pytest.raises(ValueError, match="original block"):
        rebind_prepared(replace(prepared, blocks=((0, 1),)), prepared.token_ids)
    data = prepared.cache_metadata_owner.data
    prepared.cache_metadata_owner.data = replace(data, processor_binding="different")
    with pytest.raises(ValueError, match="identity"):
        prefix_media_tags(prepared)


def test_suffix_encodes_only_uncached_frames_and_retains_absolute_rows():
    raw, _, prepared = prepared_input()
    vision, calls = tower(raw, prepared)
    cut = prepared.blocks[0][1]
    suffix = vision.encode_suffix(prepared, prepared.token_ids, cut)
    assert len(calls) == 1 and calls[0] is prepared.frames[1]
    assert suffix.blocks == prepared.blocks[1:]
    assert suffix.rows == tuple(range(*prepared.blocks[1]))
    assert torch.equal(suffix.features, torch.full((prepared.frames[1].soft_tokens, 4), 2.0))
    all_features = vision.encode(prepared, prepared.token_ids)
    assert all_features.blocks == prepared.blocks and len(calls) == 3
    assert torch.equal(all_features.features[-len(suffix.rows):], suffix.features)


def test_cut_inside_bidirectional_block_rejected_before_tower_or_torch_import(monkeypatch):
    raw, _, prepared = prepared_input()
    vision, calls = tower(raw, prepared)
    begin, end = prepared.blocks[0]
    assert end - begin > 1
    for cut in (begin + 1, -1, True, len(prepared.token_ids) + 1):
        with pytest.raises(ValueError, match="whole bidirectional"):
            vision.encode_suffix(prepared, prepared.token_ids, cut)
    assert not calls
    import builtins
    original = builtins.__import__
    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("empty suffix imported Torch")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    encoded = vision.encode_suffix(prepared, prepared.token_ids, len(prepared.token_ids))
    assert encoded.rows == encoded.blocks == () and encoded.features is None and not calls


def test_changed_full_prompt_refused_before_any_reencode():
    raw, _, prepared = prepared_input()
    vision, calls = tower(raw, prepared)
    with pytest.raises(ValueError, match="different prompt"):
        vision.encode_suffix(prepared, [*prepared.token_ids, 7], 0)
    assert not calls


def test_video_future_frames_do_not_change_earlier_atomic_tags_and_pts_are_exact():
    pixels = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    two = VideoInput(np.stack([pixels, 255 - pixels]), (0, 1), 1.0, (0.0, 1.1))
    three = VideoInput(np.stack([pixels, 255 - pixels, pixels]), (0, 1, 2), 1.0, (0.0, 1.1, 2.0))
    _, front, old = prepared_input("<|video|>", video=two)
    # Same actual processor/geometry; independent later request only adds a future frame.
    later = front.prepare("<|video|>", [], videos=[three], max_prompt_tokens=512, metadata_credit=metadata_credit,
        metadata_register=metadata_register, metadata_failed=metadata_failed)
    assert prefix_media_tags(old) == prefix_media_tags(later)[:2]
    assert old.video_hashes != later.video_hashes
    changed = VideoInput(np.stack([pixels, 255 - pixels]), (0, 1), 1.0, (0.0, 1.2))
    different = front.prepare("<|video|>", [], videos=[changed], max_prompt_tokens=512, metadata_credit=metadata_credit,
        metadata_register=metadata_register, metadata_failed=metadata_failed)
    assert old.token_ids == different.token_ids
    a, b = prefix_media_tags(old), prefix_media_tags(different)
    assert a[0] == b[0] and a[1] != b[1]
    assert isinstance(a[1][2][3], bytes) and len(a[1][2][3]) == 8
    assert front.canonical_tokens(old, "<|video|> : 123")[:len(old.token_ids)] == old.token_ids


def test_future_image_identity_not_in_prior_frame_tag():
    _, front, old = prepared_input()
    first = old.frames[0]
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    # Existing first visual input is reproduced exactly; only the later input changes.
    source = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    changed = front.prepare("<|image|> : <|image|>",
        [ImageInput(8, 4, source.tobytes()), ImageInput(8, 4, pixels.tobytes())], max_prompt_tokens=512,
        metadata_credit=metadata_credit,
        metadata_register=metadata_register, metadata_failed=metadata_failed)
    assert np.array_equal(first.pixels, changed.frames[0].pixels)
    a, b = prefix_media_tags(old), prefix_media_tags(changed)
    assert a[0] == b[0] and a[1] != b[1]


def test_credit_exhaustion_preserves_generation_but_blocks_all_media_cache():
    from tensorfold.server.errors import CapacityError
    raw, processor = standard_processor()
    frontend = GemmaVisionProcessor(raw, processor, get_aspect_ratio_preserving_size)
    image = ImageInput(8, 4, bytes(4 * 8 * 3))
    for failure in (MemoryError("store full"), CapacityError("host full")):
        calls = []
        def decline(amount):
            calls.append(amount)
            raise failure
        result = frontend.prepare("<|image|>", [image], max_prompt_tokens=512, metadata_credit=decline,
            metadata_register=metadata_register, metadata_failed=metadata_failed)
        assert calls and calls[0] > 0 and result.frames and result.token_ids
        assert result.cache_metadata_complete is False and result.cache_metadata_failure == "metadata_unfunded"
        assert result.cache_metadata_owner is None and result.frame_identities == ()
        with pytest.raises(ValueError):
            prefix_media_tags(result)
        with pytest.raises(ValueError):
            frontend.canonical_tokens(result, "<|image|> : 123")
        vision, _ = tower(raw, result)
        assert vision.encode(result, result.token_ids).features is not None


def test_credit_once_before_hashes_and_rebind_borrows_same_lease(monkeypatch):
    from tensorfold.families.gemma4.cuda import vision as module
    events = []
    original = module._frame_content_hash
    def hashed(frame):
        assert events and events[0][0] == "credit"
        events.append(("hash", len(frame)))
        return original(frame)
    monkeypatch.setattr(module, "_frame_content_hash", hashed)
    pixels = np.zeros((2, 4, 8, 3), dtype=np.uint8)
    video = VideoInput(pixels, (0, 1), 1.0)
    raw, processor = standard_processor()
    frontend = GemmaVisionProcessor(raw, processor, get_aspect_ratio_preserving_size)
    def credit(n):
        events.append(("credit", n))
        return metadata_credit(n)
    prepared = frontend.prepare("<|video|>", [], videos=[video], max_prompt_tokens=512, metadata_credit=credit,
        metadata_register=metadata_register, metadata_failed=metadata_failed)
    rebound = rebind_prepared(prepared, frontend.canonical_tokens(prepared, "<|video|> : 123"))
    assert prepared.cache_metadata_complete is True
    assert rebound.cache_metadata_owner is prepared.cache_metadata_owner
    assert prepared.cache_metadata_bytes == events[0][1]
    assert sum(kind == "credit" for kind, _ in events) == 1
    credit_owner = prepared.cache_metadata_owner.credit
    retire_cache_metadata(rebound)
    retire_cache_metadata(prepared)
    assert credit_owner.closes == 1


def test_original_request_credit_is_used_without_second_cache_debit():
    from types import SimpleNamespace
    from tensorfold.server import host_payload as hp
    pool = hp.HostPayloadPool(1 << 20, body_bound=lambda n: n, generation_bound=lambda n, o: (n + o, o),
                             encoding_bound=lambda n: n, cache_bound=lambda n, copying: n, measurement_bytes=0)
    raw, processor = standard_processor()
    frontend = GemmaVisionProcessor(raw, processor, get_aspect_ratio_preserving_size)
    def forbidden(_):
        raise AssertionError("two owners debited for the same metadata")
    with hp.handler_scope(SimpleNamespace(host_payload_pool=pool)) as owner:
        result = frontend.prepare("<|image|>", [ImageInput(8, 4, bytes(4 * 8 * 3))],
                                  max_prompt_tokens=512, metadata_credit=forbidden,
                                  metadata_register=metadata_register, metadata_failed=metadata_failed)
        assert result.cache_metadata_complete is True and result.cache_metadata_owner.credit is not None
        assert owner.claims["canonical-media"] == result.cache_metadata_bytes
        assert result.cache_metadata_registration.owner is result.cache_metadata_owner
        retire_cache_metadata(result)
        del result
    assert pool.used == 0


def test_metadata_capture_failure_retains_credit_until_registry_retirement(monkeypatch):
    from tensorfold.families.gemma4.cuda import vision as module
    raw, processor = standard_processor()
    frontend = GemmaVisionProcessor(raw, processor, get_aspect_ratio_preserving_size)
    lease, captured = metadata_credit(1), []
    def register(owner):
        captured.append(owner)
        return metadata_register(owner)
    def fail_hash(_):
        raise RuntimeError("frame hash failed")
    monkeypatch.setattr(module, "_frame_content_hash", fail_hash)
    video = VideoInput(np.zeros((1, 4, 8, 3), dtype=np.uint8), (0,), 1.0)
    with pytest.raises(RuntimeError, match="frame hash failed") as error:
        frontend.prepare("<|video|>", [], videos=[video], max_prompt_tokens=512, metadata_credit=lambda _: lease,
                         metadata_register=register, metadata_failed=metadata_failed)
    registration = captured[0].registration
    assert registration.primary is error.value
    assert registration.failed and registration.consumer_done and not registration.closed
    assert not lease.closed  # Registered failed owner, not frontend, owns eventual refund.


def test_failure_notification_error_does_not_replace_primary_or_skip_cpu_retirement(monkeypatch):
    from tensorfold.families.gemma4.cuda import vision as module
    raw, processor = standard_processor()
    frontend = GemmaVisionProcessor(raw, processor, get_aspect_ratio_preserving_size)
    lease, captured = metadata_credit(1), []
    def register(owner):
        captured.append(owner)
        return metadata_register(owner)
    def fail_hash(_):
        raise RuntimeError("frame hash failed")
    def fail_notification(primary, owner):
        metadata_failed(primary, owner)
        raise OSError("failure notification failed")
    monkeypatch.setattr(module, "_frame_content_hash", fail_hash)
    video = VideoInput(np.zeros((1, 4, 8, 3), dtype=np.uint8), (0,), 1.0)
    with pytest.raises(RuntimeError, match="frame hash failed") as error:
        frontend.prepare("<|video|>", [], videos=[video], max_prompt_tokens=512, metadata_credit=lambda _: lease,
                         metadata_register=register, metadata_failed=fail_notification)
    registration = captured[0].registration
    assert registration.failed and registration.consumer_done and not registration.closed and not lease.closed
    # Existing cleanup._transport preserves a sole cleanup cause directly.
    assert type(error.value.__cause__) is OSError
    assert error.value.__cause__.args == ("failure notification failed",)
    assert registration.primary is error.value


def test_literal_origin_canonical_roundtrip_does_not_reclassify_quoted_media(literal_tokenizer):
    import json
    from transformers import GemmaTokenizer
    from tensorfold.server.prompt_literals import PromptLiterals, RenderedPrompt
    from tensorfold.server.prompt_origin import origin_concat, trusted
    raw, processor = standard_processor()
    names = {"image_token": "<|image|>", "boi_token": "<|image>", "eoi_token": "<image|>",
             "audio_token": "<|audio|>", "boa_token": "<|audio>", "eoa_token": "<audio|>"}
    native = GemmaTokenizer(tokenizer_object=literal_tokenizer, model_specific_special_tokens=names,
                            extra_special_tokens=["<|video|>"])
    processor = type(processor)(processor.feature_extractor, processor.image_processor, native,
                                processor.video_processor, image_seq_length=70)
    raw["image_token_id"] = native.convert_tokens_to_ids("<|image|>")
    raw["video_token_id"] = native.convert_tokens_to_ids("<|video|>")
    protection = PromptLiterals(json.loads(native.backend_tokenizer.to_str()), ["template"])
    user, _ = protection.protect("user quotes <|image|>")
    tool, _ = protection.protect("tool quotes <|video|>")
    prompt = RenderedPrompt(origin_concat((user, trusted(" <|image|> "), tool,
                                           trusted(" <|video|>"))), protection.literals)
    frontend = GemmaVisionProcessor(raw, processor, get_aspect_ratio_preserving_size)
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    prepared = frontend.prepare(prompt, [ImageInput(8, 4, pixels.tobytes())],
        videos=[VideoInput(pixels[None], (0,), 1.0)], max_prompt_tokens=512,
        metadata_credit=metadata_credit, metadata_register=metadata_register, metadata_failed=metadata_failed)
    assert frontend.canonical_tokens(prepared, prompt) == prepared.token_ids
    assistant, _ = protection.protect(" assistant quotes <|image|> and <|video|>")
    completed = RenderedPrompt(origin_concat((prompt.origin, assistant)), protection.literals)
    tokens = frontend.canonical_tokens(prepared, completed)
    rebound = rebind_prepared(prepared, tokens)
    assert rebound.frames is prepared.frames and prefix_media_tags(rebound) == prefix_media_tags(prepared)
    for kind, token in (("image", raw["image_token_id"]), ("video", raw["video_token_id"])):
        assert tokens.count(token) == sum(f.soft_tokens for f in prepared.frames if f.kind == kind)
    text = native.decode(tokens, skip_special_tokens=False)
    assert "user quotes <|image|>" in text and "tool quotes <|video|>" in text
    assert "assistant quotes <|image|> and <|video|>" in text


def test_funded_payload_covers_actual_slotted_metadata_lifetime(monkeypatch):
    import dataclasses
    import sys
    from tensorfold.families.gemma4.cuda import vision as module
    measured_offsets = []
    original = module._cache_metadata_bound
    def observe(offsets, frames, row_copy_bytes):
        measured_offsets.append(offsets)
        return original(offsets, frames, row_copy_bytes)
    monkeypatch.setattr(module, "_cache_metadata_bound", observe)
    _, frontend, prepared = prepared_input()
    assert not hasattr(prepared, "__dict__")
    rebound = rebind_prepared(prepared, frontend.canonical_tokens(prepared, "<|image|> : <|image|> : 123"))
    tags = prefix_media_tags(rebound)
    # Existing immutable pixels/frames, prompt IDs, initial block/hash tuples,
    # persistent processor and separately funded registration/credit are borrowed.
    excluded = {id(value) for value in (prepared.frames, prepared.token_ids, rebound.token_ids,
        prepared.blocks, prepared.image_hashes, prepared.video_hashes, prepared.processor_owner,
        prepared.cache_metadata_owner.credit, prepared.cache_metadata_registration,
        metadata_failed)}
    def owned_bytes(roots):
        seen = set(excluded)
        def visit(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            amount = sys.getsizeof(value)
            if type(value) in (tuple, list):
                return amount + sum(visit(v) for v in value)
            if type(value) is dict:
                return amount + sum(visit(k) + visit(v) for k, v in value.items())
            if dataclasses.is_dataclass(value):
                return amount + sum(visit(getattr(value, f.name)) for f in dataclasses.fields(value))
            if type(value) is module._CacheMetadata:
                return amount + sum(visit(getattr(value, slot)) for slot in value.__slots__)
            return amount
        return sum(visit(value) for value in roots)
    prepare_peak = owned_bytes((prepared, measured_offsets[0]))
    publication_peak = owned_bytes((prepared, rebound, tags))
    assert prepared.cache_metadata_bytes >= max(prepare_peak, publication_peak)


def test_shared_optional_owner_clears_every_prepared_alias_before_credit_refund():
    from tensorfold.families.gemma4.cuda.vision import continued
    raw, frontend, prepared = prepared_input()
    direct_alias = prepared
    rebound = rebind_prepared(prepared, frontend.canonical_tokens(prepared, "<|image|> : <|image|> : 123"))
    extended = continued(prepared, (*prepared.token_ids, 7), raw)
    aliases = (prepared, direct_alias, rebound, extended)
    owner, credit = prepared.cache_metadata_owner, prepared.cache_metadata_owner.credit
    original_close = credit.close
    def credited_close():
        for alias in aliases:
            assert not alias.cache_metadata_complete
            assert alias.frame_identities == alias.image_replacements == alias.video_replacements == ()
            assert alias.processor_owner is None
        original_close()
    credit.close = credited_close
    originals = (prepared.token_ids, prepared.frames, prepared.blocks, prepared.image_hashes, prepared.video_hashes)
    before, _ = tower(raw, prepared)
    expected = before.encode(prepared, prepared.token_ids).features
    retire_cache_metadata(rebound)
    retire_cache_metadata(direct_alias)
    assert credit.closes == 1 and owner.closed and owner.credit is None
    assert owner.registration is owner.failed_callback is None
    for actual, original in zip((prepared.token_ids, prepared.frames, prepared.blocks,
                                 prepared.image_hashes, prepared.video_hashes), originals):
        assert actual is original
    after, _ = tower(raw, prepared)
    assert torch.equal(after.encode(prepared, prepared.token_ids).features, expected)
    assert extended.frames is prepared.frames and extended.token_ids[-1] == 7


def test_closed_optional_owner_does_not_pin_old_engine_callbacks():
    import weakref
    _, _, prepared = prepared_input()
    class EngineCallback:
        def failed(self, *args, **kwargs):
            raise AssertionError("no failure expected")
    engine = EngineCallback()
    reference = weakref.ref(engine)
    prepared.cache_metadata_owner.failed_callback = engine.failed
    del engine
    assert reference() is not None
    retire_cache_metadata(prepared)
    assert reference() is None
    assert not prepared.cache_metadata_complete and prepared.cache_metadata_failure == "metadata_retired"


def test_failed_retirement_is_sticky_without_retry_or_base_plan_loss():
    _, _, prepared = prepared_input()
    owner, registration = prepared.cache_metadata_owner, prepared.cache_metadata_registration
    credit, original_frames = owner.credit, prepared.frames
    primary, attempts = OSError("credit retirement uncertain"), []
    def fail_close():
        attempts.append(1)
        raise primary
    credit.close = fail_close
    with pytest.raises(OSError) as error:
        retire_cache_metadata(prepared)
    assert error.value is primary and registration.primary is primary and registration.close_failed
    assert registration.consumer_done and not registration.closed and not owner.closed
    assert owner.data is None and owner.credit is credit and prepared.frames is original_frames
    assert not prepared.cache_metadata_complete and prepared.cache_metadata_failure == "metadata_close_failed"
    with pytest.raises(OSError) as repeated:
        owner.close()
    assert repeated.value is primary and len(attempts) == 1


def test_positive_engine_retirement_clears_failed_shared_owner_before_refund():
    _, _, prepared = prepared_input()
    owner, registration = prepared.cache_metadata_owner, prepared.cache_metadata_registration
    credit = owner.credit
    metadata_failed(RuntimeError("canonical failed"), owner)
    retire_cache_metadata(prepared)  # Consumer is done; failed owner remains funded.
    assert not registration.closed and registration.consumer_done and not credit.closed
    assert owner.data is not None and not prepared.cache_metadata_complete
    registration.retiring = True
    original_close = credit.close
    def credited_close():
        assert owner.data is None and prepared.frame_identities == ()
        original_close()
    credit.close = credited_close
    owner.close()
    registration.close()
    assert registration.closed and credit.closes == 1 and owner.registration is owner.failed_callback is None


def test_concurrent_alias_retirement_refunds_one_credit_after_data_clear():
    import threading
    _, frontend, prepared = prepared_input()
    rebound = rebind_prepared(prepared, frontend.canonical_tokens(prepared, "<|image|> : <|image|> : 123"))
    owner, credit = prepared.cache_metadata_owner, prepared.cache_metadata_owner.credit
    entered, release = threading.Event(), threading.Event()
    original_close, errors = credit.close, []
    def blocked_close():
        entered.set()
        assert release.wait(2)
        original_close()
    credit.close = blocked_close
    def retire(alias):
        try:
            retire_cache_metadata(alias)
        except BaseException as error:
            errors.append(error)
    first = threading.Thread(target=retire, args=(prepared,))
    second = threading.Thread(target=retire, args=(rebound,))
    first.start()
    try:
        assert entered.wait(2)
        second.start()
        assert owner.data is None and not credit.closed
        assert not prepared.cache_metadata_complete and rebound.frame_identities == ()
    finally:
        release.set()
        first.join(2)
        if second.ident is not None:
            second.join(2)
    assert not first.is_alive() and not second.is_alive() and not errors
    assert credit.closes == 1 and owner.closed and owner.failed_callback is None
