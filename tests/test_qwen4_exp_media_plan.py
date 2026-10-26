"""Actual tokenizer/processor syntax controls, without GPU or media-quality claims."""

from dataclasses import replace
from fractions import Fraction

import pytest
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

from tensorfold.families.qwen4_exp.media_plan import QwenMediaPlanner
from tensorfold.vision.replay import MediaProbe, ReplayRef

NAMES = ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")
PROCESSOR = "qwen3vl-50e35cc0-cpu-fp32-global-t-actual-pts-v1"


@pytest.fixture
def tokenizer():
    value = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, **{v: i + 2 for i, v in enumerate(NAMES)}},
                                       unk_token="[UNK]"))
    value.pre_tokenizer = pre_tokenizers.Whitespace()
    value.add_special_tokens([AddedToken(name, normalized=False, special=True) for name in NAMES])
    return value


def planner(tokenizer, **changes):
    settings = dict(profile_identity="1" * 64, tokenizer_identity="2" * 64, template_identity="3" * 64,
                    vision_identity="4" * 64, processor_id=PROCESSOR, expected_image_id=2,
                    max_rows=512, max_segments=8, max_visual_rows=64, max_total_visual_rows=256,
                    max_rendered_bytes=4096, max_expanded_bytes=8192)
    return QwenMediaPlanner(tokenizer, **{**settings, **changes})


def probe(kind, ordinal, *, sha="a" * 64, grid=None, pts=None):
    ref = ReplayRef(f"/immutable/encoded/{ordinal}", sha, 128, ordinal, kind, "auto")
    if kind == "image":
        grid = (1, 4, 6) if grid is None else grid
        return MediaProbe(ref, "b" * 64, PROCESSOR, 96, 64, 1, (), (), None,
                          "encoded-orientation-pil-rgb-v1", (grid[1] * 16, grid[2] * 16), grid, 6144, 24576)
    grid = (2, 4, 4) if grid is None else grid
    pts = ((0, 1), (1, 10), (2, 1), (21, 10)) if pts is None else pts
    return MediaProbe(ref, "b" * 64, PROCESSOR, 64, 64, 4, (0, 1, 2, 3), pts, (30, 1),
                      "actual-pts-v1", (grid[1] * 16, grid[2] * 16), grid, 16384, 49152)


def test_interleaved_selected_ids_geometry_and_actual_pts_have_one_canonical_plan(tokenizer):
    image, video, start, end = NAMES
    rendered = f"hello {start}{image}{end} hello {start}{video}{end} hello {start}{image}{end}"
    probes = (probe("image", 0), probe("video", 1), probe("image", 2, grid=(1, 4, 4)))
    planned = planner(tokenizer)(rendered, probes, max_prompt_tokens=512, max_visual_tokens=None)
    # Exact means are .05 and 2.05; conversion/one-decimal formatting is the
    # declared pinned display rule, independently checked below.
    first, second = [f"<{float(v):.1f} seconds>" for v in (Fraction(1, 20), Fraction(41, 20))]
    expected = (f"hello {start}{image * 6}{end} hello "
                f"{first}{start}{video * 4}{end}{second}{start}{video * 4}{end} "
                f"hello {start}{image * 4}{end}")
    assert planned.token_ids == tuple(tokenizer.encode(expected, add_special_tokens=False).ids)
    assert [unit.probe_index for unit in planned.units] == [0, 1, 1, 2]
    assert [unit.run.temporal_group for unit in planned.units] == [0, 0, 1, 0]
    assert [unit.run.end - unit.run.start for unit in planned.units] == [6, 4, 4, 4]
    assert planned.probe_identities == tuple(p.identity for p in probes)


def test_content_and_profile_identity_are_independent_of_replay_path_and_token_ids(tokenizer):
    image, _, start, end = NAMES
    text = f"hello {start}{image}{end}"
    source = probe("image", 0)
    original = planner(tokenizer)(text, (source,), max_prompt_tokens=512, max_visual_tokens=None)
    relocated = replace(source, ref=replace(source.ref, path="/different/immutable/path"))
    same = planner(tokenizer)(text, (relocated,), max_prompt_tokens=512, max_visual_tokens=None)
    different = planner(tokenizer)(text, (probe("image", 0, sha="f" * 64),),
                                    max_prompt_tokens=512, max_visual_tokens=None)
    changed = planner(tokenizer, profile_identity="5" * 64)(text, (source,),
                                                           max_prompt_tokens=512, max_visual_tokens=None)
    assert original.token_ids == same.token_ids == different.token_ids == changed.token_ids
    assert original.media_identity == same.media_identity
    assert len({original.media_identity, different.media_identity, changed.media_identity}) == 3


@pytest.mark.parametrize("defect", ("order", "extra-marker", "missing-marker", "ordinal", "processor", "unit",
                                   "visual-budget", "context", "rendered-bytes", "expanded-bytes", "timestamp",
                                   "timestamp-intermediate"))
def test_invalid_media_rejects_before_target_execution(tokenizer, defect):
    image, video, start, end = NAMES
    text, sources, limits = f"{start}{image}{end}", (probe("image", 0),), {}
    options = dict(max_prompt_tokens=512, max_visual_tokens=None)
    if defect == "order":
        sources = (probe("video", 0),)
    elif defect == "extra-marker":
        text += f"{start}{image}{end}"
    elif defect == "missing-marker":
        text = "hello"
    elif defect == "ordinal":
        sources = (probe("image", 1),)
    elif defect == "processor":
        sources = (replace(sources[0], processor_id="different"),)
    elif defect == "unit":
        limits["max_visual_rows"] = 4
    elif defect == "visual-budget":
        options["max_visual_tokens"] = 5
    elif defect == "context":
        options["max_prompt_tokens"] = 3
    elif defect == "rendered-bytes":
        limits["max_rendered_bytes"] = 1
    elif defect == "expanded-bytes":
        limits["max_expanded_bytes"] = 1
    else:
        exponent = 308 if defect == "timestamp-intermediate" else 400
        text, sources = f"{start}{video}{end}", (probe("video", 0, pts=((10**exponent, 1),) * 4),)
    with pytest.raises(ValueError):
        planner(tokenizer, **limits)(text, sources, **options)


def test_selected_tokenizer_must_preserve_model_image_id_and_all_input_rows(tokenizer):
    with pytest.raises(ValueError, match="image/PLE"):
        planner(tokenizer, expected_image_id=3)
    tokenizer.enable_truncation(8)
    with pytest.raises(ValueError, match="untruncated"):
        planner(tokenizer)


def test_video_syntax_matches_whole_pinned_processor_for_cfr_metadata(tokenizer):
    import numpy as np
    from types import MethodType, SimpleNamespace
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    image, video, start, end = NAMES
    source = probe("video", 0, pts=((0, 30), (1, 30), (2, 30), (3, 30)))
    text = f"hello {start}{video}{end}"
    actual = planner(tokenizer)(text, (source,), max_prompt_tokens=512, max_visual_tokens=None)
    owner = SimpleNamespace(video_processor=SimpleNamespace(merge_size=2, temporal_patch_size=2),
                            vision_start_token=start, video_token=video, vision_end_token=end)
    # Use the actual library's own index/FPS API independently of our PTS adapter.
    owner._calculate_timestamps = MethodType(Qwen3VLProcessor._calculate_timestamps, owner)
    inputs = {"video_grid_thw": np.asarray([source.grid_thw]),
              "video_metadata": [SimpleNamespace(fps=30., frames_indices=[0, 1, 2, 3])]}
    expanded = Qwen3VLProcessor.replace_video_token(owner, inputs, 0)
    assert actual.token_ids == tuple(tokenizer.encode("hello " + expanded, add_special_tokens=False).ids)


def test_shared_contract_reconstructs_exact_leader_plan_without_tokenizer(tokenizer, monkeypatch):
    import sys

    image, video, start, end = NAMES
    owner = planner(tokenizer)
    sources = (probe("image", 0), probe("video", 1), probe("image", 2, grid=(1, 4, 4)))
    rendered = f"hello {start}{image}{end} hello {start}{video}{end} hello {start}{image}{end}"
    original = owner(rendered, sources, max_prompt_tokens=512, max_visual_tokens=None)
    with monkeypatch.context() as blocked:
        blocked.setitem(sys.modules, "tokenizers", None)
        blocked.setitem(sys.modules, "transformers", None)
        remote = owner.contract.from_expanded(original.token_ids, sources)
    assert remote == original and remote.profile_identity == "1" * 64
    assert not hasattr(owner.contract, "tokenizer")


def test_follower_factory_refuses_canonical_media_order_mismatch(tokenizer):
    image, video, start, end = NAMES
    owner = planner(tokenizer)
    sources = (probe("video", 0), probe("image", 1))
    original = owner(f"{start}{video}{end}{start}{image}{end}", sources,
                     max_prompt_tokens=512, max_visual_tokens=None)
    swapped = (replace(sources[1], ref=replace(sources[1].ref, ordinal=0)),
               replace(sources[0], ref=replace(sources[0].ref, ordinal=1)))
    with pytest.raises(ValueError, match="canonical media order"):
        owner.contract.from_expanded(original.token_ids, swapped)


@pytest.mark.parametrize("exponent", [308, 400])
def test_follower_factory_retains_timestamp_intermediate_guard(tokenizer, exponent):
    video, start, end = NAMES[1:]
    owner = planner(tokenizer)
    original = owner(f"{start}{video}{end}", (probe("video", 0),),
                     max_prompt_tokens=512, max_visual_tokens=None)
    with pytest.raises(ValueError, match="finite processor"):
        owner.contract.from_expanded(original.token_ids, (probe("video", 0, pts=((10**exponent, 1),) * 4),))



def test_startup_contract_record_roundtrip_is_checked_and_owns_immutable_values(tokenizer):
    from tensorfold.families.qwen4_exp.media_plan import MediaPlanContract

    original = planner(tokenizer).contract
    record = original.record()
    restored = MediaPlanContract.from_record(record)
    assert restored == original
    record["identities"][0] = "f" * 64
    record["roles"][0] = 9999
    assert restored == original
    for bad in ({**restored.record(), "extra": 1}, {**restored.record(), "version": True},
                {**restored.record(), "roles": [0, 1]}, {**restored.record(), "identities": "bad"},
                {**restored.record(), "max_rows": True}):
        with pytest.raises(ValueError):
            MediaPlanContract.from_record(bad)



def test_each_public_entry_validates_once_without_repeating_immutable_timestamp_work(tokenizer, monkeypatch):
    from tensorfold.families.qwen4_exp.media_plan import MediaPlanContract

    owner = planner(tokenizer)
    _, video, start, end = NAMES
    sources = (probe("video", 0),)
    calls = []
    original = MediaPlanContract.check_probes

    def checked(contract, probes, **kwargs):
        calls.append(1)
        return original(contract, probes, **kwargs)

    monkeypatch.setattr(MediaPlanContract, "check_probes", checked)
    local = owner(f"{start}{video}{end}", sources, max_prompt_tokens=512, max_visual_tokens=None)
    assert len(calls) == 1
    remote = owner.contract.from_expanded(local.token_ids, sources)
    assert len(calls) == 2 and remote == local
