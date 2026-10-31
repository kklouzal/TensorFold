"""Bounded chunk geometry compared with the independent pinned full-prefix API."""

from types import MethodType, SimpleNamespace
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp",
                    reason="requires the pinned Qwen4Exp position provider")

from tensorfold.families.qwen4_exp.media_positions import plan_media_positions  # noqa: E402

IMAGE, VIDEO, START, END = 90, 91, 92, 93


def plan(tokens, images=(), videos=(), **overrides):
    options = dict(image_grids=images, video_grids=videos, image_token_id=IMAGE, video_token_id=VIDEO,
                   vision_start_token_id=START, vision_end_token_id=END, vocab_size=100, merge=2,
                   max_rows=128, max_segments=8, max_visual_rows=32)
    options.update(overrides)
    return plan_media_positions(tokens, **options)


def reference(tokens, images, videos):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel

    owner = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    owner.get_vision_position_ids = MethodType(Qwen4ExpModel.get_vision_position_ids, owner)
    ids = torch.tensor([tokens], dtype=torch.int64)
    kinds = torch.tensor([[1 if token == IMAGE else 2 if token == VIDEO else 0 for token in tokens]])
    image_grids = torch.tensor(images, dtype=torch.int64) if images else None
    video_grids = torch.tensor(videos, dtype=torch.int64) if videos else None
    positions, delta = Qwen4ExpModel.get_rope_index(owner, ids, kinds, image_grids, video_grids)
    return positions[:, 0].t().contiguous(), delta.item()


@pytest.mark.parametrize("mode", ("text", "image", "video", "interleaved"))
def test_complete_and_chunked_coordinates_equal_pinned_full_prefix_without_retained_archive(mode):
    if mode == "text":
        tokens, images, videos = (1, 2, 3), (), ()
    elif mode == "image":
        tokens, images, videos = (1, START, *([IMAGE] * 6), END, 2), ((1, 4, 6),), ()
    elif mode == "video":
        tokens = (1, START, *([VIDEO] * 4), END, 2, START, *([VIDEO] * 4), END, 3)
        images, videos = (), ((2, 4, 4),)
    else:
        tokens = (1, START, *([IMAGE] * 6), END, 2, START, *([VIDEO] * 4), END,
                  3, START, *([IMAGE] * 4), END, 4, START, *([VIDEO] * 4), END, 5)
        images, videos = ((1, 4, 6), (1, 4, 4)), ((2, 4, 4),)
    prepared = plan(tokens, images, videos)
    expected, delta = reference(tokens, images, videos)
    assert prepared.token_ids is tokens and prepared.rope_delta == delta
    for start in range(len(tokens)):
        for count in (1, min(5, len(tokens) - start)):
            with torch.device("meta"):
                actual = prepared.coordinates(start, count, max_rows=5)
            assert actual.dtype == torch.int64 and actual.device.type == "cpu"
            torch.testing.assert_close(actual, expected[start:start + count], rtol=0, atol=0)
    assert all(not isinstance(value, torch.Tensor) for value in vars(prepared).values())
    assert len(prepared.runs) <= 2 * len(prepared.visual_runs) + 1
    assert [run.temporal_group for run in prepared.visual_runs if run.kind == "video"] in ([], [0, 1])


@pytest.mark.parametrize("defect", ("missing", "extra", "count", "start", "end", "grid", "temporal",
                                   "frame-bound", "unit-bound", "row-bound", "token", "special"))
def test_invalid_expansion_or_grid_fails_before_coordinate_allocation(monkeypatch, defect):
    tokens, images, videos, options = [1, START, *([IMAGE] * 4), END, 2], ((1, 4, 4),), (), {}
    if defect == "missing":
        images = ()
    elif defect == "extra":
        images = (*images, images[0])
    elif defect == "count":
        images = ((1, 4, 6),)
    elif defect == "start":
        tokens[1] = 2
    elif defect == "end":
        tokens[-2] = 2
    elif defect == "grid":
        images = ((1, 3, 4),)
    elif defect == "temporal":
        images = ((2, 4, 4),)
    elif defect == "frame-bound":
        videos, options = ((2**30, 4, 4),), {"max_segments": 8}
    elif defect == "unit-bound":
        options["max_visual_rows"] = 3
    elif defect == "row-bound":
        options.update(max_rows=7, max_visual_rows=7)
    elif defect == "token":
        tokens[0] = True
    else:
        options["video_token_id"] = IMAGE
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("coordinate allocation before validation"))
    with pytest.raises(ValueError):
        plan(tokens, images, videos, **options)


def test_large_axis_is_checked_before_narrowing_and_chunk_limits_are_independent():
    from tensorfold.families.qwen4_exp.position_profile import MAX_POSITION

    with pytest.raises(ValueError, match="integer THW"):
        plan((1, 2), images=((1, MAX_POSITION, 2),))
    prepared = plan((1, 2, 3, 4))
    for start, count, bound in ((-1, 1, 1), (3, 2, 2), (0, 4, 3), (0, 0, 1), (True, 1, 1)):
        with pytest.raises(ValueError, match="admitted chunk"):
            prepared.coordinates(start, count, max_rows=bound)
    assert prepared.coordinates(0, 4, max_rows=4)[:, 0].tolist() == [0, 1, 2, 3]


def test_visual_coordinate_scratch_retires_before_next_text_allocation(monkeypatch):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel

    tokens = (1, START, *([IMAGE] * 4), END, 2)
    prepared = plan(tokens, ((1, 4, 4),))
    expected, _ = reference(tokens, ((1, 4, 4),), ())
    original_helper, original_arange = Qwen4ExpModel.get_vision_position_ids, torch.arange
    previous, inside, checked = None, False, False

    def helper(*args, **kwargs):
        nonlocal previous, inside
        inside = True
        try:
            values = original_helper(*args, **kwargs)
        finally:
            inside = False
        previous = weakref.ref(values)
        return values

    def arange(*args, **kwargs):
        nonlocal checked
        if not inside and previous is not None:
            assert previous() is None, "previous visual scratch overlaps the next text allocation"
            checked = True
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(Qwen4ExpModel, "get_vision_position_ids", helper)
    monkeypatch.setattr(torch, "arange", arange)
    actual = prepared.coordinates(0, len(tokens), max_rows=len(tokens))
    assert checked
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
