"""Qwen images are sized as transformers sizes them, and a shared visual-token budget goes where it is needed.

transformers 5 keeps the checkpoint's pixel range in the processor's ``size``; the frontend read min_pixels from an
attribute that no longer exists, fell back to one merged patch, and so gave a 7x7 image 1 token where transformers
gives 64 (200x200: 36 instead of 64). And the budget was split evenly by image count, so beside a small image a large
one shrank to half the budget while most of the small image's share went unused.
"""

import json

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("PIL")

from tensorfold.vision.images import ImageInput
from tensorfold.vision.qwen_processing import QwenImageProcessor, _processor_options
from tests.test_vision_qwen_mlx import CONFIG, Tokenizer

# the Qwen3.8 checkpoint's preprocessor_config.json: 64 to 16,384 merged tokens an image
CHECKPOINT = {"size": {"shortest_edge": 65536, "longest_edge": 16777216}, "patch_size": 16, "temporal_patch_size": 2,
              "merge_size": 2, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
UNIT = 32 * 32                                          # pixels a merged token covers


@pytest.fixture
def front(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(CHECKPOINT))
    processor = transformers.Qwen2VLImageProcessor(**_processor_options(tmp_path, CONFIG["vision_config"]))
    return QwenImageProcessor(CONFIG, processor, Tokenizer())


def picture(width, height, seed=0, detail="auto"):
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return ImageInput(width, height, pixels.tobytes(), detail)


def grids(front, images, **budget):
    prepared = front.prepare("<start><image><end>" * len(images), images, **budget)
    return [tuple(int(x) for x in row) for row in prepared.image_grid_thw]


def tokens(grid):
    return int(np.prod(grid)) // 4


@pytest.mark.parametrize("width, height", [(7, 7), (40, 30), (64, 64), (200, 200), (300, 200), (500, 60),
                                           (640, 480), (1920, 1080)])
def test_an_image_gets_the_grid_and_patches_transformers_gives_it(front, width, height):
    image = picture(width, height)
    reference = front.processor(images=[image.to_pil()])           # the checkpoint's own range, no overrides
    prepared = front.prepare("<start><image><end>", [image], max_visual_tokens=16384, max_image_tokens=16384)
    assert prepared.image_grid_thw.tolist() == np.asarray(reference["image_grid_thw"]).tolist()
    np.testing.assert_array_equal(prepared.pixel_values, np.asarray(reference["pixel_values"]))
    assert tokens(prepared.image_grid_thw[0]) >= 64


def test_a_small_image_leaves_its_unused_share_to_a_large_one(front):
    small, large = picture(200, 200, 1), picture(1920, 1080, 2)
    assert [tokens(g) for g in grids(front, [small]) + grids(front, [large])] == [64, 2040]
    together = grids(front, [small, large], max_visual_tokens=2048, max_image_tokens=4096)
    assert together[0] == grids(front, [small])[0]                 # the small image keeps its full size
    assert 1024 < tokens(together[1]) and sum(map(tokens, together)) <= 2048
    caps = front._image_caps([small, large], 2048, 4096, 32, 2)
    larger = front.processor.get_number_of_image_patches(1080, 1920, {"min_pixels": 65536,
                                                                      "max_pixels": caps[1] + UNIT}) // 4
    assert 64 + larger > 2048                                       # the shared level is the highest that fits
    assert grids(front, [large, small], max_visual_tokens=2048, max_image_tokens=4096) == together[::-1]


def test_equal_images_get_equal_sizes_and_each_keeps_its_ceiling(front):
    same = grids(front, [picture(1920, 1080, 3), picture(1920, 1080, 4)], max_visual_tokens=2048)
    assert same[0] == same[1] and sum(map(tokens, same)) <= 2048
    low = picture(1920, 1080, 5, detail="low")
    assert tokens(grids(front, [low], max_visual_tokens=16384)[0]) <= 256
    capped = grids(front, [picture(4032, 3024, 6), picture(64, 64, 7)], max_visual_tokens=16384, max_image_tokens=4096)
    assert tokens(capped[0]) <= 4096 and tokens(capped[1]) == 64


def test_a_pixel_range_not_in_whole_tokens_still_sizes_as_transformers(tmp_path):
    # a ceiling of 1,000 tokens and 1,000 pixels: the shared level must not round the checkpoint's own cap down
    # (914x1184 is 972 tokens under the true cap, 945 under 1,000 whole tokens)
    size = {"shortest_edge": 65536, "longest_edge": 1000 * UNIT + 1000}
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({**CHECKPOINT, "size": size}))
    processor = transformers.Qwen2VLImageProcessor(**_processor_options(tmp_path, CONFIG["vision_config"]))
    front = QwenImageProcessor(CONFIG, processor, Tokenizer())
    image = picture(914, 1184)
    reference = processor(images=[image.to_pil()])
    prepared = front.prepare("<start><image><end>", [image], max_visual_tokens=16384)
    assert prepared.image_grid_thw.tolist() == np.asarray(reference["image_grid_thw"]).tolist()
    assert tokens(prepared.image_grid_thw[0]) == 972
