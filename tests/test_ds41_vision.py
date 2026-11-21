"""DeepSeek-V4.1 vision host pieces: resize plan, placeholder expansion, preprocessing and the vision bias file."""

import hashlib

import numpy as np
import pytest

from tensorfold.families.deepseek_v4.cuda import vision as V

# (width, height) -> (n_vit_h, n_vit_w, n_llm_h, n_llm_w, tokens): upstream image_processor's plan on the friend's
# seven numerics images (ds41/vision-ref/numerics-cpu-20261031.log, plan_equal true against upstream)
KNOWN = {(1024, 701): (51, 74, 17, 25, 444), (450, 308): (33, 47, 11, 16, 189), (448, 448): (39, 39, 13, 13, 184),
         (1024, 640): (46, 74, 16, 25, 418), (300, 1400): (100, 22, 34, 8, 308), (2000, 260): (19, 143, 7, 48, 345),
         (1400, 1400): (93, 93, 31, 31, 994)}


@pytest.mark.parametrize("size", sorted(KNOWN))
def test_plan_matches_upstream_known_answers(size):
    n_llm_h, n_llm_w, best_h, best_w = V.plan_image_grid(*size)
    assert (best_h // V.PATCH, best_w // V.PATCH, n_llm_h, n_llm_w, V.num_image_tokens(n_llm_h, n_llm_w)) == KNOWN[size]


@pytest.mark.parametrize("size", [(1, 1), (1, 5000), (5000, 1), (8192, 8192), (14, 14), (37, 1013)])
def test_plan_bounds_extreme_aspects(size):
    n_llm_h, n_llm_w, best_h, best_w = V.plan_image_grid(*size)
    assert V.num_image_tokens(n_llm_h, n_llm_w) <= V.MAX_TOKENS
    assert best_h % V.PATCH == 0 and best_w % V.PATCH == 0 and V.llm_grid(best_h, best_w) == (n_llm_h, n_llm_w)


def _frame(n_vit_h, n_vit_w, tag):
    patches = np.zeros((n_vit_h * n_vit_w, 3, V.PATCH, V.PATCH), np.uint8)
    return V.ImageFrame(patches, n_vit_h, n_vit_w, tag)


def test_expand_placeholders_spans_and_continuation():
    a, b = _frame(6, 9, "a"), _frame(3, 3, "b")           # 2 x 3 grid -> 10 tokens; 1 x 1 -> 4 tokens
    assert (a.tokens, b.tokens) == (10, 4)
    prepared = V.expand_placeholders([5, V.IMAGE_TOKEN_ID, 7, V.IMAGE_TOKEN_ID, 8], [a, b])
    assert prepared.blocks == ((1, 11), (12, 16))
    assert prepared.token_ids == (5,) + (V.IMAGE_TOKEN_ID,) * 10 + (7,) + (V.IMAGE_TOKEN_ID,) * 4 + (8,)
    assert prepared.image_hashes == ("a", "b")
    assert V.image_token_types(2, 3).tolist() == [0, 1, 1, 1, 2, 1, 1, 1, 2, 3]
    longer = prepared.continued(prepared.token_ids + (9, 10))
    assert longer.blocks == prepared.blocks and longer.token_ids[-2:] == (9, 10)
    with pytest.raises(ValueError):
        prepared.continued(prepared.token_ids + (V.IMAGE_TOKEN_ID,))
    with pytest.raises(ValueError):
        prepared.continued((1,) + prepared.token_ids)
    with pytest.raises(ValueError):
        V.expand_placeholders([V.IMAGE_TOKEN_ID], [a, b])
    with pytest.raises(ValueError):
        V.expand_placeholders([1, 2], [a])


def test_config_contract():
    raw = {"image_token_id": V.IMAGE_TOKEN_ID, "vision_config": dict(V._VISION_CONFIG, rope_theta=10000)}
    V.check_config(raw)
    with pytest.raises(ValueError):
        V.check_config(dict(raw, image_token_id=1))
    with pytest.raises(ValueError):
        V.check_config(dict(raw, vision_config=dict(raw["vision_config"], downsample_ratio=2)))
    with pytest.raises(ValueError):
        V.check_config(dict(raw, vision_config=dict(raw["vision_config"], max_wh_ratio=4)))


def test_bias_file(tmp_path):
    table = np.arange(40 * 384, dtype="<f4").reshape(40, 384) / 1000
    path = tmp_path / "b.f32"
    path.write_bytes(table.tobytes())
    got, digest = V.read_bias_vl(path)
    assert np.array_equal(got, table) and digest == hashlib.sha256(table.tobytes()).hexdigest()
    path.write_bytes(table.tobytes()[:-4])
    with pytest.raises(ValueError):
        V.read_bias_vl(path)
    bad = table.copy()
    bad[3, 7] = np.nan
    path.write_bytes(bad.tobytes())
    with pytest.raises(ValueError):
        V.read_bias_vl(path)


def test_preprocess_pads_grey_and_patchifies():
    pytest.importorskip("PIL")
    from tensorfold.vision.images import ImageInput

    w, h = 700, 500
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = np.arange(w, dtype=np.uint16)[None, :] % 251
    rgb[..., 1] = (np.arange(h, dtype=np.uint16)[:, None] * 3) % 253
    frame = V.preprocess(ImageInput(w, h, rgb.tobytes()))
    n_llm_h, n_llm_w, best_h, best_w = V.plan_image_grid(w, h)
    assert (frame.n_vit_h * V.PATCH, frame.n_vit_w * V.PATCH) == (best_h, best_w)
    assert (frame.n_llm_h, frame.n_llm_w) == (n_llm_h, n_llm_w) and not frame.patches.flags.writeable
    # unpatchify = the padded image; ImageOps.pad centres the resized image on grey 127
    image = frame.patches.reshape(frame.n_vit_h, frame.n_vit_w, 3, V.PATCH, V.PATCH).transpose(2, 0, 3, 1, 4)
    image = image.reshape(3, best_h, best_w)
    from PIL import Image, ImageOps
    expected = np.asarray(ImageOps.pad(Image.fromarray(rgb), (best_w, best_h), color=V.PAD_COLOR)).transpose(2, 0, 1)
    assert np.array_equal(image, expected)
