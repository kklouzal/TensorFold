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


def _span(frame):
    """The stand-in tower's span of ``frame``: row j is (frame tag, j)."""
    import torch
    tag = float(ord(frame.content_hash))
    return torch.arange(frame.tokens, dtype=torch.float16)[:, None].expand(-1, V.OUT_DIM) + 1000 * tag


class _Engine:
    """The Decoder's tower operations (``DS41Engine.tower_*``) on a stand-in tower; records each group of blocks
    and the live handles (what every rank would hold)."""

    def __init__(self, block_ms):
        self.tower_block_ms, self.tower_patches, self.image_features = block_ms, 54, None
        self.groups, self.states, self.spans, self.next = [], {}, {}, 0

    def tower_begin(self, frame):
        handle, self.next = self.next, self.next + 1
        self.states[handle] = [frame, 0]
        return handle

    def tower_run(self, handle, stop):
        st = self.states[handle]
        self.groups.append((st[0].content_hash, st[1], stop))
        st[1] = max(st[1], stop)

    def tower_span(self, handle):
        frame, done = self.states.pop(handle)
        assert done == V.LAYERS
        self.spans[handle] = _span(frame)

    def tower_stage(self, pieces):
        import torch
        rows = []
        for h, first, end, release in pieces:
            rows.append(self.spans[h][first:end])
            if release:
                del self.spans[h]
        self.image_features = torch.cat(rows)

    def tower_free(self, handles):
        for h in handles:
            self.states.pop(h, None)
            self.spans.pop(h, None)


def _decoder(streams=0, block_ms=40.0):
    from types import SimpleNamespace

    from tensorfold.families.deepseek_v4.cuda.engine import Decoder
    d = object.__new__(Decoder)
    d.e = _Engine(block_ms)
    d.streams = dict.fromkeys(range(streams))
    d.GAP_MS = 175.0
    d.rounds = SimpleNamespace(estimate=lambda bucket: 25.0)
    d._bucket = lambda count: count
    return d


def test_decoder_encodes_frames_in_groups_and_feeds_each_chunk_its_rows():
    import torch
    a, b = _frame(6, 9, "a"), _frame(3, 3, "b")            # 10 and 4 tokens
    prepared = V.expand_placeholders([5, V.IMAGE_TOKEN_ID, 7, V.IMAGE_TOKEN_ID, 8], [a, b])   # spans [1,11) [12,16)
    s = type("S", (), {})()
    s.vision, s.encoded, s.encoding = prepared, {}, None
    d = _decoder()
    assert d._encode(s, 0, 6)                               # nothing decoding: the whole frame at once
    assert d.e.groups == [("a", 0, V.LAYERS)]
    span_a, span_b = _span(a), _span(b)
    assert torch.equal(d.e.image_features, span_a[0:5]) and set(s.encoded) == {0}   # a straddles into the next chunk
    assert set(d.e.spans) == {s.encoded[0]}
    assert d._encode(s, 6, 17)
    assert torch.equal(d.e.image_features, torch.cat([span_a[5:10], span_b]))
    assert s.encoded == {} and len(d.e.groups) == 2         # frame a was encoded once
    assert not d.e.spans and not d.e.states                 # every rank's spans released after their last chunk
    d.e.image_features = None
    assert d._encode(s, 16, 18) and d.e.image_features is None   # a text-only chunk stages nothing
    # streams decoding: groups of floor(gap target / per-block estimate) blocks, one group a round
    from tensorfold.cuda.geometry import group_target
    d = _decoder(streams=3)
    s.encoded, s.encoding = {}, None
    rounds = 0
    while not d._encode(s, 0, 17):
        rounds += 1
    per = max(1, int(group_target(175.0, 25.0) // 40.0))
    groups = [g for g in d.e.groups if g[0] == "a"]
    assert all(stop - start <= per for _, start, stop in groups) and groups[-1][2] == V.LAYERS
    assert rounds == len(d.e.groups)                        # every group (and each frame's last) ends a round
    assert torch.equal(d.e.image_features, torch.cat([span_a, span_b]))
    assert not d.e.spans and not d.e.states


def test_decoder_releases_an_unfinished_encode():
    d = _decoder(streams=3)
    s = type("S", (), {})()
    a, b = _frame(6, 9, "a"), _frame(3, 3, "b")
    s.vision = V.expand_placeholders([V.IMAGE_TOKEN_ID, 7, V.IMAGE_TOKEN_ID], [a, b])
    s.encoded, s.encoding = {}, None
    while s.encoding is None or not s.encoded:              # frame a finished, frame b part-encoded
        assert not d._encode(s, 0, len(s.vision.token_ids))
    assert d.e.states and d.e.spans
    d._release(s)
    assert not d.e.states and not d.e.spans and s.encoded == {} and s.encoding is None


def _random_tower(gen, layers):
    """Full-size mmproj tensors (torch order, FP32) with ``layers`` blocks, the read_mmproj names."""
    import torch

    def r(*shape, scale=0.02):
        return torch.randn(*shape, generator=gen) * scale

    t = {"v.patch_embd.weight": r(V.DIM, 3, V.PATCH, V.PATCH), "v.patch_embd.bias": r(V.DIM),
         "v.post_ln.weight": 1 + r(V.DIM), "mm.1.weight": r(V.OUT_DIM, V.DIM * 9), "mm.1.bias": r(V.OUT_DIM),
         "mm.2.weight": r(V.OUT_DIM, V.OUT_DIM), "mm.2.bias": r(V.OUT_DIM), "v.token_embd.img_start": r(V.OUT_DIM),
         "v.token_embd.img_end": r(V.OUT_DIM), "v.image_newline": r(V.OUT_DIM)}
    for i in range(layers):
        p = f"v.blk.{i}."
        t.update({p + "attn_qkv.weight": r(3 * V.DIM, V.DIM, scale=0.05), p + "attn_qkv.bias": r(3 * V.DIM),
                  p + "attn_out.weight": r(V.DIM, V.DIM), p + "attn_out.bias": r(V.DIM),
                  p + "ffn_gate.weight": r(V.INTER, V.DIM), p + "ffn_up.weight": r(V.INTER, V.DIM),
                  p + "ffn_down.weight": r(V.DIM, V.INTER), p + "ln1.weight": 1 + r(V.DIM),
                  p + "ln2.weight": 1 + r(V.DIM)})
    return t


def test_take_shard_partitions_megatron_parts():
    import torch
    full = torch.arange(3 * 16 * 2, dtype=torch.float32).reshape(48, 2)       # q | k | v of 16 rows each
    parts = [V.take_shard(full, "v.blk.0.attn_qkv.weight", r, 8) for r in range(8)]
    assert all(p.shape == (6, 2) for p in parts)
    assert torch.equal(parts[3], torch.cat([full[6:8], full[22:24], full[38:40]]))
    matrix = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)
    for name in ("v.blk.0.attn_out.weight", "v.blk.0.ffn_down.weight", "mm.2.weight"):
        parts = [V.take_shard(matrix, name, r, 8) for r in range(8)]
        assert all(part.shape == (2, 4) for part in parts)
        assert torch.equal(torch.cat(parts), matrix)
    rows = torch.arange(16, dtype=torch.float32)
    assert torch.equal(torch.cat([V.take_shard(rows, "mm.1.bias", r, 8) for r in range(8)]), rows)
    for name in ("v.blk.0.attn_out.bias", "mm.2.bias"):
        assert torch.equal(V.take_shard(rows, name, 3, 8), rows[6:8])
    assert V.take_shard(full, "v.blk.0.attn_qkv.weight", 0, 1) is full
    with pytest.raises(ValueError):
        V.take_shard(torch.zeros(10, 3), "mm.1.weight", 0, 8)


def test_tp8_tower_equals_the_whole_tower(monkeypatch):
    """Eight simulated ranks gather features in rank order and reproduce the one-device FP32 span."""
    import threading

    import torch
    monkeypatch.setattr(V, "LAYERS", 2)
    # the CPU has no memory-efficient SDPA kernel: the math backend for both arms
    monkeypatch.setattr(V.VisionTower, "_attention",
                        lambda self, q, k, v: torch.nn.functional.scaled_dot_product_attention(
                            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))[0])
    full = _random_tower(torch.Generator().manual_seed(7), 2)
    gen = torch.Generator().manual_seed(3)
    frame = V.ImageFrame(torch.randint(0, 256, (7 * 8, 3, V.PATCH, V.PATCH), generator=gen,
                                       dtype=torch.uint8).numpy(), 7, 8, "x")
    one = V.VisionTower(full, "cpu", compute=torch.float32)
    whole = one.encode(frame, torch.float32)
    world = 8
    barrier, parts = threading.Barrier(world), [None] * world

    class Coll:
        def __init__(self, rank):
            self.rank = rank

        def gather(self, partial):
            parts[self.rank] = partial
            barrier.wait()
            gathered = torch.cat(parts, dim=-1)
            barrier.wait()
            return gathered

    towers = [V.VisionTower({n: V.take_shard(t, n, r, world) for n, t in full.items()}, "cpu",
                            compute=torch.float32, rank=r, world=world, comm=Coll(r)) for r in range(world)]
    assert all(t.nbytes() < 1.05 * one.nbytes() / world for t in towers)     # replicated parts are small
    out, errors = [None] * world, []

    def run(r):
        try:
            out[r] = towers[r].encode(frame, torch.float32)
        except BaseException as error:          # noqa: BLE001 - reported below
            errors.append(error)
            barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert all(torch.equal(o, out[0]) for o in out)
    assert out[0].shape == (frame.tokens, V.OUT_DIM)
    rel = float((out[0] - whole).norm() / whole.norm())
    assert rel < 1e-5, rel
    assert torch.equal(out[0][0], whole[0]) and torch.equal(out[0][-1], whole[-1])


def test_serve_admits_vision_for_deepseek_v41():
    from tensorfold.vision.config import validate_vision_config
    raw = {"model_type": "deepseek_v41", "image_token_id": V.IMAGE_TOKEN_ID,
           "vision_config": dict(V._VISION_CONFIG, rope_theta=10000)}
    assert validate_vision_config(raw, "deepseek_v41") is raw["vision_config"]
    with pytest.raises(ValueError):
        validate_vision_config(dict(raw, image_token_id=7), "deepseek_v41")
