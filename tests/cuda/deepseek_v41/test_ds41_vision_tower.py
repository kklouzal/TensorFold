"""DeepSeek-V4.1 vision tower on one V100 against upstream's FP32 math and the friend's FP16 tower.

References (read-only, outside the repository; the test skips without them):
  TENSORFOLD_DS41_MMPROJ        mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf
  TENSORFOLD_DS41_VISION_REF    dir with deepseek-ai/DeepSeek-V4.1-Flash-1017 inference/vision.py and
                                image_processor.py (rev 1bae0e10, MIT) and the example images *.jpeg
  TENSORFOLD_DS41_FRIEND_TOWER  dir with the friend's patch-150 local_ds41_vision_tower.py (optional: parity gate)
  TENSORFOLD_DS41_BIAS_VL       exp_probs_b_vl.f32 (optional: digest check)
Gates per image: our FP32 tower == upstream FP32 (rel L2 < 1e-3, the friend's gate); our FP16 production tower
(one device, world 1) no worse than the friend's FP16 tower (rel L2 and min token cosine vs upstream FP32; it keeps
the row-parallel GEMM outputs FP32 where his round to FP16, so it is not bitwise his); delimiters exact; plan and
patches equal to upstream's image_processor (patches compared after upstream's BF16 rounding). The TP8 tower's
parity is ds41_vision_tp_check.py (eight ranks).
"""

import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

MMPROJ = os.environ.get("TENSORFOLD_DS41_MMPROJ")
REF = os.environ.get("TENSORFOLD_DS41_VISION_REF")
FRIEND = os.environ.get("TENSORFOLD_DS41_FRIEND_TOWER")
BIAS = os.environ.get("TENSORFOLD_DS41_BIAS_VL")
RESULTS = os.environ.get("TENSORFOLD_DS41_VISION_RESULTS")


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic(name):
    from PIL import Image, ImageDraw, ImageFont

    if name == "number448":
        img = Image.new("RGB", (448, 448), "white")
        d = ImageDraw.Draw(img)
        d.rectangle([45, 45, 150, 150], fill=(220, 30, 30))
        d.text((45, 224), "4821", fill="black", font=ImageFont.load_default(size=56))
    elif name == "text1024":
        img = Image.new("RGB", (1024, 640), (240, 240, 230))
        d = ImageDraw.Draw(img)
        for i, line in enumerate(["THE QUICK BROWN FOX", "jumps over 13 lazy dogs", "PRICE: $42.17"]):
            d.text((40, 60 + 150 * i), line, fill=(10, 10, 10), font=ImageFont.load_default(size=48))
        for k in range(5):
            d.ellipse([700 + 50 * k, 500, 740 + 50 * k, 540], fill=(30, 60, 220))
    elif name == "tall":
        img = Image.new("RGB", (300, 1400), (20, 120, 40))
        ImageDraw.Draw(img).rectangle([50, 600, 250, 800], fill=(250, 250, 0))
    elif name == "wide":
        img = Image.new("RGB", (2000, 260), (200, 200, 255))
        ImageDraw.Draw(img).text((20, 80), "wide banner 7", fill="black", font=ImageFont.load_default(size=80))
    else:                                    # "max": 1400 x 1400 noise, the 994-token plan
        g = torch.Generator().manual_seed(0)
        img = Image.fromarray((torch.rand(1400, 1400, 3, generator=g) * 255).to(torch.uint8).numpy())
        ImageDraw.Draw(img).rectangle([300, 300, 900, 900], fill=(255, 0, 0))
    return img


def _compare(a, ref):
    a, ref = a.float(), ref.float()
    cos = torch.nn.functional.cosine_similarity(a, ref, dim=-1)
    return dict(max_abs=float((a - ref).abs().max()), rel_l2=float((a - ref).norm() / ref.norm()),
                min_token_cos=float(cos.min()), finite=bool(torch.isfinite(a).all()))


@pytest.fixture(scope="module")
def setup():
    if not (MMPROJ and REF):
        pytest.skip("set TENSORFOLD_DS41_MMPROJ and TENSORFOLD_DS41_VISION_REF")
    from tensorfold.families.deepseek_v4.cuda import vision as V

    tensors = V.read_mmproj(MMPROJ)
    ref_vision = _module("ds41_ref_vision", Path(REF) / "vision.py")
    ref_ip = _module("ds41_ref_image_processor", Path(REF) / "image_processor.py")
    # upstream builds its RoPE tables on the CPU; only their placement changes here (values unchanged)
    cpu_cos_sin = ref_vision.get_vision_cos_sin
    ref_vision.get_vision_cos_sin = lambda *a: tuple(t.cuda() for t in cpu_cos_sin(*a))
    import types

    args = types.SimpleNamespace(
        vision_patch_size=14, vision_dim=1024, vision_n_heads=16, vision_inter_dim=2816, vision_n_layers=32,
        vision_rope_theta=10000.0, vision_downsample_ratio=3, vision_max_n_token=1024, vision_min_pixels=544 * 544,
        vision_max_wh_ratio=None, dim=5120)
    sd = {"vision.patch_embed.proj.weight": tensors["v.patch_embd.weight"].reshape(1024, -1),
          "vision.patch_embed.proj.bias": tensors["v.patch_embd.bias"], "vision.norm.weight": tensors["v.post_ln.weight"],
          "aligner.w1.weight": tensors["mm.1.weight"], "aligner.w1.bias": tensors["mm.1.bias"],
          "aligner.w2.weight": tensors["mm.2.weight"], "aligner.w2.bias": tensors["mm.2.bias"]}
    for i in range(32):
        g, h = f"v.blk.{i}.", f"vision.blocks.{i}."
        sd.update({h + "attn.wqkv.weight": tensors[g + "attn_qkv.weight"], h + "attn.wqkv.bias": tensors[g + "attn_qkv.bias"],
                   h + "attn.wo.weight": tensors[g + "attn_out.weight"], h + "attn.wo.bias": tensors[g + "attn_out.bias"],
                   h + "mlp.w1.weight": torch.cat([tensors[g + "ffn_gate.weight"], tensors[g + "ffn_up.weight"]], 0),
                   h + "mlp.w2.weight": tensors[g + "ffn_down.weight"], h + "norm1.weight": tensors[g + "ln1.weight"],
                   h + "norm2.weight": tensors[g + "ln2.weight"]})
    model = torch.nn.Module()
    model.vision, model.aligner = ref_vision.ViT(args), ref_vision.Aligner(args)
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    assert not missing and not unexpected
    model = model.float().cuda().eval()
    ours32 = V.VisionTower(tensors, "cuda", compute=torch.float32)
    ours16 = V.VisionTower(tensors, "cuda")
    friend = None
    if FRIEND:
        tw = _module("local_ds41_vision_tower", Path(FRIEND) / "local_ds41_vision_tower.py")
        friend = tw.DS41VisionTower(tensors, device="cuda", dtype=torch.float16, residual="fp32")
    del tensors
    return V, ref_ip, args, model, ours32, ours16, friend


CASES = ["carrots.jpeg", "corn.jpeg", "number448", "text1024", "tall", "wide", "max"]


@pytest.mark.parametrize("name", CASES)
def test_tower_against_upstream_and_friend(setup, name):
    from PIL import Image

    from tensorfold.vision.images import ImageInput

    V, ref_ip, args, model, ours32, ours16, friend = setup
    img = Image.open(Path(REF) / name) if name.endswith(".jpeg") else _synthetic(name)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    rgb = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    frame = V.preprocess(ImageInput(rgb.width, rgb.height, rgb.tobytes()))
    ref_patches, rh, rw, rlh, rlw = ref_ip.load_image({"data": buf.getvalue()}, args)
    assert (rh, rw, rlh, rlw) == (frame.n_vit_h, frame.n_vit_w, frame.n_llm_h, frame.n_llm_w)
    patches = torch.from_numpy(np.array(frame.patches))
    assert torch.equal(((patches.float() / 255 - 0.5) / 0.5).to(torch.bfloat16), ref_patches)
    types_ = ref_ip.image_token_types(rlh, rlw).cuda()
    assert types_.numel() == frame.tokens

    with torch.inference_mode():
        norm = ((patches.float() / 255 - 0.5) / 0.5).cuda()
        grid = model.aligner(model.vision(norm, rh, rw), rh, rw)
        gold = torch.empty(types_.numel(), 5120, device="cuda")
        gold[types_ == ref_ip.IMAGE_START] = ours32.start
        gold[types_ == ref_ip.IMAGE_END] = ours32.end
        gold[types_ == ref_ip.IMAGE_NEW_LINE] = ours32.newline
        gold[types_ == ref_ip.IMAGE] = grid.float()

    entry = {"image": name, "size": list(img.size), "tokens": frame.tokens, "vit_grid": [rh, rw]}
    entry["fp32_vs_ref"] = _compare(ours32.encode(frame, torch.float32), gold)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    out16 = ours16.encode(frame)
    torch.cuda.synchronize()
    entry["fp16_cold_s"] = round(time.perf_counter() - t, 4)
    t = time.perf_counter()
    out16 = ours16.encode(frame)
    torch.cuda.synchronize()
    entry["fp16_warm_s"] = round(time.perf_counter() - t, 4)
    entry["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    entry["fp16_vs_ref"] = _compare(out16, gold)
    span = out16.float()
    assert torch.equal(span[0], ours16.start.half().float()) and torch.equal(span[-1], ours16.end.half().float())
    # the serving path encodes in groups of blocks (Decoder._encode): the same kernels, so the same bits
    reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        with torch.inference_mode():
            st = ours16.start_frame(frame)
            for first in range(0, V.LAYERS, 5):
                ours16.run(st, min(V.LAYERS, first + 5))
            assert torch.equal(ours16.span(st), out16)
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
    assert entry["fp32_vs_ref"]["rel_l2"] < 1e-3 and entry["fp32_vs_ref"]["finite"]
    assert entry["fp16_vs_ref"]["finite"]
    if friend is not None:
        reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        try:
            his = friend(patches.cuda(), rh, rw, out_dtype=torch.float16)
        finally:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
        entry["friend_fp16_vs_ref"] = _compare(his, gold)
        entry["ours_fp16_vs_friend_fp16"] = _compare(out16, his)
        entry["ours_equals_friend_bitwise"] = bool(torch.equal(out16, his))
        mine, theirs = entry["fp16_vs_ref"], entry["friend_fp16_vs_ref"]
        assert mine["rel_l2"] <= theirs["rel_l2"] * (1 + 1e-3) + 1e-7
        assert mine["min_token_cos"] >= theirs["min_token_cos"] - 1e-6
    print("DS41VISION " + json.dumps(entry), flush=True)
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(entry) + "\n")


def test_route_with_vision_selects_rows():
    from tensorfold.families.deepseek_v4.cuda import vision as V
    from tensorfold.families.deepseek_v4.cuda.kernels.router import route_topk

    g = torch.Generator(device="cuda").manual_seed(1)
    logits = torch.randn(37, 384, device="cuda", generator=g) * 3
    bias = torch.randn(384, device="cuda", generator=g) * 0.5
    bias_vl = torch.randn(384, device="cuda", generator=g) * 0.5
    image = torch.zeros(37, dtype=torch.bool, device="cuda")
    image[5:20] = True
    p, i = V.route_with_vision(logits, bias, bias_vl, image, 6, 1.5)
    tp, ti = route_topk(logits, bias, 6, 1.5)
    vp, vi = route_topk(logits, bias_vl, 6, 1.5)
    assert torch.equal(p[~image], tp[~image]) and torch.equal(i[~image], ti[~image])
    assert torch.equal(p[image], vp[image]) and torch.equal(i[image], vi[image])
    assert not torch.equal(ti[image], vi[image])
    p0, i0 = V.route_with_vision(logits, bias, bias_vl, None, 6, 1.5)
    assert torch.equal(p0, tp) and torch.equal(i0, ti)


def test_bias_file_is_upstream():
    if not BIAS:
        pytest.skip("set TENSORFOLD_DS41_BIAS_VL")
    from tensorfold.families.deepseek_v4.cuda import vision as V

    table, digest = V.read_bias_vl(BIAS)
    assert digest == V.BIAS_VL_SHA256 and table.shape == (40, 384)
