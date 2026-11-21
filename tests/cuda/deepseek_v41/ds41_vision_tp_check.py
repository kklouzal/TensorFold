"""DeepSeek-V4.1 TP8 vision tower on eight V100s against the one-device towers (a script: it launches its ranks).

usage: python ds41_vision_tp_check.py OUT.json       (8 GPUs; it starts one process per GPU, CUDA_VISIBLE_DEVICES=r)
  TENSORFOLD_DS41_MMPROJ      mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf
  TENSORFOLD_DS41_VISION_REF  dir with the example images carrots.jpeg / corn.jpeg (test_ds41_vision_tower.py's)
  TENSORFOLD_DS41_OLD_VISION  optional: an earlier vision.py whose one-device FP16 tower is the "old" reference
                              (the rank-0 tower that served before TP8)
Every rank encodes test_ds41_vision_tower.py's seven images with its TP part; rank 0 also runs the one-device FP32
oracle (world 1), the one-device FP16 tower of this source (world 1) and the old tower, and records per image the
per-token cosine (min / mean) and relative L2 of each pair, whether every rank's span is bitwise rank 0's, warm encode
times, and the time of the tower's all-reduces alone at the largest image. Gate: the TP8 tower is not less accurate
than the old one against the FP32 oracle (rel L2 within 5%, min token cosine within 1e-4) and spans agree across
ranks; delimiters exact.
"""

import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORLD = 8
HERE = Path(__file__).resolve().parent


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _frames(V, ref):
    from PIL import Image

    from tensorfold.vision.images import ImageInput
    synthetic = _module("ds41_tower_cases", HERE / "test_ds41_vision_tower.py")
    out = []
    for name in synthetic.CASES:
        img = Image.open(Path(ref) / name) if name.endswith(".jpeg") else synthetic._synthetic(name)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        rgb = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        out.append((name, V.preprocess(ImageInput(rgb.width, rgb.height, rgb.tobytes()))))
    return out


def _compare(a, ref):
    import torch
    a, ref = a.float(), ref.float()
    cos = torch.nn.functional.cosine_similarity(a, ref, dim=-1)
    return dict(rel_l2=float((a - ref).norm() / ref.norm()), min_token_cos=float(cos.min()),
                mean_token_cos=float(cos.mean()), max_abs=float((a - ref).abs().max()),
                bitwise=bool(torch.equal(a, ref)))


def _timed(fn, reps=3):
    import torch
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return out, sorted(times)[len(times) // 2]


def rank_main(rank, port, out_path):
    import torch

    from tensorfold.cuda.comm import open_comm
    from tensorfold.families.deepseek_v4.cuda import vision as V
    from tensorfold.families.deepseek_v4.cuda.collectives import Collectives

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    comm = open_comm(rank, WORLD, "127.0.0.1", port)
    coll = Collectives(comm)
    mmproj = os.environ["TENSORFOLD_DS41_MMPROJ"]
    t = time.perf_counter()
    tower = V.VisionTower(V.read_mmproj(mmproj, rank, WORLD), "cuda", rank=rank, world=WORLD, comm=coll)
    load_s = time.perf_counter() - t
    frames = _frames(V, os.environ["TENSORFOLD_DS41_VISION_REF"])
    refs = {}
    if rank == 0:
        full = V.read_mmproj(mmproj)
        refs["fp32"] = V.VisionTower(full, "cuda", compute=torch.float32)
        refs["one16"] = V.VisionTower(full, "cuda")
        old = os.environ.get("TENSORFOLD_DS41_OLD_VISION")
        if old:
            O = _module("ds41_old_vision", old)
            refs["old"] = O.VisionTower(O.read_mmproj(mmproj), "cuda")
        del full
    result = {"world": WORLD, "load_s_rank0": round(load_s, 2), "bytes_per_rank": tower.nbytes(),
              "bytes_one_device": refs["one16"].nbytes() if rank == 0 else None, "images": []}
    tower.encode(frames[-1][1])                          # warm (largest image)
    for name, frame in frames:
        span, tp_s = _timed(lambda: tower.encode(frame))
        every = torch.empty((WORLD,) + tuple(span.shape), dtype=span.dtype, device=span.device)
        comm.all_gather(span.contiguous(), every)
        if rank != 0:
            continue
        entry = {"image": name, "tokens": frame.tokens, "vit_grid": [frame.n_vit_h, frame.n_vit_w],
                 "ranks_bitwise_equal": all(bool(torch.equal(every[r], span)) for r in range(WORLD)),
                 "tp8_warm_s": round(tp_s, 4)}
        gold = refs["fp32"].encode(frame, torch.float32)
        one16, one_s = _timed(lambda: refs["one16"].encode(frame))
        entry["one16_warm_s"] = round(one_s, 4)
        entry["tp8_vs_fp32"] = _compare(span, gold)
        entry["one16_vs_fp32"] = _compare(one16, gold)
        entry["tp8_vs_one16"] = _compare(span, one16)
        if "old" in refs:
            old16, old_s = _timed(lambda: refs["old"].encode(frame))
            entry["old_warm_s"] = round(old_s, 4)
            entry["old_vs_fp32"] = _compare(old16, gold)
            entry["tp8_vs_old"] = _compare(span, old16)
            mine, theirs = entry["tp8_vs_fp32"], entry["old_vs_fp32"]
            entry["not_less_accurate"] = (mine["rel_l2"] <= theirs["rel_l2"] * 1.05
                                          and mine["min_token_cos"] >= theirs["min_token_cos"] - 1e-4)
        entry["delimiters_exact"] = bool(torch.equal(span[0], tower.start.half()) and
                                         torch.equal(span[-1], tower.end.half()))
        result["images"].append(entry)
        print("DS41VISIONTP " + json.dumps(entry), flush=True)
    # the tower's all-reduces alone at the largest image (2 per block of [patches, 1024] FP32, 1 of [tokens, 5120])
    frame = frames[-1][1]
    n, m = frame.n_vit_h * frame.n_vit_w, frame.n_llm_h * frame.n_llm_w
    a, b = torch.zeros(n, V.DIM, device="cuda"), torch.zeros(m, V.OUT_DIM, device="cuda")

    def reduces():
        for _ in range(2 * V.LAYERS):
            coll.sum(a, torch.float32)
        coll.sum(b, torch.float32)
    reduces()
    _, ar_s = _timed(reduces)
    if rank == 0:
        result["allreduce_only_s_largest"] = round(ar_s, 4)
        imgs = result["images"]
        result["pass"] = all(e["ranks_bitwise_equal"] and e["delimiters_exact"] and e.get("not_less_accurate", True)
                             for e in imgs)
        Path(out_path).write_text(json.dumps(result, indent=1))
        print("DS41VISIONTP summary " + json.dumps({k: v for k, v in result.items() if k != "images"}), flush=True)
    comm.close()
    return 0 if rank != 0 or result["pass"] else 1


def main():
    if "DS41_TP_RANK" in os.environ:
        sys.exit(rank_main(int(os.environ["DS41_TP_RANK"]), int(os.environ["DS41_TP_PORT"]), sys.argv[1]))
    port = 29600 + os.getpid() % 300
    procs = [subprocess.Popen([sys.executable, __file__, sys.argv[1]],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(r), "DS41_TP_RANK": str(r),
                                   "DS41_TP_PORT": str(port)}) for r in range(WORLD)]
    codes = [p.wait() for p in procs]
    print("rank exit codes", codes, flush=True)
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
