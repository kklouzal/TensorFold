#!/usr/bin/env python3
"""Compare CPU expert pack chunks through complete Flash Next startup and first reply.

Use the pinned CUDA verification image with its genuine 64-expert checkpoint
fixture. Every sample constructs a fresh engine, including native startup
checks and prompt/MTP warmup, then completes the same first drafted generation.
Only the host loader's chunk and its conservative loading-memory receipt vary.
This is an integrated synthetic startup comparison, not a full model or PCIe
throughput claim.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "cuda"))

import torch  # noqa: E402

from tensorfold.cuda import host_experts  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp import ram_experts  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine  # noqa: E402
from test_flashnext_tp import _checkpoint  # noqa: E402


def _mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing")


def _warm_file(path: Path) -> None:
    # Request the same buffered file-cache condition before each sample. The
    # production reader still decides whether its actual reads use O_DIRECT.
    with path.open("rb", buffering=0) as file:
        while file.read(4 * 2**20):
            pass


def _comparison(a: list[dict], b: list[dict], field: str) -> dict:
    differences = [x[field] - y[field] for x, y in zip(a, b)]
    rng = random.Random(20261006)
    bootstrap = sorted(statistics.median(rng.choices(differences, k=len(differences))) for _ in range(2000))
    return {"baseline_median_ms": statistics.median(x[field] for x in a),
            "candidate_median_ms": statistics.median(x[field] for x in b),
            "paired_median_saved_ms": statistics.median(differences),
            "paired_median_bootstrap_95pct_ms": [bootstrap[49], bootstrap[1949]],
            "candidate_faster_pairs": sum(x > 0 for x in differences)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--scratch-dir", type=Path)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 100:
        parser.error("--repeats must be 3 to 100")
    if args.scratch_dir is not None and not args.scratch_dir.is_dir():
        parser.error("--scratch-dir must be an existing owned directory")
    os.environ["TENSORFOLD_PREFILL_ROWS"] = "256"
    options = dict(depth=2, max_len=128, context_explicit=True, prefetch=False, graphs=False,
                   yarn_factor=2, draft_vocab=None, confidence=0.0)
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300]
    sampling = Sampling(seed=99, top_k=20, top_p=0.95)
    load_host, layout = host_experts.load_host_experts, ram_experts.layout
    source_root = Path(__file__).resolve().parents[1]
    source_files = ("tools/bench_ram_expert_startup.py", "src/tensorfold/cuda/host_experts.py",
                    "src/tensorfold/cuda/expert_cache.py", "src/tensorfold/cuda/capacity.py",
                    "src/tensorfold/families/qwen4_exp/ram_experts.py",
                    "src/tensorfold/families/qwen4_exp/cuda/engine.py",
                    "src/tensorfold/families/qwen4_exp/cuda/reader.py",
                    "src/tensorfold/families/qwen4_exp/cuda/weights.py", "tests/cuda/test_flashnext_tp.py")
    output = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "source_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                                for name in source_files},
              "repeats": args.repeats, "chunks": [1, 2, 4], "checkpoint_seed": 7,
              "warmup_per_chunk": 2, "constructor_options": options,
              "prefill_rows": 256,
              "cache_state": "file pages requested warm before each sample; fresh engine/hot cache; kernels compiled",
              "measurement": "completed real FlashNextEngine constructor + first completed drafted8-token reply",
              "limitations": "64-expert synthetic checkpoint, GB10; no full-model or discrete PCIe throughput claim",
              "order": [], "samples": {str(chunk): [] for chunk in (1, 2, 4)}}
    with tempfile.TemporaryDirectory(prefix="tensorfold-ram-startup-", dir=args.scratch_dir) as temporary:
        root = Path(temporary)
        _checkpoint(root)
        config_path = root / "config.json"
        config = json.loads(config_path.read_text())
        config["max_position_embeddings"] = 256
        config_path.write_text(json.dumps(config))
        file = root / "model.safetensors"
        output["checkpoint_bytes"] = file.stat().st_size
        with file.open("rb") as stream:
            output["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        output["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        native = FlashNextEngine(root, **options)
        try:
            expected = []
            native.generate(prompt, 8, sampling, lambda new: expected.extend(new), draft=True, stop_eos=False)
            if len(expected) != 8:
                raise RuntimeError("native resident generation did not finish the expected reply")
            entry_bytes = native.w.layers[0].moe.experts.bytes_per_expert()
            cache_gib = 2 * entry_bytes / 2**30
            source_layers = [layer.moe.experts for layer in native.w.layers] + [native.w.mtp.layer.moe.experts]
            expected_host_bytes = sum(ex.count * entry_bytes for ex in source_layers)
            source_layers = [(ex.up.cpu(), ex.down.cpu()) for ex in source_layers]
            # The oracle is now consumed: retain its CPU bytes and reply only,
            # rather than duplicating complete CUDA buffers during every trial.
            native.close()
            native = None
            gc.collect()
            torch.cuda.empty_cache()

            def sample(chunk: int) -> dict:
                gc.collect()
                torch.cuda.empty_cache()
                _warm_file(file)
                direct = []

                def loader(reader, name, **kw):
                    kw.pop("chunk_experts", None)
                    result = load_host(reader, name, chunk_experts=chunk, **kw)
                    direct.append(bool(reader.io.direct))
                    return result

                def sized_layout(*arguments, **kw):
                    value = layout(*arguments, **kw)
                    loading = max(64 * 2**20, 4 * value.entry_bytes * chunk + 256 * 2**10)
                    return replace(value, loading_bytes=loading)

                engine = None
                try:
                    with patch.object(host_experts, "load_host_experts", loader), \
                            patch.object(ram_experts, "layout", sized_layout):
                        available = _mem_available()
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        engine = FlashNextEngine(root, ram_experts=cache_gib, **options)
                        torch.cuda.synchronize()
                        constructed = time.perf_counter()
                        tokens = []
                        engine.generate(prompt, 8, sampling, lambda new: tokens.extend(new),
                                        draft=True, stop_eos=False)
                        torch.cuda.synchronize()
                        completed = time.perf_counter()
                    cache = engine.w.meta["expert_cache"]
                    if tokens != expected or cache.host_bytes != expected_host_bytes or cache.gpu_bytes != 2 * entry_bytes:
                        raise RuntimeError("startup candidate differs from native semantics or cache memory accounting")
                    loading = max(64 * 2**20, 4 * entry_bytes * chunk + 256 * 2**10)
                    if engine.capacity_plan["ram_experts"]["loading_bytes"] != loading:
                        raise RuntimeError("startup candidate reported a different loading-memory receipt")
                    wrapped = [layer.moe.experts for layer in engine.w.layers] + [engine.w.mtp.layer.moe.experts]
                    for oracle, expert in zip(source_layers, wrapped):
                        cold = cache._layers[expert.layer_id].source
                        if not torch.equal(cold[0], oracle[0]) or not torch.equal(cold[1], oracle[1]):
                            raise RuntimeError("startup chunk changed packed authoritative expert bytes")
                    return {"constructor_ms": (constructed - started) * 1000,
                            "generation_ms": (completed - constructed) * 1000,
                            "total_ms": (completed - started) * 1000,
                            "host_bytes": cache.host_bytes, "gpu_bytes": cache.gpu_bytes,
                            "misses": cache.misses, "copied_bytes": cache.copied_bytes,
                            "reader_direct": direct, "host_mem_available_before": available,
                            "host_mem_available_after": _mem_available(),
                            "loading_bytes_receipt": engine.capacity_plan["ram_experts"]["loading_bytes"]}
                finally:
                    if engine is not None:
                        engine.close()

            for _ in range(2):
                for chunk in (1, 2, 4):
                    sample(chunk)
            for repeat in range(args.repeats):
                order = (1, 2, 4) if repeat % 2 == 0 else (4, 2, 1)
                output["order"].append(list(order))
                results = {chunk: sample(chunk) for chunk in order}
                for field in ("host_bytes", "gpu_bytes", "misses", "copied_bytes", "reader_direct"):
                    if any(results[chunk][field] != results[1][field] for chunk in (2, 4)):
                        raise RuntimeError(f"startup variants did different work: {field}")
                for chunk in (1, 2, 4):
                    output["samples"][str(chunk)].append(results[chunk])
            output["comparisons"] = {f"1_vs_{chunk}": {field: _comparison(output["samples"]["1"],
                    output["samples"][str(chunk)], field) for field in ("constructor_ms", "generation_ms", "total_ms")}
                for chunk in (2, 4)}
            output["oracle"] = "all packed main/shared/MTP bytes and first-generation tokens equal native resident"
        finally:
            if native is not None:
                native.close()
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
