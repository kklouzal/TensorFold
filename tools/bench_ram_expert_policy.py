#!/usr/bin/env python3
"""Compare the scalar oracle and canonical cache policy through complete Flash Next forwards.

Use the pinned CUDA verification image. The default comparison uses the owned
independent scalar test oracle and the canonical production policy; an optional
owned Python candidate module supports further controlled experiments.
The model has six distinct mixed attention/DeltaNet layers, 512 routed experts
plus each layer's shared expert, and the native fixture's real head dimensions.
Packed CPU authority is built once. Native resident outputs move to CPU before
resident experts and oracle constructors are released. Each timed trial uses a
fresh sequence and either an empty cache or one completed replay's cache state.
This is integrated synthetic GB10 evidence, without a discrete PCIe speed claim.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "cuda"))

import torch  # noqa: E402

from tensorfold.cuda import expert_cache, experts as grouped  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import forward as fwd  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import LayerW  # noqa: E402
import test_flashnext_forward as fixture  # noqa: E402
from expert_cache_oracle import ScalarPolicy, same_tensor_bits  # noqa: E402


def _model():
    with patch.object(fixture, "E", 512):
        base = fixture._model(seed=19)
        cfg = replace(base.cfg, layers=6, layer_types=["linear", "attention"] * 3)
        rng = fixture._Rand(29)
        layers = list(base.layers)
        for index in range(2, 6):
            linear = cfg.layer_types[index] == "linear"
            layers.append(LayerW(index, linear, rng.hc(True), rng.hc(True),
                                 rng.gdn(cfg) if linear else None,
                                 None if linear else rng.attention(cfg), rng.moe()))
    return replace(base, cfg=cfg, layers=layers, mtp=None)


def _host_authority(model):
    sources = [grouped.Experts(ex.up.cpu(), ex.down.cpu(), ex.gs, ex.width, ex.dims, ex.limit)
               for ex in (layer.moe.experts for layer in model.layers)]
    layers = [replace(layer, moe=replace(layer.moe, experts=sources[index]))
              for index, layer in enumerate(model.layers)]
    digest = hashlib.sha256()
    for source in sources:
        for payload in (source.up, source.down):
            digest.update(memoryview(payload.numpy()).cast("B"))
    return replace(model, layers=layers), sources, digest.hexdigest()


def _tensor_bytes(value, device: str, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size() if value.device.type == device else 0
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, device, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item, device, seen) for item in value)
    if hasattr(value, "__dict__"):
        return _tensor_bytes(vars(value), device, seen)
    return 0


def _memory_plan(model, rows: int, prefill: bool, slots: int, hard_limit: int) -> dict:
    capacity = max(128, rows + 1)
    meta_model = replace(model, inv_freq=model.inv_freq.to("meta"))
    meta = (Buffers(meta_model, rows, capacity, prefill=prefill, moe_prefill=True),
            State(meta_model, capacity, rows))
    constructor, host = _tensor_bytes(meta, "meta"), _tensor_bytes(meta, "cpu")
    del meta, meta_model
    cfg = model.cfg
    front = rows * (2 * cfg.nk * cfg.dk * 4 + cfg.nv * cfg.dv * 2 + 2 * cfg.nv * 4)
    entry = model.layers[0].moe.experts.bytes_per_expert()
    output_bytes = rows * (cfg.streams * cfg.hidden + cfg.vocab) * 2
    # One constructor is live at a time. 64MiB bounds cached-window scratch
    # and native intermediate allocation slack for these declared row sizes.
    extra = constructor + front + output_bytes + slots * entry + 64 * 2**20
    cuda_free = int(torch.cuda.mem_get_info()[0])
    gpu_budget = min(hard_limit, max(0, cuda_free - 256 * 2**20))
    maximum, current = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    cgroup_budget = None
    if maximum.exists() and current.exists() and maximum.read_text().strip() != "max":
        cgroup_budget = max(0, int(maximum.read_text()) - int(current.read_text()) - 256 * 2**20)
    return {"constructor_tensor_bytes": constructor, "host_constructor_tensor_bytes": host,
            "estimated_extra_gpu_bytes": extra, "gpu_budget_bytes": gpu_budget,
            "cgroup_headroom_bytes": cgroup_budget,
            "allowed": extra <= gpu_budget and (cgroup_budget is None or extra + host + output_bytes <= cgroup_budget)}


def _policy_state(policy) -> tuple:
    return (tuple(policy.keys), tuple(sorted(policy.resident.items())),
            tuple((layer, bytes(scores)) for layer, scores in sorted(policy.frequency.items())),
            tuple(int(stamp) for stamp in policy.recency), int(policy.tick), int(policy.touches))


def _comparison(a: list[dict], b: list[dict]) -> dict:
    differences = [x["ms"] - y["ms"] for x, y in zip(a, b)]
    rng = random.Random(20261006)
    bootstrap = sorted(statistics.median(rng.choices(differences, k=len(differences))) for _ in range(2000))
    return {"scalar_median_ms": statistics.median(x["ms"] for x in a),
            "candidate_median_ms": statistics.median(x["ms"] for x in b),
            "paired_median_saved_ms": statistics.median(differences),
            "paired_median_bootstrap_95pct_ms": [bootstrap[49], bootstrap[1949]],
            "candidate_faster_pairs": sum(value > 0 for value in differences)}


def _write(path: Path | None, output: dict) -> None:
    if path is None:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(output, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _candidate(path: Path | None, class_name: str):
    """Freeze a callable and its owned source identity before any sample."""

    if path is None:
        source = Path(expert_cache.__file__).resolve()
        return expert_cache._Policy, {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("tensorfold_expert_policy_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the requested candidate module")
    module = importlib.util.module_from_spec(spec)
    # Owned experiments may import sibling helper modules. Their source hashes
    # are captured with the entry module so the executable inputs stay visible.
    sys.path.insert(0, str(path.parent))
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    selected = getattr(module, class_name)
    if not callable(selected):
        raise TypeError("candidate policy must be callable")
    sources = {path}
    for imported in tuple(sys.modules.values()):
        filename = getattr(imported, "__file__", None)
        if filename is not None:
            source = Path(filename).resolve()
            if source.suffix == ".py" and source.is_relative_to(path.parent):
                sources.add(source)
    return selected, {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in sorted(sources)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, help="optional owned Python experiment module")
    parser.add_argument("--candidate-class", default="VectorPolicy", help="class exported by --candidate")
    parser.add_argument("--output", type=Path, help="atomic progress and final JSON artifact")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--max-extra-gpu-gib", type=float, default=3.0)
    parser.add_argument("--regions", choices=("all", "decode", "prefill"), default="all")
    parser.add_argument("--slots", type=int, nargs="+", default=[128, 512], help="distinct positive hot-pool capacities")
    args = parser.parse_args()
    if args.candidate is not None and (not args.candidate.is_file() or args.candidate.suffix != ".py"):
        parser.error("--candidate must be an existing owned Python module")
    if args.candidate is None and args.candidate_class != "VectorPolicy":
        parser.error("--candidate-class requires --candidate")
    if any(not 1 <= slots <= 2**31 - 1 for slots in args.slots) or len(set(args.slots)) != len(args.slots):
        parser.error("--slots must be distinct integers within the cache's positive signed32-bit capacity range")
    if args.output is not None and not args.output.parent.is_dir():
        parser.error("--output parent directory must already exist")
    if not 3 <= args.repeats <= 100:
        parser.error("--repeats must be 3 to 100")
    if not math.isfinite(args.max_extra_gpu_gib) or args.max_extra_gpu_gib <= 0:
        parser.error("--max-extra-gpu-gib must be finite and positive")
    vector, candidate_sources = _candidate(args.candidate, args.candidate_class)
    scalar = ScalarPolicy
    source_root = Path(__file__).resolve().parents[1]
    selected_source = Path(expert_cache.__file__).resolve() if args.candidate is None else args.candidate.resolve()
    output = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "cpu_threads": torch.get_num_threads(), "repeats": args.repeats,
              "slots": args.slots, "baseline_class": f"{scalar.__module__}.{scalar.__name__}",
              "candidate_class": "tensorfold.cuda.expert_cache._Policy" if args.candidate is None else args.candidate_class,
              "model": {"routed_experts": 512, "shared_per_layer": 1, "layers": 6,
                        "layer_types": ["linear", "attention"] * 3, "mtp": False,
                        "seed": 19, "extra_layer_seed": 29, "hidden": fixture.D, "expert_width": fixture.W},
              "candidate_sha256": hashlib.sha256(selected_source.read_bytes()).hexdigest(),
              "candidate_source_sha256": candidate_sources,
              "source_sha256": {str(path.relative_to(source_root) if path.is_relative_to(source_root) else path):
                                  hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (Path(__file__).resolve(), Path(expert_cache.__file__),
                                               Path(grouped.__file__), Path(fwd.__file__), Path(fixture.__file__),
                                               Path(sys.modules[ScalarPolicy.__module__].__file__))},
              "max_extra_gpu_bytes": int(args.max_extra_gpu_gib * 2**30),
              "measurement": "completed whole six-layer Flash Next forward including CPU orchestration",
              "cache_states": {"cold": "new empty global expert cache", "warm": "same cache after one completed identical replay"},
              "sequence_state": "fresh for every replay, native oracle and candidate",
              "limitations": "synthetic GB10 replay; no discrete PCIe or production full-model speed claim",
              "status": "preparing", "regions": []}
    _write(args.output, output)
    regions = [(8, False), (256, True)] if args.regions == "all" else ([(8, False)] if args.regions == "decode" else [(256, True)])
    model = _model()
    oracles = {}
    for rows, prefill in regions:
        tokens = [(11 + i) % model.cfg.vocab for i in range(rows)]
        gc.collect()
        torch.cuda.empty_cache()
        memory = _memory_plan(model, rows, prefill, max(args.slots), output["max_extra_gpu_bytes"])
        if not memory["allowed"]:
            output.update(status="refused_native_oracle_budget", refused_rows=rows, memory_plan=memory)
            _write(args.output, output)
            print(json.dumps(output, indent=2))
            return 2
        capacity = max(128, rows + 1)
        buffers, state = (Buffers(model, rows, capacity, prefill=prefill, moe_prefill=True),
                          State(model, capacity, rows))
        expected = fwd.forward(model, state, buffers, tokens).cpu()
        expected_streams = buffers.streams[:rows].cpu()
        oracles[(rows, prefill)] = (expected, expected_streams)
        del buffers, state
    model, sources, digest = _host_authority(model)
    gc.collect()
    torch.cuda.empty_cache()
    output["authority_sha256"] = digest
    output["host_authority_bytes"] = sum((ex.up.numel() * ex.up.element_size() +
                                           ex.down.numel() * ex.down.element_size()) for ex in sources)
    output["nonexpert_gpu_baseline_bytes"] = torch.cuda.memory_allocated()
    output["status"] = "running"
    _write(args.output, output)
    refused = False
    for rows, prefill in regions:
        tokens = [(11 + i) % model.cfg.vocab for i in range(rows)]
        expected, expected_streams = oracles[(rows, prefill)]
        capacity = max(128, rows + 1)
        for slots in args.slots:
            gc.collect()
            torch.cuda.empty_cache()
            memory = _memory_plan(model, rows, prefill, slots, output["max_extra_gpu_bytes"])
            for warm in (False, True):
                region = {"rows": rows, "prefill": prefill, "hot_slots": slots,
                          "cache_state": "warm" if warm else "cold", "memory_plan": memory}
                if not memory["allowed"]:
                    output["regions"].append({**region, "status": "refused_budget"})
                    refused = True
                    _write(args.output, output)
                    continue

                def sample(policy):
                    cache = None
                    try:
                        entry = sources[0].bytes_per_expert()
                        with patch.object(expert_cache, "_Policy", policy):
                            cache = expert_cache.HostExpertCache(slots * entry, entry, "cuda")
                        layers = [replace(layer, moe=replace(layer.moe, experts=expert_cache.CachedExperts(
                            sources[index], cache, index))) for index, layer in enumerate(model.layers)]
                        cached = replace(model, layers=layers)
                        buffers = Buffers(cached, rows, capacity, prefill=prefill, moe_prefill=True)
                        if warm:
                            state = State(cached, capacity, rows)
                            fwd.forward(cached, state, buffers, tokens)
                            torch.cuda.synchronize()
                            del state
                        state = State(cached, capacity, rows)
                        before = {key: getattr(cache, key) for key in ("hits", "misses", "evictions", "copied_bytes")}
                        torch.cuda.synchronize()
                        constructor_baseline = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter()
                        got = fwd.forward(cached, state, buffers, tokens)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - started) * 1000
                        peak = torch.cuda.max_memory_allocated()
                        if peak - output["nonexpert_gpu_baseline_bytes"] > memory["gpu_budget_bytes"]:
                            raise RuntimeError("complete forward exceeded its GPU tensor budget")
                        if not same_tensor_bits(got.cpu(), expected) or not same_tensor_bits(
                                buffers.streams[:rows].cpu(), expected_streams):
                            raise RuntimeError("candidate differs from native resident logits or residual streams")
                        metrics = {key: getattr(cache, key) - before[key] for key in before}
                        union_bound = len(model.layers) * (min(model.cfg.experts, rows * model.cfg.top_k) + 1)
                        if warm and slots >= union_bound and (metrics["misses"] or metrics["copied_bytes"]):
                            raise RuntimeError("full-hit replay region unexpectedly reloaded experts")
                        metrics.update(ms=elapsed, hot_bytes=cache.gpu_bytes, host_bytes=cache.host_bytes,
                                       constructor_baseline_gpu_bytes=constructor_baseline, peak_gpu_allocated_bytes=peak,
                                       forward_peak_extra_bytes=peak - constructor_baseline,
                                       peak_above_nonexpert_baseline_bytes=peak - output["nonexpert_gpu_baseline_bytes"])
                        return metrics, _policy_state(cache._policy)
                    finally:
                        if cache is not None:
                            cache.close()

                for _ in range(2):
                    sample(scalar)
                    sample(vector)
                ordinary, candidate, order = [], [], []
                for repeat in range(args.repeats):
                    pair = {}
                    sequence = ("scalar", "candidate") if repeat % 2 == 0 else ("candidate", "scalar")
                    order.append(list(sequence))
                    for name in sequence:
                        pair[name] = sample(scalar if name == "scalar" else vector)
                    if pair["scalar"][1] != pair["candidate"][1]:
                        raise RuntimeError("policies produced different exact replacement state")
                    fields = ("hits", "misses", "evictions", "copied_bytes", "hot_bytes", "host_bytes")
                    if any(pair["scalar"][0][key] != pair["candidate"][0][key] for key in fields):
                        raise RuntimeError("policies did different expert-cache work")
                    ordinary.append(pair["scalar"][0])
                    candidate.append(pair["candidate"][0])
                output["regions"].append({**region, "status": "completed", "order": order,
                                          "scalar_samples": ordinary, "candidate_samples": candidate,
                                          "comparison": _comparison(ordinary, candidate),
                                          "oracle": "every measured logits/streams bitwise native; exact policy state and cache work identical"})
                _write(args.output, output)
    output["status"] = "refused_budget" if refused else "completed"
    _write(args.output, output)
    print(json.dumps(output, indent=2))
    return 2 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
