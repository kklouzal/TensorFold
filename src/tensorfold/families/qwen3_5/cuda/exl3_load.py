"""EXL3 packs of Qwen3.8-27B into the MLX path's ``Weights``: trellis groups on the shared linear, the rest as stored."""

from __future__ import annotations

from pathlib import Path

from tensorfold.cuda.tensor_file import checkpoint_path, read_metadata_json

import torch

SIDECARS = ("quantization_config.json", "quant_config.json")


def quant_config(model_dir: Path) -> dict | None:
    """The EXL3 block of config.json (top level or text config, either key), else of a sidecar file; None if neither."""

    from tensorfold.cuda.exl3 import format as fmt

    config = model_dir / "config.json"
    fields = fmt.config_fields(read_metadata_json(checkpoint_path(model_dir, "config.json"))) if config.exists() else {}
    if fields:
        return fields
    for name in SIDECARS:
        path = model_dir / name
        if path.exists():
            block = read_metadata_json(checkpoint_path(model_dir, name))
            if isinstance(block, dict) and str(block.get("quant_method", "")).lower() == "exl3":
                return block
    return None


def admission(geometry):
    """An EXL3 pack's admission: the MLX path's geometry plus the prompt matmuls' workspace, and its tensors' bytes."""

    from tensorfold.cuda.geometry import exl3_weights, exl3_workspace, with_fixed

    def with_workspace(text):
        from .prefill import CHUNK

        d, i = int(text["hidden_size"]), int(text["intermediate_size"])
        return with_fixed(geometry(text), exl3_workspace(d * i, CHUNK, max(d, i)))

    return with_workspace, exl3_weights


def _where(model_dir: Path) -> dict[str, str]:
    """Every tensor's file, from the safetensors headers (a group's parts may sit in different shards)."""

    from tensorfold.cuda.exl3.format import read_header

    paths = [(path.name, checkpoint_path(model_dir, path.name)) for path in sorted(model_dir.glob("*.safetensors"))]
    where = {}
    for file, path in paths:
        for name in read_header(path):
            if name in where:
                raise ValueError(f"duplicate checkpoint tensor: {name}")
            where[name] = file
    return where


def _files(model_dir: Path, where: dict[str, str]):
    """Every file's tensors, read with O_DIRECT where allowed (one header parse for the plain tensors and the groups)."""

    from tensorfold.cuda.direct_read import SafeTensors

    return SafeTensors([checkpoint_path(model_dir, name) for name in sorted(set(where.values()))])


def _read(files, where: dict[str, str], names: list[str], device: str) -> dict[str, torch.Tensor]:
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(where[name], []).append(name)
    out: dict[str, torch.Tensor] = {}
    for _, wanted in sorted(by_file.items()):
        for name in sorted(wanted, key=lambda n: files.where[n][1]):        # in file order
            out[name] = files.get(name, device)
    return out


def _read_groups(files, groups: dict, device: str, workspace=None) -> dict:
    """``Exl3`` layers for the groups, each part read from its own file; the stored trellis is dropped after the copy."""

    from tensorfold.cuda.exl3.linear import Exl3Linear

    from .weights import Exl3

    out: dict = {}
    for prefix, meta in sorted(groups.items()):
        parts = ["trellis", meta.in_scales, meta.out_scales] + (["bias"] if meta.bias else [])
        t = {p: files.get(f"{prefix}.{p}", device) for p in parts}
        layer = Exl3Linear.from_tensors(t["trellis"], t[meta.in_scales], t[meta.out_scales], meta.codebook,
                                        t.get("bias"), device=device)
        layer.split = PLANS.get((layer.bits, layer.k, layer.n), layer.split)
        out[prefix] = Exl3(layer, workspace=workspace)
        del t
    return out


# (K splits, warps) per 27B projection (bits, K, N): the shape's alone, so every row of a window keeps one reduction
PLANS: dict[tuple[float, int, int], tuple[int, int]] = {
    (3.0, 17408, 5120): (4, 2), (3.0, 5120, 10240): (2, 2), (3.0, 5120, 6144): (1, 8), (3.0, 6144, 5120): (4, 2),
    (4.0, 17408, 5120): (1, 8), (4.0, 5120, 10240): (5, 4), (4.0, 5120, 6144): (5, 2), (4.0, 6144, 5120): (16, 2),
}


def load_exl3(model_dir: str | Path, device: str = "cuda"):
    """An EXL3 pack: the model's groups as ``Exl3``, its unquantized tensors as stored; vision tower and MTP skipped."""

    from tensorfold.cuda.exl3 import format as fmt
    from tensorfold.cuda.exl3.prefill import Workspace

    from .weights import GDN, Attention, Config, Layer, Plain, Weights, _close_failed_checkpoint

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    ckpt = fmt.scan(model_dir, read_markers=False)
    prefix = "model.language_model."

    def foreign(name: str) -> bool:
        return not (name.startswith(prefix) or name.startswith("lm_head")) or ".mtp." in name

    if ckpt.bad:
        raise ValueError(f"unreadable EXL3 groups: {list(ckpt.bad.items())[:3]}")
    groups = {p: m for p, m in ckpt.groups.items() if not foreign(p)}
    if not groups:
        raise ValueError(f"{model_dir} has no EXL3 groups under {prefix!r}")
    where = _where(model_dir)
    files = _files(model_dir, where)
    try:
        plain = _read(files, where, [n for n in ckpt.plain if not foreign(n)], device)
        exl3 = _read_groups(files, groups, device, Workspace())
    except BaseException as error:
        _close_failed_checkpoint(files, error)
        raise
    files.close()                                    # drains copies and releases pinned staging before construction

    def group(name: str):
        key = prefix + name if not name.startswith("lm_head") else name
        if key not in exl3:
            raise ValueError(f"the checkpoint has no EXL3 group {key}")
        return exl3.pop(key)

    def stored(name: str) -> torch.Tensor:
        key = prefix + name
        if key not in plain:
            raise ValueError(f"the checkpoint has no plain tensor {key}")
        return plain.pop(key).contiguous()

    def norm(name: str) -> torch.Tensor:
        """A centred RMSNorm weight (stored as gamma - 1) with its 1 back, as the MLX converter wrote it."""

        w = stored(name)
        return (w.float() + 1.0).to(w.dtype)

    layers = []
    for i in range(cfg.layers):
        p = f"layers.{i}."
        gdn = attn = None
        if cfg.is_linear(i):
            gdn = GDN(qkv=group(p + "linear_attn.in_proj_qkv"), z=group(p + "linear_attn.in_proj_z"),
                      b=Plain(stored(p + "linear_attn.in_proj_b.weight")),
                      a=Plain(stored(p + "linear_attn.in_proj_a.weight")),
                      out=group(p + "linear_attn.out_proj"),
                      conv=stored(p + "linear_attn.conv1d.weight").reshape(-1, cfg.conv_kernel).contiguous(),
                      A_log=stored(p + "linear_attn.A_log").float().contiguous(),
                      dt_bias=stored(p + "linear_attn.dt_bias").float().contiguous(),
                      norm=stored(p + "linear_attn.norm.weight"))
        else:
            attn = Attention(q=group(p + "self_attn.q_proj"), k=group(p + "self_attn.k_proj"),
                             v=group(p + "self_attn.v_proj"), o=group(p + "self_attn.o_proj"),
                             q_norm=norm(p + "self_attn.q_norm.weight"),
                             k_norm=norm(p + "self_attn.k_norm.weight"))
        layers.append(Layer(linear=cfg.is_linear(i), input_norm=norm(p + "input_layernorm.weight"),
                            post_norm=norm(p + "post_attention_layernorm.weight"), gdn=gdn, attn=attn,
                            gate=group(p + "mlp.gate_proj"), up=group(p + "mlp.up_proj"),
                            down=group(p + "mlp.down_proj")))
    w = Weights(config=cfg, embed=Plain(stored("embed_tokens.weight")), layers=layers,
                norm=norm("norm.weight"), head=group("lm_head"), quant="exl3")
    half = cfg.rope_dims // 2
    inv = cfg.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)
    w.inv_freq = inv.to(torch.float32).to(device)
    if any(not layer.linear for layer in w.layers):
        _ = w.attention_origin                        # resolve before an attention runtime/capture
    if plain or exl3:
        raise ValueError(f"unused checkpoint tensors: {sorted(plain)[:3] + sorted(exl3)[:3]}")
    return w


def _storage_estimate(header, text, parse_group, itemsize, original_host_staging, *, with_drafter):
    """Return exact retained storage and conservative phase-overlap bytes.

    ``parse_group``/``itemsize`` are the maintained metadata validators. This
    operation reads no tensor payload and allocates no device state. Every
    original header is already dtype/shape/offset validated; marker values
    remain under the unchanged maintained checkpoint loader's contract.
    """
    import math

    PREFIX = "model.language_model."
    parts = ("trellis", "suh", "su", "svh", "sv", "mcg", "mul1", "bias")

    def amount(name):
        info = header[name]
        return math.prod(info["shape"]) * itemsize(info, name)

    def foreign(name):
        return not (name.startswith(PREFIX) or name.startswith("lm_head")) or ".mtp." in name

    by_prefix = {}
    for name, info in header.items():
        prefix, _, part = name.rpartition(".")
        if part in parts:
            by_prefix.setdefault(prefix, {})[part] = info
    taken, groups = set(), {}
    for prefix, group in by_prefix.items():
        if "trellis" not in group:
            continue
        # The maintained scan validates even foreign groups before filtering.
        meta = parse_group(prefix, group)
        if meta.k <= 0 or meta.n <= 0:
            raise ValueError("EXL3 dimensions must be positive: " + prefix)
        taken.update(prefix + "." + part for part in group)
        if not foreign(prefix):
            groups[prefix] = meta
    if not groups or "lm_head" not in groups:
        raise ValueError("the supported dense EXL3 groups and head are required")
    plain = {name: info for name, info in header.items() if name not in taken and not foreign(name)}
    plain_raw = sum(amount(name) for name in plain)
    plain_final = sum(math.prod(info["shape"]) *
                      (4 if name.endswith((".A_log", ".dt_bias")) else itemsize(info, name))
                      for name, info in plain.items())
    original_cast_overhang = sum(max(0, amount(name) - math.prod(info["shape"]) * 4)
                                 for name, info in plain.items() if name.endswith((".A_log", ".dt_bias")))
    resident_groups = temporary_group = sign_count = counters = 0
    for prefix, meta in groups.items():
        names = [prefix + ".trellis", prefix + "." + meta.in_scales, prefix + "." + meta.out_scales]
        if meta.bias:
            bias = header[prefix + ".bias"]
            if math.prod(bias["shape"]) != meta.n:
                raise ValueError("EXL3 bias must have one entry per output: " + prefix)
            names.append(prefix + ".bias")
        resident_groups += meta.trellis_bytes + 2 * (meta.k + meta.n + (meta.n if meta.bias else 0))
        # The original group dictionary survives the new strip copy, expanded
        # scales and optional FP16 bias creation; add all of its source bytes.
        temporary_group = max(temporary_group, sum(amount(name) for name in names))
        sign_count = max(sign_count, (meta.k if meta.in_scales == "su" else 0) +
                         (meta.n if meta.out_scales == "sv" else 0))
        counters += 8 * (meta.n // 128) * 4
    cast_temp = max((4 * math.prod(info["shape"]) for name, info in plain.items()
                     if name.endswith((".A_log", ".dt_bias"))), default=0)
    norm_temp = max((max(8, itemsize(info, name) + 4) * math.prod(info["shape"])
                     for name, info in plain.items() if name.endswith("norm.weight")), default=0)
    head_dim = int(text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"])
    rope = text.get("rope_parameters") or {}
    rope_dims = int(head_dim * float(rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))))
    inv_freq = rope_dims // 2 * 4
    subset = head_ids = head_columns = subset_counters = subset_temporary = 0
    if with_drafter:
        head = groups["lm_head"]
        vocab = int(text["vocab_size"])
        spans = tuple((a, min(b, vocab)) for a, b in ((0, 98304), (248032, 248320)) if a < vocab)
        blocks = sorted({b for a, end in spans for b in range(a // 128, -(-end // 128))})
        count = len(blocks) * 128
        picked = sum(end - a for a, end in spans)
        if head.n != vocab:
            raise ValueError("dense EXL3 head outputs must match vocabulary")
        # Input signs are shared with the full head; only strips/output signs,
        # optional bias, counters and owned ID/column maps are new.
        subset = head.trellis_bytes // (head.n // 128) * len(blocks) + 2 * count * (1 + int(head.bias))
        subset_counters = 8 * len(blocks) * 4
        head_ids = head_columns = picked * 8
        # Subset construction may hold indices, broadcasted columns, two bool
        # comparisons and their merge, plus keep/nonzero before frame release.
        subset_temporary = 2 * len(blocks) * 8 + 128 * 8 + count * (8 + 4) + picked * 8
    resident = plain_final + resident_groups + counters + inv_freq + subset + subset_counters + head_ids + head_columns
    gpu_staging = original_cast_overhang + max(temporary_group, norm_temp, cast_temp, subset_temporary)
    # Linux direct path has two 64MiB blocks, each PIECE+3*4096. Buffered
    # fallback allocates a full CPU source, while an old direct ring may remain.
    reader_pinned = 2 * ((64 << 20) + 3 * 4096)
    largest_read = max((amount(name) for name in header if not foreign(name)), default=0)
    host_staging = max(original_host_staging, reader_pinned + max(largest_read, 32 * sign_count))
    return {"resident": resident, "GPU_staging": gpu_staging, "host_staging": host_staging,
            "plain_raw_bytes": plain_raw, "plain_final_bytes": plain_final,
            "original_plain_cast_overhang": original_cast_overhang,
            "retained_group_bytes": resident_groups, "max_original_group_bytes": temporary_group,
            "persistent_full_counter_bytes": counters, "inverse_frequency_bytes": inv_freq,
            "subset_payload_bytes": subset, "subset_counter_bytes": subset_counters,
            "head_id_bytes": head_ids, "head_column_bytes": head_columns,
            "subset_temporary_bound": subset_temporary, "norm_temporary_bound": norm_temp, "cast_conversion_temporary_bound": cast_temp,
            "reader_pinned_bound": reader_pinned, "largest_source_read_bytes": largest_read,
            "target_group_count": len(groups), "packed_sign_count_max_group": sign_count,
            "with_drafter": bool(with_drafter)}


def weight_estimate(model_dir: str | Path, *, with_drafter: bool):
    """Discrete single-GPU text-only loader estimate; host staging stays conservative.

    Final GPU reads do not triple-copy the embedding. One original EXL3 group
    coexists with its prepared strips; expanded sign scales, counters and the
    optional draft-head subset are retained. Unified/vision callers keep the
    original generic estimate until their separate lifetimes are qualified.
    """
    from tensorfold.cuda.capacity import Weights, _estimate_weights_from_headers, config, headers, itemsize
    from tensorfold.cuda.geometry import exl3_weights
    from tensorfold.cuda.exl3.format import parse_group

    entries = headers(model_dir)
    generic = _estimate_weights_from_headers(entries, exl3_weights)
    estimated = _storage_estimate(entries, config(model_dir), parse_group, itemsize,
                                  generic.staging, with_drafter=with_drafter)
    return Weights(estimated["resident"], estimated["GPU_staging"], 0), estimated["host_staging"]
