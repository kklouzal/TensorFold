"""Header-only NVFP4 RAM-expert validation; no CUDA or torch imports."""

from __future__ import annotations

import re


def routed_tensor(name):
    return ".mlp.experts." in name


def bank_bases(infos, text, mtp=True):
    """Validate every routed projection before allocating authority or a pool."""
    bases = {n.split(".experts.", 1)[0] for n in infos if routed_tensor(n)
             and (mtp or not n.startswith("mtp."))}
    main = {int(m.group(1)) for b in bases if (m := re.search(r"(?:^|\.)layers\.(\d+)\.mlp$", b))
            and not b.startswith("mtp.")}
    layers = text["num_hidden_layers"]
    if len(main) != layers or min(main, default=-1) != 0 or max(main, default=-1) != layers - 1:
        raise ValueError("--vram-experts: NVFP4 checkpoint must contain every configured main expert layer")
    e, width, dims = text["num_experts"], text["moe_intermediate_size"], text["hidden_size"]

    def expect(name, shape, dtypes):
        info = infos.get(name)
        if info is None or list(info["shape"]) != list(shape) or info["dtype"] not in dtypes:
            raise ValueError(f"--vram-experts: invalid NVFP4 expert field {name}; expected {shape} {dtypes}")
        return info

    for base in bases:
        draft = base.startswith("mtp.")
        if not (re.fullmatch(r"(?:model(?:\.language_model)?|language_model\.model)\.layers\.\d+\.mlp", base)
                or base == "mtp.layers.0.mlp"):
            raise ValueError(f"--vram-experts: unrecognized NVFP4 expert layer {base}")
        if base + ".experts.gate_up_proj" in infos:
            if not draft:
                raise ValueError("--vram-experts: only MTP experts may be stacked BF16")
            expect(base + ".experts.gate_up_proj", (e, 2 * width, dims), ("BF16", "F16", "F32"))
            expect(base + ".experts.down_proj", (e, dims, width), ("BF16", "F16", "F32"))
            continue
        for expert in range(e):
            for p, n, k in (("gate_proj", width, dims), ("up_proj", width, dims), ("down_proj", dims, width)):
                b = f"{base}.experts.{expert}.{p}"
                info = infos.get(b + ".weight", {})
                if info.get("dtype") == "F8_E4M3" and draft:
                    expect(b + ".weight", (n, k), ("F8_E4M3",))
                    if b + ".weight_scale_inv" in infos:
                        expect(b + ".weight_scale_inv", (-(-n // 128), -(-k // 128)), ("BF16", "F16", "F32"))
                    else:
                        scale = infos.get(b + ".weight_scale", {})
                        if scale.get("dtype") not in ("BF16", "F16", "F32") or list(scale.get("shape", ())) not in (
                                [], [1], [n], [n, 1]):
                            raise ValueError(f"--vram-experts: invalid FP8 MTP expert scale {b}")
                else:
                    expect(b + ".weight", (n, k // 2), ("U8",))
                    expect(b + ".weight_scale", (n, k // 16), ("F8_E4M3",))
                    scale = infos.get(b + ".weight_scale_2", {})
                    if scale.get("dtype") not in ("BF16", "F16", "F32") or list(scale.get("shape", ())) not in ([], [1]):
                        raise ValueError(f"--vram-experts: invalid NVFP4 expert global scale {b}")
    return tuple(sorted(bases))
